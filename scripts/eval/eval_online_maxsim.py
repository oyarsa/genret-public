"""Batch online retrieval and evaluation with MaxSim dense scoring.

Runs online retrieval for held-out test papers against a traindev graph,
then evaluates by comparing overlap and nDCG against offline retrieval.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from genret.paper_parsing import Facets
from genret.processing.facet_evaluation import FACET_NAMES
from genret.processing.facet_retrieval import load_edge_graphs
from genret.processing.online_retrieval import load_corpus_index, retrieve_new_paper
from genret.types import CorpusId
from genret.utils import open_auto

logger = logging.getLogger(__name__)


def _load_test_facets(facets_path: Path, traindev_ids: set[int]) -> list[Facets]:
    """Load facets for test papers (those not in traindev)."""
    test_papers: list[Facets] = []
    with open_auto(facets_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            facets = Facets.model_validate(json.loads(line))
            if int(facets.corpusid) not in traindev_ids:
                test_papers.append(facets)
    return test_papers


def _load_test_embeddings(
    embeddings_path: Path, test_ids: set[int]
) -> dict[int, dict[str, npt.NDArray[np.float32]]]:
    """Load per-facet embeddings for test papers."""
    raw = np.load(embeddings_path, allow_pickle=True)
    result: dict[int, dict[str, npt.NDArray[np.float32]]] = {}
    for cid in test_ids:
        key = str(cid)
        if key not in raw:
            continue
        all_embs = raw[key].astype(np.float32)
        # Split concatenated embeddings by facet
        # Items are in FACET_NAMES order
        result[cid] = {"_all": all_embs}
    return result


def _split_embeddings_by_facet(
    all_embs: npt.NDArray[np.float32], facets: Facets
) -> dict[str, npt.NDArray[np.float32]]:
    """Split concatenated item embeddings into per-facet arrays."""
    per_facet: dict[str, npt.NDArray[np.float32]] = {}
    offset = 0
    for name in FACET_NAMES:
        items = getattr(facets, name)
        n = len(items)
        if offset + n <= len(all_embs):
            per_facet[name] = all_embs[offset : offset + n]
        else:
            per_facet[name] = np.zeros((0, all_embs.shape[1]), dtype=np.float32)
        offset += n
    return per_facet


def _load_query_references(
    references_path: Path | None,
    query_ids: set[int],
    corpus_ids: set[int],
) -> dict[int, frozenset[CorpusId]]:
    """Load reference sets for query papers, with targets kept in-corpus.

    Args:
        references_path: Path to `references.jsonl`, or None.
        query_ids: Corpus IDs of the query papers.
        corpus_ids: Corpus IDs forming the retrieval index.

    Returns:
        Dict mapping query corpus ID to its in-corpus reference set.
    """
    if references_path is None:
        return {}
    refs: dict[int, frozenset[CorpusId]] = {}
    with open_auto(references_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = int(rec["paper_id"])
            if pid not in query_ids:
                continue
            if targets := frozenset(
                CorpusId(int(r)) for r in rec["references"] if int(r) in corpus_ids
            ):
                refs[pid] = targets
    return refs


def main() -> None:
    """Run batch online retrieval and evaluate."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Online MaxSim retrieval eval")
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Directory with traindev MaxSim graphs",
    )
    parser.add_argument(
        "--facets",
        default="data/s2orc-nlp/splits-connected-10k/processed/facets.jsonl.zst",
        help="Full corpus facets file",
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Full corpus item embeddings .npz (for test paper embeddings)",
    )
    parser.add_argument(
        "--item-embeddings",
        default=None,
        help="Traindev item embeddings .npz (for MaxSim re-scoring)",
    )
    parser.add_argument(
        "--references",
        default="data/s2orc-nlp/splits-connected-10k/processed/references.jsonl",
        help="References file for bib coupling",
    )
    parser.add_argument(
        "--co-citations",
        default="data/s2orc-nlp/citations/co_citations.jsonl",
        help="Co-citations file",
    )
    parser.add_argument(
        "--bib-coupling",
        default="data/s2orc-nlp/citations/bib_coupling.jsonl",
        help="Bib coupling file",
    )
    parser.add_argument(
        "--dense-metric",
        default="mean_pool",
        choices=["mean_pool", "max_pairs"],
        help="Dense scoring method",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for online retrieval results",
    )
    parser.add_argument(
        "--offline-baseline",
        default=None,
        help="Offline retrieval JSONL for comparison",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit test papers (0=all)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.75, help="Dense similarity weight"
    )
    parser.add_argument("--beta", type=float, default=0.0, help="N-gram Jaccard weight")
    parser.add_argument(
        "--gamma", type=float, default=0.25, help="Citation signal weight"
    )
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)

    # Load corpus index
    refs_path = Path(args.references) if args.references else None
    cocit_path = Path(args.co_citations) if args.co_citations else None
    bib_path = Path(args.bib_coupling) if args.bib_coupling else None
    item_emb_path = Path(args.item_embeddings) if args.item_embeddings else None

    logger.info("Loading corpus index from %s...", graph_dir)
    corpus_index = load_corpus_index(
        graph_dir,
        Path(args.facets),
        references_path=refs_path,
        co_citations_path=cocit_path,
        bib_coupling_path=bib_path,
        item_embeddings_path=item_emb_path,
    )

    # Load edge graphs
    graphs, corpusids = load_edge_graphs(graph_dir)
    traindev_ids = {int(cid) for cid in corpusids}
    logger.info("Traindev papers: %d", len(traindev_ids))

    # Load test papers
    logger.info("Loading test paper facets...")
    test_papers = _load_test_facets(Path(args.facets), traindev_ids)
    if args.limit > 0:
        test_papers = test_papers[: args.limit]
    logger.info("Test papers: %d", len(test_papers))

    # Load test embeddings
    test_ids = {int(p.corpusid) for p in test_papers}
    logger.info("Loading test embeddings...")
    test_embs = _load_test_embeddings(Path(args.embeddings), test_ids)
    logger.info("Test embeddings loaded: %d", len(test_embs))

    # Query-side references for the citation signal. `load_corpus_index` only
    # keeps reference rows for corpus members, so a query outside the corpus
    # would otherwise get `query_references=None` and a citation score that is
    # identically zero. Targets stay restricted to the corpus, matching the
    # corpus-side rows used for bibliographic coupling.
    query_references = _load_query_references(refs_path, test_ids, traindev_ids)
    logger.info(
        "Query reference sets loaded: %d / %d", len(query_references), len(test_ids)
    )

    # Run online retrieval
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    with open(output_path, "w") as out:
        for i, paper in enumerate(test_papers):
            cid = int(paper.corpusid)
            if cid not in test_embs:
                continue

            all_embs = test_embs[cid]["_all"]
            paper_embeddings = _split_embeddings_by_facet(all_embs, paper)

            query_refs = query_references.get(
                cid
            ) or corpus_index.corpus_references.get(paper.corpusid)

            results = retrieve_new_paper(
                query_facets=paper,
                query_embeddings=paper_embeddings,
                corpus_index=corpus_index,
                edge_graphs=graphs,
                query_references=query_refs,
                dense_metric=args.dense_metric,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                top_n=args.top_n,
            )

            rec = {
                "corpusid": cid,
                "results": [
                    {
                        "corpusid": int(corpusids[c.index]),
                        "score": round(c.score, 6),
                        "facets": list(c.facets),
                    }
                    for c in results
                ],
            }
            out.write(json.dumps(rec) + "\n")

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                logger.info(
                    "  %d/%d papers (%.1f s/paper)",
                    i + 1,
                    len(test_papers),
                    elapsed / (i + 1),
                )

    elapsed = time.time() - start
    logger.info(
        "Online retrieval done: %d papers, %.1f s total", len(test_papers), elapsed
    )

    class _Paper(TypedDict):
        corpusid: int

    class _Recommendations(TypedDict):
        results: list[_Paper]
        corpusid: int

    # Compare against offline baseline if provided
    if args.offline_baseline:
        offline: dict[int, list[int]] = {}
        with open(args.offline_baseline) as f:
            for line in f:
                recs: _Recommendations = json.loads(line)
                offline[recs["corpusid"]] = [
                    r["corpusid"] for r in recs["results"][:20]
                ]

        online: dict[int, list[int]] = {}
        with open(output_path) as f:
            for line in f:
                recs = json.loads(line)
                online[recs["corpusid"]] = [r["corpusid"] for r in recs["results"][:20]]

        overlaps: list[float] = []
        for cid in online:
            if cid in offline:
                on_set = set(online[cid])
                off_set = set(offline[cid])
                if on_set and off_set:
                    overlaps.append(len(on_set & off_set) / len(on_set))

        print("\n=== Online vs Offline Comparison ===")
        print(f"  Online queries:  {len(online)}")
        print(f"  Offline queries: {len(offline)}")
        print(f"  Common queries:  {len(overlaps)}")
        print(f"  Mean Overlap@20: {np.mean(overlaps):.3f}")


if __name__ == "__main__":
    main()
