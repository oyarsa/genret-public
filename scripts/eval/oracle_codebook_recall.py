"""Upper-bound recall the codebook permits on the 500-query eval.

Assumes a perfect generative model: at decode time, the model would
always pick the L0/L1 leaves nearest the query's per-facet vector. The
recall computed here is therefore an upper bound on what any trained
generative retriever can achieve over the same codebook.

Run before committing to training to gate codebook quality. Cheap:
pure numpy, no GPU required.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from genret.utils import iter_jsonl

FACET_NAMES: tuple[str, ...] = (
    "problems",
    "methods",
    "results",
    "contributions",
)


def _load_facet_vectors(
    dense_dir: Path, facet: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    f = np.load(dense_dir / f"facet_graph_{facet}.npz")
    return (
        np.asarray(f["corpusids"], dtype=np.int64),
        np.asarray(f["corpus_vectors"], dtype=np.float32),
    )


def _load_codebooks(
    codebook_dir: Path,
) -> tuple[
    dict[str, npt.NDArray[np.float32]],
    dict[str, npt.NDArray[np.float32]],
    dict[str, dict[tuple[int, int], list[int]]],
]:
    level0: dict[str, npt.NDArray[np.float32]] = {}
    level1: dict[str, npt.NDArray[np.float32]] = {}
    for facet in FACET_NAMES:
        level0[facet] = np.load(codebook_dir / facet / "level0_centers.npy")
        level1[facet] = np.load(codebook_dir / facet / "level1_centers.npy")

    raw = json.loads((codebook_dir / "code_to_papers.json").read_text())
    code_to_papers: dict[str, dict[tuple[int, int], list[int]]] = {}
    for facet, leaves in raw.items():
        code_to_papers[facet] = {}
        for key, papers in leaves.items():
            l0_str, l1_str = key.split(",")
            code_to_papers[facet][(int(l0_str), int(l1_str))] = list(papers)
    return level0, level1, code_to_papers


def _cosine(
    query: npt.NDArray[np.float32], targets: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    q = query / (np.linalg.norm(query) + 1e-9)
    t = targets / (np.linalg.norm(targets, axis=-1, keepdims=True) + 1e-9)
    return t @ q


def _top_indices(scores: npt.NDArray[np.float32], n: int) -> list[int]:
    if n >= scores.size:
        return list(np.argsort(scores)[::-1])
    part = np.argpartition(-scores, n - 1)[:n]
    return list(part[np.argsort(-scores[part])])


def _load_labels(
    label_files: Iterable[Path],
) -> dict[int, set[int]]:
    positives: dict[int, set[int]] = defaultdict(set)
    for path in label_files:
        if not path.exists():
            continue
        for r in iter_jsonl(path):
            if not r.get("related"):
                continue
            positives[int(r["paper_a"])].add(int(r["paper_b"]))
    return positives


def oracle_recall(
    query_corpusids: list[int],
    facet_vectors_by_facet: dict[str, npt.NDArray[np.float32]],
    cid_to_idx: dict[int, int],
    level0: dict[str, npt.NDArray[np.float32]],
    level1: dict[str, npt.NDArray[np.float32]],
    code_to_papers: dict[str, dict[tuple[int, int], list[int]]],
    positives: dict[int, set[int]],
    n_l0: int,
    n_l1: int,
    k_top: int,
) -> tuple[float, float, float, float]:
    """Return (mean recall, mean precision, mean candidates, mean coverage)."""
    recalls: list[float] = []
    precisions: list[float] = []
    candidates_per_query: list[int] = []
    coverage: list[float] = []

    for qid in query_corpusids:
        idx = cid_to_idx.get(qid)
        if idx is None:
            continue
        gold = positives.get(qid, set())
        if not gold:
            continue

        retrieved_ranked: list[int] = []
        seen: set[int] = set()
        # Score by max cosine to L1 center across facets to give a
        # deterministic ordering when truncating to k_top.
        for facet in FACET_NAMES:
            q_vec = facet_vectors_by_facet[facet][idx]
            l0_scores = _cosine(q_vec, level0[facet])
            for l0 in _top_indices(l0_scores, n_l0):
                l1_scores = _cosine(q_vec, level1[facet][l0])
                for l1 in _top_indices(l1_scores, n_l1):
                    for cid in code_to_papers[facet].get((l0, l1), ()):
                        if cid == qid or cid in seen:
                            continue
                        seen.add(cid)
                        retrieved_ranked.append(cid)

        top = retrieved_ranked[:k_top]
        hits = sum(1 for c in top if c in gold)
        recalls.append(hits / len(gold))
        precisions.append(hits / max(len(top), 1))
        candidates_per_query.append(len(retrieved_ranked))
        coverage.append(min(len(top), len(gold)) / len(gold))

    return (
        sum(recalls) / max(len(recalls), 1),
        sum(precisions) / max(len(precisions), 1),
        sum(candidates_per_query) / max(len(candidates_per_query), 1),
        sum(coverage) / max(len(coverage), 1),
    )


def main() -> None:
    """Compute oracle codebook recall on the 500-query eval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codebook-dir",
        default="output/genret/codebooks",
        type=Path,
    )
    parser.add_argument(
        "--dense-dir",
        default="data/s2orc-nlp-v2/splits-connected-10k-v2/processed",
        type=Path,
    )
    parser.add_argument(
        "--queries",
        default="output/eval_500/query_papers_500.txt",
        type=Path,
    )
    parser.add_argument(
        "--label-files",
        nargs="+",
        default=[
            Path("output/eval_500/gpt5mini_judgements.jsonl"),
            Path("output/ablations/gpt5mini_missing_judgements.jsonl"),
        ],
        type=Path,
    )
    parser.add_argument(
        "--n-l0",
        type=int,
        default=4,
        help="Top L0 leaves per facet to consider.",
    )
    parser.add_argument(
        "--n-l1",
        type=int,
        default=8,
        help="Top L1 leaves within each L0 to consider.",
    )
    parser.add_argument(
        "--k-top",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        help="Truncation depths for recall.",
    )
    args = parser.parse_args()

    print(f"Loading codebooks from {args.codebook_dir}...")
    level0, level1, code_to_papers = _load_codebooks(args.codebook_dir)

    print(f"Loading facet vectors from {args.dense_dir}...")
    cid_to_idx: dict[int, int] | None = None
    facet_vectors_by_facet: dict[str, npt.NDArray[np.float32]] = {}
    for facet in FACET_NAMES:
        cids, vecs = _load_facet_vectors(args.dense_dir, facet)
        if cid_to_idx is None:
            cid_to_idx = {int(c): i for i, c in enumerate(cids)}
        facet_vectors_by_facet[facet] = vecs
    assert cid_to_idx is not None

    queries = [int(x) for x in args.queries.read_text().split() if x.strip()]
    positives = _load_labels(args.label_files)
    print(
        f"Queries: {len(queries)}; queries with positives: "
        f"{sum(1 for q in queries if positives.get(q))}"
    )

    print(
        f"\nOracle settings: top {args.n_l0} L0 x top {args.n_l1} L1 per facet "
        f"(= {args.n_l0 * args.n_l1 * len(FACET_NAMES)} leaves searched across facets)"
    )
    print(f"{'k':>6} {'recall':>8} {'precision':>10} {'cands':>8} {'coverage':>10}")
    for k in args.k_top:
        r, p, cands, cov = oracle_recall(
            queries,
            facet_vectors_by_facet,
            cid_to_idx,
            level0,
            level1,
            code_to_papers,
            positives,
            n_l0=args.n_l0,
            n_l1=args.n_l1,
            k_top=k,
        )
        print(f"{k:>6} {r:>8.3f} {p:>10.3f} {cands:>8.1f} {cov:>10.3f}")


if __name__ == "__main__":
    main()
