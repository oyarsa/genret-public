"""BM25 baseline retrieval for paper search.

Indexes papers by title + facet items and retrieves candidates using BM25
scoring. Serves as a lexical baseline to compare against the dense
facet-graph pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any

from genret._shed.bm25 import BM25Index
from genret.paper_parsing import Facets, load_facets_map
from genret.processing.facet_evaluation import FACET_NAMES
from genret.processing.facet_graph_eval import compute_metrics, load_ground_truth
from genret.processing.reranker import (
    LocalReranker,
    RerankerConfig,
    format_paper_pair,
    load_titles,
)
from genret.types import CorpusId
from genret.utils import iter_jsonl, open_auto, track_progress

logger = logging.getLogger(__name__)


def _paper_text(title: str, facets: Facets) -> str:
    """Build searchable text from title and facet items.

    Args:
        title: Paper title.
        facets: Extracted facets for this paper.

    Returns:
        Concatenated text suitable for tokenisation.
    """
    parts = [title]
    for name in FACET_NAMES:
        items = getattr(facets, name, ())
        parts.extend(items)
    return " ".join(parts)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercasing tokenizer.

    Args:
        text: Input text.

    Returns:
        List of lowercased tokens.
    """
    return text.lower().split()


def build_bm25_index(
    titles: dict[CorpusId, str],
    facets_map: dict[CorpusId, Facets],
) -> tuple[BM25Index, list[CorpusId]]:
    """Build a BM25 index over all papers with both title and facets.

    Args:
        titles: Corpus ID to title mapping.
        facets_map: Corpus ID to facets mapping.

    Returns:
        Tuple of (BM25 index, ordered list of corpus IDs matching index positions).
    """
    corpus_ids: list[CorpusId] = []
    tokenized_docs: list[list[str]] = []

    for cid in sorted(set(titles) & set(facets_map)):
        text = _paper_text(titles[cid], facets_map[cid])
        corpus_ids.append(cid)
        tokenized_docs.append(_tokenize(text))

    logger.info(f"Building BM25 index over {len(corpus_ids)} papers...")
    index = BM25Index(tokenized_docs)
    logger.info("BM25 index built")
    return index, corpus_ids


def retrieve_bm25(
    query_ids: list[CorpusId],
    titles: dict[CorpusId, str],
    facets_map: dict[CorpusId, Facets],
    index: BM25Index,
    corpus_ids: list[CorpusId],
    top_n: int = 50,
) -> dict[CorpusId, list[dict[str, Any]]]:
    """Retrieve top-N candidates for each query paper using BM25.

    Args:
        query_ids: Corpus IDs of query papers.
        titles: Corpus ID to title mapping.
        facets_map: Corpus ID to facets mapping.
        index: Pre-built BM25 index.
        corpus_ids: Ordered corpus IDs matching index positions.
        top_n: Number of results to return per query.

    Returns:
        Dict mapping query corpus ID to ordered list of result dicts.
    """
    cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    retrieval_sets: dict[CorpusId, list[dict[str, Any]]] = {}

    for qid in track_progress(query_ids, desc="BM25 retrieving..."):
        if qid not in titles or qid not in facets_map:
            continue

        query_text = _paper_text(titles[qid], facets_map[qid])
        query_tokens = _tokenize(query_text)

        scores = index.get_scores(query_tokens)
        query_idx = cid_to_idx.get(qid, -1)
        if query_idx >= 0:
            scores[query_idx] = -1.0  # exclude self

        # argsort descending
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        results: list[dict[str, Any]] = []
        for idx in ranked_indices[:top_n]:
            results.append(
                {
                    "corpusid": int(corpus_ids[idx]),
                    "score": scores[idx],
                }
            )
        retrieval_sets[qid] = results

    return retrieval_sets


def run_bm25_baseline(
    facets_path: Path,
    paper_graphs_path: Path,
    query_papers_path: Path,
    output_path: Path,
    co_cit_path: Path,
    bib_path: Path,
    *,
    top_n: int = 50,
) -> None:
    """Run BM25 baseline retrieval and evaluate.

    Args:
        facets_path: Path to facets JSONL file.
        paper_graphs_path: Path to paper_graphs JSONL file (for titles).
        query_papers_path: Path to query papers JSONL file.
        output_path: Path to write retrieval results.
        co_cit_path: Path to co_citations JSONL file.
        bib_path: Path to bib_coupling JSONL file.
        top_n: Number of candidates to retrieve per query.
    """
    facets_map = load_facets_map(facets_path)
    titles = load_titles(paper_graphs_path)

    index, corpus_ids = build_bm25_index(titles, facets_map)

    query_ids = [
        CorpusId(int(record["corpusid"])) for record in iter_jsonl(query_papers_path)
    ]
    logger.info(f"Loaded {len(query_ids)} query papers")

    retrieval_sets = retrieve_bm25(
        query_ids, titles, facets_map, index, corpus_ids, top_n=top_n
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_auto(output_path, "w") as fh:
        for qid, results in retrieval_sets.items():
            record = {"corpusid": int(qid), "results": results}
            fh.write(json.dumps(record) + "\n")
    logger.info(f"Wrote {len(retrieval_sets)} retrieval sets to {output_path}")

    result_lists = {
        int(qid): [int(r["corpusid"]) for r in results]
        for qid, results in retrieval_sets.items()
    }
    _evaluate_and_log(result_lists, co_cit_path, bib_path)


def _evaluate_and_log(
    result_lists: dict[int, list[int]],
    co_cit_path: Path,
    bib_path: Path,
) -> None:
    """Evaluate retrieval results and log metrics.

    Args:
        result_lists: Query ID to ordered result IDs.
        co_cit_path: Path to co_citations JSONL file.
        bib_path: Path to bib_coupling JSONL file.
    """
    valid_ids = frozenset(result_lists)
    combined_gt, co_cit_gt, bib_gt = load_ground_truth(co_cit_path, bib_path, valid_ids)

    for label, gt in [
        ("Combined", combined_gt),
        ("Co-citation", co_cit_gt),
        ("Bib-coupling", bib_gt),
    ]:
        metrics = compute_metrics(result_lists, gt)
        logger.info(
            f"{label}: Recall@5={metrics.recall[5]:.4f}, "
            f"Recall@10={metrics.recall[10]:.4f}, "
            f"Recall@20={metrics.recall[20]:.4f}, "
            f"MRR={metrics.mrr:.4f} "
            f"({metrics.num_queries} queries, {metrics.num_gt_pairs} GT pairs)"
        )


def rerank_retrieval_results(
    retrieval_path: Path,
    facets_map: dict[CorpusId, Facets],
    titles: dict[CorpusId, str],
    model: LocalReranker,
    *,
    top_n: int = 20,
    batch_size: int = 32,
) -> dict[CorpusId, list[dict[str, Any]]]:
    """Re-rank retrieval results with a cross-encoder.

    Formats all pairs up front, scores them in one batched pass, then splits
    scores back per query.

    Args:
        retrieval_path: Path to retrieval results JSONL.
        facets_map: Corpus ID to facets mapping.
        titles: Corpus ID to title mapping.
        model: Loaded re-ranker model.
        top_n: Number of results to return after re-ranking.
        batch_size: Pairs per forward pass.

    Returns:
        Dict mapping query corpus ID to re-ranked result list.
    """
    import torch  # noqa: PLC0415

    retrieval_sets: dict[CorpusId, list[int]] = {}
    for record in iter_jsonl(retrieval_path):
        qid = CorpusId(int(record["corpusid"]))
        retrieval_sets[qid] = [int(r["corpusid"]) for r in record["results"]]

    # Build all pairs up front
    all_texts: list[str] = []
    all_meta: list[tuple[CorpusId, int]] = []  # (query_id, candidate_id)

    for qid, candidates in retrieval_sets.items():
        if qid not in facets_map or qid not in titles:
            continue
        for cid in candidates:
            ccid = CorpusId(cid)
            if ccid in facets_map and ccid in titles:
                all_texts.append(
                    format_paper_pair(
                        titles[qid], facets_map[qid], titles[ccid], facets_map[ccid]
                    )
                )
                all_meta.append((qid, cid))

    logger.info(f"Scoring {len(all_texts)} pairs with reranker...")

    with torch.inference_mode():
        all_scores = model.score_pairs(all_texts, batch_size=batch_size)

    # Group scores by query
    query_scores: dict[CorpusId, list[tuple[int, float]]] = {}
    for (qid, cid), score in zip(all_meta, all_scores.tolist()):
        query_scores.setdefault(qid, []).append((cid, score))

    reranked: dict[CorpusId, list[dict[str, Any]]] = {}
    for qid, scored in query_scores.items():
        scored.sort(key=lambda x: x[1], reverse=True)
        reranked[qid] = [
            {"corpusid": cid, "score": score} for cid, score in scored[:top_n]
        ]

    return reranked


def run_rerank_retrieval(
    retrieval_path: Path,
    facets_path: Path,
    paper_graphs_path: Path,
    model_dir: Path,
    output_path: Path,
    co_cit_path: Path,
    bib_path: Path,
    *,
    top_n: int = 20,
    batch_size: int = 32,
    device: str = "cpu",
) -> None:
    """Re-rank retrieval results with a cross-encoder and evaluate.

    Args:
        retrieval_path: Path to retrieval results JSONL.
        facets_path: Path to facets JSONL file.
        paper_graphs_path: Path to paper_graphs JSONL file (for titles).
        model_dir: Path to the fine-tuned reranker checkpoint.
        output_path: Path to write re-ranked results.
        co_cit_path: Path to co_citations JSONL file.
        bib_path: Path to bib_coupling JSONL file.
        top_n: Number of results after re-ranking.
        batch_size: Pairs per forward pass.
        device: Device for inference.
    """
    facets_map = load_facets_map(facets_path)
    titles = load_titles(paper_graphs_path)

    config = RerankerConfig(model_name=str(model_dir), device=device)
    model = LocalReranker.from_pretrained(config)

    reranked = rerank_retrieval_results(
        retrieval_path, facets_map, titles, model, top_n=top_n, batch_size=batch_size
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_auto(output_path, "w") as fh:
        for qid, results in reranked.items():
            record = {"corpusid": int(qid), "results": results}
            fh.write(json.dumps(record) + "\n")
    logger.info(f"Wrote {len(reranked)} re-ranked sets to {output_path}")

    result_lists = {
        int(qid): [int(r["corpusid"]) for r in results]
        for qid, results in reranked.items()
    }
    _evaluate_and_log(result_lists, co_cit_path, bib_path)
