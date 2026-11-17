"""Evaluate graph retrieval quality against citation ground truth.

Computes retrieval metrics (Recall@K, MRR, Hit Rate@K) for the full pipeline
and per-facet/per-signal ablations. Also analyses signal overlap and failure
cases.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from genret.processing.facet_evaluation import FACET_NAMES

if TYPE_CHECKING:
    from scipy.sparse import csr_array
from genret.processing.facet_retrieval import (
    DEFAULT_BFS_HOPS,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_RRF_K,
    DEFAULT_TOP_N,
    load_edge_graphs,
    retrieve,
)
from genret.utils import iter_jsonl, open_auto, safediv, track_progress

logger = logging.getLogger(__name__)

_KS = (5, 10, 20)
"""Recall/Hit Rate cut-offs."""


class RetrievalMetrics(BaseModel, frozen=True):
    """Aggregated retrieval metrics over a set of query papers."""

    recall: dict[int, float]
    """Recall@K for each K in `_KS`."""
    hit_rate: dict[int, float]
    """Hit Rate@K (fraction of queries with at least one GT hit in top-K)."""
    mrr: float
    """Mean Reciprocal Rank of first ground-truth hit."""
    num_queries: int
    """Number of query papers evaluated (those with GT pairs)."""
    num_gt_pairs: int
    """Total number of ground-truth pairs across all queries."""


class SignalOverlap(BaseModel, frozen=True):
    """Agreement statistics between pairs of signals."""

    pairwise_jaccard: dict[str, float]
    """Jaccard similarity of top-20 sets between each signal pair."""
    unique_by_signal: dict[str, float]
    """Fraction of top-20 candidates unique to each signal (not in any other)."""


class FailureCase(BaseModel, frozen=True):
    """A query paper where retrieval missed known citation partners."""

    corpusid: int
    """Corpus ID of the query paper."""
    missed_partners: list[int]
    """Corpus IDs of ground-truth partners not in top-20 results."""
    retrieved: list[int]
    """Corpus IDs of top-20 retrieved papers."""
    num_gt: int
    """Total number of ground-truth partners for this query."""


class FailureSummary(BaseModel, frozen=True):
    """Aggregated failure analysis."""

    total_failures: int
    """Number of queries with recall@20 < 1.0."""
    total_missed_pairs: int
    """Total missed GT pairs across all queries."""
    worst_cases: list[FailureCase]
    """Top-10 queries by number of missed GT partners."""
    missed_by_gt_degree: dict[str, int]
    """Count of missed pairs bucketed by GT degree of the query paper."""


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


def _load_gt_pairs(
    path: Path, valid_ids: frozenset[int], value_field: str
) -> dict[int, set[int]]:
    """Load citation pairs into an adjacency dict.

    Args:
        path: Path to citation JSONL file.
        valid_ids: Only keep pairs where both papers are in this set.
        value_field: Name of the count field (for logging).

    Returns:
        Dict mapping each paper to its set of citation partners.
    """
    adj: dict[int, set[int]] = {}
    n_pairs = 0

    for record in iter_jsonl(path):
        a = int(record["paper_a"])
        b = int(record["paper_b"])
        if a in valid_ids and b in valid_ids:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
            n_pairs += 1

    logger.info(f"  {value_field}: {n_pairs} pairs, {len(adj)} papers with GT")
    return adj


def load_ground_truth(
    co_cit_path: Path,
    bib_path: Path,
    valid_ids: frozenset[int],
) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, set[int]]]:
    """Load citation ground truth as adjacency dicts.

    Args:
        co_cit_path: Path to co_citations.jsonl.
        bib_path: Path to bib_coupling.jsonl.
        valid_ids: Only keep pairs where both papers are in this set.

    Returns:
        Tuple of `(combined, co_citation, bib_coupling)` adjacency dicts.
        Each maps paper ID to set of citation partners.
    """
    logger.info("Loading ground truth...")
    co_cit = _load_gt_pairs(co_cit_path, valid_ids, "Co-citation")
    bib = _load_gt_pairs(bib_path, valid_ids, "Bib-coupling")

    combined: dict[int, set[int]] = {}
    for adj in (co_cit, bib):
        for paper, partners in adj.items():
            combined.setdefault(paper, set()).update(partners)

    logger.info(f"  Combined: {len(combined)} papers with GT")
    return combined, co_cit, bib


# ---------------------------------------------------------------------------
# Retrieval set loading
# ---------------------------------------------------------------------------


def load_retrieval_sets(path: Path) -> dict[int, list[int]]:
    """Load retrieval results from JSONL.

    Args:
        path: Path to retrieval_sets JSONL file (plain, `.gz`, or `.zst`).

    Returns:
        Dict mapping query corpus ID to ordered list of result corpus IDs.
    """
    sets: dict[int, list[int]] = {}

    with open_auto(path, "r") as fh:
        for line in fh:
            record = json.loads(line)
            cid = int(record["corpusid"])
            results = [int(r["corpusid"]) for r in record["results"]]
            sets[cid] = results

    logger.info(f"Loaded {len(sets)} retrieval sets from {path}")
    return sets


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    retrieval_sets: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
) -> RetrievalMetrics:
    """Compute Recall@K, MRR, and Hit Rate@K.

    Only evaluates query papers that have at least one ground-truth partner.

    Args:
        retrieval_sets: Query corpus ID to ordered list of result corpus IDs.
        ground_truth: Query corpus ID to set of GT partner corpus IDs.

    Returns:
        Aggregated metrics across all qualifying queries.
    """
    recall_sums: dict[int, float] = dict.fromkeys(_KS, 0.0)
    hit_sums: dict[int, int] = dict.fromkeys(_KS, 0)
    mrr_sum = 0.0
    num_queries = 0
    num_gt_pairs = 0

    for cid, results in retrieval_sets.items():
        gt = ground_truth.get(cid)
        if not gt:
            continue

        num_queries += 1
        num_gt_pairs += len(gt)

        # Recall@K and Hit Rate@K
        for k in _KS:
            top_k = set(results[:k])
            hits = len(top_k & gt)
            recall_sums[k] += hits / len(gt)
            if hits > 0:
                hit_sums[k] += 1

        # MRR
        for rank, r in enumerate(results, 1):
            if r in gt:
                mrr_sum += 1.0 / rank
                break

    return RetrievalMetrics(
        recall={k: safediv(recall_sums[k], num_queries) for k in _KS},
        hit_rate={k: safediv(hit_sums[k], num_queries) for k in _KS},
        mrr=safediv(mrr_sum, num_queries),
        num_queries=num_queries,
        num_gt_pairs=num_gt_pairs,
    )


# ---------------------------------------------------------------------------
# Per-facet ablation
# ---------------------------------------------------------------------------


def _evaluate_per_facet(
    edges_dir: Path,
    corpusids: npt.NDArray[np.int64],
    ground_truth: dict[int, set[int]],
    bfs_hops: int = DEFAULT_BFS_HOPS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, RetrievalMetrics]:
    """Run single-facet retrieval (no RRF fusion) and evaluate each facet.

    Args:
        edges_dir: Directory containing `paper_edges_*.npz` files.
        corpusids: Ordered corpus IDs (index to ID mapping).
        ground_truth: Combined GT adjacency dict.
        bfs_hops: Number of BFS hops.
        max_candidates: Hard cap on candidates per BFS walk.
        top_n: Number of results per query.

    Returns:
        Dict mapping facet name to retrieval metrics.
    """
    graphs, _ = load_edge_graphs(edges_dir)
    cid_to_idx = {int(cid): i for i, cid in enumerate(corpusids)}
    per_facet: dict[str, RetrievalMetrics] = {}

    for facet_name in FACET_NAMES:
        if facet_name not in graphs:
            continue

        graph = graphs[facet_name]
        single_graphs: dict[str, csr_array] = {facet_name: graph}

        retrieval_sets: dict[int, list[int]] = {}
        query_ids = [cid for cid in ground_truth if cid in cid_to_idx]

        for cid in track_progress(query_ids, desc=f"  {facet_name}..."):
            idx = cid_to_idx[cid]
            results = retrieve(
                idx,
                single_graphs,
                bfs_hops=bfs_hops,
                max_candidates=max_candidates,
                rrf_k=DEFAULT_RRF_K,
                top_n=top_n,
            )
            retrieval_sets[cid] = [int(corpusids[c.index]) for c in results]

        per_facet[facet_name] = compute_metrics(retrieval_sets, ground_truth)
        r20 = per_facet[facet_name].recall[20]
        logger.info(f"  {facet_name}: Recall@20={r20:.4f}")

    return per_facet


# ---------------------------------------------------------------------------
# Per-signal ablation (candidate level)
# ---------------------------------------------------------------------------


def _evaluate_per_signal(
    candidates_dir: Path,
    ground_truth: dict[int, set[int]],
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, RetrievalMetrics]:
    """Re-rank candidates by individual score components and evaluate.

    Uses the candidate-level component scores (`scores_dense`, `scores_ngram`,
    `scores_citation`) to produce alternative rankings without re-running
    candidate generation.

    Args:
        candidates_dir: Directory containing `facet_candidates_*.npz` files.
        ground_truth: Combined GT adjacency dict.
        top_n: Number of results per query.

    Returns:
        Dict mapping signal configuration name to retrieval metrics.
    """
    configs: dict[str, tuple[float, float, float]] = {
        "dense_only": (1.0, 0.0, 0.0),
        "dense+ngram": (0.75, 0.25, 0.0),
        "dense+citation": (0.75, 0.0, 0.25),
        "all_signals": (0.6, 0.2, 0.2),
    }

    # Aggregate candidates across all facets per paper, then re-rank
    # For simplicity, use the first available facet's candidates as representative
    # (the signals are per-facet anyway; we evaluate each facet's candidates)
    results_by_config: dict[str, dict[int, list[int]]] = {name: {} for name in configs}

    for facet_name in FACET_NAMES:
        npz_path = candidates_dir / f"facet_candidates_{facet_name}.npz"
        if not npz_path.exists():
            continue

        with np.load(npz_path) as npz:
            corpusids: npt.NDArray[np.int64] = npz["corpusids"]
            candidates: npt.NDArray[np.int32] = npz["candidates"]
            scores_dense: npt.NDArray[np.float16] = npz["scores_dense"]
            scores_ngram: npt.NDArray[np.float16] = npz["scores_ngram"]
            scores_citation: npt.NDArray[np.float16] = npz["scores_citation"]

        n = len(corpusids)

        for config_name, (a, b, g) in configs.items():
            combined = (
                a * scores_dense.astype(np.float32)
                + b * scores_ngram.astype(np.float32)
                + g * scores_citation.astype(np.float32)
            )

            for i in range(n):
                cid = int(corpusids[i])
                gt = ground_truth.get(cid)
                if not gt:
                    continue

                # Sort by re-weighted score descending
                order = np.argsort(-combined[i])[:top_n]
                valid = candidates[i, order]
                valid = valid[valid >= 0]
                result_cids = [int(corpusids[j]) for j in valid]

                # Merge across facets using RRF-style fusion
                existing = results_by_config[config_name].get(cid, [])
                if not existing:
                    results_by_config[config_name][cid] = result_cids
                else:
                    # Simple RRF merge across facets
                    scores: dict[int, float] = {}
                    for rank, r_cid in enumerate(existing):
                        scores[r_cid] = scores.get(r_cid, 0.0) + 1.0 / (60 + rank + 1)
                    for rank, r_cid in enumerate(result_cids):
                        scores[r_cid] = scores.get(r_cid, 0.0) + 1.0 / (60 + rank + 1)
                    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    results_by_config[config_name][cid] = [c for c, _ in merged[:top_n]]

    per_signal: dict[str, RetrievalMetrics] = {}
    for config_name, retrieval_sets in results_by_config.items():
        metrics = compute_metrics(retrieval_sets, ground_truth)
        per_signal[config_name] = metrics
        r20 = metrics.recall[20]
        logger.info(f"  {config_name}: Recall@20={r20:.4f}")

    return per_signal


# ---------------------------------------------------------------------------
# Signal overlap analysis
# ---------------------------------------------------------------------------


def _analyse_signal_overlap(
    candidates_dir: Path,
    top_n: int = DEFAULT_TOP_N,
) -> SignalOverlap:
    """Analyse agreement between the three scoring signals.

    For each paper, takes the top-`top_n` candidates by each signal and
    computes pairwise Jaccard similarity and unique-to-signal fractions.

    Args:
        candidates_dir: Directory containing `facet_candidates_*.npz` files.
        top_n: Number of top candidates to compare per signal.

    Returns:
        Signal overlap statistics.
    """
    signal_names = ("dense", "ngram", "citation")
    jaccard_sums: dict[str, float] = {}
    unique_sums: dict[str, float] = {}
    n_papers = 0

    for facet_name in FACET_NAMES:
        npz_path = candidates_dir / f"facet_candidates_{facet_name}.npz"
        if not npz_path.exists():
            continue

        with np.load(npz_path) as npz:
            candidates: npt.NDArray[np.int32] = npz["candidates"]
            scores_dense: npt.NDArray[np.float16] = npz["scores_dense"]
            scores_ngram: npt.NDArray[np.float16] = npz["scores_ngram"]
            scores_citation: npt.NDArray[np.float16] = npz["scores_citation"]

        score_arrays = {
            "dense": scores_dense,
            "ngram": scores_ngram,
            "citation": scores_citation,
        }
        n = candidates.shape[0]

        for i in range(n):
            top_sets: dict[str, set[int]] = {}
            for sig_name in signal_names:
                order = np.argsort(-score_arrays[sig_name][i].astype(np.float32))[
                    :top_n
                ]
                valid_cands = candidates[i, order]
                top_sets[sig_name] = {int(c) for c in valid_cands if c >= 0}

            # Pairwise Jaccard
            for j, s1 in enumerate(signal_names):
                for s2 in signal_names[j + 1 :]:
                    pair_key = f"{s1}_vs_{s2}"
                    a_set = top_sets[s1]
                    b_set = top_sets[s2]
                    union_size = len(a_set | b_set)
                    if union_size > 0:
                        jacc = len(a_set & b_set) / union_size
                    else:
                        jacc = 0.0
                    jaccard_sums[pair_key] = jaccard_sums.get(pair_key, 0.0) + jacc

            # Unique to each signal
            all_others = set[int]()
            for sig_name in signal_names:
                others = set[int]()
                for other_name in signal_names:
                    if other_name != sig_name:
                        others |= top_sets[other_name]
                unique = top_sets[sig_name] - others
                frac = (
                    safediv(len(unique), len(top_sets[sig_name]))
                    if top_sets[sig_name]
                    else 0.0
                )
                unique_sums[sig_name] = unique_sums.get(sig_name, 0.0) + frac
                all_others |= others  # suppress unused warning

            n_papers += 1

    total = max(n_papers, 1)
    return SignalOverlap(
        pairwise_jaccard={k: v / total for k, v in jaccard_sums.items()},
        unique_by_signal={k: v / total for k, v in unique_sums.items()},
    )


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------


def _analyse_failures(
    retrieval_sets: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    max_cases: int = 10,
) -> FailureSummary:
    """Analyse cases where retrieval missed known citation partners.

    Args:
        retrieval_sets: Query corpus ID to ordered list of result corpus IDs.
        ground_truth: Combined GT adjacency dict.
        max_cases: Number of worst-case examples to include.

    Returns:
        Failure summary with worst cases and degree-bucketed miss counts.
    """
    failures: list[FailureCase] = []
    total_missed = 0
    missed_by_degree: Counter[str] = Counter()

    for cid, results in retrieval_sets.items():
        gt = ground_truth.get(cid)
        if not gt:
            continue

        retrieved_set = set(results[:20])
        missed = gt - retrieved_set
        if not missed:
            continue

        total_missed += len(missed)
        failures.append(
            FailureCase(
                corpusid=cid,
                missed_partners=sorted(missed),
                retrieved=results[:20],
                num_gt=len(gt),
            )
        )

        # Bucket by GT degree
        degree = len(gt)
        if degree <= 5:
            bucket = "1-5"
        elif degree <= 20:
            bucket = "6-20"
        elif degree <= 50:
            bucket = "21-50"
        else:
            bucket = "51+"
        missed_by_degree[bucket] += len(missed)

    failures.sort(key=lambda f: len(f.missed_partners), reverse=True)

    return FailureSummary(
        total_failures=len(failures),
        total_missed_pairs=total_missed,
        worst_cases=failures[:max_cases],
        missed_by_gt_degree=dict(missed_by_degree),
    )


# ---------------------------------------------------------------------------
# Dense metric comparison
# ---------------------------------------------------------------------------


def _evaluate_dense_metrics(
    dense_dir: Path,
    ground_truth: dict[int, set[int]],
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, RetrievalMetrics]:
    """Compare mean_pool, max_pairs, and mean_pairs as dense signals.

    For each metric, ranks all papers by that metric's score and evaluates
    retrieval quality against citation ground truth.

    Args:
        dense_dir: Directory containing `facet_graph_*.npz` files.
        ground_truth: Combined GT adjacency dict.
        top_n: Number of results per query.

    Returns:
        Dict mapping metric name to retrieval metrics.
    """
    metric_names = ("mean_pool", "max_pairs", "mean_pairs")
    results_by_metric: dict[str, dict[int, list[int]]] = {
        name: {} for name in metric_names
    }

    for facet_name in FACET_NAMES:
        npz_path = dense_dir / f"facet_graph_{facet_name}.npz"
        if not npz_path.exists():
            continue

        with np.load(npz_path) as npz:
            corpusids: npt.NDArray[np.int64] = npz["corpusids"]
            available_metrics = {
                name: npz[name] for name in metric_names if name in npz
            }

        n = len(corpusids)

        for metric_name, matrix in available_metrics.items():
            for i in range(n):
                cid = int(corpusids[i])
                if cid not in ground_truth:
                    continue

                row = matrix[i].astype(np.float32)
                row[i] = -1.0  # exclude self
                order = np.argsort(-row)[:top_n]
                result_cids = [int(corpusids[j]) for j in order]

                # RRF merge across facets
                existing = results_by_metric[metric_name].get(cid, [])
                if not existing:
                    results_by_metric[metric_name][cid] = result_cids
                else:
                    scores: dict[int, float] = {}
                    for rank, r_cid in enumerate(existing):
                        scores[r_cid] = scores.get(r_cid, 0.0) + 1.0 / (60 + rank + 1)
                    for rank, r_cid in enumerate(result_cids):
                        scores[r_cid] = scores.get(r_cid, 0.0) + 1.0 / (60 + rank + 1)
                    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    results_by_metric[metric_name][cid] = [c for c, _ in merged[:top_n]]

    per_metric: dict[str, RetrievalMetrics] = {}
    for metric_name in metric_names:
        if metric_name not in results_by_metric or not results_by_metric[metric_name]:
            continue
        metrics = compute_metrics(results_by_metric[metric_name], ground_truth)
        per_metric[metric_name] = metrics
        r20 = metrics.recall[20]
        logger.info(f"  {metric_name}: Recall@20={r20:.4f}")

    return per_metric


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


class FullEvaluationReport(BaseModel, frozen=True):
    """Complete evaluation output including all analyses."""

    overall: RetrievalMetrics
    """Metrics for the full pipeline (BFS + RRF over all facets)."""
    by_gt_signal: dict[str, RetrievalMetrics]
    """Metrics broken down by GT source (co_citation, bib_coupling)."""
    per_facet: dict[str, RetrievalMetrics]
    """Metrics using single-facet retrieval (no RRF)."""
    per_signal: dict[str, RetrievalMetrics]
    """Metrics re-ranking candidates by individual score components."""
    dense_metrics: dict[str, RetrievalMetrics]
    """Metrics comparing mean_pool vs max_pairs vs mean_pairs."""
    signal_overlap: SignalOverlap
    """Agreement statistics between the three scoring signals."""
    failure_summary: FailureSummary
    """Summary of retrieval failures."""


def evaluate(
    retrieval_path: Path,
    edges_dir: Path,
    candidates_dir: Path,
    dense_dir: Path,
    co_cit_path: Path,
    bib_path: Path,
) -> FullEvaluationReport:
    """Run the full evaluation pipeline.

    Args:
        retrieval_path: Path to retrieval_sets JSONL file.
        edges_dir: Directory containing `paper_edges_*.npz` files.
        candidates_dir: Directory containing `facet_candidates_*.npz` files.
        dense_dir: Directory containing `facet_graph_*.npz` files.
        co_cit_path: Path to co_citations.jsonl.
        bib_path: Path to bib_coupling.jsonl.

    Returns:
        Full evaluation report with all analyses.
    """
    retrieval_sets = load_retrieval_sets(retrieval_path)
    valid_ids = frozenset(retrieval_sets.keys())

    combined_gt, co_cit_gt, bib_gt = load_ground_truth(co_cit_path, bib_path, valid_ids)

    # Load corpusids from first available edge graph
    _, corpusids = load_edge_graphs(edges_dir)

    # Overall metrics
    logger.info("Overall metrics...")
    overall = compute_metrics(retrieval_sets, combined_gt)
    logger.info(
        f"  Recall@5={overall.recall[5]:.4f}, "
        f"Recall@10={overall.recall[10]:.4f}, "
        f"Recall@20={overall.recall[20]:.4f}, "
        f"MRR={overall.mrr:.4f}"
    )

    # By GT signal
    logger.info("Metrics by GT signal...")
    by_gt_signal = {
        "co_citation": compute_metrics(retrieval_sets, co_cit_gt),
        "bib_coupling": compute_metrics(retrieval_sets, bib_gt),
    }

    # Per-facet ablation
    logger.info("Per-facet ablation...")
    per_facet = _evaluate_per_facet(edges_dir, corpusids, combined_gt)

    # Per-signal ablation
    logger.info("Per-signal ablation...")
    per_signal = _evaluate_per_signal(candidates_dir, combined_gt)

    # Dense metric comparison
    logger.info("Dense metric comparison...")
    dense_metrics = _evaluate_dense_metrics(dense_dir, combined_gt)

    # Signal overlap
    logger.info("Signal overlap analysis...")
    signal_overlap = _analyse_signal_overlap(candidates_dir)

    # Failure analysis
    logger.info("Failure analysis...")
    failure_summary = _analyse_failures(retrieval_sets, combined_gt)

    return FullEvaluationReport(
        overall=overall,
        by_gt_signal=by_gt_signal,
        per_facet=per_facet,
        per_signal=per_signal,
        dense_metrics=dense_metrics,
        signal_overlap=signal_overlap,
        failure_summary=failure_summary,
    )
