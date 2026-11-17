"""Compare reranking quality across retriever, GPT, and local judge.

Loads per-query retrieval results and two sets of pair-level scores (GPT
boolean judgements and local judge continuous scores), builds per-query
rankings from each, and computes ranking agreement metrics.
"""

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genret.processing.llm_judge import (
    JudgedPair,
    ScoredPair,
    normalise_judged_pair_dict,
)
from genret.utils import open_auto

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class QueryRankingMetrics:
    """Ranking comparison metrics for a single query."""

    corpusid: int
    """Corpus ID of the query paper."""
    degree_bin: str
    """Citation degree bin: `sparse`, `low_mid`, or `high`."""
    n_candidates: int
    """Number of candidates retrieved."""
    n_gpt_related: int
    """Number of candidates GPT marked as related."""
    overlap_at_20: float
    """Fraction of GPT top-20 that appear in local judge top-20."""
    kendall_tau: float
    """Kendall tau between GPT and local judge rankings."""
    ndcg_retriever_at_20: float
    """nDCG@20 of the retriever ranking against GPT relevance."""
    ndcg_local_at_20: float
    """nDCG@20 of the local judge ranking against GPT relevance."""


@dataclass(frozen=True, kw_only=True)
class RerankingSummary:
    """Aggregated reranking comparison results."""

    total_queries: int
    """Total number of queries evaluated."""
    per_query: list[QueryRankingMetrics]
    """Per-query metrics."""
    overall: BinSummary
    """Overall aggregated metrics."""
    by_bin: dict[str, BinSummary]
    """Metrics aggregated by degree bin."""


@dataclass(frozen=True, kw_only=True)
class BinSummary:
    """Aggregated metrics for a group of queries."""

    n_queries: int
    """Number of queries in this group."""
    mean_overlap_at_20: float
    """Mean overlap@20 across queries."""
    mean_kendall_tau: float
    """Mean Kendall tau across queries."""
    mean_ndcg_retriever_at_20: float
    """Mean nDCG@20 for the retriever ranking."""
    mean_ndcg_local_at_20: float
    """Mean nDCG@20 for the local judge ranking."""
    ndcg_lift: float
    """Mean nDCG lift from local judge over retriever."""


@dataclass(frozen=True, kw_only=True)
class ScoreResult:
    """A single candidate's retrieval score."""

    corpusid: int
    score: float


def _load_retrieval_results(path: Path) -> dict[int, list[ScoreResult]]:
    """Load retrieval results as query_corpusid -> [(candidate_id, score), ...].

    Args:
        path: Path to retrieval JSONL file.

    Returns:
        Dict mapping query corpus ID to ranked list of (candidate_id, score).
    """
    results: dict[int, list[ScoreResult]] = {}
    with open_auto(path, "r") as f:
        for line in f:
            rec: dict[str, Any] = json.loads(line)
            results[rec["corpusid"]] = [
                ScoreResult(corpusid=r["corpusid"], score=r["score"])
                for r in rec["results"]
            ]
    return results


def _load_query_metadata(path: Path) -> dict[int, str]:
    """Load query paper metadata with degree bins.

    Args:
        path: Path to query papers JSONL with `corpusid` and `bin` fields.

    Returns:
        Dict mapping corpus ID to degree bin name.
    """
    bins: dict[int, str] = {}
    with open_auto(path, "r") as f:
        for line in f:
            rec: dict[str, Any] = json.loads(line)
            bins[rec["corpusid"]] = rec["bin"]
    return bins


def _load_gpt_judgements_by_query(
    path: Path,
) -> dict[int, dict[int, JudgedPair]]:
    """Load GPT judgements grouped by query (paper_a).

    Args:
        path: Path to GPT judgements JSONL.

    Returns:
        Dict mapping query corpus ID to dict of candidate_id -> JudgedPair.
    """
    by_query: defaultdict[int, dict[int, JudgedPair]] = defaultdict(dict)
    with open_auto(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            data: dict[str, Any] = json.loads(stripped)
            pair = JudgedPair.model_validate(normalise_judged_pair_dict(data))
            by_query[pair.paper_a][pair.paper_b] = pair
    return dict(by_query)


def _load_local_scores_by_query(
    path: Path,
) -> dict[int, dict[int, float]]:
    """Load local judge scores grouped by query (paper_a).

    Args:
        path: Path to local judge scored JSONL.

    Returns:
        Dict mapping query corpus ID to dict of candidate_id -> score.
    """
    by_query: defaultdict[int, dict[int, float]] = defaultdict(dict)
    with open_auto(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            pair = ScoredPair.model_validate_json(stripped)
            by_query[pair.paper_a][pair.paper_b] = pair.score
    return dict(by_query)


def _dcg(relevances: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k.

    Args:
        relevances: Relevance scores in rank order.
        k: Cutoff rank.

    Returns:
        DCG@k value.
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def calc_ndcg(relevances: list[float], k: int) -> float:
    """Compute normalised DCG at rank k.

    Args:
        relevances: Relevance scores in rank order.
        k: Cutoff rank.

    Returns:
        nDCG@k value. Returns 0.0 if ideal DCG is zero.
    """
    actual = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def kendall_tau(ranking_a: list[int], ranking_b: list[int]) -> float:
    """Compute Kendall tau-b between two rankings of the same items.

    Args:
        ranking_a: List of item IDs in rank order (best first).
        ranking_b: List of item IDs in rank order (best first).

    Returns:
        Kendall tau-b correlation. Returns 0.0 if undefined.
    """
    common = set(ranking_a) & set(ranking_b)
    if len(common) < 2:
        return 0.0

    rank_a = {item: i for i, item in enumerate(ranking_a) if item in common}
    rank_b = {item: i for i, item in enumerate(ranking_b) if item in common}

    items = sorted(common)
    concordant = 0
    discordant = 0

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a_diff = rank_a[items[i]] - rank_a[items[j]]
            b_diff = rank_b[items[i]] - rank_b[items[j]]
            product = a_diff * b_diff
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def convert_gpt_relevance(pair: JudgedPair) -> float:
    """Convert a GPT judgement to a graded relevance score.

    Related pairs get `0.5 + 0.5 * confidence` (range 0.5-1.0).
    Unrelated pairs get `0.5 - 0.5 * confidence` (range 0.0-0.5).

    Args:
        pair: GPT judgement for a paper pair.

    Returns:
        Graded relevance score in [0, 1].
    """
    if pair.related:
        return 0.5 + 0.5 * pair.confidence
    return 0.5 - 0.5 * pair.confidence


def _aggregate_bin(metrics: list[QueryRankingMetrics]) -> BinSummary:
    """Aggregate per-query metrics into a bin summary.

    Args:
        metrics: List of per-query metrics.

    Returns:
        Aggregated summary for the group.
    """
    n = len(metrics)
    if n == 0:
        return BinSummary(
            n_queries=0,
            mean_overlap_at_20=0.0,
            mean_kendall_tau=0.0,
            mean_ndcg_retriever_at_20=0.0,
            mean_ndcg_local_at_20=0.0,
            ndcg_lift=0.0,
        )

    mean_overlap = sum(m.overlap_at_20 for m in metrics) / n
    mean_tau = sum(m.kendall_tau for m in metrics) / n
    mean_ndcg_ret = sum(m.ndcg_retriever_at_20 for m in metrics) / n
    mean_ndcg_local = sum(m.ndcg_local_at_20 for m in metrics) / n

    return BinSummary(
        n_queries=n,
        mean_overlap_at_20=mean_overlap,
        mean_kendall_tau=mean_tau,
        mean_ndcg_retriever_at_20=mean_ndcg_ret,
        mean_ndcg_local_at_20=mean_ndcg_local,
        ndcg_lift=mean_ndcg_local - mean_ndcg_ret,
    )


def compare_reranking(
    retrieval_path: Path,
    gpt_path: Path,
    local_path: Path,
    query_meta_path: Path,
    k: int = 20,
) -> RerankingSummary:
    """Compare retriever, GPT, and local judge rankings per query.

    For each query, builds three rankings of the retrieval candidates:
    1. Retriever: original RRF order
    2. GPT-5-mini: sorted by graded relevance (related + confidence)
    3. Local judge: sorted by continuous score

    Computes nDCG@k for the retriever and local judge rankings using
    GPT relevance as ground truth, plus overlap and rank correlation
    between GPT and local judge.

    Args:
        retrieval_path: Path to retrieval results JSONL.
        gpt_path: Path to GPT judgements JSONL.
        local_path: Path to local judge scored JSONL.
        query_meta_path: Path to query papers JSONL with degree bins.
        k: Cutoff rank for nDCG and overlap.

    Returns:
        Summary with per-query and aggregated metrics.
    """
    retrieval = _load_retrieval_results(retrieval_path)
    gpt_by_query = _load_gpt_judgements_by_query(gpt_path)
    local_by_query = _load_local_scores_by_query(local_path)
    query_bins = _load_query_metadata(query_meta_path)

    per_query: list[QueryRankingMetrics] = []

    for query_id, retrieval_candidates in sorted(retrieval.items()):
        gpt_candidates = gpt_by_query.get(query_id, {})
        local_candidates = local_by_query.get(query_id, {})
        degree_bin = query_bins.get(query_id, "unknown")

        candidate_ids = [score.corpusid for score in retrieval_candidates]

        # Filter to candidates with both GPT and local scores
        scored_ids = [
            cid
            for cid in candidate_ids
            if cid in gpt_candidates and cid in local_candidates
        ]

        if not scored_ids:
            logger.warning(
                "Query %d: no candidates with both GPT and local scores", query_id
            )
            continue

        # Build relevance lookup from GPT
        gpt_relevance = {
            cid: convert_gpt_relevance(gpt_candidates[cid]) for cid in scored_ids
        }
        n_related = sum(1 for cid in scored_ids if gpt_candidates[cid].related)

        # Retriever ranking: original order, filtered to scored candidates
        retriever_ranking = [cid for cid in candidate_ids if cid in gpt_relevance]
        retriever_relevances = [gpt_relevance[cid] for cid in retriever_ranking]

        # GPT ranking: sort by graded relevance descending
        gpt_ranking = sorted(
            scored_ids, key=lambda cid: gpt_relevance[cid], reverse=True
        )
        # Local judge ranking: sort by score descending
        local_ranking = sorted(
            scored_ids, key=lambda cid: local_candidates[cid], reverse=True
        )
        local_relevances = [gpt_relevance[cid] for cid in local_ranking]

        # Overlap@k: fraction of GPT top-k in local top-k
        gpt_top_k = set(gpt_ranking[:k])
        local_top_k = set(local_ranking[:k])
        effective_k = min(k, len(scored_ids))
        overlap = len(gpt_top_k & local_top_k) / effective_k if effective_k > 0 else 0.0

        # Kendall tau between GPT and local rankings
        tau = kendall_tau(gpt_ranking, local_ranking)

        # nDCG@k
        ndcg_ret = calc_ndcg(retriever_relevances, k)
        ndcg_local = calc_ndcg(local_relevances, k)

        per_query.append(
            QueryRankingMetrics(
                corpusid=query_id,
                degree_bin=degree_bin,
                n_candidates=len(scored_ids),
                n_gpt_related=n_related,
                overlap_at_20=overlap,
                kendall_tau=tau,
                ndcg_retriever_at_20=ndcg_ret,
                ndcg_local_at_20=ndcg_local,
            )
        )

    # Aggregate
    overall = _aggregate_bin(per_query)

    by_bin: dict[str, BinSummary] = {}
    bin_names = sorted({m.degree_bin for m in per_query})
    for bin_name in bin_names:
        bin_metrics = [m for m in per_query if m.degree_bin == bin_name]
        by_bin[bin_name] = _aggregate_bin(bin_metrics)

    return RerankingSummary(
        total_queries=len(per_query),
        per_query=per_query,
        overall=overall,
        by_bin=by_bin,
    )


def _rerank_at_depth(
    retriever_ranking: list[int],
    local_scores: dict[int, float],
    n: int,
) -> list[int]:
    """Rerank top `n` candidates by local score, keep the rest in retriever order.

    Args:
        retriever_ranking: Full retriever ranking (candidate IDs, best first).
        local_scores: Local judge score per candidate ID.
        n: Number of top candidates to rerank.

    Returns:
        Combined ranking: reranked top-n followed by remaining in retriever order.
    """
    top_n = retriever_ranking[:n]
    rest = retriever_ranking[n:]
    reranked = sorted(top_n, key=lambda cid: local_scores.get(cid, 0.0), reverse=True)
    return reranked + rest


@dataclass(frozen=True, kw_only=True)
class DepthResult:
    """nDCG@k results for one reranking depth across all queries."""

    rerank_depth: int
    """Number of top candidates reranked by the local judge."""
    overall_ndcg: float
    """Overall mean nDCG@k."""
    by_bin: dict[str, float]
    """Mean nDCG@k per degree bin."""


@dataclass(frozen=True, kw_only=True)
class DepthSweepResult:
    """Results of a reranking depth sweep."""

    eval_k: int
    """nDCG cutoff rank."""
    retriever_baseline: DepthResult
    """Retriever-only baseline (rerank depth 0)."""
    depths: list[DepthResult]
    """Results for each reranking depth."""


def compare_reranking_depths(
    retrieval_path: Path,
    gpt_path: Path,
    local_path: Path,
    query_meta_path: Path,
    rerank_depths: list[int],
    eval_k: int = 20,
) -> DepthSweepResult:
    """Compare nDCG@k at different reranking depths.

    For each query, simulates reranking only the top `n` retrieval candidates
    by local judge score, leaving the rest in retriever order. Measures
    nDCG@eval_k for each depth.

    Args:
        retrieval_path: Path to retrieval results JSONL.
        gpt_path: Path to GPT judgements JSONL.
        local_path: Path to local judge scored JSONL.
        query_meta_path: Path to query papers JSONL with degree bins.
        rerank_depths: List of reranking depths to evaluate (e.g. [10, 20, 50]).
        eval_k: Cutoff rank for nDCG measurement.

    Returns:
        Sweep results with baseline and per-depth metrics.
    """
    retrieval = _load_retrieval_results(retrieval_path)
    gpt_by_query = _load_gpt_judgements_by_query(gpt_path)
    local_by_query = _load_local_scores_by_query(local_path)
    query_bins = _load_query_metadata(query_meta_path)

    # Per-query, per-depth nDCG values
    per_query_results: list[dict[str, Any]] = []

    for query_id, retrieval_candidates in sorted(retrieval.items()):
        gpt_candidates = gpt_by_query.get(query_id, {})
        local_candidates = local_by_query.get(query_id, {})
        degree_bin = query_bins.get(query_id, "unknown")

        candidate_ids = [r.corpusid for r in retrieval_candidates]

        scored_ids_set = {
            cid
            for cid in candidate_ids
            if cid in gpt_candidates and cid in local_candidates
        }

        if not scored_ids_set:
            continue

        gpt_relevance = {
            cid: convert_gpt_relevance(gpt_candidates[cid]) for cid in scored_ids_set
        }

        # Retriever ranking filtered to scored candidates
        retriever_ranking = [cid for cid in candidate_ids if cid in scored_ids_set]
        retriever_rels = [gpt_relevance[cid] for cid in retriever_ranking]
        baseline_ndcg = calc_ndcg(retriever_rels, eval_k)

        depth_ndcgs: dict[int, float] = {}
        for depth in rerank_depths:
            reranked = _rerank_at_depth(retriever_ranking, local_candidates, depth)
            rels = [gpt_relevance[cid] for cid in reranked]
            depth_ndcgs[depth] = calc_ndcg(rels, eval_k)

        per_query_results.append(
            {
                "degree_bin": degree_bin,
                "baseline_ndcg": baseline_ndcg,
                "depth_ndcgs": depth_ndcgs,
            }
        )

    # Aggregate
    bin_names = sorted({r["degree_bin"] for r in per_query_results})

    def _mean_ndcg(results: list[dict[str, Any]], key: str | int) -> float:
        """Compute mean nDCG for a set of query results.

        Args:
            results: List of per-query result dicts.
            key: Either `"baseline"` for the retriever baseline or an int depth.

        Returns:
            Mean nDCG across the given results.
        """
        if not results:
            return 0.0
        if key == "baseline":
            return sum(float(r["baseline_ndcg"]) for r in results) / len(results)
        return sum(float(r["depth_ndcgs"][key]) for r in results) / len(results)

    baseline = DepthResult(
        rerank_depth=0,
        overall_ndcg=_mean_ndcg(per_query_results, "baseline"),
        by_bin={
            b: _mean_ndcg(
                [r for r in per_query_results if r["degree_bin"] == b], "baseline"
            )
            for b in bin_names
        },
    )

    depths: list[DepthResult] = []
    for depth in rerank_depths:
        depths.append(
            DepthResult(
                rerank_depth=depth,
                overall_ndcg=_mean_ndcg(per_query_results, depth),
                by_bin={
                    b: _mean_ndcg(
                        [r for r in per_query_results if r["degree_bin"] == b], depth
                    )
                    for b in bin_names
                },
            )
        )

    return DepthSweepResult(
        eval_k=eval_k,
        retriever_baseline=baseline,
        depths=depths,
    )


def _normalise_scores(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalise scores to [0, 1].

    Args:
        scores: Raw scores keyed by candidate ID.

    Returns:
        Normalised scores. Returns all zeros if min == max.
    """
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return dict.fromkeys(scores, 0.0)
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _rerank_blended(
    retriever_ranking: list[int],
    rrf_scores: dict[int, float],
    local_scores: dict[int, float],
    n: int,
    alpha: float,
) -> list[int]:
    """Rerank top `n` by blended RRF + judge score, keep the rest in order.

    Blended score: `alpha * norm_rrf + (1 - alpha) * judge_score`.

    Args:
        retriever_ranking: Full retriever ranking (candidate IDs, best first).
        rrf_scores: Raw RRF scores per candidate ID.
        local_scores: Local judge scores (0-1) per candidate ID.
        n: Number of top candidates to rerank.
        alpha: Weight for the normalised RRF score (0 = pure judge, 1 = pure RRF).

    Returns:
        Combined ranking: blended top-n followed by remaining in retriever order.
    """
    top_n = retriever_ranking[:n]
    rest = retriever_ranking[n:]

    # Normalise RRF scores within the reranking window
    window_rrf = {cid: rrf_scores.get(cid, 0.0) for cid in top_n}
    norm_rrf = _normalise_scores(window_rrf)

    def _blended(cid: int) -> float:
        """Compute blended score for a candidate.

        Args:
            cid: Candidate corpus ID.

        Returns:
            Blended score.
        """
        return alpha * norm_rrf.get(cid, 0.0) + (1 - alpha) * local_scores.get(cid, 0.0)

    reranked = sorted(top_n, key=_blended, reverse=True)
    return reranked + rest


@dataclass(frozen=True, kw_only=True)
class AlphaResult:
    """nDCG@k results for one alpha value."""

    alpha: float
    """RRF weight (0 = pure judge, 1 = pure RRF)."""
    overall_ndcg: float
    """Overall mean nDCG@k."""
    by_bin: dict[str, float]
    """Mean nDCG@k per degree bin."""


@dataclass(frozen=True, kw_only=True)
class AlphaSweepResult:
    """Results of an alpha blending sweep."""

    eval_k: int
    """nDCG cutoff rank."""
    rerank_depth: int
    """Number of top candidates reranked."""
    results: list[AlphaResult]
    """Results for each alpha value."""


def compare_alpha_blending(
    retrieval_path: Path,
    gpt_path: Path,
    local_path: Path,
    query_meta_path: Path,
    alphas: list[float],
    rerank_depth: int = 50,
    eval_k: int = 20,
) -> AlphaSweepResult:
    """Sweep alpha values for blended RRF + judge reranking.

    For each query, normalises RRF scores within the top-N window and
    computes `alpha * norm_rrf + (1 - alpha) * judge_score` to rerank.

    Args:
        retrieval_path: Path to retrieval results JSONL.
        gpt_path: Path to GPT judgements JSONL.
        local_path: Path to local judge scored JSONL.
        query_meta_path: Path to query papers JSONL with degree bins.
        alphas: List of alpha values to sweep (0 = pure judge, 1 = pure RRF).
        rerank_depth: Number of top candidates to rerank.
        eval_k: Cutoff rank for nDCG measurement.

    Returns:
        Sweep results with per-alpha metrics.
    """
    retrieval = _load_retrieval_results(retrieval_path)
    gpt_by_query = _load_gpt_judgements_by_query(gpt_path)
    local_by_query = _load_local_scores_by_query(local_path)
    query_bins = _load_query_metadata(query_meta_path)

    per_query_results: list[dict[str, Any]] = []

    for query_id, retrieval_candidates in sorted(retrieval.items()):
        gpt_candidates = gpt_by_query.get(query_id, {})
        local_candidates = local_by_query.get(query_id, {})
        degree_bin = query_bins.get(query_id, "unknown")

        candidate_ids = [r.corpusid for r in retrieval_candidates]
        rrf_scores = {r.corpusid: r.score for r in retrieval_candidates}

        scored_ids_set = {
            cid
            for cid in candidate_ids
            if cid in gpt_candidates and cid in local_candidates
        }

        if not scored_ids_set:
            continue

        gpt_relevance = {
            cid: convert_gpt_relevance(gpt_candidates[cid]) for cid in scored_ids_set
        }

        retriever_ranking = [cid for cid in candidate_ids if cid in scored_ids_set]

        alpha_ndcgs: dict[float, float] = {}
        for alpha in alphas:
            reranked = _rerank_blended(
                retriever_ranking,
                rrf_scores,
                local_candidates,
                rerank_depth,
                alpha,
            )
            rels = [gpt_relevance[cid] for cid in reranked]
            alpha_ndcgs[alpha] = calc_ndcg(rels, eval_k)

        per_query_results.append(
            {
                "degree_bin": degree_bin,
                "alpha_ndcgs": alpha_ndcgs,
            }
        )

    bin_names = sorted({r["degree_bin"] for r in per_query_results})

    results: list[AlphaResult] = []
    for alpha in alphas:
        by_bin: dict[str, float] = {}
        for b in bin_names:
            bin_results = [r for r in per_query_results if r["degree_bin"] == b]
            by_bin[b] = (
                sum(r["alpha_ndcgs"][alpha] for r in bin_results) / len(bin_results)
                if bin_results
                else 0.0
            )

        overall = sum(r["alpha_ndcgs"][alpha] for r in per_query_results) / len(
            per_query_results
        )
        results.append(AlphaResult(alpha=alpha, overall_ndcg=overall, by_bin=by_bin))

    return AlphaSweepResult(eval_k=eval_k, rerank_depth=rerank_depth, results=results)
