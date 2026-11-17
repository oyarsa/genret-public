# pyright: basic
"""Recall@k metrics and evaluation harness for generative retrieval.

Evaluates retrieval quality by measuring how often the ground-truth relevant
papers appear within the top-k results produced by constrained beam search.
Reports metrics overall and broken down by relation type (reference,
co-citation, bibliographic coupling).
"""

import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.table import Table

from genret.paper_parsing import (
    FacetEvidence,
    FacetRecordType,
    Facets,
    PaperGraph,
    RelationType,
    TrainingPair,
    evidence_from_type,
    facets_from_type,
)
from genret.training.data import format_facets_prompt
from genret.types import CorpusId
from genret.utils import LineWriter, iter_jsonl, resolve_jsonl, track_progress, unique

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from genret.training.decoding import ConstrainedDecoder, DecodedResult

logger = logging.getLogger(__name__)

DEFAULT_KS = (1, 5, 10, 20, 50, 100)
"""Default values of k for Recall@k evaluation."""


def compute_recall_at_k(
    retrieved: list[CorpusId], ground_truth: set[CorpusId], k: int
) -> float:
    """Compute Recall@k for a single query.

    Args:
        retrieved: Ordered list of retrieved paper IDs (best first).
        ground_truth: Set of relevant paper IDs.
        k: Number of top results to consider.

    Returns:
        Fraction of ground-truth papers found in the top-k results.
        Returns 0.0 if ground_truth is empty.
    """
    if not ground_truth:
        return 0.0

    top_k = set(retrieved[:k])
    return len(top_k & ground_truth) / len(ground_truth)


@dataclass(frozen=True, kw_only=True)
class RecallMetrics:
    """Recall@k scores for a set of queries.

    Attributes:
        recall_at_k: Mapping from k to average Recall@k across all queries.
        num_queries: Number of queries evaluated.
    """

    recall_at_k: dict[int, float]
    num_queries: int


@dataclass(frozen=True, kw_only=True)
class EvaluationResults:
    """Complete evaluation results with overall and per-relation metrics.

    Attributes:
        overall: Aggregate recall metrics across all relation types.
        per_relation: Recall metrics broken down by relation type.
        mean_retrieved_per_query: Mean number of unique papers retrieved across
            all queries.  Low values (< 5) indicate beam collapse.
    """

    overall: RecallMetrics
    per_relation: dict[str, RecallMetrics]
    mean_retrieved_per_query: float


def _build_ground_truth(
    pairs: list[TrainingPair],
) -> dict[CorpusId, dict[str, set[CorpusId]]]:
    """Build ground-truth mapping from training pairs.

    Args:
        pairs: List of training pairs.

    Returns:
        Nested dict: ``query_paper -> {relation_type -> set of relevant papers}``.
    """
    gt: dict[CorpusId, dict[str, set[CorpusId]]] = defaultdict(lambda: defaultdict(set))
    for pair in pairs:
        gt[pair.paper_a][pair.relation.value].add(pair.paper_b)

    return gt


def _build_title_lookup(processed_dir: Path) -> dict[CorpusId, str]:
    """Build a corpus-ID-to-title mapping from paper_graphs.jsonl.

    Returns an empty dict if the file does not exist (titles are best-effort).

    Args:
        processed_dir: Path to the ``processed/`` directory.

    Returns:
        Dict mapping corpus ID to paper title.
    """
    graphs_path = resolve_jsonl(processed_dir / "paper_graphs.jsonl.gz")
    if not graphs_path.exists():
        logger.warning(
            "paper_graphs.jsonl not found at %s — titles will be omitted", graphs_path
        )
        return {}
    titles: dict[CorpusId, str] = {}
    for raw in iter_jsonl(graphs_path):
        paper = PaperGraph.model_validate(raw)
        titles[paper.corpusid] = paper.title
    return titles


def _build_query_record(
    corpus_id: CorpusId,
    facets: Facets,
    ground_truth_by_rel: dict[str, set[CorpusId]],
    results: list[DecodedResult],
    per_query_recall: dict[int, float],
    titles: dict[CorpusId, str],
    tokeniser: PreTrainedTokenizerBase | None,
) -> dict[str, Any]:
    """Assemble a per-query record dict for JSONL output.

    Args:
        corpus_id: Query paper corpus ID.
        facets: Extracted facets for the query paper.
        ground_truth_by_rel: Ground truth paper IDs grouped by relation type.
        results: Decoded beam search results (ordered by score).
        per_query_recall: Recall@k scores for this query.
        titles: Corpus-ID-to-title mapping.
        tokeniser: Tokeniser for decoding generated tokens (optional).

    Returns:
        Dict ready for JSON serialisation.
    """

    def _paper_ref(cid: CorpusId) -> dict[str, Any]:
        """Build a compact paper reference with corpus ID and title."""
        ref: dict[str, Any] = {"corpusid": cid}
        if cid in titles:
            ref["title"] = titles[cid]
        return ref

    # Query info
    query_info: dict[str, Any] = {
        "corpusid": corpus_id,
        "title": titles.get(corpus_id, ""),
        "facets": {
            "problems": list(facets.problems),
            "methods": list(facets.methods),
            "results": list(facets.results),
            "contributions": list(facets.contributions),
        },
    }

    # Ground truth grouped by relation
    gt_section: dict[str, list[dict[str, Any]]] = {}
    for rel_type, rel_ids in sorted(ground_truth_by_rel.items()):
        gt_section[rel_type] = [_paper_ref(pid) for pid in sorted(rel_ids)]

    # Predictions
    predictions: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        pred: dict[str, Any] = {
            "rank": rank,
            "score": result.score,
            "paper_ids": [_paper_ref(pid) for pid in result.paper_ids],
        }
        if tokeniser is not None:
            pred["docid_text"] = tokeniser.decode(
                result.generated_tokens, skip_special_tokens=True
            )
        predictions.append(pred)

    # Per-query recall
    recall_section = {str(k): v for k, v in sorted(per_query_recall.items())}

    return {
        "query": query_info,
        "ground_truth": gt_section,
        "predictions": predictions,
        "recall": recall_section,
    }


def evaluate_retrieval(
    decoder: ConstrainedDecoder,
    processed_dir: Path,
    *,
    split_ids: set[CorpusId] | None = None,
    ks: Sequence[int] = DEFAULT_KS,
    facets_type: FacetRecordType,
    include_evidence: bool = True,
    output_path: Path | None = None,
    tokeniser: PreTrainedTokenizerBase | None = None,
) -> EvaluationResults:
    """Run full evaluation: decode each query and compute Recall@k.

    Args:
        decoder: Constrained beam search decoder.
        processed_dir: Path to the ``processed/`` directory with facets and training
            pairs.
        split_ids: If provided, restrict to queries in this set.
        ks: Values of k for Recall@k computation.
        facets_type: On-disk schema for facet records.
        include_evidence: Whether to include evidence in prompts.
        output_path: If set, write per-query results to this JSONL path.
        tokeniser: Tokeniser for decoding generated tokens in per-query output.

    Returns:
        Evaluation results with overall and per-relation metrics.
    """
    # Load facets and pairs
    facets_path = resolve_jsonl(processed_dir / "facets.jsonl.gz")
    pairs_path = resolve_jsonl(processed_dir / "training_pairs.jsonl.gz")

    facets_from_corpus_id: dict[CorpusId, Facets] = {}
    evidence_from_corpus_id: dict[CorpusId, FacetEvidence] = {}
    for raw in iter_jsonl(facets_path):
        f = facets_from_type(raw, facets_type)
        if split_ids is None or f.corpusid in split_ids:
            facets_from_corpus_id[f.corpusid] = f
            if include_evidence and (ev := evidence_from_type(raw, facets_type)):
                evidence_from_corpus_id[f.corpusid] = ev

    training_pairs: list[TrainingPair] = []
    for raw in iter_jsonl(pairs_path):
        p = TrainingPair.model_validate(raw)
        if split_ids is None or (p.paper_a in split_ids and p.paper_b in split_ids):
            training_pairs.append(p)

    ground_truth = _build_ground_truth(training_pairs)

    # Only evaluate queries that have both facets and ground truth
    query_ids = sorted(set(facets_from_corpus_id) & set(ground_truth))
    logger.info("Evaluating %d queries", len(query_ids))

    # Build title lookup for per-query output
    titles: dict[CorpusId, str] = {}
    if output_path is not None:
        titles = _build_title_lookup(processed_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing per-query results to %s", output_path)

    # Per-query results
    overall_recalls: dict[int, list[float]] = {k: [] for k in ks}
    relation_recalls: dict[str, dict[int, list[float]]] = {
        rt.value: {k: [] for k in ks} for rt in RelationType
    }
    retrieved_counts: list[int] = []

    with ExitStack() as stack:
        writer = (
            stack.enter_context(LineWriter.open(output_path, "w"))
            if output_path is not None
            else None
        )

        for corpus_id in track_progress(query_ids, desc="Evaluating..."):
            messages = format_facets_prompt(
                facets_from_corpus_id[corpus_id],
                evidence=evidence_from_corpus_id.get(corpus_id),
            )
            results = decoder.decode_messages(messages)
            # Collect all retrieved paper IDs (ordered by beam score)
            retrieved = unique(
                pid for r in results for pid in r.paper_ids if pid != corpus_id
            )
            retrieved_counts.append(len(retrieved))

            # Compute overall recall (union of all relation types)
            all_relevant = {
                pid
                for rel_papers in ground_truth[corpus_id].values()
                for pid in rel_papers
            }

            per_query_recall: dict[int, float] = {}
            for k in ks:
                recall = compute_recall_at_k(retrieved, all_relevant, k)
                overall_recalls[k].append(recall)
                per_query_recall[k] = recall

            # Compute per-relation recall
            for rel_type, rel_papers in ground_truth[corpus_id].items():
                for k in ks:
                    relation_recalls[rel_type][k].append(
                        compute_recall_at_k(retrieved, rel_papers, k)
                    )

            # Write per-query record
            if writer is not None:
                record = _build_query_record(
                    corpus_id=corpus_id,
                    facets=facets_from_corpus_id[corpus_id],
                    ground_truth_by_rel=ground_truth[corpus_id],
                    results=results,
                    per_query_recall=per_query_recall,
                    titles=titles,
                    tokeniser=tokeniser,
                )
                writer.write_line(json.dumps(record, ensure_ascii=False), flush=True)

    # Aggregate
    mean_retrieved = _safe_mean([float(n) for n in retrieved_counts])
    logger.info(
        "Mean unique papers retrieved per query: %.1f (beam collapse if < 5)",
        mean_retrieved,
    )

    overall = RecallMetrics(
        recall_at_k={k: _safe_mean(scores) for k, scores in overall_recalls.items()},
        num_queries=len(query_ids),
    )

    per_relation: dict[str, RecallMetrics] = {}
    for rel_type, rel_scores in relation_recalls.items():
        num_q = len(next(iter(rel_scores.values()))) if rel_scores else 0
        if num_q > 0:
            per_relation[rel_type] = RecallMetrics(
                recall_at_k={k: _safe_mean(scores) for k, scores in rel_scores.items()},
                num_queries=num_q,
            )

    return EvaluationResults(
        overall=overall,
        per_relation=per_relation,
        mean_retrieved_per_query=mean_retrieved,
    )


def _safe_mean(values: list[float]) -> float:
    """Compute mean, returning 0.0 for empty lists.

    Args:
        values: List of floats.

    Returns:
        Arithmetic mean, or 0.0 if empty.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def format_results_table(
    results: EvaluationResults, ks: tuple[int, ...] = DEFAULT_KS
) -> Table:
    """Format evaluation results as a Rich table.

    Args:
        results: Evaluation results to format.
        ks: Values of k to include in the table.

    Returns:
        Rich Table with overall and per-relation Recall@k scores and a
        retrieved-per-query diagnostic.
    """
    table = Table(
        title=f"Recall@k Results  (mean retrieved/query: {results.mean_retrieved_per_query:.1f})"
    )
    table.add_column("Relation", style="bold")
    table.add_column("Queries", justify="right")
    for k in ks:
        table.add_column(f"R@{k}", justify="right")

    def _score_cells(metrics: RecallMetrics) -> list[str]:
        return [f"{metrics.recall_at_k.get(k, 0.0):.3f}" for k in ks]

    table.add_row(
        "Overall",
        str(results.overall.num_queries),
        *_score_cells(results.overall),
        style="bold",
    )

    for rel_type in sorted(results.per_relation):
        metrics = results.per_relation[rel_type]
        table.add_row(rel_type, str(metrics.num_queries), *_score_cells(metrics))

    return table
