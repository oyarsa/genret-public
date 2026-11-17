"""Corpus-level aggregation of per-paper evaluations into a summary."""

import statistics
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from genret.utils import safediv

from .constants import FACET_NAMES
from .models import EvaluationSummary, PaperEvaluation, ScoreDistribution


def distribution(values: list[float]) -> ScoreDistribution:
    """Compute distribution statistics from a list of values."""
    if not values:
        return ScoreDistribution(
            mean=0, median=0, std=0, p10=0, p90=0, min_val=0, max_val=0
        )
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return ScoreDistribution(
        mean=statistics.mean(sorted_vals),
        median=statistics.median(sorted_vals),
        std=statistics.stdev(sorted_vals) if n > 1 else 0.0,
        p10=sorted_vals[max(0, int(n * 0.1))],
        p90=sorted_vals[min(n - 1, int(n * 0.9))],
        min_val=sorted_vals[0],
        max_val=sorted_vals[-1],
    )


def compute_summary(
    evaluations: list[PaperEvaluation],
    papers_in_file: int,
    skip_reasons: dict[str, int],
    config: dict[str, Any],
    runtime_seconds: float,
) -> EvaluationSummary:
    """Aggregate per-paper evaluations into a summary."""
    if not evaluations:
        return EvaluationSummary(
            papers_in_file=papers_in_file,
            papers_evaluated=0,
            papers_skipped=sum(skip_reasons.values()),
            skip_reasons=skip_reasons,
            score_distributions={},
            facet_coverage={},
            item_count_means={},
            top_flags={},
            config=config,
            runtime_seconds=runtime_seconds,
            evaluated_at=datetime.now(UTC).isoformat(),
        )

    # Score distributions
    score_names = ["structure", "grounding", "redundancy", "specificity", "overall"]
    score_distributions: dict[str, ScoreDistribution] = {}
    for sn in score_names:
        vals = [e.scores.as_dict()[sn] for e in evaluations]
        score_distributions[sn] = distribution(vals)

    # Facet coverage (fraction of papers with >=1 item per facet)
    facet_coverage: dict[str, float] = {}
    for name in FACET_NAMES:
        with_items = sum(1 for e in evaluations if e.facet_counts.get(name, 0) > 0)
        facet_coverage[name] = with_items / len(evaluations)

    # Item count means
    item_count_means: dict[str, float] = {}
    for name in FACET_NAMES:
        counts = [e.facet_counts.get(name, 0) for e in evaluations]
        item_count_means[name] = statistics.mean(counts)

    # Top flags
    flag_counter: Counter[str] = Counter()
    for e in evaluations:
        flag_counter.update(e.flags)
    top_flags = dict(flag_counter.most_common(20))

    # Evidence fidelity (only when at least one paper has evidence)
    per_paper_fidelity: list[float] = []
    total_verbatim = 0
    total_faithful = 0
    total_evidence = 0
    for e in evaluations:
        paper_fidelities: list[float] = []

        for items in e.item_metrics.values():
            for item in items:
                if item.evidence_fidelity is not None:
                    paper_fidelities.append(item.evidence_fidelity)
                    total_verbatim += item.evidence_verbatim_count
                    total_faithful += item.evidence_faithful_count
                    total_evidence += len(item.model_evidence)

        if paper_fidelities:
            per_paper_fidelity.append(statistics.mean(paper_fidelities))

    ev_fidelity_dist: ScoreDistribution | None = None
    ev_verbatim_frac: float | None = None
    ev_faithful_frac: float | None = None
    if per_paper_fidelity:
        ev_fidelity_dist = distribution(per_paper_fidelity)
        ev_verbatim_frac = safediv(total_verbatim, total_evidence)
        ev_faithful_frac = safediv(total_faithful, total_evidence)

    return EvaluationSummary(
        papers_in_file=papers_in_file,
        papers_evaluated=len(evaluations),
        papers_skipped=sum(skip_reasons.values()),
        skip_reasons=skip_reasons,
        score_distributions=score_distributions,
        facet_coverage=facet_coverage,
        item_count_means=item_count_means,
        top_flags=top_flags,
        evidence_fidelity=ev_fidelity_dist,
        evidence_verbatim_fraction=ev_verbatim_frac,
        evidence_faithful_fraction=ev_faithful_frac,
        config=config,
        runtime_seconds=runtime_seconds,
        evaluated_at=datetime.now(UTC).isoformat(),
    )
