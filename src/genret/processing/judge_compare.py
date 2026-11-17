"""Compare LLM judge outputs across different judging approaches.

Reads judgement JSONL files from multiple approaches and computes agreement
metrics: completion rate, Cohen's kappa, and per-field agreement statistics.
Also supports comparing continuous scores from the local judge against
boolean judgements.
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from genret.processing.llm_judge import (
    JudgedPair,
    ScoredPair,
    ScoreRange,
    normalise_judged_pair_dict,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ComparisonResult:
    """Result of comparing two sets of judgements.

    Attributes:
        approach_a: Name of the first approach.
        approach_b: Name of the second approach.
        total_pairs_a: Total pairs judged by approach A.
        total_pairs_b: Total pairs judged by approach B.
        common_pairs: Number of pairs judged by both.
        related_agreement: Fraction of common pairs where `related` matches.
        relationship_type_agreement: Fraction of common pairs where
            `relationship_type` matches.
        cohens_kappa: Cohen's kappa for the `related` field.
        mean_confidence_a: Mean confidence for approach A (common pairs).
        mean_confidence_b: Mean confidence for approach B (common pairs).
    """

    approach_a: str
    approach_b: str
    total_pairs_a: int
    total_pairs_b: int
    common_pairs: int
    related_agreement: float
    relationship_type_agreement: float
    cohens_kappa: float
    mean_confidence_a: float
    mean_confidence_b: float


def _load_judgements(path: Path) -> dict[str, JudgedPair]:
    """Load judgement JSONL file into a dict keyed by `"paper_a|paper_b"`.

    Args:
        path: Path to JSONL file.

    Returns:
        Dict mapping pair key to `JudgedPair`.
    """
    judgements: dict[str, JudgedPair] = {}
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue

            data = json.loads(stripped)
            pair = JudgedPair.model_validate(normalise_judged_pair_dict(data))
            key = f"{pair.paper_a}|{pair.paper_b}"
            judgements[key] = pair

    return judgements


def _cohens_kappa(labels_a: list[bool], labels_b: list[bool]) -> float:
    """Compute Cohen's kappa for binary classification agreement.

    Args:
        labels_a: Binary labels from approach A.
        labels_b: Binary labels from approach B.

    Returns:
        Cohen's kappa value. Returns 0.0 if undefined (all same label).
    """
    n = len(labels_a)
    if n == 0:
        return 0.0

    agree = sum(a == b for a, b in zip(labels_a, labels_b, strict=True))
    p_o = agree / n

    # Expected agreement by chance
    a_pos = sum(labels_a) / n
    b_pos = sum(labels_b) / n
    p_e = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)

    if p_e == 1.0:
        return 0.0

    return (p_o - p_e) / (1 - p_e)


def compare_approaches(
    path_a: Path,
    path_b: Path,
    name_a: str,
    name_b: str,
) -> ComparisonResult:
    """Compare two sets of judgements and compute agreement metrics.

    Args:
        path_a: Path to first approach's JSONL output.
        path_b: Path to second approach's JSONL output.
        name_a: Human-readable name for approach A.
        name_b: Human-readable name for approach B.

    Returns:
        ComparisonResult with agreement statistics.
    """
    judgements_a = _load_judgements(path_a)
    judgements_b = _load_judgements(path_b)
    common_keys = set(judgements_a) & set(judgements_b)

    if not common_keys:
        logger.warning("No common pairs between %s and %s", name_a, name_b)
        return ComparisonResult(
            approach_a=name_a,
            approach_b=name_b,
            total_pairs_a=len(judgements_a),
            total_pairs_b=len(judgements_b),
            common_pairs=0,
            related_agreement=0.0,
            relationship_type_agreement=0.0,
            cohens_kappa=0.0,
            mean_confidence_a=0.0,
            mean_confidence_b=0.0,
        )

    related_a: list[bool] = []
    related_b: list[bool] = []
    type_matches = 0
    conf_a_sum = 0.0
    conf_b_sum = 0.0

    for key in sorted(common_keys):
        pa = judgements_a[key]
        pb = judgements_b[key]
        related_a.append(pa.related)
        related_b.append(pb.related)

        if pa.relationship_type == pb.relationship_type:
            type_matches += 1

        conf_a_sum += pa.confidence
        conf_b_sum += pb.confidence

    n = len(common_keys)
    related_matches = sum(a == b for a, b in zip(related_a, related_b, strict=True))

    return ComparisonResult(
        approach_a=name_a,
        approach_b=name_b,
        total_pairs_a=len(judgements_a),
        total_pairs_b=len(judgements_b),
        common_pairs=n,
        related_agreement=related_matches / n,
        relationship_type_agreement=type_matches / n,
        cohens_kappa=_cohens_kappa(related_a, related_b),
        mean_confidence_a=conf_a_sum / n,
        mean_confidence_b=conf_b_sum / n,
    )


def compare_all(
    judgement_files: dict[str, Path],
) -> list[ComparisonResult]:
    """Compare all pairs of approaches.

    Args:
        judgement_files: Dict mapping approach name to JSONL path.

    Returns:
        List of pairwise ComparisonResults.
    """
    names = sorted(judgement_files)
    results: list[ComparisonResult] = []

    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            result = compare_approaches(
                judgement_files[name_a],
                judgement_files[name_b],
                name_a,
                name_b,
            )
            results.append(result)
            logger.info(
                "%s vs %s: %d common pairs, related agreement=%.1f%%, "
                "kappa=%.3f, type agreement=%.1f%%",
                name_a,
                name_b,
                result.common_pairs,
                result.related_agreement * 100,
                result.cohens_kappa,
                result.relationship_type_agreement * 100,
            )

    return results


def _load_scores(path: Path) -> dict[str, ScoredPair]:
    """Load scored pair JSONL file into a dict keyed by `"paper_a|paper_b"`.

    Args:
        path: Path to JSONL file with `ScoredPair` records.

    Returns:
        Dict mapping pair key to `ScoredPair`.
    """
    scores: dict[str, ScoredPair] = {}
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            pair = ScoredPair.model_validate_json(stripped)
            key = f"{pair.paper_a}|{pair.paper_b}"
            scores[key] = pair
    return scores


@dataclass(frozen=True, kw_only=True)
class ThresholdResult:
    """Agreement metrics at a given score threshold.

    Attributes:
        threshold: The score threshold used.
        precision: Fraction of predicted-related that are truly related.
        recall: Fraction of truly related that are predicted-related.
        f1: Harmonic mean of precision and recall.
        agreement: Fraction of pairs where both agree on related/unrelated.
        cohens_kappa: Cohen's kappa at this threshold.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    agreement: float
    cohens_kappa: float


@dataclass(frozen=True, kw_only=True)
class ScoredComparisonResult:
    """Result of comparing scored output against boolean judgements.

    Attributes:
        scored_name: Name of the scored approach.
        boolean_name: Name of the boolean approach.
        total_scored: Total pairs in scored output.
        total_boolean: Total pairs in boolean output.
        common_pairs: Pairs in both.
        score_range: Score range used by the scored output.
        mean_score_related: Mean score for pairs the boolean judge marked related.
        mean_score_unrelated: Mean score for pairs the boolean judge marked unrelated.
        spearman_rho: Spearman rank correlation between score and confidence.
        roc_auc: Area under ROC curve (boolean `related` as label, score as predictor).
        threshold_results: Agreement metrics at thresholds appropriate for the
            scored range.
    """

    scored_name: str
    boolean_name: str
    total_scored: int
    total_boolean: int
    common_pairs: int
    score_range: ScoreRange
    mean_score_related: float
    mean_score_unrelated: float
    spearman_rho: float
    roc_auc: float
    threshold_results: list[ThresholdResult]


def _thresholds_for_score_range(score_range: ScoreRange) -> tuple[float, ...]:
    """Return threshold sweep values for a score range.

    Args:
        score_range: Score range used by the scored output.

    Returns:
        Threshold values to evaluate.
    """
    if score_range is ScoreRange.ZERO_TO_ONE:
        return tuple(i / 10 for i in range(1, 10))
    if score_range is ScoreRange.ONE_TO_FIVE:
        return tuple(float(i) for i in range(2, 6))
    return tuple(float(i) for i in range(2, 11))


def _spearman_rank_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman rank correlation between two sequences.

    Args:
        xs: First sequence of values.
        ys: Second sequence of values.

    Returns:
        Spearman's rho. Returns 0.0 if undefined.
    """
    n = len(xs)
    if n < 2:
        return 0.0

    def _rank(values: list[float]) -> list[float]:
        """Assign average ranks to values."""
        indexed = sorted(enumerate(values), key=lambda iv: iv[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry, strict=True))
    den_x = math.sqrt(sum((a - mean_rx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_ry) ** 2 for b in ry))

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)


def _roc_auc(labels: list[bool], scores: list[float]) -> float:
    """Compute area under the ROC curve.

    Uses the trapezoidal rule over sorted thresholds.

    Args:
        labels: Binary ground-truth labels.
        scores: Continuous predictor scores.

    Returns:
        AUC value. Returns 0.0 if undefined (all same label).
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Sort by descending score
    paired = sorted(zip(scores, labels, strict=True), key=lambda sl: -sl[0])

    tp = 0
    fp = 0
    prev_tp = 0
    prev_fp = 0
    auc = 0.0
    prev_score = float("inf")

    for score, label in paired:
        if score != prev_score:
            # Trapezoid between previous and current point
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_tp = tp
            prev_fp = fp
            prev_score = score

        if label:
            tp += 1
        else:
            fp += 1

    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0

    return auc / (n_pos * n_neg)


def compare_scored_vs_boolean(
    scored_path: Path,
    boolean_path: Path,
    scored_name: str,
    boolean_name: str,
) -> ScoredComparisonResult:
    """Compare continuous scores against boolean judgements.

    Args:
        scored_path: Path to scored pairs JSONL.
        boolean_path: Path to boolean judgements JSONL.
        scored_name: Human-readable name for scored approach.
        boolean_name: Human-readable name for boolean approach.

    Returns:
        Comparison result with threshold sweep, correlation, and AUC metrics.
    """
    scored = _load_scores(scored_path)
    boolean = _load_judgements(boolean_path)
    common_keys = set(scored) & set(boolean)

    if not common_keys:
        logger.warning("No common pairs between %s and %s", scored_name, boolean_name)
        return ScoredComparisonResult(
            scored_name=scored_name,
            boolean_name=boolean_name,
            total_scored=len(scored),
            total_boolean=len(boolean),
            common_pairs=0,
            score_range=ScoreRange.ONE_TO_TEN,
            mean_score_related=0.0,
            mean_score_unrelated=0.0,
            spearman_rho=0.0,
            roc_auc=0.0,
            threshold_results=[],
        )

    score_ranges = {pair.score_range for pair in scored.values()}
    if len(score_ranges) != 1:
        raise ValueError(
            f"Expected one score range in {scored_path}, found {sorted(score_ranges)}"
        )
    score_range = next(iter(score_ranges))

    score_vals: list[float] = []
    labels: list[bool] = []
    conf_vals: list[float] = []

    related_scores: list[float] = []
    unrelated_scores: list[float] = []

    for key in sorted(common_keys):
        sp = scored[key]
        bp = boolean[key]
        score_vals.append(sp.score)
        labels.append(bp.related)
        conf_vals.append(bp.confidence)

        if bp.related:
            related_scores.append(sp.score)
        else:
            unrelated_scores.append(sp.score)

    # Threshold sweep
    threshold_results: list[ThresholdResult] = []
    for threshold in _thresholds_for_score_range(score_range):
        predicted = [s >= threshold for s in score_vals]
        tp = sum(p and lb for p, lb in zip(predicted, labels, strict=True))
        fp = sum(p and not lb for p, lb in zip(predicted, labels, strict=True))
        fn = sum(not p and lb for p, lb in zip(predicted, labels, strict=True))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        agree = sum(p == lb for p, lb in zip(predicted, labels, strict=True))
        agreement = agree / len(labels)

        kappa = _cohens_kappa(predicted, labels)

        threshold_results.append(
            ThresholdResult(
                threshold=threshold,
                precision=precision,
                recall=recall,
                f1=f1,
                agreement=agreement,
                cohens_kappa=kappa,
            )
        )

    return ScoredComparisonResult(
        scored_name=scored_name,
        boolean_name=boolean_name,
        total_scored=len(scored),
        total_boolean=len(boolean),
        common_pairs=len(common_keys),
        score_range=score_range,
        mean_score_related=(
            sum(related_scores) / len(related_scores) if related_scores else 0.0
        ),
        mean_score_unrelated=(
            sum(unrelated_scores) / len(unrelated_scores) if unrelated_scores else 0.0
        ),
        spearman_rho=_spearman_rank_correlation(score_vals, conf_vals),
        roc_auc=_roc_auc(labels, score_vals),
        threshold_results=threshold_results,
    )
