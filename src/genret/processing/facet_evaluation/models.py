"""Pydantic data models for facet evaluation records and summaries."""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel


class EvidenceSnippet(BaseModel, frozen=True):
    """A source paragraph that supports a facet item."""

    section_heading: str
    text: str
    similarity: float


class ItemEvaluation(BaseModel, frozen=True):
    """Evaluation metrics for a single facet item."""

    text: str
    char_count: int
    sentence_count: int
    grounding_score: float
    top_evidence: Sequence[EvidenceSnippet]
    model_evidence: Sequence[str]
    predicted_facet_type: str | None
    genericness_score: float
    is_near_duplicate: bool
    evidence_verbatim_count: int = 0
    """Count of evidence strings that matched verbatim in the paper."""
    evidence_faithful_count: int = 0
    """Count of evidence strings with coverage >= EVIDENCE_FIDELITY_THRESHOLD."""
    evidence_fidelity: float | None = None
    """Mean coverage ratio across evidence strings, None when no evidence."""


class PaperScores(BaseModel, frozen=True):
    """Component and overall quality scores for a paper's facets."""

    structure: float
    grounding: float
    redundancy: float
    specificity: float
    overall: float

    def as_dict(self) -> dict[str, float]:
        """Return all scores as a {name: value} mapping."""
        return {
            "structure": self.structure,
            "grounding": self.grounding,
            "redundancy": self.redundancy,
            "specificity": self.specificity,
            "overall": self.overall,
        }


class PaperEvaluation(BaseModel, frozen=True):
    """Complete evaluation record for one paper's facets."""

    corpusid: int
    title: str | None
    facet_counts: dict[str, int]
    scores: PaperScores
    flags: Sequence[str]
    item_metrics: dict[str, Sequence[ItemEvaluation]]
    evaluated_at: str


class ScoreDistribution(BaseModel, frozen=True):
    """Distribution statistics for a score across papers."""

    mean: float
    median: float
    std: float
    p10: float
    p90: float
    min_val: float
    max_val: float


class EvaluationSummary(BaseModel, frozen=True):
    """Aggregate evaluation metrics across all papers."""

    papers_in_file: int
    papers_evaluated: int
    papers_skipped: int
    skip_reasons: dict[str, int]
    score_distributions: dict[str, ScoreDistribution]
    facet_coverage: dict[str, float]
    item_count_means: dict[str, float]
    top_flags: dict[str, int]
    evidence_fidelity: ScoreDistribution | None = None
    """Distribution of per-paper mean evidence fidelity (only when evidence is present)."""
    evidence_verbatim_fraction: float | None = None
    """Fraction of all evidence strings that were verbatim matches."""
    evidence_faithful_fraction: float | None = None
    """Fraction of all evidence strings with coverage >= EVIDENCE_FIDELITY_THRESHOLD."""
    config: dict[str, Any]
    runtime_seconds: float
    evaluated_at: str
