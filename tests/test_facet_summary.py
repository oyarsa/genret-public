"""Unit tests for facet_evaluation summary aggregation."""

from conftest import approx
from genret.processing.facet_evaluation import (
    ItemEvaluation,
    PaperEvaluation,
    PaperScores,
    compute_summary,
    distribution,
)


class TestDistribution:
    """Tests for distribution helper."""

    def test_empty(self) -> None:
        """Should return zeros for empty input."""
        dist = distribution([])
        assert dist.mean == 0
        assert dist.min_val == 0

    def test_single_value(self) -> None:
        """Should handle single-element list with zero std."""
        dist = distribution([0.5])
        assert dist.mean == approx(0.5)
        assert dist.median == approx(0.5)
        assert dist.std == 0.0

    def test_multiple_values(self) -> None:
        """Should compute correct mean, min, max for multiple values."""
        dist = distribution([0.1, 0.5, 0.9])
        assert dist.mean == approx(0.5)
        assert dist.min_val == approx(0.1)
        assert dist.max_val == approx(0.9)


class TestComputeSummary:
    """Tests for compute_summary aggregation."""

    def test_empty_evaluations(self) -> None:
        """Should handle empty evaluation list gracefully."""
        summary = compute_summary([], 10, {"parse_error": 2}, {}, 1.0)
        assert summary.papers_evaluated == 0
        assert summary.papers_in_file == 10
        assert summary.papers_skipped == 2

    def test_with_evaluations(self) -> None:
        """Should compute correct aggregate statistics."""
        evals = [
            PaperEvaluation(
                corpusid=1,
                title="P1",
                facet_counts={
                    "problems": 2,
                    "methods": 1,
                    "results": 1,
                    "contributions": 1,
                },
                scores=PaperScores(
                    structure=0.9,
                    grounding=0.8,
                    redundancy=0.7,
                    specificity=0.6,
                    overall=0.75,
                ),
                flags=["some_flag"],
                item_metrics={},
                evaluated_at="2026-01-01",
            ),
            PaperEvaluation(
                corpusid=2,
                title="P2",
                facet_counts={
                    "problems": 1,
                    "methods": 0,
                    "results": 1,
                    "contributions": 1,
                },
                scores=PaperScores(
                    structure=0.8,
                    grounding=0.7,
                    redundancy=0.9,
                    specificity=0.5,
                    overall=0.70,
                ),
                flags=["some_flag", "other_flag"],
                item_metrics={},
                evaluated_at="2026-01-01",
            ),
        ]

        summary = compute_summary(evals, 5, {"bad": 1}, {"seed": 42}, 2.5)

        assert summary.papers_evaluated == 2
        assert summary.papers_in_file == 5
        assert summary.papers_skipped == 1
        assert summary.runtime_seconds == 2.5

        # Score distributions
        assert "structure" in summary.score_distributions
        assert "overall" in summary.score_distributions
        struct_dist = summary.score_distributions["structure"]
        assert struct_dist.mean == approx(0.85)

        # Facet coverage
        assert summary.facet_coverage["problems"] == 1.0
        assert summary.facet_coverage["methods"] == 0.5

        # Flags
        assert summary.top_flags["some_flag"] == 2
        assert summary.top_flags["other_flag"] == 1

        # Evidence fidelity (no evidence in these evaluations)
        assert summary.evidence_fidelity is None
        assert summary.evidence_verbatim_fraction is None


class TestComputeSummaryWithEvidence:
    """Tests for compute_summary evidence fidelity aggregation."""

    def _make_item(
        self,
        *,
        evidence_fidelity: float | None = None,
        evidence_verbatim_count: int = 0,
        evidence_faithful_count: int = 0,
        model_evidence: list[str] | None = None,
    ) -> ItemEvaluation:
        """Create a minimal ItemEvaluation with evidence fields."""
        return ItemEvaluation(
            text="test item",
            char_count=9,
            sentence_count=1,
            grounding_score=0.7,
            top_evidence=[],
            model_evidence=model_evidence or [],
            predicted_facet_type=None,
            genericness_score=0.1,
            is_near_duplicate=False,
            evidence_verbatim_count=evidence_verbatim_count,
            evidence_faithful_count=evidence_faithful_count,
            evidence_fidelity=evidence_fidelity,
        )

    def test_summary_with_evidence(self) -> None:
        """Summary populates evidence fields when items have evidence."""
        evals = [
            PaperEvaluation(
                corpusid=1,
                title="P1",
                facet_counts={
                    "problems": 1,
                    "methods": 0,
                    "results": 0,
                    "contributions": 0,
                },
                scores=PaperScores(
                    structure=0.9,
                    grounding=0.8,
                    redundancy=0.7,
                    specificity=0.6,
                    overall=0.75,
                ),
                flags=[],
                item_metrics={
                    "problems": [
                        self._make_item(
                            evidence_fidelity=0.9,
                            evidence_verbatim_count=2,
                            evidence_faithful_count=3,
                            model_evidence=["ev1", "ev2", "ev3"],
                        ),
                    ],
                },
                evaluated_at="2026-01-01",
            ),
            PaperEvaluation(
                corpusid=2,
                title="P2",
                facet_counts={
                    "problems": 1,
                    "methods": 0,
                    "results": 0,
                    "contributions": 0,
                },
                scores=PaperScores(
                    structure=0.8,
                    grounding=0.7,
                    redundancy=0.9,
                    specificity=0.5,
                    overall=0.70,
                ),
                flags=[],
                item_metrics={
                    "problems": [
                        self._make_item(
                            evidence_fidelity=0.7,
                            evidence_verbatim_count=1,
                            evidence_faithful_count=1,
                            model_evidence=["ev1", "ev2"],
                        ),
                    ],
                },
                evaluated_at="2026-01-01",
            ),
        ]

        summary = compute_summary(evals, 2, {}, {"seed": 42}, 1.0)

        assert summary.evidence_fidelity is not None
        # mean of [0.9, 0.7] = 0.8
        assert summary.evidence_fidelity.mean == approx(0.8)
        assert summary.evidence_verbatim_fraction is not None
        # 3 verbatim out of 5 total evidence strings
        assert summary.evidence_verbatim_fraction == approx(3 / 5)
        assert summary.evidence_faithful_fraction is not None
        # 4 faithful out of 5 total evidence strings
        assert summary.evidence_faithful_fraction == approx(4 / 5)

    def test_summary_without_evidence(self) -> None:
        """Summary leaves evidence fields as None when no items have evidence."""
        evals = [
            PaperEvaluation(
                corpusid=1,
                title="P1",
                facet_counts={
                    "problems": 1,
                    "methods": 0,
                    "results": 0,
                    "contributions": 0,
                },
                scores=PaperScores(
                    structure=0.9,
                    grounding=0.8,
                    redundancy=0.7,
                    specificity=0.6,
                    overall=0.75,
                ),
                flags=[],
                item_metrics={
                    "problems": [self._make_item()],
                },
                evaluated_at="2026-01-01",
            ),
        ]

        summary = compute_summary(evals, 1, {}, {"seed": 42}, 1.0)
        assert summary.evidence_fidelity is None
        assert summary.evidence_verbatim_fraction is None
        assert summary.evidence_faithful_fraction is None
