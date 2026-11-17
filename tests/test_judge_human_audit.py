"""Tests for interactive human audit helpers."""

from genret.processing.judge_human_audit import (
    HumanAuditRecord,
    ModelPrediction,
    _all_models_agree_against_human,
    _build_audit_summaries,
)
from genret.types import CorpusId


def _prediction(related: bool) -> ModelPrediction:
    """Build a compact test prediction.

    Args:
        related: Binary relatedness label.

    Returns:
        ModelPrediction for test use.
    """
    return ModelPrediction(
        related=related,
        confidence=0.9,
        relationship_type="none",
        reasoning="test",
    )


def test_all_models_agree_against_human() -> None:
    """Should flag unanimous model disagreement against the human label."""
    record = HumanAuditRecord(
        paper_a=CorpusId(1),
        paper_b=CorpusId(2),
        source="retrieved",
        human_related=False,
        model_predictions={
            "gpt-5-mini": _prediction(True),
            "gpt-5.4-medium": _prediction(True),
            "gpt-5.4-none": _prediction(True),
        },
    )

    assert _all_models_agree_against_human(record) is True


def test_build_audit_summaries_counts_agreement() -> None:
    """Should compute per-model agreement and unanimous misses."""
    records = [
        HumanAuditRecord(
            paper_a=CorpusId(1),
            paper_b=CorpusId(2),
            source="retrieved",
            human_related=True,
            model_predictions={
                "gpt-5-mini": _prediction(True),
                "gpt-5.4-medium": _prediction(True),
                "gpt-5.4-none": _prediction(False),
            },
        ),
        HumanAuditRecord(
            paper_a=CorpusId(3),
            paper_b=CorpusId(4),
            source="random_negative",
            human_related=False,
            model_predictions={
                "gpt-5-mini": _prediction(True),
                "gpt-5.4-medium": _prediction(True),
                "gpt-5.4-none": _prediction(True),
            },
        ),
    ]

    summaries, unanimous_misses = _build_audit_summaries(records)
    by_name = {summary.model_name: summary for summary in summaries}

    assert by_name["gpt-5-mini"].agreement == 0.5
    assert by_name["gpt-5.4-medium"].agreement == 0.5
    assert by_name["gpt-5.4-none"].agreement == 0.0
    assert unanimous_misses == 1
