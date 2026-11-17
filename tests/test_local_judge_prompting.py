"""Tests for local-judge prompt-compression variants."""

from genret.paper_parsing import Facets
from genret.processing.local_judge_prompting import (
    PromptVariant,
    format_facets_for_local_judge,
)
from genret.types import CorpusId


def _example_facets() -> Facets:
    """Build a small facets fixture for prompt-formatting tests."""
    return Facets(
        corpusid=CorpusId(123),
        problems=(
            "Problem one is described in detail.",
            "Problem two is also important.",
            "Problem three should be dropped in top2 variants.",
        ),
        methods=(
            "Method one.",
            "Method two.",
        ),
        results=(
            "Result one is informative.",
            "Result two is informative.",
        ),
        contributions=(
            "Contribution one.",
            "Contribution two.",
        ),
    )


def test_baseline_compact_omits_corpus_id() -> None:
    """Always omit `Corpus ID` in local-judge prompts."""
    formatted = format_facets_for_local_judge(
        _example_facets(),
        variant=PromptVariant.BASELINE_COMPACT,
    )

    assert "Corpus ID" not in formatted
    assert "Problem three should be dropped" in formatted


def test_top2_no_results_drops_extra_items_and_results() -> None:
    """Keep only two bullets per section and remove `Results`."""
    formatted = format_facets_for_local_judge(
        _example_facets(),
        variant=PromptVariant.TOP2_NO_RESULTS,
    )

    assert "**Results:**" not in formatted
    assert "Problem three should be dropped" not in formatted
    assert formatted.count("- ") == 6


def test_top2_truncated_shortens_long_items() -> None:
    """Truncate long bullets in the dedicated compact variant."""
    facets = Facets(
        corpusid=CorpusId(1),
        problems=("x" * 220,),
        methods=(),
        results=(),
        contributions=(),
    )

    formatted = format_facets_for_local_judge(
        facets,
        variant=PromptVariant.TOP2_TRUNCATED,
    )

    assert "…" in formatted
    assert len(formatted) < 220
