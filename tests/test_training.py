"""Tests for the generative retrieval training package.

Tests pure-Python logic: prompt formatting, example mixing, recall metrics,
and constrained decoding logic.  No GPU or torch dependency required.
"""

import random
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from genret.paper_parsing import FacetEvidence, Facets
from genret.training.data import (
    SYSTEM_PROMPT,
    TrainingExample,
    _mix_examples,
    format_facets_prompt,
)
from genret.training.evaluate import (
    EvaluationResults,
    RecallMetrics,
    _safe_mean,
    compute_recall_at_k,
    format_results_table,
)
from genret.types import CorpusId

_DEFAULT_CORPUSID = CorpusId(1)


def _make_facets(
    corpusid: CorpusId = _DEFAULT_CORPUSID,
    problems: Sequence[str] = ("problem A",),
    methods: Sequence[str] = ("method X", "method Y"),
    results: Sequence[str] = ("result 1",),
    contributions: Sequence[str] = (),
) -> Facets:
    """Create a Facets instance for testing."""
    return Facets(
        corpusid=corpusid,
        problems=problems,
        methods=methods,
        results=results,
        contributions=contributions,
    )


class TestFormatFacetsPrompt:
    """Tests for the format_facets_prompt function."""

    def test_returns_two_messages(self) -> None:
        """Should return system + user messages."""
        messages = format_facets_prompt(_make_facets())
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_content(self) -> None:
        """System prompt should be the defined constant."""
        messages = format_facets_prompt(_make_facets())
        assert messages[0]["content"] == SYSTEM_PROMPT

    def test_user_content_has_sections(self) -> None:
        """User message should contain facet sections with bullet lists."""
        messages = format_facets_prompt(_make_facets())
        content = messages[1]["content"]

        assert "Problems:" in content
        assert "- problem A" in content
        assert "Methods:" in content
        assert "- method X" in content
        assert "- method Y" in content
        assert "Results:" in content
        assert "- result 1" in content

    def test_empty_section_omitted(self) -> None:
        """Empty facet sections should be omitted from the user message."""
        facets = _make_facets(contributions=())
        messages = format_facets_prompt(facets)
        assert "Contributions:" not in messages[1]["content"]

    def test_all_empty_facets(self) -> None:
        """Should produce an empty user message when all facets are empty."""
        facets = _make_facets(problems=(), methods=(), results=(), contributions=())
        messages = format_facets_prompt(facets)
        assert messages[1]["content"] == ""

    def test_with_evidence(self) -> None:
        """Should include evidence quotes beneath each facet item."""
        facets = _make_facets(problems=("problem A",), methods=("method X",))
        evidence: FacetEvidence = {
            "problems": [["Quote about problem A."]],
            "methods": [["Quote 1 about method.", "Quote 2 about method."]],
        }
        messages = format_facets_prompt(facets, evidence=evidence)
        content = messages[1]["content"]
        assert "- problem A" in content
        assert '> "Quote about problem A."' in content
        assert '> "Quote 1 about method."' in content
        assert '> "Quote 2 about method."' in content

    def test_evidence_none_is_noop(self) -> None:
        """Passing evidence=None should behave like no evidence."""
        facets = _make_facets()
        with_none = format_facets_prompt(facets, evidence=None)
        without = format_facets_prompt(facets)
        assert with_none == without


def _rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


class TestMixExamples:
    """Tests for the _mix_examples function."""

    def _make_example(self, label: str) -> TrainingExample:
        """Create a dummy training example."""
        return TrainingExample(
            prompt_messages=[{"role": "user", "content": label}],
            target_docid=f"docid-{label}",
        )

    def test_empty_inputs(self) -> None:
        """Should return empty list for empty inputs."""
        assert _mix_examples([], [], 0.5, _rng()) == []

    def test_only_indexing(self) -> None:
        """With ratio 1.0, should return only indexing examples."""
        idx = [self._make_example(f"idx-{i}") for i in range(5)]
        ret = [self._make_example(f"ret-{i}") for i in range(3)]
        result = _mix_examples(idx, ret, 1.0, _rng())
        assert len(result) == 5

    def test_only_retrieval(self) -> None:
        """With ratio 0.0, should return only retrieval examples."""
        idx = [self._make_example(f"idx-{i}") for i in range(5)]
        ret = [self._make_example(f"ret-{i}") for i in range(3)]
        result = _mix_examples(idx, ret, 0.0, _rng())
        assert len(result) == 3

    def test_deterministic_shuffle(self) -> None:
        """Same seed should produce the same ordering."""
        idx = [self._make_example(f"idx-{i}") for i in range(10)]
        ret = [self._make_example(f"ret-{i}") for i in range(10)]
        result1 = _mix_examples(idx, ret, 0.5, _rng())
        result2 = _mix_examples(idx, ret, 0.5, _rng())
        assert result1 == result2

    def test_different_seeds_differ(self) -> None:
        """Different seeds should (usually) produce different orderings."""
        idx = [self._make_example(f"idx-{i}") for i in range(20)]
        ret = [self._make_example(f"ret-{i}") for i in range(20)]
        result1 = _mix_examples(idx, ret, 0.5, _rng())
        result2 = _mix_examples(idx, ret, 0.5, _rng())
        # The contents should be the same, but order should differ
        assert {e.target_docid for e in result1} == {e.target_docid for e in result2}

    def test_approximate_ratio(self) -> None:
        """Mixed result should approximately match the target ratio."""
        idx = [self._make_example(f"idx-{i}") for i in range(100)]
        ret = [self._make_example(f"ret-{i}") for i in range(100)]
        result = _mix_examples(idx, ret, 0.5, _rng())

        idx_count = sum(1 for e in result if e.target_docid.startswith("docid-idx"))
        ratio = idx_count / len(result)
        assert 0.4 <= ratio <= 0.6


class TestComputeRecallAtK:
    """Tests for the compute_recall_at_k function."""

    def test_perfect_recall(self) -> None:
        """Should return 1.0 when all relevant papers are in top-k."""
        assert (
            compute_recall_at_k(
                [CorpusId(1), CorpusId(2), CorpusId(3), CorpusId(4), CorpusId(5)],
                {CorpusId(1), CorpusId(2), CorpusId(3)},
                5,
            )
            == 1.0
        )

    def test_zero_recall(self) -> None:
        """Should return 0.0 when no relevant papers are in top-k."""
        assert (
            compute_recall_at_k(
                [CorpusId(10), CorpusId(20), CorpusId(30)],
                {CorpusId(1), CorpusId(2), CorpusId(3)},
                3,
            )
            == 0.0
        )

    def test_partial_recall(self) -> None:
        """Should return the correct fraction for partial overlap."""
        assert (
            compute_recall_at_k(
                [CorpusId(1), CorpusId(10), CorpusId(2), CorpusId(20), CorpusId(3)],
                {CorpusId(1), CorpusId(2), CorpusId(3)},
                3,
            )
            == 2.0 / 3.0
        )

    def test_empty_ground_truth(self) -> None:
        """Should return 0.0 for empty ground truth."""
        assert (
            compute_recall_at_k([CorpusId(1), CorpusId(2), CorpusId(3)], set(), 3)
            == 0.0
        )

    def test_empty_retrieved(self) -> None:
        """Should return 0.0 when nothing is retrieved."""
        assert (
            compute_recall_at_k([], {CorpusId(1), CorpusId(2), CorpusId(3)}, 3) == 0.0
        )

    def test_k_larger_than_retrieved(self) -> None:
        """Should handle k larger than the retrieved list."""
        assert (
            compute_recall_at_k(
                [CorpusId(1), CorpusId(2)],
                {CorpusId(1), CorpusId(2), CorpusId(3)},
                10,
            )
            == 2.0 / 3.0
        )

    def test_k_one(self) -> None:
        """Recall@1 should check only the first result."""
        assert (
            compute_recall_at_k(
                [CorpusId(1), CorpusId(2), CorpusId(3)], {CorpusId(1)}, 1
            )
            == 1.0
        )
        assert (
            compute_recall_at_k(
                [CorpusId(2), CorpusId(1), CorpusId(3)], {CorpusId(1)}, 1
            )
            == 0.0
        )


class TestSafeMean:
    """Tests for the _safe_mean helper."""

    def test_empty_list(self) -> None:
        """Should return 0.0 for empty input."""
        assert _safe_mean([]) == 0.0

    def test_single_value(self) -> None:
        """Should return the value for a single-element list."""
        assert _safe_mean([0.5]) == 0.5

    def test_multiple_values(self) -> None:
        """Should compute the arithmetic mean."""
        assert _safe_mean([0.0, 0.5, 1.0]) == 0.5


class TestFormatResultsTable:
    """Tests for the format_results_table function."""

    def _render(self, table: Table) -> str:
        """Render a Rich Table to a plain string."""
        console = Console()
        with console.capture() as capture:
            console.print(table)
        return capture.get()

    def test_header_present(self) -> None:
        """Table should have a header with Relation and R@k columns."""
        results = EvaluationResults(
            overall=RecallMetrics(recall_at_k={1: 0.5, 5: 0.8}, num_queries=10),
            per_relation={},
            mean_retrieved_per_query=7.0,
        )
        text = self._render(format_results_table(results, ks=(1, 5)))
        assert "Relation" in text
        assert "R@1" in text
        assert "R@5" in text

    def test_overall_row(self) -> None:
        """Table should include an Overall row with scores."""
        results = EvaluationResults(
            overall=RecallMetrics(recall_at_k={1: 0.5}, num_queries=10),
            per_relation={},
            mean_retrieved_per_query=7.0,
        )
        text = self._render(format_results_table(results, ks=(1,)))
        assert "Overall" in text
        assert "0.500" in text

    def test_per_relation_rows(self) -> None:
        """Table should include per-relation rows."""
        results = EvaluationResults(
            overall=RecallMetrics(recall_at_k={1: 0.5}, num_queries=10),
            per_relation={
                "reference": RecallMetrics(recall_at_k={1: 0.8}, num_queries=5),
                "co_citation": RecallMetrics(recall_at_k={1: 0.2}, num_queries=5),
            },
            mean_retrieved_per_query=7.0,
        )
        text = self._render(format_results_table(results, ks=(1,)))
        assert "reference" in text
        assert "co_citation" in text
