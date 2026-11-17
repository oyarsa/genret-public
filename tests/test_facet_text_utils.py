"""Unit tests for facet_evaluation text helpers."""

from genret.paper_parsing import PaperGraph, Section, SectionType
from genret.processing.facet_evaluation import (
    count_sentences,
    extract_paragraphs,
    normalise_ws,
)
from genret.types import CorpusId


class TestExtractParagraphs:
    """Tests for extract_paragraphs function."""

    def test_flat_sections(self) -> None:
        """Should extract paragraphs from flat section list."""
        paper = PaperGraph(
            corpusid=CorpusId(1),
            title="T",
            authors=[],
            sections=[
                Section(
                    heading="Intro",
                    type=SectionType.INTRODUCTION,
                    paragraphs=["First paragraph.", "Second paragraph."],
                ),
                Section(
                    heading="Method",
                    type=SectionType.METHOD,
                    paragraphs=["Method text."],
                ),
            ],
        )
        result = extract_paragraphs(paper)
        assert len(result) == 3
        assert result[0] == ("Intro", "First paragraph.")
        assert result[2] == ("Method", "Method text.")

    def test_nested_sections(self) -> None:
        """Should recursively traverse nested subsections."""
        paper = PaperGraph(
            corpusid=CorpusId(1),
            title="T",
            authors=[],
            sections=[
                Section(
                    heading="Parent",
                    type=SectionType.OTHER,
                    paragraphs=["Parent text."],
                    subsections=[
                        Section(
                            heading="Child",
                            type=SectionType.OTHER,
                            paragraphs=["Child text."],
                        )
                    ],
                )
            ],
        )
        result = extract_paragraphs(paper)
        assert len(result) == 2
        assert result[0][0] == "Parent"
        assert result[1][0] == "Child"

    def test_skips_empty_paragraphs(self) -> None:
        """Should skip whitespace-only paragraphs."""
        paper = PaperGraph(
            corpusid=CorpusId(1),
            title="T",
            authors=[],
            sections=[
                Section(
                    heading="S",
                    type=SectionType.OTHER,
                    paragraphs=["Real.", "", "   "],
                )
            ],
        )
        result = extract_paragraphs(paper)
        assert len(result) == 1

    def test_empty_paper(self) -> None:
        """Should return empty list for paper with no sections."""
        paper = PaperGraph(corpusid=CorpusId(1), title="T", authors=[], sections=[])
        assert extract_paragraphs(paper) == []


class TestCountSentences:
    """Tests for count_sentences function."""

    def test_single_sentence(self) -> None:
        """Should count one sentence ending with period."""
        assert count_sentences("This is one sentence.") == 1

    def test_two_sentences(self) -> None:
        """Should count two period-terminated sentences."""
        assert count_sentences("First. Second.") == 2

    def test_question_and_exclamation(self) -> None:
        """Should count sentences ending with ? and !."""
        assert count_sentences("Really? Yes!") == 2

    def test_empty_string(self) -> None:
        """Should return 0 for empty string."""
        assert count_sentences("") == 0

    def test_no_punctuation(self) -> None:
        """Should count text without ending punctuation as one sentence."""
        assert count_sentences("No ending punctuation") == 1


class TestNormaliseWs:
    """Tests for normalise_ws whitespace normalisation."""

    def test_collapses_spaces(self) -> None:
        """Multiple spaces become one."""
        assert normalise_ws("hello   world") == "hello world"

    def test_collapses_newlines_and_tabs(self) -> None:
        """Newlines, tabs, and mixed whitespace become single spaces."""
        assert normalise_ws("hello\n\tworld") == "hello world"

    def test_strips_leading_trailing(self) -> None:
        """Leading and trailing whitespace is stripped."""
        assert normalise_ws("  hello  ") == "hello"

    def test_empty_string(self) -> None:
        """Empty string returns empty."""
        assert normalise_ws("") == ""
