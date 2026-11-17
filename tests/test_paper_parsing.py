"""Unit tests for paper_parsing.py."""

import json
from typing import Any

import pytest

from genret.paper_parsing import (
    Facets,
    FacetsWithEvidenceRecord,
    FacetWithEvidence,
    PaperGraph,
    RelationType,
    Section,
    SectionRange,
    SectionType,
    Span,
    SpanAttributes,
    TrainingPair,
    assign_paragraphs_to_sections,
    build_section_tree,
    classify_section,
    extract_paragraph_texts,
    is_direct_child,
    parse_annotations,
    parse_paper,
    parse_section_headers,
    parse_section_number,
)
from genret.types import CorpusId


class TestClassifySection:
    """Tests for classify_section function."""

    @pytest.mark.parametrize(
        ("heading", "expected"),
        [
            ("Abstract", SectionType.ABSTRACT),
            ("1. Introduction", SectionType.INTRODUCTION),
            ("Introduction", SectionType.INTRODUCTION),
            ("2 Related Work", SectionType.RELATED_WORK),
            ("Prior Work", SectionType.RELATED_WORK),
            ("Literature Review", SectionType.RELATED_WORK),
            ("Background", SectionType.BACKGROUND),
            ("Preliminaries", SectionType.BACKGROUND),
            ("3. Method", SectionType.METHOD),
            ("Methodology", SectionType.METHOD),
            ("Our Approach", SectionType.METHOD),
            ("Model Architecture", SectionType.METHOD),
            ("System Design", SectionType.METHOD),
            ("4 Experiments", SectionType.EXPERIMENTS),
            ("Experimental Setup", SectionType.EXPERIMENTS),
            ("Evaluation", SectionType.EXPERIMENTS),
            ("Implementation Details", SectionType.EXPERIMENTS),
            ("5. Results", SectionType.RESULTS),
            ("Findings", SectionType.RESULTS),
            ("Analysis", SectionType.RESULTS),
            ("Performance", SectionType.RESULTS),
            ("Discussion", SectionType.DISCUSSION),
            ("6 Conclusion", SectionType.CONCLUSION),
            ("Conclusions and Future Work", SectionType.CONCLUSION),
            ("Summary", SectionType.CONCLUSION),
            ("Acknowledgments", SectionType.ACKNOWLEDGMENTS),
            ("Acknowledgements", SectionType.ACKNOWLEDGMENTS),
            ("References", SectionType.REFERENCES),
            ("Bibliography", SectionType.REFERENCES),
            ("A. Appendix", SectionType.APPENDIX),
            ("Appendix A", SectionType.APPENDIX),
            ("Supplementary Material", SectionType.APPENDIX),
            # Letter-numbered appendices
            ("A.4 Costs", SectionType.APPENDIX),
            ("A. Details", SectionType.APPENDIX),
            ("B.1.2 Something", SectionType.APPENDIX),
            ("C.1 More Details", SectionType.APPENDIX),
            ("Random Section", SectionType.OTHER),
            ("Data", SectionType.OTHER),
            # Case insensitive
            ("INTRODUCTION", SectionType.INTRODUCTION),
            ("introduction", SectionType.INTRODUCTION),
            ("InTrOdUcTiOn", SectionType.INTRODUCTION),
            # Ignore section numbers
            ("1.2.3 Method", SectionType.METHOD),
        ],
    )
    def test_classifies_sections_correctly(
        self, heading: str, expected: SectionType
    ) -> None:
        """Should classify common section headings correctly."""
        assert classify_section(heading) == expected


class TestParseSectionNumber:
    """Tests for parse_section_number function."""

    @pytest.mark.parametrize(
        ("input_str", "expected"),
        [
            ("1", (1,)),
            ("2", (2,)),
            ("1.1", (1, 1)),
            ("1.2", (1, 2)),
            ("1.2.3", (1, 2, 3)),
            ("3.1.4.1", (3, 1, 4, 1)),
            ("", ()),
            (None, ()),
            (" 1 . 2 ", (1, 2)),
        ],
    )
    def test_parses_section_numbers(
        self, input_str: str | None, expected: tuple[int, ...]
    ) -> None:
        """Should parse section number strings correctly."""
        assert parse_section_number(input_str) == expected


class TestParseAnnotations:
    """Tests for parse_annotations function."""

    def test_parses_valid_json(self) -> None:
        """Should parse valid annotation JSON."""
        json_str = json.dumps(
            [
                {"start": 0, "end": 10, "attributes": {"n": "1"}},
                {"start": 20, "end": 30, "attributes": None},
            ]
        )
        spans = parse_annotations(json_str)

        assert len(spans) == 2
        assert spans[0] == Span(start=0, end=10, attributes=SpanAttributes(n="1"))
        assert spans[1] == Span(start=20, end=30, attributes=None)

    def test_handles_string_offsets(self) -> None:
        """Should handle string-typed start/end values."""
        json_str = json.dumps([{"start": "0", "end": "10", "attributes": None}])
        spans = parse_annotations(json_str)

        assert len(spans) == 1
        assert spans[0].start == 0
        assert spans[0].end == 10

    def test_sorts_by_start(self) -> None:
        """Should return spans sorted by start position."""
        json_str = json.dumps(
            [
                {"start": 30, "end": 40, "attributes": None},
                {"start": 0, "end": 10, "attributes": None},
                {"start": 20, "end": 25, "attributes": None},
            ]
        )
        spans = parse_annotations(json_str)

        assert [s.start for s in spans] == [0, 20, 30]

    def test_empty_string(self) -> None:
        """Should return empty list for empty string."""
        assert parse_annotations("") == []

    def test_empty_array(self) -> None:
        """Should return empty list for empty array."""
        assert parse_annotations("[]") == []


class TestBuildSectionTree:
    """Tests for build_section_tree function."""

    def test_single_section(self) -> None:
        """Should build tree with single section."""
        body_text = "\nIntroduction\n\nThis is the first paragraph."
        headers = [Span(start=1, end=13, attributes=SpanAttributes(n="1"))]
        paragraphs = [Span(start=15, end=44, attributes=None)]

        sections = build_section_tree(headers, paragraphs, body_text)

        assert len(sections) == 1
        assert sections[0].heading == "Introduction"
        assert sections[0].type == SectionType.INTRODUCTION
        assert len(sections[0].paragraphs) == 1
        assert "first paragraph" in sections[0].paragraphs[0]

    def test_multiple_top_level_sections(self) -> None:
        """Should build multiple top-level sections."""
        body_text = "\nIntro\n\nPara 1.\n\nMethod\n\nPara 2."
        headers = [
            Span(start=1, end=6, attributes=SpanAttributes(n="1")),
            Span(start=17, end=23, attributes=SpanAttributes(n="2")),
        ]
        paragraphs = [
            Span(start=8, end=15, attributes=None),
            Span(start=25, end=32, attributes=None),
        ]

        sections = build_section_tree(headers, paragraphs, body_text)

        assert len(sections) == 2
        assert sections[0].heading == "Intro"
        assert sections[1].heading == "Method"
        assert "Para 1" in sections[0].paragraphs[0]
        assert "Para 2" in sections[1].paragraphs[0]

    def test_nested_subsections(self) -> None:
        """Should build nested subsections correctly."""
        body_text = "\nMethod\n\nOverview.\n\nDetails\n\nMore details."
        headers = [
            Span(start=1, end=7, attributes=SpanAttributes(n="1")),
            Span(start=20, end=27, attributes=SpanAttributes(n="1.1")),
        ]
        paragraphs = [
            Span(start=9, end=18, attributes=None),
            Span(start=29, end=42, attributes=None),
        ]

        sections = build_section_tree(headers, paragraphs, body_text)

        assert len(sections) == 1
        assert sections[0].heading == "Method"
        assert len(sections[0].subsections) == 1
        assert sections[0].subsections[0].heading == "Details"

    def test_no_headers(self) -> None:
        """Should create single section when no headers."""
        body_text = "This is all the content without any sections."
        paragraphs = [Span(start=0, end=45, attributes=None)]

        sections = build_section_tree([], paragraphs, body_text)

        assert len(sections) == 1
        assert sections[0].heading == ""
        assert sections[0].type == SectionType.OTHER

    def test_empty_paragraphs(self) -> None:
        """Should handle empty paragraph text."""
        body_text = "\nIntro\n\n   \n\nReal content."
        headers = [Span(start=1, end=6, attributes=SpanAttributes(n="1"))]
        paragraphs = [
            Span(start=8, end=11, attributes=None),  # Just whitespace
            Span(start=13, end=26, attributes=None),
        ]

        sections = build_section_tree(headers, paragraphs, body_text)

        # Should only have one non-empty paragraph
        assert len(sections[0].paragraphs) == 1
        assert "Real content" in sections[0].paragraphs[0]


class TestParsePaper:
    """Tests for parse_paper function."""

    def test_parses_complete_record(self) -> None:
        """Should parse a complete S2ORC record."""
        record = {
            "corpusid": CorpusId(12345),
            "title": "A Great Paper",
            "authors": ["Alice", "Bob"],
            "body": {
                "text": "\nIntroduction\n\nThis paper is great.",
                "annotations": {
                    "section_header": json.dumps(
                        [
                            {"start": 1, "end": 13, "attributes": {"n": "1"}},
                        ]
                    ),
                    "paragraph": json.dumps(
                        [
                            {"start": 15, "end": 35, "attributes": None},
                        ]
                    ),
                },
            },
        }

        paper = parse_paper(record)

        assert paper is not None
        assert paper.corpusid == CorpusId(12345)
        assert paper.title == "A Great Paper"
        assert list(paper.authors) == ["Alice", "Bob"]
        assert len(paper.sections) == 1
        assert paper.sections[0].heading == "Introduction"

    def test_returns_none_for_missing_body(self) -> None:
        """Should return None when body is missing."""
        record: dict[str, Any] = {
            "corpusid": 12345,
            "title": "A Paper",
            "authors": [],
        }
        assert parse_paper(record) is None

    def test_returns_none_for_empty_body_text(self) -> None:
        """Should return None when body text is empty."""
        record: dict[str, Any] = {
            "corpusid": 12345,
            "title": "A Paper",
            "authors": [],
            "body": {"text": "", "annotations": {}},
        }
        assert parse_paper(record) is None

    def test_returns_none_for_missing_corpusid(self) -> None:
        """Should return None when corpusid is missing."""
        record: dict[str, Any] = {
            "title": "A Paper",
            "authors": [],
            "body": {"text": "content", "annotations": {}},
        }
        assert parse_paper(record) is None

    def test_handles_no_annotations(self) -> None:
        """Should handle records with no annotations."""
        record: dict[str, Any] = {
            "corpusid": 12345,
            "title": "A Paper",
            "authors": [],
            "body": {"text": "Just plain text content."},
        }

        paper = parse_paper(record)

        assert paper is not None
        assert len(paper.sections) == 1
        assert paper.sections[0].paragraphs[0] == "Just plain text content."

    def test_handles_empty_authors(self) -> None:
        """Should handle empty or None authors."""
        record: dict[str, Any] = {
            "corpusid": 12345,
            "title": "A Paper",
            "authors": None,
            "body": {"text": "content", "annotations": {}},
        }

        paper = parse_paper(record)

        assert paper is not None
        assert list(paper.authors) == []


class TestModels:
    """Tests for Pydantic models."""

    def test_paper_graph_serialization(self) -> None:
        """PaperGraph should serialize to JSON correctly."""
        paper = PaperGraph(
            corpusid=CorpusId(123),
            title="Test Paper",
            authors=("Author 1",),
            sections=(
                Section(
                    heading="Intro",
                    type=SectionType.INTRODUCTION,
                    paragraphs=("First paragraph.",),
                    subsections=(),
                ),
            ),
        )

        json_str = paper.model_dump_json()
        data = json.loads(json_str)

        assert data["corpusid"] == 123
        assert data["title"] == "Test Paper"
        assert data["authors"] == ["Author 1"]
        assert data["sections"][0]["heading"] == "Intro"
        assert data["sections"][0]["type"] == "introduction"

    def test_facets_model(self) -> None:
        """Facets model should work correctly."""
        facets = Facets(
            corpusid=CorpusId(123),
            problems=["Problem 1", "Problem 2"],
            methods=["Method 1"],
            results=["Result 1", "Result 2", "Result 3"],
            contributions=["Contribution 1"],
        )

        assert facets.corpusid == CorpusId(123)
        assert list(facets.problems) == ["Problem 1", "Problem 2"]
        assert len(facets.results) == 3

    def test_facets_with_evidence_to_facets(self) -> None:
        """to_facets should convert objects to string facets."""
        record = FacetsWithEvidenceRecord(
            corpusid=CorpusId(123),
            problems=(FacetWithEvidence(facet="Problem A", evidence="E"),),
            methods=(FacetWithEvidence(facet="Method A", evidence="E"),),
            results=(FacetWithEvidence(facet="Result A", evidence="E"),),
            contributions=(FacetWithEvidence(facet="Contribution A", evidence="E"),),
        )

        facets = record.to_facets()

        assert facets.corpusid == CorpusId(123)
        assert tuple(facets.problems) == ("Problem A",)
        assert tuple(facets.methods) == ("Method A",)
        assert tuple(facets.results) == ("Result A",)
        assert tuple(facets.contributions) == ("Contribution A",)

    def test_training_pair_model(self) -> None:
        """TrainingPair model should work correctly."""
        pair = TrainingPair(
            paper_a=CorpusId(123),
            paper_b=CorpusId(456),
            relation=RelationType.CO_CITATION,
            score=0.8,
        )

        assert pair.paper_a == CorpusId(123)
        assert pair.paper_b == CorpusId(456)
        assert pair.relation == RelationType.CO_CITATION
        assert pair.score == 0.8


class TestExtractParagraphTexts:
    """Tests for extract_paragraph_texts function."""

    def test_extracts_non_empty_paragraphs(self) -> None:
        """Should extract non-empty paragraph text."""
        body_text = "First paragraph.\n\nSecond paragraph."
        paragraphs = [
            Span(start=0, end=16),
            Span(start=18, end=35),
        ]
        result = extract_paragraph_texts(paragraphs, body_text)
        assert result == ["First paragraph.", "Second paragraph."]

    def test_skips_empty_paragraphs(self) -> None:
        """Should skip paragraphs that are only whitespace."""
        body_text = "Content.\n\n   \n\nMore content."
        paragraphs = [
            Span(start=0, end=8),
            Span(start=10, end=13),  # Just whitespace
            Span(start=15, end=28),
        ]
        result = extract_paragraph_texts(paragraphs, body_text)
        assert result == ["Content.", "More content."]


class TestParseSectionHeaders:
    """Tests for parse_section_headers function."""

    def test_parses_headers_with_numbers(self) -> None:
        """Should parse headers into SectionRange objects."""
        body_text = "\nIntro\n\nContent.\n\nMethod\n\nMore."
        headers = [
            Span(start=1, end=6, attributes=SpanAttributes(n="1")),
            Span(start=18, end=24, attributes=SpanAttributes(n="2")),
        ]
        result = parse_section_headers(headers, body_text)

        assert len(result) == 2
        assert result[0] == SectionRange(number=(1,), start=1, end=18, heading="Intro")
        assert result[1] == SectionRange(
            number=(2,), start=18, end=len(body_text), heading="Method"
        )

    def test_handles_nested_section_numbers(self) -> None:
        """Should parse nested section numbers."""
        body_text = "\nMethod\n\nOverview\n\nDetails"
        headers = [
            Span(start=1, end=7, attributes=SpanAttributes(n="1")),
            Span(start=9, end=17, attributes=SpanAttributes(n="1.1")),
        ]
        result = parse_section_headers(headers, body_text)

        assert result[0].number == (1,)
        assert result[1].number == (1, 1)


class TestAssignParagraphsToSections:
    """Tests for assign_paragraphs_to_sections function."""

    def test_assigns_paragraphs_to_correct_sections(self) -> None:
        """Should assign paragraphs based on offset ranges."""
        body_text = "Intro\nPara1\nMethod\nPara2"
        section_ranges = [
            SectionRange(number=(1,), start=0, end=12, heading="Intro"),
            SectionRange(number=(2,), start=12, end=24, heading="Method"),
        ]
        paragraphs = [
            Span(start=6, end=11),  # Para1 - in Intro
            Span(start=19, end=24),  # Para2 - in Method
        ]

        result = assign_paragraphs_to_sections(paragraphs, section_ranges, body_text)

        assert result[(1,)] == ["Para1"]
        assert result[(2,)] == ["Para2"]


class TestIsDirectChild:
    """Tests for is_direct_child function."""

    @pytest.mark.parametrize(
        ("child", "parent", "expected"),
        [
            ((1,), (), True),  # Top-level section
            ((1, 1), (1,), True),  # Direct child
            ((1, 2, 3), (1, 2), True),  # Deeper child
            ((1, 1), (), False),  # Two levels deep from root
            ((2,), (1,), False),  # Different branch
            ((1, 2), (1, 1), False),  # Sibling, not child
            ((), (), False),  # Empty is not child of empty
        ],
    )
    def test_identifies_direct_children(
        self, child: tuple[int, ...], parent: tuple[int, ...], expected: bool
    ) -> None:
        """Should correctly identify direct parent-child relationships."""
        assert is_direct_child(child, parent) == expected
