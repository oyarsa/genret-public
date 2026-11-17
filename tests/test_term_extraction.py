"""Unit tests for term_extraction.py pure functions."""

import json
from pathlib import Path

from genret.paper_parsing import (
    AspectTerms,
    FacetEvidence,
    FacetRecordType,
    FacetsSummary,
    PaperGraph,
    PaperTerms,
    Section,
    SectionType,
)
from genret.processing.term_extraction import (
    deduplicate_terms,
    extract_abstract_text,
    format_facets_for_prompt,
    load_facets_index,
    prepare_term_prompt,
)
from genret.types import CorpusId

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_paper(sections: list[Section] | None = None, corpusid: int = 1) -> PaperGraph:
    """Build a PaperGraph with sensible defaults."""
    return PaperGraph(
        corpusid=CorpusId(corpusid),
        title="Test Paper",
        authors=["Alice"],
        sections=sections or [],
    )


def _make_facets(**kwargs: list[str]) -> FacetsSummary:
    """Build a FacetsSummary with default single items per facet."""
    return FacetsSummary(
        problems=kwargs.get("problems", ["Low-resource NER is challenging."]),
        methods=kwargs.get("methods", ["Fine-tune RoBERTa-base."]),
        results=kwargs.get("results", ["F1 improved to 89.7%."]),
        contributions=kwargs.get("contributions", ["NER-Bench suite."]),
    )


class TestExtractAbstractText:
    """Tests for extract_abstract_text function."""

    def test_with_abstract_section(self) -> None:
        """Should return abstract text when an abstract section exists."""
        paper = _make_paper(
            sections=[
                Section(
                    heading="Abstract",
                    type=SectionType.ABSTRACT,
                    paragraphs=["This paper presents a method.", "It works well."],
                ),
                Section(
                    heading="Introduction",
                    type=SectionType.INTRODUCTION,
                    paragraphs=["Background info."],
                ),
            ]
        )
        result = extract_abstract_text(paper)
        assert result is not None
        assert "This paper presents a method." in result
        assert "It works well." in result

    def test_without_abstract_section(self) -> None:
        """Should return None when no abstract section exists."""
        paper = _make_paper(
            sections=[
                Section(
                    heading="Introduction",
                    type=SectionType.INTRODUCTION,
                    paragraphs=["Background info."],
                ),
            ]
        )
        assert extract_abstract_text(paper) is None

    def test_empty_abstract(self) -> None:
        """Should return None when abstract section has only empty paragraphs."""
        paper = _make_paper(
            sections=[
                Section(
                    heading="Abstract",
                    type=SectionType.ABSTRACT,
                    paragraphs=["", "   "],
                ),
            ]
        )
        assert extract_abstract_text(paper) is None

    def test_nested_abstract(self) -> None:
        """Should find abstract in nested subsections."""
        paper = _make_paper(
            sections=[
                Section(
                    heading="Front Matter",
                    type=SectionType.OTHER,
                    paragraphs=[],
                    subsections=[
                        Section(
                            heading="Abstract",
                            type=SectionType.ABSTRACT,
                            paragraphs=["Nested abstract text."],
                        ),
                    ],
                ),
            ]
        )
        result = extract_abstract_text(paper)
        assert result == "Nested abstract text."

    def test_no_sections(self) -> None:
        """Should return None for paper with no sections."""
        paper = _make_paper(sections=[])
        assert extract_abstract_text(paper) is None


class TestFormatFacetsForPrompt:
    """Tests for format_facets_for_prompt function."""

    def test_formats_all_facets(self) -> None:
        """Should format all non-empty facets with labels and bullets."""
        facets = _make_facets()
        result = format_facets_for_prompt(facets)
        assert "Problems:" in result
        assert "Methods:" in result
        assert "Results:" in result
        assert "Contributions:" in result
        assert "  - Low-resource NER is challenging." in result

    def test_skips_empty_facets(self) -> None:
        """Should omit facets with empty item lists."""
        facets = FacetsSummary(
            problems=["A problem."],
            methods=[],
            results=[],
            contributions=[],
        )
        result = format_facets_for_prompt(facets)
        assert "Problems:" in result
        assert "Methods:" not in result

    def test_all_empty(self) -> None:
        """Should return empty string when all facets are empty."""
        facets = FacetsSummary()
        assert format_facets_for_prompt(facets) == ""

    def test_with_evidence(self) -> None:
        """Should include evidence quotes beneath each facet item."""
        facets = _make_facets()
        evidence: FacetEvidence = {
            "problems": [["Quote about the problem."]],
            "methods": [["Method quote 1.", "Method quote 2."]],
        }
        result = format_facets_for_prompt(facets, evidence=evidence)
        assert "  - Low-resource NER is challenging." in result
        assert '    > "Quote about the problem."' in result
        assert '    > "Method quote 1."' in result
        assert '    > "Method quote 2."' in result

    def test_evidence_none_is_noop(self) -> None:
        """Passing evidence=None should behave like no evidence."""
        facets = _make_facets()
        with_none = format_facets_for_prompt(facets, evidence=None)
        without = format_facets_for_prompt(facets)
        assert with_none == without


# ── prepare_term_prompt ──────────────────────────────────────────────────────


class TestPrepareTermPrompt:
    """Tests for prepare_term_prompt function."""

    def test_with_abstract(self) -> None:
        """Should include title, abstract, and facets in prompt."""
        facets = _make_facets()
        result = prepare_term_prompt("My Title", "An abstract.", facets)
        assert "Title: My Title" in result
        assert "An abstract." in result
        assert "Problems:" in result
        assert "paper_terms" in result
        assert "aspect_terms" in result

    def test_without_abstract(self) -> None:
        """Should use placeholder when abstract is None."""
        facets = _make_facets()
        result = prepare_term_prompt("My Title", None, facets)
        assert "[No abstract available]" in result

    def test_custom_counts(self) -> None:
        """Should include custom count ranges in the prompt."""
        facets = _make_facets()
        result = prepare_term_prompt(
            "Title",
            "Abstract.",
            facets,
            paper_terms_count="5-8",
            aspect_terms_count="2-3",
        )
        assert "5-8" in result
        assert "2-3" in result

    def test_json_format_instructions(self) -> None:
        """Should include JSON response format instructions."""
        facets = _make_facets()
        result = prepare_term_prompt("Title", "Abstract.", facets)
        assert '"paper_terms"' in result
        assert '"aspect_terms"' in result
        assert '"problems"' in result


class TestDeduplicateTerms:
    """Tests for deduplicate_terms function."""

    def test_removes_overlapping_aspect_terms(self) -> None:
        """Should remove aspect terms that match paper terms."""
        terms = PaperTerms(
            corpusid=CorpusId(1),
            paper_terms=["transformer architecture", "attention mechanism"],
            aspect_terms=AspectTerms(
                problems=["data scarcity"],
                methods=["transformer architecture", "beam search"],
                results=["BLEU score improvement"],
                contributions=["attention mechanism"],
            ),
        )
        result = deduplicate_terms(terms)
        assert "transformer architecture" not in result.aspect_terms.methods
        assert "beam search" in result.aspect_terms.methods
        assert "attention mechanism" not in result.aspect_terms.contributions
        assert "data scarcity" in result.aspect_terms.problems

    def test_case_insensitive(self) -> None:
        """Should match regardless of case."""
        terms = PaperTerms(
            corpusid=CorpusId(1),
            paper_terms=["Transformer Architecture"],
            aspect_terms=AspectTerms(
                problems=[],
                methods=["transformer architecture"],
                results=[],
                contributions=[],
            ),
        )
        result = deduplicate_terms(terms)
        assert len(result.aspect_terms.methods) == 0

    def test_no_overlaps_unchanged(self) -> None:
        """Should not remove anything when there are no overlaps."""
        terms = PaperTerms(
            corpusid=CorpusId(1),
            paper_terms=["term a", "term b"],
            aspect_terms=AspectTerms(
                problems=["term c"],
                methods=["term d"],
                results=["term e"],
                contributions=["term f"],
            ),
        )
        result = deduplicate_terms(terms)
        assert len(result.aspect_terms.problems) == 1
        assert len(result.aspect_terms.methods) == 1
        assert len(result.aspect_terms.results) == 1
        assert len(result.aspect_terms.contributions) == 1

    def test_paper_terms_preserved(self) -> None:
        """Should never modify paper-level terms."""
        terms = PaperTerms(
            corpusid=CorpusId(1),
            paper_terms=["kept term"],
            aspect_terms=AspectTerms(
                problems=["kept term"],
                methods=[],
                results=[],
                contributions=[],
            ),
        )
        result = deduplicate_terms(terms)
        assert "kept term" in result.paper_terms


class TestLoadFacetsIndex:
    """Tests for load_facets_index function."""

    def test_loads_from_jsonl(self, tmp_path: Path) -> None:
        """Should load facets into a corpus-ID-keyed index."""
        facets_path = tmp_path / "facets.jsonl"
        records = [
            {
                "corpusid": 100,
                "problems": ["P1"],
                "methods": ["M1"],
                "results": ["R1"],
                "contributions": ["C1"],
            },
            {
                "corpusid": 200,
                "problems": ["P2"],
                "methods": [],
                "results": [],
                "contributions": [],
            },
        ]
        facets_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )

        index, evidence = load_facets_index(
            facets_path,
            facets_type=FacetRecordType.STANDARD,
        )
        assert len(index) == 2
        assert CorpusId(100) in index
        assert CorpusId(200) in index
        assert list(index[CorpusId(100)].problems) == ["P1"]
        assert list(index[CorpusId(200)].methods) == []
        assert evidence == {}

    def test_empty_file(self, tmp_path: Path) -> None:
        """Should return empty dicts for empty file."""
        facets_path = tmp_path / "facets.jsonl"
        facets_path.write_text("", encoding="utf-8")
        index, evidence = load_facets_index(
            facets_path,
            facets_type=FacetRecordType.STANDARD,
        )
        assert index == {}
        assert evidence == {}

    def test_loads_evidence_format(self, tmp_path: Path) -> None:
        """Should parse evidence-backed facet records when requested."""
        facets_path = tmp_path / "facets.jsonl"
        record = {
            "corpusid": 100,
            "problems": [{"facet": "P1", "evidence": "E1"}],
            "methods": [{"facet": "M1", "evidence": "E2"}],
            "results": [{"facet": "R1", "evidence": "E3"}],
            "contributions": [{"facet": "C1", "evidence": "E4"}],
        }
        facets_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        index, evidence = load_facets_index(
            facets_path,
            facets_type=FacetRecordType.WITH_EVIDENCE,
        )
        assert list(index[CorpusId(100)].problems) == ["P1"]
        assert list(index[CorpusId(100)].methods) == ["M1"]
        assert CorpusId(100) in evidence
        assert evidence[CorpusId(100)]["problems"] == [["E1"]]
        assert evidence[CorpusId(100)]["methods"] == [["E2"]]
