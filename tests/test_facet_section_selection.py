"""Tests for core-section selection used by facet extraction."""

from genret.paper_parsing import PaperGraph, Section, SectionType
from genret.processing.facet_extract_cmd import (
    SectionMode,
    _prepare_prompt_paper_text,  # pyright: ignore[reportPrivateUsage]
    select_core_section_text,
)
from genret.types import CorpusId


def _section(heading: str, body: str) -> Section:
    return Section(heading=heading, type=SectionType.OTHER, paragraphs=[body])


def _paper(sections: list[Section], *, abstract: str | None = "ABS") -> PaperGraph:
    return PaperGraph(
        corpusid=CorpusId(1),
        title="Title",
        authors=[],
        abstract=abstract,
        sections=sections,
    )


def test_core_keeps_relevant_sections_and_abstract() -> None:
    """Core mode keeps the abstract and facet-relevant sections."""
    paper = _paper(
        [
            _section("Introduction", "intro body"),
            _section("Method", "method body"),
            _section("Experiments", "exp body"),
            _section("Results", "results body"),
            _section("Conclusion", "concl body"),
        ]
    )
    text = select_core_section_text(paper)
    assert "ABS" in text  # abstract included
    for body in ["intro body", "method body", "exp body", "results body", "concl body"]:
        assert body in text


def test_core_drops_backmatter_and_offtopic() -> None:
    """Core mode drops references, appendix, related work, limitations, etc."""
    paper = _paper(
        [
            _section("Introduction", "intro body"),
            _section("Related Work", "related body"),
            _section("Limitations", "limit body"),
            _section("References", "refs body"),
            _section("A Appendix", "appendix body"),
            _section("Acknowledgments", "ack body"),
        ]
    )
    text = select_core_section_text(paper)
    assert "intro body" in text
    for dropped in [
        "related body",
        "limit body",
        "refs body",
        "appendix body",
        "ack body",
    ]:
        assert dropped not in text


def test_core_keeps_data_task_system_sections() -> None:
    """Core mode keeps data/task/system sections (added keywords)."""
    paper = _paper(
        [
            _section("Dataset", "dataset body"),
            _section("Task Setup", "task body"),
            _section("System Description", "system body"),
        ]
    )
    text = select_core_section_text(paper)
    for kept in ["dataset body", "task body", "system body"]:
        assert kept in text


def test_core_falls_back_to_all_sections_when_nothing_matches() -> None:
    """With no keyword match, core mode keeps all sections (no empty paper)."""
    paper = _paper(
        [
            _section("Foobar", "foo body"),
            _section("Widgets", "widget body"),
        ],
        abstract=None,
    )
    text = select_core_section_text(paper)
    # No keyword match -> keep everything rather than emit an empty paper.
    assert "foo body" in text
    assert "widget body" in text


def test_full_mode_includes_everything() -> None:
    """Full mode passes the whole paper through (no section filtering)."""
    paper = _paper([_section("References", "refs body")])
    full = _prepare_prompt_paper_text(paper, section_mode=SectionMode.FULL)
    core = _prepare_prompt_paper_text(paper, section_mode=SectionMode.CORE)
    assert "refs body" in full
    # 'references' has no core keyword, but it's the only section -> fallback keeps it.
    assert "refs body" in core
