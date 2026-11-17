"""Parsing logic for S2ORC papers into structured paper graphs.

This module provides:
- Pydantic models for representing paper structure (Section, PaperGraph, Facets, etc.)
- Functions to parse S2ORC JSON format into these models
- Section classification based on heading text
"""

import json
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from genret.types import CorpusId
from genret.utils import iter_jsonl, resolve_jsonl, track_progress


class SectionType(StrEnum):
    """Classification of paper sections."""

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    BACKGROUND = "background"
    METHOD = "method"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    ACKNOWLEDGMENTS = "acknowledgments"
    REFERENCES = "references"
    APPENDIX = "appendix"
    OTHER = "other"


def _reignore(pat: str) -> re.Pattern[str]:
    """Compile pattern with ignorecase."""
    return re.compile(pat, re.IGNORECASE)


SECTION_PATTERNS = [
    (_reignore(r"^abstract$"), SectionType.ABSTRACT),
    (_reignore(r"acknowledg"), SectionType.ACKNOWLEDGMENTS),
    (_reignore(r"^references?$|^bibliography$"), SectionType.REFERENCES),
    (_reignore(r"^appendix|^supplementa|^[A-Z]\.(\d+\.?)*"), SectionType.APPENDIX),
    (
        _reignore(r"related\s+work|prior\s+work|literature\s+review"),
        SectionType.RELATED_WORK,
    ),
    (_reignore(r"background|preliminar"), SectionType.BACKGROUND),
    (_reignore(r"introduc"), SectionType.INTRODUCTION),
    (
        _reignore(r"method|approach|model|framework|architecture|system\s+design"),
        SectionType.METHOD,
    ),
    (_reignore(r"experiment|evaluation|setup|implementation"), SectionType.EXPERIMENTS),
    (_reignore(r"result|finding|analysis|performance"), SectionType.RESULTS),
    (_reignore(r"discuss"), SectionType.DISCUSSION),
    (_reignore(r"conclu|summary|future\s+work"), SectionType.CONCLUSION),
]
"""Regex-to-SectionType mapping for classifying section headings, ordered by specificity."""


class SpanAttributes(BaseModel, frozen=True):
    """Attributes for a span annotation."""

    n: str | None = None  # Section number like "1", "1.2", "1.2.3"


class RawAnnotation(BaseModel, frozen=True):
    """Raw annotation from S2ORC JSON before processing."""

    start: int
    end: int
    attributes: SpanAttributes | None = None


class BodyAnnotations(BaseModel, frozen=True):
    """Annotations dict from S2ORC body field."""

    section_header: str | None = None  # JSON string of header annotations
    paragraph: str | None = None  # JSON string of paragraph annotations


class S2ORCBody(BaseModel, frozen=True):
    """Body field from an S2ORC record."""

    text: str
    annotations: BodyAnnotations | None = None


class S2ORCRecord(BaseModel):
    """S2ORC record input format."""

    corpusid: CorpusId
    title: str | None = None
    authors: Sequence[str] | None = None
    body: S2ORCBody | None = None


class Span(BaseModel, frozen=True):
    """A span of text with character offsets."""

    start: int
    end: int
    attributes: SpanAttributes | None = None


class Section(BaseModel, frozen=True):
    """A section of a paper with optional nested subsections."""

    heading: str
    type: SectionType
    paragraphs: Sequence[str]
    subsections: Sequence[Section] = ()


class PaperGraph(BaseModel, frozen=True):
    """Structured representation of a paper."""

    corpusid: CorpusId
    title: str
    authors: Sequence[str]
    sections: Sequence[Section]
    abstract: str | None = None


FACET_NAMES: tuple[str, ...] = ("problems", "methods", "results", "contributions")
"""The four fixed facet categories extracted per paper, matching `Facets`' fields."""


class Facets(BaseModel, frozen=True):
    """LLM-extracted facets from a paper (1-5 items per facet)."""

    corpusid: CorpusId
    problems: Sequence[str]
    methods: Sequence[str]
    results: Sequence[str]
    contributions: Sequence[str]

    def to_dict(self) -> dict[str, list[str]]:
        """Mapping from facet name to facet items."""
        return {
            "problems": list(self.problems),
            "methods": list(self.methods),
            "results": list(self.results),
            "contributions": list(self.contributions),
        }


class FacetWithEvidence(BaseModel, frozen=True):
    """A facet item paired with a single supporting evidence string."""

    facet: str
    evidence: str


class FacetsWithEvidenceRecord(BaseModel, frozen=True):
    """Facet extraction record whose items include evidence snippets."""

    corpusid: CorpusId
    problems: Sequence[FacetWithEvidence]
    methods: Sequence[FacetWithEvidence]
    results: Sequence[FacetWithEvidence]
    contributions: Sequence[FacetWithEvidence]

    def to_facets(self) -> Facets:
        """Convert an evidence-backed facets record to plain string facets."""
        return Facets(
            corpusid=self.corpusid,
            problems=tuple(item.facet for item in self.problems),
            methods=tuple(item.facet for item in self.methods),
            results=tuple(item.facet for item in self.results),
            contributions=tuple(item.facet for item in self.contributions),
        )


class FacetWithEvidenceList(BaseModel, frozen=True):
    """A facet item paired with a list of 1-3 supporting evidence snippets."""

    facet: str
    evidence: Sequence[str]


class FacetsWithEvidenceListRecord(BaseModel, frozen=True):
    """Facet extraction record whose items include a list of evidence snippets."""

    corpusid: CorpusId
    problems: Sequence[FacetWithEvidenceList]
    methods: Sequence[FacetWithEvidenceList]
    results: Sequence[FacetWithEvidenceList]
    contributions: Sequence[FacetWithEvidenceList]

    def to_facets(self) -> Facets:
        """Convert an evidence-list-backed facets record to plain string facets."""
        return Facets(
            corpusid=self.corpusid,
            problems=tuple(item.facet for item in self.problems),
            methods=tuple(item.facet for item in self.methods),
            results=tuple(item.facet for item in self.results),
            contributions=tuple(item.facet for item in self.contributions),
        )


class FacetRecordType(StrEnum):
    """Supported on-disk facet-record schemas."""

    STANDARD = "standard"
    WITH_EVIDENCE = "evidence"
    WITH_EVIDENCE_LIST = "evidence-list"


type FacetEvidence = dict[str, list[list[str]]]
"""Per-item evidence for each facet aspect.

Keys are aspect names (``"problems"``, ``"methods"``, ``"results"``,
``"contributions"``).  Values are lists parallel to the facet items: each
entry is a list of 1-3 evidence strings for that item.
"""


def facets_from_type(raw: dict[str, Any], facets_type: FacetRecordType) -> Facets:
    """Parse a raw facets record according to the configured on-disk schema.

    Args:
        raw: Raw JSON object from facets JSONL.
        facets_type: Schema used by the input file.

    Returns:
        Parsed facets as the canonical plain-text model.
    """
    match facets_type:
        case FacetRecordType.WITH_EVIDENCE:
            return FacetsWithEvidenceRecord.model_validate(raw).to_facets()
        case FacetRecordType.WITH_EVIDENCE_LIST:
            return FacetsWithEvidenceListRecord.model_validate(raw).to_facets()
        case FacetRecordType.STANDARD:
            return Facets.model_validate(raw)


def evidence_from_type(
    raw: dict[str, Any], facets_type: FacetRecordType
) -> FacetEvidence | None:
    """Extract per-item model evidence from a raw facets record.

    Args:
        raw: Raw JSON object from facets JSONL.
        facets_type: Schema used by the input file.

    Returns:
        Per-item evidence mapping, or ``None`` for standard records that carry
        no evidence.
    """
    record: FacetsWithEvidenceRecord | FacetsWithEvidenceListRecord
    match facets_type:
        case FacetRecordType.WITH_EVIDENCE:
            record = FacetsWithEvidenceRecord.model_validate(raw)
            return {
                "problems": [[item.evidence] for item in record.problems],
                "methods": [[item.evidence] for item in record.methods],
                "results": [[item.evidence] for item in record.results],
                "contributions": [[item.evidence] for item in record.contributions],
            }
        case FacetRecordType.WITH_EVIDENCE_LIST:
            record = FacetsWithEvidenceListRecord.model_validate(raw)
            return {
                "problems": [list(item.evidence) for item in record.problems],
                "methods": [list(item.evidence) for item in record.methods],
                "results": [list(item.evidence) for item in record.results],
                "contributions": [list(item.evidence) for item in record.contributions],
            }
        case FacetRecordType.STANDARD:
            return None


class RelationType(StrEnum):
    """Type of relationship between papers."""

    REFERENCE = "reference"
    CO_CITATION = "co_citation"
    BIB_COUPLING = "bib_coupling"


class TrainingPair(BaseModel, frozen=True):
    """A training pair for contrastive learning."""

    paper_a: CorpusId
    paper_b: CorpusId
    relation: RelationType
    score: float


class RelatedPaperInfo(BaseModel, frozen=True):
    """Information about a related paper in a training pair context."""

    related_paper_id: CorpusId
    related_paper_title: str
    relation: RelationType | str
    score: float


class FacetsSummary(BaseModel, frozen=True):
    """Extracted facets from a paper (without corpus ID)."""

    problems: Sequence[str] = ()
    methods: Sequence[str] = ()
    results: Sequence[str] = ()
    contributions: Sequence[str] = ()

    def facet_items(self) -> dict[str, Sequence[str]]:
        """Return all facets as a {name: items} mapping."""
        return {
            "problems": self.problems,
            "methods": self.methods,
            "results": self.results,
            "contributions": self.contributions,
        }


@dataclass(frozen=True, kw_only=True)
class CorpusFacetItems:
    """Per-facet text items for a corpus of papers.

    Each field is a sequence of paper entries (aligned with an external corpus ID
    list), where each paper entry is a sequence of extracted text items for that
    facet.
    """

    problems: Sequence[Sequence[str]]
    """Problem items for each paper, indexed by paper position."""
    methods: Sequence[Sequence[str]]
    """Method items for each paper, indexed by paper position."""
    results: Sequence[Sequence[str]]
    """Result items for each paper, indexed by paper position."""
    contributions: Sequence[Sequence[str]]
    """Contribution items for each paper, indexed by paper position."""

    def iter_facets(
        self,
    ) -> Iterator[tuple[str, Sequence[Sequence[str]]]]:
        """Yield `(facet_name, items)` pairs for all four facets."""
        yield "problems", self.problems
        yield "methods", self.methods
        yield "results", self.results
        yield "contributions", self.contributions

    def by_name(self, name: str) -> Sequence[Sequence[str]]:
        """Look up a facet by name.

        Args:
            name: One of `"problems"`, `"methods"`, `"results"`,
                `"contributions"`.

        Returns:
            The per-paper item sequences for that facet.

        Raises:
            KeyError: If `name` is not a valid facet name.
        """
        match name:
            case "problems":
                return self.problems
            case "methods":
                return self.methods
            case "results":
                return self.results
            case "contributions":
                return self.contributions
            case _:
                raise KeyError(name)


class AspectTerms(BaseModel, frozen=True):
    """Keyphrases for each facet aspect of a paper."""

    problems: Sequence[str] = ()
    methods: Sequence[str] = ()
    results: Sequence[str] = ()
    contributions: Sequence[str] = ()


class PaperTerms(BaseModel, frozen=True):
    """Discriminative keyphrases extracted from a paper for DocID encoding."""

    corpusid: CorpusId
    paper_terms: Sequence[str]
    aspect_terms: AspectTerms


class DocIDRecord(BaseModel, frozen=True):
    """A paper's deterministic document identifier string."""

    corpusid: CorpusId
    docid: str


class IndexStats(BaseModel, frozen=True):
    """Summary statistics for a DocID inverted index."""

    vocabulary_size: int
    num_papers: int
    mean_tokens_per_docid: float
    max_tokens_per_docid: int
    mean_papers_per_token: float


class DocIDIndex(BaseModel, frozen=True):
    """Complete DocID inverted index with metadata, stats, and mappings.

    Written to `docid_index.json` for use in constrained decoding.
    """

    tokeniser: str
    stats: IndexStats
    vocabulary: list[int]
    token_to_papers: dict[str, list[int]]
    paper_to_tokens: dict[str, list[int]]


class EnrichedPaper(BaseModel, frozen=True):
    """A paper graph enriched with facets and ground truth pairs."""

    corpusid: CorpusId
    title: str
    authors: Sequence[str]
    sections: Sequence[Section]
    abstract: str | None = None
    facets: FacetsSummary | None = None
    ground_truth_pairs: Sequence[RelatedPaperInfo] = ()


@dataclass(frozen=True, kw_only=True)
class ScoredPaperPair:
    """A scored pair of papers (for co-citation or bibliographic coupling).

    Used to represent relationships like co-citations (how many times two papers
    are cited together) or bibliographic coupling (how many references they share).
    """

    paper_a: CorpusId
    paper_b: CorpusId
    score: int


@dataclass(frozen=True, kw_only=True)
class _SectionInfo:
    """Intermediate parsed section header before end offsets are computed."""

    number: tuple[int, ...]
    """Hierarchical section number, e.g. ``(1, 2, 3)``."""
    start: int
    """Start character offset in the body text."""
    heading: str
    """Section heading text."""


@dataclass(frozen=True, kw_only=True)
class SectionRange:
    """A parsed section with its text range in the document."""

    number: tuple[int, ...]  # Hierarchical section number, e.g., (1, 2, 3)
    start: int  # Start offset in body text
    end: int  # End offset in body text
    heading: str  # Section heading text


def classify_section(heading: str) -> SectionType:
    """Classify a section heading into a SectionType.

    Args:
        heading: The section heading text (may include section numbers).

    Returns:
        The classified SectionType.
    """
    # Check for appendix-style numbering (A., A.1, B.2.3, etc.) before cleaning
    if re.match(r"^[A-Z]\.(\d+\.?)*", heading):
        return SectionType.APPENDIX

    # Remove section numbers like "1.", "1.2", "A.", etc.
    cleaned = re.sub(r"^[\d.]+\s*", "", heading)
    cleaned = re.sub(r"^[A-Z]\.\s*", "", cleaned)
    cleaned = cleaned.strip()

    for pattern, section_type in SECTION_PATTERNS:
        if pattern.search(cleaned):
            return section_type

    return SectionType.OTHER


def parse_annotations(annotations_json: str) -> list[Span]:
    """Parse S2ORC annotation JSON string into Span objects.

    Args:
        annotations_json: JSON string containing annotation list.

    Returns:
        List of Span objects sorted by start position. If `annotations_json` is empty,
        returns the empty list.
    """
    if not annotations_json:
        return []

    raw_annotations = [
        RawAnnotation.model_validate(ann) for ann in json.loads(annotations_json)
    ]
    spans = [
        Span(start=ann.start, end=ann.end, attributes=ann.attributes)
        for ann in raw_annotations
    ]

    return sorted(spans, key=lambda s: s.start)


def parse_section_number(n: str | None) -> tuple[int, ...]:
    """Parse a section number string into a tuple of integers.

    Args:
        n: Section number string like "1", "1.2", "1.2.3", or None.

    Returns:
        Tuple of integers, e.g., (1, 2, 3). Empty tuple if None or invalid.
    """
    if not n:
        return ()

    parts: list[int] = []
    for raw_part in n.split("."):
        segment = raw_part.strip()
        if segment.isdigit():
            parts.append(int(segment))
        elif len(segment) == 1 and segment.isalpha():
            # Handle letter parts like "A" -> treat as position offset
            parts.append(ord(segment.upper()) - ord("A") + 100)
    return tuple(parts)


def extract_paragraph_texts(paragraphs: list[Span], body_text: str) -> list[str]:
    """Extract non-empty paragraph text from spans.

    Args:
        paragraphs: List of paragraph spans.
        body_text: The full body text.

    Returns:
        List of non-empty paragraph strings.
    """
    return [text for p in paragraphs if (text := body_text[p.start : p.end].strip())]


def parse_section_headers(headers: list[Span], body_text: str) -> list[SectionRange]:
    """Parse header spans into SectionRange objects with computed end offsets.

    Args:
        headers: List of section header spans.
        body_text: The full body text.

    Returns:
        List of SectionRange objects sorted by start position.
    """
    section_info: list[_SectionInfo] = []
    for h in headers:
        section_num = parse_section_number(h.attributes.n if h.attributes else None)
        heading_text = body_text[h.start : h.end].strip()
        section_info.append(
            _SectionInfo(number=section_num, start=h.start, heading=heading_text)
        )

    # Sort by start offset
    section_info.sort(key=lambda x: x.start)

    # Compute end offsets (start of next section or end of document)
    ranges: list[SectionRange] = []
    for i, info in enumerate(section_info):
        if i + 1 < len(section_info):
            end = section_info[i + 1].start
        else:
            end = len(body_text)
        ranges.append(
            SectionRange(
                number=info.number, start=info.start, end=end, heading=info.heading
            )
        )

    return ranges


def assign_paragraphs_to_sections(
    paragraphs: list[Span],
    section_ranges: list[SectionRange],
    body_text: str,
) -> dict[tuple[int, ...], list[str]]:
    """Assign paragraph texts to their containing sections.

    Args:
        paragraphs: List of paragraph spans.
        section_ranges: List of section ranges.
        body_text: The full body text.

    Returns:
        Dict mapping section number tuples to lists of paragraph texts.
    """
    # section_no -> [paragraph_text, ...]
    section_paragraphs: dict[tuple[int, ...], list[str]] = {
        sr.number: [] for sr in section_ranges
    }

    for p in paragraphs:
        para_text = body_text[p.start : p.end].strip()
        if not para_text:
            continue

        for sr in section_ranges:
            if p.start in range(sr.start, sr.end):
                section_paragraphs[sr.number].append(para_text)
                break

    return section_paragraphs


def is_direct_child(child_num: tuple[int, ...], parent_num: tuple[int, ...]) -> bool:
    """Check if child_num is a direct child of parent_num in the hierarchy.

    Args:
        child_num: The potential child section number.
        parent_num: The parent section number.

    Returns:
        True if child_num is exactly one level deeper and shares parent's prefix.
    """
    if len(child_num) != len(parent_num) + 1:
        return False
    if not parent_num:  # Empty parent is the root section
        return True
    return child_num[: len(parent_num)] == parent_num


def build_section_subtree(
    parent_num: tuple[int, ...],
    section_ranges: list[SectionRange],
    section_paragraphs: dict[tuple[int, ...], list[str]],
) -> list[Section]:
    """Build Section objects that are direct children of parent_num.

    Args:
        parent_num: The parent section number (empty tuple for top-level).
        section_ranges: All section ranges in the document.
        section_paragraphs: Mapping of section numbers to paragraph texts.

    Returns:
        List of Section objects that are direct children of parent_num.
    """
    result: list[Section] = []

    for sr in section_ranges:
        if not is_direct_child(sr.number, parent_num):
            continue

        subsections = build_section_subtree(
            sr.number, section_ranges, section_paragraphs
        )

        section = Section(
            heading=sr.heading,
            type=classify_section(sr.heading),
            paragraphs=section_paragraphs.get(sr.number, []),
            subsections=subsections,
        )
        result.append(section)

    return result


def collect_orphan_sections(
    section_ranges: list[SectionRange],
    section_paragraphs: dict[tuple[int, ...], list[str]],
    existing_headings: set[str],
) -> list[Section]:
    """Collect sections without proper numbering that weren't included in the tree.

    Args:
        section_ranges: All section ranges.
        section_paragraphs: Mapping of section numbers to paragraph texts.
        existing_headings: Set of headings already in the tree.

    Returns:
        List of orphan Section objects.
    """
    orphans: list[Section] = []

    for sr in section_ranges:
        is_orphan = len(sr.number) == 0 or len(sr.number) == 1
        if is_orphan and sr.heading not in existing_headings:
            orphans.append(
                Section(
                    heading=sr.heading,
                    type=classify_section(sr.heading),
                    paragraphs=section_paragraphs.get(sr.number, []),
                    subsections=[],
                )
            )
            existing_headings.add(sr.heading)

    return orphans


def build_section_tree(
    headers: list[Span], paragraphs: list[Span], body_text: str
) -> list[Section]:
    """Build a recursive section tree from flat header and paragraph spans.

    Args:
        headers: List of section header spans with 'n' attribute for section numbers.
        paragraphs: List of paragraph spans.
        body_text: The full body text to extract content from.

    Returns:
        List of top-level Section objects with nested subsections.
    """
    if not headers:
        if all_paras := extract_paragraph_texts(paragraphs, body_text):
            return [
                Section(
                    heading="",
                    type=SectionType.OTHER,
                    paragraphs=all_paras,
                    subsections=[],
                )
            ]
        return []

    section_ranges = parse_section_headers(headers, body_text)
    section_paragraphs = assign_paragraphs_to_sections(
        paragraphs, section_ranges, body_text
    )

    # Build tree from top level
    top_level = build_section_subtree((), section_ranges, section_paragraphs)

    # Add orphan sections
    existing_headings = {s.heading for s in top_level}
    orphans = collect_orphan_sections(
        section_ranges, section_paragraphs, existing_headings
    )
    top_level.extend(orphans)

    return top_level


def parse_sections_from_annotations(
    annotations: BodyAnnotations, body_text: str
) -> list[Section]:
    """Parse sections from annotation data.

    Args:
        annotations: The annotations with section_header and paragraph fields.
        body_text: The full body text.

    Returns:
        List of parsed Section objects.
    """
    header_json = annotations.section_header or ""
    para_json = annotations.paragraph or ""

    headers = parse_annotations(header_json)
    paragraphs = parse_annotations(para_json)

    return build_section_tree(headers, paragraphs, body_text)


def parse_paper(
    record: dict[str, Any], abstract: str | None = None
) -> PaperGraph | None:
    """Parse an S2ORC record into a PaperGraph.

    Args:
        record: S2ORC JSON record with corpusid, title, authors, body.
        abstract: Optional abstract text from S2 metadata to attach to the graph.

    Returns:
        PaperGraph if parsing succeeds, None if the record lacks required fields.
    """
    try:
        s2orc = S2ORCRecord.model_validate(record)
    except ValidationError:
        return None

    body = s2orc.body
    if body is None or not body.text:
        return None

    if body.annotations is not None:
        sections = parse_sections_from_annotations(body.annotations, body.text)
    elif text := body.text.strip():
        sections = [
            Section(
                heading="",
                type=SectionType.OTHER,
                paragraphs=[text],
                subsections=[],
            )
        ]
    else:
        sections: list[Section] = []

    # If no abstract provided, try to extract from an ABSTRACT-typed section
    if abstract is None:
        for section in sections:
            if section.type == SectionType.ABSTRACT and section.paragraphs:
                abstract = " ".join(section.paragraphs)
                break

    return PaperGraph(
        corpusid=s2orc.corpusid,
        title=s2orc.title or "",
        authors=list(s2orc.authors) if s2orc.authors else [],
        sections=sections,
        abstract=abstract,
    )


_logger = logging.getLogger(__name__)


def load_facets_map(facets_path: Path) -> dict[CorpusId, Facets]:
    """Load all facets into a corpusid-keyed dict.

    Args:
        facets_path: Path to facets JSONL file (plain or compressed).

    Returns:
        Mapping from corpus ID to `Facets`.
    """
    resolved = resolve_jsonl(facets_path)
    facets_map: dict[CorpusId, Facets] = {}

    for record in track_progress(iter_jsonl(resolved), desc="Loading facets..."):
        f = Facets.model_validate(record)
        facets_map[f.corpusid] = f

    _logger.info(f"Loaded {len(facets_map)} facets")
    return facets_map
