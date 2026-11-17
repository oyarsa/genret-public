"""Shared configuration and types for conference paper counting."""

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

from genret.types import PaperId

logger = logging.getLogger(__name__)


class Category(StrEnum):
    """Conference category for grouping and filtering in plots."""

    NLP = "NLP"
    ML = "ML/AI"
    CV = "Computer Vision"
    IR = "Information Retrieval"
    DATA = "Data Mining"
    OTHER = "Other"


class PdfSource(StrEnum):
    """Origin of a paper's PDF URL."""

    S2 = "S2"
    ARXIV = "ArXiv"
    ACL = "ACL"


@dataclass(frozen=True, kw_only=True)
class Paper:
    """Represents a paper with relevant metadata."""

    paper_id: PaperId
    title: str
    year: int | None
    venue: str | None
    citation_count: int | None
    fields_of_study: list[str]
    url: str
    open_access_pdf: str | None
    pdf_source: PdfSource | None = None


@dataclass
class ConferencePattern:
    """Pattern for identifying a conference across different naming conventions."""

    name: str  # Canonical name for grouping
    venue_patterns: list[str]  # Venue names to match
    category: Category = Category.OTHER


CONFERENCE_PATTERNS = [
    # *ACL / NLP Conferences
    ConferencePattern(
        name="ACL",
        venue_patterns=[
            "Annual Meeting of the Association for Computational Linguistics",
            "ACL",
        ],
        category=Category.NLP,
    ),
    ConferencePattern(
        name="EMNLP",
        venue_patterns=[
            "Conference on Empirical Methods in Natural Language Processing",
            "Empirical Methods in Natural Language Processing",
            "EMNLP",
        ],
        category=Category.NLP,
    ),
    ConferencePattern(
        name="NAACL",
        venue_patterns=[
            "North American Chapter of the Association for Computational Linguistics",
            "NAACL",
            "NAACL-HLT",
        ],
        category=Category.NLP,
    ),
    ConferencePattern(
        name="EACL",
        venue_patterns=[
            "Conference of the European Chapter of the Association for Computational Linguistics",
            "European Chapter of the Association for Computational Linguistics",
            "EACL",
        ],
        category=Category.NLP,
    ),
    ConferencePattern(
        name="AACL-IJCNLP",
        venue_patterns=[
            "AACL",
            "International Joint Conference on Natural Language Processing",
            "IJCNLP",
            "AACL-IJCNLP",
        ],
        category=Category.NLP,
    ),
    ConferencePattern(
        name="COLING",
        venue_patterns=[
            "International Conference on Computational Linguistics",
            "COLING",
        ],
        category=Category.NLP,
    ),
    ConferencePattern(
        name="CoNLL",
        venue_patterns=[
            "Conference on Computational Natural Language Learning",
            "CoNLL",
        ],
        category=Category.NLP,
    ),
    ConferencePattern(
        name="SEM",
        venue_patterns=[
            "Joint Conference on Lexical and Computational Semantics",
            "SEM",
            "StarSEM",
        ],
        category=Category.NLP,
    ),
    ConferencePattern(
        name="TACL",
        venue_patterns=[
            "Transactions of the Association for Computational Linguistics",
            "TACL",
        ],
        category=Category.NLP,
    ),
    ConferencePattern(
        name="LREC",
        venue_patterns=[
            "Language Resources and Evaluation Conference",
            "International Conference on Language Resources and Evaluation",
            "LREC",
        ],
        category=Category.NLP,
    ),
    ConferencePattern(
        name="SemEval",
        venue_patterns=[
            "International Workshop on Semantic Evaluation",
            "SemEval",
        ],
        category=Category.NLP,
    ),
    ConferencePattern(
        name="SIGDIAL",
        venue_patterns=[
            "SIGDIAL Conferences",
            "SIGDIAL",
        ],
        category=Category.NLP,
    ),
    # Major ML/AI Conferences
    ConferencePattern(
        name="NeurIPS",
        venue_patterns=[
            "Neural Information Processing Systems",
            "NeurIPS",
            "NIPS",
        ],
        category=Category.ML,
    ),
    ConferencePattern(
        name="ICML",
        venue_patterns=[
            "International Conference on Machine Learning",
            "ICML",
        ],
        category=Category.ML,
    ),
    ConferencePattern(
        name="ICLR",
        venue_patterns=[
            "International Conference on Learning Representations",
            "ICLR",
        ],
        category=Category.ML,
    ),
    ConferencePattern(
        name="AAAI",
        venue_patterns=[
            "AAAI Conference on Artificial Intelligence",
            "AAAI",
        ],
        category=Category.ML,
    ),
    ConferencePattern(
        name="IJCAI",
        venue_patterns=[
            "International Joint Conference on Artificial Intelligence",
            "IJCAI",
        ],
        category=Category.ML,
    ),
    ConferencePattern(
        name="AISTATS",
        venue_patterns=[
            "International Conference on Artificial Intelligence and Statistics",
            "AISTATS",
        ],
        category=Category.ML,
    ),
    ConferencePattern(
        name="UAI",
        venue_patterns=[
            "Conference on Uncertainty in Artificial Intelligence",
            "UAI",
        ],
        category=Category.ML,
    ),
    ConferencePattern(
        name="COLT",
        venue_patterns=[
            "Conference on Learning Theory",
            "Annual Conference Computational Learning Theory",
            "COLT",
        ],
        category=Category.ML,
    ),
    # Computer Vision
    ConferencePattern(
        name="CVPR",
        venue_patterns=[
            "Computer Vision and Pattern Recognition",
            "CVPR",
        ],
        category=Category.CV,
    ),
    ConferencePattern(
        name="ICCV",
        venue_patterns=[
            "International Conference on Computer Vision",
            "IEEE International Conference on Computer Vision",
            "ICCV",
        ],
        category=Category.CV,
    ),
    ConferencePattern(
        name="ECCV",
        venue_patterns=[
            "European Conference on Computer Vision",
            "ECCV",
        ],
        category=Category.CV,
    ),
    # Information Retrieval
    ConferencePattern(
        name="SIGIR",
        venue_patterns=[
            "Annual International ACM SIGIR Conference on Research and Development in Information Retrieval",
            "SIGIR",
        ],
        category=Category.IR,
    ),
    ConferencePattern(
        name="WSDM",
        venue_patterns=[
            "Web Search and Data Mining",
            "WSDM",
        ],
        category=Category.IR,
    ),
    ConferencePattern(
        name="CIKM",
        venue_patterns=[
            "Conference on Information and Knowledge Management",
            "International Conference on Information and Knowledge Management",
            "CIKM",
        ],
        category=Category.IR,
    ),
    ConferencePattern(
        name="RecSys",
        venue_patterns=[
            "ACM Conference on Recommender Systems",
            "RecSys",
        ],
        category=Category.IR,
    ),
    # Data Mining / Web
    ConferencePattern(
        name="KDD",
        venue_patterns=[
            "Knowledge Discovery and Data Mining",
            "KDD",
            "ACM SIGKDD",
        ],
        category=Category.DATA,
    ),
    ConferencePattern(
        name="WWW",
        venue_patterns=[
            "The Web Conference",
            "WWW",
            "World Wide Web Conference",
        ],
        category=Category.DATA,
    ),
    ConferencePattern(
        name="VLDB",
        venue_patterns=[
            "Very Large Data Bases",
            "Very Large Data Bases Conference",
            "VLDB",
            "Proceedings of the VLDB Endowment",
        ],
        category=Category.DATA,
    ),
]
"""Conference definitions with name variations and category labels."""


def get_venue_to_conference_map() -> dict[str, str]:
    """Build a mapping from venue patterns to canonical conference names."""
    mapping: dict[str, str] = {}
    for conf in CONFERENCE_PATTERNS:
        for venue in conf.venue_patterns:
            mapping[venue] = conf.name
    return mapping


_venue_alias_map: dict[str, str] | None = None
"""Lazily-built mapping from every venue pattern to its canonical conference name."""

_FINDINGS_CONF_RE = re.compile(r"findings-(\w+)", re.IGNORECASE)
"""Regex to extract conference name from ACL anthology IDs."""

_FINDINGS_CONFERENCES: dict[str, str] = {
    "acl": "ACL",
    "emnlp": "EMNLP",
    "naacl": "NAACL",
    "eacl": "EACL",
    "aacl": "AACL-IJCNLP",
}
"""Mapping from ACL ID conference fragment to canonical conference name."""


def _get_venue_alias_map() -> dict[str, str]:
    """Return the cached venue alias map, building it on first call."""
    global _venue_alias_map  # noqa: PLW0603
    if _venue_alias_map is None:
        _venue_alias_map = get_venue_to_conference_map()
    return _venue_alias_map


def normalise_venue(venue: str, acl_id: str | None = None) -> str:
    """Normalise a raw venue string to its canonical form.

    Handles three cases:

    1. ``"Findings"`` — resolved via the ACL anthology ID to e.g.
       ``"Findings of EMNLP"``, or ``"Findings (unknown)"`` if no ID.
    2. Known alias — mapped to the canonical short conference name
       (e.g. ``"NAACL-HLT"`` → ``"NAACL"``).
    3. Everything else — returned unchanged.

    Args:
        venue: Raw venue string from Semantic Scholar.
        acl_id: Optional ACL external ID (e.g. ``"2020.findings-emnlp.82"``).

    Returns:
        Normalised venue string.
    """
    if venue != "Findings":
        return _get_venue_alias_map().get(venue, venue)

    if (
        acl_id
        and (match := _FINDINGS_CONF_RE.search(acl_id))
        and (conf := _FINDINGS_CONFERENCES.get(match[1].lower()))
    ):
        return f"Findings of {conf}"
    return "Findings (unknown)"


def select_conferences(
    conference_names: list[str] | None,
) -> tuple[list[ConferencePattern], list[str]]:
    """Select conference patterns by user-provided names.

    Args:
        conference_names: Optional user-provided conference names.

    Returns:
        Tuple of:
            - Selected conference patterns (all if `conference_names` is None)
            - Unknown conference names from input
    """
    if conference_names is None:
        return list(CONFERENCE_PATTERNS), []

    conference_map = {c.name.upper(): c for c in CONFERENCE_PATTERNS}
    selected: list[ConferencePattern] = []
    unknown: list[str] = []

    for conf_name in conference_names:
        conf = conference_map.get(conf_name.upper())
        if conf is None:
            unknown.append(conf_name)
        else:
            selected.append(conf)

    return selected, unknown


def require_conferences(
    conference_names: list[str] | None,
) -> list[ConferencePattern]:
    """Select conferences by name, warning for unknowns and raising if none are valid.

    Args:
        conference_names: Optional user-provided conference names.
            If None, returns all conferences.

    Returns:
        List of selected conference patterns.

    Raises:
        ValueError: If explicit names were given but none matched.
    """
    selected, unknown = select_conferences(conference_names)
    for name in unknown:
        logger.warning(f"Unknown conference '{name}'")
    if conference_names is not None and not selected:
        raise ValueError("No valid conferences specified")
    return selected
