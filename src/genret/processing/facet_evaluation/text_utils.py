"""Text helpers: paragraph extraction, sentence counting, clipping, normalisation."""

import re

from genret.paper_parsing import PaperGraph, Section

from .constants import EVIDENCE_CLIP_CHARS


def extract_paragraphs(paper: PaperGraph) -> list[tuple[str, str]]:
    """Extract all (section_heading, paragraph_text) pairs from a paper.

    Recursively traverses the section tree.
    """
    results: list[tuple[str, str]] = []

    def _walk(section: Section) -> None:
        for para in section.paragraphs:
            text = para.strip()
            if text:
                results.append((section.heading, text))
        for sub in section.subsections:
            _walk(sub)

    for section in paper.sections:
        _walk(section)
    return results


def count_sentences(text: str) -> int:
    """Count sentences using a simple heuristic."""
    parts = re.split(r"[.!?]+(?:\s|$)", text.strip())
    return len([s for s in parts if s.strip()])


def _clip_text(text: str, max_chars: int = EVIDENCE_CLIP_CHARS) -> str:
    """Truncate text to max_chars on a word boundary, appending ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def normalise_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip.

    Used before substring/similarity checks so minor formatting differences
    (newlines, double spaces) don't cause false negatives.
    """
    return re.sub(r"\s+", " ", text).strip()
