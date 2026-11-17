"""Prompt rendering helpers for per-bullet GLEN training examples."""

from __future__ import annotations

import re

from genret.paper_parsing import FacetEvidence, Facets

_SYSTEM_PROMPT = (
    "You are a generative retrieval system. Given a research paper's "
    "{facet_type} bullet, generate the {facet_type} description of a relevant paper."
)


def _build_bullet_messages(
    facet: str,
    title: str,
    bullet_text: str,
) -> list[dict[str, str]]:
    """Build chat messages for a single-bullet D9 training example."""
    system = _SYSTEM_PROMPT.format(facet_type=facet.capitalize())
    if title:
        user = f"Title: {title}\n{facet.capitalize()}:\n{bullet_text}"
    else:
        user = f"{facet.capitalize()}:\n{bullet_text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _render_bullet(
    facets: Facets, facet: str, bullet_idx: int, evidence: FacetEvidence | None
) -> str:
    """Render one bullet with its evidence quotes as a string."""
    items = getattr(facets, facet)
    if bullet_idx >= len(items):
        return ""
    item = str(items[bullet_idx]).strip()
    if not item:
        return ""
    lines = [f"- {item}"]
    if evidence:
        item_evidence: list[list[str]] = evidence.get(facet, [])
        if bullet_idx < len(item_evidence):
            for quote in item_evidence[bullet_idx]:
                lines.append(f'  > "{quote}"')
    return "\n".join(lines)


def _word_count(text: str) -> int:
    """Approximate word count via alphanumeric token split.

    Used to pick the "longest" bullet for ablations: replace non-alphanumeric
    characters with spaces, then split on whitespace.
    """
    return len(re.sub(r"[^A-Za-z0-9]", " ", text).split())
