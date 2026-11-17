"""Title loading and paper_a / paper_b bullet selection helpers."""

from __future__ import annotations

from pathlib import Path

from genret.paper_parsing import Facets
from genret.training.codebook_dataset import _facet_text
from genret.training.glen_docids import normalise_facet_text, split_bullets
from genret.types import CorpusId
from genret.utils import iter_jsonl

from .rendering import _word_count


def _load_titles(paper_graphs_path: Path) -> dict[CorpusId, str]:
    """Return {corpusid: title} for all papers with a non-empty title."""
    titles: dict[CorpusId, str] = {}
    for record in iter_jsonl(paper_graphs_path):
        cid = CorpusId(int(record["corpusid"]))
        title = record.get("title")
        if isinstance(title, str) and title.strip():
            titles[cid] = title.strip()
    return titles


def _longest_target(facets: Facets, facet: str) -> str:
    """Return the longest bullet of ``facet`` (by word count) as the target."""
    raw = _facet_text(facets, facet, None)
    bullets = split_bullets(raw)
    if not bullets:
        return ""
    longest = max(bullets, key=_word_count)
    return normalise_facet_text(longest)


def _all_target_bullets(facets: Facets, facet: str) -> list[str]:
    """All of paper_b's bullets in ``facet`` as normalised target strings.

    Empty bullets are dropped. Used by ``target_bullet="cartesian"`` to emit
    one training example per (paper_a bullet, paper_b bullet) pair.
    """
    raw = _facet_text(facets, facet, None)
    bullets = split_bullets(raw)
    out: list[str] = []
    for b in bullets:
        norm = normalise_facet_text(b)
        if norm:
            out.append(norm)
    return out


def _longest_source_index(facets: Facets, facet: str) -> int | None:
    """Return the index of paper_a's longest bullet for ``facet``.

    Ties broken by first occurrence. Returns ``None`` when no bullets exist.
    """
    items = getattr(facets, facet)
    if not items:
        return None
    return max(range(len(items)), key=lambda i: _word_count(str(items[i])))
