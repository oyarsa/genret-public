"""Local paper title search over in-graph and in-corpus papers."""

import re
from dataclasses import dataclass

from genret.explorer.models import SearchPaperResult


@dataclass(frozen=True)
class SearchablePaper:
    """A paper indexed for local title search."""

    corpusid: int
    title: str
    title_normalized: str
    authors: list[str]
    year: int | None
    venue: str | None
    source: str  # "in_graph" or "in_corpus"


_ALPHA_ONLY = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    """Lowercase alpha-numeric only, for substring matching."""
    return _ALPHA_ONLY.sub("", title.lower())


def search_papers(
    query: str, papers: list[SearchablePaper], limit: int = 20
) -> list[SearchPaperResult]:
    """Search papers by title substring match, ranked by match quality.

    Matches rank by where the query appears (prefix matches first) and
    then by title length (tighter matches first), regardless of source.
    The previous in-graph-first ordering with early truncation made
    in-corpus papers unreachable for any query that many of the 11k graph
    titles also matched.

    Args:
        query: Search query string.
        papers: List of searchable papers.
        limit: Maximum number of results.

    Returns:
        Best-matching papers.
    """
    q = normalize_title(query)
    if not q:
        return []

    scored: list[tuple[int, int, SearchablePaper]] = []
    for p in papers:
        pos = p.title_normalized.find(q)
        if pos == -1:
            continue
        scored.append((pos, len(p.title_normalized), p))

    scored.sort(key=lambda t: (t[0], t[1]))

    return [
        SearchPaperResult(
            corpusid=p.corpusid,
            title=p.title,
            authors=p.authors,
            year=p.year,
            venue=p.venue,
            source=p.source,
        )
        for _, _, p in scored[:limit]
    ]
