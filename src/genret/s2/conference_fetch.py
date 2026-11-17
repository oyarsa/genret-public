"""Shared helpers for fetching conference papers via Semantic Scholar bulk API."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from genret.s2.types import S2BulkSearchResponse, S2Paper

PageCallback = Callable[[int], None]
ErrorCallback = Callable[[str, Exception], None]


class SearchBulkAPI(Protocol):
    """Structural protocol for API clients exposing search_papers_bulk()."""

    async def search_papers_bulk(
        self,
        venue: str,
        year: str | None = None,
        token: str | None = None,
        fields: list[str] | None = None,
    ) -> S2BulkSearchResponse:
        """Fetch one page of bulk search results.

        Args:
            venue: Venue name(s) to search, comma-separated for multiple.
            year: Optional year or range string (e.g. ``"2020"`` or ``"2019-2023"``).
            token: Pagination token from a previous response, or None for the first page.
            fields: Paper fields to include in the response.

        Returns:
            One page of bulk search results including an optional pagination token.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class FetchPapersResult[ParsedPaperT]:
    """Result of fetching and parsing papers for a venue query.

    Attributes:
        papers: All parsed papers collected across all pages of the query.
        had_error: True if at least one API request failed during fetching.
    """

    papers: list[ParsedPaperT]
    had_error: bool = False


async def fetch_papers_by_venues[ParsedPaperT](
    api: SearchBulkAPI,
    venues: list[str],
    parser: Callable[[S2Paper], ParsedPaperT],
    *,
    year_range: tuple[int, int] | None = None,
    fields: list[str] | None = None,
    on_page_count: PageCallback | None = None,
    on_error: ErrorCallback | None = None,
) -> FetchPapersResult[ParsedPaperT]:
    """Fetch and parse all papers for a set of venue aliases.

    Paginates through all results from the bulk search API, applying the provided parser
    to each raw paper record. If any page request fails, the error callback is invoked
    and pagination stops early.

    Args:
        api: Client implementing the SearchBulkAPI protocol.
        venues: Venue names or aliases to search (joined as a single query).
        parser: Callable to convert a raw S2Paper into the desired output type.
        year_range: Optional inclusive (start_year, end_year) publication year filter.
        fields: Optional list of S2 paper fields to request from the API.
        on_page_count: Optional callback invoked with the count of papers for each page.
        on_error: Optional callback invoked with (venue_query, exception) on error.

    Returns:
        FetchPapersResult containing all parsed papers and an error flag.
    """
    papers: list[ParsedPaperT] = []
    token: str | None = None
    had_error = False

    year_query: str | None = None
    if year_range:
        start, end = year_range
        year_query = f"{start}-{end}"

    venue_query = ",".join(venues)

    while True:
        try:
            result = await api.search_papers_bulk(
                venue=venue_query,
                year=year_query,
                token=token,
                fields=fields,
            )
            papers.extend(parser(paper) for paper in result.data)

            if on_page_count:
                on_page_count(len(result.data))

            token = result.token
            if not token:
                break
        except Exception as exc:
            had_error = True
            if on_error:
                on_error(venue_query, exc)
            break

    return FetchPapersResult(papers=papers, had_error=had_error)
