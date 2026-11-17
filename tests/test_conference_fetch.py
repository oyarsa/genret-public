"""Unit tests for conference_fetch.py."""

import asyncio

from genret.s2.conference_fetch import fetch_papers_by_venues
from genret.s2.types import S2BulkSearchResponse, S2Paper


class MockS2API:
    """Deterministic fake API for pagination tests."""

    def __init__(self, responses: list[S2BulkSearchResponse | Exception]) -> None:
        """Initialise with a queue of responses to return in order."""
        self._responses = responses
        self.calls: list[dict[str, str | list[str] | None]] = []
        self._index = 0

    async def search_papers_bulk(
        self,
        venue: str,
        year: str | None = None,
        token: str | None = None,
        fields: list[str] | None = None,
    ) -> S2BulkSearchResponse:
        """Return the next queued response or raise the next queued exception."""
        self.calls.append(
            {
                "venue": venue,
                "year": year,
                "token": token,
                "fields": fields,
            }
        )
        response = self._responses[self._index]
        self._index += 1
        if isinstance(response, Exception):
            raise response
        return response


def test_fetch_papers_success_with_pagination() -> None:
    """Should parse all pages and mark result as non-partial."""
    fake_api = MockS2API(
        [
            S2BulkSearchResponse(
                data=[S2Paper(paper_id="1"), S2Paper(paper_id="2")],
                token="cursor_1",  # noqa: S106
            ),
            S2BulkSearchResponse(
                data=[S2Paper(paper_id="3")],
            ),
        ]
    )
    page_sizes: list[int] = []

    result = asyncio.run(
        fetch_papers_by_venues(
            fake_api,
            venues=["ACL", "EMNLP"],
            parser=lambda paper: paper.paper_id,
            year_range=(2020, 2025),
            fields=["paperId"],
            on_page_count=page_sizes.append,
        )
    )

    assert result.papers == ["1", "2", "3"]
    assert result.had_error is False
    assert page_sizes == [2, 1]
    assert fake_api.calls[0]["venue"] == "ACL,EMNLP"
    assert fake_api.calls[0]["year"] == "2020-2025"
    assert fake_api.calls[0]["fields"] == ["paperId"]
    assert fake_api.calls[1]["token"] == "cursor_1"  # noqa: S105


def test_fetch_papers_sets_partial_on_error_and_keeps_data() -> None:
    """Should report partial fetch and retain papers from successful pages."""
    fake_api = MockS2API(
        [
            S2BulkSearchResponse(
                data=[S2Paper(paper_id="10"), S2Paper(paper_id="20")],
                token="cursor_1",  # noqa: S106
            ),
            RuntimeError("boom"),
        ]
    )
    errors: list[tuple[str, str]] = []

    def on_error(venue_query: str, exc: Exception) -> None:
        """Collect errors for assertion."""
        errors.append((venue_query, str(exc)))

    result = asyncio.run(
        fetch_papers_by_venues(
            fake_api,
            venues=["NAACL"],
            parser=lambda paper: paper.paper_id,
            on_error=on_error,
        )
    )

    assert result.papers == ["10", "20"]
    assert result.had_error is True
    assert errors == [("NAACL", "boom")]
