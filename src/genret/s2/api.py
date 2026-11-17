"""Semantic Scholar API client and shared utilities."""

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any, Self

import httpx

from genret.config import PdfSource
from genret.s2.types import (
    S2BulkSearchResponse,
    S2Paper,
    S2PaperRef,
    S2RelatedPaperItem,
    S2RelatedPapersResponse,
    S2SearchResponse,
)
from genret.utils import RateLimiter

logger = logging.getLogger(__name__)


def extract_pdf_url(paper: S2Paper) -> tuple[str | None, PdfSource | None]:
    """Extract PDF URL from an S2Paper with fallbacks.

    Tries in order:
    1. Semantic Scholar openAccessPdf
    2. ArXiv via externalIds
    3. ACL Anthology via DOI

    Args:
        paper: Paper data from Semantic Scholar API.

    Returns:
        Tuple of (pdf_url, pdf_source).
    """
    # Try Semantic Scholar's openAccessPdf first
    pdf_url = paper.open_access_pdf.url if paper.open_access_pdf else None
    if pdf_url == "":
        pdf_url = None

    pdf_source: PdfSource | None = None
    if pdf_url:
        pdf_source = PdfSource.S2
    else:
        # Fall back to ArXiv or ACL Anthology
        external_ids = paper.external_ids
        if external_ids:
            if arxiv_id := external_ids.arxiv:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                pdf_source = PdfSource.ARXIV
            elif (doi := external_ids.doi) and doi.startswith("10.18653/v1/"):
                # ACL Anthology DOIs: 10.18653/v1/2024.acl-long.123
                acl_id = doi.removeprefix("10.18653/v1/")
                pdf_url = f"https://aclanthology.org/{acl_id}.pdf"
                pdf_source = PdfSource.ACL

    return pdf_url, pdf_source


DEFAULT_S2_SEARCH_FIELDS = (
    "paperId",
    "corpusId",
    "title",
    "abstract",
    "year",
    "venue",
    "citationCount",
    "url",
    "openAccessPdf",
    "externalIds",
    "authors",
)


class SemanticScholarAPI:
    """Async client for Semantic Scholar API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    DEFAULT_RATE_LIMIT = 1.0  # Seconds between requests (with API key)

    def __init__(
        self,
        api_key: str,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        timeout: float = 60.0,
    ) -> None:
        """Initialise the API client.

        Args:
            api_key: Semantic Scholar API key
            rate_limit: Minimum seconds between requests
            timeout: HTTP request timeout in seconds
        """
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.headers = {"x-api-key": api_key}
        self.client: httpx.AsyncClient | None = None
        self._rate_limiter = RateLimiter(1.0 / rate_limit) if rate_limit > 0 else None

    async def __aenter__(self) -> Self:
        """Initialise the HTTP client."""
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Close the HTTP client."""
        await self.stop()

    async def start(self) -> None:
        """Start the client."""
        self.client = httpx.AsyncClient(timeout=self.timeout, headers=self.headers)

    async def stop(self) -> None:
        """Stop the client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def _rate_limit_wait(self) -> None:
        """Ensure minimum time between requests."""
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        params: dict[str, str] | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Make a request with retry logic for rate limits.

        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            max_retries: Maximum retry attempts for rate limiting

        Returns:
            JSON response as dictionary

        Raises:
            RuntimeError: If client not initialised or max retries exceeded
        """
        if self.client is None:
            raise RuntimeError("API client not initialised. Use 'async with' context.")

        for attempt in range(max_retries):
            await self._rate_limit_wait()
            try:
                response = await self.client.request(method, url, params=params)
                if response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise
        raise RuntimeError(f"Max retries exceeded for {url}")

    async def _post_json_list(
        self,
        url: str,
        body: dict[str, Any],
        params: dict[str, str] | None = None,
        max_retries: int = 3,
    ) -> list[Any]:
        """POST a JSON body and return a JSON array response, with retry logic.

        Args:
            url: Request URL.
            body: JSON request body.
            params: Query parameters.
            max_retries: Maximum retry attempts for rate limiting.

        Returns:
            JSON response as a list.

        Raises:
            RuntimeError: If client not initialised or max retries exceeded.
        """
        if self.client is None:
            raise RuntimeError("API client not initialised. Use 'async with' context.")

        for attempt in range(max_retries):
            await self._rate_limit_wait()
            try:
                response = await self.client.post(url, params=params, json=body)
                if response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                raise
        raise RuntimeError(f"Max retries exceeded for {url}")

    async def search_papers_bulk(
        self,
        venue: str,
        year: str | None = None,
        token: str | None = None,
        fields: list[str] | None = None,
    ) -> S2BulkSearchResponse:
        """Search for papers using the bulk API.

        Args:
            venue: Venue name(s) to filter by (comma-separated)
            year: Year or year range (e.g., "2024" or "2020-2025")
            token: Continuation token for pagination
            fields: List of fields to return

        Returns:
            Parsed bulk search response with papers, total, and optional token.
        """
        if fields is None:
            fields = [
                "paperId",
                "title",
                "abstract",
                "year",
                "venue",
                "citationCount",
                "fieldsOfStudy",
                "url",
                "openAccessPdf",
                "externalIds",
            ]

        params = {
            "venue": venue,
            "fields": ",".join(fields),
        }

        if year:
            params["year"] = year
        if token:
            params["token"] = token

        raw = await self._request_with_retry(
            "GET", f"{self.BASE_URL}/paper/search/bulk", params=params
        )
        return S2BulkSearchResponse.model_validate(raw)

    async def search_by_title(
        self,
        query: str,
        limit: int = 20,
        fields: Sequence[str] = DEFAULT_S2_SEARCH_FIELDS,
    ) -> list[S2Paper]:
        """Search for papers by title using the relevance search endpoint.

        Uses ``GET /paper/search?query=...`` which ranks results by relevance.
        Only papers with a resolvable PDF URL are returned.

        Args:
            query: Search query (title or keywords).
            limit: Maximum number of results to return.
            fields: Fields to include in the response. Defaults to
                `DEFAULT_S2_SEARCH_FIELDS`

        Returns:
            List of matching papers that have a resolvable PDF URL.
        """
        # The S2 relevance endpoint caps each page at 100 results.
        page_size = min(limit, 100)
        results: list[S2Paper] = []
        offset = 0

        while len(results) < limit:
            params = {
                "query": query,
                "limit": str(page_size),
                "offset": str(offset),
                "fields": ",".join(fields),
            }

            raw = await self._request_with_retry(
                "GET", f"{self.BASE_URL}/paper/search", params=params
            )
            response = S2SearchResponse.model_validate(raw)

            for paper in response.data:
                pdf_url, _ = extract_pdf_url(paper)
                if pdf_url:
                    results.append(paper)
                    if len(results) >= limit:
                        break

            # No more pages available
            if response.next is None or not response.data:
                break

            offset = response.next

        return results

    async def _fetch_related_papers(
        self,
        paper_id: str,
        endpoint: str,
        get_ref: Callable[[S2RelatedPaperItem], S2PaperRef | None],
        limit: int = 1000,
    ) -> list[str]:
        """Fetch paginated related-paper IDs from a Semantic Scholar endpoint.

        Args:
            paper_id: Semantic Scholar paper ID.
            endpoint: API endpoint suffix (e.g. "references" or "citations").
            get_ref: Extracts the nested paper reference from each result item.
            limit: Maximum number of related papers to fetch.

        Returns:
            List of related paper IDs.
        """
        paper_ids: list[str] = []
        offset = 0

        while True:
            params = {
                "fields": "paperId",
                "limit": str(min(limit, 1000)),
                "offset": str(offset),
            }
            try:
                raw = await self._request_with_retry(
                    "GET",
                    f"{self.BASE_URL}/paper/{paper_id}/{endpoint}",
                    params=params,
                )
                result = S2RelatedPapersResponse.model_validate(raw)
                for item in result.data:
                    ref = get_ref(item)
                    if ref and ref.paper_id:
                        paper_ids.append(ref.paper_id)

                if len(result.data) < 1000:
                    break
                offset += len(result.data)
                if offset >= limit:
                    break
            except Exception as e:
                logger.warning(
                    f"Failed to fetch {endpoint} for {paper_id} at offset {offset}: {e}"
                )
                break

        return paper_ids

    async def get_paper_references(
        self,
        paper_id: str,
        limit: int = 1000,
    ) -> list[str]:
        """Get references for a paper (papers this paper cites).

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum references to fetch

        Returns:
            List of cited paper IDs
        """
        return await self._fetch_related_papers(
            paper_id, "references", lambda item: item.cited_paper, limit
        )

    async def get_reference_corpus_ids(
        self,
        paper_id: str,
        limit: int = 200,
    ) -> list[int]:
        """Get corpus IDs of papers referenced by the given paper.

        Similar to ``get_paper_references`` but requests ``corpusId`` instead
        of ``paperId`` and returns integer corpus IDs for bib-coupling.

        Args:
            paper_id: Semantic Scholar paper ID.
            limit: Maximum references to fetch. Maximum 1000.

        Returns:
            List of corpus IDs for cited papers.
        """
        corpus_ids: list[int] = []
        offset = 0

        remaining = limit
        while remaining > 0:
            page_size = min(remaining, 1000)
            params = {
                "fields": "corpusId",
                "limit": str(page_size),
                "offset": str(offset),
            }
            try:
                raw = await self._request_with_retry(
                    "GET",
                    f"{self.BASE_URL}/paper/{paper_id}/references",
                    params=params,
                )
                result = S2RelatedPapersResponse.model_validate(raw)
                for item in result.data:
                    ref = item.cited_paper
                    if ref and ref.corpus_id is not None:
                        corpus_ids.append(ref.corpus_id)

                offset += len(result.data)
                remaining = limit - len(corpus_ids)

                if len(result.data) < page_size:
                    break

            except Exception as e:
                logger.warning(
                    f"Failed to fetch reference corpus IDs for {paper_id}"
                    f" at offset {offset}: {e}"
                )
                break

        return corpus_ids

    async def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 1000,
    ) -> list[str]:
        """Get citations for a paper (papers that cite this paper).

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum citations to fetch

        Returns:
            List of citing paper IDs
        """
        return await self._fetch_related_papers(
            paper_id, "citations", lambda item: item.citing_paper, limit
        )

    async def get_paper_batch(
        self,
        paper_ids: list[str],
        fields: list[str] | None = None,
    ) -> list[S2Paper | None]:
        """Fetch multiple papers in a single batch request.

        Uses the POST /paper/batch endpoint. Maximum 500 IDs per call.

        Args:
            paper_ids: Paper identifiers (paperId, CorpusId:123, DOI:..., etc.).
            fields: Fields to return (defaults to paperId and title).

        Returns:
            List of S2Paper objects (or None for papers not found), in the same
            order as the input IDs.
        """
        if len(paper_ids) > 500:
            raise ValueError(
                f"Batch endpoint accepts at most 500 IDs, got {len(paper_ids)}"
            )

        params: dict[str, str] = {}
        if fields:
            params["fields"] = ",".join(fields)

        raw = await self._post_json_list(
            f"{self.BASE_URL}/paper/batch",
            body={"ids": paper_ids},
            params=params,
        )
        return [
            S2Paper.model_validate(item) if item is not None else None for item in raw
        ]
