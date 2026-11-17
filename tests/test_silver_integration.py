"""Integration tests for Silver ↔ Pan (recommend + search) wiring."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from genret.explorer.models import SearchPaperResult
from genret.explorer.recommend import _acquire_out_of_corpus
from genret.explorer.silver import AcquiredPaper, save_to_cache
from genret.paper_parsing import PaperGraph, Section, SectionType
from genret.s2.api import SemanticScholarAPI
from genret.s2.types import S2Paper
from genret.types import CorpusId


def _make_graph(corpusid: int = 42, title: str = "Test Paper") -> PaperGraph:
    """Create a minimal PaperGraph for testing."""
    return PaperGraph(
        corpusid=CorpusId(corpusid),
        title=title,
        authors=["Alice"],
        sections=[
            Section(
                heading="Introduction",
                type=SectionType.INTRODUCTION,
                paragraphs=["Hello world."],
            )
        ],
        abstract="Test abstract.",
    )


def _make_acquired(corpusid: int = 42, title: str = "Test Paper") -> AcquiredPaper:
    """Create a minimal AcquiredPaper for testing."""
    return AcquiredPaper(
        graph=_make_graph(corpusid, title), references=frozenset({100})
    )


# ---------------------------------------------------------------------------
# _acquire_out_of_corpus tests
# ---------------------------------------------------------------------------


class TestAcquireOutOfCorpus:
    """Tests for the _acquire_out_of_corpus helper."""

    @pytest.mark.asyncio
    async def test_silver_cache_hit(self, tmp_path: Path) -> None:
        """Cached papers are returned without hitting the S2 API."""
        save_to_cache(tmp_path, _make_acquired(42))
        s2_api = AsyncMock(spec=SemanticScholarAPI)

        result = await _acquire_out_of_corpus(42, s2_api, tmp_path)

        assert result is not None
        assert result.graph.corpusid == CorpusId(42)
        s2_api.get_paper_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_s2_api_paper_not_found(self, tmp_path: Path) -> None:
        """Returns None when S2 API returns no results."""
        s2_api = AsyncMock(spec=SemanticScholarAPI)
        s2_api.get_paper_batch = AsyncMock(return_value=[None])

        result = await _acquire_out_of_corpus(999, s2_api, tmp_path)

        assert result is None

    @pytest.mark.asyncio
    async def test_s2_api_exception(self, tmp_path: Path) -> None:
        """Returns None when S2 API raises an exception."""
        s2_api = AsyncMock(spec=SemanticScholarAPI)
        s2_api.get_paper_batch = AsyncMock(side_effect=RuntimeError("API down"))

        result = await _acquire_out_of_corpus(999, s2_api, tmp_path)

        assert result is None

    @pytest.mark.asyncio
    async def test_acquire_failure(self, tmp_path: Path) -> None:
        """Returns None when acquire_paper raises ValueError (no PDF URL)."""
        s2_paper = S2Paper.model_validate(
            {
                "paperId": "abc123",
                "corpusId": 42,
                "title": "No PDF Paper",
            }
        )
        s2_api = AsyncMock(spec=SemanticScholarAPI)
        s2_api.get_paper_batch = AsyncMock(return_value=[s2_paper])
        s2_api.client = AsyncMock()

        result = await _acquire_out_of_corpus(42, s2_api, tmp_path)

        # Should fail because there's no PDF URL
        assert result is None


# ---------------------------------------------------------------------------
# search-papers S2 API merge tests
# ---------------------------------------------------------------------------


class TestSearchPapersS2Merge:
    """Tests for S2 API result merging in search-papers."""

    def test_search_paper_result_s2_api_source(self) -> None:
        """SearchPaperResult accepts 's2_api' as a source value."""
        result = SearchPaperResult(
            corpusid=123,
            title="External Paper",
            authors=["Bob"],
            year=2025,
            venue="ACL",
            source="s2_api",
        )
        assert result.source == "s2_api"

    def test_search_paper_result_all_sources(self) -> None:
        """All three source values are valid."""
        for source in ("in_graph", "in_corpus", "s2_api"):
            result = SearchPaperResult(
                corpusid=1,
                title="T",
                authors=[],
                year=None,
                venue=None,
                source=source,
            )
            assert result.source == source
