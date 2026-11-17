"""SSE tests for GET /api/recommend/{corpusid} across all three paper paths."""

import pytest

import genret.explorer.aleth as aleth_mod
from explorer_harness import (
    CACHED_CORPUS_CID,
    CORPUS_CIDS,
    GHOST_CIDS,
    GRAPH_CIDS,
    ExplorerHarness,
    collect_sse,
    make_facets,
)
from genret.paper_parsing import PaperGraph, Section, SectionType
from genret.types import CorpusId


def _make_paper_graph(corpusid: int) -> PaperGraph:
    """Minimal parsed paper for faking S2ORC loading."""
    return PaperGraph(
        corpusid=CorpusId(corpusid),
        title=f"Corpus Paper {corpusid} on Neural Retrieval",
        authors=["Ada Author"],
        sections=[
            Section(
                heading="Introduction",
                type=SectionType.INTRODUCTION,
                paragraphs=["Some fulltext."],
            )
        ],
        abstract="An abstract.",
    )


async def test_recommend_in_graph(harness: ExplorerHarness) -> None:
    """In-graph recommend streams checking -> retrieving -> result."""
    cid = GRAPH_CIDS[0]
    events = await collect_sse(harness.client, f"/api/recommend/{cid}")

    names = [e for e, _ in events]
    assert names == ["status", "status", "result"]
    stages = [d["stage"] for e, d in events if e == "status"]
    assert stages == ["checking", "retrieving"]

    result = events[-1][1]
    assert result["query_paper"]["source"] == "in_graph"
    assert result["papers"]
    assert all(p["corpusid"] != cid for p in result["papers"])
    # Every recommendation records which facets contributed it
    assert all(p["matched_facets"] for p in result["papers"])


async def test_recommend_in_corpus_cached(harness: ExplorerHarness) -> None:
    """In-corpus recommend with cached facets needs no LLM call."""
    events = await collect_sse(harness.client, f"/api/recommend/{CACHED_CORPUS_CID}")

    stages = [d["stage"] for e, d in events if e == "status"]
    assert stages == ["checking", "extracting", "retrieving"]
    assert events[-1][0] == "result"

    result = events[-1][1]
    assert result["query_paper"]["source"] == "in_corpus"
    returned = {p["corpusid"] for p in result["papers"]}
    assert returned
    assert returned <= set(GRAPH_CIDS)


async def test_recommend_in_corpus_cache_miss_extracts_and_caches(
    harness: ExplorerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache miss extracts facets and writes them to the Aleth cache."""
    cid = CORPUS_CIDS[1]
    assert not (harness.cache_dir / "aleth" / f"{cid}.json").exists()

    async def fake_extract(
        corpusid: int, paper: PaperGraph, extraction_model: str
    ) -> object:
        return make_facets(corpusid)

    def fake_load(
        corpusid: int,
        s2orc_index: dict[int, tuple[str, int]],
        s2orc_dir: object,
        abstract: str | None = None,
        title: str | None = None,
    ) -> PaperGraph:
        return _make_paper_graph(corpusid)

    # The fixture shard records index correctly but are not full S2ORC
    # documents, so paper loading is faked alongside the LLM call; the
    # cache read/write logic in between runs for real.
    monkeypatch.setattr(aleth_mod, "load_s2orc_paper", fake_load)
    monkeypatch.setattr(aleth_mod, "extract_facets", fake_extract)

    events = await collect_sse(harness.client, f"/api/recommend/{cid}")
    assert events[-1][0] == "result"
    assert events[-1][1]["query_paper"]["source"] == "in_corpus"

    assert (harness.cache_dir / "aleth" / f"{cid}.json").exists()


async def test_recommend_out_of_corpus_rejected_without_silver(
    harness: ExplorerHarness,
) -> None:
    """With Silver off, out-of-corpus recommends fail fast and clearly."""
    events = await collect_sse(harness.client, "/api/recommend/9999")

    names = [e for e, _ in events]
    assert names == ["status", "error"]
    assert events[0][1]["stage"] == "checking"
    message = events[-1][1]["message"]
    assert isinstance(message, str)
    assert "disabled" in message


async def test_recommend_ghost_paper_rejected_without_silver(
    harness: ExplorerHarness,
) -> None:
    """Metadata-only papers route as out-of-corpus and are rejected.

    With Silver off they must fail immediately instead of attempting PDF
    acquisition. (They are also excluded from title search, so the UI
    cannot normally reach this state.)
    """
    events = await collect_sse(harness.client, f"/api/recommend/{GHOST_CIDS[0]}")
    assert [e for e, _ in events] == ["status", "error"]
