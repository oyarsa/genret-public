"""Tests for the Silver-enabled configuration (S2 API + out-of-corpus)."""

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

import genret.explorer.server as server_mod
from explorer_harness import (
    GRAPH_CIDS,
    ExplorerDataset,
    ExplorerHarness,
    FakeEncoder,
    collect_sse,
    make_config,
)
from genret.explorer.server import create_app
from genret.s2.types import S2Paper

EXTERNAL_CID = 9990101
"""Corpus ID only known to the fake S2 API."""


class FakeS2API:
    """Stands in for SemanticScholarAPI; never touches the network."""

    def __init__(self, api_key: str) -> None:
        self.client = None

    async def start(self) -> None:
        """No-op."""

    async def stop(self) -> None:
        """No-op."""

    async def search_by_title(
        self, query: str, limit: int = 20, fields: Sequence[str] = ()
    ) -> list[S2Paper]:
        """Return one external paper plus one already-in-graph paper."""
        return [
            S2Paper.model_validate(
                {
                    "corpusId": EXTERNAL_CID,
                    "title": "External Paper on Attention",
                    "year": 2023,
                    "venue": "ExtConf",
                }
            ),
            S2Paper.model_validate(
                {"corpusId": GRAPH_CIDS[0], "title": "Duplicate of a graph paper"}
            ),
        ]

    async def get_paper_batch(
        self, ids: Sequence[str], fields: Sequence[str] = ()
    ) -> list[S2Paper | None]:
        """Simulate a paper the S2 API cannot find."""
        return []


@pytest.fixture
async def silver_harness(
    explorer_dataset: ExplorerDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[ExplorerHarness]:
    """Explorer app with Silver enabled and a fake S2 client."""
    monkeypatch.setattr(
        server_mod, "load_encoder", lambda backend, onnx_dir: FakeEncoder()
    )
    monkeypatch.setattr(server_mod, "SemanticScholarAPI", FakeS2API)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    app = create_app(make_config(explorer_dataset, cache_dir, silver=True))
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=30.0
        ) as client:
            yield ExplorerHarness(app=app, client=client, cache_dir=cache_dir)


async def test_startup_requires_s2_key_when_silver_enabled(
    explorer_dataset: ExplorerDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With Silver on and no S2 key, startup fails loudly."""
    monkeypatch.setattr(
        server_mod, "load_encoder", lambda backend, onnx_dir: FakeEncoder()
    )
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

    app = create_app(make_config(explorer_dataset, tmp_path / "cache", silver=True))
    with pytest.raises(RuntimeError, match="SEMANTIC_SCHOLAR_API_KEY"):
        async with LifespanManager(app):
            pass


async def test_search_papers_merges_s2_results(
    silver_harness: ExplorerHarness,
) -> None:
    """S2 results merge into title search, deduplicated and source-tagged."""
    # A query with no local title matches, so the fake S2 results are not
    # truncated away by the limit
    resp = await silver_harness.client.post(
        "/api/search-papers", json={"query": "external", "limit": 10}
    )
    assert resp.status_code == 200
    papers = resp.json()["papers"]
    by_cid = {p["corpusid"]: p for p in papers}

    assert by_cid[EXTERNAL_CID]["source"] == "s2_api"
    assert by_cid[GRAPH_CIDS[0]]["source"] == "in_graph"


async def test_search_papers_local_only_skips_s2(
    silver_harness: ExplorerHarness,
) -> None:
    """local_only suppresses S2 results even with Silver enabled."""
    resp = await silver_harness.client.post(
        "/api/search-papers",
        json={"query": "attention", "limit": 10, "local_only": True},
    )
    sources = {p["source"] for p in resp.json()["papers"]}
    assert "s2_api" not in sources


async def test_recommend_out_of_corpus_acquisition_failure(
    silver_harness: ExplorerHarness,
) -> None:
    """A failed Silver acquisition errors cleanly after the acquiring stage."""
    events = await collect_sse(silver_harness.client, "/api/recommend/9999")

    stages = [d["stage"] for e, d in events if e == "status"]
    assert stages == ["checking", "acquiring"]
    assert events[-1][0] == "error"
    message = events[-1][1]["message"]
    assert isinstance(message, str)
    assert "Silver" in message
