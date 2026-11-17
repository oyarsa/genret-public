"""Shared test fixtures and helpers."""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

import genret.explorer.server as server_mod
from explorer_harness import (
    CACHED_CORPUS_CID,
    ExplorerDataset,
    ExplorerHarness,
    FakeEncoder,
    build_dataset,
    make_config,
    make_facets,
)
from genret._shed.pytest import approx as approx
from genret.explorer.facet_cache import FacetCache
from genret.explorer.server import create_app


@pytest.fixture(scope="session")
def explorer_dataset(tmp_path_factory: pytest.TempPathFactory) -> ExplorerDataset:
    """Build the synthetic explorer dataset once per session (read-only)."""
    return build_dataset(tmp_path_factory.mktemp("explorer_data"))


@pytest.fixture
async def harness(
    explorer_dataset: ExplorerDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[ExplorerHarness]:
    """Explorer app with Silver disabled (the demo configuration).

    Serves the real FastAPI app in-process via httpx's ASGI transport, with
    the MiniLM encoder replaced by a deterministic fake. See
    ``explorer_harness.py`` for the dataset layout.
    """
    monkeypatch.setattr(
        server_mod, "load_encoder", lambda backend, onnx_dir: FakeEncoder()
    )

    cache_dir = tmp_path / "cache"
    aleth_dir = cache_dir / "aleth"
    aleth_dir.mkdir(parents=True)
    FacetCache(aleth_dir).put(make_facets(CACHED_CORPUS_CID))

    app = create_app(make_config(explorer_dataset, cache_dir))
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=30.0
        ) as client:
            yield ExplorerHarness(app=app, client=client, cache_dir=cache_dir)
