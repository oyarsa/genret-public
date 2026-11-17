"""The server must not advertise itself when not in dev mode.

Internet-wide scanners fingerprint hosts by fetching /openapi.json and
/docs; publishing them hands over a map of the endpoints that spend API
credit. Serving mode also needs no CORS, since the frontend is served
from the same origin.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

import genret.explorer.server as server_mod
from explorer_harness import (
    GRAPH_CIDS,
    ExplorerDataset,
    FakeEncoder,
    make_config,
)
from genret.explorer.server import create_app


@pytest.fixture
async def serving_client(
    explorer_dataset: ExplorerDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[httpx.AsyncClient]:
    """A client against the app as it runs when serving (not dev mode)."""
    monkeypatch.setattr(
        server_mod, "load_encoder", lambda backend, onnx_dir: FakeEncoder()
    )
    config = make_config(explorer_dataset, tmp_path / "cache")
    config.dev_mode = False

    app = create_app(config)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=30.0
        ) as client:
            yield client


@pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc"])
async def test_api_documentation_is_not_published(
    serving_client: httpx.AsyncClient, path: str
) -> None:
    """The schema and its UIs are absent when serving."""
    resp = await serving_client.get(path)
    assert resp.status_code == 404


async def test_no_cors_headers_when_serving(
    serving_client: httpx.AsyncClient,
) -> None:
    """Cross-origin callers get no permission from the response."""
    resp = await serving_client.post(
        "/api/search",
        json={"query": "attention", "top_k": 1},
        headers={"Origin": "https://evil.example"},
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


async def test_api_still_works_when_serving(
    serving_client: httpx.AsyncClient,
) -> None:
    """Hardening does not disturb the endpoints the frontend uses."""
    resp = await serving_client.get(f"/api/paper/{GRAPH_CIDS[0]}")
    assert resp.status_code == 200


async def test_dev_mode_keeps_the_docs(harness: object) -> None:
    """The dev-mode harness still exposes the schema for local work."""
    from explorer_harness import ExplorerHarness  # noqa: PLC0415

    assert isinstance(harness, ExplorerHarness)
    resp = await harness.client.get("/openapi.json")
    assert resp.status_code == 200
