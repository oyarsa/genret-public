"""Rate limiting for the endpoints that call an LLM, plus the health check."""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

import genret.explorer.explain as explain_mod
import genret.explorer.server as server_mod
from explorer_harness import (
    GRAPH_CIDS,
    ExplorerDataset,
    FakeEncoder,
    make_config,
)
from genret.explorer.rate_limit import RateLimitConfig, RateLimiter
from genret.explorer.server import create_app


def test_allows_requests_under_the_limit() -> None:
    """Requests pass until the window is full."""
    limiter = RateLimiter(RateLimitConfig(per_client=3, total=100))
    assert [limiter.check("a", now=0.0) for _ in range(3)] == [None, None, None]


def test_blocks_and_reports_the_wait() -> None:
    """The next request is refused with the time until a slot frees."""
    limiter = RateLimiter(RateLimitConfig(per_client=2, total=100, window_seconds=60))
    limiter.check("a", now=0.0)
    limiter.check("a", now=10.0)

    retry_after = limiter.check("a", now=20.0)
    assert retry_after is not None
    assert retry_after == pytest.approx(40.0)


def test_window_slides() -> None:
    """Old requests stop counting once they leave the window."""
    limiter = RateLimiter(RateLimitConfig(per_client=1, total=100, window_seconds=60))
    assert limiter.check("a", now=0.0) is None
    assert limiter.check("a", now=30.0) is not None
    assert limiter.check("a", now=61.0) is None


def test_clients_are_counted_separately() -> None:
    """One caller's quota does not consume another's."""
    limiter = RateLimiter(RateLimitConfig(per_client=1, total=100))
    assert limiter.check("a", now=0.0) is None
    assert limiter.check("b", now=0.0) is None
    assert limiter.check("a", now=0.0) is not None


def test_global_limit_bounds_many_clients() -> None:
    """Distinct addresses cannot together exceed the overall budget."""
    limiter = RateLimiter(RateLimitConfig(per_client=10, total=2))
    assert limiter.check("a", now=0.0) is None
    assert limiter.check("b", now=0.0) is None
    assert limiter.check("c", now=0.0) is not None


def test_zero_disables_limiting() -> None:
    """A limit of zero switches the mechanism off entirely."""
    limiter = RateLimiter(RateLimitConfig(per_client=0))
    assert not limiter.enabled
    assert all(limiter.check("a", now=0.0) is None for _ in range(50))


@pytest.fixture
async def limited_client(
    explorer_dataset: ExplorerDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[httpx.AsyncClient]:
    """App allowing a single paid request per client."""
    monkeypatch.setattr(
        server_mod, "load_encoder", lambda backend, onnx_dir: FakeEncoder()
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def fake_call(model: str, user_prompt: str) -> str:
        return "canned"

    monkeypatch.setattr(explain_mod, "_call_llm", fake_call)

    config = make_config(explorer_dataset, tmp_path / "cache")
    config.rate_limit = RateLimitConfig(per_client=1, total=10)

    app = create_app(config)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=30.0
        ) as client:
            yield client


async def test_explain_is_rate_limited(limited_client: httpx.AsyncClient) -> None:
    """A second explain from the same caller is refused with 429."""
    body = {"corpusid": GRAPH_CIDS[0], "query": "attention"}
    assert (await limited_client.post("/api/explain", json=body)).status_code == 200

    resp = await limited_client.post(
        "/api/explain", json={"corpusid": GRAPH_CIDS[1], "query": "attention"}
    )
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1


async def test_recommend_shares_the_quota(limited_client: httpx.AsyncClient) -> None:
    """Recommend draws on the same budget as explain."""
    body = {"corpusid": GRAPH_CIDS[0], "query": "attention"}
    assert (await limited_client.post("/api/explain", json=body)).status_code == 200

    async with limited_client.stream("GET", f"/api/recommend/{GRAPH_CIDS[0]}") as resp:
        assert resp.status_code == 429


async def test_free_endpoints_are_not_limited(
    limited_client: httpx.AsyncClient,
) -> None:
    """Search and paper lookup cost nothing, so they stay open."""
    await limited_client.post(
        "/api/explain", json={"corpusid": GRAPH_CIDS[0], "query": "x"}
    )
    for _ in range(5):
        resp = await limited_client.post(
            "/api/search", json={"query": "attention", "top_k": 2}
        )
        assert resp.status_code == 200
    assert (await limited_client.get(f"/api/paper/{GRAPH_CIDS[0]}")).status_code == 200


async def test_health_reports_the_loaded_corpus(
    limited_client: httpx.AsyncClient,
) -> None:
    """The health check is unauthenticated, unlimited, and informative."""
    for _ in range(3):
        resp = await limited_client.get("/api/health")
        assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["graph_papers"] == len(GRAPH_CIDS)
