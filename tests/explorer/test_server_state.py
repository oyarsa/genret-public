"""Server state and concurrency behaviour tests.

These use the in-process app reference to assert on internals that HTTP
responses alone cannot show (query cache eviction, concurrent access).
"""

import asyncio

from explorer_harness import GRAPH_CIDS, ExplorerHarness, collect_sse
from genret.explorer.server import MAX_QUERY_CACHE, ServerState


def _state(harness: ExplorerHarness) -> ServerState:
    """The live ServerState behind the app."""
    state: ServerState | None = harness.app.state.server
    assert state is not None
    return state


async def test_query_cache_evicts_oldest(harness: ExplorerHarness) -> None:
    """Filling the query cache evicts the oldest query, whose expand 404s."""
    first = (
        await harness.client.post(
            "/api/search", json={"query": "first query", "top_k": 1}
        )
    ).json()["query_id"]

    last = first
    for i in range(MAX_QUERY_CACHE):
        resp = await harness.client.post(
            "/api/search", json={"query": f"filler query {i}", "top_k": 1}
        )
        assert resp.status_code == 200
        last = resp.json()["query_id"]

    state = _state(harness)
    assert len(state.query_cache) == MAX_QUERY_CACHE
    assert first not in state.query_cache
    assert last in state.query_cache

    cid = GRAPH_CIDS[0]
    resp = await harness.client.post(
        "/api/expand", json={"corpusid": cid, "query_id": first, "top_k": 3}
    )
    assert resp.status_code == 404
    resp = await harness.client.post(
        "/api/expand", json={"corpusid": cid, "query_id": last, "top_k": 3}
    )
    assert resp.status_code == 200


async def test_concurrent_searches(harness: ExplorerHarness) -> None:
    """Concurrent searches all succeed with distinct cached query ids."""
    queries = [f"concurrent query {i}" for i in range(4)]

    async def search(q: str) -> dict[str, object]:
        resp = await harness.client.post("/api/search", json={"query": q, "top_k": 5})
        assert resp.status_code == 200
        return resp.json()

    bodies = await asyncio.gather(*(search(q) for q in queries))

    query_ids = {b["query_id"] for b in bodies}
    assert len(query_ids) == 4
    state = _state(harness)
    assert query_ids <= set(state.query_cache)


async def test_concurrent_recommends(harness: ExplorerHarness) -> None:
    """Concurrent SSE recommends all complete with a result event."""
    cids = GRAPH_CIDS[:4]
    results = await asyncio.gather(
        *(collect_sse(harness.client, f"/api/recommend/{cid}") for cid in cids)
    )
    for events in results:
        assert events[-1][0] == "result"


async def test_search_results_are_deterministic(harness: ExplorerHarness) -> None:
    """The same query twice returns the same ranking."""

    async def top_ids() -> list[int]:
        resp = await harness.client.post(
            "/api/search", json={"query": "stability check", "top_k": 10}
        )
        return [p["corpusid"] for p in resp.json()["papers"]]

    assert await top_ids() == await top_ids()
