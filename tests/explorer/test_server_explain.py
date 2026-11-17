"""Tests for POST /api/explain (rich LLM connection explanations)."""

import pytest

import genret.explorer.explain as explain_mod
from explorer_harness import (
    CACHED_CORPUS_CID,
    GHOST_CIDS,
    GRAPH_CIDS,
    ExplorerHarness,
)

CANNED = "Both papers study attention; the retrieved one extends it to graphs."


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the LLM call with a canned response; returns the call log."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[str] = []

    async def fake_call(model: str, user_prompt: str) -> str:
        calls.append(user_prompt)
        return CANNED

    monkeypatch.setattr(explain_mod, "_call_llm", fake_call)
    return calls


async def test_explain_with_query_context(
    harness: ExplorerHarness, fake_llm: list[str]
) -> None:
    """A query-context explanation returns the generated text."""
    resp = await harness.client.post(
        "/api/explain",
        json={"corpusid": GRAPH_CIDS[0], "query": "attention mechanisms"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"explanation": CANNED}
    assert len(fake_llm) == 1
    assert "attention mechanisms" in fake_llm[0]
    assert f"problems item 1 of paper {GRAPH_CIDS[0]}" in fake_llm[0]


async def test_explain_with_seed_context(
    harness: ExplorerHarness, fake_llm: list[str]
) -> None:
    """A seed-paper explanation includes both papers' facets in the prompt."""
    resp = await harness.client.post(
        "/api/explain",
        json={"corpusid": GRAPH_CIDS[1], "seed_corpusid": CACHED_CORPUS_CID},
    )
    assert resp.status_code == 200
    assert f"problems item 1 of paper {CACHED_CORPUS_CID}" in fake_llm[0]
    assert f"problems item 1 of paper {GRAPH_CIDS[1]}" in fake_llm[0]


async def test_explain_is_cached(harness: ExplorerHarness, fake_llm: list[str]) -> None:
    """The same connection is generated once and then served from disk."""
    body = {"corpusid": GRAPH_CIDS[2], "query": "graphs"}
    first = await harness.client.post("/api/explain", json=body)
    second = await harness.client.post("/api/explain", json=body)

    assert first.json() == second.json()
    assert len(fake_llm) == 1
    assert list((harness.cache_dir / "explain").glob("*.json"))


async def test_explain_different_contexts_not_conflated(
    harness: ExplorerHarness, fake_llm: list[str]
) -> None:
    """Different queries for the same paper are cached separately."""
    cid = GRAPH_CIDS[3]
    await harness.client.post("/api/explain", json={"corpusid": cid, "query": "a"})
    await harness.client.post("/api/explain", json={"corpusid": cid, "query": "b"})
    assert len(fake_llm) == 2


async def test_explain_requires_exactly_one_context(
    harness: ExplorerHarness, fake_llm: list[str]
) -> None:
    """Neither or both contexts is a 400."""
    resp = await harness.client.post("/api/explain", json={"corpusid": GRAPH_CIDS[0]})
    assert resp.status_code == 400
    resp = await harness.client.post(
        "/api/explain",
        json={"corpusid": GRAPH_CIDS[0], "query": "x", "seed_corpusid": 1},
    )
    assert resp.status_code == 400


async def test_explain_unknown_paper_404(
    harness: ExplorerHarness, fake_llm: list[str]
) -> None:
    """Unknown papers, and papers without facets, are a 404."""
    resp = await harness.client.post(
        "/api/explain", json={"corpusid": 999999999, "query": "x"}
    )
    assert resp.status_code == 404
    # Ghost papers have metadata but no facets anywhere
    resp = await harness.client.post(
        "/api/explain", json={"corpusid": GHOST_CIDS[0], "query": "x"}
    )
    assert resp.status_code == 404


async def test_explain_without_api_key_503(
    harness: ExplorerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an OpenAI key the endpoint reports itself unavailable."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resp = await harness.client.post(
        "/api/explain", json={"corpusid": GRAPH_CIDS[0], "query": "x"}
    )
    assert resp.status_code == 503


async def test_explain_llm_failure_502(
    harness: ExplorerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed LLM call surfaces as a 502, not a cached empty result."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def failing(model: str, user_prompt: str) -> None:
        return None

    monkeypatch.setattr(explain_mod, "_call_llm", failing)
    resp = await harness.client.post(
        "/api/explain", json={"corpusid": GRAPH_CIDS[0], "query": "x"}
    )
    assert resp.status_code == 502
    assert not list((harness.cache_dir / "explain").glob("*.json"))
