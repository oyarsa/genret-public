"""HTTP-level tests for the explorer server (Pan).

Grew out of the manual cURL playbook these replaced (see
docs/reference/explorer_e2e_playbook.md): assertions against the real
FastAPI app, served in-process through httpx's ASGI transport with a
synthetic fixture dataset.
"""

from explorer_harness import (
    CACHED_CORPUS_CID,
    CORPUS_CIDS,
    GHOST_CIDS,
    GRAPH_CIDS,
    ExplorerHarness,
    facet_items,
)

# --- POST /api/search ---


async def test_search_happy_path(harness: ExplorerHarness) -> None:
    """Search returns top_k papers with edges, a query id, and facet scores."""
    resp = await harness.client.post(
        "/api/search", json={"query": "attention is all you need", "top_k": 5}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["papers"]) == 5
    assert body["query_id"]
    paper = body["papers"][0]
    for field in ("corpusid", "title", "authors", "facets", "query_score"):
        assert field in paper
    assert set(paper["facet_scores"]) == {
        "problems",
        "methods",
        "results",
        "contributions",
    }
    assert "matched_facets" in paper


async def test_search_ranks_matching_paper_first(harness: ExplorerHarness) -> None:
    """Querying with a paper's own facet item text ranks that paper first."""
    target = GRAPH_CIDS[7]
    query = facet_items(target, "methods")[0]
    resp = await harness.client.post("/api/search", json={"query": query, "top_k": 3})
    top = resp.json()["papers"][0]
    assert top["corpusid"] == target
    # The identity-encoded query matches on every facet, so all four are
    # marked as having contributed
    assert set(top["matched_facets"]) == {
        "problems",
        "methods",
        "results",
        "contributions",
    }


async def test_search_empty_query_400(harness: ExplorerHarness) -> None:
    """An empty query is rejected with 400."""
    resp = await harness.client.post("/api/search", json={"query": "", "top_k": 5})
    assert resp.status_code == 400


async def test_search_whitespace_query_400(harness: ExplorerHarness) -> None:
    """A whitespace-only query is rejected with 400."""
    resp = await harness.client.post("/api/search", json={"query": "   ", "top_k": 5})
    assert resp.status_code == 400


async def test_search_top_k_one(harness: ExplorerHarness) -> None:
    """top_k=1 returns exactly one paper."""
    resp = await harness.client.post("/api/search", json={"query": "nlp", "top_k": 1})
    assert resp.status_code == 200
    assert len(resp.json()["papers"]) == 1


# --- POST /api/expand ---


async def _search(
    harness: ExplorerHarness, query: str = "test query"
) -> tuple[str, int]:
    """Run a search and return (query_id, top corpusid)."""
    body = (
        await harness.client.post("/api/search", json={"query": query, "top_k": 5})
    ).json()
    return body["query_id"], body["papers"][0]["corpusid"]


async def test_expand_happy_path(harness: ExplorerHarness) -> None:
    """Expanding a search result yields new papers and edges."""
    query_id, cid = await _search(harness)
    resp = await harness.client.post(
        "/api/expand", json={"corpusid": cid, "query_id": query_id, "top_k": 3}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["new_papers"]) == 3
    assert body["new_edges"]


async def test_expand_facet_filter(harness: ExplorerHarness) -> None:
    """Expanding with a facet filter only returns edges of that facet."""
    query_id, cid = await _search(harness)
    resp = await harness.client.post(
        "/api/expand",
        json={"corpusid": cid, "query_id": query_id, "facet": "methods", "top_k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    facets = {e["facet"] for e in body["new_edges"]}
    assert facets <= {"methods"}
    # Contributing facets follow the facet filter
    assert all(p["matched_facets"] == ["methods"] for p in body["new_papers"])


async def test_expand_unknown_query_id_404(harness: ExplorerHarness) -> None:
    """Expanding with an expired/unknown query id is a 404."""
    resp = await harness.client.post(
        "/api/expand", json={"corpusid": 1, "query_id": "nonexistent", "top_k": 3}
    )
    assert resp.status_code == 404


async def test_expand_excludes_known_papers(harness: ExplorerHarness) -> None:
    """Papers already on the caller's graph are not returned again.

    In a dense neighbourhood most of a node's nearest neighbours are
    already displayed, so without this an expansion appears to do nothing.
    """
    query_id, cid = await _search(harness)
    first = await harness.client.post(
        "/api/expand", json={"corpusid": cid, "query_id": query_id, "top_k": 3}
    )
    seen = [p["corpusid"] for p in first.json()["new_papers"]]
    assert seen

    second = await harness.client.post(
        "/api/expand",
        json={
            "corpusid": cid,
            "query_id": query_id,
            "top_k": 3,
            "known_corpusids": [*seen, cid],
        },
    )
    returned = [p["corpusid"] for p in second.json()["new_papers"]]
    assert returned
    assert not set(returned) & set(seen)
    # The source paper is never returned as its own neighbour
    assert cid not in returned


async def test_expand_links_new_papers_to_known_graph(
    harness: ExplorerHarness,
) -> None:
    """New papers come back connected to the papers already on screen.

    The fixture graph chains papers (i to i+1 and i+2), so expanding
    paper 10 with paper 8 already known must return paper 9 linked to
    paper 8 -- an edge that does not start at the expanded node.
    """
    source, known, expected = GRAPH_CIDS[10], GRAPH_CIDS[8], GRAPH_CIDS[9]
    body = (
        await harness.client.post(
            "/api/expand-from",
            json={"corpusid": source, "top_k": 3, "known_corpusids": [known]},
        )
    ).json()

    new_cids = {p["corpusid"] for p in body["new_papers"]}
    assert known not in new_cids
    assert expected in new_cids
    # Every edge touches a new paper (the caller already has the rest)
    assert all(
        e["source"] in new_cids or e["target"] in new_cids for e in body["new_edges"]
    )
    # The known paper is linked to the new one, not just the expanded node
    assert any(
        {e["source"], e["target"]} == {known, expected} for e in body["new_edges"]
    )


async def test_expand_exhausted_neighbourhood_is_empty(
    harness: ExplorerHarness,
) -> None:
    """When every neighbour is already known, nothing new comes back."""
    query_id, cid = await _search(harness)
    body = (
        await harness.client.post(
            "/api/expand",
            json={
                "corpusid": cid,
                "query_id": query_id,
                "top_k": 5,
                "known_corpusids": GRAPH_CIDS,
            },
        )
    ).json()
    assert body["new_papers"] == []
    assert body["new_edges"] == []


async def test_expand_unknown_corpusid_empty(harness: ExplorerHarness) -> None:
    """Expanding an unknown corpus id returns empty results, not an error."""
    query_id, _ = await _search(harness)
    resp = await harness.client.post(
        "/api/expand", json={"corpusid": 999999999, "query_id": query_id, "top_k": 3}
    )
    assert resp.status_code == 200
    assert resp.json() == {"new_papers": [], "new_edges": []}


# --- POST /api/expand-from ---


async def test_expand_from_in_graph(harness: ExplorerHarness) -> None:
    """Expand-from works for in-graph papers without a cached query."""
    resp = await harness.client.post(
        "/api/expand-from", json={"corpusid": GRAPH_CIDS[0], "top_k": 3}
    )
    assert resp.status_code == 200
    assert resp.json()["new_papers"]


async def test_expand_from_non_graph_404(harness: ExplorerHarness) -> None:
    """Expand-from a paper outside the retrieval graph is a 404."""
    resp = await harness.client.post(
        "/api/expand-from", json={"corpusid": CORPUS_CIDS[0], "top_k": 3}
    )
    assert resp.status_code == 404


# --- POST /api/edges ---


async def test_edges_adjacent_papers(harness: ExplorerHarness) -> None:
    """Chained fixture papers have inter-edges with full edge fields."""
    resp = await harness.client.post("/api/edges", json={"corpusids": GRAPH_CIDS[:3]})
    assert resp.status_code == 200
    edges = resp.json()["edges"]
    assert edges
    for edge in edges:
        assert {"source", "target", "facet", "score"} <= set(edge)


async def test_edges_empty_list(harness: ExplorerHarness) -> None:
    """An empty corpus id list yields no edges."""
    resp = await harness.client.post("/api/edges", json={"corpusids": []})
    assert resp.status_code == 200
    assert resp.json()["edges"] == []


async def test_edges_unknown_corpusids(harness: ExplorerHarness) -> None:
    """Unknown corpus ids yield no edges."""
    resp = await harness.client.post("/api/edges", json={"corpusids": [1, 2, 3]})
    assert resp.status_code == 200
    assert resp.json()["edges"] == []


# --- GET /api/paper/{corpusid} ---


async def test_paper_detail_in_graph(harness: ExplorerHarness) -> None:
    """In-graph paper detail has metadata, facets, url, and citations."""
    cid = GRAPH_CIDS[0]
    resp = await harness.client.get(f"/api/paper/{cid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"]
    assert body["authors"] == ["Ada Author", "Ben Builder"]
    assert set(body["facets"]) == {"problems", "methods", "results", "contributions"}
    assert body["url"] == f"https://example.org/{cid}"
    assert body["citation_count"] == 7
    # The buggy query-dependent facet_scores were removed from this endpoint
    assert "facet_scores" not in body


async def test_paper_detail_not_found(harness: ExplorerHarness) -> None:
    """Unknown papers are a 404."""
    resp = await harness.client.get("/api/paper/1")
    assert resp.status_code == 404


async def test_paper_detail_in_corpus(harness: ExplorerHarness) -> None:
    """In-corpus paper detail has metadata but no preloaded facets."""
    resp = await harness.client.get(f"/api/paper/{CORPUS_CIDS[0]}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"]
    assert body["facets"] == {}
    assert body["url"]


# --- POST /api/search-papers ---


async def test_search_papers_local(harness: ExplorerHarness) -> None:
    """Title search returns local papers only."""
    resp = await harness.client.post(
        "/api/search-papers", json={"query": "paper", "limit": 10}
    )
    assert resp.status_code == 200
    papers = resp.json()["papers"]
    assert papers
    sources = [p["source"] for p in papers]
    assert set(sources) <= {"in_graph", "in_corpus"}


async def test_search_papers_ranks_by_match_position(
    harness: ExplorerHarness,
) -> None:
    """Prefix matches outrank later matches, regardless of source.

    The old in-graph-first ordering made in-corpus papers unreachable for
    any query that graph titles also matched.
    """
    resp = await harness.client.post(
        "/api/search-papers", json={"query": "Corpus Paper", "limit": 10}
    )
    papers = resp.json()["papers"]
    assert papers
    # "Corpus Paper ..." is a prefix match only for in-corpus papers, so
    # they must fill the top slots even though every graph title also
    # contains "paper"
    assert {p["source"] for p in papers[: len(CORPUS_CIDS)]} == {"in_corpus"}


async def test_search_papers_empty_query_400(harness: ExplorerHarness) -> None:
    """An empty title query is rejected with 400."""
    resp = await harness.client.post(
        "/api/search-papers", json={"query": "", "limit": 5}
    )
    assert resp.status_code == 400


async def test_search_papers_gibberish(harness: ExplorerHarness) -> None:
    """A query matching nothing returns an empty list."""
    resp = await harness.client.post(
        "/api/search-papers", json={"query": "xyzzy999foobarbaz", "limit": 5}
    )
    assert resp.status_code == 200
    assert resp.json()["papers"] == []


async def test_search_papers_excludes_ghost_papers(harness: ExplorerHarness) -> None:
    """Metadata-only papers are absent from title search with Silver off.

    They cannot be recommended (recommend routes on fulltext membership),
    so listing them as IN CORPUS promised a flow that would fail at the
    Silver acquisition step. Regression test for that routing mismatch.
    """
    resp = await harness.client.post(
        "/api/search-papers", json={"query": "Ghost Paper", "limit": 10}
    )
    assert resp.status_code == 200
    assert resp.json()["papers"] == []
    ghost = GHOST_CIDS[0]
    resp = await harness.client.post(
        "/api/search-papers", json={"query": f"Ghost Paper {ghost}", "limit": 10}
    )
    assert resp.json()["papers"] == []


async def test_search_papers_finds_cached_corpus_paper(
    harness: ExplorerHarness,
) -> None:
    """Fulltext-backed in-corpus papers remain searchable."""
    resp = await harness.client.post(
        "/api/search-papers",
        json={"query": f"Corpus Paper {CACHED_CORPUS_CID}", "limit": 5},
    )
    papers = resp.json()["papers"]
    assert [p["corpusid"] for p in papers] == [CACHED_CORPUS_CID]
    assert papers[0]["source"] == "in_corpus"


# --- Malformed requests ---


async def test_search_missing_fields_422(harness: ExplorerHarness) -> None:
    """Requests missing required fields are a 422."""
    resp = await harness.client.post("/api/search", json={"bad": True})
    assert resp.status_code == 422


async def test_search_wrong_method_405(harness: ExplorerHarness) -> None:
    """The wrong HTTP method is a 405 in dev mode.

    In production mode the catch-all static mount turns this into a 404.
    """
    resp = await harness.client.get("/api/search")
    assert resp.status_code == 405


async def test_search_wrong_content_type_422(harness: ExplorerHarness) -> None:
    """Non-JSON bodies are a 422."""
    resp = await harness.client.post(
        "/api/search", content="hello", headers={"Content-Type": "text/plain"}
    )
    assert resp.status_code == 422
