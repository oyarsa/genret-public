"""Core graph operations for the explorer: search, expand, inter-edges."""
# pyright: basic

import logging
import time
import uuid
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from genret.explorer.data_loading import FACET_NAMES, ExplorerData
from genret.explorer.encoder import TextEncoder
from genret.explorer.models import (
    ExpandResponse,
    FacetScores,
    PaperNode,
    ProvenanceEdge,
    SearchResponse,
)
from genret.processing.facet_retrieval import (
    FusedCandidate,
    fuse_rrf,
    traverse_neighbours,
)
from genret.processing.reranker import Reranker, format_paper_pair
from genret.types import CorpusId

logger = logging.getLogger(__name__)


def _cosine_scores(
    query_vec: npt.NDArray[np.floating],
    corpus_mat: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Cosine similarity between a query vector and a corpus matrix."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    corpus_norms = corpus_mat / (
        np.linalg.norm(corpus_mat, axis=1, keepdims=True) + 1e-9
    )
    return corpus_norms @ query_norm


def compute_facet_scores(
    paper_idx: int,
    query_vectors: dict[str, npt.NDArray[np.floating]],
    data: ExplorerData,
) -> FacetScores:
    """Compute per-facet cosine similarity for a paper against query vectors."""
    scores: dict[str, float] = {}
    for facet in FACET_NAMES:
        paper_vec = data.corpus_vectors[facet][paper_idx]
        query_vec = query_vectors[facet]
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        p_norm = paper_vec / (np.linalg.norm(paper_vec) + 1e-9)
        scores[facet] = float(np.dot(q_norm, p_norm))
    return FacetScores(**scores)


def _make_paper_node(
    idx: int,
    query_score: float,
    query_vectors: dict[str, npt.NDArray[np.floating]],
    data: ExplorerData,
    contributing_facets: list[str] | None = None,
) -> PaperNode:
    """Build a PaperNode from an index, score, and preloaded data."""
    cid = int(data.corpusids[idx])
    meta = data.metadata.get(cid)
    facets_obj = data.facets_map.get(cid)

    facets: dict[str, list[str]] = {}
    if facets_obj:
        facets = {
            "problems": list(facets_obj.problems),
            "methods": list(facets_obj.methods),
            "results": list(facets_obj.results),
            "contributions": list(facets_obj.contributions),
        }

    facet_scores = compute_facet_scores(idx, query_vectors, data)

    return PaperNode(
        corpusid=cid,
        title=meta.title if meta else "",
        authors=meta.authors if meta else [],
        year=meta.year if meta else None,
        venue=meta.venue if meta else None,
        facets=facets,
        query_score=round(query_score, 4),
        facet_scores=facet_scores,
        matched_facets=sorted(contributing_facets or []),
    )


def _encode_query(
    query: str, encoder: TextEncoder
) -> dict[str, npt.NDArray[np.floating]]:
    """Encode a text query into per-facet query vectors.

    Uses the same vector for all facets since the query is free-text.
    """
    vec = encoder.encode([query])[0]
    return dict.fromkeys(FACET_NAMES, vec)


def _rerank_results(
    query_text: str,
    candidate_indices: list[int],
    data: ExplorerData,
    reranker: Reranker,
    top_k: int,
) -> list[tuple[int, float]]:
    """Rerank candidate papers using the ModernBERT cross-encoder.

    Formats each (query_text, candidate_facets) pair and scores with
    the reranker. Returns (index, score) pairs sorted by score descending.

    Args:
        query_text: The raw query string (used as Paper A).
        candidate_indices: Corpus indices of candidate papers.
        data: Explorer data with facets and metadata.
        reranker: Loaded reranker model.
        top_k: Number of results to return after reranking.

    Returns:
        List of (corpus_index, reranker_score) tuples, sorted by score.
    """
    from genret.paper_parsing import Facets  # noqa: PLC0415

    t0 = time.monotonic()

    # Build a pseudo-facets object from the query text
    query_facets = Facets(
        corpusid=CorpusId(0),
        problems=(query_text,),
        methods=(),
        results=(),
        contributions=(),
    )
    query_title = query_text

    texts: list[str] = []
    valid_indices: list[int] = []

    for idx in candidate_indices:
        cid = int(data.corpusids[idx])
        cand_facets = data.facets_map.get(cid)
        meta = data.metadata.get(cid)
        if cand_facets is None or meta is None:
            continue
        texts.append(
            format_paper_pair(query_title, query_facets, meta.title, cand_facets)
        )
        valid_indices.append(idx)

    if not texts:
        return [(idx, 0.0) for idx in candidate_indices[:top_k]]

    t_format = time.monotonic()
    logger.info("Search rerank: formatted %d pairs in %.2fs", len(texts), t_format - t0)

    scores = reranker.score_pairs(texts, batch_size=reranker.inference_batch_size())

    t_score = time.monotonic()
    logger.info(
        "Search rerank: scored %d pairs in %.2fs (batch_size=%d)",
        len(texts),
        t_score - t_format,
        reranker.inference_batch_size(),
    )

    scored = sorted(
        zip(valid_indices, scores.tolist()), key=lambda x: x[1], reverse=True
    )
    return scored[:top_k]


def search_by_text(
    query: str,
    data: ExplorerData,
    encoder: TextEncoder,
    top_k: int = 20,
    *,
    reranker: Reranker | None = None,
) -> tuple[SearchResponse, dict[str, npt.NDArray[np.floating]]]:
    """Search the corpus by free-text query using cosine + RRF + optional reranking."""
    query_vectors = _encode_query(query, encoder)

    retrieve_k = top_k * 3 if reranker else top_k

    # Per-facet cosine ranking
    per_facet_rankings: dict[str, list[int]] = {}
    for facet in FACET_NAMES:
        scores = _cosine_scores(query_vectors[facet], data.corpus_vectors[facet])
        ranked_indices = np.argsort(scores)[::-1][: retrieve_k * 5].tolist()
        per_facet_rankings[facet] = ranked_indices

    fused = fuse_rrf(per_facet_rankings)
    candidates = fused[:retrieve_k]

    facets_by_index = {c.index: c.facets for c in candidates}
    if reranker is not None:
        candidate_indices = [c.index for c in candidates]
        reranked = _rerank_results(query, candidate_indices, data, reranker, top_k)
        papers = [
            _make_paper_node(idx, score, query_vectors, data, facets_by_index.get(idx))
            for idx, score in reranked
        ]
        result_indices = {idx for idx, _ in reranked}
    else:
        top_results = candidates[:top_k]
        papers = [
            _make_paper_node(c.index, c.score, query_vectors, data, c.facets)
            for c in top_results
        ]
        result_indices = {c.index for c in top_results}

    # Build edges between result papers from all facet graphs
    edges = _collect_edges(result_indices, data)

    query_id = str(uuid.uuid4())
    return (
        SearchResponse(papers=papers, edges=edges, query_id=query_id),
        query_vectors,
    )


def _rerank(
    reranker: Reranker,
    data: ExplorerData,
    corpusid: int,
    fused: list[FusedCandidate],
    top_k: int,
) -> list[FusedCandidate]:
    import torch  # noqa: PLC0415

    t0 = time.monotonic()

    source_facets = data.facets_map.get(corpusid)
    source_meta = data.metadata.get(corpusid)

    if not source_facets or not source_meta:
        return fused[:top_k]

    rerank_pool = fused[: top_k * 3]
    pairs: list[str] = []
    valid_indices: list[int] = []

    for i, c in enumerate(rerank_pool):
        cand_cid = int(data.corpusids[c.index])
        cand_facets = data.facets_map.get(cand_cid)
        cand_meta = data.metadata.get(cand_cid)

        if cand_facets and cand_meta:
            pairs.append(
                format_paper_pair(
                    source_meta.title, source_facets, cand_meta.title, cand_facets
                )
            )
            valid_indices.append(i)

    if not pairs:
        return fused[:top_k]

    t_format = time.monotonic()
    logger.info("Expand rerank: formatted %d pairs in %.2fs", len(pairs), t_format - t0)

    with torch.inference_mode():
        scores = reranker.score_pairs(pairs, batch_size=reranker.inference_batch_size())

    t_score = time.monotonic()
    logger.info(
        "Expand rerank: scored %d pairs in %.2fs (batch_size=%d)",
        len(pairs),
        t_score - t_format,
        reranker.inference_batch_size(),
    )
    scored = sorted(
        zip(valid_indices, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [rerank_pool[i] for i, _ in scored[:top_k]]


def query_vectors_from_paper(
    corpusid: int, data: ExplorerData
) -> dict[str, npt.NDArray[np.floating]] | None:
    """Return per-facet vectors for an in-graph paper.

    Suitable as a stand-in for the cached query when expanding from
    that paper. Returns None if the paper is not present in the
    retrieval graph.
    """

    idx = data.cid_to_idx.get(corpusid)
    if idx is None:
        return None
    return {facet: data.corpus_vectors[facet][idx] for facet in FACET_NAMES}


def expand_node(
    corpusid: int,
    data: ExplorerData,
    query_vectors: dict[str, npt.NDArray[np.floating]],
    facet: str | None = None,
    top_k: int = 5,
    *,
    reranker: Reranker | None = None,
    known_corpusids: Sequence[int] = (),
) -> ExpandResponse:
    """Expand a node by walking its facet edges and optionally reranking.

    Papers in ``known_corpusids`` (and the source paper itself) are
    excluded from the results: in a dense neighbourhood most of a node's
    nearest neighbours are already displayed, so without this an
    expansion appears to do nothing. The known set is also used to link
    the new papers to the papers already on screen.
    """
    idx = data.cid_to_idx.get(corpusid)
    if idx is None:
        return ExpandResponse(new_papers=[], new_edges=[])

    facets_to_walk = [facet] if facet and facet in data.graphs else list(data.graphs)

    # One traversal per facet, reused for both provenance and rankings.
    # Ask for enough candidates that top_k survive dropping known papers.
    max_candidates = top_k * 3 + len(known_corpusids)
    all_neighbours: dict[int, list[tuple[str, float]]] = {}
    per_facet_rankings: dict[str, list[int]] = {}
    for facet_name in facets_to_walk:
        graph = data.graphs[facet_name]
        neighbours = traverse_neighbours(graph, idx, k=1, max_candidates=max_candidates)
        per_facet_rankings[facet_name] = [nbr_idx for nbr_idx, _ in neighbours]
        for nbr_idx, edge_score in neighbours:
            all_neighbours.setdefault(nbr_idx, []).append((facet_name, edge_score))

    fused = fuse_rrf(per_facet_rankings)

    known_indices = {
        i for cid in known_corpusids if (i := data.cid_to_idx.get(cid)) is not None
    }
    excluded = known_indices | {idx}
    fresh = [c for c in fused if c.index not in excluded]

    if reranker is not None:
        top_fused = _rerank(reranker, data, corpusid, fresh, top_k)
    else:
        top_fused = fresh[:top_k]

    papers: list[PaperNode] = []
    edges: list[ProvenanceEdge] = []

    for candidate in top_fused:
        nbr_idx = candidate.index
        papers.append(
            _make_paper_node(
                nbr_idx, candidate.score, query_vectors, data, candidate.facets
            )
        )

        # Record provenance edges from the source node
        for facet_name, edge_score in all_neighbours.get(nbr_idx, []):
            edges.append(
                ProvenanceEdge(
                    source=corpusid,
                    target=int(data.corpusids[nbr_idx]),
                    facet=facet_name,
                    score=round(float(edge_score), 4),
                )
            )

    # Connect the new papers to the rest of the displayed graph, so they
    # land in context instead of dangling off the expanded node. Only
    # edges touching a new paper are returned; the caller already has the
    # edges among its known papers.
    new_indices = {c.index for c in top_fused}
    if new_indices:
        new_cids = {int(data.corpusids[i]) for i in new_indices}
        walked = set(facets_to_walk)
        seen = {(e.source, e.target, e.facet) for e in edges}
        for edge in _collect_edges(new_indices | known_indices, data):
            # Honour the facet filter: "expand via methods" should not
            # pull in other facets' edges.
            if edge.facet not in walked:
                continue
            if edge.source not in new_cids and edge.target not in new_cids:
                continue
            if (edge.source, edge.target, edge.facet) in seen:
                continue
            seen.add((edge.source, edge.target, edge.facet))
            edges.append(edge)

    return ExpandResponse(new_papers=papers, new_edges=edges)


def get_inter_edges(corpusids: list[int], data: ExplorerData) -> list[ProvenanceEdge]:
    """Get all edges between the given papers across all facet graphs."""
    indices = {
        idx for cid in corpusids if (idx := data.cid_to_idx.get(cid)) is not None
    }
    return _collect_edges(indices, data)


def _collect_edges(indices: set[int], data: ExplorerData) -> list[ProvenanceEdge]:
    """Collect all edges between a set of paper indices from all facet graphs."""
    edges: list[ProvenanceEdge] = []
    idx_set = indices

    for facet_name, graph in data.graphs.items():
        for idx in idx_set:
            start = int(graph.indptr[idx])
            end = int(graph.indptr[idx + 1])
            row_indices = graph.indices[start:end]
            row_data = graph.data[start:end]

            for j, score in zip(row_indices, row_data):
                j_int = int(j)
                if j_int in idx_set and j_int > idx:
                    edges.append(
                        ProvenanceEdge(
                            source=int(data.corpusids[idx]),
                            target=int(data.corpusids[j_int]),
                            facet=facet_name,
                            score=round(float(score), 4),
                        )
                    )

    return edges
