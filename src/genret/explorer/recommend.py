"""Recommendation orchestrator: path detection → Aleth → Lyra → rerank → format.

Handles three paper paths:
- In-graph (11k): facets preloaded, BFS + RRF + rerank
- In-corpus (~44k): S2ORC fulltext → Aleth extraction → online retrieval + rerank
- Out-of-corpus: Silver acquisition → Aleth extraction → online retrieval + rerank
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

from genret.explorer.aleth import embed_facets, extract_facets, get_or_extract_facets
from genret.explorer.data_loading import FACET_NAMES, ExplorerData, PaperMeta
from genret.explorer.encoder import TextEncoder
from genret.explorer.facet_cache import FacetCache
from genret.explorer.models import (
    QueryPaper,
    RecommendEdge,
    RecommendPaper,
    RecommendResult,
    SSEError,
    SSEStatus,
)
from genret.explorer.silver import AcquiredPaper, acquire_paper, load_from_cache
from genret.paper_parsing import Facets
from genret.processing.facet_retrieval import fuse_rrf, traverse_neighbours
from genret.processing.online_retrieval import CorpusIndex, retrieve_new_paper
from genret.processing.reranker import Reranker, format_paper_pair
from genret.s2.api import SemanticScholarAPI
from genret.types import CorpusId

logger = logging.getLogger(__name__)


class RecommendSource(StrEnum):
    """Supported query source categories for recommendations."""

    IN_GRAPH = "in_graph"
    IN_CORPUS = "in_corpus"
    OUT_OF_CORPUS = "out_of_corpus"


class SSEEventType(StrEnum):
    """Types of SSE events emmited."""

    STATUS = "status"
    RESULT = "result"
    ERROR = "error"


@dataclass(frozen=True, kw_only=True)
class SSEEvent:
    """Typed wrapper for an SSE event."""

    event: SSEEventType
    data: SSEStatus | RecommendResult | SSEError

    @classmethod
    def status(cls, stage: str, message: str) -> Self:
        """Build a status SSE event."""
        return cls(
            event=SSEEventType.STATUS, data=SSEStatus(stage=stage, message=message)
        )

    @classmethod
    def error(cls, message: str) -> Self:
        """Build an error SSE event."""
        return cls(event=SSEEventType.ERROR, data=SSEError(message=message))


@dataclass(frozen=True, kw_only=True)
class RecommendationPath:
    """Resolved source path for a query paper."""

    is_in_graph: bool
    is_in_corpus: bool
    is_out_of_corpus: bool

    @property
    def source(self) -> RecommendSource:
        """Determine paper source from status conditions."""
        if self.is_in_graph:
            return RecommendSource.IN_GRAPH
        elif self.is_in_corpus:
            return RecommendSource.IN_CORPUS
        else:
            return RecommendSource.OUT_OF_CORPUS


@dataclass(frozen=True, kw_only=True)
class ResolvedQuery:
    """Query metadata and facets after path-specific resolution."""

    path: RecommendationPath
    meta: PaperMeta | None
    acquired: AcquiredPaper | None
    facets: Facets


@dataclass(frozen=True, kw_only=True)
class RetrievalResult:
    """Candidate recommendations and provenance edges before reranking."""

    candidate_cids: Sequence[CorpusId]
    candidate_scores: Sequence[float]
    edges: Sequence[RecommendEdge]


def _resolve_recommendation_path(
    corpusid: int, data: ExplorerData, s2orc_index: Mapping[int, tuple[str, int]]
) -> RecommendationPath:
    """Resolve which recommendation path applies to a query paper."""
    is_in_graph = corpusid in data.cid_to_idx
    is_in_corpus = corpusid in s2orc_index
    is_out_of_corpus = not is_in_graph and not is_in_corpus

    return RecommendationPath(
        is_in_graph=is_in_graph,
        is_in_corpus=is_in_corpus,
        is_out_of_corpus=is_out_of_corpus,
    )


def _validate_recommendation_request(
    corpusid: int,
    path: RecommendationPath,
    meta: PaperMeta | None,
) -> SSEError | None:
    """Validate metadata requirements for a recommendation request."""
    if meta is None and not path.is_out_of_corpus:
        return SSEError(message=f"Paper {corpusid} not found in metadata")

    return None


def _rerank_candidates(
    query_title: str,
    query_facets: Facets,
    candidate_cids: Sequence[CorpusId],
    candidate_scores: Sequence[float],
    metadata: Mapping[int, PaperMeta],
    facets_map: Mapping[int, Facets],
    reranker: Reranker,
    top_k: int = 20,
) -> list[tuple[CorpusId, float]]:
    """Rerank candidates with the cross-encoder.

    Args:
        query_title: Title of the query paper.
        query_facets: Facets of the query paper.
        candidate_cids: Corpus IDs of candidates.
        candidate_scores: Pre-rerank scores.
        metadata: Paper metadata.
        facets_map: All available facets.
        reranker: Cross-encoder model.
        top_k: Number of results to return.

    Returns:
        List of (corpusid, reranker_score) sorted by score descending.
    """
    import torch  # noqa: PLC0415

    t0 = time.monotonic()
    pairs: list[str] = []
    valid_cids: list[CorpusId] = []

    for cid in candidate_cids:
        cand_facets = facets_map.get(cid)
        cand_meta = metadata.get(cid)
        if cand_facets is None or cand_meta is None:
            continue

        pairs.append(
            format_paper_pair(query_title, query_facets, cand_meta.title, cand_facets)
        )
        valid_cids.append(cid)

    if not pairs:
        return list(zip(candidate_cids[:top_k], candidate_scores[:top_k]))

    t_format = time.monotonic()
    logger.info("Rerank: formatted %d pairs in %.2fs", len(pairs), t_format - t0)

    with torch.inference_mode():
        scores = reranker.score_pairs(pairs, batch_size=reranker.inference_batch_size())

    t_score = time.monotonic()
    logger.info(
        "Rerank: scored %d pairs in %.2fs (batch_size=%d)",
        len(pairs),
        t_score - t_format,
        reranker.inference_batch_size(),
    )

    scored = sorted(zip(valid_cids, scores.tolist()), key=lambda x: x[1], reverse=True)

    t_total = time.monotonic()
    logger.info("Rerank: total %.2fs", t_total - t0)
    return scored[:top_k]


def _title_from_resolved(resolved_query: ResolvedQuery) -> str:
    if resolved_query.meta:
        return resolved_query.meta.title
    elif resolved_query.acquired is not None:
        return resolved_query.acquired.graph.title
    else:
        return ""


def _build_query_paper(resolved_query: ResolvedQuery, corpusid: int) -> QueryPaper:
    """Build the query paper payload returned in the final SSE result."""
    title = _title_from_resolved(resolved_query)

    authors: Sequence[str] = []
    if resolved_query.meta:
        authors = resolved_query.meta.authors
    elif resolved_query.acquired is not None:
        authors = resolved_query.acquired.graph.authors

    return QueryPaper(
        corpusid=corpusid,
        title=title,
        authors=authors,
        source=resolved_query.path.source,
        facets=resolved_query.facets.to_dict(),
    )


def _build_recommend_papers(
    ranked: Sequence[tuple[CorpusId, float]],
    metadata: Mapping[int, PaperMeta],
    facets_map: Mapping[int, Facets],
    edges: Sequence[RecommendEdge] = (),
) -> list[RecommendPaper]:
    """Build RecommendPaper list from ranked corpus IDs."""
    contributing: dict[int, set[str]] = {}
    for edge in edges:
        contributing.setdefault(edge.target, set()).add(edge.facet)

    papers: list[RecommendPaper] = []
    for cid, score in ranked:
        meta = metadata.get(cid)
        facets = facets_map.get(cid)
        papers.append(
            RecommendPaper(
                corpusid=cid,
                title=meta.title if meta else "",
                authors=meta.authors if meta else [],
                year=meta.year if meta else None,
                venue=meta.venue if meta else None,
                facets=facets.to_dict() if facets else dict[str, list[str]](),
                score=round(float(score), 4),
                matched_facets=sorted(contributing.get(cid, set())),
            )
        )
    return papers


async def _acquire_out_of_corpus(
    corpusid: int, s2_api: SemanticScholarAPI, silver_cache_dir: Path
) -> AcquiredPaper | None:
    """Acquire an out-of-corpus paper via Silver.

    Checks the Silver cache first, then fetches paper metadata from the S2 API
    batch endpoint and runs the full acquisition pipeline.

    Args:
        corpusid: Corpus ID of the paper.
        s2_api: Semantic Scholar API client.
        silver_cache_dir: Silver cache directory.

    Returns:
        AcquiredPaper or None on failure.
    """
    # Check Silver cache first
    cached = load_from_cache(silver_cache_dir, corpusid)
    if cached is not None:
        logger.info("Silver cache hit for %d", corpusid)
        return cached

    # Fetch paper metadata from S2 API
    try:
        papers = await s2_api.get_paper_batch(
            [f"CorpusId:{corpusid}"],
            fields=[
                "paperId",
                "corpusId",
                "title",
                "abstract",
                "year",
                "venue",
                "openAccessPdf",
                "externalIds",
                "authors",
            ],
        )
    except Exception:
        logger.exception("S2 API batch fetch failed for %d", corpusid)
        return None

    if not papers or papers[0] is None:
        logger.error("Paper %d not found in S2 API", corpusid)
        return None

    s2_paper = papers[0]

    # Will be always be true if the S2 API call above succeeds
    assert s2_api.client is not None
    try:
        # Run the full Silver acquisition pipeline
        return await acquire_paper(s2_paper, silver_cache_dir, s2_api.client, s2_api)
    except (ValueError, RuntimeError):
        logger.exception("Silver acquisition failed for %d", corpusid)
        return None


def _resolve_in_graph_query(
    corpusid: int,
    path: RecommendationPath,
    meta: PaperMeta | None,
    facets_map: Mapping[int, Facets],
) -> ResolvedQuery | SSEError:
    """Resolve an in-graph query using preloaded facets."""
    facets = facets_map.get(corpusid)
    if facets is None:
        return SSEError(message=f"Paper {corpusid} is in graph but has no facets")

    return ResolvedQuery(path=path, meta=meta, acquired=None, facets=facets)


async def _resolve_in_corpus_query(
    corpusid: int,
    path: RecommendationPath,
    meta: PaperMeta | None,
    facet_cache: FacetCache,
    s2orc_index: dict[int, tuple[str, int]],
    s2orc_dir: Path,
    extraction_model: str,
) -> ResolvedQuery | SSEError:
    """Resolve an in-corpus query by extracting or loading cached facets."""
    facets = await get_or_extract_facets(
        corpusid,
        facet_cache,
        s2orc_index,
        s2orc_dir,
        extraction_model,
        title=meta.title if meta else None,
    )
    if facets is None:
        return SSEError(message=f"Failed to extract facets for paper {corpusid}")

    return ResolvedQuery(path=path, meta=meta, acquired=None, facets=facets)


async def _extract_out_of_corpus_query(
    corpusid: int,
    path: RecommendationPath,
    acquired: AcquiredPaper,
    facet_cache: FacetCache,
    extraction_model: str,
) -> ResolvedQuery | SSEError:
    """Extract facets for an already acquired out-of-corpus paper."""
    facets = await extract_facets(corpusid, acquired.graph, extraction_model)
    if facets is None:
        return SSEError(message=f"Failed to extract facets for paper {corpusid}")

    facet_cache.put(facets)
    return ResolvedQuery(path=path, meta=None, acquired=acquired, facets=facets)


def _retrieve_in_graph_candidates(
    corpusid: int,
    data: ExplorerData,
    top_k_fused: int,
    bfs_hops: int,
    max_bfs_candidates: int,
) -> RetrievalResult:
    """Retrieve candidates for an in-graph query paper."""
    idx = data.cid_to_idx[corpusid]
    per_facet_rankings: dict[str, list[int]] = {}
    edges: list[RecommendEdge] = []

    for facet_name in FACET_NAMES:
        if facet_name not in data.graphs:
            continue

        graph = data.graphs[facet_name]
        neighbours = traverse_neighbours(
            graph, idx, k=bfs_hops, max_candidates=max_bfs_candidates
        )
        per_facet_rankings[facet_name] = [n_idx for n_idx, _ in neighbours]

        for n_idx, edge_score in neighbours:
            edges.append(
                RecommendEdge(
                    target=int(data.corpusids[n_idx]),
                    facet=facet_name,
                    score=round(float(edge_score), 4),
                )
            )

    fused = fuse_rrf(per_facet_rankings)[:top_k_fused]
    return RetrievalResult(
        candidate_cids=[CorpusId(int(data.corpusids[c.index])) for c in fused],
        candidate_scores=[c.score for c in fused],
        edges=edges,
    )


def _retrieve_out_of_graph_candidates(
    facets: Facets,
    data: ExplorerData,
    corpus_index: CorpusIndex,
    embedding_model: TextEncoder,
    top_k_fused: int,
) -> RetrievalResult:
    """Retrieve candidates for in-corpus and out-of-corpus query papers."""
    item_embeddings = embed_facets(facets, embedding_model)
    fused = retrieve_new_paper(
        query_facets=facets,
        query_embeddings=item_embeddings,
        corpus_index=corpus_index,
        edge_graphs=data.graphs,
        dense_metric="mean_pool",
        alpha=0.75,
        beta=0.0,
        gamma=0.25,
        top_n=50,
    )[:top_k_fused]

    candidate_cids = [CorpusId(int(corpus_index.corpusids[c.index])) for c in fused]
    candidate_scores = [c.score for c in fused]
    edges = [
        RecommendEdge(
            target=int(corpus_index.corpusids[candidate.index]),
            facet=facet_name,
            score=round(candidate.score, 4),
        )
        for candidate in fused
        for facet_name in candidate.facets
    ]

    return RetrievalResult(
        candidate_cids=candidate_cids,
        candidate_scores=candidate_scores,
        edges=edges,
    )


def _merge_rerank_facets(
    candidate_cids: Sequence[CorpusId],
    graph_facets: Mapping[int, Facets],
    facet_cache: FacetCache,
) -> dict[int, Facets]:
    """Merge graph facets with any cached candidate facets for reranking."""
    return dict(graph_facets) | {
        cid: cached for cid in candidate_cids if (cached := facet_cache.get(cid))
    }


def _rank_candidates(
    resolved_query: ResolvedQuery,
    retrieval: RetrievalResult,
    reranker: Reranker | None,
    data: ExplorerData,
    facet_cache: FacetCache,
    all_metadata: Mapping[int, PaperMeta],
    top_k_final: int,
) -> list[tuple[CorpusId, float]]:
    """Optionally rerank candidates and return the final ranked list."""
    if reranker is None or not retrieval.candidate_cids:
        return list(
            zip(
                retrieval.candidate_cids[:top_k_final],
                retrieval.candidate_scores[:top_k_final],
            )
        )

    return _rerank_candidates(
        query_title=_title_from_resolved(resolved_query),
        facets_map=_merge_rerank_facets(
            retrieval.candidate_cids, data.facets_map, facet_cache
        ),
        query_facets=resolved_query.facets,
        candidate_cids=retrieval.candidate_cids,
        candidate_scores=retrieval.candidate_scores,
        metadata=all_metadata,
        reranker=reranker,
        top_k=top_k_final,
    )


def _build_recommend_result(
    ranked: Sequence[tuple[CorpusId, float]],
    retrieval: RetrievalResult,
    query_paper: QueryPaper,
    all_metadata: Mapping[int, PaperMeta],
    facets_map: Mapping[int, Facets],
) -> RecommendResult:
    """Build the final recommendation result payload."""
    result_cids = {cid for cid, _ in ranked}
    return RecommendResult(
        papers=_build_recommend_papers(
            ranked, all_metadata, facets_map, retrieval.edges
        ),
        edges=[edge for edge in retrieval.edges if edge.target in result_cids],
        query_paper=query_paper,
    )


async def recommend(
    corpusid: int,
    data: ExplorerData,
    corpus_index: CorpusIndex,
    embedding_model: TextEncoder,
    reranker: Reranker | None,
    facet_cache: FacetCache,
    s2orc_index: dict[int, tuple[str, int]],
    s2orc_dir: Path,
    extraction_model: str,
    all_metadata: dict[int, PaperMeta],
    s2_api: SemanticScholarAPI | None,
    silver_cache_dir: Path,
    top_k_fused: int = 50,
    top_k_final: int = 8,
    bfs_hops: int = 2,
    max_bfs_candidates: int = 150,
) -> AsyncIterator[SSEEvent]:
    """Run the recommendation pipeline, yielding SSE events.

    Args:
        corpusid: Query paper corpus ID.
        data: Preloaded explorer data (11k graph papers).
        corpus_index: Corpus index for online retrieval.
        embedding_model: Text encoder for embedding facets.
        reranker: Optional cross-encoder for reranking.
        facet_cache: Aleth facet cache.
        s2orc_index: S2ORC fulltext index.
        s2orc_dir: Path to S2ORC shard directory.
        extraction_model: LLM model for Aleth extraction.
        all_metadata: Metadata for all papers (~78k).
        s2_api: Semantic Scholar API client for Silver, or None when Silver
            (out-of-corpus acquisition) is disabled.
        silver_cache_dir: Silver cache directory.
        top_k_fused: Number of top items to keep from RRF.
        top_k_final: Number of top items to return at the end after reranking.
        bfs_hops: Number of BFS hops.
        max_bfs_candidates: Number of top item to keep from BFS traversal.
    """
    yield SSEEvent.status("checking", "Looking up paper...")

    path = _resolve_recommendation_path(corpusid, data, s2orc_index)
    meta = all_metadata.get(corpusid)

    if error := _validate_recommendation_request(corpusid, path, meta):
        yield SSEEvent(event=SSEEventType.ERROR, data=error)
        return

    resolved_query: ResolvedQuery | SSEError
    if path.is_in_graph:
        resolved_query = _resolve_in_graph_query(
            corpusid=corpusid,
            path=path,
            meta=meta,
            facets_map=data.facets_map,
        )
    elif path.is_out_of_corpus:
        if s2_api is None:
            yield SSEEvent.error(
                f"Paper {corpusid} is not in the corpus, and out-of-corpus"
                " acquisition is disabled on this server."
            )
            return

        yield SSEEvent.status("acquiring", "Acquiring paper via Silver...")
        acquired = await _acquire_out_of_corpus(corpusid, s2_api, silver_cache_dir)
        if acquired is None:
            yield SSEEvent.error(
                f"Failed to acquire out-of-corpus paper {corpusid} via Silver."
            )
            return

        yield SSEEvent.status("extracting", "Extracting facets...")
        resolved_query = await _extract_out_of_corpus_query(
            corpusid=corpusid,
            path=path,
            acquired=acquired,
            facet_cache=facet_cache,
            extraction_model=extraction_model,
        )
        if isinstance(resolved_query, SSEError):
            yield SSEEvent(event=SSEEventType.ERROR, data=resolved_query)
            return
    else:
        yield SSEEvent.status("extracting", "Extracting facets...")
        resolved_query = await _resolve_in_corpus_query(
            corpusid=corpusid,
            path=path,
            meta=meta,
            facet_cache=facet_cache,
            s2orc_index=s2orc_index,
            s2orc_dir=s2orc_dir,
            extraction_model=extraction_model,
        )

    if isinstance(resolved_query, SSEError):
        yield SSEEvent(event=SSEEventType.ERROR, data=resolved_query)
        return

    query_paper = _build_query_paper(resolved_query, corpusid)

    yield SSEEvent.status("retrieving", "Finding related papers...")

    if path.is_in_graph:
        retrieval = await asyncio.to_thread(
            _retrieve_in_graph_candidates,
            corpusid=corpusid,
            data=data,
            top_k_fused=top_k_fused,
            bfs_hops=bfs_hops,
            max_bfs_candidates=max_bfs_candidates,
        )
    else:
        retrieval = await asyncio.to_thread(
            _retrieve_out_of_graph_candidates,
            facets=resolved_query.facets,
            data=data,
            corpus_index=corpus_index,
            embedding_model=embedding_model,
            top_k_fused=top_k_fused,
        )

    if reranker is not None and retrieval.candidate_cids:
        yield SSEEvent.status("reranking", "Reranking results...")

    ranked = await asyncio.to_thread(
        _rank_candidates,
        resolved_query=resolved_query,
        retrieval=retrieval,
        reranker=reranker,
        data=data,
        facet_cache=facet_cache,
        all_metadata=all_metadata,
        top_k_final=top_k_final,
    )

    yield SSEEvent(
        event=SSEEventType.RESULT,
        data=_build_recommend_result(
            ranked=ranked,
            retrieval=retrieval,
            query_paper=query_paper,
            all_metadata=all_metadata,
            facets_map=data.facets_map,
        ),
    )
