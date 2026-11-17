"""FastAPI server for the graph explorer."""

import asyncio
import logging
import os
from collections import OrderedDict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from genret.explorer.data_loading import (
    ExplorerData,
    PaperMeta,
    build_s2orc_index,
    build_searchable_papers,
    load_explorer_data,
    load_metadata,
)
from genret.explorer.encoder import DEFAULT_ONNX_DIR, TextEncoder, load_encoder
from genret.explorer.explain import ExplanationCache, explain_connection
from genret.explorer.facet_cache import FacetCache
from genret.explorer.graph_service import (
    expand_node,
    get_inter_edges,
    query_vectors_from_paper,
    search_by_text,
)
from genret.explorer.models import (
    EdgesRequest,
    EdgesResponse,
    ExpandFromRequest,
    ExpandRequest,
    ExpandResponse,
    ExplainRequest,
    ExplainResponse,
    PaperDetail,
    SearchPaperResult,
    SearchPapersRequest,
    SearchPapersResponse,
    SearchRequest,
    SearchResponse,
)
from genret.explorer.paper_search import SearchablePaper, search_papers
from genret.explorer.rate_limit import RateLimitConfig, RateLimiter
from genret.explorer.recommend import recommend
from genret.paper_parsing import Facets
from genret.processing.online_retrieval import CorpusIndex, load_corpus_index
from genret.processing.reranker import (
    LLMReranker,
    LocalReranker,
    Reranker,
    RerankerConfig,
)
from genret.s2.api import SemanticScholarAPI

logger = logging.getLogger(__name__)

MAX_QUERY_CACHE = 100
"""Maximum number of query vectors to cache."""


V2_PROCESSED = Path("data/s2orc-nlp-v2/splits-connected-10k-v2/processed")
"""Default processed directory for the v2 dataset."""

V2_EDGES = Path("output/v2_crossfacet_meanpool")
"""Default mean_pool + cross-facet edge directory for the v2 dataset."""


@dataclass
class ServerConfig:
    """Configuration for the explorer server, set from CLI arguments."""

    edge_dir: Path = V2_EDGES
    dense_dir: Path = V2_PROCESSED
    facets_path: Path = V2_PROCESSED / "facets.jsonl.zst"
    metadata_dir: Path = Path("data/s2orc-nlp-v2/papers")
    s2orc_dir: Path = Path("data/s2orc-nlp-v2/s2orc")
    model_dir: Path = Path("output/phase2_modernbert_150k/checkpoint")
    rerank: bool = False
    rerank_model: str = "local"
    """Reranker to use: ``"local"`` for ModernBERT, or an LLM model name."""
    rerank_base_url: str | None = None
    """Override API base URL for the LLM reranker (e.g. OpenRouter)."""
    rerank_api_key: str | None = None
    """Override API key for the LLM reranker."""
    rerank_device: str = "cpu"
    extraction_model: str = "gpt-5-mini"
    explanation_model: str = "gpt-5.6-luna"
    """LLM for on-demand connection explanations. Luna is chat-tuned,
    fast, and cheap; unlike extraction there is no consistency constraint
    with the offline corpus facets."""
    cache_dir: Path = Path("cache")
    dev_mode: bool = False
    encoder_backend: str = "onnx"
    """Text encoder backend: ``"onnx"`` (no torch) or ``"torch"``."""
    onnx_dir: Path = DEFAULT_ONNX_DIR
    """Directory with the exported ONNX encoder (onnx backend only)."""
    silver: bool = False
    """Enable Silver out-of-corpus paper acquisition and live S2 title
    search. Off by default: the demo serves in-graph + in-corpus papers
    only, and the server then needs no SEMANTIC_SCHOLAR_API_KEY."""
    frontend_dir: Path | None = None
    """Built frontend to serve. None resolves to ``frontend/dist`` beside
    the source tree, which only works from a source checkout: an
    installed package sits in site-packages, so deployments must pass
    this explicitly."""
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    """Bounds on the endpoints that call an LLM when uncached."""


@dataclass
class ServerState:
    """Runtime state populated during lifespan."""

    data: ExplorerData
    encoder: TextEncoder
    query_cache: OrderedDict[str, dict[str, npt.NDArray[np.floating]]]
    all_metadata: dict[int, PaperMeta]
    searchable_papers: list[SearchablePaper]
    s2orc_index: dict[int, tuple[str, int]]
    corpus_index: CorpusIndex
    facet_cache: FacetCache
    explain_cache: ExplanationCache
    reranker: Reranker | None
    s2_api: SemanticScholarAPI | None
    """S2 API client for Silver; None when Silver is disabled."""
    silver_cache_dir: Path
    rate_limiter: RateLimiter


def _get_state(request: Request) -> ServerState:
    """Extract the server state from the request, raising if not loaded."""
    state: ServerState | None = request.app.state.server
    assert state is not None, "Explorer data not loaded"
    return state


def _cache_query(
    state: ServerState,
    query_id: str,
    vectors: dict[str, npt.NDArray[np.floating]],
) -> None:
    """Cache query vectors, evicting the oldest if the cache is full."""
    state.query_cache[query_id] = vectors
    if len(state.query_cache) > MAX_QUERY_CACHE:
        state.query_cache.popitem(last=False)


def _get_query_vectors(
    state: ServerState, query_id: str
) -> dict[str, npt.NDArray[np.floating]]:
    """Retrieve cached query vectors, raising 404 if expired."""
    vectors = state.query_cache.get(query_id)
    if vectors is None:
        raise HTTPException(
            status_code=404, detail=f"Query {query_id} not found or expired"
        )
    return vectors


def _enforce_rate_limit(state: ServerState, request: Request) -> None:
    """Reject the request when the caller is over the paid-endpoint quota.

    Behind a reverse proxy the real address comes from the forwarded
    headers, which uvicorn applies to ``request.client`` when started
    with proxy headers enabled.
    """
    client = request.client.host if request.client else "unknown"
    retry_after = state.rate_limiter.check(client)
    if retry_after is None:
        return
    raise HTTPException(
        status_code=429,
        detail="Too many requests. This demo limits the endpoints that call"
        " a language model; please try again shortly.",
        headers={"Retry-After": str(int(retry_after))},
    )


async def _search_s2_api(
    state: ServerState, query: str, limit: int
) -> list[SearchPaperResult]:
    """Search the Semantic Scholar API and return results as SearchPaperResult.

    Papers already in the local corpus are tagged with their actual source.
    """
    if state.s2_api is None:
        return []

    try:
        s2_papers = await state.s2_api.search_by_title(query, limit=limit)
    except Exception:
        logger.warning("S2 API search failed for %r", query, exc_info=True)
        return []

    results: list[SearchPaperResult] = []
    for paper in s2_papers:
        cid = paper.corpus_id
        if cid is None:
            continue

        # Determine the source for this paper
        if cid in state.data.cid_to_idx:
            source = "in_graph"
        elif cid in state.s2orc_index:
            source = "in_corpus"
        else:
            source = "s2_api"

        authors = [a.name for a in (paper.authors or []) if a.name]
        results.append(
            SearchPaperResult(
                corpusid=cid,
                title=paper.title,
                authors=authors,
                year=paper.year,
                venue=paper.venue,
                source=source,
            )
        )

    return results


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Server configuration. Uses defaults if not provided.
    """
    cfg = config or ServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        if cfg.silver and not s2_key:
            raise RuntimeError(
                "SEMANTIC_SCHOLAR_API_KEY is required when Silver is enabled"
            )

        logger.info("Loading explorer data...")
        data = await asyncio.to_thread(
            load_explorer_data,
            cfg.edge_dir,
            cfg.dense_dir,
            cfg.facets_path,
            cfg.metadata_dir,
        )
        logger.info("Loading text encoder (%s)...", cfg.encoder_backend)
        encoder = await asyncio.to_thread(
            load_encoder, cfg.encoder_backend, cfg.onnx_dir
        )

        # Load all metadata (~78k papers, not just graph)
        logger.info("Loading full metadata...")
        all_metadata = await asyncio.to_thread(load_metadata, cfg.metadata_dir, None)

        # Build S2ORC index if the directory exists
        if cfg.s2orc_dir.is_dir():
            logger.info("Building S2ORC fulltext index...")
            cache_path = cfg.cache_dir / "s2orc_index.json"
            s2orc_index = await asyncio.to_thread(
                build_s2orc_index, cfg.s2orc_dir, cache_path
            )
        else:
            logger.warning("S2ORC directory not found: %s", cfg.s2orc_dir)
            s2orc_index: dict[int, tuple[str, int]] = {}

        # Build searchable papers. Without Silver, papers that are neither
        # in-graph nor fulltext-backed cannot be recommended, so they are
        # excluded from title search entirely.
        graph_cids = set(data.cid_to_idx.keys())
        fulltext_cids = None if cfg.silver else set(s2orc_index.keys())
        searchable_papers = await asyncio.to_thread(
            build_searchable_papers, all_metadata, graph_cids, fulltext_cids
        )

        # Build corpus index for online retrieval, sharing the pooled
        # vectors already loaded into ExplorerData (~70 MB saved)
        logger.info("Loading corpus index for online retrieval...")
        corpus_index = await asyncio.to_thread(
            load_corpus_index,
            cfg.dense_dir,
            cfg.facets_path,
            preloaded_vectors=(data.corpus_vectors, data.corpusids),
        )

        # Set up Aleth facet cache
        aleth_dir = cfg.cache_dir / "aleth"
        facet_cache = FacetCache(aleth_dir)
        facet_cache.validate_on_startup()

        explain_cache = ExplanationCache(cfg.cache_dir / "explain")

        # Load reranker if configured
        reranker: Reranker | None = None
        if cfg.rerank and cfg.rerank_model != "local":
            logger.info("Using LLM reranker: %s", cfg.rerank_model)
            reranker = LLMReranker(
                cfg.rerank_model,
                base_url=cfg.rerank_base_url,
                api_key=cfg.rerank_api_key,
            )
        elif cfg.rerank:
            logger.info("Loading local reranker from %s...", cfg.model_dir)
            reranker_config = RerankerConfig(
                model_name=str(cfg.model_dir),
                max_length=2048,
                device=cfg.rerank_device,
            )
            reranker = await asyncio.to_thread(
                LocalReranker.from_pretrained, reranker_config
            )

        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning(
                "OPENAI_API_KEY not set — only in-graph papers (11k) are recommendable"
            )

        # Set up Silver: S2 API client and cache directory
        s2_api: SemanticScholarAPI | None = None
        if cfg.silver:
            s2_api = SemanticScholarAPI(api_key=s2_key)
            await s2_api.start()
            logger.info("Semantic Scholar API client initialised")
        else:
            logger.info("Silver disabled: no S2 API client, local search only")

        silver_cache_dir = cfg.cache_dir / "silver"
        silver_cache_dir.mkdir(parents=True, exist_ok=True)

        state = ServerState(
            data=data,
            encoder=encoder,
            query_cache=OrderedDict(),
            all_metadata=all_metadata,
            searchable_papers=searchable_papers,
            s2orc_index=s2orc_index,
            corpus_index=corpus_index,
            facet_cache=facet_cache,
            explain_cache=explain_cache,
            reranker=reranker,
            s2_api=s2_api,
            silver_cache_dir=silver_cache_dir,
            rate_limiter=RateLimiter(cfg.rate_limit),
        )
        app.state.server = state
        app.state.server_config = cfg
        logger.info("Explorer ready")

        yield

        if state.s2_api is not None:
            await state.s2_api.stop()
        app.state.server = None

    # Outside dev mode the frontend is served from this same origin, so no
    # CORS is needed; and the OpenAPI schema and Swagger UI are turned off
    # rather than handing scanners a map of the endpoints that spend API
    # credit (/api/explain, /api/recommend).
    middleware: list[Middleware] = []
    if cfg.dev_mode:
        middleware.append(
            Middleware(
                CORSMiddleware,
                allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                allow_methods=["*"],
                allow_headers=["*"],
            )
        )

    app = FastAPI(
        title="Graft",
        lifespan=lifespan,
        middleware=middleware,
        docs_url="/docs" if cfg.dev_mode else None,
        redoc_url="/redoc" if cfg.dev_mode else None,
        openapi_url="/openapi.json" if cfg.dev_mode else None,
    )

    app.include_router(router)

    # Mount frontend static files in production mode
    frontend_dist = (
        cfg.frontend_dir
        or Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
    )
    if not cfg.dev_mode:
        if frontend_dist.is_dir():
            app.mount("/", StaticFiles(directory=str(frontend_dist), html=True))
        else:
            logger.warning(
                "No frontend at %s: the API works but the UI will 404."
                " Pass --frontend-dir.",
                frontend_dist,
            )

    return app


router = APIRouter()


@router.post("/api/search")
async def api_search(req: SearchRequest, request: Request) -> SearchResponse:
    """Search papers by text query."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    state = _get_state(request)
    response, query_vectors = await asyncio.to_thread(
        search_by_text,
        req.query,
        state.data,
        state.encoder,
        req.top_k,
        reranker=state.reranker,
    )
    _cache_query(state, response.query_id, query_vectors)
    return response


@router.post("/api/expand")
async def api_expand(req: ExpandRequest, request: Request) -> ExpandResponse:
    """Expand a node in the graph."""
    state = _get_state(request)
    query_vectors = _get_query_vectors(state, req.query_id)
    return await asyncio.to_thread(
        expand_node,
        req.corpusid,
        state.data,
        query_vectors,
        req.facet,
        req.top_k,
        reranker=state.reranker,
        known_corpusids=req.known_corpusids,
    )


@router.post("/api/expand-from")
async def api_expand_from(req: ExpandFromRequest, request: Request) -> ExpandResponse:
    """Expand a node when no cached query exists.

    Used by the paper-anchored recommend flow: the seed paper's own
    per-facet vectors stand in for the query, so the facet_scores in
    the returned papers are cosine similarities to the seed paper.
    """
    state = _get_state(request)
    query_vectors = query_vectors_from_paper(req.corpusid, state.data)
    if query_vectors is None:
        raise HTTPException(
            status_code=404,
            detail=f"Paper {req.corpusid} not in retrieval graph",
        )
    return await asyncio.to_thread(
        expand_node,
        req.corpusid,
        state.data,
        query_vectors,
        req.facet,
        req.top_k,
        reranker=state.reranker,
        known_corpusids=req.known_corpusids,
    )


@router.post("/api/edges")
async def api_edges(req: EdgesRequest, request: Request) -> EdgesResponse:
    """Get inter-edges between papers."""
    state = _get_state(request)
    edges = await asyncio.to_thread(get_inter_edges, req.corpusids, state.data)
    return EdgesResponse(edges=edges)


@router.get("/api/paper/{corpusid}")
async def api_paper_detail(corpusid: int, request: Request) -> PaperDetail:
    """Get detailed information about a paper."""
    state = _get_state(request)
    meta = state.all_metadata.get(corpusid) or state.data.metadata.get(corpusid)
    if meta is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    facets_obj = state.data.facets_map.get(corpusid)
    facets: dict[str, list[str]] = {}
    if facets_obj:
        facets = {
            "problems": list(facets_obj.problems),
            "methods": list(facets_obj.methods),
            "results": list(facets_obj.results),
            "contributions": list(facets_obj.contributions),
        }

    return PaperDetail(
        corpusid=corpusid,
        title=meta.title,
        authors=meta.authors,
        year=meta.year,
        venue=meta.venue,
        citation_count=meta.citation_count,
        url=meta.url,
        facets=facets,
    )


# --- New endpoints ---


@router.post("/api/search-papers")
async def api_search_papers(
    req: SearchPapersRequest, request: Request
) -> SearchPapersResponse:
    """Search papers by title (local substring match + optional S2 API)."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    state = _get_state(request)
    local_results = await asyncio.to_thread(
        search_papers, req.query, state.searchable_papers, req.limit
    )

    # Merge with S2 API results unless local_only or Silver is disabled
    if not req.local_only and state.s2_api is not None:
        s2_results = await _search_s2_api(state, req.query, req.limit)
        seen = {r.corpusid for r in local_results}
        for r in s2_results:
            if r.corpusid not in seen:
                local_results.append(r)
                seen.add(r.corpusid)

    return SearchPapersResponse(papers=local_results[: req.limit])


@router.get("/api/health")
async def api_health(request: Request) -> dict[str, object]:
    """Liveness and readiness for the container and reverse proxy."""
    state = _get_state(request)
    return {
        "status": "ok",
        "graph_papers": len(state.data.cid_to_idx),
        "searchable_papers": len(state.searchable_papers),
    }


@router.post("/api/explain")
async def api_explain(req: ExplainRequest, request: Request) -> ExplainResponse:
    """Generate (or fetch cached) a rich explanation for one connection."""
    state = _get_state(request)
    cfg: ServerConfig = request.app.state.server_config
    _enforce_rate_limit(state, request)

    has_query = req.query is not None and bool(req.query.strip())
    if has_query == (req.seed_corpusid is not None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of query or seed_corpusid",
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503, detail="Explanations unavailable: OPENAI_API_KEY not set"
        )

    def facets_for(cid: int) -> Facets | None:
        return state.data.facets_map.get(cid) or state.facet_cache.get(cid)

    target_meta = state.all_metadata.get(req.corpusid) or state.data.metadata.get(
        req.corpusid
    )
    target_facets = facets_for(req.corpusid)
    if target_meta is None or target_facets is None:
        raise HTTPException(
            status_code=404, detail=f"Paper {req.corpusid} or its facets not found"
        )

    seed_title: str | None = None
    seed_facets: Facets | None = None
    if req.seed_corpusid is not None:
        seed_meta = state.all_metadata.get(req.seed_corpusid)
        seed_facets = facets_for(req.seed_corpusid)
        if seed_meta is None or seed_facets is None:
            raise HTTPException(
                status_code=404,
                detail=f"Seed paper {req.seed_corpusid} or its facets not found",
            )
        seed_title = seed_meta.title

    text = await explain_connection(
        model=cfg.explanation_model,
        cache=state.explain_cache,
        corpusid=req.corpusid,
        target_title=target_meta.title,
        target_facets=target_facets,
        query=req.query.strip() if has_query and req.query else None,
        seed_corpusid=req.seed_corpusid,
        seed_title=seed_title,
        seed_facets=seed_facets,
    )
    if text is None:
        raise HTTPException(status_code=502, detail="Explanation generation failed")
    return ExplainResponse(explanation=text)


@router.get("/api/recommend/{corpusid}")
async def api_recommend(corpusid: int, request: Request) -> StreamingResponse:
    """Get paper recommendations via SSE stream."""
    state = _get_state(request)
    cfg: ServerConfig = request.app.state.server_config
    _enforce_rate_limit(state, request)

    async def event_stream() -> AsyncGenerator[str]:
        async for event in recommend(
            corpusid=corpusid,
            data=state.data,
            corpus_index=state.corpus_index,
            embedding_model=state.encoder,
            reranker=state.reranker,
            facet_cache=state.facet_cache,
            s2orc_index=state.s2orc_index,
            s2orc_dir=cfg.s2orc_dir,
            extraction_model=cfg.extraction_model,
            all_metadata=state.all_metadata,
            s2_api=state.s2_api,
            silver_cache_dir=state.silver_cache_dir,
        ):
            data_json = event.data.model_dump_json()
            yield f"event: {event.event}\ndata: {data_json}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
