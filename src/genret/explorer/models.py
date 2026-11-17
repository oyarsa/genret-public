"""API request/response models for the graph explorer."""

from collections.abc import Sequence

from pydantic import BaseModel


class FacetScores(BaseModel, frozen=True):
    """Per-facet cosine similarity to the original query."""

    problems: float
    methods: float
    results: float
    contributions: float


class PaperNode(BaseModel, frozen=True):
    """A paper node in the explorer graph."""

    corpusid: int
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    facets: dict[str, list[str]]
    query_score: float
    facet_scores: FacetScores
    matched_facets: list[str] = []
    """Facets that contributed this paper to the results, for the UI to
    mark. May include the cross-facet signal."""


class ProvenanceEdge(BaseModel, frozen=True):
    """An edge in the explorer graph with facet provenance."""

    source: int
    target: int
    facet: str
    score: float


class SearchRequest(BaseModel, frozen=True):
    """Request body for text search."""

    query: str
    top_k: int = 8
    """Results to return. Kept small (~2 per facet) so the graph stays
    readable; users grow it by expanding nodes."""


class SearchResponse(BaseModel, frozen=True):
    """Response from a text search."""

    papers: list[PaperNode]
    edges: list[ProvenanceEdge]
    query_id: str


class ExpandRequest(BaseModel, frozen=True):
    """Request to expand a node along facet edges."""

    corpusid: int
    facet: str | None = None
    query_id: str
    top_k: int = 5
    known_corpusids: list[int] = []
    """Papers already on the caller's graph. Excluded from the results so
    every expansion adds new papers, and used to connect the new papers
    to the rest of the displayed graph."""


class ExpandFromRequest(BaseModel, frozen=True):
    """Request to expand a node when no original query exists.

    Used by the paper-anchored recommend flow, where the seed paper
    itself stands in for the query.
    """

    corpusid: int
    facet: str | None = None
    top_k: int = 5
    known_corpusids: list[int] = []
    """See ``ExpandRequest.known_corpusids``."""


class ExpandResponse(BaseModel, frozen=True):
    """Response from node expansion."""

    new_papers: list[PaperNode]
    new_edges: list[ProvenanceEdge]


class EdgesRequest(BaseModel, frozen=True):
    """Request for inter-edges among visible papers."""

    corpusids: list[int]


class EdgesResponse(BaseModel, frozen=True):
    """All inter-edges across facet graphs for the given papers."""

    edges: list[ProvenanceEdge]


class PaperDetail(BaseModel, frozen=True):
    """Full metadata and facets for a single paper."""

    corpusid: int
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    citation_count: int | None
    url: str | None
    facets: dict[str, list[str]]


# --- Paper search models ---


class SearchPapersRequest(BaseModel, frozen=True):
    """Request body for paper title search."""

    query: str
    limit: int = 20
    local_only: bool = False
    """Skip the Semantic Scholar API and only return in-graph/in-corpus papers."""


class SearchPaperResult(BaseModel, frozen=True):
    """A single paper from a title search."""

    corpusid: int
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    source: str  # "in_graph", "in_corpus", or "s2_api"


class SearchPapersResponse(BaseModel, frozen=True):
    """Response from paper title search."""

    papers: list[SearchPaperResult]


# --- Explanation models ---


class ExplainRequest(BaseModel, frozen=True):
    """Request for a rich LLM explanation of one retrieval connection.

    Exactly one of ``query`` (text-search mode) or ``seed_corpusid``
    (recommend mode) supplies the context.
    """

    corpusid: int
    query: str | None = None
    seed_corpusid: int | None = None


class ExplainResponse(BaseModel, frozen=True):
    """A generated explanation."""

    explanation: str


# --- Recommend models ---


class RecommendPaper(BaseModel, frozen=True):
    """A recommended paper with metadata and facets."""

    corpusid: int
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    facets: dict[str, list[str]]
    score: float
    matched_facets: list[str] = []
    """See ``PaperNode.matched_facets``."""


class RecommendEdge(BaseModel, frozen=True):
    """An edge between query and recommended paper."""

    target: int
    facet: str
    score: float


class QueryPaper(BaseModel, frozen=True):
    """The query paper with its resolved metadata and facets."""

    corpusid: int
    title: str
    authors: Sequence[str]
    source: str  # "in_graph", "in_corpus", or "out_of_corpus"
    facets: dict[str, list[str]]


class RecommendResult(BaseModel, frozen=True):
    """Final recommendation result sent via SSE."""

    papers: list[RecommendPaper]
    edges: list[RecommendEdge]
    query_paper: QueryPaper


class SSEStatus(BaseModel, frozen=True):
    """SSE status event data."""

    stage: str
    message: str


class SSEError(BaseModel, frozen=True):
    """SSE error event data."""

    message: str
