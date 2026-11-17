"""Online retrieval for papers not in the pre-built graph.

Computes ephemeral edges for a new paper by embedding its facets, searching
against corpus vectors, and walking the existing sparse graph.
"""
# pyright: basic

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

from genret.paper_parsing import CorpusFacetItems, Facets
from genret.processing.facet_evaluation import FACET_NAMES
from genret.processing.facet_graph_signals import (
    CitationLookup,
    facet_ngrams,
    jaccard,
    load_citation_lookup,
)
from genret.processing.facet_retrieval import (
    DEFAULT_BFS_HOPS,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_RRF_K,
    FusedCandidate,
    fuse_rrf,
    traverse_neighbours,
)
from genret.processing.facet_stopwords import STOPWORDS
from genret.types import CorpusId
from genret.utils import open_auto

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class CorpusIndex:
    """Pre-loaded corpus data for online retrieval."""

    corpusids: npt.NDArray[np.int64]
    """Ordered corpus IDs."""
    cid_to_idx: dict[CorpusId, int]
    """Mapping from corpus ID to index."""
    corpus_vectors: dict[str, npt.NDArray[np.float32]]
    """Per-facet normalised corpus vectors, shape `(N, D)`."""
    facet_items: CorpusFacetItems
    """Per-facet text items for all papers, indexed by paper position."""
    corpus_references: dict[CorpusId, frozenset[CorpusId]]
    """Per-paper reference sets for bib coupling. Maps corpus ID to the set
    of corpus IDs it references."""
    citation_lookup: CitationLookup
    """Precomputed citation scores for corpus pairs (co-citation + bib coupling)."""
    bib_log_max: float
    """Max `log1p(shared_refs)` across the corpus, for normalising new-paper
    bib coupling scores consistently with the precomputed lookup."""
    corpus_item_embeddings: dict[int, npt.NDArray[np.float32]] | None = None
    """Per-paper raw item embeddings keyed by corpus ID. Each value is
    shape `(n_items, D)` with all facet items concatenated. Only loaded
    when MaxSim dense scoring is needed."""
    corpus_facet_embeddings: dict[str, list[npt.NDArray[np.float32]]] = field(
        default_factory=dict
    )
    """Pre-split per-facet embeddings indexed by paper position. Keys are
    facet names, values are lists where ``[i]`` is shape ``(K_i, D)`` for
    paper ``i``. Built from ``corpus_item_embeddings`` at load time."""


def _load_references(
    references_path: Path, valid_ids: frozenset[CorpusId]
) -> dict[CorpusId, frozenset[CorpusId]]:
    """Load per-paper reference sets, restricted to corpus IDs.

    Args:
        references_path: Path to `references.jsonl` file.
        valid_ids: Set of corpus IDs to restrict to.

    Returns:
        Dict mapping corpus ID to its set of referenced corpus IDs.
    """
    refs: dict[CorpusId, frozenset[CorpusId]] = {}
    with open(references_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            rec: dict[str, Any] = json.loads(line)
            pid = rec["paper_id"]
            if pid not in valid_ids:
                continue

            if paper_refs := frozenset(r for r in rec["references"] if r in valid_ids):
                refs[pid] = paper_refs

    logger.info("Loaded references for %d papers", len(refs))
    return refs


def load_corpus_index(
    dense_dir: Path,
    facets_path: Path,
    references_path: Path | None = None,
    co_citations_path: Path | None = None,
    bib_coupling_path: Path | None = None,
    facet_names: list[str] | None = None,
    item_embeddings_path: Path | None = None,
    preloaded_vectors: tuple[dict[str, npt.NDArray[np.float32]], npt.NDArray[np.int64]]
    | None = None,
) -> CorpusIndex:
    """Load corpus vectors, facet items, and citation data for online retrieval.

    Args:
        dense_dir: Directory containing `facet_graph_*.npz` files with
            `corpus_vectors` arrays.
        facets_path: Path to facets JSONL file.
        references_path: Optional path to `references.jsonl` for bib coupling
            fallback on new papers.
        co_citations_path: Optional path to co-citations JSONL.
        bib_coupling_path: Optional path to bib-coupling JSONL.
        facet_names: Facet names to load. Defaults to the 4 standard facets.
        item_embeddings_path: Optional path to precomputed item embeddings.
        preloaded_vectors: Already-loaded `(corpus_vectors, corpusids)` from
            the same `facet_graph_*.npz` files, shared instead of re-read
            (~70 MB saved when the caller keeps its own copy alive).

    Returns:
        Loaded corpus index.
    """
    names = facet_names or list(FACET_NAMES)
    corpusids: npt.NDArray[np.int64] | None = None
    corpus_vectors: dict[str, npt.NDArray[np.float32]] = {}

    if preloaded_vectors is not None:
        shared_vectors, corpusids = preloaded_vectors
        corpus_vectors = {
            name: shared_vectors[name].astype(np.float32, copy=False)
            for name in names
            if name in shared_vectors
        }
        logger.info("Sharing preloaded corpus vectors for %d facets", len(names))
    else:
        for name in names:
            path = dense_dir / f"facet_graph_{name}.npz"
            if not path.exists():
                logger.warning("Missing %s, skipping", path)
                continue

            with np.load(path) as npz:
                if corpusids is None:
                    corpusids = npz["corpusids"]
                corpus_vectors[name] = npz["corpus_vectors"].astype(np.float32)

            logger.info(
                "Loaded corpus vectors for %s: %s", name, corpus_vectors[name].shape
            )

    if corpusids is None:
        msg = f"No facet graph files found in {dense_dir}"
        raise FileNotFoundError(msg)

    cid_to_idx = {CorpusId(int(cid)): i for i, cid in enumerate(corpusids)}

    # Load facet items
    problems: list[list[str]] = []
    methods: list[list[str]] = []
    results: list[list[str]] = []
    contributions: list[list[str]] = []
    with open_auto(facets_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            facets = Facets.model_validate(json.loads(line))
            problems.append(list(facets.problems))
            methods.append(list(facets.methods))
            results.append(list(facets.results))
            contributions.append(list(facets.contributions))

    facet_items = CorpusFacetItems(
        problems=problems,
        methods=methods,
        results=results,
        contributions=contributions,
    )

    valid_ids = frozenset(CorpusId(int(cid)) for cid in corpusids)

    # Load precomputed citation lookup
    citation_lookup = CitationLookup()
    bib_log_max = 0.0
    if co_citations_path is not None and bib_coupling_path is not None:
        citation_lookup = load_citation_lookup(
            co_citations_path, bib_coupling_path, valid_ids
        )
        # Estimate log_max from the precomputed scores for normalisation
        if citation_lookup:
            bib_log_max = max((v for _, v in citation_lookup.items()), default=0.0)

    # Load references for bib coupling fallback on new papers
    corpus_references: dict[CorpusId, frozenset[CorpusId]] = {}
    if references_path is not None:
        corpus_references = _load_references(references_path, valid_ids)

    # Optionally load raw item embeddings for MaxSim dense scoring
    corpus_item_embeddings: dict[int, npt.NDArray[np.float32]] | None = None
    corpus_facet_embeddings: dict[str, list[npt.NDArray[np.float32]]] = {}
    if item_embeddings_path is not None:
        raw = np.load(item_embeddings_path, allow_pickle=True)
        corpus_item_embeddings = {int(k): v.astype(np.float32) for k, v in raw.items()}
        logger.info(
            "Loaded raw item embeddings for %d papers", len(corpus_item_embeddings)
        )

        # Pre-split into per-facet embeddings for each paper
        n_papers = len(corpusids)
        for facet_name in names:
            facet_embs: list[npt.NDArray[np.float32]] = []
            for i in range(n_papers):
                cid = int(corpusids[i])
                all_embs = corpus_item_embeddings.get(cid)
                if all_embs is not None:
                    facet_embs.append(
                        _get_facet_item_embeddings(all_embs, facet_name, facet_items, i)
                    )
                else:
                    facet_embs.append(np.zeros((0, 0), dtype=np.float32))
            corpus_facet_embeddings[facet_name] = facet_embs
        logger.info("Pre-split embeddings for %d facets", len(corpus_facet_embeddings))

    return CorpusIndex(
        corpusids=corpusids,
        cid_to_idx=cid_to_idx,
        corpus_vectors=corpus_vectors,
        facet_items=facet_items,
        corpus_references=corpus_references,
        citation_lookup=citation_lookup,
        bib_log_max=bib_log_max,
        corpus_item_embeddings=corpus_item_embeddings,
        corpus_facet_embeddings=corpus_facet_embeddings,
    )


def _mean_pool_single(embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Mean-pool and L2-normalise embeddings for a single paper's facet.

    Args:
        embeddings: Shape `(K, D)` item embeddings for one facet.

    Returns:
        Shape `(D,)` normalised vector. Zero vector if no items.
    """
    if len(embeddings) == 0:
        return np.zeros(
            embeddings.shape[-1] if embeddings.ndim > 1 else 1, dtype=np.float32
        )

    vec = embeddings.mean(axis=0)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


def _maxsim_score(
    query_embs: npt.NDArray[np.float32],
    candidate_embs: npt.NDArray[np.float32],
) -> float:
    """Compute MaxSim (max item-pair cosine) between two sets of embeddings.

    Args:
        query_embs: Shape `(Q, D)` query item embeddings.
        candidate_embs: Shape `(C, D)` candidate item embeddings.

    Returns:
        Max cosine similarity across all (q, c) item pairs. Zero if either
        set is empty.
    """
    if len(query_embs) == 0 or len(candidate_embs) == 0:
        return 0.0

    # Normalise
    q_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    q_norms = np.maximum(q_norms, 1e-12)
    q_normed = query_embs / q_norms

    c_norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
    c_norms = np.maximum(c_norms, 1e-12)
    c_normed = candidate_embs / c_norms

    # (Q, C) cosine matrix, take max
    return float(np.max(q_normed @ c_normed.T))


def _get_facet_item_embeddings(
    all_embeddings: npt.NDArray[np.float32],
    facet_name: str,
    facet_items: CorpusFacetItems,
    paper_idx: int,
) -> npt.NDArray[np.float32]:
    """Extract embeddings for a specific facet from a paper's concatenated items.

    The raw embeddings NPZ stores all facet items concatenated in FACET_NAMES
    order. This function slices out the items for the requested facet.

    Args:
        all_embeddings: Shape `(n_total_items, D)` concatenated embeddings.
        facet_name: Which facet to extract.
        facet_items: Corpus facet items for looking up item counts.
        paper_idx: Index of the paper in the corpus.

    Returns:
        Shape `(K, D)` embeddings for the requested facet items.
    """
    offset = 0
    for name in FACET_NAMES:
        items = facet_items.by_name(name)[paper_idx]
        n = len(items)
        if name == facet_name:
            if offset + n > len(all_embeddings):
                return np.zeros((0, all_embeddings.shape[1]), dtype=np.float32)
            return all_embeddings[offset : offset + n]
        offset += n
    return np.zeros((0, all_embeddings.shape[1]), dtype=np.float32)


# Threshold for using GPU: if N_corpus * max_items > this, use GPU
_GPU_THRESHOLD = 50_000


def _maxsim_search_all(
    query_embs: npt.NDArray[np.float32],
    corpus_facet_embs: list[npt.NDArray[np.float32]],
    top_k: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]:
    """Compute MaxSim between query and all corpus papers, return top-k.

    For each corpus paper, computes max cosine similarity across all
    (query_item, corpus_item) pairs. Uses GPU when available and the
    problem is large enough.

    Args:
        query_embs: Shape ``(Q, D)`` query item embeddings for one facet.
        corpus_facet_embs: List of ``(K_i, D)`` arrays, one per corpus paper.
        top_k: Number of top results to return.

    Returns:
        ``(indices, scores)`` — top-k corpus indices and their MaxSim scores,
        sorted descending.
    """
    n_corpus = len(corpus_facet_embs)
    if len(query_embs) == 0 or n_corpus == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.float32)

    # Normalise query embeddings once
    q_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    q_norms = np.maximum(q_norms, 1e-12)
    q_normed = query_embs / q_norms

    # Estimate problem size for GPU decision
    max_items = max((len(e) for e in corpus_facet_embs if len(e) > 0), default=0)
    use_gpu = False

    if n_corpus * max_items > _GPU_THRESHOLD:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                use_gpu = True
        except ImportError:
            pass

    if use_gpu:
        scores = _maxsim_search_gpu(q_normed, corpus_facet_embs, n_corpus)
    else:
        scores = _maxsim_search_cpu(q_normed, corpus_facet_embs, n_corpus)

    actual_k = min(top_k, n_corpus)
    top_indices = np.argpartition(scores, -actual_k)[-actual_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return top_indices, scores[top_indices]


def _maxsim_search_cpu(
    q_normed: npt.NDArray[np.float32],
    corpus_facet_embs: list[npt.NDArray[np.float32]],
    n_corpus: int,
) -> npt.NDArray[np.float32]:
    """CPU path for MaxSim search."""
    scores = np.zeros(n_corpus, dtype=np.float32)
    for i, c_embs in enumerate(corpus_facet_embs):
        if len(c_embs) == 0:
            continue
        c_norms = np.linalg.norm(c_embs, axis=1, keepdims=True)
        c_norms = np.maximum(c_norms, 1e-12)
        c_normed = c_embs / c_norms
        scores[i] = float(np.max(q_normed @ c_normed.T))
    return scores


def _maxsim_search_gpu(
    q_normed: npt.NDArray[np.float32],
    corpus_facet_embs: list[npt.NDArray[np.float32]],
    n_corpus: int,
) -> npt.NDArray[np.float32]:
    """GPU path for MaxSim search. Pads corpus items into a 3D tensor."""
    import torch  # noqa: PLC0415

    device = torch.device("cuda")

    # Find max items and embedding dim
    dim = q_normed.shape[1]
    max_items = max((len(e) for e in corpus_facet_embs if len(e) > 0), default=0)
    if max_items == 0:
        return np.zeros(n_corpus, dtype=np.float32)

    # Pad corpus embeddings into (N, max_items, D) tensor
    padded = np.zeros((n_corpus, max_items, dim), dtype=np.float32)
    mask = np.zeros((n_corpus, max_items), dtype=np.bool_)
    for i, c_embs in enumerate(corpus_facet_embs):
        k = len(c_embs)
        if k > 0:
            norms = np.linalg.norm(c_embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            padded[i, :k, :] = c_embs / norms
            mask[i, :k] = True

    # Move to GPU
    q_t = torch.from_numpy(q_normed).to(device)  # (Q, D)
    c_t = torch.from_numpy(padded).to(device)  # (N, K, D)
    mask_t = torch.from_numpy(mask).to(device)  # (N, K)

    # Compute all cosines: (N, Q, K) via batched matmul
    # q_t: (Q, D) -> (1, Q, D); c_t: (N, K, D) -> (N, D, K)
    cos_matrix = torch.bmm(
        q_t.unsqueeze(0).expand(n_corpus, -1, -1),
        c_t.transpose(1, 2),
    )  # (N, Q, K)

    # Mask out padding
    cos_matrix.masked_fill_(~mask_t.unsqueeze(1), -1.0)

    # Max over both Q and K dimensions
    scores_t = cos_matrix.amax(dim=(1, 2))  # (N,)

    return scores_t.cpu().numpy()


def _bib_coupling_from_refs(
    query_refs: frozenset[CorpusId],
    candidate_refs: frozenset[CorpusId],
    log_max: float,
) -> float:
    """Compute bib coupling score from raw reference sets.

    Fallback for pairs not in the precomputed `CitationLookup`. Uses the
    same `log1p / log_max` normalisation as the offline pipeline.

    Args:
        query_refs: Reference set of the query paper.
        candidate_refs: Reference set of the candidate paper.
        log_max: `log1p(max_shared_refs)` from the precomputed corpus,
            used for normalisation. If zero, returns raw `log1p`.

    Returns:
        Normalised bib coupling score. Zero if no shared references.
    """
    if not query_refs or not candidate_refs:
        return 0.0
    shared = len(query_refs & candidate_refs)
    if shared == 0:
        return 0.0
    raw = float(np.log1p(shared))
    return raw / log_max if log_max > 0 else raw


def retrieve_new_paper(
    query_facets: Facets,
    query_embeddings: dict[str, npt.NDArray[np.float32]],
    corpus_index: CorpusIndex,
    edge_graphs: dict[str, csr_array],
    *,
    query_references: frozenset[CorpusId] | None = None,
    dense_top_k: int = 100,
    edge_top_k: int = 20,
    alpha: float = 0.6,
    beta: float = 0.2,
    gamma: float = 0.2,
    dense_metric: str = "mean_pool",
    bfs_hops: int = DEFAULT_BFS_HOPS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    rrf_k: int = DEFAULT_RRF_K,
    top_n: int = 50,
) -> list[FusedCandidate]:
    """Retrieve related papers for a new paper not in the graph.

    Args:
        query_facets: Extracted facets for the query paper.
        query_embeddings: Per-facet item embeddings for the query paper.
            Keys are facet names, values are `(K, D)` arrays.
        corpus_index: Pre-loaded corpus index.
        edge_graphs: Pre-loaded sparse edge graphs (from `load_edge_graphs`).
        query_references: Optional set of corpus IDs that the query paper
            references, for computing bib coupling scores.
        dense_top_k: Number of top dense candidates per facet.
        edge_top_k: Number of top edges to keep per facet after scoring.
        alpha: Dense signal weight.
        beta: N-gram Jaccard weight.
        gamma: Citation signal weight.
        dense_metric: Dense scoring method. ``"mean_pool"`` uses centroid
            cosine (fast). ``"max_pairs"`` uses MaxSim — mean_pool for
            initial candidate retrieval, then re-scores with max item-pair
            cosine. Requires ``corpus_item_embeddings`` in the corpus index.
        bfs_hops: Number of BFS hops for neighbourhood expansion.
        max_candidates: Hard cap on candidates per facet after BFS.
        rrf_k: RRF constant.
        top_n: Number of results to return.

    Returns:
        Ranked list of `FusedCandidate`, descending by RRF score.
    """

    # Get query facet items
    query_items = {
        "problems": list(query_facets.problems),
        "methods": list(query_facets.methods),
        "results": list(query_facets.results),
        "contributions": list(query_facets.contributions),
    }

    per_facet_rankings: dict[str, list[int]] = {}

    for facet_name, corpus_vecs in corpus_index.corpus_vectors.items():
        # Determine which query facet to use for this graph
        if "_x_" in facet_name:
            query_facet_name = facet_name.split("_x_")[0]
        else:
            query_facet_name = facet_name

        # Step 1: Mean-pool query embeddings for this facet
        if query_facet_name not in query_embeddings:
            logger.warning("No embeddings for facet %s, skipping", query_facet_name)
            continue

        query_vec = _mean_pool_single(query_embeddings[query_facet_name])
        if query_vec.ndim == 0 or np.linalg.norm(query_vec) < 1e-12:
            logger.warning("Zero query vector for facet %s, skipping", facet_name)
            continue

        # Step 2: Dense search
        if (
            dense_metric == "max_pairs"
            and query_facet_name in corpus_index.corpus_facet_embeddings
        ):
            # Full MaxSim search against all corpus papers
            query_facet_embs = query_embeddings.get(query_facet_name)
            if query_facet_embs is not None and len(query_facet_embs) > 0:
                top_indices, top_scores = _maxsim_search_all(
                    query_facet_embs,
                    corpus_index.corpus_facet_embeddings[query_facet_name],
                    dense_top_k,
                )
                similarities_map = {
                    int(idx): float(score)
                    for idx, score in zip(top_indices, top_scores)
                }
            else:
                # Fallback to mean_pool if no query embeddings
                similarities = corpus_vecs @ query_vec
                top_indices = np.argsort(similarities)[::-1][:dense_top_k]
                similarities_map = {
                    int(idx): float(similarities[idx]) for idx in top_indices
                }
        else:
            # Standard mean_pool search
            similarities = corpus_vecs @ query_vec
            top_indices = np.argsort(similarities)[::-1][:dense_top_k]
            similarities_map = {
                int(idx): float(similarities[idx]) for idx in top_indices
            }

        # Step 3: Score candidates with n-gram + citation
        query_ngrams = facet_ngrams(query_items.get(query_facet_name, []), STOPWORDS)

        scored_candidates: list[tuple[int, float]] = []
        for idx in top_indices:
            idx_int = int(idx)
            dense_score = similarities_map[idx_int]

            # N-gram Jaccard
            ngram_score = 0.0
            if beta > 0:
                try:
                    facet_items_seq = corpus_index.facet_items.by_name(query_facet_name)
                    corpus_ngrams = facet_ngrams(
                        list(facet_items_seq[idx_int]), STOPWORDS
                    )
                    ngram_score = jaccard(query_ngrams, corpus_ngrams)
                except KeyError:
                    pass

            # Citation signal: try precomputed lookup first, fall back to
            # bib coupling from raw references for new papers.
            cite_score = 0.0
            if gamma > 0:
                cid = CorpusId(int(corpus_index.corpusids[idx_int]))
                query_cid = query_facets.corpusid
                if (query_cid, cid) in corpus_index.citation_lookup:
                    cite_score = corpus_index.citation_lookup.get(query_cid, cid)
                elif query_references is not None:
                    candidate_refs = corpus_index.corpus_references.get(
                        cid, frozenset()
                    )
                    cite_score = _bib_coupling_from_refs(
                        query_references,
                        candidate_refs,
                        corpus_index.bib_log_max,
                    )

            combined = alpha * dense_score + beta * ngram_score + gamma * cite_score
            scored_candidates.append((idx_int, combined))

        # Step 4: Prune to top-K edges
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        ephemeral_edges = scored_candidates[:edge_top_k]

        # Step 5: BFS expansion through existing graph
        if facet_name in edge_graphs:
            graph = edge_graphs[facet_name]

            # Collect candidates from BFS starting at each ephemeral neighbour
            all_bfs_candidates: dict[int, float] = {}
            for edge_idx, edge_score in ephemeral_edges:
                bfs_results = traverse_neighbours(
                    graph,
                    edge_idx,
                    k=bfs_hops,
                    max_candidates=max_candidates,
                )
                for bfs_idx, bfs_score in bfs_results:
                    hop_score = edge_score * bfs_score
                    if hop_score > all_bfs_candidates.get(bfs_idx, 0.0):
                        all_bfs_candidates[bfs_idx] = hop_score

            # Add direct neighbours too
            for edge_idx, edge_score in ephemeral_edges:
                if edge_score > all_bfs_candidates.get(edge_idx, 0.0):
                    all_bfs_candidates[edge_idx] = edge_score

            # Sort by score and take top candidates
            ranked = sorted(
                all_bfs_candidates.items(), key=lambda x: x[1], reverse=True
            )
            per_facet_rankings[facet_name] = [
                idx for idx, _score in ranked[:max_candidates]
            ]
        else:
            # No graph for this facet — use direct dense candidates
            per_facet_rankings[facet_name] = [idx for idx, _score in ephemeral_edges]

    # Step 6: RRF fusion
    fused = fuse_rrf(per_facet_rankings, k=rrf_k)
    return fused[:top_n]
