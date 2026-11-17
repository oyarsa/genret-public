"""Build candidate graph NPZ files combining dense, n-gram, and citation signals.

For each of the 4 facet types, produces a `facet_candidates_{name}.npz` file
containing top-K candidate paper indices and per-signal scores. Also writes
a `candidate_graph_params.json` recording all run parameters.
"""

# pyright: basic

import gc
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from genret.paper_parsing import CorpusFacetItems, Facets
from genret.processing.facet_evaluation import FACET_NAMES
from genret.processing.facet_graph_signals import (
    CitationLookup,
    NgramSet,
    facet_ngrams,
    jaccard,
    load_citation_lookup,
)
from genret.processing.facet_stopwords import STOPWORDS
from genret.processing.nptypes import Float16Array2D, Int32Array2D
from genret.types import CorpusId
from genret.utils import iter_jsonl, resolve_jsonl, track_progress

logger = logging.getLogger(__name__)


class CandidateGraphParams(BaseModel, frozen=True):
    """Parameters used to build the candidate graph.

    Written to `candidate_graph_params.json` in the output directory after the job
    completes, following the convention in `connected_sampling.py`.
    """

    k: int
    """Top-K candidates retained per paper per facet."""
    alpha: float
    """Weight for the dense cosine similarity signal."""
    beta: float
    """Weight for the n-gram Jaccard signal."""
    gamma: float
    """Weight for the citation signal."""
    dense_metric: str
    """NPZ key used for dense similarity (e.g. 'mean_pool' or 'max_pairs')."""
    embedding_model: str
    """Name of the embedding model used for the dense NPZ files."""
    facets_path: str
    """Absolute path to the facets JSONL file."""
    dense_dir: str
    """Directory containing the `facet_graph_*.npz` dense matrices."""
    co_citations_path: str
    """Path to the co_citations JSONL file."""
    bib_coupling_path: str
    """Path to the bib_coupling JSONL file."""
    created_at: str
    """ISO-8601 UTC timestamp when the job completed."""


def _load_facets(
    facets_path: Path,
) -> tuple[list[CorpusId], CorpusFacetItems]:
    """Load facets JSONL and return corpusids with per-facet item text lists.

    Args:
        facets_path: Path to the facets JSONL file (plain or compressed).

    Returns:
        Tuple of:
        - `corpusids`: Ordered list of corpus IDs.
        - `items`: Per-facet item text lists, one list per paper in the same
          order as `corpusids`.
    """
    resolved = resolve_jsonl(facets_path)
    corpusids: list[CorpusId] = []
    problems: list[list[str]] = []
    methods: list[list[str]] = []
    results: list[list[str]] = []
    contributions: list[list[str]] = []

    for record in track_progress(iter_jsonl(resolved), desc="Loading facets..."):
        facets = Facets.model_validate(record)
        corpusids.append(facets.corpusid)
        problems.append(list(facets.problems))
        methods.append(list(facets.methods))
        results.append(list(facets.results))
        contributions.append(list(facets.contributions))

    logger.info(f"Loaded {len(corpusids)} facet records")
    return corpusids, CorpusFacetItems(
        problems=problems,
        methods=methods,
        results=results,
        contributions=contributions,
    )


def _precompute_ngrams(corpus_facets: CorpusFacetItems) -> dict[str, list[NgramSet]]:
    """Precompute n-gram sets for all (paper, facet) pairs.

    Args:
        corpus_facets: Per-facet item text lists for all papers.

    Returns:
        Dict mapping facet name to a list of `NgramSet` objects, one per paper.
    """
    ngrams_by_facet: dict[str, list[NgramSet]] = {}

    for facet_name, items_list in corpus_facets.iter_facets():
        ngrams_by_facet[facet_name] = [
            facet_ngrams(list(items), STOPWORDS) for items in items_list
        ]
        logger.info(f"  {facet_name}: {len(items_list)} n-gram sets computed")

    return ngrams_by_facet


def _build_facet_candidates(
    facet_name: str,
    corpusids: list[CorpusId],
    dense_dir: Path,
    output_dir: Path,
    ngrams: list[NgramSet],
    citation_lookup: CitationLookup,
    k: int,
    alpha: float,
    beta: float,
    gamma: float,
    dense_metric: str = "mean_pool",
) -> None:
    """Build and write the candidate graph for one facet type.

    Loads the dense NxN similarity matrix, selects the top-K candidates per
    paper by dense score (excluding self), then computes n-gram and citation
    scores for each candidate pair, re-ranks by the weighted combination, and
    writes a compressed NPZ file.

    Args:
        facet_name: Name of the facet type (e.g. `"problems"`).
        corpusids: Ordered list of corpus IDs (N papers).
        dense_dir: Directory containing `facet_graph_{name}.npz` files.
        output_dir: Directory to write the output NPZ file.
        ngrams: Precomputed n-gram sets for each paper (N entries).
        citation_lookup: Citation score lookup for paper pairs.
        k: Number of top candidates to keep per paper.
        alpha: Dense signal weight.
        beta: N-gram Jaccard weight.
        gamma: Citation signal weight.
        dense_metric: NPZ key for the dense similarity matrix.
    """
    dense_path = dense_dir / f"facet_graph_{facet_name}.npz"
    out_path = output_dir / f"facet_candidates_{facet_name}.npz"
    tmp_path = output_dir / f"facet_candidates_{facet_name}.tmp.npz"

    logger.info(f"=== {facet_name} ===")
    logger.info(f"  Loading dense matrix from {dense_path}")

    with np.load(dense_path) as npz:
        mean_pool: Float16Array2D = npz[dense_metric]

    n = len(corpusids)
    actual_k = min(k, n - 1)

    logger.info(f"  Selecting top-{actual_k} candidates by dense score...")
    np.fill_diagonal(mean_pool, -1.0)  # exclude self

    # Partial sort via argpartition (fast), then sort the partition descending.
    partitioned = np.argpartition(mean_pool, -actual_k, axis=1)[:, -actual_k:]
    row_idx = np.arange(n)[:, None]
    top_k_scores = mean_pool[row_idx, partitioned]

    order = np.argsort(-top_k_scores.astype(np.float32), axis=1)
    candidates_idx: Int32Array2D = partitioned[row_idx, order].astype(np.int32)
    scores_dense: Float16Array2D = top_k_scores[row_idx, order].astype(np.float16)

    # free ~200 MB before n-gram loop
    del mean_pool
    gc.collect()
    gc.collect()

    logger.info(f"  Computing signal scores for {n}x{actual_k} pairs...")
    scores_ngram = np.zeros((n, actual_k), dtype=np.float16)
    scores_citation = np.zeros((n, actual_k), dtype=np.float16)

    for i in track_progress(range(n), desc=f"{facet_name} signals"):
        cid_i = corpusids[i]
        for j_pos in range(actual_k):
            j = int(candidates_idx[i, j_pos])
            cid_j = corpusids[j]

            if beta > 0:
                scores_ngram[i, j_pos] = jaccard(ngrams[i], ngrams[j])
            scores_citation[i, j_pos] = citation_lookup.get(cid_i, cid_j)

    logger.info("  Computing combined scores and re-ranking...")
    scores_combined = (
        alpha * scores_dense.astype(np.float32)
        + beta * scores_ngram.astype(np.float32)
        + gamma * scores_citation.astype(np.float32)
    ).astype(np.float16)

    rerank_order = np.argsort(-scores_combined.astype(np.float32), axis=1)
    row_idx = np.arange(n)[:, None]
    kwds: dict[str, Any] = {
        "corpusids": np.array(corpusids, dtype=np.int64),
        "candidates": candidates_idx[row_idx, rerank_order],
        "scores_dense": scores_dense[row_idx, rerank_order],
        "scores_ngram": scores_ngram[row_idx, rerank_order],
        "scores_citation": scores_citation[row_idx, rerank_order],
        "scores_combined": scores_combined[row_idx, rerank_order],
    }
    logger.info(f"  Writing {out_path}...")
    np.savez_compressed(tmp_path, **kwds)
    tmp_path.rename(out_path)
    logger.info(f"  Saved {out_path}")


def build_candidate_graph(
    facets_path: Path,
    dense_dir: Path,
    co_citations_path: Path,
    bib_coupling_path: Path,
    output_dir: Path,
    embedding_model: str,
    k: int = 100,
    alpha: float = 0.75,
    beta: float = 0.0,
    gamma: float = 0.25,
    dense_metric: str = "mean_pool",
) -> None:
    """Build candidate graph NPZ files for all 4 facet types.

    For each facet type, writes `facet_candidates_{name}.npz` to `output_dir`.
    After all facets are processed, writes `candidate_graph_params.json`
    recording all run parameters.

    Args:
        facets_path: Path to the facets JSONL file (plain or compressed).
        dense_dir: Directory containing `facet_graph_*.npz` dense matrices.
        co_citations_path: Path to the co_citations JSONL file.
        bib_coupling_path: Path to the bib_coupling JSONL file.
        output_dir: Directory to write output files.
        embedding_model: Name of the embedding model (for params record).
        k: Top-K candidates to retain per paper per facet.
        alpha: Dense signal weight.
        beta: N-gram Jaccard weight.
        gamma: Citation signal weight.
        dense_metric: NPZ key for the dense similarity matrix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    corpusids, corpus_facets = _load_facets(facets_path)
    if not corpusids:
        logger.warning("No papers loaded; nothing to do.")
        return

    if beta > 0:
        logger.info("Precomputing n-gram sets...")
        ngrams_by_facet = _precompute_ngrams(corpus_facets)
    else:
        logger.info("Skipping n-gram precomputation (beta=0)")
        ngrams_by_facet = {name: [frozenset()] * len(corpusids) for name in FACET_NAMES}

    logger.info("Loading citation signals...")
    valid_ids = frozenset(corpusids)
    citation_lookup = load_citation_lookup(
        co_citations_path, bib_coupling_path, valid_ids
    )

    # Standard same-facet candidate graphs
    for facet_name in FACET_NAMES:
        _build_facet_candidates(
            facet_name=facet_name,
            corpusids=corpusids,
            dense_dir=dense_dir,
            output_dir=output_dir,
            ngrams=ngrams_by_facet[facet_name],
            citation_lookup=citation_lookup,
            k=k,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            dense_metric=dense_metric,
        )

    # Cross-facet candidate graphs (auto-discovered from dense dir)
    for dense_path in sorted(dense_dir.glob("facet_graph_*_x_*.npz")):
        cross_name = dense_path.stem.removeprefix("facet_graph_")
        query_facet = cross_name.split("_x_")[0]
        ngrams = ngrams_by_facet.get(query_facet, ngrams_by_facet["problems"])
        _build_facet_candidates(
            facet_name=cross_name,
            corpusids=corpusids,
            dense_dir=dense_dir,
            output_dir=output_dir,
            ngrams=ngrams,
            citation_lookup=citation_lookup,
            k=k,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            dense_metric=dense_metric,
        )

    params = CandidateGraphParams(
        k=k,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        dense_metric=dense_metric,
        embedding_model=embedding_model,
        facets_path=str(facets_path.resolve()),
        dense_dir=str(dense_dir.resolve()),
        co_citations_path=str(co_citations_path.resolve()),
        bib_coupling_path=str(bib_coupling_path.resolve()),
        created_at=datetime.now(UTC).isoformat(),
    )
    params_path = output_dir / "candidate_graph_params.json"
    params_path.write_text(params.model_dump_json(indent=2) + "\n")
    logger.info(f"Params written to {params_path}")
