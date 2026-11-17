"""Alternative set-level similarity metrics for facet embeddings.

Provides MaxSim, optimal transport, and intrinsic coherence
metrics as alternatives to the existing mean_pool/max_pairs/mean_pairs
aggregations.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from genret.processing.facet_evaluation import FACET_NAMES
from genret.processing.facet_graph import FacetArrays
from genret.utils import track_progress

logger = logging.getLogger(__name__)

type MetricFn = Callable[
    [npt.NDArray[np.float32], int, npt.NDArray[np.float32], int], float
]
"""Signature for pairwise set-level similarity functions.

Parameters: (embeddings_a, count_a, embeddings_b, count_b)"""


def maxsim(
    emb_a: npt.NDArray[np.float32],
    emb_b: npt.NDArray[np.float32],
    count_a: int,
    count_b: int,
) -> float:
    """MaxSim: mean of per-item best-match similarities.

    For each item in A, finds its best cosine match in B, then averages.
    Symmetric: computes both A->B and B->A, returns the mean.

    Args:
        emb_a: Shape `(max_k, D)` embeddings for paper A (padded).
        emb_b: Shape `(max_k, D)` embeddings for paper B (padded).
        count_a: Number of valid items in paper A.
        count_b: Number of valid items in paper B.

    Returns:
        Symmetric MaxSim score, or 0.0 if either paper has no items.
    """
    if count_a == 0 or count_b == 0:
        return 0.0

    a = emb_a[:count_a]
    b = emb_b[:count_b]

    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    sim = a_norm @ b_norm.T  # (count_a, count_b)

    # A->B: for each item in A, best match in B
    a_to_b = float(sim.max(axis=1).mean())
    # B->A: for each item in B, best match in A
    b_to_a = float(sim.max(axis=0).mean())

    return (a_to_b + b_to_a) / 2.0


def optimal_transport_similarity(
    emb_a: npt.NDArray[np.float32],
    emb_b: npt.NDArray[np.float32],
    count_a: int,
    count_b: int,
) -> float:
    """Optimal transport similarity via Hungarian algorithm.

    Computes the minimum-cost bipartite matching between item sets, where
    cost is `1 - cosine_similarity`. Returns `1 - normalised_cost` so higher
    is more similar (matching the convention of other metrics).

    For unequal set sizes, pads the smaller set with zero-similarity dummy items.

    Args:
        emb_a: Shape `(max_k, D)` embeddings for paper A (padded).
        emb_b: Shape `(max_k, D)` embeddings for paper B (padded).
        count_a: Number of valid items in paper A.
        count_b: Number of valid items in paper B.

    Returns:
        OT similarity score in [0, 1], or 0.0 if either paper has no items.
    """
    if count_a == 0 or count_b == 0:
        return 0.0

    a = emb_a[:count_a]
    b = emb_b[:count_b]

    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    sim = a_norm @ b_norm.T  # (count_a, count_b)

    # Cost = 1 - similarity (minimising cost = maximising similarity)
    cost = 1.0 - sim

    # Hungarian algorithm finds minimum cost assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    total_cost = float(cost[row_ind, col_ind].sum())
    n_matched = len(row_ind)

    # Normalise by the larger set size (unmatched items have cost 1.0)
    max_size = max(count_a, count_b)
    unmatched = max_size - n_matched
    normalised_cost = (total_cost + unmatched) / max_size

    return 1.0 - normalised_cost


def compute_pairwise_metric(
    arrays: FacetArrays,
    metric_fn: MetricFn,
    paper_indices: list[int],
    candidate_indices: npt.NDArray[np.int32],
    desc: str = "Computing...",
) -> npt.NDArray[np.float32]:
    """Compute a set-level metric for query-candidate pairs.

    Args:
        arrays: Padded embedding arrays for one facet type.
        metric_fn: Function `(emb_a, emb_b, count_a, count_b) -> float`.
        paper_indices: Indices of query papers to evaluate.
        candidate_indices: Shape `(N, K)` candidate indices per paper.
        desc: Progress bar description.

    Returns:
        Shape `(len(paper_indices), K)` score matrix.
    """
    n_queries = len(paper_indices)
    k = candidate_indices.shape[1]
    scores = np.zeros((n_queries, k), dtype=np.float32)

    for qi, paper_idx in enumerate(track_progress(paper_indices, desc=desc)):
        emb_a = arrays.embeddings[paper_idx]
        count_a = int(arrays.counts[paper_idx])

        for ci in range(k):
            cand_idx = int(candidate_indices[paper_idx, ci])
            if cand_idx < 0:
                continue
            emb_b = arrays.embeddings[cand_idx]
            count_b = int(arrays.counts[cand_idx])
            scores[qi, ci] = metric_fn(emb_a, count_a, emb_b, count_b)

    return scores


@dataclass(frozen=True, kw_only=True)
class IntrinsicCoherenceResult:
    """Intrinsic coherence statistics for a set of retrieval results."""

    mean_coherence: float
    """Mean pairwise similarity within each query's retrieved set, averaged."""
    median_coherence: float
    """Median of per-query coherence scores."""
    num_queries: int
    """Number of queries evaluated."""


def compute_intrinsic_coherence(
    retrieval_sets: dict[int, list[int]],
    dense_dir: Path,
) -> dict[str, IntrinsicCoherenceResult]:
    """Compute intrinsic coherence of retrieval results per facet.

    For each query's top-K results, computes mean pairwise cosine similarity
    using the mean_pool vectors. High coherence means the retrieved papers
    form a tight topical cluster.

    Args:
        retrieval_sets: Query corpus ID to ordered list of result corpus IDs.
        dense_dir: Directory containing `facet_graph_*.npz` files.

    Returns:
        Dict mapping facet name to coherence results.
    """
    results: dict[str, IntrinsicCoherenceResult] = {}

    for facet_name in FACET_NAMES:
        npz_path = dense_dir / f"facet_graph_{facet_name}.npz"
        if not npz_path.exists():
            continue

        with np.load(npz_path) as npz:
            corpusids: npt.NDArray[np.int64] = npz["corpusids"]
            mean_pool: npt.NDArray[np.float16] = npz["mean_pool"]

        cid_to_idx = {int(cid): i for i, cid in enumerate(corpusids)}
        coherence_scores: list[float] = []

        for result_cids in retrieval_sets.values():
            result_indices = [
                cid_to_idx[cid] for cid in result_cids if cid in cid_to_idx
            ]
            if len(result_indices) < 2:
                continue

            # Mean pairwise similarity within the retrieved set
            idx_arr = np.array(result_indices)
            sub_matrix = mean_pool[np.ix_(idx_arr, idx_arr)].astype(np.float32)

            # Extract upper triangle (exclude diagonal)
            n = len(idx_arr)
            triu_indices = np.triu_indices(n, k=1)
            pairwise_sims = sub_matrix[triu_indices]
            coherence_scores.append(float(pairwise_sims.mean()))

        if coherence_scores:
            results[facet_name] = IntrinsicCoherenceResult(
                mean_coherence=float(np.mean(coherence_scores)),
                median_coherence=float(np.median(coherence_scores)),
                num_queries=len(coherence_scores),
            )
            logger.info(
                f"  {facet_name}: mean={results[facet_name].mean_coherence:.4f}, "
                f"median={results[facet_name].median_coherence:.4f}"
            )

    return results
