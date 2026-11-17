"""Retrieval pipeline: BFS walk over sparse edge graphs with RRF fusion.

Loads per-facet sparse edge graphs produced by `facet_edges.py`, walks
neighbours via BFS, and fuses per-facet ranked lists using Reciprocal Rank
Fusion (RRF).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

from genret.utils import open_auto, track_progress

logger = logging.getLogger(__name__)

DEFAULT_BFS_HOPS = 2
"""Default number of BFS hops for neighbourhood expansion."""
DEFAULT_MAX_CANDIDATES = 500
"""Default hard cap on candidates per facet after BFS expansion."""
DEFAULT_RRF_K = 60
"""Default RRF constant (conventional value, robust without tuning)."""
DEFAULT_TOP_N = 20
"""Default number of results to return."""


@dataclass(frozen=True, kw_only=True)
class FusedCandidate:
    """A single candidate from RRF fusion with facet provenance."""

    index: int
    """Paper index (position in corpusids)."""
    score: float
    """RRF fusion score."""
    facets: list[str]
    """Sorted list of facet names that contributed this candidate."""
    facet_ranks: dict[str, int]
    """Per-facet rank (0-based) for each contributing facet."""


@dataclass(frozen=True, kw_only=True)
class RetrievalResult:
    """Retrieval result for a single query paper."""

    corpusid: int
    """Corpus ID of the query paper."""
    results: list[tuple[int, float]]
    """Ranked list of `(corpusid, rrf_score)` pairs, descending by score."""


def load_edge_graph(path: Path) -> tuple[csr_array, npt.NDArray[np.int64]]:
    """Load a `paper_edges_{name}.npz` file as a scipy CSR sparse matrix.

    Args:
        path: Path to the edge graph NPZ file.

    Returns:
        Tuple of `(csr_graph, corpusids)`. The CSR graph has shape `(N, N)`
        where N is the number of papers, with edge scores as data.
    """
    with np.load(path) as npz:
        corpusids: npt.NDArray[np.int64] = npz["corpusids"]
        indptr: npt.NDArray[np.int32] = npz["indptr"]
        indices: npt.NDArray[np.int32] = npz["indices"]
        data: npt.NDArray[np.float16] = npz["data"]

    n = len(corpusids)
    graph = csr_array(
        (data.astype(np.float32), indices, indptr),
        shape=(n, n),
    )
    return graph, corpusids


def _make_symmetric(graph: csr_array) -> csr_array:
    """Make a directed graph undirected by taking element-wise max with transpose.

    Args:
        graph: Directed CSR graph.

    Returns:
        Symmetric CSR graph where `G[i,j] = max(G[i,j], G[j,i])`.
    """
    transpose = graph.T.tocsr()
    # Element-wise max: take the larger of forward and backward edge
    max: csr_array = graph.maximum(transpose).tocsr()
    return max


def load_edge_graphs(
    edge_dir: Path,
) -> tuple[dict[str, csr_array], npt.NDArray[np.int64]]:
    """Load all facet edge graphs and make them symmetric for undirected walks.

    Auto-discovers any `paper_edges_*.npz` files in the directory, supporting
    both standard facet names and cross-facet graphs.

    Args:
        edge_dir: Directory containing `paper_edges_*.npz` files.

    Returns:
        Tuple of `(graphs_by_facet, corpusids)`. Each graph is symmetric
        (undirected) for BFS walks.
    """
    graphs: dict[str, csr_array] = {}
    corpusids: npt.NDArray[np.int64] | None = None

    edge_files = sorted(edge_dir.glob("paper_edges_*.npz"))
    if not edge_files:
        msg = f"No edge graph files found in {edge_dir}"
        raise FileNotFoundError(msg)

    for path in edge_files:
        facet_name = path.stem.removeprefix("paper_edges_")

        graph, cids = load_edge_graph(path)
        graphs[facet_name] = _make_symmetric(graph)

        if corpusids is None:
            corpusids = cids
        else:
            assert np.array_equal(corpusids, cids), (
                f"Corpus ID mismatch between facets: expected {len(corpusids)} "
                f"IDs, got {len(cids)} for {facet_name}"
            )

    assert corpusids is not None

    edges_summary = ", ".join(f"{k}={g.nnz}" for k, g in graphs.items())
    logger.info(
        f"Loaded {len(graphs)} facet graphs, "
        f"{len(corpusids)} papers, "
        f"edges per facet: {edges_summary}"
    )
    return graphs, corpusids


def traverse_neighbours(
    graph: csr_array,
    root: int,
    k: int = DEFAULT_BFS_HOPS,
    min_score: float = 0.0,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[tuple[int, float]]:
    """Walk up to `k` hops from a root node.

    Scores for multi-hop paths are the product of edge scores along the path.
    If a node is reachable via multiple paths, the maximum score is kept.

    Args:
        graph: Symmetric CSR graph (N x N) with edge scores as data.
        root: Starting node index.
        k: Maximum number of hops.
        min_score: Drop edges with score below this threshold.
        max_candidates: Hard cap on total candidates returned.

    Returns:
        List of `(index, score)` pairs sorted descending by score.
        The root node is excluded.
    """
    # best_score[node] = best score reaching that node so far
    best_score: dict[int, float] = {}
    # frontier: nodes to expand from, with their path score
    frontier: dict[int, float] = {root: 1.0}

    for _hop in range(k):
        next_frontier: dict[int, float] = {}

        for node, path_score in frontier.items():
            start = int(graph.indptr[node])
            end = int(graph.indptr[node + 1])
            row_indices = graph.indices[start:end]
            row_data = graph.data[start:end]

            for j, edge_score in zip(row_indices, row_data):
                j_int = int(j)
                if j_int == root:
                    continue

                sc = float(edge_score)
                if sc < min_score:
                    continue

                hop_score = path_score * sc
                if hop_score > best_score.get(j_int, 0.0):
                    best_score[j_int] = hop_score
                    next_frontier[j_int] = hop_score

        frontier = next_frontier
        if not frontier:
            break

    # Sort by score descending, cap at max_candidates
    ranked = sorted(best_score.items(), key=lambda x: x[1], reverse=True)
    return ranked[:max_candidates]


def fuse_rrf(
    per_facet_rankings: dict[str, list[int]],
    k: int = DEFAULT_RRF_K,
) -> list[FusedCandidate]:
    """Reciprocal Rank Fusion across per-facet ranked lists.

    Args:
        per_facet_rankings: Dict mapping facet name to ordered list of paper
            indices (best first). Papers absent from a facet contribute 0.
        k: RRF constant (default 60, conventional value).

    Returns:
        Fused ranked list of `FusedCandidate`, descending by score.
    """
    scores: dict[int, float] = {}
    provenance: dict[int, dict[str, int]] = {}

    for facet_name, ranked_list in per_facet_rankings.items():
        for rank, idx in enumerate(ranked_list):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
            provenance.setdefault(idx, {})[facet_name] = rank

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        FusedCandidate(
            index=idx,
            score=score,
            facets=sorted(provenance[idx]),
            facet_ranks=provenance[idx],
        )
        for idx, score in ranked
    ]


DEFAULT_DIVERSITY_SLOTS = 10
"""Default number of diversity slots reserved for facet-unique candidates.

With 5 facets, this gives each facet 2 unique candidates via round-robin.
Experiment showed nDCG@20 impact of -0.003 (negligible) while surfacing
577 unique candidates across 100 queries (51% related).
"""


def _apply_diversity_slots(
    fused: list[FusedCandidate],
    per_facet: dict[str, list[int]],
    top_n: int,
    diversity_slots: int,
) -> list[FusedCandidate]:
    """Reserve slots for facet-unique candidates after RRF fusion.

    Takes the top `(top_n - diversity_slots)` by RRF consensus, then fills
    remaining slots with each facet's highest-ranked candidate not already
    in the consensus set. This surfaces papers that only one facet found,
    promoting serendipitous discoveries.

    Args:
        fused: Full RRF-fused ranking.
        per_facet: Per-facet BFS rankings (index lists, best first).
        top_n: Total number of results to return.
        diversity_slots: Number of slots reserved for facet-unique candidates.

    Returns:
        Combined ranking: consensus candidates followed by diversity candidates.
    """
    consensus_n = top_n - diversity_slots
    consensus = fused[:consensus_n]
    consensus_indices = {c.index for c in consensus}

    # Build a lookup for provenance of all fused candidates
    fused_by_idx = {c.index: c for c in fused}

    diversity: list[FusedCandidate] = []
    used_indices = set(consensus_indices)

    # Round-robin across facets: each pass takes one unique candidate per
    # facet, repeating until we fill all diversity slots or run out of
    # candidates. This ensures fair allocation across facets.
    facet_names = list(per_facet.keys())
    facet_cursors = dict.fromkeys(facet_names, 0)

    while len(diversity) < diversity_slots:
        added_this_round = False

        for facet_name in facet_names:
            if len(diversity) >= diversity_slots:
                break

            ranked_indices = per_facet[facet_name]
            cursor = facet_cursors[facet_name]

            while cursor < len(ranked_indices):
                idx = ranked_indices[cursor]
                cursor += 1

                if idx not in used_indices:
                    if idx in fused_by_idx:
                        diversity.append(fused_by_idx[idx])
                    else:
                        diversity.append(
                            FusedCandidate(
                                index=idx,
                                score=0.0,
                                facets=[facet_name],
                                facet_ranks={facet_name: cursor - 1},
                            )
                        )
                    used_indices.add(idx)
                    added_this_round = True
                    break

            facet_cursors[facet_name] = cursor

        if not added_this_round:
            break

    return list(consensus) + diversity


def retrieve(
    paper_idx: int,
    graphs: dict[str, csr_array],
    bfs_hops: int = DEFAULT_BFS_HOPS,
    min_score: float = 0.0,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    rrf_k: int = DEFAULT_RRF_K,
    top_n: int = DEFAULT_TOP_N,
    diversity_slots: int = DEFAULT_DIVERSITY_SLOTS,
) -> list[FusedCandidate]:
    """Retrieve related papers for one paper using BFS + RRF fusion.

    Runs BFS per facet on pre-loaded graphs, fuses results with RRF.
    When `diversity_slots > 0`, reserves slots for each facet's unique
    candidates that RRF consensus would not surface.

    Args:
        paper_idx: Index of the query paper (position in corpusids).
        graphs: Dict mapping facet name to symmetric CSR graph.
        bfs_hops: Number of BFS hops.
        min_score: Minimum edge score for BFS expansion.
        max_candidates: Hard cap on candidates per facet after BFS.
        rrf_k: RRF constant.
        top_n: Number of results to return.
        diversity_slots: Number of slots reserved for facet-unique candidates.
            Set to 0 (default) for pure RRF ranking.

    Returns:
        Ranked list of `FusedCandidate`, descending by score.
    """
    per_facet: dict[str, list[int]] = {}

    for facet_name, graph in graphs.items():
        neighbours = traverse_neighbours(
            graph,
            paper_idx,
            k=bfs_hops,
            min_score=min_score,
            max_candidates=max_candidates,
        )
        per_facet[facet_name] = [idx for idx, _score in neighbours]

    fused = fuse_rrf(per_facet, k=rrf_k)

    if diversity_slots > 0:
        return _apply_diversity_slots(fused, per_facet, top_n, diversity_slots)

    return fused[:top_n]


def batch_retrieve(
    graphs: dict[str, csr_array],
    corpusids: npt.NDArray[np.int64],
    paper_indices: list[int],
    output_path: Path,
    bfs_hops: int = DEFAULT_BFS_HOPS,
    min_score: float = 0.0,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    rrf_k: int = DEFAULT_RRF_K,
    top_n: int = DEFAULT_TOP_N,
    diversity_slots: int = DEFAULT_DIVERSITY_SLOTS,
) -> int:
    """Retrieve for multiple papers and write results to JSONL.

    Each output line contains a corpus ID and its ranked retrieval set:
    `{"corpusid": ..., "results": [{"corpusid": ..., "score": ...}, ...]}`.

    Args:
        graphs: Dict mapping facet name to symmetric CSR graph.
        corpusids: Ordered corpus IDs (index <-> position).
        paper_indices: List of paper indices to retrieve for.
        output_path: Output JSONL path (plain, `.gz`, or `.zst`).
        bfs_hops: Number of BFS hops.
        min_score: Minimum edge score for BFS expansion.
        max_candidates: Hard cap on candidates per facet after BFS.
        rrf_k: RRF constant.
        top_n: Number of results per paper.
        diversity_slots: Number of slots reserved for facet-unique candidates.

    Returns:
        Number of papers processed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open_auto(output_path, "w") as fh:
        for paper_idx in track_progress(paper_indices, desc="Retrieving..."):
            results = retrieve(
                paper_idx,
                graphs,
                bfs_hops=bfs_hops,
                min_score=min_score,
                max_candidates=max_candidates,
                rrf_k=rrf_k,
                top_n=top_n,
                diversity_slots=diversity_slots,
            )

            record = {
                "corpusid": int(corpusids[paper_idx]),
                "results": [
                    {
                        "corpusid": int(corpusids[c.index]),
                        "score": round(c.score, 6),
                        "facets": c.facets,
                        "facet_ranks": c.facet_ranks,
                    }
                    for c in results
                ],
            }
            fh.write(json.dumps(record) + "\n")

    logger.info(
        f"Wrote retrieval sets for {len(paper_indices)} papers to {output_path}"
    )
    return len(paper_indices)
