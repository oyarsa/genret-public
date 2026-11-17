"""Hierarchical Leiden community detection over the existing facet graphs.

Drop-in replacement for the hierarchical k-means codebook in
:mod:`codebook` — produces the same ``paper_codes.jsonl`` per-facet
artefact so the existing C2T-ID builder, GLEN trie, and beam-search code
work unchanged.

Input: the per-facet candidate graphs written by
:mod:`genret.processing.facet_graph_build` as ``facet_candidates_<f>.npz``.
Each file is already a top-K kNN graph per paper, weighted by the fused
``alpha * dense + beta * n-gram + gamma * citation`` score. We feed those
edges (and their weights) to Leiden directly — no embedding recomputation,
no fresh kNN construction. The citation signal Leiden cares about is
already baked into the edge weights.

Pipeline per facet:

1. Load ``corpusids``, ``candidates``, ``scores_combined`` from the
   facet's candidates npz.
2. Symmetrise: keep an edge ``(i, j)`` if either side lists the other as
   a top-K neighbour. Edge weight = max of the two directed weights.
3. Run Leiden at a coarse resolution to get L0 communities.
4. For each L0 community, induce the sub-graph and re-run Leiden at a
   finer resolution to get L1 sub-communities.
5. Save ``corpusid -> (L0, L1)`` in the codebook directory layout.

Why hierarchical: a flat Leiden run yields too many or too few clusters
depending on resolution; a two-level approach matches the codebook's
branching factor (~32 x 16 leaves) and gives the C2T-ID a parent/child
level to label.

Leiden vs k-means: Leiden uses the graph structure (whom you cluster
with whom, weighted by combined dense+ngram+citation signal) rather than
centroid distance in pure embedding space. So it can recover topical
communities that the embedding-only k-means partition misses.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from genret.training.codebook_data import FACET_NAMES
from genret.types import CorpusId

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LeidenCodebookParams:
    """Parameters for hierarchical Leiden codebook construction."""

    resolution_l0: float = 0.6
    """Leiden resolution at the top (L0) level. Smaller = fewer, larger
    communities."""

    resolution_l1: float = 1.2
    """Leiden resolution inside each L0 sub-graph (L1 level)."""

    seed: int = 42
    """RNG seed for Leiden initialisation. Leiden is non-deterministic;
    fixing the seed gives reproducible community assignments."""

    n_iterations: int = -1
    """Number of Leiden refinement iterations (``-1`` = run until
    convergence)."""

    min_l1_size: int = 4
    """L0 communities with at most this many papers are kept as a single
    L1 cluster (no sub-clustering). Avoids degenerate Leiden runs on
    sparse sub-graphs."""

    truncate_to_top_k: int | None = None
    """If set, keep only the top-K neighbours per row of the candidates
    matrix before building the graph. ``None`` uses all columns the npz
    provides (typically 100)."""

    score_key: str = "scores_combined"
    """Which weight column in the candidates npz to use as edge weight.
    Default ``scores_combined`` = fused dense + n-gram + citation."""


@dataclass(frozen=True)
class _FacetGraph:
    """Loaded candidates graph for one facet."""

    corpusids: npt.NDArray[np.int64]
    edges: list[tuple[int, int]]
    weights: list[float]


def _load_facet_candidates(
    candidates_dir: Path, facet: str, params: LeidenCodebookParams
) -> _FacetGraph:
    """Load ``facet_candidates_<facet>.npz`` and symmetrise to an edge list."""
    path = candidates_dir / f"facet_candidates_{facet}.npz"
    f = np.load(path)
    corpusids = np.asarray(f["corpusids"], dtype=np.int64)
    candidates = np.asarray(f["candidates"], dtype=np.int64)
    scores = np.asarray(f[params.score_key], dtype=np.float32)
    if candidates.shape != scores.shape:
        raise ValueError(
            f"{path}: candidates shape {candidates.shape} mismatches "
            f"{params.score_key} shape {scores.shape}"
        )
    if params.truncate_to_top_k is not None:
        k = min(params.truncate_to_top_k, candidates.shape[1])
        candidates = candidates[:, :k]
        scores = scores[:, :k]

    n = corpusids.shape[0]
    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        for col, j in enumerate(candidates[i]):
            jj = int(j)
            if jj == i or jj < 0 or jj >= n:
                continue
            w = float(scores[i, col])
            if w <= 0:
                # Negative / zero combined scores aren't informative edges;
                # Leiden expects non-negative weights for modularity.
                continue
            a, b = (i, jj) if i < jj else (jj, i)
            prev = edges.get((a, b))
            # Symmetrise by max — strongest of the two directed weights.
            if prev is None or w > prev:
                edges[(a, b)] = w

    edge_list = list(edges.keys())
    weight_list = [edges[e] for e in edge_list]
    logger.info(
        "Facet %s: %d papers, %d undirected edges (kNN cols=%d)",
        facet,
        n,
        len(edge_list),
        candidates.shape[1],
    )
    return _FacetGraph(corpusids=corpusids, edges=edge_list, weights=weight_list)


def _run_leiden(
    edges: Sequence[tuple[int, int]],
    weights: Sequence[float],
    n_nodes: int,
    *,
    resolution: float,
    seed: int,
    n_iterations: int,
) -> list[int]:
    """Run a single Leiden pass; return per-node community label."""
    import igraph as ig  # noqa: PLC0415
    import leidenalg  # noqa: PLC0415

    g = ig.Graph(n=n_nodes, edges=list(edges), directed=False)
    g.es["weight"] = list(weights)
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=n_iterations,
    )
    return list(partition.membership)


@dataclass(frozen=True)
class LeidenFacetCodebook:
    """Result of hierarchical Leiden over one facet's papers."""

    facet: str
    paper_codes: dict[CorpusId, tuple[int, int]]
    """``corpusid -> (L0, L1)``."""

    n_l0: int
    """Number of distinct L0 communities found."""


def fit_facet_leiden(
    facet: str,
    graph: _FacetGraph,
    params: LeidenCodebookParams,
) -> LeidenFacetCodebook:
    """Fit a 2-level Leiden codebook for one facet's candidates graph."""
    logger.info(
        "Leiden L0 for %s: res=%.2f, n=%d, edges=%d",
        facet,
        params.resolution_l0,
        len(graph.corpusids),
        len(graph.edges),
    )

    n = len(graph.corpusids)
    l0_membership = _run_leiden(
        graph.edges,
        graph.weights,
        n_nodes=n,
        resolution=params.resolution_l0,
        seed=params.seed,
        n_iterations=params.n_iterations,
    )

    # Renumber L0 to contiguous ints in descending community-size order.
    l0_arr = np.asarray(l0_membership, dtype=np.int64)
    unique, counts = np.unique(l0_arr, return_counts=True)
    order = np.argsort(-counts)
    remap = {int(old): int(new) for new, old in enumerate(unique[order])}
    l0_renum = np.asarray([remap[int(c)] for c in l0_arr], dtype=np.int64)

    n_l0 = int(l0_renum.max()) + 1
    paper_codes: dict[CorpusId, tuple[int, int]] = {}

    for l0 in range(n_l0):
        member_idx = np.where(l0_renum == l0)[0]
        m = len(member_idx)
        if m == 0:
            continue
        if m <= params.min_l1_size:
            # Tiny community: single L1 bucket.
            for idx in member_idx:
                paper_codes[CorpusId(int(graph.corpusids[idx]))] = (l0, 0)
            continue

        # Induce sub-graph on member nodes and re-run Leiden.
        local_index = {int(orig): new for new, orig in enumerate(member_idx)}
        member_set = {int(x) for x in member_idx}
        sub_edges: list[tuple[int, int]] = []
        sub_weights: list[float] = []
        for (u, v), w in zip(graph.edges, graph.weights, strict=True):
            if u in member_set and v in member_set:
                sub_edges.append((local_index[u], local_index[v]))
                sub_weights.append(w)

        if not sub_edges:
            # Singleton-ish community in the induced graph: assign all to L1=0.
            for idx in member_idx:
                paper_codes[CorpusId(int(graph.corpusids[idx]))] = (l0, 0)
            continue

        l1_membership = _run_leiden(
            sub_edges,
            sub_weights,
            n_nodes=m,
            resolution=params.resolution_l1,
            seed=params.seed,
            n_iterations=params.n_iterations,
        )
        l1_arr = np.asarray(l1_membership, dtype=np.int64)
        uniq, cnts = np.unique(l1_arr, return_counts=True)
        order_local = np.argsort(-cnts)
        l1_remap = {int(old): int(new) for new, old in enumerate(uniq[order_local])}
        for new_idx, orig_idx in enumerate(member_idx):
            cid = CorpusId(int(graph.corpusids[orig_idx]))
            paper_codes[cid] = (l0, l1_remap[int(l1_arr[new_idx])])

    logger.info(
        "Facet %s: %d L0 communities (top sizes %s)",
        facet,
        n_l0,
        list(counts[order][:5]),
    )
    return LeidenFacetCodebook(facet=facet, paper_codes=paper_codes, n_l0=n_l0)


def _write_facet_artifacts(output_dir: Path, codebook: LeidenFacetCodebook) -> None:
    """Persist ``paper_codes.jsonl`` in the codebook directory layout."""
    facet_dir = output_dir / codebook.facet
    facet_dir.mkdir(parents=True, exist_ok=True)
    with (facet_dir / "paper_codes.jsonl").open("w") as fh:
        for cid in sorted(codebook.paper_codes):
            l0, l1 = codebook.paper_codes[cid]
            fh.write(json.dumps({"corpusid": int(cid), "code": [l0, l1]}) + "\n")


def _build_code_to_papers(
    codebook: LeidenFacetCodebook,
) -> dict[str, list[int]]:
    """``"L0,L1" -> [corpusids]`` index, matching the kmeans codebook format."""
    out: dict[str, list[int]] = {}
    for cid in codebook.paper_codes:
        l0, l1 = codebook.paper_codes[cid]
        out.setdefault(f"{l0},{l1}", []).append(int(cid))
    for key in out:
        out[key] = sorted(out[key])
    return out


def build_leiden_codebooks(
    candidates_dir: Path,
    output_dir: Path,
    params: LeidenCodebookParams,
    *,
    facets: Iterable[str] = FACET_NAMES,
) -> dict[str, LeidenFacetCodebook]:
    """Fit hierarchical Leiden codebooks for all facets, save artefacts.

    ``candidates_dir`` must contain ``facet_candidates_<facet>.npz`` files
    produced by :mod:`genret.processing.facet_graph_build`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, LeidenFacetCodebook] = {}
    code_to_papers: dict[str, dict[str, list[int]]] = {}
    for facet in facets:
        graph = _load_facet_candidates(candidates_dir, facet, params)
        codebook = fit_facet_leiden(facet, graph, params)
        _write_facet_artifacts(output_dir, codebook)
        code_to_papers[facet] = _build_code_to_papers(codebook)
        out[facet] = codebook

    (output_dir / "code_to_papers.json").write_text(
        json.dumps(code_to_papers, indent=2) + "\n"
    )

    (output_dir / "leiden_params.json").write_text(
        json.dumps(
            {
                "resolution_l0": params.resolution_l0,
                "resolution_l1": params.resolution_l1,
                "seed": params.seed,
                "n_iterations": params.n_iterations,
                "min_l1_size": params.min_l1_size,
                "truncate_to_top_k": params.truncate_to_top_k,
                "score_key": params.score_key,
                "facets": list(facets),
                "per_facet_n_l0": {f: cb.n_l0 for f, cb in out.items()},
            },
            indent=2,
        )
        + "\n"
    )
    return out


__all__ = [
    "LeidenCodebookParams",
    "LeidenFacetCodebook",
    "build_leiden_codebooks",
    "fit_facet_leiden",
]
