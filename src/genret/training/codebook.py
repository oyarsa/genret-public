"""Per-facet hierarchical k-means codebooks for generative retrieval.

Each facet (problems, methods, results, contributions) gets its own
two-level codebook learned from the per-facet mean-pooled paper
vectors that already power the graph retriever. The output is:

- L0/L1 cluster centers as ``.npy`` files
- a per-paper code map ``corpusid -> (L0, L1)`` as JSONL
- a code-to-papers lookup ``{facet: {"L0,L1": [corpusids]}}`` as JSON
- validation statistics (leaf occupancy distribution)

These artifacts feed the codebook-based training data builder and the
constrained-decoding logits processor downstream.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans

from genret.types import CorpusId

logger = logging.getLogger(__name__)

FACET_NAMES: tuple[str, ...] = (
    "problems",
    "methods",
    "results",
    "contributions",
)
"""Same-facet types used for codebook construction."""


@dataclass(frozen=True)
class CodebookParams:
    """Parameters for hierarchical k-means codebook construction."""

    k0: int = 32
    """Number of top-level clusters per facet."""

    k1: int = 32
    """Number of sub-clusters per L0 cluster."""

    random_state: int = 0
    """Seed for k-means initialisation (determinism)."""

    n_init: int = 4
    """Number of k-means re-runs per fit; the best inertia wins."""

    min_l1_samples: int = 2
    """If an L0 cluster has fewer points than this, it gets a single L1
    cluster (no further subdivision)."""


@dataclass(frozen=True)
class LeafStats:
    """Leaf-occupancy statistics for a single facet codebook."""

    facet: str
    leaves_total: int
    leaves_empty: int
    leaves_with_more_than_three: int
    papers_per_leaf_min: int
    papers_per_leaf_median: int
    papers_per_leaf_max: int
    fraction_empty: float
    fraction_more_than_three: float

    def passes_gate(self) -> bool:
        """Spec gate: fewer than 40% empty AND fewer than 20% >3-paper."""
        return self.fraction_empty < 0.40 and self.fraction_more_than_three < 0.20


def _load_facet_vectors(
    dense_dir: Path, facet: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Load corpus ids and mean-pooled per-facet vectors for one facet."""
    path = dense_dir / f"facet_graph_{facet}.npz"
    f = np.load(path)
    corpusids = np.asarray(f["corpusids"], dtype=np.int64)
    # The matrices on disk are float16 for size; cluster in float32.
    vectors = np.asarray(f["corpus_vectors"], dtype=np.float32)
    if vectors.shape[0] != corpusids.shape[0]:
        raise ValueError(
            f"{path}: corpus_vectors rows {vectors.shape[0]} does not match "
            f"corpusids length {corpusids.shape[0]}"
        )
    return corpusids, vectors


def _fit_level(
    vectors: npt.NDArray[np.float32], k: int, *, random_state: int, n_init: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Fit a k-means level over ``vectors``.

    Returns the cluster centers (``k, D``) and a label per row (``N``).
    If ``k`` exceeds the number of available points, each point becomes
    its own cluster and the missing centers are filled by repeating the
    last point — those clusters will simply have no extra members.
    """
    n = vectors.shape[0]
    effective_k = min(k, n)
    km = KMeans(
        n_clusters=effective_k,
        n_init=n_init,
        random_state=random_state,
    )
    labels = km.fit_predict(vectors).astype(np.int64)
    centers = np.asarray(km.cluster_centers_, dtype=np.float32)

    if effective_k < k:
        # Pad with repeats of the last center so the on-disk shape is uniform
        # across L0 clusters. Padded clusters are unreachable through training
        # because no point is assigned to them.
        padding = np.repeat(centers[-1:], k - effective_k, axis=0)
        centers = np.concatenate([centers, padding], axis=0)

    return centers, labels


@dataclass(frozen=True)
class FacetCodebook:
    """In-memory codebook for one facet."""

    facet: str
    level0_centers: npt.NDArray[np.float32]
    """L0 cluster centers, shape ``(k0, D)``."""

    level1_centers: npt.NDArray[np.float32]
    """L1 cluster centers stacked per parent L0, shape ``(k0, k1, D)``."""

    paper_codes: dict[CorpusId, tuple[int, int]]
    """Map from corpus id to ``(L0, L1)``."""


def fit_facet_codebook(
    facet: str,
    corpusids: Sequence[CorpusId],
    vectors: npt.NDArray[np.float32],
    params: CodebookParams,
) -> FacetCodebook:
    """Fit a two-level codebook for one facet."""
    logger.info(
        "Fitting %s codebook with k0=%d, k1=%d over %d papers",
        facet,
        params.k0,
        params.k1,
        len(corpusids),
    )

    level0_centers, l0_labels = _fit_level(
        vectors,
        params.k0,
        random_state=params.random_state,
        n_init=params.n_init,
    )

    dim = vectors.shape[1]
    level1_centers = np.zeros((params.k0, params.k1, dim), dtype=np.float32)
    l1_labels = np.zeros(len(corpusids), dtype=np.int64)

    for l0 in range(params.k0):
        idx = np.flatnonzero(l0_labels == l0)
        if idx.size == 0:
            continue
        if idx.size < params.min_l1_samples:
            l1_labels[idx] = 0
            level1_centers[l0, 0] = vectors[idx[0]]
            continue
        centers, labels = _fit_level(
            vectors[idx],
            params.k1,
            random_state=params.random_state + l0,
            n_init=params.n_init,
        )
        level1_centers[l0] = centers
        l1_labels[idx] = labels

    paper_codes: dict[CorpusId, tuple[int, int]] = {
        CorpusId(int(cid)): (int(l0), int(l1))
        for cid, l0, l1 in zip(corpusids, l0_labels, l1_labels, strict=True)
    }

    return FacetCodebook(
        facet=facet,
        level0_centers=level0_centers,
        level1_centers=level1_centers,
        paper_codes=paper_codes,
    )


def _build_code_to_papers(
    codebook: FacetCodebook, params: CodebookParams
) -> dict[str, list[int]]:
    """Build a `"L0,L1" -> [corpusids]` index for a facet codebook."""
    out: dict[str, list[int]] = {}
    for cid in codebook.paper_codes:
        l0, l1 = codebook.paper_codes[cid]
        key = f"{l0},{l1}"
        out.setdefault(key, []).append(int(cid))
    # Sort the per-leaf lists so the output is deterministic.
    for key in out:
        out[key].sort()
    # Include empty leaves so the lookup is total.
    for l0 in range(params.k0):
        for l1 in range(params.k1):
            key = f"{l0},{l1}"
            out.setdefault(key, [])
    return out


def _leaf_stats(
    facet: str, codebook: FacetCodebook, params: CodebookParams
) -> LeafStats:
    """Compute leaf-occupancy stats for one facet codebook."""
    counts: list[int] = []
    occupancy: dict[tuple[int, int], int] = {}
    for code in codebook.paper_codes.values():
        occupancy[code] = occupancy.get(code, 0) + 1
    for l0 in range(params.k0):
        for l1 in range(params.k1):
            counts.append(occupancy.get((l0, l1), 0))

    counts_arr = np.asarray(counts, dtype=np.int64)
    empty = int((counts_arr == 0).sum())
    big = int((counts_arr > 3).sum())
    total = int(counts_arr.size)

    return LeafStats(
        facet=facet,
        leaves_total=total,
        leaves_empty=empty,
        leaves_with_more_than_three=big,
        papers_per_leaf_min=int(counts_arr.min()),
        papers_per_leaf_median=int(np.median(counts_arr)),
        papers_per_leaf_max=int(counts_arr.max()),
        fraction_empty=empty / total,
        fraction_more_than_three=big / total,
    )


def _write_facet_artifacts(
    output_dir: Path,
    codebook: FacetCodebook,
    params: CodebookParams,
) -> dict[str, list[int]]:
    """Persist codebook artifacts for a single facet; returns its code index."""
    facet_dir = output_dir / codebook.facet
    facet_dir.mkdir(parents=True, exist_ok=True)

    np.save(facet_dir / "level0_centers.npy", codebook.level0_centers)
    np.save(facet_dir / "level1_centers.npy", codebook.level1_centers)

    with (facet_dir / "paper_codes.jsonl").open("w") as fh:
        for cid in sorted(codebook.paper_codes):
            l0, l1 = codebook.paper_codes[cid]
            fh.write(json.dumps({"corpusid": int(cid), "code": [l0, l1]}) + "\n")

    return _build_code_to_papers(codebook, params)


def build_codebooks(
    dense_dir: Path,
    output_dir: Path,
    params: CodebookParams,
    *,
    facets: Iterable[str] = FACET_NAMES,
) -> dict[str, LeafStats]:
    """Fit codebooks for all facets and write artifacts to ``output_dir``.

    Returns per-facet leaf stats so callers can apply the validation gate.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    code_to_papers: dict[str, dict[str, list[int]]] = {}
    stats: dict[str, LeafStats] = {}

    for facet in facets:
        corpusids, vectors = _load_facet_vectors(dense_dir, facet)
        codebook = fit_facet_codebook(
            facet,
            [CorpusId(int(x)) for x in corpusids],
            vectors,
            params,
        )
        code_to_papers[facet] = _write_facet_artifacts(output_dir, codebook, params)
        stats[facet] = _leaf_stats(facet, codebook, params)

    (output_dir / "code_to_papers.json").write_text(
        json.dumps(code_to_papers, indent=2) + "\n"
    )

    validation = {
        "params": {
            "k0": params.k0,
            "k1": params.k1,
            "random_state": params.random_state,
            "n_init": params.n_init,
            "min_l1_samples": params.min_l1_samples,
        },
        "facets": {
            facet: {
                "leaves_total": s.leaves_total,
                "leaves_empty": s.leaves_empty,
                "leaves_with_more_than_three": s.leaves_with_more_than_three,
                "papers_per_leaf_min": s.papers_per_leaf_min,
                "papers_per_leaf_median": s.papers_per_leaf_median,
                "papers_per_leaf_max": s.papers_per_leaf_max,
                "fraction_empty": s.fraction_empty,
                "fraction_more_than_three": s.fraction_more_than_three,
                "passes_gate": s.passes_gate(),
            }
            for facet, s in stats.items()
        },
    }
    (output_dir / "validation_stats.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )

    return stats
