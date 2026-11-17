"""Prune candidate edges to top-K and build sparse CSR edge graphs.

Takes the top-100 candidate edges from `facet_graph_build.py` and produces
the final sparse graph stored as CSR arrays in `paper_edges_{name}.npz`.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from scipy.sparse import csr_array

from genret.processing.facet_evaluation import FACET_NAMES
from genret.utils import safediv

logger = logging.getLogger(__name__)


class EdgeGraphParams(BaseModel, frozen=True):
    """Parameters used to build the sparse edge graph.

    Written to `FILENAME` in the output directory after the job completes.
    """

    FILENAME: ClassVar[str] = "edge_graph_params.json"

    top_k: int
    """Number of edges retained per paper."""
    min_score: float
    """Minimum combined score threshold for keeping an edge."""
    candidates_dir: str
    """Directory containing the `facet_candidates_*.npz` input files."""
    created_at: str
    """ISO-8601 UTC timestamp when the job completed."""


def _prune_candidates(
    candidates: npt.NDArray[np.int32],
    scores: npt.NDArray[np.float16],
    n: int,
    top_k: int,
    min_score: float,
) -> csr_array:
    """Prune candidates per paper and return a CSR sparse matrix.

    Candidates are already sorted descending by combined score. Slices to
    `top_k` and drops entries below `min_score` or with negative index
    (padding).

    Args:
        candidates: (N, K_in) candidate indices into corpusids.
        scores: (N, K_in) combined scores, sorted descending per row.
        n: Number of papers (matrix dimension).
        top_k: Maximum edges to keep per paper.
        min_score: Drop edges with score below this threshold.

    Returns:
        Sparse CSR matrix of shape `(n, n)` with pruned edge scores.
    """
    actual_k = min(top_k, candidates.shape[1])

    # Slice to top_k
    cand_slice = candidates[:, :actual_k]
    score_slice = scores[:, :actual_k].astype(np.float32)

    # Build COO entries, filtering invalid/below-threshold
    rows = np.repeat(np.arange(n, dtype=np.int32), actual_k)
    cols = cand_slice.ravel()
    data = score_slice.ravel()

    valid = (cols >= 0) & (data >= min_score)
    rows_valid = rows[valid]
    cols_valid = cols[valid]
    data_valid = data[valid]

    return csr_array((data_valid, (rows_valid, cols_valid)), shape=(n, n))


def _build_facet_edges(
    facet_name: str,
    candidates_dir: Path,
    output_dir: Path,
    top_k: int,
    min_score: float,
) -> None:
    """Build and write the sparse edge graph for one facet type.

    Args:
        facet_name: Name of the facet type (e.g. `"problems"`).
        candidates_dir: Directory containing `facet_candidates_*.npz` files.
        output_dir: Directory to write the output NPZ file.
        top_k: Maximum edges to keep per paper.
        min_score: Minimum combined score threshold.
    """
    npz_path = candidates_dir / f"facet_candidates_{facet_name}.npz"
    out_path = output_dir / f"paper_edges_{facet_name}.npz"
    tmp_path = output_dir / f"paper_edges_{facet_name}.tmp.npz"

    if not npz_path.exists():
        logger.warning(f"Missing {npz_path}, skipping")
        return

    logger.info(f"=== {facet_name} ===")
    with np.load(npz_path) as npz:
        corpusids: npt.NDArray[np.int64] = npz["corpusids"]
        candidates: npt.NDArray[np.int32] = npz["candidates"]
        scores_combined: npt.NDArray[np.float16] = npz["scores_combined"]

    n = len(corpusids)
    graph = _prune_candidates(candidates, scores_combined, n, top_k, min_score)
    graph.sort_indices()

    avg_degree = safediv(graph.nnz, n)
    logger.info(
        f"  {graph.nnz} edges ({avg_degree:.1f} avg degree), "
        f"top_k={top_k}, min_score={min_score}"
    )

    np.savez_compressed(
        tmp_path,
        corpusids=corpusids,
        indptr=graph.indptr,
        indices=graph.indices,
        data=np.asarray(graph.data, dtype=np.float16),
    )
    tmp_path.rename(out_path)
    logger.info(f"  Saved {out_path}")


def build_edges(
    candidates_dir: Path,
    output_dir: Path,
    top_k: int = 20,
    min_score: float = 0.0,
) -> None:
    """Build sparse edge graphs for all 4 facet types.

    For each facet, loads `facet_candidates_{name}.npz`, prunes to `top_k`
    edges per paper (dropping any below `min_score`), converts to CSR format,
    and writes `paper_edges_{name}.npz`. Also writes `EdgeGraphParams.FILENAME`.

    Args:
        candidates_dir: Directory containing `facet_candidates_*.npz` files.
        output_dir: Directory to write `paper_edges_*.npz` and params JSON.
        top_k: Maximum edges to keep per paper per facet.
        min_score: Minimum combined score threshold for keeping an edge.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process standard facets
    for facet_name in FACET_NAMES:
        _build_facet_edges(
            facet_name=facet_name,
            candidates_dir=candidates_dir,
            output_dir=output_dir,
            top_k=top_k,
            min_score=min_score,
        )

    # Process any cross-facet candidate graphs
    for cand_path in sorted(candidates_dir.glob("facet_candidates_*_x_*.npz")):
        cross_name = cand_path.stem.removeprefix("facet_candidates_")
        _build_facet_edges(
            facet_name=cross_name,
            candidates_dir=candidates_dir,
            output_dir=output_dir,
            top_k=top_k,
            min_score=min_score,
        )

    params = EdgeGraphParams(
        top_k=top_k,
        min_score=min_score,
        candidates_dir=str(candidates_dir.resolve()),
        created_at=datetime.now(UTC).isoformat(),
    )
    params_path = output_dir / EdgeGraphParams.FILENAME
    params_path.write_text(params.model_dump_json(indent=2) + "\n")
    logger.info(f"Params written to {params_path}")
