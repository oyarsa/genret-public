"""Per-pair graph edge weights and their sampling / loss-scaling derivations."""

from __future__ import annotations

import logging
from pathlib import Path

from genret.training.codebook_data import FACET_NAMES
from genret.types import CorpusId

logger = logging.getLogger(__name__)


def _load_facet_pair_weights(
    candidates_dir: Path,
) -> dict[tuple[str, CorpusId, CorpusId], float]:
    """Load per-(facet, a, b) ``scores_combined`` from ``facet_candidates_*.npz``.

    Returns ``{(facet, paper_a, paper_b): score}``. Pairs not present in
    any facet's per-query top-100 are absent (callers must treat absent
    as 0).
    """
    import numpy as np  # noqa: PLC0415

    out: dict[tuple[str, CorpusId, CorpusId], float] = {}
    for facet in FACET_NAMES:
        path = candidates_dir / f"facet_candidates_{facet}.npz"
        if not path.exists():
            logger.warning("%s missing; skipping in pair-weight load", path)
            continue
        f = np.load(path)
        cids = f["corpusids"].astype(np.int64)
        candidates = f["candidates"].astype(np.int64)
        scores = f["scores_combined"].astype(np.float32)
        for i in range(cids.shape[0]):
            a = CorpusId(int(cids[i]))
            for j_pos, j_idx in enumerate(candidates[i]):
                if int(j_idx) < 0:
                    continue
                b = CorpusId(int(cids[int(j_idx)]))
                out[(facet, a, b)] = float(scores[i, j_pos])
    logger.info(
        "Loaded %d (facet, a, b) graph edge weights from %s",
        len(out),
        candidates_dir,
    )
    return out


def _symmetric_weight(
    pair_weights: dict[tuple[str, CorpusId, CorpusId], float],
    facet: str,
    a: CorpusId,
    b: CorpusId,
) -> float:
    """``scores_combined`` for edge (a, b), trying either stored direction.

    The candidate graph stores only the top-K-per-query direction, so a needed
    (a, b) edge may be present only as (b, a). ``scores_combined`` is symmetric
    in value, so return whichever direction exists (0.0 if neither).
    """
    w = pair_weights.get((facet, a, b))
    if w is not None:
        return w
    return pair_weights.get((facet, b, a), 0.0)


def _replication_from_weight(w: float, facet_max: float, max_copies: int) -> int:
    """Integer replication factor for an explicit edge weight.

    Scales ``w`` (relative to the per-facet max) linearly to ``[1, max_copies]``;
    returns 1 for a non-positive weight or facet max. Used both for looked-up
    pair weights and for synthetic pseudo-gold rows that carry a precomputed
    (symmetric) edge weight.
    """
    if w <= 0 or facet_max <= 0:
        return 1
    scaled = w / facet_max  # in (0, 1]
    return max(1, round(scaled * max_copies))


def _pair_replication_count(
    paper_a: CorpusId,
    paper_b: CorpusId,
    facet: str,
    pair_weights: dict[tuple[str, CorpusId, CorpusId], float] | None,
    max_weight_for_facet: dict[str, float],
    max_copies: int,
) -> int:
    """Integer replication factor for one (paper_a, paper_b, facet) pair.

    Returns 1 (no replication) when graph weighting is disabled. Otherwise
    scales the pair's ``scores_combined`` (per-facet) linearly to the
    range [1, max_copies].
    """
    if pair_weights is None:
        return 1
    w = pair_weights.get((facet, paper_a, paper_b), 0.0)
    return _replication_from_weight(w, max_weight_for_facet.get(facet, 0.0), max_copies)


def _pair_loss_weight(
    paper_a: CorpusId,
    paper_b: CorpusId,
    facet: str,
    pair_weights: dict[tuple[str, CorpusId, CorpusId], float] | None,
    max_weight_for_facet: dict[str, float],
    max_weight: float,
    enabled: bool,
) -> float:
    """Continuous per-row CE weight for one (paper_a, paper_b, facet) pair.

    Returns 1.0 when reweighting is disabled or the pair has no graph edge.
    Otherwise scales the pair's ``scores_combined`` (per-facet) linearly to
    the range ``[1.0, max_weight]`` — the continuous analogue of
    ``_pair_replication_count``.
    """
    if not enabled or pair_weights is None:
        return 1.0
    w = pair_weights.get((facet, paper_a, paper_b), 0.0)
    if w <= 0:
        return 1.0
    facet_max = max_weight_for_facet.get(facet, 0.0)
    if facet_max <= 0:
        return 1.0
    scaled = w / facet_max  # in (0, 1]
    return max(1.0, scaled * max_weight)
