"""Per-row hard negative pools and bullet sampling from graph candidates."""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from pathlib import Path

from genret.paper_parsing import Facets
from genret.training.codebook_data import FACET_NAMES
from genret.types import CorpusId

from .selection import _all_target_bullets

logger = logging.getLogger(__name__)


def _load_hard_negative_pools(
    candidates_dir: Path,
    facets: tuple[str, ...],
    pool_top_k: int,
) -> dict[tuple[str, CorpusId], list[CorpusId]]:
    """Load per-(facet, paper_a) top-K candidate lists from ``facet_candidates_*.npz``.

    Returns ``{(facet, paper_a): [paper_b, ...]}`` ordered by candidate
    rank (strongest first), truncated to ``pool_top_k``. Gold-masking is
    done at sample time by the caller.
    """
    import numpy as np  # noqa: PLC0415

    out: dict[tuple[str, CorpusId], list[CorpusId]] = {}
    for facet in facets:
        if facet not in FACET_NAMES:
            continue
        path = candidates_dir / f"facet_candidates_{facet}.npz"
        if not path.exists():
            logger.warning("%s missing; skipping in hard-neg pool load", path)
            continue
        f = np.load(path)
        cids = f["corpusids"].astype(np.int64)
        candidates = f["candidates"].astype(np.int64)
        for i in range(cids.shape[0]):
            a = CorpusId(int(cids[i]))
            pool: list[CorpusId] = []
            for j_idx in candidates[i, :pool_top_k]:
                if int(j_idx) < 0:
                    continue
                b = CorpusId(int(cids[int(j_idx)]))
                if b != a:
                    pool.append(b)
            out[(facet, a)] = pool
    logger.info(
        "Loaded hard-neg candidate pools for %d (facet, paper_a) keys (top_k=%d)",
        len(out),
        pool_top_k,
    )
    return out


def _sample_hard_negative_bullets(
    paper_a: CorpusId,
    facet: str,
    gold_neighbours: frozenset[CorpusId],
    pool: list[CorpusId],
    facets_map: Mapping[CorpusId, Facets],
    num_neg: int,
    rng: random.Random,
) -> tuple[str, ...]:
    """Sample K hard-negative bullets for one (paper_a, facet) training row.

    Walks the candidate pool in rank order, skipping gold-masked papers,
    and emits one uniformly-sampled bullet per negative paper. If the
    pool yields fewer than ``num_neg`` after gold-masking, returns the
    available set (shorter is OK; trainer averages over what is given).
    """
    out: list[str] = []
    for cand in pool:
        if cand in gold_neighbours or cand == paper_a:
            continue
        cand_facets = facets_map.get(cand)
        if cand_facets is None:
            continue
        bullets = _all_target_bullets(cand_facets, facet)
        if not bullets:
            continue
        out.append(rng.choice(bullets))
        if len(out) >= num_neg:
            break
    return tuple(out)
