"""Graph-smoothed GLEN bullet DocIDs.

Rewrites each bullet DocID's *text* so its tokens are predictable from the
queries that should retrieve it. In GLEN, queries are bullets of related
papers, so a DocID word that the paper's graph neighbourhood never uses is
a token the decoder must generate with no evidence in the query. Smoothing
keeps the bullet's content words (original order) that are attested in the
neighbourhood's same-facet text and drops idiosyncratic ones.

Scope guarantees:

- One smoothed string per (paper, facet, bullet): GLEN's
  multiple-DocIDs-per-paper property is preserved 1:1.
- Only DocID *target* text changes. Training prompts and eval queries
  (paper_a bullets) stay verbatim — the real query distribution.
- Both the GLEN index build and the training-data builder consume the same
  precomputed map, keyed by ``(corpusid, facet, normalised bullet)``, so
  trie entries and training targets always agree.

The neighbourhood comes from the facet candidates graph
(``facet_candidates_<facet>.npz``: per-paper top-100 candidates with
``scores_combined``), the same graph Option B uses for weighted sampling.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from genret.paper_parsing import FacetRecordType, facets_from_type
from genret.training.cluster_keyword_labels import _tokenise_to_words
from genret.training.codebook_data import FACET_NAMES
from genret.training.glen_docids import normalise_facet_text, split_bullets
from genret.types import CorpusId
from genret.utils import iter_jsonl

logger = logging.getLogger(__name__)


SmoothedBulletMap = dict[tuple[CorpusId, str, str], str]
"""``(corpusid, facet, normalised original bullet) -> smoothed DocID text``."""


def smooth_bullet_words(
    words: list[str],
    attestation: Mapping[str, int],
    *,
    min_attest: int = 2,
    min_words: int = 4,
) -> list[str]:
    """Order-preserving neighbourhood filter over one bullet's content words.

    Keeps words attested by at least ``min_attest`` neighbours. When fewer
    than ``min_words`` survive, pads with the earliest unattested words (in
    original order) so the DocID keeps enough identity to stay unique.
    """
    flags = [attestation.get(w, 0) >= min_attest for w in words]
    if sum(flags) >= min_words:
        return [w for w, keep in zip(words, flags) if keep]
    keep_idx = {i for i, keep in enumerate(flags) if keep}
    for i in range(len(words)):
        if len(keep_idx) >= min_words:
            break
        keep_idx.add(i)
    return [words[i] for i in sorted(keep_idx)]


def _load_neighbours(
    candidates_dir: Path, facet: str, top_n: int
) -> dict[CorpusId, list[CorpusId]]:
    """Top-``top_n`` graph neighbours per paper by ``scores_combined``."""
    path = candidates_dir / f"facet_candidates_{facet}.npz"
    f = np.load(path)
    cids = f["corpusids"].astype(np.int64)
    candidates = f["candidates"].astype(np.int64)
    scores = f["scores_combined"].astype(np.float32)
    out: dict[CorpusId, list[CorpusId]] = {}
    for i in range(cids.shape[0]):
        order = np.argsort(-scores[i])
        neighbours: list[CorpusId] = []
        for j in order:
            idx = int(candidates[i, int(j)])
            if idx < 0:
                continue
            neighbours.append(CorpusId(int(cids[idx])))
            if len(neighbours) >= top_n:
                break
        out[CorpusId(int(cids[i]))] = neighbours
    return out


def build_smoothed_bullets(
    facets_path: Path,
    candidates_dir: Path,
    output_path: Path,
    *,
    top_neighbours: int = 20,
    min_attest: int = 2,
    min_words: int = 4,
    facets_type: FacetRecordType = FacetRecordType.STANDARD,
) -> dict[str, float]:
    """Precompute the smoothed DocID text for every (paper, facet, bullet).

    Writes JSONL rows ``{corpusid, facet, original, smoothed}`` where
    ``original`` is the normalised bullet (the join key) and ``smoothed``
    is the graph-filtered DocID text. Returns summary stats.
    """
    facet_words: dict[tuple[str, CorpusId], set[str]] = {}
    paper_bullets: dict[CorpusId, dict[str, list[str]]] = {}
    for record in iter_jsonl(facets_path):
        facets_obj = facets_from_type(record, facets_type)
        cid = facets_obj.corpusid
        per_facet: dict[str, list[str]] = {}
        for facet in FACET_NAMES:
            items = getattr(facets_obj, facet, None) or []
            bullets: list[str] = []
            for item in items:
                for b in split_bullets(str(item)):
                    norm = normalise_facet_text(b)
                    if norm:
                        bullets.append(norm)
            per_facet[facet] = bullets
            facet_words[(facet, cid)] = {
                w for b in bullets for w in _tokenise_to_words(b)
            }
        paper_bullets[cid] = per_facet

    n_bullets = 0
    n_verbatim = 0
    kept_fractions: list[float] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for facet in FACET_NAMES:
            neighbours = _load_neighbours(candidates_dir, facet, top_neighbours)
            for cid, per_facet in paper_bullets.items():
                neigh_sets = [
                    facet_words.get((facet, nb), set())
                    for nb in neighbours.get(cid, [])
                ]
                for norm in per_facet[facet]:
                    n_bullets += 1
                    words = _tokenise_to_words(norm)
                    if not words:
                        smoothed = norm
                        n_verbatim += 1
                    else:
                        attestation = {
                            w: sum(w in s for s in neigh_sets) for w in set(words)
                        }
                        kept = smooth_bullet_words(
                            words,
                            attestation,
                            min_attest=min_attest,
                            min_words=min_words,
                        )
                        smoothed = " ".join(kept) if kept else norm
                        kept_fractions.append(len(kept) / len(words))
                    fh.write(
                        json.dumps(
                            {
                                "corpusid": int(cid),
                                "facet": facet,
                                "original": norm,
                                "smoothed": smoothed,
                            }
                        )
                        + "\n"
                    )

    stats = {
        "n_bullets": float(n_bullets),
        "verbatim_fallback_fraction": n_verbatim / n_bullets if n_bullets else 0.0,
        "mean_kept_word_fraction": (
            float(np.mean(kept_fractions)) if kept_fractions else 0.0
        ),
    }
    logger.info(
        "Smoothed %d bullets: mean kept-word fraction %.2f, %.1f%% verbatim fallback",
        n_bullets,
        stats["mean_kept_word_fraction"],
        100 * stats["verbatim_fallback_fraction"],
    )
    return stats


def load_smoothed_bullets(path: Path) -> SmoothedBulletMap:
    """Read back ``(corpusid, facet, original) -> smoothed`` from JSONL."""
    out: SmoothedBulletMap = {}
    for row in iter_jsonl(path):
        key = (
            CorpusId(int(row["corpusid"])),
            str(row["facet"]),
            str(row["original"]),
        )
        out[key] = str(row["smoothed"])
    return out


__all__ = [
    "SmoothedBulletMap",
    "build_smoothed_bullets",
    "load_smoothed_bullets",
    "smooth_bullet_words",
]
