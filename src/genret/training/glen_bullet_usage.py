"""Per-bullet DocID usage analysis for GLEN training and retrieval.

Answers "which bullet DocIDs are actually used?" from two sides:

1. **Training views**: rebuilds the training dataset with the exact run
   config (one facet at a time, so every example attributes to its facet)
   and counts views per (paper, facet, bullet) target. Shows how many
   bullets ride the ``min_target_views`` floor vs earn organic views.
2. **Decode hits** (optional): consumes the side JSONL written by
   ``glen evaluate --record-docids`` and counts, per stored bullet DocID,
   how often it appears in any beam across eval queries. Dead DocIDs
   (never decoded) are trie weight that training paid for with zero
   retrieval return.

When a graph-smoothed bullet map is given (``glen smooth-bullets``
output), each bullet also gets its neighbourhood-attested word fraction,
and the report includes the rank correlation between attestation and
usage — the direct test of "are unreachable bullets the ones whose words
the neighbourhood never uses?".
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from genret._shed.transformers import decode_tokens
from genret.paper_parsing import FacetRecordType, facets_from_type
from genret.training.cluster_keyword_labels import _tokenise_to_words
from genret.training.codebook_data import FACET_NAMES
from genret.training.glen_bullet_dataset import (
    GlenBulletDatasetConfig,
    _build_bullet_examples,
)
from genret.training.glen_docids import (
    extract_docid_tokens,
    normalise_facet_text,
    split_bullets,
)
from genret.types import CorpusId
from genret.utils import iter_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class BulletUsage:
    """Usage record for one (paper, facet, bullet) DocID."""

    corpusid: CorpusId
    facet: str
    bullet: str
    train_views: int
    decode_hits: int | None
    attested_fraction: float | None


def _bullet_universe(
    facets_path: Path, facets_type: FacetRecordType
) -> dict[str, list[tuple[CorpusId, str]]]:
    """All (paper, normalised bullet) per facet, matching target enumeration."""
    out: dict[str, list[tuple[CorpusId, str]]] = {f: [] for f in FACET_NAMES}
    for record in iter_jsonl(facets_path):
        facets_obj = facets_from_type(record, facets_type)
        for facet in FACET_NAMES:
            items = getattr(facets_obj, facet, None) or []
            for item in items:
                for b in split_bullets(str(item)):
                    norm = normalise_facet_text(b)
                    if norm:
                        out[facet].append((facets_obj.corpusid, norm))
    return out


def _attested_fractions(
    smoothed_bullets_path: Path,
) -> dict[tuple[CorpusId, str, str], float]:
    """``(cid, facet, original) -> attested word fraction`` from the map."""
    out: dict[tuple[CorpusId, str, str], float] = {}
    for row in iter_jsonl(smoothed_bullets_path):
        original = str(row["original"])
        words = _tokenise_to_words(original)
        if not words:
            continue
        kept = len(str(row["smoothed"]).split())
        key = (CorpusId(int(row["corpusid"])), str(row["facet"]), original)
        out[key] = min(1.0, kept / len(words))
    return out


def _decode_hit_counts(
    decoded_docids_path: Path,
) -> dict[str, Counter[str]]:
    """``facet -> Counter(decoded docid text -> beam appearances)``."""
    out: dict[str, Counter[str]] = {}
    for row in iter_jsonl(decoded_docids_path):
        facet = str(row["facet"])
        out.setdefault(facet, Counter())[str(row["docid"]).strip()] += 1
    return out


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation; 0.0 when degenerate."""
    if len(xs) < 3:
        return 0.0
    rx = np.argsort(np.argsort(xs)).astype(np.float64)
    ry = np.argsort(np.argsort(ys)).astype(np.float64)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze_bullet_usage(
    *,
    pairs_path: Path,
    facets_path: Path,
    paper_graphs_path: Path,
    facets_type: FacetRecordType = FacetRecordType.STANDARD,
    split_ids: set[CorpusId] | None = None,
    graph_sampling: Literal["uniform", "weighted"] = "weighted",
    candidates_dir: Path | None = None,
    graph_sampling_max_copies: int = 5,
    min_target_views: int = 3,
    target_bullet: Literal["random", "longest", "cartesian"] = "random",
    source_bullets: Literal["all", "longest"] = "all",
    seed: int = 42,
    smoothed_bullets_path: Path | None = None,
    decoded_docids_path: Path | None = None,
    tokeniser: PreTrainedTokenizerBase | None = None,
    max_docid_tokens: int = 12,
) -> tuple[list[BulletUsage], dict[str, dict[str, float]]]:
    """Per-bullet usage records + per-facet summary.

    Training views are counted by rebuilding the dataset one facet at a
    time with the given config (mirroring the training run), so every
    example's facet is known without changing the dataset schema.

    Decode hits require both ``decoded_docids_path`` (from ``glen
    evaluate --record-docids``) and ``tokeniser`` (to map each stored
    bullet to its truncated decoded form, the join key).
    """
    import random  # noqa: PLC0415

    universe = _bullet_universe(facets_path, facets_type)
    attested = (
        _attested_fractions(smoothed_bullets_path)
        if smoothed_bullets_path is not None
        else None
    )
    decode_hits = (
        _decode_hit_counts(decoded_docids_path)
        if decoded_docids_path is not None
        else None
    )
    if decode_hits is not None and tokeniser is None:
        raise ValueError("decoded_docids_path requires a tokeniser for the join")

    records: list[BulletUsage] = []
    summary: dict[str, dict[str, float]] = {}
    for facet in FACET_NAMES:
        config = GlenBulletDatasetConfig(
            pairs_path=pairs_path,
            facets_path=facets_path,
            paper_graphs_path=paper_graphs_path,
            facets_type=facets_type,
            split_ids=split_ids,
            facets=(facet,),
            source_bullets=source_bullets,
            target_bullet=target_bullet,
            graph_sampling=graph_sampling,
            candidates_dir=candidates_dir,
            graph_sampling_max_copies=graph_sampling_max_copies,
            min_target_views=min_target_views,
        )
        examples = _build_bullet_examples(config, rng=random.Random(seed))
        views: Counter[str] = Counter(ex.target_docid for ex in examples)

        facet_records: list[BulletUsage] = []
        for cid, norm in universe[facet]:
            hits: int | None = None
            if decode_hits is not None and tokeniser is not None:
                toks = extract_docid_tokens(
                    norm, tokeniser, max_tokens=max_docid_tokens
                )
                decoded = decode_tokens(tokeniser, toks).strip()
                hits = decode_hits.get(facet, Counter()).get(decoded, 0)
            facet_records.append(
                BulletUsage(
                    corpusid=cid,
                    facet=facet,
                    bullet=norm,
                    train_views=views.get(norm, 0),
                    decode_hits=hits,
                    attested_fraction=(
                        attested.get((cid, facet, norm))
                        if attested is not None
                        else None
                    ),
                )
            )
        records.extend(facet_records)

        view_counts = [r.train_views for r in facet_records]
        facet_summary: dict[str, float] = {
            "n_bullets": float(len(facet_records)),
            "views_median": float(np.median(view_counts)) if view_counts else 0.0,
            "views_p90": (
                float(np.percentile(view_counts, 90)) if view_counts else 0.0
            ),
            "share_zero_views": (
                sum(v == 0 for v in view_counts) / len(view_counts)
                if view_counts
                else 0.0
            ),
            "share_at_floor": (
                sum(v == min_target_views for v in view_counts) / len(view_counts)
                if view_counts
                else 0.0
            ),
        }
        pairs_with_attest = [
            (float(r.train_views), r.attested_fraction)
            for r in facet_records
            if r.attested_fraction is not None
        ]
        if pairs_with_attest:
            facet_summary["spearman_views_vs_attested"] = _spearman(
                [v for v, _ in pairs_with_attest],
                [a for _, a in pairs_with_attest if a is not None],
            )
        hit_counts = [r.decode_hits for r in facet_records if r.decode_hits is not None]
        if hit_counts:
            facet_summary["share_never_decoded"] = sum(
                h == 0 for h in hit_counts
            ) / len(hit_counts)
            pairs_with_hits = [
                (float(r.decode_hits or 0), r.attested_fraction)
                for r in facet_records
                if r.decode_hits is not None and r.attested_fraction is not None
            ]
            if pairs_with_hits:
                facet_summary["spearman_hits_vs_attested"] = _spearman(
                    [h for h, _ in pairs_with_hits],
                    [a for _, a in pairs_with_hits if a is not None],
                )
        summary[facet] = facet_summary

    return records, summary


def save_usage_report(
    records: list[BulletUsage],
    summary: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Write per-bullet records (JSONL) + summary (JSON) next to each other."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for r in records:
            fh.write(
                json.dumps(
                    {
                        "corpusid": int(r.corpusid),
                        "facet": r.facet,
                        "bullet": r.bullet,
                        "train_views": r.train_views,
                        "decode_hits": r.decode_hits,
                        "attested_fraction": r.attested_fraction,
                    }
                )
                + "\n"
            )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Usage report: %s + %s", output_path, summary_path)


__all__ = [
    "BulletUsage",
    "analyze_bullet_usage",
    "save_usage_report",
]
