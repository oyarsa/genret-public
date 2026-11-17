"""MINDER-style fusion of per-facet GLEN decodes into a paper ranking.

Consumes the `--record-docids` dump written by `genret training glen evaluate`
(one row per decoded beam: query, facet, rank, docid, papers) and re-fuses it
into the same JSONL schema `score_glen_eval.py` already scores, so every
existing metric, gold source and graph-RRF flag applies unchanged.

Where our deployed fusion assigns each paper one reciprocal-rank term per facet
(`_fuse_facet_results`), MINDER sums evidence over *every* predicted identifier
a paper covers (Li et al., ACL 2023, Eq. 3). Two differences are therefore in
play -- evidence accumulation and the per-identifier weight -- and the levels
below separate them:

- `uniform-rrf`: reproduces `_fuse_facet_results` exactly. Validation only; the
  output must match the stored eval file for the same run.
- `b0`: rank-additive coverage. Eq. 3's accumulation with a reciprocal-rank
  weight instead of a language-model score, so it isolates accumulation and
  needs no beam scores. Runnable on dumps that predate score recording.
- `m0`: Eq. 3 as written -- sum of language-model scores over covered
  identifiers, plus the paper's per-view offset. Requires `score` in the dump.
- `e1`: `m0` plus the background-frequency normalisation from the authors'
  released code, which their paper omits (see
  `docs/plans/20260725-minder-baseline.md` §4.2.3). Requires `score`. Uses the
  same per-view additive offset as `m0` (`--offset`), not a per-token rescale:
  the released code's own fix for the same score/background magnitude
  mismatch is a flat offset (its hardcoded ``+14`` on the pseudo-query view),
  applied because that view's identifiers are far longer than the others in
  their setting. Our identifiers are all the same length (every decoded
  DocID is exactly `n_tokens`), so one offset suffices and is the smaller,
  more faithful deviation.

Usage:
    uv run python scripts/eval/minder_fusion.py DUMP --level b0 --output OUT.jsonl
    uv run python scripts/eval/score_glen_eval.py OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from genret.types import CorpusId
from genret.utils import iter_jsonl

type FusionLevel = Literal["uniform-rrf", "b0", "m0", "e1"]

DEFAULT_RRF_K = 60
"""RRF constant used by `_fuse_facet_results`, matched here for comparability."""
DEFAULT_TOP_N = 50
"""Results written per query, matching `evaluate_glen_retrieval`'s `top_n`."""
DEFAULT_SMOOTHING = 5.0
"""Additive smoothing on occurrence counts, from SEAL's `aggregate_evidence`."""


@dataclass(frozen=True, kw_only=True)
class Identifier:
    """One decoded identifier: a DocID string and the papers at its trie leaf."""

    facet: str
    """View the identifier was decoded under (a facet, in Variant B)."""
    docid: str
    """Decoded DocID text, as recorded in the dump."""
    rank: int
    """Best (lowest) beam rank this identifier reached within its facet."""
    log_prob: float | None
    """Un-normalised sum of token log-probabilities, the paper's score.

    `None` when the dump predates score recording, in which case only the
    `b0` level is available. Used by both `m0` and `e1` -- see
    `identifier_weight` for why `e1` uses an additive offset on this rather
    than a per-token rescale.
    """
    papers: tuple[CorpusId, ...]
    """Papers stored at the identifier's trie leaf."""


def load_dump(path: Path) -> dict[CorpusId, list[Identifier]]:
    """Load a `--record-docids` dump, de-duplicating identifiers per query.

    An identifier decoded from several bullets of the same facet appears once
    per bullet in the dump. MINDER counts an identifier once per document
    ("occurs at least once"), so rows are collapsed on `(facet, docid)` keeping
    the best rank and the best score.

    Args:
        path: Dump JSONL written by `genret training glen evaluate
            --record-docids`.

    Returns:
        Mapping from query paper to its de-duplicated identifiers, in dump
        order (facet blocks in decode order, beams in beam order).
    """
    best: dict[CorpusId, dict[tuple[str, str], Identifier]] = defaultdict(dict)
    for row in iter_jsonl(path):
        query = CorpusId(int(row["query"]))
        facet: str = row["facet"]
        docid: str = row["docid"]
        rank = int(row["rank"])
        papers = tuple(CorpusId(int(p)) for p in row["papers"])
        log_prob = _row_log_prob(row)
        key = (facet, docid)
        current = best[query].get(key)
        if current is not None:
            best[query][key] = Identifier(
                facet=facet,
                docid=docid,
                rank=min(current.rank, rank),
                log_prob=_better_score(current.log_prob, log_prob),
                papers=current.papers,
            )
            continue
        best[query][key] = Identifier(
            facet=facet,
            docid=docid,
            rank=rank,
            log_prob=log_prob,
            papers=papers,
        )
    return {query: list(ids.values()) for query, ids in best.items()}


def _row_log_prob(row: dict[str, Any]) -> float | None:
    """Extract a dump row's un-normalised log-probability, if recorded.

    Prefers `logprob_sum` (computed from transition scores at decode time).
    Falls back to de-normalising `score`, which HuggingFace divided by the
    generated length at the default `length_penalty` of 1.0.
    """
    logprob_sum: float | None = row.get("logprob_sum")
    if logprob_sum is not None:
        return float(logprob_sum)
    score: float | None = row.get("score")
    if score is None:
        return None
    n_tokens = int(row.get("n_tokens", 0))
    return float(score) * n_tokens if n_tokens else float(score)


def _better_score(current: float | None, candidate: float | None) -> float | None:
    """Return the higher of two optional language-model scores."""
    if current is None:
        return candidate
    if candidate is None:
        return current
    return max(current, candidate)


def positional_facet_ranks(
    path: Path,
) -> dict[CorpusId, dict[str, dict[CorpusId, int]]]:
    """Recompute the per-facet paper ranks that `_fuse_facet_results` assigns.

    Within a facet, the deployed fusion ignores beam rank and instead numbers
    papers by first appearance across the concatenated beam list. Reproducing
    that exactly is what lets `uniform-rrf` validate against the stored eval
    file, and it keeps the emitted `facet_ranks` field meaningful for anything
    downstream that layers graph-RRF on this output.

    Args:
        path: Dump JSONL, read in its original order.

    Returns:
        Nested mapping query -> facet -> paper -> positional rank.
    """
    ranks: dict[CorpusId, dict[str, dict[CorpusId, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in iter_jsonl(path):
        query = CorpusId(int(row["query"]))
        facet: str = row["facet"]
        seen = ranks[query][facet]
        for raw_paper in row["papers"]:
            paper = CorpusId(int(raw_paper))
            if paper not in seen:
                seen[paper] = len(seen)
    return {q: dict(per_facet) for q, per_facet in ranks.items()}


def identifier_weight(
    identifier: Identifier,
    *,
    level: FusionLevel,
    rrf_k: int,
    offsets: dict[str, float],
    n_identifiers: int,
    smoothing: float,
    alpha: float,
) -> float:
    """Weight one identifier's contribution to every paper it covers.

    Args:
        identifier: The de-duplicated identifier.
        level: Fusion level; `uniform-rrf` is handled separately and rejected.
        rrf_k: RRF constant for the `b0` reciprocal-rank weight.
        offsets: Per-view additive offset on the language-model score, the
            paper's device for offsetting length bias between views.
        n_identifiers: Size of the identifier inventory, the denominator of the
            background frequency in `e1`.
        smoothing: Additive smoothing on the occurrence count in `e1`.
        alpha: Exponent on the `e1` weight (1.0 keeps it linear; the released
            code uses 2.0).

    Returns:
        The identifier's weight, clipped at zero for `e1`.

    Raises:
        ValueError: If `level` needs a language-model score the dump lacks.
    """
    if level == "b0":
        return 1.0 / (rrf_k + identifier.rank)

    if identifier.log_prob is None:
        raise ValueError(
            f"level {level!r} needs language-model scores, but identifier "
            f"{identifier.docid!r} has none -- re-decode with score recording"
        )
    biased = identifier.log_prob + offsets.get(identifier.facet, 0.0)

    if level == "m0":
        return biased

    # e1: log-odds of the identifier's (offset) joint log-probability against
    # its background frequency, `exp(biased)` vs `occurrences / n_identifiers`.
    # The offset is required, not optional: a 12-token DocID's raw joint
    # log-probability (mean -13.5 here) sits far below the background of a
    # 1-in-4537 identifier (log -8.4), so without it every weight clips to
    # zero and the ranking degenerates to input order. This is the same
    # magnitude mismatch the released code's own hardcoded `+14` on the
    # pseudo-query view corrects -- ported as a flat additive offset (the
    # smaller deviation) rather than a per-token rescale, since every
    # identifier here is the same length and a per-token score is not itself
    # part of either the paper or the released code.
    capped = min(biased, -1e-9)
    occurrences = max(len(identifier.papers), 1)
    snr = min(math.log((occurrences + smoothing) / (n_identifiers + smoothing)), -1e-9)
    weight = (capped + _log1mexp(snr)) - (snr + _log1mexp(capped))
    return max(weight, 0.0) ** alpha


def _log1mexp(log_p: float) -> float:
    """Compute `log(1 - exp(log_p))` for `log_p < 0` without cancellation."""
    if log_p >= 0.0:
        raise ValueError(f"log1mexp needs a negative argument, got {log_p}")
    if log_p > -math.log(2.0):
        return math.log(-math.expm1(log_p))
    return math.log1p(-math.exp(log_p))


def fuse_query(
    identifiers: list[Identifier],
    *,
    level: FusionLevel,
    rrf_k: int,
    offsets: dict[str, float],
    n_identifiers: int,
    smoothing: float,
    alpha: float,
) -> dict[CorpusId, float]:
    """Accumulate identifier weights into per-paper scores for one query.

    Args:
        identifiers: De-duplicated identifiers decoded for the query.
        level: Fusion level.
        rrf_k: RRF constant for `b0`.
        offsets: Per-view score offsets.
        n_identifiers: Identifier-inventory size for `e1`.
        smoothing: Smoothing for `e1`.
        alpha: Exponent for `e1`.

    Returns:
        Mapping paper -> summed score over the identifiers covering it.
    """
    scores: dict[CorpusId, float] = defaultdict(float)
    for identifier in identifiers:
        weight = identifier_weight(
            identifier,
            level=level,
            rrf_k=rrf_k,
            offsets=offsets,
            n_identifiers=n_identifiers,
            smoothing=smoothing,
            alpha=alpha,
        )
        for paper in identifier.papers:
            scores[paper] += weight
    return dict(scores)


def fuse_uniform_rrf(
    facet_ranks: dict[str, dict[CorpusId, int]], *, rrf_k: int
) -> dict[CorpusId, float]:
    """Reproduce `_fuse_facet_results`: reciprocal rank fusion over facets.

    Args:
        facet_ranks: Positional per-facet paper ranks for one query.
        rrf_k: RRF constant.

    Returns:
        Mapping paper -> summed reciprocal rank.
    """
    scores: dict[CorpusId, float] = defaultdict(float)
    for ranks in facet_ranks.values():
        for paper, rank in ranks.items():
            scores[paper] += 1.0 / (rrf_k + rank + 1)
    return dict(scores)


def count_identifiers(per_query: dict[CorpusId, list[Identifier]]) -> int:
    """Count distinct DocID strings across every query, as the `e1` denominator.

    The FM-index denominator in the released code is the corpus token count;
    the analogue for an enumerated identifier set is the inventory size. Taking
    it from the decodes rather than the trie keeps this script dependent only
    on the dump.

    Args:
        per_query: Loaded dump.

    Returns:
        Number of distinct `(facet, docid)` identifiers observed.
    """
    return len({(i.facet, i.docid) for ids in per_query.values() for i in ids})


def parse_offsets(raw: list[str]) -> dict[str, float]:
    """Parse `--offset facet=value` arguments into a mapping.

    Args:
        raw: Strings of the form `facet=value`.

    Returns:
        Mapping facet -> additive offset.

    Raises:
        ValueError: If an entry is not `facet=value` with a numeric value.
    """
    offsets: dict[str, float] = {}
    for entry in raw:
        facet, _, value = entry.partition("=")
        if not facet or not value:
            raise ValueError(f"expected facet=value, got {entry!r}")
        offsets[facet] = float(value)
    return offsets


def main() -> None:
    """CLI entry point: re-fuse a DocID dump and write an eval-schema JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, help="--record-docids dump JSONL.")
    parser.add_argument(
        "--level",
        choices=["uniform-rrf", "b0", "m0", "e1"],
        default="b0",
        help="Fusion level (default: b0).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL.")
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=f"RRF constant for uniform-rrf and b0 (default: {DEFAULT_RRF_K}).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Results per query (default: {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--offset",
        nargs="+",
        default=[],
        metavar="FACET=VALUE",
        help="Per-view additive score offset for m0/e1 (default: none).",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=DEFAULT_SMOOTHING,
        help=f"Count smoothing for e1 (default: {DEFAULT_SMOOTHING}).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Exponent on the e1 weight (default: 1.0; released code uses 2.0).",
    )
    args = parser.parse_args()

    level: FusionLevel = args.level
    offsets = parse_offsets(args.offset)
    per_query = load_dump(args.dump)
    facet_ranks = positional_facet_ranks(args.dump)
    n_identifiers = count_identifiers(per_query)

    written = 0
    degenerate = 0
    with args.output.open("w") as fh:
        for query, identifiers in per_query.items():
            if level == "uniform-rrf":
                scores = fuse_uniform_rrf(facet_ranks[query], rrf_k=args.rrf_k)
            else:
                scores = fuse_query(
                    identifiers,
                    level=level,
                    rrf_k=args.rrf_k,
                    offsets=offsets,
                    n_identifiers=n_identifiers,
                    smoothing=args.smoothing,
                    alpha=args.alpha,
                )
            if scores and not any(v != 0.0 for v in scores.values()):
                degenerate += 1
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            results: list[dict[str, Any]] = []
            for paper, score in ranked:
                if paper == query:
                    continue
                if len(results) >= args.top_n:
                    break
                provenance = {
                    facet: ranks[paper]
                    for facet, ranks in facet_ranks[query].items()
                    if paper in ranks
                }
                results.append(
                    {
                        "corpusid": int(paper),
                        "score": round(score, 6),
                        "facets": sorted(provenance),
                        "facet_ranks": provenance,
                    }
                )
            fh.write(json.dumps({"corpusid": int(query), "results": results}) + "\n")
            written += 1

    print(f"level={level} queries={written} identifiers={n_identifiers}")
    if degenerate:
        # An all-zero score vector leaves `sorted` to return input order, which
        # scores like a real ranking and silently reports a meaningless number.
        print(
            f"WARNING: {degenerate}/{written} queries scored every candidate "
            f"0.0 -- the ranking is input order, not a ranking. Do not report "
            f"these numbers."
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
