# pyright: basic
"""Merge per-facet GLEN eval outputs into one multi-facet eval file.

The per-facet-adapter experiment trains one checkpoint per facet and
evaluates each with ``--facets-subset <facet>``, producing four eval
JSONLs whose records carry facet_ranks for a single facet. This script
merges them per query — unioning candidates and their facet_ranks — into
one file with the standard multi-facet schema, scoreable by
``score_glen_eval.py`` (whose graph-RRF / hybrid fusion reads
``facet_ranks``). Candidates in the merged output are ordered by uniform
RRF over the merged ranks so the file also behaves sensibly without
graph fusion.

Usage:
    uv run python scripts/eval/merge_facet_evals.py \
        out_merged.jsonl in_problems.jsonl in_methods.jsonl ...
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RRF_K = 60


def main() -> None:
    """Merge per-facet eval JSONLs given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    # query -> candidate -> {facet: rank}
    merged: dict[int, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for path in args.inputs:
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            q = int(rec["corpusid"])
            for r in rec["results"]:
                merged[q][int(r["corpusid"])].update(
                    {f: int(rk) for f, rk in (r.get("facet_ranks") or {}).items()}
                )

    with args.output.open("w") as fh:
        for q, cands in merged.items():
            scored = sorted(
                cands.items(),
                key=lambda kv: -sum(1.0 / (RRF_K + rk) for rk in kv[1].values()),
            )
            fh.write(
                json.dumps(
                    {
                        "corpusid": q,
                        "results": [
                            {
                                "corpusid": c,
                                "score": round(
                                    sum(1.0 / (RRF_K + rk) for rk in ranks.values()),
                                    6,
                                ),
                                "facets": sorted(ranks),
                                "facet_ranks": ranks,
                            }
                            for c, ranks in scored
                        ],
                    }
                )
                + "\n"
            )
    print(f"Merged {len(args.inputs)} files -> {args.output} ({len(merged)} queries)")


if __name__ == "__main__":
    main()
