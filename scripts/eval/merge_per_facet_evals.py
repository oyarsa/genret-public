"""RRF-fuse multiple per-facet eval JSONLs into a single ranked-list JSONL.

Each input file must use the standard score_glen schema:
    {"corpusid": <query>, "results": [{"corpusid": ..., "score": ..., ...}, ...]}

Output schema matches and can be passed to score_glen_eval.py /
score_glen_post_expansion.py.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from genret.utils import iter_jsonl, open_auto

RRF_K = 60


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+", help="Per-facet eval JSONL files.")
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("--rrf-k", type=int, default=RRF_K)
    args = p.parse_args()

    # qid -> paper_id -> rrf score
    fused: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for path in args.inputs:
        for row in iter_jsonl(path):
            qid = int(row["corpusid"])
            for rank, r in enumerate(row.get("results", [])):
                pid = int(r["corpusid"])
                fused[qid][pid] += 1.0 / (args.rrf_k + rank + 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open_auto(args.output, "w") as fh:
        for qid, scores in fused.items():
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])[: args.top_n]
            rec = {
                "corpusid": qid,
                "results": [
                    {
                        "corpusid": pid,
                        "score": round(s, 6),
                        "facets": ["rrf"],
                        "facet_ranks": {"rrf": rank},
                    }
                    for rank, (pid, s) in enumerate(ranked)
                ],
            }
            fh.write(json.dumps(rec) + "\n")
            n += 1
    print(f"Fused {len(args.inputs)} files -> {args.output} ({n} queries)")


if __name__ == "__main__":
    main()
