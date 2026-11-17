"""Build a bibliographic-coupling gold that can measure niche relatedness.

Citation gold is dominated by canonical papers (95.9% of pairs on the
out-of-corpus set), so a query-independent popularity ranking scores R@20 0.459
on it and it cannot measure whether a system surfaces non-obvious work.

This builds an alternative: corpus paper B is gold for query A when A and B
share at least ``--min-shared`` references, with direct citations excluded so
the relation is "sibling" rather than "lineage". The result is ~47% non-canonical
and popularity-immune (a static popularity ranking scores ~0.009 on it).

Usage:
    uv run python scripts/eval/build_bibcoupling_gold.py \
        --queries output/ooc_eval_1k/queries_ooc.txt \
        --corpus-ids data/s2orc-nlp/splits-connected-1k/corpus_ids.txt \
        --references data/s2orc-nlp/citations/references.jsonl \
        --out-dir output/ooc_eval_1k --min-shared 5
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path


def load_references(path: Path) -> dict[int, set[int]]:
    """Load per-paper outgoing reference sets keyed by corpus ID."""
    refs: dict[int, set[int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            refs[int(rec["paper_id"])] = {int(r) for r in rec["references"]}
    return refs


def corpus_in_degree(refs: dict[int, set[int]], corpus: set[int]) -> Counter[int]:
    """Count how often each corpus paper is cited, across all known papers."""
    deg: Counter[int] = Counter()
    for targets in refs.values():
        for t in targets:
            if t in corpus:
                deg[t] += 1
    return deg


def build_gold(
    queries: list[int],
    corpus: set[int],
    refs: dict[int, set[int]],
    min_shared: int,
) -> dict[int, set[int]]:
    """Pair each query with corpus papers sharing >= min_shared references.

    Direct citations (B already referenced by A) are excluded so the gold
    measures a sibling relation rather than the lineage relation that citation
    gold already covers.
    """
    gold: dict[int, set[int]] = {}
    for q in queries:
        rq = refs.get(q)
        if not rq:
            continue
        hits = {
            c
            for c in corpus
            if c not in rq and len(rq & (refs.get(c) or set())) >= min_shared
        }
        if hits:
            gold[q] = hits
    return gold


def write_pairs(path: Path, gold: dict[int, set[int]]) -> int:
    """Write gold as retrieval rows consumable by score_glen_eval.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for a in sorted(gold):
            for b in sorted(gold[a]):
                f.write(
                    json.dumps({"kind": "retrieval", "paper_a": a, "paper_b": b}) + "\n"
                )
                n += 1
    return n


def main() -> None:
    """Build the bibliographic-coupling gold and its niche-only subset."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--corpus-ids", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-shared", type=int, default=5)
    p.add_argument(
        "--report-thresholds",
        type=int,
        nargs="*",
        default=[3, 5, 8],
        help="Also report pool statistics at these thresholds.",
    )
    args = p.parse_args()

    corpus = {int(x) for x in args.corpus_ids.read_text().split()}
    queries = [int(x) for x in args.queries.read_text().split()]
    refs = load_references(args.references)
    indeg = corpus_in_degree(refs, corpus)
    median_indeg = st.median([indeg.get(c, 0) for c in corpus])
    print(
        f"corpus {len(corpus)} | queries {len(queries)} | median in-degree {median_indeg:.0f}"
    )

    for th in args.report_thresholds:
        g = build_gold(queries, corpus, refs, th)
        pairs = sum(len(v) for v in g.values())
        niche = sum(1 for v in g.values() for c in v if indeg.get(c, 0) < median_indeg)
        print(
            f"  >={th}: {len(g):>3} queries | {pairs:>5} pairs |"
            f" mean {pairs / max(len(g), 1):5.2f} | niche {100 * niche / max(pairs, 1):4.1f}%"
        )

    gold = build_gold(queries, corpus, refs, args.min_shared)
    n = write_pairs(args.out_dir / "gold_ooc_bibcoupling_pairs.jsonl", gold)
    niche_gold = {
        q: {c for c in v if indeg.get(c, 0) < median_indeg} for q, v in gold.items()
    }
    niche_gold = {q: v for q, v in niche_gold.items() if v}
    nn = write_pairs(
        args.out_dir / "gold_ooc_bibcoupling_niche_pairs.jsonl", niche_gold
    )
    print(
        f"wrote {n} pairs ({len(gold)} queries) and {nn} niche pairs ({len(niche_gold)} queries)"
    )


if __name__ == "__main__":
    main()
