"""Split retrievals by whether facet similarity would have surfaced them.

For each query, all corpus papers are ranked by cosine over mean-pooled facet
embeddings -- the representation the dense retriever indexes. A paper is
``obvious`` when it is among the query's ``--rank`` nearest and ``non-obvious``
otherwise. The cut is per query, so it adapts to queries whose neighbourhoods sit
uniformly closer or further away, and it applies to the (query, paper) pair
rather than to the paper.

Reports, for each system: how its results are allocated across the two regions,
its precision within each against that region's base rate, and -- with
``--judgements`` -- the share of its non-gold retrievals an LLM judge accepted.
With ``--examples`` it also surfaces one example per facet, preferring shallow
ranks just outside the cut where the connection is easiest to read.

Usage:
    uv run python scripts/eval/ooc_obviousness_analysis.py \
        --gold output/ooc_eval_1k/gold_ooc_cite_pairs.jsonl \
        --generative output/genret/eval/ooc-graphfallback_per_bullet.jsonl \
        --dense output/ooc_eval_1k/ret_denseonly_ooc.jsonl \
        --corpus-emb output/eval_1k/facet_embeddings_minilm_1k.npz \
        --query-emb output/ooc_eval_1k/emb_ooc_404.npz \
        --judgements output/ooc_eval_1k/judgements_nongold_beyond.jsonl \
        --pairs output/ooc_eval_1k/judge_pairs_nongold_beyond.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# score_glen_eval.py is a sibling script, not an installed module.
_spec = importlib.util.spec_from_file_location(
    "sge", str(Path(__file__).parent / "score_glen_eval.py")
)
assert _spec is not None
assert _spec.loader is not None
sge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sge)

Vec = npt.NDArray[np.float32]
FACETS = ("problems", "methods", "results", "contributions")


def mean_pooled(mat: Vec) -> Vec:
    """L2-normalise items, mean-pool them, and re-normalise."""
    m = np.asarray(mat, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    v = (m / n).mean(0)
    return v / max(float(np.linalg.norm(v)), 1e-9)


Entry = dict[str, Any] | int
"""A result row, which may be a full dict or a bare corpus id."""


def load_results(path: Path) -> dict[int, list[Entry]]:
    """Load a retrieval JSONL, keeping the full result dicts."""
    out: dict[int, list[Entry]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[int(r["corpusid"])] = r["results"]
    return out


def cid(entry: Entry) -> int:
    """Corpus ID of a result entry, which may be a dict or a bare id."""
    return int(entry["corpusid"]) if isinstance(entry, dict) else int(entry)


def main() -> None:
    """Report allocation, precision and per-facet examples by region."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--generative", type=Path, required=True)
    p.add_argument("--dense", type=Path, required=True)
    p.add_argument("--corpus-emb", type=Path, required=True)
    p.add_argument("--query-emb", type=Path, required=True)
    p.add_argument("--judgements", type=Path, default=None)
    p.add_argument("--pairs", type=Path, default=None)
    p.add_argument("--titles", type=Path, default=None)
    p.add_argument(
        "--candidates-dir", type=Path, default=Path("output/genret/facet_candidates_1k")
    )
    p.add_argument("--rank", type=int, default=50, help="Obvious/non-obvious cut.")
    p.add_argument("-k", type=int, default=20)
    p.add_argument("--examples", action="store_true")
    args = p.parse_args()

    k, cut = args.k, args.rank
    gold = sge.load_strict_gold(args.gold)
    gen_raw = load_results(args.generative)
    gen = sge.load_eval_graph_rrf_mutual(
        args.generative, sge.load_facet_weights(args.candidates_dir)
    )
    den = {q: [cid(x) for x in v] for q, v in load_results(args.dense).items()}

    craw = np.load(args.corpus_emb, allow_pickle=True)
    qraw = np.load(args.query_emb, allow_pickle=True)
    cvec = {int(key): mean_pooled(craw[key]) for key in craw.files}
    qvec = {int(key): mean_pooled(qraw[key]) for key in qraw.files}
    cids = sorted(cvec)
    mat = np.stack([cvec[c] for c in cids])
    pos = {c: i for i, c in enumerate(cids)}

    qs = [q for q in gold if gold[q] and q in gen and q in den and q in qvec]
    rank: dict[tuple[int, int], int] = {}
    for q in qs:
        order = np.argsort(-(mat @ qvec[q]))
        rk = np.empty(len(cids), dtype=int)
        rk[order] = np.arange(1, len(cids) + 1)
        for c in cids:
            rank[(q, c)] = int(rk[pos[c]])

    print(f"queries {len(qs)} | obvious = facet-similarity rank <= {cut}\n")

    print(f"== allocation of top-{k} results ==")
    for nm, res in (("generative", gen), ("dense+graph", den)):
        tot = sum(1 for q in qs for c in res[q][:k] if (q, c) in rank)
        far = sum(1 for q in qs for c in res[q][:k] if rank.get((q, c), 0) > cut)
        print(
            f"  {nm:<14} obvious {100 * (tot - far) / tot:5.1f}%  non-obvious {100 * far / tot:5.1f}%"
        )

    print("\n== precision within each region (citation gold) ==")
    for nm, res in (("generative", gen), ("dense+graph", den)):
        for lab, keep in (("obvious", False), ("non-obvious", True)):
            picks = [
                (q, c)
                for q in qs
                for c in res[q][:k]
                if (q, c) in rank and (rank[(q, c)] > cut) == keep
            ]
            hit = sum(1 for q, c in picks if c in gold[q])
            region = [(q, c) for q in qs for c in cids if (rank[(q, c)] > cut) == keep]
            base_hit = sum(1 for q, c in region if c in gold[q])
            prec, base = hit / len(picks), base_hit / len(region)
            print(
                f"  {nm:<14} {lab:<12} precision {prec:.4f}  base {base:.4f}"
                f"  lift {prec / base:5.1f}x"
            )

    if args.judgements and args.pairs:
        meta: dict[tuple[int, int], dict[str, Any]] = {}
        with open(args.pairs, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    meta[(d["paper_a"], d["paper_b"])] = d
        acc: dict[str, list[bool]] = defaultdict(list)
        with open(args.judgements, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                m = meta.get((d["paper_a"], d["paper_b"]))
                if m:
                    acc[m["system"]].append(bool(d["related"]))
        print("\n== LLM-judged relatedness of non-gold, non-obvious retrievals ==")
        for s, lab in (("gen", "generative"), ("den", "dense+graph")):
            v = acc.get(s, [])
            if v:
                print(f"  {lab:<14} {100 * st.mean(v):5.1f}% related  (n={len(v)})")

    if args.examples and args.titles and args.titles.exists():
        from genret.utils import iter_jsonl  # noqa: PLC0415

        titles = {
            int(r["corpusid"]): (r.get("title") or "").strip()
            for r in iter_jsonl(args.titles)
        }
        judged: set[tuple[int, int]] = set()
        if args.judgements:
            with open(args.judgements, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        if d.get("related"):
                            judged.add((d["paper_a"], d["paper_b"]))
        best: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
        for q in qs:
            for entry in gen_raw.get(q, [])[:k]:
                c = cid(entry)
                r = rank.get((q, c), 0)
                if not (cut < r <= 2 * cut) or (q, c) not in judged:
                    continue
                if not isinstance(entry, dict):
                    continue
                raw = entry.get("facet_ranks") or {}
                fr: dict[str, int] = {str(a): int(b) for a, b in dict(raw).items()}
                if fr:
                    f_best = min(fr.items(), key=lambda x: x[1])
                    best[f_best[0]].append((f_best[1], r, q, c))
        print(f"\n== example per facet, rank {cut + 1}-{2 * cut}, judged related ==")
        for facet in FACETS:
            for facet_rank, r, q, c in sorted(best.get(facet, []))[:1]:
                print(f"  [{facet}] facet-rank {facet_rank}, sim rank {r}")
                print(f"     Q: {titles.get(q, '?')[:76]}")
                print(f"     A: {titles.get(c, '?')[:76]}")


if __name__ == "__main__":
    main()
