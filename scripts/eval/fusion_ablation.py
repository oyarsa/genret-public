"""Fusion-ablation table for the 1k vmhk/s39g generative retriever.

Holds the model + per-facet generations fixed; varies only how the four
per-facet ranked lists are combined. Strict 1k gold (matches the leaderboard
sanity checks: direct 0.284 / graph-RRF 0.322 / R@50 0.447 / BFS 0.271).

Reuses the fusion + scoring code in ``scripts/eval/score_glen_eval.py``. Run
from the repo root (the import path and the cached-artefact paths are
repo-root relative):

    uv run python scripts/eval/fusion_ablation.py

Companion write-up: ``docs/reference/20260629-generative-fusion-ablation.md``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("sge", "scripts/eval/score_glen_eval.py")
assert spec is not None
assert spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

EVAL = Path("output/genret/eval/glen-bullets-1k-1b-78lp-optionB_per_bullet.jsonl")
CAND = Path("output/genret/facet_candidates_1k")
K = 60
FACETS = ("problems", "methods", "results", "contributions")

fw = m.load_facet_weights(CAND)
gold = m.load_strict_gold(m.DEFAULT_PAIRS_1K)


def r2050(results: dict[int, list[int]]) -> tuple[float, float]:
    """Return (recall@20, recall@50) under strict gold."""
    return (
        m.score_at_k(results, gold, 20)[1],
        m.score_at_k(results, gold, 50)[1],
    )


def single_facet_graph(facet: str) -> dict[int, list[int]]:
    """Row 1 component: rank candidates by w_f/(k+rank_f), keep w_f>0, one facet.

    Computed on the stored s39g candidate pool (same universe as the graph-RRF
    rows), using the stored per-facet ranks. NOTE: this is the contribution of
    a single facet *within the fused candidate pool*, not a standalone
    single-facet retriever (the on-disk decoded dump is a different decode and
    does not reconcile with s39g, so it was not used).
    """
    res: dict[int, list[int]] = {}
    for line in EVAL.read_text().splitlines():
        d = json.loads(line)
        a = int(d["corpusid"])
        scored: list[tuple[int, float]] = []
        for r in d["results"]:
            b = int(r["corpusid"])
            fr = r.get("facet_ranks") or {}
            if facet not in fr:
                continue
            w = fw.get(facet, {}).get(a, {}).get(b, 0.0)
            if w > 0:
                scored.append((b, w / (K + int(fr[facet]))))
        scored.sort(key=lambda x: -x[1])
        res[a] = [b for b, _ in scored]
    return res


def graph_rrf_subset(facets: tuple[str, ...]) -> dict[int, list[int]]:
    """Graph-RRF over the stored eval, restricted to `facets` (leave-one-out).

    Same candidate universe and weight convention as the full graph-RRF row
    (load_eval_graph_rrf), just summing over a facet subset.
    """
    res: dict[int, list[int]] = {}
    for line in EVAL.read_text().splitlines():
        d = json.loads(line)
        a = int(d["corpusid"])
        scored: list[tuple[int, float]] = []
        for r in d["results"]:
            b = int(r["corpusid"])
            fr = r.get("facet_ranks") or {}
            s = 0.0
            for facet in facets:
                if facet not in fr:
                    continue
                w = fw.get(facet, {}).get(a, {}).get(b, 0.0)
                if w > 0:
                    s += w / (K + int(fr[facet]))
            scored.append((b, s))
        scored.sort(key=lambda x: -x[1])
        res[a] = [b for b, _ in scored]
    return res


# --- Row 2: uniform RRF == stored direct (the stored fusion IS uniform RRF) -
uniform = m.load_eval(EVAL)

# --- Row 1: best single facet -------------------------------------------
sf = {f: r2050(single_facet_graph(f)) for f in FACETS}
best_facet = max(sf, key=lambda f: sf[f][0])
print("\nSingle-facet graph-weighted recall:")
for f in FACETS:
    print(f"  {f:14s} R@20={sf[f][0]:.4f}  R@50={sf[f][1]:.4f}")

# --- Rows 3-7: full + leave-one-out -------------------------------------
full = m.load_eval_graph_rrf(EVAL, fw, K, m.make_w_transform("identity"))
loo = {f: graph_rrf_subset(tuple(x for x in FACETS if x != f)) for f in FACETS}

# --- Row 8: graph-RRF + BFS ---------------------------------------------
bfs = m.expand_with_graph_bfs(full, fw, seed_k=20, decay=0.5)

# --- Final table --------------------------------------------------------
print("\n\n## Fusion ablation (1k vmhk/s39g, 158q strict gold)\n")
print("| # | Variant | R@20 | R@50 | source |")
print("|---|---|---|---|---|")
b20, b50 = sf[best_facet]
print(
    f"| 1 | Best single facet (**{best_facet}**) | {b20:.4f} | {b50:.4f} | regenerated (stored pool) |"
)
u20, u50 = r2050(uniform)
print(
    f"| 2 | Uniform RRF (4 facets, no graph) | {u20:.4f} | {u50:.4f} | cached = direct baseline |"
)
f20, f50 = r2050(full)
print(
    f"| 3 | Graph-RRF (4 facets, full) | {f20:.4f} | {f50:.4f} | regenerated (matches leaderboard) |"
)
for i, f in enumerate(FACETS, start=4):
    l20, l50 = r2050(loo[f])
    print(f"| {i} | Graph-RRF - {f} | {l20:.4f} | {l50:.4f} | regenerated |")
x20, x50 = r2050(bfs)
print(
    f"| 8 | Graph-RRF + 2-hop BFS (seed=20, decay=0.5) | {x20:.4f} | {x50:.4f} | regenerated (matches leaderboard) |"
)
print(f"\nBest single facet: **{best_facet}** (R@20={b20:.4f}).")
