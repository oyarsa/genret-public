"""Discovery analysis for out-of-corpus retrieval: quadrants, ceilings, examples.

Answers "what kind of papers does one system return that the other does not?"
by classifying each gold (query, paper) pair on two axes:

- obviousness  -- MiniLM mean-pooled cosine between query and paper, i.e. what a
  dense retriever can see. Split at the median gold cosine.
- canonicality -- the paper's citation in-degree within the corpus. Split at the
  median corpus in-degree.

Also reports the reachability ceiling (what fraction of gold is inside each
system's reachable output space at all), which distinguishes a ranking failure
from a generation failure, and a non-obviousness-weighted recall (NOR) that
rewards recovering gold the embedding space places far away.

NOR must always be read against the popularity-only control, since a system that
returns canonical papers scores well on it by default.

Usage:
    uv run python scripts/eval/ooc_discovery_analysis.py \
        --gold output/ooc_eval_1k/gold_ooc_cite_pairs.jsonl \
        --generative output/genret/eval/ooc-graphfallback_per_bullet.jsonl \
        --dense output/ooc_eval_1k/ret_denseonly_ooc.jsonl \
        --corpus-emb output/eval_1k/facet_embeddings_minilm_1k.npz \
        --query-emb output/ooc_eval_1k/emb_ooc_404.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path

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


def mean_pooled(mat: Vec) -> Vec:
    """L2-normalise items, mean-pool them, and re-normalise."""
    m = np.asarray(mat, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    v = (m / n).mean(0)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def load_ranked(path: Path) -> dict[int, list[int]]:
    """Load a retrieval JSONL as {query: [corpusid, ...]} in file order."""
    out: dict[int, list[int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[int(r["corpusid"])] = [
                int(x["corpusid"]) if isinstance(x, dict) else int(x)
                for x in r["results"]
            ]
    return out


def in_degree(references: Path, corpus: set[int]) -> Counter[int]:
    """Citation in-degree of each corpus paper over all known reference lists."""
    deg: Counter[int] = Counter()
    with open(references, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            for t in json.loads(line)["references"]:
                if int(t) in corpus:
                    deg[int(t)] += 1
    return deg


def boot_ci(vals: list[float], n: int = 10000) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    bs = sorted(sum(random.choices(vals, k=len(vals))) / len(vals) for _ in range(n))
    return bs[int(0.025 * n)], bs[int(0.975 * n)]


def main() -> None:
    """Run the quadrant, ceiling and NOR analyses."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--generative", type=Path, required=True)
    p.add_argument("--dense", type=Path, required=True)
    p.add_argument("--corpus-emb", type=Path, required=True)
    p.add_argument("--query-emb", type=Path, required=True)
    p.add_argument(
        "--candidates-dir", type=Path, default=Path("output/genret/facet_candidates_1k")
    )
    p.add_argument(
        "--corpus-ids",
        type=Path,
        default=Path("data/s2orc-nlp/splits-connected-1k/corpus_ids.txt"),
    )
    p.add_argument(
        "--references",
        type=Path,
        default=Path("data/s2orc-nlp/citations/references.jsonl"),
    )
    p.add_argument("--titles", type=Path, default=None)
    p.add_argument("-k", type=int, default=20)
    p.add_argument("--examples", type=int, default=4)
    args = p.parse_args()
    random.seed(0)

    k = args.k
    corpus = {int(x) for x in args.corpus_ids.read_text().split()}
    gold = sge.load_strict_gold(args.gold)
    gen = sge.load_eval_graph_rrf_mutual(
        args.generative, sge.load_facet_weights(args.candidates_dir)
    )
    den = load_ranked(args.dense)
    indeg = in_degree(args.references, corpus)

    craw = np.load(args.corpus_emb, allow_pickle=True)
    qraw = np.load(args.query_emb, allow_pickle=True)
    cvec = {int(key): mean_pooled(craw[key]) for key in craw.files}
    qvec = {int(key): mean_pooled(qraw[key]) for key in qraw.files}

    qs = [q for q in gold if gold[q] and q in gen and q in den and q in qvec]
    sim = {(q, c): float(qvec[q] @ cvec[c]) for q in qs for c in gold[q] if c in cvec}
    med_sim = st.median(list(sim.values()))
    med_ind = st.median([indeg.get(c, 0) for c in corpus])
    print(f"queries {len(qs)} | gold pairs {len(sim)}")
    print(f"thresholds: cosine {med_sim:.3f} | in-degree {med_ind:.0f}\n")

    print(f"== quadrants: recall@{k} of gold ==")
    cells: dict[str, list[int]] = {}
    for q in qs:
        g20, d20 = set(gen[q][:k]), set(den[q][:k])
        for c in gold[q]:
            if (q, c) not in sim:
                continue
            lab = (
                ("obvious" if sim[(q, c)] >= med_sim else "non-obvious")
                + "/"
                + ("canonical" if indeg.get(c, 0) >= med_ind else "niche")
            )
            cells.setdefault(lab, [0, 0, 0])
            cells[lab][0] += 1
            cells[lab][1] += c in g20
            cells[lab][2] += c in d20
    print(f"  {'quadrant':<24}{'n':>7}{'generative':>13}{'dense+graph':>13}")
    for lab in (
        "obvious/canonical",
        "obvious/niche",
        "non-obvious/canonical",
        "non-obvious/niche",
    ):
        if lab in cells:
            n, g, d = cells[lab]
            print(f"  {lab:<24}{n:>7}{g / n:>13.3f}{d / n:>13.3f}")

    print("\n== reachability ceiling (fraction of gold inside the output space) ==")
    for nm, ranked in (("generative", gen), ("dense+graph", den)):
        reach = {c for q in ranked for c in ranked[q]}
        ceil = st.mean([len(gold[q] & reach) / len(gold[q]) for q in qs])
        med = st.median([indeg.get(c, 0) for c in reach]) if reach else 0
        print(
            f"  {nm:<14} reachable {len(reach):>4}/{len(corpus)} |"
            f" median in-degree {med:>6.0f} | ceiling {ceil:.3f}"
        )

    print("\n== non-obviousness-weighted recall (NOR) ==")
    lo, hi = min(sim.values()), max(sim.values())
    w = {kk: (hi - v) / (hi - lo) for kk, v in sim.items()}
    pop = [c for c, _ in indeg.most_common()]
    rows: dict[str, list[float]] = {}
    for nm, ranked in (
        ("generative", gen),
        ("dense+graph", den),
        ("popularity-only", None),
    ):
        vals: list[float] = []
        for q in qs:
            tot = sum(w[(q, c)] for c in gold[q] if (q, c) in w)
            if tot <= 0:
                continue
            top = set(pop[:k]) if ranked is None else set(ranked[q][:k])
            vals.append(sum(w[(q, c)] for c in gold[q] & top if (q, c) in w) / tot)
        rows[nm] = vals
        print(f"  {nm:<16} NOR@{k} {st.mean(vals):.4f}")
    d = [a - b for a, b in zip(rows["generative"], rows["dense+graph"])]
    ci = boot_ci(d)
    print(
        f"  delta (gen - dense) {st.mean(d):+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]"
        f" {'SIG' if ci[0] > 0 or ci[1] < 0 else 'ns'}"
    )

    if args.titles and args.titles.exists():
        from genret.utils import iter_jsonl  # noqa: PLC0415

        titles = {
            int(r["corpusid"]): (r.get("title") or "").strip()
            for r in iter_jsonl(args.titles)
        }
        print("\n== lineage examples: generative-only, non-obvious, canonical ==")
        cand = [
            (sim[(q, c)], indeg.get(c, 0), q, c)
            for q in qs
            for c in (set(gen[q][:k]) & gold[q]) - set(den[q][:k])
            if (q, c) in sim and sim[(q, c)] < med_sim and indeg.get(c, 0) >= med_ind
        ]
        cand.sort(key=lambda x: -x[1])
        for s, ind, q, c in cand[: args.examples]:
            print(f"  cos {s:.3f} in-degree {ind}")
            print(f"     Q: {titles.get(q, '?')[:88]}")
            print(f"     A: {titles.get(c, '?')[:88]}")


if __name__ == "__main__":
    main()
