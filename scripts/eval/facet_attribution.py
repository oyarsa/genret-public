r"""Quantify the facet-level provenance of graph-RRF retrieval.

The graph-RRF score of a candidate decomposes into per-facet terms
``g(w_f) / (k + rank_f)``.  The facet with the largest term is the one the
fusion *credits* with surfacing that candidate -- the provenance claim the
qualitative table illustrates on a single query.  This script turns that
claim into numbers over every test query-result pair.

Three things are measured:

1. **Attribution concentration.**  How decisive is the credited facet?
   Reported as the credited facet's share of the total graph-RRF score and
   its margin over the runner-up facet.  A near-uniform split would mean the
   "why" is not really attributable to one facet.

2. **Cross-signal agreement.**  The dense+graph teacher attributes the same
   pair independently, through per-facet BFS ranks over dense + citation
   edges.  The generative retriever attributes it through per-facet bullet
   decode ranks.  Agreement between two signals that share no inference-time
   machinery is evidence the attribution carries information rather than
   noise.  Chance level is reported alongside.

3. **Facet precision against facet-labelled judgements** (optional).  Pass
   ``--facet-labels`` with a JSONL of ``{paper_a, paper_b, facets: [...]}``
   rows to get true facet-level precision: the share of pairs whose credited
   facet is among the facets a judge says the two papers actually share.

Note on gold: the strict gold (``codebook_training_pairs_1k.jsonl.zst``) is
citation-derived and carries **no** facet provenance -- its ``facet`` field
replicates every pair across all four facets with identical ranks.  Facet
precision therefore cannot be computed against it; ``--facet-labels`` is the
only route to a true facet-level gold.

Usage:
    uv run python scripts/eval/facet_attribution.py \\
        output/genret/eval/glen-graphfallback_per_bullet.jsonl \\
        --teacher output/eval_1k/retrieval_graph_top50.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from genret.utils import iter_jsonl

FACETS = ("problems", "methods", "results", "contributions")
RRF_K = 60

DEFAULT_CANDIDATES = Path("output/genret/facet_candidates_1k")
DEFAULT_PAIRS = Path("output/genret/codebook_training_pairs_1k.jsonl.zst")


def load_gold(pairs_path: Path) -> dict[int, set[int]]:
    """Build ``{query: {gold_neighbour, ...}}`` from the strict pairs file."""
    gold: dict[int, set[int]] = defaultdict(set)
    for row in iter_jsonl(pairs_path):
        if row.get("kind") == "retrieval":
            gold[int(row["paper_a"])].add(int(row["paper_b"]))
    return dict(gold)


def load_facet_weights(
    candidates_dir: Path,
) -> dict[str, dict[int, dict[int, float]]]:
    """Load per-``(facet, paper_a, paper_b)`` ``scores_combined`` edge weights."""
    import numpy as np  # noqa: PLC0415

    out: dict[str, dict[int, dict[int, float]]] = {}
    for facet in FACETS:
        path = candidates_dir / f"facet_candidates_{facet}.npz"
        if not path.exists():
            continue
        f = np.load(path)
        cids = f["corpusids"].astype("int64")
        candidates = f["candidates"].astype("int64")
        scores = f["scores_combined"].astype("float32")
        per_a: dict[int, dict[int, float]] = {}
        for i in range(cids.shape[0]):
            row: dict[int, float] = {}
            for j_pos, j_idx in enumerate(candidates[i]):
                jj = int(j_idx)
                if jj < 0:
                    continue
                row[int(cids[jj])] = float(scores[i, j_pos])
            per_a[int(cids[i])] = row
        out[facet] = per_a
    return out


def attribute(
    eval_path: Path,
    facet_weights: dict[str, dict[int, dict[int, float]]],
    k: int = RRF_K,
) -> dict[int, list[dict[str, Any]]]:
    """Re-run graph-RRF fusion, keeping the per-facet score decomposition.

    Returns ``{query: [candidate_record, ...]}`` in fused rank order, where
    each record carries the per-facet contributions, the credited facet, and
    the credited facet's share of the total score.
    """
    out: dict[int, list[dict[str, Any]]] = {}
    for line in eval_path.read_text().splitlines():
        d = json.loads(line)
        a = int(d["corpusid"])
        cand_ranks: dict[int, dict[str, int]] = defaultdict(dict)
        for r in d["results"]:
            b = int(r["corpusid"])
            cand_ranks[b].update(
                {f: int(rk) for f, rk in (r.get("facet_ranks") or {}).items()}
            )
        records: list[dict[str, Any]] = []
        for b, ranks in cand_ranks.items():
            terms: dict[str, float] = {}
            for facet, rk in ranks.items():
                w = facet_weights.get(facet, {}).get(a, {}).get(b, 0.0)
                if w > 0:
                    terms[facet] = w / (k + rk)
            total = sum(terms.values())
            ordered = sorted(terms.items(), key=lambda x: -x[1])
            credited = ordered[0][0] if ordered else None
            runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
            records.append(
                {
                    "paper_b": b,
                    "score": total,
                    "terms": terms,
                    "decode_ranks": dict(ranks),
                    "credited": credited,
                    "share": (ordered[0][1] / total) if total > 0 else 0.0,
                    "margin": (ordered[0][1] - runner_up) / total if total > 0 else 0.0,
                    "n_facets": len(terms),
                }
            )
        records.sort(key=lambda r: -r["score"])
        out[a] = records
    return out


def load_teacher(path: Path) -> dict[int, dict[int, dict[str, int]]]:
    """Load the dense+graph teacher's per-facet ranks per ``(query, candidate)``."""
    out: dict[int, dict[int, dict[str, int]]] = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        per_b: dict[int, dict[str, int]] = {}
        for r in d["results"]:
            ranks = {
                f: int(rk)
                for f, rk in (r.get("facet_ranks") or {}).items()
                if f in FACETS
            }
            if ranks:
                per_b[int(r["corpusid"])] = ranks
        out[int(d["corpusid"])] = per_b
    return out


def load_facet_labels(path: Path) -> dict[tuple[int, int], tuple[set[str], str | None]]:
    """Load judge-assigned ``(shared facets, primary facet)`` per pair."""
    out: dict[tuple[int, int], tuple[set[str], str | None]] = {}
    for row in iter_jsonl(path):
        key = (int(row["paper_a"]), int(row["paper_b"]))
        facets = {f for f in row.get("facets") or [] if f in FACETS}
        primary = row.get("primary_facet")
        out[key] = (facets, primary if primary in FACETS else None)
    return out


def dump_pairs(
    attributions: dict[int, list[dict[str, Any]]],
    teacher: dict[int, dict[int, dict[str, int]]] | None,
    top_k: int,
    path: Path,
) -> None:
    """Write the ``(paper_a, paper_b)`` pairs that need facet labelling.

    Covers the generative run's attributed top-``k`` pairs and, when a
    teacher file is given, its top-``k`` too, so both systems' provenance can
    be scored against the same labels.
    """
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(a: int, b: int) -> None:
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))

    for a, records in attributions.items():
        for rec in records[:top_k]:
            if rec["credited"] is not None:
                add(a, rec["paper_b"])
    if teacher is not None:
        for a, per_b in teacher.items():
            for b in list(per_b)[:top_k]:
                add(a, b)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for a, b in pairs:
            fh.write(json.dumps({"paper_a": a, "paper_b": b}) + "\n")
    print(f"wrote {len(pairs)} pairs to {path}")


def _facet_sets(
    rec: dict[str, Any],
    depth: int,
    gen_set: str,
) -> set[str]:
    """The generative side's facet set for one candidate at a common depth.

    ``raw`` keeps every facet whose bullet decode emitted the DocID within
    ``depth``; ``gated`` keeps only those contributing a nonzero term to the
    graph-RRF score, i.e. that also carry a graph edge.

    The distinction is a choice about what counts as an explanation, not a
    mechanism: under identity weighting a zero-weight facet contributes zero
    and cannot affect the ranking, so it explains nothing about where the
    candidate landed.  ``terms`` already excludes those.
    """
    ranks: dict[str, int] = rec["decode_ranks"]
    within = {f for f, rk in ranks.items() if rk < depth and f in FACETS}
    if gen_set == "raw":
        return within
    return within & set(rec["terms"])


def set_overlap(
    attributions: dict[int, list[dict[str, Any]]],
    gold: dict[int, set[int]],
    teacher: dict[int, dict[int, dict[str, int]]],
    top_k: int,
    depth: int,
    gen_set: str,
    gold_only: bool,
    permutations: int,
    seed: int,
) -> None:
    """Compare the generative and graph facet sets at a common per-facet depth.

    For every ``(query, candidate)`` pair the graph retriever attributes the
    connection to a set of facets ``F_graph`` (facets whose traversal reached
    the candidate within ``depth``); the generative retriever attributes it to
    ``F_gen``.  Reports how much the two agree, treating ``F_graph`` as the
    reference so that per-facet precision reads as "when the generative
    retriever credits *f*, the graph agrees this often".

    A permutation control re-pairs each ``F_gen`` with another pair's
    ``F_graph``, giving the agreement expected from the two set-size
    distributions alone.
    """
    import random  # noqa: PLC0415

    observed: list[tuple[set[str], set[str]]] = []
    n_total = 0
    n_missing = 0
    n_empty = 0

    for a, records in attributions.items():
        g = gold.get(a, set())
        for rec in records[:top_k]:
            b = rec["paper_b"]
            if rec["credited"] is None:
                continue
            if gold_only and b not in g:
                continue
            n_total += 1
            t_ranks = teacher.get(a, {}).get(b)
            if not t_ranks:
                n_missing += 1
                continue
            f_graph = {f for f, rk in t_ranks.items() if rk < depth}
            f_gen = _facet_sets(rec, depth, gen_set)
            if not f_graph or not f_gen:
                n_empty += 1
                continue
            observed.append((f_gen, f_graph))

    # The reference is always F_graph; `gold_only` selects WHICH pairs are
    # scored (results that are citation-gold, i.e. correct), never what the
    # facet sets are compared against.
    slice_name = (
        "correct results only (strict gold)" if gold_only else "all retrieved pairs"
    )
    print(
        f"\n=== facet-set overlap vs F_graph  depth={depth}  gen-set={gen_set}  "
        f"top-{top_k}, {slice_name}"
    )
    print(f"  generative pairs in slice            {n_total}")
    # NOT "has no graph edge": every graph-RRF survivor has one by
    # construction. These pairs fall outside the graph *retriever's* ranked
    # output, which is a much narrower set than the edge graph.
    print(
        f"  outside the graph retriever's output  {n_missing}"
        f"  ({n_missing / n_total:.1%})"
        if n_total
        else ""
    )
    print(f"  empty facet set on a side (dropped)   {n_empty}")
    print(f"  scored pairs                          {len(observed)}")
    if not observed:
        return

    def micro(pairs: list[tuple[set[str], set[str]]]) -> tuple[float, float, float]:
        inter = sum(len(gset & tset) for gset, tset in pairs)
        gen_n = sum(len(gset) for gset, _ in pairs)
        gra_n = sum(len(tset) for _, tset in pairs)
        p = inter / gen_n if gen_n else 0.0
        r = inter / gra_n if gra_n else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    jaccard = [len(gset & tset) / len(gset | tset) for gset, tset in observed]
    exact = sum(1 for gset, tset in observed if gset == tset) / len(observed)
    prec, rec, f1 = micro(observed)

    rng = random.Random(seed)
    graph_sets = [tset for _, tset in observed]
    chance_j: list[float] = []
    chance_p: list[float] = []
    for _ in range(permutations):
        shuffled = graph_sets[:]
        rng.shuffle(shuffled)
        permuted = [(gset, tset) for (gset, _), tset in zip(observed, shuffled)]
        chance_j.append(
            sum(len(g & t) / len(g | t) for g, t in permuted) / len(permuted)
        )
        chance_p.append(micro(permuted)[0])

    mean_cj = sum(chance_j) / len(chance_j)
    mean_cp = sum(chance_p) / len(chance_p)

    def lift(value: float, chance: float) -> float:
        """Share of the headroom above chance that is actually closed.

        Raw agreement is not comparable across depths: deeper cutoffs grow
        the graph's facet sets, so a random guess lands inside them more
        often.  ``(value - chance) / (1 - chance)`` normalises that away.
        """
        return (value - chance) / (1 - chance) if chance < 1 else 0.0

    print(
        f"\n  mean Jaccard(F_gen, F_graph)          {sum(jaccard) / len(jaccard):.3f}"
    )
    print(f"    permutation chance                  {mean_cj:.3f}")
    print(f"  exact set match                       {exact:.1%}")
    print(f"  micro precision (graph as reference)  {prec:.3f}")
    print(f"    permutation chance                  {mean_cp:.3f}")
    print(f"    lift over chance                    {lift(prec, mean_cp):.3f}")
    print(f"  micro recall                          {rec:.3f}")
    print(f"  micro F1                              {f1:.3f}")

    print("\n  Per-facet (graph set as reference)")
    print(f"    {'facet':<16}{'P':>8}{'R':>8}{'F1':>8}{'credited':>10}{'in graph':>10}")
    for facet in FACETS:
        tp = sum(1 for gset, tset in observed if facet in gset and facet in tset)
        gen_n = sum(1 for gset, _ in observed if facet in gset)
        gra_n = sum(1 for _, tset in observed if facet in tset)
        p = tp / gen_n if gen_n else 0.0
        r = tp / gra_n if gra_n else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"    {facet:<16}{p:>8.3f}{r:>8.3f}{f:>8.3f}{gen_n:>10}{gra_n:>10}")


def summarise(
    attributions: dict[int, list[dict[str, Any]]],
    gold: dict[int, set[int]],
    teacher: dict[int, dict[int, dict[str, int]]] | None,
    facet_labels: dict[tuple[int, int], tuple[set[str], str | None]] | None,
    top_k: int,
    gold_only: bool,
) -> None:
    """Print the attribution report for one slice of the retrieved pairs."""
    credited_counts: Counter[str] = Counter()
    shares: list[float] = []
    margins: list[float] = []
    n_facets: Counter[int] = Counter()
    teacher_hits = teacher_tot = 0
    teacher_top2_hits = 0
    teacher_chance = 0.0
    teacher_top2_chance = 0.0
    dec_hits = dec_tot = 0
    label_hits = label_tot = 0
    label_primary_hits = label_primary_tot = 0
    label_chance = 0.0
    label_unrelated = 0
    n_pairs = 0

    for a, records in attributions.items():
        g = gold.get(a, set())
        for rec in records[:top_k]:
            b = rec["paper_b"]
            if gold_only and b not in g:
                continue
            if rec["credited"] is None:
                continue
            n_pairs += 1
            credited_counts[rec["credited"]] += 1
            shares.append(rec["share"])
            margins.append(rec["margin"])
            n_facets[rec["n_facets"]] += 1

            if teacher is not None:
                t_ranks = teacher.get(a, {}).get(b)
                if t_ranks:
                    teacher_tot += 1
                    ordered_t = sorted(t_ranks, key=lambda f: t_ranks[f])
                    hit = rec["credited"] == ordered_t[0]
                    teacher_hits += hit
                    teacher_top2_hits += rec["credited"] in ordered_t[:2]
                    teacher_chance += 1.0 / len(t_ranks)
                    teacher_top2_chance += min(2, len(t_ranks)) / len(t_ranks)
                    # Attribution should be more trustworthy where fusion is
                    # decisive rather than a near-tie between facets.
                    if rec["share"] >= 0.5:
                        dec_tot += 1
                        dec_hits += hit

            if facet_labels is not None:
                labelled = facet_labels.get((a, b))
                if labelled is not None:
                    labs, primary = labelled
                    if not labs:
                        # Judge found no substantive link at all: the pair has
                        # no facet to be right about.
                        label_unrelated += 1
                        continue
                    label_tot += 1
                    label_hits += rec["credited"] in labs
                    label_chance += len(labs) / len(FACETS)
                    if primary is not None:
                        label_primary_tot += 1
                        label_primary_hits += rec["credited"] == primary

    slice_name = (
        "correct results only (strict gold)" if gold_only else "all retrieved pairs"
    )
    print(f"\n=== top-{top_k}, {slice_name}  (n = {n_pairs})")
    if not n_pairs:
        return

    print("\n  Credited-facet distribution")
    for facet, count in credited_counts.most_common():
        print(f"    {facet:<16}{count:>7}  {count / n_pairs:>7.1%}")

    print("\n  Attribution concentration")
    print(f"    mean top-facet share of score      {sum(shares) / len(shares):.3f}")
    print(f"    mean margin over runner-up         {sum(margins) / len(margins):.3f}")
    decisive = sum(1 for s in shares if s >= 0.5) / len(shares)
    print(f"    share >= 0.50 (single facet owns)  {decisive:.1%}")

    print("\n  Facets with a corroborating edge (per pair)")
    for n in sorted(n_facets):
        print(f"    {n} facet(s)     {n_facets[n]:>7}  {n_facets[n] / n_pairs:>7.1%}")

    if teacher is not None and teacher_tot:
        print(f"\n  Agreement with dense+graph teacher  (n = {teacher_tot})")
        print(
            f"    credited == teacher's best facet   {teacher_hits / teacher_tot:.1%}"
        )
        print(
            f"      chance level                     {teacher_chance / teacher_tot:.1%}"
        )
        print(
            f"    credited in teacher's top-2 facets {teacher_top2_hits / teacher_tot:.1%}"
        )
        print(
            f"      chance level                     "
            f"{teacher_top2_chance / teacher_tot:.1%}"
        )
        if dec_tot:
            print(
                f"    best-facet agreement | decisive    {dec_hits / dec_tot:.1%}"
                f"  (n = {dec_tot})"
            )

    if facet_labels is not None and label_tot:
        print(f"\n  Facet precision vs judged facets  (n = {label_tot})")
        print(f"    credited facet is a shared facet   {label_hits / label_tot:.1%}")
        print(f"      chance level                     {label_chance / label_tot:.1%}")
        if label_primary_tot:
            print(
                f"    credited == judge's primary facet  "
                f"{label_primary_hits / label_primary_tot:.1%}"
                f"  (n = {label_primary_tot})"
            )
        print(f"    pairs judged unrelated (excluded)  {label_unrelated}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_file", type=Path)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=None,
        help="Dense+graph retrieval JSONL with per-facet ranks, for "
        "cross-signal attribution agreement.",
    )
    parser.add_argument(
        "--facet-labels",
        type=Path,
        default=None,
        help="JSONL of {paper_a, paper_b, facets: [...]} judgements for "
        "true facet-level precision.",
    )
    parser.add_argument(
        "--dump-pairs",
        type=Path,
        default=None,
        help="Write the pairs that would need facet labels to this JSONL and "
        "exit. No facet-labelled gold exists; this only enumerates the pairs "
        "one would cover.",
    )
    parser.add_argument("-k", "--top-k", nargs="+", type=int, default=[20])
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[20, 50, 100],
        help="Common per-facet depths defining each side's facet set. The "
        "headline depth is 50, matching the item-12 convention; the others "
        "show the metric is not a cutoff artefact.",
    )
    parser.add_argument(
        "--gen-set",
        nargs="+",
        choices=["gated", "raw"],
        default=["gated", "raw"],
        help="Generative facet set: 'gated' is what the deployed graph-RRF "
        "score is built from; 'raw' is the bullet-decode set before graph "
        "gating.",
    )
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sets-only",
        action="store_true",
        help="Print only the facet-set overlap report.",
    )
    args = parser.parse_args()

    facet_weights = load_facet_weights(args.candidates_dir)
    if not facet_weights:
        raise SystemExit(f"no facet_candidates_*.npz under {args.candidates_dir}")
    gold = load_gold(args.pairs)
    attributions = attribute(args.eval_file, facet_weights, args.rrf_k)
    teacher = load_teacher(args.teacher) if args.teacher else None

    if args.dump_pairs is not None:
        dump_pairs(attributions, teacher, max(args.top_k), args.dump_pairs)
        return

    labels = load_facet_labels(args.facet_labels) if args.facet_labels else None

    print(f"eval: {args.eval_file}  queries: {len(attributions)}")
    for top_k in args.top_k:
        for gold_only in (False, True):
            if not args.sets_only:
                summarise(attributions, gold, teacher, labels, top_k, gold_only)
            if teacher is None:
                continue
            for gen_set in args.gen_set:
                for depth in args.depths:
                    set_overlap(
                        attributions,
                        gold,
                        teacher,
                        top_k,
                        depth,
                        gen_set,
                        gold_only,
                        args.permutations,
                        args.seed,
                    )


if __name__ == "__main__":
    main()
