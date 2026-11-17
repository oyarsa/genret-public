"""Unified per-query scoring for GLEN retrieval outputs.

Consumes the JSONL file produced by ``genret training glen evaluate`` and
prints ``hit@k``, ``recall@k``, and ``precision@k`` against either the
strict training-pairs gold or the GPT-5-mini-labelled gold (or both).

Strict gold is auto-selected per eval file: 1k evals match against
``output/genret/codebook_training_pairs_1k.jsonl.zst``, others against
``output/genret/training_pairs.jsonl.zst``. Override via ``--pairs``.

GPT-mini gold: pairs labelled ``related: true`` by GPT-5-mini across the
three judgement files under ``output/eval_500/``.

Usage:
    uv run python scripts/eval/score_glen_eval.py output/genret/eval/foo.jsonl
    uv run python scripts/eval/score_glen_eval.py one.jsonl two.jsonl  # compare
    uv run python scripts/eval/score_glen_eval.py foo.jsonl --gold gpt
    uv run python scripts/eval/score_glen_eval.py foo.jsonl -k 1 5 10 20 50
    uv run python scripts/eval/score_glen_eval.py foo.jsonl --pairs other.jsonl.zst
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from genret.utils import iter_jsonl

GoldKind = Literal["strict", "gpt", "both"]

DEFAULT_PAIRS_10K = Path("output/genret/training_pairs.jsonl.zst")
DEFAULT_PAIRS_1K = Path("output/genret/codebook_training_pairs_1k.jsonl.zst")
SPLIT_1K_TEST = Path("data/s2orc-nlp/splits-connected-1k/test.txt")


def autodetect_pairs_path(eval_path: Path) -> Path:
    """Pick the right strict-gold pairs file for an eval JSONL.

    Returns the 1k pairs file when every query corpusid in the eval lives
    in the 1k test split; otherwise the 10k pairs file. Falls back to the
    10k default when the 1k split or pairs file is missing.
    """
    if not (SPLIT_1K_TEST.exists() and DEFAULT_PAIRS_1K.exists()):
        return DEFAULT_PAIRS_10K
    ids_1k = {
        int(line.strip())
        for line in SPLIT_1K_TEST.read_text().splitlines()
        if line.strip()
    }
    query_ids: set[int] = set()
    for line in eval_path.read_text().splitlines():
        d = json.loads(line)
        query_ids.add(int(d["corpusid"]))
        if len(query_ids) >= 50:  # enough to disambiguate
            break
    return (
        DEFAULT_PAIRS_1K
        if query_ids and query_ids.issubset(ids_1k)
        else DEFAULT_PAIRS_10K
    )


def load_strict_gold(
    pairs_path: Path = DEFAULT_PAIRS_10K,
) -> dict[int, set[int]]:
    """Build {paper_a: {paper_b, ...}} from training-pairs retrieval rows."""
    gold: dict[int, set[int]] = defaultdict(set)
    for row in iter_jsonl(pairs_path):
        if row.get("kind") == "retrieval":
            gold[int(row["paper_a"])].add(int(row["paper_b"]))
    return dict(gold)


def load_gpt_gold(
    paths: list[Path] | None = None,
) -> dict[int, set[int]]:
    """Build {paper_a: {paper_b, ...}} from GPT-5-mini labels with related=true."""
    if paths is None:
        paths = [
            Path("output/eval_500/gpt5mini_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_new_embedding_judgements.jsonl"),
            Path("output/eval_500/gpt5mini_missing_embedding_judgements.jsonl"),
        ]
    gold: dict[int, set[int]] = defaultdict(set)
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            d = json.loads(line)
            if d.get("related"):
                gold[int(d["paper_a"])].add(int(d["paper_b"]))
    return dict(gold)


DEFAULT_CANDIDATES_1K = Path("output/genret/facet_candidates_1k")
FACETS = ("problems", "methods", "results", "contributions")
RRF_K = 60

WTransform = Callable[[float], float]
"""Maps a per-facet edge weight ``w_f`` to its fused contribution ``g(w_f)``.

Only applied to corroborated edges (``w > 0``); non-edges contribute nothing
regardless of the transform.
"""


def make_w_transform(kind: str, pow_: float = 0.5, shift: float = 0.0) -> WTransform:
    """Build a ``g(w)`` for the graph-RRF weight term.

    - ``identity``: ``g(w) = w`` (current behaviour).
    - ``indicator`` (alias ``uniform``): ``g(w) = 1[w > 0]`` — the corroboration
      *gate only*, discarding the weight magnitude. Each corroborated facet
      contributes ``1/(k+rank_f)``. Isolates the filtering effect from the
      weighting effect.
    - ``sqrt``: ``g(w) = sqrt(w)`` — mild damping of dominant edges.
    - ``pow``: ``g(w) = w ** pow_`` — tunable damping/sharpening.
    - ``log1p``: ``g(w) = log(1 + w)`` — strong damping of high weights.
    - ``linear``: ``g(w) = w + shift`` — floors weak corroborated edges so
      they still contribute (only applied where an edge exists).
    """
    match kind:
        case "identity":
            return lambda w: w
        case "indicator" | "uniform":
            return lambda _w: 1.0
        case "sqrt":
            return lambda w: math.sqrt(w)
        case "pow":
            return lambda w: w**pow_
        case "log1p":
            return lambda w: math.log1p(w)
        case "linear":
            return lambda w: w + shift
        case _:
            msg = f"unknown w-transform: {kind}"
            raise ValueError(msg)


def w_transform_label(kind: str, pow_: float, shift: float) -> str:
    """Short label for a transform spec, used in the fusion mode header."""
    if kind == "pow":
        return f"pow{pow_:g}"
    if kind == "linear":
        return f"lin+{shift:g}"
    return kind


def load_eval(path: Path) -> dict[int, list[int]]:
    """Build {query_paper: [retrieved_paper_ranked_top_first]} from eval JSONL."""
    results: dict[int, list[int]] = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        results[int(d["corpusid"])] = [int(r["corpusid"]) for r in d["results"]]
    return results


def load_facet_weights(
    candidates_dir: Path,
) -> dict[str, dict[int, dict[int, float]]]:
    """Load per-(facet, paper_a, paper_b) -> scores_combined for graph-RRF."""
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
            a = int(cids[i])
            d: dict[int, float] = {}
            for j_pos, j_idx in enumerate(candidates[i]):
                jj = int(j_idx)
                if jj < 0:
                    continue
                d[int(cids[jj])] = float(scores[i, j_pos])
            per_a[a] = d
        out[facet] = per_a
    return out


def load_eval_graph_rrf(
    path: Path,
    facet_weights: dict[str, dict[int, dict[int, float]]],
    k: int = RRF_K,
    transform: WTransform | None = None,
) -> dict[int, list[int]]:
    """Re-fuse per-facet ranks using ``scores_combined`` as the per-facet weight.

    Candidates with no graph edge in any facet get score 0 and sort to the
    bottom — effectively filtering the GLEN beam output to graph-corroborated
    candidates. R@K at small K typically gains +3-4 points; R@K at K = beam-size
    is unchanged (same candidate set, different order).

    ``k`` is the RRF constant; ``transform`` is ``g(w_f)`` applied to each
    corroborated edge weight (default identity).
    """
    from collections import defaultdict  # noqa: PLC0415

    g = transform or (lambda w: w)
    results: dict[int, list[int]] = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        a = int(d["corpusid"])
        cand_ranks: dict[int, dict[str, int]] = defaultdict(dict)
        for r in d["results"]:
            b = int(r["corpusid"])
            cand_ranks[b].update(
                {f: int(rk) for f, rk in (r.get("facet_ranks") or {}).items()}
            )
        scored: list[tuple[int, float]] = []
        for b, ranks in cand_ranks.items():
            s = 0.0
            for facet, rk in ranks.items():
                w = facet_weights.get(facet, {}).get(a, {}).get(b, 0.0)
                if w > 0:
                    s += g(w) / (k + rk)
            scored.append((b, s))
        scored.sort(key=lambda x: -x[1])
        results[a] = [b for b, _ in scored]
    return results


def load_eval_graph_rrf_mutual(
    path: Path,
    facet_weights: dict[str, dict[int, dict[int, float]]],
    k: int = RRF_K,
    transform: WTransform | None = None,
) -> dict[int, list[int]]:
    """Re-fuse per-facet ranks using mutual corroboration among candidates.

    Deployment-clean variant of graph-RRF for out-of-corpus queries: the
    per-facet weight for candidate b is the summed edge strength between
    b and the *other retrieved candidates* (all corpus papers, so their
    adjacency rows always exist), instead of the query->candidate edge
    (which requires the query to be a corpus member). Gold results for a
    query tend to form a mutually-connected cluster; hallucinated beam
    hits land in scattered graph locations and get weak corroboration.

    ``k`` is the RRF constant; ``transform`` is ``g`` applied to the summed
    corroboration weight per facet (default identity).
    """
    from collections import defaultdict  # noqa: PLC0415

    g = transform or (lambda w: w)
    results: dict[int, list[int]] = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        a = int(d["corpusid"])
        cand_ranks: dict[int, dict[str, int]] = defaultdict(dict)
        for r in d["results"]:
            b = int(r["corpusid"])
            cand_ranks[b].update(
                {f: int(rk) for f, rk in (r.get("facet_ranks") or {}).items()}
            )
        cand_set = set(cand_ranks)
        scored: list[tuple[int, float]] = []
        for b, ranks in cand_ranks.items():
            s = 0.0
            for facet, rk in ranks.items():
                per_a = facet_weights.get(facet, {})
                row_b = per_a.get(b, {})
                corr = 0.0
                for c in cand_set:
                    if c == b:
                        continue
                    # Symmetric: take the stronger direction of the edge.
                    w = row_b.get(c, 0.0)
                    w_rev = per_a.get(c, {}).get(b, 0.0)
                    corr += max(w, w_rev)
                if corr > 0:
                    s += g(corr) / (k + rk)
            scored.append((b, s))
        scored.sort(key=lambda x: -x[1])
        results[a] = [b for b, _ in scored]
    return results


def load_eval_graph_rrf_hybrid(
    path: Path,
    facet_weights: dict[str, dict[int, dict[int, float]]],
    k: int = RRF_K,
    transform: WTransform | None = None,
) -> dict[int, list[int]]:
    """Per-query hybrid fusion matching the deployment semantics.

    Query-edge graph-RRF when the query has an adjacency row
    (in-corpus), mutual corroboration otherwise. Corpus members get the stronger
    direct-edge fusion, new papers degrade gracefully to the
    corpus-internal variant instead of to uniform RRF.

    ``k`` and ``transform`` are forwarded to both underlying fusions.
    """
    query_edge = load_eval_graph_rrf(path, facet_weights, k=k, transform=transform)
    mutual = load_eval_graph_rrf_mutual(path, facet_weights, k=k, transform=transform)
    results: dict[int, list[int]] = {}
    for a in query_edge:
        in_corpus = any(a in facet_weights.get(f, {}) for f in FACETS)
        results[a] = query_edge[a] if in_corpus else mutual[a]
    return results


def mmr_rerank(
    results: dict[int, list[int]],
    facet_weights: dict[str, dict[int, dict[int, float]]],
    lambda_: float,
) -> dict[int, list[int]]:
    """Greedy MMR reranking of each query's ranked list.

    At each pick, score(c) = lambda * relevance(c) - (1 - lambda) *
    max_{p in picked} sim(c, p). Relevance is rank-decay (1 / (1 +
    original_rank)). Similarity is the maximum of ``scores_combined(c,
    p)`` and ``scores_combined(p, c)`` across all facets, or 0 if no
    edge exists.

    ``lambda_=1.0`` reproduces the input ordering exactly (no diversity
    penalty). ``lambda_=0`` is pure diversity (ignores relevance).
    """
    reranked: dict[int, list[int]] = {}
    for a, retrieved in results.items():
        n = len(retrieved)
        rel = {p: 1.0 / (1.0 + i) for i, p in enumerate(retrieved)}
        remaining = list(retrieved)
        picked: list[int] = []
        max_sim: dict[int, float] = dict.fromkeys(remaining, 0.0)
        for _ in range(n):
            best_p = remaining[0]
            best_score = -float("inf")
            for c in remaining:
                s = lambda_ * rel[c] - (1.0 - lambda_) * max_sim[c]
                if s > best_score:
                    best_score = s
                    best_p = c
            picked.append(best_p)
            remaining.remove(best_p)
            for c in remaining:
                sim_cp = 0.0
                for weights_by_a in facet_weights.values():
                    sim_cp = max(
                        sim_cp,
                        weights_by_a.get(c, {}).get(best_p, 0.0),
                        weights_by_a.get(best_p, {}).get(c, 0.0),
                    )
                if sim_cp > max_sim[c]:
                    max_sim[c] = sim_cp
        reranked[a] = picked
    return reranked


def expand_with_graph_bfs(
    results: dict[int, list[int]],
    facet_weights: dict[str, dict[int, dict[int, float]]],
    seed_k: int,
    decay: float,
) -> dict[int, list[int]]:
    """Augment each query's ranked list with one BFS hop along graph edges.

    For each query paper ``a``, the top-``seed_k`` retrieved papers are
    used as seeds with a rank-decayed initial score (1 / (1 + rank)).
    Each seed's per-facet candidate list (from ``facet_candidates_*.npz``)
    contributes its candidates as graph neighbours, weighted by the
    ``scores_combined`` edge and a per-hop ``decay`` factor.

    Final ranking: sort by accumulated score; any of the original
    retrieved papers not visited by BFS are appended in their original
    rank order, so recall at high K never drops below the unexpanded
    baseline.
    """
    expanded: dict[int, list[int]] = {}
    for a, retrieved in results.items():
        scores: dict[int, float] = {}
        for rank, s in enumerate(retrieved[:seed_k]):
            scores[s] = scores.get(s, 0.0) + 1.0 / (1.0 + rank)
        additions: dict[int, float] = {}
        for s, s_score in list(scores.items()):
            neighbour_acc: dict[int, float] = {}
            for weights_by_a in facet_weights.values():
                s_neighbours = weights_by_a.get(s, {})
                for n, w in s_neighbours.items():
                    if n == a:
                        continue
                    neighbour_acc[n] = neighbour_acc.get(n, 0.0) + w
            for n, agg_w in neighbour_acc.items():
                additions[n] = additions.get(n, 0.0) + decay * s_score * agg_w
        for n, w in additions.items():
            scores[n] = scores.get(n, 0.0) + w
        original_order = {p: i for i, p in enumerate(retrieved)}
        sorted_papers = sorted(
            scores.keys(),
            key=lambda p: (-scores[p], original_order.get(p, 10**9)),
        )
        seen = set(sorted_papers)
        for p in retrieved:
            if p not in seen:
                sorted_papers.append(p)
        expanded[a] = sorted_papers
    return expanded


def score_at_k(
    results: dict[int, list[int]],
    gold: dict[int, set[int]],
    k: int,
) -> tuple[float, float, float, int]:
    """Return (hit@k, recall@k, precision@k, n_queries_with_gold)."""
    hits, recall_sum, prec_sum, n = 0, 0.0, 0.0, 0
    for qid, retr in results.items():
        g = gold.get(qid, set())
        if not g:
            continue
        n += 1
        inter = sum(1 for p in retr[:k] if p in g)
        if inter > 0:
            hits += 1
        recall_sum += inter / len(g)
        prec_sum += inter / k
    return (
        hits / n if n else 0.0,
        recall_sum / n if n else 0.0,
        prec_sum / n if n else 0.0,
        n,
    )


def print_table(
    label: str,
    results: dict[int, list[int]],
    gold: dict[int, set[int]],
    ks: list[int],
) -> None:
    """Print a per-k table for one eval file under one gold."""
    print(f"\n  {label}")
    print(f"  {'k':<5}{'hit@k':>10}{'recall@k':>12}{'precision@k':>14}")
    print("  " + "-" * 45)
    for k in ks:
        h, r, p, n = score_at_k(results, gold, k)
        print(f"  {k:<5}{h:>10.4f}{r:>12.4f}{p:>14.4f}")
    print(f"  n={n}")


def main() -> None:
    """CLI entry point: score one or more GLEN eval JSONLs at the chosen cutoffs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_files", nargs="+", type=Path)
    parser.add_argument(
        "--gold",
        choices=["strict", "gpt", "both"],
        default="strict",
        help="Gold source (default: strict, training pairs).",
    )
    parser.add_argument(
        "-k",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20, 50],
        help="Cutoffs to report (default: 1 5 10 20 50).",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="Override strict-gold pairs file (default: auto-detect per eval — "
        "1k pairs for 1k evals, 10k pairs otherwise).",
    )
    parser.add_argument(
        "--gpt-labels",
        type=Path,
        nargs="+",
        default=None,
        help="Override GPT-oracle judgement file(s) for --gold gpt "
        "(default: the hardcoded eval_500 files).",
    )
    parser.add_argument(
        "--graph-rrf",
        action="store_true",
        help="Apply graph-weighted RRF re-fusion before scoring. Filters "
        "candidates not in any per-facet top-100 graph neighborhood and "
        "re-orders survivors by sum(scores_combined / (60 + rank_f)). "
        "Adds +3-4 R@20 in our experiments.",
    )
    parser.add_argument(
        "--graph-rrf-mutual",
        action="store_true",
        help="Mutual-corroboration graph-RRF: per-facet weight = summed "
        "edge strength between the candidate and the OTHER retrieved "
        "candidates (corpus-internal only — works for out-of-corpus "
        "queries, unlike --graph-rrf which needs the query's own "
        "adjacency row).",
    )
    parser.add_argument(
        "--graph-rrf-hybrid",
        action="store_true",
        help="Per-query hybrid: query-edge graph-RRF for in-corpus "
        "queries, mutual corroboration for queries with no adjacency "
        "row. The deployment fusion semantics.",
    )
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=DEFAULT_CANDIDATES_1K,
        help="Per-facet candidates dir for graph-RRF / graph-BFS weighting.",
    )
    parser.add_argument(
        "--rrf-k",
        nargs="+",
        type=int,
        default=[RRF_K],
        help="RRF constant(s) for graph-RRF fusion. Pass several to sweep, "
        "e.g. --rrf-k 30 60 90 (one result row per value). Default: 60.",
    )
    parser.add_argument(
        "--w-transform",
        nargs="+",
        choices=["identity", "indicator", "uniform", "sqrt", "pow", "log1p", "linear"],
        default=["identity"],
        help="Transform g(w_f) applied to each corroborated edge weight in "
        "graph-RRF fusion. Pass several to sweep. identity=current; "
        "indicator=gate only g(w)=1[w>0], discards weight (alias: uniform); "
        "sqrt/pow=damping (pow uses --w-pow); log1p=strong damping; "
        "linear=w+--w-shift.",
    )
    parser.add_argument(
        "--w-pow",
        type=float,
        default=0.5,
        help="Exponent for --w-transform pow (g(w)=w**p). Default: 0.5.",
    )
    parser.add_argument(
        "--w-shift",
        type=float,
        default=0.0,
        help="Offset for --w-transform linear (g(w)=w+shift). Default: 0.0.",
    )
    parser.add_argument(
        "--graph-bfs",
        action="store_true",
        help="Apply graph BFS expansion AFTER the chosen fusion mode. For "
        "each query, take the top --bfs-seed-k retrieved as seeds, walk "
        "one graph hop via scores_combined, and reorder. Composable with "
        "--graph-rrf (RRF first → BFS on reordered seeds). Designed to "
        "extend recall at K >= 50 without sacrificing R@20.",
    )
    parser.add_argument(
        "--bfs-seed-k",
        type=int,
        default=20,
        help="Number of top retrieved papers used as BFS seeds.",
    )
    parser.add_argument(
        "--bfs-decay",
        type=float,
        default=0.5,
        help="Per-hop attenuation factor on the BFS edge score.",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=None,
        help="Apply greedy MMR reranking AFTER the chosen fusion mode. "
        "score(c) = lambda * rank-decay-relevance(c) - (1-lambda) * "
        "max similarity to already-picked. Similarity = max "
        "scores_combined edge across facets. lambda=1.0 reproduces "
        "input order; lambda<1 trades relevance for diversity.",
    )
    args = parser.parse_args()

    gold_kind: GoldKind = args.gold
    gpt = load_gpt_gold(args.gpt_labels) if gold_kind in ("gpt", "both") else None
    strict_cache: dict[Path, dict[int, set[int]]] = {}
    facet_weights: dict[str, dict[int, dict[int, float]]] = {}
    graph_modes = [args.graph_rrf, args.graph_rrf_mutual, args.graph_rrf_hybrid]
    if any(graph_modes) or args.graph_bfs or args.mmr_lambda is not None:
        facet_weights = load_facet_weights(args.candidates_dir)
        if not facet_weights:
            raise SystemExit(
                f"--graph-rrf*/--graph-bfs/--mmr-lambda requested but no "
                f"facet_candidates_*.npz under {args.candidates_dir}"
            )
    if sum(graph_modes) > 1:
        raise SystemExit(
            "--graph-rrf, --graph-rrf-mutual and --graph-rrf-hybrid are exclusive"
        )

    graph_on = any(graph_modes)
    # k / w-transform only affect graph-RRF fusion; keep a single trivial
    # config for uniform-RRF so the default path is unchanged.
    rrf_ks = args.rrf_k if graph_on else [RRF_K]
    w_kinds = args.w_transform if graph_on else ["identity"]

    for path in args.eval_files:
        for k in rrf_ks:
            for w_kind in w_kinds:
                transform = make_w_transform(w_kind, args.w_pow, args.w_shift)
                w_label = w_transform_label(w_kind, args.w_pow, args.w_shift)
                grid = f"(k={k},w={w_label})"
                if args.graph_rrf:
                    results = load_eval_graph_rrf(path, facet_weights, k, transform)
                    mode_label = f"graph-RRF{grid}"
                elif args.graph_rrf_mutual:
                    results = load_eval_graph_rrf_mutual(
                        path, facet_weights, k, transform
                    )
                    mode_label = f"graph-RRF-mutual{grid}"
                elif args.graph_rrf_hybrid:
                    results = load_eval_graph_rrf_hybrid(
                        path, facet_weights, k, transform
                    )
                    mode_label = f"graph-RRF-hybrid{grid}"
                else:
                    results = load_eval(path)
                    mode_label = "uniform-RRF"
                if args.graph_bfs:
                    results = expand_with_graph_bfs(
                        results, facet_weights, args.bfs_seed_k, args.bfs_decay
                    )
                    mode_label = (
                        f"{mode_label}+BFS"
                        f"(seed={args.bfs_seed_k},decay={args.bfs_decay})"
                    )
                if args.mmr_lambda is not None:
                    results = mmr_rerank(results, facet_weights, args.mmr_lambda)
                    mode_label = f"{mode_label}+MMR(λ={args.mmr_lambda})"
                pairs_path = args.pairs or autodetect_pairs_path(path)
                print(
                    f"\n=== {path}  (n_eval={len(results)})  gold={pairs_path.name}"
                    f"  fusion={mode_label}"
                )
                if gold_kind in ("strict", "both"):
                    if pairs_path not in strict_cache:
                        strict_cache[pairs_path] = load_strict_gold(pairs_path)
                    print_table(
                        "strict (training pairs)",
                        results,
                        strict_cache[pairs_path],
                        args.k,
                    )
                if gpt is not None:
                    print_table("GPT-mini-labelled", results, gpt, args.k)


if __name__ == "__main__":
    main()
