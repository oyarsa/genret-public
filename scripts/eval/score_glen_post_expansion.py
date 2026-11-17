"""Score a GLEN eval JSONL under post-expansion recall.

Treats GLEN's top-K predictions as seed papers, runs 2-hop BFS in each of
the 4 facet edge graphs built on-the-fly from the 1k corpus facet vectors,
and scores the union (RRF-fused) against the strict-gold training pairs.

The key shift from ``score_glen_eval.py``: GLEN no longer needs to recover
gold neighbours directly. It only needs to surface *seeds* that the 2-hop
graph traversal can reach gold from.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

from genret.processing.facet_retrieval import fuse_rrf, traverse_neighbours
from genret.utils import iter_jsonl, open_auto

logger = logging.getLogger(__name__)

FACET_NAMES = ("contributions", "methods", "problems", "results")


def _load_facet_vectors(
    facet_graphs_dir: Path,
) -> tuple[dict[str, npt.NDArray[np.float32]], npt.NDArray[np.int64]]:
    """Load per-facet corpus vectors for the 1k subset."""
    vectors: dict[str, npt.NDArray[np.float32]] = {}
    corpusids: npt.NDArray[np.int64] | None = None
    for facet in FACET_NAMES:
        path = facet_graphs_dir / f"facet_graph_{facet}.npz"
        with np.load(path) as f:
            cids = np.asarray(f["corpusids"], dtype=np.int64)
            v = np.asarray(f["corpus_vectors"], dtype=np.float32)
        if corpusids is None:
            corpusids = cids
        else:
            assert np.array_equal(corpusids, cids), f"corpus id mismatch for {facet}"
        vectors[facet] = v
    assert corpusids is not None
    return vectors, corpusids


def _build_topk_graph(vectors: npt.NDArray[np.float32], top_k: int) -> csr_array:
    """Cosine-similarity top-K neighbour graph, symmetrised."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normed = vectors / norms
    sim = normed @ normed.T
    n = sim.shape[0]
    np.fill_diagonal(sim, -np.inf)
    # top-K indices per row
    if top_k >= n:
        top_idx = np.argsort(-sim, axis=1)
    else:
        top_idx = np.argpartition(-sim, kth=top_k - 1, axis=1)[:, :top_k]

    rows = np.repeat(np.arange(n, dtype=np.int32), top_idx.shape[1])
    cols = top_idx.reshape(-1).astype(np.int32)
    data = sim[rows, cols].astype(np.float32)
    # drop -inf (in case n < top_k)
    mask = np.isfinite(data)
    rows, cols, data = rows[mask], cols[mask], data[mask]
    # CSR via coo conversion
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(indptr[1:], rows, 1)
    indptr = np.cumsum(indptr)
    order = np.argsort(rows, kind="stable")
    indices = cols[order]
    sorted_data = data[order]
    graph = csr_array((sorted_data, indices, indptr), shape=(n, n))
    # symmetrise: take element-wise max with transpose
    return graph.maximum(graph.T.tocsr()).tocsr()


def _build_cross_facet_graph(
    vec_from: npt.NDArray[np.float32],
    vec_to: npt.NDArray[np.float32],
    top_k: int,
) -> csr_array:
    """Top-K cross-facet cosine graph: edges from rows of ``vec_from`` to ``vec_to``.

    Symmetrised so 2-hop BFS can traverse in either direction, matching how
    same-facet graphs are used downstream.
    """
    nf = vec_from / (np.linalg.norm(vec_from, axis=1, keepdims=True) + 1e-12)
    nt = vec_to / (np.linalg.norm(vec_to, axis=1, keepdims=True) + 1e-12)
    sim = (nf @ nt.T).astype(np.float32)
    n = sim.shape[0]
    np.fill_diagonal(sim, -np.inf)
    if top_k >= n:
        top_idx = np.argsort(-sim, axis=1)
    else:
        top_idx = np.argpartition(-sim, kth=top_k - 1, axis=1)[:, :top_k]
    rows = np.repeat(np.arange(n, dtype=np.int32), top_idx.shape[1])
    cols = top_idx.reshape(-1).astype(np.int32)
    data = sim[rows, cols]
    mask = np.isfinite(data)
    rows, cols, data = rows[mask], cols[mask], data[mask]
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(indptr[1:], rows, 1)
    indptr = np.cumsum(indptr)
    order = np.argsort(rows, kind="stable")
    indices = cols[order]
    sorted_data = data[order]
    graph = csr_array((sorted_data, indices, indptr), shape=(n, n))
    return graph.maximum(graph.T.tocsr()).tocsr()


def _load_strict_gold(
    training_pairs_path: Path, valid_ids: set[int]
) -> dict[int, set[int]]:
    """gold[paper_a] = set of paper_b across all facets, retrieval pairs only."""
    gold: dict[int, set[int]] = defaultdict(set)
    for row in iter_jsonl(training_pairs_path):
        if str(row.get("kind", "")) != "retrieval":
            continue
        a = int(row["paper_a"])
        b = int(row["paper_b"])
        if a in valid_ids and b in valid_ids:
            gold[a].add(b)
    return gold


def _load_predictions(eval_path: Path) -> dict[int, list[int]]:
    """Top-N predicted corpusids per query, ranked."""
    preds: dict[int, list[int]] = {}
    for row in iter_jsonl(eval_path):
        qid = int(row["corpusid"])
        ranked = [int(r["corpusid"]) for r in row.get("results", [])]
        preds[qid] = ranked
    return preds


def _expand_seeds(
    seeds: Sequence[int],
    graphs: dict[str, csr_array],
    cid_to_idx: dict[int, int],
    idx_to_cid: npt.NDArray[np.int64],
    *,
    hops: int,
    per_seed_cap: int,
) -> list[int]:
    """RRF-fuse 2-hop BFS expansions of each seed across all facet graphs.

    Returns ranked corpus ids (descending by fused score).
    """
    per_facet_rankings: dict[str, list[int]] = {}
    for facet, graph in graphs.items():
        scored: dict[int, float] = {}
        for seed_cid in seeds:
            root_idx = cid_to_idx.get(int(seed_cid))
            if root_idx is None:
                continue
            scored[root_idx] = max(scored.get(root_idx, 0.0), 1.0)
            walks = traverse_neighbours(
                graph,
                root=root_idx,
                k=hops,
                max_candidates=per_seed_cap,
            )
            for idx, score in walks:
                if score > scored.get(idx, 0.0):
                    scored[idx] = score
        ranked_idx = sorted(scored.keys(), key=lambda i: -scored[i])
        per_facet_rankings[facet] = ranked_idx

    fused = fuse_rrf(per_facet_rankings)
    return [int(idx_to_cid[c.index]) for c in fused]


def _score(
    preds_expanded: dict[int, list[int]],
    gold: dict[int, set[int]],
    cutoffs: Sequence[int],
) -> dict[int, dict[str, float]]:
    """Return {k: {hit, recall, precision}} averaged over queries."""
    by_k: dict[int, dict[str, float]] = {}
    n_eval = 0
    sums: dict[int, dict[str, float]] = {
        k: {"hit": 0.0, "recall": 0.0, "prec": 0.0} for k in cutoffs
    }
    for qid, ranked in preds_expanded.items():
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        n_eval += 1
        for k in cutoffs:
            top_k = ranked[:k]
            hits = sum(1 for r in top_k if r in gold_set)
            sums[k]["hit"] += 1.0 if hits else 0.0
            sums[k]["recall"] += hits / len(gold_set)
            sums[k]["prec"] += hits / max(k, 1)
    if n_eval == 0:
        return {k: {"hit": 0.0, "recall": 0.0, "prec": 0.0, "n": 0} for k in cutoffs}
    for k in cutoffs:
        by_k[k] = {
            "hit": sums[k]["hit"] / n_eval,
            "recall": sums[k]["recall"] / n_eval,
            "prec": sums[k]["prec"] / n_eval,
            "n": n_eval,
        }
    return by_k


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("eval_file", type=Path, help="GLEN prediction JSONL.")
    p.add_argument(
        "--facet-graphs-dir",
        type=Path,
        default=Path("data/s2orc-nlp/splits-connected-1k/processed"),
        help="Dir with facet_graph_{facet}.npz vector files.",
    )
    p.add_argument(
        "--training-pairs",
        type=Path,
        default=Path("output/genret/codebook_training_pairs_1k.jsonl.zst"),
        help="Pairs file with strict-gold retrieval rows.",
    )
    p.add_argument(
        "--seed-k",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 50],
        help="Number of GLEN seeds to expand (sweep).",
    )
    p.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[20, 50, 100, 200],
        help="Top-K to score the expanded set at.",
    )
    p.add_argument(
        "--cross-facet",
        action="append",
        default=[],
        help="Add a cross-facet graph as 'from:to', e.g. 'problems:contributions'. "
        "Repeatable.",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="If set, write the post-expansion ranked candidate list per query as JSONL "
        "in score_glen format. Uses the largest --seed-k value; cutoff = max(--cutoffs).",
    )
    p.add_argument("--graph-top-k", type=int, default=20)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument(
        "--per-seed-cap",
        type=int,
        default=500,
        help="Max candidates per seed after BFS (capped per traverse call).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(levelname)s %(name)s: %(message)s"
    )

    logger.info("Loading facet vectors from %s", args.facet_graphs_dir)
    vectors, corpusids = _load_facet_vectors(args.facet_graphs_dir)
    cid_to_idx = {int(c): i for i, c in enumerate(corpusids)}

    logger.info(
        "Building top-%d cosine neighbour graphs for %d facets, %d papers",
        args.graph_top_k,
        len(vectors),
        len(corpusids),
    )
    graphs = {f: _build_topk_graph(v, args.graph_top_k) for f, v in vectors.items()}
    for spec in args.cross_facet:
        from_facet, to_facet = spec.split(":")
        if from_facet not in vectors or to_facet not in vectors:
            raise ValueError(f"unknown facet in --cross-facet {spec!r}")
        name = f"{from_facet}_x_{to_facet}"
        logger.info("Building cross-facet graph %s", name)
        graphs[name] = _build_cross_facet_graph(
            vectors[from_facet], vectors[to_facet], args.graph_top_k
        )

    logger.info("Loading strict gold from %s", args.training_pairs)
    gold = _load_strict_gold(args.training_pairs, valid_ids=set(cid_to_idx))

    logger.info("Loading predictions from %s", args.eval_file)
    preds = _load_predictions(args.eval_file)

    print(f"\n=== {args.eval_file} ===")
    print(f"  graph: top-{args.graph_top_k} cosine edges, {args.hops}-hop BFS")
    print(
        f"  papers={len(corpusids)}, queries with gold={sum(1 for q in preds if gold.get(q))}\n"
    )

    print(f"  {'seed_K':>6}  {'K_post':>6}  {'hit@K':>7}  {'R@K':>8}  {'P@K':>8}")
    print(f"  {'-' * 6}  {'-' * 6}  {'-' * 7}  {'-' * 8}  {'-' * 8}")
    max_seed_k = max(args.seed_k)
    max_cutoff = max(args.cutoffs)
    for seed_k in args.seed_k:
        preds_expanded: dict[int, list[int]] = {}
        for qid, ranked in preds.items():
            seeds = ranked[:seed_k]
            preds_expanded[qid] = _expand_seeds(
                seeds,
                graphs,
                cid_to_idx,
                corpusids,
                hops=args.hops,
                per_seed_cap=args.per_seed_cap,
            )
        scored = _score(preds_expanded, gold, args.cutoffs)
        for k in args.cutoffs:
            row = scored[k]
            print(
                f"  {seed_k:>6}  {k:>6}  "
                f"{row['hit']:>7.4f}  {row['recall']:>8.4f}  {row['prec']:>8.4f}"
            )
        print()
        if args.output_jsonl is not None and seed_k == max_seed_k:
            args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open_auto(args.output_jsonl, "w") as fh:
                for qid, ranked in preds_expanded.items():
                    rec = {
                        "corpusid": int(qid),
                        "results": [
                            {
                                "corpusid": pid,
                                "score": round(1.0 / (rank + 1), 6),
                                "facets": ["post_expansion"],
                                "facet_ranks": {"post_expansion": rank},
                            }
                            for rank, pid in enumerate(ranked[:max_cutoff])
                        ],
                    }
                    fh.write(json.dumps(rec) + "\n")
            logger.info("Wrote post-expansion JSONL to %s", args.output_jsonl)


if __name__ == "__main__":
    main()
