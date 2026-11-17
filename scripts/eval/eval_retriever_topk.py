"""Compare retriever nDCG@20 across retrieval configs and top-k values.

Lightweight evaluation that only needs retrieval files and GPT labels —
no model loading required.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from genret.processing.llm_judge import JudgedPair, normalise_judged_pair_dict
from genret.processing.reranking_compare import calc_ndcg, convert_gpt_relevance
from genret.utils import mean


def load_gpt_labels(
    paths: list[str],
) -> dict[tuple[int, int], JudgedPair]:
    """Load GPT judgement labels from one or more JSONL files."""
    labels: dict[tuple[int, int], JudgedPair] = {}
    for path in paths:
        try:
            with open(path) as f:
                for line in f:
                    data = json.loads(line.strip())
                    pair = JudgedPair.model_validate(normalise_judged_pair_dict(data))
                    labels[(pair.paper_a, pair.paper_b)] = pair
        except FileNotFoundError:
            pass
    return labels


def evaluate_retriever(
    retrieval_path: str,
    top_k: int,
    gpt_labels: dict[tuple[int, int], JudgedPair],
    query_bins: dict[int, str],
) -> dict[str, float]:
    """Compute retriever nDCG@20 for a retrieval file at a given top-k."""
    ndcg_k = 20
    ndcg_by_bin: dict[str, list[float]] = defaultdict(list)
    n_queries = 0
    n_scored = 0
    n_total = 0

    with open(retrieval_path) as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["corpusid"]
            candidates = [r["corpusid"] for r in rec["results"][:top_k]]
            bin_name = query_bins.get(qid, "unknown")
            n_total += len(candidates)

            scored = [
                c
                for c in candidates
                if (qid, c) in gpt_labels or (c, qid) in gpt_labels
            ]
            if not scored:
                continue

            n_queries += 1
            n_scored += len(scored)

            gpt_rel: dict[int, float] = {}
            for c in scored:
                key = (qid, c) if (qid, c) in gpt_labels else (c, qid)
                gpt_rel[c] = convert_gpt_relevance(gpt_labels[key])

            ret_ranking = [c for c in candidates if c in gpt_rel]
            ret_rels = [gpt_rel[c] for c in ret_ranking]

            ndcg_by_bin[bin_name].append(calc_ndcg(ret_rels, ndcg_k))
            ndcg_by_bin["overall"].append(calc_ndcg(ret_rels, ndcg_k))

    result = {
        "ndcg20": mean(ndcg_by_bin["overall"]),
        "queries": n_queries,
        "scored_pairs": n_scored,
        "total_pairs": n_total,
        "coverage": n_scored / n_total if n_total else 0.0,
    }
    for b in ["sparse", "low_mid", "high"]:
        if b in ndcg_by_bin:
            result[f"ndcg20_{b}"] = mean(ndcg_by_bin[b])
    return result


def main() -> None:
    """Run retriever-only nDCG comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set",
        choices=["100q", "500q"],
        default="100q",
        help="Evaluation query set (default: 100q)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[20, 50],
        help="Top-k values to evaluate (default: 20 50)",
    )
    parser.add_argument(
        "retrieval_files",
        nargs="+",
        help="Retrieval JSONL files to compare",
    )
    args = parser.parse_args()

    # Load GPT labels and query bins for the chosen eval set
    if args.eval_set == "100q":
        label_paths = [
            "output/reranking_eval/gpt5mini_judgements.jsonl",
            "output/cross_facet_eval/gpt5mini_new_judgements.jsonl",
        ]
        bins_path = "output/reranking_eval/query_papers_100.jsonl"
    else:
        label_paths = [
            "output/eval_500/gpt5mini_judgements.jsonl",
            "output/eval_500/gpt5mini_new_embedding_judgements.jsonl",
            "output/eval_500/gpt5mini_missing_embedding_judgements.jsonl",
        ]
        bins_path = "output/eval_500/query_papers_500.jsonl"

    gpt_labels = load_gpt_labels(label_paths)
    print(f"GPT labels: {len(gpt_labels)}")

    query_bins: dict[int, str] = {}
    with open(bins_path) as f:
        for line in f:
            rec = json.loads(line)
            query_bins[rec["corpusid"]] = rec["bin"]
    print(f"Query bins: {len(query_bins)}")

    # Evaluate each retrieval file at each top-k
    rows: list[tuple[str, int, dict[str, float]]] = []
    for path in args.retrieval_files:
        label = Path(path).parent.name
        for k in args.top_k:
            metrics = evaluate_retriever(path, k, gpt_labels, query_bins)
            rows.append((label, k, metrics))
            print(
                f"  {label} top-{k}: nDCG@20={metrics['ndcg20']:.3f}"
                f"  coverage={metrics['coverage']:.1%}"
                f"  queries={metrics['queries']}"
            )

    # Summary table
    print(
        f"\n{'Config':<25s} {'Top-K':>5s} {'nDCG@20':>8s}"
        f" {'Sparse':>8s} {'LowMid':>8s} {'High':>8s} {'Cov':>6s}"
    )
    print("-" * 70)
    for label, k, m in rows:
        sparse = m.get("ndcg20_sparse", 0.0)
        low_mid = m.get("ndcg20_low_mid", 0.0)
        high = m.get("ndcg20_high", 0.0)
        print(
            f"{label:<25s} {k:>5d} {m['ndcg20']:>8.3f}"
            f" {sparse:>8.3f} {low_mid:>8.3f} {high:>8.3f}"
            f" {m['coverage']:>5.1%}"
        )


if __name__ == "__main__":
    main()
