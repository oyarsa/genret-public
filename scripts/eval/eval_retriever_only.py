"""Compute retriever-only nDCG@20 for a retrieval file at a given depth.

No reranker needed — just measures how well the retriever orders candidates.

Example:
    python scripts/eval/eval_retriever_only.py \
        --retrieval output/maxsim_minilm/retrieval_500q.jsonl \
        --depth 20 --scale 500q
"""

import argparse
import json
from collections import defaultdict

from genret.processing.llm_judge import JudgedPair, normalise_judged_pair_dict
from genret.processing.reranking_compare import calc_ndcg, convert_gpt_relevance
from genret.utils import mean


def main() -> None:
    """Evaluate retriever-only performance against ground truth."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--scale", choices=["100q", "500q"], default="500q")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    if args.scale == "500q":
        gpt_paths = ["output/eval_500/gpt5mini_judgements.jsonl"]
        bins_path = "output/eval_500/query_papers_500.jsonl"
    else:
        gpt_paths = [
            "output/reranking_eval/gpt5mini_judgements.jsonl",
            "output/cross_facet_eval/gpt5mini_new_judgements.jsonl",
        ]
        bins_path = "output/reranking_eval/query_papers_100.jsonl"

    gpt_labels: dict[tuple[int, int], JudgedPair] = {}
    for path in gpt_paths:
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                pair = JudgedPair.model_validate(normalise_judged_pair_dict(data))
                gpt_labels[(pair.paper_a, pair.paper_b)] = pair

    query_bins: dict[int, str] = {}
    with open(bins_path) as f:
        for line in f:
            rec = json.loads(line)
            query_bins[rec["corpusid"]] = rec["bin"]

    retrieval: dict[int, list[int]] = {}
    with open(args.retrieval) as f:
        for line in f:
            rec = json.loads(line)
            retrieval[rec["corpusid"]] = [
                r["corpusid"] for r in rec["results"][: args.depth]
            ]

    k = 20
    ndcg_by_bin: dict[str, list[float]] = defaultdict(list)
    unlabelled = 0

    for qid, candidates in retrieval.items():
        bin_name = query_bins.get(qid, "unknown")
        scored = [c for c in candidates if (qid, c) in gpt_labels]
        unlabelled += len(candidates) - len(scored)
        if not scored:
            continue
        rels = [convert_gpt_relevance(gpt_labels[(qid, c)]) for c in scored]
        ndcg_by_bin[bin_name].append(calc_ndcg(rels, k))
        ndcg_by_bin["overall"].append(calc_ndcg(rels, k))

    label = args.label or args.retrieval
    print(f"\n=== {label} (depth={args.depth}, {args.scale}) ===")
    print(f"  Retriever nDCG@{k}:  {mean(ndcg_by_bin['overall']):.3f}")
    for b in ["sparse", "low_mid", "high"]:
        if b in ndcg_by_bin:
            print(f"  {b:10s}:         {mean(ndcg_by_bin[b]):.3f}")
    print(f"  Queries:            {len(ndcg_by_bin['overall'])}")
    print(f"  Unlabelled pairs:   {unlabelled}")


if __name__ == "__main__":
    main()
