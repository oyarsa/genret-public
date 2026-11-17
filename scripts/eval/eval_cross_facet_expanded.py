"""Compare alternative retrieval configs with ModernBERT 150k reranker.

Compares a candidate retrieval file against the baseline (5 graphs: 4
same-facet + problems→contributions).

Examples:
    # Cross-facet expansion
    python scripts/eval/eval_cross_facet_expanded.py \
        --candidate output/cross_facet_expanded/retrieval_100q.jsonl \
        --label "Expanded (8 graphs)"

    # MaxSim retrieval
    python scripts/eval/eval_cross_facet_expanded.py \
        --candidate output/maxsim_eval_gpu/retrieval_100q.jsonl \
        --label "MaxSim (max_pairs)"
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from genret.paper_parsing import Facets, load_facets_map
from genret.processing.llm_judge import JudgedPair, normalise_judged_pair_dict
from genret.processing.reranker import (
    LocalReranker,
    RerankerConfig,
    format_paper_pair,
    load_titles,
)
from genret.processing.reranking_compare import calc_ndcg, convert_gpt_relevance
from genret.types import CorpusId
from genret.utils import mean

FACETS_PATH = Path("data/s2orc-nlp/splits-connected-10k/processed/facets.jsonl.zst")
TITLES_PATH = Path(
    "data/s2orc-nlp/splits-connected-10k/processed/paper_graphs.jsonl.zst"
)


def evaluate_retrieval(
    label: str,
    retrieval_path: str,
    facets_map: dict[CorpusId, Facets],
    titles: dict[CorpusId, str],
    gpt_labels: dict[tuple[int, int], JudgedPair],
    query_bins: dict[int, str],
    model: LocalReranker,
    batch_size: int = 4,
) -> dict[str, float] | None:
    """Evaluate one retrieval config with the reranker."""
    print(f"\n=== {label} ===")

    retrieval: dict[int, list[int]] = {}
    with open(retrieval_path) as f:
        for line in f:
            rec = json.loads(line)
            retrieval[rec["corpusid"]] = [r["corpusid"] for r in rec["results"][:50]]

    k = 20
    ret_ndcg: dict[str, list[float]] = defaultdict(list)
    rer_ndcg: dict[str, list[float]] = defaultdict(list)
    total_pairs = 0
    unlabelled = 0
    start = time.time()

    for qid, candidates in retrieval.items():
        bin_name = query_bins.get(qid, "unknown")
        qcid = CorpusId(qid)
        scored = [
            c
            for c in candidates
            if (qid, c) in gpt_labels
            and qcid in facets_map
            and CorpusId(c) in facets_map
        ]
        unlabelled += len(candidates) - len(scored)
        if not scored:
            continue

        gpt_rel = {c: convert_gpt_relevance(gpt_labels[(qid, c)]) for c in scored}
        ret_ranking = [c for c in candidates if c in gpt_rel]
        ret_rels = [gpt_rel[c] for c in ret_ranking]

        texts = [
            format_paper_pair(
                titles.get(qcid, ""),
                facets_map[qcid],
                titles.get(CorpusId(c), ""),
                facets_map[CorpusId(c)],
            )
            for c in ret_ranking
        ]
        logits = model.score_pairs(texts, batch_size=batch_size)
        total_pairs += len(texts)

        ranked_indices = np.argsort(-logits)
        reranked = [ret_ranking[int(i)] for i in ranked_indices]
        rer_rels = [gpt_rel[c] for c in reranked]

        ret_ndcg[bin_name].append(calc_ndcg(ret_rels, k))
        ret_ndcg["overall"].append(calc_ndcg(ret_rels, k))
        rer_ndcg[bin_name].append(calc_ndcg(rer_rels, k))
        rer_ndcg["overall"].append(calc_ndcg(rer_rels, k))

    elapsed = time.time() - start

    metrics = {
        "ret_ndcg20": mean(ret_ndcg["overall"]),
        "rer_ndcg20": mean(rer_ndcg["overall"]),
        "lift": mean(rer_ndcg["overall"]) - mean(ret_ndcg["overall"]),
    }

    print(f"  Retriever nDCG@20:  {metrics['ret_ndcg20']:.3f}")
    print(f"  Reranked nDCG@20:   {metrics['rer_ndcg20']:.3f}")
    print(f"  Lift:               {metrics['lift']:+.3f}")
    for b in ["sparse", "low_mid", "high"]:
        if b in rer_ndcg:
            print(
                f"  {b:10s}:         ret={mean(ret_ndcg[b]):.3f}"
                f"  rer={mean(rer_ndcg[b]):.3f}"
                f"  lift={mean(rer_ndcg[b]) - mean(ret_ndcg[b]):+.3f}"
            )
    print(f"  Pairs scored:       {total_pairs}")
    print(f"  Unlabelled pairs:   {unlabelled}")
    print(f"  Time:               {elapsed:.1f}s ({total_pairs / elapsed:.0f} pairs/s)")
    return metrics


def main() -> None:
    """Run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Compare retrieval configs with ModernBERT 150k reranker"
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to candidate retrieval JSONL file",
    )
    parser.add_argument(
        "--label",
        default="Candidate",
        help="Label for the candidate config in output",
    )
    parser.add_argument(
        "--baseline",
        default="output/cross_facet_eval/retrieval_top100.jsonl",
        help="Path to baseline retrieval JSONL file",
    )
    parser.add_argument(
        "--scale",
        choices=["100q", "500q"],
        default="100q",
        help="Evaluation scale: 100q or 500q",
    )
    args = parser.parse_args()

    # GPT label and query bin files depend on scale
    if args.scale == "500q":
        gpt_label_paths = [
            "output/eval_500/gpt5mini_judgements.jsonl",
        ]
        query_bins_path = "output/eval_500/query_papers_500.jsonl"
        if args.baseline == "output/cross_facet_eval/retrieval_top100.jsonl":
            args.baseline = "output/eval_500/retrieval_cross_facet.jsonl"
    else:
        gpt_label_paths = [
            "output/reranking_eval/gpt5mini_judgements.jsonl",
            "output/cross_facet_eval/gpt5mini_new_judgements.jsonl",
        ]
        query_bins_path = "output/reranking_eval/query_papers_100.jsonl"

    gpt_labels: dict[tuple[int, int], JudgedPair] = {}
    for path in gpt_label_paths:
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                pair = JudgedPair.model_validate(normalise_judged_pair_dict(data))
                gpt_labels[(pair.paper_a, pair.paper_b)] = pair

    facets_map = load_facets_map(FACETS_PATH)
    titles = load_titles(TITLES_PATH)

    query_bins: dict[int, str] = {}
    with open(query_bins_path) as f:
        for line in f:
            rec = json.loads(line)
            query_bins[rec["corpusid"]] = rec["bin"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"GPT labels: {len(gpt_labels)}")

    model_path = "output/phase2_modernbert_150k/checkpoint"
    config = RerankerConfig(model_name=model_path, device=device, max_length=2048)
    model = LocalReranker.from_pretrained(config)
    print(f"Model: {model_path}")

    results: dict[str, dict[str, float]] = {}

    baseline = evaluate_retrieval(
        "Baseline (mean_pool, 5 graphs)",
        args.baseline,
        facets_map,
        titles,
        gpt_labels,
        query_bins,
        model,
    )
    if baseline:
        results["Baseline"] = baseline

    candidate = evaluate_retrieval(
        args.label,
        args.candidate,
        facets_map,
        titles,
        gpt_labels,
        query_bins,
        model,
    )
    if candidate:
        results[args.label] = candidate

    # Summary
    print("\n\n=== SUMMARY ===")
    print(f"{'Config':<40s} {'Ret nDCG':>10s} {'Rer nDCG':>10s} {'Lift':>8s}")
    print("-" * 70)
    for label, m in results.items():
        print(
            f"{label:<40s} {m['ret_ndcg20']:>10.3f}"
            f" {m['rer_ndcg20']:>10.3f} {m['lift']:>+8.3f}"
        )

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
