"""Evaluate Phase 2 model checkpoints on 100-query or 500-query sets.

Uses the same format_paper_pair() used during training (titles + all facets).
Pass --scale 500 to run on the larger 500-query validation set.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

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

# Scale-dependent defaults
SCALE_CONFIG: dict[int, dict[str, Any]] = {
    100: {
        "label_paths": [
            "output/reranking_eval/gpt5mini_judgements.jsonl",
            "output/cross_facet_eval/gpt5mini_new_judgements.jsonl",
        ],
        "label_paths_optional": False,
        "query_bins_path": "output/reranking_eval/query_papers_100.jsonl",
        "retrieval_default": "output/cross_facet_eval/retrieval_top100.jsonl",
        "baselines": [
            (
                "output/distilled_reranker_round1/checkpoint",
                "Round1 DeBERTa-base (150k diverse)",
                512,
                16,
            ),
        ],
        "models": [
            (
                "output/phase2_modernbert_55k-v2/checkpoint",
                "Phase2 ModernBERT 55k-v2",
                2048,
                4,
            ),
            ("output/phase2_gte_55k-v2/checkpoint", "Phase2 GTE 55k-v2", 2048, 4),
            (
                "output/phase2_modernbert_150k/checkpoint",
                "Phase2 ModernBERT 150k",
                2048,
                4,
            ),
            ("output/phase2_gte_150k/checkpoint", "Phase2 GTE 150k", 2048, 4),
            (
                "output/phase2_modernbert_150k_no_results_h200/checkpoint",
                "Wave2 ModernBERT 150k no-results (H200)",
                2048,
                4,
            ),
        ],
    },
    500: {
        "label_paths": [
            "output/eval_500/gpt5mini_judgements.jsonl",
            "output/eval_500/gpt5mini_new_embedding_judgements.jsonl",
            "output/eval_500/gpt5mini_missing_embedding_judgements.jsonl",
        ],
        "label_paths_optional": True,
        "query_bins_path": "output/eval_500/query_papers_500.jsonl",
        "retrieval_default": "output/eval_500/retrieval_cross_facet.jsonl",
        "baselines": [],
        "models": [
            (
                "output/phase2_modernbert_150k/checkpoint",
                "ModernBERT 150k",
                2048,
                4,
            ),
            (
                "output/phase2_gte_150k/checkpoint",
                "GTE 150k",
                2048,
                4,
            ),
        ],
    },
}


def evaluate_model(
    model_path: str,
    label: str,
    retrieval: dict[int, list[int]],
    facets_map: dict[CorpusId, Facets],
    titles: dict[CorpusId, str],
    gpt_labels: dict[tuple[int, int], JudgedPair],
    query_bins: dict[int, str],
    device: str,
    batch_size: int = 16,
    max_length: int = 2048,
) -> dict[str, float] | None:
    """Evaluate one checkpoint. Returns metrics dict or None on failure."""
    print(f"\n=== {label} ===")
    print(f"  Path: {model_path}")

    if not Path(model_path).exists():
        print("  SKIPPED: checkpoint not found")
        return None

    try:
        config = RerankerConfig(
            model_name=model_path, device=device, max_length=max_length
        )
        model = LocalReranker.from_pretrained(config)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None

    k = 20
    ret_ndcg: dict[str, list[float]] = defaultdict(list)
    rer_ndcg: dict[str, list[float]] = defaultdict(list)
    total_pairs = 0
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
    print(f"  Queries evaluated:  {len(ret_ndcg['overall'])}")
    print(f"  Pairs scored:       {total_pairs}")
    print(f"  Time:               {elapsed:.1f}s ({total_pairs / elapsed:.0f} pairs/s)")

    del model
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    """Run the evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale",
        type=int,
        choices=[100, 500],
        default=100,
        help="Query set scale: 100 (default) or 500",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of retrieval candidates per query (default: 50)",
    )
    parser.add_argument(
        "--retrieval",
        type=str,
        default=None,
        help="Path to retrieval results file (default: scale-dependent)",
    )
    args = parser.parse_args()

    cfg = SCALE_CONFIG[args.scale]
    retrieval_path = args.retrieval or cfg["retrieval_default"]
    optional = cfg["label_paths_optional"]

    gpt_labels: dict[tuple[int, int], JudgedPair] = {}
    for path in cfg["label_paths"]:
        try:
            with open(path) as f:
                for line in f:
                    data = json.loads(line.strip())
                    pair = JudgedPair.model_validate(normalise_judged_pair_dict(data))
                    gpt_labels[(pair.paper_a, pair.paper_b)] = pair
        except FileNotFoundError:
            if not optional:
                raise

    facets_map = load_facets_map(FACETS_PATH)
    titles = load_titles(TITLES_PATH)

    query_bins: dict[int, str] = {}
    with open(str(cfg["query_bins_path"])) as f:
        for line in f:
            rec = json.loads(line)
            query_bins[rec["corpusid"]] = rec["bin"]

    retrieval: dict[int, list[int]] = {}
    with open(retrieval_path) as f:
        for line in f:
            rec = json.loads(line)
            retrieval[rec["corpusid"]] = [
                r["corpusid"] for r in rec["results"][: args.top_k]
            ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Scale: {args.scale}q")
    print(f"Device: {device}")
    print(f"Top-K: {args.top_k}")
    print(f"Retrieval: {retrieval_path}")
    print(f"GPT labels: {len(gpt_labels)}")
    print(f"Queries: {len(retrieval)}")
    print(f"Titles loaded: {len(titles)}")

    all_models = list(cfg["baselines"]) + list(cfg["models"])

    results: dict[str, dict[str, float]] = {}
    for model_path, label, max_len, bs in all_models:
        metrics = evaluate_model(
            model_path,
            label,
            retrieval,
            facets_map,
            titles,
            gpt_labels,
            query_bins,
            device,
            batch_size=bs,
            max_length=max_len,
        )
        if metrics:
            results[label] = metrics

    print(f"\n\n=== SUMMARY ({args.scale}-query) ===")
    print(f"{'Model':<40s} {'Ret nDCG':>10s} {'Rer nDCG':>10s} {'Lift':>8s}")
    print("-" * 70)
    for label, m in results.items():
        print(
            f"{label:<40s} {m['ret_ndcg20']:>10.3f}"
            f" {m['rer_ndcg20']:>10.3f} {m['lift']:>+8.3f}"
        )


if __name__ == "__main__":
    main()
