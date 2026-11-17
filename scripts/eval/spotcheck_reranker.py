"""Score spot-check pairs with a cross-encoder and compare against GPT oracles.

Computes per-query Kendall tau between the reranker and each oracle (GPT-5-mini,
GPT-5.4) to see which oracle the reranker agrees with more.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from genret.paper_parsing import load_facets_map
from genret.processing.llm_judge import JudgedPair, normalise_judged_pair_dict
from genret.processing.reranker import (
    LocalReranker,
    RerankerConfig,
    format_paper_pair,
    load_titles,
)
from genret.processing.reranking_compare import convert_gpt_relevance, kendall_tau
from genret.types import CorpusId

FACETS_PATH = Path("data/s2orc-nlp/splits-connected-10k/processed/facets.jsonl.zst")
TITLES_PATH = Path(
    "data/s2orc-nlp/splits-connected-10k/processed/paper_graphs.jsonl.zst"
)


def main() -> None:
    """Run the spot-check comparison."""
    # Load GPT-5-mini judgements
    gpt_mini: dict[tuple[int, int], JudgedPair] = {}
    for path in [
        "output/reranking_eval/gpt5mini_judgements.jsonl",
        "output/cross_facet_eval/gpt5mini_new_judgements.jsonl",
    ]:
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                pair = JudgedPair.model_validate(normalise_judged_pair_dict(data))
                gpt_mini[(pair.paper_a, pair.paper_b)] = pair

    # Load GPT-5.4 judgements (original + expanded)
    gpt54: dict[tuple[int, int], JudgedPair] = {}
    for path in [
        "output/reranking_eval/gpt54_spotcheck.jsonl",
        "output/reranking_eval/gpt54_spotcheck_expanded.jsonl",
    ]:
        try:
            with open(path) as f:
                for line in f:
                    data = json.loads(line.strip())
                    pair = JudgedPair.model_validate(normalise_judged_pair_dict(data))
                    gpt54[(pair.paper_a, pair.paper_b)] = pair
        except FileNotFoundError:
            pass

    # Load spot-check pairs grouped by query (original + expanded)
    queries: dict[int, list[int]] = defaultdict(list)
    for path in [
        "output/reranking_eval/spotcheck_pairs_500.jsonl",
        "output/reranking_eval/spotcheck_pairs_expanded.jsonl",
    ]:
        try:
            with open(path) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    queries[rec["paper_a"]].append(rec["paper_b"])
        except FileNotFoundError:
            pass

    print(f"GPT-5-mini labels: {len(gpt_mini)}")
    print(f"GPT-5.4 labels: {len(gpt54)}")
    print(f"Spot-check queries: {len(queries)}")
    print(f"Spot-check pairs: {sum(len(v) for v in queries.values())}")

    # Load facets and titles
    facets_map = load_facets_map(FACETS_PATH)
    titles = load_titles(TITLES_PATH)

    # Score with ModernBERT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_path = "output/phase2_modernbert_150k/checkpoint"
    config = RerankerConfig(model_name=model_path, device=device, max_length=2048)
    model = LocalReranker.from_pretrained(config)
    print(f"Model loaded: {model_path}")

    # Score all pairs and compute per-query rankings
    taus_mini: list[float] = []
    taus_54: list[float] = []
    taus_mini_vs_54: list[float] = []

    for qid, candidates in sorted(queries.items()):
        qcid = CorpusId(qid)
        if qcid not in facets_map:
            continue

        # Score with reranker
        texts = [
            format_paper_pair(
                titles.get(qcid, ""),
                facets_map[qcid],
                titles.get(CorpusId(c), ""),
                facets_map[CorpusId(c)],
            )
            for c in candidates
            if CorpusId(c) in facets_map
        ]
        valid_candidates = [c for c in candidates if CorpusId(c) in facets_map]

        if len(valid_candidates) < 2:
            continue

        logits = model.score_pairs(texts, batch_size=4)
        reranker_ranking = [valid_candidates[int(i)] for i in np.argsort(-logits)]

        # GPT-5-mini ranking
        mini_scores = {
            c: convert_gpt_relevance(gpt_mini[(qid, c)])
            for c in valid_candidates
            if (qid, c) in gpt_mini
        }
        mini_ranking = sorted(mini_scores, key=lambda c: -mini_scores[c])

        # GPT-5.4 ranking
        scores_54 = {
            c: convert_gpt_relevance(gpt54[(qid, c)])
            for c in valid_candidates
            if (qid, c) in gpt54
        }
        ranking_54 = sorted(scores_54, key=lambda c: -scores_54[c])

        tau_mini = kendall_tau(reranker_ranking, mini_ranking)
        tau_54 = kendall_tau(reranker_ranking, ranking_54)
        tau_mini_54 = kendall_tau(mini_ranking, ranking_54)

        taus_mini.append(tau_mini)
        taus_54.append(tau_54)
        taus_mini_vs_54.append(tau_mini_54)

        print(
            f"  Query {qid}: "
            f"reranker↔mini={tau_mini:+.3f}  "
            f"reranker↔5.4={tau_54:+.3f}  "
            f"mini↔5.4={tau_mini_54:+.3f}  "
            f"({'agrees with 5.4' if tau_54 > tau_mini else 'agrees with mini'})"
        )

    print(f"\n=== SUMMARY ({len(taus_mini)} queries) ===")
    print(f"  Mean Kendall tau (reranker ↔ GPT-5-mini): {np.mean(taus_mini):.3f}")
    print(f"  Mean Kendall tau (reranker ↔ GPT-5.4):    {np.mean(taus_54):.3f}")
    print(f"  Mean Kendall tau (GPT-5-mini ↔ GPT-5.4):  {np.mean(taus_mini_vs_54):.3f}")
    agrees_54 = sum(1 for a, b in zip(taus_54, taus_mini) if a > b)
    print(f"  Reranker agrees more with GPT-5.4: {agrees_54}/{len(taus_mini)} queries")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
