"""Generate expanded spot-check pairs for GPT-5.4 oracle validation.

Computes per-query Kendall tau between GPT-5-mini and Qwen3-32B local judge,
identifies the 10 queries already used in the original spot-check, then selects
the next 20 queries by disagreement (ranked 11-30) and writes their candidate
pairs to a new JSONL file.
"""

import json
from collections import defaultdict
from pathlib import Path

from genret.processing.llm_judge import (
    JudgedPair,
    ScoredPair,
    normalise_judged_pair_dict,
)
from genret.processing.reranking_compare import convert_gpt_relevance, kendall_tau
from genret.utils import open_auto

BASE = Path("output/reranking_eval")

GPT_MINI_PATH = BASE / "gpt5mini_judgements.jsonl"
LOCAL_JUDGE_PATH = BASE / "local_judge_shards" / "reranking.merged.jsonl.zst"
EXISTING_SPOTCHECK_PATH = BASE / "spotcheck_pairs_500.jsonl"
RETRIEVAL_PATH = BASE / "retrieval_top100.jsonl"
OUTPUT_PATH = BASE / "spotcheck_pairs_expanded.jsonl"


def load_gpt_mini_by_query(path: Path) -> dict[int, dict[int, float]]:
    """Load GPT-5-mini judgements as query -> {candidate -> score}."""
    by_query: defaultdict[int, dict[int, float]] = defaultdict(dict)
    with open_auto(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            pair = JudgedPair.model_validate(normalise_judged_pair_dict(data))
            by_query[pair.paper_a][pair.paper_b] = convert_gpt_relevance(pair)
    return dict(by_query)


def load_local_scores_by_query(path: Path) -> dict[int, dict[int, float]]:
    """Load local judge scores as query -> {candidate -> score}."""
    by_query: defaultdict[int, dict[int, float]] = defaultdict(dict)
    with open_auto(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            pair = ScoredPair.model_validate_json(stripped)
            by_query[pair.paper_a][pair.paper_b] = pair.score
    return dict(by_query)


def load_existing_spotcheck_queries(path: Path) -> set[int]:
    """Load query IDs already used in the existing spot-check."""
    queries: set[int] = set()
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            queries.add(rec["paper_a"])
    return queries


def load_retrieval_candidates(path: Path) -> dict[int, list[int]]:
    """Load retrieval results as query -> [candidate_ids]."""
    results: dict[int, list[int]] = {}
    with open_auto(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            results[rec["corpusid"]] = [r["corpusid"] for r in rec["results"]]
    return results


def compute_per_query_tau(
    gpt_by_query: dict[int, dict[int, float]],
    local_by_query: dict[int, dict[int, float]],
) -> list[tuple[int, float, int]]:
    """Compute Kendall tau for each query that has both GPT and local scores.

    Returns list of (query_id, tau, n_common) sorted by tau ascending
    (most disagreement first).
    """
    results: list[tuple[int, float, int]] = []

    all_queries = set(gpt_by_query.keys()) & set(local_by_query.keys())
    for qid in sorted(all_queries):
        gpt_scores = gpt_by_query[qid]
        local_scores = local_by_query[qid]

        # Only consider candidates with both scores
        common = set(gpt_scores.keys()) & set(local_scores.keys())
        if len(common) < 2:
            continue

        # Build rankings (sorted by score descending)
        gpt_ranking = sorted(common, key=lambda c: -gpt_scores[c])
        local_ranking = sorted(common, key=lambda c: -local_scores[c])

        tau = kendall_tau(gpt_ranking, local_ranking)
        results.append((qid, tau, len(common)))

    # Sort by tau ascending (most disagreement first)
    results.sort(key=lambda x: x[1])
    return results


def main() -> None:
    """Generate expanded spot-check pairs."""
    print("Loading GPT-5-mini judgements...")
    gpt_by_query = load_gpt_mini_by_query(GPT_MINI_PATH)
    print(f"  {len(gpt_by_query)} queries")

    print("Loading local judge scores...")
    local_by_query = load_local_scores_by_query(LOCAL_JUDGE_PATH)
    print(f"  {len(local_by_query)} queries")

    print("Loading existing spot-check queries...")
    existing_queries = load_existing_spotcheck_queries(EXISTING_SPOTCHECK_PATH)
    print(f"  {len(existing_queries)} queries already used")

    print("Loading retrieval candidates...")
    retrieval = load_retrieval_candidates(RETRIEVAL_PATH)
    print(f"  {len(retrieval)} queries")

    # Compute per-query Kendall tau
    print("\nComputing per-query Kendall tau (GPT-5-mini vs Qwen3-32B)...")
    tau_results = compute_per_query_tau(gpt_by_query, local_by_query)
    print(f"  {len(tau_results)} queries with both scores")

    # Show which queries were in the original 10
    print("\n=== ORIGINAL 10 QUERIES (most disagreement) ===")
    print(f"{'Rank':>4}  {'Query ID':>12}  {'Tau':>7}  {'N':>4}  {'In existing?':>12}")
    print("-" * 50)
    original_found = 0
    for rank, (qid, tau, n) in enumerate(tau_results, 1):
        in_existing = qid in existing_queries
        if rank <= 10:
            marker = "YES" if in_existing else "NO"
            print(f"{rank:4d}  {qid:12d}  {tau:+.4f}  {n:4d}  {marker:>12}")
            if in_existing:
                original_found += 1

    print(
        f"\nVerification: {original_found}/10 of the top-10 match existing spot-check"
    )

    # Find the next 20 queries (ranked 11-30), excluding already-used queries
    # First, identify all queries sorted by disagreement, skip the existing ones
    new_candidates = [
        (qid, tau, n) for qid, tau, n in tau_results if qid not in existing_queries
    ]

    # The first 10 in new_candidates correspond to ranks 11-30 (ish) if the
    # original 10 were indeed the top-10 by disagreement. Take next 20.
    selected = new_candidates[:10]

    print("\n=== NEXT 10 QUERIES (for expanded spot-check) ===")
    print(f"{'Rank':>4}  {'Query ID':>12}  {'Tau':>7}  {'N':>4}")
    print("-" * 35)
    for i, (qid, tau, n) in enumerate(selected, 1):
        print(f"{i:4d}  {qid:12d}  {tau:+.4f}  {n:4d}")

    # Extract candidate pairs from retrieval results
    total_pairs = 0
    with open(OUTPUT_PATH, "w") as f:
        for qid, _tau, _n in selected:
            candidates = retrieval.get(qid, [])
            for cid in candidates:
                rec = {"paper_a": qid, "paper_b": cid, "source": "retrieved"}
                f.write(json.dumps(rec) + "\n")
                total_pairs += 1

    print("\n=== OUTPUT ===")
    print(f"  Wrote {total_pairs} pairs for {len(selected)} queries")
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
