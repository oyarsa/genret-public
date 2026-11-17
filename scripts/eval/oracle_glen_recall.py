# ruff: noqa: PLC0415
"""Oracle recall and collision analysis for GLEN-style DocID retrieval.

Assumes a perfect generative model: at decode time the model generates the
exact DocID token sequence for every gold-related paper in the corpus. The
recall computed here is an upper bound on what any trained GLEN retriever
can achieve.

Also reports DocID collision statistics:
- leaf size distribution (papers per DocID)
- fraction of DocIDs that are unique (leaf size == 1)
- fraction of gold papers with unique DocIDs

Run locally (CPU only, ~5 min) before committing to GLEN training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from genret.training.glen_docids import (
    load_glen_index,
)
from genret.utils import iter_jsonl

FACET_NAMES = ("problems", "methods", "results", "contributions")


def _load_labels(label_files: list[Path]) -> dict[int, set[int]]:
    positives: dict[int, set[int]] = defaultdict(set)
    for path in label_files:
        if not path.exists():
            continue
        for r in iter_jsonl(path):
            if not r.get("related"):
                continue
            positives[int(r["paper_a"])].add(int(r["paper_b"]))
    return positives


def collision_stats(index_path: Path) -> None:
    """Print leaf-size distribution for the GLEN index."""

    index = load_glen_index(index_path)

    print("\n=== Collision statistics ===")
    for facet in FACET_NAMES:
        leaves = index.code_to_papers[facet]
        sizes = [len(v) for v in leaves.values()]
        total_papers = sum(sizes)
        n_unique = sum(1 for s in sizes if s == 1)
        n_gt5 = sum(1 for s in sizes if s > 5)
        n_gt10 = sum(1 for s in sizes if s > 10)
        counter = Counter(sizes)
        max_size = max(sizes) if sizes else 0
        print(
            f"\n  {facet}: {len(leaves)} leaves, {total_papers} papers"
            f"\n    unique (size=1):  {n_unique:5d}  ({100 * n_unique / max(len(leaves), 1):.1f}%)"
            f"\n    ambiguous (>5):  {n_gt5:5d}  ({100 * n_gt5 / max(len(leaves), 1):.1f}%)"
            f"\n    very ambig (>10):{n_gt10:5d}  ({100 * n_gt10 / max(len(leaves), 1):.1f}%)"
            f"\n    max leaf size:   {max_size}"
            f"\n    avg leaf size:   {total_papers / max(len(leaves), 1):.2f}"
        )
        # Top-5 most crowded leaves
        crowded = sorted(counter.items(), key=lambda x: -x[0])[:5]
        print(f"    leaf size dist (top 5): {crowded}")


def oracle_recall(
    index_path: Path,
    queries: list[int],
    positives: dict[int, set[int]],
    k_values: list[int],
) -> None:
    """Compute oracle recall assuming perfect DocID generation."""

    index = load_glen_index(index_path)

    # Build reverse map: corpusid -> {facet -> [token_sequences]}
    # A paper may have multiple DocIDs per facet (one per bullet).
    cid_to_docids: dict[int, dict[str, list[tuple[int, ...]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for facet in FACET_NAMES:
        for seq, papers in index.code_to_papers[facet].items():
            for cid in papers:
                cid_to_docids[int(cid)][facet].append(seq)

    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    n_queries_with_gold = 0
    gold_unique_rates: list[float] = []

    for qid in queries:
        gold = positives.get(qid, set())
        if not gold:
            continue
        n_queries_with_gold += 1

        # For each gold paper, collect all papers that share its DocID
        # across any facet (union of leaves). This is what a perfect model
        # would retrieve.
        retrieved: dict[int, float] = {}  # cid -> best RRF score
        for gold_cid in gold:
            docids = cid_to_docids.get(int(gold_cid), {})
            for facet, seqs in docids.items():
                for seq in seqs:
                    leaf_papers = index.code_to_papers[facet].get(seq, [])
                    for rank, pid in enumerate(leaf_papers):
                        pid_int = int(pid)
                        score = 1.0 / (60 + rank + 1)
                        retrieved[pid_int] = retrieved.get(pid_int, 0.0) + score

        ranked = sorted(retrieved, key=lambda x: -retrieved[x])
        ranked = [r for r in ranked if r != qid]

        # Fraction of gold papers with unique DocIDs per facet
        unique_fracs: list[float] = []
        for gold_cid in gold:
            docids = cid_to_docids.get(int(gold_cid), {})
            for facet, seqs in docids.items():
                for seq in seqs:
                    leaf_size = len(index.code_to_papers[facet].get(seq, []))
                    unique_fracs.append(1.0 if leaf_size == 1 else 0.0)
        if unique_fracs:
            gold_unique_rates.append(sum(unique_fracs) / len(unique_fracs))

        for k in k_values:
            top = set(ranked[:k])
            hits = len(gold & top)
            recalls[k].append(hits / len(gold))

    print("\n=== Oracle recall (perfect generation assumed) ===")
    print(f"Queries with gold positives: {n_queries_with_gold}/{len(queries)}")
    if gold_unique_rates:
        print(
            f"Avg fraction of gold papers with unique DocIDs: "
            f"{sum(gold_unique_rates) / len(gold_unique_rates):.3f}"
        )
    print(f"\n{'k':>6}  {'recall':>8}")
    for k in k_values:
        vals = recalls[k]
        r = sum(vals) / max(len(vals), 1)
        print(f"{k:>6}  {r:>8.3f}")


def main() -> None:
    """Entry point for oracle recall and collision analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--facets",
        default="data/s2orc-nlp-v2/splits-connected-10k-v2/processed/facets.jsonl.zst",
        type=Path,
    )
    parser.add_argument(
        "--glen-dir",
        default="output/genret/glen_index",
        type=Path,
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--max-docid-tokens",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--queries",
        default="output/eval_500/query_papers_500.txt",
        type=Path,
    )
    parser.add_argument(
        "--label-files",
        nargs="+",
        default=[
            "output/eval_500/gpt5mini_judgements.jsonl",
            "output/ablations/gpt5mini_missing_judgements.jsonl",
        ],
    )
    parser.add_argument(
        "--k-top",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100],
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the GLEN index even if it already exists.",
    )
    args = parser.parse_args()

    glen_dir = args.glen_dir
    if args.rebuild or not (glen_dir / "code_to_papers.json").exists():
        print(f"Building GLEN index from {args.facets} ...")
        from genret._shed.transformers import load_tokenizer
        from genret.training.glen_docids import build_glen_index, save_glen_index

        tokeniser = load_tokenizer(args.model)
        index = build_glen_index(
            Path(args.facets),
            tokeniser,
            max_tokens=args.max_docid_tokens,
        )
        save_glen_index(index, glen_dir)
        print(f"Index saved to {glen_dir}")
    else:
        print(f"Loading existing GLEN index from {glen_dir}")

    collision_stats(glen_dir)

    queries = [int(x) for x in Path(args.queries).read_text().split() if x.strip()]
    label_paths = [Path(p) for p in args.label_files]
    positives = _load_labels(label_paths)

    oracle_recall(glen_dir, queries, positives, args.k_top)


if __name__ == "__main__":
    main()
