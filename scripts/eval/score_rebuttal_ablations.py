"""Paired-bootstrap significance for the ablation claims questioned in the rebuttal.

Two analyses on the 1k corpus, 158 in-corpus test queries:

1. Supervision ablation (Table 1 lower block) and the K_min coverage-floor sweep,
   all fused with graph-RRF as in the paper. Each comparison is a paired
   bootstrap on per-query R@20 and R@50, on strict gold and citation gold.
2. GRAFT with plain (uniform) RRF, i.e. no graph at inference, against the
   non-graph baselines, raw and with the popularity head removed.

Reuses the loaders, recall and bootstrap helpers of `score_colbert_rebuttal.py`.

Usage:
    uv run python scripts/eval/score_rebuttal_ablations.py
"""

import argparse
import importlib.util
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType


def _load_sibling(name: str) -> ModuleType:
    """Import a sibling script in scripts/eval as a module."""
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).parent / f"{name}.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scr = _load_sibling("score_colbert_rebuttal")
"""score_colbert_rebuttal.py, for recall, bootstrap and head helpers."""
sge = scr.sge
"""score_glen_eval.py, for gold loading and graph-RRF fusion."""

EVAL_DIR = Path("output/genret/eval")
"""Directory with the stored GRAFT decodes."""
SUPERVISION_RUNS = {
    "Uniform sampling": "glen-bullets-1k-1b-A-allfacets_per_bullet.jsonl",
    "+ oversampling": "glen-bullets-1k-1b-78lp-graphsampled_per_bullet_baseline.jsonl",
    "K_min=1": "glen-bullets-1k-1b-78lp-optionB-kmin1_per_bullet.jsonl",
    "K_min=2": "glen-bullets-1k-1b-78lp-optionB-kmin2_per_bullet.jsonl",
    "K_min=3": "glen-bullets-1k-1b-78lp-optionB_per_bullet.jsonl",
    "K_min=4 (ckpt-34000)": "glen-bullets-1k-1b-78lp-optionB-kmin4_ckpt34000_per_bullet.jsonl",
    "+ fallback (GRAFT)": "glen-graphfallback_per_bullet.jsonl",
}
"""Stored no-title decodes behind Table 1's lower block and the K_min sweep."""
SUPERVISION_COMPARISONS = [
    ("+ oversampling", "Uniform sampling"),
    ("K_min=3", "+ oversampling"),
    ("+ fallback (GRAFT)", "K_min=3"),
    ("+ fallback (GRAFT)", "Uniform sampling"),
    ("K_min=3", "K_min=1"),
    ("K_min=3", "K_min=2"),
    ("K_min=3", "K_min=4 (ckpt-34000)"),
]
"""(system, baseline) pairs; the reported delta is system - baseline."""
GOLDS = {
    "strict gold": Path("output/genret/codebook_training_pairs_1k.jsonl.zst"),
    "direct-citation gold": Path("output/ooc_eval_1k/gold_incorpus_cite_pairs.jsonl"),
}
"""In-corpus gold files."""
TEST_QUERIES = Path("data/s2orc-nlp/splits-connected-1k/test.txt")
"""In-corpus test split."""


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean."""
    return sum(values) / len(values)


def compare(
    systems: Mapping[str, dict[int, list[int]]],
    comparisons: Sequence[tuple[str, str]],
    gold: Mapping[int, set[int]],
    removed: frozenset[int],
) -> None:
    """Print one markdown table of paired-bootstrap deltas at R@20 and R@50."""
    queries = sorted(
        q for q in scr.read_ids(TEST_QUERIES) if gold.get(q, set()) - removed
    )
    print(f"(n={len(queries)})\n")
    print("| System | Baseline | R@20 | ΔR@20 [95% CI] | R@50 | ΔR@50 [95% CI] |")
    print("|---|---|---|---|---|---|")
    for system, baseline in comparisons:
        cells: list[str] = []
        for k in (20, 50):
            a = scr.per_query_recall(systems[system], gold, queries, k, removed)
            b = scr.per_query_recall(systems[baseline], gold, queries, k, removed)
            d, lo, hi = scr.paired_bootstrap(a, b, random.Random(0))
            sig = " *" if lo > 0 or hi < 0 else ""
            cells += [
                f"{mean(a):.3f} vs {mean(b):.3f}",
                f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}",
            ]
        print(f"| {system} | {baseline} | " + " | ".join(cells) + " |")


def main() -> None:
    """Run both analyses and print markdown tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-dir", type=Path, default=Path("output/genret/facet_candidates_1k")
    )
    parser.add_argument(
        "--corpus-ids",
        type=Path,
        default=Path("data/s2orc-nlp/splits-connected-1k/corpus_ids.txt"),
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("data/s2orc-nlp/citations/references.jsonl"),
    )
    args = parser.parse_args()

    weights = sge.load_facet_weights(args.candidates_dir)
    head = scr.popularity_head(args.references, scr.read_ids(args.corpus_ids), 20)
    golds = {name: sge.load_strict_gold(path) for name, path in GOLDS.items()}

    supervision = {
        name: sge.load_eval_graph_rrf(EVAL_DIR / file, weights)
        for name, file in SUPERVISION_RUNS.items()
    }
    for gold_name, gold in golds.items():
        print(f"\n## Supervision ablation and K_min sweep, graph-RRF, {gold_name} ")
        compare(supervision, SUPERVISION_COMPARISONS, gold, frozenset())

    base = Path("output/eval_1k")
    no_graph = {
        "GRAFT (uniform RRF)": scr.load_ranked(
            EVAL_DIR / SUPERVISION_RUNS["+ fallback (GRAFT)"]
        ),
        "MiniLM (title+abstract)": scr.load_ranked(base / "retrieval_minilm.jsonl"),
        "BM25 (title+facets)": scr.load_ranked(base / "retrieval_bm25.jsonl"),
        "SPECTER2 (title+abstract)": scr.load_ranked(base / "retrieval_specter2.jsonl"),
        "Dense-only (facet)": scr.load_ranked(base / "retrieval_denseonly.jsonl"),
    }
    no_graph_comparisons = [
        ("GRAFT (uniform RRF)", name) for name in no_graph if "GRAFT" not in name
    ]
    for gold_name, gold in golds.items():
        for label, removed in (("raw", frozenset[int]()), ("head-removed", head)):
            print(f"\n## GRAFT without graph at inference, {gold_name}, {label} ")
            compare(no_graph, no_graph_comparisons, gold, removed)


if __name__ == "__main__":
    main()
