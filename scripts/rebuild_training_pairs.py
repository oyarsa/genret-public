"""Rebuild all training pairs with consistent format_paper_pair() formatting.

Reads raw scored pairs (with corpus IDs) from the original Qwen3-32B judge
outputs and the diverse-pair shards, formats each with the current
format_paper_pair() (titles + all facets), and writes a single merged file.

Every output record includes paper_a and paper_b corpus IDs so we never
lose provenance again.
"""

import json
import random
from collections.abc import Callable
from pathlib import Path

from genret.paper_parsing import Facets, load_facets_map
from genret.processing.reranker import (
    FACET_LABELS,
    facet_items,
    format_paper_pair,
    load_titles,
)
from genret.types import CorpusId

FACETS_PATH = Path("data/s2orc-nlp/splits-connected-10k/processed/facets.jsonl.zst")
TITLES_PATH = Path(
    "data/s2orc-nlp/splits-connected-10k/processed/paper_graphs.jsonl.zst"
)
FACETS_NO_RESULTS = ("problems", "methods", "contributions")

OUTPUTS: dict[str, Path] = {
    "55k-v2": Path("output/distilled_reranker_55k-v2/training_pairs.jsonl"),
    "55k-v2-no-results": Path(
        "output/distilled_reranker_55k-v2-no-results/training_pairs.jsonl"
    ),
    "150k": Path("output/distilled_reranker_round1/training_pairs.jsonl"),
    "150k-no-results": Path(
        "output/distilled_reranker_round1-no-results/training_pairs.jsonl"
    ),
}

# Original 55k sources (Qwen3-32B scored files with paper_a/paper_b/score)
ORIGINAL_SOURCES: list[Path] = [
    Path("output/llm_judge/scores_qwen3-32b-binary-sequence-gpt-pairs-v2.jsonl"),
    Path("output/llm_judge/qwen32b_binary_subset2000_top2_no_results.jsonl"),
    *sorted(Path("output/online_eval/local_judge_shards").glob("*.jsonl")),
    *sorted(Path("output/weight_sweep/local_judge_shards").glob("*.jsonl")),
    *sorted(Path("output/eval_500/local_judge_shards").glob("*.jsonl")),
]

# Diverse 95k sources (round 1 scoring shards)
DIVERSE_SHARD_PATTERN = "output/diverse_training_pairs/scores_shard_*.jsonl"

SEED = 0


def _format_paper_no_results(title: str, facets: Facets) -> str:
    lines = [title]
    for name in FACETS_NO_RESULTS:
        if items := facet_items(facets, name):
            lines.append(f"{FACET_LABELS[name]}: {'; '.join(items)}")
    return "\n".join(lines)


def format_paper_pair_no_results(
    title_a: str, facets_a: Facets, title_b: str, facets_b: Facets
) -> str:
    """Format a paper pair for training without search results."""
    block_a = _format_paper_no_results(title_a, facets_a)
    block_b = _format_paper_no_results(title_b, facets_b)
    return f"Paper A: {block_a}\n\nPaper B: {block_b}"


def _collect_pairs(
    source_files: list[Path],
) -> list[tuple[int, int, float]]:
    """Load (paper_a, paper_b, score) from scored JSONL files.

    Keeps order as-is — (a,b) and (b,a) are treated as distinct because
    the cross-encoder prompt uses positional labels (Paper A / Paper B)
    and the judge may assign different scores depending on order.

    Deduplicates exact (paper_a, paper_b) keys, keeping the first seen.
    """
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int, float]] = []
    for path in source_files:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["paper_a"], rec["paper_b"])
                if key not in seen:
                    seen.add(key)
                    pairs.append((key[0], key[1], float(rec["score"])))
    return pairs


def _format_all(
    pairs_list: list[tuple[int, int, float]],
    source_name: str,
    facets_map: dict[CorpusId, Facets],
    titles: dict[CorpusId, str],
    formatter: Callable[..., str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    skipped = 0
    for cid_a, cid_b, score in pairs_list:
        facets_a = facets_map.get(CorpusId(cid_a))
        facets_b = facets_map.get(CorpusId(cid_b))
        title_a = titles.get(CorpusId(cid_a))
        title_b = titles.get(CorpusId(cid_b))

        if facets_a is None or facets_b is None or title_a is None or title_b is None:
            skipped += 1
            continue

        text = formatter(title_a, facets_a, title_b, facets_b)
        records.append(
            {
                "paper_a": cid_a,
                "paper_b": cid_b,
                "text": text,
                "label": score,
                "source": source_name,
            }
        )
    if skipped:
        print(f"  ({skipped} skipped for missing facets/titles)")
    return records


def _write_shuffled(records: list[dict[str, object]], path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    shuffled = list(records)
    random.shuffle(shuffled)
    with open(path, "w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in shuffled)
    print(f"  {label}: {len(shuffled):,} pairs -> {path}")


def main() -> None:
    """Rebuild training pairs from facets and original sources."""
    facets_map = load_facets_map(FACETS_PATH)
    titles = load_titles(TITLES_PATH)

    # --- Collect raw pairs ---
    original_pairs = _collect_pairs(ORIGINAL_SOURCES)
    print(f"Original pairs: {len(original_pairs):,}")

    shard_files = sorted(Path().glob(DIVERSE_SHARD_PATTERN))
    print(f"Found {len(shard_files)} diverse shard files")
    diverse_pairs = _collect_pairs(shard_files)
    print(f"Diverse pairs: {len(diverse_pairs):,}")

    # --- Format with both variants ---
    for variant_label, formatter in [
        ("all facets", format_paper_pair),
        ("no results", format_paper_pair_no_results),
    ]:
        print(f"\nFormatting: {variant_label}")
        orig_recs = _format_all(
            original_pairs, "llm_judge", facets_map, titles, formatter
        )
        diverse_recs = _format_all(
            diverse_pairs, "llm_judge_diverse", facets_map, titles, formatter
        )
        merged_recs = orig_recs + diverse_recs

        print(f"  llm_judge: {len(orig_recs):,}, diverse: {len(diverse_recs):,}")

        if variant_label == "all facets":
            _write_shuffled(orig_recs, OUTPUTS["55k-v2"], "55k-v2")
            _write_shuffled(merged_recs, OUTPUTS["150k"], "150k")
        else:
            _write_shuffled(
                orig_recs, OUTPUTS["55k-v2-no-results"], "55k-v2-no-results"
            )
            _write_shuffled(merged_recs, OUTPUTS["150k-no-results"], "150k-no-results")


if __name__ == "__main__":
    main()
