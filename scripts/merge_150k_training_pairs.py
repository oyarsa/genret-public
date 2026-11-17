"""Regenerate the merged 150k training pairs from preserved Qwen3-32B shards.

Recovery of the production 150k reranker training set (152,287 pairs) following
the recipe in tmp/round1-diverse-training-pairs.md.

Inputs:
- output/distilled_reranker_55k-v2/training_pairs.jsonl (56,952 pairs, source=llm_judge)
- output/diverse_training_pairs/scores_shard_{1,2,3,4}.jsonl (95,335 pairs total)
- data/s2orc-nlp/splits-connected-10k/processed/facets.jsonl.zst
- data/s2orc-nlp/splits-connected-10k/processed/paper_graphs.jsonl.gz

Output:
- output/distilled_reranker_round1/training_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from genret.paper_parsing import load_facets_map
from genret.processing.reranker import format_paper_pair, load_titles
from genret.utils import iter_jsonl, open_auto

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--existing",
        type=Path,
        default=Path("output/distilled_reranker_55k-v2/training_pairs.jsonl"),
    )
    p.add_argument(
        "--shards-dir",
        type=Path,
        default=Path("output/diverse_training_pairs"),
    )
    p.add_argument(
        "--facets",
        type=Path,
        default=Path("data/s2orc-nlp/splits-connected-10k/processed/facets.jsonl.zst"),
    )
    p.add_argument(
        "--paper-graphs",
        type=Path,
        default=Path(
            "data/s2orc-nlp/splits-connected-10k/processed/paper_graphs.jsonl.gz"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output/distilled_reranker_round1/training_pairs.jsonl"),
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")

    logger.info("Loading facets from %s", args.facets)
    facets_map = load_facets_map(args.facets)
    logger.info("Loading titles from %s", args.paper_graphs)
    titles = load_titles(args.paper_graphs)

    rows: list[dict[str, object]] = []

    logger.info("Loading existing pairs from %s", args.existing)
    rows.extend(iter_jsonl(args.existing))
    logger.info("  loaded %d existing rows", len(rows))

    shard_paths = sorted(args.shards_dir.glob("scores_shard_*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"no scores_shard_*.jsonl under {args.shards_dir}")
    n_pre = len(rows)
    n_skipped = 0
    for shard in shard_paths:
        n_shard = 0
        for row in iter_jsonl(shard):
            pa = int(row["paper_a"])
            pb = int(row["paper_b"])
            if pa not in facets_map or pb not in facets_map:
                n_skipped += 1
                continue
            if pa not in titles or pb not in titles:
                n_skipped += 1
                continue
            text = format_paper_pair(
                titles[pa], facets_map[pa], titles[pb], facets_map[pb]
            )
            rows.append(
                {
                    "paper_a": pa,
                    "paper_b": pb,
                    "text": text,
                    "label": float(row["score"]),
                    "source": "llm_judge_diverse",
                }
            )
            n_shard += 1
        logger.info("  %s: +%d rows", shard.name, n_shard)
    logger.info(
        "Added %d shard rows (skipped %d for missing facets/titles)",
        len(rows) - n_pre,
        n_skipped,
    )

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_auto(args.output, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    logger.info("Wrote %d pairs to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
