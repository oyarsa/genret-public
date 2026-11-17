r"""Cap each paper_b's prevalence as a gold neighbour at ``cap`` train queries.

Diagnostic experiment: produce a rebalanced training pairs file where no
single paper appears as a positive for more than ``cap`` train queries. The
goal is not to improve the method but to remove the popularity attractor
from the training distribution — if GLEN trained on the rebalanced data
performs substantially better than on the original, popularity skew is
structurally the obstacle.

The corpus and test split are kept unchanged. Indexing rows (kind="indexing")
pass through untouched. Only retrieval rows are filtered.

Usage::

    uv run python scripts/build_balanced_pairs.py \\
        --pairs output/genret/codebook_training_pairs_1k.jsonl.zst \\
        --train-split data/s2orc-nlp/splits-connected-1k/train.txt \\
        --output output/genret/codebook_training_pairs_1k_balanced5pct.jsonl.zst \\
        --cap-pct 5 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from genret.utils import iter_jsonl, open_auto

logger = logging.getLogger(__name__)


def _load_train_ids(path: Path) -> set[int]:
    """Return the set of corpus IDs listed in the train split file."""
    return {int(line.strip()) for line in path.read_text().splitlines() if line.strip()}


def _rebalance(
    pairs_path: Path,
    output_path: Path,
    *,
    cap: int,
    train_ids: set[int],
    rng: random.Random,
) -> tuple[int, int]:
    """Rewrite ``pairs_path`` to ``output_path`` with capped paper_b prevalence.

    The cap applies per ``paper_b``: at most ``cap`` distinct train-set
    ``paper_a`` values are kept as gold neighbours of that paper.

    Returns ``(rows_in, rows_out)``.
    """
    rows: list[dict[str, object]] = list(iter_jsonl(pairs_path))
    rows_in = len(rows)

    # Index: paper_b -> set of train paper_a values with (paper_a, paper_b, _)
    # retrieval links.
    paper_b_to_train_a: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        if str(row.get("kind", "")) != "retrieval":
            continue
        paper_a = int(row["paper_a"])  # type: ignore[arg-type]
        paper_b = int(row["paper_b"])  # type: ignore[arg-type]
        if paper_a in train_ids:
            paper_b_to_train_a[paper_b].add(paper_a)

    # For each paper_b with more than ``cap`` train-query neighbours, randomly
    # sample ``cap`` of them; for others, keep all.
    kept_pairs: dict[int, set[int]] = {}
    for paper_b, paper_a_set in paper_b_to_train_a.items():
        if len(paper_a_set) <= cap:
            kept_pairs[paper_b] = paper_a_set
            continue
        sampled = rng.sample(sorted(paper_a_set), cap)
        kept_pairs[paper_b] = set(sampled)

    # Filter rows: indexing rows pass through; retrieval rows kept only if
    # (paper_a, paper_b) was retained AND paper_a is in train_ids (rows for
    # dev/test queries also pass through since they aren't subject to the cap).
    kept_rows: list[dict[str, object]] = []
    for row in rows:
        kind = str(row.get("kind", ""))
        if kind != "retrieval":
            kept_rows.append(row)
            continue
        paper_a = int(row["paper_a"])  # type: ignore[arg-type]
        paper_b = int(row["paper_b"])  # type: ignore[arg-type]
        if paper_a not in train_ids:
            # Out-of-train paper_a — leave the row as-is.
            kept_rows.append(row)
            continue
        if paper_b in kept_pairs and paper_a in kept_pairs[paper_b]:
            kept_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_auto(output_path, "w") as fh:
        for row in kept_rows:
            fh.write(json.dumps(row) + "\n")

    return rows_in, len(kept_rows)


def _summarise_popularity(pairs_path: Path, train_ids: set[int], label: str) -> None:
    """Log a quick popularity histogram for retrieval rows in ``pairs_path``."""
    counts: Counter[int] = Counter()
    n_rows = 0
    for row in iter_jsonl(pairs_path):
        if str(row.get("kind", "")) != "retrieval":
            continue
        n_rows += 1
        paper_a = int(row["paper_a"])  # type: ignore[arg-type]
        paper_b = int(row["paper_b"])  # type: ignore[arg-type]
        if paper_a in train_ids:
            counts[paper_b] += 1
    n_train_queries = len(train_ids)
    logger.info("%s: %d retrieval rows", label, n_rows)
    logger.info(
        "%s: top-5 paper_b prevalence (out of %d train queries):",
        label,
        n_train_queries,
    )
    for paper_b, count in counts.most_common(5):
        pct = count / n_train_queries * 100.0
        logger.info(
            "  paper_b=%d: %d rows (%.1f%% of train queries)",
            paper_b,
            count,
            pct,
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="Source training pairs JSONL[.zst] file.",
    )
    parser.add_argument(
        "--train-split",
        type=Path,
        required=True,
        help="Train split file (one corpus ID per line). Defines the "
        "denominator for the prevalence cap.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL[.zst] path for the rebalanced pairs.",
    )
    parser.add_argument(
        "--cap-pct",
        type=float,
        default=5.0,
        help="Maximum percent of train queries any single paper may appear "
        "as gold neighbour for (default: 5%%).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING")
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    train_ids = _load_train_ids(args.train_split)
    cap = round(len(train_ids) * args.cap_pct / 100.0)
    logger.info(
        "Train split: %d queries; cap = %.1f%% = %d max gold links per paper",
        len(train_ids),
        args.cap_pct,
        cap,
    )

    _summarise_popularity(args.pairs, train_ids, label="BEFORE")

    rng = random.Random(args.seed)
    rows_in, rows_out = _rebalance(
        args.pairs,
        args.output,
        cap=cap,
        train_ids=train_ids,
        rng=rng,
    )
    logger.info(
        "Wrote %d rows (from %d input rows) to %s",
        rows_out,
        rows_in,
        args.output,
    )

    _summarise_popularity(args.output, train_ids, label="AFTER")


if __name__ == "__main__":
    main()
