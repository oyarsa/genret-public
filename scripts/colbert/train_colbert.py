"""Fine-tune ColBERT on GRAFT's supervision as paper-level pairs.

Reads the pair counts written by `scripts/colbert/export_pairs.py` and trains
on (query title+abstract, target title+abstract) pairs in one of two variants:

- `replicated`: each pair repeated `count` times, reproducing the exposure
  distribution GRAFT saw (edge-weighted oversampling, coverage floor).
- `deduplicated`: each unique pair once.

Both variants run for the same number of optimiser steps; checkpoints are then
selected on the dev split with `scripts/eval/colbert_baseline.py`. PyLate is
not a project dependency, so run in an isolated environment:

    uv run --no-project --python 3.14 --with-editable . --with pylate
      python scripts/colbert/train_colbert.py ...
"""

import argparse
import logging
from pathlib import Path

import torch

from genret._shed.pylate import load_colbert, train_contrastive
from genret.utils import iter_jsonl

logger = logging.getLogger(__name__)


def load_texts(paper_graphs_path: Path) -> dict[int, str]:
    """Load "title abstract" text per paper, as the retrieval baselines do."""
    texts: dict[int, str] = {}
    for record in iter_jsonl(paper_graphs_path):
        title: str = record.get("title") or ""
        abstract: str = record.get("abstract") or ""
        texts[int(record["corpusid"])] = f"{title} {abstract}".strip()
    return texts


def build_pairs(
    pairs_path: Path, texts: dict[int, str], *, replicated: bool
) -> tuple[list[str], list[str]]:
    """Expand pair counts into aligned query and positive text lists.

    Args:
        pairs_path: JSONL rows with `query`, `positive` and `count`.
        texts: Title+abstract text per corpus ID.
        replicated: Repeat each pair `count` times instead of once.

    Returns:
        Aligned (queries, positives) lists.
    """
    queries: list[str] = []
    positives: list[str] = []
    for row in iter_jsonl(pairs_path):
        copies = int(row["count"]) if replicated else 1
        queries.extend([texts[int(row["query"])]] * copies)
        positives.extend([texts[int(row["positive"])]] * copies)
    return queries, positives


def main() -> None:
    """Build the training pairs for one variant and fine-tune ColBERT."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--paper-graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=["replicated", "deduplicated"], required=True
    )
    parser.add_argument("--model", default="lightonai/colbertv2.0")
    parser.add_argument("--query-length", type=int, default=300)
    parser.add_argument("--document-length", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    texts = load_texts(args.paper_graphs)
    queries, positives = build_pairs(
        args.pairs, texts, replicated=args.variant == "replicated"
    )
    logger.info("Variant %s: %d training rows", args.variant, len(queries))

    model = load_colbert(
        args.model,
        query_length=args.query_length,
        document_length=args.document_length,
    )
    train_contrastive(
        model,
        queries,
        positives,
        output_dir=args.output,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_steps=args.save_steps,
        bf16=torch.cuda.is_available(),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
