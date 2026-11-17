"""Backfill abstracts into paper_graphs.jsonl from the Semantic Scholar API.

One-time migration script for datasets created before the parse command started
capturing abstracts from metadata. Fetches abstracts via the S2 batch API and
merges them into the existing paper_graphs.jsonl file.

Usage:
    uv run scripts/backfill_abstracts.py <splits_dir> [--limit N] [--batch-size N]
"""

import argparse
import asyncio
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

from genret.log_config import setup_logging
from genret.s2.api import SemanticScholarAPI
from genret.utils import (
    AsyncLineWriter,
    iter_jsonl,
    open_auto,
    require_env,
    resolve_jsonl,
    track_progress,
)

logger = logging.getLogger("genret.scripts.backfill_abstracts")

BATCH_SIZE = 500


async def fetch_abstracts_batch(
    api: SemanticScholarAPI,
    corpus_ids: list[int],
) -> dict[int, str]:
    """Fetch abstracts for a batch of corpus IDs (max 500)."""
    s2_ids = [f"CorpusId:{cid}" for cid in corpus_ids]
    papers = await api.get_paper_batch(s2_ids, fields=["abstract", "externalIds"])

    return {
        cid: paper.abstract
        for cid, paper in zip(corpus_ids, papers)
        if paper is not None and paper.abstract
    }


def fetch_abstracts(
    graphs_path: Path,
    abstracts_path: Path,
    *,
    limit: int = 0,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Fetch abstracts from S2 API and write to a sidecar JSONL file.

    Resumes from existing abstracts.jsonl if present.

    Returns:
        Number of new abstracts written.
    """
    api_key = require_env("SEMANTIC_SCHOLAR_API_KEY")

    corpus_ids: list[int] = [r["corpusid"] for r in iter_jsonl(graphs_path)]
    logger.info(f"Found {len(corpus_ids)} papers in paper_graphs")

    existing_ids: set[int] = set()
    if abstracts_path.exists():
        for record in iter_jsonl(abstracts_path):
            existing_ids.add(record["corpusid"])
        logger.info(f"Already have abstracts for {len(existing_ids)} papers")

    remaining = [cid for cid in corpus_ids if cid not in existing_ids]
    if limit > 0:
        remaining = remaining[:limit]
    if not remaining:
        logger.info("All abstracts already fetched")
        return 0

    logger.info(f"Fetching abstracts for {len(remaining)} papers")
    batches = list(itertools.batched(remaining, batch_size, strict=False))

    async def run() -> int:
        found = 0
        async with (
            SemanticScholarAPI(api_key=api_key) as api,
            AsyncLineWriter.open(abstracts_path, "a") as writer,
        ):
            for batch in track_progress(batches, "Fetching abstracts..."):
                try:
                    results = await fetch_abstracts_batch(api, list(batch))
                except Exception:
                    logger.exception(f"Failed batch, skipping {len(batch)} papers")
                    continue

                for cid, abstract in results.items():
                    await writer.write_line(
                        json.dumps({"corpusid": cid, "abstract": abstract})
                    )
                found += len(results)

        return found

    total_found = asyncio.run(run())
    logger.info(f"Wrote {total_found} abstracts to {abstracts_path}")
    return total_found


def backfill_paper_graphs(
    graphs_path: Path,
    abstracts_path: Path,
) -> int:
    """Merge abstracts sidecar into paper_graphs.jsonl.

    Returns:
        Number of papers updated.
    """
    abstracts: dict[int, str] = {}
    for record in iter_jsonl(abstracts_path):
        abstracts[record["corpusid"]] = record["abstract"]
    logger.info(f"Loaded {len(abstracts)} abstracts")

    suffix = "".join(graphs_path.suffixes)
    tmp_path = graphs_path.parent / f"paper_graphs.tmp{suffix}"

    updated = 0
    total = 0
    with (
        open_auto(graphs_path, "r") as fin,
        open_auto(tmp_path, "w") as fout,
    ):
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue

            record: dict[str, Any] = json.loads(stripped)
            total += 1
            cid = record["corpusid"]
            if cid in abstracts and not record.get("abstract"):
                record["abstract"] = abstracts[cid]
                updated += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp_path.rename(graphs_path)
    logger.info(f"Updated {updated}/{total} papers with abstracts")
    return updated


def main() -> None:
    """Backfill abstracts into paper_graphs from S2 API."""
    parser = argparse.ArgumentParser(description="Backfill abstracts into paper_graphs")
    parser.add_argument("splits_dir", type=Path, help="Splits directory")
    parser.add_argument("--limit", "-n", type=int, default=0, help="Max papers (0=all)")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    setup_logging()

    processed_dir: Path = args.splits_dir / "processed"
    graphs_path = resolve_jsonl(processed_dir / "paper_graphs.jsonl.gz")
    if not graphs_path.exists():
        logger.error(f"{graphs_path} not found")
        sys.exit(1)

    abstracts_path = processed_dir / "abstracts.jsonl"

    # Step 1: fetch
    fetch_abstracts(
        graphs_path,
        abstracts_path,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    # Step 2: merge
    if not abstracts_path.exists():
        logger.info("No abstracts to merge")
        return

    backfill_paper_graphs(graphs_path, abstracts_path)


if __name__ == "__main__":
    main()
