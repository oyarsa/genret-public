"""Phase 1: Build corpus ID index and extract paper metadata."""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
import httpx

from genret.config import normalise_venue
from genret.pipeline.common import (
    API_TIMEOUT,
    Stats,
    append_ids_to_file,
    download_and_process,
    get_dataset_info,
    get_remaining_files,
    log_dry_run_files,
    output_name,
    run_parallel_file_phase,
)
from genret.types import CorpusId
from genret.utils import (
    ProgressManager,
    append_processed_file,
    backup_files,
    iter_jsonl,
    load_ids_from_file,
    load_processed_files,
    open_auto,
)

logger = logging.getLogger(__name__)

# === Data structures ===

type VenueMatcher = Callable[[str], bool]


@dataclass
class IndexResult:
    """Result of processing a papers file for indexing.

    Attributes:
        corpus_ids: Set of corpus IDs matched in this file.
        record_count: Total number of matched records written.
        acl_only: Papers matched by ACL Anthology ID only.
        venue_only: Papers matched by venue pattern only.
        both: Papers matched by both ACL ID and venue pattern.
    """

    corpus_ids: set[CorpusId] = field(default_factory=set[CorpusId])
    record_count: int = 0
    # Match breakdown
    acl_only: int = 0
    venue_only: int = 0
    both: int = 0


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Result of paper matching, tracking which criteria matched.

    Attributes:
        matched: Whether the paper passed at least one matching criterion.
        has_acl_id: Whether the paper has an ACL Anthology ID.
        has_venue_match: Whether the paper's venue matched a pattern.
    """

    matched: bool
    has_acl_id: bool = False
    has_venue_match: bool = False


# === Index logic ===


def make_venue_matcher(patterns: list[str]) -> VenueMatcher:
    """Create a function that matches venue strings against patterns.

    Args:
        patterns: List of regex patterns to compile (case-insensitive).

    Returns:
        A callable that returns True if a venue string matches any pattern.
    """
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    def matcher(venue: str) -> bool:
        return any(p.search(venue) for p in compiled)

    return matcher


def paper_matches(
    record: dict[str, Any],
    year_range: tuple[int, int],
    venue_matcher: VenueMatcher,
) -> MatchResult:
    """Check if a paper matches: has ACL ID OR matches venue pattern, within year range.

    Args:
        record: Raw paper record dict from the S2ORC dataset.
        year_range: Inclusive (start_year, end_year) publication year filter.
        venue_matcher: Callable that tests a venue string against patterns.

    Returns:
        MatchResult indicating whether and how the paper matched.
    """
    year = record.get("year")
    if not year or not (year_range[0] <= year <= year_range[1]):
        return MatchResult(False)

    external_ids: dict[str, Any] = record.get("externalids") or {}
    acl_id: str | None = external_ids.get("ACL")
    has_acl_id = acl_id is not None

    raw_venue: str = record.get("venue", "") or ""
    venue = normalise_venue(raw_venue, acl_id) if raw_venue else ""
    has_venue_match = bool(venue) and venue_matcher(venue)

    matched = has_acl_id or has_venue_match
    return MatchResult(matched, has_acl_id, has_venue_match)


def process_papers_file(
    path: Path,
    year_range: tuple[int, int],
    venue_matcher: VenueMatcher,
    output_path: Path,
) -> IndexResult:
    """Process a papers file: extract corpus IDs and write records to output file.

    Args:
        path: Path to the gzipped papers JSONL source file.
        year_range: Inclusive (start_year, end_year) publication year filter.
        venue_matcher: Callable that tests a venue string against patterns.
        output_path: Destination path for matching records (auto-compressed).

    Returns:
        IndexResult with matched corpus IDs and per-criterion match counts.
    """
    result = IndexResult()
    with (
        open_auto(path, "r") as f,
        open_auto(output_path, "w") as out,
    ):
        for record in iter_jsonl(f):
            match = paper_matches(record, year_range, venue_matcher)
            if match.matched:
                corpus_id = record.get("corpusid")
                if corpus_id:
                    result.corpus_ids.add(CorpusId(corpus_id))
                    result.record_count += 1
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    # Track match breakdown
                    if match.has_acl_id and match.has_venue_match:
                        result.both += 1
                    elif match.has_acl_id:
                        result.acl_only += 1
                    else:
                        result.venue_only += 1
    return result


def build_index_stats(
    release_id: str,
    year_range: tuple[int, int],
    venue_patterns: list[str],
    corpus_ids: set[CorpusId],
    stats: Stats,
    processed_files: set[str],
) -> dict[str, Any]:
    """Build the index statistics dictionary for saving to JSON.

    Args:
        release_id: The dataset release ID.
        year_range: (start_year, end_year) tuple.
        venue_patterns: List of venue regex patterns.
        corpus_ids: Set of matched corpus IDs.
        stats: Processing statistics from current run.
        processed_files: Set of already-processed filenames.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    return {
        "release_id": release_id,
        "year_range": list(year_range),
        "venue_patterns": venue_patterns,
        "stats": {
            "corpus_ids": len(corpus_ids),
            "files_processed": stats.files_processed + len(processed_files),
            "records_matched": stats.records_matched,
            "matched_acl_only": stats.matched_acl_only,
            "matched_venue_only": stats.matched_venue_only,
            "matched_both": stats.matched_both,
            "bytes_downloaded": stats.bytes_downloaded,
        },
    }


async def run_index(
    output_dir: Path,
    year_range: tuple[int, int],
    venue_patterns: list[str],
    limit_files: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    max_concurrent: int = 10,
) -> None:
    """Phase 1: Build corpus ID index and extract paper metadata.

    Downloads papers dataset files from Semantic Scholar, filters records by
    year range and venue patterns, and writes matching records to per-file JSONL
    outputs under papers/. Corpus IDs of matched papers are appended to
    corpus_ids.txt for use by later pipeline phases.

    Args:
        output_dir: Root directory for all pipeline outputs.
        year_range: Inclusive (start_year, end_year) publication year filter.
        venue_patterns: Regex patterns for venue name matching.
        limit_files: Process at most this many dataset files (useful for testing).
        dry_run: If True, report what would be done without downloading anything.
        force: If True, back up existing state and start fresh.
        max_concurrent: Maximum number of simultaneous file downloads.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "corpus_ids.txt"
    metadata_dir = output_dir / "papers"
    processed_path = output_dir / "processed_index.txt"
    stats_path = output_dir / "index_stats.json"

    venue_matcher = make_venue_matcher(venue_patterns)
    stats = Stats()

    if force:
        logger.info("Force mode: backing up previous state...")
        backup_files([index_path, processed_path, stats_path])
        # Note: metadata_dir files are not backed up, they're just overwritten

    processed_files = await load_processed_files(processed_path)
    corpus_ids: set[CorpusId]
    if processed_files:
        logger.info(f"Resuming: {len(processed_files)} files already processed")
        corpus_ids = load_ids_from_file(index_path)
        logger.info(f"  Loaded {len(corpus_ids):,} existing corpus IDs")
    else:
        corpus_ids = set()
        if index_path.exists():
            index_path.unlink()

    logger.info("Fetching papers dataset info...")
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client_info:
        dataset = await get_dataset_info(client_info, "papers")

    logger.info(f"  Release: {dataset.release_id}")
    logger.info(f"  Files: {len(dataset.files)}")

    remaining_files = get_remaining_files(dataset, processed_files, limit_files)

    if not remaining_files:
        logger.info("\nAll files already processed!")
        logger.info(f"Index contains {len(corpus_ids):,} corpus IDs")
        return

    if dry_run:
        log_dry_run_files(remaining_files)
        return

    metadata_dir.mkdir(parents=True, exist_ok=True)
    state_lock = asyncio.Lock()
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:

        async def _index_work(
            url: str,
            filename: str,
            progress: ProgressManager,
        ) -> None:
            metadata_output = metadata_dir / output_name(filename)
            result, bytes_dl = await download_and_process(
                url,
                filename,
                output_dir,
                "papers",
                client,
                progress,
                lambda p: process_papers_file(
                    p, year_range, venue_matcher, metadata_output
                ),
            )
            async with state_lock:
                corpus_ids.update(result.corpus_ids)
                stats.bytes_downloaded += bytes_dl
                stats.files_processed += 1
                stats.records_matched += result.record_count
                stats.matched_acl_only += result.acl_only
                stats.matched_venue_only += result.venue_only
                stats.matched_both += result.both
                await append_ids_to_file(index_path, result.corpus_ids)
                await append_processed_file(processed_path, filename)

        await run_parallel_file_phase(
            remaining_files,
            "Indexing papers",
            max_concurrent,
            _index_work,
        )

    index_stats = build_index_stats(
        dataset.release_id,
        year_range,
        venue_patterns,
        corpus_ids,
        stats,
        processed_files,
    )
    async with aiofiles.open(stats_path, "w") as f:
        await f.write(json.dumps(index_stats, indent=2))

    total_processed = stats.files_processed + len(processed_files)
    logger.info(
        f"\nIndex complete: {len(corpus_ids):,} corpus IDs\n"
        f"  Files processed this run: {stats.files_processed}\n"
        f"  Total files processed: {total_processed}\n"
        f"  Matches this run: {stats.records_matched:,}\n"
        f"    ACL ID only: {stats.matched_acl_only:,}\n"
        f"    Venue only: {stats.matched_venue_only:,}\n"
        f"    Both: {stats.matched_both:,}\n"
        f"  Index: {index_path}\n"
        f"  Metadata: {metadata_dir}"
    )
