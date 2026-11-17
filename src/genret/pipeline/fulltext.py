"""Phase 2: Extract full paper text from s2orc_v2 dataset."""

import asyncio
import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from genret.pipeline.common import (
    API_TIMEOUT,
    Stats,
    append_ids_to_file,
    download_and_process,
    get_dataset_info,
    get_remaining_files,
    log_dry_run_files,
    maybe_load_corpus_ids_for_dry_run,
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


@dataclass(frozen=True, kw_only=True)
class FulltextResult:
    """Result of processing an s2orc file for fulltext extraction.

    Attributes:
        matched_ids: Corpus IDs of papers found and written to the output file.
        match_count: Number of matching records written.
    """

    matched_ids: set[CorpusId]
    match_count: int


def process_s2orc_file(
    path: Path, corpus_ids: set[CorpusId], output_path: Path
) -> FulltextResult:
    """Process s2orc file and stream matching records directly to output file.

    Args:
        path: Path to the gzipped s2orc JSONL source file.
        corpus_ids: Set of corpus IDs to keep.
        output_path: Destination path for matching records (gzip compressed).

    Returns:
        FulltextResult with matched corpus IDs and the total match count.
    """
    matched_ids: set[CorpusId] = set()
    match_count = 0
    with (
        gzip.open(path, "rt", encoding="utf-8") as f,
        open_auto(output_path, "w") as out,
    ):
        for record in iter_jsonl(f):
            corpus_id = record.get("corpusid")
            if corpus_id and CorpusId(corpus_id) in corpus_ids:
                out.write(json.dumps(record) + "\n")
                matched_ids.add(CorpusId(corpus_id))
                match_count += 1
    return FulltextResult(matched_ids=matched_ids, match_count=match_count)


async def run_fulltext(
    output_dir: Path,
    limit_files: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    max_concurrent: int = 10,
) -> None:
    """Phase 2: Extract full paper text from s2orc_v2 dataset.

    Downloads s2orc files, filters to papers in the corpus ID index from
    phase 1, and writes matching records to gzip-compressed JSONL files
    under s2orc/.

    Args:
        output_dir: Root directory for all pipeline outputs.
        limit_files: Process at most this many dataset files (useful for testing).
        dry_run: If True, report what would be done without downloading anything.
        force: If True, back up existing state and start fresh.
        max_concurrent: Maximum number of simultaneous file downloads.
    """
    s2orc_dir = output_dir / "s2orc"
    processed_path = output_dir / "processed_fulltext.txt"
    matched_ids_path = output_dir / "matched_corpus_ids.txt"

    maybe_corpus_ids = maybe_load_corpus_ids_for_dry_run(
        output_dir, "fulltext phase", dry_run
    )
    if maybe_corpus_ids is None:
        return

    corpus_ids = maybe_corpus_ids
    original_index_size = len(corpus_ids)

    if force:
        logger.info("Force mode: backing up previous state...")
        backup_files([processed_path, matched_ids_path])
        # Note: s2orc_dir files are not backed up, they're just overwritten

    processed_files = await load_processed_files(processed_path)
    matched_ids = load_ids_from_file(matched_ids_path)
    prior_matches = len(matched_ids)
    if processed_files:
        logger.info(f"Resuming: {len(processed_files)} files already processed")
        logger.info(f"  Existing matches: {prior_matches:,}")

    stats = Stats()
    stats.records_matched = prior_matches  # Start from existing count

    logger.info("Fetching s2orc_v2 dataset info...")
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client_info:
        dataset = await get_dataset_info(client_info, "s2orc_v2")

    logger.info(f"  Release: {dataset.release_id}")
    logger.info(f"  Files: {len(dataset.files)}")

    remaining_files = get_remaining_files(dataset, processed_files, limit_files)

    if not remaining_files:
        logger.info("\nAll files already processed!")
        logger.info(f"  Total matches: {prior_matches:,}")
        coverage = (
            prior_matches / original_index_size * 100 if original_index_size else 0
        )
        logger.info(f"  Coverage: {coverage:.1f}% of index")
        return

    if dry_run:
        log_dry_run_files(remaining_files)
        return

    s2orc_dir.mkdir(parents=True, exist_ok=True)
    state_lock = asyncio.Lock()
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:

        async def _fulltext_work(
            url: str,
            filename: str,
            progress: ProgressManager,
        ) -> None:
            s2orc_output = s2orc_dir / output_name(filename)
            result, bytes_dl = await download_and_process(
                url,
                filename,
                output_dir,
                "s2orc_v2",
                client,
                progress,
                lambda p: process_s2orc_file(p, corpus_ids, s2orc_output),
            )
            async with state_lock:
                stats.bytes_downloaded += bytes_dl
                stats.files_processed += 1
                stats.records_matched += result.match_count
                await append_ids_to_file(matched_ids_path, result.matched_ids)
                await append_processed_file(processed_path, filename)

        await run_parallel_file_phase(
            remaining_files,
            "Extracting fulltext",
            max_concurrent,
            _fulltext_work,
        )

    coverage = (
        stats.records_matched / original_index_size * 100 if original_index_size else 0
    )
    logger.info(
        f"\nFull text extraction complete: {stats.records_matched:,} papers\n"
        f"  Files processed this run: {stats.files_processed}\n"
        f"  Output: {s2orc_dir}\n"
        f"  Coverage: {coverage:.1f}% of index"
    )
