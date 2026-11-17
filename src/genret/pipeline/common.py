"""Shared constants, data structures, and helpers for the pipeline modules."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import aiofiles
import httpx

from genret.types import CorpusId
from genret.utils import (
    ProgressManager,
    die,
    download_file,
    get_filename_from_url,
    load_ids_from_file,
)

logger = logging.getLogger(__name__)

# === Constants ===

DATASETS_API = "https://api.semanticscholar.org/datasets/v1"
"""Base URL for the Semantic Scholar datasets API."""

API_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
"""Timeout for API metadata requests (not file downloads)."""

DEFAULT_VENUE_PATTERNS = [
    r"\bACL\b",
    r"\bEMNLP\b",
    r"\bNAACL\b",
    r"\bCOLING\b",
    r"\bEACL\b",
    r"\bAACL\b",
    r"\bCONLL\b",
    r"\bSemEval\b",
    r"\b\*SEM\b",
    r"\bTACL\b",
    r"\bCL\b",
]
"""Regex patterns matching NLP venue names for corpus filtering."""


# === Data structures ===


@dataclass
class Stats:
    """Statistics for processing."""

    files_processed: int = 0
    records_matched: int = 0
    bytes_downloaded: int = 0
    # Match breakdown (for index phase)
    matched_acl_only: int = 0
    matched_venue_only: int = 0
    matched_both: int = 0


@dataclass(frozen=True, kw_only=True)
class DatasetInfo:
    """Information about a dataset from the API."""

    name: str
    files: list[str] = field(default_factory=list[str])
    release_id: str = ""


# === Shared helpers ===


async def get_dataset_info(client: httpx.AsyncClient, name: str) -> DatasetInfo:
    """Fetch dataset file URLs from the Semantic Scholar API."""
    response = await client.get(f"{DATASETS_API}/release/latest")
    response.raise_for_status()
    release_id = response.json()["release_id"]

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"x-api-key": api_key} if api_key else dict[str, str]()

    response = await client.get(
        f"{DATASETS_API}/release/{release_id}/dataset/{name}/",
        headers=headers,
    )
    response.raise_for_status()
    data = response.json()

    return DatasetInfo(
        name=name,
        files=data.get("files", []),
        release_id=release_id,
    )


def make_url_refresher(
    client: httpx.AsyncClient, dataset_name: str, filename: str
) -> Callable[[], Awaitable[str]]:
    """Create a URL refresher callback for a specific file.

    When called, re-fetches the dataset info and returns the fresh URL
    for the specified filename.
    """

    async def refresh() -> str:
        dataset = await get_dataset_info(client, dataset_name)
        for url in dataset.files:
            if get_filename_from_url(url) == filename:
                return url
        raise RuntimeError(f"File {filename} not found in refreshed dataset")

    return refresh


async def append_ids_to_file(path: Path, ids: set[CorpusId]) -> None:
    """Append integer IDs to a text file (one per line)."""
    if not ids:
        return
    async with aiofiles.open(path, "a") as f:
        await f.writelines(f"{cid}\n" for cid in sorted(ids))


def output_name(filename: str) -> str:
    """Convert source filename to output filename.

    Converts ``.jsonl.gz`` to ``.jsonl.zst`` so that pipeline output files use
    zstd compression.  Non-JSONL ``.gz`` files (e.g. ``.txt.gz``) have their
    ``.gz`` stripped.
    """
    if filename.endswith(".jsonl.gz"):
        return filename.removesuffix(".gz") + ".zst"
    return filename.removesuffix(".gz")


def get_remaining_files(
    dataset: DatasetInfo,
    processed_files: set[str],
    limit_files: int | None = None,
) -> list[tuple[str, str]]:
    """Get list of files to process, filtering out already-processed ones.

    Returns:
        List of (url, filename) pairs for files not yet processed.
    """
    files_to_process = dataset.files[:limit_files] if limit_files else dataset.files
    files_with_names = [(url, get_filename_from_url(url)) for url in files_to_process]
    remaining = [
        (url, name) for url, name in files_with_names if name not in processed_files
    ]

    skipped = len(files_to_process) - len(remaining)
    if skipped > 0:
        logger.info(f"  Skipping: {skipped} already done")
        logger.info(f"  Remaining: {len(remaining)} files")

    return remaining


async def _download_worker(
    filename: str,
    semaphore: asyncio.Semaphore,
    progress: ProgressManager,
    work: Callable[[], Any],
) -> None:
    """Generic download worker with error handling and progress tracking.

    Args:
        filename: Name of the file being processed (for error messages).
        semaphore: Semaphore to limit concurrency.
        progress: Progress manager for tracking.
        work: Async callable that performs the download, processing, and state update.
    """
    async with semaphore:
        try:
            await work()
            progress.advance()
        except Exception:
            progress.advance(failed=True)
            logger.exception("Failed %s", filename)


async def run_parallel_file_phase(
    remaining_files: list[tuple[str, str]],
    description: str,
    max_concurrent: int,
    work: Callable[[str, str, ProgressManager], Awaitable[None]],
) -> None:
    """Run a per-file async worker with shared progress and concurrency controls."""
    semaphore = asyncio.Semaphore(max_concurrent)
    with ProgressManager(len(remaining_files), description) as progress:
        tasks = [
            asyncio.create_task(
                _download_worker(
                    filename,
                    semaphore,
                    progress,
                    partial(work, url, filename, progress),
                )
            )
            for url, filename in remaining_files
        ]
        await asyncio.gather(*tasks)


def log_dry_run_files(
    remaining_files: list[tuple[str, str]], max_show: int = 10
) -> None:
    """Log files that would be processed in dry run mode.

    Args:
        remaining_files: List of (url, filename) tuples.
        max_show: Maximum number of files to show individually.
    """
    logger.info("\n[DRY RUN] Would download and process:")
    for i, (_, name) in enumerate(remaining_files[:max_show], 1):
        logger.info(f"  {i}. {name}")
    if len(remaining_files) > max_show:
        logger.info(f"  ... and {len(remaining_files) - max_show} more")


def require_corpus_ids(output_dir: Path) -> set[CorpusId]:
    """Load corpus IDs from index file, dying if not found or empty.

    Args:
        output_dir: Directory containing corpus_ids.txt.

    Returns:
        Set of corpus IDs.

    Raises:
        SystemExit: If index file doesn't exist or is empty.
    """
    index_path = output_dir / "corpus_ids.txt"
    if not index_path.exists():
        die(f"Index not found: {index_path}\nRun 'index' phase first.")

    logger.info(f"Loading corpus IDs from {index_path}...")
    corpus_ids = load_ids_from_file(index_path)
    logger.info(f"  Loaded {len(corpus_ids):,} corpus IDs")

    if not corpus_ids:
        die("Index is empty.")

    return corpus_ids


def maybe_load_corpus_ids_for_dry_run(
    output_dir: Path,
    phase_name: str,
    dry_run: bool,
) -> set[CorpusId] | None:
    """Load corpus IDs, but in dry-run mode report missing prerequisites instead of dying."""
    index_path = output_dir / "corpus_ids.txt"
    if index_path.exists():
        return require_corpus_ids(output_dir)

    if not dry_run:
        die(f"Index not found: {index_path}\nRun 'index' phase first.")

    logger.info(
        f"\n[DRY RUN] {phase_name} requires {index_path} (run 'index' phase first)."
    )
    return None


async def download_and_process(
    url: str,
    filename: str,
    temp_dir: Path,
    dataset_name: str,
    client: httpx.AsyncClient,
    progress: ProgressManager,
    process: Callable[[Path], Any],
) -> tuple[Any, int]:
    """Download a dataset file, process it, and return results.

    Generic helper that handles the common download-process-cleanup pattern
    used by the index, fulltext, and citations phases.

    Args:
        url: Download URL for the file.
        filename: Filename for temp file naming and URL refresh.
        temp_dir: Directory for temporary download files.
        dataset_name: Semantic Scholar dataset name (for URL refresh).
        client: HTTP client for API calls.
        progress: Progress manager for tracking.
        process: Sync callable that processes the downloaded file and returns a result.

    Returns:
        Tuple of (process_result, bytes_downloaded).
    """
    temp_path = temp_dir / f".temp_{filename}"
    refresher = make_url_refresher(client, dataset_name, filename)
    file_task = progress.add_file(filename)
    try:
        bytes_downloaded = await download_file(
            url,
            temp_path,
            url_refresher=refresher,
            on_progress=lambda b, t: progress.update_file(file_task, b, t),
        )
        progress.file_processing(file_task, filename)
        result = await asyncio.to_thread(process, temp_path)
        return result, bytes_downloaded
    finally:
        progress.remove_file(file_task)
        temp_path.unlink(missing_ok=True)
