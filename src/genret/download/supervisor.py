"""Supervisor script to download all PDFs with automatic retry.

Keeps running the download script until all files are downloaded.
Handles failures gracefully and provides progress tracking.
"""

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer

from genret.utils import countit, die, make_app, resolve_jsonl

app = make_app("Download all PDFs with automatic retry")

logger = logging.getLogger(__name__)


def count_lines(path: Path) -> int:
    """Count lines in a file. Returns 0 if file doesn't exist."""
    if not path.exists():
        return 0
    with open(path) as f:
        return countit(f)


def count_pattern(path: Path, pattern: str) -> int:
    """Count lines matching pattern in a file."""
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if pattern in line)


def get_free_space_gb(path: Path) -> float:
    """Get free disk space in GB for the filesystem containing path."""
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def cleanup_partial_downloads(pdfs_dir: Path) -> int:
    """Remove .part files from failed downloads. Returns count of files removed."""
    if not pdfs_dir.exists():
        return 0
    part_files = list(pdfs_dir.rglob("*.part"))
    for f in part_files:
        f.unlink()
    return len(part_files)


def run_download(data_dir: Path) -> int:
    """Run the download script with real-time output. Returns exit_code."""
    result = subprocess.run(
        ["uv", "run", "genret", "download-pdfs", "all", str(data_dir)],
        check=False,
    )
    return result.returncode


@app.command()
def main(
    data_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing papers/metadata.jsonl"),
    ] = Path("data/s2orc-nlp"),
    max_failures: Annotated[
        int,
        typer.Option(
            "--max-failures", "-m", help="Max consecutive failures before exit."
        ),
    ] = 50,
    retry_delay: Annotated[
        int,
        typer.Option("--retry-delay", "-d", help="Seconds to wait after failure."),
    ] = 60,
    min_disk_gb: Annotated[
        float,
        typer.Option("--min-disk", help="Minimum free disk space in GB."),
    ] = 10.0,
) -> None:
    """Download all PDFs with automatic retry until complete.

    Runs the download script repeatedly, handling failures and tracking progress.
    Exits when all PDFs are downloaded or max consecutive failures reached.
    """
    metadata_file = resolve_jsonl(data_dir / "papers" / "metadata.jsonl.gz")
    acl_tracking = data_dir / "pdfs" / "acl" / "downloaded_acl.txt"
    arxiv_tracking = data_dir / "pdfs" / "arxiv" / "downloaded_arxiv.txt"
    pdfs_dir = data_dir / "pdfs"

    if not metadata_file.exists():
        die(f"Metadata file not found: {metadata_file}")

    # Count total available
    acl_total = count_pattern(metadata_file, '"ACL": "')
    arxiv_total = count_pattern(metadata_file, '"ArXiv": "')

    logger.info(
        f"=== PDF Download Supervisor ===\n"
        f"Data directory: {data_dir}\n"
        f"Max consecutive failures: {max_failures}\n"
        f"Retry delay: {retry_delay}s\n"
        f"Min disk space: {min_disk_gb}GB\n"
        f"\n"
        f"Papers with ACL IDs: {acl_total:,}\n"
        f"Papers with ArXiv IDs: {arxiv_total:,}"
    )

    run_count = 0
    consecutive_failures = 0

    while True:
        run_count += 1
        logger.info(f"=== Run {run_count} ===")

        # Check disk space
        free_gb = get_free_space_gb(data_dir)
        if free_gb < min_disk_gb:
            die(f"Only {free_gb:.1f}GB free, need at least {min_disk_gb}GB")
        logger.info(f"Disk space: {free_gb:.1f}GB free")

        # Clean up partial downloads
        cleaned = cleanup_partial_downloads(pdfs_dir)
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} partial downloads")

        # Check current progress
        acl_downloaded = count_lines(acl_tracking)
        arxiv_downloaded = count_lines(arxiv_tracking)
        acl_remaining = acl_total - acl_downloaded
        arxiv_remaining = arxiv_total - arxiv_downloaded

        logger.info(
            f"Progress: ACL {acl_downloaded:,}/{acl_total:,},"
            f" ArXiv {arxiv_downloaded:,}/{arxiv_total:,}\n"
            f"Remaining: ACL {acl_remaining:,}, ArXiv {arxiv_remaining:,}"
        )

        # Check if done
        if acl_remaining <= 0 and arxiv_remaining <= 0:
            logger.info(
                f"\n=== All downloads complete! ===\n"
                f"Total: {acl_downloaded:,} ACL + {arxiv_downloaded:,} ArXiv PDFs"
            )
            return

        # Run download
        logger.info("\nStarting download...")
        exit_code = run_download(data_dir)

        # Check progress after run
        new_acl = count_lines(acl_tracking)
        new_arxiv = count_lines(arxiv_tracking)
        progress = (new_acl + new_arxiv) - (acl_downloaded + arxiv_downloaded)

        if exit_code != 0:
            logger.info(f"Download exited with code {exit_code}")
            consecutive_failures += 1
        elif progress > 0:
            logger.info(f"Progress this run: +{progress:,} PDFs")
            consecutive_failures = 0
        else:
            logger.info("No progress this run")
            consecutive_failures += 1

        # Check failure limit
        if consecutive_failures >= max_failures:
            die(f"\nMax consecutive failures ({max_failures}) reached. Exiting.")

        # Wait before retry if there were issues
        if consecutive_failures > 0:
            logger.info(
                f"Waiting {retry_delay}s before retry"
                f" (failure {consecutive_failures}/{max_failures})..."
            )
            time.sleep(retry_delay)


if __name__ == "__main__":
    app()
