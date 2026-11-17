"""Download PDFs from ACL Anthology and ArXiv.

Reads paper metadata and downloads PDFs using external IDs.
Supports concurrent downloads with rate limiting and resume.
"""

import asyncio
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import aiofiles
import httpx
import typer

from genret.utils import (
    ProgressManager,
    RateLimiter,
    append_processed_file,
    countit,
    die,
    iter_jsonl,
    load_processed_files,
    make_app,
    resolve_jsonl,
)

logger = logging.getLogger(__name__)

RATE_LIMIT_ARXIV = 3.0
"""ArXiv download rate limit in requests per second."""
RATE_LIMIT_ACL = 2.0
"""ACL Anthology download rate limit in requests per second."""

ACL_PDF_URL = "https://aclanthology.org/{acl_id}.pdf"
"""URL template for ACL Anthology PDF downloads."""
ARXIV_PDF_URL = "https://export.arxiv.org/pdf/{arxiv_id}.pdf"
"""URL template for ArXiv PDF downloads."""

app = make_app("Download PDFs from ACL Anthology and ArXiv")


@dataclass(frozen=True, kw_only=True)
class PdfPaper:
    """A paper identified for PDF download.

    Attributes:
        url: Full URL of the PDF to download.
        identifier: Stable source-prefixed identifier (e.g. ``acl:P19-1001``).
        corpus_id: Semantic Scholar corpus ID used as the output filename.
    """

    url: str
    identifier: str
    corpus_id: int


@dataclass
class DownloadStats:
    """Statistics for a PDF download run.

    Attributes:
        total: Total number of PDFs attempted.
        downloaded: Successfully downloaded files.
        skipped: Files skipped because they already existed on disk.
        failed: Files that failed to download or failed PDF validation.
    """

    total: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0


async def download_pdf(
    client: httpx.AsyncClient,
    url: str,
    output_path: Path,
    rate_limiter: RateLimiter,
) -> bool:
    """Download a PDF file using an atomic write, validating magic bytes.

    Acquires a rate-limiter token before making the request, writes to a
    ``.pdf.part`` temporary file, validates the ``%PDF`` magic bytes, then
    renames atomically to ``output_path``. Cleans up the temporary file on
    any error.

    Args:
        client: Shared async HTTP client to reuse connections.
        url: Full URL of the PDF to download.
        output_path: Destination path for the completed PDF file.
        rate_limiter: Limiter controlling request frequency for this source.

    Returns:
        True if the file was downloaded and validated successfully, False otherwise.
    """
    await rate_limiter.acquire()

    temp_path = output_path.with_suffix(".pdf.part")
    try:
        response = await client.get(url, follow_redirects=True)
        if response.status_code == 200:
            content = response.content
            # Validate PDF magic bytes (%PDF)
            if len(content) < 4 or content[:4] != b"%PDF":
                return False
            # Write to temp file first
            async with aiofiles.open(temp_path, "wb") as f:
                await f.write(content)
            # Atomic rename
            temp_path.rename(output_path)
            return True
        else:
            return False
    except Exception:
        # Clean up partial file on error
        if temp_path.exists():
            temp_path.unlink()
        return False


def get_pdf_url(record: dict[str, Any], source: str) -> tuple[str, str] | None:
    """Get the PDF download URL and a stable identifier for a paper record.

    Args:
        record: Raw paper metadata dict containing an ``externalids`` field.
        source: Source to resolve; either ``"acl"`` or ``"arxiv"``.

    Returns:
        A ``(url, identifier)`` tuple, or ``None`` if the paper has no ID for the
        requested source.
    """
    external_ids: dict[str, Any] = record.get("externalids", {}) or {}

    if source == "acl":
        acl_id: str | None = external_ids.get("ACL")
        if acl_id:
            return ACL_PDF_URL.format(acl_id=acl_id), f"acl:{acl_id}"
    elif source == "arxiv":
        arxiv_id: str | None = external_ids.get("ArXiv")
        if arxiv_id:
            return ARXIV_PDF_URL.format(arxiv_id=arxiv_id), f"arxiv:{arxiv_id}"

    return None


async def download_from_source(
    metadata_path: Path,
    output_dir: Path,
    source: str,
    rate_limit: float,
    max_concurrent: int,
    limit: int | None = None,
    dry_run: bool = False,
    acl_dir: Path | None = None,
) -> None:
    """Download PDFs from a single source (ACL or ArXiv) with resume support.

    Tracks completed downloads in ``downloaded_{source}.txt`` so interrupted
    runs can be resumed. PDFs are named ``{corpus_id}.pdf``.

    Args:
        metadata_path: Path to the paper metadata JSONL (plain or gzip).
        output_dir: Directory where PDFs and the progress file are written.
        source: Source identifier; either ``"acl"`` or ``"arxiv"``.
        rate_limit: Maximum requests per second for this source.
        max_concurrent: Maximum simultaneous in-flight download requests.
        limit: Cap the number of downloads (useful for testing).
        dry_run: If True, log what would be downloaded without fetching.
        acl_dir: If provided, skip papers already downloaded from ACL there.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = output_dir / f"downloaded_{source}.txt"

    # Load metadata and filter to papers with this source
    logger.info(f"Loading metadata from {metadata_path}...")
    papers: list[PdfPaper] = []
    skipped_have_acl = 0

    async with aiofiles.open(metadata_path, encoding="utf-8") as f:
        async for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            result = get_pdf_url(record, source)
            if not result:
                continue

            url, identifier = result
            corpus_id = record.get("corpusid")

            # Skip if PDF already exists in ACL directory
            if acl_dir and (acl_dir / f"{corpus_id}.pdf").exists():
                skipped_have_acl += 1
                continue

            papers.append(PdfPaper(url=url, identifier=identifier, corpus_id=corpus_id))

    logger.info(f"  Found {len(papers):,} papers with {source.upper()} IDs")
    if skipped_have_acl:
        logger.info(f"  Skipped (already have ACL PDF): {skipped_have_acl:,}")

    # Load already downloaded
    downloaded = await load_processed_files(downloaded_path)
    if downloaded:
        logger.info(f"  Already downloaded: {len(downloaded):,}")

    # Filter out already downloaded
    remaining = [p for p in papers if p.identifier not in downloaded]
    logger.info(f"  Remaining: {len(remaining):,}")

    if limit:
        remaining = remaining[:limit]
        logger.info(f"  Limited to: {len(remaining):,}")

    if dry_run:
        logger.info(
            f"\n[DRY RUN] Would download {len(remaining):,} PDFs to {output_dir}"
        )
        for p in remaining[:10]:
            logger.info(f"  {p.identifier}: {p.url}")
        if len(remaining) > 10:
            logger.info(f"  ... and {len(remaining) - 10} more")
        return

    if not remaining:
        logger.info("\nAll PDFs already downloaded!")
        return

    stats = DownloadStats(total=len(remaining))
    rate_limiter = RateLimiter(rate_limit)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def worker(
        url: str,
        identifier: str,
        corpus_id: int,
        progress: ProgressManager,
        client: httpx.AsyncClient,
    ) -> None:
        async with semaphore:
            pdf_path = output_dir / f"{corpus_id}.pdf"

            if pdf_path.exists():
                stats.skipped += 1
                await append_processed_file(downloaded_path, identifier)
                progress.advance()
            else:
                file_task = progress.add_file(f"{corpus_id}.pdf")
                try:
                    success = await download_pdf(client, url, pdf_path, rate_limiter)
                    if success:
                        stats.downloaded += 1
                        await append_processed_file(downloaded_path, identifier)
                        progress.advance()
                    else:
                        stats.failed += 1
                        progress.advance(failed=True)
                finally:
                    progress.remove_file(file_task)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30, read=60, write=30, pool=30)
    ) as client:
        with ProgressManager(
            len(remaining), f"Downloading PDFs ({source})"
        ) as progress:
            # Process in batches to manage memory
            batch_size = 1000
            for batch in itertools.batched(remaining, batch_size, strict=False):
                tasks = [
                    asyncio.create_task(
                        worker(p.url, p.identifier, p.corpus_id, progress, client)
                    )
                    for p in batch
                ]
                await asyncio.gather(*tasks)

    logger.info(
        f"Download complete!\n"
        f"  Downloaded: {stats.downloaded:,}\n"
        f"  Skipped (exists): {stats.skipped:,}\n"
        f"  Failed: {stats.failed:,}"
    )


@app.command(name="acl")
def cmd_acl(
    data_dir: Annotated[
        Path, typer.Argument(help="Directory containing papers/metadata.jsonl")
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for PDFs."),
    ] = None,
    concurrent: Annotated[
        int,
        typer.Option("--concurrent", "-c", help="Number of concurrent downloads."),
    ] = 5,
    rate: Annotated[
        float,
        typer.Option("--rate", "-r", help="Requests per second."),
    ] = RATE_LIMIT_ACL,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Limit number of downloads (for testing)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be done without downloading."),
    ] = False,
) -> None:
    """Download PDFs from ACL Anthology.

    Downloads papers that have an ACL Anthology ID in their metadata.
    PDFs are saved as {corpus_id}.pdf in the output directory.
    """
    metadata_path = resolve_jsonl(data_dir / "papers" / "metadata.jsonl.gz")
    if not metadata_path.exists():
        die(f"Metadata not found: {metadata_path}")

    output_dir = output or data_dir / "pdfs" / "acl"

    logger.info("=== Downloading PDFs from ACL Anthology ===")
    logger.info("Source: https://aclanthology.org/")
    logger.info("")

    asyncio.run(
        download_from_source(
            metadata_path, output_dir, "acl", rate, concurrent, limit, dry_run
        )
    )


@app.command(name="arxiv")
def cmd_arxiv(
    data_dir: Annotated[
        Path, typer.Argument(help="Directory containing papers/metadata.jsonl")
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for PDFs."),
    ] = None,
    concurrent: Annotated[
        int,
        typer.Option("--concurrent", "-c", help="Number of concurrent downloads."),
    ] = 5,
    rate: Annotated[
        float,
        typer.Option("--rate", "-r", help="Requests per second."),
    ] = RATE_LIMIT_ARXIV,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Limit number of downloads (for testing)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be done without downloading."),
    ] = False,
) -> None:
    """Download PDFs from ArXiv.

    Downloads papers that have an ArXiv ID in their metadata.
    Uses export.arxiv.org as recommended for programmatic access.
    PDFs are saved as {corpus_id}.pdf in the output directory.
    """
    metadata_path = resolve_jsonl(data_dir / "papers" / "metadata.jsonl.gz")
    if not metadata_path.exists():
        die(f"Metadata not found: {metadata_path}")

    output_dir = output or data_dir / "pdfs" / "arxiv"
    acl_dir = data_dir / "pdfs" / "acl"

    logger.info("=== Downloading PDFs from ArXiv ===")
    logger.info("Source: https://export.arxiv.org/")
    logger.info("")

    asyncio.run(
        download_from_source(
            metadata_path,
            output_dir,
            "arxiv",
            rate,
            concurrent,
            limit,
            dry_run,
            acl_dir,
        )
    )


@app.command(name="all")
def cmd_all(
    data_dir: Annotated[
        Path, typer.Argument(help="Directory containing papers/metadata.jsonl")
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Base output directory for PDFs."),
    ] = None,
    concurrent: Annotated[
        int,
        typer.Option("--concurrent", "-c", help="Number of concurrent downloads."),
    ] = 5,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Limit number of downloads per source."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be done without downloading."),
    ] = False,
) -> None:
    """Download PDFs from all sources (ACL first, then ArXiv).

    Runs ACL first so ArXiv can skip papers already downloaded from ACL.
    """
    metadata_path = resolve_jsonl(data_dir / "papers" / "metadata.jsonl.gz")
    if not metadata_path.exists():
        die(f"Metadata not found: {metadata_path}")

    base_output = output or data_dir / "pdfs"
    acl_dir = base_output / "acl"
    arxiv_dir = base_output / "arxiv"

    logger.info("=== Downloading PDFs from all sources ===")
    logger.info("")

    async def download_all() -> None:
        # ACL first
        await download_from_source(
            metadata_path,
            acl_dir,
            "acl",
            RATE_LIMIT_ACL,
            concurrent,
            limit,
            dry_run,
        )
        # ArXiv second, skipping papers already in ACL
        await download_from_source(
            metadata_path,
            arxiv_dir,
            "arxiv",
            RATE_LIMIT_ARXIV,
            concurrent,
            limit,
            dry_run,
            acl_dir,
        )

    asyncio.run(download_all())


@app.command(name="stats")
def cmd_stats(
    data_dir: Annotated[
        Path, typer.Argument(help="Directory containing papers/metadata.jsonl")
    ],
) -> None:
    """Show PDF download statistics."""
    metadata_path = resolve_jsonl(data_dir / "papers" / "metadata.jsonl.gz")
    if not metadata_path.exists():
        die(f"Metadata not found: {metadata_path}")

    acl_count = 0
    arxiv_count = 0
    both_count = 0
    neither_count = 0

    for record in iter_jsonl(metadata_path):
        external_ids: dict[str, Any] = record.get("externalids", {}) or {}
        has_acl = external_ids.get("ACL") is not None
        has_arxiv = external_ids.get("ArXiv") is not None

        if has_acl and has_arxiv:
            both_count += 1
        elif has_acl:
            acl_count += 1
        elif has_arxiv:
            arxiv_count += 1
        else:
            neither_count += 1

    total = acl_count + arxiv_count + both_count + neither_count
    total_downloadable = acl_count + arxiv_count + both_count

    pct = total_downloadable / total * 100
    logger.info(
        f"=== PDF availability ===\n"
        f"Total papers: {total:,}\n"
        f"\n"
        f"ACL Anthology only: {acl_count:,}\n"
        f"ArXiv only: {arxiv_count:,}\n"
        f"Both ACL + ArXiv: {both_count:,}\n"
        f"Neither (no PDF source): {neither_count:,}\n"
        f"\n"
        f"Total downloadable: {total_downloadable:,} ({pct:.1f}%)"
    )

    # Check download progress
    pdfs_dir = data_dir / "pdfs"
    for source in ["acl", "arxiv"]:
        downloaded_path = pdfs_dir / source / f"downloaded_{source}.txt"
        if downloaded_path.exists():
            with open(downloaded_path) as f:
                count = countit(f)
            logger.info(f"  Downloaded from {source}: {count:,}")


if __name__ == "__main__":
    app()
