"""Phase 3: Extract citation relationships from citations dataset."""

import asyncio
import gzip
import json
import logging
from collections import defaultdict
from pathlib import Path

import httpx

from genret.pipeline.common import (
    API_TIMEOUT,
    Stats,
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
    AsyncLineWriter,
    ProgressManager,
    append_processed_file,
    backup_files,
    iter_jsonl,
    load_processed_files,
    open_auto,
)

logger = logging.getLogger(__name__)


def process_citations_file(
    path: Path, corpus_ids: set[CorpusId], output_path: Path
) -> int:
    """Process citations file and write matching links to output file.

    Keeps only citation links where both the citing and cited paper are in
    the known corpus ID set.

    Args:
        path: Path to the gzipped citations JSONL source file.
        corpus_ids: Set of corpus IDs to filter by (both ends of each link must match).
        output_path: Destination path for matching citation link records.

    Returns:
        Count of citation links written to the output file.
    """
    match_count = 0
    with (
        gzip.open(path, "rt", encoding="utf-8") as f,
        open_auto(output_path, "w") as out,
    ):
        for record in iter_jsonl(f):
            citing = record.get("citingcorpusid")
            cited = record.get("citedcorpusid")
            if citing and cited and citing in corpus_ids and cited in corpus_ids:
                out.write(json.dumps({"citing": citing, "cited": cited}) + "\n")
                match_count += 1
    return match_count


def aggregate_citations(
    raw_dir: Path,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Aggregate raw citation links from all files into references and citations dicts.

    Args:
        raw_dir: Directory containing raw citation link JSONL files.

    Returns:
        A tuple of (references, citations). ``references[A]`` is the set of corpus
        IDs cited by paper A; ``citations[B]`` is the set of corpus IDs that cite B.
    """
    references: dict[int, set[int]] = defaultdict(set)
    citations: dict[int, set[int]] = defaultdict(set)

    for raw_file in sorted(raw_dir.iterdir()):
        for record in iter_jsonl(raw_file):
            citing = record["citing"]
            cited = record["cited"]
            references[citing].add(cited)
            citations[cited].add(citing)

    return references, citations


async def run_citations(
    output_dir: Path,
    limit_files: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    max_concurrent: int = 10,
) -> None:
    """Phase 3: Extract citation relationships from citations dataset.

    Downloads citation files, retains only links between papers in the corpus
    ID index, then aggregates all links into per-paper references.jsonl.zst and
    citations.jsonl.zst under citations/.

    Args:
        output_dir: Root directory for all pipeline outputs.
        limit_files: Process at most this many dataset files (useful for testing).
        dry_run: If True, report what would be done without downloading anything.
        force: If True, back up existing state and start fresh.
        max_concurrent: Maximum number of simultaneous file downloads.
    """
    citations_dir = output_dir / "citations"
    raw_dir = citations_dir / "raw"
    processed_path = output_dir / "processed_citations.txt"

    maybe_corpus_ids = maybe_load_corpus_ids_for_dry_run(
        output_dir,
        "citations phase",
        dry_run,
    )
    if maybe_corpus_ids is None:
        return

    corpus_ids = maybe_corpus_ids

    if force:
        logger.info("Force mode: backing up previous state...")
        backup_files([processed_path])
        # Note: raw_dir files are not backed up, they're just overwritten

    processed_files = await load_processed_files(processed_path)
    if processed_files:
        logger.info(f"Resuming: {len(processed_files)} files already processed")

    logger.info("Fetching citations dataset info...")
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client_info:
        dataset = await get_dataset_info(client_info, "citations")

    logger.info(f"  Release: {dataset.release_id}\n  Files: {len(dataset.files)}")

    remaining_files = get_remaining_files(dataset, processed_files, limit_files)

    if dry_run:
        if remaining_files:
            log_dry_run_files(remaining_files)
        else:
            logger.info("\nAll files already processed!")
        return

    stats = Stats()
    if remaining_files:
        raw_dir.mkdir(parents=True, exist_ok=True)
        state_lock = asyncio.Lock()
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:

            async def _citations_work(
                url: str,
                filename: str,
                progress: ProgressManager,
            ) -> None:
                raw_output = raw_dir / output_name(filename)
                result, bytes_dl = await download_and_process(
                    url,
                    filename,
                    output_dir,
                    "citations",
                    client,
                    progress,
                    lambda p: process_citations_file(p, corpus_ids, raw_output),
                )
                async with state_lock:
                    stats.bytes_downloaded += bytes_dl
                    stats.files_processed += 1
                    stats.records_matched += result
                    await append_processed_file(processed_path, filename)

            await run_parallel_file_phase(
                remaining_files,
                "Extracting citations",
                max_concurrent,
                _citations_work,
            )

    logger.info("\nAggregating citation links...")
    references, citations = aggregate_citations(raw_dir)

    refs_path = citations_dir / "references.jsonl.zst"
    async with AsyncLineWriter.open(refs_path, "w") as writer:
        for paper_id, cited_ids in sorted(references.items()):
            await writer.write_line(
                json.dumps({"paper_id": paper_id, "references": sorted(cited_ids)})
            )
    logger.info(f"  references.jsonl.zst: {len(references):,} papers")

    cites_path = citations_dir / "citations.jsonl.zst"
    async with AsyncLineWriter.open(cites_path, "w") as writer:
        for paper_id, citing_ids in sorted(citations.items()):
            await writer.write_line(
                json.dumps({"paper_id": paper_id, "citations": sorted(citing_ids)})
            )
    logger.info(f"  citations.jsonl.zst: {len(citations):,} papers")

    total_links = sum(len(refs) for refs in references.values())
    logger.info(f"\nCitations complete: {total_links:,} links")
