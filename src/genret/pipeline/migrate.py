"""Migrate expensive processed files from one splits directory to another.

After connected sampling creates a smaller splits directory, the expensive
LLM-extracted files (facets, terms) and large parsed files (paper_graphs)
live in the source's `processed/` subdirectory. Rather than re-extracting,
this module filters those files to only include records matching the target
split IDs.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from genret.types import CorpusId
from genret.utils import (
    LineWriter,
    backup_files,
    die,
    iter_jsonl,
    load_split_ids,
    resolve_jsonl,
    track_progress,
)

logger = logging.getLogger(__name__)

MIGRATABLE_FILES: tuple[str, ...] = (
    "facets.jsonl",
    "terms.jsonl",
    "paper_graphs.jsonl",
)
"""Base names of processed files worth migrating (expensive to regenerate)."""

SKIPPED_FILES: dict[str, str] = {
    "training_pairs.jsonl": "uv run genret paper-processing ground-truth",
    "docids.jsonl": "uv run genret docid encode",
    "docid_index.json": "uv run genret docid build-index",
}
"""Files that are cheap to regenerate, mapped to their regeneration command."""


@dataclass(frozen=True, kw_only=True)
class MigrateConfig:
    """Configuration for migrating processed files between splits directories."""

    source_dir: Path
    """Source splits directory containing `processed/`."""
    target_dir: Path
    """Target splits directory with `train.txt`, `dev.txt`, `test.txt`."""
    force: bool
    """Whether to overwrite existing files in the target."""


@dataclass(frozen=True, kw_only=True)
class MigrateResult:
    """Result of migrating a single processed file."""

    filename: str
    """Base name of the migrated file."""
    kept: int
    """Number of records kept (matched target IDs)."""
    dropped: int
    """Number of records dropped (not in target IDs)."""


def _filter_jsonl_file(
    source_path: Path,
    target_path: Path,
    target_ids: set[CorpusId],
    *,
    id_field: str = "corpusid",
) -> MigrateResult:
    """Stream-filter a JSONL file, keeping only records with IDs in `target_ids`.

    Always writes the output as `.jsonl.zst` regardless of source compression.

    Args:
        source_path: Path to the source JSONL file (any compression).
        target_path: Path to write the filtered output (should end in `.jsonl.zst`).
        target_ids: Set of corpus IDs to keep.
        id_field: Name of the JSON field containing the corpus ID.

    Returns:
        Migration result with kept/dropped counts.
    """
    kept = 0
    dropped = 0

    target_path.parent.mkdir(parents=True, exist_ok=True)

    with LineWriter.open(target_path, "w") as writer:
        for record in track_progress(
            iter_jsonl(source_path), desc=f"Filtering {source_path.name}..."
        ):
            corpus_id = record.get(id_field)
            if corpus_id is not None and int(corpus_id) in target_ids:
                writer.write_line(json.dumps(record, ensure_ascii=False))
                kept += 1
            else:
                dropped += 1

    return MigrateResult(
        filename=target_path.name,
        kept=kept,
        dropped=dropped,
    )


def run_migrate_processed(config: MigrateConfig) -> list[MigrateResult]:
    """Migrate expensive processed files from source to target splits directory.

    Loads target split IDs, then for each migratable file, filters records to
    only those matching the target IDs. Skips cheap-to-regenerate files with
    a hint about which command to run.

    Args:
        config: Migration configuration.

    Returns:
        List of migration results, one per migrated file.
    """
    source_processed = config.source_dir / "processed"
    target_processed = config.target_dir / "processed"

    if not source_processed.exists():
        die(f"Source processed directory does not exist: {source_processed}")

    target_ids = load_split_ids(config.target_dir)
    if not target_ids:
        die(f"No split IDs found in {config.target_dir} (need train/dev/test.txt)")

    logger.info(
        "Migrating processed files\n"
        f"  Source: {source_processed}\n"
        f"  Target: {target_processed}\n"
        f"  Target IDs: {len(target_ids):,}"
    )

    results: list[MigrateResult] = []

    for base_name in MIGRATABLE_FILES:
        source_path = resolve_jsonl(source_processed / base_name)
        if not source_path.exists():
            logger.info("  Skipping %s (not found in source)", base_name)
            continue

        # Target always uses .zst compression
        target_path = target_processed / base_name.removesuffix(".jsonl")
        target_path = target_processed / (
            base_name.removesuffix(".jsonl") + ".jsonl.zst"
        )
        resolved_target = resolve_jsonl(target_processed / base_name)

        if resolved_target.exists() and not config.force:
            logger.info(
                "  Skipping %s (already exists, use --force to overwrite)",
                base_name,
            )
            continue

        if resolved_target.exists() and config.force:
            backup_files([resolved_target])

        result = _filter_jsonl_file(source_path, target_path, target_ids)
        results.append(result)

        logger.info(
            "  %s: kept %s, dropped %s",
            result.filename,
            f"{result.kept:,}",
            f"{result.dropped:,}",
        )

    # Log skipped files with regeneration hints
    skipped_hints: list[str] = []
    for filename, command in SKIPPED_FILES.items():
        skipped_hints.append(f"  {filename} -> {command}")

    if skipped_hints:
        logger.info(
            "Skipped (regenerate with):\n%s",
            "\n".join(skipped_hints),
        )

    return results
