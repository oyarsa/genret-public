"""Unified pipeline for acquiring S2ORC NLP dataset.

Five phases, each resumable:
1. index: Build corpus ID index AND extract paper metadata from `papers` dataset
2. fulltext: Extract full paper text from `s2orc_v2` dataset
3. citations: Extract citation relationships from `citations` dataset
4. derive: Compute co-citations and bibliographic coupling from citation graph
5. splits: Create train/dev/test splits by publication year

Run all phases with: uv run genret pipeline all <output_dir>
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from genret.pipeline.citations import run_citations
from genret.pipeline.common import DEFAULT_VENUE_PATTERNS
from genret.pipeline.connected_sampling import (
    ConnectedSampleConfig,
    run_connected_sample,
)
from genret.pipeline.derive import run_derive, run_splits
from genret.pipeline.fulltext import run_fulltext
from genret.pipeline.index import run_index
from genret.pipeline.migrate import MigrateConfig, run_migrate_processed
from genret.utils import (
    confirm_force,
    countit,
    glob_jsonl,
    iter_jsonl,
    load_ids_from_file,
    make_app,
    require_env,
    resolve_jsonl,
)

logger = logging.getLogger(__name__)

app = make_app("Unified pipeline for acquiring S2ORC NLP dataset")


# === CLI Commands ===


@app.command(name="index")
def cmd_index(
    output_dir: Annotated[Path, typer.Argument(help="Output directory.")],
    start_year: Annotated[
        int, typer.Option("--start-year", "-s", help="Start year (inclusive).")
    ] = 2010,
    end_year: Annotated[
        int, typer.Option("--end-year", "-e", help="End year (inclusive).")
    ] = 2025,
    venues: Annotated[
        list[str] | None,
        typer.Option(
            "--venue",
            "-v",
            help="Override default venue regex patterns (repeatable).",
        ),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Limit files (for testing).")
    ] = None,
    concurrent: Annotated[
        int, typer.Option("--concurrent", "-c", help="Concurrent downloads.")
    ] = 10,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be done.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Start fresh, backing up old data.")
    ] = False,
) -> None:
    """Phase 1: Build corpus ID index and extract paper metadata.

    Filters papers by ACL Anthology ID OR venue pattern match.
    """
    if force:
        confirm_force()

    require_env("SEMANTIC_SCHOLAR_API_KEY")

    venue_patterns = list(venues) if venues else DEFAULT_VENUE_PATTERNS

    logger.info(
        f"=== Phase 1: Index + Metadata ===\n"
        f"Years: {start_year}-{end_year}\n"
        f"Venues: {venue_patterns}"
    )

    asyncio.run(
        run_index(
            output_dir,
            (start_year, end_year),
            venue_patterns,
            limit,
            dry_run,
            force,
            concurrent,
        )
    )


@app.command(name="fulltext")
def cmd_fulltext(
    output_dir: Annotated[Path, typer.Argument(help="Output directory.")],
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Limit files (for testing).")
    ] = None,
    concurrent: Annotated[
        int, typer.Option("--concurrent", "-c", help="Concurrent downloads.")
    ] = 10,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be done.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Start fresh, backing up old data.")
    ] = False,
) -> None:
    """Phase 2: Extract full paper text from s2orc_v2 dataset."""
    if force:
        confirm_force()

    require_env("SEMANTIC_SCHOLAR_API_KEY")

    logger.info("=== Phase 2: Full Text ===")

    asyncio.run(run_fulltext(output_dir, limit, dry_run, force, concurrent))


@app.command(name="citations")
def cmd_citations(
    output_dir: Annotated[Path, typer.Argument(help="Output directory.")],
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Limit files (for testing).")
    ] = None,
    concurrent: Annotated[
        int, typer.Option("--concurrent", "-c", help="Concurrent downloads.")
    ] = 10,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be done.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Start fresh, backing up old data.")
    ] = False,
) -> None:
    """Phase 3: Extract citation relationships from citations dataset."""
    if force:
        confirm_force()

    require_env("SEMANTIC_SCHOLAR_API_KEY")

    logger.info("=== Phase 3: Citations ===")

    asyncio.run(run_citations(output_dir, limit, dry_run, force, concurrent))


@app.command(name="derive")
def cmd_derive(
    output_dir: Annotated[Path, typer.Argument(help="Output directory.")],
    min_co_citations: Annotated[
        int, typer.Option("--min-co-citations", help="Min co-citation count.")
    ] = 2,
    min_shared_refs: Annotated[
        int, typer.Option("--min-shared-refs", help="Min shared references.")
    ] = 3,
    max_citing: Annotated[
        int, typer.Option("--max-citing", help="Skip refs cited by more than N papers.")
    ] = 200,
) -> None:
    """Phase 4: Compute co-citations and bibliographic coupling."""
    logger.info("=== Phase 4: Derive ===")

    run_derive(output_dir, min_co_citations, min_shared_refs, max_citing)


@app.command(name="splits")
def cmd_splits(
    data_dir: Annotated[Path, typer.Argument(help="Data directory.")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory for splits.")
    ],
    train_start: Annotated[
        int, typer.Option("--train-start", help="First year for training set (0=all).")
    ] = 2019,
    train_end: Annotated[
        int, typer.Option("--train-end", help="Last year for training set.")
    ] = 2022,
    dev_year: Annotated[
        int, typer.Option("--dev-year", help="Year for dev set.")
    ] = 2023,
    test_year: Annotated[
        int,
        typer.Option("--test-year", help="Minimum year for test set (includes later)."),
    ] = 2024,
    train_count: Annotated[
        int | None, typer.Option("--train-count", help="Sample N from train split.")
    ] = None,
    dev_count: Annotated[
        int | None, typer.Option("--dev-count", help="Sample N from dev split.")
    ] = None,
    test_count: Annotated[
        int | None, typer.Option("--test-count", help="Sample N from test split.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for sampling.")] = 42,
) -> None:
    """Phase 5: Create train/dev/test splits by publication year."""
    logger.info(f"=== Phase 5: Splits ===\nOutput: {output}")

    run_splits(
        data_dir,
        output,
        (train_start, train_end),
        dev_year,
        test_year,
        train_count,
        dev_count,
        test_count,
        seed,
    )


@app.command(name="connected-sample")
def cmd_connected_sample(
    data_dir: Annotated[
        Path, typer.Argument(help="Data directory with citations/ and papers/.")
    ],
    splits: Annotated[
        Path, typer.Option("--splits", help="Existing splits directory (universe).")
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory for new splits.")
    ],
    target_size: Annotated[
        int, typer.Option("--target-size", help="Desired total number of papers.")
    ],
    min_degree: Annotated[
        int, typer.Option("--min-degree", help="K for k-core floor filtering.")
    ] = 3,
    seed: Annotated[
        int, typer.Option("--seed", help="RNG seed for tie-breaking.")
    ] = 42,
) -> None:
    """Sample a densely-connected paper subset with balanced year distribution.

    Uses k-core filtering for citation density, then corrects year distribution
    by filling under-represented years from the 1-hop fringe of well-connected
    papers.
    """
    logger.info(
        "=== Connected Sample ===\n"
        f"Data: {data_dir}\n"
        f"Source splits: {splits}\n"
        f"Target size: {target_size:,}\n"
        f"Min degree (k-core): {min_degree}\n"
        f"Seed: {seed}"
    )

    config = ConnectedSampleConfig(
        data_dir=data_dir,
        splits_dir=splits,
        output_dir=output,
        target_size=target_size,
        min_degree=min_degree,
        seed=seed,
    )
    run_connected_sample(config)


@app.command(name="migrate-processed")
def cmd_migrate_processed(
    source: Annotated[Path, typer.Argument(help="Source splits dir with processed/.")],
    target: Annotated[
        Path, typer.Argument(help="Target splits dir with train/dev/test.txt.")
    ],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing files.")
    ] = False,
) -> None:
    """Migrate expensive processed files (facets, terms, paper_graphs) to a new split.

    Filters records from the source `processed/` directory to only those
    matching the target split IDs. Cheap files (training_pairs, docids) are
    skipped with regeneration hints.
    """
    if force:
        confirm_force()

    logger.info(f"=== Migrate Processed ===\nSource: {source}\nTarget: {target}")

    config = MigrateConfig(source_dir=source, target_dir=target, force=force)
    run_migrate_processed(config)


@app.command(name="all")
def cmd_all(
    output_dir: Annotated[Path, typer.Argument(help="Output directory.")],
    start_year: Annotated[
        int, typer.Option("--start-year", "-s", help="Start year (inclusive).")
    ] = 2010,
    end_year: Annotated[
        int, typer.Option("--end-year", "-e", help="End year (inclusive).")
    ] = 2025,
    venues: Annotated[
        list[str] | None,
        typer.Option(
            "--venue",
            "-v",
            help="Override default venue regex patterns (repeatable).",
        ),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Limit files (for testing).")
    ] = None,
    concurrent: Annotated[
        int, typer.Option("--concurrent", "-c", help="Concurrent downloads.")
    ] = 10,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be done.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Start fresh, backing up old data.")
    ] = False,
    min_co_citations: Annotated[
        int, typer.Option("--min-co-citations", help="Min co-citation count.")
    ] = 2,
    min_shared_refs: Annotated[
        int, typer.Option("--min-shared-refs", help="Min shared references.")
    ] = 3,
    max_citing: Annotated[
        int, typer.Option("--max-citing", help="Skip refs cited by more than N papers.")
    ] = 200,
    train_start: Annotated[
        int, typer.Option("--train-start", help="First year for training set (0=all).")
    ] = 2019,
    train_end: Annotated[
        int, typer.Option("--train-end", help="Last year for training set.")
    ] = 2022,
    dev_year: Annotated[
        int, typer.Option("--dev-year", help="Year for dev set.")
    ] = 2023,
    test_year: Annotated[
        int,
        typer.Option("--test-year", help="Minimum year for test set (includes later)."),
    ] = 2024,
) -> None:
    """Run all five phases: index, fulltext, citations, derive, splits.

    Each phase is resumable. If interrupted, just run again.
    """
    if force:
        confirm_force()

    require_env("SEMANTIC_SCHOLAR_API_KEY")

    venue_patterns = list(venues) if venues else DEFAULT_VENUE_PATTERNS

    logger.info(
        f"=== Full Pipeline ===\n"
        f"Years: {start_year}-{end_year}\n"
        f"Venues: {venue_patterns}"
    )

    async def _run_async_phases() -> None:
        """Run the three async pipeline phases under a single event loop."""
        logger.info("--- Phase 1: Index + Metadata ---")
        await run_index(
            output_dir,
            (start_year, end_year),
            venue_patterns,
            limit,
            dry_run,
            force,
            concurrent,
        )

        logger.info("\n--- Phase 2: Full Text ---")
        await run_fulltext(output_dir, limit, dry_run, force, concurrent)

        logger.info("\n--- Phase 3: Citations ---")
        await run_citations(output_dir, limit, dry_run, force, concurrent)

    asyncio.run(_run_async_phases())

    if dry_run:
        logger.info(
            "\n[dry run] would run phase 4 (derive) after citations are available.\n"
            "[dry run] would run phase 5 (splits) after metadata/fulltext are available."
        )
    else:
        logger.info("\n--- Phase 4: Derive ---")
        run_derive(output_dir, min_co_citations, min_shared_refs, max_citing)

        logger.info("\n--- Phase 5: Splits ---")
        run_splits(
            output_dir,
            output_dir / "splits",
            (train_start, train_end),
            dev_year,
            test_year,
        )

    logger.info("\n=== Pipeline Complete ===")


@app.command(name="stats")
def cmd_stats(
    output_dir: Annotated[Path, typer.Argument(help="Output directory.")],
) -> None:
    """Show statistics about the filtered dataset."""
    index_path = output_dir / "corpus_ids.txt"
    index_stats_path = output_dir / "index_stats.json"
    metadata_dir = output_dir / "papers"
    s2orc_dir = output_dir / "s2orc"
    citations_dir = output_dir / "citations"

    lines: list[str] = ["=== Dataset Statistics ==="]

    # Index stats
    if index_path.exists():
        corpus_ids = load_ids_from_file(index_path)
        lines.append(f"\nIndex: {len(corpus_ids):,} corpus IDs")

    if index_stats_path.exists():
        with open(index_stats_path) as f:
            index_stats = json.load(f)
        lines.append(f"  Release: {index_stats.get('release_id', 'unknown')}")
        lines.append(f"  Year range: {index_stats.get('year_range', 'unknown')}")
        if "stats" in index_stats:
            s = index_stats["stats"]
            lines.append(f"  Files processed: {s.get('files_processed', 'unknown')}")
            lines.append("  Match breakdown:")
            lines.append(f"    ACL ID only: {s.get('matched_acl_only', 0):,}")
            lines.append(f"    Venue only: {s.get('matched_venue_only', 0):,}")
            lines.append(f"    Both: {s.get('matched_both', 0):,}")

    # Metadata stats
    if metadata_dir.exists():
        files = glob_jsonl(metadata_dir)
        if files:
            total_size = sum(f.stat().st_size for f in files)
            size_mb = total_size / (1024 * 1024)
            lines.append(f"\nMetadata: {metadata_dir}/")
            lines.append(f"  Files: {len(files)}")
            lines.append(f"  Total size: {size_mb:.1f} MB")

    # Fulltext stats
    if s2orc_dir.exists():
        files = glob_jsonl(s2orc_dir)
        if files:
            total_size = sum(f.stat().st_size for f in files)
            size_mb = total_size / (1024 * 1024)
            lines.append(f"\nFull text: {s2orc_dir}/")
            lines.append(f"  Files: {len(files)}")
            lines.append(f"  Total size: {size_mb:.1f} MB")
            # Count records from matched_corpus_ids.txt (faster than reading all gzip files)
            matched_ids_path = output_dir / "matched_corpus_ids.txt"
            if matched_ids_path.exists():
                matched_ids = load_ids_from_file(matched_ids_path)
                lines.append(f"  Records: {len(matched_ids):,}")
                if index_path.exists():
                    corpus_ids = load_ids_from_file(index_path)
                    coverage = (
                        len(matched_ids) / len(corpus_ids) * 100 if corpus_ids else 0
                    )
                    lines.append(f"  Coverage: {coverage:.1f}% of index")

    # Citations stats
    refs_path = resolve_jsonl(citations_dir / "references.jsonl.gz")
    cites_path = resolve_jsonl(citations_dir / "citations.jsonl.gz")
    raw_dir = citations_dir / "raw"

    if refs_path.exists() or cites_path.exists():
        lines.append("\nCitations:")
        if refs_path.exists():
            count = countit(iter_jsonl(refs_path))
            lines.append(f"  Papers with references: {count:,}")
        if cites_path.exists():
            count = countit(iter_jsonl(cites_path))
            lines.append(f"  Papers with citations: {count:,}")
        if raw_dir.exists():
            files = glob_jsonl(raw_dir)
            if files:
                total_links = 0
                for raw_file in files:
                    total_links += countit(iter_jsonl(raw_file))
                lines.append(f"  Total citation links: {total_links:,}")

    logger.info("\n".join(lines))


if __name__ == "__main__":
    app()
