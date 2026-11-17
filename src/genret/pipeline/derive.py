"""Phase 4: Derive co-citations and bibliographic coupling. Phase 5: Splits."""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from genret.citation_graph import (
    build_inverse_index,
    count_group_cooccurrences,
    counts_to_scored_pairs,
)
from genret.paper_parsing import ScoredPaperPair
from genret.types import CorpusId, Year
from genret.utils import (
    die,
    glob_jsonl,
    iter_jsonl,
    load_ids_from_file,
    load_references,
    open_auto,
    resolve_jsonl,
    sample,
)

logger = logging.getLogger(__name__)

# === Phase 4: Derive ===


def compute_co_citation_pairs(
    refs: dict[CorpusId, list[CorpusId]],
    min_co_citations: int,
) -> list[ScoredPaperPair]:
    """Compute co-citation pairs from references.

    Two papers are co-cited if they appear together in the reference list of
    another paper. Returns pairs sorted by score descending.

    Args:
        refs: Map of paper_id -> list of cited paper IDs.
        min_co_citations: Minimum co-citation count to include a pair.

    Returns:
        List of ScoredPaperPair objects, sorted by score descending.
    """
    pair_counts, _ = count_group_cooccurrences(refs.values())
    return [
        ScoredPaperPair(
            paper_a=CorpusId(pair.id_a), paper_b=CorpusId(pair.id_b), score=pair.score
        )
        for pair in counts_to_scored_pairs(pair_counts, min_co_citations)
    ]


def compute_bib_coupling_pairs(
    refs: dict[CorpusId, list[CorpusId]],
    min_shared_refs: int,
    max_citing: int,
) -> tuple[list[ScoredPaperPair], int]:
    """Compute bibliographic coupling pairs from references.

    Two papers are bibliographically coupled if they cite the same paper.
    Skips highly-cited references to avoid combinatorial explosion.

    Args:
        refs: Map of paper_id -> list of cited paper IDs.
        min_shared_refs: Minimum shared references to include a pair.
        max_citing: Skip references cited by more than this many papers.

    Returns:
        Tuple of (pairs, skipped_refs_count) where pairs is a list of
        ScoredPaperPair objects sorted by score descending.
    """
    cited_by = build_inverse_index(refs)
    bib_counts, skipped_refs = count_group_cooccurrences(
        cited_by.values(),
        max_group_size=max_citing,
    )
    pairs = [
        ScoredPaperPair(
            paper_a=CorpusId(pair.id_a), paper_b=CorpusId(pair.id_b), score=pair.score
        )
        for pair in counts_to_scored_pairs(bib_counts, min_shared_refs)
    ]
    return pairs, skipped_refs


def filter_pairs_by_ids(
    pairs: list[ScoredPaperPair],
    valid_ids: set[CorpusId],
) -> list[ScoredPaperPair]:
    """Filter pairs to only include those where both papers are in valid_ids.

    Args:
        pairs: List of ScoredPaperPair objects.
        valid_ids: Set of valid paper IDs.

    Returns:
        Filtered list of pairs where both papers are in valid_ids.
    """
    return [p for p in pairs if p.paper_a in valid_ids and p.paper_b in valid_ids]


def load_metadata_years(metadata_dir: Path) -> dict[CorpusId, Year]:
    """Load paper years from metadata JSONL files in directory."""
    years: dict[CorpusId, Year] = {}
    files = glob_jsonl(metadata_dir)
    for file_path in tqdm(files, desc="Loading metadata files", leave=False):
        for record in iter_jsonl(file_path):
            corpus_id = record.get("corpusid")
            year = record.get("year")
            if corpus_id and year:
                years[CorpusId(corpus_id)] = Year(year)
    return years


def write_scored_pairs(
    path: Path,
    pairs: list[ScoredPaperPair],
    score_key: str,
) -> None:
    """Write scored paper pairs to JSONL file."""
    with open_auto(path, "w") as f:
        f.writelines(
            json.dumps({"paper_a": p.paper_a, "paper_b": p.paper_b, score_key: p.score})
            + "\n"
            for p in pairs
        )


def run_derive(
    output_dir: Path,
    min_co_citations: int = 2,
    min_shared_refs: int = 3,
    max_citing: int = 200,
) -> None:
    """Phase 4: Compute co-citations and bibliographic coupling from citation graph.

    Args:
        output_dir: Directory containing citations/references.jsonl.
        min_co_citations: Minimum co-citation count to include a pair.
        min_shared_refs: Minimum shared references for bibliographic coupling.
        max_citing: Skip references cited by more than this many papers (avoids
            combinatorial explosion from highly-cited foundational papers).
    """
    refs_path = resolve_jsonl(output_dir / "citations" / "references.jsonl.gz")
    fulltext_path = output_dir / "matched_corpus_ids.txt"

    if not refs_path.exists():
        die(f"References not found: {refs_path}\nRun 'citations' phase first.")

    # Load fulltext IDs to filter - only include papers with actual content
    fulltext_ids: set[CorpusId] = set()
    if fulltext_path.exists():
        fulltext_ids = load_ids_from_file(fulltext_path)
        logger.info(f"Loaded {len(fulltext_ids):,} papers with fulltext")
    else:
        logger.info("Warning: No fulltext filter (matched_corpus_ids.txt not found)")

    logger.info("Loading references...")
    refs = load_references(refs_path, show_progress=True)
    logger.info(f"  {len(refs):,} papers with references")

    # Filter to only papers with fulltext
    if fulltext_ids:
        refs = {pid: cited for pid, cited in refs.items() if pid in fulltext_ids}
        logger.info(f"  {len(refs):,} papers after fulltext filter")
    logger.info("")

    # === Co-citations ===
    logger.info("Computing co-citations...")
    co_cite_pairs = compute_co_citation_pairs(refs, min_co_citations)
    if fulltext_ids:
        co_cite_pairs = filter_pairs_by_ids(co_cite_pairs, fulltext_ids)

    co_cite_path = output_dir / "citations" / "co_citations.jsonl.zst"
    write_scored_pairs(co_cite_path, co_cite_pairs, "count")
    logger.info(f"  Wrote {len(co_cite_pairs):,} pairs to co_citations.jsonl.zst")

    # === Bibliographic coupling ===
    logger.info("Computing bibliographic coupling...")
    bib_pairs, skipped_refs = compute_bib_coupling_pairs(
        refs, min_shared_refs, max_citing
    )

    if skipped_refs:
        logger.info(f"  Skipped {skipped_refs:,} refs cited by >{max_citing} papers")
    if fulltext_ids:
        bib_pairs = filter_pairs_by_ids(bib_pairs, fulltext_ids)

    bib_path = output_dir / "citations" / "bib_coupling.jsonl.zst"
    write_scored_pairs(bib_path, bib_pairs, "shared_refs")
    logger.info(f"  Wrote {len(bib_pairs):,} pairs to bib_coupling.jsonl.zst")

    logger.info("\nDerived data complete!")


# === Phase 5: Splits ===


@dataclass(frozen=True, kw_only=True)
class SplitResult:
    """Result of splitting papers by year."""

    train: list[CorpusId]
    dev: list[CorpusId]
    test: list[CorpusId]


def split_papers_by_year(
    years: dict[CorpusId, Year],
    train_years: tuple[int, int],
    dev_year: int,
    test_year: int,
) -> SplitResult:
    """Split paper IDs by publication year.

    Args:
        years: Map of paper_id -> publication year.
        train_years: (start, end) year range for training. If start is 0,
            all years except dev/test are included.
        dev_year: Year for development set.
        test_year: Minimum year for test set (includes this year and later).

    Returns:
        SplitResult with train, dev, and test paper ID lists.
    """
    train_ids: list[CorpusId] = []
    dev_ids: list[CorpusId] = []
    test_ids: list[CorpusId] = []

    include_all_train = train_years[0] == 0

    for paper_id, year in years.items():
        if year == dev_year:
            dev_ids.append(paper_id)
        elif year >= test_year:
            test_ids.append(paper_id)
        elif include_all_train or train_years[0] <= year <= train_years[1]:
            train_ids.append(paper_id)

    return SplitResult(train=train_ids, dev=dev_ids, test=test_ids)


def run_splits(
    data_dir: Path,
    splits_dir: Path,
    train_years: tuple[int, int] = (2019, 2022),
    dev_year: int = 2023,
    test_year: int = 2024,
    train_count: int | None = None,
    dev_count: int | None = None,
    test_count: int | None = None,
    seed: int = 42,
) -> None:
    """Phase 5: Create train/dev/test splits by publication year."""
    metadata_dir = data_dir / "papers"
    fulltext_path = data_dir / "matched_corpus_ids.txt"

    if not metadata_dir.exists() or not glob_jsonl(metadata_dir):
        die(f"Metadata not found: {metadata_dir}\nRun 'index' phase first.")

    # Load fulltext IDs to filter - only include papers with actual content
    fulltext_ids: set[CorpusId] = set()
    if fulltext_path.exists():
        fulltext_ids = load_ids_from_file(fulltext_path)
        logger.info(f"Loaded {len(fulltext_ids):,} papers with fulltext")
    else:
        logger.info("Warning: No fulltext filter (matched_corpus_ids.txt not found)")

    logger.info("Loading metadata...")
    years = load_metadata_years(metadata_dir)
    logger.info(f"  {len(years):,} papers with year info")

    # Filter to only papers with fulltext
    if fulltext_ids:
        years = {pid: year for pid, year in years.items() if pid in fulltext_ids}
        logger.info(f"  {len(years):,} papers after fulltext filter")
    logger.info("")

    # Split by year
    splits = split_papers_by_year(years, train_years, dev_year, test_year)

    # Apply sampling if counts are specified
    rng = random.Random(seed)
    train_ids = sample(rng, splits.train, train_count)
    dev_ids = sample(rng, splits.dev, dev_count)
    test_ids = sample(rng, splits.test, test_count)

    # Write splits
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", train_ids), ("dev", dev_ids), ("test", test_ids)]:
        path = splits_dir / f"{name}.txt"
        with open(path, "w") as f:
            f.writelines(f"{paper_id}\n" for paper_id in sorted(ids))

    include_all_train = train_years[0] == 0
    train_range = "all" if include_all_train else f"{train_years[0]}-{train_years[1]}"

    # Write split configuration and year distribution (splits only)
    split_ids = set(train_ids) | set(dev_ids) | set(test_ids)
    year_counts: dict[int, int] = {}
    for pid, year in years.items():
        if pid in split_ids:
            year_counts[year] = year_counts.get(year, 0) + 1

    split_info = {
        "train_years": train_range,
        "dev_year": dev_year,
        "test_year": test_year,
        "seed": seed,
        "counts": {
            "total_with_fulltext": len(years),
            "total_used": len(split_ids),
            "train": len(train_ids),
            "dev": len(dev_ids),
            "test": len(test_ids),
            "excluded": len(years) - len(split_ids),
        },
        "year_distribution": dict(sorted(year_counts.items())),
    }
    config_path = splits_dir / "split_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
        f.write("\n")

    logger.info(
        f"  Train ({train_range}): {len(train_ids):,}\n"
        f"  Dev ({dev_year}): {len(dev_ids):,}\n"
        f"  Test ({test_year}+): {len(test_ids):,}\n"
        f"\nSplits complete!"
    )
