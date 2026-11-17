"""Dataset download script for S2ORC generative retrieval pipeline.

Downloads papers from Semantic Scholar with full metadata, references, and citations.
Builds citation graphs and creates train/dev/test splits.
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from genret.citation_graph import compute_overlap_pairs
from genret.config import (
    ConferencePattern,
    Paper,
    normalise_venue,
    require_conferences,
)
from genret.s2.api import SemanticScholarAPI, extract_pdf_url
from genret.s2.conference_fetch import FetchPapersResult, fetch_papers_by_venues
from genret.s2.types import S2Paper
from genret.types import AuthorId, PaperId
from genret.utils import (
    die,
    iter_jsonl,
    make_app,
    open_auto,
    require_env,
    resolve_jsonl,
)

logger = logging.getLogger(__name__)

app = make_app("Download S2ORC dataset for generative retrieval pipeline")


@dataclass(frozen=True, kw_only=True)
class Author:
    """Author information.

    Attributes:
        author_id: Semantic Scholar author ID, or None if not available.
        name: Full display name of the author.
    """

    author_id: AuthorId | None
    name: str


@dataclass(frozen=True, kw_only=True)
class PaperMetadata(Paper):
    """Extended paper metadata for the retrieval pipeline.

    Extends Paper with abstract and structured author information.

    Attributes:
        abstract: Paper abstract text, or None if not available.
        authors: Ordered list of authors with IDs and names.
    """

    abstract: str | None = None
    authors: list[Author] = field(default_factory=list[Author])


@dataclass(frozen=True, kw_only=True)
class PaperPairScore:
    """A scored pair of papers (for co-citation or bibliographic coupling).

    Attributes:
        paper_1: ID of the first paper in the pair.
        paper_2: ID of the second paper in the pair.
        score: Overlap count (number of shared citing or cited papers).
    """

    paper_1: PaperId
    paper_2: PaperId
    score: int


@dataclass(frozen=True, kw_only=True)
class CitationData:
    """References and citations for a set of papers.

    Attributes:
        references: Maps each paper ID to the list of paper IDs it cites.
        citations: Maps each paper ID to the list of paper IDs that cite it.
    """

    references: dict[PaperId, list[PaperId]]
    citations: dict[PaperId, list[PaperId]]


@dataclass(frozen=True, kw_only=True)
class DatasetSplits:
    """Train/dev/test paper ID splits.

    Attributes:
        train: Paper IDs assigned to the training set.
        dev: Paper IDs assigned to the development set.
        test: Paper IDs assigned to the test set.
    """

    train: list[PaperId]
    dev: list[PaperId]
    test: list[PaperId]


@dataclass(frozen=True, kw_only=True)
class DownloadResult:
    """Complete result of dataset download.

    Attributes:
        papers: All downloaded papers keyed by paper ID.
        citation_data: References and citations for each paper.
        co_citations: Scored paper pairs frequently cited together.
        bib_coupling: Scored paper pairs sharing common references.
        splits: Train/dev/test split assignment by year.
        failed_fetches: Names of conferences whose API fetch failed.
    """

    papers: dict[PaperId, PaperMetadata]
    citation_data: CitationData
    co_citations: list[PaperPairScore]
    bib_coupling: list[PaperPairScore]
    splits: DatasetSplits
    failed_fetches: list[str] = field(default_factory=list[str])


@dataclass
class DatasetStats:
    """Statistics about the downloaded dataset.

    Attributes:
        total_papers: Total number of papers downloaded.
        papers_with_abstract: Papers that have a non-empty abstract.
        papers_with_pdf: Papers with an open-access PDF URL.
        papers_with_references: Papers that have at least one reference.
        total_references: Sum of reference counts across all papers.
        total_citations: Sum of incoming citation counts across all papers.
        by_year: Paper count grouped by publication year.
        by_venue: Paper count grouped by normalised venue name.
    """

    total_papers: int = 0
    papers_with_abstract: int = 0
    papers_with_pdf: int = 0
    papers_with_references: int = 0
    total_references: int = 0
    total_citations: int = 0
    by_year: dict[int, int] = field(default_factory=dict[int, int])
    by_venue: dict[str, int] = field(default_factory=dict[str, int])


EXTENDED_PAPER_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "venue",
    "authors",
    "citationCount",
    "fieldsOfStudy",
    "url",
    "openAccessPdf",
    "externalIds",
]
"""Semantic Scholar API fields requested for paper metadata."""


def parse_paper_metadata(paper: S2Paper) -> PaperMetadata:
    """Parse an S2Paper into PaperMetadata.

    Args:
        paper: Raw paper record from the Semantic Scholar API.

    Returns:
        PaperMetadata with all available fields populated.
    """
    pdf_url, pdf_source = extract_pdf_url(paper)

    # Parse authors
    authors: list[Author] = []
    for s2_author in paper.authors or []:
        authors.append(
            Author(
                author_id=AuthorId(s2_author.author_id)
                if s2_author.author_id
                else None,
                name=s2_author.name,
            )
        )

    return PaperMetadata(
        paper_id=PaperId(paper.paper_id),
        title=paper.title,
        abstract=paper.abstract,
        year=paper.year,
        venue=paper.venue,
        authors=authors,
        citation_count=paper.citation_count,
        fields_of_study=paper.fields_of_study or [],
        url=paper.url,
        open_access_pdf=pdf_url,
        pdf_source=pdf_source,
    )


async def get_papers_by_venues(
    api: SemanticScholarAPI,
    venues: list[str],
    year_range: tuple[int, int] | None = None,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> FetchPapersResult[PaperMetadata]:
    """Get all papers from multiple venues with extended metadata.

    Args:
        api: Authenticated Semantic Scholar API client.
        venues: Venue name patterns to search for.
        year_range: Optional inclusive (start_year, end_year) filter.
        progress: Optional Rich Progress instance for live updates.
        task_id: Task ID within ``progress`` to advance as pages arrive.

    Returns:
        FetchPapersResult containing parsed PaperMetadata and an error flag.
    """

    def on_page_count(page_size: int) -> None:
        if progress is not None and task_id is not None:
            progress.update(task_id, advance=page_size)

    def on_error(venue_query: str, exc: Exception) -> None:
        logger.warning(f"Error fetching venues '{venue_query}': {exc}")

    return await fetch_papers_by_venues(
        api,
        venues,
        parse_paper_metadata,
        year_range=year_range,
        fields=EXTENDED_PAPER_FIELDS,
        on_page_count=on_page_count
        if progress is not None and task_id is not None
        else None,
        on_error=on_error,
    )


async def fetch_papers_for_conferences(
    api: SemanticScholarAPI,
    conferences: list[ConferencePattern],
    year_range: tuple[int, int] | None,
) -> tuple[dict[PaperId, PaperMetadata], list[str]]:
    """Fetch papers for all configured conferences, deduplicating across venues.

    Args:
        api: Authenticated Semantic Scholar API client.
        conferences: Conference definitions with venue patterns to query.
        year_range: Optional inclusive (start_year, end_year) year filter.

    Returns:
        A tuple of (papers, failed_fetches) where papers is a deduplicated
        dict keyed by paper ID and failed_fetches lists conference names
        whose API request failed.
    """
    all_papers: dict[PaperId, PaperMetadata] = {}
    failed_fetches: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        for conference in conferences:
            task_id = progress.add_task(
                f"[cyan]Fetching {conference.name}...", total=None
            )

            result = await get_papers_by_venues(
                api,
                venues=conference.venue_patterns,
                year_range=year_range,
                progress=progress,
                task_id=task_id,
            )
            if result.had_error:
                failed_fetches.append(conference.name)

            # Deduplicate
            new_count = 0
            for paper in result.papers:
                if paper.paper_id not in all_papers:
                    all_papers[paper.paper_id] = paper
                    new_count += 1

            progress.update(
                task_id,
                description=f"[green]{conference.name}: {new_count} papers[/green]",
                completed=new_count,
                total=new_count,
            )

    return all_papers, failed_fetches


async def fetch_citations_for_papers(
    api: SemanticScholarAPI,
    paper_ids: list[PaperId],
    fetch_incoming: bool = True,
) -> CitationData:
    """Fetch references and citations for all papers.

    Args:
        api: Semantic Scholar API client
        paper_ids: List of paper IDs to fetch citations for
        fetch_incoming: Whether to also fetch incoming citations

    Returns:
        CitationData containing references and citations dicts
    """
    references: dict[PaperId, list[PaperId]] = {}
    citations: dict[PaperId, list[PaperId]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(
            "[cyan]Fetching references...", total=len(paper_ids)
        )

        for paper_id in paper_ids:
            refs = await api.get_paper_references(paper_id)
            references[paper_id] = [PaperId(r) for r in refs]
            progress.advance(task_id)

        if fetch_incoming:
            progress.update(task_id, description="[cyan]Fetching citations...")
            progress.reset(task_id, total=len(paper_ids))

            for paper_id in paper_ids:
                cites = await api.get_paper_citations(paper_id)
                citations[paper_id] = [PaperId(c) for c in cites]
                progress.advance(task_id)

    return CitationData(references=references, citations=citations)


def compute_co_citations(
    citations: dict[PaperId, list[PaperId]],
    min_overlap: int = 2,
) -> list[PaperPairScore]:
    """Compute co-citation pairs (papers frequently cited together).

    Two papers are co-cited if they appear together in the reference lists
    of other papers.

    Args:
        citations: Maps each paper ID to the list of papers that cite it.
        min_overlap: Minimum number of shared citing papers to include a pair.

    Returns:
        Scored pairs sorted by descending co-citation count.
    """
    logger.info("[cyan]Computing co-citations...[/cyan]")
    return [
        PaperPairScore(
            paper_1=pair.id_a,
            paper_2=pair.id_b,
            score=pair.score,
        )
        for pair in compute_overlap_pairs(citations, min_overlap)
    ]


def compute_bibliographic_coupling(
    references: dict[PaperId, list[PaperId]],
    min_overlap: int = 2,
) -> list[PaperPairScore]:
    """Compute bibliographic coupling pairs (papers sharing common references).

    Two papers are bibliographically coupled if they cite the same papers.

    Args:
        references: Maps each paper ID to the list of papers it references.
        min_overlap: Minimum number of shared references to include a pair.

    Returns:
        Scored pairs sorted by descending shared-reference count.
    """
    logger.info("[cyan]Computing bibliographic coupling...[/cyan]")
    return [
        PaperPairScore(
            paper_1=pair.id_a,
            paper_2=pair.id_b,
            score=pair.score,
        )
        for pair in compute_overlap_pairs(references, min_overlap)
    ]


def create_splits(
    papers: dict[PaperId, PaperMetadata],
    train_years: tuple[int, int],
    dev_year: int,
    test_year: int,
) -> DatasetSplits:
    """Create train/dev/test splits based on publication year.

    Papers without a year are excluded from all splits.

    Args:
        papers: All downloaded papers keyed by paper ID.
        train_years: Inclusive (start_year, end_year) range for the training set.
        dev_year: Publication year assigned to the development set.
        test_year: Publication year assigned to the test set.

    Returns:
        DatasetSplits with paper IDs partitioned into train, dev, and test.
    """
    train_ids: list[PaperId] = []
    dev_ids: list[PaperId] = []
    test_ids: list[PaperId] = []

    for paper_id, paper in papers.items():
        if paper.year is None:
            continue
        if train_years[0] <= paper.year <= train_years[1]:
            train_ids.append(paper_id)
        elif paper.year == dev_year:
            dev_ids.append(paper_id)
        elif paper.year == test_year:
            test_ids.append(paper_id)

    return DatasetSplits(train=train_ids, dev=dev_ids, test=test_ids)


def save_dataset(
    output_dir: Path,
    papers: dict[PaperId, PaperMetadata],
    citation_data: CitationData,
    co_citations: list[PaperPairScore],
    bib_coupling: list[PaperPairScore],
    splits: DatasetSplits,
) -> None:
    """Save all dataset artifacts to disk.

    Writes the following files under ``output_dir``:
    - ``papers/metadata.jsonl.zst``: per-paper metadata
    - ``citations/references.jsonl.zst``: outgoing references per paper
    - ``citations/citations.jsonl.zst``: incoming citations per paper
    - ``citations/co_citations.jsonl.zst``: co-citation pair scores
    - ``citations/bib_coupling.jsonl.zst``: bibliographic coupling scores
    - ``splits/{train,dev,test}.txt``: paper ID lists for each split

    Args:
        output_dir: Root directory for all dataset output files.
        papers: All downloaded papers keyed by paper ID.
        citation_data: References and citations for each paper.
        co_citations: Co-citation pair scores to persist.
        bib_coupling: Bibliographic coupling pair scores to persist.
        splits: Train/dev/test paper ID assignments.
    """
    # Create directories
    papers_dir = output_dir / "papers"
    citations_dir = output_dir / "citations"
    splits_dir = output_dir / "splits"

    for d in [papers_dir, citations_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save paper metadata
    logger.info("[cyan]Saving paper metadata...[/cyan]")
    with open_auto(papers_dir / "metadata.jsonl.zst", "w") as f:
        for paper in papers.values():
            # Convert to dict, excluding full_text (not available from API)
            data = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "year": paper.year,
                "venue": paper.venue,
                "authors": [asdict(a) for a in paper.authors],
                "citation_count": paper.citation_count,
                "fields_of_study": paper.fields_of_study,
                "url": paper.url,
                "open_access_pdf": paper.open_access_pdf,
                "pdf_source": paper.pdf_source,
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # Save references
    logger.info("[cyan]Saving references...[/cyan]")
    with open_auto(citations_dir / "references.jsonl.zst", "w") as f:
        for paper_id, refs in citation_data.references.items():
            data = {"paper_id": paper_id, "references": list(refs)}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # Save citations
    logger.info("[cyan]Saving citations...[/cyan]")
    with open_auto(citations_dir / "citations.jsonl.zst", "w") as f:
        for paper_id, cites in citation_data.citations.items():
            data = {"paper_id": paper_id, "citations": list(cites)}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # Save co-citations
    logger.info("[cyan]Saving co-citations...[/cyan]")
    with open_auto(citations_dir / "co_citations.jsonl.zst", "w") as f:
        for pair in co_citations:
            data = {
                "paper_1": pair.paper_1,
                "paper_2": pair.paper_2,
                "count": pair.score,
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # Save bibliographic coupling
    logger.info("[cyan]Saving bibliographic coupling...[/cyan]")
    with open_auto(citations_dir / "bib_coupling.jsonl.zst", "w") as f:
        for pair in bib_coupling:
            data = {
                "paper_1": pair.paper_1,
                "paper_2": pair.paper_2,
                "count": pair.score,
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # Save splits
    logger.info("[cyan]Saving splits...[/cyan]")
    with open(splits_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(splits.train) + "\n")
    with open(splits_dir / "dev.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(splits.dev) + "\n")
    with open(splits_dir / "test.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(splits.test) + "\n")


def compute_stats(
    papers: dict[PaperId, PaperMetadata],
    citation_data: CitationData,
) -> DatasetStats:
    """Compute summary statistics for the downloaded dataset.

    Args:
        papers: All downloaded papers keyed by paper ID.
        citation_data: References and citations for each paper.

    Returns:
        DatasetStats with counts broken down by availability, year, and venue.
    """
    stats = DatasetStats()
    stats.total_papers = len(papers)

    by_year: dict[int, int] = defaultdict(int)
    by_venue: dict[str, int] = defaultdict(int)

    for paper in papers.values():
        if paper.abstract:
            stats.papers_with_abstract += 1
        if paper.open_access_pdf:
            stats.papers_with_pdf += 1
        if paper.year:
            by_year[paper.year] += 1
        if paper.venue:
            by_venue[normalise_venue(paper.venue)] += 1

    stats.by_year = dict(sorted(by_year.items()))
    stats.by_venue = dict(sorted(by_venue.items(), key=lambda x: -x[1]))

    for refs in citation_data.references.values():
        if refs:
            stats.papers_with_references += 1
            stats.total_references += len(refs)

    for cites in citation_data.citations.values():
        stats.total_citations += len(cites)

    return stats


async def _step_fetch_papers(
    api: SemanticScholarAPI,
    conferences: list[ConferencePattern],
    year_range: tuple[int, int],
) -> tuple[dict[PaperId, PaperMetadata], list[str]]:
    """Step 1: Fetch papers from all conferences and log a summary.

    Args:
        api: Authenticated Semantic Scholar API client.
        conferences: Conference definitions to query.
        year_range: Inclusive (start_year, end_year) publication year filter.

    Returns:
        Tuple of (papers, failed_fetches) as returned by fetch_papers_for_conferences.
    """
    logger.info("[bold]Step 1: Fetching papers...[/bold]")
    papers, failed_fetches = await fetch_papers_for_conferences(
        api, conferences, year_range
    )
    logger.info(f"[green]Found {len(papers)} unique papers[/green]\n")
    return papers, failed_fetches


async def _step_fetch_citations(
    api: SemanticScholarAPI,
    paper_ids: list[PaperId],
    skip: bool,
) -> CitationData:
    """Step 2: Fetch citation data for papers, or return empty data if skipped.

    Args:
        api: Authenticated Semantic Scholar API client.
        paper_ids: IDs of papers for which to fetch references and citations.
        skip: If True, return empty CitationData without making any API calls.

    Returns:
        CitationData with references and incoming citations for each paper.
    """
    if skip:
        return CitationData(references={}, citations={})

    logger.info("[bold]Step 2: Fetching citation data...[/bold]")
    citation_data = await fetch_citations_for_papers(
        api, paper_ids, fetch_incoming=True
    )
    logger.info(
        f"[green]Fetched references for {len(citation_data.references)} papers[/green]\n"
    )
    return citation_data


def _step_compute_graph(
    citation_data: CitationData,
    skip: bool,
    min_overlap: int,
) -> tuple[list[PaperPairScore], list[PaperPairScore]]:
    """Step 3: Compute co-citation and bibliographic coupling graphs.

    Args:
        citation_data: References and incoming citations for each paper.
        skip: If True, return empty lists without computing anything.
        min_overlap: Minimum shared-paper count to include a pair.

    Returns:
        Tuple of (co_citations, bib_coupling) scored pair lists.
    """
    if skip or not citation_data.references or not citation_data.citations:
        return [], []

    logger.info("[bold]Step 3: Computing citation graph...[/bold]")
    co_citations = compute_co_citations(
        citation_data.citations, min_overlap=min_overlap
    )
    logger.info(f"[green]Found {len(co_citations)} co-citation pairs[/green]")

    bib_coupling = compute_bibliographic_coupling(
        citation_data.references, min_overlap=min_overlap
    )
    logger.info(
        f"[green]Found {len(bib_coupling)} bibliographic coupling pairs[/green]\n"
    )
    return co_citations, bib_coupling


def _step_create_splits(
    papers: dict[PaperId, PaperMetadata],
    train_years: tuple[int, int],
    dev_year: int,
    test_year: int,
) -> DatasetSplits:
    """Step 4: Create train/dev/test splits by year and log counts.

    Args:
        papers: All downloaded papers keyed by paper ID.
        train_years: Inclusive (start_year, end_year) range for training.
        dev_year: Publication year for the development set.
        test_year: Publication year for the test set.

    Returns:
        DatasetSplits with paper IDs partitioned into train, dev, and test.
    """
    logger.info("[bold]Step 4: Creating train/dev/test splits...[/bold]")
    splits = create_splits(papers, train_years, dev_year, test_year)
    logger.info(
        f"  Train: {len(splits.train)} papers ({train_years[0]}-{train_years[1]})"
    )
    logger.info(f"  Dev: {len(splits.dev)} papers ({dev_year})")
    logger.info(f"  Test: {len(splits.test)} papers ({test_year})\n")
    return splits


@app.command()
def download(
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for dataset",
        ),
    ] = Path("data"),
    start_year: Annotated[
        int,
        typer.Option(
            "--start-year",
            help="Start year for papers (inclusive)",
        ),
    ] = 2020,
    end_year: Annotated[
        int,
        typer.Option(
            "--end-year",
            help="End year for papers (inclusive)",
        ),
    ] = 2025,
    train_end_year: Annotated[
        int,
        typer.Option(
            "--train-end",
            help="Last year for training set",
        ),
    ] = 2023,
    dev_year: Annotated[
        int,
        typer.Option(
            "--dev-year",
            help="Year for dev set",
        ),
    ] = 2024,
    test_year: Annotated[
        int,
        typer.Option(
            "--test-year",
            help="Year for test set",
        ),
    ] = 2025,
    conferences: Annotated[
        list[str] | None,
        typer.Option(
            "--conference",
            "-c",
            help="Specific conferences to include (can be repeated)",
        ),
    ] = None,
    skip_citations: Annotated[
        bool,
        typer.Option(
            "--skip-citations",
            help="Skip fetching citation data (faster)",
        ),
    ] = False,
    skip_graph: Annotated[
        bool,
        typer.Option(
            "--skip-graph",
            help="Skip computing co-citations and bibliographic coupling",
        ),
    ] = False,
    min_overlap: Annotated[
        int,
        typer.Option(
            "--min-overlap",
            help="Minimum overlap for co-citation/coupling pairs",
        ),
    ] = 2,
) -> None:
    """Download S2ORC dataset for generative retrieval pipeline.

    Downloads papers from specified conferences, fetches citation relationships,
    and creates train/dev/test splits.

    Requires SEMANTIC_SCHOLAR_API_KEY environment variable.
    """
    api_key = require_env("SEMANTIC_SCHOLAR_API_KEY")

    selected_conferences = require_conferences(conferences)

    logger.info(
        f"[bold cyan]S2ORC Dataset Download[/bold cyan]\n"
        f"  Conferences: {len(selected_conferences)}\n"
        f"  Years: {start_year}-{end_year}\n"
        f"  Train: {start_year}-{train_end_year}\n"
        f"  Dev: {dev_year}, Test: {test_year}"
    )

    async def run() -> DownloadResult:
        async with SemanticScholarAPI(api_key=api_key) as api:
            papers, failed_fetches = await _step_fetch_papers(
                api, selected_conferences, (start_year, end_year)
            )
            citation_data = await _step_fetch_citations(
                api, list(papers.keys()), skip=skip_citations
            )
            co_citations, bib_coupling = _step_compute_graph(
                citation_data, skip=skip_graph, min_overlap=min_overlap
            )
            splits = _step_create_splits(
                papers, (start_year, train_end_year), dev_year, test_year
            )
            return DownloadResult(
                papers=papers,
                citation_data=citation_data,
                co_citations=co_citations,
                bib_coupling=bib_coupling,
                splits=splits,
                failed_fetches=failed_fetches,
            )

    # Run async portion
    result = asyncio.run(run())

    # Step 5: Save everything (sync I/O)
    logger.info("[bold]Step 5: Saving dataset...[/bold]")
    save_dataset(
        output_dir,
        result.papers,
        result.citation_data,
        result.co_citations,
        result.bib_coupling,
        result.splits,
    )

    # Print statistics
    stats = compute_stats(result.papers, result.citation_data)
    logger.info(
        f"\n[bold cyan]Dataset Statistics:[/bold cyan]\n"
        f"  Total papers: {stats.total_papers}\n"
        f"  With abstract: {stats.papers_with_abstract}\n"
        f"  With PDF: {stats.papers_with_pdf}\n"
        f"  With references: {stats.papers_with_references}\n"
        f"  Total references: {stats.total_references}\n"
        f"  Total citations: {stats.total_citations}"
    )

    year_lines = "\n".join(
        f"  {year}: {count}" for year, count in stats.by_year.items()
    )
    logger.info(f"\n[bold cyan]Papers by Year:[/bold cyan]\n{year_lines}")

    # Save stats
    stats_file = output_dir / "stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)
    logger.info(f"\n[green]Dataset saved to {output_dir}[/green]")

    if result.failed_fetches:
        unique_failed = sorted(set(result.failed_fetches))
        total_conferences = len(selected_conferences)
        logger.warning(
            f"{len(unique_failed)}/{total_conferences} conference fetches failed."
        )
        logger.warning("Retry this command to fetch remaining items.")
        logger.warning(f"Failed conferences: {', '.join(unique_failed)}")


@app.command()
def resume(
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory containing partial dataset",
        ),
    ] = Path("data"),
) -> None:
    """Resume citation fetching for papers that don't have references yet.

    Useful if the initial download was interrupted or run with --skip-citations.
    """
    api_key = require_env("SEMANTIC_SCHOLAR_API_KEY")

    # Load existing data
    papers_file = resolve_jsonl(output_dir / "papers" / "metadata.jsonl.gz")
    refs_file = resolve_jsonl(output_dir / "citations" / "references.jsonl.gz")

    if not papers_file.exists():
        die(f"{papers_file} not found")

    # Load paper IDs
    paper_ids = [PaperId(r["paper_id"]) for r in iter_jsonl(papers_file)]
    logger.info(f"[cyan]Found {len(paper_ids)} papers[/cyan]")

    # Load existing references
    existing_refs: set[PaperId] = set()
    if refs_file.exists():
        existing_refs = {PaperId(r["paper_id"]) for r in iter_jsonl(refs_file)}

    # Find papers without references
    missing = [pid for pid in paper_ids if pid not in existing_refs]
    logger.info(f"[cyan]Papers needing citation fetch: {len(missing)}[/cyan]")

    if not missing:
        logger.info("[green]All papers already have citation data[/green]")
        return

    async def run() -> CitationData:
        api = SemanticScholarAPI(api_key=api_key)
        async with api:
            return await fetch_citations_for_papers(api, missing, fetch_incoming=True)

    citation_data = asyncio.run(run())

    # Append to existing files (sync I/O)
    citations_dir = output_dir / "citations"
    with open_auto(citations_dir / "references.jsonl.zst", "a") as f:
        for paper_id, refs in citation_data.references.items():
            data = {"paper_id": paper_id, "references": list(refs)}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    with open_auto(citations_dir / "citations.jsonl.zst", "a") as f:
        for paper_id, cites in citation_data.citations.items():
            data = {"paper_id": paper_id, "citations": list(cites)}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    logger.info(
        f"[green]Added citations for {len(citation_data.references)} papers[/green]"
    )


if __name__ == "__main__":
    app()
