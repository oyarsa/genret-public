"""Script to count papers from *ACL and ML/AI/NLP conferences using Semantic Scholar API.

This script searches for papers from major conferences and saves both individual
paper metadata and aggregated counts by year and venue.
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn

from genret.config import (
    ConferencePattern,
    Paper,
    normalise_venue,
    require_conferences,
)
from genret.s2.api import SemanticScholarAPI, extract_pdf_url
from genret.s2.conference_fetch import FetchPapersResult, fetch_papers_by_venues
from genret.s2.types import S2Paper
from genret.types import PaperId
from genret.utils import make_app, require_env

logger = logging.getLogger(__name__)

app = make_app("Count papers from conferences using Semantic Scholar API")


async def get_papers_by_venues(
    api: SemanticScholarAPI,
    venues: list[str],
    year_range: tuple[int, int] | None = None,
) -> FetchPapersResult[Paper]:
    """Get all papers from multiple venues using bulk API.

    Args:
        api: SemanticScholarAPI instance
        venues: List of venue names to search for
        year_range: Optional (start_year, end_year) tuple

    Returns:
        Fetch result containing papers and whether any page failed
    """

    def parse_item(item: S2Paper) -> Paper:
        pdf_url, pdf_source = extract_pdf_url(item)
        return Paper(
            paper_id=PaperId(item.paper_id),
            title=item.title,
            year=item.year,
            venue=item.venue,
            citation_count=item.citation_count,
            fields_of_study=item.fields_of_study or [],
            url=item.url,
            open_access_pdf=pdf_url,
            pdf_source=pdf_source,
        )

    def on_error(venue_query: str, exc: Exception) -> None:
        logger.warning(f"Error for venues '{venue_query}': {exc}")

    return await fetch_papers_by_venues(
        api,
        venues,
        parse_item,
        year_range=year_range,
        on_error=on_error,
    )


@dataclass
class ConferencePapers:
    """Papers fetched for a specific conference."""

    conference_name: str
    papers: list[Paper]
    had_error: bool = False


async def fetch_conference_papers(
    api: SemanticScholarAPI,
    conference: ConferencePattern,
    year_range: tuple[int, int] | None,
    progress: Progress,
    task_id: TaskID,
) -> ConferencePapers:
    """Fetch papers for a specific conference pattern.

    Args:
        api: SemanticScholarAPI instance
        conference: Conference pattern to search for
        year_range: Optional year range to filter
        progress: Rich progress bar
        task_id: Task ID for progress updates

    Returns:
        ConferencePapers with conference name and matching papers
    """
    progress.update(task_id, description=f"[cyan]Fetching {conference.name}...[/cyan]")

    # Fetch all venue patterns in a single batched request
    papers = await get_papers_by_venues(
        api,
        venues=conference.venue_patterns,
        year_range=year_range,
    )

    # Deduplicate by paper_id
    seen_ids: set[str] = set()
    unique_papers: list[Paper] = []
    for paper in papers.papers:
        if paper.paper_id not in seen_ids:
            seen_ids.add(paper.paper_id)
            unique_papers.append(paper)

    progress.update(
        task_id,
        description=f"[green]✓ {conference.name}: {len(unique_papers)} papers[/green]",
    )

    return ConferencePapers(
        conference_name=conference.name,
        papers=unique_papers,
        had_error=papers.had_error,
    )


async def fetch_all_conferences(
    api: SemanticScholarAPI,
    conferences: list[ConferencePattern],
    year_range: tuple[int, int] | None,
) -> list[ConferencePapers]:
    """Fetch papers for all conferences concurrently with rate limiting.

    Args:
        api: SemanticScholarAPI instance
        conferences: List of conference patterns to search
        year_range: Optional year range to filter

    Returns:
        List of ConferencePapers for each conference
    """
    async with api:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            tasks: list[asyncio.Task[ConferencePapers]] = []
            for conference in conferences:
                task_id = progress.add_task(f"[cyan]{conference.name}[/cyan]")
                tasks.append(
                    asyncio.create_task(
                        fetch_conference_papers(
                            api, conference, year_range, progress, task_id
                        )
                    )
                )

            results = await asyncio.gather(*tasks)

    return list(results)


@app.command()
def main(
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output JSON file for individual paper records",
        ),
    ],
    aggregated_file: Annotated[
        Path,
        typer.Option(
            "--aggregated",
            "-a",
            help="Output JSON file for aggregated counts",
        ),
    ] = Path("output/aggregated_counts.json"),
    start_year: Annotated[
        int,
        typer.Option(
            "--start-year",
            help="Start year for filtering (inclusive). Set to 0 for no filter.",
        ),
    ] = 2020,
    end_year: Annotated[
        int,
        typer.Option(
            "--end-year",
            help="End year for filtering (inclusive). Set to 0 for no filter.",
        ),
    ] = 2025,
    conferences: Annotated[
        list[str] | None,
        typer.Option(
            "--conference",
            "-c",
            help="Specific conferences to search (can be repeated). If not specified, searches all.",
        ),
    ] = None,
) -> None:
    """Count papers from *ACL and ML/AI/NLP conferences using Semantic Scholar API.

    Requires SEMANTIC_SCHOLAR_API_KEY environment variable for API access (optional but recommended
    for higher rate limits).
    """
    # Get API key from environment
    api_key = require_env("SEMANTIC_SCHOLAR_API_KEY")

    # Create API client
    api = SemanticScholarAPI(api_key=api_key)

    # Determine year range
    year_range: tuple[int, int] | None = None
    if start_year or end_year:
        start = start_year or 1900
        end = end_year or 2030
        year_range = (start, end)
        logger.info(f"[cyan]Filtering papers from {start} to {end}[/cyan]")

    # Filter conferences if specified
    selected_conferences = require_conferences(conferences)

    logger.info(f"[cyan]Searching {len(selected_conferences)} conferences...[/cyan]")

    # Fetch papers for all conferences
    conference_results = asyncio.run(
        fetch_all_conferences(api, selected_conferences, year_range)
    )
    failed_fetches = sorted(
        conf_result.conference_name
        for conf_result in conference_results
        if conf_result.had_error
    )

    # Collect all papers and remove duplicates
    unique_papers: dict[str, Paper] = {}
    for conf_papers in conference_results:
        for paper in conf_papers.papers:
            if paper.paper_id not in unique_papers:
                unique_papers[paper.paper_id] = paper

    papers_with_pdf_count = sum(1 for p in unique_papers.values() if p.open_access_pdf)
    logger.info(f"\n[green]Total unique papers found: {len(unique_papers)}[/green]")
    logger.info(f"[green]Papers with open access PDF: {papers_with_pdf_count}[/green]")

    # Save individual paper records
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(p) for p in unique_papers.values()],
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"[green]Saved paper records to {output_file}[/green]")

    # Aggregate by year and venue
    by_year: dict[int, int] = defaultdict(int)
    by_venue: dict[str, int] = defaultdict(int)
    by_year_venue: dict[tuple[int, str], int] = defaultdict(int)
    by_year_venue_with_pdf: dict[tuple[int, str], int] = defaultdict(int)
    papers_with_pdf = 0

    for paper in unique_papers.values():
        has_pdf = bool(paper.open_access_pdf)
        if has_pdf:
            papers_with_pdf += 1

        if paper.year:
            by_year[paper.year] += 1
        if paper.venue:
            venue = normalise_venue(paper.venue)
            by_venue[venue] += 1
            if paper.year:
                by_year_venue[(paper.year, venue)] += 1
                if has_pdf:
                    by_year_venue_with_pdf[(paper.year, venue)] += 1

    # Prepare aggregated output
    aggregated = {
        "total_papers": len(unique_papers),
        "papers_with_pdf": papers_with_pdf,
        "by_year": dict(sorted(by_year.items())),
        "by_venue": dict(sorted(by_venue.items(), key=lambda x: x[1], reverse=True)),
        "by_year_venue": {
            f"{year}|{venue}": count
            for (year, venue), count in sorted(by_year_venue.items())
        },
        "by_year_venue_with_pdf": {
            f"{year}|{venue}": count
            for (year, venue), count in sorted(by_year_venue_with_pdf.items())
        },
    }

    # Save aggregated counts
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)
    with open(aggregated_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    logger.info(f"[green]Saved aggregated counts to {aggregated_file}[/green]")

    # Display summary
    year_lines = "\n".join(
        f"  {year}: {by_year[year]} papers" for year in sorted(by_year.keys())
    )
    logger.info(f"\n[bold cyan]Summary by Year:[/bold cyan]\n{year_lines}")

    venue_lines = "\n".join(
        f"  {venue}: {count} papers"
        for venue, count in sorted(by_venue.items(), key=lambda x: x[1], reverse=True)[
            :20
        ]
    )
    logger.info(f"\n[bold cyan]Summary by Venue (Top 20):[/bold cyan]\n{venue_lines}")

    if failed_fetches:
        total_conferences = len(selected_conferences)
        logger.warning(
            f"\n{len(failed_fetches)}/{total_conferences} conference fetches failed."
        )
        logger.warning("Retry this command to fetch remaining items.")
        logger.warning(f"Failed conferences: {', '.join(failed_fetches)}")


if __name__ == "__main__":
    app()
