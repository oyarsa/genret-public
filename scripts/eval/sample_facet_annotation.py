r"""Sample papers' facet items for groundedness annotation.

Supports the facet-accuracy human validation study: an annotator marks
whether each sampled facet item is `valid` (grounded in the paper it was
extracted from) or `invalid`, with a free-text rationale, independent of
the structural well-formedness checks already in place. Each row carries
the paper's abstract and introduction alongside the sampled facet item, so
an annotator can judge groundedness without having to open the source
paper separately.

Sampling procedure: draw `--n-papers` distinct papers, without replacement,
from every paper that has at least one item in *all four* facet types.
Each sampled paper contributes exactly one item per facet type (4 rows),
so total rows = `--n-papers` x 4. A paper's 4 rows stay adjacent in the
output (in `problems, methods, results, contributions` order), so an
annotator judges all four with the paper still fresh in mind, rather than
having its rows scattered across the sheet.

Partial-overlap design, generalised to any number of annotators: `--overlap-papers`
of the `--n-papers` sampled papers are shared across *every* annotator's
sheet (the only rows inter-annotator agreement can be computed on); the
remainder is split as evenly as possible and exclusively across them, to
extend total coverage beyond what full duplication would allow. One sheet
is written per `--output` occurrence -- pass it once per annotator -- each
the shared core papers' rows plus that annotator's own exclusive papers'
rows, with paper-block order independently shuffled per sheet (rows within
a block stay in facet-type order).

Usage (3 annotators). Output files belong in `output/genret/facet_annotation/`
(see `docs/reference/20260728-facet-annotation-study-setup.md` for the full
inventory and rationale behind this study's setup):
```
uv run python scripts/eval/sample_facet_annotation.py \
    --facets output/genret/facets_pruned_zeroview_1k.jsonl \
    --paper-graphs data/s2orc-nlp/processed/paper_graphs.jsonl.zst \
    --n-papers 60 \
    --overlap-papers 10 \
    --output output/genret/facet_annotation/facet_annotation_a.csv \
    --output output/genret/facet_annotation/facet_annotation_b.csv \
    --output output/genret/facet_annotation/facet_annotation_c.csv
```

A single `--output` is also valid (e.g. for a quick test file):
```
uv run python scripts/eval/sample_facet_annotation.py \
    --facets output/genret/facets_pruned_zeroview_1k.jsonl \
    --paper-graphs data/s2orc-nlp/processed/paper_graphs.jsonl.zst \
    --n-papers 1 \
    --output output/genret/facet_annotation/facet_annotation_test.csv
```

Requires `SEMANTIC_SCHOLAR_API_KEY` to be set: `paper_url` is fetched live
from the Semantic Scholar API rather than constructed locally, since the
website's paper URLs are keyed by its internal `paperId`, not by corpus ID.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import itertools
import random
from collections.abc import Set
from pathlib import Path
from typing import Annotated

import typer

from genret.paper_parsing import FACET_NAMES, Facets, load_facets_map
from genret.processing.facet_extract_cmd import iter_paper_graphs
from genret.processing.term_extraction import (
    extract_abstract_text,
    extract_introduction_text,
)
from genret.s2.api import SemanticScholarAPI
from genret.types import CorpusId
from genret.utils import require_env, track_progress

S2_BATCH_SIZE = 500
"""Semantic Scholar batch-endpoint limit."""

CSV_FIELDNAMES = (
    "item_hash",
    "paper_title",
    "paper_url",
    "paper_abstract",
    "paper_introduction",
    "facet_type",
    "facet_item",
    "validity",
    "rationale",
)
"""Output CSV column order."""


async def _fetch_paper_urls(corpusids: Set[CorpusId]) -> dict[CorpusId, str]:
    """Fetch canonical Semantic Scholar URLs for specific papers.

    The Semantic Scholar website's paper URLs are keyed by its internal
    `paperId` hash, not by corpus ID -- constructing a `.../CorpusID:{id}`
    URL locally does not reliably resolve, so the real URL has to come from
    the API.

    Args:
        corpusids: Corpus IDs to look up.

    Returns:
        Mapping from corpus ID to its canonical Semantic Scholar URL. Papers
        the API doesn't return (not found, or a failed batch) are omitted.
    """
    api_key = require_env("SEMANTIC_SCHOLAR_API_KEY")
    ids = list(corpusids)
    urls: dict[CorpusId, str] = {}
    async with SemanticScholarAPI(api_key=api_key) as api:
        for batch in itertools.batched(ids, S2_BATCH_SIZE, strict=False):
            s2_ids = [f"CorpusId:{cid}" for cid in batch]
            try:
                papers = await api.get_paper_batch(s2_ids, fields=["url"])
            except Exception as exc:
                print(f"Warning: URL lookup failed for {len(batch)} papers: {exc}")
                continue
            for cid, paper in zip(batch, papers, strict=True):
                if paper is not None and paper.url:
                    urls[cid] = paper.url
    return urls


def _load_paper_context(
    paper_graphs_path: Path, wanted_ids: Set[CorpusId]
) -> dict[CorpusId, tuple[str, str, str]]:
    """Load title, abstract, and introduction text for specific papers only.

    Args:
        paper_graphs_path: Path to the paper-graphs JSONL.
        wanted_ids: Corpus IDs to load; all other papers are skipped.

    Returns:
        Mapping from corpus ID to `(title, abstract, introduction)`.
    """
    context: dict[CorpusId, tuple[str, str, str]] = {}
    papers = iter_paper_graphs(paper_graphs_path, set(), set(wanted_ids))
    for paper in track_progress(papers, desc="Loading paper context..."):
        abstract = paper.abstract or extract_abstract_text(paper) or ""
        introduction = extract_introduction_text(paper) or ""
        context[paper.corpusid] = (paper.title.strip(), abstract, introduction)
    return context


def _build_row(
    corpusid: CorpusId,
    facet_type: str,
    facets_map: dict[CorpusId, Facets],
    context: dict[CorpusId, tuple[str, str, str]],
    urls: dict[CorpusId, str],
    rng: random.Random,
) -> dict[str, str]:
    """Build one annotation row for a (paper, facet type) pair.

    Args:
        corpusid: Paper corpus ID.
        facet_type: One of `FACET_NAMES`.
        facets_map: Mapping from corpus ID to that paper's `Facets`.
        context: Mapping from corpus ID to `(title, abstract, introduction)`.
        urls: Mapping from corpus ID to its Semantic Scholar URL.
        rng: Random number generator to use.

    Returns:
        One row dict, in CSV column order.
    """
    items = facets_map[corpusid].to_dict()[facet_type]
    item = rng.choice(items)
    item_hash = hashlib.sha256(f"{corpusid}|{facet_type}|{item}".encode()).hexdigest()[
        :12
    ]
    title, abstract, introduction = context.get(corpusid, ("", "", ""))
    return {
        "item_hash": item_hash,
        "paper_title": title,
        "paper_url": urls.get(corpusid, ""),
        "paper_abstract": abstract,
        "paper_introduction": introduction,
        "facet_type": facet_type,
        "facet_item": item,
        "validity": "",
        "rationale": "",
    }


def _build_paper_block(
    corpusid: CorpusId,
    facets_map: dict[CorpusId, Facets],
    context: dict[CorpusId, tuple[str, str, str]],
    urls: dict[CorpusId, str],
    rng: random.Random,
) -> list[dict[str, str]]:
    """Build one paper's four annotation rows, one per facet type, in order.

    Args:
        corpusid: Paper corpus ID.
        facets_map: Mapping from corpus ID to that paper's `Facets`.
        context: Mapping from corpus ID to `(title, abstract, introduction)`.
        urls: Mapping from corpus ID to its Semantic Scholar URL.
        rng: Random number generator to use.

    Returns:
        Four row dicts, in `FACET_NAMES` order.
    """
    return [
        _build_row(corpusid, facet_type, facets_map, context, urls, rng)
        for facet_type in FACET_NAMES
    ]


def _split_evenly(items: list[CorpusId], n_groups: int) -> list[list[CorpusId]]:
    """Split items into `n_groups` contiguous chunks whose sizes differ by <=1.

    Args:
        items: Items to split.
        n_groups: Number of chunks to produce.

    Returns:
        `n_groups` chunks, in order, covering `items` exactly once.
    """
    n = len(items)
    return [
        items[(i * n) // n_groups : ((i + 1) * n) // n_groups] for i in range(n_groups)
    ]


def sample_annotation_rows(
    facets_path: Path,
    paper_graphs_path: Path,
    n_papers: int,
    n_overlap: int,
    n_annotators: int,
    rng: random.Random,
) -> list[list[dict[str, str]]]:
    """Sample papers for a partial-overlap groundedness annotation study.

    Args:
        facets_path: Path to the facet-extraction JSONL.
        paper_graphs_path: Path to the paper-graphs JSONL, for paper context.
        n_papers: Distinct papers to sample; each contributes 4 rows.
        n_overlap: How many of those papers every annotator rates (for IAA);
            the remainder is split as evenly as possible and exclusively
            across the annotators.
        n_annotators: Number of annotator sheets to produce.
        rng: Random number generator to use.

    Returns:
        One row list per annotator (length `n_annotators`) -- each
        annotator's full sheet (shared core papers' rows plus their own
        exclusive papers' rows), with paper-block order independently
        shuffled.
    """
    if n_overlap > n_papers:
        raise ValueError(f"overlap ({n_overlap}) cannot exceed n_papers ({n_papers})")
    if n_annotators < 1:
        raise ValueError("need at least one annotator")

    facets_map = load_facets_map(facets_path)

    papers_pool: list[CorpusId] = []
    for corpusid, facets in facets_map.items():
        facets_by_type = facets.to_dict()
        if all(facets_by_type[facet_type] for facet_type in FACET_NAMES):
            papers_pool.append(corpusid)

    if n_papers > len(papers_pool):
        raise ValueError(
            f"Requested {n_papers} papers but only {len(papers_pool)} have "
            "items in all four facet types."
        )

    sampled_papers = rng.sample(papers_pool, n_papers)
    core_papers, remaining = sampled_papers[:n_overlap], sampled_papers[n_overlap:]
    exclusive_groups = _split_evenly(remaining, n_annotators)

    context = _load_paper_context(paper_graphs_path, set(sampled_papers))
    urls = asyncio.run(_fetch_paper_urls(set(sampled_papers)))

    core_blocks = [
        _build_paper_block(cid, facets_map, context, urls, rng) for cid in core_papers
    ]

    all_rows: list[list[dict[str, str]]] = []
    for group in exclusive_groups:
        blocks = [[dict(row) for row in block] for block in core_blocks] + [
            _build_paper_block(cid, facets_map, context, urls, rng) for cid in group
        ]
        rng.shuffle(blocks)
        all_rows.append([row for block in blocks for row in block])

    return all_rows


def write_annotation_csv(rows: list[dict[str, str]], output: Path) -> None:
    """Write sampled annotation rows to a CSV file.

    Args:
        rows: Rows produced by `sample_annotation_rows`.
        output: Destination CSV path.
    """
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    facets: Annotated[
        Path, typer.Option(show_default=False, help="Facet-extraction JSONL.")
    ],
    paper_graphs: Annotated[
        Path, typer.Option(show_default=False, help="Paper-graphs JSONL, for context.")
    ],
    n_papers: Annotated[
        int,
        typer.Option(help="Distinct papers to sample; each contributes 4 rows."),
    ],
    output: Annotated[
        list[Path],
        typer.Option(
            "--output",
            help="One CSV path per annotator; repeat the flag per annotator.",
        ),
    ],
    overlap_papers: Annotated[
        int | None,
        typer.Option(
            help="Papers shared across every annotator, for IAA. Required when "
            "--n-papers > 1; ignored (forced to 0) when --n-papers is 1."
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
) -> None:
    """Sample facet items and write each annotator's partial-overlap CSV."""
    if n_papers == 1:
        if overlap_papers not in (None, 0):
            print(
                "Warning: --overlap-papers is ignored when --n-papers is 1 "
                "(there's nothing to share with a single paper); using 0."
            )
        overlap_papers = 0
    elif overlap_papers is None:
        raise ValueError("--overlap-papers is required when --n-papers > 1.")

    rng = random.Random(seed)
    all_rows = sample_annotation_rows(
        facets, paper_graphs, n_papers, overlap_papers, len(output), rng
    )

    for path, rows in zip(output, all_rows, strict=True):
        missing = sum(
            1 for row in rows if not row["paper_title"] or not row["paper_abstract"]
        )
        if missing:
            print(
                f"Warning: {missing}/{len(rows)} rows for {path} are missing "
                "a title or abstract."
            )
        write_annotation_csv(rows, path)
        print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    app()
