# pyright: basic
# ruff: noqa: PLC0415  Enable function-level imports
"""Index-building GLEN commands: DocID tries, smoothed bullets, title index."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from genret.paper_parsing import FacetRecordType

logger = logging.getLogger(__name__)


def cmd_build_glen_docids(
    facets: Annotated[
        Path,
        typer.Option(
            "--facets", help="JSONL with per-paper facet records.", show_default=False
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory to write the GLEN index.",
            show_default=False,
        ),
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.1-8B-Instruct",
    max_docid_tokens: Annotated[
        int, typer.Option("--max-docid-tokens", help="Cap on DocID token length.")
    ] = 12,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    smoothed_bullets: Annotated[
        Path | None,
        typer.Option(
            "--smoothed-bullets",
            help="Graph-smoothed DocID map (JSONL from 'glen smooth-bullets'). "
            "When set, each bullet's DocID text is its smoothed version — "
            "must match the map used for training targets.",
        ),
    ] = None,
) -> None:
    """Extract per-facet DocID tries from the corpus and write to disk.

    Each paper's DocID for a given facet is the normalised text of its
    facet bullets tokenised to at most ``--max-docid-tokens`` tokens.
    """
    from genret._shed.transformers import load_tokenizer
    from genret.training.glen_docids import build_glen_index, save_glen_index

    smoothed_map = None
    if smoothed_bullets is not None:
        from genret.training.glen_graph_smoothing import load_smoothed_bullets

        smoothed_map = load_smoothed_bullets(smoothed_bullets)

    tokeniser = load_tokenizer(model_name)
    index = build_glen_index(
        facets,
        tokeniser,
        facets_type=facets_type,
        max_tokens=max_docid_tokens,
        smoothed_bullets=smoothed_map,
    )
    save_glen_index(index, output)

    total_leaves = sum(len(v) for v in index.code_to_papers.values())
    typer.echo(
        f"GLEN index written to {output}: {total_leaves} leaves across 4 facets."
    )


def cmd_smooth_bullets(
    facets: Annotated[
        Path,
        typer.Option(
            "--facets", help="JSONL with per-paper facet records.", show_default=False
        ),
    ],
    candidates_dir: Annotated[
        Path,
        typer.Option(
            "--candidates-dir",
            help="Directory with facet_candidates_<facet>.npz (graph "
            "neighbourhood source).",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output JSONL path.", show_default=False),
    ],
    top_neighbours: Annotated[
        int,
        typer.Option(
            "--top-neighbours",
            help="Graph neighbours per paper (by scores_combined).",
        ),
    ] = 20,
    min_attest: Annotated[
        int,
        typer.Option(
            "--min-attest",
            help="Neighbours that must use a word for it to be kept.",
        ),
    ] = 2,
    min_words: Annotated[
        int,
        typer.Option(
            "--min-words",
            help="Minimum words per smoothed DocID (padded with the "
            "bullet's own words when attestation is too sparse).",
        ),
    ] = 4,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
) -> None:
    """Precompute graph-smoothed DocID text for every (paper, facet, bullet).

    Keeps each bullet's content words that the paper's graph neighbourhood
    also uses, preserving order. Output feeds ``build-docids
    --smoothed-bullets`` and ``train-bullets --smoothed-bullets``.
    """
    from genret.training.glen_graph_smoothing import build_smoothed_bullets

    stats = build_smoothed_bullets(
        facets,
        candidates_dir,
        output,
        top_neighbours=top_neighbours,
        min_attest=min_attest,
        min_words=min_words,
        facets_type=facets_type,
    )
    typer.echo(
        f"Smoothed {int(stats['n_bullets'])} bullets -> {output} "
        f"(mean kept-word fraction {stats['mean_kept_word_fraction']:.2f}, "
        f"verbatim fallback {stats['verbatim_fallback_fraction']:.1%})"
    )


def cmd_build_title_index(
    paper_graphs: Annotated[
        Path,
        typer.Option(
            "--paper-graphs",
            help="JSONL with per-paper metadata (must include 'corpusid' and 'title').",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory to write the title GLEN index.",
            show_default=False,
        ),
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.1-8B-Instruct",
    max_title_tokens: Annotated[
        int,
        typer.Option(
            "--max-title-tokens",
            help="Maximum token length for title DocIDs.",
        ),
    ] = 32,
) -> None:
    """Build a title-based GLEN index (D5 sanity check).

    Each paper's DocID is its normalised title tokenised to at most
    ``--max-title-tokens`` tokens.  The index is stored under all four
    facet keys so the existing evaluate command works unchanged.
    """
    from genret._shed.transformers import load_tokenizer
    from genret.training.glen_docids import save_glen_index
    from genret.training.glen_title_index import build_title_index

    tokeniser = load_tokenizer(model_name)
    index = build_title_index(paper_graphs, tokeniser, max_tokens=max_title_tokens)
    save_glen_index(index, output)

    total_leaves = sum(len(v) for v in index.code_to_papers.values()) // 4
    typer.echo(f"Title GLEN index written to {output}: {total_leaves} title DocIDs.")


def register(app: typer.Typer) -> None:
    """Attach DocID / index building commands to the GLEN typer app."""
    app.command(name="build-docids")(cmd_build_glen_docids)
    app.command(name="smooth-bullets")(cmd_smooth_bullets)
    app.command(name="build-title-index")(cmd_build_title_index)


__all__ = [
    "cmd_build_glen_docids",
    "cmd_build_title_index",
    "cmd_smooth_bullets",
    "register",
]
