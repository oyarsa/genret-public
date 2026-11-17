# ruff: noqa: PLC0415
"""CLI for per-bullet DocID usage analysis (``glen analyze-bullet-usage``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from genret.paper_parsing import FacetRecordType
from genret.utils import load_ids_from_file

logger = logging.getLogger(__name__)


def cmd_analyze_bullet_usage(
    pairs: Annotated[
        Path,
        typer.Option("--pairs", help="Training pairs JSONL.", show_default=False),
    ],
    facets: Annotated[
        Path,
        typer.Option(
            "--facets", help="JSONL with per-paper facet records.", show_default=False
        ),
    ],
    paper_graphs: Annotated[
        Path,
        typer.Option(
            "--paper-graphs",
            help="JSONL with per-paper metadata (titles).",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o", help="Per-bullet JSONL output path.", show_default=False
        ),
    ],
    train_split: Annotated[
        Path | None,
        typer.Option("--train-split", help="Restrict paper_a to this split."),
    ] = None,
    graph_sampling: Annotated[
        str,
        typer.Option("--graph-sampling", help="'uniform' or 'weighted'."),
    ] = "weighted",
    candidates_dir: Annotated[
        Path | None,
        typer.Option(
            "--candidates-dir",
            help="facet_candidates_*.npz dir (required for weighted sampling).",
        ),
    ] = None,
    graph_sampling_max_copies: Annotated[
        int, typer.Option("--graph-sampling-max-copies")
    ] = 5,
    min_target_views: Annotated[int, typer.Option("--min-target-views")] = 3,
    target_bullet: Annotated[
        str, typer.Option("--target-bullet", help="'random'|'longest'|'cartesian'.")
    ] = "random",
    source_bullets: Annotated[
        str, typer.Option("--source-bullets", help="'all' or 'longest'.")
    ] = "all",
    seed: Annotated[int, typer.Option("--seed")] = 42,
    smoothed_bullets: Annotated[
        Path | None,
        typer.Option(
            "--smoothed-bullets",
            help="Graph-smoothed map (glen smooth-bullets output): adds "
            "neighbourhood-attestation fraction + usage correlations.",
        ),
    ] = None,
    decoded_docids: Annotated[
        Path | None,
        typer.Option(
            "--decoded-docids",
            help="Side JSONL from 'glen evaluate --record-docids': adds "
            "per-bullet decode-hit counts.",
        ),
    ] = None,
    tokeniser_model: Annotated[
        str,
        typer.Option(
            "--tokeniser-model",
            help="Tokeniser for the decode-hit join (must match the eval run).",
        ),
    ] = "meta-llama/Llama-3.2-1B-Instruct",
    max_docid_tokens: Annotated[int, typer.Option("--max-docid-tokens")] = 12,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
) -> None:
    """Report training views (+ optional decode hits) per bullet DocID.

    Rebuilds the training dataset with the given config to count views per
    (paper, facet, bullet), then joins optional decode-hit counts and
    graph-attestation fractions. Writes a per-bullet JSONL and a
    ``.summary.json`` with per-facet distributions and correlations.
    """
    from genret.training.glen_bullet_usage import (
        analyze_bullet_usage,
        save_usage_report,
    )

    tokeniser = None
    if decoded_docids is not None:
        from genret._shed.transformers import load_tokenizer

        tokeniser = load_tokenizer(tokeniser_model)

    split_ids = load_ids_from_file(train_split) if train_split is not None else None
    records, summary = analyze_bullet_usage(
        pairs_path=pairs,
        facets_path=facets,
        paper_graphs_path=paper_graphs,
        facets_type=facets_type,
        split_ids=split_ids,
        graph_sampling=cast(Literal["uniform", "weighted"], graph_sampling),
        candidates_dir=candidates_dir,
        graph_sampling_max_copies=graph_sampling_max_copies,
        min_target_views=min_target_views,
        target_bullet=cast(Literal["random", "longest", "cartesian"], target_bullet),
        source_bullets=cast(Literal["all", "longest"], source_bullets),
        seed=seed,
        smoothed_bullets_path=smoothed_bullets,
        decoded_docids_path=decoded_docids,
        tokeniser=tokeniser,
        max_docid_tokens=max_docid_tokens,
    )
    save_usage_report(records, summary, output)

    for facet, s in summary.items():
        parts = [
            f"{facet:15s} n={int(s['n_bullets']):5d}",
            f"median views={s['views_median']:.0f}",
            f"p90={s['views_p90']:.0f}",
            f"zero={s['share_zero_views']:.1%}",
            f"floor={s['share_at_floor']:.1%}",
        ]
        if "spearman_views_vs_attested" in s:
            parts.append(f"rho(views,attest)={s['spearman_views_vs_attested']:+.3f}")
        if "share_never_decoded" in s:
            parts.append(f"never-decoded={s['share_never_decoded']:.1%}")
        if "spearman_hits_vs_attested" in s:
            parts.append(f"rho(hits,attest)={s['spearman_hits_vs_attested']:+.3f}")
        typer.echo("  ".join(parts))


def register(app: typer.Typer) -> None:
    """Attach usage-analysis commands to the GLEN typer app."""
    app.command(name="analyze-bullet-usage")(cmd_analyze_bullet_usage)


__all__ = ["cmd_analyze_bullet_usage", "register"]
