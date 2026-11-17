"""Baseline retrieval commands: bm25, rerank-retrieval, s2-recs, embedding."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from genret.processing.bm25_baseline import run_bm25_baseline, run_rerank_retrieval
from genret.processing.s2_recs_baseline import run_s2_recs_baseline

logger = logging.getLogger(__name__)


def cmd_bm25_baseline(
    facets: Annotated[
        Path,
        typer.Option("--facets", help="Path to facets JSONL file.", show_default=False),
    ],
    paper_graphs: Annotated[
        Path,
        typer.Option(
            "--paper-graphs",
            help="Path to paper_graphs JSONL file (for titles).",
            show_default=False,
        ),
    ],
    query_papers: Annotated[
        Path,
        typer.Option(
            "--query-papers",
            help="Path to query papers JSONL file.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to write retrieval results.",
            show_default=False,
        ),
    ],
    co_citations: Annotated[
        Path,
        typer.Option(
            "--co-citations",
            help="Path to co_citations JSONL file.",
            show_default=False,
        ),
    ],
    bib_coupling: Annotated[
        Path,
        typer.Option(
            "--bib-coupling",
            help="Path to bib_coupling JSONL file.",
            show_default=False,
        ),
    ],
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Number of candidates to retrieve per query."),
    ] = 50,
) -> None:
    """Run BM25 baseline retrieval and evaluate against citation ground truth.

    Indexes papers by title + facet items, retrieves candidates using BM25
    scoring, and computes Recall@K / MRR metrics.
    """
    run_bm25_baseline(
        facets_path=facets,
        paper_graphs_path=paper_graphs,
        query_papers_path=query_papers,
        output_path=output,
        co_cit_path=co_citations,
        bib_path=bib_coupling,
        top_n=top_n,
    )


def cmd_rerank_retrieval(
    retrieval: Annotated[
        Path,
        typer.Option(
            "--retrieval",
            help="Path to retrieval results JSONL.",
            show_default=False,
        ),
    ],
    facets: Annotated[
        Path,
        typer.Option("--facets", help="Path to facets JSONL file.", show_default=False),
    ],
    paper_graphs: Annotated[
        Path,
        typer.Option(
            "--paper-graphs",
            help="Path to paper_graphs JSONL file (for titles).",
            show_default=False,
        ),
    ],
    model_dir: Annotated[
        Path,
        typer.Option(
            "--model-dir",
            help="Path to the fine-tuned reranker checkpoint.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to write re-ranked results.",
            show_default=False,
        ),
    ],
    co_citations: Annotated[
        Path,
        typer.Option(
            "--co-citations",
            help="Path to co_citations JSONL file.",
            show_default=False,
        ),
    ],
    bib_coupling: Annotated[
        Path,
        typer.Option(
            "--bib-coupling",
            help="Path to bib_coupling JSONL file.",
            show_default=False,
        ),
    ],
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Number of results after re-ranking."),
    ] = 20,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Pairs per forward pass."),
    ] = 32,
    device: Annotated[
        str,
        typer.Option("--device", help="Device for inference (cuda or cpu)."),
    ] = "cpu",
) -> None:
    """Re-rank retrieval results with a cross-encoder and evaluate.

    Scores each candidate pair with the trained re-ranker, then evaluates
    against citation ground truth. Works with any retrieval JSONL input
    (BM25, graph pipeline, etc.).
    """
    run_rerank_retrieval(
        retrieval_path=retrieval,
        facets_path=facets,
        paper_graphs_path=paper_graphs,
        model_dir=model_dir,
        output_path=output,
        co_cit_path=co_citations,
        bib_path=bib_coupling,
        top_n=top_n,
        batch_size=batch_size,
        device=device,
    )


def cmd_s2_recs_baseline(
    paper_graphs: Annotated[
        Path,
        typer.Option(
            "--paper-graphs",
            help="Path to paper_graphs JSONL file (for corpus ID set).",
            show_default=False,
        ),
    ],
    query_papers: Annotated[
        Path,
        typer.Option(
            "--query-papers",
            help="Path to query papers JSONL file.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to write retrieval results.",
            show_default=False,
        ),
    ],
    co_citations: Annotated[
        Path,
        typer.Option(
            "--co-citations",
            help="Path to co_citations JSONL file.",
            show_default=False,
        ),
    ],
    bib_coupling: Annotated[
        Path,
        typer.Option(
            "--bib-coupling",
            help="Path to bib_coupling JSONL file.",
            show_default=False,
        ),
    ],
    pool: Annotated[
        str,
        typer.Option("--pool", help="S2 recommendation pool."),
    ] = "all-cs",
    limit: Annotated[
        int,
        typer.Option("--limit", help="Max recommendations per query from API."),
    ] = 200,
) -> None:
    """Fetch S2 paper recommendations and evaluate against citation GT.

    Calls the Semantic Scholar recommendations API for each query paper,
    filters to in-corpus results, and computes Recall@K / MRR metrics.
    Requires SEMANTIC_SCHOLAR_API_KEY environment variable.
    """
    run_s2_recs_baseline(
        paper_graphs_path=paper_graphs,
        query_papers_path=query_papers,
        output_path=output,
        co_cit_path=co_citations,
        bib_path=bib_coupling,
        pool=pool,
        limit=limit,
    )


def cmd_embedding_baseline(
    paper_graphs: Annotated[
        Path,
        typer.Option(
            "--paper-graphs",
            help="Path to paper_graphs JSONL file (with abstracts).",
            show_default=False,
        ),
    ],
    query_ids: Annotated[
        Path,
        typer.Option(
            "--query-ids",
            help="Path to query corpus IDs file (one per line).",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to write retrieval results.",
            show_default=False,
        ),
    ],
    co_citations: Annotated[
        Path,
        typer.Option(
            "--co-citations",
            help="Path to co_citations JSONL file.",
            show_default=False,
        ),
    ],
    bib_coupling: Annotated[
        Path,
        typer.Option(
            "--bib-coupling",
            help="Path to bib_coupling JSONL file.",
            show_default=False,
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Sentence-transformer model name."),
    ],
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Number of candidates to retrieve per query."),
    ] = 50,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Embedding batch size."),
    ] = 64,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Device for inference (default: auto)."),
    ] = None,
) -> None:
    """Run embedding similarity baseline (title+abstract) and evaluate.

    Embeds all papers with a sentence-transformer model and retrieves
    candidates by cosine similarity. No facets, graph, or reranking.
    """
    from genret.processing.embedding_baseline import (  # noqa: PLC0415
        run_embedding_baseline,
    )

    run_embedding_baseline(
        paper_graphs_path=paper_graphs,
        query_ids_path=query_ids,
        output_path=output,
        co_cit_path=co_citations,
        bib_path=bib_coupling,
        model_name=model,
        top_n=top_n,
        batch_size=batch_size,
        device=device,
    )


def register_baseline_commands(app: typer.Typer) -> None:
    """Register baseline commands on the given Typer app."""
    app.command(name="bm25-baseline")(cmd_bm25_baseline)
    app.command(name="rerank-retrieval")(cmd_rerank_retrieval)
    app.command(name="s2-recs-baseline")(cmd_s2_recs_baseline)
    app.command(name="embedding-baseline")(cmd_embedding_baseline)


__all__ = [
    "cmd_bm25_baseline",
    "cmd_embedding_baseline",
    "cmd_rerank_retrieval",
    "cmd_s2_recs_baseline",
    "register_baseline_commands",
]
