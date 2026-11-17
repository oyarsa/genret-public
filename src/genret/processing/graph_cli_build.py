"""Graph-building commands: local embeddings, facet/candidate graphs, edges."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from genret.processing.facet_edges import build_edges
from genret.processing.facet_graph import build_facet_graph
from genret.processing.facet_graph_build import build_candidate_graph
from genret.utils import die

logger = logging.getLogger(__name__)


def cmd_embed_local(
    facets: Annotated[
        Path,
        typer.Option("--facets", help="Path to facets JSONL file.", show_default=False),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output path for embeddings .npz file.",
            show_default=False,
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="HuggingFace model ID."),
    ] = "sentence-transformers/all-MiniLM-L12-v2",
    adapter: Annotated[
        str | None,
        typer.Option("--adapter", help="Optional PEFT adapter ID."),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Texts per forward pass."),
    ] = 64,
) -> None:
    """Embed facet items using a local HuggingFace model.

    Produces a ``.npz`` file with the same structure as the OpenAI embedding
    pipeline: one key per corpus ID, each a float32 array of shape
    ``(n_items, D)``.
    """
    from genret.processing.local_embeddings import embed_facets  # noqa: PLC0415

    embed_facets(
        facets_path=facets,
        output_path=output,
        model_id=model,
        adapter_id=adapter,
        batch_size=batch_size,
    )


def cmd_build_facet_graph(
    facets: Annotated[
        Path,
        typer.Option("--facets", help="Path to facets JSONL file.", show_default=False),
    ],
    embeddings: Annotated[
        Path,
        typer.Option(
            "--embeddings",
            help="Path to facet embeddings .npz cache.",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to write facet_graph_*.npz files.",
            show_default=False,
        ),
    ],
    block_size: Annotated[
        int,
        typer.Option(
            "--block-size",
            help=(
                "Query papers processed per block. "
                "Reduce to lower peak memory (~700 MB/block at default)."
            ),
        ),
    ] = 256,
    include_pairwise: Annotated[
        bool,
        typer.Option(
            "--include-pairwise",
            help=(
                "Also compute max_pairs and mean_pairs matrices. "
                "These are expensive and not used by candidate generation."
            ),
        ),
    ] = False,
    cross_facet: Annotated[
        list[str] | None,
        typer.Option(
            "--cross-facet",
            help=(
                "Cross-facet pair as 'query->key' (e.g. 'problems->contributions'). "
                "Can be specified multiple times."
            ),
        ),
    ] = None,
) -> None:
    """Build pairwise facet similarity matrices for all papers.

    For each of the 4 facet types, writes a compressed
    ``facet_graph_{name}.npz`` file containing the ``mean_pool`` ``(N, N)``
    float16 similarity matrix. With ``--include-pairwise``, also computes
    ``max_pairs`` and ``mean_pairs`` (expensive, not used downstream).

    With ``--cross-facet``, also computes cross-facet similarity matrices
    (e.g. problems vs contributions).
    """
    cross_pairs: list[tuple[str, str]] = []
    for spec in cross_facet or []:
        if "->" not in spec:
            die(f"Expected 'query->key' format, got: {spec!r}")
        query, key = spec.split("->", 1)
        cross_pairs.append((query.strip(), key.strip()))

    build_facet_graph(
        facets,
        embeddings,
        output_dir,
        block_size,
        include_pairwise=include_pairwise,
        cross_facet_pairs=cross_pairs or None,
    )


def cmd_build_candidate_graph(
    facets: Annotated[
        Path,
        typer.Option("--facets", help="Path to facets JSONL file.", show_default=False),
    ],
    dense_dir: Annotated[
        Path,
        typer.Option(
            "--dense-dir",
            help="Directory containing facet_graph_*.npz dense matrices.",
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
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to write facet_candidates_*.npz and candidate_graph_params.json.",
            show_default=False,
        ),
    ],
    embedding_model: Annotated[
        str,
        typer.Option(
            "--embedding-model",
            help="Name of the embedding model used for the dense matrices "
            "(recorded in params).",
        ),
    ] = "all-MiniLM-L12-v2",
    k: Annotated[
        int,
        typer.Option("--k", help="Top-K candidates to retain per paper per facet."),
    ] = 100,
    alpha: Annotated[
        float,
        typer.Option("--alpha", help="Weight for the dense cosine similarity signal."),
    ] = 0.75,
    beta: Annotated[
        float,
        typer.Option("--beta", help="Weight for the n-gram Jaccard signal."),
    ] = 0.0,
    gamma: Annotated[
        float,
        typer.Option("--gamma", help="Weight for the citation signal."),
    ] = 0.25,
    dense_metric: Annotated[
        str,
        typer.Option(
            "--dense-metric",
            help="NPZ key for the dense similarity matrix (e.g. 'mean_pool', 'max_pairs').",
        ),
    ] = "mean_pool",
) -> None:
    """Build candidate graph NPZ files combining dense, n-gram, and citation signals.

    For each of the 4 facet types, writes a ``facet_candidates_{name}.npz``
    file containing the top-K candidate paper indices and per-signal scores.
    Also writes ``candidate_graph_params.json`` recording all run parameters.
    """
    build_candidate_graph(
        facets_path=facets,
        dense_dir=dense_dir,
        co_citations_path=co_citations,
        bib_coupling_path=bib_coupling,
        output_dir=output_dir,
        embedding_model=embedding_model,
        k=k,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        dense_metric=dense_metric,
    )


def cmd_build_edges(
    candidates_dir: Annotated[
        Path,
        typer.Option(
            "--candidates-dir",
            help="Directory containing facet_candidates_*.npz files.",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to write paper_edges_*.npz and edge_graph_params.json.",
            show_default=False,
        ),
    ],
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Maximum edges to keep per paper per facet."),
    ] = 20,
    min_score: Annotated[
        float,
        typer.Option("--min-score", help="Minimum combined score threshold."),
    ] = 0.0,
) -> None:
    """Prune candidate edges to top-K and build sparse CSR edge graphs.

    For each of the 4 facet types, loads ``facet_candidates_{name}.npz``,
    prunes to ``top_k`` edges per paper, and writes ``paper_edges_{name}.npz``
    in CSR format. Also writes ``edge_graph_params.json``.
    """
    build_edges(
        candidates_dir=candidates_dir,
        output_dir=output_dir,
        top_k=top_k,
        min_score=min_score,
    )


def register_build_commands(app: typer.Typer) -> None:
    """Register graph-building commands on the given Typer app."""
    app.command(name="embed-local")(cmd_embed_local)
    app.command(name="build-facet-graph")(cmd_build_facet_graph)
    app.command(name="build-candidate-graph")(cmd_build_candidate_graph)
    app.command(name="build-edges")(cmd_build_edges)


__all__ = [
    "cmd_build_candidate_graph",
    "cmd_build_edges",
    "cmd_build_facet_graph",
    "cmd_embed_local",
    "register_build_commands",
]
