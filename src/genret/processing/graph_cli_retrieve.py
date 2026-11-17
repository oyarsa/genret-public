"""Retrieval and evaluation commands: retrieve, batch-retrieve, retrieve-new, evaluate-graph."""

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from genret.processing.facet_graph_eval import (
    FullEvaluationReport,
    RetrievalMetrics,
    evaluate,
)
from genret.processing.facet_retrieval import (
    DEFAULT_BFS_HOPS,
    DEFAULT_DIVERSITY_SLOTS,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_RRF_K,
    DEFAULT_TOP_N,
    batch_retrieve,
    load_edge_graphs,
    retrieve,
)
from genret.utils import die, load_ids_from_file

logger = logging.getLogger(__name__)


def _print_metrics_table(
    console: Console,
    title: str,
    rows: dict[str, RetrievalMetrics],
) -> None:
    """Print a Rich table of retrieval metrics.

    Args:
        console: Rich console instance.
        title: Table title.
        rows: Dict mapping row label to metrics.
    """
    table = Table(title=title, show_lines=True)
    table.add_column("Config", style="bold")
    table.add_column("R@5", justify="right")
    table.add_column("R@10", justify="right")
    table.add_column("R@20", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("HR@20", justify="right")
    table.add_column("Queries", justify="right")

    for label, m in rows.items():
        table.add_row(
            label,
            f"{m.recall[5]:.4f}",
            f"{m.recall[10]:.4f}",
            f"{m.recall[20]:.4f}",
            f"{m.mrr:.4f}",
            f"{m.hit_rate[20]:.4f}",
            str(m.num_queries),
        )

    console.print(table)


def _print_evaluation_report(report: FullEvaluationReport) -> None:
    """Print the full evaluation report as Rich tables.

    Args:
        report: The evaluation report to print.
    """
    console = Console()

    # Overall + by GT signal
    overall_rows = {"overall": report.overall}
    overall_rows.update(report.by_gt_signal)
    _print_metrics_table(console, "Overall & by GT Signal", overall_rows)

    # Per-facet
    _print_metrics_table(console, "Per-Facet Ablation", report.per_facet)

    # Per-signal
    _print_metrics_table(console, "Per-Signal Ablation", report.per_signal)

    # Dense metrics
    if report.dense_metrics:
        _print_metrics_table(console, "Dense Metric Comparison", report.dense_metrics)

    # Signal overlap
    console.print("\n[bold]Signal Overlap Analysis[/bold]")
    overlap = report.signal_overlap
    overlap_table = Table(show_lines=True)
    overlap_table.add_column("Pair", style="bold")
    overlap_table.add_column("Jaccard (top-20)", justify="right")
    for pair, jacc in sorted(overlap.pairwise_jaccard.items()):
        overlap_table.add_row(pair, f"{jacc:.4f}")
    console.print(overlap_table)

    unique_table = Table(show_lines=True)
    unique_table.add_column("Signal", style="bold")
    unique_table.add_column("Unique fraction", justify="right")
    for sig, frac in sorted(overlap.unique_by_signal.items()):
        unique_table.add_row(sig, f"{frac:.4f}")
    console.print(unique_table)

    # Failure summary
    fs = report.failure_summary
    console.print("\n[bold]Failure Analysis[/bold]")
    console.print(
        f"  Queries with missed GT: {fs.total_failures}  |  "
        f"Total missed pairs: {fs.total_missed_pairs}"
    )

    if fs.missed_by_gt_degree:
        degree_table = Table(title="Missed pairs by query GT degree", show_lines=True)
        degree_table.add_column("Degree bucket", style="bold")
        degree_table.add_column("Missed pairs", justify="right")
        for bucket in ("1-5", "6-20", "21-50", "51+"):
            if bucket in fs.missed_by_gt_degree:
                degree_table.add_row(bucket, str(fs.missed_by_gt_degree[bucket]))
        console.print(degree_table)

    if fs.worst_cases:
        console.print(f"\n  Top-{len(fs.worst_cases)} worst failures:")
        for case in fs.worst_cases:
            console.print(
                f"    {case.corpusid}: missed {len(case.missed_partners)}/{case.num_gt} GT partners"
            )


def cmd_retrieve(
    edges_dir: Annotated[
        Path,
        typer.Option(
            "--edges-dir",
            help="Directory containing paper_edges_*.npz files.",
            show_default=False,
        ),
    ],
    corpusid: Annotated[
        int,
        typer.Option(
            "--corpusid",
            help="Corpus ID of the paper to retrieve for.",
            show_default=False,
        ),
    ],
    bfs_hops: Annotated[
        int,
        typer.Option("--bfs-hops", help="Number of BFS hops."),
    ] = DEFAULT_BFS_HOPS,
    min_score: Annotated[
        float,
        typer.Option("--min-score", help="Minimum edge score for BFS expansion."),
    ] = 0.0,
    max_candidates: Annotated[
        int,
        typer.Option("--max-candidates", help="Hard cap on candidates per facet."),
    ] = DEFAULT_MAX_CANDIDATES,
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Number of results to return."),
    ] = DEFAULT_TOP_N,
) -> None:
    """Retrieve related papers for a single paper.

    Loads the per-facet edge graphs, runs BFS from the given paper, and
    fuses results across facets using RRF. Prints the ranked result list.
    """
    graphs, corpusids = load_edge_graphs(edges_dir)

    paper_idx = next(
        (i for i, cid in enumerate(corpusids) if int(cid) == corpusid), None
    )
    if paper_idx is None:
        die(f"Corpus ID {corpusid} not found in edge graph ({len(corpusids)} papers)")

    results = retrieve(
        paper_idx,
        graphs,
        bfs_hops=bfs_hops,
        min_score=min_score,
        max_candidates=max_candidates,
        top_n=top_n,
    )

    console = Console()
    table = Table(title=f"Retrieval results for {corpusid}")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Corpus ID", justify="right")
    table.add_column("RRF Score", justify="right")
    table.add_column("Facets", justify="left")

    for rank, candidate in enumerate(results, 1):
        facet_str = ", ".join(
            f"{f}(#{candidate.facet_ranks[f]})" for f in candidate.facets
        )
        table.add_row(
            str(rank),
            str(int(corpusids[candidate.index])),
            f"{candidate.score:.6f}",
            facet_str,
        )

    console.print(table)


def cmd_batch_retrieve(
    edges_dir: Annotated[
        Path,
        typer.Option(
            "--edges-dir",
            help="Directory containing paper_edges_*.npz files.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output JSONL path for retrieval sets.",
            show_default=False,
        ),
    ],
    split: Annotated[
        Path | None,
        typer.Option(
            "--split",
            help="Split file (one corpus ID per line). If omitted, retrieve for all papers.",
        ),
    ] = None,
    bfs_hops: Annotated[
        int,
        typer.Option("--bfs-hops", help="Number of BFS hops."),
    ] = DEFAULT_BFS_HOPS,
    min_score: Annotated[
        float,
        typer.Option("--min-score", help="Minimum edge score for BFS expansion."),
    ] = 0.0,
    max_candidates: Annotated[
        int,
        typer.Option("--max-candidates", help="Hard cap on candidates per facet."),
    ] = DEFAULT_MAX_CANDIDATES,
    rrf_k: Annotated[
        int,
        typer.Option("--rrf-k", help="RRF constant."),
    ] = DEFAULT_RRF_K,
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Number of results per paper."),
    ] = DEFAULT_TOP_N,
    diversity_slots: Annotated[
        int,
        typer.Option(
            "--diversity-slots",
            help=(
                "Reserve this many slots for facet-unique candidates."
                " Promotes serendipitous discoveries from individual facets."
                " Set to 0 (default) for pure RRF ranking."
            ),
        ),
    ] = DEFAULT_DIVERSITY_SLOTS,
) -> None:
    """Retrieve related papers for all papers in a split (or full corpus).

    Writes a JSONL file where each line contains a corpus ID and its ranked
    retrieval set. If `--split` is provided, only retrieves for papers
    listed in that file; otherwise retrieves for all papers in the graph.
    """
    graphs, corpusids = load_edge_graphs(edges_dir)

    if split is None:
        paper_indices = list(range(len(corpusids)))
    else:
        split_ids = load_ids_from_file(split)
        cid_to_idx = {int(cid): i for i, cid in enumerate(corpusids)}
        paper_indices = [cid_to_idx[cid] for cid in split_ids if cid in cid_to_idx]
        logger.info(
            f"Split file: {len(split_ids)} IDs, {len(paper_indices)} found in graph"
        )

    batch_retrieve(
        graphs=graphs,
        corpusids=corpusids,
        paper_indices=paper_indices,
        output_path=output,
        bfs_hops=bfs_hops,
        min_score=min_score,
        max_candidates=max_candidates,
        rrf_k=rrf_k,
        top_n=top_n,
        diversity_slots=diversity_slots,
    )


def cmd_retrieve_new(
    facets: Annotated[
        Path,
        typer.Option(
            "--facets",
            help="Path to corpus facets JSONL file.",
            show_default=False,
        ),
    ],
    dense_dir: Annotated[
        Path,
        typer.Option(
            "--dense-dir",
            help="Directory containing facet_graph_*.npz with corpus_vectors.",
            show_default=False,
        ),
    ],
    edges_dir: Annotated[
        Path,
        typer.Option(
            "--edges-dir",
            help="Directory containing paper_edges_*.npz files.",
            show_default=False,
        ),
    ],
    query_facets: Annotated[
        Path,
        typer.Option(
            "--query-facets",
            help="Path to JSONL file with facets for the query paper(s).",
            show_default=False,
        ),
    ],
    query_embeddings: Annotated[
        Path,
        typer.Option(
            "--query-embeddings",
            help="Path to .npz file with per-facet embeddings for the query paper(s).",
            show_default=False,
        ),
    ],
    references: Annotated[
        Path | None,
        typer.Option(
            "--references",
            help="Path to references.jsonl for bib coupling fallback.",
        ),
    ] = None,
    co_citations: Annotated[
        Path | None,
        typer.Option("--co-citations", help="Path to co_citations JSONL."),
    ] = None,
    bib_coupling: Annotated[
        Path | None,
        typer.Option("--bib-coupling", help="Path to bib_coupling JSONL."),
    ] = None,
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Number of results to return."),
    ] = DEFAULT_TOP_N,
    dense_metric: Annotated[
        str,
        typer.Option(
            "--dense-metric",
            help="Dense scoring: 'mean_pool' (fast centroid cosine) or "
            "'max_pairs' (re-score top candidates with MaxSim).",
        ),
    ] = "mean_pool",
    item_embeddings: Annotated[
        Path | None,
        typer.Option(
            "--item-embeddings",
            help="Path to raw item embeddings .npz (required for max_pairs).",
        ),
    ] = None,
) -> None:
    """Retrieve related papers for a new paper not in the corpus graph.

    The query paper's facets and embeddings must be provided separately.
    Embeddings should be a ``.npz`` file with arrays keyed by facet name
    (e.g. ``problems``, ``methods``), each of shape ``(K, D)``.
    """
    import json  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import numpy.typing as npt  # noqa: PLC0415

    from genret.paper_parsing import Facets  # noqa: PLC0415
    from genret.processing.online_retrieval import (  # noqa: PLC0415
        load_corpus_index,
        retrieve_new_paper,
    )
    from genret.utils import open_auto  # noqa: PLC0415

    if dense_metric == "max_pairs" and item_embeddings is None:
        msg = "--item-embeddings is required when --dense-metric=max_pairs"
        raise typer.BadParameter(msg)

    # Load corpus index with citation data
    corpus_index = load_corpus_index(
        dense_dir,
        facets,
        references_path=references,
        co_citations_path=co_citations,
        bib_coupling_path=bib_coupling,
        item_embeddings_path=item_embeddings,
    )

    # Load edge graphs
    graphs, corpusids = load_edge_graphs(edges_dir)

    # Load query paper(s)
    query_papers: list[Facets] = []
    with open_auto(query_facets, "r") as f:
        for line in f:
            if line.strip():
                query_papers.append(Facets.model_validate(json.loads(line)))

    # Load query embeddings
    query_emb_data = dict(np.load(query_embeddings))

    console = Console()

    for paper in query_papers:
        # Extract per-facet embeddings for this paper
        paper_embeddings: dict[str, npt.NDArray[np.float32]] = {}
        for name in ("problems", "methods", "results", "contributions"):
            key = f"{paper.corpusid}_{name}"
            if key in query_emb_data:
                paper_embeddings[name] = query_emb_data[key].astype(np.float32)
            elif name in query_emb_data:
                paper_embeddings[name] = query_emb_data[name].astype(np.float32)

        # Look up query paper's references for bib coupling
        query_refs = corpus_index.corpus_references.get(paper.corpusid)

        results = retrieve_new_paper(
            query_facets=paper,
            query_embeddings=paper_embeddings,
            corpus_index=corpus_index,
            edge_graphs=graphs,
            query_references=query_refs,
            dense_metric=dense_metric,
            top_n=top_n,
        )

        table = Table(title=f"Results for {paper.corpusid}")
        table.add_column("Rank", justify="right", style="dim")
        table.add_column("Corpus ID", justify="right")
        table.add_column("RRF Score", justify="right")
        table.add_column("Facets", justify="left")

        for rank, candidate in enumerate(results, 1):
            facet_str = ", ".join(
                f"{f}(#{candidate.facet_ranks[f]})" for f in candidate.facets
            )
            table.add_row(
                str(rank),
                str(int(corpusids[candidate.index])),
                f"{candidate.score:.6f}",
                facet_str,
            )

        console.print(table)


def cmd_evaluate_graph(
    retrieval: Annotated[
        Path,
        typer.Option(
            "--retrieval",
            help="Path to retrieval_sets JSONL file.",
            show_default=False,
        ),
    ],
    edges_dir: Annotated[
        Path,
        typer.Option(
            "--edges-dir",
            help="Directory containing paper_edges_*.npz files.",
            show_default=False,
        ),
    ],
    candidates_dir: Annotated[
        Path,
        typer.Option(
            "--candidates-dir",
            help="Directory containing facet_candidates_*.npz files.",
            show_default=False,
        ),
    ],
    dense_dir: Annotated[
        Path,
        typer.Option(
            "--dense-dir",
            help="Directory containing facet_graph_*.npz files.",
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
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional path to write full report as JSON.",
        ),
    ] = None,
) -> None:
    """Evaluate graph retrieval quality against citation ground truth.

    Runs per-signal ablation, per-facet breakdown, dense metric comparison,
    signal overlap analysis, and failure analysis. Prints Rich tables to
    the console and optionally writes the full report as JSON.
    """
    report = evaluate(
        retrieval_path=retrieval,
        edges_dir=edges_dir,
        candidates_dir=candidates_dir,
        dense_dir=dense_dir,
        co_cit_path=co_citations,
        bib_path=bib_coupling,
    )

    _print_evaluation_report(report)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2) + "\n")
        logger.info(f"Report written to {output}")


def register_retrieve_commands(app: typer.Typer) -> None:
    """Register retrieval and evaluation commands on the given Typer app."""
    app.command(name="retrieve")(cmd_retrieve)
    app.command(name="batch-retrieve")(cmd_batch_retrieve)
    app.command(name="retrieve-new")(cmd_retrieve_new)
    app.command(name="evaluate-graph")(cmd_evaluate_graph)


__all__ = [
    "cmd_batch_retrieve",
    "cmd_evaluate_graph",
    "cmd_retrieve",
    "cmd_retrieve_new",
    "register_retrieve_commands",
]
