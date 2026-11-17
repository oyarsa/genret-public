"""CLI commands for GPT and local judge workflows."""

import asyncio
import shlex
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from genret.llm import ReasoningEffort
from genret.processing.judge_compare import (
    ComparisonResult,
    ScoredComparisonResult,
    compare_all,
    compare_scored_vs_boolean,
)
from genret.processing.judge_human_audit import run_interactive_human_audit
from genret.processing.llm_judge import (
    ScoringMode,
    run_llm_judge_pairs,
    sample_judged_pairs_subset,
)
from genret.processing.local_judge_prompting import PromptVariant
from genret.processing.local_judge_sharding import merge_scored_shards
from genret.processing.reranking_compare import (
    AlphaSweepResult,
    DepthSweepResult,
    RerankingSummary,
    compare_alpha_blending,
    compare_reranking,
    compare_reranking_depths,
)
from genret.utils import die, make_app

app = make_app("Judge workflows for hosted GPT models and local models")
gpt_app = make_app("Hosted GPT judge workflows")
local_app = make_app("Local judge workflows")

app.add_typer(gpt_app, name="gpt")
app.add_typer(local_app, name="local")


def cmd_gpt_sample_subset(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            help="Path to an existing `llm-judge` JSONL file.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output path for the sampled subset JSONL.",
            show_default=False,
        ),
    ],
    count: Annotated[
        int,
        typer.Option("--count", help="Number of judged pairs to sample."),
    ] = 500,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for subset sampling."),
    ] = 0,
    low_confidence_max: Annotated[
        float,
        typer.Option(
            "--low-confidence-max",
            help="Upper bound for the low-confidence bucket.",
        ),
    ] = 1.0 / 3.0,
    high_confidence_min: Annotated[
        float,
        typer.Option(
            "--high-confidence-min",
            help="Lower bound for the high-confidence bucket.",
        ),
    ] = 2.0 / 3.0,
) -> None:
    """Sample a representative subset from an existing GPT judge file."""
    sample_judged_pairs_subset(
        input_path=input_path,
        output_path=output,
        count=count,
        seed=seed,
        low_confidence_max=low_confidence_max,
        high_confidence_min=high_confidence_min,
    )


def cmd_gpt_replay_pairs(
    facets: Annotated[
        Path,
        typer.Option("--facets", help="Path to facets JSONL file.", show_default=False),
    ],
    pairs: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help="Path to explicit pair-list JSONL file.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output path for judgements JSONL.",
            show_default=False,
        ),
    ],
    concurrent: Annotated[
        int,
        typer.Option("--concurrent", help="Number of concurrent LLM calls."),
    ] = 50,
    model: Annotated[
        str,
        typer.Option("--model", help="LLM model for judging."),
    ] = "gpt-5-mini",
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            help="LLM provider (openai, azure-openai).",
        ),
    ] = "azure-openai",
    azure_endpoint: Annotated[
        str | None,
        typer.Option("--azure-endpoint", help="Azure endpoint URL."),
    ] = None,
    api_version: Annotated[
        str | None,
        typer.Option("--api-version", help="Azure API version override."),
    ] = None,
    reasoning_effort: Annotated[
        ReasoningEffort,
        typer.Option(
            "--reasoning-effort",
            help="Reasoning effort for GPT-5 models.",
        ),
    ] = ReasoningEffort.MINIMAL,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed recorded in run metadata."),
    ] = 0,
) -> None:
    """Judge an explicit JSONL list of paper pairs using a hosted GPT model."""
    asyncio.run(
        run_llm_judge_pairs(
            facets_path=facets,
            pairs_path=pairs,
            output_path=output,
            concurrent=concurrent,
            model=model,
            reasoning_effort=reasoning_effort,
            provider=provider,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            seed=seed,
        )
    )


def _print_comparison_table(results: list[ComparisonResult]) -> None:
    """Print a Rich table summarising pairwise judge comparison results.

    Args:
        results: List of pairwise comparison results.
    """
    console = Console()
    table = Table(title="Judge Comparison", show_lines=True)
    table.add_column("A vs B", style="bold")
    table.add_column("Pairs A", justify="right")
    table.add_column("Pairs B", justify="right")
    table.add_column("Common", justify="right")
    table.add_column("Related Agr.", justify="right")
    table.add_column("Type Agr.", justify="right")
    table.add_column("Kappa", justify="right")
    table.add_column("Conf A", justify="right")
    table.add_column("Conf B", justify="right")

    for result in results:
        table.add_row(
            f"{result.approach_a} vs {result.approach_b}",
            str(result.total_pairs_a),
            str(result.total_pairs_b),
            str(result.common_pairs),
            f"{result.related_agreement:.1%}",
            f"{result.relationship_type_agreement:.1%}",
            f"{result.cohens_kappa:.3f}",
            f"{result.mean_confidence_a:.2f}",
            f"{result.mean_confidence_b:.2f}",
        )

    console.print(table)


def cmd_gpt_compare_runs(
    files: Annotated[
        list[str],
        typer.Argument(
            help=(
                "Judgement files as name=path pairs "
                "(e.g. gpt-5-mini=output/judge_a.jsonl "
                "gpt-5.4=output/judge_b.jsonl)."
            ),
        ),
    ],
) -> None:
    """Compare hosted GPT judge outputs on the common judged pairs."""
    judgement_files: dict[str, Path] = {}
    for entry in files:
        if "=" not in entry:
            die(f"Expected name=path format, got: {entry!r}")

        name, path_str = entry.split("=", 1)
        path = Path(path_str)
        if not path.exists():
            die(f"File not found: {path}")

        judgement_files[name] = path

    if len(judgement_files) < 2:
        die("Need at least 2 judgement files to compare")

    _print_comparison_table(compare_all(judgement_files))


def cmd_gpt_interactive_audit(
    facets: Annotated[
        Path,
        typer.Option("--facets", help="Path to facets JSONL file.", show_default=False),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output path for human audit JSONL.",
            show_default=False,
        ),
    ],
    files: Annotated[
        list[str],
        typer.Argument(
            help=(
                "Judgement files as name=path pairs "
                "(e.g. gpt-5-mini=output/judge_a.jsonl "
                "gpt-5.4=output/judge_b.jsonl)."
            ),
        ),
    ],
    count: Annotated[
        int,
        typer.Option("--count", help="Maximum number of new pairs to audit."),
    ] = 50,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for subset sampling."),
    ] = 0,
) -> None:
    """Run an interactive human audit over common judged pairs."""
    run_interactive_human_audit(
        facets_path=facets,
        output_path=output,
        files=files,
        count=count,
        seed=seed,
    )


def cmd_local_score(
    facets: Annotated[
        Path,
        typer.Option("--facets", help="Path to facets JSONL file.", show_default=False),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output path for scored pairs JSONL.",
            show_default=False,
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="HuggingFace model ID.", show_default=False),
    ],
    retrieval: Annotated[
        Path | None,
        typer.Option(
            "--retrieval",
            help="Path to retrieval_sets JSONL file.",
            show_default=False,
        ),
    ] = None,
    pairs: Annotated[
        Path | None,
        typer.Option(
            "--pairs",
            help="Path to explicit pair-list JSONL file.",
            show_default=False,
        ),
    ] = None,
    mode: Annotated[
        ScoringMode, typer.Option("--mode", help="Scoring mode.")
    ] = ScoringMode.BINARY_SEQUENCE,
    prompt_variant: Annotated[
        PromptVariant,
        typer.Option("--prompt-variant", help="Prompt-compression variant."),
    ] = PromptVariant.TOP2_NO_RESULTS,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Pairs per batch."),
    ] = 8,
    n_papers: Annotated[
        int,
        typer.Option("--n-papers", help="Number of query papers to sample."),
    ] = 500,
    negatives_per_query: Annotated[
        int,
        typer.Option("--negatives", help="Random negative pairs per query."),
    ] = 5,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for pair sampling."),
    ] = 0,
    quantize_4bit: Annotated[
        bool,
        typer.Option(
            "--4bit",
            help="Use 4-bit NF4 quantisation (reduces VRAM, minor quality loss).",
        ),
    ] = False,
    shard_index: Annotated[
        int | None,
        typer.Option(
            "--shard-index",
            help="Optional 0-based shard index for exact-pairs scoring.",
            show_default=False,
        ),
    ] = None,
    num_shards: Annotated[
        int | None,
        typer.Option(
            "--num-shards",
            help="Optional total shard count for exact-pairs scoring.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Score paper pairs using a local HuggingFace judge."""
    from genret.processing.local_judge import (  # noqa: PLC0415
        run_local_judge,
        run_local_judge_pairs,
    )

    if (retrieval is None) == (pairs is None):
        die("Specify exactly one of `--retrieval` or `--pairs`.")

    if (shard_index is None) != (num_shards is None):
        die("Specify both `--shard-index` and `--num-shards`, or neither.")

    if pairs is not None:
        run_local_judge_pairs(
            facets_path=facets,
            pairs_path=pairs,
            output_path=output,
            model_id=model,
            mode=mode,
            prompt_variant=prompt_variant,
            batch_size=batch_size,
            quantize_4bit=quantize_4bit,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        return

    if shard_index is not None or num_shards is not None:
        die("Sharding is only supported together with `--pairs`.")

    assert retrieval is not None
    run_local_judge(
        facets_path=facets,
        retrieval_path=retrieval,
        output_path=output,
        model_id=model,
        mode=mode,
        prompt_variant=prompt_variant,
        batch_size=batch_size,
        n_papers=n_papers,
        negatives_per_query=negatives_per_query,
        seed=seed,
        quantize_4bit=quantize_4bit,
    )


def cmd_local_merge_shards(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output path for merged scored pairs JSONL.",
            show_default=False,
        ),
    ],
    inputs: Annotated[
        list[Path],
        typer.Argument(
            help="Shard output JSONL files to merge.",
        ),
    ],
) -> None:
    """Merge multiple local-judge shard outputs into one scored file."""
    summary = merge_scored_shards(input_paths=inputs, output_path=output)
    console = Console()
    console.print(
        f"Merged {summary['pair_count']} scored pairs to {summary['output_path']}"
    )
    console.print(f"Mode: {summary['mode']}")
    console.print(f"Score range: {summary['score_range']}")


def _shell_quote(value: Path | str) -> str:
    """Quote one shell argument for display in a generated run plan.

    Args:
        value: Path or raw string argument.

    Returns:
        Shell-escaped argument text.
    """
    return shlex.quote(str(value))


def cmd_local_plan_shards(
    facets: Annotated[
        Path,
        typer.Option("--facets", help="Path to facets JSONL file.", show_default=False),
    ],
    pairs: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help="Path to explicit pair-list JSONL file.",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for per-shard scored outputs and the merged file.",
            show_default=False,
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="HuggingFace model ID.", show_default=False),
    ],
    num_shards: Annotated[
        int,
        typer.Option("--num-shards", help="Total number of shards / GPUs."),
    ],
    mode: Annotated[
        ScoringMode, typer.Option("--mode", help="Scoring mode.")
    ] = ScoringMode.BINARY_SEQUENCE,
    prompt_variant: Annotated[
        PromptVariant,
        typer.Option("--prompt-variant", help="Prompt-compression variant."),
    ] = PromptVariant.TOP2_NO_RESULTS,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Pairs per batch."),
    ] = 8,
    output_stem: Annotated[
        str,
        typer.Option(
            "--output-stem",
            help="Base name used for shard outputs and the merged file.",
        ),
    ] = "local_judge_run",
    quantize_4bit: Annotated[
        bool,
        typer.Option(
            "--4bit",
            help="Use 4-bit NF4 quantisation (reduces VRAM, minor quality loss).",
        ),
    ] = False,
) -> None:
    """Print a complete sharded local-judge run plan for multiple GPUs."""
    if num_shards <= 0:
        die("`--num-shards` must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    shard_paths = [
        output_dir / f"{output_stem}.shard-{shard_index:03d}-of-{num_shards:03d}.jsonl"
        for shard_index in range(num_shards)
    ]
    merged_path = output_dir / f"{output_stem}.merged.jsonl"

    console.print("[bold]Shard Commands[/bold]")
    for shard_index, shard_path in enumerate(shard_paths):
        command_parts = [
            "uv run genret judge local score",
            f"--facets {_shell_quote(facets)}",
            f"--pairs {_shell_quote(pairs)}",
            f"--output {_shell_quote(shard_path)}",
            f"--model {_shell_quote(model)}",
            f"--mode {mode.value}",
            f"--prompt-variant {prompt_variant.value}",
            f"--batch-size {batch_size}",
            f"--shard-index {shard_index}",
            f"--num-shards {num_shards}",
        ]
        if quantize_4bit:
            command_parts.append("--4bit")

        console.print(f"# {shard_index:02d}\n" + "\n  ".join(command_parts))
        console.print()

    merge_command = [
        "uv run genret judge local merge-shards",
        f"--output {_shell_quote(merged_path)}",
        *[_shell_quote(path) for path in shard_paths],
    ]
    console.print("[bold]Merge Command[/bold]")
    console.print(" \\\n  ".join(merge_command))


def _print_scored_comparison(result: ScoredComparisonResult) -> None:
    """Print comparison results for scored vs boolean judgements.

    Args:
        result: Scored-vs-boolean comparison result.
    """
    console = Console()

    console.print(f"\n[bold]{result.scored_name} vs {result.boolean_name}[/bold]")
    console.print(f"  Scored pairs: {result.total_scored}")
    console.print(f"  Boolean pairs: {result.total_boolean}")
    console.print(f"  Common pairs: {result.common_pairs}")
    console.print(f"  Score range: {result.score_range}")
    console.print(f"  Mean score (related): {result.mean_score_related:.2f}")
    console.print(f"  Mean score (unrelated): {result.mean_score_unrelated:.2f}")
    console.print(f"  Spearman rho: {result.spearman_rho:.3f}")
    console.print(f"  ROC AUC: {result.roc_auc:.3f}")

    table = Table(title="Threshold Sweep", show_lines=True)
    table.add_column("Threshold", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Agreement", justify="right")
    table.add_column("Kappa", justify="right")

    for threshold_result in result.threshold_results:
        table.add_row(
            f">= {threshold_result.threshold:g}",
            f"{threshold_result.precision:.1%}",
            f"{threshold_result.recall:.1%}",
            f"{threshold_result.f1:.3f}",
            f"{threshold_result.agreement:.1%}",
            f"{threshold_result.cohens_kappa:.3f}",
        )

    console.print(table)


def cmd_local_compare_to_gpt(
    scored: Annotated[
        Path,
        typer.Option(
            "--scored",
            help="Path to local scored pairs JSONL.",
            show_default=False,
        ),
    ],
    boolean: Annotated[
        Path,
        typer.Option(
            "--boolean",
            help="Path to boolean LLM judgement JSONL.",
            show_default=False,
        ),
    ],
    scored_name: Annotated[
        str,
        typer.Option("--scored-name", help="Name for the scored approach."),
    ] = "local",
    boolean_name: Annotated[
        str,
        typer.Option("--boolean-name", help="Name for the boolean approach."),
    ] = "gpt-5-mini",
) -> None:
    """Compare local judge scores against GPT boolean judgements."""
    if not scored.exists():
        die(f"Scored file not found: {scored}")
    if not boolean.exists():
        die(f"Boolean file not found: {boolean}")

    _print_scored_comparison(
        compare_scored_vs_boolean(scored, boolean, scored_name, boolean_name)
    )


def _print_reranking_summary(summary: RerankingSummary) -> None:
    """Print reranking comparison results as Rich tables.

    Args:
        summary: Reranking comparison summary.
    """
    console = Console()

    console.print(
        f"\n[bold]Reranking Comparison ({summary.total_queries} queries)[/bold]\n"
    )

    # Overall + per-bin summary table
    table = Table(title="Ranking Quality by Degree Bin", show_lines=True)
    table.add_column("Bin", style="bold")
    table.add_column("Queries", justify="right")
    table.add_column("Overlap@20", justify="right")
    table.add_column("Kendall τ", justify="right")
    table.add_column("nDCG Retriever", justify="right")
    table.add_column("nDCG Local", justify="right")
    table.add_column("nDCG Lift", justify="right")

    for bin_name in ["sparse", "low_mid", "high"]:
        if bin_name not in summary.by_bin:
            continue

        b = summary.by_bin[bin_name]
        table.add_row(
            bin_name,
            str(b.n_queries),
            f"{b.mean_overlap_at_20:.1%}",
            f"{b.mean_kendall_tau:.3f}",
            f"{b.mean_ndcg_retriever_at_20:.3f}",
            f"{b.mean_ndcg_local_at_20:.3f}",
            f"{b.ndcg_lift:+.3f}",
        )

    o = summary.overall
    table.add_row(
        "[bold]overall[/bold]",
        str(o.n_queries),
        f"{o.mean_overlap_at_20:.1%}",
        f"{o.mean_kendall_tau:.3f}",
        f"{o.mean_ndcg_retriever_at_20:.3f}",
        f"{o.mean_ndcg_local_at_20:.3f}",
        f"{o.ndcg_lift:+.3f}",
    )

    console.print(table)


def cmd_compare_reranking(
    retrieval: Annotated[
        Path, typer.Option("--retrieval", help="Path to retrieval results JSONL.")
    ],
    gpt: Annotated[Path, typer.Option("--gpt", help="Path to GPT judgements JSONL.")],
    local: Annotated[
        Path, typer.Option("--local", help="Path to local judge scored JSONL.")
    ],
    queries: Annotated[
        Path,
        typer.Option("--queries", help="Path to query papers JSONL with degree bins."),
    ],
    k: Annotated[
        int, typer.Option("--k", help="Cutoff rank for nDCG and overlap.")
    ] = 20,
) -> None:
    """Compare retriever vs local judge reranking quality against GPT oracle."""
    for path, name in [
        (retrieval, "retrieval"),
        (gpt, "gpt"),
        (local, "local"),
        (queries, "queries"),
    ]:
        if not path.exists():
            die(f"{name} file not found: {path}")

    summary = compare_reranking(
        retrieval_path=retrieval,
        gpt_path=gpt,
        local_path=local,
        query_meta_path=queries,
        k=k,
    )
    _print_reranking_summary(summary)


def _print_depth_sweep(result: DepthSweepResult) -> None:
    """Print reranking depth sweep results as a Rich table.

    Args:
        result: Depth sweep results.
    """
    console = Console()
    console.print(f"\n[bold]Reranking Depth Sweep (nDCG@{result.eval_k})[/bold]\n")

    bin_names = ["sparse", "low_mid", "high"]
    present_bins = [b for b in bin_names if b in result.retriever_baseline.by_bin]

    table = Table(title=f"nDCG@{result.eval_k} by Reranking Depth", show_lines=True)
    table.add_column("Rerank Top N", style="bold", justify="right")
    for b in present_bins:
        table.add_column(b, justify="right")
    table.add_column("Overall", justify="right", style="bold")

    # Baseline row
    baseline = result.retriever_baseline
    table.add_row(
        "baseline",
        *[f"{baseline.by_bin[b]:.3f}" for b in present_bins],
        f"{baseline.overall_ndcg:.3f}",
    )

    # Depth rows
    for depth in result.depths:
        table.add_row(
            str(depth.rerank_depth),
            *[f"{depth.by_bin.get(b, 0.0):.3f}" for b in present_bins],
            f"{depth.overall_ndcg:.3f}",
        )

    console.print(table)

    # Lift table
    lift_table = Table(title="nDCG Lift over Baseline", show_lines=True)
    lift_table.add_column("Rerank Top N", style="bold", justify="right")
    for b in present_bins:
        lift_table.add_column(b, justify="right")
    lift_table.add_column("Overall", justify="right", style="bold")

    for depth in result.depths:
        lift_table.add_row(
            str(depth.rerank_depth),
            *[
                f"{depth.by_bin.get(b, 0.0) - baseline.by_bin[b]:+.3f}"
                for b in present_bins
            ],
            f"{depth.overall_ndcg - baseline.overall_ndcg:+.3f}",
        )

    console.print(lift_table)


def cmd_rerank_depth_sweep(
    retrieval: Annotated[
        Path,
        typer.Option(
            "--retrieval",
            help="Path to retrieval results JSONL.",
            show_default=False,
        ),
    ],
    gpt: Annotated[
        Path,
        typer.Option(
            "--gpt",
            help="Path to GPT judgements JSONL.",
            show_default=False,
        ),
    ],
    local: Annotated[
        Path,
        typer.Option(
            "--local",
            help="Path to local judge scored JSONL.",
            show_default=False,
        ),
    ],
    queries: Annotated[
        Path,
        typer.Option(
            "--queries",
            help="Path to query papers JSONL with degree bins.",
            show_default=False,
        ),
    ],
    depths: Annotated[
        str,
        typer.Option(
            "--depths",
            help="Comma-separated reranking depths (e.g. '10,20,50').",
        ),
    ] = "10,20,50",
    k: Annotated[
        int,
        typer.Option("--k", help="Cutoff rank for nDCG measurement."),
    ] = 20,
) -> None:
    """Sweep reranking depths to find optimal top-N for local judge reranking."""
    for path, name in [
        (retrieval, "retrieval"),
        (gpt, "gpt"),
        (local, "local"),
        (queries, "queries"),
    ]:
        if not path.exists():
            die(f"{name} file not found: {path}")

    depth_list = [int(d.strip()) for d in depths.split(",")]

    result = compare_reranking_depths(
        retrieval_path=retrieval,
        gpt_path=gpt,
        local_path=local,
        query_meta_path=queries,
        rerank_depths=depth_list,
        eval_k=k,
    )
    _print_depth_sweep(result)


def _print_alpha_sweep(result: AlphaSweepResult) -> None:
    """Print alpha blending sweep results as a Rich table.

    Args:
        result: Alpha sweep results.
    """
    console = Console()
    console.print(
        f"\n[bold]Alpha Blending Sweep "
        f"(depth={result.rerank_depth}, nDCG@{result.eval_k})[/bold]\n"
    )

    bin_names = ["sparse", "low_mid", "high"]
    present_bins = [
        b for b in bin_names if result.results and b in result.results[0].by_bin
    ]

    table = Table(title="nDCG@20 by Alpha (0=pure judge, 1=pure RRF)", show_lines=True)
    table.add_column("Alpha", style="bold", justify="right")
    for b in present_bins:
        table.add_column(b, justify="right")
    table.add_column("Overall", justify="right", style="bold")

    for r in result.results:
        table.add_row(
            f"{r.alpha:.1f}",
            *[f"{r.by_bin.get(b, 0.0):.3f}" for b in present_bins],
            f"{r.overall_ndcg:.3f}",
        )

    console.print(table)


def cmd_alpha_sweep(
    retrieval: Annotated[
        Path,
        typer.Option(
            "--retrieval",
            help="Path to retrieval results JSONL.",
            show_default=False,
        ),
    ],
    gpt: Annotated[
        Path,
        typer.Option(
            "--gpt",
            help="Path to GPT judgements JSONL.",
            show_default=False,
        ),
    ],
    local: Annotated[
        Path,
        typer.Option(
            "--local",
            help="Path to local judge scored JSONL.",
            show_default=False,
        ),
    ],
    queries: Annotated[
        Path,
        typer.Option(
            "--queries",
            help="Path to query papers JSONL with degree bins.",
            show_default=False,
        ),
    ],
    alphas: Annotated[
        str,
        typer.Option(
            "--alphas",
            help="Comma-separated alpha values (e.g. '0.0,0.1,0.2,...,1.0').",
        ),
    ] = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    depth: Annotated[
        int,
        typer.Option("--depth", help="Reranking depth."),
    ] = 50,
    k: Annotated[
        int,
        typer.Option("--k", help="Cutoff rank for nDCG measurement."),
    ] = 20,
) -> None:
    """Sweep alpha values for blended RRF + judge reranking."""
    for path, name in [
        (retrieval, "retrieval"),
        (gpt, "gpt"),
        (local, "local"),
        (queries, "queries"),
    ]:
        if not path.exists():
            die(f"{name} file not found: {path}")

    alpha_list = [float(a.strip()) for a in alphas.split(",")]

    result = compare_alpha_blending(
        retrieval_path=retrieval,
        gpt_path=gpt,
        local_path=local,
        query_meta_path=queries,
        alphas=alpha_list,
        rerank_depth=depth,
        eval_k=k,
    )
    _print_alpha_sweep(result)


gpt_app.command(name="sample-subset", no_args_is_help=True)(cmd_gpt_sample_subset)
gpt_app.command(name="replay-pairs", no_args_is_help=True)(cmd_gpt_replay_pairs)
gpt_app.command(name="compare-runs", no_args_is_help=True)(cmd_gpt_compare_runs)
gpt_app.command(name="interactive-audit", no_args_is_help=True)(
    cmd_gpt_interactive_audit
)
local_app.command(name="score", no_args_is_help=True)(cmd_local_score)
local_app.command(name="plan-shards", no_args_is_help=True)(cmd_local_plan_shards)
local_app.command(name="merge-shards", no_args_is_help=True)(cmd_local_merge_shards)
local_app.command(name="compare-to-gpt", no_args_is_help=True)(cmd_local_compare_to_gpt)
app.command(name="compare-reranking", no_args_is_help=True)(cmd_compare_reranking)
app.command(name="rerank-depth-sweep", no_args_is_help=True)(cmd_rerank_depth_sweep)
app.command(name="alpha-sweep", no_args_is_help=True)(cmd_alpha_sweep)
