"""Re-ranker commands: rerank, train-reranker, train-distilled-reranker, apply-reranker."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from genret.processing.llm_judge import ScoringMode
from genret.processing.local_judge_prompting import PromptVariant
from genret.processing.reranker import apply_reranker
from genret.processing.reranker_train import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEV_FRACTION,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_SEED,
    DEFAULT_WARMUP_RATIO,
    build_training_pairs,
    train,
    train_distilled,
)

logger = logging.getLogger(__name__)


def cmd_rerank(
    facets: Annotated[
        Path,
        typer.Option("--facets", help="Path to facets JSONL file.", show_default=False),
    ],
    retrieval: Annotated[
        Path,
        typer.Option(
            "--retrieval",
            help="Path to retrieval results JSONL (from batch-retrieve).",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output path for reranked retrieval JSONL.",
            show_default=False,
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="HuggingFace model ID.", show_default=False),
    ],
    rerank_depth: Annotated[
        int,
        typer.Option("--rerank-depth", help="Number of top candidates to rerank."),
    ] = 50,
    mode: Annotated[
        ScoringMode, typer.Option("--mode", help="Scoring mode.")
    ] = ScoringMode.BINARY_SEQUENCE,
    prompt_variant: Annotated[
        PromptVariant,
        typer.Option("--prompt-variant", help="Prompt-compression variant."),
    ] = PromptVariant.TOP2_NO_RESULTS,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Pairs per scoring batch."),
    ] = 4,
    quantize_4bit: Annotated[
        bool,
        typer.Option(
            "--4bit",
            help="Use 4-bit NF4 quantisation (reduces VRAM, minor quality loss).",
        ),
    ] = False,
) -> None:
    """Rerank retrieval results using a local LLM judge.

    Loads a retrieval JSONL file (from ``batch-retrieve``), scores the top
    ``--rerank-depth`` candidates per query with the local judge, reorders
    by judge score, and writes the reranked results.
    """
    from genret.processing.local_judge import rerank_retrieval  # noqa: PLC0415

    rerank_retrieval(
        facets_path=facets,
        retrieval_path=retrieval,
        output_path=output,
        model_id=model,
        rerank_depth=rerank_depth,
        mode=mode,
        prompt_variant=prompt_variant,
        batch_size=batch_size,
        quantize_4bit=quantize_4bit,
    )


def cmd_train_reranker(
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
    candidates_dir: Annotated[
        Path,
        typer.Option(
            "--candidates-dir",
            help="Directory containing facet_candidates_*.npz files.",
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
            help="Directory to write training pairs, checkpoint, and params.",
            show_default=False,
        ),
    ],
    base_model: Annotated[
        str,
        typer.Option(
            "--base-model", help="HuggingFace model ID for the base cross-encoder."
        ),
    ] = DEFAULT_BASE_MODEL,
    co_cit_min: Annotated[
        int,
        typer.Option(
            "--co-cit-min", help="Minimum co-citation count for a positive pair."
        ),
    ] = 3,
    bib_min: Annotated[
        int,
        typer.Option(
            "--bib-min", help="Minimum shared references for a positive pair."
        ),
    ] = 5,
    negative_ratio: Annotated[
        int,
        typer.Option("--neg-ratio", help="Hard negative to positive ratio."),
    ] = 3,
    epochs: Annotated[
        int,
        typer.Option("--epochs", help="Number of training epochs."),
    ] = DEFAULT_MAX_EPOCHS,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Per-device training batch size."),
    ] = DEFAULT_BATCH_SIZE,
    max_length: Annotated[
        int,
        typer.Option("--max-length", help="Maximum token length for input sequences."),
    ] = DEFAULT_MAX_LENGTH,
    lr: Annotated[
        float,
        typer.Option("--lr", help="Learning rate."),
    ] = DEFAULT_LR,
    warmup_ratio: Annotated[
        float,
        typer.Option(
            "--warmup-ratio", help="Fraction of total steps for linear warmup."
        ),
    ] = DEFAULT_WARMUP_RATIO,
    dev_fraction: Annotated[
        float,
        typer.Option(
            "--dev-fraction", help="Fraction of data to hold out for evaluation."
        ),
    ] = DEFAULT_DEV_FRACTION,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed."),
    ] = DEFAULT_SEED,
) -> None:
    """Build training data and fine-tune a cross-encoder re-ranker.

    First builds training pairs from citation signals and hard negatives (or reuses
    existing pairs if present), then fine-tunes a DeBERTa-v3 sequence classifier. Pairs
    where either paper lacks a title are dropped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "training_pairs.jsonl.zst"

    if pairs_path.exists():
        logger.info(f"Reusing existing training pairs: {pairs_path}")
    else:
        build_training_pairs(
            facets_path=facets,
            candidates_dir=candidates_dir,
            paper_graphs_path=paper_graphs,
            co_cit_path=co_citations,
            bib_path=bib_coupling,
            output_path=pairs_path,
            co_cit_min=co_cit_min,
            bib_min=bib_min,
            negative_ratio=negative_ratio,
            seed=seed,
        )

    train(
        pairs_path=pairs_path,
        output_dir=output_dir,
        base_model=base_model,
        max_epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        lr=lr,
        warmup_ratio=warmup_ratio,
        dev_fraction=dev_fraction,
        seed=seed,
    )


def cmd_train_distilled_reranker(
    pairs: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help="Path to training pairs JSONL with continuous labels.",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to write the checkpoint and params.",
            show_default=False,
        ),
    ],
    base_model: Annotated[
        str,
        typer.Option("--base-model", help="HuggingFace base model ID."),
    ] = DEFAULT_BASE_MODEL,
    max_epochs: Annotated[
        int,
        typer.Option("--max-epochs", help="Number of training epochs."),
    ] = DEFAULT_MAX_EPOCHS,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Per-device training batch size."),
    ] = DEFAULT_BATCH_SIZE,
    max_length: Annotated[
        int,
        typer.Option("--max-length", help="Maximum token length for input sequences."),
    ] = DEFAULT_MAX_LENGTH,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option(
            "--gradient-accumulation-steps",
            help="Steps to accumulate gradients (effective batch = batch_size * this).",
        ),
    ] = 1,
    lr: Annotated[
        float,
        typer.Option("--lr", help="Learning rate."),
    ] = DEFAULT_LR,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed."),
    ] = 0,
) -> None:
    """Train a distilled cross-encoder from LLM judge scores.

    Uses MSE regression against continuous 0-1 scores from the LLM judge,
    distilling the judge's semantic relevance signal into a lightweight
    cross-encoder.
    """
    train_distilled(
        pairs_path=pairs,
        output_dir=output_dir,
        base_model=base_model,
        max_epochs=max_epochs,
        batch_size=batch_size,
        max_length=max_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr=lr,
        seed=seed,
    )


def cmd_apply_reranker(
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
    candidates_dir: Annotated[
        Path,
        typer.Option(
            "--candidates-dir",
            help="Directory containing facet_candidates_*.npz files.",
            show_default=False,
        ),
    ],
    model_dir: Annotated[
        Path,
        typer.Option(
            "--model-dir",
            help="Path to the fine-tuned checkpoint directory.",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to write paper_edges_*.npz files.",
            show_default=False,
        ),
    ],
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Top-K edges to retain per paper per facet."),
    ] = 20,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Pairs per forward pass."),
    ] = 32,
    device: Annotated[
        str,
        typer.Option("--device", help="Device for inference (cuda or cpu)."),
    ] = "cuda",
) -> None:
    """Score candidate pairs with a trained re-ranker and write edge files.

    For each facet type, loads the top-100 candidate pairs, scores them with the
    cross-encoder, keeps the top-K by score, and writes the result as a compressed
    `paper_edges_{name}.npz` file.
    """
    apply_reranker(
        facets_path=facets,
        paper_graphs_path=paper_graphs,
        candidates_dir=candidates_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        top_k=top_k,
        batch_size=batch_size,
        device=device,
    )


def register_rerank_commands(app: typer.Typer) -> None:
    """Register re-ranker commands on the given Typer app."""
    app.command(name="rerank")(cmd_rerank)
    app.command(name="train-reranker")(cmd_train_reranker)
    app.command(name="train-distilled-reranker")(cmd_train_distilled_reranker)
    app.command(name="apply-reranker")(cmd_apply_reranker)


__all__ = [
    "cmd_apply_reranker",
    "cmd_rerank",
    "cmd_train_distilled_reranker",
    "cmd_train_reranker",
    "register_rerank_commands",
]
