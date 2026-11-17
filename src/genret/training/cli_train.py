# pyright: basic
# ruff: noqa: PLC0415  Enable function-level imports
"""Core training/eval/generate commands for generative retrieval."""

import functools
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, override

import typer

from genret.paper_parsing import FacetRecordType
from genret.training._cli_helpers import _load_checkpoint, _validate_data_files
from genret.utils import die, load_ids_from_file, resolve_jsonl

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def cmd_train(
    splits_dir: Annotated[
        Path, typer.Argument(help="Splits directory with train/dev/test.txt")
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Directory for checkpoints")
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m", help="Base HuggingFace model")
    ] = "meta-llama/Llama-3.1-8B-Instruct",
    epochs: Annotated[
        int, typer.Option("--epochs", help="Number of training epochs")
    ] = 3,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Per-device batch size")
    ] = 4,
    learning_rate: Annotated[
        float, typer.Option("--lr", help="Peak learning rate")
    ] = 2e-4,
    lora_rank: Annotated[int, typer.Option("--lora-rank", help="LoRA rank")] = 16,
    grad_accum: Annotated[
        int, typer.Option("--grad-accum", help="Gradient accumulation steps")
    ] = 4,
    max_seq_length: Annotated[
        int, typer.Option("--max-seq-length", help="Maximum sequence length")
    ] = 4096,
    logging_steps: Annotated[
        int, typer.Option("--logging-steps", help="Log training metrics every N steps")
    ] = 10,
    eval_steps: Annotated[
        int, typer.Option("--eval-steps", help="Run evaluation every N steps")
    ] = 100,
    save_steps: Annotated[
        int, typer.Option("--save-steps", help="Save checkpoint every N steps")
    ] = 100,
    facets_type: Annotated[
        FacetRecordType,
        typer.Option(
            "--facets-type",
            help="Facet record schema in facets.jsonl.gz: standard, evidence, or evidence-list.",
        ),
    ] = FacetRecordType.STANDARD,
    include_evidence: Annotated[
        bool,
        typer.Option(
            "--include-evidence/--no-include-evidence",
            help="Include evidence quotes in training prompts.",
        ),
    ] = True,
    indexing_ratio: Annotated[
        float,
        typer.Option(
            "--indexing-ratio",
            help="Fraction of training examples that are indexing (0.0 = retrieval only).",
            min=0.0,
            max=1.0,
        ),
    ] = 0.5,
    limit: Annotated[
        int,
        typer.Option(
            "--limit", "-n", help="Limit train/dev splits to N papers (0 = all)."
        ),
    ] = 0,
) -> None:
    """Fine-tune a language model with QLoRA on GenRet indexing + retrieval data."""
    from transformers import (
        Trainer,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )

    from genret.training.config import LoRAConfig, QuantConfig, TrainingConfig
    from genret.training.data import GenRetDataset, collate_fn
    from genret.training.model import setup_model_and_tokeniser

    class _FinalStepCallback(TrainerCallback):
        """Force evaluation and checkpoint save at the final training step.

        By default, HuggingFace Trainer only evaluates and saves at intervals
        matching `eval_steps` and `save_steps`. If the total number of steps
        doesn't align with these intervals, the final model state is never
        evaluated — so `load_best_model_at_end` cannot consider it when
        selecting the best checkpoint.
        """

        @override
        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: object,
        ) -> None:
            """Trigger eval and save when the final step is reached."""
            if state.global_step >= state.max_steps:
                control.should_evaluate = True
                control.should_save = True

    processed = _validate_data_files(splits_dir)

    training_cfg = TrainingConfig(
        splits_dir=splits_dir,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
        num_epochs=epochs,
    )
    lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_rank * 2)
    quant_cfg = QuantConfig()
    rng = random.Random(training_cfg.seed)

    logger.info(
        "=== GenRet Training ===\n"
        "  Model: %s\n"
        "  Output: %s\n"
        "  Epochs: %d\n"
        "  Batch: %d x %d (effective %d)\n"
        "  LR: %s\n"
        "  LoRA rank: %d",
        model_name,
        output,
        epochs,
        batch_size,
        grad_accum,
        batch_size * grad_accum,
        learning_rate,
        lora_rank,
    )

    # Load model and tokeniser
    model, tokeniser = setup_model_and_tokeniser(model_name, lora_cfg, quant_cfg)

    # Load datasets
    train_ids = load_ids_from_file(splits_dir / "train.txt")
    dev_ids = load_ids_from_file(splits_dir / "dev.txt")

    if limit > 0:
        train_ids = set(list(train_ids)[:limit])
        dev_ids = set(list(dev_ids)[:limit])

    logger.info("Loading training dataset...")
    train_dataset = GenRetDataset(
        processed,
        rng=rng,
        split_ids=train_ids,
        facets_type=facets_type,
        include_evidence=include_evidence,
        indexing_ratio=indexing_ratio,
    )
    logger.info("Loading dev dataset...")
    dev_dataset = GenRetDataset(
        processed,
        rng=rng,
        split_ids=dev_ids,
        facets_type=facets_type,
        include_evidence=include_evidence,
        indexing_ratio=indexing_ratio,
    )

    # Build collate function with tokeniser bound
    bound_collate = functools.partial(
        collate_fn, tokeniser=tokeniser, max_seq_length=training_cfg.max_seq_length
    )

    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=training_cfg.num_epochs,
        per_device_train_batch_size=training_cfg.batch_size,
        per_device_eval_batch_size=training_cfg.batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        warmup_ratio=training_cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=bound_collate,
        callbacks=[_FinalStepCallback()],
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final adapter and tokeniser
    logger.info("Saving adapter to %s", output)
    model.save_pretrained(output)
    tokeniser.save_pretrained(output)
    logger.info("Training complete.")


def cmd_evaluate(
    splits_dir: Annotated[
        Path, typer.Argument(help="Splits directory with train/dev/test.txt")
    ],
    checkpoint: Annotated[
        Path, typer.Argument(help="Path to fine-tuned LoRA checkpoint")
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Write per-query results to this JSONL path."
        ),
    ] = None,
    beams: Annotated[int, typer.Option("--beams", help="Number of beams")] = 10,
    split: Annotated[
        str, typer.Option("--split", help="Which split to evaluate on")
    ] = "test",
    facets_type: Annotated[
        FacetRecordType,
        typer.Option(
            "--facets-type",
            help="Facet record schema in facets.jsonl.gz: standard, evidence, or evidence-list.",
        ),
    ] = FacetRecordType.STANDARD,
    include_evidence: Annotated[
        bool,
        typer.Option(
            "--include-evidence/--no-include-evidence",
            help="Include evidence quotes in evaluation prompts.",
        ),
    ] = True,
) -> None:
    """Evaluate retrieval quality with Recall@k on a test split."""
    import rich

    from genret.training.decoding import ConstrainedDecoder
    from genret.training.evaluate import evaluate_retrieval, format_results_table

    processed = _validate_data_files(splits_dir)
    index_path = processed / "docid_index.json"

    logger.info("=== GenRet Evaluation ===")
    logger.info("Checkpoint: %s", checkpoint)

    model, tokeniser = _load_checkpoint(checkpoint)

    split_ids = load_ids_from_file(splits_dir / f"{split}.txt")
    if not split_ids:
        die(f"No IDs found in {splits_dir}/{split}.txt")

    logger.info("Evaluating on %d %s papers", len(split_ids), split)

    decoder = ConstrainedDecoder(
        index_path=index_path,
        model=model,
        tokeniser=tokeniser,
        num_beams=beams,
        max_new_tokens=256,
    )

    results = evaluate_retrieval(
        decoder,
        processed,
        split_ids=split_ids,
        facets_type=facets_type,
        include_evidence=include_evidence,
        output_path=output,
        tokeniser=tokeniser,
    )
    rich.print(format_results_table(results))


def cmd_generate(
    splits_dir: Annotated[
        Path, typer.Argument(help="Splits directory with train/dev/test.txt")
    ],
    checkpoint: Annotated[
        Path, typer.Argument(help="Path to fine-tuned LoRA checkpoint")
    ],
    query: Annotated[
        str, typer.Option("--query", "-q", help="Free-text query or corpus ID")
    ] = "",
    beams: Annotated[int, typer.Option("--beams", help="Number of beams")] = 10,
    facets_type: Annotated[
        FacetRecordType,
        typer.Option(
            "--facets-type",
            help="Facet record schema in facets.jsonl.gz: standard, evidence, or evidence-list.",
        ),
    ] = FacetRecordType.STANDARD,
    include_evidence: Annotated[
        bool,
        typer.Option(
            "--include-evidence/--no-include-evidence",
            help="Include evidence quotes in the query prompt.",
        ),
    ] = True,
) -> None:
    """Run constrained decoding on a single query for debugging."""
    from genret.paper_parsing import (
        FacetEvidence,
        Facets,
        evidence_from_type,
        facets_from_type,
    )
    from genret.training.data import format_facets_prompt
    from genret.training.decoding import ConstrainedDecoder
    from genret.utils import iter_jsonl

    if not query:
        die("Provide --query with a corpus ID or free-text query.")

    processed = _validate_data_files(splits_dir)
    index_path = processed / "docid_index.json"

    logger.info("=== GenRet Generate ===")

    model, tokeniser = _load_checkpoint(checkpoint)

    decoder = ConstrainedDecoder(
        index_path=index_path,
        model=model,
        tokeniser=tokeniser,
        num_beams=beams,
        max_new_tokens=256,
    )

    # Determine if query is a corpus ID or free text
    try:
        corpus_id = int(query)
        facets_path = resolve_jsonl(processed / "facets.jsonl.gz")
        target_facets: Facets | None = None
        target_evidence: FacetEvidence | None = None

        for raw in iter_jsonl(facets_path):
            f = facets_from_type(raw, facets_type)
            if f.corpusid == corpus_id:
                target_facets = f
                if include_evidence:
                    target_evidence = evidence_from_type(raw, facets_type)
                break

        if target_facets is None:
            die(f"No facets found for corpus ID {corpus_id}")

        messages = format_facets_prompt(target_facets, evidence=target_evidence)
        logger.info("Query paper: %d", corpus_id)
    except ValueError:
        messages = [
            {
                "role": "system",
                "content": "Generate the DocID of the most relevant paper.",
            },
            {"role": "user", "content": query},
        ]

    logger.info("Running constrained beam search...")
    results = decoder.decode_messages(messages)

    print(f"\nTop results ({len(results)} beams completed):")
    for i, result in enumerate(results[:20]):
        decoded_text = tokeniser.decode(
            result.generated_tokens, skip_special_tokens=False
        )
        print(
            f"  [{i + 1}] score={result.score:.3f}  "
            f"papers={len(result.paper_ids)}  "
            f"ids={result.paper_ids[:5]}"
            f"{'...' if len(result.paper_ids) > 5 else ''}"
        )
        print(f"       tokens: {decoded_text[:200]}")


def register_train_commands(app: typer.Typer) -> None:
    """Register core training commands on the given Typer app."""
    app.command(name="train")(cmd_train)
    app.command(name="evaluate")(cmd_evaluate)
    app.command(name="generate")(cmd_generate)


__all__ = [
    "cmd_evaluate",
    "cmd_generate",
    "cmd_train",
    "register_train_commands",
]
