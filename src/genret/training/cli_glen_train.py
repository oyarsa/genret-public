# pyright: basic
# ruff: noqa: PLC0415  Enable function-level imports
"""Core GLEN training commands: standard CE, rank-distillation, split-loss."""

from __future__ import annotations

import functools
import logging
import random
from pathlib import Path
from typing import Annotated

import typer

from genret.paper_parsing import FacetRecordType
from genret.utils import die, load_ids_from_file

logger = logging.getLogger(__name__)


def cmd_train_glen(
    pairs: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help="Training pairs JSONL from build-codebook-data.",
            show_default=False,
        ),
    ],
    facets: Annotated[
        Path,
        typer.Option(
            "--facets", help="JSONL with per-paper facet records.", show_default=False
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Checkpoint output dir.")
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.1-8B-Instruct",
    epochs: Annotated[int, typer.Option("--epochs")] = 5,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 8,
    learning_rate: Annotated[float, typer.Option("--lr")] = 2e-4,
    lora_rank: Annotated[int, typer.Option("--lora-rank")] = 16,
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = 2,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length")] = 512,
    indexing_ratio: Annotated[
        float, typer.Option("--indexing-ratio", min=0.0, max=1.0)
    ] = 0.5,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 10,
    eval_steps: Annotated[int, typer.Option("--eval-steps")] = 200,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 200,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    train_split: Annotated[Path | None, typer.Option("--train-split")] = None,
    dev_split: Annotated[Path | None, typer.Option("--dev-split")] = None,
    resume_from_checkpoint: Annotated[
        Path | None,
        typer.Option(help="Resume training from this checkpoint directory."),
    ] = None,
) -> None:
    """Fine-tune a language model with QLoRA on GLEN training pairs."""
    from transformers import Trainer, TrainingArguments

    from genret.training._glen_callbacks import FinalStepCallback
    from genret.training.config import LoRAConfig, QuantConfig, TrainingConfig
    from genret.training.data import collate_fn
    from genret.training.glen_dataset import GlenDataset, GlenDatasetConfig
    from genret.training.model import setup_model_and_tokeniser
    from genret.types import CorpusId

    training_cfg = TrainingConfig(
        splits_dir=Path(),
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
        num_epochs=epochs,
        indexing_ratio=indexing_ratio,
    )
    lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_rank * 2)
    quant_cfg = QuantConfig()
    rng = random.Random(training_cfg.seed)

    # No special tokens needed — GLEN uses existing model vocabulary.
    model, tokeniser = setup_model_and_tokeniser(
        model_name, lora_cfg, quant_cfg, extra_special_tokens=[]
    )

    def _make_dataset(split_path: Path | None) -> GlenDataset:
        split_ids: set[CorpusId] | None = None
        if split_path is not None:
            if not split_path.exists():
                die(f"Split file not found: {split_path}")
            split_ids = load_ids_from_file(split_path)
            if not split_ids:
                die(f"Split file {split_path} contains no ids")
        return GlenDataset(
            GlenDatasetConfig(
                pairs_path=pairs,
                facets_path=facets,
                facets_type=facets_type,
                indexing_ratio=indexing_ratio,
                split_ids=split_ids,
            ),
            rng=rng,
        )

    train_dataset = _make_dataset(train_split)
    dev_dataset = _make_dataset(dev_split) if dev_split is not None else None

    bound_collate = functools.partial(
        collate_fn, tokeniser=tokeniser, max_seq_length=max_seq_length
    )

    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=training_cfg.warmup_ratio,
        bf16=True,
        logging_steps=logging_steps,
        eval_strategy="steps" if dev_dataset is not None else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=dev_dataset is not None,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=bound_collate,
        callbacks=[FinalStepCallback()],
    )

    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint)
        if resume_from_checkpoint
        else None
    )
    trainer.save_model(str(output))
    tokeniser.save_pretrained(str(output))


def cmd_train_glen_distill(
    pairs: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help=(
                "Training pairs JSONL from build-codebook-data with "
                "rank-derived per-pair weights. Re-run build-codebook-data "
                "if the existing file has weight=1.0 everywhere."
            ),
            show_default=False,
        ),
    ],
    facets: Annotated[
        Path,
        typer.Option(
            "--facets", help="JSONL with per-paper facet records.", show_default=False
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Checkpoint output dir.")
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.1-8B-Instruct",
    epochs: Annotated[int, typer.Option("--epochs")] = 2,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help=(
                "Logical examples per device. Effective row count is "
                "batch_size * max_targets_per_query."
            ),
        ),
    ] = 1,
    learning_rate: Annotated[float, typer.Option("--lr")] = 2e-4,
    lora_rank: Annotated[int, typer.Option("--lora-rank")] = 16,
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = 4,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length")] = 512,
    max_targets_per_query: Annotated[
        int,
        typer.Option(
            "--max-targets-per-query",
            help="K = max targets per logical example. Memory scales linearly.",
        ),
    ] = 10,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            help="Softmax temperature on teacher weights. Smaller = peakier.",
        ),
    ] = 1.0,
    indexing_ratio: Annotated[
        float, typer.Option("--indexing-ratio", min=0.0, max=1.0)
    ] = 0.0,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 10,
    eval_steps: Annotated[int, typer.Option("--eval-steps")] = 200,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 200,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    train_split: Annotated[Path | None, typer.Option("--train-split")] = None,
    dev_split: Annotated[Path | None, typer.Option("--dev-split")] = None,
    resume_from_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--resume-from-checkpoint",
            help="Checkpoint directory to resume from. Skipped if not set.",
        ),
    ] = None,
) -> None:
    """Fine-tune a language model with QLoRA on GLEN rank-distillation.

    Each logical example groups one query's top-K teacher targets (with
    per-target weights) into a single training instance; the per-group
    loss is KL(teacher || student) over the K targets.

    See ``glen train`` for the standard CE-on-single-positive training.
    This command is the rank-distillation alternative tested as the
    immediate-baseline fix for the popularity-bias failure mode
    documented in ``docs/todos/genret_diagnostics.md``.
    """
    from transformers import (
        TrainingArguments,
    )

    from genret.training._glen_callbacks import FinalStepCallback
    from genret.training.config import LoRAConfig, QuantConfig, TrainingConfig
    from genret.training.glen_distill_dataset import (
        GlenDistillDataset,
        GlenDistillDatasetConfig,
        distill_collator,
    )
    from genret.training.glen_distill_loss import DistillTrainer
    from genret.training.model import setup_model_and_tokeniser
    from genret.types import CorpusId

    training_cfg = TrainingConfig(
        splits_dir=Path(),
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
        num_epochs=epochs,
        indexing_ratio=indexing_ratio,
    )
    lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_rank * 2)
    quant_cfg = QuantConfig()
    rng = random.Random(training_cfg.seed)

    model, tokeniser = setup_model_and_tokeniser(
        model_name, lora_cfg, quant_cfg, extra_special_tokens=[]
    )

    def _make_dataset(split_path: Path | None) -> GlenDistillDataset:
        split_ids: set[CorpusId] | None = None
        if split_path is not None:
            if not split_path.exists():
                die(f"Split file not found: {split_path}")
            split_ids = load_ids_from_file(split_path)
            if not split_ids:
                die(f"Split file {split_path} contains no ids")
        return GlenDistillDataset(
            GlenDistillDatasetConfig(
                pairs_path=pairs,
                facets_path=facets,
                facets_type=facets_type,
                max_targets_per_query=max_targets_per_query,
                indexing_ratio=indexing_ratio,
                split_ids=split_ids,
            ),
            rng=rng,
        )

    train_dataset = _make_dataset(train_split)
    dev_dataset = _make_dataset(dev_split) if dev_split is not None else None

    bound_collate = functools.partial(
        distill_collator, tokeniser=tokeniser, max_seq_length=max_seq_length
    )

    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=training_cfg.warmup_ratio,
        bf16=True,
        logging_steps=logging_steps,
        eval_strategy="steps" if dev_dataset is not None else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=dev_dataset is not None,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    trainer = DistillTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=bound_collate,
        callbacks=[FinalStepCallback()],
    )
    trainer.temperature = temperature

    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint)
        if resume_from_checkpoint
        else None
    )
    trainer.save_model(str(output))
    tokeniser.save_pretrained(str(output))


def cmd_train_glen_split_loss(
    pairs: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help="Training pairs JSONL from build-codebook-data.",
            show_default=False,
        ),
    ],
    facets: Annotated[
        Path,
        typer.Option(
            "--facets", help="JSONL with per-paper facet records.", show_default=False
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Checkpoint output dir.")
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.1-8B-Instruct",
    epochs: Annotated[int, typer.Option("--epochs")] = 2,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 8,
    learning_rate: Annotated[float, typer.Option("--lr")] = 2e-4,
    lora_rank: Annotated[int, typer.Option("--lora-rank")] = 16,
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = 2,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length")] = 512,
    indexing_ratio: Annotated[
        float, typer.Option("--indexing-ratio", min=0.0, max=1.0)
    ] = 0.5,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 10,
    eval_steps: Annotated[int, typer.Option("--eval-steps")] = 200,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 200,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    train_split: Annotated[Path | None, typer.Option("--train-split")] = None,
    dev_split: Annotated[Path | None, typer.Option("--dev-split")] = None,
    resume_from_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--resume-from-checkpoint",
            help="Checkpoint directory to resume from.",
        ),
    ] = None,
) -> None:
    """Fine-tune with QLoRA and log indexing vs retrieval loss separately (D6).

    Identical to ``glen train`` but uses ``SplitLossTrainer`` which logs
    ``train_loss_indexing`` and ``train_loss_retrieval`` to detect indexing-
    pair domination.  If ``train_loss_indexing`` converges to near zero while
    ``train_loss_retrieval`` plateaus, indexing pairs dominate training and
    the D4 indexing-ratio sweep becomes high-value.
    """
    from transformers import (
        TrainingArguments,
    )

    from genret.training._glen_callbacks import FinalStepCallback
    from genret.training.config import LoRAConfig, QuantConfig, TrainingConfig
    from genret.training.glen_dataset import GlenDataset, GlenDatasetConfig
    from genret.training.glen_split_loss import SplitLossTrainer, split_collate_fn
    from genret.training.model import setup_model_and_tokeniser
    from genret.types import CorpusId

    training_cfg = TrainingConfig(
        splits_dir=Path(),
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
        num_epochs=epochs,
        indexing_ratio=indexing_ratio,
    )
    lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_rank * 2)
    quant_cfg = QuantConfig()
    rng = random.Random(training_cfg.seed)

    model, tokeniser = setup_model_and_tokeniser(
        model_name, lora_cfg, quant_cfg, extra_special_tokens=[]
    )

    def _make_dataset(split_path: Path | None) -> GlenDataset:
        split_ids: set[CorpusId] | None = None
        if split_path is not None:
            if not split_path.exists():
                die(f"Split file not found: {split_path}")
            split_ids = load_ids_from_file(split_path)
            if not split_ids:
                die(f"Split file {split_path} contains no ids")
        return GlenDataset(
            GlenDatasetConfig(
                pairs_path=pairs,
                facets_path=facets,
                facets_type=facets_type,
                indexing_ratio=indexing_ratio,
                split_ids=split_ids,
            ),
            rng=rng,
        )

    train_dataset = _make_dataset(train_split)
    dev_dataset = _make_dataset(dev_split) if dev_split is not None else None

    bound_collate = functools.partial(
        split_collate_fn, tokeniser=tokeniser, max_seq_length=max_seq_length
    )

    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=training_cfg.warmup_ratio,
        bf16=True,
        logging_steps=logging_steps,
        eval_strategy="steps" if dev_dataset is not None else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=dev_dataset is not None,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SplitLossTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=bound_collate,
        callbacks=[FinalStepCallback()],
    )

    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint)
        if resume_from_checkpoint
        else None
    )
    trainer.save_model(str(output))
    tokeniser.save_pretrained(str(output))


def register(app: typer.Typer) -> None:
    """Attach core training commands to the GLEN typer app."""
    app.command(name="train")(cmd_train_glen)
    app.command(name="train-distill")(cmd_train_glen_distill)
    app.command(name="train-split-loss")(cmd_train_glen_split_loss)


__all__ = [
    "cmd_train_glen",
    "cmd_train_glen_distill",
    "cmd_train_glen_split_loss",
    "register",
]
