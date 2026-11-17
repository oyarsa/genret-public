# pyright: basic
# ruff: noqa: PLC0415  Enable function-level imports
"""Paper-level GLEN training commands: paper-title, paper-marginal, titles."""

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


def cmd_train_glen_paper_title(
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
            "--facets",
            help="JSONL with per-paper facet records (kept for parity).",
            show_default=False,
        ),
    ],
    paper_graphs: Annotated[
        Path,
        typer.Option(
            "--paper-graphs",
            help="JSONL with per-paper title + abstract metadata.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Checkpoint output dir.")
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.2-1B-Instruct",
    epochs: Annotated[int, typer.Option("--epochs")] = 4,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 4,
    learning_rate: Annotated[float, typer.Option("--lr")] = 2e-4,
    lora_rank: Annotated[int, typer.Option("--lora-rank")] = 16,
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = 4,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length")] = 512,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 20,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 500,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 0.0,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    train_split: Annotated[Path | None, typer.Option("--train-split")] = None,
    resume_from_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--resume-from-checkpoint",
            help="Checkpoint directory to resume from.",
        ),
    ] = None,
    facets_subset: Annotated[
        str | None,
        typer.Option("--facets-subset"),
    ] = None,
    full_finetune: Annotated[
        bool,
        typer.Option("--full-finetune/--no-full-finetune"),
    ] = False,
    max_abstract_chars: Annotated[int, typer.Option("--max-abstract-chars")] = 1500,
) -> None:
    """Paper-level title-DocID GLEN training (Variant 1 alone).

    Input = paper_a title + abstract. Target = one paper_b title per row.
    Single-target CE loss with the standard ``Trainer`` and ``collate_fn``.
    """
    from transformers import Trainer, TrainingArguments

    from genret.training._glen_callbacks import FinalStepCallback
    from genret.training.codebook_data import FACET_NAMES as _FN
    from genret.training.config import LoRAConfig, QuantConfig
    from genret.training.data import collate_fn
    from genret.training.glen_paper_title_dataset import (
        GlenPaperTitleDataset,
        GlenPaperTitleDatasetConfig,
    )
    from genret.training.model import setup_model_and_tokeniser
    from genret.types import CorpusId

    rng = random.Random(0)

    facets_arg = (
        tuple(f.strip() for f in facets_subset.split(",") if f.strip())
        if facets_subset
        else _FN
    )

    lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_rank * 2)
    quant_cfg = QuantConfig()
    model, tokeniser = setup_model_and_tokeniser(
        model_name,
        lora_cfg,
        quant_cfg,
        extra_special_tokens=[],
        full_finetune=full_finetune,
    )

    split_ids: set[CorpusId] | None = None
    if train_split is not None:
        if not train_split.exists():
            die(f"Split file not found: {train_split}")
        split_ids = load_ids_from_file(train_split)
        if not split_ids:
            die(f"Split file {train_split} contains no ids")

    train_dataset = GlenPaperTitleDataset(
        GlenPaperTitleDatasetConfig(
            pairs_path=pairs,
            facets_path=facets,
            paper_graphs_path=paper_graphs,
            facets_type=facets_type,
            split_ids=split_ids,
            facets=facets_arg,
            max_abstract_chars=max_abstract_chars,
        ),
        rng=rng,
    )

    bound_collate = functools.partial(
        collate_fn, tokeniser=tokeniser, max_seq_length=max_seq_length
    )

    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        save_only_model=True,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if full_finetune else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
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


def cmd_train_glen_paper_marginal(
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
            "--facets",
            help="JSONL with per-paper facet records (kept for signature parity).",
            show_default=False,
        ),
    ],
    paper_graphs: Annotated[
        Path,
        typer.Option(
            "--paper-graphs",
            help="JSONL with per-paper title + abstract metadata.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Checkpoint output dir.")
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.2-1B-Instruct",
    epochs: Annotated[int, typer.Option("--epochs")] = 4,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 2,
    learning_rate: Annotated[float, typer.Option("--lr")] = 2e-4,
    lora_rank: Annotated[int, typer.Option("--lora-rank")] = 16,
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = 4,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length")] = 512,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 10,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 200,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 0.0,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    train_split: Annotated[Path | None, typer.Option("--train-split")] = None,
    resume_from_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--resume-from-checkpoint",
            help="Checkpoint directory to resume from.",
        ),
    ] = None,
    facets_subset: Annotated[
        str | None,
        typer.Option(
            "--facets-subset",
            help=(
                "Comma-separated facet names to collect gold links from "
                "(default: all four). Each example's target set is the "
                "union of paper_b titles across the selected facets."
            ),
        ),
    ] = None,
    full_finetune: Annotated[
        bool,
        typer.Option(
            "--full-finetune/--no-full-finetune",
            help="Skip QLoRA and fine-tune all weights in bf16.",
        ),
    ] = False,
    max_abstract_chars: Annotated[
        int,
        typer.Option(
            "--max-abstract-chars",
            help="Cap on raw abstract length before tokenisation.",
        ),
    ] = 1500,
) -> None:
    """Paper-level multi-positive marginal-likelihood training (Variants 1 + 3).

    One example per paper_a query: input = title + abstract, targets =
    titles of all gold paper_b neighbours across the selected facets.
    Loss = ``-log Σ p(title_i | input)`` per example.
    """
    from transformers import TrainingArguments

    from genret.training._glen_callbacks import FinalStepCallback
    from genret.training.codebook_data import FACET_NAMES as _FN
    from genret.training.config import LoRAConfig, QuantConfig
    from genret.training.glen_marginal_loss import MarginalLikelihoodTrainer
    from genret.training.glen_paper_marginal_dataset import (
        GlenPaperMarginalDataset,
        GlenPaperMarginalDatasetConfig,
        paper_marginal_collator,
    )
    from genret.training.model import setup_model_and_tokeniser
    from genret.types import CorpusId

    rng = random.Random(0)

    facets_arg = (
        tuple(f.strip() for f in facets_subset.split(",") if f.strip())
        if facets_subset
        else _FN
    )

    lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_rank * 2)
    quant_cfg = QuantConfig()
    model, tokeniser = setup_model_and_tokeniser(
        model_name,
        lora_cfg,
        quant_cfg,
        extra_special_tokens=[],
        full_finetune=full_finetune,
    )

    split_ids: set[CorpusId] | None = None
    if train_split is not None:
        if not train_split.exists():
            die(f"Split file not found: {train_split}")
        split_ids = load_ids_from_file(train_split)
        if not split_ids:
            die(f"Split file {train_split} contains no ids")

    # ``facets`` arg is kept on the dataset config for API parity but it
    # isn't actually read in the paper-level builder — keep the var alive
    # by passing it through so the linter doesn't flag the unused field.
    _ = facets

    train_dataset = GlenPaperMarginalDataset(
        GlenPaperMarginalDatasetConfig(
            pairs_path=pairs,
            facets_path=facets,
            paper_graphs_path=paper_graphs,
            facets_type=facets_type,
            split_ids=split_ids,
            facets=facets_arg,
            max_abstract_chars=max_abstract_chars,
        ),
        rng=rng,
    )

    bound_collate = functools.partial(
        paper_marginal_collator,
        tokeniser=tokeniser,
        max_seq_length=max_seq_length,
    )

    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        save_only_model=True,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if full_finetune else "adamw_torch",
        remove_unused_columns=False,
    )

    trainer = MarginalLikelihoodTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
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


def cmd_train_glen_titles(
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
    paper_graphs: Annotated[
        Path,
        typer.Option(
            "--paper-graphs",
            help="JSONL with per-paper metadata (must include 'title').",
            show_default=False,
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
    facets_subset: Annotated[
        str | None,
        typer.Option(
            "--facets-subset",
            help=(
                "Comma-separated facet names to train on (e.g. 'contributions'). "
                "Defaults to all four."
            ),
        ),
    ] = None,
    full_finetune: Annotated[
        bool,
        typer.Option(
            "--full-finetune/--no-full-finetune",
            help=(
                "Skip QLoRA and fine-tune all weights. The base model is "
                "loaded in bfloat16 without quantisation."
            ),
        ),
    ] = False,
) -> None:
    """Fine-tune with QLoRA (default) or full FT to generate paper titles given facet text (D5).

    Retrieval pairs only; the target is the normalised title of paper_b rather
    than its facet-text DocID.  Combine with a title-based GLEN index (built by
    ``build-title-index``) for evaluation.
    """
    from transformers import Trainer, TrainingArguments

    from genret.training._glen_callbacks import FinalStepCallback
    from genret.training.codebook_data import FACET_NAMES
    from genret.training.config import LoRAConfig, QuantConfig, TrainingConfig
    from genret.training.data import collate_fn
    from genret.training.glen_title_index import (
        GlenTitleDataset,
        GlenTitleDatasetConfig,
    )
    from genret.training.model import setup_model_and_tokeniser
    from genret.types import CorpusId

    training_cfg = TrainingConfig(
        splits_dir=Path(),
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
        num_epochs=epochs,
        indexing_ratio=0.0,
    )
    lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_rank * 2)
    quant_cfg = QuantConfig()
    rng = random.Random(training_cfg.seed)

    model, tokeniser = setup_model_and_tokeniser(
        model_name,
        lora_cfg,
        quant_cfg,
        extra_special_tokens=[],
        full_finetune=full_finetune,
    )

    def _make_dataset(split_path: Path | None) -> GlenTitleDataset:
        split_ids: set[CorpusId] | None = None
        if split_path is not None:
            if not split_path.exists():
                die(f"Split file not found: {split_path}")
            split_ids = load_ids_from_file(split_path)
            if not split_ids:
                die(f"Split file {split_path} contains no ids")
        facets_arg = (
            tuple(f.strip() for f in facets_subset.split(",") if f.strip())
            if facets_subset
            else FACET_NAMES
        )
        return GlenTitleDataset(
            GlenTitleDatasetConfig(
                pairs_path=pairs,
                facets_path=facets,
                paper_graphs_path=paper_graphs,
                facets_type=facets_type,
                split_ids=split_ids,
                facets=facets_arg,
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
        optim="paged_adamw_8bit" if full_finetune else "adamw_torch",
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


def register(app: typer.Typer) -> None:
    """Attach paper-level training commands to the GLEN typer app."""
    app.command(name="train-paper-title")(cmd_train_glen_paper_title)
    app.command(name="train-paper-marginal")(cmd_train_glen_paper_marginal)
    app.command(name="train-titles")(cmd_train_glen_titles)


__all__ = [
    "cmd_train_glen_paper_marginal",
    "cmd_train_glen_paper_title",
    "cmd_train_glen_titles",
    "register",
]
