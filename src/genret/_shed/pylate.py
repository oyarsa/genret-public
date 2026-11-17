"""Typed wrappers for PyLate (ColBERT late-interaction models).

PyLate pins an older transformers than the project, so it is not a project
dependency. Code importing this module runs in an isolated environment:

    uv run --no-project --python 3.14 --with-editable . --with pylate ...
"""

# pyright: basic

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from datasets import Dataset  # type: ignore[import-not-found]
from pylate import losses, models, utils  # type: ignore[import-not-found]
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers


def load_colbert(
    model_name: str,
    *,
    query_length: int,
    document_length: int,
    device: str | None = None,
) -> Any:
    """Load a PyLate ColBERT model.

    Args:
        model_name: HuggingFace model ID or local checkpoint path.
        query_length: Maximum query tokens (queries shorter than this are padded
            with [MASK] expansion tokens).
        document_length: Maximum document tokens.
        device: Device to use (None = auto-detect).

    Returns:
        The loaded `pylate.models.ColBERT` instance.
    """
    return models.ColBERT(
        model_name_or_path=model_name,
        query_length=query_length,
        document_length=document_length,
        device=device,
    )


def encode_multivector(
    model: Any,
    texts: Sequence[str],
    *,
    is_query: bool,
    batch_size: int,
) -> list[npt.NDArray[np.float32]]:
    """Encode texts into per-token L2-normalised embeddings.

    Args:
        model: Model returned by `load_colbert`.
        texts: Texts to encode.
        is_query: Encode as queries (query marker, [MASK] expansion) rather
            than documents.
        batch_size: Texts per forward pass.

    Returns:
        One `(num_tokens, dim)` array per text.
    """
    embeddings = model.encode(
        list(texts),
        is_query=is_query,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    return [np.asarray(e, dtype=np.float32) for e in embeddings]


def train_contrastive(
    model: Any,
    queries: Sequence[str],
    positives: Sequence[str],
    *,
    output_dir: Path,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    save_steps: int,
    bf16: bool,
    seed: int = 0,
) -> None:
    """Fine-tune a ColBERT model with in-batch-negative contrastive loss.

    Batches never repeat a query or positive text, so a paper appearing twice
    in one batch is not scored as its own negative. Intermediate checkpoints
    are written to `output_dir/checkpoint-<step>` and the final model to
    `output_dir/final`; all load with `load_colbert`.

    Args:
        model: Model returned by `load_colbert`.
        queries: Query texts, aligned with `positives`.
        positives: Positive document texts.
        output_dir: Directory for checkpoints and the final model.
        max_steps: Optimiser steps (the dataset is cycled as needed).
        batch_size: Pairs per batch (in-batch negatives = batch_size - 1).
        learning_rate: Peak learning rate.
        save_steps: Checkpoint interval in steps.
        bf16: Train in bfloat16 mixed precision.
        seed: Seed for shuffling and initialisation.
    """
    dataset = Dataset.from_dict({"query": list(queries), "positive": list(positives)})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        bf16=bf16,
        save_strategy="steps",
        save_steps=save_steps,
        save_only_model=True,
        logging_steps=50,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        seed=seed,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=losses.Contrastive(model=model),
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    trainer.train()
    model.save_pretrained(str(output_dir / "final"))
