# pyright: basic
"""Training data construction and fine-tuning loop for the cross-encoder re-ranker.

Builds positive pairs from co-citation / bib-coupling signals and hard negatives
from the candidate graph, then fine-tunes a DeBERTa-v3 sequence classifier.
"""

import json
import logging
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import override

import numpy as np
import numpy.typing as npt

from genret._shed.scipy import spearman_correlation
from genret.paper_parsing import Facets
from genret.processing.facet_evaluation import FACET_NAMES
from genret.processing.reranker import format_paper_pair, load_facets_map, load_titles
from genret.types import CorpusId
from genret.utils import (
    iter_jsonl,
    open_auto,
    resolve_jsonl,
    sample,
    track_progress,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_EPOCHS = 3
"""Default number of training epochs."""
DEFAULT_BATCH_SIZE = 16
"""Default per-device training batch size."""
DEFAULT_LR = 2e-5
"""Default learning rate."""
DEFAULT_WARMUP_RATIO = 0.06
"""Default fraction of total steps used for linear warmup."""
DEFAULT_DEV_FRACTION = 0.1
"""Default fraction of data held out for evaluation."""
DEFAULT_SEED = 42
"""Default random seed for reproducibility."""
DEFAULT_MAX_LENGTH = 512
"""Default maximum token length for input sequences."""

DEFAULT_BASE_MODEL = "microsoft/deberta-v3-small"
"""Default HuggingFace model ID for the cross-encoder base."""


class PairSource(StrEnum):
    """Source signal that produced a training pair."""

    CO_CITATION = "co_citation"
    """Pair from co-citation count above threshold."""
    BIB_COUPLING = "bib_coupling"
    """Pair from bibliographic coupling count above threshold."""
    HARD_NEGATIVE = "hard_negative"
    """Dense candidate with no citation relationship."""


@dataclass(frozen=True, kw_only=True)
class TrainingPair:
    """A single training example for the cross-encoder re-ranker."""

    text: str
    """Formatted paper pair string from `format_paper_pair`."""
    label: int
    """1 = related, 0 = unrelated."""
    source: PairSource
    """Which signal produced this pair."""


type CitationKey = tuple[CorpusId, CorpusId]


def _collect_positives(
    co_cit_path: Path,
    bib_path: Path,
    co_cit_min: int,
    bib_min: int,
    facets_map: dict[CorpusId, Facets],
    titles: dict[CorpusId, str],
) -> tuple[set[CitationKey], list[tuple[CitationKey, PairSource]]]:
    """Collect positive pairs from citation signals.

    Args:
        co_cit_path: Path to co_citations JSONL.
        bib_path: Path to bib_coupling JSONL.
        co_cit_min: Minimum co-citation count for a positive.
        bib_min: Minimum shared references for a positive.
        facets_map: Corpus ID to facets mapping.
        titles: Corpus ID to title mapping.

    Returns:
        Tuple of
        - all citation pair keys for negative exclusion,
        - list of (canonical_key, source) positive pairs
    """
    all_citation_keys: set[CitationKey] = set()
    positives: dict[CitationKey, PairSource] = {}

    def _has_data(cid: CorpusId) -> bool:
        """Check whether a paper has both facets and a title."""
        return cid in facets_map and cid in titles

    # Co-citations
    for record in track_progress(
        iter_jsonl(co_cit_path), desc="Loading co-citations..."
    ):
        a = CorpusId(int(record["paper_a"]))
        b = CorpusId(int(record["paper_b"]))
        if not _has_data(a) or not _has_data(b):
            continue

        key = (min(a, b), max(a, b))
        all_citation_keys.add(key)

        count = float(record["count"])
        if count >= co_cit_min:
            positives[key] = PairSource.CO_CITATION

    # Bib-coupling
    for record in track_progress(iter_jsonl(bib_path), desc="Loading bib-coupling..."):
        a = CorpusId(int(record["paper_a"]))
        b = CorpusId(int(record["paper_b"]))
        if not _has_data(a) or not _has_data(b):
            continue

        key = (min(a, b), max(a, b))
        all_citation_keys.add(key)

        shared = float(record["shared_refs"])
        if shared >= bib_min and key not in positives:
            positives[key] = PairSource.BIB_COUPLING

    positives_list = list(positives.items())
    logger.info(f"Collected {len(positives_list)} positive pairs")
    return all_citation_keys, positives_list


def _collect_hard_negatives(
    candidates_dir: Path,
    all_citation_keys: set[CitationKey],
    facets_map: dict[CorpusId, Facets],
    titles: dict[CorpusId, str],
) -> list[CitationKey]:
    """Collect hard negative pairs from candidate NPZ files.

    A hard negative is a dense candidate pair that has no citation relationship
    at all (not in co-citation or bib-coupling at any count).

    Args:
        candidates_dir: Directory containing `facet_candidates_*.npz` files.
        all_citation_keys: Set of all `(min_id, max_id)` pairs from citations.
        facets_map: Corpus ID to facets mapping.
        titles: Corpus ID to title mapping.

    Returns:
        Deduplicated list of `(min_id, max_id)` hard negative pairs.
    """
    negative_keys: set[CitationKey] = set()

    for facet_name in FACET_NAMES:
        npz_path = candidates_dir / f"facet_candidates_{facet_name}.npz"
        if not npz_path.exists():
            logger.warning(f"Missing {npz_path}, skipping")
            continue

        with np.load(npz_path) as npz:
            corpusids: npt.NDArray[np.int64] = npz["corpusids"]
            candidates: npt.NDArray[np.int32] = npz["candidates"]

        n = len(corpusids)
        for i in range(n):
            cid_i = CorpusId(int(corpusids[i]))
            if cid_i not in facets_map or cid_i not in titles:
                continue

            for j_idx in range(candidates.shape[1]):
                j = int(candidates[i, j_idx])
                if j < 0 or j >= n:
                    continue

                cid_j = CorpusId(int(corpusids[j]))
                if cid_j not in facets_map or cid_j not in titles:
                    continue

                key = (min(cid_i, cid_j), max(cid_i, cid_j))
                if key not in all_citation_keys:
                    negative_keys.add(key)

    logger.info(f"Collected {len(negative_keys)} unique hard negative pairs")
    return list(negative_keys)


def _format_pairs(
    pairs: Iterable[tuple[CitationKey, int, PairSource]],
    facets_map: dict[CorpusId, Facets],
    titles: dict[CorpusId, str],
) -> list[TrainingPair]:
    """Format raw pairs into `TrainingPair` objects.

    Args:
        pairs: Iterable of (canonical_key, label, source) tuples.
        facets_map: Corpus ID to facets mapping.
        titles: Corpus ID to title mapping.

    Returns:
        List of formatted training pairs.
    """
    result: list[TrainingPair] = []

    for (a, b), label, source in pairs:
        text = format_paper_pair(
            title_a=titles[a],
            facets_a=facets_map[a],
            title_b=titles[b],
            facets_b=facets_map[b],
        )
        result.append(TrainingPair(text=text, label=label, source=source))

    return result


def build_training_pairs(
    facets_path: Path,
    candidates_dir: Path,
    paper_graphs_path: Path,
    co_cit_path: Path,
    bib_path: Path,
    output_path: Path,
    *,
    co_cit_min: int = 3,
    bib_min: int = 5,
    negative_ratio: int = 3,
    seed: int = DEFAULT_SEED,
) -> int:
    """Build and write training pairs for the cross-encoder re-ranker.

    Collects positive pairs from citation signals and hard negatives from the
    candidate graph, formats them, shuffles, and writes to a JSONL file.

    Args:
        facets_path: Path to facets JSONL.
        candidates_dir: Directory containing `facet_candidates_*.npz` files.
        paper_graphs_path: Path to paper_graphs JSONL (for titles).
        co_cit_path: Path to co_citations JSONL.
        bib_path: Path to bib_coupling JSONL.
        output_path: Output JSONL path for training pairs.
        co_cit_min: Minimum co-citation count for a positive pair.
        bib_min: Minimum shared references for a positive pair.
        negative_ratio: Hard negative to positive ratio.
        seed: Random seed for negative sampling and shuffling.

    Returns:
        Total number of training pairs written.
    """
    rng = random.Random(seed)

    facets_map = load_facets_map(facets_path)
    titles = load_titles(paper_graphs_path)

    all_citation_keys, positive_list = _collect_positives(
        co_cit_path, bib_path, co_cit_min, bib_min, facets_map, titles
    )
    negative_keys = _collect_hard_negatives(
        candidates_dir, all_citation_keys, facets_map, titles
    )

    # Sample negatives
    max_negs = negative_ratio * len(positive_list)
    negative_keys = sample(rng, negative_keys, max_negs)

    logger.info(f"Using {len(negative_keys)} hard negatives ({negative_ratio}x ratio)")

    # Build raw pair tuples: (key, label, source)
    raw_pairs: list[tuple[CitationKey, int, PairSource]] = []
    for key, source in positive_list:
        raw_pairs.append((key, 1, source))
    for key in negative_keys:
        raw_pairs.append((key, 0, PairSource.HARD_NEGATIVE))

    rng.shuffle(raw_pairs)

    # Format and write
    logger.info(f"Formatting {len(raw_pairs)} pairs...")
    training_pairs = _format_pairs(raw_pairs, facets_map, titles)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_auto(output_path, "w") as fh:
        for pair in training_pairs:
            fh.write(json.dumps(asdict(pair)) + "\n")

    logger.info(f"Wrote {len(training_pairs)} training pairs to {output_path}")
    return len(training_pairs)


class TrainingMode(StrEnum):
    """Whether to train with binary labels or continuous distillation scores."""

    BINARY = "binary"
    """Binary classification with BCEWithLogitsLoss."""
    DISTILLATION = "distillation"
    """Regression with MSE loss against continuous 0-1 scores."""


def _train_core(
    pairs_path: Path,
    output_dir: Path,
    mode: TrainingMode,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    gradient_accumulation_steps: int = 1,
    lr: float = DEFAULT_LR,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: int = DEFAULT_SEED,
) -> None:
    """Shared training loop for both binary and distillation modes.

    Loads pairs, splits train/dev, tokenises, builds datasets, trains with
    HuggingFace `Trainer`, saves the best checkpoint, and writes parameters.

    Args:
        pairs_path: Path to training pairs JSONL.
        output_dir: Directory to write the checkpoint and `train_params.json`.
        mode: Training objective (binary classification or distillation).
        base_model: HuggingFace model ID for the base cross-encoder.
        max_epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        max_length: Maximum token length for input sequences.
        gradient_accumulation_steps: Number of steps to accumulate gradients
            before updating weights (effective batch = batch_size * this).
        lr: Learning rate.
        warmup_ratio: Fraction of total steps for linear warmup.
        dev_fraction: Fraction of data to hold out for evaluation.
        seed: Random seed.
    """
    import torch  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EvalPrediction,
        Trainer,
        TrainingArguments,
    )

    logger.info(f"Loading training pairs from {pairs_path}")
    records = list(iter_jsonl(resolve_jsonl(pairs_path)))
    texts = [r["text"] for r in records]
    labels = [float(r["label"]) for r in records]

    if mode is TrainingMode.DISTILLATION:
        logger.info(f"Loaded {len(records)} pairs (mean label={np.mean(labels):.3f})")
    else:
        logger.info(f"Loaded {len(records)} pairs ({int(sum(labels))} positive)")

    # Train/dev split
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    dev_size = int(len(indices) * dev_fraction)
    dev_indices = set(indices[:dev_size])
    train_indices = [i for i in indices if i not in dev_indices]

    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    dev_texts = [texts[i] for i in dev_indices]
    dev_labels = [labels[i] for i in dev_indices]
    logger.info(f"Split: {len(train_texts)} train, {len(dev_texts)} dev")

    # Tokenise lazily in __getitem__ to avoid holding all token IDs in memory.
    # With 150k pairs at 2048 tokens, eager tokenisation uses ~5GB+ of RAM.
    # DataCollatorWithPadding pads per-batch at collation time.
    use_fast = "deberta" not in base_model.lower()
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=use_fast, trust_remote_code=True
    )
    assert tokenizer is not None

    class _LazyPairDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
        """Dataset that tokenises on-the-fly to keep memory usage constant."""

        def __init__(self, pair_texts: list[str], pair_labels: list[float]) -> None:
            self._texts = pair_texts
            self._labels = pair_labels

        def __len__(self) -> int:
            return len(self._labels)

        @override
        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            """Tokenise and return a single example with its label."""
            enc = tokenizer(
                self._texts[index],
                truncation=True,
                max_length=max_length,
            )
            return {k: torch.tensor(v) for k, v in enc.items()} | {
                "labels": torch.tensor(self._labels[index], dtype=torch.float32)
            }

    train_dataset = _LazyPairDataset(train_texts, train_labels)
    dev_dataset = _LazyPairDataset(dev_texts, dev_labels)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Deferred: _shed.transformers pulls torch, which must stay out of the
    # CLI import graph (the explorer serves torch-free)
    from genret._shed.transformers import predictions_array  # noqa: PLC0415

    # Mode-specific configuration
    if mode is TrainingMode.DISTILLATION:
        problem_type = "regression"
        best_metric = "spearman_rho"

        def _compute_metrics(pred: EvalPrediction) -> dict[str, float]:
            """Compute Spearman correlation and MSE on dev predictions."""
            preds = predictions_array(pred)
            labels_arr = np.asarray(pred.label_ids, dtype=np.float64)
            mse = float(np.mean((preds - labels_arr) ** 2))
            rho = spearman_correlation(preds, labels_arr)
            return {"mse": mse, "spearman_rho": rho}
    else:
        problem_type = "multi_label_classification"
        best_metric = "auc_roc"

        def _compute_metrics(pred: EvalPrediction) -> dict[str, float]:
            """Compute AUC-ROC on dev predictions."""
            logits = predictions_array(pred)
            return {"auc_roc": float(roc_auc_score(pred.label_ids, logits))}

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        problem_type=problem_type,
        trust_remote_code=True,
    )

    checkpoint_dir = output_dir / "checkpoint"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        seed=seed,
        logging_steps=100,
        report_to="none",
    )

    # Save tokeniser at the output_dir level (parent of checkpoint dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    logger.info(f"Starting {mode.value} training...")
    trainer.train()

    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    logger.info(f"Saved best checkpoint to {checkpoint_dir}")

    params = {
        "base_model": base_model,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "dev_fraction": dev_fraction,
        "seed": seed,
        "num_train": len(train_texts),
        "num_dev": len(dev_texts),
        "training_type": mode.value,
    }
    if mode is TrainingMode.DISTILLATION:
        params["mean_label"] = float(np.mean(labels))
    else:
        params["num_positive"] = int(sum(labels))

    params_path = output_dir / "train_params.json"
    params_path.write_text(json.dumps(params, indent=2) + "\n")
    logger.info(f"Wrote {params_path}")


def train_distilled(
    pairs_path: Path,
    output_dir: Path,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    gradient_accumulation_steps: int = 1,
    lr: float = DEFAULT_LR,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: int = DEFAULT_SEED,
) -> None:
    """Train a cross-encoder distilled from LLM judge scores.

    Uses MSE loss against continuous 0-1 scores from the LLM judge, rather
    than binary classification.

    Args:
        pairs_path: Path to training pairs JSONL with continuous `label` field.
        output_dir: Directory to write the checkpoint and `train_params.json`.
        base_model: HuggingFace model ID for the base cross-encoder.
        max_epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        max_length: Maximum token length for input sequences.
        gradient_accumulation_steps: Steps to accumulate before weight update.
        lr: Learning rate.
        warmup_ratio: Fraction of total steps for linear warmup.
        dev_fraction: Fraction of data to hold out for evaluation.
        seed: Random seed.
    """
    _train_core(
        pairs_path,
        output_dir,
        TrainingMode.DISTILLATION,
        base_model=base_model,
        max_epochs=max_epochs,
        batch_size=batch_size,
        max_length=max_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr=lr,
        warmup_ratio=warmup_ratio,
        dev_fraction=dev_fraction,
        seed=seed,
    )


def train(
    pairs_path: Path,
    output_dir: Path,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    gradient_accumulation_steps: int = 1,
    lr: float = DEFAULT_LR,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: int = DEFAULT_SEED,
) -> None:
    """Fine-tune a cross-encoder re-ranker on pre-built training pairs.

    Uses HuggingFace `Trainer` with `BCEWithLogitsLoss`. Evaluates AUC-ROC on
    a held-out dev set and saves the best checkpoint.

    Args:
        pairs_path: Path to training pairs JSONL (from `build_training_pairs`).
        output_dir: Directory to write the checkpoint and `train_params.json`.
        base_model: HuggingFace model ID for the base cross-encoder.
        max_epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        max_length: Maximum token length for input sequences.
        gradient_accumulation_steps: Steps to accumulate before weight update.
        lr: Learning rate.
        warmup_ratio: Fraction of total steps for linear warmup.
        dev_fraction: Fraction of data to hold out for evaluation.
        seed: Random seed.
    """
    _train_core(
        pairs_path,
        output_dir,
        TrainingMode.BINARY,
        base_model=base_model,
        max_epochs=max_epochs,
        batch_size=batch_size,
        max_length=max_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr=lr,
        warmup_ratio=warmup_ratio,
        dev_fraction=dev_fraction,
        seed=seed,
    )
