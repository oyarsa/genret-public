# pyright: basic
# ruff: noqa: PLC0415  Defer torch import to allow CPU-only environments
"""Dataset and collator for GLEN rank-distillation training.

Each logical example groups all of a single query's retrieval targets
(under one facet) into a single (input, [(target, teacher_weight)])
record. The collator expands these into per-target rows tagged with a
group id so the custom :class:`DistillTrainer` can compute one KL loss
per group.

Differs from :mod:`glen_dataset` in three ways:

- Groups retrieval pairs by ``(paper_a, facet)`` rather than emitting
  one (query, single positive) pair at a time.
- Carries per-target teacher weights (RRF rank contributions) from the
  ``weight`` field of the training-pairs JSONL.
- Defaults to ``indexing_ratio = 0`` — indexing pairs are excluded
  unless explicitly opted in, since the goal of distillation is to
  push the model past its indexing-memorisation failure mode.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from genret._shed.transformers import (
    apply_chat_template_ids,
    get_eos_token_id,
    get_pad_token_id,
)
from genret.paper_parsing import FacetRecordType, Facets, facets_from_type
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_dataset import _build_messages, _facet_text
from genret.training.glen_docids import normalise_facet_text, split_bullets
from genret.types import CorpusId
from genret.utils import iter_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class GlenDistillTarget:
    """One (target_text, teacher_weight) within a distillation group."""

    target_text: str
    weight: float


@dataclass(frozen=True, kw_only=True)
class GlenDistillExample:
    """A single distillation example: one query + multiple weighted targets.

    The dataset emits these as logical units; the collator below
    expands each into per-target rows for batched forward passes.
    """

    prompt_messages: list[dict[str, str]]
    targets: list[GlenDistillTarget]


@dataclass(frozen=True, kw_only=True)
class GlenDistillDatasetConfig:
    """Configuration for the GLEN distillation dataset."""

    pairs_path: Path
    """JSONL produced by ``build-codebook-data`` with rank-derived weights."""
    facets_path: Path
    """JSONL with per-paper facet bullets."""
    facets_type: FacetRecordType = FacetRecordType.STANDARD
    max_targets_per_query: int = 20
    """Cap on K = number of targets per logical example. Larger K =
    richer signal but linearly more memory per logical example."""
    indexing_ratio: float = 0.0
    """Fraction of logical examples that are indexing (single-target,
    weight=1). Default 0: distillation runs on retrieval pairs only."""
    split_ids: set[CorpusId] | None = None
    """If set, restrict to query papers in this split."""


class GlenDistillDataset:
    """Dataset of distillation logical examples."""

    def __init__(
        self,
        config: GlenDistillDatasetConfig,
        *,
        rng: random.Random,
    ) -> None:
        self._examples = _build_distill_examples(config, rng=rng)

    def __len__(self) -> int:  # pragma: no cover
        """Return the number of logical examples."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> GlenDistillExample:  # pragma: no cover
        """Return the logical example at ``idx``."""
        return self._examples[idx]


def _bullet_targets(facets: Facets, facet: str, rng: random.Random) -> str:
    """Sample one bullet from ``facets``' ``facet`` field as a target."""
    raw = _facet_text(facets, facet, None)
    bullets = split_bullets(raw)
    if not bullets:
        return ""
    return normalise_facet_text(rng.choice(bullets))


def _build_distill_examples(
    config: GlenDistillDatasetConfig,
    *,
    rng: random.Random,
) -> list[GlenDistillExample]:
    """Group training pairs by (paper_a, facet) and emit distill examples."""
    facets_map: dict[CorpusId, Facets] = {}
    for raw in iter_jsonl(config.facets_path):
        f = facets_from_type(raw, config.facets_type)
        facets_map[f.corpusid] = f

    # Group retrieval pairs by (paper_a, facet)
    groups: dict[tuple[CorpusId, str], list[tuple[CorpusId, float]]] = defaultdict(list)
    indexing_rows: list[tuple[CorpusId, str]] = []
    skipped = 0
    split_ids = config.split_ids

    for row in iter_jsonl(config.pairs_path):
        paper_a = CorpusId(int(row["paper_a"]))
        paper_b = CorpusId(int(row["paper_b"]))
        facet = str(row["facet"])
        kind = str(row["kind"])
        weight = float(row.get("weight", 1.0))

        if facet not in FACET_NAMES:
            continue
        if split_ids is not None and paper_a not in split_ids:
            continue

        if paper_a not in facets_map or paper_b not in facets_map:
            skipped += 1
            continue

        if kind == "indexing":
            indexing_rows.append((paper_a, facet))
        else:
            groups[(paper_a, facet)].append((paper_b, weight))

    if skipped:
        logger.info("Skipped %d distill rows (missing facets)", skipped)

    examples: list[GlenDistillExample] = []

    for (paper_a, facet), members in groups.items():
        facets_a = facets_map[paper_a]
        input_text = _facet_text(facets_a, facet, None)
        if not input_text:
            continue

        # Sort by weight descending, cap to max_targets_per_query
        members.sort(key=lambda x: -x[1])
        top_members = members[: config.max_targets_per_query]

        targets: list[GlenDistillTarget] = []
        for paper_b, weight in top_members:
            facets_b = facets_map[paper_b]
            target_text = _bullet_targets(facets_b, facet, rng)
            if not target_text:
                continue
            targets.append(GlenDistillTarget(target_text=target_text, weight=weight))

        if not targets:
            continue

        examples.append(
            GlenDistillExample(
                prompt_messages=_build_messages(facet, input_text, kind="retrieval"),
                targets=targets,
            )
        )

    indexing_examples: list[GlenDistillExample] = []
    if config.indexing_ratio > 0.0:
        for paper_a, facet in indexing_rows:
            facets_a = facets_map[paper_a]
            input_text = _facet_text(facets_a, facet, None)
            if not input_text:
                continue
            target_text = _bullet_targets(facets_a, facet, rng)
            if not target_text:
                continue
            indexing_examples.append(
                GlenDistillExample(
                    prompt_messages=_build_messages(facet, input_text, kind="indexing"),
                    targets=[GlenDistillTarget(target_text=target_text, weight=1.0)],
                )
            )

    # Mix at configured ratio
    if config.indexing_ratio >= 1.0:
        combined = indexing_examples
    elif config.indexing_ratio <= 0.0 or not indexing_examples:
        combined = examples
    else:
        n_ret = len(examples)
        n_idx_keep = int(n_ret * config.indexing_ratio / (1.0 - config.indexing_ratio))
        rng.shuffle(indexing_examples)
        combined = examples + indexing_examples[:n_idx_keep]

    rng.shuffle(combined)

    logger.info(
        "GLEN distill dataset: %d examples (%d retrieval, %d indexing kept)",
        len(combined),
        len(examples),
        len(combined) - len(examples),
    )
    return combined


def distill_collator(
    examples: Sequence[GlenDistillExample],
    tokeniser: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> dict[str, Any]:
    """Expand logical examples into per-target rows for the model.

    For each logical example with K targets, emits K rows (each row
    being one tokenised ``input + target + EOS`` sequence). Adds
    ``group_starts``, ``group_sizes``, and ``teacher_weights_flat``
    fields so the custom Trainer can compute one KL loss per group.

    Returns:
        Dict with keys:
            - ``input_ids`` (B*K, T) — left-padded token IDs
            - ``attention_mask`` (B*K, T) — 1 for real tokens
            - ``labels`` (B*K, T) — -100 for prompt + padding, target
              tokens otherwise
            - ``group_starts`` (B,) — index in B*K where each logical
              example's rows start
            - ``group_sizes`` (B,) — number of rows per logical example
            - ``teacher_weights_flat`` (B*K,) — per-row teacher weight
    """
    import torch

    eos_id = get_eos_token_id(tokeniser)
    pad_id_val = get_pad_token_id(tokeniser)
    assert pad_id_val is not None
    pad_id = pad_id_val

    all_rows: list[list[int]] = []
    all_labels: list[list[int]] = []
    teacher_weights: list[float] = []
    group_starts: list[int] = []
    group_sizes: list[int] = []

    for ex in examples:
        prompt_ids = apply_chat_template_ids(tokeniser, ex.prompt_messages)
        group_starts.append(len(all_rows))
        group_sizes.append(len(ex.targets))

        for tgt in ex.targets:
            target_ids = tokeniser.encode(tgt.target_text, add_special_tokens=False)
            full_ids = prompt_ids + target_ids + [eos_id]

            if len(full_ids) > max_seq_length:
                full_ids = full_ids[:max_seq_length]
                full_ids[-1] = eos_id

            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            all_rows.append(full_ids)
            all_labels.append(labels)
            teacher_weights.append(tgt.weight)

    max_len = max(len(r) for r in all_rows)
    padded_input_ids: list[list[int]] = []
    padded_labels: list[list[int]] = []
    attention_masks: list[list[int]] = []

    for ids, labels in zip(all_rows, all_labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append([pad_id] * pad_len + ids)
        padded_labels.append([-100] * pad_len + labels)
        attention_masks.append([0] * pad_len + [1] * len(ids))

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "group_starts": torch.tensor(group_starts, dtype=torch.long),
        "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
        "teacher_weights_flat": torch.tensor(teacher_weights, dtype=torch.float32),
    }


__all__ = [
    "GlenDistillDataset",
    "GlenDistillDatasetConfig",
    "GlenDistillExample",
    "GlenDistillTarget",
    "distill_collator",
]
