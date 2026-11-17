# pyright: basic
# ruff: noqa: PLC0415  Defer torch import to allow CPU-only environments
"""Dataset and collator for GLEN marginal-likelihood training (Family B).

For each gold ``(paper_a bullet, paper_b)`` retrieval link, emit one logical
example whose target is the **set** of paper_b's bullets (one DocID per
bullet of the gold paper). The loss collapses this set with a logsumexp,
so every DocID for paper_b receives equal credit and the model is no
longer pushed away from valid alternatives by CE on a single sampled
target.

Shape mirrors :mod:`glen_distill_dataset` — same group_starts /
group_sizes plumbing — but without teacher weights. The collator emits
one row per target per logical example.
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
from genret.paper_parsing import FacetEvidence, FacetRecordType, Facets
from genret.training.codebook_data import FACET_NAMES
from genret.training.glen_bullet_dataset import (
    _build_bullet_messages,
    _render_bullet,
    _word_count,
)
from genret.training.glen_docids import normalise_facet_text, split_bullets
from genret.types import CorpusId
from genret.utils import iter_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class GlenMarginalExample:
    """One marginal-likelihood logical example.

    The prompt is a single paper_a bullet input. The targets are all of
    paper_b's bullets for the same facet. The collator expands each
    example into len(targets) rows that share a group id.
    """

    prompt_messages: list[dict[str, str]]
    targets: list[str]


@dataclass(frozen=True, kw_only=True)
class GlenMarginalDatasetConfig:
    """Configuration for the marginal-likelihood dataset."""

    pairs_path: Path
    facets_path: Path
    paper_graphs_path: Path
    facets_type: FacetRecordType = FacetRecordType.STANDARD
    split_ids: set[CorpusId] | None = None
    facets: tuple[str, ...] = FACET_NAMES
    """Facets whose bullets become training inputs. Defaults to all four."""
    source_bullets: str = "all"
    """``'all'`` emits one example per paper_a bullet (D9 default);
    ``'longest'`` emits a single example using paper_a's longest bullet."""


class GlenMarginalDataset:
    """Logical examples for marginal-likelihood training."""

    def __init__(
        self, config: GlenMarginalDatasetConfig, *, rng: random.Random
    ) -> None:
        self._examples = _build_marginal_examples(config, rng=rng)

    def __len__(self) -> int:  # pragma: no cover
        """Return number of logical examples."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> GlenMarginalExample:  # pragma: no cover
        """Return the logical example at index ``idx``."""
        return self._examples[idx]


def _paper_b_target_set(facets: Facets, facet: str) -> list[str]:
    """Return the full set of normalised bullet DocIDs for ``facet``."""
    raw_text = ""
    items = getattr(facets, facet)
    for item in items:
        raw_text += f"- {item}\n"
    bullets = split_bullets(raw_text)
    return [normalise_facet_text(b) for b in bullets if b.strip()]


def _build_marginal_examples(
    config: GlenMarginalDatasetConfig,
    *,
    rng: random.Random,
) -> list[GlenMarginalExample]:
    """Group retrieval pairs by (paper_a, paper_b, facet) and emit examples."""
    from genret.paper_parsing import evidence_from_type, facets_from_type

    facets_map: dict[CorpusId, Facets] = {}
    evidence_map: dict[CorpusId, FacetEvidence] = {}
    for raw in iter_jsonl(config.facets_path):
        f = facets_from_type(raw, config.facets_type)
        facets_map[f.corpusid] = f
        if ev := evidence_from_type(raw, config.facets_type):
            evidence_map[f.corpusid] = ev

    titles: dict[CorpusId, str] = {}
    for record in iter_jsonl(config.paper_graphs_path):
        cid = CorpusId(int(record["corpusid"]))
        title = record.get("title")
        if isinstance(title, str) and title.strip():
            titles[cid] = title.strip()

    # Group (paper_a, paper_b, facet) gold links.
    links: dict[tuple[CorpusId, CorpusId, str], None] = defaultdict(lambda: None)
    split_ids = config.split_ids
    for row in iter_jsonl(config.pairs_path):
        if str(row.get("kind", "")) != "retrieval":
            continue
        facet = str(row["facet"])
        if facet not in FACET_NAMES or facet not in config.facets:
            continue
        paper_a = CorpusId(int(row["paper_a"]))
        paper_b = CorpusId(int(row["paper_b"]))
        if split_ids is not None and paper_a not in split_ids:
            continue
        links[(paper_a, paper_b, facet)] = None

    examples: list[GlenMarginalExample] = []
    skipped = 0

    for paper_a, paper_b, facet in links:
        facets_a = facets_map.get(paper_a)
        facets_b = facets_map.get(paper_b)
        if facets_a is None or facets_b is None:
            skipped += 1
            continue

        target_set = _paper_b_target_set(facets_b, facet)
        if not target_set:
            skipped += 1
            continue

        title_a = titles.get(paper_a, "")
        evidence_a = evidence_map.get(paper_a)
        items = getattr(facets_a, facet)
        if not items:
            skipped += 1
            continue

        if config.source_bullets == "longest":
            best_i = max(range(len(items)), key=lambda i: _word_count(str(items[i])))
            source_indices = [best_i]
        else:
            source_indices = list(range(len(items)))

        for i in source_indices:
            bullet_text = _render_bullet(facets_a, facet, i, evidence_a)
            if not bullet_text:
                continue
            messages = _build_bullet_messages(facet, title_a, bullet_text)
            examples.append(
                GlenMarginalExample(prompt_messages=messages, targets=list(target_set))
            )

    if skipped:
        logger.info("Skipped %d marginal links (missing facets or empty)", skipped)

    logger.info(
        "GlenMarginalDataset: %d examples from %d gold links",
        len(examples),
        len(links),
    )
    return examples


def marginal_collator(
    examples: Sequence[GlenMarginalExample],
    tokeniser: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> dict[str, Any]:
    """Expand logical examples into per-target rows.

    Mirrors the distill collator shape (``group_starts`` / ``group_sizes``)
    so the trainer can locate each example's target group, but emits no
    teacher weights — the marginal loss is uniform over the target set.
    """
    import torch

    eos_id = get_eos_token_id(tokeniser)
    pad_id_val = get_pad_token_id(tokeniser)
    assert pad_id_val is not None
    pad_id = pad_id_val

    all_rows: list[list[int]] = []
    all_labels: list[list[int]] = []
    group_starts: list[int] = []
    group_sizes: list[int] = []

    for ex in examples:
        prompt_ids = apply_chat_template_ids(tokeniser, ex.prompt_messages)
        group_starts.append(len(all_rows))
        group_sizes.append(len(ex.targets))

        for target_text in ex.targets:
            target_ids = tokeniser.encode(target_text, add_special_tokens=False)
            full_ids = prompt_ids + target_ids + [eos_id]

            if len(full_ids) > max_seq_length:
                full_ids = full_ids[:max_seq_length]
                full_ids[-1] = eos_id

            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            all_rows.append(full_ids)
            all_labels.append(labels)

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
    }


__all__ = [
    "GlenMarginalDataset",
    "GlenMarginalDatasetConfig",
    "GlenMarginalExample",
    "marginal_collator",
]
