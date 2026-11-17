# pyright: basic
# ruff: noqa: PLC0415  Defer torch import to allow CPU-only environments
"""Paper-level multi-positive marginal-likelihood dataset (Variants 1 + 3).

Departs from the bullet-based dataset on two axes:

- **Query**: paper_a's *title + abstract* is the input. One example per
  paper_a, not per (paper_a, bullet) row. Denser per-query gradient
  signal; richer semantic context per query.
- **DocID**: each paper_b is identified by its *normalised title* (built
  by :func:`genret.training.glen_title_index.build_title_index`). The trie
  has 1 leaf per paper (≈1000 leaves at 1k scale) instead of 4 per paper
  for bullet DocIDs.

The loss is uniform-weighted marginal log-likelihood over paper_a's full
gold-neighbour set across all facets — one logsumexp per example.
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
from genret.paper_parsing import FacetRecordType
from genret.training.codebook_data import FACET_NAMES
from genret.training.glen_title_index import normalise_title
from genret.types import CorpusId
from genret.utils import iter_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a generative retrieval system. Given a research paper's "
    "title and abstract, generate the title of a related paper."
)


def _build_paper_messages(title: str, abstract: str) -> list[dict[str, str]]:
    """Return chat messages for a paper-level retrieval query."""
    user = f"Title: {title}"
    if abstract:
        user += f"\nAbstract: {abstract}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@dataclass(frozen=True, kw_only=True)
class GlenPaperMarginalExample:
    """One marginal-likelihood example at the paper level.

    ``targets`` is the union of paper_b titles for every (paper_a, paper_b)
    gold link across all facets — the model is taught that any paper in
    this set is a valid retrieval target.
    """

    prompt_messages: list[dict[str, str]]
    targets: list[str]


@dataclass(frozen=True, kw_only=True)
class GlenPaperMarginalDatasetConfig:
    """Configuration for the paper-level marginal dataset."""

    pairs_path: Path
    facets_path: Path
    paper_graphs_path: Path
    facets_type: FacetRecordType = FacetRecordType.STANDARD
    split_ids: set[CorpusId] | None = None
    facets: tuple[str, ...] = FACET_NAMES
    """Restrict gold-link collection to these facets (default: all)."""
    max_abstract_chars: int = 1500
    """Cap on raw abstract length before tokenisation. ~1500 chars ≈ 350
    tokens for English; combined with the title it leaves room for the
    chat-template overhead within a 512-token budget."""


class GlenPaperMarginalDataset:
    """Logical examples for paper-level multi-positive marginal training."""

    def __init__(
        self,
        config: GlenPaperMarginalDatasetConfig,
        *,
        rng: random.Random,
    ) -> None:
        self._examples = _build_examples(config, rng=rng)

    def __len__(self) -> int:  # pragma: no cover
        """Number of logical examples (one per paper_a query)."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> GlenPaperMarginalExample:  # pragma: no cover
        """Return the logical example at ``idx``."""
        return self._examples[idx]


def _load_paper_metadata(
    paper_graphs_path: Path, *, max_abstract_chars: int
) -> tuple[dict[CorpusId, str], dict[CorpusId, str]]:
    """Return ``(raw_titles, abstracts)`` keyed by corpus id."""
    titles: dict[CorpusId, str] = {}
    abstracts: dict[CorpusId, str] = {}
    for record in iter_jsonl(paper_graphs_path):
        cid = CorpusId(int(record["corpusid"]))
        title = record.get("title")
        if isinstance(title, str) and title.strip():
            titles[cid] = title.strip()
        abstract = record.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            abstracts[cid] = abstract.strip()[:max_abstract_chars]
    return titles, abstracts


def _build_examples(
    config: GlenPaperMarginalDatasetConfig,
    *,
    rng: random.Random,
) -> list[GlenPaperMarginalExample]:
    """Group retrieval pairs by paper_a, emit one example per query."""
    raw_titles, abstracts = _load_paper_metadata(
        config.paper_graphs_path, max_abstract_chars=config.max_abstract_chars
    )

    # Collect paper_b gold neighbours per paper_a across all facets.
    gold: dict[CorpusId, set[CorpusId]] = defaultdict(set)
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
        if paper_b not in raw_titles:
            # No DocID available — model can't be asked to generate it.
            continue
        gold[paper_a].add(paper_b)

    examples: list[GlenPaperMarginalExample] = []
    skipped = 0
    for paper_a, gold_set in gold.items():
        title_a = raw_titles.get(paper_a, "")
        if not title_a:
            skipped += 1
            continue
        abstract_a = abstracts.get(paper_a, "")
        targets = [
            normalise_title(raw_titles[p_b])
            for p_b in sorted(gold_set)
            if p_b in raw_titles
        ]
        if not targets:
            skipped += 1
            continue
        examples.append(
            GlenPaperMarginalExample(
                prompt_messages=_build_paper_messages(title_a, abstract_a),
                targets=targets,
            )
        )

    if skipped:
        logger.info("Skipped %d paper queries (missing title or no gold)", skipped)
    logger.info(
        "GlenPaperMarginalDataset: %d examples (avg %.1f targets per example)",
        len(examples),
        sum(len(e.targets) for e in examples) / max(len(examples), 1),
    )
    return examples


def paper_marginal_collator(
    examples: Sequence[GlenPaperMarginalExample],
    tokeniser: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> dict[str, Any]:
    """Expand examples into per-target rows for the model.

    Mirrors :func:`glen_marginal_dataset.marginal_collator` — emits
    ``group_starts`` / ``group_sizes`` so the trainer knows which rows
    share a query. No per-target weights; the marginal loss is uniform.
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
    "GlenPaperMarginalDataset",
    "GlenPaperMarginalDatasetConfig",
    "GlenPaperMarginalExample",
    "paper_marginal_collator",
]
