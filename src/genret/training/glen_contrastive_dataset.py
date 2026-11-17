# pyright: basic
# ruff: noqa: PLC0415  Defer torch import to allow CPU-only environments
"""Dataset and collator for GLEN contrastive training with popularity-aware negatives.

Extends the marginal-likelihood setup by adding ``K`` negative targets per
example, sampled with weights proportional to paper popularity in the
training data. The InfoNCE loss then forces the model to assign higher
probability to the gold paper's bullets than to the popular-but-non-gold
ones — directly attacking the popularity-collapse failure mode.

Each example carries:

- ``positive_targets``: paper_b's bullets (the gold paper's DocIDs).
- ``negative_targets``: one bullet each from ``K`` non-gold papers,
  sampled with popularity weights.

The collator expands both lists into rows and tags each row's polarity
in ``pos_mask`` so the loss can separate the two halves.
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
class GlenContrastiveExample:
    """One contrastive logical example: query + positive + negative bullets.

    ``positive_targets`` are the gold paper_b's bullets for this facet.
    ``negative_targets`` are one bullet each from ``K`` non-gold papers,
    sampled with popularity weights at construction time.
    """

    prompt_messages: list[dict[str, str]]
    positive_targets: list[str]
    negative_targets: list[str]


@dataclass(frozen=True, kw_only=True)
class GlenContrastiveDatasetConfig:
    """Configuration for the contrastive dataset."""

    pairs_path: Path
    facets_path: Path
    paper_graphs_path: Path
    facets_type: FacetRecordType = FacetRecordType.STANDARD
    split_ids: set[CorpusId] | None = None
    facets: tuple[str, ...] = FACET_NAMES
    source_bullets: str = "all"
    num_negatives: int = 8
    """Number of negative papers sampled per example. One bullet per
    sampled paper becomes a negative target."""
    popularity_temperature: float = 1.0
    """Temperature on the popularity distribution used for negative
    sampling. ``1.0`` is proportional to raw counts; lower values
    sharpen toward the most-popular papers (e.g. BERT), higher values
    flatten toward uniform."""
    seed: int = 0


class GlenContrastiveDataset:
    """Logical examples for contrastive training."""

    def __init__(
        self,
        config: GlenContrastiveDatasetConfig,
        *,
        rng: random.Random,
    ) -> None:
        self._examples = _build_contrastive_examples(config, rng=rng)

    def __len__(self) -> int:  # pragma: no cover
        """Return number of logical examples."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> GlenContrastiveExample:  # pragma: no cover
        """Return the example at index ``idx``."""
        return self._examples[idx]


def _paper_b_target_set(facets: Facets, facet: str) -> list[str]:
    """Return the full set of normalised bullet DocIDs for ``facet``."""
    raw_text = ""
    items = getattr(facets, facet)
    for item in items:
        raw_text += f"- {item}\n"
    bullets = split_bullets(raw_text)
    return [normalise_facet_text(b) for b in bullets if b.strip()]


def _build_contrastive_examples(
    config: GlenContrastiveDatasetConfig,
    *,
    rng: random.Random,
) -> list[GlenContrastiveExample]:
    """Build examples with positives + popularity-weighted negatives."""
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

    # First pass: collect retrieval links + per-paper popularity (paper_b counts).
    links: dict[tuple[CorpusId, CorpusId, str], None] = defaultdict(lambda: None)
    gold_neighbours: dict[CorpusId, set[CorpusId]] = defaultdict(set)
    paper_b_counts: dict[CorpusId, int] = defaultdict(int)
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
        gold_neighbours[paper_a].add(paper_b)
        paper_b_counts[paper_b] += 1

    # Build the negative-sampling pool: every paper that appears as a gold
    # target somewhere, weighted by frequency to the configured temperature.
    candidate_papers: list[CorpusId] = []
    candidate_weights: list[float] = []
    for paper_id, count in paper_b_counts.items():
        candidate_papers.append(paper_id)
        # weight = count ** (1 / temperature). Lower T sharpens toward the most
        # popular papers; higher T flattens toward uniform.
        weight = float(count) ** (1.0 / max(config.popularity_temperature, 1e-6))
        candidate_weights.append(weight)

    if not candidate_papers:
        logger.warning("No candidate papers for negative sampling.")
        return []

    examples: list[GlenContrastiveExample] = []
    skipped = 0

    for paper_a, paper_b, facet in links:
        facets_a = facets_map.get(paper_a)
        facets_b = facets_map.get(paper_b)
        if facets_a is None or facets_b is None:
            skipped += 1
            continue

        positive_targets = _paper_b_target_set(facets_b, facet)
        if not positive_targets:
            skipped += 1
            continue

        items = getattr(facets_a, facet)
        if not items:
            skipped += 1
            continue

        # Sample ``num_negatives`` papers that are NOT gold for paper_a.
        # Resample to avoid in-gold collisions; cap attempts to stay bounded.
        gold_set = gold_neighbours[paper_a]
        negatives: list[CorpusId] = []
        attempts = 0
        max_attempts = config.num_negatives * 8
        while len(negatives) < config.num_negatives and attempts < max_attempts:
            cand = rng.choices(candidate_papers, weights=candidate_weights, k=1)[0]
            if cand in (paper_a, paper_b) or cand in gold_set:
                attempts += 1
                continue
            negatives.append(cand)
            attempts += 1

        # Pick one bullet per negative paper (sampled uniformly).
        negative_targets: list[str] = []
        for neg_id in negatives:
            neg_facets = facets_map.get(neg_id)
            if neg_facets is None:
                continue
            neg_bullets = _paper_b_target_set(neg_facets, facet)
            if not neg_bullets:
                continue
            negative_targets.append(rng.choice(neg_bullets))

        if not negative_targets:
            # Without negatives this example is identical to marginal-LL —
            # skip to avoid an unbalanced batch.
            skipped += 1
            continue

        title_a = titles.get(paper_a, "")
        evidence_a = evidence_map.get(paper_a)
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
                GlenContrastiveExample(
                    prompt_messages=messages,
                    positive_targets=list(positive_targets),
                    negative_targets=list(negative_targets),
                )
            )

    if skipped:
        logger.info("Skipped %d contrastive links (missing facets or empty)", skipped)

    logger.info(
        "GlenContrastiveDataset: %d examples (avg %d pos + %d neg)",
        len(examples),
        config.num_negatives,
        config.num_negatives,
    )
    return examples


def contrastive_collator(
    examples: Sequence[GlenContrastiveExample],
    tokeniser: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> dict[str, Any]:
    """Expand into per-target rows, tagging polarity in ``pos_mask``.

    Each logical example contributes ``len(positive_targets) + len(negative_targets)``
    rows. ``group_starts`` / ``group_sizes`` locate each example's full row
    span; ``pos_mask`` (B*K,) is 1 for positive rows and 0 for negatives.
    The trainer reads both to compute per-group InfoNCE.
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
    pos_mask: list[int] = []

    for ex in examples:
        prompt_ids = apply_chat_template_ids(tokeniser, ex.prompt_messages)
        group_starts.append(len(all_rows))
        group_sizes.append(len(ex.positive_targets) + len(ex.negative_targets))

        for is_positive, target_text in (
            *((1, t) for t in ex.positive_targets),
            *((0, t) for t in ex.negative_targets),
        ):
            target_ids = tokeniser.encode(target_text, add_special_tokens=False)
            full_ids = prompt_ids + target_ids + [eos_id]
            if len(full_ids) > max_seq_length:
                full_ids = full_ids[:max_seq_length]
                full_ids[-1] = eos_id

            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            all_rows.append(full_ids)
            all_labels.append(labels)
            pos_mask.append(is_positive)

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
        "pos_mask": torch.tensor(pos_mask, dtype=torch.long),
    }


__all__ = [
    "GlenContrastiveDataset",
    "GlenContrastiveDatasetConfig",
    "GlenContrastiveExample",
    "contrastive_collator",
]
