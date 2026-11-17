# pyright: basic
"""Multi-label classification dataset for the paper-recommendation sanity check.

For each ``(paper_a, paper_a bullet)`` input row, the label is a multi-hot
vector over the corpus where the 1s mark every gold neighbour of ``paper_a``
(across all facets, since at inference we score on the same union). Trained
with sigmoid + binary cross-entropy.

The input prompt format matches the D9 per-bullet GLEN dataset so model
comparisons isolate the architecture change, not the input shape.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from genret.paper_parsing import (
    FacetEvidence,
    FacetRecordType,
    Facets,
    evidence_from_type,
    facets_from_type,
)
from genret.training.codebook_data import FACET_NAMES
from genret.training.glen_bullet_dataset import _build_bullet_messages, _render_bullet
from genret.types import CorpusId
from genret.utils import iter_jsonl

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ClassifierDatasetConfig:
    """Configuration for the multi-label classifier dataset."""

    pairs_path: Path
    """Training pairs JSONL from build-codebook-data."""
    facets_path: Path
    """JSONL with per-paper facet bullets and optional evidence."""
    paper_graphs_path: Path
    """JSONL with per-paper metadata; must have 'title' and 'corpusid' fields."""
    corpus_split_path: Path
    """One corpus ID per line. Defines the fixed label ordering (head size)."""
    facets_type: FacetRecordType = FacetRecordType.STANDARD
    train_split_ids: set[CorpusId] | None = None
    """Restrict ``paper_a`` of training rows to this set."""
    facets: tuple[str, ...] = FACET_NAMES
    """Facets whose paper_a bullets become training inputs. Defaults to all."""


@dataclass(frozen=True, kw_only=True)
class ClassifierExample:
    """One training row: prompt messages plus multi-hot label."""

    prompt_messages: list[dict[str, str]]
    labels: torch.Tensor
    """Float multi-hot tensor of shape ``(corpus_size,)``."""


class ClassifierDataset:
    """Per-bullet inputs paired with multi-hot label vectors over the corpus."""

    def __init__(self, config: ClassifierDatasetConfig, *, rng: random.Random) -> None:
        self.corpus_ids: list[CorpusId] = _load_corpus_ids(config.corpus_split_path)
        self.corpus_to_label_idx: dict[CorpusId, int] = {
            cid: idx for idx, cid in enumerate(self.corpus_ids)
        }
        self._examples = _build_examples(config, self.corpus_to_label_idx, rng)

    def __len__(self) -> int:  # pragma: no cover
        """Return the number of training rows."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> ClassifierExample:  # pragma: no cover
        """Return the training row at index ``idx``."""
        return self._examples[idx]

    @property
    def num_classes(self) -> int:
        """Number of output classes (= corpus size)."""
        return len(self.corpus_ids)

    def class_positive_counts(self) -> torch.Tensor:
        """Return ``(num_classes,)`` int tensor: positives per class in the dataset.

        Sums the multi-hot label vectors across all examples. Used by the
        training command to derive pos-weight and bias-init priors.
        """
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        for ex in self._examples:
            counts += ex.labels
        return counts


def _load_corpus_ids(corpus_split_path: Path) -> list[CorpusId]:
    """Return corpus IDs in the order defined by the split file."""
    ids: list[CorpusId] = []
    for raw_line in corpus_split_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        ids.append(CorpusId(int(line)))
    return ids


def _load_titles(paper_graphs_path: Path) -> dict[CorpusId, str]:
    titles: dict[CorpusId, str] = {}
    for record in iter_jsonl(paper_graphs_path):
        cid = CorpusId(int(record["corpusid"]))
        title = record.get("title")
        if isinstance(title, str) and title.strip():
            titles[cid] = title.strip()
    return titles


def _build_examples(
    config: ClassifierDatasetConfig,
    corpus_to_label_idx: dict[CorpusId, int],
    rng: random.Random,
) -> list[ClassifierExample]:
    """Stream training pairs, group by paper_a, emit per-bullet multi-label rows."""
    facets_map: dict[CorpusId, Facets] = {}
    evidence_map: dict[CorpusId, FacetEvidence] = {}
    for raw in iter_jsonl(config.facets_path):
        f = facets_from_type(raw, config.facets_type)
        facets_map[f.corpusid] = f
        if ev := evidence_from_type(raw, config.facets_type):
            evidence_map[f.corpusid] = ev

    titles = _load_titles(config.paper_graphs_path)

    # Group gold neighbours by paper_a (across all facets).
    gold_neighbours: dict[CorpusId, set[CorpusId]] = {}
    split_ids = config.train_split_ids
    for row in iter_jsonl(config.pairs_path):
        if str(row.get("kind", "")) != "retrieval":
            continue
        paper_a = CorpusId(int(row["paper_a"]))
        paper_b = CorpusId(int(row["paper_b"]))
        if split_ids is not None and paper_a not in split_ids:
            continue
        if paper_b not in corpus_to_label_idx:
            # Gold target is outside our classification head; skip — the
            # model cannot represent it as a positive.
            continue
        gold_neighbours.setdefault(paper_a, set()).add(paper_b)

    num_classes = len(corpus_to_label_idx)
    examples: list[ClassifierExample] = []
    skipped = 0

    for paper_a, golds in gold_neighbours.items():
        facets_a = facets_map.get(paper_a)
        if facets_a is None:
            skipped += 1
            continue
        if not golds:
            skipped += 1
            continue

        labels = torch.zeros(num_classes, dtype=torch.float32)
        for gold in golds:
            labels[corpus_to_label_idx[gold]] = 1.0

        title_a = titles.get(paper_a, "")
        evidence_a = evidence_map.get(paper_a)
        for facet in config.facets:
            n_bullets = len(getattr(facets_a, facet))
            for i in range(n_bullets):
                bullet_text = _render_bullet(facets_a, facet, i, evidence_a)
                if not bullet_text:
                    continue
                messages = _build_bullet_messages(facet, title_a, bullet_text)
                examples.append(
                    ClassifierExample(prompt_messages=messages, labels=labels.clone())
                )

    if skipped:
        logger.info("Skipped %d paper_a entries (missing facets or empty)", skipped)
    logger.info(
        "ClassifierDataset: %d examples over %d paper_a queries (%d classes)",
        len(examples),
        len(gold_neighbours),
        num_classes,
    )
    return examples


__all__ = [
    "ClassifierDataset",
    "ClassifierDatasetConfig",
    "ClassifierExample",
]
