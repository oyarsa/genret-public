# pyright: basic
"""Dataset constructing GLEN-style per-facet training examples.

Like the codebook dataset but the generation target is the normalised
facet text of the target paper rather than a pair of cluster-ID tokens.
No special tokens are needed: the model generates from its existing
vocabulary, constrained at inference by a prefix trie.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from genret.paper_parsing import FacetRecordType, Facets, facets_from_type
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_dataset import (
    _build_messages,
    _facet_text,
)
from genret.training.data import TrainingExample, _mix_examples
from genret.training.glen_docids import normalise_facet_text, split_bullets
from genret.types import CorpusId
from genret.utils import iter_jsonl

logger = logging.getLogger(__name__)


def _target_text(facets: Facets, facet: str, rng: random.Random) -> str:
    """Return a randomly sampled bullet as the generation target."""
    raw = _facet_text(facets, facet, None)
    bullets = split_bullets(raw)
    if not bullets:
        return ""
    return normalise_facet_text(rng.choice(bullets))


@dataclass(frozen=True, kw_only=True)
class GlenDatasetConfig:
    """Configuration for the GLEN training dataset."""

    pairs_path: Path
    """JSONL produced by ``build-codebook-data`` (reused as-is)."""
    facets_path: Path
    """JSONL with per-paper facet bullets."""
    facets_type: FacetRecordType = FacetRecordType.STANDARD
    indexing_ratio: float = 0.5
    split_ids: set[CorpusId] | None = None
    """If set, restrict examples to pairs where ``paper_a`` is in this split."""


class GlenDataset:
    """Dataset that mixes GLEN indexing and retrieval examples."""

    def __init__(
        self,
        config: GlenDatasetConfig,
        *,
        rng: random.Random,
    ) -> None:
        self._examples = _build_glen_examples(config, rng=rng)

    def __len__(self) -> int:  # pragma: no cover
        """Return the number of training examples."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> TrainingExample:  # pragma: no cover
        """Return the training example at index ``idx``."""
        return self._examples[idx]


def _build_glen_examples(
    config: GlenDatasetConfig,
    *,
    rng: random.Random,
) -> list[TrainingExample]:
    """Load pairs + facets and produce ``TrainingExample`` rows."""
    facets_map: dict[CorpusId, Facets] = {}
    # evidence not needed for GLEN — targets are plain text
    for raw in iter_jsonl(config.facets_path):
        f = facets_from_type(raw, config.facets_type)
        facets_map[f.corpusid] = f

    indexing_examples: list[TrainingExample] = []
    retrieval_examples: list[TrainingExample] = []
    skipped = 0
    split_ids = config.split_ids

    for row in iter_jsonl(config.pairs_path):
        paper_a = CorpusId(int(row["paper_a"]))
        paper_b = CorpusId(int(row["paper_b"]))
        facet = str(row["facet"])
        kind = str(row["kind"])

        if facet not in FACET_NAMES:
            continue
        if split_ids is not None and paper_a not in split_ids:
            continue

        facets_a = facets_map.get(paper_a)
        facets_b = facets_map.get(paper_b)
        if facets_a is None or facets_b is None:
            skipped += 1
            continue

        input_text = _facet_text(facets_a, facet, None)
        if not input_text:
            skipped += 1
            continue
        target = _target_text(facets_b, facet, rng)
        if not target:
            skipped += 1
            continue

        example = TrainingExample(
            prompt_messages=_build_messages(facet, input_text, kind=kind),
            target_docid=target,
            kind=kind,
        )
        if kind == "indexing":
            indexing_examples.append(example)
        else:
            retrieval_examples.append(example)

    if skipped:
        logger.info("Skipped %d GLEN rows (missing facets or empty text)", skipped)

    result = _mix_examples(
        indexing_examples, retrieval_examples, config.indexing_ratio, rng
    )
    logger.info(
        "GLEN dataset: %d examples (%d indexing, %d retrieval) at indexing_ratio=%.2f",
        len(result),
        len(indexing_examples),
        len(retrieval_examples),
        config.indexing_ratio,
    )
    return result


__all__ = [
    "GlenDataset",
    "GlenDatasetConfig",
]
