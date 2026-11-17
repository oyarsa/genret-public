"""The per-bullet D9 dataset wrapper over the example builder."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from genret.training.data import TrainingExample

from .builder import _build_bullet_examples
from .config import GlenBulletDatasetConfig

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class GlenBulletDataset:
    """Per-bullet D9 dataset: one example per (paper_a, paper_b, facet, bullet_i)."""

    def __init__(
        self,
        config: GlenBulletDatasetConfig,
        *,
        rng: random.Random,
        tokeniser: PreTrainedTokenizerBase | None = None,
    ) -> None:
        """Build the example list.

        Args:
            config: Dataset configuration.
            rng: RNG used for target sampling and augmentation.
            tokeniser: Required when ``config.target_smoothing_epsilon > 0``;
                used to tokenise target strings exactly as ``collate_fn``
                does so soft-target positions align with training labels.
        """
        self._examples = _build_bullet_examples(config, rng=rng, tokeniser=tokeniser)

    def __len__(self) -> int:  # pragma: no cover
        """Return number of training examples."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> TrainingExample:  # pragma: no cover
        """Return the training example at index ``idx``."""
        return self._examples[idx]
