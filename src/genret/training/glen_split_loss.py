# pyright: basic
# ruff: noqa: PLC0415  Defer torch import to allow CPU-only environments
"""Per-pair-type loss tracking for D6 diagnostic.

Wraps the standard CE collator and Trainer to log ``train_loss_indexing``
and ``train_loss_retrieval`` separately.  When indexing loss converges
quickly while retrieval loss plateaus, indexing-pair domination is
confirmed and the D4 indexing-ratio sweep becomes high-value.

Requires ``TrainingExample.kind`` to be set (either ``"indexing"`` or
``"retrieval"``).  The :class:`GlenDataset` sets this field; the
:class:`GlenBulletDataset` only produces retrieval examples so every
element will carry ``kind="retrieval"`` by default.

Implementation note: split losses are injected by overriding ``log()``
rather than via a ``TrainerCallback.on_log`` hook.  The hook fires
*after* ``log_history.append`` and console printing, so any keys added
there are silently discarded.  Overriding ``log()`` runs before the
snapshot, so the keys appear in trainer_state.json and on stdout.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, override

from transformers import Trainer

from genret.training.data import TrainingExample, collate_fn

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def split_collate_fn(
    examples: Sequence[TrainingExample],
    tokeniser: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> dict[str, Any]:
    """Collate with an extra ``kinds`` bool tensor (True = indexing)."""
    import torch

    batch = collate_fn(examples, tokeniser, max_seq_length)
    batch["kinds"] = torch.tensor(
        [e.kind == "indexing" for e in examples], dtype=torch.bool
    )
    return batch


class SplitLossTrainer(Trainer):
    """CE trainer that logs per-pair-type loss alongside the aggregate."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.idx_accum: float = 0.0
        self.idx_n: int = 0
        self.ret_accum: float = 0.0
        self.ret_n: int = 0

    @override
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Inject accumulated split-loss averages before the log snapshot."""
        if self.idx_n > 0:
            logs["train_loss_indexing"] = self.idx_accum / self.idx_n
            self.idx_accum = 0.0
            self.idx_n = 0
        if self.ret_n > 0:
            logs["train_loss_retrieval"] = self.ret_accum / self.ret_n
            self.ret_accum = 0.0
            self.ret_n = 0
        super().log(logs, start_time=start_time)

    @override
    def compute_loss(
        self,
        model: Module,
        inputs: dict[str, Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, Any]:
        """CE loss with per-kind accumulation for the log override."""
        import torch.nn.functional as functional

        kinds: Tensor = inputs.pop("kinds")
        labels: Tensor = inputs["labels"]

        outputs = model(**inputs)
        logits: Tensor = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Per-token CE, then average over non-masked positions per example.
        per_token = functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).clamp(min=0),
            reduction="none",
        ).view(shift_labels.shape)

        valid = (shift_labels != -100).float()
        n_valid = valid.sum(dim=-1).clamp(min=1)
        per_example = (per_token * valid).sum(dim=-1) / n_valid
        loss = per_example.mean()

        # Accumulate split losses; flushed in log() override above.
        if model.training:
            is_idx = kinds
            is_ret = ~kinds
            if is_idx.any():
                self.idx_accum += per_example[is_idx].mean().item()
                self.idx_n += 1
            if is_ret.any():
                self.ret_accum += per_example[is_ret].mean().item()
                self.ret_n += 1

        if return_outputs:
            return loss, outputs
        return loss


__all__ = ["SplitLossTrainer", "split_collate_fn"]
