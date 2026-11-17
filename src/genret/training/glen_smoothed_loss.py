# pyright: basic
# ruff: noqa: PLC0415  Defer torch import to allow CPU-only environments
"""Custom Trainer applying graph-smoothed soft targets at trie branch points.

Companion to the `target_smoothing_epsilon` option of
`glen_bullet_dataset.GlenBulletDataset`. Standard shifted cross-entropy on
all label tokens, except at positions that carry soft-target entries
(produced by the dataset and located by `collate_fn`'s `soft_*` tensors):
there the one-hot label is replaced by `(1 - epsilon)` mass on the gold
token and `epsilon` distributed over sibling branch tokens leading to other
gold papers of the same query, weighted by graph edge strength.

Unlike marginal-likelihood losses (which permit winner-take-all collapse
onto the easiest positive), the soft label is a fixed distribution, so the
model cannot zero out a sibling branch that carries label mass.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

from transformers import Trainer

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

logger = logging.getLogger(__name__)


def smoothed_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    soft_batch: Tensor,
    soft_pos: Tensor,
    soft_token: Tensor,
    soft_weight: Tensor,
    epsilon: float,
    num_items: Tensor | int | None = None,
) -> Tensor:
    """Shifted CE with soft labels at smoothed positions.

    Args:
        logits: `(B, T, V)` model output logits.
        labels: `(B, T)` token labels with -100 masking non-target positions.
        soft_batch: `(N,)` batch index of each soft-target entry.
        soft_pos: `(N,)` absolute label position of each entry in the padded
            sequence (the position of the smoothed target token itself).
        soft_token: `(N,)` alternative token ID receiving smoothing mass.
        soft_weight: `(N,)` weight of each entry; weights sum to 1 per
            (batch, position) by construction in the dataset.
        epsilon: Total smoothing mass per smoothed position.
        num_items: Denominator for the mean (the `num_items_in_batch` from
            HF Trainer under gradient accumulation). Falls back to the
            count of non-masked label tokens, matching HF's CE behaviour.

    Returns:
        Scalar loss tensor.
    """
    import torch

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logprobs = torch.log_softmax(shift_logits.float(), dim=-1)
    mask = shift_labels != -100
    safe_labels = shift_labels.clamp(min=0)
    gold_nll = -logprobs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    gold_nll = gold_nll * mask

    scale = torch.ones_like(gold_nll)
    extra = gold_nll.new_zeros(())
    if soft_batch.numel() > 0:
        # The label at sequence position p is predicted by logits at p - 1,
        # i.e. index p - 1 in the shifted tensors.
        pos = soft_pos - 1
        valid = mask[soft_batch, pos]
        sb = soft_batch[valid]
        sp = pos[valid]
        st = soft_token[valid]
        sw = soft_weight[valid]
        if sb.numel() > 0:
            # Idempotent under duplicate (batch, pos) pairs: per-position
            # weights already sum to 1, so gold keeps exactly 1 - epsilon.
            scale[sb, sp] = 1.0 - epsilon
            extra = -epsilon * (sw.float() * logprobs[sb, sp, st]).sum()

    total = (gold_nll * scale).sum() + extra
    if num_items is None:
        denom = mask.sum().clamp(min=1)
    else:
        denom = num_items
    return total / denom


class SmoothedTargetTrainer(Trainer):
    """Trainer computing graph-smoothed soft-target CE.

    Expects batches from `collate_fn` over a `GlenBulletDataset` built with
    `target_smoothing_epsilon > 0`; batches without `soft_*` tensors fall
    back to plain shifted CE.
    """

    def __init__(self, *args: Any, smoothing_epsilon: float, **kwargs: Any) -> None:
        """Store `smoothing_epsilon` and forward everything else to Trainer."""
        super().__init__(*args, **kwargs)
        self._smoothing_epsilon = smoothing_epsilon

    @override
    def compute_loss(
        self,
        model: Module,
        inputs: dict[str, Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, Any]:
        """Compute the smoothed CE over one batch."""
        import torch

        labels: Tensor = inputs.pop("labels")
        empty = torch.zeros(0, dtype=torch.long, device=labels.device)
        soft_batch: Tensor = inputs.pop("soft_batch", empty)
        soft_pos: Tensor = inputs.pop("soft_pos", empty)
        soft_token: Tensor = inputs.pop("soft_token", empty)
        soft_weight: Tensor = inputs.pop("soft_weight", empty.float())

        outputs = model(**inputs)
        loss = smoothed_cross_entropy(
            outputs.logits,
            labels,
            soft_batch=soft_batch,
            soft_pos=soft_pos,
            soft_token=soft_token,
            soft_weight=soft_weight,
            epsilon=self._smoothing_epsilon,
            num_items=num_items_in_batch,
        )
        if return_outputs:
            return loss, outputs
        return loss


__all__ = ["SmoothedTargetTrainer", "smoothed_cross_entropy"]
