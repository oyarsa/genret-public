# pyright: basic
# ruff: noqa: PLC0415  Defer torch import to allow CPU-only environments
"""Custom Trainer with per-group InfoNCE contrastive loss.

Companion to :mod:`glen_contrastive_dataset`. For each logical example
the loss is:

    L = -log( Σ_{d in P} P(d | q) / Σ_{d in P or N} P(d | q) )

which expands as ``-logsumexp(pos_log_probs) + logsumexp(all_log_probs)``.

The numerator is the marginal likelihood over the gold paper's bullet set;
the denominator adds the popularity-sampled non-gold bullets. The model
is pushed toward the gold bullets *and* away from the negatives, breaking
the popularity-collapse attractor that plain marginal-LL leaves intact.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

from transformers import Trainer

from genret.training.glen_distill_loss import _per_row_log_probs

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

logger = logging.getLogger(__name__)


def _infonce_per_group(
    student_log_probs: Tensor,
    pos_mask: Tensor,
    group_starts: Tensor,
    group_sizes: Tensor,
) -> Tensor:
    """Per-group ``-logsumexp(pos) + logsumexp(all)``, averaged across groups.

    Args:
        student_log_probs: ``(B*K,)`` per-row autoregressive log P(d|q).
        pos_mask: ``(B*K,)`` 1 for positive rows, 0 for negative rows.
        group_starts: ``(B,)`` start index of each logical group.
        group_sizes: ``(B,)`` row count for each logical group.

    Returns:
        Scalar tensor: mean per-group InfoNCE loss.
    """
    import torch

    losses: list[Tensor] = []
    for start_t, size_t in zip(group_starts.tolist(), group_sizes.tolist()):
        s = int(start_t)
        n = int(size_t)
        if n <= 0:
            continue
        group_log_probs = student_log_probs[s : s + n]
        group_pos_mask = pos_mask[s : s + n]

        pos_rows = group_log_probs[group_pos_mask == 1]
        if pos_rows.numel() == 0:
            continue

        # Numerator: logsumexp over positive rows.
        # Denominator: logsumexp over ALL rows (positives plus negatives).
        loss = -torch.logsumexp(pos_rows, dim=0) + torch.logsumexp(
            group_log_probs, dim=0
        )
        losses.append(loss)

    if not losses:
        return student_log_probs.new_zeros(())
    return torch.stack(losses).mean()


class ContrastiveTrainer(Trainer):
    """Trainer that computes per-group InfoNCE between positives and negatives.

    Expects each batch to include ``group_starts``, ``group_sizes``, and
    ``pos_mask`` produced by
    :func:`glen_contrastive_dataset.contrastive_collator`.
    """

    @override
    def compute_loss(
        self,
        model: Module,
        inputs: dict[str, Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, Any]:
        """Compute per-group InfoNCE averaged across groups."""
        group_starts: Tensor = inputs.pop("group_starts")
        group_sizes: Tensor = inputs.pop("group_sizes")
        pos_mask: Tensor = inputs.pop("pos_mask")
        labels: Tensor = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits
        student_log_probs = _per_row_log_probs(logits, labels)

        loss = _infonce_per_group(
            student_log_probs, pos_mask, group_starts, group_sizes
        )

        if return_outputs:
            return loss, outputs
        return loss


__all__ = ["ContrastiveTrainer"]
