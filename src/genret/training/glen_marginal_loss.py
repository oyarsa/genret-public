# pyright: basic
# ruff: noqa: PLC0415  Defer torch import to allow CPU-only environments
"""Custom Trainer with per-group marginal-likelihood loss.

Companion to :mod:`glen_marginal_dataset`. Each batch contains B logical
groups expanded into B*K rows; this Trainer computes the per-row
autoregressive log-likelihood of the target tokens, then for each group
returns ``-logsumexp(per_row_log_probs)``, then averages over groups.

This is the negative log marginal likelihood
``-log Σ_{d ∈ D(b)} P(d | q)`` — equal credit for any DocID in the gold
paper's bullet set, no per-row "popularity amplification."
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


def _marginal_per_group(
    student_log_probs: Tensor,
    group_starts: Tensor,
    group_sizes: Tensor,
) -> Tensor:
    """Compute mean of ``-logsumexp`` per group.

    Args:
        student_log_probs: ``(B*K,)`` per-row autoregressive
            log-probabilities of each target sequence.
        group_starts: ``(B,)`` start index of each group.
        group_sizes: ``(B,)`` number of rows per group.

    Returns:
        Scalar tensor: mean ``-logsumexp`` over groups.
    """
    import torch

    losses: list[Tensor] = []
    for start_t, size_t in zip(group_starts.tolist(), group_sizes.tolist()):
        s = int(start_t)
        n = int(size_t)
        if n <= 0:
            continue
        group_log_probs = student_log_probs[s : s + n]
        # log Σ_i exp(log_p_i) = logsumexp; negative for the loss.
        losses.append(-torch.logsumexp(group_log_probs, dim=0))

    if not losses:
        return student_log_probs.new_zeros(())
    return torch.stack(losses).mean()


class MarginalLikelihoodTrainer(Trainer):
    """Trainer that computes ``-log marginal_likelihood`` per group.

    Expects each batch to include ``group_starts`` and ``group_sizes``
    produced by :func:`glen_marginal_dataset.marginal_collator`.
    """

    @override
    def compute_loss(
        self,
        model: Module,
        inputs: dict[str, Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, Any]:
        """Compute the mean negative log marginal likelihood across groups."""
        group_starts: Tensor = inputs.pop("group_starts")
        group_sizes: Tensor = inputs.pop("group_sizes")
        labels: Tensor = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        student_log_probs = _per_row_log_probs(logits, labels)

        loss = _marginal_per_group(student_log_probs, group_starts, group_sizes)

        if return_outputs:
            return loss, outputs
        return loss


__all__ = ["MarginalLikelihoodTrainer"]
