# pyright: basic
# ruff: noqa: PLC0415  Defer torch import to allow CPU-only environments
"""Custom Trainer with per-group KL distillation loss.

Companion to :mod:`glen_distill_dataset`. Each batch contains B
logical groups expanded into B*K rows; this Trainer computes the
per-row autoregressive log-likelihood of the target tokens, then
softmax-normalises within each group to form a student distribution
over K targets, and KL-divides against the teacher distribution
(softmax of per-row teacher weights with temperature).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

from transformers import Trainer

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

logger = logging.getLogger(__name__)


def _per_row_log_probs(logits: Tensor, labels: Tensor) -> Tensor:
    """Sum log-softmax of correct token at each target position per row.

    Args:
        logits: ``(B*K, T, V)`` — model logits.
        labels: ``(B*K, T)`` — -100 for masked positions, target token
            ids elsewhere.

    Returns:
        ``(B*K,)`` — sum of token-level log-probabilities along the
        target portion of each row.
    """
    import torch

    # Standard autoregressive shift: logits at position t predict
    # the token at position t+1.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    log_softmax = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    # Replace -100 with 0 in labels for safe gather; mask zero them out below.
    mask = (shift_labels != -100).float()
    safe_labels = shift_labels.clamp(min=0)
    gathered = log_softmax.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    masked = gathered * mask
    return masked.sum(dim=-1)


def _kl_per_group(
    student_log_probs: Tensor,
    teacher_weights: Tensor,
    group_starts: Tensor,
    group_sizes: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Compute mean per-group KL(teacher || student).

    Args:
        student_log_probs: ``(B*K,)`` — per-row autoregressive
            log-probabilities (unnormalised across groups).
        teacher_weights: ``(B*K,)`` — per-row teacher weights.
        group_starts: ``(B,)`` — start index of each group in the
            flat B*K dimension.
        group_sizes: ``(B,)`` — number of rows per group.
        temperature: softmax temperature applied to teacher weights.

    Returns:
        Scalar tensor: mean KL across groups.
    """
    import torch

    losses: list[Tensor] = []
    for start_t, size_t in zip(group_starts.tolist(), group_sizes.tolist()):
        s = int(start_t)
        n = int(size_t)
        if n <= 0:
            continue

        student_logits = student_log_probs[s : s + n]
        teacher_logits = teacher_weights[s : s + n] / temperature

        # Length-normalise student log-probs by the number of target
        # tokens in each row to avoid bias toward shorter targets.
        # The cleaner approach is to compute mean-log-prob per row;
        # for simplicity (and because target lengths are bounded by
        # the trie), use sum log-prob — this biases slightly toward
        # longer targets but matches standard GR practice.

        student_log_dist = torch.log_softmax(student_logits, dim=0)
        teacher_dist = torch.softmax(teacher_logits, dim=0)

        # KL(teacher || student) = sum t * (log t - log s)
        teacher_log_dist = torch.log_softmax(teacher_logits, dim=0)
        kl = (teacher_dist * (teacher_log_dist - student_log_dist)).sum()
        losses.append(kl)

    if not losses:
        return student_log_probs.new_zeros(())

    return torch.stack(losses).mean()


class DistillTrainer(Trainer):
    """Trainer that computes a per-group KL loss instead of token CE.

    Expects each batch to include ``group_starts``, ``group_sizes``,
    and ``teacher_weights_flat`` produced by
    :func:`glen_distill_dataset.distill_collator`.

    The ``temperature`` attribute scales the teacher distribution.
    Set after construction (or read from training args).
    """

    temperature: float = 1.0

    @override
    def compute_loss(
        self,
        model: Module,
        inputs: dict[str, Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, Any]:
        """Compute KL(teacher || student) averaged across groups."""
        group_starts: Tensor = inputs.pop("group_starts")
        group_sizes: Tensor = inputs.pop("group_sizes")
        teacher_weights: Tensor = inputs.pop("teacher_weights_flat")
        labels: Tensor = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        student_log_probs = _per_row_log_probs(logits, labels)

        loss = _kl_per_group(
            student_log_probs,
            teacher_weights,
            group_starts,
            group_sizes,
            temperature=self.temperature,
        )

        if return_outputs:
            return loss, outputs
        return loss


__all__ = ["DistillTrainer"]
