# pyright: basic
# ruff: noqa: PLC0415  Defer torch import to allow CPU-only environments
"""Custom Trainer applying per-row graph-weighted cross-entropy.

Companion to the ``graph_reweighting`` option of
``glen_bullet_dataset.GlenBulletDataset``. Standard shifted cross-entropy
on all label tokens, but every row's per-token NLL is multiplied by a
scalar weight derived from the ``scores_combined`` graph edge of the
(paper_a, paper_b, facet) pair the row was built from. The denominator
is the weighted sum of label tokens, so when every row has weight 1 the
loss is bit-identical to standard CE; non-uniform weights yield a
weighted-MLE estimator.

This is the continuous analogue of graph-weighted sample replication
(``graph_sampling="weighted"``): instead of emitting N copies of a
strong-edge pair so it contributes N forward passes, the row is emitted
once and its gradient is scaled by ``w`` in expectation. Two A/B uses:

- **Stacked**: combine with sample replication (vmhk recipe) — both the
  discrete inflation and the continuous reweighting apply.
- **Replacing**: turn sample replication off and reweight only — same
  per-pair signal as sampling in expectation, fewer training rows.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

from transformers import Trainer

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

logger = logging.getLogger(__name__)


def weighted_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    row_weight: Tensor,
    num_items: Tensor | int | None = None,
) -> Tensor:
    """Shifted CE with a per-row scalar weight.

    Args:
        logits: ``(B, T, V)`` model output logits.
        labels: ``(B, T)`` token labels with -100 masking non-target positions.
        row_weight: ``(B,)`` scalar weight per row; broadcast over label tokens.
        num_items: Optional denominator. When ``None``, the loss is normalised
            by the weighted token count so weight=1 throughout matches HF's
            standard CE exactly. When provided (e.g. HF's ``num_items_in_batch``
            under gradient accumulation), it is used directly — caller is
            responsible for the weighted normalisation.

    Returns:
        Scalar loss tensor.
    """
    import torch

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logprobs = torch.log_softmax(shift_logits.float(), dim=-1)
    mask = shift_labels != -100
    safe_labels = shift_labels.clamp(min=0)
    nll = -logprobs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    nll = nll * mask
    w = row_weight.to(nll.dtype).unsqueeze(-1)
    total = (nll * w).sum()
    if num_items is None:
        denom = (mask.to(w.dtype) * w).sum().clamp(min=1.0)
    else:
        denom = num_items
    return total / denom


def unlikelihood_per_row(
    logits: Tensor,
    labels: Tensor,
) -> Tensor:
    """Per-row sequence-level unlikelihood loss: -log(1 - P(target token)).

    Computed at every non-masked label position with the same shifted
    alignment as standard CE. Returned shape ``(B,)`` — one scalar per
    input row (mean over label tokens). Rows with all-masked labels
    return 0.

    Args:
        logits: ``(B, T, V)`` model output logits.
        labels: ``(B, T)`` token labels with -100 masking non-target positions.

    Returns:
        ``(B,)`` tensor of per-row unlikelihood means.
    """
    import torch

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logprobs = torch.log_softmax(shift_logits.float(), dim=-1)
    mask = (shift_labels != -100).to(logprobs.dtype)
    safe_labels = shift_labels.clamp(min=0)
    target_log_p = logprobs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    # log(1 - P) = log1p(-P). Clamp P below 1 - eps for stability when the
    # model is very confident about the negative target.
    target_p = target_log_p.exp().clamp(max=1.0 - 1e-6)
    ul_per_token = -torch.log1p(-target_p) * mask
    denom = mask.sum(dim=-1).clamp(min=1.0)
    return ul_per_token.sum(dim=-1) / denom


class WeightedCETrainer(Trainer):
    """Trainer computing per-row weighted CE, optionally with a hard-negative UL term.

    Expects batches from ``collate_fn`` over a ``GlenBulletDataset``. When
    ``loss_weight`` tensor is absent, every row contributes weight 1 (the
    loss is bit-identical to standard CE). When ``hard_neg_owner`` and
    ``n_gold_rows`` tensors are present, rows past ``n_gold_rows`` are
    interpreted as per-row hard negatives and contribute a sequence-level
    unlikelihood term, scaled by ``hard_negatives_lambda`` and the owner
    gold row's ``loss_weight``.
    """

    def __init__(
        self, *args: Any, hard_negatives_lambda: float = 0.0, **kwargs: Any
    ) -> None:
        """Store ``hard_negatives_lambda`` and forward everything else to Trainer."""
        super().__init__(*args, **kwargs)
        self._hard_negatives_lambda = float(hard_negatives_lambda)

    @override
    def compute_loss(
        self,
        model: Module,
        inputs: dict[str, Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, Any]:
        """Compute weighted CE plus optional hard-negative unlikelihood."""
        import torch

        labels: Tensor = inputs.pop("labels")
        row_weight = inputs.pop("loss_weight", None)
        hard_neg_owner = inputs.pop("hard_neg_owner", None)
        n_gold_rows_t = inputs.pop("n_gold_rows", None)
        if row_weight is None:
            row_weight = torch.ones(labels.shape[0], device=labels.device)

        outputs = model(**inputs)
        logits = outputs.logits

        if hard_neg_owner is not None and n_gold_rows_t is not None:
            n_gold = int(n_gold_rows_t.item())
            gold_logits = logits[:n_gold]
            gold_labels = labels[:n_gold]
            gold_weight = row_weight[:n_gold]
            ce = weighted_cross_entropy(
                gold_logits,
                gold_labels,
                row_weight=gold_weight,
                num_items=num_items_in_batch,
            )
            neg_logits = logits[n_gold:]
            neg_labels = labels[n_gold:]
            if neg_logits.shape[0] > 0 and self._hard_negatives_lambda > 0:
                ul_per_neg = unlikelihood_per_row(neg_logits, neg_labels)
                # Per-owner aggregation: average UL across negatives sharing
                # the same gold row, then scale by gold's loss_weight.
                owner = hard_neg_owner
                owner_weight = gold_weight[owner]
                weighted_ul = ul_per_neg * owner_weight
                # Mean over neg rows (each neg contributes equally; owners
                # with more negs naturally get more total mass).
                ul_loss = weighted_ul.mean()
                loss = ce + self._hard_negatives_lambda * ul_loss
            else:
                loss = ce
        else:
            loss = weighted_cross_entropy(
                logits,
                labels,
                row_weight=row_weight,
                num_items=num_items_in_batch,
            )

        if return_outputs:
            return loss, outputs
        return loss


__all__ = ["WeightedCETrainer", "unlikelihood_per_row", "weighted_cross_entropy"]
