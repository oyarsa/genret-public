"""Typed wrappers for PyTorch operations with imprecise stubs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def model_to_device(model: PreTrainedModel, device: str) -> PreTrainedModel:
    """Move a HuggingFace model to a device, returning the same type.

    Args:
        model: Pre-trained model.
        device: Target device string (e.g. ``"cuda"``, ``"cpu"``).

    Returns:
        The model on the target device.
    """
    return model.to(device)  # type: ignore


def tensor_column(tensor: torch.Tensor, col: int) -> torch.Tensor:
    """Select a column from a 2-D tensor.

    Args:
        tensor: 2-D tensor of shape ``(N, C)``.
        col: Column index (supports negative indexing).

    Returns:
        1-D tensor of shape ``(N,)``.
    """
    return tensor[:, col]


def unsqueeze_mask(mask: Any) -> torch.Tensor:
    """Unsqueeze a tokeniser attention mask for broadcasting.

    ``BatchEncoding["attention_mask"]`` has an imprecise type in
    transformers stubs. This wrapper narrows it.

    Args:
        mask: The ``attention_mask`` value from a ``BatchEncoding``.

    Returns:
        Tensor with an extra trailing dimension.
    """
    t: torch.Tensor = mask
    return t.unsqueeze(-1)


def generate(model: PreTrainedModel, **kwargs: object) -> Any:
    """Call ``model.generate()`` with the given keyword arguments.

    ``PreTrainedModel.generate`` is typed as not-callable in some stub
    versions; this wrapper hides that.

    Args:
        model: HuggingFace model with a ``generate`` method.
        **kwargs: Arguments forwarded to ``model.generate()``.

    Returns:
        The raw generate output (``GenerateOutput``).
    """
    return model.generate(**kwargs)  # type: ignore


def beam_transition_scores(
    model: PreTrainedModel, generate_output: Any
) -> torch.Tensor | None:
    """Per-token log-probabilities of each returned beam, or ``None``.

    Wraps ``PreTrainedModel.compute_transition_scores``, which is untyped and
    needs ``beam_indices`` that only beam-search outputs carry. Beam-search
    ``scores`` are already log-softmax normalised, so ``normalize_logits`` is
    left off and the row sums reproduce the cumulative score the beam scorer
    accumulated (before its length division).

    Args:
        model: The model that produced ``generate_output``.
        generate_output: Output of ``generate()`` with
            ``return_dict_in_generate=True`` and ``output_scores=True``.

    Returns:
        Tensor of shape ``(num_return_sequences, generated_length)`` with
        ``-inf`` in padding positions, or ``None`` when the output lacks the
        fields needed (non-beam decoding, or a transformers version that does
        not expose them).
    """
    scores = getattr(generate_output, "scores", None)  # type: ignore[misc]
    beam_indices = getattr(generate_output, "beam_indices", None)  # type: ignore[misc]
    if scores is None or beam_indices is None:
        return None
    try:
        result: torch.Tensor = model.compute_transition_scores(  # type: ignore
            sequences=generate_output.sequences,
            scores=scores,
            beam_indices=beam_indices,
            normalize_logits=False,
        )
    except (ValueError, RuntimeError, AttributeError, TypeError):
        return None
    return result
