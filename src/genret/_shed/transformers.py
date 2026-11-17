"""Typed wrappers for HuggingFace transformers APIs with imprecise stubs."""

# pyright: basic

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from genret.types import is_type

if TYPE_CHECKING:
    from transformers import EvalPrediction


# ---------------------------------------------------------------------------
# EvalPrediction
# ---------------------------------------------------------------------------


def predictions_array(pred: EvalPrediction) -> npt.NDArray[np.float64]:
    """Extract predictions from an ``EvalPrediction`` as a 1-D float array.

    Squeezes a trailing dimension if present (e.g. ``(N, 1) → (N,)``).

    Args:
        pred: HuggingFace evaluation prediction.

    Returns:
        1-D numpy array of prediction values.
    """
    raw = pred.predictions  # type: ignore[union-attr]
    arr = np.asarray(raw, dtype=np.float64).squeeze(-1)
    return arr


# ---------------------------------------------------------------------------
# Tokeniser loading
# ---------------------------------------------------------------------------


def load_tokenizer(name: str, **kwargs: object) -> PreTrainedTokenizerBase:
    """Load a pre-trained tokeniser, returning the proper base type.

    ``AutoTokenizer.from_pretrained`` returns an opaque union that checkers
    can't narrow. This wrapper casts it.

    Args:
        name: HuggingFace model/tokeniser name or local path.
        **kwargs: Extra keyword arguments forwarded to ``from_pretrained``.

    Returns:
        Loaded tokeniser.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokeniser = AutoTokenizer.from_pretrained(name, **kwargs)
    return cast(PreTrainedTokenizerBase, tokeniser)


def load_auto_model(name: str, **kwargs: object) -> PreTrainedModel:
    """Load a base ``AutoModel``, returning the proper base type.

    ``AutoModel.from_pretrained`` returns an opaque union that checkers
    can't narrow. This wrapper casts it.

    Args:
        name: HuggingFace model name or local path.
        **kwargs: Extra keyword arguments forwarded to ``from_pretrained``.

    Returns:
        Loaded model.
    """
    from transformers import AutoModel  # noqa: PLC0415

    model = AutoModel.from_pretrained(name, **kwargs)
    return cast(PreTrainedModel, model)


# ---------------------------------------------------------------------------
# Token ID accessors
# ---------------------------------------------------------------------------


def decode_tokens(
    tokeniser: PreTrainedTokenizerBase,
    token_ids: list[int],
    *,
    skip_special_tokens: bool = True,
) -> str:
    """Decode token IDs to a string.

    transformers 5 stubs type ``decode`` with a union return; this wrapper
    pins the single-sequence ``str`` case.
    """
    out = tokeniser.decode(token_ids, skip_special_tokens=skip_special_tokens)
    return cast(str, out)


def get_eos_token_id(tokeniser: PreTrainedTokenizerBase) -> int:
    """Return the EOS token ID, asserting it exists.

    Args:
        tokeniser: HuggingFace tokeniser.

    Returns:
        Integer EOS token ID.
    """
    tid = tokeniser.eos_token_id
    assert isinstance(tid, int)
    return tid


def get_pad_token_id(tokeniser: PreTrainedTokenizerBase) -> int | None:
    """Return the pad token ID (may be ``None``).

    Args:
        tokeniser: HuggingFace tokeniser.

    Returns:
        Integer pad token ID, or ``None`` if unset.
    """
    tid = tokeniser.pad_token_id
    assert isinstance(tid, int) or tid is None
    return tid


def convert_token_to_id(tokeniser: PreTrainedTokenizerBase, token: str) -> int:
    """Resolve a single special token string to its vocab id.

    Wraps ``tokeniser.convert_tokens_to_ids`` so call sites can treat
    the return value as a plain ``int`` instead of dealing with the
    stub's ``str | list[str]`` overload union.
    """
    # pyrefly: ignore  # not-callable — HF stub union with bound method
    tid = tokeniser.convert_tokens_to_ids(token)
    return int(cast(int, tid))


# ---------------------------------------------------------------------------
# apply_chat_template variants
# ---------------------------------------------------------------------------


def apply_chat_template_ids(
    tokeniser: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Apply chat template and return token IDs.

    Calls ``apply_chat_template`` with ``tokenize=True, return_dict=False``.

    Args:
        tokeniser: HuggingFace tokeniser with chat template support.
        messages: Chat messages.
        add_generation_prompt: Whether to append the generation prompt.

    Returns:
        List of token IDs.
    """
    result = tokeniser.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=False,
    )
    assert is_type(result, list[int])
    return cast(list[int], result)


def apply_chat_template_str(
    tokeniser: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """Apply chat template and return the rendered string.

    Calls ``apply_chat_template`` with ``tokenize=False``.

    Args:
        tokeniser: HuggingFace tokeniser with chat template support.
        messages: Chat messages.
        add_generation_prompt: Whether to append the generation prompt.
        enable_thinking: Whether to enable thinking tokens.

    Returns:
        Rendered prompt string.
    """
    result = tokeniser.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    assert isinstance(result, str)
    return result


def apply_chat_template_tensor(
    tokeniser: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> torch.Tensor:
    """Apply chat template and return a tensor of token IDs.

    Calls ``apply_chat_template`` with ``tokenize=True, return_tensors="pt"``.

    Args:
        tokeniser: HuggingFace tokeniser with chat template support.
        messages: Chat messages.
        add_generation_prompt: Whether to append the generation prompt.

    Returns:
        Token IDs as a ``torch.Tensor``.
    """
    result = tokeniser.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=False,
        return_tensors="pt",
    )
    return cast(torch.Tensor, result)


# ---------------------------------------------------------------------------
# PEFT model loading
# ---------------------------------------------------------------------------


def load_peft_model(base_model: PreTrainedModel, checkpoint: str) -> PreTrainedModel:
    """Load a PEFT adapter on top of a base model.

    ``PeftModel.from_pretrained`` returns an opaque type. This wrapper
    casts it to ``PreTrainedModel``.

    Args:
        base_model: Base HuggingFace model.
        checkpoint: Path to the saved LoRA adapter directory.

    Returns:
        Model with the adapter loaded.
    """
    from peft import PeftModel  # noqa: PLC0415

    model = PeftModel.from_pretrained(base_model, checkpoint)
    return cast(PreTrainedModel, model)


def as_pretrained_model(model: object) -> PreTrainedModel:
    """Cast a PEFT or other model wrapper to ``PreTrainedModel``.

    ``PeftModel`` and similar wrappers behave as ``PreTrainedModel`` at
    runtime but aren't typed as such in stubs. This wrapper hides that.

    Args:
        model: A model instance (e.g. ``PeftModel``).

    Returns:
        The same model, typed as ``PreTrainedModel``.
    """
    return cast(PreTrainedModel, model)
