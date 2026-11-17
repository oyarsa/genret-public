"""Shared utilities for using LLM clients."""

from collections.abc import Sequence

import tiktoken

TOKENISER = tiktoken.get_encoding("o200k_base")
"""Shared tiktoken tokeniser for token estimation."""


def tokenise(text: str, disallowed_special: Sequence[str] = ()) -> list[int]:
    """Tokenise `text` with the `disallowed_special` special tokens (defaults to none)."""
    return TOKENISER.encode(text, disallowed_special=disallowed_special)


def detokenise(token_ids: list[int]) -> str:
    """Detokenise `token_ids` back to text."""
    return TOKENISER.decode(token_ids)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* tokens.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens to keep.

    Returns:
        The (possibly truncated) text.
    """
    try:
        tokens = tokenise(text)
        if len(tokens) <= max_tokens:
            return text
        return detokenise(tokens[:max_tokens])
    except Exception:
        words = text.split()
        # Rough heuristic: ~1.5 tokens per word.
        max_words = int(max_tokens / 1.5)
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])
