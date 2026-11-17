# pyright: basic
"""Typed wrapper around the HuggingFace `tokenizers` library.

The `tokenizers` library lacks proper type stubs, so this module isolates all direct
usage behind typed function signatures.  The rest of the codebase imports from here and
stays under strict type checking.
"""

import logging
from collections.abc import Sequence

from tokenizers import AddedToken, Tokenizer

from genret.utils import die

logger = logging.getLogger(__name__)


def load_tokeniser(name: str, *, special_tokens: Sequence[str] = ()) -> Tokenizer:
    """Load a HuggingFace tokeniser by name.

    Args:
        name: HuggingFace model/tokeniser name (e.g. `"meta-llama/Llama-3.1-8B"`).
        special_tokens: Additional special tokens to register in the vocabulary
            (e.g. structural DocID delimiters).  Each token gets a unique ID.

    Returns:
        The loaded Tokenizer instance.
    """
    try:
        tok = Tokenizer.from_pretrained(name)
    except Exception:
        die(f"Failed to load tokeniser {name!r}.")

    if special_tokens:
        tok.add_special_tokens([AddedToken(t, special=True) for t in special_tokens])

    return tok


def tokenise_batch(strings: Sequence[str], tokeniser: Tokenizer) -> list[list[int]]:
    """Tokenise multiple strings in parallel.

    Uses `encode_batch` which parallelises across system threads.

    Args:
        strings: Sequence of strings to tokenise.
        tokeniser: HuggingFace tokeniser instance.

    Returns:
        List of token ID lists, one per input string.
    """
    if not strings:
        return []

    return [
        enc.ids
        for enc in tokeniser.encode_batch_fast(list(strings), add_special_tokens=False)
    ]
