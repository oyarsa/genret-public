"""Typed wrappers for the ``tokenizers`` library."""

# pyright: basic

from tokenizers import Tokenizer


def tokenizer_from_pretrained(name: str) -> Tokenizer:
    """Load a pre-trained tokenizer by name.

    Args:
        name: HuggingFace model/tokenizer name.

    Returns:
        Loaded ``Tokenizer`` instance.
    """
    return Tokenizer.from_pretrained(name)


def token_to_id(tokenizer: Tokenizer, token: str) -> int | None:
    """Get the ID for a single token string.

    Args:
        tokenizer: Tokenizer instance.
        token: Token string.

    Returns:
        Integer ID, or ``None`` if the token is unknown.
    """
    return tokenizer.token_to_id(token)


def encode_no_special(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode text without special tokens and return token IDs.

    Args:
        tokenizer: Tokenizer instance.
        text: Text to encode.

    Returns:
        List of token IDs.
    """
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return list(encoded.ids)
