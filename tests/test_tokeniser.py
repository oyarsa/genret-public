"""Tests for the tokeniser wrapper module."""

import pytest
from tokenizers import Tokenizer

from genret._shed.tokenizers import (
    encode_no_special,
    token_to_id,
    tokenizer_from_pretrained,
)
from genret.tokeniser import load_tokeniser, tokenise_batch


@pytest.fixture
def tokeniser() -> Tokenizer:
    """Load a small tokeniser for testing."""
    return tokenizer_from_pretrained("bert-base-uncased")


class TestTokeniseBatch:
    """Tests for the tokenise_batch function."""

    def test_batch_tokenisation(self, tokeniser: Tokenizer) -> None:
        """Should tokenise multiple strings and return token lists."""
        docids = ["hello world", "foo bar baz"]
        result = tokenise_batch(docids, tokeniser)

        assert len(result) == 2
        assert all(isinstance(seq, list) for seq in result)
        assert all(isinstance(tok, int) for seq in result for tok in seq)

    def test_empty_input(self, tokeniser: Tokenizer) -> None:
        """Should return empty list for empty input."""
        assert tokenise_batch([], tokeniser) == []

    def test_result_length_matches_input(self, tokeniser: Tokenizer) -> None:
        """Should return one token sequence per input string."""
        docids = ["a", "b", "c"]
        result = tokenise_batch(docids, tokeniser)
        assert len(result) == len(docids)

    def test_no_bos_cls_tokens(self, tokeniser: Tokenizer) -> None:
        """Should not include BOS/CLS wrapper tokens in output.

        BERT's CLS token ID is 101 and SEP is 102. With `add_special_tokens=False`,
        neither should appear.
        """
        result = tokenise_batch(["hello"], tokeniser)
        assert len(result) == 1
        tokens = result[0]
        cls_id = 101
        sep_id = 102
        assert tokens[0] != cls_id, "CLS token should not be prepended"
        assert tokens[-1] != sep_id, "SEP token should not be appended"


class TestLoadTokeniser:
    """Tests for the load_tokeniser function."""

    def test_special_tokens_get_unique_ids(self) -> None:
        """Registering special tokens should assign unique IDs beyond the base vocab."""
        special = ("<PAPER>", "<END_PAPER>", "<ASPECT=problems>")
        tok = load_tokeniser("bert-base-uncased", special_tokens=special)

        ids = [token_to_id(tok, t) for t in special]
        # All should be valid (not None)
        assert all(i is not None for i in ids), f"Some special tokens not found: {ids}"
        # All should be distinct
        assert len(set(ids)) == len(ids), f"Duplicate IDs: {ids}"

    def test_special_tokens_roundtrip(self) -> None:
        """Special tokens should encode as single tokens, not subword pieces."""
        special = ("<PAPER>", "<END_PAPER>")
        tok = load_tokeniser("bert-base-uncased", special_tokens=special)

        for token_str in special:
            ids = encode_no_special(tok, token_str)
            assert len(ids) == 1, (
                f"{token_str!r} should be a single token but got {len(ids)} pieces"
            )

    def test_no_special_tokens_default(self) -> None:
        """Loading without special tokens should work (backward-compatible)."""
        tok = load_tokeniser("bert-base-uncased")
        assert tok is not None
