"""Unit tests for the genret.llm.utils module."""

from genret.llm.rate_limit import estimate_tokens
from genret.llm.utils import truncate_to_tokens


class TestTruncateToTokens:
    """Tests for the truncate_to_tokens function."""

    def test_short_text_unchanged(self) -> None:
        """Should return the same text when under the limit."""
        text = "Hello world"
        assert truncate_to_tokens(text, 1000) == text

    def test_truncates_long_text(self) -> None:
        """Should truncate text exceeding the limit."""
        text = "word " * 500
        result = truncate_to_tokens(text, 10)
        assert len(result) < len(text)
        assert estimate_tokens(result) <= 10

    def test_empty_string(self) -> None:
        """Should handle empty string."""
        assert truncate_to_tokens("", 10) == ""
