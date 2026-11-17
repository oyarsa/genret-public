"""Unit tests for the rate_limit module."""

import asyncio
import contextlib

from genret.llm.rate_limit import TokenRateLimiter, estimate_tokens


class TestEstimateTokens:
    """Tests for the estimate_tokens function."""

    def test_returns_positive_for_nonempty_text(self) -> None:
        """Should return a positive token count for non-empty text."""
        assert estimate_tokens("Hello, world!") > 0

    def test_returns_zero_for_empty_string(self) -> None:
        """Should return zero tokens for an empty string."""
        assert estimate_tokens("") == 0

    def test_longer_text_has_more_tokens(self) -> None:
        """Longer text should produce more tokens than shorter text."""
        short = estimate_tokens("Hello")
        long = estimate_tokens(
            "Hello world, this is a longer sentence with more tokens"
        )
        assert long > short

    def test_reasonable_ratio(self) -> None:
        """Token count should be in a reasonable range relative to word count."""
        text = "The quick brown fox jumps over the lazy dog"
        tokens = estimate_tokens(text)
        words = len(text.split())
        # Roughly 1-2 tokens per word for English text
        assert words * 0.5 <= tokens <= words * 3


class TestTokenRateLimiter:
    """Tests for the TokenRateLimiter class."""

    async def test_allows_requests_within_limits(self) -> None:
        """Should allow requests that fit within the rate limit."""
        limiter = TokenRateLimiter(request_limit=10, token_limit=100_000)
        async with limiter.limit("hello world") as update:
            await update(5)

    async def test_tracks_request_count(self) -> None:
        """Should track multiple requests in the sliding window."""
        limiter = TokenRateLimiter(request_limit=5, token_limit=1_000_000)
        for _ in range(5):
            async with limiter.limit("test") as update:
                await update(2)

    async def test_blocks_when_at_capacity(self) -> None:
        """Should block when request limit is reached."""
        limiter = TokenRateLimiter(
            request_limit=1, token_limit=1_000_000, bucket_size_in_seconds=60
        )
        # Fill the single slot
        async with limiter.limit("first") as update:
            await update(2)

            # Try to acquire a second slot — should not complete immediately
            acquired = asyncio.Event()

            async def _try_acquire() -> None:
                async with limiter.limit("second"):
                    acquired.set()

            task = asyncio.create_task(_try_acquire())
            await asyncio.sleep(0.15)
            blocked = not acquired.is_set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            assert blocked, "Second request should have been blocked"

    async def test_releases_on_exception(self) -> None:
        """Should release the slot when an exception occurs."""

        async def _fail_in_limit(rl: TokenRateLimiter) -> None:
            """Acquire a slot and raise inside it."""
            async with rl.limit("test"):
                msg = "boom"
                raise ValueError(msg)

        limiter = TokenRateLimiter(request_limit=1, token_limit=1_000_000)
        with contextlib.suppress(ValueError):
            await _fail_in_limit(limiter)

        # Should be able to acquire again
        async with limiter.limit("test again") as update:
            await update(2)
