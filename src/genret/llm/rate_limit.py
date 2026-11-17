"""Token-aware rate limiting and shared tokenisation utilities.

Provides a sliding-window rate limiter for OpenAI API calls and a shared
tiktoken tokenizer for token estimation.  Extracted from openai_encoder so
that both the embedding encoder and the LLM chat client can share these.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass

import openai

from genret.llm.utils import tokenise

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class _RequestRecord:
    """A single in-flight or completed request tracked by the rate limiter."""

    timestamp: float
    """Monotonic time when the request was admitted."""
    tokens: int
    """Estimated (or actual, once updated) token count."""
    completed: bool
    """Whether the actual token usage callback has been called."""


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text*.

    Uses tiktoken when available, falling back to a word-count heuristic
    if encoding fails for any reason.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    try:
        return len(tokenise(text))
    except Exception:
        return int(len(text.split()) * 1.5)


RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)
"""OpenAI exception types that trigger exponential backoff retries."""


class TokenRateLimiter:
    """Token-aware rate limiter for the OpenAI API using a sliding window."""

    def __init__(
        self, request_limit: int, token_limit: int, bucket_size_in_seconds: int = 60
    ) -> None:
        """Initialise the rate limiter.

        Args:
            request_limit: Maximum requests per sliding window.
            token_limit: Maximum tokens per sliding window.
            bucket_size_in_seconds: Sliding window duration in seconds.
        """
        self._request_limit = request_limit
        self._token_limit = token_limit
        self._bucket_size = bucket_size_in_seconds
        self._requests: dict[int, _RequestRecord] = {}
        self._lock = asyncio.Lock()

    async def _get_current_usage(self) -> tuple[int, int]:
        """Get current request and token usage in the sliding window."""
        current_time = time.monotonic()
        cutoff_time = current_time - self._bucket_size

        expired_ids = [
            req_id
            for req_id, record in self._requests.items()
            if record.timestamp < cutoff_time
        ]
        for req_id in expired_ids:
            del self._requests[req_id]

        current_requests = len(self._requests)
        current_tokens = sum(record.tokens for record in self._requests.values())
        return current_requests, current_tokens

    @asynccontextmanager
    async def limit(
        self, text: str
    ) -> AsyncGenerator[Callable[[int], Coroutine[None, None, None]]]:
        """Wait for capacity, then allow the API call.

        Yields a callback to update the rate limiter with actual token usage.

        Args:
            text: The text whose token count determines capacity reservation.

        Yields:
            An async callback accepting actual token usage to refine the estimate.
        """
        estimated_tokens = estimate_tokens(text)
        request_id = uuid.uuid4().int
        waited = False
        wait_start = time.monotonic()

        while True:
            async with self._lock:
                current_requests, current_tokens = await self._get_current_usage()
                if (
                    current_requests < self._request_limit
                    and current_tokens + estimated_tokens <= self._token_limit
                ):
                    if waited:
                        elapsed = time.monotonic() - wait_start
                        logger.debug(
                            "Rate limit wait %.1fs (req %d/%d, tok %d/%d, need %d)",
                            elapsed,
                            current_requests,
                            self._request_limit,
                            current_tokens,
                            self._token_limit,
                            estimated_tokens,
                        )
                    self._requests[request_id] = _RequestRecord(
                        timestamp=time.monotonic(),
                        tokens=estimated_tokens,
                        completed=False,
                    )
                    break

                if not waited:
                    waited = True
                    logger.debug(
                        "Rate limit hit, waiting (req %d/%d, tok %d/%d, need %d)",
                        current_requests,
                        self._request_limit,
                        current_tokens,
                        self._token_limit,
                        estimated_tokens,
                    )

            await asyncio.sleep(0.05)

        try:

            async def update_with_actual_usage(actual_tokens: int) -> None:
                async with self._lock:
                    if request_id in self._requests:
                        self._requests[request_id] = _RequestRecord(
                            timestamp=self._requests[request_id].timestamp,
                            tokens=actual_tokens,
                            completed=True,
                        )

            yield update_with_actual_usage
        except BaseException:
            async with self._lock:
                self._requests.pop(request_id, None)
            raise
