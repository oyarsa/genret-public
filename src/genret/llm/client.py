"""Unified LLM client abstraction for structured LLM responses.

Provides a provider-agnostic interface for making LLM requests with built-in rate
limiting, retries, concurrency control, and typed structured-output parsing.
Supports OpenAI (Responses API), Azure OpenAI (Chat Completions API), and
dry-run testing.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Self, cast, override

import backoff
import openai
from pydantic import BaseModel

from genret.llm.models import (
    CHAT_RATE_LIMITS,
    DEFAULT_RATE_LIMIT,
    FIXED_TEMPERATURE_PREFIXES,
    compute_cost,
    resolve_model,
)
from genret.llm.rate_limit import (
    RETRYABLE_ERRORS,
    TokenRateLimiter,
    estimate_tokens,
)
from genret.llm.utils import truncate_to_tokens
from genret.types import is_type

logger = logging.getLogger(__name__)

type OpenAIReasoningEffort = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh"
]


class ReasoningEffort(StrEnum):
    """Reasoning-effort policy for OpenAI reasoning models."""

    DEFAULT = "default"
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"

    def to_openai(self, model: str) -> OpenAIReasoningEffort | None:
        """Convert to OpenAI's reasoning_effort value when supported.

        Returns `None` for `DEFAULT` or non-GPT-5 models so the parameter can be
        omitted. For GPT-5.1 onwards, `"minimal"` is normalised to `"none"`.

        Args:
            model: Model name (short alias or resolved name).

        Returns:
            OpenAI reasoning effort value, or None to omit the parameter.
        """
        if self is ReasoningEffort.DEFAULT or not model.startswith("gpt-5"):
            return None

        if (
            model.startswith(("gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.6"))
            and self is ReasoningEffort.MINIMAL
        ):
            return "none"

        assert is_type(self.value, OpenAIReasoningEffort)
        return cast(OpenAIReasoningEffort, self.value)


@dataclass(frozen=True, kw_only=True)
class CompletionRequest:
    """Immutable specification for a single structured completion request.

    Attributes:
        system: System message content.
        user: User message content.
        temperature: Per-request temperature override; None uses client default.
        timeout: Per-request timeout override in seconds; None uses client default.
    """

    system: str
    user: str
    temperature: float | None = None
    timeout: float | None = None


@dataclass(frozen=True, kw_only=True)
class LLMResult[T]:
    """Wrapper for an LLM completion result with associated cost.

    Attributes:
        result: The completion result value.
        cost: Estimated API cost in USD for this call.
    """

    result: T
    cost: float


class _TextCompletionResponse(BaseModel, frozen=True):
    """Structured output model used to extract plain text completions."""

    text: str


def _get_chat_rate_limiter(provider: str, model: str) -> TokenRateLimiter:
    """Create a TokenRateLimiter configured for the given provider and model.

    Args:
        provider: Provider name (e.g. `"openai"`, `"azure-openai"`).
        model: Model name (short alias, since rate limits are per-family).

    Returns:
        A TokenRateLimiter with appropriate limits.
    """
    provider_limits = CHAT_RATE_LIMITS.get(provider, {})
    request_limit, token_limit = provider_limits.get(model, DEFAULT_RATE_LIMIT)
    return TokenRateLimiter(request_limit=request_limit, token_limit=token_limit)


class LLMClient(ABC):
    """Abstract base class for structured LLM clients.

    Provides concrete helpers for batch processing, temperature/timeout
    resolution, and error handling on top of a single abstract
    `_complete_with_retries` method that subclasses must implement.
    """

    def __init__(
        self,
        *,
        model: str,
        seed: int = 0,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_input_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.DEFAULT,
    ) -> None:
        """Initialise client-level defaults.

        Args:
            model: Model identifier.
            seed: Seed for reproducible outputs.
            temperature: Default temperature (overridable per request).
            timeout: Default per-call timeout in seconds (overridable per request).
            max_input_tokens: Safety guardrail; truncate user message if estimated
                input tokens exceed this.  None disables the check.
            reasoning_effort: OpenAI reasoning-effort policy for GPT-5 models.
        """
        self._model = model
        self._seed = seed
        self._temperature = temperature
        self._timeout = timeout
        self._max_input_tokens = max_input_tokens
        self._reasoning_effort = reasoning_effort

    def _resolve_temperature(self, request: CompletionRequest) -> float:
        """Return per-request temperature or client default.

        Args:
            request: The completion request.

        Returns:
            Resolved temperature value.
        """
        return (
            request.temperature
            if request.temperature is not None
            else self._temperature
        )

    def _resolve_timeout(self, request: CompletionRequest) -> float:
        """Return per-request timeout or client default.

        Args:
            request: The completion request.

        Returns:
            Resolved timeout value in seconds.
        """
        return request.timeout if request.timeout is not None else self._timeout

    def _prepare_request(self, request: CompletionRequest) -> CompletionRequest:
        """Truncate the user message if it would exceed max_input_tokens.

        Args:
            request: The original completion request.

        Returns:
            The request, possibly with a truncated user message.
        """
        if self._max_input_tokens is None:
            return request

        system_tokens = estimate_tokens(request.system)
        budget = self._max_input_tokens - system_tokens
        if budget <= 0:
            logger.warning(
                "System prompt alone (%d tokens) exceeds max_input_tokens (%d)",
                system_tokens,
                self._max_input_tokens,
            )
            return request

        user_tokens = estimate_tokens(request.user)
        if user_tokens <= budget:
            return request

        logger.debug(
            "Truncating user message from %d to %d tokens (max_input_tokens=%d)",
            user_tokens,
            budget,
            self._max_input_tokens,
        )
        truncated_user = truncate_to_tokens(request.user, budget)
        return CompletionRequest(
            system=request.system,
            user=truncated_user,
            temperature=request.temperature,
            timeout=request.timeout,
        )

    async def complete[T: BaseModel](
        self,
        request: CompletionRequest,
        response_model: type[T],
    ) -> LLMResult[T | None]:
        """Execute a single structured completion request.

        Executes the completion with retries (depending on the subclass), and catches
        transient or unexpected errors, returning a `None` result so the pipeline can
        continue.

        Args:
            request: The completion request specification.
            response_model: Pydantic model used for structured output parsing.

        Returns:
            An LLMResult wrapping the parsed response model (or None on failure).
        """
        try:
            return await self._complete_with_retries(request, response_model)
        except RETRYABLE_ERRORS:
            logger.warning(
                "Retries exhausted for %s request; returning None",
                self._model,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return LLMResult(result=None, cost=0.0)
        except Exception:
            logger.warning(
                "Unexpected error for %s request; returning None",
                self._model,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return LLMResult(result=None, cost=0.0)

    @abstractmethod
    async def _complete_with_retries[T: BaseModel](
        self,
        request: CompletionRequest,
        response_model: type[T],
    ) -> LLMResult[T | None]:
        """Execute a completion with retries (subclass-provided).

        Args:
            request: The completion request specification.
            response_model: Pydantic model used for structured output parsing.

        Returns:
            An LLMResult wrapping the parsed response model (or None on failure).
        """

    async def complete_text(self, request: CompletionRequest) -> LLMResult[str | None]:
        """Execute a completion and return plain text from a structured wrapper.

        Args:
            request: The completion request specification.

        Returns:
            An LLMResult containing extracted text or None if unavailable.
        """
        llm_result = await self.complete(request, _TextCompletionResponse)
        if llm_result.result is None:
            return LLMResult(result=None, cost=llm_result.cost)
        return LLMResult(result=llm_result.result.text, cost=llm_result.cost)

    async def complete_text_batch[K](
        self,
        requests: Iterable[tuple[K, CompletionRequest]],
        *,
        concurrent: int = 100,
    ) -> AsyncIterator[tuple[K, LLMResult[str | None]]]:
        """Execute multiple completion requests and yield plain-text results.

        Args:
            requests: Iterable of ``(key, request)`` pairs.
            concurrent: Maximum number of concurrent requests.

        Yields:
            Tuples of (key, LLMResult containing text or None).
        """
        async for key, llm_result in self.batch_complete(
            requests,
            response_model=_TextCompletionResponse,
            concurrent=concurrent,
        ):
            if llm_result.result is None:
                yield key, LLMResult(result=None, cost=llm_result.cost)
            else:
                yield (
                    key,
                    LLMResult(result=llm_result.result.text, cost=llm_result.cost),
                )

    async def batch_complete[K, T: BaseModel](
        self,
        requests: Iterable[tuple[K, CompletionRequest]],
        *,
        response_model: type[T],
        concurrent: int = 100,
    ) -> AsyncIterator[tuple[K, LLMResult[T | None]]]:
        """Execute multiple structured completion requests with bounded concurrency.

        Maintains up to *concurrent* requests in flight at once, yielding results as
        they complete.  The iterable is consumed lazily, so at most *concurrent* items
        are materialised at once.

        Yields ``(key, result)`` tuples where the key is the caller-supplied identifier
        for each request.  Results are yielded in completion order, not request order.

        Args:
            requests: Iterable of ``(key, request)`` pairs.
            response_model: Pydantic model used for structured output parsing.
            concurrent: Maximum number of concurrent requests.

        Yields:
            Tuples of (key, LLMResult wrapping the completion).

        Raises:
            ValueError if concurrent < 1.
        """
        if concurrent < 1:
            raise ValueError("concurrent must be >= 1")

        async def run(key: K, req: CompletionRequest) -> tuple[K, LLMResult[T | None]]:
            return key, await self.complete(req, response_model)

        pending: set[asyncio.Task[tuple[K, LLMResult[T | None]]]] = set()
        it = iter(requests)

        def submit_up_to(n: int) -> None:
            """Create up to *n* new tasks from the request iterator."""
            for _ in range(n):
                pair = next(it, None)
                if pair is None:
                    return
                pending.add(asyncio.create_task(run(*pair)))

        submit_up_to(concurrent)

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                submit_up_to(len(done))
                for task in done:
                    yield task.result()
        finally:
            if pending:
                logger.debug("Cancelling pending requests")
                for task in pending:
                    task.cancel()
                logger.debug("Waiting for pending requests to finish")
                await asyncio.gather(*pending, return_exceptions=True)

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying resources (HTTP clients, etc.)."""

    async def __aenter__(self) -> Self:
        """Enter async context."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit async context and close the client."""
        await self.close()

    @staticmethod
    def create(
        provider: str,
        *,
        model: str,
        api_key: str | None = None,
        seed: int = 0,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_input_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.DEFAULT,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ) -> LLMClient:
        """Create an LLMClient for the given provider.

        Args:
            provider: Provider name — `"openai"`, `"azure-openai"`,
                or `"dry-run"`.
            model: Model name to use for completions.
            api_key: API key.  None falls back to environment variable.
            seed: Seed for reproducible outputs.
            temperature: Default temperature.
            timeout: Default per-call timeout in seconds.
            max_input_tokens: Safety guardrail; None disables.
            reasoning_effort: OpenAI reasoning-effort policy for GPT-5 models.
            azure_endpoint: Azure endpoint URL (required for `"azure-openai"`).
            api_version: Azure API version (required for `"azure-openai"`).

        Returns:
            A configured LLMClient instance.

        Raises:
            ValueError: For unknown providers or missing parameters.
        """
        if provider == "dry-run":
            return DryRunClient(
                seed=seed,
                temperature=temperature,
                timeout=timeout,
                max_input_tokens=max_input_tokens,
                reasoning_effort=reasoning_effort,
            )

        if provider == "azure-openai":
            azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise ValueError(
                    "Azure endpoint required: pass azure_endpoint or set"
                    " AZURE_OPENAI_ENDPOINT"
                )
            api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            return AzureOpenAIClient(
                model=model,
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version or "2024-12-01-preview",
                seed=seed,
                temperature=temperature,
                timeout=timeout,
                max_input_tokens=max_input_tokens,
                reasoning_effort=reasoning_effort,
            )

        if provider == "openai":
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            return OpenAIClient(
                model=model,
                api_key=api_key,
                seed=seed,
                temperature=temperature,
                timeout=timeout,
                max_input_tokens=max_input_tokens,
                reasoning_effort=reasoning_effort,
            )

        raise ValueError(
            f"Unknown LLM provider: {provider!r}."
            " Supported: 'openai', 'azure-openai', 'dry-run'"
        )


class DryRunClient(LLMClient):
    """LLMClient that skips API calls and returns no completion payload.

    Useful for testing the full pipeline (streaming, progress bars, file I/O)
    without spending money or requiring API keys.  Collects lightweight
    prompt statistics and logs a summary on close.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_input_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.DEFAULT,
    ) -> None:
        """Initialise dry-run client with zeroed statistics.

        Args:
            seed: Seed (unused, kept for interface compatibility).
            temperature: Temperature (unused).
            timeout: Timeout (unused).
            max_input_tokens: Token limit (unused).
            reasoning_effort: Reasoning-effort policy (unused).
        """
        super().__init__(
            model="dry-run",
            seed=seed,
            temperature=temperature,
            timeout=timeout,
            max_input_tokens=max_input_tokens,
            reasoning_effort=reasoning_effort,
        )

        self._count = 0
        self._total_words = 0
        self._min_words = 0
        self._max_words = 0

    @override
    async def _complete_with_retries[T: BaseModel](
        self,
        request: CompletionRequest,
        response_model: type[T],
    ) -> LLMResult[T | None]:
        """Return no result and record prompt word count.

        Args:
            request: The completion request.
            response_model: Pydantic model used for structured output parsing.

        Returns:
            An LLMResult with ``None`` result and zero cost.
        """
        _ = response_model
        words = len(request.user.split())
        self._count += 1
        self._total_words += words

        if self._count == 1:
            self._min_words = words
        self._min_words = min(self._min_words, words)

        self._max_words = max(self._max_words, words)

        return LLMResult(result=None, cost=0.0)

    @override
    async def close(self) -> None:
        """Log prompt statistics collected during the run."""
        if self._count == 0:
            return

        avg = self._total_words / self._count if self._count > 0 else 0
        logger.info(
            "Dry-run stats: %d requests, prompt words: "
            "min=%d, avg=%.0f, max=%d, total=%d",
            self._count,
            self._min_words,
            avg,
            self._max_words,
            self._total_words,
        )


class OpenAIClient(LLMClient):
    """LLMClient implementation backed by the OpenAI Responses API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        seed: int = 0,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_input_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.DEFAULT,
    ) -> None:
        """Initialise the OpenAI-backed client.

        Args:
            model: Model name for structured completions.
            api_key: API key.  None falls back to environment variable.
            base_url: Override the API base URL (e.g. for OpenRouter).
            seed: Seed for reproducible outputs.
            temperature: Default temperature.
            timeout: Default per-call timeout in seconds.
            max_input_tokens: Safety guardrail; None disables.
            reasoning_effort: OpenAI reasoning-effort policy for GPT-5 models.
        """
        resolved = resolve_model(model)
        super().__init__(
            model=resolved,
            seed=seed,
            temperature=temperature,
            timeout=timeout,
            max_input_tokens=max_input_tokens,
            reasoning_effort=reasoning_effort,
        )
        self._short_model = model
        self._rate_limiter = _get_chat_rate_limiter("openai", model)
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    @override
    @backoff.on_exception(backoff.expo, RETRYABLE_ERRORS, max_tries=5)
    async def _complete_with_retries[T: BaseModel](
        self,
        request: CompletionRequest,
        response_model: type[T],
    ) -> LLMResult[T | None]:
        """Execute a completion with exponential backoff on transient errors."""
        request = self._prepare_request(request)

        combined_text = request.system + request.user
        async with self._rate_limiter.limit(combined_text) as update_usage:
            response = await self._client.responses.parse(
                model=self._model,
                instructions=request.system,
                input=request.user,
                reasoning=(
                    {"effort": effort}
                    if (effort := self._reasoning_effort.to_openai(self._short_model))
                    else openai.omit
                ),
                text_format=response_model,
                temperature=(
                    openai.omit
                    if self._model.startswith(FIXED_TEMPERATURE_PREFIXES)
                    else self._resolve_temperature(request)
                ),
                timeout=self._resolve_timeout(request),
            )

            if usage := response.usage:
                await update_usage(usage.total_tokens)

        cost = 0.0
        if usage := response.usage:
            cost = compute_cost(self._model, usage.input_tokens, usage.output_tokens)

        if not response.output_parsed:
            logger.exception(
                "Failed to parse structured output for model %s",
                response_model.__name__,
            )
            return LLMResult(result=None, cost=cost)

        return LLMResult(result=response.output_parsed, cost=cost)

    @override
    async def close(self) -> None:
        """Close the underlying OpenAI HTTP client."""
        await self._client.close()


class AzureOpenAIClient(LLMClient):
    """LLMClient implementation backed by the Azure OpenAI Chat Completions API.

    Uses `chat.completions.parse()` instead of the Responses API to avoid
    Azure-specific rate-limiting issues with the `/responses` endpoint.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        azure_endpoint: str,
        api_version: str = "2024-12-01-preview",
        seed: int = 0,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_input_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.DEFAULT,
    ) -> None:
        """Initialise the Azure OpenAI-backed client.

        Args:
            model: Deployment name (used directly, not resolved via synonyms).
            api_key: API key.  None falls back to environment variable.
            azure_endpoint: Azure endpoint URL.
            api_version: Azure API version.
            seed: Seed for reproducible outputs.
            temperature: Default temperature.
            timeout: Default per-call timeout in seconds.
            max_input_tokens: Safety guardrail; None disables.
            reasoning_effort: OpenAI reasoning-effort policy for GPT-5 models.
        """
        super().__init__(
            model=model,
            seed=seed,
            temperature=temperature,
            timeout=timeout,
            max_input_tokens=max_input_tokens,
            reasoning_effort=reasoning_effort,
        )
        self._short_model = model
        self._rate_limiter = _get_chat_rate_limiter("azure-openai", model)
        self._client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    @override
    @backoff.on_exception(backoff.expo, RETRYABLE_ERRORS, max_tries=5)
    async def _complete_with_retries[T: BaseModel](
        self,
        request: CompletionRequest,
        response_model: type[T],
    ) -> LLMResult[T | None]:
        """Execute a completion with exponential backoff on transient errors."""
        request = self._prepare_request(request)

        combined_text = request.system + request.user
        async with self._rate_limiter.limit(combined_text) as update_usage:
            response = await self._client.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ],
                response_format=response_model,
                reasoning_effort=(
                    effort
                    if (effort := self._reasoning_effort.to_openai(self._short_model))
                    else openai.omit
                ),
                temperature=(
                    openai.omit
                    if self._model.startswith(FIXED_TEMPERATURE_PREFIXES)
                    else self._resolve_temperature(request)
                ),
                timeout=self._resolve_timeout(request),
            )

            if usage := response.usage:
                await update_usage(usage.total_tokens)

        cost = 0.0
        if usage := response.usage:
            cost = compute_cost(
                self._model, usage.prompt_tokens, usage.completion_tokens
            )

        parsed = response.choices[0].message.parsed if response.choices else None
        if parsed is None:
            logger.exception(
                "Failed to parse structured output for model %s",
                response_model.__name__,
            )
            return LLMResult(result=None, cost=cost)

        return LLMResult(result=parsed, cost=cost)

    @override
    async def close(self) -> None:
        """Close the underlying Azure OpenAI HTTP client."""
        await self._client.close()
