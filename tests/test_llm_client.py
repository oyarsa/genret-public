"""Unit tests for the llm_client module."""

import asyncio
from dataclasses import dataclass
from typing import ClassVar, cast

import openai
import pytest
from pydantic import BaseModel

from genret.llm.client import (
    CompletionRequest,
    LLMClient,
    LLMResult,
    ReasoningEffort,
)


class _SampleResponse(BaseModel, frozen=True):
    """Simple structured output model used throughout tests."""

    value: str = "ok"


class _TextResponse(BaseModel, frozen=True):
    """Text wrapper model used by complete_text helper tests."""

    text: str


@dataclass(frozen=True, kw_only=True)
class _FakeUsage:
    """Minimal usage payload matching Responses API usage fields."""

    total_tokens: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, kw_only=True)
class _FakeResponse:
    """Minimal Responses API parse response payload."""

    usage: _FakeUsage | None
    output_parsed: BaseModel | None


class _FakeResponses:
    """Capture kwargs passed to responses.parse."""

    def __init__(self) -> None:
        """Initialise call capture storage."""
        self.last_kwargs: dict[str, object] | None = None

    async def parse(self, **kwargs: object) -> _FakeResponse:
        """Record kwargs and return a minimal successful response."""
        self.last_kwargs = dict(kwargs)
        return _FakeResponse(
            usage=_FakeUsage(total_tokens=7, input_tokens=3, output_tokens=4),
            output_parsed=_SampleResponse(value="ok"),
        )


@dataclass(frozen=True, kw_only=True)
class _FakeResponsesResource:
    """Container for fake Responses API resource."""

    parse_client: _FakeResponses

    async def parse(self, **kwargs: object) -> _FakeResponse:
        """Delegate parse calls to the underlying fake resource."""
        return await self.parse_client.parse(**kwargs)


class _FakeAsyncOpenAI:
    """Drop-in stand-in for `openai.AsyncOpenAI` used by tests."""

    last_instance: ClassVar[_FakeAsyncOpenAI | None] = None

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        """Create a fake client and expose Responses API capture."""
        self.api_key = api_key
        self.base_url = base_url
        self.responses = _FakeResponsesResource(parse_client=_FakeResponses())
        _FakeAsyncOpenAI.last_instance = self

    async def close(self) -> None:
        """No-op close method."""


class StubClient(LLMClient):
    """Minimal LLMClient subclass for testing concrete helper methods."""

    def __init__(
        self,
        responses: list[BaseModel | None] | None = None,
        *,
        seed: int = 0,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_input_tokens: int | None = None,
    ) -> None:
        """Initialise with a list of canned responses.

        Args:
            responses: Parsed responses to return from complete(), consumed in order.
            seed: Seed for reproducible outputs.
            temperature: Default temperature.
            timeout: Default timeout.
            max_input_tokens: Safety guardrail; None disables.
        """
        super().__init__(
            model="stub-model",
            seed=seed,
            temperature=temperature,
            timeout=timeout,
            max_input_tokens=max_input_tokens,
        )
        self._responses = list(responses) if responses else []
        self._call_count = 0

    async def _complete_with_retries[T: BaseModel](
        self,
        request: CompletionRequest,
        response_model: type[T],
    ) -> LLMResult[T | None]:
        """Return the next canned response wrapped in LLMResult.

        Args:
            request: The completion request (truncation applied via _prepare_request).
            response_model: Pydantic model type requested by caller.

        Returns:
            LLMResult wrapping the next response from the pre-configured list, or None.
        """
        _ = response_model
        request = self._prepare_request(request)

        if self._call_count < len(self._responses):
            result = self._responses[self._call_count]
            self._call_count += 1
            if result is None:
                return LLMResult(result=None, cost=0.0)
            return LLMResult(result=cast(T, result), cost=0.0)

        return LLMResult(result=None, cost=0.0)

    async def close(self) -> None:
        """No-op close."""


class TestCompletionRequest:
    """Tests for CompletionRequest dataclass."""

    def test_defaults(self) -> None:
        """Should have sensible defaults for optional fields."""
        req = CompletionRequest(system="sys", user="usr")
        assert req.temperature is None
        assert req.timeout is None

    def test_explicit_overrides(self) -> None:
        """Should store explicitly provided values."""
        req = CompletionRequest(
            system="sys",
            user="usr",
            temperature=0.5,
            timeout=30.0,
        )
        assert req.temperature == 0.5
        assert req.timeout == 30.0


class TestReasoningEffort:
    """Tests for reasoning-effort mapping."""

    def test_default_omits_parameter(self) -> None:
        """DEFAULT should map to omitted for all models."""
        assert ReasoningEffort.DEFAULT.to_openai("gpt-5.2") is None
        assert ReasoningEffort.DEFAULT.to_openai("gpt-4o-mini") is None

    def test_non_gpt5_omits_parameter(self) -> None:
        """Non-GPT-5 models should omit reasoning_effort."""
        assert ReasoningEffort.HIGH.to_openai("gpt-4o-mini") is None

    def test_gpt_5_2_minimal_normalises_to_none(self) -> None:
        """GPT-5.2 should map MINIMAL to NONE."""
        assert ReasoningEffort.MINIMAL.to_openai("gpt-5.2-2025-12-11") == "none"

    def test_xhigh_maps_for_gpt5(self) -> None:
        """GPT-5 models should pass through XHIGH."""
        assert ReasoningEffort.XHIGH.to_openai("gpt-5-mini-2025-08-07") == "xhigh"


class TestResolution:
    """Tests for temperature and timeout resolution logic."""

    def test_uses_client_defaults(self) -> None:
        """Should use client defaults when request fields are None."""
        client = StubClient(temperature=0.7, timeout=60.0)
        req = CompletionRequest(system="s", user="u")
        assert client._resolve_temperature(req) == 0.7
        assert client._resolve_timeout(req) == 60.0

    def test_request_overrides_client(self) -> None:
        """Should prefer request values over client defaults."""
        client = StubClient(temperature=0.7, timeout=60.0)
        req = CompletionRequest(system="s", user="u", temperature=0.2, timeout=10.0)
        assert client._resolve_temperature(req) == 0.2
        assert client._resolve_timeout(req) == 10.0


class TestBatchComplete:
    """Tests for the batch_complete async iterator."""

    async def test_yields_all_results(self) -> None:
        """Should yield all results with correct keys."""
        client = StubClient(
            responses=[
                _SampleResponse(value="a"),
                _SampleResponse(value="b"),
                _SampleResponse(value="c"),
            ]
        )
        requests = [(i, CompletionRequest(system="s", user=f"u{i}")) for i in range(3)]

        results = {
            key: llm_result.result.value if llm_result.result is not None else None
            async for key, llm_result in client.batch_complete(
                requests,
                response_model=_SampleResponse,
            )
        }
        assert results == {0: "a", 1: "b", 2: "c"}

    async def test_respects_batch_size(self) -> None:
        """Should limit concurrent calls to batch_size."""
        peak_concurrent = 0
        current_concurrent = 0

        class ConcurrencyTracker(StubClient):
            """Tracks peak concurrency."""

            async def complete[T: BaseModel](
                self,
                request: CompletionRequest,
                response_model: type[T],
            ) -> LLMResult[T | None]:
                """Track concurrency."""
                _ = request, response_model
                nonlocal peak_concurrent, current_concurrent

                current_concurrent += 1
                peak_concurrent = max(peak_concurrent, current_concurrent)

                await asyncio.sleep(0.05)

                current_concurrent -= 1
                return LLMResult(result=cast(T, _SampleResponse(value="ok")), cost=0.0)

        client = ConcurrencyTracker()
        requests = [(i, CompletionRequest(system="s", user=f"u{i}")) for i in range(10)]

        results: list[_SampleResponse | None] = []
        async for _, llm_result in client.batch_complete(
            requests,
            response_model=_SampleResponse,
            concurrent=3,
        ):
            results.append(llm_result.result)

        assert all(r is not None and r.value == "ok" for r in results)
        assert len(results) == 10
        assert peak_concurrent <= 3

    async def test_empty_requests(self) -> None:
        """Should handle empty request list."""
        client = StubClient()

        empty: list[tuple[int, CompletionRequest]] = []
        results: list[tuple[int, LLMResult[_SampleResponse | None]]] = []
        async for item in client.batch_complete(empty, response_model=_SampleResponse):
            results.append(item)

        assert results == []


class TestCompleteText:
    """Tests for plain-text convenience completion helpers."""

    async def test_complete_text_extracts_text(self) -> None:
        """Should extract `.text` from structured completion output."""
        client = StubClient(responses=[_TextResponse(text="hello")])

        result = await client.complete_text(CompletionRequest(system="s", user="u"))

        assert result.result == "hello"
        assert result.cost == 0.0

    async def test_complete_text_returns_none_when_missing(self) -> None:
        """Should preserve None result when structured output is missing."""
        client = StubClient(responses=[None])

        result = await client.complete_text(CompletionRequest(system="s", user="u"))

        assert result.result is None
        assert result.cost == 0.0

    async def test_complete_text_batch_extracts_text(self) -> None:
        """Should extract plain text for each item in batched completions."""
        client = StubClient(
            responses=[
                _TextResponse(text="a"),
                None,
                _TextResponse(text="c"),
            ]
        )
        requests = [(i, CompletionRequest(system="s", user=f"u{i}")) for i in range(3)]

        results = {
            key: llm_result.result
            async for key, llm_result in client.complete_text_batch(
                requests,
                concurrent=2,
            )
        }

        assert results == {0: "a", 1: None, 2: "c"}


class TestMaxInputTokens:
    """Tests for the max_input_tokens truncation behaviour."""

    async def test_truncates_when_exceeded(self) -> None:
        """Should truncate user message when estimated tokens exceed max."""
        long_user = "word " * 500  # ~500 tokens
        client = StubClient(
            responses=[_SampleResponse(value="ok")],
            max_input_tokens=50,
        )
        request = CompletionRequest(system="sys", user=long_user)

        result = await client.complete(request, _SampleResponse)
        assert result.result is not None
        assert result.result.value == "ok"

    async def test_allows_when_within_limit(self) -> None:
        """Should not truncate when tokens are within the limit."""
        client = StubClient(
            responses=[_SampleResponse(value="ok")],
            max_input_tokens=100_000,
        )
        request = CompletionRequest(system="hi", user="there")

        result = await client.complete(request, _SampleResponse)
        assert result.result is not None
        assert result.result.value == "ok"

    async def test_no_check_when_disabled(self) -> None:
        """Should skip check when max_input_tokens is None."""
        client = StubClient(
            responses=[_SampleResponse(value="ok")],
            max_input_tokens=None,
        )
        request = CompletionRequest(
            system="long " * 1000,
            user="text " * 1000,
        )

        result = await client.complete(request, _SampleResponse)
        assert result.result is not None
        assert result.result.value == "ok"


class TestPrepareRequest:
    """Tests for the _prepare_request truncation logic."""

    def test_returns_same_when_no_limit(self) -> None:
        """Should return the same request when max_input_tokens is None."""
        client = StubClient(max_input_tokens=None)
        req = CompletionRequest(system="s", user="u" * 10000)
        assert client._prepare_request(req) is req

    def test_returns_same_when_within_budget(self) -> None:
        """Should return the same request when within token budget."""
        client = StubClient(max_input_tokens=100_000)
        req = CompletionRequest(system="s", user="short text")
        assert client._prepare_request(req) is req

    def test_truncates_long_user_message(self) -> None:
        """Should truncate user message to fit within budget."""
        client = StubClient(max_input_tokens=20)
        long_user = "word " * 500
        req = CompletionRequest(system="s", user=long_user)
        result = client._prepare_request(req)
        assert result is not req
        assert len(result.user) < len(req.user)
        assert result.system == req.system


class TestFactory:
    """Tests for the LLMClient.create static factory."""

    def test_unknown_provider_raises(self) -> None:
        """Should raise ValueError for unknown provider."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMClient.create("unknown", model="gpt-4o")


class TestOpenAIReasoningEffort:
    """Tests for OpenAI reasoning-effort request wiring."""

    async def test_passes_normalised_reasoning_effort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should send 'none' when MINIMAL is used with GPT-5.2."""
        _FakeAsyncOpenAI.last_instance = None
        monkeypatch.setattr("genret.llm.client.openai.AsyncOpenAI", _FakeAsyncOpenAI)

        client = LLMClient.create(
            "openai",
            model="gpt-5.2",
            api_key="test-key",
            reasoning_effort=ReasoningEffort.MINIMAL,
        )
        try:
            await client.complete(
                CompletionRequest(system="s", user="u"), _SampleResponse
            )
        finally:
            await client.close()

        fake = cast(_FakeAsyncOpenAI | None, _FakeAsyncOpenAI.last_instance)
        assert fake is not None
        kwargs = fake.responses.parse_client.last_kwargs
        assert kwargs is not None
        assert kwargs["reasoning"] == {"effort": "none"}

    async def test_omits_reasoning_effort_for_non_gpt5(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should omit reasoning_effort for non-GPT-5 models."""
        _FakeAsyncOpenAI.last_instance = None
        monkeypatch.setattr("genret.llm.client.openai.AsyncOpenAI", _FakeAsyncOpenAI)

        client = LLMClient.create(
            "openai",
            model="gpt-4o-mini",
            api_key="test-key",
            reasoning_effort=ReasoningEffort.HIGH,
        )
        try:
            await client.complete(
                CompletionRequest(system="s", user="u"), _SampleResponse
            )
        finally:
            await client.close()

        fake = cast(_FakeAsyncOpenAI | None, _FakeAsyncOpenAI.last_instance)
        assert fake is not None
        kwargs = fake.responses.parse_client.last_kwargs
        assert kwargs is not None
        assert kwargs["reasoning"] is openai.omit

    async def test_uses_response_parse_with_model_text_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should pass the Pydantic model via Responses API parse text_format."""
        _FakeAsyncOpenAI.last_instance = None
        monkeypatch.setattr("genret.llm.client.openai.AsyncOpenAI", _FakeAsyncOpenAI)

        client = LLMClient.create(
            "openai",
            model="gpt-5",
            api_key="test-key",
        )
        try:
            await client.complete(
                CompletionRequest(system="s", user="u"), _SampleResponse
            )
        finally:
            await client.close()

        fake = cast(_FakeAsyncOpenAI | None, _FakeAsyncOpenAI.last_instance)
        assert fake is not None
        kwargs = fake.responses.parse_client.last_kwargs
        assert kwargs is not None
        assert kwargs["text_format"] is _SampleResponse


class TestAsyncContextManager:
    """Tests for async context manager protocol."""

    async def test_context_manager(self) -> None:
        """Should work as an async context manager."""
        async with StubClient(responses=[_SampleResponse(value="result")]) as client:
            result = await client.complete(
                CompletionRequest(system="s", user="u"),
                _SampleResponse,
            )

        assert result.result is not None
        assert result.result.value == "result"
