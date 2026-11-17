"""Standalone OpenAI text encoder for embeddings.

External dependencies: openai, backoff, numpy, tiktoken.
"""

import asyncio
import itertools
import logging
from collections.abc import Sequence
from typing import Self

import backoff
import numpy as np
import numpy.typing as npt
import openai

from genret.llm.rate_limit import RETRYABLE_ERRORS, TokenRateLimiter
from genret.llm.utils import detokenise, tokenise

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
"""Output vector dimensions for each supported embedding model."""

EMBEDDING_RATE_LIMITS = {
    "text-embedding-3-small": (10_000, 10_000_000),
    "text-embedding-3-large": (10_000, 10_000_000),
    "text-embedding-ada-002": (10_000, 10_000_000),
}
"""Per-model rate limits as (requests/min, tokens/min) tuples."""

EMBEDDING_MAX_TOKENS = 8191
"""Maximum token count for a single embedding input."""

EMBEDDING_COSTS: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}
"""Cost per million tokens for each supported embedding model."""


def _get_embedding_rate_limiter(model: str) -> TokenRateLimiter:
    """Create a TokenRateLimiter configured for the given embedding model."""
    if model not in EMBEDDING_RATE_LIMITS:
        raise ValueError(
            f"Unknown embedding model: {model}. Known: {list(EMBEDDING_RATE_LIMITS)}"
        )
    request_limit, token_limit = EMBEDDING_RATE_LIMITS[model]
    return TokenRateLimiter(request_limit=request_limit, token_limit=token_limit)


type Vector = npt.NDArray[np.float32]
type Matrix = npt.NDArray[np.float32]


class OpenAIEncoder:
    """Async OpenAI-based text encoder."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ) -> None:
        """Initialise the encoder with the given OpenAI embedding model.

        Args:
            model: OpenAI embedding model name.
            api_key: API key.  None falls back to environment variable.
            azure_endpoint: Azure endpoint URL.  When set, uses Azure client.
            api_version: Azure API version.
        """
        if model not in EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unknown model: {model}. Known: {list(EMBEDDING_DIMENSIONS)}"
            )
        self._client: openai.AsyncOpenAI
        if azure_endpoint is not None:
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version or "2024-12-01-preview",
            )
        else:
            self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._rate_limiter = _get_embedding_rate_limiter(model)
        self._total_tokens = 0
        self._cost_per_million = EMBEDDING_COSTS.get(model, 0.0)

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions for the configured model."""
        return EMBEDDING_DIMENSIONS[self._model]

    @property
    def total_cost(self) -> float:
        """Return estimated total cost in USD for all tokens processed."""
        return self._total_tokens * self._cost_per_million / 1_000_000

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def __aenter__(self) -> Self:
        """Enter async context."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit async context and close the client."""
        await self.close()

    @backoff.on_exception(backoff.expo, RETRYABLE_ERRORS, max_tries=5)
    async def encode(self, text: str) -> Vector:
        """Encode a single text string as a vector."""
        async with self._rate_limiter.limit(text) as update_usage:
            response = await self._client.embeddings.create(
                input=[text], model=self._model
            )

            if usage := response.usage:
                await update_usage(usage.total_tokens)
                self._total_tokens += usage.total_tokens

        return np.array(response.data[0].embedding, dtype=np.float32)

    @backoff.on_exception(backoff.expo, RETRYABLE_ERRORS, max_tries=5)
    async def encode_multi(self, texts: Sequence[str]) -> Matrix:
        """Encode multiple texts as a matrix."""
        if not texts:
            raise ValueError("texts must not be empty")

        async with self._rate_limiter.limit("\n".join(texts)) as update_usage:
            input = list(texts)
            response = await self._client.embeddings.create(
                input=input, model=self._model
            )

            if usage := response.usage:
                await update_usage(usage.total_tokens)
                self._total_tokens += usage.total_tokens

        return np.array([d.embedding for d in response.data], dtype=np.float32)

    async def batch_encode(
        self, texts: Sequence[str], batch_size: int = 128, max_concurrency: int = 32
    ) -> Matrix:
        """Encode texts in batches and stack into a single matrix.

        Texts exceeding EMBEDDING_MAX_TOKENS are truncated (token-level) before sending
        to the API. If an embedding request fails, that batch's rows remain zero vectors.
        """
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        processed, truncated = await asyncio.to_thread(_preprocess_texts, texts)
        if truncated:
            logger.warning(
                "Truncated %d/%d texts to %d-token embedding limit",
                truncated,
                len(texts),
                EMBEDDING_MAX_TOKENS,
            )

        # Pre-fill with zeros; failed batches stay zeros.
        result = np.zeros((len(processed), self.dimensions), dtype=np.float32)

        pending: set[asyncio.Task[None]] = set()
        offset = 0

        async def run_batch(start: int, batch: Sequence[str]) -> None:
            try:
                mat = await self.encode_multi(batch)
                mat = _normalize_and_check_embedding_matrix(
                    mat, batch_len=len(batch), dimensions=self.dimensions
                )
                result[start : start + len(batch)] = mat
            except Exception:
                logger.exception(
                    "Embedding batch failed; leaving %d rows as zeros (rows %d..%d)",
                    len(batch),
                    start,
                    start + len(batch) - 1,
                )
                return

        for batch in itertools.batched(processed, n=batch_size, strict=False):
            start = offset
            offset += len(batch)

            pending.add(asyncio.create_task(run_batch(start, batch)))
            if len(pending) >= max_concurrency:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    # propagate unexpected errors (should be none) since we already
                    # catch Exception in `run_batch()`
                    t.result()

        if pending:
            done, _ = await asyncio.wait(pending)
            for t in done:
                t.result()

        return result


def _preprocess_texts(texts: Sequence[str]) -> tuple[list[str], int]:
    """Process each text in `text`, truncating to `EMBEDDING_MAX_TOKENS` if necessary.

    Returns:
        Tuple of (processed texts, number of truncated items)
    """
    processed: list[str] = []
    truncated = 0

    for text_raw in texts:
        tokens = tokenise(text_raw)
        if len(tokens) > EMBEDDING_MAX_TOKENS:
            processed.append(detokenise(tokens[:EMBEDDING_MAX_TOKENS]))
            truncated += 1
        else:
            processed.append(text_raw)

    return processed, truncated


def _normalize_and_check_embedding_matrix(
    mat: npt.NDArray[np.float32], *, batch_len: int, dimensions: int
) -> npt.NDArray[np.float32]:
    """Normalise `mat` if singleton, and check if `mat` shapes are valid.

    Singletons `(d,)` are normalised to `(1, d)`.
    `mat` shape must be `(batch_len, dimensions)`

    Returns:
        Normalised `mat` if required, or the original.

    Raises:
        ValueError if `mat` has an invalid shape.
    """
    # Singleton normalization: (d,) -> (1, d)
    if (
        batch_len == 1
        and getattr(mat, "ndim", None) == 1
        and mat.shape == (dimensions,)
    ):
        mat = mat.reshape(1, dimensions)

    if mat.shape != (batch_len, dimensions):
        raise ValueError(
            f"encode_multi returned shape {mat.shape}, expected {(batch_len, dimensions)}"
        )

    return mat


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L12-v2"
"""Default embedding model. Local MiniLM — the project standard, no API cost.

OpenAI embedding models are still supported (pass a ``text-embedding-*`` name),
but should only be used with a specific reason; MiniLM is the default.
"""


class LocalEncoder:
    """Local sentence-transformers encoder (e.g. MiniLM-L12-v2).

    Matches the :class:`OpenAIEncoder` interface but runs a local model with no
    API calls and zero cost. ``sentence-transformers`` is imported lazily so the
    base install (without torch) can still import this module; only constructing
    a ``LocalEncoder`` requires the dependency.
    """

    def __init__(self, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Load the sentence-transformers model.

        Args:
            model: HuggingFace / sentence-transformers model name.
        """
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model_name = model
        self._st = SentenceTransformer(model)
        dim = self._st.get_sentence_embedding_dimension()
        self._dimensions = int(dim) if dim is not None else 384

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        return self._dimensions

    @property
    def total_cost(self) -> float:
        """Local inference is free."""
        return 0.0

    async def close(self) -> None:
        """No-op; kept for interface parity with :class:`OpenAIEncoder`."""
        return

    async def __aenter__(self) -> Self:
        """Enter async context."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit async context."""
        return

    def _encode_sync(self, texts: list[str]) -> Matrix:
        from genret._shed.sentence_transformers import encode_texts  # noqa: PLC0415

        return encode_texts(self._st, texts, normalize=True)

    async def encode(self, text: str) -> Vector:
        """Encode a single text string as a unit-norm vector."""
        return (await self.encode_multi([text]))[0]

    async def encode_multi(self, texts: Sequence[str]) -> Matrix:
        """Encode multiple texts as a unit-norm matrix."""
        if not texts:
            raise ValueError("texts must not be empty")
        return await asyncio.to_thread(self._encode_sync, list(texts))

    async def batch_encode(
        self, texts: Sequence[str], batch_size: int = 128, max_concurrency: int = 32
    ) -> Matrix:
        """Encode texts into a single matrix (sentence-transformers batches internally)."""
        del batch_size, max_concurrency  # handled internally by the local model
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return await asyncio.to_thread(self._encode_sync, list(texts))


type Encoder = OpenAIEncoder | LocalEncoder


def make_encoder(
    model: str = DEFAULT_EMBEDDING_MODEL,
    *,
    api_key: str | None = None,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
) -> Encoder:
    """Build an encoder for *model*, defaulting to local MiniLM.

    Known OpenAI embedding names (``text-embedding-*``) route to the API-backed
    :class:`OpenAIEncoder`; any other name is treated as a local
    sentence-transformers model via :class:`LocalEncoder` (the default path).

    Args:
        model: Embedding model name.
        api_key: OpenAI/Azure API key (ignored for local models).
        azure_endpoint: Azure endpoint (ignored for local models).
        api_version: Azure API version (ignored for local models).

    Returns:
        A ready-to-use encoder.
    """
    if model in EMBEDDING_DIMENSIONS:
        return OpenAIEncoder(
            model,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
    return LocalEncoder(model)
