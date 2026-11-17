"""Tests for the embedding-encoder factory and the local MiniLM encoder."""

import numpy as np

from genret.llm.encoder import (
    DEFAULT_EMBEDDING_MODEL,
    LocalEncoder,
    OpenAIEncoder,
    make_encoder,
)


def test_default_model_is_local_minilm() -> None:
    """The default embedding model is local MiniLM, not OpenAI."""
    assert "MiniLM" in DEFAULT_EMBEDDING_MODEL
    assert "text-embedding" not in DEFAULT_EMBEDDING_MODEL


def test_factory_routes_openai_names_to_openai_encoder() -> None:
    """A 'text-embedding-*' name routes to the API-backed encoder."""
    enc = make_encoder("text-embedding-3-small", api_key="test-key")
    assert isinstance(enc, OpenAIEncoder)


def test_factory_routes_other_names_to_local_encoder() -> None:
    """Any non-OpenAI name (the default) routes to the local encoder."""
    enc = make_encoder()
    assert isinstance(enc, LocalEncoder)


async def test_local_encoder_produces_unit_norm_384d() -> None:
    """The default local encoder yields unit-norm float32 384-d vectors."""
    async with make_encoder() as enc:
        assert enc.dimensions == 384
        assert enc.total_cost == 0.0
        mat = await enc.encode_multi(["machine translation", "word alignment"])
    assert mat.shape == (2, 384)
    assert mat.dtype == np.float32
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-3)
