"""Typed wrapper for sentence_transformers.SentenceTransformer.encode."""

# pyright: basic

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer


def encode_texts(
    model: SentenceTransformer,
    items: list[str],
    *,
    normalize: bool = True,
) -> npt.NDArray[np.float32]:
    """Encode texts with a SentenceTransformer model.

    Args:
        model: Loaded SentenceTransformer model.
        items: Texts to encode.
        normalize: Whether to L2-normalise the embeddings.

    Returns:
        Float32 array of shape ``(len(items), dim)``.
    """
    vecs = model.encode(items, normalize_embeddings=normalize)
    return np.asarray(vecs, dtype=np.float32)
