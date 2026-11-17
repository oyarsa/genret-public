"""Text encoders for the explorer: ONNX Runtime (default) or torch.

The explorer embeds free-text queries and on-demand facet items with
all-MiniLM-L12-v2. The ONNX backend runs on tokenizers + onnxruntime alone
(~150 MB RSS); the sentence-transformers backend needs the torch stack
(~800 MB RSS) and exists as the reference implementation. Both produce
L2-normalised float32 embeddings and are validated to match with cosine
similarity > 0.9999 (see ``scripts/export_minilm_onnx.py``).
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from genret._shed.onnxruntime import create_cpu_session, run_session

logger = logging.getLogger(__name__)

MAX_LENGTH = 128
"""all-MiniLM-L12-v2 truncates inputs at 128 tokens."""

DEFAULT_ONNX_DIR = Path("output/minilm_onnx")
"""Default location of the exported ONNX encoder."""


class TextEncoder(Protocol):
    """Encodes texts into L2-normalised float32 embeddings."""

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Encode texts into embeddings of shape ``(len(texts), dim)``."""
        ...


class OnnxEncoder:
    """MiniLM encoder backed by ONNX Runtime; no torch dependency.

    Expects ``model.onnx`` and ``tokenizer.json`` in ``model_dir``, produced
    by ``scripts/export_minilm_onnx.py``.
    """

    def __init__(self, model_dir: Path, batch_size: int = 32) -> None:
        from tokenizers import Tokenizer  # noqa: PLC0415

        model_path = model_dir / "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX encoder not found at {model_path}. Export it with:"
                " uv run python scripts/export_minilm_onnx.py"
            )

        self._session = create_cpu_session(model_path)
        self._batch_size = batch_size

        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=MAX_LENGTH)
        pad_id = tokenizer.token_to_id("[PAD]") or 0
        tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")  # noqa: S106
        self._tokenizer = tokenizer

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Encode texts via tokenize -> transformer -> mean pool -> normalise."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        chunks: list[npt.NDArray[np.float32]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            encodings = self._tokenizer.encode_batch(batch)
            ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)

            hidden = run_session(
                self._session,
                "last_hidden_state",
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "token_type_ids": np.zeros_like(ids),
                },
            )

            weights = mask[:, :, None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / np.maximum(
                weights.sum(axis=1), 1e-9
            )
            chunks.append(pooled)

        embeddings = np.concatenate(chunks, axis=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / np.maximum(norms, 1e-12)).astype(np.float32)


class SentenceTransformerEncoder:
    """Reference encoder wrapping sentence-transformers (requires torch)."""

    def __init__(self, model_name: str = "all-MiniLM-L12-v2") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Encode texts with the sentence-transformers reference model."""
        from genret._shed.sentence_transformers import encode_texts  # noqa: PLC0415

        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return encode_texts(self._model, list(texts), normalize=True)


def load_encoder(backend: str, onnx_dir: Path = DEFAULT_ONNX_DIR) -> TextEncoder:
    """Load the configured encoder backend.

    Args:
        backend: ``"onnx"`` or ``"torch"``.
        onnx_dir: Directory with the exported ONNX model (onnx backend only).

    Returns:
        A ready-to-use encoder.
    """
    match backend:
        case "onnx":
            logger.info("Loading ONNX MiniLM encoder from %s", onnx_dir)
            return OnnxEncoder(onnx_dir)
        case "torch":
            logger.info("Loading sentence-transformers MiniLM encoder")
            return SentenceTransformerEncoder()
        case _:
            raise ValueError(f"Unknown encoder backend: {backend!r}")
