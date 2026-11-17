"""Export all-MiniLM-L12-v2 to ONNX for torch-free explorer serving.

Writes ``model.onnx`` and ``tokenizer.json`` to the output directory, then
validates the export against the sentence-transformers reference: every
embedding must match with cosine similarity > 0.9999.

Requires the torch environment (``just sync-cuda`` or ``just sync-cpu``);
serving afterwards only needs onnxruntime + tokenizers.

Usage:
    uv run python scripts/export_minilm_onnx.py --output-dir output/minilm_onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import override

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel

from genret._shed.transformers import load_auto_model, load_tokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
MAX_LENGTH = 128
"""all-MiniLM-L12-v2 truncates inputs at 128 tokens."""

VALIDATION_TEXTS = [
    "attention mechanisms for machine translation",
    "How powerful is cross-attention alone at adapting a pretrained "
    "Transformer to a new machine-translation language pair?",
    "We propose a graph-based retrieval method using facet decomposition "
    "and reciprocal rank fusion over dense similarity signals.",
    "short",
    "A much longer text that should exercise padding and truncation "
    "behaviour together with the other batch entries, including some "
    "unusual tokens like corpusid-12345 and RRF@k fusion metrics." * 4,
]


class _PooledModel(torch.nn.Module):
    """Wraps the transformer to expose only last_hidden_state for export."""

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    @override
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return out.last_hidden_state


def export(output_dir: Path) -> None:
    """Export the model and tokenizer to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_auto_model(MODEL_NAME)
    model.eval()

    enc = tokenizer(
        ["an example sentence", "another"],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    dynamic = {0: "batch", 1: "sequence"}
    torch.onnx.export(
        _PooledModel(model),
        (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
        f=str(output_dir / "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": dynamic,
            "attention_mask": dynamic,
            "token_type_ids": dynamic,
            "last_hidden_state": dynamic,
        },
        dynamo=False,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"Exported model.onnx + tokenizer files to {output_dir}")


def validate(output_dir: Path) -> None:
    """Compare ONNX embeddings against sentence-transformers reference."""
    # Deferred so export can run before the package import graph is touched.
    from genret.explorer.encoder import OnnxEncoder  # noqa: PLC0415

    onnx_enc = OnnxEncoder(output_dir)
    got = onnx_enc.encode(VALIDATION_TEXTS)

    reference_model = SentenceTransformer(MODEL_NAME, device="cpu")
    want = np.asarray(
        reference_model.encode(VALIDATION_TEXTS, normalize_embeddings=True),
        dtype=np.float32,
    )

    cosines = (got * want).sum(axis=1)
    print(f"cosine(onnx, torch) per text: {np.round(cosines, 6)}")
    if cosines.min() <= 0.9999:
        raise SystemExit(f"Parity check FAILED: min cosine {cosines.min():.6f}")
    print("Parity check passed (all cosines > 0.9999)")


def main() -> None:
    """Export and validate the ONNX encoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/minilm_onnx"))
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip the parity check"
    )
    args = parser.parse_args()

    export(args.output_dir)
    if not args.skip_validation:
        validate(args.output_dir)


if __name__ == "__main__":
    main()
