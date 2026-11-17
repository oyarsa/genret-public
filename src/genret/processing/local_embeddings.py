# pyright: basic
"""Compute facet embeddings using a local HuggingFace model.

Produces the same `.npz` output format as the OpenAI embedding pipeline:
one key per corpus ID, mapping to a float32 array of shape
`(n_items, embedding_dim)` with all facet items concatenated in facet order.
"""

import importlib
import itertools
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from genret._shed.numpy import savez as np_savez
from genret._shed.torch import unsqueeze_mask
from genret._shed.transformers import load_tokenizer
from genret.paper_parsing import Facets
from genret.processing.facet_evaluation import FACET_NAMES
from genret.utils import iter_jsonl, track_progress

logger = logging.getLogger(__name__)


def _collect_items(facets: Facets) -> list[str]:
    """Collect all facet items in canonical order.

    Args:
        facets: Paper facets record.

    Returns:
        Flat list of facet item strings in FACET_NAMES order.
    """
    items: list[str] = []
    for name in FACET_NAMES:
        items.extend(getattr(facets, name))
    return items


def embed_texts(
    texts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 64,
    device: str = "cpu",
) -> npt.NDArray[np.float32]:
    """Embed a list of texts using mean pooling over token embeddings.

    Args:
        texts: List of text strings to embed.
        model: HuggingFace model.
        tokenizer: Matching tokeniser.
        batch_size: Number of texts per forward pass.
        device: Device to run on.

    Returns:
        Shape `(len(texts), D)` float32 embedding matrix.
    """
    all_embeddings: list[npt.NDArray[np.float32]] = []

    for batch in itertools.batched(texts, batch_size, strict=False):
        encoded = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        # Mean pooling over non-padding tokens
        token_embeddings = outputs.last_hidden_state
        attention_mask = unsqueeze_mask(encoded["attention_mask"])
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts

        # L2 normalise
        norms = pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        normalised = pooled / norms

        all_embeddings.append(normalised.cpu().numpy().astype(np.float32))

    return np.concatenate(all_embeddings, axis=0)


def embed_facets(
    facets_path: Path,
    output_path: Path,
    model_id: str = "allenai/specter2_base",
    adapter_id: str | None = "allenai/specter2_proximity",
    batch_size: int = 64,
    device: str | None = None,
) -> None:
    """Embed all facet items and write to an npz cache file.

    Produces one key per corpus ID in the output npz, with each value
    being a float32 array of shape `(n_items, D)`.

    Args:
        facets_path: Path to facets JSONL file.
        output_path: Output path for the `.npz` embeddings cache.
        model_id: HuggingFace model ID.
        adapter_id: Optional PEFT adapter ID to load on top of the base
            model (e.g. `allenai/specter2_proximity` for paper similarity).
        batch_size: Texts per forward pass.
        device: Device to use. If None, auto-detects CUDA.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model %s on %s", model_id, device)
    tokenizer = load_tokenizer(model_id)

    if adapter_id is not None:
        # 'adapters' was dropped from the training extra (it pins
        # transformers<4.58). Install ad hoc to use adapter embeddings.
        auto_adapter_model = importlib.import_module("adapters").AutoAdapterModel

        logger.info("Loading model with adapter %s", adapter_id)
        model = auto_adapter_model.from_pretrained(model_id)
        model.load_adapter(adapter_id, source="hf", set_active=True)
    else:
        model = AutoModel.from_pretrained(model_id)

    model = model.to(device)
    model.eval()

    embedding_dim = model.config.hidden_size
    logger.info("Embedding dimension: %d", embedding_dim)

    # Load all facets
    records = list(iter_jsonl(facets_path))
    logger.info("Loaded %d facet records", len(records))

    # Embed all papers
    embeddings: dict[str, npt.NDArray[np.float32]] = {}

    for record in track_progress(records, desc="Embedding facets..."):
        facets = Facets.model_validate(record)
        items = _collect_items(facets)

        if not items:
            embeddings[str(facets.corpusid)] = np.zeros(
                (0, embedding_dim), dtype=np.float32
            )
            continue

        item_embeddings = embed_texts(items, model, tokenizer, batch_size, device)
        embeddings[str(facets.corpusid)] = item_embeddings

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.npz")
    np_savez(tmp_path, embeddings)
    tmp_path.rename(output_path)

    logger.info(
        "Wrote %d paper embeddings (dim=%d) to %s",
        len(embeddings),
        embedding_dim,
        output_path,
    )
