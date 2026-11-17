# ruff: noqa: PLC0415  Lazy-import heavy sentence-transformers only when snap=cosine
"""Snap-to-nearest helpers for off-trie GLEN beam search outputs.

When a beam decodes without the trie constraint, its token sequence may not
correspond to any stored DocID. The snappers map such sequences to the
nearest real DocID by either token-level Levenshtein distance or by cosine
similarity between the decoded text's MiniLM embedding and pre-computed
DocID embeddings.

Both snappers operate per facet — they only consider DocIDs registered for
the same facet as the decode that produced them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from genret._shed.transformers import decode_tokens
from genret.training.glen_docids import GlenIndex
from genret.types import CorpusId

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"


@dataclass(frozen=True, kw_only=True)
class DocIDIndex:
    """Per-facet lookup for snap-to-nearest scoring.

    Stores parallel arrays so the i-th entry is the same DocID across fields:
    its token sequence, its decoded text, its sentence embedding (cosine
    mode only), and the list of papers that share it.
    """

    facets: tuple[str, ...]
    token_seqs: dict[str, list[tuple[int, ...]]] = field(default_factory=dict)
    texts: dict[str, list[str]] = field(default_factory=dict)
    paper_lists: dict[str, list[list[CorpusId]]] = field(default_factory=dict)
    embeddings: dict[str, npt.NDArray[np.float32]] = field(default_factory=dict)
    """L2-normalised, only populated when built for cosine snap."""


def build_docid_index(
    index: GlenIndex,
    tokeniser: PreTrainedTokenizerBase,
    *,
    facets: tuple[str, ...],
    with_embeddings: bool,
) -> DocIDIndex:
    """Materialise per-facet parallel arrays from a ``GlenIndex``.

    When ``with_embeddings`` is True, loads the MiniLM sentence encoder and
    pre-computes a sentence embedding per DocID.
    """
    token_seqs: dict[str, list[tuple[int, ...]]] = {}
    texts: dict[str, list[str]] = {}
    paper_lists: dict[str, list[list[CorpusId]]] = {}

    for facet in facets:
        seq_map = index.code_to_papers.get(facet, {})
        seqs = sorted(seq_map.keys())
        token_seqs[facet] = seqs
        # Decode each token sequence to text once.
        decoded = [decode_tokens(tokeniser, list(seq)).strip() for seq in seqs]
        texts[facet] = decoded
        paper_lists[facet] = [list(seq_map[seq]) for seq in seqs]

    embeddings: dict[str, npt.NDArray[np.float32]] = {}
    if with_embeddings:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(MINILM_MODEL_NAME)
        for facet in facets:
            facet_texts = texts[facet]
            if not facet_texts:
                embeddings[facet] = np.zeros((0, 384), dtype=np.float32)
                continue
            vecs = encoder.encode(
                facet_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            embeddings[facet] = vecs

    return DocIDIndex(
        facets=facets,
        token_seqs=token_seqs,
        texts=texts,
        paper_lists=paper_lists,
        embeddings=embeddings,
    )


def snap_levenshtein(
    decoded_toks: tuple[int, ...],
    facet: str,
    docid_index: DocIDIndex,
    *,
    top_k: int = 1,
) -> list[tuple[int, float, list[CorpusId]]]:
    """Return ``top_k`` nearest DocIDs by token-level Levenshtein distance.

    Each entry is ``(docid_index_position, similarity_score, paper_ids)``.
    Similarity is ``1 - edit_distance / max(len_a, len_b)`` so larger is better.

    Distances are computed with ``rapidfuzz.process.cdist`` so the inner loop
    runs in C against the whole facet's DocID list rather than a pure-Python
    loop per pair.
    """
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    seqs = docid_index.token_seqs.get(facet, [])
    if not seqs:
        return []
    distances = process.cdist([decoded_toks], seqs, scorer=Levenshtein.distance)[0]
    order = np.argsort(distances, kind="stable")[:top_k]
    out: list[tuple[int, float, list[CorpusId]]] = []
    for idx in order:
        i = int(idx)
        norm = max(len(decoded_toks), len(seqs[i]), 1)
        sim = 1.0 - float(distances[i]) / norm
        out.append((i, sim, docid_index.paper_lists[facet][i]))
    return out


def snap_cosine(
    decoded_text: str,
    facet: str,
    docid_index: DocIDIndex,
    encoder: object,
    *,
    top_k: int = 1,
) -> list[tuple[int, float, list[CorpusId]]]:
    """Return ``top_k`` nearest DocIDs by cosine similarity on MiniLM embeddings.

    ``encoder`` is a ``SentenceTransformer`` instance, passed in to avoid
    re-loading per call.
    """
    vecs = docid_index.embeddings.get(facet)
    if vecs is None or vecs.size == 0:
        return []
    from sentence_transformers import SentenceTransformer

    assert isinstance(encoder, SentenceTransformer)
    q = encoder.encode(
        [decoded_text],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)[0]
    sims = vecs @ q  # (N,)
    order = np.argsort(-sims)[:top_k]
    return [
        (int(idx), float(sims[idx]), docid_index.paper_lists[facet][int(idx)])
        for idx in order
    ]


__all__ = [
    "MINILM_MODEL_NAME",
    "DocIDIndex",
    "build_docid_index",
    "snap_cosine",
    "snap_levenshtein",
]
