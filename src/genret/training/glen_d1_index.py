# pyright: basic
"""GLEN + Leiden cluster-prefix DocIDs (the D1 variant).

Each (paper, facet, bullet) becomes a DocID of the form::

    <C2T-TFIDF label for paper's Leiden leaf in this facet> <bullet text>

The prefix commits the model to a topical region of the per-facet retrieval
graph; the bullet suffix preserves per-paper specificity (multiple DocIDs per
paper, like GLEN). No new tokenizer tokens are introduced — both halves are
encoded by the model's existing BPE.

Pipeline assumes an upstream C2T index already exists on a Leiden codebook
(typically built coarse, e.g. res_l0=1.0, res_l1=2.0). The C2T builder writes
``paper_c2t_docids.jsonl`` which gives us the per-(facet, paper) prefix string.

This module:

1. Loads the C2T prefix map.
2. Loads each paper's facet bullets.
3. For every (paper, facet, bullet) tuple, encodes ``"<prefix> <bullet>"`` as
   the DocID and inserts it into the per-facet trie.
4. Writes a :class:`GlenIndex` to ``output_dir`` (drop-in compatible with the
   existing GLEN evaluator and trainer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from genret.paper_parsing import FacetRecordType, facets_from_type
from genret.training.c2t_index import load_paper_c2t_docids
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_dataset import _facet_text
from genret.training.glen_docids import (
    DEFAULT_MAX_TOKENS,
    GlenIndex,
    _build_trie,
    normalise_facet_text,
    save_glen_index,
    split_bullets,
)
from genret.types import CorpusId
from genret.utils import iter_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class D1IndexParams:
    """Build-time params for the D1 prefix+bullet index."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    """DocID length cap (prefix + bullet combined)."""

    prefix_bullet_separator: str = " "
    """Inserted between the cluster prefix and the bullet text. A single
    space lets BPE merge naturally; an explicit token (e.g. ``" || "``)
    forces a hard boundary at the cost of one extra token."""


def _encode_docid(
    prefix: str,
    bullet_text: str,
    tokeniser: PreTrainedTokenizerBase,
    params: D1IndexParams,
) -> tuple[int, ...]:
    """Tokenise ``"<prefix><sep><bullet>"`` to a capped token-ID sequence."""
    bullet_norm = normalise_facet_text(bullet_text)
    if not bullet_norm:
        return ()
    docid_text = f"{prefix}{params.prefix_bullet_separator}{bullet_norm}"
    # Leading space matches the GLEN bullet tokenisation convention so that
    # BPE treats the start of the DocID as mid-sequence at decode time.
    ids: list[int] = tokeniser.encode(" " + docid_text, add_special_tokens=False)
    return tuple(ids[: params.max_tokens])


def build_d1_index(
    c2t_dir: Path,
    facets_path: Path,
    output_dir: Path,
    tokeniser: PreTrainedTokenizerBase,
    *,
    params: D1IndexParams | None = None,
    facets_type: FacetRecordType = FacetRecordType.STANDARD,
) -> GlenIndex:
    """Build a D1 prefix+bullet GLEN index.

    ``c2t_dir`` must contain ``paper_c2t_docids.jsonl`` (produced by
    ``build-c2t-index`` on a Leiden codebook). The strings in that file
    are used verbatim as prefixes.
    """
    if params is None:
        params = D1IndexParams()

    prefix_map = load_paper_c2t_docids(c2t_dir)
    # {facet: {token_seq: [corpusids...]}}
    raw: dict[str, dict[tuple[int, ...], list[CorpusId]]] = {f: {} for f in FACET_NAMES}

    n_papers = 0
    n_empty_bullets = 0
    n_missing_prefix = 0
    for record in iter_jsonl(facets_path):
        facets_obj = facets_from_type(record, facets_type)
        cid = facets_obj.corpusid
        n_papers += 1
        for facet in FACET_NAMES:
            facet_prefixes = prefix_map.get(facet, {})
            prefix = facet_prefixes.get(cid)
            if prefix is None:
                n_missing_prefix += 1
                continue
            bullets = split_bullets(_facet_text(facets_obj, facet, None))
            if not bullets:
                n_empty_bullets += 1
                continue
            for bullet in bullets:
                seq = _encode_docid(prefix, bullet, tokeniser, params)
                if not seq:
                    continue
                raw[facet].setdefault(seq, []).append(cid)

    tries = {facet: _build_trie(list(raw[facet].keys())) for facet in FACET_NAMES}
    total_leaves = sum(len(v) for v in raw.values())
    logger.info(
        "D1 index: %d papers, %d unique DocIDs across %d facets "
        "(%d facet slots with no bullets, %d with no prefix)",
        n_papers,
        total_leaves,
        len(FACET_NAMES),
        n_empty_bullets,
        n_missing_prefix,
    )

    index = GlenIndex(code_to_papers=raw, trie=tries)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_glen_index(index, output_dir)
    return index


__all__ = ["D1IndexParams", "build_d1_index"]
