# pyright: basic
"""GLEN-style per-facet DocID extraction and prefix trie.

Each paper's DocID for a given facet is the normalised text of its facet
bullets tokenised to a capped token-ID sequence.  A prefix trie built from
these sequences enables constrained beam search without a learned codebook.

Tokenisation note: DocID tokens are encoded with a leading space
(``" " + text``) so that BPE treats the first word as mid-sequence,
matching the decoding context the model sees at inference.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from genret.paper_parsing import FacetRecordType, facets_from_type
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_dataset import _facet_text
from genret.types import CorpusId
from genret.utils import iter_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 12
"""DocID length cap. Trades discriminability vs beam-search depth."""


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def normalise_facet_text(facet_text: str) -> str:
    """Strip bullet markers, lowercase, collapse whitespace."""
    lines: list[str] = []
    for raw_line in facet_text.splitlines():
        stripped = re.sub(r"^[\s\-*>\"\']+", "", raw_line).strip().lower()
        if stripped:
            lines.append(stripped)
    return " ".join(lines)


def split_bullets(text: str) -> list[str]:
    """Split multi-line facet text into individual bullet strings.

    Returns one entry per non-empty line, with leading bullet markers stripped
    but original capitalisation preserved.
    """
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = re.sub(r"^[\s\-*>\"\']+", "", raw_line).strip()
        if stripped:
            lines.append(stripped)
    return lines


def extract_docid_tokens(
    facet_text: str,
    tokeniser: PreTrainedTokenizerBase,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[int]:
    """Normalise facet bullets and return a capped token-ID sequence."""
    text = normalise_facet_text(facet_text)
    if not text:
        return []
    ids: list[int] = tokeniser.encode(" " + text, add_special_tokens=False)
    return ids[:max_tokens]


# ---------------------------------------------------------------------------
# Trie structure
# ---------------------------------------------------------------------------


def _build_trie(sequences: list[tuple[int, ...]]) -> dict:  # type: ignore[type-arg]
    """Build a nested-dict prefix trie from token-ID sequences."""
    root: dict = {}  # type: ignore[type-arg]
    for seq in sequences:
        node = root
        for tok in seq:
            node = node.setdefault(tok, {})
    return root


@dataclass
class GlenIndex:
    """Per-facet trie + sequence→paper lookup for constrained decoding."""

    code_to_papers: dict[str, dict[tuple[int, ...], list[CorpusId]]]
    """``facet -> {token_sequence -> [corpusids]}`` at populated leaves."""
    trie: dict[str, dict]  # type: ignore[type-arg]
    """``facet -> nested dict`` mapping token IDs to child nodes."""

    def valid_next(self, facet: str, prefix: tuple[int, ...]) -> list[int]:
        """Token IDs valid as the next step after ``prefix``."""
        node: dict = self.trie.get(facet, {})  # type: ignore[type-arg]
        for tok in prefix:
            node = node.get(tok)  # type: ignore[assignment]
            if node is None:
                return []
        return list(node.keys())

    def is_leaf(self, facet: str, sequence: tuple[int, ...]) -> bool:
        """True if ``sequence`` is a complete DocID with at least one paper."""
        return bool(self.code_to_papers.get(facet, {}).get(sequence))

    def papers_at(self, facet: str, sequence: tuple[int, ...]) -> list[CorpusId]:
        """Papers stored at ``sequence``; empty list if not a leaf."""
        return self.code_to_papers.get(facet, {}).get(sequence, [])

    def reachable_papers(
        self, facet: str
    ) -> dict[tuple[int, ...], frozenset[CorpusId]]:
        """For every trie prefix, the set of papers reachable from it.

        Used by paper-dedup-during-beam decoding: when every paper reachable
        from a beam's prefix is already in the found-set for this query,
        the beam contributes no new papers and can be killed.
        Computed lazily once per facet then cached on the instance.
        """
        cache_attr: dict[str, dict[tuple[int, ...], frozenset[CorpusId]]] | None = (
            getattr(self, "_reachable_cache", None)
        )
        if cache_attr is None:
            cache_attr = {}
            object.__setattr__(self, "_reachable_cache", cache_attr)
        cache = cache_attr
        if facet in cache:
            return cache[facet]
        leaves = self.code_to_papers.get(facet, {})
        out: dict[tuple[int, ...], frozenset[CorpusId]] = {}

        def walk(sub: dict, prefix: tuple[int, ...]) -> frozenset[CorpusId]:  # type: ignore[type-arg]
            accumulated: set[CorpusId] = set(leaves.get(prefix, []))
            for tok, child in sub.items():
                accumulated |= walk(child, (*prefix, tok))
            result = frozenset(accumulated)
            out[prefix] = result
            return result

        walk(self.trie.get(facet, {}), ())
        cache[facet] = out
        return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_glen_index(
    facets_path: Path,
    tokeniser: PreTrainedTokenizerBase,
    *,
    facets_type: FacetRecordType = FacetRecordType.STANDARD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    smoothed_bullets: Mapping[tuple[CorpusId, str, str], str] | None = None,
) -> GlenIndex:
    """Build per-facet tries from all papers in ``facets_path``.

    ``smoothed_bullets`` optionally rewrites each bullet's DocID text
    (graph-smoothed DocIDs): keyed by ``(corpusid, facet, normalised
    bullet)``; bullets absent from the map fall back to their normalised
    text. Must be the same map given to the training-data builder so trie
    entries and training targets agree.
    """
    raw: dict[str, dict[tuple[int, ...], list[CorpusId]]] = {f: {} for f in FACET_NAMES}

    n_papers = 0
    n_empty = 0
    for record in iter_jsonl(facets_path):
        facets_obj = facets_from_type(record, facets_type)
        cid = facets_obj.corpusid
        n_papers += 1
        for facet in FACET_NAMES:
            text = _facet_text(facets_obj, facet, None)
            bullets = split_bullets(text)
            if not bullets:
                n_empty += 1
                continue
            for bullet in bullets:
                docid_text = bullet
                if smoothed_bullets is not None:
                    norm = normalise_facet_text(bullet)
                    docid_text = smoothed_bullets.get((cid, facet, norm), norm)
                toks = extract_docid_tokens(
                    docid_text, tokeniser, max_tokens=max_tokens
                )
                if not toks:
                    n_empty += 1
                    continue
                raw[facet].setdefault(tuple(toks), []).append(cid)

    tries = {facet: _build_trie(list(raw[facet].keys())) for facet in FACET_NAMES}

    total_leaves = sum(len(v) for v in raw.values())
    logger.info(
        "GLEN index: %d papers, %d leaves across %d facets (%d empty facet slots)",
        n_papers,
        total_leaves,
        len(FACET_NAMES),
        n_empty,
    )
    return GlenIndex(code_to_papers=raw, trie=tries)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _trie_to_serial(node: dict) -> dict:  # type: ignore[type-arg]
    return {str(k): _trie_to_serial(v) for k, v in node.items()}


def _trie_from_serial(node: dict) -> dict:  # type: ignore[type-arg]
    return {int(k): _trie_from_serial(v) for k, v in node.items()}


def save_glen_index(index: GlenIndex, glen_dir: Path) -> None:
    """Serialise index to ``glen_dir/code_to_papers.json`` and ``trie.json``."""
    glen_dir.mkdir(parents=True, exist_ok=True)

    c2p: dict[str, dict[str, list[int]]] = {}
    for facet, seqmap in index.code_to_papers.items():
        c2p[facet] = {
            ",".join(map(str, seq)): [int(c) for c in papers]
            for seq, papers in seqmap.items()
        }
    (glen_dir / "code_to_papers.json").write_text(
        json.dumps(c2p, separators=(",", ":"))
    )

    trie_serial = {f: _trie_to_serial(t) for f, t in index.trie.items()}
    (glen_dir / "trie.json").write_text(json.dumps(trie_serial, separators=(",", ":")))
    logger.info("GLEN index saved to %s", glen_dir)


def load_glen_index(glen_dir: Path) -> GlenIndex:
    """Deserialise a ``GlenIndex`` from ``glen_dir``."""
    raw_c2p = json.loads((glen_dir / "code_to_papers.json").read_text())
    code_to_papers: dict[str, dict[tuple[int, ...], list[CorpusId]]] = {}
    for facet, seqmap in raw_c2p.items():
        code_to_papers[facet] = {
            tuple(int(t) for t in key.split(",")): [CorpusId(c) for c in papers]
            for key, papers in seqmap.items()
        }

    raw_trie = json.loads((glen_dir / "trie.json").read_text())
    trie = {f: _trie_from_serial(t) for f, t in raw_trie.items()}

    return GlenIndex(code_to_papers=code_to_papers, trie=trie)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "GlenIndex",
    "build_glen_index",
    "extract_docid_tokens",
    "load_glen_index",
    "normalise_facet_text",
    "save_glen_index",
    "split_bullets",
]
