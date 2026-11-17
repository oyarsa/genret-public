"""Signal computation for facet-based candidate graph generation.

Provides two supplementary signals used alongside dense cosine similarity:

- N-gram Jaccard similarity between paper facet text.
- Citation-based similarity (co-citation and bibliographic coupling combined).
"""

import itertools
import logging
import re
from collections.abc import ItemsView, KeysView
from pathlib import Path
from typing import Any, cast

import numpy as np

from genret.types import CorpusId
from genret.utils import iter_jsonl

logger = logging.getLogger(__name__)

type NgramSet = frozenset[str]
"""Unigram + bigram token set derived from one (paper, facet) pair."""


class CitationLookup:
    """Symmetric citation score map with automatic key normalisation.

    Keys are always stored as `(min_id, max_id)`, so lookup order does not
    matter: `lookup[a, b] == lookup[b, a]`.
    """

    def __init__(
        self, data: dict[tuple[CorpusId, CorpusId], float] | None = None
    ) -> None:
        """Initialise with an optional pre-built normalised-key dict.

        Args:
            data: Existing dict whose keys are already in `(min_id, max_id)` form.
                If omitted, starts empty.
        """
        self._data: dict[tuple[CorpusId, CorpusId], float] = (
            data if data is not None else {}
        )

    @staticmethod
    def _key(a: CorpusId, b: CorpusId) -> tuple[CorpusId, CorpusId]:
        """Return the canonical `(min, max)` key for a paper pair."""
        return (min(a, b), max(a, b))

    def __getitem__(self, key: tuple[CorpusId, CorpusId]) -> float:
        """Return the score for a pair; raises `KeyError` if not present."""
        return self._data[self._key(*key)]

    def __setitem__(self, key: tuple[CorpusId, CorpusId], value: float) -> None:
        """Set the score for a pair; key order does not matter."""
        self._data[self._key(*key)] = value

    def __len__(self) -> int:
        """Return the number of unique pairs stored."""
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        """Return True if the pair (in either order) is present."""
        if not isinstance(key, tuple):
            return False

        key = cast(tuple[Any, ...], key)
        if len(key) != 2:
            return False

        a, b = key
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("CitationLookup.contains takes tuple[CorpusId, CorpusId]")

        return self._key(CorpusId(a), CorpusId(b)) in self._data

    def get(self, a: CorpusId, b: CorpusId, default: float = 0.0) -> float:
        """Return the score for a pair, or `default` if not present.

        Args:
            a: First corpus ID.
            b: Second corpus ID.
            default: Value to return when the pair has no score. Defaults to 0.

        Returns:
            Stored score, or `default`.
        """
        return self._data.get(self._key(a, b), default)

    def keys(self) -> KeysView[tuple[CorpusId, CorpusId]]:
        """Return a view of all canonical `(min_id, max_id)` keys."""
        return self._data.keys()

    def items(self) -> ItemsView[tuple[CorpusId, CorpusId], float]:
        """Return a view of all `(canonical_key, score)` pairs."""
        return self._data.items()


_TOKENISE_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)


def _tokenise(text: str) -> list[str]:
    return _TOKENISE_RE.findall(text)


def facet_ngrams(items: list[str], stopwords: frozenset[str]) -> NgramSet:
    """Build a unigram + bigram token set from facet item texts.

    Tokenises each item with a regex, removes stopwords, then forms all
    consecutive bigrams from the remaining tokens.  Tokens are lowercased
    ASCII alphanumeric sequences, with hyphens and apostrophes preserved
    internally (e.g. `"fine-tuning"`, `"it's"`).

    Args:
        items: List of facet item strings for one paper.
        stopwords: Set of token strings to exclude from the output.

    Returns:
        Frozenset of unigram and bigram strings.
    """
    ngrams: set[str] = set()

    for item in items:
        tokens = [t for t in _tokenise(item) if t not in stopwords]
        ngrams.update(tokens)  # unigrams
        for token_a, token_b in itertools.pairwise(tokens):
            ngrams.add(f"{token_a} {token_b}")  # bigrams

    return frozenset(ngrams)


def jaccard(a: NgramSet, b: NgramSet) -> float:
    """Compute Jaccard similarity between two n-gram sets.

    Args:
        a: First n-gram set.
        b: Second n-gram set.

    Returns:
        `|a & b| / |a | b|`, or `0.0` if both sets are empty.
    """
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _load_citation_signal(
    path: Path, valid_ids: frozenset[CorpusId], value_field: str, label: str
) -> CitationLookup:
    """Load a citation signal JSONL and normalise counts to [0, 1] via log1p scaling.

    Only pairs where both papers are in `valid_ids` are kept.

    Args:
        path: Path to JSONL file (fields: `paper_a`, `paper_b`, `<value_field>`).
        valid_ids: Set of corpus IDs to restrict pairs to.
        value_field: Name of the numeric count field in each record.
        label: Short label used in the log message (e.g. `"Co-citation"`).

    Returns:
        `CitationLookup` mapping paper pairs to a normalised score in [0, 1].
    """
    raw = CitationLookup()
    max_val = 0.0

    for record in iter_jsonl(path):
        a = CorpusId(int(record["paper_a"]))
        b = CorpusId(int(record["paper_b"]))
        if a in valid_ids and b in valid_ids:
            val = float(record[value_field])
            raw[a, b] = val
            max_val = max(max_val, val)

    if max_val == 0:
        return CitationLookup()

    log_max = float(np.log1p(max_val))
    result = {key: float(np.log1p(v)) / log_max for key, v in raw.items()}
    logger.info(f"  {label}: {len(result)} valid pairs (max={max_val:.0f})")
    return CitationLookup(result)


def _load_co_citations(path: Path, valid_ids: frozenset[CorpusId]) -> CitationLookup:
    """Load co-citation counts and normalise to [0, 1] via log1p scaling.

    Args:
        path: Path to co_citations JSONL (fields: `paper_a`, `paper_b`, `count`).
        valid_ids: Set of corpus IDs to restrict pairs to.

    Returns:
        `CitationLookup` mapping paper pairs to a normalised score in [0, 1].
    """
    return _load_citation_signal(path, valid_ids, "count", "Co-citation")


def _load_bib_coupling(path: Path, valid_ids: frozenset[CorpusId]) -> CitationLookup:
    """Load bibliographic coupling counts and normalise to [0, 1] via log1p scaling.

    Args:
        path: Path to bib_coupling JSONL (fields: `paper_a`, `paper_b`, `shared_refs`).
        valid_ids: Set of corpus IDs to restrict pairs to.

    Returns:
        `CitationLookup` mapping paper pairs to a normalised score in [0, 1].
    """
    return _load_citation_signal(path, valid_ids, "shared_refs", "Bib-coupling")


def load_citation_lookup(
    co_cit_path: Path,
    bib_path: Path,
    valid_ids: frozenset[CorpusId],
) -> CitationLookup:
    """Combine co-citation and bib-coupling into a single citation score lookup.

    Each signal is loaded and log-normalised independently, then averaged for
    pairs that appear in both.

    Args:
        co_cit_path: Path to co_citations JSONL.
        bib_path: Path to bib_coupling JSONL.
        valid_ids: Set of corpus IDs to restrict pairs to.

    Returns:
        `CitationLookup` mapping paper pairs to a combined score in [0, 1].
    """
    co_cit = _load_co_citations(co_cit_path, valid_ids)
    bib = _load_bib_coupling(bib_path, valid_ids)

    all_keys = co_cit.keys() | bib.keys()
    combined: dict[tuple[CorpusId, CorpusId], float] = {}
    for key in all_keys:
        a, b = key
        scores: list[float] = []
        if key in co_cit:
            scores.append(co_cit.get(a, b))
        if key in bib:
            scores.append(bib.get(a, b))
        combined[key] = sum(scores) / len(scores)

    lookup = CitationLookup(combined)
    logger.info(f"  Combined citation lookup: {len(lookup)} pairs")
    return lookup
