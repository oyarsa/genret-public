"""Suffix-array substring index for the MINDER-style substring view.

MINDER (Li et al., ACL 2023) gives every passage three views of identifiers:
its title, arbitrary substrings of its body, and pseudo-queries. The substring
view's identifier set is *every contiguous token span occurring in the corpus*,
which cannot be enumerated into a trie, so the published system stores the
corpus in an FM-index and constrains generation with it.

An FM-index is used for exactly three primitives, and a suffix array answers
all three:

1. Which tokens may follow a generated span (logit masking during decoding).
2. Which papers contain a span, and at which positions (mapping an identifier
   back to papers, and the paper's rule that overlapping substrings count once).
3. How often a span occurs (the background-frequency term in the scoring).

Both structures rest on the same fact: sort every suffix of the token stream
and any substring occupies one contiguous range of that order. The FM-index is
the compressed form; at our scale (1,000 abstracts, ~230k tokens) a plain
suffix array is about 1 MB, so the compression buys nothing and the C++
dependency is avoidable. See `docs/plans/20260725-minder-baseline.md` §4.2.1.

Each paper's cleaned abstract is separated from the next by a unique sentinel
token above the tokeniser's vocabulary, so no span can straddle two papers and
no sentinel can ever be generated.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from genret.types import CorpusId

logger = logging.getLogger(__name__)

PAD_TOKEN = -1
"""Sentinel written past the end of the stream so `sa[i] + j` is always valid.

Sorts below every real token, which matches the convention that a shorter
suffix precedes a longer one sharing its prefix.
"""

_LATEX_DROP_ARG_RE = re.compile(
    r"\\(?:cite[a-zA-Z]*|ref|eqref|label|url|href|includegraphics|bibliography"
    r"|footnote)\s*(?:\[[^\]]*\])*\s*(?:\{[^{}]*\})+"
)
"""A LaTeX command whose braced argument is a key or path, not prose.

Dropped whole: a citation key like `smith2020` is a generatable span that no
query would match, so keeping it only wastes beam mass.
"""
_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z@]+\s*(\*)?(\[[^\]]*\])*")
"""A LaTeX control word with optional star and bracketed options.

Only the command is removed; its braced argument is prose and is kept, so
`\\emph{parsing}` contributes `parsing`.
"""
_LATEX_MATH_RE = re.compile(r"\$\$?.*?\$\$?", flags=re.DOTALL)
"""Inline or display maths delimited by dollars."""
_LATEX_BRACES_RE = re.compile(r"[{}]")
"""Grouping braces left behind once commands are stripped."""
_WHITESPACE_RE = re.compile(r"\s+")
"""Any whitespace run, collapsed to a single space."""


def clean_abstract(text: str) -> str:
    r"""Strip LaTeX commands and maths from an abstract.

    S2ORC abstracts carry `\cite{...}`, `$...$` and similar. Left in place they
    become generatable spans, so beam mass is spent on identifiers that no
    query would ever match.

    Maths is removed rather than unwrapped because its contents are symbols.
    Formatting commands are removed but their braced arguments kept, so
    `\emph{parsing}` contributes `parsing`; citation-like commands are removed
    together with their arguments, which are keys rather than prose.

    Args:
        text: Raw abstract text.

    Returns:
        Cleaned text with runs of whitespace collapsed.
    """
    without_maths = _LATEX_MATH_RE.sub(" ", text)
    without_keys = _LATEX_DROP_ARG_RE.sub(" ", without_maths)
    without_commands = _LATEX_COMMAND_RE.sub(" ", without_keys)
    without_braces = _LATEX_BRACES_RE.sub(" ", without_commands)
    return _WHITESPACE_RE.sub(" ", without_braces).strip()


def build_suffix_array(tokens: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Sort every suffix of `tokens` by prefix doubling.

    O(n log^2 n) with numpy doing the sorting, which is a few seconds at 9M
    tokens and immediate at the ~230k of the 1k abstract corpus -- well inside
    the budget where a compressed index would be worth its build complexity.

    Args:
        tokens: One-dimensional token stream.

    Returns:
        Positions of the suffixes in lexicographic order.
    """
    n = int(tokens.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    _, rank = np.unique(tokens, return_inverse=True)
    rank = rank.astype(np.int64)
    order = np.argsort(rank, kind="stable")
    step = 1
    while True:
        shifted = np.full(n, -1, dtype=np.int64)
        source = np.arange(n, dtype=np.int64) + step
        in_range = source < n
        shifted[in_range] = rank[source[in_range]]
        order = np.lexsort((shifted, rank))
        primary = rank[order]
        secondary = shifted[order]
        changed = (primary[1:] != primary[:-1]) | (secondary[1:] != secondary[:-1])
        ranked = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(changed)])
        rank_next = np.empty(n, dtype=np.int64)
        rank_next[order] = ranked
        rank = rank_next
        if int(rank.max()) == n - 1:
            return order
        step *= 2


@dataclass
class SubstringIndex:
    """Suffix array over a corpus token stream, with per-paper attribution.

    Not frozen: the successor caches are populated as decoding proceeds.
    """

    tokens: npt.NDArray[np.int64]
    """Token stream, padded with `PAD_TOKEN` so `sa[i] + j` never overruns."""
    suffixes: npt.NDArray[np.int64]
    """Suffix array over the unpadded stream."""
    doc_starts: npt.NDArray[np.int64]
    """Start offset of each paper's tokens, ascending."""
    corpus_ids: tuple[CorpusId, ...]
    """Paper at each offset in `doc_starts`, same order."""
    vocab_size: int
    """Tokeniser vocabulary size; ids at or above it are boundary sentinels."""
    min_span_tokens: int = 8
    """Shortest span the decoder may stop at.

    MINDER decodes body spans in a fixed length band; the band is ours to pick
    and is the lever on how discriminative a span is. Measured on the 1k
    abstract stream (199,708 tokens), mean occurrences and the share of spans
    appearing in more than one paper fall away sharply with length:

    | tokens | mean occurrences | max | share in >1 paper |
    | ------ | ---------------- | --- | ----------------- |
    | 1      | 1356.2           | 7850 | 0.99             |
    | 2      | 61.4             | 1409 | 0.68             |
    | 3      | 9.7              | 379  | 0.35             |
    | 4      | 3.5              | 237  | 0.12             |
    | 6      | 1.5              | 177  | 0.03             |
    | 8      | 1.0              | 3    | 0.01             |
    | 12     | 1.0              | 1    | 0.00             |

    Eight is the shortest length at which spans are effectively unique, and it
    sits closest to SEAL's ~10-token body n-grams; six still carries a heavy
    boilerplate tail.
    """
    max_span_tokens: int = 12
    """Longest span the decoder may generate, matched to our DocID cap."""
    _range_cache: dict[tuple[int, ...], tuple[int, int]] = field(
        default_factory=dict[tuple[int, ...], tuple[int, int]]
    )
    """Memo of span -> suffix-array range, shared across beams and queries."""
    _successor_cache: dict[tuple[int, ...], tuple[int, ...]] = field(
        default_factory=dict[tuple[int, ...], tuple[int, ...]]
    )
    """Memo of span -> legal next tokens."""

    @property
    def n_tokens(self) -> int:
        """Length of the unpadded token stream."""
        return int(self.suffixes.shape[0])

    def range_for(self, span: tuple[int, ...]) -> tuple[int, int]:
        """Find the half-open suffix-array range whose suffixes start with `span`.

        Narrows one token at a time. Within a range whose suffixes already share
        `j` tokens, the `j`-th tokens are non-decreasing, so each step is a
        binary search rather than a scan.

        Args:
            span: Token sequence to locate.

        Returns:
            `(lo, hi)`, empty when `lo == hi`.
        """
        if (cached := self._range_cache.get(span)) is not None:
            return cached
        # Reuse the parent range when it is already known: decoding always
        # extends a span by one token, so the parent is almost always cached.
        if len(span) > 1 and (parent := self._range_cache.get(span[:-1])) is not None:
            lo, hi = parent
            depth = len(span) - 1
        else:
            lo, hi = 0, self.n_tokens
            depth = 0
        for j in range(depth, len(span)):
            if lo >= hi:
                break
            column = self.tokens[self.suffixes[lo:hi] + j]
            left = int(np.searchsorted(column, span[j], side="left"))
            right = int(np.searchsorted(column, span[j], side="right"))
            lo, hi = lo + left, lo + right
        result = (lo, max(lo, hi))
        self._range_cache[span] = result
        return result

    def count(self, span: tuple[int, ...]) -> int:
        """Count occurrences of `span` in the corpus stream."""
        lo, hi = self.range_for(span)
        return hi - lo

    def valid_next(self, span: tuple[int, ...]) -> tuple[int, ...]:
        """List tokens that may legally follow `span`.

        Boundary sentinels and the end-of-stream pad are excluded, so a
        generated span can never cross from one paper into another.

        Args:
            span: Token sequence generated so far; empty for the first step.

        Returns:
            Legal next token ids, ascending.
        """
        if (cached := self._successor_cache.get(span)) is not None:
            return cached
        lo, hi = self.range_for(span)
        if lo >= hi:
            self._successor_cache[span] = ()
            return ()
        column = self.tokens[self.suffixes[lo:hi] + len(span)]
        candidates = np.unique(column)
        allowed = candidates[(candidates >= 0) & (candidates < self.vocab_size)]
        result = tuple(int(t) for t in allowed)
        self._successor_cache[span] = result
        return result

    def positions(self, span: tuple[int, ...]) -> npt.NDArray[np.int64]:
        """Return every start position of `span` in the token stream."""
        lo, hi = self.range_for(span)
        return np.sort(self.suffixes[lo:hi])

    def papers_at(self, span: tuple[int, ...]) -> list[CorpusId]:
        """List the papers whose abstract contains `span`, in corpus order.

        Assumes `span` is sentinel-free, which `valid_next` guarantees for
        anything the decoder can produce. A span containing a boundary sentinel
        would be attributed to the paper it ends, which is meaningless.

        Args:
            span: Token sequence to attribute.

        Returns:
            Distinct papers containing the span; empty if it does not occur.
        """
        found = self.positions(span)
        if found.shape[0] == 0:
            return []
        doc_index = np.searchsorted(self.doc_starts, found, side="right") - 1
        return [self.corpus_ids[int(i)] for i in np.unique(doc_index)]

    def is_stoppable(self, span: tuple[int, ...]) -> bool:
        """Report whether the decoder may emit end-of-sequence after `span`."""
        return len(span) >= self.min_span_tokens and self.count(span) > 0


def build_substring_index(
    *,
    abstracts: Mapping[CorpusId, str],
    encode: Callable[[str], Sequence[int]],
    vocab_size: int,
    min_span_tokens: int = 8,
    max_span_tokens: int = 12,
) -> SubstringIndex:
    """Build a `SubstringIndex` over cleaned abstracts.

    Args:
        abstracts: Raw abstract text per paper; iteration order fixes the
            stream layout and therefore the index's paper ordering.
        encode: Maps text to token ids, typically `tokeniser.encode` with
            special tokens disabled.
        vocab_size: Tokeniser vocabulary size; sentinels are allocated above it.
        min_span_tokens: Shortest stoppable span.
        max_span_tokens: Longest generatable span.

    Returns:
        The built index.
    """
    stream = list[int]()
    starts = list[int]()
    ids = list[CorpusId]()
    for offset, (corpus_id, raw) in enumerate(abstracts.items()):
        cleaned = clean_abstract(raw)
        if not cleaned:
            logger.warning("Paper %d has an empty abstract after cleaning", corpus_id)
        piece = list(encode(cleaned))
        starts.append(len(stream))
        ids.append(corpus_id)
        stream.extend(piece)
        # One unique sentinel per boundary: unique so it cannot be part of any
        # repeated span, above the vocabulary so it can never be generated.
        stream.append(vocab_size + offset)
    unpadded = np.asarray(stream, dtype=np.int64)
    suffixes = build_suffix_array(unpadded)
    padded = np.concatenate(
        [unpadded, np.full(max_span_tokens + 1, PAD_TOKEN, dtype=np.int64)]
    )
    logger.info(
        "Substring index: %d papers, %d tokens, %d suffixes",
        len(ids),
        unpadded.shape[0],
        suffixes.shape[0],
    )
    return SubstringIndex(
        tokens=padded,
        suffixes=suffixes,
        doc_starts=np.asarray(starts, dtype=np.int64),
        corpus_ids=tuple(ids),
        vocab_size=vocab_size,
        min_span_tokens=min_span_tokens,
        max_span_tokens=max_span_tokens,
    )
