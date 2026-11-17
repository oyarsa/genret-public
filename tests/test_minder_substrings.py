"""Tests for the MINDER-style suffix-array substring index.

Every index primitive is checked against a brute-force implementation over the
same token stream: that is the validation gate the plan requires before the
index is allowed to drive constrained decoding
(`docs/plans/20260725-minder-baseline.md` §4.2.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from genret.training.minder_substrings import (
    SubstringIndex,
    build_substring_index,
    build_suffix_array,
    clean_abstract,
)
from genret.types import CorpusId

# -- Helpers --

VOCAB = {
    word: index
    for index, word in enumerate(
        [
            "we",
            "propose",
            "a",
            "novel",
            "method",
            "for",
            "parsing",
            "dependency",
            "trees",
            "using",
            "neural",
            "networks",
            "and",
            "evaluate",
            "it",
            "on",
            "treebanks",
        ]
    )
}
"""Toy word-level vocabulary, so token streams stay readable in failures."""


def encode(text: str) -> list[int]:
    """Encode whitespace-separated words with the toy vocabulary."""
    return [VOCAB[word] for word in text.split() if word in VOCAB]


def brute_force_suffix_array(tokens: list[int]) -> list[int]:
    """Sort suffix positions by comparing the suffixes directly."""
    return sorted(range(len(tokens)), key=lambda i: tokens[i:])


def brute_force_occurrences(tokens: list[int], span: tuple[int, ...]) -> list[int]:
    """List every start position at which `span` occurs in `tokens`."""
    width = len(span)
    return [
        i
        for i in range(len(tokens) - width + 1)
        if tuple(tokens[i : i + width]) == span
    ]


def brute_force_successors(tokens: list[int], span: tuple[int, ...]) -> set[int]:
    """Collect tokens following each occurrence of `span`."""
    return {
        tokens[i + len(span)]
        for i in brute_force_occurrences(tokens, span)
        if i + len(span) < len(tokens)
    }


@pytest.fixture
def corpus() -> dict[CorpusId, str]:
    """Three short abstracts with deliberate overlap between them."""
    return {
        CorpusId(1): "we propose a novel method for parsing dependency trees",
        CorpusId(2): "we propose a neural method for parsing treebanks",
        CorpusId(3): "using neural networks and evaluate it on treebanks",
    }


@pytest.fixture
def index(corpus: dict[CorpusId, str]) -> SubstringIndex:
    """Index over the toy corpus, with a short minimum span for testing."""
    return build_substring_index(
        abstracts=corpus,
        encode=encode,
        vocab_size=len(VOCAB),
        min_span_tokens=2,
        max_span_tokens=6,
    )


def stream_of(index: SubstringIndex) -> list[int]:
    """Recover the unpadded token stream, sentinels included."""
    return [int(t) for t in index.tokens[: index.n_tokens]]


# -- Suffix array --


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_suffix_array_matches_brute_force(seed: int) -> None:
    """Prefix doubling reproduces a directly sorted suffix list."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 5, size=200, dtype=np.int64)
    built = build_suffix_array(tokens)
    assert list(built) == brute_force_suffix_array([int(t) for t in tokens])


def test_suffix_array_handles_uniform_and_empty_streams() -> None:
    """Degenerate streams do not break the doubling loop."""
    assert list(build_suffix_array(np.zeros(0, dtype=np.int64))) == []
    uniform = np.zeros(8, dtype=np.int64)
    assert list(build_suffix_array(uniform)) == brute_force_suffix_array([0] * 8)


# -- Index primitives against brute force --


def test_count_and_positions_match_brute_force(index: SubstringIndex) -> None:
    """Every span up to length 4 has the brute-force occurrence set."""
    stream = stream_of(index)
    for width in range(1, 5):
        for start in range(len(stream) - width + 1):
            span = tuple(stream[start : start + width])
            expected = brute_force_occurrences(stream, span)
            assert index.count(span) == len(expected), span
            assert list(index.positions(span)) == expected, span


def test_valid_next_matches_brute_force_minus_sentinels(
    index: SubstringIndex,
) -> None:
    """Successors match brute force once boundary sentinels are removed."""
    stream = stream_of(index)
    for width in range(1, 4):
        for start in range(len(stream) - width + 1):
            span = tuple(stream[start : start + width])
            expected = {
                token
                for token in brute_force_successors(stream, span)
                if token < index.vocab_size
            }
            assert set(index.valid_next(span)) == expected, span


def test_valid_next_is_sorted_and_within_vocabulary(index: SubstringIndex) -> None:
    """The decoder receives ascending, in-vocabulary token ids."""
    successors = index.valid_next((VOCAB["we"], VOCAB["propose"]))
    assert list(successors) == sorted(successors)
    assert all(0 <= token < index.vocab_size for token in successors)


def test_absent_span_has_no_range_no_successors_no_papers(
    index: SubstringIndex,
) -> None:
    """A span that never occurs is empty on every primitive."""
    absent = (VOCAB["treebanks"], VOCAB["we"], VOCAB["novel"])
    assert index.count(absent) == 0
    assert index.valid_next(absent) == ()
    assert index.papers_at(absent) == []


def test_cached_and_uncached_ranges_agree(index: SubstringIndex) -> None:
    """The parent-range shortcut gives the same answer as a cold lookup."""
    span = (VOCAB["propose"], VOCAB["a"], VOCAB["novel"])
    warm = index.range_for(span)
    index._range_cache.clear()
    assert index.range_for(span) == warm


# -- Paper attribution --


def test_papers_at_finds_shared_and_unique_spans(index: SubstringIndex) -> None:
    """Attribution matches which abstracts actually contain the span."""
    shared = (VOCAB["we"], VOCAB["propose"], VOCAB["a"])
    assert index.papers_at(shared) == [CorpusId(1), CorpusId(2)]

    unique = (VOCAB["dependency"], VOCAB["trees"])
    assert index.papers_at(unique) == [CorpusId(1)]

    third = (VOCAB["neural"], VOCAB["networks"])
    assert index.papers_at(third) == [CorpusId(3)]


def test_papers_at_matches_brute_force_per_document(
    index: SubstringIndex, corpus: dict[CorpusId, str]
) -> None:
    """Every span of length 2-3 is attributed exactly as a direct search would.

    Spans containing a boundary sentinel are skipped: they exist in the stream
    but `valid_next` never offers a sentinel, so the decoder cannot reach them.
    """
    per_paper = {cid: encode(text) for cid, text in corpus.items()}
    stream = stream_of(index)
    for width in (2, 3):
        for start in range(len(stream) - width + 1):
            span = tuple(stream[start : start + width])
            if any(token >= index.vocab_size for token in span):
                continue
            expected = [
                cid
                for cid, tokens in per_paper.items()
                if brute_force_occurrences(tokens, span)
            ]
            assert index.papers_at(span) == expected, span


def test_spans_cannot_cross_paper_boundaries(index: SubstringIndex) -> None:
    """A span joining the end of one abstract to the start of the next is absent."""
    first_tail = (VOCAB["dependency"], VOCAB["trees"])
    second_head = (VOCAB["we"], VOCAB["propose"])
    crossing = first_tail + second_head
    assert index.count(crossing) == 0
    assert index.valid_next(first_tail) == ()


# -- Stopping rule --


def test_is_stoppable_respects_minimum_span(index: SubstringIndex) -> None:
    """End-of-sequence is allowed only once the span is long enough and real."""
    assert not index.is_stoppable((VOCAB["we"],))
    assert index.is_stoppable((VOCAB["we"], VOCAB["propose"]))
    assert not index.is_stoppable((VOCAB["treebanks"], VOCAB["we"]))


# -- Abstract cleaning --


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain text", "plain text"),
        (r"we cite \cite{smith2020} here", "we cite here"),
        (r"see \citep[p. 4]{a,b} and \ref{fig:1}", "see and"),
        (r"the value $x_i = \alpha$ matters", "the value matters"),
        (r"display $$\sum_i x_i$$ removed", "display removed"),
        (r"\emph{parsing} keeps its argument", "parsing keeps its argument"),
        ("collapses    all   whitespace", "collapses all whitespace"),
        (r"  \textbf{bold} and trailing  ", "bold and trailing"),
    ],
)
def test_clean_abstract(raw: str, expected: str) -> None:
    """LaTeX commands and maths are stripped, braced arguments survive."""
    assert clean_abstract(raw) == expected
