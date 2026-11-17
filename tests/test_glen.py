"""Tests for GLEN DocID extraction, trie indexing, dataset, and decoding."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from genret.paper_parsing import Facets
from genret.training.glen_dataset import _target_text
from genret.training.glen_decoding import make_glen_prefix_fn
from genret.training.glen_docids import (
    GlenIndex,
    _build_trie,
    build_glen_index,
    extract_docid_tokens,
    load_glen_index,
    normalise_facet_text,
    save_glen_index,
    split_bullets,
)
from genret.types import CorpusId

# ---------------------------------------------------------------------------
# Mock tokeniser
# ---------------------------------------------------------------------------


class _MockTokeniser(PreTrainedTokenizerBase):
    """Character-level tokeniser for testing (no model required)."""

    def encode(  # type: ignore[override]
        self,
        text: str,
        *args: object,
        add_special_tokens: bool = True,
        **kwargs: object,
    ) -> list[int]:
        """Map each character to its Unicode code point."""
        return [ord(c) for c in text]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _facets_record(
    corpusid: int,
    *,
    problems: list[str] | None = None,
    methods: list[str] | None = None,
    results: list[str] | None = None,
    contributions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "corpusid": corpusid,
        "problems": problems or [],
        "methods": methods or [],
        "results": results or [],
        "contributions": contributions or [],
    }


def _write_facets(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _write_pairs(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# TestNormaliseFacetText
# ---------------------------------------------------------------------------


class TestNormaliseFacetText:
    """Tests for normalise_facet_text."""

    def test_strips_bullet_markers(self) -> None:
        """Should strip leading -, *, >, ' and whitespace."""
        assert normalise_facet_text("- hello world") == "hello world"
        assert normalise_facet_text("* hello world") == "hello world"
        assert normalise_facet_text("> hello world") == "hello world"

    def test_lowercases(self) -> None:
        """Should lowercase all text."""
        assert normalise_facet_text("- Hello World") == "hello world"

    def test_joins_multiline(self) -> None:
        """Should join multiple lines with a space."""
        text = "- line one\n- line two"
        assert normalise_facet_text(text) == "line one line two"

    def test_skips_empty_lines(self) -> None:
        """Should ignore blank lines."""
        text = "- line one\n\n- line two"
        assert normalise_facet_text(text) == "line one line two"

    def test_empty_input(self) -> None:
        """Should return empty string for empty input."""
        assert normalise_facet_text("") == ""

    def test_collapses_whitespace(self) -> None:
        """Should strip extra internal whitespace per line."""
        assert normalise_facet_text("-   spaced   out") == "spaced   out"


# ---------------------------------------------------------------------------
# TestSplitBullets
# ---------------------------------------------------------------------------


class TestSplitBullets:
    """Tests for split_bullets."""

    def test_basic_split(self) -> None:
        """Should return one entry per non-empty line."""
        text = "- line one\n- line two\n- line three"
        assert split_bullets(text) == ["line one", "line two", "line three"]

    def test_strips_markers(self) -> None:
        """Should strip bullet markers from each line."""
        assert split_bullets("* bullet") == ["bullet"]
        assert split_bullets("> bullet") == ["bullet"]
        assert split_bullets("- bullet") == ["bullet"]

    def test_preserves_capitalisation(self) -> None:
        """Should not lowercase — that is normalise_facet_text's job."""
        assert split_bullets("- Hello World") == ["Hello World"]

    def test_skips_blank_lines(self) -> None:
        """Should skip blank and whitespace-only lines."""
        text = "- a\n\n- b"
        assert split_bullets(text) == ["a", "b"]

    def test_empty_input(self) -> None:
        """Should return empty list for empty string."""
        assert split_bullets("") == []

    def test_single_bullet(self) -> None:
        """Should return a single-element list for one bullet."""
        assert split_bullets("- only one") == ["only one"]


# ---------------------------------------------------------------------------
# TestExtractDocidTokens
# ---------------------------------------------------------------------------


class TestExtractDocidTokens:
    """Tests for extract_docid_tokens."""

    def test_returns_token_list(self) -> None:
        """Should return a non-empty list for non-empty text."""
        tok = _MockTokeniser()
        result = extract_docid_tokens("hello world", tok)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_respects_max_tokens(self) -> None:
        """Should cap output at max_tokens."""
        tok = _MockTokeniser()
        result = extract_docid_tokens("hello world", tok, max_tokens=3)
        assert len(result) <= 3

    def test_empty_text_returns_empty(self) -> None:
        """Should return [] for text that normalises to empty."""
        tok = _MockTokeniser()
        assert extract_docid_tokens("", tok) == []
        assert extract_docid_tokens("- \n- ", tok) == []

    def test_leading_space_applied(self) -> None:
        """Encoding should include a leading space (BPE context hack)."""
        tok = _MockTokeniser()
        # With char-level mock: ord(' ') == 32 should be first token
        result = extract_docid_tokens("hello", tok, max_tokens=100)
        assert result[0] == ord(" ")


# ---------------------------------------------------------------------------
# TestBuildTrie
# ---------------------------------------------------------------------------


class TestBuildTrie:
    """Tests for the trie construction helper."""

    def test_single_sequence(self) -> None:
        """Trie should have a path for the sequence."""
        trie = _build_trie([(1, 2, 3)])
        assert 1 in trie
        assert 2 in trie[1]
        assert 3 in trie[1][2]

    def test_shared_prefix(self) -> None:
        """Two sequences sharing a prefix share the same trie nodes."""
        trie = _build_trie([(1, 2, 3), (1, 2, 4)])
        assert set(trie[1][2].keys()) == {3, 4}

    def test_empty_input(self) -> None:
        """Empty input should produce an empty trie."""
        assert _build_trie([]) == {}


# ---------------------------------------------------------------------------
# TestGlenIndex
# ---------------------------------------------------------------------------


def _make_index() -> GlenIndex:
    """Minimal GlenIndex with two papers in the 'methods' facet."""
    code_to_papers: dict[str, dict[tuple[int, ...], list[CorpusId]]] = {
        "problems": {},
        "methods": {
            (1, 2, 3): [CorpusId(10)],
            (1, 2, 4): [CorpusId(20)],
            (5, 6): [CorpusId(10), CorpusId(30)],
        },
        "results": {},
        "contributions": {},
    }
    trie = {
        "problems": {},
        "methods": _build_trie([(1, 2, 3), (1, 2, 4), (5, 6)]),
        "results": {},
        "contributions": {},
    }
    return GlenIndex(code_to_papers=code_to_papers, trie=trie)


class TestGlenIndex:
    """Tests for GlenIndex methods."""

    def test_valid_next_from_root(self) -> None:
        """From empty prefix, valid next tokens are roots of the trie."""
        index = _make_index()
        assert set(index.valid_next("methods", ())) == {1, 5}

    def test_valid_next_mid_sequence(self) -> None:
        """After (1, 2), valid next tokens are 3 and 4."""
        index = _make_index()
        assert set(index.valid_next("methods", (1, 2))) == {3, 4}

    def test_valid_next_beyond_trie(self) -> None:
        """A prefix not in the trie returns an empty list."""
        index = _make_index()
        assert index.valid_next("methods", (9, 9)) == []

    def test_is_leaf_true(self) -> None:
        """A complete DocID should be recognised as a leaf."""
        index = _make_index()
        assert index.is_leaf("methods", (1, 2, 3))
        assert index.is_leaf("methods", (5, 6))

    def test_is_leaf_false_for_prefix(self) -> None:
        """An incomplete prefix should not be a leaf."""
        index = _make_index()
        assert not index.is_leaf("methods", (1, 2))

    def test_is_leaf_false_for_unknown(self) -> None:
        """An unknown sequence should not be a leaf."""
        index = _make_index()
        assert not index.is_leaf("methods", (9, 9, 9))

    def test_papers_at_leaf(self) -> None:
        """Should return the correct paper list at a leaf."""
        index = _make_index()
        assert index.papers_at("methods", (1, 2, 3)) == [CorpusId(10)]
        assert set(index.papers_at("methods", (5, 6))) == {CorpusId(10), CorpusId(30)}

    def test_papers_at_non_leaf(self) -> None:
        """Should return empty list for non-leaf sequences."""
        index = _make_index()
        assert index.papers_at("methods", (1,)) == []

    def test_valid_next_empty_facet(self) -> None:
        """Unknown facet should return empty list gracefully."""
        index = _make_index()
        assert index.valid_next("problems", ()) == []


# ---------------------------------------------------------------------------
# TestBuildGlenIndex
# ---------------------------------------------------------------------------


class TestBuildGlenIndex:
    """Tests for build_glen_index with mock data."""

    def test_per_bullet_entries(self, tmp_path: Path) -> None:
        """Each bullet should produce its own trie leaf."""
        facets_path = tmp_path / "facets.jsonl"
        _write_facets(
            facets_path,
            [
                _facets_record(1, methods=["alpha beta", "gamma delta"]),
            ],
        )
        tok = _MockTokeniser()
        index = build_glen_index(facets_path, tok, max_tokens=4)
        # Two distinct bullets → two leaves in methods
        assert len(index.code_to_papers["methods"]) == 2

    def test_same_paper_multiple_leaves(self, tmp_path: Path) -> None:
        """Same paper appears in all its bullet leaves."""
        facets_path = tmp_path / "facets.jsonl"
        _write_facets(
            facets_path,
            [_facets_record(42, problems=["bullet one", "bullet two"])],
        )
        tok = _MockTokeniser()
        index = build_glen_index(facets_path, tok, max_tokens=10)
        leaves = list(index.code_to_papers["problems"].values())
        assert all(CorpusId(42) in leaf for leaf in leaves)

    def test_empty_facet_skipped(self, tmp_path: Path) -> None:
        """Papers with an empty facet should have no leaves for that facet."""
        facets_path = tmp_path / "facets.jsonl"
        _write_facets(facets_path, [_facets_record(1, methods=[])])
        tok = _MockTokeniser()
        index = build_glen_index(facets_path, tok, max_tokens=10)
        assert len(index.code_to_papers["methods"]) == 0

    def test_trie_consistent_with_leaves(self, tmp_path: Path) -> None:
        """Every leaf key should be reachable via the trie."""
        facets_path = tmp_path / "facets.jsonl"
        _write_facets(
            facets_path,
            [_facets_record(1, methods=["alpha", "beta"])],
        )
        tok = _MockTokeniser()
        index = build_glen_index(facets_path, tok, max_tokens=6)
        for seq in index.code_to_papers["methods"]:
            assert index.is_leaf("methods", seq)


# ---------------------------------------------------------------------------
# TestSaveLoadGlenIndex
# ---------------------------------------------------------------------------


class TestSaveLoadGlenIndex:
    """Tests for save_glen_index / load_glen_index round-trip."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Saved and loaded index should match the original."""
        index = _make_index()
        save_glen_index(index, tmp_path)
        loaded = load_glen_index(tmp_path)

        assert set(loaded.code_to_papers.keys()) == set(index.code_to_papers.keys())
        assert loaded.code_to_papers["methods"] == index.code_to_papers["methods"]
        assert set(loaded.valid_next("methods", ())) == {1, 5}
        assert loaded.is_leaf("methods", (1, 2, 3))

    def test_files_created(self, tmp_path: Path) -> None:
        """Should create code_to_papers.json and trie.json."""
        save_glen_index(_make_index(), tmp_path)
        assert (tmp_path / "code_to_papers.json").exists()
        assert (tmp_path / "trie.json").exists()


# ---------------------------------------------------------------------------
# TestMakeGlenPrefixFn
# ---------------------------------------------------------------------------


class TestMakeGlenPrefixFn:
    """Tests for make_glen_prefix_fn (pure Python, no model needed)."""

    def test_returns_valid_next_at_root(self) -> None:
        """At position 0 the prefix_fn should return trie root tokens."""
        index = _make_index()
        eos_id = 999
        prompt_len = 2

        fn = make_glen_prefix_fn(
            "methods",
            index=index,
            eos_id=eos_id,
            prompt_len=prompt_len,
            max_new_tokens=10,
        )
        # Beam input: [prompt_tok, prompt_tok] (no generated tokens yet)
        beam = torch.tensor([0, 0])
        allowed = fn(0, beam)
        assert set(allowed) == {1, 5}

    def test_allows_eos_at_leaf(self) -> None:
        """EOS should be allowed when the current generated prefix is a leaf."""
        index = _make_index()
        eos_id = 999
        prompt_len = 1

        fn = make_glen_prefix_fn(
            "methods",
            index=index,
            eos_id=eos_id,
            prompt_len=prompt_len,
            max_new_tokens=10,
        )
        # Generated so far: (5, 6) — a leaf in methods
        beam = torch.tensor([0, 5, 6])
        allowed = fn(0, beam)
        assert eos_id in allowed

    def test_forces_eos_at_max_tokens(self) -> None:
        """When generated length >= max_new_tokens, only EOS is allowed."""
        index = _make_index()
        eos_id = 999
        prompt_len = 1

        fn = make_glen_prefix_fn(
            "methods",
            index=index,
            eos_id=eos_id,
            prompt_len=prompt_len,
            max_new_tokens=2,
        )
        # Generated 2 tokens already (= max_new_tokens)
        beam = torch.tensor([0, 1, 2])
        allowed = fn(0, beam)
        assert allowed == [eos_id]

    def test_forces_eos_off_trie(self) -> None:
        """A prefix not in the trie should force EOS."""
        index = _make_index()
        eos_id = 999
        prompt_len = 1

        fn = make_glen_prefix_fn(
            "methods",
            index=index,
            eos_id=eos_id,
            prompt_len=prompt_len,
            max_new_tokens=10,
        )
        beam = torch.tensor([0, 99, 99])  # 99 not in trie
        allowed = fn(0, beam)
        assert allowed == [eos_id]


# ---------------------------------------------------------------------------
# TestTargetText
# ---------------------------------------------------------------------------


class TestTargetText:
    """Tests for _target_text in glen_dataset."""

    def test_returns_one_bullet(self) -> None:
        """Should return the normalised text of one bullet, not all."""
        facets = Facets(
            corpusid=CorpusId(1),
            problems=["bullet one", "bullet two", "bullet three"],
            methods=[],
            results=[],
            contributions=[],
        )
        rng = random.Random(0)
        result = _target_text(facets, "problems", rng)
        # Result must be exactly one of the bullets (normalised)
        assert result in {"bullet one", "bullet two", "bullet three"}

    def test_empty_facet_returns_empty(self) -> None:
        """Should return empty string if facet has no bullets."""
        facets = Facets(
            corpusid=CorpusId(1),
            problems=[],
            methods=[],
            results=[],
            contributions=[],
        )
        rng = random.Random(0)
        assert _target_text(facets, "problems", rng) == ""

    def test_random_sampling(self) -> None:
        """Should not always return the same bullet."""
        facets = Facets(
            corpusid=CorpusId(1),
            problems=["alpha", "beta", "gamma", "delta"],
            methods=[],
            results=[],
            contributions=[],
        )
        results = {
            _target_text(facets, "problems", random.Random(seed)) for seed in range(20)
        }
        assert len(results) > 1
