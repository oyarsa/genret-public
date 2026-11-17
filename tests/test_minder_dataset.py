"""Tests for the MINDER-style three-view training dataset.

The invariants here are the ones that silently degrade training if broken: the
published mix ratio, the exposure guarantee that every paper is a target in
every view, and substring targets that re-tokenise to something the substring
index can actually generate.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from transformers import PreTrainedTokenizerBase

from genret.training.minder_dataset import (
    DEFAULT_MIX,
    VIEWS,
    MinderDatasetConfig,
    SpanCandidate,
    _sample_spans_fuzzy,
    _score_candidates_fuzzy,
    build_minder_examples,
    build_view_messages,
    enumerate_span_candidates,
    query_text,
    select_spans,
)
from genret.training.minder_substrings import SubstringIndex, build_substring_index
from genret.types import CorpusId

# -- Toy corpus and tokeniser --

WORDS = [
    "we",
    "propose",
    "a",
    "novel",
    "neural",
    "method",
    "for",
    "parsing",
    "dependency",
    "trees",
    "using",
    "networks",
    "and",
    "evaluate",
    "on",
    "treebanks",
    "with",
    "strong",
    "results",
    "across",
    "languages",
    "models",
]
"""Toy vocabulary; token ids are indices into this list."""


class ToyTokeniser:
    """Word-level tokeniser standing in for a real BPE tokeniser.

    Round-trips exactly, which lets the span-candidate tests assert on overlap
    and exposure rather than on tokenisation quirks.
    """

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(WORDS)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode whitespace-separated known words."""
        return [WORDS.index(w) for w in text.split() if w in WORDS]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode ids back to a space-joined string."""
        return " ".join(WORDS[i] for i in ids if i < len(WORDS))


ABSTRACTS = {
    CorpusId(1): "we propose a novel neural method for parsing dependency trees",
    CorpusId(2): "using networks and evaluate on treebanks with strong results",
    CorpusId(3): "we propose models for parsing across languages with strong results",
}
"""Three abstracts long enough to yield several 4-token spans each."""


@pytest.fixture
def tokeniser() -> PreTrainedTokenizerBase:
    """Toy tokeniser, cast to the real interface for the functions under test."""
    return cast(PreTrainedTokenizerBase, ToyTokeniser())


@pytest.fixture
def index(tokeniser: PreTrainedTokenizerBase) -> SubstringIndex:
    """Substring index over the toy abstracts."""
    return build_substring_index(
        abstracts=ABSTRACTS,
        encode=lambda t: tokeniser.encode(t, add_special_tokens=False),
        vocab_size=len(tokeniser),
        min_span_tokens=2,
        max_span_tokens=4,
    )


@pytest.fixture
def corpus_files(tmp_path: Path) -> Path:
    """Write pairs, facets, paper_graphs and abstracts JSONL to `tmp_path`."""
    (tmp_path / "pairs.jsonl").write_text(
        "\n".join(
            json.dumps({"paper_a": a, "paper_b": b, "facet": f, "kind": "retrieval"})
            # The same pair under every facet: MINDER has no facets, so the
            # builder must collapse these to one pair.
            for a, b in [(1, 2), (1, 3)]
            for f in ("problems", "methods", "results", "contributions")
        )
        + "\n"
    )
    (tmp_path / "facets.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "corpusid": cid,
                    "problems": [f"problem of paper {cid}"],
                    "methods": [f"method of paper {cid}"],
                    "results": [f"result of paper {cid}"],
                    "contributions": [f"contribution of paper {cid}"],
                }
            )
            for cid in (1, 2, 3)
        )
        + "\n"
    )
    (tmp_path / "graphs.jsonl").write_text(
        "\n".join(
            json.dumps({"corpusid": cid, "title": f"Title Of Paper {cid}"})
            for cid in (1, 2, 3)
        )
        + "\n"
    )
    (tmp_path / "abstracts.jsonl").write_text(
        "\n".join(
            json.dumps({"corpusid": cid, "abstract": text})
            for cid, text in ABSTRACTS.items()
        )
        + "\n"
    )
    return tmp_path


def make_config(
    root: Path,
    *,
    mix: Mapping[str, int] | None = None,
    unsupervised_blocks: int = 1,
) -> MinderDatasetConfig:
    """Build a config pointing at the fixture files."""
    return MinderDatasetConfig(
        pairs_path=root / "pairs.jsonl",
        facets_path=root / "facets.jsonl",
        paper_graphs_path=root / "graphs.jsonl",
        abstracts_path=root / "abstracts.jsonl",
        span_tokens=4,
        mix=dict(mix) if mix is not None else dict(DEFAULT_MIX),
        unsupervised_blocks=unsupervised_blocks,
    )


# -- Query rendering and prompts --


def test_query_text_includes_title_and_cleaned_abstract() -> None:
    """The query carries both fields, with LaTeX stripped from the abstract."""
    rendered = query_text("A Title", r"An \emph{abstract} with $x$ maths")
    assert "Title: A Title" in rendered
    assert "abstract with maths" in rendered
    assert "\\emph" not in rendered


def test_query_text_truncates_abstract() -> None:
    """`max_chars` bounds the abstract, not the title."""
    rendered = query_text("T", "abcdefghij", max_chars=4)
    assert rendered == "Title: T\nAbstract: abcd"


def test_query_text_survives_missing_abstract() -> None:
    """A paper with no abstract still yields a usable query."""
    assert query_text("Only A Title", "") == "Title: Only A Title"


def test_build_view_messages_rejects_unknown_view() -> None:
    """An unknown view fails loudly rather than training on a silent default."""
    with pytest.raises(ValueError, match="unknown view"):
        build_view_messages("bullet", "text")


@pytest.mark.parametrize("view", VIEWS)
def test_build_view_messages_names_the_view(view: str) -> None:
    """Each view's system prompt differs, so the model can tell them apart."""
    messages = build_view_messages(view, "query")
    assert messages[1]["content"] == "query"
    others = {
        build_view_messages(other, "query")[0]["content"]
        for other in VIEWS
        if other != view
    }
    assert messages[0]["content"] not in others


# -- Span candidates --


def test_span_candidates_round_trip_and_stay_within_papers(
    index: SubstringIndex, tokeniser: PreTrainedTokenizerBase
) -> None:
    """Every candidate re-tokenises exactly and comes from a single paper."""
    candidates = enumerate_span_candidates(index, tokeniser, span_tokens=4)
    assert set(candidates) == set(ABSTRACTS)
    for corpus_id, spans in candidates.items():
        assert spans, corpus_id
        own_words = set(ABSTRACTS[corpus_id].split())
        for span in spans:
            assert tokeniser.decode(tokeniser.encode(span.text)) == span.text
            # A span that leaked across a boundary would contain words from
            # another abstract.
            assert set(span.text.split()) <= own_words


# -- Dataset composition --


def test_mix_ratio_matches_the_published_portion(
    corpus_files: Path, index: SubstringIndex, tokeniser: PreTrainedTokenizerBase
) -> None:
    """Supervised rows follow the 3 : 10 : 5 title : substring : pseudo-query mix."""
    examples = build_minder_examples(
        make_config(corpus_files),
        index=index,
        tokeniser=tokeniser,
        rng=random.Random(0),
    )
    kinds = Counter(ex.kind for ex in examples)
    # Two distinct pairs survive facet dedup.
    assert kinds["title"] == 3 * 2
    assert kinds["substring"] == 10 * 2
    assert kinds["pseudo-query"] == 5 * 2


def test_facet_replicated_pairs_are_collapsed(
    corpus_files: Path, index: SubstringIndex, tokeniser: PreTrainedTokenizerBase
) -> None:
    """A pair appearing under four facets contributes one pair's worth of rows."""
    examples = build_minder_examples(
        make_config(corpus_files, mix={"title": 1, "substring": 0, "pseudo-query": 0}),
        index=index,
        tokeniser=tokeniser,
        rng=random.Random(0),
    )
    assert sum(1 for ex in examples if ex.kind == "title") == 2


def test_every_paper_is_a_target_in_every_view(
    corpus_files: Path, index: SubstringIndex, tokeniser: PreTrainedTokenizerBase
) -> None:
    """The unsupervised block guarantees exposure, including for gold-isolated papers.

    Paper 1 is never a `paper_b`, so it has zero supervised exposure; it must
    still appear as a target in all three views.
    """
    examples = build_minder_examples(
        make_config(corpus_files),
        index=index,
        tokeniser=tokeniser,
        rng=random.Random(0),
    )
    unsupervised = [ex for ex in examples if ex.kind.startswith("unsupervised-")]
    by_view = Counter(ex.kind for ex in unsupervised)
    for view in VIEWS:
        assert (
            by_view[f"unsupervised-{view}"]
            == 3
            * {
                "title": 3,
                "substring": 10,
                "pseudo-query": 5,
            }[view]
        )

    titles = {ex.target_docid for ex in examples if ex.kind.endswith("title")}
    assert titles == {f"title of paper {cid}" for cid in (1, 2, 3)}


def test_unsupervised_block_can_be_disabled(
    corpus_files: Path, index: SubstringIndex, tokeniser: PreTrainedTokenizerBase
) -> None:
    """`unsupervised_blocks=0` removes the exposure rows entirely."""
    examples = build_minder_examples(
        make_config(corpus_files, unsupervised_blocks=0),
        index=index,
        tokeniser=tokeniser,
        rng=random.Random(0),
    )
    assert not [ex for ex in examples if ex.kind.startswith("unsupervised-")]


def test_substring_targets_prefer_query_overlap(
    corpus_files: Path, index: SubstringIndex, tokeniser: PreTrainedTokenizerBase
) -> None:
    """Sampled spans skew toward higher-overlap candidates, weighted not filtered.

    There is no hard cutoff any more (ported from the source's weighted
    sampling, not a top-quantile filter), so this checks the *skew* over many
    draws rather than exclusion of any candidate.
    """
    examples = build_minder_examples(
        make_config(
            corpus_files, mix={"title": 0, "substring": 200, "pseudo-query": 0}
        ),
        index=index,
        tokeniser=tokeniser,
        rng=random.Random(0),
    )
    spans = [ex.target_docid for ex in examples if ex.kind == "substring"]
    assert spans
    # Paper 1's query mentions parsing and trees; paper 3 shares "parsing" and
    # "strong results", so its selected spans should skew toward those words
    # rather than the disjoint tail, over many draws.
    overlap_hits = sum(
        1 for s in spans if "parsing" in s or "strong" in s or "results" in s
    )
    assert overlap_hits / len(spans) > 0.5


# -- Ported fuzzy span-selection algorithm --


def test_stopword_anchored_candidates_never_score(index: SubstringIndex) -> None:
    """A candidate whose first word is a stopword never scores, however similar."""
    stopword_first = SpanCandidate(text="the strong results across languages")
    real_anchor = SpanCandidate(text="strong results across languages here")
    scores = _score_candidates_fuzzy(
        ["strong", "results", "across", "languages"],
        [stopword_first, real_anchor],
    )
    assert scores[0] == 0.0
    assert scores[1] > 0.0


def test_score_candidates_fuzzy_favors_closer_anchor() -> None:
    """A candidate whose anchor closely matches a query window scores higher."""
    close = SpanCandidate(text="strong results across many languages")
    far = SpanCandidate(text="dependency trees using neural networks")
    scores = _score_candidates_fuzzy(
        ["we", "report", "strong", "results", "across", "languages"], [close, far]
    )
    assert scores[0] > scores[1]


def test_sample_spans_fuzzy_falls_back_when_every_score_is_zero() -> None:
    """An all-zero score vector falls back to the first candidate, repeated.

    Matches the source's fallback (`indices = [0]`) when no passage position
    ever scores -- not an error, and not a uniform draw across candidates.
    """
    candidates = [SpanCandidate(text="the first"), SpanCandidate(text="the second")]
    result = _sample_spans_fuzzy(
        candidates, [0.0, 0.0], count=5, temperature=1.0, rng=random.Random(0)
    )
    assert result == ["the first"] * 5


def test_sample_spans_fuzzy_empty_candidates_returns_empty() -> None:
    """No candidates means no spans, not an error."""
    assert (
        _sample_spans_fuzzy([], [], count=3, temperature=1.0, rng=random.Random(0))
        == []
    )


def test_select_spans_is_weighted_not_uniform() -> None:
    """Over many draws, a strongly-favoured candidate dominates but isn't exclusive."""
    favoured = SpanCandidate(text="strong results across many languages today")
    disfavoured = SpanCandidate(text="dependency trees using neural networks only")
    drawn = select_spans(
        "we present strong results across languages",
        [favoured, disfavoured],
        count=200,
        temperature=1.0,
        rng=random.Random(0),
    )
    favoured_share = sum(1 for d in drawn if d == favoured.text) / len(drawn)
    assert 0.5 < favoured_share < 1.0


def test_unsupervised_substring_rows_share_one_prompt_per_paper(
    corpus_files: Path, index: SubstringIndex, tokeniser: PreTrainedTokenizerBase
) -> None:
    """All substring unsupervised rows for one paper share one input bullet.

    Ported from the source: one `extract_spans` call per document, not one
    per row -- so every row from that call shares its input query.
    """
    examples = build_minder_examples(
        make_config(corpus_files, mix={"title": 0, "substring": 10, "pseudo-query": 0}),
        index=index,
        tokeniser=tokeniser,
        rng=random.Random(0),
    )
    unsupervised = [ex for ex in examples if ex.kind == "unsupervised-substring"]
    by_paper: dict[str, set[str]] = {}
    for ex in unsupervised:
        by_paper.setdefault(ex.prompt_messages[1]["content"], set()).add("x")
    # Group rows by their rendered prompt; each paper contributes exactly one
    # distinct prompt across all of its 10 substring rows.
    prompts = [ex.prompt_messages[1]["content"] for ex in unsupervised]
    assert len(prompts) == 10 * 3
    distinct_prompts_per_paper = len(set(prompts))
    assert distinct_prompts_per_paper <= 3


def test_unsupervised_pseudo_query_target_avoids_its_own_input(
    corpus_files: Path, index: SubstringIndex, tokeniser: PreTrainedTokenizerBase
) -> None:
    """The pseudo-query view doesn't target the same bullet used as input."""
    examples = build_minder_examples(
        make_config(corpus_files, mix={"title": 0, "substring": 0, "pseudo-query": 20}),
        index=index,
        tokeniser=tokeniser,
        rng=random.Random(0),
    )
    rows = [ex for ex in examples if ex.kind == "unsupervised-pseudo-query"]
    assert rows
    for ex in rows:
        assert ex.prompt_messages[1]["content"] != ex.target_docid


def test_seed_determinism(
    corpus_files: Path, index: SubstringIndex, tokeniser: PreTrainedTokenizerBase
) -> None:
    """The same seed gives the same dataset, so runs are reproducible."""
    first = build_minder_examples(
        make_config(corpus_files),
        index=index,
        tokeniser=tokeniser,
        rng=random.Random(0),
    )
    second = build_minder_examples(
        make_config(corpus_files),
        index=index,
        tokeniser=tokeniser,
        rng=random.Random(0),
    )
    assert [(e.kind, e.target_docid) for e in first] == [
        (e.kind, e.target_docid) for e in second
    ]
