"""Tests for DocID encoding and inverted index construction."""

from pathlib import Path

import pytest

from genret.docid import (
    ASPECT_END,
    ASPECT_ORDER,
    PAPER_END,
    PAPER_START,
    TERM_SEPARATOR,
    InvertedIndex,
    build_docid_index,
    build_inverted_index,
    compute_index_stats,
    encode_all_docids,
    encode_docid,
    get_aspect_terms,
)
from genret.paper_parsing import AspectTerms, DocIDRecord, PaperTerms
from genret.types import CorpusId

# -- Fixtures --


def _make_terms(
    corpusid: int = 1,
    paper_terms: tuple[str, ...] = ("term a", "term b"),
    problems: tuple[str, ...] = ("prob 1",),
    methods: tuple[str, ...] = ("method 1", "method 2"),
    results: tuple[str, ...] = (),
    contributions: tuple[str, ...] = ("contrib 1",),
) -> PaperTerms:
    """Create a PaperTerms instance for testing."""
    return PaperTerms(
        corpusid=CorpusId(corpusid),
        paper_terms=paper_terms,
        aspect_terms=AspectTerms(
            problems=problems,
            methods=methods,
            results=results,
            contributions=contributions,
        ),
    )


@pytest.fixture
def sample_terms() -> PaperTerms:
    """Standard test terms with some empty aspects."""
    return _make_terms()


# -- TestGetAspectTerms --


class TestGetAspectTerms:
    """Tests for the get_aspect_terms accessor."""

    def test_problems(self) -> None:
        """Should return problems field."""
        aspect = AspectTerms(problems=("a", "b"))
        assert get_aspect_terms(aspect, "problems") == ("a", "b")

    def test_methods(self) -> None:
        """Should return methods field."""
        aspect = AspectTerms(methods=("m1",))
        assert get_aspect_terms(aspect, "methods") == ("m1",)

    def test_results(self) -> None:
        """Should return results field."""
        aspect = AspectTerms(results=("r1", "r2"))
        assert get_aspect_terms(aspect, "results") == ("r1", "r2")

    def test_contributions(self) -> None:
        """Should return contributions field."""
        aspect = AspectTerms(contributions=("c1",))
        assert get_aspect_terms(aspect, "contributions") == ("c1",)

    def test_invalid_name_raises(self) -> None:
        """Should raise ValueError for unknown aspect names."""
        with pytest.raises(ValueError, match="Unknown aspect name"):
            get_aspect_terms(AspectTerms(), "invalid")

    def test_all_aspect_order_names_valid(self) -> None:
        """Every name in ASPECT_ORDER should be a valid accessor."""
        aspect = AspectTerms()
        for name in ASPECT_ORDER:
            get_aspect_terms(aspect, name)  # should not raise


# -- TestEncodeDocid --


class TestEncodeDocid:
    """Tests for the encode_docid function."""

    def test_full_encoding(self, sample_terms: PaperTerms) -> None:
        """Should produce the expected structured DocID string."""
        docid = encode_docid(sample_terms)

        assert docid.startswith(PAPER_START)
        assert PAPER_END in docid
        assert "<ASPECT=problems>" in docid
        assert "<ASPECT=methods>" in docid
        assert "<ASPECT=contributions>" in docid
        # results is empty, so should be omitted
        assert "<ASPECT=results>" not in docid
        assert docid.count(ASPECT_END) == 3

    def test_empty_aspects_omitted(self) -> None:
        """Should omit aspect sections when all aspects are empty."""
        terms = _make_terms(problems=(), methods=(), results=(), contributions=())
        docid = encode_docid(terms)

        assert PAPER_START in docid
        assert PAPER_END in docid
        assert "ASPECT" not in docid

    def test_empty_paper_terms(self) -> None:
        """Should still produce a valid DocID with empty paper terms."""
        terms = _make_terms(paper_terms=())
        docid = encode_docid(terms)

        # Paper section is present but empty
        assert f"{PAPER_START}  {PAPER_END}" in docid

    def test_term_separator(self) -> None:
        """Should separate multiple terms with the defined separator."""
        terms = _make_terms(paper_terms=("alpha", "beta", "gamma"))
        docid = encode_docid(terms)

        expected_paper = (
            f"{PAPER_START} alpha{TERM_SEPARATOR}beta{TERM_SEPARATOR}gamma {PAPER_END}"
        )
        assert expected_paper in docid

    def test_determinism(self, sample_terms: PaperTerms) -> None:
        """Same input should always produce the same output."""
        assert encode_docid(sample_terms) == encode_docid(sample_terms)

    def test_aspect_order(self) -> None:
        """Aspects should appear in canonical order."""
        terms = _make_terms(
            problems=("p",),
            methods=("m",),
            results=("r",),
            contributions=("c",),
        )
        docid = encode_docid(terms)

        # Find positions of each aspect tag
        positions = [docid.index(f"<ASPECT={name}>") for name in ASPECT_ORDER]
        assert positions == sorted(positions)


# -- TestEncodeAllDocids --


class TestEncodeAllDocids:
    """Tests for the encode_all_docids function."""

    def test_from_jsonl(self, tmp_path: Path) -> None:
        """Should read terms.jsonl and produce DocIDRecords."""
        terms_path = tmp_path / "terms.jsonl"
        terms1 = _make_terms(corpusid=100)
        terms2 = _make_terms(corpusid=200, paper_terms=("x",))

        terms_path.write_text(
            terms1.model_dump_json() + "\n" + terms2.model_dump_json() + "\n",
            encoding="utf-8",
        )

        records = encode_all_docids(terms_path)
        assert len(records) == 2
        assert records[0].corpusid == 100
        assert records[1].corpusid == 200

    def test_empty_file(self, tmp_path: Path) -> None:
        """Should return empty list for an empty file."""
        terms_path = tmp_path / "terms.jsonl"
        terms_path.write_text("", encoding="utf-8")

        records = encode_all_docids(terms_path)
        assert records == []

    def test_duplicate_corpusid(self, tmp_path: Path) -> None:
        """Should keep only the last occurrence of a duplicate corpus ID."""
        terms_path = tmp_path / "terms.jsonl"
        terms1 = _make_terms(corpusid=100, paper_terms=("first",))
        terms2 = _make_terms(corpusid=100, paper_terms=("second",))

        terms_path.write_text(
            terms1.model_dump_json() + "\n" + terms2.model_dump_json() + "\n",
            encoding="utf-8",
        )

        records = encode_all_docids(terms_path)
        assert len(records) == 1
        assert "second" in records[0].docid

    def test_filter_ids(self, tmp_path: Path) -> None:
        """Should only encode papers whose corpus ID is in filter_ids."""
        terms_path = tmp_path / "terms.jsonl"
        terms1 = _make_terms(corpusid=100)
        terms2 = _make_terms(corpusid=200)
        terms3 = _make_terms(corpusid=300)

        terms_path.write_text(
            terms1.model_dump_json()
            + "\n"
            + terms2.model_dump_json()
            + "\n"
            + terms3.model_dump_json()
            + "\n",
            encoding="utf-8",
        )

        records = encode_all_docids(
            terms_path, filter_ids={CorpusId(100), CorpusId(300)}
        )
        assert [r.corpusid for r in records] == [100, 300]


# -- TestBuildInvertedIndex --


class TestBuildInvertedIndex:
    """Tests for the build_inverted_index function."""

    def test_basic_index(self) -> None:
        """Should produce correct mappings for simple input."""
        records = [
            DocIDRecord(corpusid=CorpusId(1), docid="a b"),
            DocIDRecord(corpusid=CorpusId(2), docid="b c"),
        ]
        token_seqs = [[10, 20], [20, 30]]

        inv = build_inverted_index(records, token_seqs)

        assert inv.paper_to_tokens[CorpusId(1)] == [10, 20]
        assert inv.paper_to_tokens[CorpusId(2)] == [20, 30]
        assert inv.token_to_papers[10] == {CorpusId(1)}
        assert inv.token_to_papers[20] == {CorpusId(1), CorpusId(2)}
        assert inv.token_to_papers[30] == {CorpusId(2)}
        assert inv.vocabulary == {10, 20, 30}

    def test_empty_input(self) -> None:
        """Should return empty structures for empty input."""
        inv = build_inverted_index([], [])

        assert inv.token_to_papers == {}
        assert inv.paper_to_tokens == {}
        assert inv.vocabulary == set()


# -- TestComputeIndexStats --


def _make_inv_index(
    token_to_papers: dict[int, set[CorpusId]] | None = None,
    paper_to_tokens: dict[CorpusId, list[int]] | None = None,
    vocabulary: set[int] | None = None,
) -> InvertedIndex:
    """Create an InvertedIndex for testing."""
    return InvertedIndex(
        token_to_papers=token_to_papers if token_to_papers is not None else {},
        paper_to_tokens=paper_to_tokens if paper_to_tokens is not None else {},
        vocabulary=vocabulary or set(),
    )


class TestComputeIndexStats:
    """Tests for the compute_index_stats function."""

    def test_correct_values(self) -> None:
        """Should compute correct statistics."""
        inv = _make_inv_index(
            token_to_papers={10: {CorpusId(1), CorpusId(2)}, 20: {CorpusId(1)}},
            paper_to_tokens={CorpusId(1): [10, 20], CorpusId(2): [10]},
            vocabulary={10, 20},
        )
        stats = compute_index_stats(inv)

        assert stats.vocabulary_size == 2
        assert stats.num_papers == 2
        assert stats.mean_tokens_per_docid == 1.5  # (2 + 1) / 2
        assert stats.max_tokens_per_docid == 2
        assert stats.mean_papers_per_token == 1.5  # (2 + 1) / 2

    def test_empty_input(self) -> None:
        """Should handle empty input gracefully."""
        stats = compute_index_stats(_make_inv_index())

        assert stats.vocabulary_size == 0
        assert stats.num_papers == 0
        assert stats.mean_tokens_per_docid == 0.0
        assert stats.max_tokens_per_docid == 0
        assert stats.mean_papers_per_token == 0.0


# -- TestBuildDocidIndex --


class TestBuildDocidIndex:
    """Tests for the build_docid_index function."""

    def test_sorted_values(self) -> None:
        """Vocabulary and mapping values should be sorted."""
        inv = _make_inv_index(
            token_to_papers={30: {CorpusId(2), CorpusId(1)}, 10: {CorpusId(1)}},
            paper_to_tokens={CorpusId(1): [30, 10], CorpusId(2): [30]},
            vocabulary={30, 10},
        )
        index = build_docid_index(inv, "test-tokeniser")

        assert index.vocabulary == [10, 30]
        assert index.token_to_papers["30"] == [1, 2]

    def test_tokeniser_name(self) -> None:
        """Should store the tokeniser name."""
        index = build_docid_index(_make_inv_index(), "my-tokeniser")
        assert index.tokeniser == "my-tokeniser"

    def test_stats_populated(self) -> None:
        """Should compute and embed stats."""
        inv = _make_inv_index(
            token_to_papers={10: {CorpusId(1)}},
            paper_to_tokens={CorpusId(1): [10]},
            vocabulary={10},
        )
        index = build_docid_index(inv, "test-tokeniser")
        assert index.stats.num_papers == 1
        assert index.stats.vocabulary_size == 1
