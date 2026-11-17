"""Unit tests for pipeline.py extracted functions."""

from pathlib import Path

import pytest

from genret.paper_parsing import ScoredPaperPair
from genret.pipeline.common import Stats, maybe_load_corpus_ids_for_dry_run
from genret.pipeline.derive import (
    compute_bib_coupling_pairs,
    compute_co_citation_pairs,
    filter_pairs_by_ids,
    split_papers_by_year,
)
from genret.pipeline.index import build_index_stats
from genret.types import CorpusId, Year


class TestComputeCoCitationPairs:
    """Tests for compute_co_citation_pairs function."""

    def test_finds_co_citation_pairs(self) -> None:
        """Should find papers that are cited together."""
        refs = {
            CorpusId(1): [CorpusId(100), CorpusId(200), CorpusId(300)],
            CorpusId(2): [CorpusId(100), CorpusId(200)],
            CorpusId(3): [CorpusId(100)],
        }

        result = compute_co_citation_pairs(refs, min_co_citations=2)

        # (100, 200) is cited together by papers 1 and 2
        assert len(result) == 1
        pair = result[0]
        assert (pair.paper_a, pair.paper_b) == (CorpusId(100), CorpusId(200))
        assert pair.score == 2

    def test_respects_min_threshold(self) -> None:
        """Should filter out pairs below threshold."""
        refs = {
            CorpusId(1): [CorpusId(100), CorpusId(200), CorpusId(300)],
            CorpusId(2): [CorpusId(100), CorpusId(200)],
        }

        # With threshold of 3, no pairs qualify
        result = compute_co_citation_pairs(refs, min_co_citations=3)
        assert result == []

        # With threshold of 1, multiple pairs qualify
        result = compute_co_citation_pairs(refs, min_co_citations=1)
        assert len(result) == 3  # (100,200), (100,300), (200,300)

    def test_sorted_by_count_descending(self) -> None:
        """Should return pairs sorted by count descending."""
        refs = {
            CorpusId(1): [CorpusId(100), CorpusId(200)],
            CorpusId(2): [CorpusId(100), CorpusId(200)],
            CorpusId(3): [CorpusId(100), CorpusId(200)],  # (100,200) appears 3 times
            CorpusId(4): [CorpusId(100), CorpusId(300)],
            CorpusId(5): [CorpusId(100), CorpusId(300)],  # (100,300) appears 2 times
        }

        result = compute_co_citation_pairs(refs, min_co_citations=1)

        # First pair should have highest count
        assert (result[0].paper_a, result[0].paper_b) == (CorpusId(100), CorpusId(200))
        assert result[0].score == 3

    def test_empty_refs(self) -> None:
        """Should handle empty refs dict."""
        result = compute_co_citation_pairs({}, min_co_citations=1)
        assert result == []

    def test_single_citation(self) -> None:
        """Should handle papers with only one citation (no pairs)."""
        refs = {
            CorpusId(1): [CorpusId(100)],
            CorpusId(2): [CorpusId(200)],
        }
        result = compute_co_citation_pairs(refs, min_co_citations=1)
        assert result == []


class TestComputeBibCouplingPairs:
    """Tests for compute_bib_coupling_pairs function."""

    def test_finds_coupled_papers(self) -> None:
        """Should find papers that share references."""
        refs = {
            CorpusId(1): [CorpusId(100), CorpusId(200), CorpusId(300)],
            CorpusId(2): [CorpusId(100), CorpusId(200)],
            CorpusId(3): [CorpusId(400)],
        }

        pairs, skipped = compute_bib_coupling_pairs(
            refs, min_shared_refs=2, max_citing=100
        )

        assert len(pairs) == 1
        pair = pairs[0]
        assert {pair.paper_a, pair.paper_b} == {CorpusId(1), CorpusId(2)}
        assert pair.score == 2
        assert skipped == 0

    def test_respects_min_shared_refs(self) -> None:
        """Should filter pairs below min_shared_refs threshold."""
        refs = {
            CorpusId(1): [CorpusId(100), CorpusId(200)],
            CorpusId(2): [CorpusId(100)],
        }

        pairs, _ = compute_bib_coupling_pairs(refs, min_shared_refs=2, max_citing=100)
        assert pairs == []

        pairs, _ = compute_bib_coupling_pairs(refs, min_shared_refs=1, max_citing=100)
        assert len(pairs) == 1

    def test_skips_highly_cited_refs(self) -> None:
        """Should skip references cited by more than max_citing papers."""
        # Paper 100 is cited by 3 papers (above max_citing=2)
        refs = {
            CorpusId(1): [CorpusId(100)],
            CorpusId(2): [CorpusId(100)],
            CorpusId(3): [CorpusId(100)],
        }

        pairs, skipped = compute_bib_coupling_pairs(
            refs, min_shared_refs=1, max_citing=2
        )

        assert pairs == []
        assert skipped == 1

    def test_sorted_by_shared_count_descending(self) -> None:
        """Should return pairs sorted by shared count descending."""
        refs = {
            CorpusId(1): [CorpusId(100), CorpusId(200), CorpusId(300)],
            CorpusId(2): [CorpusId(100), CorpusId(200), CorpusId(300)],
            CorpusId(3): [CorpusId(100), CorpusId(200)],
            CorpusId(4): [CorpusId(100), CorpusId(200)],
        }

        pairs, _ = compute_bib_coupling_pairs(refs, min_shared_refs=1, max_citing=100)

        # Pair with more shared refs should come first
        top_pair = pairs[0]
        assert {top_pair.paper_a, top_pair.paper_b} == {CorpusId(1), CorpusId(2)}
        assert top_pair.score == 3

    def test_empty_refs(self) -> None:
        """Should handle empty refs dict."""
        pairs, skipped = compute_bib_coupling_pairs(
            {}, min_shared_refs=1, max_citing=100
        )
        assert pairs == []
        assert skipped == 0


class TestFilterPairsByIds:
    """Tests for filter_pairs_by_ids function."""

    def test_filters_to_valid_ids(self) -> None:
        """Should keep only pairs where both papers are in valid_ids."""
        pairs = [
            ScoredPaperPair(paper_a=CorpusId(1), paper_b=CorpusId(2), score=10),
            ScoredPaperPair(paper_a=CorpusId(1), paper_b=CorpusId(3), score=8),
            ScoredPaperPair(paper_a=CorpusId(2), paper_b=CorpusId(3), score=5),
            ScoredPaperPair(paper_a=CorpusId(3), paper_b=CorpusId(4), score=3),
        ]
        valid_ids = {CorpusId(1), CorpusId(2)}

        result = filter_pairs_by_ids(pairs, valid_ids)

        assert len(result) == 1
        assert result[0] == ScoredPaperPair(
            paper_a=CorpusId(1), paper_b=CorpusId(2), score=10
        )

    def test_empty_valid_ids(self) -> None:
        """Should return empty list when valid_ids is empty."""
        pairs = [ScoredPaperPair(paper_a=CorpusId(1), paper_b=CorpusId(2), score=10)]
        result = filter_pairs_by_ids(pairs, set())
        assert result == []

    def test_empty_pairs(self) -> None:
        """Should handle empty pairs list."""
        result = filter_pairs_by_ids([], {CorpusId(1), CorpusId(2), CorpusId(3)})
        assert result == []

    def test_preserves_order(self) -> None:
        """Should preserve the order of pairs."""
        pairs = [
            ScoredPaperPair(paper_a=CorpusId(1), paper_b=CorpusId(2), score=10),
            ScoredPaperPair(paper_a=CorpusId(1), paper_b=CorpusId(3), score=8),
            ScoredPaperPair(paper_a=CorpusId(2), paper_b=CorpusId(3), score=5),
        ]
        valid_ids = {CorpusId(1), CorpusId(2), CorpusId(3)}

        result = filter_pairs_by_ids(pairs, valid_ids)

        assert result == pairs


class TestSplitPapersByYear:
    """Tests for split_papers_by_year function."""

    def test_splits_by_year_ranges(self) -> None:
        """Should correctly assign papers to train/dev/test by year."""
        years = {
            CorpusId(1): Year(2020),
            CorpusId(2): Year(2021),
            CorpusId(3): Year(2022),
            CorpusId(4): Year(2023),
            CorpusId(5): Year(2024),
        }

        result = split_papers_by_year(
            years,
            train_years=(2020, 2022),
            dev_year=2023,
            test_year=2024,
        )

        assert set(result.train) == {1, 2, 3}
        assert set(result.dev) == {4}
        assert set(result.test) == {5}

    def test_excludes_papers_outside_ranges(self) -> None:
        """Should not include papers that don't match any split."""
        years = {
            CorpusId(1): Year(2018),  # Before train range
            CorpusId(2): Year(2019),  # Before train range
            CorpusId(3): Year(2020),  # In train range
        }

        result = split_papers_by_year(
            years,
            train_years=(2020, 2022),
            dev_year=2023,
            test_year=2024,
        )

        assert result.train == [3]
        assert result.dev == []
        assert result.test == []

    def test_include_all_train_mode(self) -> None:
        """When train_years[0] == 0, should include all years except dev/test."""
        years = {
            CorpusId(1): Year(2010),
            CorpusId(2): Year(2015),
            CorpusId(3): Year(2020),
            CorpusId(4): Year(2023),  # dev
            CorpusId(5): Year(2024),  # test
        }

        result = split_papers_by_year(
            years,
            train_years=(0, 2022),  # 0 means include all
            dev_year=2023,
            test_year=2024,
        )

        assert set(result.train) == {1, 2, 3}
        assert set(result.dev) == {4}
        assert set(result.test) == {5}

    def test_empty_years(self) -> None:
        """Should handle empty years dict."""
        result = split_papers_by_year(
            {},
            train_years=(2020, 2022),
            dev_year=2023,
            test_year=2024,
        )

        assert result.train == []
        assert result.dev == []
        assert result.test == []


class TestBuildIndexStats:
    """Tests for build_index_stats function."""

    def test_builds_complete_stats(self) -> None:
        """Should build stats dict with all fields."""
        stats = Stats(
            files_processed=5,
            records_matched=100,
            bytes_downloaded=1000000,
            matched_acl_only=30,
            matched_venue_only=50,
            matched_both=20,
        )
        corpus_ids = {CorpusId(i) for i in range(100)}
        processed_files = {"file1.gz", "file2.gz"}

        result = build_index_stats(
            release_id="2024-01-01",
            year_range=(2010, 2024),
            venue_patterns=[r"\bACL\b", r"\bEMNLP\b"],
            corpus_ids=corpus_ids,
            stats=stats,
            processed_files=processed_files,
        )

        assert result["release_id"] == "2024-01-01"
        assert result["year_range"] == [2010, 2024]
        assert result["venue_patterns"] == [r"\bACL\b", r"\bEMNLP\b"]
        assert result["stats"]["corpus_ids"] == 100
        assert result["stats"]["files_processed"] == 7  # 5 + 2 processed
        assert result["stats"]["records_matched"] == 100
        assert result["stats"]["bytes_downloaded"] == 1000000
        assert result["stats"]["matched_acl_only"] == 30
        assert result["stats"]["matched_venue_only"] == 50
        assert result["stats"]["matched_both"] == 20

    def test_handles_empty_stats(self) -> None:
        """Should handle empty stats correctly."""
        stats = Stats()
        corpus_ids: set[CorpusId] = set()
        processed_files: set[str] = set()

        result = build_index_stats(
            release_id="test",
            year_range=(2020, 2024),
            venue_patterns=[],
            corpus_ids=corpus_ids,
            stats=stats,
            processed_files=processed_files,
        )

        assert result["stats"]["corpus_ids"] == 0
        assert result["stats"]["files_processed"] == 0
        assert result["stats"]["records_matched"] == 0


class TestMaybeLoadCorpusIdsForDryRun:
    """Tests for maybe_load_corpus_ids_for_dry_run function."""

    def test_returns_none_when_missing_in_dry_run(self, tmp_path: Path) -> None:
        """Should not fail in dry-run mode when corpus_ids.txt is missing."""
        result = maybe_load_corpus_ids_for_dry_run(
            tmp_path,
            "fulltext phase",
            dry_run=True,
        )
        assert result is None

    def test_loads_ids_when_index_exists(self, tmp_path: Path) -> None:
        """Should load IDs when index file is present."""
        index_path = tmp_path / "corpus_ids.txt"
        index_path.write_text("1\n2\n3\n", encoding="utf-8")

        result = maybe_load_corpus_ids_for_dry_run(
            tmp_path,
            "citations phase",
            dry_run=True,
        )
        assert result == {1, 2, 3}

    def test_raises_when_missing_without_dry_run(self, tmp_path: Path) -> None:
        """Should raise SystemExit outside dry-run if prerequisite is missing."""
        with pytest.raises(SystemExit):
            maybe_load_corpus_ids_for_dry_run(
                tmp_path,
                "fulltext phase",
                dry_run=False,
            )
