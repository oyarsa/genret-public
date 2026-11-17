"""Unit tests for paper_processing.py extracted functions."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from genret.paper_parsing import (
    EnrichedPaper,
    FacetRecordType,
    FacetsSummary,
    FacetWithEvidence,
    PaperGraph,
    RelatedPaperInfo,
    RelationType,
    ScoredPaperPair,
    TrainingPair,
)
from genret.processing.cli import (
    build_pairs_index,
    collect_reference_pairs,
    collect_scored_training_pairs,
    enrich_paper_with_metadata,
    parse_facets_record,
)
from genret.processing.facet_extract_cmd import (
    FacetsWithEvidenceResponse,
    prepare_extract_state,
)
from genret.types import CorpusId


class TestCollectReferencePairs:
    """Tests for collect_reference_pairs function."""

    def test_collects_valid_pairs(self) -> None:
        """Should collect pairs where both papers are in valid_ids."""
        refs = {
            CorpusId(1): [CorpusId(2), CorpusId(3), CorpusId(4)],
            CorpusId(2): [CorpusId(3), CorpusId(5)],
        }
        valid_ids = {CorpusId(1), CorpusId(2), CorpusId(3)}

        pairs = collect_reference_pairs(refs, valid_ids)

        # Should have: (1,2), (1,3), (2,3)
        assert len(pairs) == 3
        pair_tuples = {(p.paper_a, p.paper_b) for p in pairs}
        assert pair_tuples == {
            (CorpusId(1), CorpusId(2)),
            (CorpusId(1), CorpusId(3)),
            (CorpusId(2), CorpusId(3)),
        }
        assert all(p.relation == RelationType.REFERENCE for p in pairs)
        assert all(p.score == 1.0 for p in pairs)

    def test_excludes_invalid_pairs(self) -> None:
        """Should exclude pairs where either paper is not in valid_ids."""
        refs = {
            CorpusId(1): [CorpusId(2), CorpusId(100)],
            CorpusId(100): [CorpusId(1), CorpusId(2)],
        }
        valid_ids = {CorpusId(1), CorpusId(2)}

        pairs = collect_reference_pairs(refs, valid_ids)

        assert len(pairs) == 1
        assert pairs[0].paper_a == CorpusId(1)
        assert pairs[0].paper_b == CorpusId(2)

    def test_empty_refs(self) -> None:
        """Should handle empty refs dict."""
        pairs = collect_reference_pairs({}, {CorpusId(1), CorpusId(2), CorpusId(3)})
        assert pairs == []

    def test_empty_valid_ids(self) -> None:
        """Should return empty when valid_ids is empty."""
        refs = {CorpusId(1): [CorpusId(2), CorpusId(3)]}
        pairs = collect_reference_pairs(refs, set())
        assert pairs == []


class TestCollectScoredTrainingPairs:
    """Tests for collect_scored_training_pairs function."""

    def test_collects_valid_co_citation_pairs(self) -> None:
        """Should collect co-citation pairs where both papers are in valid_ids."""
        co_cites = [
            ScoredPaperPair(paper_a=CorpusId(1), paper_b=CorpusId(2), score=5),
            ScoredPaperPair(paper_a=CorpusId(1), paper_b=CorpusId(3), score=3),
            ScoredPaperPair(paper_a=CorpusId(2), paper_b=CorpusId(100), score=2),
        ]
        valid_ids = {CorpusId(1), CorpusId(2), CorpusId(3)}

        pairs = collect_scored_training_pairs(
            co_cites, valid_ids, RelationType.CO_CITATION
        )

        assert len(pairs) == 2
        assert all(p.relation == RelationType.CO_CITATION for p in pairs)
        scores = {(p.paper_a, p.paper_b): p.score for p in pairs}
        assert scores[(CorpusId(1), CorpusId(2))] == 5.0
        assert scores[(CorpusId(1), CorpusId(3))] == 3.0

    def test_collects_valid_bib_coupling_pairs(self) -> None:
        """Should collect bib coupling pairs where both papers are in valid_ids."""
        bib_couples = [
            ScoredPaperPair(paper_a=CorpusId(1), paper_b=CorpusId(2), score=10),
            ScoredPaperPair(paper_a=CorpusId(1), paper_b=CorpusId(3), score=5),
            ScoredPaperPair(paper_a=CorpusId(2), paper_b=CorpusId(100), score=3),
        ]
        valid_ids = {CorpusId(1), CorpusId(2), CorpusId(3)}

        pairs = collect_scored_training_pairs(
            bib_couples, valid_ids, RelationType.BIB_COUPLING
        )

        assert len(pairs) == 2
        assert all(p.relation == RelationType.BIB_COUPLING for p in pairs)
        scores = {(p.paper_a, p.paper_b): p.score for p in pairs}
        assert scores[(CorpusId(1), CorpusId(2))] == 10.0
        assert scores[(CorpusId(1), CorpusId(3))] == 5.0

    def test_excludes_invalid_pairs(self) -> None:
        """Should exclude pairs where either paper is not in valid_ids."""
        scored_pairs = [
            ScoredPaperPair(paper_a=CorpusId(1), paper_b=CorpusId(100), score=5),
            ScoredPaperPair(paper_a=CorpusId(100), paper_b=CorpusId(2), score=3),
        ]
        valid_ids = {CorpusId(1), CorpusId(2)}

        pairs = collect_scored_training_pairs(
            scored_pairs, valid_ids, RelationType.CO_CITATION
        )
        assert pairs == []

    def test_empty_pairs(self) -> None:
        """Should handle empty pairs list."""
        pairs = collect_scored_training_pairs(
            [], {CorpusId(1), CorpusId(2), CorpusId(3)}, RelationType.BIB_COUPLING
        )
        assert pairs == []


class TestBuildPairsIndex:
    """Tests for build_pairs_index function."""

    def test_builds_bidirectional_index(self) -> None:
        """Should add entries to both papers' lists."""
        pairs = [
            TrainingPair(
                paper_a=CorpusId(1),
                paper_b=CorpusId(2),
                relation=RelationType.REFERENCE,
                score=1.0,
            ),
        ]
        titles = {CorpusId(1): "Paper 1", CorpusId(2): "Paper 2"}

        index = build_pairs_index(pairs, titles)

        # Both papers should have entries
        assert CorpusId(1) in index
        assert CorpusId(2) in index

        # Paper 1's entry should reference paper 2
        assert index[CorpusId(1)][0].related_paper_id == CorpusId(2)
        assert index[CorpusId(1)][0].related_paper_title == "Paper 2"

        # Paper 2's entry should reference paper 1
        assert index[CorpusId(2)][0].related_paper_id == CorpusId(1)
        assert index[CorpusId(2)][0].related_paper_title == "Paper 1"

    def test_preserves_relation_and_score(self) -> None:
        """Should preserve relation type and score."""
        pairs = [
            TrainingPair(
                paper_a=CorpusId(1),
                paper_b=CorpusId(2),
                relation=RelationType.CO_CITATION,
                score=5.0,
            ),
        ]
        titles = {CorpusId(1): "P1", CorpusId(2): "P2"}

        index = build_pairs_index(pairs, titles)

        assert index[CorpusId(1)][0].relation == RelationType.CO_CITATION
        assert index[CorpusId(1)][0].score == 5.0

    def test_handles_missing_titles(self) -> None:
        """Should use 'Unknown' for missing titles."""
        pairs = [
            TrainingPair(
                paper_a=CorpusId(1),
                paper_b=CorpusId(2),
                relation=RelationType.REFERENCE,
                score=1.0,
            ),
        ]
        titles = {CorpusId(1): "Paper 1"}  # Missing title for paper 2

        index = build_pairs_index(pairs, titles)

        assert index[CorpusId(1)][0].related_paper_title == "Unknown"

    def test_multiple_pairs_same_paper(self) -> None:
        """Should accumulate multiple pairs for the same paper."""
        pairs = [
            TrainingPair(
                paper_a=CorpusId(1),
                paper_b=CorpusId(2),
                relation=RelationType.REFERENCE,
                score=1.0,
            ),
            TrainingPair(
                paper_a=CorpusId(1),
                paper_b=CorpusId(3),
                relation=RelationType.CO_CITATION,
                score=3.0,
            ),
        ]
        titles = {CorpusId(1): "P1", CorpusId(2): "P2", CorpusId(3): "P3"}

        index = build_pairs_index(pairs, titles)

        assert len(index[CorpusId(1)]) == 2
        related_ids = {p.related_paper_id for p in index[CorpusId(1)]}
        assert related_ids == {CorpusId(2), CorpusId(3)}

    def test_empty_pairs(self) -> None:
        """Should handle empty pairs list."""
        index = build_pairs_index([], {CorpusId(1): "P1"})
        assert index == {}


class TestParseFacetsRecord:
    """Tests for parse_facets_record function."""

    def test_parses_complete_record(self) -> None:
        """Should parse all facet fields."""
        record = {
            "corpusid": 123,
            "problems": ["p1", "p2"],
            "methods": ["m1"],
            "results": ["r1", "r2", "r3"],
            "contributions": ["c1"],
        }

        result = parse_facets_record(record, facets_type=FacetRecordType.STANDARD)

        assert isinstance(result, FacetsSummary)
        assert list(result.problems) == ["p1", "p2"]
        assert list(result.methods) == ["m1"]
        assert list(result.results) == ["r1", "r2", "r3"]
        assert list(result.contributions) == ["c1"]

    def test_rejects_missing_fields(self) -> None:
        """Should reject records missing required facet fields."""
        record = {"corpusid": 123, "problems": ["p1"]}

        with pytest.raises(ValidationError):
            parse_facets_record(record, facets_type=FacetRecordType.STANDARD)

    def test_rejects_empty_record(self) -> None:
        """Should reject an empty record."""
        with pytest.raises(ValidationError):
            parse_facets_record({}, facets_type=FacetRecordType.STANDARD)

    def test_parses_evidence_object_items(self) -> None:
        """Should extract facet text from evidence-backed object items."""
        record = {
            "corpusid": 123,
            "problems": [{"facet": "p1", "evidence": "e1"}],
            "methods": [{"facet": "m1", "evidence": "e2"}],
            "results": [{"facet": "r1", "evidence": "e3"}],
            "contributions": [{"facet": "c1", "evidence": "e4"}],
        }

        result = parse_facets_record(
            record,
            facets_type=FacetRecordType.WITH_EVIDENCE,
        )

        assert list(result.problems) == ["p1"]
        assert list(result.methods) == ["m1"]
        assert list(result.results) == ["r1"]
        assert list(result.contributions) == ["c1"]


class TestEnrichPaperWithMetadata:
    """Tests for enrich_paper_with_metadata function."""

    def test_adds_facets_and_pairs(self) -> None:
        """Should return EnrichedPaper with facets and pairs added."""
        paper = PaperGraph(
            corpusid=CorpusId(1),
            title="Test Paper",
            authors=["Author 1"],
            sections=[],
        )
        facets_summary = FacetsSummary(
            problems=["p1"], methods=[], results=[], contributions=[]
        )
        facets = {CorpusId(1): facets_summary}
        pair_info = RelatedPaperInfo(
            related_paper_id=CorpusId(2),
            related_paper_title="Related Paper",
            relation=RelationType.REFERENCE,
            score=1.0,
        )
        pairs_by_paper = {CorpusId(1): [pair_info]}

        result = enrich_paper_with_metadata(paper, facets, pairs_by_paper)

        assert isinstance(result, EnrichedPaper)
        assert result.facets == facets_summary
        assert list(result.ground_truth_pairs) == [pair_info]
        assert result.corpusid == CorpusId(1)
        assert result.title == "Test Paper"

    def test_handles_missing_facets(self) -> None:
        """Should set facets to None when not found."""
        paper = PaperGraph(
            corpusid=CorpusId(1),
            title="Test Paper",
            authors=[],
            sections=[],
        )
        facets: dict[CorpusId, FacetsSummary] = {}
        pairs_by_paper: dict[CorpusId, list[RelatedPaperInfo]] = {}

        result = enrich_paper_with_metadata(paper, facets, pairs_by_paper)

        assert result.facets is None
        assert list(result.ground_truth_pairs) == []

    def test_handles_missing_pairs(self) -> None:
        """Should set pairs to empty tuple when not found."""
        paper = PaperGraph(
            corpusid=CorpusId(1),
            title="Test Paper",
            authors=[],
            sections=[],
        )
        facets_summary = FacetsSummary(
            problems=[], methods=[], results=[], contributions=[]
        )
        facets = {CorpusId(1): facets_summary}
        pairs_by_paper: dict[CorpusId, list[RelatedPaperInfo]] = {}

        result = enrich_paper_with_metadata(paper, facets, pairs_by_paper)

        assert result.facets is not None
        assert list(result.ground_truth_pairs) == []


class TestPrepareExtractState:
    """Tests for prepare_extract_state function."""

    def test_force_mode_backs_up_existing_files(self, tmp_path: Path) -> None:
        """Force mode should back up old progress/facets files and start fresh."""
        progress = tmp_path / "facets_progress.txt"
        facets = tmp_path / "facets.jsonl"
        progress.write_text("1\n2\n", encoding="utf-8")
        facets.write_text('{"corpusid":1}\n', encoding="utf-8")

        processed_ids, mode = prepare_extract_state(progress, facets, force=True)

        assert processed_ids == set()
        assert mode == "w"
        assert not progress.exists()
        assert not facets.exists()
        assert (tmp_path / "facets_progress.txt.bak").exists()
        assert (tmp_path / "facets.jsonl.bak").exists()

    def test_resume_mode_loads_progress_without_backup(self, tmp_path: Path) -> None:
        """Resume mode should keep files and return append mode."""
        progress = tmp_path / "facets_progress.txt"
        facets = tmp_path / "facets.jsonl"
        progress.write_text("10\n20\n", encoding="utf-8")
        facets.write_text('{"corpusid":10}\n', encoding="utf-8")

        processed_ids, mode = prepare_extract_state(progress, facets, force=False)

        assert processed_ids == {10, 20}
        assert mode == "a"
        assert progress.exists()
        assert facets.exists()
        assert not (tmp_path / "facets_progress.txt.bak").exists()
        assert not (tmp_path / "facets.jsonl.bak").exists()


class TestFacetsWithEvidenceResponse:
    """Tests for evidence-backed facet response conversion helpers."""

    def test_finalise_returns_evidence_record(self) -> None:
        """Finalise should preserve evidence objects in the record."""
        response = FacetsWithEvidenceResponse(
            problems=(FacetWithEvidence(facet="P1", evidence="E1"),),
            methods=(FacetWithEvidence(facet="M1", evidence="E2"),),
            results=(FacetWithEvidence(facet="R1", evidence="E3"),),
            contributions=(FacetWithEvidence(facet="C1", evidence="E4"),),
        )

        record = response.finalise(CorpusId(123))

        assert record.corpusid == 123
        assert record.problems[0].facet == "P1"
        assert record.methods[0].facet == "M1"
        assert record.results[0].facet == "R1"
        assert record.contributions[0].facet == "C1"

    def test_finalise_preserves_evidence(self) -> None:
        """Finalise should preserve evidence text for serialisation."""
        response = FacetsWithEvidenceResponse(
            problems=(FacetWithEvidence(facet="P1", evidence="Evidence"),)
        )

        record = response.finalise(CorpusId(42))

        assert record.corpusid == 42
        assert isinstance(record.problems[0], FacetWithEvidence)
        first = record.problems[0]
        assert first.facet == "P1"
        assert first.evidence == "Evidence"
