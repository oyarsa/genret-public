"""Tests for local judge sharding helpers."""

from pathlib import Path

import pytest

from genret.processing.llm_judge import (
    PairToJudge,
    ScoredPair,
    ScoreRange,
    ScoringMode,
)
from genret.processing.local_judge_sharding import (
    merge_scored_shards,
    select_pairs_for_shard,
)
from genret.types import CorpusId
from genret.utils import open_auto, resolve_jsonl


def test_select_pairs_for_shard_round_robin() -> None:
    """Select a stable round-robin subset for one shard."""
    pairs = [
        PairToJudge(paper_a=CorpusId(0), paper_b=CorpusId(index), source="retrieved")
        for index in range(7)
    ]

    shard = select_pairs_for_shard(pairs, shard_index=1, num_shards=3)

    assert [pair.paper_b for pair in shard] == [1, 4]


def test_select_pairs_for_shard_rejects_invalid_index() -> None:
    """Reject shard indexes outside the configured range."""
    pairs = [PairToJudge(paper_a=CorpusId(1), paper_b=CorpusId(2), source="retrieved")]

    with pytest.raises(ValueError, match="smaller than"):
        select_pairs_for_shard(pairs, shard_index=2, num_shards=2)


def _write_scored_pairs(path: Path, pairs: list[ScoredPair]) -> None:
    """Write a small scored-pair JSONL fixture."""
    resolved_path = resolve_jsonl(path)
    with open_auto(resolved_path, "w") as file_handle:
        for pair in pairs:
            file_handle.write(pair.model_dump_json())
            file_handle.write("\n")


def test_merge_scored_shards_merges_and_sorts(tmp_path: Path) -> None:
    """Merge compatible shard outputs into one deterministic file."""
    shard_a = tmp_path / "scores_a.jsonl"
    shard_b = tmp_path / "scores_b.jsonl"
    output = tmp_path / "merged_scores.jsonl"

    _write_scored_pairs(
        shard_a,
        [
            ScoredPair(
                paper_a=CorpusId(2),
                paper_b=CorpusId(3),
                source="retrieved",
                score=0.7,
                mode=ScoringMode.BINARY_SEQUENCE,
                score_range=ScoreRange.ZERO_TO_ONE,
            )
        ],
    )
    _write_scored_pairs(
        shard_b,
        [
            ScoredPair(
                paper_a=CorpusId(1),
                paper_b=CorpusId(4),
                source="random_negative",
                score=0.1,
                mode=ScoringMode.BINARY_SEQUENCE,
                score_range=ScoreRange.ZERO_TO_ONE,
            )
        ],
    )

    summary = merge_scored_shards(
        input_paths=[shard_a, shard_b],
        output_path=output,
    )

    assert summary["pair_count"] == 2
    assert summary["mode"] == ScoringMode.BINARY_SEQUENCE.value
    assert summary["score_range"] == ScoreRange.ZERO_TO_ONE.value

    resolved_output = resolve_jsonl(output)
    with open_auto(resolved_output, "r") as file_handle:
        merged_lines = [line.strip() for line in file_handle if line.strip()]

    assert len(merged_lines) == 2
    assert '"paper_a":1' in merged_lines[0]
    assert '"paper_a":2' in merged_lines[1]


def test_merge_scored_shards_rejects_duplicate_pair(tmp_path: Path) -> None:
    """Reject duplicate scored pairs across shard outputs."""
    duplicate_pair = ScoredPair(
        paper_a=CorpusId(1),
        paper_b=CorpusId(2),
        source="retrieved",
        score=0.9,
        mode=ScoringMode.BINARY_SEQUENCE,
        score_range=ScoreRange.ZERO_TO_ONE,
    )
    shard_a = tmp_path / "scores_a.jsonl"
    shard_b = tmp_path / "scores_b.jsonl"

    _write_scored_pairs(shard_a, [duplicate_pair])
    _write_scored_pairs(shard_b, [duplicate_pair])

    with pytest.raises(ValueError, match="Duplicate pair"):
        merge_scored_shards(
            input_paths=[shard_a, shard_b],
            output_path=tmp_path / "merged_scores.jsonl",
        )
