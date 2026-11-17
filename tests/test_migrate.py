"""Tests for pipeline migrate module."""

import json
from pathlib import Path
from typing import Any

import pytest

from genret.pipeline.migrate import (
    MigrateConfig,
    _filter_jsonl_file,
    run_migrate_processed,
)
from genret.types import CorpusId


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as plain JSONL (no compression)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record) + "\n" for record in records)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read records from a plain JSONL file."""
    results: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def _write_split_files(splits_dir: Path, ids: list[int]) -> None:
    """Write train/dev/test.txt with the given IDs split across them."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    # Put all IDs in train.txt for simplicity
    with open(splits_dir / "train.txt", "w") as f:
        f.writelines(f"{i}\n" for i in ids)
    # Empty dev and test
    (splits_dir / "dev.txt").write_text("")
    (splits_dir / "test.txt").write_text("")


# === _filter_jsonl_file ===


class TestFilterJsonlFile:
    """Tests for _filter_jsonl_file."""

    def test_filters_by_corpusid(self, tmp_path: Path) -> None:
        """Records with corpusid in target_ids are kept, others dropped."""
        source = tmp_path / "source.jsonl"
        target = tmp_path / "output.jsonl"

        records = [
            {"corpusid": 1, "data": "a"},
            {"corpusid": 2, "data": "b"},
            {"corpusid": 3, "data": "c"},
            {"corpusid": 4, "data": "d"},
        ]
        _write_jsonl(source, records)

        result = _filter_jsonl_file(source, target, {CorpusId(1), CorpusId(3)})

        assert result.kept == 2
        assert result.dropped == 2

        output_records = _read_jsonl(target)
        assert len(output_records) == 2
        assert output_records[0]["corpusid"] == 1
        assert output_records[1]["corpusid"] == 3

    def test_custom_id_field(self, tmp_path: Path) -> None:
        """Filtering works with a custom ID field name."""
        source = tmp_path / "source.jsonl"
        target = tmp_path / "output.jsonl"

        records = [
            {"paper_id": 10, "title": "x"},
            {"paper_id": 20, "title": "y"},
        ]
        _write_jsonl(source, records)

        result = _filter_jsonl_file(source, target, {CorpusId(20)}, id_field="paper_id")

        assert result.kept == 1
        assert result.dropped == 1

    def test_empty_source(self, tmp_path: Path) -> None:
        """An empty source file produces an empty output."""
        source = tmp_path / "source.jsonl"
        target = tmp_path / "output.jsonl"
        source.write_text("")

        result = _filter_jsonl_file(source, target, {CorpusId(1), CorpusId(2)})

        assert result.kept == 0
        assert result.dropped == 0

    def test_no_matching_ids(self, tmp_path: Path) -> None:
        """When no IDs match, all records are dropped."""
        source = tmp_path / "source.jsonl"
        target = tmp_path / "output.jsonl"

        _write_jsonl(source, [{"corpusid": 1}, {"corpusid": 2}])

        result = _filter_jsonl_file(source, target, {CorpusId(99)})

        assert result.kept == 0
        assert result.dropped == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Target parent directories are created if they don't exist."""
        source = tmp_path / "source.jsonl"
        target = tmp_path / "nested" / "deep" / "output.jsonl"

        _write_jsonl(source, [{"corpusid": 1}])

        result = _filter_jsonl_file(source, target, {CorpusId(1)})

        assert result.kept == 1
        assert target.exists()


# === run_migrate_processed ===


class TestRunMigrateProcessed:
    """Tests for run_migrate_processed end-to-end."""

    def _setup_dirs(self, tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
        """Set up source and target directories with test data.

        Returns:
            Tuple of (source_dir, target_dir, source_records).
        """
        source_dir = tmp_path / "source_splits"
        target_dir = tmp_path / "target_splits"

        # Source has processed/ with facets and terms
        source_processed = source_dir / "processed"

        facet_records: list[dict[str, object]] = [
            {"corpusid": 1, "facets": ["a"]},
            {"corpusid": 2, "facets": ["b"]},
            {"corpusid": 3, "facets": ["c"]},
            {"corpusid": 4, "facets": ["d"]},
            {"corpusid": 5, "facets": ["e"]},
        ]
        _write_jsonl(source_processed / "facets.jsonl", facet_records)

        term_records: list[dict[str, object]] = [
            {"corpusid": 1, "terms": ["x"]},
            {"corpusid": 2, "terms": ["y"]},
            {"corpusid": 3, "terms": ["z"]},
        ]
        _write_jsonl(source_processed / "terms.jsonl", term_records)

        # Target has split files with IDs 1, 3
        _write_split_files(target_dir, [1, 3])

        return source_dir, target_dir, facet_records

    def test_migrates_matching_records(self, tmp_path: Path) -> None:
        """Records matching target IDs are migrated."""
        source_dir, target_dir, _ = self._setup_dirs(tmp_path)

        config = MigrateConfig(
            source_dir=source_dir, target_dir=target_dir, force=False
        )
        results = run_migrate_processed(config)

        assert len(results) == 2

        facet_result = results[0]
        assert facet_result.kept == 2
        assert facet_result.dropped == 3

        term_result = results[1]
        assert term_result.kept == 2
        assert term_result.dropped == 1

    def test_skips_existing_without_force(self, tmp_path: Path) -> None:
        """Existing target files are skipped when force=False."""
        source_dir, target_dir, _ = self._setup_dirs(tmp_path)

        # Pre-create target facets file
        target_processed = target_dir / "processed"
        _write_jsonl(
            target_processed / "facets.jsonl",
            [{"corpusid": 1, "facets": ["old"]}],
        )

        config = MigrateConfig(
            source_dir=source_dir, target_dir=target_dir, force=False
        )
        results = run_migrate_processed(config)

        # Only terms should be migrated (facets skipped)
        assert len(results) == 1
        assert results[0].filename.startswith("terms")

    def test_overwrites_with_force(self, tmp_path: Path) -> None:
        """Existing target files are overwritten when force=True."""
        source_dir, target_dir, _ = self._setup_dirs(tmp_path)

        # Pre-create target facets file
        target_processed = target_dir / "processed"
        _write_jsonl(
            target_processed / "facets.jsonl",
            [{"corpusid": 99, "facets": ["old"]}],
        )

        config = MigrateConfig(source_dir=source_dir, target_dir=target_dir, force=True)
        results = run_migrate_processed(config)

        # Both files should be migrated
        assert len(results) == 2
        assert results[0].kept == 2  # facets: IDs 1 and 3

    def test_missing_source_exits(self, tmp_path: Path) -> None:
        """Missing source processed/ directory causes exit."""
        source_dir = tmp_path / "no_source"
        target_dir = tmp_path / "target"
        _write_split_files(target_dir, [1])

        config = MigrateConfig(
            source_dir=source_dir, target_dir=target_dir, force=False
        )

        with pytest.raises(SystemExit):
            run_migrate_processed(config)

    def test_empty_splits_exits(self, tmp_path: Path) -> None:
        """Empty split files cause exit."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"

        (source_dir / "processed").mkdir(parents=True)
        target_dir.mkdir(parents=True)
        (target_dir / "train.txt").write_text("")
        (target_dir / "dev.txt").write_text("")
        (target_dir / "test.txt").write_text("")

        config = MigrateConfig(
            source_dir=source_dir, target_dir=target_dir, force=False
        )

        with pytest.raises(SystemExit):
            run_migrate_processed(config)

    def test_skips_missing_source_files(self, tmp_path: Path) -> None:
        """Source files that don't exist are skipped gracefully."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"

        # Only create facets, not terms or paper_graphs
        _write_jsonl(
            source_dir / "processed" / "facets.jsonl",
            [{"corpusid": 1, "facets": ["a"]}],
        )
        _write_split_files(target_dir, [1])

        config = MigrateConfig(
            source_dir=source_dir, target_dir=target_dir, force=False
        )
        results = run_migrate_processed(config)

        assert len(results) == 1
        assert results[0].kept == 1
