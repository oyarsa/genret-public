"""Unit tests for genret.utils file-path helpers."""

from pathlib import Path

import pytest

from genret.utils import glob_jsonl, resolve_jsonl


class TestGlobJsonl:
    """Tests for glob_jsonl, verifying deduplication and priority."""

    def test_returns_empty_for_empty_directory(self, tmp_path: Path) -> None:
        """Should return an empty list when the directory has no JSONL files."""
        assert glob_jsonl(tmp_path) == []

    def test_finds_plain_jsonl(self, tmp_path: Path) -> None:
        """Should return plain .jsonl files."""
        (tmp_path / "a.jsonl").touch()
        assert glob_jsonl(tmp_path) == [tmp_path / "a.jsonl"]

    def test_finds_gz(self, tmp_path: Path) -> None:
        """Should return .jsonl.gz files."""
        (tmp_path / "a.jsonl.gz").touch()
        assert glob_jsonl(tmp_path) == [tmp_path / "a.jsonl.gz"]

    def test_finds_zst(self, tmp_path: Path) -> None:
        """Should return .jsonl.zst files."""
        (tmp_path / "a.jsonl.zst").touch()
        assert glob_jsonl(tmp_path) == [tmp_path / "a.jsonl.zst"]

    def test_zst_beats_gz(self, tmp_path: Path) -> None:
        """When both .zst and .gz exist for the same base, return only .zst."""
        (tmp_path / "a.jsonl.gz").touch()
        (tmp_path / "a.jsonl.zst").touch()
        assert glob_jsonl(tmp_path) == [tmp_path / "a.jsonl.zst"]

    def test_zst_beats_plain(self, tmp_path: Path) -> None:
        """When both .zst and plain exist for the same base, return only .zst."""
        (tmp_path / "a.jsonl").touch()
        (tmp_path / "a.jsonl.zst").touch()
        assert glob_jsonl(tmp_path) == [tmp_path / "a.jsonl.zst"]

    def test_gz_beats_plain(self, tmp_path: Path) -> None:
        """When both .gz and plain exist for the same base, return only .gz."""
        (tmp_path / "a.jsonl").touch()
        (tmp_path / "a.jsonl.gz").touch()
        assert glob_jsonl(tmp_path) == [tmp_path / "a.jsonl.gz"]

    def test_zst_beats_all_three(self, tmp_path: Path) -> None:
        """When all three variants exist, return only .zst."""
        (tmp_path / "a.jsonl").touch()
        (tmp_path / "a.jsonl.gz").touch()
        (tmp_path / "a.jsonl.zst").touch()
        assert glob_jsonl(tmp_path) == [tmp_path / "a.jsonl.zst"]

    def test_multiple_bases_deduplicated_independently(self, tmp_path: Path) -> None:
        """Each base name is deduplicated independently."""
        (tmp_path / "a.jsonl.gz").touch()
        (tmp_path / "a.jsonl.zst").touch()
        (tmp_path / "b.jsonl.gz").touch()
        (tmp_path / "c.jsonl").touch()
        result = glob_jsonl(tmp_path)
        assert result == [
            tmp_path / "a.jsonl.zst",
            tmp_path / "b.jsonl.gz",
            tmp_path / "c.jsonl",
        ]

    def test_result_is_sorted_by_name(self, tmp_path: Path) -> None:
        """Results should be sorted alphabetically by path."""
        (tmp_path / "z.jsonl").touch()
        (tmp_path / "a.jsonl").touch()
        (tmp_path / "m.jsonl").touch()
        result = glob_jsonl(tmp_path)
        assert result == [
            tmp_path / "a.jsonl",
            tmp_path / "m.jsonl",
            tmp_path / "z.jsonl",
        ]

    def test_ignores_non_jsonl_gz_files(self, tmp_path: Path) -> None:
        """Should not treat .txt.gz or bare .gz files as JSONL."""
        (tmp_path / "data.txt.gz").touch()
        (tmp_path / "data.gz").touch()
        (tmp_path / "a.jsonl").touch()
        assert glob_jsonl(tmp_path) == [tmp_path / "a.jsonl"]


class TestResolveJsonl:
    """Tests for resolve_jsonl, verifying precedence and default behaviour."""

    def test_prefers_zst_over_gz(self, tmp_path: Path) -> None:
        """When both .zst and .gz exist, .zst is returned."""
        gz = tmp_path / "f.jsonl.gz"
        zst = tmp_path / "f.jsonl.zst"
        gz.touch()
        zst.touch()
        assert resolve_jsonl(tmp_path / "f.jsonl.gz") == zst
        assert resolve_jsonl(tmp_path / "f.jsonl") == zst
        assert resolve_jsonl(tmp_path / "f.jsonl.zst") == zst

    def test_prefers_gz_over_plain(self, tmp_path: Path) -> None:
        """When both .gz and plain exist (no .zst), .gz is returned."""
        gz = tmp_path / "f.jsonl.gz"
        plain = tmp_path / "f.jsonl"
        gz.touch()
        plain.touch()
        assert resolve_jsonl(tmp_path / "f.jsonl") == gz
        assert resolve_jsonl(tmp_path / "f.jsonl.gz") == gz

    def test_returns_plain_when_only_plain_exists(self, tmp_path: Path) -> None:
        """When only the plain variant exists, it is returned."""
        plain = tmp_path / "f.jsonl"
        plain.touch()
        assert resolve_jsonl(tmp_path / "f.jsonl") == plain
        assert resolve_jsonl(tmp_path / "f.jsonl.gz") == plain

    def test_returns_zst_default_for_new_file(self, tmp_path: Path) -> None:
        """When no variant exists, defaults to the .zst path for new-file creation."""
        result = resolve_jsonl(tmp_path / "new.jsonl.gz")
        assert result == tmp_path / "new.jsonl.zst"

    def test_returns_zst_default_from_plain_path(self, tmp_path: Path) -> None:
        """Default also applies when the input path has no compression suffix."""
        result = resolve_jsonl(tmp_path / "new.jsonl")
        assert result == tmp_path / "new.jsonl.zst"

    def test_input_with_zst_suffix(self, tmp_path: Path) -> None:
        """Input path ending in .zst is handled correctly."""
        zst = tmp_path / "f.jsonl.zst"
        zst.touch()
        assert resolve_jsonl(zst) == zst

    @pytest.mark.parametrize(
        "input_name",
        ["f.jsonl", "f.jsonl.gz", "f.jsonl.zst"],
    )
    def test_all_three_present_always_returns_zst(
        self, tmp_path: Path, input_name: str
    ) -> None:
        """Regardless of input suffix, .zst wins when all three variants exist."""
        (tmp_path / "f.jsonl").touch()
        (tmp_path / "f.jsonl.gz").touch()
        (tmp_path / "f.jsonl.zst").touch()
        assert resolve_jsonl(tmp_path / input_name) == tmp_path / "f.jsonl.zst"
