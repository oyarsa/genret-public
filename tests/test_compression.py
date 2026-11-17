"""Tests for open_auto and AsyncLineWriter compression handling.

Covers roundtrip read/write for all supported formats (.jsonl, .jsonl.gz,
.jsonl.zst) and verifies that append mode works correctly across formats.

Note on zstd append: appending opens the file in binary-append mode and
writes a second independent zstd frame.  python-zstandard's stream_reader
handles concatenated frames correctly, so resume works — provided the
original file contains only complete frames.  If the first write session was
interrupted (process killed, timeout mid-stream), the resulting incomplete
frame will cause ZstdError on the next read regardless of the appended data.
"""

import asyncio
from compression.zstd import ZstdError
from pathlib import Path

import pytest

from genret.utils import AsyncLineWriter, LineWriter, OpenMode, open_auto

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_lines(path: Path, lines: list[str], mode: OpenMode = "w") -> None:
    """Write *lines* to *path* via open_auto in text mode."""
    with LineWriter.open(path, mode) as writer:
        writer.write_lines(lines)


def _read_lines(path: Path) -> list[str]:
    """Read all non-empty lines from *path* via open_auto."""
    with open_auto(path, "r") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


async def _async_write_lines(
    path: Path, lines: list[str], mode: OpenMode = "w"
) -> None:
    """Write *lines* to *path* via AsyncLineWriter."""
    async with AsyncLineWriter.open(path, mode) as writer:
        await writer.write_lines(lines)


# ---------------------------------------------------------------------------
# open_auto roundtrip
# ---------------------------------------------------------------------------


class TestOpenAutoRoundtrip:
    """open_auto read/write roundtrip across all compression formats."""

    @pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz", ".jsonl.zst"])
    def test_write_read_roundtrip(self, tmp_path: Path, suffix: str) -> None:
        """Writing and reading back should return the original lines."""
        path = tmp_path / f"data{suffix}"
        lines = ["line1", "line2", "line3"]
        _write_lines(path, lines)
        assert _read_lines(path) == lines

    @pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz", ".jsonl.zst"])
    def test_overwrite_replaces_content(self, tmp_path: Path, suffix: str) -> None:
        """Opening in 'w' mode should replace any existing content."""
        path = tmp_path / f"data{suffix}"
        _write_lines(path, ["old1", "old2"])
        _write_lines(path, ["new1"])
        assert _read_lines(path) == ["new1"]

    def test_empty_file_reads_empty(self, tmp_path: Path) -> None:
        """Reading an empty plain file should return no lines."""
        path = tmp_path / "empty.jsonl"
        _write_lines(path, [])
        assert _read_lines(path) == []


# ---------------------------------------------------------------------------
# open_auto append
# ---------------------------------------------------------------------------


class TestOpenAutoAppend:
    """open_auto append behaviour across supported formats."""

    def test_append_plain_readable(self, tmp_path: Path) -> None:
        """Appending to a plain file and reading back returns all lines."""
        path = tmp_path / "data.jsonl"
        _write_lines(path, ["a", "b"])
        _write_lines(path, ["c", "d"], mode="a")
        assert _read_lines(path) == ["a", "b", "c", "d"]

    def test_append_gz_readable(self, tmp_path: Path) -> None:
        """Appending to .gz and reading back returns all lines.

        gzip supports concatenated streams, so this works correctly.
        """
        path = tmp_path / "data.jsonl.gz"
        _write_lines(path, ["a", "b"])
        _write_lines(path, ["c", "d"], mode="a")
        assert _read_lines(path) == ["a", "b", "c", "d"]

    def test_append_zst_readable(self, tmp_path: Path) -> None:
        """Appending to .zst and reading back returns all lines.

        Each append call writes a new independent zstd frame.
        python-zstandard's stream_reader handles concatenated frames correctly.
        """
        path = tmp_path / "data.jsonl.zst"
        _write_lines(path, ["a", "b"])
        _write_lines(path, ["c", "d"], mode="a")
        assert _read_lines(path) == ["a", "b", "c", "d"]

    def test_append_zst_incomplete_first_frame_raises(self, tmp_path: Path) -> None:
        """A truncated first frame causes ZstdError even if subsequent frames are valid.

        Simulates a run that was killed mid-write: the file contains raw bytes
        that form the start of a valid zstd frame but were never finalised.
        A resume in append mode adds a complete second frame, but reading the
        file still fails because the first frame is broken.  Unlike gz, zstd
        fails loudly rather than silently dropping data.
        """
        path = tmp_path / "data.jsonl.zst"

        # Simulate a mid-write kill: write just the zstd frame magic bytes with
        # a truncated header — enough to look like a zstd file but unreadable.
        # zstd magic = 0xFD2FB528 (little-endian: 28 B5 2F FD)
        path.write_bytes(b"\x28\xb5\x2f\xfd\xff\xff\xff\xff\xff\xff\xff\xff")

        # Append a complete second frame (simulates what resume does)
        _write_lines(path, ["recovery line"], mode="a")

        with pytest.raises(ZstdError):
            _read_lines(path)


# ---------------------------------------------------------------------------
# AsyncLineWriter roundtrip
# ---------------------------------------------------------------------------


class TestAsyncLineWriterRoundtrip:
    """AsyncLineWriter read/write roundtrip across all compression formats."""

    @pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz", ".jsonl.zst"])
    def test_write_read_roundtrip(self, tmp_path: Path, suffix: str) -> None:
        """Writing via AsyncLineWriter and reading back via open_auto returns all lines."""
        path = tmp_path / f"data{suffix}"
        asyncio.run(_async_write_lines(path, ["line1", "line2", "line3"]))
        assert _read_lines(path) == ["line1", "line2", "line3"]

    @pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz", ".jsonl.zst"])
    def test_overwrite_replaces_content(self, tmp_path: Path, suffix: str) -> None:
        """Opening AsyncLineWriter in 'w' mode replaces any existing content."""
        path = tmp_path / f"data{suffix}"
        asyncio.run(_async_write_lines(path, ["old1", "old2"]))
        asyncio.run(_async_write_lines(path, ["new1"]))
        assert _read_lines(path) == ["new1"]

    def test_write_line_strips_trailing_newline(self, tmp_path: Path) -> None:
        """write_line should strip any trailing newline before appending one."""
        path = tmp_path / "data.jsonl"
        asyncio.run(_async_write_lines(path, ["line\n", "other\r\n"]))
        assert _read_lines(path) == ["line", "other"]


# ---------------------------------------------------------------------------
# AsyncLineWriter append
# ---------------------------------------------------------------------------


class TestAsyncLineWriterAppend:
    """AsyncLineWriter append behaviour — gz works, zst is currently broken."""

    def test_append_plain_readable(self, tmp_path: Path) -> None:
        """Appending to a plain file via AsyncLineWriter returns all lines."""
        path = tmp_path / "data.jsonl"
        asyncio.run(_async_write_lines(path, ["a", "b"]))
        asyncio.run(_async_write_lines(path, ["c", "d"], mode="a"))
        assert _read_lines(path) == ["a", "b", "c", "d"]

    def test_append_gz_readable(self, tmp_path: Path) -> None:
        """Appending to .gz via AsyncLineWriter and reading back returns all lines."""
        path = tmp_path / "data.jsonl.gz"
        asyncio.run(_async_write_lines(path, ["a", "b"]))
        asyncio.run(_async_write_lines(path, ["c", "d"], mode="a"))
        assert _read_lines(path) == ["a", "b", "c", "d"]

    def test_append_zst_readable(self, tmp_path: Path) -> None:
        """Appending to .zst via AsyncLineWriter and reading back returns all lines."""
        path = tmp_path / "data.jsonl.zst"
        asyncio.run(_async_write_lines(path, ["a", "b"]))
        asyncio.run(_async_write_lines(path, ["c", "d"], mode="a"))
        assert _read_lines(path) == ["a", "b", "c", "d"]
