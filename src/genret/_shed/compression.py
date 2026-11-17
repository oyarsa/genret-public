"""Typed wrappers for compressed file openers.

``gzip.open`` and ``compression.zstd.open`` return types whose stubs are
overly broad unions. These wrappers narrow the return to ``IO[str]``
so callers don't need casts.
"""

import compression.zstd
import gzip
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Literal


@contextmanager
def open_gz(
    path: Path, mode: Literal["rt", "wt", "at"], *, encoding: str = "utf-8"
) -> Generator[IO[str]]:
    """Open a gzip file in text mode.

    Args:
        path: File path.
        mode: Text-mode open string.
        encoding: Text encoding.

    Yields:
        Text-mode file handle.
    """
    with gzip.open(path, mode, encoding=encoding) as f:
        yield f


@contextmanager
def open_zstd(
    path: Path, mode: Literal["rt", "wt", "at"], *, encoding: str = "utf-8"
) -> Generator[IO[str]]:
    """Open a zstandard file in text mode.

    Args:
        path: File path.
        mode: Text-mode open string.
        encoding: Text encoding.

    Yields:
        Text-mode file handle.
    """
    with compression.zstd.open(path, mode, encoding=encoding) as f:
        yield f
