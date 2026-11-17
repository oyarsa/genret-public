"""On-disk embedding cache and cache-backed embedding computation."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Self

import numpy as np

from genret._shed.numpy import savez as np_savez
from genret.llm.encoder import Encoder, Matrix


class EmbeddingCache:
    """On-disk cache for paragraph and facet embeddings.

    Uses a single .npz file per cache type (paragraphs / facets) with
    string keys `str(corpusid)`.  Loads into memory once, flushes new
    entries on `save()`.

    When opened with ``read_only=True``, newly computed embeddings are kept
    in memory for the duration of the run but never written back to disk.
    This is safe for shared caches (e.g. paragraph embeddings) that may be
    accessed by multiple concurrent evaluation runs.
    """

    def __init__(self, path: Path, dimensions: int, *, read_only: bool = False) -> None:
        """Load or create an embedding cache at the given path.

        Args:
            path: Path to the .npz cache file.
            dimensions: Expected embedding dimensionality.
            read_only: If True, never write back to disk.  Newly computed
                embeddings are still cached in memory for the current run.
        """
        self._path = path
        self._dimensions = dimensions
        self._read_only = read_only
        self._data: dict[str, Matrix] = {}
        self._dirty = False
        if path.exists():
            with np.load(path) as npz:
                for key in npz.files:
                    self._data[key] = npz[key]

    def get(self, corpusid: int) -> Matrix | None:
        """Return cached embeddings or None."""
        key = str(corpusid)
        return self._data.get(key)

    def put(self, corpusid: int, embeddings: Matrix) -> None:
        """Store embeddings in memory (flushed to disk on `save` unless read-only)."""
        self._data[str(corpusid)] = embeddings
        if not self._read_only:
            self._dirty = True

    def save(self) -> None:
        """Flush to disk if anything changed (no-op when read-only)."""
        if self._dirty:
            np_savez(self._path, self._data)
            self._dirty = False

    @classmethod
    @contextmanager
    def open(
        cls, path: Path, dimensions: int, *, read_only: bool = False
    ) -> Generator[Self]:
        """Load or create an embedding cache at the given path. Auto-saves on exit.

        Args:
            path: Path to the .npz cache file.
            dimensions: Expected embedding dimensionality.
            read_only: If True, never write back to disk.  Safe to use
                concurrently across multiple evaluation runs.
        """
        cache = cls(path, dimensions, read_only=read_only)
        try:
            yield cache
        finally:
            cache.save()

    @property
    def dimensions(self) -> int:
        """Embedding dimensionality."""
        return self._dimensions

    def empty_matrix(self) -> Matrix:
        """Return a (0, dim) matrix for papers with no content."""
        return np.zeros((0, self._dimensions), dtype=np.float32)

    def put_empty(self, corpusid: int) -> Matrix:
        """Store empty matrix for `corpusid` (not flushed until `save`)."""
        empty = self.empty_matrix()
        self.put(corpusid, empty)
        return empty


async def get_cached_embeddings(
    texts: list[str],
    corpusid: int,
    encoder: Encoder,
    cache: EmbeddingCache,
) -> Matrix:
    """Load embeddings from cache, or compute and cache them.

    If the cached entry exists but has a different row count than ``texts``,
    it is treated as stale (e.g. from a previous extraction run with a
    different item count) and recomputed.
    """
    cached = cache.get(corpusid)
    if cached is not None and cached.shape[0] == len(texts):
        return cached

    if not texts:
        return cache.put_empty(corpusid)

    embeddings = await encoder.batch_encode(texts)
    cache.put(corpusid, embeddings)
    return embeddings
