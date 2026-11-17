"""Aleth disk cache for extracted facets."""

import json
import logging
import shutil
from pathlib import Path

from pydantic import ValidationError

from genret.paper_parsing import Facets

logger = logging.getLogger(__name__)


class FacetCache:
    """Disk-backed cache for extracted facets.

    Stores one JSON file per paper at ``{cache_dir}/{corpusid}.json``.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, corpusid: int) -> Path:
        return self._dir / f"{corpusid}.json"

    def get(self, corpusid: int) -> Facets | None:
        """Load cached facets for a paper, or None on miss."""
        path = self._path(corpusid)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Facets.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            logger.warning("Corrupt cache entry %s, removing", path)
            path.unlink(missing_ok=True)
            return None

    def put(self, facets: Facets) -> None:
        """Write facets to the cache."""
        self._path(facets.corpusid).write_text(
            facets.model_dump_json(indent=2), encoding="utf-8"
        )

    def validate_on_startup(self) -> None:
        """Validate one cached file against the Facets schema.

        If invalid, deletes the entire cache directory and recreates it.
        """
        files = list(self._dir.glob("*.json"))
        if not files:
            return

        sample = files[0]
        try:
            data = json.loads(sample.read_text(encoding="utf-8"))
            Facets.model_validate(data)
            logger.info(
                "Aleth cache validated (%d entries in %s)", len(files), self._dir
            )
        except (json.JSONDecodeError, ValidationError):
            logger.warning(
                "Aleth cache validation failed on %s, clearing cache", sample
            )
            shutil.rmtree(self._dir)
            self._dir.mkdir(parents=True, exist_ok=True)
