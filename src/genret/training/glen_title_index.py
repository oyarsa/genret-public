# pyright: basic
"""Title-generation GLEN index and dataset (D5 sanity check).

Builds a GlenIndex where each paper's DocID is its tokenised title rather than
its facet-text bullets.  This serves as a calibration baseline: title DocIDs are
short, unambiguous, and unique per paper, so a model that can generate them is
demonstrably learning to route queries to relevant papers rather than memorising
trie paths.

The index stores the same title trie under all four FACET_NAMES keys so that
``decode_per_facet_glen`` and ``evaluate_glen_retrieval`` require no changes.
Training examples use ``kind="title"`` to trigger the title-specific system
prompt from ``_build_messages``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from genret.paper_parsing import FacetRecordType, Facets, facets_from_type
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_dataset import _build_messages, _facet_text
from genret.training.data import TrainingExample
from genret.training.glen_docids import (
    GlenIndex,
    _build_trie,
    extract_docid_tokens,
)
from genret.types import CorpusId
from genret.utils import iter_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def normalise_title(title: str) -> str:
    """Lowercase and collapse whitespace in a paper title."""
    return " ".join(title.lower().split())


def build_title_index(
    paper_graphs_path: Path,
    tokeniser: PreTrainedTokenizerBase,
    *,
    max_tokens: int = 32,
) -> GlenIndex:
    """Build a GlenIndex keyed by paper title rather than facet-text DocIDs.

    Reads ``paper_graphs_path`` (JSONL with ``corpusid`` and ``title`` fields).
    Each paper with a non-empty title gets one entry: ``normalise_title(title)``
    tokenised to at most ``max_tokens`` token IDs.  The same trie is stored under
    all four FACET_NAMES keys so existing decode/eval code works unchanged.
    """
    raw: dict[str, dict[tuple[int, ...], list[CorpusId]]] = {f: {} for f in FACET_NAMES}
    n_papers = 0
    n_empty = 0

    for record in iter_jsonl(paper_graphs_path):
        cid = CorpusId(int(record["corpusid"]))
        title = record.get("title")
        if not isinstance(title, str) or not title.strip():
            n_empty += 1
            continue
        toks = extract_docid_tokens(
            normalise_title(title), tokeniser, max_tokens=max_tokens
        )
        if not toks:
            n_empty += 1
            continue
        n_papers += 1
        for facet in FACET_NAMES:
            raw[facet].setdefault(tuple(toks), []).append(cid)

    tries = {facet: _build_trie(list(raw[facet].keys())) for facet in FACET_NAMES}

    total_leaves = sum(len(v) for v in raw.values()) // len(FACET_NAMES)
    logger.info(
        "Title index: %d papers, %d unique title sequences (%d skipped)",
        n_papers,
        total_leaves,
        n_empty,
    )
    return GlenIndex(code_to_papers=raw, trie=tries)


@dataclass(frozen=True, kw_only=True)
class GlenTitleDatasetConfig:
    """Configuration for the title-generation training dataset (D5)."""

    pairs_path: Path
    """Training pairs JSONL from build-codebook-data (only retrieval rows used)."""
    facets_path: Path
    """JSONL with per-paper facet bullets."""
    paper_graphs_path: Path
    """JSONL with per-paper metadata; must include 'corpusid' and 'title' fields."""
    facets_type: FacetRecordType = FacetRecordType.STANDARD
    split_ids: set[CorpusId] | None = None
    """If set, restrict to pairs where paper_a is in this split."""
    facets: tuple[str, ...] = FACET_NAMES
    """Restrict examples to these facets. Defaults to all four."""


class GlenTitleDataset:
    """Dataset of retrieval examples whose targets are paper titles (D5)."""

    def __init__(
        self,
        config: GlenTitleDatasetConfig,
        *,
        rng: random.Random,
    ) -> None:
        self._examples = _build_title_examples(config, rng=rng)

    def __len__(self) -> int:  # pragma: no cover
        """Return number of training examples."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> TrainingExample:  # pragma: no cover
        """Return the training example at index ``idx``."""
        return self._examples[idx]


def _load_titles(paper_graphs_path: Path) -> dict[CorpusId, str]:
    """Return {corpusid: normalised_title} for papers with non-empty titles."""
    titles: dict[CorpusId, str] = {}
    for record in iter_jsonl(paper_graphs_path):
        cid = CorpusId(int(record["corpusid"]))
        title = record.get("title")
        if isinstance(title, str) and title.strip():
            titles[cid] = normalise_title(title)
    return titles


def _build_title_examples(
    config: GlenTitleDatasetConfig,
    *,
    rng: random.Random,
) -> list[TrainingExample]:
    """Load pairs + facets + titles, build one example per retrieval pair."""
    facets_map: dict[CorpusId, Facets] = {}
    for raw in iter_jsonl(config.facets_path):
        f = facets_from_type(raw, config.facets_type)
        facets_map[f.corpusid] = f

    titles = _load_titles(config.paper_graphs_path)
    logger.info("Loaded %d titles from %s", len(titles), config.paper_graphs_path)

    examples: list[TrainingExample] = []
    skipped = 0
    split_ids = config.split_ids

    for row in iter_jsonl(config.pairs_path):
        if str(row.get("kind", "")) != "retrieval":
            continue

        paper_a = CorpusId(int(row["paper_a"]))
        paper_b = CorpusId(int(row["paper_b"]))
        facet = str(row["facet"])

        if facet not in FACET_NAMES:
            continue
        if facet not in config.facets:
            continue
        if split_ids is not None and paper_a not in split_ids:
            continue

        facets_a = facets_map.get(paper_a)
        title_b = titles.get(paper_b)
        if facets_a is None or not title_b:
            skipped += 1
            continue

        input_text = _facet_text(facets_a, facet, None)
        if not input_text:
            skipped += 1
            continue

        examples.append(
            TrainingExample(
                prompt_messages=_build_messages(facet, input_text, kind="title"),
                target_docid=title_b,
                kind="title",
            )
        )

    if skipped:
        logger.info("Skipped %d title-gen pairs (missing facets or title)", skipped)

    rng.shuffle(examples)
    logger.info("GlenTitleDataset: %d retrieval examples", len(examples))
    return examples


__all__ = [
    "GlenTitleDataset",
    "GlenTitleDatasetConfig",
    "build_title_index",
    "normalise_title",
]
