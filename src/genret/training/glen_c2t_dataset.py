# pyright: basic
"""C2T-ID training dataset: per-bullet input, C2T-ID textual target.

Mirrors the structure of :mod:`glen_bullet_dataset` (78lp / D9 recipe):
one training example per (paper_a bullet, paper_b, facet) row from the
retrieval pairs file. The only differences from D9 are:

- Target is paper_b's pre-built C2T-ID string for ``facet``
  (a hyphen-joined keyword sequence like
  ``"attention-transformer-language model-question-answer-retrieval"``).
- The C2T-ID is the same for all of paper_b's bullets — there is one
  C2T-ID per (paper, facet), not one per (paper, facet, bullet).

The C2T-ID strings are loaded from a directory built by
``build-c2t-index`` (see :mod:`c2t_index`). At inference, the same
directory's trie + ``code_to_papers`` mapping powers constrained beam
search via the existing GLEN decoder.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from genret.paper_parsing import (
    FacetEvidence,
    FacetRecordType,
    Facets,
    evidence_from_type,
    facets_from_type,
)
from genret.training.c2t_index import load_paper_c2t_docids
from genret.training.codebook_data import FACET_NAMES
from genret.training.data import TrainingExample
from genret.training.glen_bullet_dataset import (
    _build_bullet_messages,
    _render_bullet,
)
from genret.types import CorpusId
from genret.utils import iter_jsonl

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class GlenC2TDatasetConfig:
    """Configuration for the per-bullet C2T-ID training dataset."""

    pairs_path: Path
    """Training pairs JSONL (only retrieval rows are used)."""
    facets_path: Path
    """JSONL with per-paper facet bullets (and optional evidence)."""
    paper_graphs_path: Path
    """JSONL with per-paper metadata; must include 'title' and 'corpusid'."""
    c2t_dir: Path
    """Directory produced by ``build-c2t-index`` containing
    ``paper_c2t_docids.jsonl`` and the trie/code-to-papers files."""
    facets_type: FacetRecordType = FacetRecordType.STANDARD
    split_ids: set[CorpusId] | None = None
    """If set, restrict to pairs where ``paper_a`` is in this split."""
    facets: tuple[str, ...] = FACET_NAMES


class GlenC2TDataset:
    """Per-bullet input, C2T-ID textual target."""

    def __init__(
        self,
        config: GlenC2TDatasetConfig,
        *,
        rng: random.Random,
    ) -> None:
        self._examples = _build_c2t_examples(config, rng=rng)

    def __len__(self) -> int:  # pragma: no cover
        """Return number of training examples."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> TrainingExample:  # pragma: no cover
        """Return the training example at index ``idx``."""
        return self._examples[idx]


def _load_titles(paper_graphs_path: Path) -> dict[CorpusId, str]:
    """Return {corpusid: title} for all papers with a non-empty title."""
    titles: dict[CorpusId, str] = {}
    for record in iter_jsonl(paper_graphs_path):
        cid = CorpusId(int(record["corpusid"]))
        title = record.get("title")
        if isinstance(title, str) and title.strip():
            titles[cid] = title.strip()
    return titles


def _build_c2t_examples(
    config: GlenC2TDatasetConfig,
    *,
    rng: random.Random,
) -> list[TrainingExample]:
    """Expand retrieval pairs into one row per paper_a bullet."""
    _ = rng  # determinism for callers that pass an RNG; not used here yet.
    facets_map: dict[CorpusId, Facets] = {}
    evidence_map: dict[CorpusId, FacetEvidence] = {}
    for raw in iter_jsonl(config.facets_path):
        f = facets_from_type(raw, config.facets_type)
        facets_map[f.corpusid] = f
        if ev := evidence_from_type(raw, config.facets_type):
            evidence_map[f.corpusid] = ev

    titles = _load_titles(config.paper_graphs_path)
    logger.info("Loaded %d titles", len(titles))

    c2t_docids = load_paper_c2t_docids(config.c2t_dir)
    n_c2t_papers = {f: len(d) for f, d in c2t_docids.items()}
    logger.info("Loaded C2T-IDs per facet: %s", n_c2t_papers)

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
        if facets_a is None:
            skipped += 1
            continue

        # Look up paper_b's C2T-ID for this facet.
        target_docid = c2t_docids.get(facet, {}).get(paper_b)
        if not target_docid:
            skipped += 1
            continue

        title_a = titles.get(paper_a, "")
        evidence_a = evidence_map.get(paper_a)

        source_indices: list[int] = list(range(len(getattr(facets_a, facet))))
        for i in source_indices:
            bullet_text = _render_bullet(facets_a, facet, i, evidence_a)
            if not bullet_text:
                continue
            messages = _build_bullet_messages(facet, title_a, bullet_text)
            examples.append(
                TrainingExample(prompt_messages=messages, target_docid=target_docid)
            )

    if skipped:
        logger.info(
            "Skipped %d pairs (missing facets, missing C2T-ID for paper_b)",
            skipped,
        )

    logger.info("GlenC2TDataset: %d examples", len(examples))
    return examples


__all__ = ["GlenC2TDataset", "GlenC2TDatasetConfig"]
