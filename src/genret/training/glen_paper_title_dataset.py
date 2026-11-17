# pyright: basic

"""Paper-level title-DocID GLEN dataset (Variant 1 alone).

Same input representation as :mod:`glen_paper_marginal_dataset` (paper_a's
title + abstract) and same DocID space (paper_b's normalised title), but
**single-target cross-entropy** instead of multi-positive marginal LL.

One training row per ``(paper_a, paper_b)`` gold retrieval link, deduplicated
across facets. The target is paper_b's title. Standard ``TrainingExample`` /
``collate_fn`` plumbing reuses the existing trainer pipeline.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from genret.paper_parsing import FacetRecordType
from genret.training.codebook_data import FACET_NAMES
from genret.training.data import TrainingExample
from genret.training.glen_title_index import normalise_title
from genret.types import CorpusId
from genret.utils import iter_jsonl

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a generative retrieval system. Given a research paper's "
    "title and abstract, generate the title of a related paper."
)


def _build_paper_messages(title: str, abstract: str) -> list[dict[str, str]]:
    """Return chat messages for a paper-level retrieval query."""
    user = f"Title: {title}"
    if abstract:
        user += f"\nAbstract: {abstract}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@dataclass(frozen=True, kw_only=True)
class GlenPaperTitleDatasetConfig:
    """Configuration for the paper-level single-target title dataset."""

    pairs_path: Path
    facets_path: Path
    paper_graphs_path: Path
    facets_type: FacetRecordType = FacetRecordType.STANDARD
    split_ids: set[CorpusId] | None = None
    facets: tuple[str, ...] = FACET_NAMES
    """Restrict gold-link collection to these facets (default: all four)."""
    max_abstract_chars: int = 1500


class GlenPaperTitleDataset:
    """Dataset of (paper_a title+abstract → paper_b title) training rows."""

    def __init__(
        self, config: GlenPaperTitleDatasetConfig, *, rng: random.Random
    ) -> None:
        self._examples = _build_examples(config, rng=rng)

    def __len__(self) -> int:  # pragma: no cover
        """Number of training rows."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> TrainingExample:  # pragma: no cover
        """Return the row at ``idx``."""
        return self._examples[idx]


def _load_paper_metadata(
    paper_graphs_path: Path, *, max_abstract_chars: int
) -> tuple[dict[CorpusId, str], dict[CorpusId, str]]:
    """Return ``(raw_titles, abstracts)`` keyed by corpus id."""
    titles: dict[CorpusId, str] = {}
    abstracts: dict[CorpusId, str] = {}
    for record in iter_jsonl(paper_graphs_path):
        cid = CorpusId(int(record["corpusid"]))
        title = record.get("title")
        if isinstance(title, str) and title.strip():
            titles[cid] = title.strip()
        abstract = record.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            abstracts[cid] = abstract.strip()[:max_abstract_chars]
    return titles, abstracts


def _build_examples(
    config: GlenPaperTitleDatasetConfig,
    *,
    rng: random.Random,
) -> list[TrainingExample]:
    """Emit one TrainingExample per deduplicated (paper_a, paper_b) gold link."""
    raw_titles, abstracts = _load_paper_metadata(
        config.paper_graphs_path, max_abstract_chars=config.max_abstract_chars
    )

    seen: dict[CorpusId, set[CorpusId]] = defaultdict(set)
    split_ids = config.split_ids
    for row in iter_jsonl(config.pairs_path):
        if str(row.get("kind", "")) != "retrieval":
            continue
        facet = str(row["facet"])
        if facet not in FACET_NAMES or facet not in config.facets:
            continue
        paper_a = CorpusId(int(row["paper_a"]))
        paper_b = CorpusId(int(row["paper_b"]))
        if split_ids is not None and paper_a not in split_ids:
            continue
        if paper_a not in raw_titles or paper_b not in raw_titles:
            continue
        seen[paper_a].add(paper_b)

    examples: list[TrainingExample] = []
    for paper_a, gold_set in seen.items():
        title_a = raw_titles[paper_a]
        abstract_a = abstracts.get(paper_a, "")
        messages = _build_paper_messages(title_a, abstract_a)
        for paper_b in sorted(gold_set):
            target_text = normalise_title(raw_titles[paper_b])
            if not target_text:
                continue
            examples.append(
                TrainingExample(
                    prompt_messages=messages,
                    target_docid=target_text,
                    kind="retrieval",
                )
            )

    rng.shuffle(examples)
    logger.info(
        "GlenPaperTitleDataset: %d rows from %d paper_a queries",
        len(examples),
        len(seen),
    )
    return examples


__all__ = [
    "GlenPaperTitleDataset",
    "GlenPaperTitleDatasetConfig",
]
