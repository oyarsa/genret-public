"""Dataset constructing per-facet codebook training examples.

Differs from the keyphrase-DocID dataset in three ways:

- Each example carries text for a single facet, not all four, so the
  model sees the same input format whether it is decoding problems
  codes or methods codes.
- The target is a four-token string ``<CODE_START> <X_L0_a>
  <X_L1_b> <CODE_END>`` rather than a long keyphrase sequence.
- Indexing and retrieval examples are pre-emitted by the
  ``build-codebook-data`` builder; this module just attaches text and
  mixes them at the requested ratio.
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
from genret.training.codebook_data import FACET_NAMES
from genret.training.config import CodebookConfig
from genret.training.data import TrainingExample, _mix_examples
from genret.types import CorpusId
from genret.utils import iter_jsonl

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_INDEXING = (
    "You are a generative retrieval system. Given a research paper's "
    "{facet_type}, generate its semantic code."
)
"""System prompt used when the target is the same paper's own code."""

SYSTEM_PROMPT_RETRIEVAL = (
    "You are a generative retrieval system. Given a research paper's "
    "{facet_type}, generate the semantic code of a relevant paper."
)
"""System prompt used when the target is a related paper's code."""

SYSTEM_PROMPT_TITLE = (
    "You are a generative retrieval system. Given a research paper's "
    "{facet_type}, generate the title of a relevant paper."
)
"""System prompt used when the target is a related paper's title (D5)."""


def _facet_text(facets: Facets, facet: str, evidence: FacetEvidence | None) -> str:
    """Render one facet's bullets (with optional evidence) as a string."""
    items = getattr(facets, facet)
    item_evidence: list[list[str]] = evidence.get(facet, []) if evidence else []
    lines: list[str] = []
    for i, item in enumerate(items):
        lines.append(f"- {item}")
        if i < len(item_evidence):
            for quote in item_evidence[i]:
                lines.append(f'  > "{quote}"')
    return "\n".join(lines)


def _build_messages(
    facet: str,
    facet_text: str,
    *,
    kind: str,
) -> list[dict[str, str]]:
    """Two-message chat conversation for a single-facet codebook example."""
    if kind == "indexing":
        template = SYSTEM_PROMPT_INDEXING
    elif kind == "title":
        template = SYSTEM_PROMPT_TITLE
    else:
        template = SYSTEM_PROMPT_RETRIEVAL
    system = template.format(facet_type=facet.capitalize())
    user = f"{facet.capitalize()}:\n{facet_text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _target_string(cfg: CodebookConfig, facet: str, code: tuple[int, int]) -> str:
    """Render an `(L0, L1)` code as the model's generation target."""
    l0, l1 = code
    return (
        f"{cfg.code_start_token} {cfg.l0_token(facet, l0)} "
        f"{cfg.l1_token(facet, l1)} {cfg.code_end_token}"
    )


def _load_facets_and_evidence(
    facets_path: Path, facets_type: FacetRecordType
) -> tuple[dict[CorpusId, Facets], dict[CorpusId, FacetEvidence]]:
    facets_map: dict[CorpusId, Facets] = {}
    evidence_map: dict[CorpusId, FacetEvidence] = {}
    for raw in iter_jsonl(facets_path):
        f = facets_from_type(raw, facets_type)
        facets_map[f.corpusid] = f
        if ev := evidence_from_type(raw, facets_type):
            evidence_map[f.corpusid] = ev
    return facets_map, evidence_map


@dataclass(frozen=True, kw_only=True)
class CodebookDatasetConfig:
    """Configuration for the codebook training dataset."""

    pairs_path: Path
    """JSONL produced by ``build-codebook-data``."""
    facets_path: Path
    """JSONL with the per-paper facet bullets (and optional evidence)."""
    codebook: CodebookConfig
    facets_type: FacetRecordType = FacetRecordType.STANDARD
    include_evidence: bool = True
    indexing_ratio: float = 0.5
    split_ids: set[CorpusId] | None = None
    """If set, restrict examples to pairs where ``paper_a`` is in this
    split. ``paper_b`` may still be drawn from outside the split."""
    facets: tuple[str, ...] = FACET_NAMES
    """Restrict examples to these facets. Defaults to all four."""


class CodebookDataset:
    """Dataset that mixes codebook indexing and retrieval examples."""

    def __init__(
        self,
        config: CodebookDatasetConfig,
        *,
        rng: random.Random,
    ) -> None:
        self._examples = _build_codebook_examples(config, rng=rng)

    def __len__(self) -> int:  # pragma: no cover - trivial wrapper
        """Return the number of training examples in the dataset."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> TrainingExample:  # pragma: no cover
        """Return the training example at index ``idx``."""
        return self._examples[idx]


def _build_codebook_examples(
    config: CodebookDatasetConfig, *, rng: random.Random
) -> list[TrainingExample]:
    """Load training pairs + facets and produce ``TrainingExample`` rows."""
    facets_map, evidence_map = _load_facets_and_evidence(
        config.facets_path, config.facets_type
    )
    if not config.include_evidence:
        evidence_map = dict[CorpusId, FacetEvidence]()

    indexing_examples: list[TrainingExample] = []
    retrieval_examples: list[TrainingExample] = []
    skipped = 0

    split_ids = config.split_ids

    for row in iter_jsonl(config.pairs_path):
        paper_a = CorpusId(int(row["paper_a"]))
        # paper_b (the target) is implicit in target_code; not needed here
        facet = str(row["facet"])
        kind = str(row["kind"])
        l0, l1 = row["target_code"]

        if facet not in FACET_NAMES:
            continue
        if facet not in config.facets:
            continue
        if split_ids is not None and paper_a not in split_ids:
            continue
        if paper_a not in facets_map:
            skipped += 1
            continue
        facet_text = _facet_text(facets_map[paper_a], facet, evidence_map.get(paper_a))
        if not facet_text:
            skipped += 1
            continue

        example = TrainingExample(
            prompt_messages=_build_messages(facet, facet_text, kind=kind),
            target_docid=_target_string(config.codebook, facet, (int(l0), int(l1))),
        )
        if kind == "indexing":
            indexing_examples.append(example)
        else:
            retrieval_examples.append(example)

    if skipped:
        logger.info(
            "Skipped %d codebook training rows (missing facets or empty text)",
            skipped,
        )

    result = _mix_examples(
        indexing_examples, retrieval_examples, config.indexing_ratio, rng
    )
    logger.info(
        "Codebook dataset: %d examples (%d indexing, %d retrieval) at "
        "indexing_ratio=%.2f",
        len(result),
        len(indexing_examples),
        len(retrieval_examples),
        config.indexing_ratio,
    )
    return result
