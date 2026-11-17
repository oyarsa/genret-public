"""Gold (facet, paper_a) → paper_b mapping from the training pairs file."""

from __future__ import annotations

from genret.training.codebook_data import FACET_NAMES
from genret.types import CorpusId
from genret.utils import iter_jsonl

from .config import GlenBulletDatasetConfig


def _load_gold_by_query(
    config: GlenBulletDatasetConfig,
) -> dict[tuple[str, CorpusId], frozenset[CorpusId]]:
    """Map (facet, paper_a) to the set of gold paper_b's, from the pairs file.

    Respects the same kind / facet / split filters as the main example loop.
    """
    out: dict[tuple[str, CorpusId], set[CorpusId]] = {}
    split_ids = config.split_ids
    for row in iter_jsonl(config.pairs_path):
        if str(row.get("kind", "")) != "retrieval":
            continue
        facet = str(row["facet"])
        if facet not in FACET_NAMES or facet not in config.facets:
            continue
        a = CorpusId(int(row["paper_a"]))
        if split_ids is not None and a not in split_ids:
            continue
        out.setdefault((facet, a), set()).add(CorpusId(int(row["paper_b"])))
    return {k: frozenset(v) for k, v in out.items()}
