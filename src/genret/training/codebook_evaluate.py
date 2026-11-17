# pyright: basic
"""Evaluation harness for the per-facet codebook generative retriever.

For each query paper, run a constrained beam search per facet, look up
papers at each generated leaf, fuse the four per-facet rankings with
RRF, and emit a flat retrieval list with per-facet attribution.

The output JSONL mirrors the schema of
``output/eval_500/retrieval_cross_facet.jsonl`` (corpus id + per-result
score, facets, facet_ranks) so downstream metric scripts can be reused
unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from genret.paper_parsing import (
    FacetEvidence,
    FacetRecordType,
    Facets,
    evidence_from_type,
    facets_from_type,
)
from genret.processing.facet_retrieval import DEFAULT_RRF_K
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_dataset import _build_messages, _facet_text
from genret.training.codebook_decoding import (
    CodebookIndex,
    CodebookTokenMap,
    FacetDecodeResult,
    decode_per_facet,
)
from genret.training.config import CodebookConfig
from genret.types import CorpusId
from genret.utils import iter_jsonl, open_auto

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class FusedResult:
    """Cross-facet fused retrieval result with attribution."""

    corpusid: CorpusId
    score: float
    facets: list[str]
    facet_ranks: dict[str, int]


def _fuse_facet_results(
    per_facet: dict[str, list[FacetDecodeResult]],
    k: int = DEFAULT_RRF_K,
) -> list[FusedResult]:
    """Run RRF across the per-facet decoded paper lists.

    Within a facet, each beam contributes its papers; the first
    occurrence of a corpus id within the facet defines its facet rank.
    Across facets, RRF fuses the rankings.
    """
    facet_ranks: dict[str, dict[CorpusId, int]] = {f: {} for f in per_facet}
    for facet, beams in per_facet.items():
        seen: set[CorpusId] = set()
        rank = 0
        for beam in beams:
            for paper in beam.paper_ids:
                if paper in seen:
                    continue
                seen.add(paper)
                facet_ranks[facet][paper] = rank
                rank += 1

    scores: dict[CorpusId, float] = {}
    provenance: dict[CorpusId, dict[str, int]] = {}
    for facet, ranks in facet_ranks.items():
        for paper, rank in ranks.items():
            scores[paper] = scores.get(paper, 0.0) + 1.0 / (k + rank + 1)
            provenance.setdefault(paper, {})[facet] = rank

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        FusedResult(
            corpusid=paper,
            score=score,
            facets=sorted(provenance[paper]),
            facet_ranks=provenance[paper],
        )
        for paper, score in ranked
    ]


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


def evaluate_codebook_retrieval(
    *,
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizerBase,
    cfg: CodebookConfig,
    index: CodebookIndex,
    tokens: CodebookTokenMap,
    queries: Sequence[CorpusId],
    facets_map: dict[CorpusId, Facets],
    evidence_map: dict[CorpusId, FacetEvidence] | None,
    output_path: Path,
    num_beams: int = 30,
    top_n: int = 50,
    facets: tuple[str, ...] = FACET_NAMES,
) -> None:
    """Run 4-facet constrained beam search + RRF for every query.

    Writes a JSONL with one record per query::

        {"corpusid": int, "results": [
            {"corpusid": int, "score": float, "facets": [...],
             "facet_ranks": {facet: int}},
            ...
        ]}

    Output schema matches the graph retriever's
    ``retrieval_cross_facet.jsonl`` so downstream evaluation scripts
    can score generative output without modification.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evidence_map = evidence_map or {}
    with open_auto(output_path, "w") as fh:
        for qid in queries:
            paper_facets = facets_map.get(qid)
            if paper_facets is None:
                logger.warning("Query %d not in facets map; skipping", qid)
                continue
            evidence = evidence_map.get(qid)
            per_facet: dict[str, list[FacetDecodeResult]] = {}
            for facet in facets:
                text = _facet_text(paper_facets, facet, evidence)
                if not text:
                    per_facet[facet] = []
                    continue
                messages = _build_messages(facet, text, kind="retrieval")
                per_facet[facet] = decode_per_facet(
                    model=model,
                    tokeniser=tokeniser,
                    messages=messages,
                    facet=facet,
                    index=index,
                    tokens=tokens,
                    num_beams=num_beams,
                )

            fused = _fuse_facet_results(per_facet)
            # Drop the query itself if it surfaced in its own results.
            fused = [r for r in fused if r.corpusid != qid][:top_n]

            record = {
                "corpusid": int(qid),
                "results": [
                    {
                        "corpusid": int(r.corpusid),
                        "score": round(r.score, 6),
                        "facets": r.facets,
                        "facet_ranks": r.facet_ranks,
                    }
                    for r in fused
                ],
            }
            fh.write(json.dumps(record) + "\n")


def load_eval_inputs(
    facets_path: Path, facets_type: FacetRecordType
) -> tuple[dict[CorpusId, Facets], dict[CorpusId, FacetEvidence]]:
    """Public wrapper around the facets/evidence loader."""
    return _load_facets_and_evidence(facets_path, facets_type)


def queries_from_split(path: Path) -> list[CorpusId]:
    """Read a ``train.txt``/``query_papers.txt`` file of corpus ids."""
    return [CorpusId(int(x)) for x in path.read_text().split() if x.strip()]


__all__: Iterable[str] = (
    "FusedResult",
    "_fuse_facet_results",
    "evaluate_codebook_retrieval",
    "load_eval_inputs",
    "queries_from_split",
)
