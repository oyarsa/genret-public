"""Build per-facet codebook training data from the graph teacher.

For each (query, candidate) pair in the graph retriever's top-50 output
and each same-facet signal the candidate was retrieved through, emit a
training example mapping the query's facet text to the candidate's code
under that facet. Also emit indexing examples mapping every paper's own
facet text to its own code, for each of the four facets.

The downstream training data loader takes this structured output and
attaches the actual prompt text + chat template at model-training time.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from genret.types import CorpusId
from genret.utils import iter_jsonl, open_auto

logger = logging.getLogger(__name__)

FACET_NAMES: tuple[str, ...] = (
    "problems",
    "methods",
    "results",
    "contributions",
)
"""Same-facet signals used to derive codebook training data."""


@dataclass(frozen=True, kw_only=True)
class CodebookTrainingExample:
    """One row of the codebook training data."""

    paper_a: CorpusId
    """Source paper — whose facet text serves as input."""

    paper_b: CorpusId
    """Target paper — whose code is the generation target."""

    facet: str
    """Facet name; the input is paper_a's bullets in this facet and the
    target is paper_b's two-token code in this facet's codebook."""

    target_code: tuple[int, int]
    """The (L0, L1) code for paper_b in the named facet."""

    kind: str
    """"indexing" if paper_a == paper_b, else "retrieval"."""

    weight: float = 1.0
    """Per-example training weight; reserved for future use."""

    def to_jsonl_record(self) -> dict[str, object]:
        """Serialise to a JSONL-compatible dict."""
        return {
            "paper_a": int(self.paper_a),
            "paper_b": int(self.paper_b),
            "facet": self.facet,
            "target_code": [self.target_code[0], self.target_code[1]],
            "kind": self.kind,
            "weight": self.weight,
        }


def load_paper_codes(
    codebook_dir: Path,
    facets: Iterable[str] = FACET_NAMES,
) -> dict[str, dict[CorpusId, tuple[int, int]]]:
    """Load the per-facet `corpusid -> (L0, L1)` maps from disk."""
    out: dict[str, dict[CorpusId, tuple[int, int]]] = {}
    for facet in facets:
        facet_codes: dict[CorpusId, tuple[int, int]] = {}
        for r in iter_jsonl(codebook_dir / facet / "paper_codes.jsonl"):
            cid = CorpusId(int(r["corpusid"]))
            l0, l1 = r["code"]
            facet_codes[cid] = (int(l0), int(l1))
        out[facet] = facet_codes
    return out


def _retrieval_examples(
    retrieval_path: Path,
    paper_codes: Mapping[str, Mapping[CorpusId, tuple[int, int]]],
    *,
    max_rank: int,
) -> Iterable[CodebookTrainingExample]:
    """Yield retrieval examples derived from the graph retriever's output."""
    for row in iter_jsonl(retrieval_path):
        query_cid = CorpusId(int(row["corpusid"]))
        for cand in row.get("results", []):
            target_cid = CorpusId(int(cand["corpusid"]))
            if target_cid == query_cid:
                continue
            facet_ranks: dict[str, int] = cand.get("facet_ranks", {}) or {}
            for facet, rank in facet_ranks.items():
                if facet not in paper_codes:
                    # Cross-facet signals (problems_x_contributions) are
                    # not part of the codebook scheme yet — skip.
                    continue
                if rank >= max_rank:
                    continue
                target_code = paper_codes[facet].get(target_cid)
                if target_code is None:
                    continue
                # RRF-style per-rank weight; consumed by rank-distillation
                # training. CE training ignores this field.
                weight = 1.0 / (60.0 + float(rank) + 1.0)
                yield CodebookTrainingExample(
                    paper_a=query_cid,
                    paper_b=target_cid,
                    facet=facet,
                    target_code=target_code,
                    kind="retrieval",
                    weight=weight,
                )


def _indexing_examples(
    paper_codes: Mapping[str, Mapping[CorpusId, tuple[int, int]]],
    *,
    corpus_ids: set[CorpusId],
) -> Iterable[CodebookTrainingExample]:
    """Yield one indexing example per (paper, facet) the corpus contains."""
    for facet, codes in paper_codes.items():
        for cid, code in codes.items():
            if cid not in corpus_ids:
                continue
            yield CodebookTrainingExample(
                paper_a=cid,
                paper_b=cid,
                facet=facet,
                target_code=code,
                kind="indexing",
            )


@dataclass(frozen=True)
class BuildStats:
    """Counts emitted by ``build_training_pairs``."""

    indexing: int
    retrieval: int
    per_facet_retrieval: dict[str, int]
    unique_query_papers: int


def build_training_pairs(
    retrieval_path: Path,
    codebook_dir: Path,
    output_path: Path,
    *,
    max_rank: int = 20,
) -> BuildStats:
    """Build codebook training pairs and write them to ``output_path``.

    Args:
        retrieval_path: JSONL produced by ``batch-retrieve`` over the
            corpus, with per-candidate ``facet_ranks``.
        codebook_dir: Directory containing ``{facet}/paper_codes.jsonl``.
        output_path: Output JSONL (``.jsonl.zst`` supported).
        max_rank: Drop candidates whose facet rank is at least this
            value. Default 20 keeps the spec's top-20 cutoff.
    """
    paper_codes = load_paper_codes(codebook_dir)
    corpus_ids: set[CorpusId] = set()
    for codes in paper_codes.values():
        corpus_ids.update(codes.keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_facet_retrieval: dict[str, int] = dict.fromkeys(FACET_NAMES, 0)
    indexing_count = 0
    retrieval_count = 0
    unique_queries: set[CorpusId] = set()

    with open_auto(output_path, "w") as fh:
        for example in _indexing_examples(paper_codes, corpus_ids=corpus_ids):
            fh.write(json.dumps(example.to_jsonl_record()) + "\n")
            indexing_count += 1

        for example in _retrieval_examples(
            retrieval_path, paper_codes, max_rank=max_rank
        ):
            fh.write(json.dumps(example.to_jsonl_record()) + "\n")
            retrieval_count += 1
            per_facet_retrieval[example.facet] += 1
            unique_queries.add(example.paper_a)

    return BuildStats(
        indexing=indexing_count,
        retrieval=retrieval_count,
        per_facet_retrieval=per_facet_retrieval,
        unique_query_papers=len(unique_queries),
    )
