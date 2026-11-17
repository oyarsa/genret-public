"""Tests for the codebook training-pair builder."""

from __future__ import annotations

import json
from pathlib import Path

from genret.training.codebook_data import (
    CodebookTrainingExample,
    _indexing_examples,
    _retrieval_examples,
    build_training_pairs,
    load_paper_codes,
)
from genret.types import CorpusId


def _write_paper_codes(
    codebook_dir: Path, facet: str, codes: dict[int, tuple[int, int]]
) -> None:
    """Helper: write a fake `paper_codes.jsonl` under `codebook_dir/facet`."""
    facet_dir = codebook_dir / facet
    facet_dir.mkdir(parents=True, exist_ok=True)
    with (facet_dir / "paper_codes.jsonl").open("w") as fh:
        for cid, (l0, l1) in codes.items():
            fh.write(json.dumps({"corpusid": cid, "code": [l0, l1]}) + "\n")


def _write_retrieval(path: Path, rows: list[dict[str, object]]) -> None:
    """Helper: write a fake `batch-retrieve` JSONL output to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_indexing_examples_one_per_paper_facet(tmp_path: Path) -> None:
    """One indexing example per (paper, facet) the codebook covers."""
    codebook_dir = tmp_path / "codebooks"
    for facet in ("problems", "methods", "results", "contributions"):
        _write_paper_codes(codebook_dir, facet, {1: (0, 0), 2: (1, 2)})
    paper_codes = load_paper_codes(codebook_dir)
    examples = list(
        _indexing_examples(
            paper_codes,
            corpus_ids={CorpusId(1), CorpusId(2)},
        )
    )
    # 2 papers x 4 facets
    assert len(examples) == 8
    assert all(e.kind == "indexing" for e in examples)
    assert all(e.paper_a == e.paper_b for e in examples)


def test_retrieval_examples_respect_max_rank(tmp_path: Path) -> None:
    """Candidates above the rank cutoff are dropped per-facet."""
    codebook_dir = tmp_path / "codebooks"
    for facet in ("problems", "methods", "results", "contributions"):
        _write_paper_codes(codebook_dir, facet, {1: (0, 0), 2: (3, 4), 3: (5, 6)})
    paper_codes = load_paper_codes(codebook_dir)

    retrieval = tmp_path / "retrieval.jsonl"
    _write_retrieval(
        retrieval,
        [
            {
                "corpusid": 1,
                "results": [
                    {
                        "corpusid": 2,
                        "facets": ["problems", "methods"],
                        "facet_ranks": {"problems": 3, "methods": 25},
                    },
                    {
                        "corpusid": 3,
                        "facets": ["results"],
                        "facet_ranks": {"results": 0},
                    },
                ],
            },
        ],
    )
    examples = list(_retrieval_examples(retrieval, paper_codes, max_rank=20))
    facets = sorted((e.facet, e.paper_b) for e in examples)
    # problems rank 3 (ok), methods rank 25 (drop), results rank 0 (ok)
    assert facets == [("problems", CorpusId(2)), ("results", CorpusId(3))]


def test_retrieval_skips_cross_facet(tmp_path: Path) -> None:
    """Cross-facet signals (problems_x_contributions) are not emitted."""
    codebook_dir = tmp_path / "codebooks"
    for facet in ("problems", "methods", "results", "contributions"):
        _write_paper_codes(codebook_dir, facet, {1: (0, 0), 2: (1, 1)})
    paper_codes = load_paper_codes(codebook_dir)

    retrieval = tmp_path / "retrieval.jsonl"
    _write_retrieval(
        retrieval,
        [
            {
                "corpusid": 1,
                "results": [
                    {
                        "corpusid": 2,
                        "facets": ["problems_x_contributions"],
                        "facet_ranks": {
                            "problems_x_contributions": 0,
                            "problems": 1,
                        },
                    },
                ],
            },
        ],
    )
    examples = list(_retrieval_examples(retrieval, paper_codes, max_rank=20))
    # Only the in-vocab facet survives; the cross-facet rank is ignored.
    assert {e.facet for e in examples} == {"problems"}


def test_build_writes_jsonl_and_returns_stats(tmp_path: Path) -> None:
    """Driver writes one line per example and reports matching counts."""
    codebook_dir = tmp_path / "codebooks"
    for facet in ("problems", "methods", "results", "contributions"):
        _write_paper_codes(codebook_dir, facet, {1: (0, 0), 2: (1, 1)})

    retrieval = tmp_path / "retrieval.jsonl"
    _write_retrieval(
        retrieval,
        [
            {
                "corpusid": 1,
                "results": [
                    {
                        "corpusid": 2,
                        "facets": ["methods"],
                        "facet_ranks": {"methods": 0},
                    },
                ],
            },
        ],
    )

    out = tmp_path / "training_pairs.jsonl"
    stats = build_training_pairs(retrieval, codebook_dir, out, max_rank=20)

    lines = out.read_text().splitlines()
    assert len(lines) == stats.indexing + stats.retrieval
    # 2 papers x 4 facets indexing + 1 retrieval (methods)
    assert stats.indexing == 8
    assert stats.retrieval == 1
    assert stats.per_facet_retrieval["methods"] == 1
    assert stats.unique_query_papers == 1


def test_to_jsonl_record_round_trip() -> None:
    """JSONL serialisation preserves all fields verbatim."""
    ex = CodebookTrainingExample(
        paper_a=CorpusId(5),
        paper_b=CorpusId(7),
        facet="problems",
        target_code=(3, 4),
        kind="retrieval",
        weight=0.5,
    )
    rec = ex.to_jsonl_record()
    assert rec == {
        "paper_a": 5,
        "paper_b": 7,
        "facet": "problems",
        "target_code": [3, 4],
        "kind": "retrieval",
        "weight": 0.5,
    }
