"""Tests for the per-facet hierarchical codebook builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from genret.training.codebook import (
    CodebookParams,
    FacetCodebook,
    _build_code_to_papers,
    _leaf_stats,
    build_codebooks,
    fit_facet_codebook,
)
from genret.types import CorpusId


def _make_fixture(tmp_path: Path) -> Path:
    """Create a fake facet_graph_*.npz dir with three well-separated clusters.

    Each facet shares the same vectors so we can predict the cluster
    assignments deterministically.
    """
    rng = np.random.default_rng(0)
    centers = np.array(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [-10.0, 0.0],
            [0.0, -10.0],
        ],
        dtype=np.float32,
    )
    n_per = 8
    vecs = np.concatenate(
        [
            c + rng.normal(scale=0.05, size=(n_per, 2)).astype(np.float32)
            for c in centers
        ]
    )
    corpusids = np.arange(1000, 1000 + vecs.shape[0], dtype=np.int64)

    out = tmp_path / "processed"
    out.mkdir(parents=True)
    for facet in ("problems", "methods", "results", "contributions"):
        np.savez(
            out / f"facet_graph_{facet}.npz",
            corpusids=corpusids,
            corpus_vectors=vecs.astype(np.float16),
        )
    return out


def test_fit_codebook_assigns_clusters() -> None:
    """k-means should cleanly separate four well-separated blobs."""
    rng = np.random.default_rng(1)
    centers = np.array([[5.0, 5.0], [-5.0, -5.0]], dtype=np.float32)
    vecs = np.concatenate(
        [c + rng.normal(scale=0.05, size=(6, 2)).astype(np.float32) for c in centers]
    )
    corpusids = [CorpusId(i) for i in range(vecs.shape[0])]

    codebook = fit_facet_codebook(
        "problems",
        corpusids,
        vecs,
        CodebookParams(k0=2, k1=2, random_state=0, n_init=4),
    )

    # Each blob should map to a single L0 cluster (two distinct values total).
    blob_a_l0 = {codebook.paper_codes[CorpusId(i)][0] for i in range(6)}
    blob_b_l0 = {codebook.paper_codes[CorpusId(i)][0] for i in range(6, 12)}
    assert len(blob_a_l0) == 1
    assert len(blob_b_l0) == 1
    assert blob_a_l0 != blob_b_l0


def test_codebook_is_deterministic() -> None:
    """Two fits with the same seed must produce identical paper codes."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((30, 16)).astype(np.float32)
    corpusids = [CorpusId(i) for i in range(30)]
    params = CodebookParams(k0=4, k1=4, random_state=0, n_init=2)

    a = fit_facet_codebook("problems", corpusids, vecs, params)
    b = fit_facet_codebook("problems", corpusids, vecs, params)
    assert a.paper_codes == b.paper_codes


def test_code_to_papers_includes_empty_leaves() -> None:
    """Every (L0, L1) cell should appear in the lookup, even with no papers."""
    codebook = FacetCodebook(
        facet="problems",
        level0_centers=np.zeros((4, 2), dtype=np.float32),
        level1_centers=np.zeros((4, 4, 2), dtype=np.float32),
        paper_codes={CorpusId(1): (0, 0), CorpusId(2): (0, 0), CorpusId(3): (1, 2)},
    )
    params = CodebookParams(k0=4, k1=4)
    idx = _build_code_to_papers(codebook, params)

    assert idx["0,0"] == [1, 2]
    assert idx["1,2"] == [3]
    # 16 total leaves, all keys present
    assert len(idx) == 16
    # Empty leaves have empty lists
    assert idx["3,3"] == []


def test_leaf_stats_passes_gate_for_clean_clusters(tmp_path: Path) -> None:
    """A well-separated 4-cluster fixture should pass the validation gate."""
    processed = _make_fixture(tmp_path)
    stats = build_codebooks(
        processed,
        tmp_path / "codebooks",
        CodebookParams(k0=4, k1=4, random_state=0, n_init=4),
    )

    # All four facets share the same vectors, so each should pass.
    for facet in ("problems", "methods", "results", "contributions"):
        assert stats[facet].passes_gate(), f"{facet}: {stats[facet]}"


def test_artifacts_written(tmp_path: Path) -> None:
    """build_codebooks should write all expected files."""
    processed = _make_fixture(tmp_path)
    out = tmp_path / "codebooks"
    build_codebooks(
        processed, out, CodebookParams(k0=4, k1=2, random_state=0, n_init=2)
    )

    assert (out / "code_to_papers.json").exists()
    assert (out / "validation_stats.json").exists()
    for facet in ("problems", "methods", "results", "contributions"):
        facet_dir = out / facet
        assert (facet_dir / "level0_centers.npy").exists()
        assert (facet_dir / "level1_centers.npy").exists()
        codes_path = facet_dir / "paper_codes.jsonl"
        assert codes_path.exists()
        # paper_codes.jsonl should have one line per corpus id
        lines = codes_path.read_text().splitlines()
        assert len(lines) == 32  # 4 centers, 8 papers each
        first = json.loads(lines[0])
        assert set(first) == {"corpusid", "code"}
        assert len(first["code"]) == 2


def test_leaf_stats_min_max_sane() -> None:
    """Direct check that leaf-stats counts match the paper assignments."""
    codebook = FacetCodebook(
        facet="problems",
        level0_centers=np.zeros((2, 2), dtype=np.float32),
        level1_centers=np.zeros((2, 2, 2), dtype=np.float32),
        paper_codes={
            CorpusId(1): (0, 0),
            CorpusId(2): (0, 0),
            CorpusId(3): (0, 0),
            CorpusId(4): (0, 0),  # 4 papers in (0, 0) → >3 leaf
            CorpusId(5): (1, 1),
        },
    )
    params = CodebookParams(k0=2, k1=2)
    stats = _leaf_stats("problems", codebook, params)
    assert stats.leaves_total == 4
    assert stats.leaves_empty == 2  # (0, 1) and (1, 0) are empty
    assert stats.leaves_with_more_than_three == 1
    assert stats.papers_per_leaf_max == 4
    assert stats.papers_per_leaf_min == 0
