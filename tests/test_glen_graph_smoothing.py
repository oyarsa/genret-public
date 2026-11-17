"""Tests for graph-smoothed GLEN bullet DocIDs."""

import json
from pathlib import Path

import numpy as np

from genret.training.glen_graph_smoothing import (
    build_smoothed_bullets,
    load_smoothed_bullets,
    smooth_bullet_words,
)
from genret.types import CorpusId


class TestSmoothBulletWords:
    """Order-preserving neighbourhood filter."""

    def test_keeps_attested_drops_idiosyncratic(self) -> None:
        """Attested words survive in order; unattested ones drop."""
        words = ["curriculum", "orders", "lexical", "overlap", "training"]
        attestation = {"curriculum": 5, "orders": 3, "training": 4}
        out = smooth_bullet_words(words, attestation, min_attest=2, min_words=2)
        assert out == ["curriculum", "orders", "training"]

    def test_pads_when_too_few_attested(self) -> None:
        """Below min_words, earliest unattested words pad the result."""
        words = ["alpha", "beta", "gamma", "delta"]
        attestation = {"delta": 9}
        out = smooth_bullet_words(words, attestation, min_attest=2, min_words=3)
        assert out == ["alpha", "beta", "delta"]

    def test_all_unattested_keeps_prefix(self) -> None:
        """No attestation at all: keep the first min_words words."""
        words = ["one", "two", "three", "four"]
        out = smooth_bullet_words(words, {}, min_attest=1, min_words=2)
        assert out == ["one", "two"]


def _write_candidates(
    path: Path, facet: str, cids: list[int], neighbours: dict[int, list[int]]
) -> None:
    """Minimal facet_candidates npz: each paper's candidate row indices."""
    index = {c: i for i, c in enumerate(cids)}
    n = len(cids)
    width = max(len(v) for v in neighbours.values())
    cand = np.full((n, width), -1, dtype=np.int64)
    scores = np.zeros((n, width), dtype=np.float32)
    for cid, nbs in neighbours.items():
        for j, nb in enumerate(nbs):
            cand[index[cid], j] = index[nb]
            scores[index[cid], j] = 1.0 - 0.1 * j
    np.savez(
        path / f"facet_candidates_{facet}.npz",
        corpusids=np.array(cids, dtype=np.int64),
        candidates=cand,
        scores_combined=scores,
    )


class TestBuildSmoothedBullets:
    """End-to-end map construction from facets + candidates graph."""

    def test_neighbour_attested_words_kept(self, tmp_path: Path) -> None:
        """Paper 1's bullet keeps words its neighbours also use."""
        facets_path = tmp_path / "facets.jsonl"
        rows = [
            {
                "corpusid": 1,
                "problems": [
                    "curriculum orders training pairs lexical overlap stabilise"
                ],
                "methods": [],
                "results": [],
                "contributions": [],
            },
            {
                "corpusid": 2,
                "problems": ["curriculum learning orders training data"],
                "methods": [],
                "results": [],
                "contributions": [],
            },
            {
                "corpusid": 3,
                "problems": ["training pairs ordering curriculum difficulty"],
                "methods": [],
                "results": [],
                "contributions": [],
            },
        ]
        with facets_path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        for facet in ("problems", "methods", "results", "contributions"):
            _write_candidates(
                tmp_path, facet, [1, 2, 3], {1: [2, 3], 2: [1, 3], 3: [1, 2]}
            )

        out_path = tmp_path / "smoothed.jsonl"
        stats = build_smoothed_bullets(
            facets_path,
            tmp_path,
            out_path,
            top_neighbours=2,
            min_attest=2,
            min_words=2,
        )
        smoothed = load_smoothed_bullets(out_path)
        key = (
            CorpusId(1),
            "problems",
            "curriculum orders training pairs lexical overlap stabilise",
        )
        # 'curriculum', 'training' attested by both neighbours; 'orders' by
        # one, 'pairs' by one; 'lexical overlap stabilise' by none.
        assert smoothed[key] == "curriculum training"
        assert stats["n_bullets"] == 3.0
        assert stats["verbatim_fallback_fraction"] == 0.0
