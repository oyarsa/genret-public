"""Tests for graph-pseudo-gold sources in min_target_views augmentation."""

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from genret.training.glen_bullet_dataset import (
    GlenBulletDatasetConfig,
    _build_bullet_examples,
)
from genret.types import CorpusId


def _write_fixture(tmp_path: Path) -> GlenBulletDatasetConfig:
    """Build a fixture where only the graph connects paper 3 to a source.

    Paper 1 cites paper 2 (gold). Paper 3 has NO gold edge but is a graph
    candidate of paper 1 and vice versa.
    """
    facets_path = tmp_path / "facets.jsonl"
    with facets_path.open("w") as fh:
        for cid, bullet in [
            (1, "query paper studies retrieval"),
            (2, "gold paper about retrieval evaluation"),
            (3, "isolated paper on graph training signal"),
        ]:
            fh.write(
                json.dumps(
                    {
                        "corpusid": cid,
                        "problems": [bullet],
                        "methods": [],
                        "results": [],
                        "contributions": [],
                    }
                )
                + "\n"
            )

    graphs_path = tmp_path / "paper_graphs.jsonl"
    with graphs_path.open("w") as fh:
        for cid in (1, 2, 3):
            fh.write(json.dumps({"corpusid": cid, "title": f"Paper {cid}"}) + "\n")

    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(
        json.dumps(
            {"kind": "retrieval", "paper_a": 1, "paper_b": 2, "facet": "problems"}
        )
        + "\n"
    )

    # Candidates graph: paper 3's top candidate is paper 1 (and 1's is 3),
    # so the graph - and only the graph - links the isolated paper to a
    # potential source.
    cids = np.array([1, 2, 3], dtype=np.int64)
    candidates = np.array([[2], [-1], [0]], dtype=np.int64)  # row idx into cids
    scores = np.array([[0.9], [0.0], [0.9]], dtype=np.float32)
    for facet in ("problems", "methods", "results", "contributions"):
        np.savez(
            tmp_path / f"facet_candidates_{facet}.npz",
            corpusids=cids,
            candidates=candidates,
            scores_combined=scores,
        )

    return GlenBulletDatasetConfig(
        pairs_path=pairs_path,
        facets_path=facets_path,
        paper_graphs_path=graphs_path,
        split_ids={CorpusId(1)},
        facets=("problems",),
        graph_sampling="weighted",
        candidates_dir=tmp_path,
        min_target_views=2,
    )


_ISOLATED_TARGET = "isolated paper on graph training signal"


class TestGraphFallbackSources:
    """Graph neighbours stand in for missing gold paper_a sources."""

    def test_without_fallback_isolated_paper_untrained(self, tmp_path: Path) -> None:
        """Default behaviour: no gold source -> no training views."""
        config = _write_fixture(tmp_path)
        examples = _build_bullet_examples(config, rng=random.Random(0))
        targets = [ex.target_docid for ex in examples]
        assert _ISOLATED_TARGET not in targets

    def test_with_fallback_isolated_paper_reaches_floor(self, tmp_path: Path) -> None:
        """Graph fallback: isolated paper's bullet hits min_target_views."""

        config = replace(_write_fixture(tmp_path), min_views_graph_fallback=5)
        examples = _build_bullet_examples(config, rng=random.Random(0))
        targets = [ex.target_docid for ex in examples]
        assert targets.count(_ISOLATED_TARGET) == 2
