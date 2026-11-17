"""Tests for the no-training C2T label evaluation harness."""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from genret.training.c2t_label_eval import evaluate_c2t_labels

_E1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_E2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
_E3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _fake_embed(texts: list[str]) -> npt.NDArray[np.float32]:
    """Route any text containing 'alpha' to e1, 'gamma' to e2, else e3."""
    rows = []
    for t in texts:
        if "alpha" in t:
            rows.append(_E1)
        elif "gamma" in t:
            rows.append(_E2)
        else:
            rows.append(_E3)
    return np.stack(rows)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build a toy C2T index fixture.

    Two leaves ('alpha beta' -> papers 1,2; 'gamma delta' -> 3,4), one
    query paper 9 citing paper 1, plus one colliding leaf in l1_labels to
    exercise collision counting.
    """
    c2t_dir = tmp_path / "c2t"
    c2t_dir.mkdir()
    rows = [
        {"corpusid": 1, "facet": "problems", "docid": "alpha beta"},
        {"corpusid": 2, "facet": "problems", "docid": "alpha beta"},
        {"corpusid": 3, "facet": "problems", "docid": "gamma delta"},
        {"corpusid": 4, "facet": "problems", "docid": "gamma delta"},
    ]
    with (c2t_dir / "paper_c2t_docids.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (c2t_dir / "l1_labels.json").write_text(
        json.dumps(
            {
                "problems": {
                    "0,0": ["alpha", "beta"],
                    "0,1": ["gamma", "delta"],
                    "0,2": ["alpha", "beta"],
                }
            }
        )
    )
    (c2t_dir / "c2t_params.json").write_text(
        json.dumps({"keyword_separator": " ", "level_separator": " "})
    )

    facets_path = tmp_path / "facets.jsonl"
    with facets_path.open("w") as fh:
        for cid, bullet in [
            (1, "alpha things"),
            (2, "alpha stuff"),
            (3, "gamma things"),
            (4, "gamma stuff"),
            (9, "alpha related query"),
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

    split_path = tmp_path / "test.txt"
    split_path.write_text("9\n")

    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(
        json.dumps({"paper_a": 9, "paper_b": 1, "relation": "reference", "score": 1.0})
        + "\n"
    )
    return c2t_dir, facets_path, split_path, pairs_path


class TestIntrinsicMetrics:
    """Collision, redundancy, faithfulness, separation on a toy index."""

    def test_metrics(self, tmp_path: Path) -> None:
        """Hand-checkable values on the two-leaf fixture."""
        c2t_dir, facets_path, split_path, pairs_path = _write_fixture(tmp_path)
        result = evaluate_c2t_labels(
            c2t_dir,
            facets_path,
            split_path,
            pairs_path,
            embed_fn=_fake_embed,
            facets=("problems",),
        )
        m = result.per_facet["problems"]
        assert m.n_leaves == 3
        assert m.n_distinct_docids == 2
        assert m.collision_rate == pytest.approx(1 / 3)
        # 'alpha beta' words -> e1 vs e3 (cos 0); 'gamma delta' -> e2 vs e3.
        assert m.mean_redundancy == pytest.approx(0.0)
        # Label 'alpha beta' -> e1; members 1,2 -> e1: cosine 1.0.
        assert m.mean_faithfulness == pytest.approx(1.0)
        # Other leaf centroid is orthogonal: separation 1.0 - 0.0.
        assert m.mean_separation == pytest.approx(1.0)


class TestZeroShotRetrieval:
    """Query routes to the right leaf and recalls the gold paper."""

    def test_fused_recall(self, tmp_path: Path) -> None:
        """Query 9 ('alpha...' -> e1) matches the leaf holding gold paper 1."""
        c2t_dir, facets_path, split_path, pairs_path = _write_fixture(tmp_path)
        result = evaluate_c2t_labels(
            c2t_dir,
            facets_path,
            split_path,
            pairs_path,
            embed_fn=_fake_embed,
            facets=("problems",),
        )
        assert result.n_queries == 1
        assert result.fused_recall_at_20 == pytest.approx(1.0)
        assert result.fused_recall_at_50 == pytest.approx(1.0)
        assert result.per_facet["problems"].recall_at_20 == pytest.approx(1.0)
