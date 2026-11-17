"""Tests for C2T keyword scoring and selection modes."""

import numpy as np
import pytest

from genret.paper_parsing import Facets
from genret.training.c2t_index import C2TIndexParams, build_c2t_facet
from genret.training.cluster_keyword_labels import aggregate_cluster_labels
from genret.types import CorpusId


def _unit(x: float, y: float) -> np.ndarray:
    v = np.array([x, y], dtype=np.float32)
    return v / np.linalg.norm(v)


class TestLogOddsScoring:
    """Log-odds with Dirichlet prior demotes ambient words."""

    def test_exclusive_word_beats_ambient_word(self) -> None:
        """A concentrated word outranks a higher-count corpus-wide word."""
        paper_keywords = {
            CorpusId(1): ["ambient"] * 10 + ["graphy"] * 6,
            CorpusId(2): ["ambient"] * 12 + ["other"] * 5,
        }
        assignments = {CorpusId(1): "a", CorpusId(2): "b"}

        freq = aggregate_cluster_labels(
            paper_keywords, assignments, top_k=1, scoring="frequency"
        )
        logodds = aggregate_cluster_labels(
            paper_keywords, assignments, top_k=1, scoring="logodds"
        )
        assert freq["a"] == ["ambient"]
        assert logodds["a"] == ["graphy"]

    def test_deterministic_tie_break(self) -> None:
        """Symmetric counts break ties alphabetically."""
        paper_keywords = {
            CorpusId(1): ["zeta", "alpha"],
            CorpusId(2): ["mu", "nu"],
        }
        assignments = {CorpusId(1): "a", CorpusId(2): "b"}
        out = aggregate_cluster_labels(
            paper_keywords, assignments, top_k=2, scoring="logodds"
        )
        assert out["a"] == ["alpha", "zeta"]

    def test_unknown_scoring_raises(self) -> None:
        """Unknown scoring mode raises ValueError."""
        with pytest.raises(ValueError, match="scoring"):
            aggregate_cluster_labels(
                {CorpusId(1): ["w"]}, {CorpusId(1): "a"}, scoring="nope"
            )


class TestEmbeddingScoring:
    """Embedding mode ranks by cosine to the cluster centroid."""

    def test_centroid_near_words_win(self) -> None:
        """With equal counts, words near the centroid outrank far ones."""
        paper_keywords = {CorpusId(1): ["apple", "zebra", "zest"]}
        assignments = {CorpusId(1): "a"}
        word_vectors = {
            "apple": _unit(0.0, 1.0),
            "zebra": _unit(1.0, 0.0),
            "zest": _unit(0.9, 0.1),
        }
        out = aggregate_cluster_labels(
            paper_keywords,
            assignments,
            top_k=2,
            scoring="embedding",
            word_vectors=word_vectors,
        )
        assert out["a"] == ["zest", "zebra"]

    def test_embedding_without_vectors_raises(self) -> None:
        """scoring='embedding' requires word_vectors."""
        with pytest.raises(ValueError, match="word_vectors"):
            aggregate_cluster_labels(
                {CorpusId(1): ["w"]}, {CorpusId(1): "a"}, scoring="embedding"
            )


class TestMmrSelection:
    """MMR swaps a redundant runner-up for a diverse third candidate."""

    def test_redundant_runner_up_replaced(self) -> None:
        """Near-identical top words: MMR keeps one, adds the diverse word."""
        paper_keywords = {
            CorpusId(1): ["graph"] * 10 + ["graphs"] * 9 + ["protein"] * 5,
        }
        assignments = {CorpusId(1): "a"}
        word_vectors = {
            "graph": _unit(1.0, 0.0),
            "graphs": _unit(0.999, 0.045),
            "protein": _unit(0.0, 1.0),
        }
        topk = aggregate_cluster_labels(
            paper_keywords, assignments, top_k=2, scoring="frequency"
        )
        mmr = aggregate_cluster_labels(
            paper_keywords,
            assignments,
            top_k=2,
            scoring="frequency",
            selection="mmr",
            mmr_lambda=0.5,
            word_vectors=word_vectors,
        )
        assert topk["a"] == ["graph", "graphs"]
        assert mmr["a"] == ["graph", "protein"]

    def test_mmr_without_vectors_raises(self) -> None:
        """selection='mmr' requires word_vectors."""
        with pytest.raises(ValueError, match="word_vectors"):
            aggregate_cluster_labels(
                {CorpusId(1): ["w"]}, {CorpusId(1): "a"}, selection="mmr"
            )

    def test_unknown_selection_raises(self) -> None:
        """Unknown selection mode raises ValueError."""
        with pytest.raises(ValueError, match="selection"):
            aggregate_cluster_labels(
                {CorpusId(1): ["w"]}, {CorpusId(1): "a"}, selection="nope"
            )


def _fake_embedder(texts: list[str]) -> np.ndarray:
    """Deterministic 2-d unit vectors: x-axis if 'graph' in the word."""
    return np.stack(
        [_unit(1.0, 0.0) if "graph" in t else _unit(0.0, 1.0) for t in texts]
    ).astype(np.float32)


class TestC2TParamsWiring:
    """New params flow from C2TIndexParams into label aggregation."""

    def test_needs_word_vectors(self) -> None:
        """Embedding scoring or MMR selection require word vectors."""
        assert not C2TIndexParams().needs_word_vectors
        assert C2TIndexParams(scoring="embedding").needs_word_vectors
        assert C2TIndexParams(selection="mmr").needs_word_vectors

    def test_build_c2t_facet_with_embedder(self) -> None:
        """build_c2t_facet builds word vectors via the injected embedder."""
        facets_map = {
            CorpusId(1): Facets(
                corpusid=CorpusId(1),
                problems=["graph subgraph networks"],
                methods=[],
                results=[],
                contributions=[],
            ),
            CorpusId(2): Facets(
                corpusid=CorpusId(2),
                problems=["protein folding dynamics"],
                methods=[],
                results=[],
                contributions=[],
            ),
        }
        paper_codes = {CorpusId(1): (0, 0), CorpusId(2): (0, 1)}
        params = C2TIndexParams(top_k=1, scoring="embedding")
        artifacts = build_c2t_facet(
            "problems",
            paper_codes,
            facets_map,
            params,
            embedder=_fake_embedder,
        )
        # Paper 1's tokens: graph, subgraph (x-axis), networks (y-axis).
        # Count-weighted centroid leans x, so a 'graph' word must win;
        # alphabetical tie-break between graph/subgraph picks 'graph'.
        assert artifacts.l1_labels[(0, 0)] == ["graph"]
