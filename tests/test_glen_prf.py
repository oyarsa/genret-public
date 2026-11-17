"""Tests for graph-PRF beam-biasing potentials and shaping."""

import math

import pytest
import torch

from genret.training.glen_docids import GlenIndex, _build_trie
from genret.training.glen_prf import GraphPrfLogitsProcessor, build_children_bias
from genret.types import CorpusId


def _make_index() -> GlenIndex:
    """Build a tiny index.

    Two DocIDs sharing a prefix: (1,2,3)->paper 10, (1,2,4)->paper 20,
    plus (5,6)->paper 30.
    """
    code_to_papers: dict[str, dict[tuple[int, ...], list[CorpusId]]] = {
        "problems": {
            (1, 2, 3): [CorpusId(10)],
            (1, 2, 4): [CorpusId(20)],
            (5, 6): [CorpusId(30)],
        }
    }
    trie = {"problems": _build_trie(list(code_to_papers["problems"].keys()))}
    return GlenIndex(code_to_papers=code_to_papers, trie=trie)


class TestBuildChildrenBias:
    """Shaping potentials telescope to lam * log1p(best edge)."""

    def test_telescoped_total_equals_leaf_potential(self) -> None:
        """Sum of biases along a path equals lam*log1p(w) of its paper."""
        index = _make_index()
        seeds = {CorpusId(99)}
        pair_weights = {("problems", CorpusId(10), CorpusId(99)): 0.8}
        bias = build_children_bias(
            "problems", index, seeds, pair_weights, lam=0.5, depth_gate=0
        )
        # Path to paper 10: () -> (1,) -> (1,2) -> (1,2,3).
        total = 0.0
        for prefix, tok in [((), 1), ((1,), 2), ((1, 2), 3)]:
            total += dict(bias.get(prefix, ())).get(tok, 0.0)
        assert total == pytest.approx(0.5 * math.log1p(0.8))

    def test_unrelated_branch_gets_no_bias(self) -> None:
        """Branches reaching no seed-neighbour are absent from the map."""
        index = _make_index()
        pair_weights = {("problems", CorpusId(10), CorpusId(99)): 0.8}
        bias = build_children_bias(
            "problems", index, {CorpusId(99)}, pair_weights, lam=0.5, depth_gate=0
        )
        all_biased_tokens = {t for kids in bias.values() for t, _ in kids}
        assert 5 not in all_biased_tokens  # paper 30's branch untouched

    def test_depth_gate_suppresses_shallow_bias(self) -> None:
        """Gate=2: no bias at depths 1-2; telescoped total preserved."""
        index = _make_index()
        pair_weights = {("problems", CorpusId(10), CorpusId(99)): 0.8}
        bias = build_children_bias(
            "problems", index, {CorpusId(99)}, pair_weights, lam=0.5, depth_gate=2
        )
        assert () not in bias
        assert (1,) not in bias
        step3 = dict(bias[(1, 2)])[3]
        assert step3 == pytest.approx(0.5 * math.log1p(0.8))


class TestGraphPrfLogitsProcessor:
    """Processor adds the mapped bias to the right beam rows/tokens."""

    def test_bias_applied_at_node(self) -> None:
        """A beam sitting at a biased node gets its child token boosted."""
        proc = GraphPrfLogitsProcessor(
            children_bias={(1, 2): [(3, 0.7)]},
            prompt_len=2,
            eos_id=0,
        )
        input_ids = torch.tensor([[9, 9, 1, 2], [9, 9, 5, 6]])
        scores = torch.zeros((2, 10))
        out = proc(input_ids, scores)
        assert out[0, 3] == pytest.approx(0.7)
        assert out[1].abs().sum() == 0.0
