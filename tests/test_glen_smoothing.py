"""Tests for graph-smoothed soft targets (builder, collator entries, loss)."""

from __future__ import annotations

import torch

from genret.paper_parsing import Facets
from genret.training.glen_bullet_dataset import _lcp_len, _SoftTargetBuilder
from genret.training.glen_smoothed_loss import smoothed_cross_entropy
from genret.types import CorpusId

EOS = 999
"""EOS token id used by the fake word-level tokeniser."""

_VOCAB: dict[str, int] = {}
"""Word-to-id map populated lazily by `_encode`."""


def _encode(text: str) -> list[int]:
    """Word-level fake tokeniser: each unique word gets a stable id."""
    ids: list[int] = []
    for word in text.split():
        if word not in _VOCAB:
            _VOCAB[word] = len(_VOCAB) + 1
        ids.append(_VOCAB[word])
    return ids


def _facets(corpusid: int, methods: list[str]) -> Facets:
    """Facets record with only the methods facet populated."""
    return Facets(
        corpusid=CorpusId(corpusid),
        problems=[],
        methods=methods,
        results=[],
        contributions=[],
    )


def _builder(
    facets_map: dict[CorpusId, Facets],
    gold: dict[tuple[str, CorpusId], frozenset[CorpusId]],
    weights: dict[tuple[str, CorpusId, CorpusId], float] | None = None,
    max_entries: int = 64,
) -> _SoftTargetBuilder:
    """Construct a builder with no leaf prefixes and the fake tokeniser."""
    return _SoftTargetBuilder(
        gold_by_query=gold,
        facets_map=facets_map,
        leaf_prefix_map={},
        leaf_prefix_separator=" ",
        pair_weights=weights,
        encode=_encode,
        eos_token_id=EOS,
        max_entries=max_entries,
    )


A, B, C, D = CorpusId(1), CorpusId(2), CorpusId(3), CorpusId(4)
"""Corpus ids used across tests: A = query, B = target paper, C/D = slice."""


class TestLcpLen:
    """Tests for the longest-common-prefix helper."""

    def test_no_overlap(self) -> None:
        """Sequences differing at position 0 have LCP 0."""
        assert _lcp_len((1, 2), (3, 4)) == 0

    def test_partial_overlap(self) -> None:
        """LCP stops at the first differing token."""
        assert _lcp_len((1, 2, 3), (1, 2, 4)) == 2

    def test_prefix(self) -> None:
        """A strict prefix has LCP equal to its own length."""
        assert _lcp_len((1, 2), (1, 2, 3)) == 2


class TestSoftTargetBuilder:
    """Tests for first-divergence entry computation."""

    def test_divergence_position_and_token(self) -> None:
        """Neighbour diverging at position 2 contributes its token there."""
        facets_map = {
            B: _facets(2, ["graph neural citation"]),
            C: _facets(3, ["graph neural node"]),
        }
        gold = {("methods", A): frozenset({B, C})}
        builder = _builder(facets_map, gold)
        entries = builder.entries_for("methods", A, B, "graph neural citation")
        assert entries is not None
        assert len(entries) == 1
        pos, token, weight = entries[0]
        assert pos == 2
        assert token == _encode("node")[0]
        assert weight == 1.0

    def test_no_neighbours_returns_none(self) -> None:
        """A slice containing only the target paper yields no entries."""
        facets_map = {B: _facets(2, ["graph neural citation"])}
        gold = {("methods", A): frozenset({B})}
        builder = _builder(facets_map, gold)
        assert builder.entries_for("methods", A, B, "graph neural citation") is None

    def test_identical_bullet_skipped(self) -> None:
        """A neighbour bullet identical to the target contributes nothing."""
        facets_map = {
            B: _facets(2, ["graph neural citation"]),
            C: _facets(3, ["graph neural citation"]),
        }
        gold = {("methods", A): frozenset({B, C})}
        builder = _builder(facets_map, gold)
        assert builder.entries_for("methods", A, B, "graph neural citation") is None

    def test_strict_prefix_neighbour_contributes_eos(self) -> None:
        """A neighbour that is a strict prefix competes with its EOS."""
        facets_map = {
            B: _facets(2, ["graph neural citation"]),
            C: _facets(3, ["graph neural"]),
        }
        gold = {("methods", A): frozenset({B, C})}
        builder = _builder(facets_map, gold)
        entries = builder.entries_for("methods", A, B, "graph neural citation")
        assert entries == ((2, EOS, 1.0),)

    def test_extension_neighbour_competes_at_target_eos(self) -> None:
        """A neighbour extending past the target lands on the EOS slot."""
        facets_map = {
            B: _facets(2, ["graph neural"]),
            C: _facets(3, ["graph neural citation"]),
        }
        gold = {("methods", A): frozenset({B, C})}
        builder = _builder(facets_map, gold)
        entries = builder.entries_for("methods", A, B, "graph neural")
        assert entries == ((2, _encode("citation")[0], 1.0),)

    def test_weights_normalised_per_position(self) -> None:
        """Two neighbours at the same position split mass by edge weight."""
        facets_map = {
            B: _facets(2, ["graph neural citation"]),
            C: _facets(3, ["graph neural node"]),
            D: _facets(4, ["graph neural edge"]),
        }
        gold = {("methods", A): frozenset({B, C, D})}
        weights = {
            ("methods", A, C): 3.0,
            ("methods", A, D): 1.0,
        }
        builder = _builder(facets_map, gold, weights)
        entries = builder.entries_for("methods", A, B, "graph neural citation")
        assert entries is not None
        by_token = {token: weight for _pos, token, weight in entries}
        assert by_token[_encode("node")[0]] == 0.75
        assert by_token[_encode("edge")[0]] == 0.25

    def test_missing_edge_gets_min_positive_weight(self) -> None:
        """A slice member without a graph edge falls back to the floor."""
        facets_map = {
            B: _facets(2, ["graph neural citation"]),
            C: _facets(3, ["graph neural node"]),
            D: _facets(4, ["graph neural edge"]),
        }
        gold = {("methods", A): frozenset({B, C, D})}
        weights = {("methods", A, C): 2.0}
        builder = _builder(facets_map, gold, weights)
        entries = builder.entries_for("methods", A, B, "graph neural citation")
        assert entries is not None
        by_token = {token: weight for _pos, token, weight in entries}
        # D has no edge -> floor = min positive = 2.0 -> equal split.
        assert by_token[_encode("node")[0]] == 0.5
        assert by_token[_encode("edge")[0]] == 0.5

    def test_max_entries_cap(self) -> None:
        """Only the strongest entries survive the cap."""
        facets_map = {
            B: _facets(2, ["graph neural citation"]),
            C: _facets(3, ["graph neural node"]),
            D: _facets(4, ["totally different bullet"]),
        }
        gold = {("methods", A): frozenset({B, C, D})}
        weights = {
            ("methods", A, C): 5.0,
            ("methods", A, D): 1.0,
        }
        builder = _builder(facets_map, gold, weights, max_entries=1)
        entries = builder.entries_for("methods", A, B, "graph neural citation")
        assert entries is not None
        assert len(entries) == 1
        assert entries[0][1] == _encode("node")[0]
        assert entries[0][2] == 1.0


class TestSmoothedCrossEntropy:
    """Tests for the loss function."""

    @staticmethod
    def _setup() -> tuple[torch.Tensor, torch.Tensor]:
        """Tiny batch: B=1, T=4, V=6; labels unmask positions 2 and 3."""
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 6)
        labels = torch.tensor([[-100, -100, 2, 3]])
        return logits, labels

    def test_no_soft_entries_matches_plain_ce(self) -> None:
        """With no entries the loss equals standard shifted CE."""
        logits, labels = self._setup()
        empty = torch.zeros(0, dtype=torch.long)
        loss = smoothed_cross_entropy(
            logits,
            labels,
            soft_batch=empty,
            soft_pos=empty,
            soft_token=empty,
            soft_weight=empty.float(),
            epsilon=0.1,
        )
        expected = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 6),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_soft_entry_mixes_gold_and_alternative(self) -> None:
        """A smoothed position contributes (1-eps) gold + eps alt mass."""
        logits, labels = self._setup()
        eps = 0.2
        # Smooth the label at sequence position 2 (gold token 2, alt token 4).
        loss = smoothed_cross_entropy(
            logits,
            labels,
            soft_batch=torch.tensor([0]),
            soft_pos=torch.tensor([2]),
            soft_token=torch.tensor([4]),
            soft_weight=torch.tensor([1.0]),
            epsilon=eps,
        )
        logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        # Shifted coords: label at seq pos 2 -> index 1; label at 3 -> index 2.
        expected = (
            -(1 - eps) * logprobs[0, 1, 2] - eps * logprobs[0, 1, 4] - logprobs[0, 2, 3]
        ) / 2
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_num_items_denominator(self) -> None:
        """An explicit num_items overrides the mask-count denominator."""
        logits, labels = self._setup()
        empty = torch.zeros(0, dtype=torch.long)
        loss_2 = smoothed_cross_entropy(
            logits,
            labels,
            soft_batch=empty,
            soft_pos=empty,
            soft_token=empty,
            soft_weight=empty.float(),
            epsilon=0.1,
            num_items=4,
        )
        loss_mask = smoothed_cross_entropy(
            logits,
            labels,
            soft_batch=empty,
            soft_pos=empty,
            soft_token=empty,
            soft_weight=empty.float(),
            epsilon=0.1,
        )
        assert torch.isclose(loss_2 * 2, loss_mask, atol=1e-5)
