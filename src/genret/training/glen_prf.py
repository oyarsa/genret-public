# pyright: basic
"""Graph-PRF: two-pass beam biasing for GLEN decoding.

Pseudo-relevance feedback realised inside trie-constrained generation:
pass 1 decodes normally and fuses to a seed set R1; pass 2 re-decodes
with branch scores tilted toward trie regions whose reachable papers
have strong corpus-graph edges (``scores_combined``) to R1. Every result
is still generated through the trie — the graph never injects candidates,
it only moves the beam's decision boundary (small ``lambda`` = model
dominates; large ``lambda`` degenerates to expansion).

Bias form: potential-based shaping. Each step from node ``n`` to child
``c`` adds ``lambda * (log1p(phi(c)) - log1p(phi(n)))`` where ``phi`` is
the max edge weight from the node's reachable papers to R1. Along any
root-to-leaf path the terms telescope, so a completed DocID's total
bonus is exactly ``lambda * log1p(best edge of its papers to R1)`` —
independent of DocID length. The ``depth_gate`` starts shaping only
below a given depth, where reachable-paper sets are small enough for
``phi`` to be discriminative (near the root every branch reaches some
neighbour of R1 and the signal is mush).

Uses only corpus-internal edge weights — no encoder, no dense index.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import torch

from genret.training.glen_docids import GlenIndex
from genret.types import CorpusId

PairWeights = dict[tuple[str, CorpusId, CorpusId], float]


def load_prf_pair_weights(candidates_dir: Path) -> PairWeights:
    """Per-(facet, a, b) ``scores_combined`` edges for PRF biasing."""
    from genret.training.glen_bullet_dataset import (  # noqa: PLC0415
        _load_facet_pair_weights,
    )

    return _load_facet_pair_weights(candidates_dir)


def build_children_bias(
    facet: str,
    index: GlenIndex,
    seeds: set[CorpusId],
    pair_weights: PairWeights,
    *,
    lam: float,
    depth_gate: int = 4,
) -> dict[tuple[int, ...], list[tuple[int, float]]]:
    """Per-trie-node child biases for one (query, facet) given seed set R1.

    Returns ``prefix -> [(next_token, bias)]`` where bias is the shaping
    term ``lam * (log1p(phi(child)) - log1p(phi(prefix)))``. Prefixes
    shorter than ``depth_gate`` are absent (no bias near the root). Zero
    biases are dropped, so untouched branches cost nothing at decode time.
    """
    code_to_papers = index.code_to_papers.get(facet, {})
    if not code_to_papers or not seeds or lam <= 0:
        return {}

    # Max edge weight to the seed set per paper (undirected); direct
    # lookups per (paper, seed) beat scanning the full edge dict.
    paper_w: dict[CorpusId, float] = {}
    relevant_papers = {p for papers in code_to_papers.values() for p in papers}
    for p in relevant_papers:
        best = 0.0
        for s in seeds:
            w = pair_weights.get((facet, p, s), 0.0)
            if w > best:
                best = w
            w = pair_weights.get((facet, s, p), 0.0)
            if w > best:
                best = w
        if best > 0:
            paper_w[p] = best

    if not paper_w:
        return {}

    # phi per node: max paper weight over the subtree, built leaf-up.
    phi: dict[tuple[int, ...], float] = defaultdict(float)
    for seq, papers in code_to_papers.items():
        leaf_w = max((paper_w.get(p, 0.0) for p in papers), default=0.0)
        if leaf_w <= 0:
            continue
        for cut in range(len(seq), -1, -1):
            prefix = seq[:cut]
            if phi[prefix] >= leaf_w:
                break  # ancestors already at least this strong
            phi[prefix] = leaf_w

    # Children biases via shaping differences, gated by depth: no bias
    # for nodes at depth <= depth_gate; the first biased step uses a zero
    # baseline so the telescoped total for a leaf is exactly
    # lam * log1p(phi(leaf)).
    out: dict[tuple[int, ...], list[tuple[int, float]]] = {}
    for node, node_phi in phi.items():
        depth = len(node)
        if depth == 0 or depth <= depth_gate:
            continue
        parent = node[:-1]
        parent_phi = 0.0 if depth == depth_gate + 1 else phi.get(parent, 0.0)
        bias = lam * (math.log1p(node_phi) - math.log1p(parent_phi))
        if abs(bias) < 1e-9:
            continue
        out.setdefault(parent, []).append((node[-1], bias))
    return out


class GraphPrfLogitsProcessor:
    """Adds graph-shaping bias to trie-child logits during beam search.

    Composes with ``prefix_allowed_tokens_fn`` (which masks invalid
    tokens) — the bias only ever touches valid trie children, and only
    below the depth gate (encoded in the children-bias map).
    """

    def __init__(
        self,
        *,
        children_bias: dict[tuple[int, ...], list[tuple[int, float]]],
        prompt_len: int,
        eos_id: int,
    ) -> None:
        self.children_bias = children_bias
        self.prompt_len = prompt_len
        self.eos_id = eos_id

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Add child biases for each beam's current trie node."""
        if not self.children_bias:
            return scores
        for b in range(input_ids.shape[0]):
            gen = tuple(int(t) for t in input_ids[b, self.prompt_len :].tolist())
            while gen and gen[-1] == self.eos_id:
                gen = gen[:-1]
            for token, bias in self.children_bias.get(gen, ()):
                scores[b, token] += bias
        return scores


__all__ = [
    "GraphPrfLogitsProcessor",
    "build_children_bias",
    "load_prf_pair_weights",
]
