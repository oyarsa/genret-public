"""First-divergence soft-target entries for graph-smoothed training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from genret.paper_parsing import Facets
from genret.types import CorpusId

from .selection import _all_target_bullets


def _lcp_len(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Length of the longest common prefix of two token sequences."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class _SoftTargetBuilder:
    """Computes first-divergence soft-target entries for graph smoothing.

    For a row (paper_a, facet) → target DocID of paper_b, the smoothing
    set is the bullet DocIDs of every other gold paper_c for
    (paper_a, facet). Each contributes mass at the first token position
    where it diverges from the target, weighted by the
    `scores_combined(paper_a, paper_c)` graph edge (papers without an
    edge fall back to the minimum positive weight in the slice; uniform
    when no slice member has an edge). Entries are capped at
    `max_entries` by descending weight and normalised to sum to 1 per
    position.

    Tokenisation must match `collate_fn` exactly (`add_special_tokens=False`,
    no leading space) so positions align with training labels. Results are
    cached per (facet, paper_a, paper_b, target) so replicated rows share
    one tuple object.
    """

    def __init__(
        self,
        *,
        gold_by_query: Mapping[tuple[str, CorpusId], frozenset[CorpusId]],
        facets_map: Mapping[CorpusId, Facets],
        leaf_prefix_map: Mapping[str, Mapping[CorpusId, str]],
        leaf_prefix_separator: str,
        pair_weights: Mapping[tuple[str, CorpusId, CorpusId], float] | None,
        encode: Callable[[str], list[int]],
        eos_token_id: int,
        max_entries: int,
    ) -> None:
        """Store inputs and initialise the token / entry caches."""
        self._gold_by_query = gold_by_query
        self._facets_map = facets_map
        self._leaf_prefix_map = leaf_prefix_map
        self._sep = leaf_prefix_separator
        self._pair_weights = pair_weights
        self._encode = encode
        self._eos_token_id = eos_token_id
        self._max_entries = max_entries
        self._token_cache: dict[str, tuple[int, ...]] = {}
        self._entry_cache: dict[
            tuple[str, CorpusId, CorpusId, str],
            tuple[tuple[int, int, float], ...] | None,
        ] = {}

    def _tokens(self, text: str) -> tuple[int, ...]:
        """Tokenise `text` as `collate_fn` does, with caching."""
        cached = self._token_cache.get(text)
        if cached is None:
            cached = tuple(self._encode(text))
            self._token_cache[text] = cached
        return cached

    def entries_for(
        self, facet: str, paper_a: CorpusId, paper_b: CorpusId, target: str
    ) -> tuple[tuple[int, int, float], ...] | None:
        """Soft-target entries for one row; `None` when nothing to smooth."""
        key = (facet, paper_a, paper_b, target)
        if key not in self._entry_cache:
            self._entry_cache[key] = self._compute(facet, paper_a, paper_b, target)
        return self._entry_cache[key]

    def _neighbour_weights(
        self, facet: str, paper_a: CorpusId, neighbours: Sequence[CorpusId]
    ) -> dict[CorpusId, float]:
        """Per-neighbour weights with min-positive fallback for missing edges."""
        raw = {
            c: (
                self._pair_weights.get((facet, paper_a, c), 0.0)
                if self._pair_weights is not None
                else 0.0
            )
            for c in neighbours
        }
        positive = [w for w in raw.values() if w > 0]
        floor = min(positive) if positive else 1.0
        return {c: (w if w > 0 else floor) for c, w in raw.items()}

    def _compute(
        self, facet: str, paper_a: CorpusId, paper_b: CorpusId, target: str
    ) -> tuple[tuple[int, int, float], ...] | None:
        """Accumulate divergence-point mass over the gold slice for one row."""
        neighbours = [
            c
            for c in self._gold_by_query.get((facet, paper_a), frozenset())
            if c != paper_b
        ]
        if not neighbours:
            return None
        target_tokens = self._tokens(target)
        if not target_tokens:
            return None

        weights = self._neighbour_weights(facet, paper_a, sorted(neighbours))
        prefix_map = self._leaf_prefix_map.get(facet, {})

        acc: dict[tuple[int, int], float] = {}
        for c, w_c in weights.items():
            facets_c = self._facets_map.get(c)
            if facets_c is None:
                continue
            prefix = prefix_map.get(c, "")
            for bullet in _all_target_bullets(facets_c, facet):
                neighbour_target = f"{prefix}{self._sep}{bullet}" if prefix else bullet
                ntok = self._tokens(neighbour_target)
                if not ntok or ntok == target_tokens:
                    continue
                p = _lcp_len(target_tokens, ntok)
                # Past the divergence point the prefix leaves the
                # neighbour's trie path, so only position p matters. A
                # neighbour that is a strict prefix of the target competes
                # with its own EOS; one that extends past the target
                # competes with the target's EOS slot at len(target).
                alt = self._eos_token_id if p == len(ntok) else ntok[p]
                gold = (
                    target_tokens[p] if p < len(target_tokens) else self._eos_token_id
                )
                if alt == gold:
                    continue
                acc[(p, alt)] = acc.get((p, alt), 0.0) + w_c
        if not acc:
            return None

        top = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[: self._max_entries]
        position_totals: dict[int, float] = {}
        for (p, _tok), w in top:
            position_totals[p] = position_totals.get(p, 0.0) + w
        return tuple(sorted((p, tok, w / position_totals[p]) for (p, tok), w in top))
