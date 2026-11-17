"""Shared helpers for deriving pairwise citation relationships."""

from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ScoredPair[IdT]:
    """A scored pair of items with a numeric overlap or co-occurrence count."""

    id_a: IdT
    id_b: IdT
    score: int


def count_group_cooccurrences[IdT: Hashable](
    groups: Iterable[Iterable[IdT]],
    *,
    max_group_size: int | None = None,
) -> tuple[Counter[tuple[IdT, IdT]], int]:
    """Count unordered pair co-occurrences across item groups.

    Args:
        groups: Iterable of item groups. Each group contributes +1 to all unordered
            item pairs within that group.
        max_group_size: If set, groups above this size are skipped.

    Returns:
        Tuple of (pair_counts, skipped_groups_count).
    """
    pair_counts: Counter[tuple[IdT, IdT]] = Counter()
    skipped_groups = 0

    for group in groups:
        # Sort by string form for deterministic ordering without requiring
        # the ID type itself to support rich ordering.
        unique_items = sorted(set(group), key=repr)
        if max_group_size is not None and len(unique_items) > max_group_size:
            skipped_groups += 1
            continue

        n_items = len(unique_items)
        for i in range(n_items):
            for j in range(i + 1, n_items):
                pair_counts[(unique_items[i], unique_items[j])] += 1

    return pair_counts, skipped_groups


def build_inverse_index[IdT: Hashable](
    mapping: Mapping[IdT, Iterable[IdT]],
) -> dict[IdT, set[IdT]]:
    """Invert a mapping of source->neighbors into neighbor->set[sources]."""
    inverse: dict[IdT, set[IdT]] = defaultdict(set)
    for source, neighbors in mapping.items():
        for neighbor in set(neighbors):
            inverse[neighbor].add(source)
    return inverse


def counts_to_scored_pairs[IdT: Hashable](
    pair_counts: Counter[tuple[IdT, IdT]],
    min_count: int,
) -> list[ScoredPair[IdT]]:
    """Convert pair counts into scored pairs, sorted by score descending."""
    scored = [
        ScoredPair(id_a=paper_a, id_b=paper_b, score=score)
        for (paper_a, paper_b), score in pair_counts.items()
        if score >= min_count
    ]
    scored.sort(key=lambda item: (-item.score, str(item.id_a), str(item.id_b)))
    return scored


def compute_overlap_pairs[IdT: Hashable](
    mapping: Mapping[IdT, Iterable[IdT]],
    min_overlap: int,
) -> list[ScoredPair[IdT]]:
    """Compute overlap pairs between mapping keys by shared neighbors.

    Example:
        mapping = {paper -> citing_papers}
        output = paper pairs scored by overlap in citing papers.
    """
    inverse = build_inverse_index(mapping)
    counts, _ = count_group_cooccurrences(inverse.values())
    return counts_to_scored_pairs(counts, min_overlap)
