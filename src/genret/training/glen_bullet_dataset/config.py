"""Configuration for the per-bullet GLEN training dataset (D9)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from genret.paper_parsing import FacetRecordType
from genret.training.codebook_data import FACET_NAMES
from genret.types import CorpusId


@dataclass(frozen=True, kw_only=True)
class GlenBulletDatasetConfig:
    """Configuration for the per-bullet GLEN training dataset (D9)."""

    pairs_path: Path
    """Training pairs JSONL from build-codebook-data (only retrieval rows used)."""
    facets_path: Path
    """JSONL with per-paper facet bullets and optional evidence."""
    paper_graphs_path: Path
    """JSONL with per-paper metadata; must have 'title' and 'corpusid' fields."""
    facets_type: FacetRecordType = FacetRecordType.STANDARD
    split_ids: set[CorpusId] | None = None
    """If set, restrict to pairs where paper_a is in this split."""
    facets: tuple[str, ...] = FACET_NAMES
    """Restrict examples to these facets. Defaults to all four."""
    source_bullets: Literal["all", "longest"] = "all"
    """How to select paper_a bullets per gold link:
    - "all" (D9 default): one example per paper_a bullet (~4 rows per pair).
    - "longest": one example using paper_a's longest bullet (1 row per pair).
    """
    target_bullet: Literal["random", "longest", "cartesian"] = "random"
    """How to select paper_b's target bullet:
    - "random" (D9 default): sample uniformly per (paper_a bullet) row.
    - "longest": always use paper_b's longest bullet (fixed target).
    - "cartesian": emit one example per (paper_a bullet, paper_b bullet) pair.
      Inflates the dataset by paper_b's bullet count for each gold pair, but
      the model sees every paper_b target deterministically rather than via
      stochastic sampling across epochs.
    """

    graph_sampling: Literal["uniform", "weighted"] = "uniform"
    """Per-pair sampling distribution:
    - "uniform" (default): every (paper_a, paper_b, facet) gold pair
      contributes its standard number of examples once.
    - "weighted": replicate each pair's examples by an integer count
      proportional to ``scores_combined`` for that pair, summed across
      facets. Pairs not in any per-facet top-100 still contribute 1
      example (a floor) so no gold is dropped; pairs with strong graph
      edges are oversampled. Requires ``candidates_dir`` set.
    """

    candidates_dir: Path | None = None
    """Directory with ``facet_candidates_<facet>.npz`` files. Required when
    ``graph_sampling="weighted"``."""

    graph_sampling_max_copies: int = 5
    """Maximum integer replication factor when ``graph_sampling="weighted"``.
    Pair with the strongest graph weight in the training data gets this
    many copies; weaker pairs scale down linearly, with a floor of 1."""

    leaf_prefix_dir: Path | None = None
    """If set, prepend the per-(facet, paper_b) C2T-TFIDF cluster label to
    each target bullet. Used for the D1 variant (graph-token-prefix
    DocIDs). The path is the same directory passed to ``build-c2t-index``
    — it must contain ``paper_c2t_docids.jsonl``. Empty / missing entries
    fall back to no prefix (raw bullet only) so the training does not
    silently drop pairs."""

    leaf_prefix_separator: str = " "
    """Inserted between the C2T-TFIDF prefix and the bullet text. Must
    match the separator used by ``build_d1_index``."""

    min_target_views: int = 0
    """Minimum number of training examples per (paper_b, facet, bullet)
    target. ``0`` (default) = disabled. When > 0, after the main
    enumeration we count target occurrences and inject extra
    (paper_a, paper_b, source_bullet, this_paper_b_bullet) rows for any
    bullet seen fewer times than this. Solves the DocID-starvation
    side-effect of graph-weighted sampling: every paper_b bullet gets
    at least ``min_target_views`` training rows regardless of how many
    gold edges paper_b has or how strong those edges are."""

    graph_reweighting: Literal["off", "weighted"] = "off"
    """Per-row CE loss reweighting:
    - "off" (default): every example contributes the standard CE loss.
    - "weighted": multiply the per-row CE by a scalar ``w`` derived from
      ``scores_combined`` for the pair, capped at
      ``graph_reweighting_max_weight``. Continuous analogue of
      ``graph_sampling="weighted"`` — can be stacked with sampling (both
      shape the gradient) or replace it (reweight only, fewer rows).
      Requires ``candidates_dir`` set."""

    graph_reweighting_max_weight: float = 5.0
    """Ceiling for per-row loss weight when ``graph_reweighting="weighted"``.
    Mirrors ``graph_sampling_max_copies``: a row whose pair has the
    strongest per-facet ``scores_combined`` gets this weight, weaker
    rows scale linearly down to a floor of 1.0."""

    target_smoothing_epsilon: float = 0.0
    """Graph-smoothed soft targets (gold-slice, first-divergence form).
    ``0`` (default) = disabled. When > 0, every example carries sparse
    soft-target entries: for each other gold paper_c of (paper_a, facet),
    the first token position where paper_c's bullet DocIDs diverge from
    the row's target gets label mass, weighted by the
    ``scores_combined(paper_a, paper_c)`` graph edge. The epsilon itself
    is applied by ``SmoothedTargetTrainer`` at loss time; the dataset
    only computes the (position, token, weight) entries. Requires a
    tokeniser passed to ``GlenBulletDataset``."""

    target_smoothing_max_entries: int = 64
    """Cap on soft-target entries per row (kept by descending weight,
    then renormalised per position). Bounds memory for queries with
    large gold slices."""

    hard_negatives_source: Literal["off", "graph_candidates"] = "off"
    """Per-row hard negative sampler:
    - "off" (default): no negatives attached to rows.
    - "graph_candidates": for each (paper_a, facet) row, sample K bullets
      from papers in the top-``hard_negatives_pool_top_k`` of
      ``facet_candidates_<facet>.npz`` for paper_a, filtered to those
      not in paper_a's gold slice. Requires ``candidates_dir`` set.
      Consumed by ``WeightedCETrainer``: a per-row unlikelihood term is
      added to the standard CE loss, suppressing log P(neg | prompt)."""

    hard_negatives_pool_top_k: int = 50
    """Depth of the per-(paper_a, facet) candidate list to draw negatives
    from. Smaller = harder, narrower pool. Top-K candidates that are
    also in the gold slice for paper_a are filtered out."""

    num_hard_negatives: int = 8
    """Number of hard negative bullets sampled per training row. One
    bullet per sampled negative paper (uniform among that paper's
    bullets in the facet)."""

    hard_negatives_lambda: float = 1.0
    """Loss-mixing weight on the per-row unlikelihood term in
    ``WeightedCETrainer``. Final loss = CE(gold) + lambda * UL(neg).
    ``1.0`` balances the two terms equally."""

    min_views_graph_fallback: int = 0
    """Graph-pseudo-gold sources for the ``min_target_views`` floor.
    ``0`` (default) = disabled. When > 0, a paper_b whose gold paper_a
    pool is empty (gold-isolated: the floor cannot synthesise rows for
    it) instead samples paper_a from its top-N graph neighbours by
    ``scores_combined`` (N = this value). The graph substitutes for
    missing supervision exactly where supervision is absent — unlike
    sampling/reweighting, which only reshape existing gold. Requires
    ``candidates_dir``; the train-split filter still applies to the
    pseudo sources."""

    min_views_weighted_source: bool = False
    """(Experiment "a") Importance-sample the ``min_target_views`` floor's
    synthetic source instead of picking it uniformly. ``False`` (default) keeps
    the uniform pick. When ``True``, for each floor row the source ``paper_a`` is
    drawn from its pool with probability proportional to the ``scores_combined``
    edge weight to ``paper_b`` (symmetric lookup) — so the strongest gold /
    graph neighbour is favoured as the pseudo-query, without changing the floor's
    exposure *count* (unlike ``pseudo_gold_oversample_sources``, which does). Only
    affects rows added by the floor; requires ``candidates_dir``."""

    pseudo_gold_oversample_sources: int = 0
    """(Experiment "b") Seed synthetic pseudo-gold BEFORE the main expansion,
    so it flows through the same edge-weight oversampling + ``min_target_views``
    floor as real gold — instead of being synthesised uniformly at the floor.
    ``0`` (default) = disabled (keep the floor-only fallback behaviour). When
    > 0, each gold-isolated ``(facet, paper_b)`` — one with no train-source
    retrieval pair targeting it — is seeded with up to this many synthetic
    ``(neighbour -> paper_b)`` rows: paper_b's strongest graph neighbours (by
    ``scores_combined``) become the paper_a sources, flipping the *symmetric*
    edge into an incoming pair. Each row carries that edge weight, so
    ``graph_sampling='weighted'`` replicates strong pseudo edges more while weak
    ones stay near the floor. The ``min_target_views`` floor still runs after
    and guarantees the minimum. Requires ``candidates_dir``; pseudo sources are
    train-split filtered."""

    smoothed_bullets_path: Path | None = None
    """Graph-smoothed DocID map (JSONL from ``glen smooth-bullets``).
    When set, every target bullet string is replaced by its smoothed
    version — content words attested in the paper's graph neighbourhood,
    original order. Prompts (paper_a bullets) stay verbatim. The eval
    index must be built with the same map (``build-docids
    --smoothed-bullets``) so trie entries match training targets."""
