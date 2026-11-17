"""Dataset constructing per-bullet GLEN training examples (D9).

Each (paper_a, paper_b, facet) retrieval pair expands into one example per
bullet of paper_a's facet (~4x expansion vs the standard per-facet dataset).

Input format (chat user turn):
    Title: {paper_a title}
    {Facet}:
    - {bullet_i}
      > "{evidence_i_0}"

Target: a randomly sampled normalised bullet of paper_b's same facet.
No indexing pairs — 99.9% of corpus papers are reachable as retrieval targets.

This package is split into focused submodules:

``config``
    ``GlenBulletDatasetConfig`` — all knobs (sampling, reweighting,
    smoothing, hard negatives, min-target-views floor).
``rendering``
    Prompt message and bullet rendering helpers.
``selection``
    Title loading and paper_a / paper_b bullet selection.
``graph_weights``
    Per-pair graph edge weights and sampling / loss scaling.
``soft_targets``
    First-divergence soft-target builder for graph smoothing.
``hard_negatives``
    Per-row hard negative pools and sampling.
``gold``
    Gold (facet, paper_a) → paper_b mapping.
``builder``
    Main example enumeration and min-target-views augmentation.
``dataset``
    ``GlenBulletDataset`` wrapper.

Public names (and the cross-module helpers other datasets reuse) are
re-exported here, so existing
``from genret.training.glen_bullet_dataset import X`` imports keep working.
"""

from __future__ import annotations

# Cross-module helpers re-exported with redundant aliases so existing
# `from genret.training.glen_bullet_dataset import _x` imports keep working.
from .builder import _augment_for_min_target_views as _augment_for_min_target_views
from .builder import _build_bullet_examples as _build_bullet_examples
from .config import GlenBulletDatasetConfig
from .dataset import GlenBulletDataset
from .gold import _load_gold_by_query as _load_gold_by_query
from .graph_weights import _load_facet_pair_weights as _load_facet_pair_weights
from .graph_weights import _pair_loss_weight as _pair_loss_weight
from .graph_weights import _pair_replication_count as _pair_replication_count
from .hard_negatives import _load_hard_negative_pools as _load_hard_negative_pools
from .hard_negatives import (
    _sample_hard_negative_bullets as _sample_hard_negative_bullets,
)
from .rendering import _build_bullet_messages as _build_bullet_messages
from .rendering import _render_bullet as _render_bullet
from .rendering import _word_count as _word_count
from .selection import _all_target_bullets as _all_target_bullets
from .selection import _load_titles as _load_titles
from .selection import _longest_source_index as _longest_source_index
from .selection import _longest_target as _longest_target
from .soft_targets import _lcp_len as _lcp_len
from .soft_targets import _SoftTargetBuilder as _SoftTargetBuilder

__all__ = [
    "GlenBulletDataset",
    "GlenBulletDatasetConfig",
]
