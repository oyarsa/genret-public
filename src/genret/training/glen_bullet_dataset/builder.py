# pyright: basic
"""Main per-bullet example enumeration and min-target-views augmentation."""

from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from genret._shed.transformers import get_eos_token_id
from genret.paper_parsing import (
    FacetEvidence,
    Facets,
    evidence_from_type,
    facets_from_type,
)
from genret.training.codebook_data import FACET_NAMES
from genret.training.data import TrainingExample
from genret.training.glen_dataset import _target_text
from genret.types import CorpusId
from genret.utils import iter_jsonl

from .config import GlenBulletDatasetConfig
from .gold import _load_gold_by_query
from .graph_weights import (
    _load_facet_pair_weights,
    _pair_loss_weight,
    _pair_replication_count,
    _replication_from_weight,
    _symmetric_weight,
)
from .hard_negatives import _load_hard_negative_pools, _sample_hard_negative_bullets
from .rendering import _build_bullet_messages, _render_bullet
from .selection import (
    _all_target_bullets,
    _load_titles,
    _longest_source_index,
    _longest_target,
)
from .soft_targets import _SoftTargetBuilder

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _build_synthetic_pseudo_gold_rows(
    config: GlenBulletDatasetConfig,
    facets_map: Mapping[CorpusId, Facets],
    pair_weights: dict[tuple[str, CorpusId, CorpusId], float] | None,
) -> list[dict[str, Any]]:
    """Seed ``neighbour -> gold-isolated`` pairs (experiment "b").

    A ``(facet, paper_b)`` is *gold-isolated* when no train-source retrieval
    pair targets it (out-edges but no in-edges — the reverse-kNN gap). For each
    such target we emit up to ``pseudo_gold_oversample_sources`` synthetic rows:
    paper_b's strongest graph neighbours (by ``scores_combined``) become the
    paper_a sources, flipping the *symmetric* edge into an incoming ``N -> B``
    pair. Each row carries ``_pseudo_weight`` (the ``(B, N)`` edge weight, which
    is the direction actually stored in the candidate graph) so the main loop
    replicates it by strength. Returned rows are chained ahead of the real gold
    so they ride the same oversampling + ``min_target_views`` floor.
    """
    n_src = config.pseudo_gold_oversample_sources
    if n_src <= 0 or pair_weights is None:
        return []

    split_ids = config.split_ids
    allowed_facets = set(config.facets)

    # Real gold targets (train-source only), per facet — what we must NOT reseed.
    gold_targets: set[tuple[str, CorpusId]] = set()
    for row in iter_jsonl(config.pairs_path):
        if str(row.get("kind", "")) != "retrieval":
            continue
        facet = str(row["facet"])
        if facet not in allowed_facets:
            continue
        a = CorpusId(int(row["paper_a"]))
        if split_ids is not None and a not in split_ids:
            continue
        gold_targets.add((facet, CorpusId(int(row["paper_b"]))))

    # paper_b's outgoing neighbours: (facet, b) -> [(weight, neighbour), ...].
    from collections import defaultdict  # noqa: PLC0415

    by_paper: dict[tuple[str, CorpusId], list[tuple[float, CorpusId]]] = defaultdict(
        list
    )
    for (facet, a, b), w in pair_weights.items():
        if facet not in allowed_facets:
            continue
        by_paper[(facet, a)].append((w, b))

    rows: list[dict[str, Any]] = []
    isolated = 0
    for paper_b, b_facets in facets_map.items():
        for facet in config.facets:
            if (facet, paper_b) in gold_targets:
                continue
            if not getattr(b_facets, facet, ()):
                continue  # no target bullets to train in this facet
            neigh = by_paper.get((facet, paper_b))
            if not neigh:
                continue
            isolated += 1
            neigh.sort(reverse=True)
            seeded = 0
            for w, n in neigh:
                if n == paper_b or n not in facets_map:
                    continue
                if split_ids is not None and n not in split_ids:
                    continue
                rows.append(
                    {
                        "paper_a": int(n),
                        "paper_b": int(paper_b),
                        "facet": facet,
                        "kind": "retrieval",
                        "_pseudo_weight": float(w),
                    }
                )
                seeded += 1
                if seeded >= n_src:
                    break

    logger.info(
        "Pseudo-gold oversampling: %d synthetic rows for %d gold-isolated "
        "(facet, paper_b) targets (up to %d source(s) each)",
        len(rows),
        isolated,
        n_src,
    )
    return rows


def _build_bullet_examples(
    config: GlenBulletDatasetConfig,
    *,
    rng: random.Random,
    tokeniser: PreTrainedTokenizerBase | None = None,
) -> list[TrainingExample]:
    """Load pairs + facets + titles, expand each pair into per-bullet examples."""
    facets_map: dict[CorpusId, Facets] = {}
    evidence_map: dict[CorpusId, FacetEvidence] = {}
    for raw in iter_jsonl(config.facets_path):
        f = facets_from_type(raw, config.facets_type)
        facets_map[f.corpusid] = f
        if ev := evidence_from_type(raw, config.facets_type):
            evidence_map[f.corpusid] = ev

    titles = _load_titles(config.paper_graphs_path)
    logger.info("Loaded %d titles from %s", len(titles), config.paper_graphs_path)

    smoothed_map: dict[tuple[CorpusId, str, str], str] | None = None
    if config.smoothed_bullets_path is not None:
        from genret.training.glen_graph_smoothing import (  # noqa: PLC0415
            load_smoothed_bullets,
        )

        smoothed_map = load_smoothed_bullets(config.smoothed_bullets_path)
        logger.info(
            "Loaded %d graph-smoothed DocID targets from %s",
            len(smoothed_map),
            config.smoothed_bullets_path,
        )

    def _smooth_target(paper_b: CorpusId, facet: str, target: str) -> str:
        if smoothed_map is None:
            return target
        return smoothed_map.get((paper_b, facet, target), target)

    leaf_prefix_map: dict[str, dict[CorpusId, str]] = {}
    if config.leaf_prefix_dir is not None:
        from genret.training.c2t_index import load_paper_c2t_docids  # noqa: PLC0415

        leaf_prefix_map = load_paper_c2t_docids(config.leaf_prefix_dir)
        logger.info(
            "Loaded leaf prefixes for %d facets from %s",
            len(leaf_prefix_map),
            config.leaf_prefix_dir,
        )

    pair_weights: dict[tuple[str, CorpusId, CorpusId], float] | None = None
    max_weight_for_facet: dict[str, float] = {}
    needs_pair_weights = (
        config.graph_sampling == "weighted"
        or config.graph_reweighting == "weighted"
        or config.pseudo_gold_oversample_sources > 0
        or config.min_views_weighted_source
    )
    if needs_pair_weights:
        if config.candidates_dir is None:
            raise ValueError(
                "graph_sampling='weighted', graph_reweighting='weighted', or "
                "pseudo_gold_oversample_sources>0 requires candidates_dir"
            )
        pair_weights = _load_facet_pair_weights(config.candidates_dir)
        # Per-facet max so the scaling is comparable across facets.
        for (f, _a, _b), w in pair_weights.items():
            if w > max_weight_for_facet.get(f, 0.0):
                max_weight_for_facet[f] = w
        logger.info(
            "Graph weighting active: per-facet max scores_combined = %s, "
            "sampling=%s (max_copies=%d), reweighting=%s (max_weight=%.2f)",
            {f: round(v, 3) for f, v in max_weight_for_facet.items()},
            config.graph_sampling,
            config.graph_sampling_max_copies,
            config.graph_reweighting,
            config.graph_reweighting_max_weight,
        )

    hard_neg_pools: dict[tuple[str, CorpusId], list[CorpusId]] | None = None
    gold_neighbours_map: dict[tuple[str, CorpusId], frozenset[CorpusId]] | None = None
    if config.hard_negatives_source == "graph_candidates":
        if config.candidates_dir is None:
            raise ValueError(
                "hard_negatives_source='graph_candidates' requires candidates_dir"
            )
        hard_neg_pools = _load_hard_negative_pools(
            config.candidates_dir,
            config.facets,
            config.hard_negatives_pool_top_k,
        )
        gold_neighbours_map = _load_gold_by_query(config)
        logger.info(
            "Hard negatives active: source=graph_candidates pool_top_k=%d K=%d lambda=%.2f",
            config.hard_negatives_pool_top_k,
            config.num_hard_negatives,
            config.hard_negatives_lambda,
        )

    smoother: _SoftTargetBuilder | None = None
    if config.target_smoothing_epsilon > 0:
        if tokeniser is None:
            raise ValueError("target_smoothing_epsilon > 0 requires a tokeniser")
        if pair_weights is None and config.candidates_dir is not None:
            pair_weights = _load_facet_pair_weights(config.candidates_dir)

        def _encode(text: str) -> list[int]:
            return tokeniser.encode(text, add_special_tokens=False)

        smoother = _SoftTargetBuilder(
            gold_by_query=_load_gold_by_query(config),
            facets_map=facets_map,
            leaf_prefix_map=leaf_prefix_map,
            leaf_prefix_separator=config.leaf_prefix_separator,
            pair_weights=pair_weights,
            encode=_encode,
            eos_token_id=get_eos_token_id(tokeniser),
            max_entries=config.target_smoothing_max_entries,
        )
        logger.info(
            "Target smoothing active: epsilon=%.3f, max_entries=%d, graph weights %s",
            config.target_smoothing_epsilon,
            config.target_smoothing_max_entries,
            "loaded" if pair_weights is not None else "absent (uniform slice)",
        )

    def _hard_negs_for(facet: str, paper_a: CorpusId) -> tuple[str, ...]:
        if hard_neg_pools is None or gold_neighbours_map is None:
            return ()
        pool = hard_neg_pools.get((facet, paper_a))
        if not pool:
            return ()
        gold = gold_neighbours_map.get((facet, paper_a), frozenset())
        return _sample_hard_negative_bullets(
            paper_a=paper_a,
            facet=facet,
            gold_neighbours=gold,
            pool=pool,
            facets_map=facets_map,
            num_neg=config.num_hard_negatives,
            rng=rng,
        )

    examples: list[TrainingExample] = []
    skipped = 0
    split_ids = config.split_ids
    sampling_active = config.graph_sampling == "weighted"

    # Experiment "b": synthetic pseudo-gold rows ride ahead of the real gold so
    # they get the same edge-weight oversampling (then the min_target_views
    # floor tops up whatever is still short).
    synthetic_rows = _build_synthetic_pseudo_gold_rows(config, facets_map, pair_weights)
    pair_rows = itertools.chain(synthetic_rows, iter_jsonl(config.pairs_path))

    for row in pair_rows:
        if str(row.get("kind", "")) != "retrieval":
            continue

        paper_a = CorpusId(int(row["paper_a"]))
        paper_b = CorpusId(int(row["paper_b"]))
        facet = str(row["facet"])

        if facet not in FACET_NAMES:
            continue
        if facet not in config.facets:
            continue
        if split_ids is not None and paper_a not in split_ids:
            continue

        facets_a = facets_map.get(paper_a)
        facets_b = facets_map.get(paper_b)
        if facets_a is None or facets_b is None:
            skipped += 1
            continue

        if config.source_bullets == "all":
            source_indices: list[int] = list(range(len(getattr(facets_a, facet))))
        else:
            idx = _longest_source_index(facets_a, facet)
            source_indices = [] if idx is None else [idx]

        title_a = titles.get(paper_a, "")
        evidence_a = evidence_map.get(paper_a)

        # When target is "random", resample per row so different paper_a bullets
        # get different paper_b targets (D9 default). When target is "longest",
        # resolve once per pair (fixed target across all paper_a rows).
        fixed_target: str | None = None
        if config.target_bullet == "longest":
            fixed_target = _longest_target(facets_b, facet)
            if not fixed_target:
                skipped += 1
                continue
            fixed_target = _smooth_target(paper_b, facet, fixed_target)

        # For cartesian, enumerate all of paper_b's bullets once.
        cartesian_targets: list[str] = []
        if config.target_bullet == "cartesian":
            cartesian_targets = [
                _smooth_target(paper_b, facet, t)
                for t in _all_target_bullets(facets_b, facet)
            ]
            if not cartesian_targets:
                skipped += 1
                continue

        pseudo_w = row.get("_pseudo_weight")
        if pseudo_w is not None:
            # Synthetic pseudo-gold: use the explicit (symmetric) edge weight.
            n_copies = (
                _replication_from_weight(
                    float(pseudo_w),
                    max_weight_for_facet.get(facet, 0.0),
                    config.graph_sampling_max_copies,
                )
                if sampling_active
                else 1
            )
        elif sampling_active:
            n_copies = _pair_replication_count(
                paper_a,
                paper_b,
                facet,
                pair_weights,
                max_weight_for_facet,
                config.graph_sampling_max_copies,
            )
        else:
            n_copies = 1
        loss_w = _pair_loss_weight(
            paper_a,
            paper_b,
            facet,
            pair_weights,
            max_weight_for_facet,
            config.graph_reweighting_max_weight,
            enabled=config.graph_reweighting == "weighted",
        )

        prefix = leaf_prefix_map.get(facet, {}).get(paper_b, "")
        sep = config.leaf_prefix_separator

        for _copy in range(n_copies):
            for i in source_indices:
                bullet_text = _render_bullet(facets_a, facet, i, evidence_a)
                if not bullet_text:
                    continue
                messages = _build_bullet_messages(facet, title_a, bullet_text)
                if config.target_bullet == "cartesian":
                    for target in cartesian_targets:
                        full_target = f"{prefix}{sep}{target}" if prefix else target
                        examples.append(
                            TrainingExample(
                                prompt_messages=messages,
                                target_docid=full_target,
                                soft_targets=smoother.entries_for(
                                    facet, paper_a, paper_b, full_target
                                )
                                if smoother
                                else None,
                                loss_weight=loss_w,
                                hard_negative_targets=_hard_negs_for(facet, paper_a),
                            )
                        )
                    continue
                if fixed_target is not None:
                    target = fixed_target
                else:
                    target = _target_text(facets_b, facet, rng)
                    if not target:
                        skipped += 1
                        continue
                    target = _smooth_target(paper_b, facet, target)
                full_target = f"{prefix}{sep}{target}" if prefix else target
                examples.append(
                    TrainingExample(
                        prompt_messages=messages,
                        target_docid=full_target,
                        soft_targets=smoother.entries_for(
                            facet, paper_a, paper_b, full_target
                        )
                        if smoother
                        else None,
                        loss_weight=loss_w,
                        hard_negative_targets=_hard_negs_for(facet, paper_a),
                    )
                )

    if skipped:
        logger.info("Skipped %d pairs (missing facets or empty target)", skipped)

    if config.min_target_views > 0:
        _augment_for_min_target_views(
            examples=examples,
            config=config,
            facets_map=facets_map,
            evidence_map=evidence_map,
            titles=titles,
            leaf_prefix_map=leaf_prefix_map,
            rng=rng,
            smoother=smoother,
            pair_weights=pair_weights,
            max_weight_for_facet=max_weight_for_facet,
            smoothed_map=smoothed_map,
            hard_neg_pools=hard_neg_pools,
            gold_neighbours_map=gold_neighbours_map,
        )

    logger.info(
        "GlenBulletDataset: %d examples (source_bullets=%s, target_bullet=%s)",
        len(examples),
        config.source_bullets,
        config.target_bullet,
    )
    return examples


def _augment_for_min_target_views(
    *,
    examples: list[TrainingExample],
    config: GlenBulletDatasetConfig,
    facets_map: Mapping[CorpusId, Facets],
    evidence_map: Mapping[CorpusId, FacetEvidence],
    titles: Mapping[CorpusId, str],
    leaf_prefix_map: Mapping[str, Mapping[CorpusId, str]],
    rng: random.Random,
    smoother: _SoftTargetBuilder | None = None,
    pair_weights: dict[tuple[str, CorpusId, CorpusId], float] | None = None,
    max_weight_for_facet: dict[str, float] | None = None,
    smoothed_map: dict[tuple[CorpusId, str, str], str] | None = None,
    hard_neg_pools: dict[tuple[str, CorpusId], list[CorpusId]] | None = None,
    gold_neighbours_map: dict[tuple[str, CorpusId], frozenset[CorpusId]] | None = None,
) -> None:
    """Inject extra rows so each (paper_b, facet, bullet) target hits a floor.

    Walks the existing examples to count target-string occurrences, then
    enumerates every (paper_b, facet, bullet) in the corpus. For each
    bullet whose target string is under-represented vs
    ``config.min_target_views``, samples a paper_a from paper_b's gold
    neighbours and appends a fresh (paper_a source bullet → this paper_b
    bullet) example.
    """
    from collections import Counter, defaultdict  # noqa: PLC0415

    # 1. Re-scan gold to build paper_b → list of paper_a's that share an
    #    edge (so we can find a source paper_a per under-trained DocID).
    reverse_gold: dict[tuple[str, CorpusId], list[CorpusId]] = defaultdict(list)
    split_ids = config.split_ids
    for row in iter_jsonl(config.pairs_path):
        if str(row.get("kind", "")) != "retrieval":
            continue
        a = CorpusId(int(row["paper_a"]))
        b = CorpusId(int(row["paper_b"]))
        facet = str(row["facet"])
        if facet not in FACET_NAMES or facet not in config.facets:
            continue
        if split_ids is not None and a not in split_ids:
            continue
        reverse_gold[(facet, b)].append(a)

    # 2. Count current target occurrences.
    target_counts: Counter[str] = Counter(ex.target_docid for ex in examples)

    # Graph-pseudo-gold fallback: for gold-isolated paper_b's, the top-N
    # graph neighbours (by scores_combined) stand in as paper_a sources.
    graph_neighbours: dict[tuple[str, CorpusId], list[CorpusId]] = {}
    if config.min_views_graph_fallback > 0 and pair_weights:
        by_paper: dict[tuple[str, CorpusId], list[tuple[float, CorpusId]]] = (
            defaultdict(list)
        )
        for (facet, a, b), w in pair_weights.items():
            by_paper[(facet, a)].append((w, b))
        for key, weighted in by_paper.items():
            weighted.sort(reverse=True)
            graph_neighbours[key] = [
                b for _, b in weighted[: config.min_views_graph_fallback]
            ]

    sep = config.leaf_prefix_separator
    added = 0
    added_via_graph = 0
    skipped_no_source = 0
    for paper_b, b_facets in facets_map.items():
        for facet in config.facets:
            prefix = leaf_prefix_map.get(facet, {}).get(paper_b, "")
            for raw_bullet in _all_target_bullets(b_facets, facet):
                bullet = (
                    smoothed_map.get((paper_b, facet, raw_bullet), raw_bullet)
                    if smoothed_map is not None
                    else raw_bullet
                )
                target = f"{prefix}{sep}{bullet}" if prefix else bullet
                deficit = config.min_target_views - target_counts.get(target, 0)
                if deficit <= 0:
                    continue
                paper_a_pool = reverse_gold.get((facet, paper_b), [])
                via_graph = False
                if not paper_a_pool and graph_neighbours:
                    via_graph = True
                    paper_a_pool = [
                        a
                        for a in graph_neighbours.get((facet, paper_b), [])
                        if a in facets_map and (split_ids is None or a in split_ids)
                    ]
                if not paper_a_pool:
                    skipped_no_source += 1
                    continue
                # (a) Importance-sample the source by edge weight (same count).
                pool_weights: list[float] | None = None
                if config.min_views_weighted_source and pair_weights is not None:
                    pw = [
                        _symmetric_weight(pair_weights, facet, a, paper_b)
                        for a in paper_a_pool
                    ]
                    if any(w > 0 for w in pw):
                        pool_weights = pw
                for _ in range(deficit):
                    paper_a = (
                        rng.choices(paper_a_pool, weights=pool_weights, k=1)[0]
                        if pool_weights is not None
                        else rng.choice(paper_a_pool)
                    )
                    facets_a = facets_map.get(paper_a)
                    if facets_a is None:
                        continue
                    items = getattr(facets_a, facet, ())
                    if not items:
                        continue
                    source_idx = rng.randrange(len(items))
                    bullet_text = _render_bullet(
                        facets_a, facet, source_idx, evidence_map.get(paper_a)
                    )
                    if not bullet_text:
                        continue
                    title_a = titles.get(paper_a, "")
                    messages = _build_bullet_messages(facet, title_a, bullet_text)
                    loss_w = _pair_loss_weight(
                        paper_a,
                        paper_b,
                        facet,
                        pair_weights,
                        max_weight_for_facet or {},
                        config.graph_reweighting_max_weight,
                        enabled=config.graph_reweighting == "weighted",
                    )
                    aug_hard_negs: tuple[str, ...] = ()
                    if hard_neg_pools is not None and gold_neighbours_map is not None:
                        pool = hard_neg_pools.get((facet, paper_a)) or []
                        gold = gold_neighbours_map.get((facet, paper_a), frozenset())
                        aug_hard_negs = _sample_hard_negative_bullets(
                            paper_a=paper_a,
                            facet=facet,
                            gold_neighbours=gold,
                            pool=pool,
                            facets_map=facets_map,
                            num_neg=config.num_hard_negatives,
                            rng=rng,
                        )
                    examples.append(
                        TrainingExample(
                            prompt_messages=messages,
                            target_docid=target,
                            soft_targets=smoother.entries_for(
                                facet, paper_a, paper_b, target
                            )
                            if smoother
                            else None,
                            loss_weight=loss_w,
                            hard_negative_targets=aug_hard_negs,
                        )
                    )
                    target_counts[target] += 1
                    added += 1
                    if via_graph:
                        added_via_graph += 1
    if config.min_views_graph_fallback > 0:
        logger.info(
            "Graph-pseudo-gold fallback (top-%d neighbours): +%d of the "
            "augmentation rows came from graph sources",
            config.min_views_graph_fallback,
            added_via_graph,
        )
    logger.info(
        "min_target_views=%d augmentation: +%d rows (skipped %d under-trained "
        "bullets with no gold paper_a source)",
        config.min_target_views,
        added,
        skipped_no_source,
    )
