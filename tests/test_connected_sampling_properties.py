"""Property-based tests for connected_sampling module.

Uses seeded random-fuzz loops to verify invariants across many random inputs,
catching edge-case regressions that fixture tests miss.
"""

import random
from collections import defaultdict
from dataclasses import dataclass

from genret.pipeline.connected_sampling import (
    CitationGraph,
    _count_by_year,
    _trim_to_target,
    compute_year_quotas,
    fill_from_fringe,
    select_from_pool,
)
from genret.types import CorpusId, Year

NUM_TRIALS = 50
"""Number of random scenarios per property test."""


# === Random scenario generator ===


@dataclass(frozen=True, kw_only=True)
class Scenario:
    """A randomly generated graph scenario for property testing."""

    graph: CitationGraph
    """Citation graph built from refs and valid_ids."""
    valid_ids: set[CorpusId]
    """All valid paper IDs (the universe)."""
    year_map: dict[CorpusId, Year]
    """Map of paper_id -> publication year."""
    years: list[Year]
    """Distinct years present in the scenario."""


def _generate_scenario(rng: random.Random) -> Scenario:
    """Generate a random graph scenario.

    Produces 10-80 nodes across 2-5 years with random edges at 5-30% density.

    Args:
        rng: Seeded random number generator.

    Returns:
        A complete scenario with graph, IDs, and year assignments.
    """
    n_nodes = rng.randint(10, 80)
    n_years = rng.randint(2, 5)
    years = [Year(y) for y in range(2019, 2019 + n_years)]

    ids = [CorpusId(i) for i in range(1, n_nodes + 1)]
    valid_ids = set(ids)
    year_map = {pid: rng.choice(years) for pid in ids}

    # Random directed edges (undirected graph built from them)
    edge_prob = rng.uniform(0.05, 0.3)
    refs: dict[CorpusId, list[CorpusId]] = {}
    for pid in ids:
        cited = [other for other in ids if other != pid and rng.random() < edge_prob]
        if cited:
            refs[pid] = cited

    graph = CitationGraph.from_citations(refs, valid_ids)
    return Scenario(
        graph=graph,
        valid_ids=valid_ids,
        year_map=year_map,
        years=years,
    )


def _make_pool_by_year(
    pool: set[CorpusId], year_map: dict[CorpusId, Year]
) -> dict[Year, list[CorpusId]]:
    """Group a pool of IDs by year.

    Args:
        pool: Set of paper IDs.
        year_map: Map of paper_id -> publication year.

    Returns:
        Map of year -> list of paper IDs.
    """
    by_year: defaultdict[Year, list[CorpusId]] = defaultdict(list)
    for pid in pool:
        yr = year_map.get(pid)
        if yr is not None:
            by_year[yr].append(pid)
    return dict(by_year)


def _run_pipeline(scenario_seed: int, pipeline_seed: int) -> set[CorpusId]:
    """Run the full select -> fill -> trim pipeline on a random scenario.

    Args:
        scenario_seed: Seed for generating the random graph scenario.
        pipeline_seed: Seed for the selection RNG.

    Returns:
        Final selected paper IDs.
    """
    scenario = _generate_scenario(random.Random(scenario_seed))
    pool = scenario.graph.k_core(k=1)
    if not pool:
        return set()

    pool_by_year = _make_pool_by_year(pool, scenario.year_map)
    year_counts = _count_by_year(scenario.valid_ids, scenario.year_map)
    target = max(1, len(pool) // 2)
    quotas = compute_year_quotas(target, year_counts)

    rng = random.Random(pipeline_seed)
    selected = select_from_pool(pool_by_year, scenario.graph, quotas, rng)
    selected = fill_from_fringe(
        selected,
        scenario.graph,
        scenario.valid_ids,
        quotas,
        scenario.year_map,
        rng,
    )
    return _trim_to_target(
        selected, scenario.graph, target, quotas, scenario.year_map, rng
    )


# === Property tests ===


class TestComputeYearQuotasProperties:
    """Property tests for compute_year_quotas."""

    def test_quotas_sum_to_target(self) -> None:
        """Quotas must sum to exactly target_size for any input."""
        rng = random.Random(1001)
        for _ in range(NUM_TRIALS):
            n_years = rng.randint(1, 6)
            year_counts = {Year(2019 + i): rng.randint(1, 500) for i in range(n_years)}
            target = rng.randint(1, 1000)

            quotas = compute_year_quotas(target, year_counts)

            assert sum(quotas.values()) == target, (
                f"target={target}, year_counts={year_counts}, quotas={quotas}"
            )

    def test_quotas_non_negative(self) -> None:
        """All quotas must be >= 0."""
        rng = random.Random(1002)
        for _ in range(NUM_TRIALS):
            n_years = rng.randint(1, 6)
            year_counts = {Year(2019 + i): rng.randint(0, 100) for i in range(n_years)}
            target = rng.randint(0, 200)

            quotas = compute_year_quotas(target, year_counts)

            for year, quota in quotas.items():
                assert quota >= 0, f"year={year}, quota={quota}"

    def test_no_quota_exceeds_target(self) -> None:
        """No single year's quota should exceed the total target."""
        rng = random.Random(1003)
        for _ in range(NUM_TRIALS):
            n_years = rng.randint(1, 6)
            year_counts = {Year(2019 + i): rng.randint(1, 100) for i in range(n_years)}
            target = rng.randint(1, 200)

            quotas = compute_year_quotas(target, year_counts)

            for year, quota in quotas.items():
                assert quota <= target, f"year={year}, quota={quota}, target={target}"


class TestSelectFromPoolProperties:
    """Property tests for select_from_pool."""

    def test_output_subset_of_pool(self) -> None:
        """Selected papers must be a subset of the pool."""
        rng = random.Random(2001)
        for _ in range(NUM_TRIALS):
            scenario = _generate_scenario(rng)
            pool = scenario.graph.k_core(k=1)
            if not pool:
                continue

            pool_by_year = _make_pool_by_year(pool, scenario.year_map)
            all_pool_ids = {pid for ids in pool_by_year.values() for pid in ids}
            year_counts = _count_by_year(scenario.valid_ids, scenario.year_map)
            target = max(1, len(pool) // 2)
            quotas = compute_year_quotas(target, year_counts)

            selected = select_from_pool(pool_by_year, scenario.graph, quotas, rng)

            assert selected <= all_pool_ids, (
                f"Selected IDs outside pool: {selected - all_pool_ids}"
            )

    def test_deterministic(self) -> None:
        """Same seed and input must produce identical output."""
        master_rng = random.Random(2002)
        for _ in range(NUM_TRIALS):
            seed = master_rng.randint(0, 10**6)
            scenario = _generate_scenario(random.Random(seed))
            pool = scenario.graph.k_core(k=1)
            if not pool:
                continue

            pool_by_year = _make_pool_by_year(pool, scenario.year_map)
            year_counts = _count_by_year(scenario.valid_ids, scenario.year_map)
            target = max(1, len(pool) // 2)
            quotas = compute_year_quotas(target, year_counts)

            result_a = select_from_pool(
                pool_by_year, scenario.graph, quotas, random.Random(42)
            )
            result_b = select_from_pool(
                pool_by_year, scenario.graph, quotas, random.Random(42)
            )
            assert result_a == result_b


class TestFillFromFringeProperties:
    """Property tests for fill_from_fringe."""

    def test_superset_of_input(self) -> None:
        """Output must be a superset of the input selected set."""
        rng = random.Random(3001)
        for _ in range(NUM_TRIALS):
            scenario = _generate_scenario(rng)
            pool = scenario.graph.k_core(k=1)
            if not pool:
                continue

            pool_by_year = _make_pool_by_year(pool, scenario.year_map)
            year_counts = _count_by_year(scenario.valid_ids, scenario.year_map)
            target = max(1, len(pool) // 2)
            quotas = compute_year_quotas(target, year_counts)

            selected = select_from_pool(pool_by_year, scenario.graph, quotas, rng)
            filled = fill_from_fringe(
                selected,
                scenario.graph,
                scenario.valid_ids,
                quotas,
                scenario.year_map,
                rng,
            )

            assert filled >= selected, f"Lost IDs: {selected - filled}"

    def test_new_ids_in_valid_set(self) -> None:
        """Newly added IDs must be in valid_ids."""
        rng = random.Random(3002)
        for _ in range(NUM_TRIALS):
            scenario = _generate_scenario(rng)
            pool = scenario.graph.k_core(k=1)
            if not pool:
                continue

            pool_by_year = _make_pool_by_year(pool, scenario.year_map)
            year_counts = _count_by_year(scenario.valid_ids, scenario.year_map)
            target = max(1, len(pool) // 2)
            quotas = compute_year_quotas(target, year_counts)

            selected = select_from_pool(pool_by_year, scenario.graph, quotas, rng)
            filled = fill_from_fringe(
                selected,
                scenario.graph,
                scenario.valid_ids,
                quotas,
                scenario.year_map,
                rng,
            )
            added = filled - selected

            assert added <= scenario.valid_ids, (
                f"Added IDs outside universe: {added - scenario.valid_ids}"
            )

    def test_new_ids_are_neighbours(self) -> None:
        """Every newly added ID must be a 1-hop neighbour of the original selection."""
        rng = random.Random(3003)
        for _ in range(NUM_TRIALS):
            scenario = _generate_scenario(rng)
            pool = scenario.graph.k_core(k=1)
            if not pool:
                continue

            pool_by_year = _make_pool_by_year(pool, scenario.year_map)
            year_counts = _count_by_year(scenario.valid_ids, scenario.year_map)
            target = max(1, len(pool) // 2)
            quotas = compute_year_quotas(target, year_counts)

            selected = select_from_pool(pool_by_year, scenario.graph, quotas, rng)
            filled = fill_from_fringe(
                selected,
                scenario.graph,
                scenario.valid_ids,
                quotas,
                scenario.year_map,
                rng,
            )
            added = filled - selected

            # Build full 1-hop fringe of original selection
            fringe: set[CorpusId] = set()
            for pid in selected:
                fringe |= scenario.graph.neighbours(pid)
            fringe -= selected

            assert added <= fringe, f"Added IDs not in 1-hop fringe: {added - fringe}"

    def test_deterministic(self) -> None:
        """Same seed and input must produce identical output."""
        master_rng = random.Random(3004)
        for _ in range(NUM_TRIALS):
            seed = master_rng.randint(0, 10**6)
            scenario = _generate_scenario(random.Random(seed))
            pool = scenario.graph.k_core(k=1)
            if not pool:
                continue

            pool_by_year = _make_pool_by_year(pool, scenario.year_map)
            year_counts = _count_by_year(scenario.valid_ids, scenario.year_map)
            target = max(1, len(pool) // 2)
            quotas = compute_year_quotas(target, year_counts)

            selected = select_from_pool(
                pool_by_year, scenario.graph, quotas, random.Random(seed)
            )

            filled_a = fill_from_fringe(
                selected,
                scenario.graph,
                scenario.valid_ids,
                quotas,
                scenario.year_map,
                random.Random(seed + 1),
            )
            filled_b = fill_from_fringe(
                selected,
                scenario.graph,
                scenario.valid_ids,
                quotas,
                scenario.year_map,
                random.Random(seed + 1),
            )
            assert filled_a == filled_b


class TestTrimToTargetProperties:
    """Property tests for _trim_to_target."""

    def test_output_subset_of_input(self) -> None:
        """Output must be a subset of the input."""
        rng = random.Random(4001)
        for _ in range(NUM_TRIALS):
            scenario = _generate_scenario(rng)
            pool = scenario.graph.k_core(k=1)
            if not pool:
                continue

            pool_by_year = _make_pool_by_year(pool, scenario.year_map)
            year_counts = _count_by_year(scenario.valid_ids, scenario.year_map)
            target = max(1, len(pool) // 3)
            quotas = compute_year_quotas(target, year_counts)

            selected = select_from_pool(pool_by_year, scenario.graph, quotas, rng)
            selected = fill_from_fringe(
                selected,
                scenario.graph,
                scenario.valid_ids,
                quotas,
                scenario.year_map,
                rng,
            )
            trimmed = _trim_to_target(
                selected, scenario.graph, target, quotas, scenario.year_map, rng
            )

            assert trimmed <= selected, (
                f"Trimmed set contains new IDs: {trimmed - selected}"
            )

    def test_never_trims_below_quota_years(self) -> None:
        """Papers in years at or below quota must never be removed."""
        rng = random.Random(4002)
        for _ in range(NUM_TRIALS):
            scenario = _generate_scenario(rng)
            pool = scenario.graph.k_core(k=1)
            if not pool:
                continue

            pool_by_year = _make_pool_by_year(pool, scenario.year_map)
            year_counts = _count_by_year(scenario.valid_ids, scenario.year_map)
            target = max(1, len(pool) // 3)
            quotas = compute_year_quotas(target, year_counts)

            selected = select_from_pool(pool_by_year, scenario.graph, quotas, rng)
            selected = fill_from_fringe(
                selected,
                scenario.graph,
                scenario.valid_ids,
                quotas,
                scenario.year_map,
                rng,
            )

            before_counts = _count_by_year(selected, scenario.year_map)
            trimmed = _trim_to_target(
                selected, scenario.graph, target, quotas, scenario.year_map, rng
            )
            after_counts = _count_by_year(trimmed, scenario.year_map)

            for year, quota in quotas.items():
                if before_counts.get(year, 0) <= quota:
                    assert after_counts.get(year, 0) == before_counts.get(year, 0), (
                        f"year={year}, quota={quota}, "
                        f"before={before_counts.get(year, 0)}, "
                        f"after={after_counts.get(year, 0)}"
                    )

    def test_reaches_target_when_surplus_exists(self) -> None:
        """When total surplus >= excess, trimmed size must equal target."""
        rng = random.Random(4003)
        for _ in range(NUM_TRIALS):
            scenario = _generate_scenario(rng)
            pool = scenario.graph.k_core(k=1)
            if not pool:
                continue

            pool_by_year = _make_pool_by_year(pool, scenario.year_map)
            year_counts = _count_by_year(scenario.valid_ids, scenario.year_map)
            target = max(1, len(pool) // 3)
            quotas = compute_year_quotas(target, year_counts)

            selected = select_from_pool(pool_by_year, scenario.graph, quotas, rng)
            selected = fill_from_fringe(
                selected,
                scenario.graph,
                scenario.valid_ids,
                quotas,
                scenario.year_map,
                rng,
            )

            if len(selected) <= target:
                continue

            excess = len(selected) - target
            before_counts = _count_by_year(selected, scenario.year_map)
            total_surplus = sum(
                max(0, before_counts.get(y, 0) - quotas.get(y, 0))
                for y in set(before_counts) | set(quotas)
            )

            trimmed = _trim_to_target(
                selected, scenario.graph, target, quotas, scenario.year_map, rng
            )

            if total_surplus >= excess:
                assert len(trimmed) == target, (
                    f"surplus={total_surplus}, excess={excess}, "
                    f"before={len(selected)}, after={len(trimmed)}, target={target}"
                )


class TestEndToEndProperties:
    """Property tests for the full select -> fill -> trim pipeline."""

    def test_output_within_universe(self) -> None:
        """Final selection must be a subset of valid_ids."""
        rng = random.Random(5001)
        for _ in range(NUM_TRIALS):
            scenario = _generate_scenario(rng)
            pool = scenario.graph.k_core(k=1)
            if not pool:
                continue

            pool_by_year = _make_pool_by_year(pool, scenario.year_map)
            year_counts = _count_by_year(scenario.valid_ids, scenario.year_map)
            target = max(1, len(pool) // 2)
            quotas = compute_year_quotas(target, year_counts)

            selected = select_from_pool(pool_by_year, scenario.graph, quotas, rng)
            selected = fill_from_fringe(
                selected,
                scenario.graph,
                scenario.valid_ids,
                quotas,
                scenario.year_map,
                rng,
            )
            selected = _trim_to_target(
                selected, scenario.graph, target, quotas, scenario.year_map, rng
            )

            assert selected <= scenario.valid_ids

    def test_deterministic(self) -> None:
        """Full pipeline must be deterministic with fixed seed."""
        master_rng = random.Random(5002)
        for _ in range(NUM_TRIALS):
            scenario_seed = master_rng.randint(0, 10**6)
            pipeline_seed = master_rng.randint(0, 10**6)

            result_a = _run_pipeline(scenario_seed, pipeline_seed)
            result_b = _run_pipeline(scenario_seed, pipeline_seed)
            assert result_a == result_b

    def test_year_quotas_preserved_after_trim(self) -> None:
        """Years at or below quota before trim must remain unchanged after trim."""
        rng = random.Random(5003)
        for _ in range(NUM_TRIALS):
            scenario = _generate_scenario(rng)
            pool = scenario.graph.k_core(k=1)
            if not pool:
                continue

            pool_by_year = _make_pool_by_year(pool, scenario.year_map)
            year_counts = _count_by_year(scenario.valid_ids, scenario.year_map)
            target = max(1, len(pool) // 3)
            quotas = compute_year_quotas(target, year_counts)

            selected = select_from_pool(pool_by_year, scenario.graph, quotas, rng)
            filled = fill_from_fringe(
                selected,
                scenario.graph,
                scenario.valid_ids,
                quotas,
                scenario.year_map,
                rng,
            )

            before_counts = _count_by_year(filled, scenario.year_map)
            trimmed = _trim_to_target(
                filled, scenario.graph, target, quotas, scenario.year_map, rng
            )
            after_counts = _count_by_year(trimmed, scenario.year_map)

            for year, quota in quotas.items():
                if before_counts.get(year, 0) <= quota:
                    assert after_counts.get(year, 0) == before_counts.get(year, 0), (
                        f"year={year}, quota={quota}, "
                        f"before={before_counts.get(year, 0)}, "
                        f"after={after_counts.get(year, 0)}"
                    )
