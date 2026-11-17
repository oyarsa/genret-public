"""Tests for connected_sampling module."""

import random
from collections import Counter
from pathlib import Path

from genret.pipeline.connected_sampling import (
    CitationGraph,
    ConnectedSampleConfig,
    OutputSplitConfig,
    SamplingReport,
    _build_removal_order,
    _count_by_year,
    _distribute_shortfall,
    _find_year_surplus,
    _parse_train_years,
    _trim_to_target,
    analyse_selection,
    compute_year_quotas,
    fill_from_fringe,
    select_from_pool,
)
from genret.pipeline.derive import SplitResult
from genret.types import CorpusId, Year

# === Test helpers ===


_id = CorpusId
_yr = Year


def _ids(*ns: int) -> set[CorpusId]:
    """Create a set of CorpusIds."""
    return {CorpusId(n) for n in ns}


def _year_map(mapping: dict[int, int]) -> dict[CorpusId, Year]:
    """Create a year map from raw ints."""
    return {CorpusId(k): Year(v) for k, v in mapping.items()}


def _refs(mapping: dict[int, list[int]]) -> dict[CorpusId, list[CorpusId]]:
    """Create a refs dict from raw ints."""
    return {CorpusId(k): [CorpusId(v) for v in vs] for k, vs in mapping.items()}


def _build_test_graph() -> tuple[CitationGraph, set[CorpusId], dict[CorpusId, Year]]:
    """Build a test graph with 12 nodes across 3 years.

    Graph structure:
        Year 2021: nodes 1-4
        Year 2022: nodes 5-8
        Year 2023: nodes 9-12

    Dense core among 1-8 (3-core = {1..8}).
    Nodes 9, 10 have degree 2; node 11 has degree 1; node 12 is isolated.

    Full-graph degrees:
        1: 5, 2: 5, 3: 5, 4: 3, 5: 6, 6: 6, 7: 4, 8: 3,
        9: 2, 10: 2, 11: 1, 12: 0
    """
    valid = _ids(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    refs = _refs(
        {
            1: [2, 3, 5, 6],
            2: [1, 3, 5, 7],
            3: [1, 2, 4, 6],
            4: [3, 5, 8],
            5: [1, 2, 4, 6, 7],
            6: [1, 3, 5, 7, 8],
            7: [2, 5, 6, 8],
            8: [4, 6, 7],
            9: [1, 5],
            10: [2, 6],
            11: [3],
        }
    )
    year_map = _year_map(
        {
            1: 2021,
            2: 2021,
            3: 2021,
            4: 2021,
            5: 2022,
            6: 2022,
            7: 2022,
            8: 2022,
            9: 2023,
            10: 2023,
            11: 2023,
            12: 2023,
        }
    )
    graph = CitationGraph.from_citations(refs, valid)
    return graph, valid, year_map


# === CitationGraph ===


class TestCitationGraph:
    """Tests for CitationGraph wrapper."""

    def test_from_citations_node_count(self) -> None:
        """Should include all valid IDs as nodes."""
        graph, valid, _ = _build_test_graph()
        assert graph.node_count() == len(valid)

    def test_from_citations_edge_count(self) -> None:
        """Should create 21 undirected edges from the test graph."""
        graph, _, _ = _build_test_graph()
        assert graph.edge_count() == 21

    def test_excludes_edges_to_invalid_nodes(self) -> None:
        """Should not create edges to nodes outside valid_ids."""
        refs = _refs({1: [2, 99], 2: [1]})
        valid = _ids(1, 2)
        graph = CitationGraph.from_citations(refs, valid)
        assert graph.edge_count() == 1
        assert graph.node_count() == 2

    def test_excludes_edges_from_invalid_nodes(self) -> None:
        """Should skip refs entries for papers not in valid_ids."""
        refs = _refs({1: [2], 99: [1, 2]})
        valid = _ids(1, 2)
        graph = CitationGraph.from_citations(refs, valid)
        assert graph.edge_count() == 1

    def test_degree(self) -> None:
        """Should return correct degree for each node."""
        graph, _, _ = _build_test_graph()
        assert graph.degree(_id(5)) == 6
        assert graph.degree(_id(12)) == 0
        assert graph.degree(_id(4)) == 3

    def test_neighbours(self) -> None:
        """Should return correct neighbour set."""
        graph, _, _ = _build_test_graph()
        assert graph.neighbours(_id(9)) == _ids(1, 5)
        assert graph.neighbours(_id(12)) == set()

    def test_k_core(self) -> None:
        """3-core should include the dense core (nodes 1-8)."""
        graph, _, _ = _build_test_graph()
        core = graph.k_core(k=3)
        assert core == _ids(1, 2, 3, 4, 5, 6, 7, 8)

    def test_k_core_lower_k(self) -> None:
        """2-core should include all nodes except 11 (degree 1) and 12 (degree 0)."""
        graph, _, _ = _build_test_graph()
        core = graph.k_core(k=2)
        assert core == _ids(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    def test_subgraph_edge_count(self) -> None:
        """Should count only edges within the subgraph."""
        graph, _, _ = _build_test_graph()
        # Subgraph {1, 2, 3}: edges 1-2, 1-3, 2-3 = 3
        assert graph.subgraph_edge_count(_ids(1, 2, 3)) == 3

    def test_subgraph_components(self) -> None:
        """Should count connected components in the subgraph."""
        graph, _, _ = _build_test_graph()
        # {1, 2, 12}: 1-2 connected, 12 isolated → 2 components
        assert graph.subgraph_components(_ids(1, 2, 12)) == 2

    def test_subgraph_components_single(self) -> None:
        """Fully connected subgraph should have 1 component."""
        graph, _, _ = _build_test_graph()
        assert graph.subgraph_components(_ids(1, 2, 3)) == 1

    def test_degree_sequence(self) -> None:
        """Should return full-graph degrees sorted ascending."""
        graph, _, _ = _build_test_graph()
        seq = graph.degree_sequence(_ids(4, 5, 12))
        assert seq == [0, 3, 6]  # 12→0, 4→3, 5→6

    def test_subgraph_degree_sequence(self) -> None:
        """Should return induced-subgraph degrees, not full-graph degrees."""
        graph, _, _ = _build_test_graph()
        # Subgraph {1, 2, 3}: each connects to the other two → all degree 2
        seq = graph.subgraph_degree_sequence(_ids(1, 2, 3))
        assert seq == [2, 2, 2]

    def test_subgraph_vs_full_degree_sequence_differs(self) -> None:
        """Subgraph degrees should differ from full-graph degrees."""
        graph, _, _ = _build_test_graph()
        nodes = _ids(1, 2, 3)
        full = graph.degree_sequence(nodes)
        induced = graph.subgraph_degree_sequence(nodes)
        assert full == [5, 5, 5]  # Full-graph degrees
        assert induced == [2, 2, 2]  # Induced-subgraph degrees


# === Quota computation ===


class TestComputeYearQuotas:
    """Tests for compute_year_quotas."""

    def test_proportional_split(self) -> None:
        """Should allocate proportionally to year counts."""
        quotas = compute_year_quotas(10, {_yr(2021): 1, _yr(2022): 2})
        assert quotas[_yr(2021)] == 3
        assert quotas[_yr(2022)] == 7

    def test_sums_to_target(self) -> None:
        """Quotas must sum to exactly target_size."""
        counts = {_yr(2021): 17, _yr(2022): 23, _yr(2023): 31, _yr(2024): 29}
        quotas = compute_year_quotas(100, counts)
        assert sum(quotas.values()) == 100

    def test_sums_to_target_prime(self) -> None:
        """Should handle a prime target that doesn't divide evenly."""
        counts = {_yr(y): c for y, c in [(2021, 3), (2022, 4), (2023, 6)]}
        quotas = compute_year_quotas(97, counts)
        assert sum(quotas.values()) == 97

    def test_empty_counts_returns_empty(self) -> None:
        """Should return empty dict for zero total."""
        assert compute_year_quotas(100, {}) == {}

    def test_single_year_gets_full_target(self) -> None:
        """Single year should receive the entire target."""
        quotas = compute_year_quotas(42, {_yr(2023): 100})
        assert quotas == {_yr(2023): 42}

    def test_even_split(self) -> None:
        """Equal counts should split evenly."""
        quotas = compute_year_quotas(10, {_yr(2021): 50, _yr(2022): 50})
        assert quotas == {_yr(2021): 5, _yr(2022): 5}


class TestCountByYear:
    """Tests for _count_by_year."""

    def test_counts_correctly(self) -> None:
        """Should count papers per year."""
        ym = _year_map({1: 2021, 2: 2021, 3: 2022})
        result = _count_by_year(_ids(1, 2, 3), ym)
        assert result == Counter({_yr(2021): 2, _yr(2022): 1})

    def test_ignores_unknown_ids(self) -> None:
        """Should skip IDs not present in year_map."""
        ym = _year_map({1: 2021})
        result = _count_by_year(_ids(1, 99), ym)
        assert result == Counter({_yr(2021): 1})

    def test_empty_ids(self) -> None:
        """Should return empty counter for empty ID set."""
        ym = _year_map({1: 2021})
        result = _count_by_year(set(), ym)
        assert result == Counter()


# === Selection ===


class TestDistributeShortfall:
    """Tests for _distribute_shortfall."""

    def test_proportional_distribution(self) -> None:
        """Should distribute proportionally to pool sizes."""
        pools: dict[Year, list[CorpusId]] = {
            _yr(2021): [_id(1), _id(2), _id(3)],
            _yr(2022): [_id(4), _id(5), _id(6)],
        }
        extras = _distribute_shortfall(4, pools, [_yr(2021), _yr(2022)])
        assert sum(extras.values()) == 4
        assert extras[_yr(2021)] == 2
        assert extras[_yr(2022)] == 2

    def test_respects_pool_capacity(self) -> None:
        """Should not allocate more than pool size per year."""
        pools: dict[Year, list[CorpusId]] = {
            _yr(2021): [_id(1)],  # Only 1 available
            _yr(2022): [_id(2), _id(3), _id(4), _id(5)],
        }
        extras = _distribute_shortfall(4, pools, [_yr(2021), _yr(2022)])
        assert extras[_yr(2021)] <= 1
        assert sum(extras.values()) == 4

    def test_empty_pools_returns_empty(self) -> None:
        """Should return empty dict when all pools are empty."""
        pools: dict[Year, list[CorpusId]] = {_yr(2021): [], _yr(2022): []}
        result = _distribute_shortfall(5, pools, [_yr(2021), _yr(2022)])
        assert result == {}

    def test_shortfall_exceeds_available(self) -> None:
        """Should allocate only what's available when shortfall exceeds capacity."""
        pools: dict[Year, list[CorpusId]] = {
            _yr(2021): [_id(1)],
            _yr(2022): [_id(2)],
        }
        extras = _distribute_shortfall(10, pools, [_yr(2021), _yr(2022)])
        assert sum(extras.values()) == 2  # Only 2 available total


class TestSelectFromPool:
    """Tests for select_from_pool."""

    def test_selects_highest_degree_papers(self) -> None:
        """Should prefer papers with higher degree."""
        graph, _, _ = _build_test_graph()
        pool_by_year: dict[Year, list[CorpusId]] = {
            _yr(2021): [_id(1), _id(2), _id(3), _id(4)],
        }
        quotas: dict[Year, int] = {_yr(2021): 2}
        rng = random.Random(42)

        selected = select_from_pool(pool_by_year, graph, quotas, rng)
        assert len(selected) == 2
        # Nodes 1, 2, 3 have degree 5; node 4 has degree 3
        # Should pick 2 of {1, 2, 3} (not 4)
        assert _id(4) not in selected

    def test_takes_all_when_pool_smaller_than_quota(self) -> None:
        """Should take all papers when pool is smaller than quota."""
        graph, _, _ = _build_test_graph()
        pool_by_year: dict[Year, list[CorpusId]] = {
            _yr(2021): [_id(1)],
        }
        quotas: dict[Year, int] = {_yr(2021): 5}
        rng = random.Random(42)

        selected = select_from_pool(pool_by_year, graph, quotas, rng)
        assert _id(1) in selected

    def test_redistributes_shortfall(self) -> None:
        """Should redistribute shortfall from under-filled to over-filled years."""
        graph, _, _ = _build_test_graph()
        pool_by_year: dict[Year, list[CorpusId]] = {
            _yr(2021): [_id(1)],  # 1 available, quota 3
            _yr(2022): [_id(5), _id(6), _id(7), _id(8)],  # 4 available, quota 3
        }
        quotas: dict[Year, int] = {_yr(2021): 3, _yr(2022): 3}
        rng = random.Random(42)

        selected = select_from_pool(pool_by_year, graph, quotas, rng)
        # 2021: takes all 1 (shortfall 2). 2022: takes top 3, 1 surplus paper left.
        # Redistribution: 1 extra from 2022's surplus → total 1 + 3 + 1 = 5.
        assert _id(1) in selected
        assert len(selected) == 5

    def test_handles_missing_year_in_pool(self) -> None:
        """Should handle years in quotas but not in pool."""
        graph, _, _ = _build_test_graph()
        pool_by_year: dict[Year, list[CorpusId]] = {
            _yr(2021): [_id(1), _id(2)],
        }
        quotas: dict[Year, int] = {_yr(2021): 2, _yr(2023): 3}
        rng = random.Random(42)

        selected = select_from_pool(pool_by_year, graph, quotas, rng)
        assert len(selected) == 2  # Only 2021 papers available

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed should produce identical results."""
        graph, _, _ = _build_test_graph()
        pool_by_year: dict[Year, list[CorpusId]] = {
            _yr(2021): [_id(1), _id(2), _id(3), _id(4)],
        }
        quotas: dict[Year, int] = {_yr(2021): 2}

        result1 = select_from_pool(pool_by_year, graph, quotas, random.Random(99))
        result2 = select_from_pool(pool_by_year, graph, quotas, random.Random(99))
        assert result1 == result2


class TestFillFromFringe:
    """Tests for fill_from_fringe."""

    def test_fills_under_quota_years(self) -> None:
        """Should add fringe papers for years that are under quota."""
        graph, valid, year_map = _build_test_graph()
        # Start with only 2021 and 2022 papers selected
        selected = _ids(1, 2, 5, 6)
        quotas: dict[Year, int] = {
            _yr(2021): 2,
            _yr(2022): 2,
            _yr(2023): 2,  # Under-filled: 0 selected
        }
        rng = random.Random(42)

        result = fill_from_fringe(selected, graph, valid, quotas, year_map, rng)
        # Should have added some 2023 papers from the fringe
        added_2023 = {
            pid for pid in result - selected if year_map.get(pid) == _yr(2023)
        }
        assert len(added_2023) > 0
        assert len(added_2023) <= 2  # At most quota deficit

    def test_does_not_modify_original(self) -> None:
        """Should return a new set, not modify the input."""
        graph, valid, year_map = _build_test_graph()
        selected = _ids(1, 2, 5, 6)
        original = set(selected)
        quotas: dict[Year, int] = {_yr(2021): 2, _yr(2022): 2, _yr(2023): 2}
        rng = random.Random(42)

        fill_from_fringe(selected, graph, valid, quotas, year_map, rng)
        assert selected == original

    def test_returns_copy_when_all_filled(self) -> None:
        """Should return a copy of selected when no years are under-filled."""
        graph, valid, year_map = _build_test_graph()
        selected = _ids(1, 2, 3, 4, 5, 6, 7, 8)
        quotas: dict[Year, int] = {_yr(2021): 4, _yr(2022): 4}
        rng = random.Random(42)

        result = fill_from_fringe(selected, graph, valid, quotas, year_map, rng)
        assert result == selected
        assert result is not selected  # Must be a copy

    def test_only_adds_valid_fringe_papers(self) -> None:
        """Fringe papers must be in valid_ids."""
        refs = _refs({1: [2, 99], 2: [1]})
        valid = _ids(1, 2)
        graph = CitationGraph.from_citations(refs, valid)
        year_map = _year_map({1: 2021, 2: 2022, 99: 2022})
        quotas: dict[Year, int] = {_yr(2021): 1, _yr(2022): 2}
        rng = random.Random(42)

        result = fill_from_fringe(_ids(1), graph, valid, quotas, year_map, rng)
        # Node 99 is not valid, so only node 2 can be added
        assert _id(99) not in result

    def test_prefers_most_connected_fringe(self) -> None:
        """Should rank fringe papers by connections to the selected set."""
        # Custom graph: selected {1,2,3} in 2021; fringe 10 (→1,2,3) vs 11 (→1 only)
        valid = _ids(1, 2, 3, 10, 11)
        refs = _refs(
            {1: [2, 3, 10, 11], 2: [1, 3, 10], 3: [1, 2, 10], 10: [1, 2, 3], 11: [1]}
        )
        graph = CitationGraph.from_citations(refs, valid)
        ym = _year_map({1: 2021, 2: 2021, 3: 2021, 10: 2023, 11: 2023})

        selected = _ids(1, 2, 3)
        quotas: dict[Year, int] = {_yr(2021): 3, _yr(2023): 1}
        rng = random.Random(42)

        result = fill_from_fringe(selected, graph, valid, quotas, ym, rng)
        # Node 10 has 3 connections to selected, node 11 has 1 → must pick 10
        assert _id(10) in result
        assert _id(11) not in result


# === Trimming ===


class TestFindYearSurplus:
    """Tests for _find_year_surplus."""

    def test_finds_over_quota_years(self) -> None:
        """Should identify years above their quota."""
        counts: Counter[Year] = Counter({_yr(2021): 5, _yr(2022): 3})
        quotas: dict[Year, int] = {_yr(2021): 3, _yr(2022): 3}
        surplus = _find_year_surplus(counts, quotas)
        assert surplus == {_yr(2021): 2}

    def test_ignores_at_or_under_quota(self) -> None:
        """Should not include years at or below quota."""
        counts: Counter[Year] = Counter({_yr(2021): 3, _yr(2022): 2})
        quotas: dict[Year, int] = {_yr(2021): 3, _yr(2022): 5}
        surplus = _find_year_surplus(counts, quotas)
        assert surplus == {}

    def test_year_not_in_quotas(self) -> None:
        """A year with count but no quota has surplus equal to its count."""
        counts: Counter[Year] = Counter({_yr(2021): 3})
        quotas: dict[Year, int] = {}
        surplus = _find_year_surplus(counts, quotas)
        assert surplus == {_yr(2021): 3}


class TestBuildRemovalOrder:
    """Tests for _build_removal_order."""

    def test_sorted_by_degree_ascending(self) -> None:
        """Papers should be sorted lowest degree first for removal."""
        graph, _, year_map = _build_test_graph()
        selected = _ids(1, 2, 4)  # degrees 5, 5, 3
        surplus = {_yr(2021): 2}
        rng = random.Random(42)

        order = _build_removal_order(selected, surplus, graph, year_map, rng)
        # Node 4 (degree 3) should come first
        assert order[_yr(2021)][0] == _id(4)

    def test_only_includes_surplus_years(self) -> None:
        """Should only build lists for years with surplus."""
        graph, _, year_map = _build_test_graph()
        selected = _ids(1, 5)
        surplus = {_yr(2021): 1}  # Only 2021
        rng = random.Random(42)

        order = _build_removal_order(selected, surplus, graph, year_map, rng)
        assert _yr(2021) in order
        assert _yr(2022) not in order


class TestTrimToTarget:
    """Tests for _trim_to_target."""

    def test_trims_from_over_quota_years(self) -> None:
        """Should only remove papers from years above their quota."""
        graph, _, year_map = _build_test_graph()
        # 2021: 4 papers (1-4), quota 2 → surplus 2
        # 2022: 4 papers (5-8), quota 4 → at quota
        selected = _ids(1, 2, 3, 4, 5, 6, 7, 8)
        quotas: dict[Year, int] = {_yr(2021): 2, _yr(2022): 4}
        rng = random.Random(42)

        result = _trim_to_target(selected, graph, 6, quotas, year_map, rng)
        assert len(result) == 6
        # All 2022 papers should remain (at quota)
        assert _ids(5, 6, 7, 8) <= result
        # Only 2 of 4 2021 papers should remain
        remaining_2021 = result & _ids(1, 2, 3, 4)
        assert len(remaining_2021) == 2

    def test_returns_copy_when_at_target(self) -> None:
        """Should return a copy when already at target size."""
        graph, _, year_map = _build_test_graph()
        selected = _ids(1, 2)
        quotas: dict[Year, int] = {_yr(2021): 2}
        rng = random.Random(42)

        result = _trim_to_target(selected, graph, 2, quotas, year_map, rng)
        assert result == selected
        assert result is not selected

    def test_returns_copy_when_below_target(self) -> None:
        """Should return a copy when below target size."""
        graph, _, year_map = _build_test_graph()
        selected = _ids(1, 2)
        quotas: dict[Year, int] = {_yr(2021): 2}
        rng = random.Random(42)

        result = _trim_to_target(selected, graph, 10, quotas, year_map, rng)
        assert result == selected
        assert result is not selected

    def test_stops_when_no_surplus_remains(self) -> None:
        """Should stop early if all years are at or below quota."""
        graph, _, year_map = _build_test_graph()
        # All years at quota — no surplus to trim
        selected = _ids(1, 2, 5, 6)
        quotas: dict[Year, int] = {_yr(2021): 2, _yr(2022): 2}
        rng = random.Random(42)

        result = _trim_to_target(selected, graph, 2, quotas, year_map, rng)
        # Can't trim: target is 2 but no over-quota papers
        assert result == selected

    def test_removes_lowest_degree_first(self) -> None:
        """Should remove lowest-degree papers first within a year."""
        graph, _, year_map = _build_test_graph()
        # 2021 nodes: 1(deg5), 2(deg5), 3(deg5), 4(deg3)
        selected = _ids(1, 2, 3, 4)
        quotas: dict[Year, int] = {_yr(2021): 2}
        rng = random.Random(42)

        result = _trim_to_target(selected, graph, 3, quotas, year_map, rng)
        # Node 4 (degree 3) should be removed first
        assert _id(4) not in result
        assert len(result) == 3

    def test_does_not_modify_original(self) -> None:
        """Should return a new set, not modify the input."""
        graph, _, year_map = _build_test_graph()
        selected = _ids(1, 2, 3, 4)
        original = set(selected)
        quotas: dict[Year, int] = {_yr(2021): 2}
        rng = random.Random(42)

        _trim_to_target(selected, graph, 2, quotas, year_map, rng)
        assert selected == original


# === Analysis ===


class TestAnalyseSelection:
    """Tests for analyse_selection."""

    def test_correct_total(self) -> None:
        """Should report correct total count."""
        graph, _, year_map = _build_test_graph()
        selected = _ids(1, 2, 3)
        quotas: dict[Year, int] = {_yr(2021): 3}
        report = analyse_selection(graph, selected, year_map, quotas)
        assert report.total == 3

    def test_induced_edge_count(self) -> None:
        """Should count edges in the induced subgraph."""
        graph, _, year_map = _build_test_graph()
        # {1, 2, 3} form a triangle → 3 edges
        selected = _ids(1, 2, 3)
        quotas: dict[Year, int] = {_yr(2021): 3}
        report = analyse_selection(graph, selected, year_map, quotas)
        assert report.edges == 3

    def test_induced_degree_stats(self) -> None:
        """Should compute degree stats from the induced subgraph."""
        graph, _, year_map = _build_test_graph()
        # {1, 2, 3} each have induced degree 2 (triangle)
        selected = _ids(1, 2, 3)
        quotas: dict[Year, int] = {_yr(2021): 3}
        report = analyse_selection(graph, selected, year_map, quotas)
        assert report.min_degree == 2
        assert report.max_degree == 2
        assert report.mean_degree == 2.0

    def test_components(self) -> None:
        """Should count connected components correctly."""
        graph, _, year_map = _build_test_graph()
        # {1, 2, 12}: 1-2 connected, 12 isolated → 2 components
        selected = _ids(1, 2, 12)
        quotas: dict[Year, int] = {_yr(2021): 2, _yr(2023): 1}
        report = analyse_selection(graph, selected, year_map, quotas)
        assert report.components == 2

    def test_year_distribution(self) -> None:
        """Should report correct year distribution."""
        graph, _, year_map = _build_test_graph()
        selected = _ids(1, 2, 5)
        quotas: dict[Year, int] = {_yr(2021): 2, _yr(2022): 1}
        report = analyse_selection(graph, selected, year_map, quotas)
        assert report.year_distribution == {_yr(2021): 2, _yr(2022): 1}

    def test_zero_deviation_when_matching_quotas(self) -> None:
        """Deviation should be 0 when distribution matches quotas exactly."""
        graph, _, year_map = _build_test_graph()
        selected = _ids(1, 2, 5, 6)
        quotas: dict[Year, int] = {_yr(2021): 2, _yr(2022): 2}
        report = analyse_selection(graph, selected, year_map, quotas)
        for dev in report.year_deviation.values():
            assert dev == 0.0

    def test_empty_selection(self) -> None:
        """Should handle empty selection gracefully."""
        graph, _, year_map = _build_test_graph()
        quotas: dict[Year, int] = {_yr(2021): 3}
        report = analyse_selection(graph, set(), year_map, quotas)
        assert report.total == 0
        assert report.edges == 0
        assert report.min_degree == 0
        assert report.max_degree == 0


# === Config parsing ===


class TestParseTrainYears:
    """Tests for _parse_train_years."""

    def test_parses_year_range(self) -> None:
        """Should parse '2019-2022' into (2019, 2022)."""
        assert _parse_train_years("2019-2022") == (2019, 2022)

    def test_parses_all(self) -> None:
        """Should parse 'all' into (0, 0)."""
        assert _parse_train_years("all") == (0, 0)

    def test_malformed_falls_back(self) -> None:
        """Should return (0, 0) for malformed input."""
        assert _parse_train_years("garbage") == (0, 0)
        assert _parse_train_years("") == (0, 0)
        assert _parse_train_years("2019-2020-2021") == (0, 0)


# === Pydantic models ===


class TestOutputSplitConfig:
    """Tests for OutputSplitConfig and its from_results factory."""

    def test_from_results_round_trip(self) -> None:
        """Factory output should serialise and deserialise correctly."""
        config = ConnectedSampleConfig(
            data_dir=Path("/data"),
            splits_dir=Path("/splits"),
            output_dir=Path("/output"),
            target_size=100,
            min_degree=3,
            seed=42,
        )
        report = SamplingReport(
            total=100,
            edges=500,
            min_degree=2,
            mean_degree=10.0,
            max_degree=50,
            components=1,
            year_distribution={_yr(2021): 50, _yr(2022): 50},
            target_distribution={_yr(2021): 50, _yr(2022): 50},
            year_deviation={_yr(2021): 0.0, _yr(2022): 0.0},
        )
        splits = SplitResult(
            train=[CorpusId(1), CorpusId(2), CorpusId(3)],
            dev=[CorpusId(4), CorpusId(5)],
            test=[CorpusId(6), CorpusId(7)],
        )

        output = OutputSplitConfig.from_results(
            config,
            report,
            splits,
            universe_size=1000,
            train_years=(2019, 2022),
            dev_year=2023,
            test_year=2024,
        )

        assert output.method == "connected_sample"
        assert output.parameters.target_size == 100
        assert output.parameters.min_degree == 3
        assert output.parameters.seed == 42
        assert output.parameters.source_splits == "/splits"
        assert output.train_years == "2019-2022"
        assert output.dev_year == 2023
        assert output.test_year == 2024
        assert output.counts.universe == 1000
        assert output.counts.total == 100
        assert output.counts.train == 3
        assert output.counts.dev == 2
        assert output.counts.test == 2
        assert output.graph_statistics.edges == 500
        assert output.graph_statistics.connected_components == 1

        # Round-trip through JSON
        json_str = output.model_dump_json(indent=2)
        restored = OutputSplitConfig.model_validate_json(json_str)
        assert restored == output

    def test_from_results_all_train_years(self) -> None:
        """Should format train_years as 'all' when (0, 0)."""
        config = ConnectedSampleConfig(
            data_dir=Path("/data"),
            splits_dir=Path("/splits"),
            output_dir=Path("/output"),
            target_size=10,
            min_degree=1,
            seed=0,
        )
        report = SamplingReport(
            total=10,
            edges=5,
            min_degree=1,
            mean_degree=1.0,
            max_degree=2,
            components=1,
            year_distribution={_yr(2021): 10},
            target_distribution={_yr(2021): 10},
            year_deviation={_yr(2021): 0.0},
        )
        splits = SplitResult(train=[CorpusId(1)], dev=[CorpusId(2)], test=[CorpusId(3)])

        output = OutputSplitConfig.from_results(
            config,
            report,
            splits,
            universe_size=100,
            train_years=(0, 0),
            dev_year=2023,
            test_year=2024,
        )
        assert output.train_years == "all"


# === Integration-style: full pipeline on test graph ===


class TestEndToEnd:
    """Integration tests combining multiple steps on the test graph."""

    def test_pool_select_fill_trim_produces_target_size(self) -> None:
        """Full select + fill + trim should produce exactly target size."""
        graph, valid, year_map = _build_test_graph()
        target = 8
        full_year_counts = _count_by_year(valid, year_map)
        quotas = compute_year_quotas(target, full_year_counts)
        assert sum(quotas.values()) == target

        # k-core(2) to get a larger pool
        pool = graph.k_core(k=2)
        pool_by_year: dict[Year, list[CorpusId]] = {}
        for pid in pool:
            yr = year_map.get(pid)
            if yr is not None:
                pool_by_year.setdefault(yr, []).append(pid)

        rng = random.Random(42)
        selected = select_from_pool(pool_by_year, graph, quotas, rng)
        selected = fill_from_fringe(selected, graph, valid, quotas, year_map, rng)
        selected = _trim_to_target(selected, graph, target, quotas, year_map, rng)

        assert len(selected) == target

    def test_analyse_after_selection(self) -> None:
        """Analysis report should be consistent with the selection."""
        graph, _, year_map = _build_test_graph()
        selected = _ids(1, 2, 3, 5, 6, 7)
        quotas: dict[Year, int] = {_yr(2021): 3, _yr(2022): 3}

        report = analyse_selection(graph, selected, year_map, quotas)
        assert report.total == 6
        assert report.components == 1  # All connected
        assert report.edges > 0
        assert report.min_degree <= report.max_degree
        # Year distribution matches quotas exactly
        assert report.year_distribution == {_yr(2021): 3, _yr(2022): 3}
        for dev in report.year_deviation.values():
            assert dev == 0.0

    def test_deterministic_full_pipeline(self) -> None:
        """Full pipeline (select + fill + trim) should be deterministic with same seed."""
        graph, valid, year_map = _build_test_graph()
        full_year_counts = _count_by_year(valid, year_map)
        target = 8

        def _run(seed: int) -> set[CorpusId]:
            """Run selection pipeline with a given seed."""
            quotas = compute_year_quotas(target, full_year_counts)
            pool = graph.k_core(k=2)
            pool_by_year: dict[Year, list[CorpusId]] = {}
            for pid in pool:
                yr = year_map.get(pid)
                if yr is not None:
                    pool_by_year.setdefault(yr, []).append(pid)
            rng = random.Random(seed)
            selected = select_from_pool(pool_by_year, graph, quotas, rng)
            selected = fill_from_fringe(selected, graph, valid, quotas, year_map, rng)
            return _trim_to_target(selected, graph, target, quotas, year_map, rng)

        assert _run(42) == _run(42)
