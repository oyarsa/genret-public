"""Connected sampling for dense paper subsets.

Hybrid sampling policy: take the majority from a k-core for citation density, then
correct year distribution by filling under-represented years from the 1-hop fringe
of well-connected papers. High connectivity is a design goal but not a hard
constraint — the k-core + fringe approach makes it very likely empirically.
"""

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from pydantic import BaseModel

from genret._shed.networkx import k_core as nx_k_core
from genret._shed.networkx import number_connected_components
from genret.pipeline.derive import (
    SplitResult,
    load_metadata_years,
    split_papers_by_year,
)
from genret.types import CorpusId, Year
from genret.utils import die, load_references, load_split_ids, resolve_jsonl, safediv

logger = logging.getLogger(__name__)


# === Graph ===


class CitationGraph:
    """Typed wrapper around a networkx undirected graph of paper IDs.

    Encapsulates all networkx operations so the rest of the module stays
    fully typed despite networkx lacking type stubs.
    """

    def __init__(self) -> None:
        """Initialise an empty graph."""
        self._g: nx.Graph[CorpusId] = nx.Graph()

    @classmethod
    def from_citations(
        cls, refs: dict[CorpusId, list[CorpusId]], valid_ids: set[CorpusId]
    ) -> CitationGraph:
        """Build an undirected citation graph with only in-corpus edges.

        Args:
            refs: Map of paper_id -> list of cited paper IDs.
            valid_ids: Set of corpus IDs to include as nodes.

        Returns:
            Graph with edges between valid papers.
        """
        graph = cls()
        graph._g.add_nodes_from(valid_ids)

        for paper_id, cited_ids in refs.items():
            if paper_id not in valid_ids:
                continue

            for cited_id in cited_ids:
                if cited_id in valid_ids:
                    graph._g.add_edge(paper_id, cited_id)

        return graph

    def node_count(self) -> int:
        """Return the number of nodes."""
        return self._g.number_of_nodes()

    def edge_count(self) -> int:
        """Return the number of edges."""
        return self._g.number_of_edges()

    def degree(self, node: CorpusId) -> int:
        """Get the degree of a single node."""
        return self._g.degree(node)

    def neighbours(self, node: CorpusId) -> set[CorpusId]:
        """Get the neighbours of a node."""
        return set(self._g.neighbors(node))

    def k_core(self, k: int) -> set[CorpusId]:
        """Compute the k-core and return the node set.

        Args:
            k: Minimum degree threshold.

        Returns:
            Set of node IDs in the k-core.
        """
        return nx_k_core(self._g, k=k)

    def subgraph_edge_count(self, nodes: set[CorpusId]) -> int:
        """Count edges in the subgraph induced by a node set."""
        return self._g.subgraph(nodes).number_of_edges()

    def subgraph_components(self, nodes: set[CorpusId]) -> int:
        """Count connected components in the subgraph induced by a node set."""
        return number_connected_components(self._g.subgraph(nodes))

    def degree_sequence(self, nodes: set[CorpusId]) -> list[int]:
        """Get the sorted full-graph degree sequence for a subset of nodes.

        Args:
            nodes: Subset of node IDs.

        Returns:
            Sorted list of full-graph degrees (ascending).
        """
        return sorted(self.degree(n) for n in nodes)

    def subgraph_degree_sequence(self, nodes: set[CorpusId]) -> list[int]:
        """Get the sorted induced-subgraph degree sequence for a subset of nodes.

        Unlike ``degree_sequence``, this counts only edges where both endpoints
        are in ``nodes``.

        Args:
            nodes: Subset of node IDs.

        Returns:
            Sorted list of induced-subgraph degrees (ascending).
        """
        sub = self._g.subgraph(nodes)
        degs = [sub.degree(n) for n in nodes]
        return sorted(degs)


# === Quota computation ===


def compute_year_quotas(
    target_size: int, year_counts: dict[Year, int]
) -> dict[Year, int]:
    """Compute proportional per-year quotas summing to exactly target_size.

    Uses largest-remainder method to distribute rounding residuals fairly.

    Args:
        target_size: Desired total number of papers.
        year_counts: Map of year -> number of papers in full corpus.

    Returns:
        Map of year -> quota, summing to target_size.
    """
    total = sum(year_counts.values())
    if total == 0:
        return {}

    # Compute exact proportions and initial floor quotas
    exact = {year: target_size * count / total for year, count in year_counts.items()}
    quotas = {year: int(v) for year, v in exact.items()}
    remainders = {year: exact[year] - quotas[year] for year in exact}

    # Distribute residual by largest remainder
    shortfall = target_size - sum(quotas.values())
    for year in sorted(remainders, key=lambda y: remainders[y], reverse=True):
        if shortfall <= 0:
            break
        quotas[year] += 1
        shortfall -= 1

    return quotas


def _count_by_year(ids: set[CorpusId], year_map: dict[CorpusId, Year]) -> Counter[Year]:
    """Count papers per year for a set of paper IDs.

    Args:
        ids: Paper IDs to count.
        year_map: Map of paper_id -> publication year.

    Returns:
        Counter mapping year -> count.
    """
    return Counter(year for pid in ids if (year := year_map.get(pid)) is not None)


# === Selection ===


def _distribute_shortfall(
    shortfall: int,
    surplus_pools: dict[Year, list[CorpusId]],
    years: list[Year],
) -> dict[Year, int]:
    """Distribute shortfall across years proportionally.

    Uses largest-remainder method. The allocated extras sum to exactly ``shortfall``
    when enough surplus papers are available, otherwise allocates up to the total
    available capacity.

    Args:
        shortfall: Total extra papers to allocate.
        surplus_pools: Map of year -> ranked candidate list (already sorted).
        years: Years with surplus, in iteration order.

    Returns:
        Map of year -> number of extras to take.
    """
    total_available = sum(len(surplus_pools[y]) for y in years)
    if total_available == 0:
        return {}

    # Floor quotas
    exact = {y: shortfall * len(surplus_pools[y]) / total_available for y in years}
    extras = {y: min(int(exact[y]), len(surplus_pools[y])) for y in years}
    remainders = {y: exact[y] - extras[y] for y in years}

    # Distribute residual by largest remainder
    remaining = shortfall - sum(extras.values())
    for y in sorted(remainders, key=lambda y: remainders[y], reverse=True):
        if remaining <= 0:
            break
        if extras[y] < len(surplus_pools[y]):
            extras[y] += 1
            remaining -= 1

    return extras


def select_from_pool(
    pool_by_year: dict[Year, list[CorpusId]],
    graph: CitationGraph,
    quotas: dict[Year, int],
    rng: random.Random,
) -> set[CorpusId]:
    """Select papers from pool, preferring highest-degree papers per year.

    For each year stratum, takes the top papers by degree (in the full graph)
    up to the quota. If a year has fewer pool papers than its quota, takes all
    and redistributes the shortfall proportionally to other years.

    Args:
        pool_by_year: Map of year -> list of paper IDs in the k-core pool.
        graph: Full citation graph (for degree ranking).
        quotas: Map of year -> desired count.
        rng: Random number generator for tie-breaking.

    Returns:
        Set of selected paper IDs.
    """
    selected: set[CorpusId] = set()
    remaining_quota: dict[Year, int] = {}

    # First pass: fill from pool, track shortfalls
    shortfall_total = 0
    years_with_surplus: list[Year] = []

    for year, quota in quotas.items():
        pool = pool_by_year.get(year, [])
        if len(pool) <= quota:
            # Take all from this year
            selected.update(pool)
            remaining_quota[year] = quota - len(pool)
            shortfall_total += remaining_quota[year]
        else:
            # Rank by degree, break ties with RNG
            ranked = sorted(
                pool,
                key=lambda pid: (graph.degree(pid), rng.random()),
                reverse=True,
            )
            selected.update(ranked[:quota])
            remaining_quota[year] = 0
            years_with_surplus.append(year)

    # Second pass: redistribute shortfall to years with surplus
    if shortfall_total > 0 and years_with_surplus:
        surplus_pools = {
            year: sorted(
                [pid for pid in pool_by_year.get(year, []) if pid not in selected],
                key=lambda pid: (graph.degree(pid), rng.random()),
                reverse=True,
            )
            for year in years_with_surplus
        }
        total_surplus = sum(len(p) for p in surplus_pools.values())

        if total_surplus > 0:
            # Distribute using largest-remainder to sum exactly to shortfall
            extras = _distribute_shortfall(
                shortfall_total, surplus_pools, years_with_surplus
            )
            for year in years_with_surplus:
                if (take := extras.get(year, 0)) > 0:
                    selected.update(surplus_pools[year][:take])

    return selected


def fill_from_fringe(
    selected: set[CorpusId],
    graph: CitationGraph,
    valid_ids: set[CorpusId],
    quotas: dict[Year, int],
    year_map: dict[CorpusId, Year],
    rng: random.Random,
) -> set[CorpusId]:
    """Fill under-quota years from 1-hop fringe papers.

    For years that are under-filled, finds papers adjacent to the selected set
    that are not yet selected, ranks them by connection count to the selected
    set, and fills remaining quota.

    Args:
        selected: Currently selected paper IDs (not modified).
        graph: Full citation graph.
        valid_ids: All valid corpus IDs (universe).
        quotas: Map of year -> desired count.
        year_map: Map of paper_id -> publication year.
        rng: Random number generator for tie-breaking.

    Returns:
        New set of selected paper IDs with fringe papers added.
    """
    # Find under-filled years
    year_counts = _count_by_year(selected, year_map)
    under_filled: dict[Year, int] = {}
    for year, quota in quotas.items():
        deficit = quota - year_counts.get(year, 0)
        if deficit > 0:
            under_filled[year] = deficit

    if not under_filled:
        return set(selected)

    # Build fringe: papers adjacent to selected but not in selected
    fringe: set[CorpusId] = set()
    for pid in selected:
        for neighbour in graph.neighbours(pid):
            if neighbour not in selected and neighbour in valid_ids:
                fringe.add(neighbour)

    # Group fringe by year
    fringe_by_year: defaultdict[Year, list[CorpusId]] = defaultdict(list)
    for pid in fringe:
        year_fringe = year_map.get(pid)
        if year_fringe is not None and year_fringe in under_filled:
            fringe_by_year[year_fringe].append(pid)

    # Add new neighbours to selected
    selected_new = set(selected)

    # Fill each under-filled year from its fringe, ranked by connections
    for year, deficit in under_filled.items():
        candidates = fringe_by_year.get(year, [])
        if not candidates:
            continue

        # Rank by number of connections to the selected set, break ties with RNG
        ranked = sorted(
            candidates,
            key=lambda pid: (len(graph.neighbours(pid) & selected_new), rng.random()),
            reverse=True,
        )
        selected_new.update(ranked[:deficit])

    return selected_new


def _find_year_surplus(
    year_counts: Counter[Year], quotas: dict[Year, int]
) -> dict[Year, int]:
    """Find over-quota years and their surplus counts.

    Args:
        year_counts: Actual paper counts per year.
        quotas: Target paper counts per year.

    Returns:
        Map of year -> surplus count (only years with surplus > 0).
    """
    surplus: dict[Year, int] = {}
    for year, count in year_counts.items():
        extra = count - quotas.get(year, 0)
        if extra > 0:
            surplus[year] = extra
    return surplus


def _build_removal_order(
    selected: set[CorpusId],
    surplus_by_year: dict[Year, int],
    graph: CitationGraph,
    year_map: dict[CorpusId, Year],
    rng: random.Random,
) -> dict[Year, list[CorpusId]]:
    """Build per-year lists of papers ranked for removal (lowest degree first).

    Args:
        selected: Currently selected paper IDs.
        surplus_by_year: Map of year -> surplus count.
        graph: Citation graph (for degree ranking).
        year_map: Map of paper_id -> publication year.
        rng: Random number generator for tie-breaking.

    Returns:
        Map of year -> papers sorted by degree ascending (worst first).
    """
    removable_by_year: dict[Year, list[CorpusId]] = {}
    for year in surplus_by_year:
        papers = [pid for pid in selected if year_map.get(pid) == year]
        removable_by_year[year] = sorted(
            papers,
            key=lambda pid: (graph.degree(pid), rng.random()),
        )
    return removable_by_year


def _trim_to_target(
    selected: set[CorpusId],
    graph: CitationGraph,
    target_size: int,
    quotas: dict[Year, int],
    year_map: dict[CorpusId, Year],
    rng: random.Random,
) -> set[CorpusId]:
    """Trim a selection towards `target_size`, preserving year balance.

    Removes papers only from over-quota years, starting with the lowest-degree papers.
    Year quotas are respected: a year at or below its quota is never trimmed. If no
    over-quota papers remain, trimming stops early and the result may still exceed
    `target_size`.

    Args:
        selected: Currently selected paper IDs (not modified).
        graph: Full citation graph.
        target_size: Desired total size.
        quotas: Map of year -> target count.
        year_map: Map of paper_id -> publication year.
        rng: Random number generator for tie-breaking.

    Returns:
        New set, at most `target_size` (may be larger if year balance prevents further
        removal).
    """
    if len(selected) <= target_size:
        return set(selected)

    excess = len(selected) - target_size
    year_counts = _count_by_year(selected, year_map)
    surplus_by_year = _find_year_surplus(year_counts, quotas)
    removable_by_year = _build_removal_order(
        selected, surplus_by_year, graph, year_map, rng
    )

    # Remove from over-quota years, round-robin by most over-represented first
    result = set(selected)
    removed = 0
    while removed < excess:
        made_progress = False
        for year in sorted(
            surplus_by_year, key=lambda y: surplus_by_year[y], reverse=True
        ):
            if surplus_by_year[year] <= 0:
                continue

            removable = removable_by_year[year]
            if not removable:
                continue

            # Remove lowest-degree paper from year
            result.discard(removable.pop(0))
            surplus_by_year[year] -= 1
            removed += 1
            made_progress = True

            if removed >= excess:
                break

        if not made_progress:
            break

    return result


# === Analysis ===


@dataclass(frozen=True, kw_only=True)
class SamplingReport:
    """Statistics about a connected sample."""

    total: int
    """Total papers selected."""
    edges: int
    """Edges in the induced subgraph."""
    min_degree: int
    """Minimum degree in the induced subgraph."""
    mean_degree: float
    """Mean degree in the induced subgraph."""
    max_degree: int
    """Maximum degree in the induced subgraph."""
    components: int
    """Number of connected components."""
    year_distribution: dict[Year, int]
    """Actual year distribution of selected papers."""
    target_distribution: dict[Year, int]
    """Target year distribution (quotas)."""
    year_deviation: dict[Year, float]
    """Per-year deviation from target (percentage points)."""


def analyse_selection(
    graph: CitationGraph,
    selected: set[CorpusId],
    year_map: dict[CorpusId, Year],
    quotas: dict[Year, int],
) -> SamplingReport:
    """Compute statistics for a selected paper set.

    Args:
        graph: Full citation graph.
        selected: Set of selected paper IDs.
        year_map: Map of paper_id -> publication year.
        quotas: Target year distribution.

    Returns:
        SamplingReport with graph and distribution statistics.
    """
    edges = graph.subgraph_edge_count(selected)
    degrees = graph.subgraph_degree_sequence(selected)

    total_selected = len(selected)
    total_quotas = sum(quotas.values())

    min_deg = degrees[0] if degrees else 0
    max_deg = degrees[-1] if degrees else 0
    mean_deg = safediv(sum(degrees), total_selected)
    components = graph.subgraph_components(selected)

    # Year distribution
    year_dist = _count_by_year(selected, year_map)

    # Compute deviation from target
    all_years = sorted(set(quotas) | set(year_dist))

    year_deviation: dict[Year, float] = {}
    for year in all_years:
        actual_pct = safediv(year_dist.get(year, 0), total_selected) * 100
        target_pct = safediv(quotas.get(year, 0), total_quotas) * 100
        year_deviation[year] = round(actual_pct - target_pct, 2)

    return SamplingReport(
        total=total_selected,
        edges=edges,
        min_degree=min_deg,
        mean_degree=round(mean_deg, 2),
        max_degree=max_deg,
        components=components,
        year_distribution=dict(sorted(year_dist.items())),
        target_distribution=dict(sorted(quotas.items())),
        year_deviation=dict(sorted(year_deviation.items())),
    )


# === Orchestrator ===


@dataclass(frozen=True, kw_only=True)
class ConnectedSampleConfig:
    """Configuration for a connected sampling run."""

    data_dir: Path
    """Directory containing citations/ and papers/."""
    splits_dir: Path
    """Existing splits directory defining the paper universe."""
    output_dir: Path
    """Output directory for new splits."""
    target_size: int
    """Desired total number of papers."""
    min_degree: int
    """Minimum degree for k-core filtering."""
    seed: int
    """Random seed for tie-breaking."""


def _parse_train_years(s: str) -> tuple[int, int]:
    """Parse train_years from split config (e.g. '2019-2022' or 'all').

    Args:
        s: Value from split_config.json.

    Returns:
        Tuple of (start_year, end_year). (0, 0) if 'all'.
    """
    if s == "all":
        return (0, 0)

    parts = s.split("-")
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])

    return (0, 0)


@dataclass(frozen=True, kw_only=True)
class SplitBoundaries:
    """Year boundaries for train/dev/test splits read from split_config.json."""

    train_years: tuple[int, int]
    """Train year range (e.g. ``(2019, 2022)``). ``(0, 0)`` means all years."""
    dev_year: int
    """Year for the dev split."""
    test_year: int
    """Year for the test split."""


class _InputConfig(BaseModel, frozen=True):
    """Year boundaries read from an existing split_config.json."""

    train_years: str = "0-0"
    """Train year range (e.g. '2019-2022' or 'all')."""
    dev_year: int = 2023
    """Year for the dev split."""
    test_year: int = 2024
    """Year for the test split."""


class SamplingParameters(BaseModel, frozen=True):
    """Parameters used for connected sampling."""

    target_size: int
    """Desired total number of papers."""
    min_degree: int
    """Minimum degree for k-core filtering."""
    seed: int
    """Random seed for tie-breaking."""
    source_splits: str
    """Path to the source splits directory."""


class SplitCounts(BaseModel, frozen=True):
    """Paper counts in the output splits."""

    universe: int
    """Total papers in the source universe."""
    total: int
    """Total papers selected."""
    train: int
    """Papers in the training split."""
    dev: int
    """Papers in the dev split."""
    test: int
    """Papers in the test split."""


class GraphStatistics(BaseModel, frozen=True):
    """Graph statistics for the selected subgraph."""

    edges: int
    """Edges in the induced subgraph."""
    min_degree: int
    """Minimum degree in the induced subgraph."""
    mean_degree: float
    """Mean degree in the induced subgraph."""
    max_degree: int
    """Maximum degree in the induced subgraph."""
    connected_components: int
    """Number of connected components."""


class OutputSplitConfig(BaseModel, frozen=True):
    """Schema for the split_config.json written by connected sampling."""

    method: str
    """Sampling method identifier."""
    parameters: SamplingParameters
    """Sampling parameters used."""
    train_years: str
    """Train year range (e.g. '2019-2022' or 'all')."""
    dev_year: int
    """Year for the dev split."""
    test_year: int
    """Year for the test split."""
    counts: SplitCounts
    """Paper counts in the output splits."""
    graph_statistics: GraphStatistics
    """Graph statistics for the selected subgraph."""
    year_distribution: dict[Year, int]
    """Actual year distribution of selected papers."""
    target_distribution: dict[Year, int]
    """Target year distribution (quotas)."""
    year_deviation_pp: dict[Year, float]
    """Per-year deviation from target (percentage points)."""

    @classmethod
    def from_results(
        cls,
        config: ConnectedSampleConfig,
        report: SamplingReport,
        splits: SplitResult,
        *,
        universe_size: int,
        train_years: tuple[int, int],
        dev_year: int,
        test_year: int,
    ) -> OutputSplitConfig:
        """Build output config from sampling results.

        Args:
            config: The sampling configuration used.
            report: Analysis report from the selected subgraph.
            splits: Train/dev/test split result.
            universe_size: Total papers in the source universe.
            train_years: Year range for training (0, 0 means 'all').
            dev_year: Year for the dev split.
            test_year: Year for the test split.

        Returns:
            Fully populated output config ready for serialisation.
        """
        return cls(
            method="connected_sample",
            parameters=SamplingParameters(
                target_size=config.target_size,
                min_degree=config.min_degree,
                seed=config.seed,
                source_splits=str(config.splits_dir),
            ),
            train_years=(
                "all" if train_years[0] == 0 else f"{train_years[0]}-{train_years[1]}"
            ),
            dev_year=dev_year,
            test_year=test_year,
            counts=SplitCounts(
                universe=universe_size,
                total=report.total,
                train=len(splits.train),
                dev=len(splits.dev),
                test=len(splits.test),
            ),
            graph_statistics=GraphStatistics(
                edges=report.edges,
                min_degree=report.min_degree,
                mean_degree=report.mean_degree,
                max_degree=report.max_degree,
                connected_components=report.components,
            ),
            year_distribution=report.year_distribution,
            target_distribution=report.target_distribution,
            year_deviation_pp=report.year_deviation,
        )


def _read_split_boundaries(
    splits_dir: Path,
) -> SplitBoundaries:
    """Read train/dev/test year boundaries from split_config.json.

    Args:
        splits_dir: Directory containing split_config.json.

    Returns:
        Parsed year boundaries for each split.
    """
    config_path = splits_dir / "split_config.json"
    if not config_path.exists():
        die(f"No split_config.json found in {splits_dir}")
    config = _InputConfig.model_validate_json(config_path.read_bytes())

    train_years = _parse_train_years(config.train_years)
    return SplitBoundaries(
        train_years=train_years, dev_year=config.dev_year, test_year=config.test_year
    )


def run_connected_sample(config: ConnectedSampleConfig) -> None:
    """Run the connected sampling pipeline.

    Steps:
    1. Load split IDs and references, build citation graph
    2. Compute k-core pool
    3. Compute year quotas matching full corpus distribution
    4. Select from pool by degree, fill from fringe
    5. Re-split into train/dev/test by year
    6. Write output splits with statistics

    Args:
        config: Sampling configuration.
    """
    # Load universe from existing splits
    logger.info("Loading split IDs from %s", config.splits_dir)

    valid_ids = load_split_ids(config.splits_dir)
    if not valid_ids:
        die(f"No IDs found in {config.splits_dir}")
    logger.info("  Universe: %s papers", f"{len(valid_ids):,}")

    # Load references and build graph
    refs_path = resolve_jsonl(config.data_dir / "citations" / "references.jsonl.gz")
    if not refs_path.exists():
        die(f"References not found: {refs_path}")

    logger.info("Loading references from %s", refs_path)
    refs = load_references(refs_path, show_progress=True)
    logger.info("  %s papers with references", f"{len(refs):,}")

    logger.info("Building citation graph...")
    graph = CitationGraph.from_citations(refs, valid_ids)
    logger.info(
        "  Graph: %s nodes, %s edges",
        f"{graph.node_count():,}",
        f"{graph.edge_count():,}",
    )

    # Compute k-core pool
    logger.info("Computing %d-core...", config.min_degree)
    pool = graph.k_core(k=config.min_degree)
    logger.info("  Pool size: %s papers", f"{len(pool):,}")

    if len(pool) < config.target_size:
        logger.warning(
            "Pool (%d) is smaller than target (%d). Consider lowering --min-degree.",
            len(pool),
            config.target_size,
        )

    # Load year info
    logger.info("Loading metadata for year distribution...")
    all_years = load_metadata_years(config.data_dir / "papers")
    year_map = {pid: year for pid, year in all_years.items() if pid in valid_ids}
    logger.info("  %s papers with year info", f"{len(year_map):,}")

    # Compute full corpus year distribution (within the split universe)
    full_year_counts = _count_by_year(valid_ids, year_map)

    # Compute quotas
    quotas = compute_year_quotas(config.target_size, full_year_counts)
    logger.info("Year quotas: %s", quotas)

    # Group pool by year
    pool_by_year: defaultdict[Year, list[CorpusId]] = defaultdict(list)
    for pid in pool:
        if (year := year_map.get(pid)) is not None:
            pool_by_year[year].append(pid)

    # Select from pool
    rng = random.Random(config.seed)
    logger.info("Selecting from pool by degree...")
    selected = select_from_pool(pool_by_year, graph, quotas, rng)
    logger.info("  Selected: %s papers", f"{len(selected):,}")

    # Fill from fringe for under-represented years
    logger.info("Filling from fringe for under-represented years...")
    selected = fill_from_fringe(selected, graph, valid_ids, quotas, year_map, rng)
    logger.info("  After fringe fill: %s papers", f"{len(selected):,}")

    # Trim to exact target size if we overshot
    if len(selected) > config.target_size:
        excess = len(selected) - config.target_size
        logger.info("  Trimming %d excess papers...", excess)
        selected = _trim_to_target(
            selected, graph, config.target_size, quotas, year_map, rng
        )
        logger.info("  After trim: %s papers", f"{len(selected):,}")

    # Read year boundaries from source splits for re-splitting
    boundaries = _read_split_boundaries(config.splits_dir)

    # Re-split by year using same boundaries
    selected_years = {pid: year_map[pid] for pid in selected if pid in year_map}
    splits = split_papers_by_year(
        selected_years,
        boundaries.train_years,
        boundaries.dev_year,
        boundaries.test_year,
    )

    # Analyse the selection
    report = analyse_selection(graph, selected, year_map, quotas)

    # Log report
    logger.info("")
    logger.info("=== Connected Sample Report ===")
    logger.info("  Total: %s", f"{report.total:,}")
    logger.info(
        "  Splits: train=%s, dev=%s, test=%s",
        f"{len(splits.train):,}",
        f"{len(splits.dev):,}",
        f"{len(splits.test):,}",
    )
    logger.info("  Edges: %s", f"{report.edges:,}")
    logger.info(
        "  Degree: min=%d, mean=%.1f, max=%d",
        report.min_degree,
        report.mean_degree,
        report.max_degree,
    )
    logger.info("  Connected components: %d", report.components)
    logger.info("  Year distribution:")
    for year in sorted(report.year_distribution):
        actual = report.year_distribution.get(year, 0)
        target = report.target_distribution.get(year, 0)
        deviation = report.year_deviation.get(year, 0.0)
        logger.info(
            "    %d: %4d (target %4d, %+.1fpp)", year, actual, target, deviation
        )

    # Write output
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for name, ids in [
        ("train", splits.train),
        ("dev", splits.dev),
        ("test", splits.test),
    ]:
        path = config.output_dir / f"{name}.txt"
        path.write_text("\n".join(map(str, sorted(ids))) + "\n")

    # Write split_config.json
    output_config = OutputSplitConfig.from_results(
        config,
        report,
        splits,
        universe_size=len(valid_ids),
        train_years=boundaries.train_years,
        dev_year=boundaries.dev_year,
        test_year=boundaries.test_year,
    )
    config_path = config.output_dir / "split_config.json"
    config_path.write_text(output_config.model_dump_json(indent=2) + "\n")

    logger.info("")
    logger.info("Wrote splits to %s", config.output_dir)
