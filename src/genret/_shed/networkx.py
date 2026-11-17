"""Typed wrappers for networkx operations (no type stubs)."""

# pyright: basic

from typing import Any

import networkx as nx

from genret.types import CorpusId


def k_core(graph: Any, *, k: int) -> set[CorpusId]:
    """Compute the k-core of a graph.

    Args:
        graph: Undirected networkx graph.
        k: Minimum degree threshold.

    Returns:
        Set of node IDs in the k-core.
    """
    core: nx.Graph = nx.k_core(graph, k=k)  # type: ignore[missing-attribute]
    return set(core.nodes())


def number_connected_components(graph: Any) -> int:
    """Count connected components in a graph.

    Args:
        graph: Undirected networkx graph (or subgraph view).

    Returns:
        Number of connected components.
    """
    return int(nx.number_connected_components(graph))  # type: ignore[missing-attribute]
