"""CLI commands for the facet graph pipeline.

Commands for building dense similarity matrices, candidate graphs, sparse
edge graphs, retrieval, and the (shelved) re-ranker. The commands are
defined in themed sibling modules and attached via their register hooks:

- ``graph_cli_build``     — embed-local, build-facet-graph,
  build-candidate-graph, build-edges.
- ``graph_cli_retrieve``  — retrieve, batch-retrieve, retrieve-new,
  evaluate-graph.
- ``graph_cli_rerank``    — rerank, train-reranker,
  train-distilled-reranker, apply-reranker.
- ``graph_cli_baselines`` — bm25-baseline, rerank-retrieval,
  s2-recs-baseline, embedding-baseline.
"""

import logging

import typer

from genret.processing.graph_cli_baselines import register_baseline_commands
from genret.processing.graph_cli_build import register_build_commands
from genret.processing.graph_cli_rerank import register_rerank_commands
from genret.processing.graph_cli_retrieve import register_retrieve_commands

logger = logging.getLogger(__name__)


def register_graph_commands(app: typer.Typer) -> None:
    """Register all graph pipeline commands on the given Typer app.

    Args:
        app: The parent Typer app to add commands to.
    """
    register_build_commands(app)
    register_retrieve_commands(app)
    register_rerank_commands(app)
    register_baseline_commands(app)


__all__ = ["register_graph_commands"]
