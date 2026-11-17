"""CLI subcommand for the graph explorer."""
# ruff: noqa: PLC0415  Enable function-level imports

from pathlib import Path
from typing import Annotated

import typer

from genret.utils import make_app

app = make_app("Interactive graph explorer for facet-decomposed paper similarity")


@app.callback()
def _callback() -> None:
    """Keep ``serve`` an explicit subcommand for the standalone entry point.

    Typer collapses single-command apps into the bare command otherwise,
    which would make ``graft serve`` fail.
    """


def main() -> None:
    """Standalone entry point (``graft``).

    Unlike ``genret explorer``, this skips the full CLI import graph, which
    pulls torch/transformers when the training extras are installed.
    """
    from dotenv import load_dotenv

    from genret.log_config import setup_logging

    load_dotenv()
    setup_logging()
    app()


@app.command()
def serve(
    edge_dir: Annotated[
        Path,
        typer.Option("--edge-dir", help="Directory with paper_edges_*.npz files"),
    ] = Path("output/v2_crossfacet_meanpool"),
    dense_dir: Annotated[
        Path,
        typer.Option("--dense-dir", help="Directory with facet_graph_*.npz files"),
    ] = Path("data/s2orc-nlp-v2/splits-connected-10k-v2/processed"),
    facets_path: Annotated[
        Path,
        typer.Option("--facets-path", help="Path to facets.jsonl(.zst)"),
    ] = Path("data/s2orc-nlp-v2/splits-connected-10k-v2/processed/facets.jsonl.zst"),
    metadata_dir: Annotated[
        Path,
        typer.Option("--metadata-dir", help="Directory with metadata_*.jsonl files"),
    ] = Path("data/s2orc-nlp-v2/papers"),
    s2orc_dir: Annotated[
        Path,
        typer.Option("--s2orc-dir", help="Directory with S2ORC fulltext shards"),
    ] = Path("data/s2orc-nlp-v2/s2orc"),
    model_dir: Annotated[
        Path,
        typer.Option("--model-dir", help="ModernBERT reranker checkpoint directory"),
    ] = Path("output/phase2_modernbert_150k/checkpoint"),
    rerank: Annotated[
        bool,
        typer.Option("--rerank/--no-rerank", help="Enable reranking"),
    ] = False,
    rerank_model: Annotated[
        str,
        typer.Option(
            "--rerank-model",
            help="Reranker: 'local' for ModernBERT, or an LLM model name.",
        ),
    ] = "local",
    rerank_base_url: Annotated[
        str | None,
        typer.Option(
            "--rerank-base-url",
            help="API base URL for LLM reranker (e.g. OpenRouter).",
        ),
    ] = None,
    rerank_api_key: Annotated[
        str | None,
        typer.Option(
            "--rerank-api-key",
            help="API key for LLM reranker (overrides env var).",
        ),
    ] = None,
    rerank_device: Annotated[
        str,
        typer.Option("--rerank-device", help="Device for local reranker (cpu or cuda)"),
    ] = "cpu",
    extraction_model: Annotated[
        str,
        typer.Option("--extraction-model", help="LLM model for facet extraction"),
    ] = "gpt-5-mini",
    explanation_model: Annotated[
        str,
        typer.Option(
            "--explanation-model",
            help="LLM model for on-demand connection explanations",
        ),
    ] = "gpt-5.6-luna",
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", help="Cache directory for Aleth and S2ORC index"),
    ] = Path("cache"),
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Host to bind to. Loopback by default: the server has no"
            " auth and its recommend/explain endpoints spend API credit,"
            " so exposing it is an explicit choice (--host 0.0.0.0).",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="Port to bind to"),
    ] = 8000,
    dev: Annotated[
        bool,
        typer.Option("--dev", help="Development mode (skip static file serving)"),
    ] = False,
    encoder: Annotated[
        str,
        typer.Option(
            "--encoder",
            help="Text encoder backend: 'onnx' (no torch) or 'torch'.",
        ),
    ] = "onnx",
    onnx_dir: Annotated[
        Path,
        typer.Option(
            "--onnx-dir",
            help="Exported ONNX encoder directory (see scripts/export_minilm_onnx.py).",
        ),
    ] = Path("output/minilm_onnx"),
    frontend_dir: Annotated[
        Path | None,
        typer.Option(
            "--frontend-dir",
            help="Built frontend to serve. Defaults to frontend/dist beside"
            " the source tree; deployments of the installed package must"
            " pass this.",
        ),
    ] = None,
    silver: Annotated[
        bool,
        typer.Option(
            "--silver/--no-silver",
            help="Enable Silver out-of-corpus acquisition and live S2 title"
            " search (requires SEMANTIC_SCHOLAR_API_KEY).",
        ),
    ] = False,
    rate_limit: Annotated[
        int,
        typer.Option(
            "--rate-limit",
            help="Requests per client per hour for the endpoints that call"
            " an LLM (recommend, explain). 0 disables limiting.",
        ),
    ] = 40,
    rate_limit_total: Annotated[
        int,
        typer.Option(
            "--rate-limit-total",
            help="Requests per hour across all clients for those endpoints.",
        ),
    ] = 400,
    proxy_headers: Annotated[
        bool,
        typer.Option(
            "--proxy-headers/--no-proxy-headers",
            help="Trust X-Forwarded-* headers. Enable only behind a reverse"
            " proxy: it decides which address the rate limiter counts.",
        ),
    ] = False,
) -> None:
    """Start the interactive graph explorer server.

    Use --rerank to enable reranking. By default uses the local ModernBERT
    cross-encoder. Pass --rerank-model gpt-5.4-mini to use an LLM instead.
    """
    import uvicorn

    from genret.explorer.rate_limit import RateLimitConfig
    from genret.explorer.server import ServerConfig, create_app

    config = ServerConfig(
        edge_dir=edge_dir,
        dense_dir=dense_dir,
        facets_path=facets_path,
        metadata_dir=metadata_dir,
        s2orc_dir=s2orc_dir,
        model_dir=model_dir,
        rerank=rerank,
        rerank_model=rerank_model,
        rerank_base_url=rerank_base_url,
        rerank_api_key=rerank_api_key,
        rerank_device=rerank_device,
        extraction_model=extraction_model,
        explanation_model=explanation_model,
        cache_dir=cache_dir,
        dev_mode=dev,
        encoder_backend=encoder,
        onnx_dir=onnx_dir,
        frontend_dir=frontend_dir,
        silver=silver,
        rate_limit=RateLimitConfig(per_client=rate_limit, total=rate_limit_total),
    )
    uvicorn.run(
        create_app(config),
        host=host,
        port=port,
        proxy_headers=proxy_headers,
        forwarded_allow_ips="*" if proxy_headers else None,
    )
