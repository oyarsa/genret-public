"""Standalone explorer server for Playwright e2e tests.

Builds the synthetic dataset from ``explorer_harness`` into a temp
directory, replaces the MiniLM encoder with the deterministic fake, and
serves the real FastAPI app in production mode (frontend/dist mounted at
/). No model artifacts, API keys, or network access needed.

Usage:
    uv run --no-sync python tests/explorer_e2e_server.py --port 8321

Run ``npm run build`` in frontend/ first so dist/ exists.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import genret.explorer.explain as explain_mod
import genret.explorer.server as server_mod
from explorer_harness import (
    CACHED_CORPUS_CID,
    FakeEncoder,
    build_dataset,
    make_config,
    make_facets,
)
from genret.explorer.encoder import DEFAULT_ONNX_DIR, TextEncoder
from genret.explorer.facet_cache import FacetCache
from genret.explorer.server import create_app


def _fake_load_encoder(backend: str, onnx_dir: Path = DEFAULT_ONNX_DIR) -> TextEncoder:
    """Deterministic encoder; the e2e server needs no model artifact."""
    return FakeEncoder()


async def _fake_call_llm(model: str, user_prompt: str) -> str:
    """Canned explanation; the e2e server makes no LLM calls."""
    return "These papers share a canned e2e connection."


def main() -> None:
    """Build the fixture dataset and serve the app."""
    import uvicorn  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    server_mod.load_encoder = _fake_load_encoder
    explain_mod._call_llm = _fake_call_llm
    os.environ.setdefault("OPENAI_API_KEY", "e2e-dummy")

    root = Path(tempfile.mkdtemp(prefix="explorer-e2e-"))
    dataset = build_dataset(root)
    cache_dir = root / "cache"
    aleth_dir = cache_dir / "aleth"
    aleth_dir.mkdir(parents=True)
    FacetCache(aleth_dir).put(make_facets(CACHED_CORPUS_CID))

    config = make_config(dataset, cache_dir)
    # Production mode so frontend/dist is mounted at /
    config.dev_mode = False

    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
