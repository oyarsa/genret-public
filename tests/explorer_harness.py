"""Shared test harness for explorer server tests.

Synthetic dataset construction, the fake deterministic encoder, and SSE
helpers. Fixtures that wire these into pytest live in ``conftest.py``.
"""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt
from fastapi import FastAPI

from genret.explorer.server import ServerConfig
from genret.paper_parsing import Facets
from genret.types import CorpusId

FACET_NAMES = ["problems", "methods", "results", "contributions"]

GRAPH_CIDS = list(range(1001, 1021))
"""In-graph papers: facets preloaded, edge graphs built."""
CORPUS_CIDS = list(range(2001, 2006))
"""In-corpus papers: metadata + S2ORC fulltext, no graph membership."""
GHOST_CIDS = list(range(3001, 3004))
"""Metadata-only papers: no fulltext, must be excluded from search."""

CACHED_CORPUS_CID = 2001
"""In-corpus paper whose facets are pre-cached in the Aleth cache."""

DIM = 384


_PAPER_MENTION = re.compile(r"paper (\d+)")


def _seed_vec(key: str) -> npt.NDArray[np.float32]:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def fake_embedding(text: str) -> npt.NDArray[np.float32]:
    """Deterministic normalised embedding derived from the text's hash.

    Texts mentioning "paper N" embed close to a per-paper base direction,
    so a query quoting a paper's facet item lands nearest that paper on
    every facet -- a stand-in for real semantic similarity that makes
    ranking assertions deterministic.
    """
    mention = _PAPER_MENTION.search(text)
    if mention is None:
        return _seed_vec(text)

    vec = 0.9 * _seed_vec(f"paper-{mention.group(1)}") + 0.1 * _seed_vec(text)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


class FakeEncoder:
    """Drop-in TextEncoder returning hash-based embeddings."""

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Encode texts deterministically."""
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.stack([fake_embedding(t) for t in texts])


def facet_items(cid: int, facet: str) -> list[str]:
    """The two facet items every synthetic paper has per facet."""
    return [f"{facet} item {k} of paper {cid}" for k in (1, 2)]


def make_facets(cid: int) -> Facets:
    """Synthetic facets for a paper, matching the corpus vectors."""
    return Facets(
        corpusid=CorpusId(cid),
        problems=tuple(facet_items(cid, "problems")),
        methods=tuple(facet_items(cid, "methods")),
        results=tuple(facet_items(cid, "results")),
        contributions=tuple(facet_items(cid, "contributions")),
    )


def paper_facet_vector(cid: int, facet: str) -> npt.NDArray[np.float32]:
    """Mean-pooled facet vector, mirroring the real pipeline."""
    vecs = np.stack([fake_embedding(t) for t in facet_items(cid, facet)])
    pooled = vecs.mean(axis=0)
    return (pooled / np.linalg.norm(pooled)).astype(np.float32)


def title_for(cid: int) -> str:
    """Distinctive title per paper tier, for title-search tests."""
    if cid in GRAPH_CIDS:
        return f"Graph Paper {cid} on Attention Mechanisms"
    if cid in CORPUS_CIDS:
        return f"Corpus Paper {cid} on Neural Retrieval"
    return f"Ghost Paper {cid} Without Fulltext"


def write_edge_graph(path: Path, n: int, corpusids: list[int]) -> None:
    """Chain graph: i -> i+1 (0.9) and i -> i+2 (0.5), directed CSR."""
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for i in range(n):
        if i + 1 < n:
            indices.append(i + 1)
            data.append(0.9)
        if i + 2 < n:
            indices.append(i + 2)
            data.append(0.5)
        indptr.append(len(indices))

    np.savez(
        path,
        corpusids=np.asarray(corpusids, dtype=np.int64),
        indptr=np.asarray(indptr, dtype=np.int32),
        indices=np.asarray(indices, dtype=np.int32),
        data=np.asarray(data, dtype=np.float16),
    )


@dataclass(frozen=True)
class ExplorerDataset:
    """Paths of the synthetic on-disk dataset."""

    edge_dir: Path
    dense_dir: Path
    facets_path: Path
    metadata_dir: Path
    s2orc_dir: Path


def build_dataset(root: Path) -> ExplorerDataset:
    """Write the synthetic dataset under root."""
    edge_dir = root / "edges"
    dense_dir = root / "dense"
    metadata_dir = root / "metadata"
    s2orc_dir = root / "s2orc"
    for d in (edge_dir, dense_dir, metadata_dir, s2orc_dir):
        d.mkdir()

    n = len(GRAPH_CIDS)
    for facet in FACET_NAMES:
        vectors = np.stack([paper_facet_vector(cid, facet) for cid in GRAPH_CIDS])
        np.savez(
            dense_dir / f"facet_graph_{facet}.npz",
            corpusids=np.asarray(GRAPH_CIDS, dtype=np.int64),
            corpus_vectors=vectors,
        )
        write_edge_graph(edge_dir / f"paper_edges_{facet}.npz", n, GRAPH_CIDS)

    facets_path = dense_dir / "facets.jsonl"
    with open(facets_path, "w", encoding="utf-8") as f:
        f.writelines(make_facets(cid).model_dump_json() + "\n" for cid in GRAPH_CIDS)

    with open(metadata_dir / "metadata_00.jsonl", "w", encoding="utf-8") as f:
        for cid in GRAPH_CIDS + CORPUS_CIDS + GHOST_CIDS:
            record = {
                "corpusid": cid,
                "title": title_for(cid),
                "authors": [{"name": "Ada Author"}, {"name": "Ben Builder"}],
                "year": 2024,
                "venue": "TestConf",
                "citationcount": 7,
                "url": f"https://example.org/{cid}",
            }
            f.write(json.dumps(record) + "\n")

    with open(s2orc_dir / "shard_00.jsonl", "w", encoding="utf-8") as f:
        for cid in CORPUS_CIDS:
            f.write(json.dumps({"corpusid": cid, "title": title_for(cid)}) + "\n")

    return ExplorerDataset(
        edge_dir=edge_dir,
        dense_dir=dense_dir,
        facets_path=facets_path,
        metadata_dir=metadata_dir,
        s2orc_dir=s2orc_dir,
    )


def make_config(
    dataset: ExplorerDataset, cache_dir: Path, *, silver: bool = False
) -> ServerConfig:
    """ServerConfig pointing at the synthetic dataset."""
    return ServerConfig(
        edge_dir=dataset.edge_dir,
        dense_dir=dataset.dense_dir,
        facets_path=dataset.facets_path,
        metadata_dir=dataset.metadata_dir,
        s2orc_dir=dataset.s2orc_dir,
        cache_dir=cache_dir,
        dev_mode=True,
        silver=silver,
    )


@dataclass
class ExplorerHarness:
    """A running in-process explorer app and a client against it."""

    app: FastAPI
    client: httpx.AsyncClient
    cache_dir: Path


async def collect_sse(
    client: httpx.AsyncClient, url: str
) -> list[tuple[str, dict[str, Any]]]:
    """Consume an SSE stream into a list of (event, data) pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    current_event = ""
    async with client.stream("GET", url) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            if line.startswith("event: "):
                current_event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((current_event, json.loads(line.removeprefix("data: "))))
    return events
