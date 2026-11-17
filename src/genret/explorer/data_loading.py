"""Startup data loading for the graph explorer."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

from genret.explorer.paper_search import SearchablePaper, normalize_title
from genret.paper_parsing import Facets
from genret.processing.facet_retrieval import load_edge_graphs
from genret.utils import open_auto

logger = logging.getLogger(__name__)

FACET_NAMES = ("problems", "methods", "results", "contributions")
"""Standard facet names used for corpus vectors."""


@dataclass(frozen=True)
class PaperMeta:
    """Lightweight paper metadata for the explorer."""

    corpusid: int
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    citation_count: int | None
    url: str | None


@dataclass(frozen=True)
class ExplorerData:
    """All preloaded data for the explorer backend."""

    graphs: dict[str, csr_array]
    corpusids: npt.NDArray[np.int64]
    cid_to_idx: dict[int, int]
    corpus_vectors: dict[str, npt.NDArray[np.float32]]
    facets_map: dict[int, Facets]
    metadata: dict[int, PaperMeta]


CROSS_FACET_NAME = "problems_x_contributions"
"""Cross-facet graph name (loaded if file exists)."""


def _load_corpus_vectors(
    dense_dir: Path,
) -> tuple[dict[str, npt.NDArray[np.float32]], npt.NDArray[np.int64]]:
    """Load per-facet corpus vectors from facet_graph_*.npz files.

    Loads only the four standard same-facet vector banks.

    Cross-facet retrieval artifacts such as `problems_x_contributions`
    are stored as pairwise similarity matrices rather than per-paper
    `corpus_vectors`, so they are not part of this dense query-time
    vector index.
    """
    vectors: dict[str, npt.NDArray[np.float32]] = {}
    corpusids: npt.NDArray[np.int64] | None = None

    for name in FACET_NAMES:
        path = dense_dir / f"facet_graph_{name}.npz"
        if not path.exists():
            msg = f"Missing corpus vector file: {path}"
            raise FileNotFoundError(msg)

        with np.load(path) as npz:
            vectors[name] = npz["corpus_vectors"].astype(np.float32)
            if corpusids is None:
                corpusids = npz["corpusids"]

    assert corpusids is not None
    logger.info(
        f"Loaded corpus vectors for {len(FACET_NAMES)} facets, "
        f"dim={next(iter(vectors.values())).shape[1]}"
    )
    return vectors, corpusids


def _parse_metadata_record(record: dict[str, Any]) -> PaperMeta:
    """Parse a single metadata JSONL record into PaperMeta."""
    authors_raw: list[Any] = record.get("authors") or []
    authors = [
        cast(dict[str, str], a)["name"] if isinstance(a, dict) else str(a)
        for a in authors_raw
    ]
    return PaperMeta(
        corpusid=record["corpusid"],
        title=record.get("title") or "",
        authors=authors,
        year=record.get("year"),
        venue=record.get("venue"),
        citation_count=record.get("citationcount"),
        url=record.get("url"),
    )


def load_metadata(
    metadata_dir: Path, corpus_cids: set[int] | None = None
) -> dict[int, PaperMeta]:
    """Load paper metadata from JSONL files.

    Args:
        metadata_dir: Directory containing metadata_*.jsonl files.
        corpus_cids: If provided, only load these corpus IDs.
            If None, loads all papers.
    """
    meta: dict[int, PaperMeta] = {}

    for path in sorted(metadata_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                record: dict[str, Any] = json.loads(line)
                cid = record["corpusid"]
                if corpus_cids is not None and cid not in corpus_cids:
                    continue
                meta[cid] = _parse_metadata_record(record)

    logger.info(f"Loaded metadata for {len(meta)} papers")
    return meta


def build_searchable_papers(
    metadata: dict[int, PaperMeta],
    graph_cids: set[int],
    fulltext_cids: set[int] | None = None,
) -> list[SearchablePaper]:
    """Build the searchable papers list from loaded metadata.

    In-graph papers (11k) get source="in_graph"; papers with S2ORC fulltext
    get "in_corpus". When ``fulltext_cids`` is given, papers with neither
    are excluded entirely: they cannot be recommended (recommend routes on
    fulltext membership), so listing them as "in_corpus" promised a flow
    that would fail at the Silver acquisition step.

    Args:
        metadata: All loaded paper metadata.
        graph_cids: Corpus IDs of in-graph papers.
        fulltext_cids: Corpus IDs with S2ORC fulltext. None includes every
            metadata paper, labeled "in_corpus" when not in the graph.
    """
    papers: list[SearchablePaper] = []
    skipped = 0
    for meta in metadata.values():
        if not meta.title.strip():
            continue

        if meta.corpusid in graph_cids:
            source = "in_graph"
        elif fulltext_cids is None or meta.corpusid in fulltext_cids:
            source = "in_corpus"
        else:
            skipped += 1
            continue

        papers.append(
            SearchablePaper(
                corpusid=meta.corpusid,
                title=meta.title,
                title_normalized=normalize_title(meta.title),
                authors=meta.authors,
                year=meta.year,
                venue=meta.venue,
                source=source,
            )
        )

    logger.info(
        f"Built searchable index with {len(papers)} papers"
        f" ({skipped} without fulltext excluded)"
    )
    return papers


def build_s2orc_index(s2orc_dir: Path, cache_path: Path) -> dict[int, tuple[str, int]]:
    """Build or load the S2ORC fulltext index.

    Maps corpusid → (shard_filename, line_number) for fast lookup.
    Cached to disk after the first build.

    Args:
        s2orc_dir: Directory containing S2ORC shard files.
        cache_path: Path to cache the index JSON.

    Returns:
        Mapping from corpus ID to (shard filename, line number).
    """
    if cache_path.exists():
        raw: dict[str, list[str | int]] = json.loads(cache_path.read_bytes())
        index = {int(k): (str(v[0]), int(v[1])) for k, v in raw.items()}
        logger.info(f"Loaded S2ORC index from cache: {len(index)} entries")
        return index

    index: dict[int, tuple[str, int]] = {}

    for shard_path in sorted(s2orc_dir.glob("*.jsonl*")):
        shard_name = shard_path.name

        with open_auto(shard_path, "r") as f:
            for line_no, line in enumerate(f):
                try:
                    record = json.loads(line)
                    cid = record.get("corpusid")
                    if cid is not None:
                        index[int(cid)] = (shard_name, line_no)
                except json.JSONDecodeError:
                    continue

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({str(k): list(v) for k, v in index.items()}, f)

    logger.info(f"Built S2ORC index: {len(index)} entries, cached to {cache_path}")
    return index


def load_explorer_data(
    edge_dir: Path,
    dense_dir: Path,
    facets_path: Path,
    metadata_dir: Path,
) -> ExplorerData:
    """Load all data needed for the explorer at startup."""
    from genret.paper_parsing import load_facets_map  # noqa: PLC0415

    graphs, graph_cids = load_edge_graphs(edge_dir)
    corpus_vectors, _vec_cids = _load_corpus_vectors(dense_dir)

    # Use graph corpusids as canonical ordering
    corpusids = graph_cids
    cid_to_idx = {int(cid): idx for idx, cid in enumerate(corpusids)}
    corpus_set = set(cid_to_idx.keys())

    facets_map_raw = load_facets_map(facets_path)
    facets_map = {int(k): v for k, v in facets_map_raw.items()}

    metadata = load_metadata(metadata_dir, corpus_set)

    logger.info(
        f"Explorer data loaded: {len(corpusids)} papers, "
        f"{len(graphs)} graphs, {len(facets_map)} facets, "
        f"{len(metadata)} metadata records"
    )

    return ExplorerData(
        graphs=graphs,
        corpusids=corpusids,
        cid_to_idx=cid_to_idx,
        corpus_vectors=corpus_vectors,
        facets_map=facets_map,
        metadata=metadata,
    )
