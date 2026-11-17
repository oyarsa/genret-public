"""Alethiometer: on-demand facet extraction for papers not in the pre-built graph.

Given a corpus ID, loads the paper's fulltext from S2ORC, extracts facets
via an LLM, caches the result, and embeds the facets.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from genret.explorer.data_loading import FACET_NAMES
from genret.explorer.encoder import TextEncoder
from genret.explorer.facet_cache import FacetCache
from genret.llm.client import CompletionRequest, OpenAIClient, ReasoningEffort
from genret.paper_parsing import Facets, PaperGraph, parse_paper
from genret.processing.facet_extract_cmd import (
    FACET_SYSTEM_PROMPT,
    FacetPromptStyle,
    FacetsResponse,
    prepare_prompt,
)
from genret.types import CorpusId
from genret.utils import open_auto

logger = logging.getLogger(__name__)


def load_s2orc_paper(
    corpusid: int,
    s2orc_index: dict[int, tuple[str, int]],
    s2orc_dir: Path,
    abstract: str | None = None,
    title: str | None = None,
) -> PaperGraph | None:
    """Load a paper from S2ORC shards using the prebuilt index.

    Args:
        corpusid: The corpus ID to look up.
        s2orc_index: Mapping of corpusid → (shard_filename, line_number).
        s2orc_dir: Directory containing the S2ORC shard files.
        abstract: Optional abstract fallback (used if not in S2ORC body).
        title: Optional title fallback (used if S2ORC record has no title).

    Returns:
        Parsed PaperGraph, or None if not found or unparseable.
    """
    entry = s2orc_index.get(corpusid)
    if entry is None:
        return None

    shard_name, line_no = entry
    shard_path = s2orc_dir / shard_name

    if not shard_path.exists():
        logger.warning("S2ORC shard not found: %s", shard_path)
        return None

    with open_auto(shard_path, "r") as f:
        for i, line in enumerate(f):
            if i == line_no:
                record: dict[str, Any] = json.loads(line)
                paper = parse_paper(record, abstract=abstract)
                if paper is not None and not paper.title and title:
                    paper = paper.model_copy(update={"title": title})
                return paper

    return None


def embed_facets(
    facets: Facets, encoder: TextEncoder
) -> dict[str, npt.NDArray[np.float32]]:
    """Embed each facet's items with the text encoder.

    Returns per-facet embeddings as (K, D) arrays where K is the number
    of items for that facet.
    """
    embeddings: dict[str, npt.NDArray[np.float32]] = {}

    for name in FACET_NAMES:
        items = list(getattr(facets, name))
        if not items:
            embeddings[name] = np.zeros((0, 384), dtype=np.float32)
            continue

        embeddings[name] = encoder.encode(items)

    return embeddings


def mean_pool_facet_embeddings(
    embeddings: dict[str, npt.NDArray[np.float32]],
) -> dict[str, npt.NDArray[np.float32]]:
    """Mean-pool per-facet item embeddings into single vectors."""
    pooled: dict[str, npt.NDArray[np.float32]] = {}
    for name, vecs in embeddings.items():
        if vecs.shape[0] == 0:
            pooled[name] = np.zeros(384, dtype=np.float32)
        else:
            pooled[name] = np.asarray(vecs.mean(axis=0), dtype=np.float32)
    return pooled


async def extract_facets(
    corpusid: int, paper: PaperGraph, extraction_model: str
) -> Facets | None:
    """Extract facets from a paper using an LLM.

    Args:
        corpusid: The corpus ID for the paper.
        paper: The parsed paper graph.
        extraction_model: OpenAI model name (e.g. "gpt-5-mini").

    Returns:
        Extracted Facets, or None on failure.
    """
    prompt = prepare_prompt(paper, prompt_style=FacetPromptStyle.DEFAULT)
    request = CompletionRequest(system=FACET_SYSTEM_PROMPT, user=prompt)

    # Minimal effort matches the offline corpus extraction; the API default
    # (medium) tripled latency for no known quality gain.
    client = OpenAIClient(
        model=extraction_model, reasoning_effort=ReasoningEffort.MINIMAL
    )
    result = await client.complete(request, FacetsResponse)

    if result.result is None:
        logger.error("Facet extraction failed for %d", corpusid)
        return None

    return result.result.finalise(CorpusId(corpusid))


async def get_or_extract_facets(
    corpusid: int,
    cache: FacetCache,
    s2orc_index: dict[int, tuple[str, int]],
    s2orc_dir: Path,
    extraction_model: str,
    title: str | None = None,
) -> Facets | None:
    """Get facets from cache or extract them from S2ORC fulltext.

    Args:
        corpusid: Paper corpus ID.
        cache: Aleth facet cache.
        s2orc_index: S2ORC fulltext index.
        s2orc_dir: Directory with S2ORC shard files.
        extraction_model: LLM model name for extraction.
        title: Optional title fallback from metadata (used if S2ORC
            record has no title).

    Returns:
        Extracted or cached Facets, or None if extraction fails.
    """
    cached = cache.get(corpusid)
    if cached is not None:
        logger.info("Aleth cache hit for %d", corpusid)
        return cached

    paper = load_s2orc_paper(corpusid, s2orc_index, s2orc_dir, title=title)
    if paper is None:
        logger.error("Could not load S2ORC fulltext for %d", corpusid)
        return None

    facets = await extract_facets(corpusid, paper, extraction_model)
    if facets is None:
        return None

    cache.put(facets)
    logger.info("Extracted and cached facets for %d", corpusid)
    return facets
