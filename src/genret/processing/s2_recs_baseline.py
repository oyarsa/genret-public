"""Semantic Scholar recommendations API baseline.

Fetches paper recommendations from the S2 API for each query paper,
filters to in-corpus results, and evaluates against citation ground truth.
Serves as an external-system baseline to compare against the graph pipeline.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TypedDict

import httpx

from genret.processing.facet_graph_eval import compute_metrics, load_ground_truth
from genret.types import CorpusId
from genret.utils import iter_jsonl, open_auto

logger = logging.getLogger(__name__)

REC_BASE_URL = "https://api.semanticscholar.org/recommendations/v1/papers"
FIELDS = "paperId,corpusId,title"


class RecItem(TypedDict):
    """A single recommendation from the S2 API."""

    corpusId: int
    paperId: str
    title: str
    source: str


async def _fetch_recs(
    client: httpx.AsyncClient,
    corpus_id: int,
    *,
    pool: str = "all-cs",
    limit: int = 200,
    max_retries: int = 3,
) -> list[RecItem]:
    """Fetch recommendations for a single paper from the S2 API.

    Args:
        client: HTTP client with API key header set.
        corpus_id: S2 corpus ID of the query paper.
        pool: Recommendation pool ("all-cs" or "recent").
        limit: Max recommendations to request.
        max_retries: Retry attempts on rate limiting.

    Returns:
        List of recommendation items.
    """
    url = f"{REC_BASE_URL}/forpaper/CorpusId:{corpus_id}"
    params = {"from": pool, "limit": str(limit), "fields": FIELDS}

    for attempt in range(max_retries):
        resp = await client.get(url, params=params)
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)
            logger.warning(f"Rate limited on {corpus_id}, waiting {wait}s...")
            await asyncio.sleep(wait)
            continue
        if resp.status_code == 404:
            logger.warning(f"Paper CorpusId:{corpus_id} not found in S2")
            return []
        resp.raise_for_status()
        data = resp.json()
        return [
            RecItem(
                corpusId=p["corpusId"],
                paperId=p["paperId"],
                title=p["title"],
                source=pool,
            )
            for p in data.get("recommendedPapers", [])
            if p.get("corpusId") is not None
        ]

    logger.error(f"Max retries exceeded for CorpusId:{corpus_id}")
    return []


def _load_corpus_ids(paper_graphs_path: Path) -> set[int]:
    """Load the set of corpus IDs from paper_graphs JSONL.

    Args:
        paper_graphs_path: Path to paper_graphs JSONL file.

    Returns:
        Set of integer corpus IDs.
    """
    ids: set[int] = set()
    for record in iter_jsonl(paper_graphs_path):
        ids.add(int(record["corpusid"]))
    return ids


async def fetch_all_recs(
    query_ids: list[CorpusId],
    *,
    pool: str = "all-cs",
    limit: int = 200,
    rate_limit_delay: float = 1.1,
) -> dict[CorpusId, list[RecItem]]:
    """Fetch S2 recommendations for all query papers.

    Args:
        query_ids: List of query corpus IDs.
        pool: Recommendation pool.
        limit: Max recommendations per query.
        rate_limit_delay: Seconds between requests (S2 allows 1 RPS).

    Returns:
        Dict mapping query ID to list of recommendation items.
    """
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    # The public recommendations endpoint works keyless. An invalid/revoked key
    # is rejected with 403, so only attach the header when a key is present.
    headers: dict[str, str] = {"x-api-key": api_key} if api_key else {}
    if not api_key:
        logger.warning("No SEMANTIC_SCHOLAR_API_KEY set; using keyless public access.")
    results: dict[CorpusId, list[RecItem]] = {}

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        for i, qid in enumerate(query_ids):
            if i > 0:
                await asyncio.sleep(rate_limit_delay)
            if (i + 1) % 50 == 0 or i == 0:
                logger.info(f"Fetching recommendations {i + 1}/{len(query_ids)}...")
            recs = await _fetch_recs(client, int(qid), pool=pool, limit=limit)
            results[qid] = recs

    return results


def filter_to_corpus(
    all_recs: dict[CorpusId, list[RecItem]],
    corpus_ids: set[int],
) -> dict[CorpusId, list[RecItem]]:
    """Filter recommendations to only papers in our corpus.

    Args:
        all_recs: Raw recommendations from S2 API.
        corpus_ids: Set of valid corpus IDs.

    Returns:
        Filtered recommendations, preserving order.
    """
    filtered: dict[CorpusId, list[RecItem]] = {}
    for qid, recs in all_recs.items():
        in_corpus = [r for r in recs if r["corpusId"] in corpus_ids]
        # Exclude self-recommendations
        in_corpus = [r for r in in_corpus if r["corpusId"] != int(qid)]
        filtered[qid] = in_corpus
    return filtered


def run_s2_recs_baseline(
    paper_graphs_path: Path,
    query_papers_path: Path,
    output_path: Path,
    co_cit_path: Path,
    bib_path: Path,
    *,
    pool: str = "all-cs",
    limit: int = 200,
) -> None:
    """Fetch S2 recommendations, filter to corpus, evaluate, and save.

    Args:
        paper_graphs_path: Path to paper_graphs JSONL (for corpus ID set).
        query_papers_path: Path to query papers JSONL.
        output_path: Path to write retrieval results.
        co_cit_path: Path to co_citations JSONL file.
        bib_path: Path to bib_coupling JSONL file.
        pool: S2 recommendation pool.
        limit: Max recommendations per query from API.
    """
    corpus_ids = _load_corpus_ids(paper_graphs_path)
    logger.info(f"Loaded {len(corpus_ids)} corpus IDs")

    query_ids = [
        CorpusId(int(record["corpusid"])) for record in iter_jsonl(query_papers_path)
    ]
    logger.info(f"Loaded {len(query_ids)} query papers")

    logger.info(
        f"Fetching S2 recommendations (pool={pool}, limit={limit})... "
        f"ETA ~{len(query_ids) * 1.1:.0f}s"
    )
    all_recs = asyncio.run(fetch_all_recs(query_ids, pool=pool, limit=limit))

    # Save raw results (before corpus filtering) for analysis
    raw_output = output_path.with_suffix(".raw.jsonl")
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with open_auto(raw_output, "w") as fh:
        for qid, recs in all_recs.items():
            record = {"corpusid": int(qid), "results": recs}
            fh.write(json.dumps(record) + "\n")
    logger.info(f"Wrote {len(all_recs)} raw result sets to {raw_output}")

    # Log coverage stats
    total_recs = sum(len(r) for r in all_recs.values())
    empty = sum(1 for r in all_recs.values() if not r)
    logger.info(
        f"Raw: {total_recs} total recommendations, "
        f"{empty}/{len(all_recs)} queries with no results"
    )

    filtered = filter_to_corpus(all_recs, corpus_ids)

    # Save filtered results in same format as BM25 baseline
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_auto(output_path, "w") as fh:
        for qid, recs in filtered.items():
            results = [
                {"corpusid": r["corpusId"], "score": float(len(recs) - i)}
                for i, r in enumerate(recs)
            ]
            out_record = {"corpusid": int(qid), "results": results}
            fh.write(json.dumps(out_record) + "\n")
    logger.info(f"Wrote {len(filtered)} filtered result sets to {output_path}")

    in_corpus_counts = [len(r) for r in filtered.values()]
    avg_in_corpus = sum(in_corpus_counts) / max(len(in_corpus_counts), 1)
    min_in_corpus = min(in_corpus_counts) if in_corpus_counts else 0
    max_in_corpus = max(in_corpus_counts) if in_corpus_counts else 0
    logger.info(
        f"In-corpus hits: avg={avg_in_corpus:.1f}, "
        f"min={min_in_corpus}, max={max_in_corpus}"
    )

    # Evaluate
    result_lists = {
        int(qid): [r["corpusId"] for r in recs] for qid, recs in filtered.items()
    }
    _evaluate_and_log(result_lists, co_cit_path, bib_path)


def _evaluate_and_log(
    result_lists: dict[int, list[int]],
    co_cit_path: Path,
    bib_path: Path,
) -> None:
    """Evaluate retrieval results and log metrics.

    Args:
        result_lists: Query ID to ordered result IDs.
        co_cit_path: Path to co_citations JSONL file.
        bib_path: Path to bib_coupling JSONL file.
    """
    valid_ids = frozenset(result_lists)
    combined_gt, co_cit_gt, bib_gt = load_ground_truth(co_cit_path, bib_path, valid_ids)

    for label, gt in [
        ("Combined", combined_gt),
        ("Co-citation", co_cit_gt),
        ("Bib-coupling", bib_gt),
    ]:
        metrics = compute_metrics(result_lists, gt)
        logger.info(
            f"{label}: Recall@5={metrics.recall[5]:.4f}, "
            f"Recall@10={metrics.recall[10]:.4f}, "
            f"Recall@20={metrics.recall[20]:.4f}, "
            f"MRR={metrics.mrr:.4f} "
            f"({metrics.num_queries} queries, {metrics.num_gt_pairs} GT pairs)"
        )
