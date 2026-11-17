"""Sample paired dense+graph / genret result rows for blind human annotation.

Supports the retrieval human-eval validation study: for each sampled query,
an annotator rates relatedness (and ideally the facet) of one dense+graph
result and one genret (deployed generative retriever) result, without
knowing which system produced which row. Recall for each system can later
be correlated against the human relatedness ratings via the answer key.

Sampling procedure: for each of `--n-queries` sampled queries, draw one
candidate from each method's top-50 ranked results (50 is genret's own
cap, so dense+graph is truncated to match for comparability), targeting a
roughly even hit/miss split per method against strict gold. Hits are the
natural minority at this depth (~12-15% of slots), so only the hit target
is actively balanced; misses are already the majority by construction.

Partial-overlap design across two annotators: `--overlap-queries` of the
`--n-queries` sampled queries go into a shared core rated by both
annotators (the only rows inter-annotator agreement can be computed on);
the remainder is split roughly evenly into two exclusive sets, one per
annotator, extending total dataset coverage without doubling effort on
every item. Each annotator's sheet is their exclusive rows plus the core
rows, shuffled together so the shared rows aren't identifiable by position.

Three files are written:

- `--output-a` / `--output-b`: the two blind annotator sheets (no method or
  gold-hit column). Each is ready to send to its annotator as-is.
- `--key-output`: the internal answer key (method, gold-hit, bucket, corpus
  IDs per `item_hash`), for joining ratings back to systems and computing
  agreement after annotation.

Usage:
    uv run python scripts/eval/sample_retrieval_annotation.py \
        --dense-graph output/eval_1k/retrieval_graph.jsonl \
        --genret output/genret/eval/glen-bullets-1k-1b-78lp-optionB_per_bullet.jsonl \
        --paper-graphs output/splits-full/processed/paper_graphs.jsonl.zst \
        --gold output/genret/codebook_training_pairs_1k.jsonl.zst \
        --n-queries 80 --overlap-queries 30 \
        --output-a output/genret/retrieval_annotation_a.csv \
        --output-b output/genret/retrieval_annotation_b.csv \
        --key-output output/genret/retrieval_annotation_key.csv

Requires `SEMANTIC_SCHOLAR_API_KEY` to be set: `query_url`/`candidate_url`
are fetched live from the Semantic Scholar API rather than constructed
locally, since the website's paper URLs are keyed by its internal
`paperId`, not by corpus ID.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib.util
import itertools
import random
from pathlib import Path
from typing import Annotated

import typer

from genret.processing.reranker import load_titles
from genret.s2.api import SemanticScholarAPI
from genret.types import CorpusId
from genret.utils import iter_jsonl, require_env

S2_BATCH_SIZE = 500
"""Semantic Scholar batch-endpoint limit."""

_spec = importlib.util.spec_from_file_location(
    "score_glen_eval", Path(__file__).parent / "score_glen_eval.py"
)
assert _spec is not None
assert _spec.loader is not None
score_glen_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_glen_eval)
"""Sibling script loaded by path (`scripts/` isn't a package), for its strict-gold loader."""

SAMPLE_DEPTH = 50
"""Shared candidate-depth cap; genret's eval only carries 50 results/query."""

DENSE_GRAPH_METHOD = "dense_graph"
"""Internal method label for the dense+graph baseline (blinded from annotators)."""

GENRET_METHOD = "genret"
"""Internal method label for the deployed generative retriever (blinded from annotators)."""

CORE_BUCKET = "core"
"""Query bucket rated by both annotators, for computing inter-annotator agreement."""

EXCLUSIVE_A_BUCKET = "exclusive_a"
"""Query bucket rated only by annotator A."""

EXCLUSIVE_B_BUCKET = "exclusive_b"
"""Query bucket rated only by annotator B."""

ANNOTATOR_FIELDNAMES = (
    "item_hash",
    "query_title",
    "query_url",
    "candidate_title",
    "candidate_url",
    "relatedness",
    "facet",
)
"""Blind annotator-sheet column order."""

KEY_FIELDNAMES = (
    "item_hash",
    "query_corpusid",
    "candidate_corpusid",
    "method",
    "gold_hit",
    "bucket",
)
"""Internal answer-key column order, for scoring once annotation returns."""


def _load_ranked_candidates(path: Path, depth: int) -> dict[CorpusId, list[CorpusId]]:
    """Load each query's top-`depth` ranked candidates.

    Args:
        path: Retrieval eval JSONL (`{"corpusid", "results": [{"corpusid", ...}]}`).
        depth: Number of top candidates to keep per query.

    Returns:
        Mapping from query corpus ID to its top-`depth` candidate corpus IDs,
        in rank order.
    """
    out: dict[CorpusId, list[CorpusId]] = {}
    for row in iter_jsonl(path):
        qid = CorpusId(int(row["corpusid"]))
        out[qid] = [CorpusId(int(r["corpusid"])) for r in row["results"][:depth]]
    return out


def _item_hash(query_id: CorpusId, method: str, candidate_id: CorpusId) -> str:
    """Stable short hash identifying a sampled (query, method, candidate) triple."""
    digest = hashlib.sha256(f"{query_id}|{method}|{candidate_id}".encode())
    return digest.hexdigest()[:12]


async def _fetch_paper_urls(corpusids: set[CorpusId]) -> dict[CorpusId, str]:
    """Fetch canonical Semantic Scholar URLs for specific papers.

    The Semantic Scholar website's paper URLs are keyed by its internal
    `paperId` hash, not by corpus ID -- constructing a `.../CorpusID:{id}`
    URL locally does not reliably resolve, so the real URL has to come from
    the API.

    Args:
        corpusids: Corpus IDs to look up.

    Returns:
        Mapping from corpus ID to its canonical Semantic Scholar URL. Papers
        the API doesn't return (not found, or a failed batch) are omitted.
    """
    api_key = require_env("SEMANTIC_SCHOLAR_API_KEY")
    ids = list(corpusids)
    urls: dict[CorpusId, str] = {}
    async with SemanticScholarAPI(api_key=api_key) as api:
        for batch in itertools.batched(ids, S2_BATCH_SIZE, strict=False):
            s2_ids = [f"CorpusId:{cid}" for cid in batch]
            try:
                papers = await api.get_paper_batch(s2_ids, fields=["url"])
            except Exception as exc:
                print(f"Warning: URL lookup failed for {len(batch)} papers: {exc}")
                continue
            for cid, paper in zip(batch, papers, strict=True):
                if paper is not None and paper.url:
                    urls[cid] = paper.url
    return urls


def _eligible_queries(
    dense_graph: dict[CorpusId, list[CorpusId]],
    genret: dict[CorpusId, list[CorpusId]],
    gold: dict[int, set[int]],
) -> list[CorpusId]:
    """Queries with >=1 hit and >=1 miss in both methods' candidate lists.

    Guarantees any sampled query can be assigned either a hit or a miss
    target for either method without running out of candidates to draw from.
    """
    eligible: list[CorpusId] = []
    for qid in sorted(dense_graph.keys() & genret.keys()):
        gset = gold.get(int(qid), set())
        has_both = True
        for candidates in (dense_graph[qid], genret[qid]):
            hits = [c for c in candidates if int(c) in gset]
            misses = [c for c in candidates if int(c) not in gset]
            if not hits or not misses:
                has_both = False
                break
        if has_both:
            eligible.append(qid)
    return eligible


def _assign_hit_targets(
    queries: list[CorpusId], rng: random.Random
) -> dict[CorpusId, bool]:
    """Randomly assign a roughly even hit/miss target split across `queries`.

    Returns:
        Mapping from query corpus ID to whether its sampled candidate should
        be a gold hit (`True`) or a miss (`False`).
    """
    shuffled = list(queries)
    rng.shuffle(shuffled)
    n_hit = len(queries) // 2
    return {qid: (i < n_hit) for i, qid in enumerate(shuffled)}


def _assign_buckets(
    queries: list[CorpusId], n_overlap: int, rng: random.Random
) -> dict[CorpusId, str]:
    """Partition queries into a shared core and two exclusive per-annotator sets.

    The core (`n_overlap` queries) is rated by both annotators, for computing
    inter-annotator agreement. The remainder is split roughly evenly between
    the two annotators' exclusive sets, extending total dataset coverage
    without doubling annotator effort on every item.

    Args:
        queries: Sampled query corpus IDs.
        n_overlap: Number of queries to put in the shared core.
        rng: Random number generator to use.

    Returns:
        Mapping from query corpus ID to its bucket (`CORE_BUCKET`,
        `EXCLUSIVE_A_BUCKET`, or `EXCLUSIVE_B_BUCKET`).
    """
    if n_overlap > len(queries):
        raise ValueError(
            f"Requested {n_overlap} overlap queries but only {len(queries)} sampled."
        )
    shuffled = list(queries)
    rng.shuffle(shuffled)
    core = shuffled[:n_overlap]
    remainder = shuffled[n_overlap:]
    half = len(remainder) // 2

    buckets: dict[CorpusId, str] = {}
    for qid in core:
        buckets[qid] = CORE_BUCKET
    for qid in remainder[:half]:
        buckets[qid] = EXCLUSIVE_A_BUCKET
    for qid in remainder[half:]:
        buckets[qid] = EXCLUSIVE_B_BUCKET
    return buckets


def sample_annotation_rows(
    dense_graph_path: Path,
    genret_path: Path,
    paper_graphs_path: Path,
    gold_path: Path,
    n_queries: int,
    n_overlap: int,
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Sample paired dense+graph / genret rows for blind relatedness annotation.

    Args:
        dense_graph_path: Dense+graph baseline eval JSONL.
        genret_path: Generative retriever (genret) eval JSONL.
        paper_graphs_path: Paper-graphs JSONL, for titles.
        gold_path: Strict-gold training-pairs file.
        n_queries: Number of distinct queries to sample (across both annotators).
        n_overlap: Number of those queries rated by both annotators.
        rng: Random number generator to use.

    Returns:
        `(sheet_a, sheet_b, key_rows)` -- the two blind per-annotator sheets
        (each in its own randomised order) and the internal answer key
        joining every row back to its method, gold label, and bucket.
    """
    dense_graph = _load_ranked_candidates(dense_graph_path, SAMPLE_DEPTH)
    genret = _load_ranked_candidates(genret_path, SAMPLE_DEPTH)
    gold = score_glen_eval.load_strict_gold(gold_path)
    titles = load_titles(paper_graphs_path)

    eligible = _eligible_queries(dense_graph, genret, gold)
    if n_queries > len(eligible):
        raise ValueError(
            f"Requested {n_queries} queries but only {len(eligible)} have "
            "both a hit and a miss candidate for both methods."
        )
    sampled_queries = rng.sample(eligible, n_queries)

    dense_graph_targets = _assign_hit_targets(sampled_queries, rng)
    genret_targets = _assign_hit_targets(sampled_queries, rng)
    buckets = _assign_buckets(sampled_queries, n_overlap, rng)

    sheet_a: list[dict[str, str]] = []
    sheet_b: list[dict[str, str]] = []
    key_rows: list[dict[str, str]] = []
    hash_to_ids: dict[str, tuple[CorpusId, CorpusId]] = {}
    for query_id in sampled_queries:
        gset = gold.get(int(query_id), set())
        bucket = buckets[query_id]
        blocks: list[tuple[str, list[CorpusId], bool]] = [
            (DENSE_GRAPH_METHOD, dense_graph[query_id], dense_graph_targets[query_id]),
            (GENRET_METHOD, genret[query_id], genret_targets[query_id]),
        ]
        rng.shuffle(blocks)

        for method, candidates, want_hit in blocks:
            hits = [c for c in candidates if int(c) in gset]
            misses = [c for c in candidates if int(c) not in gset]
            candidate_id = rng.choice(hits if want_hit else misses)
            item_hash = _item_hash(query_id, method, candidate_id)
            hash_to_ids[item_hash] = (query_id, candidate_id)

            annotator_row = {
                "item_hash": item_hash,
                "query_title": titles.get(query_id, ""),
                "query_url": "",
                "candidate_title": titles.get(candidate_id, ""),
                "candidate_url": "",
                "relatedness": "",
                "facet": "",
            }
            if bucket in (CORE_BUCKET, EXCLUSIVE_A_BUCKET):
                sheet_a.append(annotator_row.copy())
            if bucket in (CORE_BUCKET, EXCLUSIVE_B_BUCKET):
                sheet_b.append(annotator_row.copy())

            key_rows.append(
                {
                    "item_hash": item_hash,
                    "query_corpusid": str(query_id),
                    "candidate_corpusid": str(candidate_id),
                    "method": method,
                    "gold_hit": str(int(candidate_id) in gset),
                    "bucket": bucket,
                }
            )

    needed_ids = {cid for pair in hash_to_ids.values() for cid in pair}
    urls = asyncio.run(_fetch_paper_urls(needed_ids))
    for row in itertools.chain(sheet_a, sheet_b):
        query_id, candidate_id = hash_to_ids[row["item_hash"]]
        row["query_url"] = urls.get(query_id, "")
        row["candidate_url"] = urls.get(candidate_id, "")

    rng.shuffle(sheet_a)
    rng.shuffle(sheet_b)
    return sheet_a, sheet_b, key_rows


def _write_csv(
    rows: list[dict[str, str]], fieldnames: tuple[str, ...], output: Path
) -> None:
    """Write rows to a CSV file with the given column order.

    Args:
        rows: Rows to write.
        fieldnames: Column order.
        output: Destination CSV path.
    """
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main(
    dense_graph: Annotated[
        Path, typer.Option(show_default=False, help="Dense+graph baseline eval JSONL.")
    ],
    genret: Annotated[
        Path, typer.Option(show_default=False, help="Genret eval JSONL.")
    ],
    paper_graphs: Annotated[
        Path, typer.Option(show_default=False, help="Paper-graphs JSONL, for titles.")
    ],
    gold: Annotated[
        Path, typer.Option(show_default=False, help="Strict-gold training-pairs file.")
    ],
    n_queries: Annotated[
        int,
        typer.Option(
            help="Number of distinct queries to sample (across both annotators)."
        ),
    ],
    overlap_queries: Annotated[
        int, typer.Option(help="Number of those queries rated by both annotators.")
    ],
    output_a: Annotated[
        Path, typer.Option(show_default=False, help="Blind annotator-A sheet CSV path.")
    ],
    output_b: Annotated[
        Path, typer.Option(show_default=False, help="Blind annotator-B sheet CSV path.")
    ],
    key_output: Annotated[
        Path, typer.Option(show_default=False, help="Internal answer-key CSV path.")
    ],
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
) -> None:
    """Sample paired retrieval results and write both annotator sheets + answer key."""
    rng = random.Random(seed)
    sheet_a, sheet_b, key_rows = sample_annotation_rows(
        dense_graph, genret, paper_graphs, gold, n_queries, overlap_queries, rng
    )

    for label, sheet in (("A", sheet_a), ("B", sheet_b)):
        missing_titles = sum(
            1 for row in sheet if not row["query_title"] or not row["candidate_title"]
        )
        if missing_titles:
            print(
                f"Warning: {missing_titles}/{len(sheet)} rows in sheet {label} "
                "have a missing title."
            )

    _write_csv(sheet_a, ANNOTATOR_FIELDNAMES, output_a)
    _write_csv(sheet_b, ANNOTATOR_FIELDNAMES, output_b)
    _write_csv(key_rows, KEY_FIELDNAMES, key_output)
    print(f"Wrote {len(sheet_a)} rows to annotator sheet A ({output_a})")
    print(f"Wrote {len(sheet_b)} rows to annotator sheet B ({output_b})")
    print(f"Wrote {len(key_rows)} answer-key rows to {key_output}")


if __name__ == "__main__":
    typer.run(main)
