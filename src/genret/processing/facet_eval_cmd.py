"""Facet evaluation pipeline: score extracted facets against paper text.

Loads paper graphs and facet records, computes quality metrics via embeddings,
and writes per-paper evaluations and an aggregate summary.
"""

import asyncio
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import aiofiles
from pydantic import ValidationError
from tqdm import tqdm

from genret.llm.encoder import DEFAULT_EMBEDDING_MODEL, make_encoder
from genret.paper_parsing import (
    FacetRecordType,
    Facets,
    PaperGraph,
    evidence_from_type,
    facets_from_type,
)
from genret.processing.facet_evaluation import (
    EVIDENCE_MIN_MATCH_BLOCK,
    EmbeddingCache,
    PaperEvaluation,
    compute_summary,
    evaluate_paper,
)
from genret.utils import (
    AsyncLineWriter,
    count_lines,
    die,
    iter_jsonl,
    open_auto,
    resolve_jsonl,
    sample,
    track_progress,
)

logger = logging.getLogger(__name__)


async def run_evaluate_facets(
    splits_dir: Path,
    *,
    facets_input: Path | None = None,
    facets_type: FacetRecordType,
    sample_k: int = 0,
    seed: int = 0,
    concurrent: int = 100,
    grounding_top_k: int = 3,
    evidence_min_match_block: int = EVIDENCE_MIN_MATCH_BLOCK,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    grounding: bool = False,
    force: bool = False,
    api_key: str | None = None,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
) -> None:
    """Run facet evaluation pipeline.

    Args:
        splits_dir: Splits directory with processed/ subfolder.
        facets_input: Override path to the facets file.  Outputs are written to the same
            directory as this file. Defaults to processed/facets.jsonl.gz.
        facets_type: Facet record schema in the input file.
        sample_k: Random sample size (0 = all papers).
        seed: Random seed for sampling.
        concurrent: Concurrent API calls for embedding.
        grounding_top_k: Evidence snippets per facet item.
        evidence_min_match_block: Minimum matching block size counted for
            evidence fidelity coverage.
        embedding_model: OpenAI embedding model name.
        grounding: If True, compute grounding scores against paper paragraphs.
            Requires paper_graphs to be present. Disabled by default due to
            high memory and API cost.
        force: If True, recompute embedding caches.
        api_key: API key.  None falls back to environment variable.
        azure_endpoint: Azure endpoint URL.  When set, uses Azure client.
        api_version: Azure API version.
    """
    processed_dir = splits_dir / "processed"
    facets_path = resolve_jsonl(facets_input or processed_dir / "facets.jsonl.gz")
    output_dir = facets_path.parent

    model_tag = embedding_model.replace("/", "-").replace(":", "-")
    output_path = output_dir / f"facets_evaluation_{model_tag}.jsonl.zst"
    summary_path = output_dir / f"facets_evaluation_summary_{model_tag}.json"
    para_cache_path = processed_dir / f"paragraph_embeddings_{model_tag}.npz"
    facet_cache_path = output_dir / f"facet_embeddings_{model_tag}.npz"

    if not facets_path.exists():
        die(f"Facets not found: {facets_path}\nRun 'extract' first.")

    logger.info("=== Evaluating Facet Quality ===")

    # Handle embedding cache
    if force:
        facet_cache_path.unlink(missing_ok=True)
        if grounding:
            para_cache_path.unlink(missing_ok=True)
        logger.info("Cleared embedding caches")

    # Load paper graphs into index (only needed for grounding)
    papers_index: dict[int, PaperGraph] = {}

    if grounding:
        graphs_path = resolve_jsonl(processed_dir / "paper_graphs.jsonl.gz")
        if not graphs_path.exists():
            die(f"Paper graphs not found: {graphs_path}\nRun 'parse' first.")
        with open_auto(graphs_path, "r") as f:
            for line in track_progress(
                f, desc="Loading paper graphs...", total=count_lines(graphs_path)
            ):
                if not line.strip():
                    continue
                try:
                    pg = PaperGraph.model_validate_json(line)
                    papers_index[pg.corpusid] = pg
                except ValidationError:
                    logger.exception("Invalid paper graph")
        logger.info(f"Loaded {len(papers_index)} paper graphs")

    # Load and validate facets records
    facets_records: list[tuple[Facets, dict[str, list[list[str]]] | None]] = []
    skip_reasons: defaultdict[str, int] = defaultdict(int)
    papers_in_file = 0

    for record in track_progress(iter_jsonl(facets_path), desc="Loading facets..."):
        papers_in_file += 1

        try:
            facet = facets_from_type(record, facets_type)
            evidence = evidence_from_type(record, facets_type)
        except ValidationError:
            skip_reasons["invalid_facets"] += 1
            logger.exception("Invalid facets record")
            continue

        if grounding and facet.corpusid not in papers_index:
            skip_reasons["no_paper_graph"] += 1
            continue

        facets_records.append((facet, evidence))

    # Sample if requested
    if sample_k > 0:
        rng = random.Random(seed)
        facets_records = sample(rng, facets_records, sample_k)
        logger.info(f"Sampled {sample_k} papers (seed={seed})")

    logger.info(f"Papers to evaluate: {len(facets_records)}")
    if not facets_records:
        logger.info("Nothing to evaluate.")
        return

    # Set up encoder and concurrency
    semaphore = asyncio.Semaphore(concurrent)
    write_lock = asyncio.Lock()
    evaluations: list[PaperEvaluation] = []
    start_time = time.time()

    pbar = tqdm(total=len(facets_records), desc="Evaluating")
    tmp_output = output_path.with_name("facets_evaluation.tmp.jsonl.zst")

    async with (
        make_encoder(
            embedding_model,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        ) as encoder,
        AsyncLineWriter.open(tmp_output, "w") as out_writer,
    ):

        async def run_all(para_cache: EmbeddingCache | None) -> None:
            """Run all evaluation tasks with the given para_cache."""
            with EmbeddingCache.open(
                facet_cache_path, encoder.dimensions
            ) as facet_cache:

                async def process_one(
                    facets: Facets,
                    ev: dict[str, list[list[str]]] | None,
                ) -> None:
                    """Evaluate a single paper's facets."""
                    async with semaphore:
                        try:
                            paper = (
                                papers_index.get(facets.corpusid) if grounding else None
                            )
                            result = await evaluate_paper(
                                facets,
                                paper,
                                encoder,
                                para_cache,
                                facet_cache,
                                top_k=grounding_top_k,
                                facet_evidence=ev,
                                evidence_min_match_block=evidence_min_match_block,
                            )
                            async with write_lock:
                                await out_writer.write_obj(result, flush=True)
                                evaluations.append(result)
                        except Exception:
                            logger.exception("Failed to evaluate %d", facets.corpusid)
                            skip_reasons["evaluation_error"] += 1
                        finally:
                            pbar.update(1)

                tasks = [
                    asyncio.create_task(process_one(f, ev)) for f, ev in facets_records
                ]
                await asyncio.gather(*tasks)

        if grounding:
            with EmbeddingCache.open(para_cache_path, encoder.dimensions) as para_cache:
                await run_all(para_cache)
        else:
            await run_all(None)

        embedding_cost = encoder.total_cost

    pbar.close()

    logger.info(f"Embedding cost: ${embedding_cost:.4f}")

    # Move temp file to final output (only if we evaluated anything)
    if evaluations:
        tmp_output.rename(output_path)
    else:
        tmp_output.unlink(missing_ok=True)

    runtime = time.time() - start_time
    logger.info(f"\nEvaluated {len(evaluations)} papers in {runtime:.1f}s")
    if evaluations:
        logger.info(f"Per-paper results: {output_path}")

    # Compute and write summary
    config = {
        "sample": sample_k,
        "seed": seed,
        "concurrent": concurrent,
        "facets_type": facets_type.value,
        "grounding_top_k": grounding_top_k,
        "evidence_min_match_block": evidence_min_match_block,
        "embedding_model": embedding_model,
        "grounding": grounding,
    }
    summary = compute_summary(
        evaluations, papers_in_file, skip_reasons, config, runtime
    )
    async with aiofiles.open(summary_path, "w", encoding="utf-8") as af:
        await af.write(summary.model_dump_json(indent=2) + "\n")

    logger.info(f"Summary: {summary_path}")

    # Print key metrics
    if summary.score_distributions:
        score_lines: list[str] = []
        for name in ("structure", "grounding", "redundancy", "specificity", "overall"):
            dist = summary.score_distributions.get(name)
            if dist:
                score_lines.append(
                    f"  {name:14s}: mean={dist.mean:.3f}  "
                    f"median={dist.median:.3f}  p10={dist.p10:.3f}  p90={dist.p90:.3f}"
                )
        logger.info("\n--- Score Summary ---\n%s", "\n".join(score_lines))

    if summary.top_flags:
        flag_lines = "\n".join(
            f"  {flag}: {count}" for flag, count in list(summary.top_flags.items())[:10]
        )
        logger.info("\n--- Top Flags ---\n%s", flag_lines)

    if summary.evidence_fidelity is not None:
        # Compute verbatim/faithful counts for display
        evidence_items = [
            item
            for ev in evaluations
            for items in ev.item_metrics.values()
            for item in items
            if item.evidence_fidelity is not None
        ]
        total_verbatim = sum(item.evidence_verbatim_count for item in evidence_items)
        total_faithful = sum(item.evidence_faithful_count for item in evidence_items)
        total_evidence = sum(len(item.model_evidence) for item in evidence_items)

        ef = summary.evidence_fidelity
        vf = summary.evidence_verbatim_fraction or 0.0
        ff = summary.evidence_faithful_fraction or 0.0
        logger.info(
            "\n--- Evidence Fidelity ---\n"
            "  fidelity:       mean=%.3f  median=%.3f  p10=%.3f  p90=%.3f\n"
            "  faithful:       %.1f%% (%d/%d)\n"
            "  verbatim:       %.1f%% (%d/%d)",
            ef.mean,
            ef.median,
            ef.p10,
            ef.p90,
            ff * 100,
            total_faithful,
            total_evidence,
            vf * 100,
            total_verbatim,
            total_evidence,
        )
