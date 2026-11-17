"""CLI for processing S2ORC papers into structured graphs and extracting facets.

Commands:
- parse: Parse S2ORC fulltext into paper graphs
- extract: Extract facets using LLM client (async, resumable)
- ground-truth: Compile citation data into training pairs
- sample: Sample papers for manual review
- evaluate-facets: Evaluate facet extraction quality
- all: Run full pipeline

Graph pipeline commands (registered from `graph_cli.py`):
- build-facet-graph, build-candidate-graph, build-edges
- retrieve, batch-retrieve
- train-reranker, apply-reranker
"""

import asyncio
import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from genret import llm
from genret.llm.encoder import DEFAULT_EMBEDDING_MODEL
from genret.paper_parsing import (
    EnrichedPaper,
    FacetRecordType,
    FacetsSummary,
    PaperGraph,
    RelatedPaperInfo,
    RelationType,
    ScoredPaperPair,
    TrainingPair,
    facets_from_type,
    parse_paper,
)
from genret.processing.facet_eval_cmd import run_evaluate_facets
from genret.processing.facet_evaluation import EVIDENCE_MIN_MATCH_BLOCK
from genret.processing.facet_extract_cmd import (
    FACET_PROMPTS,
    FacetPromptStyle,
    SectionMode,
    run_extract,
)
from genret.processing.graph_cli import register_graph_commands
from genret.types import CorpusId
from genret.utils import (
    backup_files,
    confirm_force,
    die,
    glob_jsonl,
    iter_jsonl,
    load_references,
    load_split_ids,
    make_app,
    open_auto,
    resolve_jsonl,
    sample,
    track_progress,
)

logger = logging.getLogger(__name__)

app = make_app("Process S2ORC papers into structured graphs and extract facets")


def iter_gzip_jsonl(directory: Path) -> Iterator[dict[str, Any]]:
    """Iterate over records from all compressed JSONL files in a directory."""
    for gz_file in glob_jsonl(directory):
        yield from iter_jsonl(gz_file)


def load_scored_pairs(path: Path, score_key: str) -> list[ScoredPaperPair]:
    """Load scored paper pairs from a JSONL file.

    Args:
        path: Path to the JSONL file.
        score_key: Key name for the score field (e.g. "count", "shared_refs").

    Returns:
        List of ScoredPaperPair objects, or empty list if file doesn't exist.
    """
    if not path.exists():
        return []
    return [
        ScoredPaperPair(paper_a=r["paper_a"], paper_b=r["paper_b"], score=r[score_key])
        for r in iter_jsonl(path)
    ]


def collect_reference_pairs(
    refs: dict[CorpusId, list[CorpusId]],
    valid_ids: set[CorpusId],
) -> list[TrainingPair]:
    """Collect reference pairs where both papers are in valid_ids.

    Args:
        refs: Map of paper_id -> list of referenced paper IDs.
        valid_ids: Set of valid paper IDs to include.

    Returns:
        List of TrainingPair objects for valid reference relationships.
    """
    pairs: list[TrainingPair] = []
    for paper_id, ref_ids in refs.items():
        if paper_id in valid_ids:
            for ref_id in ref_ids:
                if ref_id in valid_ids:
                    pairs.append(
                        TrainingPair(
                            paper_a=paper_id,
                            paper_b=ref_id,
                            relation=RelationType.REFERENCE,
                            score=1.0,
                        )
                    )
    return pairs


def collect_scored_training_pairs(
    scored_pairs: list[ScoredPaperPair],
    valid_ids: set[CorpusId],
    relation: RelationType,
) -> list[TrainingPair]:
    """Collect training pairs from scored paper pairs, filtering by valid IDs.

    Args:
        scored_pairs: List of ScoredPaperPair objects.
        valid_ids: Set of valid paper IDs to include.
        relation: The relationship type to assign.

    Returns:
        List of TrainingPair objects where both papers are in valid_ids.
    """
    return [
        TrainingPair(
            paper_a=pair.paper_a,
            paper_b=pair.paper_b,
            relation=relation,
            score=float(pair.score),
        )
        for pair in scored_pairs
        if pair.paper_a in valid_ids and pair.paper_b in valid_ids
    ]


def build_pairs_index(
    pairs: list[TrainingPair],
    titles: dict[CorpusId, str],
) -> dict[CorpusId, list[RelatedPaperInfo]]:
    """Build an index of training pairs by paper ID.

    For each pair, adds an entry to both papers' lists with the related
    paper's information.

    Args:
        pairs: List of TrainingPair objects.
        titles: Map of corpus_id -> paper title.

    Returns:
        Map of corpus_id -> list of RelatedPaperInfo objects.
    """
    pairs_by_paper: dict[CorpusId, list[RelatedPaperInfo]] = {}
    for pair in pairs:
        for pid, other_pid in [
            (pair.paper_a, pair.paper_b),
            (pair.paper_b, pair.paper_a),
        ]:
            if pid not in pairs_by_paper:
                pairs_by_paper[pid] = []
            pairs_by_paper[pid].append(
                RelatedPaperInfo(
                    related_paper_id=other_pid,
                    related_paper_title=titles.get(other_pid, "Unknown"),
                    relation=pair.relation,
                    score=pair.score,
                )
            )
    return pairs_by_paper


def parse_facets_record(
    record: dict[str, Any],
    *,
    facets_type: FacetRecordType,
) -> FacetsSummary:
    """Parse a facets record into a FacetsSummary.

    Args:
        record: Raw facets record from JSONL.
        facets_type: On-disk record schema used in facets JSONL.

    Returns:
        FacetsSummary with problems, methods, results, contributions.
    """
    facets = facets_from_type(record, facets_type)
    return FacetsSummary(
        problems=facets.problems,
        methods=facets.methods,
        results=facets.results,
        contributions=facets.contributions,
    )


def enrich_paper_with_metadata(
    paper: PaperGraph,
    facets: dict[CorpusId, FacetsSummary],
    pairs_by_paper: dict[CorpusId, list[RelatedPaperInfo]],
) -> EnrichedPaper:
    """Enrich a paper with facets and ground truth pairs.

    Args:
        paper: PaperGraph to enrich.
        facets: Map of corpus_id -> FacetsSummary.
        pairs_by_paper: Map of corpus_id -> list of RelatedPaperInfo.

    Returns:
        EnrichedPaper with facets and ground_truth_pairs added.
    """
    return EnrichedPaper(
        corpusid=paper.corpusid,
        title=paper.title,
        authors=paper.authors,
        sections=paper.sections,
        abstract=paper.abstract,
        facets=facets.get(paper.corpusid),
        ground_truth_pairs=pairs_by_paper.get(paper.corpusid, []),
    )


@app.command(name="parse")
def cmd_parse(
    data_dir: Annotated[Path, typer.Argument(help="Data directory.")],
    splits_dir: Annotated[
        Path, typer.Argument(help="Splits directory with train/dev/test.txt")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory.")],
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Max papers to process (0 = all).")
    ] = 0,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing output.")
    ] = False,
) -> None:
    """Parse S2ORC fulltext into paper graphs.

    Reads from data_dir/s2orc/ and writes to splits_dir/processed/.
    Only parses papers in splits.
    """
    if force:
        confirm_force()

    if not splits_dir.exists():
        die(f"Splits directory not found: {splits_dir}")

    split_ids = load_split_ids(splits_dir)
    if not split_ids:
        die(f"No IDs found in {splits_dir}/{{train,dev,test}}.txt")

    fulltext_path = data_dir / "s2orc"
    if not fulltext_path.exists() or not glob_jsonl(fulltext_path):
        die(
            f"Fulltext not found: expected {data_dir}/s2orc/*.jsonl.gz or *.jsonl.zst files"
        )

    processed_dir = output
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "paper_graphs.jsonl.zst"
    if output_path.exists() and not force:
        logger.info(
            f"Skipping parse: {output_path} already exists.\nUse --force to override."
        )
        return
    if output_path.exists() and force:
        backup_files([output_path])

    # Load abstracts from metadata if available
    abstracts: dict[int, str] = {}
    metadata_dir = data_dir / "papers"
    if metadata_dir.exists():
        for record in track_progress(
            iter_gzip_jsonl(metadata_dir), "Loading metadata abstracts..."
        ):
            cid = record.get("corpusid")
            abstract = record.get("abstract")
            if cid is not None and abstract:
                abstracts[cid] = abstract

        if abstracts:
            logger.info(f"Loaded {len(abstracts)} abstracts from metadata")

    logger.info("=== Parsing S2ORC Papers ===")
    logger.info(f"Input: {fulltext_path}")
    logger.info(f"Splits: {splits_dir} ({len(split_ids)} papers)")
    if limit > 0:
        logger.info(f"Limit: {limit}")
    logger.info(f"Output: {output_path}")

    parsed_count = 0
    failed_count = 0
    skipped_count = 0

    with open_auto(output_path, "w") as out:
        for record in track_progress(iter_gzip_jsonl(fulltext_path), "Parsing..."):
            corpus_id = record.get("corpusid")
            if corpus_id not in split_ids:
                skipped_count += 1
                continue

            abstract = abstracts.get(corpus_id) if corpus_id is not None else None

            if paper := parse_paper(record, abstract=abstract):
                out.write(paper.model_dump_json() + "\n")
                parsed_count += 1
                if limit > 0 and parsed_count >= limit:
                    break
            else:
                failed_count += 1

    logger.info(
        f"\nParsing complete:\n"
        f"  Parsed: {parsed_count}\n"
        f"  Failed: {failed_count}\n"
        f"  Skipped: {skipped_count}\n"
        f"  Output: {output_path}"
    )


def _list_prompts_callback(value: bool) -> None:
    """Print available prompt names and exit when --list-prompts is passed."""
    if value:
        table = Table(title="Available extraction prompts")
        table.add_column("Name", header_style="green", style="bold green")
        table.add_column("Description")
        table.add_column("No. Tokens", header_style="blue", style="blue")

        for name, prompt in FACET_PROMPTS.items():
            tokens = len(llm.tokenise(prompt.template))
            table.add_row(name, prompt.description, str(tokens))

        Console().print(table)
        raise typer.Exit


@app.command(name="extract")
def cmd_extract(
    splits_dir: Annotated[
        Path, typer.Argument(help="Splits directory with train/dev/test.txt")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory.")],
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Max papers to process (0 = all).")
    ] = 0,
    concurrent: Annotated[
        int, typer.Option("--concurrent", "-c", help="Concurrent API calls.")
    ] = 100,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model to use.")
    ] = "gpt-4o-mini",
    reasoning_effort: Annotated[
        llm.ReasoningEffort,
        typer.Option(
            "--reasoning-effort",
            help=(
                "Reasoning effort for GPT-5 models: default, none, minimal, "
                "low, medium, high, xhigh."
            ),
        ),
    ] = llm.ReasoningEffort.MINIMAL,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Reprocess all papers.")
    ] = False,
    prompt: Annotated[
        FacetPromptStyle,
        typer.Option(
            "--prompt",
            help="Facet extraction prompt/schema variant.",
        ),
    ] = FacetPromptStyle.DEFAULT,
    provider: Annotated[
        str,
        typer.Option("--provider", help="LLM provider: 'openai' or 'azure-openai'."),
    ] = "openai",
    azure_endpoint: Annotated[
        str | None,
        typer.Option("--azure-endpoint", help="Azure OpenAI endpoint URL."),
    ] = None,
    api_version: Annotated[
        str | None,
        typer.Option("--api-version", help="Azure API version override."),
    ] = None,
    section_mode: Annotated[
        SectionMode,
        typer.Option(
            "--section-mode",
            help="Paper text fed to the model: 'full' (whole paper) or 'core' "
            "(title + abstract + facet-relevant sections only).",
        ),
    ] = SectionMode.FULL,
    max_input_tokens: Annotated[
        int,
        typer.Option("--max-input-tokens", help="Truncate paper text to N tokens."),
    ] = 100000,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Skip API calls; return empty results."),
    ] = False,
    _: Annotated[
        bool,
        typer.Option(
            "--list-prompts",
            callback=_list_prompts_callback,
            is_eager=True,
            help="List available prompt names and exit.",
        ),
    ] = False,
) -> None:
    """Extract facets using LLM API (async, resumable).

    Supports OpenAI and Azure OpenAI providers. Requires OPENAI_API_KEY or
    AZURE_OPENAI_API_KEY environment variable depending on provider.
    """
    if force:
        confirm_force()

    if not splits_dir.exists():
        die(f"Splits directory not found: {splits_dir}")

    if dry_run:
        logger.warning("Dry-run mode: no API calls will be made, results will be empty")
        provider = "dry-run"

    filter_ids = load_split_ids(splits_dir)
    if not filter_ids:
        die(f"No IDs found in {splits_dir}/{{train,dev,test}}.txt")

    logger.info(
        f"=== Extracting Facets ===\nSplits: {splits_dir}\nPapers: {len(filter_ids)}"
    )
    asyncio.run(
        run_extract(
            splits_dir,
            filter_ids,
            output,
            concurrent=concurrent,
            model=model,
            reasoning_effort=reasoning_effort,
            force=force,
            prompt_style=prompt,
            provider=provider,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            section_mode=section_mode,
            max_input_tokens=max_input_tokens,
            limit=limit,
        )
    )


@app.command(name="extract-local")
def cmd_extract_local(
    splits_dir: Annotated[
        Path, typer.Argument(help="Splits directory with train/dev/test.txt")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory.")],
    model: Annotated[
        str, typer.Option("--model", "-m", help="HuggingFace model ID.")
    ] = "Qwen/Qwen3-32B",
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Max papers to process (0 = all).")
    ] = 0,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Papers per generation batch.")
    ] = 8,
    max_input_tokens: Annotated[
        int, typer.Option("--max-input-tokens", help="Truncate paper text to N tokens.")
    ] = 16000,
    max_new_tokens: Annotated[
        int, typer.Option("--max-new-tokens", help="Generation budget per paper.")
    ] = 768,
    section_mode: Annotated[
        SectionMode,
        typer.Option(
            "--section-mode",
            help="Paper text fed to the model: 'full' or 'core' (title + "
            "abstract + facet-relevant sections only).",
        ),
    ] = SectionMode.CORE,
    quantize_4bit: Annotated[
        bool, typer.Option("--4bit", help="Load model in 4-bit NF4.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Reprocess all papers.")
    ] = False,
) -> None:
    """Extract facets with a locally-loaded HuggingFace model (e.g. Qwen3-32B).

    Generation-based alternative to ``extract`` that needs no API. Writes the
    same ``facets.jsonl.zst`` schema, so ``evaluate-facets`` works unchanged.
    """
    # Imported lazily: pulls in torch/transformers (cuda/training extras only).
    from genret.processing.local_facet_extract import run_local_extract  # noqa: PLC0415

    if force:
        confirm_force()
    if not splits_dir.exists():
        die(f"Splits directory not found: {splits_dir}")

    filter_ids = load_split_ids(splits_dir)
    if not filter_ids:
        die(f"No IDs found in {splits_dir}/{{train,dev,test}}.txt")

    logger.info(
        f"=== Extracting Facets (local) ===\n"
        f"Splits: {splits_dir}\nPapers: {len(filter_ids)}\nModel: {model}"
    )
    run_local_extract(
        splits_dir,
        filter_ids,
        output,
        model_id=model,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        section_mode=section_mode,
        quantize_4bit=quantize_4bit,
        force=force,
        limit=limit,
    )


@app.command(name="ground-truth")
def cmd_ground_truth(
    data_dir: Annotated[Path, typer.Argument(help="Data directory.")],
    splits_dir: Annotated[
        Path, typer.Argument(help="Splits directory with train/dev/test.txt")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory.")],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing output.")
    ] = False,
) -> None:
    """Compile citation data into training pairs.

    Creates training pairs from:
    - Direct references
    - Co-citations
    - Bibliographic coupling

    Only includes pairs where both papers are in the splits.
    """
    if force:
        confirm_force()

    if not splits_dir.exists():
        die(f"Splits directory not found: {splits_dir}")

    valid_ids = load_split_ids(splits_dir)
    if not valid_ids:
        die(f"No IDs found in {splits_dir}/{{train,dev,test}}.txt")

    citations_dir = data_dir / "citations"
    processed_dir = output
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "training_pairs.jsonl.zst"
    if output_path.exists() and not force:
        logger.info(
            f"Skipping ground-truth: {output_path} already exists.\n"
            "Use --force to override."
        )
        return
    if output_path.exists() and force:
        backup_files([output_path])

    logger.info(
        f"=== Compiling Training Pairs ===\n"
        f"Splits: {splits_dir} ({len(valid_ids)} papers)"
    )

    pairs: list[TrainingPair] = []

    refs_path = resolve_jsonl(citations_dir / "references.jsonl.gz")
    if refs_path.exists():
        refs = load_references(refs_path)
        ref_pairs = collect_reference_pairs(refs, valid_ids)
        pairs.extend(ref_pairs)
        logger.info(f"  References: {len(ref_pairs)} pairs")

    co_cite_path = resolve_jsonl(citations_dir / "co_citations.jsonl.gz")
    if co_cite_path.exists():
        co_cites = load_scored_pairs(co_cite_path, "count")
        co_cite_pairs = collect_scored_training_pairs(
            co_cites, valid_ids, RelationType.CO_CITATION
        )
        pairs.extend(co_cite_pairs)
        logger.info(f"  Co-citations: {len(co_cite_pairs)} pairs")

    bib_path = resolve_jsonl(citations_dir / "bib_coupling.jsonl.gz")
    if bib_path.exists():
        bib_couples = load_scored_pairs(bib_path, "shared_refs")
        bib_pairs = collect_scored_training_pairs(
            bib_couples, valid_ids, RelationType.BIB_COUPLING
        )
        pairs.extend(bib_pairs)
        logger.info(f"  Bib coupling: {len(bib_pairs)} pairs")

    with open_auto(output_path, "w") as f:
        f.writelines(pair.model_dump_json() + "\n" for pair in pairs)

    logger.info(f"\nTotal pairs: {len(pairs)}\nOutput: {output_path}")


@app.command(name="sample")
def cmd_sample(
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory.")],
    sample_k: Annotated[
        int, typer.Option("--sample", "-n", help="Number of papers to sample.")
    ] = 10,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed.")] = 42,
    facets_type: Annotated[
        FacetRecordType,
        typer.Option(
            "--facets-type",
            help="Facet record schema in facets.jsonl.gz: standard, evidence, or evidence-list.",
        ),
    ] = FacetRecordType.STANDARD,
) -> None:
    """Sample papers for manual review.

    Creates a JSON file with sampled papers, facets, and ground truth pairs.
    """
    processed_dir = output
    graphs_path = resolve_jsonl(processed_dir / "paper_graphs.jsonl.gz")
    facets_path = resolve_jsonl(processed_dir / "facets.jsonl.gz")
    pairs_path = resolve_jsonl(processed_dir / "training_pairs.jsonl.gz")

    if not graphs_path.exists():
        die(f"Paper graphs not found: {graphs_path}\nRun 'parse' first.")

    logger.info("=== Sampling Papers for Validation ===")

    papers = [PaperGraph.model_validate(p) for p in iter_jsonl(graphs_path)]
    titles = {p.corpusid: p.title for p in papers}
    logger.info(f"Total papers: {len(papers)}")

    facets: dict[CorpusId, FacetsSummary] = {}
    if facets_path.exists():
        for record in iter_jsonl(facets_path):
            facets[CorpusId(record["corpusid"])] = parse_facets_record(
                record, facets_type=facets_type
            )
        logger.info(f"Loaded facets: {len(facets)}")

    pairs_by_paper: dict[CorpusId, list[RelatedPaperInfo]] = {}
    if pairs_path.exists():
        pairs_list = [TrainingPair.model_validate(p) for p in iter_jsonl(pairs_path)]
        pairs_by_paper = build_pairs_index(pairs_list, titles)
        logger.info(f"Loaded training pairs for {len(pairs_by_paper)} papers")

    rng = random.Random(seed)
    sampled = sample(rng, papers, sample_k)

    enriched = [
        enrich_paper_with_metadata(paper, facets, pairs_by_paper) for paper in sampled
    ]

    output_path = processed_dir / "validation_sample.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in enriched], f, indent=2)

    logger.info(f"Sampled: {len(sampled)} papers")
    logger.info(f"Output: {output_path}")


@app.command(name="all")
def cmd_all(
    data_dir: Annotated[Path, typer.Argument(help="Data directory.")],
    splits_dir: Annotated[
        Path, typer.Argument(help="Splits directory with train/dev/test.txt")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory.")],
    limit: Annotated[
        int, typer.Option("--limit", help="Max papers to process (0 = all).")
    ] = 0,
    concurrent: Annotated[
        int, typer.Option("--concurrent", "-c", help="Concurrent API calls.")
    ] = 100,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model to use.")
    ] = "gpt-4o-mini",
    reasoning_effort: Annotated[
        llm.ReasoningEffort,
        typer.Option(
            "--reasoning-effort",
            help=(
                "Reasoning effort for GPT-5 models: default, none, minimal, "
                "low, medium, high, xhigh."
            ),
        ),
    ] = llm.ReasoningEffort.MINIMAL,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing output.")
    ] = False,
    prompt: Annotated[
        FacetPromptStyle,
        typer.Option(
            "--prompt",
            help="Facet extraction prompt/schema variant.",
        ),
    ] = FacetPromptStyle.DEFAULT,
    skip_extract: Annotated[
        bool, typer.Option("--skip-extract", help="Skip facet extraction.")
    ] = False,
    sample_k: Annotated[
        int, typer.Option("--sample", "-n", help="Validation sample size.")
    ] = 10,
    provider: Annotated[
        str,
        typer.Option("--provider", help="LLM provider: 'openai' or 'azure-openai'."),
    ] = "openai",
    azure_endpoint: Annotated[
        str | None,
        typer.Option("--azure-endpoint", help="Azure OpenAI endpoint URL."),
    ] = None,
    api_version: Annotated[
        str | None,
        typer.Option("--api-version", help="Azure API version override."),
    ] = None,
) -> None:
    """Run full pipeline: parse, extract, ground-truth, sample.

    Set --skip-extract to skip the LLM API calls.
    """
    if force:
        confirm_force()

    logger.info("=== Full Pipeline ===")

    logger.info("\n--- Phase 1: Parse ---")
    cmd_parse(data_dir, splits_dir, limit=limit, output=output, force=force)

    if not skip_extract:
        logger.info("\n--- Phase 2: Extract Facets ---")
        cmd_extract(
            splits_dir,
            limit=limit,
            output=output,
            concurrent=concurrent,
            model=model,
            reasoning_effort=reasoning_effort,
            force=force,
            prompt=prompt,
            provider=provider,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    logger.info("\n--- Phase 3: Ground Truth ---")
    cmd_ground_truth(data_dir, splits_dir, output=output, force=force)

    logger.info("\n--- Phase 4: Validation Sample ---")
    cmd_sample(
        sample_k=sample_k,
        output=output,
        facets_type=prompt.to_type(),
    )

    logger.info("\n=== Pipeline Complete ===")


@app.command(name="evaluate-facets")
def cmd_evaluate_facets(
    splits_dir: Annotated[
        Path, typer.Argument(help="Splits directory with processed/ subfolder.")
    ],
    facets_input: Annotated[
        Path | None,
        typer.Option(
            "--input",
            "-i",
            help="Facets file to evaluate. Defaults to processed/facets.jsonl.gz.",
        ),
    ] = None,
    facets_type: Annotated[
        FacetRecordType,
        typer.Option(
            "--type",
            help="Facet record schema in input file: standard, evidence, or evidence-list.",
        ),
    ] = FacetRecordType.STANDARD,
    sample_k: Annotated[
        int,
        typer.Option("--sample", "-n", help="Random sample size (0 = all papers)."),
    ] = 0,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed.")] = 0,
    concurrent: Annotated[
        int,
        typer.Option("--concurrent", "-c", help="Concurrent API calls."),
    ] = 100,
    grounding_top_k: Annotated[
        int,
        typer.Option("--grounding-top-k", help="Evidence snippets per facet item."),
    ] = 3,
    evidence_min_match_block: Annotated[
        int,
        typer.Option(
            "--evidence-min-match-block",
            help="Minimum matching block size for evidence fidelity coverage.",
            min=1,
        ),
    ] = EVIDENCE_MIN_MATCH_BLOCK,
    embedding_model: Annotated[
        str,
        typer.Option(
            "--embedding-model",
            help="Embedding model: local MiniLM by default; pass a "
            "'text-embedding-*' name to use OpenAI.",
        ),
    ] = DEFAULT_EMBEDDING_MODEL,
    grounding: Annotated[
        bool,
        typer.Option(
            "--grounding/--no-grounding",
            help="Compute grounding scores against paper paragraphs. "
            "Requires paper_graphs. Disabled by default (high memory and API cost).",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Recompute embedding caches."),
    ] = False,
    azure_endpoint: Annotated[
        str | None,
        typer.Option("--azure-endpoint", help="Azure OpenAI endpoint URL."),
    ] = None,
    api_version: Annotated[
        str | None,
        typer.Option("--api-version", help="Azure API version override."),
    ] = None,
) -> None:
    """Evaluate facet extraction quality (read-only, no gating)."""
    if force:
        confirm_force()

    asyncio.run(
        run_evaluate_facets(
            splits_dir,
            facets_input=facets_input,
            facets_type=facets_type,
            sample_k=sample_k,
            seed=seed,
            concurrent=concurrent,
            grounding_top_k=grounding_top_k,
            evidence_min_match_block=evidence_min_match_block,
            embedding_model=embedding_model,
            grounding=grounding,
            force=force,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
    )


register_graph_commands(app)


if __name__ == "__main__":
    app()
