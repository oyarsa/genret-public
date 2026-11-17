"""LLM-based discriminative keyphrase extraction from papers.

Extracts paper-level and aspect-level keyphrases using an LLM client, reading from paper
graphs and facets to produce structured term sets suitable for DocID encoding.
"""

import asyncio
import itertools
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, Field

from genret.llm.client import CompletionRequest, LLMClient, ReasoningEffort
from genret.llm.models import resolve_model, write_run_metadata
from genret.paper_parsing import (
    AspectTerms,
    FacetEvidence,
    FacetRecordType,
    FacetsSummary,
    PaperGraph,
    PaperTerms,
    Section,
    SectionType,
    evidence_from_type,
    facets_from_type,
)
from genret.processing.facet_extract_cmd import (
    append_processed_id,
    iter_paper_graphs,
    prepare_extract_state,
)
from genret.types import CorpusId
from genret.utils import (
    AsyncLineWriter,
    confirm_force,
    countit,
    die,
    iter_jsonl,
    load_split_ids,
    make_app,
    resolve_jsonl,
    track_progress,
)

logger = logging.getLogger(__name__)

app = make_app("Extract discriminative keyphrases from papers")


def extract_abstract_text(paper: PaperGraph) -> str | None:
    """Pull abstract text from a paper's section tree.

    Args:
        paper: The paper graph to search.

    Returns:
        Combined abstract paragraph text, or None if no abstract section exists.
    """

    def _find_abstract(sections: Sequence[Section]) -> str | None:
        """Recursively search for abstract section."""
        for section in sections:
            if section.type == SectionType.ABSTRACT:
                text = "\n\n".join(section.paragraphs)
                return text if text.strip() else None

            if result := _find_abstract(section.subsections):
                return result

        return None

    return _find_abstract(paper.sections)


def extract_introduction_text(paper: PaperGraph) -> str | None:
    """Pull introduction text from a paper's section tree.

    Args:
        paper: The paper graph to search.

    Returns:
        Combined introduction paragraph text, or None if no introduction
        section exists.
    """

    def _find_introduction(sections: Sequence[Section]) -> str | None:
        """Recursively search for introduction section."""
        for section in sections:
            if section.type == SectionType.INTRODUCTION:
                text = "\n\n".join(section.paragraphs)
                return text if text.strip() else None

            if result := _find_introduction(section.subsections):
                return result

        return None

    return _find_introduction(paper.sections)


def format_facets_for_prompt(
    facets: FacetsSummary,
    evidence: FacetEvidence | None = None,
) -> str:
    """Format facets as readable text for inclusion in a prompt.

    When ``evidence`` is provided, verbatim quotes from the paper are included
    beneath each item.

    Args:
        facets: The facets summary to format.
        evidence: Optional per-item evidence from extraction.

    Returns:
        Human-readable string with labelled facet lists.
    """
    parts: list[str] = []

    for name, items in facets.facet_items().items():
        if not items:
            continue

        label = name.capitalize()
        item_evidence: list[list[str]] = evidence.get(name, []) if evidence else []

        lines: list[str] = []
        for i, item in enumerate(items):
            lines.append(f"  - {item}")
            if i < len(item_evidence):
                for quote in item_evidence[i]:
                    lines.append(f'    > "{quote}"')

        parts.append(f"{label}:\n" + "\n".join(lines))

    return "\n\n".join(parts)


TERM_PROMPTS = {
    "default": """Title: {title}

Abstract:
{abstract_text}

Facets:
{facets_text}

Extract discriminative keyphrases from this paper:

1. **Paper-level terms** ({paper_terms_count} keyphrases): Core technical concepts that \
identify this paper. Use canonical terminology (e.g. "transformer architecture" not \
"the architecture proposed"). Multi-word noun phrases (2-5 words).

2. **Aspect-level terms** ({aspect_terms_count} keyphrases per aspect):
   - problems: Terms describing the research problems
   - methods: Terms describing techniques and approaches
   - results: Terms describing findings and metrics
   - contributions: Terms describing novelties and artefacts

Guidelines:
- Use standard, canonical terminology from the field
- Be specific: "byte-pair encoding tokenisation" not "tokenisation"
- Avoid generic terms like "machine learning" or "deep learning" unless central
- No repetition: aspect terms must not duplicate paper-level terms
- Each keyphrase should be a noun phrase of 2-5 words

Respond in JSON:
{{"paper_terms": [...], "aspect_terms": {{"problems": [...], "methods": [...], \
"results": [...], "contributions": [...]}}}}""",
}
"""Named prompt templates for keyphrase extraction.

Each template receives: title, abstract_text, facets_text,
paper_terms_count, aspect_terms_count.
"""


def prepare_term_prompt(
    title: str,
    abstract: str | None,
    facets: FacetsSummary,
    *,
    evidence: FacetEvidence | None = None,
    paper_terms_count: str = "8-12",
    aspect_terms_count: str = "3-5",
    prompt_key: str = "default",
) -> str:
    """Build the user prompt for keyphrase extraction.

    Args:
        title: Paper title.
        abstract: Abstract text, or None if unavailable.
        facets: Extracted facets for context.
        evidence: Optional per-item evidence from extraction.
        paper_terms_count: Range string for paper-level keyphrases (e.g. "8-12").
        aspect_terms_count: Range string for per-aspect keyphrases (e.g. "3-5").
        prompt_key: Key into TERM_PROMPTS for the template to use.

    Returns:
        Formatted prompt string.

    Raises:
        KeyError: If *prompt_key* is not found in TERM_PROMPTS.
    """
    abstract_text = abstract if abstract else "[No abstract available]"
    facets_text = format_facets_for_prompt(facets, evidence=evidence)

    return TERM_PROMPTS[prompt_key].format(
        title=title,
        abstract_text=abstract_text,
        facets_text=facets_text,
        paper_terms_count=paper_terms_count,
        aspect_terms_count=aspect_terms_count,
    )


_TERM_SYSTEM_PROMPT = (
    "You are an expert at identifying key technical concepts in academic papers. "
    "Extract discriminative keyphrases using standard canonical terminology. "
    "Multi-word noun phrases (2-5 words). Always respond with valid JSON."
)
"""System prompt for keyphrase extraction."""


class PaperTermsResponse(BaseModel, frozen=True):
    """Structured output schema for paper-level keyphrase extraction."""

    paper_terms: tuple[str, ...] = ()
    aspect_terms: AspectTerms = Field(default_factory=AspectTerms)

    def finalise(self, corpusid: CorpusId) -> PaperTerms:
        """Convert structured response into finalised :class:`PaperTerms`."""
        return PaperTerms(
            corpusid=corpusid,
            paper_terms=self.paper_terms,
            aspect_terms=self.aspect_terms,
        )


def deduplicate_terms(terms: PaperTerms) -> PaperTerms:
    """Remove aspect terms that duplicate paper-level terms (case-insensitive).

    Args:
        terms: The extracted paper terms to deduplicate.

    Returns:
        New PaperTerms with overlapping aspect terms removed.
    """
    paper_lower = {t.lower() for t in terms.paper_terms}

    def _filter(items: Sequence[str]) -> tuple[str, ...]:
        """Remove items whose lowercase form is in paper_lower."""
        return tuple(item for item in items if item.lower() not in paper_lower)

    return PaperTerms(
        corpusid=terms.corpusid,
        paper_terms=terms.paper_terms,
        aspect_terms=AspectTerms(
            problems=_filter(terms.aspect_terms.problems),
            methods=_filter(terms.aspect_terms.methods),
            results=_filter(terms.aspect_terms.results),
            contributions=_filter(terms.aspect_terms.contributions),
        ),
    )


async def extract_terms_for_paper(
    paper: PaperGraph,
    facets: FacetsSummary,
    client: LLMClient,
    *,
    evidence: FacetEvidence | None = None,
    paper_terms_count: str = "8-12",
    aspect_terms_count: str = "3-5",
    prompt_key: str = "default",
) -> PaperTerms | None:
    """Extract keyphrases for a single paper using an LLM client.

    Args:
        paper: The paper to extract terms from.
        facets: Pre-extracted facets for this paper.
        client: LLM client for structured completions.
        evidence: Optional per-item evidence from extraction.
        paper_terms_count: Range for paper-level keyphrases.
        aspect_terms_count: Range for per-aspect keyphrases.
        prompt_key: Prompt template key.

    Returns:
        Extracted PaperTerms (deduplicated), or None on failure.
    """
    abstract = extract_abstract_text(paper)
    prompt = prepare_term_prompt(
        paper.title,
        abstract,
        facets,
        evidence=evidence,
        paper_terms_count=paper_terms_count,
        aspect_terms_count=aspect_terms_count,
        prompt_key=prompt_key,
    )

    try:
        llm_result = await client.complete(
            CompletionRequest(system=_TERM_SYSTEM_PROMPT, user=prompt),
            PaperTermsResponse,
        )
        if terms_response := llm_result.result:
            return terms_response.finalise(paper.corpusid)
    except Exception:
        logger.exception("Failed to extract terms for %d", paper.corpusid)

    return None


def load_facets_index(
    facets_path: Path,
    *,
    facets_type: FacetRecordType,
) -> tuple[dict[CorpusId, FacetsSummary], dict[CorpusId, FacetEvidence]]:
    """Load facets and evidence from a JSONL file into corpus-ID-keyed indices.

    Args:
        facets_path: Path to facets.jsonl file.
        facets_type: On-disk schema for facet records.

    Returns:
        Tuple of (facets index, evidence index).  The evidence index is empty
        for standard facet records that carry no evidence.
    """
    facets_index: dict[CorpusId, FacetsSummary] = {}
    evidence_index: dict[CorpusId, FacetEvidence] = {}

    for record in iter_jsonl(facets_path):
        parsed = facets_from_type(record, facets_type)
        facets_index[parsed.corpusid] = FacetsSummary(
            problems=parsed.problems,
            methods=parsed.methods,
            results=parsed.results,
            contributions=parsed.contributions,
        )
        if ev := evidence_from_type(record, facets_type):
            evidence_index[parsed.corpusid] = ev

    return facets_index, evidence_index


def _iter_papers_with_facets(
    graphs_path: Path,
    facets_index: dict[CorpusId, FacetsSummary],
    processed_ids: set[CorpusId],
    filter_ids: set[CorpusId],
) -> Iterator[PaperGraph]:
    """Yield paper graphs that have facets and are not yet processed.

    Wraps :func:`~genret.processing.cli.iter_paper_graphs` with an additional
    check that the paper has an entry in *facets_index*.

    Args:
        graphs_path: Path to paper_graphs.jsonl file.
        facets_index: Index of available facets (papers without facets are skipped).
        processed_ids: Set of already-processed corpus IDs to skip.
        filter_ids: If non-empty, only include papers with these corpus IDs.

    Yields:
        PaperGraph objects that pass all filters and have facets.
    """
    for paper in iter_paper_graphs(graphs_path, processed_ids, filter_ids):
        if paper.corpusid in facets_index:
            yield paper


async def run_extract_terms(
    splits_dir: Path,
    filter_ids: set[CorpusId],
    concurrent: int = 100,
    model: str = "gpt-4o-mini",
    reasoning_effort: ReasoningEffort = ReasoningEffort.MINIMAL,
    force: bool = False,
    *,
    paper_terms_count: str = "8-12",
    aspect_terms_count: str = "3-5",
    prompt_key: str = "default",
    facets_type: FacetRecordType,
    include_evidence: bool = False,
    provider: str = "openai",
    azure_endpoint: str | None = None,
    api_version: str | None = None,
    output_dir: Path | None = None,
    limit: int = 0,
) -> None:
    """Run term extraction phase.

    Args:
        splits_dir: Splits directory containing processed/ subfolder.
        filter_ids: If non-empty, only process papers with these corpus IDs.
        concurrent: Number of concurrent API calls.
        model: OpenAI model to use.
        reasoning_effort: OpenAI reasoning-effort policy for GPT-5 models.
        force: If True, reprocess all papers.
        paper_terms_count: Range for paper-level keyphrases.
        aspect_terms_count: Range for per-aspect keyphrases.
        prompt_key: Prompt template key.
        facets_type: On-disk schema for facet records.
        include_evidence: Whether to include evidence in prompts.
        provider: LLM provider — ``"openai"`` or ``"azure-openai"``.
        azure_endpoint: Azure endpoint URL (required for ``"azure-openai"``).
        api_version: Azure API version override.
        output_dir: Override output directory. Defaults to splits_dir/processed/.
        limit: Maximum papers to process (0 = all).
    """
    processed_dir = splits_dir / "processed"
    out_dir = output_dir or processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_path = resolve_jsonl(processed_dir / "paper_graphs.jsonl.gz")
    facets_path = resolve_jsonl(processed_dir / "facets.jsonl.gz")
    terms_path = out_dir / "terms.jsonl.zst"
    progress_path = out_dir / "terms_progress.txt"

    if not graphs_path.exists():
        die(f"Paper graphs not found: {graphs_path}\nRun 'parse' first.")
    if not facets_path.exists():
        die(f"Facets not found: {facets_path}\nRun 'extract' first.")

    processed_ids, terms_mode = prepare_extract_state(progress_path, terms_path, force)
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} papers already processed")

    # Load facets index
    facets_index, evidence_index = load_facets_index(
        facets_path, facets_type=facets_type
    )
    if not include_evidence:
        evidence_index = dict[CorpusId, FacetEvidence]()
    logger.info(f"Loaded facets for {len(facets_index)} papers")

    # Counting pass
    available = countit(
        _iter_papers_with_facets(graphs_path, facets_index, processed_ids, filter_ids)
    )
    paper_count = min(available, limit) if limit > 0 else available
    if paper_count == 0:
        logger.info("All papers already processed!")
        return

    logger.info(
        "Papers to process: %d\nModel: %s\nReasoning effort: %s\nConcurrent: %d",
        paper_count,
        model,
        reasoning_effort.value,
        concurrent,
    )

    def request_pairs() -> Iterator[tuple[CorpusId, CompletionRequest]]:
        """Stream ``(corpusid, CompletionRequest)`` pairs from disk."""
        papers = _iter_papers_with_facets(
            graphs_path, facets_index, processed_ids, filter_ids
        )
        for paper in itertools.islice(papers, paper_count):
            facets = facets_index[paper.corpusid]
            ev = evidence_index.get(paper.corpusid)
            abstract = extract_abstract_text(paper)
            prompt = prepare_term_prompt(
                paper.title,
                abstract,
                facets,
                evidence=ev,
                paper_terms_count=paper_terms_count,
                aspect_terms_count=aspect_terms_count,
                prompt_key=prompt_key,
            )
            yield (
                paper.corpusid,
                CompletionRequest(system=_TERM_SYSTEM_PROMPT, user=prompt),
            )

    total_cost = 0.0

    async with (
        LLMClient.create(
            provider,
            model=model,
            reasoning_effort=reasoning_effort,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        ) as client,
        AsyncLineWriter.open(terms_path, mode=terms_mode) as writer,
    ):
        async for corpusid, llm_result in track_progress(
            client.batch_complete(
                request_pairs(),
                response_model=PaperTermsResponse,
                concurrent=concurrent,
            ),
            desc="Extracting terms...",
            total=paper_count,
        ):
            total_cost += llm_result.cost
            if terms_response := llm_result.result:
                await writer.write_obj(terms_response.finalise(corpusid))
                append_processed_id(progress_path, corpusid)

    write_run_metadata(
        out_dir / "terms_run.json",
        model=resolve_model(model),
        model_short=model,
        temperature=0.0,
        seed=0,
        total_cost_usd=total_cost,
        reasoning_effort=reasoning_effort.value,
    )
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Term extraction complete: {terms_path}")


@app.command(name="extract")
def cmd_extract_terms(
    splits_dir: Annotated[
        Path, typer.Argument(help="Splits directory with train/dev/test.txt")
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Output directory. Defaults to processed/."
        ),
    ] = None,
    concurrent: Annotated[
        int, typer.Option("--concurrent", "-c", help="Concurrent API calls.")
    ] = 100,
    model: Annotated[
        str, typer.Option("--model", "-m", help="OpenAI model to use.")
    ] = "gpt-4o-mini",
    reasoning_effort: Annotated[
        ReasoningEffort,
        typer.Option(
            "--reasoning-effort",
            help=(
                "Reasoning effort for GPT-5 models: default, none, minimal, "
                "low, medium, high, xhigh."
            ),
        ),
    ] = ReasoningEffort.MINIMAL,
    paper_terms: Annotated[
        str, typer.Option("--paper-terms", help="Range for paper-level keyphrases.")
    ] = "8-12",
    aspect_terms: Annotated[
        str, typer.Option("--aspect-terms", help="Range for per-aspect keyphrases.")
    ] = "3-5",
    prompt_key: Annotated[
        str, typer.Option("--prompt-key", help="Prompt template key from TERM_PROMPTS.")
    ] = "default",
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Reprocess all papers.")
    ] = False,
    facets_type: Annotated[
        FacetRecordType,
        typer.Option(
            "--facets-type",
            help="Facet record schema in facets.jsonl.gz: standard, evidence, or evidence-list.",
        ),
    ] = FacetRecordType.STANDARD,
    include_evidence: Annotated[
        bool,
        typer.Option(
            "--include-evidence/--no-include-evidence",
            help="Include evidence quotes in the keyphrase extraction prompt.",
        ),
    ] = False,
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
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Maximum papers to process (0 = all).")
    ] = 0,
) -> None:
    """Extract discriminative keyphrases from papers using an LLM.

    Reads paper_graphs.jsonl and facets.jsonl from splits_dir/processed/.
    Writes terms.jsonl and terms_progress.txt to splits_dir/processed/.
    """
    if force:
        confirm_force()

    if not splits_dir.exists():
        die(f"Splits directory not found: {splits_dir}")

    filter_ids = load_split_ids(splits_dir)
    if not filter_ids:
        die(f"No IDs found in {splits_dir}/{{train,dev,test}}.txt")

    logger.info(
        f"=== Extracting Terms ===\nSplits: {splits_dir}\nPapers: {len(filter_ids)}"
    )
    asyncio.run(
        run_extract_terms(
            splits_dir,
            filter_ids,
            concurrent,
            model,
            reasoning_effort,
            force,
            paper_terms_count=paper_terms,
            aspect_terms_count=aspect_terms,
            prompt_key=prompt_key,
            facets_type=facets_type,
            include_evidence=include_evidence,
            provider=provider,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            output_dir=output,
            limit=limit,
        )
    )
