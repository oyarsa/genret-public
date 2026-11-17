"""Facet extraction command: LLM-based extraction of problems/methods/results/contributions.

Implements the async ``run_extract`` pipeline plus all prompt definitions,
response models, and supporting helpers used by the ``extract`` CLI command.
"""

import json
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ValidationError

from genret import llm
from genret.paper_parsing import (
    FacetRecordType,
    Facets,
    FacetsWithEvidenceListRecord,
    FacetsWithEvidenceRecord,
    FacetWithEvidence,
    FacetWithEvidenceList,
    PaperGraph,
    Section,
)
from genret.types import CorpusId
from genret.utils import (
    AsyncLineWriter,
    OpenMode,
    Unreachable,
    backup_files,
    count_lines,
    countit,
    die,
    load_ids_from_file,
    open_auto,
    resolve_jsonl,
    track_progress,
)

logger = logging.getLogger(__name__)


def append_processed_id(path: Path, corpusid: CorpusId) -> None:
    """Append a corpus ID to the progress file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{corpusid}\n")


def prepare_extract_state(
    progress_path: Path, facets_path: Path, force: bool
) -> tuple[set[CorpusId], OpenMode]:
    """Prepare resume/overwrite state for facet extraction.

    Args:
        progress_path: Progress file storing processed corpus IDs.
        facets_path: Primary output JSONL path for extracted facets.
        force: Whether to overwrite previous outputs.

    Returns:
        Tuple of (already_processed_ids, facets_open_mode).
    """
    if force:
        backup_files([facets_path, progress_path])
        return set(), "w"

    return load_ids_from_file(progress_path), "a"


def section_to_text(section: Section, depth: int = 0) -> str:
    """Convert a section to plain text for prompting.

    Args:
        section: The section to convert.
        depth: Current nesting depth for indentation.

    Returns:
        Plain text representation of the section.
    """
    lines: list[str] = []
    indent = "  " * depth

    if section.heading:
        lines.append(f"{indent}## {section.heading}")

    for para in section.paragraphs:
        lines.append(f"{indent}{para}")

    for subsection in section.subsections:
        lines.append(section_to_text(subsection, depth + 1))

    return "\n\n".join(lines)


def paper_to_text(paper: PaperGraph) -> str:
    """Convert a paper graph to plain text for prompting.

    Args:
        paper: The paper graph to convert.

    Returns:
        Plain text representation of the paper.
    """
    parts = [f"# {paper.title}"]

    if paper.authors:
        parts.append(f"Authors: {', '.join(paper.authors)}")

    for section in paper.sections:
        parts.append(section_to_text(section))

    return "\n\n".join(parts)


class SectionMode(StrEnum):
    """How much of a paper to feed into the extraction prompt."""

    FULL = "full"
    """Title + authors + every section, head-truncated to the token budget."""
    CORE = "core"
    """Title + abstract + only facet-relevant sections (see CORE_SECTION_RE)."""


# Headings kept in ``core`` mode: the sections that carry problems / methods /
# results / contributions signal. Substring match on the heading, case
# insensitive. Deliberately excludes related work, limitations, background,
# ethics, acknowledgments, appendices, and references.
CORE_SECTION_RE = re.compile(
    r"introduction|method|approach|\bmodel|experiment|result|evaluation|"
    r"analysis|discussion|conclusion|contribution|preliminar|propos|"
    r"data|task|system",
    re.IGNORECASE,
)


def select_core_section_text(paper: PaperGraph) -> str:
    """Build prompt text from a paper's title, abstract, and core sections.

    Keeps only sections whose heading matches :data:`CORE_SECTION_RE`. If none
    match (unusual headings, or unlabeled body), falls back to every section so
    we never feed an empty paper.

    Args:
        paper: The paper graph to convert.

    Returns:
        Plain text of title + abstract + selected sections.
    """
    selected = [s for s in paper.sections if CORE_SECTION_RE.search(s.heading or "")]
    if not selected:
        selected = list(paper.sections)

    parts = [f"# {paper.title}"]
    if paper.abstract:
        parts.append(f"## Abstract\n\n{paper.abstract}")
    for section in selected:
        parts.append(section_to_text(section))

    return "\n\n".join(parts)


def _prepare_prompt_paper_text(
    paper: PaperGraph,
    max_tokens: int = 100000,
    section_mode: SectionMode = SectionMode.FULL,
) -> str:
    """Prepare paper text for prompting, truncating if necessary.

    Args:
        paper: The paper graph to convert.
        max_tokens: Maximum tokens for the paper content.
        section_mode: ``full`` for the whole paper, ``core`` for title +
            abstract + facet-relevant sections only.

    Returns:
        Paper text content for prompt templates.
    """
    if section_mode is SectionMode.CORE:
        paper_text = select_core_section_text(paper)
    else:
        paper_text = paper_to_text(paper)

    tokens = llm.tokenise(paper_text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        paper_text = llm.detokenise(tokens) + "\n\n[Content truncated]"

    return paper_text


class FacetPromptStyle(StrEnum):
    """Prompt/schema variants for facet extraction."""

    DEFAULT = "default"
    WITH_EVIDENCE = "evidence"
    WITH_EVIDENCE_LIST = "evidence-list"

    def response_model(
        self,
    ) -> type[
        FacetsResponse | FacetsWithEvidenceResponse | FacetsWithEvidenceListResponse
    ]:
        """Return the structured output model for this prompt style."""
        match self:
            case FacetPromptStyle.WITH_EVIDENCE:
                return FacetsWithEvidenceResponse
            case FacetPromptStyle.WITH_EVIDENCE_LIST:
                return FacetsWithEvidenceListResponse
            case FacetPromptStyle.DEFAULT:
                return FacetsResponse
            case _:
                raise Unreachable

    def to_type(self) -> FacetRecordType:
        """Convert between ``FacetPromptStyle`` and ``FacetRecordType``."""
        match self:
            case FacetPromptStyle.WITH_EVIDENCE:
                return FacetRecordType.WITH_EVIDENCE
            case FacetPromptStyle.WITH_EVIDENCE_LIST:
                return FacetRecordType.WITH_EVIDENCE_LIST
            case FacetPromptStyle.DEFAULT:
                return FacetRecordType.STANDARD
            case _:
                raise Unreachable


@dataclass(frozen=True)
class _Prompt:
    """User prompt for extraction.

    Attributes:
        description: Single-line description of the prompt, to be shown to the user.
        template: The actual prompt. Should have placeholder variables to be filled
            using ``str.format``.
    """

    description: str
    template: str


FACET_PROMPTS: dict[FacetPromptStyle, _Prompt] = {
    FacetPromptStyle.DEFAULT: _Prompt(
        "Problems, methods, results and contributions as plain strings",
        """Analyse this academic paper and extract the following facets.
For each facet, provide 1-5 distinct items (1-2 sentences each).

1. **Problems**: What problems or research questions does this paper address?
2. **Methods**: What approaches, methods, or techniques does the paper propose or use?
3. **Results**: What are the main results or findings?
4. **Contributions**: What are the contributions or novelties of this work?

Paper:
{paper_text}""",
    ),
    FacetPromptStyle.WITH_EVIDENCE: _Prompt(
        "Same facets as `default`, each paired with supporting evidence text",
        """Analyse this academic paper and extract the following facets.
For each facet, provide 1-5 distinct items (1-2 sentences each), and include
supporting evidence from the paper for every item.

For each item, return:
- "facet": the extracted claim (concise, specific, 1-2 sentences)
- "evidence": short supporting evidence grounded in the paper text

1. **Problems**: What problems or research questions does this paper address?
2. **Methods**: What approaches, methods, or techniques does the paper propose or use?
3. **Results**: What are the main results or findings?
4. **Contributions**: What are the contributions or novelties of this work?

Paper:
{paper_text}""",
    ),
    FacetPromptStyle.WITH_EVIDENCE_LIST: _Prompt(
        "Same facets as `default`, each paired with 1-3 exact quotes from the paper",
        """Analyse this academic paper and extract the following facets.
For each facet, provide 1-5 distinct items (1-2 sentences each), and include
1-3 exact quotes from the paper that directly support the facet.

For each item, return:
- "facet": the extracted claim (concise, specific, 1-2 sentences)
- "evidence": a list of 1-3 verbatim sentences or phrases copied directly from the paper text

The evidence strings must be exact copies of text from the paper — do not paraphrase or summarise.

1. **Problems**: What problems or research questions does this paper address?
2. **Methods**: What approaches, methods, or techniques does the paper propose or use?
3. **Results**: What are the main results or findings?
4. **Contributions**: What are the contributions or novelties of this work?

Paper:
{paper_text}""",
    ),
}
"""Named prompt templates for facet extraction."""


def prepare_prompt(
    paper: PaperGraph,
    *,
    max_tokens: int = 100000,
    prompt_style: FacetPromptStyle = FacetPromptStyle.DEFAULT,
    section_mode: SectionMode = SectionMode.FULL,
) -> str:
    """Prepare a prompt for facet extraction, truncating if necessary.

    Args:
        paper: The paper graph to create a prompt for.
        max_tokens: Maximum tokens for the paper content.
        prompt_style: Prompt template key controlling output schema.
        section_mode: Whole paper (``full``) or core sections only (``core``).

    Returns:
        The prepared prompt string.
    """
    paper_text = _prepare_prompt_paper_text(
        paper, max_tokens=max_tokens, section_mode=section_mode
    )
    return FACET_PROMPTS[prompt_style].template.format(paper_text=paper_text)


FACET_SYSTEM_PROMPT = (
    "You are an expert at analysing academic papers. "
    "Extract the requested facets accurately and concisely. "
    "Always respond with valid JSON."
)
"""System prompt for facet extraction."""


class FacetsResponse(BaseModel, frozen=True):
    """Structured output schema for facet extraction responses."""

    problems: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    results: tuple[str, ...] = ()
    contributions: tuple[str, ...] = ()

    def finalise(self, corpusid: CorpusId) -> Facets:
        """Convert a structured facet response into a finalised ``Facets`` object."""
        return Facets(
            corpusid=corpusid,
            problems=self.problems,
            methods=self.methods,
            results=self.results,
            contributions=self.contributions,
        )


class FacetsWithEvidenceResponse(BaseModel, frozen=True):
    """Structured output schema for evidence-backed facet extraction."""

    problems: Sequence[FacetWithEvidence] = ()
    methods: Sequence[FacetWithEvidence] = ()
    results: Sequence[FacetWithEvidence] = ()
    contributions: Sequence[FacetWithEvidence] = ()

    def finalise(self, corpusid: CorpusId) -> FacetsWithEvidenceRecord:
        """Convert a response into an evidence-backed final record."""
        return FacetsWithEvidenceRecord(
            corpusid=corpusid,
            problems=self.problems,
            methods=self.methods,
            results=self.results,
            contributions=self.contributions,
        )


class FacetsWithEvidenceListResponse(BaseModel, frozen=True):
    """Structured output schema for list-evidence-backed facet extraction."""

    problems: Sequence[FacetWithEvidenceList] = ()
    methods: Sequence[FacetWithEvidenceList] = ()
    results: Sequence[FacetWithEvidenceList] = ()
    contributions: Sequence[FacetWithEvidenceList] = ()

    def finalise(self, corpusid: CorpusId) -> FacetsWithEvidenceListRecord:
        """Convert a response into a list-evidence-backed final record."""
        return FacetsWithEvidenceListRecord(
            corpusid=corpusid,
            problems=self.problems,
            methods=self.methods,
            results=self.results,
            contributions=self.contributions,
        )


def iter_paper_graphs(
    graphs_path: Path,
    processed_ids: set[CorpusId],
    filter_ids: set[CorpusId],
    limit: int = 0,
) -> Iterator[PaperGraph]:
    """Yield paper graphs one at a time, filtering already processed ones.

    Same filtering logic as :func:`load_paper_graphs` but O(1) memory since
    graphs are yielded rather than collected into a list.

    Args:
        graphs_path: Path to paper_graphs.jsonl file.
        processed_ids: Set of already processed corpus IDs to skip.
        filter_ids: If non-empty, only include papers with these corpus IDs.
        limit: Stop after yielding this many papers (0 = no limit).

    Yields:
        PaperGraph objects matching the filter criteria.
    """
    count = 0
    with open_auto(graphs_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                paper = PaperGraph.model_validate_json(line)
            except (json.JSONDecodeError, ValidationError):
                logger.exception("Failed to parse paper")
                continue

            if paper.corpusid in processed_ids:
                continue
            if filter_ids and paper.corpusid not in filter_ids:
                continue

            yield paper

            count += 1
            if limit > 0 and count >= limit:
                break


def load_paper_graphs(
    graphs_path: Path,
    processed_ids: set[CorpusId],
    filter_ids: set[CorpusId],
    limit: int = 0,
) -> list[PaperGraph]:
    """Load paper graphs from JSONL file, filtering already processed ones.

    Args:
        graphs_path: Path to paper_graphs.jsonl file.
        processed_ids: Set of already processed corpus IDs to skip.
        filter_ids: If non-empty, only include papers with these corpus IDs.
        limit: Stop after collecting this many papers (0 = no limit).

    Returns:
        List of PaperGraph objects to process.
    """
    papers: list[PaperGraph] = []
    total = count_lines(graphs_path)
    with open_auto(graphs_path, "r") as f:
        for line in track_progress(f, desc="Loading paper graphs...", total=total):
            if not line.strip():
                continue

            try:
                paper = PaperGraph.model_validate_json(line)

                if paper.corpusid in processed_ids:
                    continue
                if filter_ids and paper.corpusid not in filter_ids:
                    continue

                papers.append(paper)
                if limit > 0 and len(papers) >= limit:
                    break
            except (json.JSONDecodeError, ValidationError):
                logger.exception("Failed to parse paper")
    return papers


async def run_extract(
    splits_dir: Path,
    filter_ids: set[CorpusId],
    output_dir: Path,
    concurrent: int = 100,
    model: str = "gpt-4o-mini",
    reasoning_effort: llm.ReasoningEffort = llm.ReasoningEffort.MINIMAL,
    force: bool = False,
    prompt_style: FacetPromptStyle = FacetPromptStyle.DEFAULT,
    provider: str = "openai",
    azure_endpoint: str | None = None,
    api_version: str | None = None,
    section_mode: SectionMode = SectionMode.FULL,
    max_input_tokens: int = 100000,
    limit: int = 0,
) -> None:
    """Run facet extraction phase.

    Args:
        splits_dir: Splits directory containing processed/paper_graphs.jsonl.
        filter_ids: If non-empty, only process papers with these corpus IDs.
        output_dir: Output directory for extracted facets.
        concurrent: Number of concurrent API calls.
        model: OpenAI model to use.
        reasoning_effort: OpenAI reasoning-effort policy for GPT-5 models.
        force: If True, reprocess all papers.
        prompt_style: Prompt/schema variant for facet extraction.
        provider: LLM provider — ``"openai"`` or ``"azure-openai"``.
        azure_endpoint: Azure endpoint URL (required for ``"azure-openai"``).
        api_version: Azure API version override.
        section_mode: Whole paper (``full``) or core sections only (``core``).
        max_input_tokens: Truncate paper text to this many tokens.
        limit: Maximum papers to process (0 = all).
    """
    input_dir = splits_dir / "processed"
    output = output_dir
    graphs_path = resolve_jsonl(input_dir / "paper_graphs.jsonl.gz")
    facets_path = output / "facets.jsonl.zst"
    progress_path = output / "facets_progress.txt"

    if not graphs_path.exists():
        die(f"Paper graphs not found: {graphs_path}\nRun 'parse' first.")

    output.mkdir(parents=True, exist_ok=True)

    processed_ids, facets_mode = prepare_extract_state(
        progress_path,
        facets_path,
        force,
    )
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} papers already processed")

    paper_count = countit(
        iter_paper_graphs(graphs_path, processed_ids, filter_ids, limit=limit)
    )
    if paper_count == 0:
        logger.info("All papers already processed!")
        return

    logger.info(f"Papers to process: {paper_count}")
    logger.info(f"Model: {model}")
    logger.info(f"Reasoning effort: {reasoning_effort.value}")
    logger.info(f"Concurrent: {concurrent}")
    logger.info(f"Prompt style: {prompt_style.value}")

    def request_pairs() -> Iterator[tuple[CorpusId, llm.CompletionRequest]]:
        """Stream ``(corpusid, CompletionRequest)`` pairs from disk."""
        for paper in iter_paper_graphs(
            graphs_path, processed_ids, filter_ids, limit=limit
        ):
            request = llm.CompletionRequest(
                system=FACET_SYSTEM_PROMPT,
                user=prepare_prompt(
                    paper,
                    prompt_style=prompt_style,
                    section_mode=section_mode,
                    max_tokens=max_input_tokens,
                ),
            )
            yield (paper.corpusid, request)

    total_cost = 0.0

    response_model = prompt_style.response_model()

    async with (
        llm.LLMClient.create(
            provider,
            model=model,
            reasoning_effort=reasoning_effort,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        ) as client,
        AsyncLineWriter.open(facets_path, mode=facets_mode) as writer,
    ):
        async for corpusid, llm_result in track_progress(
            client.batch_complete(
                request_pairs(),
                response_model=response_model,
                concurrent=concurrent,
            ),
            desc="Extracting facets...",
            total=paper_count,
        ):
            total_cost += llm_result.cost
            if result := llm_result.result:
                await writer.write_obj(result.finalise(corpusid))
                append_processed_id(progress_path, corpusid)

    llm.write_run_metadata(
        output / "facets_run.json",
        model=llm.resolve_model(model),
        model_short=model,
        temperature=0.0,
        seed=0,
        total_cost_usd=total_cost,
        reasoning_effort=reasoning_effort.value,
    )
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Facets extraction complete: {facets_path}")
