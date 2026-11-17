"""Silvertongue — out-of-corpus paper acquisition.

Acquires papers not in our corpus via:
1. S2 API title search → candidate papers with metadata
2. PDF download → temporary file
3. PyMuPDF4LLM text extraction → markdown
4. Markdown → PaperGraph (headings → sections with type classification)
5. S2 API reference fetching → corpus IDs for bib coupling
6. Disk cache to avoid re-downloading
"""

import json
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from genret._shed.pymupdf import pdf_to_markdown
from genret.paper_parsing import (
    PaperGraph,
    Section,
    SectionType,
    classify_section,
)
from genret.s2.api import SemanticScholarAPI, extract_pdf_url
from genret.s2.types import S2Paper
from genret.types import CorpusId

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class AcquiredPaper:
    """Result of acquiring an out-of-corpus paper."""

    graph: PaperGraph
    references: frozenset[int]


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------


async def download_pdf(
    url: str,
    http_client: httpx.AsyncClient,
    max_retries: int = 3,
) -> bytes:
    """Download a PDF from a URL with retries.

    Args:
        url: PDF URL to download.
        http_client: Async HTTP client for making requests.
        max_retries: Number of retry attempts on failure.

    Returns:
        Raw PDF bytes.

    Raises:
        RuntimeError: If all download attempts fail.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await http_client.get(url, follow_redirects=True)
            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"PDF download attempt {attempt + 1}/{max_retries} failed"
                    f" for {url}: {e}"
                )
            continue

        return response.content

    raise RuntimeError(
        f"Failed to download PDF from {url} after {max_retries} attempts"
    ) from last_error


# ---------------------------------------------------------------------------
# PyMuPDF4LLM extraction
# ---------------------------------------------------------------------------


def extract_markdown_from_pdf(pdf_bytes: bytes) -> str:
    """Extract markdown text from PDF bytes using PyMuPDF4LLM.

    Args:
        pdf_bytes: Raw PDF file contents.

    Returns:
        Markdown string extracted from the PDF.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "paper.pdf"
        path.write_bytes(pdf_bytes)
        return pdf_to_markdown(str(path))


# ---------------------------------------------------------------------------
# Markdown → PaperGraph
# ---------------------------------------------------------------------------

# Matches markdown headings: # Heading, ## Heading, ### Heading
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


# TODO: Use proper markdown parser
def _split_markdown_sections(
    markdown: str,
) -> list[tuple[int, str, str]]:
    """Split markdown into (level, heading, body) tuples.

    Args:
        markdown: Markdown text with headings.

    Returns:
        List of (heading_level, heading_text, body_text) tuples.
        Level 0 is used for text before the first heading.
    """
    sections: list[tuple[int, str, str]] = []
    headings: list[tuple[int, int, int, str]] = []

    for m in _HEADING_RE.finditer(markdown):
        level = len(m.group(1))
        heading_text = m.group(2).strip()
        headings.append((m.start(), m.end(), level, heading_text))

    if not headings:
        # No headings — entire document is one section
        if text := markdown.strip():
            sections.append((0, "", text))
        return sections

    # Text before first heading
    if preamble := markdown[: headings[0][0]].strip():
        sections.append((0, "", preamble))

    for i, (_start, end, level, heading) in enumerate(headings):
        next_start = headings[i + 1][0] if i + 1 < len(headings) else len(markdown)
        body = markdown[end:next_start].strip()
        sections.append((level, heading, body))

    return sections


def markdown_to_paper_graph(
    markdown: str,
    corpusid: int,
    title: str,
    authors: list[str],
    abstract: str | None = None,
) -> PaperGraph:
    """Convert extracted markdown text into a PaperGraph.

    Parses markdown headings into sections and classifies each section type
    using the same patterns as ``parse_paper()``.

    Args:
        markdown: Markdown text from PyMuPDF4LLM extraction.
        corpusid: Semantic Scholar corpus ID.
        title: Paper title.
        authors: List of author names.
        abstract: Optional abstract text from S2 metadata.

    Returns:
        Structured PaperGraph.
    """
    raw_sections = _split_markdown_sections(markdown)

    sections: list[Section] = []
    for _level, heading, body in raw_sections:
        paragraphs = [p for p_ in body.split("\n\n") if (p := p_.strip())]
        if not paragraphs and not heading:
            continue

        sections.append(
            Section(
                heading=heading,
                type=classify_section(heading) if heading else SectionType.OTHER,
                paragraphs=paragraphs,
            )
        )

    return PaperGraph(
        corpusid=CorpusId(corpusid),
        title=title,
        authors=authors,
        sections=sections,
        abstract=abstract,
    )


# ---------------------------------------------------------------------------
# Silver cache
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, corpusid: int) -> Path:
    return cache_dir / f"{corpusid}.json"


def load_from_cache(cache_dir: Path, corpusid: int) -> AcquiredPaper | None:
    """Load a previously acquired paper from the Silver cache.

    Args:
        cache_dir: Silver cache directory (``cache/silver/``).
        corpusid: Semantic Scholar corpus ID.

    Returns:
        AcquiredPaper if cached, None otherwise.
    """
    path = _cache_path(cache_dir, corpusid)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        graph = PaperGraph.model_validate(data["graph"])
        refs = frozenset(data.get("references", []))
        return AcquiredPaper(graph=graph, references=refs)
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.warning(f"Failed to load Silver cache for {corpusid}: {e}")
        return None


def save_to_cache(
    cache_dir: Path,
    paper: AcquiredPaper,
) -> None:
    """Save an acquired paper to the Silver disk cache.

    Args:
        cache_dir: Silver cache directory (``cache/silver/``).
        paper: Acquired paper to cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, paper.graph.corpusid)
    data = {
        "graph": paper.graph.model_dump(),
        "references": sorted(paper.references),
    }
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# acquire_paper() orchestrator
# ---------------------------------------------------------------------------


async def acquire_paper(
    paper: S2Paper,
    cache_dir: Path,
    http_client: httpx.AsyncClient,
    s2_api: SemanticScholarAPI,
) -> AcquiredPaper:
    """Acquire an out-of-corpus paper: download, extract, parse, and cache.

    Steps:
    1. Check Silver cache
    2. Resolve PDF URL from S2 metadata
    3. Download PDF
    4. Extract text with PyMuPDF4LLM → markdown
    5. Parse markdown into PaperGraph
    6. Fetch reference corpus IDs from S2 API
    7. Cache and return

    Args:
        paper: S2Paper metadata from search results.
        cache_dir: Silver cache directory (``cache/silver/``).
        http_client: Async HTTP client for PDF downloads.
        s2_api: Semantic Scholar API client for reference fetching.

    Returns:
        AcquiredPaper with parsed PaperGraph and reference corpus IDs.

    Raises:
        ValueError: If the paper has no corpus ID or resolvable PDF URL.
        RuntimeError: If PDF download or text extraction fails.
    """
    if paper.corpus_id is None:
        raise ValueError(f"Paper {paper.paper_id!r} has no corpus ID")

    # Step 1: check cache
    cached = load_from_cache(cache_dir, paper.corpus_id)
    if cached is not None:
        logger.info(f"Silver cache hit for corpus ID {paper.corpus_id}")
        return cached

    # Step 2: resolve PDF URL
    pdf_url, _ = extract_pdf_url(paper)
    if not pdf_url:
        raise ValueError(
            f"No resolvable PDF URL for paper {paper.paper_id!r} ({paper.title!r})"
        )

    # Step 3: download PDF
    logger.info(f"Downloading PDF for {paper.title!r} from {pdf_url}")
    pdf_bytes = await download_pdf(pdf_url, http_client)

    # Step 4: extract markdown
    logger.info(f"Extracting text from PDF for {paper.title!r}")
    markdown = extract_markdown_from_pdf(pdf_bytes)

    # Step 5: parse into PaperGraph
    authors = [a.name for a in (paper.authors or []) if a.name]
    graph = markdown_to_paper_graph(
        markdown=markdown,
        corpusid=paper.corpus_id,
        title=paper.title,
        authors=authors,
        abstract=paper.abstract,
    )

    # Step 6: fetch reference corpus IDs
    logger.info(f"Fetching references for {paper.title!r}")
    ref_corpus_ids = await s2_api.get_reference_corpus_ids(paper.paper_id)
    references = frozenset(ref_corpus_ids)

    # Step 7: cache and return
    result = AcquiredPaper(graph=graph, references=references)
    save_to_cache(cache_dir, result)
    logger.info(
        f"Acquired paper {paper.title!r}: {len(graph.sections)} sections,"
        f" {len(references)} references"
    )

    return result
