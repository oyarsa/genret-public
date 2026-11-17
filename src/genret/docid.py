"""DocID encoding and inverted index construction.

Provides deterministic conversion of extracted keyphrases (`PaperTerms`) into structured
document identifier strings, and tokenisation of those strings into an inverted index
for constrained decoding.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from genret.paper_parsing import (
    AspectTerms,
    DocIDIndex,
    DocIDRecord,
    IndexStats,
    PaperTerms,
)
from genret.tokeniser import load_tokeniser, tokenise_batch
from genret.types import CorpusId
from genret.utils import (
    LineWriter,
    backup_files,
    confirm_force,
    die,
    iter_jsonl,
    load_split_ids,
    make_app,
    resolve_jsonl,
    track_progress,
)

logger = logging.getLogger(__name__)

app = make_app("Encode DocIDs and build inverted index")

# -- Constants --

ASPECT_ORDER: tuple[str, ...] = ("problems", "methods", "results", "contributions")
"""Canonical ordering of aspect sections within a DocID."""

PAPER_START = "<PAPER>"
"""Opening delimiter for the paper-level term section."""
PAPER_END = "<END_PAPER>"
"""Closing delimiter for the paper-level term section."""
ASPECT_END = "<END_ASPECT>"
"""Closing delimiter for each aspect section."""
TERM_SEPARATOR = " ; "
"""Separator between individual terms within a section."""


def _aspect_tag(name: str) -> str:
    """Return the opening delimiter for an aspect section.

    Args:
        name: Aspect name (e.g. `"problems"`).

    Returns:
        Opening tag string (e.g. `"<ASPECT=problems>"`).
    """
    return f"<ASPECT={name}>"


SPECIAL_TOKENS = (
    PAPER_START,
    PAPER_END,
    *(_aspect_tag(name) for name in ASPECT_ORDER),
    ASPECT_END,
)
"""All structural tokens used in DocID encoding, for tokeniser vocabulary extension."""


# -- Encoding --


def get_aspect_terms(aspect_terms: AspectTerms, name: str) -> Sequence[str]:
    """Look up terms for a named aspect.

    Args:
        aspect_terms: The aspect terms container.
        name: One of `"problems"`, `"methods"`, `"results"`, `"contributions"`.

    Returns:
        The term sequence for the given aspect.

    Raises:
        ValueError: If *name* is not a recognised aspect.
    """
    match name:
        case "problems":
            return aspect_terms.problems
        case "methods":
            return aspect_terms.methods
        case "results":
            return aspect_terms.results
        case "contributions":
            return aspect_terms.contributions
        case _:
            raise ValueError(f"Unknown aspect name: {name!r}")


def encode_docid(
    terms: PaperTerms,
    *,
    max_paper_terms: int | None = None,
    no_aspects: bool = False,
    max_aspect_terms: int | None = None,
) -> str:
    """Encode a single paper's terms into a DocID string.

    Format: `<PAPER> t1 ; t2 <END_PAPER> <ASPECT=X> t3 ; t4 <END_ASPECT> ...`

    Empty aspect sections are omitted.  The paper section is always present.
    When truncation parameters are supplied, terms are truncated before encoding.

    Args:
        terms: Extracted keyphrases for one paper.
        max_paper_terms: Maximum number of paper-level terms to include.
            `None` (default) includes all terms.
        no_aspects: If `True`, all aspect sections are omitted entirely.
        max_aspect_terms: Maximum terms per aspect section.
            `None` (default) includes all terms.  Ignored when `no_aspects=True`.

    Returns:
        The encoded DocID string.
    """
    if not terms.paper_terms:
        logger.warning("Paper %d has no paper-level terms", terms.corpusid)

    parts = [
        PAPER_START,
        TERM_SEPARATOR.join(terms.paper_terms[:max_paper_terms]),
        PAPER_END,
    ]

    if not no_aspects:
        for aspect_name in ASPECT_ORDER:
            aspect = get_aspect_terms(terms.aspect_terms, aspect_name)
            if not aspect:
                continue

            aspect_terms = aspect[:max_aspect_terms]
            if not aspect_terms:
                continue

            parts.extend(
                (
                    _aspect_tag(aspect_name),
                    TERM_SEPARATOR.join(aspect_terms),
                    ASPECT_END,
                )
            )

    return " ".join(parts)


def encode_all_docids(
    terms_path: Path,
    *,
    filter_ids: set[CorpusId] | None = None,
    max_paper_terms: int | None = None,
    no_aspects: bool = False,
    max_aspect_terms: int | None = None,
) -> list[DocIDRecord]:
    """Read terms.jsonl and encode papers into DocIDRecords.

    Args:
        terms_path: Path to the terms.jsonl file.
        filter_ids: If provided, only encode papers with these corpus IDs.
        max_paper_terms: Maximum number of paper-level terms per DocID.
            `None` (default) includes all terms.
        no_aspects: If `True`, aspect sections are omitted from all DocIDs.
        max_aspect_terms: Maximum terms per aspect section.
            `None` (default) includes all terms.  Ignored when `no_aspects=True`.

    Returns:
        List of DocIDRecord with encoded docid strings.
    """
    records: list[DocIDRecord] = []
    seen: set[CorpusId] = set()

    for raw in track_progress(iter_jsonl(terms_path), desc="Encoding DocIDs..."):
        terms = PaperTerms.model_validate(raw)

        if filter_ids is not None and terms.corpusid not in filter_ids:
            continue

        if terms.corpusid in seen:
            logger.warning("Duplicate corpus ID %d — using latest", terms.corpusid)
            records = [r for r in records if r.corpusid != terms.corpusid]

        seen.add(terms.corpusid)
        records.append(
            DocIDRecord(
                corpusid=terms.corpusid,
                docid=encode_docid(
                    terms,
                    max_paper_terms=max_paper_terms,
                    no_aspects=no_aspects,
                    max_aspect_terms=max_aspect_terms,
                ),
            )
        )

    return records


# -- Index building --


@dataclass(frozen=True, kw_only=True)
class InvertedIndex:
    """Inverted index mapping between token IDs and paper corpus IDs.

    Built from tokenised DocID strings.  Used as an intermediate representation
    before serialisation to `DocIDIndex`.

    Attributes:
        token_to_papers: Maps each token ID to the set of corpus IDs whose DocID
            contains that token.  Used during constrained decoding to determine which
            papers remain reachable after generating a token.
        paper_to_tokens: Maps each corpus ID to the ordered list of token IDs in its
            DocID. Preserves the original tokenisation order.
        vocabulary: The set of all distinct token IDs across all DocIDs.
    """

    token_to_papers: dict[int, set[CorpusId]]
    paper_to_tokens: dict[CorpusId, list[int]]
    vocabulary: set[int]


def build_inverted_index(
    records: Sequence[DocIDRecord],
    token_seqs: Sequence[list[int]],
) -> InvertedIndex:
    """Build an inverted index from tokenised DocIDs.

    Args:
        records: DocIDRecord sequence (for corpus IDs).
        token_seqs: Corresponding token sequences from `tokenise_docids`.

    Returns:
        An InvertedIndex with token-paper mappings and vocabulary.

    Raises:
        ValueError if `records` and `token_seqs` are not the same length.
    """
    if len(records) != len(token_seqs):
        raise ValueError("`records` and `token_seqs` must have the same length")

    token_to_papers: dict[int, set[CorpusId]] = defaultdict(set)
    paper_to_tokens: dict[CorpusId, list[int]] = {}
    vocabulary: set[int] = set()

    for record, tokens in zip(records, token_seqs):
        paper_to_tokens[record.corpusid] = tokens
        vocabulary.update(tokens)

        for tok in tokens:
            token_to_papers[tok].add(record.corpusid)

    return InvertedIndex(
        token_to_papers=token_to_papers,
        paper_to_tokens=paper_to_tokens,
        vocabulary=vocabulary,
    )


def compute_index_stats(index: InvertedIndex) -> IndexStats:
    """Compute summary statistics for the inverted index.

    Args:
        index: The inverted index to summarise.

    Returns:
        IndexStats with vocabulary size, paper count, and token distribution stats.
    """
    num_papers = len(index.paper_to_tokens)
    token_counts = [len(toks) for toks in index.paper_to_tokens.values()]
    papers_per_token = [len(papers) for papers in index.token_to_papers.values()]

    return IndexStats(
        vocabulary_size=len(index.vocabulary),
        num_papers=num_papers,
        mean_tokens_per_docid=(
            round(sum(token_counts) / num_papers, 1) if num_papers else 0.0
        ),
        max_tokens_per_docid=max(token_counts, default=0),
        mean_papers_per_token=(
            round(sum(papers_per_token) / len(index.vocabulary), 1)
            if index.vocabulary
            else 0.0
        ),
    )


def build_docid_index(index: InvertedIndex, tokeniser_name: str) -> DocIDIndex:
    """Build a complete DocIDIndex with stats, vocabulary, and mappings.

    String keys and sorted values ensure deterministic output.

    Args:
        index: The inverted index to serialise.
        tokeniser_name: Name of the tokeniser used.

    Returns:
        A DocIDIndex ready for JSON serialisation.
    """
    stats = compute_index_stats(index)

    return DocIDIndex(
        tokeniser=tokeniser_name,
        stats=stats,
        vocabulary=sorted(index.vocabulary),
        token_to_papers={
            str(tok): sorted(papers)
            for tok, papers in sorted(index.token_to_papers.items())
        },
        paper_to_tokens={
            str(cid): list(toks) for cid, toks in sorted(index.paper_to_tokens.items())
        },
    )


# -- CLI commands --


@app.command(name="encode")
def cmd_encode(
    splits_dir: Annotated[
        Path, typer.Argument(help="Splits directory with train/dev/test.txt")
    ],
    max_paper_terms: Annotated[
        int,
        typer.Option(
            "--max-paper-terms",
            help="Maximum paper-level terms per DocID (0 = all).",
        ),
    ] = 0,
    no_aspects: Annotated[
        bool,
        typer.Option(
            "--no-aspects/--aspects",
            help="Omit all aspect sections from DocIDs.",
        ),
    ] = False,
    max_aspect_terms: Annotated[
        int,
        typer.Option(
            "--max-aspect-terms",
            help="Maximum terms per aspect section (0 = all). Ignored with --no-aspects.",
        ),
    ] = 0,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing docids.jsonl.")
    ] = False,
) -> None:
    """Encode PaperTerms into deterministic DocID strings.

    Reads processed/terms.jsonl and writes processed/docids.jsonl.
    """
    if force:
        confirm_force()

    if not splits_dir.exists():
        die(f"Splits directory not found: {splits_dir}")

    processed_dir = splits_dir / "processed"
    terms_path = resolve_jsonl(processed_dir / "terms.jsonl.gz")
    docids_path = processed_dir / "docids.jsonl.zst"

    if not terms_path.exists():
        die(f"Terms not found: {terms_path}\nRun 'term-extraction extract' first.")

    if docids_path.exists() and not force:
        die(f"Output already exists: {docids_path}\nUse --force to overwrite.")

    if force:
        backup_files([docids_path])

    filter_ids = load_split_ids(splits_dir)
    if not filter_ids:
        die(f"No IDs found in {splits_dir}/{{train,dev,test}}.txt")

    max_paper_terms_opt = max_paper_terms if max_paper_terms > 0 else None
    max_aspect_terms_opt = max_aspect_terms if max_aspect_terms > 0 else None

    logger.info(
        "=== Encoding DocIDs ===\n"
        "  Splits: %s\n"
        "  Papers: %d\n"
        "  Max paper terms: %s\n"
        "  Aspects: %s\n"
        "  Max aspect terms: %s",
        splits_dir,
        len(filter_ids),
        max_paper_terms_opt if max_paper_terms_opt is not None else "all",
        "omitted" if no_aspects else "included",
        max_aspect_terms_opt if max_aspect_terms_opt is not None else "all",
    )

    records = encode_all_docids(
        terms_path,
        filter_ids=filter_ids,
        max_paper_terms=max_paper_terms_opt,
        no_aspects=no_aspects,
        max_aspect_terms=max_aspect_terms_opt,
    )

    with LineWriter.open(docids_path) as writer:
        writer.write_objs(records)

    logger.info("Wrote %d DocIDs to %s", len(records), docids_path)


@app.command(name="build-index")
def cmd_build_index(
    splits_dir: Annotated[
        Path, typer.Argument(help="Splits directory with train/dev/test.txt")
    ],
    tokeniser_name: Annotated[
        str,
        typer.Option("--tokeniser", "-t", help="HuggingFace tokeniser name."),
    ] = "meta-llama/Llama-3.1-8B",
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing index.")
    ] = False,
) -> None:
    """Tokenise DocIDs and build an inverted index for constrained decoding.

    Reads processed/docids.jsonl and writes processed/docid_index.json.
    """
    if force:
        confirm_force()

    if not splits_dir.exists():
        die(f"Splits directory not found: {splits_dir}")

    processed_dir = splits_dir / "processed"
    docids_path = resolve_jsonl(processed_dir / "docids.jsonl.gz")
    index_path = processed_dir / "docid_index.json"

    if not docids_path.exists():
        die(f"DocIDs not found: {docids_path}\nRun 'docid encode' first.")

    if index_path.exists() and not force:
        die(f"Output already exists: {index_path}\nUse --force to overwrite.")

    if force:
        backup_files([index_path])

    logger.info("=== Building DocID Index ===\nTokeniser: %s", tokeniser_name)

    # Load DocID records
    records = [
        DocIDRecord.model_validate(raw)
        for raw in track_progress(iter_jsonl(docids_path), desc="Loading DocIDs...")
    ]
    logger.info("Loaded %d DocID records", len(records))

    # Tokenise
    tokeniser = load_tokeniser(tokeniser_name, special_tokens=SPECIAL_TOKENS)
    docid_strings = [r.docid for r in records]
    logger.info("Tokenising %d DocIDs...", len(docid_strings))
    token_seqs = tokenise_batch(docid_strings, tokeniser)

    # Build and write index
    inv_index = build_inverted_index(records, token_seqs)
    index = build_docid_index(inv_index, tokeniser_name)
    index_path.write_text(index.model_dump_json() + "\n", encoding="utf-8")

    logger.info(
        "Index written to %s\n"
        "  Papers: %d\n"
        "  Vocabulary: %d tokens\n"
        "  Mean tokens/DocID: %.1f\n"
        "  Max tokens/DocID: %d\n"
        "  Mean papers/token: %.1f",
        index_path,
        index.stats.num_papers,
        index.stats.vocabulary_size,
        index.stats.mean_tokens_per_docid,
        index.stats.max_tokens_per_docid,
        index.stats.mean_papers_per_token,
    )
