"""Compute self-containedness metrics for a dataset and its splits.

SC_out: fraction of outgoing references pointing to papers within the dataset.
SC_in:  fraction of incoming citations (by S2 total count) coming from within the
        dataset.

By default computes metrics for the full corpus. Pass --split to compute for a
specific split directory (train + dev + test combined) instead.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from genret.utils import iter_jsonl, make_app

app = make_app("Compute self-containedness metrics for a dataset and its splits")


@dataclass
class SCMetrics:
    """Self-containedness metrics for one paper set.

    Attributes:
        n_papers: Number of papers in the set.
        sc_out: Outgoing self-containedness (internal refs / total refs).
        internal_refs: References pointing to papers within the set.
        total_refs: Total outgoing references from papers in the set.
        sc_in: Incoming self-containedness (internal cites / total S2 cites).
        internal_cites: Within-set papers that cite papers in the set.
        total_cites: Total S2 citation count across all papers in the set.
    """

    n_papers: int
    sc_out: float
    internal_refs: int
    total_refs: int
    sc_in: float
    internal_cites: int
    total_cites: int


def load_citation_counts(data_dir: Path) -> dict[str, int]:
    """Load paper_id -> S2 citation count from papers/metadata.jsonl.gz.

    Papers with a null or missing citation_count are excluded.
    """
    counts: dict[str, int] = {}
    for record in iter_jsonl(data_dir / "papers" / "metadata.jsonl.gz"):
        pid = record.get("paper_id")
        cc = record.get("citation_count")
        if pid and cc is not None:
            counts[pid] = cc
    return counts


def load_references(data_dir: Path) -> dict[str, list[str]]:
    """Load paper_id -> list of referenced paper_ids from citations/references.jsonl.gz."""
    refs: dict[str, list[str]] = {}
    for record in iter_jsonl(data_dir / "citations" / "references.jsonl.gz"):
        pid = record.get("paper_id")
        if pid:
            refs[pid] = record.get("references", [])
    return refs


def load_corpus_citations(data_dir: Path) -> dict[str, list[str]]:
    """Load paper_id -> within-corpus citing paper_ids from citations/citations.jsonl.gz."""
    cites: dict[str, list[str]] = {}
    for record in iter_jsonl(data_dir / "citations" / "citations.jsonl.gz"):
        pid = record.get("paper_id")
        if pid:
            cites[pid] = record.get("citations", [])
    return cites


def load_split_ids(split_dir: Path) -> set[str]:
    """Load the union of paper_ids from train.txt, dev.txt, and test.txt."""
    ids: set[str] = set()
    for name in ("train.txt", "dev.txt", "test.txt"):
        path = split_dir / name
        if path.exists():
            ids.update(
                line.strip() for line in path.read_text().splitlines() if line.strip()
            )
    return ids


def compute_sc(
    paper_ids: set[str],
    references: dict[str, list[str]],
    corpus_citations: dict[str, list[str]],
    citation_counts: dict[str, int],
) -> SCMetrics:
    """Compute self-containedness metrics for a set of papers.

    Args:
        paper_ids: The set of paper IDs forming the "dataset".
        references: Full corpus map of paper_id -> list of referenced paper_ids.
        corpus_citations: Full corpus map of paper_id -> within-corpus citing paper_ids.
        citation_counts: Map of paper_id -> S2-wide citation count.

    Returns:
        SCMetrics with SC_out and SC_in and their constituent counts.
    """
    internal_refs = 0
    total_refs = 0
    for pid in paper_ids:
        for ref in references.get(pid, []):
            total_refs += 1
            if ref in paper_ids:
                internal_refs += 1

    internal_cites = 0
    total_cites = 0
    for pid in paper_ids:
        total_cites += citation_counts.get(pid, 0)
        for citer in corpus_citations.get(pid, []):
            if citer in paper_ids:
                internal_cites += 1

    sc_out = internal_refs / total_refs if total_refs else 0.0
    sc_in = internal_cites / total_cites if total_cites else 0.0

    return SCMetrics(
        n_papers=len(paper_ids),
        sc_out=sc_out,
        internal_refs=internal_refs,
        total_refs=total_refs,
        sc_in=sc_in,
        internal_cites=internal_cites,
        total_cites=total_cites,
    )


def print_metrics(m: SCMetrics) -> None:
    """Print a labelled metrics block to stdout."""
    lines = [
        f"Papers: {m.n_papers:>10,}",
        f"SC_out: {m.sc_out:>10.4f}  ({m.internal_refs:,} / {m.total_refs:,} refs)",
        f"SC_in:  {m.sc_in:>10.4f}  ({m.internal_cites:,} / {m.total_cites:,} citations)",
    ]
    print("\n".join(lines))


@app.command()
def main(
    data_dir: Annotated[Path, typer.Argument(help="Dataset root directory")],
    split: Annotated[
        Path | None,
        typer.Option(
            "--split",
            help="Split directory to evaluate (train+dev+test combined).",
        ),
    ] = None,
) -> None:
    """Compute self-containedness metrics for a dataset and its splits."""
    print("Loading metadata...")
    citation_counts = load_citation_counts(data_dir)
    references = load_references(data_dir)
    corpus_citations = load_corpus_citations(data_dir)

    if split is not None:
        paper_ids = load_split_ids(split)
    else:
        paper_ids = set(citation_counts) | set(references)

    print(f"{len(paper_ids):,} papers")
    metrics = compute_sc(paper_ids, references, corpus_citations, citation_counts)
    print_metrics(metrics)


if __name__ == "__main__":
    app()
