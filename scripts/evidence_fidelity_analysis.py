"""Analyse evidence fidelity from facet evaluation output.

Loads evaluation JSONL and paper graphs, re-runs fidelity checks on each
evidence string, and prints a structured diagnostic report showing where
models deviate from verbatim paper text: coverage distribution, per-facet
breakdown, worst papers, example discrepancies, and length analysis.
"""

import dataclasses
import difflib
import statistics
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from genret.paper_parsing import PaperGraph
from genret.processing.facet_evaluation import (
    EVIDENCE_FIDELITY_THRESHOLD,
    EVIDENCE_MIN_MATCH_BLOCK,
    FACET_NAMES,
    PaperEvaluation,
    best_paragraph_coverage,
    extract_paragraphs,
    normalise_ws,
)
from genret.types import CorpusId
from genret.utils import iter_jsonl, make_app, open_auto

app = make_app("Analyse evidence fidelity from facet evaluation output")

_console = Console()


# ── Data Model ────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvidenceRecord:
    """Per-evidence-string fidelity data."""

    corpusid: CorpusId
    facet_type: str
    item_text: str
    evidence: str
    is_verbatim: bool
    coverage: float
    best_paragraph: str


# ── Loading ───────────────────────────────────────────────────────────────────


def load_papers(path: Path) -> dict[CorpusId, PaperGraph]:
    """Load paper graphs into a dict keyed by corpus ID."""
    papers: dict[CorpusId, PaperGraph] = {}
    for record in iter_jsonl(path):
        pg = PaperGraph.model_validate(record)
        papers[CorpusId(pg.corpusid)] = pg
    return papers


# ── Core Logic ────────────────────────────────────────────────────────────────


def find_best_paragraph(
    normed_evidence: str,
    normed_paragraphs: list[str],
    raw_paragraphs: list[str],
) -> tuple[bool, float, str]:
    """Find the best-matching paragraph for an evidence string.

    Wraps ``best_paragraph_coverage`` and also returns the raw text of the
    best-matching paragraph.

    Args:
        normed_evidence: Whitespace-normalised evidence string.
        normed_paragraphs: Pre-normalised paragraph texts.
        raw_paragraphs: Original paragraph texts (same order as normed).

    Returns:
        ``(is_verbatim, coverage, best_raw_paragraph)``.  When no paragraphs
        are provided or the evidence is empty, returns ``("", False, 0.0)``.
    """
    if not normed_evidence or not normed_paragraphs:
        return False, 0.0, ""

    is_verbatim, coverage = best_paragraph_coverage(normed_evidence, normed_paragraphs)

    # Find the paragraph that gave the best result
    best_idx = 0
    best_cov = 0.0
    for i, np_ in enumerate(normed_paragraphs):
        if normed_evidence in np_:
            best_idx = i
            break
        _, cov = best_paragraph_coverage(normed_evidence, [np_])
        if cov > best_cov:
            best_cov = cov
            best_idx = i

    return is_verbatim, coverage, raw_paragraphs[best_idx]


def build_evidence_records(
    evaluations: list[PaperEvaluation],
    papers: dict[CorpusId, PaperGraph],
) -> list[EvidenceRecord]:
    """Build per-evidence-string fidelity records from evaluations.

    For each item's model evidence, calls ``normalise_ws`` +
    ``best_paragraph_coverage`` against the paper's paragraphs.

    Args:
        evaluations: Loaded paper evaluation records.
        papers: Paper graphs keyed by corpus ID.

    Returns:
        List of ``EvidenceRecord`` instances, one per evidence string.
    """
    records: list[EvidenceRecord] = []

    for ev in evaluations:
        cid = CorpusId(ev.corpusid)
        paper = papers.get(cid)
        if paper is None:
            continue

        paragraphs = extract_paragraphs(paper)
        raw_paras = [text for _, text in paragraphs]
        normed_paras = [normalise_ws(p) for p in raw_paras]

        for facet_name in FACET_NAMES:
            items = ev.item_metrics.get(facet_name, [])
            for item in items:
                for evidence_str in item.model_evidence:
                    normed_ev = normalise_ws(evidence_str)
                    is_verbatim, coverage, best_para = find_best_paragraph(
                        normed_ev, normed_paras, raw_paras
                    )
                    records.append(
                        EvidenceRecord(
                            corpusid=cid,
                            facet_type=facet_name,
                            item_text=item.text,
                            evidence=evidence_str,
                            is_verbatim=is_verbatim,
                            coverage=coverage,
                            best_paragraph=best_para,
                        )
                    )

    return records


# ── Diff Display ──────────────────────────────────────────────────────────────


def inline_diff(evidence: str, paragraph: str, context: int) -> str:
    """Produce a Rich-markup inline diff between evidence and paragraph.

    Uses ``difflib.SequenceMatcher`` to align the two strings, then shows
    a window around the evidence match region with deletions (in evidence
    but not paragraph) in red and insertions (in paragraph but not evidence)
    in green.

    Args:
        evidence: The evidence string.
        paragraph: The best-matching paragraph.
        context: Characters of context around the diff region.

    Returns:
        A string with Rich markup highlighting differences.
    """
    normed_ev = normalise_ws(evidence)
    normed_para = normalise_ws(paragraph)

    matcher = difflib.SequenceMatcher(None, normed_ev, normed_para, autojunk=False)

    # Find the region of the paragraph that overlaps with the evidence
    blocks = matcher.get_matching_blocks()
    if not blocks or (len(blocks) == 1 and blocks[0].size == 0):
        return "[dim](no matching region)[/dim]"

    # Build inline diff from opcodes
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(_escape_markup(normed_ev[i1:i2]))
        elif tag == "replace":
            parts.append(f"[red]{_escape_markup(normed_ev[i1:i2])}[/red]")
            parts.append(f"[green]{_escape_markup(normed_para[j1:j2])}[/green]")
        elif tag == "delete":
            parts.append(f"[red]{_escape_markup(normed_ev[i1:i2])}[/red]")
        elif tag == "insert":
            parts.append(f"[green]{_escape_markup(normed_para[j1:j2])}[/green]")

    result = "".join(parts)

    # Trim to context window if too long
    if len(result) > context * 4:
        result = result[: context * 4] + "[dim]...[/dim]"

    return result


def _escape_markup(text: str) -> str:
    """Escape Rich markup special characters in text."""
    return text.replace("[", r"\[")


# ── Report Sections ───────────────────────────────────────────────────────────


def _print_overview(records: list[EvidenceRecord]) -> None:
    """Print overview statistics."""
    total = len(records)
    if total == 0:
        _console.print("[bold red]No evidence records found.[/bold red]")
        return

    verbatim = sum(1 for r in records if r.is_verbatim)
    faithful = sum(1 for r in records if r.coverage >= EVIDENCE_FIDELITY_THRESHOLD)
    unfaithful = total - faithful
    coverages = [r.coverage for r in records]

    _console.print("\n[bold underline]1. Overview[/bold underline]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("Total evidence strings", f"{total:,}")
    table.add_row("Verbatim", f"{verbatim:,} ({100 * verbatim / total:.1f}%)")
    table.add_row(
        f"Faithful (>= {EVIDENCE_FIDELITY_THRESHOLD * 100:.0f}%)",
        f"{faithful:,} ({100 * faithful / total:.1f}%)",
    )
    table.add_row("Unfaithful", f"{unfaithful:,} ({100 * unfaithful / total:.1f}%)")
    table.add_row("", "")
    table.add_row("Mean coverage", f"{statistics.mean(coverages):.3f}")
    table.add_row("Median coverage", f"{statistics.median(coverages):.3f}")
    sorted_cov = sorted(coverages)
    n = len(sorted_cov)
    table.add_row("P10 coverage", f"{sorted_cov[max(0, int(n * 0.1))]:.3f}")
    table.add_row("P90 coverage", f"{sorted_cov[min(n - 1, int(n * 0.9))]:.3f}")
    _console.print(table)


def _print_coverage_distribution(records: list[EvidenceRecord]) -> None:
    """Print ASCII histogram of coverage values."""
    if not records:
        return

    _console.print("\n[bold underline]2. Coverage Distribution[/bold underline]\n")

    bins = [0] * 10
    for r in records:
        idx = min(int(r.coverage * 10), 9)
        bins[idx] += 1

    max_count = max(bins) if bins else 1
    bar_width = 40

    for i in range(10):
        lo = i / 10
        hi = (i + 1) / 10
        count = bins[i]
        bar_len = int(bar_width * count / max_count) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        _console.print(f"  {lo:.1f}-{hi:.1f}  {count:>6,}  {bar}")


def _print_per_facet_breakdown(records: list[EvidenceRecord]) -> None:
    """Print per-facet-type breakdown table."""
    if not records:
        return

    _console.print("\n[bold underline]3. Per-Facet Breakdown[/bold underline]\n")

    facet_data: dict[str, list[EvidenceRecord]] = {}
    for r in records:
        facet_data.setdefault(r.facet_type, []).append(r)

    table = Table()
    table.add_column("Facet type")
    table.add_column("Count", justify="right")
    table.add_column("Verbatim %", justify="right")
    table.add_column("Faithful %", justify="right")
    table.add_column("Mean cov.", justify="right")

    rows: list[tuple[str, int, float, float, float]] = []
    for facet, recs in facet_data.items():
        n = len(recs)
        verb_pct = 100 * sum(1 for r in recs if r.is_verbatim) / n
        faith_pct = (
            100 * sum(1 for r in recs if r.coverage >= EVIDENCE_FIDELITY_THRESHOLD) / n
        )
        mean_cov = statistics.mean(r.coverage for r in recs)
        rows.append((facet, n, verb_pct, faith_pct, mean_cov))

    # Sort by mean coverage ascending (worst first)
    rows.sort(key=lambda x: x[4])

    for facet, n, verb_pct, faith_pct, mean_cov in rows:
        table.add_row(
            facet,
            f"{n:,}",
            f"{verb_pct:.1f}%",
            f"{faith_pct:.1f}%",
            f"{mean_cov:.3f}",
        )

    _console.print(table)


def _print_worst_papers(records: list[EvidenceRecord], top_n: int) -> None:
    """Print top-N papers with lowest mean coverage."""
    if not records:
        return

    _console.print(
        f"\n[bold underline]4. Worst Papers (top {top_n})[/bold underline]\n"
    )

    paper_data: dict[CorpusId, list[EvidenceRecord]] = {}
    for r in records:
        paper_data.setdefault(r.corpusid, []).append(r)

    paper_stats: list[tuple[CorpusId, int, float, int, int, int]] = []
    for cid, recs in paper_data.items():
        n = len(recs)
        mean_cov = statistics.mean(r.coverage for r in recs)
        verb = sum(1 for r in recs if r.is_verbatim)
        faith = sum(1 for r in recs if r.coverage >= EVIDENCE_FIDELITY_THRESHOLD)
        unfaith = n - faith
        paper_stats.append((cid, n, mean_cov, verb, faith, unfaith))

    # Sort by mean coverage ascending
    paper_stats.sort(key=lambda x: x[2])

    table = Table()
    table.add_column("Corpus ID", justify="right")
    table.add_column("Evidence", justify="right")
    table.add_column("Mean cov.", justify="right")
    table.add_column("Verbatim", justify="right")
    table.add_column("Faithful", justify="right")
    table.add_column("Unfaithful", justify="right")

    for cid, n, mean_cov, verb, faith, unfaith in paper_stats[:top_n]:
        table.add_row(
            str(cid),
            str(n),
            f"{mean_cov:.3f}",
            str(verb),
            str(faith),
            str(unfaith),
        )

    _console.print(table)


def _print_example_discrepancies(
    records: list[EvidenceRecord],
    top_n: int,
    diff_context: int,
) -> None:
    """Print top-N lowest-coverage non-verbatim evidence with inline diffs."""
    if not records:
        return

    _console.print(
        f"\n[bold underline]5. Example Discrepancies (top {top_n})[/bold underline]\n"
    )

    non_verbatim = [r for r in records if not r.is_verbatim]
    non_verbatim.sort(key=lambda r: r.coverage)

    for i, r in enumerate(non_verbatim[:top_n]):
        _console.print(
            f"  [bold]#{i + 1}[/bold]  facet=[cyan]{r.facet_type}[/cyan]  "
            f"coverage=[yellow]{r.coverage:.3f}[/yellow]  "
            f"corpus=[dim]{r.corpusid}[/dim]"
        )
        _console.print(
            f"    [bold]Evidence:[/bold] {_escape_markup(r.evidence[:120])}"
            + ("..." if len(r.evidence) > 120 else "")
        )
        if r.best_paragraph:
            diff = inline_diff(r.evidence, r.best_paragraph, diff_context)
            _console.print(f"    [bold]Diff:[/bold]     {diff}")
        _console.print()


def _print_length_analysis(records: list[EvidenceRecord]) -> None:
    """Print length analysis: verbatim vs non-verbatim and correlation."""
    if not records:
        return

    _console.print("\n[bold underline]6. Length Analysis[/bold underline]\n")

    verbatim_lens = [len(r.evidence) for r in records if r.is_verbatim]
    non_verbatim_lens = [len(r.evidence) for r in records if not r.is_verbatim]

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(justify="right")

    if verbatim_lens:
        table.add_row(
            "Mean length (verbatim)",
            f"{statistics.mean(verbatim_lens):.1f} chars",
        )
    if non_verbatim_lens:
        table.add_row(
            "Mean length (non-verbatim)",
            f"{statistics.mean(non_verbatim_lens):.1f} chars",
        )

    # Coverage vs length correlation (Pearson)
    if len(records) > 2:
        lengths = [float(len(r.evidence)) for r in records]
        coverages = [r.coverage for r in records]
        mean_l = statistics.mean(lengths)
        mean_c = statistics.mean(coverages)
        n = len(records)
        cov_lc = sum((lengths[i] - mean_l) * (coverages[i] - mean_c) for i in range(n))
        var_l = sum((ln - mean_l) ** 2 for ln in lengths)
        var_c = sum((c - mean_c) ** 2 for c in coverages)
        denom = (var_l * var_c) ** 0.5
        correlation = cov_lc / denom if denom > 0 else 0.0
        table.add_row("", "")
        table.add_row(
            "Coverage-length correlation (Pearson r)",
            f"{correlation:.3f}",
        )

    _console.print(table)


# ── Report Orchestration ──────────────────────────────────────────────────────


def print_report(
    records: list[EvidenceRecord],
    top_n: int,
    diff_context: int,
) -> None:
    """Print the full diagnostic report to stdout.

    Args:
        records: All evidence fidelity records.
        top_n: Number of worst papers/examples to show.
        diff_context: Characters of context around diffs.
    """
    _console.print("[bold]Evidence Fidelity Analysis[/bold]")
    _console.print(
        f"Min match block: {EVIDENCE_MIN_MATCH_BLOCK}  "
        f"Fidelity threshold: {EVIDENCE_FIDELITY_THRESHOLD}"
    )

    _print_overview(records)
    _print_coverage_distribution(records)
    _print_per_facet_breakdown(records)
    _print_worst_papers(records, top_n)
    _print_example_discrepancies(records, top_n, diff_context)
    _print_length_analysis(records)


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def main(
    eval_path: Annotated[
        Path, typer.Argument(help="Path to facets_evaluation.jsonl.zst")
    ],
    graphs_path: Annotated[
        Path, typer.Argument(help="Path to paper_graphs.jsonl (.gz/.zst)")
    ],
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Number of worst papers/examples to show."),
    ] = 10,
    diff_context: Annotated[
        int,
        typer.Option("--diff-context", help="Characters of context around diffs."),
    ] = 20,
) -> None:
    """Analyse evidence fidelity from facet evaluation output."""
    _console.print(f"Loading paper graphs from [cyan]{graphs_path}[/cyan]...")
    papers = load_papers(graphs_path)
    _console.print(f"  Loaded {len(papers):,} papers")

    _console.print(f"Loading evaluations from [cyan]{eval_path}[/cyan]...")
    evaluations: list[PaperEvaluation] = []
    with open_auto(eval_path, "r") as fh:
        for record in iter_jsonl(fh):
            evaluations.append(PaperEvaluation.model_validate(record))
    _console.print(f"  Loaded {len(evaluations):,} evaluations")

    _console.print("Building evidence records...")
    records = build_evidence_records(evaluations, papers)
    _console.print(f"  Built {len(records):,} evidence records\n")

    print_report(records, top_n, diff_context)


if __name__ == "__main__":
    app()
