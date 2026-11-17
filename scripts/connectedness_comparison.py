"""Compare connectedness metrics between two split datasets.

Computes graph statistics (edges, degree distribution, components) and
self-containedness (SC_out) for both datasets side by side.
"""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from genret.pipeline.connected_sampling import (
    CitationGraph,
    analyse_selection,
)
from genret.pipeline.derive import load_metadata_years
from genret.types import CorpusId, Year
from genret.utils import load_references, load_split_ids, make_app, safediv

DATA_DIR = Path("data/s2orc-nlp")
"""Default data directory (parent of citations/ and papers/)."""

_console = Console()


def _load_data(
    data_dir: Path, splits_base: Path, splits_cmp: Path
) -> tuple[
    dict[CorpusId, list[CorpusId]], dict[CorpusId, Year], set[CorpusId], set[CorpusId]
]:
    """Load references, year metadata, and both split ID sets."""
    _console.print("[bold]Loading references...[/bold]")
    refs = load_references(data_dir / "citations" / "references.jsonl")

    _console.print("[bold]Loading metadata years...[/bold]")
    year_map = load_metadata_years(data_dir / "papers")

    _console.print("[bold]Loading split IDs...[/bold]")
    ids_base = load_split_ids(splits_base)
    ids_cmp = load_split_ids(splits_cmp)

    return refs, year_map, ids_base, ids_cmp


def _compute_sc_out(
    paper_ids: set[CorpusId], refs: dict[CorpusId, list[CorpusId]]
) -> tuple[float, int, int]:
    """Compute SC_out: fraction of outgoing references pointing within the set.

    Returns:
        (self-containedness, internal references, total references).
    """
    internal = 0
    total = 0

    for pid in paper_ids:
        for ref in refs.get(pid, []):
            total += 1
            if ref in paper_ids:
                internal += 1

    sc = safediv(internal, total)
    return sc, internal, total


app = make_app("Compare connectedness metrics between two split datasets")


@app.command()
def main(
    splits_base: Annotated[
        Path,
        typer.Option(
            "--baseline", help="Path to the first (baselines) splits directory."
        ),
    ],
    splits_cmp: Annotated[
        Path,
        typer.Option(
            "--comparison", help="Path to the second (comparison) splits directory."
        ),
    ],
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            help="Data directory containing citations/ and papers/.",
        ),
    ] = DATA_DIR,
) -> None:
    """Compare connectedness metrics between two split datasets."""
    label_base = splits_base.name
    label_cmp = splits_cmp.name

    refs, year_map, ids_base, ids_cmp = _load_data(data_dir, splits_base, splits_cmp)

    _console.print(f"\n[bold]{label_base}:[/bold] {len(ids_base):,} papers")
    _console.print(f"[bold]{label_cmp}:[/bold] {len(ids_cmp):,} papers\n")

    # Build graph from full universe
    _console.print("[bold]Building citation graph...[/bold]")
    all_ids = ids_base | ids_cmp
    graph = CitationGraph.from_citations(refs, all_ids)

    # Year quotas (just use actual distribution)
    years_base = Counter(year_map[pid] for pid in ids_base if pid in year_map)
    years_cmp = Counter(year_map[pid] for pid in ids_cmp if pid in year_map)

    # Analyse both
    _console.print(f"[bold]Analysing {label_base}...[/bold]")
    report_base = analyse_selection(graph, ids_base, year_map, dict(years_base))
    _console.print(f"[bold]Analysing {label_cmp}...[/bold]")
    report_comp = analyse_selection(graph, ids_cmp, year_map, dict(years_cmp))

    # SC_out
    _console.print("[bold]Computing self-containedness...[/bold]")
    sc_base, int_base, tot_base = _compute_sc_out(ids_base, refs)
    sc_cmp, int_cmp, tot_cmp = _compute_sc_out(ids_cmp, refs)

    # --- Results ---
    _console.print()
    table = Table(title="Connectedness Comparison", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column(label_base, justify="right")
    table.add_column(label_cmp, justify="right")
    table.add_column(f"Ratio ({label_cmp}/{label_base})", justify="right")

    def _ratio(a: float, b: float) -> str:
        return f"{a / b:.2f}x" if b else "—"

    table.add_row(
        "Papers",
        f"{report_base.total:,}",
        f"{report_comp.total:,}",
        f"{report_comp.total / report_base.total:.2%}",
    )
    table.add_row(
        "Edges (induced)",
        f"{report_base.edges:,}",
        f"{report_comp.edges:,}",
        _ratio(report_comp.edges, report_base.edges),
    )
    table.add_row(
        "Edge density",
        f"{2 * report_base.edges / (report_base.total * (report_base.total - 1)):.6f}",
        f"{2 * report_comp.edges / (report_comp.total * (report_comp.total - 1)):.6f}",
        _ratio(
            2 * report_comp.edges / (report_comp.total * (report_comp.total - 1)),
            2 * report_base.edges / (report_base.total * (report_base.total - 1)),
        ),
    )
    table.add_row(
        "Min degree",
        str(report_base.min_degree),
        str(report_comp.min_degree),
        "",
    )
    table.add_row(
        "Mean degree",
        f"{report_base.mean_degree:.2f}",
        f"{report_comp.mean_degree:.2f}",
        _ratio(report_comp.mean_degree, report_base.mean_degree),
    )
    table.add_row(
        "Max degree",
        f"{report_base.max_degree:,}",
        f"{report_comp.max_degree:,}",
        "",
    )
    table.add_row(
        "Components",
        str(report_base.components),
        str(report_comp.components),
        "",
    )
    table.add_row(
        "SC_out",
        f"{sc_base:.4f}  ({int_base:,}/{tot_base:,})",
        f"{sc_cmp:.4f}  ({int_cmp:,}/{tot_cmp:,})",
        _ratio(sc_cmp, sc_base),
    )

    _console.print(table)

    # Year distribution comparison
    _console.print()
    year_table = Table(title="Year Distribution", show_lines=True)
    year_table.add_column("Year", style="bold")
    year_table.add_column(label_base, justify="right")
    year_table.add_column(f"{label_base} %", justify="right")
    year_table.add_column(label_cmp, justify="right")
    year_table.add_column(f"{label_cmp} %", justify="right")

    all_years_sorted = sorted(set(years_base) | set(years_cmp))
    for year in all_years_sorted:
        c_base = years_base.get(year, 0)
        c_cmp = years_cmp.get(year, 0)
        year_table.add_row(
            str(year),
            f"{c_base:,}",
            f"{c_base / report_base.total * 100:.1f}%",
            f"{c_cmp:,}",
            f"{c_cmp / report_comp.total * 100:.1f}%",
        )

    _console.print(year_table)


if __name__ == "__main__":
    app()
