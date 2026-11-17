"""Compute agreement between GPT-5-mini and local model LLM-as-judge results.

Loads two JSONL files of JudgedPair records (keyed by paper_a|paper_b),
computes binary agreement, Cohen's kappa, and confusion matrices.

Usage:
    uv run python scripts/judge_agreement.py \
        --reference data/.../llm_judgements.jsonl \
        --candidate output/llm_judge/llm_judgements_qwen3-8b.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from genret.utils import make_app

app = make_app("Compare agreement between GPT-5-mini and local judge outputs")
_console = Console()


def _load_judgements(path: Path) -> dict[str, bool]:
    """Load judgements as {paper_a|paper_b: related} mapping."""
    judgements: dict[str, bool] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = f"{record['paper_a']}|{record['paper_b']}"
            judgements[key] = record["related"]
    return judgements


def _cohens_kappa(ref_labels: list[bool], cand_labels: list[bool]) -> float:
    """Compute Cohen's kappa for binary labels."""
    n = len(ref_labels)
    if n == 0:
        return 0.0

    agree = sum(r == c for r, c in zip(ref_labels, cand_labels, strict=True))
    p_o = agree / n

    ref_pos = sum(ref_labels) / n
    cand_pos = sum(cand_labels) / n
    p_e = ref_pos * cand_pos + (1 - ref_pos) * (1 - cand_pos)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


@app.command()
def compare(
    reference: Annotated[
        Path,
        typer.Option("--reference", "-r", help="Reference judgements (GPT-5-mini)."),
    ],
    candidate: Annotated[
        Path,
        typer.Option("--candidate", "-c", help="Candidate judgements (local model)."),
    ],
) -> None:
    """Compare two sets of LLM-as-judge results."""
    ref = _load_judgements(reference)
    cand = _load_judgements(candidate)

    common_keys = set(ref) & set(cand)
    _console.print(f"\nReference: {reference.name} ({len(ref)} pairs)")
    _console.print(f"Candidate: {candidate.name} ({len(cand)} pairs)")
    _console.print(f"Common pairs: {len(common_keys)}")

    if not common_keys:
        _console.print("[red]No common pairs found![/red]")
        sys.exit(1)

    ref_labels = [ref[k] for k in sorted(common_keys)]
    cand_labels = [cand[k] for k in sorted(common_keys)]

    # Confusion matrix
    tp = sum(r and c for r, c in zip(ref_labels, cand_labels, strict=True))
    tn = sum(not r and not c for r, c in zip(ref_labels, cand_labels, strict=True))
    fp = sum(not r and c for r, c in zip(ref_labels, cand_labels, strict=True))
    fn = sum(r and not c for r, c in zip(ref_labels, cand_labels, strict=True))

    agreement = (tp + tn) / len(common_keys)
    kappa = _cohens_kappa(ref_labels, cand_labels)

    _console.print()

    # Confusion matrix table
    cm_table = Table(title="Confusion Matrix (ref \\ cand)")
    cm_table.add_column("", style="bold")
    cm_table.add_column("Cand: Related")
    cm_table.add_column("Cand: Not Related")
    cm_table.add_row("Ref: Related", str(tp), str(fn))
    cm_table.add_row("Ref: Not Related", str(fp), str(tn))
    _console.print(cm_table)

    # Summary statistics
    _console.print()
    stats_table = Table(title="Agreement Statistics")
    stats_table.add_column("Metric")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Raw agreement", f"{agreement:.1%}")
    stats_table.add_row("Cohen's kappa", f"{kappa:.3f}")
    stats_table.add_row("Ref % related", f"{sum(ref_labels) / len(ref_labels):.1%}")
    stats_table.add_row("Cand % related", f"{sum(cand_labels) / len(cand_labels):.1%}")
    _console.print(stats_table)

    # Per-source breakdown
    _console.print()
    source_ref = {
        "retrieved": list[bool](),
        "random_negative": list[bool](),
    }
    source_cand = {
        "retrieved": list[bool](),
        "random_negative": list[bool](),
    }

    # Reload with source info
    ref_full: dict[str, tuple[bool, str]] = {}
    with open(reference, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = f"{record['paper_a']}|{record['paper_b']}"
            ref_full[key] = (record["related"], record.get("source", "unknown"))

    for key in sorted(common_keys):
        _, source = ref_full.get(key, (False, "unknown"))
        if source in source_ref:
            source_ref[source].append(ref[key])
            source_cand[source].append(cand[key])

    source_table = Table(title="Per-Source Agreement")
    source_table.add_column("Source")
    source_table.add_column("N", justify="right")
    source_table.add_column("Agreement", justify="right")
    source_table.add_column("Kappa", justify="right")
    source_table.add_column("Ref % related", justify="right")
    source_table.add_column("Cand % related", justify="right")

    for source in ["retrieved", "random_negative"]:
        r = source_ref[source]
        c = source_cand[source]
        if not r:
            continue
        src_agree = sum(a == b for a, b in zip(r, c, strict=True)) / len(r)
        src_kappa = _cohens_kappa(r, c)
        source_table.add_row(
            source,
            str(len(r)),
            f"{src_agree:.1%}",
            f"{src_kappa:.3f}",
            f"{sum(r) / len(r):.1%}",
            f"{sum(c) / len(c):.1%}",
        )

    _console.print(source_table)

    # Relationship type distribution (if available)
    _console.print()
    cand_types: Counter[str] = Counter()
    with open(candidate, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = f"{record['paper_a']}|{record['paper_b']}"
            if key in common_keys:
                cand_types[record.get("relationship_type", "unknown")] += 1

    if cand_types:
        type_table = Table(title="Candidate Relationship Types")
        type_table.add_column("Type")
        type_table.add_column("Count", justify="right")
        type_table.add_column("%", justify="right")
        total = sum(cand_types.values())
        for rtype, count in cand_types.most_common():
            type_table.add_row(rtype, str(count), f"{count / total:.1%}")
        _console.print(type_table)


if __name__ == "__main__":
    app()
