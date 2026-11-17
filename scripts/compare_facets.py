"""Compare two facet-extraction outputs with summary statistics.

Reports coverage, item counts, and item lengths per facet for two
``facets.jsonl.zst`` files side by side. Intended for comparing a local-model
extraction (e.g. Qwen3-32B) against the GPT baseline.

Usage:
    uv run python scripts/compare_facets.py \
        --a data/.../processed/facets.jsonl.zst --a-label gpt-5-mini \
        --b data/.../processed-qwen3-32b/facets.jsonl.zst --b-label qwen3-32b
"""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from genret.types import CorpusId
from genret.utils import iter_jsonl, make_app

_FACETS = ("problems", "methods", "results", "contributions")

app = make_app("Compare two facet extraction outputs")


@dataclass
class FacetStats:
    """Aggregate statistics for one facet across a corpus."""

    coverage: float  # fraction of papers with >= 1 item
    mean_items: float
    median_items: float
    mean_words: float  # mean words per item

    @staticmethod
    def compute(item_lists: Sequence[Sequence[str]]) -> FacetStats:
        """Compute facet stats from one list of items per paper."""
        counts = [len(items) for items in item_lists]
        words = [len(item.split()) for items in item_lists for item in items]
        covered = sum(1 for c in counts if c > 0)
        n = max(len(item_lists), 1)
        return FacetStats(
            coverage=covered / n,
            mean_items=statistics.mean(counts) if counts else 0.0,
            median_items=statistics.median(counts) if counts else 0.0,
            mean_words=statistics.mean(words) if words else 0.0,
        )


def load_facets(path: Path) -> dict[CorpusId, dict[str, list[str]]]:
    """Load a facets JSONL into {corpusid: {facet: items}}."""
    out: dict[CorpusId, dict[str, list[str]]] = {}
    for rec in iter_jsonl(path):
        out[CorpusId(rec["corpusid"])] = {f: list(rec.get(f, [])) for f in _FACETS}
    return out


def empty_record_rate(facets: dict[CorpusId, dict[str, list[str]]]) -> float:
    """Fraction of papers with zero items across ALL facets (parse failures)."""
    if not facets:
        return 0.0
    empty = sum(1 for r in facets.values() if not any(r[f] for f in _FACETS))
    return empty / len(facets)


def summarise(
    label: str, facets: dict[CorpusId, dict[str, list[str]]]
) -> dict[str, FacetStats]:
    """Compute and print per-facet stats for one extraction."""
    print(f"\n=== {label} ===")
    print(
        f"papers: {len(facets)}  |  fully-empty records: {empty_record_rate(facets):.1%}"
    )
    stats = {f: FacetStats.compute([r[f] for r in facets.values()]) for f in _FACETS}
    print(
        f"{'facet':<14}{'coverage':>10}{'mean_n':>9}{'median_n':>10}{'words/item':>12}"
    )
    for f in _FACETS:
        s = stats[f]
        print(
            f"{f:<14}{s.coverage:>9.1%}{s.mean_items:>9.2f}"
            f"{s.median_items:>10.1f}{s.mean_words:>12.1f}"
        )
    return stats


@app.command()
def main(
    a: Annotated[Path, typer.Option("--a", help="First facets.jsonl[.zst].")],
    b: Annotated[Path, typer.Option("--b", help="Second facets.jsonl[.zst].")],
    a_label: Annotated[str, typer.Option("--a-label")] = "A",
    b_label: Annotated[str, typer.Option("--b-label")] = "B",
) -> None:
    """Print side-by-side facet statistics for two extractions."""
    fa = load_facets(a)
    fb = load_facets(b)
    sa = summarise(a_label, fa)
    sb = summarise(b_label, fb)

    common = set(fa) & set(fb)
    print(
        f"\n=== overlap ===\ncommon papers: {len(common)} "
        f"(only in {a_label}: {len(set(fa) - set(fb))}, "
        f"only in {b_label}: {len(set(fb) - set(fa))})"
    )

    print("\n=== delta (B - A) per facet ===")
    print(f"{'facet':<14}{'Δcoverage':>11}{'Δmean_n':>10}{'Δwords/item':>13}")
    for f in _FACETS:
        print(
            f"{f:<14}{sb[f].coverage - sa[f].coverage:>10.1%}"
            f"{sb[f].mean_items - sa[f].mean_items:>10.2f}"
            f"{sb[f].mean_words - sa[f].mean_words:>13.1f}"
        )


if __name__ == "__main__":
    app()
