"""Pre-flight coverage checks for the D9 (bullet-level queries) experiment.

Reports three statistics needed before launching D9:

1. Bullets per (paper, facet) distribution — confirms how much the
   bullet-level query approach multiplies training data.
2. Title availability — confirms titles can be fetched for all corpus
   papers from paper_graphs.jsonl.zst.
3. Retrieval-target coverage — fraction of corpus papers that appear
   as ``paper_b`` (retrieval target) in at least one training pair.
   If most papers are reached, indexing can be safely dropped.

Run from project root:
    uv run python scripts/eval/d9_coverage_check.py
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

from genret.paper_parsing import FacetRecordType, Facets, facets_from_type
from genret.training.codebook_data import FACET_NAMES
from genret.types import CorpusId
from genret.utils import iter_jsonl


def _facet_bullets(facets: Facets, facet: str) -> list[str]:
    """Return the list of bullets for the given facet."""
    items = getattr(facets, facet)
    return [str(b).strip() for b in items if str(b).strip()]


def bullets_per_facet(facets_path: Path) -> dict[str, list[int]]:
    """Return {facet: [count_per_paper]} bullet counts."""
    counts: dict[str, list[int]] = {f: [] for f in FACET_NAMES}
    for record in iter_jsonl(facets_path):
        facets_obj = facets_from_type(record, FacetRecordType.STANDARD)
        for facet in FACET_NAMES:
            bullets = _facet_bullets(facets_obj, facet)
            counts[facet].append(len(bullets))
    return counts


def title_coverage(
    paper_graphs_path: Path, corpus_ids: set[CorpusId]
) -> tuple[int, int, list[CorpusId]]:
    """Return (with_title, without_title, missing_ids)."""
    have_title: set[CorpusId] = set()
    for record in iter_jsonl(paper_graphs_path):
        cid = CorpusId(int(record["corpusid"]))
        if cid not in corpus_ids:
            continue
        title = record.get("title")
        if isinstance(title, str) and title.strip():
            have_title.add(cid)
    missing = sorted(corpus_ids - have_title)
    return len(have_title), len(missing), missing[:10]


def retrieval_target_coverage(
    pairs_path: Path, corpus_ids: set[CorpusId]
) -> tuple[set[CorpusId], dict[str, int]]:
    """Return (paper_bs_seen_as_retrieval_target, per_facet_counts)."""
    retrieval_targets: set[CorpusId] = set()
    per_facet: dict[str, int] = dict.fromkeys(FACET_NAMES, 0)
    n_retrieval_rows = 0
    n_indexing_rows = 0
    for row in iter_jsonl(pairs_path):
        kind = str(row.get("kind", ""))
        if kind == "retrieval":
            n_retrieval_rows += 1
            cid_b = CorpusId(int(row["paper_b"]))
            if cid_b in corpus_ids:
                retrieval_targets.add(cid_b)
            facet = str(row.get("facet", ""))
            if facet in per_facet:
                per_facet[facet] += 1
        elif kind == "indexing":
            n_indexing_rows += 1
    return retrieval_targets, {
        "retrieval_rows": n_retrieval_rows,
        "indexing_rows": n_indexing_rows,
        **per_facet,
    }


def _print_distribution(label: str, values: list[int]) -> None:
    """Pretty-print a distribution summary."""
    if not values:
        print(f"  {label}: (empty)")
        return
    counter = Counter(values)
    zero_count = counter.get(0, 0)
    nonzero = [v for v in values if v > 0]
    print(f"  {label}:")
    print(
        f"    papers with 0 bullets: {zero_count} ({100 * zero_count / len(values):.1f}%)"
    )
    if nonzero:
        print(f"    mean (non-empty): {statistics.mean(nonzero):.2f}")
        print(f"    median (non-empty): {statistics.median(nonzero):.0f}")
        print(f"    min / max: {min(nonzero)} / {max(nonzero)}")
        print(f"    distribution top-10: {sorted(counter.items())[:10]}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--facets",
        type=Path,
        default=Path(
            "data/s2orc-nlp-v2/splits-connected-10k-v2/processed/facets.jsonl.zst"
        ),
    )
    parser.add_argument(
        "--paper-graphs",
        type=Path,
        default=Path(
            "data/s2orc-nlp-v2/splits-connected-10k-v2/processed/paper_graphs.jsonl.zst"
        ),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("output/genret/training_pairs.jsonl.zst"),
    )
    args = parser.parse_args()

    print("=" * 70)
    print("D9 coverage checks")
    print("=" * 70)

    # Collect corpus ids from facets
    corpus_ids: set[CorpusId] = set()
    for record in iter_jsonl(args.facets):
        corpus_ids.add(CorpusId(int(record["corpusid"])))
    print(f"\nCorpus papers (from facets): {len(corpus_ids)}")

    # 1. Bullets per (paper, facet)
    print("\n--- 1. Bullets per (paper, facet) ---")
    counts = bullets_per_facet(args.facets)
    for facet in FACET_NAMES:
        _print_distribution(facet, counts[facet])

    # Aggregate: average across-facet
    total_per_paper: list[int] = [
        sum(counts[f][i] for f in FACET_NAMES) for i in range(len(corpus_ids))
    ]
    avg_total = statistics.mean(total_per_paper)
    print(f"\n  Avg total bullets per paper (4 facets): {avg_total:.2f}")
    print(
        f"  Estimated query multiplier vs facet-level: ~{avg_total / 4:.2f}x "
        f"(per-facet queries -> per-bullet queries)"
    )

    # 2. Title availability
    print("\n--- 2. Title availability ---")
    if not args.paper_graphs.exists():
        print(f"  ERROR: {args.paper_graphs} not found")
    else:
        have, missing_n, sample_missing = title_coverage(args.paper_graphs, corpus_ids)
        coverage_pct = 100 * have / max(len(corpus_ids), 1)
        print(f"  Papers with title: {have} / {len(corpus_ids)} ({coverage_pct:.1f}%)")
        if missing_n:
            print(f"  Papers without title: {missing_n}")
            print(f"  Sample missing corpus ids: {sample_missing}")

    # 3. Retrieval-target coverage
    print("\n--- 3. Retrieval-target coverage ---")
    if not args.pairs.exists():
        print(f"  ERROR: {args.pairs} not found")
    else:
        targets, stats = retrieval_target_coverage(args.pairs, corpus_ids)
        coverage_pct = 100 * len(targets) / max(len(corpus_ids), 1)
        print(
            f"  Papers reached as retrieval target: "
            f"{len(targets)} / {len(corpus_ids)} ({coverage_pct:.1f}%)"
        )
        orphans = sorted(corpus_ids - targets)
        if orphans:
            print(f"  Orphan papers (never retrieval target): {len(orphans)}")
            print(f"  Sample orphan ids: {orphans[:10]}")
        print("\n  Pairs file row counts:")
        print(f"    retrieval rows: {stats['retrieval_rows']}")
        print(f"    indexing rows:  {stats['indexing_rows']}")
        for facet in FACET_NAMES:
            print(f"    retrieval rows ({facet}): {stats[facet]}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
