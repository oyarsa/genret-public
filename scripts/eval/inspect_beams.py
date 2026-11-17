# ruff: noqa: PLC0415
r"""Inspect raw GLEN beam outputs to diagnose beam collapse.

For a sample of queries, runs 4-facet constrained beam search and reports:
- Number of unique DocID leaves per facet and in total
- Top DocID strings (decoded) for each facet
- Leaf size distribution (how many papers behind each hit)
- Beam score distribution

This is a diagnostic tool — run before deciding whether to retrain.

Usage:
    uv run --extra cuda --extra training python scripts/eval/inspect_beams.py \
        --checkpoint output/genret/checkpoints/glen-llama-8b \
        --facets data/s2orc-nlp-v2/splits-connected-10k-v2/processed/facets.jsonl.zst \
        --glen-dir output/genret/glen_index \
        --queries output/eval_500/query_papers_500.txt \
        --n 10
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from genret.paper_parsing import FacetRecordType
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_dataset import _build_messages, _facet_text
from genret.training.glen_decoding import decode_per_facet_glen, load_glen_index
from genret.training.glen_evaluate import load_eval_inputs
from genret.types import CorpusId


def main(
    checkpoint: Annotated[Path, typer.Option(help="Fine-tuned GLEN checkpoint.")],
    facets: Annotated[Path, typer.Option(help="Facets JSONL.", show_default=False)],
    glen_dir: Annotated[Path, typer.Option(help="GLEN index directory.")],
    queries: Annotated[Path, typer.Option(help="Query corpus-id file.")],
    n: Annotated[int, typer.Option(help="Number of queries to inspect.")] = 10,
    beams: Annotated[int, typer.Option(help="Beams per facet.")] = 50,
    max_docid_tokens: Annotated[int, typer.Option()] = 32,
    seed: Annotated[int, typer.Option()] = 42,
    facets_type: Annotated[FacetRecordType, typer.Option()] = FacetRecordType.STANDARD,
) -> None:
    """Inspect raw beam outputs for beam-collapse diagnosis."""
    from genret.training.cli import _load_checkpoint

    index = load_glen_index(glen_dir)
    model, tokeniser = _load_checkpoint(checkpoint)
    model.eval()

    facets_map, evidence_map = load_eval_inputs(facets, facets_type)
    query_ids = [
        CorpusId(int(x)) for x in Path(queries).read_text().split() if x.strip()
    ]

    rng = random.Random(seed)
    sample = rng.sample(query_ids, min(n, len(query_ids)))

    for qid in sample:
        facets_obj = facets_map.get(qid)
        if facets_obj is None:
            continue
        evidence = evidence_map.get(qid) if evidence_map else None

        print(f"\n{'=' * 72}")
        print(f"Query: {qid}")

        all_leaf_papers: set[int] = set()

        for facet in FACET_NAMES:
            text = _facet_text(facets_obj, facet, evidence)
            if not text:
                print(f"\n  [{facet}] — no text")
                continue

            messages = _build_messages(facet, text, kind="retrieval")
            results = decode_per_facet_glen(
                model=model,
                tokeniser=tokeniser,
                messages=messages,
                facet=facet,
                index=index,
                num_beams=beams,
                max_new_tokens=max_docid_tokens,
            )

            unique_leaves = {r.docid_tokens for r in results}
            leaf_papers: set[int] = set()
            for r in results:
                leaf_papers.update(int(p) for p in r.paper_ids)
            all_leaf_papers.update(leaf_papers)

            scores = [r.score for r in results]
            score_range = f"{min(scores):.3f}..{max(scores):.3f}" if scores else "—"

            print(
                f"\n  [{facet}]  beams={len(results)}  unique_leaves={len(unique_leaves)}"
                f"  distinct_papers={len(leaf_papers)}  score_range={score_range}"
            )

            # Top-5 DocIDs by beam score
            top = sorted(results, key=lambda r: -r.score)[:5]
            for r in top:
                text_decoded = tokeniser.decode(
                    list(r.docid_tokens), skip_special_tokens=True
                )
                print(
                    f"    rank={r.rank:3d}  score={r.score:7.3f}"
                    f"  papers={len(r.paper_ids):3d}  docid={text_decoded!r:.60s}"
                )

            # Leaf-size distribution
            leaf_sizes = [len(r.paper_ids) for r in results]
            size_dist = sorted(Counter(leaf_sizes).items())
            print(f"    leaf_size_dist: {size_dist}")

        print(f"\n  TOTAL distinct papers across facets: {len(all_leaf_papers)}")


if __name__ == "__main__":
    typer.run(main)
