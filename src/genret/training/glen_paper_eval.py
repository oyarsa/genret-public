# pyright: basic
# ruff: noqa: PLC0415
"""Paper-level title-DocID eval for V1 / V1+V3 GLEN variants.

The standard ``evaluate_glen_retrieval`` decodes once per facet and fuses
with RRF, querying with per-facet bullet text. The V1 (paper-title) and
V1+V3 (paper-marginal) variants train on full paper title+abstract input
and a single title DocID per query, so eval here runs **one** decode per
query using the same prompt shape as training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from genret.training.glen_paper_marginal_dataset import (
    _build_paper_messages,
    _load_paper_metadata,
)
from genret.utils import open_auto

logger = logging.getLogger(__name__)


def cmd_evaluate_glen_paper(
    checkpoint: Annotated[
        Path, typer.Argument(help="Fine-tuned paper-level checkpoint dir.")
    ],
    paper_graphs: Annotated[
        Path,
        typer.Option("--paper-graphs", help="Per-paper metadata JSONL."),
    ],
    glen_dir: Annotated[
        Path,
        typer.Option("--glen-dir", help="Title GLEN index directory."),
    ],
    queries: Annotated[
        Path,
        typer.Option("--queries", help="Plain-text file of corpus ids."),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    beams: Annotated[int, typer.Option("--beams")] = 50,
    max_docid_tokens: Annotated[int, typer.Option("--max-docid-tokens")] = 32,
    top_n: Annotated[int, typer.Option("--top-n")] = 50,
    max_abstract_chars: Annotated[int, typer.Option("--max-abstract-chars")] = 1500,
) -> None:
    """Paper-level title beam search; one decode per query, no RRF."""
    from genret.training.cli import _load_checkpoint
    from genret.training.codebook_data import FACET_NAMES
    from genret.training.codebook_evaluate import queries_from_split
    from genret.training.glen_decoding import decode_per_facet_glen, load_glen_index

    index = load_glen_index(glen_dir)
    raw_titles, abstracts = _load_paper_metadata(
        paper_graphs, max_abstract_chars=max_abstract_chars
    )
    model, tokeniser = _load_checkpoint(checkpoint)
    model.eval()
    query_ids = queries_from_split(queries)
    facet = FACET_NAMES[0]  # title trie duplicated across all facet keys

    output.parent.mkdir(parents=True, exist_ok=True)
    n_queries = 0
    n_skipped = 0
    with open_auto(output, "w") as fh:
        for qid in query_ids:
            title = raw_titles.get(qid, "")
            if not title:
                n_skipped += 1
                continue
            abstract = abstracts.get(qid, "")
            messages = _build_paper_messages(title, abstract)
            beams_out = decode_per_facet_glen(
                model=model,
                tokeniser=tokeniser,
                messages=messages,
                facet=facet,
                index=index,
                num_beams=beams,
                max_new_tokens=max_docid_tokens,
            )
            best: dict[int, tuple[int, float]] = {}
            for b in beams_out:
                for pid in b.paper_ids:
                    if int(pid) == int(qid):
                        continue
                    cur = best.get(int(pid))
                    if cur is None or b.rank < cur[0]:
                        best[int(pid)] = (b.rank, b.score)
            ranked = sorted(best.items(), key=lambda kv: kv[1][0])[:top_n]
            record = {
                "corpusid": int(qid),
                "results": [
                    {
                        "corpusid": pid,
                        "score": round(rs[1], 6),
                        "facets": ["paper"],
                        "facet_ranks": {"paper": rs[0]},
                    }
                    for pid, rs in ranked
                ],
            }
            fh.write(json.dumps(record) + "\n")
            n_queries += 1
    logger.info(
        "Wrote %d queries (%d skipped no-title) to %s", n_queries, n_skipped, output
    )


def register(app: typer.Typer) -> None:
    """Attach the paper-level eval command to a typer app."""
    app.command(name="evaluate-paper")(cmd_evaluate_glen_paper)


__all__ = ["cmd_evaluate_glen_paper", "register"]
