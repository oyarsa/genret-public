"""Rerank an eval JSONL with the production cross-encoder.

Input: JSONL with ``{corpusid, results: [{corpusid, ...}, ...]}`` rows
(typically post-expansion output from ``score_glen_post_expansion.py``).

For each query, pairs query against each candidate, scores with the cross-encoder,
reorders descending by reranker score, and writes a new JSONL in the same
schema. The downstream ``score_glen_eval.py`` / ``score_glen_post_expansion.py``
can score the reranked file directly.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from genret.paper_parsing import load_facets_map
from genret.processing.bm25_baseline import rerank_retrieval_results
from genret.processing.reranker import LocalReranker, RerankerConfig, load_titles
from genret.utils import open_auto

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+", help="Input eval JSONL files.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--facets",
        type=Path,
        default=Path("data/s2orc-nlp/splits-connected-1k/processed/facets.jsonl.zst"),
    )
    p.add_argument(
        "--paper-graphs",
        type=Path,
        default=Path(
            "data/s2orc-nlp/splits-connected-1k/processed/paper_graphs.jsonl.zst"
        ),
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("output/phase2_modernbert_150k/checkpoint"),
    )
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")

    logger.info("Loading facets+titles")
    facets_map = load_facets_map(args.facets)
    titles = load_titles(args.paper_graphs)

    logger.info("Loading reranker from %s", args.model_dir)
    model = LocalReranker.from_pretrained(
        RerankerConfig(model_name=str(args.model_dir), device=args.device)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in args.inputs:
        logger.info("Reranking %s", input_path)
        reranked = rerank_retrieval_results(
            input_path,
            facets_map,
            titles,
            model,
            top_n=args.top_n,
            batch_size=args.batch_size,
        )
        out_path = args.output_dir / f"{input_path.stem}__reranked.jsonl"
        with open_auto(out_path, "w") as fh:
            for qid, results in reranked.items():
                fh.write(
                    json.dumps(
                        {
                            "corpusid": int(qid),
                            "results": [
                                {
                                    "corpusid": int(r["corpusid"]),
                                    "score": float(r["score"]),
                                    "facets": ["rerank"],
                                    "facet_ranks": {"rerank": rank},
                                }
                                for rank, r in enumerate(results)
                            ],
                        }
                    )
                    + "\n"
                )
        logger.info("Wrote %s (%d queries)", out_path, len(reranked))


if __name__ == "__main__":
    main()
