# pyright: basic
# ruff: noqa: PLC0415, SLF001
"""Rerank an eval JSONL with an int8-quantised cross-encoder via bitsandbytes.

Mirrors ``rerank_jsonl.py`` but loads the model with
``BitsAndBytesConfig(load_in_8bit=True)`` so the weight matrices are quantised
at load time. Used to measure (a) disk size after re-saving the quantised model
and (b) reranking quality vs the fp32 baseline on the same input JSONL.
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


def _load_int8_reranker(model_dir: Path, max_length: int = 2048) -> LocalReranker:
    """Load the cross-encoder reranker with bitsandbytes int8 quantisation."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )

    tokenizer_path = str(model_dir)
    if (
        Path(model_dir).is_dir()
        and not (Path(model_dir) / "tokenizer_config.json").exists()
    ):
        parent = Path(model_dir).parent
        if (parent / "tokenizer_config.json").exists():
            tokenizer_path = str(parent)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError(f"Unexpected tokenizer type: {type(tokenizer)}")

    quant = BitsAndBytesConfig(load_in_8bit=True)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            num_labels=1,
            quantization_config=quant,
            device_map="auto",
        )
    except RuntimeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            quantization_config=quant,
            device_map="auto",
        )
    if not isinstance(model, PreTrainedModel):
        raise TypeError(f"Unexpected model type: {type(model)}")

    torch.set_float32_matmul_precision("high")
    model.eval()
    logger.info("Loaded int8 reranker from %s", model_dir)
    cfg = RerankerConfig(
        model_name=str(model_dir), max_length=max_length, device="cuda"
    )
    return LocalReranker(model=model, tokenizer=tokenizer, config=cfg)


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
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
    p.add_argument(
        "--save-quantised-to",
        type=Path,
        default=None,
        help="If set, save the int8-loaded model here for disk-size measurement.",
    )
    args = p.parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")

    logger.info("Loading facets+titles")
    facets_map = load_facets_map(args.facets)
    titles = load_titles(args.paper_graphs)

    model = _load_int8_reranker(args.model_dir)

    if args.save_quantised_to is not None:
        args.save_quantised_to.mkdir(parents=True, exist_ok=True)
        model._model.save_pretrained(str(args.save_quantised_to))  # type: ignore[attr-defined]
        model._tokenizer.save_pretrained(str(args.save_quantised_to))  # type: ignore[attr-defined]
        logger.info("Saved int8 model to %s", args.save_quantised_to)

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
        out_path = args.output_dir / f"{input_path.stem}__reranked_int8.jsonl"
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
                                    "facets": ["rerank_int8"],
                                    "facet_ranks": {"rerank_int8": rank},
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
