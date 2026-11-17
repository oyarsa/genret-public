# ruff: noqa: PLC0415
"""Eval-pipeline sanity check: run GLEN retrieval on training-set queries.

Runs GLEN evaluate on N queries drawn from the training split (queries the
model has seen many times) and computes recall@k against the training-pair
labels.  If recall is near-zero on training queries, there is a plumbing bug
in the eval pipeline, not a generalisation failure.

Usage:
    uv run --extra cuda --extra training python scripts/eval/eval_train_recall.py \
        --checkpoint output/genret/checkpoints/glen-llama-8b \
        --facets data/s2orc-nlp-v2/splits-connected-10k-v2/processed/facets.jsonl.zst \
        --glen-dir output/genret/glen_index \
        --pairs output/genret/training_pairs.jsonl.zst \
        --train-split data/s2orc-nlp-v2/splits-connected-10k-v2/train.txt \
        --n 50
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer

from genret._shed.transformers import load_tokenizer
from genret.paper_parsing import FacetRecordType
from genret.types import CorpusId
from genret.utils import iter_jsonl


def _load_train_labels(
    pairs_path: Path,
    train_ids: set[int],
) -> dict[int, set[int]]:
    """Build {paper_a: {paper_b, ...}} from retrieval pairs where paper_a in train."""
    labels: dict[int, set[int]] = defaultdict(set)
    for row in iter_jsonl(pairs_path):
        if row.get("kind") != "retrieval":
            continue
        a = int(row["paper_a"])
        b = int(row["paper_b"])
        if a in train_ids:
            labels[a].add(b)
    return dict(labels)


def _recall_at_k(
    results: dict[int, list[int]],
    labels: dict[int, set[int]],
    k: int,
) -> float:
    hits = 0
    total = 0
    for qid, retrieved in results.items():
        gold = labels.get(qid)
        if not gold:
            continue
        total += 1
        if any(pid in gold for pid in retrieved[:k]):
            hits += 1
    return hits / total if total else 0.0


def main(
    checkpoint: Annotated[Path, typer.Option(help="Fine-tuned GLEN checkpoint.")],
    facets: Annotated[Path, typer.Option(show_default=False)],
    glen_dir: Annotated[Path, typer.Option(show_default=False)],
    pairs: Annotated[
        Path, typer.Option(help="Training pairs JSONL.", show_default=False)
    ],
    train_split: Annotated[Path, typer.Option(show_default=False)],
    n: Annotated[int, typer.Option(help="Number of training queries.")] = 50,
    beams: Annotated[int, typer.Option()] = 50,
    max_docid_tokens: Annotated[int, typer.Option()] = 32,
    top_n: Annotated[int, typer.Option()] = 50,
    seed: Annotated[int, typer.Option()] = 42,
    output: Annotated[Path | None, typer.Option()] = None,
    facets_type: Annotated[FacetRecordType, typer.Option()] = FacetRecordType.STANDARD,
    no_lora: Annotated[
        bool,
        typer.Option(
            help="Skip LoRA adapter loading; run the base model without fine-tuning."
        ),
    ] = False,
    base_model: Annotated[
        str, typer.Option(help="Base model name to load when --no-lora is set.")
    ] = "meta-llama/Llama-3.1-8B-Instruct",
) -> None:
    """Sanity-check the eval pipeline by running on training-set queries."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from genret.training.cli import _load_checkpoint
    from genret.training.glen_decoding import load_glen_index
    from genret.training.glen_evaluate import evaluate_glen_retrieval, load_eval_inputs

    train_ids = {int(x) for x in train_split.read_text().split() if x.strip()}
    print(f"Train split: {len(train_ids)} papers")

    print("Loading training labels...")
    labels = _load_train_labels(pairs, train_ids)
    print(f"Training labels: {len(labels)} query papers with retrieval targets")

    rng = random.Random(seed)
    sample_ids = rng.sample(sorted(labels.keys()), min(n, len(labels)))
    print(f"Evaluating {len(sample_ids)} training queries\n")

    index = load_glen_index(glen_dir)

    if no_lora:
        print(f"Loading base model (no LoRA): {base_model}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokeniser = load_tokenizer(base_model)
        tokeniser.pad_token = tokeniser.eos_token
        tokeniser.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        model.eval()
    else:
        model, tokeniser = _load_checkpoint(checkpoint)
        model.eval()

    facets_map, evidence_map = load_eval_inputs(facets, facets_type)

    out_path = output or Path(f"output/genret/eval/train_recall_n{n}.jsonl")
    evaluate_glen_retrieval(
        model=model,
        tokeniser=tokeniser,
        index=index,
        queries=[CorpusId(q) for q in sample_ids],
        facets_map=facets_map,
        evidence_map=evidence_map,
        output_path=out_path,
        num_beams=beams,
        max_new_tokens=max_docid_tokens,
        top_n=top_n,
    )

    # Load results and compute recall
    results: dict[int, list[int]] = {}
    for row in iter_jsonl(out_path):
        qid = int(row["corpusid"])
        results[qid] = [int(r["corpusid"]) for r in row.get("results", [])]

    print("\n=== Recall on training-set queries ===")
    for k in [1, 5, 10, 20, 50]:
        r = _recall_at_k(results, labels, k)
        print(f"  R@{k:<3d} = {r:.4f}")

    # Per-query breakdown for first 10
    print("\nPer-query sample (first 10):")
    for qid in sample_ids[:10]:
        gold = labels.get(qid, set())
        retrieved = results.get(qid, [])
        hit_rank = next((i + 1 for i, pid in enumerate(retrieved) if pid in gold), None)
        n_gold = len(gold)
        print(f"  qid={qid}  gold={n_gold}  hit_rank={hit_rank or 'MISS'}")


if __name__ == "__main__":
    typer.run(main)
