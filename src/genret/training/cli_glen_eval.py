# pyright: basic
# ruff: noqa: PLC0415  Enable function-level imports
"""GLEN evaluation commands: trie-constrained beam search + RRF, and smoke."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, cast

import typer

from genret._shed.transformers import load_tokenizer
from genret.paper_parsing import FacetRecordType
from genret.utils import die

logger = logging.getLogger(__name__)


def cmd_evaluate_glen(
    checkpoint: Annotated[
        Path, typer.Argument(help="Fine-tuned LoRA checkpoint directory.")
    ],
    facets: Annotated[
        Path,
        typer.Option(
            "--facets", help="JSONL with per-paper facet records.", show_default=False
        ),
    ],
    glen_dir: Annotated[
        Path,
        typer.Option(
            "--glen-dir",
            help="GLEN index directory from build-docids.",
            show_default=False,
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", show_default=False)],
    queries: Annotated[
        Path,
        typer.Option(
            "--queries", help="Plain-text file of corpus ids.", show_default=False
        ),
    ],
    beams: Annotated[int, typer.Option("--beams")] = 50,
    max_docid_tokens: Annotated[int, typer.Option("--max-docid-tokens")] = 32,
    top_n: Annotated[int, typer.Option("--top-n")] = 50,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    prompt_kind: Annotated[
        str,
        typer.Option(
            "--prompt-kind",
            help="System prompt variant: 'retrieval' (generate a related paper's DocID) or 'indexing' (generate the input paper's own DocID).",
        ),
    ] = "retrieval",
    no_lora: Annotated[
        bool,
        typer.Option(
            "--no-lora",
            help="Skip LoRA adapter loading; run the base model without fine-tuning.",
        ),
    ] = False,
    base_model: Annotated[
        str,
        typer.Option(
            "--base-model",
            help="Base model name to load when --no-lora is set.",
        ),
    ] = "meta-llama/Llama-3.1-8B-Instruct",
    per_bullet: Annotated[
        bool,
        typer.Option(
            "--per-bullet",
            help="Query each bullet of each facet independently and fuse results via RRF (~4x more beam-search calls per query).",
        ),
    ] = False,
    bullet_prompt: Annotated[
        bool,
        typer.Option(
            "--bullet-prompt",
            help="Build the query prompt exactly like per-bullet TRAINING: "
            "'Title: {title}\\n{Facet}:\\n- {bullet}' with the bullet system "
            "prompt. Requires --paper-graphs for titles. Fixes the train/eval "
            "mismatch where the codebook prompt (no title) was used instead.",
        ),
    ] = False,
    paper_graphs: Annotated[
        Path | None,
        typer.Option(
            "--paper-graphs",
            help="paper_graphs JSONL for source-paper titles (used by "
            "--bullet-prompt).",
        ),
    ] = None,
    facets_subset: Annotated[
        str | None,
        typer.Option(
            "--facets-subset",
            help=(
                "Comma-separated facet names to evaluate on (e.g. 'contributions'). "
                "Defaults to all four."
            ),
        ),
    ] = None,
    snap: Annotated[
        str,
        typer.Option(
            "--snap",
            help=(
                "Snap-to-nearest off-trie decode mode. 'off' (default) skips it. "
                "'levenshtein' runs an additional unconstrained beam and snaps "
                "each result to the nearest DocID by token-level edit distance. "
                "'cosine' snaps via MiniLM-L12-v2 sentence embedding similarity. "
                "Snapped candidates augment (do not replace) trie-constrained "
                "beams before RRF fusion."
            ),
        ),
    ] = "off",
    snap_top_k: Annotated[
        int,
        typer.Option(
            "--snap-top-k",
            help="Number of nearest DocIDs to take per off-trie decode.",
        ),
    ] = 1,
    num_beam_groups: Annotated[
        int,
        typer.Option(
            "--num-beam-groups",
            help="Diverse Beam Search group count (Vijayakumar et al., 2016). "
            "Set >1 to enable; --beams must be divisible by this. Default 1 "
            "(plain beam search).",
        ),
    ] = 1,
    diversity_penalty: Annotated[
        float,
        typer.Option(
            "--diversity-penalty",
            help="Strength of the token-level diversity penalty between beam "
            "groups. Only used when --num-beam-groups > 1.",
        ),
    ] = 0.0,
    paper_dedup: Annotated[
        bool,
        typer.Option(
            "--paper-dedup/--no-paper-dedup",
            help="Enable paper-level deduplication during beam search: kills "
            "beams whose remaining reachable subtree contains only "
            "already-found papers, freeing slots for new exploration.",
        ),
    ] = False,
    record_docids: Annotated[
        Path | None,
        typer.Option(
            "--record-docids",
            help="Side JSONL recording every decoded DocID per (query, "
            "facet, beam) — input for 'glen analyze-bullet-usage'.",
        ),
    ] = None,
    prf_lambda: Annotated[
        float,
        typer.Option(
            "--prf-lambda",
            help="Graph-PRF bias strength. 0 (default) = off. >0 enables "
            "two-pass decoding: pass 2 tilts trie branches toward graph "
            "neighbours of pass 1's fused top results (potential shaping; "
            "completed DocID bonus = lambda*log1p(best seed edge)). "
            "Requires --prf-candidates-dir.",
        ),
    ] = 0.0,
    prf_depth_gate: Annotated[
        int,
        typer.Option(
            "--prf-depth-gate",
            help="No PRF bias at trie depths <= this (root regions reach "
            "too many papers for the bias to discriminate).",
        ),
    ] = 4,
    prf_seed_k: Annotated[
        int,
        typer.Option(
            "--prf-seed-k",
            help="Pass-1 fused top-K papers used as the PRF seed set.",
        ),
    ] = 20,
    prf_candidates_dir: Annotated[
        Path | None,
        typer.Option(
            "--prf-candidates-dir",
            help="facet_candidates_*.npz dir providing scores_combined "
            "edges for PRF biasing.",
        ),
    ] = None,
    quantize: Annotated[
        str,
        typer.Option(
            "--quantize",
            help="Inference-time quantisation for full-FT checkpoints: "
            "'off' (bf16, default), '4bit' (NF4), or '8bit' (int8). "
            "Measures deployment-quantisation recall cost.",
        ),
    ] = "off",
) -> None:
    """Run GLEN trie-constrained beam search + RRF on a query split."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from genret.training.codebook_data import FACET_NAMES
    from genret.training.codebook_evaluate import queries_from_split
    from genret.training.glen_decoding import load_glen_index
    from genret.training.glen_evaluate import (
        SnapMode,
        evaluate_glen_retrieval,
        load_eval_inputs,
    )

    index = load_glen_index(glen_dir)

    if no_lora:
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
        from genret.training.cli import _load_checkpoint

        model, tokeniser = _load_checkpoint(checkpoint, quantize=quantize)

    facets_map, evidence_map = load_eval_inputs(facets, facets_type)
    query_ids = queries_from_split(queries)

    titles = None
    if bullet_prompt:
        if paper_graphs is None:
            die("--bullet-prompt requires --paper-graphs for source titles.")
        from genret.training.glen_bullet_dataset.selection import _load_titles

        titles = _load_titles(paper_graphs)

    facets_arg = (
        tuple(f.strip() for f in facets_subset.split(",") if f.strip())
        if facets_subset
        else FACET_NAMES
    )
    if snap not in {"off", "levenshtein", "cosine"}:
        die(f"--snap must be 'off', 'levenshtein', or 'cosine', got {snap!r}")
    snap_mode = cast(SnapMode, snap)
    evaluate_glen_retrieval(
        model=model,
        tokeniser=tokeniser,
        index=index,
        queries=query_ids,
        facets_map=facets_map,
        evidence_map=evidence_map,
        output_path=output,
        num_beams=beams,
        max_new_tokens=max_docid_tokens,
        top_n=top_n,
        prompt_kind=prompt_kind,
        per_bullet=per_bullet,
        bullet_prompt=bullet_prompt,
        titles=titles,
        facets=facets_arg,
        snap_mode=snap_mode,
        snap_top_k=snap_top_k,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        paper_dedup=paper_dedup,
        record_docids_path=record_docids,
        prf_lambda=prf_lambda,
        prf_depth_gate=prf_depth_gate,
        prf_seed_k=prf_seed_k,
        prf_candidates_dir=prf_candidates_dir,
    )


def cmd_smoke_glen(
    facets: Annotated[Path, typer.Option("--facets", show_default=False)],
    glen_dir: Annotated[Path, typer.Option("--glen-dir", show_default=False)],
    queries: Annotated[Path, typer.Option("--queries", show_default=False)],
    output: Annotated[Path, typer.Option("--output", "-o", show_default=False)],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.1-8B-Instruct",
    beams: Annotated[int, typer.Option("--beams")] = 8,
    max_docid_tokens: Annotated[int, typer.Option("--max-docid-tokens")] = 32,
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Number of queries (0 = all)")
    ] = 5,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
) -> None:
    """Smoke-test GLEN decoding pipeline with an untrained model."""
    from genret.training.codebook_evaluate import queries_from_split
    from genret.training.config import LoRAConfig, QuantConfig
    from genret.training.glen_decoding import load_glen_index
    from genret.training.glen_evaluate import evaluate_glen_retrieval, load_eval_inputs
    from genret.training.model import setup_model_and_tokeniser

    index = load_glen_index(glen_dir)
    model, tokeniser = setup_model_and_tokeniser(
        model_name, LoRAConfig(), QuantConfig(), extra_special_tokens=[]
    )
    model.eval()

    facets_map, evidence_map = load_eval_inputs(facets, facets_type)
    query_ids = queries_from_split(queries)
    if limit > 0:
        query_ids = query_ids[:limit]

    evaluate_glen_retrieval(
        model=model,
        tokeniser=tokeniser,
        index=index,
        queries=query_ids,
        facets_map=facets_map,
        evidence_map=evidence_map,
        output_path=output,
        num_beams=beams,
        max_new_tokens=max_docid_tokens,
        top_n=20,
    )
    logger.info("Smoke test wrote %d query results to %s", len(query_ids), output)


def register(app: typer.Typer) -> None:
    """Attach evaluation commands to the GLEN typer app."""
    app.command(name="evaluate")(cmd_evaluate_glen)
    app.command(name="smoke")(cmd_smoke_glen)


__all__ = [
    "cmd_evaluate_glen",
    "cmd_smoke_glen",
    "register",
]
