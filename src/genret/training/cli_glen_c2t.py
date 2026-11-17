# pyright: basic
# ruff: noqa: PLC0415
"""CLI for C2T-ID DocID building and training.

Two commands attached to the GLEN typer app:

- ``build-c2t-index``: produce a C2T-ID GlenIndex from an existing per-facet
  hierarchical codebook.
- ``train-c2t``: fine-tune a language model on C2T-ID textual targets,
  mirroring the per-bullet 78lp recipe (one example per paper_a bullet,
  target = paper_b's C2T-ID for the facet).
"""

from __future__ import annotations

import functools
import logging
import random
from pathlib import Path
from typing import Annotated

import typer

from genret._shed.transformers import load_tokenizer
from genret.paper_parsing import FacetRecordType
from genret.utils import die, load_ids_from_file

logger = logging.getLogger(__name__)


def cmd_build_c2t_index(
    codebook_dir: Annotated[
        Path,
        typer.Option(
            "--codebook-dir",
            help="Existing hierarchical codebook directory.",
        ),
    ],
    facets: Annotated[
        Path,
        typer.Option("--facets", help="JSONL with per-paper facet records."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Where to write the C2T GlenIndex."),
    ],
    tokeniser_model: Annotated[
        str,
        typer.Option(
            "--tokeniser-model",
            help="HF model id whose tokeniser is used to tokenise the C2T DocIDs.",
        ),
    ] = "meta-llama/Llama-3.2-1B-Instruct",
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Keywords per cluster node (C2T paper uses 3)."),
    ] = 3,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Cap on tokenised DocID length."),
    ] = 32,
    scoring: Annotated[
        str,
        typer.Option(
            "--scoring",
            help="Keyword scoring mode: 'tfidf' (demote ambient words via "
            "inverse-cluster-frequency), 'logodds' (Dirichlet-prior "
            "log-odds z-score vs rest of corpus), 'embedding' (cosine to "
            "cluster centroid, MiniLM), or 'frequency' (raw count, the "
            "C2T paper's default).",
        ),
    ] = "tfidf",
    selection: Annotated[
        str,
        typer.Option(
            "--selection",
            help="Keyword selection mode: 'topk' (top-K by score) or "
            "'mmr' (diversity-aware selection over the top candidates).",
        ),
    ] = "topk",
    mmr_lambda: Annotated[
        float,
        typer.Option(
            "--mmr-lambda",
            help="MMR relevance weight in [0, 1]; 1.0 = pure relevance.",
        ),
    ] = 0.7,
    mmr_pool: Annotated[
        int,
        typer.Option("--mmr-pool", help="Candidate pool size for MMR."),
    ] = 10,
    prior_strength: Annotated[
        float,
        typer.Option(
            "--prior-strength",
            help="Pseudo-count mass for the log-odds Dirichlet prior.",
        ),
    ] = 10.0,
    keyword_source: Annotated[
        str,
        typer.Option(
            "--keyword-source",
            help="Source of per-paper keywords: 'bullets' (facet text, "
            "paper default), 'titles' (paper title noun phrases), or "
            "'titles_abstracts' (title + abstract).",
        ),
    ] = "bullets",
    paper_graphs: Annotated[
        Path | None,
        typer.Option(
            "--paper-graphs",
            help="JSONL with per-paper metadata (title). Required for "
            "keyword sources 'titles' and 'titles_abstracts'.",
        ),
    ] = None,
    abstracts: Annotated[
        Path | None,
        typer.Option(
            "--abstracts",
            help="JSONL with per-paper abstracts. Required for keyword "
            "source 'titles_abstracts'.",
        ),
    ] = None,
    keyword_separator: Annotated[
        str,
        typer.Option(
            "--keyword-separator",
            help="Separator joining keywords within a cluster level. Default "
            "'-' produces 'foo-bar'. Use ' ' to get space-joined prefixes "
            "(fewer BPE tokens at the cost of leaving the model to predict "
            "structural spaces).",
        ),
    ] = "-",
    level_separator: Annotated[
        str,
        typer.Option(
            "--level-separator",
            help="Separator joining the L0 keyword group and the L1 keyword "
            "group. Default ' '.",
        ),
    ] = " ",
    dedup_consecutive_keywords: Annotated[
        bool,
        typer.Option(
            "--dedup-consecutive-keywords/--no-dedup-consecutive-keywords",
            help="If true, normalise the level separator to the keyword "
            "separator and drop consecutive duplicate keywords. Default true. "
            "Disable when keyword-separator=' ' to preserve space-joined "
            "prefixes.",
        ),
    ] = True,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
) -> None:
    """Build a C2T-ID GlenIndex from an existing hierarchical codebook."""

    from genret.training.c2t_index import C2TIndexParams, build_c2t_index

    tokeniser = load_tokenizer(tokeniser_model)
    params = C2TIndexParams(
        top_k=top_k,
        max_tokens=max_tokens,
        scoring=scoring,
        selection=selection,
        mmr_lambda=mmr_lambda,
        mmr_pool=mmr_pool,
        prior_strength=prior_strength,
        keyword_source=keyword_source,
        keyword_separator=keyword_separator,
        level_separator=level_separator,
        dedup_consecutive_keywords=dedup_consecutive_keywords,
    )
    index, _docids = build_c2t_index(
        codebook_dir,
        facets,
        output_dir,
        tokeniser,
        params=params,
        facets_type=facets_type,
        paper_graphs_path=paper_graphs,
        abstracts_path=abstracts,
    )
    total_leaves = sum(len(v) for v in index.code_to_papers.values())
    logger.info(
        "C2T index complete: %d facets, %d total leaves",
        len(index.code_to_papers),
        total_leaves,
    )


def cmd_train_c2t(
    pairs: Annotated[
        Path,
        typer.Option("--pairs", help="Training pairs JSONL.", show_default=False),
    ],
    facets: Annotated[
        Path,
        typer.Option(
            "--facets", help="JSONL with per-paper facet records.", show_default=False
        ),
    ],
    paper_graphs: Annotated[
        Path,
        typer.Option(
            "--paper-graphs",
            help="JSONL with per-paper metadata (must include 'title').",
            show_default=False,
        ),
    ],
    c2t_dir: Annotated[
        Path,
        typer.Option(
            "--c2t-dir",
            help="Directory written by build-c2t-index (contains "
            "paper_c2t_docids.jsonl + trie.json + code_to_papers.json).",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Checkpoint output dir.")
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.2-1B-Instruct",
    epochs: Annotated[int, typer.Option("--epochs")] = 2,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 4,
    learning_rate: Annotated[float, typer.Option("--lr")] = 5e-5,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 0.01,
    lora_rank: Annotated[int, typer.Option("--lora-rank")] = 16,
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = 4,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length")] = 512,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 20,
    eval_steps: Annotated[int, typer.Option("--eval-steps")] = 200000,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 2000,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    train_split: Annotated[
        Path | None,
        typer.Option("--train-split", help="Optional split file of train ids."),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option("--dev-split", help="Optional split file of dev ids."),
    ] = None,
    resume_from_checkpoint: Annotated[
        Path | None, typer.Option(help="Resume training from this checkpoint dir.")
    ] = None,
    facets_subset: Annotated[
        str | None,
        typer.Option(
            "--facets-subset",
            help="Comma-separated facet names to train on. Defaults to all four.",
        ),
    ] = None,
    full_finetune: Annotated[
        bool,
        typer.Option(
            "--full-finetune/--no-full-finetune",
            help="Skip QLoRA and fine-tune all weights.",
        ),
    ] = False,
) -> None:
    """Fine-tune a model with C2T-ID textual targets (per-bullet input)."""
    from transformers import Trainer, TrainingArguments

    from genret.training._glen_callbacks import FinalStepCallback
    from genret.training.codebook_data import FACET_NAMES
    from genret.training.config import LoRAConfig, QuantConfig, TrainingConfig
    from genret.training.data import collate_fn
    from genret.training.glen_c2t_dataset import (
        GlenC2TDataset,
        GlenC2TDatasetConfig,
    )
    from genret.training.model import setup_model_and_tokeniser
    from genret.types import CorpusId

    training_cfg = TrainingConfig(
        splits_dir=Path(),
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
        num_epochs=epochs,
        indexing_ratio=0.0,
    )
    lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_rank * 2)
    quant_cfg = QuantConfig()
    rng = random.Random(training_cfg.seed)

    model, tokeniser = setup_model_and_tokeniser(
        model_name,
        lora_cfg,
        quant_cfg,
        extra_special_tokens=[],
        full_finetune=full_finetune,
    )

    facets_arg = (
        tuple(f.strip() for f in facets_subset.split(",") if f.strip())
        if facets_subset
        else FACET_NAMES
    )

    def _make_dataset(split_path: Path | None) -> GlenC2TDataset:
        split_ids: set[CorpusId] | None = None
        if split_path is not None:
            if not split_path.exists():
                die(f"Split file not found: {split_path}")
            split_ids = load_ids_from_file(split_path)
            if not split_ids:
                die(f"Split file {split_path} contains no ids")
        return GlenC2TDataset(
            GlenC2TDatasetConfig(
                pairs_path=pairs,
                facets_path=facets,
                paper_graphs_path=paper_graphs,
                c2t_dir=c2t_dir,
                facets_type=facets_type,
                split_ids=split_ids,
                facets=facets_arg,
            ),
            rng=rng,
        )

    train_dataset = _make_dataset(train_split)
    dev_dataset = _make_dataset(dev_split) if dev_split is not None else None

    bound_collate = functools.partial(
        collate_fn, tokeniser=tokeniser, max_seq_length=max_seq_length
    )

    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=training_cfg.warmup_ratio,
        bf16=True,
        logging_steps=logging_steps,
        eval_strategy="steps" if dev_dataset is not None else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        save_only_model=True,
        load_best_model_at_end=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if full_finetune else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=bound_collate,
        callbacks=[FinalStepCallback()],
    )

    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint)
        if resume_from_checkpoint
        else None
    )
    trainer.save_model(str(output))
    tokeniser.save_pretrained(str(output))


def cmd_build_d1_index(
    c2t_dir: Annotated[
        Path,
        typer.Option(
            "--c2t-dir",
            help="Directory written by build-c2t-index on a Leiden codebook "
            "(coarse, e.g. res_l0=1.0 res_l1=2.0). Provides per-(facet, paper) "
            "cluster-label prefixes via paper_c2t_docids.jsonl.",
            show_default=False,
        ),
    ],
    facets: Annotated[
        Path,
        typer.Option("--facets", help="JSONL with per-paper facet records."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Where to write the D1 GlenIndex."),
    ],
    tokeniser_model: Annotated[
        str,
        typer.Option(
            "--tokeniser-model",
            help="HF model id whose tokeniser is used to tokenise DocIDs.",
        ),
    ] = "meta-llama/Llama-3.2-1B-Instruct",
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens",
            help="Cap on tokenised DocID length (prefix + bullet combined).",
        ),
    ] = 24,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
) -> None:
    """Build the D1 prefix+bullet GlenIndex (graph-token-prefix variant)."""

    from genret.training.glen_d1_index import D1IndexParams, build_d1_index

    tokeniser = load_tokenizer(tokeniser_model)
    params = D1IndexParams(max_tokens=max_tokens)
    index = build_d1_index(
        c2t_dir,
        facets,
        output_dir,
        tokeniser,
        params=params,
        facets_type=facets_type,
    )
    total_leaves = sum(len(v) for v in index.code_to_papers.values())
    logger.info(
        "D1 index complete: %d facets, %d total DocIDs",
        len(index.code_to_papers),
        total_leaves,
    )


def cmd_evaluate_c2t_labels(
    c2t_dir: Annotated[
        Path,
        typer.Option("--c2t-dir", help="Built C2T index directory."),
    ],
    facets: Annotated[
        Path,
        typer.Option("--facets", help="JSONL with per-paper facet records."),
    ],
    split: Annotated[
        Path,
        typer.Option("--split", help="Query split file (corpus ids)."),
    ],
    pairs: Annotated[
        Path,
        typer.Option("--pairs", help="Training-pairs JSONL (citation gold)."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Where to write metrics JSON. Default: <c2t-dir>/label_eval.json.",
        ),
    ] = None,
    top_leaves: Annotated[
        int,
        typer.Option(
            "--top-leaves",
            help="Leaves considered per query before paper expansion.",
        ),
    ] = 100,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
) -> None:
    """Proxy-evaluate a C2T index's labels without training a model."""
    import json

    from genret.training.c2t_label_eval import evaluate_c2t_labels
    from genret.training.cluster_keyword_labels import make_minilm_embedder

    result = evaluate_c2t_labels(
        c2t_dir,
        facets,
        split,
        pairs,
        embed_fn=make_minilm_embedder(),
        facets_type=facets_type,
        top_leaves=top_leaves,
    )
    out_path = output or c2t_dir / "label_eval.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")

    header = (
        f"{'facet':15s} {'leaves':>7s} {'coll%':>6s} {'redun':>6s} "
        f"{'faith':>6s} {'sep':>6s} {'R@20':>6s}"
    )
    lines = [header]
    for facet, m in result.per_facet.items():
        lines.append(
            f"{facet:15s} {m.n_leaves:7d} {m.collision_rate:6.1%} "
            f"{m.mean_redundancy:6.3f} {m.mean_faithfulness:6.3f} "
            f"{m.mean_separation:6.3f} {m.recall_at_20:6.3f}"
        )
    lines.append(
        f"fused (n={result.n_queries}): "
        f"R@20={result.fused_recall_at_20:.4f} "
        f"R@50={result.fused_recall_at_50:.4f}"
    )
    typer.echo("\n".join(lines))
    logger.info("Wrote %s", out_path)


def register(app: typer.Typer) -> None:
    """Attach C2T commands to the GLEN typer app."""
    app.command(name="build-c2t-index")(cmd_build_c2t_index)
    app.command(name="train-c2t")(cmd_train_c2t)
    app.command(name="build-d1-index")(cmd_build_d1_index)
    app.command(name="evaluate-c2t-labels")(cmd_evaluate_c2t_labels)


__all__ = [
    "cmd_build_c2t_index",
    "cmd_build_d1_index",
    "cmd_train_c2t",
    "register",
]
