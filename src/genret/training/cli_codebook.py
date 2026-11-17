# pyright: basic
# ruff: noqa: PLC0415  Enable function-level imports
"""Codebook commands: build, Leiden-build, build-data, train, evaluate, smoke."""

import functools
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, override

import typer

from genret.paper_parsing import FacetRecordType
from genret.training._cli_helpers import _load_checkpoint
from genret.utils import die, load_ids_from_file

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def cmd_build_codebooks(
    dense_dir: Annotated[
        Path,
        typer.Option(
            "--dense-dir",
            help=(
                "Directory containing facet_graph_*.npz files with mean-pooled "
                "per-facet corpus vectors."
            ),
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help=(
                "Directory to write per-facet codebooks, paper_codes, and "
                "code_to_papers.json. Created if missing."
            ),
            show_default=False,
        ),
    ],
    k0: Annotated[
        int,
        typer.Option("--k0", help="Number of top-level clusters per facet."),
    ] = 32,
    k1: Annotated[
        int,
        typer.Option("--k1", help="Number of sub-clusters per L0 cluster."),
    ] = 32,
    random_state: Annotated[
        int, typer.Option("--seed", help="Random seed for k-means.")
    ] = 0,
    n_init: Annotated[
        int, typer.Option("--n-init", help="k-means re-runs per fit (best wins).")
    ] = 4,
) -> None:
    """Fit per-facet hierarchical k-means codebooks for generative retrieval."""
    from rich.console import Console
    from rich.table import Table

    from genret.training.codebook import CodebookParams, build_codebooks

    params = CodebookParams(k0=k0, k1=k1, random_state=random_state, n_init=n_init)
    stats = build_codebooks(dense_dir, output_dir, params)

    console = Console()
    table = Table(title="Codebook validation")
    table.add_column("Facet", style="bold")
    table.add_column("Leaves", justify="right")
    table.add_column("Empty", justify="right")
    table.add_column(">3 papers", justify="right")
    table.add_column("min/median/max", justify="right")
    table.add_column("Gate", justify="center")
    for facet, s in stats.items():
        gate = "[green]ok[/green]" if s.passes_gate() else "[red]fail[/red]"
        table.add_row(
            facet,
            str(s.leaves_total),
            f"{s.leaves_empty} ({s.fraction_empty:.1%})",
            f"{s.leaves_with_more_than_three} ({s.fraction_more_than_three:.1%})",
            f"{s.papers_per_leaf_min}/{s.papers_per_leaf_median}/{s.papers_per_leaf_max}",
            gate,
        )
    console.print(table)

    if not all(s.passes_gate() for s in stats.values()):
        die(
            "Validation gate failed for one or more facets. Tune --k0 / --k1 "
            "before proceeding."
        )


def cmd_build_leiden_codebooks(
    candidates_dir: Annotated[
        Path,
        typer.Option(
            "--candidates-dir",
            help="Directory with facet_candidates_*.npz files produced by "
            "`paper-processing build-facet-graph`. Each contains a top-K "
            "kNN graph per paper with fused dense+ngram+citation weights.",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Where to write per-facet paper_codes.jsonl in the "
            "codebook-directory layout. C2T index builder can consume "
            "this directly.",
            show_default=False,
        ),
    ],
    resolution_l0: Annotated[
        float,
        typer.Option(
            "--resolution-l0",
            help="Leiden resolution at the top level (smaller = fewer, "
            "larger communities).",
        ),
    ] = 0.6,
    resolution_l1: Annotated[
        float,
        typer.Option(
            "--resolution-l1",
            help="Leiden resolution inside each L0 sub-graph.",
        ),
    ] = 1.2,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    n_iterations: Annotated[
        int,
        typer.Option(
            "--n-iterations",
            help="Leiden refinement iterations (-1 = run until convergence).",
        ),
    ] = -1,
    min_l1_size: Annotated[
        int,
        typer.Option(
            "--min-l1-size",
            help="L0 communities with at most this many papers are kept "
            "as a single L1 bucket (no sub-clustering).",
        ),
    ] = 4,
    truncate_to_top_k: Annotated[
        int | None,
        typer.Option(
            "--truncate-to-top-k",
            help="Cap candidate columns per row before building the graph. "
            "None uses all (typically 100).",
        ),
    ] = None,
    score_key: Annotated[
        str,
        typer.Option(
            "--score-key",
            help="Which weight column to use as edge weight: scores_combined "
            "(default), scores_dense, scores_citation, scores_ngram.",
        ),
    ] = "scores_combined",
) -> None:
    """Hierarchical Leiden codebook on the existing per-facet candidates graph.

    Writes per-facet ``paper_codes.jsonl`` so the C2T-ID index builder can be
    re-run pointing at this directory.
    """
    from genret.training.leiden_clustering import (
        LeidenCodebookParams,
        build_leiden_codebooks,
    )

    params = LeidenCodebookParams(
        resolution_l0=resolution_l0,
        resolution_l1=resolution_l1,
        seed=seed,
        n_iterations=n_iterations,
        min_l1_size=min_l1_size,
        truncate_to_top_k=truncate_to_top_k,
        score_key=score_key,
    )
    out = build_leiden_codebooks(candidates_dir, output_dir, params)
    for facet, cb in out.items():
        n_papers = len(cb.paper_codes)
        n_leaves = len({(l0, l1) for (l0, l1) in cb.paper_codes.values()})
        logger.info(
            "%s: %d papers, %d L0 communities, %d distinct (L0,L1) leaves",
            facet,
            n_papers,
            cb.n_l0,
            n_leaves,
        )


def cmd_build_codebook_data(
    retrieval: Annotated[
        Path,
        typer.Option(
            "--retrieval",
            help=(
                "JSONL produced by `paper-processing batch-retrieve` over the "
                "corpus, with per-candidate facet_ranks."
            ),
            show_default=False,
        ),
    ],
    codebook_dir: Annotated[
        Path,
        typer.Option(
            "--codebook-dir",
            help="Directory written by `training build-codebooks`.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output JSONL path (.jsonl.zst supported).",
            show_default=False,
        ),
    ],
    max_rank: Annotated[
        int,
        typer.Option(
            "--max-rank",
            help=(
                "Drop candidates whose per-facet rank meets or exceeds this "
                "value. Default 20 mirrors the chapter's top-20 cutoff."
            ),
        ),
    ] = 20,
) -> None:
    """Build codebook-style training pairs from the graph retriever output."""
    from rich.console import Console
    from rich.table import Table

    from genret.training.codebook_data import FACET_NAMES, build_training_pairs

    stats = build_training_pairs(
        retrieval_path=retrieval,
        codebook_dir=codebook_dir,
        output_path=output,
        max_rank=max_rank,
    )

    console = Console()
    table = Table(title="Codebook training pairs")
    table.add_column("Bucket", style="bold", min_width=28)
    table.add_column("Count", justify="right")
    table.add_row("indexing", f"{stats.indexing:,}")
    table.add_row("retrieval (all facets)", f"{stats.retrieval:,}")
    for facet in FACET_NAMES:
        table.add_row(
            f"  retrieval [{facet}]",
            f"{stats.per_facet_retrieval.get(facet, 0):,}",
        )
    table.add_row("total", f"{stats.indexing + stats.retrieval:,}")
    table.add_row("unique query papers", f"{stats.unique_query_papers:,}")
    console.print(table)


def cmd_train_codebook(
    pairs: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help="Codebook training pairs JSONL (from build-codebook-data).",
            show_default=False,
        ),
    ],
    facets: Annotated[
        Path,
        typer.Option(
            "--facets",
            help="JSONL with per-paper facet records.",
            show_default=False,
        ),
    ],
    codebook_dir: Annotated[
        Path,
        typer.Option(
            "--codebook-dir",
            help="Per-facet codebook directory.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Checkpoint output dir."),
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.1-8B-Instruct",
    k0: Annotated[int, typer.Option("--k0")] = 64,
    k1: Annotated[int, typer.Option("--k1")] = 256,
    epochs: Annotated[int, typer.Option("--epochs")] = 5,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 8,
    learning_rate: Annotated[float, typer.Option("--lr")] = 2e-4,
    lora_rank: Annotated[int, typer.Option("--lora-rank")] = 16,
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = 2,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length")] = 512,
    indexing_ratio: Annotated[
        float,
        typer.Option(
            "--indexing-ratio",
            help="Fraction of training examples that are indexing.",
            min=0.0,
            max=1.0,
        ),
    ] = 0.5,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 10,
    eval_steps: Annotated[int, typer.Option("--eval-steps")] = 200,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 200,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    include_evidence: Annotated[
        bool, typer.Option("--include-evidence/--no-include-evidence")
    ] = True,
    train_split: Annotated[
        Path | None,
        typer.Option(
            "--train-split",
            help="Optional split file listing train query corpus ids.",
        ),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option(
            "--dev-split",
            help="Optional split file listing dev query corpus ids.",
        ),
    ] = None,
    resume_from_checkpoint: Annotated[
        Path | None,
        typer.Option(help="Resume training from this checkpoint directory."),
    ] = None,
    facets_subset: Annotated[
        str | None,
        typer.Option(
            "--facets-subset",
            help=(
                "Comma-separated facet names to train on (e.g. 'contributions' "
                "or 'problems,methods'). Defaults to all four."
            ),
        ),
    ] = None,
    full_finetune: Annotated[
        bool,
        typer.Option(
            "--full-finetune/--no-full-finetune",
            help=(
                "Skip QLoRA and fine-tune all weights. The base model is "
                "loaded in bfloat16 without quantisation."
            ),
        ),
    ] = False,
) -> None:
    """Fine-tune a language model with QLoRA (default) or full FT on codebook training pairs."""
    from transformers import (
        Trainer,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )

    from genret.training.codebook_data import FACET_NAMES
    from genret.training.codebook_dataset import (
        CodebookDataset,
        CodebookDatasetConfig,
    )
    from genret.training.config import (
        CodebookConfig,
        LoRAConfig,
        QuantConfig,
        TrainingConfig,
    )
    from genret.training.data import collate_fn
    from genret.training.model import setup_model_and_tokeniser
    from genret.types import CorpusId

    class _FinalStepCallback(TrainerCallback):
        """Force evaluation + checkpoint at the very last training step."""

        @override
        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: object,
        ) -> None:
            """Run the final-step eval/save callback."""
            if state.global_step >= state.max_steps:
                control.should_evaluate = True
                control.should_save = True

    cfg = CodebookConfig(codebook_dir=codebook_dir, k0=k0, k1=k1)
    training_cfg = TrainingConfig(
        splits_dir=Path(),  # unused in the codebook path
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
        num_epochs=epochs,
        indexing_ratio=indexing_ratio,
    )
    lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_rank * 2)
    quant_cfg = QuantConfig()
    rng = random.Random(training_cfg.seed)

    model, tokeniser = setup_model_and_tokeniser(
        model_name,
        lora_cfg,
        quant_cfg,
        extra_special_tokens=cfg.special_tokens(),
        full_finetune=full_finetune,
    )

    def _make_dataset(split_path: Path | None) -> CodebookDataset:
        split_ids: set[CorpusId] | None = None
        if split_path is not None:
            if not split_path.exists():
                die(f"Split file not found: {split_path}")
            split_ids = load_ids_from_file(split_path)
            if not split_ids:
                die(f"Split file {split_path} contains no ids")
        facets_arg = (
            tuple(f.strip() for f in facets_subset.split(",") if f.strip())
            if facets_subset
            else FACET_NAMES
        )
        return CodebookDataset(
            CodebookDatasetConfig(
                pairs_path=pairs,
                facets_path=facets,
                codebook=cfg,
                facets_type=facets_type,
                include_evidence=include_evidence,
                indexing_ratio=indexing_ratio,
                split_ids=split_ids,
                facets=facets_arg,
            ),
            rng=rng,
        )

    train_dataset = _make_dataset(train_split)
    dev_dataset = _make_dataset(dev_split) if dev_split is not None else None

    bound_collate = functools.partial(
        collate_fn,
        tokeniser=tokeniser,
        max_seq_length=max_seq_length,
    )

    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=training_cfg.warmup_ratio,
        bf16=True,
        logging_steps=logging_steps,
        eval_strategy="steps" if dev_dataset is not None else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
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
        callbacks=[_FinalStepCallback()],
    )

    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint)
        if resume_from_checkpoint
        else None
    )
    trainer.save_model(str(output))
    tokeniser.save_pretrained(str(output))


def cmd_evaluate_codebook(
    checkpoint: Annotated[
        Path, typer.Argument(help="Fine-tuned LoRA checkpoint directory.")
    ],
    facets: Annotated[
        Path,
        typer.Option(
            "--facets",
            help="JSONL with per-paper facet records.",
            show_default=False,
        ),
    ],
    codebook_dir: Annotated[
        Path,
        typer.Option(
            "--codebook-dir",
            help="Per-facet codebook directory.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output JSONL path.",
            show_default=False,
        ),
    ],
    queries: Annotated[
        Path,
        typer.Option(
            "--queries",
            help="Plain-text file of one corpus id per line.",
            show_default=False,
        ),
    ],
    k0: Annotated[int, typer.Option("--k0")] = 64,
    k1: Annotated[int, typer.Option("--k1")] = 256,
    beams: Annotated[int, typer.Option("--beams", help="Beams per facet.")] = 30,
    top_n: Annotated[
        int, typer.Option("--top-n", help="Top-N to keep after RRF.")
    ] = 50,
    facets_type: Annotated[
        FacetRecordType,
        typer.Option("--facets-type"),
    ] = FacetRecordType.STANDARD,
    include_evidence: Annotated[
        bool, typer.Option("--include-evidence/--no-include-evidence")
    ] = True,
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
) -> None:
    """Run constrained-beam retrieval + RRF on a query split."""
    from genret.training.codebook_data import FACET_NAMES
    from genret.training.codebook_decoding import (
        build_token_map,
        load_codebook_index,
    )
    from genret.training.codebook_evaluate import (
        evaluate_codebook_retrieval,
        load_eval_inputs,
        queries_from_split,
    )
    from genret.training.config import CodebookConfig

    cfg = CodebookConfig(codebook_dir=codebook_dir, k0=k0, k1=k1)
    index = load_codebook_index(codebook_dir)

    model, tokeniser = _load_checkpoint(checkpoint)
    tokens = build_token_map(cfg, tokeniser)

    facets_map, evidence_map = load_eval_inputs(facets, facets_type)
    if not include_evidence:
        evidence_map.clear()
    query_ids = queries_from_split(queries)

    facets_arg = (
        tuple(f.strip() for f in facets_subset.split(",") if f.strip())
        if facets_subset
        else FACET_NAMES
    )
    evaluate_codebook_retrieval(
        model=model,
        tokeniser=tokeniser,
        cfg=cfg,
        index=index,
        tokens=tokens,
        queries=query_ids,
        facets_map=facets_map,
        evidence_map=evidence_map,
        output_path=output,
        num_beams=beams,
        top_n=top_n,
        facets=facets_arg,
    )


def cmd_smoke_codebook(
    facets: Annotated[
        Path,
        typer.Option("--facets", show_default=False),
    ],
    codebook_dir: Annotated[
        Path,
        typer.Option("--codebook-dir", show_default=False),
    ],
    queries: Annotated[
        Path,
        typer.Option("--queries", show_default=False),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", show_default=False),
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.1-8B-Instruct",
    k0: Annotated[int, typer.Option("--k0")] = 64,
    k1: Annotated[int, typer.Option("--k1")] = 256,
    beams: Annotated[int, typer.Option("--beams")] = 8,
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Number of queries (0 = all)")
    ] = 5,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
) -> None:
    """Smoke-test the constrained-decoding pipeline with an untrained model.

    Loads a quantised base model with codebook special tokens added,
    runs the 4-facet decode on a few queries, and writes the JSONL
    output. The retrieval quality is meaningless until the model is
    fine-tuned; this command just exercises the plumbing.
    """
    from genret.training.codebook_decoding import (
        build_token_map,
        load_codebook_index,
    )
    from genret.training.codebook_evaluate import (
        evaluate_codebook_retrieval,
        load_eval_inputs,
        queries_from_split,
    )
    from genret.training.config import (
        CodebookConfig,
        LoRAConfig,
        QuantConfig,
    )
    from genret.training.model import setup_model_and_tokeniser

    cfg = CodebookConfig(codebook_dir=codebook_dir, k0=k0, k1=k1)
    index = load_codebook_index(codebook_dir)

    model, tokeniser = setup_model_and_tokeniser(
        model_name,
        LoRAConfig(),
        QuantConfig(),
        extra_special_tokens=cfg.special_tokens(),
    )
    model.eval()
    tokens = build_token_map(cfg, tokeniser)

    facets_map, evidence_map = load_eval_inputs(facets, facets_type)
    query_ids = queries_from_split(queries)
    if limit > 0:
        query_ids = query_ids[:limit]

    evaluate_codebook_retrieval(
        model=model,
        tokeniser=tokeniser,
        cfg=cfg,
        index=index,
        tokens=tokens,
        queries=query_ids,
        facets_map=facets_map,
        evidence_map=evidence_map,
        output_path=output,
        num_beams=beams,
        top_n=20,
    )
    logger.info("Smoke test wrote %d query results to %s", len(query_ids), output)


def register_codebook_commands(app: typer.Typer) -> None:
    """Register codebook commands on the given Typer app."""
    app.command(name="build-codebooks")(cmd_build_codebooks)
    app.command(name="build-leiden-codebooks")(cmd_build_leiden_codebooks)
    app.command(name="build-codebook-data")(cmd_build_codebook_data)
    app.command(name="train-codebook")(cmd_train_codebook)
    app.command(name="evaluate-codebook")(cmd_evaluate_codebook)
    app.command(name="smoke-codebook")(cmd_smoke_codebook)


__all__ = [
    "cmd_build_codebook_data",
    "cmd_build_codebooks",
    "cmd_build_leiden_codebooks",
    "cmd_evaluate_codebook",
    "cmd_smoke_codebook",
    "cmd_train_codebook",
    "register_codebook_commands",
]
