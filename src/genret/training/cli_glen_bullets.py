# pyright: basic
# ruff: noqa: PLC0415  Enable function-level imports
"""Per-bullet GLEN training commands: bullets (D9), marginal, contrastive."""

from __future__ import annotations

import functools
import logging
import random
from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from genret.paper_parsing import FacetRecordType
from genret.utils import die, load_ids_from_file

logger = logging.getLogger(__name__)


def cmd_train_glen_bullets(
    pairs: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help="Training pairs JSONL from build-codebook-data.",
            show_default=False,
        ),
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
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Checkpoint output dir.")
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.1-8B-Instruct",
    epochs: Annotated[int, typer.Option("--epochs")] = 2,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 8,
    learning_rate: Annotated[float, typer.Option("--lr")] = 2e-4,
    lora_rank: Annotated[int, typer.Option("--lora-rank")] = 16,
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = 2,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length")] = 512,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 20,
    eval_steps: Annotated[int, typer.Option("--eval-steps")] = 500,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 500,
    weight_decay: Annotated[
        float,
        typer.Option("--weight-decay", help="Weight decay for AdamW. Default 0."),
    ] = 0.0,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    train_split: Annotated[Path | None, typer.Option("--train-split")] = None,
    dev_split: Annotated[Path | None, typer.Option("--dev-split")] = None,
    resume_from_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--resume-from-checkpoint",
            help="Checkpoint directory to resume from.",
        ),
    ] = None,
    facets_subset: Annotated[
        str | None,
        typer.Option(
            "--facets-subset",
            help=(
                "Comma-separated facet names to train on (e.g. 'contributions'). "
                "Defaults to all four."
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
    fsdp: Annotated[
        str,
        typer.Option(
            "--fsdp",
            help=(
                "Hugging Face Trainer FSDP mode, e.g. "
                "'full_shard auto_wrap'. Empty disables FSDP."
            ),
        ),
    ] = "",
    fsdp_transformer_layer_cls: Annotated[
        str,
        typer.Option(
            "--fsdp-transformer-layer-cls",
            help=(
                "Transformer block class to auto-wrap for FSDP, e.g. "
                "LlamaDecoderLayer. Required when --fsdp includes auto_wrap."
            ),
        ),
    ] = "",
    save_only_model: Annotated[
        bool,
        typer.Option(
            "--save-only-model/--save-full-state",
            help=(
                "Save only model weights, or include optimiser/scheduler state "
                "for resumable long runs. Use --save-full-state with FSDP."
            ),
        ),
    ] = True,
    optim: Annotated[
        str,
        typer.Option(
            "--optim",
            help=(
                "Trainer optimiser name. Empty keeps the existing default: "
                "paged_adamw_8bit for full FT, adamw_torch for QLoRA."
            ),
        ),
    ] = "",
    source_bullets: Annotated[
        str,
        typer.Option(
            "--source-bullets",
            help=(
                "Paper_a bullet selection: 'all' (D9 default; one example per "
                "bullet) or 'longest' (one example using the longest bullet)."
            ),
        ),
    ] = "all",
    target_bullet: Annotated[
        str,
        typer.Option(
            "--target-bullet",
            help=(
                "Paper_b target selection: 'random' (D9 default; sample per "
                "row), 'longest' (use paper_b's longest bullet, fixed), or "
                "'cartesian' (emit one example per paper_a bullet x paper_b "
                "bullet pair; deterministic full-coverage at the cost of "
                "dataset inflation)."
            ),
        ),
    ] = "random",
    graph_sampling: Annotated[
        str,
        typer.Option(
            "--graph-sampling",
            help="Pair sampling distribution: 'uniform' (default) treats every "
            "gold pair equally; 'weighted' replicates each pair by an integer "
            "factor proportional to its per-facet scores_combined edge weight "
            "(requires --candidates-dir). Pairs with no graph edge still "
            "contribute at least 1 example.",
        ),
    ] = "uniform",
    candidates_dir: Annotated[
        Path | None,
        typer.Option(
            "--candidates-dir",
            help="Directory with facet_candidates_*.npz files. Required when "
            "--graph-sampling=weighted.",
        ),
    ] = None,
    graph_sampling_max_copies: Annotated[
        int,
        typer.Option(
            "--graph-sampling-max-copies",
            help="Replication ceiling for the strongest-graph-edge pair when "
            "--graph-sampling=weighted. Larger values amplify the graph prior.",
        ),
    ] = 5,
    leaf_prefix_dir: Annotated[
        Path | None,
        typer.Option(
            "--leaf-prefix-dir",
            help="Directory containing paper_c2t_docids.jsonl (output of "
            "build-c2t-index). When set, prepends the per-(facet, paper_b) "
            "cluster-label prefix to each training target — the D1 variant.",
        ),
    ] = None,
    min_target_views: Annotated[
        int,
        typer.Option(
            "--min-target-views",
            help="Minimum training views per (paper_b, facet, bullet) target. "
            "After the main expansion, any under-trained bullet gets extra "
            "rows synthesised by sampling a paper_a from paper_b's gold "
            "neighbours. 0 = disabled (default). This is the Option B fix "
            "for DocID-training starvation under graph-weighted sampling.",
        ),
    ] = 0,
    min_views_graph_fallback: Annotated[
        int,
        typer.Option(
            "--min-views-graph-fallback",
            help="When a paper_b has NO gold paper_a (the min-target-views "
            "floor cannot reach it), sample paper_a from its top-N graph "
            "neighbours instead (N = this value). 0 = disabled (default). "
            "Requires --candidates-dir.",
        ),
    ] = 0,
    pseudo_gold_oversample_sources: Annotated[
        int,
        typer.Option(
            "--pseudo-gold-oversample-sources",
            help="Experiment 'b': seed synthetic pseudo-gold BEFORE the main "
            "expansion so it rides the same --graph-sampling oversampling and "
            "--min-target-views floor as real gold. For each gold-isolated "
            "(facet, paper_b), seed up to N synthetic (neighbour->paper_b) rows "
            "from its strongest graph neighbours (symmetric edge weight). "
            "0 = disabled (default; keep floor-only fallback). Requires "
            "--candidates-dir.",
        ),
    ] = 0,
    min_views_weighted_source: Annotated[
        bool,
        typer.Option(
            "--min-views-weighted-source/--no-min-views-weighted-source",
            help="Experiment 'a': importance-sample the --min-target-views "
            "floor's synthetic source by scores_combined edge weight (favour the "
            "strongest gold/graph neighbour) instead of uniformly. Keeps the "
            "floor's exposure count unchanged. Requires --candidates-dir.",
        ),
    ] = False,
    graph_reweighting: Annotated[
        str,
        typer.Option(
            "--graph-reweighting",
            help="Per-row CE loss reweighting: 'off' (default) or "
            "'weighted' (multiply per-row CE by a continuous weight in "
            "[1, --graph-reweighting-max-weight] derived from "
            "scores_combined; requires --candidates-dir). Continuous "
            "analogue of --graph-sampling=weighted; can be stacked with "
            "it or used in its place.",
        ),
    ] = "off",
    graph_reweighting_max_weight: Annotated[
        float,
        typer.Option(
            "--graph-reweighting-max-weight",
            help="Ceiling for per-row loss weight when "
            "--graph-reweighting=weighted. Mirrors "
            "--graph-sampling-max-copies.",
        ),
    ] = 5.0,
    hard_negatives_source: Annotated[
        str,
        typer.Option(
            "--hard-negatives-source",
            help="Per-row hard-negative sampler: 'off' (default) or "
            "'graph_candidates' (sample K bullets per row from the top of "
            "facet_candidates_<facet>.npz for paper_a, gold-masked). Adds "
            "an unlikelihood term via WeightedCETrainer. Requires "
            "--candidates-dir.",
        ),
    ] = "off",
    hard_negatives_pool_top_k: Annotated[
        int,
        typer.Option(
            "--hard-negatives-pool-top-k",
            help="Candidate-pool depth for graph hard negatives. Smaller "
            "= harder negatives but narrower pool.",
        ),
    ] = 50,
    num_hard_negatives: Annotated[
        int,
        typer.Option(
            "--num-hard-negatives",
            help="K negative bullets per training row when "
            "--hard-negatives-source=graph_candidates.",
        ),
    ] = 8,
    hard_negatives_lambda: Annotated[
        float,
        typer.Option(
            "--hard-negatives-lambda",
            help="Mixing weight on the per-row unlikelihood term: "
            "loss = CE(gold) + lambda * UL(neg).",
        ),
    ] = 1.0,
    target_smoothing_epsilon: Annotated[
        float,
        typer.Option(
            "--target-smoothing-epsilon",
            help="Graph-smoothed soft targets: at each trie branch point, "
            "keep this much label mass on sibling tokens leading to other "
            "gold papers of the same query (weighted by scores_combined). "
            "0 = disabled (default). Uses --candidates-dir weights when "
            "available; uniform slice weights otherwise.",
        ),
    ] = 0.0,
    target_smoothing_max_entries: Annotated[
        int,
        typer.Option(
            "--target-smoothing-max-entries",
            help="Cap on soft-target entries per training row (kept by "
            "descending weight, renormalised per position).",
        ),
    ] = 64,
    smoothed_bullets: Annotated[
        Path | None,
        typer.Option(
            "--smoothed-bullets",
            help="Graph-smoothed DocID map (JSONL from 'glen smooth-bullets'). "
            "Replaces every target bullet with its neighbourhood-attested "
            "version; prompts stay verbatim. The eval index must be built "
            "with the same map.",
        ),
    ] = None,
) -> None:
    """Fine-tune with QLoRA (default) or full FT on per-bullet GLEN examples (D9).

    Each (paper_a, paper_b, facet) retrieval pair becomes one example per
    bullet of paper_a, expanding the training set ~4x. Input format:

        Title: {paper_a title}
        {Facet}:
        - {bullet_i}
          > "{evidence_i}"

    No indexing pairs are included (99.9% retrieval-target coverage confirmed).
    """
    from transformers import Trainer, TrainingArguments

    from genret.training._glen_callbacks import FinalStepCallback
    from genret.training.config import LoRAConfig, QuantConfig, TrainingConfig
    from genret.training.data import collate_fn
    from genret.training.glen_bullet_dataset import (
        GlenBulletDataset,
        GlenBulletDatasetConfig,
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
    fsdp_mode = fsdp.strip()
    fsdp_layer_cls = fsdp_transformer_layer_cls.strip()

    if fsdp_mode and not full_finetune:
        die("--fsdp is only supported with --full-finetune")
    if "auto_wrap" in fsdp_mode.split() and not fsdp_layer_cls:
        die("--fsdp containing auto_wrap requires --fsdp-transformer-layer-cls")
    if resume_from_checkpoint is not None and save_only_model:
        die(
            "--resume-from-checkpoint requires --save-full-state; "
            "--save-only-model omits optimiser/scheduler state"
        )

    model, tokeniser = setup_model_and_tokeniser(
        model_name,
        lora_cfg,
        quant_cfg,
        extra_special_tokens=[],
        full_finetune=full_finetune,
        distributed_training=bool(fsdp_mode) or None,
    )

    from genret.training.codebook_data import FACET_NAMES as _FN_BULLETS

    facets_arg = (
        tuple(f.strip() for f in facets_subset.split(",") if f.strip())
        if facets_subset
        else _FN_BULLETS
    )

    if source_bullets not in {"all", "longest"}:
        die(f"--source-bullets must be 'all' or 'longest', got {source_bullets!r}")
    if target_bullet not in {"random", "longest", "cartesian"}:
        die(
            f"--target-bullet must be 'random', 'longest', or 'cartesian', "
            f"got {target_bullet!r}"
        )
    if graph_sampling not in {"uniform", "weighted"}:
        die(f"--graph-sampling must be 'uniform' or 'weighted', got {graph_sampling!r}")
    if graph_sampling == "weighted" and candidates_dir is None:
        die("--graph-sampling=weighted requires --candidates-dir")
    if graph_reweighting not in {"off", "weighted"}:
        die(
            f"--graph-reweighting must be 'off' or 'weighted', "
            f"got {graph_reweighting!r}"
        )
    if graph_reweighting == "weighted" and candidates_dir is None:
        die("--graph-reweighting=weighted requires --candidates-dir")
    if graph_reweighting_max_weight < 1.0:
        die(
            f"--graph-reweighting-max-weight must be >= 1.0, "
            f"got {graph_reweighting_max_weight}"
        )
    if hard_negatives_source not in {"off", "graph_candidates"}:
        die(
            f"--hard-negatives-source must be 'off' or 'graph_candidates', "
            f"got {hard_negatives_source!r}"
        )
    if hard_negatives_source == "graph_candidates" and candidates_dir is None:
        die("--hard-negatives-source=graph_candidates requires --candidates-dir")
    if num_hard_negatives < 1:
        die(f"--num-hard-negatives must be >= 1, got {num_hard_negatives}")
    if hard_negatives_pool_top_k < num_hard_negatives:
        die(
            f"--hard-negatives-pool-top-k ({hard_negatives_pool_top_k}) must be "
            f">= --num-hard-negatives ({num_hard_negatives})"
        )
    if hard_negatives_lambda < 0.0:
        die(f"--hard-negatives-lambda must be >= 0, got {hard_negatives_lambda}")
    if not 0.0 <= target_smoothing_epsilon < 1.0:
        die(
            f"--target-smoothing-epsilon must be in [0, 1), "
            f"got {target_smoothing_epsilon}"
        )
    source_bullets_mode = cast(Literal["all", "longest"], source_bullets)
    target_bullet_mode = cast(Literal["random", "longest", "cartesian"], target_bullet)
    graph_sampling_mode = cast(Literal["uniform", "weighted"], graph_sampling)
    graph_reweighting_mode = cast(Literal["off", "weighted"], graph_reweighting)
    hard_negatives_source_mode = cast(
        Literal["off", "graph_candidates"], hard_negatives_source
    )

    def _make_dataset(split_path: Path | None) -> GlenBulletDataset:
        split_ids: set[CorpusId] | None = None
        if split_path is not None:
            if not split_path.exists():
                die(f"Split file not found: {split_path}")
            split_ids = load_ids_from_file(split_path)
            if not split_ids:
                die(f"Split file {split_path} contains no ids")
        return GlenBulletDataset(
            GlenBulletDatasetConfig(
                pairs_path=pairs,
                facets_path=facets,
                paper_graphs_path=paper_graphs,
                facets_type=facets_type,
                split_ids=split_ids,
                facets=facets_arg,
                source_bullets=source_bullets_mode,
                target_bullet=target_bullet_mode,
                graph_sampling=graph_sampling_mode,
                candidates_dir=candidates_dir,
                graph_sampling_max_copies=graph_sampling_max_copies,
                graph_reweighting=graph_reweighting_mode,
                graph_reweighting_max_weight=graph_reweighting_max_weight,
                hard_negatives_source=hard_negatives_source_mode,
                hard_negatives_pool_top_k=hard_negatives_pool_top_k,
                num_hard_negatives=num_hard_negatives,
                hard_negatives_lambda=hard_negatives_lambda,
                leaf_prefix_dir=leaf_prefix_dir,
                min_target_views=min_target_views,
                min_views_graph_fallback=min_views_graph_fallback,
                min_views_weighted_source=min_views_weighted_source,
                pseudo_gold_oversample_sources=pseudo_gold_oversample_sources,
                target_smoothing_epsilon=target_smoothing_epsilon,
                target_smoothing_max_entries=target_smoothing_max_entries,
                smoothed_bullets_path=smoothed_bullets,
            ),
            rng=rng,
            tokeniser=tokeniser,
        )

    train_dataset = _make_dataset(train_split)
    dev_dataset = _make_dataset(dev_split) if dev_split is not None else None

    bound_collate = functools.partial(
        collate_fn, tokeniser=tokeniser, max_seq_length=max_seq_length
    )
    fsdp_config = (
        {"transformer_layer_cls_to_wrap": fsdp_layer_cls} if fsdp_layer_cls else None
    )
    optim_name = optim.strip() or (
        "paged_adamw_8bit" if full_finetune else "adamw_torch"
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
        # Only save model weights, not optimiser state — we use checkpoints for
        # post-hoc R@k selection and never resume from them. Drops per-checkpoint
        # disk usage from ~7 GB to ~2.4 GB for Llama-3.2-1B full FT.
        save_only_model=save_only_model,
        load_best_model_at_end=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optim_name,
        fsdp=fsdp_mode,
        fsdp_config=fsdp_config,
    )

    if target_smoothing_epsilon > 0 and graph_reweighting == "weighted":
        die(
            "Combining --target-smoothing-epsilon and --graph-reweighting "
            "is not implemented. Pick one."
        )
    if target_smoothing_epsilon > 0 and hard_negatives_source != "off":
        die(
            "Combining --target-smoothing-epsilon and --hard-negatives-source "
            "is not implemented. Pick one."
        )

    needs_weighted_trainer = (
        graph_reweighting == "weighted" or hard_negatives_source != "off"
    )

    if target_smoothing_epsilon > 0:
        from genret.training.glen_smoothed_loss import SmoothedTargetTrainer

        trainer = SmoothedTargetTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=bound_collate,
            callbacks=[FinalStepCallback()],
            smoothing_epsilon=target_smoothing_epsilon,
        )
    elif needs_weighted_trainer:
        from genret.training.glen_weighted_loss import WeightedCETrainer

        trainer = WeightedCETrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=bound_collate,
            callbacks=[FinalStepCallback()],
            hard_negatives_lambda=hard_negatives_lambda
            if hard_negatives_source != "off"
            else 0.0,
        )
    else:
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


def cmd_train_glen_marginal(
    pairs: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help="Training pairs JSONL from build-codebook-data.",
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
    paper_graphs: Annotated[
        Path,
        typer.Option(
            "--paper-graphs",
            help="JSONL with per-paper metadata (must include 'title').",
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
    batch_size: Annotated[int, typer.Option("--batch-size")] = 2,
    learning_rate: Annotated[float, typer.Option("--lr")] = 2e-4,
    lora_rank: Annotated[int, typer.Option("--lora-rank")] = 16,
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = 4,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length")] = 512,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 20,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 1000,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 0.0,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    train_split: Annotated[Path | None, typer.Option("--train-split")] = None,
    resume_from_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--resume-from-checkpoint",
            help="Checkpoint directory to resume from.",
        ),
    ] = None,
    facets_subset: Annotated[
        str | None,
        typer.Option(
            "--facets-subset",
            help=(
                "Comma-separated facet names whose bullets become training "
                "inputs (e.g. 'contributions'). Defaults to all four."
            ),
        ),
    ] = None,
    full_finetune: Annotated[
        bool,
        typer.Option(
            "--full-finetune/--no-full-finetune",
            help="Skip QLoRA and fine-tune all weights in bf16.",
        ),
    ] = False,
    source_bullets: Annotated[
        str,
        typer.Option(
            "--source-bullets",
            help="'all' (default) or 'longest' — paper_a bullet selection.",
        ),
    ] = "all",
) -> None:
    """Marginal-likelihood training: ``-log Σ p(d|q)`` over paper_b's bullets."""
    from transformers import TrainingArguments

    from genret.training._glen_callbacks import FinalStepCallback
    from genret.training.codebook_data import FACET_NAMES as _FN
    from genret.training.config import LoRAConfig, QuantConfig
    from genret.training.glen_marginal_dataset import (
        GlenMarginalDataset,
        GlenMarginalDatasetConfig,
        marginal_collator,
    )
    from genret.training.glen_marginal_loss import MarginalLikelihoodTrainer
    from genret.training.model import setup_model_and_tokeniser
    from genret.types import CorpusId

    rng = random.Random(0)

    facets_arg = (
        tuple(f.strip() for f in facets_subset.split(",") if f.strip())
        if facets_subset
        else _FN
    )
    if source_bullets not in {"all", "longest"}:
        die(f"--source-bullets must be 'all' or 'longest', got {source_bullets!r}")

    lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_rank * 2)
    quant_cfg = QuantConfig()
    model, tokeniser = setup_model_and_tokeniser(
        model_name,
        lora_cfg,
        quant_cfg,
        extra_special_tokens=[],
        full_finetune=full_finetune,
    )

    split_ids: set[CorpusId] | None = None
    if train_split is not None:
        if not train_split.exists():
            die(f"Split file not found: {train_split}")
        split_ids = load_ids_from_file(train_split)
        if not split_ids:
            die(f"Split file {train_split} contains no ids")

    train_dataset = GlenMarginalDataset(
        GlenMarginalDatasetConfig(
            pairs_path=pairs,
            facets_path=facets,
            paper_graphs_path=paper_graphs,
            facets_type=facets_type,
            split_ids=split_ids,
            facets=facets_arg,
            source_bullets=source_bullets,
        ),
        rng=rng,
    )

    bound_collate = functools.partial(
        marginal_collator,
        tokeniser=tokeniser,
        max_seq_length=max_seq_length,
    )

    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        save_only_model=True,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if full_finetune else "adamw_torch",
        remove_unused_columns=False,
    )

    trainer = MarginalLikelihoodTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
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


def cmd_train_glen_contrastive(
    pairs: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help="Training pairs JSONL from build-codebook-data.",
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
    paper_graphs: Annotated[
        Path,
        typer.Option(
            "--paper-graphs",
            help="JSONL with per-paper metadata (must include 'title').",
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
    batch_size: Annotated[int, typer.Option("--batch-size")] = 1,
    learning_rate: Annotated[float, typer.Option("--lr")] = 2e-4,
    lora_rank: Annotated[int, typer.Option("--lora-rank")] = 16,
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = 8,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length")] = 512,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 20,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 1000,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 0.0,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    train_split: Annotated[Path | None, typer.Option("--train-split")] = None,
    resume_from_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--resume-from-checkpoint",
            help="Checkpoint directory to resume from.",
        ),
    ] = None,
    facets_subset: Annotated[
        str | None,
        typer.Option(
            "--facets-subset",
            help=(
                "Comma-separated facet names whose bullets become training "
                "inputs (e.g. 'contributions'). Defaults to all four."
            ),
        ),
    ] = None,
    full_finetune: Annotated[
        bool,
        typer.Option(
            "--full-finetune/--no-full-finetune",
            help="Skip QLoRA and fine-tune all weights in bf16.",
        ),
    ] = False,
    source_bullets: Annotated[
        str,
        typer.Option(
            "--source-bullets",
            help="'all' (default) or 'longest' — paper_a bullet selection.",
        ),
    ] = "all",
    num_negatives: Annotated[
        int,
        typer.Option(
            "--num-negatives",
            help=(
                "Number of popularity-weighted negative papers sampled per "
                "example. One bullet per negative paper is added as a "
                "negative target."
            ),
        ),
    ] = 8,
    popularity_temperature: Annotated[
        float,
        typer.Option(
            "--popularity-temperature",
            help=(
                "Temperature on the popularity sampling distribution. "
                "1.0 = proportional to raw counts; <1 sharpens toward the "
                "most-popular papers; >1 flattens toward uniform."
            ),
        ),
    ] = 1.0,
) -> None:
    """Contrastive training: marginal-LL numerator + popularity-weighted negatives."""
    from transformers import TrainingArguments

    from genret.training._glen_callbacks import FinalStepCallback
    from genret.training.codebook_data import FACET_NAMES as _FN
    from genret.training.config import LoRAConfig, QuantConfig
    from genret.training.glen_contrastive_dataset import (
        GlenContrastiveDataset,
        GlenContrastiveDatasetConfig,
        contrastive_collator,
    )
    from genret.training.glen_contrastive_loss import ContrastiveTrainer
    from genret.training.model import setup_model_and_tokeniser
    from genret.types import CorpusId

    rng = random.Random(0)

    facets_arg = (
        tuple(f.strip() for f in facets_subset.split(",") if f.strip())
        if facets_subset
        else _FN
    )
    if source_bullets not in {"all", "longest"}:
        die(f"--source-bullets must be 'all' or 'longest', got {source_bullets!r}")

    lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_rank * 2)
    quant_cfg = QuantConfig()
    model, tokeniser = setup_model_and_tokeniser(
        model_name,
        lora_cfg,
        quant_cfg,
        extra_special_tokens=[],
        full_finetune=full_finetune,
    )

    split_ids: set[CorpusId] | None = None
    if train_split is not None:
        if not train_split.exists():
            die(f"Split file not found: {train_split}")
        split_ids = load_ids_from_file(train_split)
        if not split_ids:
            die(f"Split file {train_split} contains no ids")

    train_dataset = GlenContrastiveDataset(
        GlenContrastiveDatasetConfig(
            pairs_path=pairs,
            facets_path=facets,
            paper_graphs_path=paper_graphs,
            facets_type=facets_type,
            split_ids=split_ids,
            facets=facets_arg,
            source_bullets=source_bullets,
            num_negatives=num_negatives,
            popularity_temperature=popularity_temperature,
        ),
        rng=rng,
    )

    bound_collate = functools.partial(
        contrastive_collator,
        tokeniser=tokeniser,
        max_seq_length=max_seq_length,
    )

    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        save_only_model=True,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if full_finetune else "adamw_torch",
        remove_unused_columns=False,
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
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


def register(app: typer.Typer) -> None:
    """Attach per-bullet training commands to the GLEN typer app."""
    app.command(name="train-bullets")(cmd_train_glen_bullets)
    app.command(name="train-marginal")(cmd_train_glen_marginal)
    app.command(name="train-contrastive")(cmd_train_glen_contrastive)


__all__ = [
    "cmd_train_glen_bullets",
    "cmd_train_glen_contrastive",
    "cmd_train_glen_marginal",
    "register",
]
