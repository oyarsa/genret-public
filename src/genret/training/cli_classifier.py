# pyright: basic
# ruff: noqa: PLC0415  Keep heavy imports inside command bodies (transformers / peft).
"""CLI for the multi-label classifier sanity check.

Registered under ``training classifier`` in the main CLI. Two commands:

- ``train``    — fine-tune an LM encoder + zero-init head with BCE.
- ``evaluate`` — forward pass per query bullet, average logits, top-K corpus IDs.
"""

from __future__ import annotations

import functools
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import torch
import typer

from genret.paper_parsing import FacetEvidence, FacetRecordType, Facets
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_dataset import _build_messages, _facet_text
from genret.training.codebook_evaluate import (
    _load_facets_and_evidence,
    queries_from_split,
)
from genret.types import CorpusId
from genret.utils import die, load_ids_from_file, make_app, open_auto

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from genret.training.classifier_dataset import ClassifierDataset
    from genret.training.classifier_model import PaperClassifier

logger = logging.getLogger(__name__)

classifier_app = make_app("Multi-label paper-classifier commands")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@classifier_app.command(name="train")
def cmd_train_classifier(
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
            help="JSONL with per-paper title + corpusid metadata.",
            show_default=False,
        ),
    ],
    corpus_split: Annotated[
        Path,
        typer.Option(
            "--corpus-split",
            help=(
                "One corpus ID per line. Defines the class ordering for the "
                "head; conventionally the union of train/dev/test splits."
            ),
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Checkpoint output directory."),
    ],
    model_name: Annotated[
        str, typer.Option("--model", "-m")
    ] = "meta-llama/Llama-3.2-1B-Instruct",
    epochs: Annotated[int, typer.Option("--epochs")] = 5,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 4,
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = 4,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length")] = 512,
    learning_rate: Annotated[float, typer.Option("--lr")] = 2e-5,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 0.01,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 20,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 1000,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    train_split: Annotated[Path | None, typer.Option("--train-split")] = None,
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
    use_qlora: Annotated[
        bool,
        typer.Option(
            "--qlora/--no-qlora",
            help="Load encoder in 4-bit NF4 and attach LoRA adapters.",
        ),
    ] = False,
    lora_rank: Annotated[int, typer.Option("--lora-rank")] = 16,
    pooling: Annotated[
        str,
        typer.Option(
            "--pooling",
            help=(
                "Sequence pooling: 'mean' (mask-aware average; default) or "
                "'last' (final non-pad token; canonical for instruct LMs)."
            ),
        ),
    ] = "mean",
    pos_weight_mode: Annotated[
        str,
        typer.Option(
            "--pos-weight",
            help=(
                "BCE positive-weighting from class prevalence: 'none' "
                "(default), 'inverse_freq', or 'inverse_sqrt_freq'. The "
                "weighted variants down-weight popular classes (e.g. BERT) so "
                "the optimiser can't trivially win by always predicting them."
            ),
        ),
    ] = "none",
    head_bias_init_mode: Annotated[
        str,
        typer.Option(
            "--head-bias-init",
            help=(
                "Initialise the classifier head bias as 'zero' (default) or "
                "'prior' (logit of class prevalence). 'prior' makes the "
                "trivial bias-only solution the *starting point* rather than "
                "the attractor."
            ),
        ),
    ] = "zero",
    loss_kind: Annotated[
        str,
        typer.Option(
            "--loss",
            help=(
                "Loss function: 'bce' (default; combines with --pos-weight) "
                "or 'focal' (down-weights easy examples via (1-p_t)^gamma)."
            ),
        ),
    ] = "bce",
    focal_gamma: Annotated[
        float,
        typer.Option(
            "--focal-gamma", help="Focal loss focusing parameter (default 2.0)."
        ),
    ] = 2.0,
    focal_alpha: Annotated[
        float,
        typer.Option(
            "--focal-alpha",
            help="Focal loss class-balance parameter (default 0.25).",
        ),
    ] = 0.25,
) -> None:
    """Train a multi-label classifier head over the 1k corpus."""
    from transformers import Trainer, TrainingArguments

    from genret.training._glen_callbacks import FinalStepCallback
    from genret.training.classifier_dataset import (
        ClassifierDataset,
        ClassifierDatasetConfig,
    )
    from genret.training.classifier_model import (
        LossKind,
        PoolingMode,
        setup_classifier_model_and_tokeniser,
    )

    if pooling not in {"mean", "last"}:
        die(f"--pooling must be 'mean' or 'last', got {pooling!r}")
    if pos_weight_mode not in {"none", "inverse_freq", "inverse_sqrt_freq"}:
        die(
            f"--pos-weight must be 'none', 'inverse_freq', or "
            f"'inverse_sqrt_freq', got {pos_weight_mode!r}"
        )
    if head_bias_init_mode not in {"zero", "prior"}:
        die(f"--head-bias-init must be 'zero' or 'prior', got {head_bias_init_mode!r}")
    if loss_kind not in {"bce", "focal"}:
        die(f"--loss must be 'bce' or 'focal', got {loss_kind!r}")
    pooling_mode = cast("PoolingMode", pooling)
    loss_kind_lit = cast("LossKind", loss_kind)

    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    facets_arg = (
        tuple(f.strip() for f in facets_subset.split(",") if f.strip())
        if facets_subset
        else FACET_NAMES
    )

    train_split_ids: set[CorpusId] | None = None
    if train_split is not None:
        if not train_split.exists():
            die(f"Split file not found: {train_split}")
        train_split_ids = load_ids_from_file(train_split)
        if not train_split_ids:
            die(f"Split file {train_split} contains no ids")

    dataset_cfg = ClassifierDatasetConfig(
        pairs_path=pairs,
        facets_path=facets,
        paper_graphs_path=paper_graphs,
        corpus_split_path=corpus_split,
        facets_type=facets_type,
        train_split_ids=train_split_ids,
        facets=facets_arg,
    )
    train_dataset = ClassifierDataset(dataset_cfg, rng=rng)
    num_classes = train_dataset.num_classes

    pos_weight_tensor = _compute_pos_weight(train_dataset, mode=pos_weight_mode)
    head_bias_init = _compute_head_bias_init(train_dataset, mode=head_bias_init_mode)

    setup = setup_classifier_model_and_tokeniser(
        model_name,
        num_classes=num_classes,
        use_qlora=use_qlora,
        lora_rank=lora_rank,
        pooling=pooling_mode,
        pos_weight=pos_weight_tensor,
        head_bias_init=head_bias_init,
        loss_kind=loss_kind_lit,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )

    bound_collate = functools.partial(
        _classifier_collate,
        tokeniser=setup.tokeniser,
        max_seq_length=max_seq_length,
        num_classes=num_classes,
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
        # gradient_checkpointing is enabled directly on the encoder in
        # setup_classifier_model_and_tokeniser. The Trainer's flag would
        # try to call gradient_checkpointing_enable on PaperClassifier
        # itself (which doesn't expose that method).
        gradient_checkpointing=False,
        optim="paged_adamw_8bit" if not use_qlora else "adamw_torch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=setup.model,
        args=args,
        train_dataset=train_dataset,  # type: ignore[arg-type]
        data_collator=bound_collate,
        callbacks=[FinalStepCallback()],
    )
    trainer.train()
    trainer.save_model(str(output))
    setup.tokeniser.save_pretrained(str(output))
    _save_corpus_map(
        output,
        train_dataset.corpus_ids,
        model_name=model_name,
        use_qlora=use_qlora,
        pooling=pooling_mode,
    )


def _classifier_collate(
    batch: list[object],
    *,
    tokeniser: object,
    max_seq_length: int,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    """Collate a batch of ``ClassifierExample`` into tensors for the model."""
    from genret._shed.transformers import apply_chat_template_tensor

    input_ids_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for example in batch:
        ids = apply_chat_template_tensor(tokeniser, example.prompt_messages)  # type: ignore[attr-defined]
        ids = ids[0, :max_seq_length]
        input_ids_list.append(ids)
        labels_list.append(example.labels)  # type: ignore[attr-defined]

    pad_token_id = int(tokeniser.pad_token_id)  # type: ignore[attr-defined]
    max_len = max(t.shape[0] for t in input_ids_list)
    input_ids = torch.full(
        (len(input_ids_list), max_len), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        # Left padding (matches tokeniser.padding_side = "left" used elsewhere).
        offset = max_len - ids.shape[0]
        input_ids[i, offset:] = ids
        attention_mask[i, offset:] = 1
    labels = torch.stack(labels_list)
    assert labels.shape == (len(input_ids_list), num_classes), (
        f"label shape {labels.shape} mismatches expected "
        f"({len(input_ids_list)}, {num_classes})"
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _compute_pos_weight(
    dataset: ClassifierDataset, *, mode: str
) -> torch.Tensor | None:
    """Return BCE ``pos_weight`` tensor by class prevalence, or ``None``.

    ``inverse_freq``    → ``(N - n_pos) / n_pos``.
    ``inverse_sqrt_freq`` → ``sqrt((N - n_pos) / n_pos)`` (less extreme).
    """
    if mode == "none":
        return None
    counts = dataset.class_positive_counts()
    total = float(len(dataset))
    safe_counts = counts.clamp(min=1.0)
    raw = (total - counts) / safe_counts
    raw = raw.clamp(min=1e-6)
    if mode == "inverse_sqrt_freq":
        raw = raw.sqrt()
    # Clamp to a sane ceiling so the rarest classes don't dominate.
    raw = raw.clamp(max=100.0)
    logger.info(
        "pos_weight (%s): min=%.3f, median=%.3f, max=%.3f",
        mode,
        float(raw.min()),
        float(raw.median()),
        float(raw.max()),
    )
    return raw


def _compute_head_bias_init(
    dataset: ClassifierDataset, *, mode: str
) -> torch.Tensor | None:
    """Return per-class bias init = ``logit(prior_i)``, or ``None`` for zero-init."""
    if mode == "zero":
        return None
    counts = dataset.class_positive_counts()
    total = float(len(dataset))
    eps = 1e-4
    prior = (counts / max(total, 1.0)).clamp(min=eps, max=1.0 - eps)
    bias = torch.logit(prior)
    logger.info(
        "head bias init (%s): min=%.3f, median=%.3f, max=%.3f",
        mode,
        float(bias.min()),
        float(bias.median()),
        float(bias.max()),
    )
    return bias


def _save_corpus_map(
    output: Path,
    corpus_ids: list[CorpusId],
    *,
    model_name: str,
    use_qlora: bool,
    pooling: str,
) -> None:
    """Persist label ordering plus encoder identity for the eval-time loader."""
    data = {
        "corpus_ids": [int(cid) for cid in corpus_ids],
        "model_name": model_name,
        "use_qlora": use_qlora,
        "pooling": pooling,
    }
    (output / "classifier_corpus_map.json").write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@classifier_app.command(name="evaluate")
def cmd_evaluate_classifier(
    checkpoint: Annotated[
        Path,
        typer.Argument(
            help="Path to a checkpoint produced by ``train-classifier``.",
            show_default=False,
        ),
    ],
    facets: Annotated[
        Path,
        typer.Option("--facets", help="Facets JSONL.", show_default=False),
    ],
    queries: Annotated[
        Path,
        typer.Option(
            "--queries",
            help="Split file (one corpus ID per line) of query papers.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output JSONL path.", show_default=False),
    ],
    top_n: Annotated[
        int, typer.Option("--top-n", help="Top-N papers to write per query.")
    ] = 50,
    facets_subset: Annotated[
        str | None,
        typer.Option(
            "--facets-subset",
            help=(
                "Comma-separated facet names whose bullets feed the query "
                "encoder. Defaults to all four. Logits are mean-averaged "
                "across bullet calls."
            ),
        ),
    ] = None,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    model_override: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help=(
                "Override the encoder model identity. By default it is read "
                "from the checkpoint's classifier_corpus_map.json."
            ),
        ),
    ] = None,
) -> None:
    """Score every query paper against the trained head and write top-N JSONL."""
    from safetensors.torch import load_file

    from genret.training.classifier_model import (
        PoolingMode,
        setup_classifier_model_and_tokeniser,
    )

    facets_arg = (
        tuple(f.strip() for f in facets_subset.split(",") if f.strip())
        if facets_subset
        else FACET_NAMES
    )

    corpus_map_path = checkpoint / "classifier_corpus_map.json"
    if not corpus_map_path.exists():
        die(f"Missing classifier_corpus_map.json at {corpus_map_path}")
    corpus_map = json.loads(corpus_map_path.read_text())
    corpus_ids: list[CorpusId] = [CorpusId(int(c)) for c in corpus_map["corpus_ids"]]
    num_classes = len(corpus_ids)

    model_name = model_override or str(corpus_map.get("model_name") or "")
    if not model_name:
        die(
            f"checkpoint {checkpoint} has no model_name in "
            "classifier_corpus_map.json; pass --model to override."
        )
    use_qlora = bool(corpus_map.get("use_qlora", False))
    pooling = str(corpus_map.get("pooling", "mean"))
    if pooling not in {"mean", "last"}:
        die(f"invalid pooling mode {pooling!r} in {corpus_map_path}")
    pooling_mode = cast("PoolingMode", pooling)

    setup = setup_classifier_model_and_tokeniser(
        model_name,
        num_classes=num_classes,
        use_qlora=use_qlora,
        pooling=pooling_mode,
    )

    # Load the saved wrapper state-dict from model.safetensors into the
    # freshly-constructed PaperClassifier (encoder + head).
    state_path = checkpoint / "model.safetensors"
    if not state_path.exists():
        die(f"Missing model.safetensors at {state_path}")
    state = load_file(str(state_path))
    missing, unexpected = setup.model.load_state_dict(state, strict=False)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %d", len(unexpected))
    if missing:
        # When the encoder was QLoRA-trained, the missing keys are the
        # frozen quantised base weights; that's expected.
        logger.info("Missing keys (likely frozen base weights): %d", len(missing))
    setup.model.eval()
    device = next(setup.model.parameters()).device

    facets_map, evidence_map = _load_facets_and_evidence(facets, facets_type)
    query_ids = queries_from_split(queries)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open_auto(output, "w") as fh:
        for qid in query_ids:
            top = _score_one(
                qid=qid,
                facets_map=facets_map,
                evidence_map=evidence_map,
                facets=facets_arg,
                model=setup.model,
                tokeniser=setup.tokeniser,
                device=device,
                corpus_ids=corpus_ids,
                top_n=top_n,
            )
            if top is None:
                continue
            fh.write(json.dumps(top) + "\n")


def _score_one(
    *,
    qid: CorpusId,
    facets_map: dict[CorpusId, Facets],
    evidence_map: dict[CorpusId, FacetEvidence],
    facets: tuple[str, ...],
    model: PaperClassifier,
    tokeniser: PreTrainedTokenizerBase,
    device: torch.device,
    corpus_ids: list[CorpusId],
    top_n: int,
) -> dict[str, object] | None:
    """Score one query paper across all its bullets and pick the top-N papers."""
    from genret._shed.transformers import apply_chat_template_tensor
    from genret.training.glen_docids import split_bullets

    paper_facets = facets_map.get(qid)
    if paper_facets is None:
        logger.warning("Query %d not in facets map; skipping", qid)
        return None
    evidence = evidence_map.get(qid)

    logits_sum: torch.Tensor | None = None
    n_inputs = 0
    for facet in facets:
        text = _facet_text(paper_facets, facet, evidence)
        if not text:
            continue
        for bullet in split_bullets(text) or [text]:
            messages = _build_messages(facet, bullet, kind="retrieval")
            ids = apply_chat_template_tensor(tokeniser, messages).to(device)
            mask = torch.ones_like(ids)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask)
            logits = out["logits"].squeeze(0).float().cpu()
            logits_sum = logits if logits_sum is None else logits_sum + logits
            n_inputs += 1

    if logits_sum is None or n_inputs == 0:
        return None
    mean_logits = logits_sum / n_inputs

    # Mask out self.
    cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    self_idx = cid_to_idx.get(qid)
    if self_idx is not None:
        mean_logits[self_idx] = -float("inf")

    top_scores, top_idx = torch.topk(mean_logits, k=min(top_n, len(corpus_ids)))
    results: list[dict[str, object]] = []
    for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_idx.tolist())):
        cid = int(corpus_ids[idx])
        results.append(
            {
                "corpusid": cid,
                "score": round(float(score), 6),
                "facets": list(facets),
                "facet_ranks": dict.fromkeys(facets, rank),
            }
        )
    return {"corpusid": int(qid), "results": results}


__all__ = ["classifier_app"]
