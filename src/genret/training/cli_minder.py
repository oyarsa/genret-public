# pyright: basic
# ruff: noqa: PLC0415  Enable function-level imports
"""CLI commands for the MINDER-style three-view baseline (Variant A).

See `docs/plans/20260725-minder-baseline.md` for the design. Not a deployment
candidate: a baseline measuring MINDER's own identifier scheme and recipe
against our facet-bullet DocIDs, on our corpus and gold.
"""

from __future__ import annotations

import functools
import logging
import random
from pathlib import Path
from typing import Annotated

import typer

from genret.paper_parsing import FacetRecordType
from genret.types import CorpusId
from genret.utils import die, load_ids_from_file, make_app

logger = logging.getLogger(__name__)

minder_app = make_app("MINDER-style three-view baseline (Variant A)")


def cmd_train_minder(
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
            help="JSONL with per-paper facet bullets (the pseudo-query view).",
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
    abstracts: Annotated[
        Path,
        typer.Option(
            "--abstracts",
            help="JSONL with per-paper 'corpusid' and 'abstract' (the substring view).",
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
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = 4,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length")] = 512,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 20,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 500,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 0.0,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    train_split: Annotated[Path | None, typer.Option("--train-split")] = None,
    full_finetune: Annotated[
        bool,
        typer.Option(
            "--full-finetune/--no-full-finetune",
            help="Skip QLoRA and fine-tune all weights in bf16 (default: on, "
            "matching our own recipe -- full FT beats LoRA for DocID "
            "memorisation).",
        ),
    ] = True,
    span_tokens: Annotated[
        int,
        typer.Option(
            "--span-tokens",
            help="Substring-view span length, in tokens. Matches our DocID cap.",
        ),
    ] = 12,
    min_span_tokens: Annotated[
        int,
        typer.Option(
            "--min-span-tokens",
            help="Shortest substring-view span the decoder may stop at "
            "(see minder_substrings.SubstringIndex for the corpus measurement "
            "behind the default).",
        ),
    ] = 8,
    max_query_chars: Annotated[
        int,
        typer.Option(
            "--max-query-chars",
            help="Abstract characters kept on the query side.",
        ),
    ] = 1500,
    unsupervised_blocks: Annotated[
        int,
        typer.Option(
            "--unsupervised-blocks",
            help="Times MINDER's unsupervised-exposure mix is emitted per "
            "corpus paper. 0 disables it entirely.",
        ),
    ] = 1,
    fuzzy_temperature: Annotated[
        float,
        typer.Option(
            "--fuzzy-temperature",
            help="Softmax temperature for substring-span sampling, matching "
            "the released code's default.",
        ),
    ] = 1.0,
) -> None:
    """Fine-tune on the MINDER-style three-view dataset (title/substring/pseudo-query)."""
    from transformers import Trainer, TrainingArguments

    from genret.training._glen_callbacks import FinalStepCallback
    from genret.training.config import LoRAConfig, QuantConfig
    from genret.training.data import collate_fn
    from genret.training.minder_dataset import (
        MinderDataset,
        MinderDatasetConfig,
        load_abstracts,
    )
    from genret.training.minder_substrings import build_substring_index
    from genret.training.model import setup_model_and_tokeniser

    lora_cfg = LoRAConfig()
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

    # The substring index spans the whole corpus, not just the training
    # split: decode-time queries may target any paper, and the index must
    # already know its spans at training time to select from.
    all_abstracts = load_abstracts(abstracts, keep=_all_ids(abstracts))

    def _encode(text: str) -> list[int]:
        return list(tokeniser.encode(text, add_special_tokens=False))

    index = build_substring_index(
        abstracts=all_abstracts,
        encode=_encode,
        vocab_size=len(tokeniser),
        min_span_tokens=min_span_tokens,
        max_span_tokens=span_tokens,
    )

    rng = random.Random(0)
    train_dataset = MinderDataset(
        MinderDatasetConfig(
            pairs_path=pairs,
            facets_path=facets,
            paper_graphs_path=paper_graphs,
            abstracts_path=abstracts,
            facets_type=facets_type,
            split_ids=split_ids,
            span_tokens=span_tokens,
            max_query_chars=max_query_chars,
            unsupervised_blocks=unsupervised_blocks,
            fuzzy_temperature=fuzzy_temperature,
        ),
        index=index,
        tokeniser=tokeniser,
        rng=rng,
    )

    bound_collate = functools.partial(
        collate_fn, tokeniser=tokeniser, max_seq_length=max_seq_length
    )
    optim_name = "paged_adamw_8bit" if full_finetune else "adamw_torch"

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
        optim=optim_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=bound_collate,
        callbacks=[FinalStepCallback()],
    )
    trainer.train()
    trainer.save_model(str(output))
    tokeniser.save_pretrained(str(output))


def _all_ids(abstracts_path: Path) -> set[CorpusId]:
    """Every corpus id present in the abstracts file.

    A separate pass avoids loading abstract text twice; the file at our
    scale (1k papers) makes this cheap.
    """
    from genret.utils import iter_jsonl

    return {CorpusId(int(r["corpusid"])) for r in iter_jsonl(abstracts_path)}


def cmd_evaluate_minder(
    checkpoint: Annotated[
        Path, typer.Argument(help="Fine-tuned checkpoint directory.")
    ],
    facets: Annotated[
        Path,
        typer.Option(
            "--facets", help="JSONL with per-paper facet bullets.", show_default=False
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
    abstracts: Annotated[
        Path,
        typer.Option(
            "--abstracts",
            help="JSONL with per-paper 'corpusid' and 'abstract'.",
            show_default=False,
        ),
    ],
    queries: Annotated[
        Path,
        typer.Option(
            "--queries", help="Plain-text file of corpus ids.", show_default=False
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", show_default=False)],
    beams: Annotated[int, typer.Option("--beams")] = 50,
    max_docid_tokens: Annotated[int, typer.Option("--max-docid-tokens")] = 32,
    span_tokens: Annotated[int, typer.Option("--span-tokens")] = 12,
    min_span_tokens: Annotated[int, typer.Option("--min-span-tokens")] = 8,
    max_query_chars: Annotated[int, typer.Option("--max-query-chars")] = 1500,
    facets_type: Annotated[
        FacetRecordType, typer.Option("--facets-type")
    ] = FacetRecordType.STANDARD,
    quantize: Annotated[
        str,
        typer.Option(
            "--quantize",
            help="Inference-time quantisation: 'off' (bf16, default), '4bit', '8bit'.",
        ),
    ] = "off",
) -> None:
    """Decode all three MINDER views for a query split and dump every beam.

    Writes the same schema as `genret training glen evaluate --record-docids`
    (`query/facet/rank/docid/papers/score/n_tokens/logprob_sum`), so
    `scripts/eval/minder_fusion.py` scores this dump unchanged.
    """
    from genret.paper_parsing import Facets, facets_from_type
    from genret.training._cli_helpers import _load_checkpoint
    from genret.training.codebook_evaluate import queries_from_split
    from genret.training.minder_dataset import bullets_of, load_abstracts, load_titles
    from genret.training.minder_decoding import build_title_and_pseudo_query_index
    from genret.training.minder_evaluate import evaluate_minder_retrieval
    from genret.training.minder_substrings import build_substring_index
    from genret.types import CorpusId
    from genret.utils import iter_jsonl

    model, tokeniser = _load_checkpoint(checkpoint, quantize=quantize)

    facets_map = dict[CorpusId, Facets]()
    for raw in iter_jsonl(facets):
        parsed = facets_from_type(raw, facets_type)
        facets_map[parsed.corpusid] = parsed
    corpus_ids: set[CorpusId] = set(facets_map)
    if not corpus_ids:
        die(f"No facet records found in {facets}")

    titles = load_titles(paper_graphs)
    abstracts_map = load_abstracts(abstracts, corpus_ids)
    bullets_by_paper = {cid: bullets_of(f) for cid, f in facets_map.items()}

    def _encode(text: str) -> list[int]:
        return list(tokeniser.encode(text, add_special_tokens=False))

    substring_index = build_substring_index(
        abstracts=abstracts_map,
        encode=_encode,
        vocab_size=len(tokeniser),
        min_span_tokens=min_span_tokens,
        max_span_tokens=span_tokens,
    )
    trie_index = build_title_and_pseudo_query_index(
        titles, bullets_by_paper, tokeniser, max_tokens=max_docid_tokens
    )

    query_ids = queries_from_split(queries)

    evaluate_minder_retrieval(
        model=model,
        tokeniser=tokeniser,
        trie_index=trie_index,
        substring_index=substring_index,
        queries=query_ids,
        titles=titles,
        abstracts=abstracts_map,
        output_path=output,
        num_beams=beams,
        max_docid_tokens=max_docid_tokens,
        max_query_chars=max_query_chars,
    )


def register(app: typer.Typer) -> None:
    """Attach MINDER-baseline commands to their sub-app."""
    app.command(name="train")(cmd_train_minder)
    app.command(name="evaluate")(cmd_evaluate_minder)


register(minder_app)

__all__ = [
    "cmd_evaluate_minder",
    "cmd_train_minder",
    "minder_app",
    "register",
]
