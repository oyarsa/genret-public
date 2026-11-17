# pyright: basic
# ruff: noqa: PLC0415  Enable function-level imports
"""Shared helpers for the training CLI commands."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from genret.utils import die, resolve_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _validate_data_files(splits_dir: Path) -> Path:
    """Check that required data files exist and return the processed directory.

    Args:
        splits_dir: Root splits directory.

    Returns:
        Path to the ``processed/`` subdirectory.

    Raises:
        SystemExit: If any required file is missing.
    """
    processed = splits_dir / "processed"
    if not processed.exists():
        die(f"Processed directory not found: {processed}")

    required_jsonl = [
        "facets.jsonl.zst",
        "docids.jsonl.zst",
        "training_pairs.jsonl.zst",
    ]
    for name in required_jsonl:
        if not resolve_jsonl(processed / name).exists():
            die(f"Required file not found: {processed / name}")

    index_path = processed / "docid_index.json"
    if not index_path.exists():
        die(f"DocID index not found: {index_path}\nRun 'docid build-index' first.")

    return processed


def _load_checkpoint(
    checkpoint: Path,
    *,
    quantize: str = "off",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a fine-tuned checkpoint, auto-detecting LoRA vs full FT.

    If the checkpoint directory contains ``adapter_config.json``, it's a LoRA
    adapter — the base model is resolved from the PEFT config and loaded with
    4-bit quantisation. Otherwise, the directory is assumed to be a full
    fine-tune and loaded unquantised in bfloat16, unless ``quantize`` asks
    for inference-time quantisation.

    Args:
        checkpoint: Path to the saved checkpoint directory.
        quantize: Inference-time quantisation for full-FT checkpoints:
            ``"off"`` (bfloat16, default), ``"4bit"`` (NF4 double-quant),
            or ``"8bit"`` (bitsandbytes int8).

    Returns:
        Tuple of (model in eval mode, tokeniser).
    """
    if quantize not in {"off", "4bit", "8bit"}:
        raise ValueError(f"Unknown quantize mode: {quantize!r}")
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from genret._shed.transformers import load_peft_model, load_tokenizer

    tokeniser = load_tokenizer(str(checkpoint))
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.padding_side = "left"

    is_lora = (checkpoint / "adapter_config.json").exists()
    if is_lora:
        from peft import PeftConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        peft_config = PeftConfig.from_pretrained(str(checkpoint))
        base_model_name = peft_config.base_model_name_or_path
        assert isinstance(base_model_name, str)
        logger.info("Base model: %s (LoRA adapter)", base_model_name)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        base_model.resize_token_embeddings(len(tokeniser), mean_resizing=False)
        model = load_peft_model(base_model, str(checkpoint))
    else:
        logger.info(
            "Loading full-FT checkpoint: %s (quantize=%s)", checkpoint, quantize
        )
        quant_config = None
        if quantize == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantize == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=quant_config,
        )

    model.eval()
    return model, tokeniser


__all__ = ["_load_checkpoint", "_validate_data_files"]
