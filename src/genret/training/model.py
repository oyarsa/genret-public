# pyright: basic
"""Model loading, special token registration, and LoRA adapter setup.

Handles BitsAndBytes 4-bit quantisation, vocabulary extension for DocID structural
tokens, and PEFT LoRA application for QLoRA fine-tuning.
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import replace

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from genret._shed.transformers import as_pretrained_model, load_tokenizer
from genret.docid import SPECIAL_TOKENS
from genret.training.config import LoRAConfig, QuantConfig

logger = logging.getLogger(__name__)


def _distributed_training_active() -> bool:
    """Return whether this process is part of a multi-process training job."""
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except ValueError:
        return False


def _make_bnb_config(quant: QuantConfig) -> BitsAndBytesConfig:
    """Build a BitsAndBytesConfig from our dataclass.

    Args:
        quant: Quantisation settings.

    Returns:
        HuggingFace BitsAndBytesConfig for 4-bit NF4 loading.
    """
    compute_dtype = getattr(torch, quant.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=quant.load_in_4bit,
        bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def resolve_lora_targets(model: nn.Module, patterns: Sequence[str]) -> Sequence[str]:
    """Resolve LoRA target patterns against architectures PEFT can't fully wrap.

    Multimodal models can reuse a projection name across towers with
    different module classes — e.g. Gemma 4's ``q_proj`` is a plain
    ``nn.Linear`` in the language tower but a ``Gemma4ClippableLinear``
    wrapper in the vision/audio towers, and PEFT errors on the wrapper.
    Per pattern:

    - all matches plain ``nn.Linear`` → keep the pattern unchanged;
    - mixed → expand to the full module names of the plain-Linear matches
      only (adapts the language tower, skips the unsupported towers, which
      are inert for text-only training anyway);
    - all matches wrappers with a ``.linear`` child → retarget the pattern
      to ``<pattern>.linear``.
    """
    resolved: list[str] = []
    for pattern in patterns:
        matches = [
            (name, module)
            for name, module in model.named_modules()
            if name.endswith(f".{pattern}") or name == pattern
        ]
        if not matches or all(isinstance(m, nn.Linear) for _, m in matches):
            resolved.append(pattern)
            continue
        plain = [name for name, m in matches if isinstance(m, nn.Linear)]
        if plain:
            resolved.extend(plain)
            continue
        if all(isinstance(getattr(m, "linear", None), nn.Linear) for _, m in matches):
            resolved.append(f"{pattern}.linear")
            continue
        resolved.append(pattern)  # let PEFT raise its descriptive error
    return resolved


def _make_lora_config(lora: LoRAConfig) -> LoraConfig:
    """Build a PEFT LoraConfig from our dataclass.

    Args:
        lora: LoRA hyperparameters.

    Returns:
        PEFT LoraConfig for causal LM fine-tuning.
    """
    return LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def setup_model_and_tokeniser(
    model_name: str,
    lora: LoRAConfig,
    quant: QuantConfig,
    *,
    extra_special_tokens: Sequence[str] = (),
    full_finetune: bool = False,
    distributed_training: bool | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a model with extended vocabulary and either LoRA adapters or full fine-tune.

    Steps:
        1. Load tokeniser, set ``pad_token`` = ``eos_token``.
        2. Add DocID special tokens + any extra tokens (e.g. codebook
           tokens) if not already in vocabulary.
        3. Load base model. With ``full_finetune=False`` (default), use
           4-bit NF4 quantisation; otherwise load unquantised bfloat16.
        4. Resize embeddings if the vocabulary was extended.
        5. With ``full_finetune=False``, prepare for k-bit training and
           apply LoRA adapters. Otherwise, return the full model with
           gradient checkpointing enabled.

    Args:
        model_name: HuggingFace model identifier (e.g. ``meta-llama/Llama-3.1-8B``).
        lora: LoRA adapter configuration (ignored when ``full_finetune=True``).
        quant: Quantisation configuration (ignored when ``full_finetune=True``).
        extra_special_tokens: Additional special tokens to register
            alongside the built-in DocID tokens. Used by the codebook
            generative-retrieval scheme to add per-facet L0/L1 tokens
            and the ``<CODE_START>`` / ``<CODE_END>`` delimiters.
        full_finetune: If True, skip quantisation and LoRA — return the
            unwrapped model in bfloat16 with all weights trainable.
        distributed_training: Whether the model will be wrapped by a
            distributed trainer such as FSDP/DDP. When ``None``, infer from
            ``WORLD_SIZE``. Distributed full fine-tuning loads the model
            without ``device_map="auto"`` so the trainer owns placement and
            sharding.

    Returns:
        Tuple of (model with LoRA adapters or full FT, tokeniser with special tokens).
    """
    # 1. Tokeniser
    tokeniser = load_tokenizer(model_name)
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.padding_side = "left"

    # 2. Add special tokens
    existing = set(tokeniser.get_vocab())
    desired = list(SPECIAL_TOKENS) + [t for t in extra_special_tokens if t]
    # Preserve order while deduplicating (some callers may pass tokens that
    # overlap with SPECIAL_TOKENS).
    seen: set[str] = set()
    new_tokens: list[str] = []
    for t in desired:
        if t in existing or t in seen:
            continue
        seen.add(t)
        new_tokens.append(t)
    num_added = tokeniser.add_special_tokens({"additional_special_tokens": new_tokens})
    logger.info("Added %d special tokens to tokeniser", num_added)

    # 3. Load model — quantised for QLoRA path, unquantised bfloat16 for full FT.
    use_distributed_loading = (
        _distributed_training_active()
        if distributed_training is None
        else distributed_training
    )
    if full_finetune:
        if use_distributed_loading:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype=torch.bfloat16,
            )
    else:
        bnb_config = _make_bnb_config(quant)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
        )

    # 4. Resize embeddings if tokens were added
    if num_added > 0:
        model.resize_token_embeddings(len(tokeniser), mean_resizing=False)
        logger.info("Resized embeddings to %d", len(tokeniser))

    # 5. Either prepare for QLoRA + apply adapters, or enable gradient checkpointing
    #    on the full model and return as-is.
    if full_finetune:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Full fine-tune: trainable parameters: %s / %s (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / total if total > 0 else 0.0,
        )
        return model, tokeniser

    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    resolved_targets = resolve_lora_targets(model, lora.target_modules)
    if list(resolved_targets) != list(lora.target_modules):
        logger.info("LoRA targets retargeted to inner linears: %s", resolved_targets)
        lora = replace(lora, target_modules=resolved_targets)
    peft_config = _make_lora_config(lora)
    model_peft = get_peft_model(model, peft_config)

    trainable, total = model_peft.get_nb_trainable_parameters()
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / total if total > 0 else 0.0,
    )

    return as_pretrained_model(model_peft), tokeniser
