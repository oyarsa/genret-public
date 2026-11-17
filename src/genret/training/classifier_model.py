# pyright: basic
# ruff: noqa: PLC0415  Lazy-import torch / transformers / peft for module load speed.
"""Causal-LM-encoder + zero-init linear head for multi-label classification.

Used by the paper-recommendation sanity check: ``train-classifier`` and
``evaluate-classifier`` in ``cli_classifier``. The encoder is one of the
existing Llama / Qwen3 instruct models; the head maps mean-pooled token
representations to per-paper logits, trained with sigmoid + BCE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

PoolingMode = Literal["mean", "last"]
LossKind = Literal["bce", "focal"]


class FocalLoss(nn.Module):
    """Multi-label focal loss with per-class alpha balancing.

    Reduces the gradient contribution from easy examples (high ``p_t``) so
    the model can't trivially win by always predicting the popular classes.
    """

    def __init__(self, *, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    @override
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal BCE between sigmoid(logits) and the multi-hot labels."""
        ce = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * labels + (1.0 - p) * (1.0 - labels)
        loss = ce * (1.0 - p_t).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1.0 - self.alpha) * (1.0 - labels)
            loss = alpha_t * loss
        return loss.mean()


logger = logging.getLogger(__name__)


def _mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask-aware mean over the sequence dimension.

    ``last_hidden_state``: ``(B, T, H)``. ``attention_mask``: ``(B, T)``.
    """
    mask = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def _last_token_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Hidden state at the last non-pad position.

    With left padding (``padding_side="left"``), the final position in the
    sequence is always a real token, so this is just ``[:, -1, :]``. The
    attention mask is accepted for interface parity with :func:`_mean_pool`
    and as a guard if padding ever switches sides.
    """
    if attention_mask is not None and (attention_mask[:, -1] == 0).any():
        # Right padding: pick the last position where mask == 1 per row.
        last_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(last_hidden_state.size(0), device=last_idx.device)
        return last_hidden_state[batch_idx, last_idx]
    return last_hidden_state[:, -1, :]


class PaperClassifier(nn.Module):
    """Wrap a HF causal-LM encoder with a zero-init linear head.

    Forward signature matches HF Trainer expectations: ``input_ids`` /
    ``attention_mask`` / ``labels`` go in; ``loss`` and ``logits`` come out.
    """

    def __init__(
        self,
        encoder: PreTrainedModel,
        *,
        hidden_size: int,
        num_classes: int,
        pooling: PoolingMode = "mean",
        pos_weight: torch.Tensor | None = None,
        head_bias_init: torch.Tensor | None = None,
        loss_kind: LossKind = "bce",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.zeros_(self.head.weight)
        if head_bias_init is not None:
            # Start at the prior so the trivial bias-only solution is the
            # *origin* of optimisation, not its attractor.
            assert head_bias_init.shape == (num_classes,), (
                f"bias init shape {head_bias_init.shape} != ({num_classes},)"
            )
            with torch.no_grad():
                self.head.bias.copy_(head_bias_init)
        else:
            nn.init.zeros_(self.head.bias)
        loss_fn: nn.Module
        if loss_kind == "focal":
            loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        elif pos_weight is not None:
            assert pos_weight.shape == (num_classes,), (
                f"pos_weight shape {pos_weight.shape} != ({num_classes},)"
            )
            # BCEWithLogitsLoss stores pos_weight as an internal buffer, so it
            # follows .to(device) along with the rest of the module.
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.float())
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = loss_fn
        self.pooling: PoolingMode = pooling

    @override
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode, mean-pool, project to per-paper logits, compute BCE if labels given."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Use the encoder's last hidden state and pool per the configured mode.
        last_hidden = outputs.hidden_states[-1]  # (B, T, H)
        if self.pooling == "last":
            pooled = _last_token_pool(last_hidden, attention_mask)
        else:
            pooled = _mean_pool(last_hidden, attention_mask)
        logits = self.head(pooled.to(self.head.weight.dtype))
        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits.float(), labels.float())
        return result


@dataclass(frozen=True, kw_only=True)
class ClassifierSetup:
    """Plain bundle returned by :func:`setup_classifier_model_and_tokeniser`."""

    model: PaperClassifier
    tokeniser: PreTrainedTokenizerBase
    hidden_size: int


def setup_classifier_model_and_tokeniser(
    model_name: str,
    *,
    num_classes: int,
    use_qlora: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    pooling: PoolingMode = "mean",
    pos_weight: torch.Tensor | None = None,
    head_bias_init: torch.Tensor | None = None,
    loss_kind: LossKind = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> ClassifierSetup:
    """Load encoder + tokeniser and wrap with the classifier head.

    When ``use_qlora`` is True, the encoder is loaded in 4-bit NF4 and LoRA
    adapters are attached. The classification head is kept in bfloat16 and
    fully trainable regardless of the encoder regime.
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModel, BitsAndBytesConfig

    from genret._shed.transformers import as_pretrained_model, load_tokenizer

    tokeniser = load_tokenizer(model_name)
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token
    tokeniser.padding_side = "left"

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        encoder_raw = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        encoder_raw = prepare_model_for_kbit_training(
            encoder_raw, use_gradient_checkpointing=True
        )
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        encoder_raw = get_peft_model(encoder_raw, peft_config)
    else:
        encoder_raw = AutoModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        encoder_raw.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    encoder = as_pretrained_model(encoder_raw)
    hidden_size = int(encoder.config.hidden_size)
    classifier = PaperClassifier(
        encoder,
        hidden_size=hidden_size,
        num_classes=num_classes,
        pooling=pooling,
        pos_weight=pos_weight,
        head_bias_init=head_bias_init,
        loss_kind=loss_kind,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
    # Move the head to the same device as the encoder. ``device_map="auto"``
    # places the encoder on GPU; without this, the head stays on CPU and the
    # forward pass fails at eval time with a device-mismatch error. Training
    # works because HF Trainer moves the whole model to the args device.
    encoder_device = next(encoder.parameters()).device
    classifier.head.to(device=encoder_device, dtype=torch.bfloat16)
    return ClassifierSetup(
        model=classifier, tokeniser=tokeniser, hidden_size=hidden_size
    )


__all__ = [
    "ClassifierSetup",
    "FocalLoss",
    "LossKind",
    "PaperClassifier",
    "PoolingMode",
    "setup_classifier_model_and_tokeniser",
]
