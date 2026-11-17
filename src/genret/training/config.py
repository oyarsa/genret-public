"""Configuration dataclasses for the generative retrieval training pipeline.

Provides frozen, keyword-only dataclasses for all hyperparameters: LoRA adapter
configuration, 4-bit quantisation settings, training loop parameters, and
constrained beam search decoding options.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class LoRAConfig:
    """Low-Rank Adaptation (LoRA) hyperparameters.

    Controls the rank, scaling, dropout, and which linear layers receive
    adapters during QLoRA fine-tuning.
    """

    rank: int = 16
    """LoRA rank (dimension of the low-rank matrices)."""
    alpha: int = 32
    """LoRA scaling factor (effective scale = alpha / rank)."""
    dropout: float = 0.05
    """Dropout probability applied to LoRA layers."""
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj")
    """Names of the linear modules to attach LoRA adapters to."""


@dataclass(frozen=True, kw_only=True)
class QuantConfig:
    """BitsAndBytes 4-bit quantisation settings.

    Configures NF4 quantisation with double quantisation and bfloat16
    computation dtype for QLoRA training.
    """

    load_in_4bit: bool = True
    """Whether to load the base model in 4-bit precision."""
    bnb_4bit_quant_type: str = "nf4"
    """Quantisation type: 'nf4' (NormalFloat4) or 'fp4'."""
    bnb_4bit_use_double_quant: bool = True
    """Whether to apply nested quantisation to further reduce memory."""
    bnb_4bit_compute_dtype: str = "bfloat16"
    """Compute dtype for 4-bit operations (as a string for serialisation)."""


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    """Training loop hyperparameters.

    Controls batch sizes, learning rate schedule, sequence lengths, and the
    balance between indexing and retrieval examples.
    """

    splits_dir: Path
    """Path to the splits directory containing train/dev/test data."""
    batch_size: int = 4
    """Per-device training batch size."""
    gradient_accumulation_steps: int = 4
    """Number of gradient accumulation steps (effective batch = batch_size * this)."""
    max_seq_length: int = 4096
    """Maximum total sequence length (prompt + target DocID + EOS)."""
    learning_rate: float = 2e-4
    """Peak learning rate for the AdamW optimiser."""
    num_epochs: int = 3
    """Number of full passes over the training data."""
    warmup_ratio: float = 0.1
    """Fraction of total steps used for linear warmup."""
    indexing_ratio: float = 0.5
    """Proportion of training examples that are indexing (vs retrieval) examples."""
    seed: int = 42
    """Random seed for deterministic dataset shuffling."""


@dataclass(frozen=True, kw_only=True)
class DecodingConfig:
    """Constrained beam search decoding parameters.

    Controls the beam width, maximum generation length, and number of
    top results returned by the decoder.
    """

    num_beams: int = 10
    """Number of beams to maintain during beam search."""
    max_new_tokens: int = 256
    """Maximum number of tokens to generate per query."""
    top_k_results: int = 100
    """Number of top-scoring results to return from decoding."""


FACET_TOKEN_PREFIXES: dict[str, str] = {
    "problems": "P",
    "methods": "M",
    "results": "R",
    "contributions": "C",
}
"""One-letter token-namespace prefix per facet. Keeps the L0/L1 tokens
distinct across facets so the constrained decoder can mask cleanly."""


@dataclass(frozen=True, kw_only=True)
class CodebookConfig:
    """Per-facet hierarchical codebook parameters.

    Holds the per-facet ``k0``/``k1`` cluster sizes, the path to the
    codebook artifacts on disk, and the delimiter tokens the model
    generates around each ``(L0, L1)`` code. The codebook directory must
    follow the layout produced by ``genret training build-codebooks``:

    ``<dir>/{problems,methods,results,contributions}/paper_codes.jsonl``
    ``<dir>/{...}/level0_centers.npy``
    ``<dir>/{...}/level1_centers.npy``
    ``<dir>/code_to_papers.json``
    """

    codebook_dir: Path
    """Directory with the per-facet codebooks."""
    k0: int = 64
    """Number of L0 clusters per facet codebook."""
    k1: int = 256
    """Number of L1 sub-clusters per L0 cluster."""
    code_start_token: str = "<CODE_START>"  # noqa: S105 — model special token
    """Marks the beginning of a generated code sequence."""
    code_end_token: str = "<CODE_END>"  # noqa: S105 — model special token
    """Marks the end of a generated code sequence."""

    def special_tokens(self) -> list[str]:
        """List every codebook special token in tokeniser-add order.

        Delimiters first, then for each facet its L0 tokens followed by
        its L1 tokens. Order is deterministic and stable.
        """
        out: list[str] = [self.code_start_token, self.code_end_token]
        for prefix in FACET_TOKEN_PREFIXES.values():
            out.extend(f"<{prefix}_L0_{i}>" for i in range(self.k0))
            out.extend(f"<{prefix}_L1_{i}>" for i in range(self.k1))
        return out

    def l0_token(self, facet: str, value: int) -> str:
        """Special-token string for an L0 cluster id under a facet."""
        return f"<{FACET_TOKEN_PREFIXES[facet]}_L0_{value}>"

    def l1_token(self, facet: str, value: int) -> str:
        """Special-token string for an L1 cluster id under a facet."""
        return f"<{FACET_TOKEN_PREFIXES[facet]}_L1_{value}>"
