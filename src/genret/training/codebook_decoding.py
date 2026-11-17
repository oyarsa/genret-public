# pyright: basic
"""Constrained-decoding for the per-facet codebook generative retriever.

For each facet the decoder produces a 4-token sequence:

    <CODE_START> <X_L0_a> <X_L1_b> <CODE_END>

Where the L0 token comes from the active facet's namespace and the L1
token must correspond to a sub-cluster that exists under the generated
L0 (i.e. a populated leaf in ``code_to_papers``). The decoder runs four
times per query (one per facet), each call constrained to that facet's
token namespace and the codebook's tree structure.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from genret._shed.torch import generate as model_generate
from genret._shed.transformers import (
    apply_chat_template_tensor,
    convert_token_to_id,
    get_eos_token_id,
)
from genret.training.codebook_data import FACET_NAMES
from genret.training.config import FACET_TOKEN_PREFIXES, CodebookConfig
from genret.types import CorpusId

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class FacetDecodeResult:
    """One beam from a per-facet decode."""

    facet: str
    rank: int
    """Beam rank within the facet (0 = best)."""
    score: float
    """Beam score from HuggingFace generate (sequences_scores)."""
    l0: int
    l1: int
    paper_ids: list[CorpusId]
    """Papers living at the generated leaf, in deterministic order."""


@dataclass(frozen=True, kw_only=True)
class CodebookIndex:
    """Per-facet code/leaf membership used to build the decode masks."""

    code_to_papers: dict[str, dict[tuple[int, int], list[CorpusId]]]
    """``facet -> {(L0, L1) -> [corpusids]}``."""
    populated_l1_by_l0: dict[str, dict[int, list[int]]]
    """``facet -> {L0 -> [L1, ...]}`` over leaves that contain papers."""


def load_codebook_index(codebook_dir: Path) -> CodebookIndex:
    """Load the ``code_to_papers.json`` index and pre-compute decode masks."""
    raw = json.loads((codebook_dir / "code_to_papers.json").read_text())
    code_to_papers: dict[str, dict[tuple[int, int], list[CorpusId]]] = {}
    populated: dict[str, dict[int, list[int]]] = {}

    for facet in FACET_NAMES:
        facet_map: dict[tuple[int, int], list[CorpusId]] = {}
        per_l0: dict[int, list[int]] = {}
        for key, papers in raw.get(facet, {}).items():
            l0_str, l1_str = key.split(",")
            l0, l1 = int(l0_str), int(l1_str)
            cids = [CorpusId(int(c)) for c in papers]
            facet_map[(l0, l1)] = cids
            if cids:
                per_l0.setdefault(l0, []).append(l1)
        # Sort populated L1 lists for determinism.
        for l0 in per_l0:
            per_l0[l0].sort()
        code_to_papers[facet] = facet_map
        populated[facet] = per_l0

    return CodebookIndex(code_to_papers=code_to_papers, populated_l1_by_l0=populated)


@dataclass(frozen=True, kw_only=True)
class CodebookTokenMap:
    """Special-token id lookups derived from a CodebookConfig + tokeniser."""

    code_start_id: int
    code_end_id: int
    eos_id: int
    l0_ids: dict[str, dict[int, int]]
    """``facet -> {l0 value -> token id}``."""
    l1_ids: dict[str, dict[int, int]]
    """``facet -> {l1 value -> token id}``."""
    l0_id_to_value: dict[int, tuple[str, int]]
    """``token id -> (facet, l0 value)`` (for reading back beam outputs)."""
    l1_id_to_value: dict[int, tuple[str, int]]
    """``token id -> (facet, l1 value)``."""


def build_token_map(
    cfg: CodebookConfig, tokeniser: PreTrainedTokenizerBase
) -> CodebookTokenMap:
    """Resolve all codebook special tokens to vocab ids."""

    def _resolve(tok: str) -> int:
        tid = convert_token_to_id(tokeniser, tok)
        if tid == tokeniser.unk_token_id:
            raise ValueError(
                f"Token {tok!r} not registered in tokeniser; was the model "
                "loaded via setup_model_and_tokeniser with the codebook "
                "special_tokens?"
            )
        return tid

    code_start_id = _resolve(cfg.code_start_token)
    code_end_id = _resolve(cfg.code_end_token)
    eos_id = get_eos_token_id(tokeniser)

    l0_ids: dict[str, dict[int, int]] = {}
    l1_ids: dict[str, dict[int, int]] = {}
    l0_id_to_value: dict[int, tuple[str, int]] = {}
    l1_id_to_value: dict[int, tuple[str, int]] = {}
    for facet in FACET_NAMES:
        l0_ids[facet] = {}
        l1_ids[facet] = {}
        for value in range(cfg.k0):
            tid = _resolve(cfg.l0_token(facet, value))
            l0_ids[facet][value] = tid
            l0_id_to_value[tid] = (facet, value)
        for value in range(cfg.k1):
            tid = _resolve(cfg.l1_token(facet, value))
            l1_ids[facet][value] = tid
            l1_id_to_value[tid] = (facet, value)
    return CodebookTokenMap(
        code_start_id=code_start_id,
        code_end_id=code_end_id,
        eos_id=eos_id,
        l0_ids=l0_ids,
        l1_ids=l1_ids,
        l0_id_to_value=l0_id_to_value,
        l1_id_to_value=l1_id_to_value,
    )


def make_facet_prefix_fn(
    facet: str,
    *,
    index: CodebookIndex,
    tokens: CodebookTokenMap,
    prompt_len: int,
) -> Callable[[int, torch.Tensor], list[int]]:
    """Build a HuggingFace ``prefix_allowed_tokens_fn`` for one facet.

    Decoding proceeds through four positions (relative to ``prompt_len``):

    0. ``<CODE_START>``
    1. ``<facet_L0_x>`` for any L0 with at least one populated leaf
    2. ``<facet_L1_y>`` for L1 values populated under the chosen L0
    3. ``<CODE_END>``
    4+. ``<eos>`` (or anything; the model should have stopped by now)
    """
    populated_l0 = sorted(index.populated_l1_by_l0[facet].keys())
    populated_l1_by_l0 = index.populated_l1_by_l0[facet]
    l0_ids = tokens.l0_ids[facet]
    l1_ids = tokens.l1_ids[facet]

    # Pre-compute allowed L0 ids: only L0 values with at least one populated
    # L1. Empty L0 buckets are unreachable.
    allowed_l0_ids = [l0_ids[v] for v in populated_l0]

    code_start = tokens.code_start_id
    code_end = tokens.code_end_id
    eos = tokens.eos_id

    def prefix_fn(_batch_id: int, beam_input_ids: torch.Tensor) -> list[int]:
        gen = beam_input_ids[prompt_len:].tolist()
        step = len(gen)
        if step == 0:
            return [code_start]
        if step == 1:
            return allowed_l0_ids
        if step == 2:
            l0_token_id = int(gen[1])
            mapped = tokens.l0_id_to_value.get(l0_token_id)
            if mapped is None or mapped[0] != facet:
                return [code_end]
            l0_value = mapped[1]
            return [l1_ids[v] for v in populated_l1_by_l0.get(l0_value, ())]
        if step == 3:
            return [code_end]
        return [eos]

    return prefix_fn


def decode_per_facet(
    *,
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    facet: str,
    index: CodebookIndex,
    tokens: CodebookTokenMap,
    num_beams: int = 30,
) -> list[FacetDecodeResult]:
    """Run constrained beam search for one facet on one query."""
    prompt_ids = apply_chat_template_tensor(tokeniser, messages)
    device = next(model.parameters()).device
    input_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    prefix_fn = make_facet_prefix_fn(
        facet, index=index, tokens=tokens, prompt_len=prompt_len
    )

    out = model_generate(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=4,  # CODE_START + L0 + L1 + CODE_END
        num_beams=num_beams,
        num_return_sequences=num_beams,
        do_sample=False,
        prefix_allowed_tokens_fn=prefix_fn,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokens.eos_id,
    )

    sequences = out.sequences  # type: ignore[attr-defined]
    seq_scores = getattr(out, "sequences_scores", None)

    results: list[FacetDecodeResult] = []
    for rank in range(sequences.shape[0]):
        gen = sequences[rank, prompt_len:].tolist()
        # Expect [code_start, l0, l1, code_end] (or shorter on error).
        if len(gen) < 4:
            continue
        l0_token, l1_token = int(gen[1]), int(gen[2])
        l0_mapped = tokens.l0_id_to_value.get(l0_token)
        l1_mapped = tokens.l1_id_to_value.get(l1_token)
        if (
            l0_mapped is None
            or l1_mapped is None
            or l0_mapped[0] != facet
            or l1_mapped[0] != facet
        ):
            continue
        l0 = l0_mapped[1]
        l1 = l1_mapped[1]
        papers = index.code_to_papers[facet].get((l0, l1), [])
        score = float(seq_scores[rank]) if seq_scores is not None else 0.0
        results.append(
            FacetDecodeResult(
                facet=facet,
                rank=rank,
                score=score,
                l0=l0,
                l1=l1,
                paper_ids=list(papers),
            )
        )
    return results


def all_facets() -> Iterable[str]:
    """Yield the four same-facet types in deterministic order."""
    yield from FACET_NAMES


__all__ = [
    "FACET_TOKEN_PREFIXES",
    "CodebookIndex",
    "CodebookTokenMap",
    "FacetDecodeResult",
    "all_facets",
    "build_token_map",
    "decode_per_facet",
    "load_codebook_index",
    "make_facet_prefix_fn",
]
