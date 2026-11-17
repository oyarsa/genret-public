# pyright: basic
"""Constrained decoding for the GLEN generative retriever.

The model generates a free-text DocID token by token, constrained at each
step to token IDs that are valid continuations of at least one trie path.
When the generated prefix exactly matches a stored DocID (a trie leaf), EOS
is also allowed.  Decoding runs four times per query (one per facet) and the
per-facet results are fused with RRF downstream.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genret._shed.torch import beam_transition_scores
from genret._shed.torch import generate as model_generate
from genret._shed.transformers import (
    apply_chat_template_tensor,
    get_eos_token_id,
)
from genret.training.codebook_data import FACET_NAMES
from genret.training.glen_docids import GlenIndex, load_glen_index
from genret.types import CorpusId

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class GlenDecodeResult:
    """One beam from a per-facet GLEN decode."""

    facet: str
    rank: int
    """Beam rank within the facet (0 = best)."""
    score: float
    """Beam score from HuggingFace generate (sequences_scores)."""
    docid_tokens: tuple[int, ...]
    """Token-ID sequence that was generated."""
    paper_ids: list[CorpusId]
    """Papers stored at this trie leaf."""
    logprob_sum: float | None = None
    """Un-normalised sum of token log-probabilities, or `None` if unavailable.

    `score` is length-normalised (HuggingFace divides by the generated length
    at the default `length_penalty` of 1.0). MINDER's scoring uses the raw sum
    -- its length-bias remark only holds without normalisation, and its
    rescoring path passes `length_penalty=0.0` -- so the raw sum is recorded
    alongside for fusion rules that need it.
    """


def make_glen_prefix_fn(
    facet: str,
    *,
    index: GlenIndex,
    eos_id: int,
    prompt_len: int,
    max_new_tokens: int,
) -> Callable[[int, torch.Tensor], list[int]]:
    """Build a HuggingFace ``prefix_allowed_tokens_fn`` for one GLEN facet.

    At each generation step the function:

    1. Extracts the generated tokens so far (after the prompt).
    2. Walks the trie to find valid next tokens.
    3. If the current prefix is a populated leaf, also allows EOS.
    4. Once the sequence is longer than any trie path, forces EOS.
    """

    def prefix_fn(_batch_id: int, beam_input_ids: torch.Tensor) -> list[int]:
        generated = tuple(beam_input_ids[prompt_len:].tolist())

        if len(generated) >= max_new_tokens:
            return [eos_id]

        next_toks = index.valid_next(facet, generated)
        at_leaf = index.is_leaf(facet, generated)

        if not next_toks:
            # End of all trie paths — force EOS.
            return [eos_id]

        allowed = list(next_toks)
        if at_leaf:
            # Allow stopping here (intermediate prefix that is also a leaf).
            allowed.append(eos_id)
        return allowed

    return prefix_fn


class _PaperDedupLogitsProcessor:
    """Suppress beams whose reachable papers are all already found.

    Tracks, per query, the set of papers any beam has *completed* a DocID
    for. When another beam's current prefix can only reach already-found
    papers (the reachability map shows full overlap with the found set),
    that beam's logits are forced to -inf so it loses the next top-K
    selection — freeing the beam slot for exploration elsewhere in the
    trie.

    Implemented as a HuggingFace ``LogitsProcessor`` interface (call with
    ``(input_ids, scores)``) so it composes with ``prefix_allowed_tokens_fn``
    and the standard HF beam-search loop.
    """

    def __init__(
        self,
        *,
        facet: str,
        index: GlenIndex,
        prompt_len: int,
        eos_id: int,
    ) -> None:
        self.facet = facet
        self.code_to_papers = index.code_to_papers.get(facet, {})
        self.prompt_len = prompt_len
        self.eos_id = eos_id
        # Within a single decode call HF reuses the same processor across
        # generation steps; the found-set accumulates over those steps.
        self.found: set[CorpusId] = set()

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        n_beams = input_ids.shape[0]
        # Only suppress beams that have completed a DocID whose papers are
        # already in the found-set. Mid-decode beams are never suppressed —
        # their prefix may still navigate to new papers. The original
        # design checked prefix-reachability (subtree-under-prefix) which
        # killed too many exploratory beams.
        suppress: list[int] = []
        for b in range(n_beams):
            gen = tuple(int(t) for t in input_ids[b, self.prompt_len :].tolist())
            # Strip trailing EOS so leaf lookup works on the bare DocID.
            while gen and gen[-1] == self.eos_id:
                gen = gen[:-1]
            papers = self.code_to_papers.get(gen)
            if not papers:
                # Not at a leaf yet — keep exploring.
                continue
            paper_set = set(papers)
            if paper_set.issubset(self.found):
                # All papers at this leaf are already found via another
                # beam; suppressing this beam frees its slot.
                suppress.append(b)
            else:
                # New papers; admit them so other beams can skip this leaf.
                self.found.update(paper_set)
        # Never suppress every beam: HF beam search hangs / silently emits
        # nothing when all candidates have -inf score. Keep at least one
        # beam alive so generation can finish.
        if len(suppress) >= n_beams:
            suppress = suppress[: n_beams - 1]
        for b in suppress:
            scores[b] = float("-inf")
        return scores


def decode_per_facet_glen(
    *,
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    facet: str,
    index: GlenIndex,
    num_beams: int = 50,
    max_new_tokens: int = 32,
    num_beam_groups: int = 1,
    diversity_penalty: float = 0.0,
    paper_dedup: bool = False,
    extra_processor_factory: Callable[[int, int], object] | None = None,
) -> list[GlenDecodeResult]:
    """Run constrained beam search for one GLEN facet on one query.

    ``num_beam_groups``+``diversity_penalty`` enable HuggingFace Diverse
    Beam Search (Vijayakumar et al., 2016). ``paper_dedup`` enables
    leaf-paper deduplication during beam search via a custom
    LogitsProcessor that kills beams whose reachable subtree contains
    only already-found papers. ``extra_processor_factory`` optionally
    builds one more LogitsProcessor (e.g. graph-PRF biasing) given
    ``(prompt_len, eos_id)``; composed after the built-in ones.
    """
    eos_id = get_eos_token_id(tokeniser)
    prompt_ids = apply_chat_template_tensor(tokeniser, messages)
    device = next(model.parameters()).device
    input_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    prefix_fn = make_glen_prefix_fn(
        facet,
        index=index,
        eos_id=eos_id,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
    )

    processors: list[object] = []
    if paper_dedup:
        processors.append(
            _PaperDedupLogitsProcessor(
                facet=facet,
                index=index,
                prompt_len=prompt_len,
                eos_id=eos_id,
            )
        )
    if extra_processor_factory is not None:
        processors.append(extra_processor_factory(prompt_len, eos_id))

    generate_kwargs: dict[str, object] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "num_return_sequences": num_beams,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": eos_id,
    }
    generate_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
    if num_beam_groups > 1:
        generate_kwargs["num_beam_groups"] = num_beam_groups
        generate_kwargs["diversity_penalty"] = diversity_penalty
        # Group Beam Search moved to the transformers-community plugin in
        # recent HF versions; loading it requires trust_remote_code=True.
        # NOTE: this path is known-broken (see L5 in
        # docs/specs/20260611-beam-search-limitations.md) — investigation
        # suspended 2026-06-12.
        generate_kwargs["trust_remote_code"] = True

    if processors:
        from transformers import LogitsProcessorList  # noqa: PLC0415

        generate_kwargs["logits_processor"] = LogitsProcessorList(processors)

    out = model_generate(model, **generate_kwargs)

    sequences = out.sequences  # type: ignore[attr-defined]
    seq_scores = getattr(out, "sequences_scores", None)
    transition_scores = beam_transition_scores(model, out)

    results: list[GlenDecodeResult] = []
    for rank in range(sequences.shape[0]):
        gen = sequences[rank, prompt_len:]
        # Strip trailing EOS/pad.
        toks = tuple(int(t) for t in gen.tolist() if int(t) != eos_id)
        if not toks:
            continue
        papers = index.papers_at(facet, toks)
        if not papers:
            # Beam landed on a non-leaf — no papers to return.
            continue
        score = float(seq_scores[rank]) if seq_scores is not None else 0.0
        logprob_sum: float | None = None
        if transition_scores is not None:
            row = transition_scores[rank]
            logprob_sum = float(row[torch.isfinite(row)].sum())
        results.append(
            GlenDecodeResult(
                facet=facet,
                rank=rank,
                score=score,
                docid_tokens=toks,
                paper_ids=list(papers),
                logprob_sum=logprob_sum,
            )
        )
    return results


@dataclass(frozen=True, kw_only=True)
class UnconstrainedDecodeResult:
    """One beam from an unconstrained (off-trie) decode.

    The decoded token sequence is not constrained to any DocID and may not
    map to a real paper without a snap step.
    """

    facet: str
    rank: int
    score: float
    docid_tokens: tuple[int, ...]


def decode_unconstrained(
    *,
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    facet: str,
    num_beams: int = 50,
    max_new_tokens: int = 32,
) -> list[UnconstrainedDecodeResult]:
    """Run unconstrained beam search; no trie constraint applied.

    Used by the snap-to-nearest evaluation path. The caller must snap each
    decoded token sequence to a real DocID before scoring.
    """
    eos_id = get_eos_token_id(tokeniser)
    prompt_ids = apply_chat_template_tensor(tokeniser, messages)
    device = next(model.parameters()).device
    input_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    out = model_generate(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=eos_id,
    )

    sequences = out.sequences  # type: ignore[attr-defined]
    seq_scores = getattr(out, "sequences_scores", None)

    results: list[UnconstrainedDecodeResult] = []
    for rank in range(sequences.shape[0]):
        gen = sequences[rank, prompt_len:]
        toks = tuple(int(t) for t in gen.tolist() if int(t) != eos_id)
        if not toks:
            continue
        score = float(seq_scores[rank]) if seq_scores is not None else 0.0
        results.append(
            UnconstrainedDecodeResult(
                facet=facet, rank=rank, score=score, docid_tokens=toks
            )
        )
    return results


def all_facets() -> Iterable[str]:
    """Yield the four facet names in deterministic order."""
    yield from FACET_NAMES


__all__ = [
    "GlenDecodeResult",
    "GlenIndex",
    "UnconstrainedDecodeResult",
    "all_facets",
    "decode_per_facet_glen",
    "decode_unconstrained",
    "load_glen_index",
    "make_glen_prefix_fn",
]
