"""Decode-time plumbing for the MINDER-style three-view baseline (Variant A).

Title and pseudo-query are both enumerable identifier sets, so they reuse the
existing `GlenIndex` trie machinery unchanged -- `GlenIndex` is already
generic over an arbitrary facet-like string key (`glen_title_index.py` does
exactly this for the title view). Only the substring view needs new decode
plumbing, since its identifier set is every corpus span and is backed by a
`SubstringIndex` (suffix array) rather than a trie.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import torch

from genret._shed.torch import beam_transition_scores
from genret._shed.torch import generate as model_generate
from genret._shed.transformers import apply_chat_template_tensor, get_eos_token_id
from genret.training.glen_decoding import GlenDecodeResult
from genret.training.glen_docids import GlenIndex, _build_trie, extract_docid_tokens
from genret.training.glen_title_index import normalise_title
from genret.training.minder_substrings import SubstringIndex
from genret.types import CorpusId

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

TITLE_FACET = "title"
"""Facet-key tag for the title view in a `GlenIndex` and in dump rows."""
PSEUDO_QUERY_FACET = "pseudo-query"
"""Facet-key tag for the pooled-bullet view in a `GlenIndex` and in dump rows."""
SUBSTRING_FACET = "substring"
"""Facet-key tag for the substring view in dump rows."""


def build_title_and_pseudo_query_index(
    titles: Mapping[CorpusId, str],
    bullets_by_paper: Mapping[CorpusId, list[str]],
    tokeniser: PreTrainedTokenizerBase,
    *,
    max_tokens: int = 32,
) -> GlenIndex:
    """Build a two-facet `GlenIndex` covering the title and pseudo-query views.

    Both are enumerable identifier sets (one title per paper; a paper's own
    facet bullets, pooled, as its pseudo-queries), so they reuse the existing
    trie decode path unchanged rather than needing a new one.

    Args:
        titles: Paper titles, keyed by corpus id.
        bullets_by_paper: A paper's pooled facet bullets (the pseudo-query
            identifier set), keyed by corpus id.
        tokeniser: Tokeniser to encode identifier text with.
        max_tokens: Token cap per identifier.

    Returns:
        A `GlenIndex` with `"title"` and `"pseudo-query"` facet keys.
    """
    code_to_papers: dict[str, dict[tuple[int, ...], list[CorpusId]]] = {
        TITLE_FACET: {},
        PSEUDO_QUERY_FACET: {},
    }
    for corpus_id, title in titles.items():
        if not title.strip():
            continue
        toks = extract_docid_tokens(
            normalise_title(title), tokeniser, max_tokens=max_tokens
        )
        if toks:
            code_to_papers[TITLE_FACET].setdefault(tuple(toks), []).append(corpus_id)

    for corpus_id, bullets in bullets_by_paper.items():
        for bullet in bullets:
            toks = extract_docid_tokens(bullet, tokeniser, max_tokens=max_tokens)
            if toks:
                code_to_papers[PSEUDO_QUERY_FACET].setdefault(tuple(toks), []).append(
                    corpus_id
                )

    trie = {
        facet: _build_trie(list(seqmap.keys()))
        for facet, seqmap in code_to_papers.items()
    }
    logger.info(
        "MINDER title+pseudo-query index: %d title identifiers, %d pseudo-query "
        "identifiers",
        len(code_to_papers[TITLE_FACET]),
        len(code_to_papers[PSEUDO_QUERY_FACET]),
    )
    return GlenIndex(code_to_papers=code_to_papers, trie=trie)


def make_substring_prefix_fn(
    *,
    index: SubstringIndex,
    eos_id: int,
    prompt_len: int,
    max_new_tokens: int,
) -> Callable[[int, torch.Tensor], list[int]]:
    """Build a `prefix_allowed_tokens_fn` constraining generation to corpus spans.

    Mirrors `glen_decoding.make_glen_prefix_fn`'s shape, substituting the
    suffix array's `valid_next`/`is_stoppable` for the trie's `valid_next`/
    `is_leaf` -- the same constrained-decoding contract, backed by a
    different structure (see `minder_substrings.py`'s module docstring).
    """

    def prefix_fn(_batch_id: int, beam_input_ids: torch.Tensor) -> list[int]:
        generated = tuple(beam_input_ids[prompt_len:].tolist())

        if len(generated) >= max_new_tokens:
            return [eos_id]

        next_toks = index.valid_next(generated)
        if not next_toks:
            return [eos_id]

        allowed = list(next_toks)
        if index.is_stoppable(generated):
            allowed.append(eos_id)
        return allowed

    return prefix_fn


def decode_substring_view(
    *,
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    index: SubstringIndex,
    num_beams: int = 50,
    max_new_tokens: int = 12,
) -> list[GlenDecodeResult]:
    """Run constrained beam search for the substring view on one query.

    Mirrors `glen_decoding.decode_per_facet_glen`; the only structural
    difference is the prefix function's backing index.

    Args:
        model: Fine-tuned model.
        tokeniser: Matching tokeniser.
        messages: Chat messages for this query.
        index: Substring index over the corpus.
        num_beams: Beam width.
        max_new_tokens: Token cap, matching `index.max_span_tokens`.

    Returns:
        One `GlenDecodeResult` per beam that lands on a real span, tagged
        with `facet=SUBSTRING_FACET`.
    """
    eos_id = get_eos_token_id(tokeniser)
    prompt_ids = apply_chat_template_tensor(tokeniser, messages)
    device = next(model.parameters()).device
    input_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    prefix_fn = make_substring_prefix_fn(
        index=index,
        eos_id=eos_id,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
    )

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
        prefix_allowed_tokens_fn=prefix_fn,
    )

    sequences = out.sequences  # type: ignore[attr-defined]
    seq_scores = getattr(out, "sequences_scores", None)
    transition_scores = beam_transition_scores(model, out)

    results: list[GlenDecodeResult] = []
    for rank in range(sequences.shape[0]):
        gen = sequences[rank, prompt_len:]
        toks = tuple(int(t) for t in gen.tolist() if int(t) != eos_id)
        if not toks:
            continue
        papers = index.papers_at(toks)
        if not papers:
            continue
        score = float(seq_scores[rank]) if seq_scores is not None else 0.0
        logprob_sum: float | None = None
        if transition_scores is not None:
            row = transition_scores[rank]
            logprob_sum = float(row[torch.isfinite(row)].sum())
        results.append(
            GlenDecodeResult(
                facet=SUBSTRING_FACET,
                rank=rank,
                score=score,
                docid_tokens=toks,
                paper_ids=list(papers),
                logprob_sum=logprob_sum,
            )
        )
    return results


__all__ = [
    "PSEUDO_QUERY_FACET",
    "SUBSTRING_FACET",
    "TITLE_FACET",
    "build_title_and_pseudo_query_index",
    "decode_substring_view",
    "make_substring_prefix_fn",
]
