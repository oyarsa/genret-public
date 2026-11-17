# pyright: basic
"""Constrained beam search decoding for generative retrieval.

Uses HuggingFace `model.generate()` with a prefix trie and
`prefix_allowed_tokens_fn` to restrict generation to valid DocID prefixes.
KV cache is handled automatically by the generate() infrastructure.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from genret._shed.torch import generate as model_generate
from genret._shed.transformers import (
    apply_chat_template_tensor,
    get_eos_token_id,
    get_pad_token_id,
)
from genret.paper_parsing import DocIDIndex
from genret.types import CorpusId

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class DecodedResult:
    """A single result from constrained beam search.

    Attributes:
        paper_ids: Corpus IDs of all papers whose DocID is consistent with the generated
            token sequence.
        score: Log-probability score of this beam.
        generated_tokens: The token IDs generated (excluding the prompt).
    """

    paper_ids: list[CorpusId]
    score: float
    generated_tokens: list[int]


@dataclass
class _TrieNode:
    """A node in the DocID prefix trie.

    Attributes:
        children: Mapping from token ID to child node.
        paper_ids: Papers whose complete DocID ends exactly at this node.
    """

    children: dict[int, _TrieNode] = field(default_factory=dict)
    terminal_paper_ids: list[CorpusId] = field(default_factory=list)


def _build_trie(paper_to_tokens: dict[str, list[int]]) -> _TrieNode:
    """Build a prefix trie from the paper-to-token-sequence mapping.

    Each path from root to a node with non-empty `paper_ids` represents a complete DocID
    token sequence. Internal nodes allow constraining generation to valid prefixes.

    Args:
        paper_to_tokens: Mapping from corpus ID (as string) to the token sequence
        forming that paper's DocID.

    Returns:
        Root node of the constructed trie.
    """
    root = _TrieNode()

    for paper_id_str, tokens in paper_to_tokens.items():
        paper_id = CorpusId(int(paper_id_str))
        node = root

        for tok in tokens:
            if tok not in node.children:
                node.children[tok] = _TrieNode()
            node = node.children[tok]

        node.terminal_paper_ids.append(paper_id)

    return root


def _walk_trie(root: _TrieNode, tokens: list[int]) -> _TrieNode | None:
    """Walk the trie following a token sequence.

    Args:
        root: Starting node (typically the trie root).
        tokens: Token IDs to follow.

    Returns:
        The node reached after following all tokens, or `None` if any token has no
        matching child.
    """
    node = root

    for tok in tokens:
        child = node.children.get(tok)
        if child is None:
            return None
        node = child

    return node


def _collect_papers(node: _TrieNode) -> list[CorpusId]:
    """Collect all paper IDs reachable from a node (inclusive).

    Performs an iterative DFS to gather `paper_ids` from the given node and
    all its descendants.

    Args:
        node: Starting trie node.

    Returns:
        All paper IDs in the subtree (unsorted).
    """
    papers: list[CorpusId] = []
    stack = [node]

    while stack:
        current = stack.pop()
        papers.extend(current.terminal_paper_ids)
        stack.extend(current.children.values())

    return papers


def _extract_generated_tokens(
    sequence: torch.Tensor,
    prompt_len: int,
    eos_id: int,
    pad_id: int | None,
) -> list[int]:
    """Extract generated token IDs from a single beam sequence.

    Strips the prompt prefix, then truncates at the first EOS or padding token.

    Args:
        sequence: Full token sequence of shape `(seq_len,)`.
        prompt_len: Number of prompt tokens to skip.
        eos_id: End-of-sequence token ID.
        pad_id: Padding token ID (may be `None`).

    Returns:
        Generated token IDs (excluding EOS and padding).
    """
    tokens: list[int] = []

    for tok in sequence[prompt_len:].tolist():
        if tok in (eos_id, pad_id):
            break

        tokens.append(tok)

    return tokens


class ConstrainedDecoder:
    """Beam search decoder constrained to produce valid DocID prefixes.

    Builds a prefix trie from `docid_index.json` and uses HuggingFace's
    `model.generate()` with `prefix_allowed_tokens_fn` to constrain generation.
    Standard beam search is used; all beams are fully constrained by the trie.
    """

    def __init__(
        self,
        index_path: Path,
        model: PreTrainedModel,
        tokeniser: PreTrainedTokenizerBase,
        *,
        num_beams: int = 10,
        max_new_tokens: int = 256,
    ) -> None:
        """Load the DocID index and build the prefix trie.

        Args:
            index_path: Path to `docid_index.json`.
            model: Fine-tuned causal LM (may be a PEFT model).
            tokeniser: Tokeniser matching the model.
            num_beams: Number of beams to maintain.
            max_new_tokens: Maximum tokens to generate per query.
        """
        self._model = model
        self._tokeniser = tokeniser
        self._num_beams = num_beams
        self._max_new_tokens = max_new_tokens

        # Load the index and build the prefix trie
        index = DocIDIndex.model_validate_json(index_path.read_bytes())
        self._trie_root = _build_trie(index.paper_to_tokens)

        # EOS token ID
        self._eos_id = get_eos_token_id(tokeniser)

        logger.debug(
            "ConstrainedDecoder: %d papers, %d beams, trie built",
            len(index.paper_to_tokens),
            self._num_beams,
        )

    @property
    def device(self) -> torch.device:
        """Return the device of the underlying model."""
        return next(self._model.parameters()).device

    def decode_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[DecodedResult]:
        """Encode chat messages and run constrained beam search.

        Convenience wrapper around `decode` that handles tokenisation: applies the
        chat template, moves tensors to the model device, and constructs the attention
        mask.

        Args:
            messages: Chat messages (system + user) as produced by
                `~genret.training.data.format_facets_prompt`.

        Returns:
            List of `DecodedResult` sorted by score (best first).
        """
        prompt_ids = apply_chat_template_tensor(self._tokeniser, messages)
        input_ids = prompt_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)
        return self.decode(input_ids, attention_mask)

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[DecodedResult]:
        """Run constrained beam search on a single prompt.

        Uses `model.generate()` with `prefix_allowed_tokens_fn` to constrain decoding to
        valid DocID prefixes via the pre-built trie.  KV cache is managed automatically
        by HuggingFace.

        Args:
            input_ids: Prompt token IDs of shape `(1, seq_len)`.
            attention_mask: Attention mask of shape `(1, seq_len)`.

        Returns:
            List of `DecodedResult` sorted by score (best first), one per unique set of
            candidate papers.
        """
        prompt_len = input_ids.shape[1]
        trie_root = self._trie_root
        eos_id = self._eos_id

        def prefix_allowed_tokens_fn(
            _batch_id: int, beam_input_ids: torch.Tensor
        ) -> list[int]:
            gen_tokens = beam_input_ids[prompt_len:].tolist()
            node = _walk_trie(trie_root, gen_tokens)
            if node is None:
                return [eos_id]

            allowed = list(node.children.keys())
            # Only allow EOS at terminal nodes (complete DocIDs) or dead ends
            if node.terminal_paper_ids or not allowed:
                allowed.append(eos_id)
            return allowed

        generate_kwargs: dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self._max_new_tokens,
            "num_beams": self._num_beams,
            "num_return_sequences": self._num_beams,
            "pad_token_id": self._eos_id,
            "do_sample": False,
            "early_stopping": True,
            "output_scores": True,
            "return_dict_in_generate": True,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
        }

        output = model_generate(self._model, **generate_kwargs)

        sequences = output.sequences
        scores = output.sequences_scores
        assert isinstance(sequences, torch.Tensor)
        assert isinstance(scores, torch.Tensor)

        return self._parse_beam_sequences(sequences, scores, prompt_len)

    def _parse_beam_sequences(
        self,
        sequences: torch.Tensor,
        scores: torch.Tensor,
        prompt_len: int,
    ) -> list[DecodedResult]:
        """Convert raw beam search output into deduplicated results.

        For each returned beam sequence, extracts the generated tokens (stripping
        EOS and padding), walks the trie to find reachable papers, and deduplicates
        by the resulting paper set.

        Args:
            sequences: Token IDs of shape `(num_return_sequences, seq_len)` including
                the prompt prefix.
            scores: Beam scores of shape `(num_return_sequences,)`.
            prompt_len: Length of the prompt prefix to skip when extracting generated
                tokens.

        Returns:
            Deduplicated results sorted by score (best first).
        """
        eos_id = self._eos_id
        pad_id = get_pad_token_id(self._tokeniser)
        trie_root = self._trie_root

        results: list[DecodedResult] = []
        seen: set[tuple[CorpusId, ...]] = set()

        for i in range(sequences.shape[0]):
            gen_tokens = _extract_generated_tokens(
                sequences[i], prompt_len, eos_id, pad_id
            )

            node = _walk_trie(trie_root, gen_tokens)
            if node is None:
                continue

            # Prefer exact terminal match; fall back to subtree collection
            # (e.g. when max_new_tokens truncated the sequence)
            if node.terminal_paper_ids:
                paper_ids = sorted(node.terminal_paper_ids)
            else:
                paper_ids = sorted(_collect_papers(node))
            paper_key = tuple(paper_ids)
            if paper_key in seen:
                continue
            seen.add(paper_key)

            results.append(
                DecodedResult(
                    paper_ids=paper_ids,
                    score=float(scores[i].item()),
                    generated_tokens=gen_tokens,
                )
            )

        return results
