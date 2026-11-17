# pyright: basic
# ruff: noqa: PLC0415  Enable function-level imports
"""Dataset construction, prompt formatting, and batch collation for GenRet training.

Builds two types of training examples:
- **Indexing**: ``facets(A) -> docid(A)`` -- teach the model to map a paper's facets
  to its own document identifier.
- **Retrieval**: ``facets(A) -> docid(B)`` -- teach the model to map a query paper's
  facets to a related paper's document identifier.

The two types are mixed at a configurable ratio and shuffled deterministically.

CUDA-dependent imports (torch, transformers) are deferred so this module can
be imported without torch installed (e.g. for testing pure-Python logic).
"""

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from genret._shed.transformers import (
    apply_chat_template_ids,
    get_eos_token_id,
    get_pad_token_id,
)
from genret.paper_parsing import (
    DocIDRecord,
    FacetEvidence,
    FacetRecordType,
    Facets,
    TrainingPair,
    evidence_from_type,
    facets_from_type,
)
from genret.types import CorpusId
from genret.utils import iter_jsonl, resolve_jsonl, sample

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a generative retrieval system. Given the facets of a research paper, "
    "generate the structured document identifier (DocID) of a relevant paper."
)
"""System message used as the first turn in the Llama chat template."""


def format_facets_prompt(
    facets: Facets,
    evidence: FacetEvidence | None = None,
) -> list[dict[str, str]]:
    """Build Llama chat messages from a paper's extracted facets.

    Produces a two-message conversation: a system prompt and a user message listing the
    paper's problems, methods, results, and contributions as bullet lists.  When
    ``evidence`` is provided, verbatim quotes from the paper are included beneath each
    item.

    Args:
        facets: Extracted facets for a single paper.
        evidence: Optional per-item evidence from extraction.

    Returns:
        List of ``{"role": ..., "content": ...}`` dicts suitable for
        ``tokeniser.apply_chat_template``.
    """
    sections: list[str] = []
    for name, items in [
        ("Problems", facets.problems),
        ("Methods", facets.methods),
        ("Results", facets.results),
        ("Contributions", facets.contributions),
    ]:
        if not items:
            continue

        item_evidence: list[list[str]] = (
            evidence.get(name.lower(), []) if evidence else []
        )
        lines: list[str] = []

        for i, item in enumerate(items):
            lines.append(f"- {item}")
            if i < len(item_evidence):
                for quote in item_evidence[i]:
                    lines.append(f'  > "{quote}"')

        sections.append(f"{name}:\n" + "\n".join(lines))

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


@dataclass(frozen=True, kw_only=True)
class TrainingExample:
    """A single training example pairing a facets prompt with a target DocID.

    Attributes:
        prompt_messages: Chat messages for the input (system + user).
        target_docid: The target DocID string to generate.
        kind: Pair type — ``"indexing"`` or ``"retrieval"``.  Defaults to
            ``"retrieval"`` for backward compatibility with callers that do
            not set this field.
        soft_targets: Optional graph-smoothed soft-target entries as
            ``(position, token_id, weight)`` tuples. ``position`` indexes
            into the encoded target DocID tokens (the EOS slot is
            ``len(docid_tokens)``); weights are normalised to sum to 1 per
            position. Consumed by ``SmoothedTargetTrainer``; ignored by the
            standard CE path.
        loss_weight: Per-row weight for graph-weighted CE reweighting (the
            continuous analogue of graph-weighted sample replication).
            ``1.0`` (default) leaves the standard CE loss unchanged; values
            > 1 amplify the row's contribution. Consumed by
            ``WeightedCETrainer``; ignored by the standard CE path.
        hard_negative_targets: Optional hard-negative DocID strings for
            per-row unlikelihood training. Each string is the encoded
            DocID of a graph-similar non-gold paper's bullet. Consumed by
            ``WeightedCETrainer`` (adds a UL term per neg); ignored by the
            standard CE path.
    """

    prompt_messages: list[dict[str, str]]
    target_docid: str
    kind: str = "retrieval"
    soft_targets: tuple[tuple[int, int, float], ...] | None = None
    loss_weight: float = 1.0
    hard_negative_targets: tuple[str, ...] = ()


def _load_facets_and_evidence(
    processed_dir: Path,
    *,
    facets_type: FacetRecordType,
) -> tuple[dict[CorpusId, Facets], dict[CorpusId, FacetEvidence]]:
    """Load facets and evidence from JSONL in a single pass.

    Args:
        processed_dir: Path to the processed/ directory.
        facets_type: On-disk schema for facet records.

    Returns:
        Tuple of (facets map, evidence map).  The evidence map is empty for
        standard facet records that carry no evidence.
    """
    facets_map: dict[CorpusId, Facets] = {}
    evidence_map: dict[CorpusId, FacetEvidence] = {}

    for raw in iter_jsonl(resolve_jsonl(processed_dir / "facets.jsonl.gz")):
        f = facets_from_type(raw, facets_type)
        facets_map[f.corpusid] = f
        if ev := evidence_from_type(raw, facets_type):
            evidence_map[f.corpusid] = ev

    return facets_map, evidence_map


def _load_docids_map(processed_dir: Path) -> dict[CorpusId, str]:
    """Load DocID records from JSONL into a corpus-ID-keyed dict.

    Args:
        processed_dir: Path to the processed/ directory.

    Returns:
        Mapping from corpus ID to encoded DocID string.
    """
    result: dict[CorpusId, str] = {}

    for raw in iter_jsonl(resolve_jsonl(processed_dir / "docids.jsonl.gz")):
        record = DocIDRecord.model_validate(raw)
        result[record.corpusid] = record.docid

    return result


def _load_training_pairs(processed_dir: Path) -> list[TrainingPair]:
    """Load training pairs from JSONL.

    Args:
        processed_dir: Path to the processed/ directory.

    Returns:
        List of training pairs.
    """
    path = resolve_jsonl(Path(processed_dir) / "training_pairs.jsonl.gz")
    return [TrainingPair.model_validate(raw) for raw in iter_jsonl(path)]


def _build_examples(
    processed_dir: Path,
    *,
    indexing_ratio: float,
    rng: random.Random,
    split_ids: set[CorpusId] | None,
    facets_type: FacetRecordType,
    include_evidence: bool = True,
) -> list[TrainingExample]:
    """Load data and build the mixed example list.

    Args:
        processed_dir: Path to the processed/ directory.
        indexing_ratio: Target proportion of indexing examples (0.0-1.0).
        rng: RNG object used for sampling.
        split_ids: If provided, restrict to papers in this set.
        facets_type: On-disk schema for facet records.
        include_evidence: Whether to include evidence in prompts.

    Returns:
        Shuffled list of training examples.
    """
    facets_map, evidence_map = _load_facets_and_evidence(
        processed_dir, facets_type=facets_type
    )
    if not include_evidence:
        evidence_map = dict[CorpusId, FacetEvidence]()
    docids_map = _load_docids_map(processed_dir)
    pairs = _load_training_pairs(processed_dir)

    # Filter to split if requested
    if split_ids is not None:
        facets_map = {k: v for k, v in facets_map.items() if k in split_ids}
        evidence_map = {k: v for k, v in evidence_map.items() if k in split_ids}
        docids_map = {k: v for k, v in docids_map.items() if k in split_ids}
        pairs = [p for p in pairs if p.paper_a in split_ids and p.paper_b in split_ids]

    # Papers that have both facets and docids
    valid_ids = set(facets_map) & set(docids_map)
    logger.info(
        "Valid papers (have both facets and docid): %d / %d facets, %d docids",
        len(valid_ids),
        len(facets_map),
        len(docids_map),
    )

    # Build indexing examples -- facets(A) maps to docid(A)
    indexing_examples = [
        TrainingExample(
            prompt_messages=format_facets_prompt(
                facets_map[cid], evidence=evidence_map.get(cid)
            ),
            target_docid=docids_map[cid],
        )
        for cid in sorted(valid_ids)
    ]

    # Build retrieval examples -- facets(A) maps to docid(B)
    retrieval_examples: list[TrainingExample] = []
    skipped = 0
    for pair in pairs:
        if pair.paper_a not in valid_ids or pair.paper_b not in valid_ids:
            skipped += 1
            continue

        retrieval_examples.append(
            TrainingExample(
                prompt_messages=format_facets_prompt(
                    facets_map[pair.paper_a],
                    evidence=evidence_map.get(pair.paper_a),
                ),
                target_docid=docids_map[pair.paper_b],
            )
        )

    if skipped > 0:
        logger.info("Skipped %d training pairs (missing facets or docid)", skipped)

    result = _mix_examples(indexing_examples, retrieval_examples, indexing_ratio, rng)

    logger.info(
        "Dataset: %d examples (%d indexing, %d retrieval)",
        len(result),
        len(indexing_examples),
        len(retrieval_examples),
    )

    return result


class GenRetDataset:
    """Dataset mixing indexing and retrieval examples for GenRet training.

    Indexing examples map ``facets(A) -> docid(A)`` for every paper that has both facets
    and a DocID.  Retrieval examples map ``facets(A) -> docid(B)`` from training pairs
    where both sides have data.

    Implements ``__len__`` and ``__getitem__`` for compatibility with PyTorch
    ``DataLoader`` and HuggingFace ``Trainer``.
    """

    def __init__(
        self,
        processed_dir: Path,
        *,
        indexing_ratio: float = 0.5,
        rng: random.Random,
        split_ids: set[CorpusId] | None = None,
        facets_type: FacetRecordType,
        include_evidence: bool = True,
    ) -> None:
        """Load data and build the example list.

        Args:
            processed_dir: Path to the ``processed/`` directory containing
                ``facets.jsonl.gz``, ``docids.jsonl.gz``, and
                ``training_pairs.jsonl.gz``.
            indexing_ratio: Target proportion of indexing examples (0.0-1.0).
            rng: RNG object used for sampling.
            split_ids: If provided, restrict to papers in this set.
            facets_type: On-disk schema for facet records.
            include_evidence: Whether to include evidence in prompts.
        """
        self._examples = _build_examples(
            processed_dir,
            indexing_ratio=indexing_ratio,
            rng=rng,
            split_ids=split_ids,
            facets_type=facets_type,
            include_evidence=include_evidence,
        )

    def __len__(self) -> int:
        """Return the total number of training examples."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        """Return the training example at the given index."""
        return self._examples[idx]


def _mix_examples(
    indexing: list[TrainingExample],
    retrieval: list[TrainingExample],
    indexing_ratio: float,
    rng: random.Random,
) -> list[TrainingExample]:
    """Mix indexing and retrieval examples at the target ratio.

    If one type is overrepresented relative to the target ratio, it is subsampled. The
    result is shuffled deterministically.

    Args:
        indexing: Indexing examples (facets -> own docid).
        retrieval: Retrieval examples (facets -> related docid).
        indexing_ratio: Target proportion of indexing examples.
        rng: RNG object used for sampling.

    Returns:
        Combined and shuffled example list.
    """
    if not indexing and not retrieval:
        return []

    # Subsample to achieve the target ratio
    if indexing_ratio <= 0.0:
        combined = list(retrieval)
    elif indexing_ratio >= 1.0:
        combined = list(indexing)
    else:
        n_idx = len(indexing)
        n_ret = len(retrieval)

        # Keep all of one type and subsample the other to hit the target ratio
        target_ret_per_idx = (1.0 - indexing_ratio) / indexing_ratio
        desired_ret = int(n_idx * target_ret_per_idx)

        if desired_ret <= n_ret:
            # Subsample retrieval
            sampled_ret = sample(rng, retrieval, desired_ret)
            combined = list(indexing) + sampled_ret
        else:
            # Subsample indexing
            target_idx_per_ret = indexing_ratio / (1.0 - indexing_ratio)
            desired_idx = int(n_ret * target_idx_per_ret)
            sampled_idx = sample(rng, indexing, desired_idx)
            combined = sampled_idx + list(retrieval)

    rng.shuffle(combined)
    return combined


def collate_fn(
    examples: Sequence[TrainingExample],
    tokeniser: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> dict[str, Any]:
    """Collate training examples into a padded batch.

    For each example:
        1. Apply the chat template to get prompt token IDs.
        2. Tokenise the target DocID separately.
        3. Concatenate prompt, docid, and EOS tokens.
        4. Build labels with -100 masking the prompt portion.

    The batch is left-padded to the longest sequence.

    When any example carries ``soft_targets``, the batch additionally
    includes flat COO tensors ``soft_batch``, ``soft_pos``, ``soft_token``,
    and ``soft_weight`` locating each soft-target entry in the padded batch
    (``soft_pos`` is the absolute position of the smoothed label token).
    These keys are only present when at least one entry survives, so the
    standard ``Trainer`` path is unaffected for smoothing-off datasets.

    Args:
        examples: Sequence of training examples.
        tokeniser: Tokeniser with chat template support.
        max_seq_length: Maximum total sequence length (truncates from the right).

    Returns:
        Dict with ``input_ids``, ``attention_mask``, and ``labels`` tensors,
        plus the four ``soft_*`` tensors when soft targets are present.
    """
    import torch

    all_input_ids: list[list[int]] = []
    all_labels: list[list[int]] = []
    all_prompt_lens: list[int] = []
    hard_neg_owner: list[int] = []

    eos_id = get_eos_token_id(tokeniser)
    pad_id_val = get_pad_token_id(tokeniser)
    assert pad_id_val is not None
    pad_id = pad_id_val

    def _encode_row(
        prompt_ids: list[int], target: str
    ) -> tuple[list[int], list[int], int]:
        target_ids = tokeniser.encode(target, add_special_tokens=False)
        full_ids = prompt_ids + target_ids + [eos_id]
        if len(full_ids) > max_seq_length:
            full_ids = full_ids[:max_seq_length]
            full_ids[-1] = eos_id
        prompt_len_local = min(len(prompt_ids), len(full_ids))
        labels_local = [-100] * prompt_len_local + full_ids[prompt_len_local:]
        return full_ids, labels_local, prompt_len_local

    # Gold rows first (indices [0, B)).
    for ex in examples:
        prompt_ids = apply_chat_template_ids(tokeniser, ex.prompt_messages)
        full_ids, labels, prompt_len = _encode_row(prompt_ids, ex.target_docid)
        if prompt_len >= len(full_ids):
            logger.warning(
                "Prompt fills entire max_seq_length (%d tokens) — "
                "all labels are masked (zero loss for this example)",
                max_seq_length,
            )
        all_input_ids.append(full_ids)
        all_labels.append(labels)
        all_prompt_lens.append(prompt_len)

    # Hard-negative rows appended after (indices [B, B+N)).
    n_gold = len(examples)
    for owner_idx, ex in enumerate(examples):
        if not ex.hard_negative_targets:
            continue
        prompt_ids = apply_chat_template_ids(tokeniser, ex.prompt_messages)
        for neg_target in ex.hard_negative_targets:
            full_ids, labels, prompt_len = _encode_row(prompt_ids, neg_target)
            all_input_ids.append(full_ids)
            all_labels.append(labels)
            all_prompt_lens.append(prompt_len)
            hard_neg_owner.append(owner_idx)
    has_hard_negs = bool(hard_neg_owner)

    # Left-pad to the longest sequence in the batch
    max_len = max(len(ids) for ids in all_input_ids)

    padded_input_ids: list[list[int]] = []
    padded_labels: list[list[int]] = []
    attention_masks: list[list[int]] = []

    soft_batch: list[int] = []
    soft_pos: list[int] = []
    soft_token: list[int] = []
    soft_weight: list[float] = []
    row_loss_weights: list[float] = [float(ex.loss_weight) for ex in examples]
    has_row_weighting = any(abs(w - 1.0) > 1e-9 for w in row_loss_weights)

    for i, (ids, labels) in enumerate(zip(all_input_ids, all_labels)):
        pad_len = max_len - len(ids)
        padded_input_ids.append([pad_id] * pad_len + ids)
        padded_labels.append([-100] * pad_len + labels)
        attention_masks.append([0] * pad_len + [1] * len(ids))

        # soft_targets only apply to gold rows (indices [0, n_gold)).
        if i >= n_gold:
            continue
        ex = examples[i]
        prompt_len = all_prompt_lens[i]
        if not ex.soft_targets:
            continue
        for rel_pos, token, weight in ex.soft_targets:
            label_pos = prompt_len + rel_pos
            if label_pos >= len(ids):
                continue
            soft_batch.append(i)
            soft_pos.append(pad_len + label_pos)
            soft_token.append(token)
            soft_weight.append(weight)

    batch: dict[str, Any] = {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }
    if soft_batch:
        batch["soft_batch"] = torch.tensor(soft_batch, dtype=torch.long)
        batch["soft_pos"] = torch.tensor(soft_pos, dtype=torch.long)
        batch["soft_token"] = torch.tensor(soft_token, dtype=torch.long)
        batch["soft_weight"] = torch.tensor(soft_weight, dtype=torch.float32)
    if has_row_weighting:
        batch["loss_weight"] = torch.tensor(row_loss_weights, dtype=torch.float32)
    if has_hard_negs:
        batch["hard_neg_owner"] = torch.tensor(hard_neg_owner, dtype=torch.long)
        batch["n_gold_rows"] = torch.tensor(n_gold, dtype=torch.long)
    return batch
