"""Local LLM judge with direct score output.

Loads a HuggingFace model directly and scores paper pairs with local
sequence-scored judging modes.
"""

# pyright: basic

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from genret._shed.transformers import apply_chat_template_str, load_tokenizer
from genret.paper_parsing import Facets, load_facets_map
from genret.processing.llm_judge import (
    PairToJudge,
    ScoredPair,
    ScoreRange,
    ScoringMode,
    load_pairs_to_judge,
    sample_pairs,
)
from genret.processing.local_judge_prompting import (
    PromptVariant,
    format_facets_for_local_judge,
)
from genret.processing.local_judge_sharding import select_pairs_for_shard
from genret.types import CorpusId
from genret.utils import iter_jsonl, load_processed_strings, open_auto

logger = logging.getLogger(__name__)


_SCORE_SYSTEM_PROMPT = """\
You are an expert scientific reviewer. You will be shown the extracted facets \
(problems, methods, results, contributions) of two research papers. Rate how \
related the papers are.

Two papers are "related" if a researcher reading one would benefit from \
knowing about the other — e.g. they address similar problems, use similar \
methods, build on each other's results, or make complementary contributions.

Be strict: superficial similarity (both use neural networks, both study NLP) \
is NOT enough. The connection must be specific and substantive."""

_BINARY_SYSTEM_PROMPT = """\
You are an expert scientific reviewer. You will be shown the extracted facets \
(problems, methods, results, contributions) of two research papers. Judge \
whether the papers are related.

Two papers are "related" if a researcher reading one would benefit from \
knowing about the other — e.g. they address similar problems, use similar \
methods, build on each other's results, or make complementary contributions.

Be strict: superficial similarity (both use neural networks, both study NLP) \
is NOT enough. The connection must be specific and substantive."""

_SCORE_USER_TEMPLATE = """\
## Paper A
{paper_a}

## Paper B
{paper_b}

Rate the relatedness of these two papers on a scale from 1 to 10, where:
- 1 = completely unrelated
- 5 = some topical overlap but no substantive connection
- 10 = deeply related (same problem, shared methods, or builds directly on the other)

Respond with only the number."""

_SCORE_USER_TEMPLATE_5 = """\
## Paper A
{paper_a}

## Paper B
{paper_b}

Rate the relatedness of these two papers on a scale from 1 to 5, where:
- 1 = completely unrelated
- 3 = some topical overlap but no substantive connection
- 5 = deeply related (same problem, shared methods, or builds directly on the other)

Respond with only the number."""

_BINARY_USER_TEMPLATE = """\
## Paper A
{paper_a}

## Paper B
{paper_b}

Judge whether these two papers are substantively related.

Respond with exactly `<label>related</label>` or `<label>not_related</label>`.
Do not output any other text."""

_VALID_SCORES_10 = tuple(range(1, 11))
"""Valid score values for the 1-10 scale."""

_VALID_SCORES_5 = tuple(range(1, 6))
"""Valid score values for the 1-5 scale."""

_IM_END_MARKER = "<|im_end|>"
"""ChatML end-of-message marker used to terminate assistant responses."""

_RELATED_LABEL = "<label>related</label>"
"""Canonical positive label used by binary sequence scoring."""

_NOT_RELATED_LABEL = "<label>not_related</label>"
"""Canonical negative label used by binary sequence scoring."""


def load_model(
    model_id: str,
    *,
    quantize_4bit: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a HuggingFace model and tokeniser for local inference.

    Args:
        model_id: HuggingFace model ID (e.g. ``"Qwen/Qwen3-32B"``).
        quantize_4bit: If True, load with 4-bit NF4 quantisation via
            bitsandbytes. Reduces VRAM from ~64GB to ~18GB for a 32B model
            at the cost of some quality degradation.

    Returns:
        Tuple of ``(model, tokenizer)``.
    """
    logger.info("Loading tokeniser for %s", model_id)
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(
        model_id,
        trust_remote_code=True,
    )

    quant_label = "4-bit NF4" if quantize_4bit else "float16"
    logger.info("Loading model %s (%s, device_map=auto)", model_id, quant_label)

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    return model, tokenizer


def _build_messages(
    facets_a: Facets,
    facets_b: Facets,
    *,
    prompt_variant: PromptVariant = PromptVariant.TOP2_NO_RESULTS,
    system_prompt: str = _SCORE_SYSTEM_PROMPT,
    user_template: str = _SCORE_USER_TEMPLATE,
) -> list[dict[str, str]]:
    """Build chat messages for scoring a paper pair.

    Args:
        facets_a: Facets for paper A.
        facets_b: Facets for paper B.
        prompt_variant: Prompt-compression variant for facet formatting.
        system_prompt: System prompt describing the judging task.
        user_template: Jinja template for the user message.

    Returns:
        Chat messages for the scoring prompt.
    """
    user_content = user_template.format(
        paper_a=format_facets_for_local_judge(facets_a, variant=prompt_variant),
        paper_b=format_facets_for_local_judge(facets_b, variant=prompt_variant),
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    facets_a: Facets,
    facets_b: Facets,
    *,
    prompt_variant: PromptVariant = PromptVariant.TOP2_NO_RESULTS,
    system_prompt: str = _SCORE_SYSTEM_PROMPT,
    user_template: str = _SCORE_USER_TEMPLATE,
) -> str:
    """Build the production scoring prompt using `enable_thinking=False`.

    Args:
        tokenizer: The tokeniser used for chat-template rendering.
        facets_a: Facets for paper A.
        facets_b: Facets for paper B.
        prompt_variant: Prompt-compression variant for facet formatting.
        system_prompt: System prompt describing the judging task.
        user_template: User prompt template describing the score range.

    Returns:
        The rendered prompt string.
    """
    messages = _build_messages(
        facets_a,
        facets_b,
        prompt_variant=prompt_variant,
        system_prompt=system_prompt,
        user_template=user_template,
    )
    return apply_chat_template_str(
        tokenizer,
        messages,
        enable_thinking=False,
    )


@dataclass(frozen=True, kw_only=True)
class _ScoringResult:
    """Result from a single scoring attempt."""

    score: float | None
    """The score, or None if no valid token was found."""
    raw_token: str
    """The raw token text that was generated/selected."""


@dataclass(frozen=True, kw_only=True)
class _SequenceCandidate:
    """A canonical answer sequence used for sequence scoring."""

    score: int
    """Score value represented by this candidate."""
    text: str
    """Human-readable answer text."""
    token_ids: tuple[int, ...]
    """Token IDs for the answer continuation, including termination."""


def _score_template_for_mode(mode: ScoringMode) -> str:
    """Return the user prompt template associated with a scoring mode.

    Args:
        mode: Scoring mode.

    Returns:
        Prompt template describing the requested answer format.
    """
    if mode is ScoringMode.BINARY_SEQUENCE:
        return _BINARY_USER_TEMPLATE
    if mode is ScoringMode.SEQUENCE5:
        return _SCORE_USER_TEMPLATE_5
    return _SCORE_USER_TEMPLATE


def _system_prompt_for_mode(mode: ScoringMode) -> str:
    """Return the system prompt associated with a scoring mode.

    Args:
        mode: Scoring mode.

    Returns:
        System prompt describing the task.
    """
    if mode is ScoringMode.BINARY_SEQUENCE:
        return _BINARY_SYSTEM_PROMPT
    return _SCORE_SYSTEM_PROMPT


def _score_range_for_mode(mode: ScoringMode) -> ScoreRange:
    """Return the score range associated with a scoring mode.

    Args:
        mode: Scoring mode.

    Returns:
        Score range label for output metadata.
    """
    if mode is ScoringMode.BINARY_SEQUENCE:
        return ScoreRange.ZERO_TO_ONE
    if mode is ScoringMode.SEQUENCE5:
        return ScoreRange.ONE_TO_FIVE
    return ScoreRange.ONE_TO_TEN


def _build_numeric_sequence_candidates(
    tokenizer: PreTrainedTokenizerBase,
    scores: tuple[int, ...],
) -> list[_SequenceCandidate]:
    """Build canonical answer sequences for sequence-level scoring.

    Each candidate is the bare score followed by `<|im_end|>` so that `1`
    and `10` are disambiguated as complete answers.

    Args:
        tokenizer: The tokeniser.
        scores: Score values to encode.

    Returns:
        Canonical candidate continuations in score order.
    """
    end_token_ids = tokenizer.encode(_IM_END_MARKER, add_special_tokens=False)
    if not end_token_ids:
        raise ValueError("Could not tokenise ChatML end marker")

    candidates: list[_SequenceCandidate] = []
    for score in scores:
        score_token_ids = tokenizer.encode(str(score), add_special_tokens=False)
        candidates.append(
            _SequenceCandidate(
                score=score,
                text=str(score),
                token_ids=tuple(score_token_ids + end_token_ids),
            )
        )
    return candidates


def _build_binary_sequence_candidates(
    tokenizer: PreTrainedTokenizerBase,
) -> list[_SequenceCandidate]:
    """Build canonical binary answer sequences for sequence scoring.

    Args:
        tokenizer: The tokeniser.

    Returns:
        Binary answer candidates in negative/positive order.
    """
    end_token_ids = tokenizer.encode(_IM_END_MARKER, add_special_tokens=False)
    if not end_token_ids:
        raise ValueError("Could not tokenise ChatML end marker")

    candidates: list[_SequenceCandidate] = []
    for score, text in ((0, _NOT_RELATED_LABEL), (1, _RELATED_LABEL)):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        candidates.append(
            _SequenceCandidate(
                score=score,
                text=text,
                token_ids=tuple(token_ids + end_token_ids),
            )
        )
    return candidates


def _build_sequence_candidates_for_mode(
    tokenizer: PreTrainedTokenizerBase,
    mode: ScoringMode,
) -> list[_SequenceCandidate]:
    """Build sequence candidates for a sequence-scored mode.

    Args:
        tokenizer: The tokeniser.
        mode: Scoring mode.

    Returns:
        Candidate continuations appropriate for `mode`.
    """
    if mode is ScoringMode.SEQUENCE10:
        return _build_numeric_sequence_candidates(tokenizer, _VALID_SCORES_10)
    if mode is ScoringMode.SEQUENCE5:
        return _build_numeric_sequence_candidates(tokenizer, _VALID_SCORES_5)
    if mode is ScoringMode.BINARY_SEQUENCE:
        return _build_binary_sequence_candidates(tokenizer)
    raise ValueError(f"Mode {mode} does not use sequence scoring")


def _score_batch_sequence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    candidates: list[_SequenceCandidate],
) -> list[_ScoringResult]:
    """Score prompts with microbatched prompt-candidate sequences.

    Args:
        model: The loaded model.
        tokenizer: The tokeniser.
        prompts: Prompt strings to score.
        candidates: Canonical answer continuations.

    Returns:
        One expected-value result per prompt.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Tokeniser must define either `pad_token_id` or `eos_token_id`"
        )
    if not isinstance(tokenizer.pad_token_id, int):
        raise TypeError("Tokeniser `pad_token_id` must be an integer")

    flat_sequences: list[list[int]] = []
    row_map: list[tuple[int, int, int]] = []
    max_candidate_len = max(len(candidate.token_ids) for candidate in candidates)

    for prompt_index, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        for candidate_index, candidate in enumerate(candidates):
            flat_sequences.append(prompt_ids + list(candidate.token_ids))
            row_map.append((prompt_index, candidate_index, len(candidate.token_ids)))

    candidate_log_probs = [[0.0] * len(candidates) for _ in prompts]
    max_flat_sequence_rows = 8
    logits_to_keep = max_candidate_len + 1

    for chunk_start in range(0, len(flat_sequences), max_flat_sequence_rows):
        # Slice one microbatch of prompt+candidate rows for the GPU forward pass.
        chunk_end = min(chunk_start + max_flat_sequence_rows, len(flat_sequences))
        chunk_sequences = flat_sequences[chunk_start:chunk_end]
        chunk_rows = row_map[chunk_start:chunk_end]

        # Left-pad all rows in the chunk so they can be packed into one tensor.
        max_len = max(len(sequence) for sequence in chunk_sequences)
        padded_input_ids = [
            [tokenizer.pad_token_id] * (max_len - len(sequence)) + sequence
            for sequence in chunk_sequences
        ]
        padded_attention_masks = [
            [0] * (max_len - len(sequence)) + [1] * len(sequence)
            for sequence in chunk_sequences
        ]

        input_ids = torch.tensor(padded_input_ids, device=model.device)
        attention_mask = torch.tensor(padded_attention_masks, device=model.device)

        # Request only the candidate-tail logits to avoid materialising the full
        # prompt-length logits tensor for every row.
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
            )

        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        for row_index, (prompt_index, candidate_index, candidate_len) in enumerate(
            chunk_rows
        ):
            # Align candidate tokens to the kept tail logits for this row.
            start_index = max_candidate_len - candidate_len
            total_log_prob = 0.0
            candidate = candidates[candidate_index]

            for offset, token_id in enumerate(candidate.token_ids):
                token_position = start_index + offset
                total_log_prob += log_probs[row_index, token_position, token_id].item()

            candidate_log_probs[prompt_index][candidate_index] = total_log_prob

    results: list[_ScoringResult] = []
    for log_prob_row in candidate_log_probs:
        max_log_prob = max(log_prob_row)
        weights = [math.exp(log_prob - max_log_prob) for log_prob in log_prob_row]
        total_weight = sum(weights)
        expected_value = sum(
            candidate.score * (weight / total_weight)
            for candidate, weight in zip(candidates, weights, strict=True)
        )
        best_index = max(range(len(log_prob_row)), key=lambda x: log_prob_row[x])
        results.append(
            _ScoringResult(
                score=expected_value,
                raw_token=candidates[best_index].text,
            )
        )

    return results


def _run_local_judge_for_pairs(
    *,
    pairs: list[PairToJudge],
    facets_path: Path,
    output_path: Path,
    model_id: str,
    mode: ScoringMode,
    prompt_variant: PromptVariant,
    batch_size: int,
    quantize_4bit: bool,
    metadata: dict[str, str | int | float | bool],
) -> None:
    """Run local judging on an explicit list of paper pairs.

    Args:
        pairs: Explicit pair list to score.
        facets_path: Path to facets JSONL file.
        output_path: Output JSONL path for scored pairs.
        model_id: HuggingFace model ID.
        mode: Scoring mode.
        prompt_variant: Prompt-compression variant for local judging.
        batch_size: Number of pairs per batch.
        quantize_4bit: If True, use 4-bit NF4 quantisation.
        metadata: Additional run metadata to write alongside aggregate counts.
    """
    facets_by_id = load_facets_map(facets_path)

    pair_source: dict[str, str] = {}
    for pair in pairs:
        pair_source[f"{pair.paper_a}|{pair.paper_b}"] = pair.source

    progress_path = output_path.with_suffix(".progress.txt")
    processed_keys = load_processed_strings(progress_path)

    if processed_keys:
        logger.info("Resuming: %d pairs already scored", len(processed_keys))

    pending: list[tuple[str, CorpusId, CorpusId]] = []
    for pair in pairs:
        key = f"{pair.paper_a}|{pair.paper_b}"
        if key in processed_keys:
            continue
        if pair.paper_a not in facets_by_id or pair.paper_b not in facets_by_id:
            continue
        pending.append((key, pair.paper_a, pair.paper_b))

    if not pending:
        logger.info("All pairs already scored!")
        return

    logger.info("Pairs to score: %d (mode=%s)", len(pending), mode)

    model, tokenizer = load_model(model_id, quantize_4bit=quantize_4bit)
    system_prompt = _system_prompt_for_mode(mode)
    score_range = _score_range_for_mode(mode)
    score_template = _score_template_for_mode(mode)
    sequence_candidates = _build_sequence_candidates_for_mode(tokenizer, mode)
    logger.info(
        "Sequence candidates: %s",
        [candidate.text for candidate in sequence_candidates],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if processed_keys else "w"

    scored_count = 0
    discarded_count = 0

    attempted_count = 0
    with (
        open(output_path, open_mode, encoding="utf-8") as out_f,
        open(progress_path, "a", encoding="utf-8") as prog_f,
    ):
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]

            prompts: list[str] = []
            for _key, paper_a, paper_b in batch:
                prompt = _build_prompt(
                    tokenizer,
                    facets_by_id[paper_a],
                    facets_by_id[paper_b],
                    prompt_variant=prompt_variant,
                    system_prompt=system_prompt,
                    user_template=score_template,
                )
                prompts.append(prompt)

            results = _score_batch_sequence(
                model,
                tokenizer,
                prompts,
                sequence_candidates,
            )

            for (key, paper_a, paper_b), result in zip(batch, results, strict=True):
                if result.score is None:
                    discarded_count += 1
                    logger.debug(
                        "Discarded pair %s: invalid token %r",
                        key,
                        result.raw_token,
                    )
                    prog_f.write(f"{key}\n")
                    prog_f.flush()
                    continue

                scored = ScoredPair(
                    paper_a=paper_a,
                    paper_b=paper_b,
                    source=pair_source.get(key, "unknown"),
                    score=result.score,
                    mode=mode,
                    score_range=score_range,
                )
                out_f.write(scored.model_dump_json() + "\n")
                out_f.flush()
                prog_f.write(f"{key}\n")
                prog_f.flush()
                scored_count += 1

            attempted_count += len(batch)
            if attempted_count % (batch_size * 10) == 0 or attempted_count == len(
                pending
            ):
                logger.info(
                    "Progress: %d/%d scored, %d discarded (%.1f%% discard rate)",
                    scored_count,
                    attempted_count,
                    discarded_count,
                    100.0 * discarded_count / max(attempted_count, 1),
                )

    total_processed = scored_count + discarded_count
    discard_rate = discarded_count / max(total_processed, 1)
    logger.info(
        "Done: %d scored, %d discarded (%.1f%% discard rate)",
        scored_count,
        discarded_count,
        100.0 * discard_rate,
    )

    if discard_rate > 0.05:
        logger.warning(
            "Discard rate %.1f%% exceeds 5%% threshold. Consider using the "
            "scan fallback strategy (generate up to N tokens and use the "
            "first position where a valid score token appears in top-K).",
            100.0 * discard_rate,
        )

    meta = {
        "model_id": model_id,
        "mode": mode,
        "prompt_variant": prompt_variant,
        "score_range": score_range,
        "quantize_4bit": quantize_4bit,
        "batch_size": batch_size,
        "scored_count": scored_count,
        "discarded_count": discarded_count,
        "discard_rate": discard_rate,
    }
    meta.update(metadata)
    meta_path = output_path.with_name(output_path.stem + "_run.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Scores written to %s", output_path)
    logger.info("Run metadata written to %s", meta_path)


def run_local_judge(
    facets_path: Path,
    retrieval_path: Path,
    output_path: Path,
    model_id: str,
    mode: ScoringMode = ScoringMode.BINARY_SEQUENCE,
    prompt_variant: PromptVariant = PromptVariant.TOP2_NO_RESULTS,
    batch_size: int = 8,
    n_papers: int = 500,
    negatives_per_query: int = 5,
    seed: int = 0,
    quantize_4bit: bool = False,
) -> None:
    """Run local LLM judge scoring on sampled paper pairs.

    Loads a HuggingFace model, scores pairs using the configured local judge
    mode, and writes results to JSONL. Supports resume via progress tracking.

    Args:
        facets_path: Path to facets JSONL file.
        retrieval_path: Path to retrieval_sets JSONL file.
        output_path: Output JSONL path for scored pairs.
        model_id: HuggingFace model ID.
        mode: Scoring mode.
        prompt_variant: Prompt-compression variant for local judging.
        batch_size: Number of pairs per batch.
        n_papers: Number of query papers to sample.
        negatives_per_query: Random negative pairs per query.
        seed: Random seed for pair sampling.
        quantize_4bit: If True, use 4-bit NF4 quantisation.
    """
    sampled_pairs = sample_pairs(retrieval_path, n_papers, negatives_per_query, seed)
    pairs = [
        PairToJudge(paper_a=CorpusId(paper_a), paper_b=CorpusId(paper_b), source=source)
        for paper_a, paper_b, source in sampled_pairs
    ]
    _run_local_judge_for_pairs(
        pairs=pairs,
        facets_path=facets_path,
        output_path=output_path,
        model_id=model_id,
        mode=mode,
        prompt_variant=prompt_variant,
        batch_size=batch_size,
        quantize_4bit=quantize_4bit,
        metadata={
            "input_mode": "sampled_retrieval",
            "retrieval_path": str(retrieval_path),
            "n_papers": n_papers,
            "negatives_per_query": negatives_per_query,
            "seed": seed,
        },
    )


def run_local_judge_pairs(
    *,
    facets_path: Path,
    pairs_path: Path,
    output_path: Path,
    model_id: str,
    mode: ScoringMode = ScoringMode.BINARY_SEQUENCE,
    prompt_variant: PromptVariant = PromptVariant.TOP2_NO_RESULTS,
    batch_size: int = 8,
    quantize_4bit: bool = False,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> None:
    """Run local LLM judge scoring on an explicit JSONL pair list.

    Args:
        facets_path: Path to facets JSONL file.
        pairs_path: Path to explicit `PairToJudge` JSONL, or a subset of an
            existing `JudgedPair` file.
        output_path: Output JSONL path for scored pairs.
        model_id: HuggingFace model ID.
        mode: Scoring mode.
        prompt_variant: Prompt-compression variant for local judging.
        batch_size: Number of pairs per batch.
        quantize_4bit: If True, use 4-bit NF4 quantisation.
        shard_index: Optional 0-based shard index for distributed scoring.
        num_shards: Optional total number of shards for distributed scoring.
    """
    if (shard_index is None) != (num_shards is None):
        raise ValueError("Specify both `shard_index` and `num_shards`, or neither.")

    pairs = load_pairs_to_judge(pairs_path)
    logger.info("Loaded %d explicit pairs from %s", len(pairs), pairs_path)
    if shard_index is not None and num_shards is not None:
        pairs = select_pairs_for_shard(
            pairs,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        logger.info(
            "Selected shard %d/%d with %d pairs",
            shard_index,
            num_shards,
            len(pairs),
        )

    metadata: dict[str, str | int | float | bool] = {
        "input_mode": "explicit_pairs",
        "pairs_path": str(pairs_path),
    }
    if shard_index is not None and num_shards is not None:
        metadata["shard_index"] = shard_index
        metadata["num_shards"] = num_shards

    _run_local_judge_for_pairs(
        pairs=pairs,
        facets_path=facets_path,
        output_path=output_path,
        model_id=model_id,
        mode=mode,
        prompt_variant=prompt_variant,
        batch_size=batch_size,
        quantize_4bit=quantize_4bit,
        metadata=metadata,
    )


def rerank_retrieval(
    *,
    facets_path: Path,
    retrieval_path: Path,
    output_path: Path,
    model_id: str,
    rerank_depth: int = 50,
    mode: ScoringMode = ScoringMode.BINARY_SEQUENCE,
    prompt_variant: PromptVariant = PromptVariant.TOP2_NO_RESULTS,
    batch_size: int = 4,
    quantize_4bit: bool = False,
) -> None:
    """Rerank retrieval results using the local judge.

    Reads a retrieval JSONL file, scores the top `rerank_depth` candidates
    per query with the local judge, reorders by score, and writes a new
    retrieval JSONL file in the same format.

    Args:
        facets_path: Path to facets JSONL file.
        retrieval_path: Path to retrieval results JSONL (from `batch-retrieve`).
        output_path: Output path for reranked retrieval JSONL.
        model_id: HuggingFace model ID.
        rerank_depth: Number of top candidates per query to rerank.
        mode: Scoring mode.
        prompt_variant: Prompt-compression variant for local judging.
        batch_size: Number of pairs per batch.
        quantize_4bit: If True, use 4-bit NF4 quantisation.
    """
    facets_by_id = load_facets_map(facets_path)

    # Load retrieval results
    queries: list[dict[str, Any]] = list(iter_jsonl(retrieval_path))
    logger.info("Loaded %d queries from %s", len(queries), retrieval_path)

    # Collect all pairs to score
    all_pairs: list[tuple[int, CorpusId, CorpusId]] = []
    for query_idx, query in enumerate(queries):
        qid = CorpusId(int(query["corpusid"]))
        results_list: list[dict[str, Any]] = query["results"]
        for r in results_list[:rerank_depth]:
            cid = CorpusId(int(r["corpusid"]))
            if qid in facets_by_id and cid in facets_by_id:
                all_pairs.append((query_idx, qid, cid))

    logger.info("Pairs to score: %d", len(all_pairs))

    # Load model and score
    model, tokenizer = load_model(model_id, quantize_4bit=quantize_4bit)
    system_prompt = _system_prompt_for_mode(mode)
    score_template = _score_template_for_mode(mode)
    sequence_candidates = _build_sequence_candidates_for_mode(tokenizer, mode)

    # Score in batches
    scores: dict[tuple[int, int], float] = {}
    scored_count = 0
    discarded_count = 0

    for batch_start in range(0, len(all_pairs), batch_size):
        batch = all_pairs[batch_start : batch_start + batch_size]

        prompts: list[str] = []
        for _query_idx, paper_a, paper_b in batch:
            prompt = _build_prompt(
                tokenizer,
                facets_by_id[paper_a],
                facets_by_id[paper_b],
                prompt_variant=prompt_variant,
                system_prompt=system_prompt,
                user_template=score_template,
            )
            prompts.append(prompt)

        results = _score_batch_sequence(
            model,
            tokenizer,
            prompts,
            sequence_candidates,
        )

        for (_query_idx, paper_a, paper_b), result in zip(batch, results, strict=True):
            if result.score is not None:
                scores[(paper_a, paper_b)] = result.score
                scored_count += 1
            else:
                discarded_count += 1

        attempted = batch_start + len(batch)
        if attempted % (batch_size * 10) == 0 or attempted == len(all_pairs):
            logger.info(
                "Progress: %d/%d scored, %d discarded",
                scored_count,
                attempted,
                discarded_count,
            )

    logger.info(
        "Scoring done: %d scored, %d discarded (%.1f%%)",
        scored_count,
        discarded_count,
        100.0 * discarded_count / max(scored_count + discarded_count, 1),
    )

    # Write reranked output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_auto(output_path, "w") as out_f:
        for query in queries:
            qid = CorpusId(int(query["corpusid"]))
            query_results: list[dict[str, Any]] = query["results"]

            # Split into reranked window and tail
            window = query_results[:rerank_depth]
            tail = query_results[rerank_depth:]

            # Sort window by judge score (descending), fall back to original score
            window.sort(
                key=lambda r: scores.get((qid, CorpusId(int(r["corpusid"]))), -1.0),
                reverse=True,
            )

            # Add judge_score to each result in the window
            for r in window:
                judge_score = scores.get((qid, CorpusId(int(r["corpusid"]))))
                if judge_score is not None:
                    r["judge_score"] = round(judge_score, 4)

            record: dict[str, Any] = {
                "corpusid": int(qid),
                "results": window + tail,
            }
            out_f.write(json.dumps(record) + "\n")

    logger.info("Reranked output written to %s", output_path)
