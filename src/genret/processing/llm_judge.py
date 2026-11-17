"""LLM-as-judge evaluation for graph retrieval quality.

Samples paper pairs from retrieval results and asks an LLM to judge
whether the papers are related, producing a ground-truth-free evaluation
set.
"""

import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from genret import llm
from genret.paper_parsing import Facets, load_facets_map
from genret.processing.facet_graph_eval import load_retrieval_sets
from genret.types import CorpusId
from genret.utils import (
    AsyncLineWriter,
    CallbackColumn,
    OpenMode,
    load_processed_strings,
    sample,
    track_progress,
)

logger = logging.getLogger(__name__)


class RelationshipType(StrEnum):
    """Relationship labels emitted by the LLM judge."""

    SAME_PROBLEM = "same_problem"
    SHARED_METHOD = "shared_method"
    BUILDS_ON = "builds_on"
    COMPLEMENTARY = "complementary"
    NONE = "none"


class ScoringMode(StrEnum):
    """Scoring modes supported by the local judge."""

    BINARY_SEQUENCE = "binary_sequence"
    """Default mode to use."""
    SEQUENCE5 = "sequence5"
    """Deprecated."""
    SEQUENCE10 = "sequence10"
    """Deprecated."""


class ScoreRange(StrEnum):
    """Score ranges supported by the local judge."""

    ZERO_TO_ONE = "0-1"
    ONE_TO_FIVE = "1-5"
    ONE_TO_TEN = "1-10"


def _append_progress(path: Path, key: str) -> None:
    """Append a pair key to the progress file.

    Args:
        path: Progress file path.
        key: Pair key string (``"paper_a|paper_b"``).
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{key}\n")


_JUDGE_SYSTEM_PROMPT = """\
You are an expert scientific reviewer. You will be shown the extracted facets \
(problems, methods, results, contributions) of two research papers. Judge \
whether the papers are related.

Two papers are "related" if a researcher reading one would benefit from \
knowing about the other — e.g. they address similar problems, use similar \
methods, build on each other's results, or make complementary contributions.

Be strict: superficial similarity (both use neural networks, both study NLP) \
is NOT enough. The connection must be specific and substantive."""

_JUDGE_USER_TEMPLATE = """\
## Paper A
{paper_a}

## Paper B
{paper_b}"""


class JudgementResponse(BaseModel, frozen=True):
    """Structured response from the LLM judge."""

    related: bool
    """Whether the two papers are substantively related."""
    confidence: float
    """Confidence in the judgement, from 0.0 to 1.0."""
    reasoning: str
    """Brief explanation of why the papers are or are not related."""
    relationship_type: RelationshipType
    """Type of relationship if related."""


class JudgedPair(BaseModel, frozen=True):
    """A judged paper pair with the LLM's assessment."""

    paper_a: CorpusId
    """Corpus ID of paper A."""
    paper_b: CorpusId
    """Corpus ID of paper B."""
    source: str
    """How this pair was sampled: `"retrieved"` or `"random_negative"`."""
    related: bool
    """LLM judgement: are the papers related?"""
    confidence: float
    """LLM confidence in the judgement."""
    reasoning: str
    """LLM explanation."""
    relationship_type: RelationshipType
    """Type of relationship if related."""


class PairToJudge(BaseModel, frozen=True):
    """Explicit paper pair input for replay-style judging.

    This model intentionally matches the leading fields of `JudgedPair` so
    that existing judgement files can be reused as pair lists.
    """

    paper_a: CorpusId
    """Corpus ID of paper A."""
    paper_b: CorpusId
    """Corpus ID of paper B."""
    source: str
    """Pair origin label such as `"retrieved"` or `"random_negative"`."""


class ConfidenceBucket(StrEnum):
    """Confidence bands used when stratifying existing judgements."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ScoredPair(BaseModel, frozen=True):
    """A scored paper pair from the local judge."""

    paper_a: CorpusId
    """Corpus ID of paper A."""
    paper_b: CorpusId
    """Corpus ID of paper B."""
    source: str
    """How this pair was sampled: `"retrieved"` or `"random_negative"`."""
    score: float
    """Relatedness score on the configured scale."""
    mode: ScoringMode
    """Scoring mode used."""
    score_range: ScoreRange = ScoreRange.ONE_TO_TEN
    """Score range associated with `score`."""


def _format_facets(facets: Facets) -> str:
    """Format a paper's facets for the LLM prompt.

    Args:
        facets: Extracted facets for one paper.

    Returns:
        Human-readable facet summary.
    """
    parts: list[str] = [f"Corpus ID: {facets.corpusid}"]

    for name, items in (
        ("Problems", facets.problems),
        ("Methods", facets.methods),
        ("Results", facets.results),
        ("Contributions", facets.contributions),
    ):
        if items:
            bullet_list = "\n".join(f"- {item}" for item in items)
            parts.append(f"**{name}:**\n{bullet_list}")

    return "\n\n".join(parts)


def sample_pairs(
    retrieval_path: Path,
    n_papers: int,
    negatives_per_query: int,
    seed: int,
) -> list[tuple[int, int, str]]:
    """Sample paper pairs for LLM judgement.

    For each sampled query paper, includes:
    - All retrieved papers (source='retrieved')
    - `negatives_per_query` random papers not in the retrieval set
      (source='random_negative')

    Args:
        retrieval_path: Path to retrieval_sets JSONL file.
        n_papers: Number of query papers to sample.
        negatives_per_query: Random negatives per query paper.
        seed: Random seed for reproducibility.

    Returns:
        List of `(paper_a, paper_b, source)` tuples.
    """
    retrieval_sets = load_retrieval_sets(retrieval_path)
    all_ids = list(retrieval_sets.keys())

    rng = random.Random(seed)
    sampled_queries = sample(rng, all_ids, n_papers)

    pairs: list[tuple[int, int, str]] = []
    all_ids_set = set(all_ids)

    for query_id in sampled_queries:
        results = retrieval_sets[query_id]

        # Retrieved pairs
        for result_id in results:
            pairs.append((query_id, result_id, "retrieved"))

        # Random negatives (papers not in retrieval set)
        retrieved_set = set(results) | {query_id}
        neg_pool = list(all_ids_set - retrieved_set)
        negatives = sample(rng, neg_pool, negatives_per_query)

        for neg_id in negatives:
            pairs.append((query_id, neg_id, "random_negative"))

    logger.info(
        f"Sampled {len(pairs)} pairs from {len(sampled_queries)} queries "
        f"({len([p for p in pairs if p[2] == 'retrieved'])} retrieved, "
        f"{len([p for p in pairs if p[2] == 'random_negative'])} negatives)"
    )
    return pairs


def load_pairs_to_judge(path: Path) -> list[PairToJudge]:
    """Load explicit paper pairs from a JSONL file.

    Args:
        path: Path to a JSONL file containing `PairToJudge` records, or
            `JudgedPair`-compatible records with the same leading fields.

    Returns:
        Explicit pair list in file order.
    """
    pairs: list[PairToJudge] = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            pairs.append(PairToJudge.model_validate_json(stripped))
    return pairs


def normalise_judged_pair_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Normalise legacy `JudgedPair` payloads before validation.

    Args:
        data: Parsed JSON object for one judgement row.

    Returns:
        Normalised judgement payload.
    """
    relationship_type = data.get("relationship_type", "none")
    valid_types = {
        RelationshipType.SAME_PROBLEM.value,
        RelationshipType.SHARED_METHOD.value,
        RelationshipType.BUILDS_ON.value,
        RelationshipType.COMPLEMENTARY.value,
        RelationshipType.NONE.value,
    }
    if relationship_type not in valid_types:
        data["relationship_type"] = RelationshipType.NONE.value
    return data


def load_judged_pairs(path: Path) -> list[JudgedPair]:
    """Load judged paper pairs from a JSONL file.

    Args:
        path: Path to a JSONL file containing `JudgedPair` records.

    Returns:
        Judged pairs in file order.
    """
    judged_pairs: list[JudgedPair] = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            judged_pairs.append(
                JudgedPair.model_validate(normalise_judged_pair_dict(data))
            )
    return judged_pairs


def _confidence_bucket(
    confidence: float,
    *,
    low_max: float,
    high_min: float,
) -> ConfidenceBucket:
    """Bucket a confidence score for stratified sampling.

    Args:
        confidence: Confidence value in `[0.0, 1.0]`.
        low_max: Upper bound for the low-confidence bucket.
        high_min: Lower bound for the high-confidence bucket.

    Returns:
        Confidence bucket label.
    """
    if confidence < low_max:
        return ConfidenceBucket.LOW
    if confidence < high_min:
        return ConfidenceBucket.MEDIUM
    return ConfidenceBucket.HIGH


def _allocate_stratified_counts(
    stratum_sizes: dict[tuple[str, bool, ConfidenceBucket], int],
    total_count: int,
) -> dict[tuple[str, bool, ConfidenceBucket], int]:
    """Allocate sample counts across strata proportionally.

    Args:
        stratum_sizes: Available item counts for each stratum.
        total_count: Requested total sample size.

    Returns:
        Per-stratum sample counts summing to `total_count`.
    """
    total_available = sum(stratum_sizes.values())
    if total_available == 0:
        return dict.fromkeys(stratum_sizes, 0)

    allocated = dict.fromkeys(stratum_sizes, 0)
    non_empty_keys = [key for key, size in stratum_sizes.items() if size > 0]
    if total_count >= len(non_empty_keys):
        for key in non_empty_keys:
            allocated[key] = 1

    remaining_total = total_count - sum(allocated.values())
    remaining_sizes = {
        key: max(0, size - allocated[key]) for key, size in stratum_sizes.items()
    }
    remaining_available = sum(remaining_sizes.values())
    if remaining_available == 0:
        return allocated

    raw_targets = {
        key: (size * remaining_total) / remaining_available
        for key, size in remaining_sizes.items()
    }
    base_allocations = {
        key: min(size, int(raw_targets[key])) for key, size in remaining_sizes.items()
    }
    for key, extra in base_allocations.items():
        allocated[key] += extra

    remaining = total_count - sum(allocated.values())
    while remaining > 0:
        best_key: tuple[str, bool, ConfidenceBucket] | None = None
        best_fraction = -1.0
        for key, size in remaining_sizes.items():
            if base_allocations[key] >= size:
                continue
            fraction = raw_targets[key] - base_allocations[key]
            if fraction > best_fraction:
                best_fraction = fraction
                best_key = key
        if best_key is None:
            break
        allocated[best_key] += 1
        base_allocations[best_key] += 1
        remaining -= 1

    return allocated


def sample_judged_pairs_subset(
    input_path: Path,
    output_path: Path,
    count: int,
    seed: int,
    *,
    low_confidence_max: float = 1.0 / 3.0,
    high_confidence_min: float = 2.0 / 3.0,
) -> None:
    """Sample a representative subset from an existing judgement file.

    The subset is stratified by `source`, boolean `related` label, and a
    three-way confidence bucket. The sampled records are written back out as
    full `JudgedPair` JSONL lines so they can be reused both as a replay pair
    list and as the original GPT-5-mini reference slice.

    Args:
        input_path: Source `JudgedPair` JSONL file.
        output_path: Output JSONL path for the sampled subset.
        count: Requested subset size.
        seed: Random seed for reproducibility.
        low_confidence_max: Upper bound for the low-confidence bucket.
        high_confidence_min: Lower bound for the high-confidence bucket.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 <= low_confidence_max <= 1.0:
        raise ValueError("low_confidence_max must be in [0.0, 1.0]")
    if not 0.0 <= high_confidence_min <= 1.0:
        raise ValueError("high_confidence_min must be in [0.0, 1.0]")
    if low_confidence_max >= high_confidence_min:
        raise ValueError("low_confidence_max must be less than high_confidence_min")

    judged_pairs = load_judged_pairs(input_path)
    total_available = len(judged_pairs)
    if total_available == 0:
        raise ValueError(f"no judged pairs found in {input_path}")

    actual_count = min(count, total_available)
    if actual_count < count:
        logger.warning(
            "Requested %d pairs, but only %d are available; writing all pairs",
            count,
            total_available,
        )

    strata: dict[tuple[str, bool, ConfidenceBucket], list[int]] = defaultdict(list)
    for index, pair in enumerate(judged_pairs):
        key = (
            pair.source,
            pair.related,
            _confidence_bucket(
                pair.confidence,
                low_max=low_confidence_max,
                high_min=high_confidence_min,
            ),
        )
        strata[key].append(index)

    allocations = _allocate_stratified_counts(
        {key: len(indices) for key, indices in strata.items()},
        actual_count,
    )

    rng = random.Random(seed)
    sampled_indices: set[int] = set()
    for key, indices in strata.items():
        target = allocations[key]
        if target == 0:
            continue
        sampled_indices.update(rng.sample(indices, target))

    sampled_pairs = [
        pair for index, pair in enumerate(judged_pairs) if index in sampled_indices
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in sampled_pairs:
            f.write(pair.model_dump_json())
            f.write("\n")

    logger.info(
        "Wrote %d judged pairs to %s (from %d available)",
        len(sampled_pairs),
        output_path,
        total_available,
    )
    for key in sorted(strata):
        source, related, bucket = key
        logger.info(
            "  stratum source=%s related=%s confidence=%s: %d/%d",
            source,
            related,
            bucket.value,
            allocations[key],
            len(strata[key]),
        )


def _build_requests(
    pairs: list[PairToJudge],
    facets_by_id: dict[CorpusId, Facets],
    processed_keys: set[str],
) -> Iterator[tuple[str, llm.CompletionRequest]]:
    """Build LLM requests for unprocessed pairs.

    Args:
        pairs: Explicit paper pairs to judge.
        facets_by_id: Facets keyed by corpus ID.
        processed_keys: Set of already-processed pair keys (``"a|b"``).

    Yields:
        ``(key, CompletionRequest)`` tuples for unprocessed pairs.
    """
    for pair in pairs:
        key = f"{pair.paper_a}|{pair.paper_b}"
        if key in processed_keys:
            continue

        facets_a = facets_by_id.get(pair.paper_a)
        facets_b = facets_by_id.get(pair.paper_b)
        if facets_a is None or facets_b is None:
            continue

        user_prompt = _JUDGE_USER_TEMPLATE.format(
            paper_a=_format_facets(facets_a),
            paper_b=_format_facets(facets_b),
        )

        yield (
            key,
            llm.CompletionRequest(system=_JUDGE_SYSTEM_PROMPT, user=user_prompt),
        )


async def _run_llm_judge_for_pairs(
    pairs: list[PairToJudge],
    facets_path: Path,
    output_path: Path,
    concurrent: int,
    model: str,
    reasoning_effort: llm.ReasoningEffort,
    provider: str,
    azure_endpoint: str | None,
    api_version: str | None,
    seed: int,
) -> None:
    """Run LLM judging on an explicit list of pairs.

    Args:
        pairs: Explicit paper pairs to judge.
        facets_path: Path to facets JSONL file.
        output_path: Output JSONL path for judgements.
        concurrent: Number of concurrent LLM calls.
        model: LLM model to use.
        reasoning_effort: Reasoning effort for GPT-5 models.
        provider: LLM provider.
        azure_endpoint: Azure endpoint URL.
        api_version: Azure API version override.
        seed: Random seed recorded in metadata.
    """
    facets_by_id = load_facets_map(facets_path)
    pair_source = {f"{pair.paper_a}|{pair.paper_b}": pair.source for pair in pairs}

    progress_path = output_path.with_suffix(".progress.txt")
    processed_str_keys = load_processed_strings(progress_path)

    if processed_str_keys:
        logger.info("Resuming: %d pairs already judged", len(processed_str_keys))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode: OpenMode = "a" if processed_str_keys else "w"

    total_pairs = sum(
        1
        for pair in pairs
        if f"{pair.paper_a}|{pair.paper_b}" not in processed_str_keys
        and pair.paper_a in facets_by_id
        and pair.paper_b in facets_by_id
    )

    if total_pairs == 0:
        logger.info("All pairs already judged!")
        return

    logger.info("Pairs to judge: %d", total_pairs)
    logger.info("Model: %s, Provider: %s", model, provider)

    total_cost = 0.0
    failed_keys: list[str] = []

    async with (
        llm.LLMClient.create(
            provider,
            model=model,
            reasoning_effort=reasoning_effort,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        ) as client,
        AsyncLineWriter.open(output_path, mode=mode) as writer,
    ):
        cost_col = CallbackColumn(lambda: f"${total_cost:.4f}")
        async for key, llm_result in track_progress(
            client.batch_complete(
                _build_requests(pairs, facets_by_id, processed_str_keys),
                response_model=JudgementResponse,
                concurrent=concurrent,
            ),
            desc="Judging pairs...",
            total=total_pairs,
            extra_columns=[cost_col],
        ):
            total_cost += llm_result.cost
            if result := llm_result.result:
                paper_a_str, paper_b_str = key.split("|")
                source = pair_source.get(key, "unknown")

                judged = JudgedPair(
                    paper_a=CorpusId(int(paper_a_str)),
                    paper_b=CorpusId(int(paper_b_str)),
                    source=source,
                    related=result.related,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    relationship_type=result.relationship_type,
                )
                await writer.write_obj(judged, flush=True)
                _append_progress(progress_path, key)
            else:
                # The client returns None once retries are exhausted. Record the
                # pair rather than dropping it: it is deliberately NOT written to
                # the progress file, so a re-run retries it.
                failed_keys.append(key)

    _report_failures(failed_keys, total_pairs, output_path, pair_source)

    llm.write_run_metadata(
        output_path.with_name("judge_run.json"),
        model=llm.resolve_model(model),
        model_short=model,
        temperature=0.0,
        seed=seed,
        total_cost_usd=total_cost,
        reasoning_effort=reasoning_effort.value,
    )
    logger.info("Total cost: $%.4f", total_cost)
    logger.info("Judgements written to %s", output_path)


def _report_failures(
    failed_keys: list[str],
    total_pairs: int,
    output_path: Path,
    pair_source: dict[str, str],
) -> None:
    """Log a judged/failed summary and write failed pairs to a sidecar file.

    Failures are dropped from the judgements file by design, so without this the
    output silently contains fewer rows than the input. The sidecar lists exactly
    which pairs failed, and is written next to the judgements as
    ``<output>.failed.jsonl``. Failed pairs are absent from the progress file, so
    re-running the same command retries them.

    Args:
        failed_keys: ``"paper_a|paper_b"`` keys whose judgement returned None.
        total_pairs: Number of pairs attempted in this run.
        output_path: Judgements output path, used to derive the sidecar path.
        pair_source: Map from pair key to its origin label.
    """
    judged = total_pairs - len(failed_keys)
    if not failed_keys:
        logger.info("Judged %d/%d pairs.", judged, total_pairs)
        return

    sidecar = output_path.with_suffix(output_path.suffix + ".failed.jsonl")
    with open(sidecar, "w", encoding="utf-8") as fh:
        for key in failed_keys:
            paper_a_str, paper_b_str = key.split("|")
            fh.write(
                json.dumps(
                    {
                        "paper_a": int(paper_a_str),
                        "paper_b": int(paper_b_str),
                        "source": pair_source.get(key, "unknown"),
                    }
                )
                + "\n"
            )

    log = logger.error if judged == 0 else logger.warning
    log(
        "Judged %d/%d pairs; %d failed. Failed pairs written to %s. "
        "Re-run the same command to retry them.",
        judged,
        total_pairs,
        len(failed_keys),
        sidecar,
    )


async def run_llm_judge(
    facets_path: Path,
    retrieval_path: Path,
    output_path: Path,
    n_papers: int = 500,
    negatives_per_query: int = 5,
    concurrent: int = 50,
    model: str = "gpt-5-mini",
    reasoning_effort: llm.ReasoningEffort = llm.ReasoningEffort.MINIMAL,
    provider: str = "azure-openai",
    azure_endpoint: str | None = None,
    api_version: str | None = None,
    seed: int = 0,
) -> None:
    """Run LLM-as-judge evaluation on sampled paper pairs.

    Samples pairs from retrieval results, judges each pair with the LLM,
    and writes results to JSONL. Supports resume via progress tracking.

    Args:
        facets_path: Path to facets JSONL file.
        retrieval_path: Path to retrieval_sets JSONL file.
        output_path: Output JSONL path for judgements.
        n_papers: Number of query papers to sample.
        negatives_per_query: Random negative pairs per query.
        concurrent: Number of concurrent LLM calls.
        model: LLM model to use.
        reasoning_effort: Reasoning effort for GPT-5 models.
        provider: LLM provider (``"openai"`` or ``"azure-openai"``).
        azure_endpoint: Azure endpoint URL (for ``"azure-openai"``).
        api_version: Azure API version override.
        seed: Random seed for pair sampling.
    """
    sampled_pairs = sample_pairs(retrieval_path, n_papers, negatives_per_query, seed)
    pairs = [
        PairToJudge(paper_a=CorpusId(paper_a), paper_b=CorpusId(paper_b), source=source)
        for paper_a, paper_b, source in sampled_pairs
    ]
    await _run_llm_judge_for_pairs(
        pairs=pairs,
        facets_path=facets_path,
        output_path=output_path,
        concurrent=concurrent,
        model=model,
        reasoning_effort=reasoning_effort,
        provider=provider,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        seed=seed,
    )


async def run_llm_judge_pairs(
    facets_path: Path,
    pairs_path: Path,
    output_path: Path,
    concurrent: int = 50,
    model: str = "gpt-5-mini",
    reasoning_effort: llm.ReasoningEffort = llm.ReasoningEffort.MINIMAL,
    provider: str = "azure-openai",
    azure_endpoint: str | None = None,
    api_version: str | None = None,
    seed: int = 0,
) -> None:
    """Run LLM judging on an explicit JSONL list of pairs.

    Args:
        facets_path: Path to facets JSONL file.
        pairs_path: Path to explicit `PairToJudge` JSONL, or a subset of an
            existing `JudgedPair` file.
        output_path: Output JSONL path for judgements.
        concurrent: Number of concurrent LLM calls.
        model: LLM model to use.
        reasoning_effort: Reasoning effort for GPT-5 models.
        provider: LLM provider (``"openai"`` or ``"azure-openai"``).
        azure_endpoint: Azure endpoint URL (for ``"azure-openai"``).
        api_version: Azure API version override.
        seed: Random seed recorded in metadata.
    """
    pairs = load_pairs_to_judge(pairs_path)
    logger.info("Loaded %d explicit pairs from %s", len(pairs), pairs_path)
    await _run_llm_judge_for_pairs(
        pairs=pairs,
        facets_path=facets_path,
        output_path=output_path,
        concurrent=concurrent,
        model=model,
        reasoning_effort=reasoning_effort,
        provider=provider,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        seed=seed,
    )
