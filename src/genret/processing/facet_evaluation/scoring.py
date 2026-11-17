"""Per-dimension scoring: structure, grounding, redundancy, specificity, prediction."""

import dataclasses
import re
import statistics
from collections.abc import Sequence

import numpy as np

from genret.llm.encoder import Matrix, Vector
from genret.paper_parsing import FacetsSummary

from .constants import (
    _COMPILED_GENERIC,
    _COMPILED_META,
    CROSS_OVERLAP_THRESHOLD,
    FACET_NAMES,
    FACET_TYPE_KEYWORDS,
    GROUNDING_CEILING,
    GROUNDING_FLOOR,
    INTRA_DUP_THRESHOLD,
    MAX_ITEM_CHARS,
    MAX_ITEMS_PER_FACET,
    MAX_SENTENCES_PER_ITEM,
    MIN_ITEM_CHARS,
    SCORE_WEIGHTS,
)
from .models import EvidenceSnippet
from .text_utils import _clip_text, count_sentences


@dataclasses.dataclass(frozen=True, kw_only=True)
class GroundingResult:
    """Result of grounding evaluation for a paper's facets."""

    score: float
    """Overall grounding score (mean of per-facet scores)."""
    per_facet_scores: dict[str, float]
    """Per-facet grounding scores keyed by facet name."""
    per_facet_evidence: dict[str, list[list[EvidenceSnippet]]]
    """Per-facet top-k evidence snippets for each item."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class RedundancyResult:
    """Result of redundancy evaluation for a paper's facets."""

    score: float
    """Overall redundancy score (1.0 = no redundancy)."""
    flags: list[str]
    """Redundancy-related flags (e.g. ``intra_facet_duplicate``)."""
    near_duplicate_indices: set[int]
    """Global item indices flagged as near-duplicates."""


# ── Structural Evaluation ────────────────────────────────────────────────────


def evaluate_structure(facets: FacetsSummary) -> tuple[float, list[str]]:
    """Evaluate structural quality of facets.

    Checks formatting, length, duplicates, and meta-description prevalence.
    Returns (score, flags) where score is in [0, 1].
    """
    score = 1.0
    flags: list[str] = []
    total_items = 0
    total_meta = 0

    all_facets = facets.facet_items()
    for name in FACET_NAMES:
        items: Sequence[str] = all_facets[name]
        if len(items) == 0:
            score -= 0.20
            flags.append(f"{name}_empty")
            continue
        if len(items) > MAX_ITEMS_PER_FACET:
            score -= 0.05
            flags.append(f"{name}_too_many_items")

        meta_count = 0
        # Check individual items
        for item in items:
            stripped = item.strip()
            total_items += 1
            if not stripped:
                score -= 0.05
                if "item_empty" not in flags:
                    flags.append("item_empty")
            elif len(stripped) < MIN_ITEM_CHARS:
                score -= 0.02
                if "item_too_short" not in flags:
                    flags.append("item_too_short")
            elif len(stripped) > MAX_ITEM_CHARS:
                score -= 0.02
                if "item_too_long" not in flags:
                    flags.append("item_too_long")
            if count_sentences(stripped) > MAX_SENTENCES_PER_ITEM:
                score -= 0.01
                if "item_too_many_sentences" not in flags:
                    flags.append("item_too_many_sentences")
            if is_meta_description(stripped):
                meta_count += 1
                total_meta += 1

        # Exact duplicates within facet
        if len(items) != len(set(items)):
            score -= 0.10
            flags.append(f"{name}_exact_duplicate")

        # Per-facet meta-description prevalence
        if meta_count == len(items) and len(items) >= 2:
            score -= 0.03
            flags.append(f"{name}_all_meta_descriptions")

    # Global meta-description rate: penalise if majority of items are
    # meta-descriptions ("The paper addresses...") rather than direct statements.
    if total_items > 0:
        meta_ratio = total_meta / total_items
        if meta_ratio > 0.5:
            score -= 0.10 * meta_ratio
            if "high_meta_description_rate" not in flags:
                flags.append("high_meta_description_rate")

    return max(0.0, score), flags


# ── Grounding Evaluation ─────────────────────────────────────────────────────


def remap_grounding(raw: float) -> float:
    """Remap raw cosine similarity to a 0-1 scale for interpretability.

    OpenAI embeddings yield cosine similarities mostly in [0.4, 0.9] for related
    text. This linear remap spreads the useful range so scores are more
    discriminating.
    """
    return max(
        0.0, min(1.0, (raw - GROUNDING_FLOOR) / (GROUNDING_CEILING - GROUNDING_FLOOR))
    )


def compute_item_grounding(
    item_embedding: Vector,
    paragraph_embeddings: Matrix,
    paragraphs: list[tuple[str, str]],
    top_k: int = 3,
) -> tuple[float, list[EvidenceSnippet]]:
    """Compute grounding score for a single facet item.

    Returns (remapped_score, top_evidence_snippets).
    OpenAI embeddings are L2-normalised, so cosine similarity = dot product.
    The raw similarity is remapped through `remap_grounding` for
    interpretability.
    """
    similarities: Vector = paragraph_embeddings @ item_embedding
    top_indices = np.argsort(similarities)[::-1][:top_k]

    evidence = [
        EvidenceSnippet(
            section_heading=paragraphs[int(i)][0],
            text=_clip_text(paragraphs[int(i)][1]),
            similarity=float(similarities[i]),
        )
        for i in top_indices
    ]

    raw_score = float(similarities[top_indices[0]])
    return remap_grounding(raw_score), evidence


def evaluate_grounding(
    facet_items: dict[str, list[str]],
    facet_embeddings: Matrix,
    paragraph_embeddings: Matrix,
    paragraphs: list[tuple[str, str]],
    top_k: int = 3,
) -> GroundingResult:
    """Evaluate how well facets are grounded in the paper text.

    Returns (overall_score, per_facet_scores, per_facet_evidence).
    """
    idx = 0
    per_facet_scores: dict[str, float] = {}
    per_facet_evidence: dict[str, list[list[EvidenceSnippet]]] = {}

    for name in FACET_NAMES:
        items = facet_items.get(name, [])
        item_scores: list[float] = []
        item_evidence: list[list[EvidenceSnippet]] = []
        for _ in items:
            gs, ev = compute_item_grounding(
                facet_embeddings[idx], paragraph_embeddings, paragraphs, top_k
            )
            item_scores.append(gs)
            item_evidence.append(ev)
            idx += 1
        per_facet_scores[name] = statistics.mean(item_scores) if item_scores else 0.0
        per_facet_evidence[name] = item_evidence

    all_scores = [s for name in FACET_NAMES for s in [per_facet_scores[name]] if s > 0]
    overall = statistics.mean(all_scores) if all_scores else 0.0
    return GroundingResult(
        score=overall,
        per_facet_scores=per_facet_scores,
        per_facet_evidence=per_facet_evidence,
    )


# ── Redundancy Evaluation ────────────────────────────────────────────────────


def evaluate_redundancy(
    facet_items: dict[str, list[str]],
    all_embeddings: Matrix,
) -> RedundancyResult:
    """Evaluate redundancy within and across facets.

    Returns (score, flags, near_duplicate_indices).
    """
    n = all_embeddings.shape[0]
    if n < 2:
        return RedundancyResult(score=1.0, flags=[], near_duplicate_indices=set())

    sim_matrix: Matrix = all_embeddings @ all_embeddings.T

    # Build index ranges per facet
    ranges: dict[str, tuple[int, int]] = {}
    offset = 0
    for name in FACET_NAMES:
        count = len(facet_items.get(name, []))
        ranges[name] = (offset, offset + count)
        offset += count

    flags: list[str] = []
    near_dup_indices: set[int] = set()
    intra_severity: list[float] = []
    cross_severity: list[float] = []

    # Intra-facet duplicates
    for name in FACET_NAMES:
        start, end = ranges[name]
        for i in range(start, end):
            for j in range(i + 1, end):
                sim = float(sim_matrix[i, j])
                if sim > INTRA_DUP_THRESHOLD:
                    intra_severity.append(sim - INTRA_DUP_THRESHOLD)
                    near_dup_indices.update((i, j))

    # Cross-facet overlap
    facet_names_list = list(FACET_NAMES)
    for a_idx in range(len(facet_names_list)):
        for b_idx in range(a_idx + 1, len(facet_names_list)):
            a_start, a_end = ranges[facet_names_list[a_idx]]
            b_start, b_end = ranges[facet_names_list[b_idx]]
            for i in range(a_start, a_end):
                for j in range(b_start, b_end):
                    sim = float(sim_matrix[i, j])
                    if sim > CROSS_OVERLAP_THRESHOLD:
                        cross_severity.append(sim - CROSS_OVERLAP_THRESHOLD)

    if intra_severity:
        flags.append("intra_facet_duplicate")
    if cross_severity:
        flags.append("cross_facet_overlap_high")

    # Penalty scales with both count and severity of violations,
    # normalised by total items so papers with more facets aren't
    # penalised disproportionately.
    intra_penalty = sum(s * 3.0 for s in intra_severity) / n
    cross_penalty = sum(s * 2.0 for s in cross_severity) / n
    score = max(0.0, 1.0 - intra_penalty - cross_penalty)
    return RedundancyResult(
        score=score, flags=flags, near_duplicate_indices=near_dup_indices
    )


# ── Specificity Evaluation ───────────────────────────────────────────────────


def is_meta_description(text: str) -> bool:
    """Check if text is a meta-description of the paper rather than a direct statement."""
    return any(p.search(text) for p in _COMPILED_META)


def compute_item_genericness(text: str) -> float:
    """Score 0-1 where 0 = specific, 1 = completely generic.

    Three signals are combined:
    1. Meta-description: starts with "The paper...", "The authors...", etc.
    2. Generic pattern matches: boilerplate academic phrases.
    3. Specific indicators: numbers, acronyms, proper nouns (discounts genericness).
    """
    words = text.split()
    if not words:
        return 1.0

    # Signal 1: meta-description prefix (0.3 penalty)
    meta = 0.3 if is_meta_description(text) else 0.0

    # Signal 2: generic pattern matches (0-0.5, scaled by match count)
    generic_matches = sum(1 for p in _COMPILED_GENERIC if p.search(text))
    pattern_score = min(0.5, generic_matches * 0.2)

    # Signal 3: specific indicators (discount up to 0.3)
    numbers = len(re.findall(r"\d+\.?\d*%?", text))
    proper_nouns = len(re.findall(r"(?<=[a-z] )[A-Z][a-zA-Z]+", text))
    acronyms = len(re.findall(r"\b[A-Z]{2,}\d*\b", text))
    technical = len(re.findall(r"\b\w+-\w+(?:-\w+)*\b", text))
    specific_count = numbers + proper_nouns + acronyms + technical
    specific_discount = min(0.3, specific_count * 0.06)

    return max(0.0, min(1.0, meta + pattern_score - specific_discount))


def evaluate_specificity(facet_items: dict[str, list[str]]) -> tuple[float, list[str]]:
    """Evaluate specificity of facets (1 = specific, 0 = generic).

    Returns (score, flags).
    """
    all_scores: list[float] = []
    flags: list[str] = []

    for name in FACET_NAMES:
        for item in facet_items.get(name, []):
            genericness = compute_item_genericness(item)
            all_scores.append(1.0 - genericness)

    if not all_scores:
        return 0.0, ["no_facet_items"]

    score = statistics.mean(all_scores)
    if score < 0.3:
        flags.append("high_genericness")

    return score, flags


# ── Facet-Type Prediction (Informational Only) ───────────────────────────────


def predict_facet_type(text: str) -> str | None:
    """Predict the most likely facet type for an item based on keywords.

    Returns the predicted type name or None if no strong signal.
    This is informational only and does not affect scores.
    """
    scores: dict[str, int] = {}
    for facet_type, patterns in FACET_TYPE_KEYWORDS.items():
        scores[facet_type] = sum(1 for p in patterns if p.search(text))

    if not scores or max(scores.values()) == 0:
        return None

    best = max(scores, key=lambda k: scores[k])
    # Only predict if there's a clear winner (>= 2 matches or single match with 0 others)
    if scores[best] < 1:
        return None
    runner_up = sorted(scores.values(), reverse=True)
    if len(runner_up) > 1 and scores[best] == runner_up[1] and scores[best] <= 1:
        return None
    return best


# ── Score Combination ────────────────────────────────────────────────────────


def compute_overall_score(components: dict[str, float]) -> float:
    """Weighted combination of component scores."""
    return sum(components[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)
