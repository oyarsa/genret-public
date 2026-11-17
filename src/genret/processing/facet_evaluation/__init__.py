"""Facet quality evaluation for extracted paper facets.

Evaluates Step 1 Phase 2 output (facets.jsonl) against source paper text.
Produces per-paper quality scores and corpus-level summary statistics.
Read-only: does not modify extraction outputs.

This package is split into focused submodules:

``constants``
    Facet names, score weights, thresholds, and compiled regex patterns.
``models``
    Pydantic records (``PaperEvaluation``, ``EvaluationSummary``, etc.).
``text_utils``
    Paragraph extraction, sentence counting, whitespace normalisation.
``evidence``
    Evidence fidelity scoring against paper text.
``scoring``
    Per-dimension scoring: structure, grounding, redundancy, specificity.
``embeddings``
    On-disk embedding cache and cache-backed embedding computation.
``paper``
    Per-paper orchestration (``evaluate_paper``).
``summary``
    Corpus-level aggregation (``compute_summary``).

All public names are re-exported here, so existing
``from genret.processing.facet_evaluation import X`` imports keep working.

Output
------
Two files are written to the output directory:

``facets_evaluation.jsonl``
    One JSON object per paper (``PaperEvaluation``), containing paper-level
    scores, flags, and per-item metrics.  Written via a ``.tmp`` intermediate
    file and renamed on success so a partial run never corrupts the output.

``facets_evaluation_summary.json``
    A single ``EvaluationSummary`` object aggregating score distributions,
    facet coverage rates, mean item counts, and the most common flags across
    all evaluated papers.

Paper-Level Scoring
-------------------
Each paper receives four component scores (0-1) combined into an overall
score via weighted average.

**Structure** (weight 0.15)
    Checks that facets are well-formed: non-empty, within length bounds, no
    exact duplicates, and no excessive multi-sentence items.  Also penalises
    meta-descriptions — items phrased as "The paper proposes X" rather than
    stating X directly.  A paper with all direct, well-formatted items scores
    1.0; missing facets, duplicates, and high meta-description rates reduce
    the score.

**Grounding** (weight 0.45)
    Measures how well each facet item is supported by the paper text.  Each
    item is embedded (OpenAI text-embedding-3-small) and compared against
    paragraph embeddings via cosine similarity.  The best-matching paragraph's
    raw similarity is remapped from [0.50, 0.80] to [0, 1] so the score
    spans the full range.  Per-facet scores are averaged, and facets below a
    low-grounding threshold are flagged.

**Redundancy** (weight 0.20)
    Detects near-duplicate items within a facet (threshold 0.92) and
    semantically overlapping items across facets (threshold 0.90) using
    pairwise cosine similarity of item embeddings.  The penalty scales with
    the severity (how far above the threshold) and count of violations,
    normalised by total item count.  Papers with no near-duplicates score
    1.0.

**Specificity** (weight 0.20)
    Scores how concrete and informative each item is, as opposed to vague
    boilerplate.  Three signals are combined per item: a meta-description
    penalty (items starting with "The paper...", "The authors...", etc.), a
    generic-pattern penalty (stock academic phrases like "state-of-the-art",
    "novel approach"), and a specificity discount for numbers, acronyms, and
    proper nouns.  The paper score is the mean of (1 - genericness) across
    all items.

Per-Item Metrics
----------------
Each facet item receives an ``ItemEvaluation`` record with:

``grounding_score``
    Raw cosine similarity between the item embedding and the best-matching
    source paragraph.  Unlike the paper-level grounding score, this is
    **not** remapped — it stays in the original OpenAI similarity range
    (typically 0.4-0.9).

``top_evidence``
    The top-k (default 3) most similar source paragraphs, each an
    ``EvidenceSnippet`` with section heading, paragraph text (clipped to
    200 characters), and raw cosine similarity.  Allows tracing each item
    back to its supporting text in the paper.

``genericness_score``
    How vague the item is (0 = specific, 1 = completely generic), combining
    meta-description penalty, generic-pattern matches, and specific-indicator
    discounts (numbers, acronyms, proper nouns).

``predicted_facet_type``
    Keyword-based prediction of which facet category (problems, methods,
    results, contributions) the item best fits.  Informational only — does
    not affect scores.  ``None`` if no strong signal.

``is_near_duplicate``
    Whether this item was flagged as a near-duplicate during redundancy
    evaluation (intra-facet similarity > 0.92).

Corpus-Level Summary
--------------------
``EvaluationSummary`` aggregates across all evaluated papers:

- **Score distributions**: mean, median, std, p10, p90, min, max for each
  of the five scores (structure, grounding, redundancy, specificity, overall).
- **Facet coverage**: fraction of papers with at least one item per facet.
- **Item count means**: average number of items per facet category.
- **Top flags**: the 20 most common flags and their counts.
- **Skip reasons**: why papers were skipped (e.g. missing parsed data).

Embedding Cache
---------------
Paragraph and facet item embeddings are cached in single ``.npz`` files
(``paragraph_embeddings.npz``, ``facet_embeddings.npz``) using the
``EmbeddingCache`` class, keyed by a SHA-256 hash of the input texts.
Hashing by content means that if the texts change (e.g. after re-running
extraction with a different prompt), the cache automatically misses and
recomputes — no manual invalidation required.  The cache loads into memory
once and flushes new entries on ``save()``.  This allows re-running
evaluation without re-computing embeddings, and supports offline scoring
when the OpenAI API is unavailable.
"""

from .constants import (
    CROSS_OVERLAP_THRESHOLD,
    EVIDENCE_CLIP_CHARS,
    EVIDENCE_FIDELITY_THRESHOLD,
    EVIDENCE_MIN_MATCH_BLOCK,
    FACET_NAMES,
    FACET_TYPE_KEYWORDS,
    GENERIC_PATTERNS,
    GROUNDING_CEILING,
    GROUNDING_FLOOR,
    INTRA_DUP_THRESHOLD,
    LOW_GROUNDING_THRESHOLD,
    MAX_ITEM_CHARS,
    MAX_ITEMS_PER_FACET,
    MAX_SENTENCES_PER_ITEM,
    META_DESCRIPTION_PATTERNS,
    MIN_ITEM_CHARS,
    MIN_ITEMS_PER_FACET,
    SCORE_WEIGHTS,
)
from .embeddings import EmbeddingCache, get_cached_embeddings
from .evidence import (
    EvidenceFidelityResult,
    best_paragraph_coverage,
    evaluate_evidence_fidelity,
)
from .models import (
    EvaluationSummary,
    EvidenceSnippet,
    ItemEvaluation,
    PaperEvaluation,
    PaperScores,
    ScoreDistribution,
)
from .paper import evaluate_paper
from .scoring import (
    GroundingResult,
    RedundancyResult,
    compute_item_genericness,
    compute_item_grounding,
    compute_overall_score,
    evaluate_grounding,
    evaluate_redundancy,
    evaluate_specificity,
    evaluate_structure,
    is_meta_description,
    predict_facet_type,
    remap_grounding,
)
from .summary import compute_summary, distribution
from .text_utils import count_sentences, extract_paragraphs, normalise_ws

__all__ = [
    "CROSS_OVERLAP_THRESHOLD",
    "EVIDENCE_CLIP_CHARS",
    "EVIDENCE_FIDELITY_THRESHOLD",
    "EVIDENCE_MIN_MATCH_BLOCK",
    "FACET_NAMES",
    "FACET_TYPE_KEYWORDS",
    "GENERIC_PATTERNS",
    "GROUNDING_CEILING",
    "GROUNDING_FLOOR",
    "INTRA_DUP_THRESHOLD",
    "LOW_GROUNDING_THRESHOLD",
    "MAX_ITEMS_PER_FACET",
    "MAX_ITEM_CHARS",
    "MAX_SENTENCES_PER_ITEM",
    "META_DESCRIPTION_PATTERNS",
    "MIN_ITEMS_PER_FACET",
    "MIN_ITEM_CHARS",
    "SCORE_WEIGHTS",
    "EmbeddingCache",
    "EvaluationSummary",
    "EvidenceFidelityResult",
    "EvidenceSnippet",
    "GroundingResult",
    "ItemEvaluation",
    "PaperEvaluation",
    "PaperScores",
    "RedundancyResult",
    "ScoreDistribution",
    "best_paragraph_coverage",
    "compute_item_genericness",
    "compute_item_grounding",
    "compute_overall_score",
    "compute_summary",
    "count_sentences",
    "distribution",
    "evaluate_evidence_fidelity",
    "evaluate_grounding",
    "evaluate_paper",
    "evaluate_redundancy",
    "evaluate_specificity",
    "evaluate_structure",
    "extract_paragraphs",
    "get_cached_embeddings",
    "is_meta_description",
    "normalise_ws",
    "predict_facet_type",
    "remap_grounding",
]
