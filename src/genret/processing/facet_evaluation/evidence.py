"""Evidence fidelity: how faithfully model evidence strings reproduce paper text."""

import dataclasses
import difflib
import statistics

from .constants import EVIDENCE_FIDELITY_THRESHOLD, EVIDENCE_MIN_MATCH_BLOCK
from .text_utils import normalise_ws


def best_paragraph_coverage(
    normed_evidence: str,
    normed_paragraphs: list[str],
    *,
    min_match_block: int = EVIDENCE_MIN_MATCH_BLOCK,
) -> tuple[bool, float]:
    """Compute the best character-level coverage ratio across paragraphs.

    Checks for a verbatim substring match first (after whitespace
    normalisation).  When no verbatim match exists, uses
    ``difflib.SequenceMatcher`` to sum matching blocks that are at least
    ``min_match_block`` characters long.  The threshold filters out
    trivial coincidental overlaps (short common words, punctuation) while
    still crediting multiple substantial matching regions.

    Args:
        normed_evidence: Whitespace-normalised evidence string.
        normed_paragraphs: Pre-normalised paragraph texts.
        min_match_block: Minimum matching block size counted towards coverage.
            Must be >= 1.

    Returns:
        (is_verbatim, coverage) where is_verbatim is True when the evidence
        appears as a contiguous substring in some paragraph, and coverage is
        the fraction of evidence characters covered by non-trivial matching
        blocks (1.0 when verbatim).
    """
    if min_match_block < 1:
        msg = f"min_match_block must be >= 1, got {min_match_block}"
        raise ValueError(msg)

    if not normed_evidence:
        return False, 0.0

    best = 0.0
    for normed_paragraph in normed_paragraphs:
        if normed_evidence in normed_paragraph:
            return True, 1.0

        matcher = difflib.SequenceMatcher(
            None, normed_evidence, normed_paragraph, autojunk=False
        )
        matched_chars = sum(
            block.size
            for block in matcher.get_matching_blocks()
            if block.size >= min_match_block
        )
        coverage = min(matched_chars / len(normed_evidence), 1.0)
        best = max(best, coverage)

    return False, best


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvidenceFidelityResult:
    """Aggregated evidence fidelity metrics for one facet item."""

    verbatim_count: int
    """Number of evidence strings that matched as exact substrings."""
    faithful_count: int
    """Number of evidence strings with coverage >= EVIDENCE_FIDELITY_THRESHOLD."""
    mean_fidelity: float
    """Mean coverage ratio across all evidence strings (1.0 = all verbatim)."""


_EMPTY_FIDELITY = EvidenceFidelityResult(
    verbatim_count=0, faithful_count=0, mean_fidelity=0.0
)
"""Sentinel for items with no model evidence."""


def evaluate_evidence_fidelity(
    model_evidence: list[str],
    para_texts: list[str],
    *,
    min_match_block: int = EVIDENCE_MIN_MATCH_BLOCK,
) -> EvidenceFidelityResult:
    """Evaluate how faithfully evidence strings reproduce paper text.

    For one facet item's evidence strings, checks each against all paper
    paragraphs for verbatim substring matches (after whitespace normalisation)
    and computes contiguous character-level coverage ratios.

    Paragraph texts are normalised once and reused across all evidence strings
    to avoid redundant work.

    Args:
        model_evidence: Evidence strings produced by the model for one item.
        para_texts: All paragraph texts from the paper.
        min_match_block: Minimum matching block size used when computing
            non-verbatim coverage. Must be >= 1.

    Returns:
        An ``EvidenceFidelityResult`` with verbatim count, faithful count
        (coverage >= ``EVIDENCE_FIDELITY_THRESHOLD``), and mean coverage.
        Returns zero counts and 0.0 fidelity when there are no evidence
        strings.
    """
    if not model_evidence:
        return _EMPTY_FIDELITY

    normed_paras = [normalise_ws(p) for p in para_texts]

    verbatim = 0
    faithful = 0
    coverages: list[float] = []

    for ev in model_evidence:
        normed_ev = normalise_ws(ev)
        is_verbatim, coverage = best_paragraph_coverage(
            normed_ev,
            normed_paras,
            min_match_block=min_match_block,
        )
        coverages.append(coverage)
        verbatim += is_verbatim
        if coverage >= EVIDENCE_FIDELITY_THRESHOLD:
            faithful += 1

    return EvidenceFidelityResult(
        verbatim_count=verbatim,
        faithful_count=faithful,
        mean_fidelity=statistics.mean(coverages),
    )
