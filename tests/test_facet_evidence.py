"""Unit tests for facet_evaluation evidence fidelity primitives."""

from genret.processing.facet_evaluation import (
    EVIDENCE_FIDELITY_THRESHOLD,
    EVIDENCE_MIN_MATCH_BLOCK,
    best_paragraph_coverage,
    evaluate_evidence_fidelity,
)


class TestBestParagraphCoverage:
    """Tests for best_paragraph_coverage."""

    def test_verbatim_match(self) -> None:
        """Exact substring returns (True, 1.0)."""
        is_verb, cov = best_paragraph_coverage(
            "exact match", ["this is an exact match in context"]
        )
        assert is_verb is True
        assert cov == 1.0

    def test_no_match(self) -> None:
        """Completely unrelated text returns low coverage."""
        is_verb, cov = best_paragraph_coverage(
            "quantum entanglement", ["the cat sat on the mat"]
        )
        assert is_verb is False
        assert cov < 0.5

    def test_scattered_characters_not_verbatim(self) -> None:
        """Scattered matching characters should NOT count as verbatim.

        This is the regression case: 'abc' vs 'aXbXc' has all characters
        matching but not contiguously.
        """
        is_verb, cov = best_paragraph_coverage("abc", ["aXbXc"])
        assert is_verb is False
        assert cov < 1.0

    def test_close_but_not_verbatim(self) -> None:
        """A near-copy with small differences gets high but not 1.0 coverage."""
        evidence = "the model achieves strong performance"
        paragraph = "the model achieves very strong performance"
        is_verb, cov = best_paragraph_coverage(evidence, [paragraph])
        assert is_verb is False
        # Both sides of the insertion are substantial blocks, so coverage is high
        assert cov > 0.5

    def test_min_match_block_is_configurable(self) -> None:
        """Relaxing min block length increases coverage for split matches."""
        evidence = "abcdefghij"
        paragraph = "abcdXefghij"
        _, default_cov = best_paragraph_coverage(evidence, [paragraph])
        _, relaxed_cov = best_paragraph_coverage(
            evidence,
            [paragraph],
            min_match_block=4,
        )
        assert default_cov == 0.0
        assert relaxed_cov == 1.0

    def test_single_word_swap_high_coverage(self) -> None:
        """A single word swapped in a long sentence should still score high.

        Two large blocks on either side of the swap should both count,
        giving coverage close to 1.0 rather than ~0.5.
        """
        evidence = "the transformer model achieves state-of-the-art results on all benchmark datasets evaluated"
        paragraph = "the transformer model achieves excellent results on all benchmark datasets evaluated"
        is_verb, cov = best_paragraph_coverage(evidence, [paragraph])
        assert is_verb is False
        # ~80% of the evidence is covered by two large matching blocks
        assert cov > 0.7

    def test_empty_evidence(self) -> None:
        """Empty evidence returns (False, 0.0)."""
        is_verb, cov = best_paragraph_coverage("", ["some text"])
        assert is_verb is False
        assert cov == 0.0

    def test_empty_paragraphs(self) -> None:
        """No paragraphs returns (False, 0.0)."""
        is_verb, cov = best_paragraph_coverage("some evidence", [])
        assert is_verb is False
        assert cov == 0.0

    def test_best_across_multiple_paragraphs(self) -> None:
        """Returns the best coverage across multiple paragraphs."""
        evidence = "transformer attention mechanism"
        paragraphs = [
            "unrelated paragraph about biology",
            "the transformer attention mechanism is widely used",
            "another unrelated paragraph",
        ]
        is_verb, cov = best_paragraph_coverage(evidence, paragraphs)
        assert is_verb is True
        assert cov == 1.0


class TestEvaluateEvidenceFidelity:
    """Tests for evaluate_evidence_fidelity."""

    def test_empty_evidence(self) -> None:
        """No evidence strings returns zero result."""
        result = evaluate_evidence_fidelity([], ["some paragraph"])
        assert result.verbatim_count == 0
        assert result.faithful_count == 0
        assert result.mean_fidelity == 0.0

    def test_default_min_match_block_is_8(self) -> None:
        """Default minimum SequenceMatcher block size is 8 characters."""
        assert EVIDENCE_MIN_MATCH_BLOCK == 8

    def test_all_verbatim(self) -> None:
        """All evidence strings are exact substrings."""
        paragraphs = ["The model uses attention. It achieves high accuracy."]
        evidence = ["The model uses attention", "achieves high accuracy"]
        result = evaluate_evidence_fidelity(evidence, paragraphs)
        assert result.verbatim_count == 2
        assert result.faithful_count == 2
        assert result.mean_fidelity == 1.0

    def test_none_verbatim(self) -> None:
        """No evidence strings match verbatim."""
        paragraphs = ["The cat sat on the mat."]
        evidence = ["quantum physics is fascinating"]
        result = evaluate_evidence_fidelity(evidence, paragraphs)
        assert result.verbatim_count == 0
        assert result.mean_fidelity < 0.5

    def test_mixed_verbatim_and_fuzzy(self) -> None:
        """Mix of verbatim and non-verbatim evidence."""
        paragraphs = ["The model achieves 95% accuracy on the benchmark."]
        evidence = [
            "The model achieves 95% accuracy",  # verbatim
            "model gets 95% on benchmark",  # not verbatim
        ]
        result = evaluate_evidence_fidelity(evidence, paragraphs)
        assert result.verbatim_count == 1
        assert 0.5 < result.mean_fidelity < 1.0

    def test_whitespace_normalisation(self) -> None:
        """Whitespace differences don't prevent verbatim match."""
        paragraphs = ["The   model\nachieves   accuracy"]
        evidence = ["The model achieves accuracy"]
        result = evaluate_evidence_fidelity(evidence, paragraphs)
        assert result.verbatim_count == 1
        assert result.faithful_count == 1
        assert result.mean_fidelity == 1.0

    def test_scattered_matches_not_verbatim(self) -> None:
        """Evidence with scattered matching chars is not counted as verbatim."""
        paragraphs = ["aXbXcXdXeXf"]
        evidence = ["abcdef"]
        result = evaluate_evidence_fidelity(evidence, paragraphs)
        assert result.verbatim_count == 0
        assert result.mean_fidelity < 1.0

    def test_faithful_but_not_verbatim(self) -> None:
        """A near-copy passes the faithful threshold but not verbatim."""
        # Evidence with one word swapped — high coverage from two large blocks
        paragraphs = [
            "the transformer model achieves state-of-the-art results on all "
            "benchmark datasets evaluated in the study"
        ]
        evidence = [
            "the transformer model achieves excellent results on all "
            "benchmark datasets evaluated in the study"
        ]
        result = evaluate_evidence_fidelity(evidence, paragraphs)
        assert result.verbatim_count == 0
        assert result.faithful_count == 1
        assert result.mean_fidelity >= EVIDENCE_FIDELITY_THRESHOLD

    def test_min_match_block_changes_fidelity(self) -> None:
        """Smaller minimum block length counts more split matches as coverage."""
        paragraphs = ["abcdXefghij"]
        evidence = ["abcdefghij"]
        default_result = evaluate_evidence_fidelity(evidence, paragraphs)
        relaxed_result = evaluate_evidence_fidelity(
            evidence,
            paragraphs,
            min_match_block=4,
        )
        assert default_result.mean_fidelity == 0.0
        assert default_result.faithful_count == 0
        assert relaxed_result.mean_fidelity == 1.0
        assert relaxed_result.faithful_count == 1
