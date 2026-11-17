"""Unit tests for facet_evaluation per-dimension scoring primitives."""

import numpy as np
import numpy.typing as npt

from conftest import approx
from genret.paper_parsing import FacetsSummary
from genret.processing.facet_evaluation import (
    FACET_NAMES,
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

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_facets(**kwargs: list[str]) -> FacetsSummary:
    """Build a FacetsSummary with defaults that are direct statements (not meta)."""
    return FacetsSummary(
        problems=kwargs.get(
            "problems", ["Labeled data is scarce in low-resource NER."]
        ),
        methods=kwargs.get(
            "methods", ["Fine-tune RoBERTa-base with 3e-5 learning rate."]
        ),
        results=kwargs.get(
            "results", ["F1 improved from 82.1% to 89.7% on CoNLL-2003."]
        ),
        contributions=kwargs.get(
            "contributions", ["NER-Bench, a 50k-sample evaluation suite."]
        ),
    )


def _normalized(v: list[float]) -> npt.NDArray[np.float32]:
    """Create an L2-normalized float32 vector."""
    arr = np.array(v, dtype=np.float32)
    norm = np.float32(np.linalg.norm(arr))
    return arr / norm


class TestIsMetaDescription:
    """Tests for is_meta_description function."""

    def test_the_paper_addresses(self) -> None:
        """Should detect 'The paper addresses...' as meta-description."""
        assert is_meta_description(
            "The paper addresses the challenge of low-resource NER."
        )

    def test_it_investigates(self) -> None:
        """Should detect 'It investigates...' as meta-description."""
        assert is_meta_description(
            "It investigates the impact of data augmentation on accuracy."
        )

    def test_the_authors_propose(self) -> None:
        """Should detect 'The authors propose...' as meta-description."""
        assert is_meta_description("The authors propose a novel attention mechanism.")

    def test_this_work(self) -> None:
        """Should detect 'This work...' as meta-description."""
        assert is_meta_description(
            "This work introduces a new benchmark for evaluation."
        )

    def test_direct_statement(self) -> None:
        """Should not flag direct factual statements."""
        assert not is_meta_description("BERT achieves 95.3% F1 on SQuAD-2.0.")

    def test_method_statement(self) -> None:
        """Should not flag statements that don't start with meta prefixes."""
        assert not is_meta_description("Fine-tuning RoBERTa with learning rate 3e-5.")


class TestEvaluateStructure:
    """Tests for evaluate_structure function."""

    def test_perfect_facets(self) -> None:
        """Should return 1.0 for well-formed direct-statement facets."""
        facets = _make_facets()
        score, flags = evaluate_structure(facets)
        assert score == 1.0
        assert flags == []

    def test_empty_facet_penalized(self) -> None:
        """Should penalize and flag empty facet lists."""
        facets = _make_facets(problems=[])
        score, flags = evaluate_structure(facets)
        assert score < 1.0
        assert "problems_empty" in flags

    def test_all_empty_heavily_penalized(self) -> None:
        """Should heavily penalize when all four facets are empty."""
        facets = FacetsSummary(problems=[], methods=[], results=[], contributions=[])
        score, flags = evaluate_structure(facets)
        assert score == approx(0.2)
        assert len(flags) == 4

    def test_too_many_items(self) -> None:
        """Should flag facets with more than 5 items."""
        facets = _make_facets(
            problems=["Problem statement " + str(i) for i in range(7)]
        )
        _score, flags = evaluate_structure(facets)
        assert "problems_too_many_items" in flags

    def test_too_short_item(self) -> None:
        """Should flag items shorter than minimum character count."""
        facets = _make_facets(problems=["Short"])
        _score, flags = evaluate_structure(facets)
        assert "item_too_short" in flags

    def test_too_long_item(self) -> None:
        """Should flag items longer than maximum character count."""
        facets = _make_facets(problems=["x" * 501])
        _score, flags = evaluate_structure(facets)
        assert "item_too_long" in flags

    def test_empty_item(self) -> None:
        """Should flag empty string items."""
        facets = _make_facets(problems=["", "A valid problem statement."])
        _score, flags = evaluate_structure(facets)
        assert "item_empty" in flags

    def test_exact_duplicates(self) -> None:
        """Should flag exact duplicate items within a facet."""
        facets = _make_facets(problems=["Same problem here.", "Same problem here."])
        _score, flags = evaluate_structure(facets)
        assert "problems_exact_duplicate" in flags

    def test_score_clamped_to_zero(self) -> None:
        """Should never return a score below 0."""
        facets = FacetsSummary(
            problems=["", "", "", "", "", ""],
            methods=[],
            results=[],
            contributions=[],
        )
        score, _ = evaluate_structure(facets)
        assert score >= 0.0

    def test_meta_description_penalized(self) -> None:
        """Should penalize facets dominated by meta-descriptions."""
        facets = FacetsSummary(
            problems=[
                "The paper addresses the challenge of parsing.",
                "It investigates how models handle ambiguity.",
            ],
            methods=[
                "The authors propose a new neural architecture.",
                "It employs attention mechanisms for encoding.",
            ],
            results=[
                "The results show significant improvement.",
                "It demonstrates state-of-the-art performance.",
            ],
            contributions=[
                "This work introduces a novel framework.",
                "It provides insights into model behavior.",
            ],
        )
        score, flags = evaluate_structure(facets)
        assert score < 1.0
        assert "high_meta_description_rate" in flags

    def test_direct_statements_no_meta_penalty(self) -> None:
        """Should not penalize direct statements without meta prefixes."""
        facets = FacetsSummary(
            problems=["Labeled data is scarce.", "Models overfit on small datasets."],
            methods=["A contrastive learning approach with augmented pairs."],
            results=["Accuracy reached 94.2% on the test set."],
            contributions=["A new benchmark with 100k samples."],
        )
        score, flags = evaluate_structure(facets)
        assert "high_meta_description_rate" not in flags
        assert score == 1.0


class TestRemapGrounding:
    """Tests for remap_grounding function."""

    def test_at_floor(self) -> None:
        """Should return 0.0 at the grounding floor."""
        assert remap_grounding(0.50) == approx(0.0)

    def test_below_floor(self) -> None:
        """Should clamp to 0.0 below the floor."""
        assert remap_grounding(0.3) == 0.0

    def test_at_ceiling(self) -> None:
        """Should return 1.0 at the grounding ceiling."""
        assert remap_grounding(0.80) == approx(1.0)

    def test_above_ceiling(self) -> None:
        """Should clamp to 1.0 above the ceiling."""
        assert remap_grounding(0.95) == 1.0

    def test_midpoint(self) -> None:
        """Should return 0.5 at the midpoint between floor and ceiling."""
        mid = (0.50 + 0.80) / 2  # 0.65
        assert remap_grounding(mid) == approx(0.5)


class TestComputeItemGrounding:
    """Tests for compute_item_grounding with synthetic embeddings."""

    def test_identical_embedding_high_score(self) -> None:
        """Should return high remapped score for identical embeddings."""
        v = _normalized([1, 0, 0, 0])
        para_embs = np.stack([v, _normalized([0, 1, 0, 0])])
        paragraphs = [("S1", "text1"), ("S2", "text2")]

        score, evidence = compute_item_grounding(v, para_embs, paragraphs, top_k=1)

        # Raw cosine = 1.0, remapped = 1.0
        assert score == approx(1.0)
        assert len(evidence) == 1
        assert evidence[0].section_heading == "S1"

    def test_orthogonal_embedding_low_score(self) -> None:
        """Should return 0.0 for orthogonal embeddings (raw cosine ~0)."""
        item = _normalized([1, 0, 0, 0])
        para_embs = np.stack([_normalized([0, 1, 0, 0])])
        paragraphs = [("S", "text")]

        score, _ = compute_item_grounding(item, para_embs, paragraphs)
        assert score == 0.0

    def test_top_k_ordering(self) -> None:
        """Should return evidence ordered by descending similarity."""
        item = _normalized([1, 0, 0, 0])
        para_embs = np.stack(
            [
                _normalized([0, 1, 0, 0]),
                _normalized([0.9, 0.1, 0, 0]),
                _normalized([0.5, 0.5, 0, 0]),
            ]
        )
        paragraphs = [("S0", "low"), ("S1", "high"), ("S2", "med")]

        _, evidence = compute_item_grounding(item, para_embs, paragraphs, top_k=3)

        assert evidence[0].section_heading == "S1"
        assert evidence[0].similarity > evidence[1].similarity

    def test_evidence_text_clipped(self) -> None:
        """Should clip long evidence text with ellipsis."""
        item = _normalized([1, 0, 0, 0])
        para_embs = np.stack([item])
        long_text = str("x" * 300)
        paragraphs = [("S", long_text)]

        _, evidence = compute_item_grounding(item, para_embs, paragraphs, top_k=1)
        assert len(evidence[0].text) < 300
        assert evidence[0].text.endswith("...")


class TestEvaluateGrounding:
    """Tests for evaluate_grounding with synthetic embeddings."""

    def test_well_grounded(self) -> None:
        """Should score highly when facets match paragraph embeddings."""
        v1 = _normalized([1, 0, 0, 0])
        v2 = _normalized([0, 1, 0, 0])
        facet_items: dict[str, list[str]] = {
            "problems": ["p1"],
            "methods": ["m1"],
            "results": [],
            "contributions": [],
        }
        facet_embs = np.stack([v1, v2])
        para_embs = np.stack([v1, v2])
        paragraphs = [("S1", "t1"), ("S2", "t2")]

        result = evaluate_grounding(
            facet_items, facet_embs, para_embs, paragraphs, top_k=1
        )
        # Raw cosine = 1.0, remapped = 1.0
        assert result.score > 0.9
        assert result.per_facet_scores["problems"] == approx(1.0)

    def test_empty_facets(self) -> None:
        """Should return 0.0 when all facets are empty."""
        facet_items: dict[str, list[str]] = {name: [] for name in FACET_NAMES}
        facet_embs = np.zeros((0, 4), dtype=np.float32)
        para_embs = np.stack([_normalized([1, 0, 0, 0])])
        paragraphs = [("S", "t")]

        result = evaluate_grounding(facet_items, facet_embs, para_embs, paragraphs)
        assert result.score == 0.0
        assert all(v == 0.0 for v in result.per_facet_scores.values())


class TestEvaluateRedundancy:
    """Tests for evaluate_redundancy with synthetic embeddings."""

    def test_no_redundancy(self) -> None:
        """Should return 1.0 for orthogonal embeddings across facets."""
        items = {
            "problems": ["p1"],
            "methods": ["m1"],
            "results": ["r1"],
            "contributions": ["c1"],
        }
        embs = np.stack(
            [
                _normalized([1, 0, 0, 0]),
                _normalized([0, 1, 0, 0]),
                _normalized([0, 0, 1, 0]),
                _normalized([0, 0, 0, 1]),
            ]
        )

        result = evaluate_redundancy(items, embs)
        assert result.score == 1.0
        assert result.flags == []
        assert result.near_duplicate_indices == set()

    def test_intra_facet_duplicate(self) -> None:
        """Should flag identical items within a single facet."""
        v = _normalized([1, 0, 0, 0])
        items: dict[str, list[str]] = {
            "problems": ["p1", "p2"],
            "methods": [],
            "results": [],
            "contributions": [],
        }
        embs = np.stack([v, v])

        result = evaluate_redundancy(items, embs)
        assert result.score < 1.0
        assert "intra_facet_duplicate" in result.flags
        assert 0 in result.near_duplicate_indices
        assert 1 in result.near_duplicate_indices

    def test_cross_facet_overlap(self) -> None:
        """Should flag identical items across different facets."""
        v = _normalized([1, 0, 0, 0])
        items: dict[str, list[str]] = {
            "problems": ["p1"],
            "methods": ["m1"],
            "results": [],
            "contributions": [],
        }
        embs = np.stack([v, v])

        result = evaluate_redundancy(items, embs)
        assert result.score < 1.0
        assert "cross_facet_overlap_high" in result.flags

    def test_single_item(self) -> None:
        """Should return 1.0 when only one item exists."""
        items: dict[str, list[str]] = {
            "problems": ["p1"],
            "methods": [],
            "results": [],
            "contributions": [],
        }
        embs = np.stack([_normalized([1, 0, 0, 0])])

        result = evaluate_redundancy(items, embs)
        assert result.score == 1.0
        assert result.flags == []

    def test_penalty_scales_with_severity(self) -> None:
        """Should penalize more for higher-similarity duplicates."""
        v1 = _normalized([1, 0, 0, 0])
        v2 = _normalized([0.99, 0.01, 0, 0])  # Very similar
        v3 = _normalized([0, 1, 0, 0])  # Different
        # Two similar items in same facet
        items_similar: dict[str, list[str]] = {
            "problems": ["p1", "p2"],
            "methods": ["m1"],
            "results": [],
            "contributions": [],
        }
        embs_similar = np.stack([v1, v1, v3])  # Identical = max severity
        embs_close = np.stack([v1, v2, v3])  # Close but not identical

        result_identical = evaluate_redundancy(items_similar, embs_similar)
        result_close = evaluate_redundancy(items_similar, embs_close)
        assert result_close.score > result_identical.score


class TestComputeItemGenericness:
    """Tests for compute_item_genericness function."""

    def test_specific_text(self) -> None:
        """Should return low genericness for text with names, numbers, acronyms."""
        text = "BERT achieves 95.3% F1 on the SQuAD-2.0 benchmark dataset."
        score = compute_item_genericness(text)
        assert score < 0.2

    def test_meta_description_generic(self) -> None:
        """Should return high genericness for meta-description boilerplate."""
        text = "The paper addresses the challenge of improving performance."
        score = compute_item_genericness(text)
        assert score >= 0.3

    def test_meta_with_some_specifics(self) -> None:
        """Should still have nonzero genericness for meta + some specific terms."""
        text = "The paper addresses the challenge of BERT fine-tuning on SQuAD."
        score = compute_item_genericness(text)
        assert score > 0.0  # Meta prefix still contributes

    def test_empty_text(self) -> None:
        """Should return 1.0 for empty text."""
        assert compute_item_genericness("") == 1.0

    def test_pure_direct_statement(self) -> None:
        """Should return 0 genericness for direct specific statements."""
        text = "Cross-lingual BERT models outperform monolingual baselines by 3.2 F1."
        score = compute_item_genericness(text)
        assert score == 0.0


class TestEvaluateSpecificity:
    """Tests for evaluate_specificity function."""

    def test_specific_items(self) -> None:
        """Should score highly for items with technical details."""
        items = {
            "problems": [
                "BERT-large fails on SQuAD 2.0 adversarial examples with 40% drop."
            ],
            "methods": [
                "Fine-tune RoBERTa-base with learning rate 3e-5 for 10 epochs."
            ],
            "results": [
                "F1 improved from 82.1% to 89.7% on the CoNLL-2003 NER benchmark."
            ],
            "contributions": ["Release of NER-Bench, a 50k-sample evaluation suite."],
        }
        score, flags = evaluate_specificity(items)
        assert score > 0.5
        assert "high_genericness" not in flags

    def test_generic_items(self) -> None:
        """Should score lower for meta-description items."""
        items = {
            "problems": [
                "The paper addresses the challenge of improving model performance."
            ],
            "methods": ["The authors propose a novel approach for the task."],
            "results": ["The results demonstrate significant improvement."],
            "contributions": ["This work introduces a new framework."],
        }
        score, _flags = evaluate_specificity(items)
        assert score < 0.8

    def test_no_items(self) -> None:
        """Should return 0.0 and flag when no facet items exist."""
        items: dict[str, list[str]] = {name: [] for name in FACET_NAMES}
        score, flags = evaluate_specificity(items)
        assert score == 0.0
        assert "no_facet_items" in flags


class TestPredictFacetType:
    """Tests for predict_facet_type informational classifier."""

    def test_results_with_numbers(self) -> None:
        """Should predict 'results' for text with metrics and comparisons."""
        text = "The model achieved 95.2% accuracy, outperforming the baseline by 3.1 points."
        assert predict_facet_type(text) == "results"

    def test_methods_with_propose(self) -> None:
        """Should predict 'methods' for text describing an approach."""
        text = (
            "We propose a transformer-based architecture with cross-attention modules."
        )
        assert predict_facet_type(text) == "methods"

    def test_problems_with_challenge(self) -> None:
        """Should predict 'problems' for text framing a challenge."""
        text = (
            "The main challenge is the lack of labeled data in low-resource settings."
        )
        assert predict_facet_type(text) == "problems"

    def test_contributions_with_novel(self) -> None:
        """Should predict 'contributions' for text about novelty."""
        text = "The key contribution is a novel framework that advances the field."
        assert predict_facet_type(text) == "contributions"

    def test_no_signal_returns_none(self) -> None:
        """Should return None when no facet-type keywords match."""
        text = "Some text with no clear facet type signal at all."
        result = predict_facet_type(text)
        assert result is None


class TestComputeOverallScore:
    """Tests for compute_overall_score weighted combination."""

    def test_all_perfect(self) -> None:
        """Should return 1.0 when all components are 1.0."""
        components = {
            "structure": 1.0,
            "grounding": 1.0,
            "redundancy": 1.0,
            "specificity": 1.0,
        }
        assert compute_overall_score(components) == approx(1.0)

    def test_all_zero(self) -> None:
        """Should return 0.0 when all components are 0.0."""
        components = {
            "structure": 0.0,
            "grounding": 0.0,
            "redundancy": 0.0,
            "specificity": 0.0,
        }
        assert compute_overall_score(components) == approx(0.0)

    def test_grounding_weighted_most(self) -> None:
        """Should weight grounding (0.45) more than structure (0.15)."""
        high_grounding = {
            "structure": 0.0,
            "grounding": 1.0,
            "redundancy": 0.0,
            "specificity": 0.0,
        }
        high_structure = {
            "structure": 1.0,
            "grounding": 0.0,
            "redundancy": 0.0,
            "specificity": 0.0,
        }
        assert compute_overall_score(high_grounding) > compute_overall_score(
            high_structure
        )
