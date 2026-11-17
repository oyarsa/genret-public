"""Per-paper evaluation: run all checks and assemble a PaperEvaluation record."""

from collections.abc import Sequence
from datetime import UTC, datetime

from genret.llm.encoder import Encoder
from genret.paper_parsing import Facets, FacetsSummary, PaperGraph

from .constants import EVIDENCE_MIN_MATCH_BLOCK, FACET_NAMES, LOW_GROUNDING_THRESHOLD
from .embeddings import EmbeddingCache, get_cached_embeddings
from .evidence import _EMPTY_FIDELITY, evaluate_evidence_fidelity
from .models import EvidenceSnippet, ItemEvaluation, PaperEvaluation, PaperScores
from .scoring import (
    GroundingResult,
    RedundancyResult,
    compute_item_genericness,
    compute_overall_score,
    evaluate_grounding,
    evaluate_redundancy,
    evaluate_specificity,
    evaluate_structure,
    predict_facet_type,
)
from .text_utils import count_sentences, extract_paragraphs


async def evaluate_paper(
    facets: Facets,
    paper: PaperGraph | None,
    encoder: Encoder,
    para_cache: EmbeddingCache | None,
    facet_cache: EmbeddingCache,
    top_k: int = 3,
    facet_evidence: dict[str, list[list[str]]] | None = None,
    evidence_min_match_block: int = EVIDENCE_MIN_MATCH_BLOCK,
) -> PaperEvaluation:
    """Run all evaluation checks on a single paper's facets.

    Args:
        facets: Extracted facets for the paper.
        paper: Source paper graph, or `None` when grounding is disabled.
        encoder: Embedding encoder (local MiniLM by default).
        para_cache: Cache for paragraph embeddings, or `None` to skip grounding.
            When `None`, grounding scores default to 0 and no paragraph
            embeddings are computed.
        facet_cache: Cache for facet item embeddings.
        top_k: Number of evidence snippets per item.
        facet_evidence: Per-item model-generated evidence, keyed by facet name.
            Each value is a list (one entry per item) of evidence strings.
            ``None`` for standard extraction records with no evidence.
        evidence_min_match_block: Minimum matching block size used for
            evidence fidelity coverage. Must be >= 1.

    Returns:
        Complete evaluation record.
    """
    if evidence_min_match_block < 1:
        msg = f"evidence_min_match_block must be >= 1, got {evidence_min_match_block}"
        raise ValueError(msg)

    summary = FacetsSummary(
        problems=facets.problems,
        methods=facets.methods,
        results=facets.results,
        contributions=facets.contributions,
    )

    # Collect facet items
    facet_items: dict[str, list[str]] = {
        name: list(items) for name, items in summary.facet_items().items()
    }
    all_items = [item for name in FACET_NAMES for item in facet_items[name]]
    facet_counts = {name: len(items) for name, items in facet_items.items()}

    # Structural evaluation (no embeddings needed)
    struct_score, struct_flags = evaluate_structure(summary)

    all_flags = list(struct_flags)

    # Extract paragraphs only when grounding is enabled
    if paper is not None and para_cache is not None:
        paragraphs = extract_paragraphs(paper)
        para_texts = [text for _, text in paragraphs]
    else:
        paragraphs = list[tuple[str, str]]()
        para_texts = list[str]()

    has_items = len(all_items) > 0
    has_para_content = has_items and len(para_texts) > 0

    if has_items:
        facet_embeddings = await get_cached_embeddings(
            all_items, facets.corpusid, encoder, facet_cache
        )

        if has_para_content:
            # para_texts is only populated when para_cache is not None
            assert para_cache is not None
            para_embeddings = await get_cached_embeddings(
                para_texts, facets.corpusid, encoder, para_cache
            )
            grounding = evaluate_grounding(
                facet_items, facet_embeddings, para_embeddings, paragraphs, top_k
            )
            for name in FACET_NAMES:
                if (
                    grounding.per_facet_scores.get(name, 0) < LOW_GROUNDING_THRESHOLD
                    and facet_items[name]
                ):
                    all_flags.append(f"{name}_low_grounding")
        else:
            if paper is not None:
                all_flags.append("no_paper_content")
            grounding = GroundingResult(
                score=0.0,
                per_facet_scores=dict.fromkeys(FACET_NAMES, 0.0),
                per_facet_evidence={name: [] for name in FACET_NAMES},
            )

        redundancy = evaluate_redundancy(facet_items, facet_embeddings)
        all_flags.extend(redundancy.flags)
    else:
        if paper is not None and not para_texts:
            all_flags.append("no_paper_content")
        grounding = GroundingResult(
            score=0.0,
            per_facet_scores=dict.fromkeys(FACET_NAMES, 0.0),
            per_facet_evidence={name: [] for name in FACET_NAMES},
        )
        redundancy = RedundancyResult(score=1.0, flags=[], near_duplicate_indices=set())
        facet_embeddings = facet_cache.empty_matrix()

    # Specificity
    spec_score, spec_flags = evaluate_specificity(facet_items)
    all_flags.extend(spec_flags)

    # Overall score
    components = {
        "structure": struct_score,
        "grounding": grounding.score,
        "redundancy": redundancy.score,
        "specificity": spec_score,
    }
    overall = compute_overall_score(components)

    # Build per-item metrics
    item_metrics: dict[str, Sequence[ItemEvaluation]] = {}
    global_idx = 0
    for name in FACET_NAMES:
        evals: list[ItemEvaluation] = []
        evidence_for_facet = grounding.per_facet_evidence.get(name, [])
        model_ev_for_facet = (facet_evidence or {}).get(name, [])

        for i, item_text in enumerate(facet_items[name]):
            evidence: list[EvidenceSnippet] = (
                evidence_for_facet[i] if i < len(evidence_for_facet) else []
            )
            model_ev: list[str] = (
                model_ev_for_facet[i] if i < len(model_ev_for_facet) else []
            )

            item_grounding = float(evidence[0].similarity) if evidence else 0.0

            fidelity = _EMPTY_FIDELITY
            ev_fidelity: float | None = None
            if model_ev:
                fidelity = evaluate_evidence_fidelity(
                    list(model_ev),
                    para_texts,
                    min_match_block=evidence_min_match_block,
                )
                ev_fidelity = fidelity.mean_fidelity

            evals.append(
                ItemEvaluation(
                    text=item_text,
                    char_count=len(item_text),
                    sentence_count=count_sentences(item_text),
                    grounding_score=item_grounding,
                    top_evidence=evidence,
                    model_evidence=model_ev,
                    predicted_facet_type=predict_facet_type(item_text),
                    genericness_score=compute_item_genericness(item_text),
                    is_near_duplicate=global_idx in redundancy.near_duplicate_indices,
                    evidence_verbatim_count=fidelity.verbatim_count,
                    evidence_faithful_count=fidelity.faithful_count,
                    evidence_fidelity=ev_fidelity,
                )
            )
            global_idx += 1

        item_metrics[name] = evals

    return PaperEvaluation(
        corpusid=facets.corpusid,
        title=paper.title if paper is not None else None,
        facet_counts=facet_counts,
        scores=PaperScores(
            structure=struct_score,
            grounding=grounding.score,
            redundancy=redundancy.score,
            specificity=spec_score,
            overall=overall,
        ),
        flags=all_flags,
        item_metrics=item_metrics,
        evaluated_at=datetime.now(UTC).isoformat(),
    )
