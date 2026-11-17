# pyright: basic
# ruff: noqa: PLC0415  Lazy-import heavy sentence-transformers only when snap=cosine
"""Evaluation harness for the GLEN generative retriever.

Mirrors ``codebook_evaluate.py`` but uses trie-constrained decoding over
normalised facet-text DocIDs instead of codebook cluster-ID tokens.  The
output JSONL schema is identical to the graph retriever's output so
downstream metric scripts can be reused unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from genret._shed.transformers import decode_tokens
from genret.paper_parsing import FacetEvidence, FacetRecordType, Facets
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_dataset import _build_messages, _facet_text
from genret.training.codebook_evaluate import (
    FusedResult,
    _fuse_facet_results,
    _load_facets_and_evidence,
)
from genret.training.glen_bullet_dataset.rendering import _build_bullet_messages
from genret.training.glen_decoding import (
    GlenDecodeResult,
    GlenIndex,
    decode_per_facet_glen,
    decode_unconstrained,
)
from genret.training.glen_docids import split_bullets
from genret.training.glen_snap import (
    MINILM_MODEL_NAME,
    DocIDIndex,
    build_docid_index,
    snap_cosine,
    snap_levenshtein,
)
from genret.types import CorpusId
from genret.utils import open_auto

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

SnapMode = Literal["off", "levenshtein", "cosine"]


def evaluate_glen_retrieval(
    *,
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizerBase,
    index: GlenIndex,
    queries: Sequence[CorpusId],
    facets_map: dict[CorpusId, Facets],
    evidence_map: dict[CorpusId, FacetEvidence] | None,
    output_path: Path,
    num_beams: int = 50,
    max_new_tokens: int = 32,
    top_n: int = 50,
    prompt_kind: str = "retrieval",
    per_bullet: bool = False,
    bullet_prompt: bool = False,
    titles: Mapping[CorpusId, str] | None = None,
    facets: tuple[str, ...] = FACET_NAMES,
    snap_mode: SnapMode = "off",
    snap_top_k: int = 1,
    num_beam_groups: int = 1,
    diversity_penalty: float = 0.0,
    paper_dedup: bool = False,
    record_docids_path: Path | None = None,
    prf_lambda: float = 0.0,
    prf_depth_gate: int = 4,
    prf_seed_k: int = 20,
    prf_candidates_dir: Path | None = None,
) -> None:
    """Run 4-facet GLEN beam search + RRF for every query.

    Output JSONL matches the graph retriever's ``retrieval_cross_facet.jsonl``
    schema so downstream evaluation scripts require no changes.

    When ``per_bullet=True``, each bullet of each facet is queried independently
    and the results are aggregated within each facet before RRF fusion, producing
    up to ``4 * B`` beam-search calls per query where B is the mean bullet count.

    When ``snap_mode != "off"``, an additional **unconstrained** beam search
    runs per facet/bullet. Each of its decoded sequences is mapped (snapped)
    to the ``snap_top_k`` nearest stored DocIDs by either Levenshtein
    distance (``"levenshtein"``) or by cosine similarity on MiniLM-L12-v2
    embeddings (``"cosine"``). The snapped candidates are unioned with the
    trie-constrained ones before RRF fusion (augment, not replace).

    When ``record_docids_path`` is set, a side JSONL is written with one
    row per (query, facet, beam): the decoded DocID text, its beam rank,
    and the papers stored under it. Feeds the bullet-usage analysis
    (``glen analyze-bullet-usage``) — which stored DocIDs ever fire.

    When ``prf_lambda > 0`` (requires ``prf_candidates_dir``), graph-PRF
    two-pass decoding is enabled: pass 1 decodes normally and fuses to a
    top-``prf_seed_k`` seed set; pass 2 re-decodes with trie branches
    biased toward graph neighbours of the seeds (potential shaping,
    gated below ``prf_depth_gate``). Per facet, pass-2 beams are ranked
    first and pass-1 beams appended, so the bias can promote but never
    remove candidates.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_map = evidence_map or {}

    prf_pair_weights = None
    if prf_lambda > 0:
        if prf_candidates_dir is None:
            raise ValueError("prf_lambda > 0 requires prf_candidates_dir")
        from genret.training.glen_prf import load_prf_pair_weights

        prf_pair_weights = load_prf_pair_weights(prf_candidates_dir)
        logger.info(
            "Graph-PRF enabled: lambda=%.3f, depth_gate=%d, seed_k=%d",
            prf_lambda,
            prf_depth_gate,
            prf_seed_k,
        )

    docid_index: DocIDIndex | None = None
    encoder = None
    if snap_mode != "off":
        docid_index = build_docid_index(
            index,
            tokeniser,
            facets=facets,
            with_embeddings=snap_mode == "cosine",
        )
        logger.info(
            "Snap mode %s enabled with top_k=%d; DocID lookup built for %d facets",
            snap_mode,
            snap_top_k,
            len(facets),
        )
        if snap_mode == "cosine":
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(MINILM_MODEL_NAME)

    with ExitStack() as stack:
        fh = stack.enter_context(open_auto(output_path, "w"))
        docids_fh = None
        if record_docids_path is not None:
            record_docids_path.parent.mkdir(parents=True, exist_ok=True)
            docids_fh = stack.enter_context(open_auto(record_docids_path, "w"))

        def _decode_query(
            paper_facets: Facets,
            evidence: FacetEvidence | None,
            facet_bias: dict[str, dict[tuple[int, ...], list[tuple[int, float]]]]
            | None = None,
            title: str = "",
        ) -> dict[str, list[GlenDecodeResult]]:
            per_facet: dict[str, list[GlenDecodeResult]] = {}
            for facet in facets:
                text = _facet_text(paper_facets, facet, evidence)
                if not text:
                    per_facet[facet] = []
                    continue
                if per_bullet:
                    bullets = split_bullets(text)
                    query_texts = bullets if bullets else [text]
                else:
                    query_texts = [text]

                factory = None
                if facet_bias is not None:
                    from genret.training.glen_prf import (
                        GraphPrfLogitsProcessor,
                    )

                    children_bias = facet_bias.get(facet, {})

                    def factory(
                        prompt_len: int,
                        eos_id: int,
                        _cb: dict[
                            tuple[int, ...], list[tuple[int, float]]
                        ] = children_bias,  # bind per facet
                    ) -> GraphPrfLogitsProcessor:
                        return GraphPrfLogitsProcessor(
                            children_bias=_cb,
                            prompt_len=prompt_len,
                            eos_id=eos_id,
                        )

                facet_results: list[GlenDecodeResult] = []
                snap_rank_offset = 0
                for query_text in query_texts:
                    if bullet_prompt:
                        # Match training input exactly: title + "- {bullet}" via
                        # the per-bullet renderer's message builder.
                        messages = _build_bullet_messages(
                            facet, title, f"- {query_text}"
                        )
                    else:
                        messages = _build_messages(facet, query_text, kind=prompt_kind)
                    trie_beams = decode_per_facet_glen(
                        model=model,
                        tokeniser=tokeniser,
                        messages=messages,
                        facet=facet,
                        index=index,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                        num_beam_groups=num_beam_groups,
                        diversity_penalty=diversity_penalty,
                        paper_dedup=paper_dedup,
                        extra_processor_factory=factory,
                    )
                    facet_results.extend(trie_beams)

                    if snap_mode != "off" and docid_index is not None:
                        snapped = _run_snap_for_query(
                            model=model,
                            tokeniser=tokeniser,
                            messages=messages,
                            facet=facet,
                            num_beams=num_beams,
                            max_new_tokens=max_new_tokens,
                            snap_mode=snap_mode,
                            docid_index=docid_index,
                            encoder=encoder,
                            snap_top_k=snap_top_k,
                            rank_offset=snap_rank_offset,
                        )
                        facet_results.extend(snapped)
                        snap_rank_offset += len(snapped)
                per_facet[facet] = facet_results
            return per_facet

        for qid in queries:
            paper_facets = facets_map.get(qid)
            if paper_facets is None:
                logger.warning("Query %d not in facets map; skipping", qid)
                continue
            evidence = evidence_map.get(qid)
            title = titles.get(qid, "") if titles else ""

            per_facet = _decode_query(paper_facets, evidence, title=title)

            if prf_pair_weights is not None:
                from genret.training.glen_prf import (
                    build_children_bias,
                )

                fused_pass1: list[FusedResult] = _fuse_facet_results(per_facet)  # type: ignore[arg-type]
                seeds = {
                    r.corpusid for r in fused_pass1[:prf_seed_k] if r.corpusid != qid
                }
                facet_bias = {
                    facet: build_children_bias(
                        facet,
                        index,
                        seeds,
                        prf_pair_weights,
                        lam=prf_lambda,
                        depth_gate=prf_depth_gate,
                    )
                    for facet in facets
                }
                pass2 = _decode_query(paper_facets, evidence, facet_bias, title=title)
                # Pass-2 beams first (promotions count), pass-1 appended
                # (nothing the bias demotes is ever lost).
                per_facet = {f: pass2.get(f, []) + per_facet.get(f, []) for f in facets}

            if docids_fh is not None:
                for facet, beams in per_facet.items():
                    for beam in beams:
                        docids_fh.write(
                            json.dumps(
                                {
                                    "query": int(qid),
                                    "facet": facet,
                                    "rank": beam.rank,
                                    "docid": tokeniser.decode(
                                        list(beam.docid_tokens),
                                        skip_special_tokens=True,
                                    ),
                                    "papers": [int(p) for p in beam.paper_ids],
                                    # Score fields feed score-additive fusion
                                    # rules (scripts/eval/minder_fusion.py).
                                    # `score` is length-normalised;
                                    # `logprob_sum` is the raw sum, absent on
                                    # decode paths that cannot supply it.
                                    "score": round(beam.score, 6),
                                    "n_tokens": len(beam.docid_tokens),
                                    "logprob_sum": (
                                        None
                                        if beam.logprob_sum is None
                                        else round(beam.logprob_sum, 6)
                                    ),
                                }
                            )
                            + "\n"
                        )

            # _fuse_facet_results only accesses .rank and .paper_ids —
            # GlenDecodeResult satisfies that interface at runtime.
            fused: list[FusedResult] = _fuse_facet_results(per_facet)  # type: ignore[arg-type]
            fused = [r for r in fused if r.corpusid != qid][:top_n]

            record = {
                "corpusid": int(qid),
                "results": [
                    {
                        "corpusid": int(r.corpusid),
                        "score": round(r.score, 6),
                        "facets": r.facets,
                        "facet_ranks": r.facet_ranks,
                    }
                    for r in fused
                ],
            }
            fh.write(json.dumps(record) + "\n")

    if record_docids_path is not None:
        logger.info("Decoded DocIDs recorded to %s", record_docids_path)


def _run_snap_for_query(
    *,
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    facet: str,
    num_beams: int,
    max_new_tokens: int,
    snap_mode: SnapMode,
    docid_index: DocIDIndex,
    encoder: object,
    snap_top_k: int,
    rank_offset: int,
) -> list[GlenDecodeResult]:
    """Run an unconstrained beam search and snap each beam to nearest DocIDs.

    Returns synthesised ``GlenDecodeResult`` entries so the downstream RRF
    fusion treats them like trie-constrained beams.
    """
    unconstrained = decode_unconstrained(
        model=model,
        tokeniser=tokeniser,
        messages=messages,
        facet=facet,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )
    snapped: list[GlenDecodeResult] = []
    for beam_rank, beam in enumerate(unconstrained):
        if snap_mode == "levenshtein":
            hits = snap_levenshtein(
                beam.docid_tokens,
                facet,
                docid_index,
                top_k=snap_top_k,
            )
        else:
            decoded_text = decode_tokens(tokeniser, list(beam.docid_tokens)).strip()
            if not decoded_text:
                continue
            hits = snap_cosine(
                decoded_text,
                facet,
                docid_index,
                encoder,
                top_k=snap_top_k,
            )
        for hit_rank, (idx, sim, paper_ids) in enumerate(hits):
            snapped.append(
                GlenDecodeResult(
                    facet=facet,
                    # Offset ranks so snap beams sit AFTER trie beams in the
                    # RRF input. RRF uses rank, so order matters for fusion.
                    rank=rank_offset + beam_rank * max(snap_top_k, 1) + hit_rank,
                    score=beam.score + sim,
                    docid_tokens=docid_index.token_seqs[facet][idx],
                    paper_ids=list(paper_ids),
                )
            )
    return snapped


def load_eval_inputs(
    facets_path: Path, facets_type: FacetRecordType
) -> tuple[dict[CorpusId, Facets], dict[CorpusId, FacetEvidence]]:
    """Load facets and evidence for evaluation."""
    return _load_facets_and_evidence(facets_path, facets_type)


__all__ = [
    "SnapMode",
    "evaluate_glen_retrieval",
    "load_eval_inputs",
]
