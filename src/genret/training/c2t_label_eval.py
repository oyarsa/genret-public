"""No-training proxy evaluation of C2T keyword labels.

Given a built C2T index directory, computes two families of metrics
without training a model:

1. **Intrinsic label metrics** per facet: leaf-label collision rate,
   within-label redundancy, label-to-members faithfulness, and
   label-vs-other-leaves separation (all on MiniLM embeddings).
2. **Zero-shot leaf retrieval**: replaces the trained GLEN decoder with
   cosine similarity between query facet text and leaf label strings,
   expands leaves to member papers, fuses facets with RRF, and scores
   recall@k against citation gold pairs.

The proxy is calibrated against trained C2T leaderboard results: if the
proxy ordering matches the trained ordering on known configs, it can
rank new label configs (scoring x selection) without GPU training.

All embedding access goes through an injected ``EmbedFn`` so tests can
use deterministic fakes and the CLI wires in MiniLM.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from genret.paper_parsing import FacetRecordType, Facets, TrainingPair
from genret.training.c2t_index import _load_facets_map
from genret.training.cluster_keyword_labels import EmbedFn
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_decoding import FacetDecodeResult
from genret.training.codebook_evaluate import (
    _fuse_facet_results,
    queries_from_split,
)
from genret.training.evaluate import _build_ground_truth, compute_recall_at_k
from genret.types import CorpusId
from genret.utils import iter_jsonl

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class FacetLabelMetrics:
    """Intrinsic + retrieval metrics for one facet's labels."""

    n_leaves: int
    n_distinct_docids: int
    collision_rate: float
    mean_redundancy: float
    mean_faithfulness: float
    mean_separation: float
    recall_at_20: float


@dataclass(frozen=True, kw_only=True)
class LabelEvalResult:
    """Full proxy-evaluation result for one C2T index."""

    per_facet: dict[str, FacetLabelMetrics]
    fused_recall_at_20: float
    fused_recall_at_50: float
    n_queries: int

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "fused_recall_at_20": self.fused_recall_at_20,
            "fused_recall_at_50": self.fused_recall_at_50,
            "n_queries": self.n_queries,
            "per_facet": {f: vars(m).copy() for f, m in self.per_facet.items()},
        }


def _mean(xs: list[float]) -> float:
    """Mean of a possibly-empty list; 0.0 when empty."""
    return float(np.mean(xs)) if xs else 0.0


def _load_docid_groups(
    c2t_dir: Path,
) -> dict[str, dict[str, list[CorpusId]]]:
    """``facet -> docid string -> sorted member papers``."""
    out: dict[str, dict[str, list[CorpusId]]] = {}
    for row in iter_jsonl(c2t_dir / "paper_c2t_docids.jsonl"):
        facet = str(row["facet"])
        cid = CorpusId(int(row["corpusid"]))
        out.setdefault(facet, {}).setdefault(str(row["docid"]), []).append(cid)
    for groups in out.values():
        for papers in groups.values():
            papers.sort()
    return out


def _load_leaf_counts(c2t_dir: Path) -> dict[str, int]:
    """``facet -> number of populated leaves`` from ``l1_labels.json``."""
    labels = json.loads((c2t_dir / "l1_labels.json").read_text())
    return {facet: len(leaves) for facet, leaves in labels.items()}


def _load_separators(c2t_dir: Path) -> tuple[str, str]:
    """``(keyword_separator, level_separator)`` from ``c2t_params.json``."""
    path = c2t_dir / "c2t_params.json"
    if not path.exists():
        return "-", " "
    params = json.loads(path.read_text())
    return (
        str(params.get("keyword_separator", "-")),
        str(params.get("level_separator", " ")),
    )


def _label_words(docid: str, keyword_sep: str, level_sep: str) -> list[str]:
    """Split a composed C2T-ID back into its keyword tokens."""
    flat = docid.replace(level_sep, keyword_sep) if level_sep else docid
    return [t for t in flat.split(keyword_sep) if t]


def _facet_text(facets_map: dict[CorpusId, Facets], facet: str, cid: CorpusId) -> str:
    """Joined bullet text for one (paper, facet); empty when absent."""
    facets = facets_map.get(cid)
    if facets is None:
        return ""
    items = getattr(facets, facet, None) or []
    return "\n".join(str(item) for item in items)


def _mean_pairwise_cosine(vecs: npt.NDArray[np.float32]) -> float:
    """Mean cosine over all unordered pairs of unit-norm row vectors."""
    n = vecs.shape[0]
    sims = vecs @ vecs.T
    return float((sims.sum() - np.trace(sims)) / (n * (n - 1)))


def _facet_intrinsics(
    docid_groups: dict[str, list[CorpusId]],
    n_leaves: int,
    member_texts: dict[CorpusId, str],
    embed_fn: EmbedFn,
    keyword_sep: str,
    level_sep: str,
) -> tuple[float, float, float, float]:
    """``(collision_rate, redundancy, faithfulness, separation)``."""
    docids = sorted(docid_groups)
    collision = (n_leaves - len(docids)) / n_leaves if n_leaves else 0.0
    if not docids:
        return collision, 0.0, 0.0, 0.0

    # Redundancy: mean pairwise cosine among each label's word embeddings.
    all_words = sorted(
        {w for d in docids for w in _label_words(d, keyword_sep, level_sep)}
    )
    word_vecs: dict[str, npt.NDArray[np.float32]] = (
        dict(zip(all_words, embed_fn(all_words), strict=True)) if all_words else {}
    )
    redundancies: list[float] = []
    for docid in docids:
        words = _label_words(docid, keyword_sep, level_sep)
        if len(words) < 2:
            continue
        redundancies.append(
            _mean_pairwise_cosine(np.stack([word_vecs[w] for w in words]))
        )

    # Faithfulness + separation: label embedding vs member-text centroids.
    label_vecs = embed_fn(docids)
    texts_to_embed: list[str] = []
    spans: list[tuple[int, int]] = []
    for docid in docids:
        texts = [t for cid in docid_groups[docid] if (t := member_texts.get(cid, ""))]
        start = len(texts_to_embed)
        texts_to_embed.extend(texts)
        spans.append((start, len(texts_to_embed)))
    member_vecs = (
        embed_fn(texts_to_embed)
        if texts_to_embed
        else np.zeros((0, 1), dtype=np.float32)
    )
    centroids: list[npt.NDArray[np.float32] | None] = []
    for start, end in spans:
        if start == end:
            centroids.append(None)
            continue
        centroid = member_vecs[start:end].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroids.append(centroid / norm if norm > 0 else None)

    faiths: list[float] = []
    separations: list[float] = []
    valid_centroids = {i: c for i, c in enumerate(centroids) if c is not None}
    valid = sorted(valid_centroids)
    centroid_matrix = np.stack([valid_centroids[i] for i in valid]) if valid else None
    for pos, i in enumerate(valid):
        own = float(label_vecs[i] @ valid_centroids[i])
        faiths.append(own)
        if centroid_matrix is not None and len(valid) > 1:
            sims = label_vecs[i] @ centroid_matrix.T
            others = np.delete(sims, pos)
            separations.append(own - float(others.max()))

    return collision, _mean(redundancies), _mean(faiths), _mean(separations)


def _load_gold(
    pairs_path: Path, queries: list[CorpusId]
) -> dict[CorpusId, set[CorpusId]]:
    """``query -> union of gold papers over all relation types``."""
    wanted = set(queries)
    pairs = [
        p
        for raw in iter_jsonl(pairs_path)
        if (p := TrainingPair.model_validate(raw)).paper_a in wanted
    ]
    gt = _build_ground_truth(pairs)
    return {q: set().union(*gt[q].values()) for q in gt if gt[q]}


def _facet_retrieval_beams(
    facet: str,
    query_texts: dict[CorpusId, str],
    docid_groups: dict[str, list[CorpusId]],
    embed_fn: EmbedFn,
    top_leaves: int,
) -> dict[CorpusId, list[FacetDecodeResult]]:
    """Per-query ranked leaf "beams" via label-embedding cosine."""
    docids = sorted(docid_groups)
    if not docids or not query_texts:
        return {}
    label_vecs = embed_fn(docids)
    qids = sorted(query_texts)
    query_vecs = embed_fn([query_texts[q] for q in qids])
    sims = query_vecs @ label_vecs.T
    out: dict[CorpusId, list[FacetDecodeResult]] = {}
    for qi, qid in enumerate(qids):
        order = np.argsort(-sims[qi])[:top_leaves]
        out[qid] = [
            FacetDecodeResult(
                facet=facet,
                rank=rank,
                score=float(sims[qi, di]),
                l0=-1,
                l1=-1,
                paper_ids=docid_groups[docids[int(di)]],
            )
            for rank, di in enumerate(order)
        ]
    return out


def evaluate_c2t_labels(
    c2t_dir: Path,
    facets_path: Path,
    split_path: Path,
    pairs_path: Path,
    *,
    embed_fn: EmbedFn,
    facets: Iterable[str] = FACET_NAMES,
    facets_type: FacetRecordType = FacetRecordType.STANDARD,
    top_leaves: int = 100,
) -> LabelEvalResult:
    """Proxy-evaluate one C2T index without training. See module docstring."""
    facet_list = list(facets)
    docid_groups = _load_docid_groups(c2t_dir)
    leaf_counts = _load_leaf_counts(c2t_dir)
    keyword_sep, level_sep = _load_separators(c2t_dir)
    facets_map = _load_facets_map(facets_path, facets_type)

    queries = queries_from_split(split_path)
    gold = _load_gold(pairs_path, queries)

    per_facet_beams: dict[str, dict[CorpusId, list[FacetDecodeResult]]] = {}
    intrinsics: dict[str, tuple[int, int, float, float, float, float]] = {}

    for facet in facet_list:
        groups = docid_groups.get(facet, {})
        member_texts = {
            cid: _facet_text(facets_map, facet, cid)
            for papers in groups.values()
            for cid in papers
        }
        collision, redundancy, faithfulness, separation = _facet_intrinsics(
            groups,
            leaf_counts.get(facet, len(groups)),
            member_texts,
            embed_fn,
            keyword_sep,
            level_sep,
        )
        intrinsics[facet] = (
            leaf_counts.get(facet, len(groups)),
            len(groups),
            collision,
            redundancy,
            faithfulness,
            separation,
        )

        query_texts = {
            q: text
            for q in queries
            if q in gold and (text := _facet_text(facets_map, facet, q))
        }
        per_facet_beams[facet] = _facet_retrieval_beams(
            facet, query_texts, groups, embed_fn, top_leaves
        )

    # Per-facet recall@20 (no fusion) + fused recall over shared queries.
    eval_queries = sorted(
        q for q in gold if any(q in per_facet_beams[f] for f in facet_list)
    )
    fused_20: list[float] = []
    fused_50: list[float] = []
    per_facet_20: dict[str, list[float]] = {f: [] for f in facet_list}
    for q in eval_queries:
        per_facet = {
            f: per_facet_beams[f][q] for f in facet_list if q in per_facet_beams[f]
        }
        fused = _fuse_facet_results(per_facet)
        retrieved = [r.corpusid for r in fused]
        fused_20.append(compute_recall_at_k(retrieved, gold[q], 20))
        fused_50.append(compute_recall_at_k(retrieved, gold[q], 50))
        for f, beams in per_facet.items():
            flat: list[CorpusId] = []
            seen: set[CorpusId] = set()
            for beam in beams:
                for cid in beam.paper_ids:
                    if cid not in seen:
                        seen.add(cid)
                        flat.append(cid)
            per_facet_20[f].append(compute_recall_at_k(flat, gold[q], 20))

    per_facet_metrics = {
        f: FacetLabelMetrics(
            n_leaves=intrinsics[f][0],
            n_distinct_docids=intrinsics[f][1],
            collision_rate=intrinsics[f][2],
            mean_redundancy=intrinsics[f][3],
            mean_faithfulness=intrinsics[f][4],
            mean_separation=intrinsics[f][5],
            recall_at_20=_mean(per_facet_20[f]),
        )
        for f in facet_list
    }
    return LabelEvalResult(
        per_facet=per_facet_metrics,
        fused_recall_at_20=_mean(fused_20),
        fused_recall_at_50=_mean(fused_50),
        n_queries=len(eval_queries),
    )


__all__ = [
    "FacetLabelMetrics",
    "LabelEvalResult",
    "evaluate_c2t_labels",
]
