# pyright: basic
"""C2T-ID DocID construction (Zhang et al., SIGIR-AP 2025).

Converts a per-facet hierarchical codebook (built by ``codebook.py``) into a
DocID scheme where each cluster node's numeric label is replaced by the
top-K most frequent keywords of its member papers, joined by ``-``.

Pipeline per facet:

1. Load the existing 2-level codebook for the facet
   (``corpusid -> (L0, L1)``, plus the populated leaves).
2. Extract per-paper keywords from the facet's bullets (lowercased content
   words, stopwords filtered).
3. Aggregate per-cluster top-K labels at L0 and at (L0, L1).
4. Build the per-paper C2T-ID string:
   ``"<L0 top-k joined by -> || <L1 top-k joined by -> "``
   (top-K levels separated by ``-`` again, matching the paper's notation).
5. Tokenise each C2T-ID with the model tokeniser.
6. Build a per-facet trie + code-to-papers map in the ``GlenIndex`` format
   so the existing GLEN beam-search decoder and training pipelines work
   unchanged.

The keyword-aggregation step is delegated to
:mod:`cluster_keyword_labels`, which is clustering-scheme agnostic — when
we later add Leiden community DocIDs, this same C2T builder reuses the
labelling.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from genret.paper_parsing import FacetRecordType, Facets, facets_from_type
from genret.training.cluster_keyword_labels import (
    EmbedFn,
    aggregate_cluster_labels,
    deduplicate_consecutive,
    extract_paper_keywords,
    extract_paper_keywords_from_texts,
    make_minilm_embedder,
    merge_plural_variants,
)
from genret.training.codebook_data import FACET_NAMES
from genret.training.glen_docids import GlenIndex, _build_trie, save_glen_index
from genret.types import CorpusId
from genret.utils import iter_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class C2TIndexParams:
    """Parameters for C2T-ID DocID construction."""

    top_k: int = 3
    """Number of top keywords per cluster node (paper uses K=3)."""

    keyword_separator: str = "-"
    """Joins keywords within a cluster level (paper uses ``-``)."""

    level_separator: str = " "
    """Joins L0 and L1 label substrings. The paper uses ``-`` here too,
    but using a space keeps tokenisation cleaner with subword tokenisers
    that treat hyphens as part of a word."""

    max_tokens: int = 32
    """Cap on tokenised DocID length to limit beam-search depth."""

    min_keyword_length: int = 3
    """Minimum characters per candidate keyword."""

    dedup_consecutive_keywords: bool = True
    """If True, drop consecutive duplicates in the final C2T-ID — saves
    trie branching when L0 and L1 top-K share a leading keyword."""

    scoring: str = "tfidf"
    """Keyword scoring mode passed to ``aggregate_cluster_labels``:
    ``"tfidf"`` (count * log(N/df)), ``"logodds"`` (Dirichlet-prior
    log-odds z-score vs rest of corpus), ``"embedding"`` (cosine to the
    cluster centroid on MiniLM vectors), or ``"frequency"`` (raw count,
    the C2T paper's choice — ambient words tend to dominate)."""

    selection: str = "topk"
    """Keyword selection mode: ``"topk"`` (top-K by score) or ``"mmr"``
    (diversity-aware exhaustive selection over the top candidates)."""

    mmr_lambda: float = 0.7
    """Relevance weight in the MMR objective (1.0 = pure relevance)."""

    mmr_pool: int = 10
    """Candidate pool size for the MMR subset search."""

    prior_strength: float = 10.0
    """Total pseudo-count mass for the log-odds Dirichlet prior."""

    keyword_source: str = "bullets"
    """Where per-paper keywords come from:

    - ``"bullets"``: tokens from the paper's facet bullets (paper default).
    - ``"titles"``: noun/adj tokens from the paper title (spaCy POS-filtered,
      hyphenated compounds merged).
    - ``"titles_abstracts"``: title + abstract concatenated.
    """

    @property
    def needs_word_vectors(self) -> bool:
        """Whether this config requires word embeddings."""
        return self.scoring == "embedding" or self.selection == "mmr"


def _load_paper_codes(
    codebook_dir: Path, facet: str
) -> dict[CorpusId, tuple[int, int]]:
    """Load ``corpusid -> (L0, L1)`` for one facet from the codebook dir."""
    out: dict[CorpusId, tuple[int, int]] = {}
    path = codebook_dir / facet / "paper_codes.jsonl"
    for row in iter_jsonl(path):
        cid = CorpusId(int(row["corpusid"]))
        l0, l1 = row["code"]
        out[cid] = (int(l0), int(l1))
    return out


def _load_facets_map(
    facets_path: Path, facets_type: FacetRecordType
) -> dict[CorpusId, Facets]:
    """Load ``corpusid -> Facets`` from the per-paper facets JSONL."""
    out: dict[CorpusId, Facets] = {}
    for raw in iter_jsonl(facets_path):
        facets = facets_from_type(raw, facets_type)
        out[facets.corpusid] = facets
    return out


def _build_c2t_id_for_paper(
    l0: int,
    l1: int,
    *,
    l0_labels: dict[int, list[str]],
    l1_labels: dict[tuple[int, int], list[str]],
    params: C2TIndexParams,
) -> str:
    """Compose the natural-language C2T-ID for one (paper, facet)."""
    l0_kws = l0_labels.get(l0, [])
    l1_kws = l1_labels.get((l0, l1), [])
    # If a level has no keywords (e.g. tiny cluster, all stopwords), fall
    # back to a literal cluster index so the DocID is still unique.
    l0_part = params.keyword_separator.join(l0_kws) if l0_kws else f"c{l0}"
    l1_part = params.keyword_separator.join(l1_kws) if l1_kws else f"c{l0}-{l1}"
    docid = f"{l0_part}{params.level_separator}{l1_part}"
    if params.dedup_consecutive_keywords:
        tokens = docid.replace(params.level_separator, params.keyword_separator).split(
            params.keyword_separator
        )
        tokens = [t for t in tokens if t]
        deduped = deduplicate_consecutive(tokens)
        docid = params.keyword_separator.join(deduped)
    return docid


@dataclass(frozen=True, kw_only=True)
class C2TFacetArtifacts:
    """Per-facet build artefacts for inspection and downstream training."""

    facet: str
    l0_labels: dict[int, list[str]]
    """``L0 value -> [top-K keywords]``."""
    l1_labels: dict[tuple[int, int], list[str]]
    """``(L0, L1) -> [top-K keywords]``."""
    paper_docids: dict[CorpusId, str]
    """``corpusid -> C2T-ID string`` for this facet."""


def build_c2t_facet(
    facet: str,
    paper_codes: dict[CorpusId, tuple[int, int]],
    facets_map: dict[CorpusId, Facets],
    params: C2TIndexParams,
    *,
    global_paper_keywords: dict[CorpusId, list[str]] | None = None,
    embedder: EmbedFn | None = None,
) -> C2TFacetArtifacts:
    """Produce C2T-IDs for every paper in one facet.

    ``global_paper_keywords`` overrides per-facet bullet extraction with a
    paper-level keyword list (used by title/abstract sources, which don't
    vary per facet). ``embedder`` is required when the params use
    embedding scoring or MMR selection; it embeds the facet's keyword
    vocabulary once.
    """
    if global_paper_keywords is not None:
        paper_keywords = global_paper_keywords
    else:
        paper_keywords = extract_paper_keywords(
            facets_map, facet, min_length=params.min_keyword_length
        )

    word_vectors: dict[str, npt.NDArray[np.float32]] | None = None
    if params.needs_word_vectors:
        if embedder is None:
            raise ValueError(
                f"params (scoring={params.scoring!r}, "
                f"selection={params.selection!r}) require an embedder"
            )
        vocab = sorted({w for ws in paper_keywords.values() for w in ws})
        if vocab:
            vecs = embedder(vocab)
            word_vectors = dict(zip(vocab, vecs, strict=True))

    # Bucket assignments for both cluster levels.
    l0_assign: dict[CorpusId, int] = {cid: code[0] for cid, code in paper_codes.items()}
    l1_assign: dict[CorpusId, tuple[int, int]] = dict(paper_codes)

    l0_labels = aggregate_cluster_labels(
        paper_keywords,
        l0_assign,
        top_k=params.top_k,
        scoring=params.scoring,
        selection=params.selection,
        mmr_lambda=params.mmr_lambda,
        mmr_pool=params.mmr_pool,
        prior_strength=params.prior_strength,
        word_vectors=word_vectors,
    )
    l1_labels = aggregate_cluster_labels(
        paper_keywords,
        l1_assign,
        top_k=params.top_k,
        scoring=params.scoring,
        selection=params.selection,
        mmr_lambda=params.mmr_lambda,
        mmr_pool=params.mmr_pool,
        prior_strength=params.prior_strength,
        word_vectors=word_vectors,
    )

    paper_docids: dict[CorpusId, str] = {}
    for cid, (l0, l1) in paper_codes.items():
        paper_docids[cid] = _build_c2t_id_for_paper(
            l0,
            l1,
            l0_labels=l0_labels,
            l1_labels=l1_labels,
            params=params,
        )

    logger.info(
        "Built C2T-IDs for facet %s: %d papers, %d distinct DocID strings",
        facet,
        len(paper_docids),
        len(set(paper_docids.values())),
    )
    return C2TFacetArtifacts(
        facet=facet,
        l0_labels=l0_labels,
        l1_labels=l1_labels,
        paper_docids=paper_docids,
    )


def _tokenise(
    text: str, tokeniser: PreTrainedTokenizerBase, *, max_tokens: int
) -> tuple[int, ...]:
    """Tokenise without special tokens; cap to ``max_tokens``."""
    ids = tokeniser.encode(text, add_special_tokens=False)
    return tuple(ids[:max_tokens])


def _load_titles(paper_graphs_path: Path) -> dict[CorpusId, str]:
    return {
        CorpusId(int(r["corpusid"])): str(r.get("title") or "")
        for r in iter_jsonl(paper_graphs_path)
    }


def _load_abstracts(abstracts_path: Path) -> dict[CorpusId, str]:
    return {
        CorpusId(int(r["corpusid"])): str(r.get("abstract") or "")
        for r in iter_jsonl(abstracts_path)
    }


def _build_global_keywords(
    params: C2TIndexParams,
    *,
    paper_graphs_path: Path | None,
    abstracts_path: Path | None,
) -> dict[CorpusId, list[str]] | None:
    """For non-bullet keyword sources, build the paper-level keyword map once.

    Returns None when the source is ``"bullets"`` (per-facet, computed
    inside ``build_c2t_facet``).
    """
    if params.keyword_source == "bullets":
        return None

    import spacy  # noqa: PLC0415

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")

    if params.keyword_source == "titles":
        if paper_graphs_path is None:
            raise ValueError("keyword_source='titles' requires paper_graphs_path")
        texts = _load_titles(paper_graphs_path)
    elif params.keyword_source == "titles_abstracts":
        if paper_graphs_path is None or abstracts_path is None:
            raise ValueError(
                "keyword_source='titles_abstracts' requires both "
                "paper_graphs_path and abstracts_path"
            )
        titles = _load_titles(paper_graphs_path)
        abstracts = _load_abstracts(abstracts_path)
        texts = {
            cid: f"{titles.get(cid, '')}. {abstracts.get(cid, '')}".strip()
            for cid in titles.keys() | abstracts.keys()
        }
    else:
        raise ValueError(f"Unknown keyword_source: {params.keyword_source!r}")

    paper_keywords = extract_paper_keywords_from_texts(
        texts, nlp=nlp, min_length=params.min_keyword_length
    )
    return merge_plural_variants(paper_keywords)


def build_c2t_index(
    codebook_dir: Path,
    facets_path: Path,
    output_dir: Path,
    tokeniser: PreTrainedTokenizerBase,
    *,
    params: C2TIndexParams | None = None,
    facets: Iterable[str] = FACET_NAMES,
    facets_type: FacetRecordType = FacetRecordType.STANDARD,
    paper_graphs_path: Path | None = None,
    abstracts_path: Path | None = None,
) -> tuple[GlenIndex, dict[str, dict[CorpusId, str]]]:
    """Build a GLEN-compatible index using C2T-ID textual DocIDs.

    Returns:
        - ``GlenIndex``: facet → tokenised DocID → [papers], plus the trie.
        - ``per_facet_docids``: ``{facet: {corpusid: c2t_id_string}}`` for
          downstream training data construction (where the model is taught
          to generate the C2T-ID string for paper_b's facet).
    """
    if params is None:
        params = C2TIndexParams()
    facets_map = _load_facets_map(facets_path, facets_type)
    global_keywords = _build_global_keywords(
        params,
        paper_graphs_path=paper_graphs_path,
        abstracts_path=abstracts_path,
    )
    embedder = make_minilm_embedder() if params.needs_word_vectors else None

    code_to_papers: dict[str, dict[tuple[int, ...], list[CorpusId]]] = {}
    per_facet_docids: dict[str, dict[CorpusId, str]] = {}
    per_facet_l0_labels: dict[str, dict[int, list[str]]] = {}
    per_facet_l1_labels: dict[str, dict[str, list[str]]] = {}

    for facet in facets:
        paper_codes = _load_paper_codes(codebook_dir, facet)
        artifacts = build_c2t_facet(
            facet,
            paper_codes,
            facets_map,
            params,
            global_paper_keywords=global_keywords,
            embedder=embedder,
        )

        per_facet_docids[facet] = artifacts.paper_docids
        per_facet_l0_labels[facet] = artifacts.l0_labels
        per_facet_l1_labels[facet] = {
            f"{l0},{l1}": kws for (l0, l1), kws in artifacts.l1_labels.items()
        }

        facet_c2p: dict[tuple[int, ...], list[CorpusId]] = {}
        for cid, docid_str in artifacts.paper_docids.items():
            toks = _tokenise(docid_str, tokeniser, max_tokens=params.max_tokens)
            if not toks:
                continue
            facet_c2p.setdefault(toks, []).append(cid)
        code_to_papers[facet] = facet_c2p

    tries = {f: _build_trie(list(c2p.keys())) for f, c2p in code_to_papers.items()}
    index = GlenIndex(code_to_papers=code_to_papers, trie=tries)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_glen_index(index, output_dir)

    # Save the per-(paper, facet) C2T-ID strings as JSONL for the C2T
    # training-data builder.
    with (output_dir / "paper_c2t_docids.jsonl").open("w") as fh:
        for facet, docids in per_facet_docids.items():
            for cid in sorted(docids):
                fh.write(
                    json.dumps(
                        {
                            "corpusid": int(cid),
                            "facet": facet,
                            "docid": docids[cid],
                        }
                    )
                    + "\n"
                )

    # Save labels for inspection (helpful when debugging or reporting
    # human-readable cluster names).
    (output_dir / "l0_labels.json").write_text(
        json.dumps(
            {
                facet: {str(l0): kws for l0, kws in lab.items()}
                for facet, lab in per_facet_l0_labels.items()
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "l1_labels.json").write_text(
        json.dumps(per_facet_l1_labels, indent=2) + "\n"
    )

    params_record = {
        "top_k": params.top_k,
        "keyword_separator": params.keyword_separator,
        "level_separator": params.level_separator,
        "max_tokens": params.max_tokens,
        "min_keyword_length": params.min_keyword_length,
        "dedup_consecutive_keywords": params.dedup_consecutive_keywords,
        "scoring": params.scoring,
        "keyword_source": params.keyword_source,
        "selection": params.selection,
        "mmr_lambda": params.mmr_lambda,
        "mmr_pool": params.mmr_pool,
        "prior_strength": params.prior_strength,
        "codebook_dir": str(codebook_dir),
        "facets": list(facets),
    }
    (output_dir / "c2t_params.json").write_text(
        json.dumps(params_record, indent=2) + "\n"
    )

    return index, per_facet_docids


def load_paper_c2t_docids(
    c2t_dir: Path,
) -> dict[str, dict[CorpusId, str]]:
    """Read back per-(facet, paper) C2T-ID strings."""
    out: dict[str, dict[CorpusId, str]] = {}
    for row in iter_jsonl(c2t_dir / "paper_c2t_docids.jsonl"):
        facet = str(row["facet"])
        cid = CorpusId(int(row["corpusid"]))
        out.setdefault(facet, {})[cid] = str(row["docid"])
    return out


__all__ = [
    "C2TFacetArtifacts",
    "C2TIndexParams",
    "build_c2t_facet",
    "build_c2t_index",
    "load_paper_c2t_docids",
]
