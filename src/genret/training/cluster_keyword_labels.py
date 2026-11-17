"""Generic cluster-to-keyword label aggregation, reusable across clustering schemes.

The C2T-ID paper (Zhang et al., SIGIR-AP 2025) replaces numeric cluster labels
in a semantic-codebook DocID with the top-K most frequent keywords drawn from
the cluster's member documents. The "K" controls richness vs branching factor.

This module is **clustering-agnostic**. It accepts:

- ``paper_keywords``: per-paper keyword multiset (the "document prior" extracted
  from titles, abstracts, facet bullets, or any other metadata)
- ``cluster_assignments``: per-paper cluster id (could be hierarchical
  k-means ``(L0, L1)`` tuples, Leiden community ids, or any hashable label)

It returns ``cluster_id -> [top-k keywords]``, which is what the downstream
C2T DocID builder needs.

Reuse plan: when we add Leiden-based clustering later, the only swap is
producing a different ``cluster_assignments`` map — keyword extraction and
label aggregation stay unchanged.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping
from math import log, sqrt
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from genret.paper_parsing import Facets
from genret.types import CorpusId

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc

logger = logging.getLogger(__name__)


EmbedFn = Callable[[list[str]], npt.NDArray[np.float32]]
"""Batch text-embedding callable returning unit-norm float32 vectors."""


def make_minilm_embedder() -> EmbedFn:
    """Load MiniLM-L12-v2 once and return a batch-embedding callable."""
    import functools  # noqa: PLC0415

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    from genret._shed.sentence_transformers import encode_texts  # noqa: PLC0415
    from genret.training.glen_snap import MINILM_MODEL_NAME  # noqa: PLC0415

    model = SentenceTransformer(MINILM_MODEL_NAME)
    return functools.partial(encode_texts, model)


# A small English stopword list — sufficient for academic-paper bullets;
# avoids pulling in a heavy NLP dependency for a one-shot preprocessing step.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "we",
        "our",
        "us",
        "they",
        "them",
        "their",
        "i",
        "you",
        "he",
        "she",
        "his",
        "her",
        "him",
        "from",
        "into",
        "than",
        "then",
        "such",
        "also",
        "very",
        "can",
        "may",
        "will",
        "would",
        "should",
        "could",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "yes",
        "all",
        "any",
        "some",
        "more",
        "most",
        "less",
        "least",
        "each",
        "both",
        "while",
        "where",
        "when",
        "what",
        "which",
        "who",
        "how",
        "why",
        "so",
        "up",
        "down",
        "out",
        "over",
        "under",
        "again",
        "further",
        "use",
        "uses",
        "used",
        "using",
        "show",
        "shows",
        "showed",
        "showing",
        "propose",
        "proposed",
        "present",
        "presents",
        "presented",
        "approach",
        "method",
        "methods",
        "technique",
        "techniques",
        "result",
        "results",
        "paper",
        "papers",
        "work",
        "based",
        "however",
        "thus",
        "therefore",
        "hence",
        "since",
        "between",
        "within",
        "without",
        "via",
        "per",
        "due",
        "etc",
        "e.g",
        "i.e",
    }
)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


def _tokenise_to_words(text: str, *, min_length: int = 3) -> list[str]:
    """Lowercase alphanumeric tokens, dropping stopwords + short tokens."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        tok = raw.lower()
        if len(tok) < min_length:
            continue
        if tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


# Extended stopwords for title/abstract extraction — drops academic-paper
# fluff that would otherwise dominate noun-phrase lists.
_TITLE_FLUFF: frozenset[str] = frozenset(
    {
        "approach",
        "method",
        "study",
        "case",
        "review",
        "survey",
        "analysis",
        "framework",
        "model",
        "system",
        "task",
        "paper",
        "work",
        "research",
        "experiment",
        "evaluation",
        "datum",
        "data",
        "recent",
        "advance",
        "novel",
        "effective",
        "baseline",
        "improvement",
        "exploring",
        "understanding",
        "introduction",
        "investigation",
        "empirical",
        "comparative",
        "comparison",
        "limitation",
        "finding",
        "trend",
        "depth",
        "look",
        "general",
        "specific",
        "simple",
        "small",
        "large",
        "good",
        "better",
    }
)


def _candidate_singulars(word: str) -> Iterable[str]:
    """Yield plausible singular forms for a plural-looking word."""
    if len(word) < 4 or not word.endswith("s"):
        return
    if word.endswith("ies") and len(word) > 4:
        yield word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        yield word[:-2]
    if not word.endswith("ss"):
        yield word[:-1]


def merge_plural_variants(
    paper_keywords: Mapping[CorpusId, list[str]],
) -> dict[CorpusId, list[str]]:
    """Corpus-aware plural→singular merge.

    spaCy's lemmatiser occasionally leaves plurals intact (``biases``,
    ``hallucinations``) which then split a cluster's count budget across
    two surface forms. For each plural-looking word, try a few
    morphological singulars (`-s`, `-es`, `-ies → -y`); if any candidate
    is attested elsewhere in the corpus, rewrite to that singular.
    """
    all_words: set[str] = set()
    for ws in paper_keywords.values():
        all_words.update(ws)
    out: dict[CorpusId, list[str]] = {}
    for cid, ws in paper_keywords.items():
        fixed: list[str] = []
        for w in ws:
            singular = w
            for cand in _candidate_singulars(w):
                if cand in all_words:
                    singular = cand
                    break
            fixed.append(singular)
        out[cid] = fixed
    return out


def _merge_hyphenated_compounds(doc: Doc) -> Doc:
    """Retokenise `NOUN/PROPN/ADJ - NOUN/PROPN/ADJ ...` runs as single tokens.

    Default spaCy tokenisation splits ``cross-lingual`` into three tokens.
    Treating compound noun phrases as one keyword keeps DocIDs short and
    keeps semantically-unified terms together in cluster labels.
    """
    from spacy.tokens import Span  # noqa: PLC0415

    spans: list[Span] = []
    toks = list(doc)
    i = 0
    keep_pos = {"NOUN", "PROPN", "ADJ"}
    while i < len(toks) - 2:
        if (
            toks[i].pos_ in keep_pos
            and toks[i + 1].text == "-"
            and toks[i + 2].pos_ in keep_pos
        ):
            start = i
            end = i + 2
            while (
                end + 2 < len(toks)
                and toks[end + 1].text == "-"
                and toks[end + 2].pos_ in keep_pos
            ):
                end += 2
            spans.append(doc[start : end + 1])
            i = end + 1
        else:
            i += 1
    with doc.retokenize() as r:
        for span in spans:
            r.merge(span, attrs={"POS": "NOUN"})
    return doc


def extract_paper_keywords_from_texts(
    texts: Mapping[CorpusId, str],
    *,
    nlp: Language,
    min_length: int = 3,
) -> dict[CorpusId, list[str]]:
    """Per-paper noun/adjective tokens extracted from free-form text.

    Used for title-based and title+abstract C2T variants. spaCy POS-tags
    the text, keeps NOUN/PROPN/ADJ tokens (with hyphenated compounds
    merged), lowercases lemmas, and drops short tokens, stopwords, and
    extended academic-paper fluff.
    """
    out: dict[CorpusId, list[str]] = {}
    for cid, text in texts.items():
        if not text:
            out[cid] = []
            continue
        doc = _merge_hyphenated_compounds(nlp(text))
        words: list[str] = []
        for tok in doc:
            if tok.pos_ not in {"NOUN", "PROPN", "ADJ"}:
                continue
            if tok.is_stop or tok.is_punct or tok.is_space:
                continue
            # For merged hyphenated compounds, lemma is unreliable — use
            # the surface form instead.
            raw = tok.text.lower() if "-" in tok.text else tok.lemma_.lower()
            cleaned = re.sub(r"[^a-z\-]", "", raw).strip("-")
            if len(cleaned) < min_length:
                continue
            if cleaned in _STOPWORDS or cleaned in _TITLE_FLUFF:
                continue
            words.append(cleaned)
        out[cid] = words
    return out


def extract_paper_keywords(
    facets_map: Mapping[CorpusId, Facets],
    facet: str,
    *,
    min_length: int = 3,
) -> dict[CorpusId, list[str]]:
    """Per-paper keyword list extracted from one facet's bullets.

    Each paper's keyword list is the bag of content words across all bullets
    in the named facet. Stopwords + short tokens are filtered. Duplicate
    occurrences within a paper are kept so that a paper using "attention"
    in three bullets contributes "attention" three times when its cluster
    aggregates frequencies.
    """
    out: dict[CorpusId, list[str]] = {}
    for cid, facets in facets_map.items():
        items = getattr(facets, facet, None)
        if not items:
            continue
        words: list[str] = []
        for item in items:
            words.extend(_tokenise_to_words(str(item), min_length=min_length))
        out[cid] = words
    return out


def aggregate_cluster_labels[ClusterT: Hashable](
    paper_keywords: Mapping[CorpusId, list[str]],
    cluster_assignments: Mapping[CorpusId, ClusterT],
    *,
    top_k: int = 3,
    scoring: str = "frequency",
    prior_strength: float = 10.0,
    word_vectors: Mapping[str, npt.NDArray[np.float32]] | None = None,
    selection: str = "topk",
    mmr_lambda: float = 0.7,
    mmr_pool: int = 10,
) -> dict[ClusterT, list[str]]:
    """Top-K keywords per cluster, by descending score.

    Scoring modes:

    - ``"frequency"`` (default): score is the raw count of the keyword in
      the cluster. Ambient words ("data", "model") common to many clusters
      tend to dominate, since they're frequent everywhere.
    - ``"tfidf"``: score is ``tf * idf`` where ``tf`` is the in-cluster
      count and ``idf = log(N / df)`` with ``df`` the number of clusters
      containing the word. Demotes words that appear in many clusters,
      promoting genuinely discriminative ones.
    - ``"logodds"``: Monroe et al. log-odds ratio with an informative
      Dirichlet prior. Compares the in-cluster rate against the pooled
      rate of all other clusters, normalised by the estimated standard
      deviation. Robust to ambient words and verbose papers.
      ``prior_strength`` sets the total pseudo-count mass.
    - ``"embedding"``: cosine similarity between the word's embedding and
      the cluster centroid (count-weighted mean of member word
      embeddings). Picks semantically central words rather than
      statistically overrepresented ones. Requires ``word_vectors``
      (unit-norm embeddings keyed by word).

    Selection modes (applied after scoring):

    - ``"topk"`` (default): take the top-K ranked words.
    - ``"mmr"``: exhaustive diversity-aware selection over the top
      ``mmr_pool`` candidates, maximising ``mmr_lambda * relevance -
      (1 - mmr_lambda) * pairwise cosine redundancy``. Requires
      ``word_vectors``. Output is ordered by relevance.

    Ties broken alphabetically for determinism. Empty clusters map to
    an empty list.
    """
    if selection not in {"topk", "mmr"}:
        raise ValueError(f"Unknown selection mode: {selection!r}")
    if selection == "mmr" and word_vectors is None:
        raise ValueError("selection='mmr' requires word_vectors")
    cluster_counters: dict[ClusterT, Counter[str]] = {}
    for cid, kws in paper_keywords.items():
        cluster = cluster_assignments.get(cid)
        if cluster is None:
            continue
        if cluster not in cluster_counters:
            cluster_counters[cluster] = Counter()
        cluster_counters[cluster].update(kws)

    scorer_for = _make_scorer_factory(
        scoring,
        cluster_counters,
        word_vectors=word_vectors,
        prior_strength=prior_strength,
    )

    out: dict[ClusterT, list[str]] = {}
    for cluster, counter in cluster_counters.items():
        score_fn = scorer_for(cluster)
        # Sort by descending score, ties broken alphabetically.
        ranked = sorted(
            ((w, score_fn(w, c)) for w, c in counter.items()),
            key=lambda kv: (-kv[1], kv[0]),
        )
        if selection == "mmr" and word_vectors is not None:
            out[cluster] = _select_mmr(
                ranked[:mmr_pool],
                top_k,
                word_vectors=word_vectors,
                lam=mmr_lambda,
            )
        else:
            out[cluster] = [w for w, _ in ranked[:top_k]]
    return out


def _select_mmr(
    ranked: list[tuple[str, float]],
    top_k: int,
    *,
    word_vectors: Mapping[str, npt.NDArray[np.float32]],
    lam: float,
) -> list[str]:
    """Exact diversity-aware K-subset selection over a small candidate pool.

    Maximises ``lam * sum(min-max-normalised relevance) - (1 - lam) *
    sum(pairwise cosine)`` by enumerating all K-subsets. Candidates
    without a vector are dropped. The returned words keep relevance
    order. Ties go to the earlier-enumerated (higher-relevance) subset.
    """
    candidates = [(w, s) for w, s in ranked if w in word_vectors]
    if len(candidates) <= top_k:
        return [w for w, _ in candidates]
    scores = [s for _, s in candidates]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    rel = {w: ((s - lo) / span if span > 0 else 1.0) for w, s in candidates}
    words = [w for w, _ in candidates]
    best: tuple[str, ...] = ()
    best_obj = float("-inf")
    for subset in itertools.combinations(words, top_k):
        relevance = sum(rel[w] for w in subset)
        redundancy = sum(
            float(word_vectors[a] @ word_vectors[b])
            for a, b in itertools.combinations(subset, 2)
        )
        obj = lam * relevance - (1.0 - lam) * redundancy
        if obj > best_obj:
            best_obj = obj
            best = subset
    return list(best)


def _make_scorer_factory[ClusterT: Hashable](
    scoring: str,
    cluster_counters: Mapping[ClusterT, Counter[str]],
    *,
    word_vectors: Mapping[str, npt.NDArray[np.float32]] | None,
    prior_strength: float,
) -> Callable[[ClusterT], Callable[[str, int], float]]:
    """Return a per-cluster scorer factory for the given scoring mode.

    ``frequency`` and ``tfidf`` ignore the cluster argument; ``logodds``
    closes over rest-of-corpus counts per cluster.
    """
    if scoring == "frequency":
        return lambda _cluster: _score_frequency
    if scoring == "tfidf":
        tfidf_fn = _score_tfidf(cluster_counters)
        return lambda _cluster: tfidf_fn
    if scoring == "logodds":
        return _make_logodds_factory(cluster_counters, prior_strength)
    if scoring == "embedding":
        if word_vectors is None:
            raise ValueError("scoring='embedding' requires word_vectors")
        return _make_embedding_factory(cluster_counters, word_vectors)
    raise ValueError(f"Unknown scoring mode: {scoring!r}")


def _make_logodds_factory[ClusterT: Hashable](
    cluster_counters: Mapping[ClusterT, Counter[str]],
    prior_strength: float,
) -> Callable[[ClusterT], Callable[[str, int], float]]:
    """Log-odds ratio with informative Dirichlet prior (Monroe et al. 2008).

    For word w in cluster i vs the pooled remainder j, the z-score of the
    log-odds difference, with per-word prior mass proportional to the
    word's corpus-wide frequency.
    """
    global_counts: Counter[str] = Counter()
    for counter in cluster_counters.values():
        global_counts.update(counter)
    n_total = sum(global_counts.values())
    cluster_totals = {c: sum(cnt.values()) for c, cnt in cluster_counters.items()}

    def factory(cluster: ClusterT) -> Callable[[str, int], float]:
        n_i = cluster_totals[cluster]
        n_j = n_total - n_i

        def _score(word: str, count: int) -> float:
            g = global_counts[word]
            alpha_w = prior_strength * g / n_total
            y_j = g - count
            num_i = count + alpha_w
            den_i = max(n_i + prior_strength - count - alpha_w, 1e-9)
            num_j = y_j + alpha_w
            den_j = max(n_j + prior_strength - y_j - alpha_w, 1e-9)
            delta = log(num_i / den_i) - log(num_j / den_j)
            var = 1.0 / num_i + 1.0 / num_j
            return delta / sqrt(var)

        return _score

    return factory


def _make_embedding_factory[ClusterT: Hashable](
    cluster_counters: Mapping[ClusterT, Counter[str]],
    word_vectors: Mapping[str, npt.NDArray[np.float32]],
) -> Callable[[ClusterT], Callable[[str, int], float]]:
    """Cosine-to-centroid scorer; centroid is count-weighted and unit-norm.

    Words without a vector score -1.0 (below any cosine), so they never
    win a label slot.
    """

    def factory(cluster: ClusterT) -> Callable[[str, int], float]:
        counter = cluster_counters[cluster]
        vecs: list[npt.NDArray[np.float32]] = []
        weights: list[float] = []
        for w, c in counter.items():
            v = word_vectors.get(w)
            if v is not None:
                vecs.append(v)
                weights.append(float(c))
        if not vecs:
            return lambda _word, _count: 0.0
        centroid = np.average(np.stack(vecs), axis=0, weights=weights)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm

        def _score(word: str, _count: int) -> float:
            v = word_vectors.get(word)
            return float(v @ centroid) if v is not None else -1.0

        return _score

    return factory


def _score_frequency(_word: str, count: int) -> float:
    """Raw in-cluster count."""
    return float(count)


def _score_tfidf[ClusterT: Hashable](
    cluster_counters: Mapping[ClusterT, Counter[str]],
) -> Callable[[str, int], float]:
    """Return a TF-IDF scorer closed over corpus-level document frequencies.

    The "document frequency" for a word is the number of clusters that
    contain it (any non-zero count). IDF is the natural log of
    ``N_clusters / df``, with smoothing to avoid log(0) for words
    appearing in every cluster.
    """
    n_clusters = len(cluster_counters)
    cluster_df: Counter[str] = Counter()
    for counter in cluster_counters.values():
        for word in counter:
            cluster_df[word] += 1

    def _score(word: str, count: int) -> float:
        df = cluster_df.get(word, 0)
        if df == 0:
            return 0.0
        idf = log(n_clusters / df) if n_clusters > df else log(1.0 + 1.0 / df)
        return float(count) * idf

    return _score


def deduplicate_consecutive(words: Iterable[str]) -> list[str]:
    """Drop consecutive duplicates from a keyword sequence.

    Helpful when a parent and child cluster both have the same top-K
    keyword sequence — the duplicated label adds no information at decode
    time and just inflates the trie.
    """
    out: list[str] = []
    prev: str | None = None
    for w in words:
        if w == prev:
            continue
        out.append(w)
        prev = w
    return out


__all__ = [
    "EmbedFn",
    "aggregate_cluster_labels",
    "deduplicate_consecutive",
    "extract_paper_keywords",
    "extract_paper_keywords_from_texts",
    "make_minilm_embedder",
    "merge_plural_variants",
]
