"""MINDER-style multiview training dataset (Variant A of the MINDER baseline).

Builds one dataset covering MINDER's three identifier views (Li et al., ACL
2023): the paper's title, a substring of its abstract, and a pseudo-query. Our
facet bullets serve as the pseudo-query view -- they are LLM-generated
descriptions of the paper's content, which is what MINDER's query-generation
model produces -- pooled across facets, because MINDER's views are identifier
*types* and carry no facet.

Deliberate design points, all recorded in
`docs/plans/20260725-minder-baseline.md`:

- **One row per (pair, view), query side = title + abstract.** MINDER issues one
  decode per view from one query text, so training inputs match: the source
  paper's title plus the head of its cleaned abstract. This is *not* our
  per-bullet decomposition, and it gives the baseline roughly 5.6x less
  query-side evidence than our deployed system. Use `query_text` for both
  training and evaluation so the two can never drift -- prompt drift between
  train and eval is exactly the bug `20260701-glen-eval-prompt-mismatch.md`
  documents.
- **Mix 3 : 10 : 5** (title : substring : pseudo-query), the published portion.
- **Substring targets are selected by the released code's own algorithm**,
  ported from `extract_spans` in `scripts/training/make_generated_dataset.py`
  (`github.com/liyongqi67/MINDER`), not by a word-overlap approximation of
  the paper's one-sentence gloss ("we only keep those substrings with a high
  character overlap with the query"). That gloss undersells a real
  algorithm: every 3-word window of the query is scored against the first
  3-word window ("anchor") of every candidate span by fuzzy character-level
  string similarity, scores accumulate per candidate, a temperature-1.0
  softmax turns them into weights, and a candidate is drawn **with
  replacement, weighted, no hard threshold** -- "keep" in the paper's prose
  means "weight highly", not "discard the rest". Candidates whose anchor is
  a stopword never score (ported verbatim from the source's crude banned-word
  list) and so are never drawn. See `_score_candidates_fuzzy` /
  `_sample_spans_fuzzy`.
- **Exposure comes from MINDER's unsupervised rows only** -- input is one of a
  paper's own pseudo-queries, target one of its identifiers -- so every corpus
  paper appears as a target. No `K_min` floor and no graph-pseudo-gold: those
  are our own graph-derived augmentations and belong on our side of the
  comparison.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from genret._shed.transformers import decode_tokens
from genret.paper_parsing import FacetRecordType, Facets, facets_from_type
from genret.training.codebook_data import FACET_NAMES
from genret.training.data import TrainingExample
from genret.training.glen_title_index import normalise_title
from genret.training.minder_substrings import SubstringIndex, clean_abstract
from genret.types import CorpusId
from genret.utils import iter_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_STOPWORD_ANCHORS = frozenset(
    {
        "the",
        "The",
        "to",
        "a",
        "A",
        "an",
        "An",
        "he",
        "He",
        "his",
        "His",
        "him",
        "He's",
        "she",
        "She",
        "her",
        "Her",
        "she's",
        "She's",
        "it",
        "It",
        "its",
        "Its",
        "it's",
        "It's",
        "and",
        "And",
        "or",
        "Or",
        "this",
        "This",
        "that",
        "That",
        "those",
        "Those",
        "these",
        "These",
        '"',
        '""',
        "'",
        "''",
    }
)
"""Words that may not anchor a substring-view span.

Ported verbatim from `scripts/training/make_generated_dataset.py::banned` in
the released MINDER code, which `extract_spans`'s `span_iterator` checks
against lowercased words -- so only this set's lowercase members are ever
reachable there, matching the original's own behaviour rather than "fixing"
the apparent redundancy.
"""

VIEWS = ("title", "substring", "pseudo-query")
"""MINDER's three identifier views, in the paper's order."""

_VIEW_TARGET = {
    "title": "title",
    "substring": "passage from the abstract",
    "pseudo-query": "descriptive bullet",
}
"""What the model is asked to generate, per view, for the system prompt."""

_SYSTEM_PROMPT = (
    "You are a generative retrieval system. Given a research paper, generate "
    "the {target} of a relevant paper."
)
"""Mirrors the per-facet prompt's shape; the view replaces the facet."""

DEFAULT_MIX = {"title": 3, "substring": 10, "pseudo-query": 5}
"""Rows per gold pair per view: the paper's 3 : 10 : 5 portion."""


def query_text(title: str, abstract: str, *, max_chars: int = 1500) -> str:
    """Render the query side of a MINDER-style row.

    Shared by training and evaluation on purpose: a divergence here is a silent
    train/eval prompt mismatch.

    Args:
        title: Source paper title, raw.
        abstract: Source paper abstract, raw (cleaned here).
        max_chars: Abstract characters kept, matching the V1 title-DocID runs.

    Returns:
        The rendered query text.
    """
    cleaned = clean_abstract(abstract)[:max_chars]
    if not cleaned:
        return f"Title: {title}".strip()
    return f"Title: {title}\nAbstract: {cleaned}"


def build_view_messages(view: str, text: str) -> list[dict[str, str]]:
    """Build chat messages for one view.

    MINDER appends a literal marker (`" || title"`) to the query; we put the
    view in the system prompt instead, which is the same signal in the shape
    this codebase's prompts already take. Noted as a deviation in the plan.

    Args:
        view: One of `VIEWS`.
        text: Rendered query text.

    Returns:
        System + user messages.

    Raises:
        ValueError: If `view` is not one of `VIEWS`.
    """
    if view not in _VIEW_TARGET:
        raise ValueError(f"unknown view {view!r}, expected one of {VIEWS}")
    return [
        {"role": "system", "content": _SYSTEM_PROMPT.format(target=_VIEW_TARGET[view])},
        {"role": "user", "content": text},
    ]


@dataclass(frozen=True, kw_only=True)
class MinderDatasetConfig:
    """Configuration for the MINDER-style three-view dataset."""

    pairs_path: Path
    """Training pairs JSONL; only `retrieval` rows are used."""
    facets_path: Path
    """JSONL with per-paper facet bullets (the pseudo-query view)."""
    paper_graphs_path: Path
    """JSONL with `corpusid` and `title`."""
    abstracts_path: Path
    """JSONL with `corpusid` and `abstract`."""
    facets_type: FacetRecordType = FacetRecordType.STANDARD
    split_ids: set[CorpusId] | None = None
    """If set, keep only pairs whose source paper is in this split."""
    mix: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_MIX))
    """Rows per gold pair per view."""
    unsupervised_blocks: int = 1
    """Times the whole `mix` is emitted per corpus paper, unsupervised.

    MINDER's exposure mechanism feeds a paper's own pseudo-query as input and
    one of its own identifiers as target. It is the recipe's only guarantee
    against the zero-exposure cliff -- no `K_min`, no graph fallback -- so the
    block is emitted **per view at the mix ratio** rather than by drawing views
    at random: with a handful of random draws a paper misses a view with high
    probability (at 3 draws and a 3:10:5 mix, 58% of papers get no title row),
    and a view a paper is never a target in is a view it can never be
    retrieved by.

    One block is 18 rows per paper at the default mix, ~7% of the dataset.
    MINDER's own regime has unsupervised rows *dominating* (one per passage
    over 21M passages against ~79k labelled pairs), which cannot be
    reproduced at 1k without swamping the supervised signal; their Table 4
    puts the mechanism's worth at +1.2 hits@5, so coverage rather than volume
    is what it buys.
    """
    max_query_chars: int = 1500
    """Abstract characters kept on the query side."""
    span_tokens: int = 12
    """Length of substring-view target spans, in tokens."""
    fuzzy_temperature: float = 1.0
    """Softmax temperature for substring-span sampling, matching the source
    default in `extract_spans`."""


def load_titles(path: Path) -> dict[CorpusId, str]:
    """Load raw titles keyed by paper."""
    titles: dict[CorpusId, str] = {}
    for record in iter_jsonl(path):
        title = record.get("title")
        if isinstance(title, str) and title.strip():
            titles[CorpusId(int(record["corpusid"]))] = title.strip()
    return titles


def load_abstracts(path: Path, keep: set[CorpusId]) -> dict[CorpusId, str]:
    """Load raw abstracts for `keep`, first occurrence winning."""
    abstracts: dict[CorpusId, str] = {}
    for record in iter_jsonl(path):
        cid = CorpusId(int(record["corpusid"]))
        if cid in keep and cid not in abstracts:
            abstracts[cid] = record.get("abstract") or ""
    return abstracts


def _ordered_words(text: str) -> list[str]:
    """Whitespace-split words, case and punctuation preserved.

    Matches MINDER's own tokenisation for overlap scoring (`text.split()`,
    lowercased only at the comparison site) rather than an alphanumeric-only
    cleanup -- the released code keeps trailing punctuation attached to words.
    """
    return text.split()


@dataclass(frozen=True, kw_only=True)
class SpanCandidate:
    """One substring-view target span."""

    text: str
    """Decoded span text; re-tokenises to the index's token slice."""


def enumerate_span_candidates(
    index: SubstringIndex,
    tokeniser: PreTrainedTokenizerBase,
    *,
    span_tokens: int = 12,
) -> dict[CorpusId, list[SpanCandidate]]:
    """Enumerate every usable target span of every paper, once.

    A candidate is kept only if its decoded text re-tokenises to exactly the
    index's token slice. Otherwise the trainer would teach a token sequence the
    index cannot generate. In practice only spans that cut a multi-byte
    character fail.

    Args:
        index: Built substring index; supplies the token stream and boundaries.
        tokeniser: Same tokeniser the index was built with.
        span_tokens: Span length in tokens.

    Returns:
        Candidates per paper, in stream order.
    """
    stream = index.tokens[: index.n_tokens]
    starts = index.doc_starts
    per_paper: dict[CorpusId, list[SpanCandidate]] = {}
    dropped = 0
    for offset, corpus_id in enumerate(index.corpus_ids):
        begin = int(starts[offset])
        # One sentinel sits between papers, so the text ends one before the
        # next start (or one before the stream end for the last paper).
        end = (
            int(starts[offset + 1]) - 1
            if offset + 1 < starts.shape[0]
            else index.n_tokens - 1
        )
        candidates: list[SpanCandidate] = []
        for position in range(begin, max(begin, end - span_tokens + 1)):
            span = tuple(int(t) for t in stream[position : position + span_tokens])
            text = decode_tokens(tokeniser, list(span), skip_special_tokens=False)
            if tuple(tokeniser.encode(text, add_special_tokens=False)) != span:
                dropped += 1
                continue
            candidates.append(SpanCandidate(text=text))
        per_paper[corpus_id] = candidates
    total = sum(len(v) for v in per_paper.values())
    logger.info(
        "Span candidates: %d over %d papers (%d dropped, failed round-trip)",
        total,
        len(per_paper),
        dropped,
    )
    return per_paper


def _anchor_words(candidate: SpanCandidate, *, n: int = 3) -> list[str]:
    """First `n` words of a candidate, lowercased -- its match-scoring anchor.

    `extract_spans` scores a passage start position by its first `ngrams`
    words; a `SpanCandidate` already fixes both a start position and a length,
    so its own leading words stand in for that position.
    """
    return [w.lower() for w in _ordered_words(candidate.text)[:n]]


def _score_candidates_fuzzy(
    query_words: Sequence[str], candidates: Sequence[SpanCandidate], *, ngrams: int = 3
) -> list[float]:
    """Score each candidate by summed fuzzy overlap with every query window.

    Ports `extract_spans` from `scripts/training/make_generated_dataset.py`
    in the released MINDER code: every `ngrams`-word window of the query
    (unfiltered, matching the source) is compared against each candidate's
    anchor with `rapidfuzz.fuzz.ratio` (a character-level fuzzy similarity;
    MINDER's own `fuzzywuzzy` falls back to the same family of algorithm --
    `difflib.SequenceMatcher` -- since its `requirements.txt` does not pin
    `python-Levenshtein`), scaled to `[0, 1]` and accumulated. A candidate
    whose anchor starts with a stopword (`_STOPWORD_ANCHORS`) never scores,
    matching the source's `span_iterator` pre-filter on passage positions.

    Args:
        query_words: Query text, whitespace-split, order preserved.
        candidates: Target paper's candidate spans.
        ngrams: Window width.

    Returns:
        One accumulated score per candidate, same order as `candidates`,
        `0.0` for stopword-anchored candidates.
    """
    from rapidfuzz import fuzz  # noqa: PLC0415

    query_ngrams = [
        " ".join(query_words[i : i + ngrams]).lower() for i in range(len(query_words))
    ]
    scores = [0.0] * len(candidates)
    for idx, candidate in enumerate(candidates):
        anchor_words = _anchor_words(candidate, n=ngrams)
        if not anchor_words or anchor_words[0] in _STOPWORD_ANCHORS:
            continue
        anchor = " ".join(anchor_words)
        scores[idx] = sum(fuzz.ratio(qn, anchor) / 100.0 for qn in query_ngrams)
    return scores


def _sample_spans_fuzzy(
    candidates: Sequence[SpanCandidate],
    scores: Sequence[float],
    *,
    count: int,
    temperature: float,
    rng: random.Random,
) -> list[str]:
    """Weighted-with-replacement sample over every candidate, no hard cutoff.

    Ports `extract_spans`'s sampling tail: scores become a temperature-softmax
    distribution and `count` positions are drawn from it, so a low-scoring
    candidate is unlikely but never impossible -- unlike a top-quantile filter,
    which discards it outright. If every candidate scores zero (e.g. every
    anchor is a stopword), the source falls back to a fixed position; the
    equivalent here is the first candidate, repeated.

    Args:
        candidates: Target paper's candidate spans.
        scores: `_score_candidates_fuzzy` output, same order.
        count: Number of spans to draw.
        temperature: Softmax temperature (the source's default is 1.0).
        rng: Seeded RNG.

    Returns:
        `count` span texts, possibly repeated.
    """
    if not candidates:
        return []
    if not any(s > 0.0 for s in scores):
        return [candidates[0].text] * count
    weights = [math.exp(s / temperature) for s in scores]
    total = sum(weights)
    weights = [w / total for w in weights]
    chosen = rng.choices(candidates, weights=weights, k=count)
    return [c.text for c in chosen]


def select_spans(
    query: str,
    candidates: Sequence[SpanCandidate],
    *,
    count: int,
    temperature: float,
    rng: random.Random,
) -> list[str]:
    """Score and sample `count` substring-view targets for one query.

    Combines `_score_candidates_fuzzy` and `_sample_spans_fuzzy`; the single
    entry point both the supervised and unsupervised paths use, so a
    (query, target) pair and a paper's own (pseudo-query, self) pair are
    scored identically -- matching the released code, where `extract_spans`
    is the same function in both `make_supervised_dpr_dataset.py` and
    `make_generated_dataset.py`.

    Args:
        query: Rendered query text (a paper's `query_text`, or a bullet).
        candidates: Target paper's candidate spans.
        count: Number of spans to draw.
        temperature: Softmax temperature.
        rng: Seeded RNG.

    Returns:
        `count` span texts, possibly repeated.
    """
    scores = _score_candidates_fuzzy(_ordered_words(query), candidates)
    return _sample_spans_fuzzy(
        candidates, scores, count=count, temperature=temperature, rng=rng
    )


def bullets_of(facets: Facets) -> list[str]:
    """Pool a paper's facet bullets across facets, in facet order."""
    pooled: list[str] = []
    for facet in FACET_NAMES:
        for item in getattr(facets, facet):
            if text := str(item).strip():
                pooled.append(text)
    return pooled


def build_minder_examples(
    config: MinderDatasetConfig,
    *,
    index: SubstringIndex,
    tokeniser: PreTrainedTokenizerBase,
    rng: random.Random,
) -> list[TrainingExample]:
    """Build the three-view example list.

    Args:
        config: Dataset configuration.
        index: Substring index over the same corpus.
        tokeniser: Tokeniser the index was built with.
        rng: Seeded RNG for span and bullet sampling.

    Returns:
        Shuffled training examples across all three views.
    """
    facets_map: dict[CorpusId, Facets] = {}
    for raw in iter_jsonl(config.facets_path):
        parsed = facets_from_type(raw, config.facets_type)
        facets_map[parsed.corpusid] = parsed
    titles = load_titles(config.paper_graphs_path)
    abstracts = load_abstracts(config.abstracts_path, set(index.corpus_ids))
    spans = enumerate_span_candidates(index, tokeniser, span_tokens=config.span_tokens)

    # MINDER has no facets, so a pair replicated per facet is one pair.
    pairs: list[tuple[CorpusId, CorpusId]] = []
    seen_pairs: set[tuple[CorpusId, CorpusId]] = set()
    for row in iter_jsonl(config.pairs_path):
        if str(row.get("kind", "")) != "retrieval":
            continue
        paper_a = CorpusId(int(row["paper_a"]))
        paper_b = CorpusId(int(row["paper_b"]))
        if config.split_ids is not None and paper_a not in config.split_ids:
            continue
        if (paper_a, paper_b) in seen_pairs:
            continue
        seen_pairs.add((paper_a, paper_b))
        pairs.append((paper_a, paper_b))
    logger.info("MINDER dataset: %d distinct gold pairs after facet dedup", len(pairs))

    examples: list[TrainingExample] = []
    skipped = 0
    for paper_a, paper_b in pairs:
        query = _render_query(paper_a, titles, abstracts, config)
        if not query:
            skipped += 1
            continue
        examples.extend(
            _rows_for_target(
                query=query,
                target=paper_b,
                titles=titles,
                facets_map=facets_map,
                spans=spans,
                config=config,
                rng=rng,
            )
        )

    n_supervised = len(examples)
    examples.extend(
        _unsupervised_rows(
            facets_map=facets_map,
            titles=titles,
            spans=spans,
            corpus_ids=index.corpus_ids,
            config=config,
            rng=rng,
        )
    )
    rng.shuffle(examples)
    logger.info(
        "MINDER dataset: %d rows (%d supervised, %d unsupervised), %d pairs skipped",
        len(examples),
        n_supervised,
        len(examples) - n_supervised,
        skipped,
    )
    return examples


def _render_query(
    paper: CorpusId,
    titles: Mapping[CorpusId, str],
    abstracts: Mapping[CorpusId, str],
    config: MinderDatasetConfig,
) -> str:
    """Render a paper's query text, or empty when it has neither field."""
    return query_text(
        titles.get(paper, ""),
        abstracts.get(paper, ""),
        max_chars=config.max_query_chars,
    )


def _rows_for_target(
    *,
    query: str,
    target: CorpusId,
    titles: Mapping[CorpusId, str],
    facets_map: Mapping[CorpusId, Facets],
    spans: Mapping[CorpusId, list[SpanCandidate]],
    config: MinderDatasetConfig,
    rng: random.Random,
) -> list[TrainingExample]:
    """Emit the per-view rows for one (query, target) pair at the mix ratio."""
    rows: list[TrainingExample] = []

    if (title := titles.get(target)) and (n := config.mix.get("title", 0)):
        messages = build_view_messages("title", query)
        rows.extend(
            TrainingExample(
                prompt_messages=messages,
                target_docid=normalise_title(title),
                kind="title",
            )
            for _ in range(n)
        )

    if n := config.mix.get("substring", 0):
        messages = build_view_messages("substring", query)
        rows.extend(
            TrainingExample(
                prompt_messages=messages, target_docid=text, kind="substring"
            )
            for text in select_spans(
                query,
                spans.get(target, []),
                count=n,
                temperature=config.fuzzy_temperature,
                rng=rng,
            )
        )

    facets_b = facets_map.get(target)
    pooled = bullets_of(facets_b) if facets_b is not None else list[str]()
    if pooled and (n := config.mix.get("pseudo-query", 0)):
        messages = build_view_messages("pseudo-query", query)
        rows.extend(
            TrainingExample(
                prompt_messages=messages,
                target_docid=rng.choice(pooled),
                kind="pseudo-query",
            )
            for _ in range(n)
        )
    return rows


def _unsupervised_rows(
    *,
    facets_map: Mapping[CorpusId, Facets],
    titles: Mapping[CorpusId, str],
    spans: Mapping[CorpusId, list[SpanCandidate]],
    corpus_ids: Sequence[CorpusId],
    config: MinderDatasetConfig,
    rng: random.Random,
) -> list[TrainingExample]:
    """Build MINDER's unsupervised exposure rows.

    "The model takes as input a random pseudo-query and generates the
    corresponding passage's identifiers." Every corpus paper has
    pseudo-queries, so every paper appears as a target in every view -- this
    is the recipe's only guarantee against the zero-exposure cliff, replacing
    our `K_min` floor. Views are emitted at the mix ratio rather than sampled
    (see `MinderDatasetConfig.unsupervised_blocks`), but the row *shape*
    within each view is ported from `preprocess_file` in
    `scripts/training/make_generated_dataset.py`, which is not uniform across
    views:

    - **title, pseudo-query**: each row independently draws its own input
      bullet (with replacement).
    - **substring**: one input bullet is drawn **once per paper** and shared
      across every substring row for that paper, all scored and sampled
      together in a single `select_spans` call -- matching the source's one
      `extract_spans(text, a, num_samples, ...)` call per document, not a
      fresh draw per row.
    - **pseudo-query target selection is not "any other bullet"**: the source
      scans the paper's own pseudo-query list in order and takes the first
      one that is not string-identical to the chosen input
      (`while select_querys[sampled] == querys[num]: num += 1`), falling back
      to the input itself if every bullet is identical to it. Ported as a
      scan over `pooled` in its stored facet order.
    """
    rows: list[TrainingExample] = []
    for paper in corpus_ids:
        facets = facets_map.get(paper)
        if facets is None:
            continue
        pooled = bullets_of(facets)
        if not pooled:
            continue

        for _ in range(config.unsupervised_blocks):
            if title := titles.get(paper):
                for _row in range(config.mix.get("title", 0)):
                    prompt = rng.choice(pooled)
                    rows.append(
                        TrainingExample(
                            prompt_messages=build_view_messages("title", prompt),
                            target_docid=normalise_title(title),
                            kind="unsupervised-title",
                        )
                    )

            n_substring = config.mix.get("substring", 0)
            candidates = spans.get(paper, [])
            if n_substring and candidates:
                prompt = rng.choice(pooled)
                targets = select_spans(
                    prompt,
                    candidates,
                    count=n_substring,
                    temperature=config.fuzzy_temperature,
                    rng=rng,
                )
                rows.extend(
                    TrainingExample(
                        prompt_messages=build_view_messages("substring", prompt),
                        target_docid=target,
                        kind="unsupervised-substring",
                    )
                    for target in targets
                )

            if len(pooled) > 1:
                for _row in range(config.mix.get("pseudo-query", 0)):
                    prompt = rng.choice(pooled)
                    target = next((b for b in pooled if b != prompt), prompt)
                    rows.append(
                        TrainingExample(
                            prompt_messages=build_view_messages("pseudo-query", prompt),
                            target_docid=target,
                            kind="unsupervised-pseudo-query",
                        )
                    )
    return rows


class MinderDataset:
    """Dataset of MINDER-style multiview examples."""

    def __init__(
        self,
        config: MinderDatasetConfig,
        *,
        index: SubstringIndex,
        tokeniser: PreTrainedTokenizerBase,
        rng: random.Random,
    ) -> None:
        """Build the example list.

        Args:
            config: Dataset configuration.
            index: Substring index over the same corpus.
            tokeniser: Tokeniser the index was built with.
            rng: Seeded RNG.
        """
        self._examples = build_minder_examples(
            config, index=index, tokeniser=tokeniser, rng=rng
        )

    def __len__(self) -> int:
        """Return the number of training examples."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        """Return the training example at `idx`."""
        return self._examples[idx]
