"""Evaluation harness for the MINDER-style three-view baseline (Variant A).

Decodes one query per view from `query_text` (title + abstract head, matching
training) and writes every beam that lands on a real paper to a JSONL dump in
the same schema `glen_evaluate.py`'s `--record-docids` writer uses --
`query/facet/rank/docid/papers/score/n_tokens/logprob_sum` -- so
`scripts/eval/minder_fusion.py` (Variant B's fusion levels) scores this dump
unchanged, `facet` here holding a view name (`title`/`substring`/
`pseudo-query`) rather than a facet name.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING

from genret._shed.transformers import decode_tokens
from genret.training.glen_decoding import GlenDecodeResult, decode_per_facet_glen
from genret.training.glen_docids import GlenIndex
from genret.training.minder_dataset import build_view_messages, query_text
from genret.training.minder_decoding import (
    PSEUDO_QUERY_FACET,
    SUBSTRING_FACET,
    TITLE_FACET,
    decode_substring_view,
)
from genret.training.minder_substrings import SubstringIndex
from genret.types import CorpusId

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def evaluate_minder_retrieval(
    *,
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizerBase,
    trie_index: GlenIndex,
    substring_index: SubstringIndex,
    queries: Sequence[CorpusId],
    titles: Mapping[CorpusId, str],
    abstracts: Mapping[CorpusId, str],
    output_path: Path,
    num_beams: int = 50,
    max_docid_tokens: int = 32,
    max_query_chars: int = 1500,
) -> None:
    """Decode all three MINDER views for every query and dump every beam.

    Args:
        model: Fine-tuned MINDER-baseline model.
        tokeniser: Matching tokeniser.
        trie_index: Two-facet `GlenIndex` (`title`, `pseudo-query`) from
            `minder_decoding.build_title_and_pseudo_query_index`.
        substring_index: Substring index over the corpus.
        queries: Query paper ids to decode.
        titles: Paper titles, keyed by corpus id.
        abstracts: Paper abstracts, keyed by corpus id.
        output_path: Destination JSONL dump.
        num_beams: Beam width per view.
        max_docid_tokens: Token cap for the title/pseudo-query views.
        max_query_chars: Abstract characters kept on the query side, matching
            `MinderDatasetConfig.max_query_chars` used at training time.
    """
    written_queries = 0
    written_rows = 0
    with output_path.open("w") as fh:
        for qid in queries:
            title = titles.get(qid, "")
            abstract = abstracts.get(qid, "")
            if not title and not abstract:
                logger.warning("Query %d has no title or abstract; skipping", qid)
                continue
            query = query_text(title, abstract, max_chars=max_query_chars)

            title_results = decode_per_facet_glen(
                model=model,
                tokeniser=tokeniser,
                messages=build_view_messages(TITLE_FACET, query),
                facet=TITLE_FACET,
                index=trie_index,
                num_beams=num_beams,
                max_new_tokens=max_docid_tokens,
            )
            pseudo_query_results = decode_per_facet_glen(
                model=model,
                tokeniser=tokeniser,
                messages=build_view_messages(PSEUDO_QUERY_FACET, query),
                facet=PSEUDO_QUERY_FACET,
                index=trie_index,
                num_beams=num_beams,
                max_new_tokens=max_docid_tokens,
            )
            substring_results = decode_substring_view(
                model=model,
                tokeniser=tokeniser,
                messages=build_view_messages(SUBSTRING_FACET, query),
                index=substring_index,
                num_beams=num_beams,
                max_new_tokens=substring_index.max_span_tokens,
            )

            for result in (*title_results, *pseudo_query_results, *substring_results):
                _write_result(fh, qid, result, tokeniser)
                written_rows += 1
            written_queries += 1

    logger.info(
        "MINDER eval: %d queries, %d beam rows -> %s",
        written_queries,
        written_rows,
        output_path,
    )


def _write_result(
    fh: IO[str],
    qid: CorpusId,
    result: GlenDecodeResult,
    tokeniser: PreTrainedTokenizerBase,
) -> None:
    """Write one decoded beam as a dump row, matching `--record-docids`'s schema."""
    docid = decode_tokens(
        tokeniser, list(result.docid_tokens), skip_special_tokens=True
    )
    row = {
        "query": int(qid),
        "facet": result.facet,
        "rank": result.rank,
        "docid": docid,
        "papers": [int(p) for p in result.paper_ids],
        "score": round(result.score, 6),
        "n_tokens": len(result.docid_tokens),
        "logprob_sum": (
            None if result.logprob_sum is None else round(result.logprob_sum, 6)
        ),
    }
    fh.write(json.dumps(row) + "\n")


__all__ = ["evaluate_minder_retrieval"]
