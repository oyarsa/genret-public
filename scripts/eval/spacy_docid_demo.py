"""Demo: spaCy noun-chunk extraction on real facet bullets.

Shows raw bullet text alongside spaCy noun chunks and the resulting
candidate DocID (joined chunks, first 12 tokens) for a random sample
of papers across all four facets.

Usage:
    uv run python scripts/eval/spacy_docid_demo.py \
        --facets data/s2orc-nlp-v2/splits-connected-10k-v2/processed/facets.jsonl.zst \
        --n 10
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import spacy

from genret.paper_parsing import FacetRecordType, facets_from_type
from genret.training.codebook_data import FACET_NAMES
from genret.training.codebook_dataset import _facet_text
from genret.utils import iter_jsonl

BULLET_RE = re.compile(r"^[\s\-*>\"\']+")


def split_bullets(text: str) -> list[str]:
    """Split multi-line facet text into individual stripped bullet strings."""
    lines: list[str] = []
    for raw in text.splitlines():
        s = BULLET_RE.sub("", raw).strip()
        if s:
            lines.append(s)
    return lines


def noun_chunk_docid(nlp: spacy.Language, bullet: str, max_tokens: int = 12) -> str:
    """Return noun-chunk keyphrases from bullet, joined and word-capped."""
    doc = nlp(bullet)
    chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
    joined = " ".join(chunks) if chunks else bullet.lower()
    # Approximate token count by whitespace split; real tokenisation happens later.
    words = joined.split()
    return " ".join(words[:max_tokens])


def main() -> None:
    """Entry point for spaCy DocID demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--facets",
        default="data/s2orc-nlp-v2/splits-connected-10k-v2/processed/facets.jsonl.zst",
        type=Path,
    )
    parser.add_argument("--n", type=int, default=10, help="Papers to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

    rng = random.Random(args.seed)
    records = list(iter_jsonl(args.facets))
    sample = rng.sample(records, min(args.n, len(records)))

    for record in sample:
        facets_obj = facets_from_type(record, FacetRecordType.STANDARD)
        print(f"\n{'=' * 70}")
        print(f"Paper: {facets_obj.corpusid}")
        for facet in FACET_NAMES:
            text = _facet_text(facets_obj, facet, None)
            if not text:
                continue
            bullets = split_bullets(text)
            if not bullets:
                continue
            print(f"\n  [{facet}]")
            for bullet in bullets:
                chunks_docid = noun_chunk_docid(nlp, bullet)
                print(f"    RAW:    {bullet[:80]}")
                print(f"    CHUNKS: {chunks_docid}")
                print()


if __name__ == "__main__":
    main()
