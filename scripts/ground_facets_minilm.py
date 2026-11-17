"""Compare facet grounding for two extractions using MiniLM embeddings.

Grounding score of a facet item = max cosine similarity between the item and
any paragraph of its paper (how well the item is supported by the paper text).
Uses MiniLM-L12-v2 (local, free) — never OpenAI embeddings. Reports mean
grounding per facet for each extraction, side by side.

Usage:
    uv run python scripts/ground_facets_minilm.py \
        --splits data/s2orc-nlp/splits-connected-1k \
        --a .../processed-gpt5mini-core/facets.jsonl.zst --a-label gpt5mini-core \
        --b .../processed-qwen3-32b/facets.jsonl.zst   --b-label qwen3-32b-core
"""

import statistics
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import typer

from genret.paper_parsing import PaperGraph, Section
from genret.training.cluster_keyword_labels import EmbedFn, make_minilm_embedder
from genret.types import CorpusId
from genret.utils import iter_jsonl, make_app, open_auto, resolve_jsonl

_FACETS = ("problems", "methods", "results", "contributions")

app = make_app("MiniLM grounding comparison for two facet extractions")


def _paragraphs(section: Section) -> list[str]:
    out = list(section.paragraphs)
    for sub in section.subsections:
        out.extend(_paragraphs(sub))
    return out


def load_paragraphs(splits_dir: Path) -> dict[CorpusId, list[str]]:
    """Map corpusid -> list of paragraph texts (abstract + all sections)."""
    graphs = resolve_jsonl(splits_dir / "processed" / "paper_graphs.jsonl.gz")
    out: dict[CorpusId, list[str]] = {}
    with open_auto(graphs, "r") as f:
        for line in f:
            if not line.strip():
                continue
            p = PaperGraph.model_validate_json(line)
            paras: list[str] = [p.abstract] if p.abstract else []
            for s in p.sections:
                paras.extend(_paragraphs(s))
            out[p.corpusid] = [t for t in paras if t.strip()]
    return out


def load_facets(path: Path) -> dict[CorpusId, dict[str, list[str]]]:
    """Load a facets JSONL into {corpusid: {facet: items}}."""
    out: dict[CorpusId, dict[str, list[str]]] = {}
    for rec in iter_jsonl(path):
        out[CorpusId(rec["corpusid"])] = {f: list(rec.get(f, [])) for f in _FACETS}
    return out


def _spans(
    texts_per_id: dict[CorpusId, list[str]], common: list[CorpusId]
) -> tuple[list[str], dict[CorpusId, tuple[int, int]]]:
    """Flatten per-id text lists into one list + per-id (start, end) spans."""
    flat: list[str] = []
    spans: dict[CorpusId, tuple[int, int]] = {}
    for cid in common:
        start = len(flat)
        flat.extend(texts_per_id.get(cid, []))
        spans[cid] = (start, len(flat))
    return flat, spans


def grounding_by_facet(
    facets: dict[CorpusId, dict[str, list[str]]],
    para_vecs: npt.NDArray[np.float32],
    para_spans: dict[CorpusId, tuple[int, int]],
    embed: EmbedFn,
    common: list[CorpusId],
) -> dict[str, list[float]]:
    """Per-facet list of item grounding scores (max cosine to any paragraph).

    All facet items are embedded in one batched call; grounding is computed by
    slicing the precomputed paragraph and item matrices per paper.
    """
    scores: dict[str, list[float]] = {f: [] for f in _FACETS}
    for f in _FACETS:
        items_per_id = {cid: facets[cid][f] for cid in common}
        flat, spans = _spans(items_per_id, common)
        if not flat:
            continue
        item_vecs: npt.NDArray[np.float32] = embed(flat)
        for cid in common:
            ps, pe = para_spans[cid]
            ist, ien = spans[cid]
            if pe <= ps or ien <= ist:
                continue
            sims = item_vecs[ist:ien] @ para_vecs[ps:pe].T  # cosine (unit-norm)
            scores[f].extend(sims.max(axis=1).tolist())
    return scores


@app.command()
def main(
    splits: Annotated[Path, typer.Option("--splits")],
    a: Annotated[Path, typer.Option("--a")],
    b: Annotated[Path, typer.Option("--b")],
    a_label: Annotated[str, typer.Option("--a-label")] = "A",
    b_label: Annotated[str, typer.Option("--b-label")] = "B",
) -> None:
    """Print mean MiniLM grounding per facet for two extractions."""
    print("Loading paragraphs + facets...")
    paragraphs = load_paragraphs(splits)
    fa, fb = load_facets(a), load_facets(b)
    common = sorted(set(fa) & set(fb) & set(paragraphs))
    print(f"common papers with text: {len(common)}")

    print("Loading MiniLM-L12-v2...")
    embed = make_minilm_embedder()

    print("Embedding paragraphs (once)...")
    para_flat, para_spans = _spans(paragraphs, common)
    para_vecs: npt.NDArray[np.float32] = embed(para_flat)

    print(f"Scoring {a_label}...")
    sa = grounding_by_facet(fa, para_vecs, para_spans, embed, common)
    print(f"Scoring {b_label}...")
    sb = grounding_by_facet(fb, para_vecs, para_spans, embed, common)

    def _mean(xs: list[float]) -> float:
        return statistics.mean(xs) if xs else 0.0

    print(f"\n{'facet':<14}{a_label:>18}{b_label:>18}{'Δ(B-A)':>10}")
    for f in _FACETS:
        ma, mb = _mean(sa[f]), _mean(sb[f])
        print(f"{f:<14}{ma:>18.4f}{mb:>18.4f}{mb - ma:>+10.4f}")
    alla = [x for f in _FACETS for x in sa[f]]
    allb = [x for f in _FACETS for x in sb[f]]
    print(
        f"{'OVERALL':<14}{_mean(alla):>18.4f}{_mean(allb):>18.4f}"
        f"{_mean(allb) - _mean(alla):>+10.4f}"
    )


if __name__ == "__main__":
    app()
