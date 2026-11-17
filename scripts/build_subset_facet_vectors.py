"""Subset 10k per-bullet MiniLM embeddings to 1k per-facet mean-pooled vectors.

Produces ``facet_graph_{facet}.npz`` files with ``corpusids`` and
``corpus_vectors`` keys in the format ``build-codebooks`` expects.

Usage:
    uv run python scripts/build_subset_facet_vectors.py \
        --src-embeddings data/s2orc-nlp/splits-connected-10k/processed/facet_embeddings_minilm.npz \
        --target-facets data/s2orc-nlp/splits-connected-1k/processed/facets.jsonl.zst \
        --output-dir data/s2orc-nlp/splits-connected-1k/processed/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt

from genret.processing.facet_evaluation import FACET_NAMES
from genret.utils import iter_jsonl


def main() -> None:
    """Build per-facet mean-pooled vectors for a subset corpus from a global per-bullet file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-embeddings", type=Path, required=True)
    parser.add_argument("--target-facets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    print(f"Loading source embeddings from {args.src_embeddings}")
    src = np.load(args.src_embeddings, allow_pickle=True)
    print(f"  keys: {len(src.files)} papers")

    print(f"Reading target facets from {args.target_facets}")
    target_counts: dict[int, dict[str, int]] = {}
    for row in iter_jsonl(args.target_facets):
        cid = int(row["corpusid"])
        target_counts[cid] = {name: len(row.get(name, [])) for name in FACET_NAMES}
    print(f"  target papers: {len(target_counts)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_facet_vectors: dict[str, list[npt.NDArray[np.float32]]] = {
        name: [] for name in FACET_NAMES
    }
    per_facet_ids: dict[str, list[int]] = {name: [] for name in FACET_NAMES}
    skipped = 0
    for cid, counts in target_counts.items():
        key = str(cid)
        if key not in src.files:
            skipped += 1
            continue
        per_bullet = src[key]
        offset = 0
        for name in FACET_NAMES:
            n = counts[name]
            if n == 0:
                offset += n
                continue
            bullets = per_bullet[offset : offset + n]
            if bullets.shape[0] != n:
                raise ValueError(
                    f"corpusid {cid}: facet {name} expected {n} bullets, got {bullets.shape[0]}"
                )
            mean_vec = bullets.mean(axis=0).astype(np.float32)
            per_facet_vectors[name].append(mean_vec)
            per_facet_ids[name].append(cid)
            offset += n

    if skipped:
        print(f"  skipped {skipped} papers not present in source embeddings")

    for name in FACET_NAMES:
        ids = np.array(per_facet_ids[name], dtype=np.int64)
        vecs = np.stack(per_facet_vectors[name], axis=0).astype(np.float32)
        out_path = args.output_dir / f"facet_graph_{name}.npz"
        np.savez(out_path, corpusids=ids, corpus_vectors=vecs)
        print(f"  wrote {out_path}: {vecs.shape}")


if __name__ == "__main__":
    main()
