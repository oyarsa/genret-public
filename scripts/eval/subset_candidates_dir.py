"""Restrict a facet_candidates_*.npz directory to a subset of the corpus.

Keeps only rows whose ``corpusids`` entry is in the subset, drops candidate
entries pointing outside the subset, and remaps the stored candidate indices
to positions in the new ``corpusids`` array.

Used to build the held-out-corpus training inputs: the 1k candidate graph
restricted to train+dev, so test papers appear neither as sources nor as
candidates.

Usage:
    uv run python scripts/eval/subset_candidates_dir.py \
        --input output/genret/facet_candidates_1k_goldsource \
        --ids data/s2orc-nlp/splits-connected-1k/traindev.txt \
        --output output/heldout_842/cand_goldsource_842
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from genret._shed.numpy import savez_compressed

logger = logging.getLogger(__name__)

ARRAY_KEYS = (
    "scores_dense",
    "scores_ngram",
    "scores_citation",
    "scores_combined",
)


def subset_one(path: Path, keep: set[int], out_path: Path) -> tuple[int, int, int]:
    """Subset a single facet_candidates npz. Returns (rows, edges_in, edges_out)."""
    f = np.load(path, allow_pickle=True)
    cids = f["corpusids"].astype(np.int64)
    candidates = f["candidates"].astype(np.int64)
    # Materialise once: indexing an NpzFile re-decompresses the whole array,
    # so touching it inside the loop below is quadratic in practice.
    src_arrays = {k: f[k] for k in ARRAY_KEYS if k in f}

    keep_rows = [i for i, c in enumerate(cids) if int(c) in keep]
    new_cids = cids[keep_rows]
    # old row index -> new row index
    remap = {int(old): new for new, old in enumerate(keep_rows)}

    width = candidates.shape[1]
    new_cand = np.full((len(keep_rows), width), -1, dtype=np.int64)
    new_arrays = {
        k: np.zeros((len(keep_rows), width), dtype=np.float32)
        for k in ARRAY_KEYS
        if k in f
    }

    edges_in = 0
    edges_out = 0
    for new_i, old_i in enumerate(keep_rows):
        col = 0
        for j_pos, j_idx in enumerate(candidates[old_i]):
            j = int(j_idx)
            if j < 0:
                continue
            edges_in += 1
            if j not in remap:
                continue  # candidate outside the subset
            new_cand[new_i, col] = remap[j]
            for k, arr in new_arrays.items():
                arr[new_i, col] = src_arrays[k][old_i, j_pos]
            col += 1
            edges_out += 1

    savez_compressed(
        out_path,
        {"corpusids": new_cids, "candidates": new_cand, **new_arrays},
    )
    return len(keep_rows), edges_in, edges_out


def main() -> None:
    """Subset every facet_candidates_*.npz in a directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True, help="Text file of corpus ids")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    keep = {int(x) for x in args.ids.read_text().split() if x.strip()}
    args.output.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input.glob("facet_candidates_*.npz"))
    if not paths:
        raise SystemExit(f"no facet_candidates_*.npz under {args.input}")

    for path in paths:
        rows, ein, eout = subset_one(path, keep, args.output / path.name)
        logger.info(
            "%-46s rows %4d | edges %6d -> %6d (%.1f%% kept)",
            path.name,
            rows,
            ein,
            eout,
            100 * eout / max(1, ein),
        )
    logger.info("Wrote %d files to %s", len(paths), args.output)


if __name__ == "__main__":
    main()
