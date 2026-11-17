"""Strip the dense directory down to what the server actually loads.

Each ``facet_graph_{facet}.npz`` carries the full pairwise ``mean_pool``
similarity matrix used to build the graph offline -- 246 MB per facet
that serving never touches. The server reads only ``corpusids`` and
``corpus_vectors``, about 8 MB per facet.

Usage:
    uv run python deployment/repack-dense.py \
        --processed data/s2orc-nlp-v2/splits-connected-10k-v2/processed \
        --output /tmp/graft-dense
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from genret._shed.numpy import savez_compressed

FACETS = ("problems", "methods", "results", "contributions")


def main() -> None:
    """Write slimmed facet graphs plus the facets file to the output dir."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    total_in = total_out = 0
    for facet in FACETS:
        src = args.processed / f"facet_graph_{facet}.npz"
        dst = args.output / f"facet_graph_{facet}.npz"
        with np.load(src) as npz:
            savez_compressed(
                dst,
                {
                    "corpusids": npz["corpusids"],
                    "corpus_vectors": npz["corpus_vectors"],
                },
            )
        total_in += src.stat().st_size
        total_out += dst.stat().st_size
        print(
            f"{facet}: {src.stat().st_size / 1e6:.0f} MB -> {dst.stat().st_size / 1e6:.0f} MB"
        )

    facets_src = args.processed / "facets.jsonl.zst"
    shutil.copy2(facets_src, args.output / facets_src.name)
    print(f"facets.jsonl.zst: {facets_src.stat().st_size / 1e6:.0f} MB (copied)")

    print(f"\ntotal: {total_in / 1e6:.0f} MB -> {total_out / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
