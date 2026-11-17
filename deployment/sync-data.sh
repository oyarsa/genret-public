#!/bin/bash
# Ship Graft's read-only payload to the deployment host.
#
# Run from the repo root on the machine that holds the data:
#   deployment/sync-data.sh [user@host]
#
# Copies ~2.4 GB the first time and only differences afterwards. The
# server never writes to this tree, so it is bind-mounted read-only.
set -euo pipefail

REMOTE="${1:-${GRAFT_HOST:-graphmind.maleldil.com}}"
DATA_DIR="${GRAFT_DATA_DIR:-/opt/graft-data}"

PROCESSED="data/s2orc-nlp-v2/splits-connected-10k-v2/processed"

# source:destination pairs, relative to the repo root
PAYLOAD=(
    "output/v2_crossfacet_meanpool/:edges/"
    "data/s2orc-nlp-v2/papers/:metadata/"
    "data/s2orc-nlp-v2/s2orc/:s2orc/"
    "output/minilm_onnx/:minilm_onnx/"
)

for item in "${PAYLOAD[@]}"; do
    [ -d "${item%%:*}" ] || { echo "missing: ${item%%:*}" >&2; exit 1; }
done
[ -d "$PROCESSED" ] || { echo "missing: $PROCESSED" >&2; exit 1; }

echo "Preparing ${DATA_DIR} on ${REMOTE}"
ssh "$REMOTE" "mkdir -p ${DATA_DIR}" || {
    echo "cannot write ${DATA_DIR} on ${REMOTE}." >&2
    echo "Create it once with:  sudo mkdir -p ${DATA_DIR} && sudo chown \$USER ${DATA_DIR}" >&2
    exit 1
}

for item in "${PAYLOAD[@]}"; do
    src="${item%%:*}"
    dst="${item##*:}"
    echo "==> ${src} -> ${DATA_DIR}/${dst}"
    rsync -az --info=progress2 --delete "$src" "${REMOTE}:${DATA_DIR}/${dst}"
done

# The facet graph files are mostly the offline pairwise similarity
# matrices, which serving never reads: repack to the vectors and ids it
# does (837 MB -> 32 MB) before shipping.
DENSE_STAGE=$(mktemp -d)
trap 'rm -rf "$DENSE_STAGE"' EXIT
echo "==> repacking ${PROCESSED} -> ${DATA_DIR}/dense/"
uv run --no-sync python deployment/repack-dense.py \
    --processed "$PROCESSED" --output "$DENSE_STAGE"
rsync -az --info=progress2 --delete "$DENSE_STAGE/" "${REMOTE}:${DATA_DIR}/dense/"

echo
echo "Done. On ${REMOTE}:"
ssh "$REMOTE" "du -sh ${DATA_DIR} ${DATA_DIR}/*"
