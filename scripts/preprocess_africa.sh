#!/bin/bash
# Preprocess African training tiles into zarr caches after download completes.
# Run from /workspace inside the docker container (docker exec -it rocm bash).
#
# Usage:
#   bash /shared-docker/scripts/preprocess_africa.sh
#
# Output: /data/africa_cache/{train,forest_masks}/

set -euo pipefail

DATA_ROOT="/data/africa"
CACHE_ROOT="/data/africa_cache"

echo "=== Preprocessing African tiles ==="
echo "Data root  : $DATA_ROOT"
echo "Cache root : $CACHE_ROOT"

cd /workspace

python3 -m src.preprocess \
    --data-root "$DATA_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --split train \
    --overwrite \
    2>&1 | tee /data/africa_preprocess.log

echo "=== Preprocessing done ==="
echo "Cache written to $CACHE_ROOT"
echo ""
echo "To train with both original + African tiles, set:"
echo "  --cache-root /data/africa_cache  (or merge with existing cache)"
