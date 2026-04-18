# Handoff Summary

## Goal

Build an end-to-end baseline for the osapiens deforestation challenge that:

- preprocesses all raw train/test tiles into aligned cache artifacts
- trains a leaderboard-usable first model
- supports inference to GeoTIFF and GeoJSON submission outputs

## What Was Implemented

### 1. Preprocessing pipeline

Added a full preprocessing stack under `src/`:

- `src/index.py`
  - discovers train/test tiles and associated Sentinel-1, Sentinel-2, AEF, and label files
- `src/raster_utils.py`
  - handles grid extraction and raster reprojection
- `src/features.py`
  - builds `forest_2020` from Sentinel-2 NDVI
  - falls back to the nearest available Sentinel-2 year if 2020 is missing
- `src/labels.py`
  - decodes `RADD`, `GLAD-L`, and `GLAD-S2`
  - creates `consensus`, `weight`, `votes`, `source_mask`, and `alert_year`
- `src/preprocess.py`
  - writes chunked Zarr caches per tile
  - merges manifest entries instead of overwriting them

### 2. Training/inference baseline

Added a first model training path:

- `src/cache.py`
  - loads metadata and tile caches from `.zarr`
- `src/feature_bank.py`
  - converts tile caches into tabular per-pixel features
  - uses yearly aligned S2, S1, AEF statistics and deltas
  - handles missing yearly S1 stacks by zero-filling
- `src/dataset.py`
  - samples train pixels with a positive/negative ratio
  - keeps positives when max sample caps apply
- `src/models.py`
  - MLP baseline
- `src/metrics.py`
  - weighted precision/recall/F1 and threshold search
- `src/train.py`
  - region-held-out validation split
  - weighted BCE training
  - checkpoint/config/history/metrics output
  - skips tiles with zero trainable pixels
- `src/infer.py`
  - runs inference on cached tiles
  - writes binary GeoTIFF predictions
  - converts them to GeoJSON using `submission_utils.py`

### 3. Docs and dependencies

Updated:

- `README.md`
- `requirements.txt`
- `pyproject.toml`

## Server State

Server:

- `root@134.199.196.225`
- running container: `rocm`

Workspace paths:

- repo in container: `/workspace/gump`
- raw challenge data: `/workspace/data`
- preprocessed caches: `/workspace/cache/gump`
- training runs: `/workspace/runs/gump`

## Preprocessing Status

All tiles were preprocessed successfully.

Cache summary:

- train caches: `16`
- test caches: `5`
- total cache size: about `23G`
- manifest: `/workspace/cache/gump/manifest.json`

Known preprocessing edge cases already handled:

- test tile `47QMA_6_2` had no Sentinel-2 observations in 2020
  - forest-mask fallback used nearest year `2021`
- some tiles have no trainable pixels after `forest_mask` and `weight > 0`
  - trainer now skips them instead of crashing
- some tiles have missing S1 orbit/year data
  - feature builder now zero-fills those feature blocks

## Completed Training Run

Smoke-test baseline run completed successfully:

- run dir: `/workspace/runs/gump/20260418-081922-baseline`
- checkpoint: `/workspace/runs/gump/20260418-081922-baseline/best.pt`
- metrics: `/workspace/runs/gump/20260418-081922-baseline/metrics.json`

Validation setup:

- held-out validation region: `48P`
- train tiles: `12`
- val tiles: `4`

Epoch-1 result:

- `val_f1 = 0.5541`
- `precision = 0.3862`
- `recall = 0.9805`
- `best_threshold = 0.90`

This is the first end-to-end proof that preprocessing -> training -> validation -> checkpoint saving works.

## Long Training Run In Progress

Background run started for a stronger baseline:

- process: `23696`
- log: `/workspace/runs/gump/baseline-long.log`

Command:

```bash
cd /workspace/gump
python -m src.train \
  --cache-root /workspace/cache/gump \
  --runs-dir /workspace/runs/gump \
  --epochs 6 \
  --batch-size 16384 \
  --eval-batch-size 131072 \
  --max-samples-per-tile 120000 \
  --negative-ratio 3.0 \
  --num-workers 0 \
  --val-region 48P \
  --log-level INFO
```

At the time of writing, the run is still progressing through train tile sampling/training.

To inspect it:

```bash
ssh root@134.199.196.225
docker exec rocm bash -lc "ps -p 23696 -o pid,etimes,%cpu,%mem,cmd"
docker exec rocm bash -lc "tail -n 40 /workspace/runs/gump/baseline-long.log"
```

## Submission Path

Once a checkpoint is ready, inference can be run with:

```bash
cd /workspace/gump
python -m src.infer \
  --cache-root /workspace/cache/gump \
  --checkpoint /workspace/runs/gump/<run>/best.pt \
  --output-root /workspace/output/gump
```

Outputs:

- rasters: `/workspace/output/gump/rasters/*.tif`
- leaderboard geojson: `/workspace/output/gump/geojson/*.geojson`

## Recommended Next Steps

1. Let the long baseline training finish and inspect `metrics.json`.
2. Run `src.infer` on the best checkpoint.
3. Sanity-check GeoTIFF/GeoJSON outputs tile-by-tile.
4. Prepare leaderboard submission from the generated GeoJSONs.
5. Then improve the model, in this order:
   - better `forest_2020` prior than NDVI heuristic
   - better validation split strategy if needed
   - stronger temporal model than the current MLP
   - post-processing on prediction rasters before GeoJSON export

## Important Constraints / Notes

- The current baseline is designed to be practical and leaderboard-usable, not final.
- `val_region = 48P` is fixed explicitly in the long run for reproducibility.
- `manifest.json` now merges future preprocessing runs, but the currently written server manifest was explicitly regenerated to contain both train and test entries.
- The repo was synced into the container manually. Future code changes must be copied again unless the container workspace is turned into a mounted bind volume.
