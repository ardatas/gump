# GUMP

Preprocessing and submission utilities for the osapiens deforestation challenge.

## Preprocess Tiles

Build per-tile Zarr caches from the raw challenge folders under `data/`:

```bash
python -m src.preprocess \
  --data-root /workspace/data \
  --cache-root /workspace/cache \
  --split train
```

This pipeline:

- uses Sentinel-2 as the spatial reference grid
- reprojects Sentinel-1, AlphaEarth embeddings, and labels onto that grid
- derives a `forest_2020` mask from Sentinel-2 NDVI
- harmonizes `RADD`, `GLAD-L`, and `GLAD-S2` into consensus labels and weights
- writes chunked per-tile Zarr stores for later model training

## Train Baseline

```bash
python -m src.train \
  --cache-root /workspace/cache/gump \
  --runs-dir /workspace/runs/gump \
  --model-name resmlp \
  --loss focal \
  --amp-dtype bf16 \
  --auto-positive-class-weight
```

The baseline trainer:

- builds tabular per-pixel features from the Zarr caches
- uses a region-held-out validation split
- supports `mlp` and `resmlp`
- supports weighted BCE or focal loss, optional auto class weighting, gradient clipping, AMP, cosine LR scheduling, early stopping, and resume from `last.pt`
- saves `best.pt`, `last.pt`, metrics, and run config under `runs/`

## Run Inference

```bash
python -m src.infer \
  --cache-root /workspace/cache/gump \
  --checkpoint /workspace/runs/gump/<run>/best.pt \
  --output-root /workspace/output/gump \
  --save-probabilities \
  --closing-radius 1 \
  --opening-radius 1 \
  --min-area-ha 0.5
```

This writes:

- probability GeoTIFFs under `probabilities/` when requested
- binary GeoTIFF rasters under `rasters/`
- GeoJSON leaderboard artifacts under `geojson/`

Inference now predicts only on forest pixels and applies reusable postprocessing:

- forest mask gating
- optional persistence gating
- morphological closing/opening
- minimum object area and minimum hole filling

## Evaluate Postprocessing On Val

```bash
python -m src.evaluate \
  --cache-root /workspace/cache/gump \
  --checkpoint /workspace/runs/gump/<run>/best.pt \
  --report-path /workspace/runs/gump/<run>/val_postprocess.json \
  --threshold 0.70 \
  --threshold 0.75 \
  --threshold 0.80 \
  --closing-radius 0 \
  --closing-radius 1 \
  --opening-radius 0 \
  --opening-radius 1 \
  --min-area-ha 0.3 \
  --min-area-ha 0.5
```

This runs local postprocessing ablations on the validation region(s) from the checkpoint and writes a ranked JSON report.
