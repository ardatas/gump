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
  --runs-dir /workspace/runs/gump
```

The baseline trainer:

- builds tabular per-pixel features from the Zarr caches
- uses a region-held-out validation split
- trains an MLP with weighted BCE
- saves `best.pt`, metrics, and run config under `runs/`

## Run Inference

```bash
python -m src.infer \
  --cache-root /workspace/cache/gump \
  --checkpoint /workspace/runs/gump/<run>/best.pt \
  --output-root /workspace/output/gump
```

This writes:

- binary GeoTIFF rasters under `rasters/`
- GeoJSON leaderboard artifacts under `geojson/`
