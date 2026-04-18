# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

`gump` is the codebase for an osapiens hackathon submission: a per-pixel
deforestation detector trained on multimodal satellite time series
(Sentinel-1 RTC, Sentinel-2 L2A, AlphaEarth foundation-model embeddings) with
noisy weak-label supervision from RADD, GLAD-L, and GLAD-S2. The final
leaderboard artefacts are GeoJSON polygons in EPSG:4326 derived from binary
prediction rasters.

Long-form design notes live in `PLAN.md`, `PLAN_B.md`, `HANDOFF.md`, and
`src/data/README.md` (German). The challenge brief is in
`osapiens-challenge-full-description.md`.

## Environment & Dependencies

- Python >= 3.10 (see `pyproject.toml`). `.python-version` pins a newer
  interpreter for local tooling but is not authoritative.
- `make install` creates a `.venv` with `python3.10` and installs
  `requirements.txt` via `uv`. Activate with `source .venv/bin/activate`.
  Training/inference additionally needs PyTorch (install separately, matched
  to the target CUDA/ROCm build) — it is intentionally not pinned in
  `requirements.txt`/`pyproject.toml` so the repo stays installable off-GPU.
- Core runtime deps: `rasterio`, `zarr<3` + `numcodecs`, `geopandas`,
  `shapely`, `scikit-image`, `boto3`, `tqdm`, `numpy`/`pandas`.
- Data download: `python download_data.py` (or `make download_data_from_s3`)
  pulls the challenge bucket into `./data/` via unsigned S3.

## Common Commands

All training/inference entry points are `python -m src.<module>` scripts that
take argparse flags; there are no wrapper shell scripts. Representative
invocations (see `README.md` for the full flag set):

```bash
# 1. Preprocess raw tiles into per-tile Zarr caches (Zarr pipeline, primary)
python -m src.preprocess --data-root <data> --cache-root <cache> --split train

# 2. Train a baseline (resmlp | mlp | tcnn)
python -m src.train --cache-root <cache>/gump --runs-dir <runs>/gump \
  --model-name resmlp --loss focal --amp-dtype bf16

# 3. K-fold (leave-one-region-out) ensemble training
python -m src.train_kfold --cache-root <cache>/gump --runs-dir <runs>/gump

# 4. Calibrate binarisation thresholds on val tiles (supports ensemble)
python -m src.calibrate_thresholds --cache-root <cache>/gump \
  --checkpoint <run>/best.pt --output thresholds.json

# 5. Grid-search postprocessing on validation tiles
python -m src.evaluate --cache-root <cache>/gump --checkpoint <run>/best.pt \
  --report-path val_postprocess.json --threshold 0.75 --min-area-ha 0.5

# 6. Run inference -> rasters + submission GeoJSONs
python -m src.infer --cache-root <cache>/gump --checkpoint <run>/best.pt \
  --output-root <out>/gump --save-probabilities

# 7. Retrofit Hansen GFC forest masks into existing caches
python -m src.patch_forest_masks --cache-root <cache>/gump --split both

# 8. Smoke test (no GPU / no cache required, good for PR sanity)
python -m src.smoke_test
```

`src/infer.py` and `src/calibrate_thresholds.py` accept `--checkpoint`
multiple times to build an ensemble (averaged probabilities); all checkpoints
must share `target_years`.

### Tests & Lint

- Test framework is `pytest` with shared fixtures in `tests/conftest.py`
  (synthetic `TileData`/`TileMetadata` in memory; no cache dependency).
- There is no configured linter/formatter and no top-level `Makefile` test
  target; run tests directly:
  ```bash
  pytest                       # full suite
  pytest tests/test_models.py  # one file
  pytest tests/test_models.py::test_name -xvs  # one test
  ```
- Additionally, `python -m src.smoke_test` is a hand-rolled import &
  forward-pass check that is faster than pytest and runs on CPU.

## Architecture

### Two preprocessing pipelines (do not confuse)

The repo contains **two** preprocessing stacks that write different cache
layouts. Both pass through the same `submission_utils.raster_to_geojson`
exporter at the end but are otherwise independent:

1. **Zarr pipeline — primary, used by trainer/infer/evaluate** (`src/preprocess.py`)
   - Writes `<cache>/<split>/<tile>.zarr` per tile with groups
     `s2/`, `s1_ascending/`, `s1_descending/`, `aef/`, `forest_2020`,
     `labels/{consensus,weight,votes,source_mask,alert_year}` and
     root attrs `tile_id, split, crs, transform, width, height`.
   - Uses Sentinel-2 as the reference grid; reprojects S1/AEF/labels onto it
     via `src/raster_utils.read_reprojected` (nearest for discrete, bilinear
     for continuous).
   - Forest mask: Hansen GFC `treecover2020` by default
     (`src/data/forest_mask.get_or_build_forest_mask`), falling back to
     NDVI-from-S2 (`src/features.compute_forest_mask`) when `--no-hansen`.
   - Label harmonisation (`src/labels.build_label_products`) combines RADD /
     GLAD-L / GLAD-S2 into `consensus` (majority vote), `weight` (higher for
     multi-source agreement, zeroed outside forest), `votes`, `source_mask`,
     and `alert_year` (earliest positive year).
   - Appends to `manifest.json` (merges with existing entries, not overwrites).
   - **All downstream scripts (`src/train.py`, `src/infer.py`,
     `src/evaluate.py`, `src/calibrate_thresholds.py`,
     `src/train_kfold.py`) consume this layout via `src/cache.py`.**

2. **Legacy `.npy` pipeline** (`src/data/prepare_cache.py`, `src/data/stats.py`,
   `src/labels/*`) — historical predecessor described in `src/data/README.md`.
   It writes `<cache>/<tile>/{s2,s1_asc,s1_desc,aef}.npy` + `meta.json` for
   mmap access. Not wired into the training entry points; kept for
   experiments. Don't mix its caches with the Zarr caches above.

### Training data flow (Zarr path)

`src/cache.py`
  `list_tile_paths` → `load_tile_metadata` / `load_tile_data` parse Zarr
  stores into `TileMetadata` (region is the first 3 chars of the MGRS tile
  id) and `TileData` (dense float32 arrays). `choose_validation_regions`
  picks a held-out region so tiles of the same region never straddle the
  split.

`src/feature_bank.py`
  `build_tile_features` converts a `TileData` into a per-pixel
  `TileFeatures` cube of yearly-reduced S2 / S1 / AEF statistics,
  delta-year features, and spectral indices (NDVI/NBR/NDMI). It also
  computes a `persistence_score`/`persistence_mask` reused by
  postprocessing. Results are cached alongside the Zarr as
  `<tile>.features.zarr` — invalidate by deleting the sibling directory
  (`patch_forest_masks.py` does this automatically when forest masks
  change).
  `build_monthly_sequence` returns a dense `(T, C, H, W)` monthly stack
  (C = `SEQUENCE_CHANNELS`) used only by `tcnn` / `--use-temporal`.

`src/dataset.py`
  `sample_tile_training_data` draws positive/negative pixels per tile with
  `negative_ratio` and per-tile caps. `build_eval_matrix` and
  `build_prediction_matrix` flatten to `(N, F)` for val/inference. Year
  classes (for the optional alert-year multi-task head) are gathered here.

`src/models.py`
  Three architectures sharing the `ModelOutputs(logits, year_logits)` API so
  downstream code is model-agnostic:
  - `MLPBaseline` / `ResidualMLP`: tabular per-pixel features only.
  - `TemporalCNN` (`tcnn`): static features + 1D conv over the monthly
    sequence; requires `sequence` input. Year head is activated when
    `num_year_classes > 0`.

`src/train.py`
  Orchestrates the whole training loop. The main module also exports the
  shared helpers (`LossConfig`, `TemporalConfig`, `RuntimeConfig`,
  `Normalizer`, `fit_normalizer`, `normalize_features`, `train_one_epoch`,
  `evaluate_on_tiles`, `predict_probabilities`, `save_checkpoint`, …) that
  `train_kfold.py`, `evaluate.py`, `infer.py`, and
  `calibrate_thresholds.py` import — keep these signatures stable.
  Supports BCE / focal / GCE loss, label smoothing, auto positive class
  weight, AMP (bf16/fp16), cosine LR, grad clipping, early stopping,
  resume-from-`last.pt`, and parallel train/eval feature prep via
  `--prep-workers`. With `--gpu-resident-data auto|always`, static tensors
  for `mlp`/`resmlp` stay on device across epochs.
  Writes `best.pt`, `last.pt`, `metrics.json`, `history.json`, and the
  resolved run config under `<runs-dir>/<timestamp>-<tag>/`.

`src/infer.py`
  Loads N checkpoints (`CheckpointBundle`), averages probabilities,
  chooses the threshold (override > per-region JSON > mean of checkpoint
  thresholds), runs `postprocess.postprocess_probability_map`, writes a
  uint8 GeoTIFF, float32 probability GeoTIFF (if requested), and GeoJSON
  via `submission_utils.raster_to_geojson`. Inference predicts only inside
  `forest_mask`; all postprocessing ops reuse that gate. When
  `--predict-time-step` is set **and** at least one checkpoint exposes a
  year head, per-polygon `time_step` is populated from the argmax of
  `year_logits`.

`src/postprocess.py`
  Single source of truth for binarisation and morphology
  (`PostprocessConfig`: threshold, closing/opening radii, min-area-ha,
  min-hole-area-ha, persistence threshold). Both inference and the
  `evaluate.py` grid-search consume the same function so results are
  reproducible.

`submission_utils.py` (repo root, not under `src/`)
  Vectorises a binary raster to EPSG:4326 polygons, filters by metric area
  (ha) computed in UTM, and writes a GeoJSON FeatureCollection with a
  `time_step` field. `src.infer` imports this module by name — keep it at
  the repo root or adjust both imports.

### Region-aware validation / K-fold

Regions are derived from the first 3 chars of the MGRS tile id
(`18NWG_6_6` → `18N`). This is the unit used everywhere for splitting,
per-region thresholds, and `train_kfold.py` leave-one-region-out folds.
Passing `--val-region` multiple times overrides auto-selection.

### Configs

`configs/preprocess.yaml` and `configs/labels.yaml` hold defaults
(data paths, worker counts, S1 clip range, label weights, NDVI/NDWI
thresholds). They are **not auto-loaded** by the CLI scripts — each CLI
has its own argparse flags and defaults. Treat the YAMLs as documentation
of the intended values unless you wire them up.

### Visualisation app (`www/`)

Next.js 14 + Leaflet viewer for prediction/submission artefacts:

```bash
cd www
npm install
npm run dev      # serves on :8765
npm run build && npm run start
```

Data is expected under `www/public/data/` (whitelisted in `.gitignore`).
Unrelated to the Python pipeline; do not couple the two unless asked.

## Conventions & Gotchas

- `from __future__ import annotations` is used project-wide; frozen
  `@dataclass` objects are the default for config/IO containers. Keep new
  code consistent.
- The training/evaluation scripts all rely on `src.train` re-exporting a
  large public surface (losses, normalizer, AMP helpers, prediction
  helpers). Avoid renaming or deleting those without updating
  `train_kfold.py`, `evaluate.py`, `infer.py`, and `calibrate_thresholds.py`.
- `TemporalCNN` requires a `sequence` tensor shaped `(B, T, C+1)` where the
  extra channel is a per-timestep validity mask. `MLPBaseline`/`ResidualMLP`
  ignore `sequence`. `build_model` dispatches on `--model-name`.
- Hansen mask changes invalidate the cached feature cube
  (`*.features.zarr`) — `patch_forest_masks.py` deletes it automatically;
  do the same manually if you rewrite `forest_2020` by hand.
- `zarr` is pinned to `<3`; `numcodecs.Blosc` APIs assume this. Do not
  upgrade casually.
- `TileMetadata.region` comes from the MGRS prefix (`tile_id.split("_")[0][:3]`).
  Tiles without this format will break `choose_validation_regions` and
  k-fold splits.
- Checkpoints are loaded with `torch.load(..., weights_only=False)` because
  they contain normalizer + config dicts. Never point this at untrusted
  files.
- GDAL / rasterio is **not** thread-safe; preprocessing parallelism uses
  `multiprocessing` (see `src/data/README.md §6`). Prefer processes over
  threads for any new IO-bound raster work.

## Documentation To Consult Before Large Changes

- `PLAN.md` — end-to-end design (features, training, postprocessing).
- `PLAN_B.md` — alternative direction / ablations, especially around
  label harmonisation and forest-mask choices.
- `HANDOFF.md` — current server state, known edge-case tiles, recommended
  next steps, and the canonical training commands used for the baseline.
- `src/data/README.md` — rationale for the legacy `.npy` cache format and
  lessons learned about corrupt tiles (`18NYH_9_9`, `18NWM_9_4`).
