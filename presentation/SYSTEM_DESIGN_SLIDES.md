# System Design Slides

Use this script to present the implementation in about 90 to 120 seconds.

## Slide 1: End-to-End Architecture

Title:
Solution Architecture: From Satellite Data to Actionable Alerts

Message:
We decouple data science execution from user interaction. The API remains responsive while the worker handles GPU-heavy inference asynchronously.

Talk track:
- On the left, we ingest multimodal satellite and label sources.
- In the center, preprocessing and training generate reproducible run artifacts.
- On the right, FastAPI and the Next.js app expose results for regulators.
- The filesystem under `/workspace` is our contract between components.

Proof points:
- Queue states: `queued`, `running`, `done`, `failed`
- Artifacts include GeoJSON polygons, probability rasters, and summary metadata

## Slide 2: Data-to-Insight Pipeline

Title:
Deterministic Pipeline with Inspectable Outputs

Message:
Each stage writes explicit artifacts, so model quality and serving behavior can be audited independently.

Talk track:
- Preprocess aligns all modalities to one grid and harmonizes noisy labels.
- Feature extraction and region-held-out validation train a robust baseline.
- Inference applies thresholding and spatial cleanup before export.
- The frontend visualizes confidence and polygons without touching model code.

Proof points:
- Training artifacts in `/workspace/runs`
- Inference artifacts in `/workspace/artifacts/{run}/{tile}`

## Slide 3: Reliability and Operational Design

Title:
Operational Reliability on One Server

Message:
Single-server deployment can still be reliable when runtime responsibilities are clearly separated.

Talk track:
- API does metadata and job orchestration, not long GPU computation.
- Worker is the only process that loads model checkpoints.
- GPU lock prevents concurrent heavy jobs.
- Output contracts keep frontend and ML pipeline loosely coupled.

Proof points:
- API routes: `/api/runs`, `/api/jobs/*`
- Worker modules: `src/worker/main.py`, `src/worker/executor.py`
- Path contracts: `src/common/paths.py`

## Visual Source

For diagrams and exact component mapping, use `docs/ARCHITECTURE.md`.