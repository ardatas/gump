from typing import Any
from datetime import datetime
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"

class RunMetrics(BaseModel):
    val_f1: float | None = None
    val_iou: float | None = None
    pr_auc: float | None = None
    # Weitere Metriken können hier ergänzt werden

class RunSummary(BaseModel):
    run_id: str
    status: str
    model_type: str | None = None
    created_at: datetime | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    artifact_tile_count: int = 0

class RunListResponse(BaseModel):
    runs: list[RunSummary]

class RunDetailResponse(BaseModel):
    run_id: str
    status: str
    config: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] | None = None
    tiles: list[str] = Field(default_factory=list)
    artifacts_ready: bool = False

class TileSummaryMetrics(BaseModel):
    pixel_f1: float | None = None
    pixel_iou: float | None = None

class TileArtifacts(BaseModel):
    geojson: bool = False
    overlay_png: bool = False
    prob_tif: bool = False
    uncertainty_tif: bool = False

class TileBounds(BaseModel):
    """WGS84 bounding box [min_lon, min_lat, max_lon, max_lat]"""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

class TileSummary(BaseModel):
    run_id: str
    tile_id: str
    status: str
    threshold: float | None = None
    artifacts: TileArtifacts = Field(default_factory=TileArtifacts)
    paths: dict[str, str] = Field(default_factory=dict)
    metrics: TileSummaryMetrics = Field(default_factory=TileSummaryMetrics)
    bounds: TileBounds | None = None
    months_s2: list[tuple[int, int]] | None = None  # [[year, month], ...]

class TileListResponse(BaseModel):
    run_id: str
    tiles: list[TileSummary]

class TileDetailResponse(BaseModel):
    run_id: str
    tile_id: str
    summary: TileSummary

class CreateInferTileJobRequest(BaseModel):
    run_id: str
    tile_id: str
    threshold: float | None = None
    export_geojson: bool = True
    export_overlay: bool = True
    morph_cleanup: bool = True

class CreateJobResponse(BaseModel):
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    type: str
    status: str
    run_id: str | None = None
    tile_id: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
