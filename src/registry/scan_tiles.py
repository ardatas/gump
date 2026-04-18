import json
import logging
from pathlib import Path

from src.common.paths import ARTIFACTS_DIR
from src.api.schemas import TileSummary, TileArtifacts, TileSummaryMetrics

LOGGER = logging.getLogger(__name__)

def get_tile_summary(run_id: str, tile_id: str) -> TileSummary | None:
    tile_dir = ARTIFACTS_DIR / run_id / tile_id
    if not tile_dir.exists():
        return None
        
    summary_path = tile_dir / "summary.json"
    
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Reconstruct the schema
            return TileSummary(
                run_id=data.get("run_id", run_id),
                tile_id=data.get("tile_id", tile_id),
                status=data.get("status", "ready"),
                threshold=data.get("threshold"),
                artifacts=TileArtifacts(**data.get("artifacts", {})),
                paths=data.get("paths", {}),
                metrics=TileSummaryMetrics(**data.get("metrics", {}))
            )
        except Exception as e:
            LOGGER.warning("Failed to parse summary.json in %s: %s", tile_dir, e)
            
    # Fallback: construct it from files if no summary exists
    has_geojson = (tile_dir / "pred.geojson").exists()
    has_overlay = (tile_dir / "overlay.png").exists()
    has_prob = (tile_dir / "prob.tif").exists()
    
    paths = {}
    if has_geojson: paths["geojson"] = f"/artifacts/{run_id}/{tile_id}/pred.geojson"
    if has_overlay: paths["overlay_png"] = f"/artifacts/{run_id}/{tile_id}/overlay.png"
    if has_prob: paths["prob_tif"] = f"/artifacts/{run_id}/{tile_id}/prob.tif"
    
    return TileSummary(
        run_id=run_id,
        tile_id=tile_id,
        status="ready",
        artifacts=TileArtifacts(
            geojson=has_geojson,
            overlay_png=has_overlay,
            prob_tif=has_prob
        ),
        paths=paths
    )

def get_all_tiles_for_run(run_id: str) -> list[TileSummary]:
    run_dir = ARTIFACTS_DIR / run_id
    if not run_dir.exists():
        return []
    
    summaries = []
    for tile_dir in run_dir.iterdir():
        if tile_dir.is_dir():
            s = get_tile_summary(run_id, tile_dir.name)
            if s:
                summaries.append(s)
    return summaries
