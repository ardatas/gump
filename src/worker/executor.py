import logging
from typing import Any
import json
from pathlib import Path

LOGGER = logging.getLogger(__name__)

def execute_job(job_data: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatches the job to the correct handler based on job type."""
    job_type = job_data.get("type")
    
    if job_type == "infer_tile":
        return _execute_infer_tile(job_data)
    elif job_type == "refresh_registry":
        # Placeholder for registry refresh
        LOGGER.info("Executing refresh registry job...")
        return {"status": "refreshed"}
    else:
        raise ValueError(f"Unknown job type: {job_type}")

def _execute_infer_tile(job_data: dict[str, Any]) -> dict[str, Any]:
    run_id = job_data["run_id"]
    tile_id = job_data["tile_id"]
    params = job_data.get("params", {})
    
    from src.infer import run_tile_inference
    
    # By default, inference expects paths. We'll use the ones defined in common.paths
    from src.common.paths import CACHE_DIR, RUNS_DIR, ARTIFACTS_DIR
    
    # Use gump cache subdirectory if it exists
    actual_cache_root = CACHE_DIR / "gump" if (CACHE_DIR / "gump").exists() else CACHE_DIR
    
    # Resolving run folder
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        run_dir = RUNS_DIR / "gump" / run_id
    if not run_dir.exists():
        raise ValueError(f"Run {run_id} not found in runs directory.")
        
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise ValueError(f"No best.pt checkpoint found for run {run_id}.")
        
    # The output root is artifacts/{run_id}
    output_root = ARTIFACTS_DIR / run_id
    
    result = run_tile_inference(
        cache_root=actual_cache_root,
        checkpoint_paths=[checkpoint_path],
        output_root=output_root,
        tile_id=tile_id,
        threshold_override=params.get("threshold"),
        export_geojson=params.get("export_geojson", True),
        export_overlay=params.get("export_overlay", True),
        morph_cleanup=params.get("morph_cleanup", True)
    )
    
    # Write summary.json
    summary_path = output_root / tile_id / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary_data = {
        "run_id": run_id,
        "tile_id": tile_id,
        "status": "ready",
        "threshold": params.get("threshold", result.get("used_threshold")),
        "artifacts": {
            "geojson": (output_root / tile_id / "pred.geojson").exists(),
            "overlay_png": (output_root / tile_id / "overlay.png").exists(),
            "prob_tif": (output_root / tile_id / "prob.tif").exists()
        },
        "paths": {
            "geojson": f"/artifacts/{run_id}/{tile_id}/pred.geojson",
            "overlay_png": f"/artifacts/{run_id}/{tile_id}/overlay.png",
            "prob_tif": f"/artifacts/{run_id}/{tile_id}/prob.tif"
        },
        "metrics": {
            "pixel_f1": None,
            "pixel_iou": None
        }
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    return summary_data
