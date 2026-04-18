import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.paths import RUNS_DIR, ARTIFACTS_DIR
from src.api.schemas import RunSummary, RunDetailResponse, RunMetrics

LOGGER = logging.getLogger(__name__)

def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        LOGGER.warning("Failed to read JSON %s: %s", path, e)
        return None

def _derive_status(run_dir: Path, config: dict | None, metrics: dict | None, best_pt_exists: bool) -> str:
    if config is not None and (metrics is not None or best_pt_exists):
        return "finished"
    if config is not None:
        return "partial"
    
    # Very basic heuristic for 'created' vs 'failed' vs 'running' could be added.
    # Without an active job tracking mechanism for training itself, we default to 'created'.
    return "created"

def _get_creation_time(run_id: str, run_dir: Path) -> datetime | None:
    try:
        # Assuming ID format starts with: YYYYMMDD-HHMMSS
        date_str = run_id.split("-")[0] + run_id.split("-")[1]
        return datetime.strptime(date_str, "%Y%m%d%H%M%S")
    except Exception:
        pass
    # Fallback to dir creation time
    try:
        return datetime.fromtimestamp(run_dir.stat().st_ctime)
    except Exception:
        return None

def scan_all_runs() -> list[RunSummary]:
    runs = []
    if not RUNS_DIR.exists():
        return runs

    # Some runs are nested inside "gump" directly, handle both.
    run_dirs = []
    for p in RUNS_DIR.iterdir():
        if p.is_dir() and p.name == "gump":
            run_dirs.extend([d for d in p.iterdir() if d.is_dir()])
        elif p.is_dir() and not p.name.startswith("."):
            run_dirs.append(p)
            
    for run_dir in run_dirs:
        run_id = run_dir.name
        
        config = _safe_read_json(run_dir / "config.json")
        metrics = _safe_read_json(run_dir / "metrics.json")
        best_pt_exists = (run_dir / "best.pt").exists()
        
        status = _derive_status(run_dir, config, metrics, best_pt_exists)
        model_type = config.get("model", "unknown") if config else None
        if not model_type and "-" in run_id:
            model_type = run_id.split("-")[-1] # fallback from name
            
        created_at = _get_creation_time(run_id, run_dir)
        
        run_metrics = RunMetrics()
        if metrics:
            run_metrics.val_f1 = metrics.get("val_f1") or metrics.get("best_f1")
            run_metrics.val_iou = metrics.get("val_iou") or metrics.get("best_iou")
            
        # Count artifact tiles
        artifact_count = 0
        run_artifacts_dir = ARTIFACTS_DIR / run_id
        if run_artifacts_dir.exists():
            artifact_count = sum(1 for d in run_artifacts_dir.iterdir() if d.is_dir())
            
        runs.append(RunSummary(
            run_id=run_id,
            status=status,
            model_type=model_type,
            created_at=created_at,
            metrics=run_metrics,
            artifact_tile_count=artifact_count
        ))
        
    runs.sort(key=lambda r: r.created_at or datetime.min, reverse=True)
    return runs

def get_run_detail(run_id: str) -> RunDetailResponse | None:
    # Resolve correct run dir (could be in runs/ or runs/gump/)
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        run_dir = RUNS_DIR / "gump" / run_id
        if not run_dir.exists():
            return None
            
    config = _safe_read_json(run_dir / "config.json")
    metrics = _safe_read_json(run_dir / "metrics.json")
    history = _safe_read_json(run_dir / "history.json")
    best_pt_exists = (run_dir / "best.pt").exists()
    
    status = _derive_status(run_dir, config, metrics, best_pt_exists)
    
    # Collect tile artifacts
    tiles = []
    artifacts_ready = False
    run_artifacts_dir = ARTIFACTS_DIR / run_id
    if run_artifacts_dir.exists():
        tiles = [d.name for d in run_artifacts_dir.iterdir() if d.is_dir()]
        artifacts_ready = len(tiles) > 0
        
    return RunDetailResponse(
        run_id=run_id,
        status=status,
        config=config,
        metrics=metrics,
        history=history,
        tiles=tiles,
        artifacts_ready=artifacts_ready
    )
