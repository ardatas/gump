import json
import uuid
import os
import logging
from datetime import datetime
from typing import Any
from pathlib import Path

from src.common.paths import (
    JOBS_QUEUED_DIR,
    JOBS_RUNNING_DIR,
    JOBS_DONE_DIR,
    JOBS_FAILED_DIR,
    JOBS_DIR
)

LOGGER = logging.getLogger(__name__)

def create_job(job_type: str, run_id: str | None = None, tile_id: str | None = None, params: dict[str, Any] | None = None, requires_gpu: bool = False) -> str:
    """Create a new job and write it to the queued directory atomically."""
    job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    job_data = {
        "job_id": job_id,
        "type": job_type,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "tile_id": tile_id,
        "requires_gpu": requires_gpu,
        "params": params or {},
        "result": None,
        "error": None
    }
    
    # Write to temporary file, then move atomically
    temp_path = JOBS_DIR / f"{job_id}.tmp"
    final_path = JOBS_QUEUED_DIR / f"{job_id}.json"
    
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(job_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
        
    os.replace(temp_path, final_path)
    LOGGER.info("Created job %s of type %s", job_id, job_type)
    return job_id

def get_job_status(job_id: str) -> dict[str, Any] | None:
    """Find a job in any state directory and return its data."""
    for state_dir in [JOBS_QUEUED_DIR, JOBS_RUNNING_DIR, JOBS_DONE_DIR, JOBS_FAILED_DIR]:
        job_file = state_dir / f"{job_id}.json"
        if job_file.exists():
            try:
                with open(job_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                LOGGER.error("Could not read job file %s: %s", job_file, e)
                return None
    return None

def fetch_next_job() -> tuple[dict[str, Any], Path] | tuple[None, None]:
    """Find the next queued job, move it to running, and return it. (Worker side)"""
    # Simply sort by creation time (file name starts with timestamp)
    queued_files = sorted(JOBS_QUEUED_DIR.glob("*.json"))
    if not queued_files:
        return None, None
        
    for job_file in queued_files:
        running_path = JOBS_RUNNING_DIR / job_file.name
        try:
            os.replace(job_file, running_path)
        except OSError:
            # Another worker might have grabbed it
            continue
            
        # Successfully claimed
        with open(running_path, "r", encoding="utf-8") as f:
            job_data = json.load(f)
            
        job_data["status"] = "running"
        job_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
        _save_job_state(running_path, job_data)
        
        return job_data, running_path
        
    return None, None

def mark_job_done(job_path: Path, job_data: dict[str, Any], result: dict[str, Any] | None = None):
    job_data["status"] = "done"
    job_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    job_data["result"] = result
    _save_job_state(job_path, job_data)
    
    final_path = JOBS_DONE_DIR / job_path.name
    os.replace(job_path, final_path)

def mark_job_failed(job_path: Path, job_data: dict[str, Any], error: str):
    job_data["status"] = "failed"
    job_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    job_data["error"] = error
    _save_job_state(job_path, job_data)
    
    final_path = JOBS_FAILED_DIR / job_path.name
    os.replace(job_path, final_path)

def _save_job_state(path: Path, data: dict[str, Any]):
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)
