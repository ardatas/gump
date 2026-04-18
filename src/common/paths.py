from pathlib import Path
import os

WORKSPACE_ROOT = Path("/workspace")

RUNS_DIR = WORKSPACE_ROOT / "runs"
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"

JOBS_DIR = WORKSPACE_ROOT / "jobs"
JOBS_QUEUED_DIR = JOBS_DIR / "queued"
JOBS_RUNNING_DIR = JOBS_DIR / "running"
JOBS_DONE_DIR = JOBS_DIR / "done"
JOBS_FAILED_DIR = JOBS_DIR / "failed"

LOGS_DIR = WORKSPACE_ROOT / "logs"
STATE_DIR = WORKSPACE_ROOT / "state"
CACHE_DIR = WORKSPACE_ROOT / "cache"

GPU_LOCK_PATH = STATE_DIR / "gpu.lock"
SQLITE_PATH = STATE_DIR / "app.db"

def ensure_workspace_dirs():
    """Create all required workspace directories if they do not exist."""
    directories = [
        RUNS_DIR,
        ARTIFACTS_DIR,
        JOBS_QUEUED_DIR,
        JOBS_RUNNING_DIR,
        JOBS_DONE_DIR,
        JOBS_FAILED_DIR,
        LOGS_DIR,
        LOGS_DIR / "training",
        STATE_DIR,
        CACHE_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
