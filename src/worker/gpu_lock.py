import json
import os
import time
from datetime import datetime
from typing import Any

from src.common.paths import GPU_LOCK_PATH

class GPULockError(Exception):
    pass

class GPULock:
    def __init__(self, job_id: str, job_type: str):
        self.job_id = job_id
        self.job_type = job_type
        self.pid = os.getpid()

    def acquire(self):
        """Atomically acquire the GPU lock."""
        # Simple cross-platform check (if file exists, someone has it)
        if GPU_LOCK_PATH.exists():
            # Check if PID is still alive
            try:
                with open(GPU_LOCK_PATH, "r", encoding="utf-8") as f:
                    lock_data = json.load(f)
                
                old_pid = lock_data.get("pid")
                if old_pid:
                    try:
                        os.kill(old_pid, 0)
                        # Process is still alive, lock is valid
                        raise GPULockError(f"GPU is currently locked by Job {lock_data.get('job_id')} (PID {old_pid})")
                    except OSError:
                        # Process is dead, lock is stale, we can overwrite
                        pass
            except Exception as e:
                if isinstance(e, GPULockError):
                    raise
                # if invalid JSON, we just overwrite
                pass
                
        # Write new lock
        lock_data = {
            "job_id": self.job_id,
            "type": self.job_type,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "pid": self.pid
        }
        
        temp_path = GPU_LOCK_PATH.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(lock_data, f)
            f.flush()
            os.fsync(f.fileno())
            
        try:
            os.replace(temp_path, GPU_LOCK_PATH)
        except Exception as e:
            raise GPULockError(f"Failed to write lock file: {e}")

    def release(self):
        """Release the GPU lock."""
        if GPU_LOCK_PATH.exists():
            try:
                GPU_LOCK_PATH.unlink()
            except OSError:
                pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
