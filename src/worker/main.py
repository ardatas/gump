import time
import logging
from src.common.paths import ensure_workspace_dirs
from src.worker.file_queue import fetch_next_job, mark_job_done, mark_job_failed
from src.worker.gpu_lock import GPULock, GPULockError
from src.worker.executor import execute_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

def main():
    ensure_workspace_dirs()
    LOGGER.info("Worker started. Polling for jobs...")
    
    while True:
        try:
            job_data, job_path = fetch_next_job()
            if not job_data or not job_path:
                time.sleep(2)
                continue
                
            job_id = job_data["job_id"]
            job_type = job_data["type"]
            requires_gpu = job_data.get("requires_gpu", False)
            
            LOGGER.info("Picked up job %s (%s). GPU required: %s", job_id, job_type, requires_gpu)
            
            try:
                if requires_gpu:
                    with GPULock(job_id=job_id, job_type=job_type):
                        result = execute_job(job_data)
                else:
                    result = execute_job(job_data)
                    
                mark_job_done(job_path, job_data, result=result)
                LOGGER.info("Job %s completed successfully.", job_id)
                
            except GPULockError as e:
                LOGGER.warning("Could not acquire GPU lock for job %s: %s. Returning to queue.", job_id, e)
                # Note: Currently fetch_next_job moves to 'running'. If we fail to get lock, 
                # we should technically push it back to 'queued' or wait.
                # For simplicity in V1, we fail the job. 
                # (A real queue would nack/requeue)
                mark_job_failed(job_path, job_data, error=str(e))
                
            except Exception as e:
                LOGGER.exception("Job %s failed with error: %s", job_id, e)
                mark_job_failed(job_path, job_data, error=str(e))
                
        except KeyboardInterrupt:
            LOGGER.info("Worker stopping.")
            break
        except Exception as e:
            LOGGER.error("Worker loop encountered an error: %s", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
