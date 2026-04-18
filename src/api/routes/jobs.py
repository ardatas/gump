from fastapi import APIRouter, HTTPException
from src.api.schemas import CreateInferTileJobRequest, CreateJobResponse, JobStatusResponse
from src.worker.file_queue import create_job, get_job_status
from src.registry.scan_runs import get_run_detail

router = APIRouter()

@router.post("/infer-tile", response_model=CreateJobResponse)
def create_infer_tile_job(req: CreateInferTileJobRequest):
    # Optional Validation: Ensure run exists
    run = get_run_detail(req.run_id)
    if not run:
        raise HTTPException(status_code=400, detail=f"Run {req.run_id} not found.")
        
    job_id = create_job(
        job_type="infer_tile",
        run_id=req.run_id,
        tile_id=req.tile_id,
        requires_gpu=True, # inference requires GPU
        params={
            "threshold": req.threshold,
            "export_geojson": req.export_geojson,
            "export_overlay": req.export_overlay,
            "morph_cleanup": req.morph_cleanup
        }
    )
    return CreateJobResponse(job_id=job_id, status="queued")

@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    job_data = get_job_status(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return JobStatusResponse(**job_data)
