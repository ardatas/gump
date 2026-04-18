from fastapi import APIRouter, HTTPException
from src.api.schemas import RunListResponse, RunDetailResponse
from src.registry.scan_runs import scan_all_runs, get_run_detail

router = APIRouter()

@router.get("", response_model=RunListResponse)
def list_runs():
    runs = scan_all_runs()
    return RunListResponse(runs=runs)

@router.get("/{run_id}", response_model=RunDetailResponse)
def get_run(run_id: str):
    detail = get_run_detail(run_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Run not found")
    return detail
