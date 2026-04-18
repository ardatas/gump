from fastapi import APIRouter, HTTPException
from src.api.schemas import TileListResponse, TileDetailResponse
from src.registry.scan_tiles import get_tile_summary, get_all_tiles_for_run

router = APIRouter()

@router.get("/{run_id}/tiles", response_model=TileListResponse)
def list_tiles(run_id: str):
    tiles = get_all_tiles_for_run(run_id)
    return TileListResponse(run_id=run_id, tiles=tiles)

@router.get("/{run_id}/tiles/{tile_id}", response_model=TileDetailResponse)
def get_tile(run_id: str, tile_id: str):
    summary = get_tile_summary(run_id, tile_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Tile or artifacts not found")
    return TileDetailResponse(run_id=run_id, tile_id=tile_id, summary=summary)
