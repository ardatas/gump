from fastapi import APIRouter
from src.api.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def get_health():
    return HealthResponse(status="ok", version="1.0.0")
