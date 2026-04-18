from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.common.paths import ensure_workspace_dirs

from src.api.routes import health, runs, tiles, jobs

# Initialize workspace structure
ensure_workspace_dirs()

app = FastAPI(
    title="GUMP API",
    description="Backend API for GUMP AI Platform",
    version="1.0.0"
)

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the Next.js domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["System"])
app.include_router(runs.router, prefix="/api/runs", tags=["Runs"])
app.include_router(tiles.router, prefix="/api/runs", tags=["Tiles"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
