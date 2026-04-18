from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.common.paths import ensure_workspace_dirs, ARTIFACTS_DIR

from src.api.routes import health, runs, tiles, jobs, previews

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

# Serve static artifacts (like GeoJSON and PNG overlays)
app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")

app.include_router(health.router, prefix="/api", tags=["System"])
app.include_router(runs.router, prefix="/api/runs", tags=["Runs"])
app.include_router(tiles.router, prefix="/api/runs", tags=["Tiles"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(previews.router, prefix="/api/tiles", tags=["Previews"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
