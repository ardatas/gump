import json
import logging
from pathlib import Path

from src.common.paths import ARTIFACTS_DIR, CACHE_DIR
from src.api.schemas import TileSummary, TileArtifacts, TileSummaryMetrics, TileBounds

LOGGER = logging.getLogger(__name__)


def _load_meta(tile_id: str) -> dict | None:
    for base in [CACHE_DIR, CACHE_DIR / "gump"]:
        p = base / tile_id / "meta.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def _compute_bounds(tile_id: str) -> TileBounds | None:
    meta = _load_meta(tile_id)
    if meta is None:
        return None
    try:
        crs = meta["crs"]
        tr = meta["transform"]
        w, h = meta["width"], meta["height"]
        ox, oy, px, py = tr[2], tr[5], tr[0], tr[4]
        xs = [ox, ox + px * w]
        ys = [oy, oy + py * h]
        from pyproj import Transformer
        t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lons, lats = t.transform([min(xs), max(xs)], [min(ys), max(ys)])
        return TileBounds(min_lon=min(lons), min_lat=min(lats), max_lon=max(lons), max_lat=max(lats))
    except Exception as e:
        LOGGER.warning("Could not compute bounds for %s: %s", tile_id, e)
        return None


def _get_months(tile_id: str) -> list[tuple[int, int]] | None:
    meta = _load_meta(tile_id)
    if meta is None:
        return None
    return [tuple(m) for m in meta.get("months_s2", [])] or None

def get_tile_summary(run_id: str, tile_id: str) -> TileSummary | None:
    tile_dir = ARTIFACTS_DIR / run_id / tile_id
    if not tile_dir.exists():
        return None
        
    summary_path = tile_dir / "summary.json"
    
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TileSummary(
                run_id=data.get("run_id", run_id),
                tile_id=data.get("tile_id", tile_id),
                status=data.get("status", "ready"),
                threshold=data.get("threshold"),
                artifacts=TileArtifacts(**data.get("artifacts", {})),
                paths=data.get("paths", {}),
                metrics=TileSummaryMetrics(**data.get("metrics", {})),
                bounds=_compute_bounds(tile_id),
                months_s2=_get_months(tile_id),
            )
        except Exception as e:
            LOGGER.warning("Failed to parse summary.json in %s: %s", tile_dir, e)

    # Fallback: construct it from files if no summary exists
    has_geojson = (tile_dir / "pred.geojson").exists()
    has_overlay = (tile_dir / "overlay.png").exists()
    has_prob = (tile_dir / "prob.tif").exists()

    paths = {}
    if has_geojson: paths["geojson"] = f"/artifacts/{run_id}/{tile_id}/pred.geojson"
    if has_overlay: paths["overlay_png"] = f"/artifacts/{run_id}/{tile_id}/overlay.png"
    if has_prob: paths["prob_tif"] = f"/artifacts/{run_id}/{tile_id}/prob.tif"

    return TileSummary(
        run_id=run_id,
        tile_id=tile_id,
        status="ready",
        artifacts=TileArtifacts(
            geojson=has_geojson,
            overlay_png=has_overlay,
            prob_tif=has_prob,
        ),
        paths=paths,
        bounds=_compute_bounds(tile_id),
        months_s2=_get_months(tile_id),
    )

def get_all_tiles_for_run(run_id: str) -> list[TileSummary]:
    run_dir = ARTIFACTS_DIR / run_id
    if not run_dir.exists():
        return []
    
    summaries = []
    for tile_dir in run_dir.iterdir():
        if tile_dir.is_dir():
            s = get_tile_summary(run_id, tile_dir.name)
            if s:
                summaries.append(s)
    return summaries
