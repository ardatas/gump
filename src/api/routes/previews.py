import io
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from src.common.paths import CACHE_DIR

LOGGER = logging.getLogger(__name__)
router = APIRouter()

# S2 band indices within the (n_months, 12, H, W) array
# Order: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
S2_BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
B = {name: i for i, name in enumerate(S2_BANDS)}

COMPOSITES = {
    "rgb":        {"bands": [B["B04"], B["B03"], B["B02"]], "src": "s2"},
    "falsecolor": {"bands": [B["B08"], B["B04"], B["B03"]], "src": "s2"},
    "swir":       {"bands": [B["B12"], B["B08"], B["B04"]], "src": "s2"},
    "redge":      {"bands": [B["B06"], B["B05"], B["B04"]], "src": "s2"},
    "ndvi":       {"bands": None,                           "src": "s2"},
    "sar":        {"bands": None,                           "src": "s1"},
}


def _cache_dir(tile_id: str) -> Path:
    for base in [CACHE_DIR, CACHE_DIR / "gump"]:
        p = base / tile_id
        if p.exists():
            return p
    return CACHE_DIR / tile_id


def _best_month(s2: np.ndarray, band_idxs: list[int]) -> int:
    """Return the month index with the highest fraction of valid pixels."""
    fracs = []
    for t in range(s2.shape[0]):
        rgb = s2[t][band_idxs].astype(np.float32)
        frac = float((np.isfinite(rgb) & (rgb > 0)).all(axis=0).mean())
        fracs.append(frac)
    return int(np.argmax(fracs))


def _stretch(arr: np.ndarray) -> np.ndarray:
    """Percentile stretch to [0, 1], ignore NaN."""
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if len(valid) == 0:
        return np.zeros_like(arr)
    p2, p98 = np.percentile(valid, [2, 98])
    span = p98 - p2
    if span < 1e-6:
        span = 1.0
    return np.clip((arr - p2) / span, 0, 1)


def _render_rgb(bands_hwc: np.ndarray) -> np.ndarray:
    """bands_hwc: (3, H, W) float32 → (H, W, 4) uint8 RGBA"""
    h, w = bands_hwc.shape[1], bands_hwc.shape[2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for c in range(3):
        rgba[:, :, c] = (_stretch(bands_hwc[c]) * 255).astype(np.uint8)
    # alpha: 0 where all bands are NaN or zero
    valid = np.isfinite(bands_hwc).all(axis=0) & (bands_hwc > 0).any(axis=0)
    rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def _render_ndvi(s2_t: np.ndarray) -> np.ndarray:
    """s2_t: (12, H, W) float32 → NDVI coloured RGBA"""
    nir = s2_t[B["B08"]].astype(np.float32)
    red = s2_t[B["B04"]].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1, 1)
    h, w = ndvi.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Colormap: -1→brown, 0→yellow, 1→dark green
    pos = (ndvi + 1) / 2.0  # 0..1
    # Brown (139,90,43) → Yellow (255,230,0) → Green (0,128,0)
    def lerp(a, b, t): return (a * (1 - t) + b * t).astype(np.uint8)
    mask_lo = pos < 0.5
    t_lo = (pos / 0.5)
    t_hi = ((pos - 0.5) / 0.5)
    r = np.where(mask_lo, lerp(139, 255, t_lo), lerp(255, 0, t_hi)).astype(np.uint8)
    g = np.where(mask_lo, lerp(90, 230, t_lo), lerp(230, 128, t_hi)).astype(np.uint8)
    b_ = np.where(mask_lo, lerp(43, 0, t_lo), lerp(0, 0, t_hi)).astype(np.uint8)
    rgba[:, :, 0] = r; rgba[:, :, 1] = g; rgba[:, :, 2] = b_
    valid = np.isfinite(nir) & np.isfinite(red) & (nir + red > 0)
    rgba[:, :, 3] = np.where(valid, 220, 0).astype(np.uint8)
    return rgba


def _render_sar(s1: np.ndarray, month_idx: int | None) -> np.ndarray:
    """s1: (T, 1, H, W) float32, returns greyscale RGBA"""
    if s1.shape[0] == 0:
        raise ValueError("no S1 data")
    # find best S1 month if not specified
    if month_idx is None:
        fracs = []
        for t in range(s1.shape[0]):
            ch = s1[t, 0]
            fracs.append(float(np.isfinite(ch).mean()))
        month_idx = int(np.argmax(fracs))
    month_idx = min(month_idx, s1.shape[0] - 1)
    ch = s1[month_idx, 0].astype(np.float32)
    grey = (_stretch(ch) * 255).astype(np.uint8)
    h, w = grey.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = grey; rgba[:, :, 1] = grey; rgba[:, :, 2] = grey
    rgba[:, :, 3] = np.where(np.isfinite(ch), 220, 0).astype(np.uint8)
    return rgba


def _to_png(rgba: np.ndarray) -> bytes:
    from PIL import Image
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


@router.get("/{tile_id}/preview.png")
def get_tile_preview(
    tile_id: str,
    composite: str = Query(default="rgb"),
    month: str = Query(default="auto"),
):
    if composite not in COMPOSITES:
        raise HTTPException(400, f"Unknown composite '{composite}'. Choose from: {list(COMPOSITES)}")

    tile_dir = _cache_dir(tile_id)
    if not tile_dir.exists():
        raise HTTPException(404, f"Tile '{tile_id}' not found in cache")

    # Check PNG cache
    cache_key = f"{composite}_{month}"
    png_cache = tile_dir / f"preview_{cache_key}.png"
    if png_cache.exists():
        return Response(content=png_cache.read_bytes(), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})

    cfg = COMPOSITES[composite]

    try:
        if cfg["src"] == "s2":
            s2_path = tile_dir / "s2.npy"
            if not s2_path.exists():
                raise HTTPException(404, "s2.npy not found for this tile")
            s2 = np.load(s2_path).astype(np.float32)  # (T, 12, H, W)

            if composite == "ndvi":
                ref_bands = [B["B08"], B["B04"]]
                t = _best_month(s2, ref_bands) if month == "auto" else min(int(month), s2.shape[0] - 1)
                rgba = _render_ndvi(s2[t])
            else:
                band_idxs = cfg["bands"]
                t = _best_month(s2, band_idxs) if month == "auto" else min(int(month), s2.shape[0] - 1)
                rgba = _render_rgb(s2[t][band_idxs])

        else:  # sar
            s1_path = tile_dir / "s1_asc.npy"
            if not s1_path.exists():
                raise HTTPException(404, "s1_asc.npy not found for this tile")
            s1 = np.load(s1_path).astype(np.float32)
            t = None if month == "auto" else int(month)
            rgba = _render_sar(s1, t)

        png_bytes = _to_png(rgba)
        png_cache.write_bytes(png_bytes)
        return Response(content=png_bytes, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("Preview render failed for %s/%s", tile_id, composite)
        raise HTTPException(500, str(e))
