"""Sanity checks for label artefacts (PLAN_B §8).

Usage::

    python -m src.labels.validate [--tile 18NWG_6_6 ...]
    # exit 0 = all checks pass, exit 1 = failures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_tile(tile_dir: Path) -> list[str]:
    errors = []
    meta = json.loads((tile_dir / "meta.json").read_text())
    H, W = meta["height"], meta["width"]

    import numpy as np

    # 1. forest_2020 shape
    f = np.load(tile_dir / "forest_2020.npy")
    if f.shape != (H, W):
        errors.append(f"forest_2020 shape {f.shape} != ({H},{W})")
    if not (0.001 < f.mean() < 0.999):
        errors.append(f"forest_2020 mean {f.mean():.3f} out of expected [0.001, 0.999]")

    labels_meta_p = tile_dir / "labels_meta.json"
    if not labels_meta_p.exists():
        return errors  # test tile, no labels
    lm = json.loads(labels_meta_p.read_text())
    if lm.get("forest_only"):
        return errors

    # 2. consensus shape + value range
    c = np.load(tile_dir / "label_consensus.npy")
    if c.shape != (H, W):
        errors.append(f"label_consensus shape {c.shape} != ({H},{W})")
    if not np.isin(c, [0, 1]).all():
        errors.append("label_consensus has values outside {0, 1}")
    if not (0.0 <= c.mean() < 0.70):
        errors.append(f"label_consensus mean {c.mean():.4f} out of expected [0, 0.70]")

    # 3. weight discrete values (float32 can't represent 0.3/0.8 exactly)
    w = np.load(tile_dir / "label_weight.npy")
    if w.shape != (H, W):
        errors.append(f"label_weight shape {w.shape} != ({H},{W})")
    allowed = np.array([0.0, 0.3, 0.5, 0.8, 1.0], dtype=np.float32)
    unique_w = np.unique(w)
    for v in unique_w:
        if not np.any(np.isclose(v, allowed, atol=1e-4)):
            errors.append(f"label_weight has unexpected value: {v}")
    if (w[f] == 0).all():
        errors.append("label_weight is all-zero inside forest — likely bug")

    # 4. alert_year range
    ay = np.load(tile_dir / "alert_year.npy")
    if ay.shape != (H, W):
        errors.append(f"alert_year shape {ay.shape} != ({H},{W})")
    nonzero = ay[ay > 0]
    if len(nonzero) > 0:
        bad = nonzero[(nonzero < 2020) | (nonzero > 2030)]
        if len(bad):
            errors.append(f"alert_year has {len(bad)} values outside [2020,2030]: sample={bad[:5]}")

    # 5. weight=0 exactly where forest==0
    if not (w[~f] == 0).all():
        errors.append("label_weight != 0 outside forest_2020 — harmonize bug")

    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=Path("/workspace/cache"))
    ap.add_argument("--tile", nargs="*", dest="tiles")
    args = ap.parse_args()

    cache_root = args.cache_root
    if args.tiles:
        dirs = [cache_root / t for t in args.tiles]
    else:
        dirs = [d for d in sorted(cache_root.iterdir()) if (d / "labels_meta.json").exists()]

    all_errors: dict[str, list[str]] = {}
    for d in dirs:
        errs = check_tile(d)
        if errs:
            all_errors[d.name] = errs

    if all_errors:
        for tile, errs in all_errors.items():
            for e in errs:
                print(f"FAIL {tile}: {e}")
        sys.exit(1)
    else:
        print(f"OK — {len(dirs)} tile(s) passed all checks")
        sys.exit(0)


if __name__ == "__main__":
    main()
