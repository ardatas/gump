from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from rasterio import Affine
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_holes, remove_small_objects


@dataclass(frozen=True)
class PostprocessConfig:
    threshold: float
    closing_radius: int = 0
    opening_radius: int = 0
    min_area_ha: float = 0.5
    min_hole_area_ha: float = 0.0
    persistence_threshold: float = 0.0


def postprocess_probability_map(
    probability_map: np.ndarray,
    *,
    forest_mask: np.ndarray,
    transform: Affine,
    config: PostprocessConfig,
    persistence_score: np.ndarray | None = None,
) -> np.ndarray:
    prediction = probability_map >= config.threshold
    prediction &= forest_mask

    if persistence_score is not None and config.persistence_threshold > 0.0:
        prediction &= persistence_score >= config.persistence_threshold

    if config.closing_radius > 0:
        prediction = binary_closing(prediction, footprint=disk(config.closing_radius))

    if config.opening_radius > 0:
        prediction = binary_opening(prediction, footprint=disk(config.opening_radius))

    min_pixels = hectares_to_pixels(config.min_area_ha, transform)
    if min_pixels > 1:
        prediction = remove_small_objects(prediction.astype(bool), min_size=min_pixels)

    hole_pixels = hectares_to_pixels(config.min_hole_area_ha, transform)
    if hole_pixels > 1:
        prediction = remove_small_holes(prediction.astype(bool), area_threshold=hole_pixels)

    prediction &= forest_mask
    return prediction.astype(bool)


def hectares_to_pixels(area_ha: float, transform: Affine) -> int:
    if area_ha <= 0:
        return 0
    pixel_area_m2 = pixel_area_square_meters(transform)
    if pixel_area_m2 <= 0:
        raise ValueError("Pixel area must be positive.")
    return max(1, int(math.ceil((area_ha * 10_000.0) / pixel_area_m2)))


def pixel_area_square_meters(transform: Affine) -> float:
    return float(abs((transform.a * transform.e) - (transform.b * transform.d)))
