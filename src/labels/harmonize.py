"""Harmonize per-source deforestation signals into consensus label + weight."""

from __future__ import annotations

import numpy as np


def harmonize(
    sources: list[tuple[np.ndarray, np.ndarray]],
    forest_2020: np.ndarray,
    w_single: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine (is_defo, alert_year) arrays from available sources.

    Args:
        sources: list of (is_defo bool[H,W], alert_year uint16[H,W]) per source.
                 Pass only sources that were found (missing = excluded entirely).
        forest_2020: bool (H, W) mask.
        w_single: weight for pixels where only 1 source agrees (ablation knob).

    Returns:
        label_consensus (H,W) uint8  — 0/1 hard target
        label_weight    (H,W) float32 — per-pixel loss weight
        alert_year      (H,W) uint16  — earliest alert year across agreeing sources
    """
    n_avail = len(sources)
    H, W = forest_2020.shape

    if n_avail == 0:
        zeros_u8 = np.zeros((H, W), dtype=np.uint8)
        zeros_f32 = np.zeros((H, W), dtype=np.float32)
        zeros_u16 = np.zeros((H, W), dtype=np.uint16)
        return zeros_u8, zeros_f32, zeros_u16

    votes = np.zeros((H, W), dtype=np.int16)
    # Initialise alert_year accumulator with large sentinel; take min at end.
    _YEAR_MAX = 9999
    year_acc = np.full((H, W), _YEAR_MAX, dtype=np.int32)

    for is_defo, alert_year in sources:
        votes += is_defo.astype(np.int16)
        # Track earliest year from positive-voting sources
        positive_year = np.where(is_defo & (alert_year > 0), alert_year.astype(np.int32), _YEAR_MAX)
        year_acc = np.minimum(year_acc, positive_year)

    consensus = (votes >= 2).astype(np.uint8)

    # Per-pixel weight
    weight = np.ones((H, W), dtype=np.float32)  # default: 1.0 (sure negatives in forest)
    if n_avail >= 3:
        weight = np.where(votes == 3, 1.0,
                 np.where(votes == 2, 0.8,
                 np.where(votes == 1, w_single, 1.0))).astype(np.float32)
    elif n_avail == 2:
        weight = np.where(votes == 2, 1.0,
                 np.where(votes == 1, w_single, 1.0)).astype(np.float32)
    else:  # n_avail == 1: single source, cap at w_single
        weight = np.where(votes == 1, w_single, 1.0).astype(np.float32)

    # Pixels outside forest 2020 are irrelevant; zero out their weight
    weight[~forest_2020] = 0.0

    alert_year_out = np.where(year_acc < _YEAR_MAX, year_acc, 0).astype(np.uint16)

    return consensus, weight, alert_year_out
