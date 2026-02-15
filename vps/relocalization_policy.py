#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VPS relocalization policy helpers (import-light, unit-test friendly)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def offset_latlon_m(center_lat: float, center_lon: float, east_m: float, north_m: float) -> Tuple[float, float]:
    """Approximate ENU meter offsets to lat/lon around current latitude."""
    m_per_deg_lat = 111320.0
    m_per_deg_lon = max(1e-6, 111320.0 * np.cos(np.radians(float(center_lat))))
    d_lat = float(north_m) / m_per_deg_lat
    d_lon = float(east_m) / m_per_deg_lon
    return float(center_lat + d_lat), float(center_lon + d_lon)


def extract_xy_sigma_m(est_cov_xy: Optional[np.ndarray]) -> float:
    """Return representative XY sigma [m] from covariance matrix."""
    if est_cov_xy is None:
        return float("nan")
    try:
        c = np.array(est_cov_xy, dtype=float)
        if c.shape != (2, 2) or not np.all(np.isfinite(c)):
            return float("nan")
        c = 0.5 * (c + c.T)
        evals = np.linalg.eigvalsh(c)
        evals = np.clip(evals, 0.0, 1e12)
        return float(np.sqrt(float(np.max(evals))))
    except Exception:
        return float("nan")


def should_force_global_relocalization(*,
                                       force_global: bool,
                                       accuracy_mode: bool,
                                       objective: str,
                                       reloc_enabled: bool,
                                       fail_streak: int,
                                       fail_streak_trigger: int,
                                       since_success_sec: float,
                                       stale_success_sec: float,
                                       since_global_sec: float,
                                       global_interval_sec: float,
                                       est_cov_xy: Optional[np.ndarray],
                                       xy_sigma_trigger_m: float,
                                       phase: Optional[int] = None,
                                       force_global_on_warning_phase: bool = False) -> Tuple[bool, str]:
    """Decision policy for switching to global relocalization search."""
    if force_global:
        return True, "forced"
    if not bool(reloc_enabled):
        return False, "disabled"

    phase_now = int(phase) if phase is not None else 2
    if bool(force_global_on_warning_phase) and phase_now <= 1:
        return True, "phase_warning"

    if int(fail_streak) >= max(1, int(fail_streak_trigger)):
        return True, "fail_streak"
    if np.isfinite(float(since_success_sec)) and float(since_success_sec) >= max(0.0, float(stale_success_sec)):
        return True, "stale_success"

    xy_sigma = extract_xy_sigma_m(est_cov_xy)
    if np.isfinite(xy_sigma) and xy_sigma >= max(1e-3, float(xy_sigma_trigger_m)):
        return True, "high_covariance"

    objective_mode = str(objective).lower()
    if bool(accuracy_mode) or objective_mode == "accuracy":
        if np.isfinite(float(since_global_sec)) and float(since_global_sec) >= max(0.0, float(global_interval_sec)):
            return True, "periodic_global"

    return False, "local"


def build_relocalization_centers(est_lat: float,
                                 est_lon: float,
                                 max_centers: int,
                                 ring_radius_m: List[float],
                                 ring_samples: int) -> List[Tuple[float, float]]:
    """Generate candidate search centers (local center + ring offsets)."""
    centers: List[Tuple[float, float]] = [(float(est_lat), float(est_lon))]
    seen = {(round(float(est_lat), 8), round(float(est_lon), 8))}
    max_centers = max(1, int(max_centers))
    ring_samples = max(4, int(ring_samples))
    radii = [float(r) for r in (ring_radius_m or []) if np.isfinite(r) and float(r) > 0.0]

    for radius in radii:
        for k in range(ring_samples):
            ang = 2.0 * np.pi * float(k) / float(ring_samples)
            east = float(radius) * float(np.cos(ang))
            north = float(radius) * float(np.sin(ang))
            cand = offset_latlon_m(est_lat, est_lon, east, north)
            key = (round(cand[0], 8), round(cand[1], 8))
            if key in seen:
                continue
            seen.add(key)
            centers.append(cand)
            if len(centers) >= max_centers:
                return centers
    return centers
