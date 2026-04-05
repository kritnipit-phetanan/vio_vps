#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPS Integration Module

Handles Visual Positioning System (VPS) updates with adaptive uncertainty
and innovation gating. Includes DEM (Digital Elevation Model) altitude updates.

Author: VIO project
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R_scipy

from .ekf import ExtendedKalmanFilter, regularize_innovation_covariance
from .data_loaders import VPSItem, ProjectionCache
from .math_utils import safe_matrix_inverse


def _sanitize_vps_covariance(
    R_vps: np.ndarray,
    *,
    min_sigma_xy_m: float = 0.0,
) -> np.ndarray:
    """Return a finite symmetric 2x2 VPS covariance with a configurable XY floor."""
    R_vps = np.asarray(R_vps, dtype=float)
    if R_vps.shape != (2, 2) or not np.all(np.isfinite(R_vps)):
        raise ValueError("invalid VPS covariance")
    R_vps = 0.5 * (R_vps + R_vps.T)
    diag_floor = float(max(1e-6, float(min_sigma_xy_m) ** 2))
    diag = np.clip(np.diag(R_vps), diag_floor, 1e8)
    offdiag = float(R_vps[0, 1]) if np.isfinite(R_vps[0, 1]) else 0.0
    offdiag_limit = 0.95 * float(np.sqrt(max(diag[0] * diag[1], 1e-12)))
    offdiag = float(np.clip(offdiag, -offdiag_limit, offdiag_limit))
    return np.array(
        [
            [float(diag[0]), offdiag],
            [offdiag, float(diag[1])],
        ],
        dtype=float,
    )


def _compute_vps_mahalanobis_gate(
    *,
    xy_ref: np.ndarray,
    p_xy: np.ndarray,
    z_xy: np.ndarray,
    R_vps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute innovation, innovation covariance, and Mahalanobis distance for VPS XY."""
    xy_ref = np.asarray(xy_ref, dtype=float).reshape(2,)
    z_xy = np.asarray(z_xy, dtype=float).reshape(2,)
    p_xy = np.asarray(p_xy, dtype=float)
    if p_xy.shape != (2, 2) or not np.all(np.isfinite(p_xy)):
        raise ValueError("invalid VPS covariance block")
    p_xy = 0.5 * (p_xy + p_xy.T)
    evals, evecs = np.linalg.eigh(np.clip(p_xy, -1e8, 1e8))
    evals = np.clip(evals, 1e-9, 1e8)
    p_xy = evecs @ np.diag(evals) @ evecs.T
    innovation = (z_xy - xy_ref).reshape(-1, 1)
    S = p_xy + np.asarray(R_vps, dtype=float)
    S, _ = regularize_innovation_covariance(
        S,
        base_epsilon=1e-8,
        rel_epsilon=1e-9,
        max_epsilon=1e-3,
    )
    S_inv = safe_matrix_inverse(S, damping=1e-9, method="cholesky")
    mahalanobis_sq = float((innovation.T @ S_inv @ innovation).item())
    return innovation, S, mahalanobis_sq


def compute_vps_innovation(vps: VPSItem, kf: ExtendedKalmanFilter,
                           lat0: float, lon0: float,
                           proj_cache: ProjectionCache,
                           sigma_vps: float = 1.0,
                           min_sigma_xy_m: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute VPS innovation and S matrix for gating.
    
    Args:
        vps: VPS measurement
        kf: Extended Kalman Filter
        lat0: Origin latitude
        lon0: Origin longitude
        proj_cache: ProjectionCache instance for coordinate conversion
    
    Returns:
        vps_xy: VPS position in local frame
        innovation: Position innovation
        m2_test: Mahalanobis distance squared
    """
    vps_xy = proj_cache.latlon_to_xy(vps.lat, vps.lon, lat0, lon0)
    
    # ESKF Jacobian
    num_clones = (kf.x.shape[0] - 19) // 7  # v3.9.7: 19D nominal
    err_dim = 18 + 6 * num_clones  # v3.9.7: 18D core error
    h_xy = np.zeros((2, err_dim), dtype=float)
    h_xy[0, 0] = 1.0  # δp_x
    h_xy[1, 1] = 1.0  # δp_y
    
    # Adaptive measurement noise based on speed
    speed_xy = float(np.hypot(kf.x[3, 0], kf.x[4, 0]))
    sigma_vps = float(max(float(sigma_vps), float(min_sigma_xy_m), 1e-3))
    scale = 1.0 + max(0.0, (speed_xy - 10.0) / 10.0) if speed_xy > 10 else 1.0
    r_mat = _sanitize_vps_covariance(
        np.diag([(sigma_vps**2) * scale, (sigma_vps**2) * scale]),
        min_sigma_xy_m=min_sigma_xy_m,
    )
    
    # Innovation
    s_mat = h_xy @ kf.P @ h_xy.T + r_mat
    xy_pred = kf.x[0:2].reshape(2,)
    innovation = (vps_xy - xy_pred).reshape(-1, 1)
    
    try:
        s_mat, _ = regularize_innovation_covariance(
            s_mat,
            base_epsilon=1e-8,
            rel_epsilon=1e-9,
            max_epsilon=1e-3,
        )
        s_inv = safe_matrix_inverse(s_mat, damping=1e-9, method='cholesky')
        m2_test = float((innovation.T @ s_inv @ innovation).item())
    except (np.linalg.LinAlgError, ValueError):
        m2_test = np.inf
    
    return vps_xy, innovation, m2_test


def compute_vps_acceptance_threshold(time_since_correction: float,
                                      innovation_mag: float) -> Tuple[float, float, str]:
    """
    Compute adaptive VPS acceptance threshold based on drift time.
    
    Multi-tier strategy:
      - Tier 1 (>60s): Very long drift - accept very large innovations
      - Tier 2 (10-60s): Long drift - permissive threshold
      - Tier 3a (3-10s): Medium drift - moderate threshold
      - Tier 3b (<3s): Recent VPS - strict threshold
    
    Args:
        time_since_correction: Seconds since last absolute correction
        innovation_mag: Innovation magnitude in meters
    
    Returns:
        max_innovation_m: Maximum allowed innovation
        r_scale: Measurement noise scaling factor
        tier_name: Name of acceptance tier
    """
    if time_since_correction > 60.0:
        # TIER 1: Very long drift
        base_threshold_m = 150.0
        max_drift_rate = 6.0  # m/s equivalent budget
        r_scale = min(12.0, 6.0 + time_since_correction / 20.0)
        tier_name = "FIRST VPS"
    elif time_since_correction > 10.0:
        # TIER 2: Long drift
        base_threshold_m = 90.0
        max_drift_rate = 5.0
        r_scale = min(8.0, 1.0 + time_since_correction / 6.0)
        tier_name = "LONG DRIFT"
    elif time_since_correction > 3.0:
        # TIER 3a: Medium drift
        base_threshold_m = 55.0
        max_drift_rate = 4.0
        r_scale = min(5.0, 1.0 + time_since_correction / 4.0)
        tier_name = "MEDIUM DRIFT"
    else:
        # TIER 3b: Recent VPS
        base_threshold_m = 35.0
        max_drift_rate = 3.0
        r_scale = 1.0
        tier_name = "RECENT"
    
    max_innovation_m = base_threshold_m + max_drift_rate * time_since_correction
    max_innovation_m = min(max_innovation_m, 800.0)  # Keep delayed absolute updates bounded
    
    return max_innovation_m, r_scale, tier_name


def apply_vps_update(kf: ExtendedKalmanFilter, vps_xy: np.ndarray,
                     sigma_vps: float = 1.0, r_scale: float = 1.0) -> bool:
    """
    Apply VPS position update to EKF.
    
    Args:
        kf: Extended Kalman Filter
        vps_xy: VPS position in local frame [x, y]
        sigma_vps: Base VPS measurement noise
        r_scale: Noise scaling factor
    
    Returns:
        success: True if update was applied
    """
    num_clones = (kf.x.shape[0] - 19) // 7  # v3.9.7: 19D nominal
    err_dim = 18 + 6 * num_clones  # v3.9.7: 18D core error
    
    h_xy = np.zeros((2, err_dim), dtype=float)
    h_xy[0, 0] = 1.0
    h_xy[1, 1] = 1.0
    
    def h_fun(x, h=h_xy):
        return h
    
    def hx_fun(x, h=h_xy):
        return x[0:2].reshape(2, 1)
    
    r_mat = _sanitize_vps_covariance(
        np.diag([(sigma_vps * r_scale)**2, (sigma_vps * r_scale)**2]),
        min_sigma_xy_m=sigma_vps,
    )
    
    try:
        kf.update(
            z=vps_xy.reshape(-1, 1),
            HJacobian=h_fun,
            Hx=hx_fun,
            R=r_mat,
            update_type="VPS",
            timestamp=float('nan')
        )
        return True
    except Exception as e:
        print(f"[VPS] Update failed: {e}")
        return False


def apply_height_update(kf: ExtendedKalmanFilter, height_m: float,
                        sigma_agl_z: float = 2.5,
                        height_cov_scale: float = 1.0) -> Tuple[bool, float, float]:
    """
    Apply DEM/height measurement update to EKF.
    
    Args:
        kf: Extended Kalman Filter
        height_m: Height measurement
        sigma_agl_z: Base height measurement noise
        height_cov_scale: Noise scaling factor
    
    Returns:
        applied: True if update was applied
        innovation: Height innovation
        m2_test: Mahalanobis distance squared
    """
    if np.isnan(height_m):
        return False, 0.0, np.inf
    
    num_clones = (kf.x.shape[0] - 19) // 7  # v3.9.7: 19D nominal
    err_dim = 18 + 6 * num_clones  # v3.9.7: 18D core error
    
    h_height = np.zeros((1, err_dim), dtype=float)
    h_height[0, 2] = 1.0  # δp_z
    
    def h_fun(x, h=h_height):
        return h
    
    def hx_fun(x, h=h_height):
        return x[2:3].reshape(1, 1)
    
    r_mat = np.array([[sigma_agl_z**2 * height_cov_scale]])
    
    # Innovation and gating
    s_mat = h_height @ kf.P @ h_height.T + r_mat
    predicted_height = kf.x[2, 0]
    innovation = height_m - predicted_height
    
    try:
        m2_test = float((innovation**2) / s_mat[0, 0])
    except (ZeroDivisionError, FloatingPointError):
        m2_test = np.inf
    
    # Adaptive threshold
    xy_std = np.sqrt(np.trace(kf.P[0:2, 0:2]) / 2.0)
    if xy_std > 10.0:
        threshold = 15.0
    elif xy_std > 5.0:
        threshold = 9.21  # 99.9% confidence
    else:
        threshold = 6.63  # 99% confidence
    
    if m2_test < threshold:
        kf.update(
            z=np.array([[height_m]]),
            HJacobian=h_fun,
            Hx=hx_fun,
            R=r_mat,
            update_type="DEM",
            timestamp=float('nan')
        )
        return True, innovation, m2_test
    else:
        return False, innovation, m2_test


# =============================================================================
# Homography Scale Estimation (for nadir cameras)
# =============================================================================

def estimate_homography_scale(pts1: np.ndarray, pts2: np.ndarray, 
                              k_matrix: np.ndarray, altitude: float, 
                              r_rel: np.ndarray,
                              min_inliers: int = 15) -> Optional[Tuple[float, np.ndarray, int]]:
    """
    Estimate scale from planar homography for nadir cameras over flat terrain.
    
    Theory (Planar Homography):
    For points on a plane π with normal n and distance d from camera:
    H = K @ (R + t @ n^T / d) @ K^{-1}
    
    For nadir camera over ground plane:
    - Rotation R: relative rotation between frames
    - Translation t: [tx, ty, tz] where tz ≈ altitude change
    - Distance d: altitude above ground (AGL)
    
    Args:
        pts1: Feature points in frame 1 (N×2, normalized coordinates)
        pts2: Feature points in frame 2 (N×2, normalized coordinates)
        k_matrix: Camera intrinsic matrix (3×3)
        altitude: Estimated altitude above ground (meters)
        r_rel: Relative rotation between frames (3×3)
        min_inliers: Minimum inliers for valid homography
    
    Returns:
        (scale, t_scaled, num_inliers) or None if estimation fails
    """
    import cv2
    
    if len(pts1) < min_inliers or len(pts2) < min_inliers:
        return None
    
    # Convert normalized to pixel coordinates
    pts1_px = (k_matrix @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T
    pts2_px = (k_matrix @ np.hstack([pts2, np.ones((len(pts2), 1))]).T).T
    pts1_px = pts1_px[:, :2] / pts1_px[:, 2:3]
    pts2_px = pts2_px[:, :2] / pts2_px[:, 2:3]
    
    try:
        h_mat, mask = cv2.findHomography(
            pts1_px, pts2_px,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000,
            confidence=0.99
        )
        
        if h_mat is None or mask is None:
            return None
        
        num_inliers = int(np.sum(mask))
        if num_inliers < min_inliers:
            return None
        
        # Decompose homography
        k_inv = np.linalg.inv(k_matrix)
        h_normalized = k_inv @ h_mat @ k_matrix
        
        # Extract t/d
        r_col2 = r_rel[:, 2]
        t_over_d = h_normalized[:, 2] - r_col2
        
        # Scale recovery
        t_scaled = t_over_d * altitude
        scale = np.linalg.norm(t_scaled)
        
        # Sanity check
        if scale > 100.0 or scale < 0.01:
            return None
        
        return (scale, t_scaled, num_inliers)
        
    except (cv2.error, np.linalg.LinAlgError):
        return None


def compute_plane_constraint_jacobian(kf: ExtendedKalmanFilter, 
                                      altitude: float) -> Tuple[np.ndarray, float]:
    """
    Compute Jacobian for plane constraint measurement.
    
    Args:
        kf: Extended Kalman Filter
        altitude: Measured altitude from AGL/DEM (meters)
    
    Returns:
        h_matrix: Jacobian matrix (1, err_state_size)
        predicted_altitude: Current altitude from state
    """
    p_world = kf.x[0:3, 0]
    predicted_altitude = p_world[2]
    
    num_clones = (kf.x.shape[0] - 19) // 7  # v3.9.7: 19D nominal
    err_state_size = 18 + 6 * num_clones  # v3.9.7: 18D core error
    
    h_matrix = np.zeros((1, err_state_size), dtype=float)
    h_matrix[0, 2] = 1.0
    
    return h_matrix, predicted_altitude


# =============================================================================
# Stochastic Cloning - Delayed VPS Update
# =============================================================================

def apply_vps_delayed_update(
    kf: ExtendedKalmanFilter,
    clone_manager,  # VPSDelayedUpdateManager
    image_id: str,
    vps_lat: float,
    vps_lon: float,
    R_vps: np.ndarray,
    proj_cache: ProjectionCache,
    lat0: float,
    lon0: float,
    time_since_last_vps: float = 0.0,
    verbose: bool = False,
    mahalanobis_gate_enable: bool = True,
    mahalanobis_chi2_threshold: float = 9.21,
    mahalanobis_min_sigma_xy_m: float = 5.0,
    force_feed_enable: bool = False,
    bounded_correction_enable: bool = True,
    bounded_correction_pull_gain: float = 0.10,
    bounded_correction_min_pull_m: float = 1.0,
    bounded_correction_max_pull_m: float = 2.0,
    bounded_correction_r_inflate_min_mult: float = 1.5,
    bounded_correction_r_inflate_max_mult: float = 4.0,
    bounded_correction_p_xy_cap_m2: float = 25.0,
    bounded_correction_max_position_correction_m: float = 2.0,
) -> Tuple[bool, Optional[float], str, float]:
    """
    Apply delayed VPS update using Stochastic Cloning.
    
    This is the main entry point for VPS updates with latency handling.
    Uses the VPSDelayedUpdateManager to apply updates to cloned states
    and propagate corrections to the current state.
    
    Args:
        kf: ExtendedKalmanFilter instance (will be modified)
        clone_manager: VPSDelayedUpdateManager with pending clone
        image_id: Image identifier from clone_state()
        vps_lat: VPS measured latitude
        vps_lon: VPS measured longitude
        R_vps: 2x2 measurement covariance matrix (m²)
        proj_cache: ProjectionCache for coordinate conversion
        lat0: Origin latitude for local frame
        lon0: Origin longitude for local frame
        time_since_last_vps: Time since last VPS update (for gating)
        verbose: Print debug information
        
    Returns:
        Tuple of:
            - success: True if update was applied
            - innovation_mag: Innovation magnitude in meters (or None)
            - status: Status message for logging
            - mahalanobis_sq: Squared Mahalanobis distance used by the gate
            
    Example:
        from vps import VPSDelayedUpdateManager
        
        manager = VPSDelayedUpdateManager(max_clones=3)
        
        # At camera capture
        manager.clone_state(kf, t_capture, image_id="frame_001")
        
        # ... IMU propagation loop ...
        # manager.propagate_cross_covariance(F_k)
        
        # When VPS result arrives
        success, innov, msg = apply_vps_delayed_update(
            kf, manager, "frame_001",
            vps_lat=45.123, vps_lon=-75.456,
            R_vps=np.diag([1.0, 1.0]),
            proj_cache=proj_cache, lat0=lat0, lon0=lon0
        )
    """
    # Check if clone exists
    if not clone_manager.has_pending_clone(image_id):
        return False, None, f"REJECTED: Clone '{image_id}' not found (expired?)", np.nan

    try:
        r_check = _sanitize_vps_covariance(
            np.array(R_vps, dtype=float),
            min_sigma_xy_m=mahalanobis_min_sigma_xy_m,
        )
    except Exception:
        return False, None, "REJECTED: Bad R_vps", np.nan
    
    # Get clone age for logging
    clone_age = clone_manager.get_clone_age(image_id, t_now=0.0)  # Relative age
    
    # Compute expected position for gating check
    vps_xy = proj_cache.latlon_to_xy(vps_lat, vps_lon, lat0, lon0)
    current_xy = kf.x[0:2, 0]
    pre_innovation = np.linalg.norm(vps_xy - current_xy)
    if not np.isfinite(pre_innovation):
        clone_manager.clones.pop(image_id, None)
        return False, None, "REJECTED: Non-finite innovation", np.nan
    
    # Adaptive gating based on time since last VPS
    max_innovation_m, r_scale, tier_name = compute_vps_acceptance_threshold(
        time_since_last_vps, pre_innovation
    )
    
    # Gate check
    if pre_innovation > max_innovation_m:
        if not bool(force_feed_enable):
            clone_manager.clones.pop(image_id, None)  # Remove the useless clone
            return False, pre_innovation, f"GATED: Innovation {pre_innovation:.1f}m > {max_innovation_m:.1f}m ({tier_name})", np.nan
    
    # Scale R_vps if needed
    R_vps_scaled = _sanitize_vps_covariance(
        np.asarray(r_check, dtype=float) * float(r_scale),
        min_sigma_xy_m=mahalanobis_min_sigma_xy_m,
    )

    mahalanobis_sq = np.nan
    status_notes: list[str] = []
    vps_lat_apply = float(vps_lat)
    vps_lon_apply = float(vps_lon)
    R_vps_apply = np.array(R_vps_scaled, dtype=float, copy=True)
    position_correction_cap_apply = (
        float(bounded_correction_max_position_correction_m)
        if bool(bounded_correction_enable)
        else None
    )
    if bool(mahalanobis_gate_enable):
        try:
            clone = clone_manager.clones.get(image_id)
            if clone is None:
                raise ValueError("clone disappeared")
            clone_xy = np.asarray(clone.x_clone[0:2, 0], dtype=float).reshape(2,)
            clone_p_xy = np.asarray(clone.P_clone[0:2, 0:2], dtype=float)
            clone_innovation, _, mahalanobis_sq = _compute_vps_mahalanobis_gate(
                xy_ref=clone_xy,
                p_xy=clone_p_xy,
                z_xy=vps_xy,
                R_vps=R_vps_scaled,
            )
            clone_innovation_mag = float(np.linalg.norm(clone_innovation))
            if bool(bounded_correction_enable):
                pull_gain = float(np.clip(float(bounded_correction_pull_gain), 1e-3, 1.0))
                min_pull = float(max(0.1, float(bounded_correction_min_pull_m)))
                max_pull = float(max(min_pull, float(bounded_correction_max_pull_m)))
                applied_pull = float(
                    np.clip(float(clone_innovation_mag) * pull_gain, min_pull, max_pull)
                )
                if np.isfinite(clone_innovation_mag):
                    position_correction_cap_apply = min(
                        float(position_correction_cap_apply or max_pull),
                        float(max(0.1, min(float(clone_innovation_mag), max_pull))),
                    )
                if np.isfinite(clone_innovation_mag) and clone_innovation_mag > applied_pull:
                    pull_scale = float(applied_pull / max(clone_innovation_mag, 1e-9))
                    bounded_xy = clone_xy + np.asarray(clone_innovation[:, 0], dtype=float) * pull_scale
                    vps_lat_apply, vps_lon_apply = proj_cache.xy_to_latlon(
                        float(bounded_xy[0]),
                        float(bounded_xy[1]),
                        lat0,
                        lon0,
                    )
                    position_correction_cap_apply = float(applied_pull)
                    inflate_min = float(
                        max(1.0, float(bounded_correction_r_inflate_min_mult))
                    )
                    inflate_mult = float(
                        np.clip(
                            clone_innovation_mag / max(applied_pull, 1e-6),
                            inflate_min,
                            max(1.0, float(bounded_correction_r_inflate_max_mult)),
                        )
                    )
                    R_vps_apply = _sanitize_vps_covariance(
                        np.asarray(R_vps_scaled, dtype=float) * inflate_mult,
                        min_sigma_xy_m=mahalanobis_min_sigma_xy_m,
                    )
                    status_notes.append(
                        "ELASTIC_BOUND("
                        f"raw={clone_innovation_mag:.2f}m->used={applied_pull:.2f}m,"
                        f"gain={pull_gain:.2f},r*={inflate_mult:.2f})"
                    )
            if (not np.isfinite(mahalanobis_sq)) or mahalanobis_sq > float(mahalanobis_chi2_threshold):
                if not bool(force_feed_enable):
                    clone_manager.clones.pop(image_id, None)
                    gate_innovation_mag = float(np.linalg.norm(clone_innovation))
                    return (
                        False,
                        gate_innovation_mag,
                        (
                            "GATED_MAHALANOBIS: "
                            f"chi2={mahalanobis_sq:.2f} > {float(mahalanobis_chi2_threshold):.2f}, "
                            f"innov={gate_innovation_mag:.2f}m ({tier_name})"
                        ),
                        float(mahalanobis_sq),
                    )
                status_notes.append(
                    f"FORCE_ACCEPT_MAHALANOBIS(chi2={mahalanobis_sq:.2f}>{float(mahalanobis_chi2_threshold):.2f})"
                )
        except Exception:
            clone_manager.clones.pop(image_id, None)
            return False, pre_innovation, "REJECTED: Mahalanobis gate failure", np.nan
    
    # Apply the delayed update
    success, innovation_mag = clone_manager.apply_delayed_update(
        kf=kf,
        image_id=image_id,
        vps_lat=vps_lat_apply,
        vps_lon=vps_lon_apply,
        R_vps=R_vps_apply,
        proj_cache=proj_cache,
        lat0=lat0,
        lon0=lon0,
        p_xy_cap_m2=float(bounded_correction_p_xy_cap_m2) if bool(bounded_correction_enable) else None,
        max_position_correction_m_override=(
            float(position_correction_cap_apply)
            if bool(bounded_correction_enable) and position_correction_cap_apply is not None
            else None
        ),
    )
    
    if success:
        status = f"APPLIED: Innovation {innovation_mag:.2f}m, R_scale={r_scale:.1f} ({tier_name})"
        if status_notes:
            status = f"{status} | {' | '.join(status_notes)}"
        if verbose:
            print(f"[VPS_DELAYED] {status}")
        return True, innovation_mag, status, float(mahalanobis_sq)
    else:
        status = f"FAILED: Update computation failed ({tier_name})"
        if status_notes:
            status = f"{status} | {' | '.join(status_notes)}"
        return False, innovation_mag, status, float(mahalanobis_sq)
