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

from .ekf import ExtendedKalmanFilter
from .data_loaders import VPSItem, ProjectionCache


def compute_vps_innovation(vps: VPSItem, kf: ExtendedKalmanFilter,
                           lat0: float, lon0: float,
                           proj_cache: ProjectionCache) -> Tuple[np.ndarray, np.ndarray, float]:
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
    sigma_vps = 1.0  # Base sigma
    scale = 1.0 + max(0.0, (speed_xy - 10.0) / 10.0) if speed_xy > 10 else 1.0
    r_mat = np.diag([(sigma_vps**2) * scale, (sigma_vps**2) * scale])
    
    # Innovation
    s_mat = h_xy @ kf.P @ h_xy.T + r_mat
    xy_pred = kf.x[0:2].reshape(2,)
    innovation = (vps_xy - xy_pred).reshape(-1, 1)
    
    try:
        from .math_utils import safe_matrix_inverse
        s_inv = safe_matrix_inverse(s_mat, damping=1e-9, method='cholesky')
        m2_test = float(innovation.T @ s_inv @ innovation)
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
    
    r_mat = np.diag([(sigma_vps * r_scale)**2, (sigma_vps * r_scale)**2])
    
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
    verbose: bool = False
) -> Tuple[bool, Optional[float], str]:
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
        return False, None, f"REJECTED: Clone '{image_id}' not found (expired?)"

    try:
        r_check = np.array(R_vps, dtype=float)
        if r_check.shape != (2, 2) or not np.all(np.isfinite(r_check)):
            return False, None, "REJECTED: Invalid R_vps"
        if float(np.min(np.diag(r_check))) <= 0.0:
            return False, None, "REJECTED: Non-positive R_vps diag"
    except Exception:
        return False, None, "REJECTED: Bad R_vps"
    
    # Get clone age for logging
    clone_age = clone_manager.get_clone_age(image_id, t_now=0.0)  # Relative age
    
    # Compute expected position for gating check
    vps_xy = proj_cache.latlon_to_xy(vps_lat, vps_lon, lat0, lon0)
    current_xy = kf.x[0:2, 0]
    pre_innovation = np.linalg.norm(vps_xy - current_xy)
    if not np.isfinite(pre_innovation):
        clone_manager.clones.pop(image_id, None)
        return False, None, "REJECTED: Non-finite innovation"
    
    # Adaptive gating based on time since last VPS
    max_innovation_m, r_scale, tier_name = compute_vps_acceptance_threshold(
        time_since_last_vps, pre_innovation
    )
    
    # Gate check
    if pre_innovation > max_innovation_m:
        clone_manager.clones.pop(image_id, None)  # Remove the useless clone
        return False, pre_innovation, f"GATED: Innovation {pre_innovation:.1f}m > {max_innovation_m:.1f}m ({tier_name})"
    
    # Scale R_vps if needed
    R_vps_scaled = R_vps * r_scale
    
    # Apply the delayed update
    success, innovation_mag = clone_manager.apply_delayed_update(
        kf=kf,
        image_id=image_id,
        vps_lat=vps_lat,
        vps_lon=vps_lon,
        R_vps=R_vps_scaled,
        proj_cache=proj_cache,
        lat0=lat0,
        lon0=lon0
    )
    
    if success:
        status = f"APPLIED: Innovation {innovation_mag:.2f}m, R_scale={r_scale:.1f} ({tier_name})"
        if verbose:
            print(f"[VPS_DELAYED] {status}")
        return True, innovation_mag, status
    else:
        return False, innovation_mag, f"FAILED: Update computation failed ({tier_name})"
