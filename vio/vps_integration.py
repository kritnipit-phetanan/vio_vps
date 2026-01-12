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
from .data_loaders import VPSItem, DEMReader, ProjectionCache


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
        base_threshold_m = 200.0
        max_drift_rate = 200.0  # m/s
        r_scale = min(50.0, 10.0 + time_since_correction / 10.0)
        tier_name = "FIRST VPS"
    elif time_since_correction > 10.0:
        # TIER 2: Long drift
        base_threshold_m = 100.0
        max_drift_rate = 100.0
        r_scale = min(10.0, 1.0 + time_since_correction / 5.0)
        tier_name = "LONG DRIFT"
    elif time_since_correction > 3.0:
        # TIER 3a: Medium drift
        base_threshold_m = 50.0
        max_drift_rate = 60.0
        r_scale = min(5.0, 1.0 + time_since_correction / 3.0)
        tier_name = "MEDIUM DRIFT"
    else:
        # TIER 3b: Recent VPS
        base_threshold_m = 50.0
        max_drift_rate = 50.0
        r_scale = 1.0
        tier_name = "RECENT"
    
    max_innovation_m = base_threshold_m + max_drift_rate * time_since_correction
    max_innovation_m = min(max_innovation_m, 30000.0)  # Cap at 30km
    
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
            R=r_mat
        )
        return True
    except Exception as e:
        print(f"[VPS] Update failed: {e}")
        return False


# =============================================================================
# DEM (Digital Elevation Model) Altitude Updates
# =============================================================================

def get_dem_height(dem: DEMReader, lat: float, lon: float,
                   last_valid_dem: Optional[float] = None) -> Tuple[Optional[float], bool]:
    """
    Get DEM height at given location with fallback.
    
    Args:
        dem: DEM reader object
        lat: Latitude
        lon: Longitude
        last_valid_dem: Last valid DEM value for fallback
    
    Returns:
        dem_height: DEM height (or fallback value)
        has_valid_dem: True if DEM lookup succeeded
    """
    if dem.ds is None:
        return 0.0, False
    
    dem_now = dem.sample_m(lat, lon)
    
    if dem_now is not None and not np.isnan(dem_now):
        return dem_now, True
    elif last_valid_dem is not None:
        return last_valid_dem, True
    else:
        return 0.0, False


def compute_dem_measurement(kf: ExtendedKalmanFilter, dem_now: float,
                            has_valid_dem: bool, z_state: str,
                            msl_measured: Optional[float] = None,
                            msl0_m: float = 0.0) -> Tuple[float, str]:
    """
    Compute height measurement for DEM update.
    
    Adaptive strategy based on:
    - DEM availability
    - Current AGL
    - MSL measurements from flight log
    
    Args:
        kf: Extended Kalman Filter
        dem_now: DEM height at current position
        has_valid_dem: True if DEM is available
        z_state: State Z mode ('msl' or 'agl')
        msl_measured: MSL from flight log (if available)
        msl0_m: Initial MSL altitude
    
    Returns:
        height_m: Height measurement for update
        update_mode: Description of update strategy
    """
    if has_valid_dem:
        if z_state.lower() == "agl":
            agl_now = kf.x[2, 0]
            height_m = agl_now
            update_mode = "AGL"
        else:
            msl_now = kf.x[2, 0]
            agl_now = msl_now - dem_now
            
            if msl_measured is not None:
                height_m = msl_measured
                expected_agl = msl_measured - dem_now
                update_mode = f"MSL (flight_log, AGL={expected_agl:.1f}m)"
            else:
                # Fallback without flight_log
                min_safe_agl = 0.5
                
                if agl_now < min_safe_agl:
                    target_agl = 2.0
                    height_m = dem_now + target_agl
                    update_mode = f"MSL (emergency lift to {target_agl}m AGL)"
                elif agl_now < 30.0:
                    target_agl = max(agl_now, 5.0)
                    height_m = dem_now + target_agl
                    update_mode = f"MSL (low alt, AGL={target_agl:.1f}m)"
                elif agl_now < 150.0:
                    height_m = dem_now + agl_now
                    update_mode = f"MSL (AGL={agl_now:.1f}m)"
                else:
                    fallback_agl = 100.0
                    height_m = dem_now + fallback_agl
                    update_mode = f"MSL (high alt, fallback AGL={fallback_agl:.0f}m)"
    else:
        # No DEM
        height_m = msl_measured if msl_measured is not None else msl0_m
        source = 'interpolated' if msl_measured is not None else 'initial'
        update_mode = f"MSL (no DEM, {source})"
    
    return height_m, update_mode


def compute_height_noise_scale(kf: ExtendedKalmanFilter, t: float,
                               has_valid_dem: bool,
                               speed_ms: float = 0.0,
                               last_height: Optional[float] = None,
                               dt: float = 0.0) -> float:
    """
    Compute adaptive height measurement noise scaling.
    
    Scales based on:
    - Horizontal position uncertainty
    - Time since last correction
    - Speed
    - Vertical dynamics
    - DEM availability
    
    Args:
        kf: Extended Kalman Filter
        t: Current timestamp
        has_valid_dem: True if DEM available
        speed_ms: Current speed (m/s)
        last_height: Previous height measurement
        dt: Time since last height measurement
    
    Returns:
        height_cov_scale: Noise scaling factor
    """
    height_cov_scale = 1.0
    
    # 1. XY uncertainty
    xy_uncertainty = float(np.trace(kf.P[0:2, 0:2]))
    xy_std = np.sqrt(xy_uncertainty / 2.0)
    
    if xy_std > 1.0:
        xy_scale_factor = 1.0 + (xy_std * 0.25)
        height_cov_scale *= xy_scale_factor
    
    # 2. Time since correction
    if hasattr(kf, 'last_absolute_correction_time'):
        time_since_correction = t - kf.last_absolute_correction_time
        if time_since_correction > 10.0:
            time_scale_factor = 1.0 + (time_since_correction - 10.0) / 40.0
            height_cov_scale *= time_scale_factor
    
    # 3. Speed
    if speed_ms > 10:
        speed_scale_factor = 1.0 + (speed_ms - 10) / 15.0
        height_cov_scale *= speed_scale_factor
    
    # 4. Vertical dynamics
    if last_height is not None and dt > 0:
        current_height = kf.x[2, 0]
        height_rate = abs(current_height - last_height) / dt
        if height_rate > 2.0:
            height_cov_scale *= (1.0 + height_rate / 4.0)
    
    # 5. No DEM penalty
    if not has_valid_dem:
        height_cov_scale *= 5.0
    
    return height_cov_scale


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
            R=r_mat
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
