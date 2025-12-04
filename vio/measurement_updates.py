"""
Measurement Updates for VIO+EKF System

This module contains measurement update functions for:
- Magnetometer heading updates (MAG)
- DEM height updates (altitude constraint)
- ZUPT (Zero Velocity Update)
- Generic EKF update helpers

These updates are applied asynchronously during the main VIO loop.

Author: VIO project
"""

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from typing import Optional, Tuple, Callable
import math

# Import shared math utilities (avoid duplication)
from .math_utils import quaternion_to_yaw, mahalanobis_squared
from .output_utils import log_measurement_update


def _mahalanobis2(y: np.ndarray, S: np.ndarray) -> float:
    """
    Compute squared Mahalanobis distance (wrapper to math_utils).
    
    Args:
        y: Innovation vector
        S: Innovation covariance matrix
        
    Returns:
        Squared Mahalanobis distance
    """
    return mahalanobis_squared(y, S)


def apply_zupt_update(kf, 
                      acc_deviation: float,
                      gyro_magnitude: float,
                      velocity_magnitude: float,
                      consecutive_stationary: int,
                      zupt_acc_threshold: float = 0.5,
                      zupt_gyro_threshold: float = 0.05,
                      save_debug: bool = False,
                      residual_csv: Optional[str] = None,
                      timestamp: float = 0.0,
                      frame: int = -1) -> Tuple[bool, int]:
    """
    Apply Zero Velocity Update when stationary is detected.
    
    ZUPT constrains velocity to zero when IMU indicates no motion,
    preventing drift accumulation during hover/stationary periods.
    
    Args:
        kf: ExtendedKalmanFilter instance
        acc_deviation: Deviation of accelerometer magnitude from gravity
        gyro_magnitude: Gyroscope measurement magnitude
        velocity_magnitude: Current estimated velocity magnitude
        consecutive_stationary: Count of consecutive stationary detections
        zupt_acc_threshold: Acceleration threshold for stationary detection
        zupt_gyro_threshold: Gyro threshold for stationary detection
        save_debug: Whether to log debug data
        residual_csv: Path to residual log file
        timestamp: Current timestamp
        frame: Current frame number
        
    Returns:
        Tuple of (zupt_applied: bool, updated_consecutive_count: int)
    """
    # Check if stationary
    is_stationary = (acc_deviation < zupt_acc_threshold) and (gyro_magnitude < zupt_gyro_threshold)
    
    if not is_stationary:
        return False, 0
    
    consecutive_stationary += 1
    
    # Build ZUPT measurement
    num_clones = (kf.x.shape[0] - 16) // 7
    err_dim = 15 + 6 * num_clones
    
    H_zupt = np.zeros((3, err_dim), dtype=float)
    H_zupt[0:3, 3:6] = np.eye(3)  # Measure velocity error (δv)
    
    z_zupt = np.zeros((3, 1), dtype=float)  # Target: zero velocity
    
    # Adaptive R based on confidence
    if velocity_magnitude < 1.0:
        base_r = 0.0001
    elif velocity_magnitude < 5.0:
        base_r = 0.001
    else:
        base_r = 0.01
    
    consecutive_factor = max(1.0, min(10.0, consecutive_stationary / 100.0))
    R_zupt = np.diag([base_r / consecutive_factor] * 3)
    
    # Reject if velocity too high (might cause instability)
    if velocity_magnitude > 500.0:
        return False, consecutive_stationary
    
    # Decouple yaw from velocity before ZUPT
    kf.P[3:6, 8] = 0.0
    kf.P[8, 3:6] = 0.0
    kf.P[3:6, 6:8] = 0.0
    kf.P[6:8, 3:6] = 0.0
    
    def h_zupt_jacobian(x, h=H_zupt):
        return h
    
    def hx_zupt_fun(x, h=H_zupt):
        return x[3:6].reshape(3, 1)
    
    # Apply update
    kf.update(
        z=z_zupt,
        HJacobian=h_zupt_jacobian,
        Hx=hx_zupt_fun,
        R=R_zupt
    )
    
    return True, consecutive_stationary


# Global state for magnetometer oscillation detection
_MAG_STATE = {
    'innovation_history': [],  # Recent innovation signs
    'oscillation_count': 0,    # Consecutive sign flips
    'skip_count': 0,           # How many updates to skip
    'last_yaw_mag': None,      # For smoothing
}


def apply_magnetometer_update(kf,
                              mag_calibrated: np.ndarray,
                              mag_declination: float,
                              use_raw_heading: bool,
                              sigma_mag_yaw: float,
                              time_elapsed: float,
                              gyro_z: float = 0.0,
                              in_convergence: bool = False,
                              has_ppk_yaw: bool = False,
                              save_debug: bool = False,
                              residual_csv: Optional[str] = None,
                              timestamp: float = 0.0,
                              frame: int = -1) -> Tuple[bool, str]:
    """
    Apply magnetometer heading update.
    
    Features:
    - GPS-calibrated raw heading or tilt-compensated heading
    - Adaptive Kalman gain with phase-based K_MIN
    - Yaw correction limiting to prevent attitude destabilization
    - Cross-covariance decoupling to prevent altitude drift
    - Oscillation detection to prevent yaw bouncing near ±180°
    
    Args:
        kf: ExtendedKalmanFilter instance
        mag_calibrated: Calibrated magnetometer vector [x, y, z]
        mag_declination: Magnetic declination in radians
        use_raw_heading: Use raw body-frame heading (GPS-calibrated)
        sigma_mag_yaw: Base measurement noise for yaw
        time_elapsed: Time since start for flight phase detection
        gyro_z: Current gyro Z for consistency check
        in_convergence: Whether in initial convergence period
        has_ppk_yaw: Whether PPK provided initial yaw (skip if in convergence)
        save_debug: Enable debug logging
        residual_csv: Path to residual log
        timestamp: Current timestamp
        frame: Current frame number
        
    Returns:
        Tuple of (update_applied: bool, rejection_reason: str)
    """
    global _MAG_STATE
    from .magnetometer import compute_yaw_from_mag
    
    # Skip during convergence if PPK provided initial yaw
    if in_convergence and has_ppk_yaw:
        return False, "PPK convergence period"
    
    # Check if we're in oscillation skip mode
    if _MAG_STATE['skip_count'] > 0:
        _MAG_STATE['skip_count'] -= 1
        return False, f"Skipping due to oscillation ({_MAG_STATE['skip_count']} remaining)"
    
    # Check field strength
    mag_norm = np.linalg.norm(mag_calibrated)
    
    # Compute yaw from magnetometer
    q_state = kf.x[6:10, 0]
    yaw_mag, quality = compute_yaw_from_mag(
        mag_calibrated, q_state,
        mag_declination=mag_declination,
        use_raw_heading=use_raw_heading
    )
    
    if quality < 0.3:
        return False, f"Low quality ({quality:.3f})"
    
    # Get current yaw from state
    yaw_state = quaternion_to_yaw(q_state)
    
    # Compute innovation with angle wrapping
    yaw_innov = yaw_mag - yaw_state
    yaw_innov = np.arctan2(np.sin(yaw_innov), np.cos(yaw_innov))
    
    # =========================================================================
    # OSCILLATION DETECTION: Prevent yaw bouncing near ±180° boundary
    # =========================================================================
    innov_sign = 1 if yaw_innov > 0 else -1
    _MAG_STATE['innovation_history'].append(innov_sign)
    
    # Keep only last 10 innovations
    if len(_MAG_STATE['innovation_history']) > 10:
        _MAG_STATE['innovation_history'] = _MAG_STATE['innovation_history'][-10:]
    
    # Detect oscillation: alternating signs with large innovations
    if len(_MAG_STATE['innovation_history']) >= 4 and abs(yaw_innov) > np.radians(90.0):
        recent = _MAG_STATE['innovation_history'][-4:]
        # Check for alternating pattern: [+, -, +, -] or [-, +, -, +]
        is_alternating = all(recent[i] != recent[i+1] for i in range(3))
        
        if is_alternating:
            _MAG_STATE['oscillation_count'] += 1
            if _MAG_STATE['oscillation_count'] >= 3:
                # Skip next 50 updates to let IMU stabilize
                _MAG_STATE['skip_count'] = 50
                _MAG_STATE['oscillation_count'] = 0
                _MAG_STATE['innovation_history'] = []
                print(f"[MAG-OSC] Detected oscillation, skipping 50 updates")
                return False, "Oscillation detected, skipping"
        else:
            _MAG_STATE['oscillation_count'] = 0
    else:
        _MAG_STATE['oscillation_count'] = 0
    
    # Innovation threshold (very permissive - mag is absolute reference)
    if in_convergence:
        innovation_threshold = np.radians(180.0)
    else:
        innovation_threshold = np.radians(179.0)
    
    if abs(yaw_innov) > innovation_threshold:
        return False, f"Innovation too large ({np.degrees(yaw_innov):.1f}°)"
    
    # Build measurement model
    num_clones = (kf.x.shape[0] - 16) // 7
    err_dim = 15 + 6 * num_clones
    theta_cov_idx = 8  # δθ_z in error state
    
    def h_mag_fun(x):
        h_yaw = np.zeros((1, err_dim), dtype=float)
        h_yaw[0, 8] = 1.0  # Yaw depends on δθ_z
        return h_yaw
    
    def hx_mag_fun(x):
        q_x = x[6:10, 0]
        yaw_x = quaternion_to_yaw(q_x)
        return np.array([[yaw_x]])
    
    # Measurement covariance
    r_yaw = np.array([[sigma_mag_yaw**2]])
    
    # Phase-based Kalman gain tuning
    if time_elapsed < 15.0:  # Spinup phase
        K_MIN = 0.40
        MAX_YAW_CORRECTION = np.radians(30.0)
    else:  # Normal flight
        K_MIN = 0.20
        MAX_YAW_CORRECTION = np.radians(15.0)
    
    # Compute current Kalman gain
    P_yaw = kf.P[theta_cov_idx, theta_cov_idx]
    S_yaw = P_yaw + sigma_mag_yaw**2
    K_yaw = P_yaw / S_yaw
    
    # Enforce minimum K
    if K_yaw < K_MIN and K_MIN > 0.01:
        P_yaw_min = K_MIN * sigma_mag_yaw**2 / (1.0 - K_MIN)
        if kf.P[theta_cov_idx, theta_cov_idx] < P_yaw_min:
            kf.P[theta_cov_idx, theta_cov_idx] = P_yaw_min
            P_yaw = P_yaw_min
            K_yaw = K_MIN
    
    # Limit maximum correction
    expected_correction = K_yaw * yaw_innov
    if abs(expected_correction) > MAX_YAW_CORRECTION:
        K_target = MAX_YAW_CORRECTION / abs(yaw_innov)
        R_inflated = P_yaw * (1.0 / K_target - 1.0)
        r_yaw = np.array([[max(R_inflated, sigma_mag_yaw**2)]])
    
    # Decouple yaw from roll/pitch and velocity
    kf.P[8, 6] = 0.0; kf.P[6, 8] = 0.0
    kf.P[8, 7] = 0.0; kf.P[7, 8] = 0.0
    kf.P[8, 5] = 0.0; kf.P[5, 8] = 0.0
    
    # Apply update with angle residual
    def angle_residual(a, b):
        res = a - b
        return np.arctan2(np.sin(res), np.cos(res))
    
    try:
        kf.update(
            z=np.array([[yaw_mag]]),
            HJacobian=h_mag_fun,
            Hx=hx_mag_fun,
            R=r_yaw,
            residual=angle_residual
        )
        kf.last_mag_time = timestamp
        
        # Log measurement update for debug_residuals.csv
        if residual_csv:
            innovation = np.array([yaw_innov])
            s_mat = np.array([[S_yaw]])
            m2 = mahalanobis_squared(innovation, s_mat)
            log_measurement_update(
                residual_csv, timestamp, frame, 'MAG',
                innovation=innovation,
                mahalanobis_dist=np.sqrt(m2),
                chi2_threshold=np.inf,  # MAG always accepted if it passes above checks
                accepted=True,
                s_matrix=s_mat,
                p_prior=getattr(kf, 'P_prior', kf.P)
            )
        
        return True, ""
    except Exception as e:
        return False, f"Update failed: {e}"


def apply_dem_height_update(kf,
                            height_measurement: float,
                            sigma_height: float,
                            xy_uncertainty: float,
                            time_since_correction: float,
                            speed_ms: float,
                            has_valid_dem: bool,
                            dem_slope: float = 0.0,
                            no_vision_corrections: bool = False,
                            save_debug: bool = False,
                            residual_csv: Optional[str] = None,
                            timestamp: float = 0.0,
                            frame: int = -1) -> Tuple[bool, str]:
    """
    Apply DEM-based height update.
    
    Uses adaptive uncertainty scaling based on:
    - XY position uncertainty (affects DEM lookup accuracy)
    - Time since last absolute correction
    - Horizontal speed
    - DEM terrain slope
    
    Args:
        kf: ExtendedKalmanFilter instance
        height_measurement: Height measurement (MSL or AGL)
        sigma_height: Base height uncertainty
        xy_uncertainty: XY position uncertainty (sum of variances)
        time_since_correction: Time since last VPS/GNSS correction
        speed_ms: Current horizontal speed
        has_valid_dem: Whether DEM is available
        dem_slope: Local terrain slope magnitude
        no_vision_corrections: True if no VIO/VPS available (IMU-only mode)
        save_debug: Enable debug logging
        residual_csv: Path to residual log
        timestamp: Current timestamp
        frame: Current frame number
        
    Returns:
        Tuple of (update_applied: bool, rejection_reason: str)
    """
    if np.isnan(height_measurement):
        return False, "Invalid height measurement"
    
    num_clones = (kf.x.shape[0] - 16) // 7
    err_dim = 15 + 6 * num_clones
    
    H_height = np.zeros((1, err_dim), dtype=float)
    H_height[0, 2] = 1.0  # Height is δp_z
    
    def h_fun(x, h=H_height):
        return h
    
    def hx_fun(x, h=H_height):
        return x[2:3].reshape(1, 1)
    
    # Adaptive uncertainty scaling
    height_cov_scale = 1.0
    
    # Scale with XY uncertainty
    xy_std = np.sqrt(xy_uncertainty / 2.0)
    if xy_std > 1.0:
        height_cov_scale *= (1.0 + xy_std * 0.25)
    
    # Scale with time since correction
    if time_since_correction > 10.0:
        height_cov_scale *= (1.0 + (time_since_correction - 10.0) / 40.0)
    
    # Scale with speed
    if speed_ms > 10:
        height_cov_scale *= (1.0 + (speed_ms - 10) / 15.0)
    
    # Scale for MSL-only (no DEM)
    if not has_valid_dem:
        height_cov_scale *= 5.0
    
    r_mat = np.array([[sigma_height**2 * height_cov_scale]])
    
    # Innovation gating
    S_mat = H_height @ kf.P @ H_height.T + r_mat
    predicted_height = kf.x[2, 0]
    innovation = np.array([[height_measurement - predicted_height]])
    
    try:
        m2_test = _mahalanobis2(innovation, S_mat)
    except:
        m2_test = float('inf')
    
    # Adaptive threshold - IMPORTANT: We use higher threshold for altitude
    # since nadir camera MSCKF cannot properly constrain vertical direction
    # and we need DEM updates to bound altitude drift
    if no_vision_corrections:
        threshold = 100.0
    elif xy_std > 10.0:
        threshold = 50.0  # Increased from 15.0 - trust DEM more
    elif xy_std > 5.0:
        threshold = 25.0  # Increased from 9.21
    else:
        threshold = 15.0  # Increased from 6.63 - chi2(1, 0.999) ≈ 10.83
    
    # ALTITUDE FIX: For large innovations, apply update with increased noise
    # This prevents runaway drift while still allowing large corrections
    if m2_test >= threshold:
        # Apply update anyway but with inflated noise (soft gating)
        inflation_factor = m2_test / threshold
        r_mat_inflated = r_mat * inflation_factor
        
        kf.update(
            z=np.array([[height_measurement]]),
            HJacobian=h_fun,
            Hx=hx_fun,
            R=r_mat_inflated
        )
        
        if residual_csv:
            log_measurement_update(
                residual_csv, timestamp, frame, 'DEM',
                innovation=innovation.flatten(),
                mahalanobis_dist=np.sqrt(m2_test),
                chi2_threshold=threshold,
                accepted=True,  # Still accepted but with inflated noise
                s_matrix=S_mat
            )
        return True, f"Soft gating (m2={m2_test:.2f}, inflated R by {inflation_factor:.1f}x)"
    
    # Normal update
    kf.update(
        z=np.array([[height_measurement]]),
        HJacobian=h_fun,
        Hx=hx_fun,
        R=r_mat
    )
    
    # Log accepted update
    if residual_csv:
        log_measurement_update(
            residual_csv, timestamp, frame, 'DEM',
            innovation=innovation.flatten(),
            mahalanobis_dist=np.sqrt(m2_test),
            chi2_threshold=threshold,
            accepted=True,
            s_matrix=S_mat,
            p_prior=getattr(kf, 'P_prior', kf.P)
        )
    
    return True, ""


def apply_velocity_update(kf,
                          velocity_measurement: np.ndarray,
                          sigma_velocity: float,
                          use_vz_only: bool,
                          alignment_deg: float,
                          avg_flow_px: float,
                          sigma_scale_xy: float = 1.0,
                          sigma_scale_z: float = 1.0,
                          chi2_threshold_1d: float = 25.0,
                          chi2_threshold_3d: float = 60.0,
                          save_debug: bool = False,
                          residual_csv: Optional[str] = None,
                          timestamp: float = 0.0,
                          frame: int = -1) -> Tuple[bool, str]:
    """
    Apply VIO velocity update.
    
    Supports either full 3D velocity update or vertical-only (Vz) for nadir cameras.
    Uses adaptive uncertainty based on alignment and optical flow quality.
    
    Args:
        kf: ExtendedKalmanFilter instance
        velocity_measurement: Velocity [vx, vy, vz] in world frame
        sigma_velocity: Base velocity uncertainty
        use_vz_only: Only update vertical velocity (for nadir cameras)
        alignment_deg: Motion alignment angle in degrees
        avg_flow_px: Average optical flow magnitude in pixels
        sigma_scale_xy: XY uncertainty scale factor
        sigma_scale_z: Z uncertainty scale factor
        chi2_threshold_1d: Chi-square threshold for 1D update
        chi2_threshold_3d: Chi-square threshold for 3D update
        save_debug: Enable debug logging
        residual_csv: Path to residual log
        timestamp: Current timestamp
        frame: Current frame number
        
    Returns:
        Tuple of (update_applied: bool, rejection_reason: str)
    """
    num_clones = (kf.x.shape[0] - 16) // 7
    err_dim = 15 + 6 * num_clones
    
    # Adaptive uncertainty
    align_scale = 1.0 + alignment_deg / 45.0
    flow_scale = 1.0 + max(0.0, (avg_flow_px - 10.0) / 20.0)
    uncertainty_scale = align_scale * flow_scale
    
    if use_vz_only:
        H_vel = np.zeros((1, err_dim), dtype=float)
        H_vel[0, 5] = 1.0  # δv_z
        vel_meas = np.array([[velocity_measurement[2]]])
        r_mat = np.array([[(sigma_velocity * sigma_scale_z * uncertainty_scale)**2]])
        chi2_threshold = chi2_threshold_1d
    else:
        H_vel = np.zeros((3, err_dim), dtype=float)
        H_vel[0, 3] = 1.0  # δv_x
        H_vel[1, 4] = 1.0  # δv_y
        H_vel[2, 5] = 1.0  # δv_z
        vel_meas = velocity_measurement.reshape(-1, 1)
        r_mat = np.diag([
            (sigma_velocity * sigma_scale_xy * uncertainty_scale)**2,
            (sigma_velocity * sigma_scale_xy * uncertainty_scale)**2,
            (sigma_velocity * sigma_scale_z * uncertainty_scale)**2
        ])
        chi2_threshold = chi2_threshold_3d
    
    def h_fun(x, h=H_vel):
        return h
    
    def hx_fun(x, h=H_vel):
        if use_vz_only:
            return x[5:6].reshape(1, 1)
        else:
            return x[3:6].reshape(3, 1)
    
    # Innovation gating
    S_mat = H_vel @ kf.P @ H_vel.T + r_mat
    if use_vz_only:
        predicted_vel = kf.x[5:6, 0].reshape(1, 1)
    else:
        predicted_vel = kf.x[3:6, 0].reshape(3, 1)
    
    innovation = vel_meas - predicted_vel
    
    try:
        m2_test = _mahalanobis2(innovation, S_mat)
    except:
        m2_test = float('inf')
    
    if m2_test >= chi2_threshold:
        # Log rejected update
        if residual_csv:
            log_measurement_update(
                residual_csv, timestamp, frame, 'VIO_VEL',
                innovation=innovation.flatten(),
                mahalanobis_dist=np.sqrt(m2_test),
                chi2_threshold=chi2_threshold,
                accepted=False,
                s_matrix=S_mat
            )
        return False, f"Chi-square test failed ({m2_test:.2f} >= {chi2_threshold:.1f})"
    
    # Apply update
    kf.update(
        z=vel_meas,
        HJacobian=h_fun,
        Hx=hx_fun,
        R=r_mat
    )
    
    # Log accepted update
    if residual_csv:
        log_measurement_update(
            residual_csv, timestamp, frame, 'VIO_VEL',
            innovation=innovation.flatten(),
            mahalanobis_dist=np.sqrt(m2_test),
            chi2_threshold=chi2_threshold,
            accepted=True,
            s_matrix=S_mat,
            p_prior=getattr(kf, 'P_prior', kf.P)
        )
    
    return True, ""


def apply_plane_constraint(kf,
                           altitude_agl: float,
                           dem_elevation: float,
                           base_sigma: float = 1.0,
                           num_inliers: int = 30) -> Tuple[bool, str]:
    """
    Apply altitude plane constraint for nadir cameras.
    
    Uses homography-derived altitude estimate to constrain vertical position.
    
    Args:
        kf: ExtendedKalmanFilter instance
        altitude_agl: Estimated AGL from homography
        dem_elevation: DEM elevation at current position
        base_sigma: Base altitude uncertainty
        num_inliers: Number of homography inliers (affects uncertainty)
        
    Returns:
        Tuple of (update_applied: bool, rejection_reason: str)
    """
    num_clones = (kf.x.shape[0] - 16) // 7
    err_dim = 15 + 6 * num_clones
    
    H_plane = np.zeros((1, err_dim), dtype=float)
    H_plane[0, 2] = 1.0  # δp_z
    
    z_msl = altitude_agl + dem_elevation
    z_altitude = np.array([[z_msl]])
    
    # Uncertainty scales with inlier count
    if num_inliers < 30:
        sigma_alt = base_sigma * (30.0 / max(15, num_inliers))
    else:
        sigma_alt = base_sigma
    
    R_altitude = np.array([[sigma_alt**2]])
    
    def h_plane_fun(x, h=H_plane):
        return h
    
    def hx_plane_fun(x):
        return np.array([[x[2, 0]]])
    
    # Innovation gating
    predicted_alt = kf.x[2, 0]
    innovation = z_altitude - np.array([[predicted_alt]])
    S_alt = H_plane @ kf.P @ H_plane.T + R_altitude
    
    try:
        chi2_alt = float(innovation.T @ np.linalg.inv(S_alt) @ innovation)
    except:
        return False, "Covariance singular"
    
    # Adaptive threshold
    P_z = kf.P[2, 2]
    if P_z > 100:
        chi2_threshold = min(50.0, 6.63 * (1 + np.sqrt(P_z / 100)))
    else:
        chi2_threshold = 6.63
    
    if chi2_alt >= chi2_threshold:
        return False, f"Chi-square test failed ({chi2_alt:.2f} >= {chi2_threshold:.1f})"
    
    kf.update(
        z=z_altitude,
        HJacobian=h_plane_fun,
        Hx=hx_plane_fun,
        R=R_altitude
    )
    
    return True, ""
