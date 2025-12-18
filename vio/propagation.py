#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Propagation Module

Handles IMU state propagation with both preintegration and legacy modes.
Includes helpers for time synchronization and ZUPT (Zero Velocity Update).

Author: VIO project
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.spatial.transform import Rotation as R_scipy

from .math_utils import skew_symmetric
from .imu_preintegration import (
    IMUPreintegration, 
    compute_error_state_jacobian, 
    compute_error_state_process_noise
)
from .ekf import ExtendedKalmanFilter, ensure_covariance_valid, propagate_error_state_covariance
from .data_loaders import IMURecord


def propagate_nominal_state(kf: ExtendedKalmanFilter, 
                            rec: IMURecord, dt: float,
                            bg: np.ndarray, ba: np.ndarray,
                            imu_params: dict):
    """
    [DEPRECATED v3.5.1] Lightweight nominal state propagation - DO NOT USE!
    
    This function was based on a MISUNDERSTANDING of OpenVINS architecture.
    
    WRONG APPROACH (v3.5.0):
    - Propagate only state (x) at 400 Hz
    - Propagate covariance (P) at 20 Hz
    - Result: P is stale when VPS/MAG arrive → inconsistent Kalman gain!
    
    CORRECT APPROACH (v3.5.1+):
    - Propagate BOTH x and P at every IMU tick (400 Hz)
    - Use preintegration ONLY for MSCKF/camera constraints
    - Never skip covariance propagation between measurements!
    
    OpenVINS also propagates P at every IMU tick. The "fast" part is using
    preintegration for MSCKF instead of per-clone Jacobians.
    
    Use process_imu() instead - it correctly propagates both x and P.
    
    Args:
        kf: Kalman filter
        rec: IMU record (ang, lin)
        dt: Time step
        bg: Gyro bias
        ba: Accel bias
        imu_params: IMU parameters (g_norm)
    """
    # Bias-corrected measurements
    w_corr = rec.ang - bg
    a_corr = rec.lin - ba
    
    # Current state
    p = kf.x[0:3, 0]
    v = kf.x[3:6, 0]
    q = kf.x[6:10, 0]  # [w,x,y,z]
    
    # Rotation update (discrete integration)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    R_BW = R_scipy.from_quat(q_xyzw).as_matrix()
    
    # Discrete rotation: exp(w * dt)
    theta = np.linalg.norm(w_corr) * dt
    if theta < 1e-8:
        # Small angle approximation: exp([w×]dt) ≈ I + [w×]dt
        delta_R = np.eye(3) + skew_symmetric(w_corr * dt)
    else:
        # Rodrigues formula
        axis = w_corr / np.linalg.norm(w_corr)
        delta_R = R_scipy.from_rotvec(axis * theta).as_matrix()
    
    R_BW_new = R_BW @ delta_R
    q_new_xyzw = R_scipy.from_matrix(R_BW_new).as_quat()
    q_new = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]])
    
    # Velocity update (world frame acceleration)
    g_world = np.array([0, 0, imu_params.get('g_norm', 9.803)])
    a_world = R_BW @ a_corr - g_world
    v_new = v + a_world * dt
    
    # Position update (trapezoidal integration for better accuracy)
    p_new = p + (v + v_new) / 2.0 * dt
    
    # Write back to state (only p, v, q - NOT bias or clones)
    kf.x[0:3, 0] = p_new
    kf.x[3:6, 0] = v_new
    kf.x[6:10, 0] = q_new


def propagate_to_timestamp(kf: ExtendedKalmanFilter, target_time: float, 
                           imu_buffer: List[IMURecord], current_time: float,
                           imu_params: dict,
                           estimate_imu_bias: bool = False,
                           use_preintegration: bool = True) -> Tuple[bool, Optional[IMUPreintegration]]:
    """
    Propagate state to exact target timestamp using IMU measurements.
    
    Uses IMU preintegration (Forster et al.) for better accuracy and efficiency.
    
    Two modes:
      1. Preintegration (default): Integrate all IMU → apply once → less noise
      2. Legacy: Propagate sample-by-sample (fallback)
    
    This function handles time synchronization between sensors by:
    1. Finding IMU measurements between current_time and target_time
    2. Preintegrating all IMU measurements into ΔR, Δv, Δp
    3. Applying preintegrated delta to state (single update)
    4. Propagating covariance using preintegration Jacobians
    
    Critical for:
    - Camera-IMU synchronization (camera @ 20Hz, IMU @ 400Hz)
    - VPS measurements with latency
    - Event-driven sensor fusion
    
    Args:
        kf: Extended Kalman Filter
        target_time: Desired timestamp to propagate to
        imu_buffer: List of IMU measurements (must be sorted by time)
        current_time: Current state time
        imu_params: IMU noise parameters dict
        estimate_imu_bias: Whether IMU bias was pre-estimated
        use_preintegration: Use Forster-style preintegration (recommended)
    
    Returns:
        success: True if propagation succeeded
        preint_data: IMUPreintegration object (None if not using preintegration)
    """
    if target_time <= current_time:
        return True, None
    
    if len(imu_buffer) == 0:
        print(f"[WARNING] propagate_to_timestamp: No IMU data available")
        return False, None
    
    # Find IMU samples in the time range (current_time, target_time]
    relevant_imu = [imu for imu in imu_buffer if current_time < imu.t <= target_time]
    
    print(f"[PROPAGATE] t={current_time:.3f}→{target_time:.3f}, dt={target_time-current_time:.3f}s, found {len(relevant_imu)} IMU samples")
    
    if len(relevant_imu) == 0:
        # No IMU data in range - need to extrapolate (not recommended)
        print(f"[WARNING] propagate_to_timestamp: No IMU in range ({current_time:.3f}, {target_time:.3f}], extrapolating")
        
        # Find closest IMU before target
        past_imu = [imu for imu in imu_buffer if imu.t <= target_time]
        if len(past_imu) == 0:
            print(f"[ERROR] propagate_to_timestamp: No IMU data before target time")
            return False, None
        
        last_imu = past_imu[-1]
        dt = target_time - current_time
        
        # Propagate with constant IMU (extrapolation) - legacy mode only
        _propagate_single_imu_step(kf, last_imu, dt, estimate_imu_bias, current_time, imu_params)
        return True, None
    
    # ============================================================
    # MODE 1: IMU Preintegration (Forster et al. TRO 2017)
    # ============================================================
    if use_preintegration:
        # Extract current state
        p = kf.x[0:3, 0]
        v = kf.x[3:6, 0]
        q = kf.x[6:10, 0]  # [w,x,y,z]
        bg = kf.x[10:13, 0]
        ba = kf.x[13:16, 0]
        
        # Initialize preintegration with current bias estimates
        preint = IMUPreintegration(
            bg=bg, ba=ba,
            sigma_g=imu_params['gyr_n'],
            sigma_a=imu_params['acc_n'],
            sigma_bg=imu_params['gyr_w'],
            sigma_ba=imu_params['acc_w']
        )
        
        # Integrate all IMU measurements in the interval
        t_prev = current_time
        for imu in relevant_imu:
            dt = imu.t - t_prev
            if dt > 0:
                preint.integrate_measurement(imu.w, imu.a, dt)
                t_prev = imu.t
        
        # Handle fractional timestep if target_time doesn't align with last IMU
        if t_prev < target_time:
            last_imu = relevant_imu[-1]
            dt_frac = target_time - t_prev
            preint.integrate_measurement(last_imu.w, last_imu.a, dt_frac)
        
        # Get bias-corrected preintegrated deltas
        delta_R, delta_v, delta_p = preint.get_deltas_corrected(bg, ba)
        
        # Apply preintegrated delta to nominal state
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
        R_old = R_scipy.from_quat(q_xyzw).as_matrix()
        
        # R_WB_new = R_delta^T * R_WB_old
        R_new = delta_R.T @ R_old
        q_new_xyzw = R_scipy.from_matrix(R_new).as_quat()
        q_new = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]])  # [w,x,y,z]
        
        # R_BW = R_WB.T (Body-to-World) for projecting to world frame
        R_BW = R_old.T
        
        # Velocity: v_new = v_old + R_BW * delta_v
        v_new = v + R_BW @ delta_v
        
        # Position: p_new = p_old + v_old * dt + R_BW * delta_p
        dt_total = target_time - current_time
        p_new = p + v * dt_total + R_BW @ delta_p
        
        # Update nominal state
        kf.x[0:3, 0] = p_new
        kf.x[3:6, 0] = v_new
        kf.x[6:10, 0] = q_new
        
        # Propagate error-state covariance using preintegration Jacobians
        preint_cov = preint.get_covariance()
        J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba = preint.get_jacobians()
        
        # Build state transition matrix for error-state
        num_clones = (kf.x.shape[0] - 16) // 7
        n_err = 15 + num_clones * 6
        
        Phi = np.eye(n_err, dtype=float)
        
        # Position error propagation
        Phi[0:3, 3:6] = np.eye(3) * dt_total
        Phi[0:3, 6:9] = -R_BW @ skew_symmetric(delta_p)
        Phi[0:3, 9:12] = R_BW @ J_p_bg
        Phi[0:3, 12:15] = R_BW @ J_p_ba
        
        # Velocity error propagation
        Phi[3:6, 6:9] = -R_BW @ skew_symmetric(delta_v)
        Phi[3:6, 9:12] = R_BW @ J_v_bg
        Phi[3:6, 12:15] = R_BW @ J_v_ba
        
        # Rotation error propagation
        Phi[6:9, 9:12] = J_R_bg
        
        # Process noise
        Q = np.zeros((n_err, n_err), dtype=float)
        
        # Map preintegration covariance to error-state
        Q[0:3, 0:3] = R_BW @ preint_cov[6:9, 6:9] @ R_BW.T
        Q[0:3, 3:6] = R_BW @ preint_cov[6:9, 3:6] @ R_BW.T
        Q[0:3, 6:9] = R_BW @ preint_cov[6:9, 0:3] @ R_BW.T
        
        Q[3:6, 0:3] = R_BW @ preint_cov[3:6, 6:9] @ R_BW.T
        Q[3:6, 3:6] = R_BW @ preint_cov[3:6, 3:6] @ R_BW.T
        Q[3:6, 6:9] = R_BW @ preint_cov[3:6, 0:3] @ R_BW.T
        
        Q[6:9, 0:3] = R_BW @ preint_cov[0:3, 6:9] @ R_BW.T
        Q[6:9, 3:6] = R_BW @ preint_cov[0:3, 3:6] @ R_BW.T
        Q[6:9, 6:9] = R_BW @ preint_cov[0:3, 0:3] @ R_BW.T
        
        # Add bias random walk noise
        Q[9:12, 9:12] = np.eye(3) * (imu_params['gyr_w'] ** 2) * dt_total
        Q[12:15, 12:15] = np.eye(3) * (imu_params['acc_w'] ** 2) * dt_total
        
        # Propagate covariance
        kf.P = Phi @ kf.P @ Phi.T + Q
        
        # Ensure covariance validity
        kf.P = ensure_covariance_valid(
            kf.P, 
            label="Preintegration-Propagate",
            symmetrize=True,
            check_psd=True,
            min_eigenvalue=1e-9,
            log_condition=False
        )
        
        # Update priors
        kf.x_prior = kf.x.copy()
        kf.P_prior = kf.P.copy()
        
        return True, preint
    
    # ============================================================
    # MODE 2: Legacy Sample-by-Sample Propagation (Fallback)
    # ============================================================
    else:
        t_current = current_time
        
        for i, imu in enumerate(relevant_imu):
            if i == 0:
                dt = imu.t - t_current
            else:
                dt = imu.t - relevant_imu[i-1].t
            
            if dt > 0:
                _propagate_single_imu_step(kf, imu, dt, estimate_imu_bias, t_current, imu_params)
                t_current = imu.t
        
        # Check if we need to interpolate to exact target_time
        if t_current < target_time:
            last_imu = relevant_imu[-1]
            dt = target_time - t_current
            _propagate_single_imu_step(kf, last_imu, dt, estimate_imu_bias, t_current, imu_params)
        
        # Ensure covariance validity
        kf.P = ensure_covariance_valid(
            kf.P,
            label="Legacy-Propagate",
            symmetrize=True,
            check_psd=True,
            min_eigenvalue=1e-9
        )
        
        return True, None


def process_imu(kf: ExtendedKalmanFilter, imu: IMURecord, dt: float,
                estimate_imu_bias: bool = False, t: float = 0.0, 
                t0: float = 0.0, imu_params: dict = None):
    """
    Public wrapper for single IMU step propagation.
    
    This is the main entry point for legacy (non-preintegration) IMU propagation.
    
    Args:
        kf: Extended Kalman Filter
        imu: IMU measurement record
        dt: Time step since last sample
        estimate_imu_bias: Whether bias was pre-estimated
        t: Current timestamp
        t0: Initial timestamp (for process noise time-varying)
        imu_params: IMU noise parameters dict
    """
    if imu_params is None:
        imu_params = {'g_norm': 9.80665}
    
    if dt <= 0:
        return
    
    _propagate_single_imu_step(kf, imu, dt, estimate_imu_bias, t, imu_params)


def _propagate_single_imu_step(kf: ExtendedKalmanFilter, imu: IMURecord, dt: float,
                                estimate_imu_bias: bool, t: float, imu_params: dict):
    """
    Propagate state by one IMU step (helper for propagate_to_timestamp).
    
    Args:
        kf: Extended Kalman Filter
        imu: IMU measurement
        dt: Time step
        estimate_imu_bias: Whether bias was pre-estimated
        t: Current time (for process noise computation)
        imu_params: IMU noise parameters dict
    """
    # Extract state
    p = kf.x[0:3, 0]
    v = kf.x[3:6, 0]
    q = kf.x[6:10, 0]  # [w,x,y,z]
    bg = kf.x[10:13, 0]
    ba = kf.x[13:16, 0]
    
    # Rotation matrix (body to world)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    R_body_to_world = R_scipy.from_quat(q_xyzw).as_matrix()
    
    # Bias-corrected measurements
    w_corr = imu.w - bg
    a_corr = imu.a - ba
    
    # Rotate acceleration to world frame
    a_world = R_body_to_world @ a_corr
    
    # Nominal state propagation
    p_new = p + v * dt + 0.5 * a_world * dt**2
    v_new = v + a_world * dt
    
    # Quaternion propagation
    theta_vec = w_corr * dt
    theta = np.linalg.norm(theta_vec)
    
    if theta < 1e-6:
        dq = np.array([1.0, 0.5*theta_vec[0], 0.5*theta_vec[1], 0.5*theta_vec[2]], dtype=float).reshape(4,1)
    else:
        half_theta = theta / 2.0
        axis = theta_vec / theta
        dq = np.array([
            np.cos(half_theta),
            np.sin(half_theta) * axis[0],
            np.sin(half_theta) * axis[1],
            np.sin(half_theta) * axis[2]
        ], dtype=float).reshape(4,1)
    
    q_old = q.reshape(4,1)
    q_new = _quat_mul(q_old, dq)
    q_new = _quat_normalize(q_new).reshape(4,)
    
    # Update state
    kf.x[0:3, 0] = p_new
    kf.x[3:6, 0] = v_new
    kf.x[6:10, 0] = q_new
    
    # ESKF covariance propagation
    Phi_err = compute_error_state_jacobian(q, a_corr, w_corr, dt, R_body_to_world)
    Q_err = compute_error_state_process_noise(dt, estimate_imu_bias, t, 0.0, imu_params, 0.8)
    
    num_clones = (kf.x.shape[0] - 16) // 7
    kf.P = propagate_error_state_covariance(kf.P, Phi_err, Q_err, num_clones)
    
    # Update priors
    kf.x_prior = kf.x.copy()
    kf.P_prior = kf.P.copy()


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication q1 * q2 where q = [w,x,y,z]."""
    w1, x1, y1, z1 = q1.flatten()
    w2, x2, y2, z2 = q2.flatten()
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]).reshape(4, 1)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0]).reshape(q.shape)
    return q / norm


# =============================================================================
# ZUPT (Zero Velocity Update)
# =============================================================================

def detect_stationary(a_raw: np.ndarray, w_corr: np.ndarray, v_mag: float,
                      imu_params: dict,
                      acc_threshold: float = 0.5,
                      gyro_threshold: float = 0.05) -> Tuple[bool, float]:
    """
    Detect if IMU is stationary based on raw measurements.
    
    Uses acceleration deviation from gravity and gyro magnitude.
    
    Args:
        a_raw: Raw acceleration from IMU (includes gravity)
        w_corr: Bias-corrected angular velocity
        v_mag: Current velocity magnitude
        imu_params: IMU parameters dict with 'g_norm'
        acc_threshold: Acceleration deviation threshold (m/s²)
        gyro_threshold: Gyro magnitude threshold (rad/s)
    
    Returns:
        is_stationary: True if IMU appears stationary
        acc_deviation: Acceleration deviation from gravity
    """
    a_raw_mag = np.linalg.norm(a_raw)
    gyro_mag = np.linalg.norm(w_corr)
    acc_deviation = abs(a_raw_mag - imu_params.get("g_norm", 9.80665))
    
    is_stationary = (acc_deviation < acc_threshold) and (gyro_mag < gyro_threshold)
    
    return is_stationary, acc_deviation


def apply_zupt(kf: ExtendedKalmanFilter, v_mag: float,
               consecutive_stationary_count: int,
               max_v_for_zupt: float = 500.0,
               save_debug: bool = False,
               residual_csv: Optional[str] = None,
               timestamp: float = 0.0,
               frame: int = -1) -> Tuple[bool, float, int]:
    """
    Apply Zero Velocity Update to constrain velocity to zero.
    
    ZUPT prevents IMU drift during stationary/hover periods by constraining
    the velocity estimate to zero when the vehicle is detected as stationary.
    
    Args:
        kf: Extended Kalman Filter
        v_mag: Current velocity magnitude [m/s]
        consecutive_stationary_count: Number of consecutive stationary samples
        max_v_for_zupt: Maximum velocity for ZUPT (safety check) [m/s]
        save_debug: Enable debug logging to residual CSV
        residual_csv: Path to residual log file
        timestamp: Current timestamp [s]
        frame: Current frame number
    
    Returns:
        Tuple of:
            - applied: True if ZUPT was applied
            - v_reduction: Velocity reduction magnitude [m/s]
            - updated_consecutive_count: Updated stationary count
    """
    if v_mag >= max_v_for_zupt:
        return False, 0.0, 0
    
    # Calculate dimensions
    num_clones = (kf.x.shape[0] - 16) // 7
    err_dim = 15 + 6 * num_clones
    
    # ZUPT Jacobian
    H_zupt = np.zeros((3, err_dim), dtype=float)
    H_zupt[0:3, 3:6] = np.eye(3)
    
    def h_zupt_jacobian(x, h=H_zupt):
        return h
    
    def hx_zupt_fun(x, h=H_zupt):
        return x[3:6].reshape(3, 1)
    
    # ZUPT measurement: velocity should be zero
    z_zupt = np.zeros((3, 1), dtype=float)
    
    # Adaptive measurement noise
    if v_mag < 1.0:
        base_r = 0.0001
    elif v_mag < 5.0:
        base_r = 0.001
    else:
        base_r = 0.01
    
    # Strengthen for sustained stationary periods
    consecutive_factor = max(1.0, min(10.0, consecutive_stationary_count / 100.0))
    R_zupt = np.diag([base_r / consecutive_factor] * 3)
    
    # Compute innovation and Mahalanobis distance for logging
    predicted_vel = kf.x[3:6, 0].reshape(3, 1)
    innovation = z_zupt - predicted_vel
    S_zupt = H_zupt @ kf.P @ H_zupt.T + R_zupt
    
    try:
        mahal_squared = float(innovation.T @ np.linalg.inv(S_zupt) @ innovation)
        mahal_dist = np.sqrt(mahal_squared)
    except:
        mahal_dist = float('nan')
    
    # Decouple yaw from velocity before ZUPT
    kf.P[3:6, 8] = 0.0
    kf.P[8, 3:6] = 0.0
    kf.P[3:6, 6:8] = 0.0
    kf.P[6:8, 3:6] = 0.0
    
    v_before = kf.x[3:6, 0].copy()
    
    kf.update(
        z=z_zupt,
        HJacobian=h_zupt_jacobian,
        Hx=hx_zupt_fun,
        R=R_zupt
    )
    
    v_after = kf.x[3:6, 0].copy()
    v_reduction = np.linalg.norm(v_before) - np.linalg.norm(v_after)
    
    # Log to residual CSV if debug enabled
    if save_debug and residual_csv:
        try:
            from .output_utils import log_measurement_update
            log_measurement_update(
                residual_csv, timestamp, frame, 'ZUPT',
                innovation=innovation.flatten(),
                mahalanobis_dist=mahal_dist,
                chi2_threshold=7.81,  # Chi-square 3 DOF, 95% confidence
                accepted=True,
                s_matrix=S_zupt,
                p_prior=kf.P
            )
        except Exception as e:
            print(f"[ZUPT] Warning: Failed to log to debug_residuals.csv: {e}")
    
    updated_count = consecutive_stationary_count + 1
    
    return True, v_reduction, updated_count


# =============================================================================
# Flight Phase Detection (v3.3.1: Pure State-based for GPS-denied realtime)
# =============================================================================

def get_flight_phase(velocity: Optional[np.ndarray] = None,
                     velocity_sigma: Optional[float] = None,
                     vibration_level: Optional[float] = None,
                     altitude_change: Optional[float] = None,
                     # Configurable thresholds (v3.4.0)
                     spinup_velocity_thresh: float = 1.0,
                     spinup_vibration_thresh: float = 0.3,
                     spinup_alt_change_thresh: float = 5.0,
                     early_velocity_sigma_thresh: float = 3.0) -> Tuple[int, str]:
    """
    Determine current flight phase based on VEHICLE STATE ONLY.
    
    v3.4.0: State-based with configurable thresholds - NO TIME DEPENDENCY for GPS-denied realtime.
    
    GPS-denied Realtime Philosophy:
    -------------------------------
    - NO absolute time reference (incompatible with restart/resume)
    - NO hardcoded time thresholds (not robust to flight variations)
    - ONLY vehicle state: velocity, uncertainty, vibration, altitude
    
    State-based Detection Logic:
    ----------------------------
    SPINUP (Phase 0): Rotor spin-up / Ground operations
      - Conditions: High vibration + Low horizontal velocity + Small altitude change
      - Characteristics: Stationary on ground, rotors spinning up
      - Trust level: LOW - Don't trust IMU heading or velocity
    
    EARLY (Phase 1): Initial flight / High uncertainty
      - Conditions: High velocity uncertainty (σ_v > 3 m/s)
      - Characteristics: Just took off, filter converging
      - Trust level: MEDIUM - Some sensor drift expected
    
    NORMAL (Phase 2): Stable flight / Low uncertainty
      - Conditions: Moving with low uncertainty
      - Characteristics: Steady flight, good sensor fusion
      - Trust level: HIGH - Full trust in sensor estimates
    
    Conservative Default:
    --------------------
    - If state inputs unavailable → Default to NORMAL (phase 2)
    - Rationale: Conservative approach, full sensor trust
    - Better to trust good sensors than assume degraded state
    
    Phases:
      0: SPINUP - Rotor spin-up (high vibration, stationary)
      1: EARLY - Early flight (high uncertainty, converging)
      2: NORMAL - Stable flight (low uncertainty, trusted)
    
    Args:
        velocity: Current velocity [vx, vy, vz] (m/s), REQUIRED
        velocity_sigma: Velocity uncertainty (m/s), REQUIRED
        vibration_level: Gyro std from VibrationDetector (rad/s), REQUIRED
        altitude_change: Change in altitude from start (m), OPTIONAL
        spinup_velocity_thresh: Velocity threshold for SPINUP detection (m/s)
        spinup_vibration_thresh: Vibration threshold for SPINUP detection (rad/s)
        spinup_alt_change_thresh: Altitude change threshold for SPINUP (m)
        early_velocity_sigma_thresh: Velocity uncertainty threshold for EARLY phase (m/s)
    
    Returns:
        phase: Phase number (0, 1, or 2)
        phase_name: Human-readable phase name
        
    Example:
        >>> phase, name = get_flight_phase(
        ...     velocity=np.array([0.5, 0.3, 0.0]),  # Stationary
        ...     velocity_sigma=2.5,
        ...     vibration_level=0.4,  # High vibration
        ...     altitude_change=1.0   # On ground
        ... )
        >>> print(f"Phase: {phase} = {name}")
        Phase: 0 = SPINUP
    """
    # GPS-denied Realtime: State-based detection ONLY
    # Thresholds now passed as parameters (v3.4.0)
    if velocity is not None and vibration_level is not None:
        vel_mag = np.linalg.norm(velocity[:2]) if len(velocity) >= 2 else 0.0
        alt_change = abs(altitude_change) if altitude_change is not None else 0.0
        
        # Phase 0: SPINUP - High vibration + stationary + on ground
        # Typical: Rotors spinning up, vehicle on ground
        if vibration_level > spinup_vibration_thresh and vel_mag < spinup_velocity_thresh:
            if alt_change < spinup_alt_change_thresh:
                return 0, "SPINUP"
        
        # Phase 1: EARLY - High velocity uncertainty (filter converging)
        # Typical: Just took off, sensors still converging
        if velocity_sigma is not None and velocity_sigma > early_velocity_sigma_thresh:
            return 1, "EARLY"
        
        # Phase 2: NORMAL - Stable flight with low uncertainty
        # Typical: Steady flight, good sensor fusion
        if vel_mag > spinup_velocity_thresh or alt_change > spinup_alt_change_thresh:
            if velocity_sigma is None or velocity_sigma < early_velocity_sigma_thresh:
                return 2, "NORMAL"
    
    # Conservative default: If no state data → assume NORMAL (full sensor trust)
    # Rationale: Better to trust good sensors than assume degraded state
    return 2, "NORMAL"


# =============================================================================
# Vibration Detection
# =============================================================================

class VibrationDetector:
    """
    Detect high vibration from gyro measurements using a rolling buffer.
    
    High vibration indicates IMU integration is less reliable.
    """
    
    def __init__(self, buffer_size: int = 100, threshold: float = 0.5):
        """
        Args:
            buffer_size: Size of rolling gyro buffer (~0.25s at 400Hz)
            threshold: Gyro std threshold for vibration detection (rad/s)
        """
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.gyro_buffer = []
        self.current_vibration_level = 0.0
        self.detection_count = 0
    
    def update(self, gyro_mag: float) -> Tuple[bool, float]:
        """
        Update with new gyro magnitude and check for vibration.
        
        Args:
            gyro_mag: Magnitude of angular velocity
        
        Returns:
            is_high_vibration: True if vibration detected
            vibration_level: Current gyro std
        """
        self.gyro_buffer.append(gyro_mag)
        if len(self.gyro_buffer) > self.buffer_size:
            self.gyro_buffer.pop(0)
        
        if len(self.gyro_buffer) >= 10:
            self.current_vibration_level = float(np.std(self.gyro_buffer))
            is_high_vibration = self.current_vibration_level > self.threshold
            if is_high_vibration:
                self.detection_count += 1
            return is_high_vibration, self.current_vibration_level
        
        return False, 0.0
    
    def reset(self):
        """Reset detector state."""
        self.gyro_buffer = []
        self.current_vibration_level = 0.0


# =============================================================================
# Camera State Augmentation
# =============================================================================

def augment_state_with_camera(kf: ExtendedKalmanFilter, cam_q_wxyz: np.ndarray, 
                              cam_p: np.ndarray, cam_states: list, 
                              cam_observations: list,
                              p_quat: float = 1e-3, p_pos: float = 1.0, 
                              max_poses: int = 5) -> int:
    """
    Augment EKF state with camera pose for MSCKF sliding window.
    
    CRITICAL: Works with ESKF (Error-State Kalman Filter):
      - Nominal state x: [p,v,q,bg,ba, q_C1,p_C1, ...] (16+7M dimensions)
      - Error-state covariance P: [δp,δv,δθ,δbg,δba, δθ_C1,δp_C1, ...] (15+6M dimensions)

    Args:
        kf: Extended Kalman Filter
        cam_q_wxyz: Camera quaternion [w,x,y,z]
        cam_p: Camera position [x,y,z]
        cam_states: List of camera state metadata
        cam_observations: List of observation records
        p_quat: Prior uncertainty for quaternion (→ 3D rotation)
        p_pos: Prior uncertainty for position
        max_poses: Maximum poses in sliding window

    Returns:
        Start index of appended block
    """
    pose_size_nominal = 7  # quaternion (4) + position (3)
    pose_size_error = 6    # rotation (3) + position (3)
    core_size_nominal = 16
    core_size_error = 15
    
    old_n_nominal = kf.dim_x
    num_poses = (old_n_nominal - core_size_nominal) // pose_size_nominal
    old_n_error = core_size_error + num_poses * pose_size_error
    
    # Marginalize oldest if at max
    if num_poses >= max_poses:
        old_pose_idx_nominal = core_size_nominal
        old_pose_idx_error = core_size_error
        
        mask_nominal = np.ones(old_n_nominal, dtype=bool)
        mask_nominal[old_pose_idx_nominal:old_pose_idx_nominal + pose_size_nominal] = False
        
        mask_error = np.ones(old_n_error, dtype=bool)
        mask_error[old_pose_idx_error:old_pose_idx_error + pose_size_error] = False
        
        kf.x = kf.x[mask_nominal]
        kf.P = kf.P[np.ix_(mask_error, mask_error)]
        
        old_n_nominal = kf.x.shape[0]
        old_n_error = kf.P.shape[0]
        kf.dim_x = old_n_nominal
        
        old_cam_id = 0
        old_obs_count = len([obs for obs in cam_observations if obs['cam_id'] == old_cam_id])
        
        if len(cam_states) > 0:
            cam_states.pop(0)
        
        # Update indices
        for cs in cam_states:
            cs['start_idx'] -= pose_size_nominal
            cs['q_idx'] -= pose_size_nominal
            cs['p_idx'] -= pose_size_nominal
            cs['err_q_idx'] -= pose_size_error
            cs['err_p_idx'] -= pose_size_error
        
        # Update observation cam_ids
        new_observations = []
        for obs in cam_observations:
            if obs['cam_id'] > 0:
                new_obs = obs.copy()
                new_obs['cam_id'] = obs['cam_id'] - 1
                new_observations.append(new_obs)
        cam_observations[:] = new_observations
        
        print(f"[VIO] Marginalized oldest pose, removed {old_obs_count} observation sets, "
              f"now tracking {len(cam_states)} poses")

    # Augment with new pose
    add_n_nominal = pose_size_nominal
    add_n_error = pose_size_error
    new_n_nominal = old_n_nominal + add_n_nominal
    new_n_error = old_n_error + add_n_error

    # Augment nominal state
    new_x = np.zeros((new_n_nominal, 1), dtype=float)
    new_x[:old_n_nominal, 0] = kf.x.reshape(-1)[:old_n_nominal]
    new_x[old_n_nominal:old_n_nominal+4, 0] = cam_q_wxyz.reshape(4,)
    new_x[old_n_nominal+4:old_n_nominal+7, 0] = cam_p.reshape(3,)

    # Augment error-state covariance
    new_P = np.zeros((new_n_error, new_n_error), dtype=float)
    new_P[:old_n_error, :old_n_error] = kf.P
    new_P[old_n_error:old_n_error+3, old_n_error:old_n_error+3] = np.eye(3) * p_quat
    new_P[old_n_error+3:old_n_error+6, old_n_error+3:old_n_error+6] = np.eye(3) * p_pos

    # Update filter (ESKF fix: separate nominal and error dimensions)
    kf.x = new_x
    kf.P = new_P
    kf.dim_x = new_n_nominal    # Nominal state: 16 + 7*N
    kf.dim_err = new_n_error    # Error state: 15 + 6*N
    
    # Update error-state matrices
    kf.F = np.eye(new_n_error, dtype=float)
    kf.Q = np.eye(new_n_error, dtype=float)
    kf._I = np.eye(new_n_error, dtype=float)
    kf.K = np.zeros((new_n_error, kf.dim_z), dtype=float)

    kf.x_prior = kf.x.copy()
    kf.P_prior = kf.P.copy()
    kf.x_post = kf.x.copy()
    kf.P_post = kf.P.copy()

    return old_n_nominal


# =============================================================================
# Preintegration Application at Camera Frame
# =============================================================================

def apply_preintegration_at_camera(kf: ExtendedKalmanFilter, 
                                   ongoing_preint,
                                   t: float, imu_params: dict) -> dict:
    """
    Snapshot preintegration Jacobians and reset buffer at camera frame.
    
    CRITICAL (v3.5.1): This function does NOT propagate covariance anymore!
    - Covariance (P) is already propagated by process_imu() at every IMU tick (400Hz)
    - This function ONLY snapshots Jacobians for MSCKF/bias observability
    - Then resets preintegration buffer for next camera frame
    
    Why NOT propagate P here?
    -------------------------
    If we propagate P here with preintegration, we would DOUBLE-COUNT:
      1. P propagated every IMU tick via process_imu() (400Hz)
      2. P propagated again here with same IMU data (20Hz)
      → Result: P inflates (overconfident) → NEES wrong → gating broken
    
    Correct approach (v3.5.1):
    - P propagates continuously at IMU rate
    - Preintegration is ONLY for MSCKF Jacobians (bias observability)
    
    Args:
        kf: ExtendedKalmanFilter instance (P already propagated!)
        ongoing_preint: IMUPreintegration object
        t: Current timestamp
        imu_params: IMU noise parameters (unused now)
    
    Returns:
        dict: Preintegration Jacobians for MSCKF (J_R_bg, J_v_bg, etc.)
              Returns None if dt too small
    """
    from scipy.spatial.transform import Rotation as R_scipy
    from .math_utils import skew_symmetric
    
    dt_total = ongoing_preint.dt_sum
    if dt_total < 1e-6:
        return None
    
    # Current biases for reset
    bg = kf.x[10:13, 0].reshape(3,)
    ba = kf.x[13:16, 0].reshape(3,)
    
    # Snapshot Jacobians for MSCKF/bias observability
    # These Jacobians relate integrated motion to biases (δbg, δba)
    J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba = ongoing_preint.get_jacobians()
    
    preint_jacobians = {
        'J_R_bg': J_R_bg.copy(),
        'J_v_bg': J_v_bg.copy(),
        'J_v_ba': J_v_ba.copy(),
        'J_p_bg': J_p_bg.copy(),
        'J_p_ba': J_p_ba.copy(),
        'dt_total': dt_total  # For MSCKF temporal baseline
    }
    
    # Reset preintegration buffer for next camera frame
    ongoing_preint.reset(bg=bg, ba=ba)
    
    print(f"[PREINT] Jacobians snapshotted and reset: Δt={dt_total:.3f}s")
    
    return preint_jacobians


def clone_camera_for_msckf(kf: ExtendedKalmanFilter, t: float,
                           cam_states: list, cam_observations: list,
                           vio_fe, frame_idx: int,
                           preint_jacobians: dict = None) -> int:
    """
    Clone current IMU pose for MSCKF.
    
    Args:
        kf: ExtendedKalmanFilter instance
        t: Camera timestamp
        cam_states: List of camera clone states
        cam_observations: List of camera observations
        vio_fe: VIO frontend (for getting feature tracks)
        frame_idx: Current frame index
        preint_jacobians: Preintegration Jacobians snapshot for bias observability
            Contains: J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba
    
    Returns:
        Clone index (for FEJ tracking)
    """
    p_imu = kf.x[0:3, 0].reshape(3,)
    q_imu = kf.x[6:10, 0].reshape(4,)
    
    try:
        start_idx = augment_state_with_camera(
            kf, q_imu, p_imu,
            cam_states, cam_observations
        )
        
        # Store FEJ linearization points
        clone_idx = len(cam_states)
        err_theta_idx = 15 + 6 * clone_idx
        err_p_idx = 15 + 6 * clone_idx + 3
        
        # Record observations
        obs_data = []
        if hasattr(vio_fe, 'get_tracks_for_frame'):
            tracks = vio_fe.get_tracks_for_frame(frame_idx)
            for fid, pt in tracks:
                pt_array = np.array([[pt[0], pt[1]]], dtype=np.float32)
                pt_norm = vio_fe._undistort_pts(pt_array).reshape(2,)
                obs_data.append({
                    'fid': int(fid),
                    'pt_pixel': (float(pt[0]), float(pt[1])),
                    'pt_norm': (float(pt_norm[0]), float(pt_norm[1])),
                    'quality': 1.0
                })
        
        cam_state_entry = {
            'start_idx': start_idx,
            'q_idx': start_idx,
            'p_idx': start_idx + 4,
            'err_q_idx': err_theta_idx,
            'err_p_idx': err_p_idx,
            't': t,
            'timestamp': t,
            'frame': frame_idx,
            'q_fej': q_imu.copy(),
            'p_fej': p_imu.copy(),
            'bg_fej': kf.x[10:13, 0].copy(),
            'ba_fej': kf.x[13:16, 0].copy()
        }
        
        # Add preintegration Jacobians for bias observability
        if preint_jacobians is not None:
            cam_state_entry['J_R_bg'] = preint_jacobians.get('J_R_bg')
            cam_state_entry['J_v_bg'] = preint_jacobians.get('J_v_bg')
            cam_state_entry['J_v_ba'] = preint_jacobians.get('J_v_ba')
            cam_state_entry['J_p_bg'] = preint_jacobians.get('J_p_bg')
            cam_state_entry['J_p_ba'] = preint_jacobians.get('J_p_ba')
        
        cam_states.append(cam_state_entry)
        
        cam_observations.append({
            'cam_id': clone_idx,
            'frame': frame_idx,
            't': t,
            'observations': obs_data
        })
        
        print(f"[CLONE] Created clone {clone_idx} with {len(obs_data)} observations")
        
        return clone_idx
        
    except Exception as e:
        print(f"[CLONE] Failed: {e}")
        return -1

