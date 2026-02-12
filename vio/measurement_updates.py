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
from typing import Optional, Tuple, Callable, Dict, Any
import math

# Import shared math utilities (avoid duplication)
from .math_utils import quaternion_to_yaw, mahalanobis_squared, safe_matrix_inverse
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


# NOTE: apply_zupt_update() has been REMOVED (v2.9.6)
# Reason: Redundant - duplicates functionality of apply_zupt() in propagation.py
# Use detect_stationary() + apply_zupt() instead (more modular approach)
# See propagation.py for the active ZUPT implementation with debug logging


# Global state for magnetometer statistics (v2.9.5: oscillation detection with skip_count=2)
_MAG_STATE = {
    # Rejection statistics
    'reject_quality': 0,
    'reject_innovation': 0,
    'reject_high_rate': 0,     # From apply_mag_filter
    'reject_gyro_inconsistent': 0,  # From apply_mag_filter
    'total_attempts': 0,
    'total_accepted': 0,
    # Oscillation detection (v2.9.5: back with reduced skip)
    'innovation_history': [],  # Track recent innovation signs
    'oscillation_count': 0,    # Number of oscillations detected
    'skip_count': 0,           # Remaining updates to skip
}


def apply_magnetometer_update(kf,
                              mag_calibrated: np.ndarray,
                              mag_declination: float,
                              use_raw_heading: bool,
                              sigma_mag_yaw: float,
                              current_phase: int = 2,
                              gyro_z: float = 0.0,
                              in_convergence: bool = False,
                              has_ppk_yaw: bool = False,
                              save_debug: bool = False,
                              residual_csv: Optional[str] = None,
                              timestamp: float = 0.0,
                              frame: int = -1,
                              yaw_override: Optional[float] = None,
                              filter_info: Optional[dict] = None,
                              use_estimated_bias: bool = True,
                              r_scale_extra: float = 1.0,
                              adaptive_info: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    Apply magnetometer heading update.
    
    Features:
    - GPS-calibrated raw heading or tilt-compensated heading
    - Adaptive Kalman gain with phase-based K_MIN
    - Yaw correction limiting to prevent attitude destabilization
    - Cross-covariance decoupling to prevent altitude drift
    - Oscillation detection to prevent yaw bouncing near ±180°
    - OPTIONAL: EMA-filtered yaw can be passed via yaw_override
    
    Args:
        kf: ExtendedKalmanFilter instance
        mag_calibrated: Calibrated magnetometer vector [x, y, z]
        mag_declination: Magnetic declination in radians
        use_raw_heading: Use raw body-frame heading (GPS-calibrated)
        sigma_mag_yaw: Base measurement noise for yaw (can be scaled by filter)
        current_phase: Flight phase (0=SPINUP, 1=EARLY, 2=NORMAL)
        gyro_z: Current gyro Z for consistency check
        in_convergence: Whether in initial convergence period
        has_ppk_yaw: Whether PPK provided initial yaw (skip if in convergence)
        save_debug: Enable debug logging
        residual_csv: Path to residual log
        timestamp: Current timestamp
        frame: Current frame number
        yaw_override: If provided, use this yaw instead of computing from mag (for filtered yaw)
        filter_info: Dictionary from apply_mag_filter() with {'high_rate': bool, 'gyro_inconsistent': bool}
        use_estimated_bias: If True, include mag_bias in Jacobian. If False, freeze mag_bias states.
        
    Returns:
        Tuple of (update_applied: bool, rejection_reason: str)
    """
    global _MAG_STATE
    extra_scale = max(1e-3, float(r_scale_extra))

    def _set_adaptive_info(accepted: bool,
                           nis_norm: Optional[float],
                           chi2: Optional[float],
                           threshold: Optional[float],
                           r_scale_used: float):
        if adaptive_info is None:
            return
        adaptive_info.clear()
        adaptive_info.update({
            "sensor": "MAG",
            "accepted": bool(accepted),
            "dof": 1,
            "nis_norm": float(nis_norm) if nis_norm is not None and np.isfinite(nis_norm) else np.nan,
            "chi2": float(chi2) if chi2 is not None and np.isfinite(chi2) else np.nan,
            "threshold": float(threshold) if threshold is not None and np.isfinite(threshold) else np.nan,
            "r_scale_used": float(r_scale_used),
        })
    from .magnetometer import compute_yaw_from_mag
    
    # Track filter rejection reasons (v2.9.2)
    if filter_info is not None:
        if filter_info.get('high_rate', False):
            _MAG_STATE['reject_high_rate'] += 1
        if filter_info.get('gyro_inconsistent', False):
            _MAG_STATE['reject_gyro_inconsistent'] += 1
    
    # v3.9.0: REMOVED - PPK convergence skip was blocking ALL mag updates
    # Mag updates should still work during convergence with reduced weight
    # if in_convergence and has_ppk_yaw:
    #     return False, "PPK convergence period"
    
    # Track total attempts
    _MAG_STATE['total_attempts'] += 1
    
    # v3.4.0: Oscillation skip check - DISABLED during SPINUP phase!
    # v2.9.5: Oscillation detection (skip_count = 2, reduced from 10)
    # Check if we're in skip period - BUT NOT during SPINUP (phase 0)
    if current_phase >= 2 and _MAG_STATE['skip_count'] > 0:
        _MAG_STATE['skip_count'] -= 1
        _set_adaptive_info(False, None, None, None, r_scale_extra)
        return False, f"Oscillation skip (remaining: {_MAG_STATE['skip_count']})"
    elif current_phase < 2:
        # During SPINUP/EARLY: reset skip_count to ensure continuous updates
        _MAG_STATE['skip_count'] = 0
    
    # Check field strength
    mag_norm = np.linalg.norm(mag_calibrated)
    
    # Compute yaw from magnetometer (or use filtered yaw if provided)
    q_state = kf.x[6:10, 0]
    if yaw_override is not None:
        # Use pre-filtered yaw (EMA smoothed + gyro consistency checked)
        yaw_mag = yaw_override
        quality = 1.0  # Filter already handled quality
    else:
        # Standard path: compute yaw directly
        yaw_mag, quality = compute_yaw_from_mag(
            mag_calibrated, q_state,
            mag_declination=mag_declination,
            use_raw_heading=use_raw_heading
        )
    
    if quality < 0.3:
        _MAG_STATE['reject_quality'] += 1
        if _MAG_STATE['total_attempts'] % 100 == 0:
            print(f"[MAG-REJECT] Quality: {_MAG_STATE['reject_quality']}/{_MAG_STATE['total_attempts']} ({_MAG_STATE['reject_quality']/_MAG_STATE['total_attempts']*100:.1f}%)")
        _set_adaptive_info(False, None, None, None, r_scale_extra)
        return False, f"Low quality ({quality:.3f})"
    
    # Get current yaw from state
    yaw_state = quaternion_to_yaw(q_state)
    
    # Compute innovation with angle wrapping
    yaw_innov = yaw_mag - yaw_state
    yaw_innov = np.arctan2(np.sin(yaw_innov), np.cos(yaw_innov))
    
    # =========================================================================
    # v2.9.10.13: Oscillation Detection - DISABLED during spinup!
    # =========================================================================
    # During spinup (t<15s), we do FULL CORRECTION (K=1.0) so oscillation is
    # expected and normal - gyro drift causes yaw to swing back and forth as
    # mag corrects it. DON'T skip updates during spinup - that causes drift!
    
    # v2.9.5: Oscillation Detection (skip_count=2, reduced from v2.9.2's 10)
    # Detect rapid sign alternation in innovation (indicates erratic behavior)
    # When detected, skip 2 updates (minimal gap ~0.1s) to let system stabilize
    
    # v3.4.0: SKIP oscillation detection during SPINUP/EARLY phases!
    if current_phase >= 2:  # ONLY in NORMAL phase
        innovation_sign = 1 if yaw_innov > 0 else -1
        _MAG_STATE['innovation_history'].append(innovation_sign)
        
        # Keep only recent history (last 3 innovations)
        if len(_MAG_STATE['innovation_history']) > 3:
            _MAG_STATE['innovation_history'].pop(0)
        
        # Detect oscillation: alternating signs in recent history
        if len(_MAG_STATE['innovation_history']) >= 3:
            h = _MAG_STATE['innovation_history']
            if (h[-1] != h[-2]) and (h[-2] != h[-3]):
                # Oscillation detected: skip next 2 updates (reduced from 10)
                _MAG_STATE['skip_count'] = 2  # CRITICAL: 2 instead of 10!
                _MAG_STATE['oscillation_count'] += 1
                
                if _MAG_STATE['total_attempts'] % 100 == 0:
                    print(f"[MAG-OSC] Detected oscillation #{_MAG_STATE['oscillation_count']}, skipping next 2 updates")
                _set_adaptive_info(False, None, None, None, r_scale_extra)
                return False, "Oscillation detected (skip=2)"
    else:
        # During spinup: clear history and don't track oscillations
        _MAG_STATE['innovation_history'] = []
    
    # Adaptive R-scaling during oscillation buildup (v2.9.2 logic)
    # Apply progressive R-inflation if oscillation pattern developing
    # v3.4.0: DISABLED during SPINUP/EARLY - always use R_scale=1.0
    oscillation_r_scale = 1.0
    if current_phase >= 2 and len(_MAG_STATE['innovation_history']) >= 2:
        h = _MAG_STATE['innovation_history']
        if h[-1] != h[-2]:
            # One alternation: light R-scaling
            oscillation_r_scale = 1.0 + 2.0  # R = 3.0
        if len(_MAG_STATE['innovation_history']) >= 3 and h[-2] != h[-3]:
            # Two alternations: heavier R-scaling
            oscillation_r_scale = 1.0 + 4.0  # R = 5.0
    
    # Innovation threshold (very permissive - mag is absolute reference)
    if in_convergence:
        innovation_threshold = np.radians(180.0)
    else:
        innovation_threshold = np.radians(179.0)
    
    if abs(yaw_innov) > innovation_threshold:
        _MAG_STATE['reject_innovation'] += 1
        if _MAG_STATE['total_attempts'] % 100 == 0:
            print(f"[MAG-REJECT] Innovation: {_MAG_STATE['reject_innovation']}/{_MAG_STATE['total_attempts']} ({_MAG_STATE['reject_innovation']/_MAG_STATE['total_attempts']*100:.1f}%)")
        _set_adaptive_info(False, None, None, innovation_threshold, r_scale_extra)
        return False, f"Innovation too large ({np.degrees(yaw_innov):.1f}°)"
    
    # Apply oscillation R-scaling to measurement noise (v2.9.5)
    sigma_mag_yaw_scaled = sigma_mag_yaw * oscillation_r_scale * extra_scale
    
    # Build measurement model
    _MAG_STATE['total_accepted'] += 1
    
    # Print summary statistics periodically
    if _MAG_STATE['total_attempts'] % 500 == 0:
        acc_rate = _MAG_STATE['total_accepted'] / _MAG_STATE['total_attempts'] * 100
        print(f"[MAG-STATS] Accepted: {_MAG_STATE['total_accepted']}/{_MAG_STATE['total_attempts']} ({acc_rate:.1f}%), osc={_MAG_STATE['oscillation_count']}")
        print(f"[MAG-STATS] Reject reasons: qual={_MAG_STATE['reject_quality']}, "
              f"innov={_MAG_STATE['reject_innovation']}, high_rate={_MAG_STATE['reject_high_rate']}, gyro_incons={_MAG_STATE['reject_gyro_inconsistent']}")
    
    # Build measurement model
    num_clones = (kf.x.shape[0] - 19) // 7  # v3.9.7: 19D nominal
    err_dim = 18 + 6 * num_clones  # v3.9.7: 18D core error
    theta_cov_idx = 8  # δθ_z in error state
    
    # v2.9.10.13: Check body Z direction to determine sign for yaw correction
    # For nadir camera (body Z points DOWN = -world Z), body δθ_z = -world δyaw
    # This requires H[0,8] = -1.0 to properly map world yaw to body rotation
    q_current = kf.x[6:10, 0]
    from scipy.spatial.transform import Rotation as R_scipy_check
    q_xyzw = np.array([q_current[1], q_current[2], q_current[3], q_current[0]])
    r_check = R_scipy_check.from_quat(q_xyzw)
    body_z_world = r_check.apply(np.array([0.0, 0.0, 1.0]))
    
    # If body Z points DOWN (negative world Z), flip sign
    yaw_sign = -1.0 if body_z_world[2] < 0 else 1.0
    
    # v3.9.7: Compute mag_bias Jacobian for online estimation
    # FIXED DERIVATION:
    # yaw = atan2(-mag_y_cal, mag_x_cal) + declination
    # 
    # Step 1: ∂yaw/∂mag_cal
    #   ∂yaw/∂mag_x = -(-mag_y) / (mag_x² + mag_y²) = mag_y / denom
    #   ∂yaw/∂mag_y = -mag_x / (mag_x² + mag_y²)  ← negative from atan2(-mag_y, ...)
    #
    # Step 2: mag_cal = mag_raw - hard_iron  →  ∂mag_cal/∂hard_iron = -I
    #
    # Step 3: Chain rule: ∂yaw/∂hard_iron = ∂yaw/∂mag_cal × (-I) = -∂yaw/∂mag_cal
    #   ∂yaw/∂hi_x = -mag_y / denom
    #   ∂yaw/∂hi_y = mag_x / denom   (double negative: -(-mag_x/denom))
    #
    mag_x, mag_y = mag_calibrated[0], mag_calibrated[1]
    mag_denom = mag_x**2 + mag_y**2
    if mag_denom > 1e-6:
        # Jacobian of yaw w.r.t. hard_iron bias (FIXED sign)
        dyaw_dhix = -mag_y / mag_denom   # ∂yaw/∂hard_iron_x (negative!)
        dyaw_dhiy = mag_x / mag_denom    # ∂yaw/∂hard_iron_y (positive!)
        dyaw_dhiz = 0.0                   # Z-axis bias doesn't affect horizontal yaw
    else:
        dyaw_dhix = dyaw_dhiy = dyaw_dhiz = 0.0
    
    # v3.9.8: When use_estimated_bias=False, freeze mag_bias by setting Jacobian=0
    if not use_estimated_bias:
        dyaw_dhix = dyaw_dhiy = dyaw_dhiz = 0.0
    
    def h_mag_fun(x):
        h_yaw = np.zeros((1, err_dim), dtype=float)
        h_yaw[0, 8] = yaw_sign  # v2.9.10.13: Sign depends on body orientation!
        # v3.9.7: Add mag_bias Jacobian terms (error state indices 15:18)
        # NOTE: Do NOT multiply by yaw_sign! yaw_sign is for attitude error,
        # but mag_bias affects raw measurement directly (sensor frame).
        h_yaw[0, 15] = dyaw_dhix  # δmag_bias_x
        h_yaw[0, 16] = dyaw_dhiy  # δmag_bias_y  
        h_yaw[0, 17] = dyaw_dhiz  # δmag_bias_z
        return h_yaw
    
    def hx_mag_fun(x):
        q_x = x[6:10, 0]
        yaw_x = quaternion_to_yaw(q_x)
        return np.array([[yaw_x]])
    
    # Measurement covariance (use scaled sigma from oscillation detection)
    r_yaw = np.array([[sigma_mag_yaw_scaled**2]])
    
    # Phase-based Kalman gain tuning (v3.4.0)
    # SPINUP (0): FULL CORRECTION (K=1.0) to completely prevent drift
    # Problem: Gyro drift ~0.44°/s accumulates between mag updates (~50ms apart)
    #          Even K=0.90 leaves 10% residual → exponential drift buildup
    # Solution: K=1.0 (100% correction) in SPINUP - trust magnetometer completely
    if current_phase == 0:  # SPINUP phase - FULL CORRECTION
        K_MIN = 0.999  # Use 0.999 instead of 1.0 to avoid div by zero!
        MAX_YAW_CORRECTION = np.radians(180.0)  # NO CLAMP in SPINUP!
    elif current_phase == 1:  # EARLY phase - moderate correction
        K_MIN = 0.70  # Higher trust in mag during convergence
        MAX_YAW_CORRECTION = np.radians(90.0)  # Allow larger corrections
    else:  # NORMAL phase - gentler correction
        K_MIN = 0.40  # Standard tracking
        MAX_YAW_CORRECTION = np.radians(60.0)  # Conservative limit
    
    # Compute current Kalman gain
    P_yaw = kf.P[theta_cov_idx, theta_cov_idx]
    S_yaw = P_yaw + sigma_mag_yaw_scaled**2  # BUG FIX: Use scaled sigma!
    K_yaw = P_yaw / S_yaw
    
    # Enforce minimum K
    # v2.9.10.13: Handle K_MIN close to 1.0 to avoid division by near-zero
    if K_yaw < K_MIN and K_MIN > 0.01 and K_MIN < 0.9999:
        P_yaw_min = K_MIN * sigma_mag_yaw_scaled**2 / (1.0 - K_MIN)  # BUG FIX: Use scaled sigma!
        if kf.P[theta_cov_idx, theta_cov_idx] < P_yaw_min:
            kf.P[theta_cov_idx, theta_cov_idx] = P_yaw_min
            P_yaw = P_yaw_min
            K_yaw = K_MIN
    elif K_MIN >= 0.9999:
        # v2.9.10.13: For K_MIN ≈ 1.0, set P_yaw very large to achieve K ≈ 1
        P_yaw_target = 1e6 * sigma_mag_yaw_scaled**2  # Very large P → K → 1
        if kf.P[theta_cov_idx, theta_cov_idx] < P_yaw_target:
            kf.P[theta_cov_idx, theta_cov_idx] = P_yaw_target
            P_yaw = P_yaw_target
            S_yaw = P_yaw + sigma_mag_yaw_scaled**2
            K_yaw = P_yaw / S_yaw  # ≈ 0.999999
    
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
            residual=angle_residual,
            update_type="MAG",
            timestamp=timestamp
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
        m2 = mahalanobis_squared(np.array([yaw_innov]), np.array([[S_yaw]]))
        _set_adaptive_info(True, m2, m2, np.inf, extra_scale)
        return True, ""
    except Exception as e:
        _set_adaptive_info(False, None, None, None, extra_scale)
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
                            frame: int = -1,
                            threshold_scale: float = 1.0,
                            r_scale_extra: float = 1.0,
                            adaptive_info: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
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
    def _set_adaptive_info(accepted: bool,
                           nis_norm: Optional[float],
                           chi2: Optional[float],
                           threshold: Optional[float],
                           r_scale_used: float):
        if adaptive_info is None:
            return
        adaptive_info.clear()
        adaptive_info.update({
            "sensor": "DEM",
            "accepted": bool(accepted),
            "dof": 1,
            "nis_norm": float(nis_norm) if nis_norm is not None and np.isfinite(nis_norm) else np.nan,
            "chi2": float(chi2) if chi2 is not None and np.isfinite(chi2) else np.nan,
            "threshold": float(threshold) if threshold is not None and np.isfinite(threshold) else np.nan,
            "r_scale_used": float(r_scale_used),
        })

    if np.isnan(height_measurement):
        _set_adaptive_info(False, None, None, None, r_scale_extra)
        return False, "Invalid height measurement"
    
    num_clones = (kf.x.shape[0] - 19) // 7  # v3.9.7: 19D nominal
    err_dim = 18 + 6 * num_clones  # v3.9.7: 18D core error
    
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
    
    extra_scale = max(1e-3, float(r_scale_extra))
    r_mat = np.array([[sigma_height**2 * height_cov_scale * (extra_scale ** 2)]])
    
    # VALIDATION: Check P matrix for numerical issues before innovation computation
    # This prevents divide-by-zero and overflow from corrupted covariance
    has_invalid_p = np.any(np.isinf(kf.P)) or np.any(np.isnan(kf.P))
    if has_invalid_p:
        # P corrupted - cannot compute valid innovation, skip update
        _set_adaptive_info(False, None, None, None, extra_scale)
        return False, "Invalid P matrix (contains inf/nan)"
    
    # Clamp large P values to prevent overflow in matmul
    P_max = np.max(np.abs(kf.P))
    if P_max > 1e10:
        # Scale P down to prevent numerical explosion
        scale_factor = 1e8 / P_max
        kf.P = kf.P * scale_factor
        _set_adaptive_info(False, None, None, None, extra_scale)
        return False, f"P matrix overflow clamped (max={P_max:.2e})"
    
    from .numerical_checks import assert_finite
    
    # [TRIPWIRE] Validate inputs before S matrix computation
    if not assert_finite("height_H", H_height, extra_info={"H_norm": np.linalg.norm(H_height)}):
        _set_adaptive_info(False, None, None, None, extra_scale)
        return False, "H_height contains inf/nan"
    
    if not assert_finite("height_kf_P", kf.P, extra_info={
        "P_max": np.max(np.abs(kf.P)),
        "P_trace": np.trace(kf.P)
    }):
        _set_adaptive_info(False, None, None, None, extra_scale)
        return False, "kf.P contains inf/nan before height update"
    
    # Innovation gating with overflow protection
    # Suppress numpy warnings - we handle explicitly with tripwires
    with np.errstate(all='ignore'):
        try:
            S_mat = H_height @ kf.P @ H_height.T + r_mat
        except Exception as e:
            _set_adaptive_info(False, None, None, None, extra_scale)
            return False, f"S_mat computation failed: {e}"
    
    # [TRIPWIRE] Check S_mat validity after computation
    if not assert_finite("height_S_mat", S_mat, extra_info={
        "S_mat_max": np.max(np.abs(S_mat)),
        "r_mat_val": r_mat[0,0] if r_mat.size > 0 else 0,
        "H_norm": np.linalg.norm(H_height),
        "P_max": np.max(np.abs(kf.P))
    }):
        _set_adaptive_info(False, None, None, None, extra_scale)
        return False, "S_mat contains inf/nan after computation"
    
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
    threshold = max(1e-6, threshold * max(1e-3, float(threshold_scale)))
    
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
            R=r_mat_inflated,
            update_type="DEM",
            timestamp=timestamp
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
        _set_adaptive_info(True, m2_test, m2_test, threshold, extra_scale)
        return True, f"Soft gating (m2={m2_test:.2f}, inflated R by {inflation_factor:.1f}x)"
    
    # Normal update
    kf.update(
        z=np.array([[height_measurement]]),
        HJacobian=h_fun,
        Hx=hx_fun,
        R=r_mat,
        update_type="DEM",
        timestamp=timestamp
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
    _set_adaptive_info(True, m2_test, m2_test, threshold, extra_scale)
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
    num_clones = (kf.x.shape[0] - 19) // 7  # v3.9.7: 19D nominal
    err_dim = 18 + 6 * num_clones  # v3.9.7: 18D core error
    
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
        R=r_mat,
        update_type="VIO_VEL",
        timestamp=timestamp
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


def apply_vio_velocity_update(kf, r_vo_mat: np.ndarray, t_unit: np.ndarray,
                               t: float, dt_img: float, avg_flow_px: float,
                               imu_rec, global_config: dict, camera_view: str,
                               dem_reader, lat0: float, lon0: float,
                               use_vio_velocity: bool,
                               proj_cache,
                               save_debug: bool = False,
                               residual_csv: Optional[str] = None,
                               vio_frame: int = -1,
                               vio_fe=None,
                               state_error: Optional[np.ndarray] = None,
                               state_cov: Optional[np.ndarray] = None,
                               chi2_scale: float = 1.0,
                               r_scale_extra: float = 1.0,
                               adaptive_info: Optional[Dict[str, Any]] = None) -> bool:
    """
    Apply VIO velocity update with scale recovery and chi-square gating.
    
    This function:
    1. Recovers scale from AGL using optical flow
    2. Computes velocity in world frame
    3. Applies chi-square innovation gating
    4. Updates EKF if innovation passes gating
    
    Args:
        kf: ExtendedKalmanFilter instance
        r_vo_mat: Relative rotation matrix from Essential matrix
        t_unit: Unit translation vector from Essential matrix
        t: Current timestamp
        dt_img: Time between images
        avg_flow_px: Average optical flow in pixels
        imu_rec: Current IMU record
        global_config: Global configuration dictionary
        camera_view: Camera view mode
        dem_reader: DEM reader for AGL
        lat0, lon0: Origin coordinates
        use_vio_velocity: Whether to apply velocity update
        proj_cache: ProjectionCache instance for coordinate conversion
        save_debug: Enable debug logging
        residual_csv: Path to residual CSV
        vio_frame: Current VIO frame index
        vio_fe: VIO frontend (for flow direction)
    
    Returns:
        True if update was accepted, False otherwise
    """
    def _set_adaptive_info(accepted: bool,
                           dof: int,
                           chi2: Optional[float],
                           threshold: Optional[float],
                           r_scale_used: float):
        if adaptive_info is None:
            return
        adaptive_info.clear()
        if chi2 is not None and np.isfinite(chi2) and dof > 0:
            nis_norm = float(chi2) / float(dof)
        else:
            nis_norm = np.nan
        adaptive_info.update({
            "sensor": "VIO_VEL",
            "accepted": bool(accepted),
            "dof": int(max(1, dof)),
            "nis_norm": nis_norm,
            "chi2": float(chi2) if chi2 is not None and np.isfinite(chi2) else np.nan,
            "threshold": float(threshold) if threshold is not None and np.isfinite(threshold) else np.nan,
            "r_scale_used": float(r_scale_used),
        })

    from scipy.spatial.transform import Rotation as R_scipy
    from .config import CAMERA_VIEW_CONFIGS
    from .data_loaders import ProjectionCache
    
    kb_params = global_config.get('KB_PARAMS', {'mu': 600})
    sigma_vo = global_config.get('SIGMA_VO', 0.5)
    
    # Get camera view config from loaded YAML (not hardcoded defaults)
    view_cfg = global_config.get('CAMERA_VIEW_CONFIGS', {}).get(
        camera_view,
        CAMERA_VIEW_CONFIGS.get(camera_view, CAMERA_VIEW_CONFIGS['nadir'])
    )
    extrinsics_name = view_cfg['extrinsics']
    
    # Get extrinsics from global_config (loaded from YAML) not hardcoded
    if extrinsics_name == 'BODY_T_CAMDOWN':
        body_t_cam = global_config.get('BODY_T_CAMDOWN', np.eye(4))
    elif extrinsics_name == 'BODY_T_CAMFRONT':
        body_t_cam = global_config.get('BODY_T_CAMFRONT', np.eye(4))
    elif extrinsics_name == 'BODY_T_CAMSIDE':
        body_t_cam = global_config.get('BODY_T_CAMSIDE', np.eye(4))
    else:
        body_t_cam = global_config.get('BODY_T_CAMDOWN', np.eye(4))
    
    R_cam_to_body = body_t_cam[:3, :3]
    
    # Map direction (from VO if available, else from optical flow)
    if t_unit is not None and np.linalg.norm(t_unit) > 1e-6:
        t_norm = t_unit / (np.linalg.norm(t_unit) + 1e-12)
        t_body = R_cam_to_body @ t_norm
    else:
        # Fallback: use median optical flow direction (no VO available)
        # This is the KEY for XY drift reduction when VO fails
        if vio_fe is not None and vio_fe.last_matches is not None:
            pts_prev, pts_cur = vio_fe.last_matches
            if len(pts_prev) > 5:  # Need minimum features
                flows = pts_cur - pts_prev
                median_flow = np.median(flows, axis=0)
                flow_norm = np.linalg.norm(median_flow)
                if flow_norm > 1e-6:
                    # Flow direction in normalized camera coordinates
                    flow_dir = median_flow / flow_norm
                    # Convert to 3D camera direction (assume small angle)
                    t_cam_fallback = np.array([-flow_dir[0], -flow_dir[1], 0.0])
                    t_cam_fallback = t_cam_fallback / (np.linalg.norm(t_cam_fallback) + 1e-9)
                    t_body = R_cam_to_body @ t_cam_fallback
                else:
                    # No motion detected
                    t_body = np.array([0.0, 0.0, 1.0])  # Default: forward
            else:
                t_body = np.array([0.0, 0.0, 1.0])
        else:
            t_body = np.array([0.0, 0.0, 1.0])
    
    # Get rotation from IMU quaternion
    q_imu = imu_rec.q
    Rwb = R_scipy.from_quat(q_imu).as_matrix()
    
    # Scale recovery using AGL
    # v2.9.10.4: Use initial AGL from config to prevent feedback loop
    # Problem: z_drift -> wrong_dynamic_AGL -> wrong_VIO_scale -> more_drift  
    # Solution: Use fixed initial_agl from t=0 (GPS-denied compliant)
    initial_agl = global_config.get('INITIAL_AGL', None)
    
    if initial_agl is not None:
        # Use fixed initial AGL (GPS-denied compliant: from t=0)
        agl = initial_agl
    else:
        # Fallback: dynamic AGL from current state (original behavior)
        lat_now, lon_now = proj_cache.xy_to_latlon(kf.x[0, 0], kf.x[1, 0], lat0, lon0)
        dem_now = dem_reader.sample_m(lat_now, lon_now) if dem_reader.ds else 0.0
        if dem_now is None or np.isnan(dem_now):
            dem_now = 0.0
        
        agl = abs(kf.x[2, 0] - dem_now)
    
    agl = max(1.0, agl)
    
    # Get flow threshold from config (default 0.3px for slow motion)
    vio_config = global_config.get('vio', {})
    min_flow_px = vio_config.get('min_parallax_px', 0.3)
    
    # Optical flow-based scale
    # CRITICAL FIX v2.9.8.4: Use config threshold, not hard-coded 2.0
    focal_px = kb_params.get('mu', 600)
    if dt_img > 1e-4 and avg_flow_px > min_flow_px:
        scale_flow = agl / focal_px
        speed_final = (avg_flow_px / dt_img) * scale_flow
    else:
        speed_final = 0.0
    
    speed_final = min(speed_final, 50.0)  # Clamp to 50 m/s
    
    # Adaptive R scaling for low flow - increase uncertainty when flow is small
    # This prevents low-quality flow from dominating the filter
    flow_quality_scale = 1.0
    if avg_flow_px < 1.0:
        flow_quality_scale = 3.0  # 3x uncertainty for very low flow
    elif avg_flow_px < 2.0:
        flow_quality_scale = 1.5 + (2.0 - avg_flow_px)  # Linear scale 1.5-2.5x
    
    # Compute velocity in world frame with rotational flow compensation
    if avg_flow_px > min_flow_px and vio_fe is not None and vio_fe.last_matches is not None:
        pts_prev, pts_cur = vio_fe.last_matches
        if len(pts_prev) > 0:
            # Get gyro angular velocity in body frame
            omega_body = imu_rec.ang  # [wx, wy, wz] in rad/s
            omega_cam = R_cam_to_body.T @ omega_body  # Transform to camera frame
            
            # Compensate for rotational flow on each feature
            flows_compensated = []
            for i in range(len(pts_prev)):
                pt_prev = pts_prev[i]
                pt_cur = pts_cur[i]
                
                # Measured flow
                flow_measured = pt_cur - pt_prev
                
                # Predict rotational flow: flow_rot = omega_cam × [u, v, f]
                # For small rotations: flow_rot_u ≈ -omega_z * v + omega_y * f
                #                       flow_rot_v ≈  omega_z * u - omega_x * f
                # Simplified for typical nadir camera: mainly omega_z (yaw rotation)
                u_prev, v_prev = pt_prev[0], pt_prev[1]
                flow_rot_u = -omega_cam[2] * v_prev * dt_img
                flow_rot_v = omega_cam[2] * u_prev * dt_img
                flow_rot = np.array([flow_rot_u, flow_rot_v])
                
                # Remove rotational component
                flow_translational = flow_measured - flow_rot
                flows_compensated.append(flow_translational)
            
            # Use compensated flows
            flows_compensated = np.array(flows_compensated)
            median_flow = np.median(flows_compensated, axis=0)
            flow_norm = np.linalg.norm(median_flow)
            
            if flow_norm > 1e-6:
                flow_dir = median_flow / flow_norm
                vel_cam = np.array([-flow_dir[0], -flow_dir[1], 0.0])
                vel_cam = vel_cam / np.linalg.norm(vel_cam + 1e-9)
                vel_body = R_cam_to_body @ vel_cam * speed_final
            else:
                vel_body = t_body * speed_final
        else:
            vel_body = t_body * speed_final
    else:
        vel_body = t_body * speed_final
    
    vel_world = Rwb @ vel_body
    
    # Determine if using VZ only (for nadir cameras)
    # Allow config override for OF-velocity drift reduction
    use_vz_only_default = view_cfg.get('use_vz_only', True)
    vio_config = global_config.get('vio', {})
    use_vz_only = vio_config.get('use_vz_only', use_vz_only_default)
    
    # ESKF velocity update
    num_clones = (kf.x.shape[0] - 19) // 7  # v3.9.7: 19D nominal
    err_dim = 18 + 6 * num_clones  # v3.9.7: 18D core error
    
    extra_scale = max(1e-3, float(r_scale_extra))
    if use_vz_only:
        h_vel = np.zeros((1, err_dim), dtype=float)
        h_vel[0, 5] = 1.0
        vel_meas = np.array([[vel_world[2]]])
        # Apply flow quality scaling to uncertainty
        r_mat = np.array([[(sigma_vo * view_cfg.get('sigma_scale_z', 2.0) * flow_quality_scale * extra_scale)**2]])
        dof = 1
    else:
        h_vel = np.zeros((3, err_dim), dtype=float)
        h_vel[0, 3] = 1.0
        h_vel[1, 4] = 1.0
        h_vel[2, 5] = 1.0
        vel_meas = vel_world.reshape(-1, 1)
        scale_xy = view_cfg.get('sigma_scale_xy', 1.0)
        scale_z = view_cfg.get('sigma_scale_z', 2.0)
        # Apply flow quality scaling to uncertainty (more for XY since direction is less reliable)
        r_mat = np.diag([
            (sigma_vo * scale_xy * flow_quality_scale * 1.5 * extra_scale)**2,  # Extra 1.5x for XY
            (sigma_vo * scale_xy * flow_quality_scale * 1.5 * extra_scale)**2,
            (sigma_vo * scale_z * flow_quality_scale * extra_scale)**2
        ])
        dof = 3
    
    def h_fun(x, h=h_vel):
        return h
    
    def hx_fun(x, h=h_vel):
        if use_vz_only:
            return x[5:6].reshape(1, 1)
        else:
            return x[3:6].reshape(3, 1)
    
    # Apply update with chi-square gating
    if use_vio_velocity:
        # VALIDATION: Check P matrix for numerical issues before innovation computation
        # This prevents divide-by-zero and overflow from corrupted covariance
        has_invalid_p = np.any(np.isinf(kf.P)) or np.any(np.isnan(kf.P))
        if has_invalid_p:
            # P corrupted - cannot compute valid innovation, skip update
            print(f"[VIO] Velocity REJECTED: P matrix contains inf/nan")
            _set_adaptive_info(False, dof, None, None, extra_scale)
            return False
        
        # Clamp large P values to prevent overflow in matmul
        P_max = np.max(np.abs(kf.P))
        if P_max > 1e10:
            # Scale P down to prevent numerical explosion
            scale_factor = 1e8 / P_max
            kf.P = kf.P * scale_factor
            print(f"[VIO] Velocity REJECTED: P matrix overflow clamped (max={P_max:.2e})")
            _set_adaptive_info(False, dof, None, None, extra_scale)
            return False
        
        from .numerical_checks import assert_finite
        
        # [TRIPWIRE] Validate inputs before S matrix computation
        if not assert_finite("vel_h_vel", h_vel, extra_info={"h_vel_norm": np.linalg.norm(h_vel)}):
            print(f"[VIO] Velocity REJECTED: h_vel contains inf/nan")
            _set_adaptive_info(False, dof, None, None, extra_scale)
            return False
        
        if not assert_finite("vel_kf_P", kf.P, extra_info={
            "P_max": np.max(np.abs(kf.P)),
            "P_trace": np.trace(kf.P)
        }):
            print(f"[VIO] Velocity REJECTED: kf.P contains inf/nan")
            _set_adaptive_info(False, dof, None, None, extra_scale)
            return False
        
        # Compute innovation for gating with overflow protection
        predicted_vel = hx_fun(kf.x)
        innovation = vel_meas - predicted_vel
        
        # Suppress numpy warnings - we handle explicitly with tripwires
        with np.errstate(all='ignore'):
            try:
                s_mat = h_vel @ kf.P @ h_vel.T + r_mat
            except Exception as e:
                print(f"[VIO] Velocity REJECTED: s_mat computation failed: {e}")
                _set_adaptive_info(False, dof, None, None, extra_scale)
                return False
        
        # [TRIPWIRE] Check s_mat validity after computation
        if not assert_finite("vel_s_mat", s_mat, extra_info={
            "s_mat_max": np.max(np.abs(s_mat)),
            "r_mat_max": np.max(np.abs(r_mat)),
            "h_norm": np.linalg.norm(h_vel),
            "P_max": np.max(np.abs(kf.P))
        }):
            print(f"[VIO] Velocity REJECTED: s_mat contains inf/nan after computation")
            _set_adaptive_info(False, dof, None, None, extra_scale)
            return False
        
        # Chi-square test
        try:
            s_inv = safe_matrix_inverse(s_mat, damping=1e-9, method='cholesky')
            m2 = innovation.T @ s_inv @ innovation
            chi2_value = float(m2)
            mahal_dist = np.sqrt(chi2_value)
        except Exception:
            chi2_value = float('inf')
            mahal_dist = float('nan')
        
        # Chi-square thresholds
        # v2.9.9.7: RELAXED from 95% to 99.5% to accept more valid updates
        # Analysis: Filter overconfident (11.8σ vel error) → rejects valid VIO_VEL → divergence
        chi2_threshold = 6.63 if use_vz_only else 11.34  # 1-DOF: 99%, 3-DOF: 99.5% (was 3.84/7.81)
        chi2_threshold *= max(1e-3, float(chi2_scale))
        
        if chi2_value < chi2_threshold:
            # Accept update
            kf.update(
                z=vel_meas, HJacobian=h_fun, Hx=hx_fun, R=r_mat,
                update_type="VIO_VEL", timestamp=t
            )
            vo_mode = "VO" if (t_unit is not None and np.linalg.norm(t_unit) > 1e-6) else "OF-fallback"
            print(f"[VIO] Velocity update: speed={speed_final:.2f}m/s, vz_only={use_vz_only}, "
                  f"flow={avg_flow_px:.1f}px, R_scale={flow_quality_scale:.1f}x, mode={vo_mode}, chi2={chi2_value:.2f}")
            accepted = True
        else:
            # Reject outlier
            print(f"[VIO] Velocity REJECTED: chi2={chi2_value:.2f} > {chi2_threshold:.1f}, "
                  f"flow={avg_flow_px:.1f}px, speed={speed_final:.2f}m/s")
            accepted = False

        _set_adaptive_info(accepted, dof, chi2_value, chi2_threshold, extra_scale)
        
        # Log to debug_residuals.csv (v2.9.9.8: with NEES)
        if save_debug and residual_csv:
            log_measurement_update(
                residual_csv, t, vio_frame, 'VIO_VEL',
                innovation=innovation.flatten(),
                mahalanobis_dist=mahal_dist,
                chi2_threshold=chi2_threshold,
                accepted=accepted,
                s_matrix=s_mat,
                p_prior=getattr(kf, 'P_prior', kf.P),
                state_error=state_error,  # Ground truth error for NEES
                state_cov=state_cov       # Velocity covariance for NEES
            )
        
        return accepted
    
    _set_adaptive_info(False, dof, None, None, extra_scale)
    return False


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
    num_clones = (kf.x.shape[0] - 19) // 7  # v3.9.7: 19D nominal
    err_dim = 18 + 6 * num_clones  # v3.9.7: 18D core error
    
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
        S_alt_inv = safe_matrix_inverse(S_alt, damping=1e-9, method='cholesky')
        chi2_alt = float(innovation.T @ S_alt_inv @ innovation)
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
        R=R_altitude,
        update_type="PLANE_ALT",
        timestamp=0.0
    )
    
    return True, ""
