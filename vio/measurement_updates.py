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

import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from typing import Optional, Tuple, Callable, Dict, Any, TYPE_CHECKING
import math

# Import shared math utilities (avoid duplication)
from .math_utils import quaternion_to_yaw, mahalanobis_squared, safe_matrix_inverse
from .output_utils import log_measurement_update

if TYPE_CHECKING:
    from .policy.types import SensorPolicyDecision


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

_VIO_VEL_LOG_EVERY_N = max(1, int(os.getenv("VIO_VEL_LOG_EVERY_N", "25")))
_VIO_VEL_LOG_COUNTER = 0


def _log_vio_vel_update(message: str):
    """Throttle very chatty VIO velocity INFO logs to reduce runtime overhead."""
    global _VIO_VEL_LOG_COUNTER
    _VIO_VEL_LOG_COUNTER += 1
    if _VIO_VEL_LOG_COUNTER <= 3 or _VIO_VEL_LOG_COUNTER % _VIO_VEL_LOG_EVERY_N == 0:
        print(message)


def apply_magnetometer_update(kf,
                              mag_calibrated: np.ndarray,
                              mag_declination: float,
                              use_raw_heading: bool,
                              sigma_mag_yaw: float,
                              global_config: Optional[dict] = None,
                              current_phase: int = 2,
                              health_state: str = "HEALTHY",
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
                              adaptive_info: Optional[Dict[str, Any]] = None,
                              policy_decision: Optional["SensorPolicyDecision"] = None) -> Tuple[bool, str]:
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
    if policy_decision is not None:
        mode = str(getattr(policy_decision, "mode", "APPLY")).upper()
        if mode in ("HOLD", "SKIP"):
            if adaptive_info is not None:
                adaptive_info.clear()
                adaptive_info.update({
                    "sensor": "MAG",
                    "accepted": False,
                    "dof": 1,
                    "nis_norm": np.nan,
                    "chi2": np.nan,
                    "threshold": np.nan,
                    "r_scale_used": float(extra_scale),
                    "reason_code": f"policy_mode_{mode.lower()}",
                })
            return False, f"policy_mode_{mode.lower()}"
        try:
            health_state = str(policy_decision.extra_str("health_state", str(health_state))).upper()
        except Exception:
            pass

    def _set_adaptive_info(accepted: bool,
                           nis_norm: Optional[float],
                           chi2: Optional[float],
                           threshold: Optional[float],
                           r_scale_used: float,
                           reason_code: str = ""):
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
            "reason_code": str(reason_code) if reason_code else ("normal_accept" if accepted else "hard_reject"),
        })
    from .magnetometer import compute_yaw_from_mag
    global_config = global_config or {}
    
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
    health_key = str(health_state).upper()
    warning_r_mult = float(global_config.get("MAG_WARNING_R_MULT", 4.0))
    degraded_r_mult = float(global_config.get("MAG_DEGRADED_R_MULT", 8.0))
    warning_max_dyaw_deg = float(global_config.get("MAG_WARNING_MAX_DYAW_DEG", 1.5))
    degraded_max_dyaw_deg = float(global_config.get("MAG_DEGRADED_MAX_DYAW_DEG", 1.0))
    if policy_decision is not None:
        warning_r_mult = float(policy_decision.extra("warning_r_mult", warning_r_mult))
        degraded_r_mult = float(policy_decision.extra("degraded_r_mult", degraded_r_mult))
        warning_max_dyaw_deg = float(policy_decision.extra("warning_max_dyaw_deg", warning_max_dyaw_deg))
        degraded_max_dyaw_deg = float(policy_decision.extra("degraded_max_dyaw_deg", degraded_max_dyaw_deg))
    if health_key == "WARNING":
        sigma_mag_yaw_scaled *= float(warning_r_mult)
    elif health_key == "DEGRADED":
        sigma_mag_yaw_scaled *= float(degraded_r_mult)
    
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
    r_yaw = np.array([[sigma_mag_yaw_scaled**2]], dtype=float)
    
    # Phase-based Kalman gain tuning (v3.4.0)
    # SPINUP (0): FULL CORRECTION (K=1.0) to completely prevent drift
    # Problem: Gyro drift ~0.44°/s accumulates between mag updates (~50ms apart)
    #          Even K=0.90 leaves 10% residual → exponential drift buildup
    # Solution: K=1.0 (100% correction) in SPINUP - trust magnetometer completely
    if current_phase == 0:  # SPINUP phase - FULL CORRECTION
        K_MIN = 0.999  # Use 0.999 instead of 1.0 to avoid div by zero!
        MAX_YAW_CORRECTION = np.radians(120.0)  # Keep broad, but avoid hard snaps
    elif current_phase == 1:  # EARLY phase - moderate correction
        K_MIN = 0.70  # Higher trust in mag during convergence
        MAX_YAW_CORRECTION = np.radians(55.0)
    else:  # NORMAL phase - gentler correction
        K_MIN = 0.40  # Standard tracking
        MAX_YAW_CORRECTION = np.radians(35.0)

    health_corr_mult = {
        "HEALTHY": 1.00,
        "WARNING": 0.80,
        "DEGRADED": 0.65,
        "RECOVERY": 0.90,
    }.get(health_key, 1.00)
    MAX_YAW_CORRECTION *= float(health_corr_mult)
    if health_key == "WARNING":
        MAX_YAW_CORRECTION = min(
            MAX_YAW_CORRECTION,
            np.radians(float(warning_max_dyaw_deg))
        )
    elif health_key == "DEGRADED":
        MAX_YAW_CORRECTION = min(
            MAX_YAW_CORRECTION,
            np.radians(float(degraded_max_dyaw_deg))
        )
    
    # Compute current Kalman gain
    P_yaw = float(kf.P[theta_cov_idx, theta_cov_idx])
    if not np.isfinite(P_yaw) or P_yaw <= 0.0:
        _set_adaptive_info(False, None, None, None, extra_scale)
        return False, "Invalid P_yaw"
    S_yaw = float(P_yaw + sigma_mag_yaw_scaled**2)  # BUG FIX: Use scaled sigma!
    if not np.isfinite(S_yaw) or S_yaw <= 1e-12:
        _set_adaptive_info(False, None, None, None, extra_scale)
        return False, "Invalid S_yaw"
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
        r_yaw = np.array([[max(R_inflated, sigma_mag_yaw_scaled**2)]], dtype=float)

    if not np.all(np.isfinite(r_yaw)) or float(r_yaw[0, 0]) <= 0.0:
        _set_adaptive_info(False, None, None, None, extra_scale)
        return False, "Invalid MAG covariance"
    
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
                               soft_fail_enable: bool = False,
                               soft_fail_r_cap: float = 8.0,
                               soft_fail_hard_reject_factor: float = 3.0,
                               soft_fail_power: float = 1.0,
                               phase: int = 2,
                               health_state: str = "HEALTHY",
                               adaptive_info: Optional[Dict[str, Any]] = None,
                               policy_decision: Optional["SensorPolicyDecision"] = None) -> bool:
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
                           r_scale_used: float,
                           reason_code: str = "",
                           hard_threshold: Optional[float] = None,
                           attempted: int = 1):
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
            "attempted": int(max(0, attempted)),
            "reason_code": str(reason_code),
            "hard_threshold": float(hard_threshold) if hard_threshold is not None and np.isfinite(hard_threshold) else np.nan,
        })

    if policy_decision is not None:
        mode = str(getattr(policy_decision, "mode", "APPLY")).upper()
        if mode in ("HOLD", "SKIP"):
            _set_adaptive_info(
                False,
                2,
                None,
                None,
                max(1e-3, float(r_scale_extra)),
                reason_code=f"policy_mode_{mode.lower()}",
                attempted=0,
            )
            return False
        try:
            phase = int(round(float(policy_decision.extra("phase", float(phase)))))
        except Exception:
            pass
        try:
            health_state = str(policy_decision.extra_str("health_state", str(health_state))).upper()
        except Exception:
            pass

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
    
    # YAML extrinsics follow T_BC (body -> camera) from calibration docs.
    # For mapping camera vectors into body frame we must use R_CB = R_BC^T.
    R_body_to_cam = body_t_cam[:3, :3]
    R_cam_to_body = R_body_to_cam.T
    
    vio_config = global_config.get('vio', {})
    nadir_prefer_flow_direction = bool(vio_config.get('nadir_prefer_flow_direction', True))
    nadir_enforce_xy_motion = bool(vio_config.get('nadir_enforce_xy_motion', True))

    def _flow_direction_body() -> Optional[np.ndarray]:
        if vio_fe is None or vio_fe.last_matches is None:
            return None
        pts_prev, pts_cur = vio_fe.last_matches
        if pts_prev is None or pts_cur is None or len(pts_prev) <= 5:
            return None
        flows = pts_cur - pts_prev
        if flows.size == 0:
            return None
        median_flow = np.median(flows, axis=0)
        flow_norm = np.linalg.norm(median_flow)
        if flow_norm <= 1e-6:
            return None
        flow_dir = median_flow / flow_norm
        # For nadir, image flow direction is a better cue than essential-matrix t_unit
        # under planar-scene degeneracy.
        t_cam = np.array([-flow_dir[0], -flow_dir[1], 0.0], dtype=float)
        t_cam /= max(1e-9, np.linalg.norm(t_cam))
        return R_cam_to_body @ t_cam

    # Map direction.
    # For nadir camera, prefer optical-flow direction to reduce E-matrix translation ambiguity.
    t_body = None
    if camera_view == "nadir" and nadir_prefer_flow_direction:
        t_body = _flow_direction_body()

    if t_body is None:
        if t_unit is not None and np.linalg.norm(t_unit) > 1e-6:
            t_norm = t_unit / (np.linalg.norm(t_unit) + 1e-12)
            t_body = R_cam_to_body @ t_norm
        else:
            t_body = _flow_direction_body()

    if t_body is None:
        # Fall back to current horizontal velocity direction, then body X.
        v_curr = np.array(kf.x[3:6, 0], dtype=float)
        if np.linalg.norm(v_curr[:2]) > 1e-6:
            t_body = np.array([v_curr[0], v_curr[1], 0.0], dtype=float)
        else:
            t_body = np.array([1.0, 0.0, 0.0], dtype=float)

    if camera_view == "nadir" and nadir_enforce_xy_motion:
        t_body = np.array(t_body, dtype=float)
        t_body[2] = 0.0
        if np.linalg.norm(t_body[:2]) <= 1e-6:
            t_body = np.array([1.0, 0.0, 0.0], dtype=float)
    t_body = t_body / max(1e-9, np.linalg.norm(t_body))
    
    # Get rotation from IMU quaternion
    q_imu = imu_rec.q
    Rwb = R_scipy.from_quat(q_imu).as_matrix()
    
    # Scale recovery using AGL
    # Default behavior is HYBRID dynamic AGL so velocity scale can follow takeoff/cruise/landing.
    # Fixed-only mode can be re-enabled for debugging via config:
    #   vio.flow_agl_mode: fixed
    initial_agl = global_config.get('INITIAL_AGL', None)
    flow_agl_mode = str(global_config.get('VIO_FLOW_AGL_MODE', 'hybrid')).lower()
    flow_min_agl = float(global_config.get('VIO_FLOW_MIN_AGL', 1.0))
    flow_max_agl = float(global_config.get('VIO_FLOW_MAX_AGL', 500.0))

    lat_now, lon_now = proj_cache.xy_to_latlon(kf.x[0, 0], kf.x[1, 0], lat0, lon0)
    dem_now = dem_reader.sample_m(lat_now, lon_now) if dem_reader.ds else 0.0
    if dem_now is None or np.isnan(dem_now):
        dem_now = 0.0
    agl_dynamic = abs(float(kf.x[2, 0]) - float(dem_now))

    if flow_agl_mode == 'fixed' and initial_agl is not None:
        agl = float(initial_agl)
    elif flow_agl_mode == 'dynamic':
        agl = float(agl_dynamic)
    else:
        # hybrid: follow dynamic altitude but never collapse below a fraction
        # of initial AGL, preventing extreme scale drops from local DEM noise.
        if initial_agl is not None:
            agl_floor = max(flow_min_agl, 0.5 * float(initial_agl))
            agl = max(float(agl_dynamic), float(agl_floor))
        else:
            agl = float(agl_dynamic)

    agl = float(np.clip(agl, flow_min_agl, max(flow_min_agl, flow_max_agl)))
    
    # Get flow threshold from config (default 0.3px for slow motion)
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

    if camera_view == "nadir" and nadir_enforce_xy_motion:
        # Keep nadir VO velocity as horizontal-only cue; vertical is handled by DEM/MSL.
        vxy = np.linalg.norm(vel_body[:2])
        if vxy > 1e-6:
            scale_xy = speed_final / vxy
            vel_body[0] *= scale_xy
            vel_body[1] *= scale_xy
            vel_body[2] = 0.0
        else:
            vel_body = np.array([speed_final, 0.0, 0.0], dtype=float)
    
    vel_world = Rwb @ vel_body

    if camera_view == "nadir" and nadir_enforce_xy_motion:
        # Avoid injecting synthetic vertical velocity from nadir XY flow.
        vel_world[2] = float(kf.x[5, 0])
    
    # Determine velocity measurement mode for VIO velocity update.
    # - vz_only: legacy 1D vertical update
    # - xy_only_nadir: nadir-only horizontal update (2DOF) to avoid Z coupling
    # - full_3d: standard 3DOF update
    use_vz_only_default = view_cfg.get('use_vz_only', True)
    vio_config = global_config.get('vio', {})
    use_vz_only = vio_config.get('use_vz_only', use_vz_only_default)
    xy_only_nadir = bool(global_config.get('VIO_NADIR_XY_ONLY_VELOCITY', False))
    if policy_decision is not None:
        try:
            xy_only_nadir = bool(float(policy_decision.extra("xy_only_nadir", float(xy_only_nadir))) >= 0.5)
        except Exception:
            pass
    use_xy_only = bool(camera_view == "nadir" and xy_only_nadir and not bool(use_vz_only))
    speed_state_m_s = float(np.linalg.norm(np.asarray(kf.x[3:6, 0], dtype=float)))

    def _parse_float_list(value, default_list):
        if not isinstance(value, (list, tuple)):
            value = default_list
        out = []
        for v in value:
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv):
                out.append(fv)
        if len(out) == 0:
            out = [float(v) for v in default_list]
        return out

    speed_bp = _parse_float_list(
        global_config.get("VIO_VEL_SPEED_R_INFLATE_BREAKPOINTS_M_S", [25.0, 40.0, 55.0]),
        [25.0, 40.0, 55.0],
    )
    speed_vals = _parse_float_list(
        global_config.get("VIO_VEL_SPEED_R_INFLATE_VALUES", [1.5, 2.5, 4.0]),
        [1.5, 2.5, 4.0],
    )
    n_pairs = min(len(speed_bp), len(speed_vals))
    speed_pairs = sorted(
        [(float(speed_bp[i]), max(1.0, float(speed_vals[i]))) for i in range(n_pairs)],
        key=lambda x: x[0],
    )
    speed_r_inflate = 1.0
    for bp, val in speed_pairs:
        if np.isfinite(speed_state_m_s) and speed_state_m_s > bp:
            speed_r_inflate = float(max(speed_r_inflate, val))
    min_flow_high_speed = float(global_config.get("VIO_VEL_MIN_FLOW_PX_HIGH_SPEED", 0.8))
    if policy_decision is not None:
        min_flow_high_speed = float(policy_decision.extra("min_flow_px_high_speed", min_flow_high_speed))
    high_speed_flow_bp = float(speed_pairs[0][0]) if len(speed_pairs) > 0 else 25.0
    cfg_high_speed_bp = float(global_config.get("VIO_VEL_HIGH_SPEED_BP_M_S", high_speed_flow_bp))
    if np.isfinite(cfg_high_speed_bp) and cfg_high_speed_bp > 0.0:
        high_speed_flow_bp = cfg_high_speed_bp
    max_delta_v_xy_base = float(global_config.get("VIO_VEL_MAX_DELTA_V_XY_PER_UPDATE_M_S", 2.0))
    if policy_decision is not None:
        max_delta_v_xy_base = float(policy_decision.extra("max_delta_v_xy_m_s", max_delta_v_xy_base))
    max_delta_v_xy_high_speed = float(
        global_config.get(
            "VIO_VEL_MAX_DELTA_V_XY_HIGH_SPEED_M_S",
            max_delta_v_xy_base,
        )
    )
    delta_v_soft_enable = bool(global_config.get("VIO_VEL_DELTA_V_SOFT_ENABLE", True))
    delta_v_soft_factor = float(global_config.get("VIO_VEL_DELTA_V_SOFT_FACTOR", 2.0))
    delta_v_hard_factor = float(global_config.get("VIO_VEL_DELTA_V_HARD_FACTOR", 3.0))
    delta_v_soft_r_cap = float(global_config.get("VIO_VEL_DELTA_V_SOFT_MAX_R_MULT", 6.0))
    delta_v_clamp_enable = bool(global_config.get("VIO_VEL_DELTA_V_CLAMP_ENABLE", True))
    delta_v_clamp_max_ratio = float(global_config.get("VIO_VEL_DELTA_V_CLAMP_MAX_RATIO", 6.0))
    delta_v_clamp_r_mult = float(global_config.get("VIO_VEL_DELTA_V_CLAMP_R_MULT", 3.5))
    if policy_decision is not None:
        delta_v_clamp_max_ratio = float(policy_decision.extra("delta_v_clamp_max_ratio", delta_v_clamp_max_ratio))
        delta_v_clamp_r_mult = float(policy_decision.extra("delta_v_clamp_r_mult", delta_v_clamp_r_mult))
    enforce_nadir_xy_guard = bool(camera_view == "nadir" and not bool(use_vz_only))
    
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
    elif use_xy_only:
        h_vel = np.zeros((2, err_dim), dtype=float)
        h_vel[0, 3] = 1.0
        h_vel[1, 4] = 1.0
        vel_meas = np.array([[vel_world[0]], [vel_world[1]]], dtype=float)
        scale_xy = view_cfg.get('sigma_scale_xy', 1.0)
        # Slightly conservative XY-only R to preserve fail-soft behavior.
        r_speed = float(max(1.0, speed_r_inflate))
        r_mat = np.diag([
            (sigma_vo * scale_xy * flow_quality_scale * 1.4 * extra_scale * r_speed) ** 2,
            (sigma_vo * scale_xy * flow_quality_scale * 1.4 * extra_scale * r_speed) ** 2,
        ])
        dof = 2
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
        elif use_xy_only:
            return x[3:5].reshape(2, 1)
        else:
            return x[3:6].reshape(3, 1)
    
    # Apply update with chi-square gating
    if use_vio_velocity:
        # VALIDATION: Check P matrix for numerical issues before innovation computation
        # This prevents divide-by-zero and overflow from corrupted covariance
        has_invalid_p = np.any(np.isinf(kf.P)) or np.any(np.isnan(kf.P))
        if has_invalid_p:
            # P corrupted - cannot compute valid innovation, skip update
            _log_vio_vel_update("[VIO] Velocity REJECTED: P matrix contains inf/nan")
            _set_adaptive_info(False, dof, None, None, extra_scale, reason_code="hard_reject")
            return False
        if (
            enforce_nadir_xy_guard
            and np.isfinite(speed_state_m_s)
            and speed_state_m_s > high_speed_flow_bp
            and float(avg_flow_px) < min_flow_high_speed
        ):
            _log_vio_vel_update(
                f"[VIO] Velocity REJECTED: low flow at high speed (flow={avg_flow_px:.2f}px, "
                f"speed={speed_state_m_s:.2f}m/s, min_flow={min_flow_high_speed:.2f}px, "
                f"guard={'xy_only' if use_xy_only else 'nadir_xy'})"
            )
            _set_adaptive_info(
                False, dof, None, None, extra_scale * max(1.0, speed_r_inflate),
                reason_code="soft_reject_low_flow"
            )
            return False
        
        # Clamp large P values to prevent overflow in matmul
        P_max = np.max(np.abs(kf.P))
        if P_max > 1e10:
            # Scale P down to prevent numerical explosion
            scale_factor = 1e8 / P_max
            kf.P = kf.P * scale_factor
            _log_vio_vel_update(f"[VIO] Velocity REJECTED: P matrix overflow clamped (max={P_max:.2e})")
            _set_adaptive_info(False, dof, None, None, extra_scale, reason_code="hard_reject")
            return False
        
        from .numerical_checks import assert_finite
        
        # [TRIPWIRE] Validate inputs before S matrix computation
        if not assert_finite("vel_h_vel", h_vel, extra_info={"h_vel_norm": np.linalg.norm(h_vel)}):
            _log_vio_vel_update("[VIO] Velocity REJECTED: h_vel contains inf/nan")
            _set_adaptive_info(False, dof, None, None, extra_scale, reason_code="hard_reject")
            return False
        
        if not assert_finite("vel_kf_P", kf.P, extra_info={
            "P_max": np.max(np.abs(kf.P)),
            "P_trace": np.trace(kf.P)
        }):
            _log_vio_vel_update("[VIO] Velocity REJECTED: kf.P contains inf/nan")
            _set_adaptive_info(False, dof, None, None, extra_scale, reason_code="hard_reject")
            return False
        
        # Compute innovation for gating with overflow protection
        predicted_vel = hx_fun(kf.x)
        innovation = vel_meas - predicted_vel
        delta_v_soft_mult = 1.0
        delta_v_clamped = False
        if use_xy_only or enforce_nadir_xy_guard:
            max_delta_v_xy = float(max_delta_v_xy_base)
            if np.isfinite(speed_state_m_s) and speed_state_m_s > high_speed_flow_bp:
                max_delta_v_xy = float(
                    min(max_delta_v_xy, max(1e-3, float(max_delta_v_xy_high_speed)))
                )
            delta_v_xy = float(np.linalg.norm(np.asarray(innovation[:2]).reshape(-1)))
            if np.isfinite(max_delta_v_xy) and max_delta_v_xy > 0.0 and delta_v_xy > max_delta_v_xy:
                ratio_delta = float(delta_v_xy / max(max_delta_v_xy, 1e-9))
                allow_delta_soft = (
                    bool(delta_v_soft_enable)
                    and bool(soft_fail_enable)
                    and np.isfinite(ratio_delta)
                    and ratio_delta <= max(1.0, float(delta_v_hard_factor))
                    and float(avg_flow_px) >= (0.5 * max(0.1, min_flow_high_speed))
                )
                if allow_delta_soft:
                    soft_pow = max(0.8, min(2.0, float(delta_v_soft_factor)))
                    delta_v_soft_mult = float(
                        np.clip(ratio_delta ** soft_pow, 1.0, max(1.0, float(delta_v_soft_r_cap)))
                    )
                    r_mat = r_mat * delta_v_soft_mult
                    _log_vio_vel_update(
                        f"[VIO] Velocity delta-v soft mode: |Δv_xy|={delta_v_xy:.2f}m/s, "
                        f"cap={max_delta_v_xy:.2f}m/s, R*={delta_v_soft_mult:.2f}x "
                        f"({('xy_only' if use_xy_only else 'nadir_xy')})"
                    )
                else:
                    allow_delta_clamp = (
                        bool(delta_v_clamp_enable)
                        and np.isfinite(ratio_delta)
                        and ratio_delta <= max(1.0, float(delta_v_clamp_max_ratio))
                        and float(avg_flow_px) >= (0.4 * max(0.1, min_flow_high_speed))
                    )
                    if allow_delta_clamp:
                        clamp_scale = float(max_delta_v_xy / max(delta_v_xy, 1e-9))
                        innovation[:2] *= clamp_scale
                        vel_meas[:2, 0] = predicted_vel[:2, 0] + innovation[:2, 0]
                        clamp_mult = float(
                            np.clip(
                                max(1.0, float(delta_v_clamp_r_mult)) * max(1.0, ratio_delta),
                                1.0,
                                max(1.0, float(delta_v_soft_r_cap) * 1.8),
                            )
                        )
                        r_mat = r_mat * clamp_mult
                        delta_v_soft_mult = max(float(delta_v_soft_mult), clamp_mult)
                        delta_v_clamped = True
                        _log_vio_vel_update(
                            f"[VIO] Velocity delta-v CLAMP: |Δv_xy|={delta_v_xy:.2f}m/s -> "
                            f"{max_delta_v_xy:.2f}m/s, clamp={clamp_scale:.3f}, R*={clamp_mult:.2f}x "
                            f"({('xy_only' if use_xy_only else 'nadir_xy')})"
                        )
                    else:
                        _log_vio_vel_update(
                            f"[VIO] Velocity REJECTED: delta-v cap (|Δv_xy|={delta_v_xy:.2f}m/s > "
                            f"{max_delta_v_xy:.2f}m/s, guard={'xy_only' if use_xy_only else 'nadir_xy'})"
                        )
                        _set_adaptive_info(
                            False,
                            dof,
                            None,
                            None,
                            extra_scale * max(1.0, speed_r_inflate),
                            reason_code="soft_reject_delta_v_cap",
                        )
                        if save_debug and residual_csv:
                            try:
                                s_cap = h_vel @ kf.P @ h_vel.T + r_mat
                                log_measurement_update(
                                    residual_csv, t, vio_frame, 'VIO_VEL',
                                    innovation=innovation.flatten(),
                                    mahalanobis_dist=np.nan,
                                    chi2_threshold=np.nan,
                                    accepted=False,
                                    s_matrix=s_cap,
                                    p_prior=getattr(kf, 'P_prior', kf.P),
                                    state_error=state_error,
                                    state_cov=state_cov,
                                )
                            except Exception:
                                pass
                        return False
        
        # Suppress numpy warnings - we handle explicitly with tripwires
        with np.errstate(all='ignore'):
            try:
                s_mat = h_vel @ kf.P @ h_vel.T + r_mat
            except Exception as e:
                _log_vio_vel_update(f"[VIO] Velocity REJECTED: s_mat computation failed: {e}")
                _set_adaptive_info(False, dof, None, None, extra_scale, reason_code="hard_reject")
                return False
        
        # [TRIPWIRE] Check s_mat validity after computation
        if not assert_finite("vel_s_mat", s_mat, extra_info={
            "s_mat_max": np.max(np.abs(s_mat)),
            "r_mat_max": np.max(np.abs(r_mat)),
            "h_norm": np.linalg.norm(h_vel),
            "P_max": np.max(np.abs(kf.P))
        }):
            _log_vio_vel_update("[VIO] Velocity REJECTED: s_mat contains inf/nan after computation")
            _set_adaptive_info(False, dof, None, None, extra_scale, reason_code="hard_reject")
            return False

        def _compute_chi2(s_matrix: np.ndarray) -> Tuple[float, float]:
            try:
                s_inv = safe_matrix_inverse(s_matrix, damping=1e-9, method='cholesky')
                m2 = innovation.T @ s_inv @ innovation
                chi2_val = float(np.squeeze(m2))
                return chi2_val, float(np.sqrt(max(0.0, chi2_val)))
            except Exception:
                return float("inf"), float("nan")

        chi2_value, mahal_dist = _compute_chi2(s_mat)
        
        # Chi-square thresholds
        # v2.9.9.7: RELAXED from 95% to 99.5% to accept more valid updates
        # Analysis: Filter overconfident (11.8σ vel error) → rejects valid VIO_VEL → divergence
        if use_vz_only:
            chi2_threshold = 6.63  # 1-DOF (99%)
        elif use_xy_only:
            chi2_threshold = 9.21 * float(global_config.get("VIO_VEL_XY_ONLY_CHI2_SCALE", 1.10))  # 2-DOF
        else:
            chi2_threshold = 11.34  # 3-DOF (99.5%)
        chi2_threshold *= max(1e-3, float(chi2_scale))
        phase_key = str(max(0, min(2, int(phase))))
        health_key = str(health_state).upper()
        phase_hard_map = {
            "0": float(global_config.get("VIO_VEL_PHASE_HARD_FACTOR_0", 1.35)),
            "1": float(global_config.get("VIO_VEL_PHASE_HARD_FACTOR_1", 1.18)),
            "2": float(global_config.get("VIO_VEL_PHASE_HARD_FACTOR_2", 1.00)),
        }
        health_hard_map = {
            "HEALTHY": float(global_config.get("VIO_VEL_HEALTH_HARD_FACTOR_HEALTHY", 1.00)),
            "WARNING": float(global_config.get("VIO_VEL_HEALTH_HARD_FACTOR_WARNING", 1.20)),
            "DEGRADED": float(global_config.get("VIO_VEL_HEALTH_HARD_FACTOR_DEGRADED", 1.45)),
            "RECOVERY": float(global_config.get("VIO_VEL_HEALTH_HARD_FACTOR_RECOVERY", 1.10)),
        }
        phase_hard_mult = float(phase_hard_map.get(phase_key, 1.0))
        health_hard_mult = float(health_hard_map.get(health_key, 1.0))
        hard_threshold = (
            chi2_threshold
            * max(1.0, float(soft_fail_hard_reject_factor))
            * max(1.0, phase_hard_mult)
            * max(1.0, health_hard_mult)
        )
        accepted = False
        reason_code = "hard_reject"
        s_used = s_mat
        r_used = r_mat
        r_scale_used = extra_scale
        if use_xy_only:
            r_scale_used *= max(1.0, speed_r_inflate)
        r_scale_used *= max(1.0, float(delta_v_soft_mult))
        chi2_used = chi2_value
        mahal_used = mahal_dist

        if chi2_value < chi2_threshold:
            # Accept update
            kf.update(
                z=vel_meas, HJacobian=h_fun, Hx=hx_fun, R=r_used,
                update_type="VIO_VEL", timestamp=t
            )
            vo_mode = "VO" if (t_unit is not None and np.linalg.norm(t_unit) > 1e-6) else "OF-fallback"
            _log_vio_vel_update(
                f"[VIO] Velocity update: speed={speed_final:.2f}m/s, vz_only={use_vz_only}, "
                f"xy_only={use_xy_only}, "
                f"flow={avg_flow_px:.1f}px, R_scale={flow_quality_scale:.1f}x, mode={vo_mode}, chi2={chi2_value:.2f}"
            )
            accepted = True
            if bool(delta_v_clamped):
                reason_code = "soft_accept_clamped_delta_v_cap"
            elif float(delta_v_soft_mult) > 1.0:
                reason_code = "soft_accept_delta_v_cap"
            else:
                reason_code = "normal_accept"
        elif bool(soft_fail_enable):
            soft_ratio = max(1.0, float(chi2_value) / max(chi2_threshold, 1e-9))
            phase_cap_map = {
                "0": float(global_config.get("VIO_VEL_PHASE_SOFT_CAP_FACTOR_0", 1.20)),
                "1": float(global_config.get("VIO_VEL_PHASE_SOFT_CAP_FACTOR_1", 1.10)),
                "2": float(global_config.get("VIO_VEL_PHASE_SOFT_CAP_FACTOR_2", 1.00)),
            }
            health_cap_map = {
                "HEALTHY": float(global_config.get("VIO_VEL_HEALTH_SOFT_CAP_FACTOR_HEALTHY", 1.00)),
                "WARNING": float(global_config.get("VIO_VEL_HEALTH_SOFT_CAP_FACTOR_WARNING", 1.20)),
                "DEGRADED": float(global_config.get("VIO_VEL_HEALTH_SOFT_CAP_FACTOR_DEGRADED", 1.40)),
                "RECOVERY": float(global_config.get("VIO_VEL_HEALTH_SOFT_CAP_FACTOR_RECOVERY", 1.10)),
            }
            soft_r_cap_eff = (
                float(soft_fail_r_cap)
                * float(phase_cap_map.get(phase_key, 1.0))
                * float(health_cap_map.get(health_key, 1.0))
            )
            soft_factor = soft_ratio ** max(0.1, float(soft_fail_power))
            soft_factor = min(max(1.0, float(soft_r_cap_eff)), soft_factor)
            r_used = r_mat * soft_factor
            with np.errstate(all='ignore'):
                s_soft = h_vel @ kf.P @ h_vel.T + r_used
            if np.all(np.isfinite(s_soft)):
                chi2_soft, mahal_soft = _compute_chi2(s_soft)
                chi2_used = chi2_soft
                mahal_used = mahal_soft
                s_used = s_soft
                r_scale_used = extra_scale * soft_factor
                tail_factor = float(global_config.get("VIO_VEL_SOFT_FAIL_TAIL_FACTOR", 1.30))
                tail_gate = hard_threshold * max(1.0, tail_factor)
                flow_tail_guard = float(global_config.get("VIO_VEL_SOFT_FAIL_TAIL_FLOW_GUARD_PX", 0.8))
                allow_tail_soft = (
                    np.isfinite(chi2_soft)
                    and chi2_soft <= tail_gate
                    and float(avg_flow_px) >= flow_tail_guard
                )
                if np.isfinite(chi2_soft) and (chi2_soft <= hard_threshold):
                    kf.update(
                        z=vel_meas, HJacobian=h_fun, Hx=hx_fun, R=r_used,
                        update_type="VIO_VEL", timestamp=t
                    )
                    vo_mode = "VO" if (t_unit is not None and np.linalg.norm(t_unit) > 1e-6) else "OF-fallback"
                    _log_vio_vel_update(
                        f"[VIO] Velocity soft-accept: chi2={chi2_value:.2f}->{chi2_soft:.2f}, "
                        f"th={chi2_threshold:.2f}, hard={hard_threshold:.2f}, infl={soft_factor:.2f}x, mode={vo_mode}"
                    )
                    accepted = True
                    reason_code = "soft_accept"
                elif allow_tail_soft:
                    # Fail-soft tail: keep weakly consistent measurements alive with
                    # heavily inflated R to avoid long hard-reject bursts.
                    kf.update(
                        z=vel_meas, HJacobian=h_fun, Hx=hx_fun, R=r_used,
                        update_type="VIO_VEL", timestamp=t
                    )
                    _log_vio_vel_update(
                        f"[VIO] Velocity soft-tail accept: chi2={chi2_value:.2f}->{chi2_soft:.2f}, "
                        f"hard={hard_threshold:.2f}, tail={tail_gate:.2f}, infl={soft_factor:.2f}x"
                    )
                    accepted = True
                    reason_code = "soft_accept_tail"
                else:
                    _log_vio_vel_update(
                        f"[VIO] Velocity REJECTED: chi2={chi2_soft:.2f} > hard={hard_threshold:.2f} "
                        f"(raw={chi2_value:.2f}, infl={soft_factor:.2f}x, flow={avg_flow_px:.1f}px)"
                    )
            else:
                _log_vio_vel_update(
                    f"[VIO] Velocity REJECTED: soft S invalid (raw_chi2={chi2_value:.2f}, flow={avg_flow_px:.1f}px)"
                )
        else:
            _log_vio_vel_update(
                f"[VIO] Velocity REJECTED: chi2={chi2_value:.2f} > {chi2_threshold:.1f}, "
                f"flow={avg_flow_px:.1f}px, speed={speed_final:.2f}m/s"
            )

        if accepted and bool(delta_v_clamped) and reason_code in ("soft_accept", "soft_accept_tail"):
            reason_code = f"{reason_code}_clamped"

        _set_adaptive_info(
            accepted,
            dof,
            chi2_used,
            chi2_threshold,
            r_scale_used,
            reason_code=reason_code,
            hard_threshold=hard_threshold,
        )
        
        # Log to debug_residuals.csv (v2.9.9.8: with NEES)
        if save_debug and residual_csv:
            log_measurement_update(
                residual_csv, t, vio_frame, 'VIO_VEL',
                innovation=innovation.flatten(),
                mahalanobis_dist=mahal_used,
                chi2_threshold=chi2_threshold,
                accepted=accepted,
                s_matrix=s_used,
                p_prior=getattr(kf, 'P_prior', kf.P),
                state_error=state_error,  # Ground truth error for NEES
                state_cov=state_cov       # Velocity covariance for NEES
            )
        
        return accepted
    
    _set_adaptive_info(False, dof, None, None, extra_scale, reason_code="hard_reject")
    return False
