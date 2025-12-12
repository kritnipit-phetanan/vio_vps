"""
Magnetometer Processing Module for VIO System

This module handles magnetometer calibration, yaw computation, and filtering.
Includes hard-iron/soft-iron correction and GPS-calibrated heading computation.

Based on Bell 412 dataset calibration (MUN-FRL).

Author: VIO project
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R_scipy

# Import shared math utilities (avoid duplication)
from .math_utils import (
    quaternion_to_yaw as _quaternion_to_yaw_shared,
    quaternion_multiply as _quaternion_multiply_shared,
    angle_wrap as _angle_wrap_shared,
    yaw_to_quaternion_update as _yaw_to_quaternion_update_shared
)


# =============================================================================
# Default calibration constants (from Bell 412 dataset)
# These should be overridden by config file values in production
# =============================================================================

# Hard-iron offset: Permanent magnetic fields from electronics/structure
DEFAULT_MAG_HARD_IRON = np.array([0.0, 0.0, 0.0])

# Soft-iron matrix: Correction for magnetic field distortion
DEFAULT_MAG_SOFT_IRON = np.eye(3)

# Magnetometer filter constants
MAG_EMA_ALPHA = 0.3  # EMA smoothing factor (0 = no update, 1 = no smoothing)
MAG_MAX_YAW_RATE = np.radians(30.0)  # Max expected yaw rate [rad/s]
MAG_GYRO_CONSISTENCY_THRESHOLD = np.radians(10.0)  # Threshold for gyro vs mag consistency
MAG_CONSISTENCY_R_INFLATE = 5.0  # R inflation factor for inconsistent measurements


# =============================================================================
# Module-level state for magnetometer filter
# =============================================================================

_MAG_FILTER_STATE = {
    'yaw_ema': None,
    'last_yaw_mag': None,
    'last_yaw_t': None,
    'integrated_gyro_dz': 0.0,
    'n_updates': 0,
}


def reset_mag_filter_state():
    """Reset magnetometer filter state (call at start of each VIO run)."""
    global _MAG_FILTER_STATE
    _MAG_FILTER_STATE = {
        'yaw_ema': None,
        'last_yaw_mag': None,
        'last_yaw_t': None,
        'integrated_gyro_dz': 0.0,
        'n_updates': 0,
    }


# =============================================================================
# Calibration Functions
# =============================================================================

def calibrate_magnetometer(mag_raw: np.ndarray, 
                          hard_iron: Optional[np.ndarray] = None,
                          soft_iron: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply hard-iron and soft-iron calibration to raw magnetometer reading.
    
    Calibration model: M_calibrated = soft_iron @ (M_raw - hard_iron)
    
    Args:
        mag_raw: Raw 3D magnetic field vector [mx, my, mz]
        hard_iron: Hard-iron offset to subtract (3,)
        soft_iron: Soft-iron correction matrix (3x3) - full matrix for ellipsoid correction
    
    Returns: Calibrated magnetic field vector
    """
    if hard_iron is None:
        hard_iron = DEFAULT_MAG_HARD_IRON
    if soft_iron is None:
        soft_iron = DEFAULT_MAG_SOFT_IRON
    
    # Subtract hard-iron offset
    mag_centered = mag_raw - hard_iron
    
    # Apply soft-iron correction (full 3x3 matrix)
    mag_corrected = soft_iron @ mag_centered
    
    return mag_corrected


# =============================================================================
# Yaw Computation
# =============================================================================

def compute_yaw_from_mag(mag_body: np.ndarray, q_wxyz: np.ndarray, 
                         mag_declination: float = 0.0,
                         use_raw_heading: bool = True) -> Tuple[float, float]:
    """
    Compute yaw angle from calibrated magnetometer.
    
    Two modes:
    1. Raw heading (use_raw_heading=True): Uses body-frame mag directly
       - Formula: atan2(-mag_y, mag_x) + declination
       - Best for level flight when IMU orientation is uncertain
       - GPS-calibrated for Bell 412 dataset
       
    2. Tilt-compensated (use_raw_heading=False): Rotates mag to world frame
       - Uses IMU quaternion for tilt compensation
       - Better for non-level flight IF IMU frame is correctly aligned
    
    Args:
        mag_body: Calibrated magnetometer reading in body frame [mx, my, mz]
        q_wxyz: Current attitude quaternion [w,x,y,z]
        mag_declination: Magnetic declination in radians (East positive)
        use_raw_heading: If True, use raw body-frame heading (GPS-calibrated)
    
    Returns: (yaw_rad, quality_score)
        yaw_rad: Yaw angle in radians (0 = North in NED frame)
        quality_score: Confidence [0,1] based on horizontal field strength
    """
    # =========================================================================
    # RAW HEADING MODE (GPS-calibrated, works for near-level flight)
    # =========================================================================
    if use_raw_heading:
        # GPS-calibrated formula: atan2(-mag_y, mag_x)
        # This was determined by comparing with PPK GPS ground truth heading
        # Works best for Bell 412 dataset where roll/pitch are typically < 10°
        # 
        # This gives heading in NED frame (0 = North, positive = clockwise)
        yaw_ned = np.arctan2(-mag_body[1], mag_body[0])
        
        # Apply declination correction (includes GPS-based heading offset)
        yaw_ned += mag_declination
        
        # Convert NED yaw to ENU yaw (to match state quaternion frame)
        # ENU: 0 = East, positive = CCW (standard math convention)
        # NED: 0 = North, positive = CW (aviation convention)
        # Conversion: yaw_enu = π/2 - yaw_ned
        yaw_enu = np.pi/2 - yaw_ned
        
        # Normalize to [-π, π]
        yaw_enu = np.arctan2(np.sin(yaw_enu), np.cos(yaw_enu))
        
        # Quality based on horizontal field strength in body frame
        horizontal_strength = np.sqrt(mag_body[0]**2 + mag_body[1]**2)
        total_strength = np.linalg.norm(mag_body)
        
        if total_strength > 0:
            quality = horizontal_strength / total_strength
        else:
            quality = 0.0
        
        quality = np.clip(quality, 0.0, 1.0)
        return yaw_enu, quality
    
    # =========================================================================
    # TILT-COMPENSATED MODE (requires correct IMU frame alignment)
    # =========================================================================
    # Convert quaternion to rotation matrix (body → world)
    # Note: This assumes IMU quaternion convention matches expected frame
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    r_world_to_body = R_scipy.from_quat(q_xyzw).as_matrix()
    r_body_to_world = r_world_to_body.T  # Transpose to get body→world
    
    # Rotate magnetometer to world frame
    mag_world = r_body_to_world @ mag_body.reshape(3, 1)
    mag_world = mag_world.reshape(3)
    
    # Extract horizontal components (ENU frame: X=East, Y=North, Z=Up)
    mx_world = mag_world[0]  # East component
    my_world = mag_world[1]  # North component
    
    # Compute yaw from magnetic north (NED convention: 0=North, positive=CW)
    # atan2(East, North) gives heading where 0=North
    yaw_ned = np.arctan2(mx_world, my_world)
    
    # Apply declination correction
    yaw_ned += mag_declination
    
    # Convert NED yaw to ENU yaw (to match state quaternion frame)
    yaw_enu = np.pi/2 - yaw_ned
    
    # Normalize to [-π, π]
    yaw_enu = np.arctan2(np.sin(yaw_enu), np.cos(yaw_enu))
    
    # Quality score based on horizontal field strength
    horizontal_strength = np.sqrt(mx_world**2 + my_world**2)
    total_strength = np.linalg.norm(mag_world)
    
    if total_strength > 0:
        quality = horizontal_strength / total_strength
    else:
        quality = 0.0
    
    quality = np.clip(quality, 0.0, 1.0)
    
    return yaw_enu, quality


# Use shared implementations from math_utils
# These are re-exported for backward compatibility
def quaternion_to_yaw(q_wxyz: np.ndarray) -> float:
    """Extract yaw from quaternion (wrapper to math_utils)."""
    return _quaternion_to_yaw_shared(q_wxyz)


def yaw_to_quaternion_update(yaw_current: float, yaw_measured: float) -> np.ndarray:
    """Compute quaternion correction for yaw-only update (wrapper to math_utils)."""
    return _yaw_to_quaternion_update_shared(yaw_current, yaw_measured)


# =============================================================================
# Filtering Functions
# =============================================================================

def angle_wrap(angle: float) -> float:
    """Wrap angle to [-π, π] (wrapper to math_utils)."""
    return _angle_wrap_shared(angle)


def apply_mag_filter(yaw_mag: float, yaw_t: float, gyro_z: float, dt_imu: float, 
                     in_convergence: bool = False,
                     mag_max_yaw_rate: float = None,
                     mag_gyro_threshold: float = None,
                     mag_ema_alpha: float = None,
                     mag_consistency_r_inflate: float = None) -> Tuple[float, float, dict]:
    """
    Apply enhanced magnetometer filtering with EMA smoothing.
    
    This function maintains internal state across calls to:
    1. Apply Exponential Moving Average (EMA) to smooth yaw measurements
    2. Track gyro integration for rate-of-change monitoring (logging only)
    
    Note: R-inflation based on gyro consistency is DISABLED during convergence
    to allow magnetometer to correct large yaw drift.
    
    Args:
        yaw_mag: Raw yaw measurement from magnetometer [rad]
        yaw_t: Timestamp of magnetometer measurement [s]
        gyro_z: Current gyroscope z-axis angular velocity [rad/s]
        dt_imu: Time since last IMU update [s]
        in_convergence: If True, disable R inflation to allow yaw correction
        mag_max_yaw_rate: Override for MAG_MAX_YAW_RATE constant
        mag_gyro_threshold: Override for MAG_GYRO_CONSISTENCY_THRESHOLD constant
        mag_ema_alpha: Override for MAG_EMA_ALPHA constant
        mag_consistency_r_inflate: Override for MAG_CONSISTENCY_R_INFLATE constant
        
    Returns:
        (yaw_filtered, r_scale_factor, info_dict)
        yaw_filtered: EMA-smoothed yaw measurement [rad]
        r_scale_factor: Multiplier for measurement noise R (1.0 during convergence)
        info_dict: {'high_rate': bool, 'gyro_inconsistent': bool} for logging
    """
    global _MAG_FILTER_STATE
    
    # Use provided values or defaults
    max_yaw_rate = mag_max_yaw_rate if mag_max_yaw_rate is not None else MAG_MAX_YAW_RATE
    gyro_threshold = mag_gyro_threshold if mag_gyro_threshold is not None else MAG_GYRO_CONSISTENCY_THRESHOLD
    ema_alpha = mag_ema_alpha if mag_ema_alpha is not None else MAG_EMA_ALPHA
    consistency_r_inflate = mag_consistency_r_inflate if mag_consistency_r_inflate is not None else MAG_CONSISTENCY_R_INFLATE
    
    state = _MAG_FILTER_STATE
    r_scale = 1.0
    info = {'high_rate': False, 'gyro_inconsistent': False}  # v2.9.2 logging
    
    # Accumulate gyro integration between mag updates
    state['integrated_gyro_dz'] += gyro_z * dt_imu
    
    # First measurement - initialize state
    if state['yaw_ema'] is None:
        state['yaw_ema'] = yaw_mag
        state['last_yaw_mag'] = yaw_mag
        state['last_yaw_t'] = yaw_t
        state['integrated_gyro_dz'] = 0.0
        state['n_updates'] = 1
        # v2.9.2: Must return 3 values (yaw, r_scale, info_dict)
        return yaw_mag, 1.0, {'high_rate': False, 'gyro_inconsistent': False}
    
    # =========================================================================
    # Step 1: Rate-of-change check (fast rotation detection)
    # Only apply R inflation AFTER convergence period
    # =========================================================================
    dt_mag = yaw_t - state['last_yaw_t'] if state['last_yaw_t'] is not None else 0.01
    if dt_mag < 0.001:
        dt_mag = 0.001  # Avoid division by zero
        
    # Yaw change from previous mag measurement
    dyaw_mag = angle_wrap(yaw_mag - state['last_yaw_mag'])
    yaw_rate_mag = abs(dyaw_mag) / dt_mag
    
    # Check if rate exceeds maximum expected (from gyro)
    # Only inflate R after convergence (during convergence, we need mag to fix drift)
    if not in_convergence and yaw_rate_mag > max_yaw_rate * 1.5:
        # Mag is changing faster than gyro max rate - likely noise/interference
        r_scale = max(r_scale, 2.0)
        info['high_rate'] = True
        if state['n_updates'] % 50 == 0:
            print(f"[MAG-FILTER] High rate: {np.degrees(yaw_rate_mag):.1f}°/s > {np.degrees(max_yaw_rate*1.5):.1f}°/s limit → R×{r_scale:.1f}")
    
    # =========================================================================
    # Step 2: Gyro consistency check (DISABLED during convergence)
    # =========================================================================
    # Compare mag yaw change with integrated gyro rotation
    expected_dyaw_gyro = state['integrated_gyro_dz']  # Expected change from gyro
    
    consistency_error = abs(angle_wrap(dyaw_mag - expected_dyaw_gyro))
    
    # Only apply R inflation after convergence period
    if not in_convergence and consistency_error > gyro_threshold:
        # Mag measurement inconsistent with gyro integration
        # Scale R inflation based on how inconsistent
        r_scale_consistency = 1.0 + (consistency_error / gyro_threshold - 1.0) * (consistency_r_inflate - 1.0)
        r_scale_consistency = np.clip(r_scale_consistency, 1.0, consistency_r_inflate)
        r_scale = max(r_scale, r_scale_consistency)
        info['gyro_inconsistent'] = True
        
        if state['n_updates'] % 50 == 0:
            print(f"[MAG-FILTER] Gyro inconsistent: |d_yaw_mag({np.degrees(dyaw_mag):.1f}°) - d_yaw_gyro({np.degrees(expected_dyaw_gyro):.1f}°)| = {np.degrees(consistency_error):.1f}° → R×{r_scale:.1f}")
    
    # Reset gyro integration for next mag interval
    state['integrated_gyro_dz'] = 0.0
    
    # =========================================================================
    # Step 3: Apply EMA smoothing (in angle space, handling wraparound)
    # =========================================================================
    # Compute innovation in wrapped angle space
    yaw_diff = angle_wrap(yaw_mag - state['yaw_ema'])
    
    # Apply EMA: new_ema = old_ema + alpha * (new - old_ema)
    yaw_ema_new = angle_wrap(state['yaw_ema'] + ema_alpha * yaw_diff)
    
    # =========================================================================
    # Step 4: Update state for next iteration
    # =========================================================================
    state['yaw_ema'] = yaw_ema_new
    state['last_yaw_mag'] = yaw_mag
    state['last_yaw_t'] = yaw_t
    state['n_updates'] += 1
    
    return yaw_ema_new, r_scale, info


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (wrapper to math_utils)."""
    return _quaternion_multiply_shared(q1, q2)


# =============================================================================
# Utility Functions
# =============================================================================

def get_mag_filter_state() -> dict:
    """Get current magnetometer filter state (for debugging)."""
    return _MAG_FILTER_STATE.copy()


def set_mag_constants(ema_alpha: float = None, 
                      max_yaw_rate: float = None,
                      gyro_threshold: float = None,
                      consistency_r_inflate: float = None):
    """
    Set module-level magnetometer constants.
    
    Args:
        ema_alpha: EMA smoothing factor (0-1)
        max_yaw_rate: Maximum expected yaw rate [rad/s]
        gyro_threshold: Gyro consistency threshold [rad]
        consistency_r_inflate: R inflation factor for inconsistent measurements
    """
    global MAG_EMA_ALPHA, MAG_MAX_YAW_RATE, MAG_GYRO_CONSISTENCY_THRESHOLD, MAG_CONSISTENCY_R_INFLATE
    
    if ema_alpha is not None:
        MAG_EMA_ALPHA = ema_alpha
    if max_yaw_rate is not None:
        MAG_MAX_YAW_RATE = max_yaw_rate
    if gyro_threshold is not None:
        MAG_GYRO_CONSISTENCY_THRESHOLD = gyro_threshold
    if consistency_r_inflate is not None:
        MAG_CONSISTENCY_R_INFLATE = consistency_r_inflate
