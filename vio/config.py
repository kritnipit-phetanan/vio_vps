#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIO Configuration Module

Handles YAML configuration loading and global constants.
"""

import os
import yaml
import numpy as np
from typing import Dict, Any

# Debug verbosity control - set to False for cleaner output
VERBOSE_DEBUG = False  # Set to True for detailed per-IMU debug output
VERBOSE_DEM = False    # Set to True for per-IMU DEM update logs


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file and convert to global variables format."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert nested dictionary to flat structure for compatibility
    result = {}
    
    # Camera intrinsics - build KB_PARAMS dict from camera section
    cam = config['camera']
    result['KB_PARAMS'] = {
        'k2': cam['k2'],
        'k3': cam['k3'],
        'k4': cam['k4'],
        'k5': cam['k5'],
        'mu': cam['mu'],
        'mv': cam['mv'],
        'u0': cam['u0'],
        'v0': cam['v0'],
        'w': cam['image_width'],
        'h': cam['image_height'],
    }
    
    # Extrinsics (convert lists to numpy arrays)
    extr = config['extrinsics']
    
    # CRITICAL: Body frame convention determines whether we need R_flip!
    # - FLU body (Z-up): Original Kalibr calibration has camera Z pointing UP
    #   Need R_flip to make camera Z point DOWN for nadir camera
    # - FRD body (Z-down): Kalibr calibration already has camera Z pointing DOWN
    #   No R_flip needed!
    #
    # Bell 412 dataset uses FRD body frame (quaternion shows body Z pointing down)
    # So we should NOT apply R_flip here!
    #
    # R_flip = diag(1, -1, -1) rotates 180° around X-axis
    R_flip = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float64)
    
    # Check if we should apply R_flip based on body frame convention
    # For now, DISABLE R_flip for Bell 412 (FRD body)
    APPLY_R_FLIP = False  # Set to True for FLU body frame datasets
    
    def correct_camera_extrinsics(T_bc):
        """Apply 180° rotation around camera X-axis to fix optical axis direction."""
        if not APPLY_R_FLIP:
            return T_bc.copy()  # No correction needed for FRD body frame
        T_corrected = T_bc.copy()
        R_bc = T_bc[:3, :3]
        # New rotation: R_bc @ R_flip (apply flip in camera frame)
        T_corrected[:3, :3] = R_bc @ R_flip
        return T_corrected
    
    T_orig = np.array(extr['nadir']['transform'], dtype=np.float64)
    result['BODY_T_CAMDOWN'] = correct_camera_extrinsics(T_orig)
    
    result['BODY_T_CAMFRONT'] = correct_camera_extrinsics(
        np.array(extr['front']['transform'], dtype=np.float64))
    result['BODY_T_CAMSIDE'] = correct_camera_extrinsics(
        np.array(extr['side']['transform'], dtype=np.float64))
    
    # IMU parameters
    imu = config['imu']
    result['IMU_PARAMS'] = {
        'acc_n': imu['acc_n'],
        'gyr_n': imu['gyr_n'],
        'acc_w': imu['acc_w'],
        'gyr_w': imu['gyr_w'],
        'g_norm': imu['g_norm'],
    }
    result['IMU_PARAMS_PREINT'] = {
        'acc_n': imu['preintegration']['acc_n'],
        'gyr_n': imu['preintegration']['gyr_n'],
        'acc_w': imu['preintegration']['acc_w'],
        'gyr_w': imu['preintegration']['gyr_w'],
        'g_norm': imu['g_norm'],
    }
    
    # IMU bias settings
    result['ESTIMATE_IMU_BIAS'] = imu.get('estimate_bias', False)
    initial_gyro_bias = imu.get('initial_gyro_bias', [0.0, 0.0, 0.0])
    result['INITIAL_GYRO_BIAS'] = np.array(initial_gyro_bias, dtype=float)
    initial_accel_bias = imu.get('initial_accel_bias', [0.0, 0.0, 0.0])
    result['INITIAL_ACCEL_BIAS'] = np.array(initial_accel_bias, dtype=float)
    
    # Magnetometer calibration
    mag = config['magnetometer']
    result['MAG_HARD_IRON_OFFSET'] = np.array(mag['hard_iron_offset'], dtype=float)
    result['MAG_SOFT_IRON_MATRIX'] = np.array(mag['soft_iron_matrix'], dtype=float)
    result['MAG_DECLINATION'] = mag['declination']
    result['MAG_FIELD_STRENGTH'] = mag['expected_field_strength']
    result['MAG_MIN_FIELD_STRENGTH'] = mag['min_field_strength']
    result['MAG_MAX_FIELD_STRENGTH'] = mag['max_field_strength']
    result['MAG_UPDATE_RATE_LIMIT'] = mag['update_rate_limit']
    result['MAG_USE_RAW_HEADING'] = mag.get('use_raw_heading', True)
    result['MAG_APPLY_INITIAL_CORRECTION'] = mag.get('apply_initial_correction', True)
    result['MAG_INITIAL_CONVERGENCE_WINDOW'] = mag.get('convergence_window', 30.0)
    
    # Process noise
    pn = config['process_noise']
    result['SIGMA_ACCEL'] = pn['sigma_accel']
    result['SIGMA_VO_VEL'] = pn['sigma_vo_vel']
    result['SIGMA_VPS_XY'] = pn['sigma_vps_xy']
    result['SIGMA_AGL_Z'] = pn['sigma_agl_z']
    result['SIGMA_MAG_YAW'] = pn['sigma_mag_yaw']
    
    # VIO parameters
    vio = config['vio']
    result['MIN_PARALLAX_PX'] = vio['min_parallax_px']
    result['MIN_MSCKF_BASELINE'] = vio['min_msckf_baseline']
    result['MSCKF_CHI2_MULTIPLIER'] = vio['msckf_chi2_multiplier']
    result['MSCKF_MAX_REPROJECTION_ERROR'] = vio['msckf_max_reprojection_error']
    result['VO_MIN_INLIERS'] = vio['min_inliers']
    result['VO_RATIO_TEST'] = vio['ratio_test']
    result['VO_NADIR_ALIGN_DEG'] = vio['views']['nadir']['nadir_threshold_deg']
    result['VO_FRONT_ALIGN_DEG'] = vio['views']['front']['nadir_threshold_deg']
    
    # Camera view configs - convert to expected format
    result['CAMERA_VIEW_CONFIGS'] = {
        'nadir': {
            'extrinsics': 'BODY_T_CAMDOWN',
            'nadir_threshold': vio['views']['nadir']['nadir_threshold_deg'],
            'sigma_scale_xy': vio['views']['nadir']['sigma_scale_xy'],
            'sigma_scale_z': vio['views']['nadir']['sigma_scale_z'],
            'use_vz_only': vio['views']['nadir']['use_vz_only'],
            'min_parallax': vio['views']['nadir']['min_parallax'],
            'max_corners': vio['views']['nadir']['max_corners'],
        },
        'front': {
            'extrinsics': 'BODY_T_CAMFRONT',
            'nadir_threshold': vio['views']['front']['nadir_threshold_deg'],
            'sigma_scale_xy': vio['views']['front']['sigma_scale_xy'],
            'sigma_scale_z': vio['views']['front']['sigma_scale_z'],
            'use_vz_only': vio['views']['front']['use_vz_only'],
            'min_parallax': vio['views']['front']['min_parallax'],
            'max_corners': vio['views']['front']['max_corners'],
        },
        'side': {
            'extrinsics': 'BODY_T_CAMSIDE',
            'nadir_threshold': vio['views']['side']['nadir_threshold_deg'],
            'sigma_scale_xy': vio['views']['side']['sigma_scale_xy'],
            'sigma_scale_z': vio['views']['side']['sigma_scale_z'],
            'use_vz_only': vio['views']['side']['use_vz_only'],
            'min_parallax': vio['views']['side']['min_parallax'],
            'max_corners': vio['views']['side']['max_corners'],
        },
    }
    
    # Undistort backend
    result['USE_FISHEYE'] = config.get('use_fisheye', True)
    
    # IMU-GNSS Lever Arm (optional - defaults to zero if not specified)
    if 'imu_gnss_extrinsics' in config:
        imu_gnss = config['imu_gnss_extrinsics']
        result['BODY_T_GNSS'] = np.array(imu_gnss['transform'], dtype=np.float64)
        result['IMU_GNSS_LEVER_ARM'] = result['BODY_T_GNSS'][:3, 3]
    else:
        # Default: no lever arm (GNSS at same position as IMU)
        result['BODY_T_GNSS'] = np.eye(4, dtype=np.float64)
        result['IMU_GNSS_LEVER_ARM'] = np.zeros(3, dtype=np.float64)
    
    return result


# =============================================================================
# Default Configuration Variables (will be overridden by load_config)
# =============================================================================

# Camera parameters
KB_PARAMS = {}
BODY_T_CAMDOWN = np.eye(4, dtype=np.float64)
BODY_T_CAMFRONT = np.eye(4, dtype=np.float64)
BODY_T_CAMSIDE = np.eye(4, dtype=np.float64)

# IMU parameters
IMU_PARAMS = {}
IMU_PARAMS_PREINT = {}

# IMU-GNSS Lever Arm (will be overridden by load_config if available)
BODY_T_GNSS = np.eye(4, dtype=np.float64)
IMU_GNSS_LEVER_ARM = np.zeros(3, dtype=np.float64)

# Magnetometer parameters
MAG_HARD_IRON_OFFSET = np.zeros(3, dtype=float)
MAG_SOFT_IRON_MATRIX = np.eye(3, dtype=float)
MAG_DECLINATION = 0.0
MAG_FIELD_STRENGTH = 50.0
MAG_MIN_FIELD_STRENGTH = 0.1  # RELAXED: normalized data may have low magnitude
MAG_MAX_FIELD_STRENGTH = 100.0
MAG_UPDATE_RATE_LIMIT = 1  # Process EVERY mag sample for faster convergence
MAG_USE_RAW_HEADING = True  # GPS-calibrated raw body-frame heading
MAG_APPLY_INITIAL_CORRECTION = True  # Apply mag correction at startup
MAG_INITIAL_CONVERGENCE_WINDOW = 30.0  # 30s initial settling window
MAG_ADAPTIVE_THRESHOLD = True  # Use adaptive threshold based on yaw uncertainty

# Enhanced Magnetometer Filtering Parameters
MAG_EMA_ALPHA = 0.3  # EMA smoothing factor: 0.3 = 30% new, 70% old
MAG_GYRO_CONSISTENCY_THRESHOLD = np.radians(5.0)  # Max deviation from gyro (5°)
MAG_MAX_YAW_RATE = np.radians(45.0)  # Max expected yaw rate from gyro (45°/s)
MAG_CONSISTENCY_R_INFLATE = 4.0  # Inflate R by this factor when inconsistent

# Process noise parameters
SIGMA_ACCEL = 0.8
SIGMA_VO_VEL = 2.0
SIGMA_VPS_XY = 1.0
SIGMA_AGL_Z = 2.5
SIGMA_MAG_YAW = 0.15  # ~9° measurement noise

# VIO parameters
MIN_PARALLAX_PX = 2.0
MIN_MSCKF_BASELINE = 0.15
MSCKF_CHI2_MULTIPLIER = 4.0
MSCKF_MAX_REPROJECTION_ERROR = 3.0
VO_MIN_INLIERS = 15
VO_RATIO_TEST = 0.75
VO_NADIR_ALIGN_DEG = 30.0
VO_FRONT_ALIGN_DEG = 60.0
CAMERA_VIEW_CONFIGS = {}
USE_FISHEYE = True

# IMU bias settings
ESTIMATE_IMU_BIAS = False
INITIAL_GYRO_BIAS = np.zeros(3, dtype=float)
INITIAL_ACCEL_BIAS = np.zeros(3, dtype=float)


# =============================================================================
# MSCKF Triangulation Statistics (for debugging)
# =============================================================================
MSCKF_STATS = {
    'total_attempt': 0,
    'success': 0,
    'fail_few_obs': 0,
    'fail_baseline': 0,
    'fail_parallax': 0,
    'fail_depth_sign': 0,
    'fail_depth_large': 0,
    'fail_reproj_error': 0,
    'fail_nonlinear': 0,
    'fail_chi2': 0,
    'fail_solver': 0,
    'fail_other': 0,
}


def reset_msckf_stats():
    """Reset MSCKF statistics counters."""
    global MSCKF_STATS
    for key in MSCKF_STATS:
        MSCKF_STATS[key] = 0


def print_msckf_stats():
    """Print MSCKF triangulation statistics summary."""
    total = MSCKF_STATS['total_attempt']
    if total == 0:
        print("[MSCKF-TRI] No triangulation attempts")
        return
    
    success = MSCKF_STATS['success']
    rate = 100.0 * success / total if total > 0 else 0
    
    print(f"[MSCKF-TRI] total_attempt={total} succ={success} rate={rate:.1f}%")
    print(f"  Failure breakdown:")
    print(f"    few_obs={MSCKF_STATS['fail_few_obs']}")
    print(f"    baseline={MSCKF_STATS['fail_baseline']}")
    print(f"    parallax={MSCKF_STATS['fail_parallax']}")
    print(f"    depth_sign={MSCKF_STATS['fail_depth_sign']}")
    print(f"    depth_large={MSCKF_STATS['fail_depth_large']}")
    print(f"    reproj_error={MSCKF_STATS['fail_reproj_error']}")
    print(f"    nonlinear={MSCKF_STATS['fail_nonlinear']}")
    print(f"    chi2={MSCKF_STATS['fail_chi2']}")
    print(f"    solver={MSCKF_STATS['fail_solver']}")
    print(f"    other={MSCKF_STATS['fail_other']}")


# Runtime state for magnetometer filtering (reset per run)
MAG_FILTER_STATE = {
    'yaw_ema': None,  # EMA smoothed yaw
    'last_yaw_mag': None,  # Previous yaw_mag for rate check
    'last_yaw_t': None,  # Timestamp of last mag measurement
    'integrated_gyro_dz': 0.0,  # Integrated gyro_z since last mag update
    'n_updates': 0,  # Count of successful mag updates
}
