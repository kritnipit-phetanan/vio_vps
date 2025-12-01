#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIO + VPS + DEM fusion with EKF (IMU-driven predict)
- Removes dependency on flight_log_with_ground_truth2_with_dem_agl.csv
- Uses flight_log_from_gga.csv only for initial lat_dd/lon_dd and altitude_MSL_m
- Computes initial AGL from DEM if provided, sets initial z accordingly
- Predicts at IMU rate (400 Hz) using IMU orientation + linear acceleration
- Optional updates (auto-enabled if the corresponding inputs are present):
    * VIO (from images) → updates velocity (vx,vy, vz or only vz for near-nadir)
    * VPS (from vps_result.csv) → updates position (px, py)
    * DEM (DSM_*.tif) → used to keep state z as MSL and to compute/log AGL
      (z state is MSL; AGL = z - dem(lat,lon)).
- Asynchronous-rate handling:
    * IMU (stamp_log in seconds) @ ~400 Hz → main clock
    * Images (stamp_log in seconds) @ ~20 Hz → event updates
    * VPS (seconds) → event updates
    * flight_log_from_gga.csv (stamp_log in seconds @ 5 Hz) is used ONLY for initial lat/lon/MSL
- Output CSV adds `Frame` (VIO frame index, starts at 0 at first VIO update)
  and `dt` (seconds since previous IMU sample, starts at 0 for the first row).
- Startup prints which sources were loaded (imu / images / video / dem / vps / flight_log).

DEBUG VERSION:
- Added inline TODO comments and DEBUG print statements to systematically check:
  * IMU propagation logic, units, and sign conventions
  * VPS update logic, units, and sign conventions
  * VIO/VO update logic, units, and sign conventions
  * DEM/height update logic, units, and sign conventions
  * Gravity compensation
- Debug prints show:
  * [DEBUG][IMU] IMU propagation state before/after, accel, velocity, position
  * [DEBUG][VPS] VPS measurement, innovation, chi-square test, update status
  * [DEBUG][VIO] VIO measurement, alignment, nadir mode, update status
  * [DEBUG][DEM] DEM value, AGL/MSL, height measurement, innovation, update status
- Run with these debug prints enabled to diagnose state drift and monotonic trends

Author: VIO project
"""
from __future__ import annotations

import os
import sys
import time
import math
import glob
import csv
import yaml
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R_scipy

# NOTE: RANSAC in OpenCV uses random sampling which causes non-determinism
# For debugging, set seeds here. For production, leave unset for variety.
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# cv2.setRNGSeed(RANDOM_SEED)

import rasterio
from pyproj import CRS, Transformer

# Debug verbosity control - set to False for cleaner output
VERBOSE_DEBUG = False  # Set to True for detailed per-IMU debug output
VERBOSE_DEM = False    # Set to True for per-IMU DEM update logs

# --- Necessary imports for the ExtendedKalmanFilter class ---
from copy import deepcopy
from math import log, exp, sqrt
from numpy import dot, zeros, eye
import scipy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z

# ========================
# Configuration Loading
# ========================

def load_config(config_path: str) -> dict:
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
    result['MAG_USE_RAW_HEADING'] = mag.get('use_raw_heading', True)  # GPS-calibrated raw heading
    result['MAG_APPLY_INITIAL_CORRECTION'] = mag.get('apply_initial_correction', True)  # Apply mag correction at startup
    result['MAG_INITIAL_CONVERGENCE_WINDOW'] = mag.get('convergence_window', 30.0)  # Default 30s
    
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

# Default configuration variables (will be overridden by load_config)
KB_PARAMS = {}
BODY_T_CAMDOWN = np.eye(4, dtype=np.float64)
BODY_T_CAMFRONT = np.eye(4, dtype=np.float64)
BODY_T_CAMSIDE = np.eye(4, dtype=np.float64)
IMU_PARAMS = {}
IMU_PARAMS_PREINT = {}

# IMU-GNSS Lever Arm (will be overridden by load_config if available)
# Default: no lever arm (GNSS at same position as IMU)
BODY_T_GNSS = np.eye(4, dtype=np.float64)
IMU_GNSS_LEVER_ARM = np.zeros(3, dtype=np.float64)

MAG_HARD_IRON_OFFSET = np.zeros(3, dtype=float)
MAG_SOFT_IRON_MATRIX = np.eye(3, dtype=float)
MAG_DECLINATION = 0.0
MAG_FIELD_STRENGTH = 50.0
MAG_MIN_FIELD_STRENGTH = 0.1  # RELAXED: normalized data may have low magnitude
MAG_MAX_FIELD_STRENGTH = 100.0
MAG_UPDATE_RATE_LIMIT = 1  # Process EVERY mag sample for faster convergence (was 5)
MAG_USE_RAW_HEADING = True  # GPS-calibrated raw body-frame heading
MAG_APPLY_INITIAL_CORRECTION = True  # Apply mag correction at startup
MAG_INITIAL_CONVERGENCE_WINDOW = 30.0  # REDUCED: 30s is enough for initial settling (was 600s)
                                        # Flight is only 5min, 10min window meant NO mag updates!
MAG_ADAPTIVE_THRESHOLD = True  # Use adaptive threshold based on yaw uncertainty

# ===============
# Enhanced Magnetometer Filtering Parameters
# ===============
MAG_EMA_ALPHA = 0.3  # EMA smoothing factor: 0.3 = 30% new, 70% old
MAG_GYRO_CONSISTENCY_THRESHOLD = np.radians(5.0)  # Max deviation from gyro integration (5°)
MAG_MAX_YAW_RATE = np.radians(45.0)  # Max expected yaw rate from gyro (45°/s)
MAG_CONSISTENCY_R_INFLATE = 4.0  # Inflate R by this factor when inconsistent with gyro

# Runtime state for magnetometer filtering (reset per run)
_MAG_FILTER_STATE = {
    'yaw_ema': None,  # EMA smoothed yaw
    'last_yaw_mag': None,  # Previous yaw_mag for rate check
    'last_yaw_t': None,  # Timestamp of last mag measurement
    'integrated_gyro_dz': 0.0,  # Integrated gyro_z since last mag update
    'n_updates': 0,  # Count of successful mag updates
}

SIGMA_ACCEL = 0.8
SIGMA_VO_VEL = 2.0
SIGMA_VPS_XY = 1.0
SIGMA_AGL_Z = 2.5
SIGMA_MAG_YAW = 0.15  # REDUCED from 1.0 to 0.15 rad (~9°) for higher Kalman gain
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

# ===============
# MSCKF Triangulation Statistics (for debugging)
# ===============
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


# ===============
# Loop Closure Detection & Correction
# ===============
# Lightweight loop closure for yaw drift correction:
# 1. Store keyframes with position, yaw, and ORB descriptors
# 2. Detect loop when returning to visited location (position proximity)
# 3. Match visual features to confirm loop
# 4. Compute relative yaw and apply correction

class LoopClosureDetector:
    """
    Lightweight loop closure detector for yaw drift correction.
    
    Strategy:
    - Store sparse keyframes (every N meters or M degrees rotation)
    - When estimated position is near a stored keyframe, attempt visual matching
    - If match successful, compute yaw correction from relative pose
    - Apply yaw correction to EKF state
    
    Key insight: For helicopter returning to starting position, we mainly need
    to correct YAW drift. Position will be corrected automatically once yaw is fixed.
    
    TUNED FOR PARTIAL REVISITS:
    - Lower position threshold (30m instead of 50m) to catch near-passes
    - Lower minimum inliers (15 instead of 20) for partial view overlap
    - More aggressive keyframe addition for better coverage
    """
    
    def __init__(self, 
                 position_threshold: float = 30.0,    # REDUCED: catch partial revisits
                 min_keyframe_dist: float = 15.0,     # REDUCED: denser keyframe coverage
                 min_keyframe_yaw: float = 20.0,      # REDUCED: more keyframes during turns
                 min_frame_gap: int = 50,             # REDUCED: faster loop detection
                 min_match_ratio: float = 0.12,       # REDUCED: accept partial overlaps
                 min_inliers: int = 15):              # REDUCED: accept partial matches
        
        self.position_threshold = position_threshold
        self.min_keyframe_dist = min_keyframe_dist
        self.min_keyframe_yaw = np.radians(min_keyframe_yaw)
        self.min_frame_gap = min_frame_gap
        self.min_match_ratio = min_match_ratio
        self.min_inliers = min_inliers
        
        # Keyframe database: list of {frame_idx, position, yaw, descriptors, keypoints}
        self.keyframes = []
        
        # ORB detector for loop closure (separate from VIO front-end)
        # Increased features for better matching in partial overlaps
        self.orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
        
        # BFMatcher for descriptor matching
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Statistics
        self.stats = {
            'keyframes_added': 0,
            'loop_checks': 0,
            'loop_candidates': 0,
            'loop_detected': 0,
            'loop_rejected_few_matches': 0,
            'loop_rejected_geometry': 0,
            'yaw_corrections_applied': 0,
            'total_yaw_correction': 0.0,
        }
        
        # Last keyframe info (to avoid adding too many)
        self.last_kf_position = None
        self.last_kf_yaw = None
        self.last_kf_frame = -1000
        
    def should_add_keyframe(self, position: np.ndarray, yaw: float, frame_idx: int) -> bool:
        """Check if current frame should be stored as keyframe."""
        if self.last_kf_position is None:
            return True
            
        # Distance from last keyframe
        dist = np.linalg.norm(position[:2] - self.last_kf_position[:2])
        
        # Yaw change from last keyframe
        yaw_diff = abs(np.arctan2(np.sin(yaw - self.last_kf_yaw), np.cos(yaw - self.last_kf_yaw)))
        
        # Frame gap
        frame_gap = frame_idx - self.last_kf_frame
        
        # Add keyframe if sufficient motion or rotation
        if dist > self.min_keyframe_dist or yaw_diff > self.min_keyframe_yaw:
            if frame_gap > 20:  # At least 20 frames apart
                return True
                
        return False
    
    def add_keyframe(self, frame_idx: int, position: np.ndarray, yaw: float, 
                     gray_image: np.ndarray) -> None:
        """Add a new keyframe to the database."""
        # Detect ORB features
        keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
        
        if descriptors is None or len(keypoints) < 50:
            return  # Not enough features
            
        # Store keyframe
        kf = {
            'frame_idx': frame_idx,
            'position': position.copy(),
            'yaw': yaw,
            'keypoints': keypoints,
            'descriptors': descriptors,
        }
        self.keyframes.append(kf)
        
        # Update last keyframe info
        self.last_kf_position = position.copy()
        self.last_kf_yaw = yaw
        self.last_kf_frame = frame_idx
        
        self.stats['keyframes_added'] += 1
        
        if self.stats['keyframes_added'] % 10 == 0:
            print(f"[LOOP] Added keyframe #{self.stats['keyframes_added']} at frame {frame_idx}, "
                  f"pos=({position[0]:.1f}, {position[1]:.1f}), yaw={np.degrees(yaw):.1f}°")
    
    def find_loop_candidates(self, position: np.ndarray, frame_idx: int) -> List[int]:
        """Find keyframes that are close to current position."""
        candidates = []
        
        for i, kf in enumerate(self.keyframes):
            # Skip recent keyframes (need sufficient time gap)
            if frame_idx - kf['frame_idx'] < self.min_frame_gap:
                continue
                
            # Check position proximity
            dist = np.linalg.norm(position[:2] - kf['position'][:2])
            if dist < self.position_threshold:
                candidates.append(i)
                
        return candidates
    
    def match_keyframe(self, gray_image: np.ndarray, kf_idx: int, 
                       K: np.ndarray) -> Optional[Tuple[float, int]]:
        """
        Match current image with keyframe and compute relative yaw.
        
        Returns: (yaw_correction, num_inliers) or None if match failed
        """
        kf = self.keyframes[kf_idx]
        
        # Detect ORB in current frame
        keypoints_curr, descriptors_curr = self.orb.detectAndCompute(gray_image, None)
        
        if descriptors_curr is None or len(keypoints_curr) < 30:
            return None
            
        # Match descriptors
        try:
            matches = self.matcher.knnMatch(kf['descriptors'], descriptors_curr, k=2)
        except cv2.error:
            return None
            
        # Apply Lowe's ratio test
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # Check minimum matches
        if len(good_matches) < self.min_inliers:
            self.stats['loop_rejected_few_matches'] += 1
            return None
            
        # Get matched point coordinates
        pts_kf = np.float32([kf['keypoints'][m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([keypoints_curr[m.trainIdx].pt for m in good_matches])
        
        # Compute Essential matrix with RANSAC
        try:
            E, mask = cv2.findEssentialMat(pts_kf, pts_curr, K, method=cv2.RANSAC, 
                                           prob=0.999, threshold=1.0)
        except cv2.error:
            self.stats['loop_rejected_geometry'] += 1
            return None
            
        if E is None or mask is None:
            self.stats['loop_rejected_geometry'] += 1
            return None
            
        # Count inliers
        num_inliers = int(mask.sum())
        inlier_ratio = num_inliers / len(good_matches)
        
        if num_inliers < self.min_inliers or inlier_ratio < self.min_match_ratio:
            self.stats['loop_rejected_geometry'] += 1
            return None
            
        # Recover pose from Essential matrix
        try:
            retval, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_kf, pts_curr, K, mask=mask)
        except cv2.error:
            self.stats['loop_rejected_geometry'] += 1
            return None
            
        if retval < self.min_inliers // 2:
            self.stats['loop_rejected_geometry'] += 1
            return None
            
        # Extract yaw from rotation matrix
        # R_rel transforms from keyframe to current frame
        # Yaw = atan2(R[1,0], R[0,0]) for Z-up convention
        yaw_rel = np.arctan2(R_rel[1, 0], R_rel[0, 0])
        
        return yaw_rel, num_inliers
    
    def check_loop_closure(self, frame_idx: int, position: np.ndarray, yaw: float,
                           gray_image: np.ndarray, K: np.ndarray) -> Optional[Tuple[float, int, int]]:
        """
        Check for loop closure and return yaw correction if found.
        
        Args:
            frame_idx: Current frame index
            position: Current estimated position [x, y, z]
            yaw: Current estimated yaw [rad]
            gray_image: Current grayscale image
            K: Camera intrinsic matrix
            
        Returns:
            (yaw_correction, num_inliers, loop_kf_idx) or None if no loop detected
        """
        self.stats['loop_checks'] += 1
        
        # Find candidate keyframes
        candidates = self.find_loop_candidates(position, frame_idx)
        
        if len(candidates) == 0:
            return None
            
        self.stats['loop_candidates'] += len(candidates)
        
        # Try matching with each candidate (closest first)
        candidates_with_dist = [(i, np.linalg.norm(position[:2] - self.keyframes[i]['position'][:2])) 
                                for i in candidates]
        candidates_with_dist.sort(key=lambda x: x[1])
        
        for kf_idx, dist in candidates_with_dist[:3]:  # Try top 3 closest
            result = self.match_keyframe(gray_image, kf_idx, K)
            
            if result is not None:
                yaw_rel, num_inliers = result
                kf = self.keyframes[kf_idx]
                
                # Compute yaw correction
                # Current yaw should be: kf['yaw'] + yaw_rel
                # Error = current_yaw - expected_yaw
                expected_yaw = kf['yaw'] + yaw_rel
                yaw_error = yaw - expected_yaw
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # Wrap to [-π, π]
                
                # Yaw correction = -error
                yaw_correction = -yaw_error
                
                self.stats['loop_detected'] += 1
                
                print(f"[LOOP] ✓ DETECTED! frame={frame_idx} matched with kf={kf['frame_idx']}")
                print(f"       dist={dist:.1f}m, inliers={num_inliers}, "
                      f"yaw_rel={np.degrees(yaw_rel):.1f}°, correction={np.degrees(yaw_correction):.1f}°")
                
                return yaw_correction, num_inliers, kf_idx
                
        return None
    
    def apply_yaw_correction(self, kf: 'ExtendedKalmanFilter', yaw_correction: float,
                             sigma_yaw: float = 0.1) -> bool:
        """
        Apply yaw correction to EKF state.
        
        Uses a measurement update approach:
        - Treats loop closure as a yaw measurement
        - Correction magnitude limited by Kalman gain
        
        Args:
            kf: ExtendedKalmanFilter instance
            yaw_correction: Yaw correction in radians
            sigma_yaw: Measurement noise (smaller = more trust in loop closure)
            
        Returns:
            True if correction applied successfully
        """
        # Get current yaw from state
        q_state = kf.x[6:10, 0]
        current_yaw = quaternion_to_yaw(q_state)
        
        # Compute measured yaw (= current + correction)
        measured_yaw = current_yaw + yaw_correction
        measured_yaw = np.arctan2(np.sin(measured_yaw), np.cos(measured_yaw))
        
        # Set up measurement update
        num_clones = (kf.x.shape[0] - 16) // 7
        err_dim = 15 + 6 * num_clones
        theta_cov_idx = 8  # δθ_z in error state
        
        def h_loop_fun(x):
            H = np.zeros((1, err_dim), dtype=float)
            H[0, theta_cov_idx] = 1.0  # Yaw measurement
            return H
        
        def hx_loop_fun(x):
            q_x = x[6:10, 0]
            yaw_x = quaternion_to_yaw(q_x)
            return np.array([[yaw_x]])
        
        # Measurement noise (small = trust loop closure more)
        R_loop = np.array([[sigma_yaw**2]])
        
        # Angle residual function
        def angle_residual(a, b):
            res = a - b
            return np.arctan2(np.sin(res), np.cos(res))
        
        try:
            kf.update(
                z=np.array([[measured_yaw]]),
                HJacobian=h_loop_fun,
                Hx=hx_loop_fun,
                R=R_loop,
                residual=angle_residual
            )
            
            self.stats['yaw_corrections_applied'] += 1
            self.stats['total_yaw_correction'] += abs(yaw_correction)
            
            return True
            
        except Exception as e:
            print(f"[LOOP] Failed to apply yaw correction: {e}")
            return False
    
    def print_stats(self):
        """Print loop closure statistics."""
        print(f"[LOOP] Statistics:")
        print(f"  Keyframes added: {self.stats['keyframes_added']}")
        print(f"  Loop checks: {self.stats['loop_checks']}")
        print(f"  Loop candidates found: {self.stats['loop_candidates']}")
        print(f"  Loops detected: {self.stats['loop_detected']}")
        print(f"  Rejected (few matches): {self.stats['loop_rejected_few_matches']}")
        print(f"  Rejected (geometry): {self.stats['loop_rejected_geometry']}")
        print(f"  Yaw corrections applied: {self.stats['yaw_corrections_applied']}")
        if self.stats['yaw_corrections_applied'] > 0:
            avg_corr = np.degrees(self.stats['total_yaw_correction'] / self.stats['yaw_corrections_applied'])
            print(f"  Average yaw correction: {avg_corr:.1f}°")


# Global loop closure detector (initialized in main)
_LOOP_DETECTOR: Optional[LoopClosureDetector] = None


def init_loop_closure(position_threshold: float = 50.0) -> LoopClosureDetector:
    """Initialize global loop closure detector."""
    global _LOOP_DETECTOR
    _LOOP_DETECTOR = LoopClosureDetector(position_threshold=position_threshold)
    return _LOOP_DETECTOR


def get_loop_detector() -> Optional[LoopClosureDetector]:
    """Get global loop closure detector."""
    return _LOOP_DETECTOR


# ===============
# ESKF (Error-State Kalman Filter) - OpenVINS style
# ===============

def quat_multiply(q1, q2):
    """Quaternion multiplication: q1 * q2, both in [w,x,y,z] format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_normalize(q):
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm

# ENU→NED frame transformation matrix
# ENU: X=East, Y=North, Z=Up (Xsens IMU uses this)
# NED: X=North, Y=East, Z=Down (our state uses this)
# Transform: x_ned = y_enu, y_ned = x_enu, z_ned = -z_enu
R_NED_FROM_ENU = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1]
], dtype=np.float64)

def quat_enu_to_ned(q_enu_xyzw):
    """
    Convert quaternion from ENU frame to NED frame.
    
    Xsens MTi-30 outputs orientation in ENU frame (Z-Up).
    Our VIO state uses NED-like frame (Z-Down).
    
    Args:
        q_enu_xyzw: Quaternion [x,y,z,w] representing body-to-ENU rotation
    
    Returns:
        q_ned_xyzw: Quaternion [x,y,z,w] representing body-to-NED rotation
    """
    # Get rotation matrix from ENU quaternion
    R_body_enu = R_scipy.from_quat(q_enu_xyzw).as_matrix()
    
    # Convert to NED: R_body_ned = R_NED_FROM_ENU @ R_body_enu
    R_body_ned = R_NED_FROM_ENU @ R_body_enu
    
    # Convert back to quaternion [x,y,z,w]
    q_ned_xyzw = R_scipy.from_matrix(R_body_ned).as_quat()
    
    return q_ned_xyzw

def quat_to_rot(q):
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ])

def rot_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))

def small_angle_quat(dtheta):
    """
    Convert small angle rotation vector (3D) to quaternion.
    For small angles: q ≈ [1, θx/2, θy/2, θz/2]
    Uses exact formula for better accuracy.
    """
    theta = np.linalg.norm(dtheta)
    if theta < 1e-8:
        # First-order approximation
        return quat_normalize(np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2]))
    else:
        # Exact formula: q = [cos(θ/2), sin(θ/2) * axis]
        half_theta = theta / 2
        axis = dtheta / theta
        return np.array([
            np.cos(half_theta),
            np.sin(half_theta) * axis[0],
            np.sin(half_theta) * axis[1],
            np.sin(half_theta) * axis[2]
        ])

def quat_boxplus(q, dtheta):
    """
    Quaternion box-plus operation (manifold update).
    q_new = q ⊕ δθ = q ⊗ exp(δθ)
    where exp(δθ) converts 3D rotation vector to quaternion.
    """
    dq = small_angle_quat(dtheta)
    return quat_normalize(quat_multiply(q, dq))

def quat_boxminus(q1, q2):
    """
    Quaternion box-minus operation (manifold difference).
    δθ = q1 ⊖ q2 = log(q2^{-1} ⊗ q1)
    Returns 3D rotation vector.
    """
    # q2^{-1} = [w, -x, -y, -z] for unit quaternion
    q2_inv = np.array([q2[0], -q2[1], -q2[2], -q2[3]])
    dq = quat_multiply(q2_inv, q1)
    # log(dq) = 2 * atan2(||v||, w) * v/||v||
    w, x, y, z = dq
    vec_norm = np.sqrt(x*x + y*y + z*z)
    if vec_norm < 1e-8:
        return np.array([0.0, 0.0, 0.0])
    angle = 2.0 * np.arctan2(vec_norm, w)
    return angle * np.array([x, y, z]) / vec_norm

def skew_symmetric(v):
    """
    Create skew-symmetric matrix from 3D vector.
    [v]× such that [v]× @ u = v × u (cross product)
    
    [v]× = [ 0   -vz   vy ]
           [ vz   0   -vx ]
           [-vy   vx   0  ]
    """
    return np.array([
        [0,      -v[2],  v[1]],
        [v[2],    0,    -v[0]],
        [-v[1],   v[0],   0   ]
    ], dtype=float)

# ===============================
# IMU Preintegration (Forster et al. TRO 2017)
# ===============================
class IMUPreintegration:
    """
    IMU Preintegration on Manifold (Forster et al., TRO 2017).
    
    Preintegrates IMU measurements between keyframes to compute:
      - ΔR: Preintegrated rotation (SO(3))
      - Δv: Preintegrated velocity (3D)
      - Δp: Preintegrated position (3D)
    
    Also maintains Jacobians w.r.t. biases for bias correction:
      - ∂ΔR/∂bg: (3x3) rotation Jacobian w.r.t. gyro bias
      - ∂Δv/∂bg, ∂Δv/∂ba: (3x3) velocity Jacobians
      - ∂Δp/∂bg, ∂Δp/∂ba: (3x3) position Jacobians
    
    Key advantages over naive integration:
      1. Reduced numerical error (integrate once, apply once)
      2. Fast bias correction via Jacobians (no re-integration)
      3. Proper manifold handling for rotation (SO(3))
    
    Usage:
        preint = IMUPreintegration(bg_init, ba_init, cov_params)
        for imu in imu_buffer:
            preint.integrate_measurement(imu.w, imu.a, dt)
        delta_R, delta_v, delta_p = preint.get_deltas()
    """
    
    def __init__(self, bg: np.ndarray, ba: np.ndarray, 
                 sigma_g: float, sigma_a: float, sigma_bg: float, sigma_ba: float):
        """
        Initialize preintegration with initial biases and noise parameters.
        
        Args:
            bg: Gyroscope bias (3D)
            ba: Accelerometer bias (3D)
            sigma_g: Gyroscope measurement noise std (rad/s)
            sigma_a: Accelerometer measurement noise std (m/s²)
            sigma_bg: Gyroscope bias random walk std (rad/s/√s)
            sigma_ba: Accelerometer bias random walk std (m/s²/√s)
        """
        # Linearization point (bias at start of preintegration)
        self.bg_lin = np.copy(bg).reshape(3,)
        self.ba_lin = np.copy(ba).reshape(3,)
        
        # Noise parameters
        self.sigma_g = sigma_g
        self.sigma_a = sigma_a
        self.sigma_bg = sigma_bg
        self.sigma_ba = sigma_ba
        
        # Preintegrated measurements (identity/zero)
        self.delta_R = np.eye(3, dtype=float)  # SO(3)
        self.delta_v = np.zeros(3, dtype=float)
        self.delta_p = np.zeros(3, dtype=float)
        
        # Jacobians w.r.t. biases (for fast bias correction)
        self.J_R_bg = np.zeros((3, 3), dtype=float)  # ∂ΔR/∂bg (rotation manifold: 3D)
        self.J_v_bg = np.zeros((3, 3), dtype=float)  # ∂Δv/∂bg
        self.J_v_ba = np.zeros((3, 3), dtype=float)  # ∂Δv/∂ba
        self.J_p_bg = np.zeros((3, 3), dtype=float)  # ∂Δp/∂bg
        self.J_p_ba = np.zeros((3, 3), dtype=float)  # ∂Δp/∂ba
        
        # Preintegration covariance (9x9: rotation 3D + velocity 3D + position 3D)
        self.cov = np.zeros((9, 9), dtype=float)
        
        # Total integration time
        self.dt_sum = 0.0
        
    def reset(self, bg: np.ndarray, ba: np.ndarray):
        """Reset preintegration to identity/zero with new linearization point."""
        self.bg_lin = np.copy(bg).reshape(3,)
        self.ba_lin = np.copy(ba).reshape(3,)
        
        self.delta_R = np.eye(3, dtype=float)
        self.delta_v = np.zeros(3, dtype=float)
        self.delta_p = np.zeros(3, dtype=float)
        
        self.J_R_bg = np.zeros((3, 3), dtype=float)
        self.J_v_bg = np.zeros((3, 3), dtype=float)
        self.J_v_ba = np.zeros((3, 3), dtype=float)
        self.J_p_bg = np.zeros((3, 3), dtype=float)
        self.J_p_ba = np.zeros((3, 3), dtype=float)
        
        self.cov = np.zeros((9, 9), dtype=float)
        self.dt_sum = 0.0
        
    def integrate_measurement(self, w_meas: np.ndarray, a_meas: np.ndarray, dt: float):
        """
        Integrate one IMU measurement (gyro + accel) over time step dt.
        
        CRITICAL: IMU accelerometer data includes gravity!
        - Stationary reading: [0, 0, -9.8] in body frame (Z-down convention)
        - Must subtract gravity in BODY frame before integration
        - Gravity magnitude from IMU_PARAMS['g_norm']
        
        Updates:
          - Preintegrated deltas (ΔR, Δv, Δp)
          - Jacobians w.r.t. biases
          - Preintegration covariance
        
        Args:
            w_meas: Gyroscope measurement (3D, rad/s)
            a_meas: Accelerometer measurement (3D, m/s²) - INCLUDES GRAVITY
            dt: Time step (seconds)
        """
        # Bias-corrected measurements (using linearization point)
        w_hat = w_meas - self.bg_lin
        a_hat = a_meas - self.ba_lin
        
        # CRITICAL FIX: Preintegration MUST compensate for gravity!
        # IMU convention (DJI): Z-axis points DOWN (NED body frame)
        # - Stationary reading: a_meas = [0, 0, -9.8] (upward support force)
        # - Free fall: a_meas = [0, 0, 0] (weightless)
        # 
        # Gravity compensation: ADD gravity to cancel measurement bias
        # - a_true = a_meas + g_body, where g_body = [0, 0, +9.8] for Z-down frame
        # - Stationary: a_true = [0,0,-9.8] + [0,0,+9.8] = [0,0,0] ✓
        # - Free fall: a_true = [0,0,0] + [0,0,+9.8] = [0,0,+9.8] (gravity acceleration)
        # 
        # Why ADD (not subtract)?
        # - Accelerometer measures SPECIFIC FORCE (= true_accel - gravity)
        # - To get true acceleration: true_accel = specific_force + gravity
        # 
        # Forster et al. TRO 2017, Eq. 5: a_k = a_meas_k - b_a + R_WB @ g_world
        # In body frame: a_body = a_meas - b_a + g_body
        g_body = np.array([0.0, 0.0, IMU_PARAMS["g_norm"]], dtype=float)  # [0, 0, +9.8]
        a_hat = a_hat + g_body  # Add gravity to get true acceleration
        
        # --- Step 1: Update rotation delta ---
        # ΔR_{k+1} = ΔR_k * Exp(ω_hat * dt)
        theta_vec = w_hat * dt
        theta = np.linalg.norm(theta_vec)
        
        if theta < 1e-8:
            # Small angle: Exp(θ) ≈ I + [θ]×
            delta_R_k1 = self.delta_R @ (np.eye(3) + skew_symmetric(theta_vec))
        else:
            # Rodrigues formula: Exp(θ) = I + sin(θ)/θ [θ]× + (1-cos(θ))/θ² [θ]×²
            axis = theta_vec / theta
            skew_axis = skew_symmetric(axis)
            Exp_theta = np.eye(3) + np.sin(theta) * skew_axis + (1 - np.cos(theta)) * (skew_axis @ skew_axis)
            delta_R_k1 = self.delta_R @ Exp_theta
        
        # Right Jacobian of SO(3) for rotation error propagation
        # Jr(θ) ≈ I for small θ, or exact formula for larger θ
        if theta < 1e-8:
            Jr = np.eye(3) - 0.5 * skew_symmetric(theta_vec)
        else:
            axis = theta_vec / theta
            skew_axis = skew_symmetric(axis)
            Jr = np.eye(3) - (1 - np.cos(theta)) / theta * skew_axis + \
                 (theta - np.sin(theta)) / theta * (skew_axis @ skew_axis)
        
        # --- Step 2: Update velocity delta ---
        # Δv_{k+1} = Δv_k + ΔR_k * a_hat * dt
        delta_v_k1 = self.delta_v + self.delta_R @ a_hat * dt
        
        # --- Step 3: Update position delta ---
        # Δp_{k+1} = Δp_k + Δv_k * dt + 0.5 * ΔR_k * a_hat * dt²
        delta_p_k1 = self.delta_p + self.delta_v * dt + 0.5 * self.delta_R @ a_hat * (dt ** 2)
        
        # --- Step 4: Update Jacobians w.r.t. biases ---
        # ∂ΔR/∂bg: rotation Jacobian (manifold: 3D tangent space)
        # J_R_bg_{k+1} = J_R_bg_k - Jr * dt
        J_R_bg_k1 = self.J_R_bg - Jr * dt
        
        # ∂Δv/∂bg: velocity depends on rotation, which depends on gyro bias
        # J_v_bg_{k+1} = J_v_bg_k - ΔR_k * [a_hat]× * J_R_bg_k * dt
        J_v_bg_k1 = self.J_v_bg - self.delta_R @ skew_symmetric(a_hat) @ self.J_R_bg * dt
        
        # ∂Δv/∂ba: velocity directly affected by accel bias
        # J_v_ba_{k+1} = J_v_ba_k - ΔR_k * dt
        J_v_ba_k1 = self.J_v_ba - self.delta_R * dt
        
        # ∂Δp/∂bg: position depends on velocity, which depends on gyro bias
        # J_p_bg_{k+1} = J_p_bg_k + J_v_bg_k * dt - 0.5 * ΔR_k * [a_hat]× * J_R_bg_k * dt²
        J_p_bg_k1 = self.J_p_bg + self.J_v_bg * dt - \
                    0.5 * self.delta_R @ skew_symmetric(a_hat) @ self.J_R_bg * (dt ** 2)
        
        # ∂Δp/∂ba: position depends on velocity, which depends on accel bias
        # J_p_ba_{k+1} = J_p_ba_k + J_v_ba_k * dt - 0.5 * ΔR_k * dt²
        J_p_ba_k1 = self.J_p_ba + self.J_v_ba * dt - 0.5 * self.delta_R * (dt ** 2)
        
        # --- Step 5: Update covariance (discrete-time propagation) ---
        # State transition matrix A (9x9: rotation 3D + velocity 3D + position 3D)
        A = np.eye(9, dtype=float)
        A[0:3, 0:3] = delta_R_k1 @ self.delta_R.T  # ΔR_{k+1} / ΔR_k
        A[3:6, 0:3] = -self.delta_R @ skew_symmetric(a_hat) * dt  # ∂Δv/∂ΔR
        A[3:6, 3:6] = np.eye(3)  # Δv propagates
        A[6:9, 0:3] = -0.5 * self.delta_R @ skew_symmetric(a_hat) * (dt ** 2)  # ∂Δp/∂ΔR
        A[6:9, 3:6] = np.eye(3) * dt  # ∂Δp/∂Δv
        A[6:9, 6:9] = np.eye(3)  # Δp propagates
        
        # Noise matrix B (9x6: gyro noise 3D + accel noise 3D)
        B = np.zeros((9, 6), dtype=float)
        B[0:3, 0:3] = Jr * dt  # Rotation noise from gyro
        B[3:6, 3:6] = self.delta_R * dt  # Velocity noise from accel
        B[6:9, 3:6] = 0.5 * self.delta_R * (dt ** 2)  # Position noise from accel
        
        # Noise covariance Q (6x6: gyro + accel)
        # CRITICAL FIX: Multiply by dt to convert continuous-time spectral density
        # to discrete-time covariance (Forster TRO 2017, Eq. 47)
        # sigma_g, sigma_a are continuous-time noise densities (rad/s/√Hz, m/s²/√Hz)
        # Without dt: covariance is 400x too large (1/0.0025 = 400 for 400Hz IMU)
        Q = np.diag([
            self.sigma_g**2 * dt, self.sigma_g**2 * dt, self.sigma_g**2 * dt,
            self.sigma_a**2 * dt, self.sigma_a**2 * dt, self.sigma_a**2 * dt
        ])
        
        # Propagate covariance: Σ_{k+1} = A * Σ_k * A^T + B * Q * B^T
        self.cov = A @ self.cov @ A.T + B @ Q @ B.T
        
        # --- Step 6: Commit updates ---
        self.delta_R = delta_R_k1
        self.delta_v = delta_v_k1
        self.delta_p = delta_p_k1
        
        self.J_R_bg = J_R_bg_k1
        self.J_v_bg = J_v_bg_k1
        self.J_v_ba = J_v_ba_k1
        self.J_p_bg = J_p_bg_k1
        self.J_p_ba = J_p_ba_k1
        
        self.dt_sum += dt
        
    def get_deltas(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get preintegrated deltas (no bias correction).
        
        Returns:
            delta_R: (3x3) rotation matrix
            delta_v: (3,) velocity vector
            delta_p: (3,) position vector
        """
        return self.delta_R, self.delta_v, self.delta_p
    
    def get_deltas_corrected(self, bg_new: np.ndarray, ba_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get bias-corrected preintegrated deltas using first-order correction.
        
        This is FAST (no re-integration) and maintains FEJ consistency.
        
        Args:
            bg_new: Current gyro bias estimate (3D)
            ba_new: Current accel bias estimate (3D)
        
        Returns:
            delta_R_corr: Bias-corrected rotation matrix
            delta_v_corr: Bias-corrected velocity vector
            delta_p_corr: Bias-corrected position vector
        """
        # Bias changes from linearization point
        dbg = bg_new - self.bg_lin
        dba = ba_new - self.ba_lin
        
        # Correct rotation: ΔR_corrected = ΔR * Exp(J_R_bg * dbg)
        delta_theta = self.J_R_bg @ dbg
        theta = np.linalg.norm(delta_theta)
        
        if theta < 1e-8:
            correction_R = np.eye(3) + skew_symmetric(delta_theta)
        else:
            axis = delta_theta / theta
            skew_axis = skew_symmetric(axis)
            correction_R = np.eye(3) + np.sin(theta) * skew_axis + \
                          (1 - np.cos(theta)) * (skew_axis @ skew_axis)
        
        delta_R_corr = self.delta_R @ correction_R
        
        # Correct velocity and position (linear correction)
        delta_v_corr = self.delta_v + self.J_v_bg @ dbg + self.J_v_ba @ dba
        delta_p_corr = self.delta_p + self.J_p_bg @ dbg + self.J_p_ba @ dba
        
        return delta_R_corr, delta_v_corr, delta_p_corr
    
    def get_covariance(self) -> np.ndarray:
        """Get preintegration covariance (9x9)."""
        return self.cov.copy()
    
    def get_jacobians(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Jacobians w.r.t. biases (for measurement model).
        
        Returns:
            J_R_bg: (3x3) ∂ΔR/∂bg
            J_v_bg: (3x3) ∂Δv/∂bg
            J_v_ba: (3x3) ∂Δv/∂ba
            J_p_bg: (3x3) ∂Δp/∂bg
            J_p_ba: (3x3) ∂Δp/∂ba
        """
        return self.J_R_bg, self.J_v_bg, self.J_v_ba, self.J_p_bg, self.J_p_ba


def compute_error_state_jacobian(q, a_corr, w_corr, dt, R_body_to_world):
    """
    Compute error-state transition matrix Φ for ESKF propagation.
    
    Error state: δx = [δp, δv, δθ, δbg, δba]^T (15 dimensions)
    
    Linearized error dynamics:
      δp_k+1 = δp_k + dt * δv_k
      δv_k+1 = δv_k - R * [a_corr]× * δθ_k - R * δba_k * dt
      δθ_k+1 = exp(-[w_corr * dt]×) * δθ_k - I * δbg_k * dt
      δbg_k+1 = δbg_k  (random walk)
      δba_k+1 = δba_k  (random walk)
    
    Args:
        q: Current quaternion [w,x,y,z]
        a_corr: Bias-corrected acceleration (body frame)
        w_corr: Bias-corrected angular velocity (body frame)
        dt: Time step
        R_body_to_world: Rotation matrix from body to world frame
    
    Returns:
        Φ: 15×15 error-state transition matrix (for core state only)
    """
    # Initialize as identity
    Phi = np.eye(15, dtype=float)
    
    # Block 1: δp depends on δv
    # δp_k+1 = δp_k + dt * δv_k
    Phi[0:3, 3:6] = np.eye(3) * dt
    
    # Block 2: δv depends on δθ and δba
    # δv_k+1 = δv_k - R * [a_corr]× * δθ_k - R * δba_k * dt
    # ∂(δv)/∂(δθ) = -R * [a_corr]×
    Phi[3:6, 6:9] = -R_body_to_world @ skew_symmetric(a_corr) * dt
    # ∂(δv)/∂(δba) = -R * dt
    Phi[3:6, 12:15] = -R_body_to_world * dt
    
    # Block 3: δθ depends on previous δθ and δbg
    # δθ_k+1 ≈ (I - [w_corr * dt]×) * δθ_k - I * δbg_k * dt
    # For small angles: exp(-[ω*dt]×) ≈ I - [ω*dt]×
    theta_vec = w_corr * dt
    Phi[6:9, 6:9] = np.eye(3) - skew_symmetric(theta_vec)
    # ∂(δθ)/∂(δbg) = -I * dt
    Phi[6:9, 9:12] = -np.eye(3) * dt
    
    # Blocks 4 & 5: Biases are random walk (already identity)
    # δbg_k+1 = δbg_k
    # δba_k+1 = δba_k
    
    return Phi

def compute_error_state_process_noise(dt, estimate_imu_bias, t, t0):
    """
    Compute process noise Q for error-state covariance (15×15).
    
    Error state noise model:
      Q_δp   = (σ_acc * dt²/2)² * I  (from double integration)
      Q_δv   = (σ_acc * dt)² * I     (from single integration)
      Q_δθ   = (σ_gyro * dt)² * I    (from rotation integration)
      Q_δbg  = σ_bg_rw² * dt * I     (bias random walk)
      Q_δba  = σ_ba_rw² * dt * I     (bias random walk)
    
    Args:
        dt: Time step
        estimate_imu_bias: Whether bias was pre-estimated
        t: Current time
        t0: Start time
    
    Returns:
        Q: 15×15 process noise covariance matrix
    """
    # =====================================================================
    # ENHANCED PROCESS NOISE FOR HELICOPTER VIO
    # =====================================================================
    # Problem: Original Q used only IMU noise, leading to covariance underestimation
    # Solution: Add unmodeled dynamics noise (vibration, wind, rotor effects)
    #
    # Reference: SIGMA_ACCEL represents unmodeled acceleration (~0.3-0.8 m/s²)
    # This accounts for:
    # - Helicopter vibration (5-50 Hz, amplitude ~1-5 m/s²)
    # - Wind gusts (random acceleration ~0.5-2 m/s²)
    # - Rotor wake effects (random position/velocity disturbances)
    # =====================================================================
    
    # IMU noise (sensor noise)
    imu_acc_noise = IMU_PARAMS['acc_n']  # ~0.08 m/s²/√Hz
    imu_gyr_noise = IMU_PARAMS['gyr_n']  # ~0.004 rad/s/√Hz
    
    # Unmodeled dynamics noise (SIGMA_ACCEL from config ~0.3-0.8 m/s²)
    # This is MUCH larger than IMU noise and dominates in practice
    unmodeled_acc = SIGMA_ACCEL  # ~0.3 m/s² for helicopter
    
    # Combined acceleration noise: sqrt(imu² + unmodeled²)
    # Unmodeled dominates: sqrt(0.08² + 0.3²) ≈ 0.31
    combined_acc_noise = np.sqrt(imu_acc_noise**2 + unmodeled_acc**2)
    
    # Position noise from velocity integration error
    # Include BOTH sensor noise AND unmodeled dynamics
    q_pos = (combined_acc_noise * dt**2 / 2)**2
    
    # Velocity noise from acceleration measurement noise + unmodeled dynamics
    q_vel = (combined_acc_noise * dt)**2
    
    # Rotation noise from gyro measurement noise
    # MEMS gyro bias instability is typically ~0.0003 rad/s (1 deg/min)
    # We add small unmodeled drift but NOT too large to destabilize attitude
    unmodeled_gyr = 0.002  # rad/s (~0.1 deg/s) - reduced from 0.02
    combined_gyr_noise = np.sqrt(imu_gyr_noise**2 + unmodeled_gyr**2)
    q_theta = (combined_gyr_noise * dt)**2
    
    # =====================================================================
    # CRITICAL: Minimum yaw process noise to prevent P_yaw collapse!
    # Without this, magnetometer Kalman gain drops to ~0.01 after convergence,
    # making mag updates ineffective at correcting yaw drift.
    # 
    # FIX 2025-12-01c: Increased from 5.0 to 8.0 deg/sqrt(Hz)
    # Problem: With K_MIN=0.30, P_yaw needs to be higher to maintain K
    # P_yaw = K * R / (1-K) = 0.30 * 8.6² / 0.70 = 31.6 deg²
    # With MIN_YAW_PROCESS_NOISE=8.0, P_yaw grows by (8*√0.0025)² = 0.16°²/step
    # At 400Hz, between 20Hz mag: 20*0.16 = 3.2°² growth → adequate
    # =====================================================================
    MIN_YAW_PROCESS_NOISE = np.radians(8.0)  # 8.0 deg/sqrt(Hz) - FIX: increased from 5.0
    q_theta_z_min = (MIN_YAW_PROCESS_NOISE * np.sqrt(dt))**2
    
    # Bias random walk with adaptive tuning
    if not estimate_imu_bias:
        # Aggressive online estimation: decay over time
        time_elapsed = t - t0
        decay_factor = max(0.1, 1.0 - time_elapsed / 60.0)
        q_bg_base = IMU_PARAMS['gyr_w']
        q_ba_base = IMU_PARAMS['acc_w']
        q_bg = (q_bg_base * decay_factor)**2 * dt
        q_ba = (q_ba_base * decay_factor)**2 * dt
    else:
        # Conservative: bias already estimated
        q_bg = (IMU_PARAMS['gyr_w']**2) * dt
        q_ba = (IMU_PARAMS['acc_w']**2) * dt
    
    # Build 15×15 matrix
    Q = np.zeros((15, 15), dtype=float)
    Q[0:3, 0:3] = np.eye(3) * q_pos      # δp
    Q[3:6, 3:6] = np.eye(3) * q_vel      # δv
    Q[6:9, 6:9] = np.eye(3) * q_theta    # δθ (3D rotation!)
    
    # CRITICAL: Apply minimum yaw process noise (index 8 is δθ_z in error state)
    Q[8, 8] = max(Q[8, 8], q_theta_z_min)
    
    Q[9:12, 9:12] = np.eye(3) * q_bg     # δbg
    Q[12:15, 12:15] = np.eye(3) * q_ba   # δba
    
    return Q

def ensure_covariance_valid(P: np.ndarray, label: str = "", 
                            symmetrize: bool = True, 
                            check_psd: bool = True,
                            min_eigenvalue: float = 1e-9,
                            log_condition: bool = False) -> np.ndarray:
    """
    Ensure covariance matrix is valid (symmetric + positive semi-definite).
    
    Numerical errors during propagation/update can cause:
    1. Asymmetry: P ≠ P^T (floating-point rounding)
    2. Negative eigenvalues: loss of PSD property (catastrophic)
    
    Fixes:
    1. Symmetrization: P ← (P + P^T) / 2
    2. PSD enforcement: If λ_min < 0, add jitter P ← P + ε*I
    
    Args:
        P: Covariance matrix (n×n)
        label: Debug label for logging
        symmetrize: Force symmetry
        check_psd: Check and fix negative eigenvalues
        min_eigenvalue: Minimum allowed eigenvalue (add jitter if below)
        log_condition: Log condition number (for debugging)
    
    Returns:
        P_valid: Fixed covariance matrix
    """
    n = P.shape[0]
    
    # 1. Symmetrization (cheap, always recommended)
    if symmetrize:
        asymmetry = np.linalg.norm(P - P.T, ord='fro')
        if asymmetry > 1e-6:
            print(f"[COV_CHECK] {label}: Asymmetry detected (||P - P^T|| = {asymmetry:.3e}), symmetrizing")
        P = (P + P.T) / 2.0
    
    # 2. PSD check (expensive, optional)
    if check_psd:
        try:
            eigvals = np.linalg.eigvalsh(P)  # Symmetric eigenvalue solver (faster)
            lambda_min = eigvals[0]
            lambda_max = eigvals[-1]
            
            # Log condition number (optional)
            if log_condition and lambda_min > 1e-12:
                cond = lambda_max / lambda_min
                if cond > 1e10:
                    print(f"[COV_CHECK] {label}: High condition number κ(P) = {cond:.3e}")
            
            # Fix negative eigenvalues
            if lambda_min < min_eigenvalue:
                jitter = abs(lambda_min) + min_eigenvalue
                print(f"[COV_CHECK] {label}: Negative eigenvalue λ_min = {lambda_min:.3e}, "
                      f"adding jitter ε = {jitter:.3e}")
                P = P + jitter * np.eye(n, dtype=float)
                
                # Verify fix worked
                eigvals_new = np.linalg.eigvalsh(P)
                if eigvals_new[0] < 0:
                    print(f"[COV_CHECK] {label}: WARNING - Jitter failed to fix PSD! "
                          f"λ_min = {eigvals_new[0]:.3e}")
        
        except np.linalg.LinAlgError as e:
            print(f"[COV_CHECK] {label}: Eigenvalue computation failed: {e}")
            # Emergency fix: add large jitter
            P = P + 1e-6 * np.eye(n, dtype=float)
    
    return P


def propagate_error_state_covariance(P, Phi, Q, num_clones):
    """
    Propagate error-state covariance with camera clones.
    
    Full error state: [δp, δv, δθ, δbg, δba, δθ_C1, δp_C1, δθ_C2, δp_C2, ...]
    Dimensions: 15 (core) + 6*num_clones
    
    Clones are STATIC (no dynamics), so:
      Φ_full = [ Φ_core   0       ]  (15+6M) × (15+6M)
               [ 0        I_clones]
      
      Q_full = [ Q_core   0       ]  (15+6M) × (15+6M)
               [ 0        0       ]  (clones have no process noise)
    
    Args:
        P: Current error-state covariance (15+6M) × (15+6M)
        Phi: Core error-state transition (15×15)
        Q: Core process noise (15×15)
        num_clones: Number of camera clones
    
    Returns:
        P_new: Propagated covariance
    """
    err_dim = 15 + 6 * num_clones
    
    # Build full Φ matrix
    Phi_full = np.eye(err_dim, dtype=float)
    Phi_full[0:15, 0:15] = Phi  # Core dynamics
    # Clones remain identity (static)
    
    # Build full Q matrix
    Q_full = np.zeros((err_dim, err_dim), dtype=float)
    Q_full[0:15, 0:15] = Q  # Core process noise
    # Clones have no process noise
    
    # Propagate: P_k+1 = Φ * P_k * Φ^T + Q
    P_new = Phi_full @ P @ Phi_full.T + Q_full
    
    # Ensure symmetry (numerical stability)
    P_new = (P_new + P_new.T) / 2
    
    # =====================================================================
    # COVARIANCE FLOOR: Prevent overconfident state estimates
    # =====================================================================
    # Problem: After measurement updates (ZUPT, VPS, DEM), covariance can
    # become very small, leading to:
    # 1. Rejection of valid future measurements (outlier gating)
    # 2. Underestimation of uncertainty (filter divergence)
    #
    # Solution: Enforce minimum variance on position and velocity
    # Typical values for helicopter at ~40 m/s:
    # - Position: σ_min = 1.0 m (uncertainty in position)
    # - Velocity: σ_min = 0.5 m/s (uncertainty in velocity)
    # =====================================================================
    
    # Minimum variances (squared standard deviations)
    P_pos_min = 1.0**2  # 1 meter std
    P_vel_min = 0.5**2  # 0.5 m/s std
    
    # CRITICAL: Roll/Pitch vs Yaw have DIFFERENT observability!
    # - Roll/Pitch: Observable via gravity vector → can converge to low uncertainty
    # - Yaw: NOT observable without mag/GPS → allow convergence with working mag
    #
    # LESSON LEARNED: High P_yaw_min (5°) → K=0.25 → yaw oscillates wildly due to mag noise
    # → altitude drift! Better to let P_yaw converge → K decreases → stable yaw → good altitude
    #
    # With working mag: P_yaw converges → K decreases → yaw tracks mag smoothly
    # With bad mag: P_yaw converges → K decreases → yaw drifts but slowly (acceptable)
    P_rollpitch_min = 0.02**2  # 0.02 rad (~1.1°) std - roll/pitch observable via gravity
    P_yaw_min = 0.02**2        # 0.02 rad (~1.1°) std - let yaw converge naturally
    
    # Apply floor to position covariance (indices 0:3)
    for i in range(3):
        if P_new[i, i] < P_pos_min:
            P_new[i, i] = P_pos_min
    
    # Apply floor to velocity covariance (indices 3:6)
    for i in range(3, 6):
        if P_new[i, i] < P_vel_min:
            P_new[i, i] = P_vel_min
    
    # Apply floor to rotation covariance (indices 6:9)
    # Roll (idx 6) and Pitch (idx 7) use lower floor
    # Yaw (idx 8) uses higher floor for mag observability
    for i in range(6, 8):  # Roll, Pitch
        if P_new[i, i] < P_rollpitch_min:
            P_new[i, i] = P_rollpitch_min
    if P_new[8, 8] < P_yaw_min:  # Yaw only
        P_new[8, 8] = P_yaw_min
    
    return P_new

class ExtendedKalmanFilter(object):
    """
    Error-State Kalman Filter (ESKF) for VIO - OpenVINS style.
    
    State vector layout (nominal state):
      - Core IMU state (16 elements):
        * p (0:3): position [m]
        * v (3:6): velocity [m/s]
        * q (6:10): quaternion [w,x,y,z] (unit quaternion)
        * bg (10:13): gyro bias [rad/s]
        * ba (13:16): accel bias [m/s²]
      
      - Sliding window camera clones (7 elements each):
        * q_C (0:4): camera quaternion [w,x,y,z]
        * p_C (4:7): camera position [m]
    
    Covariance: error-state (reduced dimension)
      - Core IMU error state (15 elements):
        * δp (0:3): position error
        * δv (3:6): velocity error
        * δθ (6:9): rotation error (3D, NOT 4D quaternion!)
        * δbg (9:12): gyro bias error
        * δba (12:15): accel bias error
      
      - Camera clone error state (6 elements each):
        * δθ_C (0:3): rotation error
        * δp_C (3:6): position error
    
    Key differences from standard EKF:
    1. Quaternion stored in nominal state (4D) but covariance uses rotation vector (3D)
    2. Updates use multiplicative quaternion correction: q_new = q ⊗ exp(δθ)
    3. All Jacobians must map quaternion to rotation vector
    """
    def __init__(self, dim_x, dim_z, dim_u=0):

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1)) # state
        self.P = eye(dim_x)        # uncertainty covariance
        self.B = 0                 # control transition matrix
        self.F = np.eye(dim_x)     # state transition matrix
        self.R = eye(dim_z)        # state uncertainty
        self.Q = eye(dim_x)        # process uncertainty
        self.y = zeros((dim_z, 1)) # residual

        z = np.array([None]*self.dim_z)
        self.z = reshape_z(z, self.dim_z, self.x.ndim)

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros(self.x.shape) # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))   # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):
        """ Performs the predict/update innovation of the extended Kalman
        filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.
            If `None`, only predict step is perfomed.

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, along with the
           optional arguments in args, and returns H.

        Hx : function
            function which takes as input the state variable (self.x) along
            with the optional arguments in hx_args, and returns the measurement
            that would correspond to that state.

        args : tuple, optional, default (,)
            arguments to be passed into HJacobian after the required state
            variable.

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx after the required state
            variable.

        u : np.array or scalar
            optional control vector input to the filter.
        """
        #pylint: disable=too-many-locals

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        F = self.F
        B = self.B
        P = self.P
        Q = self.Q
        R = self.R
        x = self.x

        H = HJacobian(x, *args)

        # predict step
        x = dot(F, x) + dot(B, u)
        P = dot(F, P).dot(F.T) + Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

        # update step
        PHT = dot(P, H.T)
        self.S = dot(H, PHT) + R
        self.SI = linalg.inv(self.S)
        self.K = dot(PHT, self.SI)

        self.y = z - Hx(x, *hx_args)
        self.x = x + dot(self.K, self.y)

        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
               residual=np.subtract):
        """ Performs the update innovation of the extended Kalman filter with ESKF.

        Parameters
        ----------

        z : np.array
            measurement for this step.
            If `None`, posterior is not computed

        HJacobian : function
           function which computes the Jacobian of the H matrix w.r.t. ERROR state.
           CRITICAL: Must return derivatives w.r.t. 3D rotation vector (δθ), NOT 4D quaternion!

        Hx : function
            function which takes as input the state variable (self.x) along
            with the optional arguments in hx_args, and returns the measurement
            that would correspond to that state.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        args : tuple, optional, default (,)
            arguments to be passed into HJacobian after the required state
            variable. for robot localization you might need to pass in
            information about the map and time of day, so you might have
            `args=(map_data, time)`, where the signature of HCacobian will
            be `def HJacobian(x, map, t)`

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state
            variable.

        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """

        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        H = HJacobian(self.x, *args)

        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        
        try:
            self.K = PHT.dot(linalg.inv(self.S))
        except np.linalg.LinAlgError:
            print("[ESKF] WARNING: Singular S matrix, rejecting update")
            self.z = deepcopy(z)
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        hx = Hx(self.x, *hx_args)
        self.y = residual(z, hx)
        
        # ESKF: Compute error-state correction
        dx = dot(self.K, self.y)
        
        # DEBUG: Print yaw correction if significant
        if dx.shape[0] > 8 and abs(dx[8, 0]) > 0.001:  # >0.06 degrees
            print(f"[ESKF-DEBUG] dx[8] (yaw)={np.degrees(dx[8,0]):.2f}°, y={np.degrees(self.y[0,0]):.2f}°, K[8]={self.K[8,0]:.4f}")
        
        # Apply correction with proper manifold update
        self._apply_error_state_correction(dx)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        # NOTE: For ESKF, P is error-state covariance, so use error-state dimensions
        I_err = np.eye(self.P.shape[0])  # Error state identity
        I_KH = I_err - dot(self.K, H)
        self.P = dot(I_KH, self.P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)
        
        # =====================================================================
        # COVARIANCE FLOOR: Enforce after every update to prevent overconfidence
        # CRITICAL: Roll/Pitch vs Yaw have DIFFERENT observability!
        # =====================================================================
        P_pos_min = 1.0**2  # 1 meter std
        P_vel_min = 0.5**2  # 0.5 m/s std
        # Roll/Pitch: Observable via gravity vector → low floor OK
        # Yaw: Let converge naturally - high K causes yaw oscillation!
        P_rollpitch_min = 0.02**2  # 0.02 rad (~1.1°) std - observable via gravity
        P_yaw_min = 0.02**2        # 0.02 rad (~1.1°) std - let converge naturally
        
        for i in range(min(3, self.P.shape[0])):  # Position
            if self.P[i, i] < P_pos_min:
                self.P[i, i] = P_pos_min
        for i in range(3, min(6, self.P.shape[0])):  # Velocity
            if self.P[i, i] < P_vel_min:
                self.P[i, i] = P_vel_min
        # Roll (idx 6), Pitch (idx 7) - lower floor
        for i in range(6, min(8, self.P.shape[0])):
            if self.P[i, i] < P_rollpitch_min:
                self.P[i, i] = P_rollpitch_min
        # Yaw (idx 8) - same floor, let converge naturally
        if self.P.shape[0] > 8 and self.P[8, 8] < P_yaw_min:
            self.P[8, 8] = P_yaw_min

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
    
    def _apply_error_state_correction(self, dx):
        """
        Apply error-state correction to nominal state with proper manifold updates.
        
        ESKF correction model:
          - Additive for position, velocity, biases: x_new = x + δx
          - Multiplicative for quaternion: q_new = q ⊗ exp(δθ)
        
        dx layout (error state):
          - Core: [δp (3), δv (3), δθ (3), δbg (3), δba (3)] = 15 elements
          - Clones: [δθ_C (3), δp_C (3)] per clone = 6 elements each
        """
        # Extract core error corrections
        dp = dx[0:3, 0]
        dv = dx[3:6, 0]
        dtheta = dx[6:9, 0]
        dbg = dx[9:12, 0]
        dba = dx[12:15, 0]
        
        # Apply additive corrections
        self.x[0:3, 0] += dp   # position
        self.x[3:6, 0] += dv   # velocity
        self.x[10:13, 0] += dbg  # gyro bias
        self.x[13:16, 0] += dba  # accel bias
        
        # Apply multiplicative correction to quaternion: q_new = q ⊗ exp(δθ)
        q_old = self.x[6:10, 0]
        q_new = quat_boxplus(q_old, dtheta)
        self.x[6:10, 0] = q_new
        
        # Apply corrections to camera clones
        num_clones = (self.x.shape[0] - 16) // 7
        for i in range(num_clones):
            # Error state indices
            err_base = 15 + i * 6
            dtheta_c = dx[err_base:err_base+3, 0]
            dp_c = dx[err_base+3:err_base+6, 0]
            
            # Nominal state indices
            nom_base = 16 + i * 7
            q_c_old = self.x[nom_base:nom_base+4, 0]
            
            # Update clone quaternion
            q_c_new = quat_boxplus(q_c_old, dtheta_c)
            self.x[nom_base:nom_base+4, 0] = q_c_new
            
            # Update clone position
            self.x[nom_base+4:nom_base+7, 0] += dp_c

    def predict_x(self, u=0):
        """
        Predicts the next state of X. If you need to
        compute the next state yourself, override this function. You would
        need to do this, for example, if the usual Taylor expansion to
        generate F is not providing accurate results for you.
        """
        self.x = dot(self.F, self.x) + dot(self.B, u)

    def predict(self, u=0):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.

        Parameters
        ----------

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        self.predict_x(u)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)


    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """

        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """
        Mahalanobis distance of innovation. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(float(dot(dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis

    def __repr__(self):
        return '\n'.join([
            'KalmanFilter object',
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('F', self.F),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('K', self.K),
            pretty_str('y', self.y),
            pretty_str('S', self.S),
            pretty_str('likelihood', self.likelihood),
            pretty_str('log-likelihood', self.log_likelihood),
            pretty_str('mahalanobis', self.mahalanobis)
            ])

# ===============================
# Projections / DEM sampling
# ===============================
_proj_cache = {"origin": None, "to_xy": None, "to_ll": None}

def ensure_local_proj(origin_lat: float, origin_lon: float):
    global _proj_cache
    if _proj_cache["origin"] != (origin_lat, origin_lon) or _proj_cache["to_xy"] is None:
        crs_wgs84 = CRS.from_epsg(4326)
        crs_aeqd  = CRS.from_proj4(f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} +datum=WGS84 +units=m +no_defs")
        _proj_cache["to_xy"] = Transformer.from_crs(crs_wgs84, crs_aeqd, always_xy=True)
        _proj_cache["to_ll"] = Transformer.from_crs(crs_aeqd,  crs_wgs84, always_xy=True)
        _proj_cache["origin"] = (origin_lat, origin_lon)

def latlon_to_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> np.ndarray:
    ensure_local_proj(origin_lat, origin_lon)
    x, y = _proj_cache["to_xy"].transform(lon, lat)
    return np.array([x, y], dtype=float)

def xy_to_latlon(px: float, py: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    ensure_local_proj(origin_lat, origin_lon)
    lon, lat = _proj_cache["to_ll"].transform(px, py)
    return float(lat), float(lon)


# ===============================
# IMU-GNSS Lever Arm Compensation
# ===============================

def gnss_to_imu_position(p_gnss_enu: np.ndarray, R_body_to_world: np.ndarray) -> np.ndarray:
    """
    Convert GNSS antenna position to IMU position using lever arm.
    
    The GNSS antenna is offset from IMU in body frame by IMU_GNSS_LEVER_ARM.
    PPK ground truth gives GNSS antenna position, but VIO state tracks IMU position.
    
    Formula: p_imu = p_gnss - R_body_to_world @ lever_arm
    
    Args:
        p_gnss_enu: GNSS position in ENU world frame [x, y, z] meters
        R_body_to_world: Rotation matrix from body (FRD) to world (ENU)
        
    Returns:
        p_imu_enu: IMU position in ENU world frame [x, y, z] meters
    """
    lever_arm_world = R_body_to_world @ IMU_GNSS_LEVER_ARM
    p_imu = p_gnss_enu - lever_arm_world
    return p_imu


def imu_to_gnss_position(p_imu_enu: np.ndarray, R_body_to_world: np.ndarray) -> np.ndarray:
    """
    Convert IMU position to GNSS antenna position using lever arm.
    
    Inverse of gnss_to_imu_position().
    Formula: p_gnss = p_imu + R_body_to_world @ lever_arm
    
    Args:
        p_imu_enu: IMU position in ENU world frame [x, y, z] meters
        R_body_to_world: Rotation matrix from body (FRD) to world (ENU)
        
    Returns:
        p_gnss_enu: GNSS position in ENU world frame [x, y, z] meters
    """
    lever_arm_world = R_body_to_world @ IMU_GNSS_LEVER_ARM
    p_gnss = p_imu_enu + lever_arm_world
    return p_gnss

@dataclass
class DEMReader:
    ds: Optional[rasterio.io.DatasetReader]
    to_raster: Optional[Transformer]

    @classmethod
    def open(cls, path: Optional[str]):
        if not path or not os.path.exists(path):
            return cls(None, None)
        ds = rasterio.open(path)
        to_raster = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        return cls(ds, to_raster)

    def sample_m(self, lat: float, lon: float) -> Optional[float]:
        if self.ds is None:
            return None
        x, y = self.to_raster.transform(lon, lat)
        nodata = self.ds.nodatavals[0]
        try:
            for v in self.ds.sample([(x, y)]):
                h = float(v[0])
                if (h is not None) and (h != nodata) and math.isfinite(h):
                    return h
                return None
        except Exception:
            return None

# ===============================
# Inputs
# ===============================

@dataclass
class PPKInitialState:
    """Initial state values from PPK ground truth file."""
    lat: float           # degrees
    lon: float           # degrees
    height: float        # meters (ellipsoidal height, NOT MSL)
    roll: float          # radians (NED frame)
    pitch: float         # radians (NED frame)
    yaw: float           # radians (NED frame: 0=North, positive=CW)
    ve: float            # m/s East velocity (ENU)
    vn: float            # m/s North velocity (ENU)
    vu: float            # m/s Up velocity (ENU)
    timestamp: str       # GPST timestamp string


def load_ppk_initial_state(path: str) -> Optional[PPKInitialState]:
    """
    Load comprehensive initial state from PPK ground truth file (.pos).
    
    The .pos file format (MUN-FRL Bell 412):
    %GPST                    latitude(deg)  longitude(deg)  height(m)  Q  ns  sdn  sde  sdu  sdne  sdeu  sdun  age  ratio  roll  pitch  yaw(deg)  P_deg_s  Q_deg_s  R_deg_s  Ve_m_s  Vn_m_s  Vu_m_s
    
    Column indices (0-indexed, first 2 are date+time):
      0-1: GPST date/time
      2: latitude (deg)
      3: longitude (deg)  
      4: height (m) - ellipsoidal, NOT MSL
      5: Q (quality flag)
      6: ns (num satellites)
      7-12: std dev values
      13: age
      14: ratio
      15: roll (deg, NED frame)
      16: pitch (deg, NED frame)
      17: yaw (deg, NED frame: 0=North, positive=CW)
      18-20: P,Q,R angular rates (deg/s)
      21: Ve_m_s (East velocity, m/s)
      22: Vn_m_s (North velocity, m/s)
      23: Vu_m_s (Up velocity, m/s)
    
    Returns PPKInitialState or None if parsing fails.
    """
    if path is None or not os.path.exists(path):
        print(f"[WARN] PPK file not found: {path}")
        return None
    
    try:
        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('%')]
        
        if len(lines) == 0:
            print(f"[WARN] PPK file empty: {path}")
            return None
        
        # Parse first data line
        parts = lines[0].split()
        
        if len(parts) < 24:
            print(f"[WARN] PPK file has insufficient columns ({len(parts)}, need 24): {path}")
            return None
        
        # Extract values (indices for whitespace-split)
        timestamp = f"{parts[0]} {parts[1]}"
        lat = float(parts[2])
        lon = float(parts[3])
        height = float(parts[4])  # Ellipsoidal height, NOT MSL
        roll_deg = float(parts[15])
        pitch_deg = float(parts[16])
        yaw_deg = float(parts[17])
        ve = float(parts[21])  # East velocity m/s
        vn = float(parts[22])  # North velocity m/s
        vu = float(parts[23])  # Up velocity m/s
        
        state = PPKInitialState(
            lat=lat,
            lon=lon,
            height=height,
            roll=np.radians(roll_deg),
            pitch=np.radians(pitch_deg),
            yaw=np.radians(yaw_deg),
            ve=ve,
            vn=vn,
            vu=vu,
            timestamp=timestamp
        )
        
        print(f"\n[INIT][PPK] Loaded initial state from PPK ground truth:")
        print(f"  Timestamp: {timestamp}")
        print(f"  Position: lat={lat:.8f}°, lon={lon:.8f}°, h={height:.2f}m (ellipsoidal)")
        print(f"  Attitude (NED): roll={roll_deg:.2f}°, pitch={pitch_deg:.2f}°, yaw={yaw_deg:.2f}°")
        print(f"  Velocity (ENU): Ve={ve:.2f} m/s, Vn={vn:.2f} m/s, Vu={vu:.2f} m/s")
        
        return state
        
    except Exception as e:
        print(f"[WARN] Failed to load PPK initial state: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_ground_truth_initial_yaw(path: str) -> Optional[float]:
    """
    Load initial yaw (heading) from PPK ground truth file.
    (Legacy function - use load_ppk_initial_state() for comprehensive data)
    
    Returns yaw in radians (NED convention: 0=North, positive=CW) or None if not found.
    """
    state = load_ppk_initial_state(path)
    if state is not None:
        return state.yaw
    return None


def load_ppk_trajectory(path: str) -> Optional[pd.DataFrame]:
    """
    Load full PPK trajectory as DataFrame for error comparison.
    
    Returns DataFrame with columns:
      - stamp_log: Unix timestamp (float)
      - lat, lon: Position (degrees)
      - height: Ellipsoidal height (m)
      - roll, pitch, yaw: Attitude (radians, NED frame)
      - ve, vn, vu: Velocity (m/s, ENU frame)
    """
    if path is None or not os.path.exists(path):
        return None
    
    try:
        from datetime import datetime, timezone
        import calendar
        
        rows = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                
                parts = line.split()
                if len(parts) < 24:
                    continue
                
                # Parse GPST timestamp: "2022/05/25 21:07:15.600"
                # IMPORTANT: GPS time is essentially UTC, so we need to treat it as UTC
                gpst_str = f"{parts[0]} {parts[1]}"
                try:
                    dt = datetime.strptime(gpst_str, "%Y/%m/%d %H:%M:%S.%f")
                    # Convert to UTC Unix timestamp (GPS time ≈ UTC for practical purposes)
                    # Use calendar.timegm to avoid timezone issues
                    stamp = calendar.timegm(dt.timetuple()) + dt.microsecond / 1e6
                except:
                    continue
                
                try:
                    row = {
                        'stamp_log': stamp,
                        'lat': float(parts[2]),
                        'lon': float(parts[3]),
                        'height': float(parts[4]),
                        'quality': int(parts[5]),
                        'roll': np.radians(float(parts[15])),
                        'pitch': np.radians(float(parts[16])),
                        'yaw': np.radians(float(parts[17])),
                        've': float(parts[21]),
                        'vn': float(parts[22]),
                        'vu': float(parts[23]),
                    }
                    rows.append(row)
                except:
                    continue
        
        if len(rows) == 0:
            return None
        
        df = pd.DataFrame(rows)
        print(f"[PPK] Loaded {len(df)} trajectory points from {path}")
        print(f"[PPK] Time range: {df['stamp_log'].min():.3f} - {df['stamp_log'].max():.3f}")
        return df
        
    except Exception as e:
        print(f"[WARN] Failed to load PPK trajectory: {e}")
        return None


def load_msl_from_gga(path: str) -> Optional[float]:
    """
    Load initial MSL altitude from flight_log_from_gga.csv.
    This is the ONLY value we take from this file (rest comes from PPK).
    
    Returns MSL altitude in meters.
    """
    if not os.path.exists(path):
        print(f"[WARN] GGA file not found: {path}")
        return None
    
    try:
        df = pd.read_csv(path)
        msl_col = next((c for c in df.columns if "altitude_msl_m" in c.lower()), None)
        
        if msl_col is None:
            print(f"[WARN] altitude_MSL_m column not found in GGA file")
            return None
        
        msl_m = float(df[msl_col].iloc[0])
        print(f"[INIT][GGA] Initial MSL altitude: {msl_m:.2f}m (from {path})")
        return msl_m
        
    except Exception as e:
        print(f"[WARN] Failed to load MSL from GGA: {e}")
        return None


def load_quarry_initial(path: str) -> Tuple[float, float, float, np.ndarray]:
    """Return (initial_lat, initial_lon, initial_msl_meters, initial_velocity_ENU)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"flight_log_from_gga.csv not found: {path}")
    df = pd.read_csv(path)
    
    print(f"\n[DEBUG][INIT] Loading initial values from: {path}")
    print(f"[DEBUG][INIT] Total rows in flight_log: {len(df)}")
    print(f"[DEBUG][INIT] Columns found: {list(df.columns)}")
    
    # Column names for flight_log_from_gga.csv: lat_dd, lon_dd, altitude_MSL_m, xSpeed_mph, ySpeed_mph, zSpeed_mph
    lat_col = next((c for c in df.columns if c.lower().strip() in ["lat", "latitude", "lat_dd"]), None)
    lon_col = next((c for c in df.columns if c.lower().strip() in ["lon", "longitude", "lon_dd"]), None)
    msl_col = next((c for c in df.columns if "altitude_msl_m" in c.lower() or "altitude_above_sealevel" in c.lower()), None)
    
    print(f"[DEBUG][INIT] Matched columns: lat={lat_col}, lon={lon_col}, msl={msl_col}")
    
    # Try to read velocity components (in mph, need to convert to m/s)
    vx_col = next((c for c in df.columns if "xspeed" in c.lower().strip()), None)
    vy_col = next((c for c in df.columns if "yspeed" in c.lower().strip()), None)
    vz_col = next((c for c in df.columns if "zspeed" in c.lower().strip()), None)
    
    print(f"[DEBUG][INIT] Velocity columns: vx={vx_col}, vy={vy_col}, vz={vz_col}")
    
    if lat_col is None or lon_col is None or msl_col is None:
        raise ValueError("flight_log_from_gga.csv must contain lat_dd, lon_dd, and altitude_MSL_m")
    
    # Check if stamp_log exists
    stamp_col = next((c for c in df.columns if "stamp" in c.lower()), None)
    if stamp_col:
        print(f"[DEBUG][INIT] Timestamp column: {stamp_col}")
        print(f"[DEBUG][INIT] First timestamp: {df[stamp_col].iloc[0]:.3f} sec")
        print(f"[DEBUG][INIT] Last timestamp: {df[stamp_col].iloc[-1]:.3f} sec")
        print(f"[DEBUG][INIT] Duration: {df[stamp_col].iloc[-1] - df[stamp_col].iloc[0]:.3f} sec")
    
    row0 = df.iloc[0]
    lat0 = float(row0[lat_col])
    lon0 = float(row0[lon_col])
    # altitude_MSL_m is already in meters (not feet)
    msl_m = float(row0[msl_col])
    
    print(f"[DEBUG][INIT] Initial position: lat={lat0:.8f}°, lon={lon0:.8f}°, MSL={msl_m:.2f}m")
    
    # Read initial velocity (convert mph to m/s: 1 mph = 0.44704 m/s)
    # Dataset velocities are in ENU frame (xSpeed=East, ySpeed=North, zSpeed=Up)
    v_init = np.zeros(3, dtype=float)
    if vx_col is not None and vy_col is not None and vz_col is not None:
        vx_mph = float(row0[vx_col]) if pd.notna(row0[vx_col]) else 0.0
        vy_mph = float(row0[vy_col]) if pd.notna(row0[vy_col]) else 0.0
        vz_mph = float(row0[vz_col]) if pd.notna(row0[vz_col]) else 0.0
        v_init = np.array([vx_mph, vy_mph, vz_mph], dtype=float) * 0.44704  # mph to m/s
        print(f"[DEBUG][INIT] Initial velocity (mph): vx={vx_mph:.6f}, vy={vy_mph:.6f}, vz={vz_mph:.6f}")
        print(f"[DEBUG][INIT] Initial velocity (m/s): vx={v_init[0]:.6f}, vy={v_init[1]:.6f}, vz={v_init[2]:.6f}")
        v_mag_kmh = np.linalg.norm(v_init[:2]) * 3.6  # horizontal speed in km/h
        print(f"[DEBUG][INIT] Initial horizontal speed: {v_mag_kmh:.2f} km/h")
    else:
        print(f"[WARNING] Velocity columns not found in flight log CSV. Found cols with 'speed': {[c for c in df.columns if 'speed' in c.lower()]}")
    
    print(f"[DEBUG][INIT] Returning: lat={lat0:.8f}, lon={lon0:.8f}, msl={msl_m:.2f}m, v_init={v_init}\n")
    
    return lat0, lon0, msl_m, v_init

@dataclass
class IMURecord:
    t: float
    q: np.ndarray  # [x,y,z,w]
    ang: np.ndarray  # [wx,wy,wz] rad/s
    lin: np.ndarray  # [ax,ay,az] m/s^2 (body)
    
    @property
    def w(self) -> np.ndarray:
        """Alias for angular velocity (gyro)"""
        return self.ang
    
    @property
    def a(self) -> np.ndarray:
        """Alias for linear acceleration"""
        return self.lin

@dataclass
class MagRecord:
    t: float
    mag: np.ndarray  # [x,y,z] magnetic field in body frame (µT or normalized)


def load_imu_csv(path: str) -> List[IMURecord]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"IMU CSV not found: {path}")
    df = pd.read_csv(path)
    # required columns
    cols = ["stamp_log","ori_x","ori_y","ori_z","ori_w","ang_x","ang_y","ang_z","lin_x","lin_y","lin_z"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"IMU CSV missing column: {c}")
    df = df.sort_values("stamp_log").reset_index(drop=True)
    recs: List[IMURecord] = []
    for _, r in df.iterrows():
        recs.append(IMURecord(
            t=float(r["stamp_log"]),
            q=np.array([float(r["ori_x"]), float(r["ori_y"]), float(r["ori_z"]), float(r["ori_w"])], dtype=float),
            ang=np.array([float(r["ang_x"]), float(r["ang_y"]), float(r["ang_z"])], dtype=float),
            lin=np.array([float(r["lin_x"]), float(r["lin_y"]), float(r["lin_z"])], dtype=float),
        ))
    return recs

@dataclass
class ImageItem:
    t: float
    path: str


def load_images(images_dir: Optional[str], index_csv: Optional[str]) -> List[ImageItem]:
    if not images_dir or not os.path.isdir(images_dir):
        print(f"[Images] Directory not found: {images_dir}")
        return []
    if not index_csv or not os.path.exists(index_csv):
        print(f"[Images] Index CSV not found: {index_csv}")
        return []
    df = pd.read_csv(index_csv)
    # expected columns: stamp_log, filename (but be permissive)
    t_candidates = [c for c in df.columns if c.lower().strip().startswith("stamp") or c.lower().strip().startswith("time")]
    f_candidates = [c for c in df.columns if ("file" in c.lower()) or ("name" in c.lower()) or ("image" in c.lower()) or ("path" in c.lower())]
    tcol = t_candidates[0] if t_candidates else None
    fcol = f_candidates[0] if f_candidates else None
    if tcol is None or fcol is None:
        print(f"[Images] Bad headers in index CSV. Found columns: {list(df.columns)}")
        raise ValueError("images_index.csv must have time/stamp and filename/path columns")
    items: List[ImageItem] = []
    skipped = 0
    examples_missing = []
    for _, r in df.iterrows():
        try:
            ts = float(str(r[tcol]).strip())
        except Exception:
            continue
        fn_raw = str(r[fcol]).strip()
        candidates = []
        if os.path.isabs(fn_raw):
            candidates = [fn_raw]
        else:
            candidates = [os.path.join(images_dir, fn_raw), os.path.join(images_dir, os.path.basename(fn_raw)), fn_raw]
        chosen = None
        for cpath in candidates:
            if os.path.exists(cpath):
                chosen = cpath
                break
        if chosen is None:
            skipped += 1
            if len(examples_missing) < 3:
                examples_missing.append(fn_raw)
            continue
        items.append(ImageItem(ts, chosen))
    items.sort(key=lambda x: x.t)
    print(f"[Images] Indexed rows: {len(df)} | Valid files: {len(items)} | Missing: {skipped}")
    if skipped > 0:
        print(f"[Images] Missing examples (first 3): {examples_missing}")
    return items

@dataclass
class VPSItem:
    t: float
    lat: float
    lon: float


def load_vps_csv(path: Optional[str]) -> List[VPSItem]:
    if not path or not os.path.exists(path):
        return []
    items: List[VPSItem] = []
    # Try flexible readers: either csv header or simple lines "Ts,lat,lon"
    try:
        df = pd.read_csv(path)
        tcol = next((c for c in df.columns if c.lower().startswith("t") or "stamp" in c.lower()), None)
        latcol = next((c for c in df.columns if c.lower().startswith("lat")), None)
        loncol = next((c for c in df.columns if c.lower().startswith("lon")), None)
        if tcol and latcol and loncol:
            for _, r in df.iterrows():
                items.append(VPSItem(float(r[tcol]), float(r[latcol]), float(r[loncol])))
            items.sort(key=lambda x: x.t)
            return items
    except Exception:
        pass
    # raw text fallback
    with open(path, "r") as f:
        for line in f:
            p = [x.strip() for x in line.strip().split(",")]
            if len(p) < 3:  
                continue
            t_str = p[0].replace("s", "")
            try:
                items.append(VPSItem(float(t_str), float(p[1]), float(p[2])))
            except Exception:
                continue
    items.sort(key=lambda x: x.t)
    return items

def load_mag_csv(path: Optional[str]) -> List[MagRecord]:
    """Load magnetometer data from vector3.csv (stamp_log,x,y,z,frame_id)."""
    if not path or not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    # Expected columns: stamp_log, x, y, z, frame_id
    tcol = next((c for c in df.columns if 'stamp' in c.lower()), None)
    if tcol is None or 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
        print(f"[Mag] WARNING: vector3.csv missing required columns")
        return []
    df = df.sort_values(tcol).reset_index(drop=True)
    recs: List[MagRecord] = []
    for _, r in df.iterrows():
        recs.append(MagRecord(
            t=float(r[tcol]),
            mag=np.array([float(r['x']), float(r['y']), float(r['z'])], dtype=float)
        ))
    print(f"[Mag] Loaded {len(recs)} magnetometer samples from {path}")
    return recs

# ===============================
# Magnetometer calibration & heading
# ===============================

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
        hard_iron = MAG_HARD_IRON_OFFSET
    if soft_iron is None:
        soft_iron = MAG_SOFT_IRON_MATRIX
    
    # Subtract hard-iron offset
    mag_centered = mag_raw - hard_iron
    
    # Apply soft-iron correction (full 3x3 matrix)
    mag_corrected = soft_iron @ mag_centered
    
    return mag_corrected


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


def angle_wrap(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def apply_mag_filter(yaw_mag: float, yaw_t: float, gyro_z: float, dt_imu: float, 
                     in_convergence: bool = False) -> Tuple[float, float]:
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
        
    Returns:
        (yaw_filtered, r_scale_factor)
        yaw_filtered: EMA-smoothed yaw measurement [rad]
        r_scale_factor: Multiplier for measurement noise R (1.0 during convergence)
    """
    global _MAG_FILTER_STATE
    
    state = _MAG_FILTER_STATE
    r_scale = 1.0
    
    # Accumulate gyro integration between mag updates
    state['integrated_gyro_dz'] += gyro_z * dt_imu
    
    # First measurement - initialize state
    if state['yaw_ema'] is None:
        state['yaw_ema'] = yaw_mag
        state['last_yaw_mag'] = yaw_mag
        state['last_yaw_t'] = yaw_t
        state['integrated_gyro_dz'] = 0.0
        state['n_updates'] = 1
        return yaw_mag, 1.0
    
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
    if not in_convergence and yaw_rate_mag > MAG_MAX_YAW_RATE * 1.5:
        # Mag is changing faster than gyro max rate - likely noise/interference
        r_scale = max(r_scale, 2.0)
        if state['n_updates'] % 50 == 0:
            print(f"[MAG-FILTER] High rate: {np.degrees(yaw_rate_mag):.1f}°/s > {np.degrees(MAG_MAX_YAW_RATE*1.5):.1f}°/s limit → R×{r_scale:.1f}")
    
    # =========================================================================
    # Step 2: Gyro consistency check (DISABLED during convergence)
    # =========================================================================
    # Compare mag yaw change with integrated gyro rotation
    expected_dyaw_gyro = state['integrated_gyro_dz']  # Expected change from gyro
    
    consistency_error = abs(angle_wrap(dyaw_mag - expected_dyaw_gyro))
    
    # Only apply R inflation after convergence period
    if not in_convergence and consistency_error > MAG_GYRO_CONSISTENCY_THRESHOLD:
        # Mag measurement inconsistent with gyro integration
        # Scale R inflation based on how inconsistent
        r_scale_consistency = 1.0 + (consistency_error / MAG_GYRO_CONSISTENCY_THRESHOLD - 1.0) * (MAG_CONSISTENCY_R_INFLATE - 1.0)
        r_scale_consistency = np.clip(r_scale_consistency, 1.0, MAG_CONSISTENCY_R_INFLATE)
        r_scale = max(r_scale, r_scale_consistency)
        
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
    yaw_ema_new = angle_wrap(state['yaw_ema'] + MAG_EMA_ALPHA * yaw_diff)
    
    # =========================================================================
    # Step 4: Update state for next iteration
    # =========================================================================
    state['yaw_ema'] = yaw_ema_new
    state['last_yaw_mag'] = yaw_mag
    state['last_yaw_t'] = yaw_t
    state['n_updates'] += 1
    
    return yaw_ema_new, r_scale


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


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions q1 * q2.
    
    Args:
        q1, q2: Quaternions in [w, x, y, z] format
        
    Returns: Result quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])


def quaternion_to_yaw(q_wxyz: np.ndarray) -> float:
    """
    Extract yaw angle from quaternion (ENU frame convention).
    
    CRITICAL: Xsens MTi-30 quaternion represents BODY→WORLD rotation.
    (This is the same convention we use in imu_pose_to_camera_pose)
    The rotation matrix R is R_body_to_world directly (NO transpose needed).
    Yaw = arctan2(R[1,0], R[0,0]) where R is the body X axis in world frame.
    """
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    R_body_to_world = R_scipy.from_quat(q_xyzw).as_matrix()  # Body-to-World directly
    # Extract yaw from rotation matrix (heading of body X axis in world XY plane)
    yaw = np.arctan2(R_body_to_world[1,0], R_body_to_world[0,0])
    return yaw  # radians


def yaw_to_quaternion_update(yaw_current: float, yaw_measured: float) -> np.ndarray:
    """
    Compute quaternion correction for yaw-only update.
    
    Args:
        yaw_current: Current yaw from state (radians)
        yaw_measured: Measured yaw from magnetometer (radians)
    
    Returns: Delta quaternion [w,x,y,z] for multiplicative update
    """
    # Compute yaw error (unwrap to [-π, π])
    dyaw = yaw_measured - yaw_current
    dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
    
    # Convert to quaternion (rotation around Z axis)
    # q = [cos(θ/2), 0, 0, sin(θ/2)] for Z-axis rotation
    dq_wxyz = np.array([
        np.cos(dyaw / 2.0),
        0.0,
        0.0,
        np.sin(dyaw / 2.0)
    ])
    
    return dq_wxyz

# ===============================
# Camera model helpers (Kannala-Brandt Fisheye)
# ===============================

def kannala_brandt_unproject(pts: np.ndarray, K: np.ndarray, D: np.ndarray, 
                             max_iters: int = 10) -> np.ndarray:
    """
    Unproject 2D pixel coordinates to 3D unit rays using Kannala-Brandt model.
    
    The Kannala-Brandt model maps 3D rays to 2D pixels:
        θ = angle from optical axis
        θ_d = k1*θ + k2*θ³ + k3*θ⁵ + k4*θ⁷ + ...
        r = f * θ_d   (radial distance in pixels)
        
    Where D = [k1, k2, k3, k4] (note: k1 is typically 1.0 for normalized model)
    
    CRITICAL: cv2.fisheye uses DIFFERENT model (equidistant: r = f*θ)
    This function properly handles Kannala-Brandt as used by MUN-FRL Bell 412.
    
    Args:
        pts: Nx2 array of pixel coordinates
        K: 3x3 camera intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        D: 4-element distortion coefficients [k2, k3, k4, k5] 
           Note: In VINS convention, k2-k5 (k1=1 implicit)
        max_iters: Newton iterations for inverse mapping
    
    Returns:
        Nx2 array of normalized coordinates (x/z, y/z) on unit plane z=1
    """
    if pts.size == 0:
        return pts
    
    pts = pts.reshape(-1, 2)
    n = pts.shape[0]
    
    # Extract intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Kannala-Brandt coefficients: θ_d = k1*θ + k2*θ³ + k3*θ⁵ + k4*θ⁷
    # In VINS/Kalibr convention: k1 = 1.0 (implicit), D = [k2, k3, k4, k5]
    k1 = 1.0
    k2 = D[0] if len(D) > 0 else 0.0
    k3 = D[1] if len(D) > 1 else 0.0
    k4 = D[2] if len(D) > 2 else 0.0
    k5 = D[3] if len(D) > 3 else 0.0
    
    # Normalize to camera frame (remove principal point and focal length)
    x_dist = (pts[:, 0] - cx) / fx
    y_dist = (pts[:, 1] - cy) / fy
    
    # Radial distance in normalized image
    r_dist = np.sqrt(x_dist**2 + y_dist**2)
    
    # Angle in image plane
    phi = np.arctan2(y_dist, x_dist)
    
    # For K-B model: r_dist = θ_d where θ_d = k1*θ + k2*θ³ + k3*θ⁵ + k4*θ⁷
    # Need to solve for θ given θ_d (Newton's method)
    theta_d = r_dist
    theta = theta_d.copy()  # Initial guess
    
    for _ in range(max_iters):
        # f(θ) = k1*θ + k2*θ³ + k3*θ⁵ + k4*θ⁷ + k5*θ⁹ - θ_d = 0
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        
        f_theta = k1 * theta + k2 * theta * theta2 + k3 * theta * theta4 + \
                  k4 * theta * theta6 + k5 * theta * theta8 - theta_d
        
        # f'(θ) = k1 + 3*k2*θ² + 5*k3*θ⁴ + 7*k4*θ⁶ + 9*k5*θ⁸
        f_prime = k1 + 3*k2*theta2 + 5*k3*theta4 + 7*k4*theta6 + 9*k5*theta8
        
        # Newton update: θ_new = θ - f(θ)/f'(θ)
        # Avoid division by zero
        valid = np.abs(f_prime) > 1e-10
        theta[valid] = theta[valid] - f_theta[valid] / f_prime[valid]
        
        # Clamp to valid range [0, π/2]
        theta = np.clip(theta, 0, np.pi / 2)
    
    # Convert θ, φ to 3D unit ray
    # In camera frame: x = sin(θ)*cos(φ), y = sin(θ)*sin(φ), z = cos(θ)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Handle center point (r=0 → θ=0 → straight ahead)
    center_mask = r_dist < 1e-8
    sin_theta[center_mask] = 0.0
    cos_theta[center_mask] = 1.0
    
    # 3D ray in camera frame
    ray_x = sin_theta * np.cos(phi)
    ray_y = sin_theta * np.sin(phi)
    ray_z = cos_theta
    
    # Return normalized coordinates (x/z, y/z) on z=1 plane
    # Handle edge case where z is very small
    z_safe = np.maximum(ray_z, 1e-6)
    x_norm = ray_x / z_safe
    y_norm = ray_y / z_safe
    
    return np.column_stack([x_norm, y_norm])


def normalized_to_unit_ray(x_norm: float, y_norm: float) -> np.ndarray:
    """
    Convert normalized coordinates (x/z, y/z) back to unit ray direction.
    
    For fisheye cameras, normalized coordinates can be very large (tan(theta) 
    can be >> 1 for large angles). Simply using [x_n, y_n, 1] / norm gives 
    WRONG direction!
    
    Correct conversion:
        Given: x_n = x/z = tan(θ)*cos(φ), y_n = y/z = tan(θ)*sin(φ)
        We want: ray = [sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
    
    Derivation:
        r_n = sqrt(x_n² + y_n²) = tan(θ)
        θ = arctan(r_n)
        ray = [sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
        where cos(φ) = x_n/r_n, sin(φ) = y_n/r_n
    
    Returns:
        3D unit vector pointing in ray direction (Z positive = forward)
    """
    r_n = np.sqrt(x_norm**2 + y_norm**2)
    
    if r_n < 1e-8:
        # Center of image: straight ahead
        return np.array([0.0, 0.0, 1.0])
    
    theta = np.arctan(r_n)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Azimuth direction in image plane
    cos_phi = x_norm / r_n
    sin_phi = y_norm / r_n
    
    # Unit ray in camera frame (Z+ = forward/optical axis)
    ray = np.array([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    ])
    
    return ray


def make_KD_for_size(kb: dict, dst_w: int, dst_h: int) -> Tuple[np.ndarray, np.ndarray]:
    sx = dst_w / float(kb["w"])  # scale from calib size → runtime size
    sy = dst_h / float(kb["h"])
    fx = kb["mu"] * sx
    fy = kb["mv"] * sy
    cx = kb["u0"] * sx
    cy = kb["v0"] * sy
    K = np.array([[fx, 0,  cx], [0,  fy, cy], [0, 0, 1.0]], dtype=np.float64)
    D = np.array([kb["k2"], kb["k3"], kb["k4"], kb["k5"]], dtype=np.float64)
    return K, D

# ===============================
# VIO front-end (OpenVINS-style: Grid-based KLT)
# ===============================
class VIOFrontEnd:
    """
    OpenVINS-style visual front-end:
    1. Grid-based feature distribution: Divide image into grid, select best features per cell
    2. Shi-Tomasi corner detector: Better tracking quality than ORB
    3. Multi-stage KLT tracking: Coarse-to-fine optical flow
    4. Track quality scoring: Distance from epipolar line, temporal consistency
    5. Aggressive outlier rejection: RANSAC + chi-squared gating
    
    Reference: OpenVINS (https://github.com/rpng/open_vins)
    """
    def __init__(self, img_w: int, img_h: int, K: np.ndarray, D: np.ndarray, use_fisheye: bool = True):
        self.K = K
        self.D = D
        self.use_fisheye = use_fisheye
        self.img_w = img_w
        self.img_h = img_h
        
        # Grid-based feature extraction (OpenVINS-style)
        self.grid_x = 8  # Horizontal grid cells (was not grid-based before)
        self.grid_y = 8  # Vertical grid cells
        self.max_features_per_grid = 10  # Max features per cell (ensures distribution)
        self.max_total_features = self.grid_x * self.grid_y * self.max_features_per_grid  # 640 features
        
        # Shi-Tomasi corner detection (OpenVINS uses this, NOT ORB)
        self.feature_params = dict(
            maxCorners=self.max_total_features,
            qualityLevel=0.01,  # Min eigenvalue quality (lower = more features)
            minDistance=5,      # Min distance between features (pixels)
            blockSize=5         # Corner detection window size
        )
        
        # Multi-stage KLT parameters (OpenVINS-style: coarse-to-fine)
        self.lk_params = dict(
            winSize=(21, 21),    # Larger window for nadir camera (more motion)
            maxLevel=4,          # 4 pyramid levels (was 3) - better for large motion
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,  # Get quality measure
            minEigThreshold=0.001  # Reject poor tracking
        )
        
        # Track database: fid -> list of observations
        # Each observation: {'frame': int, 'pt': (x,y), 'quality': float, 'grid_cell': (gx, gy)}
        self.tracks = {}
        self.next_fid = 0
        self.last_gray_for_klt = None
        self.last_pts_for_klt = None      # Nx2 float32 (current tracked points)
        self.last_fids_for_klt = None     # N int64 (feature IDs for tracked points)
        
        # Keyframe state (for ORB+Essential fallback when KLT fails)
        self.keyframe_gray = None
        self.keyframe_frame_idx = -1
        self.frame_idx = -1
        self.last_matches = None  # (pts_keyframe, pts_current)
        
        # OpenVINS-style keyframe management
        self.min_tracked_ratio = 0.6  # Create new keyframe if < 60% tracked (was 0.7)
        self.min_parallax_threshold = 30.0  # pixels - OpenVINS uses 20-30px for nadir (was 25.0)
        self.max_track_length = 100  # Allow longer tracks (was 50) - don't prune too aggressively
        self.min_track_length = 4   # Need 4+ observations for good triangulation (was 3)
        
        # Track quality scoring (NEW: OpenVINS-style)
        self.quality_threshold = 0.001  # Min KLT eigenvalue (reject poor tracking)
        self.epipolar_threshold = 1.0   # Max distance from epipolar line (pixels)
        self.temporal_threshold = 50.0  # Max frame-to-frame motion (pixels)

    def _extract_grid_features(self, img_gray: np.ndarray) -> np.ndarray:
        """
        OpenVINS-style grid-based feature extraction:
        1. Divide image into grid cells
        2. Detect Shi-Tomasi corners in each cell
        3. Select best features per cell (highest quality)
        4. Return distributed feature points
        
        This ensures features are spread across the image (better conditioning)
        vs traditional detector that clusters features in textured regions.
        
        Args:
            img_gray: Grayscale image
        
        Returns:
            features: Nx2 array of (x, y) pixel coordinates
        """
        cell_w = self.img_w // self.grid_x
        cell_h = self.img_h // self.grid_y
        
        all_features = []
        
        for gx in range(self.grid_x):
            for gy in range(self.grid_y):
                # Define cell ROI
                x1 = gx * cell_w
                y1 = gy * cell_h
                x2 = min((gx + 1) * cell_w, self.img_w)
                y2 = min((gy + 1) * cell_h, self.img_h)
                
                # Skip tiny cells at image boundaries
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                
                # Extract cell region
                cell_roi = img_gray[y1:y2, x1:x2]
                
                # Detect Shi-Tomasi corners in this cell
                corners = cv2.goodFeaturesToTrack(
                    cell_roi,
                    maxCorners=self.max_features_per_grid,
                    qualityLevel=self.feature_params['qualityLevel'],
                    minDistance=self.feature_params['minDistance'],
                    blockSize=self.feature_params['blockSize']
                )
                
                if corners is not None and len(corners) > 0:
                    # Convert to global coordinates
                    corners_global = corners.reshape(-1, 2) + np.array([x1, y1], dtype=np.float32)
                    
                    # Add grid cell metadata for tracking
                    for pt in corners_global:
                        all_features.append(pt)
        
        if len(all_features) == 0:
            return np.empty((0, 2), dtype=np.float32)
        
        return np.array(all_features, dtype=np.float32)
    
    def _compute_epipolar_error(self, pts_prev: np.ndarray, pts_curr: np.ndarray, 
                                 E: np.ndarray) -> np.ndarray:
        """
        Compute distance from points to epipolar lines.
        
        For point correspondence (p1, p2), the epipolar constraint is:
        p2^T * E * p1 = 0
        
        Distance from p2 to epipolar line l2 = E * p1:
        d = |p2^T * E * p1| / ||E * p1||_2
        
        Args:
            pts_prev: Nx2 normalized coordinates from previous frame
            pts_curr: Nx2 normalized coordinates from current frame
            E: 3x3 essential matrix
        
        Returns:
            errors: N-array of epipolar distances (pixels)
        """
        if E is None or pts_prev.shape[0] == 0:
            return np.array([])
        
        # Convert to homogeneous coordinates
        pts_prev_h = np.hstack([pts_prev, np.ones((len(pts_prev), 1))])
        pts_curr_h = np.hstack([pts_curr, np.ones((len(pts_curr), 1))])
        
        # Compute epipolar lines: l2 = E * p1
        lines = (E @ pts_prev_h.T).T  # (N, 3)
        
        # Distance from p2 to line l2: |p2^T * l2| / ||l2[:2]||
        numerator = np.abs(np.sum(pts_curr_h * lines, axis=1))
        denominator = np.linalg.norm(lines[:, :2], axis=1)
        
        errors = numerator / (denominator + 1e-8)
        
        return errors

    def _undistort_pts(self, pts: np.ndarray) -> np.ndarray:
        """
        Undistort pixel coordinates to normalized camera coordinates.
        
        CRITICAL: Bell 412 uses Kannala-Brandt fisheye model, NOT cv2.fisheye (equidistant).
        Using wrong model causes ~10x error in normalized coordinates!
        
        Args:
            pts: Nx2 array of pixel coordinates
            
        Returns:
            Nx1x2 array of normalized coordinates (x/z, y/z)
        """
        # pts: Nx2
        if self.use_fisheye:
            # FIXED: Use proper Kannala-Brandt unprojection instead of cv2.fisheye
            # cv2.fisheye uses equidistant model (r = f*θ) which is WRONG for K-B
            undist = kannala_brandt_unproject(pts.reshape(-1, 2), self.K, self.D)
            
            # DEBUG: Print first few undistortions to verify KB is working
            if not hasattr(self, '_undist_dbg_count'):
                self._undist_dbg_count = 0
            if self._undist_dbg_count < 5:
                for i in range(min(3, len(pts))):
                    px = pts.reshape(-1, 2)[i]
                    un = undist[i]
                    print(f"[KB-UNDIST] pixel=({px[0]:.1f},{px[1]:.1f}) → norm=({un[0]:.4f},{un[1]:.4f}) K={self.K[0,0]:.1f},{self.K[1,1]:.1f} D={self.D.flatten()[:4]}")
                self._undist_dbg_count += 1
            
            return undist.reshape(-1, 1, 2)
        else:
            undist = cv2.undistortPoints(pts.reshape(-1,1,2), self.K, self.D)
            return undist  # Nx1x2

    def _should_create_keyframe(self, current_gray: np.ndarray) -> Tuple[bool, str]:
        """
        Decide if we should create a new keyframe based on:
        1. Tracked feature ratio
        2. Median parallax from last keyframe (STRICTER for nadir)
        3. Frame difference
        
        NEW: Nadir-aware parallax thresholds to improve triangulation conditioning.
        
        Returns: (should_create, reason)
        """
        if self.keyframe_frame_idx < 0:
            return True, "first_keyframe"
        
        # Check tracked ratio
        if self.keyframe_tracked_ratio < self.min_tracked_ratio:
            return True, f"low_tracking_ratio_{self.keyframe_tracked_ratio:.2f}"
        
        # Check parallax if we have KLT tracking
        if self.last_gray_for_klt is not None and self.last_pts_for_klt is not None and len(self.last_pts_for_klt) > 10:
            try:
                p0 = self.last_pts_for_klt.reshape(-1,1,2)
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.keyframe_gray, current_gray, p0, None, **self.lk_params)
                if p1 is not None and st is not None:
                    good = st.reshape(-1) == 1
                    if np.sum(good) > 10:
                        parallax = np.linalg.norm(p0[good] - p1[good], axis=2).reshape(-1)
                        median_parallax = np.median(parallax)
                        
                        # NEW: Adaptive parallax threshold for nadir camera
                        # Nadir cameras need MORE parallax for good triangulation
                        # (baseline mostly in image plane, less depth sensitivity)
                        parallax_threshold = self.min_parallax_threshold
                        
                        if median_parallax > parallax_threshold:
                            return True, f"high_parallax_{median_parallax:.1f}px"
            except Exception:
                pass
        
        # Check frame difference
        frames_since_keyframe = self.frame_idx - self.keyframe_frame_idx
        if frames_since_keyframe > 20:  # Force keyframe every 20 frames
            return True, f"frame_count_{frames_since_keyframe}"
        
        return False, "keep_current"
    
    def compute_track_parallax_stats(self) -> dict:
        """
        Compute parallax statistics for currently tracked features.
        
        NEW: Used for parallax-aware feature gating in MSCKF.
        
        Returns:
            stats: dict with 'median_px', 'mean_px', 'max_px', 'count'
        """
        parallax_values = []
        
        for fid, hist in self.tracks.items():
            if len(hist) < 2:
                continue
            
            # Compute max parallax across all observations
            pts = np.array([obs['pt'] for obs in hist])  # (N, 2)
            if len(pts) < 2:
                continue
            
            # Pairwise distances
            dists = []
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    dists.append(np.linalg.norm(pts[i] - pts[j]))
            
            if dists:
                parallax_values.append(max(dists))
        
        if not parallax_values:
            return {'median_px': 0.0, 'mean_px': 0.0, 'max_px': 0.0, 'count': 0}
        
        return {
            'median_px': np.median(parallax_values),
            'mean_px': np.mean(parallax_values),
            'max_px': np.max(parallax_values),
            'count': len(parallax_values)
        }

    def _prune_old_tracks(self):
        """Remove tracks that are too old or have poor quality."""
        to_remove = []
        current_frame = self.frame_idx
        
        tracks_before = len(self.tracks)
        
        for fid, hist in self.tracks.items():
            if not hist:
                to_remove.append(fid)
                continue
            
            # Remove if track is too long
            track_length = len(hist)
            if track_length > self.max_track_length:
                to_remove.append(fid)
                continue
            
            # Remove if last observation is too old (not seen in last 3 frames)
            last_frame = hist[-1]['frame']
            if current_frame - last_frame > 3:
                to_remove.append(fid)
                continue
            
            # Remove if average quality is too low
            if track_length > 0:
                avg_quality = np.mean([obs.get('quality', 1.0) for obs in hist])
                if avg_quality < 0.3:
                    to_remove.append(fid)
        
        for fid in to_remove:
            del self.tracks[fid]
        
        if len(to_remove) > 0:
            tracks_after = len(self.tracks)
            print(f"[VIO][PRUNE] Frame {current_frame}: Removed {len(to_remove)} tracks ({tracks_before} -> {tracks_after})")

    def estimate_homography_scale(self, pts_prev: np.ndarray, pts_curr: np.ndarray, 
                                   altitude_m: float, pitch_rad: float = 0.0) -> tuple:
        """
        Estimate velocity scale using planar homography decomposition.
        
        For nadir camera over flat terrain, homography provides direct scale recovery:
        H = K * (R + (1/d) * t * n^T) * K^-1
        
        where d = altitude / cos(pitch) is the distance to ground plane.
        
        Args:
            pts_prev: Nx2 normalized coordinates from previous frame
            pts_curr: Nx2 normalized coordinates from current frame  
            altitude_m: Altitude above ground (MSL - DEM or AGL)
            pitch_rad: Camera pitch angle (0 = nadir pointing down)
        
        Returns:
            (success, scale_factor, R, t_scaled, inliers)
            - scale_factor: metric scale (meters per unit translation)
            - R: 3x3 rotation matrix
            - t_scaled: 3x1 translation vector in meters
            - inliers: boolean mask of inlier correspondences
        """
        if len(pts_prev) < 8:  # Homography requires minimum 4 correspondences
            return False, 1.0, None, None, None
        
        try:
            # Find homography using RANSAC
            H, mask = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, ransacReprojThreshold=0.01)
            
            if H is None or mask is None:
                return False, 1.0, None, None, None
            
            inliers = mask.ravel().astype(bool)
            num_inliers = np.sum(inliers)
            
            if num_inliers < 8:  # Require sufficient inliers
                return False, 1.0, None, None, inliers
            
            # Decompose homography into R, t, n (plane normal)
            # For ground plane: n = [0, 0, 1]^T (pointing up in world frame)
            # Distance to plane: d = altitude / cos(pitch)
            d_plane = altitude_m / max(0.1, np.cos(pitch_rad))
            
            # Decompose H = R + (1/d) * t * n^T
            # Since we assume n = [0, 0, 1], we can extract t directly
            num_solutions, Rs, ts, normals = cv2.decomposeHomographyMat(H, np.eye(3))
            
            # Select the best solution (positive depth, smallest rotation)
            best_idx = -1
            best_score = float('inf')
            
            for i in range(num_solutions):
                R_sol = Rs[i]
                t_sol = ts[i].ravel()
                n_sol = normals[i].ravel()
                
                # Check if solution is physically plausible:
                # 1. Normal should point approximately upward (n_z > 0.5)
                # 2. Rotation should be small (trace(R) close to 3)
                # 3. Translation should be reasonable
                
                if n_sol[2] < 0.3:  # Normal not pointing up enough
                    continue
                
                trace_R = np.trace(R_sol)
                rotation_magnitude = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))
                
                # Score: prefer small rotations and normals close to [0, 0, 1]
                score = rotation_magnitude + 2.0 * (1.0 - n_sol[2])
                
                if score < best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx == -1:
                return False, 1.0, None, None, inliers
            
            R_best = Rs[best_idx]
            t_best = ts[best_idx].ravel()
            
            # Scale translation by distance to plane
            # t_scaled = d * t_unit
            t_scaled = d_plane * t_best
            
            # Compute scale factor (for compatibility with existing code)
            scale_factor = d_plane
            
            return True, scale_factor, R_best, t_scaled, inliers
            
        except Exception as e:
            print(f"[VIO][HOMOGRAPHY] Decomposition failed: {e}")
            return False, 1.0, None, None, None

    def bootstrap(self, img_gray: np.ndarray, t: float):
        """
        Initialize front-end with first frame.
        
        OpenVINS-style:
        - Use grid-based feature extraction (NOT ORB clustering)
        - Create initial tracks with uniform spatial distribution
        """
        self.keyframe_gray = img_gray.copy()
        self.last_frame_time = t
        self.frame_idx = 0
        self.keyframe_frame_idx = 0
        
        # Grid-based feature extraction (OpenVINS-style)
        try:
            features = self._extract_grid_features(img_gray)
            
            if len(features) > 0:
                self.last_pts_for_klt = features.astype(np.float32)
                self.last_gray_for_klt = img_gray.copy()
                
                # Initialize tracks
                fids = []
                for p in features:
                    fid = self.next_fid
                    self.next_fid += 1
                    self.tracks[fid] = [{'frame': self.frame_idx, 'pt': (float(p[0]), float(p[1])), 'quality': 1.0}]
                    fids.append(fid)
                
                self.last_fids_for_klt = np.array(fids, dtype=np.int64)
                
                print(f"[VIO][BOOTSTRAP] Initialized {len(features)} grid-based features (grid {self.grid_x}x{self.grid_y})")
            else:
                self.last_pts_for_klt = None
                self.last_gray_for_klt = None
                self.last_fids_for_klt = None
                print("[VIO][BOOTSTRAP] WARNING: No features detected!")
        except Exception as e:
            print(f"[VIO][BOOTSTRAP] Exception: {e}")
            self.last_pts_for_klt = None
            self.last_gray_for_klt = None
            self.last_fids_for_klt = None

    def step(self, img_gray: np.ndarray, t: float):
        """Return (success, num_inliers, R_vo, t_unit, dt_img) or (False, ...)."""
        if self.keyframe_gray is None:
            self.bootstrap(img_gray, t)
            return False, 0, None, None, 0.0
        
        # Increment frame index
        self.frame_idx += 1
        
        # --- KLT tracking from last_gray_for_klt -> img_gray ---
        num_tracked_successfully = 0
        try:
            if self.last_gray_for_klt is None:
                print(f"[VIO][TRACK][DEBUG] Frame {self.frame_idx}: last_gray_for_klt is None!")
            if self.last_pts_for_klt is None:
                print(f"[VIO][TRACK][DEBUG] Frame {self.frame_idx}: last_pts_for_klt is None!")
            
            if self.last_gray_for_klt is not None and self.last_pts_for_klt is not None and len(self.last_pts_for_klt)>0:
                p0 = self.last_pts_for_klt.reshape(-1,1,2)
                # Forward KLT tracking
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.last_gray_for_klt, img_gray, p0, None, **self.lk_params)
                
                if p1 is not None:
                    # Backward tracking for consistency check
                    p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(img_gray, self.last_gray_for_klt, p1, None, **self.lk_params)
                    
                    # Calculate forward-backward error
                    fb_err = np.linalg.norm(p0 - p0_back.reshape(-1,1,2), axis=2).reshape(-1)
                    
                    # Calculate flow magnitude for spatial consistency
                    flow_mag = np.linalg.norm(p1.reshape(-1,2) - p0.reshape(-1,2), axis=1)
                    
                    # IMPROVED quality masks for robust tracking:
                    # 1. Basic status checks
                    good_mask = (st.reshape(-1) == 1) & (st_back.reshape(-1) == 1)
                    
                    # 2. Forward-backward consistency (relaxed for fisheye distortion)
                    good_mask = good_mask & (fb_err < 2.0)  # 2.0 pixels for fisheye tolerance
                    
                    # 3. Forward tracking error (tightened to reject poor tracks)
                    good_mask = good_mask & (err.reshape(-1) < 8.0)  # Reduced from 12.0 to 8.0
                    
                    # 4. Flow magnitude sanity check (reject extreme outliers)
                    # For 20Hz camera (~0.05s), reasonable motion is < 100 pixels
                    # Extreme flows (>200px) are likely tracking failures
                    good_mask = good_mask & (flow_mag < 200.0)
                    
                    # 5. Boundary check (reject features near image edges)
                    p1_reshaped = p1.reshape(-1,2)
                    margin = 10
                    good_mask = good_mask & (p1_reshaped[:,0] >= margin) & (p1_reshaped[:,0] < self.img_w - margin)
                    good_mask = good_mask & (p1_reshaped[:,1] >= margin) & (p1_reshaped[:,1] < self.img_h - margin)
                    
                    # Apply quality mask
                    st = st.reshape(-1) * good_mask
                    p1 = p1.reshape(-1,2)
                    fb_err_final = fb_err
                    err_final = err.reshape(-1)
                    flow_mag_final = flow_mag
                    
                    # Debug: Log tracking quality statistics
                    if np.sum(good_mask) < len(good_mask) * 0.5:  # If > 50% rejected
                        print(f"[VIO][TRACKING] Quality check: {np.sum(good_mask)}/{len(good_mask)} passed")
                        print(f"  FB errors: min={fb_err.min():.2f}, med={np.median(fb_err):.2f}, max={fb_err.max():.2f}")
                        print(f"  Flow mag: min={flow_mag.min():.2f}, med={np.median(flow_mag):.2f}, max={flow_mag.max():.2f}")
                        print(f"  Rejected by FB: {np.sum(fb_err >= 2.0)}, by error: {np.sum(err.reshape(-1) >= 8.0)}, by flow: {np.sum(flow_mag >= 200.0)}")
                    
                    # Update tracks using feature ID mapping
                    if self.last_fids_for_klt is None or len(self.last_fids_for_klt) != len(p0):
                        # Fallback: create FID list if missing
                        self.last_fids_for_klt = np.array(list(self.tracks.keys())[:len(p0)], dtype=np.int64)
                    
                    tracked_fids = []
                    new_pts = []
                    new_fids = []
                    
                    for idx in range(len(p1)):
                        if idx >= len(self.last_fids_for_klt):
                            break
                        
                        fid = self.last_fids_for_klt[idx]
                        
                        if st[idx] and fid in self.tracks:
                            pt = (float(p1[idx,0]), float(p1[idx,1]))
                            # Calculate quality score based on tracking errors
                            quality = 1.0 / (1.0 + fb_err_final[idx] + err_final[idx] * 0.1)
                            self.tracks[fid].append({'frame': self.frame_idx, 'pt': pt, 'quality': quality})
                            tracked_fids.append(fid)
                            new_pts.append(pt)
                            new_fids.append(fid)
                            num_tracked_successfully += 1
                    
                    # Update tracking ratio for keyframe decision
                    total_tracks_before = len(self.last_pts_for_klt)
                    self.keyframe_tracked_ratio = num_tracked_successfully / max(1, total_tracks_before)
                    
                    # Update tracking state for next iteration
                    if len(new_pts) > 0:
                        self.last_pts_for_klt = np.array(new_pts, dtype=np.float32)
                        self.last_fids_for_klt = np.array(new_fids, dtype=np.int64)
                        self.last_gray_for_klt = img_gray.copy()
                        
                        # Debug logging
                        if self.frame_idx % 20 == 0:
                            print(f"[VIO][TRACK] Frame {self.frame_idx}: {num_tracked_successfully}/{total_tracks_before} tracked ({self.keyframe_tracked_ratio*100:.1f}%), {len(self.tracks)} total tracks")
                    else:
                        # ALL TRACKING FAILED! Replenish immediately!
                        print(f"[VIO][TRACK] WARNING: Frame {self.frame_idx} - All tracking failed! Replenishing...")
                        
                        # Extract new features immediately
                        try:
                            new_features = self._extract_grid_features(img_gray)
                            
                            if len(new_features) > 0:
                                new_fids = []
                                for p in new_features:
                                    fid = self.next_fid
                                    self.next_fid += 1
                                    self.tracks[fid] = [{'frame': self.frame_idx, 'pt': (float(p[0]), float(p[1])), 'quality': 1.0}]
                                    new_fids.append(fid)
                                
                                self.last_pts_for_klt = new_features.astype(np.float32)
                                self.last_fids_for_klt = np.array(new_fids, dtype=np.int64)
                                self.last_gray_for_klt = img_gray.copy()
                                print(f"[VIO][EMERGENCY_REPLENISH] Added {len(new_features)} new features")
                            else:
                                self.last_pts_for_klt = np.empty((0, 2), dtype=np.float32)
                                self.last_fids_for_klt = np.empty(0, dtype=np.int64)
                                self.last_gray_for_klt = None
                                print("[VIO][EMERGENCY_REPLENISH] FAILED: No features detected!")
                        except Exception as e:
                            print(f"[VIO][EMERGENCY_REPLENISH] Exception: {e}")
                            self.last_pts_for_klt = np.empty((0, 2), dtype=np.float32)
                            self.last_fids_for_klt = np.empty(0, dtype=np.int64)
                            self.last_gray_for_klt = None
        except Exception as e:
            print(f"[VIO] KLT tracking exception: {e}")
            pass
        
        # Prune old/poor quality tracks periodically
        if self.frame_idx % 5 == 0:
            self._prune_old_tracks()
        
        # OpenVINS-style pose estimation from KLT tracks (NOT ORB)
        # Use sliding window: previous frame to current frame (NOT keyframe)
        dt_img = max(1e-3, t - (self.last_frame_time or t))
        
        # Gather points from previous frame and current frame
        prev_pts = []
        curr_pts = []
        
        prev_frame_idx = self.frame_idx - 1
        
        for fid, hist in self.tracks.items():
            if len(hist) < 2:
                continue
            
            # Find observations at previous and current frame
            prev_obs = None
            curr_obs = None
            
            for obs in hist:
                if obs['frame'] == prev_frame_idx:
                    prev_obs = obs
                if obs['frame'] == self.frame_idx:
                    curr_obs = obs
            
            if prev_obs is not None and curr_obs is not None:
                prev_pts.append(prev_obs['pt'])
                curr_pts.append(curr_obs['pt'])
        
        if len(prev_pts) < VO_MIN_INLIERS:
            # Fallback: not enough tracked features, return failure
            print(f"[VIO][POSE] Insufficient frame-to-frame correspondences: {len(prev_pts)}/{VO_MIN_INLIERS} (frame {prev_frame_idx}->{self.frame_idx})")
            # Debug: print track statistics
            total_tracks = len(self.tracks)
            active_tracks = sum(1 for hist in self.tracks.values() if hist and hist[-1]['frame'] == self.frame_idx)
            print(f"[VIO][DEBUG] Total tracks: {total_tracks}, Active at frame {self.frame_idx}: {active_tracks}")
            return False, len(prev_pts), None, None, dt_img
        
        q1 = np.array(prev_pts, dtype=np.float32)
        q2 = np.array(curr_pts, dtype=np.float32)
        
        # CRITICAL: OpenVINS-style aggressive outlier rejection
        # Step 1: Flow magnitude filtering (MAD-based)
        flow_vectors = q2 - q1
        flow_mag = np.linalg.norm(flow_vectors, axis=1)
        
        median_flow = np.median(flow_vectors, axis=0)
        flow_deviations = flow_vectors - median_flow
        mad = np.median(np.linalg.norm(flow_deviations, axis=1))
        
        # Adaptive threshold: median + 5*MAD, capped at 300px, min 50px
        flow_threshold = min(300.0, max(50.0, np.median(flow_mag) + 5.0 * mad))
        flow_valid_mask = flow_mag < flow_threshold
        
        if np.sum(flow_valid_mask) < VO_MIN_INLIERS:
            print(f"[VIO][OUTLIER] Flow filtering rejected {np.sum(~flow_valid_mask)}/{len(flow_valid_mask)} features")
            return False, len(q1), None, None, dt_img
        
        q1 = q1[flow_valid_mask]
        q2 = q2[flow_valid_mask]
        
        # Step 2: Undistort and compute Essential matrix with RANSAC
        # OpenVINS uses ~2-3 pixel threshold for normalized coordinates
        # For fisheye camera with distortion, use relaxed threshold
        # TUNED: 3e-3 → 5e-3 to handle fisheye distortion residuals
        q1n = self._undistort_pts(q1)
        q2n = self._undistort_pts(q2)
        E, mask = cv2.findEssentialMat(q2n, q1n, method=cv2.RANSAC, prob=0.999, threshold=5e-3)
        
        if E is None or mask is None:
            return False, len(q1), None, None, dt_img
        
        # Step 3: Recover pose
        num_inl, R_vo, t_unit, pose_mask = cv2.recoverPose(E, q2n, q1n, mask=mask)
        
        if num_inl < VO_MIN_INLIERS:
            print(f"[VIO][POSE] Insufficient inliers after RANSAC: {num_inl}/{VO_MIN_INLIERS}")
            return False, num_inl, None, None, dt_img
        
        # Step 4: Epipolar error check (aggressive OpenVINS-style gating)
        inlier_idx = pose_mask.ravel().astype(bool)
        q1n_inliers = q1n.reshape(-1, 2)[inlier_idx]
        q2n_inliers = q2n.reshape(-1, 2)[inlier_idx]
        
        epipolar_errors = self._compute_epipolar_error(q1n_inliers, q2n_inliers, E)
        
        # OpenVINS uses ~1px threshold for epipolar errors
        final_inliers = epipolar_errors < self.epipolar_threshold
        num_final = np.sum(final_inliers)
        
        if num_final < VO_MIN_INLIERS:
            print(f"[VIO][EPIPOLAR] Rejected {len(final_inliers) - num_final} features by epipolar threshold")
            print(f"  Epipolar errors: median={np.median(epipolar_errors):.3f}, max={np.max(epipolar_errors):.3f}")
            return False, num_final, None, None, dt_img
        
        # Save inlier matched points for downstream processing
        try:
            # Rebuild full inlier mask (flow filter + RANSAC + epipolar)
            full_inlier_idx = np.zeros(len(flow_valid_mask), dtype=bool)
            full_inlier_idx[np.where(flow_valid_mask)[0][inlier_idx][final_inliers]] = True
            
            q1n_arr = self._undistort_pts(q1).reshape(-1, 2)
            q2n_arr = self._undistort_pts(q2).reshape(-1, 2)
            
            # Apply final inlier mask
            q1n_final = q1n_arr[inlier_idx][final_inliers]
            q2n_final = q2n_arr[inlier_idx][final_inliers]
            
            self.last_matches = (q1n_final.copy(), q2n_final.copy())
        except Exception as e:
            print(f"[VIO][MATCHES] Failed to save inlier matches: {e}")
            self.last_matches = None
        
        # Keyframe management decision
        should_keyframe, kf_reason = self._should_create_keyframe(img_gray)
        if should_keyframe:
            print(f"[VIO] Creating new keyframe at frame {self.frame_idx}: {kf_reason}")
            self.keyframe_gray = img_gray.copy()
            self.keyframe_frame_idx = self.frame_idx
            self.keyframe_tracked_ratio = 1.0
        
        self.last_frame_time = t
        
        # Replenish features using grid-based extraction (OpenVINS-style)
        try:
            num_active_tracks = sum(1 for v in self.tracks.values() if len(v)>0 and v[-1]['frame']==self.frame_idx)
            min_features = self.max_total_features // 2  # Maintain at least 50% feature count
            
            if num_active_tracks < min_features:
                # Extract new features using grid-based detection
                new_features = self._extract_grid_features(img_gray)
                
                if len(new_features) > 0:
                    # Filter out features too close to existing tracks (avoid clustering)
                    existing_pts = np.array([hist[-1]['pt'] for hist in self.tracks.values() 
                                            if hist and hist[-1]['frame'] == self.frame_idx])
                    
                    if len(existing_pts) > 0:
                        # Compute distances to existing features
                        from scipy.spatial.distance import cdist
                        try:
                            dists = cdist(new_features, existing_pts)
                            min_dists = np.min(dists, axis=1)
                            # Keep features at least 10px away from existing ones
                            far_enough = min_dists > 10.0
                            new_features = new_features[far_enough]
                        except:
                            pass  # If scipy not available, add all features
                    
                    # Add new tracks
                    new_fids = []
                    for p in new_features:
                        fid = self.next_fid
                        self.next_fid += 1
                        self.tracks[fid] = [{'frame': self.frame_idx, 'pt': (float(p[0]), float(p[1])), 'quality': 1.0}]
                        new_fids.append(fid)
                    
                    # Update tracking state
                    pts_all = []
                    fids_all = []
                    for fid, hist in self.tracks.items():
                        if hist and hist[-1]['frame'] == self.frame_idx:
                            pts_all.append(hist[-1]['pt'])
                            fids_all.append(fid)
                    
                    if len(pts_all) > 0:
                        self.last_pts_for_klt = np.array(pts_all, dtype=np.float32)
                        self.last_fids_for_klt = np.array(fids_all, dtype=np.int64)
                        self.last_gray_for_klt = img_gray.copy()
                        
                    print(f"[VIO][REPLENISH] Added {len(new_features)} grid-based features ({num_active_tracks} -> {len(pts_all)} total)")
        except Exception as e:
            print(f"[VIO][REPLENISH] Exception: {e}")
            pass
        
        return True, int(num_inl), R_vo, t_unit.reshape(-1), dt_img

    def get_tracks_for_frame(self, frame_idx: int):
        """Return list of (fid, pt) observed at given frame_idx."""
        res = []
        for fid, hist in self.tracks.items():
            if hist and hist[-1]['frame'] == frame_idx:
                res.append((fid, hist[-1]['pt']))
        return res
    
    def get_mature_tracks(self):
        """
        Return tracks that are ready for MSCKF update (length >= min_track_length).
        Returns: dict of fid -> track_history
        """
        mature = {}
        for fid, hist in self.tracks.items():
            if len(hist) >= self.min_track_length:
                mature[fid] = hist
        return mature
    
    def get_track_by_id(self, fid: int):
        """Get track history for a specific feature ID."""
        return self.tracks.get(fid, None)
    
    def get_multi_view_observations(self, fid: int, cam_ids: List[int], cam_states: List[dict]) -> List[dict]:
        """
        Get observations of a feature across multiple camera poses.
        
        Args:
            fid: Feature ID
            cam_ids: List of camera IDs to check
            cam_states: List of camera state metadata
        
        Returns: List of {'cam_id': int, 'frame': int, 'pt': (x,y), 'quality': float}
        """
        track = self.tracks.get(fid, None)
        if track is None:
            return []
        
        observations = []
        for obs in track:
            frame_idx = obs['frame']
            # Find matching camera state
            for i, cs in enumerate(cam_states):
                if cs['frame'] == frame_idx and i in cam_ids:
                    observations.append({
                        'cam_id': i,
                        'frame': frame_idx,
                        'pt': obs['pt'],
                        'quality': obs.get('quality', 1.0)
                    })
                    break
        return observations

# ===============================
# Helper: robust Mahalanobis distance for gating
# ===============================

def _mahalanobis2(y, S):
    y = np.asarray(y).reshape(-1, 1)  # force (n,1)
    S = np.asarray(S)
    if S.shape[0] != y.shape[0]:
        raise ValueError(f"Shape mismatch: S {S.shape}, y {y.shape}")
    try:
        sol = np.linalg.solve(S, y)
    except np.linalg.LinAlgError:
        return np.inf
    return float((y.T @ sol).ravel()[0])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in [w,x,y,z] format."""
    w1, x1, y1, z1 = q1.ravel()
    w2, x2, y2, z2 = q2.ravel()
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=float).reshape(4,1)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = q.reshape(4)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float).reshape(4,1)
    return (q / n).reshape(4,1)


def get_feature_multi_view_observations(fid: int, cam_observations: List[dict]) -> List[dict]:
    """
    Extract all observations of a specific feature across multiple camera poses.
    
    Args:
        fid: Feature ID
        cam_observations: List of observation records
    
    Returns: List of {'cam_id', 'pt_pixel', 'pt_norm', 'quality', 'frame', 't'}
    """
    multi_view_obs = []
    for obs_record in cam_observations:
        if 'observations' in obs_record:
            # New format with detailed observations
            for obs in obs_record['observations']:
                if obs['fid'] == fid:
                    multi_view_obs.append({
                        'cam_id': obs_record['cam_id'],
                        'pt_pixel': obs['pt_pixel'],
                        'pt_norm': obs['pt_norm'],
                        'quality': obs['quality'],
                        'frame': obs_record.get('frame', -1),
                        't': obs_record.get('t', 0.0)
                    })
        elif 'tracks' in obs_record:
            # Old format compatibility
            for track_fid, pt in obs_record['tracks']:
                if track_fid == fid:
                    multi_view_obs.append({
                        'cam_id': obs_record['cam_id'],
                        'pt_pixel': pt,
                        'pt_norm': pt,  # Assume already normalized in old format
                        'quality': 1.0,
                        'frame': -1,
                        't': 0.0
                    })
    return multi_view_obs


def find_mature_features_for_msckf(vio_fe: VIOFrontEnd, cam_observations: List[dict], 
                                   min_observations: int = 3) -> List[int]:
    """
    Find features that have been observed in multiple camera poses and are ready for MSCKF update.
    
    Args:
        vio_fe: VIO frontend with track information
        cam_observations: List of observation records
        min_observations: Minimum number of camera observations required
    
    Returns: List of feature IDs ready for MSCKF update
    """
    if vio_fe is None:
        return []
    
    mature_tracks = vio_fe.get_mature_tracks()
    print(f"[MSCKF][DEBUG] get_mature_tracks returned {len(mature_tracks)} tracks")
    
    mature_features = []
    
    for fid in mature_tracks.keys():
        # Count how many camera poses have observed this feature
        obs = get_feature_multi_view_observations(fid, cam_observations)
        if len(obs) >= min_observations:
            mature_features.append(fid)
    
    print(f"[MSCKF][DEBUG] {len(mature_features)} features have {min_observations}+ observations")
    
    return mature_features


# ===============================
# Helper: IMU to Camera Pose Transform
# ===============================

def imu_pose_to_camera_pose(q_imu: np.ndarray, p_imu: np.ndarray, 
                            T_body_cam: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert IMU/Body pose to camera pose using extrinsics.
    
    CRITICAL: Dataset notation is T_BC (Body→Camera), not T_CB!
    - BODY_T_CAMDOWN from calibration is Body→Camera transform
    - Must invert to get Camera→Body for pose conversion
    
    Pose transformation:
    - IMU/Body pose in world: (R_WB, t_WB) where q_imu represents R_WB
    - Camera pose in world: (R_WC, t_WC)
    - Extrinsics: T_BC (Body→Camera), need T_CB = T_BC^{-1}
    
    R_WC = R_WB @ R_BC^T  (rotate body-to-world then camera-to-body)
    t_WC = t_WB + R_WB @ t_CB  (body position + camera offset in world frame)
    
    Args:
        q_imu: IMU quaternion [w,x,y,z] representing R_WB (body-to-world) in ENU frame
        p_imu: IMU position [x,y,z] in world frame (ENU)
        T_body_cam: Body→Camera extrinsics T_BC (4x4 matrix)
    
    Returns:
        q_cam: Camera quaternion [w,x,y,z] representing R_WC (camera-to-world) in ENU
        p_cam: Camera position [x,y,z] in world frame (ENU)
    """
    if T_body_cam is None:
        T_body_cam = BODY_T_CAMDOWN
    
    # DEBUG: Check if BODY_T_CAMDOWN is correctly loaded (first call only)
    global _DEBUG_IMU_TO_CAM_CHECKED
    if '_DEBUG_IMU_TO_CAM_CHECKED' not in globals():
        _DEBUG_IMU_TO_CAM_CHECKED = True
        print(f"[DEBUG] imu_pose_to_camera_pose: BODY_T_CAMDOWN diagonal = {np.diag(T_body_cam[:3,:3])}")
        cam_z_in_body = T_body_cam[2, :3]
        print(f"[DEBUG] Camera Z axis in body frame: {cam_z_in_body}")
        print(f"[DEBUG] For nadir camera in FRD body: should point towards +Z (down)")
    
    # Extract Body→Camera transform
    R_BC = T_body_cam[:3, :3]  # Rotation: Body→Camera
    t_BC = T_body_cam[:3, 3]   # Translation: camera position in body frame
    
    # Invert to get Camera→Body
    R_CB = R_BC.T              # Camera→Body rotation
    t_CB = -R_BC.T @ t_BC      # Camera→Body translation
    
    # IMU/Body rotation in ENU frame:
    # CRITICAL: Xsens MTi-30 outputs quaternion in ENU world frame (Z-Up)
    # The state quaternion is already in ENU, NO conversion needed!
    q_imu_xyzw = np.array([q_imu[1], q_imu[2], q_imu[3], q_imu[0]])  # Convert [w,x,y,z] to scipy [x,y,z,w]
    R_BW = R_scipy.from_quat(q_imu_xyzw).as_matrix()  # Body-to-World (ENU)
    
    # Camera orientation in world frame:
    # R_BC is Body→Camera: p_cam = R_BC @ p_body
    # R_CB = R_BC.T is Camera→Body: p_body = R_CB @ p_cam
    # R_CW = R_BW @ R_CB transforms Camera→World: p_world = R_BW @ R_CB @ p_cam
    R_CW = R_BW @ R_CB  # Camera→World = (Body→World) @ (Camera→Body)
    q_cam_xyzw = R_scipy.from_matrix(R_CW).as_quat()
    q_cam = np.array([q_cam_xyzw[3], q_cam_xyzw[0], q_cam_xyzw[1], q_cam_xyzw[2]])
    
    # Camera position in world frame:
    # t_WC = t_WB + R_BW @ t_CB (body position + camera offset rotated to world)
    p_cam = p_imu + R_BW @ t_CB
    
    return q_cam, p_cam


# ===============================
# MSCKF Feature Triangulation
# ===============================

def triangulate_point_linear(observations: List[dict], cam_states: List[dict]) -> Optional[np.ndarray]:
    """
    Linear triangulation using Direct Linear Transform (DLT).
    
    Args:
        observations: List of {'cam_id', 'pt_norm': (x, y), ...}
        cam_states: List of camera state metadata with pose information
    
    Returns: 3D point in world frame [x, y, z] or None if failed
    """
    if len(observations) < 2:
        return None
    
    # Build linear system A * p = 0
    A = []
    for obs in observations:
        cam_id = obs['cam_id']
        if cam_id >= len(cam_states):
            continue
        
        # Get camera pose from state
        cs = cam_states[cam_id]
        # Note: In actual implementation, need to extract from kf.x
        # For now, assume we have R_cw and t_cw (camera to world)
        # This is placeholder - will be filled in actual MSCKF update
        
        # Normalized image coordinates
        x, y = obs['pt_norm']
        
        # Build constraint matrix (will be completed in full implementation)
        # A_i = [x*P[2,:] - P[0,:]]
        #       [y*P[2,:] - P[1,:]]
        # where P = K[R|t] projection matrix
        
    # Placeholder: return None for now, will implement with actual camera poses
    return None


def triangulate_point_nonlinear(observations: List[dict], cam_states: List[dict], 
                                p_init: np.ndarray, kf: ExtendedKalmanFilter,
                                max_iters: int = 10, debug: bool = False) -> Optional[np.ndarray]:
    """
    Nonlinear triangulation refinement using Gauss-Newton.
    
    Args:
        observations: List of observations
        cam_states: Camera state metadata
        p_init: Initial 3D point estimate
        kf: EKF with camera poses in state
        max_iters: Maximum iterations
    
    Returns: Refined 3D point or None
    """
    if p_init is None or len(observations) < 2:
        return None
    
    p = p_init.copy()
    
    for iteration in range(max_iters):
        # Build Jacobian and residual
        H = []
        r = []
        
        for obs in observations:
            cam_id = obs['cam_id']
            if cam_id >= len(cam_states):
                continue
            
            cs = cam_states[cam_id]
            # Extract IMU pose from state and transform to camera
            q_idx = cs['q_idx']
            p_idx = cs['p_idx']
            
            q_imu = kf.x[q_idx:q_idx+4, 0]  # [w,x,y,z]
            p_imu = kf.x[p_idx:p_idx+3, 0]  # [x,y,z]
            q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu)
            
            # Transform point to camera frame
            # q_cam contains R_CW (camera-to-world rotation from imu_pose_to_camera_pose)
            q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
            R_cw = R_scipy.from_quat(q_xyzw).as_matrix()  # Camera-to-World
            R_wc = R_cw.T  # World-to-Camera
            
            # Point in camera frame: p_c = R_wc @ (p_w - p_wc)
            p_c = R_wc @ (p - p_cam)
            
            if p_c[2] <= 0.1:  # Behind camera or too close
                if debug:
                    print(f"[NL-DBG] Fail: p_c[2]={p_c[2]:.2f}m (behind camera in iter {iteration})")
                    print(f"         p_init={p_init}, p={p}")
                    print(f"         p_cam={p_cam}, R_wc diag={np.diag(R_wc)}")
                return None
            
            # Project to normalized image plane
            x_pred = p_c[0] / p_c[2]
            y_pred = p_c[1] / p_c[2]
            
            # Residual
            x_obs, y_obs = obs['pt_norm']
            r.append(x_obs - x_pred)
            r.append(y_obs - y_pred)
            
            # Jacobian w.r.t. 3D point
            # d(u,v)/d(p_c) where u = x/z, v = y/z
            inv_z = 1.0 / p_c[2]
            inv_z2 = inv_z * inv_z
            
            J_proj = np.array([
                [inv_z, 0, -p_c[0] * inv_z2],
                [0, inv_z, -p_c[1] * inv_z2]
            ])
            
            # Chain rule: J = J_proj * R_wc
            J = J_proj @ R_wc
            H.append(J)
        
        if len(r) < 4:  # Need at least 2 observations
            if debug:
                print(f"[NL-DBG] Fail: len(r)={len(r)} < 4 (not enough observations)")
            return None
        
        H = np.vstack(H)  # (2N, 3)
        r = np.array(r).reshape(-1, 1)  # (2N, 1)
        
        # Levenberg-Marquardt damped update: dp = (H^T H + λI)^-1 H^T r
        # This prevents divergence for ill-conditioned systems (nadir cameras)
        try:
            HTH = H.T @ H
            HTr = H.T @ r
            
            # Adaptive damping: start with small λ, increase if diverging
            lambda_lm = 1e-3 * np.trace(HTH) / 3.0  # Scale with matrix magnitude
            lambda_lm = max(lambda_lm, 1e-6)  # Minimum damping
            
            # Add damping to diagonal (Levenberg-Marquardt)
            HTH_damped = HTH + lambda_lm * np.eye(3)
            
            dp = np.linalg.solve(HTH_damped, HTr)
            
            # Limit step size to prevent divergence
            max_step = 10.0  # Maximum update in meters
            dp_norm = np.linalg.norm(dp)
            if dp_norm > max_step:
                dp = dp * (max_step / dp_norm)
                if debug:
                    print(f"[NL-DBG] Clamped step from {dp_norm:.1f}m to {max_step:.1f}m")
            
            p = p + dp.reshape(3,)
            
            # Check convergence
            if np.linalg.norm(dp) < 1e-6:
                break
        except np.linalg.LinAlgError:
            if debug:
                print(f"[NL-DBG] Fail: LinAlgError in solve")
            return None
    
    return p


def triangulate_feature(fid: int, cam_observations: List[dict], cam_states: List[dict],
                        kf: ExtendedKalmanFilter, use_plane_constraint: bool = True,
                        ground_altitude: float = 0.0, debug: bool = False,
                        dem_reader: Optional['DEMReader'] = None,
                        origin_lat: float = 0.0, origin_lon: float = 0.0) -> Optional[dict]:
    """
    Triangulate a feature using multi-view observations.
    
    For nadir cameras: Use plane constraint (ground altitude) for better depth estimation
    since triangulation geometry is poor with downward-looking cameras.
    
    Tracks failure statistics in global MSCKF_STATS dict.
    
    1.4 INDIRECT VIO CONSTRAINT (NEW):
    - Use DEM to estimate ground elevation at camera position
    - Apply as soft constraint to triangulation (depth prior)
    - Only when xy_std < 10m (horizontal position reliable)
    - Error isolation: Bad DEM affects VIO local map, not EKF global state
    
    Args:
        fid: Feature ID
        cam_observations: All observation records
        cam_states: Camera state metadata
        kf: EKF with camera poses
        use_plane_constraint: Use ground plane to estimate depth (better for nadir)
        ground_altitude: Ground altitude in MSL (meters)
        dem_reader: DEM reader for ground plane constraint (1.4)
        origin_lat/origin_lon: Local projection origin for lat/lon conversion
    
    Returns: {'p_w': np.ndarray(3,), 'observations': List, 'quality': float} or None
    """
    # Get all observations of this feature
    obs_list = get_feature_multi_view_observations(fid, cam_observations)
    
    # Track statistics
    global MSCKF_STATS
    MSCKF_STATS['total_attempt'] += 1
    
    if len(obs_list) < 2:
        MSCKF_STATS['fail_few_obs'] += 1
        return None
    
    # ========== 1.4 INDIRECT VIO CONSTRAINT: DEM ground plane prior ==========
    # Check if DEM available and horizontal position reliable
    dem_ground_z = None
    xy_std = None
    
    if dem_reader is not None and dem_reader.ds is not None:
        # Compute XY uncertainty from covariance
        xy_uncertainty = float(np.trace(kf.P[0:2, 0:2]))
        xy_std = np.sqrt(xy_uncertainty / 2.0)
        
        # Only use DEM when horizontal position reliable (< 10m uncertainty)
        if xy_std < 10.0:
            # Get camera position in world frame (use first observation camera)
            obs0 = obs_list[0]
            cam_id0 = obs0['cam_id']
            
            if cam_id0 < len(cam_states):
                cs0 = cam_states[cam_id0]
                # Get IMU position from state
                p_imu0 = kf.x[cs0['p_idx']:cs0['p_idx']+3, 0]
                
                # Convert to lat/lon
                try:
                    lat, lon = xy_to_latlon(p_imu0[0], p_imu0[1], origin_lat, origin_lon)
                    
                    # Sample DEM at camera position
                    dem_ground_z = dem_reader.sample_m(lat, lon)
                    
                    if dem_ground_z is not None:
                        if debug:
                            print(f"[TRI-DEM] Feature {fid}: DEM ground={dem_ground_z:.1f}m, cam_z={p_imu0[2]:.1f}m, xy_std={xy_std:.2f}m")
                except Exception as e:
                    if debug:
                        print(f"[TRI-DEM] Feature {fid}: DEM lookup failed: {e}")
                    dem_ground_z = None
        else:
            if debug:
                print(f"[TRI-DEM] Feature {fid}: SKIP DEM (xy_std={xy_std:.2f}m > 10m threshold)")
    
    # Simple initialization: use first two views
    # In practice, could use linear DLT with all views
    if len(obs_list) >= 2:
        # Get first two camera poses for initialization
        obs0 = obs_list[0]
        obs1 = obs_list[1]
        
        cam_id0 = obs0['cam_id']
        cam_id1 = obs1['cam_id']
        
        if cam_id0 >= len(cam_states) or cam_id1 >= len(cam_states):
            return None
        
        cs0 = cam_states[cam_id0]
        cs1 = cam_states[cam_id1]
        
        # Extract IMU poses and convert to camera poses
        q_imu0 = kf.x[cs0['q_idx']:cs0['q_idx']+4, 0]
        p_imu0 = kf.x[cs0['p_idx']:cs0['p_idx']+3, 0]
        q0, p0 = imu_pose_to_camera_pose(q_imu0, p_imu0)
        
        q_imu1 = kf.x[cs1['q_idx']:cs1['q_idx']+4, 0]
        p_imu1 = kf.x[cs1['p_idx']:cs1['p_idx']+3, 0]
        q1, p1 = imu_pose_to_camera_pose(q_imu1, p_imu1)
        
        # Camera centers in world frame
        c0 = p0
        c1 = p1
        
        # CRITICAL: Check baseline length to prevent degenerate triangulation
        # For nadir cameras over flat terrain, insufficient baseline → poor depth estimation
        baseline = np.linalg.norm(c1 - c0)
        if debug and baseline < MIN_MSCKF_BASELINE:
            print(f"[TRI-DBG] Feature {fid}: FAIL baseline ({baseline:.3f}m < {MIN_MSCKF_BASELINE}m)")
        if baseline < MIN_MSCKF_BASELINE:
            # Insufficient baseline for reliable triangulation
            MSCKF_STATS['fail_baseline'] += 1
            return None
        
        # Ray directions in world frame
        # CRITICAL FIX: q0/q1 represent camera-to-world rotation (R_CW)
        # Quaternion → matrix gives R_CW directly (NO transpose needed!)
        # Legacy code confirms: R_body_to_world = R_scipy.from_quat(q_xyzw).as_matrix()
        q0_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]])
        R0_cw = R_scipy.from_quat(q0_xyzw).as_matrix()  # R_CW (camera-to-world) - REMOVED .T
        
        q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]])
        R1_cw = R_scipy.from_quat(q1_xyzw).as_matrix()  # R_CW (camera-to-world) - REMOVED .T
        
        # Rays in camera frame (normalized coordinates)
        # For pinhole model: ray = [x_norm, y_norm, 1] / norm
        # This is the standard convention where Z+ is forward (optical axis)
        x0, y0 = obs0['pt_norm']
        ray0_c = np.array([x0, y0, 1.0])
        ray0_c = ray0_c / np.linalg.norm(ray0_c)
        
        x1, y1 = obs1['pt_norm']
        ray1_c = np.array([x1, y1, 1.0])
        ray1_c = ray1_c / np.linalg.norm(ray1_c)
        
        # Transform rays to world frame
        ray0_w = R0_cw @ ray0_c
        ray0_w = ray0_w / np.linalg.norm(ray0_w)
        
        ray1_w = R1_cw @ ray1_c
        ray1_w = ray1_w / np.linalg.norm(ray1_w)
        
        # NEW: Check parallax angle (angular separation between rays)
        # Even with short baseline, sufficient parallax can enable triangulation
        # Minimum parallax angle: 0.5 degrees (prevents degenerate near-parallel rays)
        MIN_PARALLAX_ANGLE_DEG = 0.3  # Very permissive for nadir cameras
        ray_angle_rad = np.arccos(np.clip(np.dot(ray0_w, ray1_w), -1, 1))
        ray_angle_deg = np.degrees(ray_angle_rad)
        
        if ray_angle_deg < MIN_PARALLAX_ANGLE_DEG:
            if debug:
                print(f"[TRI-DBG] Feature {fid}: FAIL parallax angle ({ray_angle_deg:.3f}° < {MIN_PARALLAX_ANGLE_DEG}°)")
            MSCKF_STATS['fail_parallax'] += 1
            return None
        
        # DEBUG: Check ray Z direction - for nadir camera, ray should point DOWN (Z<0 in ENU)
        if debug:
            # Check camera Z axis (optical axis) in world
            cam_z_world = R0_cw @ np.array([0, 0, 1])
            print(f"[RAY-DBG] Feature {fid}:")
            print(f"  q0 (wxyz) = {q0}")
            print(f"  cam Z axis in world = {cam_z_world} (Z should be < 0 for nadir)")
            print(f"  ray0_c (unnorm) = [{x0:.4f}, {y0:.4f}, 1.0]")
            print(f"  ray0_w = {ray0_w} (Z should be < 0 for nadir cam)")
            print(f"  ray1_w = {ray1_w}")
            print(f"  parallax angle = {ray_angle_deg:.2f}°")
            print(f"  cam0_z = {c0[2]:.1f}m")  # Camera altitude
        
        # Mid-point method for initialization
        # Find closest points on two rays
        # https://en.wikipedia.org/wiki/Skew_lines#Nearest_points
        w = c0 - c1
        a = np.dot(ray0_w, ray0_w)
        b = np.dot(ray0_w, ray1_w)
        c = np.dot(ray1_w, ray1_w)
        d = np.dot(ray0_w, w)
        e = np.dot(ray1_w, w)
        
        denom = a * c - b * b
        if debug and abs(denom) < 1e-6:
            print(f"[TRI-DBG] Feature {fid}: FAIL parallel rays (denom={denom:.2e}, baseline={baseline:.3f}m)")
        if abs(denom) < 1e-6:  # Parallel rays
            MSCKF_STATS['fail_solver'] += 1
            return None
        
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        
        # Points on rays
        p_ray0 = c0 + s * ray0_w
        p_ray1 = c1 + t * ray1_w
        
        # Mid-point as initial estimate
        p_init = (p_ray0 + p_ray1) / 2.0
        
        # CRITICAL: Validate midpoint for nadir camera
        # For downward-looking camera: point should be BELOW camera (Z < cam_z)
        # If midpoint is above camera or s < 0, the triangulation failed completely
        midpoint_valid = True
        if s < 0 or t < 0:
            # Negative s/t means point is behind camera on the ray
            midpoint_valid = False
            if debug:
                print(f"[TRI-DBG] Feature {fid}: Midpoint behind camera (s={s:.2f}, t={t:.2f})")
        elif p_init[2] > c0[2]:
            # Point is above camera - impossible for ground features with nadir camera
            midpoint_valid = False
            if debug:
                print(f"[TRI-DBG] Feature {fid}: Midpoint above camera (p_z={p_init[2]:.1f}m > cam_z={c0[2]:.1f}m)")
        
        # ========== 1.4 DEM CONSTRAINT: Apply ground plane prior ==========
        # If DEM available and reliable, use it to improve depth estimate
        # CRITICAL FIX: For nadir cameras, DEM constraint is essential because
        # midpoint triangulation often fails due to near-parallel rays
        if dem_ground_z is not None:
            # Compute expected depth from camera to ground plane
            # Nadir camera: depth ≈ camera_z - ground_z (for vertical rays)
            cam_z = c0[2]  # Camera altitude MSL
            expected_depth = cam_z - dem_ground_z
            
            if expected_depth > 5.0 and expected_depth < 500.0:
                # FIX: Instead of blending Z, project ray onto ground plane
                # This maintains geometric consistency between ray and point
                
                # Compute where ray0 intersects z = dem_ground_z plane
                # Ray: p = c0 + t * ray0_w
                # Plane: z = dem_ground_z
                # Solve: c0[2] + t * ray0_w[2] = dem_ground_z
                ray_z = ray0_w[2]
                if abs(ray_z) > 0.1:  # Ray has significant vertical component
                    t_ground = (dem_ground_z - c0[2]) / ray_z
                    if t_ground > 5.0:  # Point is in front of camera, reasonable depth
                        p_dem = c0 + t_ground * ray0_w
                        
                        # CRITICAL FIX: For nadir cameras with small baseline relative to depth,
                        # the midpoint method FAILS because rays are nearly parallel.
                        # Use DEM ray intersection as PRIMARY method in this case.
                        # 
                        # Baseline-to-depth ratio determines triangulation quality:
                        # - ratio > 0.15 (15%): Reasonable triangulation geometry
                        # - ratio < 0.15: Poor geometry, use DEM constraint heavily
                        baseline_depth_ratio = baseline / t_ground
                        
                        # CRITICAL FIX: If midpoint is invalid (behind camera or above camera),
                        # use DEM 100% - don't blend with garbage
                        if not midpoint_valid:
                            weight_dem = 1.0
                            if debug:
                                print(f"[TRI-DEM] Feature {fid}: Using DEM 100% (invalid midpoint)")
                        elif baseline_depth_ratio < 0.05:
                            # Very poor triangulation geometry - use DEM 100%
                            weight_dem = 1.0
                            if debug:
                                print(f"[TRI-DEM] Feature {fid}: Using DEM 100% (poor geometry, B/D={baseline_depth_ratio:.3f})")
                        elif baseline_depth_ratio < 0.15:
                            # Marginal geometry - strong DEM prior (INCREASED from 0.1)
                            weight_dem = 0.95
                        else:
                            # Reasonable geometry - moderate DEM prior
                            weight_dem = 0.8
                        
                        p_init = weight_dem * p_dem + (1 - weight_dem) * p_init
                        
                        if debug:
                            print(f"[TRI-DEM] Feature {fid}: Applied DEM constraint (ray intersection)")
                            print(f"  Ground Z: {dem_ground_z:.1f}m, Cam Z: {cam_z:.1f}m")
                            print(f"  Ray depth to ground: {t_ground:.1f}m")
                            print(f"  B/D ratio: {baseline_depth_ratio:.3f}, DEM weight: {weight_dem:.1f}")
                            print(f"  p_dem: {p_dem}")
                            print(f"  p_init (blended): {p_init}")
                    else:
                        if debug:
                            print(f"[TRI-DEM] Feature {fid}: SKIP (t_ground={t_ground:.1f}m invalid)")
                else:
                    if debug:
                        print(f"[TRI-DEM] Feature {fid}: SKIP (ray too horizontal, ray_z={ray_z:.3f})")
            else:
                if debug:
                    print(f"[TRI-DEM] Feature {fid}: SKIP constraint (unrealistic depth={expected_depth:.1f}m)")
        
        # Check if point is in front of cameras
        depth0 = np.dot(p_init - c0, ray0_w)
        depth1 = np.dot(p_init - c1, ray1_w)
        
        # DEBUG: Print detailed triangulation info
        if debug:
            print(f"[TRI-DBG] Feature {fid}:")
            print(f"  Cam0 pos: {c0}")
            print(f"  Cam1 pos: {c1}")
            print(f"  Baseline: {baseline:.3f}m")
            print(f"  Ray0_w: {ray0_w}")
            print(f"  Ray1_w: {ray1_w}")
            print(f"  Point: {p_init}")
            print(f"  Depth0: {depth0:.1f}m, Depth1: {depth1:.1f}m")
        
        # For nadir camera: Expect depth ~30-100m (drone altitude)
        # If triangulation gives unrealistic depth, reject it
        if debug and (depth0 < 5.0 or depth1 < 5.0):
            print(f"[TRI-DBG] Feature {fid}: FAIL depth too small (d0={depth0:.1f}m, d1={depth1:.1f}m)")
        if depth0 < 5.0 or depth1 < 5.0:
            # Negative or very small depth → feature behind camera or too close
            MSCKF_STATS['fail_depth_sign'] += 1
            return None
        
        # Check if depth is unrealistically large (>500m for drone at ~60m altitude)
        if debug and (depth0 > 500.0 or depth1 > 500.0):
            print(f"[TRI-DBG] Feature {fid}: FAIL depth too large (d0={depth0:.1f}m, d1={depth1:.1f}m)")
        if depth0 > 500.0 or depth1 > 500.0:
            # Triangulation failed → parallel rays or bad geometry
            MSCKF_STATS['fail_depth_large'] += 1
            return None

        
        # Nonlinear refinement
        p_refined = triangulate_point_nonlinear(obs_list, cam_states, p_init, kf, debug=debug)
        
        if debug and p_refined is None:
            print(f"[TRI-DBG] Feature {fid}: FAIL nonlinear refinement")
        if p_refined is None:
            MSCKF_STATS['fail_nonlinear'] += 1
            return None
        
        # Compute reprojection error as quality metric
        total_error = 0.0
        for obs in obs_list:
            cam_id = obs['cam_id']
            if cam_id >= len(cam_states):
                continue
            
            cs = cam_states[cam_id]
            q_imu = kf.x[cs['q_idx']:cs['q_idx']+4, 0]
            p_imu = kf.x[cs['p_idx']:cs['p_idx']+3, 0]
            q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu)
            
            # Transform to camera frame
            # q_cam contains R_CW (camera-to-world) from imu_pose_to_camera_pose
            q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
            R_cw = R_scipy.from_quat(q_xyzw).as_matrix()  # Camera-to-World
            R_wc = R_cw.T  # World-to-Camera
            p_c = R_wc @ (p_refined - p_cam)
            
            if p_c[2] <= 0.1:
                # Feature behind camera
                MSCKF_STATS['fail_depth_sign'] += 1
                return None
            
            # Project
            x_pred = p_c[0] / p_c[2]
            y_pred = p_c[1] / p_c[2]
            
            x_obs, y_obs = obs['pt_norm']
            error = np.sqrt((x_obs - x_pred)**2 + (y_obs - y_pred)**2)
            total_error += error
        
        avg_error = total_error / len(obs_list)
        
        # Reject if reprojection error too high
        # VERY RELAXED: 0.10 for nadir cameras with DEM constraint
        # (DEM-constrained triangulation has lower XY accuracy but valid depth)
        MAX_REPROJ_ERROR = 0.10  # Increased from 0.05 for nadir cameras
        if avg_error > MAX_REPROJ_ERROR:
            if debug:
                print(f"[TRI-DBG] Feature {fid}: FAIL reproj error ({avg_error:.4f} > {MAX_REPROJ_ERROR})")
            MSCKF_STATS['fail_reproj_error'] += 1
            return None
        
        # SUCCESS!
        MSCKF_STATS['success'] += 1
        return {
            'p_w': p_refined,
            'observations': obs_list,
            'quality': 1.0 / (1.0 + avg_error * 100.0),
            'avg_reproj_error': avg_error
        }
    
    MSCKF_STATS['fail_other'] += 1
    return None


# ===============================
# MSCKF Measurement Jacobians
# ===============================

def compute_huber_weights(normalized_residuals: np.ndarray, threshold: float = 1.345) -> np.ndarray:
    """
    Compute Huber robust loss weights for outlier rejection.
    
    Huber loss:
    - ρ(r) = r²/2           if |r| ≤ δ
    - ρ(r) = δ|r| - δ²/2    if |r| > δ
    
    Weight: w(r) = √(ρ'(r)/r) where ρ'(r) is the derivative
    - w(r) = 1              if |r| ≤ δ
    - w(r) = δ/|r|          if |r| > δ
    
    Args:
        normalized_residuals: Residuals normalized by measurement std (r/σ)
        threshold: Huber threshold (default 1.345 ≈ 95% efficiency for Gaussian)
    
    Returns:
        weights: Weight for each residual element (same shape as input)
    """
    r_abs = np.abs(normalized_residuals)
    weights = np.ones_like(normalized_residuals)
    
    # For large residuals: downweight proportional to 1/|r|
    outlier_mask = r_abs > threshold
    weights[outlier_mask] = threshold / r_abs[outlier_mask]
    
    return weights


def compute_measurement_jacobian(p_w: np.ndarray, cam_state: dict, kf: ExtendedKalmanFilter,
                                 err_state_size: int, use_preint_jacobians: bool = True,
                                 T_cam_imu: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute measurement Jacobian for one camera observation in ERROR-STATE formulation.
    
    NEW (OpenVINS-style): Clone stores IMU pose, use extrinsics to get camera pose
    
    Measurement model: z = h(x_imu, T_cam_imu, p_w) where z is normalized image coordinates
    We compute: H = dh/d(δx) (Jacobian w.r.t. error state)
    
    Error state structure:
    - Core: [δp(3), δv(3), δθ(3), δbg(3), δba(3)] = 15 dims
    - Clones: [δθ_IMU(3), δp_IMU(3)] per clone = 6 dims each (IMU pose!)
    
    NEW: Preintegration Jacobians (if available):
    - IMU pose depends on biases via preintegration:
      δp_imu = ... + R * J_p_bg * δbg + R * J_p_ba * δba
      δθ_imu = ... + J_R_bg * δbg
    - This couples visual residuals → bias errors → reduces drift
    
    Args:
        p_w: 3D point in world frame
        cam_state: IMU state metadata with 'err_q_idx', 'err_p_idx', 'q_fej', 'p_fej', 'preint'
        kf: EKF with full state (nominal state x, error-state covariance P)
        err_state_size: Total ERROR state dimension (15 + 6*num_clones)
        use_preint_jacobians: Whether to add IMU preintegration terms (default: True)
        T_cam_imu: Camera-to-IMU extrinsics (4x4 transform matrix)
    
    Returns: (H_cam, H_feat) where
        H_cam: (2, err_state_size) - Jacobian w.r.t. error state (includes bias coupling)
        H_feat: (2, 3) - Jacobian w.r.t. feature position
    """
    # Default extrinsics (nadir camera) if not provided
    if T_cam_imu is None:
        T_cam_imu = BODY_T_CAMDOWN
    
    R_cam_imu = T_cam_imu[:3, :3]  # Rotation from IMU to camera
    t_cam_imu = T_cam_imu[:3, 3]   # Translation from IMU to camera
    
    # === FEJ: Use first-estimate for linearization (if available) ===
    # This ensures consistent Jacobian computation across multiple observations
    if 'q_fej' in cam_state and 'p_fej' in cam_state:
        # Use FEJ linearization point (saved at clone time) - NOW IMU POSE!
        q_imu = cam_state['q_fej']  # [w,x,y,z] IMU orientation
        p_imu = cam_state['p_fej']  # [x,y,z] IMU position
    else:
        # Fallback: Use current NOMINAL state (backward compatible)
        q_idx = cam_state['q_idx']  # Nominal quaternion index (4D)
        p_idx = cam_state['p_idx']  # Nominal position index (3D)
        
        q_imu = kf.x[q_idx:q_idx+4, 0]  # [w,x,y,z]
        p_imu = kf.x[p_idx:p_idx+3, 0]  # [x,y,z]
    
    # Transform IMU pose to camera pose via extrinsics
    q_imu_xyzw = np.array([q_imu[1], q_imu[2], q_imu[3], q_imu[0]])
    R_w_imu = R_scipy.from_quat(q_imu_xyzw).as_matrix()  # World to IMU rotation
    
    # Camera pose in world frame:
    # R_w_cam = R_w_imu @ R_imu_cam = R_w_imu @ R_cam_imu^T
    R_w_cam = R_w_imu @ R_cam_imu.T
    # p_cam = p_imu + R_w_imu @ t_cam_imu
    p_cam = p_imu + R_w_imu @ t_cam_imu
    
    # Transform point to camera frame
    # p_c = R_cam_w @ (p_w - p_cam) = R_w_cam^T @ (p_w - p_cam)
    p_rel = p_w - p_cam
    p_c = R_w_cam.T @ p_rel
    
    # Projection to normalized image plane
    # z = [u, v]^T where u = x/z, v = y/z
    inv_z = 1.0 / p_c[2]
    inv_z2 = inv_z * inv_z
    
    # Jacobian of projection w.r.t. p_c: dz/dp_c
    j_proj = np.array([
        [inv_z, 0, -p_c[0] * inv_z2],
        [0, inv_z, -p_c[1] * inv_z2]
    ])
    
    # === Jacobian w.r.t. feature position ===
    # dz/dp_w = J_proj * R_cam_w = J_proj * R_w_cam^T
    h_feat = j_proj @ R_w_cam.T  # (2, 3)
    
    # === Jacobian w.r.t. ERROR state (IMU pose!) ===
    h_cam = np.zeros((2, err_state_size))
    
    # Get error-state indices for this clone (IMU pose)
    err_theta_idx = cam_state['err_q_idx']  # Error rotation (3D) - IMU!
    err_p_idx = cam_state['err_p_idx']      # Error position (3D) - IMU!
    
    # === Chain rule for camera-IMU extrinsics ===
    # z = h(p_cam(p_imu, q_imu))
    # dz/d(δx_imu) = dz/dp_cam * dp_cam/d(δx_imu) + dz/dR_cam * dR_cam/d(δx_imu)
    
    # 1. Jacobian w.r.t. error in IMU position: ∂z/∂(δp_imu)
    # Camera position: p_cam = p_imu + R_w_imu @ t_cam_imu
    # Error: δp_cam = δp_imu + [δθ_imu]× @ (R_w_imu @ t_cam_imu)
    #              ≈ δp_imu - [R_w_imu @ t_cam_imu]× @ δθ_imu
    # 
    # ∂p_c/∂(δp_imu) = -R_cam_w = -R_w_cam^T
    h_cam[:, err_p_idx:err_p_idx+3] = j_proj @ (-R_w_cam.T)
    
    # 2. Jacobian w.r.t. error in IMU rotation: ∂z/∂(δθ_imu)
    # Camera rotation: R_w_cam = R_w_imu @ R_cam_imu^T
    # For small rotation error δθ_imu:
    # R_w_imu_perturbed = R_w_imu @ exp([δθ_imu]×) ≈ R_w_imu @ (I + [δθ_imu]×)
    # R_w_cam_perturbed = R_w_imu @ (I + [δθ_imu]×) @ R_cam_imu^T
    #                   = R_w_cam + R_w_imu @ [δθ_imu]× @ R_cam_imu^T
    # 
    # Point in camera frame:
    # p_c = R_cam_w @ p_rel = R_w_cam^T @ (p_w - p_cam)
    # 
    # Full perturbation includes both rotation and position effects:
    # δp_c = -R_cam_w @ [R_w_imu @ t_cam_imu]× @ δθ_imu - R_cam_w @ [p_rel]× @ (R_cam_imu @ δθ_imu)
    # 
    # Simplified (dominant term from p_rel rotation):
    # ∂p_c/∂(δθ_imu) ≈ -R_cam_w @ [p_rel + R_w_imu @ t_cam_imu]× @ R_cam_imu
    
    # Position offset effect
    t_cam_world = R_w_imu @ t_cam_imu
    p_rel_total = p_rel + t_cam_world
    skew_p_rel_total = skew_symmetric(p_rel_total)
    j_rot = j_proj @ (-R_w_cam.T @ skew_p_rel_total @ R_cam_imu)  # (2, 3)
    h_cam[:, err_theta_idx:err_theta_idx+3] = j_rot
    
    # === NEW: Add IMU preintegration Jacobian coupling (bias → IMU pose) ===
    # If this clone has preintegration data, IMU pose errors depend on bias errors:
    #   δθ_imu = J_R_bg * δbg
    #   δp_imu = R * J_p_bg * δbg + R * J_p_ba * δba
    # Camera pose inherits these errors via extrinsics
    # This couples visual measurements to bias estimation → reduces drift!
    
    if use_preint_jacobians and 'preint' in cam_state and cam_state['preint'] is not None:
        preint = cam_state['preint']
        
        # Get preintegration Jacobians
        J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba = preint.get_jacobians()
        
        # Get FEJ bias linearization points (for consistency)
        if 'bg_fej' in cam_state and 'ba_fej' in cam_state:
            # Use FEJ: rotation at clone time (IMU orientation!)
            R_clone = R_w_imu  # IMU-to-world rotation at clone time
            bg_fej = cam_state['bg_fej']
            ba_fej = cam_state['ba_fej']
            
            # Rotation matrix at clone time (IMU-to-world, for proper coordinate transform)
            R_clone = R_w_imu  # Already computed above from q_imu
        else:
            # Fallback: use current state
            R_clone = R_w_imu
        
        # 3. Jacobian w.r.t. gyro bias: ∂z/∂(δbg)
        # IMU rotation error: δθ_imu = J_R_bg * δbg
        # Camera inherits via extrinsics: δθ_cam includes IMU rotation error
        # Propagate through rotation Jacobian (with camera transform)
        h_cam[:, 9:12] += j_rot @ R_cam_imu @ J_R_bg  # Gyro bias [9:12]
        
        # 4. Jacobian w.r.t. accel bias: ∂z/∂(δba)
        # IMU position error: δp_imu = R_clone * J_p_ba * δba
        # Camera position inherits: δp_cam = δp_imu (extrinsics are fixed)
        j_pos = j_proj @ (-R_w_cam.T)  # Position Jacobian
        h_cam[:, 12:15] += j_pos @ R_clone @ J_p_ba  # Accel bias [12:15]
        
        # Gyro bias also affects position via rotation coupling
        h_cam[:, 9:12] += j_pos @ R_clone @ J_p_bg  # Additional coupling
    
    return h_cam, h_feat


def compute_observability_nullspace(kf: ExtendedKalmanFilter, num_clones: int) -> np.ndarray:
    """
    Compute nullspace basis for unobservable directions in MSCKF.
    
    Unobservable modes for VIO (monocular camera + IMU):
    1. Global position (X, Y): Cannot observe absolute position, only relative motion
    2. Global yaw: Cannot observe absolute heading without external reference
    
    These modes correspond to gauge freedom in the state estimation problem.
    If not constrained, the filter will gain spurious information from these directions,
    leading to overconfidence and eventual divergence.
    
    Solution (FEJ - First Estimate Jacobian):
    Build nullspace basis U where each column represents a direction of zero information:
    - u_x: Gradient of all states w.r.t. global X translation
    - u_y: Gradient of all states w.r.t. global Y translation  
    - u_yaw: Gradient of all states w.r.t. global yaw rotation
    
    Args:
        kf: EKF with current state estimate
        num_clones: Number of camera clones in state
    
    Returns:
        U: (err_state_size, 3) matrix where columns span unobservable directions
    """
    err_state_size = 15 + 6 * num_clones
    
    # Extract current orientation from nominal state
    q_imu = kf.x[6:10, 0]  # IMU quaternion [w,x,y,z]
    R_body_to_world = quat_to_rot(q_imu)
    
    # Compute yaw angle (rotation around Z-axis)
    yaw = quaternion_to_yaw(q_imu)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # Initialize nullspace basis (3 columns: X, Y, yaw)
    U = np.zeros((err_state_size, 3), dtype=float)
    
    # === Column 1: Global X translation ===
    # When entire trajectory shifts in X direction:
    # - δp_x = 1 (core IMU position)
    # - All camera clone positions shift by same amount: δp_Ci_x = 1
    U[0, 0] = 1.0  # δp_x (core IMU)
    for i in range(num_clones):
        clone_p_idx = 15 + 6*i + 3  # Error-state position index for clone i
        U[clone_p_idx, 0] = 1.0     # δp_Ci_x
    
    # === Column 2: Global Y translation ===
    # Similar to X, but in Y direction
    U[1, 1] = 1.0  # δp_y (core IMU)
    for i in range(num_clones):
        clone_p_idx = 15 + 6*i + 3
        U[clone_p_idx + 1, 1] = 1.0  # δp_Ci_y
    
    # === Column 3: Global yaw rotation ===
    # When entire trajectory rotates around Z-axis:
    # For small rotation δψ around Z-axis centered at origin:
    # - Position change: δp = [-y*δψ, x*δψ, 0]
    # - Orientation change: δθ = [0, 0, δψ]
    
    # Core IMU orientation
    U[8, 2] = 1.0  # δθ_z (yaw component of rotation error)
    
    # Core IMU position (rotate around origin)
    p_imu = kf.x[0:3, 0]
    U[0, 2] = -p_imu[1]  # δp_x = -y * δψ
    U[1, 2] =  p_imu[0]  # δp_y =  x * δψ
    
    # Camera clones
    core_size = 16  # Nominal state core size
    for i in range(num_clones):
        # Get camera clone position from nominal state
        clone_q_idx = core_size + 7*i      # Quaternion index (nominal)
        clone_p_idx = core_size + 7*i + 4  # Position index (nominal)
        
        p_cam = kf.x[clone_p_idx:clone_p_idx+3, 0]
        
        # Error-state indices
        err_theta_idx = 15 + 6*i      # δθ_Ci
        err_p_idx = 15 + 6*i + 3      # δp_Ci
        
        # Camera orientation change
        U[err_theta_idx + 2, 2] = 1.0  # δθ_Ci_z = δψ
        
        # Camera position change (rotate around origin)
        U[err_p_idx, 2]     = -p_cam[1]  # δp_Ci_x = -y * δψ
        U[err_p_idx + 1, 2] =  p_cam[0]  # δp_Ci_y =  x * δψ
    
    # Orthonormalize columns (Gram-Schmidt)
    # This ensures numerical stability when projecting
    U_ortho = np.zeros_like(U)
    for j in range(3):
        # Start with j-th column
        u_j = U[:, j].copy()
        # Subtract projections onto previous columns
        for k in range(j):
            u_j -= np.dot(U_ortho[:, k], u_j) * U_ortho[:, k]
        # Normalize
        norm = np.linalg.norm(u_j)
        if norm > 1e-10:
            U_ortho[:, j] = u_j / norm
        else:
            # Column is linearly dependent, leave as zero
            pass
    
    return U_ortho


def msckf_measurement_update(fid: int, triangulated: dict, cam_observations: List[dict],
                             cam_states: List[dict], kf: ExtendedKalmanFilter,
                             measurement_noise: float = 1e-4,
                             huber_threshold: float = 1.345,
                             chi2_max_dof: float = 15.36) -> tuple:  # RELAXED: 15.36 = chi2_4dof at 99.5% confidence (was 3.84 = 95%)
    """
    MSCKF measurement update with observability constraints (OpenVINS-style).
    
    This implements the core MSCKF update with FEJ (First Estimate Jacobian):
    1. Compute residuals for all observations
    2. Compute Jacobians (error-state: H_x is 2N × (15+6M))
    3. Apply null-space projection to remove unobservable feature depth
    4. Apply observability constraints (FEJ): project H orthogonal to unobservable modes
       - Global position (X, Y): VIO cannot observe absolute position
       - Global yaw: VIO cannot observe absolute heading
    5. Apply Huber robust weighting to downweight outliers
    6. Chi-square gating to reject bad measurements
    7. Update EKF error-state
    
    Args:
        fid: Feature ID
        triangulated: Triangulation result with 'p_w' and 'observations'
        cam_observations: All observation records
        cam_states: Camera state metadata
        kf: EKF to update (with error-state covariance)
        measurement_noise: Measurement noise variance (in normalized coords)
        huber_threshold: Huber loss threshold (default 1.345 for 95% efficiency)
        chi2_max_dof: Max chi-square per DoF for gating (default 3.84 for 95% CI)
    
    Returns: (success, innovation_norm, chi2_test) tuple
    """
    p_w = triangulated['p_w']
    obs_list = triangulated['observations']
    
    if len(obs_list) < 2:
        return (False, np.nan, np.nan)
    
    # ERROR-state dimension (not nominal state!)
    err_state_size = kf.P.shape[0]  # 15 + 6*num_clones
    num_obs = len(obs_list)
    meas_size = 2 * num_obs  # Each observation contributes 2 measurements (u, v)
    
    # === Step 1: Compute residuals and Jacobians ===
    residuals = []
    h_x_stack = []  # Jacobians w.r.t. ERROR state
    h_f_stack = []  # Jacobians w.r.t. feature
    
    for obs in obs_list:
        cam_id = obs['cam_id']
        if cam_id >= len(cam_states):
            continue
        
        cs = cam_states[cam_id]
        
        # Predicted measurement (use NOMINAL state)
        q_imu = kf.x[cs['q_idx']:cs['q_idx']+4, 0]
        p_imu = kf.x[cs['p_idx']:cs['p_idx']+3, 0]
        q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu)
        
        # q_cam contains R_CW (camera-to-world) from imu_pose_to_camera_pose
        q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
        r_cw = R_scipy.from_quat(q_xyzw).as_matrix()  # Camera-to-World
        r_wc = r_cw.T  # World-to-Camera
        p_c = r_wc @ (p_w - p_cam)
        
        if p_c[2] <= 0.1:  # Behind camera
            continue
        
        z_pred = np.array([p_c[0] / p_c[2], p_c[1] / p_c[2]])
        
        # Observed measurement
        z_obs = np.array([obs['pt_norm'][0], obs['pt_norm'][1]])
        
        # Residual
        r = z_obs - z_pred
        residuals.append(r)
        
        # Jacobians (error-state!)
        h_cam, h_feat = compute_measurement_jacobian(p_w, cs, kf, err_state_size)
        h_x_stack.append(h_cam)
        h_f_stack.append(h_feat)
    
    if len(residuals) < 2:
        return (False, np.nan, np.nan)
    
    # Stack into matrices
    r_o = np.vstack(residuals).reshape(-1, 1)  # (2N, 1)
    h_x = np.vstack(h_x_stack)  # (2N, err_state_size)
    h_f = np.vstack(h_f_stack)  # (2N, 3)
    
    # === Step 2: Nullspace projection ===
    # Key MSCKF step: project H_x to left null-space of H_f
    # This removes dependence on unobservable feature position
    
    try:
        # Compute SVD of H_f: H_f = U @ diag(S) @ V^T
        u_mat, s_mat, vh_mat = np.linalg.svd(h_f, full_matrices=True)
        
        # Determine rank (feature should have rank 3 in general position)
        tol = 1e-6 * s_mat[0] if len(s_mat) > 0 else 1e-6
        rank = np.sum(s_mat > tol)
        
        # Left null-space: columns of U beyond rank
        # Q2 = U[:, rank:] such that Q2^T @ H_f ≈ 0
        # For typical MSCKF: h_f is (2N, 3), rank=3, so null_space is (2N, 2N-3)
        null_space = u_mat[:, rank:]  # (2N, 2N-rank)
        
        # Project measurements and Jacobian to nullspace
        # This removes dependence on unobservable feature position
        h_proj = null_space.T @ h_x  # (2N-rank, err_state_size)
        r_proj = null_space.T @ r_o  # (2N-rank, 1)
        
    except np.linalg.LinAlgError:
        # Fallback: use unprojected (not recommended)
        h_proj = h_x
        r_proj = r_o
    
    # === Step 2b: Observability constraint (FEJ - First Estimate Jacobian) ===
    # Project out unobservable directions to prevent spurious information gain
    # Unobservable modes: global X, Y position and global yaw
    
    # Compute number of clones from error-state dimension
    num_clones = (err_state_size - 15) // 6
    
    try:
        # Build nullspace basis for unobservable directions
        U_obs = compute_observability_nullspace(kf, num_clones)  # (err_state_size, 3)
        
        # Project H to be orthogonal to unobservable directions
        # H_constrained = H - H @ U @ (U^T @ U)^{-1} @ U^T
        # Simplified: H_constrained = (I - U @ U^T) @ H when U is orthonormal
        
        # Since U is orthonormalized, U @ U^T is projection matrix
        projection_matrix = np.eye(err_state_size) - U_obs @ U_obs.T
        
        # Apply constraint to Jacobian
        h_constrained = h_proj @ projection_matrix  # (meas_dim, err_state_size)
        
        # Residuals remain unchanged (constraint is on information, not measurement)
        r_constrained = r_proj
        
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fallback: use unconstrained (not recommended)
        h_constrained = h_proj
        r_constrained = r_proj
    
    # === Step 3: Huber robust weighting ===
    # Compute normalized residuals: r_norm = r / σ
    measurement_std = np.sqrt(measurement_noise)
    r_normalized = r_constrained / measurement_std
    
    # Compute weights
    weights = compute_huber_weights(r_normalized.flatten(), threshold=huber_threshold)
    weight_matrix = np.diag(np.sqrt(weights))  # Use sqrt for left/right multiplication
    
    # Apply weights: H_weighted = W @ H, r_weighted = W @ r
    h_weighted = weight_matrix @ h_constrained
    r_weighted = weight_matrix @ r_constrained
    
    # Weighted measurement covariance: R_w = W @ R @ W^T
    # This ensures consistency: S = H_w @ P @ H_w^T + R_w
    meas_dim = r_weighted.shape[0]
    r_cov_original = np.eye(meas_dim) * measurement_noise
    r_cov = weight_matrix @ r_cov_original @ weight_matrix.T
    
    # === Step 4: Innovation covariance and chi-square gating ===
    # S = H * P * H^T + R
    s_mat = h_weighted @ kf.P @ h_weighted.T + r_cov
    
    # Compute innovation metrics
    innovation_norm = float(np.linalg.norm(r_weighted))
    
    try:
        s_inv = np.linalg.inv(s_mat)
        chi2_test = float(r_weighted.T @ s_inv @ r_weighted)
        
        # Chi-square gating: reject if chi2 > threshold * DoF
        # DoF = number of measurements after projection
        dof = meas_dim
        chi2_threshold = chi2_max_dof * dof
        
        if chi2_test > chi2_threshold:
            # Reject outlier measurement
            return (False, innovation_norm, chi2_test)
        
    except np.linalg.LinAlgError:
        return (False, innovation_norm, np.nan)
    
    # === Step 5: EKF update (error-state) ===
    # Kalman gain: K = P * H^T * S^-1
    try:
        k_gain = kf.P @ h_weighted.T @ s_inv
        
        # Error-state update: δx = K * r
        delta_x = k_gain @ r_weighted
        
        # Apply error-state correction to NOMINAL state
        kf._apply_error_state_correction(delta_x)
        
        # Covariance update: P = (I - K*H) * P (Joseph form for stability)
        i_kh = np.eye(err_state_size) - k_gain @ h_weighted
        kf.P = i_kh @ kf.P @ i_kh.T + k_gain @ r_cov @ k_gain.T
        
        # NEW: Ensure covariance validity after MSCKF update
        kf.P = ensure_covariance_valid(
            kf.P,
            label="MSCKF-Update",
            symmetrize=True,
            check_psd=True,
            min_eigenvalue=1e-9,
            log_condition=False
        )
        
        # Update posteriors
        kf.x_post = kf.x.copy()
        kf.P_post = kf.P.copy()
        
        return (True, innovation_norm, chi2_test)
        
    except (np.linalg.LinAlgError, ValueError) as e:
        return (False, innovation_norm if 'innovation_norm' in locals() else np.nan, 
                chi2_test if 'chi2_test' in locals() else np.nan)


def perform_msckf_updates(vio_fe: VIOFrontEnd, cam_observations: List[dict],
                          cam_states: List[dict], kf: ExtendedKalmanFilter,
                          min_observations: int = 3, max_features: int = 50,
                          msckf_dbg_path: str = None,
                          dem_reader: Optional['DEMReader'] = None,
                          origin_lat: float = 0.0, origin_lon: float = 0.0) -> int:
    """
    Perform MSCKF updates for mature features.
    
    Args:
        vio_fe: VIO frontend
        cam_observations: Observation records
        cam_states: Camera state metadata
        kf: EKF to update
        min_observations: Minimum camera observations per feature
        max_features: Maximum features to process per call
        dem_reader: DEM reader for ground plane constraint (1.4)
        origin_lat/origin_lon: Local projection origin
    
    Returns: Number of successful updates
    """
    if vio_fe is None or len(cam_states) < 2:
        print(f"[MSCKF][DEBUG] Skipping: vio_fe={vio_fe is not None}, cam_states={len(cam_states)}")
        return 0
    
    # Debug: Print state before searching for features
    print(f"[MSCKF][DEBUG] Searching features: cam_states={len(cam_states)}, cam_observations={len(cam_observations)}")
    
    # Find mature features
    mature_fids = find_mature_features_for_msckf(vio_fe, cam_observations, min_observations)
    
    print(f"[MSCKF][DEBUG] Found {len(mature_fids)} mature features (min_obs={min_observations})")
    
    print(f"[MSCKF][DEBUG] Found {len(mature_fids)} mature features (min_obs={min_observations})")
    
    if len(mature_fids) == 0:
        print(f"[MSCKF][DEBUG] No mature features - need features tracked across {min_observations}+ frames")
        return 0
    
    # Limit number of features to process
    if len(mature_fids) > max_features:
        # Prioritize by track length or quality
        # For now, just take first max_features
        mature_fids = mature_fids[:max_features]
    
    num_successful = 0
    
    for i, fid in enumerate(mature_fids):
        # Triangulate feature
        # Use ground altitude = 0 (MSL reference) for plane constraint
        # Enable debug for first 3 features to see failure reasons
        enable_debug = (i < 3)
        triangulated = triangulate_feature(fid, cam_observations, cam_states, kf, 
                                          use_plane_constraint=True, ground_altitude=0.0,
                                          debug=enable_debug,
                                          dem_reader=dem_reader,
                                          origin_lat=origin_lat, origin_lon=origin_lon)
        
        triangulation_success = triangulated is not None
        reprojection_error = triangulated.get('error', np.nan) if triangulated else np.nan
        
        if triangulated is None:
            # Log failed triangulation
            if msckf_dbg_path:
                # Count observations correctly: check inside observations list
                num_obs = sum(1 for cam_obs in cam_observations 
                             for obs in cam_obs['observations'] 
                             if obs['fid'] == fid)
                with open(msckf_dbg_path, "a", newline="") as mf:
                    mf.write(f"{vio_fe.frame_idx},{fid},{num_obs},{0},{reprojection_error:.3f},nan,0,nan\n")
            continue
        
        # Perform MSCKF update
        success, innovation_norm, chi2_test = msckf_measurement_update(fid, triangulated, cam_observations, cam_states, kf)
        
        # Log MSCKF update result
        if msckf_dbg_path:
            # Count observations correctly: check inside observations list
            num_obs = sum(1 for cam_obs in cam_observations 
                         for obs in cam_obs['observations'] 
                         if obs['fid'] == fid)
            with open(msckf_dbg_path, "a", newline="") as mf:
                mf.write(f"{vio_fe.frame_idx},{fid},{num_obs},{1},{reprojection_error:.3f},"
                        f"{innovation_norm:.3f},{int(success)},{chi2_test:.3f}\n")
        
        if success:
            num_successful += 1
    
    return num_successful


def propagate_to_timestamp(kf: ExtendedKalmanFilter, target_time: float, 
                           imu_buffer: List[IMURecord], current_time: float,
                           estimate_imu_bias: bool = False,
                           use_preintegration: bool = True) -> Tuple[bool, Optional['IMUPreintegration']]:
    """
    Propagate state to exact target timestamp using IMU measurements.
    
    NEW: Uses IMU preintegration (Forster et al.) for better accuracy and efficiency.
    
    Two modes:
      1. Preintegration (default): Integrate all IMU → apply once → less noise
      2. Legacy: Propagate sample-by-sample (fallback)
    
    This function handles time synchronization between sensors by:
    1. Finding IMU measurements between current_time and target_time
    2. [NEW] Preintegrating all IMU measurements into ΔR, Δv, Δp
    3. [NEW] Applying preintegrated delta to state (single update)
    4. [NEW] Propagating covariance using preintegration Jacobians
    
    Critical for:
    - Camera-IMU synchronization (camera @ 20Hz, IMU @ 400Hz)
    - VPS measurements with latency
    - Event-driven sensor fusion
    
    Args:
        kf: Extended Kalman Filter
        target_time: Desired timestamp to propagate to
        imu_buffer: List of IMU measurements (must be sorted by time)
        current_time: Current state time
        estimate_imu_bias: Whether IMU bias was pre-estimated
        use_preintegration: Use Forster-style preintegration (recommended)
    
    Returns:
        success: True if propagation succeeded
        preint_data: IMUPreintegration object (None if not using preintegration)
    """
    if target_time <= current_time:
        # Already at or past target time
        return True, None
    
    if len(imu_buffer) == 0:
        print(f"[WARNING] propagate_to_timestamp: No IMU data available")
        return False, None
    
    # Find IMU samples in the time range (current_time, target_time]
    relevant_imu = [imu for imu in imu_buffer if current_time < imu.t <= target_time]
    
    # DEBUG: Log IMU availability
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
        _propagate_single_imu_step(kf, last_imu, dt, estimate_imu_bias, current_time)
        return True, None
    
    # ============================================================
    # MODE 1: IMU Preintegration (Forster et al. TRO 2017)
    # ============================================================
    if use_preintegration:
        if target_time == current_time:  # First call - log mode selection
            print(f"[PROPAGATE] Using PREINTEGRATION mode (Forster et al.)")
        
        # Extract current state
        p = kf.x[0:3, 0]
        v = kf.x[3:6, 0]
        q = kf.x[6:10, 0]  # [w,x,y,z]
        bg = kf.x[10:13, 0]
        ba = kf.x[13:16, 0]
        
        # Initialize preintegration with current bias estimates
        # Use preintegration-specific noise parameters (more conservative)
        # Preintegration is sensitive to bias RW because errors accumulate
        params = IMU_PARAMS_PREINT if 'IMU_PARAMS_PREINT' in globals() else IMU_PARAMS
        preint = IMUPreintegration(
            bg=bg, ba=ba,
            sigma_g=params['gyr_n'],
            sigma_a=params['acc_n'],
            sigma_bg=params['gyr_w'],
            sigma_ba=params['acc_w']
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
        # FIX 3: Use current biases for state propagation (FEJ handled in Jacobians)
        # Preintegration was linearized at bg_lin, ba_lin stored in preint object
        # Jacobians remain at linearization point (FEJ), state uses corrected deltas
        delta_R, delta_v, delta_p = preint.get_deltas_corrected(bg, ba)
        
        # Apply preintegrated delta to nominal state
        # Rotation: q is World-to-Body (Xsens convention)
        # R_old = R_WB (World-to-Body)
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
        R_old = R_scipy.from_quat(q_xyzw).as_matrix()
        
        # R_WB_new = R_delta^T * R_WB_old
        # (Body rotates by R_delta, so World-to-Body rotates by inverse)
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
        # Biases remain constant during preintegration
        
        # Propagate error-state covariance using preintegration Jacobians
        # This is more accurate than summing individual step Jacobians
        preint_cov = preint.get_covariance()
        J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba = preint.get_jacobians()
        
        # Build state transition matrix for error-state (15+6M dimensions)
        num_clones = (kf.x.shape[0] - 16) // 7
        n_err = 15 + num_clones * 6
        
        Phi = np.eye(n_err, dtype=float)
        
        # Core state block (15x15): [δp, δv, δθ, δbg, δba]
        # FIX 2: Proper state transition using preintegration Jacobians
        # Jacobians from Forster TRO 2017 (already computed in body frame)
        
        # Position error propagation:
        # δp_k+1 = δp_k + δv_k * dt + R_BW * ∂Δp/∂θ * δθ_k + R_BW * ∂Δp/∂bg * δbg + R_BW * ∂Δp/∂ba * δba
        Phi[0:3, 3:6] = np.eye(3) * dt_total  # ∂(δp)/∂(δv)
        Phi[0:3, 6:9] = -R_BW @ skew_symmetric(delta_p)  # ∂(δp)/∂(δθ) = -R*[Δp]× (correct sign!)
        Phi[0:3, 9:12] = R_BW @ J_p_bg  # ∂(δp)/∂(δbg)
        Phi[0:3, 12:15] = R_BW @ J_p_ba  # ∂(δp)/∂(δba)
        
        # Velocity error propagation:
        # δv_k+1 = δv_k + R_BW * ∂Δv/∂θ * δθ_k + R_BW * ∂Δv/∂bg * δbg + R_BW * ∂Δv/∂ba * δba
        Phi[3:6, 6:9] = -R_BW @ skew_symmetric(delta_v)  # ∂(δv)/∂(δθ) = -R*[Δv]× (correct sign!)
        Phi[3:6, 9:12] = R_BW @ J_v_bg  # ∂(δv)/∂(δbg)
        Phi[3:6, 12:15] = R_BW @ J_v_ba  # ∂(δv)/∂(δba)
        
        # Rotation error propagation:
        # δθ_k+1 = δθ_k + ∂ΔR/∂bg * δbg (manifold tangent space)
        Phi[6:9, 9:12] = J_R_bg  # ∂(δθ)/∂(δbg) - already in tangent space
        
        # Biases are random walk (identity) - Phi[9:12,9:12] and Phi[12:15,12:15] already I
        # Clone poses propagate unchanged (identity) - already I for clones
        
        # Process noise (add preintegration covariance to core block)
        # FIX 4: Use preintegration covariance properly (already includes dt scaling now)
        Q = np.zeros((n_err, n_err), dtype=float)
        
        # Map preintegration covariance (9x9: [rotation, velocity, position])
        # to error-state covariance ([δp, δv, δθ])
        # Note: preint_cov is [rotation(0:3), velocity(3:6), position(6:9)]
        # Need to rotate from Body to World frame
        Q[0:3, 0:3] = R_BW @ preint_cov[6:9, 6:9] @ R_BW.T  # δp covariance
        Q[0:3, 3:6] = R_BW @ preint_cov[6:9, 3:6] @ R_BW.T  # δp-δv cross
        Q[0:3, 6:9] = R_BW @ preint_cov[6:9, 0:3] @ R_BW.T  # δp-δθ cross
        
        Q[3:6, 0:3] = R_BW @ preint_cov[3:6, 6:9] @ R_BW.T  # δv-δp cross
        Q[3:6, 3:6] = R_BW @ preint_cov[3:6, 3:6] @ R_BW.T  # δv covariance
        Q[3:6, 6:9] = R_BW @ preint_cov[3:6, 0:3] @ R_BW.T  # δv-δθ cross
        
        Q[6:9, 0:3] = R_BW @ preint_cov[0:3, 6:9] @ R_BW.T  # δθ-δp cross
        Q[6:9, 3:6] = R_BW @ preint_cov[0:3, 3:6] @ R_BW.T  # δθ-δv cross
        Q[6:9, 6:9] = R_BW @ preint_cov[0:3, 0:3] @ R_BW.T  # δθ covariance
        
        # Add bias random walk noise (matches legacy formulation exactly)
        # FIX 4: Use same formulation as legacy (line 676-677)
        Q[9:12, 9:12] = np.eye(3) * (IMU_PARAMS['gyr_w'] ** 2) * dt_total  # gyro bias random walk
        Q[12:15, 12:15] = np.eye(3) * (IMU_PARAMS['acc_w'] ** 2) * dt_total  # accel bias random walk
        
        # Propagate covariance: P = Φ * P * Φ^T + Q
        kf.P = Phi @ kf.P @ Phi.T + Q
        
        # NEW: Ensure covariance validity (symmetry + PSD)
        kf.P = ensure_covariance_valid(
            kf.P, 
            label="Preintegration-Propagate",
            symmetrize=True,
            check_psd=True,
            min_eigenvalue=1e-9,
            log_condition=False  # Set True for debugging
        )
        
        # Update priors
        kf.x_prior = kf.x.copy()
        kf.P_prior = kf.P.copy()
        
        return True, preint
    
    # ============================================================
    # MODE 2: Legacy Sample-by-Sample Propagation (Fallback)
    # ============================================================
    else:
        if target_time == current_time:  # First call - log mode selection
            print(f"[PROPAGATE] Using LEGACY mode (sample-by-sample)")
        
        t_current = current_time
        
        for i, imu in enumerate(relevant_imu):
            if i == 0:
                # First sample: propagate from current_time to first IMU sample
                dt = imu.t - t_current
            else:
                # Subsequent samples: propagate from previous to current
                dt = imu.t - relevant_imu[i-1].t
            
            if dt > 0:
                _propagate_single_imu_step(kf, imu, dt, estimate_imu_bias, t_current)
                t_current = imu.t
        
        # Check if we need to interpolate to exact target_time
        if t_current < target_time:
            # Last IMU sample is before target - interpolate
            last_imu = relevant_imu[-1]
            dt = target_time - t_current
            
            # Linear interpolation using last IMU measurement
            _propagate_single_imu_step(kf, last_imu, dt, estimate_imu_bias, t_current)
        
        # NEW: Ensure covariance validity after legacy propagation
        kf.P = ensure_covariance_valid(
            kf.P,
            label="Legacy-Propagate",
            symmetrize=True,
            check_psd=True,
            min_eigenvalue=1e-9
        )
        
        return True, None


def _propagate_single_imu_step(kf: ExtendedKalmanFilter, imu: IMURecord, dt: float,
                                estimate_imu_bias: bool, t: float):
    """
    Propagate state by one IMU step (helper for propagate_to_timestamp).
    
    This implements the same propagation as the main loop but as a reusable function.
    
    Args:
        kf: Extended Kalman Filter
        imu: IMU measurement
        dt: Time step
        estimate_imu_bias: Whether bias was pre-estimated
        t: Current time (for process noise computation)
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
    # CRITICAL: This IMU data is PRE-PROCESSED (gravity already compensated)
    # Stationary reading: ≈ [0, 0, 0] (NOT [0, 0, ±9.81])
    # Therefore: a_world = R * (a_measured - bias) directly gives motion acceleration
    a_world = R_body_to_world @ a_corr  # No additional gravity compensation needed!
    
    # Nominal state propagation (physics-based)
    p_new = p + v * dt + 0.5 * a_world * dt**2
    v_new = v + a_world * dt
    
    # Quaternion propagation: q_new = q ⊗ exp(ω * dt)
    theta_vec = w_corr * dt
    theta = np.linalg.norm(theta_vec)
    
    if theta < 1e-6:
        # Small angle approximation
        dq = np.array([1.0, 0.5*theta_vec[0], 0.5*theta_vec[1], 0.5*theta_vec[2]], dtype=float).reshape(4,1)
    else:
        # Exponential map
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
    
    # Biases (random walk - keep constant during propagation)
    bg_new = bg
    ba_new = ba
    
    # Update state
    kf.x[0:3, 0] = p_new
    kf.x[3:6, 0] = v_new
    kf.x[6:10, 0] = q_new
    kf.x[10:13, 0] = bg_new
    kf.x[13:16, 0] = ba_new
    
    # ESKF covariance propagation
    Phi_err = compute_error_state_jacobian(q, a_corr, w_corr, dt, R_body_to_world)
    Q_err = compute_error_state_process_noise(dt, estimate_imu_bias, t, 0.0)
    
    num_clones = (kf.x.shape[0] - 16) // 7
    kf.P = propagate_error_state_covariance(kf.P, Phi_err, Q_err, num_clones)
    
    # Update priors
    kf.x_prior = kf.x.copy()
    kf.P_prior = kf.P.copy()


def estimate_homography_scale(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray,
                               altitude: float, R_rel: np.ndarray,
                               min_inliers: int = 15) -> Optional[Tuple[float, np.ndarray, int]]:
    """
    Estimate scale from planar homography for nadir cameras over flat terrain.
    
    Theory (Planar Homography):
    For points on a plane π with normal n and distance d from camera:
    H = K @ (R + t @ n^T / d) @ K^{-1}
    
    For nadir camera over ground plane (n=[0,0,1], pointing down):
    - Rotation R: relative rotation between frames
    - Translation t: [tx, ty, tz] where tz ≈ altitude change
    - Distance d: altitude above ground (AGL)
    
    Key insight: Homography decomposition gives t/d (up to scale)
    Combined with known altitude → recover absolute scale
    
    Args:
        pts1: Feature points in frame 1 (N×2, normalized coordinates)
        pts2: Feature points in frame 2 (N×2, normalized coordinates)
        K: Camera intrinsic matrix (3×3)
        altitude: Estimated altitude above ground (meters)
        R_rel: Relative rotation between frames (3×3)
        min_inliers: Minimum inliers for valid homography
    
    Returns:
        (scale, t_scaled, num_inliers) or None if estimation fails
        - scale: Absolute scale factor (meters)
        - t_scaled: Scaled translation vector (3,)
        - num_inliers: Number of homography inliers
    """
    if len(pts1) < min_inliers or len(pts2) < min_inliers:
        return None
    
    # Convert normalized coordinates to pixel coordinates for OpenCV
    # Normalized coords are in Z=1 plane: [x_norm, y_norm, 1]
    # Pixel coords: [u, v] = K @ [x_norm, y_norm, 1]
    pts1_px = (K @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T
    pts2_px = (K @ np.hstack([pts2, np.ones((len(pts2), 1))]).T).T
    pts1_px = pts1_px[:, :2] / pts1_px[:, 2:3]  # Normalize by z
    pts2_px = pts2_px[:, :2] / pts2_px[:, 2:3]
    
    try:
        # Estimate homography using RANSAC
        H, mask = cv2.findHomography(
            pts1_px, pts2_px,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,  # pixels
            maxIters=1000,
            confidence=0.99
        )
        
        if H is None or mask is None:
            return None
        
        num_inliers = int(np.sum(mask))
        if num_inliers < min_inliers:
            return None
        
        # Decompose homography: H = K @ (R + t @ n^T / d) @ K^{-1}
        # For ground plane: n = [0, 0, 1]
        # Solve: K^{-1} @ H @ K = R + t @ n^T / d
        #                        = R + [t0/d, t1/d, t2/d] @ [0, 0, 1]
        #                        = [R[:,0], R[:,1], R[:,2] + [t0/d, t1/d, t2/d]]
        
        K_inv = np.linalg.inv(K)
        H_normalized = K_inv @ H @ K
        
        # Extract translation/depth ratio: t/d
        # For nadir: R @ n = [0, 0, 1] (approximately, camera pointing down)
        # H_normalized[:,2] ≈ R[:,2] + t/d
        R_col2 = R_rel[:, 2]  # Third column of rotation
        t_over_d = H_normalized[:, 2] - R_col2
        
        # Scale recovery: t = (t/d) * d, where d = altitude
        t_scaled = t_over_d * altitude
        
        # Compute scale magnitude
        scale = np.linalg.norm(t_scaled)
        
        # Sanity check: Scale should be reasonable (< 100 m/frame for 20Hz camera)
        if scale > 100.0 or scale < 0.01:
            return None
        
        return (scale, t_scaled, num_inliers)
        
    except (cv2.error, np.linalg.LinAlgError):
        return None


def compute_plane_constraint_jacobian(kf: ExtendedKalmanFilter, altitude: float,
                                      plane_normal: np.ndarray = np.array([0, 0, 1])) -> Tuple[np.ndarray, float]:
    """
    Compute Jacobian for plane constraint measurement.
    
    Plane constraint: For nadir camera over flat ground, altitude should match
    the distance from camera to ground plane.
    
    Measurement model: h(x) = p_z - altitude_measured
    where p_z is the Z-component of position (MSL)
    
    For error-state ESKF:
    H = ∂h/∂(δx) = [0, 0, 1, 0, ...] (only depends on position error)
    
    Args:
        kf: Extended Kalman Filter
        altitude: Measured altitude from AGL/DEM (meters)
        plane_normal: Ground plane normal in world frame (default [0,0,1])
    
    Returns:
        (H, predicted_altitude) where
        - H: Jacobian matrix (1, err_state_size)
        - predicted_altitude: Current altitude from state
    """
    # Get current position
    p_world = kf.x[0:3, 0]
    
    # Predicted altitude (assuming flat ground at z=0 for now)
    # In practice, should subtract DEM elevation
    predicted_altitude = p_world[2]
    
    # Error-state dimension
    num_clones = (kf.x.shape[0] - 16) // 7
    err_state_size = 15 + 6 * num_clones
    
    # Jacobian: ∂(p_z)/∂(δp) = [0, 0, 1]
    H = np.zeros((1, err_state_size), dtype=float)
    H[0, 2] = 1.0  # Only Z-position affects altitude
    
    return H, predicted_altitude


def augment_state_with_camera(kf: ExtendedKalmanFilter, cam_q_wxyz: np.ndarray, cam_p: np.ndarray,
                              cam_states: List[dict], cam_observations: List[dict],
                              p_quat: float = 1e-3, p_pos: float = 1.0, max_poses: int = 5) -> int:
    """
    Augment the EKF state vector with a camera pose block: [q(4,wxyz), p(3)].
    Maintains a sliding window of camera poses by marginalizing old poses when exceeding max_poses.
    
    CRITICAL: This function works with ESKF (Error-State Kalman Filter):
      - Nominal state x: [p,v,q,bg,ba, q_C1,p_C1, q_C2,p_C2, ...] (16+7M dimensions)
      - Error-state covariance P: [δp,δv,δθ,δbg,δba, δθ_C1,δp_C1, δθ_C2,δp_C2, ...] (15+6M dimensions)
    
    Note: Quaternion q is 4D in nominal state but rotation error δθ is 3D in error-state!

    Args:
        kf: Extended Kalman Filter
        cam_q_wxyz: Camera quaternion [w,x,y,z]
        cam_p: Camera position [x,y,z]
        cam_states: List of camera state metadata
        cam_observations: List of observation records with cam_id references
        p_quat: Prior uncertainty for quaternion (will be converted to 3D rotation)
        p_pos: Prior uncertainty for position
        max_poses: Maximum number of camera poses in sliding window

    Returns the start index of the appended block (int).
    """
    # ESKF dimensions
    pose_size_nominal = 7  # quaternion (4) + position (3) in nominal state
    pose_size_error = 6    # rotation (3) + position (3) in error-state
    core_size_nominal = 16 # base nominal state size
    core_size_error = 15   # base error-state size
    
    # Calculate number of poses currently in state vector
    old_n_nominal = kf.dim_x
    num_poses = (old_n_nominal - core_size_nominal) // pose_size_nominal
    
    # Error-state covariance dimension
    old_n_error = core_size_error + num_poses * pose_size_error
    # Error-state covariance dimension
    old_n_error = core_size_error + num_poses * pose_size_error
    
    # Validate covariance shape matches expected error-state dimension
    if kf.P.shape != (old_n_error, old_n_error):
        print(f"[WARNING] Covariance shape mismatch: P.shape={kf.P.shape}, expected=({old_n_error},{old_n_error})")
    
    # If we're at max poses, marginalize oldest pose first
    if num_poses >= max_poses:
        # Remove oldest pose (first pose after core state)
        old_pose_idx_nominal = core_size_nominal
        old_pose_idx_error = core_size_error
        
        # Create masks to keep everything except oldest pose
        mask_nominal = np.ones(old_n_nominal, dtype=bool)
        mask_nominal[old_pose_idx_nominal:old_pose_idx_nominal + pose_size_nominal] = False
        
        mask_error = np.ones(old_n_error, dtype=bool)
        mask_error[old_pose_idx_error:old_pose_idx_error + pose_size_error] = False
        
        # Apply mask to nominal state and error-state covariance
        kf.x = kf.x[mask_nominal]
        kf.P = kf.P[np.ix_(mask_error, mask_error)]
        
        old_n_nominal = kf.x.shape[0]
        old_n_error = kf.P.shape[0]
        kf.dim_x = old_n_nominal
        
        # Marginalize oldest camera state (index 0)
        old_cam_id = 0
        old_obs_count = len([obs for obs in cam_observations if obs['cam_id'] == old_cam_id])
        
        if len(cam_states) > 0:
            cam_states.pop(0)
        
        # Update all remaining cam_states indices (shift down)
        for cs in cam_states:
            cs['start_idx'] -= pose_size_nominal
            cs['q_idx'] -= pose_size_nominal
            cs['p_idx'] -= pose_size_nominal
            # FIX: Update error indices too!
            cs['err_q_idx'] -= pose_size_error
            cs['err_p_idx'] -= pose_size_error
        
        # Remove observations for the marginalized camera (cam_id=0)
        # and decrement cam_id for all remaining observations
        new_observations = []
        for obs in cam_observations:
            if obs['cam_id'] > 0:  # Drop observations for marginalized camera
                new_obs = obs.copy()
                new_obs['cam_id'] = obs['cam_id'] - 1
                new_observations.append(new_obs)
        cam_observations[:] = new_observations
        
        print(f"[VIO] Marginalized oldest pose (cam_id={old_cam_id}), removed {old_obs_count} observation sets, "
              f"now tracking {len(cam_states)} poses with {len(cam_observations)} observation sets")

    # Now augment with new pose
    # Nominal state: add 7 dimensions (q:4, p:3)
    # Error-state covariance: add 6 dimensions (δθ:3, δp:3)
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
    # Prior for the new camera block (rotation: 3D, position: 3D)
    new_P[old_n_error:old_n_error+3, old_n_error:old_n_error+3] = np.eye(3) * p_quat  # δθ (3D rotation error)
    new_P[old_n_error+3:old_n_error+6, old_n_error+3:old_n_error+6] = np.eye(3) * p_pos  # δp

    # Assign back to filter and resize helper matrices
    kf.x = new_x
    kf.P = new_P
    kf.dim_x = new_n_nominal
    kf.F = np.eye(new_n_error, dtype=float)  # F operates on error-state
    kf.Q = np.eye(new_n_error, dtype=float)  # Q operates on error-state
    kf._I = np.eye(new_n_error, dtype=float) # I operates on error-state

    # Update priors/posteriors to match sizes
    kf.x_prior = kf.x.copy()
    kf.P_prior = kf.P.copy()
    kf.x_post = kf.x.copy()
    kf.P_post = kf.P.copy()

    # Validation: Ensure consistency between state vector and metadata
    current_num_poses = (kf.dim_x - core_size_nominal) // pose_size_nominal
    if len(cam_states) != current_num_poses:
        # This warning is benign because the caller appends the new state immediately after this function returns
        # print(f"[WARNING] augment_state_with_camera: cam_states length {len(cam_states)} != num_poses in state {current_num_poses}")
        pass
    
    # Validate cam_id references
    for obs in cam_observations:
        if obs['cam_id'] < 0 or obs['cam_id'] >= len(cam_states):
            print(f"[WARNING] augment_state_with_camera: Invalid cam_id {obs['cam_id']} (valid range: [0, {len(cam_states)-1}])")

    return old_n_nominal

# ===============================
# Debug helper functions
# ===============================

def log_measurement_update(residual_csv, t, frame, update_type, innovation, 
                           mahalanobis_dist, chi2_threshold, accepted, 
                           S_matrix=None, P_prior=None):
    """
    Log measurement update residuals and statistics for debugging.
    
    Args:
        residual_csv: Path to residual debug CSV file
        t: Current timestamp
        frame: Current VIO frame index (-1 if not applicable)
        update_type: Type of update ('VPS', 'VIO_VEL', 'MAG', 'DEM', 'MSCKF', 'ZUPT')
        innovation: Innovation vector (residual)
        mahalanobis_dist: Mahalanobis distance (for gating)
        chi2_threshold: Chi-square threshold for acceptance
        accepted: Whether update was accepted (bool)
        S_matrix: Innovation covariance (optional, for NIS computation)
        P_prior: Prior covariance (optional, for NEES computation)
    """
    if residual_csv is None:
        return
    
    try:
        # Compute NIS (Normalized Innovation Squared)
        # NIS = y^T * S^{-1} * y (should follow chi-square distribution)
        NIS = float('nan')
        if S_matrix is not None:
            try:
                S_inv = np.linalg.inv(S_matrix)
                NIS = float(innovation.T @ S_inv @ innovation)
            except:
                pass
        
        # Compute NEES (Normalized Estimation Error Squared)
        # NEES = e^T * P^{-1} * e (requires ground truth, approximated here)
        NEES = float('nan')  # Would need ground truth for proper NEES
        
        # Format innovation vector (pad with NaN for consistent CSV width)
        innov_x = float(innovation[0]) if len(innovation) > 0 else float('nan')
        innov_y = float(innovation[1]) if len(innovation) > 1 else float('nan')
        innov_z = float(innovation[2]) if len(innovation) > 2 else float('nan')
        
        with open(residual_csv, "a", newline="") as f:
            f.write(f"{t:.6f},{frame},{update_type},{innov_x:.6f},{innov_y:.6f},{innov_z:.6f},"
                   f"{mahalanobis_dist:.6f},{chi2_threshold:.6f},{int(accepted)},{NIS:.6f},{NEES:.6f}\n")
    except Exception:
        pass  # Silent fail for debug logging


def log_fej_consistency(fej_csv, t, frame, cam_states, kf):
    """
    Log FEJ consistency metrics: compare FEJ linearization points vs. current state.
    
    NEW: Tracks drift between first-estimate and current estimate to detect
    spurious information from unobservable directions.
    
    Writes to debug_fej_consistency.csv with format:
    timestamp, frame, clone_idx, 
    pos_fej_drift_m, rot_fej_drift_deg,
    bg_fej_drift_rad_s, ba_fej_drift_m_s2
    
    Args:
        fej_csv: CSV file path for FEJ consistency logging
        t: Current timestamp
        frame: Current VIO frame index
        cam_states: List of camera clone states with FEJ data
        kf: Extended Kalman Filter with current state
    """
    if fej_csv is None or not cam_states:
        return
    
    # Current bias estimates from core state
    bg_current = kf.x[10:13, 0]
    ba_current = kf.x[13:16, 0]
    
    for i, cs in enumerate(cam_states):
        # Skip if no FEJ data
        if 'q_fej' not in cs or 'p_fej' not in cs:
            continue
        
        q_fej = cs['q_fej']
        p_fej = cs['p_fej']
        
        # Get current state estimates (nominal state)
        q_idx = cs['q_idx']
        p_idx = cs['p_idx']
        q_current = kf.x[q_idx:q_idx+4, 0]
        p_current = kf.x[p_idx:p_idx+3, 0]
        
        # Compute position drift (Euclidean distance)
        pos_drift = np.linalg.norm(p_current - p_fej)
        
        # Compute rotation drift (angle between quaternions)
        # q1^{-1} * q2 = relative rotation
        q_fej_inv = np.array([q_fej[0], -q_fej[1], -q_fej[2], -q_fej[3]])
        q_rel = _quat_mul(q_fej_inv.reshape(4,1), q_current.reshape(4,1)).reshape(4,)
        
        # Angle from quaternion: θ = 2 * arccos(w)
        w_rel = np.clip(q_rel[0], -1.0, 1.0)
        rot_drift_rad = 2.0 * np.arccos(abs(w_rel))
        rot_drift_deg = np.rad2deg(rot_drift_rad)
        
        # Compute bias drift (if FEJ bias stored)
        if 'bg_fej' in cs and 'ba_fej' in cs:
            bg_fej = cs['bg_fej']
            ba_fej = cs['ba_fej']
            bg_drift = np.linalg.norm(bg_current - bg_fej)
            ba_drift = np.linalg.norm(ba_current - ba_fej)
        else:
            bg_drift = np.nan
            ba_drift = np.nan
        
        try:
            with open(fej_csv, "a", newline="") as f:
                f.write(f"{t:.6f},{frame},{i},{pos_drift:.6f},{rot_drift_deg:.6f},"
                       f"{bg_drift:.6e},{ba_drift:.6e}\n")
        except Exception:
            pass  # Silent fail for debug logging


def save_keyframe_image_with_overlay(image, features, inliers, reprojections, 
                                     output_path, frame_id, tracking_stats=None):
    """
    Save keyframe image with feature tracking and reprojection overlays.
    
    Args:
        image: Input image (grayscale or color)
        features: List of tracked feature points (Nx2 array)
        inliers: Boolean mask of inlier features
        reprojections: Reprojected feature positions (Nx2 array, or None)
        output_path: Path to save annotated image
        frame_id: Frame index for labeling
        tracking_stats: Dict with tracking statistics (optional)
    """
    try:
        import cv2
        
        # Convert grayscale to BGR for colored overlays
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Draw all features (green circles)
        if features is not None and len(features) > 0:
            for pt in features:
                cv2.circle(vis, tuple(pt.astype(int)), 3, (0, 255, 0), 1)
        
        # Draw inliers (blue circles, thicker)
        if features is not None and inliers is not None and len(features) > 0:
            inlier_pts = features[inliers]
            for pt in inlier_pts:
                cv2.circle(vis, tuple(pt.astype(int)), 4, (255, 0, 0), 2)
        
        # Draw reprojection errors (red lines from feature to reprojection)
        if reprojections is not None and features is not None and len(features) > 0:
            for feat, reproj in zip(features, reprojections):
                if not np.any(np.isnan(reproj)):
                    pt1 = tuple(feat.astype(int))
                    pt2 = tuple(reproj.astype(int))
                    cv2.line(vis, pt1, pt2, (0, 0, 255), 1)
                    cv2.circle(vis, pt2, 2, (0, 0, 255), -1)
        
        # Add text overlay with statistics
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        cv2.putText(vis, f"Frame: {frame_id}", (10, y_offset), font, 0.7, (255, 255, 255), 2)
        
        if tracking_stats:
            y_offset += 30
            cv2.putText(vis, f"Features: {tracking_stats.get('num_features', 0)}", 
                       (10, y_offset), font, 0.6, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(vis, f"Inliers: {tracking_stats.get('num_inliers', 0)}", 
                       (10, y_offset), font, 0.6, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(vis, f"Parallax: {tracking_stats.get('parallax_px', 0):.1f}px", 
                       (10, y_offset), font, 0.6, (255, 255, 255), 1)
        
        # Legend
        y_offset = vis.shape[0] - 60
        cv2.circle(vis, (20, y_offset), 3, (0, 255, 0), 1)
        cv2.putText(vis, "Tracked", (30, y_offset + 5), font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.circle(vis, (20, y_offset), 4, (255, 0, 0), 2)
        cv2.putText(vis, "Inlier", (30, y_offset + 5), font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.line(vis, (15, y_offset), (25, y_offset), (0, 0, 255), 1)
        cv2.putText(vis, "Reproj Error", (30, y_offset + 5), font, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(output_path, vis)
    except Exception as e:
        print(f"[WARNING] Failed to save keyframe image: {e}")


# ===============================
# Main runner
# ===============================

def run(
    imu_path: str,
    quarry_path: str,
    output_dir: str,
    images_dir: Optional[str] = None,
    images_index_csv: Optional[str] = None,
    vps_csv: Optional[str] = None,
    mag_csv: Optional[str] = None,  # Magnetometer data (vector3.csv)
    dem_path: Optional[str] = None,
    downscale_size: Tuple[int,int] = (1140,1080),
    z_state: str = "msl",
    camera_view: str = "nadir",  # Camera view mode: 'nadir', 'front', or 'side'
    estimate_imu_bias: bool = False,  # Estimate IMU bias from static period (now fixed: uses world-frame calculation)
    use_magnetometer: bool = True,  # Enable magnetometer-aided heading correction
    use_vio_velocity: bool = True,  # Enable VIO velocity updates (disable if scale recovery is problematic)
    save_debug_data: bool = False,  # Save comprehensive debugging data (IMU, keyframes, residuals, etc.)
    save_keyframe_images: bool = False,  # Save keyframe images with feature/inlier overlays
    use_preintegration: bool = True,  # NEW: Use IMU preintegration (Forster et al.) instead of legacy propagation
    ground_truth_path: Optional[str] = None,  # PPK ground truth file for initial state (lat/lon/attitude/velocity)
):
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Load inputs ----------
    # NEW: Use PPK for all initial values EXCEPT MSL altitude
    ppk_state = load_ppk_initial_state(ground_truth_path) if ground_truth_path else None
    
    # Build initial rotation matrix from PPK attitude for lever arm compensation
    R_BW_init = None  # Body-to-World rotation (FRD body, ENU world)
    if ppk_state is not None:
        # Build rotation matrix from PPK Euler angles
        roll_ned = ppk_state.roll
        pitch_ned = ppk_state.pitch
        yaw_ned = ppk_state.yaw
        roll_enu = roll_ned
        pitch_enu = -pitch_ned
        yaw_enu = np.pi/2 - yaw_ned
        R_BW_flu = R_scipy.from_euler('ZYX', [yaw_enu, pitch_enu, roll_enu]).as_matrix()
        R_FLU_to_FRD = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R_BW_init = R_BW_flu @ R_FLU_to_FRD.T  # FRD body to ENU world
    
    if ppk_state is not None:
        # PPK position is at GNSS antenna - need to compensate for lever arm to get IMU position
        # Step 1: Get GNSS antenna position in local ENU frame
        lat0_gnss = ppk_state.lat
        lon0_gnss = ppk_state.lon
        
        # For origin, we use GNSS position directly (lever arm offset is small relative to lat/lon scale)
        lat0 = lat0_gnss
        lon0 = lon0_gnss
        
        # Step 2: Compute lever arm offset in ENU world frame
        # lever_arm_world = R_BW @ lever_arm_body
        lever_arm_world = R_BW_init @ IMU_GNSS_LEVER_ARM
        
        # Print lever arm info for debugging
        print(f"[INIT][LEVER ARM] IMU-GNSS lever arm in body frame: {IMU_GNSS_LEVER_ARM}")
        print(f"[INIT][LEVER ARM] IMU-GNSS lever arm in world frame (ENU): {lever_arm_world}")
        print(f"[INIT][LEVER ARM] Lever arm magnitude: {np.linalg.norm(IMU_GNSS_LEVER_ARM):.3f} m")
        
        # Note: We store the lever_arm_world for later use in position initialization
        # The actual XY offset will be applied after ensure_local_proj is called
        _init_lever_arm_world = lever_arm_world
        
        # Velocity from PPK is in ENU (Ve, Vn, Vu)
        v_init_enu = np.array([ppk_state.ve, ppk_state.vn, ppk_state.vu], dtype=float)
        print(f"[INIT] Using PPK for lat/lon/velocity (with lever arm compensation)")
    else:
        # Fallback to old GGA file
        lat0, lon0, _, v_init_enu = load_quarry_initial(quarry_path)
        _init_lever_arm_world = np.zeros(3)
        print(f"[INIT] Using GGA file for lat/lon/velocity (no PPK provided)")
    
    # ALWAYS use GGA for MSL (PPK has ellipsoidal height, not MSL)
    msl0_m = load_msl_from_gga(quarry_path)
    if msl0_m is None:
        # Fallback to load_quarry_initial if load_msl_from_gga fails
        _, _, msl0_m, _ = load_quarry_initial(quarry_path)
    
    # Load full flight_log for MSL updates (when no DEM available)
    # This allows us to use time-varying MSL measurements instead of just initial value
    flight_log_df = None
    if os.path.exists(quarry_path):
        flight_log_df = pd.read_csv(quarry_path)
        if 'stamp_log' in flight_log_df.columns and 'altitude_MSL_m' in flight_log_df.columns:
            print(f"[INFO] Loaded flight_log with {len(flight_log_df)} MSL measurements for height updates")
        else:
            print("[WARNING] flight_log missing stamp_log or altitude_MSL_m columns")
            flight_log_df = None
    
    imu = load_imu_csv(imu_path)
    imgs = load_images(images_dir, images_index_csv)
    vps_list = load_vps_csv(vps_csv)
    mag_list = load_mag_csv(mag_csv) if use_magnetometer else []
    dem = DEMReader.open(dem_path)
    
    # Load PPK trajectory for error comparison (more accurate than GGA)
    ppk_trajectory_df = load_ppk_trajectory(ground_truth_path) if ground_truth_path else None

    # ---------- Startup log ----------
    print("=== Input check ===")
    print(f"IMU: {'OK' if len(imu)>0 else 'MISSING'} ({len(imu)} samples)")

    # images loader prints its own diagnostics
    print(f"Images: {'OK' if len(imgs)>0 else 'None'} ({len(imgs)} frames)")
    print(f"VPS: {'OK' if len(vps_list)>0 else 'None'} ({len(vps_list)} items)")
    print(f"Mag: {'OK' if len(mag_list)>0 else 'None'} ({len(mag_list)} samples)")
    print(f"DEM: {'OK' if dem.ds is not None else 'None'} → {dem_path if dem.ds else ''}")
    print(f"PPK: {'OK' if ppk_state else 'None'} → {ground_truth_path if ppk_state else ''}")
    print(f"GGA (MSL only): OK → {quarry_path}")

    if len(imu) == 0:
        raise RuntimeError("IMU is required. Aborting.")

    # ---------- Origin / DEM / initial state ----------
    ensure_local_proj(lat0, lon0)
    dem0 = dem.sample_m(lat0, lon0) if dem.ds else None
    agl0 = (msl0_m - dem0) if dem0 is not None else msl0_m

    # EKF / MSCKF-style IMU state layout (reserved space)
    # We expand the state vector to hold a full IMU state (no camera poses yet).
    # State x layout (16 elements):
    #  idx 0..2   : p_I (position) [m]  -> world frame (MSL or AGL per `z_state`)
    #  idx 3..5   : v_I (velocity) [m/s]
    #  idx 6..9   : q_I (quaternion w,x,y,z) (unit quaternion)
    #  idx 10..12 : b_g (gyro bias) [rad/s]
    #  idx 13..15 : b_a (accel bias) [m/s^2]
    # Note: downstream code will be updated to use this layout for IMU propagation
    # and (later) camera-poses appended to the state for MSCKF sliding window.
    kf = ExtendedKalmanFilter(dim_x=16, dim_z=3, dim_u=3)
    kf.x = np.zeros((16, 1), dtype=float)
    
    # Initialize timestamp for adaptive VPS chi-square threshold
    # Start at first IMU timestamp so time_since_correction starts at 0
    kf.last_absolute_correction_time = imu[0].t
    
    # ======== INITIAL POSITION WITH LEVER ARM COMPENSATION ========
    # Origin (lat0, lon0) is GNSS antenna position, but we track IMU position
    # IMU position in local ENU frame = -lever_arm_world (since GNSS is at origin)
    # This compensates for the ~55cm offset between GNSS antenna and IMU
    if ppk_state is not None and np.linalg.norm(_init_lever_arm_world) > 0.01:
        # Apply lever arm offset: p_imu = p_gnss - lever_arm_world
        # Since GNSS is at origin (0, 0), IMU starts at (-lever_arm_x, -lever_arm_y)
        kf.x[0, 0] = -_init_lever_arm_world[0]  # X (East)
        kf.x[1, 0] = -_init_lever_arm_world[1]  # Y (North)
        print(f"[INIT] Initial XY position (IMU, lever-arm corrected): [{kf.x[0,0]:.3f}, {kf.x[1,0]:.3f}] m")
    else:
        kf.x[0, 0] = 0.0
        kf.x[1, 0] = 0.0
    
    # Initialize position z according to z_state (MSL or AGL)
    # Also apply lever arm Z offset (GNSS antenna is 53cm above IMU in body frame)
    # In world frame with FRD body: body Z-down means lever_arm_world[2] varies with attitude
    z_lever_offset = _init_lever_arm_world[2] if ppk_state is not None else 0.0
    if z_state.lower() == "agl" and dem0 is not None:
        kf.x[2, 0] = agl0 - z_lever_offset
        z_mode = "AGL"
        print(f"[INIT] Initial Z (AGL, lever-arm corrected): {agl0:.2f} - {z_lever_offset:.2f} = {kf.x[2,0]:.2f} m")
    else:
        kf.x[2, 0] = msl0_m - z_lever_offset
        z_mode = "MSL"
        print(f"[INIT] Initial Z (MSL, lever-arm corrected): {msl0_m:.2f} - {z_lever_offset:.2f} = {kf.x[2,0]:.2f} m")
    # Initialize velocity from PPK or GGA (ENU frame)
    kf.x[3:6, 0] = v_init_enu
    
    # ======== QUATERNION INITIALIZATION ========
    # Priority: PPK (roll/pitch/yaw) > IMU quaternion
    # PPK gives us accurate attitude from INS/GPS fusion
    if ppk_state is not None:
        # Use PPK roll/pitch/yaw directly
        # PPK uses NED frame: roll/pitch/yaw
        # Convert to ENU: roll_enu = roll_ned, pitch_enu = -pitch_ned, yaw_enu = π/2 - yaw_ned
        roll_ned = ppk_state.roll
        pitch_ned = ppk_state.pitch  
        yaw_ned = ppk_state.yaw
        
        # NED to ENU conversion for Euler angles:
        # - Roll: same (rotation about forward axis)
        # - Pitch: negated (NED pitch up is positive, ENU pitch up is negative)
        # - Yaw: ENU yaw = 90° - NED yaw (NED: 0=North CW, ENU: 0=East CCW)
        roll_enu = roll_ned
        pitch_enu = -pitch_ned
        yaw_enu = np.pi/2 - yaw_ned
        
        # Build quaternion from ENU Euler angles (ZYX convention: yaw-pitch-roll)
        # This creates a rotation for FLU body frame (Body Z-Up)
        R_BW_flu = R_scipy.from_euler('ZYX', [yaw_enu, pitch_enu, roll_enu])
        
        # CRITICAL: PPK Euler angles assume FLU body frame (Body Z pointing UP)
        # But our extrinsics and IMU data use FRD body frame (Body Z pointing DOWN)
        # We need to convert from FLU to FRD by applying 180° rotation around X-axis
        # R_FLU_to_FRD = diag(1, -1, -1) rotates FLU body to FRD body
        R_FLU_to_FRD = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R_BW_frd = R_BW_flu.as_matrix() @ R_FLU_to_FRD.T  # Apply body frame conversion
        
        # Convert back to quaternion
        q_xyzw = R_scipy.from_matrix(R_BW_frd).as_quat()
        q_init = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # [w,x,y,z]
        
        # Verify body Z direction
        body_z_world = R_BW_frd[:, 2]
        print(f"[PPK→FRD] Body Z in world: {body_z_world}, Z-component: {body_z_world[2]:.4f} (should be negative)")
        
        kf.x[6, 0] = q_init[0]  # w
        kf.x[7, 0] = q_init[1]  # x
        kf.x[8, 0] = q_init[2]  # y
        kf.x[9, 0] = q_init[3]  # z
        
        # Compare with IMU for diagnostics
        q_imu = imu[0].q  # [x,y,z,w]
        euler_imu = R_scipy.from_quat(q_imu).as_euler('ZYX')  # [yaw, pitch, roll] ENU
        
        print(f"[INIT][PPK] Using PPK attitude:")
        print(f"  NED: roll={np.degrees(roll_ned):.2f}°, pitch={np.degrees(pitch_ned):.2f}°, yaw={np.degrees(yaw_ned):.2f}°")
        print(f"  ENU: roll={np.degrees(roll_enu):.2f}°, pitch={np.degrees(pitch_enu):.2f}°, yaw={np.degrees(yaw_enu):.2f}°")
        print(f"  IMU (ENU): yaw={np.degrees(euler_imu[0]):.2f}°, pitch={np.degrees(euler_imu[1]):.2f}°, roll={np.degrees(euler_imu[2]):.2f}°")
        print(f"  Delta yaw: {np.degrees(yaw_enu - euler_imu[0]):.2f}°")
    else:
        # Fallback: Use IMU quaternion and optionally correct yaw from ground_truth
        # CRITICAL: Initialize quaternion from FIRST IMU sample, not identity!
        # The aircraft already has orientation (not aligned with world frame)
        # IMU CSV stores quaternion as [x,y,z,w], state stores as [w,x,y,z]
        # 
        # IMPORTANT: Xsens MTi-30 uses ENU world frame AND Z-Up body frame!
        # - Accelerometer lin_z ≈ -9.8 when stationary → body Z points UP
        # - Quaternion represents Body-to-ENU rotation
        # - We use ENU as our world frame (NO conversion needed!)
        q_first = imu[0].q  # [x,y,z,w] from CSV (Body-to-ENU)
        kf.x[6, 0] = q_first[3]  # w
        kf.x[7, 0] = q_first[0]  # x
        kf.x[8, 0] = q_first[1]  # y
        kf.x[9, 0] = q_first[2]  # z
        print(f"[INIT][IMU] Using IMU quaternion (no PPK available)")
        
        # Legacy: Initialize yaw from ground truth if available (yaw-only correction)
        gt_yaw_rad = load_ground_truth_initial_yaw(ground_truth_path)
        if gt_yaw_rad is not None:
            # Ground truth yaw is in NED convention: 0=North, positive=clockwise
            # Convert to ENU: yaw_enu = 90° - yaw_ned = pi/2 - yaw_ned
            gt_yaw_enu = np.pi/2 - gt_yaw_rad
            
            # Get current quaternion
            q_current = np.array([kf.x[6,0], kf.x[7,0], kf.x[8,0], kf.x[9,0]])  # [w,x,y,z]
            
            # Extract current yaw from IMU quaternion (ENU frame)
            q_xyzw = np.array([q_current[1], q_current[2], q_current[3], q_current[0]])  # scipy format
            euler_imu = R_scipy.from_quat(q_xyzw).as_euler('ZYX')  # [yaw, pitch, roll]
            yaw_imu = euler_imu[0]
            
            print(f"[INIT][GT] IMU initial yaw: {np.degrees(yaw_imu):.1f}° (ENU)")
            print(f"[INIT][GT] Ground truth yaw: {np.degrees(gt_yaw_rad):.1f}° (NED) = {np.degrees(gt_yaw_enu):.1f}° (ENU)")
            
            # Replace yaw while keeping roll and pitch from IMU
            euler_corrected = euler_imu.copy()
            euler_corrected[0] = gt_yaw_enu  # Replace yaw
            
            # Reconstruct quaternion
            q_new_xyzw = R_scipy.from_euler('ZYX', euler_corrected).as_quat()  # [x,y,z,w]
            q_new = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]])  # [w,x,y,z]
            
            # Update state
            kf.x[6, 0] = q_new[0]  # w
            kf.x[7, 0] = q_new[1]  # x
            kf.x[8, 0] = q_new[2]  # y
            kf.x[9, 0] = q_new[3]  # z
            
            yaw_new_check = R_scipy.from_quat([q_new[1], q_new[2], q_new[3], q_new[0]]).as_euler('ZYX')[0]
            print(f"[INIT][GT] Corrected yaw: {np.degrees(yaw_new_check):.1f}° (ENU), delta={np.degrees(gt_yaw_enu - yaw_imu):.1f}°")
    
    # RE-ENABLED: Magnetometer calibration now GPS-calibrated (2025-02-07)
    # Analysis using PPK GPS ground truth heading:
    #   - Found heading formula: atan2(-mag_y, mag_x)
    #   - Combined declination: -0.643 rad (-36.8°)
    #   - Residual std: ~16° (acceptable for initial alignment)
    if use_magnetometer and len(mag_list) > 0 and MAG_APPLY_INITIAL_CORRECTION:
        # Find first valid magnetometer reading
        mag_init = None
        for mag_rec in mag_list[:50]:  # Check first 50 samples
            mag_cal = calibrate_magnetometer(mag_rec.mag)
            mag_norm = np.linalg.norm(mag_cal)
            if MAG_MIN_FIELD_STRENGTH <= mag_norm <= MAG_MAX_FIELD_STRENGTH:
                mag_init = mag_cal
                break
        
        if mag_init is not None:
            # Get current quaternion from IMU
            q_imu = np.array([kf.x[6,0], kf.x[7,0], kf.x[8,0], kf.x[9,0]])  # [w,x,y,z]
            
            # Compute yaw from magnetometer (GPS-calibrated raw heading)
            yaw_mag, quality = compute_yaw_from_mag(
                mag_init, q_imu, 
                mag_declination=MAG_DECLINATION,
                use_raw_heading=MAG_USE_RAW_HEADING
            )
            
            # Get yaw from IMU quaternion
            yaw_imu = quaternion_to_yaw(q_imu)
            
            # If quality is good, correct the yaw
            if quality > 0.5:
                # Compute yaw difference
                yaw_diff = yaw_mag - yaw_imu
                yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))  # Wrap to [-π, π]
                
                # Only correct if difference is significant (>10°)
                if abs(yaw_diff) > np.radians(10):
                    print(f"[INIT][MAG] Correcting initial yaw: IMU={np.degrees(yaw_imu):.1f}° → MAG={np.degrees(yaw_mag):.1f}° (Δ={np.degrees(yaw_diff):.1f}°)")
                    
                    # SIMPLER APPROACH: Use scipy to extract Euler, modify yaw, reconstruct
                    q_xyzw = np.array([q_imu[1], q_imu[2], q_imu[3], q_imu[0]])
                    R_mat = R_scipy.from_quat(q_xyzw).as_matrix()
                    
                    # Extract roll, pitch (keep these), replace yaw with magnetometer
                    # Using ZYX (yaw-pitch-roll) convention
                    euler = R_scipy.from_matrix(R_mat).as_euler('ZYX')  # [yaw, pitch, roll]
                    old_yaw = euler[0]
                    euler[0] = yaw_mag  # Replace yaw with magnetometer yaw
                    
                    # Reconstruct quaternion
                    R_new = R_scipy.from_euler('ZYX', euler).as_matrix()
                    q_new_xyzw = R_scipy.from_matrix(R_new).as_quat()  # [x,y,z,w]
                    q_new = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]])  # [w,x,y,z]
                    
                    # Update state
                    kf.x[6, 0] = q_new[0]  # w
                    kf.x[7, 0] = q_new[1]  # x
                    kf.x[8, 0] = q_new[2]  # y
                    kf.x[9, 0] = q_new[3]  # z
                    
                    # Verify correction
                    yaw_new = quaternion_to_yaw(q_new)
                    print(f"[INIT][MAG] New yaw after correction: {np.degrees(yaw_new):.1f}° (target: {np.degrees(yaw_mag):.1f}°)")
                else:
                    print(f"[INIT][MAG] Yaw difference small ({np.degrees(yaw_diff):.1f}°), no correction needed")
            else:
                print(f"[INIT][MAG] Low quality ({quality:.2f}), using IMU yaw={np.degrees(yaw_imu):.1f}°")
    
    # Initialize IMU biases
    # Check if we have config-provided initial biases
    initial_gyro_bias = globals().get('INITIAL_GYRO_BIAS', np.zeros(3))
    initial_accel_bias = globals().get('INITIAL_ACCEL_BIAS', np.zeros(3))
    config_estimate_bias = globals().get('ESTIMATE_IMU_BIAS', False)
    
    if estimate_imu_bias or config_estimate_bias:
        if np.any(initial_gyro_bias != 0) or np.any(initial_accel_bias != 0):
            # Use config-provided initial biases
            bg_init = initial_gyro_bias
            ba_init = initial_accel_bias
            
            kf.x[10:13, 0] = bg_init  # Gyro bias
            kf.x[13:16, 0] = ba_init  # Accel bias
            
            print(f"[DEBUG][BIAS] Using config-provided initial biases:")
            print(f"[DEBUG][BIAS] bg_init: {bg_init} rad/s ({np.degrees(bg_init)}°/s)")
            print(f"[DEBUG][BIAS] ba_init: {ba_init} m/s²")
            
            bias_msg = f"from config: bg={bg_init}, ba={ba_init}"
        elif len(imu) >= 100:
            # Estimate biases from initial static period  
            # CRITICAL FIX: Estimate bias in BODY frame directly!
            # The bias correction happens in body frame BEFORE rotation to world
            n_static = min(500, len(imu))  # Use more samples for better estimate
            
            # Collect raw IMU readings in body frame
            acc_body_list = []
            gyro_list = []
            
            for rec in imu[:n_static]:
                acc_body_list.append(rec.lin.astype(float))
                gyro_list.append(rec.ang.astype(float))
            
            acc_body_mean = np.mean(acc_body_list, axis=0)
            gyro_mean = np.mean(gyro_list, axis=0)
            
            # For stationary sensor: after bias correction and rotation to world,
            # we should get a_world ≈ [0, 0, -g] in ENU (gravity points DOWN = negative Z)
            # So: R @ (a_measured - ba) = [0, 0, -g]
            # Therefore: ba = a_measured - R^T @ [0, 0, -g]
            
            # Use ENU quaternion directly (NO conversion needed!)
            quat_0 = np.array([imu[0].q[0], imu[0].q[1], imu[0].q[2], imu[0].q[3]])
            R_0 = R_scipy.from_quat(quat_0).as_matrix()
            # In ENU frame: gravity points DOWN, so g_world = [0, 0, -g]
            expected_world = np.array([0, 0, -IMU_PARAMS["g_norm"]])
            expected_body = R_0.T @ expected_world  # Transform expected to body frame
            
            ba_init = acc_body_mean - expected_body
            bg_init = gyro_mean  # Gyro bias: should read zero when stationary
            
            kf.x[10:13, 0] = bg_init  # Gyro bias
            kf.x[13:16, 0] = ba_init  # Accel bias
            
            print(f"[DEBUG][BIAS] Using {n_static} samples for bias estimation")
            print(f"[DEBUG][BIAS] Raw acc body mean: {acc_body_mean}")
            print(f"[DEBUG][BIAS] Expected body (from R^T @ [0,0,g]): {expected_body}")
            print(f"[DEBUG][BIAS] Estimated ba: {ba_init}")
            print(f"[DEBUG][BIAS] Estimated bg: {bg_init}")
            
            bias_msg = f"estimated from {n_static} static samples (body frame): bg={bg_init}, ba={ba_init}"
        else:
            bias_msg = "insufficient samples for bias estimation, using zero"
    else:
        # Biases initialized to zero - will be estimated online by EKF
        bias_msg = "initialized to zero (online estimation enabled)"
    
    # Initial covariance: ESKF uses error-state (15 dimensions)
    # Error state: [δp(3), δv(3), δθ(3), δbg(3), δba(3)]
    # Note: Quaternion error uses 3D rotation vector, NOT 4D quaternion!
    P_pos = 10.0
    P_vel = 4.0
    P_theta = 0.1  # Rotation error (3D) - INCREASED from 1e-3 to allow mag corrections (~18°)
    # Bias uncertainty: larger if not pre-estimated, smaller if estimated from static period
    if estimate_imu_bias and len(imu) >= 100:
        P_bg = (IMU_PARAMS["gyr_w"] * 10) ** 2  # Smaller uncertainty after static calibration
        P_ba = (IMU_PARAMS["acc_w"] * 10) ** 2
    else:
        P_bg = (IMU_PARAMS["gyr_w"] * 1000) ** 2  # Larger uncertainty for online estimation
        P_ba = (IMU_PARAMS["acc_w"] * 1000) ** 2
    
    # ESKF covariance is 15×15 (error-state dimension)
    kf.P = np.diag([
        P_pos, P_pos, P_pos,        # δp (3)
        P_vel, P_vel, P_vel,        # δv (3)
        P_theta, P_theta, P_theta,  # δθ (3) - rotation vector!
        P_bg, P_bg, P_bg,           # δbg (3)
        P_ba, P_ba, P_ba,           # δba (3)
    ]).astype(float)

    print("\n=== Initial State (ESKF: nominal 16D, error 15D) ===")
    print(f"[STATE] Position (p_I): [{kf.x[0,0]:.3f}, {kf.x[1,0]:.3f}, {kf.x[2,0]:.3f}] m (ENU, {z_mode})")
    print(f"[STATE] Velocity (v_I): [{kf.x[3,0]:.6f}, {kf.x[4,0]:.6f}, {kf.x[5,0]:.6f}] m/s (ENU)")
    v_mag_init_kmh = np.linalg.norm(kf.x[3:6]) * 3.6
    print(f"[STATE] Velocity magnitude: {v_mag_init_kmh:.2f} km/h")
    print(f"[STATE] Quaternion (q_I): [{kf.x[6,0]:.4f}, {kf.x[7,0]:.4f}, {kf.x[8,0]:.4f}, {kf.x[9,0]:.4f}] (w,x,y,z)")
    print(f"[STATE] Gyro bias (b_g): [{kf.x[10,0]:.6f}, {kf.x[11,0]:.6f}, {kf.x[12,0]:.6f}] rad/s")
    print(f"[STATE] Accel bias (b_a): [{kf.x[13,0]:.6f}, {kf.x[14,0]:.6f}, {kf.x[15,0]:.6f}] m/s²")
    print(f"[STATE] Bias estimation: {bias_msg}")
    
    print("\n=== Initial Covariance (ESKF error-state 15×15) ===")
    print(f"[COV] Position error (δp): {P_pos:.2e} m²")
    print(f"[COV] Velocity error (δv): {P_vel:.2e} (m/s)²")
    print(f"[COV] Rotation error (δθ): {P_theta:.2e} (3D rotation vector)")
    print(f"[COV] Gyro bias error: {P_bg:.2e} (rad/s)²")
    print(f"[COV] Accel bias error: {P_ba:.2e} (m/s²)²")
    print(f"[INFO] State dimension: x = {kf.x.shape[0]} (nominal), P = {kf.P.shape[0]}×{kf.P.shape[1]} (error-state)")
    
    print("\n--- Initial position/altitude summary ---")
    print(f"MSL0={msl0_m:.2f} m | v_init=[{v_init_enu[0]:.6f}, {v_init_enu[1]:.6f}, {v_init_enu[2]:.6f}] m/s")
    if dem0 is not None:
        print(f"DEM@init={dem0:.2f} m | AGL0=MSL0-DEM={agl0:.2f} m | z_state={z_mode}")
    else:
        print(f"DEM@init=NA | AGL0=MSL0 | z_state={z_mode}")
    print()

    # Statistics counters
    zupt_applied_count = 0
    zupt_rejected_count = 0
    zupt_total_detected = 0
    consecutive_stationary_count = 0  # Track sustained stationary periods

    # Precompute camera intrinsics at runtime size
    K, D = make_KD_for_size(KB_PARAMS, downscale_size[0], downscale_size[1])
    vio_fe: Optional[VIOFrontEnd] = None
    if len(imgs) > 0:
        vio_fe = VIOFrontEnd(downscale_size[0], downscale_size[1], K, D, use_fisheye=USE_FISHEYE)
        # Set camera view mode for VIO
        vio_fe.camera_view = camera_view
        print(f"[VIO] Camera view mode: {camera_view}")

    # Get camera extrinsics based on view mode
    view_cfg = CAMERA_VIEW_CONFIGS.get(camera_view, CAMERA_VIEW_CONFIGS['nadir'])
    extrinsics_name = view_cfg['extrinsics']
    if extrinsics_name == 'BODY_T_CAMDOWN':
        body_t_cam = BODY_T_CAMDOWN
    elif extrinsics_name == 'BODY_T_CAMFRONT':
        body_t_cam = BODY_T_CAMFRONT
    elif extrinsics_name == 'BODY_T_CAMSIDE':
        body_t_cam = BODY_T_CAMSIDE
    else:
        body_t_cam = BODY_T_CAMDOWN  # default
    
    R_cam_to_body = body_t_cam[:3,:3]
    print(f"[VIO] Using extrinsics: {extrinsics_name}")

    # MSCKF helpers: camera-poses appended to the state and simple observation log
    cam_states: List[dict] = []  # entries: {'start_idx', 'q_idx', 'p_idx', 't', 'frame'}
    cam_observations: List[dict] = []  # entries: {'cam_id', 'pts' (Nx2 array)}

    # Output files
    pose_csv = os.path.join(output_dir, "pose.csv")
    with open(pose_csv, "w", newline="") as f:
        f.write(
            "Timestamp(s),dt,Frame,PX,PY,PZ_MSL,VX,VY,VZ,lat,lon,AGL(m),"
            "vo_dx,vo_dy,vo_dz,vo_d_roll,vo_d_pitch,vo_d_yaw\n"
        )

    inf_csv = os.path.join(output_dir, "inference_log.csv")
    with open(inf_csv, "w", newline="") as f:
        f.write("Index,Inference Time (s),FPS\n")

    # Debug file to trace state variables for diagnosing monotonic AGL / v_xy
    state_dbg_csv = os.path.join(output_dir, "state_debug.csv")
    with open(state_dbg_csv, "w", newline="") as f:
        f.write("t,px,py,pz,vx,vy,vz,a_world_x,a_world_y,a_world_z,dem,agl,msl\n")

    vo_dbg = os.path.join(output_dir, "vo_debug.csv")
    with open(vo_dbg, "w", newline="") as vf:
        vf.write(
            "Frame,num_inliers,rot_angle_deg,alignment_deg,rotation_rate_deg_s,use_only_vz,skip_vo,"
            "vo_dx,vo_dy,vo_dz,vel_vx,vel_vy,vel_vz\n"
        )

    # MSCKF debug CSV for tracking multi-view geometric updates
    msckf_dbg = os.path.join(output_dir, "msckf_debug.csv")
    with open(msckf_dbg, "w", newline="") as mf:
        mf.write(
            "frame,feature_id,num_observations,triangulation_success,reprojection_error_px,"
            "innovation_norm,update_applied,chi2_test\n"
        )

    # Error logging CSV (compare VIO vs GPS ground truth)
    error_csv = os.path.join(output_dir, "error_log.csv")
    with open(error_csv, "w", newline="") as ef:
        ef.write(
            "t,pos_error_m,pos_error_E,pos_error_N,pos_error_U,"
            "vel_error_m_s,vel_error_E,vel_error_N,vel_error_U,"
            "alt_error_m,yaw_vio_deg,yaw_gps_deg,yaw_error_deg,"
            "gps_lat,gps_lon,gps_alt,vio_E,vio_N,vio_U\n"
        )

    # ===== COMPREHENSIVE DEBUG DATA LOGGING =====
    # Optional debug outputs enabled by --save_debug_data flag
    
    # 1. Raw IMU data log (for post-processing and validation)
    imu_raw_csv = None
    if save_debug_data:
        imu_raw_csv = os.path.join(output_dir, "debug_imu_raw.csv")
        with open(imu_raw_csv, "w", newline="") as f:
            f.write("t,ori_x,ori_y,ori_z,ori_w,ang_x,ang_y,ang_z,lin_x,lin_y,lin_z\n")
    
    # 2. State & covariance evolution (for observability/consistency analysis)
    state_cov_csv = None
    if save_debug_data:
        state_cov_csv = os.path.join(output_dir, "debug_state_covariance.csv")
        with open(state_cov_csv, "w", newline="") as f:
            f.write("t,frame,P_pos_xx,P_pos_yy,P_pos_zz,P_vel_xx,P_vel_yy,P_vel_zz,"
                   "P_rot_xx,P_rot_yy,P_rot_zz,P_bg_xx,P_bg_yy,P_bg_zz,P_ba_xx,P_ba_yy,P_ba_zz,"
                   "bg_x,bg_y,bg_z,ba_x,ba_y,ba_z\n")
    
    # 3. Residual & innovation log (for measurement quality analysis)
    residual_csv = None
    if save_debug_data:
        residual_csv = os.path.join(output_dir, "debug_residuals.csv")
        with open(residual_csv, "w", newline="") as f:
            f.write("t,frame,update_type,innovation_x,innovation_y,innovation_z,"
                   "mahalanobis_dist,chi2_threshold,accepted,NIS,NEES\n")
    
    # 4. Feature tracking statistics (for front-end quality)
    feature_stats_csv = None
    if save_debug_data:
        feature_stats_csv = os.path.join(output_dir, "debug_feature_stats.csv")
        with open(feature_stats_csv, "w", newline="") as f:
            f.write("frame,t,num_features_detected,num_features_tracked,num_inliers,"
                   "mean_parallax_px,max_parallax_px,tracking_ratio,inlier_ratio\n")
    
    # 5. MSCKF window & marginalization log
    msckf_window_csv = None
    if save_debug_data:
        msckf_window_csv = os.path.join(output_dir, "debug_msckf_window.csv")
        with open(msckf_window_csv, "w", newline="") as f:
            f.write("frame,t,num_camera_clones,num_tracked_features,num_mature_features,"
                   "window_start_time,window_duration,marginalized_clone_id\n")
    
    # NEW: FEJ consistency log
    fej_consistency_csv = None
    if save_debug_data:
        fej_consistency_csv = os.path.join(output_dir, "debug_fej_consistency.csv")
        with open(fej_consistency_csv, "w", newline="") as f:
            f.write("timestamp,frame,clone_idx,pos_fej_drift_m,rot_fej_drift_deg,"
                   "bg_fej_drift_rad_s,ba_fej_drift_m_s2\n")
    
    # 6. Calibration snapshot (save initial parameters for reproducibility)
    calibration_log = None
    if save_debug_data:
        calibration_log = os.path.join(output_dir, "debug_calibration.txt")
        with open(calibration_log, "w") as f:
            f.write("=== VIO System Calibration & Configuration ===\n\n")
            f.write(f"[Camera View]\n")
            f.write(f"  Mode: {camera_view}\n")
            f.write(f"  Extrinsics: {view_cfg['extrinsics']}\n")
            f.write(f"  Nadir threshold: {view_cfg['nadir_threshold']}°\n\n")
            
            f.write(f"[Camera Intrinsics - Kannala-Brandt]\n")
            for key, val in KB_PARAMS.items():
                f.write(f"  {key}: {val}\n")
            f.write(f"  Runtime size: {downscale_size[0]}x{downscale_size[1]}\n\n")
            
            f.write(f"[IMU Parameters]\n")
            for key, val in IMU_PARAMS.items():
                f.write(f"  {key}: {val}\n")
            f.write(f"\n[EKF Process Noise]\n")
            f.write(f"  SIGMA_ACCEL: {SIGMA_ACCEL}\n")
            f.write(f"  SIGMA_VO_VEL: {SIGMA_VO_VEL}\n")
            f.write(f"  SIGMA_VPS_XY: {SIGMA_VPS_XY}\n")
            f.write(f"  SIGMA_AGL_Z: {SIGMA_AGL_Z}\n")
            f.write(f"  SIGMA_MAG_YAW: {SIGMA_MAG_YAW}\n\n")
            
            f.write(f"[VIO Quality Control]\n")
            f.write(f"  MIN_PARALLAX_PX: {MIN_PARALLAX_PX}\n")
            f.write(f"  MIN_MSCKF_BASELINE: {MIN_MSCKF_BASELINE}\n")
            f.write(f"  VO_MIN_INLIERS: {VO_MIN_INLIERS}\n")
            f.write(f"  VO_RATIO_TEST: {VO_RATIO_TEST}\n\n")
            
            f.write(f"[Magnetometer Calibration]\n")
            f.write(f"  Hard-iron offset: {MAG_HARD_IRON_OFFSET}\n")
            f.write(f"  Declination: {np.rad2deg(MAG_DECLINATION):.2f}°\n")
            f.write(f"  Expected field: {MAG_FIELD_STRENGTH:.1f} µT\n")
            f.write(f"  Min/Max field: {MAG_MIN_FIELD_STRENGTH:.1f} - {MAG_MAX_FIELD_STRENGTH:.1f} µT\n\n")
            
            f.write(f"[Initial State]\n")
            f.write(f"  Origin: ({lat0:.8f}°, {lon0:.8f}°)\n")
            f.write(f"  MSL altitude: {msl0_m:.2f} m\n")
            if dem0 is not None:
                f.write(f"  DEM elevation: {dem0:.2f} m\n")
                f.write(f"  AGL altitude: {agl0:.2f} m\n")
            f.write(f"  Initial velocity: {v_init_enu} m/s\n")
            f.write(f"  Estimate IMU bias: {estimate_imu_bias}\n")
            if estimate_imu_bias:
                f.write(f"    bg_init: {kf.x[9:12,0].flatten()}\n")
                f.write(f"    ba_init: {kf.x[12:15,0].flatten()}\n")
    
    # 7. Keyframe image directory (with feature overlays)
    keyframe_dir = None
    if save_keyframe_images:
        keyframe_dir = os.path.join(output_dir, "debug_keyframes")
        os.makedirs(keyframe_dir, exist_ok=True)
        print(f"[DEBUG] Keyframe images will be saved to: {keyframe_dir}")

    # 8. Loop Closure Detector
    # Initialize loop closure detector for yaw drift correction
    loop_detector = init_loop_closure(position_threshold=50.0)
    loop_closure_enabled = True  # Can be disabled via config if needed
    print(f"[LOOP] Loop closure detector initialized (threshold={loop_detector.position_threshold}m)")

    # VPS iterator
    vps_idx = 0

    # Image iterator
    img_idx = 0
    vio_frame = -1  # first VIO frame will become 0
    
    # Flag: Did we initialize yaw from PPK ground truth?
    # If yes, skip mag updates during convergence period to preserve accurate PPK yaw
    has_ppk_initial_yaw = (ppk_state is not None)

    # Magnetometer iterator
    mag_idx = 0
    mag_update_count = 0
    mag_reject_count = 0
    
    # =====================================================
    # YAW RESET TRACKING (Goal 1)
    # Track prolonged high yaw innovation for potential reset
    # =====================================================
    high_yaw_innovation_start_time = None  # When high innovation started
    HIGH_YAW_INNOVATION_THRESHOLD = np.radians(90.0)  # 90° threshold
    HIGH_YAW_INNOVATION_DURATION = 5.0  # seconds before reset
    yaw_reset_count = 0
    
    # =====================================================
    # VIBRATION DETECTION (Goal 2)
    # Track gyro variance over sliding window for vibration detection
    # =====================================================
    gyro_buffer = []  # Rolling buffer of recent gyro readings
    GYRO_BUFFER_SIZE = 100  # ~0.25s at 400Hz
    VIBRATION_THRESHOLD = 0.5  # rad/s std - extreme vibration
    VIBRATION_NOISE_INFLATE = 3.0  # Inflate process noise by this factor during vibration
    current_vibration_level = 0.0  # Current gyro std
    vibration_detected_count = 0
    
    # =====================================================
    # SENSOR QUALITY METRICS (Goal 3)
    # Replace time-based K_MIN with quality-based approach
    # =====================================================
    mag_quality_history = []  # Recent MAG quality scores
    MAG_QUALITY_BUFFER_SIZE = 20  # ~1 second of MAG readings (20Hz)
    innovation_history = []  # Recent innovation magnitudes
    INNOVATION_BUFFER_SIZE = 20

    t0 = imu[0].t
    last_t = t0

    # =====================================================
    # FLIGHT PHASE DETECTION
    # Phase 0: Rotor spin-up (0-15s) - high vibration, don't trust IMU heading
    # Phase 1: Early flight (15-60s) - stabilizing, gradually trust mag more
    # Phase 2: Normal flight (>60s) - full trust in mag corrections
    # =====================================================
    PHASE_SPINUP_END = 15.0      # seconds - rotor spin-up period
    PHASE_EARLY_FLIGHT_END = 60.0  # seconds - early flight period
    current_flight_phase = 0
    flight_phase_names = ["SPINUP", "EARLY", "NORMAL"]

    # GRAVITY VECTOR convention:
    # - Stationary IMU after rotation to world: a_world ≈ [0, 0, +9.8] (specific force)
    # - To get motion acceleration: a_motion = a_world - expected_stationary
    # - Therefore: g_world = [0, 0, +9.803] for SUBTRACTION
    # This matches bias calculation where expected_world = [0, 0, +9.803]
    g_world = np.array([0.0, 0.0, +IMU_PARAMS["g_norm"]])

    # ====================
    # Preintegration Buffer
    # ====================
    # Initialize ongoing preintegration buffer for main loop
    # This accumulates IMU measurements between state updates
    if use_preintegration:
        ongoing_preint = IMUPreintegration(
            bg=kf.x[10:13, 0].reshape(3,),
            ba=kf.x[13:16, 0].reshape(3,),
            sigma_g=IMU_PARAMS["gyr_n"],    # Gyro measurement noise
            sigma_a=IMU_PARAMS["acc_n"],    # Accel measurement noise
            sigma_bg=IMU_PARAMS["gyr_w"],   # Gyro random walk
            sigma_ba=IMU_PARAMS["acc_w"]    # Accel random walk
        )
        preint_start_time = t0
        preint_start_state = kf.x.copy()
        print(f"[PREINT] Initialized ongoing preintegration buffer (use_preintegration={use_preintegration})")
    else:
        ongoing_preint = None

    # --------------
    # Main IMU loop
    # --------------
    print("\n=== Running (IMU-driven) ===")
    tic_all = time.time()
    for i, rec in enumerate(imu):
        tic = time.time()
        t = rec.t
        dt = max(0.0, float(t - last_t)) if i > 0 else 0.0
        last_t = t

        # ------- IMU measurement from sensor (used for VO direction scaling) -------
        # Xsens MTi-30: ENU world frame, Z-Up body frame
        # Use quaternion directly (NO ENU→NED conversion!)
        Rwb = R_scipy.from_quat(rec.q).as_matrix()  # Body-to-World (ENU)

        # IMU-driven propagation for expanded 16-element IMU state
        # State layout: [p(3), v(3), q(4:w,x,y,z), bg(3), ba(3)]
        x = kf.x.reshape(-1).copy()
        p = x[0:3].reshape(3,)
        v = x[3:6].reshape(3,)
        q = x[6:10].reshape(4,)  # [w,x,y,z]
        bg = x[10:13].reshape(3,)
        ba = x[13:16].reshape(3,)

        # measurements
        a_meas = rec.lin.astype(float)
        w_meas = rec.ang.astype(float)

        # Compute bias-corrected measurements (needed for ZUPT and other checks)
        a_corr = a_meas - ba
        w_corr = w_meas - bg

        # =====================================================
        # VIBRATION DETECTION (Goal 2)
        # Track gyro magnitude in rolling buffer to detect vibration
        # High vibration = high gyro std = unreliable integration
        # =====================================================
        gyro_mag = np.linalg.norm(w_corr)
        gyro_buffer.append(gyro_mag)
        if len(gyro_buffer) > GYRO_BUFFER_SIZE:
            gyro_buffer.pop(0)
        
        # Compute gyro std over buffer
        if len(gyro_buffer) >= 10:  # Need minimum samples
            current_vibration_level = np.std(gyro_buffer)
            is_high_vibration = current_vibration_level > VIBRATION_THRESHOLD
            if is_high_vibration:
                vibration_detected_count += 1
        else:
            is_high_vibration = False
        # Compute rotation matrix for transforming to world frame (needed for ZUPT)
        quat_xyzw = np.array([q[1], q[2], q[3], q[0]])  # to [x,y,z,w] for scipy
        R_world_to_body = R_scipy.from_quat(quat_xyzw).as_matrix()
        R_body_to_world = R_world_to_body.T  # Transpose = inverse for rotation matrices
        
        # Transform acceleration to world frame and SUBTRACT GRAVITY
        # IMU data is RAW (not gravity-compensated)
        # In ENU frame: gravity = [0, 0, -g] (points down)
        # Raw accel when stationary: a_world_raw ≈ [0, 0, +g] (reaction force)
        # Motion acceleration: a_motion = a_world_raw - [0, 0, g]
        g_world = np.array([0.0, 0.0, -IMU_PARAMS["g_norm"]])  # ENU: Z-up, gravity down
        a_world_raw = R_body_to_world @ a_corr
        a_world = a_world_raw + g_world  # Subtract gravity (add because g_world is negative)

        # ====================
        # Preintegration Mode
        # ====================
        if use_preintegration and ongoing_preint is not None:
            # IMPORTANT: IMU data is RAW (includes gravity)
            # integrate_measurement() does its own bias correction and gravity compensation
            # So we pass RAW a_meas, not a_corr!
            
            ongoing_preint.integrate_measurement(w_meas, a_meas, dt)  # Pass RAW measurements!
            
            # Update biases in buffer if they changed (e.g., from measurement updates)
            # This ensures Jacobians are computed correctly
            current_bg = kf.x[10:13, 0].reshape(3,)
            current_ba = kf.x[13:16, 0].reshape(3,)
            if not np.allclose(ongoing_preint.bg_lin, current_bg, atol=1e-6) or \
               not np.allclose(ongoing_preint.ba_lin, current_ba, atol=1e-6):
                # Bias changed - need to repropagate with new bias
                # This is expensive but necessary for correctness after measurement updates
                pass  # For now, keep existing buffer (OpenVINS assumes bias updates are small)
            
            # Save priors (state doesn't change between camera frames in preint mode)
            kf.x_prior = kf.x.copy()
            kf.P_prior = kf.P.copy()
        else:
            # Legacy discrete integration mode
            
            # Discrete integration using bias-corrected acceleration
            # a_world = R * (a_meas - ba) is in world frame
            # For RAW IMU: includes gravity component
            # For this integration, we use a_world directly
            
            # integrate state (simple discrete integration)
            p_new = p + v*dt + 0.5 * a_world * (dt*dt)
            v_new = v + a_world * dt

            # Improved quaternion integration using exponential map (more accurate than small-angle)
            # Compute rotation vector: theta = w_corr * dt
            theta_vec = w_corr * dt
            theta = np.linalg.norm(theta_vec)
            
            if theta < 1e-8:
                # Small angle: use first-order approximation
                dq = np.array([1.0, 0.5*theta_vec[0], 0.5*theta_vec[1], 0.5*theta_vec[2]], dtype=float).reshape(4,1)
            else:
                # Exponential map: dq = [cos(theta/2), sin(theta/2) * axis]
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

            # Biases evolve as random walk (no dynamics, but will be updated by measurements)
            # For now, keep constant during propagation
            bg_new = bg
            ba_new = ba

            # write back state
            kf.x[0:3,0] = p_new
            kf.x[3:6,0] = v_new
            kf.x[6:10,0] = q_new
            kf.x[10:13,0] = bg_new
            kf.x[13:16,0] = ba_new

            # ====================
            # ESKF Covariance Propagation (Legacy)
            # ====================
            # Error-state transition matrix (15×15 for core state)
            Phi_err = compute_error_state_jacobian(q, a_corr, w_corr, dt, R_body_to_world)
            
            # Error-state process noise (15×15 for core state)
            Q_err = compute_error_state_process_noise(dt, estimate_imu_bias, t, t0)
            
            # =====================================================
            # VIBRATION-BASED PROCESS NOISE INFLATION (Goal 2)
            # During high vibration, IMU integration is less reliable
            # Inflate process noise to allow corrections from MAG/VIO
            # =====================================================
            if is_high_vibration:
                # Inflate rotation and velocity noise during high vibration
                Q_err[6:9, 6:9] *= VIBRATION_NOISE_INFLATE  # Rotation noise
                Q_err[3:6, 3:6] *= VIBRATION_NOISE_INFLATE  # Velocity noise
                if i % 500 == 0:
                    print(f"[VIBRATION] t={time_elapsed:.1f}s: High vibration detected (std={current_vibration_level:.3f} rad/s) - inflating Q by {VIBRATION_NOISE_INFLATE}x")
            
            # Get number of camera clones
            num_clones = (kf.x.shape[0] - 16) // 7
            
            # Propagate error-state covariance with clones
            kf.P = propagate_error_state_covariance(kf.P, Phi_err, Q_err, num_clones)
            
            # Save priors
            kf.x_prior = kf.x.copy()
            kf.P_prior = kf.P.copy()
        
        # ===== DEBUG: Log raw IMU data =====
        if save_debug_data and imu_raw_csv:
            with open(imu_raw_csv, "a", newline="") as f:
                f.write(f"{t:.6f},{rec.q[0]:.6f},{rec.q[1]:.6f},{rec.q[2]:.6f},{rec.q[3]:.6f},"
                       f"{rec.ang[0]:.6f},{rec.ang[1]:.6f},{rec.ang[2]:.6f},"
                       f"{rec.lin[0]:.6f},{rec.lin[1]:.6f},{rec.lin[2]:.6f}\n")
        
        # ===== DEBUG: Log state & covariance =====
        if save_debug_data and state_cov_csv and i % 10 == 0:  # Every 10 samples (~25ms @ 400Hz)
            with open(state_cov_csv, "a", newline="") as f:
                # Extract diagonal elements of covariance blocks
                P_pos = [kf.P[0,0], kf.P[1,1], kf.P[2,2]]
                P_vel = [kf.P[3,3], kf.P[4,4], kf.P[5,5]]
                P_rot = [kf.P[6,6], kf.P[7,7], kf.P[8,8]]
                P_bg = [kf.P[9,9], kf.P[10,10], kf.P[11,11]]
                P_ba = [kf.P[12,12], kf.P[13,13], kf.P[14,14]]
                f.write(f"{t:.6f},{vio_frame},{P_pos[0]:.6e},{P_pos[1]:.6e},{P_pos[2]:.6e},"
                       f"{P_vel[0]:.6e},{P_vel[1]:.6e},{P_vel[2]:.6e},"
                       f"{P_rot[0]:.6e},{P_rot[1]:.6e},{P_rot[2]:.6e},"
                       f"{P_bg[0]:.6e},{P_bg[1]:.6e},{P_bg[2]:.6e},"
                       f"{P_ba[0]:.6e},{P_ba[1]:.6e},{P_ba[2]:.6e},"
                       f"{bg[0]:.6f},{bg[1]:.6f},{bg[2]:.6f},"
                       f"{ba[0]:.6f},{ba[1]:.6f},{ba[2]:.6f}\n")
        
        # Debug logging (every 1000 samples)
        if i % 1000 == 0:
            print(f"[ESKF] t={t:.1f}s | P_shape={kf.P.shape} | "
                  f"P_diag: pos={np.sqrt(kf.P[0,0]):.2f}m vel={np.sqrt(kf.P[3,3]):.2f}m/s "
                  f"theta={np.sqrt(kf.P[6,6]):.4f}rad bg={kf.P[9,9]:.2e} ba={kf.P[12,12]:.2e}")

        # -------------- ZUPT (Zero Velocity Update) --------------
        # Detect stationary periods by checking IMU measurements
        # When stationary, constrain velocity to zero to prevent drift
        
        # Compute number of clones (needed for ZUPT and other updates)
        num_clones = (kf.x.shape[0] - 16) // 7
        
        # IMPORTANT: Use RAW acceleration (not bias-corrected) for stationary detection
        # Because when bias is unknown, a_corr will be incorrect, but raw accel magnitude
        # still reflects true motion (gravity ± small noise when stationary)
        a_raw = rec.lin  # Raw acceleration from IMU
        a_raw_mag = np.linalg.norm(a_raw)
        gyro_mag = np.linalg.norm(w_corr)  # Gyro can use corrected (bias is smaller)
        v_mag = np.linalg.norm(v)  # Current velocity magnitude
        time_elapsed = t - t0  # Compute this for all modes
        acc_deviation = abs(a_raw_mag - IMU_PARAMS["g_norm"])  # For logging
        
        # ZUPT thresholds (tuned for consumer-grade IMU)
        # When hovering/stationary: acceleration ≈ gravity magnitude, gyro ≈ 0
        # IMPORTANT: ZUPT works with preintegration!
        # Preintegration handles IMU integration between keyframes, ZUPT corrects velocity bias AFTER application
        # ZUPT operates on nominal state (v) not on preintegrated Δv, so no conflict
        ZUPT_ACC_THRESHOLD = 0.5   # m/s² - same for both legacy and preintegration
        ZUPT_GYRO_THRESHOLD = 0.05  # rad/s - same for both legacy and preintegration
        # Check if stationary: raw acc magnitude close to gravity, gyro near zero
        is_stationary = (acc_deviation < ZUPT_ACC_THRESHOLD) and (gyro_mag < ZUPT_GYRO_THRESHOLD)
        
        # ADDITIONAL: Velocity-aided ZUPT (for aggressive bias correction when no initial estimate)
        # If velocity is growing unrealistically high but IMU shows low motion, apply soft ZUPT
        velocity_aided_zupt = False
        if not estimate_imu_bias and v_mag > 10.0:  # 10 m/s = 36 km/h threshold
            # Check if current acceleration is small (vehicle likely slowing down or stationary)
            a_world_mag = np.linalg.norm(a_world)
            if a_world_mag < 1.0 and gyro_mag < 0.1:  # Relaxed thresholds
                # Apply velocity damping (not full ZUPT)
                velocity_aided_zupt = True
                is_stationary = True
                if i % 100 == 0:  # Log occasionally
                    print(f"[DEBUG][ZUPT-VEL] Velocity damping at t={t:.3f} | v_mag={v_mag:.2f}m/s a_world={a_world_mag:.2f}m/s²")
        
        # Log stationary detection (every 500 samples to avoid spam)
        if i % 500 == 0:
            status = "✓ STATIONARY" if is_stationary else "✗ MOVING"
            print(f"[DEBUG][ZUPT-DETECT] t={time_elapsed:.1f}s {status} | acc_dev={acc_deviation:.3f}m/s² | "
                  f"gyro={gyro_mag:.3f}rad/s | v_mag={v_mag:.1f}m/s")
        
        if is_stationary:
            zupt_total_detected += 1
            consecutive_stationary_count += 1  # Increment for sustained stationary
            
            # Apply ZUPT: constrain velocity to zero
            # ESKF: Jacobian maps to error state (15 + 6*clones)
            num_clones = (kf.x.shape[0] - 16) // 7  # FIX: calculate from nominal state
            err_dim = 15 + 6 * num_clones
            H_zupt = np.zeros((3, err_dim), dtype=float)
            H_zupt[0:3, 3:6] = np.eye(3)  # Measure velocity error (δv)
            
            def h_zupt_jacobian(x, h=H_zupt):
                return h
            
            def hx_zupt_fun(x, h=H_zupt):
                # x is nominal state (16+7M), measurement is velocity (3)
                return x[3:6].reshape(3, 1)
            
            # ZUPT measurement: velocity should be zero
            z_zupt = np.zeros((3, 1), dtype=float)
            
            # ZUPT measurement noise - MUCH MORE AGGRESSIVE
            # Use consecutive stationary count to increase confidence
            
            # Base R values - very confident (reduced 10-100x from before)
            if v_mag < 1.0:  # Very low velocity
                base_r = 0.0001  # Was 0.01 - now 100x more confident
            elif v_mag < 5.0:  # Low-medium velocity
                base_r = 0.001   # Was 0.1 - now 100x more confident
            else:  # Higher velocity
                base_r = 0.01    # Was 0.1-10 - now 10-1000x more confident
            
            # Further strengthen R for sustained stationary periods
            consecutive_factor = min(10.0, consecutive_stationary_count / 100.0)
            if consecutive_factor < 1.0:
                consecutive_factor = 1.0
            
            # Final R matrix - divide by consecutive_factor to increase confidence
            R_zupt = np.diag([base_r / consecutive_factor, 
                             base_r / consecutive_factor, 
                             base_r / consecutive_factor])
            
            # SIMPLIFIED gating: check if IMU truly shows stationary (primary check already done)
            # Secondary check: don't apply ZUPT if velocity is unreasonably high (> 500 m/s = 1800 km/h)
            # This indicates severe drift and ZUPT might cause instability
            max_v_for_zupt = 500.0  # m/s
            
            if v_mag < max_v_for_zupt:
                zupt_applied_count += 1
                v_before = kf.x[3:6,0].copy()
                
                # =====================================================
                # CRITICAL FIX: Decouple yaw from velocity before ZUPT!
                # ZUPT observes velocity, but cross-covariance P(v,θ) can
                # cause yaw to be modified by ZUPT update via Kalman gain.
                # This was causing massive yaw drift (200°/s apparent rate)
                # even though gyro shows only ~1°/s!
                # =====================================================
                # Zero out cross-covariance between velocity and yaw
                kf.P[3:6, 8] = 0.0  # Cov(v, yaw)
                kf.P[8, 3:6] = 0.0
                # Also decouple velocity from roll/pitch (for altitude stability)
                kf.P[3:6, 6:8] = 0.0  # Cov(v, roll/pitch)
                kf.P[6:8, 3:6] = 0.0
                
                # Compute innovation for ZUPT (before update)
                zupt_innovation = z_zupt - kf.x[3:6].reshape(3,1)
                zupt_s_mat = h_zupt_jacobian(None) @ kf.P @ h_zupt_jacobian(None).T + R_zupt
                try:
                    zupt_m2_test = _mahalanobis2(zupt_innovation, zupt_s_mat)
                except Exception:
                    zupt_m2_test = 0.0
                
                kf.update(
                    z=z_zupt,
                    HJacobian=h_zupt_jacobian,
                    Hx=hx_zupt_fun,
                    R=R_zupt
                )
                v_after = kf.x[3:6,0].copy()
                v_before_mag = np.linalg.norm(v_before)
                v_after_mag = np.linalg.norm(v_after)
                v_reduction = v_before_mag - v_after_mag
                
                # Log residual for debugging
                if save_debug_data and residual_csv:
                    log_measurement_update(
                        residual_csv, t, vio_frame, 'ZUPT',
                        innovation=zupt_innovation,
                        mahalanobis_dist=np.sqrt(zupt_m2_test),
                        chi2_threshold=np.inf,  # ZUPT doesn't use chi-square gating
                        accepted=True,
                        S_matrix=zupt_s_mat,
                        P_prior=kf.P_prior
                    )
                
                # Log every ZUPT update (not too frequent)
                if zupt_applied_count % 50 == 1 or zupt_applied_count <= 10:
                    print(f"[DEBUG][ZUPT] APPLIED #{zupt_applied_count} at t={t:.3f} | "
                          f"v: {v_before_mag:.2f} → {v_after_mag:.2f} m/s (Δ={v_reduction:.2f}) | "
                          f"acc_dev={acc_deviation:.4f} gyro={gyro_mag:.4f} | R_scale={(R_zupt[0,0]/0.01):.1f}x")
            else:
                zupt_rejected_count += 1
                if zupt_rejected_count % 100 == 1:  # Log occasionally
                    print(f"[DEBUG][ZUPT] REJECTED at t={t:.3f} | v_mag={v_mag:.1f}m/s > {max_v_for_zupt}m/s (too high)")
        else:
            # Reset consecutive counter when not stationary
            consecutive_stationary_count = 0


        # -------------- Optional VPS update (XY) with adaptive uncertainty --------------
        # DEBUG: Log VPS timing
        if i == 0 and len(vps_list) > 0:
            print(f"[VPS_DEBUG] VPS range: {vps_list[0].t:.3f} - {vps_list[-1].t:.3f} ({len(vps_list)} items)")
            print(f"[VPS_DEBUG] IMU first t: {t:.3f}")
        if vps_idx < len(vps_list) and i % 100 == 0:  # Log every 100 iterations
            print(f"[VPS_DEBUG] i={i} t={t:.3f} vps_idx={vps_idx}/{len(vps_list)} vps.t={vps_list[vps_idx].t:.3f} diff={vps_list[vps_idx].t - t:.3f}")
        
        while vps_idx < len(vps_list) and vps_list[vps_idx].t <= t:
            print(f"[VPS_DEBUG] ✓ ENTERING UPDATE: vps_idx={vps_idx} vps.t={vps_list[vps_idx].t:.3f} current_t={t:.3f}")
            vps = vps_list[vps_idx]
            vps_idx += 1
            vps_xy = latlon_to_xy(vps.lat, vps.lon, lat0, lon0)
            
            # ESKF: VPS Jacobian maps to error state (15 + 6*clones)
            num_clones = (kf.x.shape[0] - 16) // 7  # FIX: calculate num_clones from nominal state
            err_dim = 15 + 6 * num_clones
            h_xy = np.zeros((2, err_dim), dtype=float)
            h_xy[0, 0] = 1.0  # Measure position error δp_x
            h_xy[1, 1] = 1.0  # Measure position error δp_y

            def h_fun(x, h=h_xy):
                return h

            def hx_fun(x, h=h_xy):
                # x is nominal state, measurement is position (2D)
                return x[0:2].reshape(2, 1)

            # Adaptive uncertainty based on speed
            speed_xy = float(math.hypot(kf.x[3,0], kf.x[4,0]))
            scale = 1.0 + max(0.0, (speed_xy-10.0)/10.0) if speed_xy>10 else 1.0
            r_mat = np.diag([(SIGMA_VPS_XY**2)*scale, (SIGMA_VPS_XY**2)*scale])

            # Innovation gating
            s_mat = h_xy @ kf.P @ h_xy.T + r_mat
            try:
                xy_pred = kf.x[0:2].reshape(2,)
                innovation = (vps_xy - xy_pred).reshape(-1,1)
                m2_test = _mahalanobis2(innovation, s_mat)
            except Exception as e:
                print(f"[VPS] ERROR computing innovation: {e}")
                m2_test = np.inf

            # Log every VPS update attempt
            print(f"[VPS] t={t:.3f} meas={vps_xy} pred={xy_pred} innov={innovation.flatten()} m2={m2_test:.1f}")
            
            # =====================================================================
            # ADAPTIVE VPS INTEGRATION STRATEGY (Real-time GPS-denied navigation)
            # =====================================================================
            # Problem: VPS arrives irregularly (t=10s, t=24s, t=40s...) with large innovations
            # Solution: Multi-tier acceptance based on drift time + innovation magnitude
            #
            # Tier 1 (>60s): FIRST VPS - use innovation magnitude threshold (not chi-square)
            #   - VIO drifted far → accept large innovation (e.g., 5km) with very low weight
            #   - Prevents rejection of first absolute position after long GPS outage
            #   - Uses R_scale=20-50x → EKF gradually pulls toward VPS (no sudden jump)
            #
            # Tier 2 (10-60s): LONG DRIFT - permissive chi-square + adaptive R scaling
            #   - Accepts innovations up to chi2=1000 (vs normal 13.82)
            #   - R_scale increases with drift time → smoother correction
            #
            # Tier 3 (<10s): RECENT VPS - strict chi-square gating
            #   - Trust VIO, reject outliers aggressively
            # =====================================================================
            
            time_since_correction = t - kf.last_absolute_correction_time
            innovation_mag = np.linalg.norm(innovation)
            
            # =====================================================================
            # INNOVATION-BASED VPS ACCEPTANCE (All tiers use innovation threshold)
            # =====================================================================
            # Problem: Chi-square test rejects valid VPS when VIO covariance is underestimated
            # Solution: Use innovation magnitude threshold that scales with drift time
            #
            # UPDATED: Helicopter VIO drift rate observed ~36+ m/s (130+ km/h flight)
            # Must allow very high drift rate to accept VPS during fast flight
            # =====================================================================
            
            # Base thresholds for each tier (INCREASED for helicopter flight)
            if time_since_correction > 60.0:
                # TIER 1: Very long drift - accept up to 200 m/s * time
                base_threshold_m = 200.0
                max_drift_rate = 200.0  # m/s assumed max drift
                r_scale = min(50.0, 10.0 + time_since_correction / 10.0)
                tier_name = "FIRST VPS"
            elif time_since_correction > 10.0:
                # TIER 2: Long drift - accept up to 100 m/s * time + 100m base
                base_threshold_m = 100.0
                max_drift_rate = 100.0  # m/s
                r_scale = min(10.0, 1.0 + time_since_correction / 5.0)
                tier_name = "LONG DRIFT"
            elif time_since_correction > 3.0:
                # TIER 3a: Medium drift (3-10s) - accept up to 60 m/s * time + 50m base
                base_threshold_m = 50.0
                max_drift_rate = 60.0  # m/s (helicopter ~36 m/s observed)
                r_scale = min(5.0, 1.0 + time_since_correction / 3.0)
                tier_name = "MEDIUM DRIFT"
            else:
                # TIER 3b: Recent VPS (<3s) - accept up to 50 m/s * time + 50m base
                # CRITICAL: Must allow 36+ m/s drift rate for helicopter
                base_threshold_m = 50.0
                max_drift_rate = 50.0  # m/s (helicopter drift ~36 m/s per second)
                r_scale = 1.0
                tier_name = "RECENT"
            
            # Calculate innovation threshold: base + drift_rate * time
            max_innovation_m = base_threshold_m + max_drift_rate * time_since_correction
            max_innovation_m = min(max_innovation_m, 30000.0)  # Cap at 30km
            
            # Accept if innovation < threshold
            chi2_accepted = innovation_mag < max_innovation_m
            
            if chi2_accepted:
                print(f"[VPS] ✓ {tier_name} ({time_since_correction:.1f}s): innov={innovation_mag:.1f}m < {max_innovation_m:.1f}m, R_scale={r_scale:.1f}x")
            else:
                print(f"[VPS] ✗ {tier_name} REJECTED ({time_since_correction:.1f}s): innov={innovation_mag:.1f}m > {max_innovation_m:.1f}m (OUTLIER)")
            
            # Apply R matrix scaling for all tiers
            r_mat_scaled = r_mat * (r_scale ** 2)
            
            # Execute VPS update if accepted
            if chi2_accepted:
                kf.update(
                    z=vps_xy.reshape(-1,1),
                    HJacobian=h_fun,
                    Hx=hx_fun,
                    R=r_mat_scaled  # Use scaled R for drift compensation
                )
                
                # Update timestamp for DEM adaptive noise
                kf.last_absolute_correction_time = t
                
                # Log residual for debugging
                if save_debug_data and residual_csv:
                    log_measurement_update(
                        residual_csv, t, vio_frame, 'VPS',
                        innovation=innovation,
                        mahalanobis_dist=np.sqrt(m2_test),
                        chi2_threshold=9.21,
                        accepted=True,
                        S_matrix=s_mat,
                        P_prior=kf.P_prior
                    )
            else:
                print(f"[VPS] ✗ UPDATE REJECTED at t={t:.3f} innov={innovation_mag:.1f}m > {max_innovation_m:.1f}m (drift={time_since_correction:.1f}s)")
                
                # Log rejected update
                if save_debug_data and residual_csv:
                    log_measurement_update(
                        residual_csv, t, vio_frame, 'VPS',
                        innovation=innovation,
                        mahalanobis_dist=np.sqrt(m2_test),
                        chi2_threshold=9.21,
                        accepted=False,
                        S_matrix=s_mat
                    )

        # -------------- Magnetometer heading update --------------
        # Process magnetometer measurements at current IMU timestamp
        while mag_idx < len(mag_list) and mag_list[mag_idx].t <= t:
            mag_rec = mag_list[mag_idx]
            mag_idx += 1
            
            # Rate limiting: only process every Nth sample to avoid over-constraining
            if (mag_idx-1) % MAG_UPDATE_RATE_LIMIT != 0:
                continue
            
            # Calibrate magnetometer
            mag_cal = calibrate_magnetometer(mag_rec.mag)
            
            # Check field strength (reject outliers)
            mag_norm = np.linalg.norm(mag_cal)
            
            if mag_norm < MAG_MIN_FIELD_STRENGTH or mag_norm > MAG_MAX_FIELD_STRENGTH:
                mag_reject_count += 1
                if i % 100 == 0:  # Print periodically
                    print(f"[MAG] t={t:.2f}: Rejected (field strength {mag_norm:.1f} µT, expect {MAG_MIN_FIELD_STRENGTH}-{MAG_MAX_FIELD_STRENGTH})")
                continue
            
            # =====================================================
            # SPEED-BASED MAG QUALITY CHECK (DISABLED)
            # FIX 2025-12-02a: REVERT speed-based rejection
            # Analysis shows speed-based rejection makes performance WORSE:
            # - With speed filtering: 3228m RMSE
            # - Without (v9): 736m RMSE
            # 
            # The issue is MAG is still needed for correction even at hover.
            # Better approach: use innovation-based rejection instead.
            # =====================================================
            speed_factor = 1.0  # Always full trust
            
            # Compute yaw from magnetometer
            # Use GPS-calibrated raw heading (default) or tilt-compensated
            q_state = kf.x[6:10, 0]  # Current quaternion [w,x,y,z]
            yaw_mag, quality = compute_yaw_from_mag(
                mag_cal, q_state, 
                mag_declination=MAG_DECLINATION,
                use_raw_heading=MAG_USE_RAW_HEADING
            )
            
            # Reject if quality too low (near vertical attitude)
            # Quality threshold: 0.3 for conservative field strength filtering
            if quality < 0.3:
                mag_reject_count += 1
                if i % 100 == 0:
                    print(f"[MAG] t={t:.2f}: Rejected (quality {quality:.3f} < 0.3)")
                continue
            
            # Extract current yaw from state
            yaw_state = quaternion_to_yaw(q_state)
            
            # =====================================================
            # YAW WRAP-AROUND HANDLING
            # When yaw approaches ±180°, naive subtraction can give
            # wrong sign. Use atan2 to properly wrap to [-π, π].
            # Also detect and log wrap-around events.
            # =====================================================
            yaw_innov = yaw_mag - yaw_state
            yaw_innov = np.arctan2(np.sin(yaw_innov), np.cos(yaw_innov))
            
            # Detect wrap-around: if state and mag are on opposite sides of ±180°
            near_wrap = abs(yaw_state) > np.radians(160.0) or abs(yaw_mag) > np.radians(160.0)
            wrapped = near_wrap and (np.sign(yaw_state) != np.sign(yaw_mag))
            
            if wrapped and i % 50 == 0:
                print(f"[MAG] WRAP-AROUND: yaw_state={np.degrees(yaw_state):.1f}°, yaw_mag={np.degrees(yaw_mag):.1f}°, innov={np.degrees(yaw_innov):.1f}°")
            
            # =====================================================
            # YAW RESET ON PROLONGED HIGH INNOVATION (Goal 1)
            # If innovation > 90° for more than 5 seconds, something is
            # seriously wrong - reset yaw to MAG value directly
            # =====================================================
            if abs(yaw_innov) > HIGH_YAW_INNOVATION_THRESHOLD:
                if high_yaw_innovation_start_time is None:
                    high_yaw_innovation_start_time = t
                    print(f"[MAG] HIGH INNOVATION START: |{np.degrees(yaw_innov):.1f}°| > {np.degrees(HIGH_YAW_INNOVATION_THRESHOLD):.0f}° at t={time_elapsed:.1f}s")
                else:
                    duration = t - high_yaw_innovation_start_time
                    if duration > HIGH_YAW_INNOVATION_DURATION:
                        # RESET YAW STATE TO MAGNETOMETER
                        print(f"[MAG] ⚠️ YAW RESET #{yaw_reset_count+1}: Innovation >{np.degrees(HIGH_YAW_INNOVATION_THRESHOLD):.0f}° for {duration:.1f}s")
                        print(f"       Before: yaw_state={np.degrees(yaw_state):.1f}°, yaw_mag={np.degrees(yaw_mag):.1f}°")
                        
                        # Extract current quaternion
                        q_current = kf.x[6:10, 0].copy()
                        
                        # Replace yaw while keeping roll/pitch
                        q_xyzw = np.array([q_current[1], q_current[2], q_current[3], q_current[0]])
                        euler = R_scipy.from_quat(q_xyzw).as_euler('ZYX')  # [yaw, pitch, roll]
                        euler[0] = yaw_mag  # Replace yaw with magnetometer
                        
                        # Reconstruct quaternion
                        q_new_xyzw = R_scipy.from_euler('ZYX', euler).as_quat()
                        q_new = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]])
                        
                        # Update state
                        kf.x[6:10, 0] = q_new
                        
                        # Inflate yaw uncertainty to reflect reset
                        kf.P[8, 8] = np.radians(30.0)**2  # 30° uncertainty after reset
                        
                        yaw_reset_count += 1
                        high_yaw_innovation_start_time = None
                        
                        print(f"       After: yaw_state={np.degrees(quaternion_to_yaw(q_new)):.1f}°")
                        
                        # Skip normal MAG update this cycle
                        continue
            else:
                # Innovation back to normal - reset timer
                if high_yaw_innovation_start_time is not None:
                    duration = t - high_yaw_innovation_start_time
                    print(f"[MAG] High innovation ended after {duration:.1f}s (no reset needed)")
                high_yaw_innovation_start_time = None
            
            # ==========================================
            # INITIAL CONVERGENCE PERIOD
            # ==========================================
            time_since_start = t - t0  # t0 is first IMU timestamp
            in_convergence_period = time_since_start < MAG_INITIAL_CONVERGENCE_WINDOW
            
            # CRITICAL FIX: If we have PPK ground truth initial yaw, SKIP mag updates
            # during convergence period to preserve the accurate PPK yaw.
            # PPK yaw is typically accurate to ~1°, while magnetometer can be off by 30-40°.
            # After convergence period, mag can help correct accumulated drift.
            if in_convergence_period and has_ppk_initial_yaw:
                if mag_update_count == 0:
                    print(f"[MAG] Skipping mag updates during convergence - using PPK initial yaw")
                mag_reject_count += 1
                continue
            
            # Get current yaw uncertainty from covariance
            theta_cov_idx = 8  # δθ_z in error state
            if kf.P.shape[0] > theta_cov_idx:
                yaw_sigma_from_P = np.sqrt(kf.P[theta_cov_idx, theta_cov_idx])
            else:
                yaw_sigma_from_P = SIGMA_MAG_YAW
            
            # =============================================================
            # FIXED: Use CONSTANT small sigma for high Kalman gain!
            # Previous bug: sigma was scaled with innovation, causing low K
            # when we need high K most (large innovation = large error)
            # =============================================================
            # Measurement noise R should reflect magnetometer uncertainty,
            # NOT the size of the innovation!
            sigma_yaw_adaptive = SIGMA_MAG_YAW  # FIXED: Always use base uncertainty
            
            # REMOVED quality scaling (was dividing by quality which inflated uncertainty)
            # Quality is already used for gating above, don't double-penalize
            sigma_yaw_scaled = sigma_yaw_adaptive
            
            # Innovation gating: ADAPTIVE threshold
            # If yaw uncertainty is high, use higher threshold to allow correction
            # This prevents rejecting mag updates when VIO is drifting
            if in_convergence_period:
                innovation_threshold = np.radians(180.0)  # Accept ANY yaw difference
            else:
                # ADAPTIVE: Use MAX of (3σ_R, 3σ_P, 179°)
                # FIX 2025-12-01b: Increased min_threshold from 150° to 179°
                # Problem: yaw drift >150° was being rejected after ~20 updates
                # Root cause: gyro bias causes ~190°/s drift, exceeding 150° threshold quickly
                # Solution: Accept almost any innovation - magnetometer is ABSOLUTE reference!
                # Note: 180° would create ambiguity (±π wrap), so use 179° max
                base_threshold = 3.0 * sigma_yaw_scaled
                state_based_threshold = 3.0 * yaw_sigma_from_P
                min_threshold = np.radians(179.0)  # FIX: Accept up to 179° innovation (was 150°)
                innovation_threshold = max(base_threshold, state_based_threshold, min_threshold)
            
            if abs(yaw_innov) > innovation_threshold:
                mag_reject_count += 1
                if i % 100 == 0:
                    print(f"[MAG] t={t:.2f}: Rejected (innovation {np.degrees(yaw_innov):.1f}° > threshold {np.degrees(innovation_threshold):.1f}°, σ_P={np.degrees(yaw_sigma_from_P):.1f}°)")
                continue
            
            # Log accepted magnetometer update
            convergence_marker = " [CONVERGING]" if in_convergence_period else ""
            if i % 100 == 0 or mag_update_count < 20 or in_convergence_period:
                print(f"[MAG] t={t:.2f}: UPDATE #{mag_update_count+1}{convergence_marker} - yaw_state={np.degrees(yaw_state):.1f}° → yaw_mag={np.degrees(yaw_mag):.1f}° (Δ={np.degrees(yaw_innov):.1f}°, quality={quality:.3f}, σ={np.degrees(sigma_yaw_scaled):.1f}°)")
            
            # ESKF: Magnetometer Jacobian w.r.t. error state
            # For yaw-only measurement: ∂yaw/∂δθ ≈ [0, 0, 1] (only Z-axis rotation affects yaw)
            num_clones = (kf.x.shape[0] - 16) // 7  # FIX: calculate from nominal state
            err_dim = 15 + 6 * num_clones
            
            def h_mag_fun(x):
                h_yaw = np.zeros((1, err_dim), dtype=float)
                # Yaw measurement depends only on rotation error around Z-axis
                h_yaw[0, 8] = 1.0  # δθ_z component (index 6+2=8 in error state)
                return h_yaw
            
            def hx_mag_fun(x):
                # x is nominal state, extract quaternion and compute yaw
                q_x = x[6:10, 0]
                yaw_x = quaternion_to_yaw(q_x)
                return np.array([[yaw_x]])
            
            # Measurement covariance
            r_yaw = np.array([[sigma_yaw_scaled**2]])
            
            # Apply EKF update with angle wrapping
            try:
                # Use residual function that properly handles angle wrapping
                def angle_residual(a, b):
                    res = a - b
                    return np.arctan2(np.sin(res), np.cos(res))
                
                # Compute innovation and Mahalanobis distance before update
                mag_h_matrix = h_mag_fun(kf.x)
                mag_s_mat = mag_h_matrix @ kf.P @ mag_h_matrix.T + r_yaw
                mag_innov_vec = np.array([[yaw_innov]])  # Already computed with wrapping
                try:
                    mag_m2_test = _mahalanobis2(mag_innov_vec, mag_s_mat)
                except Exception:
                    mag_m2_test = 0.0
                
                # Compute Kalman gain before update for logging
                P_yaw = kf.P[theta_cov_idx, theta_cov_idx]  # θ_z variance
                S_yaw = P_yaw + sigma_yaw_scaled**2  # Innovation covariance
                K_yaw = P_yaw / S_yaw  # Kalman gain for yaw
                
                # =====================================================
                # CRITICAL: Enforce minimum K to prevent yaw drift!
                # Without gyro bias estimation, gyro drift (~4°/s from
                # helicopter vibration) requires continuous mag correction.
                # 
                # FIX 2025-12-02b: SENSOR-QUALITY BASED K_MIN (Goal 3)
                # Replace time-based K_MIN with sensor quality metrics
                # - High vibration → lower K_MIN (less trust in MAG during shake)
                # - Low MAG quality → lower K_MIN
                # - Large recent innovations → higher K_MIN (need correction)
                # =====================================================
                
                # Update quality history
                mag_quality_history.append(quality)
                if len(mag_quality_history) > MAG_QUALITY_BUFFER_SIZE:
                    mag_quality_history.pop(0)
                
                innovation_history.append(abs(yaw_innov))
                if len(innovation_history) > INNOVATION_BUFFER_SIZE:
                    innovation_history.pop(0)
                
                # Compute quality metrics
                avg_quality = np.mean(mag_quality_history) if mag_quality_history else quality
                avg_innovation = np.mean(innovation_history) if innovation_history else abs(yaw_innov)
                
                # Base K_MIN from time (still useful for initial stabilization)
                if time_elapsed < 15.0:  # PHASE_SPINUP_END
                    current_flight_phase = 0
                    K_MIN_BASE = 0.40
                    MAX_YAW_CORRECTION = np.radians(30.0)
                else:
                    current_flight_phase = 1  # Normal flight
                    K_MIN_BASE = 0.20
                    MAX_YAW_CORRECTION = np.radians(15.0)
                
                # =====================================================
                # SENSOR-QUALITY MODULATION OF K_MIN
                # =====================================================
                K_MIN = K_MIN_BASE
                
                # 1. MAG Quality Factor: higher quality → higher K_MIN
                # quality ranges 0-1, avg_quality ~0.5-0.9 typically
                quality_factor = min(1.0, avg_quality / 0.7)  # Normalize to 0.7 as "good"
                
                # 2. Vibration Factor: high vibration → lower K_MIN
                # During vibration, MAG readings are less reliable
                if is_high_vibration:
                    vibration_factor = 0.5  # Halve K_MIN during high vibration
                else:
                    vibration_factor = 1.0
                
                # 3. Innovation Factor: large innovations → higher K_MIN (need correction)
                # If yaw is drifting fast, we need aggressive correction
                if avg_innovation > np.radians(30.0):  # >30° average innovation
                    innovation_factor = 1.5  # Boost K_MIN by 50%
                elif avg_innovation > np.radians(15.0):  # >15° average innovation
                    innovation_factor = 1.2  # Boost K_MIN by 20%
                else:
                    innovation_factor = 1.0
                
                # Combine factors
                K_MIN = K_MIN_BASE * quality_factor * vibration_factor * innovation_factor
                K_MIN = np.clip(K_MIN, 0.10, 0.60)  # Clamp to reasonable range
                
                if K_yaw < K_MIN:
                    # Inflate P to achieve target K
                    # K = P / (P + R) → P_new = K_min * R / (1 - K_min)
                    if K_MIN > 0.01:  # Avoid division by near-zero
                        P_yaw_min = K_MIN * sigma_yaw_scaled**2 / (1.0 - K_MIN)
                        if kf.P[theta_cov_idx, theta_cov_idx] < P_yaw_min:
                            kf.P[theta_cov_idx, theta_cov_idx] = P_yaw_min
                            P_yaw = P_yaw_min
                            K_yaw = K_MIN
                
                # Expected yaw correction = K * innovation
                expected_correction = K_yaw * yaw_innov
                
                # =====================================================
                # LIMIT MAXIMUM YAW CORRECTION
                # Large corrections can destabilize attitude matrix and
                # cause velocity projection errors → altitude drift
                # MAX_YAW_CORRECTION is set by flight phase above
                # =====================================================
                if abs(expected_correction) > MAX_YAW_CORRECTION:
                    # Inflate measurement noise to limit correction magnitude
                    # K_new * innov = MAX_CORRECTION → K_new = MAX_CORRECTION / innov
                    # K_new = P / (P + R_new) → R_new = P * (1/K_new - 1) = P * (innov/MAX - 1)
                    K_target = MAX_YAW_CORRECTION / abs(yaw_innov)
                    R_inflated = P_yaw * (1.0 / K_target - 1.0)
                    r_yaw = np.array([[max(R_inflated, sigma_yaw_scaled**2)]])  # Don't reduce below base
                    new_correction = K_target * yaw_innov
                    if i % 100 == 0:
                        print(f"[MAG] LIMITING: correction {np.degrees(expected_correction):.1f}° → {np.degrees(new_correction):.1f}°")
                
                # =====================================================
                # CRITICAL: Decouple yaw from roll/pitch before mag update!
                # Cross-covariance between yaw and roll/pitch causes
                # altitude drift when yaw is corrected.
                # =====================================================
                # Indices: roll=6, pitch=7, yaw=8 in error state
                # Zero out cross-covariance between yaw and roll/pitch
                kf.P[8, 6] = 0.0  # Cov(yaw, roll)
                kf.P[6, 8] = 0.0
                kf.P[8, 7] = 0.0  # Cov(yaw, pitch)
                kf.P[7, 8] = 0.0
                # Also decouple yaw from velocity (altitude related)
                kf.P[8, 5] = 0.0  # Cov(yaw, v_z)
                kf.P[5, 8] = 0.0
                
                kf.update(
                    z=np.array([[yaw_mag]]),
                    HJacobian=h_mag_fun,
                    Hx=hx_mag_fun,
                    R=r_yaw,
                    residual=angle_residual
                )
                mag_update_count += 1
                
                # =====================================================
                # NOTE: P_yaw floor was causing instability!
                # Floor K=0.25 made mag over-correct with noisy measurements
                # Instead, use ADAPTIVE floor based on motion state:
                # - Low floor (3°) when stationary → trust mag more
                # - No floor when moving → trust IMU integration more
                # =====================================================
                # DISABLED: P_yaw floor was causing altitude to diverge (532m error)
                # MIN_P_YAW = np.radians(5.0)**2  # 5° minimum yaw uncertainty
                # if kf.P[theta_cov_idx, theta_cov_idx] < MIN_P_YAW:
                #     kf.P[theta_cov_idx, theta_cov_idx] = MIN_P_YAW
                
                # =====================================================
                # GYRO BIAS Z ESTIMATION FROM MAGNETOMETER
                # DISABLED 2025-12-01: Causing instability!
                # Problem: Large yaw innovations (e.g., ±180° wrap-around)
                # divided by small dt creates unrealistic drift estimates
                # that corrupt the gyro bias state.
                # 
                # The Kalman filter already handles bias estimation through
                # the P(yaw,bg) cross-covariance. Explicit update is not
                # needed and causes harm.
                # =====================================================
                # DISABLED: See above
                # if mag_update_count > 1 and hasattr(kf, 'last_mag_time'):
                #     dt_since_mag = t - kf.last_mag_time
                #     if dt_since_mag > 0.1:  # At least 100ms between updates
                #         bg_z_estimate = yaw_innov / dt_since_mag
                #         ...
                
                kf.last_mag_time = t  # Save time for potential future use
                
                # Log Kalman gain periodically (every 50 updates)
                if mag_update_count % 50 == 1 or mag_update_count <= 10:
                    phase_name = "SPINUP" if current_flight_phase == 0 else "NORMAL"
                    vib_marker = " [VIB]" if is_high_vibration else ""
                    print(f"[MAG] K={K_yaw:.3f} (K_MIN={K_MIN:.2f}, qf={quality_factor:.2f}, vf={vibration_factor:.1f}, if={innovation_factor:.1f}) | P_yaw={np.degrees(np.sqrt(P_yaw)):.1f}° | correction={np.degrees(expected_correction):.1f}° | t={time_elapsed:.1f}s [{phase_name}]{vib_marker}")
                
                # Log residual for debugging
                if save_debug_data and residual_csv:
                    log_measurement_update(
                        residual_csv, t, vio_frame, 'MAG',
                        innovation=mag_innov_vec,
                        mahalanobis_dist=np.sqrt(mag_m2_test),
                        chi2_threshold=innovation_threshold,
                        accepted=True,
                        S_matrix=mag_s_mat,
                        P_prior=kf.P_prior
                    )
            except Exception as e:
                mag_reject_count += 1

        # -------------- Optional VIO (velocity) --------------
        # TODO: Check VIO/VO update logic, units, and sign conventions here
        # DEBUG: Print VIO/VO measurement, innovation, and update status
        if VERBOSE_DEBUG:
            print(f"[DEBUG][VIO] Entry check: vio_fe={'None' if vio_fe is None else 'OK'}, img_idx={img_idx}/{len(imgs)}")
        vo_dx=vo_dy=vo_dz=vo_r=vo_p=vo_y=np.nan
        vel_vx=vel_vy=vel_vz=0.0  # Initialize velocity outputs for CSV logging
        used_vo = False
        if vio_fe is not None:
            # trigger VIO when current IMU t passes image timestamp
            while img_idx < len(imgs) and imgs[img_idx].t <= t:
                if VERBOSE_DEBUG:
                    print(f"[DEBUG][VIO] Processing image idx={img_idx}/{len(imgs)} at t={t:.3f}, img_t={imgs[img_idx].t:.3f}, path={imgs[img_idx].path}")
                # load and process this image
                img = cv2.imread(imgs[img_idx].path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"[DEBUG][VIO] FAILED to read image: {imgs[img_idx].path}")
                    img_idx += 1
                    continue
                if (img.shape[1], img.shape[0]) != tuple(downscale_size):
                    img = cv2.resize(img, downscale_size, interpolation=cv2.INTER_AREA)
                
                # ========================================================
                # LOOP CLOSURE: Check and apply yaw correction
                # ========================================================
                # DISABLED: Loop closure is causing false positives on this dataset
                # (no true loop in trajectory - start/end are 694m apart)
                # False loop detections apply wrong yaw corrections → position error
                # 
                # TODO: Re-enable with stricter validation when testing on loop datasets
                # ========================================================
                if False and loop_closure_enabled and loop_detector is not None:
                    # Get current estimated state
                    current_pos = kf.x[0:3, 0].flatten()  # [x, y, z]
                    current_yaw = quaternion_to_yaw(kf.x[6:10, 0])
                    
                    # 1. Check if we should add this as a keyframe
                    if loop_detector.should_add_keyframe(current_pos, current_yaw, img_idx):
                        loop_detector.add_keyframe(img_idx, current_pos, current_yaw, img)
                    
                    # 2. Check for loop closure (only after we have some keyframes)
                    if len(loop_detector.keyframes) >= 5:
                        # Only check every 10 frames to reduce computation
                        if img_idx % 10 == 0:
                            loop_result = loop_detector.check_loop_closure(
                                img_idx, current_pos, current_yaw, img, vio_fe.K
                            )
                            
                            if loop_result is not None:
                                yaw_correction, num_inliers, loop_kf_idx = loop_result
                                
                                # Apply correction if significant (> 5 degrees)
                                if abs(yaw_correction) > np.radians(5.0):
                                    # Use lower sigma (= more trust) for high inlier count
                                    sigma_loop = 0.05 if num_inliers > 50 else 0.1
                                    loop_detector.apply_yaw_correction(kf, yaw_correction, sigma_loop)
                
                ok, ninl, r_vo_mat, t_unit, dt_img = vio_fe.step(img, imgs[img_idx].t)
                if VERBOSE_DEBUG:
                    print(f"[DEBUG][VIO] vio_fe.step() returned: ok={ok}, ninl={ninl}, r_vo_mat={'None' if r_vo_mat is None else 'OK'}, t_unit={'None' if t_unit is None else 'OK'}, dt_img={dt_img:.4f}")
                
                # CRITICAL: Detect fast rotation and skip VIO updates during aggressive maneuvers
                # Fast rotation breaks feature tracking and essential matrix decomposition
                # Use most recent IMU angular velocity (rec is last IMU measurement processed)
                angular_vel = rec.ang  # Angular velocity from IMU [rad/s]
                rotation_rate_deg_s = np.linalg.norm(angular_vel) * 180.0 / np.pi
                is_fast_rotation = rotation_rate_deg_s > 30.0  # Threshold: 30 deg/s
                
                if is_fast_rotation:
                    print(f"[DEBUG][VIO] SKIPPING VIO update due to fast rotation: {rotation_rate_deg_s:.1f} deg/s")
                    img_idx += 1
                    continue
                
                # CRITICAL: Check for sufficient parallax/baseline before VIO update
                # During hovering, insufficient translation → no parallax → poor multi-view constraints
                # Strategy: Track features but don't update EKF until sufficient motion
                # This prevents degenerate cases for nadir cameras over flat terrain
                
                # FIX: Use feature tracker's mean_parallax (computed from all tracked features)
                # This is more robust than recomputing from last_matches (which can be None)
                avg_flow_px = getattr(vio_fe, 'mean_parallax', 0.0)
                
                # Fallback: If mean_parallax not available, compute from last_matches
                if avg_flow_px == 0.0 and vio_fe.last_matches is not None:
                    focal_px = KB_PARAMS['mu']
                    pts_prev, pts_cur = vio_fe.last_matches
                    if len(pts_prev) > 0 and len(pts_cur) > 0:
                        pts_prev_px = pts_prev * focal_px + np.array([[vio_fe.img_w/2, vio_fe.img_h/2]])
                        pts_cur_px = pts_cur * focal_px + np.array([[vio_fe.img_w/2, vio_fe.img_h/2]])
                        flows = pts_cur_px - pts_prev_px
                        avg_flow_px = float(np.median(np.linalg.norm(flows, axis=1)))
                
                # Parallax threshold: Use global MIN_PARALLAX_PX (2.0 for slow motion)
                # Below this → hovering/stationary/vibrations → skip VIO velocity updates
                # Rationale: During hovering, rotor vibrations + tracking drift cause false flow
                # Only trust VIO when sufficient intentional motion exists
                is_insufficient_parallax = avg_flow_px < MIN_PARALLAX_PX
                
                print(f"[DEBUG][VIO] Parallax check: avg_flow={avg_flow_px:.2f}px, threshold={MIN_PARALLAX_PX}px, insufficient={is_insufficient_parallax}") if VERBOSE_DEBUG else None
                
                # ================================================================
                # CRITICAL FIX: Apply Preintegration at EVERY Camera Frame
                # ================================================================
                # Previously: Preintegration was only applied when cloning camera
                # Problem: Cloning happens every 1-5 seconds → IMU accumulates for 16s → massive drift
                # Solution: Apply preintegration at EVERY camera frame (~50ms) as designed by Forster et al.
                # 
                # IMPORTANT: Apply BEFORE checking parallax!
                # - IMU physics continues regardless of visual features
                # - Prevents accumulation of large deltas when camera has low parallax
                # - Matches Forster et al. design: propagate state at every measurement
                
                if use_preintegration and ongoing_preint is not None:
                    # Apply accumulated preintegration since last camera frame
                    dt_total = ongoing_preint.dt_sum
                    
                    # Get deltas before correction for debugging
                    delta_R_raw, delta_v_raw, delta_p_raw = ongoing_preint.get_deltas()
                    delta_angle_deg = np.linalg.norm(R_scipy.from_matrix(delta_R_raw).as_rotvec()) * 180 / np.pi
                    
                    print(f"[PREINT] Applying preintegration: Δt={dt_total:.3f}s, ΔR={delta_angle_deg:.2f}°, Δv={np.linalg.norm(delta_v_raw):.3f}m/s, Δp={np.linalg.norm(delta_p_raw):.3f}m")
                    
                    # Warn if accumulation is too long (should be ~50ms for 20Hz camera)
                    if dt_total > 0.2:  # More than 200ms is suspicious
                        print(f"[PREINT_DEBUG] WARNING: LARGE ACCUMULATION! dt={dt_total:.3f}s (expected ~0.05s)")
                    
                    # Current state
                    p_i = kf.x[0:3, 0].reshape(3,)
                    v_i = kf.x[3:6, 0].reshape(3,)
                    q_i = kf.x[6:10, 0].reshape(4,)  # [w,x,y,z]
                    bg = kf.x[10:13, 0].reshape(3,)
                    ba = kf.x[13:16, 0].reshape(3,)
                    
                    # Get bias-corrected preintegrated deltas
                    delta_R, delta_v, delta_p = ongoing_preint.get_deltas_corrected(bg, ba)
                    
                    # CRITICAL FIX: Quaternion stores R_BW (Body-to-World) NOT R_WB!
                    # Legacy code (line 3821): R_body_to_world = R_scipy.from_quat(q_xyzw).as_matrix()
                    # This is consistent with IMU convention: body frame rotates in world frame
                    q_i_xyzw = np.array([q_i[1], q_i[2], q_i[3], q_i[0]])
                    R_BW = R_scipy.from_quat(q_i_xyzw).as_matrix()  # R_BW (Body-to-World)
                    
                    # Rotation update (Forster et al. TRO 2017, Eq. 11):
                    # R_{k+1} = R_k @ delta_R (body-frame rotation composed on right)
                    # Since we store R_BW (body-to-world), update is:
                    # R_BW_new = R_BW_old @ delta_R
                    R_BW_new = R_BW @ delta_R
                    q_i_new_xyzw = R_scipy.from_matrix(R_BW_new).as_quat()
                    q_i_new = np.array([q_i_new_xyzw[3], q_i_new_xyzw[0], q_i_new_xyzw[1], q_i_new_xyzw[2]])
                    
                    # Position and velocity updates (Forster et al. TRO 2017, Eq. 11):
                    # delta_v and delta_p are in body frame at t=0, rotate to world frame
                    # IMPORTANT: Preintegration already compensates gravity!
                    # 
                    # v_W_{k+1} = v_W_k + R_BW_k @ delta_v
                    # p_W_{k+1} = p_W_k + v_W_k * dt + R_BW_k @ delta_p
                    # 
                    # Note: Use R_BW_k (OLD rotation) to transform deltas integrated in old body frame
                    v_i_new = v_i + R_BW @ delta_v
                    p_i_new = p_i + v_i * dt_total + R_BW @ delta_p
                    
                    # Debug: show state BEFORE update
                    print(f"[PREINT_DEBUG] BEFORE: p={p_i}, v={v_i}, q={q_i}")
                    print(f"[PREINT_DEBUG] DELTAS: ΔR={delta_R.diagonal()}, Δv={delta_v}, Δp={delta_p}")
                    print(f"[PREINT_DEBUG] AFTER:  p={p_i_new}, v={v_i_new}, q={q_i_new}")
                    
                    # Write back to state (CRITICAL: must reshape to column vectors)
                    kf.x[0:3, 0] = p_i_new.reshape(3,)
                    kf.x[3:6, 0] = v_i_new.reshape(3,)
                    kf.x[6:10, 0] = q_i_new.reshape(4,)
                    
                    # Propagate error-state covariance using preintegration Jacobians
                    preint_cov = ongoing_preint.get_covariance()
                    J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba = ongoing_preint.get_jacobians()
                    
                    # Build state transition matrix for error-state (15+6M dimensions)
                    num_clones = (kf.x.shape[0] - 16) // 7
                    
                    # Core state block (15x15): [δp, δv, δθ, δbg, δba]
                    Phi_core = np.eye(15, dtype=float)
                    Phi_core[0:3, 3:6] = np.eye(3) * dt_total  # δp ← δv
                    Phi_core[0:3, 6:9] = -R_BW @ skew_symmetric(delta_p)  # δp ← δθ (negative: perturb rotation)
                    Phi_core[0:3, 9:12] = R_BW @ J_p_bg  # δp ← δbg
                    Phi_core[0:3, 12:15] = R_BW @ J_p_ba  # δp ← δba
                    Phi_core[3:6, 6:9] = -R_BW @ skew_symmetric(delta_v)  # δv ← δθ (negative: perturb rotation)
                    Phi_core[3:6, 9:12] = R_BW @ J_v_bg  # δv ← δbg
                    Phi_core[3:6, 12:15] = R_BW @ J_v_ba  # δv ← δba
                    Phi_core[6:9, 9:12] = -J_R_bg  # δθ ← δbg
                    
                    # Process noise (rotate preintegration covariance to world frame)
                    Q_core = np.zeros((15, 15), dtype=float)
                    Q_core[0:3, 0:3] = R_BW @ preint_cov[6:9, 6:9] @ R_BW.T  # δp noise
                    Q_core[0:3, 3:6] = R_BW @ preint_cov[6:9, 3:6] @ R_BW.T
                    Q_core[0:3, 6:9] = R_BW @ preint_cov[6:9, 0:3] @ R_BW.T
                    Q_core[3:6, 0:3] = R_BW @ preint_cov[3:6, 6:9] @ R_BW.T
                    Q_core[3:6, 3:6] = R_BW @ preint_cov[3:6, 3:6] @ R_BW.T  # δv noise
                    Q_core[3:6, 6:9] = R_BW @ preint_cov[3:6, 0:3] @ R_BW.T
                    Q_core[6:9, 0:3] = R_BW @ preint_cov[0:3, 6:9] @ R_BW.T
                    Q_core[6:9, 3:6] = R_BW @ preint_cov[0:3, 3:6] @ R_BW.T
                    Q_core[6:9, 6:9] = R_BW @ preint_cov[0:3, 0:3] @ R_BW.T  # δθ noise
                    
                    # Add bias random walk
                    Q_core[9:12, 9:12] = np.eye(3) * (IMU_PARAMS["gyr_w"]**2 * dt_total)
                    Q_core[12:15, 12:15] = np.eye(3) * (IMU_PARAMS["acc_w"]**2 * dt_total)
                    
                    # Propagate covariance (function will expand to include clones)
                    kf.P = propagate_error_state_covariance(kf.P, Phi_core, Q_core, num_clones)
                    
                    # Debug: show actual state change
                    delta_pos = np.linalg.norm(p_i_new - p_i)
                    delta_vel = np.linalg.norm(v_i_new - v_i)
                    print(f"[PREINT] State updated: Δpos={delta_pos:.4f}m, Δvel={delta_vel:.4f}m/s, Δyaw={(q_i_new[0]-q_i[0])*180/np.pi:.2f}°")
                    
                    # Reset preintegration buffer for next camera frame
                    ongoing_preint.reset(bg=bg, ba=ba)
                    preint_start_time = t
                    print(f"[PREINT] Buffer reset for next integration period")
                
                # ================================================================
                # Check Parallax for VIO Velocity Update
                # ================================================================
                # After preintegration is applied, check if parallax is sufficient for VIO velocity
                if is_insufficient_parallax:
                    print(f"[DEBUG][VIO] SKIPPING VIO velocity update: insufficient parallax (flow={avg_flow_px:.2f}px < {MIN_PARALLAX_PX}px)")
                    # Continue tracking features for MSCKF but don't update velocity
                    # Preintegration was already applied above!
                    img_idx += 1
                    continue
                else:
                    print(f"[DEBUG][VIO] USING VIO velocity: sufficient parallax (flow={avg_flow_px:.2f}px >= {MIN_PARALLAX_PX}px)")
                
                # ================================================================
                # Camera Cloning Decision (Separate from Preintegration)
                # ================================================================
                # --- ARCHITECTURE FIX: Separate camera cloning from VO velocity update ---
                # OpenVINS-style: Clone camera WHENEVER valid features exist (not just on VO success)
                # This ensures MSCKF has sufficient camera poses for multi-view constraints
                
                # === IMPROVED: Motion-based Camera Cloning ===
                # Only clone when sufficient motion exists (not every frame)
                # Strategy: Clone when parallax exceeds threshold (indicates sufficient baseline)
                # This prevents state vector bloat while ensuring MSCKF has enough camera poses
                
                # CRITICAL: Use same parallax threshold as VIO velocity (MIN_PARALLAX_PX)
                # But apply ADDITIONAL motion check to prevent cloning during hover/vibrations
                CLONE_PARALLAX_THRESHOLD = MIN_PARALLAX_PX * 2.0  # 2x VIO threshold for cloning (4px)
                should_clone_camera = (avg_flow_px >= CLONE_PARALLAX_THRESHOLD) and not is_fast_rotation
                
                print(f"[DEBUG][CLONE] Cloning decision: parallax={avg_flow_px:.2f}px >= {CLONE_PARALLAX_THRESHOLD:.2f}px? {should_clone_camera}")
                
                if not should_clone_camera:
                    # Track features but don't clone camera pose
                    print(f"[DEBUG][CLONE] SKIPPING camera cloning (insufficient motion)")
                    # IMPORTANT: Still allow VIO velocity update if parallax is sufficient!
                    # Don't continue here - proceed to VO velocity estimation below
                
                # Legacy motion check (keep for backward compatibility with MSCKF baseline check)
                distance_moved = 0.0
                rotation_angle = 0.0
                
                if should_clone_camera and vio_fe.last_matches is not None and len(vio_fe.last_matches[0]) >= 10:
                    # Check motion since last clone
                    if len(cam_states) > 0:
                        # Get current and last camera positions
                        p_i_now = kf.x[0:3, 0].reshape(3,)
                        q_i_now = kf.x[6:10, 0].reshape(4,)
                        
                        # Last cloned IMU state (now stores IMU pose, not camera pose)
                        last_cs = cam_states[-1]
                        p_imu_last = kf.x[last_cs['p_idx']:last_cs['p_idx']+3, 0].reshape(3,)
                        q_imu_last = kf.x[last_cs['q_idx']:last_cs['q_idx']+4, 0].reshape(4,)
                        
                        # Distance moved (IMU frame)
                        distance_moved = float(np.linalg.norm(p_i_now - p_imu_last))
                        
                        # Rotation angle (IMU frame)
                        q_now_xyzw = np.array([q_i_now[1], q_i_now[2], q_i_now[3], q_i_now[0]])
                        q_last_xyzw = np.array([q_imu_last[1], q_imu_last[2], q_imu_last[3], q_imu_last[0]])
                        
                        R_now = R_scipy.from_quat(q_now_xyzw).as_matrix()
                        R_last = R_scipy.from_quat(q_last_xyzw).as_matrix()
                        R_delta = R_now @ R_last.T
                        
                        # Convert to angle-axis
                        angle = float(np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1, 1)))
                        rotation_angle = np.degrees(angle)
                        
                        # Clone if sufficient motion OR time-based fallback
                        # CRITICAL: Require minimum baseline for triangulation quality
                        # BUT also enforce maximum time limit to prevent preintegration explosion
                        frames_since_clone = vio_fe.frame_idx - last_cs['frame']
                        time_since_clone = t - last_cs.get('timestamp', 0.0)  # Time in seconds (0 if missing)
                        has_minimum_motion = distance_moved >= 0.05  # 5cm minimum for motion-based clone
                        
                        should_clone_camera = (
                            (has_minimum_motion and (
                                distance_moved >= 0.2 or           # 20cm translation
                                rotation_angle >= 2.0 or           # 2 degree rotation  
                                frames_since_clone >= 5            # Every 5 frames if moving
                            )) or
                            time_since_clone >= 1.0                # CRITICAL: Force clone every 1 second
                                                                   # to prevent preintegration explosion
                        )
                        
                        if should_clone_camera:
                            reason = []
                            if distance_moved >= 0.2:
                                reason.append(f"dist={distance_moved:.2f}m")
                            if rotation_angle >= 2.0:
                                reason.append(f"rot={rotation_angle:.1f}°")
                            if frames_since_clone >= 5 and has_minimum_motion:
                                reason.append(f"frames={frames_since_clone}")
                            if time_since_clone >= 1.0:
                                reason.append(f"time={time_since_clone:.1f}s")
                            print(f"[CLONE] Motion-based clone trigger: {', '.join(reason)}")
                    else:
                        # First clone: always create
                        should_clone_camera = True
                        print(f"[CLONE] Creating first camera clone")
                
                if should_clone_camera:
                    # NOTE: Preintegration is now applied at EVERY camera frame (see above)
                    # This block only handles camera cloning for MSCKF
                    
                    # --- Time Synchronization: Propagate to camera timestamp ---
                    camera_timestamp = imgs[img_idx].t
                    current_state_time = t  # Current IMU time
                    
                    if abs(camera_timestamp - current_state_time) > 0.001:  # >1ms difference
                        print(f"[TIME_SYNC] Propagating from t={current_state_time:.4f}s to camera_t={camera_timestamp:.4f}s")
                        
                        # Propagate state to camera timestamp using preintegration (or legacy)
                        success, preint_data = propagate_to_timestamp(
                            kf=kf,
                            target_time=camera_timestamp,
                            imu_buffer=imu,
                            current_time=current_state_time,
                            estimate_imu_bias=estimate_imu_bias,
                            use_preintegration=use_preintegration  # Controlled by CLI flag
                        )
                        
                        if not success:
                            print(f"[WARNING] Failed to propagate to camera timestamp, using current state")
                            preint_data = None
                    else:
                        preint_data = None
                    
                    # --- Augment EKF state with current IMU pose (OpenVINS-style) ---
                    # CRITICAL: Clone IMU pose, NOT camera pose!
                    # Camera pose is computed via extrinsics during measurement model
                    try:
                        # IMU pose (world <- body) AT CAMERA TIMESTAMP
                        p_imu = kf.x[0:3, 0].reshape(3,)
                        q_imu = kf.x[6:10, 0].reshape(4,)  # [w,x,y,z]

                        # Augment state with IMU pose (OpenVINS convention)
                        # This preserves observability: we only observe IMU motion, not camera-IMU transform
                        start_idx = augment_state_with_camera(kf, q_imu, p_imu, 
                                                             cam_states, cam_observations)
                        
                        # === FEJ: Store first-estimate linearization point ===
                        # OpenVINS-style: Save IMU pose at time of cloning for consistent Jacobian
                        q_fej = q_imu.copy()  # First-estimate quaternion [w,x,y,z] (IMU orientation)
                        p_fej = p_imu.copy()  # First-estimate position [x,y,z] (IMU position)
                        
                        # NEW: Store bias linearization points for FEJ consistency
                        bg_fej = kf.x[10:13, 0].copy()  # Gyro bias at clone time
                        ba_fej = kf.x[13:16, 0].copy()  # Accel bias at clone time
                        
                        # Store camera state with SYNCHRONIZED timestamp + FEJ
                        clone_idx = len(cam_states)  # Which clone is this (0, 1, 2, ...)
                        err_theta_idx = 15 + 6 * clone_idx      # Error rotation index
                        err_p_idx = 15 + 6 * clone_idx + 3      # Error position index
                        
                        cam_states.append({
                            'start_idx': start_idx,
                            'q_idx': start_idx,
                            'p_idx': start_idx+4,
                            'err_q_idx': err_theta_idx,
                            'err_p_idx': err_p_idx,
                            't': camera_timestamp,
                            'timestamp': t,  # IMU timestamp for time-based clone decision
                            'frame': vio_fe.frame_idx,
                            # FEJ linearization points (camera pose)
                            'q_fej': q_fej,  # First-estimate quaternion (for consistent Jacobian)
                            'p_fej': p_fej,  # First-estimate position (for consistent Jacobian)
                            # NEW: FEJ linearization points (IMU biases)
                            'bg_fej': bg_fej,  # Gyro bias at clone time (for preintegration)
                            'ba_fej': ba_fej,  # Accel bias at clone time (for preintegration)
                            # NEW: Preintegration data (for IMU constraints)
                            'preint': preint_data  # IMUPreintegration object (None if not used)
                        })

                        # Record feature observations for this camera pose
                        # === CRITICAL FIX: Ensure feature tracking works ===
                        obs_data = []
                        try:
                            # Get all tracked features at current frame
                            tracks_obs = vio_fe.get_tracks_for_frame(vio_fe.frame_idx)
                            print(f"[CLONE] Frame {vio_fe.frame_idx}: Found {len(tracks_obs)} tracked features")
                            
                            if len(tracks_obs) > 0:
                                for i_feat, (fid, pt) in enumerate(tracks_obs):
                                    try:
                                        pt_array = np.array([[pt[0], pt[1]]], dtype=np.float32)
                                        pt_norm = vio_fe._undistort_pts(pt_array).reshape(2,)
                                        
                                        # DEBUG: Print first 3 features per frame to verify undistortion
                                        if i_feat < 3 and len(cam_observations) < 5:
                                            print(f"[KB-DEBUG] fid={fid} pixel=({pt[0]:.1f},{pt[1]:.1f}) → norm=({pt_norm[0]:.4f},{pt_norm[1]:.4f})")
                                        
                                        track = vio_fe.get_track_by_id(fid)
                                        quality = track[-1].get('quality', 1.0) if track else 1.0
                                        
                                        obs_data.append({
                                            'fid': int(fid),
                                            'pt_pixel': (float(pt[0]), float(pt[1])),
                                            'pt_norm': (float(pt_norm[0]), float(pt_norm[1])),
                                            'quality': float(quality)
                                        })
                                    except Exception as e_obs:
                                        print(f"[CLONE] Warning: Failed to process feature {fid}: {e_obs}")
                                        continue
                                
                                # Store observations even if empty (for tracking purposes)
                                cam_observations.append({
                                    'cam_id': clone_idx,
                                    'frame': vio_fe.frame_idx,
                                    't': imgs[img_idx].t,
                                    'observations': obs_data
                                })
                                
                                print(f"[CLONE] ✓ Created camera clone {clone_idx} with {len(obs_data)} observations at frame {vio_fe.frame_idx}")
                            else:
                                # No features tracked - still create observation entry
                                cam_observations.append({
                                    'cam_id': clone_idx,
                                    'frame': vio_fe.frame_idx,
                                    't': imgs[img_idx].t,
                                    'observations': []
                                })
                                print(f"[CLONE] ⚠ Created camera clone {clone_idx} with 0 observations (no features tracked)")
                                
                        except Exception as e:
                            print(f"[CLONE] ✗ Failed to record observations: {type(e).__name__}: {e}")
                            # Still create empty observation entry to maintain cam_observations alignment
                            cam_observations.append({
                                'cam_id': clone_idx,
                                'frame': vio_fe.frame_idx,
                                't': imgs[img_idx].t,
                                'observations': obs_data  # May be empty
                            })
                        
                        # === DEBUG: Log feature tracking stats ===
                        if save_debug_data and feature_stats_csv:
                            try:
                                num_detected = len(obs_data)
                                num_tracked = num_detected  # All in obs_data are tracked
                                num_inliers_current = vio_fe.last_inliers if hasattr(vio_fe, 'last_inliers') else 0
                                mean_parallax = avg_flow_px if 'avg_flow_px' in locals() else 0.0
                                max_parallax = avg_flow_px if 'avg_flow_px' in locals() else 0.0
                                tracking_ratio = 1.0 if num_detected > 0 else 0.0
                                inlier_ratio = num_inliers_current / max(1, num_detected)
                                
                                with open(feature_stats_csv, "a", newline="") as f:
                                    f.write(f"{vio_fe.frame_idx},{imgs[img_idx].t:.6f},"
                                           f"{num_detected},{num_tracked},{num_inliers_current},"
                                           f"{mean_parallax:.2f},{max_parallax:.2f},"
                                           f"{tracking_ratio:.3f},{inlier_ratio:.3f}\n")
                            except Exception:
                                pass
                        
                        # === DEBUG: Save keyframe image ===
                        if save_keyframe_images and keyframe_dir:
                            try:
                                # Load image
                                img_path = imgs[img_idx].path
                                img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                if img_gray is not None:
                                    # Resize to runtime size
                                    img_resized = cv2.resize(img_gray, downscale_size)
                                    
                                    # Get features and inliers
                                    features = None
                                    inliers = None
                                    if vio_fe.last_matches is not None:
                                        pts_prev, pts_cur = vio_fe.last_matches
                                        # Convert normalized coords to pixel coords
                                        features = pts_cur * focal_px + np.array([[vio_fe.img_w/2, vio_fe.img_h/2]])
                                        # All points in last_matches are inliers from RANSAC
                                        inliers = np.ones(len(features), dtype=bool)
                                    
                                    # Tracking statistics
                                    tracking_stats = {
                                        'num_features': len(obs_data),
                                        'num_inliers': num_inliers_current if 'num_inliers_current' in locals() else 0,
                                        'parallax_px': avg_flow_px if 'avg_flow_px' in locals() else 0.0
                                    }
                                    
                                    # Save annotated image
                                    output_path = os.path.join(keyframe_dir, f"keyframe_{vio_fe.frame_idx:04d}.png")
                                    save_keyframe_image_with_overlay(
                                        img_resized, features, inliers, None,
                                        output_path, vio_fe.frame_idx, tracking_stats
                                    )
                                    print(f"[DEBUG] Saved keyframe image: {output_path}")
                            except Exception as e:
                                print(f"[WARNING] Failed to save keyframe image: {e}")
                    except Exception as e:
                        print(f"[CLONE] ✗ Failed to augment camera state: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
                        pass
                    
                    # === IMPROVED: Adaptive MSCKF Update Trigger ===
                    # Trigger based on multiple conditions (not just fixed frame count)
                    should_update_msckf = False
                    num_mature_features = 0
                    
                    if len(cam_states) >= 3:
                        # Count features with sufficient observations
                        feature_obs_count = {}
                        total_obs_count = 0
                        for obs_set in cam_observations:
                            total_obs_count += len(obs_set['observations'])
                            for obs in obs_set['observations']:
                                fid = obs['fid']
                                feature_obs_count[fid] = feature_obs_count.get(fid, 0) + 1
                        
                        # Count mature features (observed in 2+ camera poses)
                        num_mature_features = sum(1 for count in feature_obs_count.values() if count >= 2)
                        
                        print(f"[MSCKF] Stats: {len(cam_states)} clones, {total_obs_count} total obs, {len(feature_obs_count)} unique features, {num_mature_features} mature (≥2 obs)")
                        
                        # Adaptive trigger conditions
                        frames_since_last_update = vio_fe.frame_idx % 5  # Check every 5 frames
                        window_nearly_full = len(cam_states) >= 4
                        
                        should_update_msckf = (
                            num_mature_features >= 20 or               # Sufficient mature features
                            (window_nearly_full and num_mature_features >= 10) or  # Window full with some features
                            (vio_fe.frame_idx % 5 == 0 and len(cam_states) >= 3)  # Periodic fallback
                        )
                        
                        if should_update_msckf:
                            reason = []
                            if num_mature_features >= 20:
                                reason.append(f"mature_feat={num_mature_features}")
                            if window_nearly_full:
                                reason.append("window_full")
                            if vio_fe.frame_idx % 5 == 0:
                                reason.append("periodic")
                            print(f"[MSCKF] Adaptive trigger: {', '.join(reason)}")
                    
                    if should_update_msckf:
                        try:
                            print(f"[MSCKF] Attempting update at frame {vio_fe.frame_idx}, cam_states={len(cam_states)}, mature_features={num_mature_features}")
                            
                            # === DEBUG: Log MSCKF window state ===
                            if save_debug_data and msckf_window_csv:
                                try:
                                    window_start_time = cam_states[0]['t'] if len(cam_states) > 0 else 0.0
                                    window_duration = (cam_states[-1]['t'] - window_start_time) if len(cam_states) > 1 else 0.0
                                    marginalized_id = -1  # Will be updated if marginalization happens
                                    
                                    # Count tracked features across all observations
                                    all_features = set()
                                    for obs_set in cam_observations:
                                        for obs in obs_set['observations']:
                                            all_features.add(obs['fid'])
                                    
                                    with open(msckf_window_csv, "a", newline="") as f:
                                        f.write(f"{vio_fe.frame_idx},{imgs[img_idx].t:.6f},"
                                               f"{len(cam_states)},{len(all_features)},{num_mature_features},"
                                               f"{window_start_time:.6f},{window_duration:.6f},{marginalized_id}\n")
                                except Exception:
                                    pass
                            
                            num_updates = perform_msckf_updates(
                                vio_fe, cam_observations, cam_states, kf,
                                min_observations=2,
                                max_features=50,  # Increased from 30
                                msckf_dbg_path=msckf_dbg,
                                dem_reader=dem,  # 1.4: Pass DEM for indirect constraint
                                origin_lat=lat0, origin_lon=lon0
                            )
                            if num_updates > 0:
                                print(f"[MSCKF] Successfully updated {num_updates} features")
                                
                                # NEW: Log FEJ consistency after MSCKF update
                                if save_debug_data and fej_consistency_csv:
                                    log_fej_consistency(fej_consistency_csv, t, img_idx, cam_states, kf)
                            else:
                                print(f"[MSCKF] No successful updates (triangulation failed)")
                        except Exception as e:
                            print(f"[MSCKF] Error: {e}")
                            import traceback
                            traceback.print_exc()
                
                # --- VIO Velocity Update (Separate, conditional on parallax) ---
                if ok and r_vo_mat is not None and t_unit is not None:
                    # Check nadir alignment from camera view
                    zn = np.array([0,0,1.0], dtype=float)
                    t_norm = t_unit / (np.linalg.norm(t_unit)+1e-12)
                    alignment_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(t_norm, zn)), -1.0, 1.0))))
                    
                    # Get camera view config (use default 'nadir' if not set)
                    cam_view = getattr(vio_fe, 'camera_view', 'nadir')
                    view_cfg = CAMERA_VIEW_CONFIGS.get(cam_view, CAMERA_VIEW_CONFIGS['nadir'])
                    
                    # FIXED LOGIC: For nadir camera, use_vz_only should be config-driven only
                    # Original logic was inverted: (alignment < threshold) meant "nearly vertical motion"
                    # But for nadir cameras, we want to ALWAYS use VZ only because we don't know
                    # horizontal direction without absolute heading reference (MSCKF or VPS)
                    # 
                    # New logic:
                    #   - If use_vz_only config is True → always use VZ only (for nadir cameras)
                    #   - If use_vz_only config is False → check alignment (for front/side cameras)
                    if view_cfg['use_vz_only']:
                        # Nadir camera: Always use VZ only (we don't know horizontal direction)
                        use_only_vz = True
                    else:
                        # Front/side camera: Use VZ only when motion is nearly vertical
                        use_only_vz = (alignment_deg < view_cfg['nadir_threshold'])

                    # Map camera unit direction → body → world
                    t_body = R_cam_to_body @ t_norm
                    
                    # FISHEYE NADIR CAMERA GEOMETRY DEBUG:
                    # For nadir fisheye pointing down, camera Z-axis points to ground
                    # When image moves down (camera Y+), drone is moving FORWARD (body X+)
                    # t_norm is camera motion direction from Essential matrix
                    print(f"[DEBUG][VIO][GEOMETRY] t_norm (camera frame): [{t_norm[0]:.3f}, {t_norm[1]:.3f}, {t_norm[2]:.3f}]")
                    print(f"[DEBUG][VIO][GEOMETRY] t_body (body frame): [{t_body[0]:.3f}, {t_body[1]:.3f}, {t_body[2]:.3f}]")
                    
                    # IMPROVED SCALE RECOVERY for nadir cameras using AGL
                    # Method: For nadir-pointing cameras, use altitude and pitch angle
                    # to estimate feature depth, then compute velocity from optical flow
                    
                    # Get focal length first (needed for flow computation)
                    focal_px = KB_PARAMS['mu']  # average focal length
                    
                    # Compute average optical flow from tracked features
                    # Also compute MEDIAN flow vector for direction estimation
                    avg_flow_px = 0.0
                    median_flow_vec = np.array([0.0, 0.0])
                    if vio_fe.last_matches is not None:
                        pts_prev, pts_cur = vio_fe.last_matches  # Normalized coordinates from inliers
                        if len(pts_prev) > 0 and len(pts_cur) > 0:
                            # Convert normalized coordinates back to pixel coordinates for flow
                            pts_prev_px = pts_prev * focal_px + np.array([[vio_fe.img_w/2, vio_fe.img_h/2]])
                            pts_cur_px = pts_cur * focal_px + np.array([[vio_fe.img_w/2, vio_fe.img_h/2]])
                            flows = pts_cur_px - pts_prev_px  # Shape: (N, 2) - flow in pixel coordinates
                            avg_flow_px = float(np.median(np.linalg.norm(flows, axis=1)))
                            
                            # Compute median flow direction (more robust than mean)
                            median_flow_vec = np.median(flows, axis=0)  # Median of [dx, dy] in pixels
                    
                    focal_px = KB_PARAMS['mu']  # average focal length
                    
                    # CRITICAL: Get DEM elevation at current position
                    lat_temp, lon_temp = xy_to_latlon(kf.x[0,0], kf.x[1,0], lat0, lon0)
                    dem_temp = dem.sample_m(lat_temp, lon_temp) if dem.ds else 0.0
                    if dem_temp is None or np.isnan(dem_temp):
                        dem_temp = 0.0
                    
                    # Calculate AGL for scale recovery using GPS MSL (not EKF state!)
                    # Problem: If XY position is wrong, DEM sampling is wrong → AGL wrong
                    # Solution: Use GPS MSL altitude directly from flight_log
                    gps_msl_for_scale = None
                    if flight_log_df is not None and len(flight_log_df) > 0:
                        gps_idx = np.argmin(np.abs(flight_log_df['stamp_log'].values - t))
                        gps_msl_for_scale = float(flight_log_df['altitude_MSL_m'].iloc[gps_idx])
                    
                    if gps_msl_for_scale is not None:
                        # Use GPS MSL for scale (more reliable than EKF state during drift)
                        agl_temp = abs(gps_msl_for_scale - dem_temp)
                    else:
                        # Fallback: Use EKF state
                        if z_state.lower() == "agl":
                            agl_temp = abs(kf.x[2,0])  # State is already AGL
                        else:
                            agl_temp = abs(kf.x[2,0] - dem_temp)  # State is MSL, compute AGL
                    
                    agl_temp = max(1.0, agl_temp)  # Ensure minimum altitude for safety
                    
                    # For nadir cameras: Estimate feature depth from AGL and camera pitch
                    # Camera direction in world frame (z-component indicates pitch)
                    cam_z_world = Rwb @ (R_cam_to_body @ np.array([0, 0, 1]))
                    pitch_rad = np.arcsin(-cam_z_world[2])  # Negative z = downward
                    
                    # Average feature depth ≈ AGL / cos(pitch) for ground plane
                    # For near-nadir: depth ≈ AGL
                    cos_pitch = max(0.3, np.cos(pitch_rad))  # Clamp to avoid division by very small numbers
                    depth_est = agl_temp / cos_pitch
                    
                    # === Scale Recovery: Two methods ===
                    # Method 1: Optical flow-based (existing)
                    if dt_img > 1e-4 and avg_flow_px > 2.0:  # Require minimum flow
                        scale_flow = depth_est / focal_px
                        speed_from_flow = (avg_flow_px / dt_img) * scale_flow
                    else:
                        speed_from_flow = 0.0
                    
                    # Method 2: Homography-based (for nadir + flat terrain)
                    # More robust when parallax is low (hovering/near-flat trajectory)
                    speed_from_homography = 0.0
                    use_homography_scale = False
                    
                    if cam_view == 'nadir' and vio_fe.last_matches is not None:
                        pts_prev, pts_cur = vio_fe.last_matches
                        if len(pts_prev) >= 15:  # Need sufficient points for homography
                            # Build camera intrinsic matrix at normalized coordinates
                            # For normalized coordinates, K is identity (features already normalized)
                            K_norm = np.eye(3, dtype=float)
                            
                            # Estimate scale from homography
                            homography_result = estimate_homography_scale(
                                pts1=pts_prev,
                                pts2=pts_cur,
                                K=K_norm,
                                altitude=agl_temp,
                                R_rel=r_vo_mat,
                                min_inliers=15
                            )
                            
                            if homography_result is not None:
                                scale_homog, t_scaled, num_h_inliers = homography_result
                                
                                # Compute velocity from homography translation
                                if dt_img > 1e-4:
                                    # Transform scaled translation to body frame
                                    t_body_homog = R_cam_to_body @ t_scaled
                                    # Transform to world frame
                                    t_world_homog = Rwb @ t_body_homog
                                    # Velocity = displacement / time
                                    speed_from_homography = np.linalg.norm(t_world_homog) / dt_img
                                    
                                    # Sanity check
                                    if speed_from_homography < 50.0:
                                        use_homography_scale = True
                                        print(f"[HOMOGRAPHY] Inliers={num_h_inliers}, scale={scale_homog:.2f}m, "
                                              f"speed={speed_from_homography:.2f}m/s, flow_speed={speed_from_flow:.2f}m/s")
                    
                    # Choose scale method based on availability and confidence
                    if use_homography_scale and speed_from_homography > 0.1:
                        # Use homography scale (more reliable for nadir + flat terrain)
                        speed_final = speed_from_homography
                        scale_method = "homography"
                    elif speed_from_flow > 0.1:
                        # Use optical flow scale (fallback)
                        speed_final = speed_from_flow
                        scale_method = "flow"
                    else:
                        # Insufficient motion for scale estimation
                        speed_final = 0.0
                        scale_method = "none"
                    
                    # Sanity check: Limit velocity to reasonable range (< 50 m/s ≈ 180 km/h)
                    if speed_final > 50.0:
                        print(f"[DEBUG][VIO] WARNING: Computed velocity too high ({speed_final:.1f} m/s), clamping to 50 m/s")
                        speed_final = 50.0
                    
                    print(f"[SCALE] Method={scale_method}, speed={speed_final:.2f}m/s, parallax={avg_flow_px:.1f}px")
                    
                    # CRITICAL FIX: Use optical flow direction from NORMALIZED coordinates
                    # For fisheye cameras, must use normalized coords (after undistortion)
                    # not pixel coords (which have radial distortion)
                    
                    if avg_flow_px > 2.0 and vio_fe.last_matches is not None:  # Only use flow direction if significant flow exists
                        pts_prev, pts_cur = vio_fe.last_matches
                        if len(pts_prev) > 0 and len(pts_cur) > 0:
                            # Compute flow in NORMALIZED camera coordinates (undistorted)
                            flows_normalized = pts_cur - pts_prev  # Shape: (N, 2)
                            median_flow_normalized = np.median(flows_normalized, axis=0)  # Median [dx_norm, dy_norm]
                            
                            flow_norm = np.linalg.norm(median_flow_normalized)
                            if flow_norm > 1e-6:
                                flow_dir_normalized = median_flow_normalized / flow_norm
                            else:
                                flow_dir_normalized = np.array([0.0, 1.0])  # Default: downward in camera
                            
                            # CRITICAL: Optical flow is OPPOSITE to camera velocity!
                            # If camera moves right (+X), features move LEFT (-X) in image.
                            # If camera moves forward (+Z in camera), features expand OUTWARD (no Z).
                            # Therefore: camera_velocity = -optical_flow_direction
                            #
                            # For nadir camera: camera +X = body +X (East-ish after Rwb)
                            #                  camera +Y = body +Y (North-ish after Rwb)
                            # The flow direction needs to be NEGATED to get camera velocity direction.
                            vel_cam = np.array([-flow_dir_normalized[0], -flow_dir_normalized[1], 0.0])
                            vel_cam = vel_cam / np.linalg.norm(vel_cam + 1e-9)  # Normalize
                            
                            # Transform to body frame using BODY_T_CAMDOWN rotation matrix
                            vel_body = R_cam_to_body @ vel_cam
                            vel_body = vel_body * speed_final  # Use final speed (homography or flow)
                        else:
                            # Fallback: use Essential matrix direction
                            vel_body = t_body * speed_final
                    else:
                        # Fallback: use Essential matrix direction (may be less accurate)
                        vel_body = t_body * speed_final
                    
                    vel_world = Rwb @ vel_body  # Rwb maps from body to world coordinates
                    
                    # Store velocity values for CSV logging
                    vel_vx, vel_vy, vel_vz = float(vel_world[0]), float(vel_world[1]), float(vel_world[2])
                    
                    if VERBOSE_DEBUG:
                        print(f"[DEBUG][VIO] flow_px={avg_flow_px:.2f}, dt={dt_img:.4f}, AGL={agl_temp:.2f}m, speed={speed_final:.3f}m/s ({scale_method})")
                        print(f"[DEBUG][VIO] vel_world: [{vel_vx:.3f}, {vel_vy:.3f}, {vel_vz:.3f}] m/s")

                    # ESKF velocity update (vx,vy,vz) with adaptive uncertainty and gating
                    # Define measurement Jacobian in error-state dimensions
                    # Error state: [δp(3), δv(3), δθ(3), δbg(3), δba(3)] + clones [δθ_C(3), δp_C(3)]
                    num_clones = (kf.x.shape[0] - 16) // 7  # FIX: Camera clones from nominal state
                    err_dim = 15 + 6 * num_clones
                    
                    if use_only_vz:
                        h_vel = np.zeros((1, err_dim), dtype=float)
                        h_vel[0,5] = 1.0  # δv_z is at index 5 in error state
                        vel_meas = np.array([[vel_world[2]]])
                        if VERBOSE_DEBUG:
                            print(f"[DEBUG][VIO] t={t:.3f} vel_meas={vel_meas.flatten()} alignment_deg={alignment_deg:.2f} use_only_vz={use_only_vz}")
                    else:
                        h_vel = np.zeros((3, err_dim), dtype=float)
                        h_vel[0,3] = 1.0  # δv_x at index 3
                        h_vel[1,4] = 1.0  # δv_y at index 4
                        h_vel[2,5] = 1.0  # δv_z at index 5
                        vel_meas = vel_world.reshape(-1,1)
                        if VERBOSE_DEBUG:
                            print(f"[DEBUG][VIO] t={t:.3f} vel_meas={vel_meas.flatten()} alignment_deg={alignment_deg:.2f} use_only_vz={use_only_vz}")

                    def h_fun(x, h=h_vel):
                        return h

                    def hx_fun(x, h=h_vel):
                        # Extract velocity from nominal state (indices 3:6)
                        if use_only_vz:
                            return x[5:6].reshape(1, 1)
                        else:
                            return x[3:6].reshape(3, 1)

                    # Adaptive uncertainty based on config and dynamics
                    align_scale = 1.0 + alignment_deg / 45.0
                    flow_scale = 1.0 + max(0.0, (avg_flow_px - 10.0) / 20.0)  # Scale by flow magnitude
                    uncertainty_scale = align_scale * flow_scale
                    
                    # Base uncertainty scaled by config and dynamics
                    if use_only_vz:
                        r_mat = np.array([[(SIGMA_VO_VEL*view_cfg['sigma_scale_z']*uncertainty_scale)**2]])
                    else:
                        r_mat = np.diag([
                            (SIGMA_VO_VEL*view_cfg['sigma_scale_xy']*uncertainty_scale)**2,
                            (SIGMA_VO_VEL*view_cfg['sigma_scale_xy']*uncertainty_scale)**2,
                            (SIGMA_VO_VEL*view_cfg['sigma_scale_z']*uncertainty_scale)**2
                        ])

                    # Innovation gating
                    s_mat = h_vel @ kf.P @ h_vel.T + r_mat
                    try:
                        # Innovation: measurement - predicted (from nominal state)
                        if use_only_vz:
                            predicted_vel = kf.x[5:6, 0].reshape(1, 1)
                        else:
                            predicted_vel = kf.x[3:6, 0].reshape(3, 1)
                        innovation = vel_meas - predicted_vel
                        m2_test = _mahalanobis2(innovation, s_mat)
                    except Exception:
                        m2_test = np.inf

                    # Chi-square test (VERY PERMISSIVE for helicopter VIO)
                    # Standard 99%: 6.63 (1 DoF), 11.34 (3 DoF)
                    # Using very relaxed thresholds to accept more VIO updates
                    # Analysis showed mean Mahalanobis = 35.3 with threshold 30 → 85% rejected
                    chi2_threshold = 25.0 if use_only_vz else 60.0  # Much more permissive
                    
                    # Apply VIO velocity update only if enabled and passes gating
                    vio_accepted = False
                    if use_vio_velocity and m2_test < chi2_threshold:
                        kf.update(
                            z=vel_meas,
                            HJacobian=h_fun,
                            Hx=hx_fun,
                            R=r_mat
                        )
                        vio_accepted = True
                        if VERBOSE_DEBUG:
                            print(f"[DEBUG][VIO] VIO velocity update APPLIED at t={t:.3f} with {ninl} inliers, innovation={innovation.flatten()}, m2={m2_test:.2f}")
                        
                        # === DEBUG: Log residual ===
                        if save_debug_data and residual_csv:
                            log_measurement_update(
                                residual_csv, t, vio_frame, 'VIO_VEL',
                                innovation=innovation,
                                mahalanobis_dist=np.sqrt(m2_test),
                                chi2_threshold=chi2_threshold,
                                accepted=True,
                                S_matrix=s_mat,
                                P_prior=kf.P_prior
                            )
                    else:
                        if not use_vio_velocity:
                            if VERBOSE_DEBUG:
                                print(f"[DEBUG][VIO] VIO velocity update DISABLED by flag at t={t:.3f}")
                        else:
                            if VERBOSE_DEBUG:
                                print(f"[DEBUG][VIO] VIO velocity update REJECTED at t={t:.3f}, m2={m2_test:.2f} > threshold={chi2_threshold:.2f}")
                        
                        # === DEBUG: Log rejected update ===
                        if save_debug_data and residual_csv and use_vio_velocity:
                            log_measurement_update(
                                residual_csv, t, vio_frame, 'VIO_VEL',
                                innovation=innovation,
                                mahalanobis_dist=np.sqrt(m2_test),
                                chi2_threshold=chi2_threshold,
                                accepted=False,
                                S_matrix=s_mat
                            )
                    
                    # --- Plane Constraint Update (for nadir cameras) ---
                    # Add altitude constraint from ground plane to improve Z accuracy
                    if cam_view == 'nadir' and use_homography_scale and agl_temp > 1.0:
                        # Compute plane constraint Jacobian
                        H_plane, predicted_alt = compute_plane_constraint_jacobian(kf, agl_temp)
                        
                        # Measurement: observed MSL altitude from GPS and DEM
                        # predicted_alt = state Z = MSL altitude
                        # agl_temp = AGL (above ground level)
                        # MSL = AGL + DEM elevation
                        z_msl = agl_temp + dem_temp  # Convert AGL to MSL
                        z_altitude = np.array([[z_msl]])
                        
                        # Measurement covariance (uncertainty in altitude)
                        # Higher uncertainty if homography had fewer inliers
                        base_sigma_alt = 1.0  # meters
                        if 'num_h_inliers' in locals() and num_h_inliers < 30:
                            sigma_alt = base_sigma_alt * (30.0 / max(15, num_h_inliers))
                        else:
                            sigma_alt = base_sigma_alt
                        
                        R_altitude = np.array([[sigma_alt**2]])
                        
                        def h_plane_fun(x, h=H_plane):
                            return h
                        
                        def hx_plane_fun(x):
                            # Return current altitude (Z position)
                            return np.array([[x[2, 0]]])
                        
                        # Innovation gating with ADAPTIVE threshold
                        innovation_alt = z_altitude - np.array([[predicted_alt]])
                        s_alt = H_plane @ kf.P @ H_plane.T + R_altitude
                        
                        try:
                            chi2_alt = float(innovation_alt.T @ np.linalg.inv(s_alt) @ innovation_alt)
                            
                            # ADAPTIVE CHI2 THRESHOLD:
                            # When altitude uncertainty is high, use larger threshold to allow corrections
                            # Base: 6.63 (99% CI for 1 DoF)
                            # When P_z > 100m², increase threshold up to 50 to allow re-locking
                            P_z = kf.P[2, 2]  # Position Z covariance
                            if P_z > 100:
                                # Large uncertainty - allow larger innovations
                                chi2_threshold = min(50.0, 6.63 * (1 + np.sqrt(P_z / 100)))
                            else:
                                chi2_threshold = 6.63  # Standard 99% CI
                            
                            if chi2_alt < chi2_threshold:
                                kf.update(
                                    z=z_altitude,
                                    HJacobian=h_plane_fun,
                                    Hx=hx_plane_fun,
                                    R=R_altitude
                                )
                                print(f"[PLANE] Altitude constraint applied: z_msl={z_msl:.2f}m (agl={agl_temp:.2f}m + dem={dem_temp:.2f}m), "
                                      f"predicted={predicted_alt:.2f}m, innovation={float(innovation_alt):.2f}m")
                            else:
                                print(f"[PLANE] Altitude constraint rejected: chi2={chi2_alt:.2f} > {chi2_threshold:.2f} (P_z={P_z:.1f}m², z_msl={z_msl:.2f}m, pred={predicted_alt:.2f}m)")
                        except np.linalg.LinAlgError:
                            pass  # Skip if covariance is singular

                    vo_dx, vo_dy, vo_dz = float(t_norm[0]), float(t_norm[1]), float(t_norm[2])
                    r_eul = R_scipy.from_matrix(r_vo_mat).as_euler('zyx', degrees=True)
                    vo_y, vo_p, vo_r = float(r_eul[0]), float(r_eul[1]), float(r_eul[2])
                    used_vo = True

                    # write VO debug
                    with open(vo_dbg, "a", newline="") as vf:
                        vf.write(
                            f"{max(vio_fe.frame_idx,0)},{ninl},{np.degrees(np.arccos(np.clip((np.trace(r_vo_mat)-1)/2,-1,1))):.3f},"
                            f"{alignment_deg:.3f},{rotation_rate_deg_s:.3f},{int(use_only_vz)},{int(not ok)},"
                            f"{vo_dx:.6f},{vo_dy:.6f},{vo_dz:.6f},{vel_vx:.3f},{vel_vy:.3f},{vel_vz:.3f}\n"
                        )
                    # mark frame index only on VIO update
                    if vio_frame < 0:
                        vio_frame = 0
                    else:
                        vio_frame += 1
                
                img_idx += 1

        # --------- Height state update (AGL/MSL) with adaptive uncertainty ---------
        # DEBUG: Print DEM value, height measurement, innovation, and update status
        lat_now, lon_now = xy_to_latlon(kf.x[0,0], kf.x[1,0], lat0, lon0)
        dem_now = dem.sample_m(lat_now, lon_now) if dem.ds else None
        
        # Determine if we have valid DEM data
        has_valid_dem = False
        if dem_now is not None and not np.isnan(dem_now):
            has_valid_dem = True
            if not hasattr(kf, 'last_valid_dem'):
                kf.last_valid_dem = dem_now
                kf.last_valid_dem_xy = (kf.x[0,0], kf.x[1,0])  # Store position where DEM was valid
            else:
                kf.last_valid_dem = dem_now
                kf.last_valid_dem_xy = (kf.x[0,0], kf.x[1,0])
        else:
            # Try to use last known DEM value if available
            if hasattr(kf, 'last_valid_dem') and kf.last_valid_dem != 0.0:
                dem_now = kf.last_valid_dem
                has_valid_dem = True
                print(f"[WARNING][DEM] DEM lookup failed at lat={lat_now:.6f}, lon={lon_now:.6f}, using last_valid_dem={dem_now:.2f}")
            else:
                # No DEM available at all
                dem_now = 0.0
                has_valid_dem = False

        # Prepare measurement and conversion between AGL/MSL
        # Decision logic:
        # - If DEM available: use AGL for update (regardless of z_state setting)
        # - If NO DEM: use MSL for update (interpolate from flight_log_from_gga.csv by timestamp)
        
        # Try to get MSL measurement from flight_log by timestamp
        msl_measured = None
        if flight_log_df is not None and 'stamp_log' in flight_log_df.columns:
            # Find closest timestamp in flight_log
            time_diffs = np.abs(flight_log_df['stamp_log'].values - t)
            closest_idx = np.argmin(time_diffs)
            time_diff = time_diffs[closest_idx]
            
            # Only use if timestamp is close enough (within 0.5 seconds)
            if time_diff < 0.5:
                msl_measured = float(flight_log_df['altitude_MSL_m'].iloc[closest_idx])
                if VERBOSE_DEM:
                    print(f"[DEBUG][DEM] t={t:.3f} Matched flight_log idx={closest_idx} stamp={flight_log_df['stamp_log'].iloc[closest_idx]:.3f} dt={time_diff:.3f}s MSL={msl_measured:.2f}m")
            else:
                print(f"[WARNING][DEM] t={t:.3f} No close timestamp match in flight_log (closest dt={time_diff:.3f}s)")
        
        if has_valid_dem:
            # DEM available: use adaptive strategy based on AGL
            if z_state.lower() == "agl":
                agl_now = kf.x[2,0]
                msl_now = agl_now + dem_now
                height_m = agl_now  # Use AGL for update
                update_mode = "AGL"
            else:
                # Adaptive DEM Constraint Strategy
                # CRITICAL FIX: ALWAYS use msl_measured from flight_log when available!
                # The state Z (msl_now) may have drifted due to IMU integration error.
                # Only msl_measured from GNSS/baro provides absolute altitude reference.
                
                msl_now = kf.x[2,0]
                agl_now = msl_now - dem_now
                
                # PRIMARY: Use msl_measured from flight_log if available
                if msl_measured is not None:
                    height_m = msl_measured
                    expected_agl = msl_measured - dem_now
                    update_mode = f"MSL (from flight_log, expected AGL={expected_agl:.1f}m)"
                    
                    # Log when there's significant altitude error
                    altitude_error = msl_now - msl_measured
                    if abs(altitude_error) > 10.0:
                        print(f"[DEM][CORRECT] Large altitude error detected: "
                              f"state_MSL={msl_now:.1f}m vs flight_log_MSL={msl_measured:.1f}m "
                              f"(error={altitude_error:.1f}m)")
                else:
                    # FALLBACK: No flight_log MSL available
                    # Use DEM + reasonable AGL estimate
                    
                    min_safe_agl = 0.5  # meters (minimum clearance)
                    
                    if agl_now < min_safe_agl:
                        # EMERGENCY: Estimated altitude too low!
                        target_agl = 2.0
                        height_m = dem_now + target_agl
                        update_mode = f"MSL (emergency lift to {target_agl}m AGL)"
                        print(f"[DEM][EMERGENCY] Altitude {agl_now:.2f}m < {min_safe_agl}m! "
                              f"Applying emergency lift (no flight_log available)")
                    elif agl_now < 30.0:
                        # Low altitude without flight_log
                        target_agl = max(agl_now, 5.0)
                        height_m = dem_now + target_agl
                        update_mode = f"MSL (low alt, soft AGL={target_agl:.1f}m, no flight_log)"
                    elif agl_now < 150.0:
                        # Medium altitude without flight_log
                        target_agl = agl_now
                        height_m = dem_now + target_agl
                        update_mode = f"MSL (AGL={target_agl:.1f}m, no flight_log)"
                    else:
                        # High altitude without flight_log
                        fallback_agl = 100.0
                        height_m = dem_now + fallback_agl
                        update_mode = f"MSL (high alt, fallback AGL={fallback_agl:.0f}m)"
                        print(f"[DEM][ADAPTIVE] Altitude {agl_now:.1f}m > 150m, using fallback AGL={fallback_agl:.0f}m → MSL={height_m:.1f}m")
        else:
            # NO DEM: use MSL for update (from flight_log_from_gga.csv)
            if z_state.lower() == "agl":
                agl_now = kf.x[2,0]
                msl_now = agl_now  # No DEM, so AGL = MSL conceptually
                # Use time-varying MSL from flight_log if available, otherwise use initial MSL
                height_m = msl_measured if msl_measured is not None else msl0_m
                update_mode = f"MSL (no DEM, {'interpolated' if msl_measured is not None else 'initial'})"
            else:
                msl_now = kf.x[2,0]
                agl_now = msl_now  # No DEM, so AGL = MSL
                # Use time-varying MSL from flight_log if available, otherwise use initial MSL
                height_m = msl_measured if msl_measured is not None else msl0_m
                update_mode = f"MSL (no DEM, {'interpolated' if msl_measured is not None else 'initial'})"
        
        if VERBOSE_DEM:
            print(f"[DEBUG][DEM] t={t:.3f} dem_now={dem_now:.2f} agl_now={agl_now:.2f} msl_now={msl_now:.2f} height_m={height_m:.2f} mode={update_mode} px={kf.x[0,0]:.2f} py={kf.x[1,0]:.2f}")

        # Proceed with update only if we have valid height measurement
        if not np.isnan(height_m):
            # ESKF: Height measurement Jacobian in error-state dimensions
            # Error state: [δp(3), δv(3), δθ(3), δbg(3), δba(3)] + clones [δθ_C(3), δp_C(3)]
            num_clones = (kf.x.shape[0] - 16) // 7  # Camera clones
            err_dim = 15 + 6 * num_clones
            h_height = np.zeros((1, err_dim), dtype=float)
            h_height[0,2] = 1.0  # Height is δp_z (3rd element of error state)

            def h_fun(x, h=h_height):
                return h

            def hx_fun(x, h=h_height):
                # Measurement function: extract height from nominal state
                return x[2:3].reshape(1, 1)

            # ========== IMPROVED: Adaptive uncertainty scaling ==========
            height_cov_scale = 1.0
            
            # 1. Scale with HORIZONTAL POSITION UNCERTAINTY (KEY IMPROVEMENT!)
            # If XY uncertain → lat/lon unreliable → DEM lookup unreliable → increase noise
            xy_uncertainty = float(np.trace(kf.P[0:2, 0:2]))  # Sum of variances in X,Y
            xy_std = np.sqrt(xy_uncertainty / 2.0)  # Average std in meters
            
            # REDUCED scaling to avoid over-weakening DEM constraint
            # 1m std → 1.5x noise, 5m std → 2.5x noise, 10m std → 3.5x noise
            if xy_std > 1.0:
                xy_scale_factor = 1.0 + (xy_std * 0.25)  # Gentler scaling (was 1.0 + xy_std)
                height_cov_scale *= xy_scale_factor
                if VERBOSE_DEM:
                    print(f"[DEM][ADAPTIVE] XY uncertainty: std={xy_std:.2f}m → height_noise_scale={xy_scale_factor:.2f}x")
            
            # 2. Scale with time since last GNSS/VPS correction
            # If no recent absolute position corrections → XY drifting → distrust DEM more
            if not hasattr(kf, 'last_absolute_correction_time'):
                kf.last_absolute_correction_time = t  # Initialize
            time_since_correction = t - kf.last_absolute_correction_time
            
            # After 10s without GNSS/VPS, start scaling up (conservative)
            if time_since_correction > 10.0:
                time_scale_factor = 1.0 + (time_since_correction - 10.0) / 40.0  # Slower growth (was /20.0)
                height_cov_scale *= time_scale_factor
                if VERBOSE_DEM:
                    print(f"[DEM][ADAPTIVE] Time since correction: {time_since_correction:.1f}s → height_noise_scale={time_scale_factor:.2f}x")
            
            # 3. Scale with horizontal speed (higher uncertainty at higher speeds)
            speed_ms = float(np.linalg.norm(kf.x[3:6, 0]))
            if speed_ms > 10:
                speed_scale_factor = 1.0 + (speed_ms - 10) / 15.0
                height_cov_scale *= speed_scale_factor

            # 4. Scale with vertical dynamics if we have history
            if hasattr(kf, 'last_height'):
                height_rate = abs(height_m - kf.last_height) / dt
                if height_rate > 2.0:  # More than 2 m/s vertical change
                    height_cov_scale *= (1.0 + height_rate / 4.0)
            kf.last_height = height_m

            # 5. Adjust uncertainty based on update mode
            if not has_valid_dem:
                # No DEM: MSL update should have higher uncertainty (less frequent, from initial only)
                height_cov_scale *= 5.0  # Increase uncertainty for MSL-only updates

            # Set up measurement uncertainty
            r_mat = np.array([[SIGMA_AGL_Z**2 * height_cov_scale]])
            
            # Print final adaptive noise for debugging
            if VERBOSE_DEM:
                print(f"[DEM][ADAPTIVE] Final height_noise: base={SIGMA_AGL_Z:.2f}m, scale={height_cov_scale:.2f}x, final={np.sqrt(r_mat[0,0]):.2f}m")

            # ========== IMPROVED: Innovation gating with DEM slope analysis ==========
            s_mat = h_height @ kf.P @ h_height.T + r_mat
            
            # Estimate DEM slope at current position for better innovation prediction
            dem_slope_magnitude = 0.0
            if has_valid_dem and hasattr(kf, 'last_valid_dem_xy'):
                # Sample DEM at small offset to estimate local slope
                delta_m = 5.0  # 5 meter offset for numerical gradient
                lat_east, lon_east = xy_to_latlon(kf.x[0,0] + delta_m, kf.x[1,0], lat0, lon0)
                lat_north, lon_north = xy_to_latlon(kf.x[0,0], kf.x[1,0] + delta_m, lat0, lon0)
                
                dem_east = dem.sample_m(lat_east, lon_east) if dem.ds else None
                dem_north = dem.sample_m(lat_north, lon_north) if dem.ds else None
                
                if dem_east is not None and dem_north is not None:
                    slope_x = (dem_east - dem_now) / delta_m  # dz/dx
                    slope_y = (dem_north - dem_now) / delta_m  # dz/dy
                    dem_slope_magnitude = np.sqrt(slope_x**2 + slope_y**2)
                    if VERBOSE_DEM:
                        print(f"[DEM][SLOPE] Estimated slope: {dem_slope_magnitude:.4f} (dz/dx={slope_x:.4f}, dz/dy={slope_y:.4f})")
            
            try:
                # Innovation: measurement - predicted (from nominal state)
                predicted_height = kf.x[2, 0]  # Extract height from nominal state
                innovation = np.array([[height_m - predicted_height]])
                m2_test = _mahalanobis2(innovation, s_mat)
                
                # Estimate expected innovation magnitude from XY uncertainty and DEM slope
                # If XY position is uncertain by σ_xy, and terrain has slope m → expected z error ≈ σ_xy * m
                expected_innovation_from_slope = xy_std * dem_slope_magnitude if dem_slope_magnitude > 0 else 0.0
                
                if VERBOSE_DEM:
                    print(f"[DEM][INNOVATION] predicted_z={predicted_height:.2f}m, measured_z={height_m:.2f}m, innovation={innovation[0,0]:.2f}m, "
                          f"expected_from_slope={expected_innovation_from_slope:.2f}m, mahalanobis={np.sqrt(m2_test):.2f}")
            except Exception as e:
                print(f"[DEM][ERROR] Innovation computation failed: {e}")
                m2_test = np.inf

            # ========== IMPROVED: Adaptive innovation gating ==========
            # For IMU-only mode (no VIO/VPS), use relaxed threshold to handle larger drift
            # Standard: 6.63 (99% confidence, 1 DoF)
            # Conservative: 9.21 (99.9% confidence, 1 DoF) - use when XY uncertainty is high
            # IMU-only: 100.0 (very relaxed, accept most reasonable measurements)
            no_vision_corrections = len(imgs) == 0 and len(vps_list) == 0
            
            # Adaptive threshold based on horizontal uncertainty
            if no_vision_corrections:
                threshold = 100.0  # Very relaxed for IMU-only
            elif xy_std > 10.0:  # High XY uncertainty → very conservative
                threshold = 15.0  # Even more relaxed than 99.9%
                if VERBOSE_DEM:
                    print(f"[DEM][GATE] High XY uncertainty ({xy_std:.1f}m) → using relaxed threshold {threshold:.1f}")
            elif xy_std > 5.0:  # Moderate XY uncertainty → conservative
                threshold = 9.21  # 99.9% confidence
                if VERBOSE_DEM:
                    print(f"[DEM][GATE] Moderate XY uncertainty ({xy_std:.1f}m) → using conservative threshold {threshold:.1f}")
            else:
                threshold = 6.63  # Standard 99% confidence
            
            if m2_test < threshold:
                status = "APPLIED"
                if m2_test >= 6.63:
                    status += " (relaxed gate)"
                if VERBOSE_DEM:
                    print(f"[DEBUG][DEM] Height update {status} at t={t:.3f} mode={update_mode} innovation={innovation[0,0]:.3f} m2_test={m2_test:.2f} threshold={threshold:.1f}")
                kf.update(
                    z=np.array([[height_m]]),
                    HJacobian=h_fun,
                    Hx=hx_fun,
                    R=r_mat
                )
                
                # Log residual for debugging
                if save_debug_data and residual_csv:
                    log_measurement_update(
                        residual_csv, t, vio_frame, 'DEM',
                        innovation=innovation,
                        mahalanobis_dist=np.sqrt(m2_test),
                        chi2_threshold=threshold,
                        accepted=True,
                        S_matrix=s_mat,
                        P_prior=kf.P_prior
                    )
            else:
                if VERBOSE_DEM:
                    print(f"[DEBUG][DEM] Height update REJECTED at t={t:.3f} mode={update_mode} innovation={innovation[0,0]:.3f} m2_test={m2_test:.2f} threshold={threshold:.1f}")
                
                # Log rejected update
                if save_debug_data and residual_csv:
                    log_measurement_update(
                        residual_csv, t, vio_frame, 'DEM',
                        innovation=innovation,
                        mahalanobis_dist=np.sqrt(m2_test),
                        chi2_threshold=threshold,
                        accepted=False,
                        S_matrix=s_mat,
                        P_prior=kf.P_prior
                    )

        # --------- Logging ---------
        with open(pose_csv, "a", newline="") as f:
            # Frame column: set only when VIO updated at (or just before) this IMU time; else blank
            frame_str = str(vio_frame) if used_vo else ""
            f.write(
                f"{t - t0:.6f},{dt:.6f},{frame_str},"
                f"{kf.x[0,0]:.3f},{kf.x[1,0]:.3f},{msl_now:.3f},"
                f"{kf.x[3,0]:.3f},{kf.x[4,0]:.3f},{kf.x[5,0]:.3f},"
                f"{lat_now:.8f},{lon_now:.8f},{agl_now:.3f},"
                f"{'' if np.isnan(vo_dx) else f'{vo_dx:.6f}'},{'' if np.isnan(vo_dy) else f'{vo_dy:.6f}'},{'' if np.isnan(vo_dz) else f'{vo_dz:.6f}'},"
                f"{'' if np.isnan(vo_r) else f'{vo_r:.3f}'},{'' if np.isnan(vo_p) else f'{vo_p:.3f}'},{'' if np.isnan(vo_y) else f'{vo_y:.3f}'}\n"
            )

        toc = time.time()
        with open(inf_csv, "a", newline="") as f:
            dt_proc = toc - tic
            fps = (1.0/dt_proc) if dt_proc>0 else 0.0
            f.write(f"{i},{dt_proc:.6f},{fps:.2f}\n")

        # write detailed state debug line for offline inspection
        try:
            with open(state_dbg_csv, "a", newline="") as f:
                try:
                    px = float(kf.x[0,0])
                    py = float(kf.x[1,0])
                    pz = float(kf.x[2,0])
                    vx = float(kf.x[3,0])
                    vy = float(kf.x[4,0])
                    vz = float(kf.x[5,0])
                except Exception:
                    px=py=pz=vx=vy=vz=float('nan')
                try:
                    a_wx = float(a_world[0])
                    a_wy = float(a_world[1])
                    a_wz = float(a_world[2])
                except Exception:
                    a_wx=a_wy=a_wz=float('nan')
                dem_val = dem_now if dem_now is not None else float('nan')
                if z_state.lower() == 'agl':
                    agl_val = float(kf.x[2,0])
                    msl_val = agl_val + (dem_val if not np.isnan(dem_val) else 0.0)
                else:
                    msl_val = float(kf.x[2,0])
                    agl_val = msl_val - (dem_val if not np.isnan(dem_val) else 0.0)
                f.write(f"{t:.6f},{px:.6f},{py:.6f},{pz:.6f},{vx:.6f},{vy:.6f},{vz:.6f},"
                        f"{a_wx:.6f},{a_wy:.6f},{a_wz:.6f},{dem_val:.6f},{agl_val:.6f},{msl_val:.6f}\n")
        except Exception:
            pass

        # Error logging: Compare VIO prediction vs PPK ground truth (preferred) or GGA (fallback)
        try:
            gt_df = ppk_trajectory_df if ppk_trajectory_df is not None else flight_log_df
            use_ppk = ppk_trajectory_df is not None
            
            if gt_df is not None and len(gt_df) > 0:
                # Find closest ground truth timestamp
                gt_idx = np.argmin(np.abs(gt_df['stamp_log'].values - t))
                gt_row = gt_df.iloc[gt_idx]
                
                # Ground truth position
                if use_ppk:
                    gt_lat = gt_row['lat']
                    gt_lon = gt_row['lon']
                    gt_alt = gt_row['height']  # PPK uses ellipsoidal height
                    gt_yaw_rad = gt_row['yaw']  # NED frame
                    gt_ve = gt_row['ve']
                    gt_vn = gt_row['vn']
                    gt_vu = gt_row['vu']
                else:
                    gt_lat = gt_row['lat_dd']
                    gt_lon = gt_row['lon_dd']
                    gt_alt = gt_row['altitude_MSL_m']
                    gt_yaw_rad = None
                    gt_ve = gt_vn = gt_vu = None
                
                # Convert to local ENU
                gt_E_gnss, gt_N_gnss = latlon_to_xy(gt_lat, gt_lon, lat0, lon0)
                gt_U_gnss = gt_alt
                
                # Apply lever arm compensation: PPK position is at GNSS antenna
                # We track IMU position, so need to compensate for lever arm
                # gt_imu = gt_gnss - R_body_to_world @ lever_arm
                if use_ppk and 'roll' in gt_row and 'pitch' in gt_row and 'yaw' in gt_row:
                    # Compute R_body_to_world from PPK attitude at this timestep
                    gt_roll_ned = gt_row['roll']
                    gt_pitch_ned = gt_row['pitch']
                    gt_yaw_ned_rad = gt_row['yaw']
                    gt_roll_enu = gt_roll_ned
                    gt_pitch_enu = -gt_pitch_ned
                    gt_yaw_enu = np.pi/2 - gt_yaw_ned_rad
                    R_BW_flu_gt = R_scipy.from_euler('ZYX', [gt_yaw_enu, gt_pitch_enu, gt_roll_enu]).as_matrix()
                    R_FLU_to_FRD = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    R_BW_gt = R_BW_flu_gt @ R_FLU_to_FRD.T
                    lever_arm_world_gt = R_BW_gt @ IMU_GNSS_LEVER_ARM
                    gt_E = gt_E_gnss - lever_arm_world_gt[0]
                    gt_N = gt_N_gnss - lever_arm_world_gt[1]
                    gt_U = gt_U_gnss - lever_arm_world_gt[2]
                else:
                    # No attitude data - use GNSS position directly
                    gt_E = gt_E_gnss
                    gt_N = gt_N_gnss
                    gt_U = gt_U_gnss
                
                # VIO prediction
                vio_E = float(kf.x[0,0])
                vio_N = float(kf.x[1,0])
                vio_U = msl_val
                
                # Position errors
                err_E = vio_E - gt_E
                err_N = vio_N - gt_N
                err_U = vio_U - gt_U
                pos_error = np.sqrt(err_E**2 + err_N**2 + err_U**2)
                
                # Velocity errors
                vio_vE = float(kf.x[3,0])
                vio_vN = float(kf.x[4,0])
                vio_vU = float(kf.x[5,0])
                
                if use_ppk and gt_ve is not None:
                    # PPK has direct velocity
                    vel_err_E = vio_vE - gt_ve
                    vel_err_N = vio_vN - gt_vn
                    vel_err_U = vio_vU - gt_vu
                elif gt_idx > 0:
                    # Compute velocity from position derivative
                    dt_gt = gt_row['stamp_log'] - gt_df.iloc[gt_idx-1]['stamp_log']
                    if dt_gt > 0.01:
                        if use_ppk:
                            prev_lat = gt_df.iloc[gt_idx-1]['lat']
                            prev_lon = gt_df.iloc[gt_idx-1]['lon']
                            prev_alt = gt_df.iloc[gt_idx-1]['height']
                        else:
                            prev_lat = gt_df.iloc[gt_idx-1]['lat_dd']
                            prev_lon = gt_df.iloc[gt_idx-1]['lon_dd']
                            prev_alt = gt_df.iloc[gt_idx-1]['altitude_MSL_m']
                        prev_E, prev_N = latlon_to_xy(prev_lat, prev_lon, lat0, lon0)
                        gt_vE = (gt_E - prev_E) / dt_gt
                        gt_vN = (gt_N - prev_N) / dt_gt
                        gt_vU = (gt_alt - prev_alt) / dt_gt
                        vel_err_E = vio_vE - gt_vE
                        vel_err_N = vio_vN - gt_vN
                        vel_err_U = vio_vU - gt_vU
                    else:
                        vel_err_E = vel_err_N = vel_err_U = 0.0
                else:
                    vel_err_E = vel_err_N = vel_err_U = 0.0
                    
                vel_error = np.sqrt(vel_err_E**2 + vel_err_N**2 + vel_err_U**2)
                
                # Yaw comparison
                q_vio = kf.x[6:10,0]
                yaw_vio = np.rad2deg(quaternion_to_yaw(q_vio))
                
                if use_ppk and gt_yaw_rad is not None:
                    # PPK yaw is NED (0=North, CW positive)
                    # Convert to ENU for comparison: yaw_enu = 90° - yaw_ned
                    yaw_gt_enu = 90.0 - np.rad2deg(gt_yaw_rad)
                    yaw_gt = yaw_gt_enu
                    yaw_error = ((yaw_vio - yaw_gt + 180) % 360) - 180
                else:
                    # Fallback: compute bearing from trajectory
                    if gt_idx > 0 and gt_idx < len(gt_df) - 1:
                        if use_ppk:
                            next_lat = gt_df.iloc[gt_idx+1]['lat']
                            next_lon = gt_df.iloc[gt_idx+1]['lon']
                        else:
                            next_lat = gt_df.iloc[gt_idx+1]['lat_dd']
                            next_lon = gt_df.iloc[gt_idx+1]['lon_dd']
                        next_E, next_N = latlon_to_xy(next_lat, next_lon, lat0, lon0)
                        dE = next_E - gt_E
                        dN = next_N - gt_N
                        if np.sqrt(dE**2 + dN**2) > 0.01:
                            yaw_gt = np.degrees(np.arctan2(dE, dN))
                            yaw_error = ((yaw_vio - yaw_gt + 180) % 360) - 180
                        else:
                            yaw_gt = float('nan')
                            yaw_error = float('nan')
                    else:
                        yaw_gt = float('nan')
                        yaw_error = float('nan')
                
                # Write to error log
                with open(error_csv, "a", newline="") as ef:
                    ef.write(
                        f"{t:.6f},{pos_error:.3f},{err_E:.3f},{err_N:.3f},{err_U:.3f},"
                        f"{vel_error:.3f},{vel_err_E:.3f},{vel_err_N:.3f},{vel_err_U:.3f},"
                        f"{err_U:.3f},{yaw_vio:.2f},{yaw_gt:.2f},{yaw_error:.2f},"
                        f"{gt_lat:.8f},{gt_lon:.8f},{gt_alt:.3f},"
                        f"{vio_E:.3f},{vio_N:.3f},{vio_U:.3f}\n"
                    )
        except Exception as e:
            pass  # Silent fail for error logging

        # progress line
        speed_ms = float(np.linalg.norm(kf.x[3:6,0]))
        mode_bits = ["IMU"]
        if dem.ds: mode_bits.append("DEM")
        if vio_fe is not None: mode_bits.append("VIO")
        if len(vps_list) > 0: mode_bits.append("VPS")
        mode_str = "+".join(mode_bits)
        print(
            f"t={t - t0:8.3f}s | pos(lat,lon,AGL)=({lat_now:.6f},{lon_now:.6f},{agl_now:.2f}m) "
            f"| v_xy={speed_ms*3.6:5.1f}km/h | {mode_str}", end="\r"
        )

    print("\n\n--- Done ---")
    print(f"Total IMU samples: {len(imu)} | Images used: {max(0,vio_frame+1) if vio_frame>=0 else 0} | VPS used: {vps_idx}")
    print(f"Magnetometer: {mag_update_count} updates applied | {mag_reject_count} rejected | {len(mag_list)} total samples")
    print(f"  Yaw resets: {yaw_reset_count} | Vibration detected: {vibration_detected_count} samples")
    print(f"ZUPT: {zupt_applied_count} applied | {zupt_rejected_count} rejected | {zupt_total_detected} detected stationary")
    
    # Print MSCKF triangulation statistics
    print_msckf_stats()
    
    # Print loop closure statistics
    if loop_detector is not None:
        loop_detector.print_stats()
    
    print(f"Outputs: {pose_csv}, {inf_csv}, {vo_dbg}")
    
    # Print error statistics (if error_log was generated)
    try:
        error_df = pd.read_csv(error_csv)
        if len(error_df) > 0:
            print("\n=== Error Statistics (VIO vs GPS Ground Truth) ===")
            print(f"Position Error:")
            print(f"  Mean: {error_df['pos_error_m'].mean():.2f} m")
            print(f"  Median: {error_df['pos_error_m'].median():.2f} m")
            print(f"  Max: {error_df['pos_error_m'].max():.2f} m")
            print(f"  Final: {error_df['pos_error_m'].iloc[-1]:.2f} m")
            print(f"Velocity Error:")
            print(f"  Mean: {error_df['vel_error_m_s'].mean():.3f} m/s")
            print(f"  Final: {error_df['vel_error_m_s'].iloc[-1]:.3f} m/s")
            print(f"Altitude Error:")
            print(f"  Mean: {error_df['alt_error_m'].mean():.2f} m")
            print(f"  Final: {error_df['alt_error_m'].iloc[-1]:.2f} m")
            
            # Yaw error (filter out NaN values)
            yaw_errors = error_df['yaw_error_deg'].dropna()
            if len(yaw_errors) > 0:
                print(f"Yaw Error:")
                print(f"  Mean: {yaw_errors.mean():.1f}°")
                print(f"  Final: {yaw_errors.iloc[-1]:.1f}°")
            print(f"\nDetailed errors saved to: {error_csv}")
    except Exception:
        pass
    
    toc_all = time.time()
    dt_proc_all = toc_all - tic_all
    print(f"\n=== Finished in {dt_proc_all:.2f} seconds ===")

# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="IMU-driven EKF with optional VIO/VPS/DEM")
    ap.add_argument("--config", default="config_dji_m600_quarry.yaml", 
                    help="Path to YAML configuration file (default: config_dji_m600_quarry.yaml)")
    ap.add_argument("--imu", required=True, help="Path to imu.csv (stamp_log, ori_*, ang_*, lin_*)")
    ap.add_argument("--quarry", required=True, help="Path to flight_log_from_gga.csv (for initial lat_dd/lon_dd/altitude_MSL_m only)")
    ap.add_argument("--output", default="out_vio_imu_ekf", help="Output directory")

    ap.add_argument("--images_dir", default=None, help="Directory of images (e.g., ./camera__image_mono/images)")
    ap.add_argument("--images_index", default=None, help="CSV with (stamp_log,filename)")
    
    # Multi-camera support
    ap.add_argument("--front_images_dir", default=None, help="Directory of front camera images (optional, for multi-camera VIO)")
    ap.add_argument("--front_images_index", default=None, help="CSV for front camera (stamp_log,filename)")
    ap.add_argument("--side_images_dir", default=None, help="Directory of side camera images (optional, for multi-camera VIO)")
    ap.add_argument("--side_images_index", default=None, help="CSV for side camera (stamp_log,filename)")

    ap.add_argument("--vps", default=None, help="vps_result.csv (t,lat,lon)")
    ap.add_argument("--mag", default=None, help="Magnetometer CSV (e.g., vector3.csv with stamp_log,x,y,z,frame_id)")
    ap.add_argument("--dem", default=None, help="DEM GeoTIFF (e.g., DSM_10_N47_00_W054_00_AOI.tif)")

    ap.add_argument("--img_w", type=int, default=1140, help="Runtime image width (resized)")
    ap.add_argument("--img_h", type=int, default=1080, help="Runtime image height (resized)")
    ap.add_argument("--z_state", choices=["msl","agl"], default="msl", help="Store EKF z as MSL or AGL (if DEM available)")
    ap.add_argument("--camera_view", choices=["nadir","front","side","multi"], default="nadir", 
                    help="Camera view mode: nadir (downward), front (forward), side (oblique), or multi (auto-detect from inputs). "
                         "Multi-camera VIO uses all provided cameras for better robustness.")
    ap.add_argument("--estimate_imu_bias", action="store_true", default=False,
                    help="Estimate IMU bias from initial static period. If False, relies on online estimation with adaptive process noise.")
    ap.add_argument("--use_magnetometer", action="store_true", default=True,
                    help="Enable magnetometer-aided heading correction (reduces yaw drift). Default: False (use IMU quaternion only)")
    ap.add_argument("--disable_vio_velocity", action="store_true", default=False,
                    help="Disable VIO velocity updates (use only MSCKF feature constraints). Useful if scale recovery is problematic.")
    ap.add_argument("--save_debug_data", action="store_true", default=False,
                    help="Save comprehensive debug data (raw IMU, state/covariance, residuals, feature stats, MSCKF window, calibration)")
    ap.add_argument("--save_keyframe_images", action="store_true", default=False,
                    help="Save keyframe images with feature/inlier/reprojection overlays (requires --save_debug_data)")
    ap.add_argument("--use_legacy_propagation", action="store_true", default=False,
                    help="Use legacy sample-by-sample propagation instead of IMU preintegration (for benchmarking)")
    ap.add_argument("--ground_truth", default=None, 
                    help="Path to PPK ground truth file (e.g., bell412_dataset3_frl.pos) for initial yaw. "
                         "If provided, uses the yaw from first row to initialize EKF heading instead of IMU quaternion.")

    args = ap.parse_args()
    
    # Auto-detect multi-camera configuration
    cameras_available = []
    if args.images_dir:
        cameras_available.append('nadir')
    if args.front_images_dir:
        cameras_available.append('front')
    if args.side_images_dir:
        cameras_available.append('side')
    
    # Override camera_view if multi-camera detected and mode is 'multi'
    if args.camera_view == 'multi' or (len(cameras_available) > 1 and args.camera_view == 'nadir'):
        if len(cameras_available) > 1:
            print(f"[MULTI-CAM] Detected {len(cameras_available)} cameras: {', '.join(cameras_available)}")
            print(f"[MULTI-CAM] Multi-camera VIO will be used for better robustness")
            args.camera_view = 'multi'
        elif len(cameras_available) == 1:
            args.camera_view = cameras_available[0]
            print(f"[SINGLE-CAM] Using {args.camera_view} camera only")
        else:
            print(f"[WARNING] No camera images provided, VIO will be disabled")

    # Load configuration from YAML
    print("=" * 80)
    print(f"[CONFIG] Loading configuration from: {args.config}")
    try:
        config = load_config(args.config)
        # Update global variables
        globals().update(config)
        print(f"[CONFIG] Successfully loaded configuration")
        print(f"[CONFIG] Camera: {config['KB_PARAMS']['w']}x{config['KB_PARAMS']['h']}, "
              f"mu={config['KB_PARAMS']['mu']:.2f}, mv={config['KB_PARAMS']['mv']:.2f}")
        print(f"[CONFIG] VIO: min_parallax={config['MIN_PARALLAX_PX']}px, "
              f"min_baseline={config['MIN_MSCKF_BASELINE']}m, "
              f"chi2_mult={config['MSCKF_CHI2_MULTIPLIER']}")
        print(f"[CONFIG] Magnetometer: field_strength={config['MAG_FIELD_STRENGTH']} µT, "
              f"declination={np.rad2deg(config['MAG_DECLINATION']):.1f}°, "
              f"convergence_window={config.get('MAG_INITIAL_CONVERGENCE_WINDOW', 30.0)}s")
        # Print lever arm configuration
        lever_arm = config.get('IMU_GNSS_LEVER_ARM', np.zeros(3))
        if np.any(lever_arm != 0):
            print(f"[CONFIG] IMU-GNSS Lever Arm: X={lever_arm[0]:.4f}m, Y={lever_arm[1]:.4f}m, Z={lever_arm[2]:.4f}m")
        else:
            print(f"[CONFIG] IMU-GNSS Lever Arm: Not configured (assuming GNSS at IMU position)")
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found: {args.config}")
        print(f"[ERROR] Please provide a valid YAML config file with --config")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        sys.exit(1)

    # DEBUG: Print propagation mode selection
    preint_mode = not args.use_legacy_propagation
    print(f"[CONFIG] IMU Propagation Mode: {'PREINTEGRATION (Forster et al.)' if preint_mode else 'LEGACY (sample-by-sample)'}")
    print(f"[CONFIG] CLI Flag --use_legacy_propagation: {args.use_legacy_propagation}")
    print(f"[CONFIG] use_preintegration parameter: {preint_mode}")
    print("=" * 80)
    print()

    run(
        imu_path=args.imu,
        quarry_path=args.quarry,
        output_dir=args.output,
        images_dir=args.images_dir,
        images_index_csv=args.images_index,
        vps_csv=args.vps,
        mag_csv=args.mag,
        dem_path=args.dem,
        downscale_size=(args.img_w, args.img_h),
        z_state=args.z_state,
        camera_view=args.camera_view,
        estimate_imu_bias=args.estimate_imu_bias,
        use_magnetometer=args.use_magnetometer,
        use_vio_velocity=not args.disable_vio_velocity,
        save_debug_data=args.save_debug_data,
        save_keyframe_images=args.save_keyframe_images,
        use_preintegration=not args.use_legacy_propagation,  # NEW: Control preintegration mode
        ground_truth_path=args.ground_truth,  # PPK ground truth for initial yaw
    )

