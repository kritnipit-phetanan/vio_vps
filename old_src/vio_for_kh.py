#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
import math
import glob
import csv
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R_scipy

# import rasterio
# from pyproj import CRS, Transformer

# --- Necessary imports for the ExtendedKalmanFilter class ---
from copy import deepcopy
from math import log, exp, sqrt
from numpy import dot, zeros, eye
import scipy.linalg as linalg
# from filterpy.stats import logpdf
# from filterpy.common import pretty_str, reshape_z

# ========================
# Configuration & Tuning
# ========================

# Fisheye (Kannala-Brandt) camera from user (DJI M600 dataset)
KB_PARAMS = {
    "k2": -0.07937700,
    "k3":  0.02228435,
    "k4": -0.03852023,
    "k5":  0.01346873,
    "mu":  854.383024,
    "mv":  853.285954,
    "u0":  780.324522,
    "v0":  520.690672,
    "w":   1440,
    "h":   1080,
}

# Camera-IMU extrinsics
# BODY_T_CAMDOWN: Camera pointing down (nadir view)
BODY_T_CAMDOWN = np.array([
    [ 0.00235643,  0.99997843, -0.00613037, -0.25805624],
    [-0.99960218,  0.00218315, -0.02811962, -0.01138283],
    [-0.02810563,  0.00619420,  0.99958577,  0.09243762],
    [ 0.0,         0.0,         0.0,         1.0       ],
], dtype=np.float64)

# BODY_T_CAMFRONT: Camera pointing forward (front view)
# Typical front-facing camera: X-right, Y-down, Z-forward in camera frame
# Maps to body frame: X-forward, Y-right, Z-down
BODY_T_CAMFRONT = np.array([
    [ -0.04200713, -0.01166497,  0.99904921,  0.17666233],
    [ 0.99899047,  0.01544242,  0.04218496, -0.05171531],
    [ -0.01591983,  0.99981271,  0.01100451, -0.04656282],
    [ 0.0,         0.0,         0.0,         1.0],
], dtype=np.float64)

# BODY_T_CAMSIDE: Camera pointing to the side (oblique view)
# Typical side camera: rotated 90° from front
BODY_T_CAMSIDE = np.array([
    [ 1.0,  0.0,  0.0,  0.0],   # Camera X → Body X
    [ 0.0,  0.0,  1.0,  0.2],   # Camera Z → Body Y (side)
    [ 0.0,  1.0,  0.0, -0.05],  # Camera Y → Body Z
    [ 0.0,  0.0,  0.0,  1.0],
], dtype=np.float64)

# IMU parameters (noise etc.)
IMU_PARAMS = {
    "acc_n": 0.08,      # m/s^2 (measurement noise) - from IMU Intrinsic Calibration
    "gyr_n": 0.004,     # rad/s (measurement noise) - from IMU Intrinsic Calibration
    "acc_w": 0.001,     # m/s^2 (random walk) - INCREASED 25x from 0.00004 to reduce position drift (was 0.0004)
    "gyr_w": 0.0001,    # rad/s (random walk) - increased significantly for aggressive online tuning
    "g_norm": 9.803,    # gravity magnitude
}

# Preintegration-specific noise parameters (more conservative)
# NOTE: Preintegration is MORE SENSITIVE to noise mistuning than legacy
# because errors accumulate over integration period (20-50ms typical)
# Legacy has implicit numerical damping from step-by-step updates
IMU_PARAMS_PREINT = {
    "acc_n": 0.08,      # Same measurement noise
    "gyr_n": 0.004,     # Same measurement noise
    "acc_w": 0.0001,    # 10x smaller than legacy (was 0.001) - prevent bias over-drift
    "gyr_w": 0.00002,   # 5x smaller than legacy (was 0.0001) - more conservative
    "g_norm": 9.803,
}

# EKF noises (tunable)
SIGMA_ACCEL = 0.8          # process accel magnitude (for Q)
# CRITICAL: VIO velocity noise should reflect actual measurement uncertainty
# - Homography scale has ~10-30% error depending on altitude, texture, parallax
# - Was 2.5 m/s: too conservative (over-trusts IMU, ignores vision)
# - Now 1.5 m/s: more balanced (similar to OpenVINS default for good features)
SIGMA_VO_VEL = 2.0         # m/s, VO velocity measurement stdev (TUNED for basic front-end: ORB tracks are noisy)
SIGMA_VPS_XY = 1.0         # m, VPS XY position measurement stdev (base)
SIGMA_AGL_Z = 0.8          # m, AGL height measurement stdev (if used)
SIGMA_MAG_YAW = 0.5        # rad (~28.6 degrees), magnetometer yaw measurement stdev (increased for robustness)

# VIO Quality Control
# CRITICAL: Parallax threshold must balance triangulation quality vs update frequency
# - Too high (8px): No updates → pure IMU drift
# - Too low (1px): Noisy measurements → instability
# - Sweet spot (2-3px): Regular updates with acceptable noise
# REVISED: MIN_PARALLAX_PX = 2.0 → enable updates with small motion (was 8.0)
MIN_PARALLAX_PX = 2.0      # Minimum optical flow (pixels) for VIO updates (REDUCED to enable updates - was 8.0)
MIN_MSCKF_BASELINE = 0.3   # Minimum baseline (meters) for MSCKF triangulation (INCREASED - was 0.1)

# Magnetometer calibration parameters (REVERTED - original calibration works better)
# Note: Field strength (9.1 µT) is technically incorrect, but conservative filtering
# rejects most samples, keeping only high-quality measurements → 1.5% improvement
MAG_HARD_IRON_OFFSET = np.array([0.071295, 1.002700, -7.844761], dtype=float)  # Hard-iron offset
MAG_SOFT_IRON_MATRIX = np.array([  # Soft-iron correction matrix (full 3x3)
    [3.275489, 0.082112, -0.221314],
    [0.082112, 4.173449, -0.691909],
    [-0.221314, -0.691909, 8.867694],
], dtype=float)
MAG_DECLINATION = -0.340  # radians (-19.5° for Newfoundland)
MAG_FIELD_STRENGTH = 79.8  # Expected field strength after calibration (µT)
MAG_MIN_FIELD_STRENGTH = 5.0  # Reject measurements below this (adjusted for OLD calibration ~9 µT)
MAG_MAX_FIELD_STRENGTH = 100.0  # Reject measurements above this
MAG_UPDATE_RATE_LIMIT = 5  # Apply magnetometer updates every N samples to avoid over-constraining

# VO / VIO settings
VO_MIN_INLIERS = 15  # Minimum inliers for VIO (was 10, increased to match vio_vps.py)
VO_RATIO_TEST = 0.75
# Nadir alignment threshold: if camera Z-axis aligns with world Z within this angle, use VZ-only
VO_NADIR_ALIGN_DEG = 30.0  # degrees (for nadir/downward cameras)
VO_FRONT_ALIGN_DEG = 60.0  # degrees (for forward cameras: less strict on Z alignment)

# Camera view configuration
CAMERA_VIEW_CONFIGS = {
    "nadir": {
        "extrinsics": "BODY_T_CAMDOWN",
        "nadir_threshold": 30.0,      # Strict threshold for nadir alignment
        "sigma_scale_xy": 1.5,         # Higher XY uncertainty (mainly Z motion)
        "sigma_scale_z": 0.7,          # Lower Z uncertainty (good depth info)
        "use_vz_only": True,           # Prefer VZ-only updates for nadir
        "min_parallax": 3,             # VERY REDUCED - accept very small motion for MSCKF
        "max_corners": 1500,
    },
    "front": {
        "extrinsics": "BODY_T_CAMFRONT",
        "nadir_threshold": 60.0,      # Relaxed threshold (not nadir-focused)
        "sigma_scale_xy": 0.8,         # Lower XY uncertainty (good lateral motion)
        "sigma_scale_z": 1.5,          # Higher Z uncertainty (depth ambiguity)
        "use_vz_only": False,          # Use full 3D velocity
        "min_parallax": 8,             # Lower parallax OK (more lateral motion)
        "max_corners": 2000,
    },
    "side": {
        "extrinsics": "BODY_T_CAMSIDE",
        "nadir_threshold": 60.0,
        "sigma_scale_xy": 1.0,
        "sigma_scale_z": 1.2,
        "use_vz_only": False,
        "min_parallax": 10,
        "max_corners": 2000,
    },
}

# Undistort backend: use OpenCV fisheye for KB model
USE_FISHEYE = True

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
    # Position noise from velocity integration error
    q_pos = (IMU_PARAMS['acc_n'] * dt**2 / 2)**2
    # Velocity noise from acceleration measurement noise
    q_vel = (IMU_PARAMS['acc_n'] * dt)**2
    # Rotation noise from gyro measurement noise
    q_theta = (IMU_PARAMS['gyr_n'] * dt)**2
    
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
        self.z = np.array([None]*self.dim_z).reshape(self.dim_z, 1)

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
        
        # Apply correction with proper manifold update
        self._apply_error_state_correction(dx)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        # NOTE: For ESKF, P is error-state covariance, so use error-state dimensions
        I_err = np.eye(self.P.shape[0])  # Error state identity
        I_KH = I_err - dot(self.K, H)
        self.P = dot(I_KH, self.P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

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
            self._log_likelihood = 0.0 # logpdf(x=self.y, cov=self.S)
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
            str(self.x),
            str(self.P),
            str(self.x_prior),
            str(self.P_prior),
            str(self.F),
            str(self.Q),
            str(self.R),
            str(self.K),
            str(self.y),
            str(self.S),
            str(self.likelihood),
            str(self.log_likelihood),
            str(self.mahalanobis)
            ])

# Simple WGS84 approximation
R_EARTH = 6378137.0

def latlon_to_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> np.ndarray:
    dlat = np.radians(lat - origin_lat)
    dlon = np.radians(lon - origin_lon)
    lat0_rad = np.radians(origin_lat)
    
    x = R_EARTH * dlon * np.cos(lat0_rad)
    y = R_EARTH * dlat
    return np.array([x, y], dtype=float)

def xy_to_latlon(px: float, py: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    lat0_rad = np.radians(origin_lat)
    dlat = py / R_EARTH
    dlon = px / (R_EARTH * np.cos(lat0_rad))
    
    lat = origin_lat + np.degrees(dlat)
    lon = origin_lon + np.degrees(dlon)
    return float(lat), float(lon)

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
    ang: np.ndarray  # [wx,wy,wz] rad/s (unused for now)
    lin: np.ndarray  # [ax,ay,az] m/s^2 (body)

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
# ===============================
# VIO front-end (OpenVINS-style: Grid-based KLT)
# ===============================

def make_KD_for_size(kb: dict, dst_w: int, dst_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Kannala-Brandt fisheye parameters to camera matrix K and distortion D at runtime size."""
    sx = dst_w / float(kb["w"])  # scale from calib size → runtime size
    sy = dst_h / float(kb["h"])
    fx = kb["mu"] * sx
    fy = kb["mv"] * sy
    cx = kb["u0"] * sx
    cy = kb["v0"] * sy
    K = np.array([[fx, 0,  cx], [0,  fy, cy], [0, 0, 1.0]], dtype=np.float64)
    D = np.array([kb["k2"], kb["k3"], kb["k4"], kb["k5"]], dtype=np.float64)
    return K, D

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
        # pts: Nx2
        if self.use_fisheye:
            undist = cv2.fisheye.undistortPoints(pts.reshape(-1,1,2), self.K, self.D)
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
            # Note: pts are normalized, so K is identity
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


def detect_zupt(imu_window, threshold_accel=0.5, threshold_gyro=0.1):
    """
    Detect Zero Velocity Update (ZUPT) condition from IMU window.
    Returns True if vehicle is stationary.
    """
    if len(imu_window) < 10:
        return False
    
    accels = np.array([rec.lin for rec in imu_window])
    gyros = np.array([rec.ang for rec in imu_window])
    
    accel_var = np.var(accels, axis=0).sum()
    gyro_var = np.var(gyros, axis=0).sum()
    
    return (accel_var < threshold_accel**2) and (gyro_var < threshold_gyro**2)



# ===============================
# Ground Truth Loading & Error Calculation
# ===============================

def load_ground_truth_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load ground truth from flight_log_from_gga.csv.
    
    CSV format:
        stamp_log, lat_dd, lon_dd, altitude_MSL_m, xSpeed_mph, ySpeed_mph, zSpeed_mph
    
    Args:
        csv_path: Path to flight_log_from_gga.csv
        
    Returns:
        DataFrame with columns: [timestamp, lat, lon, alt_msl, vx_mph, vy_mph, vz_mph]
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Verify required columns
        required_cols = ['stamp_log', 'lat_dd', 'lon_dd', 'altitude_MSL_m']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Rename for consistency
        df = df.rename(columns={
            'stamp_log': 'timestamp',
            'lat_dd': 'lat',
            'lon_dd': 'lon',
            'altitude_MSL_m': 'alt_msl'
        })
        
        # Optional velocity columns (may not exist in all datasets)
        if 'xSpeed_mph' in df.columns:
            df = df.rename(columns={
                'xSpeed_mph': 'vx_mph',
                'ySpeed_mph': 'vy_mph',
                'zSpeed_mph': 'vz_mph'
            })
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\n[GT] Loaded {len(df)} ground truth samples from {csv_path}")
        print(f"[GT] Time range: {df['timestamp'].min():.2f} - {df['timestamp'].max():.2f} s")
        print(f"[GT] Lat range: {df['lat'].min():.6f}° - {df['lat'].max():.6f}°")
        print(f"[GT] Lon range: {df['lon'].min():.6f}° - {df['lon'].max():.6f}°")
        print(f"[GT] Alt range: {df['alt_msl'].min():.1f} - {df['alt_msl'].max():.1f} m MSL")
        
        return df
    
    except FileNotFoundError:
        print(f"[WARNING] Ground truth file not found: {csv_path}")
        return None
    except Exception as e:
        print(f"[WARNING] Failed to load ground truth: {e}")
        return None


def compute_trajectory_errors_against_gt(
    pose_csv_path: str,
    gt_df: pd.DataFrame,
    origin_lat: float,
    origin_lon: float,
    output_csv_path: str,
    t_start_epoch: float = 0.0
) -> Optional[pd.DataFrame]:
    """
    Compute trajectory errors between estimated poses and ground truth.
    
    Args:
        pose_csv_path: Path to output pose.csv
        gt_df: Ground truth DataFrame from load_ground_truth_from_csv()
        origin_lat, origin_lon: Local coordinate origin (degrees)
        output_csv_path: Path to save error CSV
        
    Returns:
        DataFrame with error metrics, or None if failed
    """
    if gt_df is None or len(gt_df) == 0:
        print("[ERROR] No ground truth data available for error calculation")
        return None
    
    try:
        # Load estimated trajectory
        est_df = pd.read_csv(pose_csv_path)
        print(f"\n[ERROR_CALC] Loaded {len(est_df)} estimated poses from {pose_csv_path}")
        
        # Convert GT lat/lon to ENU coordinates
        gt_df = gt_df.copy()
        gt_xy = []
        for _, row in gt_df.iterrows():
            xy = latlon_to_xy(row['lat'], row['lon'], origin_lat, origin_lon)
            gt_xy.append(xy)
        
        gt_df['gt_px'] = [xy[0] for xy in gt_xy]
        gt_df['gt_py'] = [xy[1] for xy in gt_xy]
        gt_df['gt_pz'] = gt_df['alt_msl']
        
        # Get estimated positions (already in ENU from pose.csv)
        if 'px' in est_df.columns and 'py' in est_df.columns:
            est_df['est_px'] = est_df['px']
            est_df['est_py'] = est_df['py']
        else:
            print("[ERROR] Estimated trajectory must contain px, py columns")
            return None
        
        if 'pz' in est_df.columns:
            est_df['est_pz'] = est_df['pz']
        else:
            print("[WARNING] No pz in estimated trajectory, using 0")
            est_df['est_pz'] = 0.0
        
        # Interpolate GT to match estimated timestamps
        from scipy.interpolate import interp1d
        
        gt_times = gt_df['timestamp'].values
        interp_px = interp1d(gt_times, gt_df['gt_px'].values, 
                             kind='linear', fill_value='extrapolate')
        interp_py = interp1d(gt_times, gt_df['gt_py'].values, 
                             kind='linear', fill_value='extrapolate')
        interp_pz = interp1d(gt_times, gt_df['gt_pz'].values, 
                             kind='linear', fill_value='extrapolate')
        
        # Interpolate to estimated timestamps
        est_times = est_df['t'].values + t_start_epoch  # pose.csv uses 't' (relative), add start epoch
        est_df['gt_px_interp'] = interp_px(est_times)
        est_df['gt_py_interp'] = interp_py(est_times)
        est_df['gt_pz_interp'] = interp_pz(est_times)
        
        # Compute errors
        est_df['error_px'] = est_df['est_px'] - est_df['gt_px_interp']
        est_df['error_py'] = est_df['est_py'] - est_df['gt_py_interp']
        est_df['error_pz'] = est_df['est_pz'] - est_df['gt_pz_interp']
        
        # 2D horizontal error (XY plane)
        est_df['error_2d'] = np.sqrt(est_df['error_px']**2 + est_df['error_py']**2)
        
        # 3D error
        est_df['error_3d'] = np.sqrt(
            est_df['error_px']**2 + 
            est_df['error_py']**2 + 
            est_df['error_pz']**2
        )
        
        # Print summary statistics
        print("\n" + "="*60)
        print("TRAJECTORY ERROR STATISTICS")
        print("="*60)
        print(f"Number of samples: {len(est_df)}")
        print(f"\n2D Horizontal Error (XY plane):")
        print(f"  Mean:   {est_df['error_2d'].mean():.3f} m")
        print(f"  Median: {est_df['error_2d'].median():.3f} m")
        print(f"  Std:    {est_df['error_2d'].std():.3f} m")
        print(f"  Max:    {est_df['error_2d'].max():.3f} m")
        print(f"  RMSE:   {np.sqrt((est_df['error_2d']**2).mean()):.3f} m")
        
        print(f"\n3D Position Error:")
        print(f"  Mean:   {est_df['error_3d'].mean():.3f} m")
        print(f"  Median: {est_df['error_3d'].median():.3f} m")
        print(f"  Std:    {est_df['error_3d'].std():.3f} m")
        print(f"  Max:    {est_df['error_3d'].max():.3f} m")
        print(f"  RMSE:   {np.sqrt((est_df['error_3d']**2).mean()):.3f} m")
        
        print(f"\nPer-Axis Errors:")
        print(f"  X (East):  mean={est_df['error_px'].mean():+.3f} m, std={est_df['error_px'].std():.3f} m")
        print(f"  Y (North): mean={est_df['error_py'].mean():+.3f} m, std={est_df['error_py'].std():.3f} m")
        print(f"  Z (Up):    mean={est_df['error_pz'].mean():+.3f} m, std={est_df['error_pz'].std():.3f} m")
        print("="*60 + "\n")
        
        # Save error CSV
        columns_to_save = [
            't', # Changed from 'time' to 't'
            'est_px', 'est_py', 'est_pz',
            'gt_px_interp', 'gt_py_interp', 'gt_pz_interp',
            'error_px', 'error_py', 'error_pz',
            'error_2d', 'error_3d'
        ]
        available_cols = [col for col in columns_to_save if col in est_df.columns]
        est_df[available_cols].to_csv(output_csv_path, index=False, float_format='%.6f')
        print(f"[SAVE] Error metrics saved to: {output_csv_path}\n")
        
        return est_df
        
    except Exception as e:
        print(f"[ERROR] Failed to compute trajectory errors: {e}")
        import traceback
        traceback.print_exc()
        return None


def run(imu_path, quarry_path, output_dir,
        images_dir=None, images_index_csv=None,
        downscale_size=(1440, 1080),
        camera_view="nadir"):
    """
    Streamlined VIO runner with IMU preintegration + VIO frontend + MSCKF + ZUPT.
    Uses ONLY preintegration mode (legacy mode removed).
    Removed: VPS, Magnetometer, DEM updates, legacy propagation.
    """
    import os
    import time
    import csv
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Output files
    pose_csv = os.path.join(output_dir, "pose.csv")
    error_csv = os.path.join(output_dir, "error_log.csv")
    
    # Initialize CSV headers
    with open(pose_csv, "w", newline="") as f:
        f.write("t,dt,Frame,px,py,pz,vx,vy,vz,lat,lon\n")
    
    with open(error_csv, "w", newline="") as f:
        f.write("t,pos_error_m,err_E,err_N,err_U,vel_error_m_s\n")
    
    # Load inputs
    print("\n=== Loading Inputs ===")
    imu = load_imu_csv(imu_path)
    print(f"[IMU] Loaded {len(imu)} samples")
    
    lat0, lon0, msl0_m, v_init = load_quarry_initial(quarry_path)
    print(f"[Init] lat={lat0:.6f}, lon={lon0:.6f}, MSL={msl0_m:.2f}m")
    
    imgs = load_images(images_dir, images_index_csv) if images_dir else []
    print(f"[Images] Loaded {len(imgs)} images")
    
    # Initialize VIO frontend
    vio_fe = None
    if len(imgs) > 0:
        img_w, img_h = downscale_size
        # Precompute camera intrinsics at runtime size
        K, D = make_KD_for_size(KB_PARAMS, img_w, img_h)
        vio_fe = VIOFrontEnd(img_w, img_h, K, D, use_fisheye=USE_FISHEYE)
        print(f"[VIO] Initialized frontend (camera_view={camera_view})")

    
    # Initialize EKF (core state only: 16D nominal, 15D error)
    nom_dim = 16
    err_dim = 15
    kf = ExtendedKalmanFilter(nom_dim, 3)
    
    # Set initial state
    kf.x = np.zeros((nom_dim, 1))
    kf.x[0:3, 0] = [0.0, 0.0, msl0_m]  # Position (local ENU origin)
    kf.x[3:6, 0] = v_init  # Velocity
    kf.x[6:10, 0] = imu[0].q[[3,0,1,2]]  # Quaternion [w,x,y,z]
    kf.x[10:13, 0] = [0.0, 0.0, 0.0]  # Gyro bias
    kf.x[13:16, 0] = [0.0, 0.0, 0.0]  # Accel bias
    
    # Initial covariance
    kf.P = np.eye(err_dim) * 0.01
    kf.P[0:3, 0:3] *= 10.0  # Position uncertainty
    kf.P[3:6, 3:6] *= 1.0   # Velocity uncertainty
    kf.P[6:9, 6:9] *= 0.1   # Rotation uncertainty
    
    # IMU preintegration (ONLY mode - no legacy)
    preint = IMUPreintegration(
        bg=kf.x[10:13, 0],
        ba=kf.x[13:16, 0],
        sigma_g=IMU_PARAMS_PREINT['gyr_n'],
        sigma_a=IMU_PARAMS_PREINT['acc_n'],
        sigma_bg=IMU_PARAMS_PREINT['gyr_w'],
        sigma_ba=IMU_PARAMS_PREINT['acc_w']
    )
    
    # Main loop
    print("\n=== Processing ===")
    t0 = imu[0].t
    img_idx = 0
    vio_frame = -1
    zupt_count = 0
    zupt_applied = 0
    imu_window = []
    last_preint_time = t0
    
    for i, imu_rec in enumerate(imu):
        t = imu_rec.t
        dt = t - imu[i-1].t if i > 0 else 0.0
        
        # Extract current state
        q = kf.x[6:10, 0]  # Quaternion [w,x,y,z]
        v = kf.x[3:6, 0]   # Velocity
        p = kf.x[0:3, 0]   # Position
        bg = kf.x[10:13, 0]  # Gyro bias
        ba = kf.x[13:16, 0]  # Accel bias
        
        # Bias-corrected IMU measurements
        w_corr = imu_rec.ang - bg
        a_corr = imu_rec.lin - ba
        
        # Rotation matrix (body to world)
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
        R_bw = R_scipy.from_quat(q_xyzw).as_matrix()
        
        # ========== IMU PROPAGATION (Preintegration ONLY) ==========
        if i > 0:
            # Accumulate IMU measurements in preintegration
            preint.integrate_measurement(imu_rec.ang, imu_rec.lin, dt)
            
            # Apply preintegration every 20ms or at image timestamps
            apply_preint = False
            if (t - last_preint_time) >= 0.02:  # 20ms = 50Hz
                apply_preint = True
            elif vio_fe and img_idx < len(imgs) and abs(imgs[img_idx].t - t) < 0.01:
                apply_preint = True
            
            if apply_preint:
                # Get bias-corrected deltas
                delta_R, delta_v, delta_p = preint.get_deltas_corrected(bg, ba)
                
                # CRITICAL FIX: Gravity compensation
                # Preintegration already handled gravity in body frame
                # Now we apply deltas and add gravity in world frame
                
                # Rotation: q_new = q_old ⊗ delta_q
                delta_q = rot_to_quat(delta_R)
                q_new = quat_multiply(q, delta_q)
                kf.x[6:10, 0] = quat_normalize(q_new)
                
                # Velocity: v_new = v_old + R_old @ delta_v + g_world * dt
                # Note: delta_v from preintegration is in body frame, already gravity-compensated
                R_old = quat_to_rot(q)
                g_world = np.array([0.0, 0.0, -IMU_PARAMS['g_norm']])  # ENU: gravity points down
                kf.x[3:6, 0] = v + R_old @ delta_v + g_world * preint.dt_sum
                
                # Position: p_new = p_old + v_old * dt + R_old @ delta_p + 0.5 * g_world * dt^2
                kf.x[0:3, 0] = p + v * preint.dt_sum + R_old @ delta_p + 0.5 * g_world * (preint.dt_sum ** 2)
                
                # Propagate covariance
                Phi_err = compute_error_state_jacobian(q, a_corr, w_corr, preint.dt_sum, R_bw)
                Q_err = compute_error_state_process_noise(preint.dt_sum, False, t, t0)
                num_clones = (kf.x.shape[0] - 16) // 7
                kf.P = propagate_error_state_covariance(kf.P, Phi_err, Q_err, num_clones)
                
                # Reset preintegration
                preint.reset(bg, ba)
                last_preint_time = t
        
        # ========== VIO UPDATE ==========
        if vio_fe and img_idx < len(imgs) and abs(imgs[img_idx].t - t) < 0.01:
            img = cv2.imread(imgs[img_idx].path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, downscale_size)
                ok, ninl, R_vo, t_unit, dt_img = vio_fe.step(img, t)
                
                if ok and R_vo is not None and t_unit is not None:
                    # Camera extrinsics
                    R_cam_to_body = BODY_T_CAMDOWN[:3, :3].T
                    
                    # Check nadir alignment
                    zn = np.array([0, 0, 1.0], dtype=float)
                    t_norm = t_unit / (np.linalg.norm(t_unit) + 1e-12)
                    alignment_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(t_norm, zn)), -1.0, 1.0))))
                    
                    # Get camera view config
                    view_cfg = CAMERA_VIEW_CONFIGS.get(camera_view, CAMERA_VIEW_CONFIGS['nadir'])
                    use_only_vz = (alignment_deg < view_cfg['nadir_threshold']) and view_cfg['use_vz_only']
                    
                    # Map camera unit direction → body → world
                    t_body = R_cam_to_body @ t_norm
                    
                    # Get focal length for flow computation
                    focal_px = KB_PARAMS['mu']
                    
                    # Compute average optical flow from tracked features
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
                            median_flow_vec = np.median(flows, axis=0)  # Median of [dx, dy] in pixels
                    
                    # Check parallax threshold
                    if avg_flow_px < MIN_PARALLAX_PX:
                        print(f"[DEBUG][VIO] SKIPPING VIO velocity update: insufficient parallax (flow={avg_flow_px:.2f}px < {MIN_PARALLAX_PX}px)")
                        img_idx += 1
                        continue
                    
                    # Estimate AGL (heuristic: 30m if no DEM available)
                    agl_temp = 30.0
                    
                    # Estimate feature depth from AGL and camera pitch
                    cam_z_world = R_bw @ (R_cam_to_body @ np.array([0, 0, 1]))
                    pitch_rad = np.arcsin(-cam_z_world[2])
                    cos_pitch = max(0.3, np.cos(pitch_rad))
                    depth_est = agl_temp / cos_pitch
                    
                    # === Scale Recovery: Two methods ===
                    # Method 1: Optical flow-based
                    if dt_img > 1e-4 and avg_flow_px > 2.0:
                        scale_flow = depth_est / focal_px
                        speed_from_flow = (avg_flow_px / dt_img) * scale_flow
                    else:
                        speed_from_flow = 0.0
                    
                    # Method 2: Homography-based (for nadir + flat terrain)
                    speed_from_homography = 0.0
                    use_homography_scale = False
                    
                    if camera_view == 'nadir' and vio_fe.last_matches is not None:
                        pts_prev, pts_cur = vio_fe.last_matches
                        if len(pts_prev) >= 15:
                            # Build camera intrinsic matrix at normalized coordinates
                            K_norm = np.eye(3, dtype=float)
                            
                            # Estimate scale from homography
                            try:
                                # Compute homography
                                H, h_mask = cv2.findHomography(pts_prev, pts_cur, cv2.RANSAC, 3e-3)
                                if H is not None and h_mask is not None:
                                    num_h_inliers = int(np.sum(h_mask))
                                    if num_h_inliers >= 15:
                                        # Decompose homography to get scale
                                        # For nadir camera: scale ≈ altitude / (altitude - translation_z)
                                        # Simplified: use homography translation magnitude
                                        t_homog = H[:3, 2] / (H[2, 2] + 1e-9)
                                        scale_homog = np.linalg.norm(t_homog) * agl_temp
                                        
                                        # Apply scale to Essential matrix translation
                                        t_scaled = t_unit * scale_homog
                                        
                                        if dt_img > 1e-4:
                                            # Transform scaled translation to body frame
                                            t_body_homog = R_cam_to_body @ t_scaled
                                            # Transform to world frame
                                            t_world_homog = R_bw @ t_body_homog
                                            # Velocity = displacement / time
                                            speed_from_homography = np.linalg.norm(t_world_homog) / dt_img
                                            
                                            if speed_from_homography < 50.0:
                                                use_homography_scale = True
                                                print(f"[HOMOGRAPHY] Inliers={num_h_inliers}, scale={scale_homog:.2f}m, "
                                                      f"speed={speed_from_homography:.2f}m/s, flow_speed={speed_from_flow:.2f}m/s")
                            except Exception as e:
                                print(f"[HOMOGRAPHY] Failed: {e}")
                    
                    # Choose scale method
                    if use_homography_scale and speed_from_homography > 0.1:
                        speed_final = speed_from_homography
                        scale_method = "homography"
                    elif speed_from_flow > 0.1:
                        speed_final = speed_from_flow
                        scale_method = "flow"
                    else:
                        speed_final = 0.0
                        scale_method = "none"
                    
                    # Sanity check
                    if speed_final > 50.0:
                        print(f"[DEBUG][VIO] WARNING: Computed velocity too high ({speed_final:.1f} m/s), clamping to 50 m/s")
                        speed_final = 50.0
                    
                    print(f"[SCALE] Method={scale_method}, speed={speed_final:.2f}m/s, parallax={avg_flow_px:.1f}px")
                    
                    # Use optical flow direction from normalized coordinates
                    if avg_flow_px > 2.0 and vio_fe.last_matches is not None:
                        pts_prev, pts_cur = vio_fe.last_matches
                        if len(pts_prev) > 0 and len(pts_cur) > 0:
                            # Compute flow in normalized camera coordinates
                            flows_normalized = pts_cur - pts_prev
                            median_flow_normalized = np.median(flows_normalized, axis=0)
                            
                            flow_norm = np.linalg.norm(median_flow_normalized)
                            if flow_norm > 1e-6:
                                flow_dir_normalized = median_flow_normalized / flow_norm
                            else:
                                flow_dir_normalized = np.array([0.0, 1.0])
                            
                            # Normalized camera coordinates: [x_norm, y_norm, 0.0] in Z=1 plane
                            vel_cam = np.array([flow_dir_normalized[0], flow_dir_normalized[1], 0.0])
                            vel_cam = vel_cam / (np.linalg.norm(vel_cam) + 1e-9)
                            
                            # Transform to body frame
                            vel_body = R_cam_to_body @ vel_cam
                            vel_body = vel_body * speed_final
                        else:
                            vel_body = t_body * speed_final
                    else:
                        vel_body = t_body * speed_final
                    
                    vel_world = R_bw @ vel_body
                    
                    vel_vx, vel_vy, vel_vz = float(vel_world[0]), float(vel_world[1]), float(vel_world[2])
                    
                    print(f"[DEBUG][VIO] flow_px={avg_flow_px:.2f}, dt={dt_img:.4f}, AGL={agl_temp:.2f}m, speed={speed_final:.3f}m/s ({scale_method})")
                    print(f"[DEBUG][VIO] vel_world: [{vel_vx:.3f}, {vel_vy:.3f}, {vel_vz:.3f}] m/s")
                    
                    # ESKF velocity update with adaptive uncertainty and gating
                    num_clones = (kf.x.shape[0] - 16) // 7
                    err_dim = 15 + 6 * num_clones
                    
                    if use_only_vz:
                        h_vel = np.zeros((1, err_dim), dtype=float)
                        h_vel[0, 5] = 1.0  # δv_z at index 5
                        vel_meas = np.array([[vel_world[2]]])
                        print(f"[DEBUG][VIO] t={t:.3f} vel_meas={vel_meas.flatten()} alignment_deg={alignment_deg:.2f} use_only_vz={use_only_vz}")
                    else:
                        h_vel = np.zeros((3, err_dim), dtype=float)
                        h_vel[0, 3] = 1.0  # δv_x at index 3
                        h_vel[1, 4] = 1.0  # δv_y at index 4
                        h_vel[2, 5] = 1.0  # δv_z at index 5
                        vel_meas = vel_world.reshape(-1, 1)
                        print(f"[DEBUG][VIO] t={t:.3f} vel_meas={vel_meas.flatten()} alignment_deg={alignment_deg:.2f} use_only_vz={use_only_vz}")
                    
                    def h_fun(x, h=h_vel):
                        return h
                    
                    def hx_fun(x, h=h_vel):
                        if use_only_vz:
                            return x[5:6].reshape(1, 1)
                        else:
                            return x[3:6].reshape(3, 1)
                    
                    # Adaptive uncertainty
                    align_scale = 1.0 + alignment_deg / 45.0
                    flow_scale = 1.0 + max(0.0, (avg_flow_px - 10.0) / 20.0)
                    uncertainty_scale = align_scale * flow_scale
                    
                    if use_only_vz:
                        r_mat = np.array([[(SIGMA_VO_VEL * view_cfg['sigma_scale_z'] * uncertainty_scale)**2]])
                    else:
                        r_mat = np.diag([
                            (SIGMA_VO_VEL * view_cfg['sigma_scale_xy'] * uncertainty_scale)**2,
                            (SIGMA_VO_VEL * view_cfg['sigma_scale_xy'] * uncertainty_scale)**2,
                            (SIGMA_VO_VEL * view_cfg['sigma_scale_z'] * uncertainty_scale)**2
                        ])
                    
                    # Innovation gating
                    s_mat = h_vel @ kf.P @ h_vel.T + r_mat
                    try:
                        if use_only_vz:
                            predicted_vel = kf.x[5:6, 0].reshape(1, 1)
                        else:
                            predicted_vel = kf.x[3:6, 0].reshape(3, 1)
                        innovation = vel_meas - predicted_vel
                        m2_test = _mahalanobis2(innovation, s_mat)
                    except Exception:
                        m2_test = np.inf
                    
                    # Chi-square test (99%)
                    chi2_threshold = 6.63 if use_only_vz else 11.34
                    
                    if m2_test < chi2_threshold:
                        kf.update(
                            z=vel_meas,
                            HJacobian=h_fun,
                            Hx=hx_fun,
                            R=r_mat
                        )
                        vio_frame += 1
                        print(f"[DEBUG][VIO] VIO velocity update APPLIED at t={t:.3f} with {ninl} inliers, innovation={innovation.flatten()}, m2={m2_test:.2f}")
                    else:
                        print(f"[DEBUG][VIO] VIO velocity update REJECTED at t={t:.3f}, m2={m2_test:.2f} > threshold={chi2_threshold:.2f}")
            
            img_idx += 1
        
        # ========== ZUPT UPDATE ==========
        imu_window.append(imu_rec)
        if len(imu_window) > 20:
            imu_window.pop(0)
        
        if len(imu_window) >= 10 and detect_zupt(imu_window):
            zupt_count += 1
            
            # Apply zero velocity constraint
            num_clones = (kf.x.shape[0] - 16) // 7
            err_dim = 15 + 6 * num_clones
            
            H_zupt = np.zeros((3, err_dim), dtype=float)
            H_zupt[0:3, 3:6] = np.eye(3)
            
            def h_zupt_jac(x, h=H_zupt):
                return h
            
            def hx_zupt(x):
                return x[3:6].reshape(3, 1)
            
            R_zupt = np.eye(3) * 0.01  # Very low uncertainty (confident in zero velocity)
            
            # Innovation gating
            v_current = kf.x[3:6, 0]
            if np.linalg.norm(v_current) < 5.0:  # Only apply if velocity is reasonable
                try:
                    kf.update(
                        z=np.zeros((3, 1)),
                        HJacobian=h_zupt_jac,
                        Hx=hx_zupt,
                        R=R_zupt
                    )
                    zupt_applied += 1
                except:
                    pass
        
        # ========== LOGGING ==========
        lat_now, lon_now = xy_to_latlon(kf.x[0,0], kf.x[1,0], lat0, lon0)
        with open(pose_csv, "a") as f:
            f.write(f"{t-t0:.6f},{dt:.6f},{vio_frame if vio_frame>=0 else ''},")
            f.write(f"{kf.x[0,0]:.3f},{kf.x[1,0]:.3f},{kf.x[2,0]:.3f},")
            f.write(f"{kf.x[3,0]:.3f},{kf.x[4,0]:.3f},{kf.x[5,0]:.3f},")
            f.write(f"{lat_now:.8f},{lon_now:.8f}\n")
        
        if i % 100 == 0:
            print(f"t={t-t0:.2f}s | pos=({kf.x[0,0]:.1f},{kf.x[1,0]:.1f},{kf.x[2,0]:.1f}) | "
                  f"vel=({kf.x[3,0]:.1f},{kf.x[4,0]:.1f},{kf.x[5,0]:.1f})", end="\r")
    
    
    print(f"\n\nDone! Processed {len(imu)} IMU samples, {vio_frame+1 if vio_frame>=0 else 0} VIO frames, "
          f"{zupt_applied}/{zupt_count} ZUPT updates applied")
    print(f"Output: {pose_csv}, {error_csv}")
    
    # ========== GROUND TRUTH ERROR CALCULATION ==========
    print("\n" + "="*60)
    print("GROUND TRUTH ERROR CALCULATION")
    print("="*60)
    
    # Load ground truth from the same quarry_path
    gt_df = load_ground_truth_from_csv(quarry_path)
    
    # Compute errors if ground truth is available
    if gt_df is not None:
        compute_trajectory_errors_against_gt(
            pose_csv_path=pose_csv,
            gt_df=gt_df,
            origin_lat=lat0,
            origin_lon=lon0,
            output_csv_path=error_csv,
            t_start_epoch=t0
        )
    else:
        print("[WARNING] Skipping error calculation (no ground truth available)\n")



if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Streamlined VIO with IMU preintegration (preintegration-only mode)")
    ap.add_argument("--imu", required=True, help="Path to IMU CSV file")
    ap.add_argument("--quarry", required=True, help="Path to flight_log_from_gga.csv")
    ap.add_argument("--output", default="out_vio_streamlined", help="Output directory")
    ap.add_argument("--images_dir", default=None, help="Directory containing images")
    ap.add_argument("--images_index", default=None, help="CSV with image timestamps")
    ap.add_argument("--img_w", type=int, default=1440, help="Image width for processing")
    ap.add_argument("--img_h", type=int, default=1080, help="Image height for processing")
    ap.add_argument("--camera_view", choices=["nadir","front","side"], default="nadir", help="Camera orientation")
    
    args = ap.parse_args()
    
    run(
        imu_path=args.imu,
        quarry_path=args.quarry,
        output_dir=args.output,
        images_dir=args.images_dir,
        images_index_csv=args.images_index,
        downscale_size=(args.img_w, args.img_h),
        camera_view=args.camera_view
    )

