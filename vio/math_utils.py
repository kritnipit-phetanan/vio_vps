#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIO Math Utilities Module
=========================

Contains quaternion operations, rotation matrices, and mathematical helpers
for Visual-Inertial Odometry.

Quaternion Convention:
----------------------
All quaternions use Hamilton convention with [w, x, y, z] ordering:
- w is the scalar (real) part
- [x, y, z] is the vector (imaginary) part
- q = w + xi + yj + zk

Quaternion represents rotation R such that:
    v' = q ⊗ v ⊗ q* (passive/frame rotation)
    
where v is a pure quaternion [0, vx, vy, vz] and q* is the conjugate.

Frame Conventions:
------------------
- ENU (East-North-Up): X=East, Y=North, Z=Up
  Used by Xsens IMU output
  
- NED (North-East-Down): X=North, Y=East, Z=Down  
  Used by aviation/aerospace convention
  
- FRD (Forward-Right-Down): Body frame for aircraft
  X=Forward, Y=Right, Z=Down

Key Operations:
---------------
- quat_multiply: Hamilton quaternion product
- quat_normalize: Ensure unit quaternion
- quat_to_rot: Convert to 3x3 rotation matrix
- rot_to_quat: Convert rotation matrix to quaternion
- quat_boxplus: Quaternion ⊞ rotation vector (perturbation)
- quat_boxminus: Extract rotation vector between quaternions
- skew_symmetric: Create 3x3 skew-symmetric matrix for cross product

Author: VIO project
"""

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy


# =============================================================================
# Frame Transformation Constants
# =============================================================================

# ENU→NED frame transformation matrix
# ENU: X=East, Y=North, Z=Up (Xsens IMU uses this)
# NED: X=North, Y=East, Z=Down (our state uses this)
# Transform: x_ned = y_enu, y_ned = x_enu, z_ned = -z_enu
R_NED_FROM_ENU = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1]
], dtype=np.float64)


# =============================================================================
# Quaternion Operations (all use [w, x, y, z] Hamilton convention)
# =============================================================================

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion multiplication: q1 ⊗ q2, both in [w,x,y,z] format.
    
    Implements Hamilton product following right-hand convention:
    q1 ⊗ q2 represents rotation q2 followed by rotation q1.
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        
    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion to unit length.
    
    Unit quaternions (||q|| = 1) represent valid rotations.
    Numerical errors can cause drift from unit norm.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Normalized quaternion with ||q|| = 1
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse (conjugate for unit quaternion)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_enu_to_ned(q_enu_xyzw: np.ndarray) -> np.ndarray:
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
    r_body_enu = R_scipy.from_quat(q_enu_xyzw).as_matrix()
    
    # Convert to NED: R_body_ned = R_NED_FROM_ENU @ R_body_enu
    r_body_ned = R_NED_FROM_ENU @ r_body_enu
    
    # Convert back to quaternion [x,y,z,w]
    q_ned_xyzw = R_scipy.from_matrix(r_body_ned).as_quat()
    
    return q_ned_xyzw


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
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


def small_angle_quat(dtheta: np.ndarray) -> np.ndarray:
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


def quat_boxplus(q: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """
    Quaternion box-plus operation (manifold update).
    q_new = q ⊕ δθ = q ⊗ exp(δθ)
    where exp(δθ) converts 3D rotation vector to quaternion.
    """
    dq = small_angle_quat(dtheta)
    return quat_normalize(quat_multiply(q, dq))


def quat_boxminus(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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


def quaternion_to_yaw(q_wxyz: np.ndarray) -> float:
    """
    Extract yaw angle from quaternion [w,x,y,z] in ENU frame.
    Yaw = rotation around Z-axis (Up in ENU)
    
    Returns:
        Yaw angle in radians [-π, π], where 0 = East, π/2 = North
    """
    w, x, y, z = q_wxyz
    # Using atan2 formula for yaw from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion multiplication: q1 * q2
    Both quaternions in [w, x, y, z] format
    Result represents rotation q2 applied first, then q1
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])


def yaw_to_quaternion_update(yaw_current: float, yaw_measured: float) -> np.ndarray:
    """
    Create a quaternion that rotates from current yaw to measured yaw.
    
    Args:
        yaw_current: Current yaw angle in radians
        yaw_measured: Target yaw angle in radians
    
    Returns:
        Quaternion [w, x, y, z] representing the rotation correction
    """
    # Compute yaw difference and wrap to [-π, π]
    dyaw = yaw_measured - yaw_current
    while dyaw > np.pi:
        dyaw -= 2 * np.pi
    while dyaw < -np.pi:
        dyaw += 2 * np.pi
    
    # Create quaternion for rotation around Z-axis by dyaw
    half_dyaw = dyaw / 2
    return np.array([
        np.cos(half_dyaw),
        0.0,
        0.0,
        np.sin(half_dyaw)
    ])


# =============================================================================
# Matrix Operations
# =============================================================================

def skew_symmetric(v: np.ndarray) -> np.ndarray:
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


def angle_wrap(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


# =============================================================================
# Mahalanobis Distance
# =============================================================================

def mahalanobis_squared(y: np.ndarray, S: np.ndarray) -> float:
    """
    Compute squared Mahalanobis distance: d² = y^T @ S^{-1} @ y
    Uses Cholesky decomposition for numerical stability.
    
    Args:
        y: Innovation vector
        S: Innovation covariance matrix
    
    Returns:
        Squared Mahalanobis distance
    """
    try:
        L = np.linalg.cholesky(S)
        z = np.linalg.solve(L, y)
        return float(np.dot(z, z))
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if Cholesky fails
        return float(y @ np.linalg.pinv(S) @ y)
