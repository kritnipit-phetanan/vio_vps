#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Model Helpers Module

Contains Kannala-Brandt fisheye camera model functions and related utilities.

Author: VIO project
"""

import numpy as np
from typing import Tuple


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
    """
    Create K and D matrices for a given image size from calibration parameters.
    
    Args:
        kb: Calibration dict with keys: 'mu', 'mv', 'u0', 'v0', 'w', 'h', 'k2', 'k3', 'k4', 'k5'
        dst_w: Target image width
        dst_h: Target image height
    
    Returns:
        K: 3x3 camera intrinsic matrix
        D: 4-element distortion coefficients
    """
    sx = dst_w / float(kb["w"])  # scale from calib size → runtime size
    sy = dst_h / float(kb["h"])
    fx = kb["mu"] * sx
    fy = kb["mv"] * sy
    cx = kb["u0"] * sx
    cy = kb["v0"] * sy
    K = np.array([[fx, 0,  cx], [0,  fy, cy], [0, 0, 1.0]], dtype=np.float64)
    D = np.array([kb["k2"], kb["k3"], kb["k4"], kb["k5"]], dtype=np.float64)
    return K, D


def project_point_to_normalized(p_camera: np.ndarray) -> np.ndarray:
    """
    Project 3D point in camera frame to normalized image coordinates.
    
    Args:
        p_camera: 3D point in camera frame [x, y, z]
    
    Returns:
        2D normalized coordinates [x/z, y/z]
    """
    if p_camera[2] <= 0:
        return None
    
    return np.array([p_camera[0] / p_camera[2], p_camera[1] / p_camera[2]])


def kannala_brandt_project(pts_3d: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Project 3D points to 2D pixels using Kannala-Brandt model.
    
    Args:
        pts_3d: Nx3 array of 3D points in camera frame
        K: 3x3 camera intrinsic matrix
        D: 4-element distortion coefficients [k2, k3, k4, k5]
    
    Returns:
        Nx2 array of pixel coordinates
    """
    if pts_3d.size == 0:
        return pts_3d[:, :2] if pts_3d.ndim == 2 else np.empty((0, 2))
    
    pts_3d = pts_3d.reshape(-1, 3)
    
    # Extract intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # K-B coefficients
    k1 = 1.0
    k2 = D[0] if len(D) > 0 else 0.0
    k3 = D[1] if len(D) > 1 else 0.0
    k4 = D[2] if len(D) > 2 else 0.0
    k5 = D[3] if len(D) > 3 else 0.0
    
    # Compute theta (angle from optical axis)
    r_3d = np.sqrt(pts_3d[:, 0]**2 + pts_3d[:, 1]**2)
    theta = np.arctan2(r_3d, pts_3d[:, 2])
    
    # Azimuth angle
    phi = np.arctan2(pts_3d[:, 1], pts_3d[:, 0])
    
    # K-B distortion: θ_d = k1*θ + k2*θ³ + k3*θ⁵ + k4*θ⁷ + k5*θ⁹
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    
    theta_d = k1 * theta + k2 * theta * theta2 + k3 * theta * theta4 + \
              k4 * theta * theta6 + k5 * theta * theta8
    
    # Distorted image coordinates
    x_dist = theta_d * np.cos(phi)
    y_dist = theta_d * np.sin(phi)
    
    # Pixel coordinates
    u = fx * x_dist + cx
    v = fy * y_dist + cy
    
    return np.column_stack([u, v])
