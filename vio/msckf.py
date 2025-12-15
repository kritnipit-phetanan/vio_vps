#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSCKF Backend Module

Multi-State Constraint Kalman Filter implementation for VIO.
Includes feature triangulation, measurement Jacobians, and EKF updates.

References:
- Mourikis & Roumeliotis, "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation"
- OpenVINS (https://github.com/rpng/open_vins)

Author: VIO project
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from scipy.spatial.transform import Rotation as R_scipy

from .math_utils import quat_to_rot, quaternion_to_yaw, skew_symmetric
from .ekf import ExtendedKalmanFilter, ensure_covariance_valid
from .camera import normalized_to_unit_ray


# Default extrinsics (will be overridden when used)
BODY_T_CAMDOWN = np.eye(4, dtype=np.float64)

# Triangulation constants
MIN_MSCKF_BASELINE = 0.15
MIN_PARALLAX_ANGLE_DEG = 0.3

# MSCKF statistics
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
    """Reset MSCKF statistics."""
    global MSCKF_STATS
    for key in MSCKF_STATS:
        MSCKF_STATS[key] = 0


def print_msckf_stats():
    """Print MSCKF triangulation statistics."""
    total = MSCKF_STATS['total_attempt']
    if total == 0:
        print("[MSCKF-STATS] No triangulation attempts")
        return
    
    success = MSCKF_STATS['success']
    print(f"[MSCKF-STATS] Total: {total}, Success: {success} ({100*success/total:.1f}%)")
    
    for key, val in MSCKF_STATS.items():
        if key.startswith('fail_') and val > 0:
            print(f"  {key}: {val} ({100*val/total:.1f}%)")


def imu_pose_to_camera_pose(q_imu: np.ndarray, p_imu: np.ndarray, 
                            T_body_cam: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert IMU/Body pose to camera pose using extrinsics.
    
    Args:
        q_imu: IMU quaternion [w,x,y,z] representing R_WB (body-to-world)
        p_imu: IMU position [x,y,z] in world frame
        T_body_cam: Body→Camera extrinsics T_BC (4x4 matrix)
    
    Returns:
        q_cam: Camera quaternion [w,x,y,z] representing R_WC (camera-to-world)
        p_cam: Camera position [x,y,z] in world frame
    """
    if T_body_cam is None:
        T_body_cam = BODY_T_CAMDOWN
    
    # Extract Body→Camera transform
    R_BC = T_body_cam[:3, :3]
    t_BC = T_body_cam[:3, 3]
    
    # Invert to get Camera→Body
    R_CB = R_BC.T
    t_CB = -R_BC.T @ t_BC
    
    # IMU/Body rotation
    q_imu_xyzw = np.array([q_imu[1], q_imu[2], q_imu[3], q_imu[0]])
    R_BW = R_scipy.from_quat(q_imu_xyzw).as_matrix()
    
    # Camera orientation: R_CW = R_BW @ R_CB
    R_CW = R_BW @ R_CB
    q_cam_xyzw = R_scipy.from_matrix(R_CW).as_quat()
    q_cam = np.array([q_cam_xyzw[3], q_cam_xyzw[0], q_cam_xyzw[1], q_cam_xyzw[2]])
    
    # Camera position: t_WC = t_WB + R_BW @ t_CB
    p_cam = p_imu + R_BW @ t_CB
    
    return q_cam, p_cam


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
            for track_fid, pt in obs_record['tracks']:
                if track_fid == fid:
                    multi_view_obs.append({
                        'cam_id': obs_record['cam_id'],
                        'pt_pixel': pt,
                        'pt_norm': pt,
                        'quality': 1.0,
                        'frame': -1,
                        't': 0.0
                    })
    return multi_view_obs


def find_mature_features_for_msckf(vio_fe, cam_observations: List[dict], 
                                   min_observations: int = 3) -> List[int]:
    """
    Find features ready for MSCKF update.
    
    Args:
        vio_fe: VIO frontend with track information
        cam_observations: List of observation records
        min_observations: Minimum observations required
    
    Returns: List of mature feature IDs
    """
    if vio_fe is None:
        return []
    
    mature_tracks = vio_fe.get_mature_tracks()
    mature_features = []
    
    for fid in mature_tracks.keys():
        obs = get_feature_multi_view_observations(fid, cam_observations)
        if len(obs) >= min_observations:
            mature_features.append(fid)
    
    return mature_features


def triangulate_point_linear(observations: List[dict], cam_states: List[dict]) -> Optional[np.ndarray]:
    """
    Linear triangulation using DLT (Direct Linear Transform).
    
    Solves the linear least squares problem for 3D point from multiple views.
    Used as initial estimate for nonlinear refinement.
    
    Args:
        observations: List of observation dicts with 'uv' (normalized coordinates)
        cam_states: List of camera state dicts with 'q' and 'p'
        
    Returns:
        3D point in world frame, or None if insufficient observations
    """
    if len(observations) < 2:
        return None
        
    # Build DLT matrix
    A_rows = []
    
    for obs in observations:
        # Find matching camera state
        cam_state = None
        for cs in cam_states:
            if cs.get('frame_idx') == obs.get('frame_idx') or cs.get('clone_id') == obs.get('clone_id'):
                cam_state = cs
                break
        
        if cam_state is None:
            continue
            
        # Get camera pose (world to camera)
        q_wc = cam_state['q']  # [w,x,y,z]
        p_w = cam_state['p']   # Camera position in world
        
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R_scipy
        q_xyzw = np.array([q_wc[1], q_wc[2], q_wc[3], q_wc[0]])
        R_wc = R_scipy.from_quat(q_xyzw).as_matrix()  # World to camera
        R_cw = R_wc.T  # Camera to world
        
        # Projection: p_cam = R_wc @ (p_world - p_w)
        # Normalized: [u, v, 1]^T ~ R_wc @ (p_world - p_w)
        # Let t = -R_wc @ p_w, then p_cam = R_wc @ p_world + t
        t_cw = -R_wc @ p_w
        
        # Build 2 equations from each observation
        uv = obs['uv']  # Normalized image coordinates
        u, v = uv[0], uv[1]
        
        # Row for u: u*(r3^T @ P + t3) - (r1^T @ P + t1) = 0
        # Row for v: v*(r3^T @ P + t3) - (r2^T @ P + t2) = 0
        r1 = R_wc[0, :]
        r2 = R_wc[1, :]
        r3 = R_wc[2, :]
        
        A_rows.append(u * r3 - r1)
        A_rows.append(v * r3 - r2)
    
    if len(A_rows) < 4:  # Need at least 2 observations
        return None
    
    A = np.array(A_rows)
    
    # Build b vector
    b_rows = []
    for i, obs in enumerate(observations):
        cam_state = None
        for cs in cam_states:
            if cs.get('frame_idx') == obs.get('frame_idx') or cs.get('clone_id') == obs.get('clone_id'):
                cam_state = cs
                break
        if cam_state is None:
            continue
            
        q_wc = cam_state['q']
        p_w = cam_state['p']
        q_xyzw = np.array([q_wc[1], q_wc[2], q_wc[3], q_wc[0]])
        R_wc = R_scipy.from_quat(q_xyzw).as_matrix()
        t_cw = -R_wc @ p_w
        
        uv = obs['uv']
        u, v = uv[0], uv[1]
        
        b_rows.append(t_cw[0] - u * t_cw[2])
        b_rows.append(t_cw[1] - v * t_cw[2])
    
    b = np.array(b_rows)
    
    # Solve least squares: A @ p = b
    try:
        p_world, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return p_world
    except np.linalg.LinAlgError:
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
        H = []
        r = []
        
        for obs in observations:
            cam_id = obs['cam_id']
            if cam_id >= len(cam_states):
                continue
            
            cs = cam_states[cam_id]
            q_idx = cs['q_idx']
            p_idx = cs['p_idx']
            
            q_imu = kf.x[q_idx:q_idx+4, 0]
            p_imu = kf.x[p_idx:p_idx+3, 0]
            q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu)
            
            # Transform point to camera frame
            q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
            R_cw = R_scipy.from_quat(q_xyzw).as_matrix()
            R_wc = R_cw.T
            
            p_c = R_wc @ (p - p_cam)
            
            if p_c[2] <= 0.1:
                return None
            
            # Project to normalized plane
            x_pred = p_c[0] / p_c[2]
            y_pred = p_c[1] / p_c[2]
            
            x_obs, y_obs = obs['pt_norm']
            r.append(x_obs - x_pred)
            r.append(y_obs - y_pred)
            
            # Jacobian
            inv_z = 1.0 / p_c[2]
            inv_z2 = inv_z * inv_z
            
            J_proj = np.array([
                [inv_z, 0, -p_c[0] * inv_z2],
                [0, inv_z, -p_c[1] * inv_z2]
            ])
            
            J = J_proj @ R_wc
            H.append(J)
        
        if len(r) < 4:
            return None
        
        H = np.vstack(H)
        r = np.array(r).reshape(-1, 1)
        
        try:
            HTH = H.T @ H
            HTr = H.T @ r
            
            lambda_lm = max(1e-3 * np.trace(HTH) / 3.0, 1e-6)
            HTH_damped = HTH + lambda_lm * np.eye(3)
            
            dp = np.linalg.solve(HTH_damped, HTr)
            
            # Limit step size
            dp_norm = np.linalg.norm(dp)
            if dp_norm > 10.0:
                dp = dp * (10.0 / dp_norm)
            
            p = p + dp.reshape(3,)
            
            if np.linalg.norm(dp) < 1e-6:
                break
        except np.linalg.LinAlgError:
            return None
    
    return p


def triangulate_feature(fid: int, cam_observations: List[dict], cam_states: List[dict],
                        kf: ExtendedKalmanFilter, use_plane_constraint: bool = True,
                        ground_altitude: float = 0.0, debug: bool = False,
                        dem_reader = None,
                        origin_lat: float = 0.0, origin_lon: float = 0.0) -> Optional[dict]:
    """
    Triangulate a feature using multi-view observations.
    
    Args:
        fid: Feature ID
        cam_observations: All observation records
        cam_states: Camera state metadata
        kf: EKF with camera poses
        use_plane_constraint: Use ground plane for depth estimation
        ground_altitude: Ground altitude in MSL
        dem_reader: DEM reader for ground plane constraint
        origin_lat/origin_lon: Local projection origin
    
    Returns: {'p_w': np.ndarray, 'observations': List, 'quality': float} or None
    """
    global MSCKF_STATS
    
    obs_list = get_feature_multi_view_observations(fid, cam_observations)
    MSCKF_STATS['total_attempt'] += 1
    
    if len(obs_list) < 2:
        MSCKF_STATS['fail_few_obs'] += 1
        return None
    
    # Get first two camera poses
    obs0 = obs_list[0]
    obs1 = obs_list[1]
    
    cam_id0 = obs0['cam_id']
    cam_id1 = obs1['cam_id']
    
    if cam_id0 >= len(cam_states) or cam_id1 >= len(cam_states):
        return None
    
    cs0 = cam_states[cam_id0]
    cs1 = cam_states[cam_id1]
    
    # Extract poses
    q_imu0 = kf.x[cs0['q_idx']:cs0['q_idx']+4, 0]
    p_imu0 = kf.x[cs0['p_idx']:cs0['p_idx']+3, 0]
    q0, p0 = imu_pose_to_camera_pose(q_imu0, p_imu0)
    
    q_imu1 = kf.x[cs1['q_idx']:cs1['q_idx']+4, 0]
    p_imu1 = kf.x[cs1['p_idx']:cs1['p_idx']+3, 0]
    q1, p1 = imu_pose_to_camera_pose(q_imu1, p_imu1)
    
    c0, c1 = p0, p1
    
    # Check baseline
    baseline = np.linalg.norm(c1 - c0)
    if baseline < MIN_MSCKF_BASELINE:
        MSCKF_STATS['fail_baseline'] += 1
        return None
    
    # Ray directions
    q0_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]])
    R0_cw = R_scipy.from_quat(q0_xyzw).as_matrix()
    
    q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]])
    R1_cw = R_scipy.from_quat(q1_xyzw).as_matrix()
    
    # CRITICAL FIX: For fisheye cameras, normalized coords can be very large (|x| > 1)
    # Using [x,y,1]/norm gives WRONG direction! Must use arctan to get proper ray angle.
    x0, y0 = obs0['pt_norm']
    x1, y1 = obs1['pt_norm']
    
    # Filter extreme fisheye angles: |norm| > 2 corresponds to angle > 63 degrees
    # For nadir camera, extreme angles give near-horizontal rays that fail triangulation
    MAX_NORM_COORD = 1.5  # ~56 degrees from optical axis
    if np.sqrt(x0*x0 + y0*y0) > MAX_NORM_COORD or np.sqrt(x1*x1 + y1*y1) > MAX_NORM_COORD:
        MSCKF_STATS['fail_other'] += 1  # Use 'other' for now, can add specific counter later
        return None
    
    ray0_c = normalized_to_unit_ray(x0, y0)
    ray1_c = normalized_to_unit_ray(x1, y1)
    
    ray0_w = R0_cw @ ray0_c
    ray0_w = ray0_w / np.linalg.norm(ray0_w)
    
    ray1_w = R1_cw @ ray1_c
    ray1_w = ray1_w / np.linalg.norm(ray1_w)
    
    # Check parallax angle
    ray_angle_rad = np.arccos(np.clip(np.dot(ray0_w, ray1_w), -1, 1))
    ray_angle_deg = np.degrees(ray_angle_rad)
    
    if ray_angle_deg < MIN_PARALLAX_ANGLE_DEG:
        MSCKF_STATS['fail_parallax'] += 1
        return None
    
    # Mid-point method
    w = c0 - c1
    a = np.dot(ray0_w, ray0_w)
    b = np.dot(ray0_w, ray1_w)
    c = np.dot(ray1_w, ray1_w)
    d = np.dot(ray0_w, w)
    e = np.dot(ray1_w, w)
    
    denom = a * c - b * b
    if abs(denom) < 1e-6:
        MSCKF_STATS['fail_solver'] += 1
        return None
    
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    
    p_ray0 = c0 + s * ray0_w
    p_ray1 = c1 + t * ray1_w
    p_init = (p_ray0 + p_ray1) / 2.0
    
    # Validate midpoint
    midpoint_valid = (s > 0 and t > 0 and p_init[2] <= c0[2])
    
    # Check depth
    depth0 = np.dot(p_init - c0, ray0_w)
    depth1 = np.dot(p_init - c1, ray1_w)
    
    if depth0 < 5.0 or depth1 < 5.0:
        MSCKF_STATS['fail_depth_sign'] += 1
        return None
    
    if depth0 > 500.0 or depth1 > 500.0:
        MSCKF_STATS['fail_depth_large'] += 1
        return None
    
    # Nonlinear refinement
    p_refined = triangulate_point_nonlinear(obs_list, cam_states, p_init, kf, debug=debug)
    
    if p_refined is None:
        MSCKF_STATS['fail_nonlinear'] += 1
        return None
    
    # =========================================================================
    # NEW (Phase 2): MSCKF Reprojection Validation using kannala_brandt_project
    # =========================================================================
    # Validate triangulated point by reprojecting to all observed frames
    # This catches errors from incorrect camera model or calibration issues
    
    from .camera import kannala_brandt_project, make_KD_for_size
    
    # Attempt to get camera intrinsics (K, D) from global config
    # If not available, fall back to normalized coordinate reprojection
    use_pixel_reprojection = False
    try:
        from .config import KB_PARAMS
        if KB_PARAMS:
            # Reconstruct K and D from KB params
            K, D = make_KD_for_size(KB_PARAMS, 
                                     int(KB_PARAMS.get('w', 1440)), 
                                     int(KB_PARAMS.get('h', 1080)))
            use_pixel_reprojection = True
    except Exception:
        pass  # Fall back to normalized coordinate method
    
    # Compute reprojection error (ENHANCED with pixel-level validation)
    total_error = 0.0
    max_pixel_error = 0.0
    
    for obs in obs_list:
        cam_id = obs['cam_id']
        if cam_id >= len(cam_states):
            continue
        
        cs = cam_states[cam_id]
        q_imu = kf.x[cs['q_idx']:cs['q_idx']+4, 0]
        p_imu = kf.x[cs['p_idx']:cs['p_idx']+3, 0]
        q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu)
        
        q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
        R_cw = R_scipy.from_quat(q_xyzw).as_matrix()
        R_wc = R_cw.T
        
        # Transform point to camera frame
        p_c = R_wc @ (p_refined - p_cam)
        
        if p_c[2] <= 0.1:
            MSCKF_STATS['fail_depth_sign'] += 1
            return None
        
        # Compute normalized coordinates (legacy method - always used)
        x_pred = p_c[0] / p_c[2]
        y_pred = p_c[1] / p_c[2]
        
        x_obs, y_obs = obs['pt_norm']
        norm_error = np.sqrt((x_obs - x_pred)**2 + (y_obs - y_pred)**2)
        total_error += norm_error
        
        # NEW: Pixel-level reprojection validation (if K, D available)
        if use_pixel_reprojection:
            # Project 3D point to pixel coordinates
            pts_reproj = kannala_brandt_project(p_c.reshape(1, 3), K, D)
            
            if pts_reproj.size > 0:
                # Get observed pixel coordinates (if stored in obs)
                if 'pt_px' in obs:
                    pt_obs_px = np.array(obs['pt_px'])
                    pt_pred_px = pts_reproj[0]
                    
                    pixel_error = np.linalg.norm(pt_pred_px - pt_obs_px)
                    max_pixel_error = max(max_pixel_error, pixel_error)
                    
                    # Reject if pixel error exceeds threshold
                    MAX_PIXEL_REPROJ_ERROR = 3.0  # pixels
                    if pixel_error > MAX_PIXEL_REPROJ_ERROR:
                        MSCKF_STATS['fail_reproj_error'] += 1
                        if debug:
                            print(f"[MSCKF-TRI] REJECT: pixel_error={pixel_error:.2f}px > {MAX_PIXEL_REPROJ_ERROR}px")
                        return None
    
    avg_error = total_error / len(obs_list)
    
    if avg_error > 0.10:
        MSCKF_STATS['fail_reproj_error'] += 1
        return None
    
    MSCKF_STATS['success'] += 1
    result = {
        'p_w': p_refined,
        'observations': obs_list,
        'quality': 1.0 / (1.0 + avg_error * 100.0),
        'avg_reproj_error': avg_error
    }
    
    # Add pixel-level error if available
    if use_pixel_reprojection and max_pixel_error > 0:
        result['max_pixel_reproj_error'] = max_pixel_error
    
    return result


def compute_huber_weights(normalized_residuals: np.ndarray, 
                          threshold: float = 1.345) -> np.ndarray:
    """Compute Huber robust loss weights."""
    r_abs = np.abs(normalized_residuals)
    weights = np.ones_like(normalized_residuals)
    
    outlier_mask = r_abs > threshold
    weights[outlier_mask] = threshold / r_abs[outlier_mask]
    
    return weights


def compute_measurement_jacobian(p_w: np.ndarray, cam_state: dict, 
                                 kf: ExtendedKalmanFilter,
                                 err_state_size: int, 
                                 use_preint_jacobians: bool = True,
                                 T_cam_imu: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute measurement Jacobian for one camera observation.
    
    Args:
        p_w: 3D point in world frame
        cam_state: Camera state metadata
        kf: EKF with full state
        err_state_size: Error state dimension
        use_preint_jacobians: Add IMU preintegration terms
        T_cam_imu: Camera-to-IMU extrinsics
    
    Returns: (H_cam, H_feat)
    """
    if T_cam_imu is None:
        T_cam_imu = BODY_T_CAMDOWN
    
    R_cam_imu = T_cam_imu[:3, :3]
    t_cam_imu = T_cam_imu[:3, 3]
    
    # Get IMU pose (FEJ if available)
    if 'q_fej' in cam_state and 'p_fej' in cam_state:
        q_imu = cam_state['q_fej']
        p_imu = cam_state['p_fej']
    else:
        q_idx = cam_state['q_idx']
        p_idx = cam_state['p_idx']
        q_imu = kf.x[q_idx:q_idx+4, 0]
        p_imu = kf.x[p_idx:p_idx+3, 0]
    
    # Transform to camera pose
    q_imu_xyzw = np.array([q_imu[1], q_imu[2], q_imu[3], q_imu[0]])
    R_w_imu = R_scipy.from_quat(q_imu_xyzw).as_matrix()
    
    R_w_cam = R_w_imu @ R_cam_imu.T
    p_cam = p_imu + R_w_imu @ t_cam_imu
    
    # Transform point to camera frame
    p_rel = p_w - p_cam
    p_c = R_w_cam.T @ p_rel
    
    # Projection Jacobian
    inv_z = 1.0 / p_c[2]
    inv_z2 = inv_z * inv_z
    
    j_proj = np.array([
        [inv_z, 0, -p_c[0] * inv_z2],
        [0, inv_z, -p_c[1] * inv_z2]
    ])
    
    # Jacobian w.r.t. feature
    h_feat = j_proj @ R_w_cam.T
    
    # Jacobian w.r.t. error state
    h_cam = np.zeros((2, err_state_size))
    
    err_theta_idx = cam_state['err_q_idx']
    err_p_idx = cam_state['err_p_idx']
    
    # Position Jacobian
    h_cam[:, err_p_idx:err_p_idx+3] = j_proj @ (-R_w_cam.T)
    
    # Rotation Jacobian
    t_cam_world = R_w_imu @ t_cam_imu
    p_rel_total = p_rel + t_cam_world
    skew_p_rel = skew_symmetric(p_rel_total)
    j_rot = j_proj @ (-R_w_cam.T @ skew_p_rel @ R_cam_imu)
    h_cam[:, err_theta_idx:err_theta_idx+3] = j_rot
    
    # Preintegration Jacobians (bias coupling)
    if use_preint_jacobians and 'preint' in cam_state and cam_state['preint'] is not None:
        preint = cam_state['preint']
        J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba = preint.get_jacobians()
        
        R_clone = R_w_imu
        
        # Gyro bias
        h_cam[:, 9:12] += j_rot @ R_cam_imu @ J_R_bg
        j_pos = j_proj @ (-R_w_cam.T)
        h_cam[:, 9:12] += j_pos @ R_clone @ J_p_bg
        
        # Accel bias
        h_cam[:, 12:15] += j_pos @ R_clone @ J_p_ba
    
    return h_cam, h_feat


def compute_observability_nullspace(kf: ExtendedKalmanFilter, 
                                   num_clones: int) -> np.ndarray:
    """
    Compute nullspace basis for unobservable directions.
    
    Unobservable modes: global X, Y position and global yaw.
    
    Args:
        kf: EKF with current state
        num_clones: Number of camera clones
    
    Returns: (err_state_size, 3) nullspace matrix
    """
    err_state_size = 15 + 6 * num_clones
    
    q_imu = kf.x[6:10, 0]
    yaw = quaternion_to_yaw(q_imu)
    
    U = np.zeros((err_state_size, 3), dtype=float)
    
    # Global X translation
    U[0, 0] = 1.0
    for i in range(num_clones):
        clone_p_idx = 15 + 6*i + 3
        U[clone_p_idx, 0] = 1.0
    
    # Global Y translation
    U[1, 1] = 1.0
    for i in range(num_clones):
        clone_p_idx = 15 + 6*i + 3
        U[clone_p_idx + 1, 1] = 1.0
    
    # Global yaw rotation
    U[8, 2] = 1.0
    
    p_imu = kf.x[0:3, 0]
    U[0, 2] = -p_imu[1]
    U[1, 2] = p_imu[0]
    
    core_size = 16
    for i in range(num_clones):
        clone_p_idx = core_size + 7*i + 4
        p_cam = kf.x[clone_p_idx:clone_p_idx+3, 0]
        
        err_theta_idx = 15 + 6*i
        err_p_idx = 15 + 6*i + 3
        
        U[err_theta_idx + 2, 2] = 1.0
        U[err_p_idx, 2] = -p_cam[1]
        U[err_p_idx + 1, 2] = p_cam[0]
    
    # Orthonormalize
    U_ortho = np.zeros_like(U)
    for j in range(3):
        u_j = U[:, j].copy()
        for k in range(j):
            u_j -= np.dot(U_ortho[:, k], u_j) * U_ortho[:, k]
        norm = np.linalg.norm(u_j)
        if norm > 1e-10:
            U_ortho[:, j] = u_j / norm
    
    return U_ortho


def msckf_measurement_update(fid: int, triangulated: dict, cam_observations: List[dict],
                             cam_states: List[dict], kf: ExtendedKalmanFilter,
                             measurement_noise: float = 1e-4,
                             huber_threshold: float = 1.345,
                             chi2_max_dof: float = 15.36) -> Tuple[bool, float, float]:
    """
    MSCKF measurement update with observability constraints.
    
    Args:
        fid: Feature ID
        triangulated: Triangulation result
        cam_observations: Observation records
        cam_states: Camera state metadata
        kf: EKF to update
        measurement_noise: Measurement noise variance
        huber_threshold: Huber loss threshold
        chi2_max_dof: Max chi-square per DoF
    
    Returns: (success, innovation_norm, chi2_test)
    """
    p_w = triangulated['p_w']
    obs_list = triangulated['observations']
    
    if len(obs_list) < 2:
        return (False, np.nan, np.nan)
    
    err_state_size = kf.P.shape[0]
    
    # Compute residuals and Jacobians
    residuals = []
    h_x_stack = []
    h_f_stack = []
    
    for obs in obs_list:
        cam_id = obs['cam_id']
        if cam_id >= len(cam_states):
            continue
        
        cs = cam_states[cam_id]
        
        # Predicted measurement
        q_imu = kf.x[cs['q_idx']:cs['q_idx']+4, 0]
        p_imu = kf.x[cs['p_idx']:cs['p_idx']+3, 0]
        q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu)
        
        q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
        r_cw = R_scipy.from_quat(q_xyzw).as_matrix()
        r_wc = r_cw.T
        p_c = r_wc @ (p_w - p_cam)
        
        if p_c[2] <= 0.1:
            continue
        
        z_pred = np.array([p_c[0] / p_c[2], p_c[1] / p_c[2]])
        z_obs = np.array([obs['pt_norm'][0], obs['pt_norm'][1]])
        
        r = z_obs - z_pred
        residuals.append(r)
        
        h_cam, h_feat = compute_measurement_jacobian(p_w, cs, kf, err_state_size)
        h_x_stack.append(h_cam)
        h_f_stack.append(h_feat)
    
    if len(residuals) < 2:
        return (False, np.nan, np.nan)
    
    r_o = np.vstack(residuals).reshape(-1, 1)
    h_x = np.vstack(h_x_stack)
    h_f = np.vstack(h_f_stack)
    
    # Nullspace projection
    try:
        u_mat, s_mat, vh_mat = np.linalg.svd(h_f, full_matrices=True)
        tol = 1e-6 * s_mat[0] if len(s_mat) > 0 else 1e-6
        rank = np.sum(s_mat > tol)
        null_space = u_mat[:, rank:]
        
        h_proj = null_space.T @ h_x
        r_proj = null_space.T @ r_o
    except np.linalg.LinAlgError:
        h_proj = h_x
        r_proj = r_o
    
    # Observability constraint
    num_clones = (err_state_size - 15) // 6
    
    try:
        U_obs = compute_observability_nullspace(kf, num_clones)
        projection_matrix = np.eye(err_state_size) - U_obs @ U_obs.T
        h_constrained = h_proj @ projection_matrix
        r_constrained = r_proj
    except (np.linalg.LinAlgError, ValueError):
        h_constrained = h_proj
        r_constrained = r_proj
    
    # Huber weighting
    measurement_std = np.sqrt(measurement_noise)
    r_normalized = r_constrained / measurement_std
    weights = compute_huber_weights(r_normalized.flatten(), threshold=huber_threshold)
    weight_matrix = np.diag(np.sqrt(weights))
    
    h_weighted = weight_matrix @ h_constrained
    r_weighted = weight_matrix @ r_constrained
    
    meas_dim = r_weighted.shape[0]
    r_cov_original = np.eye(meas_dim) * measurement_noise
    r_cov = weight_matrix @ r_cov_original @ weight_matrix.T
    
    # Innovation and chi-square gating
    s_mat = h_weighted @ kf.P @ h_weighted.T + r_cov
    innovation_norm = float(np.linalg.norm(r_weighted))
    
    try:
        s_inv = np.linalg.inv(s_mat)
        chi2_test = float(r_weighted.T @ s_inv @ r_weighted)
        
        dof = meas_dim
        chi2_threshold = chi2_max_dof * dof
        
        if chi2_test > chi2_threshold:
            return (False, innovation_norm, chi2_test)
    except np.linalg.LinAlgError:
        return (False, innovation_norm, np.nan)
    
    # EKF update
    try:
        k_gain = kf.P @ h_weighted.T @ s_inv
        delta_x = k_gain @ r_weighted
        
        kf._apply_error_state_correction(delta_x)
        
        i_kh = np.eye(err_state_size) - k_gain @ h_weighted
        kf.P = i_kh @ kf.P @ i_kh.T + k_gain @ r_cov @ k_gain.T
        
        kf.P = ensure_covariance_valid(kf.P, label="MSCKF-Update", 
                                        symmetrize=True, check_psd=True)
        
        kf.x_post = kf.x.copy()
        kf.P_post = kf.P.copy()
        
        return (True, innovation_norm, chi2_test)
    except (np.linalg.LinAlgError, ValueError):
        return (False, innovation_norm, np.nan)


def msckf_measurement_update_with_plane(fid: int, triangulated: dict, 
                                       cam_observations: List[dict],
                                       cam_states: List[dict], 
                                       kf: ExtendedKalmanFilter,
                                       plane,
                                       plane_config: dict) -> Tuple[bool, float, float]:
    """
    MSCKF measurement update with stacked plane constraint.
    
    Stacked measurement model (following OpenVINS ov_plane):
        [z_bearing]   [H_bearing  ]   [0]
        [z_plane  ] = [H_plane    ] + [ε]
    
    where:
        - z_bearing: 2D bearing measurements (standard MSCKF)
        - z_plane: 1D point-on-plane constraint (n^T * p + d = 0)
    
    Args:
        fid: Feature ID
        triangulated: Triangulation result with 'p_w' (3D point)
        cam_observations: All observations
        cam_states: Camera states
        kf: EKF
        plane: Detected plane (with normal n and distance d)
        plane_config: Plane configuration dict
    
    Returns: (success, innovation_norm, chi2_test)
    """
    from .plane_utils import compute_plane_jacobian
    
    # =========================================================================
    # Part 1: Standard MSCKF bearing measurements
    # =========================================================================
    point_world = triangulated['p_w']
    obs_list = triangulated['observations']
    
    if len(obs_list) < 2:
        return (False, 0.0, 0.0)
    
    # Build bearing Jacobian (same as standard MSCKF)
    err_state_size = 15 + 6 * len(cam_states)
    meas_dim_bearing = 2 * len(obs_list)
    
    h_bearing = np.zeros((meas_dim_bearing, err_state_size))
    r_bearing = np.zeros(meas_dim_bearing)
    
    for i, obs_data in enumerate(obs_list):
        cam_id = obs_data['cam_id']
        if cam_id >= len(cam_states):
            continue
        
        cs = cam_states[cam_id]
        q_imu = kf.x[cs['q_idx']:cs['q_idx']+4, 0]
        p_imu = kf.x[cs['p_idx']:cs['p_idx']+3, 0]
        
        q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu)
        q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
        R_cw = R_scipy.from_quat(q_xyzw).as_matrix()
        R_wc = R_cw.T
        
        p_c = R_wc @ (point_world - p_cam)
        
        if p_c[2] < 0.1:
            return (False, 0.0, 0.0)
        
        # Bearing measurement
        xn_pred = p_c[0] / p_c[2]
        yn_pred = p_c[1] / p_c[2]
        
        xn_obs, yn_obs = obs_data['pt_norm']
        
        r_bearing[2*i] = xn_obs - xn_pred
        r_bearing[2*i+1] = yn_obs - yn_pred
        
        # Jacobian (standard MSCKF)
        z_inv = 1.0 / p_c[2]
        z_inv2 = z_inv * z_inv
        
        J_proj = np.array([
            [z_inv, 0, -p_c[0]*z_inv2],
            [0, z_inv, -p_c[1]*z_inv2]
        ])
        
        J_point = J_proj @ R_wc
        J_q = -J_proj @ R_cw @ skew_symmetric(p_c)
        J_p = -J_proj @ R_wc
        
        clone_idx_err = 15 + 6 * cam_id
        h_bearing[2*i:2*i+2, clone_idx_err:clone_idx_err+3] = J_q
        h_bearing[2*i:2*i+2, clone_idx_err+3:clone_idx_err+6] = J_p
    
    # =========================================================================
    # Part 2: Plane constraint measurement
    # =========================================================================
    # Measurement: z_plane = n^T * p + d = 0 (ideal)
    # Residual: r_plane = n^T * p_w + d
    
    residual_plane = plane.point_distance(point_world)
    
    # Jacobian w.r.t. point (not used here, but for reference)
    # H_point = n^T (1x3)
    
    # Jacobian w.r.t. camera states (through triangulated point)
    # Since point is not in state, we approximate H_plane from bearing sensitivities
    # Following ov_plane: use weighted combination of bearing Jacobians
    
    # Simplified approach: H_plane ≈ weighted average of bearing contributions
    # More rigorous: recompute ∂p/∂cam_states from triangulation
    
    h_plane = np.zeros((1, err_state_size))
    n = plane.normal
    
    # Approximate: weight bearing Jacobians by point sensitivity
    for i, obs_data in enumerate(obs_list):
        cam_id = obs_data['cam_id']
        if cam_id >= len(cam_states):
            continue
        
        cs = cam_states[cam_id]
        q_imu = kf.x[cs['q_idx']:cs['q_idx']+4, 0]
        p_imu = kf.x[cs['p_idx']:cs['p_idx']+3, 0]
        
        q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu)
        q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
        R_cw = R_scipy.from_quat(q_xyzw).as_matrix()
        R_wc = R_cw.T
        
        p_c = R_wc @ (point_world - p_cam)
        depth = p_c[2]
        
        # Sensitivity of point to camera pose (simplified)
        clone_idx_err = 15 + 6 * cam_id
        
        # ∂p_w/∂θ_cam ≈ -R_wc @ [p_c]×
        J_p_theta = -R_wc @ skew_symmetric(p_c)
        
        # ∂p_w/∂p_cam = R_wc @ (-I) = -R_wc
        J_p_pos = -R_wc
        
        # ∂(n^T*p)/∂cam_state
        h_plane[0, clone_idx_err:clone_idx_err+3] += n @ J_p_theta / len(obs_list)
        h_plane[0, clone_idx_err+3:clone_idx_err+6] += n @ J_p_pos / len(obs_list)
    
    # =========================================================================
    # Part 3: Stack measurements and update
    # =========================================================================
    meas_dim_total = meas_dim_bearing + 1
    h_stacked = np.vstack([h_bearing, h_plane])
    r_stacked = np.concatenate([r_bearing, [residual_plane]])
    
    # Measurement noise
    sigma_bearing = 1e-4  # Standard MSCKF bearing noise
    sigma_plane = plane_config.get('PLANE_SIGMA', 0.05)
    
    R_stacked = np.eye(meas_dim_total)
    R_stacked[:meas_dim_bearing, :meas_dim_bearing] *= sigma_bearing
    R_stacked[-1, -1] = sigma_plane ** 2
    
    # Observability constraint (same as standard MSCKF)
    U_nullspace = compute_observability_nullspace(kf, len(cam_states))
    
    if U_nullspace is not None and U_nullspace.shape[0] == h_stacked.shape[1]:
        A_proj = np.eye(h_stacked.shape[1]) - U_nullspace @ U_nullspace.T
        h_weighted = h_stacked @ A_proj
        r_weighted = r_stacked
    else:
        h_weighted = h_stacked
        r_weighted = r_stacked
    
    # Chi-square gating
    s_mat = h_weighted @ kf.P @ h_weighted.T + R_stacked
    innovation_norm = float(np.linalg.norm(r_weighted))
    
    try:
        s_inv = np.linalg.inv(s_mat)
        chi2_test = float(r_weighted.T @ s_inv @ r_weighted)
        
        chi2_threshold = 15.36 * meas_dim_total  # Same as standard MSCKF
        
        if chi2_test > chi2_threshold:
            return (False, innovation_norm, chi2_test)
    except np.linalg.LinAlgError:
        return (False, innovation_norm, np.nan)
    
    # EKF update
    try:
        k_gain = kf.P @ h_weighted.T @ s_inv
        delta_x = k_gain @ r_weighted
        
        kf._apply_error_state_correction(delta_x)
        
        i_kh = np.eye(err_state_size) - k_gain @ h_weighted
        kf.P = i_kh @ kf.P @ i_kh.T + k_gain @ R_stacked @ k_gain.T
        
        kf.P = ensure_covariance_valid(kf.P, label="MSCKF-Plane-Update", 
                                        symmetrize=True, check_psd=True)
        
        kf.x_post = kf.x.copy()
        kf.P_post = kf.P.copy()
        
        return (True, innovation_norm, chi2_test)
    except (np.linalg.LinAlgError, ValueError):
        return (False, innovation_norm, np.nan)


def perform_msckf_updates(vio_fe, cam_observations: List[dict],
                          cam_states: List[dict], kf: ExtendedKalmanFilter,
                          min_observations: int = 3, max_features: int = 50,
                          msckf_dbg_path: str = None,
                          dem_reader = None,
                          origin_lat: float = 0.0, origin_lon: float = 0.0,
                          plane_detector = None,
                          plane_config: dict = None) -> int:
    """
    Perform MSCKF updates for mature features.
    
    Optionally uses plane-aided MSCKF with stacked measurements:
    - Standard bearing measurements (2D per observation)
    - Point-on-plane constraint (1D per plane-associated feature)
    
    Args:
        vio_fe: VIO frontend
        cam_observations: Observation records
        cam_states: Camera state metadata
        kf: EKF to update
        min_observations: Minimum observations per feature
        max_features: Maximum features to process
        msckf_dbg_path: Debug log path
        dem_reader: DEM reader for ground constraint
        origin_lat/origin_lon: Local projection origin
        plane_detector: PlaneDetector for plane-aided MSCKF (optional)
        plane_config: Plane configuration dict (optional)
    
    Returns: Number of successful updates
    """
    if vio_fe is None or len(cam_states) < 2:
        return 0
    
    mature_fids = find_mature_features_for_msckf(vio_fe, cam_observations, min_observations)
    
    if len(mature_fids) == 0:
        return 0
    
    if len(mature_fids) > max_features:
        mature_fids = mature_fids[:max_features]
    
    num_successful = 0
    
    # =========================================================================
    # Plane Detection (if enabled)
    # =========================================================================
    detected_planes = []
    triangulated_points = {}
    
    if plane_detector is not None and plane_config is not None:
        # First pass: triangulate all mature features to build point cloud
        for fid in mature_fids:
            tri_result = triangulate_feature(fid, cam_observations, cam_states, kf,
                                            use_plane_constraint=True, ground_altitude=0.0,
                                            debug=False,
                                            dem_reader=dem_reader,
                                            origin_lat=origin_lat, origin_lon=origin_lon)
            if tri_result is not None:
                triangulated_points[fid] = tri_result['p_w']
        
        # Detect planes from triangulated point cloud
        if len(triangulated_points) >= 10:
            points_array = np.array(list(triangulated_points.values()))
            try:
                detected_planes = plane_detector.detect_planes(points_array)
                if len(detected_planes) > 0:
                    print(f"[MSCKF-PLANE] Detected {len(detected_planes)} planes from {len(points_array)} points")
            except Exception as e:
                print(f"[MSCKF-PLANE] Plane detection failed: {e}")
    
    # =========================================================================
    # MSCKF Updates with Optional Plane Constraints
    # =========================================================================
    for i, fid in enumerate(mature_fids):
        enable_debug = (i < 3)
        triangulated = triangulate_feature(fid, cam_observations, cam_states, kf, 
                                          use_plane_constraint=True, ground_altitude=0.0,
                                          debug=enable_debug,
                                          dem_reader=dem_reader,
                                          origin_lat=origin_lat, origin_lon=origin_lon)
        
        if triangulated is None:
            if msckf_dbg_path:
                num_obs = sum(1 for cam_obs in cam_observations 
                             for obs in cam_obs.get('observations', []) 
                             if obs.get('fid') == fid)
                with open(msckf_dbg_path, "a", newline="") as mf:
                    mf.write(f"{vio_fe.frame_idx},{fid},{num_obs},0,nan,nan,0,nan\n")
            continue
        
        # Check if feature is associated with a plane
        associated_plane = None
        if len(detected_planes) > 0 and plane_config is not None:
            point_3d = triangulated['p_w']
            distance_threshold = plane_config.get('PLANE_DISTANCE_THRESHOLD', 0.15)
            
            # Find nearest plane
            min_dist = float('inf')
            for plane in detected_planes:
                dist = abs(plane.point_distance(point_3d))
                if dist < min_dist and dist < distance_threshold:
                    min_dist = dist
                    associated_plane = plane
        
        # MSCKF update with optional plane constraint
        if associated_plane is not None and plane_config.get('PLANE_USE_CONSTRAINTS', True):
            # Stacked measurement: bearing (2D per obs) + plane constraint (1D)
            success, innovation_norm, chi2_test = msckf_measurement_update_with_plane(
                fid, triangulated, cam_observations, cam_states, kf, associated_plane, plane_config)
        else:
            # Standard MSCKF update (bearing only)
            success, innovation_norm, chi2_test = msckf_measurement_update(
                fid, triangulated, cam_observations, cam_states, kf)
        
        if msckf_dbg_path:
            num_obs = sum(1 for cam_obs in cam_observations 
                         for obs in cam_obs.get('observations', []) 
                         if obs.get('fid') == fid)
            avg_reproj = triangulated.get('avg_reproj_error', np.nan)
            with open(msckf_dbg_path, "a", newline="") as mf:
                mf.write(f"{vio_fe.frame_idx},{fid},{num_obs},1,{avg_reproj:.3f},"
                        f"{innovation_norm:.3f},{int(success)},{chi2_test:.3f}\n")
        
        if success:
            num_successful += 1
    
    return num_successful


# =============================================================================
# High-level MSCKF Trigger Function
# =============================================================================

def trigger_msckf_update(kf, cam_states: list, cam_observations: list,
                         vio_fe, t: float = 0.0,
                         msckf_dbg_csv: Optional[str] = None,
                         dem_reader=None, origin_lat: float = 0.0,
                         origin_lon: float = 0.0,
                         plane_detector=None,
                         plane_config: dict = None) -> int:
    """
    Trigger MSCKF multi-view geometric update.
    
    This function decides when to perform MSCKF updates based on:
    - Number of mature features (seen in multiple views)
    - Number of camera clones in sliding window
    - Frame intervals
    
    Args:
        kf: ExtendedKalmanFilter instance
        cam_states: List of camera clone states
        cam_observations: List of camera observations
        vio_fe: VIO frontend
        t: Current timestamp for logging
        msckf_dbg_csv: Optional debug CSV path
        dem_reader: Optional DEM reader for ground constraints
        origin_lat/origin_lon: Local projection origin
    
    Returns:
        Number of successful MSCKF updates
    """
    if vio_fe is None or len(cam_states) < 2:
        return 0
    
    # Count mature features
    feature_obs_count = {}
    for obs_set in cam_observations:
        for obs in obs_set['observations']:
            fid = obs['fid']
            feature_obs_count[fid] = feature_obs_count.get(fid, 0) + 1
    
    num_mature = sum(1 for c in feature_obs_count.values() if c >= 2)
    
    # Decide if we should update
    should_update = (
        num_mature >= 20 or                                    # Many mature features
        len(cam_states) >= 4 or                               # Window getting full
        (vio_fe.frame_idx % 5 == 0 and len(cam_states) >= 3)  # Periodic update
    )
    
    if should_update:
        try:
            num_updates = perform_msckf_updates(
                vio_fe,
                cam_observations,
                cam_states,
                kf,
                min_observations=2,
                max_features=50,
                msckf_dbg_path=msckf_dbg_csv,
                dem_reader=dem_reader,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                plane_detector=plane_detector,
                plane_config=plane_config
            )
            if num_updates > 0:
                print(f"[MSCKF] Updated {num_updates} features at t={t:.3f}s")
            return num_updates
        except Exception as e:
            print(f"[MSCKF] Error: {e}")
            return 0
    
    return 0

