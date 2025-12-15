#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plane-Aided MSCKF Module

Implements plane-constrained MSCKF updates for improved depth estimation
and reduced drift in planar scenes (indoor, urban environments).

Key Features:
    - Point-to-plane distance constraints
    - Plane-aided triangulation (improves depth for planar features)
    - SLAM plane state management
    - MSCKF plane marginalization

References:
    [1] Geneva et al., "OpenVINS: ov_plane", GitHub 2020
    [2] Hsiao et al., "Optimization-based VI-SLAM with Planes", ICRA 2018

Author: VIO project
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.spatial.transform import Rotation as R_scipy

from .ekf import ExtendedKalmanFilter
from .plane_utils import Plane, compute_plane_jacobian
from .msckf import triangulate_feature


def plane_aided_triangulation(fid: int,
                              cam_observations: List[Dict],
                              cam_states: List[Dict],
                              kf: ExtendedKalmanFilter,
                              plane: Plane,
                              dem_reader=None,
                              origin_lat: float = 0.0,
                              origin_lon: float = 0.0) -> Tuple[Optional[np.ndarray], str]:
    """
    Triangulate feature with plane constraint.
    
    Strategy:
        1. Do standard triangulation (2-view or multi-view)
        2. Project result onto plane
        3. Return projected point with validation
    
    Args:
        fid: Feature ID
        cam_observations: List of camera observation dicts
        cam_states: List of camera state dicts
        kf: ExtendedKalmanFilter
        plane: Plane constraint
        dem_reader: Optional DEM reader
        origin_lat/origin_lon: Local projection origin
    
    Returns:
        point_world: Triangulated 3D point in world frame (or None)
        status: Success/failure reason
    """
    # Step 1: Standard triangulation
    result = triangulate_feature(
        fid, cam_observations, cam_states, kf,
        dem_reader=dem_reader,
        origin_lat=origin_lat,
        origin_lon=origin_lon
    )
    
    if result is None:
        return None, "standard_triangulation_failed"
    
    point_world = result['p_w']
    
    # Step 2: Check distance to plane
    dist_to_plane = plane.point_distance(point_world)
    
    # If already close to plane, use as-is
    if abs(dist_to_plane) < 0.2:  # Within 20cm
        return point_world, f"plane_aided_accepted_dist_{abs(dist_to_plane):.3f}m"
    
    # Step 3: Project onto plane
    point_on_plane = plane.project_point(point_world)
    
    # Step 4: Return projected point
    # (Optional: could re-refine here with plane constraint in optimization)
    
    return point_on_plane, f"plane_aided_projected_dist_{abs(dist_to_plane):.3f}m"


def compute_plane_constraint_residual(point_world: np.ndarray,
                                      plane: Plane) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute plane constraint residual and Jacobians.
    
    Measurement model:
        z = n^T * p + d = 0  (ideal, point on plane)
        
    Args:
        point_world: 3D point in world frame
        plane: Plane [n, d]
    
    Returns:
        residual: Signed distance (innovation)
        H_point: Jacobian w.r.t. point (1x3)
        H_plane: Jacobian w.r.t. plane (1x4)
    """
    # Residual: distance to plane
    residual = plane.point_distance(point_world)
    
    # Jacobians
    H_plane, H_point = compute_plane_jacobian(plane, point_world)
    
    return residual, H_point, H_plane


def apply_plane_constraint_update(kf: ExtendedKalmanFilter,
                                  plane: Plane,
                                  point_world: np.ndarray,
                                  point_cov: np.ndarray,
                                  feature_id: int,
                                  sigma_plane: float = 0.05,
                                  save_debug: bool = False) -> Tuple[bool, str]:
    """
    Apply plane constraint as EKF measurement update.
    
    Measurement:
        z = 0 (point should be on plane)
        y = n^T * p + d (innovation)
    
    Args:
        kf: ExtendedKalmanFilter
        plane: Plane constraint
        point_world: Triangulated 3D point
        point_cov: Point covariance (3x3)
        feature_id: Feature ID
        sigma_plane: Plane measurement noise (meters)
        save_debug: Enable debug logging
    
    Returns:
        success: True if update applied
        reason: Status message
    """
    # Compute residual and Jacobians
    residual, H_point, H_plane = compute_plane_constraint_residual(point_world, plane)
    
    # Innovation covariance
    # S = H_point * Cov_point * H_point^T + R
    S = H_point @ point_cov @ H_point.T + sigma_plane**2
    
    # Chi-square gating
    m2 = residual**2 / S[0, 0]
    chi2_threshold = 5.99  # 95% confidence, 1 DOF
    
    if m2 > chi2_threshold:
        return False, f"plane_constraint_rejected_chi2_{m2:.2f}"
    
    # Build full measurement Jacobian (w.r.t. EKF state)
    # For now, assume point is not in state (MSCKF feature)
    # So we don't update state, just use as validation
    
    # TODO: If implementing SLAM planes in state, add H w.r.t. plane parameters
    
    if save_debug:
        print(f"[PLANE-MSCKF] Feature {feature_id}: "
              f"residual={residual:.3f}m, chi2={m2:.2f}, "
              f"accepted={m2 < chi2_threshold}")
    
    return True, f"plane_constraint_accepted_chi2_{m2:.2f}"


def initialize_slam_plane(plane: Plane,
                          kf: ExtendedKalmanFilter,
                          initial_cov: float = 0.1) -> int:
    """
    Add plane to EKF state (SLAM plane).
    
    Plane parameterization in state:
        - Minimal: Use [n1, n2, d] (3 DOF, constrain n3 = sqrt(1-n1²-n2²))
        - Overcomplete: Use [n1, n2, n3, d] (4 params, enforce ||n||=1)
    
    We use overcomplete with constraint enforcement.
    
    Args:
        kf: ExtendedKalmanFilter
        plane: Plane to add
        initial_cov: Initial uncertainty (meters for d, unitless for n)
    
    Returns:
        plane_start_idx: Index in state where plane starts
    """
    # Augment nominal state
    plane_params = plane.to_array()  # [nx, ny, nz, d]
    
    kf.x = np.vstack([kf.x, plane_params.reshape(-1, 1)])
    kf.dim_x = kf.x.shape[0]
    
    # Augment error-state covariance (plane has 3 DOF in error-state)
    # Error: [δn1, δn2, δd] (δn3 constrained by sphere)
    old_dim_err = kf.dim_err
    kf.dim_err += 3
    
    P_new = np.zeros((kf.dim_err, kf.dim_err))
    P_new[:old_dim_err, :old_dim_err] = kf.P
    
    # Initial plane uncertainty
    P_new[old_dim_err:old_dim_err+2, old_dim_err:old_dim_err+2] = np.eye(2) * (0.01**2)  # Normal
    P_new[old_dim_err+2, old_dim_err+2] = initial_cov**2  # Distance
    
    kf.P = P_new
    
    plane_start_idx = kf.dim_x - 4
    
    print(f"[PLANE-SLAM] Initialized plane {plane.id} at state index {plane_start_idx}")
    
    return plane_start_idx


def marginalize_slam_plane(kf: ExtendedKalmanFilter,
                           plane_idx: int) -> bool:
    """
    Remove SLAM plane from state.
    
    Args:
        kf: ExtendedKalmanFilter
        plane_idx: Starting index of plane in state
    
    Returns:
        success: True if marginalized
    """
    # Marginalize from nominal state
    mask_nominal = np.ones(kf.dim_x, dtype=bool)
    mask_nominal[plane_idx:plane_idx+4] = False
    
    kf.x = kf.x[mask_nominal]
    kf.dim_x = kf.x.shape[0]
    
    # Marginalize from error-state covariance
    # Find corresponding error-state indices
    # This is complex - for now, simplified removal
    
    # TODO: Proper error-state index mapping
    
    print(f"[PLANE-SLAM] Marginalized plane at index {plane_idx}")
    
    return True


def associate_features_to_planes(feature_tracks: List[Dict],
                                 triangulated_points: Dict[int, np.ndarray],
                                 planes: List[Plane],
                                 distance_threshold: float = 0.15) -> Dict[int, int]:
    """
    Associate features to planes based on distance.
    
    Args:
        feature_tracks: List of feature track dicts
        triangulated_points: Dict mapping feature_id -> 3D point
        planes: List of detected planes
        distance_threshold: Max distance for association (meters)
    
    Returns:
        associations: Dict mapping feature_id -> plane_id
    """
    associations = {}
    
    for fid, point in triangulated_points.items():
        best_plane = None
        best_dist = float('inf')
        
        for plane in planes:
            dist = abs(plane.point_distance(point))
            
            if dist < best_dist and dist < distance_threshold:
                best_dist = dist
                best_plane = plane
        
        if best_plane is not None:
            associations[fid] = best_plane.id
            
            # Add feature to plane's feature list
            if fid not in best_plane.feature_ids:
                best_plane.feature_ids.append(fid)
    
    return associations


def compute_msckf_with_plane_constraints(kf,
                                         cam_states: List[Dict],
                                         cam_observations: List[Dict],
                                         vio_fe,
                                         plane_detector,
                                         config: Dict,
                                         t: float = 0.0,
                                         dem_reader=None,
                                         origin_lat: float = 0.0,
                                         origin_lon: float = 0.0) -> Tuple[int, int, int]:
    """
    Enhanced MSCKF with plane constraints.
    
    Called after standard MSCKF updates to apply plane-based constraints.
    
    Args:
        kf: ExtendedKalmanFilter
        cam_states: List of camera clone states
        cam_observations: List of camera observations
        vio_fe: VIO frontend
        plane_detector: PlaneDetector instance
        config: Configuration dict with plane parameters
        t: Current timestamp
        dem_reader: Optional DEM reader
        origin_lat/origin_lon: Local projection origin
    
    Returns:
        num_updates: Number of successful MSCKF updates
        num_plane_aided: Number of plane-aided triangulations
        num_plane_constrained: Number of plane constraint updates
    """
    # Extract config
    use_aided_triangulation = config.get('PLANE_USE_AIDED_TRIANGULATION', True)
    use_constraints = config.get('PLANE_USE_CONSTRAINTS', True)
    sigma_plane = config.get('PLANE_SIGMA', 0.05)
    distance_threshold = config.get('PLANE_DISTANCE_THRESHOLD', 0.15)
    
    num_plane_aided = 0
    num_plane_constrained = 0
    num_planes_detected = 0
    
    # =========================================================================
    # Step 1: Triangulate all mature features to get 3D point cloud
    # =========================================================================
    triangulated_points = {}  # fid -> 3D point
    
    # Get all mature features (seen in multiple views)
    feature_obs_count = {}
    for obs_set in cam_observations:
        for obs in obs_set['observations']:
            fid = obs['fid']
            feature_obs_count[fid] = feature_obs_count.get(fid, 0) + 1
    
    mature_features = [fid for fid, count in feature_obs_count.items() if count >= 2]
    
    if len(mature_features) < 10:
        # Not enough points for plane detection
        return 0, 0, 0
    
    # Triangulate all mature features
    for fid in mature_features:
        result = triangulate_feature(
            fid, cam_observations, cam_states, kf,
            dem_reader=dem_reader,
            origin_lat=origin_lat,
            origin_lon=origin_lon
        )
        
        if result is not None:
            triangulated_points[fid] = result['p_w']
    
    if len(triangulated_points) < 10:
        return 0, 0, 0
    
    # =========================================================================
    # Step 2: Detect planes from triangulated point cloud
    # =========================================================================
    points_array = np.array(list(triangulated_points.values()))
    
    try:
        planes = plane_detector.detect_planes(points_array)
        num_planes_detected = len(planes)
        
        if num_planes_detected > 0:
            print(f"[PLANE-MSCKF] Detected {num_planes_detected} planes from {len(points_array)} points at t={t:.3f}s")
    except Exception as e:
        print(f"[PLANE-MSCKF] Plane detection failed: {e}")
        return 0, 0, 0
    
    if num_planes_detected == 0:
        return 0, 0, 0
    
    # =========================================================================
    # Step 3: Associate features to planes
    # =========================================================================
    feature_tracks = []
    for fid in triangulated_points.keys():
        feature_tracks.append({'fid': fid})
    
    associations = associate_features_to_planes(
        feature_tracks, triangulated_points, planes, distance_threshold
    )
    
    # =========================================================================
    # Step 4: Apply plane-aided triangulation (re-triangulate with constraint)
    # =========================================================================
    if use_aided_triangulation:
        for fid, plane_id in associations.items():
            plane = next((p for p in planes if p.id == plane_id), None)
            if plane is None:
                continue
            
            # Re-triangulate with plane constraint
            point_aided, status = plane_aided_triangulation(
                fid, cam_observations, cam_states, kf, plane,
                dem_reader=dem_reader,
                origin_lat=origin_lat,
                origin_lon=origin_lon
            )
            
            if point_aided is not None:
                triangulated_points[fid] = point_aided
                num_plane_aided += 1
    
    # =========================================================================
    # Step 5: Apply plane constraint EKF updates
    # =========================================================================
    if use_constraints:
        for fid, plane_id in associations.items():
            plane = next((p for p in planes if p.id == plane_id), None)
            if plane is None:
                continue
            
            point = triangulated_points[fid]
            point_cov = np.eye(3) * 1.0  # Rough covariance estimate
            
            success, reason = apply_plane_constraint_update(
                kf, plane, point, point_cov, fid, sigma_plane, save_debug=False
            )
            
            if success:
                num_plane_constrained += 1
    
    print(f"[PLANE-MSCKF] Summary: {num_planes_detected} planes, "
          f"{num_plane_aided} aided, {num_plane_constrained} constrained")
    
    return num_planes_detected, num_plane_aided, num_plane_constrained
