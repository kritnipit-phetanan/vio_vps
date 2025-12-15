#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plane Utilities Module

Helper functions for plane representation, manipulation, and geometric operations.
Follows OpenVINS ov_plane conventions.

Plane Representation:
    π = [n^T, d]^T where n·x + d = 0
    - n: unit normal vector [nx, ny, nz] (3D)
    - d: signed distance from origin (scalar)
    - Constraint: ||n|| = 1

References:
    [1] Geneva et al., "OpenVINS: A Research Platform for Visual-Inertial Estimation", 2020
    [2] Hsiao et al., "Optimization-based Visual-Inertial SLAM with Planes", 2018

Author: VIO project
"""

import numpy as np
from typing import Tuple, List, Optional


class Plane:
    """
    Plane representation for SLAM/MSCKF.
    
    Attributes:
        normal: Unit normal vector [nx, ny, nz]
        d: Signed distance from origin
        id: Unique plane ID
        feature_ids: List of feature IDs on this plane
        observation_count: Number of times observed
        last_seen_time: Last observation timestamp
        is_slam: True if SLAM plane (in state), False if MSCKF plane
    """
    
    def __init__(self, normal: np.ndarray, d: float, plane_id: int = -1):
        """
        Initialize plane.
        
        Args:
            normal: Normal vector (will be normalized)
            d: Distance from origin
            plane_id: Unique identifier
        """
        self.normal = normal / (np.linalg.norm(normal) + 1e-12)
        self.d = float(d)
        self.id = plane_id
        self.feature_ids = []
        self.observation_count = 0
        self.last_seen_time = 0.0
        self.is_slam = False
        
    def point_distance(self, point: np.ndarray) -> float:
        """
        Compute signed distance from point to plane.
        
        Args:
            point: 3D point [x, y, z]
        
        Returns:
            Signed distance (positive = same side as normal)
        """
        return float(np.dot(self.normal, point) + self.d)
    
    def project_point(self, point: np.ndarray) -> np.ndarray:
        """
        Project point onto plane.
        
        Args:
            point: 3D point [x, y, z]
        
        Returns:
            Projected point on plane
        """
        dist = self.point_distance(point)
        return point - dist * self.normal
    
    def angle_to(self, other: 'Plane') -> float:
        """
        Compute angle between two planes (via normals).
        
        Args:
            other: Another plane
        
        Returns:
            Angle in radians [0, π]
        """
        cos_angle = np.clip(np.dot(self.normal, other.normal), -1.0, 1.0)
        return np.arccos(abs(cos_angle))  # abs for undirected angle
    
    def is_similar_to(self, other: 'Plane', 
                      angle_threshold: float = np.radians(10.0),
                      distance_threshold: float = 0.5) -> bool:
        """
        Check if two planes are similar (for merging).
        
        Args:
            other: Another plane
            angle_threshold: Maximum angle difference (radians)
            distance_threshold: Maximum distance difference (meters)
        
        Returns:
            True if planes are similar
        """
        angle_diff = self.angle_to(other)
        
        # Distance difference (project origin to both planes)
        # If normals are aligned: dist_diff ≈ |d1 - d2|
        # If normals are opposite: dist_diff ≈ |d1 + d2|
        if np.dot(self.normal, other.normal) > 0:
            dist_diff = abs(self.d - other.d)
        else:
            dist_diff = abs(self.d + other.d)
        
        return angle_diff < angle_threshold and dist_diff < distance_threshold
    
    def to_array(self) -> np.ndarray:
        """Convert to array [nx, ny, nz, d]."""
        return np.array([self.normal[0], self.normal[1], self.normal[2], self.d])
    
    @staticmethod
    def from_array(arr: np.ndarray, plane_id: int = -1) -> 'Plane':
        """Create plane from array [nx, ny, nz, d]."""
        return Plane(arr[0:3], arr[3], plane_id)
    
    @staticmethod
    def from_points(points: np.ndarray, method: str = 'svd') -> Optional['Plane']:
        """
        Fit plane to 3D points.
        
        Args:
            points: Nx3 array of points
            method: 'svd' or 'lstsq'
        
        Returns:
            Fitted plane or None if fitting fails
        """
        if points.shape[0] < 3:
            return None
        
        # Center points
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        if method == 'svd':
            # SVD: normal is smallest singular vector
            try:
                U, S, Vt = np.linalg.svd(centered)
                normal = Vt[-1, :]  # Last row = smallest singular vector
            except np.linalg.LinAlgError:
                return None
        else:  # lstsq
            # Solve Ax + By + Cz = D where [A,B,C] is normal
            A = np.c_[centered[:, 0], centered[:, 1], np.ones(points.shape[0])]
            b = -centered[:, 2]
            
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                normal = np.array([x[0], x[1], 1.0])
                normal = normal / np.linalg.norm(normal)
            except np.linalg.LinAlgError:
                return None
        
        # Compute d = -n·centroid
        d = -np.dot(normal, centroid)
        
        return Plane(normal, d)
    
    def __repr__(self) -> str:
        return (f"Plane(id={self.id}, normal=[{self.normal[0]:.3f}, {self.normal[1]:.3f}, "
                f"{self.normal[2]:.3f}], d={self.d:.3f}, obs={self.observation_count}, "
                f"slam={self.is_slam})")


def merge_planes(planes: List[Plane], 
                 angle_threshold: float = np.radians(10.0),
                 distance_threshold: float = 0.5) -> List[Plane]:
    """
    Merge similar planes.
    
    Args:
        planes: List of planes
        angle_threshold: Maximum angle for merging
        distance_threshold: Maximum distance for merging
    
    Returns:
        List of merged planes
    """
    if len(planes) <= 1:
        return planes
    
    merged = []
    used = set()
    
    for i, p1 in enumerate(planes):
        if i in used:
            continue
        
        # Find all planes similar to p1
        group = [p1]
        group_ids = [i]
        
        for j, p2 in enumerate(planes[i+1:], start=i+1):
            if j in used:
                continue
            
            if p1.is_similar_to(p2, angle_threshold, distance_threshold):
                group.append(p2)
                group_ids.append(j)
        
        # Merge group by averaging normals and d
        if len(group) == 1:
            merged.append(p1)
        else:
            # Collect all points from features
            all_normals = np.array([p.normal for p in group])
            all_d = np.array([p.d for p in group])
            
            # Average normal and normalize
            avg_normal = np.mean(all_normals, axis=0)
            avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-12)
            
            avg_d = np.mean(all_d)
            
            merged_plane = Plane(avg_normal, avg_d, p1.id)
            merged_plane.observation_count = sum(p.observation_count for p in group)
            merged_plane.last_seen_time = max(p.last_seen_time for p in group)
            
            # Merge feature IDs
            for p in group:
                merged_plane.feature_ids.extend(p.feature_ids)
            merged_plane.feature_ids = list(set(merged_plane.feature_ids))
            
            merged.append(merged_plane)
        
        used.update(group_ids)
    
    return merged


def compute_plane_jacobian(plane: Plane, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Jacobian of plane-to-point distance w.r.t. plane parameters and point.
    
    Distance: z = n^T·p + d
    
    Args:
        plane: Plane
        point: 3D point
    
    Returns:
        H_plane: Jacobian w.r.t. plane [∂z/∂n, ∂z/∂d] (1x4)
        H_point: Jacobian w.r.t. point [∂z/∂p] (1x3)
    """
    # ∂z/∂n = p^T (point coordinates)
    # ∂z/∂d = 1
    H_plane = np.zeros((1, 4))
    H_plane[0, 0:3] = point
    H_plane[0, 3] = 1.0
    
    # ∂z/∂p = n^T (plane normal)
    H_point = plane.normal.reshape(1, 3)
    
    return H_plane, H_point


def filter_outlier_points(points: np.ndarray, plane: Plane, 
                          threshold: float = 0.1) -> np.ndarray:
    """
    Filter points that are too far from plane.
    
    Args:
        points: Nx3 array of points
        plane: Reference plane
        threshold: Maximum distance (meters)
    
    Returns:
        Filtered points (subset of input)
    """
    distances = np.abs([plane.point_distance(p) for p in points])
    mask = distances < threshold
    return points[mask]


def plane_intersection_line(p1: Plane, p2: Plane) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute line of intersection between two planes.
    
    Args:
        p1: First plane
        p2: Second plane
    
    Returns:
        point: A point on the intersection line
        direction: Direction vector of line
    
    Raises:
        ValueError: If planes are parallel
    """
    # Direction: perpendicular to both normals
    direction = np.cross(p1.normal, p2.normal)
    
    if np.linalg.norm(direction) < 1e-6:
        raise ValueError("Planes are parallel")
    
    direction = direction / np.linalg.norm(direction)
    
    # Find a point on the line
    # Solve: n1·p + d1 = 0, n2·p + d2 = 0
    # Set z=0 and solve for x,y
    A = np.array([
        [p1.normal[0], p1.normal[1]],
        [p2.normal[0], p2.normal[1]]
    ])
    b = np.array([-p1.d, -p2.d])
    
    try:
        xy = np.linalg.solve(A, b)
        point = np.array([xy[0], xy[1], 0.0])
    except np.linalg.LinAlgError:
        # Try y=0 instead
        A = np.array([
            [p1.normal[0], p1.normal[2]],
            [p2.normal[0], p2.normal[2]]
        ])
        xz = np.linalg.solve(A, b)
        point = np.array([xz[0], 0.0, xz[1]])
    
    return point, direction
