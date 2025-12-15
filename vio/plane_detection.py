#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plane Detection Module

Detect planar structures from 3D point clouds for plane-aided MSCKF.
Uses Delaunay triangulation and normal clustering.

Algorithm:
    1. Build Delaunay triangulation on projected 2D points
    2. Compute triangle normals
    3. Average normals at vertices
    4. Cluster vertices by normal similarity
    5. Fit plane to each cluster
    6. Filter small/unstable planes

References:
    [1] Geneva et al., "OpenVINS ov_plane", GitHub
    [2] Hsiao et al., "Optimization-based VI-SLAM with Planes", ICRA 2018

Author: VIO project
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.spatial import Delaunay
from collections import defaultdict

from .plane_utils import Plane, merge_planes, filter_outlier_points


class PlaneDetector:
    """
    Detect planes from triangulated 3D points.
    
    Configuration:
        min_points_per_plane: Minimum points to form a plane
        normal_angle_threshold: Max angle between normals for clustering (radians)
        distance_threshold: Max point-to-plane distance (meters)
        min_plane_area: Minimum plane area (m²)
    """
    
    def __init__(self,
                 min_points_per_plane: int = 10,
                 normal_angle_threshold: float = np.radians(15.0),
                 distance_threshold: float = 0.15,
                 min_plane_area: float = 0.5):
        """Initialize plane detector."""
        self.min_points = min_points_per_plane
        self.angle_threshold = normal_angle_threshold
        self.dist_threshold = distance_threshold
        self.min_area = min_plane_area
        
        # Next plane ID
        self.next_plane_id = 0
    
    def detect_planes(self, points: np.ndarray, 
                      feature_ids: Optional[List[int]] = None) -> List[Plane]:
        """
        Detect planes from 3D point cloud.
        
        Args:
            points: Nx3 array of 3D points in world frame
            feature_ids: Optional list of feature IDs corresponding to points
        
        Returns:
            List of detected planes
        """
        if points.shape[0] < self.min_points:
            return []
        
        if feature_ids is None:
            feature_ids = list(range(points.shape[0]))
        
        # Step 1: Build Delaunay triangulation (2D projection)
        triangles, vertex_triangles = self._triangulate(points)
        
        if triangles is None or len(triangles) == 0:
            return []
        
        # Step 2: Compute triangle normals
        triangle_normals = self._compute_triangle_normals(points, triangles)
        
        # Step 3: Compute vertex normals (average of adjacent triangles)
        vertex_normals = self._compute_vertex_normals(
            points.shape[0], vertex_triangles, triangle_normals
        )
        
        # Step 4: Cluster vertices by normal similarity
        clusters = self._cluster_by_normals(vertex_normals)
        
        # Step 5: Fit plane to each cluster
        planes = []
        for cluster_idx, vertex_indices in clusters.items():
            if len(vertex_indices) < self.min_points:
                continue
            
            cluster_points = points[vertex_indices]
            cluster_fids = [feature_ids[i] for i in vertex_indices]
            
            # Fit plane
            plane = Plane.from_points(cluster_points, method='svd')
            
            if plane is None:
                continue
            
            # Filter outliers
            filtered_points = filter_outlier_points(
                cluster_points, plane, self.dist_threshold
            )
            
            if filtered_points.shape[0] < self.min_points:
                continue
            
            # Refit after filtering
            plane = Plane.from_points(filtered_points, method='svd')
            
            if plane is None:
                continue
            
            # Check plane area (approximate from point spread)
            area = self._estimate_plane_area(filtered_points)
            
            if area < self.min_area:
                continue
            
            # Assign plane ID and feature IDs
            plane.id = self.next_plane_id
            self.next_plane_id += 1
            plane.feature_ids = cluster_fids
            plane.observation_count = 1
            
            planes.append(plane)
        
        # Step 6: Merge similar planes
        if len(planes) > 1:
            planes = merge_planes(
                planes,
                angle_threshold=self.angle_threshold,
                distance_threshold=self.dist_threshold
            )
        
        return planes
    
    def _triangulate(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Build Delaunay triangulation.
        
        Args:
            points: Nx3 array
        
        Returns:
            triangles: Mx3 array of vertex indices
            vertex_triangles: Dict mapping vertex_idx -> list of triangle indices
        """
        # Project to 2D (drop Z or use PCA)
        # Simple: use XY projection
        points_2d = points[:, 0:2]
        
        try:
            tri = Delaunay(points_2d)
            triangles = tri.simplices  # Mx3 array
            
            # Build vertex -> triangles mapping
            vertex_triangles = defaultdict(list)
            for tri_idx, triangle in enumerate(triangles):
                for vertex_idx in triangle:
                    vertex_triangles[vertex_idx].append(tri_idx)
            
            return triangles, dict(vertex_triangles)
        
        except Exception as e:
            print(f"[PLANE] Triangulation failed: {e}")
            return None, None
    
    def _compute_triangle_normals(self, points: np.ndarray, 
                                   triangles: np.ndarray) -> np.ndarray:
        """
        Compute normal for each triangle.
        
        Args:
            points: Nx3 points
            triangles: Mx3 triangle indices
        
        Returns:
            normals: Mx3 unit normals
        """
        normals = []
        
        for tri in triangles:
            p0, p1, p2 = points[tri[0]], points[tri[1]], points[tri[2]]
            
            # Cross product
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0, 1.0])  # Degenerate triangle
            
            normals.append(normal)
        
        return np.array(normals)
    
    def _compute_vertex_normals(self, num_vertices: int,
                                vertex_triangles: Dict[int, List[int]],
                                triangle_normals: np.ndarray) -> np.ndarray:
        """
        Compute normal at each vertex (average of adjacent triangles).
        
        Args:
            num_vertices: Number of vertices
            vertex_triangles: Mapping vertex -> triangle indices
            triangle_normals: Mx3 triangle normals
        
        Returns:
            vertex_normals: Nx3 unit normals
        """
        vertex_normals = np.zeros((num_vertices, 3))
        
        for v_idx in range(num_vertices):
            if v_idx not in vertex_triangles:
                vertex_normals[v_idx] = np.array([0.0, 0.0, 1.0])
                continue
            
            # Average normals of adjacent triangles
            adj_tri_indices = vertex_triangles[v_idx]
            adj_normals = triangle_normals[adj_tri_indices]
            
            avg_normal = np.mean(adj_normals, axis=0)
            norm = np.linalg.norm(avg_normal)
            
            if norm > 1e-9:
                vertex_normals[v_idx] = avg_normal / norm
            else:
                vertex_normals[v_idx] = np.array([0.0, 0.0, 1.0])
        
        return vertex_normals
    
    def _cluster_by_normals(self, vertex_normals: np.ndarray) -> Dict[int, List[int]]:
        """
        Cluster vertices by normal similarity (simple greedy clustering).
        
        Args:
            vertex_normals: Nx3 normals
        
        Returns:
            clusters: Dict mapping cluster_id -> list of vertex indices
        """
        num_vertices = vertex_normals.shape[0]
        labels = -np.ones(num_vertices, dtype=int)
        cluster_id = 0
        
        for i in range(num_vertices):
            if labels[i] >= 0:
                continue  # Already clustered
            
            # Start new cluster
            labels[i] = cluster_id
            seed_normal = vertex_normals[i]
            
            # Find similar normals
            for j in range(i+1, num_vertices):
                if labels[j] >= 0:
                    continue
                
                # Angle between normals
                cos_angle = np.dot(seed_normal, vertex_normals[j])
                angle = np.arccos(np.clip(abs(cos_angle), 0.0, 1.0))
                
                if angle < self.angle_threshold:
                    labels[j] = cluster_id
            
            cluster_id += 1
        
        # Build clusters dict
        clusters = defaultdict(list)
        for v_idx, c_id in enumerate(labels):
            if c_id >= 0:
                clusters[c_id].append(v_idx)
        
        return dict(clusters)
    
    def _estimate_plane_area(self, points: np.ndarray) -> float:
        """
        Estimate plane area from point spread.
        
        Args:
            points: Nx3 points on plane
        
        Returns:
            Estimated area (m²)
        """
        if points.shape[0] < 3:
            return 0.0
        
        # Simple: bounding box area
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        
        ranges = max_vals - min_vals
        
        # Use two largest ranges
        sorted_ranges = np.sort(ranges)[::-1]
        area = sorted_ranges[0] * sorted_ranges[1]
        
        return area


def update_plane_observations(planes: List[Plane], current_time: float,
                              max_age: float = 5.0) -> List[Plane]:
    """
    Update plane observation times and filter stale planes.
    
    Args:
        planes: List of planes
        current_time: Current timestamp
        max_age: Maximum age without observation (seconds)
    
    Returns:
        Active planes (not stale)
    """
    active = []
    
    for plane in planes:
        age = current_time - plane.last_seen_time
        
        if age < max_age:
            active.append(plane)
    
    return active


def promote_to_slam_plane(plane: Plane, min_observations: int = 5) -> bool:
    """
    Check if MSCKF plane should be promoted to SLAM plane.
    
    Args:
        plane: Plane to check
        min_observations: Minimum observation count
    
    Returns:
        True if should be promoted
    """
    return plane.observation_count >= min_observations
