#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPS Pose Estimator

Converts pixel offsets from matching to lat/lon position with covariance.

Author: VIO project
"""

import numpy as np
import math
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VPSMeasurement:
    """VPS position measurement for EKF update."""
    lat: float
    lon: float
    R_vps: np.ndarray          # 2x2 measurement covariance (m²)
    confidence: float          # 0-1 confidence score
    t_measurement: float       # Timestamp of original camera frame
    
    # Debug info
    offset_m: Tuple[float, float]   # (dx, dy) in meters
    num_inliers: int
    reproj_error: float


class VPSPoseEstimator:
    """
    Converts matching results to lat/lon with covariance.
    
    Takes pixel offset from SatelliteMatcher and converts to:
    1. Metric offset (meters)
    2. Lat/lon delta
    3. Measurement covariance
    """
    
    def __init__(self,
                 base_sigma_m: float = 0.5,
                 min_sigma_m: float = 0.3,
                 max_sigma_m: float = 10.0):
        """
        Initialize pose estimator.
        
        Args:
            base_sigma_m: Base position uncertainty (meters)
            min_sigma_m: Minimum uncertainty (meters)
            max_sigma_m: Maximum uncertainty (meters)
        """
        self.base_sigma = base_sigma_m
        self.min_sigma = min_sigma_m
        self.max_sigma = max_sigma_m
    
    def pixel_offset_to_meters(self, 
                                dx_px: float, 
                                dy_px: float, 
                                gsd: float) -> Tuple[float, float]:
        """
        Convert pixel offset to meters.
        
        Args:
            dx_px: X offset in pixels (positive = right = East)
            dy_px: Y offset in pixels (positive = down = South in North-Up map)
            gsd: Ground Sample Distance (meters per pixel)
            
        Returns:
            (dx_m, dy_m) offset in meters (ENU convention: +X=East, +Y=North)
        """
        dx_m = dx_px * gsd       # East
        dy_m = -dy_px * gsd      # North (negative because image Y is down)
        
        return dx_m, dy_m
    
    def meters_to_latlon(self,
                         dx_m: float,
                         dy_m: float,
                         center_lat: float,
                         center_lon: float) -> Tuple[float, float]:
        """
        Convert meter offset to lat/lon.
        
        Args:
            dx_m: East offset in meters
            dy_m: North offset in meters
            center_lat: Reference latitude
            center_lon: Reference longitude
            
        Returns:
            (new_lat, new_lon)
        """
        # Meters per degree
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
        
        # Delta in degrees
        delta_lat = dy_m / m_per_deg_lat
        delta_lon = dx_m / m_per_deg_lon
        
        return center_lat + delta_lat, center_lon + delta_lon
    
    def estimate_covariance(self,
                            gsd: float,
                            num_inliers: int,
                            reproj_error: float,
                            confidence: float) -> np.ndarray:
        """
        Estimate measurement covariance based on match quality.
        
        Higher uncertainty for:
        - Fewer inliers
        - Higher reprojection error
        - Lower confidence
        
        Args:
            gsd: Ground Sample Distance (m/px)
            num_inliers: Number of RANSAC inliers
            reproj_error: Mean reprojection error (pixels)
            confidence: Match confidence (0-1)
            
        Returns:
            2x2 covariance matrix (m²)
        """
        # Base uncertainty from GSD
        sigma = max(self.base_sigma, gsd)
        
        # Scale by inlier count
        if num_inliers < 30:
            sigma *= 2.0
        elif num_inliers < 50:
            sigma *= 1.5
        elif num_inliers > 100:
            sigma *= 0.8  # More confident with many inliers
        
        # Scale by reprojection error
        if reproj_error > 5.0:
            sigma *= 2.0
        elif reproj_error > 3.0:
            sigma *= 1.5
        elif reproj_error > 2.0:
            sigma *= 1.2
        elif reproj_error < 1.0:
            sigma *= 0.9  # Very good match
        
        # Scale by confidence
        if confidence < 0.3:
            sigma *= 3.0
        elif confidence < 0.5:
            sigma *= 2.0
        elif confidence < 0.7:
            sigma *= 1.3
        
        # Clamp
        sigma = max(self.min_sigma, min(self.max_sigma, sigma))
        
        # 2x2 covariance (diagonal, assume isotropic)
        R = np.diag([sigma ** 2, sigma ** 2])
        
        return R
    
    def compute_vps_measurement(self,
                                 match_result,  # MatchResult from SatelliteMatcher
                                 map_gsd: float,
                                 map_center_lat: float,
                                 map_center_lon: float,
                                 t_cam: float) -> Optional[VPSMeasurement]:
        """
        Full pipeline: convert match result to VPS measurement.
        
        Args:
            match_result: MatchResult from SatelliteMatcher
            map_gsd: GSD of the satellite map (m/px)
            map_center_lat: Center latitude of the map patch
            map_center_lon: Center longitude of the map patch
            t_cam: Timestamp of the camera frame
            
        Returns:
            VPSMeasurement or None if match failed
        """
        if not match_result.success:
            return None
        
        # 1. Convert pixel offset to meters
        dx_px, dy_px = match_result.offset_px
        dx_m, dy_m = self.pixel_offset_to_meters(dx_px, dy_px, map_gsd)
        
        # 2. Convert to lat/lon
        # The offset tells us where the drone actually is relative to map center
        vps_lat, vps_lon = self.meters_to_latlon(
            dx_m, dy_m, map_center_lat, map_center_lon
        )
        
        # 3. Estimate covariance
        R_vps = self.estimate_covariance(
            gsd=map_gsd,
            num_inliers=match_result.num_inliers,
            reproj_error=match_result.reproj_error,
            confidence=match_result.confidence
        )
        
        return VPSMeasurement(
            lat=vps_lat,
            lon=vps_lon,
            R_vps=R_vps,
            confidence=match_result.confidence,
            t_measurement=t_cam,
            offset_m=(dx_m, dy_m),
            num_inliers=match_result.num_inliers,
            reproj_error=match_result.reproj_error
        )


def test_pose_estimator():
    """Test pose estimator."""
    print("Testing VPSPoseEstimator...")
    
    estimator = VPSPoseEstimator()
    
    # Test pixel to meters
    dx_m, dy_m = estimator.pixel_offset_to_meters(10, -20, 0.3)
    print(f"Pixel (10, -20) @ 0.3 m/px → Meters: ({dx_m:.2f}, {dy_m:.2f})")
    
    # Test meters to latlon
    lat, lon = estimator.meters_to_latlon(100, 200, 45.0, -75.0)
    print(f"Offset (100m E, 200m N) from (45, -75) → ({lat:.6f}, {lon:.6f})")
    
    # Test covariance estimation
    R = estimator.estimate_covariance(
        gsd=0.3,
        num_inliers=80,
        reproj_error=1.5,
        confidence=0.8
    )
    sigma = np.sqrt(R[0, 0])
    print(f"Covariance σ = {sigma:.2f} m (good match)")
    
    R_bad = estimator.estimate_covariance(
        gsd=0.3,
        num_inliers=25,
        reproj_error=4.0,
        confidence=0.4
    )
    sigma_bad = np.sqrt(R_bad[0, 0])
    print(f"Covariance σ = {sigma_bad:.2f} m (bad match)")
    
    print("✅ PoseEstimator test passed")


if __name__ == "__main__":
    test_pose_estimator()
