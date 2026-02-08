#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPS Debug Logger

Handles VPS frame processing logs for debugging and analysis.
Similar to vio/output_utils.py but specific to VPS operations.

Author: VIO project
"""

import os
import time
from typing import Optional


class VPSDebugLogger:
    """
    Manages VPS debug logging (similar to vio/output_utils.py).
    
    Creates CSV files:
    - debug_vps_attempts.csv: All VPS processing attempts
    - debug_vps_matches.csv: Successful matches with metrics
    
    Usage:
        logger = VPSDebugLogger(output_dir="output/", enabled=True)
        logger.log_attempt(t=1.0, frame=10, est_lat=45.3, est_lon=-75.6, 
                          success=True, reason="matched")
        logger.log_match(t=1.0, frame=10, vps_lat=45.301, vps_lon=-75.601,
                        innovation_m=5.2, num_features=150, ...)
    """
    
    def __init__(self, output_dir: str, enabled: bool = False):
        """
        Initialize VPS debug logger.
        
        Args:
            output_dir: Output directory for CSV files
            enabled: Whether to enable logging
        """
        self.output_dir = output_dir
        self.enabled = enabled
        
        self.attempts_csv = None
        self.matches_csv = None
        
        if self.enabled:
            self._init_files()
    
    def _init_files(self):
        """Initialize CSV files with headers."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # VPS Attempts CSV
        self.attempts_csv = os.path.join(self.output_dir, "debug_vps_attempts.csv")
        with open(self.attempts_csv, "w", newline="") as f:
            f.write("t,frame,est_lat,est_lon,est_alt,est_yaw_deg,success,reason,processing_time_ms\n")
        
        # VPS Matches CSV
        self.matches_csv = os.path.join(self.output_dir, "debug_vps_matches.csv")
        with open(self.matches_csv, "w", newline="") as f:
            f.write("t,frame,vps_lat,vps_lon,innovation_x,innovation_y,innovation_mag,"
                   "num_features,num_inliers,confidence,tile_zoom,delayed_update\n")
    
    def log_attempt(self, 
                   t: float,
                   frame: int,
                   est_lat: float,
                   est_lon: float,
                   est_alt: float,
                   est_yaw_deg: float,
                   success: bool,
                   reason: str,
                   processing_time_ms: float = 0.0):
        """
        Log a VPS processing attempt.
        
        Args:
            t: Timestamp
            frame: VIO frame index (-1 if not applicable)
            est_lat: Estimated latitude
            est_lon: Estimated longitude
            est_alt: Estimated altitude
            est_yaw_deg: Estimated yaw in degrees
            success: Whether VPS succeeded
            reason: Reason for success/failure ("matched", "no_coverage", "match_failed", etc.)
            processing_time_ms: Processing time in milliseconds
        """
        if not self.enabled or self.attempts_csv is None:
            return
        
        try:
            with open(self.attempts_csv, "a", newline="") as f:
                f.write(f"{t:.6f},{frame},{est_lat:.8f},{est_lon:.8f},{est_alt:.2f},"
                       f"{est_yaw_deg:.2f},{int(success)},{reason},{processing_time_ms:.2f}\n")
        except Exception:
            pass
    
    def log_match(self,
                 t: float,
                 frame: int,
                 vps_lat: float,
                 vps_lon: float,
                 innovation_x: float,
                 innovation_y: float,
                 innovation_mag: float,
                 num_features: int,
                 num_inliers: int,
                 confidence: float,
                 tile_zoom: int = 19,
                 delayed_update: bool = False):
        """
        Log a successful VPS match.
        
        Args:
            t: Timestamp
            frame: VIO frame index (-1 if not applicable)
            vps_lat: VPS measured latitude
            vps_lon: VPS measured longitude
            innovation_x: Innovation in X (meters)
            innovation_y: Innovation in Y (meters)
            innovation_mag: Innovation magnitude (meters)
            num_features: Number of features matched
            num_inliers: Number of inlier features
            confidence: Match confidence score
            tile_zoom: Tile zoom level used
            delayed_update: Whether using delayed update (stochastic cloning)
        """
        if not self.enabled or self.matches_csv is None:
            return
        
        try:
            with open(self.matches_csv, "a", newline="") as f:
                f.write(f"{t:.6f},{frame},{vps_lat:.8f},{vps_lon:.8f},"
                       f"{innovation_x:.3f},{innovation_y:.3f},{innovation_mag:.3f},"
                       f"{num_features},{num_inliers},{confidence:.3f},"
                       f"{tile_zoom},{int(delayed_update)}\n")
        except Exception:
            pass
