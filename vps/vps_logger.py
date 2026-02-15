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
        self.profile_csv = None
        self.diagnostics_csv = None
        
        if self.enabled:
            self._init_files()
    
    def _init_files(self):
        """Initialize CSV files with headers."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # VPS Attempts CSV
        self.attempts_csv = os.path.join(self.output_dir, "debug_vps_attempts.csv")
        with open(self.attempts_csv, "w", newline="") as f:
            f.write(
                "t,frame,est_lat,est_lon,est_alt,est_yaw_deg,success,reason,processing_time_ms,"
                "patch_size_px,candidate_idx,num_candidates,selected_yaw_deg,selected_scale_mult,"
                "selected_content_ratio,selected_texture_std,best_num_matches,best_num_inliers,"
                "best_reproj_error,best_confidence\n"
            )
        
        # VPS Matches CSV
        self.matches_csv = os.path.join(self.output_dir, "debug_vps_matches.csv")
        with open(self.matches_csv, "w", newline="") as f:
            f.write("t,frame,vps_lat,vps_lon,innovation_x,innovation_y,innovation_mag,"
                   "num_features,num_inliers,confidence,tile_zoom,delayed_update\n")

        # VPS Stage Profile CSV (tile/preprocess/match/pose breakdown)
        self.profile_csv = os.path.join(self.output_dir, "debug_vps_profile.csv")
        with open(self.profile_csv, "w", newline="") as f:
            f.write(
                "t,frame,success,reason,total_ms,tile_ms,preprocess_ms,match_ms,pose_ms,"
                "num_matches,num_inliers,reproj_error,confidence,content_ratio,texture_std\n"
            )

        # Candidate diagnostics (one row per hypothesis)
        self.diagnostics_csv = os.path.join(self.output_dir, "debug_vps_diagnostics.csv")
        with open(self.diagnostics_csv, "w", newline="") as f:
            f.write(
                "t,frame,candidate_idx,num_candidates,patch_size_px,yaw_delta_deg,scale_mult,"
                "rotation_deg,content_ratio,texture_std,num_matches,num_inliers,reproj_error,"
                "confidence,match_success,decision\n"
            )
    
    def log_attempt(self, 
                   t: float,
                   frame: int,
                   est_lat: float,
                   est_lon: float,
                   est_alt: float,
                   est_yaw_deg: float,
                   success: bool,
                   reason: str,
                   processing_time_ms: float = 0.0,
                   patch_size_px: int = 0,
                   candidate_idx: int = -1,
                   num_candidates: int = 0,
                   selected_yaw_deg: float = float("nan"),
                   selected_scale_mult: float = float("nan"),
                   selected_content_ratio: float = float("nan"),
                   selected_texture_std: float = float("nan"),
                   best_num_matches: int = 0,
                   best_num_inliers: int = 0,
                   best_reproj_error: float = float("nan"),
                   best_confidence: float = float("nan")):
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
                f.write(
                    f"{t:.6f},{frame},{est_lat:.8f},{est_lon:.8f},{est_alt:.2f},"
                    f"{est_yaw_deg:.2f},{int(success)},{reason},{processing_time_ms:.2f},"
                    f"{int(patch_size_px)},{int(candidate_idx)},{int(num_candidates)},"
                    f"{selected_yaw_deg:.2f},{selected_scale_mult:.3f},"
                    f"{selected_content_ratio:.4f},{selected_texture_std:.3f},"
                    f"{int(best_num_matches)},{int(best_num_inliers)},"
                    f"{best_reproj_error:.3f},{best_confidence:.3f}\n"
                )
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

    def log_profile(self,
                    t: float,
                    frame: int,
                    success: bool,
                    reason: str,
                    total_ms: float,
                    tile_ms: float,
                    preprocess_ms: float,
                    match_ms: float,
                    pose_ms: float,
                    num_matches: int = 0,
                    num_inliers: int = 0,
                    reproj_error: float = float("nan"),
                    confidence: float = float("nan"),
                    content_ratio: float = float("nan"),
                    texture_std: float = float("nan")):
        """Log per-stage VPS processing time for profiling."""
        if not self.enabled or self.profile_csv is None:
            return
        try:
            with open(self.profile_csv, "a", newline="") as f:
                f.write(
                    f"{t:.6f},{frame},{int(success)},{reason},"
                    f"{total_ms:.2f},{tile_ms:.2f},{preprocess_ms:.2f},"
                    f"{match_ms:.2f},{pose_ms:.2f},"
                    f"{int(num_matches)},{int(num_inliers)},{reproj_error:.3f},"
                    f"{confidence:.3f},{content_ratio:.4f},{texture_std:.3f}\n"
                )
        except Exception:
            pass

    def log_candidate(self,
                      t: float,
                      frame: int,
                      candidate_idx: int,
                      num_candidates: int,
                      patch_size_px: int,
                      yaw_delta_deg: float,
                      scale_mult: float,
                      rotation_deg: float,
                      content_ratio: float,
                      texture_std: float,
                      num_matches: int,
                      num_inliers: int,
                      reproj_error: float,
                      confidence: float,
                      match_success: bool,
                      decision: str):
        """Log per-candidate VPS matching diagnostics."""
        if not self.enabled or self.diagnostics_csv is None:
            return
        try:
            with open(self.diagnostics_csv, "a", newline="") as f:
                f.write(
                    f"{t:.6f},{frame},{int(candidate_idx)},{int(num_candidates)},{int(patch_size_px)},"
                    f"{yaw_delta_deg:.2f},{scale_mult:.3f},{rotation_deg:.2f},"
                    f"{content_ratio:.4f},{texture_std:.3f},{int(num_matches)},{int(num_inliers)},"
                    f"{reproj_error:.3f},{confidence:.3f},{int(match_success)},{decision}\n"
                )
        except Exception:
            pass
