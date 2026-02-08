#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPS Module - Offline Visual Positioning System

This module provides satellite-based position correction for GPS-denied navigation.
Uses pre-cached satellite tiles and feature matching to estimate position.

Components:
- tile_prefetcher: Pre-flight tile download
- tile_cache: In-flight tile query
- image_preprocessor: Drone image preprocessing
- satellite_matcher: LightGlue + SuperPoint matching
- vps_pose_estimator: Position estimation from matches
- vps_runner: Main orchestrator

Usage:
    # Pre-flight: Download tiles
    python -m vps.tile_prefetcher --center 45.315,-75.670 --radius 500 --output mission.mbtiles
    
    # In-flight: Use in VIO
    from vps import VPSRunner
    vps = VPSRunner(config)
    result = vps.process_frame(img, t_cam, est_lat, est_lon, est_yaw, est_alt)
"""

# Core tile management - always available (no heavy dependencies)
from .tile_prefetcher import (
    TilePrefetcher,
    lat_lon_to_tile,
    tile_to_lat_lon,
    get_tile_gsd,
)

# Components that require numpy/opencv/torch - import conditionally
_OPTIONAL_IMPORTS_AVAILABLE = True
_IMPORT_ERROR = None

try:
    from .tile_cache import TileCache, MapPatch
    from .image_preprocessor import VPSImagePreprocessor, PreprocessResult
    from .satellite_matcher import SatelliteMatcher, MatchResult
    from .vps_pose_estimator import VPSPoseEstimator, VPSMeasurement
    from .vps_runner import VPSRunner, VPSConfig
    from .vps_delayed_update import VPSDelayedUpdateManager, VPSStateClone
    from .vps_logger import VPSDebugLogger
except ImportError as e:
    _OPTIONAL_IMPORTS_AVAILABLE = False
    _IMPORT_ERROR = str(e)
    
    # Set to None for conditional checking
    TileCache = None
    MapPatch = None
    VPSImagePreprocessor = None
    PreprocessResult = None
    SatelliteMatcher = None
    MatchResult = None
    VPSPoseEstimator = None
    VPSMeasurement = None
    VPSRunner = None
    VPSConfig = None
    VPSDelayedUpdateManager = None
    VPSStateClone = None
    VPSDebugLogger = None


__all__ = [
    # Always available
    'TilePrefetcher',
    'lat_lon_to_tile',
    'tile_to_lat_lon',
    'get_tile_gsd',
    # Conditionally available
    'TileCache',
    'MapPatch',
    'VPSImagePreprocessor',
    'PreprocessResult',
    'SatelliteMatcher', 
    'MatchResult',
    'VPSPoseEstimator',
    'VPSMeasurement',
    'VPSRunner',
    'VPSConfig',
    'VPSDelayedUpdateManager',
    'VPSStateClone',
    'VPSDebugLogger',
]


__version__ = '0.1.0'

def check_dependencies() -> bool:
    """Check if all VPS dependencies are available."""
    if not _OPTIONAL_IMPORTS_AVAILABLE:
        print(f"[VPS] Warning: Some dependencies missing: {_IMPORT_ERROR}")
        print("[VPS] Install with: pip install numpy opencv-python torch")
        return False
    return True
