#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPS Runner - Main Orchestrator

Coordinates all VPS components for position estimation.
Called at camera frame rate from VIO main loop.

Author: VIO project
"""

import time
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .tile_cache import TileCache, MapPatch
from .image_preprocessor import VPSImagePreprocessor, PreprocessResult
from .satellite_matcher import SatelliteMatcher, MatchResult
from .vps_pose_estimator import VPSPoseEstimator, VPSMeasurement


@dataclass
class VPSConfig:
    """Configuration for VPS system."""
    mbtiles_path: str
    
    # Processing
    output_size: tuple = (512, 512)
    
    # Quality thresholds
    min_inliers: int = 20
    max_reproj_error: float = 5.0
    min_confidence: float = 0.3
    
    # Altitude limits (AGL)
    min_altitude: float = 30.0
    max_altitude: float = 500.0
    
    # Timing
    min_update_interval: float = 0.5  # seconds
    
    # Matcher
    device: str = 'cuda'
    max_keypoints: int = 2048


class VPSRunner:
    """
    Main VPS orchestrator.
    
    Workflow:
    1. Get map patch from TileCache centered at VIO estimate
    2. Preprocess drone image (undistort, rotate, scale)
    3. Match drone image against satellite map
    4. Estimate lat/lon position with covariance
    
    Called from VIO main loop when camera frame is available.
    """
    
    def __init__(self,
                 mbtiles_path: str,
                 fisheye_rectifier=None,
                 camera_intrinsics: Optional[Dict] = None,
                 config: Optional[VPSConfig] = None,
                 device: str = 'cuda',
                 camera_yaw_offset_rad: float = 0.0,
                 save_matches_dir: Optional[str] = None):
        """
        Initialize VPS runner.
        
        Args:
            mbtiles_path: Path to cached satellite tiles
            fisheye_rectifier: Optional FisheyeRectifier from vio module
            camera_intrinsics: Camera intrinsics (KB_PARAMS dict)
            config: VPSConfig, uses defaults if None
            device: 'cuda' or 'cpu' for matcher
            camera_yaw_offset_rad: Camera-to-body yaw offset in radians
                                   (from extrinsics calibration)
            save_matches_dir: Optional directory to save match visualization images
        """
        # Config
        if config is None:
            config = VPSConfig(mbtiles_path=mbtiles_path, device=device)
        self.config = config
        
        # Initialize components
        print("[VPSRunner] Initializing...")
        
        # 1. Tile cache
        self.tile_cache = TileCache(mbtiles_path)
        print(f"[VPSRunner] TileCache loaded: {self.tile_cache.get_tile_count()} tiles")
        
        # 2. Image preprocessor
        self.preprocessor = VPSImagePreprocessor(
            fisheye_rectifier=fisheye_rectifier,
            camera_intrinsics=camera_intrinsics,
            output_size=config.output_size
        )
        
        # 3. Satellite matcher
        self.matcher = SatelliteMatcher(
            device=config.device,
            max_keypoints=config.max_keypoints,
            min_inliers=config.min_inliers
        )
        
        # 4. Pose estimator
        self.pose_estimator = VPSPoseEstimator()
        
        # Camera-Body extrinsics offset
        self.camera_yaw_offset_rad = camera_yaw_offset_rad
        if abs(camera_yaw_offset_rad) > 0.01:
            import math
            print(f"[VPSRunner] Camera yaw offset: {math.degrees(camera_yaw_offset_rad):.2f}°")
        
        # State
        self.last_update_time = 0.0
        self.last_result: Optional[VPSMeasurement] = None
        
        # Debug logger (set via set_logger)
        self.logger = None
        
        # Save match visualizations directory
        self.save_matches_dir = save_matches_dir
        if save_matches_dir:
            import os
            os.makedirs(save_matches_dir, exist_ok=True)
            print(f"[VPSRunner] Match visualizations will be saved to: {save_matches_dir}")
        
        # Flag for delayed update (stochastic cloning) - set by caller
        self.delayed_update_enabled = False
        
        # Statistics
        self.stats = {
            'total_attempts': 0,
            'success': 0,
            'fail_no_coverage': 0,
            'fail_match': 0,
            'fail_quality': 0,
        }
        
        print("[VPSRunner] Ready")
    
    @classmethod
    def create_from_config(cls, 
                           mbtiles_path: str,
                           config_path: str,
                           device: str = 'cuda') -> 'VPSRunner':
        """
        Factory method to create VPSRunner from VIO config file.
        
        Automatically loads camera intrinsics and creates fisheye rectifier.
        
        Args:
            mbtiles_path: Path to MBTiles file
            config_path: Path to VIO YAML config file
            device: 'cuda' or 'cpu' for matcher
            
        Returns:
            VPSRunner instance with proper camera configuration
            
        Example:
            vps = VPSRunner.create_from_config(
                "mission.mbtiles",
                "configs/config_bell412_dataset3.yaml"
            )
        """
        import yaml
        import os
        
        fisheye_rectifier = None
        camera_intrinsics = None
        
        # Load camera intrinsics from config
        if os.path.exists(config_path):
            print(f"[VPSRunner] Loading config: {config_path}")
            with open(config_path, 'r') as f:
                config_yaml = yaml.safe_load(f)
            
            if 'camera' in config_yaml:
                cam = config_yaml['camera']
                camera_intrinsics = {
                    'mu': cam.get('mu', 500),
                    'mv': cam.get('mv', 500),
                    'u0': cam.get('u0', 720),
                    'v0': cam.get('v0', 540),
                    'w': cam.get('image_width', 1440),
                    'h': cam.get('image_height', 1080),
                    'k2': cam.get('k2', 0),
                    'k3': cam.get('k3', 0),
                    'k4': cam.get('k4', 0),
                    'k5': cam.get('k5', 0),
                }
                print(f"[VPSRunner] Camera: mu={camera_intrinsics['mu']:.1f}, "
                      f"mv={camera_intrinsics['mv']:.1f}")
            
            # Create fisheye rectifier
            try:
                import sys
                # Add parent for vio imports
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from vio.fisheye_rectifier import create_rectifier_from_config
                
                rectifier_config = {'KB_PARAMS': camera_intrinsics}
                src_size = (camera_intrinsics['w'], camera_intrinsics['h'])
                fisheye_rectifier = create_rectifier_from_config(
                    rectifier_config, 
                    src_size=src_size,
                    fov_deg=90.0
                )
                print(f"[VPSRunner] Fisheye rectifier created (FOV=90°)")
            except Exception as e:
                print(f"[VPSRunner] Warning: Fisheye rectifier failed: {e}")
            
            # Load extrinsics and calculate camera yaw offset
            camera_yaw_offset_rad = 0.0
            try:
                import math
                # Get camera view from VPS config (default: nadir)
                vps_cfg = config_yaml.get('vps', {})
                camera_view = vps_cfg.get('camera_view', 'nadir')
                
                if 'extrinsics' in config_yaml and camera_view in config_yaml['extrinsics']:
                    t_matrix = config_yaml['extrinsics'][camera_view]['transform']
                    R00 = t_matrix[0][0]
                    R10 = t_matrix[1][0]
                    
                    # Calculate yaw offset from rotation matrix
                    # atan2(R10, R00) gives azimuth of Camera X-axis (Right)
                    # Subtract 90° to get azimuth of Camera Up-axis (-Y)
                    yaw_x_axis = math.atan2(R10, R00)
                    camera_yaw_offset_rad = yaw_x_axis - math.radians(90)
                    
                    print(f"[VPSRunner] Extrinsics loaded: {camera_view} -> "
                          f"yaw offset = {math.degrees(camera_yaw_offset_rad):.2f}°")
            except Exception as e:
                print(f"[VPSRunner] Warning: Failed to load extrinsics: {e}")
        else:
            print(f"[VPSRunner] Warning: Config not found: {config_path}")
            camera_yaw_offset_rad = 0.0
        
        # Create VPSRunner
        return cls(
            mbtiles_path=mbtiles_path,
            fisheye_rectifier=fisheye_rectifier,
            camera_intrinsics=camera_intrinsics,
            device=device,
            camera_yaw_offset_rad=camera_yaw_offset_rad
        )
    
    def process_frame(self,
                      img: np.ndarray,
                      t_cam: float,
                      est_lat: float,
                      est_lon: float,
                      est_yaw: float,
                      est_alt: float,
                      frame_idx: int = -1) -> Optional[VPSMeasurement]:
        """
        Process single camera frame for VPS position.
        
        This is the MAIN entry point, called from VIO main loop
        when a new camera frame is available.
        
        Args:
            img: Camera image (can be fisheye or rectified)
            t_cam: Camera timestamp
            est_lat: Estimated latitude from VIO
            est_lon: Estimated longitude from VIO
            est_yaw: Estimated yaw in radians (ENU)
            est_alt: Estimated altitude AGL in meters
            
        Returns:
            VPSMeasurement if successful, None otherwise
        
        Note:
            frame_idx: Optional frame index from VIO for logging. Default -1 if not provided.
        """
        t_start = time.time()
        self.stats['total_attempts'] += 1
        t_tile_start = t_start
        t_preprocess_start = t_start
        t_match_start = t_start
        t_pose_start = t_start
        tile_ms = preprocess_ms = match_ms = pose_ms = 0.0

        def _log_attempt_and_profile(success: bool, reason: str):
            processing_time_ms = (time.time() - t_start) * 1000.0
            if self.logger:
                import math
                self.logger.log_attempt(
                    t=t_cam,
                    frame=frame_idx,
                    est_lat=est_lat,
                    est_lon=est_lon,
                    est_alt=est_alt,
                    est_yaw_deg=math.degrees(est_yaw),
                    success=success,
                    reason=reason,
                    processing_time_ms=processing_time_ms,
                )
                self.logger.log_profile(
                    t=t_cam,
                    frame=frame_idx,
                    success=success,
                    reason=reason,
                    total_ms=processing_time_ms,
                    tile_ms=tile_ms,
                    preprocess_ms=preprocess_ms,
                    match_ms=match_ms,
                    pose_ms=pose_ms,
                )
            return processing_time_ms

        def _save_match_visualization(tag: str,
                                      preprocess_result: Optional[PreprocessResult] = None,
                                      map_patch: Optional[MapPatch] = None,
                                      match_result: Optional[MatchResult] = None):
            """Persist VPS matching visualization for both success and failure cases."""
            if not self.save_matches_dir or frame_idx < 0:
                return
            if preprocess_result is None or map_patch is None or match_result is None:
                return
            try:
                import os
                safe_tag = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(tag))
                output_path = os.path.join(
                    self.save_matches_dir,
                    f"vps_{safe_tag}_{frame_idx:06d}_{t_cam:.2f}s.jpg",
                )
                self.matcher.visualize_matches(
                    drone_img=preprocess_result.image,
                    sat_img=map_patch.image,
                    result=match_result,
                    output_path=output_path,
                )
            except Exception as e:
                print(f"[VPS] Failed to save match visualization ({tag}): {e}")
        
        # 1. Check update interval
        if t_cam - self.last_update_time < self.config.min_update_interval:
            return None
        
        # 2. Check altitude limits
        if est_alt < self.config.min_altitude or est_alt > self.config.max_altitude:
            return None
        
        # 3. Check tile coverage
        if not self.tile_cache.is_position_in_cache(est_lat, est_lon):
            self.stats['fail_no_coverage'] += 1
            _log_attempt_and_profile(False, "no_coverage")
            return None
        
        # 4. Get satellite map patch
        t_tile_start = time.time()
        map_patch = self.tile_cache.get_map_patch(
            est_lat, est_lon, 
            patch_size_px=self.config.output_size[0]
        )
        tile_ms = (time.time() - t_tile_start) * 1000.0
        
        if map_patch is None:
            self.stats['fail_no_coverage'] += 1
            _log_attempt_and_profile(False, "no_map_patch")
            return None
        if map_patch.image is None or map_patch.image.size == 0:
            self.stats['fail_match'] += 1
            _log_attempt_and_profile(False, "empty_map_patch")
            return None
        if map_patch.image.shape[0] <= 1 or map_patch.image.shape[1] <= 1:
            self.stats['fail_match'] += 1
            _log_attempt_and_profile(False, "degenerate_map_patch")
            return None
        if not np.isfinite(float(map_patch.meters_per_pixel)) or float(map_patch.meters_per_pixel) <= 0.0:
            self.stats['fail_match'] += 1
            _log_attempt_and_profile(False, "invalid_map_gsd")
            return None
        if img is None or getattr(img, "size", 0) == 0:
            self.stats['fail_match'] += 1
            _log_attempt_and_profile(False, "empty_input_image")
            return None
        
        # 5. Preprocess drone image
        # Apply camera-body yaw offset (from extrinsics calibration)
        camera_yaw = est_yaw + self.camera_yaw_offset_rad
        t_preprocess_start = time.time()
        try:
            preprocess_result = self.preprocessor.preprocess(
                img=img,
                yaw_rad=camera_yaw,
                altitude_m=est_alt,
                target_gsd=map_patch.meters_per_pixel,
                grayscale=True
            )
        except Exception:
            self.stats['fail_match'] += 1
            _log_attempt_and_profile(False, "preprocess_failed")
            return None
        preprocess_ms = (time.time() - t_preprocess_start) * 1000.0
        if preprocess_result.image is None or preprocess_result.image.size == 0:
            self.stats['fail_match'] += 1
            _log_attempt_and_profile(False, "empty_preprocessed")
            return None
        if preprocess_result.image.shape[0] <= 1 or preprocess_result.image.shape[1] <= 1:
            self.stats['fail_match'] += 1
            _log_attempt_and_profile(False, "degenerate_preprocessed")
            return None
        
        # 6. Match against satellite
        t_match_start = time.time()
        try:
            match_result = self.matcher.match_with_homography(
                drone_img=preprocess_result.image,
                sat_img=map_patch.image if len(map_patch.image.shape) == 2 
                        else map_patch.image[:,:,0]  # Use single channel
            )
        except Exception:
            self.stats['fail_match'] += 1
            _log_attempt_and_profile(False, "matcher_failed")
            return None
        match_ms = (time.time() - t_match_start) * 1000.0
        
        if not match_result.success:
            self.stats['fail_match'] += 1
            _save_match_visualization("match_failed", preprocess_result, map_patch, match_result)
            _log_attempt_and_profile(False, "match_failed")
            return None
        
        # 7. Quality check
        if match_result.num_inliers < self.config.min_inliers:
            self.stats['fail_quality'] += 1
            _save_match_visualization("quality_inliers", preprocess_result, map_patch, match_result)
            _log_attempt_and_profile(False, "quality_inliers")
            return None
        
        if match_result.reproj_error > self.config.max_reproj_error:
            self.stats['fail_quality'] += 1
            _save_match_visualization("quality_reproj", preprocess_result, map_patch, match_result)
            _log_attempt_and_profile(False, "quality_reproj")
            return None
        
        if match_result.confidence < self.config.min_confidence:
            self.stats['fail_quality'] += 1
            _save_match_visualization("quality_confidence", preprocess_result, map_patch, match_result)
            _log_attempt_and_profile(False, "quality_confidence")
            return None
        
        # 8. Compute VPS measurement
        t_pose_start = time.time()
        try:
            vps_measurement = self.pose_estimator.compute_vps_measurement(
                match_result=match_result,
                map_gsd=map_patch.meters_per_pixel,
                map_center_lat=map_patch.center_lat,
                map_center_lon=map_patch.center_lon,
                t_cam=t_cam
            )
        except Exception:
            self.stats['fail_match'] += 1
            _log_attempt_and_profile(False, "pose_estimation_exception")
            return None
        pose_ms = (time.time() - t_pose_start) * 1000.0
        
        if vps_measurement is None:
            self.stats['fail_match'] += 1
            _save_match_visualization("pose_failed", preprocess_result, map_patch, match_result)
            _log_attempt_and_profile(False, "pose_estimation_failed")
            return None
        
        # Success!
        self.stats['success'] += 1
        self.last_update_time = t_cam
        self.last_result = vps_measurement
        
        # Log processing time
        processing_time_ms = _log_attempt_and_profile(True, "matched")
        
        # Debug logging
        if self.logger:
            # Log match details
            self.logger.log_match(
                t=t_cam,
                frame=frame_idx,
                vps_lat=vps_measurement.lat,
                vps_lon=vps_measurement.lon,
                innovation_x=vps_measurement.offset_m[0],
                innovation_y=vps_measurement.offset_m[1],
                innovation_mag=float(np.linalg.norm(vps_measurement.offset_m)),
                num_features=match_result.num_matches,
                num_inliers=match_result.num_inliers,
                confidence=match_result.confidence,
                tile_zoom=19,  # Default zoom level
                delayed_update=self.delayed_update_enabled  # From stochastic cloning setup
            )
        
        # Console log
        print(f"[VPS] t={t_cam:.2f}: "
              f"inliers={match_result.num_inliers}, "
              f"err={match_result.reproj_error:.2f}px, "
              f"Δ=({vps_measurement.offset_m[0]:.1f}, {vps_measurement.offset_m[1]:.1f})m, "
              f"σ={np.sqrt(vps_measurement.R_vps[0,0]):.2f}m, "
              f"time={processing_time_ms:.0f}ms")
        
        # Save match visualization if directory is set
        _save_match_visualization("matched", preprocess_result, map_patch, match_result)
        
        return vps_measurement
    
    def get_last_result(self) -> Optional[VPSMeasurement]:
        """Get last successful VPS measurement."""
        return self.last_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total = self.stats['total_attempts']
        return {
            **self.stats,
            'success_rate': self.stats['success'] / total if total > 0 else 0,
        }
    
    def print_statistics(self):
        """Print statistics summary."""
        stats = self.get_statistics()
        print(f"\n[VPS] Statistics:")
        print(f"  Total attempts: {stats['total_attempts']}")
        print(f"  Success: {stats['success']} ({stats['success_rate']*100:.1f}%)")
        print(f"  No coverage: {stats['fail_no_coverage']}")
        print(f"  Match failed: {stats['fail_match']}")
        print(f"  Quality reject: {stats['fail_quality']}")
    
    def set_logger(self, logger):
        """
        Attach debug logger for VPS processing.
        
        Args:
            logger: VPSDebugLogger instance
        """
        self.logger = logger
    
    def close(self):
        """Release resources."""
        self.tile_cache.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_vps_runner():
    """Test VPS runner with proper camera config."""
    import sys
    import os
    import argparse
    
    # Parse arguments using argparse (consistent with run_vio.py)
    parser = argparse.ArgumentParser(
        description="VPS Runner Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults:
  python -m vps.vps_runner --mbtiles mission.mbtiles
  
  # With custom config:
  python -m vps.vps_runner --mbtiles mission.mbtiles --config configs/config.yaml
  
  # With custom position:
  python -m vps.vps_runner --mbtiles mission.mbtiles --lat 45.315 --lon -75.670
        """
    )
    
    parser.add_argument("--mbtiles", type=str, 
                        default="mission.mbtiles",
                        help="Path to MBTiles file (default: mission.mbtiles)")
    parser.add_argument("--config", type=str,
                        default="configs/config_bell412_dataset3.yaml",
                        help="Path to YAML config file (default: configs/config_bell412_dataset3.yaml)")
    parser.add_argument("--lat", type=float,
                        default=45.315721787845,
                        help="Test latitude (default: 45.315721787845)")
    parser.add_argument("--lon", type=float,
                        default=-75.670671305696,
                        help="Test longitude (default: -75.670671305696)")
    parser.add_argument("--device", type=str,
                        default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for matcher (default: cpu)")
    
    args = parser.parse_args()
    
    mbtiles_path = args.mbtiles
    config_path = args.config
    test_lat = args.lat
    test_lon = args.lon
    
    print("=" * 60)
    print("Testing VPSRunner with Real Camera Config")
    print("=" * 60)
    print(f"  MBTiles: {mbtiles_path}")
    print(f"  Config:  {config_path}")
    print(f"  Test position: ({test_lat:.6f}, {test_lon:.6f})")
    print(f"  Device: {args.device}")
    
    # Check if mbtiles exists
    if not os.path.exists(mbtiles_path):
        print(f"\n❌ Error: MBTiles file not found: {mbtiles_path}")
        print("Please provide a valid MBTiles file or create one using:")
        print("  python -m vps.tile_prefetcher --center LAT,LON --radius 500 -o mission.mbtiles")
        return
    
    # Try to load real image from bell412 dataset
    images_dir = "/Users/france/Downloads/vio_dataset/bell412_dataset3/extracted_data_new/cam_data/camera__image_mono/images"
    import cv2
    
    if os.path.exists(images_dir):
        import random
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        if image_files:
            selected_file = random.choice(image_files)
            img = cv2.imread(os.path.join(images_dir, selected_file))
            print(f"  Image: {selected_file} ({img.shape})")
        else:
            img = np.random.randint(50, 200, (1080, 1440, 3), dtype=np.uint8)
            print("  Image: synthetic (no images found)")
    else:
        img = np.random.randint(50, 200, (1080, 1440, 3), dtype=np.uint8)
        print("  Image: synthetic (dataset not found)")
    
    print("-" * 60)
    
    # Create VPSRunner with proper config
    if os.path.exists(config_path):
        vps = VPSRunner.create_from_config(
            mbtiles_path=mbtiles_path,
            config_path=config_path,
            device=args.device
        )
    else:
        print(f"Config not found, using defaults")
        vps = VPSRunner(mbtiles_path, device=args.device)
    
    with vps:
        result = vps.process_frame(
            img=img,
            t_cam=1.0,
            est_lat=test_lat,
            est_lon=test_lon,
            est_yaw=0.0,
            est_alt=100.0
        )
        
        if result:
            print(f"\n✅ VPS Result:")
            print(f"  Position: ({result.lat:.6f}, {result.lon:.6f})")
            print(f"  Offset: ({result.offset_m[0]:.2f}, {result.offset_m[1]:.2f}) m")
            print(f"  Sigma: {np.sqrt(result.R_vps[0,0]):.2f} m")
        else:
            print("\n⚠️ VPS returned no result (may need real imagery matching location)")
        
        vps.print_statistics()


if __name__ == "__main__":
    test_vps_runner()
