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
                 camera_yaw_offset_rad: float = 0.0):
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
                      est_alt: float) -> Optional[VPSMeasurement]:
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
        """
        t_start = time.time()
        self.stats['total_attempts'] += 1
        
        # 1. Check update interval
        if t_cam - self.last_update_time < self.config.min_update_interval:
            return None
        
        # 2. Check altitude limits
        if est_alt < self.config.min_altitude or est_alt > self.config.max_altitude:
            return None
        
        # 3. Check tile coverage
        if not self.tile_cache.is_position_in_cache(est_lat, est_lon):
            self.stats['fail_no_coverage'] += 1
            return None
        
        # 4. Get satellite map patch
        map_patch = self.tile_cache.get_map_patch(
            est_lat, est_lon, 
            patch_size_px=self.config.output_size[0]
        )
        
        if map_patch is None:
            self.stats['fail_no_coverage'] += 1
            return None
        
        # 5. Preprocess drone image
        # Apply camera-body yaw offset (from extrinsics calibration)
        camera_yaw = est_yaw + self.camera_yaw_offset_rad
        
        preprocess_result = self.preprocessor.preprocess(
            img=img,
            yaw_rad=camera_yaw,
            altitude_m=est_alt,
            target_gsd=map_patch.meters_per_pixel,
            grayscale=True
        )
        
        # 6. Match against satellite
        match_result = self.matcher.match_with_homography(
            drone_img=preprocess_result.image,
            sat_img=map_patch.image if len(map_patch.image.shape) == 2 
                    else map_patch.image[:,:,0]  # Use single channel
        )
        
        if not match_result.success:
            self.stats['fail_match'] += 1
            return None
        
        # 7. Quality check
        if match_result.num_inliers < self.config.min_inliers:
            self.stats['fail_quality'] += 1
            return None
        
        if match_result.reproj_error > self.config.max_reproj_error:
            self.stats['fail_quality'] += 1
            return None
        
        if match_result.confidence < self.config.min_confidence:
            self.stats['fail_quality'] += 1
            return None
        
        # 8. Compute VPS measurement
        vps_measurement = self.pose_estimator.compute_vps_measurement(
            match_result=match_result,
            map_gsd=map_patch.meters_per_pixel,
            map_center_lat=map_patch.center_lat,
            map_center_lon=map_patch.center_lon,
            t_cam=t_cam
        )
        
        if vps_measurement is None:
            self.stats['fail_match'] += 1
            return None
        
        # Success!
        self.stats['success'] += 1
        self.last_update_time = t_cam
        self.last_result = vps_measurement
        
        # Log
        processing_time = time.time() - t_start
        print(f"[VPS] t={t_cam:.2f}: "
              f"inliers={match_result.num_inliers}, "
              f"err={match_result.reproj_error:.2f}px, "
              f"Δ=({vps_measurement.offset_m[0]:.1f}, {vps_measurement.offset_m[1]:.1f})m, "
              f"σ={np.sqrt(vps_measurement.R_vps[0,0]):.2f}m, "
              f"time={processing_time*1000:.0f}ms")
        
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
    
    def close(self):
        """Clean up resources."""
        self.tile_cache.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_vps_runner():
    """Test VPS runner with proper camera config."""
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python vps_runner.py <mbtiles_path> [config_path] [lat] [lon]")
        print("Skipping test (no mbtiles file provided)")
        return
    
    mbtiles_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "configs/config_bell412_dataset3.yaml"
    test_lat = float(sys.argv[3]) if len(sys.argv) > 3 else 45.315721787845
    test_lon = float(sys.argv[4]) if len(sys.argv) > 4 else -75.670671305696
    
    print("=" * 60)
    print("Testing VPSRunner with Real Camera Config")
    print("=" * 60)
    print(f"  MBTiles: {mbtiles_path}")
    print(f"  Config:  {config_path}")
    print(f"  Test position: ({test_lat:.6f}, {test_lon:.6f})")
    
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
            device='cpu'
        )
    else:
        print(f"Config not found, using defaults")
        vps = VPSRunner(mbtiles_path, device='cpu')
    
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
