#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPS Standalone Test Script

Tests the VPS pipeline without VIO integration.
Measures inference time for each component.

Usage:
    # Basic test with defaults
    python -m vps.test_vps_standalone
    
    # Use VIO config for camera intrinsics
    python -m vps.test_vps_standalone --vio-config configs/config_dji_m600_quarry.yaml
    
    # Use position from quarry file
    python -m vps.test_vps_standalone --quarry /path/to/quarry.csv

Inputs required:
    1. MBTiles file (satellite tiles)
    2. Camera images (or synthetic)
    3. Approximate position (lat, lon, yaw, alt)
    4. Camera intrinsics (from VIO config or defaults)

Author: VIO project
"""

import os
import sys
import time
import argparse
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not available")


def load_vio_config(config_path: str) -> dict:
    """
    Load camera intrinsics from VIO YAML config.
    
    Returns:
        dict with 'mu', 'mv', 'u0', 'v0', 'w', 'h' keys
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from vio.config import load_config
        
        config = load_config(config_path)
        kb_params = config._raw_config.get('KB_PARAMS', {})
        print(f"[Config] Loaded camera intrinsics from {config_path}")
        print(f"  Focal length: ({kb_params.get('mu', 'N/A')}, {kb_params.get('mv', 'N/A')}) px")
        print(f"  Image size: {kb_params.get('w', 'N/A')} x {kb_params.get('h', 'N/A')}")
        return kb_params
    except Exception as e:
        print(f"[Config] Failed to load VIO config: {e}")
        return {'mu': 500, 'mv': 500, 'u0': 720, 'v0': 540, 'w': 1440, 'h': 1080}


def load_position_from_ppk(ppk_path: str) -> dict:
    """
    Load initial position from PPK ground truth file (.pos).
    
    Uses VIO's data_loaders.load_ppk_initial_state() for parsing.
    Falls back to manual parsing if VIO module not available.
    
    Returns:
        dict with 'lat', 'lon', 'alt', 'yaw' keys
    """
    try:
        # Try to use VIO's loader first
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from vio.data_loaders import load_ppk_initial_state
        
        ppk_state = load_ppk_initial_state(ppk_path)
        if ppk_state:
            result = {
                'lat': ppk_state.lat,
                'lon': ppk_state.lon,
                'alt': ppk_state.height,
                'yaw': ppk_state.yaw,  # Already in radians
            }
            print(f"[PPK] Loaded position from {ppk_path}")
            print(f"  Position: ({result['lat']:.6f}, {result['lon']:.6f})")
            print(f"  Altitude: {result['alt']:.1f} m, Yaw: {np.degrees(result['yaw']):.1f}°")
            return result
    except ImportError:
        print("[PPK] VIO module not available, using manual parsing")
    except Exception as e:
        print(f"[PPK] VIO loader failed: {e}")
    
    # Fallback: Manual PPK parsing
    try:
        with open(ppk_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('%')]
        
        if len(lines) == 0:
            raise ValueError("Empty PPK file")
        
        parts = lines[0].split()
        if len(parts) < 18:
            raise ValueError(f"PPK format error: expected 18+ columns, got {len(parts)}")
        
        result = {
            'lat': float(parts[2]),
            'lon': float(parts[3]),
            'alt': float(parts[4]),  # Ellipsoidal height
            'yaw': np.radians(float(parts[17])),  # Yaw in degrees -> radians
        }
        
        print(f"[PPK] Loaded position from {ppk_path} (manual parse)")
        print(f"  Position: ({result['lat']:.6f}, {result['lon']:.6f})")
        print(f"  Altitude: {result['alt']:.1f} m, Yaw: {np.degrees(result['yaw']):.1f}°")
        return result
        
    except Exception as e:
        print(f"[PPK] Failed to load: {e}")
        return {'lat': 45.315721787845, 'lon': -75.670671305696, 'alt': 100.0, 'yaw': 0.0}


def create_synthetic_image(width: int = 1440, height: int = 1080) -> np.ndarray:
    """Create synthetic test image with features."""
    img = np.random.randint(50, 150, (height, width, 3), dtype=np.uint8)
    
    # Add some structure (buildings, roads)
    for i in range(5):
        x = np.random.randint(100, width - 100)
        y = np.random.randint(100, height - 100)
        w = np.random.randint(50, 150)
        h = np.random.randint(50, 150)
        cv2.rectangle(img, (x, y), (x + w, y + h), 
                     (np.random.randint(100, 200),) * 3, -1)
    
    # Roads
    for i in range(3):
        y = np.random.randint(0, height)
        cv2.line(img, (0, y), (width, y), (80, 80, 80), 20)
    
    return img


def test_tile_cache(mbtiles_path: str, lat: float, lon: float):
    """Test tile cache component."""
    from vps.tile_cache import TileCache
    
    print("\n" + "="*60)
    print("Testing TileCache")
    print("="*60)
    
    t0 = time.time()
    cache = TileCache(mbtiles_path)
    init_time = (time.time() - t0) * 1000
    print(f"Init time: {init_time:.1f} ms")
    print(f"Total tiles: {cache.get_tile_count()}")
    print(f"GSD at location: {cache.get_gsd(lat):.4f} m/px")
    
    # Test coverage check
    t0 = time.time()
    has_coverage = cache.is_position_in_cache(lat, lon)
    check_time = (time.time() - t0) * 1000
    print(f"Coverage check: {has_coverage} ({check_time:.2f} ms)")
    
    if not has_coverage:
        print("❌ No tile coverage at this location!")
        return None, None
    
    # Test map patch retrieval
    t0 = time.time()
    patch = cache.get_map_patch(lat, lon, patch_size_px=512)
    patch_time = (time.time() - t0) * 1000
    
    if patch:
        print(f"Map patch: {patch.image.shape}, GSD={patch.meters_per_pixel:.4f} m/px ({patch_time:.1f} ms)")
        return cache, patch
    else:
        print("❌ Failed to get map patch")
        return cache, None


def test_preprocessor(camera_img: np.ndarray, yaw: float, altitude: float, target_gsd: float):
    """Test image preprocessor component."""
    from vps.image_preprocessor import VPSImagePreprocessor
    
    print("\n" + "="*60)
    print("Testing VPSImagePreprocessor")
    print("="*60)
    
    t0 = time.time()
    preprocessor = VPSImagePreprocessor(
        camera_intrinsics={'fx': 500, 'fy': 500},
        output_size=(512, 512)
    )
    init_time = (time.time() - t0) * 1000
    print(f"Init time: {init_time:.1f} ms")
    
    # Preprocess
    t0 = time.time()
    result = preprocessor.preprocess(
        img=camera_img,
        yaw_rad=yaw,
        altitude_m=altitude,
        target_gsd=target_gsd,
        grayscale=True
    )
    preprocess_time = (time.time() - t0) * 1000
    
    print(f"Input: {camera_img.shape}")
    print(f"Output: {result.image.shape}")
    print(f"Drone GSD: {result.drone_gsd:.4f} m/px")
    print(f"Scale factor: {result.scale_factor:.3f}")
    print(f"Rotation: {result.rotation_deg:.1f}°")
    print(f"Preprocess time: {preprocess_time:.1f} ms")
    
    return result


def test_matcher(drone_img: np.ndarray, sat_img: np.ndarray, device: str = 'cpu'):
    """Test satellite matcher component."""
    from vps.satellite_matcher import SatelliteMatcher
    
    print("\n" + "="*60)
    print("Testing SatelliteMatcher")
    print("="*60)
    
    t0 = time.time()
    matcher = SatelliteMatcher(device=device, min_inliers=10)
    init_time = (time.time() - t0) * 1000
    print(f"Matcher type: {'LightGlue' if matcher.use_lightglue else 'ORB'}")
    print(f"Init time: {init_time:.1f} ms")
    
    # Match
    t0 = time.time()
    result = matcher.match_with_homography(drone_img, sat_img)
    match_time = (time.time() - t0) * 1000
    
    print(f"Success: {result.success}")
    print(f"Matches: {result.num_matches}")
    print(f"Inliers: {result.num_inliers}")
    print(f"Reproj error: {result.reproj_error:.2f} px")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Offset: ({result.offset_px[0]:.1f}, {result.offset_px[1]:.1f}) px")
    print(f"Match time: {match_time:.1f} ms")
    
    return result, match_time


def test_pose_estimator(match_result, gsd: float, center_lat: float, center_lon: float):
    """Test pose estimator component."""
    from vps.vps_pose_estimator import VPSPoseEstimator
    
    print("\n" + "="*60)
    print("Testing VPSPoseEstimator")
    print("="*60)
    
    t0 = time.time()
    estimator = VPSPoseEstimator()
    measurement = estimator.compute_vps_measurement(
        match_result=match_result,
        map_gsd=gsd,
        map_center_lat=center_lat,
        map_center_lon=center_lon,
        t_cam=1.0
    )
    estimate_time = (time.time() - t0) * 1000
    
    if measurement:
        print(f"VPS Position: ({measurement.lat:.6f}, {measurement.lon:.6f})")
        print(f"Offset: ({measurement.offset_m[0]:.2f}, {measurement.offset_m[1]:.2f}) m")
        print(f"Sigma: {np.sqrt(measurement.R_vps[0,0]):.2f} m")
        print(f"Estimate time: {estimate_time:.1f} ms")
    else:
        print("❌ Failed to compute measurement")
    
    return measurement


def test_full_pipeline(mbtiles_path: str, lat: float, lon: float, 
                       yaw: float = 0.0, altitude: float = 100.0,
                       camera_img: np.ndarray = None,
                       device: str = 'cpu',
                       num_runs: int = 5):
    """Test full VPS pipeline and measure inference time."""
    from vps.vps_runner import VPSRunner
    
    print("\n" + "="*60)
    print("Full Pipeline Test (VPSRunner)")
    print("="*60)
    
    if camera_img is None:
        print("Creating synthetic camera image...")
        camera_img = create_synthetic_image()
    
    # Initialize
    t0 = time.time()
    vps = VPSRunner(mbtiles_path, device=device)
    init_time = (time.time() - t0) * 1000
    print(f"VPSRunner init time: {init_time:.1f} ms")
    
    # Run multiple times for timing
    times = []
    results = []
    
    print(f"\nRunning {num_runs} iterations...")
    for i in range(num_runs):
        t0 = time.time()
        result = vps.process_frame(
            img=camera_img,
            t_cam=float(i + 1),  # Different timestamp each time
            est_lat=lat,
            est_lon=lon,
            est_yaw=yaw,
            est_alt=altitude
        )
        elapsed = (time.time() - t0) * 1000
        times.append(elapsed)
        results.append(result)
        
        status = "✓" if result else "✗"
        print(f"  Run {i+1}: {elapsed:.1f} ms {status}")
    
    # Statistics
    times = np.array(times)
    print(f"\nTiming Statistics:")
    print(f"  Mean: {np.mean(times):.1f} ms")
    print(f"  Std:  {np.std(times):.1f} ms")
    print(f"  Min:  {np.min(times):.1f} ms")
    print(f"  Max:  {np.max(times):.1f} ms")
    print(f"  FPS:  {1000 / np.mean(times):.1f}")
    
    vps.print_statistics()
    vps.close()
    
    return times, results


def visualize_results(mbtiles_path: str, lat: float, lon: float,
                      camera_img: np.ndarray, output_dir: str = None):
    """Visualize VPS matching results."""
    from vps.tile_cache import TileCache
    from vps.image_preprocessor import VPSImagePreprocessor
    from vps.satellite_matcher import SatelliteMatcher
    
    if output_dir is None:
        output_dir = os.path.dirname(mbtiles_path)
    
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # Get map patch
    cache = TileCache(mbtiles_path)
    patch = cache.get_map_patch(lat, lon, patch_size_px=512)
    
    if patch is None:
        print("❌ No coverage")
        return
    
    # Preprocess
    preprocessor = VPSImagePreprocessor(output_size=(512, 512))
    prep_result = preprocessor.preprocess(
        img=camera_img,
        yaw_rad=0.0,
        altitude_m=100.0,
        target_gsd=patch.meters_per_pixel,
        grayscale=False
    )
    
    # Match
    matcher = SatelliteMatcher(device='cpu')
    
    # Convert to grayscale for matching
    drone_gray = cv2.cvtColor(prep_result.image, cv2.COLOR_BGR2GRAY) if len(prep_result.image.shape) == 3 else prep_result.image
    sat_gray = cv2.cvtColor(patch.image, cv2.COLOR_BGR2GRAY) if len(patch.image.shape) == 3 else patch.image
    
    match_result = matcher.match_with_homography(drone_gray, sat_gray)
    
    # Visualize
    vis_path = os.path.join(output_dir, "vps_match_visualization.jpg")
    vis = matcher.visualize_matches(prep_result.image, patch.image, match_result, vis_path)
    print(f"Saved visualization to: {vis_path}")
    
    # Save individual images
    cv2.imwrite(os.path.join(output_dir, "vps_drone_preprocessed.jpg"), prep_result.image)
    cv2.imwrite(os.path.join(output_dir, "vps_satellite_patch.jpg"), patch.image)
    print(f"Saved preprocessed drone image and satellite patch")
    
    cache.close()


def main():
    parser = argparse.ArgumentParser(description='Test VPS pipeline standalone')
    parser.add_argument('--mbtiles', '-m', type=str, 
                        default='vps/test_tiles_ottawa.mbtiles',
                        help='Path to MBTiles file')
    parser.add_argument('--lat', type=float, default=None,
                        help='Test latitude (overrides --quarry)')
    parser.add_argument('--lon', type=float, default=None,
                        help='Test longitude (overrides --quarry)')
    parser.add_argument('--yaw', type=float, default=0.0,
                        help='Test yaw (radians)')
    parser.add_argument('--altitude', type=float, default=100.0,
                        help='Test altitude AGL (meters)')
    parser.add_argument('--image', '-i', type=str, default=None,
                        help='Camera image path (uses synthetic if not provided)')
    parser.add_argument('--device', '-d', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for matcher')
    parser.add_argument('--runs', '-n', type=int, default=5,
                        help='Number of timing runs')
    parser.add_argument('--component', '-c', type=str, default='all',
                        choices=['all', 'cache', 'preprocessor', 'matcher', 'pipeline'],
                        help='Component to test')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate visualization images')
    
    # VIO integration options
    parser.add_argument('--vio-config', type=str, default=None,
                        help='Path to VIO YAML config for camera intrinsics')
    parser.add_argument('--ppk', type=str, default=None,
                        help='Path to PPK .pos file for initial position')
    
    args = parser.parse_args()
    
    # Load camera intrinsics from VIO config
    kb_params = None
    if args.vio_config and os.path.exists(args.vio_config):
        kb_params = load_vio_config(args.vio_config)
    
    # Load position from PPK if not manually specified
    if args.ppk and os.path.exists(args.ppk):
        pos_data = load_position_from_ppk(args.ppk)
        if args.lat is None:
            args.lat = pos_data['lat']
        if args.lon is None:
            args.lon = pos_data['lon']
        args.altitude = pos_data['alt']
        args.yaw = pos_data['yaw']
    
    # Use defaults if still not set
    if args.lat is None:
        args.lat = 45.315721787845
    if args.lon is None:
        args.lon = -75.670671305696
    
    print("="*60)
    print("VPS Standalone Test")
    print("="*60)
    print(f"MBTiles: {args.mbtiles}")
    print(f"Location: ({args.lat:.6f}, {args.lon:.6f})")
    print(f"Altitude: {args.altitude} m")
    print(f"Device: {args.device}")
    if kb_params:
        print(f"Camera: fx={kb_params.get('mu', 'N/A')}, fy={kb_params.get('mv', 'N/A')}")
    
    # Check mbtiles exists
    if not os.path.exists(args.mbtiles):
        print(f"\n❌ MBTiles file not found: {args.mbtiles}")
        print("Please provide a valid MBTiles file or download tiles first:")
        print("  python -m vps.tile_prefetcher --center LAT,LON --radius 500 -o tiles.mbtiles")
        return 1
    
    # Load or create camera image
    if args.image and os.path.exists(args.image):
        print(f"Loading camera image: {args.image}")
        camera_img = cv2.imread(args.image)
    else:
        print("Using synthetic camera image")
        camera_img = create_synthetic_image()
    
    # Run tests
    if args.component in ['all', 'cache']:
        cache, patch = test_tile_cache(args.mbtiles, args.lat, args.lon)
        
    if args.component in ['all', 'preprocessor']:
        gsd = 0.3 if args.component == 'preprocessor' else patch.meters_per_pixel
        prep_result = test_preprocessor(camera_img, args.yaw, args.altitude, gsd)
        
    if args.component in ['all', 'matcher']:
        if args.component == 'all' and patch:
            drone_gray = prep_result.image
            sat_gray = cv2.cvtColor(patch.image, cv2.COLOR_BGR2GRAY) if len(patch.image.shape) == 3 else patch.image
            match_result, match_time = test_matcher(drone_gray, sat_gray, args.device)
            
            if match_result.success:
                measurement = test_pose_estimator(
                    match_result, 
                    patch.meters_per_pixel, 
                    args.lat, args.lon
                )
        else:
            # Standalone matcher test with synthetic images
            img1 = create_synthetic_image(512, 512)[:,:,0]
            img2 = create_synthetic_image(512, 512)[:,:,0]
            test_matcher(img1, img2, args.device)
    
    if args.component in ['all', 'pipeline']:
        test_full_pipeline(
            args.mbtiles, args.lat, args.lon,
            args.yaw, args.altitude, camera_img,
            args.device, args.runs
        )
    
    if args.visualize:
        visualize_results(args.mbtiles, args.lat, args.lon, camera_img)
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
