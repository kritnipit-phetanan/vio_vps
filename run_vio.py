#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIO Standalone Entry Point (run_vio.py)

This script runs the VIO+EKF pipeline using the modular vio/ package.
It is completely independent of vio_vps.py and uses VIORunner directly.

The modular package provides:
- vio.config: Configuration loading
- vio.data_loaders: IMU, MAG, VPS, DEM data loading  
- vio.ekf: Extended Kalman Filter
- vio.imu_preintegration: IMU preintegration (Forster et al.)
- vio.magnetometer: Magnetometer calibration & yaw
- vio.vio_frontend: Visual front-end (feature tracking)
- vio.msckf: Multi-State Constraint Kalman Filter
- vio.propagation: IMU propagation, ZUPT
- vio.vps_integration: VPS/DEM updates
- vio.state_manager: State initialization
- vio.measurement_updates: Measurement update functions
- vio.output_utils: Debug output
- vio.main_loop: VIORunner orchestrator class

Usage:
    python run_vio.py --config configs/config_bell412_dataset3.yaml \\
        --imu path/to/imu.csv \\
        --quarry path/to/flight_log_from_gga.csv \\
        --output output_dir/

Author: VIO project
Version: 2.5.1 (Standalone - No dependency on vio_vps.py)
"""

import argparse
import sys
import os

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    """Parse command line arguments (same as vio_vps.py)."""
    parser = argparse.ArgumentParser(
        description="VIO+EKF Pipeline - Lightweight Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required inputs
    parser.add_argument("--imu", type=str, required=True, 
                        help="Path to IMU CSV file")
    parser.add_argument("--quarry", type=str, required=True, 
                        help="Path to flight_log_from_gga.csv")
    parser.add_argument("--output", type=str, required=True, 
                        help="Output directory")
    
    # Configuration
    parser.add_argument("--config", type=str, 
                        default="configs/config_bell412_dataset3.yaml",
                        help="Path to YAML config file")
    
    # Optional inputs
    parser.add_argument("--images_dir", type=str, default=None, 
                        help="Directory containing images")
    parser.add_argument("--images_index", type=str, default=None, 
                        help="Path to images_index.csv")
    parser.add_argument("--vps", type=str, default=None, 
                        help="Path to VPS results CSV")
    parser.add_argument("--mag", type=str, default=None, 
                        help="Path to magnetometer CSV")
    parser.add_argument("--dem", type=str, default=None, 
                        help="Path to DEM/DSM TIF file")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Path to PPK ground truth file")
    
    # Image processing
    parser.add_argument("--img_w", type=int, default=1440, 
                        help="Downscaled image width")
    parser.add_argument("--img_h", type=int, default=1080, 
                        help="Downscaled image height")
    
    # State configuration
    parser.add_argument("--z_state", type=str, default="msl", 
                        choices=["msl", "agl"],
                        help="Z state representation")
    parser.add_argument("--camera_view", type=str, default="nadir",
                        choices=["nadir", "front", "side"],
                        help="Camera view configuration")
    
    # Optional features
    parser.add_argument("--use_magnetometer", action="store_true", 
                        help="Enable magnetometer updates")
    parser.add_argument("--estimate_imu_bias", action="store_true", 
                        help="Enable IMU bias estimation")
    parser.add_argument("--use_vio_velocity", action="store_true", 
                        help="Enable VIO velocity updates")
    parser.add_argument("--save_debug_data", action="store_true", 
                        help="Save debug CSV files")
    parser.add_argument("--save_keyframe_images", action="store_true", 
                        help="Save keyframe overlay images")
    
    # Performance optimization (v2.9.9)
    parser.add_argument("--fast_mode", action="store_true",
                        help="Enable fast mode: reduce features + faster KLT (60%% speedup)")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Process every N frames (1=all, 2=half speed, etc.)")
    
    return parser.parse_args()


def main():
    """Main entry point - uses modular VIORunner directly."""
    args = parse_args()
    
    # Print header
    print("=" * 70)
    print("VIO+EKF Pipeline - Standalone Modular Entry Point")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"IMU: {args.imu}")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    # Import from modular vio package (no vio_vps.py dependency!)
    try:
        from vio import __version__
        from vio.main_loop import VIORunner, VIOConfig
        
        print(f"Using vio package version: {__version__}")
        
        # Create VIOConfig from command line arguments
        vio_config = VIOConfig(
            imu_path=args.imu,
            quarry_path=args.quarry,
            output_dir=args.output,
            config_yaml=args.config,
            images_dir=args.images_dir,
            images_index_csv=args.images_index,
            vps_csv=args.vps,
            mag_csv=args.mag,
            dem_path=args.dem,
            ground_truth_path=args.ground_truth,
            downscale_size=(args.img_w, args.img_h),
            z_state=args.z_state,
            camera_view=args.camera_view,
            estimate_imu_bias=args.estimate_imu_bias,
            use_magnetometer=args.use_magnetometer,
            use_vio_velocity=args.use_vio_velocity,
            save_debug_data=args.save_debug_data,
            save_keyframe_images=args.save_keyframe_images,
            use_preintegration=True,
            fast_mode=args.fast_mode,        # v2.9.9: Performance optimization
            frame_skip=args.frame_skip,      # v2.9.9: Process every N frames
        )
        
        # Create and run VIORunner
        runner = VIORunner(vio_config)
        runner.run()
        
        print("=" * 70)
        print("✅ VIO pipeline completed successfully")
        print(f"   Output: {args.output}")
        print("=" * 70)
        
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("\nMake sure you are in the vio_vps_repo directory.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running VIO pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
