#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIO Standalone Entry Point (run_vio.py)

This script runs the VIO+EKF pipeline using the modular vio/ package.
It is completely independent of vio_vps.py and uses VIORunner directly.

Configuration Model (v3.9.9):
-----------------------------
    YAML config is the single source of truth for ALL settings.
    CLI provides only data paths and debug flags.
    
    YAML controls:
    - Sensor calibration (IMU, Camera, Magnetometer)
    - Algorithm toggles (use_magnetometer, estimate_imu_bias, use_vio_velocity, etc.)
    - Performance tuning (fast_mode, frame_skip)
    - Camera view selection (default_camera_view)
    
    CLI provides:
    - Required: --config, --imu, --output
    - Optional: --quarry
    - Optional: --images_dir, --images_index, --vps, --mag, --dem, --ground_truth
    - Debug flags: --save_debug_data, --save_keyframe_images

The modular package provides:
- vio.config: Configuration loading, VIOConfig dataclass
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
        --output output_dir/
    
    # With all inputs:
    python run_vio.py --config config.yaml --imu imu.csv \\
        --output out/ --mag mag.csv --vps vps.csv --dem dem.tif \\
        --save_debug_data

Author: VIO project
Version: 3.2.0 (YAML single source of truth - all settings from YAML)
"""

import argparse
import sys
import os
import numpy as np

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# STRICT NUMERICAL ERROR DETECTION (debug mode)
# Uncomment to catch inf/nan at the EXACT line where it first occurs
# WARNING: This will crash the program immediately on first inf/nan
# np.seterr(all='raise')  # Raise FloatingPointError on divide-by-zero, overflow, invalid
# 
# For production, use 'warn' to log but continue:
np.seterr(divide='warn', over='warn', invalid='warn', under='ignore')


def parse_args():
    """
    Parse command line arguments.
    
    v3.2.0 Model:
    -------------
    - YAML config is the single source of truth for ALL settings
    - CLI provides only data paths and debug flags
    - No algorithm toggles in CLI (use YAML)
    """
    parser = argparse.ArgumentParser(
        description="VIO+EKF Pipeline - Entry Point (v3.2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Model:
  YAML = single source of truth for ALL settings
  CLI  = data paths + debug flags only

Settings in YAML:
  imu.estimate_bias, imu.use_preintegration
  magnetometer.enabled
  vio.use_vio_velocity
  default_camera_view
  fast_mode.use_fast_mode, fast_mode.frame_skip

Examples:
  # Basic usage:
  python run_vio.py --config config.yaml --imu imu.csv --output out/
  
  # With debug output:
  python run_vio.py --config config.yaml --imu imu.csv \\
      --output out/ --save_debug_data --save_keyframe_images
  
  # With all inputs:
  python run_vio.py --config config.yaml --imu imu.csv \\
      --output out/ --mag mag.csv --vps vps.csv --dem dem.tif --ground_truth gt.pos
        """
    )
    
    # Required inputs
    parser.add_argument("--imu", type=str, required=True, 
                        help="Path to IMU CSV file")
    parser.add_argument("--output", type=str, required=True, 
                        help="Output directory")
    
    # Configuration (YAML is the single source of truth)
    parser.add_argument("--config", type=str, 
                        default="configs/config_bell412_dataset3.yaml",
                        help="Path to YAML config file (single source of truth)")
    
    # Optional data inputs (paths only)
    parser.add_argument("--images_dir", type=str, default=None, 
                        help="Directory containing images")
    parser.add_argument("--images_index", type=str, default=None, 
                        help="Path to images_index.csv")
    parser.add_argument("--timeref_csv", type=str, default=None,
                        help="Path to camera timeref.csv (time_ref for unified clock)")
    parser.add_argument("--timeref_pps_csv", type=str, default=None,
                        help="Path to PPS timeref.csv (time_ref <-> UTC/GNSS anchor for time audit)")
    parser.add_argument("--vps", type=str, default=None, 
                        help="Path to VPS results CSV")
    parser.add_argument("--vps_tiles", type=str, default=None,
                        help="Path to MBTiles file for VPS real-time processing")
    parser.add_argument("--mag", type=str, default=None, 
                        help="Path to magnetometer CSV")
    parser.add_argument("--dem", type=str, default=None, 
                        help="Path to DEM/DSM TIF file")
    parser.add_argument("--quarry", type=str, default=None,
                        help="Path to flight_log_from_gga.csv for msl datasets")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Path to PPK ground truth file")
    
    # Debug/output flags (simple store_true - off by default)
    parser.add_argument("--save_debug_data", action="store_true",
                        help="Save debug CSV files (ekf_*.csv, vio_*.csv)")
    parser.add_argument("--save_keyframe_images", action="store_true",
                        help="Save keyframe overlay images")
    
    return parser.parse_args()


def main():
    """Main entry point - load YAML config and run VIO."""
    args = parse_args()
    
    # Print header
    print("=" * 70)
    print("VIO+EKF Pipeline - Entry Point (v3.2.0)")
    print("Config Model: YAML = single source of truth")
    print("=" * 70)
    
    # Import from modular vio package
    try:
        from vio import __version__
        from vio.config import load_config
        from vio.main_loop import VIORunner
        
        print(f"Using vio package version: {__version__}")
        
        # =================================================================
        # Step 1: Load YAML config → returns VIOConfig dataclass
        # =================================================================
        print(f"\nLoading config: {args.config}")
        config = load_config(args.config)
        
        # =================================================================
        # Step 2: Set CLI-provided paths
        # =================================================================
        # Required paths
        config.imu_path = args.imu
        config.quarry_path = args.quarry
        config.output_dir = args.output
        
        # Optional paths (only if provided)
        if args.images_dir:
            config.images_dir = args.images_dir
        if args.images_index:
            config.images_index_csv = args.images_index
        if args.timeref_csv:
            config.timeref_csv = args.timeref_csv
        if args.timeref_pps_csv:
            config.timeref_pps_csv = args.timeref_pps_csv
        if args.vps:
            config.vps_csv = args.vps
        if args.mag:
            config.mag_csv = args.mag
        if args.dem:
            config.dem_path = args.dem
        if args.ground_truth:
            config.ground_truth_path = args.ground_truth
        
        # MBTiles path for VPS
        if args.vps_tiles:
            config.mbtiles_path = args.vps_tiles
        
        # Debug flags (CLI flags enable if specified)
        if args.save_debug_data:
            config.save_debug_data = True
        if args.save_keyframe_images:
            config.save_keyframe_images = True
        
        # =================================================================
        # Step 3: Print config summary
        # =================================================================
        print("\n" + "=" * 70)
        print("Configuration Summary:")
        print("=" * 70)
        print(f"  Config file: {args.config}")
        print(f"  IMU path: {config.imu_path}")
        print(f"  Quarry path: {config.quarry_path or 'None'}")
        print(f"  Output dir: {config.output_dir}")
        print(f"\nAlgorithm Settings (from YAML):")
        print(f"  camera_view: {config.camera_view}")
        print(f"  estimate_imu_bias: {config.estimate_imu_bias}")
        print(f"  use_magnetometer: {config.use_magnetometer}")
        print(f"  use_mag_estimated_bias: {config.use_mag_estimated_bias} (sigma_init={config.sigma_mag_bias_init}, sigma_proc={config.sigma_mag_bias})")
        print(f"  use_vio_velocity: {config.use_vio_velocity}")
        print(f"  estimator_mode: {config.estimator_mode}")
        print(f"  fast_mode: {config.fast_mode}")
        print(f"  frame_skip: {config.frame_skip}")
        print(f"\nDebug Settings:")
        print(f"  save_debug_data: {config.save_debug_data}")
        print(f"  save_keyframe_images: {config.save_keyframe_images}")
        print("=" * 70)
        
        # =================================================================
        # [TRIPWIRE] Check which files are being used
        # =================================================================
        from vio.numerical_checks import check_file_being_used
        check_file_being_used()
        
        # =================================================================
        # Step 4: Create output directory and save CLI command
        # =================================================================
        os.makedirs(args.output, exist_ok=True)
        
        # Save CLI command for reproducibility
        cli_command = " ".join(sys.argv)
        cli_log_path = os.path.join(args.output, "cli_command.txt")
        with open(cli_log_path, 'w') as f:
            f.write(f"# VIO Pipeline CLI Command\n")
            f.write(f"# Config: {args.config}\n")
            f.write(f"# Version: 3.2.0\n\n")
            f.write(cli_command + "\n")
        print(f"CLI command saved: {cli_log_path}")
        
        # =================================================================
        # Step 5: Create and run VIORunner
        # =================================================================
        runner = VIORunner(config)
        runner.run()
        
        print("=" * 70)
        print("✅ VIO pipeline completed successfully")
        print(f"   Output: {args.output}")
        print(f"   Config used: {args.config}")
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
