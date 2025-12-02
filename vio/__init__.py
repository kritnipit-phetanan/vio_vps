"""
VIO (Visual-Inertial Odometry) Package

Complete modularized implementation of the VIO+ESKF+MSCKF system
for helicopter navigation. Version 2.6.0 - Standalone (no vio_vps.py dependency).

Version: 2.6.0 (Standalone modular package)
Modules: 17
Total Lines: ~8,900

Changes in v2.6.0:
- BREAKING: run_vio.py now uses VIORunner directly (no vio_vps.py dependency)
- Refactored redundant code: math_utils is now single source of truth for:
  * quaternion_to_yaw, quaternion_multiply, angle_wrap, yaw_to_quaternion_update
- magnetometer.py and measurement_updates.py now import from math_utils
- Reduced code duplication by ~100 lines
- Package is now fully self-contained and independent of vio_vps.py

Changes in v2.5.1:
- Added comprehensive debug output setup in main_loop.py
- Added detailed docstrings to config.py, math_utils.py, imu_preintegration.py

Changes in v2.5.0:
- Added complete VIO processing in main_loop.py (process_vio method)
- Added preintegration application at camera frames
- Added camera cloning for MSCKF
- Added VIO velocity update with scale recovery
- Added error logging vs ground truth
- Full integration of run() logic into VIORunner class

Submodules:
- config: Configuration loading and global constants
- math_utils: Quaternion operations, rotation matrices
- imu_preintegration: IMU preintegration (Forster et al.)
- ekf: Extended Kalman Filter core
- data_loaders: IMU, MAG, VPS, PPK, DEM data loaders
- magnetometer: Magnetometer calibration and yaw computation
- camera: Kannala-Brandt fisheye camera model helpers
- vio_frontend: OpenVINS-style visual front-end for feature tracking
- msckf: Multi-State Constraint Kalman Filter backend
- propagation: IMU propagation, ZUPT, flight phase detection
- vps_integration: VPS position updates, DEM height updates
- output_utils: Debug logging, visualization, error statistics
- state_manager: EKF state initialization, lever arm compensation
- measurement_updates: MAG, DEM, ZUPT measurement updates
- main_loop: VIORunner class for complete pipeline
- loop_closure: Loop closure detection for yaw drift correction

Author: VIO project

Usage:
    # Import specific modules (lazy loading)
    from vio import config
    from vio import math_utils
    from vio import imu_preintegration
    from vio import ekf
    from vio import data_loaders
    from vio import magnetometer
    from vio import camera
    from vio import vio_frontend
    from vio import msckf
    from vio import propagation
    from vio import vps_integration
    from vio import output_utils
    from vio import state_manager
    from vio import measurement_updates
    from vio import main_loop
    from vio import loop_closure
    
    # Or import specific functions
    from vio.config import load_config
    from vio.math_utils import quat_multiply, quat_to_rot
    from vio.magnetometer import calibrate_magnetometer, compute_yaw_from_mag
    from vio.camera import kannala_brandt_unproject
    from vio.vio_frontend import VIOFrontEnd
    from vio.msckf import perform_msckf_updates
    from vio.propagation import propagate_to_timestamp, apply_zupt
    from vio.vps_integration import apply_vps_update, apply_height_update
    from vio.output_utils import DebugCSVWriters, print_error_statistics
    from vio.state_manager import initialize_ekf_state
    from vio.main_loop import VIORunner, VIOConfig
    from vio.loop_closure import LoopClosureDetector, init_loop_closure
"""

__version__ = "2.6.0"

# Lazy module imports - access as vio.config, vio.math_utils, etc.
# This avoids importing all dependencies at once
import importlib

# Available submodules
_SUBMODULES = {
    "config", "math_utils", "imu_preintegration", "ekf", 
    "data_loaders", "magnetometer", "camera", "vio_frontend", "msckf",
    "propagation", "vps_integration", "output_utils",
    "state_manager", "measurement_updates", "main_loop", "loop_closure"
}

def __getattr__(name):
    """Lazy module loading to avoid importing all dependencies at once."""
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # Cache in globals to avoid repeated import
        return module
    raise AttributeError(f"module 'vio' has no attribute '{name}'")


# For explicit imports: from vio import load_config, etc.
def __dir__():
    """List available submodules."""
    return list(_SUBMODULES)


# Direct function imports for convenience (loaded on-demand)
__all__ = list(_SUBMODULES)
