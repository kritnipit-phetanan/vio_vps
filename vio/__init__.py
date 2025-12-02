"""
VIO (Visual-Inertial Odometry) Package

This package contains modularized components of the VIO+EKF system:
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

__version__ = "2.4.0"

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
