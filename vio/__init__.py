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
    
    # Or import specific functions
    from vio.config import load_config
    from vio.math_utils import quat_multiply, quat_to_rot
    from vio.magnetometer import calibrate_magnetometer, compute_yaw_from_mag
    from vio.camera import kannala_brandt_unproject
    from vio.vio_frontend import VIOFrontEnd
    from vio.msckf import perform_msckf_updates
"""

__version__ = "2.1.0"

# Lazy module imports - access as vio.config, vio.math_utils, etc.
# This avoids importing all dependencies at once

def __getattr__(name):
    """Lazy module loading to avoid importing all dependencies at once."""
    if name == "config":
        from . import config
        return config
    elif name == "math_utils":
        from . import math_utils
        return math_utils
    elif name == "imu_preintegration":
        from . import imu_preintegration
        return imu_preintegration
    elif name == "ekf":
        from . import ekf
        return ekf
    elif name == "data_loaders":
        from . import data_loaders
        return data_loaders
    elif name == "magnetometer":
        from . import magnetometer
        return magnetometer
    elif name == "camera":
        from . import camera
        return camera
    elif name == "vio_frontend":
        from . import vio_frontend
        return vio_frontend
    elif name == "msckf":
        from . import msckf
        return msckf
    
    raise AttributeError(f"module 'vio' has no attribute '{name}'")


# For explicit imports: from vio import load_config, etc.
def __dir__():
    """List available submodules."""
    return [
        "config", "math_utils", "imu_preintegration", "ekf", 
        "data_loaders", "magnetometer", "camera", "vio_frontend", "msckf"
    ]


# Direct function imports for convenience (loaded on-demand)
__all__ = [
    # Submodules
    "config", "math_utils", "imu_preintegration", "ekf", 
    "data_loaders", "magnetometer", "camera", "vio_frontend", "msckf",
]
