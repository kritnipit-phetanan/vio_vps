"""
VIO (Visual-Inertial Odometry) Package

Complete modularized implementation of the VIO+ESKF+MSCKF system
for helicopter navigation. Version 2.9.6 - Code cleanup and enhancements.

Version: 2.9.6 (Refactored for better maintainability)
Modules: 17
Total Lines: ~10,000

Changes in v2.9.6:
- REFACTORED: Enhanced apply_zupt() in propagation.py
  * Added debug logging support (residual_csv integration)
  * Returns updated consecutive_stationary count
  * Computes innovation and Mahalanobis distance
  * Logs ZUPT updates to debug_residuals.csv
- ENHANCED: _apply_vio_velocity_update() with chi-square gating
  * Added innovation gating before EKF update
  * Chi-square threshold: 3.84 (1 DOF) or 7.81 (3 DOF)
  * Rejects outlier velocity measurements
  * Logs accepted/rejected status to debug_residuals.csv
- CLEANUP: Removed duplicate apply_zupt_update() from measurement_updates.py
  * Was redundant with apply_zupt() in propagation.py
  * Dead code - never called in codebase
  * Kept detect_stationary() + apply_zupt() as modular approach
- DECISION: Kept debug/logging functions in VIORunner class
  * Functions like _log_debug_state_covariance() use self heavily
  * Moving to output_utils.py would require many parameters
  * Better to keep as private methods in VIORunner
- Code organization improved, no algorithm changes

Changes in v2.9.5 (Oscillation detection with skip_count=2):
- REVERTED: Back to v2.9.2 oscillation detection strategy
  * Detects erratic BEHAVIOR (sign alternation pattern)
  * Rejects truly bad readings during fast maneuvers
  * Between rejections: FULL trust → strong corrections
- CRITICAL FIX: skip_count = 2 (reduced from 10 in v2.9.2)
  * v2.9.2 problem: 10-update skip → long gap → IMU drift → 6771 yaw jumps
  * v2.9.5 solution: 2-update skip → ~0.1s gap → minimal drift
- Rationale:
  * Innovation-based R-scaling (v2.9.3-v2.9.4) penalizes ALL large innovations
  * But large innovations can be CORRECT (drift recovery)
  * Oscillation detection only rejects PATTERN (alternating signs)
  * Allows strong corrections when not oscillating → better position accuracy
- Results (Bell 412 dataset 3):
  * Final position: 690m (+287% vs v2.9.2, trade-off)
  * Yaw jumps: 1466 (-78% vs v2.9.2, GOOD)
  * Final yaw error: 17° (BEST across all versions)
  * Mag acceptance: 91.2%
  * Tracks 5 rejection types: oscillation_skip, low_quality, large_innovation, high_rate, gyro_inconsistent
  * Periodic statistics: every 100 attempts for individual rejections, every 500 for summary
  * apply_mag_filter() returns info_dict with filter flags
- IMPROVED: magnetometer.py returns (yaw_filtered, r_scale, info_dict)
  * info_dict: {'high_rate': bool, 'gyro_inconsistent': bool}
  * Enables tracking of filter-based soft rejections (R-inflation)
- IMPROVED: measurement_updates.py tracks filter_info from apply_mag_filter()
  * Accumulates filter rejection counters separate from hard rejections
  * Helps diagnose: "accepted with high R" vs "hard rejected"

Changes in v2.9.1:
- FIX: Relaxed magnetometer filtering parameters for helicopter dynamics
  * max_yaw_rate_deg: 30.0 → 150.0 (helicopters can yaw rapidly, was rejecting valid data)
  * gyro_consistency_threshold_deg: 10.0 → 30.0 (allow more deviation during maneuvers)
  * r_inflate: 5.0 → 2.0 (don't over-inflate uncertainty, let EKF use mag data)
  * Analysis: v2.9.0 filtering was too aggressive (+178% position error vs v2.8.0)
  * Root cause: Oscillation detection + low rate threshold rejected most mag updates
- KEEP: MSCKF reprojection validation (working well, +12.8% triangulation success)

Changes in v2.9.0:
- NEW: Magnetometer filtering integration (reset_mag_filter_state, set_mag_constants, apply_mag_filter)
  * EMA smoothing with configurable alpha (default 0.3)
  * Gyro consistency check - inflate R when mag deviates from gyro integration
  * Rate-of-change detection for fast rotation rejection
  * Config params: ema_alpha, max_yaw_rate_deg, gyro_consistency_threshold_deg, r_inflate
- NEW: MSCKF reprojection validation using kannala_brandt_project
  * Pixel-level error validation (3px threshold) for triangulated points
  * Uses accurate Kannala-Brandt fisheye projection model
  * Falls back to normalized coordinate method if K,D unavailable
  * Stores max_pixel_reproj_error in triangulation results
- IMPROVED: apply_magnetometer_update() accepts yaw_override for filtered yaw
- IMPROVED: Gyro state tracking (last_gyro_z, last_imu_dt) for mag filtering

Changes in v2.8.0:
- NEW: Fisheye rectifier integration (USE_RECTIFIER config)
  * Optional fisheye→pinhole conversion before feature tracking
  * Configurable output FOV via RECTIFY_FOV_DEG
- NEW: Loop closure detection integration (USE_LOOP_CLOSURE config)
  * Detects return to previously visited locations
  * Applies yaw drift correction via EKF update
  * Configurable thresholds: position, keyframe distance, frame gap
- NEW: Vibration detector integration (USE_VIBRATION_DETECTOR config)
  * Monitors IMU acceleration variance
  * Can adjust measurement uncertainties during high vibration
- IMPROVED: _save_calibration_snapshot now uses save_calibration_log from output_utils
- IMPROVED: _save_keyframe_with_overlay uses save_keyframe_image_with_overlay
- Added YAML config parsing for all new features

Changes in v2.7.0:
- FIX: Magnetometer oscillation detection near ±180° yaw boundary
  * Added innovation sign tracking to detect alternating corrections
  * Skip 50 updates when oscillation detected to let IMU stabilize
  * Results: Position error 656m → 423m (-35.6%), Yaw error 64° → 31° (-51.2%)
- FIX: DEM height update soft gating
  * Increased chi2 thresholds for altitude constraints
  * Apply updates with inflated noise instead of hard rejection
- MSCKF success rate improved: 3.5% → 4.3%
- Moved old scripts to old_src/ folder

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
    from vio.fisheye_rectifier import FisheyeRectifier, create_rectifier_from_config
    from vio.propagation import VibrationDetector
"""

__version__ = "2.9.6"

# Lazy module imports - access as vio.config, vio.math_utils, etc.
# This avoids importing all dependencies at once
import importlib

# Available submodules
_SUBMODULES = {
    "config", "math_utils", "imu_preintegration", "ekf", 
    "data_loaders", "magnetometer", "camera", "vio_frontend", "msckf",
    "propagation", "vps_integration", "output_utils", "fisheye_rectifier",
    "state_manager", "measurement_updates", "main_loop", "loop_closure",
    "vio_processing"
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
