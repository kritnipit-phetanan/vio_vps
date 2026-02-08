"""VIO (Visual-Inertial Odometry) Package

Complete modularized implementation of the VIO+ESKF+MSCKF system
for helicopter navigation. Version 3.9.10 - Config Cleanup.

Version: 3.9.10 (Config.py Refactoring)
Modules: 18
Total Lines: ~11,400

Changes in v3.9.10:
- REFACTOR: config.py cleanup (647 → 571 lines, -12%)
  * Removed dead code: VERBOSE_DEBUG, VERBOSE_DEM (never imported)
  * Removed dead code: R_flip, APPLY_R_FLIP (always False)
  * Removed dead code: MAG_ADAPTIVE_THRESHOLD, SIGMA_VO_VEL
  * Removed duplicates: MSCKF_STATS (use msckf.py), MAG_FILTER_STATE (use magnetometer.py)
  * Simplified correct_camera_extrinsics() to pass-through for FRD body frame

Changes in v3.9.9:
- NO-MAG FREEZE: When magnetometer disabled or mag_params=None
  * P_mag = 1e-12 (frozen covariance)
  * Q_mag = 0 (no process noise)
  * Prevents unbounded covariance growth

Changes in v3.9.8:
- PROPER FREEZE: When use_estimated_bias=false
  * Sets Jacobian H[15:18] = 0
  * Sets process noise sigma_mag_bias = 0
  * Sets initial covariance P_mag = 1e-12

Changes in v3.9.7:
- EKF MAG BIAS ESTIMATION: Added magnetometer hard iron online estimation
  * State augmented from 16D to 19D nominal (15D to 18D error)
  * New state: [p, v, q, bg, ba, mag_bias, clones...]
  * Process noise for mag_bias random walk (sigma_mag_bias = 0.001)
  * Initial uncertainty configurable via sigma_mag_bias_init
  
- FILES MODIFIED (8 files, ~30 locations):
  * ekf.py: CORE_NOMINAL_DIM=19, CORE_ERROR_DIM=18, new IDX_MAG_BIAS
  * state_manager.py: 18x18 covariance, mag_bias initialization
  * imu_preintegration.py: Phi 18x18, Q 18x18 with mag noise
  * propagation.py: Clone indices updated (15→18, 16→19)
  * msckf.py: Clone indices, observability nullspace
  * measurement_updates.py: Error state dimension
  * vps_integration.py: Error state dimension  
  * loop_closure.py: Error state dimension
  * main_loop.py: EKF initialization (dim_x=19)
  * config.yaml: sigma_mag_bias, sigma_mag_bias_init, use_estimated_bias

Changes in v3.2.0:
- VIOCONFIG DATACLASS: config.py is now pure YAML reader returning VIOConfig
  * VIOConfig dataclass holds all settings from YAML
  * load_config() returns VIOConfig directly (not raw dict)
  * _raw_config Dict preserved for backward compatibility with legacy code
  
- CLI SIMPLIFIED: Removed camera_view from CLI
  * camera_view now configured in YAML only
  * CLI provides only: paths, save_debug_data, save_keyframe_images
  * run_vio.py uses VIOConfig throughout
  
- CLEANED UP: Removed resolved_config remnants
  * save_calibration_log now accepts vio_config parameter
  * main_loop.py imports VIOConfig from config.py (single definition)
  * No more duplicate VIOConfig definitions

Changes in v3.1.0:
- SIMPLIFIED CONFIG: YAML is now single source of truth for algorithm settings
  * Removed CLI overrides for: use_magnetometer, estimate_imu_bias, use_vio_velocity, 
    use_preintegration, fast_mode, frame_skip
  * CLI now provides only: paths, camera_view, save_debug_data, save_keyframe_images
  * Algorithm toggles configured in YAML config file only
  
- YAML CONFIG CHANGES:
  * Added imu.use_preintegration toggle
  * Renamed performance section to fast_mode section
  * fast_mode: { use_fast_mode: bool, frame_skip: int }
  
- REMOVED: config_resolver.py module (merged functionality into config.py)
- REMOVED: Tri-state boolean flags (BooleanOptionalAction)
- SIMPLIFIED: run_vio.py entry point

Changes in v3.0.0:
- PRODUCTION CONFIG: New config_resolver.py module for clean config management
  * Clear precedence: CODE DEFAULTS → YAML base config → CLI overrides
  * Tri-state booleans: --flag / --no-flag / (not specified = use YAML)
  * ResolvedConfig: Single source of truth for runtime configuration
  * Reproducibility: resolved_config.yaml saved with every run
  * Full audit: debug_calibration.txt includes CLI command and source tracking

- CLI IMPROVEMENTS: argparse.BooleanOptionalAction for proper override behavior
  * --use_magnetometer: Explicitly enable (override YAML)
  * --no-use_magnetometer: Explicitly disable (override YAML)
  * (not specified): Use YAML value (or code default)
  
- DEPRECATED: Global constants in config.py for runtime decisions
  * Old: Read KB_PARAMS, IMU_PARAMS directly from config.py globals
  * New: All runtime values flow through ResolvedConfig/VIOConfig

Changes in v2.9.10.0:
- PRIORITY 1: PPK Initial Heading Calibration (HIGHEST IMPACT)
  * Use PPK trajectory (first 30s) to extract accurate initial heading
  * Complies with GPS-denied constraints: GT only as initializer, not continuous
  * Expected: 863m → 200-300m (65% improvement!)
  * Eliminates 739m North bias from 5-10° heading error
  
- PRIORITY 2: Adaptive MSCKF Reprojection Threshold (CRITICAL)
  * Start permissive (20px) during initialization
  * Tighten to 10px as filter converges (based on velocity covariance)
  * Expected: MSCKF rate 0.5 Hz → 3-4 Hz
  * More landmark updates = better scale constraint
  
- PRIORITY 3: Multi-Baseline Triangulation (REFINEMENT)
  * Use 3-5 best frame pairs instead of just 2
  * Select maximum baseline for better geometry
  * Reduces depth errors and improves triangulation quality
- Config: performance.fast_mode, performance.frame_skip
- Flags: --fast_mode, --frame_skip N
- Trade-off: Minimal accuracy loss (<5%) for 2-2.5x speedup

Changes in v2.9.8.8:
- CRITICAL FIX: VIO velocity updates now enabled via --use_vio_velocity flag
  * Bug: Flag missing in benchmark_modular.sh → 0 velocity updates
  * Fix: Added flag → expect ~18,000 updates (1 per frame)
  * Impact: XY drift 10 m/min → 2-3 m/min (5x improvement)
- OPTIMIZATION: Disabled loop closure for outdoor long-range flights
  * No loops → 319 keyframes wasted 0.3-0.5s each
  * CPU savings: ~3% inference time improvement
  * Alternative: Magnetometer provides continuous yaw correction
- OPTIMIZATION: Disabled ZUPT for helicopters
  * Continuous rotor vibration prevents zero-velocity state
  * 0 detections in 5-minute flight (feature not applicable)
  * Alternative: Vibration detection for adaptive R scaling
- Config changes: loop_closure.use_loop_closure=false, zupt.enabled=false
- Expected improvements: Position error 200m → 50-100m (2-4x better)

Changes in v2.9.8.7:
- FIX: Loop closure variable name collision bug
  * Line 553: kf = keyframes[idx] overwrote EKF parameter
  * Caused AttributeError: 'dict' object has no attribute 'x'
  * Fix: Renamed to stored_kf to preserve EKF object
  * Result: ~30 loop closure errors eliminated

Changes in v2.9.8.6:
- FIX: Config migration sigma_vo_vel → sigma_vo
  * Consolidated duplicate parameter (process_noise + vio sections)
  * Removed sigma_vo_vel, use vio.sigma_vo only
  * Updated config.py, measurement_updates.py, main_loop.py

Changes in v2.9.8.5:
- CLEANUP: Removed global vio.use_vz_only override
  * Now per-view controlled via views.nadir.use_vz_only
  * Nadir: use_vz_only=false (enable XY velocity)
  * Front: use_vz_only=false (enable full 3D velocity)

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

__version__ = "3.9.10"

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
