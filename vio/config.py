#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIO Configuration Module
========================

Handles YAML configuration loading and defines VIOConfig dataclass for the
VIO+ESKF+MSCKF system.

Configuration Model (v3.2.0):
-----------------------------
YAML is the single source of truth for ALL settings:
- Sensor calibration (IMU, Camera, Magnetometer)
- Algorithm toggles (use_magnetometer, estimate_imu_bias, use_vio_velocity, etc.)
- Performance tuning (fast_mode, frame_skip)
- Camera view selection (default_camera_view)

CLI provides ONLY:
- Data paths (--imu, --quarry, --images_dir, etc.)
- Output directory (--output)
- Debug flags (--save_debug_data, --save_keyframe_images)

load_config() returns a VIOConfig dataclass ready for VIORunner.

Frame Conventions:
------------------
- Body Frame: FRD (Forward-Right-Down) for Bell 412
- World Frame: ENU (East-North-Up) - local tangent plane
- Camera Frame: OpenCV convention (X-right, Y-down, Z-forward)
- Quaternion: [w, x, y, z] Hamilton convention

Author: VIO project
"""

import os
import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

SUPPORTED_ESTIMATOR_MODES = {
    "imu_step_preint_cache",
    "event_queue_output_predictor",
}


def _load_yaml_file(config_path: str) -> Dict[str, Any]:
    """Read YAML file and return dict."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping (dict), got: {type(cfg)}")
    return cfg


def _validate_yaml_config(cfg: Dict[str, Any], config_path: str) -> None:
    """Validate required sections and critical enum values before compilation."""
    required_top = ["camera", "extrinsics", "imu", "magnetometer", "process_noise", "vio"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required sections: {missing} ({config_path})")
    
    est_mode = cfg.get("imu", {}).get("estimator_mode", "imu_step_preint_cache")
    if est_mode not in SUPPORTED_ESTIMATOR_MODES:
        raise ValueError(
            f"Invalid imu.estimator_mode='{est_mode}' in {config_path}. "
            f"Supported: {sorted(SUPPORTED_ESTIMATOR_MODES)}"
        )


# =============================================================================
# VIOConfig Dataclass - Single source of truth for VIO runtime settings
# =============================================================================

@dataclass
class VIOConfig:
    """
    Configuration for VIO runner.
    
    This dataclass holds ALL runtime settings for the VIO pipeline.
    It is populated by load_config() from YAML and can have paths
    overridden by CLI arguments.
    
    Usage:
        config = load_config("config.yaml")  # Returns VIOConfig
        config.imu_path = "/path/to/imu.csv"  # Override from CLI
        runner = VIORunner(config)
        runner.run()
    """
    # Required paths (set by CLI)
    imu_path: str = ""
    output_dir: str = ""
    
    # Optional data paths (set by CLI)
    quarry_path: Optional[str] = None
    images_dir: Optional[str] = None
    images_index_csv: Optional[str] = None
    timeref_csv: Optional[str] = None
    vps_csv: Optional[str] = None
    mbtiles_path: Optional[str] = None  # VPS MBTiles file (like dem_path)
    mag_csv: Optional[str] = None
    dem_path: Optional[str] = None
    ground_truth_path: Optional[str] = None
    config_yaml: Optional[str] = None
    
    # Image processing (from YAML camera section)
    downscale_size: Tuple[int, int] = (1440, 1080)
    
    camera_view: str = "nadir"  # "nadir", "front", "side"
    
    # Algorithm options (from YAML)
    estimate_imu_bias: bool = False
    use_magnetometer: bool = True
    use_mag_estimated_bias: bool = True  # v3.9.7: Online hard-iron estimation
    sigma_mag_bias_init: float = 0.1     # v3.9.7: Initial uncertainty
    sigma_mag_bias: float = 0.001        # v3.9.7: Process noise
    use_vio_velocity: bool = True
    estimator_mode: str = "imu_step_preint_cache"  # "imu_step_preint_cache" or "event_queue_output_predictor"
    
    # Performance options (from YAML fast_mode section)
    fast_mode: bool = False
    frame_skip: int = 1
    
    # Debug options (set by CLI flags)
    save_debug_data: bool = False
    save_keyframe_images: bool = False
    
    # Internal: store the raw YAML config dict for advanced access
    _raw_config: Dict[str, Any] = field(default_factory=dict)  # Flat dict
    _yaml_config: Dict[str, Any] = field(default_factory=dict)  # Original YAML


def load_config(config_path: str) -> VIOConfig:
    """
    Load YAML configuration file and return VIOConfig dataclass.
    
    This function reads the YAML config and creates a VIOConfig instance
    with all algorithm settings populated. CLI should only override
    paths (imu_path, quarry_path, etc.) and debug flags.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        VIOConfig dataclass with all settings populated from YAML
        
    Also populates module-level globals for backward compatibility:
        - KB_PARAMS: Kannala-Brandt camera intrinsics
        - BODY_T_CAMDOWN: 4x4 transform from body to down camera
        - IMU_PARAMS: IMU noise parameters
        - etc.
        
    Example:
        >>> config = load_config("configs/config_bell412_dataset3.yaml")
        >>> print(f"Use magnetometer: {config.use_magnetometer}")
        >>> print(f"Camera view: {config.camera_view}")
    """
    config = _load_yaml_file(config_path)
    _validate_yaml_config(config, config_path)
    
    # Convert nested dictionary to flat structure for compatibility
    result = {}
    
    # ========================================
    # Camera Intrinsics (Kannala-Brandt Model)
    # ========================================
    # The Kannala-Brandt model handles fisheye distortion:
    # r(θ) = k1*θ + k2*θ³ + k3*θ⁵ + k4*θ⁷ + k5*θ⁹
    # where θ is the angle from optical axis
    cam = config['camera']
    result['KB_PARAMS'] = {
        'k2': cam['k2'],  # Note: k1 is implicit (=1 for equidistant)
        'k3': cam['k3'],
        'k4': cam['k4'],
        'k5': cam['k5'],
        'mu': cam['mu'],  # Focal length in x (pixels)
        'mv': cam['mv'],  # Focal length in y (pixels)
        'u0': cam['u0'],  # Principal point x (pixels)
        'v0': cam['v0'],  # Principal point y (pixels)
        'w': cam['image_width'],
        'h': cam['image_height'],
    }
    
    # ========================================
    # Plane-Aided MSCKF Configuration
    # ========================================
    plane_config = config.get('plane', {})
    result['USE_PLANE_MSCKF'] = plane_config.get('enabled', False)
    result['PLANE_MIN_POINTS'] = plane_config.get('min_points_per_plane', 10)
    result['PLANE_ANGLE_THRESHOLD'] = np.radians(plane_config.get('angle_threshold_deg', 15.0))
    result['PLANE_DISTANCE_THRESHOLD'] = plane_config.get('distance_threshold_m', 0.15)
    result['PLANE_MIN_AREA'] = plane_config.get('min_area_m2', 0.5)
    result['PLANE_SIGMA'] = plane_config.get('measurement_noise_m', 0.05)
    result['PLANE_USE_AIDED_TRIANGULATION'] = plane_config.get('use_aided_triangulation', True)
    result['PLANE_USE_CONSTRAINTS'] = plane_config.get('use_constraints', True)
    result['PLANE_MAX_PLANES'] = plane_config.get('max_planes', 10)
    
    # ========================================
    # Extrinsics (Camera-to-Body Transforms)
    # ========================================
    extr = config['extrinsics']
    
    # ========================================================================
    # NOTE: R_flip logic removed - Bell 412 uses FRD body frame (no flip needed)
    # For FLU body frame datasets, this would need to be re-implemented
    # ========================================================================
    
    def correct_camera_extrinsics(T_bc):
        """Pass-through for FRD body frame (no correction needed)."""
        return T_bc.copy()
    

    T_orig = np.array(extr['nadir']['transform'], dtype=np.float64)
    result['BODY_T_CAMDOWN'] = correct_camera_extrinsics(T_orig)
    
    result['BODY_T_CAMFRONT'] = correct_camera_extrinsics(
        np.array(extr['front']['transform'], dtype=np.float64))
    result['BODY_T_CAMSIDE'] = correct_camera_extrinsics(
        np.array(extr['side']['transform'], dtype=np.float64))
    
    # IMU parameters
    imu = config['imu']
    result['IMU_PARAMS'] = {
        'acc_n': imu['acc_n'],
        'gyr_n': imu['gyr_n'],
        'acc_w': imu['acc_w'],
        'gyr_w': imu['gyr_w'],
        'g_norm': imu['g_norm'],
        'accel_includes_gravity': imu.get('accel_includes_gravity', True),
    }
    result['IMU_PARAMS_PREINT'] = {
        'acc_n': imu['preintegration']['acc_n'],
        'gyr_n': imu['preintegration']['gyr_n'],
        'acc_w': imu['preintegration']['acc_w'],
        'gyr_w': imu['preintegration']['gyr_w'],
        'g_norm': imu['g_norm'],
        'accel_includes_gravity': imu.get('accel_includes_gravity', True),
    }
    
    # IMU bias settings
    result['ESTIMATE_IMU_BIAS'] = imu.get('estimate_bias', False)
    initial_gyro_bias = imu.get('initial_gyro_bias', [0.0, 0.0, 0.0])
    result['INITIAL_GYRO_BIAS'] = np.array(initial_gyro_bias, dtype=float)
    initial_accel_bias = imu.get('initial_accel_bias', [0.0, 0.0, 0.0])
    result['INITIAL_ACCEL_BIAS'] = np.array(initial_accel_bias, dtype=float)
    
    # Estimator mode (v3.7.0: enum instead of boolean)
    # "imu_step_preint_cache": IMU-driven with preintegration cache (was use_preintegration: false)
    # "event_queue_output_predictor": Event-driven with propagate-to-timestamp (was use_preintegration: true)
    estimator_mode_raw = imu.get('estimator_mode', None)
    result['ESTIMATOR_MODE'] = estimator_mode_raw
    
    # Flight phase detection thresholds (v3.4.0: State-based)
    phase_detection = imu.get('phase_detection', {})
    result['PHASE_SPINUP_VELOCITY_THRESH'] = phase_detection.get('spinup_velocity_thresh', 1.0)
    result['PHASE_SPINUP_VIBRATION_THRESH'] = phase_detection.get('spinup_vibration_thresh', 0.3)
    result['PHASE_SPINUP_ALT_CHANGE_THRESH'] = phase_detection.get('spinup_altitude_change_thresh', 5.0)
    result['PHASE_EARLY_VELOCITY_SIGMA_THRESH'] = phase_detection.get('early_velocity_sigma_thresh', 3.0)
    
    # Magnetometer calibration
    mag = config['magnetometer']
    # v2.9.10.4: Add enabled flag to allow disabling mag from config
    result['MAG_ENABLED'] = mag.get('enabled', True)
    result['MAG_HARD_IRON_OFFSET'] = np.array(mag['hard_iron_offset'], dtype=float)
    result['MAG_SOFT_IRON_MATRIX'] = np.array(mag['soft_iron_matrix'], dtype=float)
    result['MAG_DECLINATION'] = mag['declination']
    result['MAG_FIELD_STRENGTH'] = mag['expected_field_strength']
    result['MAG_MIN_FIELD_STRENGTH'] = mag['min_field_strength']
    result['MAG_MAX_FIELD_STRENGTH'] = mag['max_field_strength']
    result['MAG_UPDATE_RATE_LIMIT'] = mag['update_rate_limit']
    result['MAG_USE_RAW_HEADING'] = mag.get('use_raw_heading', True)
    result['MAG_APPLY_INITIAL_CORRECTION'] = mag.get('apply_initial_correction', True)
    result['MAG_INITIAL_CONVERGENCE_WINDOW'] = mag.get('convergence_window', 30.0)
    
    # v3.9.7: Online Mag Bias Estimation
    result['MAG_USE_ESTIMATED_BIAS'] = mag.get('use_estimated_bias', True)
    result['SIGMA_MAG_BIAS_INIT'] = mag.get('sigma_mag_bias_init', 0.1)
    result['SIGMA_MAG_BIAS'] = mag.get('sigma_mag_bias', 0.001)
    
    # Enhanced Magnetometer Filtering Parameters (v2.9.0+)
    result['MAG_EMA_ALPHA'] = mag.get('ema_alpha', 0.3)
    result['MAG_MAX_YAW_RATE_DEG'] = mag.get('max_yaw_rate_deg', 30.0)
    result['MAG_GYRO_THRESHOLD_DEG'] = mag.get('gyro_consistency_threshold_deg', 10.0)
    result['MAG_R_INFLATE'] = mag.get('r_inflate', 5.0)
    
    # Process noise
    pn = config['process_noise']
    result['SIGMA_ACCEL'] = pn['sigma_accel']
    result['SIGMA_VPS_XY'] = pn['sigma_vps_xy']
    result['SIGMA_AGL_Z'] = pn['sigma_agl_z']
    result['SIGMA_MAG_YAW'] = pn['sigma_mag_yaw']
    result['SIGMA_UNMODELED_GYR'] = pn.get('sigma_unmodeled_gyr', 0.002)
    result['MIN_YAW_PROCESS_NOISE_DEG'] = pn.get('min_yaw_process_noise_deg', 3.0)
    
    # Inject process-noise overrides into IMU param dicts for propagation modules
    result['IMU_PARAMS']['sigma_accel'] = result['SIGMA_ACCEL']
    result['IMU_PARAMS']['sigma_unmodeled_gyr'] = result['SIGMA_UNMODELED_GYR']
    result['IMU_PARAMS']['min_yaw_process_noise_deg'] = result['MIN_YAW_PROCESS_NOISE_DEG']
    result['IMU_PARAMS_PREINT']['sigma_accel'] = result['SIGMA_ACCEL']
    result['IMU_PARAMS_PREINT']['sigma_unmodeled_gyr'] = result['SIGMA_UNMODELED_GYR']
    result['IMU_PARAMS_PREINT']['min_yaw_process_noise_deg'] = result['MIN_YAW_PROCESS_NOISE_DEG']
    
    # Compiler metadata (for debug/traceability)
    result['CONFIG_COMPILE_META'] = {
        'source': os.path.abspath(config_path),
        'estimator_mode': result['ESTIMATOR_MODE'],
        'accel_includes_gravity': bool(result['IMU_PARAMS']['accel_includes_gravity']),
    }
    
    # VIO parameters
    vio = config['vio']
    result['SIGMA_VO'] = vio['sigma_vo']  # Velocity measurement uncertainty
    result['MIN_PARALLAX_PX'] = vio['min_parallax_px']
    result['MIN_MSCKF_BASELINE'] = vio['min_msckf_baseline']
    result['MSCKF_CHI2_MULTIPLIER'] = vio['msckf_chi2_multiplier']
    result['MSCKF_MAX_REPROJECTION_ERROR'] = vio['msckf_max_reprojection_error']
    result['VO_MIN_INLIERS'] = vio['min_inliers']
    result['VO_RATIO_TEST'] = vio['ratio_test']
    
    # MSCKF sliding window parameters (v3.9.1)
    msckf_cfg = vio.get('msckf', {})
    result['MSCKF_MAX_CLONE_SIZE'] = msckf_cfg.get('max_clone_size', 11)
    result['MSCKF_MIN_TRACK_LENGTH'] = msckf_cfg.get('min_track_length', 4)
    result['VO_NADIR_ALIGN_DEG'] = vio['views']['nadir']['nadir_threshold_deg']
    result['VO_FRONT_ALIGN_DEG'] = vio['views']['front']['nadir_threshold_deg']
    
    # VIO velocity toggle (v3.1.0)
    result['USE_VIO_VELOCITY'] = vio.get('use_vio_velocity', True)
    
    # v2.9.10.5: Store raw vio config dict for access to all parameters
    # This includes new parameters like initial_agl_override
    result['vio'] = vio
    
    # Camera view configs - convert to expected format
    result['CAMERA_VIEW_CONFIGS'] = {
        'nadir': {
            'extrinsics': 'BODY_T_CAMDOWN',
            'nadir_threshold': vio['views']['nadir']['nadir_threshold_deg'],
            'sigma_scale_xy': vio['views']['nadir']['sigma_scale_xy'],
            'sigma_scale_z': vio['views']['nadir']['sigma_scale_z'],
            'use_vz_only': vio['views']['nadir']['use_vz_only'],
            'min_parallax': vio['views']['nadir']['min_parallax'],
            'max_corners': vio['views']['nadir']['max_corners'],
        },
        'front': {
            'extrinsics': 'BODY_T_CAMFRONT',
            'nadir_threshold': vio['views']['front']['nadir_threshold_deg'],
            'sigma_scale_xy': vio['views']['front']['sigma_scale_xy'],
            'sigma_scale_z': vio['views']['front']['sigma_scale_z'],
            'use_vz_only': vio['views']['front']['use_vz_only'],
            'min_parallax': vio['views']['front']['min_parallax'],
            'max_corners': vio['views']['front']['max_corners'],
        },
        'side': {
            'extrinsics': 'BODY_T_CAMSIDE',
            'nadir_threshold': vio['views']['side']['nadir_threshold_deg'],
            'sigma_scale_xy': vio['views']['side']['sigma_scale_xy'],
            'sigma_scale_z': vio['views']['side']['sigma_scale_z'],
            'use_vz_only': vio['views']['side']['use_vz_only'],
            'min_parallax': vio['views']['side']['min_parallax'],
            'max_corners': vio['views']['side']['max_corners'],
        },
    }
    
    # Undistort backend
    result['USE_FISHEYE'] = config.get('use_fisheye', True)
    
    # Default camera view (v3.1.0)
    result['DEFAULT_CAMERA_VIEW'] = config.get('default_camera_view', 'nadir')
    
    # =========================================================================
    # Fast Mode / Performance (v3.1.0)
    # =========================================================================
    if 'fast_mode' in config:
        fm = config['fast_mode']
        result['FAST_MODE'] = fm.get('use_fast_mode', False)
        result['FRAME_SKIP'] = fm.get('frame_skip', 1)
    else:
        # Legacy support: check 'performance' section
        perf = config.get('performance', {})
        result['FAST_MODE'] = perf.get('fast_mode', False)
        result['FRAME_SKIP'] = perf.get('frame_skip', 1)
    
    # IMU-GNSS Lever Arm (optional - defaults to zero if not specified)
    if 'imu_gnss_extrinsics' in config:
        imu_gnss = config['imu_gnss_extrinsics']
        result['BODY_T_GNSS'] = np.array(imu_gnss['transform'], dtype=np.float64)
        result['IMU_GNSS_LEVER_ARM'] = result['BODY_T_GNSS'][:3, 3]
    else:
        # Default: no lever arm (GNSS at same position as IMU)
        result['BODY_T_GNSS'] = np.eye(4, dtype=np.float64)
        result['IMU_GNSS_LEVER_ARM'] = np.zeros(3, dtype=np.float64)
    
    # =========================================================================
    # NEW: Fisheye Rectification (v2.8.0)
    # =========================================================================
    if 'rectification' in config:
        rect = config['rectification']
        result['USE_RECTIFIER'] = rect.get('use_rectifier', False)
        result['RECTIFY_FOV_DEG'] = rect.get('rectify_fov_deg', 90.0)
    else:
        result['USE_RECTIFIER'] = False
        result['RECTIFY_FOV_DEG'] = 90.0
    
    # =========================================================================
    # NEW: Loop Closure Detection (v2.8.0)
    # =========================================================================
    if 'loop_closure' in config:
        lc = config['loop_closure']
        result['USE_LOOP_CLOSURE'] = lc.get('use_loop_closure', True)
        result['LOOP_POSITION_THRESHOLD'] = lc.get('position_threshold', 30.0)
        result['LOOP_MIN_KEYFRAME_DIST'] = lc.get('min_keyframe_dist', 15.0)
        result['LOOP_MIN_KEYFRAME_YAW'] = lc.get('min_keyframe_yaw', 20.0)
        result['LOOP_MIN_FRAME_GAP'] = lc.get('min_frame_gap', 50)
        result['LOOP_MIN_INLIERS'] = lc.get('min_inliers', 15)
    else:
        result['USE_LOOP_CLOSURE'] = True
        result['LOOP_POSITION_THRESHOLD'] = 30.0
        result['LOOP_MIN_KEYFRAME_DIST'] = 15.0
        result['LOOP_MIN_KEYFRAME_YAW'] = 20.0
        result['LOOP_MIN_FRAME_GAP'] = 50
        result['LOOP_MIN_INLIERS'] = 15
    
    # =========================================================================
    # NEW: Vibration Detection (v2.8.0)
    # =========================================================================
    if 'vibration' in config:
        vib = config['vibration']
        result['USE_VIBRATION_DETECTOR'] = vib.get('use_vibration_detector', True)
        result['VIBRATION_WINDOW_SIZE'] = vib.get('window_size', 50)
        result['VIBRATION_THRESHOLD_MULT'] = vib.get('threshold_multiplier', 5.0)
    else:
        result['USE_VIBRATION_DETECTOR'] = True
        result['VIBRATION_WINDOW_SIZE'] = 50
        result['VIBRATION_THRESHOLD_MULT'] = 5.0
    
    # =========================================================================
    # NEW: Terrain Referenced Navigation (TRN) v3.3.0
    # =========================================================================
    if 'trn' in config:
        trn_cfg = config['trn']
        result['TRN_ENABLED'] = trn_cfg.get('enabled', False)
        result['TRN_PROFILE_WINDOW'] = trn_cfg.get('profile_window_sec', 30.0)
        result['TRN_MIN_SAMPLES'] = trn_cfg.get('min_samples', 20)
        result['TRN_SEARCH_RADIUS'] = trn_cfg.get('search_radius_m', 500.0)
        result['TRN_SEARCH_STEP'] = trn_cfg.get('search_step_m', 30.0)
        result['TRN_MIN_TERRAIN_VAR'] = trn_cfg.get('min_terrain_variation_m', 10.0)
        result['TRN_CORR_THRESHOLD'] = trn_cfg.get('max_correlation_threshold', 0.7)
        result['TRN_MIN_ALT_VAR'] = trn_cfg.get('min_altitude_variation_m', 5.0)
        result['TRN_UPDATE_INTERVAL'] = trn_cfg.get('update_interval_sec', 10.0)
        result['TRN_SIGMA_XY'] = trn_cfg.get('sigma_trn_xy', 50.0)
        result['trn'] = trn_cfg  # Store full config for TRN module
    else:
        result['TRN_ENABLED'] = False
        result['trn'] = {}
    
    # =========================================================================
    # Create VIOConfig dataclass from parsed YAML (v3.2.0)
    # =========================================================================
    cam = config['camera']
    vio_config = VIOConfig(
        # Paths will be set by CLI - leave as defaults
        imu_path="",
        quarry_path="",
        output_dir="",
        config_yaml=config_path,
        
        # Image processing from camera section
        downscale_size=(cam.get('image_width', 1440), cam.get('image_height', 1080)),
        
        # State options
        camera_view=result['DEFAULT_CAMERA_VIEW'],
        
        # Algorithm options from YAML
        estimate_imu_bias=result['ESTIMATE_IMU_BIAS'],
        use_magnetometer=result['MAG_ENABLED'],
        use_mag_estimated_bias=result['MAG_USE_ESTIMATED_BIAS'],
        sigma_mag_bias_init=result['SIGMA_MAG_BIAS_INIT'],
        sigma_mag_bias=result['SIGMA_MAG_BIAS'],
        use_vio_velocity=result['USE_VIO_VELOCITY'],
        estimator_mode=result['ESTIMATOR_MODE'],
        
        # Performance options
        fast_mode=result['FAST_MODE'],
        frame_skip=result['FRAME_SKIP'],
        
        # Debug options (default off, CLI can enable)
        save_debug_data=False,
        save_keyframe_images=False,
        
        # CRITICAL: Store BOTH for backward compatibility
        # _raw_config: Flat dict for legacy code (IMU_PARAMS, KB_PARAMS, etc.)
        # _yaml_config: Original YAML dict for new modules (VPS, TRN, etc.)
        _raw_config=result,  # ← Flat dict (backward compatible)
        _yaml_config=config  # ← Original YAML dict (for VPS/TRN)
    )
    
    return vio_config


# =============================================================================
# Default Configuration Variables (will be overridden by load_config)
# =============================================================================

# Camera parameters
KB_PARAMS = {}
BODY_T_CAMDOWN = np.eye(4, dtype=np.float64)
BODY_T_CAMFRONT = np.eye(4, dtype=np.float64)
BODY_T_CAMSIDE = np.eye(4, dtype=np.float64)

# IMU parameters
IMU_PARAMS = {}
IMU_PARAMS_PREINT = {}

# IMU-GNSS Lever Arm (will be overridden by load_config if available)
BODY_T_GNSS = np.eye(4, dtype=np.float64)
IMU_GNSS_LEVER_ARM = np.zeros(3, dtype=np.float64)

# Magnetometer parameters
MAG_HARD_IRON_OFFSET = np.zeros(3, dtype=float)
MAG_SOFT_IRON_MATRIX = np.eye(3, dtype=float)
MAG_DECLINATION = 0.0
MAG_FIELD_STRENGTH = 50.0
MAG_MIN_FIELD_STRENGTH = 0.1  # RELAXED: normalized data may have low magnitude
MAG_MAX_FIELD_STRENGTH = 100.0
MAG_UPDATE_RATE_LIMIT = 1  # Process EVERY mag sample for faster convergence
MAG_USE_RAW_HEADING = True  # GPS-calibrated raw body-frame heading
MAG_APPLY_INITIAL_CORRECTION = True  # Apply mag correction at startup
MAG_INITIAL_CONVERGENCE_WINDOW = 30.0  # 30s initial settling window
# REMOVED: MAG_ADAPTIVE_THRESHOLD (was dead code - never referenced)

# Enhanced Magnetometer Filtering Parameters
MAG_EMA_ALPHA = 0.3  # EMA smoothing factor: 0.3 = 30% new, 70% old
MAG_GYRO_CONSISTENCY_THRESHOLD = np.radians(5.0)  # Max deviation from gyro (5°)
MAG_MAX_YAW_RATE = np.radians(45.0)  # Max expected yaw rate from gyro (45°/s)
MAG_CONSISTENCY_R_INFLATE = 4.0  # Inflate R by this factor when inconsistent

# Process noise parameters
SIGMA_ACCEL = 0.8
# REMOVED: SIGMA_VO_VEL (was dead code - never referenced, use _raw_config['SIGMA_VO'] instead)
SIGMA_VPS_XY = 1.0
SIGMA_AGL_Z = 2.5
SIGMA_MAG_YAW = 0.15  # ~9° measurement noise

# VIO parameters
MIN_PARALLAX_PX = 2.0
MIN_MSCKF_BASELINE = 0.15
MSCKF_CHI2_MULTIPLIER = 4.0
MSCKF_MAX_REPROJECTION_ERROR = 3.0
VO_MIN_INLIERS = 15
VO_RATIO_TEST = 0.75
VO_NADIR_ALIGN_DEG = 30.0
VO_FRONT_ALIGN_DEG = 60.0

# Camera view configurations with default values
# These may be overridden by load_config()
CAMERA_VIEW_CONFIGS = {
    'nadir': {
        'extrinsics': 'BODY_T_CAMDOWN',
        'nadir_threshold': 30.0,
        'sigma_scale_xy': 2.0,
        'sigma_scale_z': 1.0,
        'use_vz_only': True,
        'min_parallax': 2.0,
        'max_corners': 150,
        'align_degrees': 30.0,
        'vel_obs_type': 'z',
        'h_vel_idx': [2],
        'T_body_cam': np.eye(4),
    },
    'front': {
        'extrinsics': 'BODY_T_CAMFRONT',
        'nadir_threshold': 60.0,
        'sigma_scale_xy': 2.0,
        'sigma_scale_z': 2.0,
        'use_vz_only': False,
        'min_parallax': 2.0,
        'max_corners': 200,
        'align_degrees': 60.0, 
        'vel_obs_type': 'xyz',
        'h_vel_idx': [0, 1, 2],
        'T_body_cam': np.eye(4),
    },
    'side': {
        'extrinsics': 'BODY_T_CAMSIDE',
        'nadir_threshold': 45.0,
        'sigma_scale_xy': 2.0,
        'sigma_scale_z': 2.0,
        'use_vz_only': False,
        'min_parallax': 2.0,
        'max_corners': 150,
        'align_degrees': 45.0,
        'vel_obs_type': 'xy',
        'h_vel_idx': [0, 1],
        'T_body_cam': np.eye(4),
    },
}
USE_FISHEYE = True

# IMU bias settings
ESTIMATE_IMU_BIAS = False
INITIAL_GYRO_BIAS = np.zeros(3, dtype=float)
INITIAL_ACCEL_BIAS = np.zeros(3, dtype=float)


# =============================================================================
# REMOVED: MSCKF_STATS, reset_msckf_stats(), print_msckf_stats()
# These are now defined in vio/msckf.py (single source of truth)
# Import from msckf module: from vio.msckf import MSCKF_STATS, reset_msckf_stats
# =============================================================================

# =============================================================================
# REMOVED: MAG_FILTER_STATE
# This is now defined in vio/magnetometer.py as _MAG_FILTER_STATE
# Import from magnetometer module: from vio.magnetometer import get_mag_filter_state
# =============================================================================
