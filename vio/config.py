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


def _merge_dict_defaults(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge user config on top of defaults."""
    merged = dict(defaults)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


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
    timeref_pps_csv: Optional[str] = None
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
    result['PHASE_HYSTERESIS_ENABLED'] = bool(phase_detection.get('hysteresis_enabled', True))
    result['PHASE_UP_HOLD_SEC'] = float(phase_detection.get('up_hold_sec', 0.75))
    result['PHASE_DOWN_HOLD_SEC'] = float(phase_detection.get('down_hold_sec', 6.0))
    result['PHASE_ALLOW_NORMAL_TO_EARLY'] = bool(phase_detection.get('allow_normal_to_early', True))
    result['PHASE_REVERT_MAX_SPEED'] = float(phase_detection.get('revert_max_speed', 18.0))
    result['PHASE_REVERT_MAX_ALT_CHANGE'] = float(phase_detection.get('revert_max_alt_change', 60.0))

    # ZUPT defaults (used by IMU helper path; adaptive may scale these values)
    zupt = config.get('zupt', {})
    result['ZUPT_ENABLED'] = bool(zupt.get('enabled', True))
    result['ZUPT_ACCEL_THRESHOLD'] = float(zupt.get('accel_threshold', 0.5))
    result['ZUPT_GYRO_THRESHOLD'] = float(zupt.get('gyro_threshold', 0.05))
    result['ZUPT_VELOCITY_THRESHOLD'] = float(zupt.get('velocity_threshold', 0.3))
    result['ZUPT_MAX_V_FOR_UPDATE'] = float(zupt.get('max_v_for_zupt', 20.0))
    result['ZUPT_FLOW_GUARD_PX'] = float(zupt.get('flow_guard_px', 1.0))
    result['ZUPT_FLOW_GUARD_WINDOW_SEC'] = float(zupt.get('flow_guard_window_sec', 0.30))
    
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
    result['MAG_VISION_HEADING_CONSISTENCY_ENABLE'] = bool(
        mag.get('vision_heading_consistency_enable', True)
    )
    result['MAG_VISION_HEADING_MAX_AGE_SEC'] = float(
        mag.get('vision_heading_max_age_sec', 1.0)
    )
    result['MAG_VISION_HEADING_MIN_QUALITY'] = float(
        mag.get('vision_heading_min_quality', 0.15)
    )
    result['MAG_VISION_HEADING_SOFT_DEG'] = float(
        mag.get('vision_heading_soft_deg', 30.0)
    )
    result['MAG_VISION_HEADING_HARD_DEG'] = float(
        mag.get('vision_heading_hard_deg', 85.0)
    )
    result['MAG_VISION_HEADING_R_INFLATE_MAX'] = float(
        mag.get('vision_heading_r_inflate_max', 6.0)
    )
    result['MAG_VISION_HEADING_STRONG_QUALITY'] = float(
        mag.get('vision_heading_strong_quality', 0.60)
    )
    result['MAG_VISION_HEADING_PHASE_SOFT_DEG'] = mag.get(
        'vision_heading_phase_soft_deg',
        {'0': 36.0, '1': 28.0, '2': 24.0}
    )
    result['MAG_VISION_HEADING_PHASE_HARD_DEG'] = mag.get(
        'vision_heading_phase_hard_deg',
        {'0': 95.0, '1': 78.0, '2': 62.0}
    )
    result['MAG_VISION_HEADING_HEALTH_THRESHOLD_MULT'] = mag.get(
        'vision_heading_health_threshold_mult',
        {'HEALTHY': 1.0, 'WARNING': 0.90, 'DEGRADED': 0.80, 'RECOVERY': 0.95}
    )
    result['MAG_VISION_HEADING_HEALTH_R_MULT'] = mag.get(
        'vision_heading_health_r_mult',
        {'HEALTHY': 1.0, 'WARNING': 1.20, 'DEGRADED': 1.40, 'RECOVERY': 1.10}
    )
    result['MAG_VISION_HEADING_HIGH_SPEED_M_S'] = float(
        mag.get('vision_heading_high_speed_m_s', 45.0)
    )
    result['MAG_VISION_HEADING_HIGH_SPEED_THRESH_MULT'] = float(
        mag.get('vision_heading_high_speed_thresh_mult', 0.85)
    )
    result['MAG_VISION_HEADING_HIGH_SPEED_R_MULT'] = float(
        mag.get('vision_heading_high_speed_r_mult', 1.20)
    )
    warning_weak_yaw = mag.get('warning_weak_yaw', {})
    result['MAG_WARNING_R_MULT'] = float(warning_weak_yaw.get('warning_r_mult', 4.0))
    result['MAG_DEGRADED_R_MULT'] = float(warning_weak_yaw.get('degraded_r_mult', 8.0))
    result['MAG_WARNING_MAX_DYAW_DEG'] = float(warning_weak_yaw.get('warning_max_dyaw_deg', 1.5))
    result['MAG_DEGRADED_MAX_DYAW_DEG'] = float(warning_weak_yaw.get('degraded_max_dyaw_deg', 1.0))
    result['MAG_WARNING_SKIP_VISION_HARD_MISMATCH'] = bool(
        warning_weak_yaw.get('skip_vision_hard_mismatch', True)
    )
    result['MAG_CONDITIONING_GUARD_ENABLE'] = bool(
        warning_weak_yaw.get('conditioning_guard_enable', True)
    )
    result['MAG_CONDITIONING_GUARD_WARN_PCOND'] = float(
        warning_weak_yaw.get('conditioning_guard_warn_pcond', 8e11)
    )
    result['MAG_CONDITIONING_GUARD_DEGRADED_PCOND'] = float(
        warning_weak_yaw.get('conditioning_guard_degraded_pcond', 1e11)
    )
    result['MAG_CONDITIONING_GUARD_WARN_PMAX'] = float(
        warning_weak_yaw.get('conditioning_guard_warn_pmax', 8e6)
    )
    result['MAG_CONDITIONING_GUARD_DEGRADED_PMAX'] = float(
        warning_weak_yaw.get('conditioning_guard_degraded_pmax', 2e6)
    )
    result['MAG_CONDITIONING_GUARD_HARD_PCOND'] = float(
        warning_weak_yaw.get('conditioning_guard_hard_pcond', 1e12)
    )
    result['MAG_CONDITIONING_GUARD_HARD_PMAX'] = float(
        warning_weak_yaw.get('conditioning_guard_hard_pmax', 1e7)
    )
    result['MAG_BIAS_FREEZE_WARN_PCOND'] = float(
        warning_weak_yaw.get('bias_freeze_warn_pcond', 1e10)
    )
    result['MAG_BIAS_FREEZE_DEGRADED_PCOND'] = float(
        warning_weak_yaw.get('bias_freeze_degraded_pcond', 5e9)
    )
    result['MAG_WARNING_EXTRA_R_MULT'] = float(
        warning_weak_yaw.get('warning_extra_r_mult', 2.0)
    )
    result['MAG_DEGRADED_EXTRA_R_MULT'] = float(
        warning_weak_yaw.get('degraded_extra_r_mult', 2.8)
    )
    result['VISION_HEADING_MIN_INLIERS'] = int(
        mag.get('vision_heading_min_inliers', 25)
    )
    result['VISION_HEADING_MIN_PARALLAX_PX'] = float(
        mag.get('vision_heading_min_parallax_px', 1.0)
    )
    result['VISION_HEADING_MAX_DELTA_DEG'] = float(
        mag.get('vision_heading_max_delta_deg', 20.0)
    )
    result['VISION_HEADING_QUALITY_DECAY'] = float(
        mag.get('vision_heading_quality_decay', 0.92)
    )
    
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
    
    # VIO parameters
    vio = config['vio']
    result['SIGMA_VO'] = vio['sigma_vo']  # Velocity measurement uncertainty
    result['MIN_PARALLAX_PX'] = vio['min_parallax_px']
    result['MIN_MSCKF_BASELINE'] = vio['min_msckf_baseline']
    result['MSCKF_CHI2_MULTIPLIER'] = vio['msckf_chi2_multiplier']
    result['MSCKF_MAX_REPROJECTION_ERROR'] = vio['msckf_max_reprojection_error']
    result['VO_MIN_INLIERS'] = vio['min_inliers']
    result['VO_RATIO_TEST'] = vio['ratio_test']
    # Optical-flow velocity scale mode: fixed | dynamic | hybrid
    result['VIO_FLOW_AGL_MODE'] = str(vio.get('flow_agl_mode', 'hybrid')).lower()
    result['VIO_FLOW_MIN_AGL'] = float(vio.get('flow_min_agl', 1.0))
    result['VIO_FLOW_MAX_AGL'] = float(vio.get('flow_max_agl', 500.0))
    
    # MSCKF sliding window parameters (v3.9.1)
    msckf_cfg = vio.get('msckf', {})
    result['MSCKF_MAX_CLONE_SIZE'] = msckf_cfg.get('max_clone_size', 11)
    result['MSCKF_MIN_TRACK_LENGTH'] = msckf_cfg.get('min_track_length', 4)
    result['MSCKF_PHASE_CHI2_SCALE'] = msckf_cfg.get(
        'phase_chi2_scale',
        {'0': 1.20, '1': 1.08, '2': 1.00}
    )
    result['MSCKF_PHASE_REPROJ_SCALE'] = msckf_cfg.get(
        'phase_reproj_scale',
        {'0': 1.15, '1': 1.05, '2': 0.95}
    )
    result['MSCKF_RAY_SOFT_FACTOR'] = float(msckf_cfg.get('ray_soft_factor', 1.8))
    result['MSCKF_PIXEL_SOFT_FACTOR'] = float(msckf_cfg.get('pixel_soft_factor', 1.6))
    result['MSCKF_AVG_REPROJ_GATE_FACTOR'] = float(msckf_cfg.get('avg_reproj_gate_factor', 2.0))
    result['MSCKF_PHASE_REPROJ_GATE_SCALE'] = msckf_cfg.get(
        'phase_reproj_gate_scale',
        {'0': 1.20, '1': 1.08, '2': 1.00}
    )
    result['MSCKF_HEALTH_REPROJ_GATE_SCALE'] = msckf_cfg.get(
        'health_reproj_gate_scale',
        {'HEALTHY': 1.0, 'WARNING': 1.10, 'DEGRADED': 1.20, 'RECOVERY': 1.05}
    )
    result['MSCKF_PHASE_AVG_REPROJ_GATE_SCALE'] = msckf_cfg.get(
        'phase_avg_reproj_gate_scale',
        {'0': 1.15, '1': 1.05, '2': 1.00}
    )
    result['MSCKF_HEALTH_AVG_REPROJ_GATE_SCALE'] = msckf_cfg.get(
        'health_avg_reproj_gate_scale',
        {'HEALTHY': 1.0, 'WARNING': 1.08, 'DEGRADED': 1.15, 'RECOVERY': 1.04}
    )
    reproj_state_aware = msckf_cfg.get('reproj_state_aware', {})
    result['MSCKF_REPROJ_QUALITY_HIGH_TH'] = float(reproj_state_aware.get('quality_high_th', 0.75))
    result['MSCKF_REPROJ_QUALITY_LOW_TH'] = float(reproj_state_aware.get('quality_low_th', 0.45))
    result['MSCKF_REPROJ_QUALITY_MID_GATE_MULT'] = float(reproj_state_aware.get('mid_gate_mult', 1.15))
    result['MSCKF_REPROJ_QUALITY_LOW_REJECT'] = bool(reproj_state_aware.get('low_quality_reject', True))
    result['MSCKF_REPROJ_WARNING_SCALE'] = float(reproj_state_aware.get('warning_scale', 1.20))
    result['MSCKF_REPROJ_DEGRADED_SCALE'] = float(reproj_state_aware.get('degraded_scale', 1.35))
    result['VO_NADIR_ALIGN_DEG'] = vio['views']['nadir']['nadir_threshold_deg']
    result['VO_FRONT_ALIGN_DEG'] = vio['views']['front']['nadir_threshold_deg']
    
    # VIO velocity toggle (v3.1.0)
    result['USE_VIO_VELOCITY'] = vio.get('use_vio_velocity', True)
    result['VIO_NADIR_XY_ONLY_VELOCITY'] = bool(vio.get('nadir_xy_only_velocity', False))
    vio_vel_cfg = config.get('vio_vel', {})
    result['VIO_VEL_XY_ONLY_CHI2_SCALE'] = float(vio_vel_cfg.get('xy_only_chi2_scale', 1.10))
    result['VIO_VEL_SPEED_R_INFLATE_BREAKPOINTS_M_S'] = list(
        vio_vel_cfg.get('speed_r_inflate_breakpoints_m_s', [25.0, 40.0, 55.0])
    )
    result['VIO_VEL_SPEED_R_INFLATE_VALUES'] = list(
        vio_vel_cfg.get('speed_r_inflate_values', [1.5, 2.5, 4.0])
    )
    result['VIO_VEL_MAX_DELTA_V_XY_PER_UPDATE_M_S'] = float(
        vio_vel_cfg.get('max_delta_v_xy_per_update_m_s', 2.0)
    )
    result['VIO_VEL_MAX_DELTA_V_XY_HIGH_SPEED_M_S'] = float(
        vio_vel_cfg.get(
            'max_delta_v_xy_high_speed_m_s',
            vio_vel_cfg.get('max_delta_v_xy_per_update_m_s', 2.0),
        )
    )
    result['VIO_VEL_DELTA_V_SOFT_ENABLE'] = bool(
        vio_vel_cfg.get('delta_v_soft_enable', True)
    )
    result['VIO_VEL_DELTA_V_SOFT_FACTOR'] = float(
        vio_vel_cfg.get('delta_v_soft_factor', 2.0)
    )
    result['VIO_VEL_DELTA_V_HARD_FACTOR'] = float(
        vio_vel_cfg.get('delta_v_hard_factor', 3.0)
    )
    result['VIO_VEL_DELTA_V_SOFT_MAX_R_MULT'] = float(
        vio_vel_cfg.get('delta_v_soft_max_r_mult', 6.0)
    )
    result['VIO_VEL_HIGH_SPEED_BP_M_S'] = float(
        vio_vel_cfg.get('high_speed_bp_m_s', 25.0)
    )
    result['VIO_VEL_MIN_FLOW_PX_HIGH_SPEED'] = float(
        vio_vel_cfg.get('min_flow_px_high_speed', 0.8)
    )
    
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

    # Runtime logging policy (separate from adaptive.logging)
    runtime_logging = config.get("logging", {}) if isinstance(config.get("logging", {}), dict) else {}
    runtime_verbosity_default = os.environ.get("VIO_RUNTIME_VERBOSITY", "debug")
    result["LOG_RUNTIME_VERBOSITY"] = str(
        runtime_logging.get("runtime_verbosity", runtime_verbosity_default)
    ).lower()
    result["LOG_RUNTIME_MIN_INTERVAL_SEC"] = float(runtime_logging.get("runtime_log_interval_sec", 1.0))
    result["INFERENCE_LOG_FLUSH_STRIDE"] = int(
        max(1, runtime_logging.get("inference_log_flush_stride", 200))
    )
    
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
        result['LOOP_MIN_MATCH_RATIO'] = lc.get('min_match_ratio', 0.12)
        result['LOOP_MIN_INLIERS'] = lc.get('min_inliers', 15)
        qg = lc.get('quality_gate', {})
        result['LOOP_MIN_INLIERS_HARD'] = int(qg.get('min_inliers_hard', 35))
        result['LOOP_MIN_INLIERS_FAILSOFT'] = int(qg.get('min_inliers_failsoft', 20))
        result['LOOP_MIN_SPATIAL_SPREAD'] = float(qg.get('min_spatial_spread', 0.18))
        result['LOOP_MAX_REPROJ_PX'] = float(qg.get('max_reproj_px', 2.5))
        result['LOOP_YAW_RESIDUAL_BOUND_DEG'] = float(qg.get('yaw_residual_bound_deg', 25.0))
        result['LOOP_DOUBLE_CONFIRM_ENABLE'] = bool(qg.get('double_confirm_enable', True))
        result['LOOP_DOUBLE_CONFIRM_WINDOW_SEC'] = float(qg.get('double_confirm_window_sec', 2.0))
        result['LOOP_DOUBLE_CONFIRM_YAW_DEG'] = float(qg.get('double_confirm_yaw_deg', 8.0))
        result['LOOP_COOLDOWN_SEC'] = float(qg.get('cooldown_sec', 2.0))
        result['LOOP_PHASE_DYNAMIC_INLIER_MULT'] = float(qg.get('phase_dynamic_inlier_mult', 1.15))
        result['LOOP_WARNING_INLIER_MULT'] = float(qg.get('warning_inlier_mult', 1.10))
        result['LOOP_DEGRADED_INLIER_MULT'] = float(qg.get('degraded_inlier_mult', 1.20))
        fs = lc.get('fail_soft', {})
        result['LOOP_FAIL_SOFT_ENABLE'] = bool(fs.get('enable', True))
        result['LOOP_FAIL_SOFT_SIGMA_YAW_DEG'] = float(fs.get('fail_soft_sigma_yaw_deg', 18.0))
        result['LOOP_BASE_SIGMA_YAW_DEG'] = float(fs.get('base_sigma_yaw_deg', 5.0))
        result['LOOP_MAX_ABS_YAW_CORR_DEG'] = float(fs.get('max_abs_yaw_corr_deg', 4.0))
        result['LOOP_MIN_ABS_YAW_CORR_DEG'] = float(fs.get('min_abs_yaw_corr_deg', 1.5))
        result['LOOP_DYNAMIC_PHASE_SIGMA_MULT'] = float(fs.get('dynamic_phase_sigma_mult', 1.15))
        result['LOOP_WARNING_SIGMA_MULT'] = float(fs.get('warning_sigma_mult', 1.20))
        result['LOOP_DEGRADED_SIGMA_MULT'] = float(fs.get('degraded_sigma_mult', 1.40))
        result['LOOP_SPEED_SKIP_M_S'] = float(fs.get('speed_skip_m_s', 35.0))
        result['LOOP_SPEED_SIGMA_INFLATE_M_S'] = float(fs.get('speed_sigma_inflate_m_s', 25.0))
        result['LOOP_SPEED_SIGMA_MULT'] = float(fs.get('speed_sigma_mult', 1.5))
    else:
        result['USE_LOOP_CLOSURE'] = True
        result['LOOP_POSITION_THRESHOLD'] = 30.0
        result['LOOP_MIN_KEYFRAME_DIST'] = 15.0
        result['LOOP_MIN_KEYFRAME_YAW'] = 20.0
        result['LOOP_MIN_FRAME_GAP'] = 50
        result['LOOP_MIN_MATCH_RATIO'] = 0.12
        result['LOOP_MIN_INLIERS'] = 15
        result['LOOP_MIN_INLIERS_HARD'] = 35
        result['LOOP_MIN_INLIERS_FAILSOFT'] = 20
        result['LOOP_MIN_SPATIAL_SPREAD'] = 0.18
        result['LOOP_MAX_REPROJ_PX'] = 2.5
        result['LOOP_YAW_RESIDUAL_BOUND_DEG'] = 25.0
        result['LOOP_DOUBLE_CONFIRM_ENABLE'] = True
        result['LOOP_DOUBLE_CONFIRM_WINDOW_SEC'] = 2.0
        result['LOOP_DOUBLE_CONFIRM_YAW_DEG'] = 8.0
        result['LOOP_COOLDOWN_SEC'] = 2.0
        result['LOOP_PHASE_DYNAMIC_INLIER_MULT'] = 1.15
        result['LOOP_WARNING_INLIER_MULT'] = 1.10
        result['LOOP_DEGRADED_INLIER_MULT'] = 1.20
        result['LOOP_FAIL_SOFT_ENABLE'] = True
        result['LOOP_FAIL_SOFT_SIGMA_YAW_DEG'] = 18.0
        result['LOOP_BASE_SIGMA_YAW_DEG'] = 5.0
        result['LOOP_MAX_ABS_YAW_CORR_DEG'] = 4.0
        result['LOOP_MIN_ABS_YAW_CORR_DEG'] = 1.5
        result['LOOP_DYNAMIC_PHASE_SIGMA_MULT'] = 1.15
        result['LOOP_WARNING_SIGMA_MULT'] = 1.20
        result['LOOP_DEGRADED_SIGMA_MULT'] = 1.40
        result['LOOP_SPEED_SKIP_M_S'] = 35.0
        result['LOOP_SPEED_SIGMA_INFLATE_M_S'] = 25.0
        result['LOOP_SPEED_SIGMA_MULT'] = 1.5

    # =========================================================================
    # Kinematic consistency guard
    # =========================================================================
    kin_cfg = config.get('kinematic_guard', {})
    result['KIN_GUARD_ENABLED'] = bool(kin_cfg.get('enabled', True))
    result['KIN_GUARD_WINDOW_SEC'] = float(kin_cfg.get('window_sec', 0.5))
    result['KIN_GUARD_VEL_MISMATCH_WARN'] = float(kin_cfg.get('vel_mismatch_warn', 8.0))
    result['KIN_GUARD_VEL_MISMATCH_HARD'] = float(kin_cfg.get('vel_mismatch_hard', 15.0))
    result['KIN_GUARD_MAX_INFLATE'] = float(kin_cfg.get('max_inflate', 1.25))
    result['KIN_GUARD_HARD_BLEND_ALPHA'] = float(kin_cfg.get('hard_blend_alpha', 0.0))
    result['KIN_GUARD_MIN_ACTION_DT_SEC'] = float(kin_cfg.get('min_action_dt_sec', 0.25))
    result['KIN_GUARD_MAX_STATE_SPEED_M_S'] = float(kin_cfg.get('max_state_speed_m_s', 120.0))
    result['KIN_GUARD_MAX_BLEND_SPEED_M_S'] = float(kin_cfg.get('max_blend_speed_m_s', 60.0))
    result['KIN_GUARD_MAX_KIN_SPEED_M_S'] = float(kin_cfg.get('max_kin_speed_m_s', 80.0))
    result['KIN_GUARD_ABS_SPEED_SANITY_M_S'] = float(
        kin_cfg.get('abs_speed_sanity_m_s', max(120.0, 1.8 * float(kin_cfg.get('max_state_speed_m_s', 120.0))))
    )
    result['KIN_GUARD_HARD_HOLD_SEC'] = float(kin_cfg.get('hard_hold_sec', 0.30))
    result['KIN_GUARD_SPEED_HARD_M_S'] = float(
        kin_cfg.get('speed_hard_m_s', kin_cfg.get('max_state_speed_m_s', 120.0))
    )
    result['KIN_GUARD_SPEED_HARD_HOLD_SEC'] = float(
        kin_cfg.get('speed_hard_hold_sec', kin_cfg.get('hard_hold_sec', 0.30))
    )
    result['KIN_GUARD_RELEASE_HYSTERESIS_RATIO'] = float(
        kin_cfg.get('release_hysteresis_ratio', 0.75)
    )
    result['KIN_GUARD_SPEED_RELEASE_HYSTERESIS_RATIO'] = float(
        kin_cfg.get(
            'speed_release_hysteresis_ratio',
            kin_cfg.get('release_hysteresis_ratio', 0.75),
        )
    )
    result['KIN_GUARD_SPEED_BLEND_ALPHA'] = float(
        kin_cfg.get('speed_blend_alpha', max(0.2, float(kin_cfg.get('hard_blend_alpha', 0.0))))
    )
    result['KIN_GUARD_SPEED_INFLATE'] = float(kin_cfg.get('speed_inflate', 1.12))
    
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
    # Objective / optimization mode (accuracy-first vs runtime-first)
    # =========================================================================
    opt_cfg = config.get('optimization', {})
    result['OBJECTIVE_MODE'] = str(opt_cfg.get('objective', 'stability')).lower()
    result['COMPUTE_BUDGET'] = str(opt_cfg.get('compute_budget', 'low')).lower()

    # =========================================================================
    # VPS accuracy mode + relocalization policy
    # =========================================================================
    vps_cfg = config.get('vps', {})
    result['VPS_ACCURACY_MODE'] = bool(vps_cfg.get('accuracy_mode', False))
    result['VPS_APPLY_MIN_INLIERS'] = int(vps_cfg.get('apply_min_inliers', 8))
    result['VPS_APPLY_MIN_CONFIDENCE'] = float(vps_cfg.get('apply_min_confidence', 0.18))
    result['VPS_APPLY_MAX_REPROJ_ERROR'] = float(vps_cfg.get('apply_max_reproj_error', 1.2))
    result['VPS_APPLY_MAX_SPEED_M_S'] = float(vps_cfg.get('apply_max_speed_m_s', 80.0))
    result['VPS_APPLY_WARNING_INLIER_BONUS'] = int(vps_cfg.get('apply_warning_inlier_bonus', 2))
    result['VPS_APPLY_DEGRADED_INLIER_BONUS'] = int(vps_cfg.get('apply_degraded_inlier_bonus', 4))
    result['VPS_APPLY_WARNING_CONF_MULT'] = float(vps_cfg.get('apply_warning_conf_mult', 1.15))
    result['VPS_APPLY_DEGRADED_CONF_MULT'] = float(vps_cfg.get('apply_degraded_conf_mult', 1.30))
    result['VPS_APPLY_WARNING_REPROJ_MULT'] = float(vps_cfg.get('apply_warning_reproj_mult', 0.90))
    result['VPS_APPLY_DEGRADED_REPROJ_MULT'] = float(vps_cfg.get('apply_degraded_reproj_mult', 0.80))
    result['VPS_APPLY_FAILSOFT_ENABLE'] = bool(vps_cfg.get('apply_failsoft_enable', True))
    result['VPS_APPLY_FAILSOFT_MIN_INLIERS'] = int(
        vps_cfg.get('apply_failsoft_min_inliers', vps_cfg.get('min_inliers_failsoft', 5))
    )
    result['VPS_APPLY_FAILSOFT_MIN_CONFIDENCE'] = float(
        vps_cfg.get('apply_failsoft_min_confidence', vps_cfg.get('min_confidence_failsoft', 0.12))
    )
    result['VPS_APPLY_FAILSOFT_MAX_REPROJ_ERROR'] = float(
        vps_cfg.get('apply_failsoft_max_reproj_error', vps_cfg.get('max_reproj_error_failsoft', 1.2))
    )
    result['VPS_APPLY_FAILSOFT_MAX_SPEED_M_S'] = float(
        vps_cfg.get('apply_failsoft_max_speed_m_s', vps_cfg.get('apply_max_speed_m_s', 80.0))
    )
    result['VPS_APPLY_FAILSOFT_R_MULT'] = float(
        vps_cfg.get('apply_failsoft_r_mult', 1.5)
    )
    result['VPS_APPLY_FAILSOFT_MAX_OFFSET_M'] = float(
        vps_cfg.get('apply_failsoft_max_offset_m', 180.0)
    )
    result['VPS_APPLY_FAILSOFT_MAX_DIR_CHANGE_DEG'] = float(
        vps_cfg.get('apply_failsoft_max_dir_change_deg', 60.0)
    )
    result['VPS_APPLY_FAILSOFT_LARGE_OFFSET_CONFIRM_M'] = float(
        vps_cfg.get('apply_failsoft_large_offset_confirm_m', 80.0)
    )
    result['VPS_APPLY_FAILSOFT_LARGE_OFFSET_CONFIRM_HITS'] = int(
        vps_cfg.get('apply_failsoft_large_offset_confirm_hits', 2)
    )
    result['VPS_APPLY_FAILSOFT_ALLOW_WARNING'] = bool(
        vps_cfg.get('apply_failsoft_allow_warning', True)
    )
    result['VPS_APPLY_FAILSOFT_ALLOW_DEGRADED'] = bool(
        vps_cfg.get('apply_failsoft_allow_degraded', False)
    )
    result['VPS_ABS_HARD_REJECT_OFFSET_M'] = float(
        vps_cfg.get('abs_hard_reject_offset_m', 180.0)
    )
    result['VPS_ABS_HARD_REJECT_DIR_CHANGE_DEG'] = float(
        vps_cfg.get('abs_hard_reject_dir_change_deg', 75.0)
    )
    result['VPS_ABS_MAX_APPLY_DP_XY_M'] = float(
        vps_cfg.get('abs_max_apply_dp_xy_m', 25.0)
    )

    vps_reloc_defaults = {
        'enabled': True,
        'global_interval_sec': 12.0,
        'fail_streak_trigger': 6,
        'stale_success_sec': 8.0,
        'xy_sigma_trigger_m': 35.0,
        'max_centers': 10,
        'ring_radius_m': [35.0, 80.0],
        'ring_samples': 8,
        'global_yaw_hypotheses_deg': [0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0],
        'global_scale_hypotheses': [0.80, 0.90, 1.00, 1.10, 1.20],
        'force_global_on_warning_phase': False,
    }
    result['VPS_RELOCALIZATION'] = _merge_dict_defaults(
        vps_reloc_defaults, vps_cfg.get('relocalization', {})
    )
    _vps_reloc = result['VPS_RELOCALIZATION']
    result['VPS_RELOC_ENABLED'] = bool(_vps_reloc.get('enabled', True))
    result['VPS_RELOC_GLOBAL_INTERVAL_SEC'] = float(_vps_reloc.get('global_interval_sec', 12.0))
    result['VPS_RELOC_FAIL_STREAK_TRIGGER'] = int(_vps_reloc.get('fail_streak_trigger', 6))
    result['VPS_RELOC_STALE_SUCCESS_SEC'] = float(_vps_reloc.get('stale_success_sec', 8.0))
    result['VPS_RELOC_XY_SIGMA_TRIGGER_M'] = float(_vps_reloc.get('xy_sigma_trigger_m', 35.0))
    result['VPS_RELOC_MAX_CENTERS'] = int(_vps_reloc.get('max_centers', 10))
    result['VPS_RELOC_RING_RADIUS_M'] = list(_vps_reloc.get('ring_radius_m', [35.0, 80.0]))
    result['VPS_RELOC_RING_SAMPLES'] = int(_vps_reloc.get('ring_samples', 8))
    result['VPS_RELOC_GLOBAL_YAW_HYPOTHESES_DEG'] = list(
        _vps_reloc.get('global_yaw_hypotheses_deg', [0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0])
    )
    result['VPS_RELOC_GLOBAL_SCALE_HYPOTHESES'] = list(
        _vps_reloc.get('global_scale_hypotheses', [0.80, 0.90, 1.00, 1.10, 1.20])
    )
    result['VPS_RELOC_FORCE_GLOBAL_ON_WARNING_PHASE'] = bool(
        _vps_reloc.get('force_global_on_warning_phase', False)
    )

    # =========================================================================
    # Async backend optimizer (fixed-lag, non-blocking frontend)
    # =========================================================================
    backend_cfg = config.get('backend', {})
    result['BACKEND_ENABLED'] = bool(backend_cfg.get('enabled', False))
    result['BACKEND_FIXED_LAG_WINDOW'] = int(backend_cfg.get('fixed_lag_window', 10))
    result['BACKEND_OPTIMIZE_RATE_HZ'] = float(backend_cfg.get('optimize_rate_hz', 2.0))
    result['BACKEND_MAX_ITERATION_MS'] = float(backend_cfg.get('max_iteration_ms', 35.0))
    result['BACKEND_POLL_INTERVAL_SEC'] = float(backend_cfg.get('poll_interval_sec', 0.5))
    result['BACKEND_MAX_CORRECTION_AGE_SEC'] = float(backend_cfg.get('max_correction_age_sec', 2.0))
    result['BACKEND_BLEND_STEPS'] = int(backend_cfg.get('blend_steps', 3))
    result['BACKEND_MAX_APPLY_DYAW_DEG'] = float(backend_cfg.get('max_apply_dyaw_deg', 2.5))
    result['BACKEND_MAX_APPLY_DP_XY_M'] = float(backend_cfg.get('max_apply_dp_xy_m', 25.0))
    result['BACKEND_CORRECTION_WEIGHT'] = float(backend_cfg.get('correction_weight', 1.0))
    result['BACKEND_APPLY_COV_INFLATE'] = float(backend_cfg.get('apply_cov_inflate', 1.05))
    result['BACKEND_MIN_QUALITY_SCORE'] = float(backend_cfg.get('min_quality_score', 0.20))
    result['BACKEND_MAX_ABS_DP_XY_M'] = float(backend_cfg.get('max_abs_dp_xy_m', 60.0))
    result['BACKEND_MAX_ABS_DYAW_DEG'] = float(backend_cfg.get('max_abs_dyaw_deg', 8.0))

    # =========================================================================
    # NEW: Adaptive + State-aware control policy (IMU-driven)
    # =========================================================================
    adaptive_defaults = {
        'mode': 'off',  # off | shadow | active
        'objective': 'stability_first',
        'health': {
            'warning': {
                'p_cond': 1e10,
                'p_max': 1e6,
                'growth_ratio': 1.05,
            },
            'degraded': {
                'p_cond': 1e12,
                'p_max': 1e7,
                'growth_ratio': 1.10,
                'hold_sec': 0.25,
            },
            'recovery': {
                'healthy_sec': 3.0,
                'to_healthy_sec': 3.0,
            },
            'aiding_age': {
                'full_sec': 0.25,
                'partial_sec': 2.0,
            },
        },
        'nis_feedback': {
            'enabled': True,
            'alpha': 0.1,
            'high': 2.5,
            'low': 0.5,
            'accept_rate_hi': 0.85,
            'window': 200,
            'high_r_scale': 1.5,
            'high_chi2_scale': 0.9,
            'low_r_scale': 0.9,
            'low_chi2_scale': 1.05,
        },
        'process_noise': {
            'aiding_multiplier': {
                'FULL': 1.0,
                'PARTIAL': 0.85,
                'NONE': 0.60,
            },
            'health_profile': {
                'HEALTHY': {
                    'sigma_accel_scale': 1.0,
                    'gyr_w_scale': 1.0,
                    'acc_w_scale': 1.0,
                    'sigma_unmodeled_gyr_scale': 1.0,
                    'min_yaw_scale': 1.0,
                },
                'WARNING': {
                    'sigma_accel_scale': 0.85,
                    'gyr_w_scale': 0.70,
                    'acc_w_scale': 0.70,
                    'sigma_unmodeled_gyr_scale': 0.80,
                    'min_yaw_scale': 0.80,
                },
                'DEGRADED': {
                    'sigma_accel_scale': 0.70,
                    'gyr_w_scale': 0.50,
                    'acc_w_scale': 0.50,
                    'sigma_unmodeled_gyr_scale': 0.65,
                    'min_yaw_scale': 0.65,
                },
                'RECOVERY': {
                    'sigma_accel_scale': 0.90,
                    'gyr_w_scale': 0.80,
                    'acc_w_scale': 0.80,
                    'sigma_unmodeled_gyr_scale': 0.90,
                    'min_yaw_scale': 0.85,
                },
            },
            'min_yaw_floor_deg': 0.05,
            'scale_clamp': {
                'sigma_accel': [0.20, 2.0],
                'gyr_w': [0.10, 2.0],
                'acc_w': [0.10, 2.0],
                'sigma_unmodeled_gyr': [0.10, 2.0],
                'min_yaw': [0.20, 2.0],
            },
        },
        'measurement': {
            'sensor_defaults': {
                'MAG': {'r_scale': 1.0},
                'DEM': {'r_scale': 1.0, 'threshold_scale': 1.0},
                'VIO_VEL': {'r_scale': 1.0, 'chi2_scale': 1.0},
                'MSCKF': {'chi2_scale': 1.0, 'reproj_scale': 1.0},
                'ZUPT': {'r_scale': 1.0, 'chi2_scale': 1.0},
                'VPS': {'r_scale': 1.0, 'chi2_scale': 1.0},
                'GRAVITY_RP': {'r_scale': 1.0, 'chi2_scale': 1.0},
                'YAW_AID': {'r_scale': 1.0, 'chi2_scale': 1.0},
                'BIAS_GUARD': {'r_scale': 1.0, 'chi2_scale': 1.0},
            },
            'clamp': {
                'r_scale': [0.5, 10.0],
                'chi2_scale': [0.5, 2.0],
                'threshold_scale': [0.5, 2.0],
                'reproj_scale': [0.5, 2.0],
            },
            'degraded_r_extra': 1.2,
            'degraded_chi2_extra': 0.95,
            'phase_profiles': {
                'VIO_VEL': {
                    '0': {'chi2_scale': 1.20, 'r_scale': 1.35},
                    '1': {'chi2_scale': 1.10, 'r_scale': 1.20},
                    '2': {'chi2_scale': 1.00, 'r_scale': 1.00},
                },
                'ZUPT': {
                    '0': {'chi2_scale': 0.80, 'r_scale': 1.50, 'acc_threshold_scale': 0.80, 'gyro_threshold_scale': 0.80, 'max_v_scale': 0.70},
                    '1': {'chi2_scale': 0.90, 'r_scale': 1.20, 'acc_threshold_scale': 0.90, 'gyro_threshold_scale': 0.90, 'max_v_scale': 0.85},
                    '2': {'chi2_scale': 1.00, 'r_scale': 1.00, 'acc_threshold_scale': 1.00, 'gyro_threshold_scale': 1.00, 'max_v_scale': 1.00},
                },
                'YAW_AID': {
                    '0': {'chi2_scale': 1.05, 'r_scale': 0.90},
                    '1': {'chi2_scale': 0.90, 'r_scale': 1.40},
                    '2': {'chi2_scale': 0.95, 'r_scale': 1.20},
                },
                'BIAS_GUARD': {
                    '0': {'r_scale': 0.95},
                    '1': {'r_scale': 1.05},
                    '2': {'r_scale': 1.00},
                },
            },
            'zupt_fail_soft': {
                'enabled': True,
                'hard_reject_factor': 4.5,
                'max_r_scale': 40.0,
                'inflate_power': 1.0,
                'health_hard_factor': {
                    'HEALTHY': 1.0,
                    'WARNING': 1.15,
                    'DEGRADED': 1.35,
                    'RECOVERY': 1.1,
                },
                'health_r_cap_factor': {
                    'HEALTHY': 1.0,
                    'WARNING': 1.2,
                    'DEGRADED': 1.4,
                    'RECOVERY': 1.1,
                },
            },
            'vio_vel_fail_soft': {
                'enabled': True,
                'hard_reject_factor': 3.0,
                'max_r_scale': 12.0,
                'inflate_power': 1.0,
                'health_hard_factor': {
                    'HEALTHY': 1.0,
                    'WARNING': 1.15,
                    'DEGRADED': 1.30,
                    'RECOVERY': 1.10,
                },
                'health_r_cap_factor': {
                    'HEALTHY': 1.0,
                    'WARNING': 1.15,
                    'DEGRADED': 1.30,
                    'RECOVERY': 1.10,
                },
            },
            'gravity_alignment': {
                'enabled_imu_only': True,
                'phase_sigma_deg': {'0': 3.5, '1': 10.0, '2': 8.5},
                'phase_period_steps': {'0': 1, '1': 2, '2': 2},
                'phase_acc_norm_tolerance': {'0': 0.15, '1': 0.35, '2': 0.30},
                'phase_max_gyro_rad_s': {'0': 0.25, '1': 0.65, '2': 0.50},
                'health_sigma_mult': {
                    'HEALTHY': 1.0,
                    'WARNING': 1.2,
                    'DEGRADED': 1.5,
                    'RECOVERY': 1.1,
                },
                'acc_norm_tolerance': 0.25,
                'max_gyro_rad_s': 0.40,
                'chi2_scale': 1.0,
            },
            'yaw_aid': {
                'enabled_imu_only': True,
                'phase_sigma_deg': {'0': 20.0, '1': 50.0, '2': 38.0},
                'phase_min_sigma_deg': {'0': 18.0, '1': 35.0, '2': 28.0},
                'phase_period_steps': {'0': 4, '1': 24, '2': 16},
                'max_accept_rate_for_active': 0.98,
                'high_accept_backoff_factor': 1.5,
                'high_accept_r_scale': 1.1,
                'phase_acc_norm_tolerance': {'0': 0.15, '1': 0.30, '2': 0.25},
                'phase_max_gyro_rad_s': {'0': 0.20, '1': 0.55, '2': 0.40},
                'health_sigma_mult': {
                    'HEALTHY': 1.0,
                    'WARNING': 1.2,
                    'DEGRADED': 1.5,
                    'RECOVERY': 1.1,
                },
                'high_speed_m_s': 180.0,
                'high_speed_sigma_mult': 1.15,
                'high_speed_period_mult': 1.2,
                'motion_consistency_enable': True,
                'motion_min_speed_m_s': 20.0,
                'motion_speed_full_m_s': 90.0,
                'motion_weight_max': 0.18,
                'motion_max_yaw_error_deg': 70.0,
                'chi2_scale': 1.0,
                'ref_alpha': 0.005,
                'dynamic_ref_alpha': 0.05,
                'soft_fail': {
                    'enabled': True,
                    'hard_reject_factor': 3.0,
                    'max_r_scale': 12.0,
                    'inflate_power': 1.0,
                },
            },
            'bias_guard': {
                'enabled_imu_only': True,
                'apply_when_aiding_level': ['NONE'],
                'period_steps': 8,
                'phase_period_steps': {'0': 8, '1': 24, '2': 20},
                'max_accept_rate_for_active': 0.98,
                'high_accept_backoff_factor': 1.6,
                'high_accept_r_scale': 1.05,
                'phase_sigma_bg_deg_s': {'0': 0.25, '1': 0.45, '2': 0.35},
                'phase_sigma_ba_m_s2': {'0': 0.08, '1': 0.25, '2': 0.18},
                'phase_acc_norm_tolerance': {'0': 0.20, '1': 0.20, '2': 0.22},
                'phase_max_gyro_rad_s': {'0': 0.20, '1': 0.30, '2': 0.32},
                'high_speed_m_s': 180.0,
                'high_speed_period_mult': 1.2,
                'health_sigma_mult': {
                    'HEALTHY': 1.0,
                    'WARNING': 0.8,
                    'DEGRADED': 0.6,
                    'RECOVERY': 0.9,
                },
                'chi2_scale': 1.0,
                'max_bg_norm_rad_s': 0.20,
                'max_ba_norm_m_s2': 2.5,
                'soft_fail': {
                    'enabled': True,
                    'hard_reject_factor': 4.0,
                    'max_r_scale': 8.0,
                    'inflate_power': 1.0,
                },
            },
            'conditioning_backoff': {
                'HEALTHY': {'period_mult': 1.0, 'r_scale_mult': 1.0},
                'WARNING': {'period_mult': 1.2, 'r_scale_mult': 1.05},
                'DEGRADED': {'period_mult': 1.5, 'r_scale_mult': 1.15},
                'RECOVERY': {'period_mult': 1.1, 'r_scale_mult': 1.02},
            },
        },
        'conditioning': {
            'caps': {
                'HEALTHY': 1e8,
                'WARNING': 1e7,
                'DEGRADED': 1e6,
                'RECOVERY': 1e7,
            },
            'cond_hard': {
                'HEALTHY': 1.2e12,
                'WARNING': 1.2e12,
                'DEGRADED': 1.1e12,
                'RECOVERY': 1.2e12,
            },
            'cond_hard_window': {
                'HEALTHY': 16,
                'WARNING': 16,
                'DEGRADED': 16,
                'RECOVERY': 16,
            },
            'projection_min_interval_steps': {
                'HEALTHY': 64,
                'WARNING': 64,
                'DEGRADED': 64,
                'RECOVERY': 64,
            },
        },
        'logging': {
            'enabled': True,
            # Sensor-health CSV downsampling:
            # - debug tier (with --save_debug_data): keep every row
            # - light tier (without --save_debug_data): keep every Nth accepted row
            'sensor_health_stride_debug': 1,
            'sensor_health_stride_release': 10,
        },
    }
    result['ADAPTIVE'] = _merge_dict_defaults(adaptive_defaults, config.get('adaptive', {}))

    # =========================================================================
    # Magnetometer accuracy policy (quality-aware, weak fail-soft)
    # =========================================================================
    mag_accuracy_defaults = {
        'enabled': True,
        'good_min_score': 0.72,
        'mid_min_score': 0.45,
        'r_inflate_mid': 1.8,
        'r_inflate_bad': 3.0,
        'norm_window': 200,
        'norm_dev_good': 0.12,
        'norm_dev_bad': 0.30,
        'gyro_delta_soft_deg': 20.0,
        'gyro_delta_hard_deg': 65.0,
        'vision_delta_soft_deg': 30.0,
        'vision_delta_hard_deg': 90.0,
        'vision_weight': 0.25,
        'skip_on_bad': True,
    }
    result['MAG_ACCURACY_POLICY'] = _merge_dict_defaults(
        mag_accuracy_defaults, mag.get('accuracy_policy', {})
    )
    _mag_acc = result['MAG_ACCURACY_POLICY']
    result['MAG_ACCURACY_ENABLED'] = bool(_mag_acc.get('enabled', True))
    result['MAG_ACCURACY_GOOD_MIN_SCORE'] = float(_mag_acc.get('good_min_score', 0.72))
    result['MAG_ACCURACY_MID_MIN_SCORE'] = float(_mag_acc.get('mid_min_score', 0.45))
    result['MAG_ACCURACY_R_INFLATE_MID'] = float(_mag_acc.get('r_inflate_mid', 1.8))
    result['MAG_ACCURACY_R_INFLATE_BAD'] = float(_mag_acc.get('r_inflate_bad', 3.0))
    result['MAG_ACCURACY_NORM_WINDOW'] = int(_mag_acc.get('norm_window', 200))
    result['MAG_ACCURACY_NORM_DEV_GOOD'] = float(_mag_acc.get('norm_dev_good', 0.12))
    result['MAG_ACCURACY_NORM_DEV_BAD'] = float(_mag_acc.get('norm_dev_bad', 0.30))
    result['MAG_ACCURACY_GYRO_DELTA_SOFT_DEG'] = float(_mag_acc.get('gyro_delta_soft_deg', 20.0))
    result['MAG_ACCURACY_GYRO_DELTA_HARD_DEG'] = float(_mag_acc.get('gyro_delta_hard_deg', 65.0))
    result['MAG_ACCURACY_VISION_DELTA_SOFT_DEG'] = float(_mag_acc.get('vision_delta_soft_deg', 30.0))
    result['MAG_ACCURACY_VISION_DELTA_HARD_DEG'] = float(_mag_acc.get('vision_delta_hard_deg', 90.0))
    result['MAG_ACCURACY_VISION_WEIGHT'] = float(_mag_acc.get('vision_weight', 0.25))
    result['MAG_ACCURACY_SKIP_ON_BAD'] = bool(_mag_acc.get('skip_on_bad', True))

    # Compiler metadata (for debug/traceability)
    result['CONFIG_COMPILE_META'] = {
        'source': os.path.abspath(config_path),
        'estimator_mode': result['ESTIMATOR_MODE'],
        'accel_includes_gravity': bool(result['IMU_PARAMS']['accel_includes_gravity']),
        'adaptive_mode': result['ADAPTIVE']['mode'],
        'objective_mode': result['OBJECTIVE_MODE'],
    }
    
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

# Adaptive policy defaults (compiled from YAML adaptive section)
ADAPTIVE = {}

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
