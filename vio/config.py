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


def _as_float_list(value: Any, default: Tuple[float, ...]) -> list:
    """Best-effort parse list/tuple into list[float] with safe fallback."""
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            try:
                out.append(float(v))
            except Exception:
                continue
        if len(out) > 0:
            return out
    return [float(v) for v in default]


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
    mag_cal = mag.get('calibration', {}) if isinstance(mag.get('calibration', {}), dict) else {}
    hard_iron = mag_cal.get('hard_iron', mag.get('hard_iron_offset', [0.0, 0.0, 0.0]))
    soft_iron = mag_cal.get(
        'soft_iron',
        mag.get('soft_iron_matrix', np.eye(3, dtype=float).tolist()),
    )
    result['MAG_HARD_IRON_OFFSET'] = np.array(hard_iron, dtype=float)
    result['MAG_SOFT_IRON_MATRIX'] = np.array(soft_iron, dtype=float)
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
    result['MAG_CONDITIONING_GUARD_EXTREME_PCOND'] = float(
        warning_weak_yaw.get(
            'conditioning_guard_extreme_pcond',
            8.0 * float(warning_weak_yaw.get('conditioning_guard_hard_pcond', 1e12)),
        )
    )
    result['MAG_CONDITIONING_GUARD_EXTREME_PMAX'] = float(
        warning_weak_yaw.get(
            'conditioning_guard_extreme_pmax',
            4.0 * float(warning_weak_yaw.get('conditioning_guard_hard_pmax', 1e7)),
        )
    )
    result['MAG_CONDITIONING_GUARD_EXTREME_SOFT_ENABLE'] = bool(
        warning_weak_yaw.get('conditioning_guard_extreme_soft_enable', True)
    )
    result['MAG_CONDITIONING_GUARD_EXTREME_SOFT_R_MULT'] = float(
        warning_weak_yaw.get('conditioning_guard_extreme_soft_r_mult', 2.0)
    )
    result['MAG_CONDITIONING_GUARD_SOFT_ENABLE'] = bool(
        warning_weak_yaw.get('conditioning_guard_soft_enable', True)
    )
    result['MAG_CONDITIONING_GUARD_SOFT_R_MULT'] = float(
        warning_weak_yaw.get('conditioning_guard_soft_r_mult', 4.0)
    )
    result['MAG_CONDITIONING_GUARD_SOFT_R_MULT_WARNING'] = float(
        warning_weak_yaw.get('conditioning_guard_soft_r_mult_warning', 1.0)
    )
    result['MAG_CONDITIONING_GUARD_SOFT_R_MULT_DEGRADED'] = float(
        warning_weak_yaw.get('conditioning_guard_soft_r_mult_degraded', 1.3)
    )
    result['MAG_CONDITIONING_GUARD_SOFT_R_MULT_RECOVERY'] = float(
        warning_weak_yaw.get('conditioning_guard_soft_r_mult_recovery', 1.0)
    )
    result['MAG_CONDITIONING_GUARD_SOFT_R_MULT_HEALTHY'] = float(
        warning_weak_yaw.get('conditioning_guard_soft_r_mult_healthy', 1.6)
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
    result['MAG_CHOL_COOLDOWN_SEC'] = float(
        warning_weak_yaw.get('chol_cooldown_sec', 0.30)
    )
    result['MAG_CHOL_COOLDOWN_STREAK_MULT'] = float(
        warning_weak_yaw.get('chol_cooldown_streak_mult', 0.25)
    )
    result['MAG_CHOL_COOLDOWN_MAX_SEC'] = float(
        warning_weak_yaw.get('chol_cooldown_max_sec', 2.0)
    )
    mag_ablation = mag.get('ablation', {})
    result['MAG_MODE'] = str(mag_ablation.get('mode', 'normal')).lower()
    result['MAG_ABLATION_R_MULT'] = float(mag_ablation.get('r_mult', 1.0))
    result['MAG_ABLATION_MAX_DYAW_DEG'] = float(
        mag_ablation.get('max_dyaw_deg', 180.0)
    )
    result['MAG_ABLATION_MAX_UPDATE_DYAW_DEG'] = float(
        mag_ablation.get('max_update_dyaw_deg', 180.0)
    )
    result['MAG_ABLATION_WEAK_SKIP_HARD_MISMATCH'] = bool(
        mag_ablation.get('weak_skip_hard_mismatch', False)
    )
    heading_arb = mag.get('heading_arbitration', {})
    result['MAG_HEADING_ARB_ENABLE'] = bool(heading_arb.get('enable', False))
    result['MAG_HEADING_ARB_HARD_MISMATCH_DEG'] = float(
        heading_arb.get('hard_mismatch_deg', 95.0)
    )
    result['MAG_HEADING_ARB_MIN_VISION_QUALITY'] = float(
        heading_arb.get('min_vision_quality', 0.55)
    )
    result['MAG_HEADING_ARB_STREAK_TO_HOLD'] = int(
        heading_arb.get('streak_to_hold', 3)
    )
    result['MAG_HEADING_ARB_HOLD_SEC'] = float(
        heading_arb.get('hold_sec', 1.8)
    )
    result['MAG_HEADING_ARB_SOFT_R_MULT'] = float(
        heading_arb.get('soft_r_mult', 2.0)
    )
    result['MAG_HEADING_ARB_MAX_VISION_AGE_SEC'] = float(
        heading_arb.get('max_vision_age_sec', 1.0)
    )
    result['MAG_HEADING_ARB_SCORE_EMA_ALPHA'] = float(
        heading_arb.get('score_ema_alpha', 0.20)
    )
    result['MAG_HEADING_ARB_SCORE_SOFT_THRESHOLD'] = float(
        heading_arb.get('score_soft_threshold', 0.55)
    )
    result['MAG_HEADING_ARB_SCORE_HOLD_THRESHOLD'] = float(
        heading_arb.get('score_hold_threshold', 0.30)
    )
    result['MAG_HEADING_ARB_SCORE_VIS_GOOD_DEG'] = float(
        heading_arb.get('score_vis_good_deg', 18.0)
    )
    result['MAG_HEADING_ARB_SCORE_GYRO_GOOD_DEG'] = float(
        heading_arb.get('score_gyro_good_deg', 10.0)
    )
    result['MAG_HEADING_ARB_SCORE_GYRO_BAD_DEG'] = float(
        heading_arb.get('score_gyro_bad_deg', 50.0)
    )
    result['MAG_HEADING_ARB_SCORE_STATE_GOOD_DEG'] = float(
        heading_arb.get('score_state_good_deg', 25.0)
    )
    result['MAG_HEADING_ARB_SCORE_STATE_BAD_DEG'] = float(
        heading_arb.get('score_state_bad_deg', 110.0)
    )
    result['MAG_HEADING_ARB_SCORE_VIS_WEIGHT'] = float(
        heading_arb.get('score_vis_weight', 0.50)
    )
    result['MAG_HEADING_ARB_SCORE_GYRO_WEIGHT'] = float(
        heading_arb.get('score_gyro_weight', 0.30)
    )
    result['MAG_HEADING_ARB_SCORE_STATE_WEIGHT'] = float(
        heading_arb.get('score_state_weight', 0.20)
    )
    result['MAG_HEADING_ARB_YAW_BUDGET_WINDOW_SEC'] = float(
        heading_arb.get('yaw_budget_window_sec', 6.0)
    )
    result['MAG_HEADING_ARB_YAW_BUDGET_ABS_DEG'] = float(
        heading_arb.get('yaw_budget_abs_deg', 8.0)
    )
    result['MAG_HEADING_ARB_YAW_BUDGET_MIN_REMAINING_DEG'] = float(
        heading_arb.get('yaw_budget_min_remaining_deg', 0.6)
    )
    result['MAG_HEADING_ARB_RECOVER_CONFIRM_HITS'] = int(
        heading_arb.get('recover_confirm_hits', 3)
    )
    result['MAG_HEADING_ARB_RECOVER_MIN_SCORE'] = float(
        heading_arb.get('recover_min_score', 0.60)
    )
    result['MAG_HEADING_ARB_RECOVER_SOFT_R_MULT'] = float(
        heading_arb.get('recover_soft_r_mult', 1.6)
    )
    result['MAG_HEADING_ARB_RECOVER_MAX_UPDATE_DYAW_DEG'] = float(
        heading_arb.get('recover_max_update_dyaw_deg', 0.30)
    )
    mag_preproc = mag.get('preprocessing', {})
    mag_quality = mag.get('quality', {}) if isinstance(mag.get('quality', {}), dict) else {}
    result['MAG_PREPROC_ENABLE'] = bool(mag_preproc.get('enable', True))
    result['MAG_PREPROC_NORM_RANGE_ENABLE'] = bool(
        mag_preproc.get('norm_range_enable', True)
    )
    preproc_norm_dev = float(mag_preproc.get('rolling_norm_dev_max', 0.45))
    preproc_gyro_deg = float(mag_preproc.get('gyro_delta_max_deg', 95.0))
    preproc_vision_deg = float(mag_preproc.get('vision_delta_max_deg', 120.0))
    preproc_ewma_alpha = float(mag_preproc.get('ewma_alpha', 0.08))
    # Canonical source: quality.* (if present). preprocessing.* acts as legacy alias.
    q_norm_ewma = float(mag_quality.get('norm_ewma_alpha', preproc_ewma_alpha))
    q_norm_mad = float(mag_quality.get('norm_mad_thresh', preproc_norm_dev))
    q_gyro_consistency = float(mag_quality.get('gyro_consistency_deg_s', preproc_gyro_deg))
    q_vision_consistency = float(mag_quality.get('vision_consistency_deg', preproc_vision_deg))
    result['MAG_QUALITY_NORM_EWMA_ALPHA'] = q_norm_ewma
    result['MAG_QUALITY_NORM_MAD_THRESH'] = q_norm_mad
    result['MAG_QUALITY_GYRO_CONSISTENCY_DEG_S'] = q_gyro_consistency
    result['MAG_QUALITY_VISION_CONSISTENCY_DEG'] = q_vision_consistency
    # Runtime de-coupling: keep preproc keys as aliases to canonical quality values
    # so no dual-threshold disagreement can happen in policy consumers.
    result['MAG_PREPROC_NORM_DEV_MAX'] = q_norm_mad
    result['MAG_PREPROC_GYRO_DELTA_MAX_DEG'] = q_gyro_consistency
    result['MAG_PREPROC_VISION_DELTA_MAX_DEG'] = q_vision_consistency
    result['MAG_PREPROC_EWMA_ALPHA'] = q_norm_ewma
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

    # =========================================================================
    # Global Yaw Authority (single-owner yaw governance)
    # =========================================================================
    yaw_auth = config.get('yaw_authority', {})
    result['YAW_AUTH_ENABLE'] = bool(yaw_auth.get('enable', False))
    result['YAW_AUTH_STAGE'] = int(yaw_auth.get('activation_stage', 0))
    result['YAW_AUTH_SCORE_EMA_ALPHA'] = float(yaw_auth.get('score_ema_alpha', 0.22))
    result['YAW_AUTH_MIN_SOURCE_SCORE'] = float(yaw_auth.get('min_source_score', 0.35))
    result['YAW_AUTH_MIN_SOURCE_SCORE_MAP'] = yaw_auth.get(
        'min_source_score_map',
        {},
    )
    result['YAW_AUTH_SWITCH_MARGIN'] = float(yaw_auth.get('switch_margin', 0.12))
    result['YAW_AUTH_SWITCH_MIN_INTERVAL_SEC'] = float(
        yaw_auth.get('switch_min_interval_sec', 0.75)
    )
    result['YAW_AUTH_OWNER_MIN_DWELL_SEC'] = float(
        yaw_auth.get('owner_min_dwell_sec', 0.0)
    )
    result['YAW_AUTH_HOLD_SCORE_THRESHOLD'] = float(
        yaw_auth.get('hold_score_threshold', 0.18)
    )
    result['YAW_AUTH_HOLD_SEC'] = float(yaw_auth.get('hold_sec', 0.8))
    result['YAW_AUTH_HOLD_RECLAIM_SOURCES'] = yaw_auth.get(
        'hold_reclaim_sources',
        ['LOOP', 'BACKEND'],
    )
    result['YAW_AUTH_HOLD_RECLAIM_MIN_CONFIDENCE'] = float(
        yaw_auth.get('hold_reclaim_min_confidence', 0.20)
    )
    result['YAW_AUTH_HOLD_ESCAPE_MIN_SCORE'] = float(
        yaw_auth.get('hold_escape_min_score', 0.16)
    )
    result['YAW_AUTH_HOLD_ESCAPE_MIN_SEC'] = float(
        yaw_auth.get('hold_escape_min_sec', 0.35)
    )
    result['YAW_AUTH_HOLD_BYPASS_SOURCES'] = yaw_auth.get(
        'hold_bypass_sources',
        ['LOOP', 'BACKEND'],
    )
    result['YAW_AUTH_HOLD_BYPASS_MIN_CONFIDENCE'] = float(
        yaw_auth.get('hold_bypass_min_confidence', 0.55)
    )
    result['YAW_AUTH_HOLD_BYPASS_MIN_FRESHNESS'] = float(
        yaw_auth.get('hold_bypass_min_freshness', 0.20)
    )
    result['YAW_AUTH_HOLD_BYPASS_MIN_CONFIDENCE_MAP'] = yaw_auth.get(
        'hold_bypass_min_confidence_map',
        {},
    )
    result['YAW_AUTH_HOLD_MAG_ANCHOR_ENABLE'] = bool(
        yaw_auth.get('hold_mag_anchor_enable', True)
    )
    result['YAW_AUTH_HOLD_MAG_ANCHOR_MIN_CONFIDENCE'] = float(
        yaw_auth.get('hold_mag_anchor_min_confidence', 0.55)
    )
    result['YAW_AUTH_HOLD_MAG_ANCHOR_REQUIRE_ALT_QUIET'] = bool(
        yaw_auth.get('hold_mag_anchor_require_alt_quiet', True)
    )
    result['YAW_AUTH_HOLD_MAG_ANCHOR_ALT_WINDOW_SEC'] = float(
        yaw_auth.get('hold_mag_anchor_alt_window_sec', 8.0)
    )
    result['YAW_AUTH_HOLD_MAG_ANCHOR_MAX_ALT_APPLIES'] = int(
        yaw_auth.get('hold_mag_anchor_max_alt_applies', 0)
    )
    result['YAW_AUTH_HOLD_MAG_ANCHOR_R_MULT'] = float(
        yaw_auth.get('hold_mag_anchor_r_mult', 4.0)
    )
    result['YAW_AUTH_HOLD_MAG_ANCHOR_MAX_DYAW_DEG'] = float(
        yaw_auth.get('hold_mag_anchor_max_dyaw_deg', 0.20)
    )
    result['YAW_AUTH_MAG_WARMUP_MIN_SCORE'] = float(
        yaw_auth.get('mag_warmup_min_score', 0.55)
    )
    result['YAW_AUTH_OWNER_REQUIRE_APPLY_RECENT_ENABLE'] = bool(
        yaw_auth.get('owner_require_apply_recent_enable', True)
    )
    result['YAW_AUTH_OWNER_APPLY_RECENT_WINDOW_SEC'] = float(
        yaw_auth.get('owner_apply_recent_window_sec', 8.0)
    )
    result['YAW_AUTH_OWNER_MIN_APPLY_RECENT_MAP'] = yaw_auth.get(
        'owner_min_apply_recent_map',
        {'BACKEND': 2},
    )
    result['YAW_AUTH_BACKEND_BOOTSTRAP_MIN_REQUESTS'] = int(
        yaw_auth.get('backend_bootstrap_min_requests', 3)
    )
    result['YAW_AUTH_BACKEND_BOOTSTRAP_MIN_SCORE'] = float(
        yaw_auth.get('backend_bootstrap_min_score', 0.28)
    )
    result['YAW_AUTH_BACKEND_REQUEST_ALIVE_ENABLE'] = bool(
        yaw_auth.get('backend_request_alive_enable', True)
    )
    result['YAW_AUTH_BACKEND_REQUEST_ALIVE_WINDOW_SEC'] = float(
        yaw_auth.get('backend_request_alive_window_sec', yaw_auth.get('owner_apply_recent_window_sec', 8.0))
    )
    result['YAW_AUTH_BACKEND_REQUEST_ALIVE_MIN_REQUESTS'] = int(
        yaw_auth.get('backend_request_alive_min_requests', yaw_auth.get('backend_bootstrap_min_requests', 3))
    )
    result['YAW_AUTH_BACKEND_REQUEST_ALIVE_MIN_SCORE'] = float(
        yaw_auth.get('backend_request_alive_min_score', yaw_auth.get('backend_bootstrap_min_score', 0.28))
    )
    result['YAW_AUTH_LOOP_REQUEST_ALIVE_ENABLE'] = bool(
        yaw_auth.get('loop_request_alive_enable', True)
    )
    result['YAW_AUTH_LOOP_REQUEST_ALIVE_WINDOW_SEC'] = float(
        yaw_auth.get('loop_request_alive_window_sec', yaw_auth.get('owner_apply_recent_window_sec', 8.0))
    )
    result['YAW_AUTH_LOOP_REQUEST_ALIVE_MIN_REQUESTS'] = int(
        yaw_auth.get('loop_request_alive_min_requests', 2)
    )
    result['YAW_AUTH_LOOP_REQUEST_ALIVE_MIN_SCORE'] = float(
        yaw_auth.get('loop_request_alive_min_score', 0.24)
    )
    result['YAW_AUTH_REQUEST_ALIVE_WINDOW_SEC_MAP'] = yaw_auth.get(
        'request_alive_window_sec_map',
        {},
    )
    result['YAW_AUTH_REQUEST_ALIVE_MIN_REQUESTS_MAP'] = yaw_auth.get(
        'request_alive_min_requests_map',
        {},
    )
    result['YAW_AUTH_REQUEST_ALIVE_MIN_SCORE_MAP'] = yaw_auth.get(
        'request_alive_min_score_map',
        {},
    )
    result['YAW_AUTH_REQUEST_ALIVE_MIN_FRESHNESS_MAP'] = yaw_auth.get(
        'request_alive_min_freshness_map',
        {},
    )
    result['YAW_AUTH_YAW_BUDGET_WINDOW_SEC'] = float(
        yaw_auth.get('yaw_budget_window_sec', 8.0)
    )
    result['YAW_AUTH_YAW_BUDGET_ABS_DEG'] = float(
        yaw_auth.get('yaw_budget_abs_deg', 10.0)
    )
    result['YAW_AUTH_YAW_RATE_MAX_DEG_S'] = float(
        yaw_auth.get('yaw_rate_max_deg_s', 4.0)
    )
    result['YAW_AUTH_BUDGET_SOFT_R_MULT'] = float(
        yaw_auth.get('budget_soft_r_mult', 1.6)
    )
    # Whether MSCKF quality is allowed to update yaw-owner confidence scores.
    # Keep default True for backward compatibility; can be disabled per config
    # to make instrumentation-only phases behavior-neutral.
    result['YAW_AUTH_USE_MSCKF_CONFIDENCE'] = bool(
        yaw_auth.get('use_msckf_confidence', True)
    )
    result['YAW_AUTH_MSCKF_CONF_BLEND'] = yaw_auth.get(
        'msckf_conf_blend',
        {'MAG': 0.55, 'LOOP': 0.25, 'BACKEND': 0.20},
    )
    result['YAW_AUTH_SOFT_ONLY_HIGH_SPEED_M_S'] = float(
        yaw_auth.get('soft_only_high_speed_m_s', 22.0)
    )
    result['YAW_AUTH_SOFT_ONLY_UNSTABLE_PMAX'] = float(
        yaw_auth.get('soft_only_unstable_pmax', 1.0e6)
    )
    result['YAW_AUTH_SOFT_ONLY_UNSTABLE_PCOND'] = float(
        yaw_auth.get('soft_only_unstable_pcond', 1.0e12)
    )
    result['YAW_AUTH_SOFT_ONLY_UNSTABLE_HEALTH'] = yaw_auth.get(
        'soft_only_unstable_health',
        ['WARNING', 'DEGRADED'],
    )
    result['YAW_AUTH_SOFT_ONLY_MAX_DYAW_DEG'] = float(
        yaw_auth.get('soft_only_max_dyaw_deg', 1.2)
    )
    result['YAW_AUTH_SOFT_ONLY_R_MULT'] = float(
        yaw_auth.get('soft_only_r_mult', 1.5)
    )
    result['YAW_AUTH_SOURCE_PRIORITY'] = yaw_auth.get(
        'source_priority',
        {'MAG': 1.0, 'LOOP': 1.0, 'BACKEND': 0.9},
    )
    # Stage-1/2 controlled owner handoff (pre-stage3 confidence switching)
    result['YAW_AUTH_STAGE12_CLAIM_ENABLE'] = bool(
        yaw_auth.get('stage12_claim_enable', True)
    )
    result['YAW_AUTH_STAGE12_MIN_SWITCH_INTERVAL_SEC'] = float(
        yaw_auth.get('stage12_min_switch_interval_sec', yaw_auth.get('switch_min_interval_sec', 0.75))
    )
    result['YAW_AUTH_STAGE12_OWNER_TIMEOUT_SEC'] = float(
        yaw_auth.get('stage12_owner_timeout_sec', 1.2)
    )
    result['YAW_AUTH_STAGE12_CLAIM_MIN_SCORE'] = float(
        yaw_auth.get('stage12_claim_min_score', 0.45)
    )
    result['YAW_AUTH_STAGE12_CLAIM_MARGIN'] = float(
        yaw_auth.get('stage12_claim_margin', 0.05)
    )
    result['YAW_AUTH_STAGE12_LOOP_FORCE_CLAIM_MIN_SCORE'] = float(
        yaw_auth.get('stage12_loop_force_claim_min_score', 0.62)
    )
    result['YAW_AUTH_STAGE12_LOOP_FORCE_CLAIM_MAX_SPEED_M_S'] = float(
        yaw_auth.get('stage12_loop_force_claim_max_speed_m_s', 35.0)
    )
    result['YAW_AUTH_STAGE12_ALLOW_STALE_RECLAIM_ANY'] = bool(
        yaw_auth.get('stage12_allow_stale_reclaim_any', True)
    )
    result['YAW_AUTH_STAGE12_CLAIM_SOURCES'] = yaw_auth.get(
        'stage12_claim_sources',
        ['LOOP', 'BACKEND'],
    )
    result['YAW_AUTH_BACKEND_MIN_REQUEST_DYAW_DEG'] = float(
        yaw_auth.get('backend_min_request_dyaw_deg', 0.12)
    )
    result['YAW_AUTH_BACKEND_NOOP_MAX_SCORE'] = float(
        yaw_auth.get('backend_noop_max_score', 0.18)
    )
    result['YAW_AUTH_MAG_OWNER_MIN_ACCEPT_RATE'] = float(
        yaw_auth.get('mag_owner_min_accept_rate', 0.15)
    )
    result['YAW_AUTH_MAG_OWNER_ACCEPT_WINDOW_SEC'] = float(
        yaw_auth.get('mag_owner_accept_window_sec', 8.0)
    )
    result['YAW_AUTH_MAG_OWNER_MIN_SAMPLES'] = int(
        yaw_auth.get('mag_owner_min_samples', 20)
    )
    result['YAW_AUTH_MAG_OWNER_BAN_SEC'] = float(
        yaw_auth.get('mag_owner_ban_sec', 2.5)
    )
    result['YAW_AUTH_MAG_BLOCK_REQUIRE_ALT_SOURCE'] = bool(
        yaw_auth.get('mag_block_require_alt_source', False)
    )
    result['YAW_AUTH_MAG_BLOCK_MIN_ALT_APPLIES'] = int(
        yaw_auth.get('mag_block_min_alt_applies', 1)
    )
    result['YAW_AUTH_MAG_BLOCK_REQUIRE_ALT_WINDOW_SEC'] = float(
        yaw_auth.get('mag_block_require_alt_window_sec', yaw_auth.get('mag_owner_accept_window_sec', 8.0))
    )
    result['YAW_AUTH_MAG_BLOCK_FORCE_RATE'] = float(
        yaw_auth.get('mag_block_force_rate', 0.03)
    )
    result['YAW_AUTH_MAG_BLOCK_FORCE_BAN_SEC'] = float(
        yaw_auth.get('mag_block_force_ban_sec', 8.0)
    )
    result['YAW_AUTH_MAG_SCORE_ACCEPT_ENABLE'] = bool(
        yaw_auth.get('mag_score_accept_enable', True)
    )
    result['YAW_AUTH_MAG_SCORE_ACCEPT_WINDOW_SEC'] = float(
        yaw_auth.get('mag_score_accept_window_sec', yaw_auth.get('mag_owner_accept_window_sec', 8.0))
    )
    result['YAW_AUTH_MAG_SCORE_ACCEPT_REF_RATE'] = float(
        yaw_auth.get('mag_score_accept_ref_rate', 0.20)
    )
    result['YAW_AUTH_MAG_SCORE_MIN_SAMPLES'] = int(
        yaw_auth.get('mag_score_min_samples', yaw_auth.get('mag_owner_min_samples', 20))
    )
    result['YAW_AUTH_MAG_SCORE_FLOOR'] = float(
        yaw_auth.get('mag_score_floor', 0.15)
    )
    result['YAW_AUTH_MAG_SCORE_COLD_START_SCALE'] = float(
        yaw_auth.get('mag_score_cold_start_scale', 1.0)
    )
    result['YAW_AUTH_MAG_LOW_ACCEPT_SOFT_R_MULT'] = float(
        yaw_auth.get('mag_low_accept_soft_r_mult', 3.0)
    )
    result['YAW_AUTH_MAG_LOW_ACCEPT_MAX_DYAW_DEG'] = float(
        yaw_auth.get('mag_low_accept_max_dyaw_deg', 0.35)
    )
    result['YAW_AUTH_MAG_COUNT_OWNER_SKIP_AS_REJECT'] = bool(
        yaw_auth.get('mag_count_owner_skip_as_reject', False)
    )
    result['YAW_AUTH_POSITION_FIRST_DISABLE_MAG_OWNER'] = bool(
        yaw_auth.get('position_first_disable_mag_owner', True)
    )
    result['YAW_AUTH_OWNER_DEAD_ENABLE'] = bool(
        yaw_auth.get('owner_dead_enable', True)
    )
    result['YAW_AUTH_OWNER_DEAD_TIMEOUT_SEC'] = float(
        yaw_auth.get('owner_dead_timeout_sec', 2.5)
    )
    result['YAW_AUTH_OWNER_DEAD_TIMEOUT_SEC_MAP'] = yaw_auth.get(
        'owner_dead_timeout_sec_map',
        {},
    )
    result['YAW_AUTH_OWNER_DEAD_HOLD_SEC'] = float(
        yaw_auth.get('owner_dead_hold_sec', 0.35)
    )
    result['YAW_AUTH_OWNER_DEAD_REARM_SEC'] = float(
        yaw_auth.get('owner_dead_rearm_sec', 0.8)
    )
    result['YAW_AUTH_SOURCE_STALE_SEC'] = float(
        yaw_auth.get('source_stale_sec', yaw_auth.get('owner_dead_timeout_sec', 2.5))
    )
    result['YAW_AUTH_SOURCE_STALE_SEC_MAP'] = yaw_auth.get(
        'source_stale_sec_map',
        {},
    )
    result['YAW_AUTH_SOURCE_STALE_TAU_SEC'] = float(
        yaw_auth.get('source_stale_tau_sec', yaw_auth.get('source_stale_sec', yaw_auth.get('owner_dead_timeout_sec', 2.5)))
    )
    result['YAW_AUTH_SOURCE_STALE_TAU_SEC_MAP'] = yaw_auth.get(
        'source_stale_tau_sec_map',
        {},
    )
    result['YAW_AUTH_OWNER_DEAD_MIN_REQUESTS_MAP'] = yaw_auth.get(
        'owner_dead_min_requests_map',
        {'BACKEND': 2, 'LOOP': 1},
    )
    result['YAW_AUTH_OWNER_DEAD_MIN_FRESHNESS'] = float(
        yaw_auth.get('owner_dead_min_freshness', 0.20)
    )
    result['YAW_AUTH_MIN_EFFECTIVE_SCORE'] = float(
        yaw_auth.get('min_effective_score', 0.06)
    )
    result['YAW_AUTH_OWNER_DEAD_RECLAIM_MIN_SCORE'] = float(
        yaw_auth.get('owner_dead_reclaim_min_score', max(yaw_auth.get('min_source_score', 0.35), 0.45))
    )
    result['YAW_AUTH_OWNER_DEAD_RECLAIM_MIN_FRESHNESS'] = float(
        yaw_auth.get('owner_dead_reclaim_min_freshness', 0.35)
    )
    result['YAW_AUTH_ACTIVITY_ENABLE'] = bool(
        yaw_auth.get('activity_enable', True)
    )
    result['YAW_AUTH_ACTIVITY_WINDOW_SEC'] = float(
        yaw_auth.get('activity_window_sec', 8.0)
    )
    result['YAW_AUTH_ACTIVITY_MIN_SAMPLES'] = int(
        yaw_auth.get('activity_min_samples', 8)
    )
    result['YAW_AUTH_ACTIVITY_RATIO_FLOOR'] = float(
        yaw_auth.get('activity_ratio_floor', 0.25)
    )
    result['YAW_AUTH_SWITCH_EVAL_MIN_INTERVAL_SEC'] = float(
        yaw_auth.get('switch_eval_min_interval_sec', 0.25)
    )
    result['YAW_AUTH_STRICT_SINGLE_PATH_ENABLE'] = bool(
        yaw_auth.get('strict_single_path_enable', True)
    )
    result['YAW_AUTH_STRICT_HOLD_RECLAIM_SOURCES'] = yaw_auth.get(
        'strict_hold_reclaim_sources',
        ['BACKEND', 'LOOP'],
    )
    result['YAW_AUTH_STRICT_HOLD_RECLAIM_MIN_CONFIDENCE'] = float(
        yaw_auth.get('strict_hold_reclaim_min_confidence', 0.22)
    )
    result['YAW_AUTH_STRICT_HOLD_RECLAIM_MIN_CONFIDENCE_MAP'] = yaw_auth.get(
        'strict_hold_reclaim_min_confidence_map',
        {},
    )
    result['YAW_AUTH_STRICT_HOLD_RECLAIM_MIN_FRESHNESS'] = float(
        yaw_auth.get('strict_hold_reclaim_min_freshness', 0.15)
    )
    result['YAW_AUTH_LOOP_RECLAIM_ENABLE'] = bool(
        yaw_auth.get('loop_reclaim_enable', False)
    )
    result['YAW_AUTH_LOOP_RECLAIM_ALLOW_FROM_HOLD'] = bool(
        yaw_auth.get('loop_reclaim_allow_from_hold', False)
    )
    result['YAW_AUTH_LOOP_RECLAIM_HOLD_REQUIRE_BACKEND_DEAD'] = bool(
        yaw_auth.get('loop_reclaim_hold_require_backend_dead', True)
    )
    result['YAW_AUTH_LOOP_RECLAIM_HOLD_ALLOW_NO_DEAD'] = bool(
        yaw_auth.get('loop_reclaim_hold_allow_no_dead', False)
    )
    result['YAW_AUTH_LOOP_RECLAIM_HOLD_MIN_SEC'] = float(
        yaw_auth.get('loop_reclaim_hold_min_sec', 0.35)
    )
    result['YAW_AUTH_LOOP_RECLAIM_HOLD_COOLDOWN_SEC'] = float(
        yaw_auth.get('loop_reclaim_hold_cooldown_sec', 1.5)
    )
    result['YAW_AUTH_LOOP_RECLAIM_BYPASS_MIN_INTERVAL_ON_BACKEND_DEAD'] = bool(
        yaw_auth.get('loop_reclaim_bypass_min_interval_on_backend_dead', True)
    )
    result['YAW_AUTH_LOOP_RECLAIM_MIN_INTERVAL_SEC'] = float(
        yaw_auth.get('loop_reclaim_min_interval_sec', yaw_auth.get('switch_min_interval_sec', 0.75))
    )
    result['YAW_AUTH_LOOP_RECLAIM_WINDOW_SEC'] = float(
        yaw_auth.get('loop_reclaim_window_sec', 6.0)
    )
    result['YAW_AUTH_LOOP_RECLAIM_MIN_REQUESTS'] = int(
        yaw_auth.get('loop_reclaim_min_requests', 3)
    )
    result['YAW_AUTH_LOOP_RECLAIM_MIN_REQUESTS_ON_BACKEND_DEAD'] = int(
        yaw_auth.get('loop_reclaim_min_requests_on_backend_dead', 1)
    )
    result['YAW_AUTH_LOOP_RECLAIM_MIN_SCORE'] = float(
        yaw_auth.get('loop_reclaim_min_score', 0.62)
    )
    result['YAW_AUTH_LOOP_RECLAIM_MIN_SCORE_ON_BACKEND_DEAD'] = float(
        yaw_auth.get('loop_reclaim_min_score_on_backend_dead', yaw_auth.get('loop_reclaim_min_score', 0.62))
    )
    result['YAW_AUTH_LOOP_RECLAIM_MARGIN'] = float(
        yaw_auth.get('loop_reclaim_margin', 0.05)
    )
    result['YAW_AUTH_LOOP_RECLAIM_MARGIN_ON_BACKEND_DEAD'] = float(
        yaw_auth.get('loop_reclaim_margin_on_backend_dead', yaw_auth.get('loop_reclaim_margin', 0.05))
    )
    result['YAW_AUTH_LOOP_RECLAIM_MIN_FRESHNESS'] = float(
        yaw_auth.get('loop_reclaim_min_freshness', 0.10)
    )
    result['YAW_AUTH_LOOP_RECLAIM_MAX_SPEED_M_S'] = float(
        yaw_auth.get('loop_reclaim_max_speed_m_s', 120.0)
    )
    result['YAW_AUTH_LOOP_RECLAIM_MIN_APPLY_EFF'] = float(
        yaw_auth.get('loop_reclaim_min_apply_eff', 0.0)
    )
    result['YAW_AUTH_LOOP_RECLAIM_MIN_APPLY_SAMPLES'] = int(
        yaw_auth.get('loop_reclaim_min_apply_samples', 0)
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
    fe_cfg = vio.get('frontend', {}) if isinstance(vio.get('frontend', {}), dict) else {}
    result['VIO_FRONTEND_MAX_TRACK_HISTORY'] = int(fe_cfg.get('max_track_history', 90))
    result['VIO_FRONTEND_MAX_TOTAL_TRACKS'] = int(fe_cfg.get('max_total_tracks', 5000))
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
    result['MSCKF_REPROJ_FAILSOFT_ENABLE'] = bool(reproj_state_aware.get('failsoft_enable', True))
    result['MSCKF_REPROJ_FAILSOFT_MAX_MULT'] = float(reproj_state_aware.get('failsoft_max_mult', 1.35))
    result['MSCKF_REPROJ_FAILSOFT_MIN_OBS'] = int(reproj_state_aware.get('failsoft_min_obs', 3))
    result['MSCKF_REPROJ_FAILSOFT_MIN_QUALITY'] = float(
        reproj_state_aware.get('failsoft_min_quality', 0.35)
    )
    msckf_quality_gate = msckf_cfg.get('quality_gate', {}) if isinstance(msckf_cfg.get('quality_gate', {}), dict) else {}
    result['MSCKF_QUALITY_GATE_TRACK_MIN'] = int(msckf_quality_gate.get('track_min', 10))
    result['MSCKF_QUALITY_GATE_INLIER_MIN'] = float(msckf_quality_gate.get('inlier_min', 0.30))
    result['MSCKF_QUALITY_GATE_PARALLAX_MIN_PX'] = float(msckf_quality_gate.get('parallax_min_px', 1.2))
    result['MSCKF_QUALITY_GATE_DEPTH_POSITIVE_MIN'] = float(
        msckf_quality_gate.get('depth_positive_min', 0.62)
    )
    result['MSCKF_QUALITY_GATE_REPROJ_P95_MAX'] = float(
        msckf_quality_gate.get('reproj_p95_max', 0.06)
    )
    result['MSCKF_PHASE_RAY_SOFT_SCALE'] = msckf_cfg.get(
        'phase_ray_soft_scale',
        {'0': 1.12, '1': 1.06, '2': 1.00}
    )
    result['MSCKF_HEALTH_RAY_SOFT_SCALE'] = msckf_cfg.get(
        'health_ray_soft_scale',
        {'HEALTHY': 1.0, 'WARNING': 1.10, 'DEGRADED': 1.20, 'RECOVERY': 1.04}
    )
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
    result['VIO_VEL_DELTA_V_CLAMP_ENABLE'] = bool(
        vio_vel_cfg.get('delta_v_clamp_enable', True)
    )
    result['VIO_VEL_DELTA_V_CLAMP_MAX_RATIO'] = float(
        vio_vel_cfg.get('delta_v_clamp_max_ratio', 6.0)
    )
    result['VIO_VEL_DELTA_V_CLAMP_R_MULT'] = float(
        vio_vel_cfg.get('delta_v_clamp_r_mult', 3.5)
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
        result['LOOP_MAX_KEYFRAMES'] = int(lc.get('max_keyframes', 220))
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
        speed_gate = lc.get('speed_gate', {})
        speed_gate_normal = speed_gate.get('normal', {})
        speed_gate_failsoft = speed_gate.get('fail_soft', speed_gate.get('failsoft', {}))
        result['LOOP_SPEED_SKIP_M_S_NORMAL'] = float(
            speed_gate_normal.get('speed_skip_m_s', fs.get('speed_skip_m_s_normal', fs.get('speed_skip_m_s', 35.0)))
        )
        result['LOOP_SPEED_SIGMA_INFLATE_M_S_NORMAL'] = float(
            speed_gate_normal.get(
                'speed_sigma_inflate_m_s',
                fs.get('speed_sigma_inflate_m_s_normal', fs.get('speed_sigma_inflate_m_s', 25.0)),
            )
        )
        result['LOOP_SPEED_SIGMA_MULT_NORMAL'] = float(
            speed_gate_normal.get('speed_sigma_mult', fs.get('speed_sigma_mult_normal', fs.get('speed_sigma_mult', 1.5)))
        )
        result['LOOP_SPEED_SKIP_M_S_FAILSOFT'] = float(
            speed_gate_failsoft.get('speed_skip_m_s', fs.get('speed_skip_m_s_failsoft', fs.get('speed_skip_m_s', 35.0)))
        )
        result['LOOP_SPEED_SIGMA_INFLATE_M_S_FAILSOFT'] = float(
            speed_gate_failsoft.get(
                'speed_sigma_inflate_m_s',
                fs.get('speed_sigma_inflate_m_s_failsoft', fs.get('speed_sigma_inflate_m_s', 25.0)),
            )
        )
        result['LOOP_SPEED_SIGMA_MULT_FAILSOFT'] = float(
            speed_gate_failsoft.get('speed_sigma_mult', fs.get('speed_sigma_mult_failsoft', fs.get('speed_sigma_mult', 1.5)))
        )
        speed_yaw_cap = lc.get('speed_yaw_cap', {})
        result['LOOP_SPEED_YAW_CAP_BREAKPOINTS_M_S'] = _as_float_list(
            speed_yaw_cap.get('breakpoints_m_s', [20.0, 35.0, 50.0]),
            (20.0, 35.0, 50.0),
        )
        result['LOOP_SPEED_YAW_CAP_NORMAL_DEG'] = _as_float_list(
            speed_yaw_cap.get('normal_cap_deg', [3.0, 2.2, 1.5]),
            (3.0, 2.2, 1.5),
        )
        result['LOOP_SPEED_YAW_CAP_FAILSOFT_DEG'] = _as_float_list(
            speed_yaw_cap.get('fail_soft_cap_deg', speed_yaw_cap.get('failsoft_cap_deg', [2.5, 1.8, 1.2])),
            (2.5, 1.8, 1.2),
        )
        temporal_apply = lc.get('temporal_apply', {})
        result['LOOP_APPLY_CONFIRM_ENABLE'] = bool(temporal_apply.get('enable', True))
        result['LOOP_APPLY_CONFIRM_WINDOW_SEC'] = float(temporal_apply.get('confirm_window_sec', 3.0))
        result['LOOP_APPLY_CONFIRM_YAW_DEG'] = float(temporal_apply.get('confirm_yaw_deg', 6.0))
        result['LOOP_APPLY_CONFIRM_HITS_NORMAL'] = int(temporal_apply.get('confirm_hits_normal', 1))
        result['LOOP_APPLY_CONFIRM_HITS_FAILSOFT'] = int(temporal_apply.get('confirm_hits_failsoft', 2))
        result['LOOP_APPLY_CONFIRM_SPEED_M_S'] = float(temporal_apply.get('confirm_speed_m_s', 25.0))
        result['LOOP_APPLY_CONFIRM_EXTRA_HITS_HIGH_SPEED'] = int(
            temporal_apply.get('confirm_extra_hits_high_speed', 1)
        )
        result['LOOP_APPLY_CONFIRM_PHASE_DYNAMIC_MIN_HITS'] = int(
            temporal_apply.get('phase_dynamic_min_hits', 2)
        )
        result['LOOP_COOLDOWN_SEC_NORMAL'] = float(
            temporal_apply.get('cooldown_sec_normal', qg.get('cooldown_sec', 2.0))
        )
        result['LOOP_COOLDOWN_SEC_FAILSOFT'] = float(
            temporal_apply.get('cooldown_sec_failsoft', max(qg.get('cooldown_sec', 2.0), 3.0))
        )
        result['LOOP_COOLDOWN_SPEED_M_S'] = float(
            temporal_apply.get(
                'cooldown_speed_m_s',
                max(
                    fs.get('speed_sigma_inflate_m_s', 25.0),
                    speed_gate_normal.get('speed_sigma_inflate_m_s', fs.get('speed_sigma_inflate_m_s', 25.0)),
                ),
            )
        )
        result['LOOP_COOLDOWN_SPEED_MULT'] = float(temporal_apply.get('cooldown_speed_mult', 1.35))
        result['LOOP_BURST_WINDOW_SEC'] = float(temporal_apply.get('burst_window_sec', 12.0))
        result['LOOP_BURST_MAX_CORRECTIONS'] = int(temporal_apply.get('burst_max_corrections', 2))
        result['LOOP_BURST_COOLDOWN_SEC'] = float(temporal_apply.get('burst_cooldown_sec', 6.0))
        result['LOOP_BURST_ABS_YAW_BUDGET_DEG'] = float(
            temporal_apply.get('burst_abs_yaw_budget_deg', 8.0)
        )
        result['LOOP_BURST_RATE_WINDOW_SEC'] = float(
            temporal_apply.get('burst_rate_window_sec', 1.0)
        )
        result['LOOP_BURST_RATE_MAX_DEG_PER_SEC'] = float(
            temporal_apply.get('burst_rate_max_deg_per_sec', 2.4)
        )
        result['LOOP_BURST_MIN_APPLY_DEG'] = float(
            temporal_apply.get('burst_min_apply_deg', 0.18)
        )
        result['LOOP_HIGH_SPEED_SOFT_APPLY_ENABLE'] = bool(
            temporal_apply.get('high_speed_soft_apply_enable', True)
        )
        result['LOOP_HIGH_SPEED_SOFT_CAP_DEG'] = float(
            temporal_apply.get('high_speed_soft_cap_deg', 0.9)
        )
        result['LOOP_HIGH_SPEED_SOFT_R_MULT'] = float(
            temporal_apply.get('high_speed_soft_r_mult', 3.0)
        )
        result['LOOP_HIGH_SPEED_SOFT_EXTRA_CONFIRM_HITS'] = int(
            temporal_apply.get('high_speed_soft_extra_confirm_hits', 1)
        )
    else:
        result['USE_LOOP_CLOSURE'] = True
        result['LOOP_POSITION_THRESHOLD'] = 30.0
        result['LOOP_MIN_KEYFRAME_DIST'] = 15.0
        result['LOOP_MIN_KEYFRAME_YAW'] = 20.0
        result['LOOP_MIN_FRAME_GAP'] = 50
        result['LOOP_MAX_KEYFRAMES'] = 220
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
        result['LOOP_SPEED_SKIP_M_S_NORMAL'] = 35.0
        result['LOOP_SPEED_SIGMA_INFLATE_M_S_NORMAL'] = 25.0
        result['LOOP_SPEED_SIGMA_MULT_NORMAL'] = 1.5
        result['LOOP_SPEED_SKIP_M_S_FAILSOFT'] = 35.0
        result['LOOP_SPEED_SIGMA_INFLATE_M_S_FAILSOFT'] = 25.0
        result['LOOP_SPEED_SIGMA_MULT_FAILSOFT'] = 1.5
        result['LOOP_SPEED_YAW_CAP_BREAKPOINTS_M_S'] = [20.0, 35.0, 50.0]
        result['LOOP_SPEED_YAW_CAP_NORMAL_DEG'] = [3.0, 2.2, 1.5]
        result['LOOP_SPEED_YAW_CAP_FAILSOFT_DEG'] = [2.5, 1.8, 1.2]
        result['LOOP_APPLY_CONFIRM_ENABLE'] = True
        result['LOOP_APPLY_CONFIRM_WINDOW_SEC'] = 3.0
        result['LOOP_APPLY_CONFIRM_YAW_DEG'] = 6.0
        result['LOOP_APPLY_CONFIRM_HITS_NORMAL'] = 1
        result['LOOP_APPLY_CONFIRM_HITS_FAILSOFT'] = 2
        result['LOOP_APPLY_CONFIRM_SPEED_M_S'] = 25.0
        result['LOOP_APPLY_CONFIRM_EXTRA_HITS_HIGH_SPEED'] = 1
        result['LOOP_APPLY_CONFIRM_PHASE_DYNAMIC_MIN_HITS'] = 2
        result['LOOP_COOLDOWN_SEC_NORMAL'] = 2.0
        result['LOOP_COOLDOWN_SEC_FAILSOFT'] = 3.0
        result['LOOP_COOLDOWN_SPEED_M_S'] = 25.0
        result['LOOP_COOLDOWN_SPEED_MULT'] = 1.35
        result['LOOP_BURST_WINDOW_SEC'] = 12.0
        result['LOOP_BURST_MAX_CORRECTIONS'] = 2
        result['LOOP_BURST_COOLDOWN_SEC'] = 6.0
        result['LOOP_BURST_ABS_YAW_BUDGET_DEG'] = 8.0
        result['LOOP_BURST_RATE_WINDOW_SEC'] = 1.0
        result['LOOP_BURST_RATE_MAX_DEG_PER_SEC'] = 2.4
        result['LOOP_BURST_MIN_APPLY_DEG'] = 0.18
        result['LOOP_HIGH_SPEED_SOFT_APPLY_ENABLE'] = True
        result['LOOP_HIGH_SPEED_SOFT_CAP_DEG'] = 0.9
        result['LOOP_HIGH_SPEED_SOFT_R_MULT'] = 3.0
        result['LOOP_HIGH_SPEED_SOFT_EXTRA_CONFIRM_HITS'] = 1

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
    result['KIN_GUARD_CERTAINTY_ENABLE'] = bool(kin_cfg.get('certainty_enable', False))
    result['KIN_GUARD_CERTAINTY_MISMATCH_MULT'] = float(kin_cfg.get('certainty_mismatch_mult', 1.6))
    result['KIN_GUARD_CERTAINTY_SPEED_MULT'] = float(kin_cfg.get('certainty_speed_mult', 1.15))
    result['KIN_GUARD_CERTAINTY_INFLATE'] = float(kin_cfg.get('certainty_inflate', 1.35))
    result['KIN_GUARD_CERTAINTY_BLEND_ALPHA'] = float(kin_cfg.get('certainty_blend_alpha', 0.35))
    result['KIN_GUARD_CERTAINTY_SPEED_CAP_M_S'] = float(
        kin_cfg.get('certainty_speed_cap_m_s', kin_cfg.get('max_state_speed_m_s', 120.0))
    )
    result['KIN_GUARD_CERTAINTY_MIN_ACTION_DT_SEC'] = float(
        kin_cfg.get('certainty_min_action_dt_sec', kin_cfg.get('min_action_dt_sec', 0.25))
    )
    result['KIN_GUARD_CERTAINTY_REQUIRE_BOTH'] = bool(kin_cfg.get('certainty_require_both', False))
    
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
        result['trn'] = trn_cfg  # Store full config for TRN module
    else:
        result['TRN_ENABLED'] = False
        result['trn'] = {}

    # =========================================================================
    # Objective / optimization mode (accuracy-first vs runtime-first)
    # =========================================================================
    opt_cfg = config.get('optimization', {})
    result['OBJECTIVE_MODE'] = str(opt_cfg.get('objective', 'stability')).lower()
    result['POSITION_FIRST_LANE'] = bool(opt_cfg.get('position_first_lane', False))

    # =========================================================================
    # VPS accuracy mode + relocalization policy
    # =========================================================================
    vps_cfg = config.get('vps', {})
    result['VPS_ACCURACY_MODE'] = bool(vps_cfg.get('accuracy_mode', False))
    result['VPS_MATCHER_MODE'] = str(vps_cfg.get('matcher_mode', 'orb')).lower()
    result['VPS_MIN_UPDATE_INTERVAL'] = float(vps_cfg.get('min_update_interval', 0.5))
    result['VPS_MAX_TOTAL_CANDIDATES'] = int(vps_cfg.get('max_total_candidates', 0))
    result['VPS_MAX_FRAME_TIME_MS_LOCAL'] = float(vps_cfg.get('max_frame_time_ms_local', 0.0))
    result['VPS_MAX_FRAME_TIME_MS_GLOBAL'] = float(vps_cfg.get('max_frame_time_ms_global', 0.0))
    result['VPS_TILE_CACHE_MAX_TILES'] = int(vps_cfg.get('tile_cache_max_tiles', 50))
    result['VPS_MPS_CACHE_CLEAR_INTERVAL'] = int(vps_cfg.get('mps_cache_clear_interval', 8))
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
    result['VPS_APPLY_FAILSOFT_DIR_GATE_MAX_SPEED_M_S'] = float(
        vps_cfg.get('apply_failsoft_dir_gate_max_speed_m_s', 12.0)
    )
    result['VPS_APPLY_FAILSOFT_DIR_GATE_MAX_YAW_RATE_DEG_S'] = float(
        vps_cfg.get('apply_failsoft_dir_gate_max_yaw_rate_deg_s', 25.0)
    )
    result['VPS_APPLY_FAILSOFT_DIR_GATE_RELAX_MULT_SPEED'] = float(
        vps_cfg.get('apply_failsoft_dir_gate_relax_mult_speed', 1.6)
    )
    result['VPS_APPLY_FAILSOFT_DIR_GATE_RELAX_MULT_YAW_RATE'] = float(
        vps_cfg.get('apply_failsoft_dir_gate_relax_mult_yaw_rate', 1.6)
    )
    result['VPS_APPLY_FAILSOFT_DIR_GATE_DISABLE_SPEED_M_S'] = float(
        vps_cfg.get('apply_failsoft_dir_gate_disable_speed_m_s', 24.0)
    )
    result['VPS_APPLY_FAILSOFT_DIR_GATE_DISABLE_YAW_RATE_DEG_S'] = float(
        vps_cfg.get('apply_failsoft_dir_gate_disable_yaw_rate_deg_s', 45.0)
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
    result['VPS_ABS_HARD_REJECT_DIR_CHANGE_MAX_SPEED_M_S'] = float(
        vps_cfg.get('abs_hard_reject_dir_change_max_speed_m_s', 12.0)
    )
    result['VPS_ABS_HARD_REJECT_DIR_CHANGE_MAX_YAW_RATE_DEG_S'] = float(
        vps_cfg.get('abs_hard_reject_dir_change_max_yaw_rate_deg_s', 25.0)
    )
    result['VPS_ABS_HARD_REJECT_DIR_CHANGE_MIN_ACCEPTS'] = int(
        vps_cfg.get('abs_hard_reject_dir_change_min_accepts', 3)
    )
    result['VPS_ABS_MAX_APPLY_DP_XY_M'] = float(
        vps_cfg.get('abs_max_apply_dp_xy_m', 25.0)
    )
    result['VPS_XY_DRIFT_RECOVERY_ENABLE'] = bool(
        vps_cfg.get('xy_drift_recovery_enable', True)
    )
    result['VPS_XY_DRIFT_RECOVERY_MIN_OFFSET_M'] = float(
        vps_cfg.get('xy_drift_recovery_min_offset_m', 120.0)
    )
    result['VPS_XY_DRIFT_RECOVERY_MAX_OFFSET_M'] = float(
        vps_cfg.get('xy_drift_recovery_max_offset_m', 500.0)
    )
    result['VPS_XY_DRIFT_RECOVERY_MIN_NO_APPLY_SEC'] = float(
        vps_cfg.get('xy_drift_recovery_min_no_apply_sec', 5.0)
    )
    result['VPS_XY_DRIFT_RECOVERY_MIN_INLIERS'] = int(
        vps_cfg.get('xy_drift_recovery_min_inliers', 6)
    )
    result['VPS_XY_DRIFT_RECOVERY_MIN_CONFIDENCE'] = float(
        vps_cfg.get('xy_drift_recovery_min_confidence', 0.10)
    )
    result['VPS_XY_DRIFT_RECOVERY_MAX_REPROJ_ERROR'] = float(
        vps_cfg.get('xy_drift_recovery_max_reproj_error', 1.6)
    )
    result['VPS_XY_DRIFT_RECOVERY_MAX_SPEED_M_S'] = float(
        vps_cfg.get('xy_drift_recovery_max_speed_m_s', 95.0)
    )
    result['VPS_XY_DRIFT_RECOVERY_R_MULT'] = float(
        vps_cfg.get('xy_drift_recovery_r_mult', 2.8)
    )
    result['VPS_XY_DRIFT_RECOVERY_MAX_APPLY_DP_XY_M'] = float(
        vps_cfg.get('xy_drift_recovery_max_apply_dp_xy_m', 55.0)
    )
    result['VPS_XY_DRIFT_RECOVERY_ALLOW_DIR_CHANGE_BYPASS'] = bool(
        vps_cfg.get('xy_drift_recovery_allow_dir_change_bypass', True)
    )
    result['VPS_XY_DRIFT_RECOVERY_ALLOW_OFFSET_BYPASS'] = bool(
        vps_cfg.get('xy_drift_recovery_allow_offset_bypass', True)
    )
    result['VPS_XY_DRIFT_RECOVERY_SKIP_TEMPORAL_GATE'] = bool(
        vps_cfg.get('xy_drift_recovery_skip_temporal_gate', True)
    )
    result['VPS_XY_DRIFT_RECOVERY_REPORT_BACKEND_HINT_ON_REJECT'] = bool(
        vps_cfg.get('xy_drift_recovery_report_backend_hint_on_reject', True)
    )
    result['VPS_XY_DRIFT_RECOVERY_HINT_QUALITY_SCALE'] = float(
        vps_cfg.get('xy_drift_recovery_hint_quality_scale', 0.65)
    )
    result['VPS_POSITION_FIRST_SOFT_ENABLE'] = bool(
        vps_cfg.get('position_first_soft_enable', True)
    )
    result['VPS_POSITION_FIRST_SOFT_MIN_OFFSET_M'] = float(
        vps_cfg.get('position_first_soft_min_offset_m', 30.0)
    )
    result['VPS_POSITION_FIRST_SOFT_MIN_NO_APPLY_SEC'] = float(
        vps_cfg.get('position_first_soft_min_no_apply_sec', 2.0)
    )
    result['VPS_POSITION_FIRST_SOFT_MIN_INLIERS'] = int(
        vps_cfg.get('position_first_soft_min_inliers', 4)
    )
    result['VPS_POSITION_FIRST_SOFT_MIN_CONFIDENCE'] = float(
        vps_cfg.get('position_first_soft_min_confidence', 0.06)
    )
    result['VPS_POSITION_FIRST_SOFT_MAX_REPROJ_ERROR'] = float(
        vps_cfg.get('position_first_soft_max_reproj_error', 2.2)
    )
    result['VPS_POSITION_FIRST_SOFT_MAX_SPEED_M_S'] = float(
        vps_cfg.get('position_first_soft_max_speed_m_s', 110.0)
    )
    result['VPS_POSITION_FIRST_SOFT_SKIP_TEMPORAL_GATE'] = bool(
        vps_cfg.get('position_first_soft_skip_temporal_gate', True)
    )
    result['VPS_POSITION_FIRST_SOFT_IGNORE_POLICY_HOLD'] = bool(
        vps_cfg.get('position_first_soft_ignore_policy_hold', True)
    )
    result['VPS_POSITION_FIRST_SOFT_R_MULT'] = float(
        vps_cfg.get('position_first_soft_r_mult', 4.0)
    )
    result['VPS_POSITION_FIRST_SOFT_MAX_APPLY_DP_XY_M'] = float(
        vps_cfg.get('position_first_soft_max_apply_dp_xy_m', 80.0)
    )
    result['VPS_POSITION_FIRST_SOFT_HINT_QUALITY_SCALE'] = float(
        vps_cfg.get('position_first_soft_hint_quality_scale', 0.85)
    )
    # Position-first VPS controller:
    # Canonical source is vps.position_controller.* with legacy vps.position_first_direct_xy_* aliases.
    vps_pos_ctrl = vps_cfg.get("position_controller", {})
    if not isinstance(vps_pos_ctrl, dict):
        vps_pos_ctrl = {}

    def _pc_alias_value(canonical_key: str, legacy_key: str, default: Any) -> Any:
        if canonical_key in vps_pos_ctrl:
            return vps_pos_ctrl.get(canonical_key, default)
        return vps_cfg.get(legacy_key, default)

    def _pc_alias_bool(canonical_key: str, legacy_key: str, default: bool) -> bool:
        return bool(_pc_alias_value(canonical_key, legacy_key, default))

    def _pc_alias_int(canonical_key: str, legacy_key: str, default: int) -> int:
        return int(_pc_alias_value(canonical_key, legacy_key, default))

    def _pc_alias_float(canonical_key: str, legacy_key: str, default: float) -> float:
        return float(_pc_alias_value(canonical_key, legacy_key, default))

    # Canonical controller knobs (exported for new code paths)
    result['VPS_POSITION_CONTROLLER_ENABLE'] = _pc_alias_bool(
        'enable', 'position_first_direct_xy_enable', True
    )
    result['VPS_POSITION_CONTROLLER_CONSENSUS_WINDOW_SEC'] = _pc_alias_float(
        'consensus_window_sec', 'position_first_direct_xy_consensus_window_sec', 4.0
    )
    result['VPS_POSITION_CONTROLLER_MIN_SAMPLES'] = _pc_alias_int(
        'min_samples', 'position_first_direct_xy_consensus_min_samples', 3
    )
    result['VPS_POSITION_CONTROLLER_MAX_APPLY_STEP_M'] = _pc_alias_float(
        'max_apply_step_m',
        'position_first_direct_xy_max_apply_dp_xy_m',
        float(vps_cfg.get('position_first_soft_max_apply_dp_xy_m', 80.0)),
    )
    result['VPS_POSITION_CONTROLLER_WINDOW_BUDGET_M'] = _pc_alias_float(
        'window_budget_m', 'position_first_direct_xy_budget_max_total_dp_xy_m', 22.0
    )
    result['VPS_POSITION_CONTROLLER_FORCE_FAILSOFT_ON_REJECT'] = _pc_alias_bool(
        'force_failsoft_on_reject', 'position_first_direct_xy_force_failsoft_on_reject', False
    )
    result['VPS_POSITION_CONTROLLER_HIGH_SPEED_SOFT_CLAMP_ENABLE'] = _pc_alias_bool(
        'high_speed_soft_clamp_enable',
        'position_first_direct_xy_high_speed_soft_clamp_enable',
        True,
    )
    vps_apply_gate_cfg = vps_cfg.get("apply_gate", {}) if isinstance(vps_cfg.get("apply_gate", {}), dict) else {}
    result['VPS_APPLY_GATE_STRICT_SCORE_TH'] = float(vps_apply_gate_cfg.get('strict_score_th', 0.74))
    result['VPS_APPLY_GATE_FAILSOFT_SCORE_TH'] = float(vps_apply_gate_cfg.get('failsoft_score_th', 0.54))
    result['VPS_APPLY_GATE_HINT_SCORE_TH'] = float(vps_apply_gate_cfg.get('hint_score_th', 0.32))
    result['VPS_APPLY_GATE_CONSENSUS_WEIGHT'] = float(vps_apply_gate_cfg.get('consensus_weight', 0.35))
    result['VPS_APPLY_GATE_GEOMETRY_WEIGHT'] = float(vps_apply_gate_cfg.get('geometry_weight', 0.45))
    result['VPS_APPLY_GATE_MOTION_WEIGHT'] = float(vps_apply_gate_cfg.get('motion_weight', 0.20))
    result['VPS_APPLY_GATE_MOTION_HIGH_SPEED_M_S'] = float(
        vps_apply_gate_cfg.get('motion_high_speed_m_s', 45.0)
    )
    result['VPS_APPLY_GATE_MOTION_HIGH_YAWRATE_DEG_S'] = float(
        vps_apply_gate_cfg.get('motion_high_yawrate_deg_s', 55.0)
    )
    result['VPS_APPLY_GATE_FAILSOFT_DEFAULT_HINT_ONLY'] = bool(
        vps_apply_gate_cfg.get(
            'failsoft_default_hint_only',
            vps_cfg.get('position_first_direct_only_failsoft', True),
        )
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_ENABLE'] = bool(
        vps_apply_gate_cfg.get('bounded_soft_enable', True)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_R_MULT'] = float(
        vps_apply_gate_cfg.get('bounded_soft_r_mult', 2.6)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_MAX_APPLY_DP_XY_M'] = float(
        vps_apply_gate_cfg.get('bounded_soft_max_apply_dp_xy_m', 12.0)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_MIN_INTERVAL_SEC'] = float(
        vps_apply_gate_cfg.get('bounded_soft_min_interval_sec', 0.35)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_HIGH_SPEED_M_S'] = float(
        vps_apply_gate_cfg.get('bounded_soft_high_speed_m_s', 20.0)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_HIGH_SPEED_MAX_APPLY_DP_XY_M'] = float(
        vps_apply_gate_cfg.get('bounded_soft_high_speed_max_apply_dp_xy_m', 7.0)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_SCORE_TH'] = float(
        vps_apply_gate_cfg.get('bounded_soft_score_th', vps_apply_gate_cfg.get('hint_score_th', 0.32))
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_ENABLE'] = bool(
        vps_apply_gate_cfg.get('bounded_soft_policy_hold_bypass_enable', True)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MIN_NO_APPLY_SEC'] = float(
        vps_apply_gate_cfg.get('bounded_soft_policy_hold_bypass_min_no_apply_sec', 3.0)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MIN_SCORE_TH'] = float(
        vps_apply_gate_cfg.get('bounded_soft_policy_hold_bypass_min_score_th', 0.40)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MAX_SPEED_M_S'] = float(
        vps_apply_gate_cfg.get('bounded_soft_policy_hold_bypass_max_speed_m_s', 95.0)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MAX_OFFSET_M'] = float(
        vps_apply_gate_cfg.get('bounded_soft_policy_hold_bypass_max_offset_m', 180.0)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MIN_INLIERS'] = int(
        vps_apply_gate_cfg.get('bounded_soft_policy_hold_bypass_min_inliers', 5)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MIN_CONFIDENCE'] = float(
        vps_apply_gate_cfg.get('bounded_soft_policy_hold_bypass_min_confidence', 0.12)
    )
    result['VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MAX_REPROJ_ERROR'] = float(
        vps_apply_gate_cfg.get('bounded_soft_policy_hold_bypass_max_reproj_error', 1.2)
    )

    # Backward-compatible runtime keys (single alias chain via canonical source).
    result['VPS_POSITION_FIRST_DIRECT_XY_ENABLE'] = bool(result['VPS_POSITION_CONTROLLER_ENABLE'])
    result['VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_WINDOW_SEC'] = float(
        result['VPS_POSITION_CONTROLLER_CONSENSUS_WINDOW_SEC']
    )
    result['VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_MIN_SAMPLES'] = int(
        result['VPS_POSITION_CONTROLLER_MIN_SAMPLES']
    )
    result['VPS_POSITION_FIRST_DIRECT_XY_MAX_APPLY_DP_XY_M'] = float(
        result['VPS_POSITION_CONTROLLER_MAX_APPLY_STEP_M']
    )
    result['VPS_POSITION_FIRST_DIRECT_XY_BUDGET_MAX_TOTAL_DP_XY_M'] = float(
        result['VPS_POSITION_CONTROLLER_WINDOW_BUDGET_M']
    )
    result['VPS_POSITION_FIRST_DIRECT_XY_FORCE_FAILSOFT_ON_REJECT'] = bool(
        result['VPS_POSITION_CONTROLLER_FORCE_FAILSOFT_ON_REJECT']
    )
    result['VPS_POSITION_FIRST_DIRECT_XY_HIGH_SPEED_SOFT_CLAMP_ENABLE'] = bool(
        result['VPS_POSITION_CONTROLLER_HIGH_SPEED_SOFT_CLAMP_ENABLE']
    )

    direct_alias_specs = [
        ('VPS_POSITION_FIRST_DIRECT_XY_MIN_INTERVAL_SEC', 'min_interval_sec', 'position_first_direct_xy_min_interval_sec', 0.8, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_QUALITY_SCALE', 'quality_scale', 'position_first_direct_xy_quality_scale', 0.90, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_FALLBACK_ON_APPLY_FAIL', 'fallback_on_apply_fail', 'position_first_direct_xy_fallback_on_apply_fail', True, bool),
        ('VPS_POSITION_FIRST_DIRECT_XY_FORCE_FAILSOFT_MAX_OFFSET_M', 'force_failsoft_max_offset_m', 'position_first_direct_xy_force_failsoft_max_offset_m', 180.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_FORCE_FAILSOFT_MIN_INLIERS', 'force_failsoft_min_inliers', 'position_first_direct_xy_force_failsoft_min_inliers', 5, int),
        ('VPS_POSITION_FIRST_DIRECT_XY_FORCE_FAILSOFT_MAX_REPROJ_ERROR', 'force_failsoft_max_reproj_error', 'position_first_direct_xy_force_failsoft_max_reproj_error', 1.4, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_FORCE_FAILSOFT_MAX_SPEED_M_S', 'force_failsoft_max_speed_m_s', 'position_first_direct_xy_force_failsoft_max_speed_m_s', 95.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_MIN_OFFSET_M', 'min_offset_m', 'position_first_direct_xy_min_offset_m', 8.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_MAX_OFFSET_M', 'max_offset_m', 'position_first_direct_xy_max_offset_m', float(vps_cfg.get('xy_drift_recovery_max_offset_m', 1800.0)), float),
        ('VPS_POSITION_FIRST_DIRECT_XY_MIN_INLIERS', 'min_inliers', 'position_first_direct_xy_min_inliers', 3, int),
        ('VPS_POSITION_FIRST_DIRECT_XY_MIN_CONFIDENCE', 'min_confidence', 'position_first_direct_xy_min_confidence', 0.02, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_MAX_REPROJ_ERROR', 'max_reproj_error', 'position_first_direct_xy_max_reproj_error', 4.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_MAX_SPEED_M_S', 'max_speed_m_s', 'position_first_direct_xy_max_speed_m_s', 120.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_HIGH_SPEED_TH_M_S', 'high_speed_th_m_s', 'position_first_direct_xy_high_speed_th_m_s', 18.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_MIN_INLIERS_HIGH_SPEED', 'min_inliers_high_speed', 'position_first_direct_xy_min_inliers_high_speed', 6, int),
        ('VPS_POSITION_FIRST_DIRECT_XY_MAX_REPROJ_HIGH_SPEED', 'max_reproj_high_speed', 'position_first_direct_xy_max_reproj_high_speed', 1.2, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_MAX_OFFSET_HIGH_SPEED_M', 'max_offset_high_speed_m', 'position_first_direct_xy_max_offset_high_speed_m', 140.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_IGNORE_POLICY_HOLD', 'ignore_policy_hold', 'position_first_direct_xy_ignore_policy_hold', False, bool),
        ('VPS_POSITION_FIRST_DIRECT_XY_IGNORE_HARD_REJECT', 'ignore_hard_reject', 'position_first_direct_xy_ignore_hard_reject', False, bool),
        ('VPS_POSITION_FIRST_DIRECT_XY_ALLOW_HARD_REJECT_WITH_CONSENSUS', 'allow_hard_reject_with_consensus', 'position_first_direct_xy_allow_hard_reject_with_consensus', True, bool),
        ('VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_ENABLE', 'consensus_enable', 'position_first_direct_xy_consensus_enable', True, bool),
        ('VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_MAX_DEV_M', 'consensus_max_dev_m', 'position_first_direct_xy_consensus_max_dev_m', 30.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_MAX_DIR_DEG', 'consensus_max_dir_deg', 'position_first_direct_xy_consensus_max_dir_deg', 70.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_MAX_KEEP', 'consensus_max_keep', 'position_first_direct_xy_consensus_max_keep', 24, int),
        ('VPS_POSITION_FIRST_DIRECT_XY_USE_CONSENSUS_VECTOR', 'use_consensus_vector', 'position_first_direct_xy_use_consensus_vector', True, bool),
        ('VPS_POSITION_FIRST_DIRECT_XY_QUALITY_CAP_MIN_M', 'quality_cap_min_m', 'position_first_direct_xy_quality_cap_min_m', 6.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_BUDGET_WINDOW_SEC', 'budget_window_sec', 'position_first_direct_xy_budget_window_sec', 5.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MIN_OFFSET_M', 'recovery_min_offset_m', 'position_first_direct_xy_recovery_min_offset_m', 120.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MIN_NO_APPLY_SEC', 'recovery_min_no_apply_sec', 'position_first_direct_xy_recovery_min_no_apply_sec', 6.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MAX_SPEED_M_S', 'recovery_max_speed_m_s', 'position_first_direct_xy_recovery_max_speed_m_s', 14.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MIN_INLIERS', 'recovery_min_inliers', 'position_first_direct_xy_recovery_min_inliers', 7, int),
        ('VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MAX_REPROJ_ERROR', 'recovery_max_reproj_error', 'position_first_direct_xy_recovery_max_reproj_error', 0.9, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MAX_DP_MULT', 'recovery_max_dp_mult', 'position_first_direct_xy_recovery_max_dp_mult', 1.8, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MAX_DP_CAP_M', 'recovery_max_dp_cap_m', 'position_first_direct_xy_recovery_max_dp_cap_m', 90.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_BUDGET_MULT', 'recovery_budget_mult', 'position_first_direct_xy_recovery_budget_mult', 2.5, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_BUDGET_CAP_M', 'recovery_budget_cap_m', 'position_first_direct_xy_recovery_budget_cap_m', 120.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_GUARD_ENABLE', 'guard_enable', 'position_first_direct_xy_guard_enable', True, bool),
        ('VPS_POSITION_FIRST_DIRECT_XY_GUARD_FAIL_STREAK_TH', 'guard_fail_streak_th', 'position_first_direct_xy_guard_fail_streak_th', 18, int),
        ('VPS_POSITION_FIRST_DIRECT_XY_GUARD_NO_COVERAGE_STREAK_TH', 'guard_no_coverage_streak_th', 'position_first_direct_xy_guard_no_coverage_streak_th', 6, int),
        ('VPS_POSITION_FIRST_DIRECT_XY_GUARD_MAX_APPLY_DP_XY_M', 'guard_max_apply_dp_xy_m', 'position_first_direct_xy_guard_max_apply_dp_xy_m', 14.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_GUARD_BUDGET_MAX_TOTAL_DP_XY_M', 'guard_budget_max_total_dp_xy_m', 'position_first_direct_xy_guard_budget_max_total_dp_xy_m', 18.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_GUARD_QUALITY_MULT', 'guard_quality_mult', 'position_first_direct_xy_guard_quality_mult', 0.78, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_GUARD_MIN_QUALITY', 'guard_min_quality', 'position_first_direct_xy_guard_min_quality', 0.28, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_GUARD_EXTRA_CONFIRM_HITS', 'guard_extra_confirm_hits', 'position_first_direct_xy_guard_extra_confirm_hits', 1, int),
        ('VPS_POSITION_FIRST_DIRECT_XY_GUARD_MAX_DIR_CHANGE_DEG', 'guard_max_dir_change_deg', 'position_first_direct_xy_guard_max_dir_change_deg', 55.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_GUARD_MIN_DIR_CHECK_M', 'guard_min_dir_check_m', 'position_first_direct_xy_guard_min_dir_check_m', 8.0, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_RECOVERY_MIN_INLIERS', 'failsoft_recovery_min_inliers', 'position_first_direct_xy_failsoft_recovery_min_inliers', 5, int),
        ('VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_RECOVERY_MAX_REPROJ_ERROR', 'failsoft_recovery_max_reproj_error', 'position_first_direct_xy_failsoft_recovery_max_reproj_error', 1.3, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_GUARD_LOWQ_ENABLE', 'failsoft_guard_lowq_enable', 'position_first_direct_xy_failsoft_guard_lowq_enable', True, bool),
        ('VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_GUARD_LOWQ_MIN_INLIERS', 'failsoft_guard_lowq_min_inliers', 'position_first_direct_xy_failsoft_guard_lowq_min_inliers', 4, int),
        ('VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_GUARD_LOWQ_MAX_REPROJ_ERROR', 'failsoft_guard_lowq_max_reproj_error', 'position_first_direct_xy_failsoft_guard_lowq_max_reproj_error', 1.6, float),
        ('VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_GUARD_LOWQ_MAX_APPLY_DP_XY_M', 'failsoft_guard_lowq_max_apply_dp_xy_m', 'position_first_direct_xy_failsoft_guard_lowq_max_apply_dp_xy_m', 7.0, float),
    ]
    for out_key, canonical_key, legacy_key, default_value, cast_type in direct_alias_specs:
        raw_value = _pc_alias_value(canonical_key, legacy_key, default_value)
        result[out_key] = cast_type(raw_value)
    result['VPS_WORKER_BUSY_FORCE_LOCAL_STREAK'] = int(
        vps_cfg.get('worker_busy_force_local_streak', 120)
    )
    result['VPS_WORKER_BUSY_FORCE_LOCAL_SEC'] = float(
        vps_cfg.get('worker_busy_force_local_sec', 8.0)
    )
    result['VPS_WORKER_BUSY_BACKPRESSURE_SEC'] = float(
        vps_cfg.get('worker_busy_backpressure_sec', 0.40)
    )
    result['VPS_WORKER_BUSY_BACKPRESSURE_MAX_SEC'] = float(
        vps_cfg.get('worker_busy_backpressure_max_sec', 2.5)
    )
    result['VPS_WORKER_BUSY_BACKPRESSURE_STEP'] = int(
        vps_cfg.get('worker_busy_backpressure_step', 30)
    )

    # =========================================================================
    # Runtime memory watchdog / compaction
    # =========================================================================
    mem_cfg = config.get('memory', {}) if isinstance(config.get('memory', {}), dict) else {}
    result['MEMORY_WATCHDOG_ENABLE'] = bool(mem_cfg.get('watchdog_enable', True))
    result['MEMORY_CHECK_INTERVAL_SEC'] = float(mem_cfg.get('check_interval_sec', 1.0))
    result['MEMORY_SOFT_LIMIT_GB'] = float(mem_cfg.get('soft_limit_gb', 16.0))
    result['MEMORY_HARD_LIMIT_GB'] = float(mem_cfg.get('hard_limit_gb', 24.0))
    result['MEMORY_COMPACT_COOLDOWN_SEC'] = float(mem_cfg.get('compact_cooldown_sec', 2.0))
    result['MEMORY_PERIODIC_COMPACT_SEC'] = float(mem_cfg.get('periodic_compact_sec', 10.0))
    result['MEMORY_VPS_PAUSE_SEC'] = float(mem_cfg.get('vps_pause_sec', 4.0))
    result['MEMORY_COMPACT_FRONTEND_TRACK_LEN'] = int(
        mem_cfg.get('compact_frontend_track_len', result.get('VIO_FRONTEND_MAX_TRACK_HISTORY', 90))
    )
    result['MEMORY_COMPACT_FRONTEND_MAX_TRACKS'] = int(
        mem_cfg.get('compact_frontend_max_tracks', result.get('VIO_FRONTEND_MAX_TOTAL_TRACKS', 5000))
    )
    result['MEMORY_COMPACT_LOOP_MAX_KEYFRAMES'] = int(
        mem_cfg.get('compact_loop_max_keyframes', result.get('LOOP_MAX_KEYFRAMES', 220))
    )
    result['MEMORY_COMPACT_VPS_TILE_CACHE_MAX'] = int(
        mem_cfg.get('compact_vps_tile_cache_max', result.get('VPS_TILE_CACHE_MAX_TILES', 50))
    )
    result['MEMORY_COMPACT_VPS_WALL_SAMPLES'] = int(mem_cfg.get('compact_vps_wall_samples', 1200))

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
    result['BACKEND_MIN_EMIT_DP_XY_M'] = float(backend_cfg.get('min_emit_dp_xy_m', 0.5))
    result['BACKEND_MIN_EMIT_DYAW_DEG'] = float(backend_cfg.get('min_emit_dyaw_deg', 0.2))
    result['BACKEND_MAX_ABS_DP_XY_M'] = float(backend_cfg.get('max_abs_dp_xy_m', 60.0))
    result['BACKEND_MAX_ABS_DYAW_DEG'] = float(backend_cfg.get('max_abs_dyaw_deg', 8.0))
    robust_yaw_cfg = backend_cfg.get('robust_yaw', {})
    switchable_cfg = backend_cfg.get('switchable_constraints', {})
    blend_cfg = backend_cfg.get('correction_blend', {})
    factor_lite_cfg = backend_cfg.get('hybrid_factor_lite', {})
    transport_cfg = backend_cfg.get('transport', {})
    contract_cfg = backend_cfg.get('contract', {})
    contract_v1_cfg = backend_cfg.get('contract_v1', {})
    if not isinstance(contract_v1_cfg, dict):
        contract_v1_cfg = {}
    result['BACKEND_ROBUST_YAW_ENABLE'] = bool(robust_yaw_cfg.get('enable', True))
    result['BACKEND_ROBUST_YAW_HUBER_DEG'] = float(robust_yaw_cfg.get('huber_deg', 6.0))
    result['BACKEND_SWITCHABLE_CONSTRAINTS_ENABLE'] = bool(switchable_cfg.get('enable', True))
    result['BACKEND_SWITCHABLE_QUALITY_FLOOR'] = float(switchable_cfg.get('quality_floor', 0.08))
    result['BACKEND_SWITCHABLE_RESIDUAL_XY_M'] = float(switchable_cfg.get('residual_xy_m', 12.0))
    result['BACKEND_SWITCHABLE_RESIDUAL_YAW_DEG'] = float(switchable_cfg.get('residual_yaw_deg', 15.0))
    result['BACKEND_SWITCHABLE_MIN_WEIGHT'] = float(switchable_cfg.get('min_weight', 0.15))
    result['BACKEND_BLEND_MODE'] = str(blend_cfg.get('mode', 'linear')).lower()
    result['BACKEND_BLEND_ALPHA'] = float(blend_cfg.get('alpha', 0.45))
    result['BACKEND_BLEND_MAX_STEP_DP_XY_M'] = float(blend_cfg.get('max_step_dp_xy_m', 8.0))
    result['BACKEND_BLEND_MAX_STEP_DYAW_DEG'] = float(blend_cfg.get('max_step_dyaw_deg', 0.8))
    result['BACKEND_BLEND_QUALITY_SOFT_SCALE'] = bool(blend_cfg.get('quality_soft_scale', True))
    result['BACKEND_HYBRID_FACTOR_LITE_ENABLE'] = bool(factor_lite_cfg.get('enable', False))
    result['BACKEND_HYBRID_FACTOR_LITE_WINDOW'] = int(factor_lite_cfg.get('window', 10))
    result['BACKEND_HYBRID_FACTOR_LITE_LOSS_XY_M'] = float(factor_lite_cfg.get('loss_xy_m', 8.0))
    result['BACKEND_HYBRID_FACTOR_LITE_LOSS_YAW_DEG'] = float(factor_lite_cfg.get('loss_yaw_deg', 10.0))
    result['BACKEND_HYBRID_FACTOR_LITE_USE_IMU'] = bool(factor_lite_cfg.get('use_imu', True))
    result['BACKEND_HYBRID_FACTOR_LITE_USE_VISUAL'] = bool(factor_lite_cfg.get('use_visual', True))
    result['BACKEND_HYBRID_FACTOR_LITE_USE_VPS'] = bool(factor_lite_cfg.get('use_vps', True))
    result['BACKEND_HYBRID_FACTOR_LITE_USE_DEM'] = bool(factor_lite_cfg.get('use_dem', True))
    result['BACKEND_HYBRID_FACTOR_LITE_USE_VPS_XY'] = bool(
        factor_lite_cfg.get('use_vps_xy', factor_lite_cfg.get('use_vps', True))
    )
    result['BACKEND_HYBRID_FACTOR_LITE_USE_VPS_YAW'] = bool(
        factor_lite_cfg.get('use_vps_yaw', factor_lite_cfg.get('use_vps', True))
    )
    result['BACKEND_HYBRID_FACTOR_LITE_VPS_YAW_QUALITY_FLOOR'] = float(
        factor_lite_cfg.get('vps_yaw_quality_floor', 0.30)
    )
    result['BACKEND_HYBRID_FACTOR_LITE_VPS_YAW_CAP_DEG'] = float(
        factor_lite_cfg.get('vps_yaw_cap_deg', backend_cfg.get('max_abs_dyaw_deg', 8.0))
    )
    result['BACKEND_HYBRID_FACTOR_LITE_USE_LOOP_YAW'] = bool(
        factor_lite_cfg.get('use_loop_yaw', factor_lite_cfg.get('use_visual', True))
    )
    result['BACKEND_HYBRID_FACTOR_LITE_USE_MAG_YAW'] = bool(
        factor_lite_cfg.get('use_mag_yaw', True)
    )
    result['BACKEND_TRANSPORT_LATEST_WINS_ENABLE'] = bool(
        transport_cfg.get('latest_wins_enable', True)
    )
    result['BACKEND_TRANSPORT_DROP_STALE_ON_EMIT'] = bool(
        transport_cfg.get('drop_stale_on_emit', True)
    )
    result['BACKEND_TRANSPORT_POLL_ON_CAMERA_TICK_ONLY'] = bool(
        transport_cfg.get('poll_on_camera_tick_only', True)
    )
    result['BACKEND_TRANSPORT_POLL_MIN_INTERVAL_SEC'] = float(
        transport_cfg.get('poll_min_interval_sec', backend_cfg.get('poll_interval_sec', 0.5))
    )
    result['BACKEND_CONTRACT_STRICT_REQUIRE_SOURCE_MIX'] = bool(
        contract_v1_cfg.get('strict_require_source_mix', contract_cfg.get('strict_require_source_mix', True))
    )
    result['BACKEND_CONTRACT_STRICT_REQUIRE_RESIDUAL_SUMMARY'] = bool(
        contract_v1_cfg.get('strict_require_residual_summary', contract_cfg.get('strict_require_residual_summary', True))
    )
    result['BACKEND_CONTRACT_V1_STRICT_REQUIRE_VERSION'] = bool(
        contract_v1_cfg.get('strict_require_version', True)
    )
    result['BACKEND_CONTRACT_V1_EXPECTED_VERSION'] = str(
        contract_v1_cfg.get('expected_version', "v1")
    )
    result['BACKEND_VPS_YAW_HINT_ENABLE'] = bool(
        backend_cfg.get('vps_yaw_hint_enable', True)
    )
    result['BACKEND_VPS_YAW_HINT_GAIN'] = float(
        backend_cfg.get('vps_yaw_hint_gain', 0.35)
    )
    result['BACKEND_VPS_YAW_HINT_MAX_ABS_DEG'] = float(
        backend_cfg.get('vps_yaw_hint_max_abs_deg', 35.0)
    )
    result['BACKEND_VPS_YAW_HINT_MIN_QUALITY'] = float(
        backend_cfg.get('vps_yaw_hint_min_quality', 0.45)
    )
    result['BACKEND_VPS_YAW_HINT_MIN_INLIERS'] = int(
        backend_cfg.get('vps_yaw_hint_min_inliers', 8)
    )
    result['BACKEND_VPS_YAW_HINT_MAX_REPROJ'] = float(
        backend_cfg.get('vps_yaw_hint_max_reproj', 1.2)
    )
    result['BACKEND_VPS_YAW_HINT_MAX_APPLY_DEG'] = float(
        backend_cfg.get('vps_yaw_hint_max_apply_deg', 2.0)
    )
    result['BACKEND_POSITION_FIRST_FORCE_XY_ONLY'] = bool(
        backend_cfg.get('position_first_force_xy_only', True)
    )
    result['BACKEND_POSITION_FIRST_MIN_VPS_MIX'] = float(
        backend_cfg.get('position_first_min_vps_mix', 0.35)
    )
    result['BACKEND_POSITION_FIRST_MAX_APPLY_DP_XY_M'] = float(
        backend_cfg.get('position_first_max_apply_dp_xy_m', backend_cfg.get('max_apply_dp_xy_m', 25.0))
    )
    result['BACKEND_POSITION_FIRST_CORR_WEIGHT'] = float(
        backend_cfg.get('position_first_corr_weight', backend_cfg.get('correction_weight', 1.0))
    )
    result['BACKEND_POSITION_FIRST_BLEND_STEPS'] = int(
        backend_cfg.get('position_first_blend_steps', backend_cfg.get('blend_steps', 3))
    )
    result['BACKEND_POSITION_FIRST_DIRECT_XY_MAX_APPLY_DP_XY_M'] = float(
        backend_cfg.get(
            'position_first_direct_xy_max_apply_dp_xy_m',
            min(
                float(backend_cfg.get('position_first_max_apply_dp_xy_m', backend_cfg.get('max_apply_dp_xy_m', 25.0))),
                28.0,
            ),
        )
    )
    result['BACKEND_POSITION_FIRST_DIRECT_XY_CORR_WEIGHT'] = float(
        backend_cfg.get(
            'position_first_direct_xy_corr_weight',
            min(
                float(backend_cfg.get('position_first_corr_weight', backend_cfg.get('correction_weight', 1.0))),
                0.72,
            ),
        )
    )
    result['BACKEND_POSITION_FIRST_DIRECT_XY_MIN_BLEND_STEPS'] = int(
        backend_cfg.get('position_first_direct_xy_min_blend_steps', 3)
    )
    result['BACKEND_XY_PRIORITY_ENABLE'] = bool(
        backend_cfg.get('xy_priority_enable', True)
    )
    result['BACKEND_XY_PRIORITY_MIN_VPS_MIX'] = float(
        backend_cfg.get('xy_priority_min_vps_mix', 0.70)
    )
    result['BACKEND_XY_PRIORITY_MAX_LOOP_MIX'] = float(
        backend_cfg.get('xy_priority_max_loop_mix', 0.10)
    )
    result['BACKEND_XY_PRIORITY_MAX_MAG_MIX'] = float(
        backend_cfg.get('xy_priority_max_mag_mix', 0.15)
    )
    result['BACKEND_XY_PRIORITY_QUALITY_TH'] = float(
        backend_cfg.get('xy_priority_quality_th', 0.55)
    )
    result['BACKEND_XY_PRIORITY_MAX_DYAW_DEG'] = float(
        backend_cfg.get('xy_priority_max_dyaw_deg', 0.35)
    )
    result['BACKEND_XY_PRIORITY_LOWQ_MAX_DYAW_DEG'] = float(
        backend_cfg.get('xy_priority_lowq_max_dyaw_deg', 0.15)
    )
    result['BACKEND_YAW_AUTH_MIN_REQUEST_DEG'] = float(
        backend_cfg.get('yaw_auth_min_request_deg', 0.05)
    )
    result['BACKEND_YAW_AUTH_ZERO_REQUEST_DEADBAND_DEG'] = float(
        backend_cfg.get('yaw_auth_zero_request_deadband_deg', 0.02)
    )
    result['BACKEND_YAW_AUTH_ZERO_DYAW_AS_REQUEST'] = bool(
        backend_cfg.get('yaw_auth_zero_dyaw_as_request', False)
    )

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
    _mag_acc = _merge_dict_defaults(
        mag_accuracy_defaults, mag.get('accuracy_policy', {})
    )
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
