"""
Main VIO Loop Runner

This module provides the VIORunner class that orchestrates the complete
VIO+EKF pipeline, integrating all modular components:

- Data loading (config, IMU, images, VPS, MAG, DEM)
- State initialization (position, velocity, orientation, biases)
- IMU propagation (preintegration or legacy)
- Measurement updates (VPS, MAG, DEM, VIO velocity, MSCKF)
- Output logging (pose, error, debug CSVs)

Usage:
    from vio.main_loop import VIORunner, VIOConfig
    
    config = VIOConfig(
        imu_path="imu.csv",
        quarry_path="gga.csv",
        output_dir="output/",
        # ... other parameters
    )
    
    runner = VIORunner(config)
    runner.run()

Author: VIO project
"""

import os
import time
import math
import threading
import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

# Performance-critical imports moved to top-level to avoid per-iteration overhead
from .config import load_config, VIOConfig, CAMERA_VIEW_CONFIGS
from .data_loaders import (
    load_imu_csv, load_images,
    load_mag_csv, load_ppk_initial_state, load_ppk_trajectory,
    DEMReader, ProjectionCache, load_flight_log_msl
)
from .ekf import ExtendedKalmanFilter
from .state_manager import initialize_ekf_state, imu_to_gnss_position
from .vio_frontend import VIOFrontEnd
from .camera import make_KD_for_size
from .propagation import (
    propagate_error_state_covariance, 
    augment_state_with_camera, detect_stationary, apply_zupt,
    apply_gravity_roll_pitch_update,
    apply_yaw_pseudo_update, apply_bias_observability_guard,
    apply_preintegration_at_camera, clone_camera_for_msckf,
    get_flight_phase  # v3.3.0: State-based phase detection
)
from .vps_integration import apply_vps_update, compute_vps_innovation
from .msckf import perform_msckf_updates, print_msckf_stats, trigger_msckf_update
from .plane_detection import PlaneDetector
from .measurement_updates import (
    apply_magnetometer_update, apply_dem_height_update, apply_vio_velocity_update
)
from .magnetometer import (
    calibrate_magnetometer, reset_mag_filter_state, 
    set_mag_constants, apply_mag_filter
)
from .math_utils import quaternion_to_yaw, skew_symmetric
from .loop_closure import (
    get_loop_detector, init_loop_closure, LoopClosureDetector,
    check_loop_closure, apply_loop_closure_correction
)
from .imu_preintegration import IMUPreintegration
from .fisheye_rectifier import FisheyeRectifier, create_rectifier_from_config
from .propagation import VibrationDetector
from .output_utils import (
    save_calibration_log, save_keyframe_image_with_overlay,
    save_keyframe_with_overlay,
    log_state_debug, log_vo_debug, log_msckf_window, log_fej_consistency,
    log_adaptive_decision, log_sensor_health,
    append_benchmark_health_summary,
    DebugCSVWriters, init_output_csvs, get_ground_truth_error,
    build_calibration_params
)
from .adaptive_controller import AdaptiveController, AdaptiveContext, AdaptiveDecision
from .trn import TerrainReferencedNavigation, TRNConfig, create_trn_from_config
from .imu_driven import run_imu_driven_loop
from .event_driven import run_event_driven_loop

@dataclass
class VIOState:
    """Runtime state container."""
    # Indices and counters
    imu_idx: int = 0
    img_idx: int = 0
    vps_idx: int = 0
    mag_idx: int = 0
    vio_frame: int = 0  # Changed from -1: first frame is 0
    
    # Statistics
    zupt_applied: int = 0
    zupt_rejected: int = 0
    zupt_detected: int = 0
    mag_updates: int = 0
    mag_rejects: int = 0
    consecutive_stationary: int = 0
    imu_propagation_count: int = 0  # For debug logging
    
    # Timing
    t0: float = 0.0
    last_t: float = 0.0
    
    # Flight phase
    current_phase: int = 0  # 0=SPINUP, 1=EARLY, 2=NORMAL
    
    # Feature tracking
    cam_states: List[dict] = field(default_factory=list)
    cam_observations: List[dict] = field(default_factory=list)


class VIORunner:
    """
    Main VIO+EKF runner class.
    
    Orchestrates the complete visual-inertial odometry pipeline:
    1. Data loading
    2. State initialization
    3. IMU-driven propagation loop
    4. Asynchronous measurement updates
    5. Output logging
    """
    
    # Flight phase constants (loaded from YAML config in v3.2.0)
    # NOTE: Time-based phases NOT suitable for GPS-denied realtime!
    # TODO v3.3.0: Replace with state-based detection (velocity, altitude, vibration)
    PHASE_NAMES = ["SPINUP", "EARLY", "NORMAL"]
    
    def __init__(self, config: VIOConfig):
        """
        Initialize VIO runner.
        
        Args:
            config: VIOConfig instance with all settings
        """
        self.config = config
        self.state = VIOState()
        
        # Flight phase detection thresholds (v3.4.0: State-based)
        self.PHASE_SPINUP_VELOCITY_THRESH = 1.0
        self.PHASE_SPINUP_VIBRATION_THRESH = 0.3
        self.PHASE_SPINUP_ALT_CHANGE_THRESH = 5.0
        self.PHASE_EARLY_VELOCITY_SIGMA_THRESH = 3.0
        
        # These will be initialized in run()
        self.kf = None
        self.imu = None
        self.imgs = None
        self.vps_list = None
        self.mag_list = None
        self.dem = None
        self.vio_fe = None
        self.plane_detector = None  # Plane detection for plane-aided MSCKF
        
        # Origin coordinates
        self.lat0 = 0.0
        self.lon0 = 0.0
        self.msl0 = 0.0
        self.dem0 = None
        self.initial_msl = 0.0  # For TRN altitude change tracking
        self.msl_interpolator = None  # Direct MSL update source (Barometer/GNSS)
        
        # Global config (loaded from YAML)
        self.global_config = {}
        
        # Output file handles
        self.pose_csv = None
        self.error_csv = None
        self.state_dbg_csv = None
        self.time_sync_csv = None
        self.cov_health_csv = None
        self.adaptive_debug_csv = None
        self.sensor_health_csv = None
        self.conditioning_events_csv = None
        self.benchmark_health_summary_csv = None
        
        # Fisheye rectifier (optional)
        self.rectifier: Optional[FisheyeRectifier] = None
        
        # Loop closure detector (optional)
        self.loop_detector: Optional[LoopClosureDetector] = None
        
        # Vibration detector (optional)
        self.vibration_detector: Optional[VibrationDetector] = None
        
        # TRN (Terrain Referenced Navigation) - v3.3.0
        self.trn: Optional[TerrainReferencedNavigation] = None
        
        # Last gyro for mag filtering
        self.last_gyro_z = 0.0
        self.last_imu_dt = 0.01
        
        # Projection cache for coordinate conversion
        self.proj_cache = ProjectionCache()
        
        # IMU-GNSS lever arm (will be populated from config)
        self.lever_arm = np.zeros(3)
        
        # VPS timing tracking
        self.last_vps_update_time = 0.0
        
        # Timestamp base tracking (for GT/error alignment)
        self.imu_time_col: Optional[str] = None
        self.error_time_scale: float = 1.0
        self.error_time_offset: float = 0.0
        self.error_time_mode: str = "identity"
        self._warned_gt_time_mismatch: bool = False

        # Adaptive controller runtime
        self.adaptive_controller: Optional[AdaptiveController] = None
        self.current_adaptive_decision: Optional[AdaptiveDecision] = None
        self._adaptive_last_aiding_time: Optional[float] = None
        self._adaptive_prev_pmax: Optional[float] = None
        self._adaptive_log_enabled: bool = True
        self._legacy_covariance_cap: float = 1e8
        self._adaptive_last_policy_t: Optional[float] = None
        self._conditioning_warning_sec: float = 0.0
        self.imu_only_mode: bool = False
        self._imu_only_yaw_ref: Optional[float] = None
        self._imu_only_bg_ref: Optional[np.ndarray] = None
        self._imu_only_ba_ref: Optional[np.ndarray] = None
    
    def load_config(self) -> dict:
        """Load YAML configuration file."""
        if self.config.config_yaml:
            cfg = load_config(self.config.config_yaml)
            self.global_config = cfg
            
            # Load flight phase detection thresholds from YAML (v3.4.0)
            self.PHASE_SPINUP_VELOCITY_THRESH = cfg.get('PHASE_SPINUP_VELOCITY_THRESH', 1.0)
            self.PHASE_SPINUP_VIBRATION_THRESH = cfg.get('PHASE_SPINUP_VIBRATION_THRESH', 0.3)
            self.PHASE_SPINUP_ALT_CHANGE_THRESH = cfg.get('PHASE_SPINUP_ALT_CHANGE_THRESH', 5.0)
            self.PHASE_EARLY_VELOCITY_SIGMA_THRESH = cfg.get('PHASE_EARLY_VELOCITY_SIGMA_THRESH', 3.0)
            
            return cfg
        return {}
    
    def load_data(self):
        """Load all input data sources."""
        # Load PPK ground truth if available
        self.ppk_state = None
        if self.config.ground_truth_path:
            self.ppk_state = load_ppk_initial_state(self.config.ground_truth_path)
        
        # Load initial position from PPK (required for GPS-denied operation)
        if self.ppk_state is not None:
            self.lat0 = self.ppk_state.lat
            self.lon0 = self.ppk_state.lon
            v_init = np.array([self.ppk_state.ve, self.ppk_state.vn, self.ppk_state.vu])
            print(f"[INIT] Using PPK for lat/lon/velocity")
        else:
            raise RuntimeError("[INIT] PPK file is required for initialization. Check ppk_csv path in config.")
        
        self.v_init = v_init
        
        # Use PPK ellipsoidal height
        self.msl0 = self.ppk_state.height  # Use PPK ellipsoidal height as fallback
        print(f"[INIT] Using PPK ellipsoidal height: {self.msl0:.1f}m")
        
        # Load main data
        self.imu, self.imu_time_col = load_imu_csv(self.config.imu_path, return_time_col=True)
        # v3.9.0: Pass cam_timeref_csv for time_ref matching
        self.imgs = load_images(
            self.config.images_dir, 
            self.config.images_index_csv,
            self.config.timeref_csv
        )
        
        # v3.10.x: Offline VPS list path is deprecated in runtime mode
        # Keep as empty list for compatibility with legacy checks/logging.
        self.vps_list = []
        
        # v3.2.0: use_magnetometer is already populated from YAML (magnetometer.enabled)
        # No need for redundant check - VIOConfig.use_magnetometer is the final decision
        # v3.9.4: Pass timeref_csv to magnetometer for hardware clock synchronization
        self.mag_list = load_mag_csv(self.config.mag_csv, timeref_csv=self.config.timeref_csv) if self.config.use_magnetometer else []
        
        self.dem = DEMReader.open(self.config.dem_path)
        
        # Load PPK trajectory for error comparison
        self.ppk_trajectory = None
        if self.config.ground_truth_path:
            self.ppk_trajectory = load_ppk_trajectory(self.config.ground_truth_path)
        
        # Configure mapping from filter time base -> absolute epoch for GT lookup
        self._configure_error_time_mapping()

        # v3.9.12: Load Flight Log MSL for direct altitude updates (Barometer/GNSS)
        # This allows using "barometer" altitude instead of DEM+AGL logic
        self.msl_interpolator = load_flight_log_msl(
            self.config.quarry_path,
            self.config.timeref_csv
        )
        if self.msl_interpolator:
            # v3.9.13: Auto-Align Flight Log to Ground Truth (Plotting Logic)
            # The 3D plot aligns trajectories by subtracting the start value (z - z0).
            # We do the same here in real-time: Calculate offset at start and remove it.
            self.msl_offset = 0.0
            if self.ppk_state and self.ppk_state.timestamp:
                 # Find MSL from flight log at the same time as PPK start
                 t_start = self.msl_interpolator.times[0]
                 msl_log_start = self.msl_interpolator.get_msl(t_start)
                 
                 if msl_log_start is not None:
                     # Offset = FlightLog_Start - GT_Start
                     # This forces VIO to start at the exact same altitude as GT.
                     self.msl_offset = msl_log_start - self.ppk_state.height
                     print(f"[INIT] Auto-Aligned MSL to PPK: Offset {self.msl_offset:.3f}m")
                     print(f"       (Log: {msl_log_start:.3f}m, GT: {self.ppk_state.height:.3f}m)")
            else:
                 print(f"[INIT] Loaded Flight Log MSL (No Ground Truth for Alignment)")
        
        # Print summary
        print("=== Input check ===")
        print(f"IMU: {'OK' if len(self.imu)>0 else 'MISSING'} ({len(self.imu)} samples)")
        print(f"Images: {'OK' if len(self.imgs)>0 else 'None'} ({len(self.imgs)} frames)")
        print(f"Mag: {'OK' if len(self.mag_list)>0 else 'None'} ({len(self.mag_list)} samples)")
        print(f"DEM: {'OK' if self.dem.ds else 'None'}")
        print(f"PPK: {'OK' if self.ppk_state else 'None'}")
        
        if len(self.imu) == 0:
            raise RuntimeError("IMU is required. Aborting.")
    
    def _configure_error_time_mapping(self):
        """
        Configure timestamp mapping used by log_error().
        
        Goal:
        - If filter runs on time_ref, convert filter time to absolute epoch
          before matching with PPK stamp_log.
        - If filter already runs on absolute timestamps, use identity mapping.
        """
        self.error_time_scale = 1.0
        self.error_time_offset = 0.0
        self.error_time_mode = "identity"
        
        if self.imu_time_col != "time_ref":
            print(f"[TIME] Error/GT lookup uses direct filter time ({self.imu_time_col or 'unknown'})")
            return
        
        try:
            header_cols = pd.read_csv(self.config.imu_path, nrows=0).columns.tolist()
            abs_col = next((c for c in ["stamp_bag", "stamp_msg", "stamp_log"] if c in header_cols), None)
            if abs_col is None:
                print("[TIME] WARNING: IMU has time_ref but no absolute stamp column for GT mapping")
                return
            
            df = pd.read_csv(self.config.imu_path, usecols=["time_ref", abs_col])
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) < 2:
                print("[TIME] WARNING: Not enough IMU rows to fit time_ref->absolute mapping")
                return
            
            t_ref = df["time_ref"].astype(float).values
            t_abs = df[abs_col].astype(float).values
            scale, offset = np.polyfit(t_ref, t_abs, 1)
            pred = scale * t_ref + offset
            rms_ms = float(np.sqrt(np.mean((pred - t_abs) ** 2)) * 1000.0)
            
            self.error_time_scale = float(scale)
            self.error_time_offset = float(offset)
            self.error_time_mode = f"time_ref_to_{abs_col}"
            
            print(
                f"[TIME] GT mapping enabled: time_ref -> {abs_col} | "
                f"scale={self.error_time_scale:.12f}, offset={self.error_time_offset:.6f}, rms={rms_ms:.3f}ms"
            )
        except Exception as e:
            print(f"[TIME] WARNING: Failed to configure GT time mapping: {e}")
            self.error_time_scale = 1.0
            self.error_time_offset = 0.0
            self.error_time_mode = "identity"
    
    def _filter_time_to_gt_time(self, t_filter: float) -> float:
        """
        Convert filter timestamp to GT lookup timestamp (absolute epoch).
        """
        if self.error_time_mode.startswith("time_ref_to_"):
            return self.error_time_scale * t_filter + self.error_time_offset
        return t_filter
    
    def initialize_ekf(self):
        """Initialize EKF state and covariance."""
        # Ensure local projection is set up
        self.proj_cache.ensure_proj(self.lat0, self.lon0)
        
        # Sample DEM at origin
        self.dem0 = self.dem.sample_m(self.lat0, self.lon0) if self.dem.ds else None
        
        # v2.9.10.5: Check for manual AGL override in config (for helicopter flights)
        # Problem: DEM/GPS datum mismatch causes wrong AGL (e.g., 6m instead of 100m)
        # Solution: Allow config to override with reasonable helicopter cruise AGL
        vio_cfg = self.global_config.get('vio', {})
        agl_override = vio_cfg.get('initial_agl_override', None)
        
        if agl_override is not None and agl_override > 0:
            # Use manual override (GPS-denied compliant: fixed value from config)
            self.initial_agl = agl_override
            computed_agl = abs(self.msl0 - self.dem0) if self.dem0 is not None else 0.0
            print(f"[VIO] Initial AGL = {self.initial_agl:.1f}m (OVERRIDE, computed was {computed_agl:.1f}m)")
        elif self.dem0 is not None:
            # v2.9.10.4: Compute initial AGL for VIO velocity scale (GPS-denied compliant)
            # Uses only t=0 values: msl0 - dem0
            self.initial_agl = abs(self.msl0 - self.dem0)
            print(f"[VIO] Initial AGL = {self.initial_agl:.1f}m (MSL={self.msl0:.1f}, DEM={self.dem0:.1f})")
        else:
            self.initial_agl = 100.0  # Default fallback
            print(f"[VIO] WARNING: No DEM, using default initial_agl = {self.initial_agl:.1f}m")
        
        # Store in global config for VIO velocity update to access
        self.global_config['INITIAL_AGL'] = self.initial_agl
        
        # Create EKF (v3.9.7: 19D nominal state with mag_bias)
        self.kf = ExtendedKalmanFilter(dim_x=19, dim_z=3, dim_u=3)
        self.kf.x = np.zeros((19, 1), dtype=float)
        
        # Get config parameters from YAML (v3.2.0: no fallback - require proper config)
        imu_params = self.global_config.get('IMU_PARAMS')
        if imu_params is None:
            raise RuntimeError("IMU_PARAMS not found in config. Check YAML file.")
        
        self.lever_arm = self.global_config.get('IMU_GNSS_LEVER_ARM', np.zeros(3))
        
        # v3.9.10: Use PPK ATTITUDE yaw for initial heading (GPS-denied compliant)
        # NOTE: This is ATTITUDE yaw from PPK, NOT velocity heading!
        # Convert from NED frame (PPK file) to ENU frame (EKF uses ENU)
        # NED: 0 = North, positive = clockwise
        # ENU: 0 = East, positive = counter-clockwise
        # Conversion: yaw_enu = π/2 - yaw_ned
        ppk_initial_yaw = None
        if self.ppk_state is not None:
            yaw_ned = self.ppk_state.yaw  # Attitude yaw from PPK file (NED frame)
            yaw_enu = np.pi/2 - yaw_ned  # Convert to ENU frame
            yaw_enu = np.arctan2(np.sin(yaw_enu), np.cos(yaw_enu))  # Normalize to [-π, π]
            ppk_initial_yaw = yaw_enu
        
        # MAG params for initial correction
        # v2.9.10.12: Include hard_iron and soft_iron for proper initial calibration
        mag_params = None
        if self.config.use_magnetometer and len(self.mag_list) > 0:
            mag_params = {
                'declination': self.global_config.get('MAG_DECLINATION', 0.0),
                'use_raw_heading': self.global_config.get('MAG_USE_RAW_HEADING', True),
                'min_field': self.global_config.get('MAG_MIN_FIELD_STRENGTH', 0.1),
                'max_field': self.global_config.get('MAG_MAX_FIELD_STRENGTH', 100.0),
                'hard_iron': self.global_config.get('MAG_HARD_IRON_OFFSET', None),
                'soft_iron': self.global_config.get('MAG_SOFT_IRON_MATRIX', None),
                'sigma_mag_bias_init': self.global_config.get('SIGMA_MAG_BIAS_INIT', 0.1),  # v3.9.7
                'sigma_mag_bias': self.global_config.get('SIGMA_MAG_BIAS', 0.001),         # v3.9.7
                'use_estimated_bias': self.config.use_mag_estimated_bias,                  # v3.9.8
            }
        
        # Initialize state
        init_state = initialize_ekf_state(
            kf=self.kf,
            ppk_state=self.ppk_state,
            imu_records=self.imu,
            imu_params=imu_params,
            lever_arm=self.lever_arm,
            lat0=self.lat0, lon0=self.lon0,
            msl0=self.msl0, dem0=self.dem0,
            v_init_enu=self.v_init,
            estimate_imu_bias=self.config.estimate_imu_bias,
            initial_gyro_bias=self.global_config.get('INITIAL_GYRO_BIAS'),
            initial_accel_bias=self.global_config.get('INITIAL_ACCEL_BIAS'),
            mag_records=self.mag_list if mag_params else None,
            mag_params=mag_params,
            ppk_initial_yaw=ppk_initial_yaw  # PPK attitude yaw (ENU)
        )
        
        return init_state
    
    def initialize_vio_frontend(self):
        """Initialize VIO frontend if images are available."""
        if len(self.imgs) == 0:
            return
        
        kb_params = self.global_config.get('KB_PARAMS', {})
        use_fisheye = self.global_config.get('USE_FISHEYE', True)
        
        K, D = make_KD_for_size(kb_params, 
                                self.config.downscale_size[0],
                                self.config.downscale_size[1])
        
        self.vio_fe = VIOFrontEnd(
            self.config.downscale_size[0],
            self.config.downscale_size[1],
            K, D,
            use_fisheye=use_fisheye,
            fast_mode=self.config.fast_mode  # v2.9.9: Performance optimization
        )
        
        # Override min_track_length from config (v3.9.1)
        self.vio_fe.min_track_length = self.global_config.get('MSCKF_MIN_TRACK_LENGTH', 4)
        
        print(f"[VIO] Camera view mode: {self.config.camera_view}")
        print(f"[MSCKF] max_clone_size={self.global_config.get('MSCKF_MAX_CLONE_SIZE', 11)}, "
              f"min_track_length={self.vio_fe.min_track_length}")
        
        # Initialize plane detector if enabled
        if self.global_config.get('USE_PLANE_MSCKF', False):
            self.plane_detector = PlaneDetector(
                min_points_per_plane=self.global_config.get('PLANE_MIN_POINTS', 10),
                normal_angle_threshold=self.global_config.get('PLANE_ANGLE_THRESHOLD', np.radians(15.0)),
                distance_threshold=self.global_config.get('PLANE_DISTANCE_THRESHOLD', 0.15),
                min_plane_area=self.global_config.get('PLANE_MIN_AREA', 0.5)
            )
            print(f"[Plane-MSCKF] Enabled with min_points={self.plane_detector.min_points}, "
                  f"angle_thresh={np.degrees(self.plane_detector.angle_threshold):.1f}°")
    
    def _initialize_rectifier(self):
        """
        Initialize fisheye rectifier for converting fisheye images to pinhole.
        
        This is optional and can improve feature tracking by removing 
        fisheye distortion before processing. The rectified images use
        a virtual pinhole camera with specified FOV.
        """
        # Check if rectification is enabled in config
        use_rectifier = self.global_config.get('USE_RECTIFIER', False)
        if not use_rectifier:
            print("[RECTIFY] Fisheye rectification disabled")
            return
        
        if len(self.imgs) == 0:
            return
        
        try:
            rectify_fov = self.global_config.get('RECTIFY_FOV_DEG', 90.0)
            self.rectifier = create_rectifier_from_config(
                self.global_config,
                src_size=self.config.downscale_size,
                fov_deg=rectify_fov,
                dst_size=self.config.downscale_size
            )
            print(f"[RECTIFY] Fisheye rectifier initialized (FOV={rectify_fov}°)")
        except Exception as e:
            print(f"[RECTIFY] WARNING: Failed to initialize rectifier: {e}")
            self.rectifier = None
    
    def _initialize_loop_closure(self):
        """
        Initialize loop closure detector for yaw drift correction.
        
        Loop closure detects when the vehicle returns to a previously
        visited location and corrects accumulated yaw drift by matching
        visual features.
        """
        use_loop_closure = self.global_config.get('USE_LOOP_CLOSURE', True)
        if not use_loop_closure:
            print("[LOOP] Loop closure detection disabled")
            return
        
        if len(self.imgs) == 0:
            print("[LOOP] No images available, loop closure disabled")
            return
        
        try:
            # Use tuned parameters for helicopter operations
            position_threshold = self.global_config.get('LOOP_POSITION_THRESHOLD', 30.0)
            self.loop_detector = init_loop_closure(position_threshold=position_threshold)
            print(f"[LOOP] Loop closure detector initialized (threshold={position_threshold}m)")
        except Exception as e:
            print(f"[LOOP] WARNING: Failed to initialize loop closure: {e}")
            self.loop_detector = None
    
    def setup_output_files(self):
        """
        Create output directory and all CSV files.
        
        Uses output_utils.init_output_csvs() and DebugCSVWriters.
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Use output_utils to initialize core CSVs
        csv_paths = init_output_csvs(self.config.output_dir)
        self.pose_csv = csv_paths['pose_csv']
        self.error_csv = csv_paths['error_csv']
        self.state_dbg_csv = csv_paths['state_dbg_csv']
        self.time_sync_csv = csv_paths.get('time_sync_csv')
        self.cov_health_csv = csv_paths.get('cov_health_csv')
        self.adaptive_debug_csv = csv_paths.get('adaptive_debug_csv')
        self.sensor_health_csv = csv_paths.get('sensor_health_csv')
        self.conditioning_events_csv = csv_paths.get('conditioning_events_csv')
        self.benchmark_health_summary_csv = csv_paths.get('benchmark_health_summary_csv')
        self.inf_csv = csv_paths['inf_csv']
        self.vo_dbg_csv = csv_paths['vo_dbg']
        self.msckf_dbg_csv = csv_paths['msckf_dbg']
        
        if self.kf is not None and hasattr(self.kf, "enable_cov_health_logging") and self.cov_health_csv:
            self.kf.enable_cov_health_logging(self.cov_health_csv)
        if self.kf is not None and hasattr(self.kf, "enable_conditioning_event_logging") and self.conditioning_events_csv:
            self.kf.enable_conditioning_event_logging(self.conditioning_events_csv)
        
        # Use DebugCSVWriters for optional debug files
        self.debug_writers = DebugCSVWriters(self.config.output_dir, self.config.save_debug_data)
        self.imu_raw_csv = self.debug_writers.imu_raw_csv
        self.state_cov_csv = self.debug_writers.state_cov_csv
        self.residual_csv = self.debug_writers.residual_csv
        self.feature_stats_csv = self.debug_writers.feature_stats_csv
        self.msckf_window_csv = self.debug_writers.msckf_window_csv
        self.fej_csv = self.debug_writers.fej_consistency_csv
        
        self.keyframe_dir = None
        if self.config.save_debug_data:
            print(f"[DEBUG] Debug data logging enabled")
        
        if self.config.save_keyframe_images:
            self.keyframe_dir = os.path.join(self.config.output_dir, "debug_keyframes")
            os.makedirs(self.keyframe_dir, exist_ok=True)
            print(f"[DEBUG] Keyframe images will be saved to: {self.keyframe_dir}")
            
            # VPS match visualization directory
            self.vps_matches_dir = os.path.join(self.config.output_dir, "debug_vps_matches")
            os.makedirs(self.vps_matches_dir, exist_ok=True)
            print(f"[DEBUG] VPS match images will be saved to: {self.vps_matches_dir}")
        else:
            self.vps_matches_dir = None
        
        # VPS Debug Logger (for future VPSRunner integration)
        # When VPSRunner is integrated, attach logger via: vps_runner.set_logger(self.vps_logger)
        self.vps_logger = None
        if self.config.save_debug_data:
            # Check if VPS is enabled in YAML config (nested structure)
            vps_cfg = {}
            if hasattr(self.config, '_yaml_config') and self.config._yaml_config:
                vps_cfg = self.config._yaml_config.get('vps', {})
            if vps_cfg.get('enabled', False):
                try:
                    from vps import VPSDebugLogger
                    self.vps_logger = VPSDebugLogger(
                        output_dir=self.config.output_dir,
                        enabled=True
                    )
                    print(f"[DEBUG] VPS logger enabled (debug_vps_attempts.csv, debug_vps_matches.csv)")
                except ImportError:
                    print(f"[WARNING] VPS module not available, VPS logging disabled")
        
        # VPS Real-time Runner (for live VPS processing)
        self.vps_runner = None
        # Use _yaml_config (original YAML) not global_config (flat dict)
        vps_cfg = {}
        if hasattr(self.config, '_yaml_config') and self.config._yaml_config:
            vps_cfg = self.config._yaml_config.get('vps', {})
        
        if vps_cfg.get('enabled', False):
            try:
                from vps import VPSRunner
                
                # Get MBTiles path
                # Priority: 1. CLI (config.mbtiles_path), 2. YAML (vps.mbtiles_path), 3. Default
                mbtiles_path = self.config.mbtiles_path  # From CLI
                if not mbtiles_path:
                    # Fallback to YAML config
                    mbtiles_path = vps_cfg.get('mbtiles_path', 'mission.mbtiles')
                
                if os.path.exists(mbtiles_path):
                    self.vps_runner = VPSRunner.create_from_config(
                        mbtiles_path=mbtiles_path,
                        config_path=self.config.config_yaml,
                        device=vps_cfg.get('device', 'cpu')
                    )
                    
                    # Inject logger (Dependency Injection!)
                    if self.vps_logger is not None:
                        self.vps_runner.set_logger(self.vps_logger)
                        print(f"[VPS] Real-time VPS enabled with logging")
                    else:
                        print(f"[VPS] Real-time VPS enabled with logging")
                    
                    # Set match visualization directory
                    if hasattr(self, 'vps_matches_dir') and self.vps_matches_dir:
                        self.vps_runner.save_matches_dir = self.vps_matches_dir
                        print(f"[VPS] Match visualizations will be saved")
                    
                    # Initialize VPS delayed update manager (stochastic cloning)
                    # CRITICAL: Must be created HERE, not lazily, so clone_state works
                    from vps import VPSDelayedUpdateManager
                    vps_delay_cfg = vps_cfg.get('delayed_update', {})
                    self.vps_clone_manager = VPSDelayedUpdateManager(
                        max_delay_sec=vps_delay_cfg.get('max_delay_sec', 0.5),
                        max_clones=vps_delay_cfg.get('max_clones', 3)
                    )
                    # Tell VPSRunner that stochastic cloning is active for logging
                    self.vps_runner.delayed_update_enabled = True
                    print(f"[VPS] Delayed update manager initialized (stochastic cloning enabled)")
                else:
                    print(f"[WARNING] VPS enabled but MBTiles not found: {mbtiles_path}")
                    print(f"          Please create MBTiles using: python -m vps.tile_prefetcher")
            except ImportError as e:
                print(f"[WARNING] VPS module not available: {e}")
        
        # Save calibration snapshot using output_utils
        if self.config.save_debug_data:
            cal_path = os.path.join(self.config.output_dir, "debug_calibration.txt")
            
            # Use helper function to build all calibration params
            cal_params = build_calibration_params(
                global_config=self.global_config,
                vio_config=self.config,
                lat0=self.lat0,
                lon0=self.lon0,
                alt0=getattr(self, 'alt0', 0.0)
            )
            
            save_calibration_log(output_path=cal_path, **cal_params)
            print(f"[DEBUG] Calibration snapshot saved: {cal_path}")

    def initialize_adaptive_controller(self):
        """Initialize adaptive controller from compiled YAML config."""
        adaptive_cfg = {}
        if isinstance(self.global_config, dict):
            adaptive_cfg = self.global_config.get("ADAPTIVE", {})
        if self.config.estimator_mode != "imu_step_preint_cache":
            adaptive_cfg = dict(adaptive_cfg)
            adaptive_cfg["mode"] = "off"
        try:
            self.adaptive_controller = AdaptiveController(adaptive_cfg)
        except Exception as exc:
            print(f"[ADAPTIVE] WARNING: failed to initialize controller: {exc}")
            self.adaptive_controller = AdaptiveController({"mode": "off"})

        self.current_adaptive_decision = self.adaptive_controller.last_decision
        self._adaptive_prev_pmax = None
        self._adaptive_last_aiding_time = None
        self._adaptive_last_policy_t = None
        self._conditioning_warning_sec = 0.0
        self._adaptive_log_enabled = bool(
            adaptive_cfg.get("logging", {}).get("enabled", True)
        )
        mode = self.current_adaptive_decision.mode if self.current_adaptive_decision else "off"
        print(f"[ADAPTIVE] mode={mode} (IMU-driven policy)")

    def _get_sensor_adaptive_scales(self, sensor: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get policy scales and applied scales for a sensor."""
        default = {
            "r_scale": 1.0,
            "chi2_scale": 1.0,
            "threshold_scale": 1.0,
            "reproj_scale": 1.0,
            "acc_threshold_scale": 1.0,
            "gyro_threshold_scale": 1.0,
            "max_v_scale": 1.0,
            "fail_soft_enable": 0.0,
            "hard_reject_factor": 3.0,
            "soft_r_cap": 20.0,
            "soft_r_power": 1.0,
            "sigma_deg": 7.0,
            "acc_norm_tolerance": 0.25,
            "max_gyro_rad_s": 0.40,
            "enabled_imu_only": 1.0,
            "ref_alpha": 0.005,
            "dynamic_ref_alpha": 0.05,
            "period_steps": 8.0,
            "sigma_bg_rad_s": np.deg2rad(0.30),
            "sigma_ba_m_s2": 0.15,
            "enable_when_no_aiding": 1.0,
            "enable_when_partial_aiding": 0.0,
            "enable_when_full_aiding": 0.0,
            "max_bg_norm_rad_s": 0.20,
            "max_ba_norm_m_s2": 2.5,
        }
        decision = self.current_adaptive_decision
        if decision is None:
            return dict(default), dict(default)
        policy = decision.sensor_scale(sensor)
        applied = dict(policy) if decision.apply_measurement else dict(default)
        return policy, applied

    def update_adaptive_policy(self, t: float, phase: int) -> Optional[AdaptiveDecision]:
        """Evaluate adaptive controller for current filter health snapshot."""
        if self.adaptive_controller is None or self.kf is None:
            return None

        try:
            p = self.kf.P
            p_max = float(np.max(np.abs(p)))
            p_trace = float(np.trace(p))
            core_dim = min(18, p.shape[0])
            p_cond = float(np.linalg.cond(p[:core_dim, :core_dim]))
        except Exception:
            p_max = float("nan")
            p_trace = float("nan")
            p_cond = float("inf")

        if self._adaptive_prev_pmax is None or not np.isfinite(self._adaptive_prev_pmax) or self._adaptive_prev_pmax <= 0:
            growth_ratio = 1.0
        else:
            growth_ratio = p_max / self._adaptive_prev_pmax if np.isfinite(p_max) else float("nan")
        self._adaptive_prev_pmax = p_max if np.isfinite(p_max) else self._adaptive_prev_pmax

        if self._adaptive_last_aiding_time is None:
            aiding_age_sec = 1e9
        else:
            aiding_age_sec = max(0.0, float(t - self._adaptive_last_aiding_time))

        ctx = AdaptiveContext(
            timestamp=float(t),
            phase=int(phase),
            p_cond=float(p_cond),
            p_max=float(p_max),
            p_trace=float(p_trace),
            p_growth_ratio=float(growth_ratio) if np.isfinite(growth_ratio) else float("nan"),
            aiding_age_sec=float(aiding_age_sec),
        )
        decision = self.adaptive_controller.step(ctx)
        self.current_adaptive_decision = decision

        if self._adaptive_last_policy_t is None:
            dt_policy = 0.0
        else:
            dt_policy = max(0.0, float(t - self._adaptive_last_policy_t))
        self._adaptive_last_policy_t = float(t)
        if decision.health_state in ("WARNING", "DEGRADED"):
            self._conditioning_warning_sec += dt_policy
        else:
            self._conditioning_warning_sec = 0.0

        # Keep legacy cap for off/shadow to guarantee backward compatibility.
        cov_cap = self._legacy_covariance_cap
        if decision.apply_process_noise:
            cov_cap = float(decision.conditioning_max_value)
        if hasattr(self.kf, "set_covariance_max_value"):
            self.kf.set_covariance_max_value(cov_cap)
        if hasattr(self.kf, "conditioning_cond_hard"):
            self.kf.conditioning_cond_hard = float(decision.conditioning_cond_hard)
        if hasattr(self.kf, "conditioning_cond_hard_window"):
            self.kf.conditioning_cond_hard_window = int(max(1, decision.conditioning_cond_hard_window))
        if hasattr(self.kf, "conditioning_projection_min_interval_steps"):
            self.kf.conditioning_projection_min_interval_steps = int(
                max(1, decision.conditioning_projection_min_interval_steps)
            )

        if self.adaptive_debug_csv and self._adaptive_log_enabled:
            log_adaptive_decision(
                self.adaptive_debug_csv,
                t=float(t),
                mode=decision.mode,
                health_state=decision.health_state,
                phase=int(phase),
                aiding_age_sec=float(aiding_age_sec),
                p_max=float(p_max),
                p_cond=float(p_cond),
                p_growth_ratio=float(growth_ratio) if np.isfinite(growth_ratio) else np.nan,
                sigma_accel_scale=float(decision.sigma_accel_scale),
                gyr_w_scale=float(decision.gyr_w_scale),
                acc_w_scale=float(decision.acc_w_scale),
                sigma_unmodeled_gyr_scale=float(decision.sigma_unmodeled_gyr_scale),
                min_yaw_scale=float(decision.min_yaw_scale),
                conditioning_max_value=float(cov_cap),
                reason=decision.reason,
            )

        return decision

    def build_step_imu_params(self, imu_params_base: dict, sigma_accel_base: float) -> Tuple[dict, float]:
        """Build per-step IMU params (active mode only)."""
        imu_params_step = dict(imu_params_base)
        sigma_accel_step = float(sigma_accel_base)

        decision = self.current_adaptive_decision
        if decision is None or not decision.apply_process_noise:
            return imu_params_step, sigma_accel_step

        sigma_accel_step *= float(decision.sigma_accel_scale)
        if "gyr_w" in imu_params_step:
            imu_params_step["gyr_w"] = float(imu_params_step["gyr_w"]) * float(decision.gyr_w_scale)
        if "acc_w" in imu_params_step:
            imu_params_step["acc_w"] = float(imu_params_step["acc_w"]) * float(decision.acc_w_scale)
        if "sigma_unmodeled_gyr" in imu_params_step:
            imu_params_step["sigma_unmodeled_gyr"] = (
                float(imu_params_step["sigma_unmodeled_gyr"]) * float(decision.sigma_unmodeled_gyr_scale)
            )
        if "min_yaw_process_noise_deg" in imu_params_step:
            min_yaw = float(imu_params_step["min_yaw_process_noise_deg"]) * float(decision.min_yaw_scale)
            imu_params_step["min_yaw_process_noise_deg"] = max(float(decision.min_yaw_floor_deg), min_yaw)

        return imu_params_step, sigma_accel_step

    def record_adaptive_measurement(self,
                                    sensor: str,
                                    adaptive_info: Optional[Dict[str, Any]],
                                    timestamp: float,
                                    policy_scales: Optional[Dict[str, float]] = None,
                                    counts_as_aiding: bool = True):
        """Feed measurement acceptance/NIS back into adaptive controller."""
        if self.adaptive_controller is None or adaptive_info is None or len(adaptive_info) == 0:
            return
        attempted = int(adaptive_info.get("attempted", 1))
        if attempted <= 0:
            return

        accepted = bool(adaptive_info.get("accepted", False))
        nis_norm = adaptive_info.get("nis_norm", np.nan)
        nis_norm = float(nis_norm) if nis_norm is not None and np.isfinite(nis_norm) else None

        feedback = self.adaptive_controller.record_measurement(
            sensor=sensor,
            accepted=accepted,
            nis_norm=nis_norm,
            timestamp=float(timestamp),
        )
        if accepted and counts_as_aiding:
            self._adaptive_last_aiding_time = float(timestamp)

        decision = self.current_adaptive_decision
        mode = decision.mode if decision is not None else "off"
        health_state = decision.health_state if decision is not None else "HEALTHY"
        if policy_scales is None and decision is not None:
            policy_scales = decision.sensor_scale(sensor)
        if policy_scales is None:
            policy_scales = {"r_scale": 1.0, "chi2_scale": 1.0, "threshold_scale": 1.0, "reproj_scale": 1.0}

        if self.sensor_health_csv and self._adaptive_log_enabled:
            log_sensor_health(
                self.sensor_health_csv,
                t=float(timestamp),
                sensor=sensor,
                accepted=accepted,
                nis_norm=nis_norm,
                nis_ewma=float(feedback.get("nis_ewma", 1.0)),
                accept_rate=float(feedback.get("accept_rate", 1.0)),
                mode=mode,
                health_state=health_state,
                r_scale=float(policy_scales.get("r_scale", 1.0)),
                chi2_scale=float(policy_scales.get("chi2_scale", 1.0)),
                threshold_scale=float(policy_scales.get("threshold_scale", 1.0)),
                reproj_scale=float(policy_scales.get("reproj_scale", 1.0)),
                reason_code=str(adaptive_info.get("reason_code", "")),
            )
    
    def _update_imu_helpers(self, rec, dt: float, imu_params: dict,
                            zupt_scales: Optional[Dict[str, float]] = None):
        """
        Update IMU-related helper variables (ZUPT, mag filter).
        Shared by both preintegration and legacy modes.
        
        Args:
            rec: IMU record
            dt: Time delta since last sample
            imu_params: IMU parameters dict
        """
        # Get current state
        x = self.kf.x.reshape(-1)
        bg = x[10:13]
        ba = x[13:16]
        
        # Bias-corrected measurements
        a_corr = rec.lin.astype(float) - ba
        w_corr = rec.ang.astype(float) - bg
        
        # Store gyro_z and dt for mag filtering (CRITICAL for preintegration mode)
        self.last_gyro_z = float(w_corr[2])
        self.last_imu_dt = dt
        
        # Check for stationary (phase-aware thresholds for ZUPT gating)
        v_mag = np.linalg.norm(x[3:6])
        zupt_enabled = bool(self.global_config.get("ZUPT_ENABLED", True))
        base_acc_threshold = float(self.global_config.get("ZUPT_ACCEL_THRESHOLD", 0.5))
        base_gyro_threshold = float(self.global_config.get("ZUPT_GYRO_THRESHOLD", 0.05))
        base_max_v_for_zupt = float(self.global_config.get("ZUPT_MAX_V_FOR_UPDATE", 20.0))
        zupt_policy_scales, zupt_apply_scales = self._get_sensor_adaptive_scales("ZUPT")
        if zupt_scales is not None:
            zupt_apply_scales = dict(zupt_apply_scales)
            zupt_apply_scales.update(zupt_scales)
        acc_threshold = base_acc_threshold * float(zupt_apply_scales.get("acc_threshold_scale", 1.0))
        gyro_threshold = base_gyro_threshold * float(zupt_apply_scales.get("gyro_threshold_scale", 1.0))

        is_stationary, _ = detect_stationary(
            a_raw=rec.lin,
            w_corr=w_corr,
            v_mag=v_mag,
            imu_params=imu_params,
            acc_threshold=acc_threshold,
            gyro_threshold=gyro_threshold,
        )
        
        if zupt_enabled and is_stationary:
            self.state.zupt_detected += 1
            zupt_adaptive_info: Dict[str, Any] = {}
            
            applied, v_reduction, updated_count = apply_zupt(
                self.kf,
                v_mag=v_mag,
                consecutive_stationary_count=self.state.consecutive_stationary,
                max_v_for_zupt=base_max_v_for_zupt * float(zupt_apply_scales.get("max_v_scale", 1.0)),
                save_debug=self.config.save_debug_data,
                residual_csv=getattr(self, 'residual_csv', None),
                timestamp=rec.t,
                frame=self.state.imu_propagation_count,
                r_scale=float(zupt_apply_scales.get("r_scale", 1.0)),
                chi2_scale=float(zupt_apply_scales.get("chi2_scale", 1.0)),
                soft_fail_enable=bool(float(zupt_apply_scales.get("fail_soft_enable", 0.0)) >= 0.5),
                soft_fail_r_cap=float(zupt_apply_scales.get("soft_r_cap", 20.0)),
                soft_fail_hard_reject_factor=float(zupt_apply_scales.get("hard_reject_factor", 3.0)),
                soft_fail_power=float(zupt_apply_scales.get("soft_r_power", 1.0)),
                adaptive_info=zupt_adaptive_info,
            )
            self.record_adaptive_measurement(
                "ZUPT",
                adaptive_info=zupt_adaptive_info,
                timestamp=rec.t,
                policy_scales=zupt_policy_scales,
                counts_as_aiding=False,
            )
            
            if applied:
                self.state.zupt_applied += 1
                self.state.consecutive_stationary = updated_count
            else:
                self.state.zupt_rejected += 1
        else:
            self.state.consecutive_stationary = 0

        # IMU-only stabilization stack: gravity RP, weak yaw aid, bias guard.
        if getattr(self, "imu_only_mode", False):
            if self._imu_only_yaw_ref is None:
                self._imu_only_yaw_ref = float(quaternion_to_yaw(self.kf.x[6:10, 0].reshape(4,)))
            if self._imu_only_bg_ref is None:
                self._imu_only_bg_ref = self.kf.x[10:13, 0].astype(float).copy()
            if self._imu_only_ba_ref is None:
                self._imu_only_ba_ref = self.kf.x[13:16, 0].astype(float).copy()
            warning_backoff_active = bool(self._conditioning_warning_sec >= 1.5)
            warning_period_mult = 2 if warning_backoff_active else 1

            grav_policy_scales, grav_apply_scales = self._get_sensor_adaptive_scales("GRAVITY_RP")
            grav_period_steps = max(1, int(round(float(grav_apply_scales.get("period_steps", 1.0))))) * warning_period_mult
            if (
                bool(float(grav_apply_scales.get("enabled_imu_only", 1.0)) >= 0.5)
                and (self.state.imu_propagation_count % grav_period_steps == 0)
            ):
                grav_adaptive_info: Dict[str, Any] = {}
                apply_gravity_roll_pitch_update(
                    self.kf,
                    a_raw=rec.lin.astype(float),
                    w_corr=w_corr,
                    imu_params=imu_params,
                    sigma_rad=np.deg2rad(float(grav_apply_scales.get("sigma_deg", 7.0))),
                    r_scale=float(grav_apply_scales.get("r_scale", 1.0)),
                    chi2_scale=float(grav_apply_scales.get("chi2_scale", 1.0)),
                    acc_norm_tolerance=float(grav_apply_scales.get("acc_norm_tolerance", 0.25)),
                    max_gyro_rad_s=float(grav_apply_scales.get("max_gyro_rad_s", 0.40)),
                    save_debug=self.config.save_debug_data,
                    residual_csv=getattr(self, 'residual_csv', None),
                    timestamp=rec.t,
                    frame=self.state.imu_propagation_count,
                    adaptive_info=grav_adaptive_info,
                )
                self.record_adaptive_measurement(
                    "GRAVITY_RP",
                    adaptive_info=grav_adaptive_info,
                    timestamp=rec.t,
                    policy_scales=grav_policy_scales,
                    counts_as_aiding=False,
                )

            yaw_policy_scales, yaw_apply_scales = self._get_sensor_adaptive_scales("YAW_AID")
            yaw_period_steps = max(1, int(round(float(yaw_apply_scales.get("period_steps", 8.0))))) * warning_period_mult
            speed_ms_now = float(v_mag)
            if speed_ms_now > float(yaw_apply_scales.get("high_speed_m_s", 1e9)):
                yaw_period_steps = int(
                    max(1, round(yaw_period_steps * float(yaw_apply_scales.get("high_speed_period_mult", 1.0))))
                )
            if (
                bool(float(yaw_apply_scales.get("enabled_imu_only", 1.0)) >= 0.5)
                and (self.state.imu_propagation_count % yaw_period_steps == 0)
            ):
                yaw_sigma_deg = float(yaw_apply_scales.get("sigma_deg", 35.0))
                if speed_ms_now > float(yaw_apply_scales.get("high_speed_m_s", 1e9)):
                    yaw_sigma_deg *= float(yaw_apply_scales.get("high_speed_sigma_mult", 1.0))
                yaw_adaptive_info: Dict[str, Any] = {}
                apply_yaw_pseudo_update(
                    self.kf,
                    a_raw=rec.lin.astype(float),
                    w_corr=w_corr,
                    imu_params=imu_params,
                    yaw_ref_rad=float(self._imu_only_yaw_ref),
                    sigma_rad=np.deg2rad(yaw_sigma_deg),
                    r_scale=float(yaw_apply_scales.get("r_scale", 1.0)),
                    chi2_scale=float(yaw_apply_scales.get("chi2_scale", 1.0)),
                    acc_norm_tolerance=float(yaw_apply_scales.get("acc_norm_tolerance", 0.25)),
                    max_gyro_rad_s=float(yaw_apply_scales.get("max_gyro_rad_s", 0.40)),
                    soft_fail_enable=bool(float(yaw_apply_scales.get("fail_soft_enable", 0.0)) >= 0.5),
                    soft_fail_r_cap=float(yaw_apply_scales.get("soft_r_cap", 12.0)),
                    soft_fail_hard_reject_factor=float(yaw_apply_scales.get("hard_reject_factor", 3.0)),
                    soft_fail_power=float(yaw_apply_scales.get("soft_r_power", 1.0)),
                    save_debug=self.config.save_debug_data,
                    residual_csv=getattr(self, 'residual_csv', None),
                    timestamp=rec.t,
                    frame=self.state.imu_propagation_count,
                    adaptive_info=yaw_adaptive_info,
                )
                self.record_adaptive_measurement(
                    "YAW_AID",
                    adaptive_info=yaw_adaptive_info,
                    timestamp=rec.t,
                    policy_scales=yaw_policy_scales,
                    counts_as_aiding=False,
                )
                yaw_now = float(quaternion_to_yaw(self.kf.x[6:10, 0].reshape(4,)))
                attempted = int(yaw_adaptive_info.get("attempted", 1))
                if attempted <= 0:
                    yaw_alpha = float(yaw_apply_scales.get("dynamic_ref_alpha", 0.05))
                else:
                    yaw_alpha = float(yaw_apply_scales.get("ref_alpha", 0.005))
                yaw_alpha = max(0.0, min(1.0, yaw_alpha))
                self._imu_only_yaw_ref = float(np.arctan2(
                    np.sin((1.0 - yaw_alpha) * float(self._imu_only_yaw_ref) + yaw_alpha * yaw_now),
                    np.cos((1.0 - yaw_alpha) * float(self._imu_only_yaw_ref) + yaw_alpha * yaw_now),
                ))

            bias_policy_scales, bias_apply_scales = self._get_sensor_adaptive_scales("BIAS_GUARD")
            bias_enabled = bool(float(bias_apply_scales.get("enabled_imu_only", 1.0)) >= 0.5)
            period_steps = max(1, int(round(float(bias_apply_scales.get("period_steps", 8.0)))))
            decision = self.current_adaptive_decision
            aiding_level = decision.aiding_level if decision is not None else "FULL"
            health_state = decision.health_state if decision is not None else "HEALTHY"
            phase_now = int(self.state.current_phase)
            phase_period_mult = 2 if phase_now >= 1 else 1
            health_period_mult = 2 if health_state in ("WARNING", "DEGRADED") else 1
            period_steps = period_steps * phase_period_mult * health_period_mult * warning_period_mult
            if speed_ms_now > float(bias_apply_scales.get("high_speed_m_s", 1e9)):
                period_steps = int(
                    max(1, round(period_steps * float(bias_apply_scales.get("high_speed_period_mult", 1.0))))
                )
            allow_level = (
                (aiding_level == "NONE" and float(bias_apply_scales.get("enable_when_no_aiding", 1.0)) >= 0.5)
                or (aiding_level == "PARTIAL" and float(bias_apply_scales.get("enable_when_partial_aiding", 0.0)) >= 0.5)
                or (aiding_level == "FULL" and float(bias_apply_scales.get("enable_when_full_aiding", 0.0)) >= 0.5)
            )
            g_norm = float(imu_params.get("g_norm", 9.80665))
            accel_includes_gravity = bool(imu_params.get("accel_includes_gravity", True))
            expected_acc_mag = g_norm if accel_includes_gravity else 0.0
            acc_dev = abs(float(np.linalg.norm(rec.lin.astype(float))) - expected_acc_mag)
            gyro_mag = float(np.linalg.norm(w_corr))
            low_dynamic_for_bias = (
                acc_dev <= float(bias_apply_scales.get("acc_norm_tolerance", 0.15))
                and gyro_mag <= float(bias_apply_scales.get("max_gyro_rad_s", 0.25))
            )
            if bias_enabled and allow_level and low_dynamic_for_bias and (self.state.imu_propagation_count % period_steps == 0):
                bias_adaptive_info: Dict[str, Any] = {}
                apply_bias_observability_guard(
                    self.kf,
                    bg_ref=self._imu_only_bg_ref,
                    ba_ref=self._imu_only_ba_ref,
                    sigma_bg_rad_s=float(bias_apply_scales.get("sigma_bg_rad_s", np.deg2rad(0.30))),
                    sigma_ba_m_s2=float(bias_apply_scales.get("sigma_ba_m_s2", 0.15)),
                    r_scale=float(bias_apply_scales.get("r_scale", 1.0)),
                    chi2_scale=float(bias_apply_scales.get("chi2_scale", 1.0)),
                    soft_fail_enable=bool(float(bias_apply_scales.get("fail_soft_enable", 0.0)) >= 0.5),
                    soft_fail_r_cap=float(bias_apply_scales.get("soft_r_cap", 8.0)),
                    soft_fail_hard_reject_factor=float(bias_apply_scales.get("hard_reject_factor", 4.0)),
                    soft_fail_power=float(bias_apply_scales.get("soft_r_power", 1.0)),
                    max_bg_norm_rad_s=float(bias_apply_scales.get("max_bg_norm_rad_s", 0.20)),
                    max_ba_norm_m_s2=float(bias_apply_scales.get("max_ba_norm_m_s2", 2.5)),
                    save_debug=self.config.save_debug_data,
                    residual_csv=getattr(self, 'residual_csv', None),
                    timestamp=rec.t,
                    frame=self.state.imu_propagation_count,
                    adaptive_info=bias_adaptive_info,
                )
                self.record_adaptive_measurement(
                    "BIAS_GUARD",
                    adaptive_info=bias_adaptive_info,
                    timestamp=rec.t,
                    policy_scales=bias_policy_scales,
                    counts_as_aiding=False,
                )
                if bool(bias_adaptive_info.get("accepted", False)):
                    bg_now = self.kf.x[10:13, 0].astype(float)
                    ba_now = self.kf.x[13:16, 0].astype(float)
                    self._imu_only_bg_ref = 0.99 * self._imu_only_bg_ref + 0.01 * bg_now
                    self._imu_only_ba_ref = 0.99 * self._imu_only_ba_ref + 0.01 * ba_now
        
        # Save priors
        self.kf.x_prior = self.kf.x.copy()
        self.kf.P_prior = self.kf.P.copy()
    
    def _log_time_sync_debug(self, t_filter: float, t_gt_mapped: float,
                             dt_gt: float, gt_idx: int,
                             gt_stamp_log: float, matched: int):
        """Append one per-frame time sync debug row."""
        if not self.time_sync_csv:
            return
        try:
            with open(self.time_sync_csv, "a", newline="") as f:
                f.write(
                    f"{t_filter:.6f},{t_gt_mapped:.6f},{dt_gt:.6f},{gt_idx},"
                    f"{gt_stamp_log:.6f},{matched},{self.error_time_mode}\n"
                )
        except Exception:
            pass

    def process_vio(self, rec, t: float, ongoing_preint=None):
        """
        Process VIO (Visual-Inertial Odometry) + VPS updates.
        
        This handles:
        1. Image loading and preprocessing
        2. VPS real-time processing (satellite matching)
        3. Feature tracking (VIO frontend)
        4. Loop closure detection (optional)
        5. Preintegration application at camera frame
        6. Camera cloning for MSCKF
        7. MSCKF multi-view updates
        8. VIO velocity updates with scale recovery
        9. Plane constraint updates
        
        Args:
            rec: Current IMU record (for rotation rate check)
            t: Current timestamp
            ongoing_preint: Preintegration buffer (if enabled)
            
        Returns:
            Tuple of (used_vo, vo_data dict)
        """
        # Get configuration
        kb_params = self.global_config.get('KB_PARAMS', {'mu': 600, 'mv': 600})
        min_parallax = self.global_config.get('MIN_PARALLAX_PX', 2.0)
        imu_params = self.global_config.get('IMU_PARAMS', {})
        
        used_vo = False
        vo_data = None
        vo_dx = vo_dy = vo_dz = vo_r = vo_p = vo_y = np.nan
        
        if self.vio_fe is None:
            return used_vo, vo_data
        
        # Check for fast rotation - skip VIO during aggressive maneuvers
        rotation_rate_deg_s = np.linalg.norm(rec.ang) * 180.0 / np.pi
        is_fast_rotation = rotation_rate_deg_s > 30.0
        
        # Process images up to current time
        while self.state.img_idx < len(self.imgs) and self.imgs[self.state.img_idx].t <= t:
            # Get camera timestamp (CRITICAL: use this instead of IMU time t)
            # Prevents timestamp mismatch when cloning/resetting preintegration
            t_cam = self.imgs[self.state.img_idx].t
            
            # FRAME SKIP (v2.9.9): Process every N frames for speedup
            # frame_skip=1 → all frames, frame_skip=2 → every other frame (50% faster)
            if self.config.frame_skip > 1 and (self.state.img_idx % self.config.frame_skip) != 0:
                self.state.img_idx += 1
                continue
            
            img_path = self.imgs[self.state.img_idx].path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                self.state.img_idx += 1
                continue
            
            # Resize if needed
            if (img.shape[1], img.shape[0]) != tuple(self.config.downscale_size):
                img = cv2.resize(img, self.config.downscale_size, interpolation=cv2.INTER_AREA)
            
            # Apply fisheye rectification if enabled
            img_for_tracking = img
            if self.rectifier is not None:
                img_for_tracking = self.rectifier.rectify(img)
            
            # ================================================================
            # VPS Real-time Processing (Parallel with Threading)
            # ================================================================
            # VPS runs in background thread while VIO continues immediately
            # This reduces total latency and utilizes multi-core CPUs
            vps_thread = None
            vps_result_container = [None]  # Thread-safe result storage
            
            if self.vps_runner is not None:
                # Get current EKF estimates for VPS processing
                # Extract IMU position from EKF state
                p_imu_enu = self.kf.x[:3].flatten()  # Position in ENU
                
                # Extract rotation matrix from quaternion
                q = self.kf.x[6:10].flatten()  # Quaternion [w, x, y, z]
                from .math_utils import quat_to_rot
                R_body_to_world = quat_to_rot(q)
                
                # Get GNSS position (apply lever arm)
                # Compute lever arm in ENU world frame
                lever_arm_world = R_body_to_world @ self.lever_arm
                p_gnss_enu = imu_to_gnss_position(
                    p_imu_enu, R_body_to_world, self.lever_arm
                )
                
                # Convert ENU to lat/lon
                lat, lon = self.proj_cache.xy_to_latlon(
                    p_gnss_enu[0], p_gnss_enu[1], self.lat0, self.lon0
                )
                alt = p_gnss_enu[2]  # MSL altitude
                
                # Get terrain height (DEM) to compute AGL for VPS filtering
                dem_height = self.dem.sample_m(lat, lon)
                if dem_height is None:
                    dem_height = 0.0
                agl = alt - dem_height
                
                est_yaw = self.kf.x[8]  # Yaw from EKF state
                
                # Define VPS processing function for thread
                def run_vps_in_thread():
                    try:
                        result = self.vps_runner.process_frame(
                            img=img,  # Use original image (not rectified)
                            t_cam=t_cam,
                            est_lat=lat,
                            est_lon=lon,
                            est_yaw=est_yaw,
                            est_alt=agl,  # Send AGL for 30m min_altitude check
                            frame_idx=self.state.img_idx
                        )
                        vps_result_container[0] = result
                    except Exception as e:
                        print(f"[VPS] Thread error: {e}")
                        vps_result_container[0] = None
                
                # Start VPS processing in background thread
                vps_thread = threading.Thread(target=run_vps_in_thread, daemon=True)
                vps_thread.start()
                
                # Clone EKF state for delayed update (stochastic cloning)
                if hasattr(self, 'vps_clone_manager'):
                    clone_id = f"vps_{self.state.img_idx}"
                    self.vps_clone_manager.clone_state(self.kf, t_cam, clone_id)
            
            # ================================================================
            # VIO Processing (continues immediately, parallel with VPS!)
            # ================================================================
            
            if is_fast_rotation:
                print(f"[VIO] SKIPPING due to fast rotation: {rotation_rate_deg_s:.1f} deg/s")
                self.state.img_idx += 1
                continue
            
            # Run VIO frontend
            ok, ninl, r_vo_mat, t_unit, dt_img = self.vio_fe.step(img_for_tracking, t_cam)
            
            # Loop closure detection - check when we have sufficient position estimate
            match_result = check_loop_closure(
                loop_detector=self.loop_detector,
                img_gray=img,
                t=t_cam,
                kf=self.kf,
                global_config=self.global_config,
                vio_fe=self.vio_fe
            )
            if match_result is not None:
                relative_yaw, kf_idx, num_inliers = match_result
                apply_loop_closure_correction(
                    kf=self.kf,
                    relative_yaw=relative_yaw,
                    kf_idx=kf_idx,
                    num_inliers=num_inliers,
                    t=t_cam,
                    cam_states=self.state.cam_states,
                    loop_detector=self.loop_detector
                )
            
            # Compute average optical flow (parallax) - use the new attributes
            # mean_parallax is now computed in step() EVERY frame, even if ok=False
            avg_flow_px = self.vio_fe.mean_parallax
            
            # Fallback to last_matches only if mean_parallax is still ~0 and matches exist
            if avg_flow_px < 0.01 and self.vio_fe.last_matches is not None:
                focal_px = kb_params.get('mu', 600)
                pts_prev, pts_cur = self.vio_fe.last_matches
                if len(pts_prev) > 0:
                    pts_prev_px = pts_prev * focal_px + np.array([[self.vio_fe.img_w/2, self.vio_fe.img_h/2]])
                    pts_cur_px = pts_cur * focal_px + np.array([[self.vio_fe.img_w/2, self.vio_fe.img_h/2]])
                    flows = pts_cur_px - pts_prev_px
                    avg_flow_px = float(np.median(np.linalg.norm(flows, axis=1)))
            
            # Debug: log feature statistics - use actual tracked features from VIOFrontEnd
            num_features = self.vio_fe.last_num_tracked
            num_inliers = self.vio_fe.last_num_inliers
            tracking_ratio = 1.0 if num_features > 0 else 0.0
            inlier_ratio = num_inliers / max(1, num_features)
            self.debug_writers.log_feature_stats(
                self.vio_fe.frame_idx, t_cam, num_features, num_features, num_inliers,
                avg_flow_px, avg_flow_px, tracking_ratio, inlier_ratio
            )
            
            # Apply preintegration at EVERY camera frame
            # Store Jacobians for bias observability in MSCKF
            preint_jacobians = None
            if ongoing_preint is not None:
                preint_jacobians = apply_preintegration_at_camera(self.kf, ongoing_preint, t_cam, imu_params)
            
            # Check parallax for different purposes
            # CRITICAL CHANGE: Separate low-parallax handling for velocity vs MSCKF/plane
            # - Low parallax (<2px): Skip velocity update, but ALLOW cloning/MSCKF
            # - This lets plane-aided MSCKF help even in nadir scenarios
            is_insufficient_parallax_for_velocity = avg_flow_px < min_parallax
            
            # Camera cloning for MSCKF (lower threshold than velocity)
            # Allow cloning even with low parallax to enable plane-aided MSCKF
            clone_threshold = min_parallax * 0.5  # Much lower: 1px instead of 4px
            should_clone = avg_flow_px >= clone_threshold and not is_fast_rotation
            
            if should_clone:
                clone_idx = clone_camera_for_msckf(
                    kf=self.kf,
                    t=t_cam,  # Use camera time for accurate cloning
                    cam_states=self.state.cam_states,
                    cam_observations=self.state.cam_observations,
                    vio_fe=self.vio_fe,
                    frame_idx=self.vio_fe.frame_idx,
                    preint_jacobians=preint_jacobians,  # Pass Jacobians for bias observability
                    max_clone_size=self.global_config.get('MSCKF_MAX_CLONE_SIZE', 11)
                )
                
                # Log MSCKF window state
                if clone_idx >= 0:
                    num_tracked = len(self.state.cam_observations[-1]['observations']) if self.state.cam_observations else 0
                    window_start = self.state.cam_states[0]['t'] if self.state.cam_states else t_cam
                    log_msckf_window(
                        msckf_window_csv=self.msckf_window_csv,
                        frame=self.vio_fe.frame_idx,
                        t=t_cam,
                        num_clones=len(self.state.cam_states),
                        num_tracked=num_tracked,
                        num_mature=0,
                        window_start=window_start,
                        marginalized_clone=-1
                    )
                    
                    # Trigger MSCKF update if enough clones
                    if len(self.state.cam_states) >= 3:
                        msckf_policy_scales, msckf_apply_scales = self._get_sensor_adaptive_scales("MSCKF")
                        msckf_adaptive_info: Dict[str, Any] = {}
                        num_updates = trigger_msckf_update(
                            kf=self.kf,
                            cam_states=self.state.cam_states,
                            cam_observations=self.state.cam_observations,
                            vio_fe=self.vio_fe,
                            t=t_cam,
                            msckf_dbg_csv=self.msckf_dbg_csv if hasattr(self, 'msckf_dbg_csv') else None,
                            dem_reader=self.dem,
                            origin_lat=self.lat0,
                            origin_lon=self.lon0,
                            plane_detector=self.plane_detector,
                            plane_config=self.global_config if self.plane_detector else None,
                            global_config=self.global_config,
                            chi2_scale=float(msckf_apply_scales.get("chi2_scale", 1.0)),
                            reproj_scale=float(msckf_apply_scales.get("reproj_scale", 1.0)),
                            adaptive_info=msckf_adaptive_info,
                        )
                        self.record_adaptive_measurement(
                            "MSCKF",
                            adaptive_info=msckf_adaptive_info,
                            timestamp=t_cam,
                            policy_scales=msckf_policy_scales,
                        )
                        
                        # Log FEJ consistency after MSCKF update
                        if num_updates > 0 and self.config.save_debug_data:
                            log_fej_consistency(
                                fej_csv=self.fej_csv,
                                t=t_cam,
                                frame=self.vio_fe.frame_idx if self.vio_fe else 0,
                                cam_states=self.state.cam_states,
                                kf=self.kf
                            )
            
            # Optical Flow Velocity Update - run EVERY camera frame as XY drift reduction fallback
            # This is independent of VO (Essential matrix) success and works even with low parallax
            # Key for outdoor flights: reduce XY drift when VPS unavailable
            if self.config.use_vio_velocity and avg_flow_px > 0.5:  # Very low threshold: any motion
                # Get ground truth error for NEES calculation (v2.9.9.8)
                vel_error, vel_cov = get_ground_truth_error(
                    t_cam, self.kf, self.ppk_trajectory,
                    self.lat0, self.lon0, self.proj_cache, 'velocity')
                vio_policy_scales, vio_apply_scales = self._get_sensor_adaptive_scales("VIO_VEL")
                vio_adaptive_info: Dict[str, Any] = {}
                
                apply_vio_velocity_update(
                    kf=self.kf,
                    r_vo_mat=r_vo_mat if ok else None,  # Can be None - will use optical flow direction
                    t_unit=t_unit if ok else None,      # Can be None - will use optical flow direction
                    t=t_cam,  # Use camera time for velocity update
                    dt_img=dt_img,
                    avg_flow_px=avg_flow_px,
                    imu_rec=rec,
                    global_config=self.global_config,
                    camera_view=self.config.camera_view,
                    dem_reader=self.dem,
                    lat0=self.lat0,
                    lon0=self.lon0,
                    use_vio_velocity=True,  # Always True when entering this block
                    proj_cache=self.proj_cache,
                    save_debug=self.config.save_debug_data,
                    residual_csv=self.residual_csv if self.config.save_debug_data else None,
                    vio_frame=self.state.vio_frame,
                    vio_fe=self.vio_fe,
                    state_error=vel_error,  # For NEES calculation
                    state_cov=vel_cov,      # For NEES calculation
                    chi2_scale=float(vio_apply_scales.get("chi2_scale", 1.0)),
                    r_scale_extra=float(vio_apply_scales.get("r_scale", 1.0)),
                    adaptive_info=vio_adaptive_info,
                )
                self.record_adaptive_measurement(
                    "VIO_VEL",
                    adaptive_info=vio_adaptive_info,
                    timestamp=t_cam,
                    policy_scales=vio_policy_scales,
                )
                
                # Increment VIO frame
                self.state.vio_frame += 1
                used_vo = (ok and r_vo_mat is not None)  # Only True if VO succeeded
                
                # Store VO data (if available)
                if ok and r_vo_mat is not None and t_unit is not None:
                    t_norm = t_unit / (np.linalg.norm(t_unit) + 1e-12)
                    vo_dx, vo_dy, vo_dz = float(t_norm[0]), float(t_norm[1]), float(t_norm[2])
                    r_eul = R_scipy.from_matrix(r_vo_mat).as_euler('zyx', degrees=True)
                    vo_y, vo_p, vo_r = float(r_eul[0]), float(r_eul[1]), float(r_eul[2])
                    
                    vo_data = {
                        'dx': vo_dx, 'dy': vo_dy, 'dz': vo_dz,
                        'roll': vo_r, 'pitch': vo_p, 'yaw': vo_y
                    }
                    
                    rot_angle_deg = np.degrees(np.arccos(np.clip((np.trace(r_vo_mat)-1)/2, -1, 1)))
                else:
                    # OF-velocity fallback mode (no VO)
                    vo_dx = vo_dy = vo_dz = 0.0
                    vo_r = vo_p = vo_y = 0.0
                    vo_data = None
                    rot_angle_deg = 0.0
                
                # Log VO debug
                vel_vx = float(self.kf.x[3, 0])
                vel_vy = float(self.kf.x[4, 0])
                vel_vz = float(self.kf.x[5, 0])
                
                # Determine use_vz_only from config (for logging)
                view_cfg = CAMERA_VIEW_CONFIGS.get(self.config.camera_view, CAMERA_VIEW_CONFIGS['nadir'])
                use_vz_only_default = view_cfg.get('use_vz_only', True)
                # Check if overridden by config
                vio_config = self.global_config.get('vio', {})
                use_vz_only = vio_config.get('use_vz_only', use_vz_only_default)
                
                log_vo_debug(
                    self.vo_dbg_csv, self.vio_fe.frame_idx, num_inliers, rot_angle_deg,
                    0.0,  # alignment_deg
                    rotation_rate_deg_s, use_vz_only,
                    not used_vo,  # skip_vo (True if using OF-velocity fallback)
                    vo_dx, vo_dy, vo_dz, vel_vx, vel_vy, vel_vz
                )
                
                # ================================================================
                # VPS Result Collection (after VIO completes)
                # ================================================================
                # Wait for VPS thread to finish and apply updates
                if vps_thread is not None:
                    # Wait for VPS processing to complete
                    vps_thread.join(timeout=2.0)  # Max 2 seconds wait
                    
                    # Get result from thread
                    vps_result = vps_result_container[0]
                    
                    # Apply VPS update using stochastic cloning
                    if vps_result is not None:
                        from .vps_integration import apply_vps_delayed_update
                        
                        # Create clone manager if not exists
                        if not hasattr(self, 'vps_clone_manager'):
                            from vps import VPSDelayedUpdateManager
                            self.vps_clone_manager = VPSDelayedUpdateManager(
                                max_delay_sec=0.5,  # From config
                                max_clones=3
                            )
                        
                        # Apply delayed update with stochastic cloning
                        clone_id = f"vps_{self.state.img_idx}"
                        vps_applied, _, _ = apply_vps_delayed_update(
                            kf=self.kf,
                            clone_manager=self.vps_clone_manager,
                            image_id=clone_id,
                            vps_lat=vps_result.lat,
                            vps_lon=vps_result.lon,
                            R_vps=vps_result.R_vps,
                            proj_cache=self.proj_cache,
                            lat0=self.lat0,
                            lon0=self.lon0,
                            time_since_last_vps=(t_cam - self.last_vps_update_time)
                        )
                        if vps_applied:
                            self.last_vps_update_time = t_cam
                            self.state.vps_idx += 1
                
                # Save keyframe image with visualization overlay
                if self.config.save_keyframe_images and hasattr(self, 'keyframe_dir'):
                    save_keyframe_with_overlay(img, self.vio_fe.frame_idx, 
                                              self.keyframe_dir, self.vio_fe)
            
            self.state.img_idx += 1
        
        return used_vo, vo_data
    
    def log_error(self, t: float):
        """
        Log VIO error vs ground truth.
        
        Args:
            t: Current timestamp
        """
        gt_df = self.ppk_trajectory
        if gt_df is None or len(gt_df) == 0:
            self._log_time_sync_debug(
                t_filter=float(t),
                t_gt_mapped=float("nan"),
                dt_gt=float("nan"),
                gt_idx=-1,
                gt_stamp_log=float("nan"),
                matched=0
            )
            return
        
        try:
            t_gt = self._filter_time_to_gt_time(t)
            
            # Find closest ground truth
            gt_diffs = np.abs(gt_df['stamp_log'].values - t_gt)
            gt_idx = int(np.argmin(gt_diffs))
            gt_row = gt_df.iloc[gt_idx]
            dt_gt = float(gt_diffs[gt_idx])
            gt_stamp_log = float(gt_row['stamp_log']) if 'stamp_log' in gt_row else float("nan")
            is_matched = 1 if dt_gt <= 1.0 else 0
            self._log_time_sync_debug(
                t_filter=float(t),
                t_gt_mapped=float(t_gt),
                dt_gt=dt_gt,
                gt_idx=gt_idx,
                gt_stamp_log=gt_stamp_log,
                matched=is_matched
            )
            if dt_gt > 1.0:
                if not self._warned_gt_time_mismatch:
                    print(
                        f"[ERROR_LOG] WARNING: Large GT time mismatch ({dt_gt:.3f}s). "
                        f"mode={self.error_time_mode}. Skipping unmatched rows."
                    )
                    self._warned_gt_time_mismatch = True
                return
            
            use_ppk = self.ppk_trajectory is not None
            
            if use_ppk:
                gt_lat = gt_row['lat']
                gt_lon = gt_row['lon']
                gt_alt = gt_row['height']
            else:
                gt_lat = gt_row['lat_dd']
                gt_lon = gt_row['lon_dd']
                gt_alt = gt_row['altitude_MSL_m']
            
            gt_E, gt_N = self.proj_cache.latlon_to_xy(gt_lat, gt_lon, self.lat0, self.lon0)
            gt_U = gt_alt
            
            # VIO prediction (IMU position)
            vio_E = float(self.kf.x[0, 0])
            vio_N = float(self.kf.x[1, 0])
            vio_U = float(self.kf.x[2, 0])
            
            # Ground truth is GNSS position - convert VIO (IMU) to GNSS for fair comparison
            # Use instance lever_arm (already set in initialize_ekf)
            if np.linalg.norm(self.lever_arm) > 0.01:
                from scipy.spatial.transform import Rotation as R_scipy
                q_vio = self.kf.x[6:10, 0]
                q_xyzw = np.array([q_vio[1], q_vio[2], q_vio[3], q_vio[0]])
                R_body_to_world = R_scipy.from_quat(q_xyzw).as_matrix()
                
                p_imu_enu = np.array([vio_E, vio_N, vio_U])
                p_gnss_enu = imu_to_gnss_position(p_imu_enu, R_body_to_world, self.lever_arm)
                vio_E, vio_N, vio_U = p_gnss_enu[0], p_gnss_enu[1], p_gnss_enu[2]
            
            # Errors
            err_E = vio_E - gt_E
            err_N = vio_N - gt_N
            err_U = vio_U - gt_U
            pos_error = np.sqrt(err_E**2 + err_N**2 + err_U**2)
            
            # Velocity error (compute from consecutive GPS positions)
            # FIX v3.9.3: Use PPK velocity columns directly instead of position difference
            # Position difference method fails when gt_idx doesn't change between samples
            vel_error = 0.0
            vel_err_E = vel_err_N = vel_err_U = 0.0
            
            if use_ppk and 've' in gt_row and 'vn' in gt_row and 'vu' in gt_row:
                # PPK provides direct velocity measurements
                gt_vel_E = float(gt_row['ve'])
                gt_vel_N = float(gt_row['vn'])
                gt_vel_U = float(gt_row['vu'])
                
                # VIO velocity
                vio_vel_E = float(self.kf.x[3, 0])
                vio_vel_N = float(self.kf.x[4, 0])
                vio_vel_U = float(self.kf.x[5, 0])
                
                # Velocity error
                vel_err_E = vio_vel_E - gt_vel_E
                vel_err_N = vio_vel_N - gt_vel_N
                vel_err_U = vio_vel_U - gt_vel_U
                vel_error = np.sqrt(vel_err_E**2 + vel_err_N**2 + vel_err_U**2)
            elif gt_idx > 0 and gt_idx < len(gt_df) - 1:
                # Fallback: compute from consecutive GPS positions (for flight_log)
                gt_row_prev = gt_df.iloc[gt_idx - 1]
                gt_row_next = gt_df.iloc[gt_idx + 1]
                
                dt = gt_row_next['stamp_log'] - gt_row_prev['stamp_log']
                if dt > 0.01:  # Avoid division by zero
                    if use_ppk:
                        gt_E_prev, gt_N_prev = self.proj_cache.latlon_to_xy(gt_row_prev['lat'], gt_row_prev['lon'], self.lat0, self.lon0)
                        gt_E_next, gt_N_next = self.proj_cache.latlon_to_xy(gt_row_next['lat'], gt_row_next['lon'], self.lat0, self.lon0)
                        gt_U_prev = gt_row_prev['height']
                        gt_U_next = gt_row_next['height']
                    else:
                        gt_E_prev, gt_N_prev = self.proj_cache.latlon_to_xy(gt_row_prev['lat_dd'], gt_row_prev['lon_dd'], self.lat0, self.lon0)
                        gt_E_next, gt_N_next = self.proj_cache.latlon_to_xy(gt_row_next['lat_dd'], gt_row_next['lon_dd'], self.lat0, self.lon0)
                        gt_U_prev = gt_row_prev['altitude_MSL_m']
                        gt_U_next = gt_row_next['altitude_MSL_m']
                    
                    # Ground truth velocity
                    gt_vel_E = (gt_E_next - gt_E_prev) / dt
                    gt_vel_N = (gt_N_next - gt_N_prev) / dt
                    gt_vel_U = (gt_U_next - gt_U_prev) / dt
                    
                    # VIO velocity
                    vio_vel_E = float(self.kf.x[3, 0])
                    vio_vel_N = float(self.kf.x[4, 0])
                    vio_vel_U = float(self.kf.x[5, 0])
                    
                    # Velocity error
                    vel_err_E = vio_vel_E - gt_vel_E
                    vel_err_N = vio_vel_N - gt_vel_N
                    vel_err_U = vio_vel_U - gt_vel_U
                    vel_error = np.sqrt(vel_err_E**2 + vel_err_N**2 + vel_err_U**2)
            
            # Yaw
            q_vio = self.kf.x[6:10, 0]
            yaw_vio = np.rad2deg(quaternion_to_yaw(q_vio))
            
            yaw_gt = np.nan
            yaw_error = np.nan
            if use_ppk and 'yaw' in gt_row:
                yaw_gt = 90.0 - np.rad2deg(gt_row['yaw'])
                yaw_error = ((yaw_vio - yaw_gt + 180) % 360) - 180
            
            with open(self.error_csv, "a", newline="") as ef:
                ef.write(
                    f"{t:.6f},{pos_error:.3f},{err_E:.3f},{err_N:.3f},{err_U:.3f},"
                    f"{vel_error:.3f},{vel_err_E:.3f},{vel_err_N:.3f},{vel_err_U:.3f},"
                    f"{err_U:.3f},{yaw_vio:.2f},{yaw_gt:.2f},{yaw_error:.2f},"
                    f"{gt_lat:.8f},{gt_lon:.8f},{gt_alt:.3f},"
                    f"{vio_E:.3f},{vio_N:.3f},{vio_U:.3f}\n"
                )
        except Exception:
            pass
    
    def process_magnetometer(self, t: float):
        """
        Process magnetometer measurements up to current time.
        
        Args:
            t: Current timestamp
        """
        sigma_mag = self.global_config.get('SIGMA_MAG_YAW', 0.15)
        declination = self.global_config.get('MAG_DECLINATION', 0.0)
        use_raw = self.global_config.get('MAG_USE_RAW_HEADING', True)
        rate_limit = self.global_config.get('MAG_UPDATE_RATE_LIMIT', 1)
        
        while (self.state.mag_idx < len(self.mag_list) and 
               self.mag_list[self.state.mag_idx].t <= t):
            
            mag_rec = self.mag_list[self.state.mag_idx]
            self.state.mag_idx += 1
            
            # Rate limiting
            if (self.state.mag_idx - 1) % rate_limit != 0:
                continue
            
            # v3.9.7: Use EKF estimated mag_bias instead of static config hard_iron
            # This enables online hard iron estimation for time-varying interference
            if self.config.use_mag_estimated_bias:
                # Use EKF state mag_bias (indices 16:19) as hard iron
                hard_iron = self.kf.x[16:19, 0].flatten()
            else:
                # Fallback to static config hard iron
                hard_iron = self.global_config.get('MAG_HARD_IRON_OFFSET', None)
            soft_iron = self.global_config.get('MAG_SOFT_IRON_MATRIX', None)
            mag_cal = calibrate_magnetometer(mag_rec.mag, hard_iron=hard_iron, soft_iron=soft_iron)
            
            # Compute raw yaw from calibrated magnetometer
            from .magnetometer import compute_yaw_from_mag
            q_current = self.kf.x[6:10, 0].flatten()
            yaw_mag_raw, quality = compute_yaw_from_mag(
                mag_body=mag_cal,
                q_wxyz=q_current,
                mag_declination=declination,
                use_raw_heading=use_raw
            )
            
            # Apply magnetometer filter (EMA smoothing + gyro consistency check)
            # v3.4.0: Use phase-based convergence (state-based, not time-based)
            in_convergence = self.state.current_phase < 2  # SPINUP or EARLY phase
            yaw_mag_filtered, r_scale, filter_info = apply_mag_filter(
                yaw_mag=yaw_mag_raw,
                yaw_t=mag_rec.t,
                gyro_z=self.last_gyro_z,
                dt_imu=self.last_imu_dt,
                in_convergence=in_convergence,
                mag_max_yaw_rate=self.global_config.get('MAG_MAX_YAW_RATE_DEG', 30.0) * np.pi / 180.0,
                mag_gyro_threshold=self.global_config.get('MAG_GYRO_THRESHOLD_DEG', 10.0) * np.pi / 180.0,
                mag_ema_alpha=self.global_config.get('MAG_EMA_ALPHA', 0.3),
                mag_consistency_r_inflate=self.global_config.get('MAG_R_INFLATE', 5.0)
            )
            
            # Scale measurement noise based on filter confidence
            sigma_mag_scaled = sigma_mag * r_scale
            mag_policy_scales, mag_apply_scales = self._get_sensor_adaptive_scales("MAG")
            mag_adaptive_info: Dict[str, Any] = {}
            
            # Use filtered yaw instead of raw calibrated mag
            has_ppk = self.ppk_state is not None
            
            # Get residual_csv path if debug data is enabled
            residual_path = self.residual_csv if self.config.save_debug_data and hasattr(self, 'residual_csv') else None
            
            # Apply magnetometer update using FILTERED yaw
            # CRITICAL: Pass filtered yaw directly instead of raw mag
            applied, reason = apply_magnetometer_update(
                self.kf,
                mag_calibrated=mag_cal,  # Still needed for compute_yaw_from_mag inside
                mag_declination=declination,
                use_raw_heading=use_raw,
                sigma_mag_yaw=sigma_mag_scaled,  # Use scaled sigma
                current_phase=self.state.current_phase,  # v3.4.0: state-based phase
                in_convergence=in_convergence,
                has_ppk_yaw=has_ppk,
                timestamp=mag_rec.t,
                residual_csv=residual_path,
                frame=self.state.vio_frame,
                yaw_override=yaw_mag_filtered,  # NEW: pass filtered yaw directly
                filter_info=filter_info,  # v2.9.2: track filter rejection reasons
                use_estimated_bias=self.config.use_mag_estimated_bias,  # v3.9.8: freeze states if disabled
                r_scale_extra=float(mag_apply_scales.get("r_scale", 1.0)),
                adaptive_info=mag_adaptive_info,
            )
            self.record_adaptive_measurement(
                "MAG",
                adaptive_info=mag_adaptive_info,
                timestamp=mag_rec.t,
                policy_scales=mag_policy_scales,
            )
            
            if applied:
                self.state.mag_updates += 1
            else:
                self.state.mag_rejects += 1
    
    def _process_single_vps(self, vps, t: float):
        """
        Process a single VPS measurement (for event-driven mode).
        
        Args:
            vps: VPS record with lat, lon, etc.
            t: VPS timestamp
        """
        from .vps_integration import (
            compute_vps_innovation, compute_vps_acceptance_threshold
        )
        
        sigma_vps = self.global_config.get('SIGMA_VPS_XY', 1.0)
        
        # Compute innovation and Mahalanobis distance
        vps_xy, innovation, m2_test = compute_vps_innovation(
            vps, self.kf, self.lat0, self.lon0, self.proj_cache
        )
        
        # Compute adaptive acceptance threshold
        time_since_correction = t - getattr(self.kf, 'last_absolute_correction_time', t)
        innovation_mag = float(np.linalg.norm(innovation))
        max_innovation_m, r_scale, tier_name = compute_vps_acceptance_threshold(
            time_since_correction, innovation_mag
        )
        
        # Chi-square threshold (2 DOF)
        chi2_threshold = 5.99  # 95% confidence
        
        # Gate by innovation magnitude OR chi-square test
        if innovation_mag > max_innovation_m:
            print(f"[VPS] REJECTED: innovation {innovation_mag:.1f}m > {max_innovation_m:.1f}m "
                  f"({tier_name}, drift={time_since_correction:.1f}s)")
            return
        
        if m2_test > chi2_threshold * 10:  # Very permissive for first VPS
            print(f"[VPS] REJECTED: chi2={m2_test:.1f} >> {chi2_threshold} "
                  f"(innovation={innovation_mag:.1f}m)")
            return
        
        # Apply update with adaptive R scaling
        applied = apply_vps_update(
            self.kf,
            vps_xy=vps_xy,
            sigma_vps=sigma_vps,
            r_scale=r_scale
        )
        
        if applied:
            self.kf.last_absolute_correction_time = t
            print(f"[VPS] Applied at t={t:.3f}, innovation={innovation_mag:.1f}m, "
                  f"tier={tier_name}, R_scale={r_scale:.1f}x")
    
    def _process_single_mag(self, mag_rec, t: float):
        """
        Process a single magnetometer measurement (for event-driven mode).
        
        v3.9.4 CRITICAL: Magnetometer time synchronization via time_ref
        
        CONDITION 1: Filter must use SAME TIME BASE for all sensors
        - IMU: time_ref (hardware monotonic clock)
        - Camera: time_ref via timeref.csv interpolation
        - Magnetometer: time_ref via interpolation in load_mag_csv()
        
        CONDITION 2: mag_rec.t must represent TRUE measurement time
        - After interpolation, mag_rec.t is hardware time (consistent with IMU)
        - State is already at time t (propagated by IMU loop)
        - If |t - mag_rec.t| > threshold, skip measurement (timing issue)
        
        NOTE: Full state propagation to mag_rec.t requires storing IMU buffer.
        Current implementation: Use state at closest IMU time if within threshold.
        
        Args:
            mag_rec: Magnetometer record with hardware-synchronized timestamp
            t: Current IMU timestamp (after propagation)
        """
        # Check time synchronization (should be very close if time_ref works)
        if abs(t - mag_rec.t) > 0.05:  # 50ms threshold
            print(f"[Mag] WARNING: Large time difference {t - mag_rec.t:.3f}s - may indicate clock sync issue")
        
        sigma_mag = self.global_config.get('SIGMA_MAG_YAW', 0.15)
        declination = self.global_config.get('MAG_DECLINATION', 0.0)
        use_raw = self.global_config.get('MAG_USE_RAW_HEADING', True)
        
        # v3.9.7: Use EKF estimated mag_bias instead of static config hard_iron
        if self.config.use_mag_estimated_bias:
            hard_iron = self.kf.x[16:19, 0].flatten()
        else:
            hard_iron = self.global_config.get('MAG_HARD_IRON_OFFSET', None)
        soft_iron = self.global_config.get('MAG_SOFT_IRON_MATRIX', None)
        mag_cal = calibrate_magnetometer(mag_rec.mag, hard_iron=hard_iron, soft_iron=soft_iron)
        
        # Compute raw yaw from calibrated magnetometer
        from .magnetometer import compute_yaw_from_mag
        q_current = self.kf.x[6:10, 0].flatten()
        yaw_mag_raw, quality = compute_yaw_from_mag(
            mag_body=mag_cal,
            q_wxyz=q_current,
            mag_declination=declination,
            use_raw_heading=use_raw
        )
        
        # Apply magnetometer filter (EMA smoothing + gyro consistency check)
        in_convergence = self.state.current_phase < 2  # SPINUP or EARLY phase
        yaw_mag_filtered, r_scale, filter_info = apply_mag_filter(
            yaw_mag=yaw_mag_raw,
            yaw_t=mag_rec.t,
            gyro_z=self.last_gyro_z,
            dt_imu=self.last_imu_dt,
            in_convergence=in_convergence,
            mag_max_yaw_rate=self.global_config.get('MAG_MAX_YAW_RATE_DEG', 30.0) * np.pi / 180.0,
            mag_gyro_threshold=self.global_config.get('MAG_GYRO_THRESHOLD_DEG', 10.0) * np.pi / 180.0,
            mag_ema_alpha=self.global_config.get('MAG_EMA_ALPHA', 0.3),
            mag_consistency_r_inflate=self.global_config.get('MAG_R_INFLATE', 5.0)
        )
        
        # Scale measurement noise based on filter confidence
        sigma_mag_scaled = sigma_mag * r_scale
        
        # Use filtered yaw instead of raw calibrated mag
        has_ppk = self.ppk_state is not None
        
        # Get residual_csv path if debug data is enabled
        residual_path = self.residual_csv if self.config.save_debug_data and hasattr(self, 'residual_csv') else None
        
        # Apply magnetometer update using FILTERED yaw
        applied, reason = apply_magnetometer_update(
            self.kf,
            mag_calibrated=mag_cal,
            mag_declination=declination,
            use_raw_heading=use_raw,
            sigma_mag_yaw=sigma_mag_scaled,
            current_phase=self.state.current_phase,
            in_convergence=in_convergence,
            has_ppk_yaw=has_ppk,
            timestamp=mag_rec.t,
            residual_csv=residual_path,
            frame=self.state.vio_frame,
            yaw_override=yaw_mag_filtered,
            filter_info=filter_info,
            use_estimated_bias=self.config.use_mag_estimated_bias  # v3.9.8: freeze states if disabled
        )
        
        if applied:
            self.state.mag_updates += 1
        else:
            self.state.mag_rejects += 1
    
    def process_dem_height(self, t: float):
        """
        Process DEM height update.
        
        Args:
            t: Current timestamp
        """
        sigma_height = self.global_config.get('SIGMA_AGL_Z', 2.5)
        
        # Get current position
        lat_now, lon_now = self.proj_cache.xy_to_latlon(
            self.kf.x[0, 0], self.kf.x[1, 0],
            self.lat0, self.lon0
        )
        
        # Sample DEM
        dem_now = self.dem.sample_m(lat_now, lon_now) if self.dem.ds else None
        
        if dem_now is None or np.isnan(dem_now):
            return
        
        # Compute height measurement
        # v3.9.12: Try Direct MSL from Flight Log (Barometer/GNSS) first
        # This replaces DEM+AGL logic if available
        msl_direct = None
        if self.msl_interpolator:
            raw_msl = self.msl_interpolator.get_msl(t)
            if raw_msl is not None:
                # Apply the alignment offset (Result = Log - Offset)
                msl_direct = raw_msl - getattr(self, 'msl_offset', 0.0)
            
        if msl_direct is not None:
            # Direct MSL update
            height_m = msl_direct
            # Disable XY uncertainty scaling (baro/GNSS doesn't depend on XY)
            xy_uncertainty = 0.0 
            has_valid_dem = True # Treat as valid measurement
        else:
            # Fallback: Compute height measurement from DEM + estimated AGL
            # CRITICAL FIX v3.9.2: DO NOT use GPS MSL - that causes innovation=0!
            # DEM update must constrain drift by comparing: (DEM + estimated_AGL) vs state
            # State is MSL, measurement = DEM + estimated_AGL
            
            # Estimate AGL from current MSL state and DEM
            current_msl = self.kf.x[2, 0]
            estimated_agl = current_msl - dem_now
            
            # Clamp AGL to reasonable range (helicopter doesn't fly underground or >500m AGL)
            estimated_agl = np.clip(estimated_agl, 0.0, 500.0)
            
            # Reconstruct MSL measurement from DEM + estimated AGL
            height_m = dem_now + estimated_agl
            has_valid_dem = True # Since we checked dem_now is not None/NaN before
        
        # Compute uncertainties
        if msl_direct is None:
            # Only use XY uncertainty for DEM-based update
            xy_uncertainty = float(np.trace(self.kf.P[0:2, 0:2]))
        
        time_since_correction = t - getattr(self.kf, 'last_absolute_correction_time', t)
        speed = float(np.linalg.norm(self.kf.x[3:6, 0]))
        dem_policy_scales, dem_apply_scales = self._get_sensor_adaptive_scales("DEM")
        dem_adaptive_info: Dict[str, Any] = {}
        
        # Get residual_csv path if debug data is enabled
        residual_path = self.residual_csv if self.config.save_debug_data and hasattr(self, 'residual_csv') else None
        
        applied, reason = apply_dem_height_update(
            self.kf,
            height_measurement=height_m,
            sigma_height=sigma_height,
            xy_uncertainty=xy_uncertainty,
            time_since_correction=time_since_correction,
            speed_ms=speed,
            has_valid_dem=True,
            no_vision_corrections=(len(self.imgs) == 0 and len(self.vps_list) == 0),
            timestamp=t,
            residual_csv=residual_path,
            frame=self.state.vio_frame,
            threshold_scale=float(dem_apply_scales.get("threshold_scale", 1.0)),
            r_scale_extra=float(dem_apply_scales.get("r_scale", 1.0)),
            adaptive_info=dem_adaptive_info,
        )
        self.record_adaptive_measurement(
            "DEM",
            adaptive_info=dem_adaptive_info,
            timestamp=t,
            policy_scales=dem_policy_scales,
        )
    
    def log_pose(self, t: float, dt: float, used_vo: bool,
                 vo_data: Optional[Dict] = None,
                 msl_now: Optional[float] = None,
                 agl_now: Optional[float] = None,
                 lat_now: Optional[float] = None,
                 lon_now: Optional[float] = None):
        """
        Log current pose to CSV.
        
        Args:
            t: Current timestamp
            dt: Time delta
            used_vo: Whether VO was used this frame
            vo_data: VO output data (optional)
            msl_now: Pre-computed MSL altitude (optional, for consistency with vio_vps.py)
            agl_now: Pre-computed AGL altitude (optional)
            lat_now: Pre-computed latitude (optional)
            lon_now: Pre-computed longitude (optional)
        """
        # Use pre-computed values if provided, otherwise compute from current state
        if lat_now is None or lon_now is None:
            lat_now, lon_now = self.proj_cache.xy_to_latlon(
                self.kf.x[0, 0], self.kf.x[1, 0],
                self.lat0, self.lon0
            )
        
        if msl_now is None or agl_now is None:
            # Compute MSL/AGL from current state (post-DEM update)
            dem_now = self.dem.sample_m(lat_now, lon_now) if self.dem.ds else 0.0
            if dem_now is None:
                dem_now = 0.0
            
            msl_now = self.kf.x[2, 0]
            agl_now = msl_now - dem_now
        
        frame_str = str(self.state.vio_frame) if used_vo else ""
        
        vo_dx = vo_data.get('dx', np.nan) if vo_data else np.nan
        vo_dy = vo_data.get('dy', np.nan) if vo_data else np.nan
        vo_dz = vo_data.get('dz', np.nan) if vo_data else np.nan
        vo_r = vo_data.get('roll', np.nan) if vo_data else np.nan
        vo_p = vo_data.get('pitch', np.nan) if vo_data else np.nan
        vo_y = vo_data.get('yaw', np.nan) if vo_data else np.nan
        
        with open(self.pose_csv, "a", newline="") as f:
            f.write(
                f"{t - self.state.t0:.6f},{dt:.6f},{frame_str},"
                f"{self.kf.x[0,0]:.3f},{self.kf.x[1,0]:.3f},{msl_now:.3f},"
                f"{self.kf.x[3,0]:.3f},{self.kf.x[4,0]:.3f},{self.kf.x[5,0]:.3f},"
                f"{lat_now:.8f},{lon_now:.8f},{agl_now:.3f},"
                f"{'' if np.isnan(vo_dx) else f'{vo_dx:.6f}'},"
                f"{'' if np.isnan(vo_dy) else f'{vo_dy:.6f}'},"
                f"{'' if np.isnan(vo_dz) else f'{vo_dz:.6f}'},"
                f"{'' if np.isnan(vo_r) else f'{vo_r:.3f}'},"
                f"{'' if np.isnan(vo_p) else f'{vo_p:.3f}'},"
                f"{'' if np.isnan(vo_y) else f'{vo_y:.3f}'}\n"
            )
    
    def print_summary(self):
        """Print final summary statistics."""
        print("\n\n--- Done ---")
        print(f"Total IMU samples: {len(self.imu)}")
        print(f"Images used: {self.state.vio_frame}")
        print(f"VPS used: {self.state.vps_idx}")
        print(f"Magnetometer: {self.state.mag_updates} updates | "
              f"{self.state.mag_rejects} rejected")
        print(f"ZUPT: {self.state.zupt_applied} applied | "
              f"{self.state.zupt_rejected} rejected | "
              f"{self.state.zupt_detected} detected")
        
        if self.kf is not None and hasattr(self.kf, "get_cov_growth_summary"):
            cov_summary = self.kf.get_cov_growth_summary()
            if len(cov_summary) > 0:
                print("\nCovariance Growth Sources (top 8):")
                for row in cov_summary[:8]:
                    print(
                        f"  {row['update_type']}: samples={row['samples']}, "
                        f"growth={row['growth_events']} ({100.0*row['growth_rate']:.1f}%), "
                        f"largeP={row['large_events']} ({100.0*row['large_rate']:.1f}%), "
                        f"max|P|={row['max_pmax']:.2e}"
                    )
        
        print_msckf_stats()
        
        # Print error statistics using dedicated function
        from .output_utils import print_error_statistics
        print_error_statistics(self.error_csv)
        self._write_benchmark_health_summary()

    def _write_benchmark_health_summary(self):
        """Write one-row benchmark health summary CSV for before/after comparison."""
        if not self.benchmark_health_summary_csv:
            return

        projection_count = 0
        first_projection_time = float("nan")
        pcond_max_stats = float("nan")
        if self.kf is not None and hasattr(self.kf, "get_conditioning_stats"):
            try:
                cstats = self.kf.get_conditioning_stats()
                projection_count = int(cstats.get("projection_count", 0))
                first_projection_time = float(cstats.get("first_projection_time", float("nan")))
                pcond_max_stats = float(cstats.get("max_cond_seen", float("nan")))
            except Exception:
                pass

        pcond_max_cov = float("nan")
        pmax_max = float("nan")
        cov_large_rate = float("nan")
        if self.cov_health_csv and os.path.isfile(self.cov_health_csv):
            try:
                cov_df = pd.read_csv(self.cov_health_csv)
                if len(cov_df) > 0:
                    pcond_max_cov = float(pd.to_numeric(cov_df["p_cond"], errors="coerce").max())
                    pmax_max = float(pd.to_numeric(cov_df["p_max"], errors="coerce").max())
                    cov_large_rate = float(pd.to_numeric(cov_df["large_flag"], errors="coerce").mean())
            except Exception:
                pass

        pcond_max = pcond_max_stats
        if np.isfinite(pcond_max_cov):
            pcond_max = max(pcond_max, pcond_max_cov) if np.isfinite(pcond_max) else pcond_max_cov

        pos_rmse = float("nan")
        final_pos_err = float("nan")
        final_alt_err = float("nan")
        if self.error_csv and os.path.isfile(self.error_csv):
            try:
                err_df = pd.read_csv(self.error_csv)
                if len(err_df) > 0:
                    pos_vals = pd.to_numeric(err_df["pos_error_m"], errors="coerce").to_numpy(dtype=float)
                    alt_vals = pd.to_numeric(err_df["alt_error_m"], errors="coerce").to_numpy(dtype=float)
                    pos_rmse = float(np.sqrt(np.nanmean(pos_vals ** 2)))
                    final_pos_err = float(pos_vals[-1])
                    final_alt_err = float(alt_vals[-1])
            except Exception:
                pass

        output_dir_norm = os.path.normpath(self.config.output_dir)
        if os.path.basename(output_dir_norm) == "preintegration":
            run_id = os.path.basename(os.path.dirname(output_dir_norm))
        else:
            run_id = os.path.basename(output_dir_norm)
        if not run_id:
            run_id = output_dir_norm

        append_benchmark_health_summary(
            self.benchmark_health_summary_csv,
            run_id=run_id,
            projection_count=projection_count,
            first_projection_time=first_projection_time,
            pcond_max=pcond_max,
            pmax_max=pmax_max,
            cov_large_rate=cov_large_rate,
            pos_rmse=pos_rmse,
            final_pos_err=final_pos_err,
            final_alt_err=final_alt_err,
        )
        print(
            f"[SUMMARY] projection_count={projection_count}, "
            f"first_projection_time={first_projection_time:.3f}, "
            f"pcond_max={pcond_max:.2e}, cov_large_rate={cov_large_rate:.3f}"
        )
    
    def run(self):
        """
        Run the complete VIO pipeline.
        
        Architecture Router (v3.7.0):
        - estimator_mode: "imu_step_preint_cache" → IMU-driven (with preintegration cache)
        - estimator_mode: "event_queue_output_predictor" → Event-driven (TODO - not implemented yet)
        
        IMU-driven: Process all IMU samples sequentially @ 400Hz with sub-sample precision
        Event-driven: Process measurements in chronological order using priority queue
        """
        print("=" * 80)
        print("VIO+EKF Pipeline Starting (v3.6.0)")
        print("=" * 80)
        
        # =================================================================
        # Load configuration from VIOConfig._raw_config (v3.2.0)
        # =================================================================
        # VIOConfig already has all settings from YAML
        # _raw_config contains the parsed YAML dict for backward compatibility
        if hasattr(self.config, '_raw_config') and self.config._raw_config:
            self.global_config = self.config._raw_config
        elif self.config.config_yaml:
            # Fallback: load from YAML path if _raw_config not set
            self.load_config()
        
        # Print algorithm settings (already set in VIOConfig from YAML)
        print(f"\n[CONFIG] Algorithm settings (from YAML):")
        print(f"  config_yaml: {self.config.config_yaml}")
        print(f"  camera_view: {self.config.camera_view}")
        print(f"  use_vio_velocity: {self.config.use_vio_velocity}")
        print(f"  use_magnetometer: {self.config.use_magnetometer}")
        print(f"  estimate_imu_bias: {self.config.estimate_imu_bias}")
        print(f"  estimator_mode: {self.config.estimator_mode}")
        print(f"  fast_mode: {self.config.fast_mode}")
        print(f"  frame_skip: {self.config.frame_skip}")
        self.initialize_adaptive_controller()
        
        # =================================================================
        # Architecture Selection (v3.8.0: enum-based with full event-driven support)
        # =================================================================
        if self.config.estimator_mode == "event_queue_output_predictor":
            # Event-driven mode: propagate to each measurement time
            # Key guarantee: filter_time == measurement_time before EVERY update
            print("\n[ARCH] Selected: Event-driven with output predictor (event_queue_output_predictor)")
            print("[ARCH] Features:")
            print("  - Priority queue orders all sensor events by timestamp")
            print("  - Propagate-to-event ensures state_time == measurement_time")
            print("  - Fast-propagate output layer for low-latency logging")
            run_event_driven_loop(self)
        elif self.config.estimator_mode == "imu_step_preint_cache":
            # IMU-driven mode: process all IMU @ 400Hz with sub-sample precision
            print("\n[ARCH] Selected: IMU-driven with sub-sample timestamp precision (imu_step_preint_cache)")
            # Cleanup expired VPS clones (stochastic cloning)
            # Note: t_current is not available here, this cleanup should be inside the loop
            # or called with a relevant timestamp. Assuming this is a placeholder for
            # a call that will be made within run_imu_driven_loop or its main loop.
            # For now, we'll add it as a comment or a call with a dummy timestamp if needed.
            # If vps_clone_manager exists, it implies it's used in this mode.
            if hasattr(self, 'vps_clone_manager') and self.vps_clone_manager is not None:
                # This call needs a current timestamp 't_current' which is only available inside the loop.
                # The instruction implies it should be part of the update cycle.
                # For now, we'll add a placeholder comment.
                # self.vps_clone_manager.cleanup_expired_clones(t_current) # t_current needs to be defined
                pass # The actual call will be placed inside run_imu_driven_loop
            run_imu_driven_loop(self)
        else:
            raise ValueError(f"Unknown estimator_mode: {self.config.estimator_mode}")
