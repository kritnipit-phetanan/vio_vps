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
import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

# Performance-critical imports moved to top-level to avoid per-iteration overhead
from .config import load_config, VIOConfig, CAMERA_VIEW_CONFIGS
from .data_loaders import (
    load_imu_csv, load_images, load_vps_csv, 
    load_mag_csv, load_ppk_initial_state, load_ppk_trajectory,
    load_msl_from_gga, load_quarry_initial,
    DEMReader, ensure_local_proj
)
from .ekf import ExtendedKalmanFilter
from .state_manager import initialize_ekf_state, imu_to_gnss_position
from .vio_frontend import VIOFrontEnd
from .camera import make_KD_for_size
from .propagation import (
    process_imu, propagate_error_state_covariance, 
    augment_state_with_camera, detect_stationary, apply_zupt,
    apply_preintegration_at_camera, clone_camera_for_msckf,
    get_flight_phase  # v3.3.0: State-based phase detection
)
from .vps_integration import apply_vps_update, xy_to_latlon, latlon_to_xy
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
    DebugCSVWriters, init_output_csvs, get_ground_truth_error
)
from .trn import TerrainReferencedNavigation, TRNConfig, create_trn_from_config

# VIOConfig is imported from config.py (single source of truth)

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
        
        # Global config (loaded from YAML)
        self.global_config = {}
        
        # Output file handles
        self.pose_csv = None
        self.error_csv = None
        self.state_dbg_csv = None
        
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
        
        # Load initial position from PPK or GGA
        if self.ppk_state is not None:
            self.lat0 = self.ppk_state.lat
            self.lon0 = self.ppk_state.lon
            v_init = np.array([self.ppk_state.ve, self.ppk_state.vn, self.ppk_state.vu])
            print(f"[INIT] Using PPK for lat/lon/velocity")
        else:
            self.lat0, self.lon0, _, v_init = load_quarry_initial(self.config.quarry_path)
            print(f"[INIT] Using GGA for lat/lon/velocity")
        
        self.v_init = v_init
        
        # Always use GGA for MSL (PPK has ellipsoidal height)
        self.msl0 = load_msl_from_gga(self.config.quarry_path)
        if self.msl0 is None:
            _, _, self.msl0, _ = load_quarry_initial(self.config.quarry_path)
        
        # Load main data
        self.imu = load_imu_csv(self.config.imu_path)
        self.imgs = load_images(self.config.images_dir, self.config.images_index_csv)
        self.vps_list = load_vps_csv(self.config.vps_csv)
        
        # v3.2.0: use_magnetometer is already populated from YAML (magnetometer.enabled)
        # No need for redundant check - VIOConfig.use_magnetometer is the final decision
        self.mag_list = load_mag_csv(self.config.mag_csv) if self.config.use_magnetometer else []
        
        self.dem = DEMReader.open(self.config.dem_path)
        
        # Load PPK trajectory for error comparison
        self.ppk_trajectory = None
        if self.config.ground_truth_path:
            self.ppk_trajectory = load_ppk_trajectory(self.config.ground_truth_path)
        
        # Load flight log for MSL updates
        self.flight_log_df = None
        if os.path.exists(self.config.quarry_path):
            df = pd.read_csv(self.config.quarry_path)
            if 'stamp_log' in df.columns and 'altitude_MSL_m' in df.columns:
                self.flight_log_df = df
        
        # Print summary
        print("=== Input check ===")
        print(f"IMU: {'OK' if len(self.imu)>0 else 'MISSING'} ({len(self.imu)} samples)")
        print(f"Images: {'OK' if len(self.imgs)>0 else 'None'} ({len(self.imgs)} frames)")
        print(f"VPS: {'OK' if len(self.vps_list)>0 else 'None'} ({len(self.vps_list)} items)")
        print(f"Mag: {'OK' if len(self.mag_list)>0 else 'None'} ({len(self.mag_list)} samples)")
        print(f"DEM: {'OK' if self.dem.ds else 'None'}")
        print(f"PPK: {'OK' if self.ppk_state else 'None'}")
        
        if len(self.imu) == 0:
            raise RuntimeError("IMU is required. Aborting.")
    
    def initialize_ekf(self):
        """Initialize EKF state and covariance."""
        # Ensure local projection is set up
        ensure_local_proj(self.lat0, self.lon0)
        
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
        
        # Create EKF
        self.kf = ExtendedKalmanFilter(dim_x=16, dim_z=3, dim_u=3)
        self.kf.x = np.zeros((16, 1), dtype=float)
        
        # Get config parameters from YAML (v3.2.0: no fallback - require proper config)
        imu_params = self.global_config.get('IMU_PARAMS')
        if imu_params is None:
            raise RuntimeError("IMU_PARAMS not found in config. Check YAML file.")
        
        lever_arm = self.global_config.get('IMU_GNSS_LEVER_ARM', np.zeros(3))
        
        # v2.9.10.1 FIX: Extract PPK initial heading (GPS-denied: 2 samples only)
        # FIXED: Now uses only first 2 samples (t=0 velocity) instead of 30s trajectory
        ppk_initial_heading = None
        if self.ppk_trajectory is not None:
            from .data_loaders import get_ppk_initial_heading
            ppk_initial_heading = get_ppk_initial_heading(
                self.ppk_trajectory, self.lat0, self.lon0
            )
        
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
            }
        
        # Initialize state
        init_state = initialize_ekf_state(
            kf=self.kf,
            ppk_state=self.ppk_state,
            imu_records=self.imu,
            imu_params=imu_params,
            lever_arm=lever_arm,
            lat0=self.lat0, lon0=self.lon0,
            msl0=self.msl0, dem0=self.dem0,
            v_init_enu=self.v_init,
            z_state=self.config.z_state,
            estimate_imu_bias=self.config.estimate_imu_bias,
            initial_gyro_bias=self.global_config.get('INITIAL_GYRO_BIAS'),
            initial_accel_bias=self.global_config.get('INITIAL_ACCEL_BIAS'),
            mag_records=self.mag_list if mag_params else None,
            mag_params=mag_params,
            ppk_initial_heading=ppk_initial_heading  # v2.9.10.0 Priority 1
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
        self.vio_fe.camera_view = self.config.camera_view
        
        print(f"[VIO] Camera view mode: {self.config.camera_view}")
        
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
        self.inf_csv = csv_paths['inf_csv']
        self.vo_dbg_csv = csv_paths['vo_dbg']
        self.msckf_dbg_csv = csv_paths['msckf_dbg']
        
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
        
        # Save calibration snapshot using output_utils
        if self.config.save_debug_data:
            cal_path = os.path.join(self.config.output_dir, "debug_calibration.txt")
            
            # Get camera view config from loaded YAML (not hardcoded defaults)
            view_cfg = self.global_config.get('CAMERA_VIEW_CONFIGS', {}).get(
                self.config.camera_view,
                CAMERA_VIEW_CONFIGS.get(self.config.camera_view, CAMERA_VIEW_CONFIGS['nadir'])
            )
            
            kb_params = self.global_config.get('KB_PARAMS', {})
            imu_params = self.global_config.get('IMU_PARAMS', {})
            mag_params = {
                'declination': self.global_config.get('MAG_DECLINATION', 0.0),
                'hard_iron': self.global_config.get('MAG_HARD_IRON_OFFSET', None),
                'soft_iron': self.global_config.get('MAG_SOFT_IRON_MATRIX', None),
            }
            noise_params = {
                'sigma_vo_vel': self.global_config.get('SIGMA_VO', 0.5),
                'sigma_mag_yaw': self.global_config.get('SIGMA_MAG_YAW', 0.15),
                'sigma_agl_z': self.global_config.get('SIGMA_AGL_Z', 2.5),
            }
            vio_params = {
                'use_vio_velocity': self.config.use_vio_velocity,
                'use_magnetometer': self.config.use_magnetometer,
                'camera_view': self.config.camera_view,
                'z_state': self.config.z_state,
            }
            initial_state = {
                'lat0': self.lat0,
                'lon0': self.lon0,
                'alt0': getattr(self, 'alt0', 0.0),
            }
            # Get plane config
            plane_config = self.global_config.get('plane', None)
            
            save_calibration_log(
                output_path=cal_path,
                camera_view=self.config.camera_view,
                view_cfg=view_cfg,
                kb_params=kb_params,
                imu_params=imu_params,
                mag_params=mag_params,
                noise_params=noise_params,
                vio_params=vio_params,
                initial_state=initial_state,
                estimate_imu_bias=self.config.estimate_imu_bias,
                plane_config=plane_config,
                global_config=self.global_config,
                vio_config=self.config  # Pass VIOConfig for full audit
            )
            print(f"[DEBUG] Calibration snapshot saved: {cal_path}")
    
    def _update_imu_helpers(self, rec, dt: float, imu_params: dict):
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
        
        # Check for stationary (ZUPT)
        v_mag = np.linalg.norm(x[3:6])
        is_stationary, _ = detect_stationary(
            a_raw=rec.lin,
            w_corr=w_corr,
            v_mag=v_mag,
            imu_params=imu_params
        )
        
        if is_stationary:
            self.state.zupt_detected += 1
            
            applied, v_reduction, updated_count = apply_zupt(
                self.kf,
                v_mag=v_mag,
                consecutive_stationary_count=self.state.consecutive_stationary,
                save_debug=self.config.save_debug_data,
                residual_csv=getattr(self, 'residual_csv', None),
                timestamp=rec.t,
                frame=self.state.imu_propagation_count
            )
            
            if applied:
                self.state.zupt_applied += 1
                self.state.consecutive_stationary = updated_count
            else:
                self.state.zupt_rejected += 1
        else:
            self.state.consecutive_stationary = 0
        
        # Save priors
        self.kf.x_prior = self.kf.x.copy()
        self.kf.P_prior = self.kf.P.copy()
    
    def process_imu_sample(self, rec, dt: float):
        """
        Process single IMU sample (legacy mode: propagation + helpers).
        
        Args:
            rec: IMU record
            dt: Time delta since last sample
        """
        imu_params = self.global_config.get('IMU_PARAMS', {'g_norm': 9.803})
        
        # Propagate state + covariance
        process_imu(
            self.kf, rec, dt,
            estimate_imu_bias=self.config.estimate_imu_bias,
            t=rec.t, t0=self.state.t0,
            imu_params=imu_params
        )
        
        # Update helpers (ZUPT, mag filter)
        self._update_imu_helpers(rec, dt, imu_params)
    
    def process_vps(self, t: float):
        """
        Process VPS measurements up to current time with innovation gating.
        
        Args:
            t: Current timestamp
        """
        from .vps_integration import (
            compute_vps_innovation, compute_vps_acceptance_threshold
        )
        
        sigma_vps = self.global_config.get('SIGMA_VPS_XY', 1.0)
        
        while (self.state.vps_idx < len(self.vps_list) and 
               self.vps_list[self.state.vps_idx].t <= t):
            
            vps = self.vps_list[self.state.vps_idx]
            self.state.vps_idx += 1
            
            # Compute innovation and Mahalanobis distance
            vps_xy, innovation, m2_test = compute_vps_innovation(
                vps, self.kf, self.lat0, self.lon0
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
                continue
            
            if m2_test > chi2_threshold * 10:  # Very permissive for first VPS
                print(f"[VPS] REJECTED: chi2={m2_test:.1f} >> {chi2_threshold} "
                      f"(innovation={innovation_mag:.1f}m)")
                continue
            
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
    
    def process_vio(self, rec, t: float, ongoing_preint=None):
        """
        Process VIO (Visual-Inertial Odometry) updates.
        
        This handles:
        1. Image loading and feature tracking
        2. Loop closure detection (optional)
        3. Preintegration application at camera frame
        4. Camera cloning for MSCKF
        5. MSCKF multi-view updates
        6. VIO velocity updates with scale recovery
        7. Plane constraint updates
        
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
                    preint_jacobians=preint_jacobians  # Pass Jacobians for bias observability
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
                            global_config=self.global_config
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
                    t_cam, self.kf, self.ppk_trajectory, self.flight_log_df,
                    self.lat0, self.lon0, 'velocity')
                
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
                    z_state=self.config.z_state,
                    use_vio_velocity=True,  # Always True when entering this block
                    save_debug=self.config.save_debug_data,
                    residual_csv=self.residual_csv if self.config.save_debug_data else None,
                    vio_frame=self.state.vio_frame,
                    vio_fe=self.vio_fe,
                    state_error=vel_error,  # For NEES calculation
                    state_cov=vel_cov       # For NEES calculation
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
        gt_df = self.ppk_trajectory if self.ppk_trajectory is not None else self.flight_log_df
        if gt_df is None or len(gt_df) == 0:
            return
        
        try:
            # Find closest ground truth
            gt_idx = np.argmin(np.abs(gt_df['stamp_log'].values - t))
            gt_row = gt_df.iloc[gt_idx]
            
            use_ppk = self.ppk_trajectory is not None
            
            if use_ppk:
                gt_lat = gt_row['lat']
                gt_lon = gt_row['lon']
                gt_alt = gt_row['height']
            else:
                gt_lat = gt_row['lat_dd']
                gt_lon = gt_row['lon_dd']
                gt_alt = gt_row['altitude_MSL_m']
            
            gt_E, gt_N = latlon_to_xy(gt_lat, gt_lon, self.lat0, self.lon0)
            gt_U = gt_alt
            
            # VIO prediction (IMU position)
            vio_E = float(self.kf.x[0, 0])
            vio_N = float(self.kf.x[1, 0])
            vio_U = float(self.kf.x[2, 0])
            
            # Ground truth is GNSS position - convert VIO (IMU) to GNSS for fair comparison
            lever_arm = self.global_config.get('IMU_GNSS_LEVER_ARM', np.zeros(3))
            if np.linalg.norm(lever_arm) > 0.01:
                from scipy.spatial.transform import Rotation as R_scipy
                q_vio = self.kf.x[6:10, 0]
                q_xyzw = np.array([q_vio[1], q_vio[2], q_vio[3], q_vio[0]])
                R_body_to_world = R_scipy.from_quat(q_xyzw).as_matrix()
                
                p_imu_enu = np.array([vio_E, vio_N, vio_U])
                p_gnss_enu = imu_to_gnss_position(p_imu_enu, R_body_to_world, lever_arm)
                vio_E, vio_N, vio_U = p_gnss_enu[0], p_gnss_enu[1], p_gnss_enu[2]
            
            # Errors
            err_E = vio_E - gt_E
            err_N = vio_N - gt_N
            err_U = vio_U - gt_U
            pos_error = np.sqrt(err_E**2 + err_N**2 + err_U**2)
            
            # Velocity error (compute from consecutive GPS positions)
            vel_error = 0.0
            vel_err_E = vel_err_N = vel_err_U = 0.0
            
            if gt_idx > 0 and gt_idx < len(gt_df) - 1:
                # Use central difference for velocity
                gt_row_prev = gt_df.iloc[gt_idx - 1]
                gt_row_next = gt_df.iloc[gt_idx + 1]
                
                dt = gt_row_next['stamp_log'] - gt_row_prev['stamp_log']
                if dt > 0.01:  # Avoid division by zero
                    if use_ppk:
                        gt_E_prev, gt_N_prev = latlon_to_xy(gt_row_prev['lat'], gt_row_prev['lon'], self.lat0, self.lon0)
                        gt_E_next, gt_N_next = latlon_to_xy(gt_row_next['lat'], gt_row_next['lon'], self.lat0, self.lon0)
                        gt_U_prev = gt_row_prev['height']
                        gt_U_next = gt_row_next['height']
                    else:
                        gt_E_prev, gt_N_prev = latlon_to_xy(gt_row_prev['lat_dd'], gt_row_prev['lon_dd'], self.lat0, self.lon0)
                        gt_E_next, gt_N_next = latlon_to_xy(gt_row_next['lat_dd'], gt_row_next['lon_dd'], self.lat0, self.lon0)
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
            
            # Calibrate using hard-iron and soft-iron from config
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
                timestamp=t,
                residual_csv=residual_path,
                frame=self.state.vio_frame,
                yaw_override=yaw_mag_filtered,  # NEW: pass filtered yaw directly
                filter_info=filter_info  # v2.9.2: track filter rejection reasons
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
        lat_now, lon_now = xy_to_latlon(
            self.kf.x[0, 0], self.kf.x[1, 0],
            self.lat0, self.lon0
        )
        
        # Sample DEM
        dem_now = self.dem.sample_m(lat_now, lon_now) if self.dem.ds else None
        
        if dem_now is None or np.isnan(dem_now):
            return
        
        # Compute height measurement
        if self.config.z_state.lower() == "agl":
            height_m = self.kf.x[2, 0]  # State is AGL
        else:
            # Use flight log MSL if available
            msl_measured = None
            if self.flight_log_df is not None:
                idx = np.argmin(np.abs(self.flight_log_df['stamp_log'].values - t))
                if abs(self.flight_log_df['stamp_log'].iloc[idx] - t) < 0.5:
                    msl_measured = float(self.flight_log_df['altitude_MSL_m'].iloc[idx])
            
            if msl_measured is not None:
                height_m = msl_measured
            else:
                height_m = self.kf.x[2, 0]
        
        # Compute uncertainties
        xy_uncertainty = float(np.trace(self.kf.P[0:2, 0:2]))
        time_since_correction = t - getattr(self.kf, 'last_absolute_correction_time', t)
        speed = float(np.linalg.norm(self.kf.x[3:6, 0]))
        
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
            frame=self.state.vio_frame
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
            lat_now, lon_now = xy_to_latlon(
                self.kf.x[0, 0], self.kf.x[1, 0],
                self.lat0, self.lon0
            )
        
        if msl_now is None or agl_now is None:
            # Compute MSL/AGL from current state (post-DEM update)
            dem_now = self.dem.sample_m(lat_now, lon_now) if self.dem.ds else 0.0
            if dem_now is None:
                dem_now = 0.0
            
            if self.config.z_state.lower() == "agl":
                agl_now = self.kf.x[2, 0]
                msl_now = agl_now + dem_now
            else:
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
        
        print_msckf_stats()
        
        # Print error statistics using dedicated function
        from .output_utils import print_error_statistics
        print_error_statistics(self.error_csv)
    
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
        
        # =================================================================
        # Architecture Selection (v3.7.0: enum-based)
        # =================================================================
        if self.config.estimator_mode == "event_queue_output_predictor":
            # Event-driven mode: propagate to each measurement time
            print("\n[ARCH] Selected: Event-driven (event_queue_output_predictor)")
            print("[TODO] Event-driven architecture not implemented yet!")
            print("[TODO] Will use propagate_to_timestamp() for exact timing")
            print("[TODO] See analysis in conversation for implementation details")
            raise NotImplementedError(
                "Event-driven architecture not yet implemented. "
                "Set estimator_mode: imu_step_preint_cache in YAML to use IMU-driven mode."
            )
        elif self.config.estimator_mode == "imu_step_preint_cache":
            # IMU-driven mode: process all IMU @ 400Hz with sub-sample precision
            print("\n[ARCH] Selected: IMU-driven with sub-sample timestamp precision (imu_step_preint_cache)")
            self._run_imu_driven()
        else:
            raise ValueError(f"Unknown estimator_mode: {self.config.estimator_mode}")
    
    def _run_imu_driven(self):
        """
        IMU-driven VIO loop (v3.6.0).
        
        Architecture:
        - Iterate through all IMU samples @ 400Hz
        - Accumulate preintegration buffer for MSCKF Jacobians
        - Propagate state + covariance every IMU tick
        - Process measurements when their timestamps are reached
        
        Advantages:
        - Simple sequential processing
        - Streaming-friendly (no need to buffer all data)
        - Clear separation between propagation and update
        
        Disadvantages:
        - State can lag measurement time by ~2.5ms (1 IMU sample)
        - Need careful timestamp handling for camera events
        """
        # Load data
        self.load_data()
        
        # Initialize EKF
        self.initialize_ekf()
        
        # Initialize VIO frontend
        self.initialize_vio_frontend()
        
        # Initialize fisheye rectifier (optional - for converting fisheye to pinhole)
        self._initialize_rectifier()
        
        # Initialize loop closure detector (for yaw drift correction)
        self._initialize_loop_closure()
        
        # Initialize magnetometer filter state
        reset_mag_filter_state()
        
        # Set magnetometer constants from config
        mag_ema_alpha = self.global_config.get('MAG_EMA_ALPHA', 0.3)
        mag_max_yaw_rate_deg = self.global_config.get('MAG_MAX_YAW_RATE_DEG', 30.0)
        mag_gyro_threshold_deg = self.global_config.get('MAG_GYRO_THRESHOLD_DEG', 10.0)
        mag_r_inflate = self.global_config.get('MAG_R_INFLATE', 5.0)
        set_mag_constants(
            ema_alpha=mag_ema_alpha,
            max_yaw_rate=np.radians(mag_max_yaw_rate_deg),
            gyro_threshold=np.radians(mag_gyro_threshold_deg),
            consistency_r_inflate=mag_r_inflate
        )
        print(f"[MAG-FILTER] Initialized: EMA_α={mag_ema_alpha:.2f}, max_rate={mag_max_yaw_rate_deg:.1f}°/s, gyro_thresh={mag_gyro_threshold_deg:.1f}°")
        
        # Initialize vibration detector
        imu_params = self.global_config.get('IMU_PARAMS', {})
        vib_buffer_size = self.global_config.get('VIBRATION_WINDOW_SIZE', 50)
        vib_threshold_mult = self.global_config.get('VIBRATION_THRESHOLD_MULT', 5.0)
        self.vibration_detector = VibrationDetector(
            buffer_size=vib_buffer_size,
            threshold=imu_params.get('acc_n', 0.1) * vib_threshold_mult
        )
        
        # Initialize TRN (Terrain Referenced Navigation) - v3.3.0
        if self.global_config.get('TRN_ENABLED', False):
            self.trn = create_trn_from_config(self.dem, self.global_config)
            if self.trn:
                print(f"[TRN] Initialized: search_radius={self.global_config.get('TRN_SEARCH_RADIUS', 500)}m, "
                      f"update_interval={self.global_config.get('TRN_UPDATE_INTERVAL', 10)}s")
            else:
                print("[TRN] WARNING: Failed to initialize TRN")
        else:
            print("[TRN] Disabled in config")
        
        # Store initial MSL for altitude change tracking
        self.initial_msl = self.msl0
        
        # Setup output files
        self.setup_output_files()
        
        # Initialize timing
        self.state.t0 = self.imu[0].t
        self.state.last_t = self.state.t0
        
        # Initialize preintegration cache for MSCKF Jacobians
        imu_params = self.global_config.get('IMU_PARAMS', {})
        ongoing_preint = IMUPreintegration(
            bg=self.kf.x[10:13, 0].reshape(3,),
            ba=self.kf.x[13:16, 0].reshape(3,),
            sigma_g=imu_params.get('gyr_n', 0.01),
            sigma_a=imu_params.get('acc_n', 0.1),
            sigma_bg=imu_params.get('gyr_w', 0.0001),
            sigma_ba=imu_params.get('acc_w', 0.001)
        )
        print(f"[PREINT] Initialized preintegration cache for MSCKF Jacobians")
        
        # Main IMU-driven loop
        print("\n=== Running (IMU-driven with preintegration cache) ===")
        tic_all = time.time()
        
        # Store last a_world for state_debug logging
        last_a_world = np.array([0.0, 0.0, 0.0])
        
        for i, rec in enumerate(self.imu):
            # Start timing for inference_log
            tic_iter = time.time()
            
            t = rec.t
            dt = max(0.0, float(t - self.state.last_t)) if i > 0 else 0.0
            
            time_elapsed = t - self.state.t0
            
            # Update vibration detector
            gyro_mag = np.linalg.norm(rec.ang)
            is_high_vibration, vibration_level = self.vibration_detector.update(gyro_mag)
            
            # Get current velocity and uncertainty for state-based phase detection
            velocity = self.kf.x[3:6, 0].flatten() if self.kf is not None else None
            velocity_sigma = None
            if self.kf is not None and self.kf.P.shape[0] > 5:
                velocity_sigma = np.sqrt(np.mean(np.diag(self.kf.P[3:6, 3:6])))
            
            altitude_change = self.kf.x[2, 0] - self.initial_msl if self.kf is not None else 0.0
            
            # Get flight phase (v3.4.0: pure state-based with config thresholds)
            phase_num, _ = get_flight_phase(
                velocity=velocity,
                velocity_sigma=velocity_sigma,
                vibration_level=vibration_level,
                altitude_change=altitude_change,
                spinup_velocity_thresh=self.PHASE_SPINUP_VELOCITY_THRESH,
                spinup_vibration_thresh=self.PHASE_SPINUP_VIBRATION_THRESH,
                spinup_alt_change_thresh=self.PHASE_SPINUP_ALT_CHANGE_THRESH,
                early_velocity_sigma_thresh=self.PHASE_EARLY_VELOCITY_SIGMA_THRESH
            )
            self.state.current_phase = phase_num
            
            # =====================================================================
            # IMU Propagation (v3.7.0: IMU-driven with sub-sample timestamp precision)
            # =====================================================================
            # Architecture: Always propagate x+P at every IMU tick (400Hz)
            # Sub-sample precision: Split dt when crossing camera timestamp (OpenVINS-style)
            # Preintegration cache is ONLY for MSCKF Jacobians, NOT to skip propagation
            
            # Check if we cross camera timestamp during this IMU step
            next_cam_time = self.imgs[self.state.img_idx].t if self.state.img_idx < len(self.imgs) else float('inf')
            t_last = self.state.last_t
            t_current = rec.t
            
            # Detect camera crossing: last_t < t_cam <= current_t
            if t_last < next_cam_time <= t_current and dt > 0:
                # Split dt at camera timestamp for sub-sample precision
                dt_before = next_cam_time - t_last  # Propagate to camera time
                dt_after = t_current - next_cam_time  # Propagate remaining
                
                # Step 1a: Propagate to camera time (exact timestamp)
                ongoing_preint.integrate_measurement(rec.ang, rec.lin, dt_before)
                process_imu(
                    self.kf, rec, dt_before,
                    estimate_imu_bias=self.config.estimate_imu_bias,
                    t=next_cam_time, t0=self.state.t0,
                    imu_params=imu_params
                )
                self._update_imu_helpers(rec, dt_before, imu_params)
                self.state.last_t = next_cam_time
                
                # Process camera at exact timestamp
                # VIO updates (feature tracking, MSCKF, velocity) at t_cam
                used_vo, vo_data = self.process_vio(rec, next_cam_time, ongoing_preint)
                
                # Step 1b: Propagate remaining dt after camera
                if dt_after > 1e-6:  # Only if significant remaining time
                    ongoing_preint.integrate_measurement(rec.ang, rec.lin, dt_after)
                    process_imu(
                        self.kf, rec, dt_after,
                        estimate_imu_bias=self.config.estimate_imu_bias,
                        t=t_current, t0=self.state.t0,
                        imu_params=imu_params
                    )
                    self._update_imu_helpers(rec, dt_after, imu_params)
                self.state.last_t = t_current
            else:
                # Normal propagation (no camera crossing)
                # Step 1: Accumulate IMU measurements in preintegration cache for MSCKF
                ongoing_preint.integrate_measurement(rec.ang, rec.lin, dt)
                
                # Step 2: Propagate state + covariance (REQUIRED for consistent EKF updates)
                # This ensures BOTH x and P are at current time when VPS/MAG/DEM arrive
                process_imu(
                    self.kf, rec, dt,
                    estimate_imu_bias=self.config.estimate_imu_bias,
                    t=rec.t, t0=self.state.t0,
                    imu_params=imu_params
                )
                
                # Step 3: Update mag filter variables and check ZUPT
                self._update_imu_helpers(rec, dt, imu_params)
                
                # Update state time
                self.state.last_t = t
                
                # VIO updates (feature tracking, MSCKF, velocity) if no camera crossed
                used_vo, vo_data = self.process_vio(rec, t, ongoing_preint)
            
            # Increment counter for debug logging
            self.state.imu_propagation_count += 1
            
            # Debug logging: raw IMU and state covariance
            self.debug_writers.log_imu_raw(t, rec)
            if i % 10 == 0:  # Every 10 samples (~25ms @ 400Hz)
                bg = self.kf.x[10:13, 0]
                ba = self.kf.x[13:16, 0]
                self.debug_writers.log_state_covariance(t, self.state.vio_frame, self.kf, bg, ba)
            
            # VPS updates
            self.process_vps(t)
            
            # Magnetometer updates
            if self.config.use_magnetometer:
                self.process_magnetometer(t)
            
            # Capture current MSL/AGL BEFORE DEM update (matches vio_vps.py behavior)
            lat_now, lon_now = xy_to_latlon(
                self.kf.x[0, 0], self.kf.x[1, 0],
                self.lat0, self.lon0
            )
            dem_now = self.dem.sample_m(lat_now, lon_now) if self.dem.ds else 0.0
            if dem_now is None:
                dem_now = 0.0
            if self.config.z_state.lower() == "agl":
                agl_now = self.kf.x[2, 0]
                msl_now = agl_now + dem_now
            else:
                msl_now = self.kf.x[2, 0]
                agl_now = msl_now - dem_now
            
            # DEM height updates (modifies kf.x[2,0], but we use pre-update values for logging)
            self.process_dem_height(t)
            
            # TRN (Terrain Referenced Navigation) updates - v3.3.0
            if self.trn is not None:
                # Update altitude history for profile matching
                self.trn.update_altitude_history(
                    timestamp=t,
                    msl_altitude=msl_now,
                    estimated_x=self.kf.x[0, 0],
                    estimated_y=self.kf.x[1, 0]
                )
                
                # Try TRN position fix
                if self.trn.check_update_needed(t):
                    trn_success = self.trn.apply_trn_update(
                        kf=self.kf,
                        lat0=self.lat0,
                        lon0=self.lon0,
                        current_time=t
                    )
            
            # Log error vs ground truth (every sample, like vio_vps.py)
            self.log_error(t)
            
            # Log pose (using pre-DEM-update msl_now/agl_now for consistency with vio_vps.py)
            self.log_pose(t, dt, used_vo, vo_data, msl_now=msl_now, agl_now=agl_now, 
                          lat_now=lat_now, lon_now=lon_now)
            
            # Log inference timing (every sample, like vio_vps.py)
            toc_iter = time.time()
            with open(self.inf_csv, "a", newline="") as f:
                dt_proc = toc_iter - tic_iter
                fps = (1.0 / dt_proc) if dt_proc > 0 else 0.0
                f.write(f"{i},{dt_proc:.6f},{fps:.2f}\n")
            
            # Log state debug (every sample, like vio_vps.py)
            log_state_debug(self.state_dbg_csv, t, self.kf, dem_now, agl_now, msl_now, last_a_world)
            
            # Update last_a_world for next iteration
            # Get a_world from last IMU sample processing
            try:
                # Convert quaternion to rotation matrix (state is [w,x,y,z])
                q = self.kf.x[6:10, 0].flatten()
                quat_xyzw = np.array([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w]
                r_body_to_world = R_scipy.from_quat(quat_xyzw).as_matrix()
                
                # Compute world-frame acceleration with gravity subtraction
                a_body = rec.lin - self.kf.x[13:16, 0].flatten()  # bias-corrected
                g_norm = self.global_config.get('IMU_PARAMS', {}).get('g_norm', 9.8066)
                g_world = np.array([0.0, 0.0, -g_norm])  # ENU: gravity points down
                a_world_raw = r_body_to_world @ a_body
                last_a_world = a_world_raw + g_world  # Subtract gravity
            except Exception:
                pass
            
            # Progress
            if i % 1000 == 0:
                speed_ms = float(np.linalg.norm(self.kf.x[3:6, 0]))
                print(f"t={time_elapsed:8.3f}s | speed={speed_ms*3.6:5.1f}km/h | "
                      f"phase={self.PHASE_NAMES[self.state.current_phase]}", end="\r")
        
        toc_all = time.time()
        
        # Print summary
        self.print_summary()
        print(f"\n=== Finished in {toc_all - tic_all:.2f} seconds ===")


def run_vio(config: VIOConfig):
    """
    Convenience function to run VIO pipeline.
    
    Args:
        config: VIOConfig instance
    """
    runner = VIORunner(config)
    runner.run()
