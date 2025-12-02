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


@dataclass
class VIOConfig:
    """Configuration for VIO runner."""
    # Required paths
    imu_path: str
    quarry_path: str
    output_dir: str
    
    # Optional data paths
    images_dir: Optional[str] = None
    images_index_csv: Optional[str] = None
    vps_csv: Optional[str] = None
    mag_csv: Optional[str] = None
    dem_path: Optional[str] = None
    ground_truth_path: Optional[str] = None
    config_yaml: Optional[str] = None
    
    # Image processing
    downscale_size: Tuple[int, int] = (1140, 1080)
    
    # State options
    z_state: str = "msl"  # "msl" or "agl"
    camera_view: str = "nadir"  # "nadir", "front", "side", "multi"
    
    # Algorithm options
    estimate_imu_bias: bool = False
    use_magnetometer: bool = True
    use_vio_velocity: bool = True
    use_preintegration: bool = True
    
    # Debug options
    save_debug_data: bool = False
    save_keyframe_images: bool = False


@dataclass
class VIOState:
    """Runtime state container."""
    # Indices and counters
    imu_idx: int = 0
    img_idx: int = 0
    vps_idx: int = 0
    mag_idx: int = 0
    vio_frame: int = -1
    
    # Statistics
    zupt_applied: int = 0
    zupt_rejected: int = 0
    zupt_detected: int = 0
    mag_updates: int = 0
    mag_rejects: int = 0
    consecutive_stationary: int = 0
    
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
    
    # Flight phase constants
    PHASE_SPINUP_END = 15.0      # seconds
    PHASE_EARLY_END = 60.0       # seconds
    PHASE_NAMES = ["SPINUP", "EARLY", "NORMAL"]
    
    def __init__(self, config: VIOConfig):
        """
        Initialize VIO runner.
        
        Args:
            config: VIOConfig instance with all settings
        """
        self.config = config
        self.state = VIOState()
        
        # These will be initialized in run()
        self.kf = None
        self.imu = None
        self.imgs = None
        self.vps_list = None
        self.mag_list = None
        self.dem = None
        self.vio_fe = None
        
        # Origin coordinates
        self.lat0 = 0.0
        self.lon0 = 0.0
        self.msl0 = 0.0
        self.dem0 = None
        
        # Global config (loaded from YAML)
        self.global_config = {}
        
        # Output file handles
        self.pose_csv = None
        self.error_csv = None
        self.state_dbg_csv = None
    
    def load_config(self) -> dict:
        """Load YAML configuration file."""
        from .config import load_config
        
        if self.config.config_yaml:
            cfg = load_config(self.config.config_yaml)
            self.global_config = cfg
            return cfg
        return {}
    
    def load_data(self):
        """Load all input data sources."""
        from .data_loaders import (
            load_imu_csv, load_images, load_vps_csv, load_mag_csv,
            DEMReader, load_ppk_initial_state, load_msl_from_gga,
            load_quarry_initial, load_ppk_trajectory
        )
        
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
        from .ekf import ExtendedKalmanFilter
        from .state_manager import initialize_ekf_state
        from .vps_integration import ensure_local_proj
        
        # Ensure local projection is set up
        ensure_local_proj(self.lat0, self.lon0)
        
        # Sample DEM at origin
        self.dem0 = self.dem.sample_m(self.lat0, self.lon0) if self.dem.ds else None
        
        # Create EKF
        self.kf = ExtendedKalmanFilter(dim_x=16, dim_z=3, dim_u=3)
        self.kf.x = np.zeros((16, 1), dtype=float)
        
        # Get config parameters
        imu_params = self.global_config.get('IMU_PARAMS', {
            'g_norm': 9.803,
            'gyr_w': 0.0001,
            'acc_w': 0.001,
            'gyr_n': 0.01,
            'acc_n': 0.1
        })
        lever_arm = self.global_config.get('IMU_GNSS_LEVER_ARM', np.zeros(3))
        
        # MAG params for initial correction
        mag_params = None
        if self.config.use_magnetometer and len(self.mag_list) > 0:
            mag_params = {
                'declination': self.global_config.get('MAG_DECLINATION', 0.0),
                'use_raw_heading': self.global_config.get('MAG_USE_RAW_HEADING', True),
                'min_field': self.global_config.get('MAG_MIN_FIELD_STRENGTH', 0.1),
                'max_field': self.global_config.get('MAG_MAX_FIELD_STRENGTH', 100.0),
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
            mag_params=mag_params
        )
        
        return init_state
    
    def initialize_vio_frontend(self):
        """Initialize VIO frontend if images are available."""
        from .vio_frontend import VIOFrontEnd
        from .camera import make_KD_for_size
        
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
            use_fisheye=use_fisheye
        )
        self.vio_fe.camera_view = self.config.camera_view
        
        print(f"[VIO] Camera view mode: {self.config.camera_view}")
    
    def setup_output_files(self):
        """
        Create output directory and all CSV files.
        
        Creates the following output files:
        - pose.csv: Main trajectory output (position, velocity, orientation)
        - error_log.csv: Error comparison vs ground truth
        - state_debug.csv: Detailed state evolution for debugging
        - inference_log.csv: Processing time per frame
        - vo_debug.csv: Visual odometry debug info
        - msckf_debug.csv: MSCKF update statistics
        
        If save_debug_data is enabled, also creates:
        - debug_imu_raw.csv: Raw IMU measurements
        - debug_state_covariance.csv: Covariance evolution
        - debug_residuals.csv: Innovation/residual statistics
        - debug_feature_stats.csv: Feature tracking quality
        - debug_msckf_window.csv: MSCKF sliding window state
        - debug_fej_consistency.csv: First-estimate Jacobian consistency
        - debug_calibration.txt: Calibration parameters snapshot
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # ===== CORE OUTPUT FILES =====
        
        # Pose output - main trajectory
        self.pose_csv = os.path.join(self.config.output_dir, "pose.csv")
        with open(self.pose_csv, "w") as f:
            f.write("Timestamp(s),dt,Frame,PX,PY,PZ_MSL,VX,VY,VZ,lat,lon,AGL(m),"
                    "vo_dx,vo_dy,vo_dz,vo_d_roll,vo_d_pitch,vo_d_yaw\n")
        
        # Error log - comparison with ground truth
        self.error_csv = os.path.join(self.config.output_dir, "error_log.csv")
        with open(self.error_csv, "w") as f:
            f.write("t,pos_error_m,pos_error_E,pos_error_N,pos_error_U,"
                    "vel_error_m_s,vel_error_E,vel_error_N,vel_error_U,"
                    "alt_error_m,yaw_vio_deg,yaw_gps_deg,yaw_error_deg,"
                    "gps_lat,gps_lon,gps_alt,vio_E,vio_N,vio_U\n")
        
        # State debug log - detailed state evolution
        self.state_dbg_csv = os.path.join(self.config.output_dir, "state_debug.csv")
        with open(self.state_dbg_csv, "w") as f:
            f.write("t,px,py,pz,vx,vy,vz,a_world_x,a_world_y,a_world_z,dem,agl,msl\n")
        
        # Inference timing log
        self.inf_csv = os.path.join(self.config.output_dir, "inference_log.csv")
        with open(self.inf_csv, "w") as f:
            f.write("Index,Inference Time (s),FPS\n")
        
        # VO debug log
        self.vo_dbg_csv = os.path.join(self.config.output_dir, "vo_debug.csv")
        with open(self.vo_dbg_csv, "w") as f:
            f.write("Frame,num_inliers,rot_angle_deg,alignment_deg,rotation_rate_deg_s,"
                    "use_only_vz,skip_vo,vo_dx,vo_dy,vo_dz,vel_vx,vel_vy,vel_vz\n")
        
        # MSCKF debug log
        self.msckf_dbg_csv = os.path.join(self.config.output_dir, "msckf_debug.csv")
        with open(self.msckf_dbg_csv, "w") as f:
            f.write("frame,feature_id,num_observations,triangulation_success,"
                    "reprojection_error_px,innovation_norm,update_applied,chi2_test\n")
        
        # ===== OPTIONAL DEBUG DATA FILES =====
        
        self.imu_raw_csv = None
        self.state_cov_csv = None
        self.residual_csv = None
        self.feature_stats_csv = None
        self.msckf_window_csv = None
        self.fej_csv = None
        self.keyframe_dir = None
        
        if self.config.save_debug_data:
            # Raw IMU data log
            self.imu_raw_csv = os.path.join(self.config.output_dir, "debug_imu_raw.csv")
            with open(self.imu_raw_csv, "w") as f:
                f.write("t,ori_x,ori_y,ori_z,ori_w,ang_x,ang_y,ang_z,lin_x,lin_y,lin_z\n")
            
            # State & covariance evolution
            self.state_cov_csv = os.path.join(self.config.output_dir, "debug_state_covariance.csv")
            with open(self.state_cov_csv, "w") as f:
                f.write("t,frame,P_pos_xx,P_pos_yy,P_pos_zz,P_vel_xx,P_vel_yy,P_vel_zz,"
                        "P_rot_xx,P_rot_yy,P_rot_zz,P_bg_xx,P_bg_yy,P_bg_zz,"
                        "P_ba_xx,P_ba_yy,P_ba_zz,bg_x,bg_y,bg_z,ba_x,ba_y,ba_z\n")
            
            # Residual & innovation log
            self.residual_csv = os.path.join(self.config.output_dir, "debug_residuals.csv")
            with open(self.residual_csv, "w") as f:
                f.write("t,frame,update_type,innovation_x,innovation_y,innovation_z,"
                        "mahalanobis_dist,chi2_threshold,accepted,NIS,NEES\n")
            
            # Feature tracking statistics
            self.feature_stats_csv = os.path.join(self.config.output_dir, "debug_feature_stats.csv")
            with open(self.feature_stats_csv, "w") as f:
                f.write("frame,t,num_features_detected,num_features_tracked,num_inliers,"
                        "mean_parallax_px,max_parallax_px,tracking_ratio,inlier_ratio\n")
            
            # MSCKF window & marginalization log
            self.msckf_window_csv = os.path.join(self.config.output_dir, "debug_msckf_window.csv")
            with open(self.msckf_window_csv, "w") as f:
                f.write("frame,t,num_camera_clones,num_tracked_features,num_mature_features,"
                        "window_start_time,window_duration,marginalized_clone_id\n")
            
            # FEJ consistency log
            self.fej_csv = os.path.join(self.config.output_dir, "debug_fej_consistency.csv")
            with open(self.fej_csv, "w") as f:
                f.write("timestamp,frame,clone_idx,pos_fej_drift_m,rot_fej_drift_deg,"
                        "bg_fej_drift_rad_s,ba_fej_drift_m_s2\n")
            
            print(f"[DEBUG] Debug data logging enabled")
        
        if self.config.save_keyframe_images:
            self.keyframe_dir = os.path.join(self.config.output_dir, "debug_keyframes")
            os.makedirs(self.keyframe_dir, exist_ok=True)
            print(f"[DEBUG] Keyframe images will be saved to: {self.keyframe_dir}")
        
        # Save calibration snapshot
        if self.config.save_debug_data:
            self._save_calibration_snapshot()
    
    def _save_calibration_snapshot(self):
        """Save calibration parameters for reproducibility."""
        cal_path = os.path.join(self.config.output_dir, "debug_calibration.txt")
        with open(cal_path, "w") as f:
            f.write("=== VIO System Calibration & Configuration ===\n\n")
            f.write(f"[Camera View]\n")
            f.write(f"  Mode: {self.config.camera_view}\n\n")
            
            f.write(f"[Image Processing]\n")
            f.write(f"  Downscale size: {self.config.downscale_size[0]}x{self.config.downscale_size[1]}\n\n")
            
            f.write(f"[Camera Intrinsics - Kannala-Brandt]\n")
            kb_params = self.global_config.get('KB_PARAMS', {})
            for key, val in kb_params.items():
                f.write(f"  {key}: {val}\n")
            f.write(f"\n")
            
            f.write(f"[IMU Parameters]\n")
            imu_params = self.global_config.get('IMU_PARAMS', {})
            for key, val in imu_params.items():
                f.write(f"  {key}: {val}\n")
            f.write(f"\n")
            
            f.write(f"[EKF Configuration]\n")
            f.write(f"  z_state: {self.config.z_state}\n")
            f.write(f"  estimate_imu_bias: {self.config.estimate_imu_bias}\n")
            f.write(f"  use_preintegration: {self.config.use_preintegration}\n")
            f.write(f"  use_magnetometer: {self.config.use_magnetometer}\n")
            f.write(f"  use_vio_velocity: {self.config.use_vio_velocity}\n\n")
            
            f.write(f"[Initial State]\n")
            f.write(f"  Origin: ({self.lat0:.8f}°, {self.lon0:.8f}°)\n")
            f.write(f"  MSL altitude: {self.msl0:.2f} m\n")
            if self.dem0 is not None:
                f.write(f"  DEM elevation: {self.dem0:.2f} m\n")
            f.write(f"  Initial velocity: {self.v_init} m/s\n")
        
        print(f"[DEBUG] Calibration snapshot saved: {cal_path}")
    
    def get_flight_phase(self, time_elapsed: float) -> int:
        """
        Determine current flight phase.
        
        Args:
            time_elapsed: Time since start in seconds
            
        Returns:
            Phase index: 0=SPINUP, 1=EARLY, 2=NORMAL
        """
        if time_elapsed < self.PHASE_SPINUP_END:
            return 0
        elif time_elapsed < self.PHASE_EARLY_END:
            return 1
        else:
            return 2
    
    def process_imu_sample(self, rec, dt: float, time_elapsed: float):
        """
        Process single IMU sample (propagation + ZUPT).
        
        Args:
            rec: IMU record
            dt: Time delta since last sample
            time_elapsed: Time since start
        """
        from .propagation import (
            propagate_single_imu_step, detect_stationary, apply_zupt
        )
        
        imu_params = self.global_config.get('IMU_PARAMS', {'g_norm': 9.803})
        
        # Get current state
        x = self.kf.x.reshape(-1)
        bg = x[10:13]
        ba = x[13:16]
        
        # Bias-corrected measurements
        a_corr = rec.lin.astype(float) - ba
        w_corr = rec.ang.astype(float) - bg
        
        # Propagate state
        if not self.config.use_preintegration:
            # Legacy propagation
            propagate_single_imu_step(
                self.kf, rec, dt,
                estimate_imu_bias=self.config.estimate_imu_bias,
                t=rec.t, t0=self.state.t0,
                imu_params=imu_params
            )
        
        # Check for stationary (ZUPT)
        is_stationary = detect_stationary(
            acc_raw=rec.lin,
            gyro_corrected=w_corr,
            g_norm=imu_params.get('g_norm', 9.803)
        )
        
        if is_stationary:
            self.state.zupt_detected += 1
            v_mag = np.linalg.norm(x[3:6])
            
            applied, self.state.consecutive_stationary = apply_zupt(
                self.kf,
                velocity_magnitude=v_mag,
                consecutive_count=self.state.consecutive_stationary
            )
            
            if applied:
                self.state.zupt_applied += 1
            else:
                self.state.zupt_rejected += 1
        else:
            self.state.consecutive_stationary = 0
        
        # Save priors
        self.kf.x_prior = self.kf.x.copy()
        self.kf.P_prior = self.kf.P.copy()
    
    def process_vps(self, t: float):
        """
        Process VPS measurements up to current time.
        
        Args:
            t: Current timestamp
        """
        from .vps_integration import apply_vps_update
        
        sigma_vps = self.global_config.get('SIGMA_VPS_XY', 1.0)
        
        while (self.state.vps_idx < len(self.vps_list) and 
               self.vps_list[self.state.vps_idx].t <= t):
            
            vps = self.vps_list[self.state.vps_idx]
            self.state.vps_idx += 1
            
            applied, reason = apply_vps_update(
                self.kf,
                vps_lat=vps.lat,
                vps_lon=vps.lon,
                origin_lat=self.lat0,
                origin_lon=self.lon0,
                sigma_base=sigma_vps,
                timestamp=t
            )
            
            if applied:
                print(f"[VPS] Applied at t={t:.3f}")
            else:
                print(f"[VPS] Rejected: {reason}")
    
    def process_vio(self, rec, t: float, time_elapsed: float, ongoing_preint=None):
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
            time_elapsed: Time since start
            ongoing_preint: Preintegration buffer (if enabled)
            
        Returns:
            Tuple of (used_vo, vo_data dict)
        """
        from .vio_frontend import VIOFrontEnd
        from .msckf import (
            augment_state_with_camera, perform_msckf_updates,
            compute_plane_constraint_jacobian
        )
        from .math_utils import quaternion_to_yaw, skew_symmetric
        from .propagation import propagate_error_state_covariance
        from .loop_closure import get_loop_detector
        
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
            img_path = self.imgs[self.state.img_idx].path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                self.state.img_idx += 1
                continue
            
            # Resize if needed
            if (img.shape[1], img.shape[0]) != tuple(self.config.downscale_size):
                img = cv2.resize(img, self.config.downscale_size, interpolation=cv2.INTER_AREA)
            
            if is_fast_rotation:
                print(f"[VIO] SKIPPING due to fast rotation: {rotation_rate_deg_s:.1f} deg/s")
                self.state.img_idx += 1
                continue
            
            # Run VIO frontend
            ok, ninl, r_vo_mat, t_unit, dt_img = self.vio_fe.step(img, self.imgs[self.state.img_idx].t)
            
            # Compute average optical flow (parallax)
            avg_flow_px = getattr(self.vio_fe, 'mean_parallax', 0.0)
            if avg_flow_px == 0.0 and self.vio_fe.last_matches is not None:
                focal_px = kb_params.get('mu', 600)
                pts_prev, pts_cur = self.vio_fe.last_matches
                if len(pts_prev) > 0:
                    pts_prev_px = pts_prev * focal_px + np.array([[self.vio_fe.img_w/2, self.vio_fe.img_h/2]])
                    pts_cur_px = pts_cur * focal_px + np.array([[self.vio_fe.img_w/2, self.vio_fe.img_h/2]])
                    flows = pts_cur_px - pts_prev_px
                    avg_flow_px = float(np.median(np.linalg.norm(flows, axis=1)))
            
            # Apply preintegration at EVERY camera frame
            if self.config.use_preintegration and ongoing_preint is not None:
                self._apply_preintegration_at_camera(ongoing_preint, t, imu_params)
            
            # Check parallax for VIO velocity update
            is_insufficient_parallax = avg_flow_px < min_parallax
            
            if is_insufficient_parallax:
                print(f"[VIO] SKIPPING velocity: parallax={avg_flow_px:.2f}px < {min_parallax}px")
                self.state.img_idx += 1
                continue
            
            # Camera cloning for MSCKF
            clone_threshold = min_parallax * 2.0
            should_clone = avg_flow_px >= clone_threshold and not is_fast_rotation
            
            if should_clone:
                self._clone_camera_for_msckf(t)
            
            # VIO velocity update (if frontend succeeded)
            if ok and r_vo_mat is not None and t_unit is not None:
                self._apply_vio_velocity_update(
                    r_vo_mat, t_unit, t, dt_img, 
                    avg_flow_px, rec
                )
                
                # Increment VIO frame
                self.state.vio_frame = max(0, self.state.vio_frame + 1)
                used_vo = True
                
                # Store VO data
                t_norm = t_unit / (np.linalg.norm(t_unit) + 1e-12)
                vo_dx, vo_dy, vo_dz = float(t_norm[0]), float(t_norm[1]), float(t_norm[2])
                r_eul = R_scipy.from_matrix(r_vo_mat).as_euler('zyx', degrees=True)
                vo_y, vo_p, vo_r = float(r_eul[0]), float(r_eul[1]), float(r_eul[2])
                
                vo_data = {
                    'dx': vo_dx, 'dy': vo_dy, 'dz': vo_dz,
                    'roll': vo_r, 'pitch': vo_p, 'yaw': vo_y
                }
            
            self.state.img_idx += 1
        
        return used_vo, vo_data
    
    def _apply_preintegration_at_camera(self, ongoing_preint, t: float, imu_params: dict):
        """
        Apply accumulated preintegration at camera frame.
        
        This is called at EVERY camera frame to prevent IMU accumulation explosion.
        Following Forster et al. TRO 2017 design.
        
        Args:
            ongoing_preint: IMUPreintegration object
            t: Current timestamp
            imu_params: IMU noise parameters
        """
        from .propagation import propagate_error_state_covariance
        from .math_utils import skew_symmetric
        
        dt_total = ongoing_preint.dt_sum
        if dt_total < 1e-6:
            return
        
        # Get deltas
        delta_R, delta_v, delta_p = ongoing_preint.get_deltas()
        
        # Current state
        p_i = self.kf.x[0:3, 0].reshape(3,)
        v_i = self.kf.x[3:6, 0].reshape(3,)
        q_i = self.kf.x[6:10, 0].reshape(4,)  # [w,x,y,z]
        bg = self.kf.x[10:13, 0].reshape(3,)
        ba = self.kf.x[13:16, 0].reshape(3,)
        
        # Get bias-corrected deltas
        delta_R_corr, delta_v_corr, delta_p_corr = ongoing_preint.get_deltas_corrected(bg, ba)
        
        # R_BW (Body-to-World)
        q_i_xyzw = np.array([q_i[1], q_i[2], q_i[3], q_i[0]])
        R_BW = R_scipy.from_quat(q_i_xyzw).as_matrix()
        
        # Rotation update
        R_BW_new = R_BW @ delta_R_corr
        q_i_new_xyzw = R_scipy.from_matrix(R_BW_new).as_quat()
        q_i_new = np.array([q_i_new_xyzw[3], q_i_new_xyzw[0], q_i_new_xyzw[1], q_i_new_xyzw[2]])
        
        # Position and velocity updates
        v_i_new = v_i + R_BW @ delta_v_corr
        p_i_new = p_i + v_i * dt_total + R_BW @ delta_p_corr
        
        # Write back to state
        self.kf.x[0:3, 0] = p_i_new.reshape(3,)
        self.kf.x[3:6, 0] = v_i_new.reshape(3,)
        self.kf.x[6:10, 0] = q_i_new.reshape(4,)
        
        # Propagate covariance
        preint_cov = ongoing_preint.get_covariance()
        J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba = ongoing_preint.get_jacobians()
        
        num_clones = (self.kf.x.shape[0] - 16) // 7
        
        # Build state transition matrix
        Phi_core = np.eye(15, dtype=float)
        Phi_core[0:3, 3:6] = np.eye(3) * dt_total
        Phi_core[0:3, 6:9] = -R_BW @ skew_symmetric(delta_p_corr)
        Phi_core[0:3, 9:12] = R_BW @ J_p_bg
        Phi_core[0:3, 12:15] = R_BW @ J_p_ba
        Phi_core[3:6, 6:9] = -R_BW @ skew_symmetric(delta_v_corr)
        Phi_core[3:6, 9:12] = R_BW @ J_v_bg
        Phi_core[3:6, 12:15] = R_BW @ J_v_ba
        Phi_core[6:9, 9:12] = -J_R_bg
        
        # Process noise
        Q_core = np.zeros((15, 15), dtype=float)
        Q_core[0:3, 0:3] = R_BW @ preint_cov[6:9, 6:9] @ R_BW.T
        Q_core[3:6, 3:6] = R_BW @ preint_cov[3:6, 3:6] @ R_BW.T
        Q_core[6:9, 6:9] = R_BW @ preint_cov[0:3, 0:3] @ R_BW.T
        Q_core[9:12, 9:12] = np.eye(3) * (imu_params.get('gyr_w', 0.0001)**2 * dt_total)
        Q_core[12:15, 12:15] = np.eye(3) * (imu_params.get('acc_w', 0.001)**2 * dt_total)
        
        self.kf.P = propagate_error_state_covariance(self.kf.P, Phi_core, Q_core, num_clones)
        
        # Reset preintegration buffer
        ongoing_preint.reset(bg=bg, ba=ba)
        
        print(f"[PREINT] Applied: Δt={dt_total:.3f}s, Δpos={np.linalg.norm(p_i_new - p_i):.4f}m")
    
    def _clone_camera_for_msckf(self, t: float):
        """
        Clone current IMU pose for MSCKF.
        
        Args:
            t: Camera timestamp
        """
        from .msckf import augment_state_with_camera
        
        p_imu = self.kf.x[0:3, 0].reshape(3,)
        q_imu = self.kf.x[6:10, 0].reshape(4,)
        
        try:
            start_idx = augment_state_with_camera(
                self.kf, q_imu, p_imu,
                self.state.cam_states, self.state.cam_observations
            )
            
            # Store FEJ linearization points
            clone_idx = len(self.state.cam_states)
            err_theta_idx = 15 + 6 * clone_idx
            err_p_idx = 15 + 6 * clone_idx + 3
            
            # Record observations
            obs_data = []
            if hasattr(self.vio_fe, 'get_tracks_for_frame'):
                tracks = self.vio_fe.get_tracks_for_frame(self.vio_fe.frame_idx)
                for fid, pt in tracks:
                    pt_array = np.array([[pt[0], pt[1]]], dtype=np.float32)
                    pt_norm = self.vio_fe._undistort_pts(pt_array).reshape(2,)
                    obs_data.append({
                        'fid': int(fid),
                        'pt_pixel': (float(pt[0]), float(pt[1])),
                        'pt_norm': (float(pt_norm[0]), float(pt_norm[1])),
                        'quality': 1.0
                    })
            
            self.state.cam_states.append({
                'start_idx': start_idx,
                'q_idx': start_idx,
                'p_idx': start_idx + 4,
                'err_q_idx': err_theta_idx,
                'err_p_idx': err_p_idx,
                't': t,
                'timestamp': t,
                'frame': self.vio_fe.frame_idx,
                'q_fej': q_imu.copy(),
                'p_fej': p_imu.copy(),
                'bg_fej': self.kf.x[10:13, 0].copy(),
                'ba_fej': self.kf.x[13:16, 0].copy()
            })
            
            self.state.cam_observations.append({
                'cam_id': clone_idx,
                'frame': self.vio_fe.frame_idx,
                't': t,
                'observations': obs_data
            })
            
            print(f"[CLONE] Created clone {clone_idx} with {len(obs_data)} observations")
            
            # Trigger MSCKF update if enough clones
            if len(self.state.cam_states) >= 3:
                self._trigger_msckf_update()
                
        except Exception as e:
            print(f"[CLONE] Failed: {e}")
    
    def _trigger_msckf_update(self):
        """Trigger MSCKF multi-view geometric update."""
        from .msckf import perform_msckf_updates
        
        # Count mature features
        feature_obs_count = {}
        for obs_set in self.state.cam_observations:
            for obs in obs_set['observations']:
                fid = obs['fid']
                feature_obs_count[fid] = feature_obs_count.get(fid, 0) + 1
        
        num_mature = sum(1 for c in feature_obs_count.values() if c >= 2)
        
        should_update = (
            num_mature >= 20 or
            len(self.state.cam_states) >= 4 or
            (self.vio_fe.frame_idx % 5 == 0 and len(self.state.cam_states) >= 3)
        )
        
        if should_update:
            try:
                num_updates = perform_msckf_updates(
                    self.vio_fe, 
                    self.state.cam_observations,
                    self.state.cam_states, 
                    self.kf,
                    min_observations=2,
                    max_features=50,
                    msckf_dbg_path=None,
                    dem_reader=self.dem,
                    origin_lat=self.lat0,
                    origin_lon=self.lon0
                )
                if num_updates > 0:
                    print(f"[MSCKF] Updated {num_updates} features")
            except Exception as e:
                print(f"[MSCKF] Error: {e}")
    
    def _apply_vio_velocity_update(self, r_vo_mat, t_unit, t: float, dt_img: float,
                                    avg_flow_px: float, rec):
        """
        Apply VIO velocity update with scale recovery.
        
        Args:
            r_vo_mat: Relative rotation matrix from Essential matrix
            t_unit: Unit translation vector from Essential matrix
            t: Current timestamp
            dt_img: Time between images
            avg_flow_px: Average optical flow in pixels
            rec: Current IMU record
        """
        from .math_utils import quaternion_to_yaw
        from .vps_integration import xy_to_latlon
        from .config import CAMERA_VIEW_CONFIGS
        
        kb_params = self.global_config.get('KB_PARAMS', {'mu': 600})
        sigma_vo = self.global_config.get('SIGMA_VO_VEL', 0.5)
        
        # Get camera extrinsics
        view_cfg = CAMERA_VIEW_CONFIGS.get(self.config.camera_view, CAMERA_VIEW_CONFIGS['nadir'])
        extrinsics_name = view_cfg['extrinsics']
        
        from .config import BODY_T_CAMDOWN, BODY_T_CAMFRONT, BODY_T_CAMSIDE
        if extrinsics_name == 'BODY_T_CAMDOWN':
            body_t_cam = BODY_T_CAMDOWN
        elif extrinsics_name == 'BODY_T_CAMFRONT':
            body_t_cam = BODY_T_CAMFRONT
        else:
            body_t_cam = BODY_T_CAMDOWN
        
        R_cam_to_body = body_t_cam[:3, :3]
        
        # Map direction
        t_norm = t_unit / (np.linalg.norm(t_unit) + 1e-12)
        t_body = R_cam_to_body @ t_norm
        
        # Get rotation from IMU quaternion
        q_imu = rec.q
        Rwb = R_scipy.from_quat(q_imu).as_matrix()
        
        # Scale recovery using AGL
        lat_now, lon_now = xy_to_latlon(self.kf.x[0, 0], self.kf.x[1, 0], self.lat0, self.lon0)
        dem_now = self.dem.sample_m(lat_now, lon_now) if self.dem.ds else 0.0
        if dem_now is None or np.isnan(dem_now):
            dem_now = 0.0
        
        if self.config.z_state.lower() == "agl":
            agl = abs(self.kf.x[2, 0])
        else:
            agl = abs(self.kf.x[2, 0] - dem_now)
        agl = max(1.0, agl)
        
        # Optical flow-based scale
        focal_px = kb_params.get('mu', 600)
        if dt_img > 1e-4 and avg_flow_px > 2.0:
            scale_flow = agl / focal_px
            speed_final = (avg_flow_px / dt_img) * scale_flow
        else:
            speed_final = 0.0
        
        speed_final = min(speed_final, 50.0)  # Clamp to 50 m/s
        
        # Compute velocity in world frame
        if avg_flow_px > 2.0 and self.vio_fe.last_matches is not None:
            pts_prev, pts_cur = self.vio_fe.last_matches
            if len(pts_prev) > 0:
                flows_normalized = pts_cur - pts_prev
                median_flow = np.median(flows_normalized, axis=0)
                flow_norm = np.linalg.norm(median_flow)
                if flow_norm > 1e-6:
                    flow_dir = median_flow / flow_norm
                    vel_cam = np.array([-flow_dir[0], -flow_dir[1], 0.0])
                    vel_cam = vel_cam / np.linalg.norm(vel_cam + 1e-9)
                    vel_body = R_cam_to_body @ vel_cam * speed_final
                else:
                    vel_body = t_body * speed_final
            else:
                vel_body = t_body * speed_final
        else:
            vel_body = t_body * speed_final
        
        vel_world = Rwb @ vel_body
        
        # Determine if using VZ only (for nadir cameras)
        use_only_vz = view_cfg.get('use_vz_only', True)
        
        # ESKF velocity update
        num_clones = (self.kf.x.shape[0] - 16) // 7
        err_dim = 15 + 6 * num_clones
        
        if use_only_vz:
            h_vel = np.zeros((1, err_dim), dtype=float)
            h_vel[0, 5] = 1.0
            vel_meas = np.array([[vel_world[2]]])
            r_mat = np.array([[(sigma_vo * view_cfg.get('sigma_scale_z', 2.0))**2]])
        else:
            h_vel = np.zeros((3, err_dim), dtype=float)
            h_vel[0, 3] = 1.0
            h_vel[1, 4] = 1.0
            h_vel[2, 5] = 1.0
            vel_meas = vel_world.reshape(-1, 1)
            scale_xy = view_cfg.get('sigma_scale_xy', 1.0)
            scale_z = view_cfg.get('sigma_scale_z', 2.0)
            r_mat = np.diag([(sigma_vo * scale_xy)**2, (sigma_vo * scale_xy)**2, (sigma_vo * scale_z)**2])
        
        def h_fun(x, h=h_vel):
            return h
        
        def hx_fun(x, h=h_vel):
            if use_only_vz:
                return x[5:6].reshape(1, 1)
            else:
                return x[3:6].reshape(3, 1)
        
        # Apply update
        if self.config.use_vio_velocity:
            self.kf.update(z=vel_meas, HJacobian=h_fun, Hx=hx_fun, R=r_mat)
            print(f"[VIO] Velocity update: speed={speed_final:.2f}m/s, vz_only={use_only_vz}")
    
    def log_error(self, t: float):
        """
        Log VIO error vs ground truth.
        
        Args:
            t: Current timestamp
        """
        from .vps_integration import xy_to_latlon, latlon_to_xy
        from .math_utils import quaternion_to_yaw
        from .state_manager import imu_to_gnss_position
        
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
            
            # VIO prediction
            vio_E = float(self.kf.x[0, 0])
            vio_N = float(self.kf.x[1, 0])
            vio_U = float(self.kf.x[2, 0])
            
            # Errors
            err_E = vio_E - gt_E
            err_N = vio_N - gt_N
            err_U = vio_U - gt_U
            pos_error = np.sqrt(err_E**2 + err_N**2 + err_U**2)
            
            # Yaw
            q_vio = self.kf.x[6:10, 0]
            yaw_vio = np.rad2deg(quaternion_to_yaw(q_vio))
            
            yaw_gt = np.nan
            yaw_error = np.nan
            if use_ppk and 'yaw' in gt_row:
                yaw_gt = 90.0 - np.rad2deg(gt_row['yaw'])
                yaw_error = ((yaw_vio - yaw_gt + 180) % 360) - 180
            
            with open(self.error_csv, "a") as ef:
                ef.write(
                    f"{t:.6f},{pos_error:.3f},{err_E:.3f},{err_N:.3f},{err_U:.3f},"
                    f"0.0,0.0,0.0,0.0,"
                    f"{err_U:.3f},{yaw_vio:.2f},{yaw_gt:.2f},{yaw_error:.2f},"
                    f"{gt_lat:.8f},{gt_lon:.8f},{gt_alt:.3f},"
                    f"{vio_E:.3f},{vio_N:.3f},{vio_U:.3f}\n"
                )
        except Exception:
            pass
    
    def process_magnetometer(self, t: float, time_elapsed: float):
        """
        Process magnetometer measurements up to current time.
        
        Args:
            t: Current timestamp
            time_elapsed: Time since start
        """
        from .measurement_updates import apply_magnetometer_update
        from .magnetometer import calibrate_magnetometer
        
        sigma_mag = self.global_config.get('SIGMA_MAG_YAW', 0.15)
        declination = self.global_config.get('MAG_DECLINATION', 0.0)
        use_raw = self.global_config.get('MAG_USE_RAW_HEADING', True)
        rate_limit = self.global_config.get('MAG_UPDATE_RATE_LIMIT', 1)
        convergence_window = self.global_config.get('MAG_INITIAL_CONVERGENCE_WINDOW', 30.0)
        
        while (self.state.mag_idx < len(self.mag_list) and 
               self.mag_list[self.state.mag_idx].t <= t):
            
            mag_rec = self.mag_list[self.state.mag_idx]
            self.state.mag_idx += 1
            
            # Rate limiting
            if (self.state.mag_idx - 1) % rate_limit != 0:
                continue
            
            # Calibrate
            mag_cal = calibrate_magnetometer(mag_rec.mag)
            
            in_convergence = time_elapsed < convergence_window
            has_ppk = self.ppk_state is not None
            
            applied, reason = apply_magnetometer_update(
                self.kf,
                mag_calibrated=mag_cal,
                mag_declination=declination,
                use_raw_heading=use_raw,
                sigma_mag_yaw=sigma_mag,
                time_elapsed=time_elapsed,
                in_convergence=in_convergence,
                has_ppk_yaw=has_ppk,
                timestamp=t
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
        from .measurement_updates import apply_dem_height_update
        from .vps_integration import xy_to_latlon
        
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
        
        applied, reason = apply_dem_height_update(
            self.kf,
            height_measurement=height_m,
            sigma_height=sigma_height,
            xy_uncertainty=xy_uncertainty,
            time_since_correction=time_since_correction,
            speed_ms=speed,
            has_valid_dem=True,
            no_vision_corrections=(len(self.imgs) == 0 and len(self.vps_list) == 0),
            timestamp=t
        )
    
    def log_pose(self, t: float, dt: float, used_vo: bool,
                 vo_data: Optional[Dict] = None):
        """
        Log current pose to CSV.
        
        Args:
            t: Current timestamp
            dt: Time delta
            used_vo: Whether VO was used this frame
            vo_data: VO output data (optional)
        """
        from .vps_integration import xy_to_latlon
        
        lat_now, lon_now = xy_to_latlon(
            self.kf.x[0, 0], self.kf.x[1, 0],
            self.lat0, self.lon0
        )
        
        # Compute MSL/AGL
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
        
        with open(self.pose_csv, "a") as f:
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
        from .msckf import print_msckf_stats
        
        print("\n\n--- Done ---")
        print(f"Total IMU samples: {len(self.imu)}")
        print(f"Images used: {max(0, self.state.vio_frame + 1)}")
        print(f"VPS used: {self.state.vps_idx}")
        print(f"Magnetometer: {self.state.mag_updates} updates | "
              f"{self.state.mag_rejects} rejected")
        print(f"ZUPT: {self.state.zupt_applied} applied | "
              f"{self.state.zupt_rejected} rejected | "
              f"{self.state.zupt_detected} detected")
        
        print_msckf_stats()
        
        # Print error statistics
        try:
            error_df = pd.read_csv(self.error_csv)
            if len(error_df) > 0:
                print("\n=== Error Statistics ===")
                print(f"Position Error:")
                print(f"  Mean: {error_df['pos_error_m'].mean():.2f} m")
                print(f"  Max: {error_df['pos_error_m'].max():.2f} m")
                print(f"  Final: {error_df['pos_error_m'].iloc[-1]:.2f} m")
        except Exception:
            pass
    
    def run(self):
        """
        Run the complete VIO pipeline.
        
        This is the main entry point that orchestrates:
        1. Configuration loading
        2. Data loading
        3. EKF initialization
        4. IMU-driven loop with measurement updates
        5. Output generation
        """
        print("=" * 80)
        print("VIO+EKF Pipeline Starting")
        print("=" * 80)
        
        # Load configuration
        if self.config.config_yaml:
            self.load_config()
        
        # Load data
        self.load_data()
        
        # Initialize EKF
        self.initialize_ekf()
        
        # Initialize VIO frontend
        self.initialize_vio_frontend()
        
        # Setup output files
        self.setup_output_files()
        
        # Initialize timing
        self.state.t0 = self.imu[0].t
        self.state.last_t = self.state.t0
        
        # Initialize preintegration if enabled
        ongoing_preint = None
        if self.config.use_preintegration:
            from .imu_preintegration import IMUPreintegration
            
            imu_params = self.global_config.get('IMU_PARAMS', {})
            ongoing_preint = IMUPreintegration(
                bg=self.kf.x[10:13, 0].reshape(3,),
                ba=self.kf.x[13:16, 0].reshape(3,),
                sigma_g=imu_params.get('gyr_n', 0.01),
                sigma_a=imu_params.get('acc_n', 0.1),
                sigma_bg=imu_params.get('gyr_w', 0.0001),
                sigma_ba=imu_params.get('acc_w', 0.001)
            )
            print(f"[PREINT] Initialized preintegration buffer")
        
        # Main loop
        print("\n=== Running (IMU-driven) ===")
        tic_all = time.time()
        
        for i, rec in enumerate(self.imu):
            t = rec.t
            dt = max(0.0, float(t - self.state.last_t)) if i > 0 else 0.0
            self.state.last_t = t
            
            time_elapsed = t - self.state.t0
            self.state.current_phase = self.get_flight_phase(time_elapsed)
            
            # IMU propagation
            if self.config.use_preintegration and ongoing_preint is not None:
                ongoing_preint.integrate_measurement(rec.ang, rec.lin, dt)
            else:
                self.process_imu_sample(rec, dt, time_elapsed)
            
            # VPS updates
            self.process_vps(t)
            
            # Magnetometer updates
            if self.config.use_magnetometer:
                self.process_magnetometer(t, time_elapsed)
            
            # VIO updates (feature tracking, MSCKF, velocity)
            used_vo, vo_data = self.process_vio(rec, t, time_elapsed, ongoing_preint)
            
            # DEM height updates
            self.process_dem_height(t)
            
            # Log error vs ground truth
            if i % 100 == 0:  # Log every 100 samples to reduce file size
                self.log_error(t)
            
            # Log pose
            self.log_pose(t, dt, used_vo, vo_data)
            
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
