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
from .config import load_config, CAMERA_VIEW_CONFIGS
from .config import BODY_T_CAMDOWN, BODY_T_CAMFRONT, BODY_T_CAMSIDE
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
    augment_state_with_camera, detect_stationary, apply_zupt
)
from .vps_integration import apply_vps_update, xy_to_latlon, latlon_to_xy
from .vps_integration import compute_plane_constraint_jacobian
from .msckf import perform_msckf_updates, print_msckf_stats
from .measurement_updates import apply_magnetometer_update, apply_dem_height_update
from .magnetometer import calibrate_magnetometer
from .math_utils import quaternion_to_yaw, skew_symmetric
from .loop_closure import get_loop_detector, init_loop_closure, LoopClosureDetector
from .imu_preintegration import IMUPreintegration
from .fisheye_rectifier import FisheyeRectifier, create_rectifier_from_config
from .propagation import VibrationDetector
from .output_utils import save_calibration_log, save_keyframe_image_with_overlay


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
        
        # Fisheye rectifier (optional)
        self.rectifier: Optional[FisheyeRectifier] = None
        
        # Loop closure detector (optional)
        self.loop_detector: Optional[LoopClosureDetector] = None
        
        # Vibration detector (optional)
        self.vibration_detector: Optional[VibrationDetector] = None
    
    def load_config(self) -> dict:
        """Load YAML configuration file."""
        if self.config.config_yaml:
            cfg = load_config(self.config.config_yaml)
            self.global_config = cfg
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
        with open(self.pose_csv, "w", newline="") as f:
            f.write("Timestamp(s),dt,Frame,PX,PY,PZ_MSL,VX,VY,VZ,lat,lon,AGL(m),"
                    "vo_dx,vo_dy,vo_dz,vo_d_roll,vo_d_pitch,vo_d_yaw\n")
        
        # Error log - comparison with ground truth
        self.error_csv = os.path.join(self.config.output_dir, "error_log.csv")
        with open(self.error_csv, "w", newline="") as f:
            f.write("t,pos_error_m,pos_error_E,pos_error_N,pos_error_U,"
                    "vel_error_m_s,vel_error_E,vel_error_N,vel_error_U,"
                    "alt_error_m,yaw_vio_deg,yaw_gps_deg,yaw_error_deg,"
                    "gps_lat,gps_lon,gps_alt,vio_E,vio_N,vio_U\n")
        
        # State debug log - detailed state evolution
        self.state_dbg_csv = os.path.join(self.config.output_dir, "state_debug.csv")
        with open(self.state_dbg_csv, "w", newline="") as f:
            f.write("t,px,py,pz,vx,vy,vz,a_world_x,a_world_y,a_world_z,dem,agl,msl\n")
        
        # Inference timing log
        self.inf_csv = os.path.join(self.config.output_dir, "inference_log.csv")
        with open(self.inf_csv, "w", newline="") as f:
            f.write("Index,Inference Time (s),FPS\n")
        
        # VO debug log
        self.vo_dbg_csv = os.path.join(self.config.output_dir, "vo_debug.csv")
        with open(self.vo_dbg_csv, "w", newline="") as f:
            f.write("Frame,num_inliers,rot_angle_deg,alignment_deg,rotation_rate_deg_s,"
                    "use_only_vz,skip_vo,vo_dx,vo_dy,vo_dz,vel_vx,vel_vy,vel_vz\n")
        
        # MSCKF debug log
        self.msckf_dbg_csv = os.path.join(self.config.output_dir, "msckf_debug.csv")
        with open(self.msckf_dbg_csv, "w", newline="") as f:
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
            with open(self.imu_raw_csv, "w", newline="") as f:
                f.write("t,ori_x,ori_y,ori_z,ori_w,ang_x,ang_y,ang_z,lin_x,lin_y,lin_z\n")
            
            # State & covariance evolution
            self.state_cov_csv = os.path.join(self.config.output_dir, "debug_state_covariance.csv")
            with open(self.state_cov_csv, "w", newline="") as f:
                f.write("t,frame,P_pos_xx,P_pos_yy,P_pos_zz,P_vel_xx,P_vel_yy,P_vel_zz,"
                        "P_rot_xx,P_rot_yy,P_rot_zz,P_bg_xx,P_bg_yy,P_bg_zz,"
                        "P_ba_xx,P_ba_yy,P_ba_zz,bg_x,bg_y,bg_z,ba_x,ba_y,ba_z\n")
            
            # Residual & innovation log
            self.residual_csv = os.path.join(self.config.output_dir, "debug_residuals.csv")
            with open(self.residual_csv, "w", newline="") as f:
                f.write("t,frame,update_type,innovation_x,innovation_y,innovation_z,"
                        "mahalanobis_dist,chi2_threshold,accepted,NIS,NEES\n")
            
            # Feature tracking statistics
            self.feature_stats_csv = os.path.join(self.config.output_dir, "debug_feature_stats.csv")
            with open(self.feature_stats_csv, "w", newline="") as f:
                f.write("frame,t,num_features_detected,num_features_tracked,num_inliers,"
                        "mean_parallax_px,max_parallax_px,tracking_ratio,inlier_ratio\n")
            
            # MSCKF window & marginalization log
            self.msckf_window_csv = os.path.join(self.config.output_dir, "debug_msckf_window.csv")
            with open(self.msckf_window_csv, "w", newline="") as f:
                f.write("frame,t,num_camera_clones,num_tracked_features,num_mature_features,"
                        "window_start_time,window_duration,marginalized_clone_id\n")
            
            # FEJ consistency log
            self.fej_csv = os.path.join(self.config.output_dir, "debug_fej_consistency.csv")
            with open(self.fej_csv, "w", newline="") as f:
                f.write("timestamp,frame,clone_idx,pos_fej_drift_m,rot_fej_drift_deg,"
                        "bg_fej_drift_rad_s,ba_fej_drift_m_s2\n")
            
            print(f"[DEBUG] Debug data logging enabled")
        
        if self.config.save_keyframe_images:
            self.keyframe_dir = os.path.join(self.config.output_dir, "debug_keyframes")
            os.makedirs(self.keyframe_dir, exist_ok=True)
            print(f"[DEBUG] Keyframe images will be saved to: {self.keyframe_dir}")
        
        # Save calibration snapshot using output_utils
        if self.config.save_debug_data:
            self._save_calibration_snapshot()
    
    def _save_calibration_snapshot(self):
        """Save calibration parameters for reproducibility using output_utils."""
        cal_path = os.path.join(self.config.output_dir, "debug_calibration.txt")
        
        # Gather all parameters for save_calibration_log
        kb_params = self.global_config.get('KB_PARAMS', {})
        imu_params = self.global_config.get('IMU_PARAMS', {})
        
        # Camera view config
        view_cfg = CAMERA_VIEW_CONFIGS.get(self.config.camera_view, {})
        
        # Magnetometer params
        mag_params = {
            'hard_iron': str(self.global_config.get('MAG_HARD_IRON_OFFSET', np.zeros(3))),
            'soft_iron': 'calibrated',
            'declination': self.global_config.get('MAG_DECLINATION', 0.0),
        }
        
        # Process noise params
        noise_params = {
            'sigma_accel': self.global_config.get('SIGMA_ACCEL', 0.8),
            'sigma_vo_vel': self.global_config.get('SIGMA_VO_VEL', 2.0),
            'sigma_vps_xy': self.global_config.get('SIGMA_VPS_XY', 1.0),
            'sigma_agl_z': self.global_config.get('SIGMA_AGL_Z', 2.5),
            'sigma_mag_yaw': self.global_config.get('SIGMA_MAG_YAW', 0.15),
        }
        
        # VIO quality control params  
        vio_params = {
            'min_parallax_px': self.global_config.get('MIN_PARALLAX_PX', 2.0),
            'use_preintegration': self.config.use_preintegration,
            'use_magnetometer': self.config.use_magnetometer,
            'use_vio_velocity': self.config.use_vio_velocity,
            'use_rectifier': self.rectifier is not None,
            'use_loop_closure': self.loop_detector is not None,
        }
        
        # Initial state
        initial_state = {
            'origin_lat': f"{self.lat0:.8f}°",
            'origin_lon': f"{self.lon0:.8f}°",
            'msl_altitude': f"{self.msl0:.2f} m",
            'dem_elevation': f"{self.dem0:.2f} m" if self.dem0 else "N/A",
            'initial_velocity': str(self.v_init) + " m/s",
            'z_state': self.config.z_state,
            'downscale_size': f"{self.config.downscale_size[0]}x{self.config.downscale_size[1]}",
        }
        
        # Call the modular save function
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
            estimate_imu_bias=self.config.estimate_imu_bias
        )
        
        print(f"[DEBUG] Calibration snapshot saved: {cal_path}")
    
    def _log_debug_state_covariance(self, t: float, rec, i: int):
        """Log state and covariance to debug CSV (every 10 samples)."""
        if not self.config.save_debug_data or not hasattr(self, 'state_cov_csv'):
            return
        if i % 10 != 0:  # Every 10 samples (~25ms @ 400Hz)
            return
        
        try:
            bg = self.kf.x[10:13, 0]
            ba = self.kf.x[13:16, 0]
            
            # Extract diagonal elements of covariance blocks
            P_pos = [self.kf.P[0,0], self.kf.P[1,1], self.kf.P[2,2]]
            P_vel = [self.kf.P[3,3], self.kf.P[4,4], self.kf.P[5,5]]
            P_rot = [self.kf.P[6,6], self.kf.P[7,7], self.kf.P[8,8]]
            P_bg = [self.kf.P[9,9], self.kf.P[10,10], self.kf.P[11,11]]
            P_ba = [self.kf.P[12,12], self.kf.P[13,13], self.kf.P[14,14]]
            
            with open(self.state_cov_csv, "a", newline="") as f:
                f.write(f"{t:.6f},{self.state.vio_frame},{P_pos[0]:.6e},{P_pos[1]:.6e},{P_pos[2]:.6e},"
                       f"{P_vel[0]:.6e},{P_vel[1]:.6e},{P_vel[2]:.6e},"
                       f"{P_rot[0]:.6e},{P_rot[1]:.6e},{P_rot[2]:.6e},"
                       f"{P_bg[0]:.6e},{P_bg[1]:.6e},{P_bg[2]:.6e},"
                       f"{P_ba[0]:.6e},{P_ba[1]:.6e},{P_ba[2]:.6e},"
                       f"{bg[0]:.6f},{bg[1]:.6f},{bg[2]:.6f},"
                       f"{ba[0]:.6f},{ba[1]:.6f},{ba[2]:.6f}\n")
        except Exception:
            pass
    
    def _log_debug_imu_raw(self, t: float, rec):
        """Log raw IMU data to debug CSV."""
        if not self.config.save_debug_data or not hasattr(self, 'imu_raw_csv'):
            return
        
        try:
            with open(self.imu_raw_csv, "a", newline="") as f:
                f.write(f"{t:.6f},{rec.q[0]:.6f},{rec.q[1]:.6f},{rec.q[2]:.6f},{rec.q[3]:.6f},"
                       f"{rec.ang[0]:.6f},{rec.ang[1]:.6f},{rec.ang[2]:.6f},"
                       f"{rec.lin[0]:.6f},{rec.lin[1]:.6f},{rec.lin[2]:.6f}\n")
        except Exception:
            pass
    
    def _log_debug_feature_stats(self, frame: int, t: float, num_features: int, 
                                  num_tracked: int, num_inliers: int,
                                  mean_parallax: float, max_parallax: float):
        """Log feature tracking statistics to debug CSV."""
        if not self.config.save_debug_data or not hasattr(self, 'feature_stats_csv'):
            return
        
        try:
            tracking_ratio = 1.0 if num_features > 0 else 0.0
            inlier_ratio = num_inliers / max(1, num_features)
            
            with open(self.feature_stats_csv, "a", newline="") as f:
                f.write(f"{frame},{t:.6f},{num_features},{num_tracked},{num_inliers},"
                       f"{mean_parallax:.2f},{max_parallax:.2f},"
                       f"{tracking_ratio:.3f},{inlier_ratio:.3f}\n")
        except Exception:
            pass
    
    def _log_debug_msckf_window(self, frame: int, t: float, num_clones: int,
                                 num_tracked: int, num_mature: int,
                                 window_start: float, marginalized_clone: int):
        """Log MSCKF window state to debug CSV."""
        if not self.config.save_debug_data or not hasattr(self, 'msckf_window_csv'):
            return
        
        try:
            window_duration = t - window_start if window_start > 0 else 0.0
            
            with open(self.msckf_window_csv, "a", newline="") as f:
                f.write(f"{frame},{t:.6f},{num_clones},{num_tracked},{num_mature},"
                       f"{window_start:.6f},{window_duration:.3f},{marginalized_clone}\n")
        except Exception:
            pass

    def _log_state_debug(self, t: float, dem_now: float, agl_now: float, 
                         msl_now: float, a_world: np.ndarray):
        """
        Log full state debug (every sample, like vio_vps.py state_dbg_csv).
        
        Args:
            t: Current timestamp
            dem_now: Current DEM height
            agl_now: Current AGL
            msl_now: Current MSL
            a_world: World-frame acceleration [3]
        """
        try:
            px = float(self.kf.x[0, 0])
            py = float(self.kf.x[1, 0])
            pz = float(self.kf.x[2, 0])
            vx = float(self.kf.x[3, 0])
            vy = float(self.kf.x[4, 0])
            vz = float(self.kf.x[5, 0])
        except Exception:
            px = py = pz = vx = vy = vz = float('nan')
        
        try:
            a_wx = float(a_world[0])
            a_wy = float(a_world[1])
            a_wz = float(a_world[2])
        except Exception:
            a_wx = a_wy = a_wz = float('nan')
        
        dem_val = dem_now if dem_now is not None else float('nan')
        agl_val = agl_now if agl_now is not None else float('nan')
        msl_val = msl_now if msl_now is not None else float('nan')
        
        try:
            with open(self.state_dbg_csv, "a", newline="") as f:
                f.write(f"{t:.6f},{px:.6f},{py:.6f},{pz:.6f},"
                        f"{vx:.6f},{vy:.6f},{vz:.6f},"
                        f"{a_wx:.6f},{a_wy:.6f},{a_wz:.6f},"
                        f"{dem_val:.6f},{agl_val:.6f},{msl_val:.6f}\n")
        except Exception:
            pass

    def _log_vo_debug(self, frame: int, num_inliers: int, rot_angle_deg: float,
                      alignment_deg: float, rotation_rate_deg_s: float,
                      use_only_vz: bool, skip_vo: bool,
                      vo_dx: float, vo_dy: float, vo_dz: float,
                      vel_vx: float, vel_vy: float, vel_vz: float):
        """
        Log VO debug info (when VIO processes a frame, like vio_vps.py vo_dbg).
        
        Args:
            frame: Frame index
            num_inliers: Number of inlier matches
            rot_angle_deg: Rotation angle from VO (degrees)
            alignment_deg: Alignment with expected motion (degrees)
            rotation_rate_deg_s: Rotation rate (degrees/sec)
            use_only_vz: Whether only vertical velocity is used
            skip_vo: Whether VO was skipped
            vo_dx, vo_dy, vo_dz: VO translation
            vel_vx, vel_vy, vel_vz: VIO velocities
        """
        try:
            with open(self.vo_dbg_csv, "a", newline="") as f:
                f.write(
                    f"{max(frame, 0)},{num_inliers},{rot_angle_deg:.3f},"
                    f"{alignment_deg:.3f},{rotation_rate_deg_s:.3f},"
                    f"{int(use_only_vz)},{int(skip_vo)},"
                    f"{vo_dx:.6f},{vo_dy:.6f},{vo_dz:.6f},"
                    f"{vel_vx:.3f},{vel_vy:.3f},{vel_vz:.3f}\n"
                )
        except Exception:
            pass

    def _log_fej_consistency(self, t: float, frame: int):
        """
        Log FEJ consistency metrics (like vio_vps.py log_fej_consistency).
        
        Compares FEJ linearization points vs. current state to detect
        spurious information from unobservable directions.
        
        Args:
            t: Current timestamp
            frame: Current VIO frame index
        """
        if not self.state.cam_states:
            return
        
        if not hasattr(self, 'fej_csv') or not self.fej_csv:
            return
        
        # Current bias estimates from core state
        bg_current = self.kf.x[10:13, 0]
        ba_current = self.kf.x[13:16, 0]
        
        try:
            with open(self.fej_csv, "a", newline="") as f:
                for i, cs in enumerate(self.state.cam_states):
                    # Skip if no FEJ data
                    if 'q_fej' not in cs or 'p_fej' not in cs:
                        continue
                    
                    q_fej = cs['q_fej']
                    p_fej = cs['p_fej']
                    bg_fej = cs.get('bg_fej', bg_current)
                    ba_fej = cs.get('ba_fej', ba_current)
                    
                    # Get current state estimates (nominal state)
                    q_idx = cs['q_idx']
                    p_idx = cs['p_idx']
                    q_current = self.kf.x[q_idx:q_idx+4, 0]
                    p_current = self.kf.x[p_idx:p_idx+3, 0]
                    
                    # Compute position drift (Euclidean distance)
                    pos_drift = np.linalg.norm(p_current - p_fej)
                    
                    # Compute rotation drift (angle in degrees)
                    try:
                        # Quaternion difference: q_diff = q_current * q_fej^-1
                        r_current = R_scipy.from_quat([q_current[1], q_current[2], 
                                                       q_current[3], q_current[0]])
                        r_fej = R_scipy.from_quat([q_fej[1], q_fej[2], q_fej[3], q_fej[0]])
                        r_diff = r_current * r_fej.inv()
                        rot_drift = np.linalg.norm(r_diff.as_rotvec()) * 180.0 / np.pi
                    except Exception:
                        rot_drift = 0.0
                    
                    # Bias drifts
                    bg_drift = np.linalg.norm(bg_current - bg_fej)
                    ba_drift = np.linalg.norm(ba_current - ba_fej)
                    
                    f.write(f"{t:.6f},{frame},{i},{pos_drift:.6f},{rot_drift:.6f},"
                            f"{bg_drift:.9f},{ba_drift:.9f}\n")
        except Exception:
            pass
    
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
            process_imu(
                self.kf, rec, dt,
                estimate_imu_bias=self.config.estimate_imu_bias,
                t=rec.t, t0=self.state.t0,
                imu_params=imu_params
            )
        
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
            
            applied, _ = apply_zupt(
                self.kf,
                v_mag=v_mag,
                consecutive_stationary_count=self.state.consecutive_stationary
            )
            
            if applied:
                self.state.zupt_applied += 1
                self.state.consecutive_stationary += 1
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
        sigma_vps = self.global_config.get('SIGMA_VPS_XY', 1.0)
        
        while (self.state.vps_idx < len(self.vps_list) and 
               self.vps_list[self.state.vps_idx].t <= t):
            
            vps = self.vps_list[self.state.vps_idx]
            self.state.vps_idx += 1
            
            # Convert VPS lat/lon to local XY
            vps_x, vps_y = latlon_to_xy(vps.lat, vps.lon, self.lat0, self.lon0)
            vps_xy = np.array([vps_x, vps_y])
            
            applied = apply_vps_update(
                self.kf,
                vps_xy=vps_xy,
                sigma_vps=sigma_vps
            )
            
            if applied:
                print(f"[VPS] Applied at t={t:.3f}")
    
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
            
            # Apply fisheye rectification if enabled
            img_for_tracking = img
            if self.rectifier is not None:
                img_for_tracking = self.rectifier.rectify(img)
            
            if is_fast_rotation:
                print(f"[VIO] SKIPPING due to fast rotation: {rotation_rate_deg_s:.1f} deg/s")
                self.state.img_idx += 1
                continue
            
            # Run VIO frontend
            ok, ninl, r_vo_mat, t_unit, dt_img = self.vio_fe.step(img_for_tracking, self.imgs[self.state.img_idx].t)
            
            # Loop closure detection - check when we have sufficient position estimate
            self._check_loop_closure(img, t)
            
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
            self._log_debug_feature_stats(
                frame=self.vio_fe.frame_idx,
                t=t,
                num_features=num_features,
                num_tracked=num_features,
                num_inliers=num_inliers,
                mean_parallax=avg_flow_px,
                max_parallax=avg_flow_px
            )
            
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
                
                # Log VO debug (like vio_vps.py)
                rot_angle_deg = np.degrees(np.arccos(np.clip((np.trace(r_vo_mat)-1)/2, -1, 1)))
                vel_vx = float(self.kf.x[3, 0])
                vel_vy = float(self.kf.x[4, 0])
                vel_vz = float(self.kf.x[5, 0])
                self._log_vo_debug(
                    frame=self.vio_fe.frame_idx,
                    num_inliers=num_inliers,
                    rot_angle_deg=rot_angle_deg,
                    alignment_deg=0.0,  # TODO: compute alignment with expected motion
                    rotation_rate_deg_s=rotation_rate_deg_s,
                    use_only_vz=False,  # Not implemented in modular version
                    skip_vo=False,
                    vo_dx=vo_dx, vo_dy=vo_dy, vo_dz=vo_dz,
                    vel_vx=vel_vx, vel_vy=vel_vy, vel_vz=vel_vz
                )
                
                # Save keyframe image with visualization overlay
                if self.config.save_keyframe_images and hasattr(self, 'keyframe_dir'):
                    self._save_keyframe_with_overlay(img, self.vio_fe.frame_idx)
            
            self.state.img_idx += 1
        
        return used_vo, vo_data
    
    def _save_keyframe_with_overlay(self, img_gray: np.ndarray, frame_id: int):
        """
        Save keyframe image with feature tracking overlay.
        
        Uses save_keyframe_image_with_overlay from output_utils for consistent
        visualization across the codebase.
        
        Args:
            img_gray: Grayscale image
            frame_id: Frame index
        """
        try:
            # Get current tracked features from frontend
            features = None
            inliers = None
            reprojections = None
            
            if self.vio_fe is not None:
                # Get current features
                if hasattr(self.vio_fe, 'prev_pts') and self.vio_fe.prev_pts is not None:
                    features = self.vio_fe.prev_pts.copy()
                
                # Get tracking stats
                tracking_stats = {
                    'num_tracked': self.vio_fe.last_num_tracked,
                    'num_inliers': self.vio_fe.last_num_inliers,
                    'mean_parallax': self.vio_fe.mean_parallax,
                }
            else:
                tracking_stats = None
            
            output_path = os.path.join(self.keyframe_dir, f"keyframe_{frame_id:06d}.jpg")
            
            save_keyframe_image_with_overlay(
                image=img_gray,
                features=features,
                inliers=inliers,
                reprojections=reprojections,
                output_path=output_path,
                frame_id=frame_id,
                tracking_stats=tracking_stats
            )
            
        except Exception as e:
            print(f"[KEYFRAME] Failed to save keyframe {frame_id}: {e}")
    
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
            
            # Debug: log MSCKF window state
            num_tracked = len(obs_data)
            window_start = self.state.cam_states[0]['t'] if self.state.cam_states else t
            self._log_debug_msckf_window(
                frame=self.vio_fe.frame_idx,
                t=t,
                num_clones=len(self.state.cam_states),
                num_tracked=num_tracked,
                num_mature=0,  # Will be computed in trigger
                window_start=window_start,
                marginalized_clone=-1
            )
            
            # Trigger MSCKF update if enough clones
            if len(self.state.cam_states) >= 3:
                self._trigger_msckf_update(t)
                
        except Exception as e:
            print(f"[CLONE] Failed: {e}")
    
    def _check_loop_closure(self, img_gray: np.ndarray, t: float):
        """
        Check for loop closure and apply yaw correction if detected.
        
        This method:
        1. Gets current position and yaw from EKF state
        2. Checks if we should add current frame as keyframe
        3. Searches for loop closure candidates
        4. If match found, applies yaw correction to EKF
        
        Args:
            img_gray: Current grayscale image
            t: Current timestamp
        """
        if self.loop_detector is None:
            return
        
        try:
            # Get current state
            position = self.kf.x[0:3, 0].flatten()[:2]  # XY position only
            yaw = quaternion_to_yaw(self.kf.x[6:10, 0].flatten())
            frame_idx = self.vio_fe.frame_idx if self.vio_fe else 0
            
            # Check if we should add keyframe
            if self.loop_detector.should_add_keyframe(position, yaw, frame_idx):
                self.loop_detector.add_keyframe(frame_idx, position, yaw, img_gray)
            
            # Try to find and match loop closure
            candidates = self.loop_detector.find_loop_candidates(position, frame_idx)
            
            if len(candidates) > 0:
                # Get camera intrinsics for matching
                kb_params = self.global_config.get('KB_PARAMS', {})
                K = np.array([
                    [kb_params.get('mu', 600), 0, img_gray.shape[1] / 2],
                    [0, kb_params.get('mv', 600), img_gray.shape[0] / 2],
                    [0, 0, 1]
                ], dtype=np.float64)
                
                # Try to match with each candidate
                for kf_idx in candidates:
                    result = self.loop_detector.match_keyframe(img_gray, kf_idx, K)
                    
                    if result is not None:
                        relative_yaw, num_inliers = result
                        
                        # Apply yaw correction via EKF update
                        self._apply_loop_closure_correction(relative_yaw, kf_idx, num_inliers, t)
                        break  # Only apply one correction per frame
                        
        except Exception as e:
            print(f"[LOOP] Error in loop closure check: {e}")
    
    def _apply_loop_closure_correction(self, relative_yaw: float, kf_idx: int, 
                                        num_inliers: int, t: float):
        """
        Apply yaw correction from loop closure detection.
        
        Args:
            relative_yaw: Measured yaw difference from loop closure
            kf_idx: Index of matched keyframe
            num_inliers: Number of feature match inliers
            t: Current timestamp
        """
        if self.loop_detector is None:
            return
        
        try:
            # Get expected yaw difference from stored keyframe
            kf = self.loop_detector.keyframes[kf_idx]
            current_yaw = quaternion_to_yaw(self.kf.x[6:10, 0].flatten())
            expected_yaw_diff = current_yaw - kf['yaw']
            
            # Yaw correction (innovation)
            yaw_error = relative_yaw - expected_yaw_diff
            
            # Wrap to [-π, π]
            while yaw_error > np.pi:
                yaw_error -= 2 * np.pi
            while yaw_error < -np.pi:
                yaw_error += 2 * np.pi
            
            # Only apply if correction is significant but not too large
            if np.abs(yaw_error) < np.radians(2.0):  # < 2° skip
                return
            if np.abs(yaw_error) > np.radians(30.0):  # > 30° suspicious
                print(f"[LOOP] REJECT: yaw_error={np.degrees(yaw_error):.1f}° too large")
                return
            
            # Build EKF update for yaw
            num_clones = len(self.state.cam_states)
            err_dim = 15 + 6 * num_clones
            
            H_loop = np.zeros((1, err_dim), dtype=float)
            H_loop[0, 8] = 1.0  # Yaw error index
            
            z_loop = np.array([[yaw_error]])
            
            # Measurement noise (inversely proportional to inliers)
            base_sigma = np.radians(5.0)  # 5° base uncertainty
            sigma_loop = base_sigma / np.sqrt(max(num_inliers, 1) / 15.0)
            R_loop = np.array([[sigma_loop**2]])
            
            # Apply EKF update
            def h_loop_jacobian(x, h=H_loop):
                return h
            
            def hx_loop_fun(x, h=H_loop):
                return np.zeros((1, 1))  # Zero residual (error is already computed)
            
            self.kf.update(
                z=z_loop,
                HJacobian=h_loop_jacobian,
                Hx=hx_loop_fun,
                R=R_loop
            )
            
            print(f"[LOOP] CORRECTION at t={t:.2f}s: Δyaw={np.degrees(yaw_error):.2f}° "
                  f"(kf={kf_idx}, inliers={num_inliers})")
            
        except Exception as e:
            print(f"[LOOP] Failed to apply correction: {e}")
    
    def _trigger_msckf_update(self, t: float = 0.0):
        """Trigger MSCKF multi-view geometric update.
        
        Args:
            t: Current timestamp for logging
        """
        
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
                # Pass msckf_dbg_csv path if debug data enabled
                msckf_dbg_path = self.msckf_dbg_csv if hasattr(self, 'msckf_dbg_csv') else None
                
                num_updates = perform_msckf_updates(
                    self.vio_fe, 
                    self.state.cam_observations,
                    self.state.cam_states, 
                    self.kf,
                    min_observations=2,
                    max_features=50,
                    msckf_dbg_path=msckf_dbg_path,
                    dem_reader=self.dem,
                    origin_lat=self.lat0,
                    origin_lon=self.lon0
                )
                if num_updates > 0:
                    print(f"[MSCKF] Updated {num_updates} features")
                    
                    # Log FEJ consistency after MSCKF update
                    if self.config.save_debug_data and hasattr(self, 'fej_csv') and self.fej_csv:
                        self._log_fej_consistency(t, self.vio_fe.frame_idx)
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
        kb_params = self.global_config.get('KB_PARAMS', {'mu': 600})
        sigma_vo = self.global_config.get('SIGMA_VO_VEL', 0.5)
        
        # Get camera extrinsics
        view_cfg = CAMERA_VIEW_CONFIGS.get(self.config.camera_view, CAMERA_VIEW_CONFIGS['nadir'])
        extrinsics_name = view_cfg['extrinsics']
        
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
            # Compute innovation for logging
            predicted_vel = hx_fun(self.kf.x)
            innovation = vel_meas - predicted_vel
            s_mat = h_vel @ self.kf.P @ h_vel.T + r_mat
            try:
                m2 = innovation.T @ np.linalg.inv(s_mat) @ innovation
                mahal_dist = np.sqrt(float(m2))
            except Exception:
                mahal_dist = float('nan')
            
            self.kf.update(z=vel_meas, HJacobian=h_fun, Hx=hx_fun, R=r_mat)
            print(f"[VIO] Velocity update: speed={speed_final:.2f}m/s, vz_only={use_only_vz}")
            
            # Log to debug_residuals.csv
            if self.config.save_debug_data and hasattr(self, 'residual_csv') and self.residual_csv:
                from .output_utils import log_measurement_update
                log_measurement_update(
                    self.residual_csv, t, self.state.vio_frame, 'VIO_VEL',
                    innovation=innovation.flatten(),
                    mahalanobis_dist=mahal_dist,
                    chi2_threshold=9.21,
                    accepted=True,
                    s_matrix=s_mat,
                    p_prior=getattr(self.kf, 'P_prior', self.kf.P)
                )
    
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
            
            with open(self.error_csv, "a", newline="") as ef:
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
            
            # Calibrate using hard-iron and soft-iron from config
            hard_iron = self.global_config.get('MAG_HARD_IRON_OFFSET', None)
            soft_iron = self.global_config.get('MAG_SOFT_IRON_MATRIX', None)
            mag_cal = calibrate_magnetometer(mag_rec.mag, hard_iron=hard_iron, soft_iron=soft_iron)
            
            in_convergence = time_elapsed < convergence_window
            has_ppk = self.ppk_state is not None
            
            # Get residual_csv path if debug data is enabled
            residual_path = self.residual_csv if self.config.save_debug_data and hasattr(self, 'residual_csv') else None
            
            applied, reason = apply_magnetometer_update(
                self.kf,
                mag_calibrated=mag_cal,
                mag_declination=declination,
                use_raw_heading=use_raw,
                sigma_mag_yaw=sigma_mag,
                time_elapsed=time_elapsed,
                in_convergence=in_convergence,
                has_ppk_yaw=has_ppk,
                timestamp=t,
                residual_csv=residual_path,
                frame=self.state.vio_frame
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
        
        # Initialize fisheye rectifier (optional - for converting fisheye to pinhole)
        self._initialize_rectifier()
        
        # Initialize loop closure detector (for yaw drift correction)
        self._initialize_loop_closure()
        
        # Initialize vibration detector
        imu_params = self.global_config.get('IMU_PARAMS', {})
        vib_buffer_size = self.global_config.get('VIBRATION_WINDOW_SIZE', 50)
        vib_threshold_mult = self.global_config.get('VIBRATION_THRESHOLD_MULT', 5.0)
        self.vibration_detector = VibrationDetector(
            buffer_size=vib_buffer_size,
            threshold=imu_params.get('acc_n', 0.1) * vib_threshold_mult
        )
        
        # Setup output files
        self.setup_output_files()
        
        # Initialize timing
        self.state.t0 = self.imu[0].t
        self.state.last_t = self.state.t0
        
        # Initialize preintegration if enabled
        ongoing_preint = None
        if self.config.use_preintegration:
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
        
        # Store last a_world for state_debug logging
        last_a_world = np.array([0.0, 0.0, 0.0])
        
        for i, rec in enumerate(self.imu):
            # Start timing for inference_log
            tic_iter = time.time()
            
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
            
            # Debug logging: raw IMU and state covariance
            self._log_debug_imu_raw(t, rec)
            self._log_debug_state_covariance(t, rec, i)
            
            # VPS updates
            self.process_vps(t)
            
            # Magnetometer updates
            if self.config.use_magnetometer:
                self.process_magnetometer(t, time_elapsed)
            
            # VIO updates (feature tracking, MSCKF, velocity)
            used_vo, vo_data = self.process_vio(rec, t, time_elapsed, ongoing_preint)
            
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
            self._log_state_debug(t, dem_now, agl_now, msl_now, last_a_world)
            
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
