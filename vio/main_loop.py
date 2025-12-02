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
        """Create output directory and CSV files."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Pose output
        self.pose_csv = os.path.join(self.config.output_dir, "pose.csv")
        with open(self.pose_csv, "w") as f:
            f.write("Timestamp(s),dt,Frame,PX,PY,PZ_MSL,VX,VY,VZ,lat,lon,AGL(m),"
                    "vo_dx,vo_dy,vo_dz,vo_d_roll,vo_d_pitch,vo_d_yaw\n")
        
        # Error log
        self.error_csv = os.path.join(self.config.output_dir, "error_log.csv")
        with open(self.error_csv, "w") as f:
            f.write("t,pos_error_m,pos_error_E,pos_error_N,pos_error_U,"
                    "vel_error_m_s,vel_error_E,vel_error_N,vel_error_U,"
                    "alt_error_m,yaw_vio_deg,yaw_gps_deg,yaw_error_deg,"
                    "gps_lat,gps_lon,gps_alt,vio_E,vio_N,vio_U\n")
        
        # State debug log
        self.state_dbg_csv = os.path.join(self.config.output_dir, "state_debug.csv")
        with open(self.state_dbg_csv, "w") as f:
            f.write("t,px,py,pz,vx,vy,vz,a_world_x,a_world_y,a_world_z,dem,agl,msl\n")
    
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
            
            # DEM height updates
            self.process_dem_height(t)
            
            # VIO updates (simplified - full implementation in vio_vps.py)
            used_vo = False
            vo_data = None
            
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
