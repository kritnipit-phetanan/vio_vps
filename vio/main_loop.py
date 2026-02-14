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

import numpy as np
from typing import Dict, Optional

from .config import VIOConfig
from .data_loaders import ProjectionCache
from .adaptive_controller import AdaptiveController, AdaptiveDecision
from .fisheye_rectifier import FisheyeRectifier
from .loop_closure import LoopClosureDetector
from .propagation import VibrationDetector
from .state_container import RunnerState
from .trn import TerrainReferencedNavigation
from .imu_driven import run_imu_driven_loop
from .event_driven import run_event_driven_loop
from .services.adaptive_service import AdaptiveService
from .services.bootstrap_service import BootstrapService
from .services.output_reporting_service import OutputReportingService
from .services.phase_service import PhaseService
from .services.magnetometer_service import MagnetometerService
from .services.dem_service import DEMService
from .services.vps_service import VPSService
from .services.vio_service import VIOService
from .services.imu_update_service import IMUUpdateService


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
        self.state = RunnerState()
        
        # Flight phase detection thresholds (v3.4.0: State-based)
        self.PHASE_SPINUP_VELOCITY_THRESH = 1.0
        self.PHASE_SPINUP_VIBRATION_THRESH = 0.3
        self.PHASE_SPINUP_ALT_CHANGE_THRESH = 5.0
        self.PHASE_EARLY_VELOCITY_SIGMA_THRESH = 3.0
        self.PHASE_HYSTERESIS_ENABLED = True
        self.PHASE_UP_HOLD_SEC = 0.75
        self.PHASE_DOWN_HOLD_SEC = 6.0
        self.PHASE_ALLOW_NORMAL_TO_EARLY = True
        self.PHASE_REVERT_MAX_SPEED = 18.0
        self.PHASE_REVERT_MAX_ALT_CHANGE = 60.0
        
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
        self.inf_csv = None
        self.state_dbg_csv = None
        self.vo_dbg_csv = None
        self.msckf_dbg_csv = None
        self.time_sync_csv = None
        self.cov_health_csv = None
        self.convention_csv = None
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
        self._vps_last_attempt_t = -1e9
        self._vps_next_allowed_t = 0.0
        self._vps_skip_streak = 0
        
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
        self._sensor_health_log_stride: int = 1
        self._sensor_health_log_counts: Dict[str, int] = {}
        self._legacy_covariance_cap: float = 1e8
        self._adaptive_last_policy_t: Optional[float] = None
        self._conditioning_warning_sec: float = 0.0
        self.imu_only_mode: bool = False
        self._imu_only_yaw_ref: Optional[float] = None
        self._imu_only_bg_ref: Optional[np.ndarray] = None
        self._imu_only_ba_ref: Optional[np.ndarray] = None
        self._vision_yaw_ref: Optional[float] = None
        self._vision_yaw_last_t: Optional[float] = None
        self._vision_heading_quality: float = 0.0
        self._convention_warn_counts: Dict[str, int] = {}
        self.phase_service = PhaseService(self)
        self.adaptive_service = AdaptiveService(self)
        self.output_reporting = OutputReportingService(self)
        self.bootstrap_service = BootstrapService(self)
        self.magnetometer_service = MagnetometerService(self)
        self.dem_service = DEMService(self)
        self.vps_service = VPSService(self)
        self.vio_service = VIOService(self)
        self.imu_update_service = IMUUpdateService(self)
    
    def run(self):
        """
        Run the complete VIO pipeline as a pure orchestrator.
        """
        print("=" * 80)
        print("VIO+EKF Pipeline Starting (v3.6.0)")
        print("=" * 80)

        # Compile runtime config and initialize adaptive policy controller.
        self.bootstrap_service.prepare_runtime_config()
        self.adaptive_service.initialize_adaptive_controller()

        duration_sec = None
        if self.config.estimator_mode == "event_queue_output_predictor":
            print("\n[ARCH] Selected: Event-driven with output predictor (event_queue_output_predictor)")
            print("[ARCH] Features:")
            print("  - Priority queue orders all sensor events by timestamp")
            print("  - Propagate-to-event ensures state_time == measurement_time")
            print("  - Fast-propagate output layer for low-latency logging")
            duration_sec = run_event_driven_loop(self)
        elif self.config.estimator_mode == "imu_step_preint_cache":
            print("\n[ARCH] Selected: IMU-driven with sub-sample timestamp precision (imu_step_preint_cache)")
            duration_sec = run_imu_driven_loop(self)
        else:
            raise ValueError(f"Unknown estimator_mode: {self.config.estimator_mode}")

        # Finalize reporting in one place.
        self.output_reporting.print_summary()
        return duration_sec
