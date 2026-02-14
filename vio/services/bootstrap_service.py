"""Bootstrap/init service for VIORunner."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from ..camera import make_KD_for_size
from ..config import load_config
from ..data_loaders import (
    DEMReader,
    load_flight_log_msl,
    load_images,
    load_imu_csv,
    load_mag_csv,
    load_ppk_initial_state,
    load_ppk_trajectory,
)
from ..ekf import ExtendedKalmanFilter
from ..fisheye_rectifier import create_rectifier_from_config
from ..loop_closure import init_loop_closure
from ..output_utils import (
    DebugCSVWriters,
    build_calibration_params,
    init_output_csvs,
    save_calibration_log,
)
from ..plane_detection import PlaneDetector
from ..state_manager import initialize_ekf_state
from ..vio_frontend import VIOFrontEnd


class BootstrapService:
    """Encapsulates data loading + initialization for VIORunner."""

    def __init__(self, runner: Any):
        self.runner = runner

    def prepare_runtime_config(self):
        """
        Load effective runtime config into `runner.global_config`.

        Preference:
        1) compiled config from `VIOConfig._raw_config`
        2) load from YAML path
        """
        if hasattr(self.runner.config, "_raw_config") and self.runner.config._raw_config:
            self.runner.global_config = self.runner.config._raw_config
        elif self.runner.config.config_yaml:
            self.load_config()

        print(f"\n[CONFIG] Algorithm settings (from YAML):")
        print(f"  config_yaml: {self.runner.config.config_yaml}")
        print(f"  camera_view: {self.runner.config.camera_view}")
        print(f"  use_vio_velocity: {self.runner.config.use_vio_velocity}")
        print(f"  use_magnetometer: {self.runner.config.use_magnetometer}")
        print(f"  estimate_imu_bias: {self.runner.config.estimate_imu_bias}")
        print(f"  estimator_mode: {self.runner.config.estimator_mode}")
        print(f"  fast_mode: {self.runner.config.fast_mode}")
        print(f"  frame_skip: {self.runner.config.frame_skip}")

    def load_config(self) -> dict:
        """Load YAML configuration file."""
        if self.runner.config.config_yaml:
            cfg = load_config(self.runner.config.config_yaml)
            self.runner.global_config = cfg

            # Load flight phase detection thresholds from YAML (v3.4.0)
            self.runner.PHASE_SPINUP_VELOCITY_THRESH = cfg.get("PHASE_SPINUP_VELOCITY_THRESH", 1.0)
            self.runner.PHASE_SPINUP_VIBRATION_THRESH = cfg.get("PHASE_SPINUP_VIBRATION_THRESH", 0.3)
            self.runner.PHASE_SPINUP_ALT_CHANGE_THRESH = cfg.get("PHASE_SPINUP_ALT_CHANGE_THRESH", 5.0)
            self.runner.PHASE_EARLY_VELOCITY_SIGMA_THRESH = cfg.get("PHASE_EARLY_VELOCITY_SIGMA_THRESH", 3.0)
            self.runner.PHASE_HYSTERESIS_ENABLED = bool(cfg.get("PHASE_HYSTERESIS_ENABLED", True))
            self.runner.PHASE_UP_HOLD_SEC = float(cfg.get("PHASE_UP_HOLD_SEC", 0.75))
            self.runner.PHASE_DOWN_HOLD_SEC = float(cfg.get("PHASE_DOWN_HOLD_SEC", 6.0))
            self.runner.PHASE_ALLOW_NORMAL_TO_EARLY = bool(cfg.get("PHASE_ALLOW_NORMAL_TO_EARLY", True))
            self.runner.PHASE_REVERT_MAX_SPEED = float(cfg.get("PHASE_REVERT_MAX_SPEED", 18.0))
            self.runner.PHASE_REVERT_MAX_ALT_CHANGE = float(cfg.get("PHASE_REVERT_MAX_ALT_CHANGE", 60.0))

            return cfg
        return {}

    def load_data(self):
        """Load all input data sources."""
        runner = self.runner

        # Load PPK ground truth if available
        runner.ppk_state = None
        if runner.config.ground_truth_path:
            runner.ppk_state = load_ppk_initial_state(runner.config.ground_truth_path)

        # Load initial position from PPK (required for GPS-denied operation)
        if runner.ppk_state is not None:
            runner.lat0 = runner.ppk_state.lat
            runner.lon0 = runner.ppk_state.lon
            v_init = np.array([runner.ppk_state.ve, runner.ppk_state.vn, runner.ppk_state.vu])
            print(f"[INIT] Using PPK for lat/lon/velocity")
        else:
            raise RuntimeError("[INIT] PPK file is required for initialization. Check ppk_csv path in config.")

        runner.v_init = v_init

        # Use PPK ellipsoidal height
        runner.msl0 = runner.ppk_state.height
        print(f"[INIT] Using PPK ellipsoidal height: {runner.msl0:.1f}m")

        # Load main data
        runner.imu, runner.imu_time_col = load_imu_csv(runner.config.imu_path, return_time_col=True)
        runner.imgs = load_images(
            runner.config.images_dir,
            runner.config.images_index_csv,
            runner.config.timeref_csv,
        )

        # v3.10.x: Offline VPS list path is deprecated in runtime mode
        runner.vps_list = []

        # v3.9.4: Pass timeref_csv to magnetometer for hardware clock synchronization
        runner.mag_list = (
            load_mag_csv(runner.config.mag_csv, timeref_csv=runner.config.timeref_csv)
            if runner.config.use_magnetometer
            else []
        )

        runner.dem = DEMReader.open(runner.config.dem_path)

        # Load PPK trajectory for error comparison
        runner.ppk_trajectory = None
        if runner.config.ground_truth_path:
            runner.ppk_trajectory = load_ppk_trajectory(runner.config.ground_truth_path)

        # Configure mapping from filter time base -> absolute epoch for GT lookup
        self.configure_error_time_mapping()

        # v3.9.12: Load Flight Log MSL for direct altitude updates (Barometer/GNSS)
        runner.msl_interpolator = load_flight_log_msl(
            runner.config.quarry_path,
            runner.config.timeref_csv,
        )
        if runner.msl_interpolator:
            runner.msl_offset = 0.0
            if runner.ppk_state and runner.ppk_state.timestamp:
                t_start = runner.msl_interpolator.times[0]
                msl_log_start = runner.msl_interpolator.get_msl(t_start)
                if msl_log_start is not None:
                    runner.msl_offset = msl_log_start - runner.ppk_state.height
                    print(f"[INIT] Auto-Aligned MSL to PPK: Offset {runner.msl_offset:.3f}m")
                    print(f"       (Log: {msl_log_start:.3f}m, GT: {runner.ppk_state.height:.3f}m)")
            else:
                print(f"[INIT] Loaded Flight Log MSL (No Ground Truth for Alignment)")

        # Print summary
        print("=== Input check ===")
        print(f"IMU: {'OK' if len(runner.imu)>0 else 'MISSING'} ({len(runner.imu)} samples)")
        print(f"Images: {'OK' if len(runner.imgs)>0 else 'None'} ({len(runner.imgs)} frames)")
        print(f"Mag: {'OK' if len(runner.mag_list)>0 else 'None'} ({len(runner.mag_list)} samples)")
        print(f"DEM: {'OK' if runner.dem.ds else 'None'}")
        print(f"PPK: {'OK' if runner.ppk_state else 'None'}")

        if len(runner.imu) == 0:
            raise RuntimeError("IMU is required. Aborting.")

    def configure_error_time_mapping(self):
        """
        Configure timestamp mapping used by log_error().

        Goal:
        - If filter runs on time_ref, convert filter time to absolute epoch
          before matching with PPK stamp_log.
        - If filter already runs on absolute timestamps, use identity mapping.
        """
        runner = self.runner
        runner.error_time_scale = 1.0
        runner.error_time_offset = 0.0
        runner.error_time_mode = "identity"

        if runner.imu_time_col != "time_ref":
            print(f"[TIME] Error/GT lookup uses direct filter time ({runner.imu_time_col or 'unknown'})")
            return

        try:
            header_cols = pd.read_csv(runner.config.imu_path, nrows=0).columns.tolist()
            abs_col = next((c for c in ["stamp_bag", "stamp_msg", "stamp_log"] if c in header_cols), None)
            if abs_col is None:
                print("[TIME] WARNING: IMU has time_ref but no absolute stamp column for GT mapping")
                return

            df = pd.read_csv(runner.config.imu_path, usecols=["time_ref", abs_col])
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) < 2:
                print("[TIME] WARNING: Not enough IMU rows to fit time_ref->absolute mapping")
                return

            t_ref = df["time_ref"].astype(float).values
            t_abs = df[abs_col].astype(float).values
            scale, offset = np.polyfit(t_ref, t_abs, 1)
            pred = scale * t_ref + offset
            rms_ms = float(np.sqrt(np.mean((pred - t_abs) ** 2)) * 1000.0)

            runner.error_time_scale = float(scale)
            runner.error_time_offset = float(offset)
            runner.error_time_mode = f"time_ref_to_{abs_col}"

            print(
                f"[TIME] GT mapping enabled: time_ref -> {abs_col} | "
                f"scale={runner.error_time_scale:.12f}, offset={runner.error_time_offset:.6f}, rms={rms_ms:.3f}ms"
            )
        except Exception as e:
            print(f"[TIME] WARNING: Failed to configure GT time mapping: {e}")
            runner.error_time_scale = 1.0
            runner.error_time_offset = 0.0
            runner.error_time_mode = "identity"

    def filter_time_to_gt_time(self, t_filter: float) -> float:
        """Convert filter timestamp to GT lookup timestamp (absolute epoch)."""
        if self.runner.error_time_mode.startswith("time_ref_to_"):
            return self.runner.error_time_scale * t_filter + self.runner.error_time_offset
        return t_filter

    def initialize_ekf(self):
        """Initialize EKF state and covariance."""
        runner = self.runner

        # Ensure local projection is set up
        runner.proj_cache.ensure_proj(runner.lat0, runner.lon0)

        # Sample DEM at origin
        runner.dem0 = runner.dem.sample_m(runner.lat0, runner.lon0) if runner.dem.ds else None

        # v2.9.10.5: Check for manual AGL override in config (for helicopter flights)
        vio_cfg = runner.global_config.get("vio", {})
        agl_override = vio_cfg.get("initial_agl_override", None)

        if agl_override is not None and agl_override > 0:
            runner.initial_agl = agl_override
            computed_agl = abs(runner.msl0 - runner.dem0) if runner.dem0 is not None else 0.0
            print(f"[VIO] Initial AGL = {runner.initial_agl:.1f}m (OVERRIDE, computed was {computed_agl:.1f}m)")
        elif runner.dem0 is not None:
            runner.initial_agl = abs(runner.msl0 - runner.dem0)
            print(f"[VIO] Initial AGL = {runner.initial_agl:.1f}m (MSL={runner.msl0:.1f}, DEM={runner.dem0:.1f})")
        else:
            runner.initial_agl = 100.0
            print(f"[VIO] WARNING: No DEM, using default initial_agl = {runner.initial_agl:.1f}m")

        # Store in global config for VIO velocity update to access
        runner.global_config["INITIAL_AGL"] = runner.initial_agl

        # Create EKF (v3.9.7: 19D nominal state with mag_bias)
        runner.kf = ExtendedKalmanFilter(dim_x=19, dim_z=3, dim_u=3)
        runner.kf.x = np.zeros((19, 1), dtype=float)

        # Get config parameters from YAML (v3.2.0: no fallback - require proper config)
        imu_params = runner.global_config.get("IMU_PARAMS")
        if imu_params is None:
            raise RuntimeError("IMU_PARAMS not found in config. Check YAML file.")

        runner.lever_arm = runner.global_config.get("IMU_GNSS_LEVER_ARM", np.zeros(3))

        # v3.9.10: Use PPK ATTITUDE yaw for initial heading
        ppk_initial_yaw = None
        if runner.ppk_state is not None:
            yaw_ned = runner.ppk_state.yaw
            yaw_enu = np.pi / 2 - yaw_ned
            yaw_enu = np.arctan2(np.sin(yaw_enu), np.cos(yaw_enu))
            ppk_initial_yaw = yaw_enu

        # MAG params for initial correction
        mag_params = None
        if runner.config.use_magnetometer and len(runner.mag_list) > 0:
            mag_params = {
                "declination": runner.global_config.get("MAG_DECLINATION", 0.0),
                "use_raw_heading": runner.global_config.get("MAG_USE_RAW_HEADING", True),
                "min_field": runner.global_config.get("MAG_MIN_FIELD_STRENGTH", 0.1),
                "max_field": runner.global_config.get("MAG_MAX_FIELD_STRENGTH", 100.0),
                "hard_iron": runner.global_config.get("MAG_HARD_IRON_OFFSET", None),
                "soft_iron": runner.global_config.get("MAG_SOFT_IRON_MATRIX", None),
                "sigma_mag_bias_init": runner.global_config.get("SIGMA_MAG_BIAS_INIT", 0.1),
                "sigma_mag_bias": runner.global_config.get("SIGMA_MAG_BIAS", 0.001),
                "use_estimated_bias": runner.config.use_mag_estimated_bias,
            }

        # Initialize state
        init_state = initialize_ekf_state(
            kf=runner.kf,
            ppk_state=runner.ppk_state,
            imu_records=runner.imu,
            imu_params=imu_params,
            lever_arm=runner.lever_arm,
            lat0=runner.lat0,
            lon0=runner.lon0,
            msl0=runner.msl0,
            dem0=runner.dem0,
            v_init_enu=runner.v_init,
            estimate_imu_bias=runner.config.estimate_imu_bias,
            initial_gyro_bias=runner.global_config.get("INITIAL_GYRO_BIAS"),
            initial_accel_bias=runner.global_config.get("INITIAL_ACCEL_BIAS"),
            mag_records=runner.mag_list if mag_params else None,
            mag_params=mag_params,
            ppk_initial_yaw=ppk_initial_yaw,
        )

        return init_state

    def initialize_vio_frontend(self):
        """Initialize VIO frontend if images are available."""
        runner = self.runner
        if len(runner.imgs) == 0:
            return

        kb_params = runner.global_config.get("KB_PARAMS", {})
        use_fisheye = runner.global_config.get("USE_FISHEYE", True)

        K, D = make_KD_for_size(
            kb_params,
            runner.config.downscale_size[0],
            runner.config.downscale_size[1],
        )

        runner.vio_fe = VIOFrontEnd(
            runner.config.downscale_size[0],
            runner.config.downscale_size[1],
            K,
            D,
            use_fisheye=use_fisheye,
            fast_mode=runner.config.fast_mode,
        )

        # Override min_track_length from config
        runner.vio_fe.min_track_length = runner.global_config.get("MSCKF_MIN_TRACK_LENGTH", 4)

        print(f"[VIO] Camera view mode: {runner.config.camera_view}")
        print(
            f"[MSCKF] max_clone_size={runner.global_config.get('MSCKF_MAX_CLONE_SIZE', 11)}, "
            f"min_track_length={runner.vio_fe.min_track_length}"
        )

        if runner.global_config.get("USE_PLANE_MSCKF", False):
            runner.plane_detector = PlaneDetector(
                min_points_per_plane=runner.global_config.get("PLANE_MIN_POINTS", 10),
                normal_angle_threshold=runner.global_config.get("PLANE_ANGLE_THRESHOLD", np.radians(15.0)),
                distance_threshold=runner.global_config.get("PLANE_DISTANCE_THRESHOLD", 0.15),
                min_plane_area=runner.global_config.get("PLANE_MIN_AREA", 0.5),
            )
            print(
                f"[Plane-MSCKF] Enabled with min_points={runner.plane_detector.min_points}, "
                f"angle_thresh={np.degrees(runner.plane_detector.angle_threshold):.1f}°"
            )

    def initialize_rectifier(self):
        """Initialize optional fisheye rectifier."""
        runner = self.runner
        use_rectifier = runner.global_config.get("USE_RECTIFIER", False)
        if not use_rectifier:
            print("[RECTIFY] Fisheye rectification disabled")
            return

        if len(runner.imgs) == 0:
            return

        try:
            rectify_fov = runner.global_config.get("RECTIFY_FOV_DEG", 90.0)
            runner.rectifier = create_rectifier_from_config(
                runner.global_config,
                src_size=runner.config.downscale_size,
                fov_deg=rectify_fov,
                dst_size=runner.config.downscale_size,
            )
            print(f"[RECTIFY] Fisheye rectifier initialized (FOV={rectify_fov}°)")
        except Exception as e:
            print(f"[RECTIFY] WARNING: Failed to initialize rectifier: {e}")
            runner.rectifier = None

    def initialize_loop_closure(self):
        """Initialize optional loop-closure detector."""
        runner = self.runner
        use_loop_closure = runner.global_config.get("USE_LOOP_CLOSURE", True)
        if not use_loop_closure:
            print("[LOOP] Loop closure detection disabled")
            return

        if len(runner.imgs) == 0:
            print("[LOOP] No images available, loop closure disabled")
            return

        try:
            position_threshold = runner.global_config.get("LOOP_POSITION_THRESHOLD", 30.0)
            runner.loop_detector = init_loop_closure(position_threshold=position_threshold)
            print(f"[LOOP] Loop closure detector initialized (threshold={position_threshold}m)")
        except Exception as e:
            print(f"[LOOP] WARNING: Failed to initialize loop closure: {e}")
            runner.loop_detector = None

    def setup_output_files(self):
        """Create output directory and all CSV files."""
        runner = self.runner
        os.makedirs(runner.config.output_dir, exist_ok=True)

        # Use output_utils to initialize CSVs. Heavy debug files are opt-in.
        csv_paths = init_output_csvs(
            runner.config.output_dir,
            save_debug_data=bool(runner.config.save_debug_data),
        )
        runner.pose_csv = csv_paths["pose_csv"]
        runner.error_csv = csv_paths["error_csv"]
        runner.state_dbg_csv = csv_paths.get("state_dbg_csv")
        runner.time_sync_csv = csv_paths.get("time_sync_csv")
        runner.cov_health_csv = csv_paths.get("cov_health_csv")
        runner.convention_csv = csv_paths.get("convention_csv")
        runner.adaptive_debug_csv = csv_paths.get("adaptive_debug_csv")
        runner.sensor_health_csv = csv_paths.get("sensor_health_csv")
        runner.conditioning_events_csv = csv_paths.get("conditioning_events_csv")
        runner.benchmark_health_summary_csv = csv_paths.get("benchmark_health_summary_csv")
        runner.inf_csv = csv_paths["inf_csv"]
        runner.vo_dbg_csv = csv_paths.get("vo_dbg")
        runner.msckf_dbg_csv = csv_paths.get("msckf_dbg")

        if runner.kf is not None and hasattr(runner.kf, "enable_cov_health_logging") and runner.cov_health_csv:
            runner.kf.enable_cov_health_logging(runner.cov_health_csv)
        if runner.kf is not None and hasattr(runner.kf, "enable_conditioning_event_logging") and runner.conditioning_events_csv:
            runner.kf.enable_conditioning_event_logging(runner.conditioning_events_csv)

        # Use DebugCSVWriters for optional debug files
        runner.debug_writers = DebugCSVWriters(runner.config.output_dir, runner.config.save_debug_data)
        runner.imu_raw_csv = runner.debug_writers.imu_raw_csv
        runner.state_cov_csv = runner.debug_writers.state_cov_csv
        runner.residual_csv = runner.debug_writers.residual_csv
        runner.feature_stats_csv = runner.debug_writers.feature_stats_csv
        runner.msckf_window_csv = runner.debug_writers.msckf_window_csv
        runner.fej_csv = runner.debug_writers.fej_consistency_csv

        runner.keyframe_dir = None
        if runner.config.save_debug_data:
            print(f"[DEBUG] Debug data logging enabled")
        else:
            print("[DEBUG] Debug tier=light (heavy CSV debug disabled)")

        if runner.config.save_keyframe_images:
            runner.keyframe_dir = os.path.join(runner.config.output_dir, "debug_keyframes")
            os.makedirs(runner.keyframe_dir, exist_ok=True)
            print(f"[DEBUG] Keyframe images will be saved to: {runner.keyframe_dir}")

            runner.vps_matches_dir = os.path.join(runner.config.output_dir, "debug_vps_matches")
            os.makedirs(runner.vps_matches_dir, exist_ok=True)
            print(f"[DEBUG] VPS match images will be saved to: {runner.vps_matches_dir}")
        else:
            runner.vps_matches_dir = None

        # VPS Debug Logger (for future VPSRunner integration)
        runner.vps_logger = None
        if runner.config.save_debug_data:
            vps_cfg = {}
            if hasattr(runner.config, "_yaml_config") and runner.config._yaml_config:
                vps_cfg = runner.config._yaml_config.get("vps", {})
            if vps_cfg.get("enabled", False):
                try:
                    from vps import VPSDebugLogger

                    runner.vps_logger = VPSDebugLogger(
                        output_dir=runner.config.output_dir,
                        enabled=True,
                    )
                    print(f"[DEBUG] VPS logger enabled (debug_vps_attempts.csv, debug_vps_matches.csv)")
                except ImportError:
                    print(f"[WARNING] VPS module not available, VPS logging disabled")

        # VPS Real-time Runner (for live VPS processing)
        runner.vps_runner = None
        vps_cfg = {}
        if hasattr(runner.config, "_yaml_config") and runner.config._yaml_config:
            vps_cfg = runner.config._yaml_config.get("vps", {})

        if vps_cfg.get("enabled", False):
            try:
                from vps import VPSRunner

                mbtiles_path = runner.config.mbtiles_path
                if not mbtiles_path:
                    mbtiles_path = vps_cfg.get("mbtiles_path", "mission.mbtiles")

                if os.path.exists(mbtiles_path):
                    runner.vps_runner = VPSRunner.create_from_config(
                        mbtiles_path=mbtiles_path,
                        config_path=runner.config.config_yaml,
                        device=vps_cfg.get("device", "cpu"),
                    )

                    if runner.vps_logger is not None:
                        runner.vps_runner.set_logger(runner.vps_logger)
                        print(f"[VPS] Real-time VPS enabled with logging")
                    else:
                        print(f"[VPS] Real-time VPS enabled with logging")

                    if hasattr(runner, "vps_matches_dir") and runner.vps_matches_dir:
                        runner.vps_runner.save_matches_dir = runner.vps_matches_dir
                        print(f"[VPS] Match visualizations will be saved")

                    from vps import VPSDelayedUpdateManager

                    vps_delay_cfg = vps_cfg.get("delayed_update", {})
                    runner.vps_clone_manager = VPSDelayedUpdateManager(
                        max_delay_sec=vps_delay_cfg.get("max_delay_sec", 0.5),
                        max_clones=vps_delay_cfg.get("max_clones", 3),
                    )
                    runner.vps_runner.delayed_update_enabled = True
                    print(f"[VPS] Delayed update manager initialized (stochastic cloning enabled)")
                else:
                    print(f"[WARNING] VPS enabled but MBTiles not found: {mbtiles_path}")
                    print(f"          Please create MBTiles using: python -m vps.tile_prefetcher")
            except ImportError as e:
                print(f"[WARNING] VPS module not available: {e}")

        # Save calibration snapshot using output_utils
        if runner.config.save_debug_data:
            cal_path = os.path.join(runner.config.output_dir, "debug_calibration.txt")
            cal_params = build_calibration_params(
                global_config=runner.global_config,
                vio_config=runner.config,
                lat0=runner.lat0,
                lon0=runner.lon0,
                alt0=getattr(runner, "alt0", 0.0),
            )
            save_calibration_log(output_path=cal_path, **cal_params)
            print(f"[DEBUG] Calibration snapshot saved: {cal_path}")
