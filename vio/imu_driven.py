"""
IMU-driven VIO Loop Implementation

This module contains the IMU-driven architecture implementation (v3.7.0).
Separated from main_loop.py for modularity.

Author: VIO project
"""
import os
import time
import gc
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import resource

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from .imu_preintegration import IMUPreintegration
from .propagation import (
    process_imu, VibrationDetector
)
from .magnetometer import reset_mag_filter_state, set_mag_constants
from .trn import create_trn_from_config
from .data_loaders import ProjectionCache
from .output_utils import log_state_debug


def _backend_poll_due(global_config, t_now: float, last_poll_t: float, cam_tick: bool) -> bool:
    """Return whether backend correction poll/apply should run at this step."""
    poll_on_cam = bool(global_config.get("BACKEND_TRANSPORT_POLL_ON_CAMERA_TICK_ONLY", True))
    if poll_on_cam:
        if bool(cam_tick):
            return True
        min_interval = float(
            global_config.get(
                "BACKEND_TRANSPORT_POLL_MIN_INTERVAL_SEC",
                global_config.get("BACKEND_POLL_INTERVAL_SEC", 0.5),
            )
        )
        return (float(t_now) - float(last_poll_t)) >= max(0.05, float(min_interval))
    poll_interval_sec = float(global_config.get("BACKEND_POLL_INTERVAL_SEC", 0.5))
    return (float(t_now) - float(last_poll_t)) >= max(0.01, float(poll_interval_sec))


def _process_memory_mb() -> tuple[float, float, float]:
    """Best-effort current process memory: (rss_mb, vms_mb, uss_mb)."""
    rss_mb = float("nan")
    vms_mb = float("nan")
    uss_mb = float("nan")
    if psutil is not None:
        try:
            proc = psutil.Process()
            mi = proc.memory_info()
            rss_mb = float(mi.rss / (1024.0 * 1024.0))
            vms_mb = float(mi.vms / (1024.0 * 1024.0))
            try:
                mfi = proc.memory_full_info()
                uss = getattr(mfi, "uss", None)
                if uss is not None:
                    uss_mb = float(float(uss) / (1024.0 * 1024.0))
            except Exception:
                pass
            return rss_mb, vms_mb, uss_mb
        except Exception:
            pass
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss_mb = float(ru / (1024.0 * 1024.0))
        else:
            rss_mb = float(ru / 1024.0)
    except Exception:
        pass
    return rss_mb, vms_mb, uss_mb


def _memory_watchdog_tick(runner, t_now: float) -> None:
    """Runtime memory guard with cache compaction + temporary VPS backpressure."""
    cfg = runner.global_config if isinstance(runner.global_config, dict) else {}
    if not bool(cfg.get("MEMORY_WATCHDOG_ENABLE", True)):
        return
    check_interval = max(0.25, float(cfg.get("MEMORY_CHECK_INTERVAL_SEC", 1.0)))
    if float(t_now) - float(getattr(runner, "_mem_last_check_t", -1e9)) < check_interval:
        return
    runner._mem_last_check_t = float(t_now)

    rss_mb, vms_mb, uss_mb = _process_memory_mb()
    if not np.isfinite(rss_mb):
        return
    runner._memory_peak_rss_mb = float(max(float(getattr(runner, "_memory_peak_rss_mb", 0.0)), float(rss_mb)))
    if np.isfinite(vms_mb):
        runner._memory_peak_vms_mb = float(
            max(float(getattr(runner, "_memory_peak_vms_mb", 0.0)), float(vms_mb))
        )
    if np.isfinite(uss_mb):
        runner._memory_peak_uss_mb = float(
            max(float(getattr(runner, "_memory_peak_uss_mb", 0.0)), float(uss_mb))
        )
    soft_mb = max(512.0, float(cfg.get("MEMORY_SOFT_LIMIT_GB", 16.0)) * 1024.0)
    hard_mb = max(soft_mb + 512.0, float(cfg.get("MEMORY_HARD_LIMIT_GB", 24.0)) * 1024.0)
    compact_cd = max(0.25, float(cfg.get("MEMORY_COMPACT_COOLDOWN_SEC", 2.0)))
    periodic_compact_sec = max(0.0, float(cfg.get("MEMORY_PERIODIC_COMPACT_SEC", 10.0)))
    periodic_due = bool(
        periodic_compact_sec > 0.0
        and (float(t_now) - float(getattr(runner, "_mem_last_periodic_compact_t", -1e9))) >= periodic_compact_sec
    )
    pressure_due = bool(rss_mb >= soft_mb)
    need_compact = bool(periodic_due or pressure_due)

    if need_compact and (float(t_now) - float(getattr(runner, "_mem_last_compact_t", -1e9))) >= compact_cd:
        try:
            if runner.vio_fe is not None:
                runner.vio_fe.compact_memory(
                    max_track_length=int(cfg.get("MEMORY_COMPACT_FRONTEND_TRACK_LEN", 90)),
                    max_total_tracks=int(cfg.get("MEMORY_COMPACT_FRONTEND_MAX_TRACKS", 5000)),
                )
        except Exception:
            pass
        try:
            if getattr(runner, "loop_detector", None) is not None:
                runner.loop_detector.compact_memory(
                    max_keyframes=int(cfg.get("MEMORY_COMPACT_LOOP_MAX_KEYFRAMES", 220))
                )
        except Exception:
            pass
        try:
            if getattr(runner, "vps_runner", None) is not None:
                runner.vps_runner.compact_runtime_caches(
                    max_cached_tiles=int(cfg.get("MEMORY_COMPACT_VPS_TILE_CACHE_MAX", 50)),
                    keep_attempt_wall_samples=int(cfg.get("MEMORY_COMPACT_VPS_WALL_SAMPLES", 1200)),
                )
        except Exception:
            pass
        gc.collect()
        runner._mem_last_compact_t = float(t_now)
        if periodic_due:
            runner._mem_last_periodic_compact_t = float(t_now)
        runner._memory_compact_count = int(getattr(runner, "_memory_compact_count", 0)) + 1

    if rss_mb >= hard_mb:
        pause_sec = max(0.5, float(cfg.get("MEMORY_VPS_PAUSE_SEC", 4.0)))
        runner._vps_memory_pressure_until_t = max(
            float(getattr(runner, "_vps_memory_pressure_until_t", -1e9)),
            float(t_now) + float(pause_sec),
        )
        runner._memory_pressure_events = int(getattr(runner, "_memory_pressure_events", 0)) + 1
        last_warn = float(getattr(runner, "_mem_last_warn_t", -1e9))
        if (float(t_now) - last_warn) > 2.0:
            if bool(getattr(runner.config, "save_debug_data", False)):
                print(
                    "[MEM] pressure: "
                    f"rss={rss_mb/1024.0:.2f}GB (soft={soft_mb/1024.0:.1f}, hard={hard_mb/1024.0:.1f}), "
                    f"pause_vps={pause_sec:.1f}s"
                )
            runner._mem_last_warn_t = float(t_now)


def run_imu_driven_loop(runner):
    """
    IMU-driven VIO loop (v3.7.0).
    
    Architecture:
    - Iterate through all IMU samples @ 400Hz
    - Accumulate preintegration buffer for MSCKF Jacobians
    - Propagate state + covariance every IMU tick
    - Process measurements when their timestamps are reached
    
    Sub-sample Timestamp Precision:
    - Split dt when crossing camera timestamp (OpenVINS-style)
    - Propagate [t_last → t_cam] → process camera → [t_cam → t]
    - Zero timestamp lag at camera events
    
    Advantages:
    - Simple sequential processing
    - Streaming-friendly (no need to buffer all data)
    - Clear separation between propagation and update
    - Exact timestamp synchronization
    
    Args:
        runner: VIORunner instance with initialized config and state
    """
    # Load data
    runner.bootstrap_service.load_data()
    
    # Initialize EKF
    runner.bootstrap_service.initialize_ekf()
    
    # Initialize VIO frontend
    runner.bootstrap_service.initialize_vio_frontend()
    
    # Initialize fisheye rectifier (optional - for converting fisheye to pinhole)
    runner.bootstrap_service.initialize_rectifier()
    
    # Initialize loop closure detector (for yaw drift correction)
    runner.bootstrap_service.initialize_loop_closure()
    
    # Initialize magnetometer filter state
    reset_mag_filter_state()
    
    # Set magnetometer constants from config
    mag_ema_alpha = runner.global_config.get('MAG_EMA_ALPHA', 0.3)
    mag_max_yaw_rate_deg = runner.global_config.get('MAG_MAX_YAW_RATE_DEG', 30.0)
    mag_gyro_threshold_deg = runner.global_config.get('MAG_GYRO_THRESHOLD_DEG', 10.0)
    mag_r_inflate = runner.global_config.get('MAG_R_INFLATE', 5.0)
    set_mag_constants(
        ema_alpha=mag_ema_alpha,
        max_yaw_rate=np.radians(mag_max_yaw_rate_deg),
        gyro_threshold=np.radians(mag_gyro_threshold_deg),
        consistency_r_inflate=mag_r_inflate
    )
    print(f"[MAG-FILTER] Initialized: EMA_α={mag_ema_alpha:.2f}, max_rate={mag_max_yaw_rate_deg:.1f}°/s, gyro_thresh={mag_gyro_threshold_deg:.1f}°")
    
    # Initialize vibration detector
    imu_params = runner.global_config.get('IMU_PARAMS', {})
    sigma_accel = runner.global_config.get('SIGMA_ACCEL', 0.8)
    
    # IMU-only stability profile:
    # If no aiding sensors are available, use conservative process noise to prevent
    # covariance blow-up from an aided-flight tuning profile.
    has_images = len(runner.imgs) > 0
    has_mag = len(runner.mag_list) > 0 and runner.config.use_magnetometer
    has_dem = bool(runner.dem is not None and runner.dem.ds is not None)
    has_vps = bool(getattr(runner, 'vps_runner', None) is not None)
    imu_only_mode = not (has_images or has_mag or has_dem or has_vps)
    runner.imu_only_mode = bool(imu_only_mode)
    if imu_only_mode:
        imu_params = dict(imu_params)
        sigma_accel = min(float(sigma_accel), float(imu_params.get('acc_n', 0.08)) * 1.5)
        imu_params['gyr_w'] = min(float(imu_params.get('gyr_w', 1e-4)), 1e-5)
        imu_params['acc_w'] = min(float(imu_params.get('acc_w', 4e-5)), 1e-5)
        imu_params['sigma_unmodeled_gyr'] = min(float(imu_params.get('sigma_unmodeled_gyr', 0.002)), 5e-4)
        imu_params['min_yaw_process_noise_deg'] = min(
            float(imu_params.get('min_yaw_process_noise_deg', 3.0)), 0.5
        )
        print(
            f"[IMU-ONLY] Conservative process-noise profile enabled: "
            f"sigma_accel={sigma_accel:.4f}, gyr_w={imu_params['gyr_w']:.2e}, "
            f"acc_w={imu_params['acc_w']:.2e}, sigma_unmodeled_gyr={imu_params['sigma_unmodeled_gyr']:.2e}, "
            f"min_yaw_q={imu_params['min_yaw_process_noise_deg']:.2f}deg"
        )
    imu_params_base = dict(imu_params)
    sigma_accel_base = float(sigma_accel)
    vib_buffer_size = runner.global_config.get('VIBRATION_WINDOW_SIZE', 50)
    vib_threshold_mult = runner.global_config.get('VIBRATION_THRESHOLD_MULT', 5.0)
    runner.vibration_detector = VibrationDetector(
        buffer_size=vib_buffer_size,
        threshold=imu_params.get('acc_n', 0.1) * vib_threshold_mult
    )
    
    # v3.9.7: Prepare mag params for process noise
    # v3.9.8: Include use_estimated_bias flag to freeze states when disabled
    mag_params = {
        'sigma_mag_bias': runner.config.sigma_mag_bias,
        'use_estimated_bias': runner.config.use_mag_estimated_bias
    }
    
    # Initialize TRN (Terrain Referenced Navigation) - v3.3.0
    if runner.global_config.get('TRN_ENABLED', False):
        runner.trn = create_trn_from_config(runner.dem, runner.global_config)
        if runner.trn:
            trn_cfg = runner.global_config.get('trn', {}) or {}
            print(
                f"[TRN] Initialized: search_radius={trn_cfg.get('search_radius_m', 500)}m, "
                f"update_interval={trn_cfg.get('update_interval_sec', 10)}s"
            )
        else:
            print("[TRN] WARNING: Failed to initialize TRN")
    else:
        print("[TRN] Disabled in config")
    
    # Store initial MSL for altitude change tracking
    runner.initial_msl = runner.msl0
    
    # Setup output files
    runner.bootstrap_service.setup_output_files()
    runner.bootstrap_service.initialize_backend_optimizer()
    
    # Initialize timing
    runner.state.t0 = runner.imu[0].t
    runner.state.last_t = runner.state.t0
    runner.output_reporting.run_bootstrap_convention_checks()
    runner.output_reporting.run_sensor_time_audit()
    
    # Initialize preintegration cache for MSCKF Jacobians
    # Keep runtime-adjusted imu_params (e.g., imu_only conservative profile).
    ongoing_preint = IMUPreintegration(
        bg=runner.kf.x[10:13, 0].reshape(3,),
        ba=runner.kf.x[13:16, 0].reshape(3,),
        sigma_g=imu_params.get('gyr_n', 0.01),
        sigma_a=imu_params.get('acc_n', 0.1),
        sigma_bg=imu_params.get('gyr_w', 0.0001),
        sigma_ba=imu_params.get('acc_w', 0.001)
    )
    print(f"[PREINT] Initialized preintegration cache for MSCKF Jacobians")
    
    # Main IMU-driven loop
    print("\n=== Running (IMU-driven with preintegration cache) ===")
    tic_all = time.time()
    stop_time_env = os.getenv("VIO_STOP_TIME", "").strip()
    try:
        stop_time_sim = float(stop_time_env) if stop_time_env else float("nan")
    except Exception:
        stop_time_sim = float("nan")
    
    # Store last a_world for state_debug logging
    last_a_world = np.array([0.0, 0.0, 0.0])
    
    for i, rec in enumerate(runner.imu):
        # Start timing for inference_log
        tic_iter = time.time()
        
        t = rec.t
        dt = max(0.0, float(t - runner.state.last_t)) if i > 0 else 0.0
        
        time_elapsed = t - runner.state.t0
        
        # Update vibration detector
        gyro_mag = np.linalg.norm(rec.ang)
        is_high_vibration, vibration_level = runner.vibration_detector.update(gyro_mag)
        
        # Get current velocity and uncertainty for state-based phase detection
        velocity = runner.kf.x[3:6, 0].flatten() if runner.kf is not None else None
        velocity_sigma = None
        if runner.kf is not None and runner.kf.P.shape[0] > 5:
            velocity_sigma = np.sqrt(np.mean(np.diag(runner.kf.P[3:6, 3:6])))
        
        altitude_change = runner.kf.x[2, 0] - runner.initial_msl if runner.kf is not None else 0.0
        
        # Flight phase with hysteresis/one-way protection.
        phase_num = runner.phase_service.estimate_flight_phase(
            velocity=velocity,
            velocity_sigma=velocity_sigma,
            vibration_level=vibration_level,
            altitude_change=altitude_change,
            dt=dt,
        )
        runner.state.current_phase = phase_num
        runner.adaptive_service.update_adaptive_policy(t=t, phase=phase_num)
        if getattr(runner, "policy_runtime_service", None) is not None:
            try:
                runner.policy_runtime_service.build_snapshot(float(t))
            except Exception:
                pass
        imu_params_step, sigma_accel_step = runner.adaptive_service.build_step_imu_params(
            imu_params_base, sigma_accel_base
        )
        _, zupt_apply_scales = runner.adaptive_service.get_sensor_adaptive_scales("ZUPT")
        
        # =====================================================================
        # IMU Propagation (v3.7.0: IMU-driven with sub-sample timestamp precision)
        # =====================================================================
        # Architecture: Always propagate x+P at every IMU tick (400Hz)
        # Sub-sample precision: Split dt when crossing camera timestamp (OpenVINS-style)
        # Preintegration cache is ONLY for MSCKF Jacobians, NOT to skip propagation
        
        # Check if we cross camera timestamp during this IMU step
        next_cam_time = runner.imgs[runner.state.img_idx].t if runner.state.img_idx < len(runner.imgs) else float('inf')
        t_last = runner.state.last_t
        t_current = rec.t
        
        # Detect camera crossing: last_t < t_cam <= current_t
        vio_frame_before = int(getattr(runner.state, "vio_frame", 0))
        if t_last < next_cam_time <= t_current and dt > 0:
            # Split dt at camera timestamp for sub-sample precision
            dt_before = next_cam_time - t_last  # Propagate to camera time
            dt_after = t_current - next_cam_time  # Propagate remaining
            
            # Step 1a: Propagate to camera time (exact timestamp)
            ongoing_preint.integrate_measurement(rec.ang, rec.lin, dt_before)
            process_imu(
                runner.kf, rec, dt_before,
                estimate_imu_bias=runner.config.estimate_imu_bias,
                t=next_cam_time, t0=runner.state.t0,
                imu_params=imu_params_step,
                mag_params=mag_params,
                sigma_accel=sigma_accel_step
            )
            # VPS Stochastic Cloning: Propagate cross-covariance
            if hasattr(runner, 'vps_clone_manager') and runner.vps_clone_manager is not None:
                Phi_approx = np.eye(18, dtype=float)
                Phi_approx[0:3, 3:6] = np.eye(3) * dt_before
                runner.vps_clone_manager.propagate_cross_covariance(Phi_approx)
            runner.imu_update_service.update_imu_helpers(rec, dt_before, imu_params_step, zupt_scales=zupt_apply_scales)
            runner.state.last_t = next_cam_time
            
            # Process camera at exact timestamp
            # VIO updates (feature tracking, MSCKF, velocity) at t_cam
            used_vo, vo_data = runner.vio_service.process_vio(rec, next_cam_time, ongoing_preint)
            
            # Step 1b: Propagate remaining dt after camera
            if dt_after > 1e-6:  # Only if significant remaining time
                ongoing_preint.integrate_measurement(rec.ang, rec.lin, dt_after)
                process_imu(
                    runner.kf, rec, dt_after,
                    estimate_imu_bias=runner.config.estimate_imu_bias,
                    t=t_current, t0=runner.state.t0,
                    imu_params=imu_params_step,
                    mag_params=mag_params,
                    sigma_accel=sigma_accel_step
                )
                # VPS Stochastic Cloning: Propagate cross-covariance
                if hasattr(runner, 'vps_clone_manager') and runner.vps_clone_manager is not None:
                    Phi_approx = np.eye(18, dtype=float)
                    Phi_approx[0:3, 3:6] = np.eye(3) * dt_after
                    runner.vps_clone_manager.propagate_cross_covariance(Phi_approx)
                runner.imu_update_service.update_imu_helpers(rec, dt_after, imu_params_step, zupt_scales=zupt_apply_scales)
            runner.state.last_t = t_current
        else:
            # Normal propagation (no camera crossing)
            # Step 1: Accumulate IMU measurements in preintegration cache for MSCKF
            ongoing_preint.integrate_measurement(rec.ang, rec.lin, dt)
            
            # Step 2: Propagate state + covariance (REQUIRED for consistent EKF updates)
            # This ensures BOTH x and P are at current time when VPS/MAG/DEM arrive
            process_imu(
                runner.kf, rec, dt,
                estimate_imu_bias=runner.config.estimate_imu_bias,
                t=rec.t, t0=runner.state.t0,
                imu_params=imu_params_step,
                mag_params=mag_params,
                sigma_accel=sigma_accel_step
            )
            
            # VPS Stochastic Cloning: Propagate cross-covariance for delayed updates
            # Use approximate Phi (identity for small dt) since process_imu doesn't return it
            if hasattr(runner, 'vps_clone_manager') and runner.vps_clone_manager is not None:
                # For small dt (~2.5ms @ 400Hz), identity is good first-order approximation
                # Full Phi would require extracting from process_imu internals
                Phi_approx = np.eye(18, dtype=float)
                Phi_approx[0:3, 3:6] = np.eye(3) * dt  # d(pos)/dt = vel
                runner.vps_clone_manager.propagate_cross_covariance(Phi_approx)
            
            # Step 3: Update mag filter variables and check ZUPT
            runner.imu_update_service.update_imu_helpers(rec, dt, imu_params_step, zupt_scales=zupt_apply_scales)
            
            # Update state time
            runner.state.last_t = t
            
            # VIO updates (feature tracking, MSCKF, velocity) if no camera crossed
            used_vo, vo_data = runner.vio_service.process_vio(rec, t, ongoing_preint)
        
        # Increment counter for debug logging
        runner.state.imu_propagation_count += 1
        
        # Debug logging: raw IMU and state covariance
        runner.debug_writers.log_imu_raw(t, rec)
        if i % 10 == 0:  # Every 10 samples (~25ms @ 400Hz)
            bg = runner.kf.x[10:13, 0]
            ba = runner.kf.x[13:16, 0]
            runner.debug_writers.log_state_covariance(t, runner.state.vio_frame, runner.kf, bg, ba)
            
        # Cleanup expired VPS clones (every ~0.1s, i.e. 40 samples)
        if i % 40 == 0 and hasattr(runner, 'vps_clone_manager') and runner.vps_clone_manager is not None:
             runner.vps_clone_manager.cleanup_expired_clones(t)
        
        # Magnetometer updates
        if runner.config.use_magnetometer:
            runner.magnetometer_service.process_magnetometer(t)
        
        # Capture current MSL/AGL BEFORE DEM update (matches vio_vps.py behavior)
        lat_now, lon_now = runner.proj_cache.xy_to_latlon(
            runner.kf.x[0, 0], runner.kf.x[1, 0],
            runner.lat0, runner.lon0
        )
        dem_now = runner.dem.sample_m(lat_now, lon_now) if runner.dem.ds else 0.0
        if dem_now is None:
            dem_now = 0.0
        msl_now = runner.kf.x[2, 0]
        agl_now = msl_now - dem_now
        
        # DEM height updates (modifies kf.x[2,0], but we use pre-update values for logging)
        runner.dem_service.process_dem_height(t)
        
        # TRN (Terrain Referenced Navigation) updates - v3.3.0
        if runner.trn is not None:
            # Update altitude history for profile matching
            runner.trn.update_altitude_history(
                timestamp=t,
                msl_altitude=msl_now,
                estimated_x=runner.kf.x[0, 0],
                estimated_y=runner.kf.x[1, 0]
            )
            
            # Try TRN position fix
            if runner.trn.check_update_needed(t):
                trn_success = runner.trn.apply_trn_update(
                    kf=runner.kf,
                    lat0=runner.lat0,
                    lon0=runner.lon0,
                    current_time=t
                )

        # Position-velocity consistency guard (state-aware kinematic sanity).
        runner.kinematic_guard_service.apply(t)

        # Async backend correction polling + blended apply (non-blocking frontend).
        if runner.backend_optimizer is not None:
            cam_tick = int(getattr(runner.state, "vio_frame", 0)) > int(vio_frame_before)
            if _backend_poll_due(
                runner.global_config,
                t_now=float(t),
                last_poll_t=float(getattr(runner, "_backend_last_poll_t", -1e9)),
                cam_tick=bool(cam_tick),
            ):
                runner._backend_last_poll_t = float(t)
                runner._backend_poll_count = int(getattr(runner, "_backend_poll_count", 0)) + 1
                try:
                    b_stats = getattr(runner.backend_optimizer, "stats", {})
                    runner._backend_emit_count = int(b_stats.get("corrections_emitted", runner._backend_emit_count))
                    runner._backend_emit_stale_drop_count = int(
                        b_stats.get("corrections_emit_stale_dropped", getattr(runner, "_backend_emit_stale_drop_count", 0))
                    )
                    runner._backend_overwrite_count = int(
                        b_stats.get("corrections_overwritten", getattr(runner, "_backend_overwrite_count", 0))
                    )
                except Exception:
                    pass
                corr = runner.backend_optimizer.poll_correction(t_now=float(t))
                stale_drops = int(getattr(runner.backend_optimizer, "last_poll_stale_drops", 0))
                if stale_drops > 0:
                    runner._backend_stale_drop_count = int(getattr(runner, "_backend_stale_drop_count", 0)) + stale_drops
                if corr is not None:
                    max_age = float(runner.global_config.get("BACKEND_MAX_CORRECTION_AGE_SEC", 2.0))
                    if float(getattr(corr, "age_sec", 0.0)) > max_age:
                        runner._backend_stale_drop_count = int(getattr(runner, "_backend_stale_drop_count", 0)) + 1
                        counts = getattr(runner, "_backend_reject_reason_counts", None)
                        if isinstance(counts, dict):
                            counts["stale_reject"] = int(counts.get("stale_reject", 0)) + 1
                    else:
                        runner.vio_service.schedule_backend_correction(corr)
            # Deterministic blend apply cadence:
            # - poll is gated by camera/fallback timing
            # - apply of already scheduled correction runs every IMU tick
            runner.vio_service.apply_pending_backend_blend(float(t))

        # Runtime memory watchdog (prevents long-run RAM ballooning).
        _memory_watchdog_tick(runner, float(t))

        # Log error vs ground truth (every sample, like vio_vps.py)
        runner.output_reporting.log_error(t)
        
        # Log pose (using pre-DEM-update msl_now/agl_now for consistency with vio_vps.py)
        runner.output_reporting.log_pose(
            t,
            dt,
            used_vo,
            vo_data,
            msl_now=msl_now,
            agl_now=agl_now,
            lat_now=lat_now,
            lon_now=lon_now,
        )
        
        # Log inference timing (every sample, like vio_vps.py).
        # Use long-lived file handle to reduce per-sample I/O overhead.
        toc_iter = time.time()
        dt_proc = toc_iter - tic_iter
        fps = (1.0 / dt_proc) if dt_proc > 0 else 0.0
        if runner._inf_fh is not None:
            runner._inf_fh.write(f"{i},{dt_proc:.6f},{fps:.2f}\n")
            runner._inf_since_flush = int(getattr(runner, "_inf_since_flush", 0)) + 1
            flush_stride = int(max(1, getattr(runner, "_inf_flush_stride", 200)))
            if runner._inf_since_flush >= flush_stride:
                try:
                    runner._inf_fh.flush()
                except Exception:
                    pass
                runner._inf_since_flush = 0
        else:
            with open(runner.inf_csv, "a", newline="") as f:
                f.write(f"{i},{dt_proc:.6f},{fps:.2f}\n")
        
        # Heavy state debug is optional (enabled by --save_debug_data).
        if runner.state_dbg_csv:
            log_state_debug(runner.state_dbg_csv, t, runner.kf, dem_now, agl_now, msl_now, last_a_world)
        
        # Update last_a_world for next iteration
        # Get a_world from last IMU sample processing
        try:
            # Convert quaternion to rotation matrix (state is [w,x,y,z])
            q = runner.kf.x[6:10, 0].flatten()
            quat_xyzw = np.array([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w]
            r_body_to_world = R_scipy.from_quat(quat_xyzw).as_matrix()
            
            # Compute world-frame kinematic acceleration for debugging.
            a_body = rec.lin - runner.kf.x[13:16, 0].flatten()  # bias-corrected
            g_norm = runner.global_config.get('IMU_PARAMS', {}).get('g_norm', 9.8066)
            g_world = np.array([0.0, 0.0, -g_norm])  # ENU: gravity points down
            accel_includes_gravity = bool(imu_params_step.get('accel_includes_gravity', True))
            a_world_raw = r_body_to_world @ a_body
            if accel_includes_gravity:
                # Remove gravity from gravity-included acceleration.
                last_a_world = a_world_raw - g_world
            else:
                # Convert specific force to kinematic acceleration.
                last_a_world = a_world_raw + g_world
        except Exception:
            pass
        
        # Progress
        if i % 1000 == 0:
            speed_ms = float(np.linalg.norm(runner.kf.x[3:6, 0]))
            print(f"t={time_elapsed:8.3f}s | speed={speed_ms*3.6:5.1f}km/h | "
                    f"phase={runner.PHASE_NAMES[runner.state.current_phase]}", end="\r")
        if np.isfinite(stop_time_sim) and float(t) >= float(stop_time_sim):
            print(f"\n[STOP] VIO_STOP_TIME reached at t={float(t):.6f}s")
            break
    
    toc_all = time.time()
    
    duration_sec = toc_all - tic_all
    print(f"\n=== Finished in {duration_sec:.2f} seconds ===")
    return duration_sec
