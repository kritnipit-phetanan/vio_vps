"""Vision/MSCKF/VPS (threaded) service for VIORunner."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from ..config import CAMERA_VIEW_CONFIGS
from ..loop_closure import check_loop_closure, apply_loop_closure_correction
from ..math_utils import quaternion_to_yaw
from ..measurement_updates import apply_vio_velocity_update
from ..msckf import trigger_msckf_update, find_mature_features_for_msckf
from ..output_utils import (
    get_ground_truth_error,
    log_fej_consistency,
    log_msckf_window,
    log_vo_debug,
    save_keyframe_with_overlay,
)
from ..propagation import apply_preintegration_at_camera, clone_camera_for_msckf
from ..state_manager import imu_to_gnss_position


class VIOService:
    """Encapsulates camera/VIO frontend, MSCKF, and VPS-thread interaction."""

    def __init__(self, runner: Any):
        self.runner = runner

    def _compute_vps_quality_metrics(self, img: np.ndarray) -> Dict[str, float]:
        """Compute lightweight image-quality metrics for VPS call-rate adaptation."""
        if img is None or getattr(img, "size", 0) == 0:
            return {"lap_var": 0.0, "contrast_std": 0.0, "sat_ratio": 1.0}
        img_u8 = img.astype(np.uint8, copy=False)
        lap_var = float(cv2.Laplacian(img_u8, cv2.CV_64F).var())
        contrast_std = float(np.std(img_u8))
        sat_ratio = float(np.mean((img_u8 <= 4) | (img_u8 >= 251)))
        return {
            "lap_var": lap_var,
            "contrast_std": contrast_std,
            "sat_ratio": sat_ratio,
        }

    def _update_visual_heading_reference(self,
                                         t_cam: float,
                                         yaw_delta_deg: float,
                                         num_inliers: int,
                                         avg_flow_px: float,
                                         accepted: bool):
        """
        Maintain a weak visual-heading reference from VO rotation increments.

        This is used by MAG service as a consistency aid in full-sensor mode.
        """
        runner = self.runner
        decay = float(runner.global_config.get("VISION_HEADING_QUALITY_DECAY", 0.92))
        decay = min(0.999, max(0.0, decay))
        if not accepted:
            runner._vision_heading_quality *= decay
            return

        if not np.isfinite(yaw_delta_deg):
            runner._vision_heading_quality *= decay
            return

        max_delta_deg = float(runner.global_config.get("VISION_HEADING_MAX_DELTA_DEG", 20.0))
        min_inliers = int(runner.global_config.get("VISION_HEADING_MIN_INLIERS", 25))
        min_parallax = float(runner.global_config.get("VISION_HEADING_MIN_PARALLAX_PX", 1.0))
        if abs(float(yaw_delta_deg)) > max_delta_deg or int(num_inliers) < min_inliers or float(avg_flow_px) < min_parallax:
            runner._vision_heading_quality *= decay
            return

        if runner._vision_yaw_ref is None:
            runner._vision_yaw_ref = float(quaternion_to_yaw(runner.kf.x[6:10, 0]))
        yaw_next = float(runner._vision_yaw_ref) + float(np.deg2rad(yaw_delta_deg))
        yaw_next = float(np.arctan2(np.sin(yaw_next), np.cos(yaw_next)))
        yaw_state = float(quaternion_to_yaw(runner.kf.x[6:10, 0]))
        runner._vision_yaw_last_t = float(t_cam)

        q_inlier = min(1.0, float(num_inliers) / 120.0)
        q_parallax = min(1.0, float(avg_flow_px) / 10.0)
        q_now = q_inlier * q_parallax
        yaw_delta_state_deg = abs(np.degrees(np.arctan2(np.sin(yaw_next - yaw_state), np.cos(yaw_next - yaw_state))))
        if yaw_delta_state_deg > 45.0:
            q_now *= 0.6
        anchor_alpha = float(np.clip(0.08 + 0.30 * q_now, 0.08, 0.38))
        yaw_blend = (1.0 - anchor_alpha) * yaw_next + anchor_alpha * yaw_state
        runner._vision_yaw_ref = float(np.arctan2(np.sin(yaw_blend), np.cos(yaw_blend)))
        runner._vision_heading_quality = float(np.clip(0.80 * runner._vision_heading_quality + 0.20 * q_now, 0.0, 1.0))

    def _should_run_vps(self, t_cam: float, img: np.ndarray) -> Tuple[bool, str]:
        """
        Decide whether to invoke VPS matcher on this frame.

        Adaptive factors:
        - flight phase
        - current image quality
        - accumulated VPS matcher failure rate
        """
        runner = self.runner
        if runner.vps_runner is None:
            return False, "disabled"

        phase = int(getattr(runner.state, "current_phase", 2))
        objective_mode = str(runner.global_config.get("OBJECTIVE_MODE", "stability")).lower()
        vps_accuracy_mode = bool(runner.global_config.get("VPS_ACCURACY_MODE", False))
        if objective_mode == "accuracy" or vps_accuracy_mode:
            phase_mult = {0: 1.8, 1: 1.2, 2: 0.85}.get(phase, 1.0)
        else:
            phase_mult = {0: 2.5, 1: 1.6, 2: 1.0}.get(phase, 1.0)
        base_interval = float(getattr(runner.vps_runner.config, "min_update_interval", 0.5))

        quality = self._compute_vps_quality_metrics(img)
        lap_var = float(quality["lap_var"])
        contrast_std = float(quality["contrast_std"])
        sat_ratio = float(quality["sat_ratio"])

        min_lap = float(runner.global_config.get("VPS_QUALITY_MIN_LAPLACIAN_VAR", 18.0))
        min_contrast = float(runner.global_config.get("VPS_QUALITY_MIN_CONTRAST_STD", 16.0))
        max_sat_ratio = float(runner.global_config.get("VPS_QUALITY_MAX_SAT_RATIO", 0.40))

        hard_lap = float(runner.global_config.get("VPS_QUALITY_HARD_MIN_LAPLACIAN_VAR", max(4.0, 0.25 * min_lap)))
        hard_contrast = float(runner.global_config.get("VPS_QUALITY_HARD_MIN_CONTRAST_STD", max(3.0, 0.25 * min_contrast)))
        hard_sat_ratio = float(runner.global_config.get("VPS_QUALITY_HARD_MAX_SAT_RATIO", min(0.85, max_sat_ratio + 0.25)))

        if lap_var < hard_lap or contrast_std < hard_contrast or sat_ratio > hard_sat_ratio:
            runner._vps_skip_streak = int(getattr(runner, "_vps_skip_streak", 0)) + 1
            return False, "quality_hard_skip"

        quality_mult = 1.0
        if lap_var < min_lap:
            quality_mult *= 1.35
        if contrast_std < min_contrast:
            quality_mult *= 1.25
        if sat_ratio > max_sat_ratio:
            quality_mult *= 1.25

        stats = getattr(runner.vps_runner, "stats", {})
        total_attempts = max(1, int(stats.get("total_attempts", 0)))
        fail_match = int(stats.get("fail_match", 0))
        fail_quality = int(stats.get("fail_quality", 0))
        fail_rate = float(fail_match + fail_quality) / float(total_attempts)
        fail_mult = 1.0 + min(
            2.0,
            (1.25 * fail_rate) if (objective_mode == "accuracy" or vps_accuracy_mode) else (2.0 * fail_rate),
        )

        adaptive_interval = max(0.05, base_interval * phase_mult * quality_mult * fail_mult)
        next_allowed = float(getattr(runner, "_vps_next_allowed_t", 0.0))
        max_skip_streak = int(runner.global_config.get("VPS_ADAPTIVE_MAX_SKIP_STREAK", 20))

        if t_cam < next_allowed and int(getattr(runner, "_vps_skip_streak", 0)) < max_skip_streak:
            runner._vps_skip_streak = int(getattr(runner, "_vps_skip_streak", 0)) + 1
            return False, "interval_hold"

        runner._vps_last_attempt_t = float(t_cam)
        runner._vps_next_allowed_t = float(t_cam + adaptive_interval)
        runner._vps_skip_streak = 0
        return True, "run"

    def _evaluate_vps_failsoft_temporal_gate(self, vps_offset: np.ndarray, vps_offset_m: float) -> Tuple[bool, str]:
        """
        Evaluate fail-soft temporal consistency for VPS update apply path.

        Rules:
        - Reject if offset direction changes too sharply vs last accepted VPS.
        - For large offsets, require N consecutive consistent observations.
        """
        runner = self.runner
        if vps_offset is None or np.size(vps_offset) < 2:
            return True, ""
        cur = np.array(vps_offset[:2], dtype=float)
        cur_norm = float(np.linalg.norm(cur))
        if not np.isfinite(cur_norm) or cur_norm <= 1e-6:
            return True, ""

        max_dir_change_deg = float(
            runner.global_config.get("VPS_APPLY_FAILSOFT_MAX_DIR_CHANGE_DEG", 60.0)
        )
        last_vec = getattr(runner, "_vps_last_accepted_offset_vec", None)
        if isinstance(last_vec, np.ndarray) and last_vec.size >= 2:
            last = np.array(last_vec[:2], dtype=float)
            last_norm = float(np.linalg.norm(last))
            if np.isfinite(last_norm) and last_norm > 1e-6:
                cosang = float(np.clip(np.dot(cur, last) / (cur_norm * last_norm), -1.0, 1.0))
                dir_delta_deg = float(np.degrees(np.arccos(cosang)))
                if dir_delta_deg > max_dir_change_deg:
                    runner._vps_jump_reject_count = int(getattr(runner, "_vps_jump_reject_count", 0)) + 1
                    return False, f"SOFT_REJECT_DIR_CHANGE: dir={dir_delta_deg:.1f}deg>{max_dir_change_deg:.1f}deg"

        large_offset_m = float(
            runner.global_config.get("VPS_APPLY_FAILSOFT_LARGE_OFFSET_CONFIRM_M", 80.0)
        )
        required_hits = max(
            1,
            int(runner.global_config.get("VPS_APPLY_FAILSOFT_LARGE_OFFSET_CONFIRM_HITS", 2)),
        )
        if np.isfinite(vps_offset_m) and vps_offset_m > large_offset_m and required_hits > 1:
            pending_vec = getattr(runner, "_vps_pending_large_offset_vec", None)
            pending_hits = int(getattr(runner, "_vps_pending_large_offset_hits", 0))
            consistent = False
            if isinstance(pending_vec, np.ndarray) and pending_vec.size >= 2:
                prev = np.array(pending_vec[:2], dtype=float)
                prev_norm = float(np.linalg.norm(prev))
                if np.isfinite(prev_norm) and prev_norm > 1e-6:
                    cos_prev = float(np.clip(np.dot(cur, prev) / (cur_norm * prev_norm), -1.0, 1.0))
                    prev_delta_deg = float(np.degrees(np.arccos(cos_prev)))
                    consistent = prev_delta_deg <= max_dir_change_deg
            pending_hits = pending_hits + 1 if consistent else 1
            runner._vps_pending_large_offset_vec = cur.copy()
            runner._vps_pending_large_offset_hits = int(pending_hits)
            if pending_hits < required_hits:
                return False, (
                    f"SOFT_REJECT_LARGE_OFFSET_PENDING: hits={pending_hits}/{required_hits}, "
                    f"offset={vps_offset_m:.1f}m"
                )
            runner._vps_temporal_confirm_count = int(getattr(runner, "_vps_temporal_confirm_count", 0)) + 1
            runner._vps_pending_large_offset_vec = None
            runner._vps_pending_large_offset_hits = 0
        elif np.isfinite(vps_offset_m) and vps_offset_m <= large_offset_m:
            runner._vps_pending_large_offset_vec = None
            runner._vps_pending_large_offset_hits = 0

        return True, ""

    def _check_vps_hard_reject(self,
                               vps_offset: np.ndarray,
                               vps_offset_m: float) -> Tuple[bool, str]:
        """
        Hard safety gate for absolute correction before delayed-update apply.

        Rejects obviously unsafe updates:
        - huge offset magnitude
        - abrupt direction change vs previous accepted offset
        """
        runner = self.runner
        hard_max_offset_m = float(runner.global_config.get("VPS_ABS_HARD_REJECT_OFFSET_M", 180.0))
        if np.isfinite(vps_offset_m) and vps_offset_m > hard_max_offset_m:
            runner._vps_jump_reject_count = int(getattr(runner, "_vps_jump_reject_count", 0)) + 1
            return False, f"HARD_REJECT_OFFSET: {vps_offset_m:.1f}m>{hard_max_offset_m:.1f}m"

        hard_max_dir_change_deg = float(runner.global_config.get("VPS_ABS_HARD_REJECT_DIR_CHANGE_DEG", 75.0))
        cur = np.asarray(vps_offset[:2], dtype=float).reshape(-1)
        if cur.size < 2 or not np.all(np.isfinite(cur[:2])):
            return True, ""
        cur_norm = float(np.linalg.norm(cur[:2]))
        if cur_norm <= 1e-6:
            return True, ""

        last_vec = getattr(runner, "_vps_last_accepted_offset_vec", None)
        if isinstance(last_vec, np.ndarray) and last_vec.size >= 2:
            prev = np.asarray(last_vec[:2], dtype=float).reshape(-1)
            prev_norm = float(np.linalg.norm(prev[:2]))
            if np.isfinite(prev_norm) and prev_norm > 1e-6:
                cosang = float(np.clip(np.dot(cur[:2], prev[:2]) / (cur_norm * prev_norm), -1.0, 1.0))
                dir_delta_deg = float(np.degrees(np.arccos(cosang)))
                if dir_delta_deg > hard_max_dir_change_deg:
                    runner._vps_jump_reject_count = int(getattr(runner, "_vps_jump_reject_count", 0)) + 1
                    return False, (
                        f"HARD_REJECT_DIR_CHANGE: {dir_delta_deg:.1f}deg>"
                        f"{hard_max_dir_change_deg:.1f}deg"
                    )
        return True, ""

    def _clamp_vps_latlon(self,
                          current_xy: np.ndarray,
                          vps_lat: float,
                          vps_lon: float) -> Tuple[float, float, float]:
        """
        Clamp one-shot VPS correction magnitude in XY before delayed-update apply.
        """
        runner = self.runner
        max_apply_dp_xy = float(runner.global_config.get("VPS_ABS_MAX_APPLY_DP_XY_M", 25.0))
        vps_xy = np.array(
            runner.proj_cache.latlon_to_xy(float(vps_lat), float(vps_lon), runner.lat0, runner.lon0),
            dtype=float,
        ).reshape(2,)
        cur_xy = np.asarray(current_xy, dtype=float).reshape(2,)
        delta = vps_xy - cur_xy
        delta_norm = float(np.linalg.norm(delta))
        if not np.isfinite(delta_norm) or delta_norm <= max_apply_dp_xy or delta_norm <= 1e-9:
            return float(vps_lat), float(vps_lon), 1.0

        scale = float(max_apply_dp_xy / max(delta_norm, 1e-9))
        vps_xy_clamped = cur_xy + delta * scale
        lat_c, lon_c = runner.proj_cache.xy_to_latlon(vps_xy_clamped[0], vps_xy_clamped[1], runner.lat0, runner.lon0)
        return float(lat_c), float(lon_c), scale

    def process_vio(self, rec, t: float, ongoing_preint=None) -> Tuple[bool, Optional[Dict[str, float]]]:
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
        runner = self.runner

        # Get configuration
        kb_params = runner.global_config.get("KB_PARAMS", {"mu": 600, "mv": 600})
        min_parallax = runner.global_config.get("MIN_PARALLAX_PX", 2.0)
        imu_params = runner.global_config.get("IMU_PARAMS", {})

        used_vo = False
        vo_data = None
        vo_dx = vo_dy = vo_dz = vo_r = vo_p = vo_y = np.nan

        if runner.vio_fe is None:
            return used_vo, vo_data

        # Check for fast rotation - skip VIO during aggressive maneuvers
        rotation_rate_deg_s = np.linalg.norm(rec.ang) * 180.0 / np.pi
        is_fast_rotation = rotation_rate_deg_s > 30.0

        def _collect_finished_vps_result():
            """
            Collect async VPS result if the single in-flight worker has finished.

            Returns:
                (result, meta) where meta contains clone_id/frame/time info, or (None, None).
            """
            th = getattr(runner, "_vps_inflight_thread", None)
            if th is None:
                return None, None
            if th.is_alive():
                return None, None
            result = getattr(runner, "_vps_inflight_result", None)
            meta = getattr(runner, "_vps_inflight_meta", None)
            runner._vps_inflight_thread = None
            runner._vps_inflight_result = None
            runner._vps_inflight_meta = None
            return result, (meta if isinstance(meta, dict) else {})

        def _get_policy(sensor: str, ts: float):
            decision = None
            scales = {
                "r_scale": 1.0,
                "chi2_scale": 1.0,
                "threshold_scale": 1.0,
                "reproj_scale": 1.0,
            }
            policy_runtime = getattr(runner, "policy_runtime_service", None)
            if policy_runtime is not None:
                try:
                    decision = policy_runtime.get_sensor_decision(str(sensor), float(ts))
                    scales = {
                        "r_scale": float(decision.r_scale),
                        "chi2_scale": float(decision.chi2_scale),
                        "threshold_scale": float(decision.threshold_scale),
                        "reproj_scale": float(decision.reproj_scale),
                    }
                    return decision, scales
                except Exception:
                    decision = None
            # Backward-compatible fallback.
            try:
                _, apply_scales = runner.adaptive_service.get_sensor_adaptive_scales(str(sensor))
                for k in scales.keys():
                    if k in apply_scales:
                        scales[k] = float(apply_scales[k])
            except Exception:
                pass
            return decision, scales

        # Process images up to current time
        while runner.state.img_idx < len(runner.imgs) and runner.imgs[runner.state.img_idx].t <= t:
            # Get camera timestamp (CRITICAL: use this instead of IMU time t)
            # Prevents timestamp mismatch when cloning/resetting preintegration
            t_cam = runner.imgs[runner.state.img_idx].t
            cam_lag = abs(float(t) - float(t_cam))
            cam_sync_threshold = float(runner.global_config.get("CAM_TIME_SYNC_THRESHOLD_SEC", 0.10))
            runner.output_reporting.log_convention_check(
                t=float(t_cam),
                sensor="CAM",
                check="imu_cam_abs_dt",
                value=float(cam_lag),
                threshold=float(cam_sync_threshold),
                status="PASS" if cam_lag <= cam_sync_threshold else "WARN",
                note=f"t_imu={float(t):.6f}",
            )

            # FRAME SKIP (v2.9.9): Process every N frames for speedup
            # frame_skip=1 -> all frames, frame_skip=2 -> every other frame (50% faster)
            if runner.config.frame_skip > 1 and (runner.state.img_idx % runner.config.frame_skip) != 0:
                runner.state.img_idx += 1
                continue

            img_path = runner.imgs[runner.state.img_idx].path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                runner.state.img_idx += 1
                continue

            # Resize if needed
            if (img.shape[1], img.shape[0]) != tuple(runner.config.downscale_size):
                img = cv2.resize(img, runner.config.downscale_size, interpolation=cv2.INTER_AREA)

            # Apply fisheye rectification if enabled
            img_for_tracking = img
            if runner.rectifier is not None:
                img_for_tracking = runner.rectifier.rectify(img)

            # ================================================================
            # VPS Real-time Processing (Parallel with Threading)
            # ================================================================
            # VPS runs in background thread while VIO continues immediately.
            # Single-flight guard: never allow more than one in-flight VPS worker.
            # Without this, slow VPS frames can pile up threads and explode memory.
            vps_result, vps_result_meta = _collect_finished_vps_result()

            if runner.vps_runner is not None:
                inflight = getattr(runner, "_vps_inflight_thread", None)
                worker_busy = bool(inflight is not None and inflight.is_alive())
                busy_force_streak = int(
                    runner.global_config.get("VPS_WORKER_BUSY_FORCE_LOCAL_STREAK", 120)
                )
                busy_force_sec = float(
                    runner.global_config.get("VPS_WORKER_BUSY_FORCE_LOCAL_SEC", 8.0)
                )
                if is_fast_rotation:
                    should_run_vps, skip_reason = False, "fast_rotation"
                elif worker_busy:
                    should_run_vps, skip_reason = False, "thread_busy"
                else:
                    should_run_vps, skip_reason = self._should_run_vps(t_cam=t_cam, img=img)
                if not should_run_vps:
                    if skip_reason == "thread_busy":
                        runner._vps_thread_busy_skip_count = int(
                            getattr(runner, "_vps_thread_busy_skip_count", 0)
                        ) + 1
                        runner._vps_thread_busy_streak = int(
                            getattr(runner, "_vps_thread_busy_streak", 0)
                        ) + 1
                        if busy_force_streak > 0 and runner._vps_thread_busy_streak >= busy_force_streak:
                            prev_until = float(getattr(runner, "_vps_force_local_until_t", -1e9))
                            new_until = float(t_cam + max(0.0, busy_force_sec))
                            runner._vps_force_local_until_t = max(prev_until, new_until)
                            if prev_until < float(t_cam):
                                print(
                                    f"[VPS] busy guard: force_local enabled for {busy_force_sec:.1f}s "
                                    f"(streak={runner._vps_thread_busy_streak})"
                                )
                        if (runner._vps_thread_busy_skip_count % 100) == 0:
                            print(
                                f"[VPS] worker busy; skipped {runner._vps_thread_busy_skip_count} frame(s) "
                                "while previous VPS attempt is still running"
                            )
                    else:
                        runner._vps_thread_busy_streak = 0
                    if runner.vps_logger is not None and skip_reason != "interval_hold":
                        try:
                            runner.vps_logger.log_attempt(
                                t=float(t_cam),
                                frame=int(runner.state.img_idx),
                                est_lat=float(runner.lat0),
                                est_lon=float(runner.lon0),
                                est_alt=float(runner.kf.x[2, 0]),
                                est_yaw_deg=float(np.degrees(quaternion_to_yaw(runner.kf.x[6:10, 0]))),
                                success=False,
                                reason=f"adaptive_skip_{skip_reason}",
                                processing_time_ms=0.0,
                            )
                        except Exception:
                            pass
                else:
                    runner._vps_thread_busy_streak = 0
                    runner._vps_attempt_count = int(getattr(runner, "_vps_attempt_count", 0)) + 1
                    # Get current EKF estimates for VPS processing
                    # Extract IMU position from EKF state
                    p_imu_enu = runner.kf.x[:3].flatten()  # Position in ENU
                    v_imu_enu = runner.kf.x[3:6].flatten()  # Velocity in ENU

                    # Extract rotation matrix from quaternion
                    q = runner.kf.x[6:10].flatten()  # Quaternion [w, x, y, z]
                    from ..math_utils import quat_to_rot

                    r_body_to_world = quat_to_rot(q)

                    # Get GNSS position (apply lever arm)
                    p_gnss_enu = imu_to_gnss_position(
                        p_imu_enu, r_body_to_world, runner.lever_arm
                    )

                    # Convert ENU to lat/lon
                    lat, lon = runner.proj_cache.xy_to_latlon(
                        p_gnss_enu[0], p_gnss_enu[1], runner.lat0, runner.lon0
                    )
                    alt = p_gnss_enu[2]  # MSL altitude
                    state_speed_m_s = float(np.linalg.norm(v_imu_enu))

                    # Get terrain height (DEM) to compute AGL for VPS filtering
                    dem_height = runner.dem.sample_m(lat, lon)
                    if dem_height is None:
                        dem_height = 0.0
                    agl = alt - dem_height

                    q_wxyz = runner.kf.x[6:10, 0].astype(float)
                    est_yaw = float(quaternion_to_yaw(q_wxyz))
                    frame_idx_for_vps = int(runner.state.img_idx)
                    force_local_busy_guard = bool(
                        float(t_cam) < float(getattr(runner, "_vps_force_local_until_t", -1e9))
                    )
                    if force_local_busy_guard and (runner._vps_attempt_count % 20) == 0:
                        rem = float(getattr(runner, "_vps_force_local_until_t", -1e9)) - float(t_cam)
                        print(f"[VPS] force_local active (remaining={max(0.0, rem):.1f}s)")

                    # Define VPS processing function for thread
                    def run_vps_in_thread():
                        try:
                            est_cov_xy = np.array(runner.kf.P[0:2, 0:2], dtype=float)
                            result = runner.vps_runner.process_frame(
                                img=img,  # Use original image (not rectified)
                                t_cam=t_cam,
                                est_lat=lat,
                                est_lon=lon,
                                est_yaw=est_yaw,
                                est_alt=agl,  # Send AGL for 30m min_altitude check
                                frame_idx=frame_idx_for_vps,
                                est_cov_xy=est_cov_xy,
                                phase=int(getattr(runner.state, "current_phase", 2)),
                                state_speed_m_s=state_speed_m_s,
                                force_local=force_local_busy_guard,
                                objective=str(runner.global_config.get("OBJECTIVE_MODE", "stability")),
                            )
                            runner._vps_inflight_result = result
                        except Exception as e:
                            print(f"[VPS] Thread error: {e}")
                            runner._vps_inflight_result = None

                    clone_id = f"vps_{frame_idx_for_vps}"
                    # Clone EKF state for delayed update (stochastic cloning)
                    if hasattr(runner, "vps_clone_manager"):
                        runner.vps_clone_manager.clone_state(runner.kf, t_cam, clone_id)

                    runner._vps_inflight_result = None
                    runner._vps_inflight_meta = {
                        "clone_id": clone_id,
                        "frame_idx": frame_idx_for_vps,
                        "t_cam": float(t_cam),
                    }
                    runner._vps_inflight_thread = threading.Thread(
                        target=run_vps_in_thread, daemon=True
                    )
                    runner._vps_inflight_thread.start()

            # ================================================================
            # VIO Processing (continues immediately, parallel with VPS!)
            # ================================================================

            if is_fast_rotation:
                print(f"[VIO] SKIPPING due to fast rotation: {rotation_rate_deg_s:.1f} deg/s")
                runner.state.img_idx += 1
                continue

            # Run VIO frontend
            ok, ninl, r_vo_mat, t_unit, dt_img = runner.vio_fe.step(img_for_tracking, t_cam)

            # Loop closure detection - check when we have sufficient position estimate
            current_phase = int(getattr(runner.state, "current_phase", 2))
            current_health = str(
                getattr(getattr(runner, "current_adaptive_decision", None), "health_state", "HEALTHY")
            )
            match_result = check_loop_closure(
                loop_detector=runner.loop_detector,
                img_gray=img,
                t=t_cam,
                kf=runner.kf,
                global_config=runner.global_config,
                vio_fe=runner.vio_fe,
                phase=current_phase,
                health_state=current_health,
            )
            if match_result is not None:
                loop_decision, _ = _get_policy("LOOP_CLOSURE", t_cam)
                loop_applied = apply_loop_closure_correction(
                    kf=runner.kf,
                    loop_info=match_result,
                    t=t_cam,
                    cam_states=runner.state.cam_states,
                    loop_detector=runner.loop_detector,
                    global_config=runner.global_config,
                    policy_decision=loop_decision,
                )
                if (
                    loop_decision is not None
                    and str(getattr(loop_decision, "mode", "APPLY")).upper() in ("HOLD", "SKIP")
                    and bool(loop_applied)
                    and getattr(runner, "policy_runtime_service", None) is not None
                ):
                    runner.policy_runtime_service.record_conflict(
                        sensor="LOOP_CLOSURE",
                        t=float(t_cam),
                        expected_mode=str(loop_decision.mode),
                        actual_mode="APPLY",
                        note="applied_despite_hold_skip",
                    )

            # Compute average optical flow (parallax) - use the new attributes
            # mean_parallax is now computed in step() EVERY frame, even if ok=False
            avg_flow_px = runner.vio_fe.mean_parallax

            # Fallback to last_matches only if mean_parallax is still ~0 and matches exist
            if avg_flow_px < 0.01 and runner.vio_fe.last_matches is not None:
                focal_px = kb_params.get("mu", 600)
                pts_prev, pts_cur = runner.vio_fe.last_matches
                if len(pts_prev) > 0:
                    pts_prev_px = pts_prev * focal_px + np.array([[runner.vio_fe.img_w / 2, runner.vio_fe.img_h / 2]])
                    pts_cur_px = pts_cur * focal_px + np.array([[runner.vio_fe.img_w / 2, runner.vio_fe.img_h / 2]])
                    flows = pts_cur_px - pts_prev_px
                    avg_flow_px = float(np.median(np.linalg.norm(flows, axis=1)))

            # Debug: log feature statistics - use actual tracked features from VIOFrontEnd
            num_features = runner.vio_fe.last_num_tracked
            num_inliers = runner.vio_fe.last_num_inliers
            runner._cam_frames_processed += 1
            if int(num_inliers) > 0:
                runner._cam_frames_inlier_nonzero += 1
            tracking_ratio = 1.0 if num_features > 0 else 0.0
            inlier_ratio = num_inliers / max(1, num_features)
            runner.debug_writers.log_feature_stats(
                runner.vio_fe.frame_idx, t_cam, num_features, num_features, num_inliers,
                avg_flow_px, avg_flow_px, tracking_ratio, inlier_ratio
            )

            # Apply preintegration at EVERY camera frame
            # Store Jacobians for bias observability in MSCKF
            preint_jacobians = None
            if ongoing_preint is not None:
                preint_jacobians = apply_preintegration_at_camera(runner.kf, ongoing_preint, t_cam, imu_params)

            # Check parallax for different purposes
            # CRITICAL CHANGE: Separate low-parallax handling for velocity vs MSCKF/plane
            # - Low parallax (<2px): Skip velocity update, but ALLOW cloning/MSCKF
            # - This lets plane-aided MSCKF help even in nadir scenarios
            is_insufficient_parallax_for_velocity = avg_flow_px < min_parallax
            current_tracks = runner.vio_fe.get_tracks_for_frame(runner.vio_fe.frame_idx)
            current_track_count = len(current_tracks)

            # Camera cloning for MSCKF (lower threshold than velocity)
            # Allow cloning even with low parallax to enable plane-aided MSCKF
            clone_threshold = min_parallax * 0.5  # Much lower: 1px instead of 4px
            should_clone = (
                avg_flow_px >= clone_threshold
                and not is_fast_rotation
                and current_track_count > 0
            )

            if should_clone:
                clone_idx = clone_camera_for_msckf(
                    kf=runner.kf,
                    t=t_cam,  # Use camera time for accurate cloning
                    cam_states=runner.state.cam_states,
                    cam_observations=runner.state.cam_observations,
                    vio_fe=runner.vio_fe,
                    frame_idx=runner.vio_fe.frame_idx,
                    preint_jacobians=preint_jacobians,  # Pass Jacobians for bias observability
                    max_clone_size=runner.global_config.get("MSCKF_MAX_CLONE_SIZE", 11),
                )

                # Log MSCKF window state
                if clone_idx >= 0:
                    num_tracked = len(runner.state.cam_observations[-1]["observations"]) if runner.state.cam_observations else 0
                    try:
                        num_mature = len(
                            find_mature_features_for_msckf(
                                runner.vio_fe,
                                runner.state.cam_observations,
                                min_observations=2,
                            )
                        )
                    except Exception:
                        num_mature = 0
                    window_start = runner.state.cam_states[0]["t"] if runner.state.cam_states else t_cam
                    log_msckf_window(
                        msckf_window_csv=runner.msckf_window_csv,
                        frame=runner.vio_fe.frame_idx,
                        t=t_cam,
                        num_clones=len(runner.state.cam_states),
                        num_tracked=num_tracked,
                        num_mature=num_mature,
                        window_start=window_start,
                        marginalized_clone=-1,
                    )

                    # Trigger MSCKF update if enough clones
                    if len(runner.state.cam_states) >= 3:
                        msckf_decision, msckf_policy_scales = _get_policy("MSCKF", t_cam)
                        msckf_adaptive_info: Dict[str, Any] = {}
                        phase_key = str(int(getattr(runner.state, "current_phase", 2)))
                        phase_chi2 = float(
                            runner.global_config.get("MSCKF_PHASE_CHI2_SCALE", {}).get(phase_key, 1.0)
                        )
                        phase_reproj = float(
                            runner.global_config.get("MSCKF_PHASE_REPROJ_SCALE", {}).get(phase_key, 1.0)
                        )
                        num_updates = trigger_msckf_update(
                            kf=runner.kf,
                            cam_states=runner.state.cam_states,
                            cam_observations=runner.state.cam_observations,
                            vio_fe=runner.vio_fe,
                            t=t_cam,
                            msckf_dbg_csv=runner.msckf_dbg_csv if hasattr(runner, "msckf_dbg_csv") else None,
                            dem_reader=runner.dem,
                            origin_lat=runner.lat0,
                            origin_lon=runner.lon0,
                            plane_detector=runner.plane_detector,
                            plane_config=runner.global_config if runner.plane_detector else None,
                            global_config=runner.global_config,
                            chi2_scale=float(msckf_policy_scales.get("chi2_scale", 1.0)) * phase_chi2,
                            reproj_scale=float(msckf_policy_scales.get("reproj_scale", 1.0)) * phase_reproj,
                            phase=int(getattr(runner.state, "current_phase", 2)),
                            health_state=str(
                                getattr(getattr(runner, "current_adaptive_decision", None), "health_state", "HEALTHY")
                            ),
                            adaptive_info=msckf_adaptive_info,
                            policy_decision=msckf_decision,
                        )
                        if (
                            msckf_decision is not None
                            and str(getattr(msckf_decision, "mode", "APPLY")).upper() in ("HOLD", "SKIP")
                            and int(num_updates) > 0
                            and getattr(runner, "policy_runtime_service", None) is not None
                        ):
                            runner.policy_runtime_service.record_conflict(
                                sensor="MSCKF",
                                t=float(t_cam),
                                expected_mode=str(msckf_decision.mode),
                                actual_mode="APPLY",
                                note=f"num_updates={int(num_updates)}",
                            )
                        runner.adaptive_service.record_adaptive_measurement(
                            "MSCKF",
                            adaptive_info=msckf_adaptive_info,
                            timestamp=t_cam,
                            policy_scales=msckf_policy_scales,
                        )

                        # Log FEJ consistency after MSCKF update
                        if num_updates > 0 and runner.config.save_debug_data:
                            log_fej_consistency(
                                fej_csv=runner.fej_csv,
                                t=t_cam,
                                frame=runner.vio_fe.frame_idx if runner.vio_fe else 0,
                                cam_states=runner.state.cam_states,
                                kf=runner.kf,
                            )

            # Optical Flow Velocity Update - run EVERY camera frame as XY drift reduction fallback
            # This is independent of VO (Essential matrix) success and works even with low parallax.
            is_dt_valid = np.isfinite(float(dt_img)) and (1e-4 < float(dt_img) < 0.25)
            if (
                runner.config.use_vio_velocity
                and avg_flow_px > 0.5
                and current_track_count > 0
                and is_dt_valid
            ):
                vel_error, vel_cov = get_ground_truth_error(
                    t_cam, runner.kf, runner.ppk_trajectory, runner.lat0, runner.lon0, runner.proj_cache, "velocity"
                )
                vio_decision, vio_policy_scales = _get_policy("VIO_VEL", t_cam)
                vio_adaptive_info: Dict[str, Any] = {}
                runner._vio_vel_attempt_count += 1
                vio_vel_accepted = apply_vio_velocity_update(
                    kf=runner.kf,
                    r_vo_mat=r_vo_mat if ok else None,
                    t_unit=t_unit if ok else None,
                    t=t_cam,
                    dt_img=dt_img,
                    avg_flow_px=avg_flow_px,
                    imu_rec=rec,
                    global_config=runner.global_config,
                    camera_view=runner.config.camera_view,
                    dem_reader=runner.dem,
                    lat0=runner.lat0,
                    lon0=runner.lon0,
                    use_vio_velocity=True,
                    proj_cache=runner.proj_cache,
                    save_debug=runner.config.save_debug_data,
                    residual_csv=runner.residual_csv if runner.config.save_debug_data else None,
                    vio_frame=runner.state.vio_frame,
                    vio_fe=runner.vio_fe,
                    state_error=vel_error,
                    state_cov=vel_cov,
                    chi2_scale=float(vio_policy_scales.get("chi2_scale", 1.0)),
                    r_scale_extra=float(vio_policy_scales.get("r_scale", 1.0)),
                    soft_fail_enable=bool(runner.global_config.get("VIO_VEL_SOFT_FAIL_ENABLE", True)),
                    soft_fail_r_cap=float(runner.global_config.get("VIO_VEL_SOFT_FAIL_MAX_R_MULT", 8.0)),
                    soft_fail_hard_reject_factor=float(runner.global_config.get("VIO_VEL_SOFT_FAIL_HARD_REJECT_FACTOR", 3.0)),
                    soft_fail_power=float(runner.global_config.get("VIO_VEL_SOFT_FAIL_POWER", 1.0)),
                    phase=int(getattr(runner.state, "current_phase", 2)),
                    health_state=str(
                        getattr(getattr(runner, "current_adaptive_decision", None), "health_state", "HEALTHY")
                    ),
                    adaptive_info=vio_adaptive_info,
                    policy_decision=vio_decision,
                )
                if bool(vio_vel_accepted):
                    runner._vio_vel_accept_count += 1
                if (
                    vio_decision is not None
                    and str(getattr(vio_decision, "mode", "APPLY")).upper() in ("HOLD", "SKIP")
                    and bool(vio_vel_accepted)
                    and getattr(runner, "policy_runtime_service", None) is not None
                ):
                    runner.policy_runtime_service.record_conflict(
                        sensor="VIO_VEL",
                        t=float(t_cam),
                        expected_mode=str(vio_decision.mode),
                        actual_mode="APPLY",
                        note="accepted_despite_hold_skip",
                    )
                runner.adaptive_service.record_adaptive_measurement(
                    "VIO_VEL",
                    adaptive_info=vio_adaptive_info,
                    timestamp=t_cam,
                    policy_scales=vio_policy_scales,
                )
                used_vo = bool(ok and r_vo_mat is not None)
            elif runner.config.use_vio_velocity and (
                avg_flow_px > 0.5 and (current_track_count == 0 or not is_dt_valid)
            ):
                print(
                    f"[VIO] Skip velocity update (stale guard): "
                    f"tracks={current_track_count}, dt_img={float(dt_img):.3f}, flow={float(avg_flow_px):.2f}"
                )

            # Store VO increments / visual-heading hint for MAG consistency policy.
            rot_angle_deg = 0.0
            if ok and r_vo_mat is not None and t_unit is not None:
                t_norm = t_unit / (np.linalg.norm(t_unit) + 1e-12)
                vo_dx, vo_dy, vo_dz = float(t_norm[0]), float(t_norm[1]), float(t_norm[2])
                r_eul = R_scipy.from_matrix(r_vo_mat).as_euler("zyx", degrees=True)
                vo_y, vo_p, vo_r = float(r_eul[0]), float(r_eul[1]), float(r_eul[2])
                self._update_visual_heading_reference(
                    t_cam=t_cam,
                    yaw_delta_deg=vo_y,
                    num_inliers=num_inliers,
                    avg_flow_px=avg_flow_px,
                    accepted=True,
                )
                vo_data = {
                    "dx": vo_dx, "dy": vo_dy, "dz": vo_dz,
                    "roll": vo_r, "pitch": vo_p, "yaw": vo_y,
                }
                rot_angle_deg = np.degrees(np.arccos(np.clip((np.trace(r_vo_mat) - 1) / 2, -1, 1)))
            else:
                vo_dx = vo_dy = vo_dz = 0.0
                vo_r = vo_p = vo_y = 0.0
                vo_data = None
                self._update_visual_heading_reference(
                    t_cam=t_cam,
                    yaw_delta_deg=0.0,
                    num_inliers=num_inliers,
                    avg_flow_px=avg_flow_px,
                    accepted=False,
                )

            # Increment VIO frame once per processed camera frame.
            runner.state.vio_frame += 1

            if runner.backend_optimizer is not None:
                try:
                    yaw_deg_state = float(np.degrees(quaternion_to_yaw(runner.kf.x[6:10, 0])))
                    q_inlier = min(1.0, float(num_inliers) / 120.0)
                    q_flow = min(1.0, float(avg_flow_px) / 10.0)
                    quality = float(np.clip(0.65 * q_inlier + 0.35 * q_flow, 0.0, 1.0))
                    runner.backend_optimizer.push_keyframe(
                        t_ref=float(t_cam),
                        p_enu=np.array(runner.kf.x[0:3, 0], dtype=float),
                        yaw_deg=yaw_deg_state,
                        quality_score=quality,
                    )
                except Exception:
                    pass

            # Log VO debug
            vel_vx = float(runner.kf.x[3, 0])
            vel_vy = float(runner.kf.x[4, 0])
            vel_vz = float(runner.kf.x[5, 0])

            view_cfg = CAMERA_VIEW_CONFIGS.get(runner.config.camera_view, CAMERA_VIEW_CONFIGS["nadir"])
            use_vz_only_default = view_cfg.get("use_vz_only", True)
            vio_config = runner.global_config.get("vio", {})
            use_vz_only = vio_config.get("use_vz_only", use_vz_only_default)

            if runner.vo_dbg_csv:
                log_vo_debug(
                    runner.vo_dbg_csv, runner.vio_fe.frame_idx, num_inliers, rot_angle_deg,
                    0.0,
                    rotation_rate_deg_s, use_vz_only,
                    not used_vo,
                    vo_dx, vo_dy, vo_dz, vel_vx, vel_vy, vel_vz,
                )

            # ================================================================
            # VPS Result Collection (after VIO completes)
            # ================================================================
            # Non-blocking poll: apply VPS only when worker has finished.
            if vps_result is None:
                vps_result, vps_result_meta = _collect_finished_vps_result()
            if vps_result is not None:
                    from ..vps_integration import apply_vps_delayed_update

                    vps_decision, vps_policy_scales = _get_policy("VPS", t_cam)
                    vps_adaptive_info: Dict[str, Any] = {}
                    vps_sync_threshold = float(runner.global_config.get("VPS_TIME_SYNC_THRESHOLD_SEC", 0.25))
                    vps_t_ref = float(vps_result_meta.get("t_cam", getattr(vps_result, "t_measurement", t_cam)))
                    vps_dt = abs(float(getattr(vps_result, "t_measurement", vps_t_ref)) - float(vps_t_ref))
                    runner.output_reporting.log_convention_check(
                        t=float(t_cam),
                        sensor="VPS",
                        check="cam_vps_abs_dt",
                        value=float(vps_dt),
                        threshold=float(vps_sync_threshold),
                        status="PASS" if vps_dt <= vps_sync_threshold else "WARN",
                        note=f"vps_t={float(getattr(vps_result, 't_measurement', vps_t_ref)):.6f}",
                    )

                    # Create clone manager if not exists
                    if not hasattr(runner, "vps_clone_manager"):
                        from vps import VPSDelayedUpdateManager

                        runner.vps_clone_manager = VPSDelayedUpdateManager(
                            max_delay_sec=0.5,
                            max_clones=3,
                        )

                    clone_id = str(vps_result_meta.get("clone_id", f"vps_{runner.state.img_idx}"))
                    health_key = str(
                        getattr(getattr(runner, "current_adaptive_decision", None), "health_state", "HEALTHY")
                    ).upper()
                    speed_now = float(np.linalg.norm(np.array(runner.kf.x[3:6, 0], dtype=float)))
                    min_inliers_apply = int(runner.global_config.get("VPS_APPLY_MIN_INLIERS", 8))
                    min_conf_apply = float(runner.global_config.get("VPS_APPLY_MIN_CONFIDENCE", 0.18))
                    max_reproj_apply = float(runner.global_config.get("VPS_APPLY_MAX_REPROJ_ERROR", 1.2))
                    max_speed_apply = float(runner.global_config.get("VPS_APPLY_MAX_SPEED_M_S", 80.0))
                    if health_key == "WARNING":
                        min_inliers_apply += int(runner.global_config.get("VPS_APPLY_WARNING_INLIER_BONUS", 2))
                        min_conf_apply *= float(runner.global_config.get("VPS_APPLY_WARNING_CONF_MULT", 1.15))
                        max_reproj_apply *= float(runner.global_config.get("VPS_APPLY_WARNING_REPROJ_MULT", 0.90))
                    elif health_key == "DEGRADED":
                        min_inliers_apply += int(runner.global_config.get("VPS_APPLY_DEGRADED_INLIER_BONUS", 4))
                        min_conf_apply *= float(runner.global_config.get("VPS_APPLY_DEGRADED_CONF_MULT", 1.30))
                        max_reproj_apply *= float(runner.global_config.get("VPS_APPLY_DEGRADED_REPROJ_MULT", 0.80))

                    vps_num_inliers = int(getattr(vps_result, "num_inliers", 0))
                    vps_conf = float(getattr(vps_result, "confidence", 0.0))
                    vps_reproj = float(getattr(vps_result, "reproj_error", float("inf")))
                    current_xy = np.array(runner.kf.x[0:2, 0], dtype=float).reshape(2,)
                    vps_xy = np.array(
                        runner.proj_cache.latlon_to_xy(float(vps_result.lat), float(vps_result.lon), runner.lat0, runner.lon0),
                        dtype=float,
                    ).reshape(2,)
                    abs_offset_vec = vps_xy - current_xy
                    abs_offset_m = float(np.linalg.norm(abs_offset_vec))
                    strict_quality_ok = (
                        np.isfinite(vps_conf)
                        and np.isfinite(vps_reproj)
                        and vps_num_inliers >= max(1, min_inliers_apply)
                        and vps_conf >= min_conf_apply
                        and vps_reproj <= max_reproj_apply
                        and speed_now <= max_speed_apply
                    )
                    vps_offset = np.asarray(getattr(vps_result, "offset_m", (np.nan, np.nan)), dtype=float).reshape(-1)
                    vps_offset_m = float(np.linalg.norm(vps_offset[:2])) if vps_offset.size >= 2 else float("nan")
                    failsoft_enabled = bool(runner.global_config.get("VPS_APPLY_FAILSOFT_ENABLE", True))
                    failsoft_allowed_state = (
                        (health_key == "HEALTHY")
                        or (health_key == "WARNING" and bool(runner.global_config.get("VPS_APPLY_FAILSOFT_ALLOW_WARNING", True)))
                        or (health_key == "DEGRADED" and bool(runner.global_config.get("VPS_APPLY_FAILSOFT_ALLOW_DEGRADED", False)))
                    )
                    fs_min_inliers = int(runner.global_config.get("VPS_APPLY_FAILSOFT_MIN_INLIERS", 5))
                    fs_min_conf = float(runner.global_config.get("VPS_APPLY_FAILSOFT_MIN_CONFIDENCE", 0.12))
                    fs_max_reproj = float(runner.global_config.get("VPS_APPLY_FAILSOFT_MAX_REPROJ_ERROR", 1.2))
                    fs_max_speed = float(runner.global_config.get("VPS_APPLY_FAILSOFT_MAX_SPEED_M_S", max_speed_apply))
                    fs_max_offset_m = float(runner.global_config.get("VPS_APPLY_FAILSOFT_MAX_OFFSET_M", 180.0))
                    failsoft_quality_ok = (
                        failsoft_enabled
                        and failsoft_allowed_state
                        and np.isfinite(vps_conf)
                        and np.isfinite(vps_reproj)
                        and np.isfinite(vps_offset_m)
                        and vps_num_inliers >= max(1, fs_min_inliers)
                        and vps_conf >= fs_min_conf
                        and vps_reproj <= fs_max_reproj
                        and speed_now <= fs_max_speed
                        and vps_offset_m <= fs_max_offset_m
                    )
                    quality_mode = "strict" if strict_quality_ok else ("failsoft" if failsoft_quality_ok else "reject")
                    r_scale_apply = float(vps_policy_scales.get("r_scale", 1.0))
                    policy_reject_note = ""
                    if vps_decision is not None and str(getattr(vps_decision, "mode", "APPLY")).upper() in ("HOLD", "SKIP"):
                        quality_mode = "reject"
                        policy_reject_note = f"policy_mode_{str(getattr(vps_decision, 'mode', 'skip')).lower()}"
                    temporal_reject_note = ""
                    hard_reject_note = ""
                    hard_ok, hard_note = self._check_vps_hard_reject(abs_offset_vec, abs_offset_m)
                    if not hard_ok:
                        quality_mode = "reject"
                        hard_reject_note = str(hard_note)

                    if quality_mode == "failsoft":
                        temporal_ok, temporal_note = self._evaluate_vps_failsoft_temporal_gate(
                            vps_offset=abs_offset_vec,
                            vps_offset_m=abs_offset_m,
                        )
                        if not temporal_ok:
                            quality_mode = "reject"
                            temporal_reject_note = str(temporal_note)

                    if quality_mode == "reject":
                        if hasattr(runner, "vps_clone_manager"):
                            runner.vps_clone_manager.clones.pop(clone_id, None)
                        vps_applied, vps_innovation_m = False, None
                        vps_status = (
                            f"SKIPPED_QUALITY: inliers={vps_num_inliers}/{min_inliers_apply}, "
                            f"conf={vps_conf:.3f}/{min_conf_apply:.3f}, "
                            f"reproj={vps_reproj:.3f}/{max_reproj_apply:.3f}, "
                            f"speed={speed_now:.2f}/{max_speed_apply:.2f}, "
                            f"offset={abs_offset_m:.1f}/{fs_max_offset_m:.1f}"
                        )
                        if hard_reject_note:
                            vps_status = f"{hard_reject_note} | {vps_status}"
                        if temporal_reject_note:
                            vps_status = f"{temporal_reject_note} | {vps_status}"
                        if policy_reject_note:
                            vps_status = f"{policy_reject_note} | {vps_status}"
                    else:
                        if quality_mode == "failsoft":
                            r_scale_apply *= float(runner.global_config.get("VPS_APPLY_FAILSOFT_R_MULT", 1.5))
                            if health_key == "WARNING":
                                r_scale_apply *= 1.25
                            elif health_key == "DEGRADED":
                                r_scale_apply *= 1.5
                        vps_lat_apply = float(vps_result.lat)
                        vps_lon_apply = float(vps_result.lon)
                        clamp_scale = 1.0
                        if quality_mode == "failsoft":
                            vps_lat_apply, vps_lon_apply, clamp_scale = self._clamp_vps_latlon(
                                current_xy=current_xy,
                                vps_lat=vps_lat_apply,
                                vps_lon=vps_lon_apply,
                            )
                        vps_applied, vps_innovation_m, vps_status = apply_vps_delayed_update(
                            kf=runner.kf,
                            clone_manager=runner.vps_clone_manager,
                            image_id=clone_id,
                            vps_lat=vps_lat_apply,
                            vps_lon=vps_lon_apply,
                            R_vps=np.array(vps_result.R_vps, dtype=float) * float(r_scale_apply),
                            proj_cache=runner.proj_cache,
                            lat0=runner.lat0,
                            lon0=runner.lon0,
                            time_since_last_vps=(t_cam - runner.last_vps_update_time),
                        )
                        if vps_applied and clamp_scale < 0.999:
                            vps_status = f"{vps_status} | CLAMPED(scale={clamp_scale:.3f})"
                    if quality_mode == "failsoft" and vps_applied:
                        reason_code = "soft_accept"
                    else:
                        reason_code = "normal_accept" if vps_applied else "hard_reject"
                    nis_norm_vps = np.nan
                    if isinstance(vps_status, str):
                        status_lower = vps_status.lower()
                        if "clone" in status_lower:
                            reason_code = "skip_missing_clone"
                        elif "skipped_quality" in status_lower:
                            reason_code = "skip_low_quality"
                            if "soft_reject" in status_lower:
                                reason_code = "soft_reject"
                        elif "hard_reject" in status_lower:
                            reason_code = "abs_corr_hard_reject"
                        elif "policy_mode_" in status_lower:
                            reason_code = "policy_hold"
                        elif "large_offset_pending" in status_lower:
                            reason_code = "abs_corr_temporal_wait"
                        elif quality_mode == "failsoft" and "gated" in status_lower:
                            reason_code = "soft_reject"
                        elif "gated" in status_lower:
                            reason_code = "gated"
                        elif "failed" in status_lower:
                            reason_code = "soft_reject" if quality_mode == "failsoft" else "hard_reject"
                        elif "clamped" in status_lower:
                            reason_code = "abs_corr_clamped_apply"
                        elif "applied" in status_lower:
                            reason_code = "abs_corr_soft_apply" if quality_mode == "failsoft" else "normal_accept"
                    vps_adaptive_info.update({
                        "sensor": "VPS",
                        "accepted": bool(vps_applied),
                        "attempted": 1,
                        "dof": 2,
                        "nis_norm": nis_norm_vps,
                        "chi2": float(vps_innovation_m) if vps_innovation_m is not None and np.isfinite(float(vps_innovation_m)) else np.nan,
                        "threshold": np.nan,
                        "r_scale_used": float(r_scale_apply),
                        "reason_code": reason_code,
                    })
                    runner.adaptive_service.record_adaptive_measurement(
                        "VPS",
                        adaptive_info=vps_adaptive_info,
                        timestamp=float(t_cam),
                        policy_scales=vps_policy_scales,
                    )
                    if reason_code.startswith("soft_accept") or reason_code == "abs_corr_soft_apply":
                        runner._vps_soft_accept_count = int(getattr(runner, "_vps_soft_accept_count", 0)) + 1
                    elif reason_code.startswith("soft_reject") or reason_code == "abs_corr_temporal_wait":
                        runner._vps_soft_reject_count = int(getattr(runner, "_vps_soft_reject_count", 0)) + 1
                    if vps_applied:
                        runner._abs_corr_apply_count = int(getattr(runner, "_abs_corr_apply_count", 0)) + 1
                        if quality_mode == "failsoft":
                            runner._abs_corr_soft_count = int(getattr(runner, "_abs_corr_soft_count", 0)) + 1
                        runner.last_vps_update_time = t_cam
                        runner.state.vps_idx += 1
                        runner._vps_pending_large_offset_vec = None
                        runner._vps_pending_large_offset_hits = 0
                        if abs_offset_vec.size >= 2:
                            runner._vps_last_accepted_offset_vec = np.array(abs_offset_vec[:2], dtype=float)
                        if runner.backend_optimizer is not None:
                            try:
                                quality_score = float(
                                    np.clip(
                                        0.45 * np.clip(vps_conf, 0.0, 1.0)
                                        + 0.35 * np.clip(vps_num_inliers / 80.0, 0.0, 1.0)
                                        + 0.20 * np.clip(1.2 / max(vps_reproj, 0.1), 0.0, 1.0),
                                        0.0,
                                        1.0,
                                    )
                                )
                                runner.backend_optimizer.report_absolute_hint(
                                    t_ref=float(t_cam),
                                    dp_enu=np.array([abs_offset_vec[0], abs_offset_vec[1], 0.0], dtype=float),
                                    dyaw_deg=0.0,
                                    quality_score=quality_score,
                                )
                            except Exception:
                                pass
                    if (
                        vps_decision is not None
                        and str(getattr(vps_decision, "mode", "APPLY")).upper() in ("HOLD", "SKIP")
                        and bool(vps_applied)
                        and getattr(runner, "policy_runtime_service", None) is not None
                    ):
                        runner.policy_runtime_service.record_conflict(
                            sensor="VPS",
                            t=float(t_cam),
                            expected_mode=str(vps_decision.mode),
                            actual_mode="APPLY",
                            note="applied_despite_hold_skip",
                        )

            # Save keyframe image with visualization overlay
            if runner.config.save_keyframe_images and hasattr(runner, "keyframe_dir"):
                save_keyframe_with_overlay(
                    img_for_tracking,
                    runner.vio_fe.frame_idx,
                    runner.keyframe_dir,
                    runner.vio_fe,
                )

            runner.state.img_idx += 1

        return used_vo, vo_data

    def schedule_backend_correction(self, corr) -> None:
        """
        Schedule backend correction for gradual blend-in to avoid EKF state jumps.
        """
        runner = self.runner
        if corr is None:
            return
        try:
            dp = np.asarray(getattr(corr, "dp_enu", np.zeros(3)), dtype=float).reshape(3,)
        except Exception:
            return
        if not np.all(np.isfinite(dp)):
            return
        dyaw_deg = float(getattr(corr, "dyaw_deg", 0.0))
        if not np.isfinite(dyaw_deg):
            dyaw_deg = 0.0

        max_dp = float(runner.global_config.get("BACKEND_MAX_APPLY_DP_XY_M", 25.0))
        dp_xy_norm = float(np.linalg.norm(dp[:2]))
        if dp_xy_norm > max_dp and dp_xy_norm > 1e-9:
            dp[:2] *= float(max_dp / dp_xy_norm)
        max_dyaw_deg = float(runner.global_config.get("BACKEND_MAX_APPLY_DYAW_DEG", 2.5))
        dyaw_deg = float(np.clip(dyaw_deg, -max_dyaw_deg, max_dyaw_deg))
        corr_weight = float(runner.global_config.get("BACKEND_CORRECTION_WEIGHT", 1.0))
        corr_weight = float(np.clip(corr_weight, 0.0, 1.0))
        if corr_weight < 1.0:
            dp *= corr_weight
            dyaw_deg *= corr_weight

        blend_steps = max(1, int(runner.global_config.get("BACKEND_BLEND_STEPS", 3)))
        runner._backend_pending_dp_enu = dp.copy()
        runner._backend_pending_dyaw_deg = float(dyaw_deg)
        runner._backend_pending_steps_left = int(blend_steps)

    def apply_pending_backend_blend(self, t_now: float) -> bool:
        """
        Apply one blend step from pending backend correction.
        """
        runner = self.runner
        steps_left = int(getattr(runner, "_backend_pending_steps_left", 0))
        if steps_left <= 0:
            return False
        dp_rem = getattr(runner, "_backend_pending_dp_enu", None)
        if dp_rem is None:
            runner._backend_pending_steps_left = 0
            return False

        dp_rem = np.asarray(dp_rem, dtype=float).reshape(3,)
        dyaw_rem = float(getattr(runner, "_backend_pending_dyaw_deg", 0.0))
        step_dp = dp_rem / float(steps_left)
        step_dyaw_deg = dyaw_rem / float(steps_left)

        inflate = float(runner.global_config.get("BACKEND_APPLY_COV_INFLATE", 1.05))
        if np.isfinite(inflate) and inflate > 1.0:
            runner.kf.P[0:6, 0:6] *= float(min(inflate, 1.25))

        runner.kf.x[0:3, 0] = runner.kf.x[0:3, 0] + step_dp.reshape(3,)
        if abs(step_dyaw_deg) > 1e-9:
            q_wxyz = runner.kf.x[6:10, 0].astype(float).reshape(4,)
            q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)
            r_now = R_scipy.from_quat(q_xyzw)
            yaw, pitch, roll = r_now.as_euler("zyx", degrees=False)
            yaw_new = float(np.arctan2(np.sin(yaw + np.deg2rad(step_dyaw_deg)), np.cos(yaw + np.deg2rad(step_dyaw_deg))))
            r_new = R_scipy.from_euler("zyx", [yaw_new, pitch, roll], degrees=False)
            q_new_xyzw = r_new.as_quat()
            runner.kf.x[6:10, 0] = np.array(
                [q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]],
                dtype=float,
            )

        runner._backend_pending_dp_enu = dp_rem - step_dp
        runner._backend_pending_dyaw_deg = dyaw_rem - step_dyaw_deg
        runner._backend_pending_steps_left = steps_left - 1
        runner._backend_apply_count = int(getattr(runner, "_backend_apply_count", 0)) + 1

        if int(runner._backend_pending_steps_left) <= 0:
            runner._backend_pending_dp_enu = None
            runner._backend_pending_dyaw_deg = 0.0
            runner._backend_pending_steps_left = 0
        return True
