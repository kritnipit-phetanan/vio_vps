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
            runner._vps_pending_large_offset_vec = None
            runner._vps_pending_large_offset_hits = 0
        elif np.isfinite(vps_offset_m) and vps_offset_m <= large_offset_m:
            runner._vps_pending_large_offset_vec = None
            runner._vps_pending_large_offset_hits = 0

        return True, ""

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
            # VPS runs in background thread while VIO continues immediately
            # This reduces total latency and utilizes multi-core CPUs
            vps_thread = None
            vps_result_container = [None]  # Thread-safe result storage

            if runner.vps_runner is not None:
                if is_fast_rotation:
                    should_run_vps, skip_reason = False, "fast_rotation"
                else:
                    should_run_vps, skip_reason = self._should_run_vps(t_cam=t_cam, img=img)
                if not should_run_vps:
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
                    # Get current EKF estimates for VPS processing
                    # Extract IMU position from EKF state
                    p_imu_enu = runner.kf.x[:3].flatten()  # Position in ENU

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

                    # Get terrain height (DEM) to compute AGL for VPS filtering
                    dem_height = runner.dem.sample_m(lat, lon)
                    if dem_height is None:
                        dem_height = 0.0
                    agl = alt - dem_height

                    q_wxyz = runner.kf.x[6:10, 0].astype(float)
                    est_yaw = float(quaternion_to_yaw(q_wxyz))

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
                                frame_idx=runner.state.img_idx,
                                est_cov_xy=est_cov_xy,
                                phase=int(getattr(runner.state, "current_phase", 2)),
                                objective=str(runner.global_config.get("OBJECTIVE_MODE", "stability")),
                            )
                            vps_result_container[0] = result
                        except Exception as e:
                            print(f"[VPS] Thread error: {e}")
                            vps_result_container[0] = None

                    # Start VPS processing in background thread
                    vps_thread = threading.Thread(target=run_vps_in_thread, daemon=True)
                    vps_thread.start()

                    # Clone EKF state for delayed update (stochastic cloning)
                    if hasattr(runner, "vps_clone_manager"):
                        clone_id = f"vps_{runner.state.img_idx}"
                        runner.vps_clone_manager.clone_state(runner.kf, t_cam, clone_id)

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
                apply_loop_closure_correction(
                    kf=runner.kf,
                    loop_info=match_result,
                    t=t_cam,
                    cam_states=runner.state.cam_states,
                    loop_detector=runner.loop_detector,
                    global_config=runner.global_config,
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
                        msckf_policy_scales, msckf_apply_scales = runner.adaptive_service.get_sensor_adaptive_scales("MSCKF")
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
                            chi2_scale=float(msckf_apply_scales.get("chi2_scale", 1.0)) * phase_chi2,
                            reproj_scale=float(msckf_apply_scales.get("reproj_scale", 1.0)) * phase_reproj,
                            phase=int(getattr(runner.state, "current_phase", 2)),
                            health_state=str(
                                getattr(getattr(runner, "current_adaptive_decision", None), "health_state", "HEALTHY")
                            ),
                            adaptive_info=msckf_adaptive_info,
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
                vio_policy_scales, vio_apply_scales = runner.adaptive_service.get_sensor_adaptive_scales("VIO_VEL")
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
                    chi2_scale=float(vio_apply_scales.get("chi2_scale", 1.0)),
                    r_scale_extra=float(vio_apply_scales.get("r_scale", 1.0)),
                    soft_fail_enable=bool(float(vio_apply_scales.get("fail_soft_enable", 0.0)) >= 0.5),
                    soft_fail_r_cap=float(vio_apply_scales.get("soft_r_cap", 8.0)),
                    soft_fail_hard_reject_factor=float(vio_apply_scales.get("hard_reject_factor", 3.0)),
                    soft_fail_power=float(vio_apply_scales.get("soft_r_power", 1.0)),
                    phase=int(getattr(runner.state, "current_phase", 2)),
                    health_state=str(
                        getattr(getattr(runner, "current_adaptive_decision", None), "health_state", "HEALTHY")
                    ),
                    adaptive_info=vio_adaptive_info,
                )
                if bool(vio_vel_accepted):
                    runner._vio_vel_accept_count += 1
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
            if vps_thread is not None:
                vps_thread.join(timeout=2.0)
                vps_result = vps_result_container[0]
                if vps_result is not None:
                    from ..vps_integration import apply_vps_delayed_update

                    vps_policy_scales, vps_apply_scales = runner.adaptive_service.get_sensor_adaptive_scales("VPS")
                    vps_adaptive_info: Dict[str, Any] = {}
                    vps_sync_threshold = float(runner.global_config.get("VPS_TIME_SYNC_THRESHOLD_SEC", 0.25))
                    vps_dt = abs(float(getattr(vps_result, "t_measurement", t_cam)) - float(t_cam))
                    runner.output_reporting.log_convention_check(
                        t=float(t_cam),
                        sensor="VPS",
                        check="cam_vps_abs_dt",
                        value=float(vps_dt),
                        threshold=float(vps_sync_threshold),
                        status="PASS" if vps_dt <= vps_sync_threshold else "WARN",
                        note=f"vps_t={float(getattr(vps_result, 't_measurement', t_cam)):.6f}",
                    )

                    # Create clone manager if not exists
                    if not hasattr(runner, "vps_clone_manager"):
                        from vps import VPSDelayedUpdateManager

                        runner.vps_clone_manager = VPSDelayedUpdateManager(
                            max_delay_sec=0.5,
                            max_clones=3,
                        )

                    clone_id = f"vps_{runner.state.img_idx}"
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
                    r_scale_apply = float(vps_apply_scales.get("r_scale", 1.0))
                    temporal_reject_note = ""

                    if quality_mode == "failsoft":
                        temporal_ok, temporal_note = self._evaluate_vps_failsoft_temporal_gate(
                            vps_offset=vps_offset,
                            vps_offset_m=vps_offset_m,
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
                            f"offset={vps_offset_m:.1f}/{fs_max_offset_m:.1f}"
                        )
                        if temporal_reject_note:
                            vps_status = f"{temporal_reject_note} | {vps_status}"
                    else:
                        if quality_mode == "failsoft":
                            r_scale_apply *= float(runner.global_config.get("VPS_APPLY_FAILSOFT_R_MULT", 1.5))
                            if health_key == "WARNING":
                                r_scale_apply *= 1.25
                            elif health_key == "DEGRADED":
                                r_scale_apply *= 1.5
                        vps_applied, vps_innovation_m, vps_status = apply_vps_delayed_update(
                            kf=runner.kf,
                            clone_manager=runner.vps_clone_manager,
                            image_id=clone_id,
                            vps_lat=vps_result.lat,
                            vps_lon=vps_result.lon,
                            R_vps=np.array(vps_result.R_vps, dtype=float) * float(r_scale_apply),
                            proj_cache=runner.proj_cache,
                            lat0=runner.lat0,
                            lon0=runner.lon0,
                            time_since_last_vps=(t_cam - runner.last_vps_update_time),
                        )
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
                        elif quality_mode == "failsoft" and "gated" in status_lower:
                            reason_code = "soft_reject"
                        elif "gated" in status_lower:
                            reason_code = "gated"
                        elif "failed" in status_lower:
                            reason_code = "soft_reject" if quality_mode == "failsoft" else "hard_reject"
                        elif "applied" in status_lower:
                            reason_code = "soft_accept" if quality_mode == "failsoft" else "normal_accept"
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
                    if reason_code.startswith("soft_accept"):
                        runner._vps_soft_accept_count = int(getattr(runner, "_vps_soft_accept_count", 0)) + 1
                    elif reason_code.startswith("soft_reject"):
                        runner._vps_soft_reject_count = int(getattr(runner, "_vps_soft_reject_count", 0)) + 1
                    if vps_applied:
                        runner.last_vps_update_time = t_cam
                        runner.state.vps_idx += 1
                        runner._vps_pending_large_offset_vec = None
                        runner._vps_pending_large_offset_hits = 0
                        if vps_offset.size >= 2:
                            runner._vps_last_accepted_offset_vec = np.array(vps_offset[:2], dtype=float)

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
