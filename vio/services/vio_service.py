"""Vision/MSCKF/VPS (threaded) service for VIORunner."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from ..backend_optimizer import BackendCorrection, BackendCorrectionContractV1
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
from ..policy.types import ApplyDecisionRecord, SensorPolicyDecision
from .vps_position_controller import VPSPositionController, VpsMatchEvidence


class VIOService:
    """Encapsulates camera/VIO frontend, MSCKF, and VPS-thread interaction."""

    def __init__(self, runner: Any):
        self.runner = runner
        self.vps_position_controller = VPSPositionController(runner=runner, vio_service=self)

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

    def _evaluate_vps_failsoft_temporal_gate(
        self,
        vps_offset: np.ndarray,
        vps_offset_m: float,
        speed_m_s: float = float("nan"),
        yaw_rate_deg_s: float = float("nan"),
    ) -> Tuple[bool, str]:
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
        dir_gate_relax_max_speed = float(
            runner.global_config.get("VPS_APPLY_FAILSOFT_DIR_GATE_MAX_SPEED_M_S", 12.0)
        )
        dir_gate_relax_max_yaw_rate = float(
            runner.global_config.get("VPS_APPLY_FAILSOFT_DIR_GATE_MAX_YAW_RATE_DEG_S", 25.0)
        )
        dir_gate_relax_mult_speed = float(
            runner.global_config.get("VPS_APPLY_FAILSOFT_DIR_GATE_RELAX_MULT_SPEED", 1.6)
        )
        dir_gate_relax_mult_yaw = float(
            runner.global_config.get("VPS_APPLY_FAILSOFT_DIR_GATE_RELAX_MULT_YAW_RATE", 1.6)
        )
        dir_gate_disable_speed = float(
            runner.global_config.get("VPS_APPLY_FAILSOFT_DIR_GATE_DISABLE_SPEED_M_S", 24.0)
        )
        dir_gate_disable_yaw_rate = float(
            runner.global_config.get("VPS_APPLY_FAILSOFT_DIR_GATE_DISABLE_YAW_RATE_DEG_S", 45.0)
        )
        speed_high = np.isfinite(speed_m_s) and float(speed_m_s) > float(dir_gate_relax_max_speed)
        yaw_high = np.isfinite(yaw_rate_deg_s) and float(yaw_rate_deg_s) > float(dir_gate_relax_max_yaw_rate)
        if speed_high:
            max_dir_change_deg *= max(1.0, float(dir_gate_relax_mult_speed))
        if yaw_high:
            max_dir_change_deg *= max(1.0, float(dir_gate_relax_mult_yaw))
        max_dir_change_deg = float(np.clip(max_dir_change_deg, 5.0, 175.0))
        if (
            (np.isfinite(speed_m_s) and float(speed_m_s) > float(dir_gate_disable_speed))
            or (np.isfinite(yaw_rate_deg_s) and float(yaw_rate_deg_s) > float(dir_gate_disable_yaw_rate))
        ):
            return True, "SOFT_BYPASS_DIR_DYNAMIC"
        last_vec = getattr(runner, "_vps_last_accepted_offset_vec", None)
        if isinstance(last_vec, np.ndarray) and last_vec.size >= 2:
            last = np.array(last_vec[:2], dtype=float)
            last_norm = float(np.linalg.norm(last))
            if np.isfinite(last_norm) and last_norm > 1e-6:
                cosang = float(np.clip(np.dot(cur, last) / (cur_norm * last_norm), -1.0, 1.0))
                dir_delta_deg = float(np.degrees(np.arccos(cosang)))
                if dir_delta_deg > max_dir_change_deg:
                    runner._vps_jump_reject_count = int(getattr(runner, "_vps_jump_reject_count", 0)) + 1
                    runner._vps_temporal_consensus_block_count = int(
                        getattr(runner, "_vps_temporal_consensus_block_count", 0)
                    ) + 1
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
                runner._vps_temporal_consensus_block_count = int(
                    getattr(runner, "_vps_temporal_consensus_block_count", 0)
                ) + 1
                return False, (
                    f"SOFT_REJECT_LARGE_OFFSET_PENDING: hits={pending_hits}/{required_hits}, "
                    f"offset={vps_offset_m:.1f}m"
                )
            runner._vps_temporal_confirm_count = int(getattr(runner, "_vps_temporal_confirm_count", 0)) + 1
            runner._vps_temporal_consensus_pass_count = int(
                getattr(runner, "_vps_temporal_consensus_pass_count", 0)
            ) + 1
            runner._vps_pending_large_offset_vec = None
            runner._vps_pending_large_offset_hits = 0
        elif np.isfinite(vps_offset_m) and vps_offset_m <= large_offset_m:
            runner._vps_pending_large_offset_vec = None
            runner._vps_pending_large_offset_hits = 0

        return True, ""

    def _check_vps_hard_reject(self,
                               vps_offset: np.ndarray,
                               vps_offset_m: float,
                               speed_m_s: float = float("nan"),
                               yaw_rate_deg_s: float = float("nan"),
                               allow_dir_change_override: bool = False) -> Tuple[bool, str]:
        """
        Hard safety gate for absolute correction before delayed-update apply.

        Rejects obviously unsafe updates:
        - huge offset magnitude
        - abrupt direction change vs previous accepted offset
        """
        runner = self.runner
        hard_max_offset_m = float(runner.global_config.get("VPS_ABS_HARD_REJECT_OFFSET_M", 180.0))
        if np.isfinite(vps_offset_m) and vps_offset_m > hard_max_offset_m:
            allow_offset_bypass = bool(
                bool(allow_dir_change_override)
                and bool(runner.global_config.get("VPS_XY_DRIFT_RECOVERY_ALLOW_OFFSET_BYPASS", True))
            )
            if allow_offset_bypass:
                max_recovery_offset = float(
                    runner.global_config.get("VPS_XY_DRIFT_RECOVERY_MAX_OFFSET_M", 500.0)
                )
                if float(vps_offset_m) <= max(0.0, max_recovery_offset):
                    return True, (
                        f"OFFSET_SOFT_BYPASS: {vps_offset_m:.1f}m>{hard_max_offset_m:.1f}m"
                    )
            runner._vps_jump_reject_count = int(getattr(runner, "_vps_jump_reject_count", 0)) + 1
            return False, f"HARD_REJECT_OFFSET: {vps_offset_m:.1f}m>{hard_max_offset_m:.1f}m"

        hard_max_dir_change_deg = float(runner.global_config.get("VPS_ABS_HARD_REJECT_DIR_CHANGE_DEG", 75.0))
        dir_check_max_speed = float(
            runner.global_config.get("VPS_ABS_HARD_REJECT_DIR_CHANGE_MAX_SPEED_M_S", 12.0)
        )
        dir_check_max_yaw_rate = float(
            runner.global_config.get("VPS_ABS_HARD_REJECT_DIR_CHANGE_MAX_YAW_RATE_DEG_S", 25.0)
        )
        if np.isfinite(speed_m_s) and float(speed_m_s) > float(dir_check_max_speed):
            return True, ""
        if np.isfinite(yaw_rate_deg_s) and float(yaw_rate_deg_s) > float(dir_check_max_yaw_rate):
            return True, ""
        min_accepts_for_dir = int(
            max(1, round(float(runner.global_config.get("VPS_ABS_HARD_REJECT_DIR_CHANGE_MIN_ACCEPTS", 3.0))))
        )
        if int(getattr(runner, "_vps_last_accepted_offset_count", 0)) < min_accepts_for_dir:
            return True, ""
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
                    if bool(allow_dir_change_override):
                        return True, (
                            f"DIR_CHANGE_SOFT_BYPASS: {dir_delta_deg:.1f}deg>"
                            f"{hard_max_dir_change_deg:.1f}deg"
                        )
                    runner._vps_jump_reject_count = int(getattr(runner, "_vps_jump_reject_count", 0)) + 1
                    return False, (
                        f"HARD_REJECT_DIR_CHANGE: {dir_delta_deg:.1f}deg>"
                        f"{hard_max_dir_change_deg:.1f}deg"
                    )
        return True, ""

    def _clamp_vps_latlon(self,
                          current_xy: np.ndarray,
                          vps_lat: float,
                          vps_lon: float,
                          max_apply_dp_xy_override: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Clamp one-shot VPS correction magnitude in XY before delayed-update apply.
        """
        runner = self.runner
        max_apply_dp_xy = float(runner.global_config.get("VPS_ABS_MAX_APPLY_DP_XY_M", 25.0))
        if max_apply_dp_xy_override is not None and np.isfinite(float(max_apply_dp_xy_override)):
            max_apply_dp_xy = float(max(0.5, max_apply_dp_xy_override))
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

    def _should_use_vps_xy_drift_recovery(
        self,
        t_cam: float,
        abs_offset_m: float,
        speed_m_s: float,
        vps_num_inliers: int,
        vps_conf: float,
        vps_reproj: float,
        policy_mode: str,
    ) -> Tuple[bool, str]:
        """Return True when XY-only VPS recovery should be attempted despite strict-quality reject."""
        runner = self.runner
        if not bool(runner.global_config.get("VPS_XY_DRIFT_RECOVERY_ENABLE", True)):
            return False, ""
        if str(policy_mode).upper() in ("HOLD", "SKIP"):
            return False, "policy_hold_skip"
        if not np.isfinite(abs_offset_m):
            return False, "offset_nan"

        min_offset_m = float(runner.global_config.get("VPS_XY_DRIFT_RECOVERY_MIN_OFFSET_M", 120.0))
        max_offset_m = float(runner.global_config.get("VPS_XY_DRIFT_RECOVERY_MAX_OFFSET_M", 500.0))
        if abs_offset_m < max(1.0, min_offset_m):
            return False, "offset_small"
        if abs_offset_m > max(min_offset_m, max_offset_m):
            return False, "offset_too_large"

        since_last_vps = float(t_cam) - float(getattr(runner, "last_vps_update_time", -1e9))
        min_no_apply_sec = float(runner.global_config.get("VPS_XY_DRIFT_RECOVERY_MIN_NO_APPLY_SEC", 5.0))
        if np.isfinite(since_last_vps) and since_last_vps < max(0.0, min_no_apply_sec):
            return False, "recent_apply"

        min_inliers = int(runner.global_config.get("VPS_XY_DRIFT_RECOVERY_MIN_INLIERS", 6))
        min_conf = float(runner.global_config.get("VPS_XY_DRIFT_RECOVERY_MIN_CONFIDENCE", 0.10))
        max_reproj = float(runner.global_config.get("VPS_XY_DRIFT_RECOVERY_MAX_REPROJ_ERROR", 1.6))
        max_speed = float(runner.global_config.get("VPS_XY_DRIFT_RECOVERY_MAX_SPEED_M_S", 95.0))
        if int(vps_num_inliers) < max(1, min_inliers):
            return False, "quality_inliers"
        if not np.isfinite(vps_conf) or float(vps_conf) < float(min_conf):
            return False, "quality_conf"
        if not np.isfinite(vps_reproj) or float(vps_reproj) > float(max_reproj):
            return False, "quality_reproj"
        if np.isfinite(speed_m_s) and float(speed_m_s) > float(max_speed):
            return False, "speed_high"
        return True, "xy_drift_recovery"

    def _should_use_vps_position_first_soft_apply(
        self,
        t_cam: float,
        abs_offset_m: float,
        speed_m_s: float,
        vps_num_inliers: int,
        vps_conf: float,
        vps_reproj: float,
        policy_mode: str,
    ) -> Tuple[bool, str]:
        """
        Position-first fallback:
        allow conservative XY soft-apply when strict/failsoft gates reject too often.
        """
        runner = self.runner
        if not bool(runner.global_config.get("POSITION_FIRST_LANE", False)):
            return False, ""
        if not bool(runner.global_config.get("VPS_POSITION_FIRST_SOFT_ENABLE", True)):
            return False, ""
        if str(policy_mode).upper() in ("HOLD", "SKIP"):
            if not bool(runner.global_config.get("VPS_POSITION_FIRST_SOFT_IGNORE_POLICY_HOLD", True)):
                return False, "policy_hold_skip"
        if not np.isfinite(abs_offset_m):
            return False, "offset_nan"
        min_offset = float(runner.global_config.get("VPS_POSITION_FIRST_SOFT_MIN_OFFSET_M", 30.0))
        if abs_offset_m < max(0.0, min_offset):
            return False, "offset_small"
        since_last_vps = float(t_cam) - float(getattr(runner, "last_vps_update_time", -1e9))
        min_no_apply_sec = float(runner.global_config.get("VPS_POSITION_FIRST_SOFT_MIN_NO_APPLY_SEC", 2.0))
        if np.isfinite(since_last_vps) and since_last_vps < max(0.0, min_no_apply_sec):
            return False, "recent_apply"

        min_inliers = int(runner.global_config.get("VPS_POSITION_FIRST_SOFT_MIN_INLIERS", 4))
        min_conf = float(runner.global_config.get("VPS_POSITION_FIRST_SOFT_MIN_CONFIDENCE", 0.06))
        max_reproj = float(runner.global_config.get("VPS_POSITION_FIRST_SOFT_MAX_REPROJ_ERROR", 2.2))
        max_speed = float(runner.global_config.get("VPS_POSITION_FIRST_SOFT_MAX_SPEED_M_S", 110.0))
        if int(vps_num_inliers) < max(1, min_inliers):
            return False, "quality_inliers"
        if not np.isfinite(vps_conf) or float(vps_conf) < float(min_conf):
            return False, "quality_conf"
        if not np.isfinite(vps_reproj) or float(vps_reproj) > float(max_reproj):
            return False, "quality_reproj"
        if np.isfinite(speed_m_s) and float(speed_m_s) > float(max_speed):
            return False, "speed_high"
        return True, "position_first_soft_apply"

    def _apply_position_first_direct_xy_correction(
        self,
        *,
        t_cam: float,
        abs_offset_vec: np.ndarray,
        abs_offset_m: float,
        vps_num_inliers: int,
        vps_conf: float,
        vps_reproj: float,
        base_quality: float,
        failsoft_selected: bool = False,
    ) -> Tuple[bool, str]:
        """
        Schedule deterministic XY-only backend correction directly from VPS hint.
        This path bypasses VPS EKF accept requirement for position-first recovery.
        """
        runner = self.runner
        if not bool(runner.global_config.get("POSITION_FIRST_LANE", False)):
            return False, "position_first_lane_off"
        if not bool(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_ENABLE", True)):
            return False, "position_first_direct_xy_disabled"
        if not np.isfinite(abs_offset_m) or float(abs_offset_m) <= 1e-6:
            return False, "position_first_direct_xy_invalid_offset"

        last_t = float(getattr(runner, "_position_first_direct_last_t", -1e9))
        min_interval = float(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_MIN_INTERVAL_SEC", 0.8))
        if float(t_cam) - last_t < max(0.0, float(min_interval)):
            return False, "position_first_direct_xy_cooldown"
        try:
            speed_m_s = float(np.linalg.norm(np.asarray(runner.kf.x[3:6, 0], dtype=float)))
        except Exception:
            speed_m_s = float("nan")
        if not np.isfinite(speed_m_s):
            speed_m_s = float("nan")
        vps_runner = getattr(runner, "vps_runner", None)
        fail_streak = int(getattr(vps_runner, "_fail_streak", 0)) if vps_runner is not None else 0
        no_coverage_streak = int(getattr(vps_runner, "_no_coverage_streak", 0)) if vps_runner is not None else 0
        guarded_mode = bool(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_GUARD_ENABLE", True)) and (
            fail_streak >= int(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_GUARD_FAIL_STREAK_TH", 18))
            or no_coverage_streak >= int(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_GUARD_NO_COVERAGE_STREAK_TH", 6))
        )

        vec_in = np.asarray(abs_offset_vec, dtype=float).reshape(-1,)
        if vec_in.size < 2 or (not np.all(np.isfinite(vec_in[:2]))):
            return False, "position_first_direct_xy_invalid_vec"
        vec_xy = np.array([float(vec_in[0]), float(vec_in[1])], dtype=float)
        # Prefer robust consensus center over raw one-shot offset to reduce
        # outlier-driven XY jumps in position-first lane.
        use_consensus_vec = bool(
            runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_USE_CONSENSUS_VECTOR", True)
        )
        if use_consensus_vec:
            hist_raw = getattr(runner, "_position_first_direct_hint_hist", None)
            if isinstance(hist_raw, list):
                cons_window = max(
                    0.5,
                    float(
                        runner.global_config.get(
                            "VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_WINDOW_SEC",
                            4.0,
                        )
                    ),
                )
                cons_min_samples = max(
                    1,
                    int(
                        round(
                            runner.global_config.get(
                                "VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_MIN_SAMPLES",
                                3,
                            )
                        )
                    ),
                )
                cutoff = float(t_cam) - float(cons_window)
                vals = []
                for item in hist_raw:
                    try:
                        tt = float(item[0])
                        vv = np.asarray(item[1], dtype=float).reshape(2,)
                    except Exception:
                        continue
                    if np.isfinite(tt) and np.all(np.isfinite(vv)) and tt >= cutoff:
                        vals.append(vv)
                if len(vals) >= cons_min_samples:
                    arr = np.asarray(vals, dtype=float)
                    med_vec = np.nanmedian(arr, axis=0)
                    if np.all(np.isfinite(med_vec)):
                        vec_xy = med_vec.astype(float)

        dp = np.array([float(vec_xy[0]), float(vec_xy[1]), 0.0], dtype=float)
        max_dp = float(
            runner.global_config.get(
                "VPS_POSITION_FIRST_DIRECT_XY_MAX_APPLY_DP_XY_M",
                runner.global_config.get(
                    "BACKEND_POSITION_FIRST_MAX_APPLY_DP_XY_M",
                    runner.global_config.get("BACKEND_MAX_APPLY_DP_XY_M", 25.0),
                ),
            )
        )
        max_dp = max(1.0, float(max_dp))
        if guarded_mode:
            max_dp = min(
                max_dp,
                max(
                    2.0,
                    float(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_GUARD_MAX_APPLY_DP_XY_M", 14.0)),
                ),
            )
        since_last_vps = float(t_cam) - float(getattr(runner, "last_vps_update_time", -1e9))
        recovery_boost_max_speed = float(
            runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MAX_SPEED_M_S", 14.0)
        )
        recovery_boost_min_inliers = int(
            runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MIN_INLIERS", 7)
        )
        recovery_boost_max_reproj = float(
            runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MAX_REPROJ_ERROR", 0.9)
        )
        recovery_quality_ok = bool(
            int(vps_num_inliers) >= max(1, int(recovery_boost_min_inliers))
            and np.isfinite(vps_reproj)
            and float(vps_reproj) <= float(recovery_boost_max_reproj)
        )
        # When VPS already marked this measurement as failsoft-success, keep the
        # same deterministic guardrails but allow a relaxed recovery-quality gate.
        if bool(failsoft_selected) and not bool(recovery_quality_ok):
            fs_rec_min_inliers = max(
                1,
                int(
                    round(
                        runner.global_config.get(
                            "VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_RECOVERY_MIN_INLIERS",
                            5,
                        )
                    )
                ),
            )
            fs_rec_max_reproj = float(
                runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_RECOVERY_MAX_REPROJ_ERROR",
                    1.3,
                )
            )
            recovery_quality_ok = bool(
                int(vps_num_inliers) >= fs_rec_min_inliers
                and np.isfinite(vps_reproj)
                and float(vps_reproj) <= float(fs_rec_max_reproj)
            )
        speed_recovery_ok = bool(
            (not np.isfinite(speed_m_s))
            or float(speed_m_s) <= float(recovery_boost_max_speed)
        )
        recovery_boost = bool(
            np.isfinite(float(abs_offset_m))
            and float(abs_offset_m)
            >= float(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MIN_OFFSET_M", 120.0))
            and np.isfinite(since_last_vps)
            and since_last_vps
            >= max(
                0.0,
                float(
                    runner.global_config.get(
                        "VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MIN_NO_APPLY_SEC",
                        6.0,
                    )
                ),
            )
            and speed_recovery_ok
            and recovery_quality_ok
        )
        if recovery_boost:
            boost_mult = max(
                1.0,
                float(
                    runner.global_config.get(
                        "VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MAX_DP_MULT",
                        1.8,
                    )
                ),
            )
            boost_cap = max(
                max_dp,
                float(
                    runner.global_config.get(
                        "VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_MAX_DP_CAP_M",
                        90.0,
                    )
                ),
            )
            max_dp = min(boost_cap, max_dp * boost_mult)
        dp_norm = float(np.linalg.norm(dp[:2]))
        if dp_norm > max_dp and dp_norm > 1e-9:
            dp[:2] *= float(max_dp / dp_norm)
            dp_norm = float(np.linalg.norm(dp[:2]))

        quality_scale = float(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_QUALITY_SCALE", 0.90))
        if guarded_mode:
            quality_scale *= float(
                np.clip(
                    runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_GUARD_QUALITY_MULT", 0.78),
                    0.2,
                    1.0,
                )
            )
        quality_scale = float(np.clip(quality_scale, 0.05, 2.0))
        q = float(np.clip(float(base_quality) * quality_scale, 0.0, 1.0))
        if guarded_mode:
            min_q_guard = float(
                np.clip(
                    runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_GUARD_MIN_QUALITY", 0.28),
                    0.0,
                    1.0,
                )
            )
            if float(q) < float(min_q_guard):
                allow_lowq_failsoft = bool(
                    bool(failsoft_selected)
                    and bool(
                        runner.global_config.get(
                            "VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_GUARD_LOWQ_ENABLE",
                            True,
                        )
                    )
                )
                if not allow_lowq_failsoft:
                    return False, "position_first_direct_xy_guard_low_quality"
                lowq_min_inliers = max(
                    1,
                    int(
                        round(
                            runner.global_config.get(
                                "VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_GUARD_LOWQ_MIN_INLIERS",
                                4,
                            )
                        )
                    ),
                )
                lowq_max_reproj = float(
                    runner.global_config.get(
                        "VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_GUARD_LOWQ_MAX_REPROJ_ERROR",
                        1.6,
                    )
                )
                if int(vps_num_inliers) < lowq_min_inliers:
                    return False, "position_first_direct_xy_guard_low_quality"
                if not np.isfinite(vps_reproj) or float(vps_reproj) > float(lowq_max_reproj):
                    return False, "position_first_direct_xy_guard_low_quality"
                lowq_max_dp = max(
                    1.0,
                    float(
                        runner.global_config.get(
                            "VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_GUARD_LOWQ_MAX_APPLY_DP_XY_M",
                            7.0,
                        )
                    ),
                )
                max_dp = min(float(max_dp), float(lowq_max_dp))
        # Quality-aware deterministic cap (logic, not threshold-only tuning):
        # low-confidence hints get a much smaller per-apply XY displacement.
        q_cap_floor = max(
            0.5,
            float(
                runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_QUALITY_CAP_MIN_M",
                    6.0,
                )
            ),
        )
        q_cap = float(q_cap_floor + (max_dp - q_cap_floor) * np.clip(q, 0.0, 1.0))
        q_cap = max(0.5, min(max_dp, q_cap))
        if dp_norm > q_cap and dp_norm > 1e-9:
            dp[:2] *= float(q_cap / dp_norm)
            dp_norm = float(np.linalg.norm(dp[:2]))

        # Deterministic gain shaping to avoid bursty/oscillatory XY injections:
        # - lower gain when quality is weak
        # - lower gain at high speed
        # - lower gain when proposed vector flips against recent applied direction
        apply_gain = float(
            np.clip(
                runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_APPLY_GAIN_BASE",
                    0.72,
                ),
                0.10,
                1.00,
            )
        )
        q_gain_floor = float(
            np.clip(
                runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_APPLY_GAIN_QUALITY_FLOOR",
                    0.45,
                ),
                0.10,
                1.00,
            )
        )
        q_gain = float(q_gain_floor + (1.0 - q_gain_floor) * np.clip(q, 0.0, 1.0))
        apply_gain *= float(np.clip(q_gain, 0.10, 1.00))

        high_speed_gain_th = float(
            runner.global_config.get(
                "VPS_POSITION_FIRST_DIRECT_XY_APPLY_GAIN_HIGH_SPEED_TH_M_S",
                18.0,
            )
        )
        if np.isfinite(speed_m_s) and float(speed_m_s) >= float(high_speed_gain_th):
            apply_gain *= float(
                np.clip(
                    runner.global_config.get(
                        "VPS_POSITION_FIRST_DIRECT_XY_APPLY_GAIN_HIGH_SPEED_MULT",
                        0.72,
                    ),
                    0.10,
                    1.00,
                )
            )

        if bool(failsoft_selected):
            apply_gain *= float(
                np.clip(
                    runner.global_config.get(
                        "VPS_POSITION_FIRST_DIRECT_XY_APPLY_GAIN_FAILSOFT_MULT",
                        0.82,
                    ),
                    0.10,
                    1.00,
                )
            )

        last_apply_vec = getattr(runner, "_position_first_direct_last_apply_vec", None)
        if isinstance(last_apply_vec, np.ndarray) and last_apply_vec.size >= 2:
            prev_vec = np.asarray(last_apply_vec[:2], dtype=float).reshape(2,)
            prev_n = float(np.linalg.norm(prev_vec))
            cur_n = float(np.linalg.norm(dp[:2]))
            reverse_check_min_m = float(
                max(
                    0.5,
                    runner.global_config.get(
                        "VPS_POSITION_FIRST_DIRECT_XY_APPLY_GAIN_REVERSE_DIR_CHECK_MIN_M",
                        6.0,
                    ),
                )
            )
            if prev_n >= reverse_check_min_m and cur_n >= reverse_check_min_m:
                cosang = float(np.clip(np.dot(prev_vec, dp[:2]) / max(1e-9, prev_n * cur_n), -1.0, 1.0))
                reverse_cos = float(
                    np.cos(
                        np.deg2rad(
                            float(
                                np.clip(
                                    runner.global_config.get(
                                        "VPS_POSITION_FIRST_DIRECT_XY_APPLY_GAIN_REVERSE_DIR_DEG",
                                        120.0,
                                    ),
                                    90.0,
                                    179.0,
                                )
                            )
                        )
                    )
                )
                if cosang <= reverse_cos:
                    apply_gain *= float(
                        np.clip(
                            runner.global_config.get(
                                "VPS_POSITION_FIRST_DIRECT_XY_APPLY_GAIN_REVERSE_DIR_MULT",
                                0.55,
                            ),
                            0.10,
                            1.00,
                        )
                    )

        apply_gain = float(np.clip(apply_gain, 0.10, 1.00))
        if apply_gain < 0.999 and dp_norm > 1e-9:
            dp[:2] *= float(apply_gain)
            dp_norm = float(np.linalg.norm(dp[:2]))

        # Deterministic multi-hit commit for large direct-XY jumps:
        # avoid applying single-frame outliers (especially at high speed tails).
        confirm_offset_m = max(
            1.0,
            float(
                runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_CONFIRM_OFFSET_M",
                    35.0,
                )
            ),
        )
        confirm_hits = max(
            1,
            int(
                round(
                    runner.global_config.get(
                        "VPS_POSITION_FIRST_DIRECT_XY_CONFIRM_HITS",
                        2,
                    )
                )
            ),
        )
        confirm_window_sec = max(
            0.2,
            float(
                runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_CONFIRM_WINDOW_SEC",
                    2.0,
                )
            ),
        )
        confirm_max_dev_m = max(
            0.5,
            float(
                runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_CONFIRM_MAX_DEV_M",
                    18.0,
                )
            ),
        )
        confirm_max_dir_deg = float(
            np.clip(
                runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_CONFIRM_MAX_DIR_DEG",
                    25.0,
                ),
                3.0,
                180.0,
            )
        )
        high_speed_m_s = float(
            runner.global_config.get(
                "VPS_POSITION_FIRST_DIRECT_XY_CONFIRM_HIGH_SPEED_M_S",
                18.0,
            )
        )
        high_speed_extra_hits = max(
            0,
            int(
                round(
                    runner.global_config.get(
                        "VPS_POSITION_FIRST_DIRECT_XY_CONFIRM_HIGH_SPEED_EXTRA_HITS",
                        1,
                    )
                )
            ),
        )
        required_hits = int(confirm_hits)
        if np.isfinite(speed_m_s) and float(speed_m_s) >= float(high_speed_m_s):
            required_hits += int(high_speed_extra_hits)
        if guarded_mode:
            required_hits += int(
                max(
                    0,
                    int(
                        round(
                            runner.global_config.get(
                                "VPS_POSITION_FIRST_DIRECT_XY_GUARD_EXTRA_CONFIRM_HITS",
                                1,
                            )
                        )
                    ),
                )
            )
        required_hits = max(1, int(required_hits))

        if float(dp_norm) >= float(confirm_offset_m):
            pending = getattr(runner, "_position_first_direct_confirm_pending", None)
            cur_vec = np.array([float(dp[0]), float(dp[1])], dtype=float)
            new_count = 1
            if isinstance(pending, dict):
                try:
                    prev_t = float(pending.get("t", -1e9))
                    prev_vec = np.asarray(pending.get("vec", [0.0, 0.0]), dtype=float).reshape(2,)
                    prev_count = int(pending.get("count", 0))
                except Exception:
                    prev_t = -1e9
                    prev_vec = np.array([0.0, 0.0], dtype=float)
                    prev_count = 0
                dt = float(t_cam) - float(prev_t)
                if np.isfinite(dt) and dt <= float(confirm_window_sec):
                    dev_m = float(np.linalg.norm(cur_vec - prev_vec))
                    dir_ok = True
                    prev_n = float(np.linalg.norm(prev_vec))
                    cur_n = float(np.linalg.norm(cur_vec))
                    if prev_n > 1e-6 and cur_n > 1e-6:
                        cosang = float(np.clip(np.dot(cur_vec, prev_vec) / max(1e-9, cur_n * prev_n), -1.0, 1.0))
                        dir_ok = bool(cosang >= float(np.cos(np.deg2rad(confirm_max_dir_deg))))
                    if dev_m <= float(confirm_max_dev_m) and dir_ok:
                        new_count = int(prev_count) + 1
            runner._position_first_direct_confirm_pending = {
                "t": float(t_cam),
                "vec": cur_vec.copy(),
                "count": int(new_count),
            }
            if int(new_count) < int(required_hits):
                return False, "position_first_direct_xy_confirm_wait"
        else:
            runner._position_first_direct_confirm_pending = None

        # Windowed XY budget avoids bursty cumulative injections from repeated
        # direct-lane triggers in a short time span.
        budget_window_sec = max(
            0.5,
            float(
                runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_BUDGET_WINDOW_SEC",
                    5.0,
                )
            ),
        )
        budget_max_total = max(
            1.0,
            float(
                runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_BUDGET_MAX_TOTAL_DP_XY_M",
                    max(2.0 * q_cap_floor, 35.0),
                )
            ),
        )
        if guarded_mode:
            budget_max_total = min(
                budget_max_total,
                max(
                    2.0,
                    float(
                        runner.global_config.get(
                            "VPS_POSITION_FIRST_DIRECT_XY_GUARD_BUDGET_MAX_TOTAL_DP_XY_M",
                            18.0,
                        )
                    ),
                ),
            )
        if recovery_boost:
            budget_mult = max(
                1.0,
                float(
                    runner.global_config.get(
                        "VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_BUDGET_MULT",
                        2.5,
                    )
                ),
            )
            budget_cap = max(
                budget_max_total,
                float(
                    runner.global_config.get(
                        "VPS_POSITION_FIRST_DIRECT_XY_RECOVERY_BUDGET_CAP_M",
                        120.0,
                    )
                ),
            )
            budget_max_total = min(budget_cap, budget_max_total * budget_mult)
        budget_t0 = float(getattr(runner, "_position_first_direct_budget_t0", -1e9))
        budget_used = float(getattr(runner, "_position_first_direct_budget_used_m", 0.0))
        if (float(t_cam) - budget_t0) > float(budget_window_sec):
            budget_t0 = float(t_cam)
            budget_used = 0.0
        remaining = max(0.0, float(budget_max_total) - float(budget_used))
        if remaining <= 1e-6:
            runner._position_first_direct_budget_t0 = float(budget_t0)
            runner._position_first_direct_budget_used_m = float(budget_used)
            return False, "position_first_direct_xy_budget_exhausted"
        if dp_norm > remaining and dp_norm > 1e-9:
            dp[:2] *= float(remaining / dp_norm)
            dp_norm = float(np.linalg.norm(dp[:2]))
        if guarded_mode:
            last_apply_vec = getattr(runner, "_position_first_direct_last_apply_vec", None)
            if isinstance(last_apply_vec, np.ndarray) and last_apply_vec.size >= 2:
                prev_vec = np.asarray(last_apply_vec[:2], dtype=float)
                prev_n = float(np.linalg.norm(prev_vec))
                cur_n = float(np.linalg.norm(dp[:2]))
                min_dir_check_m = float(
                    max(
                        0.5,
                        runner.global_config.get(
                            "VPS_POSITION_FIRST_DIRECT_XY_GUARD_MIN_DIR_CHECK_M",
                            8.0,
                        ),
                    )
                )
                if prev_n >= min_dir_check_m and cur_n >= min_dir_check_m:
                    cosang = float(np.clip(np.dot(prev_vec, dp[:2]) / max(1e-9, prev_n * cur_n), -1.0, 1.0))
                    max_dir_change_deg = float(
                        np.clip(
                            runner.global_config.get(
                                "VPS_POSITION_FIRST_DIRECT_XY_GUARD_MAX_DIR_CHANGE_DEG",
                                55.0,
                            ),
                            5.0,
                            180.0,
                        )
                    )
                    if cosang < float(np.cos(np.deg2rad(max_dir_change_deg))):
                        return False, "position_first_direct_xy_guard_dir_change"

        cov_scale = float(np.clip(1.0 + 0.9 * (1.0 - q), 1.0, 2.0))
        corr = BackendCorrection(
            t_ref=float(t_cam),
            dp_enu=dp,
            dyaw_deg=0.0,
            cov_scale=cov_scale,
            age_sec=0.0,
            quality_score=q,
            source_mix={"VPS": 1.0},
            residual_summary={
                "direct_xy_lane": 1.0,
                "offset_m": float(abs_offset_m),
                "offset_clamped_m": float(dp_norm),
                "offset_budget_remaining_m": float(max(0.0, remaining - dp_norm)),
                "apply_gain": float(apply_gain),
                "recovery_boost": 1.0 if recovery_boost else 0.0,
                "guarded_mode": 1.0 if guarded_mode else 0.0,
                "fail_streak": float(fail_streak),
                "no_coverage_streak": float(no_coverage_streak),
                "since_last_vps_sec": float(since_last_vps) if np.isfinite(since_last_vps) else float("nan"),
                "inliers": float(max(0, int(vps_num_inliers))),
                "confidence": float(vps_conf) if np.isfinite(vps_conf) else 0.0,
                "reproj_error": float(vps_reproj) if np.isfinite(vps_reproj) else 999.0,
            },
            contract_version="v1",
        )
        self.schedule_backend_correction(corr)
        runner._position_first_direct_last_t = float(t_cam)
        runner._position_first_direct_last_apply_vec = np.array([float(dp[0]), float(dp[1])], dtype=float)
        runner._position_first_direct_confirm_pending = None
        runner._position_first_direct_budget_t0 = float(budget_t0)
        runner._position_first_direct_budget_used_m = float(budget_used + dp_norm)
        runner._position_first_direct_xy_count = int(getattr(runner, "_position_first_direct_xy_count", 0)) + 1
        return True, "position_first_direct_xy_apply"

    def _should_use_vps_position_first_direct_xy(
        self,
        *,
        t_cam: float,
        abs_offset_vec: np.ndarray,
        abs_offset_m: float,
        speed_m_s: float,
        vps_num_inliers: int,
        vps_conf: float,
        vps_reproj: float,
        policy_mode: str,
        hard_reject_note: str,
        failsoft_selected: bool = False,
        failsoft_min_inliers: Optional[int] = None,
        failsoft_min_conf: Optional[float] = None,
        failsoft_max_reproj: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Eligibility gate for direct XY lane (position-first)."""
        runner = self.runner
        try:
            off = np.asarray(abs_offset_vec, dtype=float).reshape(-1,)
        except Exception:
            return False, "offset_vec_invalid"
        if off.size < 2 or (not np.all(np.isfinite(off[:2]))):
            return False, "offset_vec_invalid"
        if not bool(runner.global_config.get("POSITION_FIRST_LANE", False)):
            return False, "position_first_lane_off"
        if not bool(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_ENABLE", True)):
            return False, "position_first_direct_xy_disabled"
        if not np.isfinite(abs_offset_m):
            return False, "offset_nan"

        if str(policy_mode).upper() in ("HOLD", "SKIP"):
            if not bool(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_IGNORE_POLICY_HOLD", False)):
                return False, "policy_hold_skip"
        hard_reject_soft_path = False
        if hard_reject_note and not bool(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_IGNORE_HARD_REJECT", False)):
            # Logic path: allow recovery even after hard reject only when consensus
            # is explicitly enabled and later passes strict checks.
            allow_hard_reject_with_consensus = bool(
                runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_ALLOW_HARD_REJECT_WITH_CONSENSUS",
                    True,
                )
            )
            if not allow_hard_reject_with_consensus:
                return False, "hard_reject"
            if not bool(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_ENABLE", True)):
                return False, "hard_reject_no_consensus"
            hard_reject_soft_path = True

        min_offset = float(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_MIN_OFFSET_M", 8.0))
        if float(abs_offset_m) < max(0.0, min_offset):
            return False, "offset_small"
        max_offset = float(
            runner.global_config.get(
                "VPS_POSITION_FIRST_DIRECT_XY_MAX_OFFSET_M",
                runner.global_config.get("VPS_XY_DRIFT_RECOVERY_MAX_OFFSET_M", 1800.0),
            )
        )
        if np.isfinite(max_offset) and float(abs_offset_m) > max(1.0, float(max_offset)):
            return False, "offset_too_large"

        min_inliers = int(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_MIN_INLIERS", 3))
        min_conf = float(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_MIN_CONFIDENCE", 0.02))
        max_reproj = float(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_MAX_REPROJ_ERROR", 4.0))
        max_speed = float(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_MAX_SPEED_M_S", 120.0))
        high_speed_th = float(
            runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_HIGH_SPEED_TH_M_S", 18.0)
        )
        high_speed_soft_clamp_enable = bool(
            runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_HIGH_SPEED_SOFT_CLAMP_ENABLE", True)
        )
        if bool(failsoft_selected):
            # VPS matched_failsoft is already a filtered success path from runner.
            # Use a relaxed direct-lane entry gate, then rely on consensus+budget+clamp.
            fs_inl = int(failsoft_min_inliers) if failsoft_min_inliers is not None else max(3, int(min_inliers))
            fs_conf = float(failsoft_min_conf) if failsoft_min_conf is not None else float(min_conf)
            fs_repr = float(failsoft_max_reproj) if failsoft_max_reproj is not None else float(max_reproj)
            min_inliers = min(int(min_inliers), max(3, int(fs_inl)))
            min_conf = min(float(min_conf), max(0.0, float(fs_conf)))
            max_reproj = max(float(max_reproj), min(5.0, max(0.6, float(fs_repr))))
        if (
            high_speed_soft_clamp_enable
            and np.isfinite(speed_m_s)
            and float(speed_m_s) >= float(high_speed_th)
        ):
            max_offset = min(
                float(max_offset),
                float(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_MAX_OFFSET_HIGH_SPEED_M", 140.0)),
            )
            # Do not over-tighten matched_failsoft candidates at high speed here;
            # downstream deterministic caps/consensus/budget handle safety.
            if not bool(failsoft_selected):
                min_inliers = max(
                    int(min_inliers),
                    int(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_MIN_INLIERS_HIGH_SPEED", 6)),
                )
                max_reproj = min(
                    float(max_reproj),
                    float(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_MAX_REPROJ_HIGH_SPEED", 1.2)),
                )
        if int(vps_num_inliers) < max(1, min_inliers):
            return False, "quality_inliers"
        if not np.isfinite(vps_conf) or float(vps_conf) < float(min_conf):
            return False, "quality_conf"
        if not np.isfinite(vps_reproj) or float(vps_reproj) > float(max_reproj):
            return False, "quality_reproj"
        if np.isfinite(speed_m_s) and float(speed_m_s) > float(max_speed):
            return False, "speed_high"

        if bool(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_ENABLE", True)):
            cons_window = max(
                0.5,
                float(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_WINDOW_SEC", 4.0)),
            )
            cons_min_samples = max(
                1,
                int(round(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_MIN_SAMPLES", 3))),
            )
            if hard_reject_soft_path:
                cons_min_samples = max(cons_min_samples, 5)
            cons_max_dev = max(
                1.0,
                float(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_MAX_DEV_M", 35.0)),
            )
            if hard_reject_soft_path:
                cons_max_dev = min(float(cons_max_dev), 20.0)
            cons_max_dir_deg = float(
                np.clip(
                    runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_MAX_DIR_DEG", 70.0),
                    5.0,
                    180.0,
                )
            )
            if hard_reject_soft_path:
                cons_max_dir_deg = min(float(cons_max_dir_deg), 35.0)
            cons_max_keep = max(
                4,
                int(round(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_CONSENSUS_MAX_KEEP", 24))),
            )
            cur_vec = np.array([float(off[0]), float(off[1])], dtype=float)
            hist_raw = getattr(runner, "_position_first_direct_hint_hist", None)
            hist: list[tuple[float, np.ndarray]] = []
            if isinstance(hist_raw, list):
                cutoff = float(t_cam) - float(cons_window)
                for item in hist_raw:
                    try:
                        tt = float(item[0])
                        vv = np.asarray(item[1], dtype=float).reshape(2,)
                    except Exception:
                        continue
                    if np.isfinite(tt) and np.all(np.isfinite(vv)) and tt >= cutoff:
                        hist.append((tt, vv))
            if len(hist) < cons_min_samples:
                hist.append((float(t_cam), cur_vec.copy()))
                if len(hist) > cons_max_keep:
                    hist = hist[-cons_max_keep:]
                runner._position_first_direct_hint_hist = hist
                return False, "consensus_warmup"
            hist_arr = np.asarray([v for _, v in hist], dtype=float)
            med_vec = np.nanmedian(hist_arr, axis=0)
            cur_dev = float(np.linalg.norm(cur_vec - med_vec))
            if cur_dev > cons_max_dev:
                runner._position_first_direct_hint_hist = hist
                return False, "consensus_dev"
            med_norm = float(np.linalg.norm(med_vec))
            cur_norm = float(np.linalg.norm(cur_vec))
            if med_norm > 1e-6 and cur_norm > 1e-6:
                cosang = float(np.clip(np.dot(cur_vec, med_vec) / max(1e-9, cur_norm * med_norm), -1.0, 1.0))
                min_cos = float(np.cos(np.deg2rad(cons_max_dir_deg)))
                if cosang < min_cos:
                    runner._position_first_direct_hint_hist = hist
                    return False, "consensus_dir"
            hist.append((float(t_cam), cur_vec.copy()))
            if len(hist) > cons_max_keep:
                hist = hist[-cons_max_keep:]
            runner._position_first_direct_hint_hist = hist
        if hard_reject_soft_path:
            return True, "position_first_direct_xy_hard_reject_soft_recovery"
        return True, "position_first_direct_xy_eligible"

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
        deterministic_vps_sync = bool(
            runner.global_config.get("VPS_DETERMINISTIC_SYNC_ENABLE", False)
        )

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
            decision: SensorPolicyDecision = SensorPolicyDecision(sensor=str(sensor))
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
                except Exception:
                    decision = SensorPolicyDecision(sensor=str(sensor))
            # Consume-only contract: no adaptive/local fallback policy here.
            return decision, scales

        def _decision_phase(decision: SensorPolicyDecision, default_phase: int = 2) -> int:
            try:
                return int(round(float(decision.extra("phase", float(default_phase)))))
            except Exception:
                return int(default_phase)

        def _decision_health(decision: SensorPolicyDecision, default_health: str = "HEALTHY") -> str:
            try:
                return str(decision.extra_str("health_state", default_health)).upper()
            except Exception:
                return str(default_health).upper()

        def _merge_alignment_action(a: str, b: str) -> str:
            rank = {"PASS": 0, "HINT_ONLY": 1, "REJECT": 2}
            a_u = str(a or "PASS").upper()
            b_u = str(b or "PASS").upper()
            return a_u if rank.get(a_u, 0) >= rank.get(b_u, 0) else b_u

        # Process images up to current time
        while runner.state.img_idx < len(runner.imgs) and runner.imgs[runner.state.img_idx].t <= t:
            # Get camera timestamp (CRITICAL: use this instead of IMU time t)
            # Prevents timestamp mismatch when cloning/resetting preintegration
            t_cam = runner.imgs[runner.state.img_idx].t
            frame_alignment_action = "PASS"
            cam_lag = abs(float(t) - float(t_cam))
            cam_sync_threshold = float(runner.global_config.get("CAM_TIME_SYNC_THRESHOLD_SEC", 0.10))
            cam_status = "PASS" if cam_lag <= cam_sync_threshold else "WARN"
            cam_action = runner.output_reporting.log_convention_check(
                t=float(t_cam),
                sensor="CAM",
                check="imu_cam_abs_dt",
                value=float(cam_lag),
                threshold=float(cam_sync_threshold),
                status=cam_status,
                note=f"t_imu={float(t):.6f}",
            )
            frame_alignment_action = _merge_alignment_action(frame_alignment_action, cam_action)

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
                elif float(t_cam) < float(getattr(runner, "_vps_memory_pressure_until_t", -1e9)):
                    should_run_vps, skip_reason = False, "memory_pressure"
                elif worker_busy:
                    busy_until = float(getattr(runner, "_vps_busy_backpressure_until_t", -1e9))
                    if float(t_cam) < busy_until:
                        should_run_vps, skip_reason = False, "busy_backpressure"
                    else:
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
                        bp_base_sec = float(
                            runner.global_config.get("VPS_WORKER_BUSY_BACKPRESSURE_SEC", 0.40)
                        )
                        bp_max_sec = float(
                            runner.global_config.get("VPS_WORKER_BUSY_BACKPRESSURE_MAX_SEC", 2.5)
                        )
                        bp_step = max(
                            1,
                            int(runner.global_config.get("VPS_WORKER_BUSY_BACKPRESSURE_STEP", 30)),
                        )
                        busy_mult = 1.0 + float(max(0, runner._vps_thread_busy_streak - 1)) / float(bp_step)
                        busy_bp_sec = min(max(bp_base_sec, 0.05) * busy_mult, max(bp_max_sec, 0.05))
                        prev_bp_until = float(getattr(runner, "_vps_busy_backpressure_until_t", -1e9))
                        runner._vps_busy_backpressure_until_t = max(
                            prev_bp_until,
                            float(t_cam + busy_bp_sec),
                        )
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
                    elif skip_reason == "memory_pressure":
                        runner._vps_thread_busy_streak = 0
                        if (int(runner._vps_attempt_count) % 100) == 0 and bool(getattr(runner.config, "save_debug_data", False)):
                            rem = float(getattr(runner, "_vps_memory_pressure_until_t", -1e9)) - float(t_cam)
                            print(f"[VPS] memory pressure: suppress VPS submit (remaining={max(0.0, rem):.1f}s)")
                    elif skip_reason != "busy_backpressure":
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
                    if deterministic_vps_sync:
                        # Deterministic cadence lock: run VPS inline on camera tick.
                        run_vps_in_thread()
                        vps_result = runner._vps_inflight_result
                        vps_result_meta = dict(runner._vps_inflight_meta or {})
                        runner._vps_inflight_thread = None
                        runner._vps_inflight_result = None
                        runner._vps_inflight_meta = None
                    else:
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
            loop_decision, _ = _get_policy("LOOP_CLOSURE", t_cam)
            current_phase = _decision_phase(loop_decision, int(getattr(runner.state, "current_phase", 2)))
            current_health = _decision_health(loop_decision, "HEALTHY")
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
                loop_applied = apply_loop_closure_correction(
                    kf=runner.kf,
                    loop_info=match_result,
                    t=t_cam,
                    cam_states=runner.state.cam_states,
                    loop_detector=runner.loop_detector,
                    global_config=runner.global_config,
                    policy_decision=loop_decision,
                    yaw_authority_service=getattr(runner, "yaw_authority_service", None),
                )
                # Feed loop yaw as an absolute yaw hint into backend (hybrid track).
                # This is robustly fused by backend switchable constraints/huber yaw.
                if loop_applied and runner.backend_optimizer is not None:
                    try:
                        loop_inliers = float(max(0, int(match_result.get("num_inliers", 0))))
                        loop_spread = float(match_result.get("spread_ratio", 0.0))
                        loop_reproj = float(match_result.get("reproj_p95_px", np.inf))
                        q_inlier = float(np.clip(loop_inliers / 50.0, 0.0, 1.0))
                        q_spread = float(np.clip(loop_spread / 0.25, 0.0, 1.0))
                        q_reproj = float(np.clip(2.5 / max(loop_reproj, 0.25), 0.0, 1.0))
                        loop_quality = float(np.clip(0.45 * q_inlier + 0.30 * q_spread + 0.25 * q_reproj, 0.0, 1.0))
                        dyaw_deg_hint = float(np.degrees(float(match_result.get("yaw_correction", 0.0))))
                        runner.backend_optimizer.report_absolute_hint(
                            t_ref=float(t_cam),
                            dp_enu=np.zeros(3, dtype=float),
                            dyaw_deg=float(dyaw_deg_hint),
                            quality_score=loop_quality,
                            source="LOOP",
                        )
                    except Exception:
                        pass
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
                        msckf_phase = _decision_phase(msckf_decision, int(getattr(runner.state, "current_phase", 2)))
                        msckf_health = _decision_health(msckf_decision, "HEALTHY")
                        phase_chi2 = float(msckf_decision.extra("phase_chi2_scale", 1.0))
                        phase_reproj = float(msckf_decision.extra("phase_reproj_scale", 1.0))
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
                            phase=int(msckf_phase),
                            health_state=str(msckf_health),
                            adaptive_info=msckf_adaptive_info,
                            policy_decision=msckf_decision,
                            runner=runner,
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
                vio_phase = _decision_phase(vio_decision, int(getattr(runner.state, "current_phase", 2)))
                vio_health = _decision_health(vio_decision, "HEALTHY")
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
                    phase=int(vio_phase),
                    health_state=str(vio_health),
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
                    vps_sync_status = "PASS" if vps_dt <= vps_sync_threshold else "WARN"
                    vps_sync_action = runner.output_reporting.log_convention_check(
                        t=float(t_cam),
                        sensor="VPS",
                        check="cam_vps_abs_dt",
                        value=float(vps_dt),
                        threshold=float(vps_sync_threshold),
                        status=vps_sync_status,
                        note=f"vps_t={float(getattr(vps_result, 't_measurement', vps_t_ref)):.6f}",
                    )
                    frame_alignment_action = _merge_alignment_action(frame_alignment_action, vps_sync_action)

                    # Create clone manager if not exists
                    if not hasattr(runner, "vps_clone_manager"):
                        from vps import VPSDelayedUpdateManager

                        runner.vps_clone_manager = VPSDelayedUpdateManager(
                            max_delay_sec=0.5,
                            max_clones=3,
                        )

                    clone_id = str(vps_result_meta.get("clone_id", f"vps_{runner.state.img_idx}"))
                    health_key = _decision_health(vps_decision, "HEALTHY")
                    speed_now = float(np.linalg.norm(np.array(runner.kf.x[3:6, 0], dtype=float)))
                    yaw_rate_deg_s = float("nan")
                    try:
                        ang = np.asarray(getattr(rec, "ang", np.zeros(3)), dtype=float).reshape(-1,)
                        if ang.size >= 3 and np.isfinite(float(ang[2])):
                            yaw_rate_deg_s = float(abs(float(ang[2])) * 180.0 / np.pi)
                        elif ang.size > 0 and np.any(np.isfinite(ang)):
                            yaw_rate_deg_s = float(np.linalg.norm(np.nan_to_num(ang, nan=0.0)) * 180.0 / np.pi)
                    except Exception:
                        yaw_rate_deg_s = float("nan")
                    try:
                        p_abs = np.asarray(runner.kf.P, dtype=float)
                        p_max = float(np.nanmax(np.abs(p_abs))) if p_abs.size > 0 else float("nan")
                        p_cond = float("inf")
                        if p_abs.ndim == 2 and p_abs.shape[0] > 0 and p_abs.shape[1] > 0:
                            dim = min(48, int(p_abs.shape[0]))
                            p_block = 0.5 * (p_abs[:dim, :dim] + p_abs[:dim, :dim].T)
                            p_cond = float(np.linalg.cond(p_block))
                    except Exception:
                        p_max = float("nan")
                        p_cond = float("inf")
                    min_inliers_apply = int(round(float(vps_decision.extra("strict_min_inliers", 8.0))))
                    min_conf_apply = float(vps_decision.extra("strict_min_conf", 0.18))
                    max_reproj_apply = float(vps_decision.extra("strict_max_reproj", 1.2))
                    max_speed_apply = float(vps_decision.extra("strict_max_speed", 80.0))
                    if health_key == "WARNING":
                        min_inliers_apply += int(round(float(vps_decision.extra("warning_inlier_bonus", 2.0))))
                        min_conf_apply *= float(vps_decision.extra("warning_conf_mult", 1.15))
                        max_reproj_apply *= float(vps_decision.extra("warning_reproj_mult", 0.90))
                    elif health_key == "DEGRADED":
                        min_inliers_apply += int(round(float(vps_decision.extra("degraded_inlier_bonus", 4.0))))
                        min_conf_apply *= float(vps_decision.extra("degraded_conf_mult", 1.30))
                        max_reproj_apply *= float(vps_decision.extra("degraded_reproj_mult", 0.80))

                    vps_num_inliers = int(getattr(vps_result, "num_inliers", 0))
                    vps_conf = float(getattr(vps_result, "confidence", 0.0))
                    vps_reproj = float(getattr(vps_result, "reproj_error", float("inf")))
                    vps_match_reason = str(getattr(vps_result, "match_reason", "")).lower()
                    vps_match_failsoft = bool(
                        bool(getattr(vps_result, "match_is_failsoft", False))
                        or vps_match_reason == "matched_failsoft"
                    )
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
                    force_soft_only_note = ""
                    if getattr(runner, "yaw_authority_service", None) is not None:
                        try:
                            soft_only, soft_reason = runner.yaw_authority_service.should_force_soft_only(
                                source="VPS",
                                timestamp=float(t_cam),
                                speed_m_s=float(speed_now),
                                health_state=str(health_key),
                                p_max=float(p_max),
                                p_cond=float(p_cond),
                            )
                        except Exception:
                            soft_only, soft_reason = False, ""
                        if bool(soft_only):
                            strict_quality_ok = False
                            force_soft_only_note = f"soft_only_{soft_reason}" if soft_reason else "soft_only"
                    vps_offset = np.asarray(getattr(vps_result, "offset_m", (np.nan, np.nan)), dtype=float).reshape(-1)
                    vps_offset_m = float(np.linalg.norm(vps_offset[:2])) if vps_offset.size >= 2 else float("nan")
                    failsoft_enabled = bool(float(vps_decision.extra("failsoft_enabled", 1.0)) >= 0.5)
                    failsoft_allowed_state = (
                        (health_key == "HEALTHY")
                        or (health_key == "WARNING" and bool(float(vps_decision.extra("failsoft_allow_warning", 1.0)) >= 0.5))
                        or (health_key == "DEGRADED" and bool(float(vps_decision.extra("failsoft_allow_degraded", 0.0)) >= 0.5))
                    )
                    fs_min_inliers = int(round(float(vps_decision.extra("failsoft_min_inliers", 5.0))))
                    fs_min_conf = float(vps_decision.extra("failsoft_min_conf", 0.12))
                    fs_max_reproj = float(vps_decision.extra("failsoft_max_reproj", 1.2))
                    fs_max_speed = float(vps_decision.extra("failsoft_max_speed", max_speed_apply))
                    fs_max_offset_m = float(vps_decision.extra("failsoft_max_offset_m", 180.0))
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
                    msckf_q_snap = getattr(runner, "_msckf_quality_snapshot", None)
                    msckf_quality_score = float(
                        getattr(msckf_q_snap, "quality_score", np.nan)
                    ) if msckf_q_snap is not None else float("nan")
                    msckf_stable_geometry_flag = bool(
                        getattr(msckf_q_snap, "stable_geometry_flag", False)
                    ) if msckf_q_snap is not None else False
                    r_scale_apply = float(vps_policy_scales.get("r_scale", 1.0))
                    position_first_direct_xy_applied = False
                    evidence = VpsMatchEvidence(
                        t_cam=float(t_cam),
                        frame_idx=int(runner.state.img_idx),
                        policy_mode=str(getattr(vps_decision, "mode", "APPLY")),
                        health_key=str(health_key),
                        speed_m_s=float(speed_now),
                        yaw_rate_deg_s=float(yaw_rate_deg_s),
                        abs_offset_vec=np.asarray(abs_offset_vec, dtype=float).reshape(-1,),
                        abs_offset_m=float(abs_offset_m),
                        vps_num_inliers=int(vps_num_inliers),
                        vps_conf=float(vps_conf),
                        vps_reproj=float(vps_reproj),
                        strict_quality_ok=bool(strict_quality_ok),
                        failsoft_quality_ok=bool(failsoft_quality_ok),
                        vps_match_failsoft=bool(vps_match_failsoft),
                        fs_min_inliers=int(fs_min_inliers),
                        fs_min_conf=float(fs_min_conf),
                        fs_max_reproj=float(fs_max_reproj),
                        msckf_quality_score=float(msckf_quality_score),
                        msckf_stable_geometry_flag=bool(msckf_stable_geometry_flag),
                        rescue_trigger_reason=str(getattr(vps_result, "rescue_trigger_reason", "none")),
                        quality_subscores=str(getattr(vps_result, "quality_subscores", "")),
                        temporal_hits=int(
                            getattr(
                                vps_result,
                                "temporal_hits",
                                getattr(runner, "_vps_pending_large_offset_hits", 0),
                            )
                        ),
                        scale_pruned_band=str(getattr(vps_result, "scale_pruned_band", "full")),
                    )
                    pos_decision = self.vps_position_controller.decide_and_classify(evidence)
                    quality_mode = str(pos_decision.quality_mode)
                    decision_lane = str(pos_decision.decision_lane)
                    force_hint_only = bool(pos_decision.force_hint_only)
                    policy_reject_note = str(pos_decision.policy_reject_note)
                    hard_reject_note = str(pos_decision.hard_reject_note)
                    temporal_reject_note = str(pos_decision.temporal_reject_note)
                    drift_recovery_candidate = bool(pos_decision.drift_recovery_candidate)
                    drift_recovery_active = bool(pos_decision.drift_recovery_active)
                    drift_recovery_note = str(pos_decision.drift_recovery_note)
                    position_first_soft_active = bool(pos_decision.position_first_soft_active)
                    position_first_soft_note = str(pos_decision.position_first_soft_note)
                    position_first_direct_xy_candidate = bool(pos_decision.position_first_direct_xy_candidate)
                    position_first_direct_xy_note = str(pos_decision.position_first_direct_xy_note)
                    late_reclaim_active = bool(getattr(pos_decision, "late_reclaim_active", False))
                    late_reclaim_note = str(getattr(pos_decision, "late_reclaim_note", ""))
                    alignment_action = str(frame_alignment_action).upper()
                    if hasattr(runner.output_reporting, "current_alignment_action"):
                        try:
                            alignment_action = _merge_alignment_action(
                                alignment_action,
                                runner.output_reporting.current_alignment_action(float(t_cam)),
                            )
                        except Exception:
                            alignment_action = str(frame_alignment_action).upper()
                    if alignment_action == "REJECT":
                        quality_mode = "reject"
                        decision_lane = "REJECT"
                        force_hint_only = False
                        note = "alignment_lock_reject"
                        policy_reject_note = f"{policy_reject_note}|{note}" if policy_reject_note else note
                    elif alignment_action == "HINT_ONLY" and quality_mode != "reject":
                        quality_mode = "reject"
                        decision_lane = "HINT_ONLY"
                        force_hint_only = True
                        note = "alignment_lock_hint_only"
                        policy_reject_note = f"{policy_reject_note}|{note}" if policy_reject_note else note
                    if str(decision_lane) == "HINT_ONLY":
                        runner._vps_hint_only_count = int(getattr(runner, "_vps_hint_only_count", 0)) + 1
                    if str(decision_lane) == "BOUNDED_SOFT_APPLY":
                        runner._vps_bounded_soft_candidate_count = int(
                            getattr(runner, "_vps_bounded_soft_candidate_count", 0)
                        ) + 1
                    try:
                        sc = float(getattr(pos_decision, "apply_score", float("nan")))
                        if np.isfinite(sc):
                            hist = getattr(runner, "_vps_apply_score_history", None)
                            if not isinstance(hist, list):
                                hist = []
                                runner._vps_apply_score_history = hist
                            hist.append(float(sc))
                            if len(hist) > 20000:
                                del hist[:-20000]
                    except Exception:
                        pass

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
                        if force_soft_only_note:
                            vps_status = f"{force_soft_only_note} | {vps_status}"
                        if drift_recovery_note:
                            vps_status = f"{drift_recovery_note} | {vps_status}"
                        if position_first_soft_note:
                            vps_status = f"{position_first_soft_note} | {vps_status}"
                        if position_first_direct_xy_note:
                            vps_status = f"{position_first_direct_xy_note} | {vps_status}"
                        hint_quality: Optional[float] = None
                        if np.isfinite(float(pos_decision.hint_quality)):
                            hint_quality = float(pos_decision.hint_quality)
                        if (
                            bool(drift_recovery_candidate)
                            and runner.backend_optimizer is not None
                            and bool(
                                runner.global_config.get(
                                    "VPS_XY_DRIFT_RECOVERY_REPORT_BACKEND_HINT_ON_REJECT",
                                    True,
                                )
                            )
                        ):
                            try:
                                if hint_quality is None:
                                    hint_quality = 0.0
                                runner.backend_optimizer.report_absolute_hint(
                                    t_ref=float(t_cam),
                                    dp_enu=np.array([abs_offset_vec[0], abs_offset_vec[1], 0.0], dtype=float),
                                    dyaw_deg=0.0,
                                    quality_score=float(hint_quality),
                                    source="VPS",
                                )
                                vps_status = f"{vps_status} | XY_DRIFT_RECOVERY_HINT_ONLY"
                            except Exception:
                                pass
                        allow_direct_xy_apply = bool(pos_decision.allow_direct_xy_apply) and (not bool(force_hint_only))
                        if bool(allow_direct_xy_apply) and hint_quality is not None:
                            try:
                                direct_ok, direct_note = self.vps_position_controller.apply_direct_xy(
                                    evidence=evidence,
                                    base_quality=float(hint_quality),
                                )
                                if direct_ok:
                                    position_first_direct_xy_applied = True
                                    vps_status = f"{vps_status} | POSITION_FIRST_DIRECT_XY_APPLIED"
                                elif str(direct_note).strip():
                                    vps_status = f"{vps_status} | {direct_note}"
                            except Exception:
                                pass
                    else:
                        if quality_mode == "failsoft":
                            r_scale_apply *= float(vps_decision.extra("failsoft_r_mult", 1.5))
                            if health_key == "WARNING":
                                r_scale_apply *= 1.25
                            elif health_key == "DEGRADED":
                                r_scale_apply *= 1.5
                            if drift_recovery_active:
                                r_scale_apply *= float(
                                    runner.global_config.get("VPS_XY_DRIFT_RECOVERY_R_MULT", 2.8)
                                )
                            if position_first_soft_active:
                                r_scale_apply *= float(
                                    runner.global_config.get("VPS_POSITION_FIRST_SOFT_R_MULT", 4.0)
                                )
                            if str(decision_lane) == "BOUNDED_SOFT_APPLY":
                                r_scale_apply *= float(
                                    runner.global_config.get("VPS_APPLY_GATE_BOUNDED_SOFT_R_MULT", 2.6)
                                )
                                if bool(late_reclaim_active):
                                    r_scale_apply *= float(
                                        runner.global_config.get(
                                            "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_RECLAIM_R_MULT",
                                            1.8,
                                        )
                                    )
                        vps_lat_apply = float(vps_result.lat)
                        vps_lon_apply = float(vps_result.lon)
                        clamp_scale = 1.0
                        if quality_mode == "failsoft":
                            clamp_override = None
                            if drift_recovery_active:
                                clamp_override = float(
                                    runner.global_config.get(
                                        "VPS_XY_DRIFT_RECOVERY_MAX_APPLY_DP_XY_M",
                                        runner.global_config.get("VPS_ABS_MAX_APPLY_DP_XY_M", 25.0),
                                    )
                                )
                            if position_first_soft_active:
                                clamp_override = float(
                                    runner.global_config.get(
                                        "VPS_POSITION_FIRST_SOFT_MAX_APPLY_DP_XY_M",
                                        runner.global_config.get(
                                            "VPS_XY_DRIFT_RECOVERY_MAX_APPLY_DP_XY_M",
                                            runner.global_config.get("VPS_ABS_MAX_APPLY_DP_XY_M", 25.0),
                                        ),
                                    )
                                )
                            if str(decision_lane) == "BOUNDED_SOFT_APPLY":
                                bounded_clamp = float(getattr(pos_decision, "bounded_clamp_m", np.nan))
                                if np.isfinite(bounded_clamp) and bounded_clamp > 0.0:
                                    if clamp_override is None:
                                        clamp_override = float(bounded_clamp)
                                    else:
                                        clamp_override = min(float(clamp_override), float(bounded_clamp))
                            vps_lat_apply, vps_lon_apply, clamp_scale = self._clamp_vps_latlon(
                                current_xy=current_xy,
                                vps_lat=vps_lat_apply,
                                vps_lon=vps_lon_apply,
                                max_apply_dp_xy_override=clamp_override,
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
                        if drift_recovery_active:
                            vps_status = f"{vps_status} | XY_DRIFT_RECOVERY"
                        if position_first_soft_active:
                            vps_status = f"{vps_status} | POSITION_FIRST_SOFT"
                        if str(decision_lane) == "BOUNDED_SOFT_APPLY":
                            vps_status = f"{vps_status} | BOUNDED_SOFT_APPLY"
                        if bool(late_reclaim_active) and str(late_reclaim_note).strip():
                            vps_status = f"{vps_status} | {late_reclaim_note}"
                        bounded_missing_clone_reclaim = bool(
                            str(decision_lane) == "BOUNDED_SOFT_APPLY"
                            and bool(
                                runner.global_config.get(
                                    "VPS_APPLY_GATE_BOUNDED_SOFT_MISSING_CLONE_DIRECT_FALLBACK_ENABLE",
                                    True,
                                )
                            )
                            and ("clone" in str(vps_status).lower())
                        )
                        if (
                            not bool(vps_applied)
                            and bool(quality_mode == "failsoft")
                            and (
                                bool(position_first_soft_active)
                                or bool(drift_recovery_active)
                                or bool(position_first_direct_xy_candidate)
                                or bool(bounded_missing_clone_reclaim)
                            )
                            and bool(runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_FALLBACK_ON_APPLY_FAIL", True))
                        ):
                            try:
                                fallback_quality = float(
                                    np.clip(
                                        0.45 * np.clip(vps_conf, 0.0, 1.0)
                                        + 0.35 * np.clip(vps_num_inliers / 80.0, 0.0, 1.0)
                                        + 0.20 * np.clip(1.2 / max(vps_reproj, 0.1), 0.0, 1.0),
                                        0.0,
                                        1.0,
                                    )
                                )
                                direct_ok, direct_note = self.vps_position_controller.apply_direct_xy(
                                    evidence=evidence,
                                    base_quality=float(fallback_quality),
                                )
                                if direct_ok:
                                    position_first_direct_xy_applied = True
                                    vps_status = f"{vps_status} | POSITION_FIRST_DIRECT_XY_APPLIED"
                                elif str(direct_note).strip():
                                    vps_status = f"{vps_status} | {direct_note}"
                            except Exception:
                                pass
                    reason_code = self.vps_position_controller.resolve_reason_code(
                        vps_applied=bool(vps_applied),
                        quality_mode=str(quality_mode),
                        vps_status=str(vps_status),
                        position_first_direct_xy_applied=bool(position_first_direct_xy_applied),
                        drift_recovery_active=bool(drift_recovery_active),
                        position_first_soft_active=bool(position_first_soft_active),
                    )
                    nis_norm_vps = np.nan
                    try:
                        self.vps_position_controller.log_trace(
                            evidence=evidence,
                            decision=pos_decision,
                            applied=bool(vps_applied or position_first_direct_xy_applied),
                            reason_code=str(reason_code),
                        )
                    except Exception:
                        pass
                    vps_adaptive_info.update({
                        "sensor": "VPS",
                        "accepted": bool(vps_applied or position_first_direct_xy_applied),
                        "attempted": 1,
                        "dof": 2,
                        "nis_norm": nis_norm_vps,
                        "chi2": float(vps_innovation_m) if vps_innovation_m is not None and np.isfinite(float(vps_innovation_m)) else np.nan,
                        "threshold": np.nan,
                        "r_scale_used": float(r_scale_apply),
                        "reason_code": reason_code,
                        "decision_lane": str(decision_lane),
                        "apply_score": float(getattr(pos_decision, "apply_score", np.nan)),
                    })
                    runner.adaptive_service.record_adaptive_measurement(
                        "VPS",
                        adaptive_info=vps_adaptive_info,
                        timestamp=float(t_cam),
                        policy_scales=vps_policy_scales,
                    )
                    if self.vps_position_controller.is_soft_accept_reason(str(reason_code)):
                        runner._vps_soft_accept_count = int(getattr(runner, "_vps_soft_accept_count", 0)) + 1
                    elif self.vps_position_controller.is_soft_reject_reason(str(reason_code)):
                        runner._vps_soft_reject_count = int(getattr(runner, "_vps_soft_reject_count", 0)) + 1
                    if vps_applied or position_first_direct_xy_applied:
                        runner._abs_corr_apply_count = int(getattr(runner, "_abs_corr_apply_count", 0)) + 1
                        if quality_mode == "failsoft" or position_first_direct_xy_applied:
                            runner._abs_corr_soft_count = int(getattr(runner, "_abs_corr_soft_count", 0)) + 1
                        if str(decision_lane) == "BOUNDED_SOFT_APPLY":
                            runner._vps_bounded_soft_apply_count = int(
                                getattr(runner, "_vps_bounded_soft_apply_count", 0)
                            ) + 1
                            runner._vps_bounded_soft_last_apply_time = float(t_cam)
                        if bool(vps_match_failsoft):
                            runner._vps_failsoft_applied_count = int(
                                getattr(runner, "_vps_failsoft_applied_count", 0)
                            ) + 1
                        runner.last_vps_update_time = t_cam
                        runner.state.vps_idx += 1
                        runner._vps_pending_large_offset_vec = None
                        runner._vps_pending_large_offset_hits = 0
                        if abs_offset_vec.size >= 2:
                            runner._vps_last_accepted_offset_vec = np.array(abs_offset_vec[:2], dtype=float)
                        runner._vps_last_accepted_offset_count = int(
                            getattr(runner, "_vps_last_accepted_offset_count", 0)
                        ) + 1
                        if vps_applied and runner.backend_optimizer is not None:
                            try:
                                use_vps_yaw_hint = bool(runner.global_config.get("BACKEND_VPS_YAW_HINT_ENABLE", True))
                                if drift_recovery_active or position_first_soft_active:
                                    use_vps_yaw_hint = False
                                yaw_hint_gain = float(runner.global_config.get("BACKEND_VPS_YAW_HINT_GAIN", 0.35))
                                yaw_hint_max_abs_deg = float(runner.global_config.get("BACKEND_VPS_YAW_HINT_MAX_ABS_DEG", 35.0))
                                yaw_hint_min_quality = float(runner.global_config.get("BACKEND_VPS_YAW_HINT_MIN_QUALITY", 0.45))
                                yaw_hint_min_inliers = int(runner.global_config.get("BACKEND_VPS_YAW_HINT_MIN_INLIERS", 8))
                                yaw_hint_max_reproj = float(runner.global_config.get("BACKEND_VPS_YAW_HINT_MAX_REPROJ", 1.2))
                                yaw_hint_max_apply = float(runner.global_config.get("BACKEND_VPS_YAW_HINT_MAX_APPLY_DEG", 2.0))
                                dyaw_deg_hint = 0.0
                                if use_vps_yaw_hint:
                                    # Use VPS result payload directly; keep robust fallbacks.
                                    raw_yaw_hint = float(getattr(vps_result, "yaw_delta_deg", np.nan))
                                    raw_yaw_q = float(getattr(vps_result, "yaw_hint_quality", np.nan))
                                    if (
                                        np.isfinite(raw_yaw_hint)
                                        and np.isfinite(raw_yaw_q)
                                        and abs(raw_yaw_hint) <= max(1.0, yaw_hint_max_abs_deg)
                                        and float(raw_yaw_q) >= float(np.clip(yaw_hint_min_quality, 0.0, 1.0))
                                        and int(vps_num_inliers) >= int(max(1, yaw_hint_min_inliers))
                                        and np.isfinite(float(vps_reproj))
                                        and float(vps_reproj) <= float(max(0.05, yaw_hint_max_reproj))
                                    ):
                                        dyaw_deg_hint = float(
                                            np.clip(
                                                float(raw_yaw_hint) * float(max(0.0, yaw_hint_gain)),
                                                -max(0.05, yaw_hint_max_apply),
                                                +max(0.05, yaw_hint_max_apply),
                                            )
                                        )
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
                                    dyaw_deg=float(dyaw_deg_hint),
                                    quality_score=quality_score,
                                    source="VPS",
                                )
                            except Exception as exc:
                                if bool(getattr(runner.config, "save_debug_data", False)):
                                    print(f"[BACKEND] report_absolute_hint failed: {exc}")
                    if (
                        vps_decision is not None
                        and str(getattr(vps_decision, "mode", "APPLY")).upper() in ("HOLD", "SKIP")
                        and bool(vps_applied)
                        and getattr(runner, "policy_runtime_service", None) is not None
                    ):
                        hold_bypass_apply = bool(
                            str(decision_lane) == "BOUNDED_SOFT_APPLY"
                            and str(policy_reject_note)
                            in (
                                "policy_mode_hold_bypass_bounded_soft",
                                "policy_mode_hold_reclaim_bounded_soft",
                            )
                        )
                        if bool(hold_bypass_apply):
                            runner._vps_hold_bypass_apply_count = int(
                                getattr(runner, "_vps_hold_bypass_apply_count", 0)
                            ) + 1
                        else:
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

    def _validate_backend_contract_v1(
        self, corr: Any
    ) -> tuple[Optional[BackendCorrectionContractV1], str]:
        """Validate backend correction payload against Contract V1."""
        runner = self.runner
        strict_source_mix = bool(runner.global_config.get("BACKEND_CONTRACT_STRICT_REQUIRE_SOURCE_MIX", True))
        strict_residual_summary = bool(
            runner.global_config.get("BACKEND_CONTRACT_STRICT_REQUIRE_RESIDUAL_SUMMARY", True)
        )
        strict_contract_version = bool(
            runner.global_config.get(
                "BACKEND_CONTRACT_V1_STRICT_REQUIRE_VERSION",
                (
                    bool(runner.global_config.get("BACKEND_CONTRACT_STRICT_REQUIRE_SOURCE_MIX", True))
                    or bool(runner.global_config.get("BACKEND_CONTRACT_STRICT_REQUIRE_RESIDUAL_SUMMARY", True))
                ),
            )
        )
        expected_version = str(
            runner.global_config.get("BACKEND_CONTRACT_V1_EXPECTED_VERSION", "v1")
        ).strip()

        contract, note = BackendCorrectionContractV1.from_payload(
            corr,
            strict_version=bool(strict_contract_version),
            expected_version=str(expected_version),
            require_source_mix=bool(strict_source_mix),
            require_residual_summary=bool(strict_residual_summary),
        )
        if contract is None:
            return None, str(note)
        return contract, ""

    def _backend_reject_counts(self) -> Dict[str, int]:
        """Single-source reason counter map for backend correction rejects."""
        runner = self.runner
        counts = getattr(runner, "_backend_reject_reason_counts", None)
        if not isinstance(counts, dict):
            counts = {}
            runner._backend_reject_reason_counts = counts
        return counts

    def _backend_source_reliability_map(self) -> Dict[str, float]:
        """Return mutable source reliability map with defaults initialized."""
        runner = self.runner
        rel_map = getattr(runner, "_backend_source_reliability", None)
        if not isinstance(rel_map, dict):
            rel_map = {}
            runner._backend_source_reliability = rel_map
        default_rel = float(np.clip(runner.global_config.get("BACKEND_SOURCE_RELIABILITY_DEFAULT", 0.50), 0.0, 1.0))
        for key in ("VPS", "LOOP", "BACKEND", "MAG"):
            if key not in rel_map or not np.isfinite(float(rel_map.get(key, np.nan))):
                rel_map[key] = default_rel
        return rel_map

    def _backend_source_reliability_hist_map(self) -> Dict[str, list[float]]:
        """Return mutable source reliability history map."""
        runner = self.runner
        hist_map = getattr(runner, "_backend_source_reliability_hist", None)
        if not isinstance(hist_map, dict):
            hist_map = {}
            runner._backend_source_reliability_hist = hist_map
        for key in ("VPS", "LOOP", "BACKEND", "MAG"):
            if not isinstance(hist_map.get(key, None), list):
                hist_map[key] = []
        return hist_map

    def _backend_update_source_reliability(
        self,
        *,
        source: str,
        quality_score: float,
        residual_xy: float,
        committed: bool,
    ) -> float:
        """Update rolling source reliability after commit/reject decisions."""
        runner = self.runner
        source_u = str(source or "BACKEND").upper()
        rel_map = self._backend_source_reliability_map()
        hist_map = self._backend_source_reliability_hist_map()
        prev = float(np.clip(rel_map.get(source_u, 0.5), 0.0, 1.0))
        if not bool(runner.global_config.get("BACKEND_SOURCE_RELIABILITY_ENABLE", True)):
            hist = hist_map.setdefault(source_u, [])
            hist.append(prev)
            if len(hist) > 20000:
                del hist[:-20000]
            return prev

        alpha = float(np.clip(runner.global_config.get("BACKEND_SOURCE_RELIABILITY_ALPHA", 0.10), 0.0, 1.0))
        penalty = float(np.clip(runner.global_config.get("BACKEND_SOURCE_RELIABILITY_REJECT_PENALTY", 0.12), 0.0, 1.0))
        residual_ref_m = float(max(0.1, runner.global_config.get("BACKEND_SOURCE_RELIABILITY_RESIDUAL_REF_M", 12.0)))
        q = float(np.clip(quality_score, 0.0, 1.0)) if np.isfinite(float(quality_score)) else 0.0
        r = float(residual_xy) if np.isfinite(float(residual_xy)) else residual_ref_m
        obs = float(np.clip(q * np.clip(1.0 / (1.0 + max(0.0, r) / residual_ref_m), 0.0, 1.0), 0.0, 1.0))
        if committed:
            new_rel = float(np.clip((1.0 - alpha) * prev + alpha * obs, 0.0, 1.0))
        else:
            new_rel = float(np.clip(prev * (1.0 - penalty), 0.0, 1.0))
        rel_map[source_u] = new_rel
        hist = hist_map.setdefault(source_u, [])
        hist.append(new_rel)
        if len(hist) > 20000:
            del hist[:-20000]
        return new_rel

    def _log_backend_apply_trace(self, record: ApplyDecisionRecord) -> None:
        """Append one deterministic backend apply funnel trace row."""
        runner = self.runner
        if not bool(runner.global_config.get("BACKEND_APPLY_TRACE_ENABLE", True)):
            return
        trace_csv = getattr(runner, "backend_apply_trace_csv", None)
        if not trace_csv:
            return
        try:
            with open(trace_csv, "a", newline="") as f:
                f.write(
                    f"{float(record.timestamp):.6f},{str(record.source)},"
                    f"{str(record.decision_state)},{str(record.reason).replace(',', ';')},"
                    f"{float(record.quality_score):.6f},{float(record.residual_xy):.6f},"
                    f"{float(record.dp_xy_in):.6f},{float(record.dp_xy_applied):.6f},"
                    f"{float(record.age_sec):.6f},{float(record.t_ref):.6f},"
                    f"{int(bool(record.time_aligned_used))}\n"
                )
        except Exception:
            pass

    def _enqueue_backend_pending(
        self,
        *,
        dp: np.ndarray,
        dyaw_deg: float,
        blend_steps: int,
        corr_quality: float,
        corr_age_sec: float,
        residual_xy_metric: float,
        source: str,
        t_ref: float,
        time_aligned_used: bool,
        reason: str,
    ) -> None:
        """Commit backend correction payload into pending blend queue."""
        runner = self.runner
        runner._backend_pending_dp_enu = np.asarray(dp, dtype=float).reshape(3,).copy()
        runner._backend_pending_dyaw_deg = float(dyaw_deg)
        runner._backend_pending_steps_left = int(max(1, int(blend_steps)))
        runner._backend_pending_quality = float(np.clip(corr_quality, 0.0, 1.0))
        runner._backend_pending_source = str(source or "BACKEND").upper()
        runner._backend_pending_t_ref = float(t_ref) if np.isfinite(float(t_ref)) else float("nan")
        runner._backend_pending_age_sec = float(corr_age_sec) if np.isfinite(float(corr_age_sec)) else float("nan")
        runner._backend_pending_residual_xy = float(residual_xy_metric) if np.isfinite(float(residual_xy_metric)) else float("nan")
        runner._backend_pending_dp_xy_in = float(np.linalg.norm(np.asarray(dp[:2], dtype=float)))
        runner._backend_pending_reason = str(reason)
        runner._backend_pending_time_aligned_used = bool(time_aligned_used)

    def _process_backend_probation(self, t_now: float) -> None:
        """Advance PROBATION state and commit/reject deterministically."""
        runner = self.runner
        candidate = getattr(runner, "_backend_probation_candidate", None)
        if not isinstance(candidate, dict):
            return
        # Do not evaluate probation while pending correction is still blending.
        if int(getattr(runner, "_backend_pending_steps_left", 0)) > 0:
            return
        min_interval = float(max(0.01, runner.global_config.get("BACKEND_SUPERVISOR_PROBATION_MIN_INTERVAL_SEC", 0.08)))
        last_eval_t = float(candidate.get("last_eval_t", -1e9))
        if np.isfinite(float(t_now)) and (float(t_now) - last_eval_t) < min_interval:
            return
        candidate["last_eval_t"] = float(t_now)
        ticks_seen = int(candidate.get("ticks_seen", 0)) + 1
        candidate["ticks_seen"] = ticks_seen
        required_ticks = int(max(1, runner.global_config.get("BACKEND_SUPERVISOR_PROBATION_TICKS", 2)))

        ok_now = True
        fail_reason = ""
        magnitude_consistency_pass = True
        direction_consistency_pass = True
        dp = np.asarray(candidate.get("dp", np.zeros(3, dtype=float)), dtype=float).reshape(3,)
        dp_xy_norm = float(np.linalg.norm(dp[:2]))
        source = str(candidate.get("source", "BACKEND")).upper()
        quality = float(candidate.get("corr_quality", np.nan))
        source_rel = float(self._backend_source_reliability_map().get(source, 0.5))
        counts = self._backend_reject_counts()

        def _shape_probation_payload(
            dp_in: np.ndarray,
            dyaw_in_deg: float,
            blend_steps_in: int,
            *,
            bounded: bool,
        ) -> tuple[np.ndarray, float, int, bool]:
            """Deterministic energy shaping for probation commits."""
            dp_shaped = np.asarray(dp_in, dtype=float).reshape(3,).copy()
            dyaw_shaped = float(dyaw_in_deg)
            shaped = False
            if bounded:
                max_total_dp_xy = float(
                    max(
                        0.1,
                        runner.global_config.get(
                            "BACKEND_SUPERVISOR_BOUNDED_COMMIT_MAX_DP_XY_M",
                            3.0,
                        ),
                    )
                )
                step_cap_m = float(
                    max(
                        0.1,
                        runner.global_config.get(
                            "BACKEND_SUPERVISOR_BOUNDED_COMMIT_STEP_CAP_M",
                            0.9,
                        ),
                    )
                )
                min_blend_steps = int(
                    max(
                        1,
                        runner.global_config.get(
                            "BACKEND_SUPERVISOR_BOUNDED_COMMIT_MIN_BLEND_STEPS",
                            6,
                        ),
                    )
                )
                max_dyaw_deg = float(
                    max(
                        0.01,
                        runner.global_config.get(
                            "BACKEND_SUPERVISOR_BOUNDED_COMMIT_MAX_DYAW_DEG",
                            0.35,
                        ),
                    )
                )
            else:
                max_total_dp_xy = float(
                    max(
                        0.1,
                        runner.global_config.get(
                            "BACKEND_SUPERVISOR_COMMIT_MAX_DP_XY_M",
                            10.0,
                        ),
                    )
                )
                step_cap_m = float(
                    max(
                        0.1,
                        runner.global_config.get(
                            "BACKEND_SUPERVISOR_COMMIT_STEP_CAP_M",
                            2.0,
                        ),
                    )
                )
                min_blend_steps = int(
                    max(
                        1,
                        runner.global_config.get(
                            "BACKEND_SUPERVISOR_COMMIT_MIN_BLEND_STEPS",
                            4,
                        ),
                    )
                )
                max_dyaw_deg = float(
                    max(
                        0.01,
                        runner.global_config.get(
                            "BACKEND_SUPERVISOR_COMMIT_MAX_DYAW_DEG",
                            1.0,
                        ),
                    )
                )

            dp_xy_local = float(np.linalg.norm(dp_shaped[:2]))
            if np.isfinite(dp_xy_local) and dp_xy_local > max_total_dp_xy and max_total_dp_xy > 1e-6:
                scale = float(np.clip(max_total_dp_xy / max(dp_xy_local, 1e-9), 0.0, 1.0))
                dp_shaped[:2] *= scale
                dyaw_shaped *= scale
                shaped = True
            if np.isfinite(dyaw_shaped) and abs(dyaw_shaped) > max_dyaw_deg:
                dyaw_shaped = float(np.clip(dyaw_shaped, -max_dyaw_deg, max_dyaw_deg))
                shaped = True

            dp_xy_local = float(np.linalg.norm(dp_shaped[:2]))
            blend_steps_out = int(max(1, int(blend_steps_in)))
            if step_cap_m > 1e-6 and np.isfinite(dp_xy_local) and dp_xy_local > 1e-9:
                blend_steps_out = int(max(blend_steps_out, int(np.ceil(dp_xy_local / step_cap_m))))
            blend_steps_out = int(max(blend_steps_out, int(min_blend_steps)))
            if blend_steps_out > int(max(1, int(blend_steps_in))):
                shaped = True
            return dp_shaped, float(dyaw_shaped), int(blend_steps_out), bool(shaped)

        # C4.1 (logic-first): dynamic magnitude consistency in probation.
        # Clamp oversized short-horizon correction energy before commit decision
        # so throughput is preserved while suppressing burst tails.
        try:
            vel_xy_mag = float(
                np.linalg.norm(np.asarray(runner.kf.x[3:5, 0], dtype=float).reshape(2,))
            )
        except Exception:
            vel_xy_mag = 0.0
        cand_age = float(candidate.get("corr_age_sec", np.nan))
        prob_horizon_sec = float(
            max(
                0.05,
                runner.global_config.get(
                    "BACKEND_SUPERVISOR_MAGNITUDE_HORIZON_SEC",
                    0.8,
                ),
            )
        )
        if np.isfinite(cand_age):
            prob_horizon_sec = float(
                np.clip(
                    max(prob_horizon_sec, cand_age + 0.05),
                    0.05,
                    max(
                        prob_horizon_sec,
                        float(
                            runner.global_config.get(
                                "BACKEND_SUPERVISOR_MAGNITUDE_HORIZON_MAX_SEC",
                                2.0,
                            )
                        ),
                    ),
                )
            )
        prob_mag_base_m = float(
            max(0.0, runner.global_config.get("BACKEND_SUPERVISOR_MAGNITUDE_BASE_M", 8.0))
        )
        prob_mag_speed_gain = float(
            max(0.0, runner.global_config.get("BACKEND_SUPERVISOR_MAGNITUDE_SPEED_GAIN", 1.2))
        )
        prob_mag_max_m = float(
            max(0.1, runner.global_config.get("BACKEND_SUPERVISOR_MAGNITUDE_MAX_M", 35.0))
        )
        prob_max_allow_m = float(
            min(
                prob_mag_max_m,
                prob_mag_base_m + prob_mag_speed_gain * max(0.0, vel_xy_mag) * prob_horizon_sec,
            )
        )
        probation_mag_clamp_enable = bool(
            runner.global_config.get("BACKEND_SUPERVISOR_MAGNITUDE_CLAMP_ENABLE", True)
        )
        if np.isfinite(dp_xy_norm) and dp_xy_norm > prob_max_allow_m and prob_max_allow_m > 1e-6:
            magnitude_consistency_pass = False
            if probation_mag_clamp_enable:
                scale = float(np.clip(prob_max_allow_m / max(dp_xy_norm, 1e-9), 0.0, 1.0))
                dp[:2] *= scale
                candidate["dp"] = np.asarray(dp, dtype=float).reshape(3,).copy()
                dyaw_prev = float(candidate.get("dyaw_deg", 0.0))
                candidate["dyaw_deg"] = float(dyaw_prev * scale)
                candidate["probation_clamped"] = True
                dp_xy_norm = float(np.linalg.norm(dp[:2]))
                counts["kinematic_clamp_magnitude"] = int(
                    counts.get("kinematic_clamp_magnitude", 0)
                ) + 1
                runner._backend_kinematic_budget_clamp_count = int(
                    getattr(runner, "_backend_kinematic_budget_clamp_count", 0)
                ) + 1
            else:
                ok_now = False
                fail_reason = "kinematic_reject"

        # Motion consistency check in probation window.
        try:
            vel_xy = np.asarray(runner.kf.x[3:5, 0], dtype=float).reshape(2,)
        except Exception:
            vel_xy = np.zeros(2, dtype=float)
        speed_xy = float(np.linalg.norm(vel_xy))
        min_speed = float(runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_MIN_SPEED_M_S", 8.0))
        min_dp_xy = float(runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_MIN_DP_XY_M", 4.0))
        min_cos = float(runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_MIN_COS", -0.25))
        if (
            np.isfinite(speed_xy)
            and np.isfinite(dp_xy_norm)
            and speed_xy >= max(0.0, min_speed)
            and dp_xy_norm >= max(0.0, min_dp_xy)
            and speed_xy > 1e-6
            and dp_xy_norm > 1e-6
        ):
            cos_dir = float(np.dot(np.asarray(dp[:2], dtype=float), vel_xy) / (speed_xy * dp_xy_norm))
            if np.isfinite(cos_dir) and cos_dir < float(min_cos):
                direction_consistency_pass = False
                ok_now = False
                fail_reason = "kinematic_reject"

        # Innovation evidence check in probation window.
        # Require either: (a) residual improvement vs running baseline, or
        # (b) bounded proxy evidence for sparse/noisy residual cases.
        residual_xy = float(candidate.get("residual_xy_metric", np.nan))
        improve_ratio_max = float(max(0.05, runner.global_config.get("BACKEND_SUPERVISOR_INNOVATION_IMPROVE_RATIO_MAX", 1.05)))
        innovation_evidence = False
        residual_eval_metric = float("nan")
        if ok_now:
            hist = np.asarray(getattr(runner, "_backend_apply_residual_xy_history", []), dtype=float)
            hist = hist[np.isfinite(hist) & (hist > 1e-6)]
            baseline = float(np.percentile(hist, 50)) if hist.size > 0 else float("nan")
            if np.isfinite(residual_xy) and residual_xy > 1e-6:
                if np.isfinite(baseline) and baseline > 1e-9:
                    innovation_evidence = bool(float(residual_xy) <= float(improve_ratio_max * baseline))
                    residual_eval_metric = float(residual_xy / baseline)
                else:
                    abs_residual_cap = float(
                        max(
                            0.1,
                            runner.global_config.get(
                                "BACKEND_SUPERVISOR_RESIDUAL_ABS_MAX_M",
                                12.0,
                            ),
                        )
                    )
                    innovation_evidence = bool(float(residual_xy) <= abs_residual_cap)
                    residual_eval_metric = float(residual_xy)

            # Sparse/no-residual fallback: bounded proxy evidence.
            proxy_evidence_enable = bool(
                runner.global_config.get("BACKEND_SUPERVISOR_PROXY_EVIDENCE_ENABLE", True)
            )
            if proxy_evidence_enable and not innovation_evidence:
                proxy_ref_m = float(
                    max(0.1, runner.global_config.get("BACKEND_SUPERVISOR_PROXY_REF_M", 12.0))
                )
                proxy_max = float(
                    max(0.05, runner.global_config.get("BACKEND_SUPERVISOR_PROXY_MAX", 1.0))
                )
                proxy_min_quality = float(
                    np.clip(
                        runner.global_config.get("BACKEND_SUPERVISOR_PROXY_MIN_QUALITY", 0.35),
                        0.0,
                        1.0,
                    )
                )
                proxy_min_source_rel = float(
                    np.clip(
                        runner.global_config.get("BACKEND_SUPERVISOR_PROXY_MIN_SOURCE_REL", 0.30),
                        0.0,
                        1.0,
                    )
                )
                proxy_max_speed = float(
                    max(
                        0.0,
                        runner.global_config.get(
                            "BACKEND_SUPERVISOR_PROXY_MAX_SPEED_M_S",
                            1.0e9,
                        ),
                    )
                )
                proxy_speed_ok = bool(
                    (not np.isfinite(speed_xy))
                    or float(speed_xy) <= float(proxy_max_speed)
                )
                proxy_norm = float(dp_xy_norm / proxy_ref_m) if np.isfinite(dp_xy_norm) else float("nan")
                innovation_evidence = bool(
                    proxy_speed_ok
                    and
                    np.isfinite(proxy_norm)
                    and proxy_norm <= proxy_max
                    and np.isfinite(quality)
                    and float(quality) >= proxy_min_quality
                    and np.isfinite(source_rel)
                    and float(source_rel) >= proxy_min_source_rel
                )
                if (not np.isfinite(residual_eval_metric)) and np.isfinite(proxy_norm):
                    residual_eval_metric = float(proxy_norm)
                if (not innovation_evidence) and (not proxy_speed_ok):
                    counts["proxy_speed_block"] = int(
                        counts.get("proxy_speed_block", 0)
                    ) + 1

            require_evidence = bool(
                runner.global_config.get("BACKEND_SUPERVISOR_REQUIRE_EVIDENCE", True)
            )
            if require_evidence and not innovation_evidence:
                ok_now = False
                fail_reason = "quality_reject"
                runner._backend_probation_evidence_fail_count = int(
                    getattr(runner, "_backend_probation_evidence_fail_count", 0)
                ) + 1

        residual_eval_history = candidate.get("residual_eval_history", [])
        if not isinstance(residual_eval_history, list):
            residual_eval_history = []
        if np.isfinite(residual_eval_metric):
            residual_eval_history.append(float(residual_eval_metric))
            if len(residual_eval_history) > 16:
                del residual_eval_history[:-16]
        candidate["residual_eval_history"] = residual_eval_history

        residual_monotonic_enable = bool(
            runner.global_config.get(
                "BACKEND_SUPERVISOR_RESIDUAL_MONOTONIC_COMMIT_ENABLE",
                False,
            )
        )
        residual_monotonic_ticks = int(
            max(
                2,
                runner.global_config.get(
                    "BACKEND_SUPERVISOR_RESIDUAL_MONOTONIC_TICKS",
                    2,
                ),
            )
        )
        residual_monotonic_min_drop = float(
            max(
                0.0,
                runner.global_config.get(
                    "BACKEND_SUPERVISOR_RESIDUAL_MONOTONIC_MIN_DROP",
                    0.0,
                ),
            )
        )
        residual_monotonic_ready = False
        residual_monotonic_pass = True
        if residual_monotonic_enable:
            if len(residual_eval_history) >= residual_monotonic_ticks:
                window = np.asarray(
                    residual_eval_history[-residual_monotonic_ticks:],
                    dtype=float,
                )
                if np.all(np.isfinite(window)):
                    residual_monotonic_ready = True
                    deltas = np.diff(window)
                    residual_monotonic_pass = bool(
                        np.all(
                            deltas
                            <= (-float(residual_monotonic_min_drop) + 1e-12)
                        )
                    )
                else:
                    residual_monotonic_pass = False
            else:
                residual_monotonic_pass = False
            candidate["residual_monotonic_ready"] = bool(residual_monotonic_ready)
            candidate["residual_monotonic_pass"] = bool(residual_monotonic_pass)
            if (
                ok_now
                and ticks_seen >= required_ticks
                and (not residual_monotonic_ready or not residual_monotonic_pass)
            ):
                ok_now = False
                fail_reason = "probation_residual_non_monotonic"
                counts["probation_residual_non_monotonic"] = int(
                    counts.get("probation_residual_non_monotonic", 0)
                ) + 1
                runner._backend_probation_residual_non_monotonic_fail_count = int(
                    getattr(
                        runner,
                        "_backend_probation_residual_non_monotonic_fail_count",
                        0,
                    )
                ) + 1

        defer_first_enable = bool(
            runner.global_config.get("BACKEND_SUPERVISOR_DEFER_FIRST_ENABLE", True)
        )
        max_probation_ticks = int(
            max(
                required_ticks,
                runner.global_config.get(
                    "BACKEND_SUPERVISOR_MAX_PROBATION_TICKS",
                    max(required_ticks, 3),
                ),
            )
        )
        soft_fail_reason = str(fail_reason) in {
            "quality_reject",
            "probation_residual_non_monotonic",
            "probation_evidence_insufficient",
        }
        candidate["all_pass"] = bool(ok_now)
        candidate["probation_direction_consistent"] = bool(direction_consistency_pass)
        candidate["probation_magnitude_consistent"] = bool(magnitude_consistency_pass)
        if bool(innovation_evidence):
            candidate["evidence_pass_ticks"] = int(candidate.get("evidence_pass_ticks", 0)) + 1
        runner._backend_probation_candidate = candidate
        if (
            (not ok_now)
            and bool(defer_first_enable)
            and bool(soft_fail_reason)
            and ticks_seen < max_probation_ticks
        ):
            counts["probation_deferred"] = int(counts.get("probation_deferred", 0)) + 1
            candidate["probation_defer_reason"] = str(fail_reason or "quality_reject")
            runner._backend_probation_candidate = candidate
            return
        if ticks_seen < required_ticks:
            return

        committed = bool(ok_now)
        evidence_min_ticks = int(
            max(
                1,
                runner.global_config.get(
                    "BACKEND_SUPERVISOR_EVIDENCE_MIN_TICKS",
                    required_ticks,
                ),
            )
        )
        if committed and int(candidate.get("evidence_pass_ticks", 0)) < evidence_min_ticks:
            committed = False
            fail_reason = "probation_evidence_insufficient"
            counts["probation_evidence_insufficient"] = int(
                counts.get("probation_evidence_insufficient", 0)
            ) + 1
        if committed:
            commit_energy_shaping_enable = bool(
                runner.global_config.get(
                    "BACKEND_SUPERVISOR_COMMIT_ENERGY_SHAPING_ENABLE",
                    False,
                )
            )
            if commit_energy_shaping_enable:
                shaped_dp, shaped_dyaw, shaped_steps, energy_shaped = _shape_probation_payload(
                    np.asarray(candidate.get("dp", np.zeros(3, dtype=float)), dtype=float),
                    float(candidate.get("dyaw_deg", 0.0)),
                    int(candidate.get("blend_steps", 1)),
                    bounded=False,
                )
            else:
                shaped_dp = np.asarray(candidate.get("dp", np.zeros(3, dtype=float)), dtype=float).reshape(3,)
                shaped_dyaw = float(candidate.get("dyaw_deg", 0.0))
                shaped_steps = int(max(1, int(candidate.get("blend_steps", 1))))
                energy_shaped = False
            if bool(energy_shaped):
                counts["probation_commit_energy_shaped"] = int(
                    counts.get("probation_commit_energy_shaped", 0)
                ) + 1
            q_hist = getattr(runner, "_backend_apply_quality_history", None)
            if isinstance(q_hist, list):
                q_hist.append(float(quality))
                if len(q_hist) > 20000:
                    del q_hist[:-20000]
            lat_hist = getattr(runner, "_backend_apply_latency_ms_history", None)
            cand_age = float(candidate.get("corr_age_sec", np.nan))
            if isinstance(lat_hist, list) and np.isfinite(cand_age):
                lat_hist.append(float(max(0.0, cand_age) * 1000.0))
                if len(lat_hist) > 20000:
                    del lat_hist[:-20000]
            resid_hist = getattr(runner, "_backend_apply_residual_xy_history", None)
            cand_resid = float(candidate.get("residual_xy_metric", np.nan))
            if isinstance(resid_hist, list) and np.isfinite(cand_resid):
                resid_hist.append(float(cand_resid))
                if len(resid_hist) > 20000:
                    del resid_hist[:-20000]
            self._enqueue_backend_pending(
                dp=np.asarray(shaped_dp, dtype=float),
                dyaw_deg=float(shaped_dyaw),
                blend_steps=int(shaped_steps),
                corr_quality=quality,
                corr_age_sec=float(candidate.get("corr_age_sec", np.nan)),
                residual_xy_metric=float(candidate.get("residual_xy_metric", np.nan)),
                source=source,
                t_ref=float(candidate.get("t_ref", np.nan)),
                time_aligned_used=bool(candidate.get("time_aligned_used", False)),
                reason=(
                    "probation_commit_energy_shaped"
                    if bool(energy_shaped)
                    else "probation_commit"
                ),
            )
            dyaw_commit = float(shaped_dyaw)
            if abs(dyaw_commit) > 1e-9 and getattr(runner, "yaw_authority_service", None) is not None:
                try:
                    runner.yaw_authority_service.register_applied(
                        source="BACKEND",
                        timestamp=float(candidate.get("t_ref", np.nan)),
                        abs_yaw_deg=float(abs(dyaw_commit)),
                    )
                except Exception:
                    pass
            runner._backend_probation_commit_count = int(getattr(runner, "_backend_probation_commit_count", 0)) + 1
            if bool(candidate.get("time_aligned_used", False)):
                runner._backend_time_aligned_apply_count = int(
                    getattr(runner, "_backend_time_aligned_apply_count", 0)
                ) + 1
            self._backend_update_source_reliability(
                source=source,
                quality_score=quality,
                residual_xy=float(candidate.get("residual_xy_metric", np.nan)),
                committed=True,
            )
            self._log_backend_apply_trace(
                ApplyDecisionRecord(
                    timestamp=float(t_now),
                    source=source,
                    decision_state="COMMIT",
                    reason=(
                        "probation_commit_clamped_energy_shaped"
                        if bool(candidate.get("probation_clamped", False)) and bool(energy_shaped)
                        else "probation_commit_clamped"
                        if bool(candidate.get("probation_clamped", False))
                        else "probation_commit_energy_shaped"
                        if bool(energy_shaped)
                        else "probation_commit"
                    ),
                    quality_score=quality,
                    residual_xy=float(candidate.get("residual_xy_metric", np.nan)),
                    dp_xy_in=float(np.linalg.norm(np.asarray(candidate.get("dp", np.zeros(3))[:2], dtype=float))),
                    dp_xy_applied=0.0,
                    age_sec=float(candidate.get("corr_age_sec", np.nan)),
                    t_ref=float(candidate.get("t_ref", np.nan)),
                    time_aligned_used=bool(candidate.get("time_aligned_used", False)),
                )
            )
        else:
            reason = fail_reason if str(fail_reason).strip() else "quality_reject"
            counts[str(reason)] = int(counts.get(str(reason), 0)) + 1
            runner._backend_probation_reject_count = int(getattr(runner, "_backend_probation_reject_count", 0)) + 1
            self._backend_update_source_reliability(
                source=source,
                quality_score=quality,
                residual_xy=float(candidate.get("residual_xy_metric", np.nan)),
                committed=False,
            )
            reject_to_hint = bool(runner.global_config.get("BACKEND_SUPERVISOR_REJECT_TO_HINT_ONLY", True))
            state_name = "HINT_ONLY" if reject_to_hint else "REJECT"
            self._log_backend_apply_trace(
                ApplyDecisionRecord(
                    timestamp=float(t_now),
                    source=source,
                    decision_state=state_name,
                    reason=reason,
                    quality_score=quality,
                    residual_xy=float(candidate.get("residual_xy_metric", np.nan)),
                    dp_xy_in=float(np.linalg.norm(np.asarray(candidate.get("dp", np.zeros(3))[:2], dtype=float))),
                    dp_xy_applied=0.0,
                    age_sec=float(candidate.get("corr_age_sec", np.nan)),
                    t_ref=float(candidate.get("t_ref", np.nan)),
                    time_aligned_used=False,
                )
            )

        runner._backend_probation_candidate = None

    def schedule_backend_correction(self, corr) -> None:
        """
        Schedule backend correction for gradual blend-in to avoid EKF state jumps.
        """
        runner = self.runner
        if corr is None:
            return
        state_t_now = float(getattr(getattr(runner, "state", None), "t", 0.0))
        contract, contract_note = self._validate_backend_contract_v1(corr)
        if contract is None:
            runner._backend_contract_violation_count = int(getattr(runner, "_backend_contract_violation_count", 0)) + 1
            counts = self._backend_reject_counts()
            counts["contract_violation"] = int(counts.get("contract_violation", 0)) + 1
            counts["backend_contract_violation"] = int(counts.get("backend_contract_violation", 0)) + 1
            self._log_backend_apply_trace(
                ApplyDecisionRecord(
                    timestamp=state_t_now,
                    source=str(getattr(corr, "source", "BACKEND")).upper(),
                    decision_state="REJECT",
                    reason="contract_violation",
                    quality_score=float(getattr(corr, "quality_score", np.nan)),
                    residual_xy=float("nan"),
                    dp_xy_in=float("nan"),
                    dp_xy_applied=0.0,
                    age_sec=float(getattr(corr, "age_sec", np.nan)),
                    t_ref=float(getattr(corr, "t_ref", np.nan)),
                    time_aligned_used=False,
                )
            )
            if bool(getattr(runner.config, "save_debug_data", False)):
                print(f"[BACKEND] contract violation: correction rejected ({contract_note})")
            return

        dp = np.asarray(contract.dp_enu, dtype=float).reshape(3,)
        dyaw_deg = float(contract.dyaw_deg)
        yaw_auth_reason = ""
        src_mix = dict(contract.source_mix)
        resid = dict(contract.residual_summary)
        source_name = str(getattr(corr, "source", "")).strip().upper()
        if not source_name:
            source_name = (
                str(max(src_mix.items(), key=lambda kv: float(kv[1]))[0]).upper()
                if len(src_mix) > 0
                else "BACKEND"
            )
        direct_xy_lane = bool(float(resid.get("direct_xy_lane", 0.0)) >= 0.5)
        corr_quality = float(np.clip(float(contract.quality_score), 0.0, 1.0))
        corr_age_sec = float(contract.age_sec)
        residual_xy_metric = float(
            resid.get(
                "residual_xy_p50",
                resid.get("residual_xy_mean", np.nan),
            )
        )
        if (not np.isfinite(state_t_now)) or float(state_t_now) <= 0.0:
            t_ref_trace = float(contract.t_ref)
            if np.isfinite(t_ref_trace):
                state_t_now = float(t_ref_trace)
        supervisor_enable = bool(runner.global_config.get("BACKEND_SUPERVISOR_ENABLE", True))
        if getattr(runner, "state", None) is None:
            supervisor_enable = False
        runner._backend_proposed_count = int(getattr(runner, "_backend_proposed_count", 0)) + 1
        self._log_backend_apply_trace(
            ApplyDecisionRecord(
                timestamp=state_t_now,
                source=source_name,
                decision_state="PROPOSE",
                reason="proposed",
                quality_score=float(corr_quality),
                residual_xy=float(residual_xy_metric),
                dp_xy_in=float(np.linalg.norm(dp[:2])),
                dp_xy_applied=0.0,
                age_sec=float(corr_age_sec),
                t_ref=float(contract.t_ref),
                time_aligned_used=False,
            )
        )

        # Source reliability quarantine: low-reliability sources are downgraded
        # to deterministic HINT_ONLY unless reliability recovers.
        source_rel_map = self._backend_source_reliability_map()
        source_rel = float(source_rel_map.get(source_name, 0.5))
        if bool(runner.global_config.get("BACKEND_SOURCE_RELIABILITY_ENABLE", True)):
            quarantine_th = float(np.clip(runner.global_config.get("BACKEND_SOURCE_RELIABILITY_QUARANTINE_TH", 0.20), 0.0, 1.0))
            hint_only_below = bool(runner.global_config.get("BACKEND_SOURCE_RELIABILITY_HINT_ONLY_BELOW_TH", True))
            if hint_only_below and np.isfinite(source_rel) and source_rel < quarantine_th:
                counts = self._backend_reject_counts()
                counts["quality_reject"] = int(counts.get("quality_reject", 0)) + 1
                self._backend_update_source_reliability(
                    source=source_name,
                    quality_score=float(corr_quality),
                    residual_xy=float(residual_xy_metric),
                    committed=False,
                )
                self._log_backend_apply_trace(
                    ApplyDecisionRecord(
                        timestamp=state_t_now,
                        source=source_name,
                        decision_state="HINT_ONLY",
                        reason="quality_reject",
                        quality_score=float(corr_quality),
                        residual_xy=float(residual_xy_metric),
                        dp_xy_in=float(np.linalg.norm(dp[:2])),
                        dp_xy_applied=0.0,
                        age_sec=float(corr_age_sec),
                        t_ref=float(contract.t_ref),
                        time_aligned_used=False,
                    )
                )
                return

        position_first_lane = bool(runner.global_config.get("POSITION_FIRST_LANE", False))
        max_dp = float(runner.global_config.get("BACKEND_MAX_APPLY_DP_XY_M", 25.0))
        if position_first_lane:
            max_dp = float(runner.global_config.get("BACKEND_POSITION_FIRST_MAX_APPLY_DP_XY_M", max_dp))
            if direct_xy_lane:
                # Direct XY lane carries raw absolute offsets; keep per-apply cap
                # tighter than generic position-first to reduce drift bursts.
                max_dp = float(
                    runner.global_config.get(
                        "BACKEND_POSITION_FIRST_DIRECT_XY_MAX_APPLY_DP_XY_M",
                        min(max_dp, 28.0),
                    )
                )
        dp_xy_norm = float(np.linalg.norm(dp[:2]))
        if dp_xy_norm > max_dp and dp_xy_norm > 1e-9:
            dp[:2] *= float(max_dp / dp_xy_norm)
        max_dyaw_deg = float(runner.global_config.get("BACKEND_MAX_APPLY_DYAW_DEG", 2.5))
        dyaw_deg = float(np.clip(dyaw_deg, -max_dyaw_deg, max_dyaw_deg))

        # XY-priority fallback: if backend correction is dominated by VPS XY hints
        # with weak yaw confidence, suppress/limit yaw injection and keep XY correction.
        if bool(runner.global_config.get("BACKEND_XY_PRIORITY_ENABLE", True)):
            vps_mix = float(src_mix.get("VPS", src_mix.get("ABS", 0.0)))
            loop_mix = float(src_mix.get("LOOP", 0.0))
            mag_mix = float(src_mix.get("MAG", 0.0))
            min_vps_mix = float(runner.global_config.get("BACKEND_XY_PRIORITY_MIN_VPS_MIX", 0.70))
            max_loop_mix = float(runner.global_config.get("BACKEND_XY_PRIORITY_MAX_LOOP_MIX", 0.10))
            max_mag_mix = float(runner.global_config.get("BACKEND_XY_PRIORITY_MAX_MAG_MIX", 0.15))
            if (
                np.isfinite(vps_mix)
                and np.isfinite(loop_mix)
                and np.isfinite(mag_mix)
                and float(vps_mix) >= float(min_vps_mix)
                and float(loop_mix) <= float(max_loop_mix)
                and float(mag_mix) <= float(max_mag_mix)
            ):
                q_th = float(runner.global_config.get("BACKEND_XY_PRIORITY_QUALITY_TH", 0.55))
                if float(corr_quality) < float(q_th):
                    lowq_cap = float(
                        max(0.0, runner.global_config.get("BACKEND_XY_PRIORITY_LOWQ_MAX_DYAW_DEG", 0.15))
                    )
                    if lowq_cap <= 1e-6:
                        dyaw_deg = 0.0
                        yaw_auth_reason = "xy_priority_zero_yaw_low_quality"
                    else:
                        dyaw_prev = float(dyaw_deg)
                        dyaw_deg = float(np.clip(dyaw_prev, -lowq_cap, lowq_cap))
                        yaw_auth_reason = (
                            "xy_priority_lowq_cap_yaw"
                            if abs(dyaw_prev) > lowq_cap
                            else "xy_priority_lowq_keep_yaw"
                        )
                else:
                    xy_cap = float(max(0.05, runner.global_config.get("BACKEND_XY_PRIORITY_MAX_DYAW_DEG", 0.35)))
                    if abs(dyaw_deg) > xy_cap:
                        dyaw_deg = float(np.clip(dyaw_deg, -xy_cap, xy_cap))
                        yaw_auth_reason = "xy_priority_cap_yaw"
        if position_first_lane and bool(runner.global_config.get("BACKEND_POSITION_FIRST_FORCE_XY_ONLY", True)):
            vps_mix_pf = float(src_mix.get("VPS", src_mix.get("ABS", 0.0)))
            min_vps_mix_pf = float(runner.global_config.get("BACKEND_POSITION_FIRST_MIN_VPS_MIX", 0.35))
            if np.isfinite(vps_mix_pf) and float(vps_mix_pf) >= float(min_vps_mix_pf):
                dyaw_deg = 0.0
                yaw_auth_reason = "position_first_xy_only"

        if getattr(runner, "yaw_authority_service", None) is not None:
            yaw_denied = False
            speed_now = float(np.linalg.norm(np.asarray(runner.kf.x[3:6, 0], dtype=float)))
            adaptive_decision = getattr(runner, "current_adaptive_decision", None)
            health_state = str(getattr(adaptive_decision, "health_state", "HEALTHY")).upper()
            try:
                p_abs = np.asarray(runner.kf.P, dtype=float)
                p_max = float(np.nanmax(np.abs(p_abs))) if p_abs.size > 0 else float("nan")
                p_cond = float("inf")
                if p_abs.ndim == 2 and p_abs.shape[0] > 0 and p_abs.shape[1] > 0:
                    dim = min(48, int(p_abs.shape[0]))
                    p_block = 0.5 * (p_abs[:dim, :dim] + p_abs[:dim, :dim].T)
                    p_cond = float(np.linalg.cond(p_block))
            except Exception:
                p_max = float("nan")
                p_cond = float("inf")
            conf = float(corr_quality)
            req_abs_dyaw = float(abs(dyaw_deg))
            zero_req_deadband = max(
                0.0,
                float(runner.global_config.get("BACKEND_YAW_AUTH_ZERO_REQUEST_DEADBAND_DEG", 0.02)),
            )
            count_zero_as_request = bool(
                runner.global_config.get("BACKEND_YAW_AUTH_ZERO_DYAW_AS_REQUEST", False)
            )
            if req_abs_dyaw <= zero_req_deadband and not count_zero_as_request:
                # Deterministic no-op path: XY-only backend corrections should not
                # claim yaw ownership nor keep yaw-owner activity alive.
                yaw_auth_reason = "apply:backend_zero_dyaw_bypass"
                dyaw_deg = 0.0
            else:
                req_abs_dyaw = float(
                    max(
                        req_abs_dyaw,
                        float(runner.global_config.get("BACKEND_YAW_AUTH_MIN_REQUEST_DEG", 0.05)),
                    )
                )
                auth_dec = runner.yaw_authority_service.request_decision(
                    source="BACKEND",
                    timestamp=float(getattr(corr, "t_ref", 0.0)),
                    requested_abs_dyaw_deg=float(req_abs_dyaw),
                    confidence=float(conf),
                    speed_m_s=float(speed_now),
                    health_state=str(health_state),
                    p_max=float(p_max),
                    p_cond=float(p_cond),
                )
                if (not bool(auth_dec.allow)) or str(auth_dec.mode).upper() in ("HOLD", "SKIP"):
                    yaw_auth_reason = f"skip:{auth_dec.reason}"
                    dyaw_deg = 0.0
                    yaw_denied = True
                else:
                    cap = float(auth_dec.max_update_dyaw_deg) if np.isfinite(float(auth_dec.max_update_dyaw_deg)) else float(max_dyaw_deg)
                    cap = max(0.05, min(float(max_dyaw_deg), cap))
                    dyaw_deg = float(np.clip(dyaw_deg, -cap, cap))
                    if str(auth_dec.mode).upper() == "SOFT_APPLY":
                        yaw_auth_reason = f"soft:{auth_dec.reason}"
                    else:
                        yaw_auth_reason = f"apply:{auth_dec.reason}"
            if yaw_denied and not bool(runner.global_config.get("BACKEND_XY_YAW_DECOUPLE_ENABLE", True)):
                counts = self._backend_reject_counts()
                counts["quality_reject"] = int(counts.get("quality_reject", 0)) + 1
                self._backend_update_source_reliability(
                    source=source_name,
                    quality_score=float(corr_quality),
                    residual_xy=float(residual_xy_metric),
                    committed=False,
                )
                self._log_backend_apply_trace(
                    ApplyDecisionRecord(
                        timestamp=state_t_now,
                        source=source_name,
                        decision_state="REJECT",
                        reason="quality_reject",
                        quality_score=float(corr_quality),
                        residual_xy=float(residual_xy_metric),
                        dp_xy_in=float(np.linalg.norm(np.asarray(dp[:2], dtype=float))),
                        dp_xy_applied=0.0,
                        age_sec=float(corr_age_sec),
                        t_ref=float(contract.t_ref),
                        time_aligned_used=False,
                    )
                )
                return

        # Deterministic kinematic consistency gate (C5):
        # 1) magnitude-consistency: reject physically implausible short-horizon jumps
        # 2) direction-consistency: reject strong opposite-direction pushes
        if bool(runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_ENABLE", True)):
            try:
                vel_xy = np.asarray(runner.kf.x[3:5, 0], dtype=float).reshape(2,)
            except Exception:
                vel_xy = np.zeros(2, dtype=float)
            speed_xy = float(np.linalg.norm(vel_xy))
            dp_xy = np.asarray(dp[:2], dtype=float).reshape(2,)
            dp_xy_norm = float(np.linalg.norm(dp_xy))
            min_speed = float(runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_MIN_SPEED_M_S", 8.0))
            min_dp_xy = float(runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_MIN_DP_XY_M", 4.0))
            min_cos = float(runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_MIN_COS", -0.25))
            magnitude_enable = bool(
                runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_MAGNITUDE_ENABLE", True)
            )
            magnitude_horizon_sec = float(
                runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_MAGNITUDE_HORIZON_SEC", 0.8)
            )
            magnitude_base_m = float(
                runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_MAGNITUDE_BASE_M", 6.0)
            )
            magnitude_speed_gain = float(
                runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_MAGNITUDE_SPEED_GAIN", 1.5)
            )
            magnitude_max_m = float(
                runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_MAGNITUDE_MAX_M", 45.0)
            )
            if (
                np.isfinite(speed_xy)
                and np.isfinite(dp_xy_norm)
                and speed_xy >= max(0.0, min_speed)
                and dp_xy_norm >= max(0.0, min_dp_xy)
                and speed_xy > 1e-6
                and dp_xy_norm > 1e-6
            ):
                counts = self._backend_reject_counts()

                # Magnitude-consistency: allow correction size proportional to
                # short-horizon kinematic travel plus a fixed base margin.
                if bool(magnitude_enable):
                    horizon = max(0.05, float(magnitude_horizon_sec))
                    base_allow = max(0.0, float(magnitude_base_m))
                    speed_gain = max(0.0, float(magnitude_speed_gain))
                    max_allow = float(base_allow + speed_gain * speed_xy * horizon)
                    if np.isfinite(magnitude_max_m) and magnitude_max_m > 0.0:
                        max_allow = float(min(max_allow, float(magnitude_max_m)))
                    max_allow = float(max(max(0.0, min_dp_xy), max_allow))
                    if np.isfinite(max_allow) and dp_xy_norm > max_allow:
                        defer_mag_to_probation = bool(supervisor_enable) and bool(
                            runner.global_config.get(
                                "BACKEND_SUPERVISOR_SKIP_SCHEDULE_MAG_CLAMP",
                                True,
                            )
                        )
                        if bool(defer_mag_to_probation):
                            counts["kinematic_clamp_magnitude_probation_deferred"] = int(
                                counts.get("kinematic_clamp_magnitude_probation_deferred", 0)
                            ) + 1
                        else:
                            # Supervisor disabled: keep deterministic clamp in schedule stage.
                            orig_dp_xy_norm = float(dp_xy_norm)
                            scale = float(np.clip(max_allow / max(dp_xy_norm, 1e-9), 0.0, 1.0))
                            dp[:2] = np.asarray(dp[:2], dtype=float) * scale
                            dyaw_deg = float(dyaw_deg * scale)
                            dp_xy = np.asarray(dp[:2], dtype=float).reshape(2,)
                            dp_xy_norm = float(np.linalg.norm(dp_xy))
                            counts["kinematic_clamp_magnitude"] = int(
                                counts.get("kinematic_clamp_magnitude", 0)
                            ) + 1
                            runner._backend_kinematic_magnitude_clamp_count = int(
                                getattr(runner, "_backend_kinematic_magnitude_clamp_count", 0)
                            ) + 1
                            if bool(getattr(runner.config, "save_debug_data", False)):
                                print(
                                    "[BACKEND] kinematic_clamp(magnitude):"
                                    f" orig_dp_xy={orig_dp_xy_norm:.2f},"
                                    f" clamped_dp_xy={dp_xy_norm:.2f}, max_allow={max_allow:.2f},"
                                    f" speed_xy={speed_xy:.2f}, horizon={horizon:.2f},"
                                    f" base={base_allow:.2f}, gain={speed_gain:.2f}, scale={scale:.3f}"
                                )

                if dp_xy_norm > 1e-6:
                    cos_dir = float(np.dot(dp_xy, vel_xy) / (speed_xy * dp_xy_norm))
                    if np.isfinite(cos_dir) and cos_dir < float(min_cos):
                        if bool(supervisor_enable) and bool(
                            runner.global_config.get("BACKEND_SUPERVISOR_DEFER_DIRECTION_GATE", True)
                        ):
                            counts["kinematic_reject_direction_probation_deferred"] = int(
                                counts.get("kinematic_reject_direction_probation_deferred", 0)
                            ) + 1
                        else:
                            runner._backend_kinematic_reject_count = int(
                                getattr(runner, "_backend_kinematic_reject_count", 0)
                            ) + 1
                            counts["kinematic_reject"] = int(counts.get("kinematic_reject", 0)) + 1
                            counts["kinematic_reject_direction"] = int(
                                counts.get("kinematic_reject_direction", 0)
                            ) + 1
                            counts["backend_kinematic_reject"] = int(
                                counts.get("backend_kinematic_reject", 0)
                            ) + 1
                            self._backend_update_source_reliability(
                                source=source_name,
                                quality_score=float(corr_quality),
                                residual_xy=float(residual_xy_metric),
                                committed=False,
                            )
                            self._log_backend_apply_trace(
                                ApplyDecisionRecord(
                                    timestamp=state_t_now,
                                    source=source_name,
                                    decision_state="REJECT",
                                    reason="kinematic_reject",
                                    quality_score=float(corr_quality),
                                    residual_xy=float(residual_xy_metric),
                                    dp_xy_in=float(np.linalg.norm(np.asarray(dp[:2], dtype=float))),
                                    dp_xy_applied=0.0,
                                    age_sec=float(corr_age_sec),
                                    t_ref=float(contract.t_ref),
                                    time_aligned_used=False,
                                )
                            )
                            if bool(getattr(runner.config, "save_debug_data", False)):
                                print(
                                    "[BACKEND] kinematic_reject(direction):"
                                    f" cos={cos_dir:.3f}<min_cos={float(min_cos):.3f},"
                                    f" speed_xy={speed_xy:.2f}, dp_xy={dp_xy_norm:.2f}"
                                )
                            return

        corr_weight = float(runner.global_config.get("BACKEND_CORRECTION_WEIGHT", 1.0))
        if position_first_lane:
            corr_weight = float(runner.global_config.get("BACKEND_POSITION_FIRST_CORR_WEIGHT", corr_weight))
            if direct_xy_lane:
                corr_weight = float(
                    runner.global_config.get(
                        "BACKEND_POSITION_FIRST_DIRECT_XY_CORR_WEIGHT",
                        min(corr_weight, 0.72),
                    )
                )
        corr_weight = float(np.clip(corr_weight, 0.0, 1.0))
        if bool(runner.global_config.get("BACKEND_BLEND_QUALITY_SOFT_SCALE", True)) and not direct_xy_lane:
            q_corr = float(corr_quality)
            corr_weight *= float(np.clip(0.55 + 0.45 * q_corr, 0.25, 1.0))
        # Pseudo-measurement gain (C5): replace raw offset-style authority with
        # deterministic normalized gain from quality + residual.
        if bool(runner.global_config.get("BACKEND_PSEUDOMEAS_ENABLE", True)):
            gain_base = float(np.clip(runner.global_config.get("BACKEND_PSEUDOMEAS_GAIN_BASE", 0.75), 0.0, 2.0))
            gain_q = float(np.clip(runner.global_config.get("BACKEND_PSEUDOMEAS_GAIN_QUALITY_GAIN", 0.35), 0.0, 2.0))
            gain_ref_m = float(max(0.1, runner.global_config.get("BACKEND_PSEUDOMEAS_GAIN_RESIDUAL_REF_M", 12.0)))
            gain_min = float(np.clip(runner.global_config.get("BACKEND_PSEUDOMEAS_GAIN_MIN", 0.25), 0.0, 2.0))
            gain_max = float(np.clip(runner.global_config.get("BACKEND_PSEUDOMEAS_GAIN_MAX", 1.0), 0.0, 2.0))
            q_term = float(np.clip(corr_quality, 0.0, 1.0))
            residual_term = 1.0
            if np.isfinite(float(residual_xy_metric)):
                residual_term = float(np.clip(1.0 / (1.0 + max(0.0, float(residual_xy_metric)) / gain_ref_m), 0.0, 1.0))
            pseudo_gain = float(np.clip((gain_base + gain_q * q_term) * residual_term, gain_min, max(gain_min, gain_max)))
            corr_weight *= pseudo_gain

        # Source reliability escalation/de-escalation in commit authority.
        if bool(runner.global_config.get("BACKEND_SOURCE_RELIABILITY_ENABLE", True)):
            source_rel = float(self._backend_source_reliability_map().get(source_name, 0.5))
            escalate_th = float(np.clip(runner.global_config.get("BACKEND_SOURCE_RELIABILITY_ESCALATE_TH", 0.75), 0.0, 1.0))
            if np.isfinite(source_rel):
                rel_scale = float(np.clip(0.60 + 0.70 * source_rel, 0.20, 1.20))
                if source_rel >= escalate_th:
                    rel_scale = float(np.clip(rel_scale * 1.05, 0.20, 1.30))
                corr_weight *= rel_scale
        if corr_weight < 1.0:
            dp *= corr_weight
            dyaw_deg *= corr_weight

        blend_steps = max(1, int(runner.global_config.get("BACKEND_BLEND_STEPS", 3)))
        if position_first_lane:
            blend_steps = max(1, int(runner.global_config.get("BACKEND_POSITION_FIRST_BLEND_STEPS", blend_steps)))
            if direct_xy_lane:
                blend_steps = max(
                    blend_steps,
                    int(
                        max(
                            1,
                            round(
                                runner.global_config.get(
                                    "BACKEND_POSITION_FIRST_DIRECT_XY_MIN_BLEND_STEPS",
                                    3,
                                )
                            ),
                        )
                    ),
                )

        # C4.2: deterministic time-aligned apply gating.
        time_aligned_enable = bool(runner.global_config.get("BACKEND_TIME_ALIGNED_APPLY_ENABLE", True))
        time_aligned_max_age_sec = float(max(0.0, runner.global_config.get("BACKEND_TIME_ALIGNED_MAX_AGE_SEC", 0.35)))
        time_aligned_reject_stale = bool(runner.global_config.get("BACKEND_TIME_ALIGNED_REJECT_STALE", True))
        time_aligned_used = bool(time_aligned_enable and np.isfinite(corr_age_sec) and corr_age_sec <= time_aligned_max_age_sec)
        if (
            bool(time_aligned_enable)
            and bool(time_aligned_reject_stale)
            and np.isfinite(corr_age_sec)
            and corr_age_sec > time_aligned_max_age_sec
        ):
            counts = self._backend_reject_counts()
            counts["stale_reject"] = int(counts.get("stale_reject", 0)) + 1
            self._backend_update_source_reliability(
                source=source_name,
                quality_score=float(corr_quality),
                residual_xy=float(residual_xy_metric),
                committed=False,
            )
            self._log_backend_apply_trace(
                ApplyDecisionRecord(
                    timestamp=state_t_now,
                    source=source_name,
                    decision_state="REJECT",
                    reason="stale_reject",
                    quality_score=float(corr_quality),
                    residual_xy=float(residual_xy_metric),
                    dp_xy_in=float(np.linalg.norm(np.asarray(dp[:2], dtype=float))),
                    dp_xy_applied=0.0,
                    age_sec=float(corr_age_sec),
                    t_ref=float(contract.t_ref),
                    time_aligned_used=False,
                )
            )
            return

        def _record_schedule_histories() -> None:
            q_hist = getattr(runner, "_backend_apply_quality_history", None)
            if isinstance(q_hist, list):
                q_hist.append(float(corr_quality))
                if len(q_hist) > 20000:
                    del q_hist[:-20000]
            lat_hist = getattr(runner, "_backend_apply_latency_ms_history", None)
            if isinstance(lat_hist, list) and np.isfinite(corr_age_sec):
                lat_hist.append(float(max(0.0, corr_age_sec) * 1000.0))
                if len(lat_hist) > 20000:
                    del lat_hist[:-20000]
            resid_hist = getattr(runner, "_backend_apply_residual_xy_history", None)
            if isinstance(resid_hist, list) and np.isfinite(residual_xy_metric):
                resid_hist.append(float(residual_xy_metric))
                if len(resid_hist) > 20000:
                    del resid_hist[:-20000]

        # C4.1: supervisor lane (PROPOSE -> PROBATION -> COMMIT|REJECT).
        if supervisor_enable:
            existing_candidate = getattr(runner, "_backend_probation_candidate", None)
            if isinstance(existing_candidate, dict):
                source_rel_map = self._backend_source_reliability_map()
                existing_source = str(existing_candidate.get("source", "BACKEND")).upper()
                existing_rel = float(source_rel_map.get(existing_source, 0.5))
                existing_quality = float(existing_candidate.get("corr_quality", np.nan))
                existing_score = (
                    float(np.clip(existing_quality, 0.0, 1.0)) * float(np.clip(existing_rel, 0.0, 1.0))
                    if np.isfinite(existing_quality)
                    else 0.0
                )
                new_score = float(np.clip(corr_quality, 0.0, 1.0)) * float(
                    np.clip(source_rel, 0.0, 1.0) if np.isfinite(source_rel) else 0.5
                )
                replace_min_delta = float(
                    max(
                        0.0,
                        runner.global_config.get("BACKEND_SUPERVISOR_REPLACE_MIN_SCORE_DELTA", 0.08),
                    )
                )
                force_replace_max_age_sec = float(
                    max(
                        0.0,
                        runner.global_config.get("BACKEND_SUPERVISOR_REPLACE_FORCE_MAX_AGE_SEC", 0.45),
                    )
                )
                existing_age = float(existing_candidate.get("corr_age_sec", np.nan))
                force_replace = bool(np.isfinite(existing_age) and existing_age > force_replace_max_age_sec)
                if (not force_replace) and (
                    new_score <= (existing_score + replace_min_delta)
                ):
                    counts = self._backend_reject_counts()
                    counts["probation_deferred"] = int(counts.get("probation_deferred", 0)) + 1
                    self._log_backend_apply_trace(
                        ApplyDecisionRecord(
                            timestamp=state_t_now,
                            source=source_name,
                            decision_state="HINT_ONLY",
                            reason="probation_deferred",
                            quality_score=float(corr_quality),
                            residual_xy=float(residual_xy_metric),
                            dp_xy_in=float(np.linalg.norm(np.asarray(dp[:2], dtype=float))),
                            dp_xy_applied=0.0,
                            age_sec=float(corr_age_sec),
                            t_ref=float(contract.t_ref),
                            time_aligned_used=bool(time_aligned_used),
                        )
                    )
                    return
                runner._backend_probation_replace_count = int(
                    getattr(runner, "_backend_probation_replace_count", 0)
                ) + 1
            runner._backend_probation_count = int(getattr(runner, "_backend_probation_count", 0)) + 1
            runner._backend_probation_candidate = {
                "dp": np.asarray(dp, dtype=float).reshape(3,).copy(),
                "dyaw_deg": float(dyaw_deg),
                "blend_steps": int(blend_steps),
                "corr_quality": float(corr_quality),
                "corr_age_sec": float(corr_age_sec),
                "residual_xy_metric": float(residual_xy_metric),
                "source": str(source_name),
                "t_ref": float(contract.t_ref),
                "time_aligned_used": bool(time_aligned_used),
                "ticks_seen": 0,
                "last_eval_t": -1e9,
                "all_pass": True,
            }
            self._log_backend_apply_trace(
                ApplyDecisionRecord(
                    timestamp=state_t_now,
                    source=source_name,
                    decision_state="PROBATION",
                    reason="probation_start",
                    quality_score=float(corr_quality),
                    residual_xy=float(residual_xy_metric),
                    dp_xy_in=float(np.linalg.norm(np.asarray(dp[:2], dtype=float))),
                    dp_xy_applied=0.0,
                    age_sec=float(corr_age_sec),
                    t_ref=float(contract.t_ref),
                    time_aligned_used=bool(time_aligned_used),
                )
            )
            return

        # Direct commit fallback when supervisor lane is disabled.
        _record_schedule_histories()
        self._enqueue_backend_pending(
            dp=np.asarray(dp, dtype=float).reshape(3,),
            dyaw_deg=float(dyaw_deg),
            blend_steps=int(blend_steps),
            corr_quality=float(corr_quality),
            corr_age_sec=float(corr_age_sec),
            residual_xy_metric=float(residual_xy_metric),
            source=str(source_name),
            t_ref=float(contract.t_ref),
            time_aligned_used=bool(time_aligned_used),
            reason="direct_commit",
        )
        if bool(time_aligned_used):
            runner._backend_time_aligned_apply_count = int(getattr(runner, "_backend_time_aligned_apply_count", 0)) + 1
        self._backend_update_source_reliability(
            source=source_name,
            quality_score=float(corr_quality),
            residual_xy=float(residual_xy_metric),
            committed=True,
        )
        self._log_backend_apply_trace(
            ApplyDecisionRecord(
                timestamp=state_t_now,
                source=source_name,
                decision_state="COMMIT",
                reason="direct_commit",
                quality_score=float(corr_quality),
                residual_xy=float(residual_xy_metric),
                dp_xy_in=float(np.linalg.norm(np.asarray(dp[:2], dtype=float))),
                dp_xy_applied=0.0,
                age_sec=float(corr_age_sec),
                t_ref=float(contract.t_ref),
                time_aligned_used=bool(time_aligned_used),
            )
        )
        if abs(dyaw_deg) > 1e-9 and getattr(runner, "yaw_authority_service", None) is not None:
            try:
                runner.yaw_authority_service.register_applied(
                    source="BACKEND",
                    timestamp=float(getattr(corr, "t_ref", 0.0)),
                    abs_yaw_deg=float(abs(dyaw_deg)),
                )
            except Exception:
                pass
        if yaw_auth_reason and getattr(runner, "config", None) is not None and bool(getattr(runner.config, "save_debug_data", False)):
            print(f"[BACKEND] yaw_auth {yaw_auth_reason}")

    def apply_pending_backend_blend(self, t_now: float) -> bool:
        """
        Apply one blend step from pending backend correction.
        """
        runner = self.runner
        # Advance C4.1 probation lane on deterministic cadence.
        self._process_backend_probation(float(t_now))
        steps_left = int(getattr(runner, "_backend_pending_steps_left", 0))
        if steps_left <= 0:
            return False
        dp_rem = getattr(runner, "_backend_pending_dp_enu", None)
        if dp_rem is None:
            runner._backend_pending_steps_left = 0
            return False

        dp_rem = np.asarray(dp_rem, dtype=float).reshape(3,)
        dyaw_rem = float(getattr(runner, "_backend_pending_dyaw_deg", 0.0))
        q_pending = float(np.clip(float(getattr(runner, "_backend_pending_quality", 0.6)), 0.0, 1.0))
        blend_mode = str(runner.global_config.get("BACKEND_BLEND_MODE", "linear")).strip().lower()
        blend_alpha = float(np.clip(runner.global_config.get("BACKEND_BLEND_ALPHA", 0.45), 0.05, 0.95))
        max_step_dp_xy = float(max(0.1, runner.global_config.get("BACKEND_BLEND_MAX_STEP_DP_XY_M", 8.0)))
        max_step_dyaw = float(max(0.05, runner.global_config.get("BACKEND_BLEND_MAX_STEP_DYAW_DEG", 0.8)))
        quality_soft_scale = bool(runner.global_config.get("BACKEND_BLEND_QUALITY_SOFT_SCALE", True))

        if steps_left <= 1:
            step_dp = dp_rem.copy()
            step_dyaw_deg = float(dyaw_rem)
        else:
            if blend_mode == "exp":
                frac = float(blend_alpha)
                if quality_soft_scale:
                    frac = float(np.clip(frac * (0.65 + 0.35 * q_pending), 0.05, 0.95))
                step_dp = dp_rem * frac
                step_dyaw_deg = float(dyaw_rem * frac)
            else:
                step_dp = dp_rem / float(steps_left)
                step_dyaw_deg = float(dyaw_rem / float(steps_left))

            step_xy_norm = float(np.linalg.norm(step_dp[:2]))
            if step_xy_norm > max_step_dp_xy and step_xy_norm > 1e-9:
                step_dp[:2] *= float(max_step_dp_xy / step_xy_norm)
            if abs(step_dyaw_deg) > max_step_dyaw:
                step_dyaw_deg = float(np.clip(step_dyaw_deg, -max_step_dyaw, max_step_dyaw))

        step_xy_norm = float(np.linalg.norm(np.asarray(step_dp[:2], dtype=float)))

        # Deterministic short-horizon energy shaping:
        # block or clamp per-step XY correction if accumulated applied energy
        # within a recent window exceeds physically plausible short-term travel.
        if bool(runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_BUDGET_ENABLE", True)):
            try:
                vel_xy_now = np.asarray(runner.kf.x[3:5, 0], dtype=float).reshape(2,)
                speed_xy_now = float(np.linalg.norm(vel_xy_now))
            except Exception:
                speed_xy_now = 0.0
            window_sec = float(
                runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_BUDGET_WINDOW_SEC", 2.0)
            )
            base_budget_m = float(
                runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_BUDGET_BASE_M", 18.0)
            )
            speed_gain = float(
                runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_BUDGET_SPEED_GAIN", 0.80)
            )
            max_budget_m = float(
                runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_BUDGET_MAX_M", 60.0)
            )
            min_step_m = float(
                runner.global_config.get("BACKEND_KINEMATIC_CONSISTENCY_BUDGET_MIN_STEP_M", 0.30)
            )
            now_t = float(t_now)
            if not np.isfinite(now_t):
                now_t = float(getattr(runner.state, "t", 0.0))
            hist = getattr(runner, "_backend_apply_energy_history", None)
            if not isinstance(hist, list):
                hist = []
                runner._backend_apply_energy_history = hist
            if np.isfinite(window_sec) and window_sec > 0.05 and np.isfinite(now_t):
                cutoff = float(now_t - max(0.05, window_sec))
                hist[:] = [
                    (float(tt), float(mm))
                    for tt, mm in hist
                    if np.isfinite(tt) and np.isfinite(mm) and float(tt) >= cutoff and float(mm) >= 0.0
                ]
            recent_energy_m = float(sum(float(mm) for _, mm in hist))
            budget_allow_m = float(max(0.0, base_budget_m + speed_gain * max(0.0, speed_xy_now) * max(0.05, window_sec)))
            if np.isfinite(max_budget_m) and max_budget_m > 0.0:
                budget_allow_m = float(min(budget_allow_m, max_budget_m))
            remaining_budget_m = float(budget_allow_m - recent_energy_m)
            if step_xy_norm > 1e-9 and np.isfinite(remaining_budget_m):
                counts = self._backend_reject_counts()
                floor_budget_m = max(0.0, float(min_step_m))
                target_budget_m = float(max(0.0, remaining_budget_m))
                if target_budget_m < floor_budget_m:
                    target_budget_m = float(floor_budget_m)
                if step_xy_norm > target_budget_m:
                    scale = float(np.clip(target_budget_m / step_xy_norm, 0.0, 1.0))
                    step_dp[:2] *= scale
                    step_dyaw_deg *= scale
                    step_xy_norm = float(np.linalg.norm(np.asarray(step_dp[:2], dtype=float)))
                    counts["kinematic_budget_clamp"] = int(
                        counts.get("kinematic_budget_clamp", 0)
                    ) + 1
                    if remaining_budget_m <= floor_budget_m:
                        counts["kinematic_budget_floor_clamp"] = int(
                            counts.get("kinematic_budget_floor_clamp", 0)
                        ) + 1
                    runner._backend_kinematic_budget_clamp_count = int(
                        getattr(runner, "_backend_kinematic_budget_clamp_count", 0)
                    ) + 1
                    if bool(getattr(runner.config, "save_debug_data", False)):
                        print(
                            "[BACKEND] kinematic_budget_clamp:"
                            f" scale={scale:.3f}, step={step_xy_norm:.2f}m,"
                            f" remaining={remaining_budget_m:.2f}m,"
                            f" target={target_budget_m:.2f}m,"
                            f" floor={floor_budget_m:.2f}m"
                        )

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
        self._log_backend_apply_trace(
            ApplyDecisionRecord(
                timestamp=float(t_now),
                source=str(getattr(runner, "_backend_pending_source", "BACKEND")),
                decision_state="COMMIT",
                reason=str(getattr(runner, "_backend_pending_reason", "apply_step")),
                quality_score=float(getattr(runner, "_backend_pending_quality", np.nan)),
                residual_xy=float(getattr(runner, "_backend_pending_residual_xy", np.nan)),
                dp_xy_in=float(getattr(runner, "_backend_pending_dp_xy_in", np.nan)),
                dp_xy_applied=float(step_xy_norm),
                age_sec=float(getattr(runner, "_backend_pending_age_sec", np.nan)),
                t_ref=float(getattr(runner, "_backend_pending_t_ref", np.nan)),
                time_aligned_used=bool(getattr(runner, "_backend_pending_time_aligned_used", False)),
            )
        )
        dp_hist = getattr(runner, "_backend_apply_dp_xy_history", None)
        if isinstance(dp_hist, list):
            if np.isfinite(step_xy_norm):
                dp_hist.append(step_xy_norm)
                if len(dp_hist) > 20000:
                    del dp_hist[:-20000]
        energy_hist = getattr(runner, "_backend_apply_energy_history", None)
        if isinstance(energy_hist, list):
            t_apply = float(t_now)
            if not np.isfinite(t_apply):
                t_apply = float(getattr(runner.state, "t", 0.0))
            if np.isfinite(t_apply) and np.isfinite(step_xy_norm) and step_xy_norm >= 0.0:
                energy_hist.append((t_apply, step_xy_norm))
                if len(energy_hist) > 20000:
                    del energy_hist[:-20000]

        if int(runner._backend_pending_steps_left) <= 0:
            runner._backend_pending_dp_enu = None
            runner._backend_pending_dyaw_deg = 0.0
            runner._backend_pending_steps_left = 0
            runner._backend_pending_quality = 0.0
            runner._backend_pending_source = ""
            runner._backend_pending_t_ref = float("nan")
            runner._backend_pending_age_sec = float("nan")
            runner._backend_pending_residual_xy = float("nan")
            runner._backend_pending_dp_xy_in = float("nan")
            runner._backend_pending_reason = ""
            runner._backend_pending_time_aligned_used = False
        return True
