"""Single-authority runtime policy service.

This service builds one immutable snapshot per step from:
1) YAML base configuration
2) adaptive controller decision
3) phase/health runtime state

Sensor/update paths should consume decisions from this service rather than
re-deciding policy locally.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from ..policy.types import PolicySnapshot, SensorPolicyDecision
from ..output_utils import (
    log_policy_trace,
    log_policy_conflict,
    write_policy_owner_map_rows,
)


class PolicyRuntimeService:
    """Build/serve per-step policy decisions with basic conflict tracing."""

    _SENSORS = (
        "MAG",
        "DEM",
        "VIO_VEL",
        "MSCKF",
        "LOOP_CLOSURE",
        "VPS",
        "ZUPT",
        "KIN_GUARD",
        "GRAVITY_RP",
        "YAW_AID",
        "BIAS_GUARD",
    )

    _OWNER_MAP: Mapping[str, str] = {
        "VIO_NADIR_XY_ONLY_VELOCITY": "yaml.vio",
        "VIO_VEL_MAX_DELTA_V_XY_PER_UPDATE_M_S": "yaml.vio_vel",
        "VIO_VEL_DELTA_V_CLAMP_MAX_RATIO": "yaml.vio_vel",
        "VIO_VEL_DELTA_V_CLAMP_R_MULT": "yaml.vio_vel",
        "VIO_VEL_MIN_FLOW_PX_HIGH_SPEED": "yaml.vio_vel",
        "MSCKF_PHASE_REPROJ_SCALE": "yaml.vio.msckf",
        "MSCKF_REPROJ_FAILSOFT_MAX_MULT": "yaml.vio.msckf.reproj_state_aware",
        "LOOP_SPEED_SKIP_M_S": "yaml.loop_closure.fail_soft",
        "LOOP_SPEED_SIGMA_INFLATE_M_S": "yaml.loop_closure.fail_soft",
        "LOOP_SPEED_SIGMA_MULT": "yaml.loop_closure.fail_soft",
        "LOOP_SPEED_SKIP_M_S_NORMAL": "yaml.loop_closure.speed_gate.normal",
        "LOOP_SPEED_SKIP_M_S_FAILSOFT": "yaml.loop_closure.speed_gate.fail_soft",
        "LOOP_SPEED_YAW_CAP_BREAKPOINTS_M_S": "yaml.loop_closure.speed_yaw_cap",
        "LOOP_SPEED_YAW_CAP_NORMAL_DEG": "yaml.loop_closure.speed_yaw_cap",
        "LOOP_SPEED_YAW_CAP_FAILSOFT_DEG": "yaml.loop_closure.speed_yaw_cap",
        "LOOP_APPLY_CONFIRM_ENABLE": "yaml.loop_closure.temporal_apply",
        "LOOP_APPLY_CONFIRM_HITS_NORMAL": "yaml.loop_closure.temporal_apply",
        "LOOP_APPLY_CONFIRM_HITS_FAILSOFT": "yaml.loop_closure.temporal_apply",
        "LOOP_APPLY_CONFIRM_WINDOW_SEC": "yaml.loop_closure.temporal_apply",
        "LOOP_APPLY_CONFIRM_YAW_DEG": "yaml.loop_closure.temporal_apply",
        "LOOP_COOLDOWN_SEC_NORMAL": "yaml.loop_closure.temporal_apply",
        "LOOP_COOLDOWN_SEC_FAILSOFT": "yaml.loop_closure.temporal_apply",
        "LOOP_BURST_WINDOW_SEC": "yaml.loop_closure.temporal_apply",
        "LOOP_BURST_MAX_CORRECTIONS": "yaml.loop_closure.temporal_apply",
        "LOOP_BURST_COOLDOWN_SEC": "yaml.loop_closure.temporal_apply",
        "MAG_CONDITIONING_GUARD_HARD_PCOND": "yaml.magnetometer.warning_weak_yaw",
        "MAG_CONDITIONING_GUARD_HARD_PMAX": "yaml.magnetometer.warning_weak_yaw",
        "MAG_CONDITIONING_GUARD_EXTREME_PCOND": "yaml.magnetometer.warning_weak_yaw",
        "MAG_CONDITIONING_GUARD_EXTREME_PMAX": "yaml.magnetometer.warning_weak_yaw",
        "MAG_CONDITIONING_GUARD_EXTREME_SOFT_ENABLE": "yaml.magnetometer.warning_weak_yaw",
        "MAG_CONDITIONING_GUARD_EXTREME_SOFT_R_MULT": "yaml.magnetometer.warning_weak_yaw",
        "MAG_WARNING_EXTRA_R_MULT": "yaml.magnetometer.warning_weak_yaw",
        "MAG_WARNING_MAX_DYAW_DEG": "yaml.magnetometer.warning_weak_yaw",
        "MAG_DEGRADED_EXTRA_R_MULT": "yaml.magnetometer.warning_weak_yaw",
        "MAG_DEGRADED_MAX_DYAW_DEG": "yaml.magnetometer.warning_weak_yaw",
        "MAG_MODE": "yaml.magnetometer.ablation",
        "MAG_ABLATION_R_MULT": "yaml.magnetometer.ablation",
        "MAG_ABLATION_MAX_DYAW_DEG": "yaml.magnetometer.ablation",
        "MAG_ABLATION_MAX_UPDATE_DYAW_DEG": "yaml.magnetometer.ablation",
        "MAG_ABLATION_WEAK_SKIP_HARD_MISMATCH": "yaml.magnetometer.ablation",
        "MAG_HEADING_ARB_ENABLE": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_HARD_MISMATCH_DEG": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_MIN_VISION_QUALITY": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_STREAK_TO_HOLD": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_HOLD_SEC": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_SOFT_R_MULT": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_MAX_VISION_AGE_SEC": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_SCORE_EMA_ALPHA": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_SCORE_SOFT_THRESHOLD": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_SCORE_HOLD_THRESHOLD": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_SCORE_VIS_GOOD_DEG": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_SCORE_GYRO_GOOD_DEG": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_SCORE_GYRO_BAD_DEG": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_SCORE_STATE_GOOD_DEG": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_SCORE_STATE_BAD_DEG": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_SCORE_VIS_WEIGHT": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_SCORE_GYRO_WEIGHT": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_SCORE_STATE_WEIGHT": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_YAW_BUDGET_WINDOW_SEC": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_YAW_BUDGET_ABS_DEG": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_YAW_BUDGET_MIN_REMAINING_DEG": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_RECOVER_CONFIRM_HITS": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_RECOVER_MIN_SCORE": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_RECOVER_SOFT_R_MULT": "yaml.magnetometer.heading_arbitration",
        "MAG_HEADING_ARB_RECOVER_MAX_UPDATE_DYAW_DEG": "yaml.magnetometer.heading_arbitration",
        "MAG_PREPROC_ENABLE": "yaml.magnetometer.preprocessing",
        "MAG_PREPROC_NORM_RANGE_ENABLE": "yaml.magnetometer.preprocessing",
        "MAG_PREPROC_NORM_DEV_MAX": "yaml.magnetometer.preprocessing",
        "MAG_PREPROC_GYRO_DELTA_MAX_DEG": "yaml.magnetometer.preprocessing",
        "MAG_PREPROC_VISION_DELTA_MAX_DEG": "yaml.magnetometer.preprocessing",
        "MAG_PREPROC_EWMA_ALPHA": "yaml.magnetometer.preprocessing",
        "MAG_HARD_IRON_OFFSET": "yaml.magnetometer.calibration",
        "MAG_SOFT_IRON_MATRIX": "yaml.magnetometer.calibration",
        "MAG_QUALITY_NORM_EWMA_ALPHA": "yaml.magnetometer.quality",
        "MAG_QUALITY_NORM_MAD_THRESH": "yaml.magnetometer.quality",
        "MAG_QUALITY_GYRO_CONSISTENCY_DEG_S": "yaml.magnetometer.quality",
        "MAG_QUALITY_VISION_CONSISTENCY_DEG": "yaml.magnetometer.quality",
        "YAW_AUTH_ENABLE": "yaml.yaw_authority",
        "YAW_AUTH_STAGE": "yaml.yaw_authority",
        "YAW_AUTH_OWNER_MIN_DWELL_SEC": "yaml.yaw_authority",
        "YAW_AUTH_MIN_SOURCE_SCORE": "yaml.yaw_authority",
        "YAW_AUTH_SWITCH_MARGIN": "yaml.yaw_authority",
        "YAW_AUTH_SWITCH_MIN_INTERVAL_SEC": "yaml.yaw_authority",
        "YAW_AUTH_STAGE12_CLAIM_ENABLE": "yaml.yaw_authority",
        "YAW_AUTH_STAGE12_MIN_SWITCH_INTERVAL_SEC": "yaml.yaw_authority",
        "YAW_AUTH_STAGE12_OWNER_TIMEOUT_SEC": "yaml.yaw_authority",
        "YAW_AUTH_STAGE12_CLAIM_MIN_SCORE": "yaml.yaw_authority",
        "YAW_AUTH_STAGE12_CLAIM_MARGIN": "yaml.yaw_authority",
        "YAW_AUTH_STAGE12_LOOP_FORCE_CLAIM_MIN_SCORE": "yaml.yaw_authority",
        "YAW_AUTH_STAGE12_LOOP_FORCE_CLAIM_MAX_SPEED_M_S": "yaml.yaw_authority",
        "YAW_AUTH_STAGE12_ALLOW_STALE_RECLAIM_ANY": "yaml.yaw_authority",
        "YAW_AUTH_MAG_OWNER_MIN_ACCEPT_RATE": "yaml.yaw_authority",
        "YAW_AUTH_MAG_OWNER_ACCEPT_WINDOW_SEC": "yaml.yaw_authority",
        "YAW_AUTH_MAG_OWNER_MIN_SAMPLES": "yaml.yaw_authority",
        "YAW_AUTH_MAG_OWNER_BAN_SEC": "yaml.yaw_authority",
        "YAW_AUTH_OWNER_DEAD_ENABLE": "yaml.yaw_authority",
        "YAW_AUTH_OWNER_DEAD_TIMEOUT_SEC": "yaml.yaw_authority",
        "YAW_AUTH_OWNER_DEAD_HOLD_SEC": "yaml.yaw_authority",
        "YAW_AUTH_YAW_BUDGET_WINDOW_SEC": "yaml.yaw_authority",
        "YAW_AUTH_YAW_BUDGET_ABS_DEG": "yaml.yaw_authority",
        "YAW_AUTH_YAW_RATE_MAX_DEG_S": "yaml.yaw_authority",
        "YAW_AUTH_SOFT_ONLY_HIGH_SPEED_M_S": "yaml.yaw_authority",
        "YAW_AUTH_SOFT_ONLY_UNSTABLE_PMAX": "yaml.yaw_authority",
        "YAW_AUTH_SOFT_ONLY_UNSTABLE_PCOND": "yaml.yaw_authority",
        "YAW_AUTH_SOFT_ONLY_MAX_DYAW_DEG": "yaml.yaw_authority",
        "YAW_AUTH_SOFT_ONLY_R_MULT": "yaml.yaw_authority",
        "BACKEND_TRANSPORT_LATEST_WINS_ENABLE": "yaml.backend.transport",
        "BACKEND_TRANSPORT_DROP_STALE_ON_EMIT": "yaml.backend.transport",
        "BACKEND_TRANSPORT_POLL_ON_CAMERA_TICK_ONLY": "yaml.backend.transport",
        "BACKEND_TRANSPORT_POLL_MIN_INTERVAL_SEC": "yaml.backend.transport",
        "BACKEND_CONTRACT_STRICT_REQUIRE_SOURCE_MIX": "yaml.backend.contract",
        "BACKEND_CONTRACT_STRICT_REQUIRE_RESIDUAL_SUMMARY": "yaml.backend.contract",
        "VPS_MATCHER_MODE": "yaml.vps",
        "VPS_MIN_UPDATE_INTERVAL": "yaml.vps",
        "VPS_MAX_TOTAL_CANDIDATES": "yaml.vps",
        "VPS_MAX_FRAME_TIME_MS_LOCAL": "yaml.vps",
        "VPS_MAX_FRAME_TIME_MS_GLOBAL": "yaml.vps",
        "KIN_GUARD_VEL_MISMATCH_WARN": "yaml.kinematic_guard",
        "KIN_GUARD_VEL_MISMATCH_HARD": "yaml.kinematic_guard",
        "KIN_GUARD_MAX_STATE_SPEED_M_S": "yaml.kinematic_guard",
        "KIN_GUARD_MAX_INFLATE": "yaml.kinematic_guard",
        "KIN_GUARD_HARD_BLEND_ALPHA": "yaml.kinematic_guard",
        "KIN_GUARD_HARD_HOLD_SEC": "yaml.kinematic_guard",
        "KIN_GUARD_RELEASE_HYSTERESIS_RATIO": "yaml.kinematic_guard",
        "KIN_GUARD_MIN_ACTION_DT_SEC": "yaml.kinematic_guard",
        "KIN_GUARD_MAX_BLEND_SPEED_M_S": "yaml.kinematic_guard",
        "KIN_GUARD_MAX_KIN_SPEED_M_S": "yaml.kinematic_guard",
        "KIN_GUARD_SPEED_HARD_M_S": "yaml.kinematic_guard",
        "KIN_GUARD_SPEED_HARD_HOLD_SEC": "yaml.kinematic_guard",
        "KIN_GUARD_SPEED_RELEASE_HYSTERESIS_RATIO": "yaml.kinematic_guard",
        "KIN_GUARD_SPEED_BLEND_ALPHA": "yaml.kinematic_guard",
        "KIN_GUARD_SPEED_INFLATE": "yaml.kinematic_guard",
        "KIN_GUARD_ABS_SPEED_SANITY_M_S": "yaml.kinematic_guard",
        "KIN_GUARD_CERTAINTY_ENABLE": "yaml.kinematic_guard",
        "KIN_GUARD_CERTAINTY_MISMATCH_MULT": "yaml.kinematic_guard",
        "KIN_GUARD_CERTAINTY_SPEED_MULT": "yaml.kinematic_guard",
        "KIN_GUARD_CERTAINTY_INFLATE": "yaml.kinematic_guard",
        "KIN_GUARD_CERTAINTY_BLEND_ALPHA": "yaml.kinematic_guard",
        "KIN_GUARD_CERTAINTY_SPEED_CAP_M_S": "yaml.kinematic_guard",
        "KIN_GUARD_CERTAINTY_MIN_ACTION_DT_SEC": "yaml.kinematic_guard",
        "KIN_GUARD_CERTAINTY_REQUIRE_BOTH": "yaml.kinematic_guard",
        "IMU_HELPER_PERIODS": "adaptive.controller",
        "ADAPTIVE_SENSOR_R_SCALE": "adaptive.controller",
        "ADAPTIVE_SENSOR_CHI2_SCALE": "adaptive.controller",
        "PHASE_CURRENT": "phase_service",
        "HEALTH_STATE": "adaptive.controller",
        "AIDING_LEVEL": "adaptive.controller",
    }

    def __init__(self, runner: Any):
        self.runner = runner
        self._last_snapshot_t: float = float("nan")
        self._owner_map_written: bool = False

    def _current_health_state(self) -> str:
        decision = getattr(self.runner, "current_adaptive_decision", None)
        if decision is not None and hasattr(decision, "health_state"):
            return str(getattr(decision, "health_state", "HEALTHY")).upper()
        return "HEALTHY"

    def _sensor_scales_from_adaptive(self, sensor: str) -> Dict[str, float]:
        if hasattr(self.runner, "adaptive_service"):
            _, apply_scales = self.runner.adaptive_service.get_sensor_adaptive_scales(sensor)
            return dict(apply_scales)
        return {
            "r_scale": 1.0,
            "chi2_scale": 1.0,
            "threshold_scale": 1.0,
            "reproj_scale": 1.0,
        }

    def _build_sensor_decision(self, sensor: str, t: float, phase: int, health_state: str, speed_m_s: float) -> SensorPolicyDecision:
        cfg = self.runner.global_config if isinstance(self.runner.global_config, dict) else {}
        scales = self._sensor_scales_from_adaptive(sensor)
        mode = "APPLY"
        reasons = []
        extras: Dict[str, Any] = {}
        adaptive_decision = getattr(self.runner, "current_adaptive_decision", None)
        aiding_level = str(getattr(adaptive_decision, "aiding_level", "FULL")).upper()

        # Common runtime context for all sensor decisions.
        extras["phase"] = float(phase)
        extras["health_state"] = str(health_state).upper()
        extras["speed_m_s"] = float(speed_m_s)
        extras["aiding_level"] = str(aiding_level)

        msckf_q = getattr(self.runner, "_msckf_quality_snapshot", None)
        if msckf_q is not None:
            try:
                extras["msckf_quality_score"] = float(getattr(msckf_q, "quality_score", np.nan))
                extras["msckf_inlier_ratio"] = float(getattr(msckf_q, "inlier_ratio", np.nan))
                extras["msckf_reproj_p95_norm"] = float(getattr(msckf_q, "reproj_p95_norm", np.nan))
                extras["msckf_depth_positive_ratio"] = float(getattr(msckf_q, "depth_positive_ratio", np.nan))
                extras["msckf_parallax_med_px"] = float(getattr(msckf_q, "parallax_med_px", np.nan))
                extras["msckf_stable_geometry_flag"] = (
                    1.0 if bool(getattr(msckf_q, "stable_geometry_flag", False)) else 0.0
                )
                extras["msckf_conditioning_risk"] = float(getattr(msckf_q, "conditioning_risk", np.nan))
                extras["msckf_feature_track_health"] = float(
                    getattr(msckf_q, "feature_track_health", np.nan)
                )
                extras["msckf_unstable_reason_code"] = str(
                    getattr(msckf_q, "unstable_reason_code", "stable")
                )
            except Exception:
                pass

        # Carry adaptive per-sensor extras into snapshot so consumers don't read adaptive directly.
        for key, val in scales.items():
            if key in ("r_scale", "chi2_scale", "threshold_scale", "reproj_scale"):
                continue
            if isinstance(val, (int, float, np.number)):
                extras[str(key)] = float(val)

        if sensor == "VIO_VEL":
            extras["xy_only_nadir"] = 1.0 if bool(cfg.get("VIO_NADIR_XY_ONLY_VELOCITY", False)) else 0.0
            extras["max_delta_v_xy_m_s"] = float(cfg.get("VIO_VEL_MAX_DELTA_V_XY_PER_UPDATE_M_S", 2.0))
            extras["delta_v_clamp_max_ratio"] = float(cfg.get("VIO_VEL_DELTA_V_CLAMP_MAX_RATIO", 6.0))
            extras["delta_v_clamp_r_mult"] = float(cfg.get("VIO_VEL_DELTA_V_CLAMP_R_MULT", 3.5))
            extras["min_flow_px_high_speed"] = float(cfg.get("VIO_VEL_MIN_FLOW_PX_HIGH_SPEED", 0.8))
            extras["phase_chi2_scale"] = float(cfg.get("VIO_VEL_XY_ONLY_CHI2_SCALE", 1.10))
        elif sensor == "MSCKF":
            extras["reproj_failsoft_max_mult"] = float(cfg.get("MSCKF_REPROJ_FAILSOFT_MAX_MULT", 1.35))
            extras["reproj_failsoft_min_quality"] = float(cfg.get("MSCKF_REPROJ_FAILSOFT_MIN_QUALITY", 0.35))
            extras["reproj_failsoft_min_obs"] = float(cfg.get("MSCKF_REPROJ_FAILSOFT_MIN_OBS", 3))
            extras["quality_score"] = float(extras.get("msckf_quality_score", np.nan))
            extras["inlier_ratio"] = float(extras.get("msckf_inlier_ratio", np.nan))
            extras["reproj_p95_norm"] = float(extras.get("msckf_reproj_p95_norm", np.nan))
            extras["depth_positive_ratio"] = float(extras.get("msckf_depth_positive_ratio", np.nan))
            extras["parallax_med_px"] = float(extras.get("msckf_parallax_med_px", np.nan))
            extras["stable_geometry_flag"] = float(extras.get("msckf_stable_geometry_flag", 0.0))
            extras["conditioning_risk"] = float(extras.get("msckf_conditioning_risk", np.nan))
            extras["feature_track_health"] = float(extras.get("msckf_feature_track_health", np.nan))
            extras["unstable_reason_code"] = str(extras.get("msckf_unstable_reason_code", "stable"))
            msckf_stable = bool(float(extras.get("stable_geometry_flag", 0.0)) >= 0.5)
            unstable_reason = str(extras.get("unstable_reason_code", "stable"))
            extras["msckf_geometry_bucket"] = "stable" if msckf_stable else "unstable"
            extras["msckf_geometry_unstable_reason"] = unstable_reason
            if not msckf_stable:
                if unstable_reason and unstable_reason != "stable":
                    reasons.append(f"msckf_unstable:{unstable_reason}")
                unstable_mode = str(cfg.get("MSCKF_UNSTABLE_POLICY_MODE", "SOFT_APPLY")).upper()
                if mode == "APPLY" and unstable_mode in ("SOFT_APPLY", "HOLD", "SKIP"):
                    mode = unstable_mode
            phase_key = str(max(0, min(2, int(phase))))
            extras["phase_chi2_scale"] = float(cfg.get("MSCKF_PHASE_CHI2_SCALE", {}).get(phase_key, 1.0))
            extras["phase_reproj_scale"] = float(cfg.get("MSCKF_PHASE_REPROJ_SCALE", {}).get(phase_key, 1.0))
        elif sensor == "LOOP_CLOSURE":
            # Apply speed gate to both normal/fail-soft paths via explicit snapshot extras.
            extras["min_abs_yaw_corr_deg"] = float(cfg.get("LOOP_MIN_ABS_YAW_CORR_DEG", 1.5))
            extras["speed_skip_m_s_normal"] = float(cfg.get("LOOP_SPEED_SKIP_M_S_NORMAL", cfg.get("LOOP_SPEED_SKIP_M_S", 35.0)))
            extras["speed_sigma_inflate_m_s_normal"] = float(
                cfg.get("LOOP_SPEED_SIGMA_INFLATE_M_S_NORMAL", cfg.get("LOOP_SPEED_SIGMA_INFLATE_M_S", 25.0))
            )
            extras["speed_sigma_mult_normal"] = float(
                cfg.get("LOOP_SPEED_SIGMA_MULT_NORMAL", cfg.get("LOOP_SPEED_SIGMA_MULT", 1.5))
            )
            extras["speed_skip_m_s_failsoft"] = float(
                cfg.get("LOOP_SPEED_SKIP_M_S_FAILSOFT", cfg.get("LOOP_SPEED_SKIP_M_S", 35.0))
            )
            extras["speed_sigma_inflate_m_s_failsoft"] = float(
                cfg.get("LOOP_SPEED_SIGMA_INFLATE_M_S_FAILSOFT", cfg.get("LOOP_SPEED_SIGMA_INFLATE_M_S", 25.0))
            )
            extras["speed_sigma_mult_failsoft"] = float(
                cfg.get("LOOP_SPEED_SIGMA_MULT_FAILSOFT", cfg.get("LOOP_SPEED_SIGMA_MULT", 1.5))
            )
            extras["max_abs_yaw_corr_deg"] = float(cfg.get("LOOP_MAX_ABS_YAW_CORR_DEG", 4.0))
            extras["reject_abs_yaw_corr_deg"] = float(cfg.get("LOOP_YAW_RESIDUAL_BOUND_DEG", 25.0))
            extras["speed_yaw_cap_breakpoints_m_s"] = list(
                cfg.get("LOOP_SPEED_YAW_CAP_BREAKPOINTS_M_S", [20.0, 35.0, 50.0])
            )
            extras["speed_yaw_cap_normal_deg"] = list(
                cfg.get("LOOP_SPEED_YAW_CAP_NORMAL_DEG", [3.0, 2.2, 1.5])
            )
            extras["speed_yaw_cap_failsoft_deg"] = list(
                cfg.get("LOOP_SPEED_YAW_CAP_FAILSOFT_DEG", [2.5, 1.8, 1.2])
            )
            extras["base_sigma_yaw_deg"] = float(cfg.get("LOOP_BASE_SIGMA_YAW_DEG", 5.0))
            extras["fail_soft_sigma_yaw_deg"] = float(cfg.get("LOOP_FAIL_SOFT_SIGMA_YAW_DEG", 18.0))
            extras["dynamic_phase_sigma_mult"] = float(cfg.get("LOOP_DYNAMIC_PHASE_SIGMA_MULT", 1.15))
            extras["warning_sigma_mult"] = float(cfg.get("LOOP_WARNING_SIGMA_MULT", 1.20))
            extras["degraded_sigma_mult"] = float(cfg.get("LOOP_DEGRADED_SIGMA_MULT", 1.40))
            extras["apply_confirm_enable"] = 1.0 if bool(cfg.get("LOOP_APPLY_CONFIRM_ENABLE", True)) else 0.0
            extras["apply_confirm_window_sec"] = float(cfg.get("LOOP_APPLY_CONFIRM_WINDOW_SEC", 3.0))
            extras["apply_confirm_yaw_deg"] = float(cfg.get("LOOP_APPLY_CONFIRM_YAW_DEG", 6.0))
            extras["apply_confirm_hits_normal"] = float(cfg.get("LOOP_APPLY_CONFIRM_HITS_NORMAL", 1))
            extras["apply_confirm_hits_failsoft"] = float(cfg.get("LOOP_APPLY_CONFIRM_HITS_FAILSOFT", 2))
            extras["apply_confirm_speed_m_s"] = float(cfg.get("LOOP_APPLY_CONFIRM_SPEED_M_S", 25.0))
            extras["apply_confirm_extra_hits_high_speed"] = float(
                cfg.get("LOOP_APPLY_CONFIRM_EXTRA_HITS_HIGH_SPEED", 1)
            )
            extras["apply_confirm_phase_dynamic_min_hits"] = float(
                cfg.get("LOOP_APPLY_CONFIRM_PHASE_DYNAMIC_MIN_HITS", 2)
            )
            extras["cooldown_sec_normal"] = float(cfg.get("LOOP_COOLDOWN_SEC_NORMAL", cfg.get("LOOP_COOLDOWN_SEC", 2.0)))
            extras["cooldown_sec_failsoft"] = float(
                cfg.get("LOOP_COOLDOWN_SEC_FAILSOFT", max(cfg.get("LOOP_COOLDOWN_SEC", 2.0), 3.0))
            )
            extras["cooldown_speed_m_s"] = float(cfg.get("LOOP_COOLDOWN_SPEED_M_S", 25.0))
            extras["cooldown_speed_mult"] = float(cfg.get("LOOP_COOLDOWN_SPEED_MULT", 1.35))
            extras["burst_window_sec"] = float(cfg.get("LOOP_BURST_WINDOW_SEC", 12.0))
            extras["burst_max_corrections"] = float(cfg.get("LOOP_BURST_MAX_CORRECTIONS", 2))
            extras["burst_cooldown_sec"] = float(cfg.get("LOOP_BURST_COOLDOWN_SEC", 6.0))
        elif sensor == "KIN_GUARD":
            extras["vel_mismatch_warn"] = float(cfg.get("KIN_GUARD_VEL_MISMATCH_WARN", 8.0))
            extras["vel_mismatch_hard"] = float(cfg.get("KIN_GUARD_VEL_MISMATCH_HARD", 15.0))
            extras["max_state_speed_m_s"] = float(cfg.get("KIN_GUARD_MAX_STATE_SPEED_M_S", 120.0))
            extras["max_inflate"] = float(cfg.get("KIN_GUARD_MAX_INFLATE", 1.25))
            extras["hard_blend_alpha"] = float(cfg.get("KIN_GUARD_HARD_BLEND_ALPHA", 0.08))
            extras["hard_hold_sec"] = float(cfg.get("KIN_GUARD_HARD_HOLD_SEC", 0.30))
            extras["min_action_dt_sec"] = float(cfg.get("KIN_GUARD_MIN_ACTION_DT_SEC", 0.25))
            extras["release_hysteresis_ratio"] = float(cfg.get("KIN_GUARD_RELEASE_HYSTERESIS_RATIO", 0.75))
            extras["max_blend_speed_m_s"] = float(cfg.get("KIN_GUARD_MAX_BLEND_SPEED_M_S", 60.0))
            extras["max_kin_speed_m_s"] = float(cfg.get("KIN_GUARD_MAX_KIN_SPEED_M_S", 80.0))
            extras["speed_hard_m_s"] = float(cfg.get("KIN_GUARD_SPEED_HARD_M_S", extras["max_state_speed_m_s"]))
            extras["speed_hard_hold_sec"] = float(cfg.get("KIN_GUARD_SPEED_HARD_HOLD_SEC", extras["hard_hold_sec"]))
            extras["speed_release_hysteresis_ratio"] = float(
                cfg.get("KIN_GUARD_SPEED_RELEASE_HYSTERESIS_RATIO", extras["release_hysteresis_ratio"])
            )
            extras["speed_blend_alpha"] = float(cfg.get("KIN_GUARD_SPEED_BLEND_ALPHA", max(0.2, extras["hard_blend_alpha"])))
            extras["speed_inflate"] = float(cfg.get("KIN_GUARD_SPEED_INFLATE", 1.12))
            extras["abs_speed_sanity_m_s"] = float(cfg.get("KIN_GUARD_ABS_SPEED_SANITY_M_S", max(120.0, 1.8 * extras["max_state_speed_m_s"])))
            extras["certainty_enable"] = 1.0 if bool(cfg.get("KIN_GUARD_CERTAINTY_ENABLE", False)) else 0.0
            extras["certainty_mismatch_mult"] = float(cfg.get("KIN_GUARD_CERTAINTY_MISMATCH_MULT", 1.6))
            extras["certainty_speed_mult"] = float(cfg.get("KIN_GUARD_CERTAINTY_SPEED_MULT", 1.15))
            extras["certainty_inflate"] = float(cfg.get("KIN_GUARD_CERTAINTY_INFLATE", 1.35))
            extras["certainty_blend_alpha"] = float(cfg.get("KIN_GUARD_CERTAINTY_BLEND_ALPHA", 0.35))
            extras["certainty_speed_cap_m_s"] = float(cfg.get("KIN_GUARD_CERTAINTY_SPEED_CAP_M_S", extras["max_state_speed_m_s"]))
            extras["certainty_min_action_dt_sec"] = float(
                cfg.get("KIN_GUARD_CERTAINTY_MIN_ACTION_DT_SEC", extras["min_action_dt_sec"])
            )
            extras["certainty_require_both"] = 1.0 if bool(cfg.get("KIN_GUARD_CERTAINTY_REQUIRE_BOTH", False)) else 0.0
        elif sensor == "ZUPT":
            extras["acc_threshold_scale"] = float(scales.get("acc_threshold_scale", 1.0))
            extras["gyro_threshold_scale"] = float(scales.get("gyro_threshold_scale", 1.0))
            extras["vel_threshold_scale"] = float(scales.get("vel_threshold_scale", 1.0))
            extras["max_v_scale"] = float(scales.get("max_v_scale", 1.0))
            extras["fail_soft_enable"] = float(scales.get("fail_soft_enable", 0.0))
            extras["soft_r_cap"] = float(scales.get("soft_r_cap", 20.0))
            extras["hard_reject_factor"] = float(scales.get("hard_reject_factor", 3.0))
            extras["soft_r_power"] = float(scales.get("soft_r_power", 1.0))
        elif sensor == "GRAVITY_RP":
            extras["enabled_imu_only"] = float(scales.get("enabled_imu_only", 1.0))
            extras["period_steps"] = float(scales.get("period_steps", 1.0))
            extras["sigma_deg"] = float(scales.get("sigma_deg", 7.0))
            extras["acc_norm_tolerance"] = float(scales.get("acc_norm_tolerance", 0.25))
            extras["max_gyro_rad_s"] = float(scales.get("max_gyro_rad_s", 0.40))
        elif sensor == "YAW_AID":
            extras["enabled_imu_only"] = float(scales.get("enabled_imu_only", 1.0))
            extras["period_steps"] = float(scales.get("period_steps", 8.0))
            extras["sigma_deg"] = float(scales.get("sigma_deg", 35.0))
            extras["high_speed_m_s"] = float(scales.get("high_speed_m_s", 1e9))
            extras["high_speed_period_mult"] = float(scales.get("high_speed_period_mult", 1.0))
            extras["high_speed_sigma_mult"] = float(scales.get("high_speed_sigma_mult", 1.0))
            extras["acc_norm_tolerance"] = float(scales.get("acc_norm_tolerance", 0.25))
            extras["max_gyro_rad_s"] = float(scales.get("max_gyro_rad_s", 0.40))
            extras["motion_consistency_enable"] = float(scales.get("motion_consistency_enable", 0.0))
            extras["motion_min_speed_m_s"] = float(scales.get("motion_min_speed_m_s", 20.0))
            extras["motion_speed_full_m_s"] = float(scales.get("motion_speed_full_m_s", 90.0))
            extras["motion_weight_max"] = float(scales.get("motion_weight_max", 0.18))
            extras["motion_max_yaw_error_deg"] = float(scales.get("motion_max_yaw_error_deg", 70.0))
            extras["fail_soft_enable"] = float(scales.get("fail_soft_enable", 0.0))
            extras["soft_r_cap"] = float(scales.get("soft_r_cap", 12.0))
            extras["hard_reject_factor"] = float(scales.get("hard_reject_factor", 3.0))
            extras["soft_r_power"] = float(scales.get("soft_r_power", 1.0))
            extras["dynamic_ref_alpha"] = float(scales.get("dynamic_ref_alpha", 0.05))
            extras["ref_alpha"] = float(scales.get("ref_alpha", 0.005))
        elif sensor == "BIAS_GUARD":
            extras["enabled_imu_only"] = float(scales.get("enabled_imu_only", 1.0))
            extras["period_steps"] = float(scales.get("period_steps", 8.0))
            extras["enable_when_no_aiding"] = float(scales.get("enable_when_no_aiding", 1.0))
            extras["enable_when_partial_aiding"] = float(scales.get("enable_when_partial_aiding", 0.0))
            extras["enable_when_full_aiding"] = float(scales.get("enable_when_full_aiding", 0.0))
            extras["high_speed_m_s"] = float(scales.get("high_speed_m_s", 1e9))
            extras["high_speed_period_mult"] = float(scales.get("high_speed_period_mult", 1.0))
            extras["acc_norm_tolerance"] = float(scales.get("acc_norm_tolerance", 0.15))
            extras["max_gyro_rad_s"] = float(scales.get("max_gyro_rad_s", 0.25))
            extras["sigma_bg_rad_s"] = float(scales.get("sigma_bg_rad_s", np.deg2rad(0.30)))
            extras["sigma_ba_m_s2"] = float(scales.get("sigma_ba_m_s2", 0.15))
            extras["fail_soft_enable"] = float(scales.get("fail_soft_enable", 0.0))
            extras["soft_r_cap"] = float(scales.get("soft_r_cap", 8.0))
            extras["hard_reject_factor"] = float(scales.get("hard_reject_factor", 4.0))
            extras["soft_r_power"] = float(scales.get("soft_r_power", 1.0))
            extras["max_bg_norm_rad_s"] = float(scales.get("max_bg_norm_rad_s", 0.20))
            extras["max_ba_norm_m_s2"] = float(scales.get("max_ba_norm_m_s2", 2.5))
        elif sensor == "VPS":
            allow_degraded = bool(cfg.get("VPS_APPLY_FAILSOFT_ALLOW_DEGRADED", False))
            if health_state == "DEGRADED" and not allow_degraded:
                mode = "HOLD"
                reasons.append("degraded_hold")
            matcher_mode = str(cfg.get("VPS_MATCHER_MODE", "orb")).lower()
            extras["matcher_mode_orb"] = 1.0 if matcher_mode == "orb" else 0.0
            extras["matcher_mode"] = matcher_mode
            extras["min_update_interval"] = float(cfg.get("VPS_MIN_UPDATE_INTERVAL", 0.5))
            extras["max_total_candidates"] = float(cfg.get("VPS_MAX_TOTAL_CANDIDATES", 0))
            extras["max_frame_time_ms_local"] = float(cfg.get("VPS_MAX_FRAME_TIME_MS_LOCAL", 0.0))
            extras["max_frame_time_ms_global"] = float(cfg.get("VPS_MAX_FRAME_TIME_MS_GLOBAL", 0.0))
            extras["strict_min_inliers"] = float(cfg.get("VPS_APPLY_MIN_INLIERS", 8))
            extras["strict_min_conf"] = float(cfg.get("VPS_APPLY_MIN_CONFIDENCE", 0.18))
            extras["strict_max_reproj"] = float(cfg.get("VPS_APPLY_MAX_REPROJ_ERROR", 1.2))
            extras["strict_max_speed"] = float(cfg.get("VPS_APPLY_MAX_SPEED_M_S", 80.0))
            extras["warning_inlier_bonus"] = float(cfg.get("VPS_APPLY_WARNING_INLIER_BONUS", 2))
            extras["warning_conf_mult"] = float(cfg.get("VPS_APPLY_WARNING_CONF_MULT", 1.15))
            extras["warning_reproj_mult"] = float(cfg.get("VPS_APPLY_WARNING_REPROJ_MULT", 0.90))
            extras["degraded_inlier_bonus"] = float(cfg.get("VPS_APPLY_DEGRADED_INLIER_BONUS", 4))
            extras["degraded_conf_mult"] = float(cfg.get("VPS_APPLY_DEGRADED_CONF_MULT", 1.30))
            extras["degraded_reproj_mult"] = float(cfg.get("VPS_APPLY_DEGRADED_REPROJ_MULT", 0.80))
            extras["failsoft_enabled"] = 1.0 if bool(cfg.get("VPS_APPLY_FAILSOFT_ENABLE", True)) else 0.0
            extras["failsoft_allow_warning"] = 1.0 if bool(cfg.get("VPS_APPLY_FAILSOFT_ALLOW_WARNING", True)) else 0.0
            extras["failsoft_allow_degraded"] = 1.0 if bool(cfg.get("VPS_APPLY_FAILSOFT_ALLOW_DEGRADED", False)) else 0.0
            extras["failsoft_min_inliers"] = float(cfg.get("VPS_APPLY_FAILSOFT_MIN_INLIERS", 5))
            extras["failsoft_min_conf"] = float(cfg.get("VPS_APPLY_FAILSOFT_MIN_CONFIDENCE", 0.12))
            extras["failsoft_max_reproj"] = float(cfg.get("VPS_APPLY_FAILSOFT_MAX_REPROJ_ERROR", 1.2))
            extras["failsoft_max_speed"] = float(cfg.get("VPS_APPLY_FAILSOFT_MAX_SPEED_M_S", cfg.get("VPS_APPLY_MAX_SPEED_M_S", 80.0)))
            extras["failsoft_max_offset_m"] = float(cfg.get("VPS_APPLY_FAILSOFT_MAX_OFFSET_M", 180.0))
            extras["failsoft_r_mult"] = float(cfg.get("VPS_APPLY_FAILSOFT_R_MULT", 1.5))
        elif sensor == "MAG":
            extras["warning_r_mult"] = float(cfg.get("MAG_WARNING_R_MULT", 4.0))
            extras["degraded_r_mult"] = float(cfg.get("MAG_DEGRADED_R_MULT", 8.0))
            extras["warning_max_dyaw_deg"] = float(cfg.get("MAG_WARNING_MAX_DYAW_DEG", 1.5))
            extras["degraded_max_dyaw_deg"] = float(cfg.get("MAG_DEGRADED_MAX_DYAW_DEG", 1.0))
            extras["mag_mode"] = str(cfg.get("MAG_MODE", "normal")).lower()
            extras["ablation_r_mult"] = float(cfg.get("MAG_ABLATION_R_MULT", 1.0))
            extras["ablation_max_dyaw_deg"] = float(cfg.get("MAG_ABLATION_MAX_DYAW_DEG", 180.0))
            extras["ablation_max_update_dyaw_deg"] = float(
                cfg.get("MAG_ABLATION_MAX_UPDATE_DYAW_DEG", 180.0)
            )
            extras["ablation_weak_skip_hard_mismatch"] = (
                1.0 if bool(cfg.get("MAG_ABLATION_WEAK_SKIP_HARD_MISMATCH", False)) else 0.0
            )
            extras["heading_arb_enable"] = 1.0 if bool(cfg.get("MAG_HEADING_ARB_ENABLE", False)) else 0.0
            extras["heading_arb_hard_mismatch_deg"] = float(
                cfg.get("MAG_HEADING_ARB_HARD_MISMATCH_DEG", 95.0)
            )
            extras["heading_arb_min_vision_quality"] = float(
                cfg.get("MAG_HEADING_ARB_MIN_VISION_QUALITY", 0.55)
            )
            extras["heading_arb_streak_to_hold"] = float(
                cfg.get("MAG_HEADING_ARB_STREAK_TO_HOLD", 3)
            )
            extras["heading_arb_hold_sec"] = float(cfg.get("MAG_HEADING_ARB_HOLD_SEC", 1.8))
            extras["heading_arb_soft_r_mult"] = float(
                cfg.get("MAG_HEADING_ARB_SOFT_R_MULT", 2.0)
            )
            extras["heading_arb_max_vision_age_sec"] = float(
                cfg.get("MAG_HEADING_ARB_MAX_VISION_AGE_SEC", 1.0)
            )
            extras["heading_arb_score_ema_alpha"] = float(
                cfg.get("MAG_HEADING_ARB_SCORE_EMA_ALPHA", 0.20)
            )
            extras["heading_arb_score_soft_threshold"] = float(
                cfg.get("MAG_HEADING_ARB_SCORE_SOFT_THRESHOLD", 0.55)
            )
            extras["heading_arb_score_hold_threshold"] = float(
                cfg.get("MAG_HEADING_ARB_SCORE_HOLD_THRESHOLD", 0.30)
            )
            extras["heading_arb_score_vis_good_deg"] = float(
                cfg.get("MAG_HEADING_ARB_SCORE_VIS_GOOD_DEG", 18.0)
            )
            extras["heading_arb_score_gyro_good_deg"] = float(
                cfg.get("MAG_HEADING_ARB_SCORE_GYRO_GOOD_DEG", 10.0)
            )
            extras["heading_arb_score_gyro_bad_deg"] = float(
                cfg.get("MAG_HEADING_ARB_SCORE_GYRO_BAD_DEG", 50.0)
            )
            extras["heading_arb_score_state_good_deg"] = float(
                cfg.get("MAG_HEADING_ARB_SCORE_STATE_GOOD_DEG", 25.0)
            )
            extras["heading_arb_score_state_bad_deg"] = float(
                cfg.get("MAG_HEADING_ARB_SCORE_STATE_BAD_DEG", 110.0)
            )
            extras["heading_arb_score_vis_weight"] = float(
                cfg.get("MAG_HEADING_ARB_SCORE_VIS_WEIGHT", 0.50)
            )
            extras["heading_arb_score_gyro_weight"] = float(
                cfg.get("MAG_HEADING_ARB_SCORE_GYRO_WEIGHT", 0.30)
            )
            extras["heading_arb_score_state_weight"] = float(
                cfg.get("MAG_HEADING_ARB_SCORE_STATE_WEIGHT", 0.20)
            )
            extras["heading_arb_yaw_budget_window_sec"] = float(
                cfg.get("MAG_HEADING_ARB_YAW_BUDGET_WINDOW_SEC", 6.0)
            )
            extras["heading_arb_yaw_budget_abs_deg"] = float(
                cfg.get("MAG_HEADING_ARB_YAW_BUDGET_ABS_DEG", 8.0)
            )
            extras["heading_arb_yaw_budget_min_remaining_deg"] = float(
                cfg.get("MAG_HEADING_ARB_YAW_BUDGET_MIN_REMAINING_DEG", 0.6)
            )
            extras["heading_arb_recover_confirm_hits"] = float(
                cfg.get("MAG_HEADING_ARB_RECOVER_CONFIRM_HITS", 3)
            )
            extras["heading_arb_recover_min_score"] = float(
                cfg.get("MAG_HEADING_ARB_RECOVER_MIN_SCORE", 0.60)
            )
            extras["heading_arb_recover_soft_r_mult"] = float(
                cfg.get("MAG_HEADING_ARB_RECOVER_SOFT_R_MULT", 1.6)
            )
            extras["heading_arb_recover_max_update_dyaw_deg"] = float(
                cfg.get("MAG_HEADING_ARB_RECOVER_MAX_UPDATE_DYAW_DEG", 0.30)
            )
            extras["quality_norm_ewma_alpha"] = float(cfg.get("MAG_QUALITY_NORM_EWMA_ALPHA", 0.08))
            extras["quality_norm_mad_thresh"] = float(cfg.get("MAG_QUALITY_NORM_MAD_THRESH", 0.45))
            extras["quality_gyro_consistency_deg_s"] = float(
                cfg.get("MAG_QUALITY_GYRO_CONSISTENCY_DEG_S", 95.0)
            )
            extras["quality_vision_consistency_deg"] = float(
                cfg.get("MAG_QUALITY_VISION_CONSISTENCY_DEG", 120.0)
            )
            extras["conditioning_warn_pcond"] = float(cfg.get("MAG_CONDITIONING_GUARD_WARN_PCOND", 8e11))
            extras["conditioning_warn_pmax"] = float(cfg.get("MAG_CONDITIONING_GUARD_WARN_PMAX", 8e6))
            extras["conditioning_degraded_pcond"] = float(cfg.get("MAG_CONDITIONING_GUARD_DEGRADED_PCOND", 1e11))
            extras["conditioning_degraded_pmax"] = float(cfg.get("MAG_CONDITIONING_GUARD_DEGRADED_PMAX", 1e7))
            extras["conditioning_hard_pcond"] = float(cfg.get("MAG_CONDITIONING_GUARD_HARD_PCOND", 1e12))
            extras["conditioning_hard_pmax"] = float(cfg.get("MAG_CONDITIONING_GUARD_HARD_PMAX", 1e7))
            extras["conditioning_extreme_pcond"] = float(
                cfg.get("MAG_CONDITIONING_GUARD_EXTREME_PCOND", 8e12)
            )
            extras["conditioning_extreme_pmax"] = float(
                cfg.get("MAG_CONDITIONING_GUARD_EXTREME_PMAX", 4e7)
            )
            extras["conditioning_extreme_soft_enable"] = 1.0 if bool(
                cfg.get("MAG_CONDITIONING_GUARD_EXTREME_SOFT_ENABLE", True)
            ) else 0.0
            extras["conditioning_extreme_soft_r_mult"] = float(
                cfg.get("MAG_CONDITIONING_GUARD_EXTREME_SOFT_R_MULT", 2.0)
            )
            extras["conditioning_soft_enable"] = 1.0 if bool(cfg.get("MAG_CONDITIONING_GUARD_SOFT_ENABLE", True)) else 0.0
            extras["conditioning_soft_r_mult"] = float(cfg.get("MAG_CONDITIONING_GUARD_SOFT_R_MULT", 4.0))
            extras["conditioning_soft_r_mult_healthy"] = float(
                cfg.get("MAG_CONDITIONING_GUARD_SOFT_R_MULT_HEALTHY", 1.0)
            )
            extras["conditioning_soft_r_mult_warning"] = float(
                cfg.get("MAG_CONDITIONING_GUARD_SOFT_R_MULT_WARNING", 1.0)
            )
            extras["conditioning_soft_r_mult_degraded"] = float(
                cfg.get("MAG_CONDITIONING_GUARD_SOFT_R_MULT_DEGRADED", 1.0)
            )
            extras["conditioning_soft_r_mult_recovery"] = float(
                cfg.get("MAG_CONDITIONING_GUARD_SOFT_R_MULT_RECOVERY", 1.0)
            )
            extras["warning_extra_r_mult"] = float(cfg.get("MAG_WARNING_EXTRA_R_MULT", 2.0))
            extras["degraded_extra_r_mult"] = float(cfg.get("MAG_DEGRADED_EXTRA_R_MULT", 2.8))

        # Generic phase/health reason tags for traceability.
        reasons.append(f"phase={int(phase)}")
        reasons.append(f"health={health_state}")
        reasons.append(f"speed={float(speed_m_s):.2f}")

        return SensorPolicyDecision(
            sensor=str(sensor),
            mode=str(mode),
            r_scale=float(scales.get("r_scale", 1.0)),
            chi2_scale=float(scales.get("chi2_scale", 1.0)),
            threshold_scale=float(scales.get("threshold_scale", 1.0)),
            reproj_scale=float(scales.get("reproj_scale", 1.0)),
            reason_codes=tuple(reasons),
            extras=dict(extras),
            valid_from_t=float(t),
            valid_to_t=float(t + 1.0),
        )

    def build_snapshot(self, t: float) -> PolicySnapshot:
        phase = int(getattr(self.runner.state, "current_phase", 2))
        health_state = self._current_health_state()
        speed_m_s = float(np.linalg.norm(np.asarray(self.runner.kf.x[3:6, 0], dtype=float))) if getattr(self.runner, "kf", None) is not None else float("nan")
        decisions = {
            sensor: self._build_sensor_decision(sensor, float(t), phase, health_state, speed_m_s)
            for sensor in self._SENSORS
        }
        snapshot = PolicySnapshot(
            timestamp=float(t),
            phase=phase,
            health_state=health_state,
            speed_m_s=float(speed_m_s),
            decisions=decisions,
            owner_map=dict(self._OWNER_MAP),
        )
        self.runner.current_policy_snapshot = snapshot
        self._last_snapshot_t = float(t)

        if not self._owner_map_written:
            write_policy_owner_map_rows(
                getattr(self.runner, "policy_owner_map_csv", None),
                [{"key": k, "owner": v} for k, v in self._OWNER_MAP.items()],
            )
            self._owner_map_written = True

        return snapshot

    def _snapshot_stale(self, t: float) -> bool:
        snap = getattr(self.runner, "current_policy_snapshot", None)
        if snap is None:
            return True
        if not np.isfinite(float(snap.timestamp)):
            return True
        return abs(float(t) - float(snap.timestamp)) > 1e-6

    def get_sensor_decision(self, sensor: str, t: float) -> SensorPolicyDecision:
        if self._snapshot_stale(float(t)):
            self.build_snapshot(float(t))
        snap: PolicySnapshot = self.runner.current_policy_snapshot
        decision = snap.decision_for(str(sensor))
        log_policy_trace(
            getattr(self.runner, "policy_trace_csv", None),
            t=float(t),
            sensor=str(sensor),
            mode=str(decision.mode),
            phase=int(snap.phase),
            health_state=str(snap.health_state),
            speed_m_s=float(snap.speed_m_s),
            r_scale=float(decision.r_scale),
            chi2_scale=float(decision.chi2_scale),
            threshold_scale=float(decision.threshold_scale),
            reproj_scale=float(decision.reproj_scale),
            reason="|".join(decision.reason_codes),
        )
        return decision

    def record_conflict(self, sensor: str, t: float, expected_mode: str, actual_mode: str, note: str):
        self.runner._policy_conflict_count = int(getattr(self.runner, "_policy_conflict_count", 0)) + 1
        log_policy_conflict(
            getattr(self.runner, "policy_conflict_csv", None),
            t=float(t),
            sensor=str(sensor),
            expected_mode=str(expected_mode),
            actual_mode=str(actual_mode),
            note=str(note),
        )
