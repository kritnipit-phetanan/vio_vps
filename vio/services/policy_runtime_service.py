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

    _SENSORS = ("MAG", "DEM", "VIO_VEL", "MSCKF", "LOOP_CLOSURE", "VPS", "ZUPT", "KIN_GUARD")

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
        "KIN_GUARD_VEL_MISMATCH_WARN": "yaml.kinematic_guard",
        "KIN_GUARD_VEL_MISMATCH_HARD": "yaml.kinematic_guard",
        "KIN_GUARD_MAX_STATE_SPEED_M_S": "yaml.kinematic_guard",
        "ADAPTIVE_SENSOR_R_SCALE": "adaptive.controller",
        "ADAPTIVE_SENSOR_CHI2_SCALE": "adaptive.controller",
        "PHASE_CURRENT": "phase_service",
        "HEALTH_STATE": "adaptive.controller",
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
        extras: Dict[str, float] = {}

        if sensor == "VIO_VEL":
            extras["xy_only_nadir"] = 1.0 if bool(cfg.get("VIO_NADIR_XY_ONLY_VELOCITY", False)) else 0.0
            extras["max_delta_v_xy_m_s"] = float(cfg.get("VIO_VEL_MAX_DELTA_V_XY_PER_UPDATE_M_S", 2.0))
            extras["delta_v_clamp_max_ratio"] = float(cfg.get("VIO_VEL_DELTA_V_CLAMP_MAX_RATIO", 6.0))
            extras["delta_v_clamp_r_mult"] = float(cfg.get("VIO_VEL_DELTA_V_CLAMP_R_MULT", 3.5))
            extras["min_flow_px_high_speed"] = float(cfg.get("VIO_VEL_MIN_FLOW_PX_HIGH_SPEED", 0.8))
        elif sensor == "MSCKF":
            extras["reproj_failsoft_max_mult"] = float(cfg.get("MSCKF_REPROJ_FAILSOFT_MAX_MULT", 1.35))
            extras["reproj_failsoft_min_quality"] = float(cfg.get("MSCKF_REPROJ_FAILSOFT_MIN_QUALITY", 0.35))
            extras["reproj_failsoft_min_obs"] = float(cfg.get("MSCKF_REPROJ_FAILSOFT_MIN_OBS", 3))
        elif sensor == "LOOP_CLOSURE":
            # Apply speed gate to both normal/fail-soft paths via explicit snapshot extras.
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
        elif sensor == "KIN_GUARD":
            extras["vel_mismatch_warn"] = float(cfg.get("KIN_GUARD_VEL_MISMATCH_WARN", 8.0))
            extras["vel_mismatch_hard"] = float(cfg.get("KIN_GUARD_VEL_MISMATCH_HARD", 15.0))
            extras["max_state_speed_m_s"] = float(cfg.get("KIN_GUARD_MAX_STATE_SPEED_M_S", 120.0))
            extras["hard_hold_sec"] = float(cfg.get("KIN_GUARD_HARD_HOLD_SEC", 0.30))
            extras["release_hysteresis_ratio"] = float(cfg.get("KIN_GUARD_RELEASE_HYSTERESIS_RATIO", 0.75))
        elif sensor == "VPS":
            allow_degraded = bool(cfg.get("VPS_APPLY_FAILSOFT_ALLOW_DEGRADED", False))
            if health_state == "DEGRADED" and not allow_degraded:
                mode = "HOLD"
                reasons.append("degraded_hold")
            extras["matcher_mode_orb"] = 1.0 if str(cfg.get("VPS_MATCHER_MODE", "orb")).lower() == "orb" else 0.0
        elif sensor == "MAG":
            extras["warning_r_mult"] = float(cfg.get("MAG_WARNING_R_MULT", 4.0))
            extras["degraded_r_mult"] = float(cfg.get("MAG_DEGRADED_R_MULT", 8.0))

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
