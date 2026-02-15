#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive + State-Aware policy controller for IMU-driven VIO.

This module centralizes adaptive logic so sensor-specific update code can stay
focused on measurement models while policy decisions remain configurable.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

import numpy as np


HEALTHY = "HEALTHY"
WARNING = "WARNING"
DEGRADED = "DEGRADED"
RECOVERY = "RECOVERY"

_KNOWN_SENSORS = ("MAG", "DEM", "VIO_VEL", "MSCKF", "ZUPT", "VPS", "GRAVITY_RP", "YAW_AID", "BIAS_GUARD")


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


@dataclass
class AdaptiveContext:
    timestamp: float
    phase: int
    p_cond: float
    p_max: float
    p_trace: float
    p_growth_ratio: float
    aiding_age_sec: float


@dataclass
class AdaptiveDecision:
    mode: str
    health_state: str
    apply_process_noise: bool
    apply_measurement: bool
    sigma_accel_scale: float = 1.0
    gyr_w_scale: float = 1.0
    acc_w_scale: float = 1.0
    sigma_unmodeled_gyr_scale: float = 1.0
    min_yaw_scale: float = 1.0
    min_yaw_floor_deg: float = 0.05
    conditioning_max_value: float = 1e8
    conditioning_cond_hard: float = 1e12
    conditioning_cond_hard_window: int = 8
    conditioning_projection_min_interval_steps: int = 64
    aiding_level: str = "FULL"
    measurement_scales: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reason: str = ""

    def sensor_scale(self, sensor: str) -> Dict[str, float]:
        base = {
            "r_scale": 1.0,
            "chi2_scale": 1.0,
            "threshold_scale": 1.0,
            "reproj_scale": 1.0,
        }
        base.update(self.measurement_scales.get(sensor, {}))
        return base


class AdaptiveController:
    """
    Rule+NIS hybrid adaptive controller with health-state transitions.
    """

    DEFAULTS: Dict[str, Any] = {
        "mode": "off",  # off|shadow|active
        "objective": "stability_first",
        "health": {
            "warning": {
                "p_cond": 1e10,
                "p_max": 1e6,
                "growth_ratio": 1.05,
            },
            "degraded": {
                "p_cond": 1e12,
                "p_max": 1e7,
                "growth_ratio": 1.10,
                "hold_sec": 0.25,
            },
            "recovery": {
                "healthy_sec": 3.0,
                "to_healthy_sec": 3.0,
            },
            "aiding_age": {
                "full_sec": 0.25,
                "partial_sec": 2.0,
            },
        },
        "nis_feedback": {
            "enabled": True,
            "alpha": 0.1,
            "high": 2.5,
            "low": 0.5,
            "accept_rate_hi": 0.85,
            "window": 200,
            "high_r_scale": 1.5,
            "high_chi2_scale": 0.9,
            "low_r_scale": 0.9,
            "low_chi2_scale": 1.05,
        },
        "process_noise": {
            "aiding_multiplier": {
                "FULL": 1.0,
                "PARTIAL": 0.85,
                "NONE": 0.60,
            },
            "health_profile": {
                HEALTHY: {
                    "sigma_accel_scale": 1.0,
                    "gyr_w_scale": 1.0,
                    "acc_w_scale": 1.0,
                    "sigma_unmodeled_gyr_scale": 1.0,
                    "min_yaw_scale": 1.0,
                },
                WARNING: {
                    "sigma_accel_scale": 0.85,
                    "gyr_w_scale": 0.70,
                    "acc_w_scale": 0.70,
                    "sigma_unmodeled_gyr_scale": 0.80,
                    "min_yaw_scale": 0.80,
                },
                DEGRADED: {
                    "sigma_accel_scale": 0.70,
                    "gyr_w_scale": 0.50,
                    "acc_w_scale": 0.50,
                    "sigma_unmodeled_gyr_scale": 0.65,
                    "min_yaw_scale": 0.65,
                },
                RECOVERY: {
                    "sigma_accel_scale": 0.90,
                    "gyr_w_scale": 0.80,
                    "acc_w_scale": 0.80,
                    "sigma_unmodeled_gyr_scale": 0.90,
                    "min_yaw_scale": 0.85,
                },
            },
            "min_yaw_floor_deg": 0.05,
            "scale_clamp": {
                "sigma_accel": [0.20, 2.0],
                "gyr_w": [0.10, 2.0],
                "acc_w": [0.10, 2.0],
                "sigma_unmodeled_gyr": [0.10, 2.0],
                "min_yaw": [0.20, 2.0],
            },
        },
        "measurement": {
            "sensor_defaults": {
                "MAG": {"r_scale": 1.0},
                "DEM": {"r_scale": 1.0, "threshold_scale": 1.0},
                "VIO_VEL": {"r_scale": 1.0, "chi2_scale": 1.0},
                "MSCKF": {"chi2_scale": 1.0, "reproj_scale": 1.0},
                "ZUPT": {"r_scale": 1.0, "chi2_scale": 1.0},
                "VPS": {"r_scale": 1.0, "chi2_scale": 1.0},
                "GRAVITY_RP": {"r_scale": 1.0, "chi2_scale": 1.0},
                "YAW_AID": {"r_scale": 1.0, "chi2_scale": 1.0},
                "BIAS_GUARD": {"r_scale": 1.0, "chi2_scale": 1.0},
            },
            "clamp": {
                "r_scale": [0.5, 10.0],
                "chi2_scale": [0.5, 2.0],
                "threshold_scale": [0.5, 2.0],
                "reproj_scale": [0.5, 2.0],
            },
            "degraded_r_extra": 1.2,
            "degraded_chi2_extra": 0.95,
            "phase_profiles": {
                "VIO_VEL": {
                    "0": {"chi2_scale": 1.20, "r_scale": 1.35},
                    "1": {"chi2_scale": 1.10, "r_scale": 1.20},
                    "2": {"chi2_scale": 1.00, "r_scale": 1.00},
                },
                "ZUPT": {
                    "0": {"chi2_scale": 0.80, "r_scale": 1.50, "acc_threshold_scale": 0.80, "gyro_threshold_scale": 0.80, "max_v_scale": 0.70},
                    "1": {"chi2_scale": 0.90, "r_scale": 1.20, "acc_threshold_scale": 0.90, "gyro_threshold_scale": 0.90, "max_v_scale": 0.85},
                    "2": {"chi2_scale": 1.00, "r_scale": 1.00, "acc_threshold_scale": 1.00, "gyro_threshold_scale": 1.00, "max_v_scale": 1.00},
                },
                "YAW_AID": {
                    "0": {"chi2_scale": 1.05, "r_scale": 0.90},
                    "1": {"chi2_scale": 0.90, "r_scale": 1.40},
                    "2": {"chi2_scale": 0.95, "r_scale": 1.20},
                },
                "BIAS_GUARD": {
                    "0": {"r_scale": 0.95},
                    "1": {"r_scale": 1.05},
                    "2": {"r_scale": 1.00},
                },
            },
            "zupt_fail_soft": {
                "enabled": True,
                "hard_reject_factor": 4.5,
                "max_r_scale": 40.0,
                "inflate_power": 1.0,
                "health_hard_factor": {
                    HEALTHY: 1.0,
                    WARNING: 1.15,
                    DEGRADED: 1.35,
                    RECOVERY: 1.1,
                },
                "health_r_cap_factor": {
                    HEALTHY: 1.0,
                    WARNING: 1.2,
                    DEGRADED: 1.4,
                    RECOVERY: 1.1,
                },
            },
            "vio_vel_fail_soft": {
                "enabled": True,
                "hard_reject_factor": 3.0,
                "max_r_scale": 12.0,
                "inflate_power": 1.0,
                "health_hard_factor": {
                    HEALTHY: 1.0,
                    WARNING: 1.15,
                    DEGRADED: 1.30,
                    RECOVERY: 1.10,
                },
                "health_r_cap_factor": {
                    HEALTHY: 1.0,
                    WARNING: 1.15,
                    DEGRADED: 1.30,
                    RECOVERY: 1.10,
                },
            },
            "gravity_alignment": {
                "enabled_imu_only": True,
                "phase_sigma_deg": {"0": 3.5, "1": 10.0, "2": 8.5},
                "phase_period_steps": {"0": 1, "1": 2, "2": 2},
                "phase_acc_norm_tolerance": {"0": 0.15, "1": 0.35, "2": 0.30},
                "phase_max_gyro_rad_s": {"0": 0.25, "1": 0.65, "2": 0.50},
                "health_sigma_mult": {
                    HEALTHY: 1.0,
                    WARNING: 1.2,
                    DEGRADED: 1.5,
                    RECOVERY: 1.1,
                },
                "acc_norm_tolerance": 0.25,
                "max_gyro_rad_s": 0.40,
                "chi2_scale": 1.0,
            },
            "yaw_aid": {
                "enabled_imu_only": True,
                "phase_sigma_deg": {"0": 20.0, "1": 50.0, "2": 38.0},
                "phase_min_sigma_deg": {"0": 18.0, "1": 35.0, "2": 28.0},
                "phase_period_steps": {"0": 4, "1": 24, "2": 16},
                "max_accept_rate_for_active": 0.98,
                "high_accept_backoff_factor": 1.5,
                "high_accept_r_scale": 1.1,
                "phase_acc_norm_tolerance": {"0": 0.15, "1": 0.30, "2": 0.25},
                "phase_max_gyro_rad_s": {"0": 0.20, "1": 0.55, "2": 0.40},
                "health_sigma_mult": {
                    HEALTHY: 1.0,
                    WARNING: 1.2,
                    DEGRADED: 1.5,
                    RECOVERY: 1.1,
                },
                "high_speed_m_s": 180.0,
                "high_speed_sigma_mult": 1.15,
                "high_speed_period_mult": 1.2,
                "motion_consistency_enable": True,
                "motion_min_speed_m_s": 20.0,
                "motion_speed_full_m_s": 90.0,
                "motion_weight_max": 0.18,
                "motion_max_yaw_error_deg": 70.0,
                "chi2_scale": 1.0,
                "ref_alpha": 0.005,
                "dynamic_ref_alpha": 0.05,
                "soft_fail": {
                    "enabled": True,
                    "hard_reject_factor": 3.0,
                    "max_r_scale": 12.0,
                    "inflate_power": 1.0,
                },
            },
            "bias_guard": {
                "enabled_imu_only": True,
                "apply_when_aiding_level": ["NONE"],
                "period_steps": 8,
                "phase_period_steps": {"0": 8, "1": 24, "2": 20},
                "max_accept_rate_for_active": 0.98,
                "high_accept_backoff_factor": 1.6,
                "high_accept_r_scale": 1.05,
                "phase_sigma_bg_deg_s": {"0": 0.25, "1": 0.45, "2": 0.35},
                "phase_sigma_ba_m_s2": {"0": 0.08, "1": 0.25, "2": 0.18},
                "phase_acc_norm_tolerance": {"0": 0.20, "1": 0.20, "2": 0.22},
                "phase_max_gyro_rad_s": {"0": 0.20, "1": 0.30, "2": 0.32},
                "high_speed_m_s": 180.0,
                "high_speed_period_mult": 1.2,
                "health_sigma_mult": {
                    HEALTHY: 1.0,
                    WARNING: 0.8,
                    DEGRADED: 0.6,
                    RECOVERY: 0.9,
                },
                "chi2_scale": 1.0,
                "max_bg_norm_rad_s": 0.20,
                "max_ba_norm_m_s2": 2.5,
                "soft_fail": {
                    "enabled": True,
                    "hard_reject_factor": 4.0,
                    "max_r_scale": 8.0,
                    "inflate_power": 1.0,
                },
            },
            "conditioning_backoff": {
                HEALTHY: {"period_mult": 1.0, "r_scale_mult": 1.0},
                WARNING: {"period_mult": 1.2, "r_scale_mult": 1.05},
                DEGRADED: {"period_mult": 1.5, "r_scale_mult": 1.15},
                RECOVERY: {"period_mult": 1.1, "r_scale_mult": 1.02},
            },
        },
        "conditioning": {
            "caps": {
                HEALTHY: 1e8,
                WARNING: 1e7,
                DEGRADED: 1e6,
                RECOVERY: 1e7,
            },
            "cond_hard": {
                HEALTHY: 1.2e12,
                WARNING: 1.2e12,
                DEGRADED: 1.1e12,
                RECOVERY: 1.2e12,
            },
            "cond_hard_window": {
                HEALTHY: 16,
                WARNING: 16,
                DEGRADED: 16,
                RECOVERY: 16,
            },
            "projection_min_interval_steps": {
                HEALTHY: 64,
                WARNING: 64,
                DEGRADED: 64,
                RECOVERY: 64,
            },
        },
        "logging": {
            "enabled": True,
        },
    }

    def __init__(self, adaptive_cfg: Optional[Dict[str, Any]] = None):
        self.cfg = _deep_merge(self.DEFAULTS, adaptive_cfg or {})
        self.mode = str(self.cfg.get("mode", "off")).lower()
        if self.mode not in ("off", "shadow", "active"):
            self.mode = "off"

        self.health_state = HEALTHY
        self.current_phase = 2
        self.prev_pmax: Optional[float] = None
        self.last_t: Optional[float] = None
        self.degraded_hold_accum = 0.0
        self.healthy_accum = 0.0
        self.recovery_accum = 0.0

        nis_cfg = self.cfg["nis_feedback"]
        self.nis_alpha = float(nis_cfg.get("alpha", 0.1))
        self.nis_ewma: Dict[str, float] = {sensor: 1.0 for sensor in _KNOWN_SENSORS}
        self.accept_windows: Dict[str, Deque[int]] = defaultdict(
            lambda: deque(maxlen=int(nis_cfg.get("window", 200)))
        )
        self.last_decision = AdaptiveDecision(
            mode=self.mode,
            health_state=self.health_state,
            apply_process_noise=(self.mode == "active"),
            apply_measurement=(self.mode == "active"),
        )

    def step(self, ctx: AdaptiveContext) -> AdaptiveDecision:
        self.current_phase = int(ctx.phase)
        dt = 0.0
        if self.last_t is not None:
            dt = max(0.0, float(ctx.timestamp - self.last_t))
        self.last_t = float(ctx.timestamp)

        if self.prev_pmax is None:
            growth_ratio = 1.0
        elif self.prev_pmax <= 0:
            growth_ratio = 1.0
        else:
            growth_ratio = float(ctx.p_max) / float(self.prev_pmax)
        self.prev_pmax = float(ctx.p_max)

        # Prefer context-provided ratio if caller computed it.
        if np.isfinite(ctx.p_growth_ratio) and ctx.p_growth_ratio > 0:
            growth_ratio = float(ctx.p_growth_ratio)

        warning_cfg = self.cfg["health"]["warning"]
        degraded_cfg = self.cfg["health"]["degraded"]
        recovery_cfg = self.cfg["health"]["recovery"]

        warning_condition = (
            ctx.p_cond > float(warning_cfg["p_cond"])
            or ctx.p_max > float(warning_cfg["p_max"])
            or growth_ratio > float(warning_cfg["growth_ratio"])
        )
        degraded_condition = (
            ctx.p_cond > float(degraded_cfg["p_cond"])
            or ctx.p_max > float(degraded_cfg["p_max"])
            or growth_ratio > float(degraded_cfg["growth_ratio"])
        )

        if degraded_condition:
            self.degraded_hold_accum += dt
        else:
            self.degraded_hold_accum = 0.0

        if warning_condition:
            self.healthy_accum = 0.0
            if self.health_state == RECOVERY:
                self.recovery_accum = 0.0
        else:
            self.healthy_accum += dt

        if self.health_state != DEGRADED and self.degraded_hold_accum >= float(degraded_cfg["hold_sec"]):
            self.health_state = DEGRADED
            self.recovery_accum = 0.0
        elif self.health_state == DEGRADED:
            if self.healthy_accum >= float(recovery_cfg["healthy_sec"]):
                self.health_state = RECOVERY
                self.recovery_accum = 0.0
        elif self.health_state == RECOVERY:
            if warning_condition:
                self.health_state = WARNING
                self.recovery_accum = 0.0
            else:
                self.recovery_accum += dt
                if self.recovery_accum >= float(recovery_cfg["to_healthy_sec"]):
                    self.health_state = HEALTHY
                    self.recovery_accum = 0.0
        elif warning_condition:
            self.health_state = WARNING
        else:
            self.health_state = HEALTHY

        aiding_cfg = self.cfg["health"]["aiding_age"]
        if ctx.aiding_age_sec <= float(aiding_cfg["full_sec"]):
            aiding_level = "FULL"
        elif ctx.aiding_age_sec <= float(aiding_cfg["partial_sec"]):
            aiding_level = "PARTIAL"
        else:
            aiding_level = "NONE"

        pn_cfg = self.cfg["process_noise"]
        profile = dict(pn_cfg["health_profile"].get(self.health_state, {}))
        aiding_mult = float(pn_cfg["aiding_multiplier"].get(aiding_level, 1.0))
        clamp_cfg = pn_cfg["scale_clamp"]

        sigma_accel_scale = _clamp(
            float(profile.get("sigma_accel_scale", 1.0)) * aiding_mult,
            float(clamp_cfg["sigma_accel"][0]),
            float(clamp_cfg["sigma_accel"][1]),
        )
        gyr_w_scale = _clamp(
            float(profile.get("gyr_w_scale", 1.0)) * aiding_mult,
            float(clamp_cfg["gyr_w"][0]),
            float(clamp_cfg["gyr_w"][1]),
        )
        acc_w_scale = _clamp(
            float(profile.get("acc_w_scale", 1.0)) * aiding_mult,
            float(clamp_cfg["acc_w"][0]),
            float(clamp_cfg["acc_w"][1]),
        )
        sigma_unmodeled_gyr_scale = _clamp(
            float(profile.get("sigma_unmodeled_gyr_scale", 1.0)) * aiding_mult,
            float(clamp_cfg["sigma_unmodeled_gyr"][0]),
            float(clamp_cfg["sigma_unmodeled_gyr"][1]),
        )
        min_yaw_scale = _clamp(
            float(profile.get("min_yaw_scale", 1.0)) * aiding_mult,
            float(clamp_cfg["min_yaw"][0]),
            float(clamp_cfg["min_yaw"][1]),
        )

        conditioning_caps = self.cfg["conditioning"]["caps"]
        conditioning_cap = float(conditioning_caps.get(self.health_state, 1e8))
        cond_hard_cfg = self.cfg["conditioning"].get("cond_hard", {})
        cond_hard = float(cond_hard_cfg.get(self.health_state, 1e12))
        cond_window_cfg = self.cfg["conditioning"].get("cond_hard_window", {})
        cond_hard_window = int(max(1, float(cond_window_cfg.get(self.health_state, 8))))
        proj_interval_cfg = self.cfg["conditioning"].get("projection_min_interval_steps", {})
        projection_min_interval = int(max(1, float(proj_interval_cfg.get(self.health_state, 64))))

        measurement_scales = self._build_measurement_scales()
        reason = (
            f"health={self.health_state}, aiding={aiding_level}, "
            f"growth={growth_ratio:.3f}, p_max={ctx.p_max:.2e}, p_cond={ctx.p_cond:.2e}"
        )

        decision = AdaptiveDecision(
            mode=self.mode,
            health_state=self.health_state,
            apply_process_noise=(self.mode == "active"),
            apply_measurement=(self.mode == "active"),
            sigma_accel_scale=sigma_accel_scale,
            gyr_w_scale=gyr_w_scale,
            acc_w_scale=acc_w_scale,
            sigma_unmodeled_gyr_scale=sigma_unmodeled_gyr_scale,
            min_yaw_scale=min_yaw_scale,
            min_yaw_floor_deg=float(pn_cfg.get("min_yaw_floor_deg", 0.05)),
            conditioning_max_value=conditioning_cap,
            conditioning_cond_hard=cond_hard,
            conditioning_cond_hard_window=cond_hard_window,
            conditioning_projection_min_interval_steps=projection_min_interval,
            aiding_level=aiding_level,
            measurement_scales=measurement_scales,
            reason=reason,
        )
        self.last_decision = decision
        return decision

    def _build_measurement_scales(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        meas_cfg = self.cfg["measurement"]
        nis_cfg = self.cfg["nis_feedback"]
        clamp_cfg = meas_cfg["clamp"]
        phase_profiles = meas_cfg.get("phase_profiles", {})
        phase_key = str(max(0, min(2, int(self.current_phase))))
        do_feedback = bool(nis_cfg.get("enabled", True))
        hi = float(nis_cfg.get("high", 2.5))
        lo = float(nis_cfg.get("low", 0.5))
        accept_rate_hi = float(nis_cfg.get("accept_rate_hi", 0.85))
        high_r = float(nis_cfg.get("high_r_scale", 1.5))
        high_chi2 = float(nis_cfg.get("high_chi2_scale", 0.9))
        low_r = float(nis_cfg.get("low_r_scale", 0.9))
        low_chi2 = float(nis_cfg.get("low_chi2_scale", 1.05))

        for sensor, defaults in meas_cfg["sensor_defaults"].items():
            sensor_scales = dict(defaults)
            sensor_phase_cfg = phase_profiles.get(sensor, {}).get(phase_key, {})
            for key, scale_val in sensor_phase_cfg.items():
                sensor_scales[key] = sensor_scales.get(key, 1.0) * float(scale_val)
            nis = float(self.nis_ewma.get(sensor, 1.0))
            history = self.accept_windows[sensor]
            accept_rate = float(sum(history) / len(history)) if len(history) > 0 else 1.0

            if do_feedback:
                if nis > hi:
                    sensor_scales["r_scale"] = sensor_scales.get("r_scale", 1.0) * high_r
                    sensor_scales["chi2_scale"] = sensor_scales.get("chi2_scale", 1.0) * high_chi2
                    sensor_scales["threshold_scale"] = sensor_scales.get("threshold_scale", 1.0) * high_chi2
                    sensor_scales["reproj_scale"] = sensor_scales.get("reproj_scale", 1.0) * high_chi2
                elif nis < lo and accept_rate >= accept_rate_hi:
                    sensor_scales["r_scale"] = sensor_scales.get("r_scale", 1.0) * low_r
                    sensor_scales["chi2_scale"] = sensor_scales.get("chi2_scale", 1.0) * low_chi2
                    sensor_scales["threshold_scale"] = sensor_scales.get("threshold_scale", 1.0) * low_chi2
                    sensor_scales["reproj_scale"] = sensor_scales.get("reproj_scale", 1.0) * low_chi2

            if self.health_state == DEGRADED:
                sensor_scales["r_scale"] = sensor_scales.get("r_scale", 1.0) * float(
                    meas_cfg.get("degraded_r_extra", 1.2)
                )
                sensor_scales["chi2_scale"] = sensor_scales.get("chi2_scale", 1.0) * float(
                    meas_cfg.get("degraded_chi2_extra", 0.95)
                )
                sensor_scales["threshold_scale"] = sensor_scales.get("threshold_scale", 1.0) * float(
                    meas_cfg.get("degraded_chi2_extra", 0.95)
                )
                sensor_scales["reproj_scale"] = sensor_scales.get("reproj_scale", 1.0) * float(
                    meas_cfg.get("degraded_chi2_extra", 0.95)
                )

            if sensor == "ZUPT":
                zupt_cfg = meas_cfg.get("zupt_fail_soft", {})
                hard_factor = float(zupt_cfg.get("hard_reject_factor", 3.0))
                hard_factor *= float(
                    zupt_cfg.get("health_hard_factor", {}).get(self.health_state, 1.0)
                )
                r_cap = float(zupt_cfg.get("max_r_scale", 20.0))
                r_cap *= float(
                    zupt_cfg.get("health_r_cap_factor", {}).get(self.health_state, 1.0)
                )
                sensor_scales["fail_soft_enable"] = 1.0 if bool(zupt_cfg.get("enabled", True)) else 0.0
                sensor_scales["hard_reject_factor"] = max(1.0, hard_factor)
                sensor_scales["soft_r_cap"] = max(1.0, r_cap)
                sensor_scales["soft_r_power"] = max(0.1, float(zupt_cfg.get("inflate_power", 1.0)))

            if sensor == "VIO_VEL":
                vio_soft_cfg = meas_cfg.get("vio_vel_fail_soft", {})
                hard_factor = float(vio_soft_cfg.get("hard_reject_factor", 3.0))
                hard_factor *= float(
                    vio_soft_cfg.get("health_hard_factor", {}).get(self.health_state, 1.0)
                )
                r_cap = float(vio_soft_cfg.get("max_r_scale", 12.0))
                r_cap *= float(
                    vio_soft_cfg.get("health_r_cap_factor", {}).get(self.health_state, 1.0)
                )
                # State-aware fail-soft boost:
                # If VIO_VEL enters reject burst / high-NIS mode, relax gate and
                # increase soft-fail envelope instead of hard-reject bursts.
                burst_accept_rate_low = float(vio_soft_cfg.get("burst_accept_rate_low", 0.35))
                burst_nis_high = float(vio_soft_cfg.get("burst_nis_high", 5.0))
                if accept_rate <= burst_accept_rate_low and nis >= burst_nis_high:
                    sensor_scales["chi2_scale"] = sensor_scales.get("chi2_scale", 1.0) * float(
                        vio_soft_cfg.get("burst_chi2_relax", 1.35)
                    )
                    sensor_scales["threshold_scale"] = sensor_scales.get("threshold_scale", 1.0) * float(
                        vio_soft_cfg.get("burst_threshold_relax", 1.35)
                    )
                    sensor_scales["r_scale"] = sensor_scales.get("r_scale", 1.0) * float(
                        vio_soft_cfg.get("burst_r_boost", 1.25)
                    )

                if nis >= hi:
                    sensor_scales["chi2_scale"] = sensor_scales.get("chi2_scale", 1.0) * float(
                        vio_soft_cfg.get("high_nis_relax_chi2_mult", 1.15)
                    )
                    sensor_scales["threshold_scale"] = sensor_scales.get("threshold_scale", 1.0) * float(
                        vio_soft_cfg.get("high_nis_relax_threshold_mult", 1.15)
                    )

                health_relax = float(
                    vio_soft_cfg.get("health_chi2_relax", {}).get(self.health_state, 1.0)
                )
                sensor_scales["chi2_scale"] = sensor_scales.get("chi2_scale", 1.0) * health_relax
                sensor_scales["threshold_scale"] = sensor_scales.get("threshold_scale", 1.0) * health_relax

                sensor_scales["fail_soft_enable"] = 1.0 if bool(vio_soft_cfg.get("enabled", True)) else 0.0
                sensor_scales["hard_reject_factor"] = max(1.0, hard_factor)
                sensor_scales["soft_r_cap"] = max(1.0, r_cap)
                sensor_scales["soft_r_power"] = max(0.1, float(vio_soft_cfg.get("inflate_power", 1.0)))

            if sensor == "GRAVITY_RP":
                grav_cfg = meas_cfg.get("gravity_alignment", {})
                sigma_deg = float(
                    grav_cfg.get("phase_sigma_deg", {}).get(phase_key, grav_cfg.get("sigma_deg", 7.0))
                )
                sigma_deg *= float(
                    grav_cfg.get("health_sigma_mult", {}).get(self.health_state, 1.0)
                )
                sensor_scales["sigma_deg"] = max(0.5, sigma_deg)
                phase_acc_tol = grav_cfg.get("phase_acc_norm_tolerance", {}).get(
                    phase_key, grav_cfg.get("acc_norm_tolerance", 0.25)
                )
                sensor_scales["acc_norm_tolerance"] = max(
                    0.05, float(phase_acc_tol)
                )
                phase_max_gyro = grav_cfg.get("phase_max_gyro_rad_s", {}).get(
                    phase_key, grav_cfg.get("max_gyro_rad_s", 0.40)
                )
                sensor_scales["max_gyro_rad_s"] = max(
                    0.05, float(phase_max_gyro)
                )
                sensor_scales["enabled_imu_only"] = 1.0 if bool(
                    grav_cfg.get("enabled_imu_only", True)
                ) else 0.0
                grav_period = grav_cfg.get("phase_period_steps", {}).get(
                    phase_key, grav_cfg.get("period_steps", 1)
                )
                sensor_scales["period_steps"] = max(1.0, float(grav_period))
                sensor_scales["chi2_scale"] = sensor_scales.get("chi2_scale", 1.0) * float(
                    grav_cfg.get("chi2_scale", 1.0)
                )

            if sensor == "YAW_AID":
                yaw_cfg = meas_cfg.get("yaw_aid", {})
                sigma_deg = float(
                    yaw_cfg.get("phase_sigma_deg", {}).get(phase_key, yaw_cfg.get("sigma_deg", 35.0))
                )
                sigma_deg *= float(
                    yaw_cfg.get("health_sigma_mult", {}).get(self.health_state, 1.0)
                )
                sigma_floor = float(
                    yaw_cfg.get("phase_min_sigma_deg", {}).get(
                        phase_key, yaw_cfg.get("min_sigma_deg", 5.0)
                    )
                )
                sensor_scales["sigma_deg"] = max(sigma_floor, sigma_deg)
                phase_acc_tol = yaw_cfg.get("phase_acc_norm_tolerance", {}).get(
                    phase_key, yaw_cfg.get("acc_norm_tolerance", 0.25)
                )
                sensor_scales["acc_norm_tolerance"] = max(0.05, float(phase_acc_tol))
                phase_max_gyro = yaw_cfg.get("phase_max_gyro_rad_s", {}).get(
                    phase_key, yaw_cfg.get("max_gyro_rad_s", 0.40)
                )
                sensor_scales["max_gyro_rad_s"] = max(0.05, float(phase_max_gyro))
                sensor_scales["enabled_imu_only"] = 1.0 if bool(
                    yaw_cfg.get("enabled_imu_only", True)
                ) else 0.0
                sensor_scales["chi2_scale"] = sensor_scales.get("chi2_scale", 1.0) * float(
                    yaw_cfg.get("chi2_scale", 1.0)
                )
                sensor_scales["high_speed_m_s"] = max(1.0, float(yaw_cfg.get("high_speed_m_s", 60.0)))
                sensor_scales["high_speed_sigma_mult"] = max(
                    1.0, float(yaw_cfg.get("high_speed_sigma_mult", 1.0))
                )
                sensor_scales["high_speed_period_mult"] = max(
                    1.0, float(yaw_cfg.get("high_speed_period_mult", 1.0))
                )
                sensor_scales["motion_consistency_enable"] = 1.0 if bool(
                    yaw_cfg.get("motion_consistency_enable", False)
                ) else 0.0
                sensor_scales["motion_min_speed_m_s"] = max(
                    0.1, float(yaw_cfg.get("motion_min_speed_m_s", 20.0))
                )
                sensor_scales["motion_speed_full_m_s"] = max(
                    float(sensor_scales["motion_min_speed_m_s"]) + 1e-3,
                    float(yaw_cfg.get("motion_speed_full_m_s", 90.0)),
                )
                sensor_scales["motion_weight_max"] = _clamp(
                    float(yaw_cfg.get("motion_weight_max", 0.18)), 0.0, 0.5
                )
                sensor_scales["motion_max_yaw_error_deg"] = _clamp(
                    float(yaw_cfg.get("motion_max_yaw_error_deg", 70.0)), 5.0, 179.0
                )
                sensor_scales["ref_alpha"] = max(0.0, float(yaw_cfg.get("ref_alpha", 0.005)))
                sensor_scales["dynamic_ref_alpha"] = max(
                    0.0, float(yaw_cfg.get("dynamic_ref_alpha", 0.05))
                )
                yaw_period = yaw_cfg.get("phase_period_steps", {}).get(
                    phase_key, yaw_cfg.get("period_steps", 8)
                )
                max_accept = float(yaw_cfg.get("max_accept_rate_for_active", 1.01))
                if accept_rate >= max_accept:
                    yaw_period = float(yaw_period) * float(yaw_cfg.get("high_accept_backoff_factor", 1.5))
                    sensor_scales["r_scale"] = sensor_scales.get("r_scale", 1.0) * float(
                        yaw_cfg.get("high_accept_r_scale", 1.1)
                    )
                sensor_scales["period_steps"] = max(1.0, float(yaw_period))
                yaw_soft_cfg = yaw_cfg.get("soft_fail", {})
                sensor_scales["fail_soft_enable"] = 1.0 if bool(
                    yaw_soft_cfg.get("enabled", True)
                ) else 0.0
                sensor_scales["hard_reject_factor"] = max(
                    1.0, float(yaw_soft_cfg.get("hard_reject_factor", 3.0))
                )
                sensor_scales["soft_r_cap"] = max(1.0, float(yaw_soft_cfg.get("max_r_scale", 12.0)))
                sensor_scales["soft_r_power"] = max(0.1, float(yaw_soft_cfg.get("inflate_power", 1.0)))

            if sensor == "BIAS_GUARD":
                bias_cfg = meas_cfg.get("bias_guard", {})
                sigma_bg_deg_s = float(
                    bias_cfg.get("phase_sigma_bg_deg_s", {}).get(
                        phase_key, bias_cfg.get("sigma_bg_deg_s", 0.30)
                    )
                )
                sigma_ba = float(
                    bias_cfg.get("phase_sigma_ba_m_s2", {}).get(
                        phase_key, bias_cfg.get("sigma_ba_m_s2", 0.15)
                    )
                )
                sigma_mult = float(
                    bias_cfg.get("health_sigma_mult", {}).get(self.health_state, 1.0)
                )
                sensor_scales["sigma_bg_rad_s"] = np.deg2rad(max(0.01, sigma_bg_deg_s)) * sigma_mult
                sensor_scales["sigma_ba_m_s2"] = max(1e-4, sigma_ba * sigma_mult)
                sensor_scales["acc_norm_tolerance"] = max(
                    0.01,
                    float(
                        bias_cfg.get("phase_acc_norm_tolerance", {}).get(
                            phase_key, bias_cfg.get("acc_norm_tolerance", 0.15)
                        )
                    ),
                )
                sensor_scales["max_gyro_rad_s"] = max(
                    0.01,
                    float(
                        bias_cfg.get("phase_max_gyro_rad_s", {}).get(
                            phase_key, bias_cfg.get("max_gyro_rad_s", 0.25)
                        )
                    ),
                )
                sensor_scales["high_speed_m_s"] = max(1.0, float(bias_cfg.get("high_speed_m_s", 45.0)))
                sensor_scales["high_speed_period_mult"] = max(
                    1.0, float(bias_cfg.get("high_speed_period_mult", 1.0))
                )
                base_period = bias_cfg.get("phase_period_steps", {}).get(
                    phase_key, bias_cfg.get("period_steps", 8)
                )
                max_accept = float(bias_cfg.get("max_accept_rate_for_active", 1.01))
                if accept_rate >= max_accept:
                    base_period = float(base_period) * float(bias_cfg.get("high_accept_backoff_factor", 2.0))
                    sensor_scales["r_scale"] = sensor_scales.get("r_scale", 1.0) * float(
                        bias_cfg.get("high_accept_r_scale", 1.1)
                    )
                sensor_scales["period_steps"] = max(1.0, float(base_period))
                sensor_scales["enabled_imu_only"] = 1.0 if bool(
                    bias_cfg.get("enabled_imu_only", True)
                ) else 0.0
                allowed_levels = bias_cfg.get("apply_when_aiding_level", ["NONE"])
                sensor_scales["enable_when_no_aiding"] = 1.0 if "NONE" in allowed_levels else 0.0
                sensor_scales["enable_when_partial_aiding"] = 1.0 if "PARTIAL" in allowed_levels else 0.0
                sensor_scales["enable_when_full_aiding"] = 1.0 if "FULL" in allowed_levels else 0.0
                sensor_scales["chi2_scale"] = sensor_scales.get("chi2_scale", 1.0) * float(
                    bias_cfg.get("chi2_scale", 1.0)
                )
                sensor_scales["max_bg_norm_rad_s"] = max(
                    1e-3, float(bias_cfg.get("max_bg_norm_rad_s", 0.20))
                )
                sensor_scales["max_ba_norm_m_s2"] = max(
                    1e-3, float(bias_cfg.get("max_ba_norm_m_s2", 2.5))
                )
                bias_soft_cfg = bias_cfg.get("soft_fail", {})
                sensor_scales["fail_soft_enable"] = 1.0 if bool(
                    bias_soft_cfg.get("enabled", True)
                ) else 0.0
                sensor_scales["hard_reject_factor"] = max(
                    1.0, float(bias_soft_cfg.get("hard_reject_factor", 4.0))
                )
                sensor_scales["soft_r_cap"] = max(1.0, float(bias_soft_cfg.get("max_r_scale", 8.0)))
                sensor_scales["soft_r_power"] = max(0.1, float(bias_soft_cfg.get("inflate_power", 1.0)))

            if sensor in ("GRAVITY_RP", "YAW_AID", "BIAS_GUARD"):
                backoff_cfg = meas_cfg.get("conditioning_backoff", {}).get(self.health_state, {})
                period_mult = float(backoff_cfg.get("period_mult", 1.0))
                r_mult = float(backoff_cfg.get("r_scale_mult", 1.0))
                if "period_steps" in sensor_scales:
                    sensor_scales["period_steps"] = max(1.0, float(sensor_scales["period_steps"]) * period_mult)
                sensor_scales["r_scale"] = sensor_scales.get("r_scale", 1.0) * r_mult

            # Clamp values.
            for key in ("r_scale", "chi2_scale", "threshold_scale", "reproj_scale"):
                if key not in sensor_scales:
                    continue
                lim = clamp_cfg.get(key, [0.5, 2.0] if key != "r_scale" else [0.5, 10.0])
                sensor_scales[key] = _clamp(float(sensor_scales[key]), float(lim[0]), float(lim[1]))

            out[sensor] = sensor_scales

        return out

    def record_measurement(
        self,
        sensor: str,
        accepted: bool,
        nis_norm: Optional[float],
        timestamp: float,
    ) -> Dict[str, float]:
        sensor_key = sensor if sensor in _KNOWN_SENSORS else sensor.upper()
        if sensor_key not in _KNOWN_SENSORS:
            sensor_key = sensor
            if sensor_key not in self.nis_ewma:
                self.nis_ewma[sensor_key] = 1.0
                self.accept_windows[sensor_key] = deque(maxlen=int(self.cfg["nis_feedback"]["window"]))

        self.accept_windows[sensor_key].append(1 if accepted else 0)
        if nis_norm is not None and np.isfinite(nis_norm):
            prev = float(self.nis_ewma.get(sensor_key, 1.0))
            alpha = float(self.nis_alpha)
            self.nis_ewma[sensor_key] = (1.0 - alpha) * prev + alpha * float(nis_norm)

        hist = self.accept_windows[sensor_key]
        accept_rate = float(sum(hist) / len(hist)) if len(hist) > 0 else 1.0
        return {
            "timestamp": float(timestamp),
            "sensor": sensor_key,
            "accepted": float(1 if accepted else 0),
            "nis_ewma": float(self.nis_ewma.get(sensor_key, 1.0)),
            "accept_rate": accept_rate,
        }

    def get_measurement_scale(self, sensor: str) -> Dict[str, float]:
        return self.last_decision.sensor_scale(sensor)
