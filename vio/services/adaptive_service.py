"""Adaptive policy service for VIORunner."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..adaptive_controller import AdaptiveController, AdaptiveContext, AdaptiveDecision
from ..output_utils import log_adaptive_decision, log_sensor_health


class AdaptiveService:
    """Encapsulates adaptive-controller lifecycle and per-measurement feedback."""

    _DEFAULT_SENSOR_SCALES: Dict[str, float] = {
        "r_scale": 1.0,
        "chi2_scale": 1.0,
        "threshold_scale": 1.0,
        "reproj_scale": 1.0,
        "acc_threshold_scale": 1.0,
        "gyro_threshold_scale": 1.0,
        "max_v_scale": 1.0,
        "fail_soft_enable": 0.0,
        "hard_reject_factor": 3.0,
        "soft_r_cap": 20.0,
        "soft_r_power": 1.0,
        "sigma_deg": 7.0,
        "acc_norm_tolerance": 0.25,
        "max_gyro_rad_s": 0.40,
        "enabled_imu_only": 1.0,
        "ref_alpha": 0.005,
        "dynamic_ref_alpha": 0.05,
        "period_steps": 8.0,
        "motion_consistency_enable": 0.0,
        "motion_min_speed_m_s": 8.0,
        "motion_speed_full_m_s": 90.0,
        "motion_weight_max": 0.20,
        "motion_max_yaw_error_deg": 75.0,
        "sigma_bg_rad_s": np.deg2rad(0.30),
        "sigma_ba_m_s2": 0.15,
        "enable_when_no_aiding": 1.0,
        "enable_when_partial_aiding": 0.0,
        "enable_when_full_aiding": 0.0,
        "max_bg_norm_rad_s": 0.20,
        "max_ba_norm_m_s2": 2.5,
    }

    def __init__(self, runner: Any):
        self.runner = runner

    def initialize_adaptive_controller(self):
        """Initialize adaptive controller from compiled YAML config."""
        adaptive_cfg = {}
        if isinstance(self.runner.global_config, dict):
            adaptive_cfg = self.runner.global_config.get("ADAPTIVE", {})
        if self.runner.config.estimator_mode != "imu_step_preint_cache":
            adaptive_cfg = dict(adaptive_cfg)
            adaptive_cfg["mode"] = "off"
        try:
            self.runner.adaptive_controller = AdaptiveController(adaptive_cfg)
        except Exception as exc:
            print(f"[ADAPTIVE] WARNING: failed to initialize controller: {exc}")
            self.runner.adaptive_controller = AdaptiveController({"mode": "off"})

        self.runner.current_adaptive_decision = self.runner.adaptive_controller.last_decision
        self.runner._adaptive_prev_pmax = None
        self.runner._adaptive_last_aiding_time = None
        self.runner._adaptive_last_policy_t = None
        self.runner._conditioning_warning_sec = 0.0
        self.runner._adaptive_log_enabled = bool(
            adaptive_cfg.get("logging", {}).get("enabled", True)
        )
        logging_cfg = adaptive_cfg.get("logging", {}) if isinstance(adaptive_cfg, dict) else {}
        stride_debug = int(logging_cfg.get("sensor_health_stride_debug", 1))
        stride_release = int(logging_cfg.get("sensor_health_stride_release", 10))
        self.runner._sensor_health_log_stride = max(
            1,
            stride_debug if self.runner.config.save_debug_data else stride_release,
        )
        self.runner._sensor_health_log_counts = {}
        mode = self.runner.current_adaptive_decision.mode if self.runner.current_adaptive_decision else "off"
        print(
            f"[ADAPTIVE] mode={mode} (IMU-driven policy, "
            f"sensor_health_stride={self.runner._sensor_health_log_stride})"
        )

    def get_sensor_adaptive_scales(self, sensor: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get policy scales and applied scales for a sensor."""
        default = dict(self._DEFAULT_SENSOR_SCALES)
        decision = self.runner.current_adaptive_decision
        if decision is None:
            return dict(default), dict(default)
        policy = decision.sensor_scale(sensor)
        applied = dict(policy) if decision.apply_measurement else dict(default)
        return policy, applied

    def update_adaptive_policy(self, t: float, phase: int) -> Optional[AdaptiveDecision]:
        """Evaluate adaptive controller for current filter health snapshot."""
        if self.runner.adaptive_controller is None or self.runner.kf is None:
            return None

        try:
            p = self.runner.kf.P
            p_max = float(np.max(np.abs(p)))
            p_trace = float(np.trace(p))
            core_dim = min(18, p.shape[0])
            p_cond = float(np.linalg.cond(p[:core_dim, :core_dim]))
        except Exception:
            p_max = float("nan")
            p_trace = float("nan")
            p_cond = float("inf")

        if self.runner._adaptive_prev_pmax is None or not np.isfinite(self.runner._adaptive_prev_pmax) or self.runner._adaptive_prev_pmax <= 0:
            growth_ratio = 1.0
        else:
            growth_ratio = p_max / self.runner._adaptive_prev_pmax if np.isfinite(p_max) else float("nan")
        self.runner._adaptive_prev_pmax = p_max if np.isfinite(p_max) else self.runner._adaptive_prev_pmax

        if self.runner._adaptive_last_aiding_time is None:
            aiding_age_sec = 1e9
        else:
            aiding_age_sec = max(0.0, float(t - self.runner._adaptive_last_aiding_time))

        ctx = AdaptiveContext(
            timestamp=float(t),
            phase=int(phase),
            p_cond=float(p_cond),
            p_max=float(p_max),
            p_trace=float(p_trace),
            p_growth_ratio=float(growth_ratio) if np.isfinite(growth_ratio) else float("nan"),
            aiding_age_sec=float(aiding_age_sec),
        )
        decision = self.runner.adaptive_controller.step(ctx)
        self.runner.current_adaptive_decision = decision

        if self.runner._adaptive_last_policy_t is None:
            dt_policy = 0.0
        else:
            dt_policy = max(0.0, float(t - self.runner._adaptive_last_policy_t))
        self.runner._adaptive_last_policy_t = float(t)
        if decision.health_state in ("WARNING", "DEGRADED"):
            self.runner._conditioning_warning_sec += dt_policy
        else:
            self.runner._conditioning_warning_sec = 0.0

        # Keep legacy cap for off/shadow to guarantee backward compatibility.
        cov_cap = self.runner._legacy_covariance_cap
        if decision.apply_process_noise:
            cov_cap = float(decision.conditioning_max_value)
        if hasattr(self.runner.kf, "set_covariance_max_value"):
            self.runner.kf.set_covariance_max_value(cov_cap)
        if hasattr(self.runner.kf, "conditioning_cond_hard"):
            self.runner.kf.conditioning_cond_hard = float(decision.conditioning_cond_hard)
        if hasattr(self.runner.kf, "conditioning_cond_hard_window"):
            self.runner.kf.conditioning_cond_hard_window = int(max(1, decision.conditioning_cond_hard_window))
        if hasattr(self.runner.kf, "conditioning_projection_min_interval_steps"):
            self.runner.kf.conditioning_projection_min_interval_steps = int(
                max(1, decision.conditioning_projection_min_interval_steps)
            )

        if self.runner.adaptive_debug_csv and self.runner._adaptive_log_enabled:
            log_adaptive_decision(
                self.runner.adaptive_debug_csv,
                t=float(t),
                mode=decision.mode,
                health_state=decision.health_state,
                phase=int(phase),
                aiding_age_sec=float(aiding_age_sec),
                p_max=float(p_max),
                p_cond=float(p_cond),
                p_growth_ratio=float(growth_ratio) if np.isfinite(growth_ratio) else np.nan,
                sigma_accel_scale=float(decision.sigma_accel_scale),
                gyr_w_scale=float(decision.gyr_w_scale),
                acc_w_scale=float(decision.acc_w_scale),
                sigma_unmodeled_gyr_scale=float(decision.sigma_unmodeled_gyr_scale),
                min_yaw_scale=float(decision.min_yaw_scale),
                conditioning_max_value=float(cov_cap),
                reason=decision.reason,
            )

        return decision

    def build_step_imu_params(self, imu_params_base: dict, sigma_accel_base: float) -> Tuple[dict, float]:
        """Build per-step IMU params (active mode only)."""
        imu_params_step = dict(imu_params_base)
        sigma_accel_step = float(sigma_accel_base)

        decision = self.runner.current_adaptive_decision
        if decision is None or not decision.apply_process_noise:
            return imu_params_step, sigma_accel_step

        sigma_accel_step *= float(decision.sigma_accel_scale)
        if "gyr_w" in imu_params_step:
            imu_params_step["gyr_w"] = float(imu_params_step["gyr_w"]) * float(decision.gyr_w_scale)
        if "acc_w" in imu_params_step:
            imu_params_step["acc_w"] = float(imu_params_step["acc_w"]) * float(decision.acc_w_scale)
        if "sigma_unmodeled_gyr" in imu_params_step:
            imu_params_step["sigma_unmodeled_gyr"] = (
                float(imu_params_step["sigma_unmodeled_gyr"]) * float(decision.sigma_unmodeled_gyr_scale)
            )
        if "min_yaw_process_noise_deg" in imu_params_step:
            min_yaw = float(imu_params_step["min_yaw_process_noise_deg"]) * float(decision.min_yaw_scale)
            imu_params_step["min_yaw_process_noise_deg"] = max(float(decision.min_yaw_floor_deg), min_yaw)

        return imu_params_step, sigma_accel_step

    def record_adaptive_measurement(self,
                                    sensor: str,
                                    adaptive_info: Optional[Dict[str, Any]],
                                    timestamp: float,
                                    policy_scales: Optional[Dict[str, float]] = None,
                                    counts_as_aiding: bool = True):
        """Feed measurement acceptance/NIS back into adaptive controller."""
        if self.runner.adaptive_controller is None or adaptive_info is None or len(adaptive_info) == 0:
            return
        attempted = int(adaptive_info.get("attempted", 1))
        if attempted <= 0:
            return

        accepted = bool(adaptive_info.get("accepted", False))
        nis_norm = adaptive_info.get("nis_norm", np.nan)
        nis_norm = float(nis_norm) if nis_norm is not None and np.isfinite(nis_norm) else None

        feedback = self.runner.adaptive_controller.record_measurement(
            sensor=sensor,
            accepted=accepted,
            nis_norm=nis_norm,
            timestamp=float(timestamp),
        )
        if accepted and counts_as_aiding:
            self.runner._adaptive_last_aiding_time = float(timestamp)

        decision = self.runner.current_adaptive_decision
        mode = decision.mode if decision is not None else "off"
        health_state = decision.health_state if decision is not None else "HEALTHY"
        if policy_scales is None and decision is not None:
            policy_scales = decision.sensor_scale(sensor)
        if policy_scales is None:
            policy_scales = {
                "r_scale": 1.0,
                "chi2_scale": 1.0,
                "threshold_scale": 1.0,
                "reproj_scale": 1.0,
            }
        reason_code = str(adaptive_info.get("reason_code", "")).strip()
        if reason_code == "":
            reason_code = "normal_accept" if accepted else "hard_reject"

        if self.runner.sensor_health_csv and self.runner._adaptive_log_enabled:
            stride = int(getattr(self.runner, "_sensor_health_log_stride", 1))
            counts = getattr(self.runner, "_sensor_health_log_counts", None)
            if not isinstance(counts, dict):
                counts = {}
            sample_count = int(counts.get(sensor, 0)) + 1
            counts[sensor] = sample_count
            self.runner._sensor_health_log_counts = counts
            # Downsample accepted rows in light debug mode; always keep rejects/events.
            if accepted and stride > 1 and (sample_count % stride) != 0:
                return
            log_sensor_health(
                self.runner.sensor_health_csv,
                t=float(timestamp),
                sensor=sensor,
                accepted=accepted,
                nis_norm=nis_norm,
                nis_ewma=float(feedback.get("nis_ewma", 1.0)),
                accept_rate=float(feedback.get("accept_rate", 1.0)),
                mode=mode,
                health_state=health_state,
                r_scale=float(policy_scales.get("r_scale", 1.0)),
                chi2_scale=float(policy_scales.get("chi2_scale", 1.0)),
                threshold_scale=float(policy_scales.get("threshold_scale", 1.0)),
                reproj_scale=float(policy_scales.get("reproj_scale", 1.0)),
                reason_code=reason_code,
            )
