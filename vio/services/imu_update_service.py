"""IMU helper-update service for VIORunner."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..policy.types import SensorPolicyDecision
from ..math_utils import quaternion_to_yaw
from ..propagation import (
    apply_bias_observability_guard,
    apply_gravity_roll_pitch_update,
    apply_yaw_pseudo_update,
    apply_zupt,
    detect_stationary,
)


class IMUUpdateService:
    """Encapsulates IMU helper updates (ZUPT + IMU-only stabilization helpers)."""

    def __init__(self, runner: Any):
        self.runner = runner

    def _policy_for_sensor(self, sensor: str, t: float) -> tuple[SensorPolicyDecision, Dict[str, float]]:
        decision = SensorPolicyDecision(sensor=str(sensor))
        if getattr(self.runner, "policy_runtime_service", None) is not None:
            try:
                decision = self.runner.policy_runtime_service.get_sensor_decision(str(sensor), float(t))
            except Exception:
                decision = SensorPolicyDecision(sensor=str(sensor))
        policy_scales = {
            "r_scale": float(decision.r_scale),
            "chi2_scale": float(decision.chi2_scale),
            "threshold_scale": float(decision.threshold_scale),
            "reproj_scale": float(decision.reproj_scale),
        }
        return decision, policy_scales

    @staticmethod
    def _decision_is_hold_skip(decision: SensorPolicyDecision) -> bool:
        mode = str(getattr(decision, "mode", "APPLY")).upper()
        return mode in ("HOLD", "SKIP")

    def update_imu_helpers(
        self,
        rec,
        dt: float,
        imu_params: dict,
        zupt_scales: Optional[Dict[str, float]] = None,
    ):
        """
        Update IMU-related helper variables shared by propagation paths.

        Args:
            rec: IMU record
            dt: Time delta since last sample
            imu_params: IMU parameters dict
            zupt_scales: Optional override scales for ZUPT (per-step)
        """
        runner = self.runner

        # Get current state
        x = runner.kf.x.reshape(-1)
        bg = x[10:13]

        # Bias-corrected angular rate
        w_corr = rec.ang.astype(float) - bg

        # Store gyro_z and dt for mag filtering (critical for preintegration mode)
        runner.last_gyro_z = float(w_corr[2])
        runner.last_imu_dt = dt

        # Check stationary using phase-aware thresholds for ZUPT gating
        v_mag = np.linalg.norm(x[3:6])
        zupt_enabled = bool(runner.global_config.get("ZUPT_ENABLED", True))
        base_acc_threshold = float(runner.global_config.get("ZUPT_ACCEL_THRESHOLD", 0.5))
        base_gyro_threshold = float(runner.global_config.get("ZUPT_GYRO_THRESHOLD", 0.05))
        base_vel_threshold = float(runner.global_config.get("ZUPT_VELOCITY_THRESHOLD", 0.3))
        base_max_v_for_zupt = float(runner.global_config.get("ZUPT_MAX_V_FOR_UPDATE", 20.0))
        zupt_decision, zupt_policy_scales = self._policy_for_sensor("ZUPT", float(rec.t))
        acc_threshold_scale = float(zupt_decision.extra("acc_threshold_scale", 1.0))
        gyro_threshold_scale = float(zupt_decision.extra("gyro_threshold_scale", 1.0))
        vel_threshold_scale = float(zupt_decision.extra("vel_threshold_scale", 1.0))
        max_v_scale = float(zupt_decision.extra("max_v_scale", 1.0))
        zupt_r_scale = float(zupt_decision.r_scale)
        zupt_chi2_scale = float(zupt_decision.chi2_scale)
        zupt_fail_soft_enable = bool(float(zupt_decision.extra("fail_soft_enable", 0.0)) >= 0.5)
        zupt_soft_r_cap = float(zupt_decision.extra("soft_r_cap", 20.0))
        zupt_hard_reject_factor = float(zupt_decision.extra("hard_reject_factor", 3.0))
        zupt_soft_r_power = float(zupt_decision.extra("soft_r_power", 1.0))

        # Optional override from caller (legacy hook), applied on top of snapshot decision.
        if zupt_scales is not None:
            acc_threshold_scale = float(zupt_scales.get("acc_threshold_scale", acc_threshold_scale))
            gyro_threshold_scale = float(zupt_scales.get("gyro_threshold_scale", gyro_threshold_scale))
            vel_threshold_scale = float(zupt_scales.get("vel_threshold_scale", vel_threshold_scale))
            max_v_scale = float(zupt_scales.get("max_v_scale", max_v_scale))
            zupt_r_scale = float(zupt_scales.get("r_scale", zupt_r_scale))
            zupt_chi2_scale = float(zupt_scales.get("chi2_scale", zupt_chi2_scale))
            zupt_fail_soft_enable = bool(float(zupt_scales.get("fail_soft_enable", float(zupt_fail_soft_enable))) >= 0.5)
            zupt_soft_r_cap = float(zupt_scales.get("soft_r_cap", zupt_soft_r_cap))
            zupt_hard_reject_factor = float(zupt_scales.get("hard_reject_factor", zupt_hard_reject_factor))
            zupt_soft_r_power = float(zupt_scales.get("soft_r_power", zupt_soft_r_power))

        acc_threshold = base_acc_threshold * acc_threshold_scale
        gyro_threshold = base_gyro_threshold * gyro_threshold_scale

        is_stationary, _ = detect_stationary(
            a_raw=rec.lin,
            w_corr=w_corr,
            v_mag=v_mag,
            imu_params=imu_params,
            acc_threshold=acc_threshold,
            gyro_threshold=gyro_threshold,
            vel_threshold=base_vel_threshold * vel_threshold_scale,
        )

        # Camera-motion guard: if recent optical flow indicates motion, suppress ZUPT.
        # This prevents false stationary detection during smooth forward flight.
        flow_guard_px = float(runner.global_config.get("ZUPT_FLOW_GUARD_PX", 1.0))
        flow_guard_window_sec = float(runner.global_config.get("ZUPT_FLOW_GUARD_WINDOW_SEC", 0.30))
        if is_stationary and getattr(runner, "vio_fe", None) is not None:
            try:
                last_cam_t = getattr(runner.vio_fe, "last_frame_time", None)
                last_flow_px = float(getattr(runner.vio_fe, "last_flow_px", 0.0))
                if (
                    last_cam_t is not None
                    and np.isfinite(last_cam_t)
                    and abs(float(rec.t) - float(last_cam_t)) <= flow_guard_window_sec
                    and np.isfinite(last_flow_px)
                    and last_flow_px >= flow_guard_px
                ):
                    is_stationary = False
            except Exception:
                pass

        if zupt_enabled and is_stationary:
            runner.state.zupt_detected += 1
            zupt_adaptive_info: Dict[str, Any] = {}
            if self._decision_is_hold_skip(zupt_decision):
                zupt_adaptive_info.update(
                    {
                        "sensor": "ZUPT",
                        "accepted": False,
                        "attempted": 0,
                        "dof": 3,
                        "nis_norm": np.nan,
                        "chi2": np.nan,
                        "threshold": np.nan,
                        "r_scale_used": float(zupt_r_scale),
                        "reason_code": f"policy_mode_{str(zupt_decision.mode).lower()}",
                    }
                )
                runner.adaptive_service.record_adaptive_measurement(
                    "ZUPT",
                    adaptive_info=zupt_adaptive_info,
                    timestamp=rec.t,
                    policy_scales=zupt_policy_scales,
                    counts_as_aiding=False,
                )
                runner.state.consecutive_stationary = 0
                runner.state.zupt_rejected += 1
            else:
                applied, v_reduction, updated_count = apply_zupt(
                    runner.kf,
                    v_mag=v_mag,
                    consecutive_stationary_count=runner.state.consecutive_stationary,
                    max_v_for_zupt=base_max_v_for_zupt * max_v_scale,
                    save_debug=runner.config.save_debug_data,
                    residual_csv=getattr(runner, "residual_csv", None),
                    timestamp=rec.t,
                    frame=runner.state.imu_propagation_count,
                    r_scale=float(zupt_r_scale),
                    chi2_scale=float(zupt_chi2_scale),
                    soft_fail_enable=bool(zupt_fail_soft_enable),
                    soft_fail_r_cap=float(zupt_soft_r_cap),
                    soft_fail_hard_reject_factor=float(zupt_hard_reject_factor),
                    soft_fail_power=float(zupt_soft_r_power),
                    adaptive_info=zupt_adaptive_info,
                )
                runner.adaptive_service.record_adaptive_measurement(
                    "ZUPT",
                    adaptive_info=zupt_adaptive_info,
                    timestamp=rec.t,
                    policy_scales=zupt_policy_scales,
                    counts_as_aiding=False,
                )

                if applied:
                    runner.state.zupt_applied += 1
                    runner.state.consecutive_stationary = updated_count
                else:
                    runner.state.zupt_rejected += 1
        else:
            runner.state.consecutive_stationary = 0

        # IMU-only stabilization stack: gravity RP, weak yaw aid, bias guard.
        if getattr(runner, "imu_only_mode", False):
            if runner._imu_only_yaw_ref is None:
                runner._imu_only_yaw_ref = float(quaternion_to_yaw(runner.kf.x[6:10, 0].reshape(4,)))
            if runner._imu_only_bg_ref is None:
                runner._imu_only_bg_ref = runner.kf.x[10:13, 0].astype(float).copy()
            if runner._imu_only_ba_ref is None:
                runner._imu_only_ba_ref = runner.kf.x[13:16, 0].astype(float).copy()
            warning_backoff_active = bool(runner._conditioning_warning_sec >= 1.5)
            warning_period_mult = 2 if warning_backoff_active else 1

            grav_decision, grav_policy_scales = self._policy_for_sensor("GRAVITY_RP", float(rec.t))
            grav_period_steps = max(1, int(round(float(grav_decision.extra("period_steps", 1.0))))) * warning_period_mult
            if (
                (not self._decision_is_hold_skip(grav_decision))
                and bool(float(grav_decision.extra("enabled_imu_only", 1.0)) >= 0.5)
                and (runner.state.imu_propagation_count % grav_period_steps == 0)
            ):
                grav_adaptive_info: Dict[str, Any] = {}
                apply_gravity_roll_pitch_update(
                    runner.kf,
                    a_raw=rec.lin.astype(float),
                    w_corr=w_corr,
                    imu_params=imu_params,
                    sigma_rad=np.deg2rad(float(grav_decision.extra("sigma_deg", 7.0))),
                    r_scale=float(grav_decision.r_scale),
                    chi2_scale=float(grav_decision.chi2_scale),
                    acc_norm_tolerance=float(grav_decision.extra("acc_norm_tolerance", 0.25)),
                    max_gyro_rad_s=float(grav_decision.extra("max_gyro_rad_s", 0.40)),
                    save_debug=runner.config.save_debug_data,
                    residual_csv=getattr(runner, "residual_csv", None),
                    timestamp=rec.t,
                    frame=runner.state.imu_propagation_count,
                    adaptive_info=grav_adaptive_info,
                )
                runner.adaptive_service.record_adaptive_measurement(
                    "GRAVITY_RP",
                    adaptive_info=grav_adaptive_info,
                    timestamp=rec.t,
                    policy_scales=grav_policy_scales,
                    counts_as_aiding=False,
                )
            elif self._decision_is_hold_skip(grav_decision):
                runner.adaptive_service.record_adaptive_measurement(
                    "GRAVITY_RP",
                    adaptive_info={
                        "sensor": "GRAVITY_RP",
                        "accepted": False,
                        "attempted": 0,
                        "dof": 2,
                        "nis_norm": np.nan,
                        "chi2": np.nan,
                        "threshold": np.nan,
                        "r_scale_used": float(grav_decision.r_scale),
                        "reason_code": f"policy_mode_{str(grav_decision.mode).lower()}",
                    },
                    timestamp=rec.t,
                    policy_scales=grav_policy_scales,
                    counts_as_aiding=False,
                )

            yaw_decision, yaw_policy_scales = self._policy_for_sensor("YAW_AID", float(rec.t))
            yaw_period_steps = max(1, int(round(float(yaw_decision.extra("period_steps", 8.0))))) * warning_period_mult
            speed_ms_now = float(v_mag)
            if speed_ms_now > float(yaw_decision.extra("high_speed_m_s", 1e9)):
                yaw_period_steps = int(
                    max(1, round(yaw_period_steps * float(yaw_decision.extra("high_speed_period_mult", 1.0))))
                )
            if (
                (not self._decision_is_hold_skip(yaw_decision))
                and bool(float(yaw_decision.extra("enabled_imu_only", 1.0)) >= 0.5)
                and (runner.state.imu_propagation_count % yaw_period_steps == 0)
            ):
                yaw_sigma_deg = float(yaw_decision.extra("sigma_deg", 35.0))
                if speed_ms_now > float(yaw_decision.extra("high_speed_m_s", 1e9)):
                    yaw_sigma_deg *= float(yaw_decision.extra("high_speed_sigma_mult", 1.0))
                yaw_adaptive_info: Dict[str, Any] = {}
                apply_yaw_pseudo_update(
                    runner.kf,
                    a_raw=rec.lin.astype(float),
                    w_corr=w_corr,
                    imu_params=imu_params,
                    yaw_ref_rad=float(runner._imu_only_yaw_ref),
                    velocity_enu=x[3:6].astype(float),
                    sigma_rad=np.deg2rad(yaw_sigma_deg),
                    r_scale=float(yaw_decision.r_scale),
                    chi2_scale=float(yaw_decision.chi2_scale),
                    acc_norm_tolerance=float(yaw_decision.extra("acc_norm_tolerance", 0.25)),
                    max_gyro_rad_s=float(yaw_decision.extra("max_gyro_rad_s", 0.40)),
                    motion_consistency_enable=bool(float(yaw_decision.extra("motion_consistency_enable", 0.0)) >= 0.5),
                    motion_min_speed_m_s=float(yaw_decision.extra("motion_min_speed_m_s", 20.0)),
                    motion_speed_full_m_s=float(yaw_decision.extra("motion_speed_full_m_s", 90.0)),
                    motion_weight_max=float(yaw_decision.extra("motion_weight_max", 0.18)),
                    motion_max_yaw_error_deg=float(yaw_decision.extra("motion_max_yaw_error_deg", 70.0)),
                    soft_fail_enable=bool(float(yaw_decision.extra("fail_soft_enable", 0.0)) >= 0.5),
                    soft_fail_r_cap=float(yaw_decision.extra("soft_r_cap", 12.0)),
                    soft_fail_hard_reject_factor=float(yaw_decision.extra("hard_reject_factor", 3.0)),
                    soft_fail_power=float(yaw_decision.extra("soft_r_power", 1.0)),
                    save_debug=runner.config.save_debug_data,
                    residual_csv=getattr(runner, "residual_csv", None),
                    timestamp=rec.t,
                    frame=runner.state.imu_propagation_count,
                    adaptive_info=yaw_adaptive_info,
                )
                runner.adaptive_service.record_adaptive_measurement(
                    "YAW_AID",
                    adaptive_info=yaw_adaptive_info,
                    timestamp=rec.t,
                    policy_scales=yaw_policy_scales,
                    counts_as_aiding=False,
                )
                yaw_now = float(quaternion_to_yaw(runner.kf.x[6:10, 0].reshape(4,)))
                attempted = int(yaw_adaptive_info.get("attempted", 1))
                if attempted <= 0:
                    yaw_alpha = float(yaw_decision.extra("dynamic_ref_alpha", 0.05))
                else:
                    yaw_alpha = float(yaw_decision.extra("ref_alpha", 0.005))
                yaw_alpha = max(0.0, min(1.0, yaw_alpha))
                runner._imu_only_yaw_ref = float(
                    np.arctan2(
                        np.sin((1.0 - yaw_alpha) * float(runner._imu_only_yaw_ref) + yaw_alpha * yaw_now),
                        np.cos((1.0 - yaw_alpha) * float(runner._imu_only_yaw_ref) + yaw_alpha * yaw_now),
                    )
                )
            elif self._decision_is_hold_skip(yaw_decision):
                runner.adaptive_service.record_adaptive_measurement(
                    "YAW_AID",
                    adaptive_info={
                        "sensor": "YAW_AID",
                        "accepted": False,
                        "attempted": 0,
                        "dof": 1,
                        "nis_norm": np.nan,
                        "chi2": np.nan,
                        "threshold": np.nan,
                        "r_scale_used": float(yaw_decision.r_scale),
                        "reason_code": f"policy_mode_{str(yaw_decision.mode).lower()}",
                    },
                    timestamp=rec.t,
                    policy_scales=yaw_policy_scales,
                    counts_as_aiding=False,
                )

            bias_decision, bias_policy_scales = self._policy_for_sensor("BIAS_GUARD", float(rec.t))
            bias_enabled = bool(float(bias_decision.extra("enabled_imu_only", 1.0)) >= 0.5)
            period_steps = max(1, int(round(float(bias_decision.extra("period_steps", 8.0)))))
            aiding_level = str(bias_decision.extra_str("aiding_level", "FULL")).upper()
            health_state = str(bias_decision.extra_str("health_state", "HEALTHY")).upper()
            phase_now = int(runner.state.current_phase)
            phase_period_mult = 2 if phase_now >= 1 else 1
            health_period_mult = 2 if health_state in ("WARNING", "DEGRADED") else 1
            period_steps = period_steps * phase_period_mult * health_period_mult * warning_period_mult
            if speed_ms_now > float(bias_decision.extra("high_speed_m_s", 1e9)):
                period_steps = int(
                    max(1, round(period_steps * float(bias_decision.extra("high_speed_period_mult", 1.0))))
                )
            allow_level = (
                (aiding_level == "NONE" and float(bias_decision.extra("enable_when_no_aiding", 1.0)) >= 0.5)
                or (aiding_level == "PARTIAL" and float(bias_decision.extra("enable_when_partial_aiding", 0.0)) >= 0.5)
                or (aiding_level == "FULL" and float(bias_decision.extra("enable_when_full_aiding", 0.0)) >= 0.5)
            )
            g_norm = float(imu_params.get("g_norm", 9.80665))
            accel_includes_gravity = bool(imu_params.get("accel_includes_gravity", True))
            expected_acc_mag = g_norm if accel_includes_gravity else 0.0
            acc_dev = abs(float(np.linalg.norm(rec.lin.astype(float))) - expected_acc_mag)
            gyro_mag = float(np.linalg.norm(w_corr))
            low_dynamic_for_bias = (
                acc_dev <= float(bias_decision.extra("acc_norm_tolerance", 0.15))
                and gyro_mag <= float(bias_decision.extra("max_gyro_rad_s", 0.25))
            )
            if (
                (not self._decision_is_hold_skip(bias_decision))
                and bias_enabled
                and allow_level
                and low_dynamic_for_bias
                and (runner.state.imu_propagation_count % period_steps == 0)
            ):
                bias_adaptive_info: Dict[str, Any] = {}
                apply_bias_observability_guard(
                    runner.kf,
                    bg_ref=runner._imu_only_bg_ref,
                    ba_ref=runner._imu_only_ba_ref,
                    sigma_bg_rad_s=float(bias_decision.extra("sigma_bg_rad_s", np.deg2rad(0.30))),
                    sigma_ba_m_s2=float(bias_decision.extra("sigma_ba_m_s2", 0.15)),
                    r_scale=float(bias_decision.r_scale),
                    chi2_scale=float(bias_decision.chi2_scale),
                    soft_fail_enable=bool(float(bias_decision.extra("fail_soft_enable", 0.0)) >= 0.5),
                    soft_fail_r_cap=float(bias_decision.extra("soft_r_cap", 8.0)),
                    soft_fail_hard_reject_factor=float(bias_decision.extra("hard_reject_factor", 4.0)),
                    soft_fail_power=float(bias_decision.extra("soft_r_power", 1.0)),
                    max_bg_norm_rad_s=float(bias_decision.extra("max_bg_norm_rad_s", 0.20)),
                    max_ba_norm_m_s2=float(bias_decision.extra("max_ba_norm_m_s2", 2.5)),
                    save_debug=runner.config.save_debug_data,
                    residual_csv=getattr(runner, "residual_csv", None),
                    timestamp=rec.t,
                    frame=runner.state.imu_propagation_count,
                    adaptive_info=bias_adaptive_info,
                )
                runner.adaptive_service.record_adaptive_measurement(
                    "BIAS_GUARD",
                    adaptive_info=bias_adaptive_info,
                    timestamp=rec.t,
                    policy_scales=bias_policy_scales,
                    counts_as_aiding=False,
                )
                if bool(bias_adaptive_info.get("accepted", False)):
                    bg_now = runner.kf.x[10:13, 0].astype(float)
                    ba_now = runner.kf.x[13:16, 0].astype(float)
                    runner._imu_only_bg_ref = 0.99 * runner._imu_only_bg_ref + 0.01 * bg_now
                    runner._imu_only_ba_ref = 0.99 * runner._imu_only_ba_ref + 0.01 * ba_now
            elif self._decision_is_hold_skip(bias_decision):
                runner.adaptive_service.record_adaptive_measurement(
                    "BIAS_GUARD",
                    adaptive_info={
                        "sensor": "BIAS_GUARD",
                        "accepted": False,
                        "attempted": 0,
                        "dof": 6,
                        "nis_norm": np.nan,
                        "chi2": np.nan,
                        "threshold": np.nan,
                        "r_scale_used": float(bias_decision.r_scale),
                        "reason_code": f"policy_mode_{str(bias_decision.mode).lower()}",
                    },
                    timestamp=rec.t,
                    policy_scales=bias_policy_scales,
                    counts_as_aiding=False,
                )

        # Save priors
        runner.kf.x_prior = runner.kf.x.copy()
        runner.kf.P_prior = runner.kf.P.copy()
