"""Magnetometer update service for VIORunner."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict

import numpy as np

from ..magnetometer import calibrate_magnetometer, apply_mag_filter
from ..measurement_updates import apply_magnetometer_update
from ..output_utils import log_mag_quality


class MagnetometerService:
    """Encapsulates magnetometer update logic (imu-driven + event-driven)."""

    def __init__(self, runner: Any):
        self.runner = runner
        self._mag_norm_hist = deque(maxlen=200)
        self._mag_norm_ewma: float = float("nan")
        self._last_mag_yaw: float | None = None
        self._last_mag_t: float | None = None

    @staticmethod
    def _wrap_angle(rad: float) -> float:
        return float(np.arctan2(np.sin(float(rad)), np.cos(float(rad))))

    def _score_linear(self, value: float, good: float, bad: float) -> float:
        if not np.isfinite(value):
            return float("nan")
        if value <= good:
            return 1.0
        if value >= bad:
            return 0.0
        return float(1.0 - (value - good) / max(1e-9, bad - good))

    def _evaluate_accuracy_policy(self,
                                  yaw_mag_filtered: float,
                                  mag_norm: float,
                                  timestamp: float) -> Dict[str, Any]:
        """
        Quality-aware magnetometer policy for accuracy-first mode.

        Returns dict:
          decision: good | mid_inflate | bad_skip | bad_inflate | disabled
          r_scale: multiplicative R factor
          quality_score: [0,1]
          norm_ewma, norm_dev, gyro_delta_deg, vision_delta_deg
        """
        cfg = self.runner.global_config
        enabled = bool(cfg.get("MAG_ACCURACY_ENABLED", True))
        if not enabled:
            return {
                "decision": "disabled",
                "r_scale": 1.0,
                "quality_score": 1.0,
                "norm_ewma": float(mag_norm) if np.isfinite(mag_norm) else np.nan,
                "norm_dev": 0.0,
                "gyro_delta_deg": np.nan,
                "vision_delta_deg": np.nan,
                "reason": "policy_disabled",
            }

        window = max(20, int(cfg.get("MAG_ACCURACY_NORM_WINDOW", 200)))
        if self._mag_norm_hist.maxlen != window:
            self._mag_norm_hist = deque(self._mag_norm_hist, maxlen=window)
        if np.isfinite(mag_norm):
            self._mag_norm_hist.append(float(mag_norm))

        alpha = float(np.clip(2.0 / float(window + 1), 0.01, 0.5))
        if np.isfinite(self._mag_norm_ewma):
            self._mag_norm_ewma = (1.0 - alpha) * self._mag_norm_ewma + alpha * float(mag_norm)
        else:
            self._mag_norm_ewma = float(mag_norm)

        norm_ewma = float(self._mag_norm_ewma)
        norm_dev = abs(float(mag_norm) - norm_ewma) / max(1e-6, abs(norm_ewma))
        norm_score = self._score_linear(
            norm_dev,
            float(cfg.get("MAG_ACCURACY_NORM_DEV_GOOD", 0.12)),
            float(cfg.get("MAG_ACCURACY_NORM_DEV_BAD", 0.30)),
        )

        gyro_delta_deg = np.nan
        if self._last_mag_yaw is not None and self._last_mag_t is not None:
            dt = float(timestamp) - float(self._last_mag_t)
            if dt > 1e-4:
                yaw_pred_delta = float(self.runner.last_gyro_z) * dt
                yaw_meas_delta = self._wrap_angle(float(yaw_mag_filtered) - float(self._last_mag_yaw))
                gyro_delta_deg = abs(np.degrees(self._wrap_angle(yaw_meas_delta - yaw_pred_delta)))
        gyro_score = self._score_linear(
            gyro_delta_deg,
            float(cfg.get("MAG_ACCURACY_GYRO_DELTA_SOFT_DEG", 20.0)),
            float(cfg.get("MAG_ACCURACY_GYRO_DELTA_HARD_DEG", 65.0)),
        )

        vision_delta_deg = np.nan
        vis_yaw = getattr(self.runner, "_vision_yaw_ref", None)
        vis_t = getattr(self.runner, "_vision_yaw_last_t", None)
        if vis_yaw is not None and vis_t is not None:
            vis_age = float(timestamp) - float(vis_t)
            vis_max_age = float(self.runner.global_config.get("MAG_VISION_HEADING_MAX_AGE_SEC", 1.0))
            if 0.0 <= vis_age <= vis_max_age:
                vision_delta_deg = abs(
                    np.degrees(self._wrap_angle(float(yaw_mag_filtered) - float(vis_yaw)))
                )
        vision_score = self._score_linear(
            vision_delta_deg,
            float(cfg.get("MAG_ACCURACY_VISION_DELTA_SOFT_DEG", 30.0)),
            float(cfg.get("MAG_ACCURACY_VISION_DELTA_HARD_DEG", 90.0)),
        )

        # Weighted fusion of available quality terms.
        vision_w = float(np.clip(cfg.get("MAG_ACCURACY_VISION_WEIGHT", 0.25), 0.0, 0.5))
        gyro_w = 0.35
        norm_w = max(0.0, 1.0 - gyro_w - vision_w)
        score_terms = []
        weight_terms = []
        if np.isfinite(norm_score):
            score_terms.append(float(norm_score))
            weight_terms.append(norm_w)
        if np.isfinite(gyro_score):
            score_terms.append(float(gyro_score))
            weight_terms.append(gyro_w)
        if np.isfinite(vision_score):
            score_terms.append(float(vision_score))
            weight_terms.append(vision_w)
        if len(score_terms) == 0 or np.sum(weight_terms) <= 1e-9:
            quality_score = 0.5
        else:
            quality_score = float(np.sum(np.array(score_terms) * np.array(weight_terms)) / np.sum(weight_terms))

        good_min = float(cfg.get("MAG_ACCURACY_GOOD_MIN_SCORE", 0.72))
        mid_min = float(cfg.get("MAG_ACCURACY_MID_MIN_SCORE", 0.45))
        r_mid = float(max(1.0, cfg.get("MAG_ACCURACY_R_INFLATE_MID", 1.8)))
        r_bad = float(max(r_mid, cfg.get("MAG_ACCURACY_R_INFLATE_BAD", 3.0)))
        skip_on_bad = bool(cfg.get("MAG_ACCURACY_SKIP_ON_BAD", True))

        if quality_score >= good_min:
            decision = "good"
            r_scale = 1.0
        elif quality_score >= mid_min:
            decision = "mid_inflate"
            r_scale = r_mid
        else:
            if skip_on_bad:
                decision = "bad_skip"
            else:
                decision = "bad_inflate"
            r_scale = r_bad

        self._last_mag_yaw = float(yaw_mag_filtered)
        self._last_mag_t = float(timestamp)
        return {
            "decision": decision,
            "r_scale": float(r_scale),
            "quality_score": float(quality_score),
            "norm_ewma": float(norm_ewma),
            "norm_dev": float(norm_dev),
            "gyro_delta_deg": float(gyro_delta_deg) if np.isfinite(gyro_delta_deg) else np.nan,
            "vision_delta_deg": float(vision_delta_deg) if np.isfinite(vision_delta_deg) else np.nan,
            "reason": f"score={quality_score:.3f}",
        }

    def _apply_visual_heading_consistency(self,
                                          yaw_mag_filtered: float,
                                          sigma_mag_scaled: float,
                                          timestamp: float) -> tuple[bool, float, str]:
        """
        Fail-soft MAG heading consistency with VO-derived heading reference.

        Returns:
            (skip_update, sigma_scaled, reason_code)
        """
        runner = self.runner
        cfg = runner.global_config
        if not bool(cfg.get("MAG_VISION_HEADING_CONSISTENCY_ENABLE", True)):
            return False, float(sigma_mag_scaled), ""
        if bool(getattr(runner, "imu_only_mode", False)):
            return False, float(sigma_mag_scaled), ""

        vis_yaw = getattr(runner, "_vision_yaw_ref", None)
        vis_t = getattr(runner, "_vision_yaw_last_t", None)
        vis_q = float(getattr(runner, "_vision_heading_quality", 0.0))
        if vis_yaw is None or vis_t is None:
            return False, float(sigma_mag_scaled), ""

        max_age = float(cfg.get("MAG_VISION_HEADING_MAX_AGE_SEC", 1.0))
        age = float(timestamp) - float(vis_t)
        if age < 0.0 or age > max_age:
            return False, float(sigma_mag_scaled), ""

        min_quality = float(cfg.get("MAG_VISION_HEADING_MIN_QUALITY", 0.15))
        if vis_q < min_quality:
            return False, float(sigma_mag_scaled), ""

        yaw_diff = self._wrap_angle(float(yaw_mag_filtered) - float(vis_yaw))
        yaw_diff_deg = abs(float(np.degrees(yaw_diff)))

        soft_deg = float(cfg.get("MAG_VISION_HEADING_SOFT_DEG", 30.0))
        hard_deg = max(soft_deg + 1.0, float(cfg.get("MAG_VISION_HEADING_HARD_DEG", 85.0)))
        max_r = max(1.0, float(cfg.get("MAG_VISION_HEADING_R_INFLATE_MAX", 6.0)))
        strong_quality = float(cfg.get("MAG_VISION_HEADING_STRONG_QUALITY", 0.60))
        strong_quality = max(min_quality, min(1.0, strong_quality))

        runner.output_reporting.log_convention_check(
            t=float(timestamp),
            sensor="MAG",
            check="vision_heading_delta_deg",
            value=float(yaw_diff_deg),
            threshold=float(hard_deg),
            status="PASS" if yaw_diff_deg <= hard_deg else "WARN",
            note=f"vision_quality={vis_q:.3f}",
        )

        if yaw_diff_deg >= hard_deg:
            # Fail-soft: only hard reject when vision heading confidence is truly strong.
            # Otherwise keep MAG update alive with aggressive R inflation.
            if vis_q >= strong_quality:
                return True, float(sigma_mag_scaled), "skip_vision_heading_mismatch"
            sigma_scaled = float(sigma_mag_scaled) * float(max_r)
            return False, sigma_scaled, "vision_heading_hard_soften"
        if yaw_diff_deg <= soft_deg:
            return False, float(sigma_mag_scaled), ""

        frac = (yaw_diff_deg - soft_deg) / max(1e-6, hard_deg - soft_deg)
        r_inflate = 1.0 + frac * (max_r - 1.0)
        sigma_scaled = float(sigma_mag_scaled) * float(r_inflate)
        return False, sigma_scaled, "vision_heading_soft_inflate"

    def process_magnetometer(self, t: float):
        """
        Process magnetometer measurements up to current time.

        Args:
            t: Current timestamp
        """
        sigma_mag = self.runner.global_config.get("SIGMA_MAG_YAW", 0.15)
        declination = self.runner.global_config.get("MAG_DECLINATION", 0.0)
        use_raw = self.runner.global_config.get("MAG_USE_RAW_HEADING", True)
        rate_limit = self.runner.global_config.get("MAG_UPDATE_RATE_LIMIT", 1)

        while (
            self.runner.state.mag_idx < len(self.runner.mag_list)
            and self.runner.mag_list[self.runner.state.mag_idx].t <= t
        ):
            mag_rec = self.runner.mag_list[self.runner.state.mag_idx]
            self.runner.state.mag_idx += 1

            # Rate limiting
            if (self.runner.state.mag_idx - 1) % rate_limit != 0:
                continue

            sync_threshold = float(self.runner.global_config.get("MAG_TIME_SYNC_THRESHOLD_SEC", 0.05))
            dt_sync = abs(float(t) - float(mag_rec.t))
            sync_status = "PASS" if dt_sync <= sync_threshold else "FAIL"
            self.runner.output_reporting.log_convention_check(
                t=float(mag_rec.t),
                sensor="MAG",
                check="time_base_abs_dt",
                value=float(dt_sync),
                threshold=float(sync_threshold),
                status=sync_status,
                note=f"t_filter={float(t):.6f}",
            )
            if dt_sync > sync_threshold:
                mag_adaptive_info = {
                    "sensor": "MAG",
                    "accepted": False,
                    "attempted": 1,
                    "dof": 1,
                    "nis_norm": np.nan,
                    "chi2": np.nan,
                    "threshold": np.nan,
                    "r_scale_used": 1.0,
                    "reason_code": "skip_time_mismatch",
                }
                self.runner.adaptive_service.record_adaptive_measurement(
                    "MAG",
                    adaptive_info=mag_adaptive_info,
                    timestamp=float(mag_rec.t),
                )
                self.runner.state.mag_rejects += 1
                continue

            # v3.9.7: Use EKF estimated mag_bias instead of static config hard_iron
            # This enables online hard iron estimation for time-varying interference
            if self.runner.config.use_mag_estimated_bias:
                # Use EKF state mag_bias (indices 16:19) as hard iron
                hard_iron = self.runner.kf.x[16:19, 0].flatten()
            else:
                # Fallback to static config hard iron
                hard_iron = self.runner.global_config.get("MAG_HARD_IRON_OFFSET", None)
            soft_iron = self.runner.global_config.get("MAG_SOFT_IRON_MATRIX", None)
            mag_cal = calibrate_magnetometer(mag_rec.mag, hard_iron=hard_iron, soft_iron=soft_iron)
            mag_norm = float(np.linalg.norm(mag_cal))

            # Compute raw yaw from calibrated magnetometer
            from ..magnetometer import compute_yaw_from_mag

            q_current = self.runner.kf.x[6:10, 0].flatten()
            yaw_mag_raw, quality = compute_yaw_from_mag(
                mag_body=mag_cal,
                q_wxyz=q_current,
                mag_declination=declination,
                use_raw_heading=use_raw,
            )

            # Apply magnetometer filter (EMA smoothing + gyro consistency check)
            # v3.4.0: Use phase-based convergence (state-based, not time-based)
            in_convergence = self.runner.state.current_phase < 2  # SPINUP or EARLY phase
            yaw_mag_filtered, r_scale, filter_info = apply_mag_filter(
                yaw_mag=yaw_mag_raw,
                yaw_t=mag_rec.t,
                gyro_z=self.runner.last_gyro_z,
                dt_imu=self.runner.last_imu_dt,
                in_convergence=in_convergence,
                mag_max_yaw_rate=self.runner.global_config.get("MAG_MAX_YAW_RATE_DEG", 30.0) * np.pi / 180.0,
                mag_gyro_threshold=self.runner.global_config.get("MAG_GYRO_THRESHOLD_DEG", 10.0) * np.pi / 180.0,
                mag_ema_alpha=self.runner.global_config.get("MAG_EMA_ALPHA", 0.3),
                mag_consistency_r_inflate=self.runner.global_config.get("MAG_R_INFLATE", 5.0),
            )

            # Scale measurement noise based on filter confidence
            sigma_mag_scaled = sigma_mag * r_scale
            mag_policy_scales, mag_apply_scales = self.runner.adaptive_service.get_sensor_adaptive_scales("MAG")
            mag_adaptive_info: Dict[str, Any] = {}
            mag_quality = self._evaluate_accuracy_policy(
                yaw_mag_filtered=yaw_mag_filtered,
                mag_norm=mag_norm,
                timestamp=float(mag_rec.t),
            )
            sigma_mag_scaled *= float(mag_quality.get("r_scale", 1.0))
            skip_mag, sigma_mag_scaled, consistency_reason = self._apply_visual_heading_consistency(
                yaw_mag_filtered=yaw_mag_filtered,
                sigma_mag_scaled=sigma_mag_scaled,
                timestamp=float(mag_rec.t),
            )
            decision = str(mag_quality.get("decision", "good"))
            if decision == "bad_skip":
                skip_mag = True
                if not consistency_reason:
                    consistency_reason = "skip_quality_bad"
            log_mag_quality(
                getattr(self.runner, "mag_quality_csv", None),
                t=float(mag_rec.t),
                raw_norm=float(mag_norm),
                norm_ewma=float(mag_quality.get("norm_ewma", np.nan)),
                norm_dev=float(mag_quality.get("norm_dev", np.nan)),
                gyro_delta_deg=float(mag_quality.get("gyro_delta_deg", np.nan)),
                vision_delta_deg=float(mag_quality.get("vision_delta_deg", np.nan)),
                quality_score=float(mag_quality.get("quality_score", np.nan)),
                decision=decision if consistency_reason == "" else f"{decision}+{consistency_reason}",
                r_scale=float(max(1e-6, sigma_mag_scaled / max(1e-6, sigma_mag))),
                reason=str(mag_quality.get("reason", "")),
            )
            if skip_mag:
                mag_adaptive_info = {
                    "sensor": "MAG",
                    "accepted": False,
                    "attempted": 1,
                    "dof": 1,
                    "nis_norm": np.nan,
                    "chi2": np.nan,
                    "threshold": np.nan,
                    "r_scale_used": float(max(1e-6, sigma_mag_scaled / max(1e-6, sigma_mag))),
                    "reason_code": consistency_reason,
                }
                self.runner.adaptive_service.record_adaptive_measurement(
                    "MAG",
                    adaptive_info=mag_adaptive_info,
                    timestamp=float(mag_rec.t),
                    policy_scales=mag_policy_scales,
                )
                self.runner.state.mag_rejects += 1
                continue

            # Use filtered yaw instead of raw calibrated mag
            has_ppk = self.runner.ppk_state is not None

            # Get residual_csv path if debug data is enabled
            residual_path = (
                self.runner.residual_csv
                if self.runner.config.save_debug_data and hasattr(self.runner, "residual_csv")
                else None
            )

            # Apply magnetometer update using FILTERED yaw
            # CRITICAL: Pass filtered yaw directly instead of raw mag
            applied, reason = apply_magnetometer_update(
                self.runner.kf,
                mag_calibrated=mag_cal,  # Still needed for compute_yaw_from_mag inside
                mag_declination=declination,
                use_raw_heading=use_raw,
                sigma_mag_yaw=sigma_mag_scaled,  # Use scaled sigma
                current_phase=self.runner.state.current_phase,  # v3.4.0: state-based phase
                in_convergence=in_convergence,
                has_ppk_yaw=has_ppk,
                timestamp=mag_rec.t,
                residual_csv=residual_path,
                frame=self.runner.state.vio_frame,
                yaw_override=yaw_mag_filtered,  # NEW: pass filtered yaw directly
                filter_info=filter_info,  # v2.9.2: track filter rejection reasons
                use_estimated_bias=self.runner.config.use_mag_estimated_bias,  # v3.9.8: freeze states if disabled
                r_scale_extra=float(mag_apply_scales.get("r_scale", 1.0)),
                adaptive_info=mag_adaptive_info,
            )
            if consistency_reason:
                mag_adaptive_info["reason_code"] = consistency_reason
            self.runner.adaptive_service.record_adaptive_measurement(
                "MAG",
                adaptive_info=mag_adaptive_info,
                timestamp=mag_rec.t,
                policy_scales=mag_policy_scales,
            )

            if applied:
                self.runner.state.mag_updates += 1
            else:
                self.runner.state.mag_rejects += 1

    def process_single_mag(self, mag_rec, t: float):
        """
        Process a single magnetometer measurement (for event-driven mode).

        v3.9.4 CRITICAL: Magnetometer time synchronization via time_ref

        CONDITION 1: Filter must use SAME TIME BASE for all sensors
        - IMU: time_ref (hardware monotonic clock)
        - Camera: time_ref via timeref.csv interpolation
        - Magnetometer: time_ref via interpolation in load_mag_csv()

        CONDITION 2: mag_rec.t must represent TRUE measurement time
        - After interpolation, mag_rec.t is hardware time (consistent with IMU)
        - State is already at time t (propagated by IMU loop)
        - If |t - mag_rec.t| > threshold, skip measurement (timing issue)

        NOTE: Full state propagation to mag_rec.t requires storing IMU buffer.
        Current implementation: Use state at closest IMU time if within threshold.

        Args:
            mag_rec: Magnetometer record with hardware-synchronized timestamp
            t: Current IMU timestamp (after propagation)
        """
        # Check time synchronization (should be very close if time_ref works)
        if abs(t - mag_rec.t) > 0.05:  # 50ms threshold
            print(f"[Mag] WARNING: Large time difference {t - mag_rec.t:.3f}s - may indicate clock sync issue")

        sigma_mag = self.runner.global_config.get("SIGMA_MAG_YAW", 0.15)
        declination = self.runner.global_config.get("MAG_DECLINATION", 0.0)
        use_raw = self.runner.global_config.get("MAG_USE_RAW_HEADING", True)

        # v3.9.7: Use EKF estimated mag_bias instead of static config hard_iron
        if self.runner.config.use_mag_estimated_bias:
            hard_iron = self.runner.kf.x[16:19, 0].flatten()
        else:
            hard_iron = self.runner.global_config.get("MAG_HARD_IRON_OFFSET", None)
        soft_iron = self.runner.global_config.get("MAG_SOFT_IRON_MATRIX", None)
        mag_cal = calibrate_magnetometer(mag_rec.mag, hard_iron=hard_iron, soft_iron=soft_iron)

        # Compute raw yaw from calibrated magnetometer
        from ..magnetometer import compute_yaw_from_mag

        q_current = self.runner.kf.x[6:10, 0].flatten()
        yaw_mag_raw, quality = compute_yaw_from_mag(
            mag_body=mag_cal,
            q_wxyz=q_current,
            mag_declination=declination,
            use_raw_heading=use_raw,
        )

        # Apply magnetometer filter (EMA smoothing + gyro consistency check)
        in_convergence = self.runner.state.current_phase < 2  # SPINUP or EARLY phase
        yaw_mag_filtered, r_scale, filter_info = apply_mag_filter(
            yaw_mag=yaw_mag_raw,
            yaw_t=mag_rec.t,
            gyro_z=self.runner.last_gyro_z,
            dt_imu=self.runner.last_imu_dt,
            in_convergence=in_convergence,
            mag_max_yaw_rate=self.runner.global_config.get("MAG_MAX_YAW_RATE_DEG", 30.0) * np.pi / 180.0,
            mag_gyro_threshold=self.runner.global_config.get("MAG_GYRO_THRESHOLD_DEG", 10.0) * np.pi / 180.0,
            mag_ema_alpha=self.runner.global_config.get("MAG_EMA_ALPHA", 0.3),
            mag_consistency_r_inflate=self.runner.global_config.get("MAG_R_INFLATE", 5.0),
        )

        # Scale measurement noise based on filter confidence
        sigma_mag_scaled = sigma_mag * r_scale
        mag_quality = self._evaluate_accuracy_policy(
            yaw_mag_filtered=yaw_mag_filtered,
            mag_norm=float(np.linalg.norm(mag_cal)),
            timestamp=float(mag_rec.t),
        )
        sigma_mag_scaled *= float(mag_quality.get("r_scale", 1.0))
        skip_mag, sigma_mag_scaled, _ = self._apply_visual_heading_consistency(
            yaw_mag_filtered=yaw_mag_filtered,
            sigma_mag_scaled=sigma_mag_scaled,
            timestamp=float(mag_rec.t),
        )
        if str(mag_quality.get("decision", "")) == "bad_skip":
            skip_mag = True
        log_mag_quality(
            getattr(self.runner, "mag_quality_csv", None),
            t=float(mag_rec.t),
            raw_norm=float(np.linalg.norm(mag_cal)),
            norm_ewma=float(mag_quality.get("norm_ewma", np.nan)),
            norm_dev=float(mag_quality.get("norm_dev", np.nan)),
            gyro_delta_deg=float(mag_quality.get("gyro_delta_deg", np.nan)),
            vision_delta_deg=float(mag_quality.get("vision_delta_deg", np.nan)),
            quality_score=float(mag_quality.get("quality_score", np.nan)),
            decision=str(mag_quality.get("decision", "good")),
            r_scale=float(max(1e-6, sigma_mag_scaled / max(1e-6, sigma_mag))),
            reason=str(mag_quality.get("reason", "")),
        )
        if skip_mag:
            self.runner.state.mag_rejects += 1
            return

        # Use filtered yaw instead of raw calibrated mag
        has_ppk = self.runner.ppk_state is not None

        # Get residual_csv path if debug data is enabled
        residual_path = (
            self.runner.residual_csv
            if self.runner.config.save_debug_data and hasattr(self.runner, "residual_csv")
            else None
        )

        # Apply magnetometer update using FILTERED yaw
        applied, reason = apply_magnetometer_update(
            self.runner.kf,
            mag_calibrated=mag_cal,
            mag_declination=declination,
            use_raw_heading=use_raw,
            sigma_mag_yaw=sigma_mag_scaled,
            current_phase=self.runner.state.current_phase,
            in_convergence=in_convergence,
            has_ppk_yaw=has_ppk,
            timestamp=mag_rec.t,
            residual_csv=residual_path,
            frame=self.runner.state.vio_frame,
            yaw_override=yaw_mag_filtered,
            filter_info=filter_info,
            use_estimated_bias=self.runner.config.use_mag_estimated_bias,  # v3.9.8: freeze states if disabled
        )

        if applied:
            self.runner.state.mag_updates += 1
        else:
            self.runner.state.mag_rejects += 1
