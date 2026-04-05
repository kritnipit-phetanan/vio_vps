"""DEM/height update service for VIORunner."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..measurement_updates import apply_altitude_anchor_update, apply_dem_height_update


class DEMService:
    """Encapsulates DEM/barometer height update logic."""

    def __init__(self, runner: Any):
        self.runner = runner

    def process_dem_height(self, t: float):
        """
        Process DEM height update.

        Args:
            t: Current timestamp
        """
        sigma_height = self.runner.global_config.get("SIGMA_AGL_Z", 2.5)
        alt_anchor_enabled = bool(self.runner.global_config.get("ALT_ANCHOR_ENABLE", True))
        alt_anchor_use_direct_msl = bool(self.runner.global_config.get("ALT_ANCHOR_USE_DIRECT_MSL", True))
        alt_sigma = float(self.runner.global_config.get("ALT_ANCHOR_SIGMA_Z_M", sigma_height))
        alt_base_threshold = float(self.runner.global_config.get("ALT_ANCHOR_BASE_THRESHOLD", 12.0))
        alt_no_vision_threshold = float(
            self.runner.global_config.get("ALT_ANCHOR_NO_VISION_THRESHOLD", 60.0)
        )
        alt_soft_gate_enable = bool(
            self.runner.global_config.get("ALT_ANCHOR_SOFT_GATE_ENABLE", True)
        )
        alt_soft_gate_max_r_inflation = float(
            self.runner.global_config.get("ALT_ANCHOR_SOFT_GATE_MAX_R_INFLATION", 16.0)
        )
        alt_bias_estimation_enable = bool(
            self.runner.global_config.get("ALT_ANCHOR_BIAS_ESTIMATION_ENABLE", True)
        )
        alt_bias_alpha = float(self.runner.global_config.get("ALT_ANCHOR_BIAS_ALPHA", 1e-4))
        alt_bias_max_abs_m = float(self.runner.global_config.get("ALT_ANCHOR_BIAS_MAX_ABS_M", 20.0))
        alt_bias_update_gate_m = float(
            self.runner.global_config.get("ALT_ANCHOR_BIAS_UPDATE_GATE_M", 12.0)
        )
        alt_bias_freeze_no_vision = bool(
            self.runner.global_config.get("ALT_ANCHOR_BIAS_FREEZE_NO_VISION", True)
        )
        alt_max_abs_innovation_m = float(
            self.runner.global_config.get("ALT_ANCHOR_MAX_ABS_INNOVATION_M", 25.0)
        )

        # Prefer direct MSL altitude anchor when available.
        msl_direct = None
        if alt_anchor_enabled and alt_anchor_use_direct_msl and self.runner.msl_interpolator:
            raw_msl = self.runner.msl_interpolator.get_msl(t)
            if raw_msl is not None:
                msl_direct = raw_msl - getattr(self.runner, "msl_offset", 0.0)

        # Get current position
        lat_now, lon_now = self.runner.proj_cache.xy_to_latlon(
            self.runner.kf.x[0, 0], self.runner.kf.x[1, 0], self.runner.lat0, self.runner.lon0
        )

        # Sample DEM
        dem_now = self.runner.dem.sample_m(lat_now, lon_now) if self.runner.dem.ds else None

        if msl_direct is None and (dem_now is None or np.isnan(dem_now)):
            return

        # Get residual_csv path if debug data is enabled
        residual_path = (
            self.runner.residual_csv
            if self.runner.config.save_debug_data and hasattr(self.runner, "residual_csv")
            else None
        )
        time_since_correction = t - getattr(self.runner.kf, "last_absolute_correction_time", t)
        speed = float(np.linalg.norm(self.runner.kf.x[3:6, 0]))
        no_vision_corrections = (len(self.runner.imgs) == 0 and len(self.runner.vps_list) == 0)

        if msl_direct is not None:
            alt_decision = None
            alt_policy_scales = {
                "r_scale": 1.0,
                "chi2_scale": 1.0,
                "threshold_scale": 1.0,
                "reproj_scale": 1.0,
            }
            if getattr(self.runner, "policy_runtime_service", None) is not None:
                try:
                    alt_decision = self.runner.policy_runtime_service.get_sensor_decision("ALT", float(t))
                    alt_policy_scales = {
                        "r_scale": float(alt_decision.r_scale),
                        "chi2_scale": float(alt_decision.chi2_scale),
                        "threshold_scale": float(alt_decision.threshold_scale),
                        "reproj_scale": float(alt_decision.reproj_scale),
                    }
                except Exception:
                    alt_decision = None
            if alt_decision is not None and str(alt_decision.mode).upper() in ("HOLD", "SKIP"):
                alt_adaptive_info: Dict[str, Any] = {
                    "sensor": "ALT",
                    "accepted": False,
                    "attempted": 0,
                    "dof": 1,
                    "nis_norm": np.nan,
                    "chi2": np.nan,
                    "threshold": np.nan,
                    "r_scale_used": float(alt_policy_scales.get("r_scale", 1.0)),
                    "reason_code": f"policy_mode_{str(alt_decision.mode).lower()}",
                }
                self.runner.adaptive_service.record_adaptive_measurement(
                    "ALT",
                    adaptive_info=alt_adaptive_info,
                    timestamp=t,
                    policy_scales=alt_policy_scales,
                )
                return

            alt_adaptive_info: Dict[str, Any] = {}
            apply_altitude_anchor_update(
                self.runner.kf,
                altitude_measurement=float(msl_direct),
                sigma_altitude=alt_sigma,
                no_vision_corrections=no_vision_corrections,
                timestamp=t,
                residual_csv=residual_path,
                frame=self.runner.state.vio_frame,
                threshold_scale=float(alt_policy_scales.get("threshold_scale", 1.0)),
                r_scale_extra=float(alt_policy_scales.get("r_scale", 1.0)),
                soft_gate_enable=alt_soft_gate_enable,
                soft_gate_max_r_inflation=alt_soft_gate_max_r_inflation,
                base_threshold=alt_base_threshold,
                no_vision_threshold=alt_no_vision_threshold,
                bias_estimation_enable=alt_bias_estimation_enable,
                bias_alpha=alt_bias_alpha,
                bias_max_abs_m=alt_bias_max_abs_m,
                bias_update_gate_m=alt_bias_update_gate_m,
                bias_freeze_no_vision=alt_bias_freeze_no_vision,
                max_abs_innovation_m=alt_max_abs_innovation_m,
                adaptive_info=alt_adaptive_info,
            )
            self.runner.adaptive_service.record_adaptive_measurement(
                "ALT",
                adaptive_info=alt_adaptive_info,
                timestamp=t,
                policy_scales=alt_policy_scales,
            )
            return

        # Fallback: Compute height measurement from DEM + estimated AGL
        # CRITICAL FIX v3.9.2: DO NOT use GPS MSL - that causes innovation=0!
        # DEM update must constrain drift by comparing: (DEM + estimated_AGL) vs state
        # State is MSL, measurement = DEM + estimated_AGL
        current_msl = self.runner.kf.x[2, 0]
        estimated_agl = np.clip(current_msl - dem_now, 0.0, 500.0)
        height_m = dem_now + estimated_agl
        xy_uncertainty = float(np.trace(self.runner.kf.P[0:2, 0:2]))
        has_valid_dem = True

        dem_decision = None
        dem_policy_scales = {
            "r_scale": 1.0,
            "chi2_scale": 1.0,
            "threshold_scale": 1.0,
            "reproj_scale": 1.0,
        }
        if getattr(self.runner, "policy_runtime_service", None) is not None:
            try:
                dem_decision = self.runner.policy_runtime_service.get_sensor_decision("DEM", float(t))
                dem_policy_scales = {
                    "r_scale": float(dem_decision.r_scale),
                    "chi2_scale": float(dem_decision.chi2_scale),
                    "threshold_scale": float(dem_decision.threshold_scale),
                    "reproj_scale": float(dem_decision.reproj_scale),
                }
            except Exception:
                dem_decision = None
        if dem_decision is not None and str(dem_decision.mode).upper() in ("HOLD", "SKIP"):
            dem_adaptive_info: Dict[str, Any] = {
                "sensor": "DEM",
                "accepted": False,
                "attempted": 0,
                "dof": 1,
                "nis_norm": np.nan,
                "chi2": np.nan,
                "threshold": np.nan,
                "r_scale_used": float(dem_policy_scales.get("r_scale", 1.0)),
                "reason_code": f"policy_mode_{str(dem_decision.mode).lower()}",
            }
            self.runner.adaptive_service.record_adaptive_measurement(
                "DEM",
                adaptive_info=dem_adaptive_info,
                timestamp=t,
                policy_scales=dem_policy_scales,
            )
            return

        dem_adaptive_info: Dict[str, Any] = {}
        apply_dem_height_update(
            self.runner.kf,
            height_measurement=height_m,
            sigma_height=sigma_height,
            xy_uncertainty=xy_uncertainty,
            time_since_correction=time_since_correction,
            speed_ms=speed,
            has_valid_dem=has_valid_dem,
            no_vision_corrections=no_vision_corrections,
            timestamp=t,
            residual_csv=residual_path,
            frame=self.runner.state.vio_frame,
            threshold_scale=float(dem_policy_scales.get("threshold_scale", 1.0)),
            r_scale_extra=float(dem_policy_scales.get("r_scale", 1.0)),
            adaptive_info=dem_adaptive_info,
        )
        self.runner.adaptive_service.record_adaptive_measurement(
            "DEM",
            adaptive_info=dem_adaptive_info,
            timestamp=t,
            policy_scales=dem_policy_scales,
        )
