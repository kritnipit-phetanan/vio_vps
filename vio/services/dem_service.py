"""DEM/height update service for VIORunner."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..measurement_updates import apply_dem_height_update


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

        # Get current position
        lat_now, lon_now = self.runner.proj_cache.xy_to_latlon(
            self.runner.kf.x[0, 0], self.runner.kf.x[1, 0], self.runner.lat0, self.runner.lon0
        )

        # Sample DEM
        dem_now = self.runner.dem.sample_m(lat_now, lon_now) if self.runner.dem.ds else None

        if dem_now is None or np.isnan(dem_now):
            return

        # Compute height measurement
        # v3.9.12: Try Direct MSL from Flight Log (Barometer/GNSS) first
        # This replaces DEM+AGL logic if available
        msl_direct = None
        if self.runner.msl_interpolator:
            raw_msl = self.runner.msl_interpolator.get_msl(t)
            if raw_msl is not None:
                # Apply the alignment offset (Result = Log - Offset)
                msl_direct = raw_msl - getattr(self.runner, "msl_offset", 0.0)

        if msl_direct is not None:
            # Direct MSL update
            height_m = msl_direct
            # Disable XY uncertainty scaling (baro/GNSS doesn't depend on XY)
            xy_uncertainty = 0.0
            has_valid_dem = True  # Treat as valid measurement
        else:
            # Fallback: Compute height measurement from DEM + estimated AGL
            # CRITICAL FIX v3.9.2: DO NOT use GPS MSL - that causes innovation=0!
            # DEM update must constrain drift by comparing: (DEM + estimated_AGL) vs state
            # State is MSL, measurement = DEM + estimated_AGL

            # Estimate AGL from current MSL state and DEM
            current_msl = self.runner.kf.x[2, 0]
            estimated_agl = current_msl - dem_now

            # Clamp AGL to reasonable range (helicopter doesn't fly underground or >500m AGL)
            estimated_agl = np.clip(estimated_agl, 0.0, 500.0)

            # Reconstruct MSL measurement from DEM + estimated AGL
            height_m = dem_now + estimated_agl
            has_valid_dem = True  # Since we checked dem_now is not None/NaN before

        # Compute uncertainties
        if msl_direct is None:
            # Only use XY uncertainty for DEM-based update
            xy_uncertainty = float(np.trace(self.runner.kf.P[0:2, 0:2]))

        time_since_correction = t - getattr(self.runner.kf, "last_absolute_correction_time", t)
        speed = float(np.linalg.norm(self.runner.kf.x[3:6, 0]))
        dem_policy_scales, dem_apply_scales = self.runner.adaptive_service.get_sensor_adaptive_scales("DEM")
        dem_adaptive_info: Dict[str, Any] = {}

        # Get residual_csv path if debug data is enabled
        residual_path = (
            self.runner.residual_csv
            if self.runner.config.save_debug_data and hasattr(self.runner, "residual_csv")
            else None
        )

        applied, reason = apply_dem_height_update(
            self.runner.kf,
            height_measurement=height_m,
            sigma_height=sigma_height,
            xy_uncertainty=xy_uncertainty,
            time_since_correction=time_since_correction,
            speed_ms=speed,
            has_valid_dem=has_valid_dem,
            no_vision_corrections=(len(self.runner.imgs) == 0 and len(self.runner.vps_list) == 0),
            timestamp=t,
            residual_csv=residual_path,
            frame=self.runner.state.vio_frame,
            threshold_scale=float(dem_apply_scales.get("threshold_scale", 1.0)),
            r_scale_extra=float(dem_apply_scales.get("r_scale", 1.0)),
            adaptive_info=dem_adaptive_info,
        )
        self.runner.adaptive_service.record_adaptive_measurement(
            "DEM",
            adaptive_info=dem_adaptive_info,
            timestamp=t,
            policy_scales=dem_policy_scales,
        )
