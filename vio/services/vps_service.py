"""VPS update service for VIORunner."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..vps_integration import apply_vps_update, compute_vps_innovation, compute_vps_acceptance_threshold


class VPSService:
    """Encapsulates VPS update logic (event-driven single-update path)."""

    def __init__(self, runner: Any):
        self.runner = runner

    def process_single_vps(self, vps, t: float):
        """
        Process a single VPS measurement (for event-driven mode).

        Args:
            vps: VPS record with lat, lon, etc.
            t: VPS timestamp
        """
        sigma_vps = self.runner.global_config.get("SIGMA_VPS_XY", 1.0)

        # Compute innovation and Mahalanobis distance
        vps_xy, innovation, m2_test = compute_vps_innovation(
            vps, self.runner.kf, self.runner.lat0, self.runner.lon0, self.runner.proj_cache
        )

        # Compute adaptive acceptance threshold
        time_since_correction = t - getattr(self.runner.kf, "last_absolute_correction_time", t)
        innovation_mag = float(np.linalg.norm(innovation))
        max_innovation_m, r_scale, tier_name = compute_vps_acceptance_threshold(
            time_since_correction, innovation_mag
        )

        # Chi-square threshold (2 DOF)
        chi2_threshold = 5.99  # 95% confidence

        # Gate by innovation magnitude OR chi-square test
        if innovation_mag > max_innovation_m:
            print(
                f"[VPS] REJECTED: innovation {innovation_mag:.1f}m > {max_innovation_m:.1f}m "
                f"({tier_name}, drift={time_since_correction:.1f}s)"
            )
            return

        if m2_test > chi2_threshold * 10:  # Very permissive for first VPS
            print(
                f"[VPS] REJECTED: chi2={m2_test:.1f} >> {chi2_threshold} "
                f"(innovation={innovation_mag:.1f}m)"
            )
            return

        # Apply update with adaptive R scaling
        applied = apply_vps_update(
            self.runner.kf,
            vps_xy=vps_xy,
            sigma_vps=sigma_vps,
            r_scale=r_scale,
        )

        if applied:
            self.runner.kf.last_absolute_correction_time = t
            print(
                f"[VPS] Applied at t={t:.3f}, innovation={innovation_mag:.1f}m, "
                f"tier={tier_name}, R_scale={r_scale:.1f}x"
            )
