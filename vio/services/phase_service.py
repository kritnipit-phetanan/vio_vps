"""Flight phase estimation service for VIORunner."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..propagation import get_flight_phase


class PhaseService:
    """Encapsulates flight-phase estimation + hysteresis state transitions."""

    def __init__(self, runner: Any):
        self.runner = runner

    def estimate_flight_phase(self,
                              velocity: Optional[np.ndarray],
                              velocity_sigma: Optional[float],
                              vibration_level: Optional[float],
                              altitude_change: Optional[float],
                              dt: float) -> int:
        """
        Estimate flight phase with hysteresis / one-way protection.

        This avoids frequent NORMAL->EARLY regression when velocity covariance
        momentarily spikes late in flight.
        """
        raw_phase, _ = get_flight_phase(
            velocity=velocity,
            velocity_sigma=velocity_sigma,
            vibration_level=vibration_level,
            altitude_change=altitude_change,
            spinup_velocity_thresh=self.runner.PHASE_SPINUP_VELOCITY_THRESH,
            spinup_vibration_thresh=self.runner.PHASE_SPINUP_VIBRATION_THRESH,
            spinup_alt_change_thresh=self.runner.PHASE_SPINUP_ALT_CHANGE_THRESH,
            early_velocity_sigma_thresh=self.runner.PHASE_EARLY_VELOCITY_SIGMA_THRESH,
        )
        raw_phase = int(raw_phase)
        self.runner.state.phase_raw = raw_phase

        if not self.runner.state.phase_initialized:
            self.runner.state.current_phase = raw_phase
            self.runner.state.phase_initialized = True
            self.runner.state.phase_candidate = -1
            self.runner.state.phase_candidate_hold_sec = 0.0
            return self.runner.state.current_phase

        if not bool(self.runner.PHASE_HYSTERESIS_ENABLED):
            self.runner.state.current_phase = raw_phase
            self.runner.state.phase_candidate = -1
            self.runner.state.phase_candidate_hold_sec = 0.0
            return self.runner.state.current_phase

        current_phase = int(self.runner.state.current_phase)
        target_phase = raw_phase
        speed_xy = float(np.linalg.norm(velocity[:2])) if velocity is not None and len(velocity) >= 2 else 0.0
        alt_abs = abs(float(altitude_change)) if altitude_change is not None and np.isfinite(altitude_change) else 0.0

        # One-way protection: keep NORMAL unless we explicitly allow safe revert.
        if current_phase == 2 and raw_phase < 2:
            if not bool(self.runner.PHASE_ALLOW_NORMAL_TO_EARLY):
                target_phase = 2
            else:
                allow_revert = speed_xy <= max(0.1, float(self.runner.PHASE_REVERT_MAX_SPEED))
                allow_revert = allow_revert and alt_abs <= max(0.1, float(self.runner.PHASE_REVERT_MAX_ALT_CHANGE))
                if not allow_revert:
                    target_phase = 2

        # Avoid jumping SPINUP->NORMAL directly.
        if current_phase == 0 and target_phase == 2:
            target_phase = 1

        dt_eff = max(0.0, float(dt))
        if target_phase != current_phase:
            if int(self.runner.state.phase_candidate) != int(target_phase):
                self.runner.state.phase_candidate = int(target_phase)
                self.runner.state.phase_candidate_hold_sec = dt_eff
            else:
                self.runner.state.phase_candidate_hold_sec += dt_eff

            hold_sec = (
                float(self.runner.PHASE_UP_HOLD_SEC)
                if target_phase > current_phase
                else float(self.runner.PHASE_DOWN_HOLD_SEC)
            )
            if self.runner.state.phase_candidate_hold_sec >= max(0.0, hold_sec):
                self.runner.state.current_phase = int(target_phase)
                self.runner.state.phase_candidate = -1
                self.runner.state.phase_candidate_hold_sec = 0.0
        else:
            self.runner.state.phase_candidate = -1
            self.runner.state.phase_candidate_hold_sec = 0.0

        return int(self.runner.state.current_phase)
