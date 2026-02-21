"""Policy snapshot dataclasses.

These immutable-ish types are shared across services so a single runtime
authority can decide policy once per step and all sensor paths consume the
same decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class SensorPolicyDecision:
    """Decision for one sensor/update path at a given step."""

    sensor: str
    mode: str = "APPLY"  # APPLY|SOFT_APPLY|HOLD|SKIP
    r_scale: float = 1.0
    chi2_scale: float = 1.0
    threshold_scale: float = 1.0
    reproj_scale: float = 1.0
    reason_codes: Tuple[str, ...] = field(default_factory=tuple)
    extras: Mapping[str, Any] = field(default_factory=dict)
    valid_from_t: float = 0.0
    valid_to_t: float = float("inf")

    def extra(self, key: str, default: float) -> float:
        val = self.extras.get(key, default)
        try:
            return float(val)
        except Exception:
            return float(default)

    def extra_str(self, key: str, default: str) -> str:
        val = self.extras.get(key, default)
        try:
            return str(val)
        except Exception:
            return str(default)

    def as_trace_dict(self) -> Dict[str, float]:
        return {
            "sensor": str(self.sensor),
            "mode": str(self.mode),
            "r_scale": float(self.r_scale),
            "chi2_scale": float(self.chi2_scale),
            "threshold_scale": float(self.threshold_scale),
            "reproj_scale": float(self.reproj_scale),
            "valid_from_t": float(self.valid_from_t),
            "valid_to_t": float(self.valid_to_t),
        }


@dataclass(frozen=True)
class PolicySnapshot:
    """Single-step policy bundle consumed by all update services."""

    timestamp: float
    phase: int
    health_state: str
    speed_m_s: float
    decisions: Mapping[str, SensorPolicyDecision] = field(default_factory=dict)
    owner_map: Mapping[str, str] = field(default_factory=dict)

    def decision_for(self, sensor: str) -> SensorPolicyDecision:
        return self.decisions.get(
            str(sensor),
            SensorPolicyDecision(sensor=str(sensor)),
        )
