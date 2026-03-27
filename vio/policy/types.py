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


@dataclass(frozen=True)
class MsckfQualitySnapshot:
    """Compact MSCKF quality view exported to policy/runtime logs."""

    timestamp: float
    track_count: int
    inlier_ratio: float
    parallax_med_px: float
    reproj_p95_norm: float
    depth_positive_ratio: float
    quality_score: float
    stable_geometry_flag: bool = False
    conditioning_risk: float = float("nan")
    feature_track_health: float = float("nan")
    unstable_reason_code: str = "stable"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "track_count": float(self.track_count),
            "inlier_ratio": float(self.inlier_ratio),
            "parallax_med_px": float(self.parallax_med_px),
            "reproj_p95_norm": float(self.reproj_p95_norm),
            "depth_positive_ratio": float(self.depth_positive_ratio),
            "quality_score": float(self.quality_score),
            "stable_geometry_flag": float(bool(self.stable_geometry_flag)),
            "conditioning_risk": float(self.conditioning_risk),
            "feature_track_health": float(self.feature_track_health),
            "unstable_reason_code": str(self.unstable_reason_code),
        }


@dataclass(frozen=True)
class HeadingOwnerState:
    """Runtime owner state snapshot for yaw authority tracing."""

    timestamp: float
    owner: str
    source: str
    mode: str
    reason: str
    score_mag: float
    score_loop: float
    score_backend: float
    score_msckf: float
    switched: bool = False


@dataclass(frozen=True)
class CorrectionDecisionState:
    """State-machine snapshot for backend correction decision lifecycle."""

    state: str = "PROPOSE"  # PROPOSE|PROBATION|COMMIT|REJECT|HINT_ONLY
    reason: str = ""
    source: str = "UNKNOWN"
    quality_score: float = float("nan")
    residual_xy: float = float("nan")
    age_sec: float = float("nan")
    t_ref: float = float("nan")


@dataclass(frozen=True)
class ApplyDecisionRecord:
    """Compact apply/reject trace payload for deterministic funnel telemetry."""

    timestamp: float
    source: str
    decision_state: str
    reason: str
    quality_score: float
    residual_xy: float
    dp_xy_in: float
    dp_xy_applied: float
    age_sec: float
    t_ref: float
    time_aligned_used: bool = False
    q_bucket: int = 0
    kinematic_stage: str = ""
    continuity_mode: str = ""
    direction_consistency_pass: int = -1
    magnitude_consistency_pass: int = -1
    magnitude_bounded_for_continuity: int = -1
    direction_soft_fail: int = -1
    continuity_eligible: int = -1
    source_rel: float = float("nan")
    kinematic_max_allow_m: float = float("nan")
    residual_eval_metric: float = float("nan")
