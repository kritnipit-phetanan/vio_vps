"""Structured runtime state container for VIORunner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class IndexState:
    """Progress indices across sensor streams."""

    imu_idx: int = 0
    img_idx: int = 0
    vps_idx: int = 0
    mag_idx: int = 0
    vio_frame: int = 0


@dataclass
class CounterState:
    """Update and helper counters used during runtime."""

    zupt_applied: int = 0
    zupt_rejected: int = 0
    zupt_detected: int = 0
    mag_updates: int = 0
    mag_rejects: int = 0
    consecutive_stationary: int = 0
    imu_propagation_count: int = 0


@dataclass
class TimingState:
    """Core runtime timing markers."""

    t0: float = 0.0
    last_t: float = 0.0


@dataclass
class PhaseState:
    """Flight-phase tracking with hysteresis state."""

    current_phase: int = 0  # 0=SPINUP, 1=EARLY, 2=NORMAL
    phase_initialized: bool = False
    phase_raw: int = 0
    phase_candidate: int = -1
    phase_candidate_hold_sec: float = 0.0


@dataclass
class FeatureState:
    """MSCKF/vision feature buffers."""

    cam_states: List[dict] = field(default_factory=list)
    cam_observations: List[dict] = field(default_factory=list)


@dataclass
class RunnerState:
    """
    Structured state container.

    Keeps grouped state explicit (`indices/counters/timing/phase/features`) while
    preserving legacy flat attribute access via compatibility mapping.
    """

    indices: IndexState = field(default_factory=IndexState)
    counters: CounterState = field(default_factory=CounterState)
    timing: TimingState = field(default_factory=TimingState)
    phase: PhaseState = field(default_factory=PhaseState)
    features: FeatureState = field(default_factory=FeatureState)

    _compat_map: Dict[str, Tuple[str, str]] = field(
        default_factory=lambda: {
            "imu_idx": ("indices", "imu_idx"),
            "img_idx": ("indices", "img_idx"),
            "vps_idx": ("indices", "vps_idx"),
            "mag_idx": ("indices", "mag_idx"),
            "vio_frame": ("indices", "vio_frame"),
            "zupt_applied": ("counters", "zupt_applied"),
            "zupt_rejected": ("counters", "zupt_rejected"),
            "zupt_detected": ("counters", "zupt_detected"),
            "mag_updates": ("counters", "mag_updates"),
            "mag_rejects": ("counters", "mag_rejects"),
            "consecutive_stationary": ("counters", "consecutive_stationary"),
            "imu_propagation_count": ("counters", "imu_propagation_count"),
            "t0": ("timing", "t0"),
            "last_t": ("timing", "last_t"),
            "current_phase": ("phase", "current_phase"),
            "phase_initialized": ("phase", "phase_initialized"),
            "phase_raw": ("phase", "phase_raw"),
            "phase_candidate": ("phase", "phase_candidate"),
            "phase_candidate_hold_sec": ("phase", "phase_candidate_hold_sec"),
            "cam_states": ("features", "cam_states"),
            "cam_observations": ("features", "cam_observations"),
        },
        repr=False,
    )

    def __getattr__(self, name: str):
        compat = self.__dict__.get("_compat_map", {})
        if name in compat:
            group_name, field_name = compat[name]
            return getattr(getattr(self, group_name), field_name)
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value):
        compat = self.__dict__.get("_compat_map")
        if compat and name in compat:
            group_name, field_name = compat[name]
            setattr(getattr(self, group_name), field_name, value)
            return
        super().__setattr__(name, value)
