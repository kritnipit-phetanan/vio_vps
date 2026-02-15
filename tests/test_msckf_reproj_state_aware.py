import math

from vio.msckf import _state_aware_reproj_policy


def test_state_aware_reproj_policy_relaxes_in_degraded():
    cfg = {
        "MSCKF_REPROJ_QUALITY_HIGH_TH": 0.75,
        "MSCKF_REPROJ_QUALITY_LOW_TH": 0.45,
        "MSCKF_REPROJ_QUALITY_MID_GATE_MULT": 1.15,
        "MSCKF_REPROJ_QUALITY_LOW_REJECT": True,
        "MSCKF_REPROJ_WARNING_SCALE": 1.20,
        "MSCKF_REPROJ_DEGRADED_SCALE": 1.35,
    }
    healthy_high = _state_aware_reproj_policy(
        feature_quality=0.9, phase=2, health_state="HEALTHY", global_config=cfg
    )
    degraded_mid = _state_aware_reproj_policy(
        feature_quality=0.6, phase=1, health_state="DEGRADED", global_config=cfg
    )

    assert healthy_high["quality_band"] == "high"
    assert degraded_mid["quality_band"] == "mid"
    assert degraded_mid["gate_mult"] > healthy_high["gate_mult"]
    assert degraded_mid["avg_gate_mult"] >= 1.0
    assert math.isfinite(degraded_mid["gate_mult"])


def test_state_aware_reproj_policy_low_quality_reject():
    cfg = {"MSCKF_REPROJ_QUALITY_LOW_REJECT": True}
    low = _state_aware_reproj_policy(
        feature_quality=0.1, phase=2, health_state="HEALTHY", global_config=cfg
    )
    assert low["quality_band"] == "low"
    assert low["low_quality_reject"] is True

