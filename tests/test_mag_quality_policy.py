import numpy as np

from vio.services.magnetometer_service import MagnetometerService


class _DummyRunner:
    def __init__(self):
        self.global_config = {
            "MAG_ACCURACY_ENABLED": True,
            "MAG_ACCURACY_SKIP_ON_BAD": True,
            "MAG_ACCURACY_NORM_WINDOW": 64,
            "MAG_ACCURACY_GOOD_MIN_SCORE": 0.72,
            "MAG_ACCURACY_MID_MIN_SCORE": 0.45,
            "MAG_ACCURACY_R_INFLATE_MID": 1.8,
            "MAG_ACCURACY_R_INFLATE_BAD": 3.0,
            "MAG_ACCURACY_NORM_DEV_GOOD": 0.12,
            "MAG_ACCURACY_NORM_DEV_BAD": 0.30,
            "MAG_ACCURACY_GYRO_DELTA_SOFT_DEG": 20.0,
            "MAG_ACCURACY_GYRO_DELTA_HARD_DEG": 65.0,
            "MAG_ACCURACY_VISION_DELTA_SOFT_DEG": 30.0,
            "MAG_ACCURACY_VISION_DELTA_HARD_DEG": 90.0,
            "MAG_ACCURACY_VISION_WEIGHT": 0.25,
            "MAG_VISION_HEADING_MAX_AGE_SEC": 1.0,
        }
        self.last_gyro_z = 0.0
        self._vision_yaw_ref = None
        self._vision_yaw_last_t = None


def test_mag_quality_good_case():
    runner = _DummyRunner()
    svc = MagnetometerService(runner)

    out = svc._evaluate_accuracy_policy(
        yaw_mag_filtered=float(np.deg2rad(5.0)),
        mag_norm=0.63,
        timestamp=1.0,
    )
    assert out["decision"] == "good"
    assert out["r_scale"] == 1.0
    assert out["quality_score"] >= 0.72


def test_mag_quality_bad_skip_case():
    runner = _DummyRunner()
    svc = MagnetometerService(runner)

    # Seed history with a nominal sample first.
    svc._evaluate_accuracy_policy(
        yaw_mag_filtered=0.0,
        mag_norm=0.63,
        timestamp=0.0,
    )

    # Force strong gyro/vision inconsistency and norm deviation.
    svc._last_mag_yaw = 0.0
    svc._last_mag_t = 0.0
    runner.last_gyro_z = 0.0
    runner._vision_yaw_ref = 0.0
    runner._vision_yaw_last_t = 0.95

    out = svc._evaluate_accuracy_policy(
        yaw_mag_filtered=float(np.deg2rad(120.0)),
        mag_norm=2.0,
        timestamp=1.0,
    )
    assert out["decision"] == "bad_skip"
    assert out["r_scale"] >= runner.global_config["MAG_ACCURACY_R_INFLATE_BAD"]
    assert out["quality_score"] < runner.global_config["MAG_ACCURACY_MID_MIN_SCORE"]


def test_mag_quality_disabled_passthrough():
    runner = _DummyRunner()
    runner.global_config["MAG_ACCURACY_ENABLED"] = False
    svc = MagnetometerService(runner)

    out = svc._evaluate_accuracy_policy(
        yaw_mag_filtered=0.0,
        mag_norm=0.63,
        timestamp=1.0,
    )
    assert out["decision"] == "disabled"
    assert out["r_scale"] == 1.0
