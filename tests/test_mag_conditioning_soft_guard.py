import numpy as np

from vio.ekf import ExtendedKalmanFilter
from vio.services.magnetometer_service import MagnetometerService


class _DummyRunner:
    def __init__(self):
        self.global_config = {
            "MAG_CONDITIONING_GUARD_ENABLE": True,
            "MAG_CONDITIONING_GUARD_HARD_PCOND": 1.0e12,
            "MAG_CONDITIONING_GUARD_HARD_PMAX": 1.0e7,
        }
        self.kf = ExtendedKalmanFilter(dim_x=19, dim_z=3, dim_u=3)
        self.kf.x = np.zeros((19, 1), dtype=float)
        self.kf.x[6, 0] = 1.0
        self.kf.P = np.eye(18, dtype=float) * 0.05
        self.config = type("Cfg", (), {"use_mag_estimated_bias": True})()


def test_mag_conditioning_guard_warning_is_soft_until_hard_threshold():
    svc = MagnetometerService(_DummyRunner())

    skip_warn, reason_warn = svc._check_conditioning_guard(
        health_key="WARNING",
        p_max=9.0e6,
        p_cond=9.0e11,
    )
    assert skip_warn is False
    assert reason_warn == ""

    skip_hard, reason_hard = svc._check_conditioning_guard(
        health_key="WARNING",
        p_max=1.1e7,
        p_cond=9.0e11,
    )
    assert skip_hard is True
    assert reason_hard == "skip_conditioning_hard"
