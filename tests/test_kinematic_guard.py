import numpy as np

from vio.ekf import ExtendedKalmanFilter
from vio.services.kinematic_guard_service import KinematicGuardService


class _DummyReporting:
    def log_convention_check(self, **_kwargs):
        return None


class _DummyRunner:
    def __init__(self):
        self.kf = ExtendedKalmanFilter(dim_x=19, dim_z=3, dim_u=3)
        self.kf.x = np.zeros((19, 1), dtype=float)
        self.kf.x[6, 0] = 1.0
        self.kf.P = np.eye(18, dtype=float) * 1e-2
        self.global_config = {
            "KIN_GUARD_ENABLED": True,
            "KIN_GUARD_WINDOW_SEC": 2.0,
            "KIN_GUARD_VEL_MISMATCH_WARN": 8.0,
            "KIN_GUARD_VEL_MISMATCH_HARD": 15.0,
            "KIN_GUARD_MAX_INFLATE": 1.5,
            "KIN_GUARD_HARD_BLEND_ALPHA": 0.20,
        }
        self.output_reporting = _DummyReporting()
        self._kin_guard_samples = 0
        self._kin_guard_trigger_count = 0
        self._kin_guard_hard_count = 0
        self._kin_guard_last_mismatch = np.nan


def test_kinematic_guard_inflates_cov_and_blends_velocity():
    runner = _DummyRunner()
    guard = KinematicGuardService(runner)

    # First sample.
    runner.kf.x[0:3, 0] = np.array([0.0, 0.0, 0.0], dtype=float)
    runner.kf.x[3:6, 0] = np.array([30.0, 0.0, 0.0], dtype=float)
    p_before = runner.kf.P[3:6, 3:6].copy()
    guard.apply(0.0)

    # Position indicates only 1 m/s; state says 30 m/s -> hard mismatch.
    runner.kf.x[0:3, 0] = np.array([1.0, 0.0, 0.0], dtype=float)
    guard.apply(1.0)

    assert int(runner._kin_guard_samples) >= 1
    assert int(runner._kin_guard_trigger_count) >= 1
    assert int(runner._kin_guard_hard_count) >= 1
    assert float(runner._kin_guard_last_mismatch) > 15.0
    assert np.trace(runner.kf.P[3:6, 3:6]) > np.trace(p_before)
    # Hard blend should reduce exaggerated speed.
    assert float(runner.kf.x[3, 0]) < 30.0
