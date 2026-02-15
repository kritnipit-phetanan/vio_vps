import numpy as np

from vio.ekf import ExtendedKalmanFilter
from vio.services.kinematic_guard_service import KinematicGuardService


class _DummyOutput:
    @staticmethod
    def log_convention_check(**_kwargs):
        return None


class _DummyRunner:
    def __init__(self):
        self.global_config = {
            "KIN_GUARD_ENABLED": True,
            "KIN_GUARD_WINDOW_SEC": 0.5,
            "KIN_GUARD_VEL_MISMATCH_WARN": 6.0,
            "KIN_GUARD_VEL_MISMATCH_HARD": 12.0,
            "KIN_GUARD_MAX_INFLATE": 1.08,
            "KIN_GUARD_HARD_BLEND_ALPHA": 0.12,
            "KIN_GUARD_MIN_ACTION_DT_SEC": 0.0,
            "KIN_GUARD_MAX_STATE_SPEED_M_S": 70.0,
            "KIN_GUARD_MAX_BLEND_SPEED_M_S": 70.0,
            "KIN_GUARD_MAX_KIN_SPEED_M_S": 65.0,
            "KIN_GUARD_HARD_HOLD_SEC": 0.30,
            "KIN_GUARD_RELEASE_HYSTERESIS_RATIO": 0.75,
        }
        self.kf = ExtendedKalmanFilter(dim_x=19, dim_z=3, dim_u=3)
        self.kf.x = np.zeros((19, 1), dtype=float)
        self.kf.x[6, 0] = 1.0
        self.kf.P = np.eye(18, dtype=float) * 0.05
        self.output_reporting = _DummyOutput()
        self._kin_guard_samples = 0
        self._kin_guard_trigger_count = 0
        self._kin_guard_hard_count = 0
        self._kin_guard_speed_clamp_count = 0


def test_kinematic_guard_hard_clamp_after_dwell():
    runner = _DummyRunner()
    svc = KinematicGuardService(runner)

    # Seed history.
    runner.kf.x[0:3, 0] = np.array([0.0, 0.0, 0.0], dtype=float)
    runner.kf.x[3:6, 0] = np.array([100.0, 0.0, 0.0], dtype=float)
    svc.apply(0.0)

    # First hard mismatch sample: starts dwell timer, should not hard-clamp yet.
    runner.kf.x[0:3, 0] = np.array([0.0, 0.0, 0.0], dtype=float)
    svc.apply(0.1)
    assert float(np.linalg.norm(runner.kf.x[3:6, 0])) > 70.0

    # After hold duration, hard condition activates and speed is clamped.
    svc.apply(0.5)
    assert float(np.linalg.norm(runner.kf.x[3:6, 0])) <= 70.0 + 1e-6
    assert int(runner._kin_guard_hard_count) >= 1
    assert int(runner._kin_guard_speed_clamp_count) >= 1
