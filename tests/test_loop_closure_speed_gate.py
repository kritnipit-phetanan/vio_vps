import numpy as np

from vio.ekf import ExtendedKalmanFilter
from vio.loop_closure import apply_loop_closure_correction


class _DummyLoopDetector:
    def __init__(self):
        self.stats = {"loop_speed_skipped": 0}


def _make_kf(speed_m_s: float) -> ExtendedKalmanFilter:
    kf = ExtendedKalmanFilter(dim_x=19, dim_z=3, dim_u=3)
    kf.x = np.zeros((19, 1), dtype=float)
    kf.x[6, 0] = 1.0
    kf.x[3, 0] = float(speed_m_s)
    kf.P = np.eye(18, dtype=float) * 0.05
    return kf


def test_loop_closure_skips_apply_when_speed_too_high():
    kf = _make_kf(speed_m_s=45.0)
    loop_info = {
        "yaw_correction": np.deg2rad(3.0),
        "num_inliers": 40,
        "kf_idx": 0,
        "phase": 2,
        "health_state": "HEALTHY",
        "fail_soft": False,
    }
    detector = _DummyLoopDetector()

    ok = apply_loop_closure_correction(
        kf=kf,
        loop_info=loop_info,
        t=10.0,
        cam_states=[],
        loop_detector=detector,
        global_config={
            "LOOP_SPEED_SKIP_M_S": 35.0,
            "LOOP_MIN_ABS_YAW_CORR_DEG": 1.0,
            "LOOP_YAW_RESIDUAL_BOUND_DEG": 30.0,
        },
    )

    assert ok is False
    assert int(detector.stats.get("loop_speed_skipped", 0)) == 1
