import numpy as np

from vio.ekf import ExtendedKalmanFilter
from vio.propagation import apply_zupt, apply_gravity_roll_pitch_update


def _make_kf() -> ExtendedKalmanFilter:
    kf = ExtendedKalmanFilter(dim_x=19, dim_z=3, dim_u=3)
    kf.x = np.zeros((19, 1), dtype=float)
    kf.x[6, 0] = 1.0  # unit quaternion [w, x, y, z]
    kf.P = np.eye(18, dtype=float) * 1e-3
    kf.P[3:6, 3:6] = np.eye(3, dtype=float) * 1e-3
    return kf


def test_apply_zupt_soft_accept_path():
    kf = _make_kf()
    kf.x[3:6, 0] = np.array([0.24, 0.0, 0.0], dtype=float)
    info = {}

    applied, _, _ = apply_zupt(
        kf,
        v_mag=0.24,
        consecutive_stationary_count=10,
        soft_fail_enable=True,
        soft_fail_hard_reject_factor=6.0,
        soft_fail_r_cap=20.0,
        soft_fail_power=1.0,
        adaptive_info=info,
    )

    assert applied is True
    assert info.get("reason_code") == "soft_accept"
    assert info.get("accepted") is True


def test_apply_zupt_hard_reject_path():
    kf = _make_kf()
    kf.x[3:6, 0] = np.array([0.30, 0.0, 0.0], dtype=float)
    info = {}

    applied, _, _ = apply_zupt(
        kf,
        v_mag=0.30,
        consecutive_stationary_count=10,
        soft_fail_enable=False,
        adaptive_info=info,
    )

    assert applied is False
    assert info.get("reason_code") == "hard_reject"
    assert info.get("accepted") is False


def test_gravity_roll_pitch_pseudo_update_at_rest():
    kf = _make_kf()
    imu_params = {"g_norm": 9.80665, "accel_includes_gravity": True}
    info = {}

    applied, reason = apply_gravity_roll_pitch_update(
        kf,
        a_raw=np.array([0.0, 0.0, -9.80665], dtype=float),
        w_corr=np.array([0.0, 0.0, 0.0], dtype=float),
        imu_params=imu_params,
        adaptive_info=info,
    )

    assert applied is True
    assert reason == "applied"
    assert info.get("reason_code") == "normal_accept"
