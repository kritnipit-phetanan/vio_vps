import numpy as np

from vio.ekf import ExtendedKalmanFilter, ensure_covariance_valid


def _make_kf() -> ExtendedKalmanFilter:
    kf = ExtendedKalmanFilter(dim_x=19, dim_z=3, dim_u=3)
    kf.x = np.zeros((19, 1), dtype=float)
    kf.x[6, 0] = 1.0  # unit quaternion [w, x, y, z]
    kf.P = np.eye(18, dtype=float) * 1e-3
    return kf


def test_conditioning_psd_matrix_skips_projection():
    kf = _make_kf()
    p_new = ensure_covariance_valid(
        kf.P.copy(),
        label="unit_psd",
        conditioner=kf,
        timestamp=10.0,
        stage="UNIT",
        max_value=1e8,
    )
    stats = kf.get_conditioning_stats()
    assert np.all(np.isfinite(p_new))
    assert stats["projection_count"] == 0
    assert stats["chol_fail_count"] == 0


def test_conditioning_indefinite_matrix_uses_fallback_projection():
    kf = _make_kf()
    p_bad = kf.P.copy()
    p_bad[0, 1] = 2.0
    p_bad[1, 0] = 2.0  # creates a negative eigenvalue in 2x2 block
    p_new = ensure_covariance_valid(
        p_bad,
        label="unit_indef",
        conditioner=kf,
        timestamp=20.0,
        stage="UNIT",
        max_value=1e8,
    )
    stats = kf.get_conditioning_stats()
    assert np.all(np.isfinite(p_new))
    assert np.min(np.linalg.eigvalsh((p_new + p_new.T) / 2.0)) >= 0.0
    assert stats["projection_count"] >= 1
    assert stats["chol_fail_count"] >= 1


def test_conditioning_non_finite_matrix_triggers_reset_counter():
    kf = _make_kf()
    p_bad = kf.P.copy()
    p_bad[0, 0] = np.nan
    p_bad[1, 1] = np.inf
    p_new = ensure_covariance_valid(
        p_bad,
        label="unit_non_finite",
        conditioner=kf,
        timestamp=30.0,
        stage="UNIT",
        max_value=1e8,
    )
    stats = kf.get_conditioning_stats()
    assert np.all(np.isfinite(p_new))
    assert stats["projection_count"] >= 1
    assert stats["reset_count"] >= 1


def test_first_projection_time_is_set_once():
    kf = _make_kf()
    p_bad = kf.P.copy()
    p_bad[0, 1] = 2.0
    p_bad[1, 0] = 2.0

    ensure_covariance_valid(
        p_bad,
        label="first_proj_1",
        conditioner=kf,
        timestamp=111.0,
        stage="UNIT",
        max_value=1e8,
    )
    ensure_covariance_valid(
        p_bad,
        label="first_proj_2",
        conditioner=kf,
        timestamp=222.0,
        stage="UNIT",
        max_value=1e8,
    )
    stats = kf.get_conditioning_stats()
    assert np.isfinite(stats["first_projection_time"])
    assert abs(stats["first_projection_time"] - 111.0) < 1e-9
