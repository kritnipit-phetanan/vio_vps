import numpy as np

from vio.ekf import ExtendedKalmanFilter
from vio.math_utils import quaternion_to_yaw
from vio.measurement_updates import apply_magnetometer_update


def _make_kf_with_yaw(yaw_rad: float) -> ExtendedKalmanFilter:
    kf = ExtendedKalmanFilter(dim_x=19, dim_z=3, dim_u=3)
    kf.x = np.zeros((19, 1), dtype=float)
    kf.x[6, 0] = np.cos(0.5 * yaw_rad)
    kf.x[9, 0] = np.sin(0.5 * yaw_rad)
    kf.P = np.eye(18, dtype=float) * 1e-2
    kf.P[8, 8] = 0.25
    return kf


def test_mag_warning_mode_is_weaker_than_healthy():
    cfg = {
        "MAG_WARNING_R_MULT": 4.0,
        "MAG_DEGRADED_R_MULT": 8.0,
        "MAG_WARNING_MAX_DYAW_DEG": 1.5,
        "MAG_DEGRADED_MAX_DYAW_DEG": 1.0,
    }
    # Raw heading from this vector is near 22deg; state starts at 55deg.
    mag_vec = np.array([0.93, -0.38, 0.1], dtype=float)
    yaw_init = np.deg2rad(55.0)

    kf_healthy = _make_kf_with_yaw(yaw_init)
    y_before_h = float(quaternion_to_yaw(kf_healthy.x[6:10, 0]))
    ok_h, _ = apply_magnetometer_update(
        kf_healthy,
        mag_calibrated=mag_vec,
        mag_declination=0.0,
        use_raw_heading=True,
        sigma_mag_yaw=0.10,
        global_config=cfg,
        current_phase=2,
        health_state="HEALTHY",
    )
    y_after_h = float(quaternion_to_yaw(kf_healthy.x[6:10, 0]))

    kf_warn = _make_kf_with_yaw(yaw_init)
    y_before_w = float(quaternion_to_yaw(kf_warn.x[6:10, 0]))
    ok_w, _ = apply_magnetometer_update(
        kf_warn,
        mag_calibrated=mag_vec,
        mag_declination=0.0,
        use_raw_heading=True,
        sigma_mag_yaw=0.10,
        global_config=cfg,
        current_phase=2,
        health_state="WARNING",
    )
    y_after_w = float(quaternion_to_yaw(kf_warn.x[6:10, 0]))

    assert ok_h and ok_w
    delta_h = abs(float(np.arctan2(np.sin(y_after_h - y_before_h), np.cos(y_after_h - y_before_h))))
    delta_w = abs(float(np.arctan2(np.sin(y_after_w - y_before_w), np.cos(y_after_w - y_before_w))))
    assert delta_w < delta_h

