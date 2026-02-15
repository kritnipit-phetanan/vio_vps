import numpy as np

from vio.services.output_reporting_service import build_sensor_time_audit_rows


def _row_by_sensor(rows, sensor):
    for row in rows:
        if row["sensor"] == sensor:
            return row
    raise AssertionError(f"sensor row not found: {sensor}")


def test_sensor_time_audit_nominal_pass():
    sensor_times = {
        "IMU": np.array([0.0, 0.01, 0.02, 0.03, 0.04]),
        "CAM": np.array([0.001, 0.011, 0.021, 0.031]),
        "MAG": np.array([0.0, 0.02, 0.04]),
    }
    rows = build_sensor_time_audit_rows(sensor_times, reference_sensor="IMU")
    cam = _row_by_sensor(rows, "CAM")

    assert cam["warn"] == 0.0
    assert cam["in_range_frac"] == 1.0
    assert cam["nn_dt_p95_s"] < 0.005


def test_sensor_time_audit_warns_for_low_overlap_and_large_dt():
    sensor_times = {
        "IMU": np.array([100.0, 100.01, 100.02, 100.03]),
        "CAM": np.array([99.0, 100.0, 101.0]),
    }
    rows = build_sensor_time_audit_rows(sensor_times, reference_sensor="IMU")
    cam = _row_by_sensor(rows, "CAM")

    assert cam["warn"] == 1.0
    assert cam["in_range_frac"] < 0.995
