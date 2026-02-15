import numpy as np

from vio.ekf import ExtendedKalmanFilter
from vio.measurement_updates import apply_vio_velocity_update


class _DummyIMU:
    def __init__(self):
        self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)  # scipy xyzw
        self.ang = np.zeros(3, dtype=float)


class _DummyDEM:
    ds = None

    @staticmethod
    def sample_m(_lat, _lon):
        return 0.0


class _DummyProj:
    @staticmethod
    def xy_to_latlon(_x, _y, lat0, lon0):
        return float(lat0), float(lon0)


def _make_cfg():
    return {
        "KB_PARAMS": {"mu": 829.0},
        "SIGMA_VO": 0.8,
        "CAMERA_VIEW_CONFIGS": {
            "nadir": {
                "extrinsics": "BODY_T_CAMDOWN",
                "sigma_scale_xy": 1.0,
                "sigma_scale_z": 2.0,
                "use_vz_only": False,
            }
        },
        "BODY_T_CAMDOWN": np.eye(4, dtype=float),
        "vio": {
            "use_vz_only": False,
            "min_parallax_px": 0.3,
            "nadir_prefer_flow_direction": False,
            "nadir_enforce_xy_motion": True,
        },
        "VIO_NADIR_XY_ONLY_VELOCITY": True,
        "VIO_VEL_XY_ONLY_CHI2_SCALE": 1.10,
        "VIO_VEL_SPEED_R_INFLATE_BREAKPOINTS_M_S": [25.0, 40.0, 55.0],
        "VIO_VEL_SPEED_R_INFLATE_VALUES": [1.5, 2.5, 4.0],
        "VIO_VEL_MAX_DELTA_V_XY_PER_UPDATE_M_S": 2.0,
        "VIO_VEL_MIN_FLOW_PX_HIGH_SPEED": 0.8,
    }


def _make_kf(speed_xy_m_s: float) -> ExtendedKalmanFilter:
    kf = ExtendedKalmanFilter(dim_x=19, dim_z=3, dim_u=3)
    kf.x = np.zeros((19, 1), dtype=float)
    kf.x[6, 0] = 1.0
    kf.x[3, 0] = float(speed_xy_m_s)
    kf.P = np.eye(18, dtype=float) * 0.05
    return kf


def test_vio_vel_rejects_low_flow_at_high_speed():
    kf = _make_kf(30.0)
    imu = _DummyIMU()
    info = {}
    ok = apply_vio_velocity_update(
        kf=kf,
        r_vo_mat=np.eye(3, dtype=float),
        t_unit=np.array([1.0, 0.0, 0.0], dtype=float),
        t=1.0,
        dt_img=0.05,
        avg_flow_px=0.3,
        imu_rec=imu,
        global_config=_make_cfg(),
        camera_view="nadir",
        dem_reader=_DummyDEM(),
        lat0=45.0,
        lon0=-75.0,
        use_vio_velocity=True,
        proj_cache=_DummyProj(),
        save_debug=False,
        adaptive_info=info,
    )
    assert ok is False
    assert info.get("reason_code") == "soft_reject_low_flow"


def test_vio_vel_rejects_when_delta_v_xy_exceeds_cap():
    kf = _make_kf(30.0)
    imu = _DummyIMU()
    info = {}
    ok = apply_vio_velocity_update(
        kf=kf,
        r_vo_mat=np.eye(3, dtype=float),
        t_unit=np.array([1.0, 0.0, 0.0], dtype=float),
        t=2.0,
        dt_img=0.05,
        avg_flow_px=5.0,
        imu_rec=imu,
        global_config=_make_cfg(),
        camera_view="nadir",
        dem_reader=_DummyDEM(),
        lat0=45.0,
        lon0=-75.0,
        use_vio_velocity=True,
        proj_cache=_DummyProj(),
        save_debug=False,
        adaptive_info=info,
    )
    assert ok is False
    assert info.get("reason_code") == "soft_reject_delta_v_cap"
