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


def _make_kf() -> ExtendedKalmanFilter:
    kf = ExtendedKalmanFilter(dim_x=19, dim_z=3, dim_u=3)
    kf.x = np.zeros((19, 1), dtype=float)
    kf.x[6, 0] = 1.0
    kf.P = np.eye(18, dtype=float) * 0.05
    return kf


def test_vio_vel_xy_only_mode_uses_2dof():
    kf = _make_kf()
    imu = _DummyIMU()
    info = {}
    cfg = {
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
    }

    applied = apply_vio_velocity_update(
        kf=kf,
        r_vo_mat=np.eye(3, dtype=float),
        t_unit=np.array([1.0, 0.0, 0.0], dtype=float),
        t=1.0,
        dt_img=0.05,
        avg_flow_px=8.0,
        imu_rec=imu,
        global_config=cfg,
        camera_view="nadir",
        dem_reader=_DummyDEM(),
        lat0=45.0,
        lon0=-75.0,
        use_vio_velocity=True,
        proj_cache=_DummyProj(),
        save_debug=False,
        chi2_scale=10.0,
        adaptive_info=info,
    )

    assert applied is True
    assert int(info.get("dof", -1)) == 2
    assert info.get("accepted") is True

