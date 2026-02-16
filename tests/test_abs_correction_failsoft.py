import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vio.services.vio_service import VIOService


class _DummyProj:
    @staticmethod
    def latlon_to_xy(lat, lon, lat0, lon0):
        return (lon - lon0) * 100000.0, (lat - lat0) * 100000.0

    @staticmethod
    def xy_to_latlon(x, y, lat0, lon0):
        return lat0 + y / 100000.0, lon0 + x / 100000.0


class _DummyRunner:
    def __init__(self):
        self.global_config = {
            "VPS_ABS_HARD_REJECT_OFFSET_M": 180.0,
            "VPS_ABS_HARD_REJECT_DIR_CHANGE_DEG": 75.0,
            "VPS_ABS_MAX_APPLY_DP_XY_M": 25.0,
            "VPS_APPLY_FAILSOFT_MAX_DIR_CHANGE_DEG": 60.0,
            "VPS_APPLY_FAILSOFT_LARGE_OFFSET_CONFIRM_M": 80.0,
            "VPS_APPLY_FAILSOFT_LARGE_OFFSET_CONFIRM_HITS": 2,
        }
        self.proj_cache = _DummyProj()
        self.lat0 = 0.0
        self.lon0 = 0.0
        self._vps_last_accepted_offset_vec = np.array([1.0, 0.0], dtype=float)
        self._vps_pending_large_offset_vec = None
        self._vps_pending_large_offset_hits = 0
        self._vps_jump_reject_count = 0
        self._vps_temporal_confirm_count = 0


def test_vps_hard_reject_on_large_offset():
    runner = _DummyRunner()
    svc = VIOService(runner)

    ok, reason = svc._check_vps_hard_reject(np.array([220.0, 0.0]), 220.0)
    assert not ok
    assert "HARD_REJECT_OFFSET" in reason
    assert runner._vps_jump_reject_count == 1


def test_vps_clamp_limits_large_apply_step():
    runner = _DummyRunner()
    svc = VIOService(runner)

    lat, lon, scale = svc._clamp_vps_latlon(
        current_xy=np.array([0.0, 0.0]),
        vps_lat=0.0,
        vps_lon=0.002,  # 200m east in dummy projection
    )
    assert scale < 1.0
    x, y = runner.proj_cache.latlon_to_xy(lat, lon, runner.lat0, runner.lon0)
    assert np.hypot(x, y) <= 25.0 + 1e-6
