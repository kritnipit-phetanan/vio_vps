import numpy as np

from vio.services.vio_service import VIOService


class _DummyRunner:
    def __init__(self):
        self.global_config = {
            "VPS_APPLY_FAILSOFT_MAX_DIR_CHANGE_DEG": 60.0,
            "VPS_APPLY_FAILSOFT_LARGE_OFFSET_CONFIRM_M": 80.0,
            "VPS_APPLY_FAILSOFT_LARGE_OFFSET_CONFIRM_HITS": 2,
        }
        self._vps_last_accepted_offset_vec = None
        self._vps_pending_large_offset_vec = None
        self._vps_pending_large_offset_hits = 0


def test_vps_failsoft_temporal_rejects_large_direction_change():
    runner = _DummyRunner()
    runner._vps_last_accepted_offset_vec = np.array([1.0, 0.0], dtype=float)
    svc = VIOService(runner)

    ok, reason = svc._evaluate_vps_failsoft_temporal_gate(
        vps_offset=np.array([0.0, 90.0], dtype=float),
        vps_offset_m=90.0,
    )
    assert ok is False
    assert "SOFT_REJECT_DIR_CHANGE" in reason


def test_vps_failsoft_large_offset_requires_consecutive_confirmation():
    runner = _DummyRunner()
    svc = VIOService(runner)
    offset = np.array([90.0, 5.0], dtype=float)

    ok1, reason1 = svc._evaluate_vps_failsoft_temporal_gate(
        vps_offset=offset,
        vps_offset_m=float(np.linalg.norm(offset)),
    )
    assert ok1 is False
    assert "SOFT_REJECT_LARGE_OFFSET_PENDING" in reason1
    assert int(runner._vps_pending_large_offset_hits) == 1

    ok2, reason2 = svc._evaluate_vps_failsoft_temporal_gate(
        vps_offset=offset,
        vps_offset_m=float(np.linalg.norm(offset)),
    )
    assert ok2 is True
    assert reason2 == ""
    assert int(runner._vps_pending_large_offset_hits) == 0
