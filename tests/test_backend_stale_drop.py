import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vio.backend_optimizer import BackendOptimizer


def test_backend_optimizer_stale_drop_count_increments():
    backend = BackendOptimizer(
        fixed_lag_window=3,
        optimize_rate_hz=50.0,
        max_iteration_ms=100.0,
        max_correction_age_sec=0.01,
        min_quality_score=0.05,
    )
    backend.start()
    try:
        backend.push_keyframe(1.00, np.array([0.0, 0.0, 0.0]), 0.0, 0.8)
        backend.push_keyframe(1.10, np.array([1.0, 0.0, 0.0]), 1.0, 0.8)
        backend.push_keyframe(1.20, np.array([2.0, 0.1, 0.0]), 2.0, 0.8)
        backend.report_absolute_hint(1.15, np.array([8.0, 1.0, 0.0]), 0.2, 0.9)

        time.sleep(0.08)
        corr = backend.poll_correction(t_now=10.0)
        assert corr is None
        assert int(getattr(backend, "last_poll_stale_drops", 0)) >= 1
    finally:
        backend.stop()
