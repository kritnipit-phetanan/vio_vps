import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vio.backend_optimizer import BackendOptimizer


def test_backend_optimizer_emits_bounded_correction():
    backend = BackendOptimizer(
        fixed_lag_window=3,
        optimize_rate_hz=40.0,
        max_iteration_ms=100.0,
        max_correction_age_sec=2.0,
        min_quality_score=0.05,
        max_abs_dp_xy_m=60.0,
        max_abs_dyaw_deg=8.0,
    )
    backend.start()
    try:
        backend.push_keyframe(1.00, np.array([0.0, 0.0, 0.0]), 0.0, 0.8)
        backend.push_keyframe(1.10, np.array([1.0, 0.0, 0.0]), 1.0, 0.8)
        backend.push_keyframe(1.20, np.array([2.0, 0.1, 0.0]), 2.0, 0.8)
        backend.report_absolute_hint(1.05, np.array([10.0, 2.0, 0.0]), 0.3, 0.7)
        backend.report_absolute_hint(1.15, np.array([8.0, 1.0, 0.0]), 0.2, 0.9)

        corr = None
        deadline = time.time() + 1.0
        while time.time() < deadline:
            corr = backend.poll_correction(t_now=1.30)
            if corr is not None:
                break
            time.sleep(0.01)

        assert corr is not None
        assert np.isfinite(float(np.linalg.norm(corr.dp_enu[:2])))
        assert float(np.linalg.norm(corr.dp_enu[:2])) <= 60.0 + 1e-6
        assert abs(float(corr.dyaw_deg)) <= 8.0 + 1e-6
    finally:
        backend.stop()
