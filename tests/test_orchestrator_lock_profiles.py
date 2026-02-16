import importlib.util
from pathlib import Path

import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[1]
ORCH_PATH = REPO_DIR / "scripts" / "orchestrate_phase_tuning.py"
SPEC = importlib.util.spec_from_file_location("orch_tune", ORCH_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def _make_run_dir(tmp_path: Path, vps_used: int) -> Path:
    run_dir = tmp_path / "bench_run" / "preintegration"
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "run_id": "bench_run",
                "projection_count": 10,
                "first_projection_time": 100.0,
                "pcond_max": 1.0e9,
                "pmax_max": 1.0e5,
                "cov_large_rate": 0.0,
                "pos_rmse": 100.0,
                "final_pos_err": 120.0,
                "final_alt_err": 5.0,
                "frames_inlier_nonzero_ratio": 0.5,
                "vio_vel_accept_ratio_vs_cam": 0.4,
                "mag_cholfail_rate": 0.05,
                "loop_applied_rate": 0.1,
                "speed_max_m_s": 30.0,
                "speed_p99_m_s": 20.0,
                "loop_corr_count": 3,
                "loop_abs_yaw_corr_sum_deg": 4.0,
                "vps_soft_accept_count": 5,
                "vps_soft_reject_count": 2,
                "mag_accept_rate": 0.6,
                "vps_jump_reject_count": 2,
                "vps_temporal_confirm_count": 3,
                "abs_corr_apply_count": 7,
                "abs_corr_soft_count": 4,
                "backend_apply_count": 0,
                "backend_stale_drop_count": 0,
                "backend_poll_count": 1,
                "vps_attempt_count": 10,
                "rtf_proc_sim": 1.0,
            }
        ]
    ).to_csv(run_dir / "benchmark_health_summary.csv", index=False)

    pd.DataFrame([{"heading_final_abs_deg": 5.0}]).to_csv(
        run_dir / "accuracy_first_summary.csv", index=False
    )

    with (run_dir / "run.log").open("w", encoding="utf-8") as f:
        f.write(f"VPS used: {vps_used}\n")

    return run_dir


def test_pre_backend_profile_passes_with_stage_a_thresholds(tmp_path):
    run_dir = _make_run_dir(tmp_path, vps_used=12)
    ok, detail = MOD.evaluate_lock_profile(run_dir, "pre_backend")
    assert ok
    assert detail["profile_name"] == "pre_backend"


def test_backend_profile_fails_when_vps_used_too_low(tmp_path):
    run_dir = _make_run_dir(tmp_path, vps_used=12)
    ok, detail = MOD.evaluate_lock_profile(run_dir, "backend")
    assert not ok
    assert "VPS used >= 20" in detail.get("failed_items", [])
