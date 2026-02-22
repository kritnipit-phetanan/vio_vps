#!/usr/bin/env python3
"""Phase-by-phase accuracy tuning orchestrator with stage-aware lock profiles.

Implements one-knob-at-a-time protocol:
- fixed order phases per stage-set (stageA/stageE/stageF)
- each phase mutates only that phase keys
- rollback immediately when lock profile fails or err3d_mean regresses beyond threshold
- persist best run + baseline pointer automatically
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML required: {exc}")


DEFAULT_PHASE_SEQUENCE_STAGE_A = ["A1", "A2", "A3", "A4", "A5"]
DEFAULT_PHASE_SEQUENCE_STAGE_E = ["E1", "E2", "E3", "E4", "E5"]
DEFAULT_PHASE_SEQUENCE_STAGE_F = ["F1", "F2", "F3", "F4", "F5"]
DEFAULT_PHASE_SEQUENCE_STAGE_G = ["G5_1", "G5_2", "G6_1", "G7_1", "G7_2"]
DEFAULT_PHASE_SEQUENCE_STAGE_G_CONT = ["G7_1_HALF", "G5_3"]
DEFAULT_PHASE_SEQUENCE_STAGE_G_MAG = ["G5_MAG_SAFE", "G5_MAG_COND"]
DEFAULT_PHASE_SEQUENCE = list(DEFAULT_PHASE_SEQUENCE_STAGE_A)

# One-knob assignments for Stage-A pre-backend tuning (Phase4-base roadmap).
PHASE_ASSIGNMENTS: dict[str, dict[str, Any]] = {
    "A1": {
        "vps.apply_failsoft_allow_warning": True,
        "vps.apply_failsoft_r_mult": 2.2,
        "vps.apply_failsoft_large_offset_confirm_hits": 2,
        "vps.apply_failsoft_max_dir_change_deg": 55.0,
        "vps.apply_failsoft_max_offset_m": 130.0,
    },
    "A2": {
        "magnetometer.warning_weak_yaw.conditioning_guard_warn_pcond": 8.0e11,
        "magnetometer.warning_weak_yaw.conditioning_guard_warn_pmax": 8.0e6,
        "magnetometer.warning_weak_yaw.warning_extra_r_mult": 2.0,
        "magnetometer.warning_weak_yaw.degraded_extra_r_mult": 2.8,
    },
    "A3": {
        "vio.nadir_xy_only_velocity": True,
        "vio_vel.speed_r_inflate_breakpoints_m_s": [25.0, 40.0, 55.0],
        "vio_vel.speed_r_inflate_values": [1.5, 2.5, 4.0],
        "vio_vel.max_delta_v_xy_per_update_m_s": 2.0,
        "vio_vel.min_flow_px_high_speed": 0.8,
    },
    "A4": {
        "vio.msckf.reproj_state_aware.warning_scale": 1.20,
        "vio.msckf.reproj_state_aware.degraded_scale": 1.35,
        "vio.msckf.reproj_state_aware.mid_gate_mult": 1.15,
    },
    "A5": {
        # Absolute-source quality-first tuning (VPS retrieval/geometry policy)
        # Keep candidate budget bounded for near-RT; tighten content/texture gates.
        "vps.max_candidates": 5,
        "vps.global_max_candidates": 10,
        "vps.min_content_ratio": 0.24,
        "vps.min_texture_std": 10.0,
        "vps.relocalization.fail_streak_trigger": 4,
        "vps.relocalization.xy_sigma_trigger_m": 25.0,
        "vps.relocalization.max_centers": 8,
    },
    "E1": {
        # Phase-E base marker (no knob change by design).
    },
    "E2": {
        # Runtime knob #1: VPS candidate budget
        "vps.global_max_candidates": 20,
        "vps.max_candidates": 5,
        "vps.relocalization.max_centers": 6,
    },
    "E3": {
        # Runtime knob #2: VPS cadence
        "vps.min_update_interval": 0.9,
    },
    "E4": {
        # Runtime knob #3: backend cadence
        "backend.optimize_rate_hz": 0.5,
        "backend.poll_interval_sec": 1.5,
    },
    "E5": {
        # Runtime knob #4: log throttling
        "logging.runtime_verbosity": "release",
        "logging.runtime_log_interval_sec": 1.0,
        "vps.runtime_verbosity": "release",
        "vps.runtime_log_interval_sec": 1.0,
    },
    "F1": {
        # LightGlue containment for near-RT recovery: force ORB-only matcher.
        "vps.matcher_mode": "orb",
    },
    "F2": {
        # VPS hard runtime budget: cap total candidates and wall-time per frame.
        "vps.max_total_candidates": 16,
        "vps.max_frame_time_ms_local": 250.0,
        "vps.max_frame_time_ms_global": 700.0,
    },
    "F3": {
        # Recover VIO_VEL aiding coverage (relax over-tight high-speed guard).
        "vio_vel.max_delta_v_xy_per_update_m_s": 5.0,
        "vio_vel.min_flow_px_high_speed": 0.50,
        "vio_vel.speed_r_inflate_values": [1.0, 1.5, 2.0],
    },
    "F4": {
        # MAG usefulness recovery: postpone hard conditioning skip to extreme cases.
        "magnetometer.warning_weak_yaw.conditioning_guard_hard_pcond": 2.0e12,
        "magnetometer.warning_weak_yaw.conditioning_guard_hard_pmax": 2.0e7,
    },
    "F5": {
        # Optional bounded LightGlue rescue re-enable after F1-F4 stability.
        "vps.matcher_mode": "orb_lightglue_rescue",
    },
    "G5_1": {
        # Loop temporal tighten for heading stability (normal path + stricter cap).
        "loop_closure.temporal_apply.confirm_hits_normal": 2,
        "loop_closure.temporal_apply.cooldown_sec_normal": 2.4,
        "loop_closure.fail_soft.max_abs_yaw_corr_deg": 2.2,
        "loop_closure.quality_gate.yaw_residual_bound_deg": 18.0,
    },
    "G5_2": {
        # MAG soft-inflate stabilization: delay hard-skip, tighten per-update dyaw.
        "magnetometer.warning_weak_yaw.conditioning_guard_hard_pcond": 7.0e10,
        "magnetometer.warning_weak_yaw.conditioning_guard_hard_pmax": 2.2e6,
        "magnetometer.warning_weak_yaw.warning_max_dyaw_deg": 0.8,
        "magnetometer.warning_weak_yaw.warning_extra_r_mult": 3.5,
    },
    "G6_1": {
        # MSCKF reprojection retune after heading is stable.
        "vio.msckf.avg_reproj_gate_factor": 2.4,
        "vio.msckf.reproj_state_aware.mid_gate_mult": 1.20,
    },
    "G7_1": {
        # VPS coverage/backoff duty-cycle and bounded candidate budget.
        "vps.matcher_mode": "orb",
        "vps.max_total_candidates": 16,
        "vps.max_frame_time_ms_local": 250.0,
        "vps.max_frame_time_ms_global": 700.0,
        "vps.relocalization.global_backoff_probe_every_attempts": 8,
        "vps.relocalization.global_backoff_probe_min_interval_sec": 1.5,
        "vps.relocalization.global_probe_on_no_coverage": True,
        "vps.relocalization.global_probe_no_coverage_streak": 2,
        "vps.relocalization.no_coverage_recovery_streak": 2,
        "vps.relocalization.no_coverage_use_last_success": True,
        "vps.relocalization.no_coverage_use_last_coverage": True,
        "vps.relocalization.no_coverage_recovery_radius_m": [20.0, 50.0],
        "vps.relocalization.no_coverage_recovery_samples": 5,
        "vps.relocalization.no_coverage_recovery_max_centers": 4,
    },
    "G7_2": {
        # Near-RT refinement once coverage is recovered.
        "vps.min_update_interval": 0.6,
        "backend.optimize_rate_hz": 0.8,
        "backend.poll_interval_sec": 1.2,
    },
    "G7_1_HALF": {
        # Half-step runtime/backoff from G7_1: keep runtime gains without over-pruning coverage.
        "vps.matcher_mode": "orb",
        "vps.max_total_candidates": 24,
        "vps.max_frame_time_ms_local": 320.0,
        "vps.max_frame_time_ms_global": 900.0,
        "vps.relocalization.global_backoff_probe_every_attempts": 10,
        "vps.relocalization.global_backoff_probe_min_interval_sec": 1.8,
    },
    "G5_3": {
        # MAG-only follow-up: reduce hard-extreme skips while keeping yaw clamp conservative.
        "magnetometer.warning_weak_yaw.conditioning_guard_hard_pcond": 9.0e10,
        "magnetometer.warning_weak_yaw.conditioning_guard_hard_pmax": 2.8e6,
        "magnetometer.warning_weak_yaw.conditioning_guard_extreme_pcond": 1.2e12,
        "magnetometer.warning_weak_yaw.conditioning_guard_extreme_pmax": 1.4e7,
    },
    "G5_MAG_SAFE": {
        # G5 follow-up (MAG safety only, conservative):
        # keep fail-soft style but reduce yaw injection per update.
        "magnetometer.warning_weak_yaw.warning_extra_r_mult": 3.8,
        "magnetometer.warning_weak_yaw.warning_max_dyaw_deg": 0.7,
    },
    "G5_MAG_COND": {
        # G5 follow-up (MAG conditioning only, micro-step):
        # slight relaxation from G5_2 to avoid hard-extreme flood but keep guard tight.
        "magnetometer.warning_weak_yaw.conditioning_guard_hard_pcond": 7.4e10,
        "magnetometer.warning_weak_yaw.conditioning_guard_hard_pmax": 2.3e6,
        "magnetometer.warning_weak_yaw.conditioning_guard_extreme_pcond": 1.45e11,
        "magnetometer.warning_weak_yaw.conditioning_guard_extreme_pmax": 2.7e6,
    },
}


def list_run_dirs(repo_dir: Path) -> list[Path]:
    patterns = [
        "outputs/benchmark_modular_*",
        "benchmark_modular_*",
        "benchmark_modular_*/preintegration",
    ]
    runs: list[Path] = []
    for pat in patterns:
        runs.extend(repo_dir.glob(pat))
    uniq: dict[Path, Path] = {}
    for p in runs:
        if p.is_dir():
            uniq[p.resolve()] = p.resolve()
    return sorted(
        uniq.values(),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
    )


def resolve_path(repo_dir: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = (repo_dir / p).resolve()
    return p


def resolve_baseline(repo_dir: Path, explicit: str, baseline_file: Path) -> Path | None:
    if explicit:
        p = resolve_path(repo_dir, explicit)
        if p.is_dir():
            return p
        return None

    if baseline_file.is_file():
        val = baseline_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        if val:
            p = resolve_path(repo_dir, val[0].strip())
            if p.is_dir():
                return p

    runs = list_run_dirs(repo_dir)
    return runs[-1] if runs else None


def set_nested(cfg: dict[str, Any], key: str, value: Any) -> None:
    cur: dict[str, Any] = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


_MISSING = object()
_NO_DEFAULT = object()


def get_nested(cfg: dict[str, Any], key: str, default: Any = _NO_DEFAULT) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            if default is _NO_DEFAULT:
                raise KeyError(key)
            return default
        cur = cur[part]
    return cur


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(values_equal(x, y) for x, y in zip(a, b))
    try:
        af = float(a)
        bf = float(b)
        if np.isfinite(af) and np.isfinite(bf):
            return bool(np.isclose(af, bf, rtol=1e-12, atol=1e-12))
    except Exception:
        pass
    return a == b


def phase_changed_keys(cfg: dict[str, Any], phase: str) -> list[str]:
    changed: list[str] = []
    for key, target in PHASE_ASSIGNMENTS.get(phase, {}).items():
        cur_val = get_nested(cfg, key, default=_MISSING)
        if cur_val is _MISSING or not values_equal(cur_val, target):
            changed.append(key)
    return changed


def apply_phase(cfg: dict[str, Any], phase: str) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    assignments = PHASE_ASSIGNMENTS.get(phase, {})
    for key, value in assignments.items():
        set_nested(out, key, value)
    return out


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def run_benchmark(
    repo_dir: Path,
    config_path: Path,
    baseline_run: Path | None,
    run_mode: str,
    save_debug_data: int,
    save_keyframe_images: int,
    lock_profile: str,
) -> tuple[int, Path | None]:
    before = {p.resolve() for p in list_run_dirs(repo_dir)}
    env = os.environ.copy()
    env["CONFIG"] = str(config_path)
    env["RUN_MODE"] = run_mode
    env["SAVE_DEBUG_DATA"] = str(int(save_debug_data))
    env["SAVE_KEYFRAME_IMAGES"] = str(int(save_keyframe_images))
    env["LOCK_PROFILE"] = str(lock_profile)
    if baseline_run is not None:
        env["BASELINE_RUN"] = str(baseline_run)

    cmd = ["bash", "scripts/benchmark_modular.sh"]
    proc = subprocess.Popen(
        cmd,
        cwd=repo_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    captured: list[str] = []
    for line in proc.stdout:
        print(line, end="")
        captured.append(line)
    ret = proc.wait()

    after = {p.resolve() for p in list_run_dirs(repo_dir)}
    created = sorted(after - before, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    if created:
        return ret, created[-1]

    m = re.search(r"Output directory:\s*([^\s]+)", "".join(captured))
    if m:
        p = resolve_path(repo_dir, m.group(1))
        if p.is_dir():
            return ret, p

    runs = list_run_dirs(repo_dir)
    return ret, (runs[-1] if runs else None)


def read_err3d_mean(run_dir: Path | None) -> float:
    if run_dir is None:
        return float("nan")
    p = run_dir / "accuracy_first_summary.csv"
    if not p.is_file():
        return float("nan")
    try:
        df = pd.read_csv(p)
        if len(df) == 0:
            return float("nan")
        return float(pd.to_numeric(df.iloc[-1].get("err3d_mean"), errors="coerce"))
    except Exception:
        return float("nan")


def read_health_row(run_dir: Path | None) -> pd.Series | None:
    if run_dir is None:
        return None
    p = run_dir / "benchmark_health_summary.csv"
    if not p.is_file():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if len(df) == 0:
        return None
    return df.iloc[-1]


def read_heading_final_abs(run_dir: Path | None) -> float:
    if run_dir is None:
        return float("nan")
    p = run_dir / "accuracy_first_summary.csv"
    if not p.is_file():
        return float("nan")
    try:
        df = pd.read_csv(p)
        if len(df) == 0:
            return float("nan")
        return float(pd.to_numeric(df.iloc[-1].get("heading_final_abs_deg"), errors="coerce"))
    except Exception:
        return float("nan")


def parse_vps_used(run_log: Path) -> float:
    if not run_log.is_file():
        return float("nan")
    val = float("nan")
    for line in run_log.read_text(errors="ignore").splitlines():
        m = re.search(r"VPS used:\s*(\d+)", line)
        if m:
            val = float(m.group(1))
    return val


def overflow_hits(run_log: Path) -> list[str]:
    if not run_log.is_file():
        return []
    patterns = ("overflow", "non-finite", "contains inf/nan", "runtimewarning")
    hits: list[str] = []
    for line in run_log.read_text(errors="ignore").splitlines():
        low = line.strip().lower()
        if any(p in low for p in patterns):
            if "no overflow" in low:
                continue
            hits.append(line.strip())
    return hits


def _to_float(row: pd.Series, key: str) -> float:
    try:
        return float(pd.to_numeric(row.get(key), errors="coerce"))
    except Exception:
        return float("nan")


def evaluate_lock_profile(run_dir: Path | None, profile_name: str) -> tuple[bool, dict[str, Any]]:
    row = read_health_row(run_dir)
    if row is None:
        return False, {
            "reason": "missing benchmark_health_summary.csv",
            "checks": [],
            "failed_items": ["missing benchmark_health_summary.csv"],
        }

    run_log = (run_dir / "run.log") if run_dir else Path("")
    over = overflow_hits(run_log)

    mag_cholfail = _to_float(row, "mag_cholfail_rate")
    cov_large = _to_float(row, "cov_large_rate")
    pmax_max = _to_float(row, "pmax_max")
    vps_used = parse_vps_used(run_log)
    heading_final_abs = read_heading_final_abs(run_dir)
    vps_jump_reject_count = _to_float(row, "vps_jump_reject_count")
    vps_attempt_count = _to_float(row, "vps_attempt_count")
    backend_stale_drop_count = _to_float(row, "backend_stale_drop_count")
    backend_poll_count = _to_float(row, "backend_poll_count")

    jump_ratio = float("nan")
    if np.isfinite(vps_jump_reject_count) and np.isfinite(vps_attempt_count) and vps_attempt_count > 0:
        jump_ratio = float(vps_jump_reject_count / vps_attempt_count)

    stale_ratio = float("nan")
    if np.isfinite(backend_stale_drop_count) and np.isfinite(backend_poll_count) and backend_poll_count > 0:
        stale_ratio = float(backend_stale_drop_count / backend_poll_count)

    checks: list[tuple[str, bool, str]] = []
    checks.append((
        "cov_large_rate == 0",
        bool(np.isfinite(cov_large) and abs(cov_large) <= 1e-12),
        f"value={cov_large:.6f}" if np.isfinite(cov_large) else "value=nan",
    ))
    checks.append((
        "pmax_max <= 1e6",
        bool(np.isfinite(pmax_max) and pmax_max <= 1.0e6),
        f"value={pmax_max:.3e}" if np.isfinite(pmax_max) else "value=nan",
    ))
    checks.append((
        "no overflow/non-finite flood",
        bool(len(over) == 0),
        f"hits={len(over)}",
    ))

    profile = str(profile_name).strip().lower()
    if profile == "pre_backend":
        checks.extend([
            (
                "mag_cholfail_rate <= 0.10",
                bool(np.isfinite(mag_cholfail) and mag_cholfail <= 0.10),
                f"value={mag_cholfail:.6f}" if np.isfinite(mag_cholfail) else "value=nan",
            ),
            (
                "VPS used >= 10",
                bool(np.isfinite(vps_used) and vps_used >= 10.0),
                f"value={vps_used:.0f}" if np.isfinite(vps_used) else "value=nan",
            ),
            (
                "vps_jump_reject_count/vps_attempts <= 0.5",
                bool(np.isfinite(jump_ratio) and jump_ratio <= 0.5),
                f"value={jump_ratio:.4f}" if np.isfinite(jump_ratio) else "value=nan",
            ),
            (
                "heading_final_abs_deg <= 15",
                bool(np.isfinite(heading_final_abs) and heading_final_abs <= 15.0),
                f"value={heading_final_abs:.3f}" if np.isfinite(heading_final_abs) else "value=nan",
            ),
        ])
    elif profile in ("backend", "near_rt_backend"):
        checks.extend([
            (
                "mag_cholfail_rate <= 0.08",
                bool(np.isfinite(mag_cholfail) and mag_cholfail <= 0.08),
                f"value={mag_cholfail:.6f}" if np.isfinite(mag_cholfail) else "value=nan",
            ),
            (
                "VPS used >= 20",
                bool(np.isfinite(vps_used) and vps_used >= 20.0),
                f"value={vps_used:.0f}" if np.isfinite(vps_used) else "value=nan",
            ),
            (
                "backend_stale_drop_count/backend_poll_count <= 0.2",
                bool(np.isfinite(stale_ratio) and stale_ratio <= 0.2),
                f"value={stale_ratio:.4f}" if np.isfinite(stale_ratio) else "value=nan",
            ),
            (
                "heading_final_abs_deg <= 10",
                bool(np.isfinite(heading_final_abs) and heading_final_abs <= 10.0),
                f"value={heading_final_abs:.3f}" if np.isfinite(heading_final_abs) else "value=nan",
            ),
        ])
        if profile == "near_rt_backend":
            rtf_proc_sim = _to_float(row, "rtf_proc_sim")
            checks.append(
                (
                    "rtf_proc_sim <= 1.2",
                    bool(np.isfinite(rtf_proc_sim) and rtf_proc_sim <= 1.2),
                    f"value={rtf_proc_sim:.6f}" if np.isfinite(rtf_proc_sim) else "value=nan",
                )
            )
    else:
        checks.append(("lock profile recognized", False, f"profile={profile}"))

    failed_items = [name for name, ok, _ in checks if not ok]
    ok = len(failed_items) == 0
    detail = {
        "checks": checks,
        "failed_items": failed_items,
        "profile_name": profile,
        "mag_cholfail_rate": mag_cholfail,
        "cov_large_rate": cov_large,
        "pmax_max": pmax_max,
        "vps_used": vps_used,
        "heading_final_abs_deg": heading_final_abs,
        "vps_jump_reject_ratio": jump_ratio,
        "backend_stale_drop_ratio": stale_ratio,
        "overflow_hits": len(over),
        "rtf_proc_sim": _to_float(row, "rtf_proc_sim"),
    }
    return ok, detail


def classify_rtf_pass_tier(rtf_proc_sim: float) -> str:
    """Classify runtime tier for near-RT tracking dashboards."""
    if not np.isfinite(rtf_proc_sim):
        return "nan"
    if rtf_proc_sim <= 1.2:
        return "<=1.2"
    if rtf_proc_sim <= 2.0:
        return "<=2.0"
    if rtf_proc_sim <= 3.0:
        return "<=3.0"
    return ">3.0"


def save_phase_results(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_phase_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No phase results.")
        return
    print("\n=== Phase Tuning Summary ===")
    print("phase  profile      status    err3d_mean(m)  delta_vs_baseline(%)  locks  rtf_tier  run_dir")
    for r in rows:
        d = r.get("err3d_delta_pct", np.nan)
        d_s = f"{d:+.2f}" if np.isfinite(d) else "n/a"
        print(
            f"{str(r['phase']):>5s}  {str(r.get('profile_name', '')):<11s}  {str(r['status']):<8s}  "
            f"{float(r.get('err3d_mean', np.nan)):>13.3f}  {d_s:>21s}  "
            f"{str(r.get('locks_ok', np.nan)):<5s}  {str(r.get('rtf_pass_tier', 'nan')):<8s}  {r.get('run_dir', '')}"
        )


def snapshot_path_for_profile(repo_dir: Path, profile_name: str) -> Path:
    profile = str(profile_name).lower()
    if profile == "pre_backend":
        return (repo_dir / "configs" / "config_bell412_dataset3_backend_stage1.yaml").resolve()
    if profile in ("backend", "near_rt_backend"):
        return (repo_dir / "configs" / "config_bell412_dataset3_backend_stage2.yaml").resolve()
    return (repo_dir / "configs" / "config_bell412_dataset3_accuracy_stage2.yaml").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrate Bell412 one-knob phase tuning.")
    parser.add_argument("--config", default="configs/config_bell412_dataset3.yaml")
    parser.add_argument("--base_phase_config", default="", help="Optional config used as phase base (read-only source)")
    parser.add_argument("--run_mode", default="auto")
    parser.add_argument("--save_debug_data", type=int, default=0, choices=[0, 1])
    parser.add_argument("--save_keyframe_images", type=int, default=0, choices=[0, 1])
    parser.add_argument("--baseline_run", default="")
    parser.add_argument("--lock_profile", default="pre_backend", choices=["pre_backend", "backend", "near_rt_backend"])
    parser.add_argument("--max_regress_pct", type=float, default=15.0)
    parser.add_argument("--apply_best_config", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--phase_set",
        default="stageA",
        choices=["stageA", "stageE", "stageF", "stageG", "stageG_cont", "stageG_mag", "custom"],
    )
    parser.add_argument("--phases", default="")
    parser.add_argument("--dry_run", action="store_true", help="Plan and emit phase table without running benchmarks")
    parser.add_argument("--force_run_noop", action="store_true", help="Run phase even when no keys would change")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    config_path = resolve_path(repo_dir, args.config)
    if not config_path.is_file():
        print(f"❌ config not found: {config_path}")
        return 2

    base_cfg_path = resolve_path(repo_dir, args.base_phase_config) if args.base_phase_config else config_path
    if not base_cfg_path.is_file():
        print(f"❌ base_phase_config not found: {base_cfg_path}")
        return 2

    baseline_file = (repo_dir / "scripts" / "baseline_run.txt").resolve()
    baseline_run = resolve_baseline(repo_dir, args.baseline_run, baseline_file)
    baseline_err = read_err3d_mean(baseline_run)
    print(f"Baseline run: {baseline_run if baseline_run else 'None'}")
    print(f"Baseline err3d_mean: {baseline_err:.3f} m" if np.isfinite(baseline_err) else "Baseline err3d_mean: nan")
    print(f"Lock profile: {args.lock_profile}")
    print(f"Phase set: {args.phase_set}")
    print(f"Phase base config: {base_cfg_path}")

    with base_cfg_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    if args.phase_set == "stageA":
        default_phases = DEFAULT_PHASE_SEQUENCE_STAGE_A
    elif args.phase_set == "stageE":
        default_phases = DEFAULT_PHASE_SEQUENCE_STAGE_E
    elif args.phase_set == "stageF":
        default_phases = DEFAULT_PHASE_SEQUENCE_STAGE_F
    elif args.phase_set == "stageG":
        default_phases = DEFAULT_PHASE_SEQUENCE_STAGE_G
    elif args.phase_set == "stageG_cont":
        default_phases = DEFAULT_PHASE_SEQUENCE_STAGE_G_CONT
    elif args.phase_set == "stageG_mag":
        default_phases = DEFAULT_PHASE_SEQUENCE_STAGE_G_MAG
    else:
        default_phases = DEFAULT_PHASE_SEQUENCE
    phases_raw = args.phases if str(args.phases).strip() else ",".join(default_phases)
    phases = [p.strip() for p in phases_raw.split(",") if p.strip()]
    for p in phases:
        if p not in PHASE_ASSIGNMENTS:
            print(f"❌ Unknown phase: {p}")
            return 2

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root_raw = os.environ.get("OUTPUT_ROOT", "outputs")
    output_root = Path(output_root_raw)
    if not output_root.is_absolute():
        output_root = (repo_dir / output_root).resolve()
    orchestration_dir = output_root / f"benchmark_phase_tuning_{ts}"
    orchestration_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = orchestration_dir / "phase_results.csv"

    current_cfg = copy.deepcopy(base_cfg)
    cfg_by_run: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for phase in phases:
        print(f"\n================ Phase {phase} ================")
        assignments = PHASE_ASSIGNMENTS.get(phase, {})
        for k, v in assignments.items():
            print(f"  set {k} = {v}")
        changed_keys = phase_changed_keys(current_cfg, phase)
        if changed_keys:
            print(f"  changed keys ({len(changed_keys)}):")
            for ck in changed_keys:
                print(f"    - {ck}")
        else:
            print("  changed keys (0): phase is already at target values")

        if not changed_keys and not args.force_run_noop:
            rows.append(
                {
                    "phase": phase,
                    "profile_name": args.lock_profile,
                    "status": "SKIP",
                    "reason": "no_change",
                    "lock_failed_items": "",
                    "run_dir": "",
                    "returncode": 0,
                    "err3d_mean": baseline_err,
                    "err3d_baseline": baseline_err,
                    "err3d_delta_pct": 0.0,
                    "locks_ok": np.nan,
                    "vps_used": np.nan,
                    "heading_final_abs_deg": np.nan,
                    "mag_cholfail_rate": np.nan,
                    "cov_large_rate": np.nan,
                    "pmax_max": np.nan,
                    "overflow_hits": np.nan,
                    "rtf_proc_sim": np.nan,
                    "rtf_pass_tier": "nan",
                    "changed_key_count": 0,
                    "changed_keys": "",
                }
            )
            continue

        candidate_cfg = apply_phase(current_cfg, phase)

        if args.dry_run:
            rows.append(
                {
                    "phase": phase,
                    "profile_name": args.lock_profile,
                    "status": "DRYRUN",
                    "reason": "dry_run",
                    "lock_failed_items": "",
                    "run_dir": "",
                    "returncode": 0,
                    "err3d_mean": np.nan,
                    "err3d_baseline": baseline_err,
                    "err3d_delta_pct": np.nan,
                    "locks_ok": np.nan,
                    "vps_used": np.nan,
                    "heading_final_abs_deg": np.nan,
                    "mag_cholfail_rate": np.nan,
                    "cov_large_rate": np.nan,
                    "pmax_max": np.nan,
                    "overflow_hits": np.nan,
                    "rtf_proc_sim": np.nan,
                    "rtf_pass_tier": "nan",
                    "changed_key_count": len(changed_keys),
                    "changed_keys": "|".join(changed_keys),
                }
            )
            current_cfg = candidate_cfg
            continue

        temp_cfg = orchestration_dir / f"config_phase_{phase}.yaml"
        write_yaml(temp_cfg, candidate_cfg)

        ret, run_dir = run_benchmark(
            repo_dir=repo_dir,
            config_path=temp_cfg,
            baseline_run=baseline_run,
            run_mode=args.run_mode,
            save_debug_data=args.save_debug_data,
            save_keyframe_images=args.save_keyframe_images,
            lock_profile=args.lock_profile,
        )

        if run_dir is None:
            rows.append(
                {
                    "phase": phase,
                    "profile_name": args.lock_profile,
                    "status": "FAIL",
                    "reason": "no_run_dir_detected",
                    "lock_failed_items": "",
                    "run_dir": "",
                    "returncode": ret,
                    "err3d_mean": np.nan,
                    "err3d_baseline": baseline_err,
                    "err3d_delta_pct": np.nan,
                    "locks_ok": False,
                    "vps_used": np.nan,
                    "heading_final_abs_deg": np.nan,
                    "mag_cholfail_rate": np.nan,
                    "cov_large_rate": np.nan,
                    "pmax_max": np.nan,
                    "overflow_hits": np.nan,
                    "rtf_proc_sim": np.nan,
                    "rtf_pass_tier": "nan",
                    "changed_key_count": len(changed_keys),
                    "changed_keys": "|".join(changed_keys),
                }
            )
            print(f"[Phase {phase}] FAIL: no run directory detected")
            continue

        run_cfg_snapshot = run_dir / "phase_config_snapshot.yaml"
        write_yaml(run_cfg_snapshot, candidate_cfg)
        cfg_by_run[str(run_dir.resolve())] = copy.deepcopy(candidate_cfg)

        err_cur = read_err3d_mean(run_dir)
        delta_pct = np.nan
        regress_fail = False
        if np.isfinite(err_cur) and np.isfinite(baseline_err) and abs(baseline_err) > 1e-12:
            delta_pct = 100.0 * (err_cur - baseline_err) / abs(baseline_err)
            regress_fail = bool(delta_pct > args.max_regress_pct)

        locks_ok, lock_detail = evaluate_lock_profile(run_dir, args.lock_profile)
        phase_ok = bool(ret == 0 and locks_ok and not regress_fail and np.isfinite(err_cur))
        reason = []
        if ret != 0:
            reason.append(f"run_failed(rc={ret})")
        if regress_fail:
            reason.append(f"err_regress={delta_pct:.2f}%>{args.max_regress_pct:.2f}%")
        if not locks_ok:
            reason.append("lock_fail")
        if not np.isfinite(err_cur):
            reason.append("missing_err3d")
        reason_text = "|".join(reason) if reason else "pass"

        row = {
            "phase": phase,
            "profile_name": args.lock_profile,
            "status": "PASS" if phase_ok else "FAIL",
            "reason": reason_text,
            "lock_failed_items": "|".join(lock_detail.get("failed_items", [])),
            "run_dir": str(run_dir),
            "returncode": ret,
            "err3d_mean": err_cur,
            "err3d_baseline": baseline_err,
            "err3d_delta_pct": delta_pct,
            "locks_ok": locks_ok,
            "vps_used": lock_detail.get("vps_used", np.nan),
            "heading_final_abs_deg": lock_detail.get("heading_final_abs_deg", np.nan),
            "mag_cholfail_rate": lock_detail.get("mag_cholfail_rate", np.nan),
            "cov_large_rate": lock_detail.get("cov_large_rate", np.nan),
            "pmax_max": lock_detail.get("pmax_max", np.nan),
            "overflow_hits": lock_detail.get("overflow_hits", np.nan),
            "rtf_proc_sim": lock_detail.get("rtf_proc_sim", np.nan),
            "rtf_pass_tier": classify_rtf_pass_tier(float(lock_detail.get("rtf_proc_sim", np.nan))),
            "changed_key_count": len(changed_keys),
            "changed_keys": "|".join(changed_keys),
        }
        rows.append(row)

        if phase_ok:
            current_cfg = candidate_cfg
            baseline_run = run_dir
            baseline_err = err_cur
            print(f"[Phase {phase}] PASS -> new baseline: {run_dir} (err3d_mean={err_cur:.3f} m)")
        else:
            print(f"[Phase {phase}] ROLLBACK ({reason_text})")

        save_phase_results(summary_csv, rows)

    if args.dry_run:
        save_phase_results(summary_csv, rows)
        print_phase_table(rows)
        print(f"\nSaved summary CSV: {summary_csv}")
        return 0

    accepted = [r for r in rows if r["status"] == "PASS" and np.isfinite(r["err3d_mean"])]
    best_run: Path | None = None
    best_row: dict[str, Any] | None = None
    if accepted:
        best_row = min(accepted, key=lambda r: float(r["err3d_mean"]))
        best_run = Path(str(best_row["run_dir"])).resolve()

    snapshot_path = snapshot_path_for_profile(repo_dir, args.lock_profile)
    if best_run is not None:
        cfg_src = best_run / "phase_config_snapshot.yaml"
        if cfg_src.is_file():
            shutil.copy2(cfg_src, snapshot_path)
            print(f"Best config snapshot: {snapshot_path} (from {best_run})")
            if int(args.apply_best_config) == 1:
                shutil.copy2(cfg_src, config_path)
                print(f"Applied best config to: {config_path}")
        else:
            write_yaml(snapshot_path, current_cfg)
            print(f"Best snapshot fallback saved: {snapshot_path}")
            if int(args.apply_best_config) == 1:
                write_yaml(config_path, current_cfg)
                print(f"Applied fallback best config to: {config_path}")

        should_update_baseline = True
        if str(args.lock_profile).lower() == "near_rt_backend":
            best_rtf = float(best_row.get("rtf_proc_sim", np.nan)) if isinstance(best_row, dict) else np.nan
            should_update_baseline = bool(np.isfinite(best_rtf) and best_rtf <= 1.2)

        if should_update_baseline:
            baseline_file.write_text(str(best_run) + "\n", encoding="utf-8")
            print(f"Updated baseline pointer: {baseline_file} -> {best_run}")
        else:
            candidate_cfg_path = (repo_dir / "configs" / "config_bell412_dataset3_backend_stage2_candidate.yaml").resolve()
            cfg_candidate = cfg_by_run.get(str(best_run.resolve()), current_cfg)
            write_yaml(candidate_cfg_path, cfg_candidate)
            print(f"Near-RT candidate saved (baseline unchanged): {candidate_cfg_path}")
    elif str(args.lock_profile).lower() == "near_rt_backend":
        finite_rows = [r for r in rows if np.isfinite(float(r.get("err3d_mean", np.nan))) and str(r.get("run_dir", "")).strip()]
        if finite_rows:
            best_fail = min(finite_rows, key=lambda r: float(r["err3d_mean"]))
            run_str = str(best_fail.get("run_dir", "")).strip()
            cfg_candidate = current_cfg
            if run_str:
                cfg_candidate = cfg_by_run.get(str(Path(run_str).resolve()), current_cfg)
            candidate_cfg_path = (repo_dir / "configs" / "config_bell412_dataset3_backend_stage2_candidate.yaml").resolve()
            write_yaml(candidate_cfg_path, cfg_candidate)
            print(f"Near-RT candidate saved from best finite run (baseline unchanged): {candidate_cfg_path}")

    save_phase_results(summary_csv, rows)
    print_phase_table(rows)
    print(f"\nSaved summary CSV: {summary_csv}")

    if accepted:
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
