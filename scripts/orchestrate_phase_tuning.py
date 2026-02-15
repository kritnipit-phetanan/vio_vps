#!/usr/bin/env python3
"""Phase-by-phase accuracy tuning orchestrator with auto rollback.

Implements One-Knob-at-a-Time protocol:
  - fixed order: Phase 1 -> 2 -> 3 -> 4 -> 5
  - each phase mutates only that phase keys
  - rollback immediately when hard locks fail or err3d_mean regresses > threshold
  - persist best run and baseline pointer automatically
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


PHASE_SEQUENCE = ["1", "2", "3", "4", "5"]

PHASE_ASSIGNMENTS: dict[str, dict[str, Any]] = {
    "1": {
        "loop_closure.fail_soft.max_abs_yaw_corr_deg": 2.5,
        "loop_closure.quality_gate.cooldown_sec": 4.0,
    },
    "2": {
        "vio.nadir_xy_only_velocity": True,
        "vio_vel.speed_r_inflate_breakpoints_m_s": [25.0, 40.0, 55.0],
        "vio_vel.speed_r_inflate_values": [1.5, 2.5, 4.0],
        "vio_vel.max_delta_v_xy_per_update_m_s": 2.0,
        "vio_vel.min_flow_px_high_speed": 0.8,
    },
    "3": {
        "kinematic_guard.vel_mismatch_warn": 6.0,
        "kinematic_guard.vel_mismatch_hard": 12.0,
        "kinematic_guard.hard_blend_alpha": 0.12,
        "kinematic_guard.max_state_speed_m_s": 70.0,
        "kinematic_guard.max_kin_speed_m_s": 65.0,
        "kinematic_guard.hard_hold_sec": 0.30,
        "kinematic_guard.release_hysteresis_ratio": 0.75,
    },
    "4": {
        "magnetometer.warning_weak_yaw.conditioning_guard_warn_pcond": 8.0e11,
        "magnetometer.warning_weak_yaw.conditioning_guard_warn_pmax": 8.0e6,
        "magnetometer.warning_weak_yaw.conditioning_guard_hard_pcond": 1.0e12,
        "magnetometer.warning_weak_yaw.conditioning_guard_hard_pmax": 1.0e7,
        "magnetometer.warning_weak_yaw.warning_extra_r_mult": 2.0,
        "magnetometer.warning_weak_yaw.degraded_extra_r_mult": 2.8,
    },
    "5": {
        "vps.apply_failsoft_r_mult": 2.5,
        "vps.apply_failsoft_max_offset_m": 140.0,
        "vps.apply_failsoft_max_dir_change_deg": 60.0,
        "vps.apply_failsoft_large_offset_confirm_m": 80.0,
        "vps.apply_failsoft_large_offset_confirm_hits": 2,
        "vps.apply_failsoft_allow_warning": True,
        "vps.apply_failsoft_allow_degraded": False,
    },
}


def list_run_dirs(repo_dir: Path) -> list[Path]:
    runs = sorted(
        repo_dir.glob("benchmark_modular_*/preintegration"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
    )
    return runs


def resolve_baseline(repo_dir: Path, explicit: str, baseline_file: Path) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (repo_dir / p).resolve()
        if p.is_dir():
            return p
        return None

    if baseline_file.is_file():
        val = baseline_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        if val:
            p = Path(val[0].strip())
            if not p.is_absolute():
                p = (repo_dir / p).resolve()
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
) -> tuple[int, Path | None]:
    before = {p.resolve() for p in list_run_dirs(repo_dir)}
    env = os.environ.copy()
    env["CONFIG"] = str(config_path)
    env["RUN_MODE"] = run_mode
    env["SAVE_DEBUG_DATA"] = str(int(save_debug_data))
    env["SAVE_KEYFRAME_IMAGES"] = str(int(save_keyframe_images))
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

    m = re.search(r"Output directory:\s*([^\s]+/preintegration)", "".join(captured))
    if m:
        p = Path(m.group(1))
        if not p.is_absolute():
            p = (repo_dir / p).resolve()
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


def evaluate_hard_locks(run_dir: Path | None) -> tuple[bool, dict[str, Any]]:
    row = read_health_row(run_dir)
    if row is None:
        return False, {"reason": "missing benchmark_health_summary.csv"}

    mag_cholfail = float(pd.to_numeric(row.get("mag_cholfail_rate"), errors="coerce"))
    cov_large = float(pd.to_numeric(row.get("cov_large_rate"), errors="coerce"))
    pmax_max = float(pd.to_numeric(row.get("pmax_max"), errors="coerce"))
    vps_used = parse_vps_used((run_dir / "run.log") if run_dir else Path(""))
    over = overflow_hits((run_dir / "run.log") if run_dir else Path(""))

    checks = {
        "mag_cholfail_rate<=0.08": bool(np.isfinite(mag_cholfail) and mag_cholfail <= 0.08),
        "vps_used>=20": bool(np.isfinite(vps_used) and vps_used >= 20.0),
        "cov_large_rate==0": bool(np.isfinite(cov_large) and abs(cov_large) <= 1e-12),
        "pmax_max<=1e6": bool(np.isfinite(pmax_max) and pmax_max <= 1.0e6),
        "no_overflow_nonfinite": bool(len(over) == 0),
    }
    ok = all(checks.values())
    detail = {
        "checks": checks,
        "mag_cholfail_rate": mag_cholfail,
        "cov_large_rate": cov_large,
        "pmax_max": pmax_max,
        "vps_used": vps_used,
        "overflow_hits": len(over),
    }
    return ok, detail


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
    print(
        "phase  status    err3d_mean(m)  delta_vs_baseline(%)  locks  run_dir"
    )
    for r in rows:
        d = r.get("err3d_delta_pct", np.nan)
        d_s = f"{d:+.2f}" if np.isfinite(d) else "n/a"
        print(
            f"{r['phase']:>5s}  {r['status']:<8s}  "
            f"{r['err3d_mean']:>13.3f}  {d_s:>21s}  "
            f"{str(r['locks_ok']):<5s}  {r['run_dir']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrate Bell412 one-knob phase tuning.")
    parser.add_argument("--config", default="configs/config_bell412_dataset3.yaml")
    parser.add_argument("--run_mode", default="auto")
    parser.add_argument("--save_debug_data", type=int, default=0, choices=[0, 1])
    parser.add_argument("--save_keyframe_images", type=int, default=0, choices=[0, 1])
    parser.add_argument("--baseline_run", default="")
    parser.add_argument("--max_regress_pct", type=float, default=15.0)
    parser.add_argument("--apply_best_config", type=int, default=0, choices=[0, 1])
    parser.add_argument("--phases", default="1,2,3,4,5")
    parser.add_argument("--dry_run", action="store_true", help="Plan and emit phase table without running benchmarks")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    config_path = (repo_dir / args.config).resolve()
    if not config_path.is_file():
        print(f"❌ config not found: {config_path}")
        return 2

    baseline_file = (repo_dir / "scripts" / "baseline_run.txt").resolve()
    baseline_run = resolve_baseline(repo_dir, args.baseline_run, baseline_file)
    baseline_err = read_err3d_mean(baseline_run)
    print(f"Baseline run: {baseline_run if baseline_run else 'None'}")
    print(f"Baseline err3d_mean: {baseline_err:.3f} m" if np.isfinite(baseline_err) else "Baseline err3d_mean: nan")

    with config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    for p in phases:
        if p not in PHASE_ASSIGNMENTS:
            print(f"❌ Unknown phase: {p}")
            return 2

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    orchestration_dir = repo_dir / f"benchmark_phase_tuning_{ts}"
    orchestration_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = orchestration_dir / "phase_results.csv"

    current_cfg = copy.deepcopy(base_cfg)
    accepted_cfg_by_run: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for phase in phases:
        print(f"\n================ Phase {phase} ================")
        candidate_cfg = apply_phase(current_cfg, phase)
        assignments = PHASE_ASSIGNMENTS.get(phase, {})
        for k, v in assignments.items():
            print(f"  set {k} = {v}")

        if args.dry_run:
            rows.append(
                {
                    "phase": phase,
                    "status": "DRYRUN",
                    "reason": "dry_run",
                    "run_dir": "",
                    "returncode": 0,
                    "err3d_mean": np.nan,
                    "err3d_baseline": baseline_err,
                    "err3d_delta_pct": np.nan,
                    "locks_ok": np.nan,
                    "vps_used": np.nan,
                    "mag_cholfail_rate": np.nan,
                    "cov_large_rate": np.nan,
                    "pmax_max": np.nan,
                    "overflow_hits": np.nan,
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
        )

        if run_dir is None:
            rows.append(
                {
                    "phase": phase,
                    "status": "FAIL",
                    "reason": "no_run_dir_detected",
                    "run_dir": "",
                    "returncode": ret,
                    "err3d_mean": np.nan,
                    "err3d_baseline": baseline_err,
                    "err3d_delta_pct": np.nan,
                    "locks_ok": False,
                    "vps_used": np.nan,
                    "mag_cholfail_rate": np.nan,
                    "cov_large_rate": np.nan,
                    "pmax_max": np.nan,
                    "overflow_hits": np.nan,
                }
            )
            print(f"[Phase {phase}] FAIL: no run directory detected")
            continue

        run_cfg_snapshot = run_dir / "phase_config_snapshot.yaml"
        write_yaml(run_cfg_snapshot, candidate_cfg)

        err_cur = read_err3d_mean(run_dir)
        delta_pct = np.nan
        regress_fail = False
        if np.isfinite(err_cur) and np.isfinite(baseline_err) and abs(baseline_err) > 1e-12:
            delta_pct = 100.0 * (err_cur - baseline_err) / abs(baseline_err)
            regress_fail = bool(delta_pct > args.max_regress_pct)

        locks_ok, lock_detail = evaluate_hard_locks(run_dir)
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
            "status": "PASS" if phase_ok else "FAIL",
            "reason": reason_text,
            "run_dir": str(run_dir),
            "returncode": ret,
            "err3d_mean": err_cur,
            "err3d_baseline": baseline_err,
            "err3d_delta_pct": delta_pct,
            "locks_ok": locks_ok,
            "vps_used": lock_detail.get("vps_used", np.nan),
            "mag_cholfail_rate": lock_detail.get("mag_cholfail_rate", np.nan),
            "cov_large_rate": lock_detail.get("cov_large_rate", np.nan),
            "pmax_max": lock_detail.get("pmax_max", np.nan),
            "overflow_hits": lock_detail.get("overflow_hits", np.nan),
        }
        rows.append(row)

        if phase_ok:
            current_cfg = candidate_cfg
            baseline_run = run_dir
            baseline_err = err_cur
            accepted_cfg_by_run[str(run_dir)] = copy.deepcopy(candidate_cfg)
            print(f"[Phase {phase}] PASS -> new baseline: {run_dir} (err3d_mean={err_cur:.3f} m)")
        else:
            print(f"[Phase {phase}] ROLLBACK ({reason_text})")

        save_phase_results(summary_csv, rows)

    if args.dry_run:
        save_phase_results(summary_csv, rows)
        print_phase_table(rows)
        print(f"\nSaved summary CSV: {summary_csv}")
        return 0

    # Pick best run among accepted phase runs (lowest err3d_mean).
    accepted = [r for r in rows if r["status"] == "PASS" and np.isfinite(r["err3d_mean"])]
    best_run: Path | None = None
    if accepted:
        best = min(accepted, key=lambda r: float(r["err3d_mean"]))
        best_run = Path(str(best["run_dir"])).resolve()
    elif baseline_run is not None and baseline_run.is_dir():
        best_run = baseline_run

    snapshot_path = (repo_dir / "configs" / "config_bell412_dataset3_accuracy_stage2.yaml").resolve()
    if best_run is not None:
        cfg_src = best_run / "phase_config_snapshot.yaml"
        if cfg_src.is_file():
            shutil.copy2(cfg_src, snapshot_path)
            print(f"Best config snapshot: {snapshot_path} (from {best_run})")
            if int(args.apply_best_config) == 1:
                shutil.copy2(cfg_src, config_path)
                print(f"Applied best config to: {config_path}")
        else:
            # Fallback: save current accepted config.
            write_yaml(snapshot_path, current_cfg)
            print(f"Best snapshot fallback saved: {snapshot_path}")
            if int(args.apply_best_config) == 1:
                write_yaml(config_path, current_cfg)
                print(f"Applied fallback best config to: {config_path}")

        baseline_file.write_text(str(best_run) + "\n", encoding="utf-8")
        print(f"Updated baseline pointer: {baseline_file} -> {best_run}")

    save_phase_results(summary_csv, rows)
    print_phase_table(rows)
    print(f"\nSaved summary CSV: {summary_csv}")

    if accepted:
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
