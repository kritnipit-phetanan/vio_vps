#!/usr/bin/env python3
"""Compare benchmark outputs with baseline and run hard-lock checks."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


HEALTH_COLUMNS = [
    "run_id",
    "projection_count",
    "first_projection_time",
    "pcond_max",
    "pmax_max",
    "cov_large_rate",
    "pos_rmse",
    "final_pos_err",
    "final_alt_err",
    "frames_inlier_nonzero_ratio",
    "vio_vel_accept_ratio_vs_cam",
    "mag_cholfail_rate",
    "loop_applied_rate",
    "speed_max_m_s",
    "speed_p99_m_s",
    "loop_corr_count",
    "loop_abs_yaw_corr_sum_deg",
    "vps_soft_accept_count",
    "vps_soft_reject_count",
    "mag_accept_rate",
    "vps_jump_reject_count",
    "vps_temporal_confirm_count",
    "abs_corr_apply_count",
    "abs_corr_soft_count",
    "backend_apply_count",
    "backend_stale_drop_count",
    "backend_poll_count",
    "vps_attempt_count",
    "vps_worker_busy_skips",
    "vps_attempt_ms_p50",
    "vps_attempt_ms_p95",
    "vps_time_budget_stops",
    "vps_evaluated_candidates_mean",
    "policy_conflict_count",
    "rtf_proc_sim",
]


def _to_float(v: object, default: float = np.nan) -> float:
    try:
        out = float(pd.to_numeric(v, errors="coerce"))
        return out
    except Exception:
        return float(default)


def _run_id_from_dir(run_dir: Path) -> str:
    if run_dir.name == "preintegration" and run_dir.parent.name:
        return run_dir.parent.name
    return run_dir.name


def _load_last_row(csv_path: Path) -> pd.Series | None:
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if len(df) == 0:
        return None
    row = df.iloc[-1].copy()
    for col in HEALTH_COLUMNS:
        if col not in row.index:
            row[col] = np.nan
    return row


def _build_health_summary_fallback(run_dir: Path, out_csv: Path) -> Path | None:
    run_log = run_dir / "run.log"
    cov_csv = run_dir / "cov_health.csv"
    err_csv = run_dir / "error_log.csv"

    projection_count = np.nan
    first_projection_time = np.nan
    pcond_max = np.nan
    pmax_max = np.nan
    cov_large_rate = np.nan
    pos_rmse = np.nan
    final_pos_err = np.nan
    final_alt_err = np.nan

    if run_log.is_file():
        try:
            lines = run_log.read_text(errors="ignore").splitlines()
            projection_count = float(
                sum(
                    1
                    for line in lines
                    if ("Eigenvalue projection" in line or "[EKF-COND]" in line)
                )
            )
            for line in lines:
                m = re.search(r"\[EKF-COND\]\s+t=([0-9.+-eE]+)", line)
                if m:
                    first_projection_time = float(m.group(1))
                    break
        except Exception:
            pass

    if cov_csv.is_file():
        try:
            cov_df = pd.read_csv(cov_csv)
            if len(cov_df) > 0:
                pcond_max = _to_float(pd.to_numeric(cov_df.get("p_cond"), errors="coerce").max())
                pmax_max = _to_float(pd.to_numeric(cov_df.get("p_max"), errors="coerce").max())
                cov_large_rate = _to_float(pd.to_numeric(cov_df.get("large_flag"), errors="coerce").mean())
        except Exception:
            pass

    if err_csv.is_file():
        try:
            err_df = pd.read_csv(err_csv)
            if len(err_df) > 0 and "pos_error_m" in err_df.columns:
                pos = pd.to_numeric(err_df["pos_error_m"], errors="coerce").to_numpy(dtype=float)
                alt = pd.to_numeric(err_df.get("alt_error_m"), errors="coerce").to_numpy(dtype=float)
                pos_rmse = float(np.sqrt(np.nanmean(pos ** 2)))
                final_pos_err = float(pos[-1])
                if alt.size:
                    final_alt_err = float(alt[-1])
        except Exception:
            pass

    row = {col: np.nan for col in HEALTH_COLUMNS}
    row["run_id"] = _run_id_from_dir(run_dir)
    row["projection_count"] = projection_count
    row["first_projection_time"] = first_projection_time
    row["pcond_max"] = pcond_max
    row["pmax_max"] = pmax_max
    row["cov_large_rate"] = cov_large_rate
    row["pos_rmse"] = pos_rmse
    row["final_pos_err"] = final_pos_err
    row["final_alt_err"] = final_alt_err

    try:
        pd.DataFrame([row], columns=HEALTH_COLUMNS).to_csv(out_csv, index=False)
        return out_csv
    except Exception:
        return None


def ensure_health_summary(run_dir: Path) -> Path | None:
    summary_csv = run_dir / "benchmark_health_summary.csv"
    row = _load_last_row(summary_csv)
    if row is not None:
        return summary_csv
    fallback = run_dir / "_benchmark_health_summary_fallback.csv"
    return _build_health_summary_fallback(run_dir, fallback)


def _print_accuracy_delta(cur_dir: Path, base_dir: Path) -> None:
    cur_csv = cur_dir / "accuracy_first_summary.csv"
    base_csv = base_dir / "accuracy_first_summary.csv"
    if not (cur_csv.is_file() and base_csv.is_file()):
        return
    try:
        cur = pd.read_csv(cur_csv).iloc[-1]
        base = pd.read_csv(base_csv).iloc[-1]
    except Exception:
        return

    keys = [
        "err3d_mean",
        "err3d_median",
        "err3d_final",
        "heading_final_abs_deg",
        "vps_used",
        "msckf_fail_depth_sign",
        "msckf_fail_reproj_error",
        "msckf_fail_nonlinear",
    ]
    print(f"Baseline accuracy delta vs: {base_dir}")
    for k in keys:
        c = _to_float(cur.get(k))
        b = _to_float(base.get(k))
        if np.isfinite(b) and abs(b) > 1e-12 and np.isfinite(c):
            pct = 100.0 * (c - b) / abs(b)
            print(f"{k:24s} base={b: .6e} cur={c: .6e} delta={pct:+7.2f}%")
        else:
            print(f"{k:24s} base={b: .6e} cur={c: .6e} delta=   n/a")
    print("")


def _print_spectacular_delta(cur_dir: Path, base_dir: Path) -> None:
    cur_csv = cur_dir / "spectacular_style_metrics.csv"
    base_csv = base_dir / "spectacular_style_metrics.csv"
    if not (cur_csv.is_file() and base_csv.is_file()):
        return
    try:
        cur = pd.read_csv(cur_csv).iloc[-1]
        base = pd.read_csv(base_csv).iloc[-1]
    except Exception:
        return

    keys = [
        "raw_h_cep50_m",
        "raw_h_cep95_m",
        "raw_h_rmse_m",
        "aligned4dof_3d_rmse_m",
        "aligned4dof_h_cep50_m",
        "aligned4dof_h_cep95_m",
        "heading_mae_deg",
        "heading_final_abs_deg",
    ]
    print(f"Baseline spectacular-style delta vs: {base_dir}")
    for k in keys:
        c = _to_float(cur.get(k))
        b = _to_float(base.get(k))
        if np.isfinite(b) and abs(b) > 1e-12 and np.isfinite(c):
            pct = 100.0 * (c - b) / abs(b)
            print(f"{k:24s} base={b: .6e} cur={c: .6e} delta={pct:+7.2f}%")
        else:
            print(f"{k:24s} base={b: .6e} cur={c: .6e} delta=   n/a")
    print("")


def _print_health_summary(current_row: pd.Series) -> None:
    print("=== Conditioning Health Summary ===")
    print(f"projection_count   : {int(_to_float(current_row['projection_count'], 0.0))}")
    print(f"first_projection_t : {_to_float(current_row['first_projection_time']):.3f} s")
    print(f"pcond_max          : {_to_float(current_row['pcond_max']):.3e}")
    print(f"pmax_max           : {_to_float(current_row['pmax_max']):.3e}")
    print(f"cov_large_rate     : {_to_float(current_row['cov_large_rate']):.4f}")
    print(f"pos_rmse           : {_to_float(current_row['pos_rmse']):.3f} m")
    print(f"final_pos_err      : {_to_float(current_row['final_pos_err']):.3f} m")
    print(f"final_alt_err      : {_to_float(current_row['final_alt_err']):.3f} m")
    for col in [
        "frames_inlier_nonzero_ratio",
        "vio_vel_accept_ratio_vs_cam",
        "mag_cholfail_rate",
        "loop_applied_rate",
        "speed_max_m_s",
        "speed_p99_m_s",
        "loop_corr_count",
        "loop_abs_yaw_corr_sum_deg",
        "vps_soft_accept_count",
        "vps_soft_reject_count",
        "mag_accept_rate",
        "vps_jump_reject_count",
        "vps_temporal_confirm_count",
        "abs_corr_apply_count",
        "abs_corr_soft_count",
        "backend_apply_count",
        "backend_stale_drop_count",
        "backend_poll_count",
        "vps_attempt_count",
        "vps_worker_busy_skips",
        "vps_attempt_ms_p50",
        "vps_attempt_ms_p95",
        "vps_time_budget_stops",
        "vps_evaluated_candidates_mean",
        "policy_conflict_count",
        "rtf_proc_sim",
    ]:
        if col in current_row.index:
            print(f"{col:20s}: {_to_float(current_row[col]):.6f}")
    print("")


def _print_health_delta(current_row: pd.Series, base_row: pd.Series, baseline_run: Path) -> None:
    print("=== Before/After vs Baseline ===")
    print(f"Baseline: {baseline_run}")
    for m in HEALTH_COLUMNS[1:]:
        c = _to_float(current_row.get(m))
        b = _to_float(base_row.get(m))
        if np.isfinite(b) and abs(b) > 1e-12 and np.isfinite(c):
            pct = 100.0 * (c - b) / abs(b)
            print(f"{m:20s}  base={b: .6e}  cur={c: .6e}  delta={pct:+7.2f}%")
        else:
            print(f"{m:20s}  base={b: .6e}  cur={c: .6e}  delta=   n/a")
    print("")


def _load_heading_final_abs_deg(output_dir: Path) -> float:
    p = output_dir / "accuracy_first_summary.csv"
    if not p.is_file():
        return float("nan")
    try:
        df = pd.read_csv(p)
    except Exception:
        return float("nan")
    if len(df) == 0:
        return float("nan")
    return _to_float(df.iloc[-1].get("heading_final_abs_deg"))


def evaluate_locks(
    current_row: pd.Series,
    output_dir: Path,
    run_log: Path,
    profile_name: str,
) -> tuple[bool, list[tuple[str, bool, str]], list[str], float]:
    mag_cholfail = _to_float(current_row.get("mag_cholfail_rate"))
    cov_large = _to_float(current_row.get("cov_large_rate"))
    pmax_max = _to_float(current_row.get("pmax_max"))
    vps_jump_reject_count = _to_float(current_row.get("vps_jump_reject_count"))
    vps_attempt_count = _to_float(current_row.get("vps_attempt_count"))
    backend_stale_drop_count = _to_float(current_row.get("backend_stale_drop_count"))
    backend_poll_count = _to_float(current_row.get("backend_poll_count"))
    policy_conflict_count = _to_float(current_row.get("policy_conflict_count"))
    vps_used = _to_float(current_row.get("vps_used"))
    heading_final_abs_deg = _load_heading_final_abs_deg(output_dir)
    overflow_hits: list[str] = []

    if run_log.is_file():
        lines = run_log.read_text(errors="ignore").splitlines()
        for line in lines:
            m = re.search(r"VPS used:\s*(\d+)", line)
            if m and not np.isfinite(vps_used):
                vps_used = float(m.group(1))

        warn_patterns = ("overflow", "non-finite", "contains inf/nan", "runtimewarning")
        for line in lines:
            low = line.strip().lower()
            if any(p in low for p in warn_patterns):
                if "no overflow" in low:
                    continue
                overflow_hits.append(line.strip())

    jump_ratio = np.nan
    if np.isfinite(vps_jump_reject_count) and np.isfinite(vps_attempt_count) and vps_attempt_count > 0:
        jump_ratio = float(vps_jump_reject_count / vps_attempt_count)
    stale_ratio = np.nan
    if np.isfinite(backend_stale_drop_count) and np.isfinite(backend_poll_count) and backend_poll_count > 0:
        stale_ratio = float(backend_stale_drop_count / backend_poll_count)

    checks: list[tuple[str, bool, str]] = [
        (
            "cov_large_rate == 0",
            np.isfinite(cov_large) and abs(cov_large) <= 1e-12,
            f"value={cov_large:.6f}" if np.isfinite(cov_large) else "value=nan",
        ),
        (
            "pmax_max <= 1e6",
            np.isfinite(pmax_max) and pmax_max <= 1.0e6,
            f"value={pmax_max:.3e}" if np.isfinite(pmax_max) else "value=nan",
        ),
        (
            "no overflow/non-finite flood in run.log",
            len(overflow_hits) == 0,
            f"hits={len(overflow_hits)}",
        ),
        (
            "policy_conflict_count == 0",
            np.isfinite(policy_conflict_count) and abs(policy_conflict_count) <= 1e-12,
            f"value={policy_conflict_count:.0f}" if np.isfinite(policy_conflict_count) else "value=nan",
        ),
    ]

    profile = str(profile_name).strip().lower()
    if profile == "pre_backend":
        checks.extend(
            [
                (
                    "mag_cholfail_rate <= 0.10",
                    np.isfinite(mag_cholfail) and mag_cholfail <= 0.10,
                    f"value={mag_cholfail:.6f}" if np.isfinite(mag_cholfail) else "value=nan",
                ),
                (
                    "VPS used >= 10",
                    np.isfinite(vps_used) and vps_used >= 10.0,
                    f"value={vps_used:.0f}" if np.isfinite(vps_used) else "value=nan",
                ),
                (
                    "vps_jump_reject_count/vps_attempts <= 0.5",
                    np.isfinite(jump_ratio) and jump_ratio <= 0.5,
                    f"value={jump_ratio:.6f}" if np.isfinite(jump_ratio) else "value=nan",
                ),
                (
                    "heading_final_abs_deg <= 15",
                    np.isfinite(heading_final_abs_deg) and heading_final_abs_deg <= 15.0,
                    f"value={heading_final_abs_deg:.3f}" if np.isfinite(heading_final_abs_deg) else "value=nan",
                ),
            ]
        )
    elif profile in ("backend", "near_rt_backend"):
        checks.extend(
            [
                (
                    "mag_cholfail_rate <= 0.08",
                    np.isfinite(mag_cholfail) and mag_cholfail <= 0.08,
                    f"value={mag_cholfail:.6f}" if np.isfinite(mag_cholfail) else "value=nan",
                ),
                (
                    "VPS used >= 20",
                    np.isfinite(vps_used) and vps_used >= 20.0,
                    f"value={vps_used:.0f}" if np.isfinite(vps_used) else "value=nan",
                ),
                (
                    "backend_stale_drop_count/backend_poll_count <= 0.2",
                    np.isfinite(stale_ratio) and stale_ratio <= 0.2,
                    f"value={stale_ratio:.6f}" if np.isfinite(stale_ratio) else "value=nan",
                ),
                (
                    "heading_final_abs_deg <= 10",
                    np.isfinite(heading_final_abs_deg) and heading_final_abs_deg <= 10.0,
                    f"value={heading_final_abs_deg:.3f}" if np.isfinite(heading_final_abs_deg) else "value=nan",
                ),
            ]
        )
        if profile == "near_rt_backend":
            rtf_proc_sim = _to_float(current_row.get("rtf_proc_sim"))
            checks.append(
                (
                    "rtf_proc_sim <= 1.2",
                    np.isfinite(rtf_proc_sim) and rtf_proc_sim <= 1.2,
                    f"value={rtf_proc_sim:.6f}" if np.isfinite(rtf_proc_sim) else "value=nan",
                )
            )
    else:
        checks.append(
            ("lock profile recognized", False, f"profile={profile}")
        )
    all_ok = all(ok for _, ok, _ in checks)
    return all_ok, checks, overflow_hits, vps_used


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare benchmark output with baseline and check locks.")
    parser.add_argument("--output_dir", required=True, help="Current run output directory")
    parser.add_argument("--baseline_run", default="", help="Optional baseline run directory")
    parser.add_argument("--lock_profile", default="backend", choices=["pre_backend", "backend", "near_rt_backend"])
    parser.add_argument("--enforce_locks", action="store_true", help="Return non-zero when any hard lock fails")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    if not out_dir.is_dir():
        print(f"❌ output_dir not found: {out_dir}")
        return 2

    current_summary = ensure_health_summary(out_dir)
    if current_summary is None:
        print("❌ Could not produce benchmark_health_summary.csv")
        return 2
    cur_row = _load_last_row(current_summary)
    if cur_row is None:
        print("❌ benchmark_health_summary.csv is empty")
        return 2

    _print_health_summary(cur_row)

    baseline_dir = Path(args.baseline_run).resolve() if args.baseline_run else None
    if baseline_dir and baseline_dir.is_dir():
        _print_accuracy_delta(out_dir, baseline_dir)
        _print_spectacular_delta(out_dir, baseline_dir)
        base_summary = ensure_health_summary(baseline_dir)
        if base_summary is not None:
            base_row = _load_last_row(base_summary)
            if base_row is not None:
                _print_health_delta(cur_row, base_row, baseline_dir)
    else:
        print("=== Before/After vs Baseline ===")
        print("No baseline summary found; skipping before/after diff.")
        print("")

    print("=== Hard Lock Checks ===")
    print(f"Profile: {args.lock_profile}")
    all_ok, checks, overflow_hits, _ = evaluate_locks(
        cur_row, out_dir, out_dir / "run.log", args.lock_profile
    )
    for name, ok, detail in checks:
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name:40s} ({detail})")
    if all_ok:
        print("✅ LOCK RESULT: PASS")
    else:
        print("❌ LOCK RESULT: REGRESSION")
        if overflow_hits:
            print("  overflow/non-finite samples:")
            for line in overflow_hits[:5]:
                print(f"    - {line}")
    print("")
    if args.enforce_locks and not all_ok:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
