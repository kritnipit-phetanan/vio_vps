#!/usr/bin/env python3
"""Analyze one benchmark run directory and emit per-run reports.

This script is intentionally baseline-agnostic for main metrics.
Baseline deltas are handled by scripts/compare_benchmark.py.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _to_num(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _run_id_from_out_dir(out_dir: Path) -> str:
    if out_dir.name == "preintegration" and out_dir.parent.name:
        return out_dir.parent.name
    return out_dir.name


def _print_missing(path: Path) -> None:
    print(f"❌ Missing required file: {path}")


def print_accuracy_analysis(out_dir: Path) -> None:
    print("=== Accuracy Analysis ===")
    err_csv = out_dir / "error_log.csv"
    if not err_csv.is_file():
        _print_missing(err_csv)
        print("")
        return

    try:
        df = pd.read_csv(err_csv)
    except Exception as exc:
        print(f"❌ Failed reading {err_csv}: {exc}")
        print("")
        return

    if len(df) == 0:
        print("❌ error_log.csv is empty")
        print("")
        return

    if len(df) > 50:
        df = df.iloc[50:]

    pos = _to_num(df.get("pos_error_m", pd.Series(dtype=float)))
    vel = _to_num(df.get("vel_error_m_s", pd.Series(dtype=float)))
    alt = _to_num(df.get("alt_error_m", pd.Series(dtype=float)))
    e = _to_num(df.get("pos_error_E", pd.Series(dtype=float)))
    n = _to_num(df.get("pos_error_N", pd.Series(dtype=float)))
    u = _to_num(df.get("pos_error_U", pd.Series(dtype=float)))

    if pos.size:
        print(f"Position RMSE: {np.sqrt(np.nanmean(pos ** 2)):.3f} m")
    if alt.size:
        print(f"Altitude Mean Error: {np.nanmean(np.abs(alt)):.3f} m")
    if vel.size:
        print(f"Velocity RMSE: {np.sqrt(np.nanmean(vel ** 2)):.3f} m/s")
    if pos.size:
        print(f"Final Position Error: {float(pos[-1]):.3f} m")
    print("")
    if e.size:
        print(f"E mean error: {np.nanmean(e):.3f} m")
    if n.size:
        print(f"N mean error: {np.nanmean(n):.3f} m")
    if u.size:
        print(f"U mean error: {np.nanmean(u):.3f} m")
    print("")


def summarize_accuracy_first(out_dir: Path) -> dict:
    run_id = _run_id_from_out_dir(out_dir)
    result = {
        "run_id": run_id,
        "err3d_mean": np.nan,
        "err3d_median": np.nan,
        "err3d_final": np.nan,
        "heading_final_abs_deg": np.nan,
        "vps_used": np.nan,
        "msckf_fail_depth_sign": np.nan,
        "msckf_fail_reproj_error": np.nan,
        "msckf_fail_nonlinear": np.nan,
    }

    err_csv = out_dir / "error_log.csv"
    if err_csv.is_file():
        try:
            err_df = pd.read_csv(err_csv)
            if len(err_df) > 0:
                pos = _to_num(err_df.get("pos_error_m", pd.Series(dtype=float)))
                yaw = _to_num(err_df.get("yaw_error_deg", pd.Series(dtype=float)))
                pos = pos[np.isfinite(pos)]
                yaw = yaw[np.isfinite(yaw)]
                if pos.size:
                    result["err3d_mean"] = float(np.nanmean(pos))
                    result["err3d_median"] = float(np.nanmedian(pos))
                    result["err3d_final"] = float(pos[-1])
                if yaw.size:
                    result["heading_final_abs_deg"] = float(abs(yaw[-1]))
        except Exception:
            pass

    run_log = out_dir / "run.log"
    if run_log.is_file():
        try:
            with run_log.open("r", errors="ignore") as f:
                for line in f:
                    m = re.search(r"VPS used:\s*(\d+)", line)
                    if m:
                        result["vps_used"] = float(m.group(1))
                    m = re.search(r"fail_depth_sign:\s*(\d+)", line)
                    if m:
                        result["msckf_fail_depth_sign"] = float(m.group(1))
                    m = re.search(r"fail_reproj_error:\s*(\d+)", line)
                    if m:
                        result["msckf_fail_reproj_error"] = float(m.group(1))
                    m = re.search(r"fail_nonlinear:\s*(\d+)", line)
                    if m:
                        result["msckf_fail_nonlinear"] = float(m.group(1))
        except Exception:
            pass

    out_csv = out_dir / "accuracy_first_summary.csv"
    pd.DataFrame([result]).to_csv(out_csv, index=False)

    print("=== Accuracy-First Summary ===")
    print(f"mean 3D error      : {result['err3d_mean']:.3f} m")
    print(f"median 3D error    : {result['err3d_median']:.3f} m")
    print(f"final 3D error     : {result['err3d_final']:.3f} m")
    print(f"final |heading err|: {result['heading_final_abs_deg']:.3f} deg")
    vps_used = result["vps_used"]
    if np.isfinite(vps_used):
        print(f"VPS used           : {int(vps_used)}")
    else:
        print("VPS used           : nan")
    print(
        "MSCKF fails        : "
        f"depth_sign={int(result['msckf_fail_depth_sign']) if np.isfinite(result['msckf_fail_depth_sign']) else 'nan'}, "
        f"reproj={int(result['msckf_fail_reproj_error']) if np.isfinite(result['msckf_fail_reproj_error']) else 'nan'}, "
        f"nonlinear={int(result['msckf_fail_nonlinear']) if np.isfinite(result['msckf_fail_nonlinear']) else 'nan'}"
    )
    print(f"saved: {out_csv}")
    print("")
    return result


def summarize_spectacular_style(out_dir: Path) -> dict:
    run_id = _run_id_from_out_dir(out_dir)
    res = {
        "run_id": run_id,
        "samples": np.nan,
        "raw_h_cep50_m": np.nan,
        "raw_h_cep95_m": np.nan,
        "raw_h_rmse_m": np.nan,
        "raw_3d_rmse_m": np.nan,
        "aligned4dof_h_cep50_m": np.nan,
        "aligned4dof_h_cep95_m": np.nan,
        "aligned4dof_h_rmse_m": np.nan,
        "aligned4dof_3d_rmse_m": np.nan,
        "aligned4dof_final_3d_m": np.nan,
        "yaw_align_deg": np.nan,
        "heading_mae_deg": np.nan,
        "heading_final_abs_deg": np.nan,
        "aligned4dof_status": "not_computed",
        "aligned4dof_skip_reason": "",
    }

    err_csv = out_dir / "error_log.csv"
    if not err_csv.is_file():
        out_csv = out_dir / "spectacular_style_metrics.csv"
        pd.DataFrame([res]).to_csv(out_csv, index=False)
        print("=== Spectacular-Style Metrics ===")
        _print_missing(err_csv)
        print("")
        return res

    try:
        df = pd.read_csv(err_csv)
    except Exception as exc:
        out_csv = out_dir / "spectacular_style_metrics.csv"
        pd.DataFrame([res]).to_csv(out_csv, index=False)
        print("=== Spectacular-Style Metrics ===")
        print(f"❌ Failed reading {err_csv}: {exc}")
        print("")
        return res

    if len(df) > 0:
        res["samples"] = float(len(df))

    required = {"pos_error_E", "pos_error_N", "pos_error_U", "yaw_error_deg"}
    if required.issubset(df.columns):
        err_e = _to_num(df["pos_error_E"])
        err_n = _to_num(df["pos_error_N"])
        err_u = _to_num(df["pos_error_U"])
        yaw_err = np.abs(_to_num(df["yaw_error_deg"]))

        mask_raw = np.isfinite(err_e) & np.isfinite(err_n) & np.isfinite(err_u)
        if mask_raw.sum() >= 5:
            he = np.sqrt(err_e[mask_raw] ** 2 + err_n[mask_raw] ** 2)
            e3 = np.sqrt(err_e[mask_raw] ** 2 + err_n[mask_raw] ** 2 + err_u[mask_raw] ** 2)
            res["raw_h_cep50_m"] = float(np.percentile(he, 50))
            res["raw_h_cep95_m"] = float(np.percentile(he, 95))
            res["raw_h_rmse_m"] = float(np.sqrt(np.mean(he ** 2)))
            res["raw_3d_rmse_m"] = float(np.sqrt(np.mean(e3 ** 2)))

        yaw_ok = yaw_err[np.isfinite(yaw_err)]
        if yaw_ok.size:
            res["heading_mae_deg"] = float(np.mean(yaw_ok))
            res["heading_final_abs_deg"] = float(yaw_ok[-1])

        # 4-DoF alignment (yaw + xyz translation) reconstructed from error_log.
        if {"vio_E", "vio_N", "vio_U"}.issubset(df.columns):
            v_e = _to_num(df["vio_E"])
            v_n = _to_num(df["vio_N"])
            v_u = _to_num(df["vio_U"])
            g_e = v_e - err_e
            g_n = v_n - err_n
            g_u = v_u - err_u

            mask = (
                np.isfinite(v_e) & np.isfinite(v_n) & np.isfinite(v_u) &
                np.isfinite(g_e) & np.isfinite(g_n) & np.isfinite(g_u)
            )
            if mask.sum() >= 5:
                v_xy = np.column_stack([v_e[mask], v_n[mask]])
                g_xy = np.column_stack([g_e[mask], g_n[mask]])
                v_z = v_u[mask]
                g_z = g_u[mask]
                max_abs_guard = 1.0e7
                if (
                    np.nanmax(np.abs(v_xy)) > max_abs_guard
                    or np.nanmax(np.abs(g_xy)) > max_abs_guard
                    or np.nanmax(np.abs(v_z)) > max_abs_guard
                    or np.nanmax(np.abs(g_z)) > max_abs_guard
                ):
                    res["aligned4dof_status"] = "skipped"
                    res["aligned4dof_skip_reason"] = "magnitude_guard"
                else:
                    try:
                        with np.errstate(over="raise", invalid="raise", divide="raise"):
                            v_xy_mu = np.mean(v_xy, axis=0)
                            g_xy_mu = np.mean(g_xy, axis=0)
                            v_xy_c = v_xy - v_xy_mu
                            g_xy_c = g_xy - g_xy_mu

                            c_val = float(np.sum(v_xy_c[:, 0] * g_xy_c[:, 0] + v_xy_c[:, 1] * g_xy_c[:, 1]))
                            s_val = float(np.sum(v_xy_c[:, 0] * g_xy_c[:, 1] - v_xy_c[:, 1] * g_xy_c[:, 0]))
                            yaw = float(np.arctan2(s_val, c_val))
                            cy, sy = np.cos(yaw), np.sin(yaw)
                            r2 = np.array([[cy, -sy], [sy, cy]], dtype=float)

                            v_xy_aligned = (r2 @ v_xy.T).T
                            t_xy = g_xy_mu - np.mean(v_xy_aligned, axis=0)
                            v_xy_aligned = v_xy_aligned + t_xy
                            t_z = float(np.mean(g_z - v_z))
                            v_z_aligned = v_z + t_z

                            err_xy = v_xy_aligned - g_xy
                            err_z = v_z_aligned - g_z
                            err_h = np.sqrt(np.sum(err_xy ** 2, axis=1))
                            err_3d = np.sqrt(np.sum(err_xy ** 2, axis=1) + err_z ** 2)

                        finite_mask = np.isfinite(err_h) & np.isfinite(err_3d)
                        if finite_mask.sum() >= 5:
                            err_h = err_h[finite_mask]
                            err_3d = err_3d[finite_mask]
                            res["aligned4dof_h_cep50_m"] = float(np.percentile(err_h, 50))
                            res["aligned4dof_h_cep95_m"] = float(np.percentile(err_h, 95))
                            res["aligned4dof_h_rmse_m"] = float(np.sqrt(np.mean(err_h ** 2)))
                            res["aligned4dof_3d_rmse_m"] = float(np.sqrt(np.mean(err_3d ** 2)))
                            res["aligned4dof_final_3d_m"] = float(err_3d[-1])
                            res["yaw_align_deg"] = float(np.degrees(yaw))
                            res["aligned4dof_status"] = "ok"
                            res["aligned4dof_skip_reason"] = ""
                        else:
                            res["aligned4dof_status"] = "skipped"
                            res["aligned4dof_skip_reason"] = "nonfinite_aligned_error"
                    except FloatingPointError:
                        res["aligned4dof_status"] = "skipped"
                        res["aligned4dof_skip_reason"] = "numeric_guard_floating_point"
                    except Exception:
                        res["aligned4dof_status"] = "skipped"
                        res["aligned4dof_skip_reason"] = "alignment_exception"
            else:
                res["aligned4dof_status"] = "skipped"
                res["aligned4dof_skip_reason"] = "insufficient_valid_samples"
        else:
            res["aligned4dof_status"] = "skipped"
            res["aligned4dof_skip_reason"] = "missing_vio_columns"

    out_csv = out_dir / "spectacular_style_metrics.csv"
    pd.DataFrame([res]).to_csv(out_csv, index=False)

    print("=== Spectacular-Style Metrics ===")
    print("Horizontal CEP (raw ENU error):")
    print(f"  CEP50          : {res['raw_h_cep50_m']:.3f} m")
    print(f"  CEP95          : {res['raw_h_cep95_m']:.3f} m")
    print(f"  RMSE (horizontal): {res['raw_h_rmse_m']:.3f} m")
    print("")
    print("4-DoF aligned ATE (yaw + translation):")
    print(f"  ATE RMSE 3D    : {res['aligned4dof_3d_rmse_m']:.3f} m")
    print(f"  ATE final 3D   : {res['aligned4dof_final_3d_m']:.3f} m")
    print(f"  CEP50/95 (2D)  : {res['aligned4dof_h_cep50_m']:.3f} / {res['aligned4dof_h_cep95_m']:.3f} m")
    print(f"  best yaw align : {res['yaw_align_deg']:.3f} deg")
    if res.get("aligned4dof_status") != "ok":
        print(
            f"  aligned4dof status: {res.get('aligned4dof_status')} "
            f"({res.get('aligned4dof_skip_reason', '')})"
        )
    print("")
    print("Heading error:")
    print(f"  MAE            : {res['heading_mae_deg']:.3f} deg")
    print(f"  final |err|    : {res['heading_final_abs_deg']:.3f} deg")
    print(f"saved: {out_csv}")
    print("")
    return res


def print_runtime_profiling(out_dir: Path) -> None:
    print("=== Runtime Profiling (quick) ===")
    inf_csv = out_dir / "inference_log.csv"
    pose_csv = out_dir / "pose.csv"
    vps_attempts_csv = out_dir / "debug_vps_attempts.csv"
    vps_profile_csv = out_dir / "debug_vps_profile.csv"

    def _read_last_pose_time(path: Path) -> float:
        if not path.is_file():
            return float("nan")
        try:
            p = pd.read_csv(path)
            if len(p) == 0:
                return float("nan")
            return float(pd.to_numeric(p.iloc[-1, 0], errors="coerce"))
        except Exception:
            return float("nan")

    if inf_csv.is_file():
        inf = pd.read_csv(inf_csv)
        if len(inf) > 0:
            dt = pd.to_numeric(inf.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
            dt = dt[np.isfinite(dt)]
            proc_total = float(np.nansum(dt)) if dt.size else float("nan")
            avg_dt = float(np.nanmean(dt)) if dt.size else float("nan")
            max_dt = float(np.nanmax(dt)) if dt.size else float("nan")
            sim_time = _read_last_pose_time(pose_csv)
            rtf = proc_total / sim_time if np.isfinite(sim_time) and sim_time > 1e-9 else float("nan")
            print(f"inference rows   : {len(inf)}")
            print(f"proc_total       : {proc_total:.3f} s")
            print(f"avg_dt           : {avg_dt:.6f} s (avg_fps={1.0/avg_dt:.2f})")
            print(f"max_dt           : {max_dt:.6f} s")
            print(f"sim_time         : {sim_time:.3f} s")
            print(f"RTF (proc/sim)   : {rtf:.3f}x")
    else:
        print("No inference_log.csv")

    if vps_attempts_csv.is_file():
        a = pd.read_csv(vps_attempts_csv)
        if len(a) > 0 and "processing_time_ms" in a.columns:
            t = pd.to_numeric(a["processing_time_ms"], errors="coerce").to_numpy(dtype=float)
            t = t[np.isfinite(t)]
            if t.size:
                print(f"VPS attempts     : {len(a)}")
                print(f"VPS total        : {np.sum(t):.2f} ms")
                print(f"VPS avg / p95    : {np.mean(t):.2f} / {np.percentile(t,95):.2f} ms")

    if vps_profile_csv.is_file():
        vp = pd.read_csv(vps_profile_csv)
        if len(vp) > 0:
            ok = vp[vp["success"] == 1] if "success" in vp.columns else vp
            src = ok if len(ok) > 0 else vp
            cols = ["tile_ms", "preprocess_ms", "match_ms", "pose_ms", "total_ms"]
            summary = ", ".join(
                f"{c}={pd.to_numeric(src[c], errors='coerce').mean():.2f}ms"
                for c in cols if c in src.columns
            )
            print(f"VPS stage means  : {summary}")

    csv_sizes: list[tuple[str, int]] = []
    for path in out_dir.iterdir():
        if path.suffix.lower() == ".csv":
            try:
                csv_sizes.append((path.name, path.stat().st_size))
            except OSError:
                pass
    csv_sizes.sort(key=lambda x: x[1], reverse=True)
    if csv_sizes:
        print("Top CSV size     :")
        for name, size in csv_sizes[:8]:
            print(f"  {name:28s} {size/1024/1024:8.2f} MB")
    print("")


def summarize_sensor_phase(out_dir: Path, baseline_run: Path | None = None) -> None:
    print("=== Sensor Health by Phase (accept-rate / NIS EWMA) ===")
    sensor_csv = out_dir / "sensor_health.csv"
    adaptive_csv = out_dir / "adaptive_debug.csv"
    if not (sensor_csv.is_file() and adaptive_csv.is_file()):
        print("Missing sensor_health.csv or adaptive_debug.csv")
        print("")
        return

    sensor_df = pd.read_csv(sensor_csv)
    adaptive_df = pd.read_csv(adaptive_csv)
    if len(sensor_df) == 0 or len(adaptive_df) == 0 or "phase" not in adaptive_df.columns:
        print("No sensor/adaptive rows for phase summary")
        print("")
        return

    adaptive_df = adaptive_df[["t", "phase"]].copy()
    sensor_df = sensor_df.sort_values("t")
    adaptive_df = adaptive_df.sort_values("t")
    merged = pd.merge_asof(sensor_df, adaptive_df, on="t", direction="backward")
    merged["phase"] = merged["phase"].fillna(-1).astype(int)

    summary = (
        merged.groupby(["sensor", "phase"], dropna=False)
        .agg(
            samples=("accepted", "count"),
            logged_accept_ratio=("accepted", "mean"),
            accept_rate_mean=("accept_rate", "mean"),
            accept_rate_last=("accept_rate", "last"),
            nis_ewma_mean=("nis_ewma", "mean"),
            nis_ewma_last=("nis_ewma", "last"),
        )
        .reset_index()
        .sort_values(["sensor", "phase"])
    )
    out_csv = out_dir / "sensor_phase_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(summary.to_string(index=False))
    print(f"saved: {out_csv}")

    if baseline_run is None:
        print("")
        return

    b_sensor = baseline_run / "sensor_health.csv"
    b_adaptive = baseline_run / "adaptive_debug.csv"
    if not (b_sensor.is_file() and b_adaptive.is_file()):
        print("")
        return
    b_s = pd.read_csv(b_sensor)
    b_a = pd.read_csv(b_adaptive)
    if len(b_s) == 0 or len(b_a) == 0 or "phase" not in b_a.columns:
        print("")
        return

    b_a = b_a[["t", "phase"]].copy()
    b_s = b_s.sort_values("t")
    b_a = b_a.sort_values("t")
    b_m = pd.merge_asof(b_s, b_a, on="t", direction="backward")
    b_m["phase"] = b_m["phase"].fillna(-1).astype(int)

    b_sum = (
        b_m.groupby(["sensor", "phase"], dropna=False)
        .agg(
            logged_accept_ratio_base=("accepted", "mean"),
            accept_rate_mean_base=("accept_rate", "mean"),
            accept_rate_last_base=("accept_rate", "last"),
            nis_ewma_mean_base=("nis_ewma", "mean"),
        )
        .reset_index()
    )
    joined = summary.merge(b_sum, on=["sensor", "phase"], how="left")
    joined["accept_rate_mean_delta"] = joined["accept_rate_mean"] - joined["accept_rate_mean_base"]
    joined["accept_rate_last_delta"] = joined["accept_rate_last"] - joined["accept_rate_last_base"]
    joined["nis_ewma_delta"] = joined["nis_ewma_mean"] - joined["nis_ewma_mean_base"]
    print("\nBaseline delta (sensor+phase):")
    print(
        joined[
            ["sensor", "phase", "accept_rate_mean_delta", "accept_rate_last_delta", "nis_ewma_delta"]
        ].to_string(index=False)
    )
    print("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze one benchmark output directory.")
    parser.add_argument("--output_dir", required=True, help="Run output directory (usually outputs/benchmark_modular_*)")
    parser.add_argument("--baseline_run", default="", help="Optional baseline run directory for sensor-phase delta")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    baseline_dir = Path(args.baseline_run).resolve() if args.baseline_run else None
    if not out_dir.is_dir():
        print(f"❌ output_dir not found: {out_dir}")
        return 2

    print_accuracy_analysis(out_dir)
    summarize_accuracy_first(out_dir)
    summarize_spectacular_style(out_dir)
    print_runtime_profiling(out_dir)
    summarize_sensor_phase(out_dir, baseline_dir if baseline_dir and baseline_dir.is_dir() else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
