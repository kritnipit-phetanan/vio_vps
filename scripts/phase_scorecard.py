#!/usr/bin/env python3
"""Build one-run scorecard against a frozen baseline.

Usage:
  python3 scripts/phase_scorecard.py \
    --run outputs/benchmark_modular_YYYYmmdd_HHMMSS \
    --baseline outputs/benchmark_modular_YYYYmmdd_HHMMSS
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        out = float(pd.to_numeric(v, errors="coerce"))
        return out
    except Exception:
        return float(default)


def _load_accuracy(run_dir: Path) -> Dict[str, float]:
    p = run_dir / "accuracy_first_summary.csv"
    if not p.is_file():
        return {"err3d_mean": float("nan"), "err3d_final": float("nan")}
    df = pd.read_csv(p)
    if len(df) <= 0:
        return {"err3d_mean": float("nan"), "err3d_final": float("nan")}
    row = df.iloc[-1]
    return {
        "err3d_mean": _safe_float(row.get("err3d_mean")),
        "err3d_final": _safe_float(row.get("err3d_final")),
    }


def _load_health_row(run_dir: Path) -> pd.Series:
    p = run_dir / "benchmark_health_summary.csv"
    if not p.is_file():
        return pd.Series(dtype=float)
    df = pd.read_csv(p)
    if len(df) <= 0:
        return pd.Series(dtype=float)
    return df.iloc[-1]


def _quarter_means(run_dir: Path) -> Dict[str, float]:
    p = run_dir / "error_log.csv"
    if not p.is_file():
        return {f"q{i}_mean": float("nan") for i in range(1, 5)}
    df = pd.read_csv(p)
    if len(df) <= 0 or "t" not in df.columns or "pos_error_m" not in df.columns:
        return {f"q{i}_mean": float("nan") for i in range(1, 5)}
    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(df["pos_error_m"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(t) & np.isfinite(e)
    if not np.any(m):
        return {f"q{i}_mean": float("nan") for i in range(1, 5)}
    t = t[m]
    e = e[m]
    t0 = float(np.min(t))
    t1 = float(np.max(t))
    span = max(1e-9, t1 - t0)
    q = np.clip(((t - t0) / span * 4.0).astype(int) + 1, 1, 4)
    out: Dict[str, float] = {}
    for i in range(1, 5):
        vals = e[q == i]
        out[f"q{i}_mean"] = float(np.mean(vals)) if vals.size > 0 else float("nan")
    return out


def _backend_q_commits(run_dir: Path) -> Dict[str, float]:
    p = run_dir / "backend_apply_trace.csv"
    if not p.is_file():
        return {f"commit_q{i}": float("nan") for i in range(1, 5)}
    df = pd.read_csv(p)
    if len(df) <= 0 or "state" not in df.columns or "q_bucket" not in df.columns:
        return {f"commit_q{i}": float("nan") for i in range(1, 5)}
    c = df[df["state"].astype(str).str.upper() == "COMMIT"].copy()
    qb = pd.to_numeric(c["q_bucket"], errors="coerce").to_numpy(dtype=float)
    out: Dict[str, float] = {}
    for i in range(1, 5):
        out[f"commit_q{i}"] = float(np.sum(qb == float(i)))
    return out


def _pct_change(cur: float, base: float) -> float:
    if not (math.isfinite(cur) and math.isfinite(base)):
        return float("nan")
    if abs(base) <= 1e-12:
        return float("nan")
    return float((cur - base) / abs(base) * 100.0)


@dataclass
class GuardResult:
    ok: bool
    mean_regress_pct: float
    final_regress_pct: float
    runtime_delta_pct: float


def _evaluate_guards(
    run_acc: Dict[str, float],
    base_acc: Dict[str, float],
    run_health: pd.Series,
    base_health: pd.Series,
) -> GuardResult:
    mean_regress = _pct_change(run_acc.get("err3d_mean", float("nan")), base_acc.get("err3d_mean", float("nan")))
    final_regress = _pct_change(run_acc.get("err3d_final", float("nan")), base_acc.get("err3d_final", float("nan")))
    run_rt = _safe_float(run_health.get("vps_attempt_ms_p95"))
    base_rt = _safe_float(base_health.get("vps_attempt_ms_p95"))
    runtime_delta = _pct_change(run_rt, base_rt)
    ok = True
    if math.isfinite(mean_regress) and mean_regress > 10.0:
        ok = False
    if math.isfinite(final_regress) and final_regress > 15.0:
        ok = False
    if math.isfinite(runtime_delta) and runtime_delta > 15.0:
        ok = False
    return GuardResult(
        ok=bool(ok),
        mean_regress_pct=float(mean_regress),
        final_regress_pct=float(final_regress),
        runtime_delta_pct=float(runtime_delta),
    )


def build_scorecard(run_dir: Path, baseline_dir: Path) -> pd.DataFrame:
    run_acc = _load_accuracy(run_dir)
    base_acc = _load_accuracy(baseline_dir)
    run_health = _load_health_row(run_dir)
    base_health = _load_health_row(baseline_dir)
    run_q = _quarter_means(run_dir)
    run_commit_q = _backend_q_commits(run_dir)

    probation = _safe_float(run_health.get("backend_probation_count"))
    probation_commit = _safe_float(run_health.get("backend_probation_commit_count"))
    prob_commit_ratio = float("nan")
    if math.isfinite(probation) and probation > 0 and math.isfinite(probation_commit):
        prob_commit_ratio = float(probation_commit / probation)

    guards = _evaluate_guards(run_acc, base_acc, run_health, base_health)
    row = {
        "baseline_run": baseline_dir.name,
        "run_id": run_dir.name,
        "err3d_mean": run_acc["err3d_mean"],
        "err3d_final": run_acc["err3d_final"],
        "err3d_mean_regress_pct": guards.mean_regress_pct,
        "err3d_final_regress_pct": guards.final_regress_pct,
        "vps_attempt_ms_p95": _safe_float(run_health.get("vps_attempt_ms_p95")),
        "vps_attempt_ms_p95_delta_pct": guards.runtime_delta_pct,
        "backend_probation_count": probation,
        "backend_probation_commit_count": probation_commit,
        "probation_to_commit_ratio": prob_commit_ratio,
        "backend_no_commit_streak_max": _safe_float(run_health.get("backend_no_commit_streak_max")),
        "backend_continuity_try_count": _safe_float(run_health.get("backend_continuity_try_count")),
        "backend_continuity_bounded_commit_count": _safe_float(
            run_health.get("backend_continuity_bounded_commit_count")
        ),
        "guard_status": "PASS" if guards.ok else "ROLLBACK",
    }
    row.update(run_q)
    row.update(run_commit_q)
    return pd.DataFrame([row])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one-run scorecard against frozen baseline")
    parser.add_argument("--run", required=True, help="Current run directory")
    parser.add_argument("--baseline", required=True, help="Baseline run directory")
    parser.add_argument("--write_csv", default="", help="Optional explicit output csv path")
    args = parser.parse_args()

    run_dir = Path(args.run)
    baseline_dir = Path(args.baseline)
    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir}")
    if not baseline_dir.is_dir():
        raise SystemExit(f"baseline dir not found: {baseline_dir}")

    score = build_scorecard(run_dir, baseline_dir)
    out_csv = Path(args.write_csv) if str(args.write_csv).strip() else (run_dir / "phase_scorecard.csv")
    score.to_csv(out_csv, index=False)

    row = score.iloc[0]
    print(f"run_id={row['run_id']}")
    print(f"baseline={row['baseline_run']}")
    print(f"guard_status={row['guard_status']}")
    print(
        "err3d_mean={:.3f} ({:+.2f}%) err3d_final={:.3f} ({:+.2f}%)".format(
            _safe_float(row["err3d_mean"]),
            _safe_float(row["err3d_mean_regress_pct"]),
            _safe_float(row["err3d_final"]),
            _safe_float(row["err3d_final_regress_pct"]),
        )
    )
    print(
        "runtime_p95={:.3f} ({:+.2f}%) probation_to_commit={:.4f}".format(
            _safe_float(row["vps_attempt_ms_p95"]),
            _safe_float(row["vps_attempt_ms_p95_delta_pct"]),
            _safe_float(row["probation_to_commit_ratio"]),
        )
    )
    print(
        "Q-mean: Q1={:.2f} Q2={:.2f} Q3={:.2f} Q4={:.2f}".format(
            _safe_float(row["q1_mean"]),
            _safe_float(row["q2_mean"]),
            _safe_float(row["q3_mean"]),
            _safe_float(row["q4_mean"]),
        )
    )
    print(
        "Q-commit: Q1={:.0f} Q2={:.0f} Q3={:.0f} Q4={:.0f}".format(
            _safe_float(row["commit_q1"]),
            _safe_float(row["commit_q2"]),
            _safe_float(row["commit_q3"]),
            _safe_float(row["commit_q4"]),
        )
    )
    print(f"scorecard_csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
