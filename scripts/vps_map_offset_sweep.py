#!/usr/bin/env python3
"""
Sweep VPS map EN offsets over a short diagnostic window and summarize matcher hits.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import yaml


REPO_DIR = Path(__file__).resolve().parents[1]
DATASET_BASE = Path("/Users/france/Downloads/vio_dataset/bell412_dataset3")
DEFAULT_CONFIG = REPO_DIR / "configs" / "config_bell412_dataset3_stageG_base.yaml"
DEFAULT_OUTPUT_ROOT = REPO_DIR / "outputs"


def _python_bin() -> str:
    venv_python = REPO_DIR / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.exists() else Path(sys.executable))


def _offsets_from_spec(spec: str) -> list[float]:
    vals: list[float] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("empty offset spec")
    return vals


def _set_nested(cfg: dict, path: Iterable[str], value) -> None:
    cur = cfg
    parts = list(path)
    for key in parts[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _prepare_config(base_cfg_path: Path, out_cfg_path: Path, *, offset_e: float, offset_n: float,
                    start_t: float, end_t: float, max_frames: int) -> None:
    with base_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid yaml root in {base_cfg_path}")

    updates = {
        ("vps", "map_offset_e_m"): float(offset_e),
        ("vps", "map_offset_n_m"): float(offset_n),
        ("vps", "debug_match_dump_enable"): True,
        ("vps", "debug_match_dump_start_t"): float(start_t),
        ("vps", "debug_match_dump_end_t"): float(end_t),
        ("vps", "debug_match_dump_max_frames"): int(max_frames),
        ("vps", "debug_match_dump_failures"): True,
        ("vps", "debug_match_dump_successes"): True,
    }
    for path, value in updates.items():
        _set_nested(cfg, path, value)

    out_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with out_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_single(config_path: Path, output_dir: Path, *, stop_t: float) -> None:
    cmd = [
        _python_bin(),
        "run_vio.py",
        "--config", str(config_path),
        "--imu", str(DATASET_BASE / "extracted_data_new/imu_data/imu__data_stamped/imu_with_ref.csv"),
        "--quarry", str(DATASET_BASE / "flight_log_from_gga.csv"),
        "--output", str(output_dir),
        "--images_dir", str(DATASET_BASE / "extracted_data_new/cam_data/camera__image_mono/images"),
        "--images_index", str(DATASET_BASE / "extracted_data_new/cam_data/camera__image_mono/images_index.csv"),
        "--timeref_csv", str(DATASET_BASE / "extracted_data_new/imu_data/imu__time_ref_cam/timeref.csv"),
        "--timeref_pps_csv", str(DATASET_BASE / "extracted_data_new/imu_data/imu__time_ref_pps/timeref.csv"),
        "--mag", str(DATASET_BASE / "extracted_data_new/imu_data/imu__mag/vector3.csv"),
        "--dem", str(DATASET_BASE / "Copernicus_DSM_10_N45_00_W076_00_DEM.tif"),
        "--ground_truth", str(DATASET_BASE / "bell412_dataset3_frl.pos"),
        "--vps_tiles", str(REPO_DIR / "mission.mbtiles"),
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["VIO_STOP_TIME"] = f"{float(stop_t):.6f}"
    env.setdefault("VIO_RUNTIME_VERBOSITY", "release")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run.log"
    with run_log.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_DIR),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"run failed for {output_dir.name} rc={proc.returncode}")


def _summarize_run(run_dir: Path) -> dict:
    summary = {
        "max_inliers": -1,
        "max_score": float("-inf"),
        "best_t": float("nan"),
        "rows": 0,
        "rows_inliers_gt0": 0,
        "best_reason": "",
        "best_img": "",
    }
    csv_path = run_dir / "vps_reloc_summary.csv"
    if not csv_path.exists():
        return summary
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary["rows"] += 1
            try:
                inliers = int(float(row.get("best_inliers", "0") or 0))
            except Exception:
                inliers = 0
            try:
                score = float(row.get("best_score", "nan"))
            except Exception:
                score = float("nan")
            if inliers > 0:
                summary["rows_inliers_gt0"] += 1
            if (inliers > int(summary["max_inliers"])) or (
                inliers == int(summary["max_inliers"]) and score > float(summary["max_score"])
            ):
                summary["max_inliers"] = inliers
                summary["max_score"] = score
                try:
                    summary["best_t"] = float(row.get("t", "nan"))
                except Exception:
                    summary["best_t"] = float("nan")
                summary["best_reason"] = str(row.get("reason", ""))
    debug_dir = run_dir / "debug_vps_matches"
    if debug_dir.exists():
        imgs = sorted(debug_dir.glob("*.jpg"))
        if imgs:
            summary["best_img"] = str(imgs[0])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid-sweep VPS map offset on a short diagnostic window.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--offsets-e", default="-40,-20,0,20,40")
    parser.add_argument("--offsets-n", default="-40,-20,0,20,40")
    parser.add_argument("--start-t", type=float, default=1750.0)
    parser.add_argument("--end-t", type=float, default=1753.0)
    parser.add_argument("--max-frames", type=int, default=4)
    args = parser.parse_args()

    base_cfg = Path(args.config).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).resolve() if args.output_root else (DEFAULT_OUTPUT_ROOT / f"vps_offset_sweep_{stamp}")
    temp_cfg_dir = output_root / "temp_configs"
    output_root.mkdir(parents=True, exist_ok=True)

    offsets_e = _offsets_from_spec(args.offsets_e)
    offsets_n = _offsets_from_spec(args.offsets_n)

    rows: list[dict] = []
    total = len(offsets_e) * len(offsets_n)
    idx = 0
    for off_e in offsets_e:
        for off_n in offsets_n:
            idx += 1
            tag = f"E{off_e:+.0f}_N{off_n:+.0f}".replace("+", "p").replace("-", "m")
            run_dir = output_root / tag
            cfg_path = temp_cfg_dir / f"{tag}.yaml"
            print(f"[{idx:02d}/{total:02d}] sweep offset E={off_e:+.1f} N={off_n:+.1f} -> {run_dir}")
            _prepare_config(
                base_cfg,
                cfg_path,
                offset_e=off_e,
                offset_n=off_n,
                start_t=float(args.start_t),
                end_t=float(args.end_t),
                max_frames=int(args.max_frames),
            )
            _run_single(cfg_path, run_dir, stop_t=float(args.end_t))
            result = _summarize_run(run_dir)
            row = {
                "offset_e_m": off_e,
                "offset_n_m": off_n,
                "run_dir": str(run_dir),
                **result,
            }
            rows.append(row)
            print(
                "  "
                f"max_inliers={row['max_inliers']} "
                f"max_score={row['max_score']:.3f} "
                f"best_t={row['best_t']:.3f} "
                f"reason={row['best_reason']}"
            )

    rows.sort(key=lambda r: (int(r["max_inliers"]), float(r["max_score"])), reverse=True)
    summary_csv = output_root / "sweep_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "offset_e_m",
                "offset_n_m",
                "run_dir",
                "max_inliers",
                "max_score",
                "best_t",
                "rows",
                "rows_inliers_gt0",
                "best_reason",
                "best_img",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    best = rows[0] if rows else None
    if best is not None:
        print("\n=== Best Offset Candidate ===")
        print(
            f"E={float(best['offset_e_m']):+.1f}m "
            f"N={float(best['offset_n_m']):+.1f}m | "
            f"max_inliers={int(best['max_inliers'])} "
            f"max_score={float(best['max_score']):.3f} "
            f"best_t={float(best['best_t']):.3f}"
        )
        print(f"run_dir={best['run_dir']}")
        if best.get("best_img"):
            print(f"best_img={best['best_img']}")
    print(f"summary_csv={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
