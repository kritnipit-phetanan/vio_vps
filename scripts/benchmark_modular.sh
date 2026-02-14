#!/bin/bash
# ============================================================================
# Benchmark: VIO using Modular Components (without VPS)
# ============================================================================
# This script uses the lightweight entry point (run_vio.py) which
# imports from the modular vio/ package.
#
# Key differences from original:
#   - Uses run_vio.py instead of vio_vps.py directly
#   - Same command-line interface, but cleaner separation
#
# v3.2.0: VIOConfig Dataclass Model
#   - config.py is pure YAML reader, returns VIOConfig dataclass
#   - CLI provides only: paths, save_debug_data, save_keyframe_images
#   - camera_view now in YAML only (no CLI override)
#   - Algorithm toggles (use_magnetometer, etc.) controlled via YAML
#
# Usage:
#   bash scripts/benchmark_modular.sh
# ============================================================================

set -Eeuo pipefail  # Fail-fast on command errors (including pipelines)

echo "============================================================================"
echo "VIO IMU PREINTEGRATION - MODULAR VERSION (v3.2.0)"
echo "============================================================================"
echo "Features: VIOConfig Dataclass - Pure YAML Reader"
echo "Config: All settings from YAML, only paths from CLI"
echo ""

# Configuration
TEST_ID=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Activated virtual environment (.venv)"
else
    echo "⚠️  Warning: .venv not found, using system Python"
fi

# Verify modular package imports work
echo "=== Verifying VIO Module Imports ==="
python3 << 'EOF'
import vio
from vio.main_loop import VIORunner
from vio.config import load_config, VIOConfig
print('✅ All VIO modules imported successfully')
print(f'   Package version: {vio.__version__}')
print('   Config model: VIOConfig dataclass from YAML')
EOF

if [ $? -ne 0 ]; then
    echo "❌ Failed to import VIO modules. Check installation."
    exit 1
fi

echo ""

# Dataset paths
CONFIG="configs/config_bell412_dataset3.yaml"
DATASET_BASE="/Users/france/Downloads/vio_dataset/bell412_dataset3"
# v3.9.0: Use imu_with_ref.csv (has time_ref for unified clock)
IMU_PATH="${DATASET_BASE}/extracted_data_new/imu_data/imu__data_stamped/imu_with_ref.csv"
QUARRY_PATH="${DATASET_BASE}/flight_log_from_gga.csv"
IMAGES_DIR="${DATASET_BASE}/extracted_data_new/cam_data/camera__image_mono/images"
IMAGES_INDEX="${DATASET_BASE}/extracted_data_new/cam_data/camera__image_mono/images_index.csv"
# v3.9.0: Add timeref.csv for camera time_ref matching
TIMEREF_CSV="${DATASET_BASE}/extracted_data_new/imu_data/imu__time_ref_cam/timeref.csv"
MAG_PATH="${DATASET_BASE}/extracted_data_new/imu_data/imu__mag/vector3.csv"
DEM_PATH="${DATASET_BASE}/Copernicus_DSM_10_N45_00_W076_00_DEM.tif"
GROUND_TRUTH="${DATASET_BASE}/bell412_dataset3_frl.pos"

# Run mode
#   auto: enable optional sensors only if files exist
#   full: require IMU+GT+CAM+MAG+DEM
#   imu_cam: require IMU+GT+CAM
#   imu_only: IMU+GT only
RUN_MODE="${RUN_MODE:-auto}"
SAVE_DEBUG_DATA="${SAVE_DEBUG_DATA:-0}"  # 1 => pass --save_debug_data (heavy CSV logs)

# Output directory
OUTPUT_DIR="benchmark_modular_${TEST_ID}/preintegration"
mkdir -p "$OUTPUT_DIR"

# Baseline run for before/after diff.
# Priority: explicit BASELINE_RUN > latest previous benchmark run.
BASELINE_RUN="${BASELINE_RUN:-}"
if [ -z "$BASELINE_RUN" ]; then
    BASELINE_RUN="$(ls -dt benchmark_modular_*/preintegration 2>/dev/null | grep -v "^${OUTPUT_DIR}$" | head -n 1 || true)"
fi

# VPS Configuration (optional)
# Uncomment to enable VPS real-time processing
MBTILES_PATH="mission.mbtiles"

fail_fast() {
    echo "❌ $1"
    exit 1
}

# Required inputs
[ -f "$CONFIG" ] || fail_fast "Config not found: $CONFIG"
[ -f "$IMU_PATH" ] || fail_fast "IMU file not found: $IMU_PATH"
[ -f "$GROUND_TRUTH" ] || fail_fast "Ground truth file not found: $GROUND_TRUTH"

HAS_CAM=0
HAS_MAG=0
HAS_DEM=0
HAS_TIMEREF=0

[ -d "$IMAGES_DIR" ] && [ -f "$IMAGES_INDEX" ] && HAS_CAM=1
[ -f "$MAG_PATH" ] && HAS_MAG=1
[ -f "$DEM_PATH" ] && HAS_DEM=1
[ -f "$TIMEREF_CSV" ] && HAS_TIMEREF=1

if { [ -d "$IMAGES_DIR" ] && [ ! -f "$IMAGES_INDEX" ]; } || { [ ! -d "$IMAGES_DIR" ] && [ -f "$IMAGES_INDEX" ]; }; then
    fail_fast "Camera inputs incomplete (need both images dir and images index): $IMAGES_DIR | $IMAGES_INDEX"
fi

USE_CAM=0
USE_MAG=0
USE_DEM=0
USE_TIMEREF=0
USE_QUARRY=0

case "$RUN_MODE" in
    full)
        [ "$HAS_CAM" -eq 1 ] || fail_fast "RUN_MODE=full requires camera inputs"
        [ "$HAS_TIMEREF" -eq 1 ] || fail_fast "RUN_MODE=full requires timeref CSV"
        [ "$HAS_MAG" -eq 1 ] || fail_fast "RUN_MODE=full requires magnetometer CSV"
        [ "$HAS_DEM" -eq 1 ] || fail_fast "RUN_MODE=full requires DEM file"
        USE_CAM=1; USE_MAG=1; USE_DEM=1; USE_TIMEREF=1; USE_QUARRY=1
        ;;
    imu_cam)
        [ "$HAS_CAM" -eq 1 ] || fail_fast "RUN_MODE=imu_cam requires camera inputs"
        [ "$HAS_TIMEREF" -eq 1 ] || fail_fast "RUN_MODE=imu_cam requires timeref CSV"
        USE_CAM=1; USE_TIMEREF=1; USE_QUARRY=0
        ;;
    imu_only)
        USE_CAM=0; USE_MAG=0; USE_DEM=0; USE_TIMEREF=0; USE_QUARRY=0
        ;;
    auto)
        USE_CAM="$HAS_CAM"
        USE_MAG="$HAS_MAG"
        USE_DEM="$HAS_DEM"
        USE_QUARRY="$HAS_DEM"
        if [ "$HAS_TIMEREF" -eq 1 ] && { [ "$USE_CAM" -eq 1 ] || [ "$USE_MAG" -eq 1 ]; }; then
            USE_TIMEREF=1
        fi
        ;;
    *)
        fail_fast "Unknown RUN_MODE='$RUN_MODE' (supported: auto|full|imu_cam|imu_only)"
        ;;
esac

RUN_MODE_LABEL="IMU+GT"
[ "$USE_CAM" -eq 1 ] && RUN_MODE_LABEL="${RUN_MODE_LABEL}+CAM"
[ "$USE_MAG" -eq 1 ] && RUN_MODE_LABEL="${RUN_MODE_LABEL}+MAG"
[ "$USE_DEM" -eq 1 ] && RUN_MODE_LABEL="${RUN_MODE_LABEL}+DEM"
[ "$USE_QUARRY" -eq 1 ] && RUN_MODE_LABEL="${RUN_MODE_LABEL}+MSL"
[ -f "$MBTILES_PATH" ] && RUN_MODE_LABEL="${RUN_MODE_LABEL}+VPS"

echo "Test Configuration:"
echo "  Test ID: ${TEST_ID}"
echo "  Output directory: ${OUTPUT_DIR}/"
echo "  RUN_MODE: ${RUN_MODE}"
echo "  SAVE_DEBUG_DATA: ${SAVE_DEBUG_DATA}"
echo "  Active sensors: ${RUN_MODE_LABEL}"
[ -n "$BASELINE_RUN" ] && echo "  Baseline run: ${BASELINE_RUN}"
[ "$USE_CAM" -eq 1 ] && echo "    - Camera: ${IMAGES_DIR}"
[ "$USE_MAG" -eq 1 ] && echo "    - Magnetometer: ${MAG_PATH}"
[ "$USE_DEM" -eq 1 ] && echo "    - DEM: ${DEM_PATH}"
[ "$USE_TIMEREF" -eq 1 ] && echo "    - TimeRef: ${TIMEREF_CSV}"
[ "$USE_QUARRY" -eq 1 ] && echo "    - Quarry: ${QUARRY_PATH}"
if [ -f "$MBTILES_PATH" ]; then
    echo "    - VPS Tiles: ${MBTILES_PATH}"
fi
echo "  Entry point: run_vio.py (modular)"
echo "  Algorithm settings: From YAML config"
echo "  Performance: From YAML (fast_mode section)"
echo ""

# ============================================================================
# Run VIO using modular entry point
# ============================================================================
echo "=== Running VIO (${RUN_MODE_LABEL}) ==="
echo ""

START_TIME=$(date +%s.%N)

# CLI provides only: paths, debug flags
# v3.9.0: Add --timeref_csv for camera time_ref matching
# v4.0.0: Add --vps_tiles for VPS real-time processing
PYTHON_ARGS=(
    python3 run_vio.py
    --config "$CONFIG"
    --imu "$IMU_PATH"
    --ground_truth "$GROUND_TRUTH"
    --output "$OUTPUT_DIR"
)
if [ "$SAVE_DEBUG_DATA" = "1" ]; then
    PYTHON_ARGS+=(--save_debug_data)
fi
if [ "$USE_QUARRY" -eq 1 ]; then
    [ -f "$QUARRY_PATH" ] || fail_fast "RUN_MODE=${RUN_MODE} requires quarry file: $QUARRY_PATH"
    PYTHON_ARGS+=(--quarry "$QUARRY_PATH")
fi

if [ "$USE_CAM" -eq 1 ]; then
    PYTHON_ARGS+=(--images_dir "$IMAGES_DIR" --images_index "$IMAGES_INDEX")
fi
if [ "$USE_MAG" -eq 1 ]; then
    PYTHON_ARGS+=(--mag "$MAG_PATH")
fi
if [ "$USE_DEM" -eq 1 ]; then
    PYTHON_ARGS+=(--dem "$DEM_PATH")
fi
if [ "$USE_TIMEREF" -eq 1 ]; then
    PYTHON_ARGS+=(--timeref_csv "$TIMEREF_CSV")
fi

# Add MBTiles path if VPS is enabled
if [ -f "$MBTILES_PATH" ]; then
    PYTHON_ARGS+=(--vps_tiles "$MBTILES_PATH")
    echo "✅ VPS enabled with MBTiles: $MBTILES_PATH"
fi

if ! "${PYTHON_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/run.log"; then
    fail_fast "VIO run failed (see $OUTPUT_DIR/run.log)"
fi

END_TIME=$(date +%s.%N)
RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "✅ Test completed in ${RUNTIME}s"
echo ""

# ============================================================================
# Quick Analysis
# ============================================================================
echo "============================================================================"
echo "RESULTS (v3.2.0 - VIOConfig Dataclass Model)"
echo "============================================================================"
echo ""

# Function to extract error statistics
analyze_errors() {
    local csv_file="$1"
    
    if [ ! -f "$csv_file" ]; then
        echo "❌ error_log.csv not found"
        return
    fi
    
    # Use Python to compute RMSE
    python3 - <<EOF
import pandas as pd
import numpy as np

df = pd.read_csv("$csv_file")
if len(df) > 50:
    df = df.iloc[50:]  # Skip initial convergence

pos_err = np.sqrt((df['pos_error_m']**2).mean())
vel_err = np.sqrt((df['vel_error_m_s']**2).mean())
alt_err = df['alt_error_m'].abs().mean()
final_pos_err = df['pos_error_m'].iloc[-1]

print(f"Position RMSE: {pos_err:.3f} m")
print(f"Altitude Mean Error: {alt_err:.3f} m")
print(f"Velocity RMSE: {vel_err:.3f} m/s")
print(f"Final Position Error: {final_pos_err:.3f} m")
print()

# Axis breakdown
print(f"E mean error: {df['pos_error_E'].mean():.3f} m")
print(f"N mean error: {df['pos_error_N'].mean():.3f} m")
print(f"U mean error: {df['pos_error_U'].mean():.3f} m")
EOF
}

echo "=== Accuracy Analysis ==="
analyze_errors "$OUTPUT_DIR/error_log.csv"
echo ""

CURRENT_SUMMARY="$OUTPUT_DIR/benchmark_health_summary.csv"
BASELINE_SUMMARY=""
if [ -n "$BASELINE_RUN" ] && [ -f "$BASELINE_RUN/benchmark_health_summary.csv" ]; then
    BASELINE_SUMMARY="$BASELINE_RUN/benchmark_health_summary.csv"
elif [ -n "$BASELINE_RUN" ]; then
    BASELINE_SUMMARY="$OUTPUT_DIR/_baseline_health_summary_fallback.csv"
    python3 - <<EOF
import os
import re
import numpy as np
import pandas as pd

base_dir = "$BASELINE_RUN"
out_csv = "$BASELINE_SUMMARY"

projection_count = np.nan
first_projection_time = np.nan
pcond_max = np.nan
pmax_max = np.nan
cov_large_rate = np.nan
pos_rmse = np.nan
final_pos_err = np.nan
final_alt_err = np.nan

run_log = os.path.join(base_dir, "run.log")
if os.path.isfile(run_log):
    with open(run_log, "r", errors="ignore") as f:
        lines = f.readlines()
    projection_count = float(sum(1 for line in lines if ("Eigenvalue projection" in line or "[EKF-COND]" in line)))
    for line in lines:
        m = re.search(r"\\[EKF-COND\\]\\s+t=([0-9.+-eE]+)", line)
        if m:
            first_projection_time = float(m.group(1))
            break

cov_csv = os.path.join(base_dir, "cov_health.csv")
if os.path.isfile(cov_csv):
    cov_df = pd.read_csv(cov_csv)
    if len(cov_df) > 0:
        pcond_max = float(pd.to_numeric(cov_df["p_cond"], errors="coerce").max())
        pmax_max = float(pd.to_numeric(cov_df["p_max"], errors="coerce").max())
        cov_large_rate = float(pd.to_numeric(cov_df["large_flag"], errors="coerce").mean())

err_csv = os.path.join(base_dir, "error_log.csv")
if os.path.isfile(err_csv):
    err_df = pd.read_csv(err_csv)
    if len(err_df) > 0:
        pos = pd.to_numeric(err_df["pos_error_m"], errors="coerce").to_numpy(dtype=float)
        alt = pd.to_numeric(err_df["alt_error_m"], errors="coerce").to_numpy(dtype=float)
        pos_rmse = float(np.sqrt(np.nanmean(pos ** 2)))
        final_pos_err = float(pos[-1])
        final_alt_err = float(alt[-1])

run_id = os.path.basename(os.path.dirname(os.path.normpath(base_dir))) if os.path.basename(os.path.normpath(base_dir)) == "preintegration" else os.path.basename(os.path.normpath(base_dir))
with open(out_csv, "w", newline="") as f:
    f.write("run_id,projection_count,first_projection_time,pcond_max,pmax_max,cov_large_rate,pos_rmse,final_pos_err,final_alt_err\\n")
    f.write(f"{run_id},{projection_count},{first_projection_time},{pcond_max},{pmax_max},{cov_large_rate},{pos_rmse},{final_pos_err},{final_alt_err}\\n")
EOF
    if [ ! -f "$BASELINE_SUMMARY" ]; then
        BASELINE_SUMMARY=""
    fi
fi

echo "=== Conditioning Health Summary ==="
if [ -f "$CURRENT_SUMMARY" ]; then
    python3 - <<EOF
import pandas as pd
import numpy as np

cur = pd.read_csv("$CURRENT_SUMMARY")
if len(cur) == 0:
    print("❌ benchmark_health_summary.csv is empty")
else:
    row = cur.iloc[-1]
    print(f"projection_count   : {int(row['projection_count'])}")
    print(f"first_projection_t : {row['first_projection_time']:.3f} s")
    print(f"pcond_max          : {row['pcond_max']:.3e}")
    print(f"pmax_max           : {row['pmax_max']:.3e}")
    print(f"cov_large_rate     : {row['cov_large_rate']:.4f}")
    print(f"pos_rmse           : {row['pos_rmse']:.3f} m")
    print(f"final_pos_err      : {row['final_pos_err']:.3f} m")
    print(f"final_alt_err      : {row['final_alt_err']:.3f} m")
EOF
else
    echo "❌ Missing $CURRENT_SUMMARY"
fi
echo ""

echo "=== Before/After vs Baseline ==="
if [ -f "$CURRENT_SUMMARY" ] && [ -n "$BASELINE_SUMMARY" ]; then
    python3 - <<EOF
import pandas as pd
import numpy as np

cur_df = pd.read_csv("$CURRENT_SUMMARY")
base_df = pd.read_csv("$BASELINE_SUMMARY")
if len(cur_df) == 0 or len(base_df) == 0:
    print("Baseline/current summary has no data rows; skipping before/after diff.")
    raise SystemExit(0)

cur = cur_df.iloc[-1]
base = base_df.iloc[-1]
metrics = [
    "projection_count",
    "first_projection_time",
    "pcond_max",
    "pmax_max",
    "cov_large_rate",
    "pos_rmse",
    "final_pos_err",
    "final_alt_err",
]
print(f"Baseline: $BASELINE_RUN")
for m in metrics:
    c = float(cur[m])
    b = float(base[m])
    if np.isfinite(b) and abs(b) > 1e-12:
        pct = 100.0 * (c - b) / abs(b)
        print(f"{m:20s}  base={b: .6e}  cur={c: .6e}  delta={pct:+7.2f}%")
    else:
        print(f"{m:20s}  base={b: .6e}  cur={c: .6e}  delta=   n/a")
EOF
else
    echo "No baseline summary found; skipping before/after diff."
fi
echo ""

echo "=== Runtime Profiling (quick) ==="
python3 - <<EOF
import os
import pandas as pd
import numpy as np

out_dir = "$OUTPUT_DIR"
inf_csv = os.path.join(out_dir, "inference_log.csv")
pose_csv = os.path.join(out_dir, "pose.csv")
vps_attempts_csv = os.path.join(out_dir, "debug_vps_attempts.csv")
vps_profile_csv = os.path.join(out_dir, "debug_vps_profile.csv")

def _read_last_pose_time(path):
    if not os.path.isfile(path):
        return float("nan")
    try:
        p = pd.read_csv(path)
        if len(p) == 0:
            return float("nan")
        return float(pd.to_numeric(p.iloc[-1, 0], errors="coerce"))
    except Exception:
        return float("nan")

if os.path.isfile(inf_csv):
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

if os.path.isfile(vps_attempts_csv):
    a = pd.read_csv(vps_attempts_csv)
    if len(a) > 0 and "processing_time_ms" in a.columns:
        t = pd.to_numeric(a["processing_time_ms"], errors="coerce").to_numpy(dtype=float)
        t = t[np.isfinite(t)]
        if t.size:
            print(f"VPS attempts     : {len(a)}")
            print(f"VPS total        : {np.sum(t):.2f} ms")
            print(f"VPS avg / p95    : {np.mean(t):.2f} / {np.percentile(t,95):.2f} ms")

if os.path.isfile(vps_profile_csv):
    vp = pd.read_csv(vps_profile_csv)
    if len(vp) > 0:
        ok = vp[vp["success"] == 1] if "success" in vp.columns else vp
        src = ok if len(ok) > 0 else vp
        cols = ["tile_ms", "preprocess_ms", "match_ms", "pose_ms", "total_ms"]
        print("VPS stage means  :", ", ".join(
            f"{c}={pd.to_numeric(src[c], errors='coerce').mean():.2f}ms"
            for c in cols if c in src.columns
        ))

# CSV I/O footprint (proxy for logging overhead)
csv_sizes = []
for name in os.listdir(out_dir):
    if name.endswith(".csv"):
        path = os.path.join(out_dir, name)
        try:
            csv_sizes.append((name, os.path.getsize(path)))
        except OSError:
            pass
csv_sizes.sort(key=lambda x: x[1], reverse=True)
if csv_sizes:
    top = csv_sizes[:8]
    print("Top CSV size     :")
    for n, s in top:
        print(f"  {n:28s} {s/1024/1024:8.2f} MB")
EOF
echo ""

echo "=== Sensor Health by Phase (accept-rate / NIS EWMA) ==="
if [ -f "$OUTPUT_DIR/sensor_health.csv" ] && [ -f "$OUTPUT_DIR/adaptive_debug.csv" ]; then
    python3 - <<EOF
import pandas as pd
import numpy as np

sensor_df = pd.read_csv("$OUTPUT_DIR/sensor_health.csv")
adaptive_df = pd.read_csv("$OUTPUT_DIR/adaptive_debug.csv")[["t", "phase"]]

if len(sensor_df) == 0 or len(adaptive_df) == 0:
    print("No sensor/adaptive rows for phase summary")
else:
    sensor_df = sensor_df.sort_values("t")
    adaptive_df = adaptive_df.sort_values("t")
    merged = pd.merge_asof(sensor_df, adaptive_df, on="t", direction="backward")
    merged["phase"] = merged["phase"].fillna(-1).astype(int)
    summary = (
        merged.groupby(["sensor", "phase"], dropna=False)
        .agg(
            samples=("accepted", "count"),
            accept_rate=("accepted", "mean"),
            nis_ewma_mean=("nis_ewma", "mean"),
            nis_ewma_last=("nis_ewma", "last"),
        )
        .reset_index()
        .sort_values(["sensor", "phase"])
    )
    out_csv = "$OUTPUT_DIR/sensor_phase_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(summary.to_string(index=False))
    print(f"saved: {out_csv}")

    baseline_sensor = "$BASELINE_RUN/sensor_health.csv"
    baseline_adaptive = "$BASELINE_RUN/adaptive_debug.csv"
    if "$BASELINE_RUN" and pd.notna("$BASELINE_RUN") and \
       baseline_sensor and baseline_adaptive and \
       __import__("os").path.isfile(baseline_sensor) and __import__("os").path.isfile(baseline_adaptive):
        b_s = pd.read_csv(baseline_sensor)
        b_a = pd.read_csv(baseline_adaptive)[["t", "phase"]]
        if len(b_s) > 0 and len(b_a) > 0:
            b_s = b_s.sort_values("t")
            b_a = b_a.sort_values("t")
            b_m = pd.merge_asof(b_s, b_a, on="t", direction="backward")
            b_m["phase"] = b_m["phase"].fillna(-1).astype(int)
            b_sum = (
                b_m.groupby(["sensor", "phase"], dropna=False)
                .agg(
                    accept_rate_base=("accepted", "mean"),
                    nis_ewma_mean_base=("nis_ewma", "mean"),
                )
                .reset_index()
            )
            joined = summary.merge(b_sum, on=["sensor", "phase"], how="left")
            joined["accept_rate_delta"] = joined["accept_rate"] - joined["accept_rate_base"]
            joined["nis_ewma_delta"] = joined["nis_ewma_mean"] - joined["nis_ewma_mean_base"]
            print("\\nBaseline delta (sensor+phase):")
            print(joined[["sensor","phase","accept_rate_delta","nis_ewma_delta"]].to_string(index=False))
EOF
else
    echo "Missing sensor_health.csv or adaptive_debug.csv"
fi
echo ""

echo "=== Runtime ==="
echo "Time: ${RUNTIME}s"
echo ""

echo "============================================================================"
echo "OUTPUT FILES"
echo "============================================================================"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "  - pose.csv         : Full trajectory"
echo "  - error_log.csv    : Error statistics"
echo "  - cov_health.csv   : Covariance health timeline"
echo "  - conditioning_events.csv : Conditioning fallback events"
echo "  - benchmark_health_summary.csv : Run health one-line summary"
echo "  - sensor_phase_summary.csv : accept-rate/NIS summary by sensor+phase"
echo "  - run.log          : Full debug output"
echo ""
echo "Modular VIO package: vio/"
ls -la vio/*.py | head -20
echo "============================================================================"

exit 0
