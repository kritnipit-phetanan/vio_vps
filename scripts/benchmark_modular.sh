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

# Ensure script is executed by bash (not sh/zsh/source)
if [ -z "${BASH_VERSION:-}" ]; then
    exec /usr/bin/env bash "$0" "$@"
fi
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "❌ Please run (not source): bash ${BASH_SOURCE[0]}"
    return 1 2>/dev/null || exit 1
fi

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
CONFIG="${CONFIG:-configs/config_bell412_dataset3.yaml}"
DATASET_BASE="/Users/france/Downloads/vio_dataset/bell412_dataset3"
# v3.9.0: Use imu_with_ref.csv (has time_ref for unified clock)
IMU_PATH="${DATASET_BASE}/extracted_data_new/imu_data/imu__data_stamped/imu_with_ref.csv"
QUARRY_PATH="${DATASET_BASE}/flight_log_from_gga.csv"
IMAGES_DIR="${DATASET_BASE}/extracted_data_new/cam_data/camera__image_mono/images"
IMAGES_INDEX="${DATASET_BASE}/extracted_data_new/cam_data/camera__image_mono/images_index.csv"
# v3.9.0: Add timeref.csv for camera time_ref matching
TIMEREF_CSV="${DATASET_BASE}/extracted_data_new/imu_data/imu__time_ref_cam/timeref.csv"
TIMEREF_PPS_CSV="${DATASET_BASE}/extracted_data_new/imu_data/imu__time_ref_pps/timeref.csv"
MAG_PATH="${DATASET_BASE}/extracted_data_new/imu_data/imu__mag/vector3.csv"
DEM_PATH="${DATASET_BASE}/Copernicus_DSM_10_N45_00_W076_00_DEM.tif"
GROUND_TRUTH="${DATASET_BASE}/bell412_dataset3_frl.pos"
# VPS Configuration (optional)
MBTILES_PATH="mission.mbtiles"
HAS_VPS=0
[ -f "$MBTILES_PATH" ] && HAS_VPS=1

# Run mode
#   auto: enable optional sensors only if files exist
#   full: require IMU+GT+CAM+MAG+DEM
#   imu_cam: require IMU+GT+CAM
#   imu_only: IMU+GT only
RUN_MODE="${RUN_MODE:-auto}"
SAVE_DEBUG_DATA="${SAVE_DEBUG_DATA:-1}"  # 1 => pass --save_debug_data (heavy CSV logs)
SAVE_KEYFRAME_IMAGES="${SAVE_KEYFRAME_IMAGES:-1}"  # 1 => pass --save_keyframe_images (keyframe JPGs)

# Output directory
OUTPUT_DIR="benchmark_modular_${TEST_ID}/preintegration"
mkdir -p "$OUTPUT_DIR"

# Baseline run for before/after diff.
# Priority: explicit BASELINE_RUN > scripts/baseline_run.txt > latest previous benchmark run.
BASELINE_FILE="$SCRIPT_DIR/baseline_run.txt"
BASELINE_RUN="${BASELINE_RUN:-}"
if [ -z "$BASELINE_RUN" ] && [ -f "$BASELINE_FILE" ]; then
    BASELINE_RUN="$(head -n 1 "$BASELINE_FILE" | tr -d '\r')"
fi
if [ -z "$BASELINE_RUN" ]; then
    BASELINE_RUN="$(
        python3 - <<EOF
import csv
import glob
import os

exclude = os.path.normpath("$OUTPUT_DIR")
cands = sorted(
    glob.glob("benchmark_modular_*/preintegration"),
    key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0,
    reverse=True,
)

def has_nonempty_summary(run_dir: str) -> bool:
    p = os.path.join(run_dir, "benchmark_health_summary.csv")
    if not os.path.isfile(p):
        return False
    try:
        with open(p, newline="") as f:
            rows = list(csv.reader(f))
        return len(rows) >= 2
    except Exception:
        return False

def has_error_log(run_dir: str) -> bool:
    p = os.path.join(run_dir, "error_log.csv")
    return os.path.isfile(p) and os.path.getsize(p) > 0

for p in cands:
    if os.path.normpath(p) == exclude:
        continue
    if has_nonempty_summary(p):
        print(p)
        raise SystemExit(0)

for p in cands:
    if os.path.normpath(p) == exclude:
        continue
    if has_error_log(p):
        print(p)
        raise SystemExit(0)
EOF
    )"
fi
if [ -n "$BASELINE_RUN" ] && [ ! -d "$BASELINE_RUN" ]; then
    echo "⚠️  Baseline path not found, ignore: $BASELINE_RUN"
    BASELINE_RUN=""
fi

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
HAS_TIMEREF_PPS=0

[ -d "$IMAGES_DIR" ] && [ -f "$IMAGES_INDEX" ] && HAS_CAM=1
[ -f "$MAG_PATH" ] && HAS_MAG=1
[ -f "$DEM_PATH" ] && HAS_DEM=1
[ -f "$TIMEREF_CSV" ] && HAS_TIMEREF=1
[ -f "$TIMEREF_PPS_CSV" ] && HAS_TIMEREF_PPS=1

if { [ -d "$IMAGES_DIR" ] && [ ! -f "$IMAGES_INDEX" ]; } || { [ ! -d "$IMAGES_DIR" ] && [ -f "$IMAGES_INDEX" ]; }; then
    fail_fast "Camera inputs incomplete (need both images dir and images index): $IMAGES_DIR | $IMAGES_INDEX"
fi

USE_CAM=0
USE_MAG=0
USE_DEM=0
USE_TIMEREF=0
USE_TIMEREF_PPS=0
USE_QUARRY=0
USE_VPS=0

case "$RUN_MODE" in
    full)
        [ "$HAS_CAM" -eq 1 ] || fail_fast "RUN_MODE=full requires camera inputs"
        [ "$HAS_TIMEREF" -eq 1 ] || fail_fast "RUN_MODE=full requires timeref CSV"
        [ "$HAS_MAG" -eq 1 ] || fail_fast "RUN_MODE=full requires magnetometer CSV"
        [ "$HAS_DEM" -eq 1 ] || fail_fast "RUN_MODE=full requires DEM file"
        USE_CAM=1; USE_MAG=1; USE_DEM=1; USE_TIMEREF=1; USE_QUARRY=1
        [ "$HAS_VPS" -eq 1 ] && USE_VPS=1
        [ "$HAS_TIMEREF_PPS" -eq 1 ] && USE_TIMEREF_PPS=1
        ;;
    imu_cam)
        [ "$HAS_CAM" -eq 1 ] || fail_fast "RUN_MODE=imu_cam requires camera inputs"
        [ "$HAS_TIMEREF" -eq 1 ] || fail_fast "RUN_MODE=imu_cam requires timeref CSV"
        USE_CAM=1; USE_TIMEREF=1; USE_QUARRY=0
        [ "$HAS_VPS" -eq 1 ] && USE_VPS=1
        [ "$HAS_TIMEREF_PPS" -eq 1 ] && USE_TIMEREF_PPS=1
        ;;
    imu_only)
        USE_CAM=0; USE_MAG=0; USE_DEM=0; USE_TIMEREF=0; USE_TIMEREF_PPS=0; USE_QUARRY=0; USE_VPS=0
        ;;
    auto)
        USE_CAM="$HAS_CAM"
        USE_MAG="$HAS_MAG"
        USE_DEM="$HAS_DEM"
        USE_QUARRY="$HAS_DEM"
        if [ "$HAS_VPS" -eq 1 ] && [ "$USE_CAM" -eq 1 ]; then
            USE_VPS=1
        fi
        if [ "$HAS_TIMEREF" -eq 1 ] && { [ "$USE_CAM" -eq 1 ] || [ "$USE_MAG" -eq 1 ]; }; then
            USE_TIMEREF=1
        fi
        if [ "$USE_TIMEREF" -eq 1 ] && [ "$HAS_TIMEREF_PPS" -eq 1 ]; then
            USE_TIMEREF_PPS=1
        fi
        ;;
    *)
        fail_fast "Unknown RUN_MODE='$RUN_MODE' (supported: auto|full|imu_cam|imu_only)"
        ;;
esac

case "$SAVE_DEBUG_DATA" in
    0|1) ;;
    *) fail_fast "SAVE_DEBUG_DATA must be 0 or 1 (got '$SAVE_DEBUG_DATA')" ;;
esac
case "$SAVE_KEYFRAME_IMAGES" in
    0|1) ;;
    *) fail_fast "SAVE_KEYFRAME_IMAGES must be 0 or 1 (got '$SAVE_KEYFRAME_IMAGES')" ;;
esac

if [ "$SAVE_KEYFRAME_IMAGES" = "1" ] && [ "$USE_CAM" -ne 1 ]; then
    echo "⚠️  SAVE_KEYFRAME_IMAGES=1 ignored (camera input not enabled in RUN_MODE=${RUN_MODE})"
    SAVE_KEYFRAME_IMAGES=0
fi
if [ "$USE_TIMEREF" -eq 1 ] && [ "$USE_TIMEREF_PPS" -eq 0 ]; then
    echo "⚠️  TimeRef PPS CSV not found; running without --timeref_pps_csv"
fi

RUN_MODE_LABEL="IMU+GT"
[ "$USE_CAM" -eq 1 ] && RUN_MODE_LABEL="${RUN_MODE_LABEL}+CAM"
[ "$USE_MAG" -eq 1 ] && RUN_MODE_LABEL="${RUN_MODE_LABEL}+MAG"
[ "$USE_DEM" -eq 1 ] && RUN_MODE_LABEL="${RUN_MODE_LABEL}+DEM"
[ "$USE_QUARRY" -eq 1 ] && RUN_MODE_LABEL="${RUN_MODE_LABEL}+MSL"
[ "$USE_VPS" -eq 1 ] && RUN_MODE_LABEL="${RUN_MODE_LABEL}+VPS"

echo "Test Configuration:"
echo "  Test ID: ${TEST_ID}"
echo "  Output directory: ${OUTPUT_DIR}/"
echo "  RUN_MODE: ${RUN_MODE}"
echo "  SAVE_DEBUG_DATA: ${SAVE_DEBUG_DATA}"
echo "  SAVE_KEYFRAME_IMAGES: ${SAVE_KEYFRAME_IMAGES}"
echo "  Active sensors: ${RUN_MODE_LABEL}"
[ -n "$BASELINE_RUN" ] && echo "  Baseline run: ${BASELINE_RUN}"
[ "$USE_CAM" -eq 1 ] && echo "    - Camera: ${IMAGES_DIR}"
[ "$USE_MAG" -eq 1 ] && echo "    - Magnetometer: ${MAG_PATH}"
[ "$USE_DEM" -eq 1 ] && echo "    - DEM: ${DEM_PATH}"
[ "$USE_TIMEREF" -eq 1 ] && echo "    - TimeRef: ${TIMEREF_CSV}"
[ "$USE_TIMEREF_PPS" -eq 1 ] && echo "    - TimeRef PPS: ${TIMEREF_PPS_CSV}"
[ "$USE_QUARRY" -eq 1 ] && echo "    - Quarry: ${QUARRY_PATH}"
if [ "$USE_VPS" -eq 1 ]; then
    echo "    - VPS Tiles: ${MBTILES_PATH}"
fi
echo "  Entry point: run_vio.py (modular)"
echo "  Algorithm settings: From YAML config"
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
if [ "$SAVE_KEYFRAME_IMAGES" = "1" ]; then
    PYTHON_ARGS+=(--save_keyframe_images)
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
if [ "$USE_TIMEREF_PPS" -eq 1 ]; then
    PYTHON_ARGS+=(--timeref_pps_csv "$TIMEREF_PPS_CSV")
fi

# Add MBTiles path if VPS is enabled for selected mode
if [ "$USE_VPS" -eq 1 ]; then
    PYTHON_ARGS+=(--vps_tiles "$MBTILES_PATH")
    echo "✅ VPS enabled with MBTiles: $MBTILES_PATH"
elif [ "$HAS_VPS" -eq 1 ]; then
    echo "ℹ️  VPS disabled for RUN_MODE=${RUN_MODE} (camera/VPS policy)"
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
# Quick Analysis (delegated to Python scripts)
# ============================================================================
echo "============================================================================"
echo "RESULTS (v3.2.0 - VIOConfig Dataclass Model)"
echo "============================================================================"
echo ""

ANALYZE_ARGS=(python3 scripts/analyze_benchmark.py --output_dir "$OUTPUT_DIR")
if [ -n "$BASELINE_RUN" ]; then
    ANALYZE_ARGS+=(--baseline_run "$BASELINE_RUN")
fi
"${ANALYZE_ARGS[@]}"

echo ""
COMPARE_ARGS=(python3 scripts/compare_benchmark.py --output_dir "$OUTPUT_DIR")
if [ -n "$BASELINE_RUN" ]; then
    COMPARE_ARGS+=(--baseline_run "$BASELINE_RUN")
fi
"${COMPARE_ARGS[@]}"

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
echo "  - accuracy_first_summary.csv : Accuracy-first metrics"
echo "  - spectacular_style_metrics.csv : CEP + 4DoF-aligned ATE style metrics"
echo "  - sensor_phase_summary.csv : accept-rate/NIS summary by sensor+phase"
echo "  - run.log          : Full debug output"
echo ""
echo "Modular VIO package: vio/"
ls -la vio/*.py | head -20
echo "============================================================================"

exit 0
