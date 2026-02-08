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

set -e  # Exit on error

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

# Output directory
OUTPUT_DIR="benchmark_modular_${TEST_ID}/preintegration"
mkdir -p "$OUTPUT_DIR"

# VPS Configuration (optional)
# Uncomment to enable VPS real-time processing
MBTILES_PATH="mission.mbtiles"
VPS_ENABLED=true  # Set to true to enable VPS

echo "Test Configuration:"
echo "  Test ID: ${TEST_ID}"
echo "  Output directory: ${OUTPUT_DIR}/"
if [ "$VPS_ENABLED" = true ] && [ -f "$MBTILES_PATH" ]; then
    echo "  Mode: Preintegration WITH VPS (Real-time)"
    echo "  MBTiles: ${MBTILES_PATH}"
else
    echo "  Mode: Preintegration WITHOUT VPS"
fi
echo "  Entry point: run_vio.py (modular)"
echo "  Algorithm settings: From YAML config"
echo "  Performance: From YAML (fast_mode section)"
echo ""

# ============================================================================
# Run VIO using modular entry point
# ============================================================================
echo "=== Running VIO without VPS (Modular Entry Point) ==="
echo ""

START_TIME=$(date +%s.%N)

# CLI provides only: paths, debug flags
# v3.9.0: Add --timeref_csv for camera time_ref matching
# v4.0.0: Add --vps_tiles for VPS real-time processing
PYTHON_CMD="python3 run_vio.py \
    --config \"$CONFIG\" \
    --imu \"$IMU_PATH\" \
    --quarry \"$QUARRY_PATH\" \
    --images_dir \"$IMAGES_DIR\" \
    --images_index \"$IMAGES_INDEX\" \
    --timeref_csv \"$TIMEREF_CSV\" \
    --mag \"$MAG_PATH\" \
    --dem \"$DEM_PATH\" \
    --ground_truth \"$GROUND_TRUTH\" \
    --output \"$OUTPUT_DIR\" \
    --save_debug_data"

# Add MBTiles path if VPS is enabled
if [ "$VPS_ENABLED" = true ] && [ -f "$MBTILES_PATH" ]; then
    PYTHON_CMD="$PYTHON_CMD --vps_tiles \"$MBTILES_PATH\""
    echo "✅ VPS enabled with MBTiles: $MBTILES_PATH"
fi

eval "$PYTHON_CMD" 2>&1 | tee "$OUTPUT_DIR/run.log"

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

echo "=== Runtime ==="
echo "Time: ${RUNTIME}s"
echo ""

echo "============================================================================"
echo "OUTPUT FILES"
echo "============================================================================"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "  - pose.csv         : Full trajectory"
echo "  - error_log.csv    : Error statistics"
echo "  - run.log          : Full debug output"
echo ""
echo "Modular VIO package: vio/"
ls -la vio/*.py | head -20
echo "============================================================================"

exit 0
