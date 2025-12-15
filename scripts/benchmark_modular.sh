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
# Usage:
#   bash scripts/benchmark_modular.sh
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "VIO IMU PREINTEGRATION - MODULAR VERSION (WITHOUT VPS)"
echo "============================================================================"
echo ""

# Configuration
TEST_ID=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate 3D_Building_DepthAnyThingV2 2>/dev/null || {
        echo "⚠️  Warning: Could not activate conda environment"
    }
fi

# Verify modular package imports work
echo "=== Verifying VIO Module Imports ==="
python3 << 'EOF'
import vio
from vio.main_loop import VIORunner, VIOConfig
print('✅ All VIO modules imported successfully')
print(f'   Package version: {vio.__version__}')
EOF

if [ $? -ne 0 ]; then
    echo "❌ Failed to import VIO modules. Check installation."
    exit 1
fi

echo ""

# Dataset paths
CONFIG="configs/config_bell412_dataset3.yaml"
DATASET_BASE="/mnt/External_Storage/CV_team/vio_dataset/bell412_dataset3"
IMU_PATH="${DATASET_BASE}/extracted_data/imu_data/imu__data/imu.csv"
QUARRY_PATH="${DATASET_BASE}/flight_log_from_gga.csv"
IMAGES_DIR="${DATASET_BASE}/extracted_data/cam_data/camera__image_mono/images"
IMAGES_INDEX="${DATASET_BASE}/extracted_data/cam_data/camera__image_mono/images_index.csv"
MAG_PATH="${DATASET_BASE}/extracted_data/imu_data/imu__mag/vector3.csv"
DEM_PATH="/mnt/External_Storage/CV_team/vio_dataset/Copernicus_DSM_10_N45_00_W076_00_DEM.tif"
GROUND_TRUTH="${DATASET_BASE}/bell412_dataset3_frl.pos"

# Output directory
OUTPUT_DIR="benchmark_modular_${TEST_ID}/preintegration"
mkdir -p "$OUTPUT_DIR"

echo "Test Configuration:"
echo "  Test ID: ${TEST_ID}"
echo "  Output directory: ${OUTPUT_DIR}/"
echo "  Mode: Preintegration WITHOUT VPS"
echo "  Entry point: run_vio.py (modular)"
echo ""

# ============================================================================
# Run VIO using modular entry point
# ============================================================================
echo "=== Running VIO without VPS (Modular Entry Point) ==="
echo ""

START_TIME=$(date +%s.%N)

# You can use either run_vio.py or vio_vps.py - they produce the same results
# run_vio.py is cleaner but currently just calls vio_vps.run()
python3 run_vio.py \
    --config "$CONFIG" \
    --imu "$IMU_PATH" \
    --quarry "$QUARRY_PATH" \
    --images_dir "$IMAGES_DIR" \
    --images_index "$IMAGES_INDEX" \
    --mag "$MAG_PATH" \
    --dem "$DEM_PATH" \
    --ground_truth "$GROUND_TRUTH" \
    --output "$OUTPUT_DIR" \
    --img_w 1440 \
    --img_h 1080 \
    --z_state msl \
    --camera_view nadir \
    --use_magnetometer \
    --use_vio_velocity \
    --save_debug_data \
    2>&1 | tee "$OUTPUT_DIR/run.log"

END_TIME=$(date +%s.%N)
RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "✅ Test completed in ${RUNTIME}s"
echo ""

# ============================================================================
# Quick Analysis
# ============================================================================
echo "============================================================================"
echo "RESULTS (WITHOUT VPS - MODULAR)"
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
