#!/bin/bash
# ============================================================================
# Benchmark: VIO WITHOUT VPS (to see how much VPS helps)
# ============================================================================

echo "============================================================================"
echo "VIO IMU PREINTEGRATION - WITHOUT VPS"
echo "============================================================================"
echo ""

# Configuration
TEST_ID=$(date +%Y%m%d_%H%M%S)

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate 3D_Building_DepthAnyThingV2 2>/dev/null || {
        echo "⚠️  Warning: Could not activate conda environment"
    }
fi

# Common arguments WITHOUT VPS
COMMON_ARGS="
    --config config_bell412_dataset3.yaml
    --imu /mnt/External_Storage/CV_team/vio_dataset/bell412_dataset3/extracted_data/imu_data/imu__data/imu.csv
    --quarry /mnt/External_Storage/CV_team/vio_dataset/bell412_dataset3/flight_log_from_gga.csv
    --images_dir /mnt/External_Storage/CV_team/vio_dataset/bell412_dataset3/extracted_data/cam_data/camera__image_mono/images
    --images_index /mnt/External_Storage/CV_team/vio_dataset/bell412_dataset3/extracted_data/cam_data/camera__image_mono/images_index.csv
    --mag /mnt/External_Storage/CV_team/vio_dataset/bell412_dataset3/extracted_data/imu_data/imu__mag/vector3.csv
    --dem /mnt/External_Storage/CV_team/vio_dataset/Copernicus_DSM_10_N45_00_W076_00_DEM.tif
    --ground_truth /mnt/External_Storage/CV_team/vio_dataset/bell412_dataset3/bell412_dataset3_frl.pos
    --img_w 1440
    --img_h 1080
    --z_state msl
    --camera_view nadir
    --use_magnetometer
    --save_debug_data
"

echo "Test Configuration:"
echo "  Test ID: ${TEST_ID}"
echo "  Output directory: benchmark_no_vps_${TEST_ID}/preintegration/"
echo "  Mode: Preintegration WITHOUT VPS"
echo ""

# ============================================================================
# Test: Without VPS
# ============================================================================
echo "=== Running VIO without VPS ==="
echo ""

PREINT_OUT="benchmark_no_vps_${TEST_ID}/preintegration"
mkdir -p "$PREINT_OUT"

START_TIME=$(date +%s.%N)

python3 vio_vps.py \
    $COMMON_ARGS \
    --output "$PREINT_OUT" \
    2>&1 | tee "$PREINT_OUT/run.log"

END_TIME=$(date +%s.%N)
PREINT_RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "✅ Test completed in ${PREINT_RUNTIME}s"
echo ""

# ============================================================================
# Quick Analysis
# ============================================================================
echo "============================================================================"
echo "RESULTS (WITHOUT VPS)"
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

echo "=== Accuracy (WITHOUT VPS) ==="
analyze_errors "$PREINT_OUT/error_log.csv"
echo ""

echo "=== Runtime ==="
echo "Time: ${PREINT_RUNTIME}s"
echo ""

echo "============================================================================"
echo "OUTPUT FILES"
echo "============================================================================"
echo "Results saved to: benchmark_no_vps_${TEST_ID}/preintegration/"
echo "  - pose.csv         : Full trajectory"
echo "  - error_log.csv    : Error statistics"
echo "  - run.log          : Full debug output"
echo "============================================================================"

exit 0
