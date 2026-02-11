#!/bin/bash
# ============================================================================
# Benchmark Script: IMU Preintegration vs. Legacy
# ============================================================================
# Purpose: Compare performance and accuracy between:
#   1. NEW: IMU preintegration (Forster et al.)
#   2. OLD: Legacy sample-by-sample propagation
#
# Metrics:
#   - Position drift (RMSE vs. GPS ground truth)
#   - Velocity accuracy (RMSE)
#   - Computation time (seconds)
#   - Covariance consistency (NIS, NEES)
# ============================================================================

echo "============================================================================"
echo "VIO IMU PREINTEGRATION BENCHMARK"
echo "============================================================================"
echo ""

# Configuration
DURATION=480  # seconds (8 minutes)
TEST_ID=$(date +%Y%m%d_%H%M%S)

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate 3D_Building_DepthAnyThingV2 2>/dev/null || {
        echo "⚠️  Warning: Could not activate conda environment"
    }
fi

# Common arguments
COMMON_ARGS="
    --config config_dji_m600_quarry.yaml
    --imu imu.csv
    --quarry flight_log_from_gga.csv
    --images_dir camera__image_mono/images
    --images_index camera__image_mono/images_index.csv
    --mag vector3.csv
    --dem DSM_10_N47_00_W054_00_AOI.tif
    --img_w 1440
    --img_h 1080
    --camera_view nadir
    --save_debug_data
"

echo "Test Configuration:"
echo "  Duration: ${DURATION}s"
echo "  Test ID: ${TEST_ID}"
echo "  Output directory: benchmark_${TEST_ID}/"
echo ""

# ============================================================================
# Test 1: Legacy (Sample-by-Sample Propagation)
# ============================================================================
echo "=== Test 1/2: Legacy Propagation ==="
echo "Running VIO with legacy sample-by-sample propagation..."
echo ""

LEGACY_OUT="benchmark_${TEST_ID}/legacy"
mkdir -p "$LEGACY_OUT"

START_TIME=$(date +%s.%N)

python3 vio_vps.py \
    $COMMON_ARGS \
    --output "$LEGACY_OUT" \
    --use_legacy_propagation \
    2>&1 | tee "$LEGACY_OUT/run.log"

END_TIME=$(date +%s.%N)
LEGACY_RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "✅ Legacy test completed in ${LEGACY_RUNTIME}s"
echo ""
sleep 2

# ============================================================================
# Test 2: Preintegration (Forster et al.)
# ============================================================================
echo "=== Test 2/2: IMU Preintegration ==="
echo "Running VIO with IMU preintegration (Forster et al.)..."
echo ""

PREINT_OUT="benchmark_${TEST_ID}/preintegration"
mkdir -p "$PREINT_OUT"

START_TIME=$(date +%s.%N)

python3 vio_vps.py \
    $COMMON_ARGS \
    --output "$PREINT_OUT" \
    2>&1 | tee "$PREINT_OUT/run.log"

END_TIME=$(date +%s.%N)
PREINT_RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "✅ Preintegration test completed in ${PREINT_RUNTIME}s"
echo ""

# ============================================================================
# Analysis
# ============================================================================
echo "============================================================================"
echo "BENCHMARK RESULTS"
echo "============================================================================"
echo ""

# Function to extract error statistics from error_log.csv
analyze_errors() {
    local csv_file="$1"
    local label="$2"
    
    if [ ! -f "$csv_file" ]; then
        echo "❌ $label: error_log.csv not found"
        return
    fi
    
    # Use Python to compute RMSE (skip header + last 50 samples for steady-state)
    python3 - <<EOF
import pandas as pd
import numpy as np

df = pd.read_csv("$csv_file")
if len(df) > 50:
    df = df.iloc[50:]  # Skip initial convergence

pos_err = np.sqrt(df['pos_error_m'].mean()) if 'pos_error_m' in df.columns else np.nan
vel_err = np.sqrt(df['vel_error_m_s'].mean()) if 'vel_error_m_s' in df.columns else np.nan
final_pos_err = df['pos_error_m'].iloc[-1] if 'pos_error_m' in df.columns else np.nan

print(f"$label")
print(f"  Position RMSE: {pos_err:.3f} m")
print(f"  Velocity RMSE: {vel_err:.3f} m/s")
print(f"  Final Position Error: {final_pos_err:.3f} m")
EOF
}

echo "=== Accuracy Comparison ==="
analyze_errors "$LEGACY_OUT/error_log.csv" "Legacy"
echo ""
analyze_errors "$PREINT_OUT/error_log.csv" "Preintegration"
echo ""

echo "=== Computation Time ==="
echo "Legacy:           ${LEGACY_RUNTIME}s"
echo "Preintegration:   ${PREINT_RUNTIME}s"
TIME_SPEEDUP=$(echo "scale=2; 100 * (1 - $PREINT_RUNTIME / $LEGACY_RUNTIME)" | bc)
echo "Speedup:          ${TIME_SPEEDUP}%"
echo ""

echo "=== Covariance Consistency ==="
echo "Check debug_state_covariance.csv for PSD violations and condition numbers"
echo ""

# Count covariance warnings
LEGACY_COV_WARN=$(grep -c "\[COV_CHECK\]" "$LEGACY_OUT/run.log" 2>/dev/null || echo 0)
PREINT_COV_WARN=$(grep -c "\[COV_CHECK\]" "$PREINT_OUT/run.log" 2>/dev/null || echo 0)

echo "Legacy covariance warnings:          $LEGACY_COV_WARN"
echo "Preintegration covariance warnings:  $PREINT_COV_WARN"
echo ""

echo "=== FEJ Consistency ==="
if [ -f "$PREINT_OUT/debug_fej_consistency.csv" ]; then
    echo "FEJ drift analysis (preintegration):"
    python3 - <<EOF
import pandas as pd
import numpy as np

df = pd.read_csv("$PREINT_OUT/debug_fej_consistency.csv")
if len(df) > 0:
    print(f"  Median position drift:   {df['pos_fej_drift_m'].median():.6f} m")
    print(f"  Median rotation drift:   {df['rot_fej_drift_deg'].median():.6f} deg")
    print(f"  Median gyro bias drift:  {df['bg_fej_drift_rad_s'].median():.6e} rad/s")
    print(f"  Median accel bias drift: {df['ba_fej_drift_m_s2'].median():.6e} m/s²")
EOF
else
    echo "⚠️  FEJ consistency log not found"
fi
echo ""

echo "============================================================================"
echo "SUMMARY"
echo "============================================================================"

# Generate improvement percentage
python3 - <<EOF
import pandas as pd
import numpy as np

# Load both error logs
try:
    df_legacy = pd.read_csv("$LEGACY_OUT/error_log.csv").iloc[50:]
    df_preint = pd.read_csv("$PREINT_OUT/error_log.csv").iloc[50:]
    
    legacy_rmse = np.sqrt(df_legacy['pos_error_m'].mean())
    preint_rmse = np.sqrt(df_preint['pos_error_m'].mean())
    
    improvement = 100 * (1 - preint_rmse / legacy_rmse)
    
    print(f"Position RMSE:")
    print(f"  Legacy:         {legacy_rmse:.3f} m")
    print(f"  Preintegration: {preint_rmse:.3f} m")
    print(f"  Improvement:    {improvement:+.1f}%")
    print()
    
    if improvement > 5:
        print("✅ IMU Preintegration: SIGNIFICANT IMPROVEMENT")
    elif improvement > 0:
        print("✅ IMU Preintegration: Marginal improvement")
    else:
        print("⚠️  IMU Preintegration: No improvement (check tuning)")

except Exception as e:
    print(f"⚠️  Could not compute improvement: {e}")
EOF

echo ""
echo "Results saved to: benchmark_${TEST_ID}/"
echo "  - legacy/pose.csv, legacy/error_log.csv"
echo "  - preintegration/pose.csv, preintegration/error_log.csv"
echo ""
echo "============================================================================"

exit 0
