#!/bin/bash
# ============================================================================
# Bell 412 Multi-Camera VIO Benchmark
# ============================================================================
# Purpose: Test multi-camera VIO with Bell 412 dataset (nadir + front cameras)
# Expected: Better performance than single-camera due to:
#   1. More features (2× more observations)
#   2. Better motion observability (nadir sees Z, front sees XY)
#   3. Reduced scale ambiguity (forward motion + downward motion)
# ============================================================================

echo "============================================================================"
echo "BELL 412 MULTI-CAMERA VIO BENCHMARK"
echo "============================================================================"
echo ""

# Configuration
TEST_ID=$(date +%Y%m%d_%H%M%S)
DATASET_BASE="/mnt/External_Storage/CV_team/vio_dataset/bell412_dataset3"

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate 3D_Building_DepthAnyThingV2 2>/dev/null || {
        echo "⚠️  Warning: Could not activate conda environment"
    }
fi

# Common arguments
COMMON_ARGS="
    --config config_bell412_dataset3.yaml
    --imu ${DATASET_BASE}/extracted_data/imu_data/imu__data/imu.csv
    --quarry ${DATASET_BASE}/flight_log_from_gga.csv
    --mag ${DATASET_BASE}/extracted_data/imu_data/imu__mag/vector3.csv
    --dem /mnt/External_Storage/CV_team/vio_dataset/Copernicus_DSM_10_N45_00_W076_00_DEM.tif
    --img_w 1440
    --img_h 1080
    --use_magnetometer
    --save_debug_data
"

echo "Dataset: Bell 412 Dataset 3 (Forward Flight)"
echo "Cameras: Nadir (down) + Front (forward)"
echo "Expected improvement: 20-40% better RMSE vs single-camera"
echo ""

# ============================================================================
# Test 1: Single Camera (Nadir Only) - Baseline
# ============================================================================
echo "=== Test 1: Single Camera (Nadir Only) - Baseline ==="
echo ""

NADIR_OUT="benchmark_${TEST_ID}/nadir_only"
mkdir -p "$NADIR_OUT"

START_TIME=$(date +%s.%N)

python3 vio_vps.py \
    $COMMON_ARGS \
    --images_dir ${DATASET_BASE}/extracted_data/cam_data/camera__image_mono/images \
    --images_index ${DATASET_BASE}/extracted_data/cam_data/camera__image_mono/images_index.csv \
    --camera_view nadir \
    --disable_vio_velocity \
    --output "$NADIR_OUT" \
    2>&1 | tee "$NADIR_OUT/run.log"

END_TIME=$(date +%s.%N)
NADIR_RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "✅ Nadir-only test completed in ${NADIR_RUNTIME}s"
echo ""

# ============================================================================
# Test 2: Multi-Camera (Nadir + Front) - Enhanced
# ============================================================================
echo "=== Test 2: Multi-Camera (Nadir + Front) - Enhanced ==="
echo ""

MULTI_OUT="benchmark_${TEST_ID}/multi_camera"
mkdir -p "$MULTI_OUT"

START_TIME=$(date +%s.%N)

python3 vio_vps.py \
    $COMMON_ARGS \
    --images_dir ${DATASET_BASE}/extracted_data/cam_data/camera__image_mono/images \
    --images_index ${DATASET_BASE}/extracted_data/cam_data/camera__image_mono/images_index.csv \
    --front_images_dir ${DATASET_BASE}/extracted_data/cam_data/front_camera__image_mono/images \
    --front_images_index ${DATASET_BASE}/extracted_data/cam_data/front_camera__image_mono/images_index.csv \
    --camera_view multi \
    --disable_vio_velocity \
    --output "$MULTI_OUT" \
    2>&1 | tee "$MULTI_OUT/run.log"

END_TIME=$(date +%s.%N)
MULTI_RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "✅ Multi-camera test completed in ${MULTI_RUNTIME}s"
echo ""

# ============================================================================
# Results Analysis
# ============================================================================
echo "============================================================================"
echo "RESULTS COMPARISON"
echo "============================================================================"
echo ""

# Function to extract error statistics
analyze_errors() {
    local csv_file="$1"
    local test_name="$2"
    
    if [ ! -f "$csv_file" ]; then
        echo "❌ $test_name: error_log.csv not found"
        return
    fi
    
    # Use Python to compute RMSE
    python3 - "$csv_file" "$test_name" <<'EOF'
import pandas as pd
import numpy as np
import sys

csv_file = sys.argv[1]
test_name = sys.argv[2]

df = pd.read_csv(csv_file)
if len(df) > 50:
    df = df.iloc[50:]  # Skip initial convergence

pos_err = np.sqrt(df['pos_error_m'].mean()) if 'pos_error_m' in df.columns else np.nan
vel_err = np.sqrt(df['vel_error_m_s'].mean()) if 'vel_error_m_s' in df.columns else np.nan
final_pos_err = df['pos_error_m'].iloc[-1] if 'pos_error_m' in df.columns else np.nan

print(f"{'='*60}")
print(f"{test_name:^60}")
print(f"{'='*60}")
print(f"  Position RMSE:       {pos_err:8.2f} m")
print(f"  Velocity RMSE:       {vel_err:8.3f} m/s")
print(f"  Final Position Error:{final_pos_err:8.2f} m")

# Quick assessment
if pos_err < 10:
    print(f"  Status: ✅ EXCELLENT (RMSE < 10m)")
elif pos_err < 30:
    print(f"  Status: ✅ GOOD (RMSE < 30m)")
elif pos_err < 100:
    print(f"  Status: ⚠️  ACCEPTABLE (RMSE < 100m)")
else:
    print(f"  Status: ❌ POOR (RMSE > 100m)")
print()
EOF
}

echo "=== Accuracy Comparison ==="
echo ""
analyze_errors "$NADIR_OUT/error_log.csv" "Nadir Only (Baseline)"
analyze_errors "$MULTI_OUT/error_log.csv" "Multi-Camera (Nadir + Front)"

# Compute improvement
if [ -f "$NADIR_OUT/error_log.csv" ] && [ -f "$MULTI_OUT/error_log.csv" ]; then
    python3 - "$NADIR_OUT/error_log.csv" "$MULTI_OUT/error_log.csv" <<'EOF'
import pandas as pd
import numpy as np
import sys

nadir_csv = sys.argv[1]
multi_csv = sys.argv[2]

df_nadir = pd.read_csv(nadir_csv)
df_multi = pd.read_csv(multi_csv)

if len(df_nadir) > 50:
    df_nadir = df_nadir.iloc[50:]
if len(df_multi) > 50:
    df_multi = df_multi.iloc[50:]

pos_nadir = np.sqrt(df_nadir['pos_error_m'].mean())
pos_multi = np.sqrt(df_multi['pos_error_m'].mean())

improvement = ((pos_nadir - pos_multi) / pos_nadir) * 100

print(f"{'='*60}")
print(f"{'IMPROVEMENT ANALYSIS':^60}")
print(f"{'='*60}")
print(f"  Baseline (Nadir):   {pos_nadir:8.2f} m")
print(f"  Multi-Camera:       {pos_multi:8.2f} m")
print(f"  Improvement:        {improvement:8.1f} %")
print()

if improvement > 20:
    print("  ✅ SIGNIFICANT: Multi-camera provides major improvement!")
elif improvement > 10:
    print("  ✅ GOOD: Multi-camera helps noticeably")
elif improvement > 0:
    print("  ✓  MODEST: Small improvement from multi-camera")
else:
    print("  ⚠️  DEGRADATION: Multi-camera performed worse (check calibration)")
print()
EOF
fi

echo ""
echo "=== Runtime Comparison ==="
echo "  Nadir Only:     ${NADIR_RUNTIME}s"
echo "  Multi-Camera:   ${MULTI_RUNTIME}s"
MULTI_OVERHEAD=$(echo "scale=1; ($MULTI_RUNTIME / $NADIR_RUNTIME - 1) * 100" | bc)
echo "  Overhead:       ${MULTI_OVERHEAD}% (expected: 50-80% due to 2× features)"
echo ""

echo "=== Feature Statistics (from debug logs) ==="
if [ -f "$NADIR_OUT/vo_debug.csv" ] && [ -f "$MULTI_OUT/vo_debug.csv" ]; then
    python3 - "$NADIR_OUT/vo_debug.csv" "$MULTI_OUT/vo_debug.csv" <<'EOF'
import pandas as pd
import numpy as np
import sys

nadir_csv = sys.argv[1]
multi_csv = sys.argv[2]

df_nadir = pd.read_csv(nadir_csv)
df_multi = pd.read_csv(multi_csv)

# Check available columns
nadir_cols = df_nadir.columns.tolist()
multi_cols = df_multi.columns.tolist()

print(f"  Nadir Only:")
if 'num_tracks' in nadir_cols:
    print(f"    - Avg features tracked: {df_nadir['num_tracks'].mean():.0f}")
if 'num_inliers' in nadir_cols:
    print(f"    - Avg inliers:          {df_nadir['num_inliers'].mean():.0f}")
if 'avg_parallax_px' in nadir_cols:
    print(f"    - Avg parallax:         {df_nadir['avg_parallax_px'].mean():.1f} px")
if 'avg_flow_px' in nadir_cols:
    print(f"    - Avg optical flow:     {df_nadir['avg_flow_px'].mean():.1f} px")
print()

print(f"  Multi-Camera:")
if 'num_tracks' in multi_cols:
    print(f"    - Avg features tracked: {df_multi['num_tracks'].mean():.0f}")
if 'num_inliers' in multi_cols:
    print(f"    - Avg inliers:          {df_multi['num_inliers'].mean():.0f}")
if 'avg_parallax_px' in multi_cols:
    print(f"    - Avg parallax:         {df_multi['avg_parallax_px'].mean():.1f} px")
if 'avg_flow_px' in multi_cols:
    print(f"    - Avg optical flow:     {df_multi['avg_flow_px'].mean():.1f} px")
print()
EOF
fi

echo "============================================================================"
echo "OUTPUT FILES"
echo "============================================================================"
echo "Results saved to: benchmark_${TEST_ID}/"
echo "  - nadir_only/       : Single camera baseline"
echo "  - multi_camera/     : Multi-camera enhanced"
echo ""
echo "Key takeaways:"
echo "  1. Multi-camera should reduce position error by 20-40%"
echo "  2. Forward motion benefits most from nadir+front combination"
echo "  3. Computational cost increases proportionally to number of features"
echo "============================================================================"

exit 0
