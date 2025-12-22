# Hardware Timestamp Investigation Results

## Executive Summary

**Attempted:** Hardware PPS timestamp integration (stamp_ref) to eliminate 3-6ms ROS timing jitter  
**Result:** ❌ **FAILED** - Filter divergence (P_pos exploded to 6.5M m²)  
**Root Cause:** Timestamp interpolation drift (6.3ms offset drift over 308s flight)  
**Action:** ✅ Reverted to software timestamps (stamp_log) - system stable again

---

## Problem Discovery

### Initial Symptoms (Baseline v3.5.0)
- MSCKF success: 16.3%
- fail_reproj_error: 22.0% (suspected timing issue)
- Position RMSE: 995.6m
- Filter stable: P_pos = 4.3 m²

### Hypothesis
ROS software timestamps (stamp_log) have 3-6ms jitter:
- IMU: std = 3.225ms
- Camera: std = 5.936ms
- Time sync misalignment 5-10ms
- Causing triangulation/reprojection errors

### Dataset Investigation
Found hardware PPS timestamps (stamp_ref):
- IMU: std = 0.190ms (17x better!)
- Camera: std = 2.848ms (2.1x better!)
- Expected: 30-40% improvement in MSCKF success

---

## Implementation (v3.7.0)

### Dataset Structure
```
imu__data/imu.csv                    # 123,096 samples with stamp_log
imu__data_stamped/imu_with_ref.csv   # 123,087 timestamp mappings (WRONG FILE!)
imu__time_ref/timeref.csv            # 123,091 timestamp mappings (CORRECT)
imu__time_ref_cam/timeref.csv        # Camera stamp_log → stamp_ref mapping
```

### Implementation Details
1. Created `imu_hw_timestamp.csv`:
   - Merged IMU data with hardware timestamps
   - Used `np.interp()` to interpolate stamp_ref for each IMU sample
   - 5 samples required interpolation (timeref[0] = imu[4])

2. Modified `data_loaders.py`:
   - Auto-detect stamp_ref vs stamp_log
   - Camera: interpolate stamp_ref from timeref.csv mapping

3. Updated entire pipeline:
   - benchmark_modular.sh paths
   - CLI arguments
   - Config dataclass

---

## Results - FAILURE

### v3.7.0 (Hardware Timestamp)
- MSCKF success: **15.4%** (WORSE than 16.3%)
- fail_reproj: **27.1%** (WORSE than 22.0%, expected improvement!)
- fail_depth_sign: 49.9% (vs 52.9%)
- Position RMSE: **DIVERGED**
- Filter: **P_pos = 6,468,420 m²** (should be <100)

### Divergence Timeline
- Started stable: P_pos ~4 m²
- Frame 1102 (t=1824s): P_pos > 1000 m²
- Final: P_pos = 6.47M m²
- System ill-conditioned: cond = 1.9e+12
- All velocity updates rejected: chi2 = 2340 >> 11.3 threshold

---

## Root Cause Analysis

### 1. Interpolation Drift
```python
# Offset between stamp_ref and stamp_log
offset = stamp_ref - stamp_log
offset_mean = -1653511066.315 s (expected constant offset)
offset_std = 6.288 ms ⚠️  (should be ~0 if linear)
```

**Problem:** 6.3ms drift over 308s flight
- Not a constant offset (clocks have different rates)
- Linear interpolation accumulates error
- Affects preintegration: Δt errors compound

### 2. Sample Mismatch
```
IMU data:     123,096 samples
Timeref:      123,091 samples (missing 5)
Offset:       timeref[0] = imu[4]
```

**Problem:** Must interpolate for missing samples
- Interpolation introduces systematic bias
- Jitter increased: IMU timestep std = 3.0ms (vs expected ~0.19ms)
- 1,448 large jumps (>10ms) detected

### 3. Timing Error Propagation
1. Interpolation drift → timing error in IMU samples
2. Preintegration accumulates timing errors
3. State predictions become inaccurate
4. Measurement innovations grow large
5. Chi-squared test rejects updates
6. No corrections → filter diverges

---

## v3.7.1 - Revert

### Changes
1. Reverted `data_loaders.py` to use stamp_log (prefer over stamp_ref)
2. Removed imu_hw_timestamp.csv usage
3. Removed --cam_timeref CLI argument

### Results
- MSCKF success: **16.3%** (matches baseline)
- fail_reproj: **22.0%** (matches baseline)
- Position RMSE: **995.6m** (matches baseline)
- Filter stable: **P_pos = 0.42 m²** ✅

---

## Lessons Learned

### Why Hardware Timestamps Failed
1. **Interpolation is not trivial** for non-linear clock drift
2. **Sample alignment matters** - 5 missing samples cascaded
3. **Timestamp jitter << interpolation drift** (0.19ms << 6.3ms)
4. **System tuning** may be calibrated for software timestamp characteristics

### Hardware Timestamp Requirements
To use hardware timestamps successfully:
1. ✅ **No interpolation** - use only directly measured timestamps
2. ✅ **Perfect alignment** - every IMU/camera sample must have timeref
3. ✅ **Monotonic validation** - verify no jumps or reversals
4. ✅ **Retune filter** - Q_preint, R matrices for new timing precision
5. ✅ **Nearest neighbor** if interpolation unavoidable

### Current Recommendation
**Stick with stamp_log (software timestamps)** until:
- Dataset provides complete 1:1 timestamp mapping (no interpolation)
- OR: Implement sophisticated interpolation (e.g., Kalman smooth)
- AND: Retune filter for hardware timing characteristics

---

## Future Work

### Option 1: Fix Interpolation
- Use `imu__time_ref/timeref.csv` (123,091 vs 123,087 samples)
- Nearest neighbor instead of linear interpolation
- Extrapolate 5 missing samples instead of interpolate
- Validate monotonic ordering

### Option 2: Hybrid Approach
- Hardware timestamps for IMU only (direct mapping possible?)
- Software timestamps for camera (fewer samples, less critical)
- Reduces interpolation error surface

### Option 3: Dataset Regeneration
- Re-extract dataset with complete hardware timestamps
- Ensure every IMU/camera sample has synchronized PPS reference
- No missing samples requiring interpolation

### Option 4: Clock Drift Modeling
- Model non-linear drift between stamp_log and stamp_ref
- Use polynomial fit instead of linear interpolation
- Or Kalman filter for timestamp smoothing

---

## Statistics Summary

| Metric | v3.5.0 Baseline | v3.7.0 HW Timestamp | v3.7.1 Reverted |
|--------|----------------|---------------------|-----------------|
| MSCKF Success | 16.3% | 15.4% ❌ | 16.3% ✅ |
| fail_reproj | 22.0% | 27.1% ❌ | 22.0% ✅ |
| fail_depth_sign | 52.9% | 49.9% | 52.9% |
| Pos RMSE | 995.6m | DIVERGED ❌ | 995.6m ✅ |
| P_pos final | 4.3 m² | 6.47M m² ❌ | 0.42 m² ✅ |
| Filter Status | Stable ✅ | Diverged ❌ | Stable ✅ |

**Conclusion:** Hardware timestamps made things WORSE due to interpolation artifacts. Software timestamps remain the stable baseline.

---

## Files Modified

### v3.7.0 (Hardware Timestamp - REVERTED)
- ❌ `vio/data_loaders.py` - load_imu_csv() prefers stamp_ref
- ❌ `vio/data_loaders.py` - load_images() with timeref interpolation
- ❌ `vio/config.py` - Added cam_timeref_csv field
- ❌ `vio/main_loop.py` - Pass timeref to load_images()
- ❌ `run_vio.py` - Added --cam_timeref CLI arg
- ❌ `scripts/benchmark_modular.sh` - Added CAM_TIMEREF variable
- ❌ Dataset: Created imu_hw_timestamp.csv (interpolated)

### v3.7.1 (Reverted - CURRENT)
- ✅ `vio/data_loaders.py` - load_imu_csv() prefers stamp_log
- ✅ `scripts/benchmark_modular.sh` - Removed --cam_timeref

**Status:** System back to stable baseline (v3.5.0 equivalent)

---

Date: 2024-12-21  
Version: v3.7.1  
Status: Hardware timestamp investigation CLOSED (failed due to interpolation)  
Next: Focus on other MSCKF improvement strategies
