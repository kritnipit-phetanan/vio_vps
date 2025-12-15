# VIO Debug Files Guide

คู่มือการใช้งานไฟล์ debug สำหรับวิเคราะห์ VIO performance

## Overview

เมื่อรัน VIO pipeline จะสร้างไฟล์ debug หลายไฟล์ใน `benchmark_modular_*/preintegration/`:

```
benchmark_modular_YYYYMMDD_HHMMSS/preintegration/
├── debug_residuals.csv          # Measurement updates (DEM, MAG, VPS, ZUPT)
├── debug_feature_stats.csv      # Feature tracking statistics
├── debug_msckf_window.csv       # MSCKF sliding window state
├── msckf_debug.csv              # Per-feature MSCKF triangulation/update
├── debug_state_covariance.csv   # State uncertainty over time
├── debug_fej_consistency.csv    # First-Estimate Jacobian consistency
├── debug_imu_raw.csv            # Raw IMU measurements
├── error_log.csv                # Position/velocity/attitude errors vs GT
├── pose.csv                     # VIO estimated trajectory
└── state_debug.csv              # Complete state vector over time
```

---

## 1. debug_residuals.csv

**Purpose**: บันทึกทุก measurement update (innovation, chi-square test, acceptance)

**Columns**:
- `t`: timestamp
- `frame`: frame number (-1 = IMU-only update)
- `update_type`: DEM, MAG, VPS, ZUPT, MSCKF
- `innovation_x/y/z`: measurement residual (observed - predicted)
- `mahalanobis_dist`: normalized innovation (chi-square test statistic)
- `chi2_threshold`: acceptance threshold
- `accepted`: 1 = accepted, 0 = rejected
- `NIS`: Normalized Innovation Squared
- `NEES`: Normalized Estimation Error Squared

**How to check if MSCKF/ZUPT is updating**:
```bash
# Count MSCKF updates
grep ",MSCKF," debug_residuals.csv | wc -l

# Count ZUPT updates
grep ",ZUPT," debug_residuals.csv | wc -l

# Check last 10 MSCKF updates
grep ",MSCKF," debug_residuals.csv | tail -10

# Check innovation magnitude for MSCKF
grep ",MSCKF," debug_residuals.csv | awk -F',' '{print $4,$5,$6}'
```

**Expected values**:
- DEM: Many updates (every frame, innovation ~0.001-5m)
- MAG: Frequent updates (innovation ~0.5-2°)
- MSCKF: Should appear when triangulation succeeds
- ZUPT: Should appear during stationary periods
- VPS: Only if VPS measurements available

**ปัญหา**: Currently ZUPT not logging → Fixed in v2.9.8.1

---

## 2. msckf_debug.csv

**Purpose**: บันทึก per-feature MSCKF processing (triangulation success/failure, updates)

**Columns**:
- `frame`: camera frame number
- `feature_id`: unique feature ID
- `num_observations`: number of times feature was tracked
- `triangulation_success`: 1 = success, 0 = failed
- `reprojection_error_px`: pixel reprojection error (if success)
- `innovation_norm`: measurement innovation magnitude (if updated)
- `update_applied`: 1 = EKF updated, 0 = rejected by chi-square
- `chi2_test`: chi-square test statistic

**How to check MSCKF performance**:
```bash
# Success rate
awk -F',' 'NR>1 {total++; if($4==1) success++} END {print success"/"total" = "100*success/total"%"}' msckf_debug.csv

# Count updates applied
awk -F',' 'NR>1 && $7==1 {count++} END {print count" updates applied"}' msckf_debug.csv

# Average reprojection error (for successful triangulations)
awk -F',' 'NR>1 && $4==1 && $5!="nan" {sum+=$5; count++} END {print "Mean reproj error: "sum/count" px"}' msckf_debug.csv

# Features by frame
awk -F',' 'NR>1 {frame_count[$1]++} END {for(f in frame_count) print f, frame_count[f]}' msckf_debug.csv | sort -n | tail -20
```

**Interpretation**:
- `triangulation_success=1, update_applied=1`: Feature helped improve state ✅
- `triangulation_success=1, update_applied=0`: Triangulated but rejected by chi-square (innovation too large) ⚠️
- `triangulation_success=0`: Failed triangulation (baseline, parallax, depth, reproj error) ❌

**Expected values**:
- Triangulation success rate: 20-40% (good), <10% (poor)
- Reprojection error: 0.5-2px (good), >5px (bad calibration)
- Update applied rate: 60-80% of successful triangulations

---

## 3. debug_msckf_window.csv

**Purpose**: บันทึก MSCKF sliding window state (จำนวน camera clones, features, marginalization)

**Columns**:
- `frame`: camera frame number
- `t`: timestamp
- `num_camera_clones`: number of camera poses in sliding window
- `num_tracked_features`: total features being tracked
- `num_mature_features`: features ready for MSCKF (observed ≥3 times)
- `window_start_time`: timestamp of oldest clone
- `window_duration`: time span of window (seconds)
- `marginalized_clone_id`: which clone was removed (-1 = none)

**How to analyze**:
```bash
# Check window size over time
awk -F',' 'NR>1 {print $1,$3}' debug_msckf_window.csv | tail -50

# Check feature counts
awk -F',' 'NR>1 {print $1,$4,$5}' debug_msckf_window.csv | tail -50

# Average window duration
awk -F',' 'NR>1 {sum+=$7; count++} END {print "Mean window duration: "sum/count" sec"}' debug_msckf_window.csv
```

**Interpretation**:
- `num_camera_clones`: Should be 3-10 (config: `max_cam_states`)
- `num_tracked_features`: 200-800 (depends on scene texture)
- `num_mature_features`: 50-400 (features observed ≥3 times)
- `window_duration`: 0.2-1.0s (depends on camera rate and window size)

**Red flags**:
- `num_tracked_features` dropping rapidly → feature tracking failure
- `num_mature_features = 0` → parallax too low or cloning disabled
- `window_duration > 2s` → camera rate too slow or window too large

---

## 4. debug_feature_stats.csv

**Purpose**: บันทึก feature tracking statistics (detection, tracking, inliers)

**Columns**:
- `frame`: camera frame number
- `t`: timestamp
- `num_features_detected`: new features detected in this frame
- `num_features_tracked`: features successfully tracked from previous frame
- `num_inliers`: features passing RANSAC/chi-square validation
- `mean_parallax_px`: average optical flow magnitude (pixels)
- `max_parallax_px`: maximum optical flow magnitude
- `tracking_ratio`: tracked / total (1.0 = 100%)
- `inlier_ratio`: inliers / tracked

**How to analyze**:
```bash
# Check tracking quality
awk -F',' 'NR>1 {print $1,$8,$9}' debug_feature_stats.csv | tail -50

# Average parallax
awk -F',' 'NR>1 && $6>0 {sum+=$6; count++} END {print "Mean parallax: "sum/count" px"}' debug_feature_stats.csv

# Frames with low parallax (<1px)
awk -F',' 'NR>1 && $6<1.0 && $6>0 {print $1,$2,$6}' debug_feature_stats.csv | head -20
```

**Interpretation**:
- `tracking_ratio > 0.9`: Good feature tracking ✅
- `inlier_ratio > 0.7`: Good feature quality ✅
- `mean_parallax_px < 1.0`: Low motion → MSCKF struggles ⚠️
- `mean_parallax_px > 10`: Fast motion → may trigger rotation filter ⚠️

**Expected values**:
- Tracking ratio: 0.85-0.98 (good), <0.7 (poor lighting/blur)
- Inlier ratio: 0.70-0.95 (good), <0.5 (bad calibration/outliers)
- Mean parallax: 2-8px (normal), <1px (hovering), >20px (aggressive maneuver)

---

## 5. error_log.csv

**Purpose**: บันทึก errors เทียบกับ ground truth (GPS/PPK)

**Columns**:
- `t`: timestamp
- `pos_error_m`: 3D position error (meters)
- `vel_error_m_s`: 3D velocity error (m/s)
- `alt_error_m`: altitude error (meters)
- `yaw_error_deg`: heading error (degrees)

**How to analyze**:
```bash
# Position error statistics
awk -F',' 'NR>50 {sum+=$2; if(NR==50){min=max=$2} if($2<min)min=$2; if($2>max)max=$2; count++} END {print "Mean: "sum/count"m, Min: "min"m, Max: "max"m"}' error_log.csv

# Final errors
tail -1 error_log.csv

# Error over time (plot)
awk -F',' 'NR>1 {print $1,$2}' error_log.csv > pos_error.dat
```

**Expected values** (Bell 412 dataset):
- Position error: <200m (excellent), 200-500m (good), >1000m (poor)
- Velocity error: <5 m/s (good), >15 m/s (poor)
- Altitude error: <10m (good), >50m (poor)
- Yaw error: <15° (good), >45° (poor drift)

---

## Common Analysis Workflows

### Check if MSCKF is working
```bash
# 1. Check triangulation success rate (from run.log)
grep "MSCKF-STATS" run.log

# 2. Check if updates are applied
grep ",MSCKF," debug_residuals.csv | wc -l

# 3. Check per-feature updates
awk -F',' '$7==1' msckf_debug.csv | wc -l

# 4. If no updates, check failure reasons
grep "fail_" run.log | sort | uniq -c
```

### Check if ZUPT is working
```bash
# 1. Check ZUPT detections (from run.log)
grep "ZUPT:" run.log

# 2. Check ZUPT logs in residuals
grep ",ZUPT," debug_residuals.csv

# 3. If no detections, check velocity magnitudes
awk -F',' 'NR>1 {print $1,$4}' error_log.csv | head -100
```

### Diagnose position drift
```bash
# 1. Check position error over time
awk -F',' 'NR>1 {print $1,$2}' error_log.csv > drift.dat

# 2. Check when drift starts
awk -F',' 'NR>1 && $2>100 {print "Drift >100m at frame "$1; exit}' error_log.csv

# 3. Check MSCKF performance at that time
# (use frame number from step 2)
awk -F',' -v frame=XXX '$1>=frame && $1<=frame+50' msckf_debug.csv

# 4. Check feature parallax at that time
awk -F',' -v frame=XXX '$1>=frame && $1<=frame+50' debug_feature_stats.csv
```

### Compare two runs
```bash
# Compare MSCKF success rates
echo "Run 1:" && grep "Success:" run1/run.log
echo "Run 2:" && grep "Success:" run2/run.log

# Compare position errors
echo "Run 1:" && tail -1 run1/error_log.csv
echo "Run 2:" && tail -1 run2/error_log.csv

# Compare ZUPT detections
echo "Run 1:" && grep ",ZUPT," run1/debug_residuals.csv | wc -l
echo "Run 2:" && grep ",ZUPT," run2/debug_residuals.csv | wc -l
```

---

## Troubleshooting

### Problem: ZUPT not logging to debug_residuals.csv
**Symptoms**: `ZUPT: 0 detected` in run.log, no ZUPT entries in debug_residuals.csv

**Causes**:
1. ZUPT disabled in config (`zupt.enabled: false`)
2. Thresholds too high for helicopter vibration
3. Logging failed (exception in output_utils.py)

**Solution**: Check config, lower thresholds, check for warning messages

**Fixed in**: v2.9.8.1 (added warning message if logging fails)

---

### Problem: MSCKF success rate <10%
**Symptoms**: Most features fail triangulation

**Causes**:
1. `fail_depth_sign` high → depth threshold or sign issue
2. `fail_other` high → fisheye filter too restrictive
3. `fail_reproj_error` high → bad camera calibration
4. `fail_baseline` high → camera poses too close (low parallax)

**Solution**:
- Adjust `MAX_NORM_COORD` (fisheye filter)
- Fix depth validation logic (separate sign vs magnitude)
- Check camera intrinsics (K, D matrices)
- Increase `min_parallax` or lower clone threshold

**Fixed in**: v2.9.8 (fisheye 1.5→2.5), v2.9.8.1 (depth logic fix)

---

### Problem: Features detected but no MSCKF updates
**Symptoms**: `num_tracked_features > 0` but `triangulation_success = 0` in msckf_debug.csv

**Check**:
1. `num_mature_features` in debug_msckf_window.csv → should be >0
2. Parallax in debug_feature_stats.csv → should be >1px
3. `num_camera_clones` → should be ≥2

**Common causes**:
- Parallax threshold too high (`min_parallax > 2px`)
- Clone threshold too high (not creating clones)
- Window size too small (`max_cam_states < 3`)

---

## File Size Reference

Typical file sizes for 308s flight (4625 frames):

- debug_residuals.csv: ~20-50 MB (depends on update rate)
- msckf_debug.csv: ~30-50 MB (all features × frames)
- debug_feature_stats.csv: ~500 KB
- debug_msckf_window.csv: ~500 KB
- error_log.csv: ~1-2 MB
- pose.csv: ~5-10 MB

If files are much larger/smaller, something may be wrong with logging.

---

## Version History

- **v2.9.8**: Fixed velocity error calculation, added ZUPT config, relaxed fisheye/depth
- **v2.9.8.1**: Fixed depth validation logic, fixed chi-square threshold (5.99→3.84), added ZUPT logging warning
- **v2.9.7**: Added rigorous plane Jacobian, decoupled parallax, dynamic state size

---

## Summary: ตอบคำถาม 4 ข้อ

### 1. ZUPT หากมีอัปเดท แต่ไม่ถูกเพิ่มเข้า log (debug_residuals.csv)
**คำตอบ**: ZUPT should be logged via `log_measurement_update()` in `propagation.py` line 495-508
- ✅ **Fixed**: Added warning message if logging fails (v2.9.8.1)
- **Check**: `grep ",ZUPT," debug_residuals.csv` to verify
- **Expected**: Should see entries like `t,frame,ZUPT,innovation_x,innovation_y,innovation_z,...`

### 2. หากอยากดูว่า MSCKF ถูกอัปเดทไหม
**คำตอบ**: ใช้ 3 ไฟล์ร่วมกัน:

**msckf_debug.csv**: Per-feature updates
```bash
# Count features that updated EKF
awk -F',' 'NR>1 && $7==1' msckf_debug.csv | wc -l

# Show recent successful updates
awk -F',' 'NR>1 && $7==1 {print $1,$2,$5,$6}' msckf_debug.csv | tail -20
```

**debug_residuals.csv**: Measurement updates (if MSCKF logs here)
```bash
grep ",MSCKF," debug_residuals.csv | wc -l
```

**debug_msckf_window.csv**: Window state (indirect indicator)
```bash
# Features ready for MSCKF
awk -F',' 'NR>1 {print $1,$5}' debug_msckf_window.csv | tail -50
```

**ข้อจำกัด**: Currently MSCKF might not log to debug_residuals.csv (only per-feature in msckf_debug.csv)

### 3. plane_msckf.py ค่า chi-square threshold เขียนเป็น 5.99 แต่คอมเมนต์บอก 1 DOF
**คำตอบ**: ✅ **Fixed** in v2.9.8.1
- เดิม: `chi2_threshold = 5.99  # 95% confidence, 1 DOF` ❌ (5.99 = 2 DOF)
- แก้แล้ว: `chi2_threshold = 3.84  # 95% confidence, 1 DOF` ✅
- เหตุผล: Plane constraint is scalar (point-to-plane distance) = 1 DOF

### 4. ตรวจสอบผลการรัน benchmark_modular_20251215_135044
**วิเคราะห์**:

✅ **Improvements from v2.9.7**:
- MSCKF success: 4.2% → 8.1% (+92% improvement)
- fail_other: 32.2% → 2.8% (-91% improvement, fisheye filter working!)
- Position error: 1433m → 716m (-50% improvement)
- Velocity error: Now computed correctly (was 0.0 bug)
- Plane detections: 86 → 113 (+31%)

❌ **Problems found**:
- fail_depth_sign: 36.6% → 54.0% (+48% WORSE!)
  - **Root cause**: `depth < 1.0` catches valid features at 0.5-0.9m
  - **Fix**: Changed to `depth <= 0.0` (sign) + `depth < 0.1` (minimum)
- ZUPT: Still 0 detections
  - **Possible**: Thresholds still too high or helicopter never truly stationary
  - **Fix**: Added warning message to see if detection occurs but logging fails

⚠️ **Still needs investigation**:
- fail_reproj_error: 24.2% (was 4.1% in v2.9.7) → Camera calibration or model issue?
- Images used: 44% (no significant change)

**Next benchmark (v2.9.8.1) should show**:
- fail_depth_sign: 54% → 20-30% (depth logic fix)
- MSCKF success: 8.1% → 25-35% (if depth fix works)
- ZUPT: Warning messages if detection occurs

