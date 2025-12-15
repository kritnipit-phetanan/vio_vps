# Benchmark Analysis Report - v2.9.7 Plane-Aided MSCKF
**Date:** 2025-12-15  
**Benchmark:** `benchmark_modular_20251215_130642`  
**Duration:** 308 seconds flight time (979s processing)

---

## 📊 Overview Results

| Metric | Value | Status |
|--------|-------|--------|
| **Final Position Error** | 1,432.7 m | ❌ CRITICAL |
| **MSCKF Success Rate** | 4.2% (19,145/455,945) | ❌ CRITICAL |
| **Plane Detections** | 86 events | ✅ Good |
| **Images Used** | 2,048 / ~4,625 frames | ⚠️ Only 44% |
| **ZUPT Detections** | 0 | ❌ Missing |
| **Velocity Error** | 0.000 m/s (all samples) | ❌ BUG |

---

## 🔴 Critical Issues Found

### 1. **MSCKF Triangulation Failure (4.2% success)**

**Problem:** Only 19,145 successful triangulations out of 455,945 attempts.

**Breakdown:**
- ❌ `fail_baseline`: 101,643 (22.3%) - Camera positions too close
- ❌ `fail_depth_sign`: 166,773 (36.6%) - **Negative depth (behind camera)**
- ❌ `fail_other`: 146,757 (32.2%) - Extreme fisheye angles filtered
- ⚠️ `fail_reproj_error`: 18,569 (4.1%) - High reprojection error

**Root Cause:**
```python
# In msckf.py line ~464
MAX_NORM_COORD = 1.5  # Filters points with |norm| > 1.5 (>56° from optical axis)

# For nadir camera, this is TOO RESTRICTIVE
# Most features are at edges (high distortion) → rejected as "fail_other"
```

**Impact:** Without MSCKF updates, VIO relies ONLY on:
- Magnetometer (heading)
- Velocity estimates (unreliable without scale)
- Result: 1.4 km position drift in 5 minutes

---

### 2. **Velocity Error Calculation Bug**

**Evidence:**
```csv
vel_error_m_s,vel_error_E,vel_error_N,vel_error_U
0.0,0.0,0.0,0.0  # ALL samples are zero!
```

**Root Cause:** Ground truth velocity not computed from PPK trajectory.

**Location:** `vio/main_loop.py` in `log_error()` method - likely missing velocity calculation from GPS positions.

---

### 3. **ZUPT (Zero Velocity Update) Not Detecting**

**Problem:** ZUPT should detect stationary periods (e.g., 4s at start before takeoff).

**Evidence:**
```
ZUPT: 0 applied | 0 rejected | 0 detected
```

**Possible causes:**
- Vibration threshold too high (helicopter rotor always vibrating)
- Acceleration noise threshold not tuned for Bell 412
- Need to check `detect_stationary()` parameters in config

---

### 4. **Only 44% of Images Used**

**Problem:** 2,048 images used out of ~4,625 available frames.

**Causes:**
1. Low parallax: Only ~2 frames show "SKIPPING velocity" message
2. **More likely:** Many frames rejected BEFORE reaching VIO processing
3. Possible: Frame rate mismatch or image loading issues

**Need to investigate:** Why 56% of frames never reach VIO frontend.

---

## ✅ What's Working

### 1. **Decoupled Parallax Check**
```
[VIO] SKIPPING velocity: parallax=0.14px < 0.3px (MSCKF/plane still active)
[VIO] SKIPPING velocity: parallax=0.25px < 0.3px (MSCKF/plane still active)
```
✅ **Confirmed:** Velocity skipped but MSCKF/plane processing continues.

### 2. **Plane Detection Active**
```
86 plane detections total (was 59 in v2.9.6)
Detecting 1-2 planes from 26-50 triangulated points
```
✅ **46% increase** in plane detection frequency.

### 3. **No Dimension Mismatch Errors**
✅ Dynamic error state size (`kf.P.shape[0]`) working correctly.

---

## 🔧 Recommended Fixes

### Priority 1: Fix MSCKF Triangulation Failure

**Issue:** 32.2% failures due to extreme fisheye angle filtering.

**Solution 1 - Relax fisheye filter:**
```python
# vio/msckf.py line ~464
MAX_NORM_COORD = 2.5  # Increase from 1.5 to 2.5 (~68° FOV)
# Or remove entirely for nadir camera (already calibrated)
```

**Solution 2 - Better depth validation:**
```python
# Check depth in camera frame (not world frame)
if depth0 < 5.0 or depth1 < 5.0:
    # CURRENT: Rejects anything <5m (nadir hovering = 3-4m altitude!)
    # FIX: Use 1.0m for nadir camera
```

**Expected improvement:** 32% → 10% failure rate (recover 100,000 triangulations)

---

### Priority 2: Fix Velocity Error Calculation

**File:** `vio/main_loop.py` around line ~876

**Current code probably missing:**
```python
# Need to compute velocity from consecutive GPS positions
gt_vel_e = (gt_E_next - gt_E_prev) / dt
gt_vel_n = (gt_N_next - gt_N_prev) / dt
gt_vel_u = (gt_U_next - gt_U_prev) / dt

# Then compute error
vel_error_e = vio_vel_e - gt_vel_e
vel_error_n = vio_vel_n - gt_vel_n
vel_error_u = vio_vel_u - gt_vel_u
```

---

### Priority 3: Tune ZUPT Detection

**File:** `configs/config_bell412_dataset3.yaml`

**Add/modify:**
```yaml
zupt:
  enabled: true
  accel_threshold: 0.5      # m/s² (reduce from default ~1.0)
  gyro_threshold: 0.05      # rad/s (reduce from default ~0.1)
  min_duration: 2.0         # seconds
  velocity_threshold: 0.2   # m/s
```

**Rationale:** Helicopter vibration is high, need more permissive thresholds.

---

### Priority 4: Investigate Image Usage

**Check:**
1. How many images in dataset: `ls -1 $IMAGES_DIR | wc -l`
2. How many loaded: Check `len(self.imgs)` in main_loop.py
3. Why skipped: Add logging before `if is_fast_rotation` check

**Possible fix:** Lower `clone_threshold` more aggressively:
```python
# Current: min_parallax * 0.5 = 0.15px
clone_threshold = 0.1  # Even lower for nadir hover
```

---

## 📈 Expected Performance After Fixes

| Metric | Current | Expected After Fixes |
|--------|---------|---------------------|
| MSCKF Success | 4.2% | **25-35%** |
| Position Error | 1,432 m | **200-400 m** |
| Images Used | 44% | **70-80%** |
| ZUPT Detections | 0 | **2-5 events** |

---

## 🧪 Testing Plan

1. **Fix 1 (Fisheye filter):** Apply → Benchmark → Check MSCKF stats
2. **Fix 2 (Velocity error):** Apply → Verify error_log.csv has non-zero values
3. **Fix 3 (ZUPT):** Add config → Check for detection logs
4. **Fix 4 (Image usage):** Investigate → Add logging → Identify bottleneck

**Test command:**
```bash
bash scripts/benchmark_modular.sh
grep "MSCKF-STATS" benchmark_*/preintegration/run.log
```

---

## 📝 Answers to Specific Questions

### ปล.1 Images ไม่ได้ใช้ทุกภาพหรอ?

**ตอบ:** ใช่ ใช้แค่ **2,048 / ~4,625 frames (44%)**

**สาเหตุที่เป็นไปได้:**
1. ❌ **Fast rotation filter:** `rotation_rate_deg_s > 30.0` (ฮ.บินเร็ว/หมุน)
2. ❌ **Frame loading issues:** อาจมีภาพบางส่วน corrupted/missing
3. ⚠️ **Parallax too low:** แต่พบแค่ 2 ครั้งที่ skip → ไม่ใช่สาเหตุหลัก

**วิธีตรวจสอบ:**
```bash
# Count actual images in directory
ls -1 /mnt/External_Storage/.../images/ | wc -l

# Check log for "SKIPPING due to fast rotation"
grep "fast rotation" benchmark_*/preintegration/run.log | wc -l
```

### ปล.2 ZUPT ทำไมไม่ detect ช่วงนิ่งๆ?

**ตอบ:** **Bug หรือ config ไม่เหมาะกับ helicopter**

**สาเหตุที่น่าจะเป็น:**
1. **Vibration threshold สูงเกิน:** Helicopter rotor สั่นสะเทือนตลอด
2. **ไม่มี config สำหรับ ZUPT:** ใช้ default values ที่เหมาะกับ ground vehicle
3. **Detection logic ไม่เหมาะกับ hovering:** ต้องดูจาก velocity + acceleration

**แนะนำ:** เพิ่ม ZUPT config ใน yaml (ดูข้างบน Priority 3)

### ปล.3 Velocity Error = 0.000 ทั้งหมด (BUG)

**ตอบ:** **Bug แน่นอน** - Ground truth velocity ไม่ได้คำนวณ

**Root cause:** ใน `vio/main_loop.py::log_error()` คำนวณ position error แต่ไม่ได้คำนวณ velocity error จาก GPS consecutive positions

**ต้องแก้ไข:** เพิ่มการคำนวณ velocity จาก `(pos_t - pos_t-1) / dt`

---

## 🎯 Conclusion

**Plane-aided MSCKF (v2.9.7) improvements ใช้งานได้แล้ว แต่ระบบโดยรวมยังมีปัญหาใหญ่:**

✅ **ที่ดีแล้ว:**
- Decoupled parallax: ทำงานถูกต้อง
- Plane detection: เพิ่มขึ้น 46%
- Dynamic state size: ไม่มี dimension error

❌ **ปัญหาหลัก (Critical):**
1. **MSCKF triangulation: 4.2% success** (ต่ำเกินไป)
   - Fisheye filter เข้มงวดเกิน (32% rejected)
   - Depth threshold ไม่เหมาะกับ nadir hovering
   
2. **Velocity error = 0:** Bug ในการคำนวณ ground truth

3. **ZUPT ไม่ทำงาน:** Config/threshold ไม่เหมาะกับ helicopter

4. **ใช้แค่ 44% ของภาพ:** ต้องหาสาเหตุว่าทำไมโดน reject

**Priority:** แก้ Fix #1 (MSCKF triangulation) ก่อน → คาดว่าจะลด position error จาก 1.4km → 200-400m

---

**Status:** Ready for implementation  
**Next Step:** Apply Priority 1 fix and re-benchmark
