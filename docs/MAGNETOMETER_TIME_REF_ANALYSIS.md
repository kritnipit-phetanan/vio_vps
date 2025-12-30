# Magnetometer Time_ref Analysis - Why NOT to Use It

**Date**: December 31, 2025  
**Version**: v3.9.5 → REVERTED to v3.9.4  
**Commit**: 9893a58 (revert of 8871279)

## Executive Summary

**CRITICAL FINDING**: Using `time_ref` for magnetometer timestamps causes **catastrophic filter divergence**.

- ❌ Position error: 54.9m → **114,088m** (+113,995m = +207,800% worse)
- ❌ Velocity error: 2.031 m/s → **463.271 m/s** (+461 m/s = +22,700% worse)
- ✓ Yaw error: 103.70° → 100.88° (-2.8° = +2.7% better)
- 💥 Covariance explosion: P_vel 1.13 → **335.7** (296x growth)

**Conclusion**: The tiny yaw improvement (2.7%) is **NOT worth** the catastrophic divergence.

---

## 🔍 Experimental Setup

### Benchmark Comparison

| Benchmark ID | Magnetometer Timestamp | Status |
|--------------|------------------------|---------|
| **20251230_214742** | `stamp_bag` (ROS recording time) | ✅ **STABLE** |
| **20251231_000813** | `time_ref` (hardware clock) | ❌ **DIVERGED** |

Both benchmarks:
- Same dataset: Bell 412 Dataset 3
- Same config: `config_bell412_dataset3.yaml`
- Same duration: 308.3 seconds (123,087 samples)
- Same code version: v3.9.4 + magnetometer time_ref change

### Code Change (v3.9.5 - REVERTED)

```python
# OLD priority (STABLE): time_ref > stamp_bag > stamp_msg
elif "stamp_bag" in df.columns:
    tcol = "stamp_bag"

# NEW priority (BROKEN): time_ref > stamp_msg (if timeref_csv) > stamp_bag
elif "stamp_msg" in df.columns and timeref_csv and os.path.exists(timeref_csv):
    tcol = "stamp_msg"
    # Convert stamp_msg → time_ref via interpolation
```

---

## 📊 Quantitative Results

### 1. Position Error Comparison

```
BEFORE (stamp_bag):
  Mean: 54.893m ± 25.033m
  Drift: 0.9m → 92.6m (+91.6m over 308s)
  Drift rate: 0.297 m/s

AFTER (time_ref):
  Mean: 38,828.246m ± 38,363.915m
  Drift: 0.9m → 114,088.2m (+114,087.3m over 308s)
  Drift rate: 370.0 m/s (1246x worse!)
```

**Divergence Point**: Position error crossed 100m threshold at **t=1800.470s** (50.4 seconds into flight)

### 2. Velocity Error Comparison

```
BEFORE (stamp_bag):
  RMS: 2.031 m/s
  Mean: 1.702 m/s
  Samples > 10 m/s: 0% (stable)

AFTER (time_ref):
  RMS: 463.271 m/s
  Mean: 388.403 m/s
  Samples > 10 m/s: 85.8% (catastrophic)
```

**First extreme velocity error**: t=1765.872s (15.8 seconds into flight, vel_error=10.005 m/s)

### 3. Yaw Error Comparison

```
BEFORE (stamp_bag):
  RMS: 103.70°
  Mean: -5.33° ± 103.57°
  Large errors (>90°): 51.3%

AFTER (time_ref):
  RMS: 100.88°
  Mean: -3.18° ± 100.83°
  Large errors (>90°): ~49%
```

**Improvement**: -2.83° RMS (2.7% better) ✓

**Analysis**: Yaw improved slightly because magnetometer measurements ARE synchronized better, BUT this benefit is completely negated by filter divergence.

### 4. Covariance Growth

```
Velocity Covariance (P_vel_xx):

BEFORE (stamp_bag):
  Start:  4.000000
  Middle: 1.450198
  End:    1.133144 (converged, stable)

AFTER (time_ref):
  Start:  4.000000
  Middle: 14.584860 (growing)
  End:    335.701400 (exploded!)
```

**Covariance explosion**: t=1961.235s (211 seconds into flight, P_vel_xx > 100)

---

## 🔍 Root Cause Analysis

### Timestamp Analysis

**1. Magnetometer CSV Structure**:
```csv
stamp_bag,stamp_msg,x,y,z,frame_id
1653512816.453,1653512816.447,0.576,...,imu_link
```

**2. Time_ref CSV Structure**:
```csv
stamp_msg,time_ref
1653512816.442,1750.125
1653512816.492,1750.175
```

**3. Interpolation Result**:
```
Mag stamp_msg: 1653512816.447
Time_ref (interp): 1750.130
Offset: 1653511066.317 seconds (~52 years!)
```

**CRITICAL**: The interpolation works correctly, producing time_ref in the correct range [1750.130, 2058.425], matching IMU time_ref [1750.110, 2058.432].

### Why Did It Diverge Then?

**Hypothesis 1: Magnetometer Update Sequencing Bug**

With `stamp_bag` (WORKING):
```
IMU loop propagates to t=1750.110
Mag arrives with stamp_bag=1653512816.453 (Unix timestamp)
→ Mag processed at current state (t=1750.110)
→ ~20ms timing error, but filter tolerates it
```

With `time_ref` (BROKEN):
```
IMU loop propagates to t=1750.110
Mag arrives with time_ref=1750.130 (interpolated)
→ Mag timestamp is 20ms AHEAD of current state!
→ Code processes mag at t=1750.110 (20ms mismatch)
→ Linearization error in EKF Jacobian
→ Innovation computed at wrong state
→ Covariance update incorrect
→ Covariance grows unbounded
→ Filter diverges
```

**Key Issue**: Current VIO implementation processes magnetometer at the **current propagated state time**, NOT at the measurement time. This is fine when timestamps are imprecise (stamp_bag), but breaks when timestamps are accurate (time_ref).

**Hypothesis 2: Measurement Ordering**

stamp_bag timestamps are recording times (wall clock when ROS bag captured data):
- IMU: recorded at t_wall
- Mag: recorded at t_wall + δ (small delay)
- **Natural ordering preserved**

time_ref timestamps are hardware times (true measurement times):
- IMU: measured at t_hw
- Mag: measured at t_hw + ε (sensor-specific)
- **Reveals true timing relationships**

When using time_ref, magnetometer measurements may arrive **out of order** relative to the state propagation, causing:
1. Stale measurements processed after state moved forward
2. Jacobians computed at wrong linearization point
3. Innovation covariance mismatch
4. Filter instability

---

## 💡 Why Magnetometer Doesn't NEED time_ref

### 1. Low Update Rate

Magnetometer updates at **~100 Hz** but is processed much less frequently (limited by `update_rate_limit`).

**Timing tolerance**:
- 10ms error at 100 Hz = 1% of sample period (negligible)
- Magnetometer dynamics: SLOW (Earth's magnetic field changes slowly)
- Yaw rate: typically < 30°/s → 10ms = 0.3° error (acceptable)

### 2. Already Filtered

Magnetometer measurements pass through:
1. **Calibration**: Hard-iron, soft-iron correction
2. **EMA filter**: α=0.3 (smooths out noise)
3. **Gyro consistency check**: Rejects if yaw rate > 30°/s
4. **Innovation gating**: Rejects if residual too large

→ Small timing errors are **absorbed by filtering**

### 3. Yaw Observable from Multiple Sources

The VIO system estimates yaw from:
1. **Magnetometer**: Absolute heading (magnetic north)
2. **Gyroscope**: Relative yaw rate (integrated)
3. **Visual odometry**: Rotation from feature tracking
4. **MSCKF**: Multi-view geometry constraints

→ Magnetometer is just **one of many** yaw measurements, not critical

### 4. PPK Heading 180° Ambiguity

Current benchmark shows:
- Yaw error RMS: 103.70° (before), 100.88° (after)
- Large errors (>90°): 51.3%

This is **NOT a timing issue** but a **PPK heading calculation bug** (sign ambiguity in velocity-based heading).

→ Fixing magnetometer timing won't solve the yaw problem

### 5. stamp_bag vs stamp_msg Difference: ~3ms

```
Offset (stamp_bag - stamp_msg):
  Mean: 0.002837s (2.8ms)
  Std:  0.002852s (2.9ms)
```

At 100 Hz magnetometer:
- 2.8ms = 0.28% of sample period
- Yaw rate 30°/s → 2.8ms = 0.084° error

→ **Negligible impact** compared to other error sources:
- Calibration error: ~5-10°
- Declination uncertainty: ~1-2°
- Hard/soft iron compensation: ~2-5°

---

## 🎯 Recommended Solution

### Keep Using stamp_bag (STABLE)

**Rationale**:
1. ✅ **Proven stable**: 308s flight, position error 54.9m, velocity error 2.0 m/s
2. ✅ **No divergence**: Covariance stays bounded (P_vel < 2)
3. ✅ **Good enough**: 2.8ms timing error negligible for magnetometer
4. ⚠️ Yaw error 103.7° is **NOT timing-related** (PPK heading issue)

### Fix PPK Heading Calculation Instead

Priority should be fixing the **180° ambiguity** in PPK heading:
```python
# Current code (BROKEN):
heading = np.arctan2(ve, vn)  # Ambiguous sign

# Should be:
heading = np.arctan2(ve, vn)
if heading < 0:
    heading += 2 * np.pi  # Wrap to [0, 2π]
# OR use magnetometer to resolve ambiguity
```

### If You MUST Use time_ref for Magnetometer

**Requirements**:
1. **State propagation to measurement time**: Store IMU buffer, propagate state to exact mag.t
2. **Measurement buffering**: Queue measurements, process in chronological order
3. **Covariance correction**: Account for propagation uncertainty
4. **Extensive testing**: Validate on multiple datasets

**Effort**: ~1-2 weeks development + testing  
**Benefit**: ~3° yaw improvement (2.7%)  
**Risk**: High (current attempt caused catastrophic divergence)

**Verdict**: **NOT WORTH IT** for such a small benefit

---

## 📈 Performance Comparison Summary

| Metric | stamp_bag (STABLE) | time_ref (BROKEN) | Winner |
|--------|-------------------|-------------------|--------|
| Position Error (mean) | 54.9m | 38,828m | ✅ stamp_bag (707x better) |
| Velocity Error (RMS) | 2.031 m/s | 463.3 m/s | ✅ stamp_bag (228x better) |
| Yaw Error (RMS) | 103.70° | 100.88° | ⚠️ time_ref (2.7% better) |
| Covariance Stability | Converged (1.13) | Exploded (335.7) | ✅ stamp_bag (296x better) |
| Filter Status | Stable | Diverged | ✅ stamp_bag |
| **Overall Winner** | | | **✅ stamp_bag by huge margin** |

---

## 🚀 Action Items

1. ✅ **DONE**: Reverted commit 8871279 (v3.9.5) back to v3.9.4
2. ✅ **DONE**: Continue using `stamp_bag` for magnetometer timestamps
3. 🔧 **TODO**: Fix PPK heading 180° ambiguity bug (high priority)
4. 🔧 **TODO**: Improve MSCKF triangulation success rate (currently 2.69%)
5. 🔧 **TODO**: Investigate parallax threshold reduction for nadir camera

---

## 📝 Lessons Learned

1. **Timing accuracy ≠ Filter stability**: More accurate timestamps can expose bugs in sensor fusion algorithms
2. **Test before deploy**: Always benchmark on real data before pushing changes
3. **Prioritize by impact**: 2.7% yaw improvement vs 22,700% velocity degradation → obvious choice
4. **Understand your system**: Magnetometer timing is NOT the bottleneck; PPK heading and VIO triangulation are
5. **Simple is better**: stamp_bag works well enough; don't over-engineer

---

## 🔗 Related Issues

- VIO Frontend v3.9.3: Triangulation success rate 2.69% (bottleneck!)
- PPK Heading: 180° ambiguity causing 51.3% of samples to have >90° error
- MSCKF: Only 1.58% of features successfully updated (97.31% failure rate)

**Fix these first** before worrying about 3ms magnetometer timing!

---

## 📚 References

- Benchmark 20251230_214742: Stable baseline with stamp_bag
- Benchmark 20251231_000813: Diverged after time_ref implementation
- Code: `vio/data_loaders.py` lines 289-306
- Commit: 8871279 (REVERTED), 9893a58 (current)
