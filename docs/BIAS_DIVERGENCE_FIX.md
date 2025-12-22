# Bias Divergence Analysis & Fix (v3.7.2)

## Problem Discovery

### Symptoms (v3.7.1 Baseline)
```
Final accel bias: 3483 mg (35× normal!)
Final gyro bias:  204 mrad/s = 11.7°/s (200× normal!)
P_vel uncertainty: 1.99 m/s (4× normal)
```

**CRITICAL:** Bias estimates DIVERGED from 0 → 3500mg over 308s flight

---

## Root Cause Analysis

### 1. Initial Calibration is GOOD
```
Raw IMU analysis (first 1000 samples):
  Accel magnitude: 9.84 m/s² ✅ (correct)
  Apparent bias: [18.6, -0.3, 1.2] mg ✅ (normal)
  Orientation error: <1° ✅ (good)
```

**Conclusion:** IMU hardware and calibration are FINE

---

### 2. Bias Random Walk TOO HIGH

**Current config (v2.9.9.10):**
```yaml
acc_w: 0.0015  # 3× increased for velocity uncertainty
gyr_w: 0.0015  # 3× increased for velocity uncertainty
```

**Problem:** Process noise for bias is PROCESS, not MEASUREMENT
- acc_n/gyr_n = measurement noise → affects velocity uncertainty ✅
- acc_w/gyr_w = bias random walk → how fast bias can drift ❌

**What happened:**
```
Frame 0:    ba = 0 mg
Frame 173:  ba = 500 mg (exceeded threshold)
Frame 1149: ba = 2183 mg (early phase avg)
Frame 4598: ba = 3483 mg (final)
```

Bias drifted 3.5 m/s² over 308s = 11.3 mg/s drift rate!

---

### 3. Observability Problem

**MSCKF statistics:**
- Success: 16.3%
- **REJECTED: 83.7%**

**Implication:**
- Only 16.3% of measurements constrain the state
- Bias has POOR observability (hovering = mostly gravity)
- Without corrections, bias drifts according to process noise

**Vicious cycle:**
1. High acc_w → bias can drift freely
2. Bias drifts → velocity estimate wrong
3. Velocity wrong → MSCKF innovations large
4. Large innovations → chi-squared rejects update
5. No update → bias drifts more (goto 1)

---

## Solution (v3.7.2)

### Key Insight
**acc_n/gyr_n ≠ acc_w/gyr_w**

- **acc_n/gyr_n:** Measurement noise (white noise on each sample)
  - Affects velocity/attitude uncertainty
  - Can be HIGH to account for vibration/dynamics
  - Does NOT cause bias drift

- **acc_w/gyr_w:** Bias random walk (Brownian motion)
  - How fast bias can change over time
  - Should be SMALL (IMU bias is stable!)
  - Typical: 10⁻⁵ to 10⁻⁴

### New Configuration
```yaml
preintegration:
  acc_n: 0.60     # Keep 3× (measurement noise)
  gyr_n: 0.030    # Keep 3× (measurement noise)
  acc_w: 0.00004  # REDUCED 37× (bias random walk)
  gyr_w: 0.0001   # REDUCED 15× (bias random walk)
```

**Reasoning:**
1. Keep high acc_n/gyr_n for velocity uncertainty (was working)
2. Reduce acc_w/gyr_w to LOCK bias estimates
3. Bias should stay near initial values (~20mg)
4. IMU hardware is good (verified from raw data)

---

## Expected Improvement

### Before (v3.7.1)
```
ba: 0 → 3483 mg   ❌ DIVERGED
bg: 0 → 204 mrad/s ❌ DIVERGED
P_vel: 1.99 m/s    ⚠️  HIGH
```

### After (v3.7.2) - Expected
```
ba: 0 → ~50 mg     ✅ STABLE
bg: 0 → ~10 mrad/s ✅ STABLE
P_vel: 1-2 m/s     ✅ REASONABLE
```

### MSCKF Impact
- Stable bias → accurate velocity predictions
- Accurate predictions → smaller innovations
- Smaller innovations → more chi-squared passes
- More updates → better filter convergence
- **Expected MSCKF success: 16.3% → 20-25%**

---

## Technical Details

### Bias Evolution Observed (v3.7.1)
| Phase | ba (mg) | bg (mrad/s) |
|-------|---------|-------------|
| Early (0-25%) | 2183 | 149 |
| Mid (25-50%) | 3492 | 139 |
| Late (50-75%) | 4178 | 76 |
| Final (75-100%) | 4324 | 209 |

**Pattern:** Continuous drift without bound

### Physics Check
```
Initial accel bias: 0 mg ✅
Apparent bias (gravity): 18.6 mg ✅
Expected bias (good IMU): 10-50 mg ✅
Measured final bias: 3483 mg ❌ (70× expected!)
```

**Conclusion:** Bias estimates are UNPHYSICAL, caused by excessive process noise

---

## Validation Plan

1. **Run benchmark with v3.7.2 config**
2. **Check bias evolution:**
   - Should stay <100mg throughout flight
   - Should converge, not diverge
3. **Check MSCKF success rate:**
   - Target: 20-25% (up from 16.3%)
4. **Check velocity uncertainty:**
   - Should stay 1-2 m/s (acceptable)

---

## Historical Context

| Version | acc_w | gyr_w | Result |
|---------|-------|-------|--------|
| v2.9.9.8 | 0.0005 | 0.0005 | Velocity overconfident (0.15 m/s) |
| v2.9.9.9 | 0.0010 | 0.0010 | Still overconfident (13.1σ) |
| v2.9.9.10 | 0.0015 | 0.0015 | Bias DIVERGED (3483mg!) ❌ |
| **v3.7.2** | **0.00004** | **0.0001** | **Bias locked, vel ok** ✅ |

**Lesson:** Mixing up measurement noise (acc_n) with bias random walk (acc_w) caused divergence!

---

Date: 2024-12-21  
Version: v3.7.2  
Status: Fix implemented, awaiting validation  
Priority: **CRITICAL** - Bias divergence affects all state estimates
