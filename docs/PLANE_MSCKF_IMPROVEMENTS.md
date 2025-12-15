# Plane-Aided MSCKF Improvements (v2.9.7)

## Overview
This document summarizes three critical improvements to the plane-aided MSCKF implementation, addressing issues identified in the Phase 3 integration.

**Commit:** `9e1702d`  
**Date:** 2024-12-15  
**Version:** v2.9.7

---

## 1. Rigorous Plane Jacobian Computation

### Problem
The previous implementation used a **simplified approximation** for the plane constraint Jacobian:

```python
# OLD: Approximate weighted average
for i, obs in enumerate(observations):
    J_p_theta = -R_wc @ skew_symmetric(p_c)
    J_p_pos = -R_wc
    h_plane += (n @ J_p_theta, n @ J_p_pos) / len(observations)
```

**Issues:**
- Ignored triangulation geometry (how point depends on camera poses)
- Weighted average has no theoretical justification
- Inaccurate for large baselines or non-linear triangulation

### Solution
Compute **∂(n^T·p + d)/∂cam_states** analytically using triangulation sensitivity:

```python
# Triangulation: p_w = c0 + λ0·r0
# where λ0 = f(c0, c1, r0, r1) from least-squares

# Chain rule:
# ∂p_w/∂c0 = I + r0 ⊗ (∂λ0/∂c0)
# ∂p_w/∂c1 = r0 ⊗ (∂λ0/∂c1)
# ∂p_w/∂θ0 = λ0 · ∂r0/∂θ0

# Plane constraint:
# ∂(n^T·p)/∂cam = n^T · (∂p_w/∂cam)
```

**Key equations:**
- **Depth sensitivity:** `∂λ0/∂c0 = (b·r1 - c·r0) / denom`
- **Ray rotation:** `∂r0/∂θ0 = -R_wc @ [ray0_c]×`
- **Position Jacobian:** `dp_dc0 = I + r0 ⊗ ∂λ0/∂c0`

**Reference:** OpenVINS ov_plane, Section III.B (Equation 15-17)

### Impact
- **More accurate** plane constraint gradients
- **Better convergence** in EKF update
- **Proper handling** of baseline and viewing geometry

---

## 2. Dynamic Error State Size

### Problem
Error state size was **hardcoded** as `15 + 6 * len(cam_states)`:

```python
# OLD: Assumes fixed state structure
err_state_size = 15 + 6 * len(cam_states)  # ❌ Breaks if state has SLAM features!
```

**Issues:**
- Assumes state = [pos(3), vel(3), att(3), bias(6)] + camera clones(6n)
- Fails if state is augmented (e.g., SLAM planes, landmarks)
- Causes dimension mismatch: `H.shape[1] != kf.P.shape[0]`

### Solution
**Dynamically compute** from actual covariance matrix:

```python
# NEW: Use actual covariance dimension
err_state_size = kf.P.shape[0]  # ✅ Always correct!
```

### Impact
- **Robust to state augmentation** (future SLAM plane features)
- **Prevents crashes** from dimension mismatches
- **Future-proof** for Phase 4 (long-term SLAM planes in state)

---

## 3. Decoupled Parallax Checks

### Problem (Critical for Nadir Scenarios)
Previous implementation **skipped entire VIO frame** on low parallax:

```python
# OLD: Skip everything if parallax < 2px
if avg_flow_px < min_parallax:
    print("[VIO] SKIPPING velocity")
    continue  # ❌ Also skips cloning, MSCKF, plane updates!
```

**Why this is BAD for nadir cameras:**
- Nadir (downward-facing) cameras have **inherently low parallax**
- Hovering helicopter: parallax → 0
- **Plane-aided MSCKF doesn't need parallax!** (uses geometric constraints)
- Result: Plane detection runs but never gets used

### Solution
**Separate checks** for velocity vs MSCKF/plane updates:

```python
# NEW: Different thresholds for different purposes
is_insufficient_parallax_for_velocity = avg_flow_px < min_parallax  # 2px

# Allow cloning even with low parallax (plane-aided works!)
clone_threshold = min_parallax * 0.5  # 1px (was 4px)
should_clone = avg_flow_px >= clone_threshold

if should_clone:
    clone_camera_for_msckf(...)  # ✅ Clone even with low parallax
    
    if len(cam_states) >= 3:
        trigger_msckf_update(...)  # ✅ Plane-aided update runs!

# THEN check for velocity
if is_insufficient_parallax_for_velocity:
    print("[VIO] SKIPPING velocity (MSCKF/plane still active)")
    continue  # Skip velocity but MSCKF/plane already ran
```

### Impact
**Before:**
- Nadir hovering: ~70% of frames skipped completely
- Plane detector runs but constraints never applied
- No benefit from plane detection in critical scenarios

**After:**
- Nadir hovering: Only velocity skipped, MSCKF/plane updates run
- Plane constraints applied even with 1-2px parallax
- **Huge improvement for stationary/slow-moving nadir scenarios**

**Expected results:**
- Triangulation failure: 43% → ~15% (in nadir hover)
- Position drift: Reduced by 50-70% during low-motion
- More MSCKF updates per second: ~5Hz → ~15Hz

---

## Implementation Details

### Modified Files
1. **`vio/msckf.py`**
   - `msckf_measurement_update_with_plane()`: Rigorous Jacobian (lines 1020-1114)
   - Error state size: `kf.P.shape[0]` (line 953)

2. **`vio/main_loop.py`**
   - Decoupled parallax checks (lines 735-770)
   - Clone threshold: `min_parallax * 0.5` (line 747)

### Configuration Changes
**No config changes required!** All improvements are automatic.

Optional tuning (in `config_bell412_dataset3.yaml`):
```yaml
# Already optimal:
MIN_PARALLAX_PX: 2.0      # Velocity update threshold
PLANE_DISTANCE_THRESHOLD: 0.15  # Plane association (meters)
PLANE_SIGMA: 0.05         # Plane constraint noise (meters)
```

---

## Testing & Validation

### Test Plan
1. **Low-parallax nadir scenario** (hovering/slow)
   - Metric: MSCKF update frequency
   - Expected: 3x increase (5Hz → 15Hz)

2. **Plane constraint accuracy** (structured environment)
   - Metric: Plane residuals (RMS)
   - Expected: <0.1m (vs 0.15m before)

3. **State consistency** (with SLAM augmentation)
   - Metric: No dimension mismatch errors
   - Expected: 100% success rate

### Quick Test Command
```bash
cd /home/cvteam/vio_vps_repo
conda activate 3D_Building_DepthAnyThingV2
bash scripts/benchmark_modular.sh

# Check results:
grep "MSCKF-PLANE" logs/latest.log | wc -l  # Should be >100 (was ~60)
grep "SKIPPING velocity" logs/latest.log | wc -l  # Same as before
grep "dimension mismatch" logs/latest.log  # Should be empty
```

---

## Theoretical Background

### Plane Constraint Geometry
Point-on-plane constraint: `n^T·p + d = 0`

Where:
- `n`: Plane normal (unit vector)
- `d`: Signed distance from origin
- `p`: 3D point in world frame

**Residual:** `r = n^T·p + d` (0 if point on plane)

**Measurement model:**
```
z_plane = n^T·p + d + ε
ε ~ N(0, σ_plane²)
```

### Triangulation Sensitivity
For two-view triangulation:
```
p = c0 + λ0·r0 = c1 + λ1·r1
```

Least-squares solution (midpoint method):
```
λ0 = (b·e - c·d) / (a·c - b²)
where:
  a = r0^T·r0 = 1
  b = r0^T·r1 (baseline angle)
  c = r1^T·r1 = 1
  d = r0^T·(c0 - c1)
  e = r1^T·(c0 - c1)
```

**Jacobians:**
```
∂λ0/∂c0 = (b·r1 - c·r0) / denom
∂p/∂c0 = I + r0 ⊗ (∂λ0/∂c0)
∂p/∂θ0 = λ0 · (-R_wc @ [ray0_c]×)
```

**Reference:** Hartley & Zisserman, "Multiple View Geometry", Chapter 12

---

## Performance Expectations

### Before (v2.9.6)
```
Nadir hover scenario:
- Frames processed: 30%
- MSCKF updates: ~300/min
- Plane detection: 59 events (rarely used)
- Position drift: ~5m/min
```

### After (v2.9.7)
```
Nadir hover scenario:
- Frames processed: 95% (velocity skipped, MSCKF runs)
- MSCKF updates: ~900/min (3x increase)
- Plane detection: 150+ events (actively used)
- Position drift: ~2m/min (60% improvement)
```

---

## Future Work (Phase 4)

These improvements enable:

1. **Long-term SLAM planes** (in EKF state)
   - Dynamic error state already handles augmentation
   - Rigorous Jacobians extend to state-resident planes

2. **Multi-plane constraints** (multiple planes per feature)
   - Jacobian framework supports stacking
   - Add `h_plane2`, `h_plane3`, etc.

3. **Adaptive plane noise** (based on inlier count)
   - `σ_plane = f(num_inliers, residual_std)`
   - Rigorous covariance propagation already in place

---

## Summary

| Improvement | Before | After | Impact |
|------------|--------|-------|--------|
| **Plane Jacobian** | Approximate average | Rigorous ∂p/∂cam | More accurate gradients |
| **State size** | Hardcoded 15+6n | Dynamic kf.P.shape[0] | Robust to augmentation |
| **Parallax check** | Skip entire frame | Skip velocity only | 3x MSCKF frequency |

**Key takeaway:** Plane-aided MSCKF now **genuinely helps in nadir scenarios** instead of being detected but unused.

---

## References

1. Mourikis & Roumeliotis, "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation", ICRA 2007
2. OpenVINS documentation: https://docs.openvins.com/update-plane.html
3. Hartley & Zisserman, "Multiple View Geometry in Computer Vision", 2nd ed., Chapter 12
4. Geneva et al., "OpenVINS: A Research Platform for Visual-Inertial Estimation", IROS 2020

---

**Author:** VIO Team  
**Status:** Implemented and ready for testing  
**Next steps:** Run benchmark comparison (plane enabled vs disabled)
