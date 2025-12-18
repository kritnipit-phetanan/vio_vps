# Preintegration Architecture Analysis & Recommendations

**Date:** December 18, 2025  
**Version:** VIO v3.4.0  
**Issue:** State propagation inconsistency with measurement updates

---

## คำถาม 1: ความแตกต่างระหว่าง `use_preintegration: True` vs `False`

### **A. use_preintegration: False (Legacy Mode)**

**ตำแหน่งโค้ด:** [main_loop.py:1410-1414](vio/main_loop.py#L1410-L1414)

```python
# Main loop (every IMU sample @ 400Hz)
for i, rec in enumerate(self.imu):
    if self.config.use_preintegration and ongoing_preint is not None:
        ongoing_preint.integrate_measurement(rec.ang, rec.lin, dt)  # ✅ Accumulate only
    else:
        self.process_imu_sample(rec, dt)  # ✅ Propagate immediately
```

**การทำงาน:**
1. **Propagate ทุก IMU sample** (400 Hz)
   - เรียก `process_imu()` → update `kf.x`, `kf.P` ทันที
   - State อยู่ที่เวลา `t` ปัจจุบันเสมอ

2. **Measurement updates** (VPS/MAG) apply ได้ทันที
   - State propagated แล้ว → ไม่มีปัญหา timing

3. **ข้อเสีย:**
   - Propagate 400 ครั้ง/วินาที → slow (แต่ accurate)
   - Noise accumulation สูง (propagate หลายครั้ง)

---

### **B. use_preintegration: True (Current Implementation)**

**ตำแหน่งโค้ด:** [main_loop.py:1410-1414](vio/main_loop.py#L1410-L1414) + [main_loop.py:768-770](vio/main_loop.py#L768-L770)

```python
# Main loop (every IMU sample @ 400Hz)
for i, rec in enumerate(self.imu):
    if self.config.use_preintegration and ongoing_preint is not None:
        ongoing_preint.integrate_measurement(rec.ang, rec.lin, dt)  # ❌ Accumulate only, NO state update
    
    # ❌ PROBLEM: State NOT propagated yet!
    self.process_vps(t)           # Apply VPS update on OLD state
    self.process_magnetometer(t)   # Apply MAG update on OLD state
    
    # ✅ State update happens ONLY at camera frame (20 Hz)
    used_vo, vo_data = self.process_vio(rec, t, ongoing_preint)
    # Inside process_vio():
    if self.config.use_preintegration and ongoing_preint is not None:
        apply_preintegration_at_camera(kf, ongoing_preint, t, imu_params)  # Update state
```

**การทำงาน:**
1. **Accumulate IMU ไว้** (400 Hz) → ไม่ propagate state ทันที
   - State ยังอยู่ที่เวลา `t_camera_last` (เก่า)
   
2. **Apply preintegration** เฉพาะเวลา camera frame (20 Hz)
   - Propagate state ด้วย accumulated Δ (20 ครั้ง/วินาที แทน 400 ครั้ง)
   - Less noise, faster

3. **❌ ปัญหา:** Measurement updates (VPS/MAG) ถูก apply บน **stale state**
   - VPS/MAG เรียกทุก IMU step แต่ state ยังไม่ถูก propagate
   - Innovation/residuals ผิดพลาด → poor filter performance

---

## คำถาม 2: ความแตกต่างของ naming convention `initialize_vio_frontend` vs `_initialize_rectifier`

### **Python Naming Convention:**

| Pattern | Visibility | Usage | Example |
|---------|-----------|-------|---------|
| `method_name` | **Public** | ควรถูกเรียกจากภายนอก class | `initialize_vio_frontend()` |
| `_method_name` | **Protected** | Internal helper, ไม่ควรเรียกจากภายนอก | `_initialize_rectifier()` |
| `__method_name` | **Private** | Name mangling, ห้ามเรียกจากภายนอก | `__compute_jacobian()` |

### **ใน VIORunner:**

```python
class VIORunner:
    # ===== Public methods (เรียกจาก run() หลัก) =====
    def initialize_vio_frontend(self):
        """Initialize VIO frontend - CORE functionality"""
        # Setup camera, feature tracking, MSCKF
    
    def process_imu_sample(self, rec, dt):
        """Process IMU - CORE functionality"""
        # IMU propagation + ZUPT
    
    # ===== Protected methods (optional subsystems) =====
    def _initialize_rectifier(self):
        """Initialize fisheye rectifier - OPTIONAL feature"""
        # Only if USE_RECTIFIER=true in config
    
    def _initialize_loop_closure(self):
        """Initialize loop closure - OPTIONAL feature"""
        # Only if USE_LOOP_CLOSURE=true in config
```

**เหตุผล:**
- **Public (`initialize_vio_frontend`)**: CORE functionality, จำเป็นต่อการทำงาน
- **Protected (`_initialize_rectifier`)**: OPTIONAL subsystem, อาจไม่ถูกใช้

---

## คำถาม 3: ปัญหา Architecture - Measurement Updates on Stale State

### **สาเหตุหลัก:**

```python
# CURRENT IMPLEMENTATION (v3.4.0) - ❌ WRONG!
for i, rec in enumerate(self.imu):
    if use_preintegration:
        ongoing_preint.integrate_measurement(rec.ang, rec.lin, dt)  # Only accumulate
        # ❌ State is STALE (still at last camera time)
    
    self.process_vps(t)           # ❌ Apply VPS on STALE state
    self.process_magnetometer(t)  # ❌ Apply MAG on STALE state
    
    # State updated ONLY at camera frame (20 Hz)
    if is_camera_frame:
        apply_preintegration_at_camera(...)  # ✅ Finally propagate state
```

**ปัญหา:**
1. State อยู่ที่ `t_camera_last` (เก่า)
2. VPS/MAG measurement อยู่ที่ `t_current` (ใหม่)
3. **Time mismatch** → innovation/residuals ผิดพลาด

**ตัวอย่าง:**
```
t=0.00s: Camera frame → State propagated to t=0.00s
t=0.05s: VPS measurement → Apply on state at t=0.00s (❌ 50ms stale!)
t=0.10s: MAG measurement → Apply on state at t=0.00s (❌ 100ms stale!)
t=0.15s: Camera frame → State finally propagated to t=0.15s
```

---

## แนวทางแก้ไข (OpenVINS-style Nominal Propagation)

### **Architecture ที่ถูกต้อง:**

```python
# ===== PROPOSED SOLUTION (OpenVINS-style) =====

for i, rec in enumerate(self.imu):
    # ===== STEP 1: Nominal Propagation (LIGHTWEIGHT) =====
    # Propagate state ทุก IMU sample แต่เบา ๆ (ไม่ต้อง update P)
    propagate_nominal_state(kf, rec, dt, bg, ba)
    # → State ถูก propagate ไปที่ t_current แล้ว
    
    # ===== STEP 2: Accumulate for Preintegration =====
    if use_preintegration:
        ongoing_preint.integrate_measurement(rec.ang, rec.lin, dt)
        # → เก็บ Δ ไว้สำหรับ covariance propagation
    
    # ===== STEP 3: Measurement Updates (on FRESH state) =====
    self.process_vps(t)           # ✅ State at t_current
    self.process_magnetometer(t)  # ✅ State at t_current
    
    # ===== STEP 4: Covariance Propagation (at camera frame only) =====
    if is_camera_frame:
        # Use preintegrated Δ to propagate P (20 Hz instead of 400 Hz)
        propagate_covariance_with_preintegration(kf, ongoing_preint, ...)
        ongoing_preint.reset()
```

### **ประโยชน์:**
1. ✅ **State fresh** → Measurement updates ถูกต้อง
2. ✅ **Covariance propagation เบา** → 20 Hz แทน 400 Hz
3. ✅ **Best of both worlds** → Accuracy + Performance

---

## Implementation Details

### **1. Nominal Propagation (Light)**

**ไฟล์:** `vio/propagation.py`

```python
def propagate_nominal_state(kf: ExtendedKalmanFilter, 
                            rec: IMURecord, dt: float,
                            bg: np.ndarray, ba: np.ndarray,
                            imu_params: dict):
    """
    Lightweight nominal state propagation (position, velocity, rotation).
    Does NOT update covariance (P matrix) - that's done with preintegration.
    
    This ensures state is always at current time for measurement updates.
    
    Args:
        kf: Kalman filter
        rec: IMU record (ang, lin)
        dt: Time step
        bg: Gyro bias
        ba: Accel bias
        imu_params: IMU parameters (g_norm)
    """
    from scipy.spatial.transform import Rotation as R_scipy
    from .math_utils import skew_symmetric
    
    # Bias-corrected measurements
    w_corr = rec.ang - bg
    a_corr = rec.lin - ba
    
    # Current state
    p = kf.x[0:3, 0]
    v = kf.x[3:6, 0]
    q = kf.x[6:10, 0]  # [w,x,y,z]
    
    # Rotation update (discrete integration)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    R_BW = R_scipy.from_quat(q_xyzw).as_matrix()
    
    # Discrete rotation: exp(w * dt) ≈ I + sin(θ)/θ * [w×] + (1-cos(θ))/θ² * [w×]²
    theta = np.linalg.norm(w_corr) * dt
    if theta < 1e-8:
        delta_R = np.eye(3) + skew_symmetric(w_corr * dt)
    else:
        axis = w_corr / np.linalg.norm(w_corr)
        delta_R = R_scipy.from_rotvec(axis * theta).as_matrix()
    
    R_BW_new = R_BW @ delta_R
    q_new_xyzw = R_scipy.from_matrix(R_BW_new).as_quat()
    q_new = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]])
    
    # Velocity update (world frame acceleration)
    g_world = np.array([0, 0, imu_params.get('g_norm', 9.803)])
    a_world = R_BW @ a_corr - g_world
    v_new = v + a_world * dt
    
    # Position update (trapezoidal integration)
    p_new = p + (v + v_new) / 2.0 * dt
    
    # Write back to state (only p, v, q - NOT bias or clones)
    kf.x[0:3, 0] = p_new
    kf.x[3:6, 0] = v_new
    kf.x[6:10, 0] = q_new
```

### **2. Modified Main Loop**

**ไฟล์:** `vio/main_loop.py`

```python
def run(self):
    # ... (initialization) ...
    
    for i, rec in enumerate(self.imu):
        t = rec.t
        dt = max(0.0, float(t - self.state.last_t)) if i > 0 else 0.0
        
        # ===== NEW: Always propagate nominal state (LIGHT) =====
        if self.config.use_preintegration and ongoing_preint is not None:
            # Accumulate for covariance propagation
            ongoing_preint.integrate_measurement(rec.ang, rec.lin, dt)
            
            # Lightweight nominal propagation (keep state fresh)
            bg = self.kf.x[10:13, 0]
            ba = self.kf.x[13:16, 0]
            propagate_nominal_state(self.kf, rec, dt, bg, ba, imu_params)
        else:
            # Legacy mode: full propagation (state + covariance)
            self.process_imu_sample(rec, dt)
        
        # ===== Measurement updates (now on FRESH state) =====
        self.process_vps(t)
        if self.config.use_magnetometer:
            self.process_magnetometer(t)
        
        # ===== Camera frame: Propagate covariance =====
        used_vo, vo_data = self.process_vio(rec, t, ongoing_preint)
        # Inside process_vio() at camera frame:
        if self.config.use_preintegration:
            # Use preintegrated Δ to propagate covariance only
            propagate_covariance_with_preintegration(self.kf, ongoing_preint, ...)
            ongoing_preint.reset()
```

### **3. Modified Covariance Propagation**

**ไฟล์:** `vio/propagation.py`

```python
def propagate_covariance_with_preintegration(kf: ExtendedKalmanFilter,
                                             ongoing_preint,
                                             imu_params: dict) -> dict:
    """
    Propagate covariance using preintegrated measurements.
    
    NOTE: State (x) is already updated by nominal propagation!
    This function ONLY updates covariance (P).
    
    Returns:
        preint_jacobians: For bias observability in MSCKF
    """
    dt_total = ongoing_preint.dt_sum
    if dt_total < 1e-6:
        return None
    
    # Get Jacobians from preintegration
    J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba = ongoing_preint.get_jacobians()
    preint_cov = ongoing_preint.get_covariance()
    
    # Current rotation for projection
    q = kf.x[6:10, 0]
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    R_BW = R_scipy.from_quat(q_xyzw).as_matrix()
    
    # Build state transition matrix (error-state)
    num_clones = (kf.x.shape[0] - 16) // 7
    Phi = build_state_transition(R_BW, J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba, 
                                  dt_total, num_clones)
    
    # Process noise
    Q = build_process_noise(R_BW, preint_cov, imu_params, dt_total, num_clones)
    
    # Propagate covariance: P = Φ P Φ^T + Q
    kf.P = propagate_error_state_covariance(kf.P, Phi, Q, num_clones)
    
    return {
        'J_R_bg': J_R_bg, 'J_v_bg': J_v_bg, 'J_v_ba': J_v_ba,
        'J_p_bg': J_p_bg, 'J_p_ba': J_p_ba
    }
```

---

## Performance Comparison

| Mode | State Updates | Covariance Updates | Measurement Timing | Performance |
|------|---------------|-------------------|-------------------|-------------|
| **Legacy** | 400 Hz | 400 Hz | ✅ Fresh | Slow but accurate |
| **Current Preint** | 20 Hz | 20 Hz | ❌ Stale | Fast but inconsistent |
| **Proposed (OpenVINS)** | 400 Hz (light) | 20 Hz | ✅ Fresh | **Fast AND accurate** |

---

## References

1. **Forster et al. TRO 2017:** "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"
2. **OpenVINS:** [github.com/rpng/open_vins](https://github.com/rpng/open_vins)
   - See `Propagator::propagate_and_clone()` for reference implementation
3. **OKVIS:** "Keyframe-based visual-inertial odometry using nonlinear optimization"

---

## Summary

### **Current Issues:**
1. ❌ State not propagated at IMU rate when `use_preintegration=True`
2. ❌ Measurement updates (VPS/MAG) applied on stale state
3. ❌ Time mismatch causes poor innovation/residual quality

### **Proposed Solution:**
1. ✅ Lightweight nominal propagation at 400 Hz (keep state fresh)
2. ✅ Covariance propagation at 20 Hz (use preintegrated Δ)
3. ✅ Measurement updates on fresh state (correct timing)
4. ✅ Best performance + accuracy balance (OpenVINS-style)

### **Implementation Priority:**
1. **Phase 1:** Add `propagate_nominal_state()` function
2. **Phase 2:** Modify main loop to call nominal propagation
3. **Phase 3:** Refactor `apply_preintegration_at_camera()` → `propagate_covariance_with_preintegration()`
4. **Phase 4:** Benchmark performance vs current implementation

---

**Status:** Architecture recommendation ready for implementation  
**Next Step:** Implement Phase 1 - `propagate_nominal_state()` function
