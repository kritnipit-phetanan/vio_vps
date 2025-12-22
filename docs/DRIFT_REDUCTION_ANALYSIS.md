# Drift Reduction Analysis and Recommendations

## Current Performance (v3.9.0)
- **Position RMSE**: ~10-15 m over 308s (~5 min flight)
- **Altitude Error**: ~1 m (excellent with DEM updates)
- **Yaw Error**: ~10-15° (magnetometer-corrected)
- **Velocity RMSE**: Good correlation with optical flow

## Root Causes of Remaining Drift

### 1. **IMU Bias Accumulation** ⚠️ PRIMARY ISSUE
**Problem**: Gyro and accel biases drift over time, especially during temperature changes
- Gyro bias → yaw drift → horizontal position drift
- Accel bias → velocity error → position drift (quadratic)

**Current Status**: 
- `estimate_imu_bias: false` in config
- No online bias estimation

**Recommended Solutions**:
```yaml
imu:
  estimate_bias: true
  initial_gyro_bias: [0.001, -0.002, 0.003]  # From stationary calibration
  initial_accel_bias: [0.05, -0.03, 0.02]     # From static period
```

**Expected Improvement**: 30-50% drift reduction

---

### 2. **MSCKF Observability Issues**
**Problem**: XY position drift because MSCKF mainly observes:
- ✅ Altitude (Z) - excellent
- ✅ Yaw - good (mag correction)
- ⚠️ Horizontal (XY) - weak without VPS

**Why XY is weak**:
- Nadir camera: poor XY parallax (scale ambiguity)
- No absolute XY updates (VPS disabled)
- Optical flow helps but doesn't eliminate drift

**Solutions**:

#### A. Enable VPS Updates (Highest Impact)
```yaml
vio:
  use_vps: true  # Absolute XY position corrections
```
**Expected**: 70-80% XY drift reduction

#### B. Use Front/Side Camera (if available)
```yaml
vio:
  default_camera_view: front  # Better XY parallax
```
**Expected**: 20-30% drift reduction

#### C. Increase MSCKF Window Size
```yaml
msckf:
  max_clone_size: 15  # Default: 11
  min_track_length: 6  # Longer feature tracks
```
**Expected**: 10-15% improvement

---

### 3. **Feature Tracking Quality**
**Current Stats** (from debug logs):
- Features detected: 100-200 per frame
- Inliers: 30-80 (depends on motion)
- Parallax: 2-5 px (nadir view, low altitude)

**Issues**:
- Low parallax → scale drift
- Feature loss during fast rotation
- Blurry images (motion blur)

**Solutions**:

#### A. Adaptive Feature Detection
```python
# In vio_frontend.py
self.max_corners = 300  # Increase from 150
self.grid_x = 6         # Finer grid
self.grid_y = 5
```

#### B. ORB Features (rotation-invariant)
```python
self.detector = cv2.ORB_create(
    nfeatures=500,
    scaleFactor=1.2,
    nlevels=8
)
```

#### C. Enable Image Rectification (fisheye → pinhole)
```yaml
rectification:
  enabled: true
  fov_deg: 100  # Virtual pinhole FOV
```
**Expected**: 15-20% better feature tracking

---

### 4. **Magnetometer Yaw Drift**
**Current**: 10° yaw error → ~1.7m horizontal drift per 10m forward motion

**Root Cause**:
- Magnetic interference from motors
- Soft-iron effects from helicopter structure
- Dynamic calibration needed

**Solutions**:

#### A. Improve Mag Calibration
```python
# Run in-flight calibration
from vio.magnetometer import calibrate_magnetometer

# Collect data during figure-8 maneuver
mag_calib = calibrate_magnetometer(mag_samples_3d, method='ellipsoid_fit')
```

#### B. Adaptive Magnetometer Gating
```yaml
magnetometer:
  adaptive_gating: true
  max_yaw_rate_deg: 20  # Stricter gate during fast turns
  gyro_consistency_threshold_deg: 5
```

#### C. GPS Heading Aiding (when available)
Use velocity vector from GPS for heading initialization

**Expected**: 5° yaw error → 30% XY drift reduction

---

### 5. **Preintegration Noise Parameters**
**Problem**: Tuning affects state estimation accuracy

**Current**:
```yaml
imu:
  preintegration:
    acc_n: 0.08   # May be too optimistic
    gyr_n: 0.004
```

**Recommended**: Run Allan Variance analysis
```bash
python scripts/allan_variance.py --imu imu_static.csv
```

**Expected Output**:
```
Gyro noise density: 0.006 rad/s/√Hz
Accel noise density: 0.12 m/s²/√Hz
```

Update config accordingly → Better filter convergence

---

### 6. **DEM Vertical Constraint** ✅ Working Well
**Current**: Excellent altitude tracking (~1m error)

**Further Optimization**:
- Use higher resolution DEM (1m instead of 10m)
- Terrain-referenced navigation (TRN) for XY updates

```yaml
trn:
  enabled: true
  dem_resolution_m: 1.0
  update_interval_s: 2.0
```

---

## Recommended Priority Implementation

### **Phase 1: Quick Wins** (1-2 days)
1. ✅ Fix `timeref_csv` bug (DONE)
2. Enable IMU bias estimation
3. Tune MSCKF window size
4. Improve magnetometer calibration

**Expected**: 40-50% drift reduction

### **Phase 2: Feature Tracking** (3-5 days)
1. Increase feature detection density
2. Enable fisheye rectification
3. Implement ORB features for rotation-invariance

**Expected**: Additional 20-30% improvement

### **Phase 3: VPS Integration** (1 week)
1. Re-enable VPS updates
2. Implement loop closure detection
3. Add TRN for terrain-based XY updates

**Expected**: 70-80% total drift reduction

---

## Validation Metrics

Monitor these in `error_log.csv`:

| Metric | Current | Target (Phase 1) | Target (Phase 3) |
|--------|---------|------------------|------------------|
| Position RMSE | 12 m | 7 m | 3 m |
| Yaw Error | 12° | 7° | 3° |
| Altitude Error | 1 m | 0.8 m | 0.5 m |
| Final Drift Rate | 0.04 m/s | 0.02 m/s | 0.01 m/s |

---

## Code Changes Summary

### Immediate (config.yaml):
```yaml
imu:
  estimate_bias: true
  
msckf:
  max_clone_size: 15
  min_track_length: 6
  
magnetometer:
  adaptive_gating: true
  max_yaw_rate_deg: 20
```

### Next Sprint (vio_frontend.py):
```python
self.max_corners = 300
self.use_orb = True  # Switch to ORB
```

### Long-term (main_loop.py):
```python
if vps_available:
    apply_vps_update(...)

if trn_enabled:
    apply_trn_update(...)
```

---

## References
- OpenVINS: https://docs.openvins.com
- MSCKF Observability: Mourikis & Roumeliotis, ICRA 2007
- IMU Calibration: Furgale et al., JFR 2013
