# VIO/MSCKF Configuration Files

## Overview

The VIO system now uses YAML configuration files to store dataset-specific calibration parameters. This allows easy switching between different platforms (DJI M600, Bell 412, etc.) without modifying code.

## Available Configurations

### 1. `config_dji_m600_quarry.yaml`
- **Platform:** DJI M600 drone
- **Dataset:** Original quarry dataset (Newfoundland)
- **Status:** ✅ Complete calibration (verified with ground truth)
- **Camera:** Kannala-Brandt fisheye (1440×1080)
- **Notes:** 
  - Camera-IMU extrinsics calibrated using Kalibr
  - Magnetometer calibration includes hard/soft iron correction
  - MSCKF parameters tuned for hover + slow motion

### 2. `config_bell412_dataset3.yaml`
- **Platform:** Bell 412 helicopter
- **Dataset:** bell412_dataset3 (forward flight)
- **Status:** ⚠️ Placeholder values - needs calibration
- **Camera:** Assumed similar fisheye model (needs verification)
- **Notes:**
  - Relaxed MSCKF parameters for forward motion
  - All calibration values marked with TODO comments
  - **REQUIRES CALIBRATION BEFORE PRODUCTION USE**

## Usage

### Running with specific config:

```bash
# DJI M600 dataset (default)
python vio_vps.py \
    --config config_dji_m600_quarry.yaml \
    --imu path/to/imu.csv \
    --quarry path/to/flight_log_from_gga.csv \
    --images_dir path/to/images \
    --images_index path/to/images_index.csv

# Bell 412 dataset (forward motion)
python vio_vps.py \
    --config config_bell412_dataset3.yaml \
    --imu path/to/bell412_dataset3/imu.csv \
    --quarry path/to/bell412_dataset3/flight_log_from_gga.csv \
    --images_dir path/to/bell412_dataset3/images \
    --images_index path/to/bell412_dataset3/images_index.csv
```

### Default behavior:
If `--config` is omitted, the system loads `config_dji_m600_quarry.yaml` by default.

## Configuration Structure

### Camera Parameters

```yaml
camera:
  intrinsics:  # Kannala-Brandt fisheye model
    k2, k3, k4, k5: Radial distortion coefficients
    mu, mv: Focal lengths (pixels)
    u0, v0: Principal point (pixels)
    w, h: Image resolution (pixels)
  
  extrinsics:  # Camera-IMU transformations (4×4 SE(3))
    BODY_T_CAMDOWN: Nadir (downward) camera pose
    BODY_T_CAMFRONT: Forward-facing camera pose
    BODY_T_CAMSIDE: Side-facing camera pose
  
  view_configs:  # Per-view VIO parameters
    nadir: Optimized for UAV downward cameras
    front: Optimized for forward motion (cars, helicopters)
    side: Optimized for oblique views
```

### IMU Parameters

```yaml
imu:
  noise:  # Allan variance or datasheet values
    acc_n: Accelerometer noise density (m/s²/√Hz)
    gyr_n: Gyroscope noise density (rad/s/√Hz)
    acc_w: Accelerometer random walk (m/s³/√Hz)
    gyr_w: Gyroscope random walk (rad/s²/√Hz)
    g_norm: Local gravity magnitude (m/s²)
  
  preintegration:  # Conservative values for IMU preintegration
    # Typically 5-10× smaller than legacy to prevent bias drift
```

### Magnetometer Calibration

```yaml
magnetometer:
  hard_iron_offset: [x, y, z]  # µT offset (3D vector)
  soft_iron_matrix: [[...], [...], [...]]  # 3×3 correction matrix
  declination: radians  # Magnetic declination (local)
  field_strength: µT  # Expected total field after calibration
  min_field_strength: µT  # Reject measurements below this
  max_field_strength: µT  # Reject measurements above this
  update_rate_limit: N  # Apply updates every N samples
```

### VIO/MSCKF Parameters

```yaml
vio:
  min_parallax_px: Minimum optical flow for VIO updates (pixels)
  min_msckf_baseline: Minimum baseline for triangulation (meters)
  msckf_chi2_multiplier: Chi-square threshold multiplier
  msckf_max_reprojection_error: Max reprojection error (pixels)
  vo_min_inliers: Minimum inliers for valid VO pose
  vo_ratio_test: Lowe's ratio test threshold
  vo_nadir_align_deg: Nadir alignment threshold (degrees)
  vo_front_align_deg: Forward alignment threshold (degrees)
```

### Process Noise

```yaml
process_noise:
  sigma_accel: Process acceleration magnitude (m/s²)
  sigma_vo_vel: VIO velocity measurement stdev (m/s)
  sigma_vps_xy: VPS position measurement stdev (m)
  sigma_agl_z: Height measurement stdev (m)
  sigma_mag_yaw: Magnetometer yaw measurement stdev (rad)
```

## Calibration Guide for New Datasets

### Step 1: Camera Intrinsics

**Tools:** Kalibr, OpenCV calibration scripts

**Procedure:**
1. Collect checkerboard/AprilTag calibration images (30-50 images)
2. Run Kalibr camera calibration:
   ```bash
   kalibr_calibrate_cameras \
       --target checkerboard.yaml \
       --bag calibration.bag \
       --models omni-radtan \
       --topics /camera/image_raw
   ```
3. Extract `k2, k3, k4, k5, mu, mv, u0, v0` from output YAML
4. Update `camera.intrinsics` section in config file

**Expected Accuracy:**
- Reprojection error < 0.5 pixels for good calibration
- Verify undistortion visually on straight lines

### Step 2: Camera-IMU Extrinsics

**Tools:** Kalibr IMU-camera calibration

**Procedure:**
1. Collect synchronized camera-IMU data with dynamic motion (60-120 seconds)
2. Run Kalibr IMU-camera calibration:
   ```bash
   kalibr_calibrate_imu_camera \
       --target checkerboard.yaml \
       --bag calibration.bag \
       --cam camera_calib.yaml \
       --imu imu_calib.yaml
   ```
3. Extract `BODY_T_CAMDOWN` (4×4 transformation matrix)
4. Update `camera.extrinsics` section

**Critical Checks:**
- Time synchronization: IMU and camera timestamps aligned (< 1ms error)
- Motion excitation: Must include rotation in all 3 axes + translation
- Reprojection error: < 1.0 pixels after extrinsic calibration

### Step 3: IMU Noise Parameters

**Tools:** Allan variance analysis or manufacturer datasheet

**Procedure:**
1. Collect 2-4 hours of stationary IMU data
2. Run Allan variance analysis:
   ```python
   from allan_variance import AllanVariance
   av = AllanVariance(imu_data)
   av.plot()  # Identify noise parameters from slopes
   ```
3. Extract:
   - `acc_n`: Accelerometer noise (flat region at τ=1s)
   - `gyr_n`: Gyroscope noise (flat region at τ=1s)
   - `acc_w`: Accelerometer random walk (slope=+0.5)
   - `gyr_w`: Gyroscope random walk (slope=+0.5)
4. Update `imu.noise` section

**Fallback:** Use manufacturer datasheet values (conservative)

### Step 4: Magnetometer Calibration

**Tools:** MotionCal, custom Python scripts

**Procedure:**
1. Collect magnetometer data while rotating platform in all directions (sphere coverage)
2. Run hard/soft iron calibration:
   ```python
   from mag_calibration import calibrate_magnetometer
   hard_iron, soft_iron = calibrate_magnetometer(mag_data)
   ```
3. Verify field strength after correction: should be ~50-60 µT (Earth's field)
4. Update `magnetometer` section
5. Lookup local magnetic declination: https://www.ngdc.noaa.gov/geomag/declination.shtml

**Quality Check:**
- Residual field variation after calibration < 5 µT
- Visualize 3D scatter plot: should form sphere centered at origin

### Step 5: Tuning VIO Parameters

**Initial Values (Conservative):**
- `min_parallax_px: 3-5` (higher for hover, lower for forward motion)
- `min_msckf_baseline: 0.10-0.20` (meters, based on typical motion speed)
- `msckf_chi2_multiplier: 5.0` (relaxed for initial testing)
- `msckf_max_reprojection_error: 4.0` (pixels)

**Tuning Strategy:**
1. Run VIO with debug output enabled (`--save_debug_data`)
2. Check `msckf_debug.csv`:
   - Triangulation success rate (target: > 50%)
   - Chi-square test rejection rate (target: < 30%)
   - Reprojection errors (target: < 2px for inliers)
3. Adjust parameters:
   - If too few triangulations: **REDUCE** `min_msckf_baseline`, `min_parallax_px`
   - If too many chi-square rejections: **INCREASE** `msckf_chi2_multiplier`
   - If large reprojection errors: **DECREASE** `msckf_max_reprojection_error`
4. Iterate until position RMSE converges

### Step 6: Process Noise Tuning

**Goal:** Balance between trusting IMU vs external measurements

**Procedure:**
1. Run VIO with default values
2. Check position/velocity covariance growth in `state_debug.csv`
3. Adjust:
   - `sigma_vo_vel`: Lower if VIO is accurate (good features), higher if noisy
   - `sigma_vps_xy`: Lower if GPS/VPS is accurate (< 1m), higher for noisy signals
   - `sigma_agl_z`: Tune based on DEM accuracy (typically 1-5m)
4. Monitor innovation (measurement - prediction):
   - Consistent bias → adjust measurement noise
   - Random scatter → noise parameters OK

## Motion-Specific Tuning

### Hover/Stationary (UAV)
- **Challenge:** Low parallax, insufficient motion for triangulation
- **Config:**
  - `min_parallax_px: 1-2` (very permissive)
  - `min_msckf_baseline: 0.05` (accept small baselines)
  - `sigma_vo_vel: 2.0` (higher noise due to scale uncertainty)
  - Enable ZUPT (Zero Velocity Updates) if available

### Forward Motion (Helicopter, Car)
- **Challenge:** Dominant forward motion, limited lateral excitation
- **Config:**
  - `min_parallax_px: 2-4` (moderate)
  - `min_msckf_baseline: 0.10-0.15` (meters)
  - `sigma_vo_vel: 1.0-1.5` (lower noise, good scale observability)
  - `msckf_chi2_multiplier: 5.0` (relaxed for fast motion)

### Aggressive Maneuvers (Acrobatic Flight)
- **Challenge:** High dynamics, large IMU biases, motion blur
- **Config:**
  - `min_parallax_px: 5-8` (strict, avoid motion blur)
  - `acc_w: 0.001` (increase bias uncertainty)
  - `sigma_vo_vel: 3.0` (expect noisy measurements)
  - Consider disabling VIO during rapid rotation (> 180°/s)

## Troubleshooting

### Symptom: VIO velocity never updates (parallax = 0px)
**Causes:**
- Camera-IMU extrinsics incorrect (wrong frame convention)
- Feature tracking failure (poor lighting, texture)
- Image resolution mismatch (config vs actual)

**Fix:**
1. Verify `BODY_T_CAM*` transformations (visualize in 3D)
2. Check feature debug output: `vo_debug.csv` (avg_parallax_px column)
3. Visualize feature tracks on images (`--save_keyframe_images`)

### Symptom: MSCKF triangulation success < 10%
**Causes:**
- `min_msckf_baseline` too large for motion speed
- `min_parallax_px` too high (rejects valid features)
- Poor feature tracks (noisy images, rolling shutter)

**Fix:**
1. **REDUCE** `min_msckf_baseline` (try 0.10 → 0.05)
2. **REDUCE** `min_parallax_px` (try 5 → 2)
3. Check `msckf_debug.csv`: `baseline_m`, `parallax_px`, `chi2_test` columns

### Symptom: Position drift despite VIO updates
**Causes:**
- Scale observability poor (nadir camera, insufficient parallax)
- IMU noise parameters too conservative (IMU dominates)
- VIO measurement noise (`sigma_vo_vel`) too high (filter ignores vision)

**Fix:**
1. **REDUCE** `sigma_vo_vel` (e.g., 2.5 → 1.5 m/s) to trust vision more
2. Verify homography scale recovery in `vo_debug.csv` (scale_factor column)
3. Check if using correct camera view mode (`--camera_view front` for helicopters)

### Symptom: Magnetometer updates rejected (99%+)
**Causes:**
- Uncalibrated magnetometer (hard/soft iron)
- Magnetic interference (motors, metal frame)
- Incorrect field strength threshold

**Fix:**
1. Run magnetometer calibration procedure (Step 4 above)
2. Increase `MAG_MIN_FIELD_STRENGTH` to accept weaker signals
3. Disable magnetometer if uncalibrable: `--use_magnetometer False`

## Performance Benchmarks

### DJI M600 Quarry Dataset
- **Duration:** 308 seconds
- **Motion:** Hover + slow translation (< 2 m/s)
- **Position RMSE:** 27m (with bug fixes)
- **Configuration:** `config_dji_m600_quarry.yaml`
- **VIO Updates:** ~180/4625 frames (3.9%)
- **MSCKF Triangulations:** 50-60% success rate

### Bell 412 Dataset3 (Needs Calibration)
- **Duration:** 308 seconds
- **Motion:** Forward flight (5-10 m/s)
- **Position RMSE:** TBD (placeholder calibration)
- **Configuration:** `config_bell412_dataset3.yaml` ⚠️
- **Notes:** Requires Steps 1-5 above before production use

## References

1. **IMU Preintegration:** Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry", TRO 2017
2. **MSCKF:** Mourikis & Roumeliotis, "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation", ICRA 2007
3. **OpenVINS:** Geneva et al., "OpenVINS: A Research Platform for Visual-Inertial Estimation", IROS 2020
4. **Kalibr Calibration:** Furgale et al., "Unified Temporal and Spatial Calibration for Multi-Sensor Systems", IROS 2013
5. **Allan Variance:** IEEE Standard 952-1997, "Allan Variance Method for IMU Characterization"

## TODO

- [ ] Complete Bell 412 calibration (all steps)
- [ ] Add config validator (check parameter ranges)
- [ ] Add config auto-tuning from calibration data
- [ ] Support multiple camera configs (stereo, multi-cam)
- [ ] Add rolling shutter compensation parameters
