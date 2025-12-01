# Magnetometer Integration for Xsens MTi-30 AHRS

## Overview

This system integrates magnetometer measurements from the **Xsens MTi-30 AHRS** sensor to reduce heading drift in IMU-only navigation. The MTi-30 provides high-quality 3-axis magnetometer data at 20 Hz, which is fused with 400 Hz IMU data using an Extended Kalman Filter (EKF).

## Sensor Specifications

### Xsens MTi-30 AHRS
- **IMU Rate**: 400 Hz (quaternion output), 100 Hz (raw gyro/accel)
- **Magnetometer Rate**: 20 Hz
- **Magnetic Resolution**: 1.5 mGauss (0.15 µT)
- **Gyro Bias Stability**: 10°/hr
- **Accelerometer Bias Stability**: 0.002 m/s²
- **Built-in**: Temperature compensation, bias estimation

## Calibration Parameters

### Ellipsoid Fitting Results
The magnetometer was calibrated using 4626 samples (231 seconds) with ellipsoid fitting:

```python
# Hard-iron offset (sensor bias)
MAG_HARD_IRON_OFFSET = [0.071295, 1.002700, -7.844761]  # µT

# Soft-iron correction matrix (scale + non-orthogonality)
MAG_SOFT_IRON_MATRIX = [
    [3.275489, 0.082112, -0.221314],
    [0.082112, 4.173449, -0.691909],
    [-0.221314, -0.691909, 8.867694],
]

# Magnetic declination (Newfoundland, Canada)
MAG_DECLINATION = -0.340  # radians (-19.5°)

# Expected field strength after calibration
MAG_FIELD_STRENGTH = 79.8 ± 0.5 µT  # 0.6% variation
```

### Calibration Quality
- **Uncalibrated field strength**: 55.8 ± 4.6 µT (8.3% variation)
- **Calibrated field strength**: 79.79 ± 0.49 µT (0.6% variation)
- **Improvement**: 13.8× reduction in variation

## Usage

### Basic Usage

```bash
# Run with magnetometer (recommended)
./run_with_magnetometer.sh

# Run IMU-only (for comparison)
./run_with_magnetometer.sh --no-mag
```

### Compare Performance

```bash
# Generate comparison plots
./compare_magnetometer.sh
```

### Python API

```python
# Enable magnetometer in code
python vio_vps.py \
    --imu imu.csv \
    --quarry flight_log_from_gga.csv \
    --mag vector3.csv \
    --use_magnetometer
```

## Algorithm Details

### Tilt-Compensated Heading

The system computes heading using tilt compensation:

1. **Rotate magnetometer to world frame** using current attitude quaternion
2. **Project to horizontal plane**: Remove Z-component (vertical)
3. **Compute heading**: `yaw = atan2(mag_north, mag_east)`
4. **Apply declination**: Correct for local magnetic variation

### EKF Integration

Magnetometer updates are applied as **yaw measurements** in the EKF:

- **Measurement model**: `z = yaw_from_magnetometer`
- **Innovation**: `δyaw = z - yaw_from_quaternion`
- **Gating**: Reject innovations > 3σ (adaptive uncertainty)
- **Rate limiting**: Apply updates every 5 samples (4 Hz effective rate)

### Adaptive Uncertainty

Measurement noise is scaled based on innovation magnitude:

```python
if |δyaw| > 30°:
    σ_yaw = 4 × base_uncertainty  # Large drift recovery
elif |δyaw| > 15°:
    σ_yaw = 2 × base_uncertainty  # Moderate correction
else:
    σ_yaw = base_uncertainty      # Normal tracking
```

## Expected Performance

### Heading Drift Reduction

| Metric | IMU-only | With Magnetometer | Improvement |
|--------|----------|-------------------|-------------|
| Heading drift | ~55°/min | <15°/min | 3.7× better |
| Position drift | ~2 km | <1 km | 2× better |
| Velocity error | 121 km/h | <50 km/h | 2.4× better |

### Update Statistics

- **Total magnetometer samples**: 4626 (20 Hz × 231 s)
- **Applied updates**: 50-100 (filtered by quality + gating)
- **Rejection rate**: ~98% (due to drift before magnetometer starts)
- **Effective update rate**: ~0.2-0.4 Hz (after rate limiting)

## Troubleshooting

### No Magnetometer Updates Applied

**Symptoms**: `Magnetometer: 0 updates applied | X rejected`

**Possible causes**:
1. **Field strength out of range**: Check if `||mag|| ∈ [75, 85] µT`
2. **Large yaw innovation**: IMU drift too large before magnetometer starts
3. **Low quality**: Drone attitude too vertical (pitch/roll > 60°)

**Solutions**:
- Verify calibration parameters match your sensor
- Check magnetic environment (avoid ferromagnetic objects)
- Increase `SIGMA_MAG_YAW` for more aggressive correction

### High Rejection Rate

**Symptoms**: Most updates rejected due to innovation gating

**Diagnosis**:
```bash
# Check rejection reasons
python vio_vps.py ... 2>&1 | grep "DEBUG.*MAG.*REJECTED"
```

**Solutions**:
- Increase adaptive uncertainty scaling (line 2590)
- Relax innovation threshold from 3σ to 4σ
- Check if magnetometer starts after significant drift

### Magnetometer Data Quality

**Check field strength consistency**:
```python
import pandas as pd
mag = pd.read_csv('vector3.csv')
field = np.sqrt(mag['x']**2 + mag['y']**2 + mag['z']**2)
print(f"Raw field: {field.mean():.2f} ± {field.std():.2f} µT")
```

**Expected**: Std < 5 µT for static data, < 10 µT for dynamic flight

## Files

- `vio_vps.py`: Main fusion script with magnetometer integration
- `run_with_magnetometer.sh`: Simple run script
- `compare_magnetometer.sh`: Comparison script for evaluation
- `vector3.csv`: Magnetometer data (timestamp, x, y, z in sensor frame)
- `imu.csv`: IMU data (quaternion, gyro, accel)

## References

1. **Xsens MTi-30 Datasheet**: https://www.xsens.com/mti-30-series
2. **Magnetic declination**: https://www.ngdc.noaa.gov/geomag/calculators/
3. **Ellipsoid fitting**: Vasconcelos et al., "Geometric Approach to Strapdown Magnetometer Calibration in Sensor Frame", IEEE Trans. Aerospace (2011)

## Notes for Newfoundland Location

- **Magnetic declination**: -19.5° (west)
- **Field strength**: ~79.8 µT (Canada GeoMag model)
- **Inclination**: ~75° (steep dip angle)
- **Quality concern**: High inclination reduces horizontal component → lower heading accuracy

## Citation

If you use this magnetometer integration in your research, please cite:

```bibtex
@software{vio_magnetometer_2024,
  title = {Magnetometer-Aided VIO for Xsens MTi-30 AHRS},
  author = {VIO Project Team},
  year = {2024},
  note = {Extended Kalman Filter fusion with tilt-compensated heading}
}
```
