# VIO+VPS+EKF System Documentation

## Overview

This is a Visual-Inertial Odometry (VIO) system using Multi-State Constraint Kalman Filter (MSCKF) 
for Bell 412 helicopter navigation without GPS.

## System Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Main Entry Point               │
                    │              vio_vps.py                     │
                    └─────────────────────────────────────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
          ▼                              ▼                              ▼
┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
│   Data Loaders  │            │   Core Filter   │            │  VIO Frontend   │
│ (lines 2288-2420)│            │ (lines 1489-1917)│           │ (lines 2926-3755)│
└─────────────────┘            └─────────────────┘            └─────────────────┘
        │                              │                              │
        ▼                              ▼                              ▼
  - load_imu_csv               - ExtendedKalmanFilter          - VIOFrontEnd class
  - load_mag_csv               - ensure_covariance_valid       - Feature tracking
  - load_vps_csv               - propagate_error_state         - ORB detection
  - DEMReader class                                            - Optical flow
  - PPKInitialState                                            
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│ IMU Preintegration│         │  MSCKF Backend  │          │  Magnetometer   │
│ (lines 883-1174) │         │ (lines 3757-5103)│         │ (lines 2445-2700)│
└─────────────────┘          └─────────────────┘          └─────────────────┘
        │                            │                            │
        ▼                            ▼                            ▼
  - IMUPreintegration          - triangulate_feature        - calibrate_mag
  - Forster et al. TRO         - compute_measurement_jacobian - compute_yaw_from_mag
  - Bias Jacobians             - msckf_measurement_update   - apply_mag_filter
```

## File Structure

### Current (Monolithic)
```
vio_vps_repo/
├── vio_vps.py           # Main script (9000+ lines)
├── scripts/
│   └── benchmark_no_vps.sh
├── configs/
│   └── config_bell412_dataset3.yaml
└── docs/
    └── ARCHITECTURE.md  # This file
```

### Proposed (Modular) - ✅ CREATED
```
vio_vps_repo/
├── vio_vps.py           # Minimal entry point
├── vio/                 # ✅ Package created
│   ├── __init__.py      # ✅ Lazy loading
│   ├── config.py        # ✅ Configuration loading
│   ├── math_utils.py    # ✅ Quaternion & rotation utilities
│   ├── imu_preintegration.py  # ✅ IMU preintegration
│   ├── ekf.py           # ✅ Extended Kalman Filter
│   ├── data_loaders.py  # ✅ IMU/MAG/VPS/DEM loaders
│   ├── magnetometer.py  # ✅ MAG calibration & yaw
│   ├── vio_frontend.py  # ⏳ (Visual feature tracking - TODO)
│   ├── msckf.py         # ⏳ (MSCKF backend - TODO)
│   └── propagation.py   # ⏳ (IMU propagation helpers - TODO)
├── scripts/
│   └── benchmark_no_vps.sh
└── configs/
    └── config_bell412_dataset3.yaml
```

## Code Organization by Function

### 1. Configuration (lines 1-325)
- `load_config()`: Load YAML config file
- Global constants: `KB_PARAMS`, `IMU_PARAMS`, `MAG_*`
- Debug flags: `VERBOSE_DEBUG`, `VERBOSE_DEM`

### 2. Math Utilities (lines 727-880)
- Quaternion operations: `quat_multiply`, `quat_normalize`, `quat_boxplus`
- Rotation conversion: `quat_to_rot`, `rot_to_quat`
- Matrix utilities: `skew_symmetric`

### 3. IMU Preintegration (lines 883-1174)
- `IMUPreintegration` class (Forster et al. TRO 2017)
- Bias Jacobians for fast correction
- `compute_error_state_jacobian()`
- `compute_error_state_process_noise()`

### 4. Extended Kalman Filter (lines 1327-1917)
- `ensure_covariance_valid()`: Fix numerical issues
- `propagate_error_state_covariance()`: ESKF propagation
- `ExtendedKalmanFilter` class: Full ESKF implementation

### 5. Data Loaders (lines 1918-2450)
- `DEMReader`: GeoTIFF DEM sampling
- `PPKInitialState`: PPK ground truth loading
- `load_imu_csv()`, `load_mag_csv()`, `load_vps_csv()`
- `load_images()`: Image sequence loading

### 6. Magnetometer (lines 2445-2700)
- `calibrate_magnetometer()`: Hard/soft iron correction
- `compute_yaw_from_mag()`: Yaw from magnetometer
- `apply_mag_filter()`: EMA + gyro consistency check

### 7. VIO Frontend (lines 2764-3755)
- `kannala_brandt_unproject()`: Fisheye undistortion
- `VIOFrontEnd` class: Feature tracking, ORB, optical flow

### 8. MSCKF Backend (lines 3757-5103)
- `triangulate_feature()`: Multi-view triangulation
- `compute_measurement_jacobian()`: MSCKF Jacobian
- `msckf_measurement_update()`: Feature update
- `perform_msckf_updates()`: Batch processing

### 9. Propagation (lines 5105-5450)
- `propagate_to_timestamp()`: IMU propagation with preintegration
- `_propagate_single_imu_step()`: Legacy sample-by-sample

### 10. Main Loop (lines 5929-8989)
- `run()`: Main VIO+EKF loop
- State initialization
- Measurement updates (VPS, DEM, MAG, VIO)

### 11. CLI (lines 9026-9154)
- Argument parsing
- Configuration loading
- Entry point

## Key State Variables

### EKF State Vector (nominal state)
| Index | Name | Size | Description |
|-------|------|------|-------------|
| 0:3   | p    | 3    | Position [m] |
| 3:6   | v    | 3    | Velocity [m/s] |
| 6:10  | q    | 4    | Quaternion [w,x,y,z] |
| 10:13 | bg   | 3    | Gyro bias [rad/s] |
| 13:16 | ba   | 3    | Accel bias [m/s²] |
| 16+7i | clone_i | 7 | Camera clones |

### Error State (covariance)
| Index | Name | Size | Description |
|-------|------|------|-------------|
| 0:3   | δp   | 3    | Position error |
| 3:6   | δv   | 3    | Velocity error |
| 6:9   | δθ   | 3    | Rotation error (3D!) |
| 9:12  | δbg  | 3    | Gyro bias error |
| 12:15 | δba  | 3    | Accel bias error |
| 15+6i | clone_i | 6 | Clone errors |

## Critical Implementation Notes

### 1. Quaternion Convention
- Format: [w, x, y, z] (scalar-first)
- Error state uses 3D rotation vector δθ
- Update: q_new = q ⊗ exp(δθ)

### 2. Coordinate Frames
- **ENU**: East-North-Up (world frame)
- **FRD**: Forward-Right-Down (body frame)
- IMU outputs in ENU, converted internally

### 3. Magnetometer Issues (Bell 412)
- **Rotor interference**: Severe mag noise during flight
- **Solution**: Phase-based K_MIN (0.40→0.20 after 15s)
- **Yaw correction limit**: MAX_YAW_CORRECTION = 30°/15°

### 4. IMU Preintegration
- Forster et al. TRO 2017
- Gravity compensation: a_hat = a_meas - ba + g_body
- Bias Jacobians for fast correction

## Debugging Tips

### Problem: Position Drift
1. Check `pose.csv` for velocity accumulation
2. Look at `error_log.csv` for error trends
3. Enable `--save_debug_data` for detailed logs

### Problem: Yaw Drift
1. Check MAG update count in output
2. Look for "YAW RESET" messages
3. Verify K_MIN values in MAG log

### Problem: Height Drift
1. Check DEM updates (if available)
2. Look at altitude in `error_log.csv`
3. Verify initial MSL from GGA file

## Running Benchmarks

```bash
# Full benchmark
./scripts/benchmark_no_vps.sh

# Quick test (first 30 seconds)
python vio_vps.py \
    --config configs/config_bell412_dataset3.yaml \
    --imu /path/to/imu.csv \
    --quarry /path/to/gga.csv \
    --images_dir /path/to/images \
    --ground_truth /path/to/.pos \
    --output test_output
```

## Version History

- **v9**: Best results (736m RMSE)
- **v11**: Sensor-quality features (regression to 1262m)
- **v12**: Reverted to v9 logic (818m RMSE)
