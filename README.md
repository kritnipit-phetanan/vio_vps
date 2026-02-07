# VIO/VPS - Visual-Inertial Odometry with Visual Place Recognition

A robust Visual-Inertial Odometry (VIO) system based on MSCKF (Multi-State Constraint Kalman Filter) designed for aerial platforms (Bell 412 helicopter, UAV) in GPS-denied environments.

**Version: 3.9.9** (EKF Mag Bias with Proper Freeze)

## Features

- **ESKF (Error-State Kalman Filter)** - OpenVINS-style quaternion manifold updates
  - 19D nominal state / 18D error state (with mag_bias)
  - Supports dynamic state freezing for disabled sensors
- **MSCKF** - Multi-State Constraint Kalman Filter for visual-inertial fusion
- **IMU Preintegration** - Forster et al. TRO 2017 on-manifold preintegration
- **Fisheye Camera Support** - Kannala-Brandt distortion model
- **Multi-Camera Backend** - Support for nadir, front, and side cameras
- **Magnetometer Integration** - Yaw constraint with hard/soft iron calibration
  - **NEW (v3.9.7+)**: Online hard iron estimation via EKF state augmentation
  - Configurable via `use_estimated_bias`, `sigma_mag_bias_init`, `sigma_mag_bias`
- **DEM Fusion** - Terrain-relative altitude constraints
- **Loop Closure** - Lightweight yaw drift correction

## Quick Start

### Prerequisites

```bash
# Python dependencies
pip install numpy scipy opencv-python pandas filterpy pyproj rasterio
```

### Running Benchmark

```bash
# Run modular benchmark
./scripts/benchmark_modular.sh
```

### Basic Usage

```bash
python run_vio.py \
    --config configs/config_bell412_dataset3.yaml \
    --imu /path/to/imu.csv \
    --quarry /path/to/flight_log_from_gga.csv \
    --output output/ \
    --mag /path/to/mag.csv \
    --dem /path/to/dem.tif \
    --save_debug_data
```

## Project Structure

```
vio_vps/
├── run_vio.py              # Main entry point (uses modular vio/ package)
│
├── vio/                    # Modular VIO package (v3.9.9)
│   ├── __init__.py         # Package info & version
│   ├── config.py           # VIOConfig dataclass & YAML loader
│   ├── ekf.py              # Error-State Kalman Filter (19D/18D)
│   ├── state_manager.py    # State initialization (18x18 covariance)
│   ├── propagation.py      # IMU propagation & ZUPT
│   ├── imu_preintegration.py  # Forster et al. preintegration
│   ├── measurement_updates.py # Mag, DEM, velocity updates
│   ├── msckf.py            # Multi-State Constraint KF
│   ├── main_loop.py        # VIORunner orchestrator
│   ├── magnetometer.py     # Mag calibration & yaw
│   ├── vps_integration.py  # VPS/GPS updates
│   └── ...                 # Other modules
│
├── configs/                # Configuration files
│   └── config_bell412_dataset3.yaml
│
├── scripts/                # Benchmark and test scripts
│   └── benchmark_modular.sh
│
└── docs/                   # Documentation
```

## Configuration

See `docs/CONFIG_README.md` for detailed configuration options.

Key parameters:
- Camera intrinsics (Kannala-Brandt model)
- IMU noise characteristics
- Magnetometer calibration (hard_iron, soft_iron, declination)
- **NEW**: Mag bias estimation (`use_estimated_bias`, `sigma_mag_bias_init`, `sigma_mag_bias`)
- MSCKF thresholds

### Magnetometer Bias Estimation (v3.9.7+)

```yaml
magnetometer:
  enabled: true
  use_estimated_bias: true      # Enable online hard iron estimation
  sigma_mag_bias_init: 0.01     # Initial uncertainty
  sigma_mag_bias: 0.0001        # Process noise (random walk)
```

## Algorithm Overview

1. **IMU Propagation** - Predict state at IMU rate (~400 Hz)
2. **Visual Front-end** - Grid-based KLT feature tracking
3. **MSCKF Update** - Triangulate features and apply multi-state constraints
4. **Magnetometer Update** - Yaw constraint with adaptive gating + online bias estimation
5. **DEM Constraint** - Terrain-relative altitude (optional)

## State Vector (v3.9.9)

| State | Dimension | Description |
|-------|-----------|-------------|
| Position | 3 | ENU coordinates (m) |
| Velocity | 3 | ENU velocity (m/s) |
| Quaternion | 4 | Body orientation (w,x,y,z) |
| Gyro Bias | 3 | Gyroscope bias (rad/s) |
| Accel Bias | 3 | Accelerometer bias (m/s²) |
| **Mag Bias** | **3** | **Hard iron offset (normalized)** |
| Camera Clones | 7×N | Clone poses |

**Total: 19D nominal + 7N clones** (18D error + 6N clones)

## TODO

- [ ] Optimize event-driven mode for real-time performance
- [ ] Develop mag bias dynamic adaptation (time-varying interference)

## Known Issues

- VIO performance degrades in low-parallax scenarios (hovering)
- Magnetometer requires careful calibration in magnetically disturbed environments
- See `docs/BELL412_ISSUES_AND_SOLUTIONS.md` for platform-specific issues

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Current version (19D/18D with mag_bias) |
| `15states` | Original version (16D/15D without mag_bias) |

## References

- [MSCKF 3.0](https://github.com/uzh-rpg/rpg_svo_pro_open) - ETH Zurich
- [OpenVINS](https://github.com/rpng/open_vins) - University of Delaware
- [Forster et al. TRO 2017](https://arxiv.org/abs/1512.02325) - On-Manifold Preintegration

## License

MIT License

## Authors

VIO Project Team
