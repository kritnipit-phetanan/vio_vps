# VIO/VPS - Visual-Inertial Odometry with Visual Place Recognition

A robust Visual-Inertial Odometry (VIO) system based on MSCKF (Multi-State Constraint Kalman Filter) designed for aerial platforms (Bell 412 helicopter, UAV) in GPS-denied environments.

## Features

- **ESKF (Error-State Kalman Filter)** - OpenVINS-style quaternion manifold updates
- **MSCKF** - Multi-State Constraint Kalman Filter for visual-inertial fusion
- **IMU Preintegration** - Forster et al. TRO 2017 on-manifold preintegration
- **Fisheye Camera Support** - Kannala-Brandt distortion model
- **Multi-Camera Backend** - Support for nadir, front, and side cameras
- **Magnetometer Integration** - Yaw constraint with hard/soft iron calibration
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
# Activate conda environment
conda activate 3D_Building_DepthAnyThingV2

# Run benchmark without VPS
cd scripts
./benchmark_no_vps.sh
```

### Basic Usage

```bash
python vio_vps.py \
    --imu /path/to/imu.csv \
    --mag /path/to/mag.csv \
    --video /path/to/video.mp4 \
    --config configs/config_bell412_dataset3.yaml \
    --output output/ \
    --no-vps
```

## Project Structure

```
vio_vps_repo/
├── vio_vps.py              # Main VIO/VPS implementation (~9000 lines)
├── vio_kh.py               # Simplified VIO implementation
├── vio_for_kh.py           # Alternative VIO variant
├── MULTI_CAMERA_BACKEND.py # Multi-camera support
│
├── configs/                # Configuration files
│   ├── config_bell412_dataset3.yaml
│   ├── config_dji_m600_quarry.yaml
│   └── config_dji_m600_quarry_relaxed.yaml
│
├── scripts/                # Benchmark and test scripts
│   ├── benchmark_no_vps.sh
│   ├── benchmark_bell412_multicam.sh
│   └── benchmark_preintegration.sh
│
├── docs/                   # Documentation
│   ├── MSCKF_IMPLEMENTATION.md
│   ├── MAGNETOMETER_README.md
│   ├── CONFIG_README.md
│   └── ...
│
├── evaluate_*.py           # Evaluation tools
├── analyze_*.py            # Analysis scripts
├── plot_trajectory_*.py    # Visualization
├── mag_calibration.py      # Magnetometer calibration
└── ground_truth_loader.py  # Ground truth handling
```

## Datasets

Tested on:
- **MUN-FRL Bell 412 Dataset** - Helicopter flights with fisheye camera
- **DJI M600 Quarry Dataset** - UAV flights with nadir camera

## Configuration

See `docs/CONFIG_README.md` for detailed configuration options.

Key parameters:
- Camera intrinsics (Kannala-Brandt model)
- IMU noise characteristics
- Magnetometer calibration
- MSCKF thresholds

## Algorithm Overview

1. **IMU Propagation** - Predict state at IMU rate (~400 Hz)
2. **Visual Front-end** - Grid-based KLT feature tracking
3. **MSCKF Update** - Triangulate features and apply multi-state constraints
4. **Magnetometer Update** - Yaw constraint with adaptive gating
5. **DEM Constraint** - Terrain-relative altitude (optional)

## Known Issues

- VIO performance degrades in low-parallax scenarios (hovering)
- Magnetometer requires careful calibration in magnetically disturbed environments
- See `docs/BELL412_ISSUES_AND_SOLUTIONS.md` for platform-specific issues

## References

- [MSCKF 3.0](https://github.com/uzh-rpg/rpg_svo_pro_open) - ETH Zurich
- [OpenVINS](https://github.com/rpng/open_vins) - University of Delaware
- [Forster et al. TRO 2017](https://arxiv.org/abs/1512.02325) - On-Manifold Preintegration

## License

MIT License

## Authors

VIO Project Team
