# VIO/VPS - Visual-Inertial Odometry with Visual Place Recognition

A modular VIO + VPS system for GPS-denied navigation using IMU, monocular camera, and optional aiding sensors.

## Features

- ESKF (Error-State Kalman Filter)
  - 19D nominal / 18D error-state formulation for robust state propagation and update.
- MSCKF visual-inertial update
  - Multi-state constraints from tracked visual features to reduce drift.
- IMU preintegration
  - On-manifold IMU integration for high-rate propagation between camera updates.
- Magnetometer yaw aiding (optional)
  - Weak heading constraints with quality-aware gating/fail-soft logic.
- DEM / barometric altitude aiding (optional)
  - Vertical-state stabilization using terrain/altitude measurements.
- VPS absolute correction (optional)
  - Satellite-map-based global correction with safety guards and delayed update path.
- Adaptive/state-aware policy
  - Health-based scaling of process/measurement behavior to reduce divergence.

## Required and Optional Inputs

### Required (minimum to run)

- `--config` : YAML configuration file
- `--imu` : IMU CSV
- `--ground_truth` : GT/PPK file (used for initialization and evaluation in current workflow)
- `--output` : output directory

### Optional (sensor aids)

- `--images_dir` + `--images_index` : camera frames and index CSV (needed for VIO/MSCKF)
- `--timeref_csv` : camera/mag time reference mapping
- `--timeref_pps_csv` : PPS time-reference map for time-audit quality checks
- `--mag` : magnetometer CSV
- `--dem` : DEM raster for terrain height aiding
- `--quarry` : flight log / MSL source (if used in your config)
- `--vps_tiles` : MBTiles map database for VPS
- `--save_debug_data` : enable heavy debug CSV logs
- `--save_keyframe_images` : save debug keyframe images

## Quick Start

### 1) Install requirements

```bash
pip install -r requirements.txt
```

### 2) Run with `run_vio.py`

```bash
python3 run_vio.py \
  --config configs/config_bell412_dataset3.yaml \
  --imu /path/to/imu_with_ref.csv \
  --ground_truth /path/to/bell412_dataset3_frl.pos \
  --output benchmark_run/preintegration \
  --images_dir /path/to/images \
  --images_index /path/to/images_index.csv \
  --timeref_csv /path/to/timeref.csv \
  --mag /path/to/vector3.csv \
  --dem /path/to/dem.tif \
  --vps_tiles /path/to/mission.mbtiles
```

Minimal IMU-only style run:

```bash
python3 run_vio.py \
  --config configs/config_bell412_dataset3.yaml \
  --imu /path/to/imu_with_ref.csv \
  --ground_truth /path/to/bell412_dataset3_frl.pos \
  --output benchmark_run/preintegration
```

## Project Structure

```text
vio_vps/
├── run_vio.py                  # Main entry point
├── requirements.txt            # Python dependencies
├── configs/                    # YAML configurations
├── scripts/                    # Utilities, benchmark helpers, analysis scripts
├── tests/                      # Unit and integration-oriented tests
├── vio/                        # Core VIO package
│   ├── config.py               # YAML compiler/loader
│   ├── main_loop.py            # High-level runner/orchestration
│   ├── imu_driven.py           # IMU-driven runtime loop
│   ├── ekf.py                  # ESKF core
│   ├── msckf.py                # MSCKF logic
│   ├── measurement_updates.py  # Sensor update functions
│   ├── propagation.py          # IMU propagation
│   ├── backend_optimizer.py    # Async backend correction path
│   └── services/               # Runtime services (VIO/MAG/DEM/VPS/reporting)
└── vps/                        # VPS pipeline modules
```

## Configuration (Main Sections)

Primary YAML blocks you usually tune:

- `camera`, `extrinsics`
  - Intrinsics and frame transforms.
- `imu`, `process_noise`
  - IMU model, bias behavior, propagation noise.
- `vio`, `vio_vel`
  - Frontend/MSCKF thresholds and velocity-aid behavior.
- `magnetometer`
  - Heading aid quality/gating policy.
- `vps`
  - Absolute correction thresholds/fail-soft behavior.
- `adaptive`
  - State-aware and health-aware runtime policy.
- `backend`
  - Async fixed-lag optimizer options (if enabled).

## TODO

- Optimize event-driven mode for real-time performance.
- Improve policy management and policy separation per sensor/phase.
- Optimize state handling and service boundaries for maintainability.
- Continue one-knob parameter tuning with stronger acceptance criteria.
- Expand automated regression checks and long-horizon stability tests.

## Known Issues

- Accuracy can still degrade significantly in long-horizon GPS-denied runs.
- Runtime may exceed near-real-time targets depending on enabled sensors and debug I/O.
- Performance is sensitive to low-feature datasets / low-parallax segments.
- Sensor timing/frame-convention mismatch can still cause model mismatch drift if configs are wrong.

## References

- MSCKF 3.0 - ETH Zurich
- OpenVINS - University of Delaware
- Forster et al. TRO 2017 - On-Manifold Preintegration

## License

MIT License
