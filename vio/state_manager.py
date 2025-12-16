"""
State Manager for VIO+EKF System

This module handles:
- EKF state initialization (position, velocity, quaternion, biases)
- Lever arm compensation for IMU-GNSS offset
- Quaternion initialization from PPK or IMU
- Magnetometer-based initial yaw correction
- Covariance initialization

State Layout (16 elements):
    idx 0..2   : p_I (position) [m]
    idx 3..5   : v_I (velocity) [m/s]
    idx 6..9   : q_I (quaternion w,x,y,z)
    idx 10..12 : b_g (gyro bias) [rad/s]
    idx 13..15 : b_a (accel bias) [m/s^2]

Error State (15 + 6*M for M camera clones):
    idx 0..2   : δp (position error)
    idx 3..5   : δv (velocity error)
    idx 6..8   : δθ (rotation error - 3D!)
    idx 9..11  : δbg (gyro bias error)
    idx 12..14 : δba (accel bias error)
    idx 15+6i  : camera clone i errors

Author: VIO project
"""

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class InitialState:
    """Container for initial VIO state."""
    position: np.ndarray      # [x, y, z] in local ENU
    velocity: np.ndarray      # [vx, vy, vz] in ENU
    quaternion: np.ndarray    # [w, x, y, z]
    gyro_bias: np.ndarray     # [bx, by, bz] rad/s
    accel_bias: np.ndarray    # [bx, by, bz] m/s^2
    origin_lat: float         # Origin latitude
    origin_lon: float         # Origin longitude
    msl0: float               # Initial MSL altitude
    dem0: Optional[float]     # DEM elevation at origin (if available)
    agl0: Optional[float]     # Initial AGL (if DEM available)


def compute_lever_arm_world(R_body_to_world: np.ndarray, 
                            lever_arm_body: np.ndarray) -> np.ndarray:
    """
    Transform IMU-GNSS lever arm from body frame to world frame.
    
    Args:
        R_body_to_world: 3x3 rotation matrix (body to world)
        lever_arm_body: Lever arm in body frame [x, y, z]
        
    Returns:
        Lever arm in world frame (ENU)
    """
    return R_body_to_world @ lever_arm_body


def build_rotation_from_ppk(roll_ned: float, pitch_ned: float, yaw_ned: float) -> np.ndarray:
    """
    Build rotation matrix from PPK Euler angles (NED convention).
    
    Converts NED angles to ENU and handles FLU to FRD body frame conversion.
    
    Args:
        roll_ned: Roll angle in radians (NED)
        pitch_ned: Pitch angle in radians (NED)
        yaw_ned: Yaw angle in radians (NED: 0=North, CW positive)
        
    Returns:
        R_BW_frd: 3x3 rotation matrix (FRD body to ENU world)
    """
    # NED to ENU conversion for Euler angles
    roll_enu = roll_ned
    pitch_enu = -pitch_ned
    yaw_enu = np.pi/2 - yaw_ned
    
    # Build quaternion from ENU Euler angles (ZYX convention)
    R_BW_flu = R_scipy.from_euler('ZYX', [yaw_enu, pitch_enu, roll_enu])
    
    # Convert from FLU to FRD body frame
    R_FLU_to_FRD = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    R_BW_frd = R_BW_flu.as_matrix() @ R_FLU_to_FRD.T
    
    return R_BW_frd


def quaternion_from_rotation(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [w, x, y, z].
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion as [w, x, y, z]
    """
    q_xyzw = R_scipy.from_matrix(R).as_quat()  # scipy returns [x, y, z, w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def initialize_quaternion_from_ppk(ppk_state: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize quaternion from PPK ground truth attitude.
    
    Args:
        ppk_state: PPKInitialState object with roll, pitch, yaw
        
    Returns:
        Tuple of (quaternion [w,x,y,z], R_body_to_world 3x3)
    """
    R_BW = build_rotation_from_ppk(ppk_state.roll, ppk_state.pitch, ppk_state.yaw)
    q_init = quaternion_from_rotation(R_BW)
    
    # Verify body Z direction (should be negative for FRD body)
    body_z_world = R_BW[:, 2]
    print(f"[PPK→FRD] Body Z in world: {body_z_world}, Z-component: {body_z_world[2]:.4f}")
    
    return q_init, R_BW


def initialize_quaternion_from_imu(imu_quat_xyzw: np.ndarray,
                                   ground_truth_yaw_ned: Optional[float] = None) -> np.ndarray:
    """
    Initialize quaternion from IMU measurement, optionally correcting yaw.
    
    Args:
        imu_quat_xyzw: IMU quaternion [x, y, z, w] (Body-to-ENU)
        ground_truth_yaw_ned: Optional ground truth yaw in radians (NED: 0=North)
        
    Returns:
        Quaternion as [w, x, y, z]
    """
    # Convert from [x,y,z,w] to [w,x,y,z]
    q_init = np.array([imu_quat_xyzw[3], imu_quat_xyzw[0], 
                       imu_quat_xyzw[1], imu_quat_xyzw[2]])
    
    if ground_truth_yaw_ned is not None:
        # Convert NED yaw to ENU: yaw_enu = 90° - yaw_ned
        gt_yaw_enu = np.pi/2 - ground_truth_yaw_ned
        
        # Get current yaw from quaternion
        q_xyzw = np.array([q_init[1], q_init[2], q_init[3], q_init[0]])
        euler_imu = R_scipy.from_quat(q_xyzw).as_euler('ZYX')
        yaw_imu = euler_imu[0]
        
        print(f"[INIT][GT] IMU initial yaw: {np.degrees(yaw_imu):.1f}° (ENU)")
        print(f"[INIT][GT] Ground truth yaw: {np.degrees(ground_truth_yaw_ned):.1f}° (NED) = "
              f"{np.degrees(gt_yaw_enu):.1f}° (ENU)")
        
        # Replace yaw while keeping roll and pitch
        euler_corrected = euler_imu.copy()
        euler_corrected[0] = gt_yaw_enu
        
        # Reconstruct quaternion
        q_new_xyzw = R_scipy.from_euler('ZYX', euler_corrected).as_quat()
        q_init = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]])
        
    return q_init


def estimate_imu_bias_from_static(imu_records: list, 
                                  imu_params: dict,
                                  n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate gyro and accelerometer biases from initial static period.
    
    Uses world-frame gravity compensation for accurate bias estimation.
    
    Args:
        imu_records: List of IMU records with .lin, .ang, .q attributes
        imu_params: Dict with 'g_norm' key for gravity magnitude
        n_samples: Number of samples to use for estimation
        
    Returns:
        Tuple of (gyro_bias [3], accel_bias [3])
    """
    n_static = min(n_samples, len(imu_records))
    
    acc_body_list = []
    gyro_list = []
    
    for rec in imu_records[:n_static]:
        acc_body_list.append(rec.lin.astype(float))
        gyro_list.append(rec.ang.astype(float))
    
    acc_body_mean = np.mean(acc_body_list, axis=0)
    gyro_mean = np.mean(gyro_list, axis=0)
    
    # For stationary sensor: R @ (a_measured - ba) = [0, 0, -g] in ENU
    # Therefore: ba = a_measured - R^T @ [0, 0, -g]
    quat_0 = np.array([imu_records[0].q[0], imu_records[0].q[1], 
                       imu_records[0].q[2], imu_records[0].q[3]])
    R_0 = R_scipy.from_quat(quat_0).as_matrix()
    
    # In ENU frame: gravity points DOWN, so g_world = [0, 0, -g]
    expected_world = np.array([0, 0, -imu_params["g_norm"]])
    expected_body = R_0.T @ expected_world
    
    ba_init = acc_body_mean - expected_body
    bg_init = gyro_mean  # Gyro should read zero when stationary
    
    print(f"[DEBUG][BIAS] Using {n_static} samples for bias estimation")
    print(f"[DEBUG][BIAS] Raw acc body mean: {acc_body_mean}")
    print(f"[DEBUG][BIAS] Expected body (from R^T @ [0,0,-g]): {expected_body}")
    print(f"[DEBUG][BIAS] Estimated ba: {ba_init}")
    print(f"[DEBUG][BIAS] Estimated bg: {bg_init}")
    
    return bg_init, ba_init


def apply_mag_yaw_correction(q_current: np.ndarray,
                             mag_calibrated: np.ndarray,
                             mag_declination: float,
                             use_raw_heading: bool = True) -> Tuple[np.ndarray, bool]:
    """
    Apply magnetometer-based yaw correction to quaternion.
    
    Args:
        q_current: Current quaternion [w, x, y, z]
        mag_calibrated: Calibrated magnetometer vector
        mag_declination: Magnetic declination in radians
        use_raw_heading: Use GPS-calibrated raw heading
        
    Returns:
        Tuple of (corrected quaternion [w,x,y,z], correction_applied bool)
    """
    from .magnetometer import compute_yaw_from_mag, calibrate_magnetometer
    
    # Compute yaw from magnetometer
    yaw_mag, quality = compute_yaw_from_mag(
        mag_calibrated, q_current,
        mag_declination=mag_declination,
        use_raw_heading=use_raw_heading
    )
    
    if quality < 0.5:
        print(f"[INIT][MAG] Low quality ({quality:.2f}), no correction applied")
        return q_current, False
    
    # Get current yaw from quaternion
    q_xyzw = np.array([q_current[1], q_current[2], q_current[3], q_current[0]])
    R_mat = R_scipy.from_quat(q_xyzw).as_matrix()
    euler = R_scipy.from_matrix(R_mat).as_euler('ZYX')
    yaw_imu = euler[0]
    
    # Compute yaw difference
    yaw_diff = yaw_mag - yaw_imu
    yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
    
    # Only correct if difference is significant (>10°)
    if abs(yaw_diff) <= np.radians(10):
        print(f"[INIT][MAG] Yaw difference small ({np.degrees(yaw_diff):.1f}°), no correction needed")
        return q_current, False
    
    print(f"[INIT][MAG] Correcting initial yaw: IMU={np.degrees(yaw_imu):.1f}° → "
          f"MAG={np.degrees(yaw_mag):.1f}° (Δ={np.degrees(yaw_diff):.1f}°)")
    
    # Replace yaw with magnetometer yaw
    euler[0] = yaw_mag
    
    # Reconstruct quaternion
    R_new = R_scipy.from_euler('ZYX', euler).as_matrix()
    q_new_xyzw = R_scipy.from_matrix(R_new).as_quat()
    q_new = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]])
    
    return q_new, True


def initialize_covariance(imu_params: dict,
                          estimate_imu_bias: bool = False,
                          has_static_calibration: bool = False) -> np.ndarray:
    """
    Initialize ESKF covariance matrix (15x15 for core state).
    
    Args:
        imu_params: Dict with IMU noise parameters
        estimate_imu_bias: Whether bias estimation is enabled
        has_static_calibration: Whether bias was estimated from static period
        
    Returns:
        15x15 diagonal covariance matrix
    """
    P_pos = 10.0     # Position uncertainty (m²)
    P_vel = 4.0      # Velocity uncertainty (m/s)²
    P_theta = 0.1    # Rotation uncertainty (rad²)
    
    # Bias uncertainty depends on calibration state
    # NOTE: Even with static calibration, keep bias uncertainty high enough
    # to allow MSCKF to refine bias estimates during flight (for observability)
    if estimate_imu_bias and has_static_calibration:
        # Increased from 10x to 100x to allow MSCKF bias updates
        P_bg = (imu_params["gyr_w"] * 100) ** 2  # ~0.01 rad/s std
        P_ba = (imu_params["acc_w"] * 100) ** 2  # ~0.1 m/s² std
    else:
        P_bg = (imu_params["gyr_w"] * 1000) ** 2
        P_ba = (imu_params["acc_w"] * 1000) ** 2
    
    P = np.diag([
        P_pos, P_pos, P_pos,        # δp (3)
        P_vel, P_vel, P_vel,        # δv (3)
        P_theta, P_theta, P_theta,  # δθ (3)
        P_bg, P_bg, P_bg,           # δbg (3)
        P_ba, P_ba, P_ba,           # δba (3)
    ]).astype(float)
    
    print(f"\n=== Initial Covariance (ESKF error-state 15×15) ===")
    print(f"[COV] Position error (δp): {P_pos:.2e} m²")
    print(f"[COV] Velocity error (δv): {P_vel:.2e} (m/s)²")
    print(f"[COV] Rotation error (δθ): {P_theta:.2e}")
    print(f"[COV] Gyro bias error: {P_bg:.2e} (rad/s)²")
    print(f"[COV] Accel bias error: {P_ba:.2e} (m/s²)²")
    
    return P


def initialize_ekf_state(kf,
                         ppk_state: Optional[Any],
                         imu_records: list,
                         imu_params: dict,
                         lever_arm: np.ndarray,
                         lat0: float, lon0: float,
                         msl0: float, dem0: Optional[float],
                         v_init_enu: np.ndarray,
                         z_state: str = "msl",
                         estimate_imu_bias: bool = False,
                         initial_gyro_bias: Optional[np.ndarray] = None,
                         initial_accel_bias: Optional[np.ndarray] = None,
                         mag_records: Optional[list] = None,
                         mag_params: Optional[dict] = None,
                         ppk_initial_heading: Optional[float] = None) -> InitialState:
    """
    Complete EKF state initialization.
    
    Handles:
    - Position with lever arm compensation
    - Velocity from PPK or GGA
    - Quaternion from PPK, IMU, or MAG-corrected
    - Bias estimation or initialization
    - Covariance setup
    
    Args:
        kf: ExtendedKalmanFilter instance
        ppk_state: PPKInitialState or None
        imu_records: List of IMU records
        imu_params: IMU parameters dict
        lever_arm: IMU-GNSS lever arm [x, y, z] in body frame
        lat0, lon0: Origin coordinates
        msl0: Initial MSL altitude
        dem0: DEM elevation at origin (None if unavailable)
        v_init_enu: Initial velocity [vx, vy, vz]
        z_state: "msl" or "agl"
        estimate_imu_bias: Enable bias estimation
        initial_gyro_bias: Config-provided gyro bias (optional)
        initial_accel_bias: Config-provided accel bias (optional)
        mag_records: Magnetometer records for initial correction
        mag_params: Magnetometer parameters dict
        
    Returns:
        InitialState object with all initialization data
    """
    # Initialize state timestamp
    kf.last_absolute_correction_time = imu_records[0].t
    
    # Compute lever arm offset in world frame
    R_BW_init = None
    lever_arm_world = np.zeros(3)
    
    if ppk_state is not None:
        R_BW_init = build_rotation_from_ppk(ppk_state.roll, ppk_state.pitch, ppk_state.yaw)
        lever_arm_world = compute_lever_arm_world(R_BW_init, lever_arm)
        
        print(f"[INIT][LEVER ARM] IMU-GNSS lever arm in body frame: {lever_arm}")
        print(f"[INIT][LEVER ARM] IMU-GNSS lever arm in world frame (ENU): {lever_arm_world}")
        print(f"[INIT][LEVER ARM] Lever arm magnitude: {np.linalg.norm(lever_arm):.3f} m")
    
    # Initialize position with lever arm compensation
    if ppk_state is not None and np.linalg.norm(lever_arm_world) > 0.01:
        kf.x[0, 0] = -lever_arm_world[0]  # X (East)
        kf.x[1, 0] = -lever_arm_world[1]  # Y (North)
        print(f"[INIT] Initial XY position (IMU, lever-arm corrected): "
              f"[{kf.x[0,0]:.3f}, {kf.x[1,0]:.3f}] m")
    else:
        kf.x[0, 0] = 0.0
        kf.x[1, 0] = 0.0
    
    # Initialize Z position
    z_lever_offset = lever_arm_world[2] if ppk_state is not None else 0.0
    agl0 = (msl0 - dem0) if dem0 is not None else msl0
    
    if z_state.lower() == "agl" and dem0 is not None:
        kf.x[2, 0] = agl0 - z_lever_offset
        z_mode = "AGL"
    else:
        kf.x[2, 0] = msl0 - z_lever_offset
        z_mode = "MSL"
    
    print(f"[INIT] Initial Z ({z_mode}, lever-arm corrected): {kf.x[2,0]:.2f} m")
    
    # Initialize velocity
    kf.x[3:6, 0] = v_init_enu
    
    # Initialize quaternion
    # v2.9.10.2 FIX: PPK heading priority order (all are single t=0 values, GPS-denied compliant)
    #   1. PPK velocity heading (if moving at t=0) - computed from 2 samples
    #   2. PPK attitude yaw (if stationary at t=0) - single t=0 value, SKIP MAG CORRECTION!
    #   3. IMU quaternion (if no PPK available)
    
    use_ppk_attitude = False  # Flag to skip magnetometer correction
    
    if ppk_initial_heading is not None:
        # Use PPK heading from velocity (moving at t=0)
        from scipy.spatial.transform import Rotation as R_scipy
        R_init = R_scipy.from_euler('ZYX', [ppk_initial_heading, 0.0, 0.0])
        q_xyzw = R_init.as_quat()  # [x, y, z, w]
        q_init = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # [w, x, y, z]
        print(f"[INIT][PPK VELOCITY] Using PPK initial heading from t=0 velocity: {np.degrees(ppk_initial_heading):.1f}° (ENU)")
        use_ppk_attitude = True  # Skip magnetometer correction
    elif ppk_state is not None:
        # v2.9.10.2: Use PPK attitude yaw (stationary at t=0)
        # This is ALSO a single t=0 value, GPS-denied compliant!
        q_init, _ = initialize_quaternion_from_ppk(ppk_state)
        print(f"[INIT][PPK ATTITUDE] Using PPK attitude yaw at t=0 (stationary case)")
        print(f"[INIT][PPK ATTITUDE] PPK yaw (NED): {np.degrees(ppk_state.yaw):.1f}° → ENU: {90 - np.degrees(ppk_state.yaw):.1f}°")
        # v2.9.10.12: NO LONGER skip magnetometer - will be corrected below if mag available
        use_ppk_attitude = False  # CHANGED: Allow mag correction to align with reference
    else:
        q_init = initialize_quaternion_from_imu(imu_records[0].q)
        print(f"[INIT][IMU] Using IMU quaternion")
    
    kf.x[6, 0] = q_init[0]  # w
    kf.x[7, 0] = q_init[1]  # x
    kf.x[8, 0] = q_init[2]  # y
    kf.x[9, 0] = q_init[3]  # z
    
    # Apply magnetometer yaw correction if available 
    # v2.9.10.2 FIX: SKIP if using PPK attitude (either velocity or attitude yaw)
    if not use_ppk_attitude and mag_records is not None and mag_params is not None and len(mag_records) > 0:
        from .magnetometer import calibrate_magnetometer
        
        # v2.9.10.12: Use calibration params from config
        hard_iron = mag_params.get('hard_iron', None)
        soft_iron = mag_params.get('soft_iron', None)
        
        for mag_rec in mag_records[:50]:
            mag_cal = calibrate_magnetometer(mag_rec.mag, hard_iron=hard_iron, soft_iron=soft_iron)
            mag_norm = np.linalg.norm(mag_cal)
            
            if mag_params.get('min_field', 0.1) <= mag_norm <= mag_params.get('max_field', 100.0):
                q_state = kf.x[6:10, 0]
                q_corrected, applied = apply_mag_yaw_correction(
                    q_state, mag_cal,
                    mag_params.get('declination', 0.0),
                    mag_params.get('use_raw_heading', True)
                )
                
                if applied:
                    kf.x[6:10, 0] = q_corrected
                    print(f"[INIT][MAG] Applied magnetometer yaw correction (no PPK available)")
                break
    elif use_ppk_attitude:
        print(f"[INIT][MAG] Skipping magnetometer correction (using PPK yaw at t=0)")
    
    # Initialize biases
    has_static_calibration = False
    bg_init = np.zeros(3)
    ba_init = np.zeros(3)
    
    if initial_gyro_bias is not None and np.any(initial_gyro_bias != 0):
        bg_init = initial_gyro_bias
        ba_init = initial_accel_bias if initial_accel_bias is not None else np.zeros(3)
        print(f"[DEBUG][BIAS] Using config-provided initial biases")
        has_static_calibration = True
    elif estimate_imu_bias and len(imu_records) >= 100:
        bg_init, ba_init = estimate_imu_bias_from_static(imu_records, imu_params)
        has_static_calibration = True
    
    kf.x[10:13, 0] = bg_init
    kf.x[13:16, 0] = ba_init
    
    # Initialize covariance
    kf.P = initialize_covariance(imu_params, estimate_imu_bias, has_static_calibration)
    
    # Print state summary
    print(f"\n=== Initial State (ESKF: nominal 16D, error 15D) ===")
    print(f"[STATE] Position (p_I): [{kf.x[0,0]:.3f}, {kf.x[1,0]:.3f}, {kf.x[2,0]:.3f}] m (ENU, {z_mode})")
    print(f"[STATE] Velocity (v_I): [{kf.x[3,0]:.6f}, {kf.x[4,0]:.6f}, {kf.x[5,0]:.6f}] m/s (ENU)")
    print(f"[STATE] Quaternion (q_I): [{kf.x[6,0]:.4f}, {kf.x[7,0]:.4f}, "
          f"{kf.x[8,0]:.4f}, {kf.x[9,0]:.4f}] (w,x,y,z)")
    print(f"[STATE] Gyro bias (b_g): [{kf.x[10,0]:.6f}, {kf.x[11,0]:.6f}, {kf.x[12,0]:.6f}] rad/s")
    print(f"[STATE] Accel bias (b_a): [{kf.x[13,0]:.6f}, {kf.x[14,0]:.6f}, {kf.x[15,0]:.6f}] m/s²")
    
    return InitialState(
        position=kf.x[0:3, 0].copy(),
        velocity=kf.x[3:6, 0].copy(),
        quaternion=kf.x[6:10, 0].copy(),
        gyro_bias=bg_init,
        accel_bias=ba_init,
        origin_lat=lat0,
        origin_lon=lon0,
        msl0=msl0,
        dem0=dem0,
        agl0=agl0 if dem0 is not None else None
    )


def get_num_clones(kf) -> int:
    """
    Get number of camera clones in state vector.
    
    Args:
        kf: ExtendedKalmanFilter instance
        
    Returns:
        Number of camera clones
    """
    return (kf.x.shape[0] - 16) // 7


def get_error_state_dim(kf) -> int:
    """
    Get error state dimension (15 + 6*M).
    
    Args:
        kf: ExtendedKalmanFilter instance
        
    Returns:
        Error state dimension
    """
    num_clones = get_num_clones(kf)
    return 15 + 6 * num_clones


# ===============================
# IMU-GNSS Lever Arm Compensation
# ===============================

def gnss_to_imu_position(p_gnss_enu: np.ndarray, 
                         R_body_to_world: np.ndarray,
                         lever_arm: np.ndarray) -> np.ndarray:
    """
    Convert GNSS antenna position to IMU position using lever arm.
    
    The GNSS antenna is offset from IMU in body frame by lever_arm.
    PPK ground truth gives GNSS antenna position, but VIO state tracks IMU position.
    
    Formula: p_imu = p_gnss - R_body_to_world @ lever_arm
    
    Args:
        p_gnss_enu: GNSS position in ENU world frame [x, y, z] meters
        R_body_to_world: Rotation matrix from body (FRD) to world (ENU)
        lever_arm: IMU-GNSS lever arm in body frame [x, y, z] meters
        
    Returns:
        p_imu_enu: IMU position in ENU world frame [x, y, z] meters
    """
    lever_arm_world = R_body_to_world @ lever_arm
    p_imu = p_gnss_enu - lever_arm_world
    return p_imu


def imu_to_gnss_position(p_imu_enu: np.ndarray, 
                         R_body_to_world: np.ndarray,
                         lever_arm: np.ndarray) -> np.ndarray:
    """
    Convert IMU position to GNSS antenna position using lever arm.
    
    Inverse of gnss_to_imu_position().
    Formula: p_gnss = p_imu + R_body_to_world @ lever_arm
    
    Args:
        p_imu_enu: IMU position in ENU world frame [x, y, z] meters
        R_body_to_world: Rotation matrix from body (FRD) to world (ENU)
        lever_arm: IMU-GNSS lever arm in body frame [x, y, z] meters
        
    Returns:
        p_gnss_enu: GNSS position in ENU world frame [x, y, z] meters
    """
    lever_arm_world = R_body_to_world @ lever_arm
    p_gnss = p_imu_enu + lever_arm_world
    return p_gnss


def load_ground_truth_initial_yaw(path: str) -> Optional[float]:
    """
    Load initial yaw (heading) from PPK ground truth file.
    (Legacy function - use load_ppk_initial_state() for comprehensive data)
    
    Returns yaw in radians (NED convention: 0=North, positive=CW) or None if not found.
    """
    from .data_loaders import load_ppk_initial_state
    state = load_ppk_initial_state(path)
    if state is not None:
        return state.yaw
    return None
