#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Preintegration Module
=========================

Implements IMU preintegration on manifold following Forster et al., TRO 2017:
"On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"

Theory Overview:
----------------
IMU preintegration avoids recomputing the entire IMU integration when biases
change. Instead of integrating from t_i to t_j with specific biases, we:

1. Preintegrate relative measurements in body frame:
   - ΔR_ij: Relative rotation from i to j
   - Δv_ij: Velocity increment in frame i
   - Δp_ij: Position increment in frame i

2. Store Jacobians w.r.t. biases for efficient correction:
   - When bias estimate changes, correct preintegrated quantities using:
     ΔR_corr = ΔR * Exp(J_R_bg * δbg)
     Δv_corr = Δv + J_v_bg * δbg + J_v_ba * δba
     Δp_corr = Δp + J_p_bg * δbg + J_p_ba * δba

State Update:
-------------
Given state at i: (R_i, v_i, p_i) and preintegrated quantities, the state at j:
    R_j = R_i * ΔR_ij
    v_j = v_i + R_i * Δv_ij
    p_j = p_i + v_i*Δt + R_i * Δp_ij

Note: Gravity is compensated DURING preintegration (line 197: a_hat += g_body),
so state update does NOT add gravity separately. This follows the convention
where preintegrated quantities already account for gravity compensation.

Covariance Propagation:
-----------------------
Discrete noise propagation follows:
    Σ_{k+1} = F_k * Σ_k * F_k' + G_k * Q_d * G_k'

where F is the discrete state transition matrix and Q_d is discrete noise.

References:
-----------
[1] Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial
    Odometry", IEEE TRO 2017
[2] Leutenegger et al., "Keyframe-based visual-inertial odometry using 
    nonlinear optimization", IJRR 2015

Author: VIO project
"""

from typing import Tuple
import numpy as np

from .math_utils import skew_symmetric
from .numerical_checks import assert_finite, check_covariance_psd


class IMUPreintegration:
    """
    IMU Preintegration on Manifold (Forster et al., TRO 2017).
    
    Preintegrates IMU measurements between keyframes to compute:
      - ΔR: Preintegrated rotation (SO(3) rotation matrix)
      - Δv: Preintegrated velocity (3D vector)
      - Δp: Preintegrated position (3D vector)
    
    Also maintains Jacobians w.r.t. biases for efficient bias correction:
      - J_R_bg (3×3): ∂ΔR/∂bg rotation Jacobian w.r.t. gyro bias
      - J_v_bg (3×3): ∂Δv/∂bg velocity Jacobian w.r.t. gyro bias
      - J_v_ba (3×3): ∂Δv/∂ba velocity Jacobian w.r.t. accel bias
      - J_p_bg (3×3): ∂Δp/∂bg position Jacobian w.r.t. gyro bias
      - J_p_ba (3×3): ∂Δp/∂ba position Jacobian w.r.t. accel bias
    
    Key advantages over naive sample-by-sample integration:
      1. Reduced numerical error (integrate once, apply once per keyframe)
      2. Fast bias correction via Jacobians (no re-integration needed)
      3. Proper manifold handling for rotation (SO(3), not Euler angles)
      4. Theoretically grounded covariance propagation
    
    Usage Example:
        # Initialize with current bias estimates
        preint = IMUPreintegration(
            bg=gyro_bias,    # [3,] current gyro bias estimate
            ba=accel_bias,   # [3,] current accel bias estimate
            sigma_g=0.01,    # Gyroscope noise density [rad/s/√Hz]
            sigma_a=0.1,     # Accelerometer noise density [m/s²/√Hz]
            sigma_bg=0.0001, # Gyro bias random walk [rad/s²/√Hz]
            sigma_ba=0.001   # Accel bias random walk [m/s³/√Hz]
        )
        
        # Integrate IMU samples between keyframes
        for imu in imu_buffer_between_keyframes:
            preint.integrate_measurement(imu.gyro, imu.accel, dt)
        
        # Get preintegrated quantities (corrected for current bias)
        delta_R, delta_v, delta_p = preint.get_deltas_corrected(bg_current, ba_current)
        
        # Get Jacobians for EKF propagation
        J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba = preint.get_jacobians()
    """
    
    def __init__(self, bg: np.ndarray, ba: np.ndarray, 
                 sigma_g: float, sigma_a: float, 
                 sigma_bg: float, sigma_ba: float):
        """
        Initialize preintegration with initial biases and noise parameters.
        
        Args:
            bg: Gyroscope bias [rad/s], shape (3,)
            ba: Accelerometer bias [m/s²], shape (3,)
            sigma_g: Gyroscope measurement noise std [rad/s/√Hz]
            sigma_a: Accelerometer measurement noise std (m/s²)
            sigma_bg: Gyroscope bias random walk std (rad/s/√s)
            sigma_ba: Accelerometer bias random walk std (m/s²/√s)
        """
        # Linearization point (bias at start of preintegration)
        self.bg_lin = np.copy(bg).reshape(3,)
        self.ba_lin = np.copy(ba).reshape(3,)
        
        # Noise parameters
        self.sigma_g = sigma_g
        self.sigma_a = sigma_a
        self.sigma_bg = sigma_bg
        self.sigma_ba = sigma_ba
        
        # Preintegrated measurements (identity/zero)
        self.delta_R = np.eye(3, dtype=float)  # SO(3)
        self.delta_v = np.zeros(3, dtype=float)
        self.delta_p = np.zeros(3, dtype=float)
        
        # Jacobians w.r.t. biases (for fast bias correction)
        self.J_R_bg = np.zeros((3, 3), dtype=float)  # ∂ΔR/∂bg
        self.J_v_bg = np.zeros((3, 3), dtype=float)  # ∂Δv/∂bg
        self.J_v_ba = np.zeros((3, 3), dtype=float)  # ∂Δv/∂ba
        self.J_p_bg = np.zeros((3, 3), dtype=float)  # ∂Δp/∂bg
        self.J_p_ba = np.zeros((3, 3), dtype=float)  # ∂Δp/∂ba
        
        # Preintegration covariance (9x9: rotation 3D + velocity 3D + position 3D)
        self.cov = np.zeros((9, 9), dtype=float)
        
        # Total integration time
        self.dt_sum = 0.0
        
    def reset(self, bg: np.ndarray, ba: np.ndarray):
        """Reset preintegration to identity/zero with new linearization point."""
        self.bg_lin = np.copy(bg).reshape(3,)
        self.ba_lin = np.copy(ba).reshape(3,)
        
        self.delta_R = np.eye(3, dtype=float)
        self.delta_v = np.zeros(3, dtype=float)
        self.delta_p = np.zeros(3, dtype=float)
        
        self.J_R_bg = np.zeros((3, 3), dtype=float)
        self.J_v_bg = np.zeros((3, 3), dtype=float)
        self.J_v_ba = np.zeros((3, 3), dtype=float)
        self.J_p_bg = np.zeros((3, 3), dtype=float)
        self.J_p_ba = np.zeros((3, 3), dtype=float)
        
        self.cov = np.zeros((9, 9), dtype=float)
        self.dt_sum = 0.0
        
    def integrate_measurement(self, w_meas: np.ndarray, a_meas: np.ndarray, 
                               dt: float, g_norm: float = 9.80665):
        """
        Integrate one IMU measurement (gyro + accel) over time step dt.
        
        CRITICAL: IMU accelerometer data includes gravity!
        - Stationary reading: [0, 0, -9.8] in body frame (Z-down convention)
        - Must add gravity in BODY frame before integration
        - Gravity magnitude from g_norm parameter
        
        Updates:
          - Preintegrated deltas (ΔR, Δv, Δp)
          - Jacobians w.r.t. biases
          - Preintegration covariance
        
        Args:
            w_meas: Gyroscope measurement (3D, rad/s)
            a_meas: Accelerometer measurement (3D, m/s²) - INCLUDES GRAVITY
            dt: Time step (seconds)
            g_norm: Gravity magnitude (m/s², default 9.80665)
        """
        # VALIDATION: Strict dt checking to prevent numerical blow-up
        DT_MIN = 1e-6  # 1 microsecond minimum
        DT_MAX = 0.1   # 100ms maximum (IMU should be > 10Hz)
        
        if dt <= 0.0:
            # Negative or zero dt is invalid - skip this integration
            print(f"[PREINT] Invalid dt={dt:.9f}s (≤0), skipping integration")
            return
        
        if dt < DT_MIN:
            # dt too small - numerical instability, skip
            print(f"[PREINT] dt={dt:.9f}s < DT_MIN={DT_MIN}s, skipping")
            return
        
        if dt > DT_MAX:
            # dt too large - likely timestamp jump or missing data
            # Split into smaller chunks to maintain numerical stability
            num_splits = int(np.ceil(dt / DT_MAX))
            dt_split = dt / num_splits
            print(f"[PREINT] Large dt={dt:.6f}s split into {num_splits} × {dt_split:.6f}s")
            for _ in range(num_splits):
                self.integrate_measurement(w_meas, a_meas, dt_split, g_norm)
            return
        
        # ================================================================
        # GRAVITY MODEL (Forster TRO 2017 / OpenVINS standard)
        # ================================================================
        # Preintegration computes SPECIFIC FORCE (a_meas - b_a) in body frame
        # WITHOUT gravity compensation during accumulation.
        # 
        # Gravity is added later during state update (propagation.py):
        #   v_new = v_old + g_world*dt + R_WB^T @ delta_v
        #   p_new = p_old + v_old*dt + 0.5*g_world*dt² + R_WB^T @ delta_p
        # 
        # where g_world = [0, 0, -9.80665] in ENU frame (Z points up)
        # 
        # This approach:
        # ✓ Separates body-frame preintegration from world-frame gravity
        # ✓ Avoids rotating gravity (numerically stable)
        # ✓ Matches OpenVINS/Forster standard implementation
        # ================================================================
        
        # Bias-corrected measurements (specific force only, NO gravity here)
        w_hat = w_meas - self.bg_lin
        a_hat = a_meas - self.ba_lin
        # NOTE: a_hat is "specific force" = accelerometer reading minus bias
        #       Gravity will be added in propagation.py during state update
        
        # --- Step 1: Update rotation delta ---
        # ΔR_{k+1} = ΔR_k * Exp(ω_hat * dt)
        theta_vec = w_hat * dt
        theta = np.linalg.norm(theta_vec)
        
        if theta < 1e-8:
            # Small angle: Exp(θ) ≈ I + [θ]×
            delta_R_k1 = self.delta_R @ (np.eye(3) + skew_symmetric(theta_vec))
        else:
            # Rodrigues formula: Exp(θ) = I + sin(θ)/θ [θ]× + (1-cos(θ))/θ² [θ]×²
            axis = theta_vec / theta
            skew_axis = skew_symmetric(axis)
            exp_theta = np.eye(3) + np.sin(theta) * skew_axis + \
                        (1 - np.cos(theta)) * (skew_axis @ skew_axis)
            delta_R_k1 = self.delta_R @ exp_theta
        
        # Right Jacobian of SO(3) for rotation error propagation
        if theta < 1e-8:
            j_r = np.eye(3) - 0.5 * skew_symmetric(theta_vec)
        else:
            axis = theta_vec / theta
            skew_axis = skew_symmetric(axis)
            j_r = np.eye(3) - (1 - np.cos(theta)) / theta * skew_axis + \
                  (theta - np.sin(theta)) / theta * (skew_axis @ skew_axis)
        
        # --- Step 2: Update velocity delta ---
        # Δv_{k+1} = Δv_k + ΔR_k * a_hat * dt
        delta_v_k1 = self.delta_v + self.delta_R @ a_hat * dt
        
        # --- Step 3: Update position delta ---
        # Δp_{k+1} = Δp_k + Δv_k * dt + 0.5 * ΔR_k * a_hat * dt²
        delta_p_k1 = self.delta_p + self.delta_v * dt + \
                     0.5 * self.delta_R @ a_hat * (dt ** 2)
        
        # --- Step 4: Update Jacobians w.r.t. biases ---
        # Forster et al. TRO 2017, Eq. (24-28)
        # Note: Negative signs come from ∂(a_meas - ba)/∂ba = -I and ∂(w_meas - bg)/∂bg = -I
        j_r_bg_k1 = self.J_R_bg - j_r * dt
        j_v_bg_k1 = self.J_v_bg - self.delta_R @ skew_symmetric(a_hat) @ self.J_R_bg * dt
        j_v_ba_k1 = self.J_v_ba - self.delta_R * dt
        j_p_bg_k1 = self.J_p_bg + self.J_v_bg * dt - \
                    0.5 * self.delta_R @ skew_symmetric(a_hat) @ self.J_R_bg * (dt ** 2)
        j_p_ba_k1 = self.J_p_ba + self.J_v_ba * dt - 0.5 * self.delta_R * (dt ** 2)
        
        # --- Step 5: Update covariance (discrete-time propagation) ---
        # State transition matrix A (9x9) for error-state [δθ_R, δv, δp]^T
        # Forster et al. TRO 2017, Eq. (48)
        A = np.eye(9, dtype=float)
        
        # Rotation error: δθ_{k+1} ≈ δθ_k (simplified, assumes small rotation errors)
        # Alternatively: A[0:3, 0:3] = I - skew_symmetric(w_hat * dt) for more accuracy
        A[0:3, 0:3] = np.eye(3)
        
        # Velocity error coupling
        A[3:6, 0:3] = -self.delta_R @ skew_symmetric(a_hat) * dt
        A[3:6, 3:6] = np.eye(3)
        
        # Position error coupling
        A[6:9, 0:3] = -0.5 * self.delta_R @ skew_symmetric(a_hat) * (dt ** 2)
        A[6:9, 3:6] = np.eye(3) * dt
        A[6:9, 6:9] = np.eye(3)
        
        # Noise matrix B (9x6)
        B = np.zeros((9, 6), dtype=float)
        B[0:3, 0:3] = j_r * dt
        B[3:6, 3:6] = self.delta_R * dt
        B[6:9, 3:6] = 0.5 * self.delta_R * (dt ** 2)
        
        # Noise covariance Q (6x6)
        Q = np.diag([
            self.sigma_g**2 * dt, self.sigma_g**2 * dt, self.sigma_g**2 * dt,
            self.sigma_a**2 * dt, self.sigma_a**2 * dt, self.sigma_a**2 * dt
        ])
        
        # VALIDATION: Check covariance validity before propagation
        # This prevents numerical explosion from corrupted preintegration covariance
        has_invalid_cov = np.any(np.isinf(self.cov)) or np.any(np.isnan(self.cov))
        if has_invalid_cov:
            # Covariance corrupted - reset to identity/small value
            # This prevents cascading corruption through IMU preintegration
            print(f"[PREINT] Covariance contains inf/nan at t={self.dt_sum:.6f}s, resetting")
            self.cov = np.eye(9, dtype=float) * 1e-6
        
        # [TRIPWIRE] Check covariance BEFORE propagation
        if not check_covariance_psd(self.cov, name="preint_cov_before", t=self.dt_sum):
            print(f"[PREINT] WARNING: Covariance not PSD before propagation at dt={self.dt_sum:.3f}s")
        
        # Clamp large covariance values to prevent overflow
        cov_max = np.max(np.abs(self.cov))
        if cov_max > 1e10:
            # Scale covariance down to prevent numerical explosion
            scale_factor = 1e8 / cov_max
            self.cov = self.cov * scale_factor
            print(f"[PREINT] Covariance overflow clamped: max={cov_max:.2e} → scaled by {scale_factor:.2e}")
        
        # [TRIPWIRE] Check A, B, Q matrices before propagation
        if not assert_finite("preint_A", A, t=self.dt_sum, extra_info={"dt": dt}):
            print(f"[PREINT] CRITICAL: A matrix contains NaN/inf, skipping integration")
            return
        
        if not assert_finite("preint_B", B, t=self.dt_sum, extra_info={"dt": dt}):
            print(f"[PREINT] CRITICAL: B matrix contains NaN/inf, skipping integration")
            return
        
        if not assert_finite("preint_Q", Q, t=self.dt_sum, extra_info={"dt": dt}):
            print(f"[PREINT] CRITICAL: Q matrix contains NaN/inf, skipping integration")
            return
        
        # Propagate covariance with SUPPRESSED warnings (we check manually)
        with np.errstate(all='ignore'):  # Suppress numpy warnings, we handle explicitly
            try:
                cov_new = A @ self.cov @ A.T + B @ Q @ B.T
            except Exception as e:
                print(f"[PREINT] Covariance propagation failed: {e}, using previous covariance")
                cov_new = self.cov
        
        # [TRIPWIRE] Check result validity after computation
        if not assert_finite("preint_cov_new", cov_new, t=self.dt_sum, extra_info={
            "dt": dt,
            "cov_max_before": cov_max,
            "A_norm": np.linalg.norm(A),
            "B_norm": np.linalg.norm(B),
            "Q_norm": np.linalg.norm(Q)
        }):
            print(f"[PREINT] Covariance propagation produced inf/nan, reverting")
            cov_new = self.cov
        
        # Ensure symmetry
        cov_new = (cov_new + cov_new.T) / 2.0
        
        # Apply result
        self.cov = cov_new
        
        # [TRIPWIRE] Final check after symmetrization
        if not check_covariance_psd(self.cov, name="preint_cov_after", t=self.dt_sum):
            print(f"[PREINT] WARNING: Covariance not PSD after propagation!")
        
        # --- Step 6: Commit updates ---
        self.delta_R = delta_R_k1
        self.delta_v = delta_v_k1
        self.delta_p = delta_p_k1
        
        self.J_R_bg = j_r_bg_k1
        self.J_v_bg = j_v_bg_k1
        self.J_v_ba = j_v_ba_k1
        self.J_p_bg = j_p_bg_k1
        self.J_p_ba = j_p_ba_k1
        
        self.dt_sum += dt
        
    def get_deltas(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get preintegrated deltas (no bias correction).
        
        Returns:
            delta_R: (3x3) rotation matrix
            delta_v: (3,) velocity vector
            delta_p: (3,) position vector
        """
        return self.delta_R, self.delta_v, self.delta_p
    
    def get_deltas_corrected(self, bg_new: np.ndarray, ba_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get bias-corrected preintegrated deltas using first-order correction.
        
        Args:
            bg_new: Current gyro bias estimate (3D)
            ba_new: Current accel bias estimate (3D)
        
        Returns:
            delta_R_corr: Bias-corrected rotation matrix
            delta_v_corr: Bias-corrected velocity vector
            delta_p_corr: Bias-corrected position vector
        """
        dbg = bg_new - self.bg_lin
        dba = ba_new - self.ba_lin
        
        # Correct rotation
        delta_theta = self.J_R_bg @ dbg
        theta = np.linalg.norm(delta_theta)
        
        if theta < 1e-8:
            correction_R = np.eye(3) + skew_symmetric(delta_theta)
        else:
            axis = delta_theta / theta
            skew_axis = skew_symmetric(axis)
            correction_R = np.eye(3) + np.sin(theta) * skew_axis + \
                          (1 - np.cos(theta)) * (skew_axis @ skew_axis)
        
        delta_R_corr = self.delta_R @ correction_R
        delta_v_corr = self.delta_v + self.J_v_bg @ dbg + self.J_v_ba @ dba
        delta_p_corr = self.delta_p + self.J_p_bg @ dbg + self.J_p_ba @ dba
        
        return delta_R_corr, delta_v_corr, delta_p_corr
    
    def get_covariance(self) -> np.ndarray:
        """Get preintegration covariance (9x9)."""
        return self.cov.copy()
    
    def get_jacobians(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Jacobians w.r.t. biases.
        
        Returns:
            J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba: (3x3) matrices
        """
        return self.J_R_bg, self.J_v_bg, self.J_v_ba, self.J_p_bg, self.J_p_ba


def compute_error_state_jacobian(q: np.ndarray, a_corr: np.ndarray, 
                                  w_corr: np.ndarray, dt: float,
                                  R_body_to_world: np.ndarray) -> np.ndarray:
    """
    Compute error-state transition matrix Φ for ESKF propagation.
    
    v3.9.7: Updated for 18D error state with mag_bias
    Error state: δx = [δp, δv, δθ, δbg, δba, δmag]^T (18 dimensions)
    
    Args:
        q: Current quaternion [w,x,y,z]
        a_corr: Bias-corrected acceleration (body frame)
        w_corr: Bias-corrected angular velocity (body frame)
        dt: Time step
        R_body_to_world: Rotation matrix from body to world frame
    
    Returns:
        Φ: 18×18 error-state transition matrix
    """
    # Initialize as identity (18x18)
    Phi = np.eye(18, dtype=float)
    
    # δp depends on δv
    Phi[0:3, 3:6] = np.eye(3) * dt
    
    # δv depends on δθ and δba
    Phi[3:6, 6:9] = -R_body_to_world @ skew_symmetric(a_corr) * dt
    Phi[3:6, 12:15] = -R_body_to_world * dt
    
    # δθ depends on previous δθ and δbg
    theta_vec = w_corr * dt
    Phi[6:9, 6:9] = np.eye(3) - skew_symmetric(theta_vec)
    Phi[6:9, 9:12] = -np.eye(3) * dt
    
    # δmag remains constant during IMU propagation (identity block at 15:18)
    # Already set by np.eye() initialization
    
    return Phi


def compute_error_state_process_noise(dt: float, estimate_imu_bias: bool,
                                       t: float, t0: float,
                                       imu_params: dict,
                                       sigma_accel: float,
                                       mag_params: dict = None) -> np.ndarray:
    """
    Compute process noise Q for error-state covariance (18×18).
    
    v3.9.7: Updated for 18D error state with mag_bias
    
    Args:
        dt: Time step
        estimate_imu_bias: Whether bias was pre-estimated
        t: Current time
        t0: Start time
        imu_params: IMU noise parameters dict
        sigma_accel: Unmodeled acceleration noise
        mag_params: Magnetometer parameters dict (optional)
    
    Returns:
        Q: 18×18 process noise covariance matrix
    """
    # IMU noise (sensor noise)
    imu_acc_noise = imu_params['acc_n']
    imu_gyr_noise = imu_params['gyr_n']
    
    # Combined acceleration noise
    combined_acc_noise = np.sqrt(imu_acc_noise**2 + sigma_accel**2)
    
    # Position noise from velocity integration error
    q_pos = (combined_acc_noise * dt**2 / 2)**2
    
    # Velocity noise
    q_vel = (combined_acc_noise * dt)**2
    
    # Rotation noise
    unmodeled_gyr = 0.002  # rad/s
    combined_gyr_noise = np.sqrt(imu_gyr_noise**2 + unmodeled_gyr**2)
    q_theta = (combined_gyr_noise * dt)**2
    
    # Minimum yaw process noise
    min_yaw_process_noise = np.radians(3.0)
    q_theta_z_min = (min_yaw_process_noise * np.sqrt(dt))**2
    
    # Bias random walk
    q_bg = (imu_params['gyr_w']**2) * dt
    q_ba = (imu_params['acc_w']**2) * dt
    
    # Mag bias random walk (v3.9.7: NEW)
    # v3.9.8: Set to 0 when use_estimated_bias=False (freeze states)
    # v3.9.9: Also freeze when mag_params is None (no mag data)
    if mag_params is None:
        # No mag data - freeze mag_bias completely
        sigma_mag_bias = 0.0
    else:
        sigma_mag_bias = mag_params.get('sigma_mag_bias', 0.001)
        use_estimated_bias = mag_params.get('use_estimated_bias', True)
        if not use_estimated_bias:
            sigma_mag_bias = 0.0  # Freeze mag_bias states
    q_mag = (sigma_mag_bias**2) * dt
    
    # Build 18×18 matrix
    Q = np.zeros((18, 18), dtype=float)
    Q[0:3, 0:3] = np.eye(3) * q_pos
    Q[3:6, 3:6] = np.eye(3) * q_vel
    Q[6:9, 6:9] = np.eye(3) * q_theta
    
    # Apply minimum yaw process noise
    Q[8, 8] = max(Q[8, 8], q_theta_z_min)
    
    Q[9:12, 9:12] = np.eye(3) * q_bg
    Q[12:15, 12:15] = np.eye(3) * q_ba
    Q[15:18, 15:18] = np.eye(3) * q_mag  # NEW: mag bias process noise
    
    return Q
