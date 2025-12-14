#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Kalman Filter Module

Error-State Kalman Filter (ESKF) implementation for VIO - OpenVINS style.
"""

import sys
from copy import deepcopy
from math import log, exp, sqrt

import numpy as np
from numpy import dot, zeros, eye
import scipy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z

from .math_utils import quat_boxplus, skew_symmetric


def ensure_covariance_valid(P: np.ndarray, label: str = "", 
                            symmetrize: bool = True, 
                            check_psd: bool = True,
                            min_eigenvalue: float = 1e-9,
                            log_condition: bool = False) -> np.ndarray:
    """
    Ensure covariance matrix is valid (symmetric + positive semi-definite).
    
    Numerical errors during propagation/update can cause:
    1. Asymmetry: P ≠ P^T (floating-point rounding)
    2. Negative eigenvalues: loss of PSD property (catastrophic)
    
    Args:
        P: Covariance matrix (n×n)
        label: Debug label for logging
        symmetrize: Force symmetry
        check_psd: Check and fix negative eigenvalues
        min_eigenvalue: Minimum allowed eigenvalue
        log_condition: Log condition number
    
    Returns:
        P_valid: Fixed covariance matrix
    """
    n = P.shape[0]
    
    # 1. Symmetrization
    if symmetrize:
        asymmetry = np.linalg.norm(P - P.T, ord='fro')
        if asymmetry > 1e-6:
            print(f"[COV_CHECK] {label}: Asymmetry detected (||P - P^T|| = {asymmetry:.3e}), symmetrizing")
        P = (P + P.T) / 2.0
    
    # 2. PSD check
    if check_psd:
        try:
            eigvals = np.linalg.eigvalsh(P)
            lambda_min = eigvals[0]
            lambda_max = eigvals[-1]
            
            if log_condition and lambda_min > 1e-12:
                cond = lambda_max / lambda_min
                if cond > 1e10:
                    print(f"[COV_CHECK] {label}: High condition number κ(P) = {cond:.3e}")
            
            if lambda_min < min_eigenvalue:
                jitter = abs(lambda_min) + min_eigenvalue
                print(f"[COV_CHECK] {label}: Negative eigenvalue λ_min = {lambda_min:.3e}, "
                      f"adding jitter ε = {jitter:.3e}")
                P = P + jitter * np.eye(n, dtype=float)
        
        except np.linalg.LinAlgError as e:
            print(f"[COV_CHECK] {label}: Eigenvalue computation failed: {e}")
            P = P + 1e-6 * np.eye(n, dtype=float)
    
    return P


def propagate_error_state_covariance(P: np.ndarray, Phi: np.ndarray, 
                                      Q: np.ndarray, num_clones: int) -> np.ndarray:
    """
    Propagate error-state covariance with camera clones.
    
    Full error state: [δp, δv, δθ, δbg, δba, δθ_C1, δp_C1, ...]
    Dimensions: 15 (core) + 6*num_clones
    
    Args:
        P: Current error-state covariance
        Phi: Core error-state transition (15×15)
        Q: Core process noise (15×15)
        num_clones: Number of camera clones
    
    Returns:
        P_new: Propagated covariance
    """
    err_dim = 15 + 6 * num_clones
    
    # Build full Φ matrix
    phi_full = np.eye(err_dim, dtype=float)
    phi_full[0:15, 0:15] = Phi
    
    # Build full Q matrix
    q_full = np.zeros((err_dim, err_dim), dtype=float)
    q_full[0:15, 0:15] = Q
    
    # Propagate: P_k+1 = Φ * P_k * Φ^T + Q
    p_new = phi_full @ P @ phi_full.T + q_full
    
    # Ensure symmetry
    p_new = (p_new + p_new.T) / 2
    
    # Covariance floor
    p_pos_min = 1.0**2
    p_vel_min = 0.5**2
    p_rollpitch_min = 0.02**2
    p_yaw_min = 0.02**2
    
    for i in range(3):
        if p_new[i, i] < p_pos_min:
            p_new[i, i] = p_pos_min
    
    for i in range(3, 6):
        if p_new[i, i] < p_vel_min:
            p_new[i, i] = p_vel_min
    
    for i in range(6, 8):
        if p_new[i, i] < p_rollpitch_min:
            p_new[i, i] = p_rollpitch_min
    if p_new[8, 8] < p_yaw_min:
        p_new[8, 8] = p_yaw_min
    
    return p_new


class ExtendedKalmanFilter:
    """
    Error-State Kalman Filter (ESKF) for VIO - OpenVINS style.
    
    State vector layout (nominal state x):
      - Core IMU state (16 elements):
        * p (0:3): position [m]
        * v (3:6): velocity [m/s]
        * q (6:10): quaternion [w,x,y,z]
        * bg (10:13): gyro bias [rad/s]
        * ba (13:16): accel bias [m/s²]
      
      - Camera clones (7 elements each):
        * q_C (0:4): camera quaternion
        * p_C (4:7): camera position
    
    Error-state covariance P uses 3D rotation vector instead of 4D quaternion:
      - Core error state (15 elements):
        * δp (0:3): position error
        * δv (3:6): velocity error
        * δθ (6:9): rotation error (3D)
        * δbg (9:12): gyro bias error
        * δba (12:15): accel bias error
      
      - Camera clone errors (6 elements each):
        * δθ_C (0:3): rotation error
        * δp_C (3:6): position error
    """
    
    def __init__(self, dim_x: int, dim_z: int, dim_u: int = 0):
        """
        Initialize ESKF.
        
        Args:
            dim_x: Nominal state dimension (16 + 7*N)
            dim_z: Measurement dimension
            dim_u: Control input dimension
        """
        self.dim_x = dim_x  # Nominal state dimension
        self.dim_z = dim_z
        self.dim_u = dim_u
        
        # Nominal state (16 + 7*N)
        self.x = zeros((dim_x, 1))
        
        # Error-state covariance (15 + 6*N) - CRITICAL FIX!
        self.dim_err = self._compute_error_dim(dim_x)
        self.P = eye(self.dim_err)
        
        self.B = 0
        self.F = np.eye(self.dim_err)  # Use error dimension
        self.R = eye(dim_z)
        self.Q = eye(self.dim_err)  # Use error dimension
        self.y = zeros((dim_z, 1))

        z = np.array([None]*self.dim_z)
        self.z = reshape_z(z, self.dim_z, self.x.ndim)

        self.K = np.zeros((self.dim_err, dim_z))  # Kalman gain uses error dimension
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))
        self.SI = np.zeros((dim_z, dim_z))

        self._I = np.eye(self.dim_err)  # Identity uses error dimension

        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
    
    def _compute_error_dim(self, nominal_dim: int) -> int:
        """
        Compute error-state dimension from nominal state dimension.
        
        Nominal: 16 + 7*N (quaternion = 4D)
        Error:   15 + 6*N (rotation = 3D)
        
        Args:
            nominal_dim: Nominal state dimension
        
        Returns:
            Error-state dimension
        """
        if nominal_dim < 16:
            raise ValueError(f"Nominal dimension {nominal_dim} < 16 (minimum core state)")
        
        num_clones = (nominal_dim - 16) // 7
        if (nominal_dim - 16) % 7 != 0:
            raise ValueError(f"Invalid nominal dimension {nominal_dim}: not 16+7*N")
        
        return 15 + 6 * num_clones
    
    def nominal_to_error_idx(self, nominal_idx: int) -> int:
        """
        Map nominal state index to error-state index.
        
        Nominal state:
          [p(0:3), v(3:6), q(6:10), bg(10:13), ba(13:16), q_C1(16:20), p_C1(20:23), ...]
        
        Error state:
          [δp(0:3), δv(3:6), δθ(6:9), δbg(9:12), δba(12:15), δθ_C1(15:18), δp_C1(18:21), ...]
        
        Args:
            nominal_idx: Index in nominal state
        
        Returns:
            Index in error state (-1 if quaternion component w/x/y)
        """
        # Core state
        if nominal_idx < 6:
            return nominal_idx  # p, v: direct mapping
        elif nominal_idx < 10:
            # Quaternion (6-9) → rotation (6-8)
            # Only map to first component (w→δθ_x), others handled by rotation update
            if nominal_idx == 6:
                return 6  # w → δθ_x
            else:
                return -1  # x,y,z components don't map directly
        elif nominal_idx < 16:
            return nominal_idx - 1  # bg, ba: shift by 1
        else:
            # Camera clones
            clone_offset = nominal_idx - 16
            clone_id = clone_offset // 7
            within_clone = clone_offset % 7
            
            if within_clone < 4:
                # Quaternion → rotation
                if within_clone == 0:
                    return 15 + 6 * clone_id  # w → δθ_x
                else:
                    return -1
            else:
                # Position
                return 15 + 6 * clone_id + (within_clone - 4) + 3

    def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):
        """Performs the predict/update innovation."""
        if not isinstance(args, tuple):
            args = (args,)
        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        F = self.F
        B = self.B
        P = self.P
        Q = self.Q
        R = self.R
        x = self.x

        H = HJacobian(x, *args)

        x = dot(F, x) + dot(B, u)
        P = dot(F, P).dot(F.T) + Q

        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

        PHT = dot(P, H.T)
        self.S = dot(H, PHT) + R
        self.SI = linalg.inv(self.S)
        self.K = dot(PHT, self.SI)

        self.y = z - Hx(x, *hx_args)
        self.x = x + dot(self.K, self.y)

        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
               residual=np.subtract):
        """Performs the update innovation with ESKF."""
        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(args, tuple):
            args = (args,)
        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        H = HJacobian(self.x, *args)

        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        
        try:
            self.K = PHT.dot(linalg.inv(self.S))
        except np.linalg.LinAlgError:
            print("[ESKF] WARNING: Singular S matrix, rejecting update")
            self.z = deepcopy(z)
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        hx = Hx(self.x, *hx_args)
        self.y = residual(z, hx)
        
        # Compute error-state correction (δx)
        dx = dot(self.K, self.y)
        
        if dx.shape[0] > 8 and abs(dx[8, 0]) > 0.001:
            print(f"[ESKF-DEBUG] δx[8] (δyaw)={np.degrees(dx[8,0]):.2f}°, "
                  f"innovation={np.degrees(self.y[0,0]) if self.y.shape[0]==1 else 'N/A'}°")
        
        # Apply error-state correction to nominal state
        self._apply_error_state_correction(dx)

        # Update error-state covariance (Joseph form)
        I_err = np.eye(self.P.shape[0])
        I_KH = I_err - dot(self.K, H)
        self.P = dot(I_KH, self.P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)
        
        # Ensure covariance validity after update
        self.P = ensure_covariance_valid(
            self.P,
            label="EKF-Update",
            symmetrize=True,
            check_psd=True,
            min_eigenvalue=1e-9
        )
        
        # Covariance floor
        p_pos_min = 1.0**2
        p_vel_min = 0.5**2
        p_rollpitch_min = 0.02**2
        p_yaw_min = 0.02**2
        
        for i in range(min(3, self.P.shape[0])):
            if self.P[i, i] < p_pos_min:
                self.P[i, i] = p_pos_min
        for i in range(3, min(6, self.P.shape[0])):
            if self.P[i, i] < p_vel_min:
                self.P[i, i] = p_vel_min
        for i in range(6, min(8, self.P.shape[0])):
            if self.P[i, i] < p_rollpitch_min:
                self.P[i, i] = p_rollpitch_min
        if self.P.shape[0] > 8 and self.P[8, 8] < p_yaw_min:
            self.P[8, 8] = p_yaw_min

        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
    
    def _apply_error_state_correction(self, dx):
        """
        Apply error-state correction to nominal state.
        
        Error state δx (15+6N): [δp, δv, δθ, δbg, δba, δθ_C1, δp_C1, ...]
        Nominal state x (16+7N): [p, v, q, bg, ba, q_C1, p_C1, ...]
        
        Args:
            dx: Error-state correction (15+6N)
        """
        # Core state corrections
        dp = dx[0:3, 0]      # Position error
        dv = dx[3:6, 0]      # Velocity error
        dtheta = dx[6:9, 0]  # Rotation error (3D)
        dbg = dx[9:12, 0]    # Gyro bias error
        dba = dx[12:15, 0]   # Accel bias error
        
        # Additive corrections for p, v, biases
        self.x[0:3, 0] += dp
        self.x[3:6, 0] += dv
        self.x[10:13, 0] += dbg
        self.x[13:16, 0] += dba
        
        # Multiplicative correction for quaternion (manifold update)
        q_old = self.x[6:10, 0]
        q_new = quat_boxplus(q_old, dtheta)
        self.x[6:10, 0] = q_new
        
        # Camera clone corrections
        num_clones = (self.x.shape[0] - 16) // 7
        for i in range(num_clones):
            # Error-state indices (15 + 6*i)
            err_base = 15 + i * 6
            dtheta_c = dx[err_base:err_base+3, 0]    # Clone rotation error
            dp_c = dx[err_base+3:err_base+6, 0]      # Clone position error
            
            # Nominal state indices (16 + 7*i)
            nom_base = 16 + i * 7
            
            # Multiplicative quaternion update
            q_c_old = self.x[nom_base:nom_base+4, 0]
            q_c_new = quat_boxplus(q_c_old, dtheta_c)
            self.x[nom_base:nom_base+4, 0] = q_c_new
            
            # Additive position update
            self.x[nom_base+4:nom_base+7, 0] += dp_c

    def predict_x(self, u=0):
        """Predicts the next state of X."""
        self.x = dot(self.F, self.x) + dot(self.B, u)

    def predict(self, u=0):
        """Predict next state using Kalman filter propagation."""
        self.predict_x(u)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    @property
    def log_likelihood(self):
        """log-likelihood of the last measurement."""
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """Computed from the log-likelihood."""
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """Mahalanobis distance of innovation."""
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(float(dot(dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis

    def __repr__(self):
        return '\n'.join([
            'ExtendedKalmanFilter object',
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('F', self.F),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('K', self.K),
            pretty_str('y', self.y),
            pretty_str('S', self.S),
            pretty_str('likelihood', self.likelihood),
            pretty_str('log-likelihood', self.log_likelihood),
            pretty_str('mahalanobis', self.mahalanobis)
        ])
