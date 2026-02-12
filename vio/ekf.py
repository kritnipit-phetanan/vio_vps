#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Kalman Filter Module

Error-State Kalman Filter (ESKF) implementation for VIO - OpenVINS style.
"""

import sys
from copy import deepcopy
from math import log, exp, sqrt
from collections import defaultdict

import numpy as np
from numpy import dot, zeros, eye
import scipy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z

from .math_utils import quat_boxplus, skew_symmetric, safe_matrix_inverse
from .numerical_checks import assert_finite, check_quaternion, check_covariance_psd


def ensure_covariance_valid(P: np.ndarray, label: str = "",
                            symmetrize: bool = True,
                            check_psd: bool = True,
                            min_eigenvalue: float = 1e-7,
                            log_condition: bool = False,
                            max_value: float = None,
                            conditioner: "ExtendedKalmanFilter" = None,
                            timestamp: float = float("nan"),
                            stage: str = "") -> np.ndarray:
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
        max_value: Optional absolute clamp on covariance entries
    
    Returns:
        P_valid: Fixed covariance matrix
    """
    n = P.shape[0]
    core_dim = min(CORE_ERROR_DIM, n)
    notes = []
    event = "NONE"
    action = "PASS"
    projected = False
    reset_applied = False
    chol_failed = False
    if conditioner is not None:
        conditioner._conditioning_tick()
    
    # 1. Symmetrization
    if symmetrize:
        if np.all(np.isfinite(P)):
            asymmetry = np.linalg.norm(P - P.T, ord='fro')
            if asymmetry > 1e-6:
                notes.append(f"asym={asymmetry:.2e}")
        P = (P + P.T) / 2.0

    # 1.5 Cheap guard: finite sanitize + core diagonal floor.
    max_abs = float(max_value) if (max_value is not None and np.isfinite(max_value) and max_value > 0) else 1e8
    if not np.all(np.isfinite(P)):
        notes.append("non_finite")
        event = "NON_FINITE"
        action = "SANITIZE_RESET"
        P = np.nan_to_num(P, nan=0.0, posinf=max_abs, neginf=-max_abs)
        projected = True
        reset_applied = True

    if check_psd:
        floor = float(max(min_eigenvalue, 1e-12))
        for i in range(core_dim):
            if not np.isfinite(P[i, i]) or P[i, i] < floor:
                P[i, i] = floor

        force_projection = projected
        cond_hard = getattr(conditioner, "conditioning_cond_hard", 1e12) if conditioner is not None else 1e12
        cond_window = int(max(1, getattr(conditioner, "conditioning_cond_hard_window", 8))) if conditioner is not None else 8

        p_max = float(np.max(np.abs(P)))
        p_cond = float("inf")
        try:
            p_cond = float(np.linalg.cond(P[:core_dim, :core_dim]))
        except Exception:
            p_cond = float("inf")
        if conditioner is not None:
            conditioner._update_conditioning_max_cond(p_cond)

        if max_value is not None and np.isfinite(max_value) and max_value > 0 and p_max > float(max_value):
            notes.append(f"cap_hard={p_max:.2e}>{float(max_value):.2e}")
            force_projection = True
            if event == "NONE":
                event = "HARD_CAP"
                action = "EIGEN_PROJECT"

        cond_over = bool(np.isfinite(p_cond) and p_cond > float(cond_hard))
        if conditioner is not None:
            streak = conditioner._update_conditioning_cond_streak(cond_over)
        else:
            streak = 1 if cond_over else 0
        if cond_over and streak >= cond_window:
            allow_projection = True
            if conditioner is not None and (not conditioner._can_project_now()):
                allow_projection = False
                notes.append("cond_hard_cooldown")
            if allow_projection:
                notes.append(f"p_cond={p_cond:.2e}|th={float(cond_hard):.2e}")
                force_projection = True
                if event == "NONE":
                    event = "COND_HARD"
                    action = "EIGEN_PROJECT"

        # 2) Cholesky guard: cheap PSD test before eigen projection.
        if not force_projection:
            chol_ok = False
            for jitter in (0.0, floor * 1e-3, floor):
                try:
                    np.linalg.cholesky(P + jitter * np.eye(n, dtype=float))
                    chol_ok = True
                    break
                except np.linalg.LinAlgError:
                    continue
            if not chol_ok:
                chol_failed = True
                force_projection = True
                event = "CHOLESKY_FAIL"
                action = "EIGEN_PROJECT"
                notes.append("chol_fail")

        # 3) Eigen fallback path only when required.
        if force_projection:
            projected = True
            eig_floor = float(max(min_eigenvalue, 1e-12))
            eig_ceil = float(max_value) if (max_value is not None and np.isfinite(max_value) and max_value > eig_floor) else None
            try:
                eigvals, eigvecs = np.linalg.eigh(P)
                eigvals_proj = np.maximum(eigvals, eig_floor)
                if eig_ceil is not None:
                    eigvals_proj = np.minimum(eigvals_proj, eig_ceil)
                with np.errstate(all="ignore"):
                    P = eigvecs @ np.diag(eigvals_proj) @ eigvecs.T
                if not np.all(np.isfinite(P)):
                    raise np.linalg.LinAlgError("eigen reconstruction non-finite")
                P = (P + P.T) / 2.0
            except np.linalg.LinAlgError:
                reset_applied = True
                event = "EIGEN_FAIL"
                action = "DIAG_RESET"
                notes.append("diag_reset")
                reset_scale = eig_ceil if eig_ceil is not None else max(eig_floor, 1e-6)
                P = np.eye(n, dtype=float) * float(reset_scale)

        # Final finite-safe fallback.
        if not np.all(np.isfinite(P)):
            reset_applied = True
            event = "POST_NON_FINITE"
            action = "DIAG_RESET"
            notes.append("post_non_finite")
            P = np.eye(n, dtype=float) * float(max(min_eigenvalue, 1e-6))

        # Optional condition print for coarse debugging.
        if log_condition:
            try:
                cond = float(np.linalg.cond(P[:core_dim, :core_dim]))
                if cond > 1e10:
                    notes.append(f"cond={cond:.2e}")
            except Exception:
                pass

    if conditioner is not None and (projected or chol_failed or reset_applied):
        try:
            p_min_eig = float(np.linalg.eigvalsh((P + P.T) / 2.0)[0]) if P.size > 0 else float("nan")
        except Exception:
            p_min_eig = float("nan")
        try:
            p_cond_final = float(np.linalg.cond(P[:core_dim, :core_dim])) if core_dim > 0 else float("nan")
        except Exception:
            p_cond_final = float("inf")
        conditioner._record_conditioning_action(
            timestamp=float(timestamp),
            event=event if event != "NONE" else "PROJECTION",
            stage=str(stage or label or "unknown"),
            p_min_eig=p_min_eig,
            p_max=float(np.max(np.abs(P))) if P.size > 0 else float("nan"),
            p_cond=p_cond_final,
            action=action,
            notes="|".join(notes) if len(notes) > 0 else "",
            projected=bool(projected),
            reset_applied=bool(reset_applied),
            chol_failed=bool(chol_failed),
        )

    return P


def propagate_error_state_covariance(P: np.ndarray, Phi: np.ndarray,
                                      Q: np.ndarray, num_clones: int,
                                      conditioner: "ExtendedKalmanFilter" = None,
                                      timestamp: float = float("nan"),
                                      stage: str = "IMU_PROP") -> np.ndarray:
    """
    Propagate error-state covariance with camera clones.
    
    v3.9.7: Updated for 18D core error state (includes δmag_bias)
    Full error state: [δp, δv, δθ, δbg, δba, δmag, δθ_C1, δp_C1, ...]
    Dimensions: 18 (core) + 6*num_clones
    
    Args:
        P: Current error-state covariance
        Phi: Core error-state transition (18×18)
        Q: Core process noise (18×18)
        num_clones: Number of camera clones
    
    Returns:
        P_new: Propagated covariance
    """
    from .numerical_checks import assert_finite
    
    err_dim = CORE_ERROR_DIM + CLONE_ERROR_DIM * num_clones
    
    # Finite guard before propagation (conditioning path handles PSD repair).
    if not np.all(np.isfinite(P)):
        print(f"[EKF-PROP] CRITICAL: non-finite P before propagation, resetting")
        P = np.eye(err_dim, dtype=float) * 1e-2
    
    # [DIAGNOSTIC] Check P growth - log if abnormally large
    # Monitor FULL position covariance (x,y,z) to catch divergence in any axis
    P_trace = np.trace(P)
    P_max = np.max(np.abs(P))
    P_pos_trace = P[0,0] + P[1,1] + P[2,2]  # Total position variance
    P_pos_max = max(P[0,0], P[1,1], P[2,2])  # Worst axis
    
    if P_max > 1e6:
        print(f"[EKF-PROP] WARNING: P growing large: max={P_max:.2e}, trace={P_trace:.2e}")
        print(f"[EKF-PROP]   → P_pos: xx={P[0,0]:.1f}, yy={P[1,1]:.1f}, zz={P[2,2]:.1f}, trace={P_pos_trace:.1f}")
        print(f"[EKF-PROP]   → P_vel={P[3,3]:.2f}, P_yaw={P[8,8]:.4f}")
    
    # VALIDATION: Check P matrix validity before propagation
    # This prevents divide-by-zero and overflow from corrupted covariance
    if not assert_finite("ekf_P_before", P, extra_info={
        "P_max": P_max,
        "P_trace": P_trace,
        "err_dim": err_dim
    }):
        # P corrupted - reset with safe diagonal values
        print(f"[EKF-PROP] P contains inf/nan at entry, resetting to safe diagonal")
        P = np.eye(err_dim, dtype=float) * 1e-2
    
    # Clamp large P values to prevent overflow in matmul
    P_max = np.max(np.abs(P))
    if P_max > 1e10:
        # Scale P down to prevent numerical explosion
        scale_factor = 1e8 / P_max
        P = P * scale_factor
        print(f"[EKF-PROP] P overflow clamped: max={P_max:.2e} → scaled by {scale_factor:.2e}")
    
    # Build full Φ matrix (v3.9.7: 18D core)
    phi_full = np.eye(err_dim, dtype=float)
    phi_full[0:CORE_ERROR_DIM, 0:CORE_ERROR_DIM] = Phi
    
    # Build full Q matrix (v3.9.7: 18D core)
    q_full = np.zeros((err_dim, err_dim), dtype=float)
    q_full[0:CORE_ERROR_DIM, 0:CORE_ERROR_DIM] = Q
    
    # [TRIPWIRE] Validate Phi and Q matrices before propagation
    if not assert_finite("ekf_Phi", Phi, extra_info={"Phi_norm": np.linalg.norm(Phi)}):
        print(f"[EKF-PROP] CRITICAL: Phi contains inf/nan, using identity")
        phi_full = np.eye(err_dim, dtype=float)
        return P  # Early return - cannot propagate safely
    
    if not assert_finite("ekf_Q", Q, extra_info={"Q_norm": np.linalg.norm(Q)}):
        print(f"[EKF-PROP] CRITICAL: Q contains inf/nan, skipping propagation")
        return P  # Early return - cannot add process noise safely
    
    # Propagate: P_k+1 = Φ * P_k * Φ^T + Q with overflow protection
    # Suppress numpy warnings - we handle explicitly with tripwires
    with np.errstate(all='ignore'):
        try:
            p_new = phi_full @ P @ phi_full.T + q_full
        except Exception as e:
            print(f"[EKF-PROP] P propagation failed: {e}, using previous P")
            p_new = P
    
    # [TRIPWIRE] Check result validity after computation
    if not assert_finite("ekf_P_new", p_new, extra_info={
        "P_new_max": np.max(np.abs(p_new)),
        "P_new_trace": np.trace(p_new),
        "Phi_norm": np.linalg.norm(phi_full),
        "Q_norm": np.linalg.norm(q_full),
        "P_old_max": P_max
    }):
        print(f"[EKF-PROP] P propagation produced inf/nan, using previous P")
        p_new = P
    
    # Stage-aware conditioning path:
    # 1) cheap guard (finite + diag floor), 2) Cholesky guard, 3) eigen fallback only if needed.
    p_new = ensure_covariance_valid(
        p_new,
        label="EKF-Propagate",
        symmetrize=True,
        check_psd=True,
        min_eigenvalue=1e-7,
        max_value=getattr(conditioner, "covariance_max_value", 1e8) if conditioner is not None else 1e8,
        conditioner=conditioner,
        timestamp=timestamp,
        stage=stage,
    )
    
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


# State dimension constants
# v3.9.7: Added mag_bias (3D) for online hard iron estimation
CORE_NOMINAL_DIM = 19  # p(3) + v(3) + q(4) + bg(3) + ba(3) + mag_bias(3)
CORE_ERROR_DIM = 18    # δp(3) + δv(3) + δθ(3) + δbg(3) + δba(3) + δmag(3)
CLONE_NOMINAL_DIM = 7  # q(4) + p(3)
CLONE_ERROR_DIM = 6    # δθ(3) + δp(3)

# State indices (nominal)
IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_QUAT = slice(6, 10)
IDX_GYRO_BIAS = slice(10, 13)
IDX_ACCEL_BIAS = slice(13, 16)
IDX_MAG_BIAS = slice(16, 19)  # NEW

# Error state indices
IDX_ERR_POS = slice(0, 3)
IDX_ERR_VEL = slice(3, 6)
IDX_ERR_ROT = slice(6, 9)
IDX_ERR_GYRO_BIAS = slice(9, 12)
IDX_ERR_ACCEL_BIAS = slice(12, 15)
IDX_ERR_MAG_BIAS = slice(15, 18)  # NEW


class ExtendedKalmanFilter:
    """
    Error-State Kalman Filter (ESKF) for VIO - OpenVINS style.
    
    State vector layout (nominal state x):
      - Core IMU state (19 elements):  # UPDATED from 16
        * p (0:3): position [m]
        * v (3:6): velocity [m/s]
        * q (6:10): quaternion [w,x,y,z]
        * bg (10:13): gyro bias [rad/s]
        * ba (13:16): accel bias [m/s²]
        * mag_bias (16:19): magnetometer hard iron [normalized]  # NEW
      
      - Camera clones (7 elements each):
        * q_C (0:4): camera quaternion
        * p_C (4:7): camera position
    
    Error-state covariance P uses 3D rotation vector instead of 4D quaternion:
      - Core error state (18 elements):  # UPDATED from 15
        * δp (0:3): position error
        * δv (3:6): velocity error
        * δθ (6:9): rotation error (3D)
        * δbg (9:12): gyro bias error
        * δba (12:15): accel bias error
        * δmag (15:18): mag bias error  # NEW
      
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
        
        # Covariance health logging
        self._cov_health_csv = None
        self._cov_prev_pmax = None
        self._cov_prev_trace = None
        self.covariance_max_value = 1e8
        self.conditioning_cond_hard = 1e12
        self.conditioning_cond_hard_window = 8
        self.conditioning_projection_min_interval_steps = 64
        self._conditioning_cond_streak = 0
        self._conditioning_steps_since_projection = 10**9
        self._conditioning_event_csv = None
        self._conditioning_last_event = ""
        self._conditioning_last_action = ""
        self._conditioning_print_every = 500
        self._conditioning_stats = {
            "projection_count": 0,
            "reset_count": 0,
            "chol_fail_count": 0,
            "first_projection_time": float("nan"),
            "max_cond_seen": 0.0,
        }
        self.verbose_yaw_debug = False
        self._cov_growth_stats = defaultdict(lambda: {
            "samples": 0,
            "growth_events": 0,
            "large_events": 0,
            "max_pmax": 0.0
        })
    
    def enable_cov_health_logging(self, csv_path: str):
        """Enable covariance health CSV logging."""
        self._cov_health_csv = csv_path

    def enable_conditioning_event_logging(self, csv_path: str):
        """Enable conditioning event CSV logging."""
        self._conditioning_event_csv = csv_path

    def set_covariance_max_value(self, max_value: float):
        """Set adaptive absolute cap used by covariance conditioning."""
        try:
            value = float(max_value)
        except (TypeError, ValueError):
            return
        if np.isfinite(value) and value > 0:
            self.covariance_max_value = value

    def _update_conditioning_cond_streak(self, is_over: bool) -> int:
        """Track consecutive steps above hard condition-number threshold."""
        if is_over:
            self._conditioning_cond_streak += 1
        else:
            self._conditioning_cond_streak = 0
        return int(self._conditioning_cond_streak)

    def _conditioning_tick(self):
        """Advance conditioning-step counter."""
        self._conditioning_steps_since_projection += 1

    def _can_project_now(self) -> bool:
        """Rate-limit projection fallback to avoid projection flood."""
        return int(self._conditioning_steps_since_projection) >= int(
            max(1, self.conditioning_projection_min_interval_steps)
        )

    def _update_conditioning_max_cond(self, cond_value: float):
        """Update maximum condition number seen this run."""
        if cond_value is None:
            return
        try:
            cond_val = float(cond_value)
        except (TypeError, ValueError):
            return
        if np.isfinite(cond_val):
            self._conditioning_stats["max_cond_seen"] = max(
                float(self._conditioning_stats.get("max_cond_seen", 0.0)),
                cond_val
            )

    def _record_conditioning_action(self,
                                    timestamp: float,
                                    event: str,
                                    stage: str,
                                    p_min_eig: float,
                                    p_max: float,
                                    p_cond: float,
                                    action: str,
                                    notes: str = "",
                                    projected: bool = False,
                                    reset_applied: bool = False,
                                    chol_failed: bool = False):
        """Record conditioning action in counters and optional CSV."""
        if projected:
            self._conditioning_stats["projection_count"] = int(
                self._conditioning_stats.get("projection_count", 0)
            ) + 1
            self._conditioning_steps_since_projection = 0
            if not np.isfinite(float(self._conditioning_stats.get("first_projection_time", float("nan")))):
                if np.isfinite(timestamp):
                    self._conditioning_stats["first_projection_time"] = float(timestamp)
        if reset_applied:
            self._conditioning_stats["reset_count"] = int(
                self._conditioning_stats.get("reset_count", 0)
            ) + 1
        if chol_failed:
            self._conditioning_stats["chol_fail_count"] = int(
                self._conditioning_stats.get("chol_fail_count", 0)
            ) + 1

        self._update_conditioning_max_cond(p_cond)

        if self._conditioning_event_csv:
            try:
                with open(self._conditioning_event_csv, "a", newline="") as f:
                    f.write(
                        f"{timestamp:.6f},{event},{stage},{p_min_eig:.6e},"
                        f"{p_max:.6e},{p_cond:.6e},{action},{notes}\n"
                    )
            except Exception:
                pass

        projection_count = int(self._conditioning_stats.get("projection_count", 0))
        transition = (event != self._conditioning_last_event) or (action != self._conditioning_last_action)
        if transition or (projection_count > 0 and projection_count % int(max(1, self._conditioning_print_every)) == 0):
            print(
                f"[EKF-COND] t={timestamp:.3f} stage={stage} event={event} action={action} "
                f"p_max={p_max:.2e} p_cond={p_cond:.2e} note={notes}"
            )
            self._conditioning_last_event = str(event)
            self._conditioning_last_action = str(action)

    def get_conditioning_stats(self) -> dict:
        """Return runtime conditioning stats."""
        return {
            "projection_count": int(self._conditioning_stats.get("projection_count", 0)),
            "reset_count": int(self._conditioning_stats.get("reset_count", 0)),
            "chol_fail_count": int(self._conditioning_stats.get("chol_fail_count", 0)),
            "first_projection_time": float(self._conditioning_stats.get("first_projection_time", float("nan"))),
            "max_cond_seen": float(self._conditioning_stats.get("max_cond_seen", 0.0)),
        }
    
    def log_cov_health(self, update_type: str, timestamp: float = float('nan'),
                       stage: str = "post"):
        """Record covariance health sample and update per-type counters."""
        try:
            p = self.P
            p_trace = float(np.trace(p))
            p_max = float(np.max(np.abs(p)))
            try:
                eigvals = np.linalg.eigvalsh((p + p.T) / 2.0)
                p_min_eig = float(eigvals[0])
            except Exception:
                p_min_eig = float('nan')
            try:
                p_cond = float(np.linalg.cond(p))
            except Exception:
                p_cond = float('inf')
            
            growth_flag = 0
            if self._cov_prev_pmax is not None and self._cov_prev_pmax > 0:
                growth_flag = 1 if p_max > (1.05 * self._cov_prev_pmax) else 0
            large_flag = 1 if p_max > 1e6 else 0
            
            stats = self._cov_growth_stats[update_type]
            stats["samples"] += 1
            stats["growth_events"] += growth_flag
            stats["large_events"] += large_flag
            stats["max_pmax"] = max(stats["max_pmax"], p_max)
            
            if self._cov_health_csv:
                with open(self._cov_health_csv, "a", newline="") as f:
                    f.write(
                        f"{timestamp:.6f},{update_type},{stage},"
                        f"{p_trace:.6e},{p_max:.6e},{p_min_eig:.6e},{p_cond:.6e},"
                        f"{growth_flag},{large_flag}\n"
                    )
            
            self._cov_prev_pmax = p_max
            self._cov_prev_trace = p_trace
        except Exception:
            pass
    
    def get_cov_growth_summary(self):
        """Return per-update covariance growth summary sorted by large/growth count."""
        rows = []
        for update_type, s in self._cov_growth_stats.items():
            samples = max(1, int(s["samples"]))
            rows.append({
                "update_type": update_type,
                "samples": int(s["samples"]),
                "growth_events": int(s["growth_events"]),
                "large_events": int(s["large_events"]),
                "growth_rate": float(s["growth_events"]) / samples,
                "large_rate": float(s["large_events"]) / samples,
                "max_pmax": float(s["max_pmax"]),
            })
        rows.sort(key=lambda r: (r["large_events"], r["growth_events"], r["max_pmax"]), reverse=True)
        return rows
    
    def _compute_error_dim(self, nominal_dim: int) -> int:
        """
        Compute error-state dimension from nominal state dimension.
        
        v3.9.7: Updated for mag_bias
        Nominal: 19 + 7*N (quaternion = 4D, includes mag_bias)
        Error:   18 + 6*N (rotation = 3D, includes δmag_bias)
        
        Args:
            nominal_dim: Nominal state dimension
        
        Returns:
            Error-state dimension
        """
        if nominal_dim < CORE_NOMINAL_DIM:
            raise ValueError(f"Nominal dimension {nominal_dim} < {CORE_NOMINAL_DIM} (minimum core state)")
        
        num_clones = (nominal_dim - CORE_NOMINAL_DIM) // CLONE_NOMINAL_DIM
        if (nominal_dim - CORE_NOMINAL_DIM) % CLONE_NOMINAL_DIM != 0:
            raise ValueError(f"Invalid nominal dimension {nominal_dim}: not {CORE_NOMINAL_DIM}+7*N")
        
        return CORE_ERROR_DIM + CLONE_ERROR_DIM * num_clones
    
    def nominal_to_error_idx(self, nominal_idx: int) -> int:
        """
        Map nominal state index to error-state index.
        
        v3.9.7: Updated for mag_bias
        Nominal state:
          [p(0:3), v(3:6), q(6:10), bg(10:13), ba(13:16), mag(16:19), q_C1(19:23), p_C1(23:26), ...]
        
        Error state:
          [δp(0:3), δv(3:6), δθ(6:9), δbg(9:12), δba(12:15), δmag(15:18), δθ_C1(18:21), δp_C1(21:24), ...]
        
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
            if nominal_idx == 6:
                return 6  # w → δθ_x
            else:
                return -1  # x,y,z components don't map directly
        elif nominal_idx < 16:
            return nominal_idx - 1  # bg, ba: shift by 1 (due to quat 4→3)
        elif nominal_idx < 19:
            return nominal_idx - 1  # mag_bias: also shift by 1 (16,17,18 → 15,16,17)
        else:
            # Camera clones (start at 19 now, not 16)
            clone_offset = nominal_idx - CORE_NOMINAL_DIM  # 19
            clone_id = clone_offset // CLONE_NOMINAL_DIM   # 7
            within_clone = clone_offset % CLONE_NOMINAL_DIM
            
            if within_clone < 4:
                # Quaternion → rotation
                if within_clone == 0:
                    return CORE_ERROR_DIM + CLONE_ERROR_DIM * clone_id  # w → δθ_x
                else:
                    return -1
            else:
                # Position
                return CORE_ERROR_DIM + CLONE_ERROR_DIM * clone_id + (within_clone - 4) + 3

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
        self.SI = safe_matrix_inverse(self.S, damping=1e-9, method='cholesky')
        self.K = dot(PHT, self.SI)

        self.y = z - Hx(x, *hx_args)
        self.x = x + dot(self.K, self.y)

        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)
        
        # [TRIPWIRE] Check state and covariance after update
        if not assert_finite("state_after_update", self.x, extra_info={
            "innovation_norm": np.linalg.norm(self.y),
            "K_norm": np.linalg.norm(self.K)
        }):
            print(f"[EKF-UPDATE] CRITICAL: State contains NaN/inf after update!")
        
        if not check_covariance_psd(self.P, name="P_after_update"):
            print(f"[EKF-UPDATE] WARNING: Covariance not PSD after update!")

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
               residual=np.subtract, update_type: str = "EKF_UPDATE",
               timestamp: float = float('nan')):
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

        # NUMERICAL STABILITY: Symmetrize P before computing PHT
        # This prevents accumulation of asymmetry from floating-point errors
        self.P = (self.P + self.P.T) / 2.0
        
        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        
        # NUMERICAL STABILITY: Symmetrize S (innovation covariance)
        self.S = (self.S + self.S.T) / 2.0
        
        # NUMERICAL STABILITY: Use solve() instead of inv() for Kalman gain
        # K = P @ H.T @ inv(S) is rewritten as K = solve(S.T, (H @ P).T).T
        # This avoids explicit matrix inversion which is numerically unstable
        try:
            self.K = linalg.solve(self.S.T, PHT.T).T
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
        
        if self.verbose_yaw_debug and dx.shape[0] > 8 and abs(dx[8, 0]) > 0.001:
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
            min_eigenvalue=1e-7,
            max_value=getattr(self, "covariance_max_value", 1e8),
            conditioner=self,
            timestamp=timestamp,
            stage=update_type,
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
        self.log_cov_health(update_type=update_type, timestamp=timestamp, stage="post_update")
    
    def _apply_error_state_correction(self, dx):
        """
        Apply error-state correction to nominal state.
        
        v3.9.7: Updated for mag_bias
        Error state δx (18+6N): [δp, δv, δθ, δbg, δba, δmag, δθ_C1, δp_C1, ...]
        Nominal state x (19+7N): [p, v, q, bg, ba, mag, q_C1, p_C1, ...]
        
        Args:
            dx: Error-state correction (18+6N)
        """
        # Core state corrections
        dp = dx[IDX_ERR_POS, 0]           # Position error (0:3)
        dv = dx[IDX_ERR_VEL, 0]           # Velocity error (3:6)
        dtheta = dx[IDX_ERR_ROT, 0]       # Rotation error (6:9)
        dbg = dx[IDX_ERR_GYRO_BIAS, 0]    # Gyro bias error (9:12)
        dba = dx[IDX_ERR_ACCEL_BIAS, 0]   # Accel bias error (12:15)
        dmag = dx[IDX_ERR_MAG_BIAS, 0]    # Mag bias error (15:18) - NEW
        
        # Additive corrections for p, v, biases
        self.x[IDX_POS, 0] += dp
        self.x[IDX_VEL, 0] += dv
        self.x[IDX_GYRO_BIAS, 0] += dbg
        self.x[IDX_ACCEL_BIAS, 0] += dba
        self.x[IDX_MAG_BIAS, 0] += dmag   # NEW
        
        # Multiplicative correction for quaternion (manifold update)
        q_old = self.x[IDX_QUAT, 0]
        q_new = quat_boxplus(q_old, dtheta)
        self.x[IDX_QUAT, 0] = q_new
        
        # Camera clone corrections
        num_clones = (self.x.shape[0] - CORE_NOMINAL_DIM) // CLONE_NOMINAL_DIM
        for i in range(num_clones):
            # Error-state indices (18 + 6*i)
            err_base = CORE_ERROR_DIM + i * CLONE_ERROR_DIM
            dtheta_c = dx[err_base:err_base+3, 0]    # Clone rotation error
            dp_c = dx[err_base+3:err_base+6, 0]      # Clone position error
            
            # Nominal state indices (19 + 7*i)
            nom_base = CORE_NOMINAL_DIM + i * CLONE_NOMINAL_DIM
            
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
