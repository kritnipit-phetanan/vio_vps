#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical Validation and Tripwire Module
=========================================

Provides defensive checks to catch NaN/inf propagation at source.
Dumps comprehensive diagnostic information when numerical issues detected.
"""

import numpy as np


def assert_finite(name, M, t=None, extra_info=None, raise_on_fail=False):
    """
    Tripwire: Check matrix/vector for inf/nan and dump diagnostics if found.
    
    Parameters:
    -----------
    name : str
        Descriptive name of the quantity being checked
    M : np.ndarray
        Matrix or vector to validate
    t : float, optional
        Timestamp (for logging context)
    extra_info : dict, optional
        Additional diagnostic information to dump
    raise_on_fail : bool
        If True, raises ValueError on failure. If False, only prints warning.
    
    Returns:
    --------
    bool : True if finite, False if inf/nan detected
    """
    if M is None:
        print(f"[TRIPWIRE] {name}: is None!")
        return False
    
    is_finite = np.all(np.isfinite(M))
    
    if not is_finite:
        # Dump comprehensive diagnostics
        print(f"\n{'='*70}")
        print(f"[TRIPWIRE] ⚠️  NaN/inf DETECTED in {name}")
        print(f"{'='*70}")
        
        if t is not None:
            print(f"Timestamp: {t:.6f}")
        
        # Matrix statistics
        print(f"\nMatrix shape: {M.shape}")
        print(f"Has NaN: {np.any(np.isnan(M))}")
        print(f"Has inf: {np.any(np.isinf(M))}")
        
        if M.size <= 100:  # Print small matrices fully
            print(f"\nFull matrix:\n{M}")
        else:  # Print summary for large matrices
            print(f"\nMatrix stats:")
            print(f"  min: {np.nanmin(M) if not np.all(np.isnan(M)) else 'all NaN'}")
            print(f"  max: {np.nanmax(M) if not np.all(np.isnan(M)) else 'all NaN'}")
            print(f"  mean: {np.nanmean(M) if not np.all(np.isnan(M)) else 'all NaN'}")
        
        # Find exact locations of inf/nan
        if np.any(np.isnan(M)):
            nan_locs = np.argwhere(np.isnan(M))
            print(f"\nNaN locations (first 10): {nan_locs[:10].tolist()}")
        
        if np.any(np.isinf(M)):
            inf_locs = np.argwhere(np.isinf(M))
            print(f"Inf locations (first 10): {inf_locs[:10].tolist()}")
        
        # Extra diagnostic info
        if extra_info:
            print(f"\nAdditional context:")
            for key, val in extra_info.items():
                if isinstance(val, np.ndarray):
                    if val.size <= 10:
                        print(f"  {key}: {val.ravel()}")
                    else:
                        print(f"  {key}: shape={val.shape}, norm={np.linalg.norm(val):.6e}")
                else:
                    print(f"  {key}: {val}")
        
        print(f"{'='*70}\n")
        
        if raise_on_fail:
            raise ValueError(f"NaN/inf detected in {name}")
        
        return False
    
    return True


def check_quaternion(q, name="quaternion", normalize=True, t=None):
    """
    Validate quaternion and optionally normalize.
    
    Parameters:
    -----------
    q : np.ndarray
        Quaternion [w, x, y, z]
    name : str
        Descriptive name for logging
    normalize : bool
        If True, normalize the quaternion
    t : float, optional
        Timestamp for logging
    
    Returns:
    --------
    q_out : np.ndarray
        Validated (and possibly normalized) quaternion
    is_valid : bool
        True if quaternion is valid
    """
    # Check for NaN/inf
    if not assert_finite(name, q, t=t):
        return q, False
    
    # Check norm
    q_norm = np.linalg.norm(q)
    
    if q_norm < 1e-8:
        print(f"[TRIPWIRE] {name}: norm near zero ({q_norm:.6e}) at t={t}")
        return q, False
    
    # Check if norm is far from 1 (unnormalized)
    if abs(q_norm - 1.0) > 0.1:
        print(f"[TRIPWIRE] {name}: norm far from 1 ({q_norm:.6f}) at t={t}")
    
    # Normalize if requested
    if normalize:
        q_out = q / q_norm
        return q_out, True
    
    return q, True


def check_covariance_psd(P, name="covariance", min_eigenvalue=1e-12, t=None):
    """
    Validate covariance matrix is positive semi-definite.
    
    Parameters:
    -----------
    P : np.ndarray
        Covariance matrix
    name : str
        Descriptive name for logging
    min_eigenvalue : float
        Minimum acceptable eigenvalue
    t : float, optional
        Timestamp for logging
    
    Returns:
    --------
    is_valid : bool
        True if PSD, False otherwise
    """
    # Check for NaN/inf
    if not assert_finite(name, P, t=t):
        return False
    
    # Check symmetry
    if not np.allclose(P, P.T, rtol=1e-5):
        asymmetry = np.max(np.abs(P - P.T))
        print(f"[TRIPWIRE] {name}: not symmetric (max diff={asymmetry:.6e}) at t={t}")
        return False
    
    # Check eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(P)
        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)
        
        if min_eig < -min_eigenvalue:
            print(f"[TRIPWIRE] {name}: negative eigenvalue ({min_eig:.6e}) at t={t}")
            print(f"  Eigenvalue range: [{min_eig:.6e}, {max_eig:.6e}]")
            return False
        
        # Check condition number
        if max_eig > 1e-12:
            cond = max_eig / max(abs(min_eig), 1e-12)
            if cond > 1e12:
                print(f"[TRIPWIRE] {name}: ill-conditioned (cond={cond:.3e}) at t={t}")
    
    except np.linalg.LinAlgError:
        print(f"[TRIPWIRE] {name}: eigenvalue computation failed at t={t}")
        return False
    
    return True


def check_state_validity(x, t=None, check_quaternion_norm=True):
    """
    Comprehensive state vector validation.
    
    Expected state layout (VIO):
    x[0:3]   = position
    x[3:6]   = velocity  
    x[6:10]  = quaternion [w,x,y,z]
    x[10:13] = gyro bias
    x[13:16] = accel bias
    
    Parameters:
    -----------
    x : np.ndarray
        State vector
    t : float, optional
        Timestamp
    check_quaternion_norm : bool
        If True, checks quaternion normalization
    
    Returns:
    --------
    is_valid : bool
    """
    # Check entire state for NaN/inf
    if not assert_finite("state_x", x, t=t):
        return False
    
    # Check position (reasonable bounds)
    p = x[0:3, 0] if x.ndim == 2 else x[0:3]
    if np.linalg.norm(p) > 1e6:
        print(f"[TRIPWIRE] Position magnitude unreasonable: {np.linalg.norm(p):.3e} at t={t}")
        return False
    
    # Check velocity (reasonable bounds)
    v = x[3:6, 0] if x.ndim == 2 else x[3:6]
    if np.linalg.norm(v) > 500:  # 500 m/s = Mach 1.5
        print(f"[TRIPWIRE] Velocity unreasonable: {np.linalg.norm(v):.3f} m/s at t={t}")
        return False
    
    # Check quaternion
    if check_quaternion_norm and x.shape[0] >= 10:
        q = x[6:10, 0] if x.ndim == 2 else x[6:10]
        q_norm = np.linalg.norm(q)
        if abs(q_norm - 1.0) > 0.01:
            print(f"[TRIPWIRE] Quaternion norm = {q_norm:.6f} (should be 1.0) at t={t}")
            return False
    
    return True


def dump_imu_sample(w, a, dt, t=None):
    """
    Dump IMU sample for diagnostics.
    
    Parameters:
    -----------
    w : np.ndarray
        Angular velocity [rad/s]
    a : np.ndarray
        Specific force [m/s²]
    dt : float
        Time step
    t : float, optional
        Timestamp
    """
    print(f"\n[IMU Sample] t={t}")
    print(f"  ω: {w.ravel()}")
    print(f"  |ω|: {np.linalg.norm(w):.6f} rad/s")
    print(f"  a: {a.ravel()}")
    print(f"  |a|: {np.linalg.norm(a):.6f} m/s²")
    print(f"  dt: {dt:.6f} s")


def check_file_being_used():
    """
    Print which files are actually being imported at runtime.
    Use this to verify you're not running old cached versions.
    """
    import vio.propagation as prop
    import vio.imu_preintegration as preint
    import vio.ekf as ekf
    
    print(f"\n{'='*70}")
    print("[FILE CHECK] Modules being used:")
    print(f"{'='*70}")
    print(f"propagation:       {prop.__file__}")
    print(f"imu_preintegration: {preint.__file__}")
    print(f"ekf:               {ekf.__file__}")
    print(f"{'='*70}\n")
