#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPS Delayed Update with Stochastic Cloning

Handles VPS measurement latency using Camera State Augmentation.
When a camera frame is captured, the EKF state is "cloned" and stored.
When the VPS result arrives (50-200ms later), the update is applied
to the cloned state and propagated to the current state via cross-covariance.

Reference: Mourikis & Roumeliotis, "A Multi-State Constraint Kalman Filter
for Vision-aided Inertial Navigation" (MSCKF)

Author: VIO project
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class VPSStateClone:
    """
    Cloned EKF state at camera capture time.
    
    Used for delayed VPS updates via stochastic cloning.
    """
    t_capture: float              # Timestamp of camera capture
    x_clone: np.ndarray           # Nominal state at capture (19D or 16D)
    P_clone: np.ndarray           # Error covariance at capture
    cross_cov: np.ndarray         # Cross-covariance P(current, clone)
    image_id: str                 # Unique image identifier
    position_at_capture: np.ndarray  # Position [x, y, z] for debugging


class VPSDelayedUpdateManager:
    """
    Manages stochastic cloning for delayed VPS updates.
    
    Workflow:
    1. clone_state(): Called when camera frame is captured
    2. propagate_cross_covariance(): Called after each IMU propagation
    3. apply_delayed_update(): Called when VPS result arrives
    4. cleanup_expired_clones(): Called periodically to remove old clones
    
    Example:
        manager = VPSDelayedUpdateManager(max_clones=3, max_delay_sec=0.5)
        
        # At camera capture
        manager.clone_state(kf, t_capture=0.1, image_id="img_001")
        
        # During IMU propagation (in main loop)
        manager.propagate_cross_covariance(F_k)
        
        # When VPS result arrives
        success = manager.apply_delayed_update(
            kf, "img_001", vps_lat, vps_lon, R_vps, proj_cache, lat0, lon0
        )
    """
    
    def __init__(self, 
                 max_clones: int = 3,
                 max_delay_sec: float = 0.5,
                 error_state_dim: int = 18,
                 max_position_correction_m: float = 80.0,
                 max_velocity_correction_m_s: float = 25.0):
        """
        Initialize delayed update manager.
        
        Args:
            max_clones: Maximum number of clones to store
            max_delay_sec: Maximum delay before clone is expired
            error_state_dim: Dimension of error state (15 or 18)
        """
        self.max_clones = max_clones
        self.max_delay_sec = max_delay_sec
        self.error_state_dim = error_state_dim
        self.max_position_correction_m = float(max_position_correction_m)
        self.max_velocity_correction_m_s = float(max_velocity_correction_m_s)
        
        self.clones: Dict[str, VPSStateClone] = {}
        
        # Statistics
        self.stats = {
            'clones_created': 0,
            'clones_expired': 0,
            'updates_applied': 0,
            'updates_failed': 0,
            'updates_clamped': 0,
        }
    
    def clone_state(self, 
                    kf, 
                    t_capture: float, 
                    image_id: str) -> bool:
        """
        Clone current EKF state for delayed VPS update.
        
        Should be called immediately when camera frame is captured.
        
        Args:
            kf: ExtendedKalmanFilter instance
            t_capture: Timestamp of camera capture
            image_id: Unique identifier for this image
            
        Returns:
            True if clone was created successfully
        """
        # Remove oldest clone if at capacity
        if len(self.clones) >= self.max_clones:
            oldest_id = min(self.clones.keys(), 
                           key=lambda k: self.clones[k].t_capture)
            del self.clones[oldest_id]
            self.stats['clones_expired'] += 1
        
        # Extract core error-state covariance (without MSCKF clones)
        # For 18-state: first 18x18 of P
        # For 15-state: first 15x15 of P
        err_dim = min(self.error_state_dim, kf.P.shape[0])
        P_core = kf.P[:err_dim, :err_dim].copy()
        
        # Store clone
        clone = VPSStateClone(
            t_capture=t_capture,
            x_clone=kf.x.copy(),
            P_clone=P_core,
            cross_cov=P_core.copy(),  # Initially P(current, clone) = P
            image_id=image_id,
            position_at_capture=kf.x[0:3, 0].copy()
        )
        
        self.clones[image_id] = clone
        self.stats['clones_created'] += 1
        
        return True
    
    def propagate_cross_covariance(self, F_k: np.ndarray) -> None:
        """
        Update cross-covariance after IMU propagation step.
        
        Must be called after each EKF prediction step.
        
        The cross-covariance between current state and cloned state
        evolves as: P_cross_new = F_k @ P_cross
        
        Args:
            F_k: State transition Jacobian from IMU propagation (error state)
        """
        for clone in self.clones.values():
            # P(current, clone) = F_k @ P(previous, clone)
            # Cross-covariance is propagated forward with current state
            
            # Ensure dimension compatibility
            f_dim = min(F_k.shape[0], clone.cross_cov.shape[0])
            
            # Numerical safeguard: Check for inf/nan before multiplication
            if not np.all(np.isfinite(F_k[:f_dim, :f_dim])) or not np.all(np.isfinite(clone.cross_cov[:f_dim, :f_dim])):
                # Propagate with identity if corrupted (fail-safe)
                continue
                
            # Clamp input matrices to prevent overflow (reduced to 1e4 for safety)
            # 1e4 * 1e4 = 1e8, well within float32 limits
            F_k_safe = np.clip(F_k[:f_dim, :f_dim], -1e4, 1e4)
            cross_cov_safe = np.clip(clone.cross_cov[:f_dim, :f_dim], -1e4, 1e4)
            
            # Safe matrix multiplication with warning suppression
            with np.errstate(all='ignore'):
                try:
                    cross_cov_new = F_k_safe @ cross_cov_safe
                    
                    # Check result validity immediately
                    if not np.all(np.isfinite(cross_cov_new)):
                        cross_cov_new = cross_cov_safe
                except Exception:
                    cross_cov_new = cross_cov_safe  # Keep previous if failed
            
            # Numerical safeguard: clamp output
            cross_cov_new = np.clip(cross_cov_new, -1e10, 1e10)
            cross_cov_new = np.nan_to_num(cross_cov_new, nan=0.0, posinf=1e10, neginf=-1e10)
            
            clone.cross_cov[:f_dim, :f_dim] = cross_cov_new

    
    def apply_delayed_update(self,
                             kf,
                             image_id: str,
                             vps_lat: float,
                             vps_lon: float,
                             R_vps: np.ndarray,
                             proj_cache,
                             lat0: float,
                             lon0: float) -> Tuple[bool, Optional[float]]:
        """
        Apply delayed VPS update using stochastic cloning.
        
        This is the core algorithm:
        1. Find the cloned state for this image
        2. Compute innovation at cloned state's position
        3. Compute Kalman gain for clone
        4. Update clone state
        5. Propagate correction to current state via cross-covariance
        
        Args:
            kf: ExtendedKalmanFilter instance (will be modified)
            image_id: Image identifier from clone_state()
            vps_lat: VPS measured latitude
            vps_lon: VPS measured longitude
            R_vps: 2x2 measurement covariance matrix
            proj_cache: ProjectionCache for coordinate conversion
            lat0: Origin latitude
            lon0: Origin longitude
            
        Returns:
            (success, innovation_magnitude_m)
        """
        if image_id not in self.clones:
            self.stats['updates_failed'] += 1
            return False, None
        
        clone = self.clones[image_id]
        
        # Convert VPS lat/lon to local XY
        vps_xy = proj_cache.latlon_to_xy(vps_lat, vps_lon, lat0, lon0)
        
        # Measurement Jacobian (measures position)
        H = np.zeros((2, clone.P_clone.shape[0]))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        
        # Innovation: z - H @ x_clone
        xy_clone = clone.x_clone[0:2, 0]
        innovation = (vps_xy - xy_clone).reshape(-1, 1)
        innovation_mag = float(np.linalg.norm(innovation))
        
        # Innovation covariance (overflow-safe: only XY block is needed)
        try:
            r_vps_safe = np.array(R_vps, dtype=float, copy=True)
            r_vps_safe = 0.5 * (r_vps_safe + r_vps_safe.T)
            r_diag = np.clip(np.diag(r_vps_safe), 1e-6, 1e6)
            r_vps_safe = np.diag(r_diag)

            p_xy = np.array(clone.P_clone[0:2, 0:2], dtype=float, copy=True)
            if p_xy.shape != (2, 2) or not np.all(np.isfinite(p_xy)):
                raise ValueError("invalid clone covariance block")
            p_xy = 0.5 * (p_xy + p_xy.T)
            p_xy = np.clip(p_xy, -1e8, 1e8)
            eigvals, eigvecs = np.linalg.eigh(p_xy)
            eigvals = np.clip(eigvals, 1e-9, 1e8)
            p_xy = eigvecs @ np.diag(eigvals) @ eigvecs.T

            S = p_xy + r_vps_safe
            S = 0.5 * (S + S.T)
            S[np.diag_indices(2)] += 1e-9
            if not np.all(np.isfinite(S)):
                raise ValueError("non-finite S")

            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S, rcond=1e-9)
            if not np.all(np.isfinite(S_inv)):
                raise ValueError("non-finite S_inv")
        except Exception:
            self.stats['updates_failed'] += 1
            return False, innovation_mag
        
        # === KEY: Propagate correction to current state ===
        # dx_current = P_cross @ P_clone_inv @ dx_clone
        # 
        # More numerically stable form:
        # dx_current = P_cross @ H.T @ S_inv @ innovation
        
        # Cross-covariance contribution (bounded to avoid numerical explosion)
        cross_cov_safe = np.array(clone.cross_cov, dtype=float, copy=True)
        cross_cov_safe = np.nan_to_num(cross_cov_safe, nan=0.0, posinf=1e8, neginf=-1e8)
        cross_cov_safe = np.clip(cross_cov_safe, -1e8, 1e8)
        dx_current = cross_cov_safe @ H.T @ S_inv @ innovation
        dx_current = np.nan_to_num(dx_current, nan=0.0, posinf=1e4, neginf=-1e4)
        dx_current = np.clip(dx_current, -1e4, 1e4)

        # Guard against large delayed-correction jumps that can destabilize state.
        pos_corr_norm = float(np.linalg.norm(dx_current[0:2, 0])) if dx_current.shape[0] >= 2 else 0.0
        vel_corr_norm = float(np.linalg.norm(dx_current[3:5, 0])) if dx_current.shape[0] >= 5 else 0.0
        corr_scale = 1.0
        if np.isfinite(pos_corr_norm) and self.max_position_correction_m > 1e-6:
            corr_scale = min(corr_scale, self.max_position_correction_m / max(pos_corr_norm, 1e-9))
        if np.isfinite(vel_corr_norm) and self.max_velocity_correction_m_s > 1e-6:
            corr_scale = min(corr_scale, self.max_velocity_correction_m_s / max(vel_corr_norm, 1e-9))
        if corr_scale < 0.05:
            self.stats['updates_failed'] += 1
            del self.clones[image_id]
            return False, innovation_mag
        if corr_scale < 1.0:
            dx_current *= float(corr_scale)
            self.stats['updates_clamped'] += 1
        
        # Apply correction to current state
        # Position
        kf.x[0, 0] += dx_current[0, 0]
        kf.x[1, 0] += dx_current[1, 0]
        if dx_current.shape[0] > 2:
            kf.x[2, 0] += dx_current[2, 0]
        
        # Velocity
        if dx_current.shape[0] >= 6:
            kf.x[3, 0] += dx_current[3, 0]
            kf.x[4, 0] += dx_current[4, 0]
            kf.x[5, 0] += dx_current[5, 0]
        
        # Update covariance
        # P_current -= P_cross @ H.T @ S_inv @ H @ P_cross.T
        err_dim = min(clone.cross_cov.shape[0], kf.P.shape[0])
        P_cross = cross_cov_safe[:err_dim, :]
        HSH = H.T @ S_inv @ H
        with np.errstate(all='ignore'):
            dP = P_cross @ HSH @ P_cross.T
        if not np.all(np.isfinite(dP)):
            dP = np.zeros((err_dim, err_dim), dtype=float)
        dP = np.nan_to_num(dP, nan=0.0, posinf=1e8, neginf=-1e8)
        dP = np.clip(dP, -1e8, 1e8)
        kf.P[:err_dim, :err_dim] -= dP

        # Ensure symmetry + finite + PSD floor on updated core block
        kf.P = 0.5 * (kf.P + kf.P.T)
        kf.P = np.nan_to_num(kf.P, nan=0.0, posinf=1e8, neginf=-1e8)
        core = kf.P[:err_dim, :err_dim]
        core = np.clip(0.5 * (core + core.T), -1e8, 1e8)
        try:
            evals_core, evecs_core = np.linalg.eigh(core)
            evals_core = np.clip(evals_core, 1e-9, 1e8)
            with np.errstate(all='ignore'):
                core_psd = evecs_core @ np.diag(evals_core) @ evecs_core.T
            if not np.all(np.isfinite(core_psd)):
                raise ValueError("non-finite core_psd")
            kf.P[:err_dim, :err_dim] = core_psd
        except Exception:
            diag = np.diag(core) if core.ndim == 2 else np.array([], dtype=float)
            diag = np.nan_to_num(diag, nan=1e-6, posinf=1e8, neginf=1e-6)
            diag = np.clip(diag, 1e-9, 1e8)
            if diag.size != err_dim:
                diag = np.full((err_dim,), 1e-3, dtype=float)
            kf.P[:err_dim, :err_dim] = np.diag(diag)
        
        # Remove used clone
        del self.clones[image_id]
        self.stats['updates_applied'] += 1
        
        return True, innovation_mag
    
    def cleanup_expired_clones(self, t_now: float) -> int:
        """
        Remove clones older than max_delay_sec.
        
        Args:
            t_now: Current timestamp
            
        Returns:
            Number of clones removed
        """
        expired = [
            img_id for img_id, clone in self.clones.items()
            if (t_now - clone.t_capture) > self.max_delay_sec
        ]
        
        for img_id in expired:
            del self.clones[img_id]
            self.stats['clones_expired'] += 1
        
        return len(expired)
    
    def has_pending_clone(self, image_id: str) -> bool:
        """Check if a clone exists for the given image."""
        return image_id in self.clones
    
    def get_clone_age(self, image_id: str, t_now: float) -> Optional[float]:
        """Get age of clone in seconds."""
        if image_id not in self.clones:
            return None
        return t_now - self.clones[image_id].t_capture
    
    def print_stats(self) -> None:
        """Print statistics."""
        print(f"[VPSDelayed] Stats: "
              f"created={self.stats['clones_created']}, "
              f"applied={self.stats['updates_applied']}, "
              f"expired={self.stats['clones_expired']}, "
              f"failed={self.stats['updates_failed']}")


def test_delayed_update():
    """Test the delayed update manager with mock EKF."""
    print("=" * 60)
    print("Testing VPSDelayedUpdateManager")
    print("=" * 60)
    
    # Mock EKF
    class MockEKF:
        def __init__(self):
            self.x = np.zeros((19, 1))
            self.x[0, 0] = 100.0  # x position
            self.x[1, 0] = 200.0  # y position
            self.P = np.eye(18) * 1.0
    
    # Mock ProjectionCache
    class MockProjCache:
        def latlon_to_xy(self, lat, lon, lat0, lon0):
            # Simple linear approximation
            dx = (lon - lon0) * 111320 * np.cos(np.radians(lat0))
            dy = (lat - lat0) * 111320
            return np.array([dx, dy])
    
    kf = MockEKF()
    proj_cache = MockProjCache()
    manager = VPSDelayedUpdateManager(max_clones=3, max_delay_sec=0.5)
    
    # Test 1: Clone state
    print("\n[Test 1] Cloning state...")
    t_capture = 0.0
    manager.clone_state(kf, t_capture, "img_001")
    print(f"  Clone created: {manager.has_pending_clone('img_001')}")
    assert manager.has_pending_clone("img_001")
    
    # Test 2: Propagate cross-covariance
    print("\n[Test 2] Propagating cross-covariance...")
    F_k = np.eye(18)
    F_k[0, 3] = 0.01  # Position += velocity * dt
    F_k[1, 4] = 0.01
    
    for i in range(5):
        manager.propagate_cross_covariance(F_k)
        # Simulate state change
        kf.x[0, 0] += 1.0  # Move 1m in x
        kf.x[1, 0] += 0.5  # Move 0.5m in y
    
    print(f"  State moved from (100, 200) to ({kf.x[0,0]:.1f}, {kf.x[1,0]:.1f})")
    
    # Test 3: Apply delayed update
    print("\n[Test 3] Applying delayed VPS update...")
    
    # Suppose VPS says true position at t_capture was (100.5, 200.3)
    # But we recorded (100, 200). Innovation = (0.5, 0.3)
    # Convert to lat/lon (mock)
    lat0, lon0 = 45.0, -75.0
    # Inverse mock: lat = lat0 + dy/111320, lon = lon0 + dx/(111320*cos(lat0))
    vps_lat = lat0 + 200.3 / 111320
    vps_lon = lon0 + 100.5 / (111320 * np.cos(np.radians(lat0)))
    
    R_vps = np.diag([0.5**2, 0.5**2])  # 0.5m uncertainty
    
    print(f"  State before update: ({kf.x[0,0]:.3f}, {kf.x[1,0]:.3f})")
    
    success, innov_mag = manager.apply_delayed_update(
        kf, "img_001", vps_lat, vps_lon, R_vps, proj_cache, lat0, lon0
    )
    
    print(f"  Update success: {success}")
    print(f"  Innovation magnitude: {innov_mag:.3f} m")
    print(f"  State after update: ({kf.x[0,0]:.3f}, {kf.x[1,0]:.3f})")
    
    assert success
    
    # Test 4: Cleanup expired clones
    print("\n[Test 4] Testing clone expiration...")
    manager.clone_state(kf, 0.0, "img_002")
    manager.clone_state(kf, 0.1, "img_003")
    print(f"  Clones before cleanup: {len(manager.clones)}")
    
    removed = manager.cleanup_expired_clones(t_now=1.0)  # 1 second later
    print(f"  Clones removed: {removed}")
    print(f"  Clones after cleanup: {len(manager.clones)}")
    
    manager.print_stats()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_delayed_update()
