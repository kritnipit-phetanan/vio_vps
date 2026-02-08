#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIO Output Utilities Module

Handles logging, debug output, and visualization for VIO system.
Includes CSV writers, keyframe visualization, and error statistics.

Author: VIO project
"""

import os
import numpy as np
from typing import Optional, Dict, List, Any, Tuple, Tuple


# =============================================================================
# Ground Truth Error Computation
# =============================================================================

def get_ground_truth_error(t: float, kf, ppk_trajectory, flight_log_df, 
                           lat0: float, lon0: float, proj_cache,
                           error_type: str = 'position') -> Tuple:
    """
    Get ground truth error for NEES calculation.
    
    Args:
        t: Current timestamp
        kf: Kalman filter (for state and covariance)
        ppk_trajectory: PPK trajectory DataFrame (higher priority)
        flight_log_df: Flight log DataFrame (fallback)
        lat0, lon0: Reference origin for coordinate conversion
        proj_cache: ProjectionCache instance for coordinate conversion
        error_type: 'position', 'velocity', or 'both'
        
    Returns:
        Tuple of (state_error, state_cov) or (None, None) if unavailable
        
    Note:
        Returns (None, None) gracefully if no ground truth is available.
        This ensures the system continues to work without GT data.
    """
    gt_df = ppk_trajectory if ppk_trajectory is not None else flight_log_df
    if gt_df is None or len(gt_df) == 0:
        return None, None
    
    try:
        gt_idx = np.argmin(np.abs(gt_df['stamp_log'].values - t))
        gt_row = gt_df.iloc[gt_idx]
        use_ppk = ppk_trajectory is not None
        
        if error_type in ['position', 'both']:
            # Position error
            if use_ppk:
                gt_lat, gt_lon, gt_alt = gt_row['lat'], gt_row['lon'], gt_row['height']
            else:
                gt_lat, gt_lon = gt_row['lat_dd'], gt_row['lon_dd']
                gt_alt = gt_row['altitude_MSL_m']
            
            gt_E, gt_N = proj_cache.latlon_to_xy(gt_lat, gt_lon, lat0, lon0)
            gt_pos = np.array([[gt_E], [gt_N], [gt_alt]])
            vio_pos = kf.x[0:3, 0:1]
            pos_error = gt_pos - vio_pos
            pos_cov = kf.P[0:3, 0:3]
            
            if error_type == 'position':
                return pos_error, pos_cov
        
        if error_type in ['velocity', 'both']:
            # Velocity error (from finite difference)
            if gt_idx > 0 and gt_idx < len(gt_df) - 1:
                gt_row_prev = gt_df.iloc[gt_idx - 1]
                gt_row_next = gt_df.iloc[gt_idx + 1]
                dt = gt_row_next['stamp_log'] - gt_row_prev['stamp_log']
                
                if dt > 0.01:
                    if use_ppk:
                        gt_E_prev, gt_N_prev = proj_cache.latlon_to_xy(gt_row_prev['lat'], gt_row_prev['lon'], lat0, lon0)
                        gt_E_next, gt_N_next = proj_cache.latlon_to_xy(gt_row_next['lat'], gt_row_next['lon'], lat0, lon0)
                        gt_U_prev, gt_U_next = gt_row_prev['height'], gt_row_next['height']
                    else:
                        gt_E_prev, gt_N_prev = proj_cache.latlon_to_xy(gt_row_prev['lat_dd'], gt_row_prev['lon_dd'], lat0, lon0)
                        gt_E_next, gt_N_next = proj_cache.latlon_to_xy(gt_row_next['lat_dd'], gt_row_next['lon_dd'], lat0, lon0)
                        gt_U_prev, gt_U_next = gt_row_prev['altitude_MSL_m'], gt_row_next['altitude_MSL_m']
                    
                    gt_vel = np.array([[(gt_E_next - gt_E_prev) / dt],
                                      [(gt_N_next - gt_N_prev) / dt],
                                      [(gt_U_next - gt_U_prev) / dt]])
                    vio_vel = kf.x[3:6, 0:1]
                    vel_error = gt_vel - vio_vel
                    vel_cov = kf.P[3:6, 3:6]
                    
                    if error_type == 'velocity':
                        return vel_error, vel_cov
                    elif error_type == 'both':
                        combined_error = np.vstack([pos_error, vel_error])
                        combined_cov = kf.P[0:6, 0:6]
                        return combined_error, combined_cov
    except Exception:
        pass
    
    return None, None


# =============================================================================
# Debug Logging Functions
# =============================================================================

def log_measurement_update(residual_csv: Optional[str], t: float, frame: int,
                           update_type: str, innovation: np.ndarray,
                           mahalanobis_dist: float, chi2_threshold: float,
                           accepted: bool, s_matrix: Optional[np.ndarray] = None,
                           p_prior: Optional[np.ndarray] = None,
                           state_error: Optional[np.ndarray] = None,
                           state_cov: Optional[np.ndarray] = None):
    """
    Log measurement update residuals and statistics for debugging.
    
    Args:
        residual_csv: Path to residual debug CSV file
        t: Current timestamp
        frame: Current VIO frame index (-1 if not applicable)
        update_type: Type of update ('VPS', 'VIO_VEL', 'MAG', 'DEM', 'MSCKF', 'ZUPT')
        innovation: Innovation vector (residual) - can be 1D or 2D (Nx1)
        mahalanobis_dist: Mahalanobis distance (for gating)
        chi2_threshold: Chi-square threshold for acceptance
        accepted: Whether update was accepted
        s_matrix: Innovation covariance (optional, for NIS computation)
        p_prior: Prior covariance (optional)
        state_error: Ground truth error (x_true - x_est) for NEES calculation (optional)
        state_cov: State covariance matrix P for NEES calculation (optional)
    
    NEES (Normalized Estimation Error Squared):
        NEES = (x_true - x_est)ᵀ P⁻¹ (x_true - x_est)
        
        - Checks filter consistency: Does uncertainty P match actual error?
        - Should follow chi-square distribution with n degrees of freedom
        - If NEES >> n: Filter OVERCONFIDENT (P too small, underestimates uncertainty)
        - If NEES << n: Filter UNDERCONFIDENT (P too large, overestimates uncertainty)
        - Ideal: NEES ≈ n (e.g., 3.0 for 3D velocity, within [0.35, 9.35] at 95% CI)
        
        To enable NEES logging:
            1. Load ground truth trajectory (PPK/GPS)
            2. Compute state_error = ground_truth - vio_estimate at timestamp
            3. Extract relevant state_cov submatrix (e.g., P[3:6,3:6] for velocity)
            4. Pass both to log_measurement_update()
        
        Example:
            # In main_loop.py after VIO_VEL update
            gt_vel = get_ground_truth_velocity(t)  # [vx, vy, vz]
            vio_vel = kf.x[3:6, 0]
            state_error = (gt_vel - vio_vel).reshape(-1, 1)
            state_cov = kf.P[3:6, 3:6]
            log_measurement_update(..., state_error=state_error, state_cov=state_cov)
    """
    if residual_csv is None:
        return
    
    try:
        # Compute NIS (Normalized Innovation Squared)
        nis = float('nan')
        if s_matrix is not None:
            try:
                s_inv = np.linalg.inv(s_matrix)
                nis = float(innovation.T @ s_inv @ innovation)
            except np.linalg.LinAlgError:
                pass
        
        # Flatten innovation to 1D array
        innov_flat = innovation.flatten()
        
        # Map innovation to correct axes based on update type
        # DEM: altitude update → Z-axis
        # MAG: yaw update → Z-axis (rotation around Z)
        # VIO_VEL: velocity (vx,vy,vz) → X,Y,Z axes
        # VPS: position (x,y,z) → X,Y,Z axes
        innov_x = float('nan')
        innov_y = float('nan')
        innov_z = float('nan')
        
        if len(innov_flat) == 1:
            # 1D measurement
            if update_type in ['DEM', 'MAG', 'ZUPT']:
                # Altitude/Yaw → Z-axis
                innov_z = float(innov_flat[0])
            else:
                # Unknown 1D update → default to X
                innov_x = float(innov_flat[0])
        elif len(innov_flat) == 2:
            # 2D measurement (VPS)
            innov_x = float(innov_flat[0])
            innov_y = float(innov_flat[1])
        elif len(innov_flat) >= 3:
            # 3D measurement (VIO_VEL)
            innov_x = float(innov_flat[0])
            innov_y = float(innov_flat[1])
            innov_z = float(innov_flat[2])
        
        # Compute NEES (Normalized Estimation Error Squared) if ground truth available
        # NEES = (x_true - x_est)ᵀ P⁻¹ (x_true - x_est)
        # For consistency check: NEES should follow chi-square distribution
        nees = float('nan')
        # v2.9.9.11: Skip NEES during initialization (frame < 100) to avoid NaN from unaligned GT
        if frame >= 100 and state_error is not None and state_cov is not None:
            try:
                # Invert covariance matrix
                P_inv = np.linalg.inv(state_cov)
                # v2.9.9.10: FIX NaN issue - ensure correct matrix dimensions
                # state_error might be (3,1) or (3,), need to flatten for proper multiplication
                err_flat = state_error.flatten()  # Shape (n,)
                # Compute NEES: scalar = (n,) @ (n,n) @ (n,) 
                nees = float(err_flat @ P_inv @ err_flat)
            except (np.linalg.LinAlgError, ValueError):
                nees = float('nan')
        
        with open(residual_csv, "a", newline="") as f:
            f.write(f"{t:.6f},{frame},{update_type},{innov_x:.6f},{innov_y:.6f},"
                    f"{innov_z:.6f},{mahalanobis_dist:.6f},{chi2_threshold:.6f},"
                    f"{int(accepted)},{nis:.6f},{nees:.6f}\n")
    except Exception:
        pass


def log_fej_consistency(fej_csv: Optional[str], t: float, frame: int,
                        cam_states: List[Dict], kf: Any):
    """
    Log FEJ consistency metrics: compare FEJ linearization points vs. current state.
    
    Args:
        fej_csv: CSV file path for FEJ consistency logging
        t: Current timestamp
        frame: Current VIO frame index
        cam_states: List of camera clone states with FEJ data
        kf: Extended Kalman Filter with current state
    """
    if fej_csv is None or not cam_states:
        return
    
    # Current bias estimates
    bg_current = kf.x[10:13, 0]
    ba_current = kf.x[13:16, 0]
    
    for i, cs in enumerate(cam_states):
        if 'q_fej' not in cs or 'p_fej' not in cs:
            continue
        
        q_fej = cs['q_fej']
        p_fej = cs['p_fej']
        
        # Current state
        q_idx = cs['q_idx']
        p_idx = cs['p_idx']
        q_current = kf.x[q_idx:q_idx+4, 0]
        p_current = kf.x[p_idx:p_idx+3, 0]
        
        # Position drift
        pos_drift = np.linalg.norm(p_current - p_fej)
        
        # Rotation drift (angle between quaternions)
        q_fej_inv = np.array([q_fej[0], -q_fej[1], -q_fej[2], -q_fej[3]])
        q_rel = _quat_mul(q_fej_inv.reshape(4, 1), q_current.reshape(4, 1)).reshape(4,)
        w_rel = np.clip(q_rel[0], -1.0, 1.0)
        rot_drift_rad = 2.0 * np.arccos(abs(w_rel))
        rot_drift_deg = np.rad2deg(rot_drift_rad)
        
        # Bias drift
        if 'bg_fej' in cs and 'ba_fej' in cs:
            bg_drift = np.linalg.norm(bg_current - cs['bg_fej'])
            ba_drift = np.linalg.norm(ba_current - cs['ba_fej'])
        else:
            bg_drift = np.nan
            ba_drift = np.nan
        
        try:
            with open(fej_csv, "a", newline="") as f:
                f.write(f"{t:.6f},{frame},{i},{pos_drift:.6f},{rot_drift_deg:.6f},"
                        f"{bg_drift:.6e},{ba_drift:.6e}\n")
        except Exception:
            pass


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication q1 * q2 where q = [w,x,y,z]."""
    w1, x1, y1, z1 = q1.flatten()
    w2, x2, y2, z2 = q2.flatten()
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]).reshape(4, 1)


# =============================================================================
# Keyframe Visualization
# =============================================================================

def save_keyframe_image_with_overlay(image: np.ndarray, features: Optional[np.ndarray],
                                     inliers: Optional[np.ndarray],
                                     reprojections: Optional[np.ndarray],
                                     output_path: str, frame_id: int,
                                     tracking_stats: Optional[Dict] = None):
    """
    Save keyframe image with feature tracking and reprojection overlays.
    
    Args:
        image: Input image (grayscale or color)
        features: List of tracked feature points (Nx2 array)
        inliers: Boolean mask of inlier features
        reprojections: Reprojected feature positions (Nx2 array, or None)
        output_path: Path to save annotated image
        frame_id: Frame index for labeling
        tracking_stats: Dict with tracking statistics (optional)
    """
    try:
        import cv2
        
        # Convert grayscale to BGR
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Draw all features (green)
        if features is not None and len(features) > 0:
            for pt in features:
                cv2.circle(vis, tuple(pt.astype(int)), 3, (0, 255, 0), 1)
        
        # Draw inliers (blue, thicker)
        if features is not None and inliers is not None and len(features) > 0:
            inlier_pts = features[inliers]
            for pt in inlier_pts:
                cv2.circle(vis, tuple(pt.astype(int)), 4, (255, 0, 0), 2)
        
        # Draw reprojection errors (red lines)
        if reprojections is not None and features is not None and len(features) > 0:
            for feat, reproj in zip(features, reprojections):
                if not np.any(np.isnan(reproj)):
                    pt1 = tuple(feat.astype(int))
                    pt2 = tuple(reproj.astype(int))
                    cv2.line(vis, pt1, pt2, (0, 0, 255), 1)
                    cv2.circle(vis, pt2, 2, (0, 0, 255), -1)
        
        # Text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        cv2.putText(vis, f"Frame: {frame_id}", (10, y_offset), font, 0.7, (255, 255, 255), 2)
        
        if tracking_stats:
            y_offset += 30
            cv2.putText(vis, f"Features: {tracking_stats.get('num_features', 0)}",
                        (10, y_offset), font, 0.6, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(vis, f"Inliers: {tracking_stats.get('num_inliers', 0)}",
                        (10, y_offset), font, 0.6, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(vis, f"Parallax: {tracking_stats.get('parallax_px', 0):.1f}px",
                        (10, y_offset), font, 0.6, (255, 255, 255), 1)
        
        # Legend
        y_offset = vis.shape[0] - 60
        cv2.circle(vis, (20, y_offset), 3, (0, 255, 0), 1)
        cv2.putText(vis, "Tracked", (30, y_offset + 5), font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.circle(vis, (20, y_offset), 4, (255, 0, 0), 2)
        cv2.putText(vis, "Inlier", (30, y_offset + 5), font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.line(vis, (15, y_offset), (25, y_offset), (0, 0, 255), 1)
        cv2.putText(vis, "Reproj Error", (30, y_offset + 5), font, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(output_path, vis)
    except Exception as e:
        print(f"[WARNING] Failed to save keyframe image: {e}")


def save_keyframe_with_overlay(img_gray: np.ndarray, frame_id: int, keyframe_dir: str,
                               vio_fe=None):
    """
    Save keyframe image with feature tracking overlay (high-level wrapper).
    
    Args:
        img_gray: Grayscale image
        frame_id: Frame index
        keyframe_dir: Directory to save keyframe images
        vio_fe: VIO frontend instance (optional, for feature extraction)
    """
    try:
        # Get current tracked features from frontend
        features = None
        inliers = None
        reprojections = None
        
        if vio_fe is not None:
            # Get current tracked points (KLT features)
            if hasattr(vio_fe, 'last_pts_for_klt') and vio_fe.last_pts_for_klt is not None:
                # Convert from normalized coordinates to pixel coordinates
                focal = vio_fe.K[0, 0]  # Assume fx = fy
                cx = vio_fe.K[0, 2]
                cy = vio_fe.K[1, 2]
                features = vio_fe.last_pts_for_klt * focal + np.array([[cx, cy]])
            
            # Get inlier mask if available
            if hasattr(vio_fe, 'last_inlier_mask') and vio_fe.last_inlier_mask is not None:
                inliers = vio_fe.last_inlier_mask
            
            # Get tracking stats
            tracking_stats = {
                'num_tracked': vio_fe.last_num_tracked,
                'num_inliers': vio_fe.last_num_inliers,
                'mean_parallax': vio_fe.mean_parallax,
            }
        else:
            tracking_stats = None
        
        output_path = os.path.join(keyframe_dir, f"keyframe_{frame_id:06d}.jpg")
        
        save_keyframe_image_with_overlay(
            image=img_gray,
            features=features,
            inliers=inliers,
            reprojections=reprojections,
            output_path=output_path,
            frame_id=frame_id,
            tracking_stats=tracking_stats
        )
        
    except Exception as e:
        print(f"[KEYFRAME] Failed to save keyframe {frame_id}: {e}")


# =============================================================================
# CSV Writers
# =============================================================================

class DebugCSVWriters:
    """
    Manages debug CSV file writers for VIO system.
    
    Creates and manages:
    - Raw IMU data log
    - State & covariance evolution
    - Residual & innovation log
    - Feature tracking statistics
    - MSCKF window log
    - FEJ consistency log
    """
    
    def __init__(self, output_dir: str, save_debug_data: bool = False):
        """
        Initialize debug CSV writers.
        
        Args:
            output_dir: Output directory path
            save_debug_data: Whether to enable debug logging
        """
        self.output_dir = output_dir
        self.enabled = save_debug_data
        
        self.imu_raw_csv = None
        self.state_cov_csv = None
        self.residual_csv = None
        self.feature_stats_csv = None
        self.msckf_window_csv = None
        self.fej_consistency_csv = None
        self.calibration_log = None
        
        if self.enabled:
            self._init_files()
    
    def _init_files(self):
        """Initialize all debug CSV files."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Raw IMU
        self.imu_raw_csv = os.path.join(self.output_dir, "debug_imu_raw.csv")
        with open(self.imu_raw_csv, "w", newline="") as f:
            f.write("t,ori_x,ori_y,ori_z,ori_w,ang_x,ang_y,ang_z,lin_x,lin_y,lin_z\n")
        
        # State & covariance
        self.state_cov_csv = os.path.join(self.output_dir, "debug_state_covariance.csv")
        with open(self.state_cov_csv, "w", newline="") as f:
            f.write("t,frame,P_pos_xx,P_pos_yy,P_pos_zz,P_vel_xx,P_vel_yy,P_vel_zz,"
                    "P_rot_xx,P_rot_yy,P_rot_zz,P_bg_xx,P_bg_yy,P_bg_zz,P_ba_xx,P_ba_yy,P_ba_zz,"
                    "bg_x,bg_y,bg_z,ba_x,ba_y,ba_z\n")
        
        # Residuals
        self.residual_csv = os.path.join(self.output_dir, "debug_residuals.csv")
        with open(self.residual_csv, "w", newline="") as f:
            f.write("t,frame,update_type,innovation_x,innovation_y,innovation_z,"
                    "mahalanobis_dist,chi2_threshold,accepted,NIS,NEES\n")
        
        # Feature stats
        self.feature_stats_csv = os.path.join(self.output_dir, "debug_feature_stats.csv")
        with open(self.feature_stats_csv, "w", newline="") as f:
            f.write("frame,t,num_features_detected,num_features_tracked,num_inliers,"
                    "mean_parallax_px,max_parallax_px,tracking_ratio,inlier_ratio\n")
        
        # MSCKF window
        self.msckf_window_csv = os.path.join(self.output_dir, "debug_msckf_window.csv")
        with open(self.msckf_window_csv, "w", newline="") as f:
            f.write("frame,t,num_camera_clones,num_tracked_features,num_mature_features,"
                    "window_start_time,window_duration,marginalized_clone_id\n")
        
        # FEJ consistency
        self.fej_consistency_csv = os.path.join(self.output_dir, "debug_fej_consistency.csv")
        with open(self.fej_consistency_csv, "w", newline="") as f:
            f.write("timestamp,frame,clone_idx,pos_fej_drift_m,rot_fej_drift_deg,"
                    "bg_fej_drift_rad_s,ba_fej_drift_m_s2\n")
    
    def log_imu_raw(self, t: float, imu_rec):
        """Log raw IMU measurement."""
        if not self.enabled or self.imu_raw_csv is None:
            return
        try:
            with open(self.imu_raw_csv, "a", newline="") as f:
                f.write(f"{t:.6f},{imu_rec.q[0]:.6f},{imu_rec.q[1]:.6f},"
                        f"{imu_rec.q[2]:.6f},{imu_rec.q[3]:.6f},"
                        f"{imu_rec.ang[0]:.6f},{imu_rec.ang[1]:.6f},{imu_rec.ang[2]:.6f},"
                        f"{imu_rec.lin[0]:.6f},{imu_rec.lin[1]:.6f},{imu_rec.lin[2]:.6f}\n")
        except Exception:
            pass
    
    def log_state_covariance(self, t: float, frame: int, kf: Any, bg: np.ndarray, ba: np.ndarray):
        """Log state and covariance."""
        if not self.enabled or self.state_cov_csv is None:
            return
        try:
            p_pos = [kf.P[0, 0], kf.P[1, 1], kf.P[2, 2]]
            p_vel = [kf.P[3, 3], kf.P[4, 4], kf.P[5, 5]]
            p_rot = [kf.P[6, 6], kf.P[7, 7], kf.P[8, 8]]
            p_bg = [kf.P[9, 9], kf.P[10, 10], kf.P[11, 11]]
            p_ba = [kf.P[12, 12], kf.P[13, 13], kf.P[14, 14]]
            
            with open(self.state_cov_csv, "a", newline="") as f:
                f.write(f"{t:.6f},{frame},"
                        f"{p_pos[0]:.6e},{p_pos[1]:.6e},{p_pos[2]:.6e},"
                        f"{p_vel[0]:.6e},{p_vel[1]:.6e},{p_vel[2]:.6e},"
                        f"{p_rot[0]:.6e},{p_rot[1]:.6e},{p_rot[2]:.6e},"
                        f"{p_bg[0]:.6e},{p_bg[1]:.6e},{p_bg[2]:.6e},"
                        f"{p_ba[0]:.6e},{p_ba[1]:.6e},{p_ba[2]:.6e},"
                        f"{bg[0]:.6f},{bg[1]:.6f},{bg[2]:.6f},"
                        f"{ba[0]:.6f},{ba[1]:.6f},{ba[2]:.6f}\n")
        except Exception:
            pass
    
    def log_feature_stats(self, frame: int, t: float, num_detected: int,
                          num_tracked: int, num_inliers: int,
                          mean_parallax: float, max_parallax: float,
                          tracking_ratio: float, inlier_ratio: float):
        """Log feature tracking statistics."""
        if not self.enabled or self.feature_stats_csv is None:
            return
        try:
            with open(self.feature_stats_csv, "a", newline="") as f:
                f.write(f"{frame},{t:.6f},"
                        f"{num_detected},{num_tracked},{num_inliers},"
                        f"{mean_parallax:.2f},{max_parallax:.2f},"
                        f"{tracking_ratio:.3f},{inlier_ratio:.3f}\n")
        except Exception:
            pass


def build_calibration_params(global_config: Dict, vio_config: Any,
                             lat0: float = 0.0, lon0: float = 0.0, 
                             alt0: float = 0.0) -> Dict:
    """
    Build calibration parameters dict from configs.
    
    This extracts all calibration-related parameters from global_config
    and vio_config into a single dict that can be passed to save_calibration_log.
    
    Args:
        global_config: Complete YAML configuration dict
        vio_config: VIOConfig dataclass
        lat0, lon0, alt0: Initial position coordinates
        
    Returns:
        Dict with all parameters needed for save_calibration_log
    """
    from .config import CAMERA_VIEW_CONFIGS
    
    # Get camera view config
    view_cfg = global_config.get('CAMERA_VIEW_CONFIGS', {}).get(
        vio_config.camera_view,
        CAMERA_VIEW_CONFIGS.get(vio_config.camera_view, CAMERA_VIEW_CONFIGS['nadir'])
    )
    
    return {
        'camera_view': vio_config.camera_view,
        'view_cfg': view_cfg,
        'kb_params': global_config.get('KB_PARAMS', {}),
        'imu_params': global_config.get('IMU_PARAMS', {}),
        'mag_params': {
            'declination': global_config.get('MAG_DECLINATION', 0.0),
            'hard_iron': global_config.get('MAG_HARD_IRON_OFFSET', None),
            'soft_iron': global_config.get('MAG_SOFT_IRON_MATRIX', None),
            'use_estimated_bias': getattr(vio_config, 'use_mag_estimated_bias', False),
            'sigma_mag_bias_init': getattr(vio_config, 'sigma_mag_bias_init', 0.1),
            'sigma_mag_bias': getattr(vio_config, 'sigma_mag_bias', 0.001),
        },
        'noise_params': {
            'sigma_vo_vel': global_config.get('SIGMA_VO', 0.5),
            'sigma_mag_yaw': global_config.get('SIGMA_MAG_YAW', 0.15),
            'sigma_agl_z': global_config.get('SIGMA_AGL_Z', 2.5),
        },
        'vio_params': {
            'use_vio_velocity': vio_config.use_vio_velocity,
            'use_magnetometer': vio_config.use_magnetometer,
            'camera_view': vio_config.camera_view,
            'z_state': vio_config.z_state,
        },
        'initial_state': {
            'lat0': lat0,
            'lon0': lon0,
            'alt0': alt0,
        },
        'estimate_imu_bias': vio_config.estimate_imu_bias,
        'plane_config': global_config.get('plane', None),
        'global_config': global_config,
        'vio_config': vio_config,
    }


def save_calibration_log(output_path: str, camera_view: str, view_cfg: Dict,
                         kb_params: Dict, imu_params: Dict, mag_params: Dict,
                         noise_params: Dict, vio_params: Dict,
                         initial_state: Dict, estimate_imu_bias: bool,
                         plane_config: Optional[Dict] = None,
                         vio_config: Optional[Any] = None,
                         global_config: Optional[Dict] = None):
    """
    Save calibration snapshot for reproducibility.
    
    Args:
        output_path: Path to save calibration log
        camera_view: Camera view mode
        view_cfg: View configuration
        kb_params: Kannala-Brandt camera parameters
        imu_params: IMU noise parameters
        mag_params: Magnetometer calibration parameters
        noise_params: EKF process noise parameters
        vio_params: VIO quality control parameters
        initial_state: Initial state values
        estimate_imu_bias: Whether bias estimation was enabled
        plane_config: Plane-Aided MSCKF configuration (optional)
        vio_config: VIOConfig dataclass (optional, for full audit)
        global_config: Complete YAML configuration dict (optional, for full audit)
    """
    from datetime import datetime
    
    try:
        with open(output_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("VIO System Configuration & Calibration Log\n")
            f.write("=" * 70 + "\n\n")
            
            # === Timestamp ===
            f.write("[Timestamp]\n")
            f.write(f"  Generated at: {datetime.now().isoformat()}\n\n")
            
            # === Runtime Options (from VIOConfig) ===
            if vio_config is not None:
                f.write("[Runtime Options (from VIOConfig)]\n")
                f.write("-" * 50 + "\n")
                f.write(f"  config_yaml: {getattr(vio_config, 'config_yaml', 'N/A')}\n")
                f.write(f"  camera_view: {getattr(vio_config, 'camera_view', camera_view)}\n")
                f.write(f"  use_magnetometer: {getattr(vio_config, 'use_magnetometer', 'N/A')}\n")
                f.write(f"  estimate_imu_bias: {getattr(vio_config, 'estimate_imu_bias', estimate_imu_bias)}\n")
                f.write(f"  use_vio_velocity: {getattr(vio_config, 'use_vio_velocity', 'N/A')}\n")
                f.write(f"  use_preintegration: {getattr(vio_config, 'use_preintegration', 'N/A')}\n")
                f.write(f"  fast_mode: {getattr(vio_config, 'fast_mode', 'N/A')}\n")
                f.write(f"  frame_skip: {getattr(vio_config, 'frame_skip', 'N/A')}\n")
                f.write(f"  z_state: {getattr(vio_config, 'z_state', 'N/A')}\n")
                f.write(f"  save_debug_data: {getattr(vio_config, 'save_debug_data', 'N/A')}\n")
                f.write(f"  save_keyframe_images: {getattr(vio_config, 'save_keyframe_images', 'N/A')}\n")
                f.write(f"  downscale_size: {getattr(vio_config, 'downscale_size', 'N/A')}\n")
                f.write("\n")
                
                # === Input Paths ===
                f.write("[Input Paths]\n")
                f.write("-" * 50 + "\n")
                f.write(f"  imu_path: {getattr(vio_config, 'imu_path', 'N/A')}\n")
                f.write(f"  quarry_path: {getattr(vio_config, 'quarry_path', 'N/A')}\n")
                f.write(f"  images_dir: {getattr(vio_config, 'images_dir', None) or 'None'}\n")
                f.write(f"  images_index_csv: {getattr(vio_config, 'images_index_csv', None) or 'None'}\n")
                f.write(f"  mbtiles_path: {getattr(vio_config, 'mbtiles_path', None) or 'None'}\n")
                f.write(f"  mag_csv: {getattr(vio_config, 'mag_csv', None) or 'None'}\n")
                f.write(f"  dem_path: {getattr(vio_config, 'dem_path', None) or 'None'}\n")
                f.write(f"  ground_truth_path: {getattr(vio_config, 'ground_truth_path', None) or 'None'}\n")
                f.write(f"  output_dir: {getattr(vio_config, 'output_dir', 'N/A')}\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("Sensor Calibration Parameters (from YAML)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("[Camera View]\n")
            f.write(f"  Mode: {camera_view}\n")
            f.write(f"  Extrinsics: {view_cfg.get('extrinsics', 'N/A')}\n")
            f.write(f"  Nadir threshold: {view_cfg.get('nadir_threshold', 'N/A')}°\n\n")
            
            # VIO parameters from view config
            f.write("[VIO View Parameters]\n")
            f.write(f"  use_vz_only: {view_cfg.get('use_vz_only', 'N/A')}\n")
            f.write(f"  sigma_scale_xy: {view_cfg.get('sigma_scale_xy', 'N/A')}\n")
            f.write(f"  sigma_scale_z: {view_cfg.get('sigma_scale_z', 'N/A')}\n")
            f.write(f"  min_parallax: {view_cfg.get('min_parallax', 'N/A')}\n")
            f.write(f"  max_corners: {view_cfg.get('max_corners', 'N/A')}\n\n")
            
            f.write("[Camera Intrinsics - Kannala-Brandt]\n")
            for key, val in kb_params.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")
            
            f.write("[IMU Parameters]\n")
            for key, val in imu_params.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")
            
            f.write("[EKF Process Noise]\n")
            for key, val in noise_params.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")
            
            f.write("[VIO Quality Control]\n")
            for key, val in vio_params.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")
            
            # Plane-Aided MSCKF
            if plane_config:
                f.write("[Plane-Aided MSCKF]\n")
                f.write(f"  Enabled: {plane_config.get('enabled', False)}\n")
                if plane_config.get('enabled', False):
                    f.write(f"  Min points per plane: {plane_config.get('min_points_per_plane', 10)}\n")
                    f.write(f"  Angle threshold: {plane_config.get('angle_threshold_deg', 15.0)}°\n")
                    f.write(f"  Distance threshold: {plane_config.get('distance_threshold_m', 0.15)} m\n")
                    f.write(f"  Min plane area: {plane_config.get('min_area_m2', 0.5)} m²\n")
                    f.write(f"  Use aided triangulation: {plane_config.get('use_aided_triangulation', True)}\n")
                    f.write(f"  Use constraints: {plane_config.get('use_constraints', True)}\n")
                f.write("\n")
            
            # TRN (Terrain Referenced Navigation)
            if global_config:
                trn_cfg = global_config.get('trn', {})
                if trn_cfg:
                    f.write("[Terrain Referenced Navigation (TRN)]\n")
                    f.write(f"  Enabled: {trn_cfg.get('enabled', False)}\n")
                    if trn_cfg.get('enabled', False):
                        f.write(f"  Profile window: {trn_cfg.get('profile_window_sec', 30.0)} sec\n")
                        f.write(f"  Min samples: {trn_cfg.get('min_samples', 20)}\n")
                        f.write(f"  Search radius: {trn_cfg.get('search_radius_m', 500.0)} m\n")
                        f.write(f"  Search step: {trn_cfg.get('search_step_m', 30.0)} m\n")
                        f.write(f"  Min terrain variation: {trn_cfg.get('min_terrain_variation_m', 10.0)} m\n")
                        f.write(f"  Correlation threshold: {trn_cfg.get('max_correlation_threshold', 0.7)}\n")
                        f.write(f"  Update interval: {trn_cfg.get('update_interval_sec', 10.0)} sec\n")
                        f.write(f"  Sigma TRN XY: {trn_cfg.get('sigma_trn_xy', 50.0)} m\n")
                    f.write("\n")
                
                # Loop Closure Detection
                loop_cfg = global_config.get('loop_closure', {})
                if loop_cfg:
                    f.write("[Loop Closure Detection]\n")
                    f.write(f"  Enabled: {loop_cfg.get('use_loop_closure', False)}\n")
                    if loop_cfg.get('use_loop_closure', False):
                        f.write(f"  Position threshold: {loop_cfg.get('position_threshold', 30.0)} m\n")
                        f.write(f"  Min keyframe distance: {loop_cfg.get('min_keyframe_dist', 15.0)} m\n")
                        f.write(f"  Min keyframe yaw: {loop_cfg.get('min_keyframe_yaw', 20.0)}°\n")
                        f.write(f"  Min frame gap: {loop_cfg.get('min_frame_gap', 50)}\n")
                        f.write(f"  Min inliers: {loop_cfg.get('min_inliers', 15)}\n")
                    f.write("\n")
                
                # Vibration Detection
                vib_cfg = global_config.get('vibration', {})
                if vib_cfg:
                    f.write("[Vibration Detection]\n")
                    f.write(f"  Enabled: {vib_cfg.get('use_vibration_detector', True)}\n")
                    if vib_cfg.get('use_vibration_detector', True):
                        f.write(f"  Window size: {vib_cfg.get('window_size', 50)}\n")
                        f.write(f"  Threshold multiplier: {vib_cfg.get('threshold_multiplier', 5.0)}\n")
                    f.write("\n")
                
                # ZUPT (Zero Velocity Update)
                zupt_cfg = global_config.get('zupt', {})
                if zupt_cfg:
                    f.write("[ZUPT (Zero Velocity Update)]\n")
                    f.write(f"  Enabled: {zupt_cfg.get('enabled', False)}\n")
                    if zupt_cfg.get('enabled', False):
                        f.write(f"  Accel threshold: {zupt_cfg.get('accel_threshold', 0.5)} m/s²\n")
                        f.write(f"  Gyro threshold: {zupt_cfg.get('gyro_threshold', 0.05)} rad/s\n")
                        f.write(f"  Min duration: {zupt_cfg.get('min_duration', 2.0)} sec\n")
                        f.write(f"  Velocity threshold: {zupt_cfg.get('velocity_threshold', 0.2)} m/s\n")
                        f.write(f"  Consecutive required: {zupt_cfg.get('consecutive_required', 5)}\n")
                    f.write("\n")
                
                # Fast Mode
                fast_cfg = global_config.get('fast_mode', {})
                if fast_cfg:
                    f.write("[Performance Optimization (Fast Mode)]\n")
                    f.write(f"  Use fast mode: {fast_cfg.get('use_fast_mode', False)}\n")
                    f.write(f"  Frame skip: {fast_cfg.get('frame_skip', 1)}\n")
                    f.write("\n")
                
                # Phase Detection Thresholds (v3.4.0)
                phase_cfg = global_config.get('PHASE_SPINUP_VELOCITY_THRESH', None)
                if phase_cfg is not None:
                    f.write("[Flight Phase Detection Thresholds (v3.4.0)]\n")
                    f.write(f"  SPINUP velocity thresh: {global_config.get('PHASE_SPINUP_VELOCITY_THRESH', 1.0)} m/s\n")
                    f.write(f"  SPINUP vibration thresh: {global_config.get('PHASE_SPINUP_VIBRATION_THRESH', 0.3)} rad/s\n")
                    f.write(f"  SPINUP altitude change thresh: {global_config.get('PHASE_SPINUP_ALT_CHANGE_THRESH', 5.0)} m\n")
                    f.write(f"  EARLY velocity sigma thresh: {global_config.get('PHASE_EARLY_VELOCITY_SIGMA_THRESH', 3.0)} m/s\n")
                    f.write("\n")
            
            f.write("[Magnetometer Calibration]\n")
            for key, val in mag_params.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")
            
            # VPS Configuration
            if global_config:
                vps_cfg = global_config.get('vps', {})
                if vps_cfg:
                    f.write("[VPS Configuration]\n")
                    f.write(f"  Enabled: {vps_cfg.get('enabled', False)}\n")
                    if vps_cfg.get('enabled', False):
                        f.write(f"  MBTiles path: {vps_cfg.get('mbtiles_path', 'N/A')}\n")
                        f.write(f"  Camera view: {vps_cfg.get('camera_view', 'nadir')}\n")
                        f.write(f"  Min inliers: {vps_cfg.get('min_inliers', 20)}\n")
                        f.write(f"  Min confidence: {vps_cfg.get('min_confidence', 0.3)}\n")
                        f.write(f"  Max reproj error: {vps_cfg.get('max_reproj_error', 5.0)} px\n")
                        f.write(f"  Altitude range: {vps_cfg.get('min_altitude', 30)} - {vps_cfg.get('max_altitude', 500)} m\n")
                        f.write(f"  Min update interval: {vps_cfg.get('min_update_interval', 0.5)} sec\n")
                        f.write(f"  Max latency: {vps_cfg.get('max_delay_sec', 0.5)} sec (stochastic cloning)\n")
                        f.write(f"  Max clones: {vps_cfg.get('max_clones', 3)}\n")
                        f.write(f"  Device: {vps_cfg.get('device', 'cpu')}\n")
                    f.write("\n")
            
            f.write("[Initial State]\n")
            for key, val in initial_state.items():
                f.write(f"  {key}: {val}\n")
            f.write(f"  Estimate IMU bias: {estimate_imu_bias}\n")
            f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("END OF CONFIGURATION LOG\n")
            f.write("=" * 70 + "\n")
            
    except Exception as e:
        print(f"[WARNING] Failed to save calibration log: {e}")


# =============================================================================
# Error Statistics
# =============================================================================

def print_error_statistics(error_csv: str):
    """
    Print error statistics from error log CSV.
    
    Args:
        error_csv: Path to error log CSV file
    """
    try:
        import pandas as pd
        
        error_df = pd.read_csv(error_csv)
        if len(error_df) == 0:
            return
        
        print("\n=== Error Statistics (VIO vs GPS Ground Truth) ===")
        print("Position Error:")
        print(f"  Mean: {error_df['pos_error_m'].mean():.2f} m")
        print(f"  Median: {error_df['pos_error_m'].median():.2f} m")
        print(f"  Max: {error_df['pos_error_m'].max():.2f} m")
        print(f"  Final: {error_df['pos_error_m'].iloc[-1]:.2f} m")
        
        print("Velocity Error:")
        print(f"  Mean: {error_df['vel_error_m_s'].mean():.3f} m/s")
        print(f"  Final: {error_df['vel_error_m_s'].iloc[-1]:.3f} m/s")
        
        print("Altitude Error:")
        print(f"  Mean: {error_df['alt_error_m'].mean():.2f} m")
        print(f"  Final: {error_df['alt_error_m'].iloc[-1]:.2f} m")
        
        # Yaw error
        yaw_errors = error_df['yaw_error_deg'].dropna()
        if len(yaw_errors) > 0:
            print("Yaw Error:")
            print(f"  Mean: {yaw_errors.mean():.1f}°")
            print(f"  Final: {yaw_errors.iloc[-1]:.1f}°")
        
        print(f"\nDetailed errors saved to: {error_csv}")
    except Exception:
        pass


def compute_rmse(error_df, column: str) -> float:
    """
    Compute RMSE for a column in error dataframe.
    
    Args:
        error_df: Pandas DataFrame with errors
        column: Column name
    
    Returns:
        RMSE value
    """
    try:
        return float(np.sqrt(np.mean(error_df[column].values ** 2)))
    except Exception:
        return float('nan')


# =============================================================================
# Additional Logging Functions (from VIORunner)
# =============================================================================

def log_state_debug(state_dbg_csv: str, t: float, kf: Any,
                    dem_now: float, agl_now: float, msl_now: float,
                    a_world: np.ndarray):
    """
    Log full state debug information.
    
    Args:
        state_dbg_csv: Path to state debug CSV
        t: Current timestamp
        kf: ExtendedKalmanFilter instance
        dem_now: Current DEM height
        agl_now: Current AGL
        msl_now: Current MSL
        a_world: World-frame acceleration [3]
    """
    try:
        px = float(kf.x[0, 0])
        py = float(kf.x[1, 0])
        pz = float(kf.x[2, 0])
        vx = float(kf.x[3, 0])
        vy = float(kf.x[4, 0])
        vz = float(kf.x[5, 0])
    except Exception:
        px = py = pz = vx = vy = vz = float('nan')
    
    try:
        a_wx = float(a_world[0])
        a_wy = float(a_world[1])
        a_wz = float(a_world[2])
    except Exception:
        a_wx = a_wy = a_wz = float('nan')
    
    dem_val = dem_now if dem_now is not None else float('nan')
    agl_val = agl_now if agl_now is not None else float('nan')
    msl_val = msl_now if msl_now is not None else float('nan')
    
    try:
        with open(state_dbg_csv, "a", newline="") as f:
            f.write(f"{t:.6f},{px:.6f},{py:.6f},{pz:.6f},"
                    f"{vx:.6f},{vy:.6f},{vz:.6f},"
                    f"{a_wx:.6f},{a_wy:.6f},{a_wz:.6f},"
                    f"{dem_val:.6f},{agl_val:.6f},{msl_val:.6f}\n")
    except Exception:
        pass


def log_vo_debug(vo_dbg_csv: str, frame: int, num_inliers: int,
                 rot_angle_deg: float, alignment_deg: float,
                 rotation_rate_deg_s: float, use_only_vz: bool,
                 skip_vo: bool, vo_dx: float, vo_dy: float, vo_dz: float,
                 vel_vx: float, vel_vy: float, vel_vz: float):
    """
    Log visual odometry debug information.
    
    Args:
        vo_dbg_csv: Path to VO debug CSV
        frame: Frame index
        num_inliers: Number of inlier matches
        rot_angle_deg: Rotation angle from VO (degrees)
        alignment_deg: Alignment with expected motion (degrees)
        rotation_rate_deg_s: Rotation rate (degrees/sec)
        use_only_vz: Whether only vertical velocity is used
        skip_vo: Whether VO was skipped
        vo_dx, vo_dy, vo_dz: VO translation
        vel_vx, vel_vy, vel_vz: VIO velocities
    """
    try:
        with open(vo_dbg_csv, "a", newline="") as f:
            f.write(f"{max(frame, 0)},{num_inliers},{rot_angle_deg:.3f},"
                    f"{alignment_deg:.3f},{rotation_rate_deg_s:.3f},"
                    f"{int(use_only_vz)},{int(skip_vo)},"
                    f"{vo_dx:.6f},{vo_dy:.6f},{vo_dz:.6f},"
                    f"{vel_vx:.3f},{vel_vy:.3f},{vel_vz:.3f}\n")
    except Exception:
        pass


def log_msckf_window(msckf_window_csv: str, frame: int, t: float,
                     num_clones: int, num_tracked: int, num_mature: int,
                     window_start: float, marginalized_clone: int):
    """
    Log MSCKF window state.
    
    Args:
        msckf_window_csv: Path to MSCKF window CSV
        frame: Frame index
        t: Current timestamp
        num_clones: Number of camera clones
        num_tracked: Number of tracked features
        num_mature: Number of mature features
        window_start: Start time of sliding window
        marginalized_clone: Index of marginalized clone (-1 if none)
    """
    try:
        window_duration = t - window_start if window_start > 0 else 0.0
        with open(msckf_window_csv, "a", newline="") as f:
            f.write(f"{frame},{t:.6f},{num_clones},{num_tracked},{num_mature},"
                    f"{window_start:.6f},{window_duration:.3f},{marginalized_clone}\n")
    except Exception:
        pass


# =============================================================================
# Output CSV Initialization
# =============================================================================

def init_output_csvs(output_dir: str) -> Dict[str, str]:
    """
    Initialize output CSV files.
    
    Args:
        output_dir: Output directory path
    
    Returns:
        Dictionary of CSV file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    # Pose CSV
    paths['pose_csv'] = os.path.join(output_dir, "pose.csv")
    with open(paths['pose_csv'], "w", newline="") as f:
        f.write("Timestamp(s),dt,Frame,PX,PY,PZ_MSL,VX,VY,VZ,lat,lon,AGL(m),"
                "vo_dx,vo_dy,vo_dz,vo_d_roll,vo_d_pitch,vo_d_yaw\n")
    
    # Inference log
    paths['inf_csv'] = os.path.join(output_dir, "inference_log.csv")
    with open(paths['inf_csv'], "w", newline="") as f:
        f.write("Index,Inference Time (s),FPS\n")
    
    # State debug
    paths['state_dbg_csv'] = os.path.join(output_dir, "state_debug.csv")
    with open(paths['state_dbg_csv'], "w", newline="") as f:
        f.write("t,px,py,pz,vx,vy,vz,a_world_x,a_world_y,a_world_z,dem,agl,msl\n")
    
    # VO debug
    paths['vo_dbg'] = os.path.join(output_dir, "vo_debug.csv")
    with open(paths['vo_dbg'], "w", newline="") as f:
        f.write("Frame,num_inliers,rot_angle_deg,alignment_deg,rotation_rate_deg_s,"
                "use_only_vz,skip_vo,vo_dx,vo_dy,vo_dz,vel_vx,vel_vy,vel_vz\n")
    
    # MSCKF debug
    paths['msckf_dbg'] = os.path.join(output_dir, "msckf_debug.csv")
    with open(paths['msckf_dbg'], "w", newline="") as f:
        f.write("frame,feature_id,num_observations,triangulation_success,"
                "reprojection_error_px,innovation_norm,update_applied,chi2_test\n")
    
    # Error log
    paths['error_csv'] = os.path.join(output_dir, "error_log.csv")
    with open(paths['error_csv'], "w", newline="") as f:
        f.write("t,pos_error_m,pos_error_E,pos_error_N,pos_error_U,"
                "vel_error_m_s,vel_error_E,vel_error_N,vel_error_U,"
                "alt_error_m,yaw_vio_deg,yaw_gps_deg,yaw_error_deg,"
                "gps_lat,gps_lon,gps_alt,vio_E,vio_N,vio_U\n")
    
    return paths
