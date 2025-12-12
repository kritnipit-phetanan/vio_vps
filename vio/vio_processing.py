#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIO Processing Functions Module

High-level VIO processing functions that combine multiple modules.
These functions are too complex to be instance methods but need
coordination across multiple modules.

Author: VIO project
"""

import numpy as np
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R_scipy

from .config import CAMERA_VIEW_CONFIGS, BODY_T_CAMDOWN, BODY_T_CAMFRONT
from .vps_integration import xy_to_latlon
from .output_utils import log_measurement_update


def apply_vio_velocity_update(kf, r_vo_mat: np.ndarray, t_unit: np.ndarray,
                               t: float, dt_img: float, avg_flow_px: float,
                               imu_rec, global_config: dict, camera_view: str,
                               dem_reader, lat0: float, lon0: float,
                               z_state: str, use_vio_velocity: bool,
                               save_debug: bool = False,
                               residual_csv: Optional[str] = None,
                               vio_frame: int = -1,
                               vio_fe=None) -> bool:
    """
    Apply VIO velocity update with scale recovery and chi-square gating.
    
    This function:
    1. Recovers scale from AGL using optical flow
    2. Computes velocity in world frame
    3. Applies chi-square innovation gating
    4. Updates EKF if innovation passes gating
    
    Args:
        kf: ExtendedKalmanFilter instance
        r_vo_mat: Relative rotation matrix from Essential matrix
        t_unit: Unit translation vector from Essential matrix
        t: Current timestamp
        dt_img: Time between images
        avg_flow_px: Average optical flow in pixels
        imu_rec: Current IMU record
        global_config: Global configuration dictionary
        camera_view: Camera view mode
        dem_reader: DEM reader for AGL
        lat0, lon0: Origin coordinates
        z_state: Z state representation ("msl" or "agl")
        use_vio_velocity: Whether to apply velocity update
        save_debug: Enable debug logging
        residual_csv: Path to residual CSV
        vio_frame: Current VIO frame index
        vio_fe: VIO frontend (for flow direction)
    
    Returns:
        True if update was accepted, False otherwise
    """
    kb_params = global_config.get('KB_PARAMS', {'mu': 600})
    sigma_vo = global_config.get('SIGMA_VO_VEL', 0.5)
    
    # Get camera extrinsics
    view_cfg = CAMERA_VIEW_CONFIGS.get(camera_view, CAMERA_VIEW_CONFIGS['nadir'])
    extrinsics_name = view_cfg['extrinsics']
    
    if extrinsics_name == 'BODY_T_CAMDOWN':
        body_t_cam = BODY_T_CAMDOWN
    elif extrinsics_name == 'BODY_T_CAMFRONT':
        body_t_cam = BODY_T_CAMFRONT
    else:
        body_t_cam = BODY_T_CAMDOWN
    
    R_cam_to_body = body_t_cam[:3, :3]
    
    # Map direction
    t_norm = t_unit / (np.linalg.norm(t_unit) + 1e-12)
    t_body = R_cam_to_body @ t_norm
    
    # Get rotation from IMU quaternion
    q_imu = imu_rec.q
    Rwb = R_scipy.from_quat(q_imu).as_matrix()
    
    # Scale recovery using AGL
    lat_now, lon_now = xy_to_latlon(kf.x[0, 0], kf.x[1, 0], lat0, lon0)
    dem_now = dem_reader.sample_m(lat_now, lon_now) if dem_reader.ds else 0.0
    if dem_now is None or np.isnan(dem_now):
        dem_now = 0.0
    
    if z_state.lower() == "agl":
        agl = abs(kf.x[2, 0])
    else:
        agl = abs(kf.x[2, 0] - dem_now)
    agl = max(1.0, agl)
    
    # Optical flow-based scale
    focal_px = kb_params.get('mu', 600)
    if dt_img > 1e-4 and avg_flow_px > 2.0:
        scale_flow = agl / focal_px
        speed_final = (avg_flow_px / dt_img) * scale_flow
    else:
        speed_final = 0.0
    
    speed_final = min(speed_final, 50.0)  # Clamp to 50 m/s
    
    # Compute velocity in world frame
    if avg_flow_px > 2.0 and vio_fe is not None and vio_fe.last_matches is not None:
        pts_prev, pts_cur = vio_fe.last_matches
        if len(pts_prev) > 0:
            flows_normalized = pts_cur - pts_prev
            median_flow = np.median(flows_normalized, axis=0)
            flow_norm = np.linalg.norm(median_flow)
            if flow_norm > 1e-6:
                flow_dir = median_flow / flow_norm
                vel_cam = np.array([-flow_dir[0], -flow_dir[1], 0.0])
                vel_cam = vel_cam / np.linalg.norm(vel_cam + 1e-9)
                vel_body = R_cam_to_body @ vel_cam * speed_final
            else:
                vel_body = t_body * speed_final
        else:
            vel_body = t_body * speed_final
    else:
        vel_body = t_body * speed_final
    
    vel_world = Rwb @ vel_body
    
    # Determine if using VZ only (for nadir cameras)
    use_only_vz = view_cfg.get('use_vz_only', True)
    
    # ESKF velocity update
    num_clones = (kf.x.shape[0] - 16) // 7
    err_dim = 15 + 6 * num_clones
    
    if use_only_vz:
        h_vel = np.zeros((1, err_dim), dtype=float)
        h_vel[0, 5] = 1.0
        vel_meas = np.array([[vel_world[2]]])
        r_mat = np.array([[(sigma_vo * view_cfg.get('sigma_scale_z', 2.0))**2]])
    else:
        h_vel = np.zeros((3, err_dim), dtype=float)
        h_vel[0, 3] = 1.0
        h_vel[1, 4] = 1.0
        h_vel[2, 5] = 1.0
        vel_meas = vel_world.reshape(-1, 1)
        scale_xy = view_cfg.get('sigma_scale_xy', 1.0)
        scale_z = view_cfg.get('sigma_scale_z', 2.0)
        r_mat = np.diag([(sigma_vo * scale_xy)**2, (sigma_vo * scale_xy)**2, (sigma_vo * scale_z)**2])
    
    def h_fun(x, h=h_vel):
        return h
    
    def hx_fun(x, h=h_vel):
        if use_only_vz:
            return x[5:6].reshape(1, 1)
        else:
            return x[3:6].reshape(3, 1)
    
    # Apply update with chi-square gating
    if use_vio_velocity:
        # Compute innovation for gating
        predicted_vel = hx_fun(kf.x)
        innovation = vel_meas - predicted_vel
        s_mat = h_vel @ kf.P @ h_vel.T + r_mat
        
        # Chi-square test
        try:
            m2 = innovation.T @ np.linalg.inv(s_mat) @ innovation
            chi2_value = float(m2)
            mahal_dist = np.sqrt(chi2_value)
        except Exception:
            chi2_value = float('inf')
            mahal_dist = float('nan')
        
        # Chi-square thresholds (95% confidence)
        chi2_threshold = 3.84 if use_only_vz else 7.81  # 1 DOF vs 3 DOF
        
        if chi2_value < chi2_threshold:
            # Accept update
            kf.update(z=vel_meas, HJacobian=h_fun, Hx=hx_fun, R=r_mat)
            print(f"[VIO] Velocity update: speed={speed_final:.2f}m/s, vz_only={use_only_vz}, "
                  f"chi2={chi2_value:.2f}")
            accepted = True
        else:
            # Reject outlier
            print(f"[VIO] Velocity REJECTED: chi2={chi2_value:.2f} > {chi2_threshold:.1f}, "
                  f"mahal={mahal_dist:.2f}")
            accepted = False
        
        # Log to debug_residuals.csv
        if save_debug and residual_csv:
            log_measurement_update(
                residual_csv, t, vio_frame, 'VIO_VEL',
                innovation=innovation.flatten(),
                mahalanobis_dist=mahal_dist,
                chi2_threshold=chi2_threshold,
                accepted=accepted,
                s_matrix=s_mat,
                p_prior=getattr(kf, 'P_prior', kf.P)
            )
        
        return accepted
    
    return False
