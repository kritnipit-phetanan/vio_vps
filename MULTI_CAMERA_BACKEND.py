#!/usr/bin/env python3
"""
Multi-Camera VIO Backend - Patch for vio_vps.py
Add this code to vio_vps.py to enable multi-camera support
"""

# Add after line 4467 (before def run())

def create_multi_camera_vio_frontend(camera_configs: dict, kb_params: dict, 
                                      downscale_size: tuple, use_fisheye: bool) -> dict:
    """
    Create VIO front-end instances for multiple cameras.
    
    Args:
        camera_configs: Dict mapping camera name to config (e.g., {'nadir': {...}, 'front': {...}})
        kb_params: Kannala-Brandt parameters
        downscale_size: (width, height) for processing
        use_fisheye: Whether to use fisheye undistortion
    
    Returns:
        Dict mapping camera name to VIOFrontEnd instance
    """
    vio_frontends = {}
    
    for cam_name, cam_config in camera_configs.items():
        print(f"[MULTI-CAM] Initializing {cam_name} camera VIO frontend")
        print(f"[MULTI-CAM]   - max_corners: {cam_config['max_corners']}")
        print(f"[MULTI-CAM]   - min_parallax: {cam_config['min_parallax']}px")
        print(f"[MULTI-CAM]   - use_vz_only: {cam_config['use_vz_only']}")
        
        vio_fe = VIOFrontEnd(
            kb_params=kb_params,
            downscale_w=downscale_size[0],
            downscale_h=downscale_size[1],
            max_corners=cam_config['max_corners'],
            use_fisheye=use_fisheye
        )
        vio_frontends[cam_name] = vio_fe
    
    return vio_frontends


def process_multi_camera_vio(vio_frontends: dict, images_dict: dict, t: float, kf,
                              cam_observations_dict: dict, cam_states_dict: dict,
                              use_vio_velocity: bool, min_parallax: float) -> dict:
    """
    Process VIO updates from multiple cameras.
    
    Args:
        vio_frontends: Dict of VIOFrontEnd instances per camera
        images_dict: Dict mapping camera name to current image path
        t: Current timestamp
        kf: ExtendedKalmanFilter instance
        cam_observations_dict: Dict of camera observations per camera
        cam_states_dict: Dict of camera states per camera
        use_vio_velocity: Whether to use velocity updates
        min_parallax: Minimum parallax threshold
    
    Returns:
        Dict with results: {
            'any_success': bool,
            'results': {cam_name: {...}},
            'total_inliers': int,
            'combined_flow': float
        }
    """
    results = {}
    any_success = False
    total_inliers = 0
    combined_flow = 0.0
    n_cameras = 0
    
    for cam_name, vio_fe in vio_frontends.items():
        if cam_name not in images_dict or images_dict[cam_name] is None:
            continue
        
        img_path = images_dict[cam_name]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"[MULTI-CAM][{cam_name}] Failed to load image: {img_path}")
            continue
        
        # Process VIO for this camera
        dt_img = t - vio_fe.last_t if vio_fe.last_t > 0 else 0.0
        ok, ninl, r_vo_mat, t_unit = vio_fe.step(img, t)
        
        avg_flow = vio_fe.compute_avg_flow()
        
        results[cam_name] = {
            'ok': ok,
            'ninl': ninl,
            'r_vo_mat': r_vo_mat,
            't_unit': t_unit,
            'dt_img': dt_img,
            'avg_flow': avg_flow
        }
        
        if ok and ninl > 0:
            any_success = True
            total_inliers += ninl
            combined_flow += avg_flow
            n_cameras += 1
            
            print(f"[MULTI-CAM][{cam_name}] Success: {ninl} inliers, flow={avg_flow:.2f}px")
        else:
            print(f"[MULTI-CAM][{cam_name}] Failed: {ninl} inliers, flow={avg_flow:.2f}px")
    
    # Average flow across cameras
    if n_cameras > 0:
        combined_flow /= n_cameras
    
    return {
        'any_success': any_success,
        'results': results,
        'total_inliers': total_inliers,
        'combined_flow': combined_flow,
        'n_cameras': n_cameras
    }


def fuse_multi_camera_msckf_updates(vio_frontends: dict, cam_observations_dict: dict,
                                     cam_states_dict: dict, kf, max_features: int = 50,
                                     dem_reader=None, origin_lat: float = 0.0, 
                                     origin_lon: float = 0.0) -> int:
    """
    Fuse MSCKF updates from multiple cameras.
    
    Instead of separate updates per camera, we:
    1. Find mature features from all cameras
    2. Triangulate each feature using all available observations
    3. Stack residuals and Jacobians from all cameras
    4. Perform a single EKF update with combined measurement
    
    Returns:
        Total number of features used for update
    """
    total_features_used = 0
    
    # Collect mature features from all cameras
    all_mature_features = {}  # fid -> list of (cam_name, observations)
    
    for cam_name, vio_fe in vio_frontends.items():
        if cam_name not in cam_observations_dict:
            continue
        
        cam_obs = cam_observations_dict[cam_name]
        cam_states = cam_states_dict[cam_name]
        
        mature_fids = find_mature_features_for_msckf(vio_fe, cam_obs, min_observations=3)
        
        for fid in mature_fids:
            if fid not in all_mature_features:
                all_mature_features[fid] = []
            all_mature_features[fid].append((cam_name, cam_obs, cam_states))
    
    if len(all_mature_features) == 0:
        return 0
    
    # Process features (limited to max_features)
    features_to_process = list(all_mature_features.keys())[:max_features]
    
    # Stack residuals and Jacobians
    all_residuals = []
    all_jacobians = []
    all_R_blocks = []
    
    for fid in features_to_process:
        camera_observations = all_mature_features[fid]
        
        # Triangulate using all camera observations
        # (This requires merging observations from multiple cameras)
        combined_obs = []
        combined_states = []
        
        for cam_name, cam_obs, cam_states in camera_observations:
            obs = get_feature_multi_view_observations(fid, cam_obs)
            combined_obs.extend(obs)
            combined_states.extend(cam_states)
        
        if len(combined_obs) < 3:
            continue
        
        # Triangulate with combined observations
        triangulated = triangulate_feature(
            fid, combined_obs, combined_states, kf,
            use_plane_constraint=True, dem_reader=dem_reader,
            origin_lat=origin_lat, origin_lon=origin_lon
        )
        
        if triangulated is None:
            continue
        
        # Compute measurement update for this feature
        success, ninl, res, H, R = msckf_measurement_update(
            fid, triangulated, combined_obs, combined_states, kf,
            measurement_noise=1e-4, chi2_max_dof=15.36
        )
        
        if success and ninl > 0:
            all_residuals.append(res)
            all_jacobians.append(H)
            all_R_blocks.append(R)
            total_features_used += 1
    
    # Fuse all measurements into single update
    if len(all_residuals) > 0:
        from scipy.linalg import block_diag
        
        z_combined = np.concatenate(all_residuals)
        H_combined = np.vstack(all_jacobians)
        R_combined = block_diag(*all_R_blocks)
        
        # Single EKF update
        try:
            kf.update(z_combined, HJacobian=H_combined, R=R_combined)
            print(f"[MULTI-CAM][MSCKF] Fused {total_features_used} features from multiple cameras")
        except Exception as e:
            print(f"[MULTI-CAM][MSCKF] Update failed: {e}")
            total_features_used = 0
    
    return total_features_used


# Modify run() function to add multi-camera support:
# 
# 1. After loading nadir images (line ~4510), add:
"""
    # Multi-camera support: load additional cameras
    front_imgs = []
    side_imgs = []
    
    if camera_view == 'multi':
        if front_images_dir and front_images_index_csv:
            front_imgs = load_images(front_images_dir, front_images_index_csv)
            print(f"[MULTI-CAM] Front camera: {len(front_imgs)} frames loaded")
        
        if side_images_dir and side_images_index_csv:
            side_imgs = load_images(side_images_dir, side_images_index_csv)
            print(f"[MULTI-CAM] Side camera: {len(side_imgs)} frames loaded")
"""

# 2. After creating VIOFrontEnd (line ~4610), add:
"""
    # Create multi-camera VIO frontends
    vio_frontends = {}
    cam_observations_dict = {}
    cam_states_dict = {}
    
    if camera_view == 'multi':
        # Build camera configs
        camera_configs = {}
        if len(imgs) > 0:
            camera_configs['nadir'] = CAMERA_VIEW_CONFIGS['nadir']
            cam_observations_dict['nadir'] = []
            cam_states_dict['nadir'] = []
        if len(front_imgs) > 0:
            camera_configs['front'] = CAMERA_VIEW_CONFIGS['front']
            cam_observations_dict['front'] = []
            cam_states_dict['front'] = []
        if len(side_imgs) > 0:
            camera_configs['side'] = CAMERA_VIEW_CONFIGS['side']
            cam_observations_dict['side'] = []
            cam_states_dict['side'] = []
        
        vio_frontends = create_multi_camera_vio_frontend(
            camera_configs, KB_PARAMS, downscale_size, USE_FISHEYE
        )
    else:
        # Single camera mode (original code)
        vio_frontends = {camera_view: vio_fe}
        cam_observations_dict = {camera_view: cam_observations}
        cam_states_dict = {camera_view: cam_states}
"""

# 3. In main loop where VIO is processed (line ~5200), replace with:
"""
    if vio_fe is not None and img_idx < len(imgs):
        img_item = imgs[img_idx]
        if abs(t - img_item.t) < 0.1:  # Within 100ms
            
            if camera_view == 'multi':
                # Multi-camera processing
                images_dict = {}
                
                # Match images from all cameras
                if 'nadir' in vio_frontends:
                    images_dict['nadir'] = imgs[img_idx].path if img_idx < len(imgs) else None
                
                if 'front' in vio_frontends:
                    # Find matching front camera frame
                    front_idx = find_closest_image_index(front_imgs, t, max_dt=0.1)
                    images_dict['front'] = front_imgs[front_idx].path if front_idx >= 0 else None
                
                if 'side' in vio_frontends:
                    # Find matching side camera frame
                    side_idx = find_closest_image_index(side_imgs, t, max_dt=0.1)
                    images_dict['side'] = side_imgs[side_idx].path if side_idx >= 0 else None
                
                # Process all cameras
                mc_result = process_multi_camera_vio(
                    vio_frontends, images_dict, t, kf,
                    cam_observations_dict, cam_states_dict,
                    use_vio_velocity, MIN_PARALLAX_PX
                )
                
                if mc_result['any_success']:
                    print(f"[MULTI-CAM] Total: {mc_result['total_inliers']} inliers from {mc_result['n_cameras']} cameras")
                    
                    # VIO velocity update (if enabled and sufficient motion)
                    if use_vio_velocity and mc_result['combined_flow'] >= MIN_PARALLAX_PX:
                        # Fuse velocity measurements from all successful cameras
                        # (Implementation depends on how you want to combine velocities)
                        pass
                    
                    # MSCKF update (fuse features from all cameras)
                    if len(cam_states_dict) > 0:
                        features_used = fuse_multi_camera_msckf_updates(
                            vio_frontends, cam_observations_dict, cam_states_dict,
                            kf, max_features=50, dem_reader=dem,
                            origin_lat=lat0, origin_lon=lon0
                        )
            else:
                # Single camera mode (original code)
                ...
"""

print("[INFO] Multi-camera VIO backend code generated")
print("[INFO] To integrate:")
print("  1. Add helper functions to vio_vps.py (before run() function)")
print("  2. Modify run() function as indicated in comments")
print("  3. Add find_closest_image_index() helper function")
