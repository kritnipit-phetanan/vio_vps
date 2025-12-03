#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fisheye to Pinhole Rectification Module

Converts fisheye (Kannala-Brandt) images to pinhole/rectilinear images 
for use with standard VIO algorithms.

Key concepts:
1. Creates a rectification map once at initialization
2. Applies remap to each frame efficiently (cv2.remap)
3. Outputs new pinhole intrinsics (K_new) for VIO

Author: VIO project
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class FisheyeRectifier:
    """
    Converts fisheye images to pinhole using Kannala-Brandt model.
    
    The Kannala-Brandt model maps angles to radial distances:
        θ_d = k1*θ + k2*θ³ + k3*θ⁵ + k4*θ⁷ + ...
        r = f * θ_d
    
    This class creates a virtual pinhole camera looking at a subset
    of the fisheye field of view, producing undistorted images.
    """
    
    def __init__(self, 
                 K_fisheye: np.ndarray,
                 D_fisheye: np.ndarray,
                 src_size: Tuple[int, int],
                 fov_deg: float = 90.0,
                 dst_size: Optional[Tuple[int, int]] = None,
                 balance: float = 0.0):
        """
        Initialize rectifier with fisheye calibration parameters.
        
        Args:
            K_fisheye: 3x3 original fisheye camera matrix
            D_fisheye: Kannala-Brandt distortion [k2, k3, k4, k5] or [k1,k2,k3,k4]
            src_size: (width, height) of source fisheye images
            fov_deg: Field of view for virtual pinhole camera (degrees)
            dst_size: (width, height) of output rectified images (default: same as src)
            balance: 0=zoom out (black corners), 1=zoom in (lose corners)
        """
        self.K_fisheye = K_fisheye.astype(np.float64)
        self.D_fisheye = D_fisheye.astype(np.float64)
        self.src_size = src_size  # (w, h)
        self.fov_deg = fov_deg
        self.dst_size = dst_size if dst_size else src_size
        self.balance = balance
        
        # Build rectification maps
        self._build_maps()
        
        print(f"[RECTIFY] Initialized: FOV={fov_deg}°, src={src_size}, dst={self.dst_size}")
        print(f"[RECTIFY] K_new:\n{self.K_new}")
    
    def _build_maps(self):
        """
        Build rectification maps for cv2.remap.
        
        Creates a virtual pinhole camera and computes which fisheye pixel
        maps to each output pixel.
        """
        dst_w, dst_h = self.dst_size
        src_w, src_h = self.src_size
        
        # Virtual pinhole camera intrinsics
        # FOV determines focal length: f = w / (2 * tan(FOV/2))
        fov_rad = np.radians(self.fov_deg)
        f_new = dst_w / (2.0 * np.tan(fov_rad / 2.0))
        
        self.K_new = np.array([
            [f_new, 0, dst_w / 2.0],
            [0, f_new, dst_h / 2.0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Build reverse map: for each dst pixel, find src pixel
        # Create mesh grid of destination coordinates
        u_dst = np.arange(dst_w, dtype=np.float32)
        v_dst = np.arange(dst_h, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u_dst, v_dst)
        
        # Convert dst pixels to normalized coordinates
        fx_new, fy_new = self.K_new[0, 0], self.K_new[1, 1]
        cx_new, cy_new = self.K_new[0, 2], self.K_new[1, 2]
        
        x_norm = (u_grid - cx_new) / fx_new
        y_norm = (v_grid - cy_new) / fy_new
        
        # Convert normalized coords to 3D unit rays (pinhole model)
        r_norm = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan(r_norm)  # angle from optical axis
        
        # Apply Kannala-Brandt forward projection
        # θ_d = k1*θ + k2*θ³ + k3*θ⁵ + k4*θ⁷ + k5*θ⁹
        k1 = 1.0  # Usually implicit
        k2 = self.D_fisheye[0] if len(self.D_fisheye) > 0 else 0
        k3 = self.D_fisheye[1] if len(self.D_fisheye) > 1 else 0
        k4 = self.D_fisheye[2] if len(self.D_fisheye) > 2 else 0
        k5 = self.D_fisheye[3] if len(self.D_fisheye) > 3 else 0
        
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        
        theta_d = k1 * theta + k2 * theta * theta2 + k3 * theta * theta4 + \
                  k4 * theta * theta6 + k5 * theta * theta8
        
        # Radial distance in normalized fisheye
        # r_fisheye = θ_d (for normalized)
        # But we need pixel coordinates
        
        # Direction in image plane (same as x_norm, y_norm but scaled)
        # Handle r_norm = 0 (center)
        scale = np.where(r_norm > 1e-8, theta_d / r_norm, 1.0)
        
        x_fisheye_norm = x_norm * scale
        y_fisheye_norm = y_norm * scale
        
        # Convert to fisheye pixel coordinates
        fx_fish, fy_fish = self.K_fisheye[0, 0], self.K_fisheye[1, 1]
        cx_fish, cy_fish = self.K_fisheye[0, 2], self.K_fisheye[1, 2]
        
        u_src = x_fisheye_norm * fx_fish + cx_fish
        v_src = y_fisheye_norm * fy_fish + cy_fish
        
        # Store maps
        self.map_x = u_src.astype(np.float32)
        self.map_y = v_src.astype(np.float32)
        
        # Compute valid mask (pixels that map to valid fisheye region)
        valid = (u_src >= 0) & (u_src < src_w) & (v_src >= 0) & (v_src < src_h)
        self.valid_mask = valid.astype(np.uint8) * 255
    
    def rectify(self, img_fisheye: np.ndarray) -> np.ndarray:
        """
        Convert fisheye image to rectified pinhole image.
        
        Args:
            img_fisheye: Input fisheye image (grayscale or BGR)
            
        Returns:
            Rectified image with pinhole projection
        """
        return cv2.remap(img_fisheye, self.map_x, self.map_y, 
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                        borderValue=0)
    
    def get_new_intrinsics(self) -> np.ndarray:
        """Get the virtual pinhole camera intrinsic matrix."""
        return self.K_new.copy()
    
    def get_valid_mask(self) -> np.ndarray:
        """Get mask of valid pixels in rectified image."""
        return self.valid_mask.copy()


def create_rectifier_from_config(config: dict, 
                                 src_size: Tuple[int, int],
                                 fov_deg: float = 90.0,
                                 dst_size: Optional[Tuple[int, int]] = None) -> FisheyeRectifier:
    """
    Create FisheyeRectifier from VIO config dictionary.
    
    Args:
        config: Config dict with KB_PARAMS
        src_size: Source image size (w, h)
        fov_deg: Output field of view
        dst_size: Output image size (w, h)
        
    Returns:
        Initialized FisheyeRectifier
    """
    kb = config.get('KB_PARAMS', {})
    
    # Build K matrix from KB params
    fx = kb.get('mu', 500)
    fy = kb.get('mv', 500)
    cx = kb.get('u0', src_size[0] / 2)
    cy = kb.get('v0', src_size[1] / 2)
    
    # Scale for actual image size
    calib_w = kb.get('w', 1440)
    calib_h = kb.get('h', 1080)
    scale_x = src_size[0] / calib_w
    scale_y = src_size[1] / calib_h
    
    K = np.array([
        [fx * scale_x, 0, cx * scale_x],
        [0, fy * scale_y, cy * scale_y],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Distortion coefficients
    D = np.array([
        kb.get('k2', 0),
        kb.get('k3', 0),
        kb.get('k4', 0),
        kb.get('k5', 0)
    ], dtype=np.float64)
    
    return FisheyeRectifier(K, D, src_size, fov_deg, dst_size)


# Test function
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/cvteam/vio_vps_repo')
    from vio.config import load_config
    
    config = load_config('configs/config_bell412_dataset3.yaml')
    
    # Test with sample parameters
    src_size = (360, 240)  # Downscaled
    rectifier = create_rectifier_from_config(config, src_size, fov_deg=90)
    
    print(f"\nK_fisheye:\n{rectifier.K_fisheye}")
    print(f"D_fisheye: {rectifier.D_fisheye}")
    print(f"K_new:\n{rectifier.K_new}")
    
    # Test with dummy image
    dummy = np.random.randint(0, 255, (src_size[1], src_size[0]), dtype=np.uint8)
    rectified = rectifier.rectify(dummy)
    print(f"Rectified shape: {rectified.shape}")
