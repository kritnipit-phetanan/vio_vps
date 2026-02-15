#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPS Image Preprocessor

Prepares drone camera images for satellite matching by:
1. Undistorting fisheye lens
2. Rotating to north-up orientation
3. Scaling to match satellite GSD

Author: VIO project
"""

import numpy as np
import cv2
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PreprocessResult:
    """Result of image preprocessing."""
    image: np.ndarray          # Preprocessed image (grayscale or BGR)
    scale_factor: float        # Scale applied to match satellite GSD
    rotation_deg: float        # Rotation applied (degrees, clockwise)
    drone_gsd: float           # GSD of drone image after preprocessing
    target_gsd: float          # Target GSD (satellite)
    content_ratio: float = 0.0 # Non-empty pixel ratio in final image
    texture_std: float = 0.0   # Std-dev of valid pixels (texture proxy)
    target_gsd_scale: float = 1.0  # Applied multiplier on target_gsd


class VPSImagePreprocessor:
    """
    Preprocesses drone images for satellite matching.
    
    Three-stage pipeline:
    1. Undistort: Remove fisheye distortion → pinhole model
    2. Rotate: Compensate yaw to align with North-Up satellite map
    3. Scale: Match GSD so drone and satellite images have same scale
    """
    
    def __init__(self, 
                 fisheye_rectifier=None,
                 camera_intrinsics: Optional[dict] = None,
                 output_size: Tuple[int, int] = (512, 512)):
        """
        Initialize preprocessor.
        
        Args:
            fisheye_rectifier: Optional FisheyeRectifier from vio module
            camera_intrinsics: Camera intrinsics dict (KB_PARAMS format)
                               Must have 'fx', 'fy' or 'f' for focal length
            output_size: Output image size (width, height)
        """
        self.rectifier = fisheye_rectifier
        self.intrinsics = camera_intrinsics
        self.output_size = output_size
        
        # Get focal length in pixels (for GSD calculation)
        self.focal_px = self._get_focal_px()
    
    def _get_focal_px(self) -> float:
        """Get focal length in pixels from intrinsics."""
        if self.intrinsics is None:
            # Default for common drone cameras
            return 500.0
        
        if 'fx' in self.intrinsics:
            return float(self.intrinsics['fx'])
        elif 'f' in self.intrinsics:
            return float(self.intrinsics['f'])
        elif 'mu' in self.intrinsics and 'mv' in self.intrinsics:
            # KB format uses mu, mv
            return (float(self.intrinsics['mu']) + float(self.intrinsics['mv'])) / 2
        
        return 500.0  # Fallback
    
    def undistort(self, img: np.ndarray) -> np.ndarray:
        """
        Apply fisheye undistortion.
        
        Args:
            img: Input fisheye image
            
        Returns:
            Undistorted image (pinhole model)
        """
        if self.rectifier is None:
            return img
        
        return self.rectifier.rectify(img)
    
    def rotate_to_north_up(self, img: np.ndarray, yaw_rad: float) -> np.ndarray:
        """
        Rotate image to compensate for drone yaw.
        
        Satellite maps are North-Up. Drone camera points wherever the drone
        is heading. We rotate the drone image by -yaw to align with satellite.
        
        Args:
            img: Input image
            yaw_rad: Drone yaw in radians (ENU: 0=East, π/2=North)
            
        Returns:
            Rotated image aligned with North-Up
        """
        # Convert to degrees (OpenCV uses degrees)
        # ENU yaw: 0=East, π/2=North, π=West, -π/2=South
        # We need to rotate by -yaw to align with North-Up map
        yaw_deg = math.degrees(yaw_rad)
        
        # For ENU: rotate by (90 - yaw_deg) to align North with image top
        # Or simply rotate by -yaw_deg to compensate heading
        rotation_angle = -yaw_deg  # Compensate yaw
        
        # Get rotation matrix
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Use reflected border to avoid large black corners destroying features.
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return rotated
    
    def compute_drone_gsd(self, altitude_m: float) -> float:
        """
        Compute drone image GSD (ground sample distance).
        
        GSD = altitude / focal_length_px
        
        Args:
            altitude_m: Altitude above ground (AGL) in meters
            
        Returns:
            GSD in meters per pixel
        """
        if self.focal_px <= 0:
            return 0.3  # Fallback
        
        return altitude_m / self.focal_px
    
    def scale_to_gsd(self, img: np.ndarray, 
                     drone_gsd: float, 
                     target_gsd: float) -> Tuple[np.ndarray, float]:
        """
        Scale image to match satellite GSD.
        
        Args:
            img: Input image
            drone_gsd: Current GSD of drone image (m/px)
            target_gsd: Target GSD (satellite map) (m/px)
            
        Returns:
            (scaled_image, scale_factor)
        """
        if drone_gsd <= 0 or target_gsd <= 0:
            return img, 1.0
        if img is None or getattr(img, "size", 0) == 0:
            return img, 1.0
        
        # Scale factor: if drone GSD is smaller (higher res), scale down
        scale = drone_gsd / target_gsd
        if not np.isfinite(scale) or scale <= 0:
            return img, 1.0
        
        if abs(scale - 1.0) < 0.05:
            # Close enough, no scaling needed
            return img, 1.0
        
        h, w = img.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Limit size
        max_size = 2048
        if new_w > max_size or new_h > max_size:
            limit_scale = max_size / max(new_w, new_h)
            new_w = int(new_w * limit_scale)
            new_h = int(new_h * limit_scale)
            scale *= limit_scale
        
        # Guard against degenerate output size before resize.
        if new_w <= 0 or new_h <= 0:
            return img, 1.0
        
        if new_w < 64 or new_h < 64:
            # Too small, don't scale
            return img, 1.0
        
        try:
            scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        except cv2.error:
            return img, 1.0
        
        return scaled, scale
    
    def center_crop(self, img: np.ndarray, 
                    output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Center crop image to output size.
        
        Args:
            img: Input image
            output_size: Target size (width, height), uses self.output_size if None
            
        Returns:
            Center-cropped image
        """
        if output_size is None:
            output_size = self.output_size
        
        h, w = img.shape[:2]
        target_w, target_h = output_size
        
        if w <= target_w and h <= target_h:
            # Image smaller than target, pad instead
            result = np.zeros((target_h, target_w, 3) if len(img.shape) == 3 
                             else (target_h, target_w), dtype=img.dtype)
            x_off = (target_w - w) // 2
            y_off = (target_h - h) // 2
            result[y_off:y_off+h, x_off:x_off+w] = img
            return result
        
        # Center crop
        x1 = (w - target_w) // 2
        y1 = (h - target_h) // 2
        
        return img[y1:y1+target_h, x1:x1+target_w]

    def center_crop_valid_content(self,
                                  img: np.ndarray,
                                  output_size: Optional[Tuple[int, int]] = None,
                                  intensity_threshold: float = 8.0) -> np.ndarray:
        """
        Crop around valid texture content instead of strict image center.

        This reduces degenerate crops where most of the patch is empty/invalid.
        """
        if output_size is None:
            output_size = self.output_size

        h, w = img.shape[:2]
        target_w, target_h = output_size

        if w <= target_w and h <= target_h:
            return self.center_crop(img, output_size=output_size)

        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        valid = gray > float(intensity_threshold)
        if not np.any(valid):
            return self.center_crop(img, output_size=output_size)

        ys, xs = np.nonzero(valid)
        cx = int(np.round(np.median(xs)))
        cy = int(np.round(np.median(ys)))

        x1 = int(np.clip(cx - target_w // 2, 0, max(0, w - target_w)))
        y1 = int(np.clip(cy - target_h // 2, 0, max(0, h - target_h)))
        return img[y1:y1+target_h, x1:x1+target_w]

    def compute_content_metrics(self, img: np.ndarray, intensity_threshold: float = 8.0) -> Tuple[float, float]:
        """
        Return lightweight content diagnostics for matcher quality policy.
        """
        if img is None or getattr(img, "size", 0) == 0:
            return 0.0, 0.0
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        valid = gray > float(intensity_threshold)
        content_ratio = float(np.mean(valid)) if valid.size > 0 else 0.0
        if np.any(valid):
            texture_std = float(np.std(gray[valid]))
        else:
            texture_std = float(np.std(gray))
        return content_ratio, texture_std
    
    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed."""
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def preprocess(self, 
                   img: np.ndarray,
                   yaw_rad: float,
                   altitude_m: float,
                   target_gsd: float,
                   grayscale: bool = True,
                   target_gsd_scale: float = 1.0) -> PreprocessResult:
        """
        Full preprocessing pipeline.
        
        Args:
            img: Input drone camera image
            yaw_rad: Drone yaw in radians (ENU convention)
            altitude_m: Altitude AGL in meters
            target_gsd: Satellite map GSD (m/px)
            grayscale: Convert to grayscale (most matchers prefer this)
            target_gsd_scale: Multiplier for target GSD (for scale hypothesis search)
            
        Returns:
            PreprocessResult with processed image and metadata
        """
        # 1. Undistort
        undistorted = self.undistort(img)
        
        # 2. Rotate to North-Up
        rotated = self.rotate_to_north_up(undistorted, yaw_rad)
        
        # 3. Scale to match satellite GSD
        drone_gsd = self.compute_drone_gsd(altitude_m)
        scale_mult = float(target_gsd_scale) if np.isfinite(target_gsd_scale) else 1.0
        if scale_mult <= 0.0:
            scale_mult = 1.0
        target_gsd_eff = float(target_gsd) * scale_mult
        scaled, scale_factor = self.scale_to_gsd(rotated, drone_gsd, target_gsd_eff)
        
        # 4. Crop around valid content to avoid empty/degenerate patches.
        cropped = self.center_crop_valid_content(scaled)
        
        # 5. Grayscale (optional)
        if grayscale:
            final = self.to_grayscale(cropped)
        else:
            final = cropped

        content_ratio, texture_std = self.compute_content_metrics(final)
        
        return PreprocessResult(
            image=final,
            scale_factor=scale_factor,
            rotation_deg=-math.degrees(yaw_rad),
            drone_gsd=drone_gsd,
            target_gsd=target_gsd_eff,
            content_ratio=content_ratio,
            texture_std=texture_std,
            target_gsd_scale=scale_mult,
        )


def test_preprocessor():
    """Test preprocessor with real image and camera intrinsics from VIO config."""
    import os
    import sys
    import random
    
    print("="*60)
    print("Testing VPSImagePreprocessor with Real Camera Parameters")
    print("="*60)
    
    # Paths
    config_path = "configs/config_bell412_dataset3.yaml"
    images_dir = "/Users/france/Downloads/vio_dataset/bell412_dataset3/extracted_data_new/cam_data/camera__image_mono/images"
    output_dir = "vps/test_output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to load VIO config and fisheye rectifier
    fisheye_rectifier = None
    camera_intrinsics = None
    
    try:
        # Add parent directory to path for vio imports
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from vio.config import load_config
        
        if os.path.exists(config_path):
            print(f"\n📁 Loading config: {config_path}")
            config = load_config(config_path)
            
            # Load YAML directly for camera intrinsics
            import yaml
            with open(config_path, 'r') as f:
                config_yaml = yaml.safe_load(f)
            
            # Get camera intrinsics from YAML
            if 'camera' in config_yaml:
                cam = config_yaml['camera']
                camera_intrinsics = {
                    'mu': cam.get('mu', 500),
                    'mv': cam.get('mv', 500),
                    'u0': cam.get('u0', 720),
                    'v0': cam.get('v0', 540),
                    'w': cam.get('image_width', 1440),
                    'h': cam.get('image_height', 1080),
                    'k2': cam.get('k2', 0),
                    'k3': cam.get('k3', 0),
                    'k4': cam.get('k4', 0),
                    'k5': cam.get('k5', 0),
                }
                print(f"  Camera intrinsics loaded:")
                print(f"    Focal: mu={camera_intrinsics['mu']:.2f}, mv={camera_intrinsics['mv']:.2f}")
                print(f"    Principal: u0={camera_intrinsics['u0']:.2f}, v0={camera_intrinsics['v0']:.2f}")
                print(f"    Size: {camera_intrinsics['w']}x{camera_intrinsics['h']}")
                print(f"    Distortion: k2={camera_intrinsics['k2']:.4f}, k3={camera_intrinsics['k3']:.4f}")
            
            # Create fisheye rectifier
            try:
                from vio.fisheye_rectifier import create_rectifier_from_config
                # Wrap in KB_PARAMS format expected by create_rectifier_from_config
                rectifier_config = {'KB_PARAMS': camera_intrinsics}
                src_size = (camera_intrinsics['w'], camera_intrinsics['h'])
                fisheye_rectifier = create_rectifier_from_config(
                    rectifier_config, 
                    src_size=src_size,
                    fov_deg=90.0
                )
                print(f"  ✅ Fisheye rectifier created (FOV=90°)")
            except Exception as e:
                print(f"  ⚠️ Fisheye rectifier failed: {e}")
        else:
            print(f"⚠️ Config not found: {config_path}")
            
    except ImportError as e:
        print(f"⚠️ VIO module not available: {e}")
        print("  Using fallback camera intrinsics...")
    
    # Fallback intrinsics if VIO not available
    if camera_intrinsics is None:
        camera_intrinsics = {'mu': 350.0, 'mv': 350.0, 'u0': 720, 'v0': 540, 'w': 1440, 'h': 1080}
        print(f"  Using fallback intrinsics: fx={camera_intrinsics['mu']}")
    
    # Load image
    selected_file = None
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if image_files:
            selected_file = random.choice(image_files)
            image_path = os.path.join(images_dir, selected_file)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                return
            print(f"\n📷 Selected image: {selected_file}")
    
    if selected_file is None:
        print("Using synthetic image...")
        img = np.random.randint(0, 255, (1080, 1440, 3), dtype=np.uint8)
        cv2.circle(img, (720, 540), 100, (0, 255, 0), -1)
        selected_file = "synthetic.jpg"
    
    # Create preprocessor with real parameters
    preprocessor = VPSImagePreprocessor(
        fisheye_rectifier=fisheye_rectifier,
        camera_intrinsics=camera_intrinsics,
        output_size=(512, 512)
    )
    
    print(f"  Focal length used: {preprocessor.focal_px:.2f} px")
    print(f"  Using undistortion: {'✅ Yes' if fisheye_rectifier else '❌ No'}")
    
    # Test parameters
    yaw_angles = [0, math.pi/4, math.pi/2, math.pi]  # 0°, 45°, 90°, 180°
    altitude_m = 100.0
    target_gsd = 0.21  # Satellite GSD at zoom 19
    
    print(f"\n📊 Processing Parameters:")
    print(f"  Input shape: {img.shape}")
    print(f"  Altitude: {altitude_m}m")
    print(f"  Target GSD: {target_gsd} m/px (satellite)")
    print(f"  Drone GSD: {preprocessor.compute_drone_gsd(altitude_m):.3f} m/px")
    print("-" * 60)
    
    base_name = os.path.splitext(selected_file)[0]
    
    # Save original
    original_output = os.path.join(output_dir, f"{base_name}_0_original.jpg")
    cv2.imwrite(original_output, img)
    print(f"Original:      {img.shape} → {original_output}")
    
    # Process with different yaw angles
    for yaw_rad in yaw_angles:
        yaw_deg = math.degrees(yaw_rad)
        
        result = preprocessor.preprocess(
            img=img,
            yaw_rad=yaw_rad,
            altitude_m=altitude_m,
            target_gsd=target_gsd,
            grayscale=False
        )
        
        output_file = os.path.join(output_dir, f"{base_name}_yaw{int(yaw_deg):03d}.jpg")
        cv2.imwrite(output_file, result.image)
        
        print(f"Yaw {yaw_deg:3.0f}°:      {result.image.shape}, "
              f"GSD={result.drone_gsd:.3f}→{result.target_gsd:.3f}, "
              f"scale={result.scale_factor:.3f}")
    
    # Save undistorted-only (no rotation/scaling) for comparison
    if fisheye_rectifier:
        undistorted = preprocessor.undistort(img)
        undist_output = os.path.join(output_dir, f"{base_name}_1_undistorted.jpg")
        cv2.imwrite(undist_output, undistorted)
        print(f"\nUndistorted:   {undistorted.shape} → {undist_output}")
    
    # Save grayscale final
    result_gray = preprocessor.preprocess(
        img=img, yaw_rad=0, altitude_m=altitude_m, target_gsd=target_gsd, grayscale=True
    )
    gray_output = os.path.join(output_dir, f"{base_name}_grayscale.jpg")
    cv2.imwrite(gray_output, result_gray.image)
    print(f"Grayscale:     {result_gray.image.shape} → {gray_output}")
    
    print("\n" + "=" * 60)
    print(f"✅ Test passed - outputs saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    test_preprocessor()
