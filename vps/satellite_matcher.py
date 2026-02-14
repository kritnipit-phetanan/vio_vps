#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Satellite Matcher for VPS

Image matching between drone camera and satellite imagery using LightGlue + SuperPoint.
Falls back to ORB if LightGlue is not available.

Author: VIO project
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

# Try to import LightGlue (optional dependency)
try:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import numpy_image_to_torch, rbd
    HAS_LIGHTGLUE = True
except ImportError:
    HAS_LIGHTGLUE = False
    torch = None


@dataclass
class MatchResult:
    """Result of satellite matching."""
    success: bool
    H: Optional[np.ndarray]        # 3x3 Homography matrix (drone → satellite)
    num_matches: int               # Total matches found
    num_inliers: int               # Inliers after RANSAC
    reproj_error: float            # Mean reprojection error (pixels)
    confidence: float              # 0-1 confidence score
    offset_px: Tuple[float, float] # (dx, dy) pixel offset of drone center in sat image
    keypoints_drone: Optional[np.ndarray]     # Nx2 keypoints in drone image
    keypoints_sat: Optional[np.ndarray]       # Nx2 keypoints in satellite image


class SatelliteMatcher:
    """
    Matches drone camera images against satellite imagery.
    
    Uses LightGlue + SuperPoint for robust matching, with ORB fallback.
    Computes homography to estimate position offset.
    """
    
    def __init__(self, 
                 device: str = 'cuda',
                 max_keypoints: int = 2048,
                 min_inliers: int = 20,
                 reproj_threshold: float = 3.0):
        """
        Initialize matcher.
        
        Args:
            device: 'cuda' or 'cpu' for LightGlue
            max_keypoints: Maximum keypoints to extract
            min_inliers: Minimum inliers for valid match
            reproj_threshold: RANSAC reprojection threshold (pixels)
        """
        self.device = device
        self.max_keypoints = max_keypoints
        self.min_inliers = min_inliers
        self.reproj_threshold = reproj_threshold
        
        self.use_lightglue = HAS_LIGHTGLUE
        
        if self.use_lightglue:
            self._init_lightglue()
        else:
            self._init_orb_fallback()
    
    def _init_lightglue(self):
        """Initialize LightGlue + SuperPoint."""
        print("[SatelliteMatcher] Initializing LightGlue + SuperPoint...")
        
        # Check CUDA availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("[SatelliteMatcher] CUDA not available, using CPU")
            self.device = 'cpu'
        
        # SuperPoint feature extractor
        self.extractor = SuperPoint(max_num_keypoints=self.max_keypoints).eval()
        self.extractor = self.extractor.to(self.device)
        
        # LightGlue matcher
        self.matcher = LightGlue(features='superpoint').eval()
        self.matcher = self.matcher.to(self.device)
        
        print(f"[SatelliteMatcher] LightGlue ready on {self.device}")
    
    def _init_orb_fallback(self):
        """Initialize ORB fallback matcher."""
        print("[SatelliteMatcher] Using ORB fallback (LightGlue not available)")
        self.orb = cv2.ORB_create(nfeatures=4000)  # Increased from 2048
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_test_threshold = 0.78  # Balanced setting for VPS matching
    
    def extract_features_lightglue(self, img: np.ndarray) -> Dict[str, "torch.Tensor"]:
        """Extract SuperPoint features."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Convert to torch tensor
        img_tensor = numpy_image_to_torch(img).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.extractor.extract(img_tensor)
        
        return features
    
    def match_lightglue(self, 
                        drone_img: np.ndarray, 
                        sat_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match images using LightGlue.
        
        Returns:
            (kp_drone, kp_sat, match_confidence) - matched keypoint pairs
        """
        # Extract features
        feats0 = self.extract_features_lightglue(drone_img)
        feats1 = self.extract_features_lightglue(sat_img)
        
        # Match
        with torch.no_grad():
            matches_dict = self.matcher({'image0': feats0, 'image1': feats1})
        
        # Get matched keypoints
        feats0, feats1, matches_dict = [rbd(x) for x in [feats0, feats1, matches_dict]]
        
        matches = matches_dict['matches']
        kpts0 = feats0['keypoints']
        kpts1 = feats1['keypoints']
        
        # Convert to numpy
        if len(matches) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Get matched points
        valid = matches > -1
        mkpts0 = kpts0[valid].cpu().numpy()
        mkpts1 = kpts1[matches[valid]].cpu().numpy()
        
        # Get confidence scores if available
        if 'scores' in matches_dict:
            scores = matches_dict['scores'][valid].cpu().numpy()
        else:
            scores = np.ones(len(mkpts0))
        
        return mkpts0, mkpts1, scores
    
    def extract_features_orb(self, img: np.ndarray) -> Tuple[List, np.ndarray]:
        """Extract ORB features (fallback)."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        kp, desc = self.orb.detectAndCompute(img, None)
        return kp, desc
    
    def match_orb(self, 
                  drone_img: np.ndarray, 
                  sat_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match images using ORB (fallback).
        
        Returns:
            (kp_drone, kp_sat, match_scores) - matched keypoint pairs
        """
        kp0, desc0 = self.extract_features_orb(drone_img)
        kp1, desc1 = self.extract_features_orb(sat_img)
        
        if desc0 is None or desc1 is None or len(kp0) < 10 or len(kp1) < 10:
            return np.array([]), np.array([]), np.array([])
        
        # KNN match
        matches = self.bf_matcher.knnMatch(desc0, desc1, k=2)
        
        # Ratio test
        good_matches = []
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < self.ratio_test_threshold * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            return np.array([]), np.array([]), np.array([])

        # Enforce one-to-one correspondence on satellite keypoints.
        # KNN+ratio can still produce many-to-one matches (multiple drone points
        # mapping to the same satellite keypoint), which causes degenerate
        # homography and misleading debug visualizations.
        best_by_train: Dict[int, Any] = {}
        for m in good_matches:
            prev = best_by_train.get(int(m.trainIdx))
            if prev is None or m.distance < prev.distance:
                best_by_train[int(m.trainIdx)] = m

        # Keep best per query index as an additional safety guard.
        best_by_query: Dict[int, Any] = {}
        for m in best_by_train.values():
            prev = best_by_query.get(int(m.queryIdx))
            if prev is None or m.distance < prev.distance:
                best_by_query[int(m.queryIdx)] = m

        good_matches = sorted(best_by_query.values(), key=lambda mm: mm.distance)
        if len(good_matches) < 4:
            return np.array([]), np.array([]), np.array([])
        
        # Extract matched points
        pts0 = np.array([kp0[m.queryIdx].pt for m in good_matches])
        pts1 = np.array([kp1[m.trainIdx].pt for m in good_matches])
        scores = np.array([1.0 - m.distance / 256.0 for m in good_matches])
        
        return pts0, pts1, scores
    
    def match(self, 
              drone_img: np.ndarray, 
              sat_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match drone image against satellite image.
        
        Args:
            drone_img: Preprocessed drone image (grayscale preferred)
            sat_img: Satellite map patch (same GSD, North-Up)
            
        Returns:
            (kp_drone, kp_sat, scores) - Matched keypoint pairs and confidence scores
        """
        if self.use_lightglue:
            return self.match_lightglue(drone_img, sat_img)
        else:
            return self.match_orb(drone_img, sat_img)
    
    def estimate_homography(self, 
                            pts_drone: np.ndarray, 
                            pts_sat: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
        """
        Estimate homography using RANSAC.
        
        Args:
            pts_drone: Nx2 points in drone image
            pts_sat: Nx2 points in satellite image
            
        Returns:
            (H, inlier_mask, reproj_error)
        """
        if len(pts_drone) < 4:
            return None, np.array([]), float('inf')
        
        # Find homography with RANSAC
        H, mask = cv2.findHomography(
            pts_drone, pts_sat,
            cv2.RANSAC,
            ransacReprojThreshold=self.reproj_threshold
        )
        
        if H is None or mask is None:
            return None, np.array([]), float('inf')
        
        mask = mask.ravel().astype(bool)
        
        # Compute reprojection error for inliers
        if np.sum(mask) > 0:
            inlier_pts_drone = pts_drone[mask]
            inlier_pts_sat = pts_sat[mask]
            
            # Project drone points to satellite space
            ones = np.ones((len(inlier_pts_drone), 1))
            pts_h = np.hstack([inlier_pts_drone, ones])
            projected = (H @ pts_h.T).T
            denom = projected[:, 2]
            valid_proj = np.isfinite(denom) & (np.abs(denom) > 1e-9)
            if not np.any(valid_proj):
                return None, np.array([]), float('inf')
            projected_xy = np.full((len(projected), 2), np.nan, dtype=float)
            projected_xy[valid_proj] = projected[valid_proj, :2] / denom[valid_proj, None]
            
            # Compute error
            valid_err = valid_proj & np.all(np.isfinite(inlier_pts_sat), axis=1)
            if not np.any(valid_err):
                return None, np.array([]), float('inf')
            errors = np.linalg.norm(projected_xy[valid_err] - inlier_pts_sat[valid_err], axis=1)
            reproj_error = float(np.mean(errors))
        else:
            reproj_error = float('inf')
        
        return H, mask, reproj_error
    
    def compute_center_offset(self, 
                              H: np.ndarray, 
                              drone_img_size: Tuple[int, int]) -> Tuple[float, float]:
        """
        Compute pixel offset of drone image center in satellite image.
        
        Args:
            H: Homography matrix (drone → satellite)
            drone_img_size: (width, height) of drone image
            
        Returns:
            (dx, dy) pixel offset from satellite image center
        """
        w, h = drone_img_size
        
        # Drone image center
        drone_center = np.array([[w / 2, h / 2, 1.0]])
        
        # Transform to satellite space
        sat_point = (H @ drone_center.T).T
        w_h = float(sat_point[0, 2])
        if (not np.isfinite(w_h)) or abs(w_h) <= 1e-9:
            raise ValueError("Degenerate homography center projection (w≈0)")
        sat_x = float(sat_point[0, 0] / w_h)
        sat_y = float(sat_point[0, 1] / w_h)
        if not (np.isfinite(sat_x) and np.isfinite(sat_y)):
            raise ValueError("Non-finite center projection")
        
        # Offset from satellite image center (assuming same size)
        offset_x = sat_x - w / 2
        offset_y = sat_y - h / 2
        
        return offset_x, offset_y
    
    def match_with_homography(self, 
                              drone_img: np.ndarray, 
                              sat_img: np.ndarray) -> MatchResult:
        """
        Full matching pipeline: feature matching + homography estimation.
        
        Args:
            drone_img: Preprocessed drone image
            sat_img: Satellite map patch
            
        Returns:
            MatchResult with homography and offset
        """
        # Match features
        pts_drone, pts_sat, scores = self.match(drone_img, sat_img)
        
        num_matches = len(pts_drone)
        
        if num_matches < 4:
            return MatchResult(
                success=False,
                H=None,
                num_matches=num_matches,
                num_inliers=0,
                reproj_error=float('inf'),
                confidence=0.0,
                offset_px=(0.0, 0.0),
                keypoints_drone=pts_drone if len(pts_drone) > 0 else None,
                keypoints_sat=pts_sat if len(pts_sat) > 0 else None
            )
        
        # Estimate homography
        H, inlier_mask, reproj_error = self.estimate_homography(pts_drone, pts_sat)
        
        num_inliers = int(np.sum(inlier_mask)) if len(inlier_mask) > 0 else 0
        
        # Check minimum inliers
        if H is None or num_inliers < self.min_inliers:
            return MatchResult(
                success=False,
                H=H,
                num_matches=num_matches,
                num_inliers=num_inliers,
                reproj_error=reproj_error,
                confidence=0.0,
                offset_px=(0.0, 0.0),
                keypoints_drone=pts_drone,
                keypoints_sat=pts_sat
            )
        
        # Reject inlier sets that are too spatially concentrated.
        # This avoids false "good" matches where many correspondences collapse
        # to (almost) one map point.
        inlier_pts_drone = pts_drone[inlier_mask] if len(inlier_mask) > 0 else np.empty((0, 2))
        inlier_pts_sat = pts_sat[inlier_mask] if len(inlier_mask) > 0 else np.empty((0, 2))
        if len(inlier_pts_drone) >= 4 and len(inlier_pts_sat) >= 4:
            min_span_px = max(8.0, 0.02 * float(min(sat_img.shape[0], sat_img.shape[1])))
            span_sat_x = float(np.ptp(inlier_pts_sat[:, 0]))
            span_sat_y = float(np.ptp(inlier_pts_sat[:, 1]))
            span_drone_x = float(np.ptp(inlier_pts_drone[:, 0]))
            span_drone_y = float(np.ptp(inlier_pts_drone[:, 1]))
            unique_sat = int(len(np.unique(np.round(inlier_pts_sat, 1), axis=0)))

            sat_collapsed = (span_sat_x < min_span_px) and (span_sat_y < min_span_px)
            drone_collapsed = (span_drone_x < min_span_px) and (span_drone_y < min_span_px)
            low_uniqueness = unique_sat < max(4, int(0.35 * num_inliers))
            if sat_collapsed or drone_collapsed or low_uniqueness:
                return MatchResult(
                    success=False,
                    H=H,
                    num_matches=num_matches,
                    num_inliers=num_inliers,
                    reproj_error=reproj_error,
                    confidence=0.0,
                    offset_px=(0.0, 0.0),
                    keypoints_drone=inlier_pts_drone,
                    keypoints_sat=inlier_pts_sat
                )

        # Compute center offset
        drone_size = (drone_img.shape[1], drone_img.shape[0])
        try:
            offset_px = self.compute_center_offset(H, drone_size)
        except ValueError:
            return MatchResult(
                success=False,
                H=H,
                num_matches=num_matches,
                num_inliers=num_inliers,
                reproj_error=float('inf'),
                confidence=0.0,
                offset_px=(0.0, 0.0),
                keypoints_drone=pts_drone,
                keypoints_sat=pts_sat
            )
        
        # Compute confidence
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
        reproj_score = max(0, 1 - reproj_error / 10.0)  # 0-1 score
        confidence = inlier_ratio * reproj_score
        
        return MatchResult(
            success=True,
            H=H,
            num_matches=num_matches,
            num_inliers=num_inliers,
            reproj_error=reproj_error,
            confidence=confidence,
            offset_px=offset_px,
            keypoints_drone=pts_drone[inlier_mask] if len(inlier_mask) > 0 else None,
            keypoints_sat=pts_sat[inlier_mask] if len(inlier_mask) > 0 else None
        )
    
    def visualize_matches(self, 
                          drone_img: np.ndarray, 
                          sat_img: np.ndarray,
                          result: MatchResult,
                          output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize matches between drone and satellite images.
        
        Args:
            drone_img: Drone image
            sat_img: Satellite image
            result: MatchResult from match_with_homography
            output_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        # Ensure both images are BGR
        if len(drone_img.shape) == 2:
            drone_img = cv2.cvtColor(drone_img, cv2.COLOR_GRAY2BGR)
        if len(sat_img.shape) == 2:
            sat_img = cv2.cvtColor(sat_img, cv2.COLOR_GRAY2BGR)
        
        # Create side-by-side image
        h1, w1 = drone_img.shape[:2]
        h2, w2 = sat_img.shape[:2]
        
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = drone_img
        vis[:h2, w1:w1+w2] = sat_img
        
        # Draw matches
        if result.keypoints_drone is not None and result.keypoints_sat is not None:
            for i in range(len(result.keypoints_drone)):
                pt1 = tuple(result.keypoints_drone[i].astype(int))
                pt2 = tuple((result.keypoints_sat[i] + np.array([w1, 0])).astype(int))
                
                color = (0, 255, 0) if result.success else (0, 0, 255)
                cv2.circle(vis, pt1, 3, color, -1)
                cv2.circle(vis, pt2, 3, color, -1)
                cv2.line(vis, pt1, pt2, color, 1)
        
        # Add text
        status = "SUCCESS" if result.success else "FAILED"
        text = f"{status}: {result.num_inliers}/{result.num_matches} inliers, err={result.reproj_error:.2f}px"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if result.success:
            offset_text = f"Offset: ({result.offset_px[0]:.1f}, {result.offset_px[1]:.1f}) px"
            cv2.putText(vis, offset_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis)
        
        return vis


def test_matcher():
    """Test matcher with real images from bell412 dataset and satellite tiles."""
    import os
    import sys
    import random
    
    print("=" * 60)
    print("Testing SatelliteMatcher with Real Data")
    print("=" * 60)
    
    # Paths
    mbtiles_path = "mission.mbtiles"
    config_path = "configs/config_bell412_dataset3.yaml"
    images_dir = "/Users/france/Downloads/vio_dataset/bell412_dataset3/extracted_data_new/cam_data/camera__image_mono/images"
    output_dir = "vps/match_test_output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Add parent for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check if we have real data
    use_real_data = os.path.exists(mbtiles_path) and os.path.exists(images_dir)
    
    if use_real_data:
        print(f"  MBTiles: {mbtiles_path}")
        print(f"  Images: {images_dir}")
        
        # Load tile cache
        from vps.tile_cache import TileCache
        tile_cache = TileCache(mbtiles_path)
        
        # Load random drone image FIRST, then sync with PPK
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        if not image_files:
            use_real_data = False
        else:
            selected_file = random.choice(image_files)
            drone_img_color = cv2.imread(os.path.join(images_dir, selected_file))
            print(f"  Drone image: {selected_file}")
            
            # Extract timestamp from filename (nanoseconds)
            try:
                img_timestamp_ns = int(selected_file.split('.')[0])
                img_timestamp_s = img_timestamp_ns / 1e9  # Convert to seconds
                print(f"  Image timestamp: {img_timestamp_s:.6f} s")
            except:
                img_timestamp_s = None
                print(f"  ⚠️ Could not parse timestamp from filename")
        
        # Get position and yaw from PPK synchronized with image timestamp
        test_lat = 45.315721787845
        test_lon = -75.670671305696
        test_yaw_deg = 0.0  # Default: assume north-facing
        
        ppk_path = "/Users/france/Downloads/vio_dataset/bell412_dataset3/bell412_dataset3_frl.pos"
        if os.path.exists(ppk_path) and use_real_data:
            try:
                import pandas as pd
                # PPK format: GPST (week seconds), lat, lon, height, ...
                ppk_df = pd.read_csv(ppk_path, comment='%', sep=r'\s+', header=None)
                
                if img_timestamp_s is not None:
                    # Find closest timestamp in PPK
                    # Column 1 is GPS week seconds, need to match with image timestamp
                    ppk_times = ppk_df.iloc[:, 1].values  # GPS week seconds
                    
                    # Image timestamp is in Unix epoch, PPK is GPS week seconds
                    # We need to find the offset - use first image timestamp
                    # For now, just use relative position in the flight
                    
                    # Get all image timestamps
                    all_img_files = sorted(image_files)
                    all_timestamps = []
                    for f in all_img_files:
                        try:
                            all_timestamps.append(int(f.split('.')[0]))
                        except:
                            pass
                    
                    
                    # --- ACCURATE TIMESTAMP SYNC ---
                    # Parse PPK timestamps (Datum + Time columns)
                    try:
                        # Combine Date (col 0) and Time (col 1)
                        # PPK format: 2022/05/25 21:07:15.600
                        ppk_datetimes = pd.to_datetime(ppk_df.iloc[:, 0] + ' ' + ppk_df.iloc[:, 1])
                        
                        # Convert both to common time base (Unix timestamp in seconds)
                        # Note: GPST is different from UTC by ~18s, but typically datasets align them or use same base
                        ppk_timestamps = ppk_datetimes.view(np.int64) / 1e9 # seconds
                        
                        img_ts_seconds = img_timestamp_ns / 1e9
                        
                        # Find index with minimum time difference
                        time_diffs = np.abs(ppk_timestamps - img_ts_seconds)
                        ppk_idx = np.argmin(time_diffs)
                        min_diff = time_diffs[ppk_idx]
                        
                        print(f"  Timestamp Sync: Image={img_ts_seconds:.3f} vs PPK={ppk_timestamps[ppk_idx]:.3f} (Diff={min_diff:.3f}s)")
                        
                        if min_diff > 1.0:
                            print(f"  ⚠️ Warning: Large timestamp difference ({min_diff:.3f}s). Sync may be off.")
                        
                        test_lat = ppk_df.iloc[ppk_idx, 2]
                        test_lon = ppk_df.iloc[ppk_idx, 3]
                        
                        # Extract yaw from column 16 (yaw(deg))
                        try:
                            test_yaw_deg = float(ppk_df.iloc[ppk_idx, 16])
                        except:
                            test_yaw_deg = 0.0
                            
                        # Extract altitude
                        try:
                            test_height_m = float(ppk_df.iloc[ppk_idx, 4])
                            ground_elevation = 80.0
                            test_altitude_agl = max(50.0, test_height_m - ground_elevation)
                        except:
                            test_altitude_agl = 100.0
                            
                        print(f"  PPK sync: row {ppk_idx}/{len(ppk_df)}")
                        print(f"  Position: ({test_lat:.12f}, {test_lon:.12f})")
                        print(f"  Yaw: {test_yaw_deg:.1f}°, Altitude: {test_altitude_agl:.1f}m AGL")
                        
                    except Exception as e:
                        print(f"  ⚠️ Accurate sync failed: {e}")
                        print("  Falling back to approximate sync...")
                        # Fallback to old logic or just fail
                        # For now, simpler fallback
                        ppk_idx = 0
                        test_lat = ppk_df.iloc[0, 2]
                        test_lon = ppk_df.iloc[0, 3]
                        test_yaw_deg = 0.0
                        test_altitude_agl = 100.0
                else:
                    # Fallback to first row
                    test_lat = ppk_df.iloc[0, 2]
                    test_lon = ppk_df.iloc[0, 3]
                    test_altitude_agl = 100.0  # Default altitude
                    print(f"  Position (fallback): ({test_lat:.6f}, {test_lon:.6f})")
            except Exception as e:
                print(f"  PPK sync failed: {e}")
                print(f"  Using default position: ({test_lat:.6f}, {test_lon:.6f})")
        
        # Get satellite patch at synchronized position
        map_patch = tile_cache.get_map_patch(test_lat, test_lon, patch_size_px=512)
        if map_patch is None:
            print("  ⚠️ No tile coverage at this position, using synthetic test")
            use_real_data = False
            tile_info = None
        else:
            sat_img = map_patch.image
            if len(sat_img.shape) == 3:
                sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2GRAY)
            
            # Calculate center tile coordinates for filename
            from vps.tile_cache import lat_lon_to_tile
            center_tx, center_ty = lat_lon_to_tile(test_lat, test_lon, tile_cache.zoom)
            tile_info = f"tile_{tile_cache.zoom}_{center_tx}_{center_ty}"
            
            print(f"  Satellite patch: {sat_img.shape}, GSD={map_patch.meters_per_pixel:.3f} m/px")
            print(f"  Center tile: {tile_info}")
        
        # Preprocess drone image
        if use_real_data:
            
            # Preprocess drone image
            try:
                from vps.image_preprocessor import VPSImagePreprocessor
                import yaml
                
                # Load camera intrinsics
                with open(config_path, 'r') as f:
                    config_yaml = yaml.safe_load(f)
                
                cam = config_yaml.get('camera', {})
                camera_intrinsics = {
                    'mu': cam.get('mu', 500),
                    'mv': cam.get('mv', 500),
                    'u0': cam.get('u0', 720),
                    'v0': cam.get('v0', 540),
                    'w': cam.get('image_width', 1440),
                    'h': cam.get('image_height', 1080),
                }
                
                # Create rectifier
                try:
                    from vio.fisheye_rectifier import create_rectifier_from_config
                    rectifier_config = {'KB_PARAMS': camera_intrinsics}
                    src_size = (camera_intrinsics['w'], camera_intrinsics['h'])
                    rectifier = create_rectifier_from_config(rectifier_config, src_size, fov_deg=90.0)
                except:
                    rectifier = None
                
                preprocessor = VPSImagePreprocessor(
                    fisheye_rectifier=rectifier,
                    camera_intrinsics=camera_intrinsics,
                    output_size=(512, 512)
                )
                
                # Convert yaw from degrees to radians
                # PPK yaw is in NED convention: 0=North, positive=clockwise (East)
                import math
                
                # Load Configuration and DEM
                import yaml
                import math
                from vps.dem_handler import DEMHandler
                
                config_path = "/Users/france/Downloads/vio_vps/configs/config_bell412_dataset3.yaml"
                dem_path = "/Users/france/Downloads/vio_dataset/bell412_dataset3/Copernicus_DSM_10_N45_00_W076_00_DEM.tif"
                
                # 1. Calculate Camera Offset from Config
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Get Nadir camera extrinsics (Body -> Camera transform)
                    # R matrix is top-left 3x3 of the 4x4 transform
                    t_matrix = config['extrinsics']['nadir']['transform']
                    R00 = t_matrix[0][0]
                    R10 = t_matrix[1][0]
                    
                    # Calculate Yaw rotation (rotation around Z-axis)
                    # Yaw = atan2(R[1,0], R[0,0])
                    # Note: This gives the azimuth of the Camera X-axis (Right).
                    # We need the azimuth of the Camera Up-axis (-Y) for North-Up rotation.
                    # Up is 90 deg CCW from Right (in standard image coords)? 
                    # No, Up is -Y, Right is +X. Angle from X to -Y is -90 deg.
                    # So subtract 90 degrees.
                    yaw_offset_rad = math.atan2(R10, R00) - math.radians(90)
                    CAMERA_YAW_OFFSET_DEG = math.degrees(yaw_offset_rad)
                    print(f"  [Config] Camera Extrinsics Loaded. Yaw Offset: {CAMERA_YAW_OFFSET_DEG:.2f}°")
                except Exception as e:
                    print(f"  ⚠️ Failed to load config/extrinsics: {e}")
                    CAMERA_YAW_OFFSET_DEG = 90.0  # Fallback
                    
                # 2. Key: Use DEM for Ground Elevation
                dem_handler = None
                try:
                    if os.path.exists(dem_path):
                        dem_handler = DEMHandler(dem_path)
                        # Re-calculate Altitude AGL using DEM if available
                        elev = dem_handler.get_elevation(test_lat, test_lon)
                        if elev is not None:
                            ground_elevation = elev
                            # Ensure test_height_m exists (from PPK sync)
                            if 'test_height_m' in locals():
                                test_altitude_agl = max(50.0, test_height_m - ground_elevation)
                                print(f"  [DEM] Elevation: {elev:.2f}m -> updated Altitude AGL: {test_altitude_agl:.2f}m")
                            else:
                                print(f"  [DEM] Elevation: {elev:.2f}m (No PPK height available)")
                    else:
                        print(f"  ⚠️ DEM file not found: {dem_path}")
                except Exception as e:
                    print(f"  ⚠️ Failed to use DEM: {e}")

                adjusted_yaw_deg = test_yaw_deg + CAMERA_YAW_OFFSET_DEG
                yaw_rad = math.radians(adjusted_yaw_deg)
        
        # Update altitude extraction logic to use DEM inside the loop
        # (This replacement targets the parameter setup block before preprocessing)
                
                result = preprocessor.preprocess(
                    img=drone_img_color,
                    yaw_rad=yaw_rad,  # Use PPK yaw for North-Up rotation
                    altitude_m=test_altitude_agl,  # Use PPK altitude for correct GSD
                    target_gsd=map_patch.meters_per_pixel,
                    grayscale=True
                )
                drone_img = result.image
                print(f"  Preprocessed: {drone_img.shape}, yaw={test_yaw_deg:.1f}°, alt={test_altitude_agl:.0f}m")
            except Exception as e:
                print(f"  ⚠️ Preprocessing failed: {e}")
                drone_img = cv2.cvtColor(drone_img_color, cv2.COLOR_BGR2GRAY)
                drone_img = cv2.resize(drone_img, (512, 512))
        else:
            use_real_data = False
    
    if not use_real_data:
        print("Using synthetic test images...")
        # Create simple test images with known features
        drone_img = np.zeros((512, 512), dtype=np.uint8)
        sat_img = np.zeros((512, 512), dtype=np.uint8)
        
        # Add checkerboard pattern
        for i in range(0, 512, 64):
            for j in range(0, 512, 64):
                if (i // 64 + j // 64) % 2 == 0:
                    drone_img[i:i+32, j:j+32] = 255
                    si, sj = i + 10, j + 15
                    if si + 32 < 512 and sj + 32 < 512:
                        sat_img[si:si+32, sj:sj+32] = 255
    
    print("-" * 60)
    
    # Create matcher
    matcher = SatelliteMatcher(device='cpu', min_inliers=4)
    
    # Match
    result = matcher.match_with_homography(drone_img, sat_img)
    
    print(f"\n📊 Match Results:")
    print(f"  Matcher type: {'LightGlue' if matcher.use_lightglue else 'ORB'}")
    print(f"  Success: {result.success}")
    print(f"  Matches: {result.num_matches}")
    print(f"  Inliers: {result.num_inliers}")
    print(f"  Reproj error: {result.reproj_error:.2f} px")
    print(f"  Confidence: {result.confidence:.3f}")
    
    # Calculate offset in meters
    import math
    dx, dy = result.offset_px
    dist_px = math.sqrt(dx**2 + dy**2)
    dist_m = dist_px * 0.21  # Approx 0.21 m/px GSD
    print(f"  Offset: ({dx:.1f}, {dy:.1f}) px -> Approx {dist_m:.2f} meters")
    
    if dist_m > 50.0 and result.success:
         print(f"  ⚠️ Match found but offset is large (>50m). Potential false positive or large GPS error.")
    
    # Save visualization with unique filename based on drone image
    # Use selected_file if available, otherwise use timestamp
    if use_real_data and 'selected_file' in dir():
        base_name = selected_file.split('.')[0]
    else:
        import time
        base_name = f"synthetic_{int(time.time())}"
    
    vis_path = os.path.join(output_dir, f"match_{base_name}.jpg")
    vis = matcher.visualize_matches(drone_img, sat_img, result, vis_path)
    print(f"\n  Visualization saved: {vis_path}")
    
    # Also save individual images with unique names
    # Also save individual images with unique names
    if use_real_data:
        drone_filename = f"drone_{base_name}_{test_lat:.12f}_{test_lon:.12f}.jpg"
    else:
        drone_filename = f"drone_{base_name}.jpg"
        
    cv2.imwrite(os.path.join(output_dir, drone_filename), drone_img)
    
    # Save satellite with tile info for cross-checking
    sat_filename = f"sat_{base_name}_{tile_info}.jpg" if (use_real_data and tile_info) else f"sat_{base_name}.jpg"
    cv2.imwrite(os.path.join(output_dir, sat_filename), sat_img)
    print(f"  Saved: {drone_filename}, {sat_filename}")
    
    print("\n" + "=" * 60)
    if result.success:
        print("✅ Matcher test passed")
    else:
        print("⚠️ Match failed (expected for mismatched drone/satellite imagery)")
    print("=" * 60)


if __name__ == "__main__":
    test_matcher()
