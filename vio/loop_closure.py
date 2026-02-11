#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loop Closure Detection & Correction Module

Lightweight loop closure for yaw drift correction:
1. Store keyframes with position, yaw, and ORB descriptors
2. Detect loop when returning to visited location (position proximity)
3. Match visual features to confirm loop
4. Compute relative yaw and apply correction

Key insight: For helicopter returning to starting position, we mainly need
to correct YAW drift. Position will be corrected automatically once yaw is fixed.

TUNED FOR PARTIAL REVISITS:
- Lower position threshold (30m instead of 50m) to catch near-passes
- Lower minimum inliers (15 instead of 20) for partial view overlap
- More aggressive keyframe addition for better coverage

Author: VIO project
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List

from .math_utils import quaternion_to_yaw


class LoopClosureDetector:
    """
    Lightweight loop closure detector for yaw drift correction.
    
    Strategy:
    - Store sparse keyframes (every N meters or M degrees rotation)
    - When estimated position is near a stored keyframe, attempt visual matching
    - If match successful, compute yaw correction from relative pose
    - Apply yaw correction to EKF state
    
    Key insight: For helicopter returning to starting position, we mainly need
    to correct YAW drift. Position will be corrected automatically once yaw is fixed.
    
    TUNED FOR PARTIAL REVISITS:
    - Lower position threshold (30m instead of 50m) to catch near-passes
    - Lower minimum inliers (15 instead of 20) for partial view overlap
    - More aggressive keyframe addition for better coverage
    """
    
    def __init__(self, 
                 position_threshold: float = 30.0,    # REDUCED: catch partial revisits
                 min_keyframe_dist: float = 15.0,     # REDUCED: denser keyframe coverage
                 min_keyframe_yaw: float = 20.0,      # REDUCED: more keyframes during turns
                 min_frame_gap: int = 50,             # REDUCED: faster loop detection
                 min_match_ratio: float = 0.12,       # REDUCED: accept partial overlaps
                 min_inliers: int = 15):              # REDUCED: accept partial matches
        """
        Initialize loop closure detector.
        
        Args:
            position_threshold: Distance threshold for loop candidate detection (m)
            min_keyframe_dist: Minimum distance between keyframes (m)
            min_keyframe_yaw: Minimum yaw change for new keyframe (degrees)
            min_frame_gap: Minimum frames between loop closure candidates
            min_match_ratio: Minimum inlier ratio for geometry check
            min_inliers: Minimum number of inliers for valid match
        """
        self.position_threshold = position_threshold
        self.min_keyframe_dist = min_keyframe_dist
        self.min_keyframe_yaw = np.radians(min_keyframe_yaw)
        self.min_frame_gap = min_frame_gap
        self.min_match_ratio = min_match_ratio
        self.min_inliers = min_inliers
        
        # Keyframe database: list of {frame_idx, position, yaw, descriptors, keypoints}
        self.keyframes: List[dict] = []
        
        # ORB detector for loop closure (separate from VIO front-end)
        # Increased features for better matching in partial overlaps
        self.orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
        
        # BFMatcher for descriptor matching
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Statistics
        self.stats = {
            'keyframes_added': 0,
            'loop_checks': 0,
            'loop_candidates': 0,
            'loop_detected': 0,
            'loop_rejected_few_matches': 0,
            'loop_rejected_geometry': 0,
            'yaw_corrections_applied': 0,
            'total_yaw_correction': 0.0,
        }
        
        # Last keyframe info (to avoid adding too many)
        self.last_kf_position: Optional[np.ndarray] = None
        self.last_kf_yaw: Optional[float] = None
        self.last_kf_frame: int = -1000
        
    def should_add_keyframe(self, position: np.ndarray, yaw: float, frame_idx: int) -> bool:
        """
        Check if current frame should be stored as keyframe.
        
        Args:
            position: Current position [x, y, z]
            yaw: Current yaw angle (radians)
            frame_idx: Current frame index
            
        Returns:
            True if keyframe should be added
        """
        if self.last_kf_position is None:
            return True
            
        # Distance from last keyframe
        dist = np.linalg.norm(position[:2] - self.last_kf_position[:2])
        
        # Yaw change from last keyframe
        yaw_diff = abs(np.arctan2(np.sin(yaw - self.last_kf_yaw), 
                                   np.cos(yaw - self.last_kf_yaw)))
        
        # Frame gap
        frame_gap = frame_idx - self.last_kf_frame
        
        # Add keyframe if sufficient motion or rotation
        if dist > self.min_keyframe_dist or yaw_diff > self.min_keyframe_yaw:
            if frame_gap > 20:  # At least 20 frames apart
                return True
                
        return False
    
    def add_keyframe(self, frame_idx: int, position: np.ndarray, yaw: float, 
                     gray_image: np.ndarray) -> None:
        """
        Add a new keyframe to the database.
        
        Args:
            frame_idx: Frame index
            position: Position [x, y, z]
            yaw: Yaw angle (radians)
            gray_image: Grayscale image for feature extraction
        """
        # Detect ORB features
        keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
        
        if descriptors is None or len(keypoints) < 50:
            return  # Not enough features
            
        # Store keyframe
        kf = {
            'frame_idx': frame_idx,
            'position': position.copy(),
            'yaw': yaw,
            'keypoints': keypoints,
            'descriptors': descriptors,
        }
        self.keyframes.append(kf)
        
        # Update last keyframe info
        self.last_kf_position = position.copy()
        self.last_kf_yaw = yaw
        self.last_kf_frame = frame_idx
        
        self.stats['keyframes_added'] += 1
        
        if self.stats['keyframes_added'] % 10 == 0:
            print(f"[LOOP] Added keyframe #{self.stats['keyframes_added']} at frame {frame_idx}, "
                  f"pos=({position[0]:.1f}, {position[1]:.1f}), yaw={np.degrees(yaw):.1f}°")
    
    def find_loop_candidates(self, position: np.ndarray, frame_idx: int) -> List[int]:
        """
        Find keyframes that are close to current position.
        
        Args:
            position: Current position [x, y, z]
            frame_idx: Current frame index
            
        Returns:
            List of keyframe indices that are loop candidates
        """
        candidates = []
        
        for i, kf in enumerate(self.keyframes):
            # Skip recent keyframes (need sufficient time gap)
            if frame_idx - kf['frame_idx'] < self.min_frame_gap:
                continue
                
            # Check position proximity
            dist = np.linalg.norm(position[:2] - kf['position'][:2])
            if dist < self.position_threshold:
                candidates.append(i)
                
        return candidates
    
    def match_keyframe(self, gray_image: np.ndarray, kf_idx: int, 
                       K: np.ndarray) -> Optional[Tuple[float, int]]:
        """
        Match current image with keyframe and compute relative yaw.
        
        Args:
            gray_image: Current grayscale image
            kf_idx: Index of keyframe to match against
            K: Camera intrinsic matrix (3x3)
        
        Returns:
            (yaw_correction, num_inliers) or None if match failed
        """
        kf = self.keyframes[kf_idx]
        
        # Detect ORB in current frame
        keypoints_curr, descriptors_curr = self.orb.detectAndCompute(gray_image, None)
        
        if descriptors_curr is None or len(keypoints_curr) < 30:
            return None
            
        # Match descriptors
        try:
            matches = self.matcher.knnMatch(kf['descriptors'], descriptors_curr, k=2)
        except cv2.error:
            return None
            
        # Apply Lowe's ratio test
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # Check minimum matches
        if len(good_matches) < self.min_inliers:
            self.stats['loop_rejected_few_matches'] += 1
            return None
            
        # Get matched point coordinates
        pts_kf = np.float32([kf['keypoints'][m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([keypoints_curr[m.trainIdx].pt for m in good_matches])
        
        # Compute Essential matrix with RANSAC
        try:
            E, mask = cv2.findEssentialMat(pts_kf, pts_curr, K, method=cv2.RANSAC, 
                                           prob=0.999, threshold=1.0)
        except cv2.error:
            self.stats['loop_rejected_geometry'] += 1
            return None
            
        if E is None or mask is None:
            self.stats['loop_rejected_geometry'] += 1
            return None
            
        # Count inliers
        num_inliers = int(mask.sum())
        inlier_ratio = num_inliers / len(good_matches)
        
        if num_inliers < self.min_inliers or inlier_ratio < self.min_match_ratio:
            self.stats['loop_rejected_geometry'] += 1
            return None
            
        # Recover pose from Essential matrix
        try:
            retval, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_kf, pts_curr, K, mask=mask)
        except cv2.error:
            self.stats['loop_rejected_geometry'] += 1
            return None
            
        if retval < self.min_inliers // 2:
            self.stats['loop_rejected_geometry'] += 1
            return None
            
        # Extract yaw from rotation matrix
        # R_rel transforms from keyframe to current frame
        # Yaw = atan2(R[1,0], R[0,0]) for Z-up convention
        yaw_rel = np.arctan2(R_rel[1, 0], R_rel[0, 0])
        
        return yaw_rel, num_inliers
    
    def check_loop_closure(self, frame_idx: int, position: np.ndarray, yaw: float,
                           gray_image: np.ndarray, K: np.ndarray) -> Optional[Tuple[float, int, int]]:
        """
        Check for loop closure and return yaw correction if found.
        
        Args:
            frame_idx: Current frame index
            position: Current estimated position [x, y, z]
            yaw: Current estimated yaw [rad]
            gray_image: Current grayscale image
            K: Camera intrinsic matrix
            
        Returns:
            (yaw_correction, num_inliers, loop_kf_idx) or None if no loop detected
        """
        self.stats['loop_checks'] += 1
        
        # Find candidate keyframes
        candidates = self.find_loop_candidates(position, frame_idx)
        
        if len(candidates) == 0:
            return None
            
        self.stats['loop_candidates'] += len(candidates)
        
        # Try matching with each candidate (closest first)
        candidates_with_dist = [(i, np.linalg.norm(position[:2] - self.keyframes[i]['position'][:2])) 
                                for i in candidates]
        candidates_with_dist.sort(key=lambda x: x[1])
        
        for kf_idx, dist in candidates_with_dist[:3]:  # Try top 3 closest
            result = self.match_keyframe(gray_image, kf_idx, K)
            
            if result is not None:
                yaw_rel, num_inliers = result
                kf = self.keyframes[kf_idx]
                
                # Compute yaw correction
                # Current yaw should be: kf['yaw'] + yaw_rel
                # Error = current_yaw - expected_yaw
                expected_yaw = kf['yaw'] + yaw_rel
                yaw_error = yaw - expected_yaw
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # Wrap to [-π, π]
                
                # Yaw correction = -error
                yaw_correction = -yaw_error
                
                self.stats['loop_detected'] += 1
                
                print(f"[LOOP] ✓ DETECTED! frame={frame_idx} matched with kf={kf['frame_idx']}")
                print(f"       dist={dist:.1f}m, inliers={num_inliers}, "
                      f"yaw_rel={np.degrees(yaw_rel):.1f}°, correction={np.degrees(yaw_correction):.1f}°")
                
                return yaw_correction, num_inliers, kf_idx
                
        return None
    
    def apply_yaw_correction(self, kf, yaw_correction: float,
                             sigma_yaw: float = 0.1) -> bool:
        """
        Apply yaw correction to EKF state.
        
        Uses a measurement update approach:
        - Treats loop closure as a yaw measurement
        - Correction magnitude limited by Kalman gain
        
        Args:
            kf: ExtendedKalmanFilter instance
            yaw_correction: Yaw correction in radians
            sigma_yaw: Measurement noise (smaller = more trust in loop closure)
            
        Returns:
            True if correction applied successfully
        """
        # Get current yaw from state
        q_state = kf.x[6:10, 0]
        current_yaw = quaternion_to_yaw(q_state)
        
        # Compute measured yaw (= current + correction)
        measured_yaw = current_yaw + yaw_correction
        measured_yaw = np.arctan2(np.sin(measured_yaw), np.cos(measured_yaw))
        
        # Set up measurement update
        num_clones = (kf.x.shape[0] - 19) // 7  # v3.9.7: 19D nominal
        err_dim = 18 + 6 * num_clones  # v3.9.7: 18D core error
        theta_cov_idx = 8  # δθ_z in error state
        
        def h_loop_fun(x):
            H = np.zeros((1, err_dim), dtype=float)
            H[0, theta_cov_idx] = 1.0  # Yaw measurement
            return H
        
        def hx_loop_fun(x):
            q_x = x[6:10, 0]
            yaw_x = quaternion_to_yaw(q_x)
            return np.array([[yaw_x]])
        
        # Measurement noise (small = trust loop closure more)
        R_loop = np.array([[sigma_yaw**2]])
        
        # Angle residual function
        def angle_residual(a, b):
            res = a - b
            return np.arctan2(np.sin(res), np.cos(res))
        
        try:
            kf.update(
                z=np.array([[measured_yaw]]),
                HJacobian=h_loop_fun,
                Hx=hx_loop_fun,
                R=R_loop,
                residual=angle_residual,
                update_type="LOOP_CLOSURE",
                timestamp=float('nan')
            )
            
            self.stats['yaw_corrections_applied'] += 1
            self.stats['total_yaw_correction'] += abs(yaw_correction)
            
            return True
            
        except Exception as e:
            print(f"[LOOP] Failed to apply yaw correction: {e}")
            return False
    
    def print_stats(self) -> None:
        """Print loop closure statistics."""
        print(f"[LOOP] Statistics:")
        print(f"  Keyframes added: {self.stats['keyframes_added']}")
        print(f"  Loop checks: {self.stats['loop_checks']}")
        print(f"  Loop candidates found: {self.stats['loop_candidates']}")
        print(f"  Loops detected: {self.stats['loop_detected']}")
        print(f"  Rejected (few matches): {self.stats['loop_rejected_few_matches']}")
        print(f"  Rejected (geometry): {self.stats['loop_rejected_geometry']}")
        print(f"  Yaw corrections applied: {self.stats['yaw_corrections_applied']}")
        if self.stats['yaw_corrections_applied'] > 0:
            avg_corr = np.degrees(self.stats['total_yaw_correction'] / 
                                  self.stats['yaw_corrections_applied'])
            print(f"  Average yaw correction: {avg_corr:.1f}°")
    
    def reset(self) -> None:
        """Reset detector state for new run."""
        self.keyframes = []
        self.last_kf_position = None
        self.last_kf_yaw = None
        self.last_kf_frame = -1000
        for key in self.stats:
            if isinstance(self.stats[key], (int, float)):
                self.stats[key] = 0 if isinstance(self.stats[key], int) else 0.0


# ===============================
# Global Loop Closure Detector
# ===============================
_LOOP_DETECTOR: Optional[LoopClosureDetector] = None


def init_loop_closure(position_threshold: float = 50.0) -> LoopClosureDetector:
    """
    Initialize global loop closure detector.
    
    Args:
        position_threshold: Distance threshold for loop candidate detection (m)
        
    Returns:
        LoopClosureDetector instance
    """
    global _LOOP_DETECTOR
    _LOOP_DETECTOR = LoopClosureDetector(position_threshold=position_threshold)
    return _LOOP_DETECTOR


def get_loop_detector() -> Optional[LoopClosureDetector]:
    """
    Get global loop closure detector.
    
    Returns:
        LoopClosureDetector instance or None if not initialized
    """
    return _LOOP_DETECTOR


def reset_loop_detector() -> None:
    """Reset global loop closure detector."""
    global _LOOP_DETECTOR
    if _LOOP_DETECTOR is not None:
        _LOOP_DETECTOR.reset()


# =============================================================================
# Loop Closure Processing Functions
# =============================================================================

def check_loop_closure(loop_detector, img_gray: np.ndarray, t: float, kf,
                       global_config: dict, vio_fe=None) -> Optional[Tuple[float, int, int]]:
    """
    Check for loop closure and return correction if detected.
    
    Args:
        loop_detector: LoopClosureDetector instance
        img_gray: Current grayscale image
        t: Current timestamp
        kf: ExtendedKalmanFilter instance
        global_config: Global configuration dictionary
        vio_fe: VIO frontend (for frame_idx)
    
    Returns:
        Tuple of (relative_yaw, kf_idx, num_inliers) if loop detected, else None
    """
    if loop_detector is None:
        return None
    
    try:
        from .math_utils import quaternion_to_yaw
        
        # Get current state
        position = kf.x[0:3, 0].flatten()[:2]  # XY position only
        yaw = quaternion_to_yaw(kf.x[6:10, 0].flatten())
        frame_idx = vio_fe.frame_idx if vio_fe else 0
        
        # Check if we should add keyframe
        if loop_detector.should_add_keyframe(position, yaw, frame_idx):
            loop_detector.add_keyframe(frame_idx, position, yaw, img_gray)
        
        # Try to find and match loop closure
        candidates = loop_detector.find_loop_candidates(position, frame_idx)
        
        if len(candidates) > 0:
            # Get camera intrinsics for matching
            kb_params = global_config.get('KB_PARAMS', {})
            K = np.array([
                [kb_params.get('mu', 600), 0, img_gray.shape[1] / 2],
                [0, kb_params.get('mv', 600), img_gray.shape[0] / 2],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Try to match with each candidate
            for kf_idx in candidates:
                result = loop_detector.match_keyframe(img_gray, kf_idx, K)
                
                if result is not None:
                    relative_yaw, num_inliers = result
                    return (relative_yaw, kf_idx, num_inliers)
        
        return None
        
    except Exception as e:
        print(f"[LOOP] Error in loop closure check: {e}")
        return None


def apply_loop_closure_correction(kf, relative_yaw: float, kf_idx: int,
                                   num_inliers: int, t: float,
                                   cam_states: list, loop_detector) -> bool:
    """
    Apply yaw correction from loop closure detection.
    
    Args:
        kf: ExtendedKalmanFilter instance
        relative_yaw: Measured yaw difference from loop closure
        kf_idx: Index of matched keyframe
        num_inliers: Number of feature match inliers
        t: Current timestamp
        cam_states: List of camera clone states
        loop_detector: LoopClosureDetector instance
    
    Returns:
        True if correction applied, False otherwise
    """
    if loop_detector is None:
        return False
    
    try:
        from .math_utils import quaternion_to_yaw
        
        # Get expected yaw difference from stored keyframe
        stored_kf = loop_detector.keyframes[kf_idx]
        current_yaw = quaternion_to_yaw(kf.x[6:10, 0].flatten())
        expected_yaw_diff = current_yaw - stored_kf['yaw']
        
        # Yaw correction (innovation)
        yaw_error = relative_yaw - expected_yaw_diff
        
        # Wrap to [-π, π]
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        
        # Only apply if correction is significant but not too large
        if np.abs(yaw_error) < np.radians(2.0):  # < 2° skip
            return False
        if np.abs(yaw_error) > np.radians(30.0):  # > 30° suspicious
            print(f"[LOOP] REJECT: yaw_error={np.degrees(yaw_error):.1f}° too large")
            return False
        
        # Build EKF update for yaw
        num_clones = len(cam_states)
        err_dim = 18 + 6 * num_clones  # v3.9.7: 18D core error
        
        H_loop = np.zeros((1, err_dim), dtype=float)
        H_loop[0, 8] = 1.0  # Yaw error index
        
        z_loop = np.array([[yaw_error]])
        
        # Measurement noise (inversely proportional to inliers)
        base_sigma = np.radians(5.0)  # 5° base uncertainty
        sigma_loop = base_sigma / np.sqrt(max(num_inliers, 1) / 15.0)
        R_loop = np.array([[sigma_loop**2]])
        
        # Apply EKF update
        def h_loop_jacobian(x, h=H_loop):
            return h
        
        def hx_loop_fun(x, h=H_loop):
            return np.zeros((1, 1))  # Zero residual (error is already computed)
        
        kf.update(
            z=z_loop,
            HJacobian=h_loop_jacobian,
            Hx=hx_loop_fun,
            R=R_loop,
            update_type="LOOP_CLOSURE",
            timestamp=t
        )
        
        print(f"[LOOP] CORRECTION at t={t:.2f}s: Δyaw={np.degrees(yaw_error):.2f}° "
              f"(kf={kf_idx}, inliers={num_inliers})")
        
        return True
        
    except Exception as e:
        print(f"[LOOP] Failed to apply correction: {e}")
        return False
