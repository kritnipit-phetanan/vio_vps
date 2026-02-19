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
from typing import Optional, Tuple, List, Dict, Any

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
                 min_inliers: int = 15,               # REDUCED: accept partial matches
                 quality_gate: Optional[Dict[str, Any]] = None,
                 fail_soft: Optional[Dict[str, Any]] = None):
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
        self.quality_gate = dict(quality_gate or {})
        self.fail_soft = dict(fail_soft or {})
        
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
            'loop_quality_rejected': 0,
            'loop_pending_confirm': 0,
            'loop_borderline': 0,
            'loop_cooldown_skipped': 0,
            'loop_rejected_few_matches': 0,
            'loop_rejected_geometry': 0,
            'loop_speed_skipped': 0,
            'yaw_corrections_applied': 0,
            'total_yaw_correction': 0.0,
        }
        
        # Last keyframe info (to avoid adding too many)
        self.last_kf_position: Optional[np.ndarray] = None
        self.last_kf_yaw: Optional[float] = None
        self.last_kf_frame: int = -1000
        self._pending_match: Optional[Dict[str, Any]] = None
        self._last_apply_t: float = -1e9
        
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
    
    def _quality_gate_value(self, key: str, default: float) -> float:
        try:
            return float(self.quality_gate.get(key, default))
        except Exception:
            return float(default)

    def _phase_gate_mult(self, phase: int) -> float:
        # Takeoff/landing phases are more dynamic: require stronger loop quality.
        if int(phase) <= 1:
            return max(1.0, self._quality_gate_value("phase_dynamic_inlier_mult", 1.15))
        return 1.0

    def _health_gate_mult(self, health_state: str) -> float:
        health = str(health_state).upper()
        if health == "WARNING":
            return max(1.0, self._quality_gate_value("warning_inlier_mult", 1.10))
        if health == "DEGRADED":
            return max(1.0, self._quality_gate_value("degraded_inlier_mult", 1.20))
        return 1.0

    def _spatial_spread_ratio(self, pts: np.ndarray, image_shape: Tuple[int, int]) -> float:
        if pts is None or len(pts) < 4:
            return 0.0
        try:
            min_xy = np.min(pts, axis=0)
            max_xy = np.max(pts, axis=0)
            spread = np.linalg.norm(max_xy - min_xy)
            h, w = image_shape[:2]
            diag = float(np.hypot(w, h))
            return float(spread / max(1e-6, diag))
        except Exception:
            return 0.0

    def _compute_epipolar_reproj_px(self, E: np.ndarray, K: np.ndarray,
                                    pts1: np.ndarray, pts2: np.ndarray) -> float:
        """Return P95 Sampson error (px) as loop quality proxy."""
        try:
            K_inv = np.linalg.inv(K)
            F = K_inv.T @ E @ K_inv
            F = F / max(1e-9, np.linalg.norm(F))
            errs = []
            for p1, p2 in zip(pts1, pts2):
                x1 = np.array([p1[0], p1[1], 1.0], dtype=float)
                x2 = np.array([p2[0], p2[1], 1.0], dtype=float)
                err2 = cv2.sampsonDistance(x1.reshape(3, 1), x2.reshape(3, 1), F)
                if np.isfinite(err2):
                    errs.append(float(np.sqrt(max(0.0, err2))))
            if len(errs) == 0:
                return float("inf")
            return float(np.percentile(np.asarray(errs, dtype=float), 95.0))
        except Exception:
            return float("inf")

    def match_keyframe(self, gray_image: np.ndarray, kf_idx: int,
                       K: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Match current image with keyframe and compute relative yaw.
        
        Args:
            gray_image: Current grayscale image
            kf_idx: Index of keyframe to match against
            K: Camera intrinsic matrix (3x3)
        
        Returns:
            Dict with match metrics (yaw_rel, inliers, spread, reproj, ...)
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

        inlier_mask = mask_pose.reshape(-1).astype(bool) if mask_pose is not None else mask.reshape(-1).astype(bool)
        pts_kf_in = pts_kf[inlier_mask] if np.any(inlier_mask) else pts_kf
        pts_curr_in = pts_curr[inlier_mask] if np.any(inlier_mask) else pts_curr
        spread_ratio = self._spatial_spread_ratio(pts_curr_in, gray_image.shape)
        reproj_p95_px = self._compute_epipolar_reproj_px(E, K, pts_kf_in, pts_curr_in)

        return {
            "yaw_rel": float(yaw_rel),
            "num_inliers": int(num_inliers),
            "inlier_ratio": float(inlier_ratio),
            "spread_ratio": float(spread_ratio),
            "reproj_p95_px": float(reproj_p95_px),
            "num_good_matches": int(len(good_matches)),
        }

    def check_loop_closure(self, frame_idx: int, position: np.ndarray, yaw: float,
                           gray_image: np.ndarray, K: np.ndarray,
                           timestamp: float,
                           phase: int = 2,
                           health_state: str = "HEALTHY",
                           global_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Check for loop closure and return yaw correction if found.
        
        Args:
            frame_idx: Current frame index
            position: Current estimated position [x, y, z]
            yaw: Current estimated yaw [rad]
            gray_image: Current grayscale image
            K: Camera intrinsic matrix
            
        Returns:
            Loop match dict ready for correction, or None.
        """
        self.stats['loop_checks'] += 1

        cooldown_sec = self._quality_gate_value("cooldown_sec", 2.0)
        if float(timestamp) - float(self._last_apply_t) < cooldown_sec:
            self.stats['loop_cooldown_skipped'] += 1
            return None
        
        # Find candidate keyframes
        candidates = self.find_loop_candidates(position, frame_idx)
        
        if len(candidates) == 0:
            return None
            
        self.stats['loop_candidates'] += len(candidates)
        
        # Try matching with each candidate (closest first)
        candidates_with_dist = [(i, np.linalg.norm(position[:2] - self.keyframes[i]['position'][:2])) 
                                for i in candidates]
        candidates_with_dist.sort(key=lambda x: x[1])
        
        phase_mult = self._phase_gate_mult(phase)
        health_mult = self._health_gate_mult(health_state)
        gate_inliers_hard = int(round(self._quality_gate_value("min_inliers_hard", self.min_inliers) * phase_mult * health_mult))
        gate_inliers_soft = int(round(self._quality_gate_value("min_inliers_failsoft", max(6, self.min_inliers // 2)) * phase_mult))
        gate_spread = self._quality_gate_value("min_spatial_spread", 0.18)
        gate_reproj = self._quality_gate_value("max_reproj_px", 2.5)
        gate_yaw = self._quality_gate_value("yaw_residual_bound_deg", 25.0)
        double_confirm = bool(self.quality_gate.get("double_confirm_enable", True))
        confirm_window_sec = self._quality_gate_value("double_confirm_window_sec", 2.0)
        confirm_yaw_deg = self._quality_gate_value("double_confirm_yaw_deg", 8.0)
        use_fail_soft = bool(self.fail_soft.get("enable", True))

        for kf_idx, dist in candidates_with_dist[:3]:  # Try top 3 closest
            result = self.match_keyframe(gray_image, kf_idx, K)
            
            if result is not None:
                yaw_rel = float(result["yaw_rel"])
                num_inliers = int(result["num_inliers"])
                kf = self.keyframes[kf_idx]
                
                # Compute yaw correction
                # Current yaw should be: kf['yaw'] + yaw_rel
                # Error = current_yaw - expected_yaw
                expected_yaw = kf['yaw'] + yaw_rel
                yaw_error = yaw - expected_yaw
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # Wrap to [-π, π]
                
                # Yaw correction = -error
                yaw_correction = float(-yaw_error)
                yaw_abs_deg = abs(np.degrees(yaw_correction))
                spread_ratio = float(result.get("spread_ratio", 0.0))
                reproj_p95_px = float(result.get("reproj_p95_px", np.inf))
                inlier_ratio = float(result.get("inlier_ratio", 0.0))

                hard_ok = (
                    num_inliers >= gate_inliers_hard
                    and spread_ratio >= gate_spread
                    and reproj_p95_px <= gate_reproj
                    and yaw_abs_deg <= gate_yaw
                )
                soft_ok = (
                    use_fail_soft
                    and num_inliers >= gate_inliers_soft
                    and spread_ratio >= (0.7 * gate_spread)
                    and reproj_p95_px <= (1.6 * gate_reproj)
                    and yaw_abs_deg <= (1.35 * gate_yaw)
                )

                if not (hard_ok or soft_ok):
                    self.stats['loop_quality_rejected'] += 1
                    continue

                if not hard_ok and soft_ok:
                    self.stats['loop_borderline'] += 1

                loop_info = {
                    "yaw_correction": yaw_correction,
                    "num_inliers": num_inliers,
                    "kf_idx": kf_idx,
                    "dist_m": float(dist),
                    "yaw_rel": yaw_rel,
                    "spread_ratio": spread_ratio,
                    "reproj_p95_px": reproj_p95_px,
                    "inlier_ratio": inlier_ratio,
                    "fail_soft": bool(not hard_ok and soft_ok),
                    "phase": int(phase),
                    "health_state": str(health_state).upper(),
                    "timestamp": float(timestamp),
                }

                if double_confirm:
                    pending = self._pending_match
                    if pending is None:
                        self._pending_match = loop_info
                        self.stats['loop_pending_confirm'] += 1
                        return None
                    same_kf = int(pending.get("kf_idx", -1)) == int(kf_idx)
                    dt_ok = (float(timestamp) - float(pending.get("timestamp", -1e9))) <= confirm_window_sec
                    yaw_ok = abs(np.degrees(loop_info["yaw_correction"] - float(pending.get("yaw_correction", 0.0)))) <= confirm_yaw_deg
                    if same_kf and dt_ok and yaw_ok:
                        self._pending_match = None
                        self.stats['loop_detected'] += 1
                        print(f"[LOOP] ✓ DETECTED (double-confirmed) frame={frame_idx}, kf={kf['frame_idx']}, "
                              f"inliers={num_inliers}, spread={spread_ratio:.3f}, reproj95={reproj_p95_px:.2f}px")
                        return loop_info
                    self._pending_match = loop_info
                    self.stats['loop_pending_confirm'] += 1
                    return None

                self.stats['loop_detected'] += 1
                print(f"[LOOP] ✓ DETECTED frame={frame_idx}, kf={kf['frame_idx']}, "
                      f"inliers={num_inliers}, spread={spread_ratio:.3f}, reproj95={reproj_p95_px:.2f}px")
                return loop_info
                
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


def init_loop_closure(position_threshold: float = 50.0,
                      min_keyframe_dist: float = 15.0,
                      min_keyframe_yaw: float = 20.0,
                      min_frame_gap: int = 50,
                      min_match_ratio: float = 0.12,
                      min_inliers: int = 15,
                      quality_gate: Optional[Dict[str, Any]] = None,
                      fail_soft: Optional[Dict[str, Any]] = None) -> LoopClosureDetector:
    """
    Initialize global loop closure detector.
    
    Args:
        position_threshold: Distance threshold for loop candidate detection (m)
        
    Returns:
        LoopClosureDetector instance
    """
    global _LOOP_DETECTOR
    _LOOP_DETECTOR = LoopClosureDetector(
        position_threshold=position_threshold,
        min_keyframe_dist=min_keyframe_dist,
        min_keyframe_yaw=min_keyframe_yaw,
        min_frame_gap=min_frame_gap,
        min_match_ratio=min_match_ratio,
        min_inliers=min_inliers,
        quality_gate=quality_gate,
        fail_soft=fail_soft,
    )
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
                       global_config: dict, vio_fe=None,
                       phase: int = 2,
                       health_state: str = "HEALTHY") -> Optional[Dict[str, Any]]:
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
        Loop info dictionary if a quality-gated loop is detected, else None.
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
        
        # Get camera intrinsics for matching
        kb_params = global_config.get('KB_PARAMS', {})
        K = np.array([
            [kb_params.get('mu', 600), 0, img_gray.shape[1] / 2],
            [0, kb_params.get('mv', 600), img_gray.shape[0] / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        # Run detector quality-gated check
        result = loop_detector.check_loop_closure(
            frame_idx=frame_idx,
            position=position,
            yaw=yaw,
            gray_image=img_gray,
            K=K,
            timestamp=float(t),
            phase=int(phase),
            health_state=str(health_state),
            global_config=global_config,
        )
        if result is not None:
            return result
        
        return None
        
    except Exception as e:
        print(f"[LOOP] Error in loop closure check: {e}")
        return None


def apply_loop_closure_correction(kf, loop_info: Dict[str, Any], t: float,
                                   cam_states: list, loop_detector,
                                   global_config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Apply yaw correction from loop closure detection.
    
    Args:
        kf: ExtendedKalmanFilter instance
        loop_info: Loop match dictionary from check_loop_closure
        t: Current timestamp
        cam_states: List of camera clone states
        loop_detector: LoopClosureDetector instance
    
    Returns:
        True if correction applied, False otherwise
    """
    if loop_detector is None:
        return False
    
    try:
        global_config = global_config or {}
        yaw_error = float(loop_info.get("yaw_correction", 0.0))
        num_inliers = int(loop_info.get("num_inliers", 0))
        kf_idx = int(loop_info.get("kf_idx", -1))
        phase = int(loop_info.get("phase", 2))
        health = str(loop_info.get("health_state", "HEALTHY")).upper()
        fail_soft = bool(loop_info.get("fail_soft", False))

        # Rejection/clamp bounds
        min_abs_deg = float(global_config.get("LOOP_MIN_ABS_YAW_CORR_DEG", 1.5))
        max_abs_deg = float(global_config.get("LOOP_MAX_ABS_YAW_CORR_DEG", 4.0))
        reject_abs_deg = float(global_config.get("LOOP_YAW_RESIDUAL_BOUND_DEG", 25.0))
        speed_skip_m_s = float(global_config.get("LOOP_SPEED_SKIP_M_S", 35.0))
        speed_sigma_inflate_m_s = float(global_config.get("LOOP_SPEED_SIGMA_INFLATE_M_S", 25.0))
        speed_sigma_mult = float(global_config.get("LOOP_SPEED_SIGMA_MULT", 1.5))
        speed_now = float(np.linalg.norm(np.asarray(kf.x[3:6, 0], dtype=float)))

        if abs(np.degrees(yaw_error)) < min_abs_deg:
            return False
        if abs(np.degrees(yaw_error)) > reject_abs_deg:
            print(f"[LOOP] REJECT: yaw_error={np.degrees(yaw_error):.1f}° > {reject_abs_deg:.1f}°")
            return False
        if np.isfinite(speed_now) and speed_now > speed_skip_m_s:
            if loop_detector is not None and hasattr(loop_detector, "stats"):
                loop_detector.stats["loop_speed_skipped"] = int(loop_detector.stats.get("loop_speed_skipped", 0)) + 1
            print(
                f"[LOOP] SKIP apply at t={t:.2f}s: speed={speed_now:.2f}m/s > {speed_skip_m_s:.2f}m/s"
            )
            return False

        # Clamp correction magnitude (fail-soft by design)
        yaw_abs = abs(yaw_error)
        yaw_sign = 1.0 if yaw_error >= 0.0 else -1.0
        yaw_error = yaw_sign * min(yaw_abs, np.deg2rad(max_abs_deg))

        # Build EKF update for yaw
        num_clones = len(cam_states)
        err_dim = 18 + 6 * num_clones  # v3.9.7: 18D core error
        
        H_loop = np.zeros((1, err_dim), dtype=float)
        H_loop[0, 8] = 1.0  # Yaw error index
        
        z_loop = np.array([[yaw_error]])
        
        # Measurement noise (inversely proportional to inliers), with fail-soft profile.
        base_sigma_deg = float(global_config.get("LOOP_BASE_SIGMA_YAW_DEG", 5.0))
        fail_soft_sigma_deg = float(global_config.get("LOOP_FAIL_SOFT_SIGMA_YAW_DEG", 18.0))
        sigma_loop = np.deg2rad(base_sigma_deg) / np.sqrt(max(num_inliers, 1) / 15.0)
        if fail_soft:
            sigma_loop = max(sigma_loop, np.deg2rad(fail_soft_sigma_deg))
        if (
            np.isfinite(speed_now)
            and speed_now > speed_sigma_inflate_m_s
            and speed_sigma_mult > 1.0
        ):
            sigma_loop *= float(speed_sigma_mult)

        # Dynamic phase/health inflation (conservative during dynamic phases)
        if phase <= 1:
            sigma_loop *= float(global_config.get("LOOP_DYNAMIC_PHASE_SIGMA_MULT", 1.15))
        if health == "WARNING":
            sigma_loop *= float(global_config.get("LOOP_WARNING_SIGMA_MULT", 1.20))
        elif health == "DEGRADED":
            sigma_loop *= float(global_config.get("LOOP_DEGRADED_SIGMA_MULT", 1.40))
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

        # Keep detector statistics consistent with actually-applied EKF corrections.
        if hasattr(loop_detector, "stats") and isinstance(loop_detector.stats, dict):
            loop_detector.stats["yaw_corrections_applied"] = int(
                loop_detector.stats.get("yaw_corrections_applied", 0)
            ) + 1
            loop_detector.stats["total_yaw_correction"] = float(
                loop_detector.stats.get("total_yaw_correction", 0.0)
            ) + abs(float(yaw_error))
        
        loop_detector._last_apply_t = float(t)
        tag = "FAILSOFT" if fail_soft else "NORMAL"
        print(f"[LOOP] {tag} correction at t={t:.2f}s: Δyaw={np.degrees(yaw_error):.2f}° "
              f"(kf={kf_idx}, inliers={num_inliers}, sigma={np.degrees(sigma_loop):.2f}°)")
        
        return True
        
    except Exception as e:
        print(f"[LOOP] Failed to apply correction: {e}")
        return False
