#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIO Frontend Module

OpenVINS-style visual front-end with:
1. Grid-based feature distribution
2. Shi-Tomasi corner detector
3. Multi-stage KLT tracking
4. Track quality scoring
5. Aggressive outlier rejection

Reference: OpenVINS (https://github.com/rpng/open_vins)

Author: VIO project
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict

from .camera import kannala_brandt_unproject


# Default constants
VO_MIN_INLIERS = 15
VO_NADIR_ALIGN_DEG = 30.0


class VIOFrontEnd:
    """
    OpenVINS-style visual front-end:
    1. Grid-based feature distribution: Divide image into grid, select best features per cell
    2. Shi-Tomasi corner detector: Better tracking quality than ORB
    3. Multi-stage KLT tracking: Coarse-to-fine optical flow
    4. Track quality scoring: Distance from epipolar line, temporal consistency
    5. Aggressive outlier rejection: RANSAC + chi-squared gating
    
    Reference: OpenVINS (https://github.com/rpng/open_vins)
    """
    
    def __init__(self, img_w: int, img_h: int, K: np.ndarray, D: np.ndarray, 
                 use_fisheye: bool = True):
        self.K = K
        self.D = D
        self.use_fisheye = use_fisheye
        self.img_w = img_w
        self.img_h = img_h
        
        # Grid-based feature extraction (OpenVINS-style)
        self.grid_x = 8  # Horizontal grid cells
        self.grid_y = 8  # Vertical grid cells
        self.max_features_per_grid = 10  # Max features per cell
        self.max_total_features = self.grid_x * self.grid_y * self.max_features_per_grid
        
        # Shi-Tomasi corner detection
        self.feature_params = dict(
            maxCorners=self.max_total_features,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=5
        )
        
        # Multi-stage KLT parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            minEigThreshold=0.001
        )
        
        # Track database: fid -> list of observations
        self.tracks: Dict[int, List[dict]] = {}
        self.next_fid = 0
        self.last_gray_for_klt = None
        self.last_pts_for_klt = None
        self.last_fids_for_klt = None
        
        # Keyframe state
        self.keyframe_gray = None
        self.keyframe_frame_idx = -1
        self.keyframe_tracked_ratio = 1.0
        self.frame_idx = -1
        self.last_matches = None
        self.last_frame_time = None
        
        # OpenVINS-style keyframe management
        self.min_tracked_ratio = 0.6
        self.min_parallax_threshold = 30.0
        self.max_track_length = 100
        self.min_track_length = 4
        
        # Track quality scoring
        self.quality_threshold = 0.001
        self.epipolar_threshold = 1.0
        self.temporal_threshold = 50.0
        
        # Debug counter
        self._undist_dbg_count = 0

    def _extract_grid_features(self, img_gray: np.ndarray) -> np.ndarray:
        """
        OpenVINS-style grid-based feature extraction.
        """
        cell_w = self.img_w // self.grid_x
        cell_h = self.img_h // self.grid_y
        
        all_features = []
        
        for gx in range(self.grid_x):
            for gy in range(self.grid_y):
                x1 = gx * cell_w
                y1 = gy * cell_h
                x2 = min((gx + 1) * cell_w, self.img_w)
                y2 = min((gy + 1) * cell_h, self.img_h)
                
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                
                cell_roi = img_gray[y1:y2, x1:x2]
                
                corners = cv2.goodFeaturesToTrack(
                    cell_roi,
                    maxCorners=self.max_features_per_grid,
                    qualityLevel=self.feature_params['qualityLevel'],
                    minDistance=self.feature_params['minDistance'],
                    blockSize=self.feature_params['blockSize']
                )
                
                if corners is not None and len(corners) > 0:
                    corners_global = corners.reshape(-1, 2) + np.array([x1, y1], dtype=np.float32)
                    for pt in corners_global:
                        all_features.append(pt)
        
        if len(all_features) == 0:
            return np.empty((0, 2), dtype=np.float32)
        
        return np.array(all_features, dtype=np.float32)
    
    def _compute_epipolar_error(self, pts_prev: np.ndarray, pts_curr: np.ndarray, 
                                 E: np.ndarray) -> np.ndarray:
        """Compute distance from points to epipolar lines."""
        if E is None or pts_prev.shape[0] == 0:
            return np.array([])
        
        pts_prev_h = np.hstack([pts_prev, np.ones((len(pts_prev), 1))])
        pts_curr_h = np.hstack([pts_curr, np.ones((len(pts_curr), 1))])
        
        lines = (E @ pts_prev_h.T).T
        numerator = np.abs(np.sum(pts_curr_h * lines, axis=1))
        denominator = np.linalg.norm(lines[:, :2], axis=1)
        
        errors = numerator / (denominator + 1e-8)
        return errors

    def _undistort_pts(self, pts: np.ndarray) -> np.ndarray:
        """Undistort pixel coordinates to normalized camera coordinates."""
        if self.use_fisheye:
            undist = kannala_brandt_unproject(pts.reshape(-1, 2), self.K, self.D)
            
            if self._undist_dbg_count < 5:
                for i in range(min(3, len(pts))):
                    px = pts.reshape(-1, 2)[i]
                    un = undist[i]
                    print(f"[KB-UNDIST] pixel=({px[0]:.1f},{px[1]:.1f}) → norm=({un[0]:.4f},{un[1]:.4f})")
                self._undist_dbg_count += 1
            
            return undist.reshape(-1, 1, 2)
        else:
            undist = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.D)
            return undist

    def _should_create_keyframe(self, current_gray: np.ndarray) -> Tuple[bool, str]:
        """Decide if we should create a new keyframe."""
        if self.keyframe_frame_idx < 0:
            return True, "first_keyframe"
        
        if self.keyframe_tracked_ratio < self.min_tracked_ratio:
            return True, f"low_tracking_ratio_{self.keyframe_tracked_ratio:.2f}"
        
        if self.last_gray_for_klt is not None and self.last_pts_for_klt is not None and len(self.last_pts_for_klt) > 10:
            try:
                p0 = self.last_pts_for_klt.reshape(-1, 1, 2)
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.keyframe_gray, current_gray, p0, None, **self.lk_params)
                if p1 is not None and st is not None:
                    good = st.reshape(-1) == 1
                    if np.sum(good) > 10:
                        parallax = np.linalg.norm(p0[good] - p1[good], axis=2).reshape(-1)
                        median_parallax = np.median(parallax)
                        
                        if median_parallax > self.min_parallax_threshold:
                            return True, f"high_parallax_{median_parallax:.1f}px"
            except Exception:
                pass
        
        frames_since_keyframe = self.frame_idx - self.keyframe_frame_idx
        if frames_since_keyframe > 20:
            return True, f"frame_count_{frames_since_keyframe}"
        
        return False, "keep_current"
    
    def compute_track_parallax_stats(self) -> dict:
        """Compute parallax statistics for currently tracked features."""
        parallax_values = []
        
        for fid, hist in self.tracks.items():
            if len(hist) < 2:
                continue
            
            pts = np.array([obs['pt'] for obs in hist])
            if len(pts) < 2:
                continue
            
            dists = []
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    dists.append(np.linalg.norm(pts[i] - pts[j]))
            
            if dists:
                parallax_values.append(max(dists))
        
        if not parallax_values:
            return {'median_px': 0.0, 'mean_px': 0.0, 'max_px': 0.0, 'count': 0}
        
        return {
            'median_px': np.median(parallax_values),
            'mean_px': np.mean(parallax_values),
            'max_px': np.max(parallax_values),
            'count': len(parallax_values)
        }

    def _prune_old_tracks(self):
        """Remove tracks that are too old or have poor quality."""
        to_remove = []
        current_frame = self.frame_idx
        
        for fid, hist in self.tracks.items():
            if not hist:
                to_remove.append(fid)
                continue
            
            if len(hist) > self.max_track_length:
                to_remove.append(fid)
                continue
            
            last_frame = hist[-1]['frame']
            if current_frame - last_frame > 3:
                to_remove.append(fid)
                continue
            
            if len(hist) > 0:
                avg_quality = np.mean([obs.get('quality', 1.0) for obs in hist])
                if avg_quality < 0.3:
                    to_remove.append(fid)
        
        for fid in to_remove:
            del self.tracks[fid]

    def bootstrap(self, img_gray: np.ndarray, t: float):
        """Initialize front-end with first frame."""
        self.keyframe_gray = img_gray.copy()
        self.last_frame_time = t
        self.frame_idx = 0
        self.keyframe_frame_idx = 0
        
        try:
            features = self._extract_grid_features(img_gray)
            
            if len(features) > 0:
                self.last_pts_for_klt = features.astype(np.float32)
                self.last_gray_for_klt = img_gray.copy()
                
                fids = []
                for p in features:
                    fid = self.next_fid
                    self.next_fid += 1
                    self.tracks[fid] = [{'frame': self.frame_idx, 'pt': (float(p[0]), float(p[1])), 'quality': 1.0}]
                    fids.append(fid)
                
                self.last_fids_for_klt = np.array(fids, dtype=np.int64)
                print(f"[VIO][BOOTSTRAP] Initialized {len(features)} grid-based features")
            else:
                self.last_pts_for_klt = None
                self.last_gray_for_klt = None
                self.last_fids_for_klt = None
                print("[VIO][BOOTSTRAP] WARNING: No features detected!")
        except Exception as e:
            print(f"[VIO][BOOTSTRAP] Exception: {e}")
            self.last_pts_for_klt = None
            self.last_gray_for_klt = None
            self.last_fids_for_klt = None

    def step(self, img_gray: np.ndarray, t: float):
        """Process frame and return pose estimate."""
        if self.keyframe_gray is None:
            self.bootstrap(img_gray, t)
            return False, 0, None, None, 0.0
        
        self.frame_idx += 1
        num_tracked_successfully = 0
        
        # KLT tracking
        try:
            if self.last_gray_for_klt is not None and self.last_pts_for_klt is not None and len(self.last_pts_for_klt) > 0:
                p0 = self.last_pts_for_klt.reshape(-1, 1, 2)
                
                # Forward tracking
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.last_gray_for_klt, img_gray, p0, None, **self.lk_params)
                
                if p1 is not None:
                    # Backward tracking for consistency
                    p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(img_gray, self.last_gray_for_klt, p1, None, **self.lk_params)
                    
                    fb_err = np.linalg.norm(p0 - p0_back.reshape(-1, 1, 2), axis=2).reshape(-1)
                    flow_mag = np.linalg.norm(p1.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
                    
                    # Quality masks
                    good_mask = (st.reshape(-1) == 1) & (st_back.reshape(-1) == 1)
                    good_mask = good_mask & (fb_err < 2.0)
                    good_mask = good_mask & (err.reshape(-1) < 8.0)
                    good_mask = good_mask & (flow_mag < 200.0)
                    
                    p1_reshaped = p1.reshape(-1, 2)
                    margin = 10
                    good_mask = good_mask & (p1_reshaped[:, 0] >= margin) & (p1_reshaped[:, 0] < self.img_w - margin)
                    good_mask = good_mask & (p1_reshaped[:, 1] >= margin) & (p1_reshaped[:, 1] < self.img_h - margin)
                    
                    st = st.reshape(-1) * good_mask
                    p1 = p1.reshape(-1, 2)
                    
                    if self.last_fids_for_klt is None or len(self.last_fids_for_klt) != len(p0):
                        self.last_fids_for_klt = np.array(list(self.tracks.keys())[:len(p0)], dtype=np.int64)
                    
                    tracked_fids = []
                    new_pts = []
                    new_fids = []
                    
                    for idx in range(len(p1)):
                        if idx >= len(self.last_fids_for_klt):
                            break
                        
                        fid = self.last_fids_for_klt[idx]
                        
                        if st[idx] and fid in self.tracks:
                            pt = (float(p1[idx, 0]), float(p1[idx, 1]))
                            quality = 1.0 / (1.0 + fb_err[idx] + err.reshape(-1)[idx] * 0.1)
                            self.tracks[fid].append({'frame': self.frame_idx, 'pt': pt, 'quality': quality})
                            tracked_fids.append(fid)
                            new_pts.append(pt)
                            new_fids.append(fid)
                            num_tracked_successfully += 1
                    
                    total_tracks_before = len(self.last_pts_for_klt)
                    self.keyframe_tracked_ratio = num_tracked_successfully / max(1, total_tracks_before)
                    
                    if len(new_pts) > 0:
                        self.last_pts_for_klt = np.array(new_pts, dtype=np.float32)
                        self.last_fids_for_klt = np.array(new_fids, dtype=np.int64)
                        self.last_gray_for_klt = img_gray.copy()
                    else:
                        # Emergency replenish
                        self._emergency_replenish(img_gray)
        except Exception as e:
            print(f"[VIO] KLT tracking exception: {e}")
        
        # Prune old tracks
        if self.frame_idx % 5 == 0:
            self._prune_old_tracks()
        
        # Pose estimation
        dt_img = max(1e-3, t - (self.last_frame_time or t))
        
        prev_pts = []
        curr_pts = []
        prev_frame_idx = self.frame_idx - 1
        
        for fid, hist in self.tracks.items():
            if len(hist) < 2:
                continue
            
            prev_obs = None
            curr_obs = None
            
            for obs in hist:
                if obs['frame'] == prev_frame_idx:
                    prev_obs = obs
                if obs['frame'] == self.frame_idx:
                    curr_obs = obs
            
            if prev_obs is not None and curr_obs is not None:
                prev_pts.append(prev_obs['pt'])
                curr_pts.append(curr_obs['pt'])
        
        if len(prev_pts) < VO_MIN_INLIERS:
            return False, len(prev_pts), None, None, dt_img
        
        q1 = np.array(prev_pts, dtype=np.float32)
        q2 = np.array(curr_pts, dtype=np.float32)
        
        # Flow filtering
        flow_vectors = q2 - q1
        flow_mag = np.linalg.norm(flow_vectors, axis=1)
        median_flow = np.median(flow_vectors, axis=0)
        flow_deviations = flow_vectors - median_flow
        mad = np.median(np.linalg.norm(flow_deviations, axis=1))
        
        flow_threshold = min(300.0, max(50.0, np.median(flow_mag) + 5.0 * mad))
        flow_valid_mask = flow_mag < flow_threshold
        
        if np.sum(flow_valid_mask) < VO_MIN_INLIERS:
            return False, len(q1), None, None, dt_img
        
        q1 = q1[flow_valid_mask]
        q2 = q2[flow_valid_mask]
        
        # Essential matrix estimation
        q1n = self._undistort_pts(q1)
        q2n = self._undistort_pts(q2)
        E, mask = cv2.findEssentialMat(q2n, q1n, method=cv2.RANSAC, prob=0.999, threshold=5e-3)
        
        if E is None or mask is None:
            return False, len(q1), None, None, dt_img
        
        num_inl, R_vo, t_unit, pose_mask = cv2.recoverPose(E, q2n, q1n, mask=mask)
        
        if num_inl < VO_MIN_INLIERS:
            return False, num_inl, None, None, dt_img
        
        # Epipolar error check
        inlier_idx = pose_mask.ravel().astype(bool)
        q1n_inliers = q1n.reshape(-1, 2)[inlier_idx]
        q2n_inliers = q2n.reshape(-1, 2)[inlier_idx]
        
        epipolar_errors = self._compute_epipolar_error(q1n_inliers, q2n_inliers, E)
        final_inliers = epipolar_errors < self.epipolar_threshold
        num_final = np.sum(final_inliers)
        
        if num_final < VO_MIN_INLIERS:
            return False, num_final, None, None, dt_img
        
        # Save inlier matches
        try:
            q1n_arr = self._undistort_pts(q1).reshape(-1, 2)
            q2n_arr = self._undistort_pts(q2).reshape(-1, 2)
            q1n_final = q1n_arr[inlier_idx][final_inliers]
            q2n_final = q2n_arr[inlier_idx][final_inliers]
            self.last_matches = (q1n_final.copy(), q2n_final.copy())
        except Exception:
            self.last_matches = None
        
        # Keyframe management
        should_keyframe, kf_reason = self._should_create_keyframe(img_gray)
        if should_keyframe:
            print(f"[VIO] Creating new keyframe at frame {self.frame_idx}: {kf_reason}")
            self.keyframe_gray = img_gray.copy()
            self.keyframe_frame_idx = self.frame_idx
            self.keyframe_tracked_ratio = 1.0
        
        self.last_frame_time = t
        
        # Replenish features
        self._replenish_features(img_gray)
        
        return True, int(num_inl), R_vo, t_unit.reshape(-1), dt_img

    def _emergency_replenish(self, img_gray: np.ndarray):
        """Emergency feature replenishment when all tracking fails."""
        print(f"[VIO][TRACK] WARNING: Frame {self.frame_idx} - All tracking failed! Replenishing...")
        
        try:
            new_features = self._extract_grid_features(img_gray)
            
            if len(new_features) > 0:
                new_fids = []
                for p in new_features:
                    fid = self.next_fid
                    self.next_fid += 1
                    self.tracks[fid] = [{'frame': self.frame_idx, 'pt': (float(p[0]), float(p[1])), 'quality': 1.0}]
                    new_fids.append(fid)
                
                self.last_pts_for_klt = new_features.astype(np.float32)
                self.last_fids_for_klt = np.array(new_fids, dtype=np.int64)
                self.last_gray_for_klt = img_gray.copy()
                print(f"[VIO][EMERGENCY_REPLENISH] Added {len(new_features)} new features")
            else:
                self.last_pts_for_klt = np.empty((0, 2), dtype=np.float32)
                self.last_fids_for_klt = np.empty(0, dtype=np.int64)
                self.last_gray_for_klt = None
        except Exception as e:
            print(f"[VIO][EMERGENCY_REPLENISH] Exception: {e}")

    def _replenish_features(self, img_gray: np.ndarray):
        """Replenish features if count is low."""
        try:
            num_active_tracks = sum(1 for v in self.tracks.values() if len(v) > 0 and v[-1]['frame'] == self.frame_idx)
            min_features = self.max_total_features // 2
            
            if num_active_tracks < min_features:
                new_features = self._extract_grid_features(img_gray)
                
                if len(new_features) > 0:
                    existing_pts = np.array([hist[-1]['pt'] for hist in self.tracks.values() 
                                            if hist and hist[-1]['frame'] == self.frame_idx])
                    
                    if len(existing_pts) > 0:
                        try:
                            from scipy.spatial.distance import cdist
                            dists = cdist(new_features, existing_pts)
                            min_dists = np.min(dists, axis=1)
                            far_enough = min_dists > 10.0
                            new_features = new_features[far_enough]
                        except Exception:
                            pass
                    
                    new_fids = []
                    for p in new_features:
                        fid = self.next_fid
                        self.next_fid += 1
                        self.tracks[fid] = [{'frame': self.frame_idx, 'pt': (float(p[0]), float(p[1])), 'quality': 1.0}]
                        new_fids.append(fid)
                    
                    pts_all = []
                    fids_all = []
                    for fid, hist in self.tracks.items():
                        if hist and hist[-1]['frame'] == self.frame_idx:
                            pts_all.append(hist[-1]['pt'])
                            fids_all.append(fid)
                    
                    if len(pts_all) > 0:
                        self.last_pts_for_klt = np.array(pts_all, dtype=np.float32)
                        self.last_fids_for_klt = np.array(fids_all, dtype=np.int64)
                        self.last_gray_for_klt = img_gray.copy()
                        
                    print(f"[VIO][REPLENISH] Added {len(new_features)} grid-based features")
        except Exception as e:
            print(f"[VIO][REPLENISH] Exception: {e}")

    def get_tracks_for_frame(self, frame_idx: int) -> List[Tuple[int, Tuple[float, float]]]:
        """Return list of (fid, pt) observed at given frame_idx."""
        res = []
        for fid, hist in self.tracks.items():
            if hist and hist[-1]['frame'] == frame_idx:
                res.append((fid, hist[-1]['pt']))
        return res
    
    def get_mature_tracks(self) -> Dict[int, List[dict]]:
        """Return tracks ready for MSCKF update (length >= min_track_length)."""
        mature = {}
        for fid, hist in self.tracks.items():
            if len(hist) >= self.min_track_length:
                mature[fid] = hist
        return mature
    
    def get_track_by_id(self, fid: int) -> Optional[List[dict]]:
        """Get track history for a specific feature ID."""
        return self.tracks.get(fid, None)
    
    def get_multi_view_observations(self, fid: int, cam_ids: List[int], 
                                    cam_states: List[dict]) -> List[dict]:
        """Get observations of a feature across multiple camera poses."""
        track = self.tracks.get(fid, None)
        if track is None:
            return []
        
        observations = []
        for obs in track:
            frame_idx = obs['frame']
            for i, cs in enumerate(cam_states):
                if cs['frame'] == frame_idx and i in cam_ids:
                    observations.append({
                        'cam_id': i,
                        'frame': frame_idx,
                        'pt': obs['pt'],
                        'quality': obs.get('quality', 1.0)
                    })
                    break
        return observations
