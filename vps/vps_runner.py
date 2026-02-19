#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPS Runner - Main Orchestrator

Coordinates all VPS components for position estimation.
Called at camera frame rate from VIO main loop.

Author: VIO project
"""

import os
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from .tile_cache import TileCache, MapPatch
from .image_preprocessor import VPSImagePreprocessor, PreprocessResult
from .satellite_matcher import SatelliteMatcher, MatchResult
from .vps_pose_estimator import VPSPoseEstimator, VPSMeasurement
from .relocalization_policy import (
    build_relocalization_centers,
    extract_xy_sigma_m,
    should_force_global_relocalization,
)


@dataclass
class VPSConfig:
    """Configuration for VPS system."""
    mbtiles_path: str
    
    # Processing
    output_size: tuple = (512, 512)
    patch_size_px: int = 512
    patch_size_failover_px: int = 768
    failover_fail_streak: int = 6
    
    # Quality thresholds
    min_inliers: int = 20
    min_inliers_failsoft: int = 5
    max_reproj_error: float = 5.0
    max_reproj_error_failsoft: float = 1.2
    min_confidence: float = 0.3
    min_confidence_failsoft: float = 0.12
    max_offset_px_failsoft: float = 900.0
    failsoft_r_inflate: float = 4.0
    allow_failsoft_accept: bool = True
    
    # Altitude limits (AGL)
    min_altitude: float = 30.0
    max_altitude: float = 500.0
    
    # Timing
    min_update_interval: float = 0.5  # seconds

    # Multi-hypothesis preprocess policy
    yaw_hypotheses_deg: tuple = (0.0, 180.0, 90.0, -90.0)
    scale_hypotheses: tuple = (1.0, 0.90, 1.10)
    max_candidates: int = 6
    min_content_ratio: float = 0.20
    min_texture_std: float = 8.0
    
    # Matcher
    device: str = 'cuda'
    max_keypoints: int = 2048
    matcher_mode: str = "auto"  # auto|orb|lightglue|orb_lightglue_rescue
    rescue_min_inliers: int = 8
    rescue_min_confidence: float = 0.12
    rescue_max_reproj_error: float = 2.5
    max_image_side: int = 1024
    mps_cache_clear_interval: int = 0

    # Accuracy-first controls
    accuracy_mode: bool = False
    global_max_candidates: int = 12

    # Relocalization (local/global search)
    reloc_enabled: bool = True
    reloc_global_interval_sec: float = 12.0
    reloc_fail_streak_trigger: int = 6
    reloc_stale_success_sec: float = 8.0
    reloc_xy_sigma_trigger_m: float = 35.0
    reloc_max_centers: int = 10
    reloc_ring_radius_m: tuple = (35.0, 80.0)
    reloc_ring_samples: int = 8
    reloc_global_yaw_hypotheses_deg: tuple = (0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0)
    reloc_global_scale_hypotheses: tuple = (0.80, 0.90, 1.00, 1.10, 1.20)
    reloc_force_global_on_warning_phase: bool = False
    runtime_verbosity: str = "debug"
    runtime_log_interval_sec: float = 1.0


class VPSRunner:
    """
    Main VPS orchestrator.
    
    Workflow:
    1. Get map patch from TileCache centered at VIO estimate
    2. Preprocess drone image (undistort, rotate, scale)
    3. Match drone image against satellite map
    4. Estimate lat/lon position with covariance
    
    Called from VIO main loop when camera frame is available.
    """
    
    def __init__(self,
                 mbtiles_path: str,
                 fisheye_rectifier=None,
                 camera_intrinsics: Optional[Dict] = None,
                 config: Optional[VPSConfig] = None,
                 device: str = 'cuda',
                 camera_yaw_offset_rad: float = 0.0,
                 save_matches_dir: Optional[str] = None):
        """
        Initialize VPS runner.
        
        Args:
            mbtiles_path: Path to cached satellite tiles
            fisheye_rectifier: Optional FisheyeRectifier from vio module
            camera_intrinsics: Camera intrinsics (KB_PARAMS dict)
            config: VPSConfig, uses defaults if None
            device: 'cuda' or 'cpu' for matcher
            camera_yaw_offset_rad: Camera-to-body yaw offset in radians
                                   (from extrinsics calibration)
            save_matches_dir: Optional directory to save match visualization images
        """
        # Config
        if config is None:
            config = VPSConfig(mbtiles_path=mbtiles_path, device=device)
        self.config = config
        self._runtime_verbosity = str(self.config.runtime_verbosity).lower()
        self._runtime_quiet = self._runtime_verbosity in ("release", "quiet", "minimal")
        self._runtime_log_interval_sec = max(0.0, float(self.config.runtime_log_interval_sec))
        self._last_runtime_log_ts: Dict[str, float] = {}
        
        # Initialize components
        print("[VPSRunner] Initializing...")
        
        # 1. Tile cache
        self.tile_cache = TileCache(mbtiles_path)
        print(f"[VPSRunner] TileCache loaded: {self.tile_cache.get_tile_count()} tiles")
        
        # 2. Image preprocessor
        self.preprocessor = VPSImagePreprocessor(
            fisheye_rectifier=fisheye_rectifier,
            camera_intrinsics=camera_intrinsics,
            output_size=config.output_size
        )
        
        # 3. Satellite matcher
        self.matcher = SatelliteMatcher(
            device=config.device,
            max_keypoints=config.max_keypoints,
            min_inliers=config.min_inliers,
            match_mode=config.matcher_mode,
            rescue_min_inliers=config.rescue_min_inliers,
            rescue_min_confidence=config.rescue_min_confidence,
            rescue_max_reproj_error=config.rescue_max_reproj_error,
            max_image_side=config.max_image_side,
            mps_cache_clear_interval=config.mps_cache_clear_interval,
        )
        
        # 4. Pose estimator
        self.pose_estimator = VPSPoseEstimator()
        
        # Camera-Body extrinsics offset
        self.camera_yaw_offset_rad = camera_yaw_offset_rad
        if abs(camera_yaw_offset_rad) > 0.01:
            import math
            print(f"[VPSRunner] Camera yaw offset: {math.degrees(camera_yaw_offset_rad):.2f}°")
        
        # State
        self.last_update_time = 0.0
        self.last_result: Optional[VPSMeasurement] = None
        self._fail_streak = 0
        self._last_success_time = -1e9
        self._last_global_search_time = -1e9
        self.reloc_summary_csv: Optional[str] = None
        
        # Debug logger (set via set_logger)
        self.logger = None
        
        # Save match visualizations directory
        self.save_matches_dir = save_matches_dir
        if save_matches_dir:
            import os
            os.makedirs(save_matches_dir, exist_ok=True)
            print(f"[VPSRunner] Match visualizations will be saved to: {save_matches_dir}")
        
        # Flag for delayed update (stochastic cloning) - set by caller
        self.delayed_update_enabled = False
        
        # Statistics
        self.stats = {
            'total_attempts': 0,
            'success': 0,
            'fail_no_coverage': 0,
            'fail_match': 0,
            'fail_quality': 0,
            'global_search': 0,
            'local_search': 0,
        }
        
        print("[VPSRunner] Ready")

    def _runtime_log(self, key: str, msg: str, force: bool = False):
        """Rate-limited logging for per-frame runtime messages."""
        if force or not self._runtime_quiet:
            print(msg)
            self._last_runtime_log_ts[key] = time.time()
            return
        now = time.time()
        last = self._last_runtime_log_ts.get(key, -1e9)
        if (now - last) >= self._runtime_log_interval_sec:
            print(msg)
            self._last_runtime_log_ts[key] = now
    
    @classmethod
    def create_from_config(cls, 
                           mbtiles_path: str,
                           config_path: str,
                           device: str = 'cuda') -> 'VPSRunner':
        """
        Factory method to create VPSRunner from VIO config file.
        
        Automatically loads camera intrinsics and creates fisheye rectifier.
        
        Args:
            mbtiles_path: Path to MBTiles file
            config_path: Path to VIO YAML config file
            device: 'cuda' or 'cpu' for matcher
            
        Returns:
            VPSRunner instance with proper camera configuration
            
        Example:
            vps = VPSRunner.create_from_config(
                "mission.mbtiles",
                "configs/config_bell412_dataset3.yaml"
            )
        """
        import yaml
        import os
        
        fisheye_rectifier = None
        camera_intrinsics = None
        camera_yaw_offset_rad = 0.0
        vps_cfg: Dict[str, Any] = {}
        logging_cfg: Dict[str, Any] = {}
        vps_config = VPSConfig(mbtiles_path=mbtiles_path, device=device)
        
        # Load camera intrinsics from config
        if os.path.exists(config_path):
            print(f"[VPSRunner] Loading config: {config_path}")
            with open(config_path, 'r') as f:
                config_yaml = yaml.safe_load(f)
            
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
                print(f"[VPSRunner] Camera: mu={camera_intrinsics['mu']:.1f}, "
                      f"mv={camera_intrinsics['mv']:.1f}")
            vps_cfg = config_yaml.get("vps", {}) if isinstance(config_yaml, dict) else {}
            logging_cfg = config_yaml.get("logging", {}) if isinstance(config_yaml, dict) else {}
            if not isinstance(logging_cfg, dict):
                logging_cfg = {}
            
            # Create fisheye rectifier
            try:
                import sys
                # Add parent for vio imports
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from vio.fisheye_rectifier import create_rectifier_from_config
                
                rectifier_config = {'KB_PARAMS': camera_intrinsics}
                src_size = (camera_intrinsics['w'], camera_intrinsics['h'])
                fisheye_rectifier = create_rectifier_from_config(
                    rectifier_config, 
                    src_size=src_size,
                    fov_deg=90.0
                )
                print(f"[VPSRunner] Fisheye rectifier created (FOV=90°)")
            except Exception as e:
                print(f"[VPSRunner] Warning: Fisheye rectifier failed: {e}")
            
            # Load extrinsics and calculate camera yaw offset
            try:
                import math
                # Get camera view from VPS config (default: nadir)
                camera_view = vps_cfg.get('camera_view', 'nadir')
                
                if 'extrinsics' in config_yaml and camera_view in config_yaml['extrinsics']:
                    t_matrix = config_yaml['extrinsics'][camera_view]['transform']
                    R00 = t_matrix[0][0]
                    R10 = t_matrix[1][0]
                    
                    # Calculate yaw offset from rotation matrix
                    # atan2(R10, R00) gives azimuth of Camera X-axis (Right)
                    # Subtract 90° to get azimuth of Camera Up-axis (-Y)
                    yaw_x_axis = math.atan2(R10, R00)
                    camera_yaw_offset_rad = yaw_x_axis - math.radians(90)
                    
                    print(f"[VPSRunner] Extrinsics loaded: {camera_view} -> "
                          f"yaw offset = {math.degrees(camera_yaw_offset_rad):.2f}°")
            except Exception as e:
                print(f"[VPSRunner] Warning: Failed to load extrinsics: {e}")
        else:
            print(f"[VPSRunner] Warning: Config not found: {config_path}")

        # Build VPS config from YAML (single source of truth).
        if isinstance(vps_cfg, dict):
            output_size = tuple(vps_cfg.get("output_size", [512, 512]))
            if len(output_size) != 2:
                output_size = (512, 512)
            yaw_hyp = tuple(float(v) for v in vps_cfg.get("yaw_hypotheses_deg", [0.0, 180.0, 90.0, -90.0]))
            scale_hyp = tuple(float(v) for v in vps_cfg.get("scale_hypotheses", [1.0, 0.90, 1.10]))
            reloc_cfg = vps_cfg.get("relocalization", {}) if isinstance(vps_cfg.get("relocalization", {}), dict) else {}
            reloc_ring_radius = tuple(float(v) for v in reloc_cfg.get("ring_radius_m", [35.0, 80.0]))
            reloc_global_yaw = tuple(float(v) for v in reloc_cfg.get(
                "global_yaw_hypotheses_deg", [0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0]
            ))
            reloc_global_scale = tuple(float(v) for v in reloc_cfg.get(
                "global_scale_hypotheses", [0.80, 0.90, 1.00, 1.10, 1.20]
            ))

            runtime_verbosity_default = os.environ.get("VIO_RUNTIME_VERBOSITY", "debug")
            runtime_log_interval_default = float(
                os.environ.get("VIO_RUNTIME_LOG_INTERVAL_SEC", "1.0")
            )

            vps_config = VPSConfig(
                mbtiles_path=mbtiles_path,
                output_size=(int(output_size[0]), int(output_size[1])),
                patch_size_px=int(vps_cfg.get("patch_size_px", output_size[0])),
                patch_size_failover_px=int(vps_cfg.get("patch_size_failover_px", max(output_size[0], 768))),
                failover_fail_streak=int(vps_cfg.get("failover_fail_streak", 6)),
                min_inliers=int(vps_cfg.get("min_inliers", 20)),
                min_inliers_failsoft=int(vps_cfg.get("min_inliers_failsoft", 5)),
                max_reproj_error=float(vps_cfg.get("max_reproj_error", 5.0)),
                max_reproj_error_failsoft=float(vps_cfg.get("max_reproj_error_failsoft", 1.2)),
                min_confidence=float(vps_cfg.get("min_confidence", 0.3)),
                min_confidence_failsoft=float(vps_cfg.get("min_confidence_failsoft", 0.12)),
                max_offset_px_failsoft=float(vps_cfg.get("max_offset_px_failsoft", 900.0)),
                failsoft_r_inflate=float(vps_cfg.get("failsoft_r_inflate", 4.0)),
                allow_failsoft_accept=bool(vps_cfg.get("allow_failsoft_accept", True)),
                min_altitude=float(vps_cfg.get("min_altitude", 30.0)),
                max_altitude=float(vps_cfg.get("max_altitude", 500.0)),
                min_update_interval=float(vps_cfg.get("min_update_interval", 0.5)),
                device=str(vps_cfg.get("device", device)),
                max_keypoints=int(vps_cfg.get("max_keypoints", 2048)),
                matcher_mode=str(vps_cfg.get("matcher_mode", "auto")),
                rescue_min_inliers=int(vps_cfg.get("rescue_min_inliers", 8)),
                rescue_min_confidence=float(vps_cfg.get("rescue_min_confidence", 0.12)),
                rescue_max_reproj_error=float(vps_cfg.get("rescue_max_reproj_error", 2.5)),
                max_image_side=int(vps_cfg.get("max_image_side", 1024)),
                mps_cache_clear_interval=int(vps_cfg.get("mps_cache_clear_interval", 0)),
                yaw_hypotheses_deg=yaw_hyp if len(yaw_hyp) > 0 else (0.0, 180.0, 90.0, -90.0),
                scale_hypotheses=scale_hyp if len(scale_hyp) > 0 else (1.0, 0.90, 1.10),
                max_candidates=int(vps_cfg.get("max_candidates", 6)),
                global_max_candidates=int(vps_cfg.get("global_max_candidates", 12)),
                min_content_ratio=float(vps_cfg.get("min_content_ratio", 0.20)),
                min_texture_std=float(vps_cfg.get("min_texture_std", 8.0)),
                accuracy_mode=bool(vps_cfg.get("accuracy_mode", False)),
                reloc_enabled=bool(reloc_cfg.get("enabled", True)),
                reloc_global_interval_sec=float(reloc_cfg.get("global_interval_sec", 12.0)),
                reloc_fail_streak_trigger=int(reloc_cfg.get("fail_streak_trigger", 6)),
                reloc_stale_success_sec=float(reloc_cfg.get("stale_success_sec", 8.0)),
                reloc_xy_sigma_trigger_m=float(reloc_cfg.get("xy_sigma_trigger_m", 35.0)),
                reloc_max_centers=int(reloc_cfg.get("max_centers", 10)),
                reloc_ring_radius_m=reloc_ring_radius if len(reloc_ring_radius) > 0 else (35.0, 80.0),
                reloc_ring_samples=int(reloc_cfg.get("ring_samples", 8)),
                reloc_global_yaw_hypotheses_deg=reloc_global_yaw if len(reloc_global_yaw) > 0 else (0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0),
                reloc_global_scale_hypotheses=reloc_global_scale if len(reloc_global_scale) > 0 else (0.80, 0.90, 1.00, 1.10, 1.20),
                reloc_force_global_on_warning_phase=bool(reloc_cfg.get("force_global_on_warning_phase", False)),
                runtime_verbosity=str(
                    vps_cfg.get(
                        "runtime_verbosity",
                        logging_cfg.get("runtime_verbosity", runtime_verbosity_default),
                    )
                ),
                runtime_log_interval_sec=float(
                    vps_cfg.get(
                        "runtime_log_interval_sec",
                        logging_cfg.get("runtime_log_interval_sec", runtime_log_interval_default),
                    )
                ),
            )
            print(
                f"[VPSRunner] VPS cfg: patch={vps_config.patch_size_px}/{vps_config.patch_size_failover_px}, "
                f"min_inliers={vps_config.min_inliers}, failsoft={vps_config.min_inliers_failsoft}, "
                f"candidates<={vps_config.max_candidates}, matcher={vps_config.matcher_mode}, "
                f"max_image_side={vps_config.max_image_side}"
            )
        
        # Create VPSRunner
        return cls(
            mbtiles_path=mbtiles_path,
            fisheye_rectifier=fisheye_rectifier,
            camera_intrinsics=camera_intrinsics,
            config=vps_config,
            device=vps_config.device,
            camera_yaw_offset_rad=camera_yaw_offset_rad
        )

    def _select_patch_size(self) -> int:
        """Adaptive map patch size based on consecutive VPS failures."""
        base = max(256, int(self.config.patch_size_px))
        failover = max(base, int(self.config.patch_size_failover_px))
        if self._fail_streak >= int(self.config.failover_fail_streak):
            return failover
        return base

    def _build_preprocess_candidates(self,
                                     global_mode: bool = False,
                                     objective: str = "stability") -> List[Tuple[float, float]]:
        """
        Build (yaw_delta_deg, target_gsd_scale) candidates.
        Keeps runtime bounded with max_candidates.
        """
        if global_mode:
            yaw_list = [float(v) for v in self.config.reloc_global_yaw_hypotheses_deg]
            scale_list = [float(v) for v in self.config.reloc_global_scale_hypotheses]
            max_cands = max(1, int(self.config.global_max_candidates))
        else:
            yaw_list = [float(v) for v in self.config.yaw_hypotheses_deg]
            scale_list = [float(v) for v in self.config.scale_hypotheses]
            max_cands = max(1, int(self.config.max_candidates))

        if str(objective).lower() == "accuracy":
            max_cands = max(max_cands, int(self.config.max_candidates))
        if not yaw_list:
            yaw_list = [0.0]
        if not scale_list:
            scale_list = [1.0]

        # Prioritize primary candidate first, then yaw fix, then scale jitter.
        ordered: List[Tuple[float, float]] = [(0.0, 1.0)]
        for yaw_d in yaw_list:
            if abs(yaw_d) > 1e-6:
                ordered.append((yaw_d, 1.0))
        for sc in scale_list:
            if abs(sc - 1.0) > 1e-6:
                ordered.append((0.0, sc))

        # Cartesian fallback when fail streak grows.
        if self._fail_streak >= int(self.config.failover_fail_streak):
            for yaw_d in yaw_list:
                for sc in scale_list:
                    ordered.append((yaw_d, sc))

        dedup: List[Tuple[float, float]] = []
        seen = set()
        for yaw_d, sc in ordered:
            key = (round(float(yaw_d), 3), round(float(sc), 3))
            if key in seen:
                continue
            seen.add(key)
            dedup.append((float(yaw_d), float(sc)))
            if len(dedup) >= max_cands:
                break
        return dedup

    def _should_accept_failsoft(self, match_result: MatchResult) -> bool:
        """Allow low-inlier but geometrically coherent matches with conservative thresholds."""
        if not bool(self.config.allow_failsoft_accept):
            return False
        if match_result is None:
            return False
        if match_result.H is None:
            return False
        if int(match_result.num_inliers) < int(self.config.min_inliers_failsoft):
            return False
        if int(match_result.num_matches) < max(12, 2 * int(self.config.min_inliers_failsoft)):
            return False
        if not np.isfinite(float(match_result.reproj_error)):
            return False
        if float(match_result.reproj_error) > float(self.config.max_reproj_error_failsoft):
            return False
        if float(match_result.confidence) < float(self.config.min_confidence_failsoft):
            return False
        off = np.asarray(getattr(match_result, "offset_px", (0.0, 0.0)), dtype=float).reshape(-1)
        if off.size < 2 or not np.all(np.isfinite(off[:2])):
            return False
        off_norm_px = float(np.linalg.norm(off[:2]))
        if off_norm_px > float(self.config.max_offset_px_failsoft):
            return False
        return True

    def _log_reloc_summary(self,
                           t_cam: float,
                           frame_idx: int,
                           mode: str,
                           force_global: bool,
                           trigger_reason: str,
                           est_lat: float,
                           est_lon: float,
                           best_center_lat: float,
                           best_center_lon: float,
                           best_score: float,
                           best_inliers: int,
                           best_reproj: float,
                           best_conf: float,
                           selected_yaw_deg: float,
                           selected_scale_mult: float,
                           success: bool,
                           reason: str):
        """Append one VPS relocalization summary row."""
        if not self.reloc_summary_csv:
            return
        try:
            reason_txt = str(reason).replace(",", ";")
            trig_txt = str(trigger_reason).replace(",", ";")
            with open(self.reloc_summary_csv, "a", newline="") as f:
                f.write(
                    f"{float(t_cam):.6f},{int(frame_idx)},{mode},{int(force_global)},{trig_txt},"
                    f"{float(est_lat):.8f},{float(est_lon):.8f},"
                    f"{float(best_center_lat):.8f},{float(best_center_lon):.8f},"
                    f"{float(best_score):.6f},{int(best_inliers)},{float(best_reproj):.6f},"
                    f"{float(best_conf):.6f},{float(selected_yaw_deg):.3f},{float(selected_scale_mult):.4f},"
                    f"{int(success)},{reason_txt}\n"
                )
        except Exception:
            pass
    
    def process_frame(self,
                      img: np.ndarray,
                      t_cam: float,
                      est_lat: float,
                      est_lon: float,
                      est_yaw: float,
                      est_alt: float,
                      frame_idx: int = -1,
                      est_cov_xy: Optional[np.ndarray] = None,
                      phase: Optional[int] = None,
                      force_global: bool = False,
                      objective: str = "accuracy") -> Optional[VPSMeasurement]:
        """
        Process single camera frame for VPS position.
        
        This is the MAIN entry point, called from VIO main loop
        when a new camera frame is available.
        
        Args:
            img: Camera image (can be fisheye or rectified)
            t_cam: Camera timestamp
            est_lat: Estimated latitude from VIO
            est_lon: Estimated longitude from VIO
            est_yaw: Estimated yaw in radians (ENU)
            est_alt: Estimated altitude AGL in meters
            
        Returns:
            VPSMeasurement if successful, None otherwise
        
        Note:
            frame_idx: Optional frame index from VIO for logging. Default -1 if not provided.
        """
        t_start = time.time()
        self.stats['total_attempts'] += 1
        t_tile_start = t_start
        t_pose_start = t_start
        tile_ms = preprocess_ms = match_ms = pose_ms = 0.0
        patch_size_px = int(self._select_patch_size())

        num_candidates = 0
        selected_candidate_idx = -1
        selected_yaw_deg = float("nan")
        selected_scale_mult = float("nan")
        selected_reason = "match_failed"
        selected_preprocess: Optional[PreprocessResult] = None
        selected_match: Optional[MatchResult] = None

        best_num_matches = 0
        best_num_inliers = 0
        best_reproj = float("nan")
        best_conf = float("nan")
        best_score = -1e9
        selected_center_lat = float(est_lat)
        selected_center_lon = float(est_lon)
        selected_map_patch: Optional[MapPatch] = None

        force_global_requested = bool(force_global)
        objective_mode = str(objective).lower()
        if objective_mode == "":
            objective_mode = "accuracy" if bool(self.config.accuracy_mode) else "stability"

        since_success_sec = float(t_cam - self._last_success_time) if np.isfinite(self._last_success_time) else float("inf")
        since_global_sec = float(t_cam - self._last_global_search_time) if np.isfinite(self._last_global_search_time) else float("inf")
        global_mode, trigger_reason = should_force_global_relocalization(
            force_global=bool(force_global),
            accuracy_mode=bool(self.config.accuracy_mode),
            objective=objective_mode,
            reloc_enabled=bool(self.config.reloc_enabled),
            fail_streak=int(self._fail_streak),
            fail_streak_trigger=int(self.config.reloc_fail_streak_trigger),
            since_success_sec=since_success_sec,
            stale_success_sec=float(self.config.reloc_stale_success_sec),
            since_global_sec=since_global_sec,
            global_interval_sec=float(self.config.reloc_global_interval_sec),
            est_cov_xy=est_cov_xy,
            xy_sigma_trigger_m=float(self.config.reloc_xy_sigma_trigger_m),
            phase=phase,
            force_global_on_warning_phase=bool(self.config.reloc_force_global_on_warning_phase),
        )
        search_centers = (
            build_relocalization_centers(
                est_lat=float(est_lat),
                est_lon=float(est_lon),
                max_centers=int(self.config.reloc_max_centers),
                ring_radius_m=list(self.config.reloc_ring_radius_m),
                ring_samples=int(self.config.reloc_ring_samples),
            )
            if global_mode
            else [(float(est_lat), float(est_lon))]
        )
        if global_mode:
            self.stats['global_search'] += 1
            self._last_global_search_time = float(t_cam)
        else:
            self.stats['local_search'] += 1

        def _log_attempt_and_profile(success: bool, reason: str):
            processing_time_ms = (time.time() - t_start) * 1000.0
            if self.logger:
                import math
                self.logger.log_attempt(
                    t=t_cam,
                    frame=frame_idx,
                    est_lat=est_lat,
                    est_lon=est_lon,
                    est_alt=est_alt,
                    est_yaw_deg=math.degrees(est_yaw),
                    success=success,
                    reason=reason,
                    processing_time_ms=processing_time_ms,
                    patch_size_px=patch_size_px,
                    candidate_idx=selected_candidate_idx,
                    num_candidates=num_candidates,
                    selected_yaw_deg=selected_yaw_deg,
                    selected_scale_mult=selected_scale_mult,
                    selected_content_ratio=float(getattr(selected_preprocess, "content_ratio", float("nan"))),
                    selected_texture_std=float(getattr(selected_preprocess, "texture_std", float("nan"))),
                    best_num_matches=best_num_matches,
                    best_num_inliers=best_num_inliers,
                    best_reproj_error=best_reproj,
                    best_confidence=best_conf,
                )
                self.logger.log_profile(
                    t=t_cam,
                    frame=frame_idx,
                    success=success,
                    reason=reason,
                    total_ms=processing_time_ms,
                    tile_ms=tile_ms,
                    preprocess_ms=preprocess_ms,
                    match_ms=match_ms,
                    pose_ms=pose_ms,
                    num_matches=int(getattr(selected_match, "num_matches", 0)),
                    num_inliers=int(getattr(selected_match, "num_inliers", 0)),
                    reproj_error=float(getattr(selected_match, "reproj_error", float("nan"))),
                    confidence=float(getattr(selected_match, "confidence", float("nan"))),
                    content_ratio=float(getattr(selected_preprocess, "content_ratio", float("nan"))),
                    texture_std=float(getattr(selected_preprocess, "texture_std", float("nan"))),
                )
            return processing_time_ms

        def _save_match_visualization(tag: str,
                                      preprocess_result: Optional[PreprocessResult] = None,
                                      map_patch: Optional[MapPatch] = None,
                                      match_result: Optional[MatchResult] = None):
            """Persist VPS matching visualization for both success and failure cases."""
            if not self.save_matches_dir or frame_idx < 0:
                return
            if preprocess_result is None or map_patch is None or match_result is None:
                return
            try:
                import os
                safe_tag = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(tag))
                output_path = os.path.join(
                    self.save_matches_dir,
                    f"vps_{safe_tag}_{frame_idx:06d}_{t_cam:.2f}s.jpg",
                )
                self.matcher.visualize_matches(
                    drone_img=preprocess_result.image,
                    sat_img=map_patch.image,
                    result=match_result,
                    output_path=output_path,
                )
            except Exception as e:
                print(f"[VPS] Failed to save match visualization ({tag}): {e}")

        def _log_reloc(success: bool, reason: str):
            self._log_reloc_summary(
                t_cam=t_cam,
                frame_idx=frame_idx,
                mode="global" if global_mode else "local",
                force_global=force_global_requested,
                trigger_reason=trigger_reason,
                est_lat=est_lat,
                est_lon=est_lon,
                best_center_lat=selected_center_lat,
                best_center_lon=selected_center_lon,
                best_score=best_score,
                best_inliers=best_num_inliers,
                best_reproj=best_reproj,
                best_conf=best_conf,
                selected_yaw_deg=selected_yaw_deg,
                selected_scale_mult=selected_scale_mult,
                success=success,
                reason=reason,
            )
        
        # 1. Check update interval
        if t_cam - self.last_update_time < self.config.min_update_interval:
            return None
        
        # 2. Check altitude limits
        if est_alt < self.config.min_altitude or est_alt > self.config.max_altitude:
            _log_reloc(False, "altitude_out_of_range")
            return None

        if img is None or getattr(img, "size", 0) == 0:
            self.stats['fail_match'] += 1
            self._fail_streak += 1
            _log_attempt_and_profile(False, "empty_input_image")
            _log_reloc(False, "empty_input_image")
            return None

        # 3/4/5/6. Search centers + preprocess hypotheses + matching
        camera_yaw_base = est_yaw + self.camera_yaw_offset_rad
        candidates = self._build_preprocess_candidates(global_mode=global_mode, objective=objective_mode)
        num_candidates = len(candidates) * max(1, len(search_centers))
        stop_search = False
        coverage_found = False

        for center_idx, (center_lat, center_lon) in enumerate(search_centers):
            if not self.tile_cache.is_position_in_cache(center_lat, center_lon):
                continue

            t_tile_start = time.time()
            map_patch = self.tile_cache.get_map_patch(
                center_lat, center_lon,
                patch_size_px=patch_size_px
            )
            tile_ms += (time.time() - t_tile_start) * 1000.0
            if map_patch is None:
                continue
            if map_patch.image is None or map_patch.image.size == 0:
                continue
            if map_patch.image.shape[0] <= 1 or map_patch.image.shape[1] <= 1:
                continue
            if not np.isfinite(float(map_patch.meters_per_pixel)) or float(map_patch.meters_per_pixel) <= 0.0:
                continue
            coverage_found = True

            sat_img = map_patch.image if len(map_patch.image.shape) == 2 else map_patch.image[:, :, 0]
            for cand_idx, (yaw_delta_deg, scale_mult) in enumerate(candidates):
                candidate_idx = center_idx * max(1, len(candidates)) + cand_idx
                yaw_used_rad = float(camera_yaw_base + np.deg2rad(float(yaw_delta_deg)))

                t_preprocess_start = time.time()
                try:
                    preprocess_result = self.preprocessor.preprocess(
                        img=img,
                        yaw_rad=yaw_used_rad,
                        altitude_m=est_alt,
                        target_gsd=map_patch.meters_per_pixel,
                        grayscale=True,
                        target_gsd_scale=float(scale_mult),
                    )
                except Exception:
                    preprocess_ms += (time.time() - t_preprocess_start) * 1000.0
                    if self.logger:
                        self.logger.log_candidate(
                            t=t_cam,
                            frame=frame_idx,
                            candidate_idx=candidate_idx,
                            num_candidates=num_candidates,
                            patch_size_px=patch_size_px,
                            yaw_delta_deg=float(yaw_delta_deg),
                            scale_mult=float(scale_mult),
                            rotation_deg=float("nan"),
                            content_ratio=0.0,
                            texture_std=0.0,
                            num_matches=0,
                            num_inliers=0,
                            reproj_error=float("nan"),
                            confidence=float("nan"),
                            match_success=False,
                            decision="preprocess_failed",
                        )
                    continue
                preprocess_ms += (time.time() - t_preprocess_start) * 1000.0

                if preprocess_result.image is None or preprocess_result.image.size == 0:
                    if self.logger:
                        self.logger.log_candidate(
                            t=t_cam, frame=frame_idx, candidate_idx=candidate_idx, num_candidates=num_candidates,
                            patch_size_px=patch_size_px, yaw_delta_deg=float(yaw_delta_deg),
                            scale_mult=float(scale_mult), rotation_deg=float(preprocess_result.rotation_deg),
                            content_ratio=float(preprocess_result.content_ratio),
                            texture_std=float(preprocess_result.texture_std),
                            num_matches=0, num_inliers=0, reproj_error=float("nan"), confidence=float("nan"),
                            match_success=False, decision="empty_preprocessed",
                        )
                    continue

                if (
                    preprocess_result.content_ratio < float(self.config.min_content_ratio)
                    or preprocess_result.texture_std < float(self.config.min_texture_std)
                ):
                    if self.logger:
                        self.logger.log_candidate(
                            t=t_cam, frame=frame_idx, candidate_idx=candidate_idx, num_candidates=num_candidates,
                            patch_size_px=patch_size_px, yaw_delta_deg=float(yaw_delta_deg),
                            scale_mult=float(scale_mult), rotation_deg=float(preprocess_result.rotation_deg),
                            content_ratio=float(preprocess_result.content_ratio),
                            texture_std=float(preprocess_result.texture_std),
                            num_matches=0, num_inliers=0, reproj_error=float("nan"), confidence=float("nan"),
                            match_success=False, decision="low_content",
                        )
                    continue

                t_match_start = time.time()
                try:
                    match_result = self.matcher.match_with_homography(
                        drone_img=preprocess_result.image,
                        sat_img=sat_img,
                    )
                except Exception:
                    match_ms += (time.time() - t_match_start) * 1000.0
                    if self.logger:
                        self.logger.log_candidate(
                            t=t_cam, frame=frame_idx, candidate_idx=candidate_idx, num_candidates=num_candidates,
                            patch_size_px=patch_size_px, yaw_delta_deg=float(yaw_delta_deg),
                            scale_mult=float(scale_mult), rotation_deg=float(preprocess_result.rotation_deg),
                            content_ratio=float(preprocess_result.content_ratio),
                            texture_std=float(preprocess_result.texture_std),
                            num_matches=0, num_inliers=0, reproj_error=float("nan"), confidence=float("nan"),
                            match_success=False, decision="matcher_failed",
                        )
                    continue
                match_ms += (time.time() - t_match_start) * 1000.0

                num_m = int(getattr(match_result, "num_matches", 0))
                num_i = int(getattr(match_result, "num_inliers", 0))
                reproj = float(getattr(match_result, "reproj_error", float("nan")))
                conf = float(getattr(match_result, "confidence", 0.0))
                score = float(num_i) + 0.1 * float(num_m) + 6.0 * float(conf)
                if np.isfinite(reproj):
                    score -= 0.3 * max(0.0, reproj)

                if score > best_score:
                    best_score = score
                    best_num_matches = num_m
                    best_num_inliers = num_i
                    best_reproj = reproj
                    best_conf = conf
                    selected_candidate_idx = candidate_idx
                    selected_yaw_deg = float(np.degrees(yaw_used_rad))
                    selected_scale_mult = float(scale_mult)
                    selected_preprocess = preprocess_result
                    selected_match = match_result
                    selected_reason = "match_failed"
                    selected_center_lat = float(map_patch.center_lat)
                    selected_center_lon = float(map_patch.center_lon)
                    selected_map_patch = map_patch

                strict_ok = bool(match_result.success)
                failsoft_ok = (not strict_ok) and self._should_accept_failsoft(match_result)
                if failsoft_ok:
                    match_result.success = True

                decision = "matched_strict" if strict_ok else ("matched_failsoft" if failsoft_ok else "match_failed")
                if self.logger:
                    self.logger.log_candidate(
                        t=t_cam,
                        frame=frame_idx,
                        candidate_idx=candidate_idx,
                        num_candidates=num_candidates,
                        patch_size_px=patch_size_px,
                        yaw_delta_deg=float(yaw_delta_deg),
                        scale_mult=float(scale_mult),
                        rotation_deg=float(preprocess_result.rotation_deg),
                        content_ratio=float(preprocess_result.content_ratio),
                        texture_std=float(preprocess_result.texture_std),
                        num_matches=num_m,
                        num_inliers=num_i,
                        reproj_error=reproj,
                        confidence=conf,
                        match_success=bool(strict_ok or failsoft_ok),
                        decision=decision,
                    )

                if strict_ok or failsoft_ok:
                    accepted_score = score + (120.0 if strict_ok else 90.0)
                    if accepted_score >= best_score:
                        best_score = accepted_score
                        best_num_matches = num_m
                        best_num_inliers = num_i
                        best_reproj = reproj
                        best_conf = conf
                        selected_candidate_idx = candidate_idx
                        selected_yaw_deg = float(np.degrees(yaw_used_rad))
                        selected_scale_mult = float(scale_mult)
                        selected_preprocess = preprocess_result
                        selected_match = match_result
                        selected_reason = "matched_failsoft" if failsoft_ok else "matched"
                        selected_center_lat = float(map_patch.center_lat)
                        selected_center_lon = float(map_patch.center_lon)
                        selected_map_patch = map_patch
                    if objective_mode != "accuracy" and not global_mode:
                        stop_search = True
                        break
            # End candidate loop
            if stop_search:
                break
        # End center loop
        if stop_search:
            pass

        if selected_match is None:
            if coverage_found:
                self.stats['fail_match'] += 1
            else:
                self.stats['fail_no_coverage'] += 1
            self._fail_streak += 1
            fail_reason = "match_failed" if coverage_found else "no_coverage"
            _log_attempt_and_profile(False, fail_reason)
            _log_reloc(False, fail_reason)
            return None

        preprocess_result = selected_preprocess
        match_result = selected_match
        map_patch = selected_map_patch
        if selected_reason == "match_failed":
            self.stats['fail_match'] += 1
            self._fail_streak += 1
            _save_match_visualization("match_failed", preprocess_result, map_patch, match_result)
            _log_attempt_and_profile(False, "match_failed")
            _log_reloc(False, "match_failed")
            return None

        # 7. Quality check
        if selected_reason != "matched_failsoft" and match_result.num_inliers < self.config.min_inliers:
            self.stats['fail_quality'] += 1
            self._fail_streak += 1
            _save_match_visualization("quality_inliers", preprocess_result, map_patch, match_result)
            _log_attempt_and_profile(False, "quality_inliers")
            _log_reloc(False, "quality_inliers")
            return None
        
        if selected_reason != "matched_failsoft" and match_result.reproj_error > self.config.max_reproj_error:
            self.stats['fail_quality'] += 1
            self._fail_streak += 1
            _save_match_visualization("quality_reproj", preprocess_result, map_patch, match_result)
            _log_attempt_and_profile(False, "quality_reproj")
            _log_reloc(False, "quality_reproj")
            return None
        
        if selected_reason != "matched_failsoft" and match_result.confidence < self.config.min_confidence:
            self.stats['fail_quality'] += 1
            self._fail_streak += 1
            _save_match_visualization("quality_confidence", preprocess_result, map_patch, match_result)
            _log_attempt_and_profile(False, "quality_confidence")
            _log_reloc(False, "quality_confidence")
            return None
        
        # 8. Compute VPS measurement
        t_pose_start = time.time()
        try:
            vps_measurement = self.pose_estimator.compute_vps_measurement(
                match_result=match_result,
                map_gsd=map_patch.meters_per_pixel,
                map_center_lat=map_patch.center_lat,
                map_center_lon=map_patch.center_lon,
                t_cam=t_cam
            )
        except Exception:
            self.stats['fail_match'] += 1
            self._fail_streak += 1
            _log_attempt_and_profile(False, "pose_estimation_exception")
            _log_reloc(False, "pose_estimation_exception")
            return None
        pose_ms = (time.time() - t_pose_start) * 1000.0
        
        if vps_measurement is None:
            self.stats['fail_match'] += 1
            self._fail_streak += 1
            _save_match_visualization("pose_failed", preprocess_result, map_patch, match_result)
            _log_attempt_and_profile(False, "pose_estimation_failed")
            _log_reloc(False, "pose_estimation_failed")
            return None

        if selected_reason == "matched_failsoft":
            # Fail-soft matches carry higher uncertainty to avoid over-trusting weak geometry.
            inflate = max(1.0, float(self.config.failsoft_r_inflate))
            vps_measurement.R_vps = np.array(vps_measurement.R_vps, dtype=float) * inflate
        
        # Success!
        self.stats['success'] += 1
        self._fail_streak = 0
        self.last_update_time = t_cam
        self._last_success_time = float(t_cam)
        self.last_result = vps_measurement
        
        # Log processing time
        success_reason = "matched_failsoft" if selected_reason == "matched_failsoft" else "matched"
        processing_time_ms = _log_attempt_and_profile(True, success_reason)
        
        # Debug logging
        if self.logger:
            # Log match details
            self.logger.log_match(
                t=t_cam,
                frame=frame_idx,
                vps_lat=vps_measurement.lat,
                vps_lon=vps_measurement.lon,
                innovation_x=vps_measurement.offset_m[0],
                innovation_y=vps_measurement.offset_m[1],
                innovation_mag=float(np.linalg.norm(vps_measurement.offset_m)),
                num_features=match_result.num_matches,
                num_inliers=match_result.num_inliers,
                confidence=match_result.confidence,
                tile_zoom=19,  # Default zoom level
                delayed_update=self.delayed_update_enabled  # From stochastic cloning setup
            )
        
        # Console log
        self._runtime_log(
            "vps_result",
            f"[VPS] t={t_cam:.2f}: "
            f"cand={selected_candidate_idx+1}/{max(1, num_candidates)}, "
            f"patch={patch_size_px}px, "
            f"inliers={match_result.num_inliers}, "
            f"err={match_result.reproj_error:.2f}px, "
            f"Δ=({vps_measurement.offset_m[0]:.1f}, {vps_measurement.offset_m[1]:.1f})m, "
            f"σ={np.sqrt(vps_measurement.R_vps[0,0]):.2f}m, "
            f"time={processing_time_ms:.0f}ms",
        )
        
        # Save match visualization if directory is set
        if success_reason == "matched_failsoft":
            _save_match_visualization("matched_failsoft", preprocess_result, map_patch, match_result)
        else:
            _save_match_visualization("matched", preprocess_result, map_patch, match_result)
        _log_reloc(True, success_reason)
        
        return vps_measurement
    
    def get_last_result(self) -> Optional[VPSMeasurement]:
        """Get last successful VPS measurement."""
        return self.last_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total = self.stats['total_attempts']
        return {
            **self.stats,
            'success_rate': self.stats['success'] / total if total > 0 else 0,
        }
    
    def print_statistics(self):
        """Print statistics summary."""
        stats = self.get_statistics()
        print(f"\n[VPS] Statistics:")
        print(f"  Total attempts: {stats['total_attempts']}")
        print(f"  Success: {stats['success']} ({stats['success_rate']*100:.1f}%)")
        print(f"  No coverage: {stats['fail_no_coverage']}")
        print(f"  Match failed: {stats['fail_match']}")
        print(f"  Quality reject: {stats['fail_quality']}")
    
    def set_logger(self, logger):
        """
        Attach debug logger for VPS processing.
        
        Args:
            logger: VPSDebugLogger instance
        """
        self.logger = logger
    
    def close(self):
        """Release resources."""
        self.tile_cache.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_vps_runner():
    """Test VPS runner with proper camera config."""
    import sys
    import os
    import argparse
    
    # Parse arguments using argparse (consistent with run_vio.py)
    parser = argparse.ArgumentParser(
        description="VPS Runner Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults:
  python -m vps.vps_runner --mbtiles mission.mbtiles
  
  # With custom config:
  python -m vps.vps_runner --mbtiles mission.mbtiles --config configs/config.yaml
  
  # With custom position:
  python -m vps.vps_runner --mbtiles mission.mbtiles --lat 45.315 --lon -75.670
        """
    )
    
    parser.add_argument("--mbtiles", type=str, 
                        default="mission.mbtiles",
                        help="Path to MBTiles file (default: mission.mbtiles)")
    parser.add_argument("--config", type=str,
                        default="configs/config_bell412_dataset3.yaml",
                        help="Path to YAML config file (default: configs/config_bell412_dataset3.yaml)")
    parser.add_argument("--lat", type=float,
                        default=45.315721787845,
                        help="Test latitude (default: 45.315721787845)")
    parser.add_argument("--lon", type=float,
                        default=-75.670671305696,
                        help="Test longitude (default: -75.670671305696)")
    parser.add_argument("--device", type=str,
                        default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for matcher (default: cpu)")
    
    args = parser.parse_args()
    
    mbtiles_path = args.mbtiles
    config_path = args.config
    test_lat = args.lat
    test_lon = args.lon
    
    print("=" * 60)
    print("Testing VPSRunner with Real Camera Config")
    print("=" * 60)
    print(f"  MBTiles: {mbtiles_path}")
    print(f"  Config:  {config_path}")
    print(f"  Test position: ({test_lat:.6f}, {test_lon:.6f})")
    print(f"  Device: {args.device}")
    
    # Check if mbtiles exists
    if not os.path.exists(mbtiles_path):
        print(f"\n❌ Error: MBTiles file not found: {mbtiles_path}")
        print("Please provide a valid MBTiles file or create one using:")
        print("  python -m vps.tile_prefetcher --center LAT,LON --radius 500 -o mission.mbtiles")
        return
    
    # Try to load real image from bell412 dataset
    images_dir = "/Users/france/Downloads/vio_dataset/bell412_dataset3/extracted_data_new/cam_data/camera__image_mono/images"
    import cv2
    
    if os.path.exists(images_dir):
        import random
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        if image_files:
            selected_file = random.choice(image_files)
            img = cv2.imread(os.path.join(images_dir, selected_file))
            print(f"  Image: {selected_file} ({img.shape})")
        else:
            img = np.random.randint(50, 200, (1080, 1440, 3), dtype=np.uint8)
            print("  Image: synthetic (no images found)")
    else:
        img = np.random.randint(50, 200, (1080, 1440, 3), dtype=np.uint8)
        print("  Image: synthetic (dataset not found)")
    
    print("-" * 60)
    
    # Create VPSRunner with proper config
    if os.path.exists(config_path):
        vps = VPSRunner.create_from_config(
            mbtiles_path=mbtiles_path,
            config_path=config_path,
            device=args.device
        )
    else:
        print(f"Config not found, using defaults")
        vps = VPSRunner(mbtiles_path, device=args.device)
    
    with vps:
        result = vps.process_frame(
            img=img,
            t_cam=1.0,
            est_lat=test_lat,
            est_lon=test_lon,
            est_yaw=0.0,
            est_alt=100.0
        )
        
        if result:
            print(f"\n✅ VPS Result:")
            print(f"  Position: ({result.lat:.6f}, {result.lon:.6f})")
            print(f"  Offset: ({result.offset_m[0]:.2f}, {result.offset_m[1]:.2f}) m")
            print(f"  Sigma: {np.sqrt(result.R_vps[0,0]):.2f} m")
        else:
            print("\n⚠️ VPS returned no result (may need real imagery matching location)")
        
        vps.print_statistics()


if __name__ == "__main__":
    test_vps_runner()
