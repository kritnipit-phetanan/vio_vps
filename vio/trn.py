#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terrain Referenced Navigation (TRN) Module
==========================================

Provides XY position fixes using terrain profile matching against DEM/DSM.
This is a GPS-denied compliant navigation aid that uses pre-loaded terrain data.

Algorithm Overview:
-------------------
1. Collect altitude history over a window (e.g., 30-60 seconds)
2. Extract terrain profile from current estimated position
3. Search nearby positions for best terrain profile match
4. Apply position correction to EKF when good match found

Key Concepts:
- Uses radar/baro altimeter for AGL (Above Ground Level) measurements
- Compares measured AGL profile against DEM terrain
- Best match location = most likely actual position

For helicopters:
- Works well when terrain has variation (hills, valleys)
- Less effective over flat terrain (ocean, plains)
- 30m DSM resolution adequate for regional navigation

References:
- Bergman, N. "Recursive Bayesian Estimation: Navigation and Tracking Applications"
- Hostetler, L. and Andreas, R. "Nonlinear Kalman Filtering Techniques for Terrain-Aided Navigation"

Author: VIO project
Version: 3.3.0
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from scipy.ndimage import correlate1d
from scipy.optimize import minimize


@dataclass
class TRNConfig:
    """Configuration for Terrain Referenced Navigation."""
    
    # Enable/disable TRN
    enabled: bool = True
    
    # Altitude history window for profile matching
    profile_window_sec: float = 30.0  # seconds of altitude history
    min_samples: int = 20  # minimum samples for valid profile
    
    # Search grid parameters
    search_radius_m: float = 500.0  # meters - search radius around estimate
    search_step_m: float = 30.0  # meters - grid step (match DEM resolution)
    
    # Matching thresholds
    min_terrain_variation_m: float = 10.0  # minimum terrain variation for match
    max_correlation_threshold: float = 0.7  # correlation threshold for valid fix
    min_altitude_variation_m: float = 5.0  # minimum altitude variation in profile
    
    # Update rate
    update_interval_sec: float = 10.0  # seconds between TRN updates
    
    # Measurement noise
    sigma_trn_xy: float = 50.0  # meters - position uncertainty from TRN


@dataclass
class TRNState:
    """Runtime state for TRN."""
    
    # Altitude history
    altitude_history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    # Each entry: (timestamp, estimated_msl, estimated_x, estimated_y)
    
    # Last update
    last_update_time: float = 0.0
    last_fix_position: Optional[np.ndarray] = None
    
    # Statistics
    total_updates: int = 0
    successful_fixes: int = 0
    failed_matches: int = 0


class TerrainReferencedNavigation:
    """
    Terrain Referenced Navigation (TRN) for GPS-denied position fixes.
    
    Uses terrain profile correlation to estimate XY position from:
    - Measured altitude (MSL or AGL)
    - DEM/DSM terrain data
    - Estimated position history
    """
    
    def __init__(self, dem_reader, config: Optional[TRNConfig] = None):
        """
        Initialize TRN.
        
        Args:
            dem_reader: DEMReader instance from data_loaders
            config: TRNConfig (uses defaults if None)
        """
        self.dem = dem_reader
        self.config = config or TRNConfig()
        self.state = TRNState()
        
        # Cache for terrain profiles
        self._terrain_cache: Dict[Tuple[int, int], float] = {}
        
    def update_altitude_history(self, 
                                timestamp: float,
                                msl_altitude: float,
                                estimated_x: float,
                                estimated_y: float):
        """
        Add new altitude measurement to history.
        
        Args:
            timestamp: Current time (seconds)
            msl_altitude: Mean Sea Level altitude (meters)
            estimated_x: Current estimated X position (meters, ENU)
            estimated_y: Current estimated Y position (meters, ENU)
        """
        self.state.altitude_history.append((
            timestamp, msl_altitude, estimated_x, estimated_y
        ))
        
        # Prune old entries
        min_time = timestamp - self.config.profile_window_sec
        self.state.altitude_history = [
            entry for entry in self.state.altitude_history
            if entry[0] >= min_time
        ]
    
    def check_update_needed(self, current_time: float) -> bool:
        """Check if TRN update is needed based on interval."""
        if not self.config.enabled:
            return False
        if self.dem.ds is None:
            return False
        time_since_update = current_time - self.state.last_update_time
        return time_since_update >= self.config.update_interval_sec
    
    def compute_position_fix(self,
                             lat0: float, lon0: float,
                             current_x: float, current_y: float,
                             current_msl: float) -> Optional[Tuple[np.ndarray, float, float]]:
        """
        Compute TRN position fix using terrain profile matching.
        
        Args:
            lat0, lon0: Origin coordinates (for local→latlon conversion)
            current_x, current_y: Current estimated position (meters, ENU)
            current_msl: Current MSL altitude (meters)
        
        Returns:
            If successful: (position_correction [dx, dy], innovation_x, innovation_y)
            If failed: None
        """
        if not self.config.enabled or self.dem.ds is None:
            return None
        
        # Check we have enough altitude history
        if len(self.state.altitude_history) < self.config.min_samples:
            return None
        
        # Extract altitude profile
        profile = self._extract_altitude_profile()
        if profile is None:
            return None
        
        # Check altitude variation (flat profile = no information)
        alt_variation = np.std([p[1] for p in profile])
        if alt_variation < self.config.min_altitude_variation_m:
            return None
        
        # Search for best matching position
        best_pos, correlation = self._search_best_match(
            profile, lat0, lon0, current_x, current_y
        )
        
        if best_pos is None or correlation < self.config.max_correlation_threshold:
            self.state.failed_matches += 1
            return None
        
        # Compute position correction
        dx = best_pos[0] - current_x
        dy = best_pos[1] - current_y
        
        self.state.successful_fixes += 1
        self.state.last_fix_position = best_pos
        
        return np.array([dx, dy]), dx, dy
    
    def _extract_altitude_profile(self) -> Optional[List[Tuple[float, float, float, float]]]:
        """Extract altitude profile from history."""
        if len(self.state.altitude_history) < self.config.min_samples:
            return None
        
        # Sample evenly from history
        n = len(self.state.altitude_history)
        step = max(1, n // self.config.min_samples)
        profile = [self.state.altitude_history[i] for i in range(0, n, step)]
        
        return profile
    
    def _search_best_match(self,
                           profile: List[Tuple[float, float, float, float]],
                           lat0: float, lon0: float,
                           center_x: float, center_y: float
                           ) -> Tuple[Optional[np.ndarray], float]:
        """
        Search for best terrain profile match around current position.
        
        Uses grid search + refinement for efficiency.
        
        Args:
            profile: Altitude profile [(t, msl, x, y), ...]
            lat0, lon0: Origin coordinates
            center_x, center_y: Search center (current estimate)
        
        Returns:
            best_position: Best matching [x, y] or None
            correlation: Correlation score (0-1)
        """
        radius = self.config.search_radius_m
        step = self.config.search_step_m
        
        # Grid search
        best_corr = -1.0
        best_pos = None
        
        # Generate search grid
        x_range = np.arange(center_x - radius, center_x + radius + step, step)
        y_range = np.arange(center_y - radius, center_y + radius + step, step)
        
        for test_x in x_range:
            for test_y in y_range:
                # Compute offset from profile positions
                offset_x = test_x - center_x
                offset_y = test_y - center_y
                
                # Compute correlation at this offset
                corr = self._compute_terrain_correlation(
                    profile, lat0, lon0, offset_x, offset_y
                )
                
                if corr > best_corr:
                    best_corr = corr
                    best_pos = np.array([test_x, test_y])
        
        # Local refinement (optional - for better accuracy)
        if best_pos is not None and best_corr > 0.5:
            def neg_corr(offset):
                return -self._compute_terrain_correlation(
                    profile, lat0, lon0, offset[0], offset[1]
                )
            
            result = minimize(
                neg_corr,
                x0=[best_pos[0] - center_x, best_pos[1] - center_y],
                method='Nelder-Mead',
                options={'maxiter': 20, 'xatol': 5.0}
            )
            
            if result.success and -result.fun > best_corr:
                best_pos = np.array([
                    center_x + result.x[0],
                    center_y + result.x[1]
                ])
                best_corr = -result.fun
        
        return best_pos, best_corr
    
    def _compute_terrain_correlation(self,
                                     profile: List[Tuple[float, float, float, float]],
                                     lat0: float, lon0: float,
                                     offset_x: float, offset_y: float) -> float:
        """
        Compute correlation between measured altitude profile and terrain.
        
        Args:
            profile: Altitude profile [(t, msl, x, y), ...]
            lat0, lon0: Origin coordinates
            offset_x, offset_y: Position offset to test
        
        Returns:
            correlation: Correlation coefficient (0-1)
        """
        measured_alts = []
        terrain_alts = []
        
        for t, msl, x, y in profile:
            # Apply offset to get test position
            test_x = x + offset_x
            test_y = y + offset_y
            
            # Convert to lat/lon
            # Simple local tangent plane approximation
            lat = lat0 + test_y / 111320.0
            lon = lon0 + test_x / (111320.0 * np.cos(np.radians(lat0)))
            
            # Sample terrain
            terrain_h = self.dem.sample_m(lat, lon)
            if terrain_h is None:
                continue
            
            # Compute expected AGL (MSL - terrain)
            measured_agl = msl - terrain_h
            measured_alts.append(measured_agl)
            terrain_alts.append(terrain_h)
        
        if len(measured_alts) < 5:
            return 0.0
        
        # Normalize and compute correlation
        measured = np.array(measured_alts)
        terrain = np.array(terrain_alts)
        
        # Check terrain variation at test location
        if np.std(terrain) < self.config.min_terrain_variation_m:
            return 0.0  # Flat terrain = no information
        
        # Correlation based on AGL consistency
        # If position is correct, AGL should be relatively constant
        agl_std = np.std(measured)
        
        # Lower AGL variation = better match (assuming constant flight altitude)
        # Normalize to 0-1 range
        max_expected_std = 50.0  # Expected AGL std for wrong position
        correlation = max(0.0, 1.0 - agl_std / max_expected_std)
        
        return correlation
    
    def apply_trn_update(self,
                         kf,
                         lat0: float, lon0: float,
                         current_time: float) -> bool:
        """
        Apply TRN position fix to EKF.
        
        Args:
            kf: ExtendedKalmanFilter instance
            lat0, lon0: Origin coordinates
            current_time: Current timestamp
        
        Returns:
            success: True if TRN update was applied
        """
        if not self.check_update_needed(current_time):
            return False
        
        # Get current state
        pos = kf.x[0:3, 0]
        current_x, current_y, current_z = pos[0], pos[1], pos[2]
        
        # Get current MSL (assuming z_state = MSL)
        current_msl = current_z
        
        # Compute position fix
        result = self.compute_position_fix(
            lat0, lon0, current_x, current_y, current_msl
        )
        
        if result is None:
            return False
        
        correction, dx, dy = result
        
        # Apply EKF update
        # Observation: [x_measured, y_measured]
        z = np.array([[current_x + dx], [current_y + dy]])
        
        # Observation matrix: observe position directly
        H = np.zeros((2, kf.P.shape[0]))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        
        # Measurement noise
        R = np.eye(2) * (self.config.sigma_trn_xy ** 2)
        
        # Predicted measurement
        z_pred = np.array([[current_x], [current_y]])
        
        # Innovation
        y = z - z_pred
        
        # Kalman gain
        S = H @ kf.P @ H.T + R
        try:
            K = kf.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        
        # State update (error-state)
        dx_state = K @ y
        
        # Apply correction to nominal state
        kf.x[0, 0] += dx_state[0, 0]  # x
        kf.x[1, 0] += dx_state[1, 0]  # y
        
        # Covariance update
        I_KH = np.eye(kf.P.shape[0]) - K @ H
        kf.P = I_KH @ kf.P @ I_KH.T + K @ R @ K.T
        
        # Update state
        self.state.last_update_time = current_time
        self.state.total_updates += 1
        
        print(f"[TRN] Position fix applied: dx={dx:.1f}m, dy={dy:.1f}m, "
              f"total_fixes={self.state.successful_fixes}")
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TRN statistics."""
        return {
            'total_updates': self.state.total_updates,
            'successful_fixes': self.state.successful_fixes,
            'failed_matches': self.state.failed_matches,
            'history_length': len(self.state.altitude_history),
            'last_fix_position': self.state.last_fix_position,
        }


def create_trn_from_config(dem_reader, global_config: dict) -> Optional[TerrainReferencedNavigation]:
    """
    Create TRN instance from YAML config.
    
    Args:
        dem_reader: DEMReader instance
        global_config: Parsed YAML config dict
    
    Returns:
        TRN instance if enabled, None otherwise
    """
    trn_cfg = global_config.get('trn', {})
    
    if not trn_cfg.get('enabled', False):
        return None
    
    config = TRNConfig(
        enabled=trn_cfg.get('enabled', True),
        profile_window_sec=trn_cfg.get('profile_window_sec', 30.0),
        min_samples=trn_cfg.get('min_samples', 20),
        search_radius_m=trn_cfg.get('search_radius_m', 500.0),
        search_step_m=trn_cfg.get('search_step_m', 30.0),
        min_terrain_variation_m=trn_cfg.get('min_terrain_variation_m', 10.0),
        max_correlation_threshold=trn_cfg.get('max_correlation_threshold', 0.7),
        min_altitude_variation_m=trn_cfg.get('min_altitude_variation_m', 5.0),
        update_interval_sec=trn_cfg.get('update_interval_sec', 10.0),
        sigma_trn_xy=trn_cfg.get('sigma_trn_xy', 50.0),
    )
    
    return TerrainReferencedNavigation(dem_reader, config)
