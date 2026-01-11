#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIO Data Loaders Module

Data loading utilities for IMU, MAG, VPS, DEM, and PPK files.
"""

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer


# =============================================================================
# Projection Cache
# =============================================================================

class ProjectionCache:
    """
    Projection cache for coordinate transformation.
    
    Manages projection transformers between WGS84 (lat/lon) and local 
    Azimuthal Equidistant coordinates (XY in meters).
    
    Use this class directly when you need isolated projection state 
    (e.g., multiple VIO sessions). For single-session use, the module-level 
    wrapper functions (latlon_to_xy, xy_to_latlon) are available.
    
    Example:
        # Using class directly
        proj = ProjectionCache()
        xy = proj.latlon_to_xy(35.0, 139.0, 35.0, 139.0)
        
        # Using wrapper functions (backward compatible)
        from vio.data_loaders import latlon_to_xy
        xy = latlon_to_xy(35.0, 139.0, 35.0, 139.0)
    """
    
    def __init__(self):
        """Initialize empty projection cache."""
        self._origin: Optional[Tuple[float, float]] = None
        self._to_xy: Optional[Transformer] = None
        self._to_ll: Optional[Transformer] = None
    
    def ensure_proj(self, origin_lat: float, origin_lon: float) -> None:
        """Initialize local projection centered at origin if needed."""
        if self._origin != (origin_lat, origin_lon) or self._to_xy is None:
            crs_wgs84 = CRS.from_epsg(4326)
            crs_aeqd = CRS.from_proj4(
                f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} +datum=WGS84 +units=m +no_defs"
            )
            self._to_xy = Transformer.from_crs(crs_wgs84, crs_aeqd, always_xy=True)
            self._to_ll = Transformer.from_crs(crs_aeqd, crs_wgs84, always_xy=True)
            self._origin = (origin_lat, origin_lon)
    
    def latlon_to_xy(self, lat: float, lon: float, 
                     origin_lat: float, origin_lon: float) -> np.ndarray:
        """Convert lat/lon to local XY (meters) centered at origin."""
        self.ensure_proj(origin_lat, origin_lon)
        x, y = self._to_xy.transform(lon, lat)
        return np.array([x, y], dtype=float)
    
    def xy_to_latlon(self, px: float, py: float, 
                     origin_lat: float, origin_lon: float) -> Tuple[float, float]:
        """Convert local XY (meters) to lat/lon."""
        self.ensure_proj(origin_lat, origin_lon)
        lon, lat = self._to_ll.transform(px, py)
        return float(lat), float(lon)
    
    def reset(self) -> None:
        """Reset cache for new session."""
        self._origin = None
        self._to_xy = None
        self._to_ll = None
    
    @property
    def origin(self) -> Optional[Tuple[float, float]]:
        """Current origin (lat, lon) or None if not initialized."""
        return self._origin


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IMURecord:
    """Single IMU measurement."""
    t: float  # timestamp (seconds)
    q: np.ndarray  # quaternion [x,y,z,w]
    ang: np.ndarray  # angular velocity [wx,wy,wz] rad/s
    lin: np.ndarray  # linear acceleration [ax,ay,az] m/s²
    
    @property
    def w(self) -> np.ndarray:
        """Alias for angular velocity (gyro)."""
        return self.ang
    
    @property
    def a(self) -> np.ndarray:
        """Alias for linear acceleration."""
        return self.lin


@dataclass
class MagRecord:
    """Single magnetometer measurement."""
    t: float  # timestamp (seconds)
    mag: np.ndarray  # magnetic field [x,y,z] in body frame


@dataclass
class ImageItem:
    """Image file with timestamp."""
    t: float  # timestamp (seconds)
    path: str  # file path


@dataclass
class VPSItem:
    """Visual Positioning System measurement."""
    t: float  # timestamp (seconds)
    lat: float  # latitude (degrees)
    lon: float  # longitude (degrees)


@dataclass
class PPKInitialState:
    """Initial state from PPK ground truth file (GPS-denied: single t=0 value).
    
    v3.9.0: All values come from a single row at t=0, ensuring GPS-denied compliance.
    Uses yaw directly from PPK file (attitude yaw, not computed).
    """
    lat: float  # degrees
    lon: float  # degrees
    height: float  # meters (ellipsoidal)
    roll: float  # radians (NED frame)
    pitch: float  # radians (NED frame)
    yaw: float  # radians (NED frame) - attitude yaw
    ve: float  # m/s East velocity
    vn: float  # m/s North velocity
    vu: float  # m/s Up velocity
    timestamp: str  # GPST timestamp string


# =============================================================================
# DEM Reader
# =============================================================================

@dataclass
class DEMReader:
    """Digital Elevation Model reader for GeoTIFF files."""
    ds: Optional[rasterio.io.DatasetReader]
    to_raster: Optional[Transformer]

    @classmethod
    def open(cls, path: Optional[str]) -> 'DEMReader':
        """Open DEM file."""
        if not path or not os.path.exists(path):
            return cls(None, None)
        ds = rasterio.open(path)
        to_raster = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        return cls(ds, to_raster)

    def sample_m(self, lat: float, lon: float) -> Optional[float]:
        """Sample elevation at lat/lon. Returns meters or None."""
        if self.ds is None:
            return None
        x, y = self.to_raster.transform(lon, lat)
        nodata = self.ds.nodatavals[0]
        try:
            for v in self.ds.sample([(x, y)]):
                h = float(v[0])
                if h is not None and h != nodata and math.isfinite(h):
                    return h
                return None
        except Exception:
            return None


# =============================================================================
# Loader Functions
# =============================================================================

def load_imu_csv(path: str) -> List[IMURecord]:
    """Load IMU data from CSV file.
    
    v3.9.0: Uses time_ref (hardware monotonic clock) as unified timestamp.
    time_ref = Hardware monotonic clock synchronized via PPS (Pulse Per Second)
    stamp_msg = ROS header.stamp (may have clock offset between sensors)
    stamp_bag = ROS bag recording time (has network/disk latency)
    
    Priority: time_ref (unified HW clock) > stamp_bag (ROS synchronized) > stamp_msg (has offset)
    
    Background: MUN-FRL dataset has clock desynchronization between IMU and Camera
    when using stamp_msg (offset drifts 47ms over 308s), causing VIO failure.
    time_ref solves this by providing a single hardware clock source for all sensors.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"IMU CSV not found: {path}")
    
    df = pd.read_csv(path)
    
    # Priority: time_ref (unified HW) > stamp_bag (ROS sync) > stamp_msg (has offset)
    if "time_ref" in df.columns:
        t_col = "time_ref"
        print(f"[IMU] Using unified hardware clock (time_ref) - monotonic clock")
    elif "stamp_bag" in df.columns:
        t_col = "stamp_bag"
        print(f"[IMU] Using ROS synchronized timestamp (stamp_bag)")
    elif "stamp_msg" in df.columns:
        t_col = "stamp_msg"
        print(f"[IMU] WARNING: Using stamp_msg (may have clock offset with camera)")
    elif "stamp_log" in df.columns:
        t_col = "stamp_log"
        print(f"[IMU] WARNING: Using legacy timestamp (stamp_log) - deprecated")
    else:
        raise ValueError(f"IMU CSV missing timestamp column (time_ref, stamp_bag, stamp_msg, or stamp_log)")
    
    # Check required columns
    cols = [t_col, "ori_x", "ori_y", "ori_z", "ori_w",
            "ang_x", "ang_y", "ang_z", "lin_x", "lin_y", "lin_z"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"IMU CSV missing column: {c}")
    
    df = df.sort_values(t_col).reset_index(drop=True)
    recs = []
    for _, r in df.iterrows():
        recs.append(IMURecord(
            t=float(r[t_col]),
            q=np.array([r["ori_x"], r["ori_y"], r["ori_z"], r["ori_w"]], dtype=float),
            ang=np.array([r["ang_x"], r["ang_y"], r["ang_z"]], dtype=float),
            lin=np.array([r["lin_x"], r["lin_y"], r["lin_z"]], dtype=float),
        ))
    
    print(f"[IMU] Loaded {len(recs)} samples using {t_col}")
    return recs


def interpolate_time_ref(t_ros_array: np.ndarray, 
                        time_ref_pairs: List[Tuple[float, float]]) -> np.ndarray:
    """
    Interpolate hardware timestamps from ROS timestamps using time_ref mapping.
    
    CRITICAL: This ensures magnetometer uses same time base as IMU (hardware clock).
    
    Args:
        t_ros_array: ROS timestamps to convert (stamp_msg from magnetometer)
        time_ref_pairs: List of (t_ros, t_hw) from /imu/time_ref topic
    
    Returns:
        Hardware timestamps corresponding to t_ros_array
        
    Implementation:
        - Pre-computes numpy arrays for O(N log M) complexity instead of O(N×M)
        - Uses linear interpolation between nearest time_ref samples
        - Handles edge cases: before first sample → use first, after last → use last
        - Guards against division by zero when consecutive timestamps are identical
    """
    if len(time_ref_pairs) == 0:
        return t_ros_array  # Fallback: no conversion
    
    # Pre-compute arrays once (avoid list comprehension in loop)
    t_ros_refs = np.array([pair[0] for pair in time_ref_pairs])
    t_hw_refs = np.array([pair[1] for pair in time_ref_pairs])
    
    # Find insertion indices for each query timestamp
    indices = np.searchsorted(t_ros_refs, t_ros_array)
    
    # Vectorized interpolation
    t_hw_array = np.zeros_like(t_ros_array)
    
    for i, (t_ros, idx) in enumerate(zip(t_ros_array, indices)):
        # Edge case 1: before first time_ref sample
        if idx == 0:
            t_hw_array[i] = t_hw_refs[0]
            continue
        
        # Edge case 2: after last time_ref sample
        if idx >= len(t_ros_refs):
            t_hw_array[i] = t_hw_refs[-1]
            continue
        
        # Normal case: linear interpolation
        t_ros_0 = t_ros_refs[idx - 1]
        t_ros_1 = t_ros_refs[idx]
        t_hw_0 = t_hw_refs[idx - 1]
        t_hw_1 = t_hw_refs[idx]
        
        # Guard against division by zero
        dt_ros = t_ros_1 - t_ros_0
        if dt_ros < 1e-9:  # Duplicate timestamps
            t_hw_array[i] = t_hw_0
        else:
            alpha = (t_ros - t_ros_0) / dt_ros
            t_hw_array[i] = t_hw_0 + alpha * (t_hw_1 - t_hw_0)
    
    return t_hw_array


def load_mag_csv(path: Optional[str], timeref_csv: Optional[str] = None) -> List[MagRecord]:
    """
    Load magnetometer data from vector3.csv.
    
    v3.9.4: Uses time_ref to synchronize with IMU clock (same as camera).
    
    CRITICAL: Magnetometer MUST use same time base as IMU for correct EKF updates.
    Without time_ref, stamp_msg may have offset → incorrect propagation → filter divergence.
    
    Args:
        path: Path to magnetometer CSV (usually extracted_data/imu_data/imu__mag/vector3.csv)
        timeref_csv: Path to time_ref CSV from /imu/time_ref topic (hardware clock mapping)
    
    Returns:
        List of MagRecord with hardware-synchronized timestamps
    """
    if not path or not os.path.exists(path):
        return []
    
    df = pd.read_csv(path)
    
    # Find timestamp column (priority: time_ref > stamp_bag > stamp_msg)
    if "time_ref" in df.columns:
        tcol = "time_ref"
        print(f"[Mag] Using unified hardware clock (time_ref) - already synchronized!")
    elif "stamp_bag" in df.columns:
        tcol = "stamp_bag"
        print(f"[Mag] Using stamp_bag (ROS recording time)")
    elif "stamp_msg" in df.columns:
        tcol = "stamp_msg"
        print(f"[Mag] Using stamp_msg (header.stamp from sensor)")
    else:
        print(f"[Mag] WARNING: No timestamp column found")
        return []
    
    if 'x' not in df.columns:
        print(f"[Mag] WARNING: vector3.csv missing magnetic field columns")
        return []
    
    df = df.sort_values(tcol).reset_index(drop=True)
    
    # If using stamp_msg or stamp_bag and time_ref CSV is available, convert to hardware time
    if tcol in ["stamp_msg", "stamp_bag"] and timeref_csv and os.path.exists(timeref_csv):
        try:
            # Load time_ref mapping
            timeref_df = pd.read_csv(timeref_csv)
            
            # Try to find matching timestamp column in time_ref CSV
            if "time_ref" in timeref_df.columns:
                # Use the same column name that exists in timeref CSV
                if tcol in timeref_df.columns:
                    ros_col = tcol
                elif "stamp_msg" in timeref_df.columns:
                    ros_col = "stamp_msg"
                elif "stamp_bag" in timeref_df.columns:
                    ros_col = "stamp_bag"
                else:
                    ros_col = None
                    print(f"[Mag] WARNING: time_ref CSV missing ROS timestamp column")
                
                if ros_col:
                    time_ref_pairs = list(zip(
                        timeref_df[ros_col].values,
                        timeref_df["time_ref"].values
                    ))
                    
                    # Convert all magnetometer timestamps using interpolation
                    t_ros_array = df[tcol].values
                    t_hw_array = interpolate_time_ref(t_ros_array, time_ref_pairs)
                    
                    # Replace timestamp column with hardware time
                    df["time_ref"] = t_hw_array
                    tcol = "time_ref"
                    
                    print(f"[Mag] Converted {len(df)} timestamps: {ros_col} → time_ref (hardware clock)")
                    print(f"[Mag] Time range: {t_hw_array[0]:.3f} → {t_hw_array[-1]:.3f} s")
            else:
                print(f"[Mag] WARNING: time_ref CSV missing time_ref column")
        except Exception as e:
            print(f"[Mag] WARNING: Failed to load time_ref mapping: {e}")
    
    # Build MagRecord list
    recs = []
    for _, r in df.iterrows():
        recs.append(MagRecord(
            t=float(r[tcol]),
            mag=np.array([r['x'], r['y'], r['z']], dtype=float)
        ))
    
    print(f"[Mag] Loaded {len(recs)} magnetometer samples using {tcol}")
    return recs


def load_images(images_dir: Optional[str], index_csv: Optional[str], 
                timeref_csv: Optional[str] = None) -> List[ImageItem]:
    """Load image list from directory and index CSV.
    
    v3.9.0: Uses time_ref (hardware monotonic clock) to match IMU timestamps.
    time_ref = Hardware monotonic clock (same as IMU - synchronized)
    stamp_bag = ROS bag recording time (synchronized by ROS)
    stamp_msg = camera trigger time (may have offset from IMU clock)
    
    Priority: timeref_csv with time_ref > stamp_bag from index > stamp_msg (has offset)
    
    Args:
        images_dir: Directory containing image files
        index_csv: CSV with stamp_bag/stamp_msg and filename (actual captured images)
        timeref_csv: CSV with time_ref for camera triggers (includes dropped frames)
    """
    if not images_dir or not os.path.isdir(images_dir):
        print(f"[Images] Directory not found: {images_dir}")
        return []
    if not index_csv or not os.path.exists(index_csv):
        print(f"[Images] Index CSV not found: {index_csv}")
        return []
    
    # Load actual image list (4625 images)
    df_images = pd.read_csv(index_csv)
    
    # Check if we have timeref.csv with time_ref (6149 camera triggers)
    if timeref_csv and os.path.exists(timeref_csv):
        print(f"[Images] Loading camera time_ref from {os.path.basename(timeref_csv)}")
        df_timeref = pd.read_csv(timeref_csv)
        
        # Match each image to its closest timeref entry to get time_ref
        if "time_ref" in df_timeref.columns and "stamp_msg" in df_timeref.columns:
            # Use stamp_msg to match between images_index and timeref
            match_col = "stamp_msg" if "stamp_msg" in df_images.columns else "stamp_bag"
            
            timestamps = []
            filenames = []
            fcol = next((c for c in df_images.columns if "file" in c.lower() or "name" in c.lower()), None)
            
            for _, img_row in df_images.iterrows():
                img_stamp = float(img_row[match_col])
                # Find closest timeref entry (within 50ms for 20Hz timeref @ 50ms interval)
                time_diffs = np.abs(df_timeref["stamp_msg"].values - img_stamp)
                closest_idx = np.argmin(time_diffs)
                
                if time_diffs[closest_idx] < 0.050:  # 50ms threshold (relaxed from 20ms)
                    time_ref = float(df_timeref["time_ref"].iloc[closest_idx])
                    timestamps.append(time_ref)
                    filenames.append(str(img_row[fcol]).strip())
            
            print(f"[Images] Matched {len(timestamps)}/{len(df_images)} images to time_ref")
            print(f"[Images] Using unified hardware clock (time_ref) - synchronized with IMU")
            
            # Build items using matched data
            items = []
            skipped = 0
            for ts, fn_raw in zip(timestamps, filenames):
                candidates = [
                    os.path.join(images_dir, fn_raw),
                    os.path.join(images_dir, os.path.basename(fn_raw)),
                    fn_raw
                ]
                chosen = None
                for cpath in candidates:
                    if os.path.exists(cpath):
                        chosen = cpath
                        break
                if chosen is None:
                    skipped += 1
                    continue
                items.append(ImageItem(ts, chosen))
            
            items.sort(key=lambda x: x.t)
            print(f"[Images] Loaded {len(items)} images using time_ref | Missing: {skipped}")
            return items
    
    # Fallback: use timestamps directly from images_index.csv
    # Priority: stamp_bag (ROS sync) > stamp_msg (has offset)
    if "stamp_bag" in df_images.columns:
        tcol = "stamp_bag"
        print(f"[Images] Using ROS synchronized timestamp (stamp_bag)")
    elif "stamp_msg" in df_images.columns:
        tcol = "stamp_msg"
        print(f"[Images] WARNING: Using stamp_msg (may have clock offset with IMU)")
    else:
        # Legacy: auto-detect timestamp column
        t_cols = [c for c in df_images.columns if c.lower().startswith("stamp") or c.lower().startswith("time")]
        tcol = t_cols[0] if t_cols else None
    
    f_cols = [c for c in df_images.columns if "file" in c.lower() or "name" in c.lower()]
    fcol = f_cols[0] if f_cols else None
    
    if tcol is None or fcol is None:
        raise ValueError("images_index.csv must have timestamp and filename columns")
    
    items = []
    skipped = 0
    for _, r in df_images.iterrows():
        try:
            ts = float(str(r[tcol]).strip())
        except Exception:
            continue
        
        fn_raw = str(r[fcol]).strip()
        candidates = [
            os.path.join(images_dir, fn_raw),
            os.path.join(images_dir, os.path.basename(fn_raw)),
            fn_raw
        ]
        
        chosen = None
        for cpath in candidates:
            if os.path.exists(cpath):
                chosen = cpath
                break
        
        if chosen is None:
            skipped += 1
            continue
        items.append(ImageItem(ts, chosen))
    
    items.sort(key=lambda x: x.t)
    print(f"[Images] Loaded {len(items)} images using {tcol} | Missing: {skipped}")
    return items


def load_vps_csv(path: Optional[str]) -> List[VPSItem]:
    """Load VPS data from CSV file."""
    if not path or not os.path.exists(path):
        return []
    
    items = []
    try:
        df = pd.read_csv(path)
        tcol = next((c for c in df.columns if c.lower().startswith("t")), None)
        latcol = next((c for c in df.columns if c.lower().startswith("lat")), None)
        loncol = next((c for c in df.columns if c.lower().startswith("lon")), None)
        
        if tcol and latcol and loncol:
            for _, r in df.iterrows():
                items.append(VPSItem(float(r[tcol]), float(r[latcol]), float(r[loncol])))
    except Exception:
        # Fallback: raw text parsing
        with open(path, "r") as f:
            for line in f:
                p = [x.strip() for x in line.strip().split(",")]
                if len(p) >= 3:
                    try:
                        items.append(VPSItem(float(p[0]), float(p[1]), float(p[2])))
                    except Exception:
                        continue
    
    items.sort(key=lambda x: x.t)
    return items


def load_ppk_initial_state(path: str) -> Optional[PPKInitialState]:
    """Load initial state from PPK ground truth file (.pos).
    
    v3.9.0: GPS-DENIED COMPLIANT - All values from single row at t=0.
    
    Args:
        path: Path to PPK .pos file
        
    Returns:
        PPKInitialState with initial position, attitude, and velocity
    """
    if path is None or not os.path.exists(path):
        return None
    
    try:
        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('%')]
        
        if len(lines) == 0:
            return None
        
        parts = lines[0].split()
        if len(parts) < 24:
            return None
        
        timestamp = f"{parts[0]} {parts[1]}"
        
        state = PPKInitialState(
            lat=float(parts[2]),
            lon=float(parts[3]),
            height=float(parts[4]),
            roll=np.radians(float(parts[15])),
            pitch=np.radians(float(parts[16])),
            yaw=np.radians(float(parts[17])),
            ve=float(parts[21]),
            vn=float(parts[22]),
            vu=float(parts[23]),
            timestamp=timestamp
        )
        
        vel_mag = np.sqrt(state.ve**2 + state.vn**2 + state.vu**2)
        
        print(f"\n[PPK] Initial state loaded (GPS-denied compliant: single t=0 value):")
        print(f"  Position: lat={state.lat:.6f}°, lon={state.lon:.6f}°, h={state.height:.1f}m")
        print(f"  Attitude (NED): roll={np.degrees(state.roll):.1f}°, "
              f"pitch={np.degrees(state.pitch):.1f}°, yaw={np.degrees(state.yaw):.1f}°")
        print(f"  Velocity: ve={state.ve:.2f}, vn={state.vn:.2f}, vu={state.vu:.2f} m/s (|v|={vel_mag:.2f})")
        
        return state
        
    except Exception as e:
        print(f"[WARN] Failed to load PPK: {e}")
        return None



def load_ppk_trajectory(path: str) -> Optional[pd.DataFrame]:
    """Load full PPK trajectory as DataFrame."""
    if path is None or not os.path.exists(path):
        return None
    
    try:
        from datetime import datetime
        import calendar
        
        rows = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                
                parts = line.split()
                if len(parts) < 24:
                    continue
                
                gpst_str = f"{parts[0]} {parts[1]}"
                try:
                    dt = datetime.strptime(gpst_str, "%Y/%m/%d %H:%M:%S.%f")
                    stamp = calendar.timegm(dt.timetuple()) + dt.microsecond / 1e6
                except Exception:
                    continue
                
                try:
                    rows.append({
                        'stamp_log': stamp,
                        'lat': float(parts[2]),
                        'lon': float(parts[3]),
                        'height': float(parts[4]),
                        'roll': np.radians(float(parts[15])),
                        'pitch': np.radians(float(parts[16])),
                        'yaw': np.radians(float(parts[17])),
                        've': float(parts[21]),
                        'vn': float(parts[22]),
                        'vu': float(parts[23]),
                    })
                except Exception:
                    continue
        
        if len(rows) == 0:
            return None
        
        df = pd.DataFrame(rows)
        print(f"[PPK] Loaded {len(df)} trajectory points")
        return df
        
    except Exception as e:
        print(f"[WARN] Failed to load PPK trajectory: {e}")
        return None


def load_msl_from_gga(path: str) -> Optional[float]:
    """Load initial MSL altitude from GGA file."""
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path)
        msl_col = next((c for c in df.columns if "altitude_msl_m" in c.lower()), None)
        
        if msl_col is None:
            return None
        
        msl_m = float(df[msl_col].iloc[0])
        print(f"[GGA] Initial MSL: {msl_m:.2f}m")
        return msl_m
        
    except Exception:
        return None


def load_quarry_initial(path: str) -> Tuple[float, float, float, np.ndarray]:
    """Load initial position and velocity from GGA file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"GGA file not found: {path}")
    
    df = pd.read_csv(path)
    
    lat_col = next((c for c in df.columns if c.lower() in ["lat", "lat_dd"]), None)
    lon_col = next((c for c in df.columns if c.lower() in ["lon", "lon_dd"]), None)
    msl_col = next((c for c in df.columns if "altitude_msl_m" in c.lower()), None)
    
    if lat_col is None or lon_col is None or msl_col is None:
        raise ValueError("GGA file must have lat_dd, lon_dd, altitude_MSL_m")
    
    row0 = df.iloc[0]
    lat0 = float(row0[lat_col])
    lon0 = float(row0[lon_col])
    msl_m = float(row0[msl_col])
    
    # Read velocity if available
    v_init = np.zeros(3, dtype=float)
    vx_col = next((c for c in df.columns if "xspeed" in c.lower()), None)
    vy_col = next((c for c in df.columns if "yspeed" in c.lower()), None)
    vz_col = next((c for c in df.columns if "zspeed" in c.lower()), None)
    
    if vx_col and vy_col and vz_col:
        v_init = np.array([
            float(row0[vx_col]) * 0.44704,  # mph to m/s
            float(row0[vy_col]) * 0.44704,
            float(row0[vz_col]) * 0.44704,
        ], dtype=float)
    
    print(f"[GGA] Initial: lat={lat0:.6f}°, lon={lon0:.6f}°, MSL={msl_m:.1f}m")
    return lat0, lon0, msl_m, v_init
