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

_proj_cache = {"origin": None, "to_xy": None, "to_ll": None}


def ensure_local_proj(origin_lat: float, origin_lon: float):
    """Initialize local projection centered at origin."""
    global _proj_cache
    if _proj_cache["origin"] != (origin_lat, origin_lon) or _proj_cache["to_xy"] is None:
        crs_wgs84 = CRS.from_epsg(4326)
        crs_aeqd = CRS.from_proj4(
            f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} +datum=WGS84 +units=m +no_defs"
        )
        _proj_cache["to_xy"] = Transformer.from_crs(crs_wgs84, crs_aeqd, always_xy=True)
        _proj_cache["to_ll"] = Transformer.from_crs(crs_aeqd, crs_wgs84, always_xy=True)
        _proj_cache["origin"] = (origin_lat, origin_lon)


def latlon_to_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> np.ndarray:
    """Convert lat/lon to local XY (meters) centered at origin."""
    ensure_local_proj(origin_lat, origin_lon)
    x, y = _proj_cache["to_xy"].transform(lon, lat)
    return np.array([x, y], dtype=float)


def xy_to_latlon(px: float, py: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """Convert local XY (meters) to lat/lon."""
    ensure_local_proj(origin_lat, origin_lon)
    lon, lat = _proj_cache["to_ll"].transform(px, py)
    return float(lat), float(lon)


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
    """Initial state from PPK ground truth file."""
    lat: float  # degrees
    lon: float  # degrees
    height: float  # meters (ellipsoidal)
    roll: float  # radians (NED frame)
    pitch: float  # radians (NED frame)
    yaw: float  # radians (NED frame)
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
    """Load IMU data from CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"IMU CSV not found: {path}")
    
    df = pd.read_csv(path)
    cols = ["stamp_log", "ori_x", "ori_y", "ori_z", "ori_w",
            "ang_x", "ang_y", "ang_z", "lin_x", "lin_y", "lin_z"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"IMU CSV missing column: {c}")
    
    df = df.sort_values("stamp_log").reset_index(drop=True)
    recs = []
    for _, r in df.iterrows():
        recs.append(IMURecord(
            t=float(r["stamp_log"]),
            q=np.array([r["ori_x"], r["ori_y"], r["ori_z"], r["ori_w"]], dtype=float),
            ang=np.array([r["ang_x"], r["ang_y"], r["ang_z"]], dtype=float),
            lin=np.array([r["lin_x"], r["lin_y"], r["lin_z"]], dtype=float),
        ))
    return recs


def load_mag_csv(path: Optional[str]) -> List[MagRecord]:
    """Load magnetometer data from vector3.csv."""
    if not path or not os.path.exists(path):
        return []
    
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if 'stamp' in c.lower()), None)
    if tcol is None or 'x' not in df.columns:
        print(f"[Mag] WARNING: vector3.csv missing required columns")
        return []
    
    df = df.sort_values(tcol).reset_index(drop=True)
    recs = []
    for _, r in df.iterrows():
        recs.append(MagRecord(
            t=float(r[tcol]),
            mag=np.array([r['x'], r['y'], r['z']], dtype=float)
        ))
    print(f"[Mag] Loaded {len(recs)} magnetometer samples")
    return recs


def load_images(images_dir: Optional[str], index_csv: Optional[str]) -> List[ImageItem]:
    """Load image list from directory and index CSV."""
    if not images_dir or not os.path.isdir(images_dir):
        print(f"[Images] Directory not found: {images_dir}")
        return []
    if not index_csv or not os.path.exists(index_csv):
        print(f"[Images] Index CSV not found: {index_csv}")
        return []
    
    df = pd.read_csv(index_csv)
    t_cols = [c for c in df.columns if c.lower().startswith("stamp") or c.lower().startswith("time")]
    f_cols = [c for c in df.columns if "file" in c.lower() or "name" in c.lower()]
    
    tcol = t_cols[0] if t_cols else None
    fcol = f_cols[0] if f_cols else None
    
    if tcol is None or fcol is None:
        raise ValueError("images_index.csv must have time/stamp and filename columns")
    
    items = []
    skipped = 0
    for _, r in df.iterrows():
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
    print(f"[Images] Valid: {len(items)} | Missing: {skipped}")
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
    """Load initial state from PPK ground truth file (.pos)."""
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
        
        print(f"\n[PPK] Initial state loaded:")
        print(f"  Position: lat={state.lat:.6f}°, lon={state.lon:.6f}°")
        print(f"  Attitude (NED): roll={np.degrees(state.roll):.1f}°, "
              f"pitch={np.degrees(state.pitch):.1f}°, yaw={np.degrees(state.yaw):.1f}°")
        return state
        
    except Exception as e:
        print(f"[WARN] Failed to load PPK: {e}")
        return None


def get_ppk_initial_heading(ppk_trajectory: Optional[pd.DataFrame], 
                            lat0: float, lon0: float, 
                            duration: float = 30.0) -> Optional[float]:
    """Extract initial heading from PPK trajectory (first 30s).
    
    v2.9.10.0: Use ONLY initial heading from ground truth for initialization.
    This complies with GPS-denied constraints - we use GT only as initializer,
    not for continuous updates.
    
    Args:
        ppk_trajectory: PPK ground truth DataFrame
        lat0: Origin latitude (degrees)
        lon0: Origin longitude (degrees)
        duration: Duration to extract heading from (seconds)
    
    Returns:
        Median heading in radians (ENU frame), or None if unavailable
    """
    if ppk_trajectory is None or len(ppk_trajectory) == 0:
        return None
    
    try:
        # Extract first 30s of trajectory
        t_start = ppk_trajectory['stamp_log'].min()
        ppk_30s = ppk_trajectory[
            ppk_trajectory['stamp_log'] <= t_start + duration
        ].copy()
        
        if len(ppk_30s) < 5:  # Need at least 5 samples
            return None
        
        # Convert lat/lon to local ENU coordinates
        lats = ppk_30s['lat'].values
        lons = ppk_30s['lon'].values
        
        x_vals = []
        y_vals = []
        for lat, lon in zip(lats, lons):
            xy = latlon_to_xy(lat, lon, lat0, lon0)
            x_vals.append(xy[0])
            y_vals.append(xy[1])
        
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        
        # Compute heading from velocity vector
        dx = np.diff(x_vals)
        dy = np.diff(y_vals)
        dt = np.diff(ppk_30s['stamp_log'].values)
        
        # Avoid division by zero
        valid_idx = dt > 1e-6
        if not np.any(valid_idx):
            return None
        
        vx = dx[valid_idx] / dt[valid_idx]
        vy = dy[valid_idx] / dt[valid_idx]
        
        # Filter out stationary periods (velocity < 0.5 m/s)
        vel_mag = np.sqrt(vx**2 + vy**2)
        moving_idx = vel_mag > 0.5
        
        if not np.any(moving_idx):
            return None
        
        vx = vx[moving_idx]
        vy = vy[moving_idx]
        
        # Compute heading: atan2(vy, vx) in ENU frame
        headings = np.arctan2(vy, vx)
        
        # Use median to avoid outliers
        median_heading = np.median(headings)
        
        print(f"[PPK Init Heading] Extracted from first {duration}s:")
        print(f"  Samples: {len(headings)} moving periods")
        print(f"  Heading: {np.degrees(median_heading):.1f}° (ENU)")
        print(f"  Std: {np.degrees(np.std(headings)):.1f}°")
        
        return float(median_heading)
        
    except Exception as e:
        print(f"[WARN] Failed to extract PPK initial heading: {e}")
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
