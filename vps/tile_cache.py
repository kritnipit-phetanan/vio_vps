#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tile Cache for In-Flight VPS

Reads satellite tiles from MBTiles and provides map patches for VPS matching.
Used during flight to query cached satellite imagery.

Author: VIO project
"""

import math
import sqlite3
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class MapPatch:
    """Result of get_map_patch() with image and metadata."""
    image: np.ndarray          # BGR image (OpenCV format)
    center_lat: float
    center_lon: float
    meters_per_pixel: float    # GSD at center latitude
    width_m: float             # Patch width in meters
    height_m: float            # Patch height in meters
    patch_origin_lat: float    # Top-left corner latitude
    patch_origin_lon: float    # Top-left corner longitude


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates (Web Mercator / Slippy Map)."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lat_lon(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to lat/lon (top-left corner of tile)."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def lat_lon_to_pixel_in_tile(lat: float, lon: float, tile_x: int, tile_y: int, 
                              zoom: int, tile_size: int = 256) -> Tuple[int, int]:
    """
    Get pixel position within a tile for given lat/lon.
    
    Args:
        lat, lon: Position to convert
        tile_x, tile_y: Tile coordinates
        zoom: Zoom level
        tile_size: Tile size in pixels (default 256)
        
    Returns:
        (px, py) pixel position within tile (0 to tile_size-1)
    """
    n = 2 ** zoom
    
    # Get fractional tile coordinates
    x_frac = (lon + 180.0) / 360.0 * n - tile_x
    y_frac = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n - tile_y
    
    # Convert to pixel
    px = int(x_frac * tile_size)
    py = int(y_frac * tile_size)
    
    # Clamp to valid range
    px = max(0, min(tile_size - 1, px))
    py = max(0, min(tile_size - 1, py))
    
    return px, py


def pixel_to_lat_lon(tile_x: int, tile_y: int, px: int, py: int, 
                     zoom: int, tile_size: int = 256) -> Tuple[float, float]:
    """
    Convert pixel position in tile to lat/lon.
    
    Args:
        tile_x, tile_y: Tile coordinates
        px, py: Pixel position within tile
        zoom: Zoom level
        tile_size: Tile size in pixels
        
    Returns:
        (lat, lon) tuple
    """
    n = 2 ** zoom
    
    # Fractional tile position
    x_frac = tile_x + px / tile_size
    y_frac = tile_y + py / tile_size
    
    # Convert to lat/lon
    lon = x_frac / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_frac / n))))
    
    return lat, lon


def get_tile_gsd(lat: float, zoom: int) -> float:
    """Get Ground Sample Distance (meters per pixel) for tile at given latitude."""
    earth_circumference = 40075016.686  # meters at equator
    tile_size = 256
    meters_per_pixel = earth_circumference * math.cos(math.radians(lat)) / (tile_size * (2 ** zoom))
    return meters_per_pixel


class TileCache:
    """
    In-flight tile cache for VPS matching.
    
    Reads tiles from MBTiles database and provides map patches for matching.
    Uses LRU cache for frequently accessed tiles.
    """
    
    TILE_SIZE = 256  # Standard Web Mercator tile size
    
    def __init__(self, mbtiles_path: str, max_cached_tiles: int = 50):
        """
        Initialize tile cache.
        
        Args:
            mbtiles_path: Path to .mbtiles file
            max_cached_tiles: Maximum tiles to keep in memory cache
        """
        if not Path(mbtiles_path).exists():
            raise FileNotFoundError(f"MBTiles not found: {mbtiles_path}")
        
        self.mbtiles_path = mbtiles_path
        self.conn = sqlite3.connect(mbtiles_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Get zoom level from metadata
        self.zoom = self._get_zoom_level()
        
        # Get bounds info
        self.bounds = self._get_bounds()
        
        # In-memory tile cache
        self._tile_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._max_cached = max_cached_tiles
        self._cache_order = []  # LRU order
        
        print(f"[TileCache] Loaded: {mbtiles_path}")
        print(f"[TileCache] Zoom: {self.zoom}, Bounds: {self.bounds}")
    
    def _get_zoom_level(self) -> int:
        """Get zoom level from metadata or infer from tiles."""
        # Try metadata first
        self.cursor.execute("SELECT value FROM metadata WHERE name='maxzoom'")
        row = self.cursor.fetchone()
        if row:
            return int(row[0])
        
        # Infer from tiles
        self.cursor.execute("SELECT MAX(zoom_level) FROM tiles")
        row = self.cursor.fetchone()
        if row and row[0] is not None:
            return int(row[0])
        
        return 19  # Default
    
    def _get_bounds(self) -> Optional[Dict[str, float]]:
        """Get bounds from metadata."""
        self.cursor.execute("SELECT value FROM metadata WHERE name='bounds'")
        row = self.cursor.fetchone()
        if row:
            parts = row[0].split(',')
            if len(parts) == 4:
                return {
                    'min_lon': float(parts[0]),
                    'min_lat': float(parts[1]),
                    'max_lon': float(parts[2]),
                    'max_lat': float(parts[3]),
                }
        return None
    
    def get_gsd(self, lat: float) -> float:
        """Get GSD at given latitude."""
        return get_tile_gsd(lat, self.zoom)
    
    def is_position_in_cache(self, lat: float, lon: float) -> bool:
        """
        Check if position has tile coverage.
        
        Args:
            lat, lon: Position to check
            
        Returns:
            True if tile exists for this position
        """
        tile_x, tile_y = lat_lon_to_tile(lat, lon, self.zoom)
        return self._tile_exists(tile_x, tile_y)
    
    def _tile_exists(self, tile_x: int, tile_y: int) -> bool:
        """Check if tile exists in database."""
        y_tms = (2 ** self.zoom) - 1 - tile_y
        self.cursor.execute(
            "SELECT 1 FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
            (self.zoom, tile_x, y_tms)
        )
        return self.cursor.fetchone() is not None
    
    def get_tile(self, tile_x: int, tile_y: int) -> Optional[np.ndarray]:
        """
        Get single tile image from cache.
        
        Args:
            tile_x, tile_y: Tile coordinates (Google XYZ format)
            
        Returns:
            BGR image array (256x256x3) or None if not found
        """
        cache_key = (tile_x, tile_y)
        
        # Check memory cache
        if cache_key in self._tile_cache:
            # Update LRU order
            if cache_key in self._cache_order:
                self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._tile_cache[cache_key]
        
        # Load from database
        y_tms = (2 ** self.zoom) - 1 - tile_y
        self.cursor.execute(
            "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
            (self.zoom, tile_x, y_tms)
        )
        row = self.cursor.fetchone()
        
        if row is None:
            return None
        
        # Decode image
        tile_data = row[0]
        tile_array = np.frombuffer(tile_data, dtype=np.uint8)
        tile_img = cv2.imdecode(tile_array, cv2.IMREAD_COLOR)
        
        if tile_img is None:
            return None
        
        # Add to cache
        self._tile_cache[cache_key] = tile_img
        self._cache_order.append(cache_key)
        
        # Evict oldest if over limit
        while len(self._cache_order) > self._max_cached:
            oldest = self._cache_order.pop(0)
            if oldest in self._tile_cache:
                del self._tile_cache[oldest]
        
        return tile_img
    
    def get_map_patch(self, center_lat: float, center_lon: float, 
                      patch_size_px: int = 512) -> Optional[MapPatch]:
        """
        Get stitched map patch centered at position.
        
        This is the MAIN function for VPS matching. Returns a satellite image
        patch centered at the estimated drone position.
        
        Args:
            center_lat: Center latitude (from VIO estimate)
            center_lon: Center longitude (from VIO estimate)
            patch_size_px: Output patch size in pixels (default 512x512)
            
        Returns:
            MapPatch with image and metadata, or None if no coverage
        """
        # Get center tile
        center_tile_x, center_tile_y = lat_lon_to_tile(center_lat, center_lon, self.zoom)
        
        # Get pixel position within center tile
        center_px, center_py = lat_lon_to_pixel_in_tile(
            center_lat, center_lon, center_tile_x, center_tile_y, self.zoom
        )
        
        # Calculate how many tiles we need (usually 2x2 or 3x3)
        half_patch = patch_size_px // 2
        tiles_needed_x = math.ceil((half_patch + self.TILE_SIZE) / self.TILE_SIZE)
        tiles_needed_y = math.ceil((half_patch + self.TILE_SIZE) / self.TILE_SIZE)
        
        # Create large canvas by stitching tiles
        canvas_w = (2 * tiles_needed_x + 1) * self.TILE_SIZE
        canvas_h = (2 * tiles_needed_y + 1) * self.TILE_SIZE
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        tiles_found = 0
        for dx in range(-tiles_needed_x, tiles_needed_x + 1):
            for dy in range(-tiles_needed_y, tiles_needed_y + 1):
                tx = center_tile_x + dx
                ty = center_tile_y + dy
                
                tile_img = self.get_tile(tx, ty)
                if tile_img is not None:
                    # Position in canvas
                    cx = (dx + tiles_needed_x) * self.TILE_SIZE
                    cy = (dy + tiles_needed_y) * self.TILE_SIZE
                    canvas[cy:cy + self.TILE_SIZE, cx:cx + self.TILE_SIZE] = tile_img
                    tiles_found += 1
        
        if tiles_found == 0:
            return None
        
        # Calculate center position in canvas
        canvas_center_x = tiles_needed_x * self.TILE_SIZE + center_px
        canvas_center_y = tiles_needed_y * self.TILE_SIZE + center_py
        
        # Crop patch from canvas
        x1 = canvas_center_x - half_patch
        y1 = canvas_center_y - half_patch
        x2 = x1 + patch_size_px
        y2 = y1 + patch_size_px
        
        # Clamp to valid range
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(canvas_w, x2)
        y2 = min(canvas_h, y2)
        
        patch = canvas[y1:y2, x1:x2]
        
        # Resize if needed (edge case)
        if patch.shape[0] != patch_size_px or patch.shape[1] != patch_size_px:
            patch = cv2.resize(patch, (patch_size_px, patch_size_px))
        
        # Calculate metadata
        gsd = self.get_gsd(center_lat)
        
        # Get top-left corner position
        origin_lat, origin_lon = pixel_to_lat_lon(
            center_tile_x - tiles_needed_x, 
            center_tile_y - tiles_needed_y,
            x1 - (tiles_needed_x * self.TILE_SIZE - tiles_needed_x * self.TILE_SIZE), 
            y1 - (tiles_needed_y * self.TILE_SIZE - tiles_needed_y * self.TILE_SIZE),
            self.zoom
        )
        
        return MapPatch(
            image=patch,
            center_lat=center_lat,
            center_lon=center_lon,
            meters_per_pixel=gsd,
            width_m=patch_size_px * gsd,
            height_m=patch_size_px * gsd,
            patch_origin_lat=origin_lat,
            patch_origin_lon=origin_lon
        )
    
    def pixel_offset_to_latlon(self, center_lat: float, center_lon: float,
                                dx_px: float, dy_px: float) -> Tuple[float, float]:
        """
        Convert pixel offset to lat/lon offset.
        
        Args:
            center_lat, center_lon: Reference position
            dx_px, dy_px: Pixel offset (positive = right/down)
            
        Returns:
            (new_lat, new_lon) tuple
        """
        gsd = self.get_gsd(center_lat)
        
        # Convert to meters
        dx_m = dx_px * gsd
        dy_m = dy_px * gsd
        
        # Convert to lat/lon delta
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
        
        delta_lat = -dy_m / m_per_deg_lat  # Negative because y increases downward
        delta_lon = dx_m / m_per_deg_lon
        
        return center_lat + delta_lat, center_lon + delta_lon
    
    def get_tile_count(self) -> int:
        """Get total number of tiles in cache."""
        self.cursor.execute("SELECT COUNT(*) FROM tiles")
        row = self.cursor.fetchone()
        return row[0] if row else 0
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Test function
def test_tile_cache(mbtiles_path: str, test_lat: float, test_lon: float):
    """Test tile cache functionality."""
    print(f"\n{'='*60}")
    print(f"Testing TileCache")
    print(f"{'='*60}")
    
    with TileCache(mbtiles_path) as cache:
        print(f"\nTotal tiles: {cache.get_tile_count()}")
        print(f"GSD at test location: {cache.get_gsd(test_lat):.3f} m/px")
        
        # Check coverage
        has_coverage = cache.is_position_in_cache(test_lat, test_lon)
        print(f"Has coverage at ({test_lat:.6f}, {test_lon:.6f}): {has_coverage}")
        
        if has_coverage:
            # Get map patch
            patch = cache.get_map_patch(test_lat, test_lon, patch_size_px=512)
            
            if patch:
                print(f"\nMap patch retrieved:")
                print(f"  Image shape: {patch.image.shape}")
                print(f"  GSD: {patch.meters_per_pixel:.3f} m/px")
                print(f"  Coverage: {patch.width_m:.0f}m x {patch.height_m:.0f}m")
                
                # Save preview
                output_path = Path(mbtiles_path).parent / "test_map_patch.jpg"
                cv2.imwrite(str(output_path), patch.image)
                print(f"  Saved to: {output_path}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tile_cache.py <mbtiles_path> [lat] [lon]")
        sys.exit(1)
    
    mbtiles_path = sys.argv[1]
    test_lat = float(sys.argv[2]) if len(sys.argv) > 2 else 45.315721787845
    test_lon = float(sys.argv[3]) if len(sys.argv) > 3 else -75.670671305696
    
    test_tile_cache(mbtiles_path, test_lat, test_lon)
