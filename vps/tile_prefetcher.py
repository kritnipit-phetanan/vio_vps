#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tile Prefetcher for Offline VPS

Downloads satellite tiles from ESRI World Imagery and stores in MBTiles format.
Run this BEFORE flight to cache tiles for the mission area.

Usage:
    python -m vps.tile_prefetcher --center 45.315,--75.670 --radius 500 --output mission.mbtiles

Author: VIO project
"""

import os
import math
import sqlite3
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Optional: requests is only needed for downloading, not for reading cached tiles
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None


# Tile source configurations
TILE_SOURCES = {
    'esri': {
        'url': "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        'format': 'jpg',
        'max_zoom': 19,
    },
    'bing': {
        # Bing uses quadkey format, would need different implementation
        'url': None,
        'format': 'jpg',
        'max_zoom': 19,
    }
}


@dataclass
class TileBounds:
    """Bounds of tile coverage area."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    zoom: int
    tile_count: int


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """
    Convert lat/lon to tile coordinates (Web Mercator / Slippy Map).
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        zoom: Zoom level (0-19)
        
    Returns:
        (tile_x, tile_y) tuple
    """
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lat_lon(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """
    Convert tile coordinates to lat/lon (top-left corner of tile).
    
    Args:
        x: Tile X coordinate
        y: Tile Y coordinate
        zoom: Zoom level
        
    Returns:
        (lat, lon) tuple
    """
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def tile_to_lat_lon_center(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """Get center of tile in lat/lon."""
    lat_nw, lon_nw = tile_to_lat_lon(x, y, zoom)
    lat_se, lon_se = tile_to_lat_lon(x + 1, y + 1, zoom)
    return (lat_nw + lat_se) / 2, (lon_nw + lon_se) / 2


def get_tile_gsd(lat: float, zoom: int) -> float:
    """
    Get Ground Sample Distance (meters per pixel) for tile at given latitude.
    
    Tile size is 256x256 pixels. Earth circumference at equator is ~40,075 km.
    GSD varies with latitude due to Mercator projection.
    """
    earth_circumference = 40075016.686  # meters at equator
    tile_size = 256
    meters_per_pixel = earth_circumference * math.cos(math.radians(lat)) / (tile_size * (2 ** zoom))
    return meters_per_pixel


class TilePrefetcher:
    """
    Downloads and caches satellite tiles to MBTiles format.
    
    MBTiles is a SQLite-based format for storing map tiles.
    Uses TMS (y-flipped) coordinate system internally.
    """
    
    def __init__(self, output_path: str, tile_source: str = 'esri', zoom: int = 19):
        """
        Initialize tile prefetcher.
        
        Args:
            output_path: Path to output .mbtiles file
            tile_source: Tile source ('esri', 'bing')
            zoom: Zoom level (default 19 for ~0.3m GSD)
        """
        self.output_path = output_path
        self.zoom = zoom
        
        if tile_source not in TILE_SOURCES:
            raise ValueError(f"Unknown tile source: {tile_source}. Available: {list(TILE_SOURCES.keys())}")
        
        self.source_config = TILE_SOURCES[tile_source]
        if self.source_config['url'] is None:
            raise ValueError(f"Tile source '{tile_source}' not yet implemented")
        
        self.tile_url = self.source_config['url']
        self.tile_format = self.source_config['format']
        
        # Initialize database
        self._init_database()
        
        # Statistics
        self.stats = {
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
        }
    
    def _init_database(self):
        """Initialize MBTiles SQLite database."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.output_path)
        self.cursor = self.conn.cursor()
        
        # Create tables (MBTiles 1.3 spec)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tiles (
                zoom_level INTEGER,
                tile_column INTEGER,
                tile_row INTEGER,
                tile_data BLOB,
                PRIMARY KEY (zoom_level, tile_column, tile_row)
            )
        """)
        
        # Create index for faster queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS tiles_idx ON tiles (zoom_level, tile_column, tile_row)
        """)
        
        self.conn.commit()
    
    def _set_metadata(self, name: str, value: str):
        """Set metadata value."""
        self.cursor.execute(
            "INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)",
            (name, value)
        )
    
    def get_tiles_in_radius(self, center_lat: float, center_lon: float, 
                            radius_m: float) -> List[Tuple[int, int]]:
        """
        Get all tile coordinates within radius of center point.
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_m: Radius in meters
            
        Returns:
            List of (tile_x, tile_y) tuples
        """
        # Approximate meters per degree at this latitude
        m_per_deg_lat = 111320
        m_per_deg_lon = 111320 * math.cos(math.radians(center_lat))
        
        # Bounding box with some margin
        margin = 1.2  # 20% margin
        lat_min = center_lat - (radius_m * margin) / m_per_deg_lat
        lat_max = center_lat + (radius_m * margin) / m_per_deg_lat
        lon_min = center_lon - (radius_m * margin) / m_per_deg_lon
        lon_max = center_lon + (radius_m * margin) / m_per_deg_lon
        
        # Convert to tiles (note: y increases southward in tile coords)
        x_min, y_max = lat_lon_to_tile(lat_min, lon_min, self.zoom)
        x_max, y_min = lat_lon_to_tile(lat_max, lon_max, self.zoom)
        
        tiles = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tiles.append((x, y))
        
        return tiles
    
    def get_tiles_in_bbox(self, min_lat: float, min_lon: float,
                          max_lat: float, max_lon: float) -> List[Tuple[int, int]]:
        """
        Get all tiles covering a bounding box.
        
        Args:
            min_lat, min_lon: Southwest corner
            max_lat, max_lon: Northeast corner
            
        Returns:
            List of (tile_x, tile_y) tuples
        """
        x_min, y_max = lat_lon_to_tile(min_lat, min_lon, self.zoom)
        x_max, y_min = lat_lon_to_tile(max_lat, max_lon, self.zoom)
        
        tiles = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tiles.append((x, y))
        
        return tiles
    
    def _tile_exists(self, x: int, y: int) -> bool:
        """Check if tile already exists in database."""
        y_tms = (2 ** self.zoom) - 1 - y
        self.cursor.execute(
            "SELECT 1 FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
            (self.zoom, x, y_tms)
        )
        return self.cursor.fetchone() is not None
    
    def download_tile(self, x: int, y: int, timeout: int = 10) -> Optional[bytes]:
        """
        Download single tile from server.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            timeout: Request timeout in seconds
            
        Returns:
            JPEG/PNG image data as bytes, or None on failure
        """
        if not HAS_REQUESTS:
            raise ImportError("'requests' module required for downloading. Install with: pip install requests")
        
        url = self.tile_url.format(z=self.zoom, x=x, y=y)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (VIO-VPS-Research)',
            'Accept': 'image/jpeg,image/png,image/*'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"[TilePrefetcher] Download failed for ({x}, {y}): {e}")
            return None
    
    def _store_tile(self, x: int, y: int, tile_data: bytes):
        """Store tile in database (TMS y-flip)."""
        y_tms = (2 ** self.zoom) - 1 - y
        self.cursor.execute(
            "INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)",
            (self.zoom, x, y_tms, tile_data)
        )
    
    def prefetch_area(self, center_lat: float, center_lon: float, 
                      radius_m: float, rate_limit: float = 0.2,
                      skip_existing: bool = True) -> TileBounds:
        """
        Download all tiles in area and save to MBTiles.
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_m: Radius in meters
            rate_limit: Seconds between requests (be nice to servers)
            skip_existing: Skip already cached tiles
            
        Returns:
            TileBounds with coverage information
        """
        tiles = self.get_tiles_in_radius(center_lat, center_lon, radius_m)
        
        # Get GSD at this location
        gsd = get_tile_gsd(center_lat, self.zoom)
        
        print(f"\n{'='*60}")
        print(f"Tile Prefetch: ESRI World Imagery")
        print(f"{'='*60}")
        print(f"Center: ({center_lat:.6f}, {center_lon:.6f})")
        print(f"Radius: {radius_m:.0f}m")
        print(f"Zoom: {self.zoom}")
        print(f"GSD: {gsd:.3f} m/px")
        print(f"Tiles to download: {len(tiles)}")
        print(f"Output: {self.output_path}")
        print(f"{'='*60}\n")
        
        # Reset stats
        self.stats = {'downloaded': 0, 'skipped': 0, 'failed': 0}
        
        for i, (x, y) in enumerate(tiles):
            # Check if already cached
            if skip_existing and self._tile_exists(x, y):
                self.stats['skipped'] += 1
                continue
            
            # Download
            tile_data = self.download_tile(x, y)
            
            if tile_data is None:
                self.stats['failed'] += 1
                continue
            
            # Store
            self._store_tile(x, y, tile_data)
            self.stats['downloaded'] += 1
            
            lat, lon = tile_to_lat_lon_center(x, y, self.zoom)
            print(f"[{i+1}/{len(tiles)}] Downloaded ({x}, {y}) -> ({lat:.4f}, {lon:.4f})")
            
            # Commit every 10 tiles
            if self.stats['downloaded'] % 10 == 0:
                self.conn.commit()
            
            # Rate limit
            if rate_limit > 0:
                time.sleep(rate_limit)
        
        # Final commit
        self.conn.commit()
        
        # Update metadata
        self._update_metadata(center_lat, center_lon, radius_m, tiles)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Download Complete!")
        print(f"  Downloaded: {self.stats['downloaded']}")
        print(f"  Skipped (cached): {self.stats['skipped']}")
        print(f"  Failed: {self.stats['failed']}")
        print(f"  Output: {self.output_path}")
        print(f"{'='*60}\n")
        
        # Return bounds info
        if tiles:
            x_coords = [t[0] for t in tiles]
            y_coords = [t[1] for t in tiles]
            lat_nw, lon_nw = tile_to_lat_lon(min(x_coords), min(y_coords), self.zoom)
            lat_se, lon_se = tile_to_lat_lon(max(x_coords) + 1, max(y_coords) + 1, self.zoom)
            
            return TileBounds(
                min_lat=lat_se,
                max_lat=lat_nw,
                min_lon=lon_nw,
                max_lon=lon_se,
                zoom=self.zoom,
                tile_count=len(tiles)
            )
        
        return TileBounds(0, 0, 0, 0, self.zoom, 0)
    
    def _update_metadata(self, center_lat: float, center_lon: float, 
                         radius_m: float, tiles: List[Tuple[int, int]]):
        """Update MBTiles metadata."""
        self._set_metadata('name', 'VPS Satellite Tiles')
        self._set_metadata('format', self.tile_format)
        self._set_metadata('type', 'overlay')
        self._set_metadata('minzoom', str(self.zoom))
        self._set_metadata('maxzoom', str(self.zoom))
        
        # Bounds
        if tiles:
            x_coords = [t[0] for t in tiles]
            y_coords = [t[1] for t in tiles]
            lat_nw, lon_nw = tile_to_lat_lon(min(x_coords), min(y_coords), self.zoom)
            lat_se, lon_se = tile_to_lat_lon(max(x_coords) + 1, max(y_coords) + 1, self.zoom)
            self._set_metadata('bounds', f"{lon_nw},{lat_se},{lon_se},{lat_nw}")
            self._set_metadata('center', f"{center_lon},{center_lat},{self.zoom}")
        
        # Custom metadata
        self._set_metadata('vps_center_lat', str(center_lat))
        self._set_metadata('vps_center_lon', str(center_lon))
        self._set_metadata('vps_radius_m', str(radius_m))
        self._set_metadata('vps_gsd', str(get_tile_gsd(center_lat, self.zoom)))
        
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Download satellite tiles for offline VPS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 500m radius around Ottawa
  python -m vps.tile_prefetcher --center 45.315,-75.670 --radius 500 --output mission.mbtiles
  
  # Download with bounding box
  python -m vps.tile_prefetcher --bbox 45.31,--75.68,45.32,-75.66 --output mission.mbtiles
        """
    )
    
    parser.add_argument('--center', type=str, 
                        help='Center point as lat,lon (e.g., 45.315,-75.670)')
    parser.add_argument('--radius', type=float, default=500,
                        help='Radius in meters (default: 500)')
    parser.add_argument('--bbox', type=str,
                        help='Bounding box as min_lat,min_lon,max_lat,max_lon')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output .mbtiles file path')
    parser.add_argument('--zoom', type=int, default=19,
                        help='Zoom level (default: 19, ~0.3m GSD)')
    parser.add_argument('--source', type=str, default='esri',
                        choices=['esri'],
                        help='Tile source (default: esri)')
    parser.add_argument('--rate-limit', type=float, default=0.2,
                        help='Seconds between requests (default: 0.2)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.center and not args.bbox:
        parser.error("Either --center or --bbox is required")
    
    with TilePrefetcher(args.output, tile_source=args.source, zoom=args.zoom) as prefetcher:
        if args.center:
            lat, lon = map(float, args.center.split(','))
            prefetcher.prefetch_area(lat, lon, args.radius, rate_limit=args.rate_limit)
        elif args.bbox:
            coords = list(map(float, args.bbox.split(',')))
            if len(coords) != 4:
                parser.error("--bbox requires 4 values: min_lat,min_lon,max_lat,max_lon")
            # For bbox, use center point
            center_lat = (coords[0] + coords[2]) / 2
            center_lon = (coords[1] + coords[3]) / 2
            # Estimate radius from bbox
            from math import sqrt
            dlat = (coords[2] - coords[0]) * 111320 / 2
            dlon = (coords[3] - coords[1]) * 111320 * math.cos(math.radians(center_lat)) / 2
            radius = sqrt(dlat**2 + dlon**2)
            prefetcher.prefetch_area(center_lat, center_lon, radius, rate_limit=args.rate_limit)


if __name__ == "__main__":
    main()
