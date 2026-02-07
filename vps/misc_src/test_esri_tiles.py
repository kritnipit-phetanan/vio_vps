#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESRI World Imagery Tile Download Test Script

Downloads satellite tiles from ESRI for a test location.
Test coordinates: Ottawa, Canada (45.315721787845, -75.670671305696)

Usage:
    python test_esri_tiles.py

Author: VIO project
"""

import os
import math
import sqlite3
import requests
import time
from pathlib import Path
from typing import List, Tuple


# ESRI World Imagery REST endpoint (free, no API key required)
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


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


def get_tiles_in_radius(center_lat: float, center_lon: float, 
                        radius_m: float, zoom: int = 19) -> List[Tuple[int, int]]:
    """
    Get all tile coordinates within radius of center point.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        zoom: Zoom level
        
    Returns:
        List of (tile_x, tile_y) tuples
    """
    # Approximate meters per degree at this latitude
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * math.cos(math.radians(center_lat))
    
    # Bounding box
    lat_min = center_lat - radius_m / m_per_deg_lat
    lat_max = center_lat + radius_m / m_per_deg_lat
    lon_min = center_lon - radius_m / m_per_deg_lon
    lon_max = center_lon + radius_m / m_per_deg_lon
    
    # Convert to tiles
    x_min, y_max = lat_lon_to_tile(lat_min, lon_min, zoom)
    x_max, y_min = lat_lon_to_tile(lat_max, lon_max, zoom)
    
    tiles = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append((x, y))
    
    return tiles


def download_tile(x: int, y: int, zoom: int = 19, timeout: int = 10) -> bytes:
    """
    Download single tile from ESRI World Imagery.
    
    Args:
        x: Tile X coordinate
        y: Tile Y coordinate
        zoom: Zoom level
        timeout: Request timeout in seconds
        
    Returns:
        JPEG image data as bytes
    """
    url = ESRI_URL.format(z=zoom, x=x, y=y)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (VIO-VPS-Research)',
        'Accept': 'image/jpeg,image/png,image/*'
    }
    
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    
    return response.content


def init_mbtiles(output_path: str) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Initialize MBTiles SQLite database.
    
    Args:
        output_path: Path to output .mbtiles file
        
    Returns:
        (connection, cursor) tuple
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()
    
    # Create tables (MBTiles spec)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            name TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tiles (
            zoom_level INTEGER,
            tile_column INTEGER,
            tile_row INTEGER,
            tile_data BLOB,
            PRIMARY KEY (zoom_level, tile_column, tile_row)
        )
    """)
    
    # Add metadata
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('name', 'ESRI Test Tiles')")
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('format', 'jpg')")
    cursor.execute("INSERT OR REPLACE INTO metadata VALUES ('type', 'overlay')")
    
    conn.commit()
    
    return conn, cursor


def prefetch_tiles(center_lat: float, center_lon: float, radius_m: float,
                   output_path: str, zoom: int = 19) -> int:
    """
    Download all tiles in area and save to MBTiles.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        output_path: Output .mbtiles file path
        zoom: Zoom level
        
    Returns:
        Number of tiles downloaded
    """
    tiles = get_tiles_in_radius(center_lat, center_lon, radius_m, zoom)
    
    print(f"\n{'='*60}")
    print(f"ESRI World Imagery Tile Download")
    print(f"{'='*60}")
    print(f"Center: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"Radius: {radius_m}m")
    print(f"Zoom: {zoom}")
    print(f"Tiles to download: {len(tiles)}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    conn, cursor = init_mbtiles(output_path)
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for i, (x, y) in enumerate(tiles):
        # Check if already cached
        cursor.execute(
            "SELECT 1 FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
            (zoom, x, (2 ** zoom) - 1 - y)  # TMS y-flip
        )
        if cursor.fetchone():
            skipped += 1
            continue
        
        try:
            # Download tile
            tile_data = download_tile(x, y, zoom)
            
            # MBTiles uses TMS y-flip
            y_tms = (2 ** zoom) - 1 - y
            
            # Store
            cursor.execute(
                "INSERT OR REPLACE INTO tiles VALUES (?, ?, ?, ?)",
                (zoom, x, y_tms, tile_data)
            )
            
            downloaded += 1
            
            lat, lon = tile_to_lat_lon(x, y, zoom)
            print(f"[{i+1}/{len(tiles)}] Downloaded tile ({x}, {y}) -> lat={lat:.4f}, lon={lon:.4f}")
            
            # Commit every 5 tiles
            if downloaded % 5 == 0:
                conn.commit()
            
            # Rate limiting (be nice to ESRI servers)
            time.sleep(0.2)
            
        except requests.exceptions.RequestException as e:
            failed += 1
            print(f"[{i+1}/{len(tiles)}] FAILED tile ({x}, {y}): {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"Download Complete!")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")
    
    return downloaded


def test_read_tiles(mbtiles_path: str, center_lat: float, center_lon: float):
    """
    Test reading tiles from MBTiles and visualize.
    
    Args:
        mbtiles_path: Path to .mbtiles file
        center_lat: Center latitude to test
        center_lon: Center longitude to test
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV not installed, skipping visualization test")
        return
    
    if not os.path.exists(mbtiles_path):
        print(f"MBTiles not found: {mbtiles_path}")
        return
    
    conn = sqlite3.connect(mbtiles_path)
    cursor = conn.cursor()
    
    zoom = 19
    tile_x, tile_y = lat_lon_to_tile(center_lat, center_lon, zoom)
    y_tms = (2 ** zoom) - 1 - tile_y
    
    cursor.execute(
        "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
        (zoom, tile_x, y_tms)
    )
    row = cursor.fetchone()
    
    if row is None:
        print(f"Tile not found for ({center_lat}, {center_lon})")
        conn.close()
        return
    
    # Decode image
    tile_data = row[0]
    tile_array = np.frombuffer(tile_data, dtype=np.uint8)
    tile_img = cv2.imdecode(tile_array, cv2.IMREAD_COLOR)
    
    # Save for inspection
    output_dir = Path(mbtiles_path).parent
    output_file = output_dir / "test_tile_preview.jpg"
    cv2.imwrite(str(output_file), tile_img)
    
    print(f"\nTile preview saved to: {output_file}")
    print(f"Tile shape: {tile_img.shape}")
    print(f"Tile coordinates: ({tile_x}, {tile_y})")
    
    # Calculate GSD
    meters_per_pixel = 156543.03 * math.cos(math.radians(center_lat)) / (2 ** zoom)
    print(f"GSD at this latitude: {meters_per_pixel:.3f} m/px")
    
    conn.close()

if __name__ == "__main__":
    # Test coordinates: Ottawa, Canada
    TEST_LAT = 45.315721787845
    TEST_LON = -75.670671305696
    TEST_RADIUS = 200  # meters (small for testing)
    
    # Output path
    output_dir = Path(__file__).parent
    output_path = output_dir / "test_tiles_ottawa.mbtiles"
    
    # Download tiles
    prefetch_tiles(
        center_lat=TEST_LAT,
        center_lon=TEST_LON,
        radius_m=TEST_RADIUS,
        output_path=str(output_path),
        zoom=19
    )
    
    # Test reading
    test_read_tiles(str(output_path), TEST_LAT, TEST_LON)
