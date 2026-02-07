#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-flight Coverage Check Tool

Check if positions are covered by an MBTiles file.
Requires only sqlite3 and basic math (no numpy/opencv).

Usage:
    python3 vps/check_coverage.py mission.mbtiles 45.309,-75.667
    python3 vps/check_coverage.py mission.mbtiles --flight-path path.csv
"""

import os
import sys
import math
import sqlite3
import argparse
from typing import Tuple, List, Optional


def lat_lon_to_tile(lat: float, lon: float, zoom: int = 19) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters."""
    R = 6371000  # Earth radius in meters
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_r)*math.cos(lat2_r)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


class CoverageChecker:
    """Check MBTiles coverage for positions."""
    
    def __init__(self, mbtiles_path: str):
        if not os.path.exists(mbtiles_path):
            raise FileNotFoundError(f"MBTiles not found: {mbtiles_path}")
        
        self.conn = sqlite3.connect(mbtiles_path)
        self.cursor = self.conn.cursor()
        
        # Get zoom level
        self.cursor.execute("SELECT zoom_level FROM tiles LIMIT 1")
        row = self.cursor.fetchone()
        self.zoom = row[0] if row else 19
        
        # Get tile count
        self.cursor.execute("SELECT COUNT(*) FROM tiles")
        self.tile_count = self.cursor.fetchone()[0]
        
        # Get bounds
        self._compute_bounds()
    
    def _compute_bounds(self):
        """Compute geographic bounds from tiles."""
        self.cursor.execute("""
            SELECT MIN(tile_column), MAX(tile_column), 
                   MIN(tile_row), MAX(tile_row) 
            FROM tiles WHERE zoom_level = ?
        """, (self.zoom,))
        row = self.cursor.fetchone()
        
        if row and row[0] is not None:
            self.x_min, self.x_max = row[0], row[1]
            # TMS uses flipped Y
            y_tms_min, y_tms_max = row[2], row[3]
            n = 2 ** self.zoom
            self.y_min = n - 1 - y_tms_max
            self.y_max = n - 1 - y_tms_min
            
            # Convert corners to lat/lon
            self.lat_min, self.lon_min = self._tile_to_latlon(self.x_min, self.y_max + 1)
            self.lat_max, self.lon_max = self._tile_to_latlon(self.x_max + 1, self.y_min)
        else:
            self.lat_min = self.lat_max = self.lon_min = self.lon_max = 0
    
    def _tile_to_latlon(self, x: int, y: int) -> Tuple[float, float]:
        """Convert tile coordinates to lat/lon."""
        n = 2.0 ** self.zoom
        lon = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = math.degrees(lat_rad)
        return lat, lon
    
    def is_position_covered(self, lat: float, lon: float) -> bool:
        """Check if a position has tile coverage."""
        x, y = lat_lon_to_tile(lat, lon, self.zoom)
        
        # Convert to TMS
        n = 2 ** self.zoom
        y_tms = n - 1 - y
        
        self.cursor.execute("""
            SELECT 1 FROM tiles 
            WHERE zoom_level = ? AND tile_column = ? AND tile_row = ?
            LIMIT 1
        """, (self.zoom, x, y_tms))
        
        return self.cursor.fetchone() is not None
    
    def check_coverage_radius(self, center_lat: float, center_lon: float, 
                              check_radius_m: float = 100.0) -> dict:
        """Check coverage around a center point."""
        # Check center
        center_covered = self.is_position_covered(center_lat, center_lon)
        
        # Check 8 points around perimeter
        perimeter_covered = 0
        total_points = 8
        
        for i in range(total_points):
            angle = 2 * math.pi * i / total_points
            # Approximate offset in degrees
            dlat = (check_radius_m / 111000) * math.cos(angle)
            dlon = (check_radius_m / (111000 * math.cos(math.radians(center_lat)))) * math.sin(angle)
            
            if self.is_position_covered(center_lat + dlat, center_lon + dlon):
                perimeter_covered += 1
        
        return {
            'center_covered': center_covered,
            'perimeter_coverage': perimeter_covered / total_points,
            'fully_covered': center_covered and perimeter_covered == total_points
        }
    
    def get_info(self) -> dict:
        """Get MBTiles information."""
        return {
            'zoom': self.zoom,
            'tile_count': self.tile_count,
            'bounds': {
                'lat_min': self.lat_min,
                'lat_max': self.lat_max,
                'lon_min': self.lon_min,
                'lon_max': self.lon_max,
            }
        }
    
    def close(self):
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='Check MBTiles coverage for positions')
    parser.add_argument('mbtiles', type=str, help='Path to MBTiles file')
    parser.add_argument('position', type=str, nargs='?', default=None,
                        help='Position to check: LAT,LON (e.g., 45.315,-75.670)')
    parser.add_argument('--positions', '-p', type=str, nargs='+',
                        help='Multiple positions to check')
    parser.add_argument('--flight-path', '-f', type=str,
                        help='CSV file with flight path (lat,lon columns)')
    parser.add_argument('--radius', '-r', type=float, default=100.0,
                        help='Check radius around each point (meters)')
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = CoverageChecker(args.mbtiles)
    info = checker.get_info()
    
    print("="*60)
    print("MBTiles Coverage Check")
    print("="*60)
    print(f"File: {args.mbtiles}")
    print(f"Zoom: {info['zoom']}")
    print(f"Tiles: {info['tile_count']}")
    print(f"Bounds:")
    print(f"  Lat: {info['bounds']['lat_min']:.6f} to {info['bounds']['lat_max']:.6f}")
    print(f"  Lon: {info['bounds']['lon_min']:.6f} to {info['bounds']['lon_max']:.6f}")
    print()
    
    # Parse positions to check
    positions = []
    
    if args.position:
        parts = args.position.split(',')
        if len(parts) == 2:
            positions.append((float(parts[0]), float(parts[1]), "CLI argument"))
    
    if args.positions:
        for i, pos in enumerate(args.positions):
            parts = pos.split(',')
            if len(parts) == 2:
                positions.append((float(parts[0]), float(parts[1]), f"Position {i+1}"))
    
    if args.flight_path and os.path.exists(args.flight_path):
        try:
            with open(args.flight_path, 'r') as f:
                lines = f.readlines()
            
            # Simple CSV parsing
            for i, line in enumerate(lines[1:]):  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        lat, lon = float(parts[0]), float(parts[1])
                        positions.append((lat, lon, f"Path point {i+1}"))
                    except ValueError:
                        continue
            
            print(f"Loaded {len(positions)} points from flight path")
        except Exception as e:
            print(f"Warning: Failed to load flight path: {e}")
    
    # Check coverage
    if not positions:
        print("No positions specified. Use --position or --positions")
        checker.close()
        return 1
    
    print("-"*60)
    print("Coverage Results:")
    print("-"*60)
    
    all_covered = True
    for lat, lon, name in positions:
        result = checker.check_coverage_radius(lat, lon, args.radius)
        
        status = "✅" if result['fully_covered'] else ("⚠️" if result['center_covered'] else "❌")
        coverage_pct = result['perimeter_coverage'] * 100
        
        print(f"{status} {name}")
        print(f"    ({lat:.6f}, {lon:.6f})")
        print(f"    Center: {'covered' if result['center_covered'] else 'NOT COVERED'}")
        print(f"    {args.radius}m radius: {coverage_pct:.0f}% covered")
        
        if not result['fully_covered']:
            all_covered = False
    
    print()
    print("="*60)
    if all_covered:
        print("✅ All positions fully covered - SAFE TO FLY")
    else:
        print("⚠️  Some positions have incomplete coverage")
        print("   Consider downloading more tiles before flight!")
    print("="*60)
    
    checker.close()
    return 0 if all_covered else 1


if __name__ == "__main__":
    sys.exit(main())
