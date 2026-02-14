#!/usr/bin/env python3
"""
Soft Iron Calibration Script for Magnetometer

This script analyzes magnetometer data to compute hard iron and soft iron
calibration parameters using ellipsoid fitting.

Usage:
    python3 calibrate_soft_iron.py <mag_csv_path>

The ideal calibrated magnetometer data should form a sphere centered at origin.
Deviations indicate:
- Hard iron: Sphere center offset from origin
- Soft iron: Ellipsoid instead of sphere (different scale/rotation per axis)
"""
import csv
import math
import sys
from typing import List, Tuple

def load_mag_data(path: str) -> Tuple[List[float], List[float], List[float]]:
    """Load magnetometer x, y, z data from CSV."""
    x_vals, y_vals, z_vals = [], [], []
    
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_vals.append(float(row['x']))
            y_vals.append(float(row['y']))
            z_vals.append(float(row['z']))
    
    return x_vals, y_vals, z_vals

def compute_hard_iron(x: List[float], y: List[float], z: List[float]) -> Tuple[float, float, float]:
    """Compute hard iron offset as the center of the data."""
    # Simple method: use min/max center
    hi_x = (max(x) + min(x)) / 2
    hi_y = (max(y) + min(y)) / 2
    hi_z = (max(z) + min(z)) / 2
    return hi_x, hi_y, hi_z

def compute_soft_iron_diagonal(x: List[float], y: List[float], z: List[float],
                                hi_x: float, hi_y: float, hi_z: float) -> Tuple[float, float, float]:
    """
    Compute diagonal soft iron matrix (scale factors only).
    
    This normalizes each axis so that the data spans the same range.
    """
    # Center the data
    x_c = [xi - hi_x for xi in x]
    y_c = [yi - hi_y for yi in y]
    z_c = [zi - hi_z for zi in z]
    
    # Compute range for each axis
    range_x = max(x_c) - min(x_c)
    range_y = max(y_c) - min(y_c)
    range_z = max(z_c) - min(z_c)
    
    # Average range (target)
    avg_range = (range_x + range_y + range_z) / 3
    
    # Scale factors to normalize to average range
    # soft_iron[i] = avg_range / range[i]
    si_x = avg_range / range_x if range_x > 0 else 1.0
    si_y = avg_range / range_y if range_y > 0 else 1.0
    si_z = avg_range / range_z if range_z > 0 else 1.0
    
    return si_x, si_y, si_z

def apply_calibration(x: List[float], y: List[float], z: List[float],
                      hi_x: float, hi_y: float, hi_z: float,
                      si_x: float, si_y: float, si_z: float) -> Tuple[List[float], List[float], List[float]]:
    """Apply hard iron and soft iron calibration."""
    x_cal = [(xi - hi_x) * si_x for xi in x]
    y_cal = [(yi - hi_y) * si_y for yi in y]
    z_cal = [(zi - hi_z) * si_z for zi in z]
    return x_cal, y_cal, z_cal

def compute_heading_error(x_cal: List[float], y_cal: List[float], 
                          ppk_yaw_ned: float, declination: float = 0.0) -> float:
    """Compute mean heading error compared to PPK ground truth."""
    # Compute heading from first 100 samples
    headings = []
    for i in range(min(100, len(x_cal))):
        h = math.atan2(-y_cal[i], x_cal[i]) + declination
        headings.append(math.degrees(h))
    
    mag_heading_mean = sum(headings) / len(headings)
    error = mag_heading_mean - ppk_yaw_ned
    return error

def main():
    # Path to magnetometer data
    if len(sys.argv) > 1:
        mag_path = sys.argv[1]
    else:
        mag_path = '/Users/france/Downloads/vio_dataset/bell412_dataset3/extracted_data_new/imu_data/imu__mag/vector3.csv'
    
    print("=" * 60)
    print("SOFT IRON CALIBRATION ANALYSIS")
    print("=" * 60)
    
    # Load data
    x, y, z = load_mag_data(mag_path)
    print(f"\nLoaded {len(x)} magnetometer samples")
    
    # Current config values
    current_hi = [0.243467, -0.194837, 0.341668]
    current_si = [[0.702036, 0.0, 0.0],
                  [0.0, 0.715855, 0.0],
                  [0.0, 0.0, 5.597808]]
    
    print("\n" + "-" * 60)
    print("CURRENT CALIBRATION (from config)")
    print("-" * 60)
    print(f"Hard Iron: {current_hi}")
    print(f"Soft Iron diagonal: [{current_si[0][0]:.4f}, {current_si[1][1]:.4f}, {current_si[2][2]:.4f}]")
    
    # Compute new hard iron
    hi_x, hi_y, hi_z = compute_hard_iron(x, y, z)
    
    print("\n" + "-" * 60)
    print("NEW CALIBRATION (computed from data)")
    print("-" * 60)
    print(f"\n=== Hard Iron (center offset) ===")
    print(f"New:     [{hi_x:.6f}, {hi_y:.6f}, {hi_z:.6f}]")
    print(f"Current: [{current_hi[0]:.6f}, {current_hi[1]:.6f}, {current_hi[2]:.6f}]")
    print(f"Diff:    [{hi_x - current_hi[0]:.6f}, {hi_y - current_hi[1]:.6f}, {hi_z - current_hi[2]:.6f}]")
    
    # Compute soft iron (diagonal)
    si_x, si_y, si_z = compute_soft_iron_diagonal(x, y, z, hi_x, hi_y, hi_z)
    
    print(f"\n=== Soft Iron (scale factors - diagonal) ===")
    print(f"New:     [{si_x:.6f}, {si_y:.6f}, {si_z:.6f}]")
    print(f"Current: [{current_si[0][0]:.6f}, {current_si[1][1]:.6f}, {current_si[2][2]:.6f}]")
    
    # Data range analysis
    print(f"\n=== Data Range Analysis ===")
    print(f"X: min={min(x):.4f}, max={max(x):.4f}, range={max(x)-min(x):.4f}")
    print(f"Y: min={min(y):.4f}, max={max(y):.4f}, range={max(y)-min(y):.4f}")
    print(f"Z: min={min(z):.4f}, max={max(z):.4f}, range={max(z)-min(z):.4f}")
    
    # Center the data with new hard iron
    x_c = [xi - hi_x for xi in x]
    y_c = [yi - hi_y for yi in y]
    z_c = [zi - hi_z for zi in z]
    
    print(f"\n=== Data Range After Hard Iron Correction ===")
    print(f"X: min={min(x_c):.4f}, max={max(x_c):.4f}")
    print(f"Y: min={min(y_c):.4f}, max={max(y_c):.4f}")
    print(f"Z: min={min(z_c):.4f}, max={max(z_c):.4f}")
    
    # Apply new calibration
    x_cal, y_cal, z_cal = apply_calibration(x, y, z, hi_x, hi_y, hi_z, si_x, si_y, si_z)
    
    # Compute magnitude statistics
    mags_raw = [math.sqrt(xi**2 + yi**2 + zi**2) for xi, yi, zi in zip(x, y, z)]
    mags_cal = [math.sqrt(xi**2 + yi**2 + zi**2) for xi, yi, zi in zip(x_cal, y_cal, z_cal)]
    
    print(f"\n=== Magnitude After Calibration ===")
    print(f"Raw: mean={sum(mags_raw)/len(mags_raw):.4f}, std={stdev(mags_raw):.4f}")
    print(f"Cal: mean={sum(mags_cal)/len(mags_cal):.4f}, std={stdev(mags_cal):.4f}")
    
    # Heading analysis
    ppk_yaw_ned = -41.5  # degrees (from PPK at t=0)
    
    print(f"\n=== Heading Analysis (first 100 samples) ===")
    print(f"PPK yaw (NED): {ppk_yaw_ned}°")
    
    # With new calibration
    error_new = compute_heading_error(x_cal, y_cal, ppk_yaw_ned, declination=0.0)
    print(f"New calibration heading error (no declination): {error_new:.1f}°")
    print(f"=> Required declination: {-error_new:.1f}° ({math.radians(-error_new):.4f} rad)")
    
    # YAML output
    print("\n" + "=" * 60)
    print("YAML CONFIG (copy to config file)")
    print("=" * 60)
    print(f"""
magnetometer:
  # Hard-iron offset (NEW - computed from data range center)
  hard_iron_offset: [{hi_x:.6f}, {hi_y:.6f}, {hi_z:.6f}]
  
  # Soft-iron matrix (NEW - diagonal scale factors)
  soft_iron_matrix:
    - [{si_x:.6f}, 0.0, 0.0]
    - [0.0, {si_y:.6f}, 0.0]
    - [0.0, 0.0, {si_z:.6f}]
  
  # Declination (NEW - computed from PPK ground truth)
  declination: {math.radians(-error_new):.4f}  # {-error_new:.1f}°
""")

def stdev(data):
    """Compute standard deviation."""
    n = len(data)
    mean = sum(data) / n
    return math.sqrt(sum((x - mean) ** 2 for x in data) / n)

if __name__ == "__main__":
    main()
