#!/usr/bin/env python3
"""
Magnetometer Calibration Tool for Bell 412
Uses ellipsoid fitting to estimate hard-iron and soft-iron distortion
"""
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def load_magnetometer_data(csv_path):
    """Load raw magnetometer data"""
    df = pd.read_csv(csv_path)
    
    # Assuming columns: timestamp, mx, my, mz
    mag_x = df.iloc[:, 1].values
    mag_y = df.iloc[:, 2].values
    mag_z = df.iloc[:, 3].values
    
    return np.column_stack([mag_x, mag_y, mag_z])

def fit_ellipsoid(data):
    """
    Fit ellipsoid to magnetometer data using least squares
    Returns: center (hard-iron offset), radii (soft-iron scaling)
    
    Ellipsoid equation: (x-cx)²/rx² + (y-cy)²/ry² + (z-cz)²/rz² = 1
    """
    # Initial guess: center at mean, radius = std
    x0 = [
        data[:, 0].mean(), data[:, 1].mean(), data[:, 2].mean(),  # center
        data[:, 0].std(), data[:, 1].std(), data[:, 2].std()      # radii
    ]
    
    def error_func(params):
        cx, cy, cz, rx, ry, rz = params
        
        # Normalize points to unit sphere
        x_norm = (data[:, 0] - cx) / rx
        y_norm = (data[:, 1] - cy) / ry
        z_norm = (data[:, 2] - cz) / rz
        
        # Distance from unit sphere
        dist = np.sqrt(x_norm**2 + y_norm**2 + z_norm**2) - 1.0
        
        return np.sum(dist**2)
    
    # Optimize
    result = optimize.minimize(error_func, x0, method='Nelder-Mead')
    
    cx, cy, cz, rx, ry, rz = result.x
    
    # Hard-iron offset
    hard_iron = np.array([cx, cy, cz])
    
    # Soft-iron matrix (diagonal scaling)
    # Normalize to average radius
    r_avg = (rx + ry + rz) / 3.0
    soft_iron = np.diag([r_avg/rx, r_avg/ry, r_avg/rz])
    
    return hard_iron, soft_iron, r_avg

def apply_calibration(data, hard_iron, soft_iron):
    """Apply calibration to magnetometer data"""
    # Remove hard-iron offset
    data_centered = data - hard_iron
    
    # Apply soft-iron correction
    data_calibrated = data_centered @ soft_iron.T
    
    return data_calibrated

def plot_calibration(data_raw, data_calibrated, hard_iron):
    """Plot before/after calibration"""
    fig = plt.figure(figsize=(15, 5))
    
    # Raw data
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(data_raw[:, 0], data_raw[:, 1], data_raw[:, 2], 
                c='r', marker='.', alpha=0.3)
    ax1.scatter([hard_iron[0]], [hard_iron[1]], [hard_iron[2]], 
                c='k', marker='x', s=100, label='Hard-iron')
    ax1.set_xlabel('X (µT)')
    ax1.set_ylabel('Y (µT)')
    ax1.set_zlabel('Z (µT)')
    ax1.set_title('Raw Magnetometer Data')
    ax1.legend()
    
    # Calibrated data
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(data_calibrated[:, 0], data_calibrated[:, 1], data_calibrated[:, 2], 
                c='g', marker='.', alpha=0.3)
    ax2.set_xlabel('X (µT)')
    ax2.set_ylabel('Y (µT)')
    ax2.set_zlabel('Z (µT)')
    ax2.set_title('Calibrated Magnetometer Data')
    
    # Field strength distribution
    ax3 = fig.add_subplot(133)
    
    field_raw = np.linalg.norm(data_raw, axis=1)
    field_cal = np.linalg.norm(data_calibrated, axis=1)
    
    ax3.hist(field_raw, bins=50, alpha=0.5, label='Raw', color='r')
    ax3.hist(field_cal, bins=50, alpha=0.5, label='Calibrated', color='g')
    ax3.axvline(50.0, color='k', linestyle='--', label='Expected (50 µT)')
    ax3.set_xlabel('Field Strength (µT)')
    ax3.set_ylabel('Count')
    ax3.set_title('Field Strength Distribution')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('magnetometer_calibration.png', dpi=150)
    print("Calibration plot saved to: magnetometer_calibration.png")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python mag_calibration.py <magnetometer_csv_path>")
        print("Example: python mag_calibration.py /mnt/External_Storage/CV_team/vio_dataset/bell412_dataset3/extracted_data/mag_data/vector3.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    print("=" * 80)
    print("MAGNETOMETER CALIBRATION - Bell 412")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading magnetometer data...")
    data_raw = load_magnetometer_data(csv_path)
    print(f"Loaded {len(data_raw)} samples")
    print()
    
    # Analyze raw data
    print("=== RAW DATA STATISTICS ===")
    field_raw = np.linalg.norm(data_raw, axis=1)
    print(f"  Mean field strength: {field_raw.mean():.2f} µT")
    print(f"  Std dev:             {field_raw.std():.2f} µT")
    print(f"  Min/Max:             {field_raw.min():.2f} / {field_raw.max():.2f} µT")
    print(f"  Expected (Earth):    ~50.0 µT (Newfoundland)")
    print()
    print(f"  Mean X: {data_raw[:, 0].mean():.2f} µT")
    print(f"  Mean Y: {data_raw[:, 1].mean():.2f} µT")
    print(f"  Mean Z: {data_raw[:, 2].mean():.2f} µT")
    print()
    
    # Fit ellipsoid
    print("Fitting ellipsoid to magnetometer data...")
    hard_iron, soft_iron, field_expected = fit_ellipsoid(data_raw)
    print()
    
    # Apply calibration
    data_calibrated = apply_calibration(data_raw, hard_iron, soft_iron)
    
    # Analyze calibrated data
    print("=== CALIBRATED DATA STATISTICS ===")
    field_cal = np.linalg.norm(data_calibrated, axis=1)
    print(f"  Mean field strength: {field_cal.mean():.2f} µT")
    print(f"  Std dev:             {field_cal.std():.2f} µT")
    print(f"  Min/Max:             {field_cal.min():.2f} / {field_cal.max():.2f} µT")
    print()
    
    # Improvement
    improvement = (field_raw.std() - field_cal.std()) / field_raw.std() * 100
    print(f"  Improvement: {improvement:.1f}% reduction in field variance")
    print()
    
    # Calibration parameters
    print("=== CALIBRATION PARAMETERS ===")
    print()
    print("Hard-Iron Offset (µT):")
    print(f"  X: {hard_iron[0]:.6f}")
    print(f"  Y: {hard_iron[1]:.6f}")
    print(f"  Z: {hard_iron[2]:.6f}")
    print()
    print("Soft-Iron Matrix:")
    for i in range(3):
        print(f"  [{soft_iron[i, 0]:.6f}, {soft_iron[i, 1]:.6f}, {soft_iron[i, 2]:.6f}]")
    print()
    
    # YAML format
    print("=== YAML CONFIG (for config_bell412_dataset3.yaml) ===")
    print()
    print("magnetometer:")
    print("  # Hard-iron offset [µT] (CALIBRATED)")
    print(f"  hard_iron_offset: [{hard_iron[0]:.6f}, {hard_iron[1]:.6f}, {hard_iron[2]:.6f}]")
    print()
    print("  # Soft-iron matrix (CALIBRATED)")
    print("  soft_iron_matrix:")
    for i in range(3):
        print(f"    - [{soft_iron[i, 0]:.6f}, {soft_iron[i, 1]:.6f}, {soft_iron[i, 2]:.6f}]")
    print()
    print("  # Magnetic declination [radians]")
    print("  declination: -0.340  # -19.5° for Newfoundland")
    print()
    print("  # Field strength bounds [µT]")
    print(f"  expected_field_strength: {field_cal.mean():.2f}")
    print(f"  min_field_strength: {field_cal.mean() - 2*field_cal.std():.2f}")
    print(f"  max_field_strength: {field_cal.mean() + 2*field_cal.std():.2f}")
    print()
    
    # Generate plot
    plot_calibration(data_raw, data_calibrated, hard_iron)
    
    print("=" * 80)
    print("NEXT STEPS:")
    print("  1. Copy the YAML config above into config_bell412_dataset3.yaml")
    print("  2. Re-run the VIO benchmark: ./benchmark_bell412_multicam.sh")
    print("  3. Expected improvement: 50-80% reduction in position error")
    print("=" * 80)

if __name__ == "__main__":
    main()
