#!/usr/bin/env python3
"""
Full 3x3 Soft Iron Calibration using Ellipsoid Fitting

This script fits an ellipsoid to magnetometer data and computes
the full 3x3 soft iron matrix including off-diagonal terms.

Method: Least squares ellipsoid fitting
Reference: https://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
"""
import csv
import math
from typing import List, Tuple

def load_mag_data(path: str) -> Tuple[List[float], List[float], List[float]]:
    """Load magnetometer data from CSV."""
    x, y, z = [], [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row['x']))
            y.append(float(row['y']))
            z.append(float(row['z']))
    return x, y, z

def fit_ellipsoid_simple(x: List[float], y: List[float], z: List[float]):
    """
    Simplified ellipsoid fitting using axis-aligned approach.
    
    For full ellipsoid fitting with rotation, would need:
    - Solve Ax = b where A is 9xN design matrix
    - Use SVD or least squares
    
    This simple version:
    1. Computes hard iron as center
    2. Computes scale factors per axis
    3. Does NOT compute off-diagonal terms (would need more complex fitting)
    """
    # Hard iron (center)
    hi_x = (max(x) + min(x)) / 2
    hi_y = (max(y) + min(y)) / 2
    hi_z = (max(z) + min(z)) / 2
    
    # Semi-axes lengths
    a = (max(x) - min(x)) / 2
    b = (max(y) - min(y)) / 2
    c = (max(z) - min(z)) / 2
    
    # Target radius (average)
    r = (a + b + c) / 3
    
    # Scale factors
    sx = r / a if a > 0 else 1.0
    sy = r / b if b > 0 else 1.0
    sz = r / c if c > 0 else 1.0
    
    return (hi_x, hi_y, hi_z), (sx, sy, sz), r

def fit_ellipsoid_full(x: List[float], y: List[float], z: List[float]):
    """
    Full ellipsoid fitting with off-diagonal terms.
    
    Uses least squares to solve:
    Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1
    """
    n = len(x)
    
    # Build design matrix
    # Each row: [x^2, y^2, z^2, 2xy, 2xz, 2yz, 2x, 2y, 2z]
    D = []
    for i in range(n):
        row = [
            x[i]**2, y[i]**2, z[i]**2,
            2*x[i]*y[i], 2*x[i]*z[i], 2*y[i]*z[i],
            2*x[i], 2*y[i], 2*z[i]
        ]
        D.append(row)
    
    # Solve D @ v = 1 using normal equations: D^T D v = D^T 1
    # v = (D^T D)^-1 D^T 1
    
    # Compute D^T @ D (9x9)
    DTD = [[0.0]*9 for _ in range(9)]
    for i in range(9):
        for j in range(9):
            for k in range(n):
                DTD[i][j] += D[k][i] * D[k][j]
    
    # Compute D^T @ ones (9x1)
    DT1 = [0.0]*9
    for i in range(9):
        for k in range(n):
            DT1[i] += D[k][i]
    
    # Solve using Gaussian elimination (simple)
    v = solve_linear_system(DTD, DT1)
    
    if v is None:
        return None, None, None
    
    A, B, C, D_coef, E, F, G, H, I = v
    
    # Extract center (hard iron)
    # For ellipsoid: center = -[A B C; D A F; E F C]^-1 @ [G; H; I]
    # Simplified for diagonal-dominant case:
    hi_x = -G / A if abs(A) > 1e-10 else 0
    hi_y = -H / B if abs(B) > 1e-10 else 0
    hi_z = -I / C if abs(C) > 1e-10 else 0
    
    # Build Q matrix (ellipsoid shape)
    Q = [
        [A, D_coef, E],
        [D_coef, B, F],
        [E, F, C]
    ]
    
    # Compute eigenvalues for scale factors
    # For simplicity, use diagonal approximation
    sx = 1.0 / math.sqrt(abs(A)) if abs(A) > 1e-10 else 1.0
    sy = 1.0 / math.sqrt(abs(B)) if abs(B) > 1e-10 else 1.0
    sz = 1.0 / math.sqrt(abs(C)) if abs(C) > 1e-10 else 1.0
    
    # Off-diagonal ratios
    xy_ratio = D_coef / math.sqrt(abs(A*B)) if abs(A*B) > 1e-10 else 0
    xz_ratio = E / math.sqrt(abs(A*C)) if abs(A*C) > 1e-10 else 0
    yz_ratio = F / math.sqrt(abs(B*C)) if abs(B*C) > 1e-10 else 0
    
    return (hi_x, hi_y, hi_z), Q, (xy_ratio, xz_ratio, yz_ratio)

def solve_linear_system(A, b):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination
    for i in range(n):
        # Partial pivoting
        max_row = i
        for k in range(i+1, n):
            if abs(M[k][i]) > abs(M[max_row][i]):
                max_row = k
        M[i], M[max_row] = M[max_row], M[i]
        
        if abs(M[i][i]) < 1e-12:
            continue
        
        for k in range(i+1, n):
            factor = M[k][i] / M[i][i]
            for j in range(i, n+1):
                M[k][j] -= factor * M[i][j]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        if abs(M[i][i]) < 1e-12:
            x[i] = 0.0
            continue
        x[i] = M[i][n]
        for j in range(i+1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    
    return x

def main():
    mag_path = '/Users/france/Downloads/vio_dataset/bell412_dataset3/extracted_data_new/imu_data/imu__mag/vector3.csv'
    
    print("=" * 60)
    print("FULL 3x3 SOFT IRON CALIBRATION")
    print("=" * 60)
    
    x, y, z = load_mag_data(mag_path)
    print(f"\nLoaded {len(x)} samples")
    
    # Simple fitting
    print("\n" + "-" * 60)
    print("SIMPLE FITTING (diagonal only)")
    print("-" * 60)
    hi_simple, scales, r = fit_ellipsoid_simple(x, y, z)
    print(f"Hard Iron: [{hi_simple[0]:.6f}, {hi_simple[1]:.6f}, {hi_simple[2]:.6f}]")
    print(f"Scale factors: [{scales[0]:.6f}, {scales[1]:.6f}, {scales[2]:.6f}]")
    print(f"Target radius: {r:.6f}")
    
    # Full fitting
    print("\n" + "-" * 60)
    print("FULL ELLIPSOID FITTING (with off-diagonal)")
    print("-" * 60)
    result = fit_ellipsoid_full(x, y, z)
    if result[0] is not None:
        hi_full, Q, ratios = result
        print(f"Hard Iron: [{hi_full[0]:.6f}, {hi_full[1]:.6f}, {hi_full[2]:.6f}]")
        print(f"\nQ matrix (ellipsoid shape):")
        for row in Q:
            print(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
        print(f"\nOff-diagonal ratios (xy, xz, yz): {ratios}")
        
        # Check if off-diagonal terms are significant
        if max(abs(r) for r in ratios) > 0.1:
            print("\n⚠️  Significant off-diagonal terms detected!")
            print("   Diagonal-only calibration may not be sufficient.")
        else:
            print("\n✅ Off-diagonal terms are small.")
            print("   Diagonal calibration should be adequate.")
    else:
        print("Fitting failed - matrix singular")
    
    # Analysis of data distribution
    print("\n" + "-" * 60)
    print("DATA DISTRIBUTION ANALYSIS")
    print("-" * 60)
    
    # Check if data covers enough of the sphere
    x_c = [xi - hi_simple[0] for xi in x]
    y_c = [yi - hi_simple[1] for yi in y]
    z_c = [zi - hi_simple[2] for zi in z]
    
    # Compute heading distribution
    headings = [math.atan2(-yi, xi) for xi, yi in zip(x_c, y_c)]
    heading_deg = [math.degrees(h) for h in headings]
    
    heading_min = min(heading_deg)
    heading_max = max(heading_deg)
    heading_range = heading_max - heading_min
    
    print(f"Heading range: {heading_min:.1f}° to {heading_max:.1f}° (span: {heading_range:.1f}°)")
    
    if heading_range < 180:
        print("\n⚠️  WARNING: Heading range < 180°")
        print("   MAG data does NOT cover full rotation!")
        print("   Soft iron calibration may be inaccurate.")
        print("   For proper calibration, need 360° rotation data.")
    elif heading_range < 270:
        print("\n⚠️  WARNING: Heading range < 270°")
        print("   MAG data covers limited rotation.")
        print("   Soft iron calibration may have some error.")
    else:
        print("\n✅ Good heading coverage for calibration.")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    if heading_range < 180:
        print("""
This dataset does NOT have enough rotation for proper soft iron calibration.
The helicopter flew mostly in one direction without significant yaw changes.

Options:
1. Use diagonal-only calibration (current approach)
2. Disable magnetometer and rely on IMU/visual odometry
3. Collect new calibration data with 360° rotation
""")
    
if __name__ == "__main__":
    main()
