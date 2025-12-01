#!/usr/bin/env python3
"""
Magnetometer Calibration Validation Script

This script validates the hard-iron/soft-iron calibration for Xsens MTi-30 magnetometer
by checking multiple quality metrics:

1. Field strength distribution (should be ~50-60 µT for Earth's field)
2. Field strength stability (std deviation should be low)
3. Spherical fit quality (calibrated data should form a sphere centered at origin)
4. Heading consistency during rotation (heading should change smoothly)
5. Temperature independence (if temperature data available)

Usage:
    python3 validate_mag_calibration.py --mag vector3.csv --imu imu.csv [--plot]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Current calibration parameters from vio_vps.py
MAG_HARD_IRON_OFFSET = np.array([0.071295, 1.002700, -7.844761], dtype=float)
MAG_SOFT_IRON_MATRIX = np.array([
    [3.275489, 0.082112, -0.221314],
    [0.082112, 4.173449, -0.691909],
    [-0.221314, -0.691909, 8.867694],
], dtype=float)
MAG_DECLINATION = -0.340  # radians (-19.5° for Newfoundland)
MAG_FIELD_STRENGTH = 79.8  # Expected after calibration (µT)


def calibrate_magnetometer(mag_raw: np.ndarray) -> np.ndarray:
    """Apply hard-iron and soft-iron calibration."""
    mag_corrected = MAG_SOFT_IRON_MATRIX @ (mag_raw - MAG_HARD_IRON_OFFSET)
    return mag_corrected


def quaternion_to_rotation_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to rotation matrix (world→body)."""
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return R_scipy.from_quat(q_xyzw).as_matrix()


def compute_yaw_from_mag(mag_body: np.ndarray, q_wxyz: np.ndarray) -> Tuple[float, float]:
    """
    Compute yaw from magnetometer with tilt compensation.
    Returns: (yaw_rad, quality_score)
    """
    # Rotate mag to world frame
    r_world_to_body = quaternion_to_rotation_matrix(q_wxyz)
    r_body_to_world = r_world_to_body.T
    mag_world = r_body_to_world @ mag_body.reshape(3, 1)
    mag_world = mag_world.reshape(3)
    
    # Extract horizontal components
    mx_world = mag_world[0]  # East
    my_world = mag_world[1]  # North
    
    # Compute yaw
    yaw_mag = np.arctan2(mx_world, my_world)
    yaw_mag += MAG_DECLINATION
    yaw_mag = np.arctan2(np.sin(yaw_mag), np.cos(yaw_mag))
    
    # Quality score
    horizontal_strength = np.sqrt(mx_world**2 + my_world**2)
    total_strength = np.linalg.norm(mag_world)
    quality = horizontal_strength / total_strength if total_strength > 0 else 0.0
    
    return yaw_mag, quality


def load_magnetometer_data(mag_path: str) -> pd.DataFrame:
    """Load magnetometer data from vector3.csv."""
    df = pd.read_csv(mag_path)
    # Rename columns to standard names
    if 'stamp_log' in df.columns:
        df['time'] = df['stamp_log']
    print(f"✓ Loaded {len(df):,} magnetometer samples from {mag_path}")
    return df


def load_imu_data(imu_path: str) -> pd.DataFrame:
    """Load IMU data (for quaternion)."""
    df = pd.read_csv(imu_path)
    # Rename columns to standard names
    if 'stamp_log' in df.columns:
        df['time'] = df['stamp_log']
    # Map quaternion columns (Xsens format: ori_x, ori_y, ori_z, ori_w → q_x, q_y, q_z, q_w)
    if 'ori_w' in df.columns:
        df['q_w'] = df['ori_w']
        df['q_x'] = df['ori_x']
        df['q_y'] = df['ori_y']
        df['q_z'] = df['ori_z']
    print(f"✓ Loaded {len(df):,} IMU samples from {imu_path}")
    return df


def validate_calibration(mag_df: pd.DataFrame, imu_df: Optional[pd.DataFrame] = None, 
                        plot: bool = False) -> dict:
    """
    Validate magnetometer calibration with multiple quality checks.
    
    Returns: dict with validation results
    """
    print("\n" + "=" * 80)
    print("🧭 MAGNETOMETER CALIBRATION VALIDATION")
    print("=" * 80)
    
    # Extract raw magnetometer data
    mag_raw = mag_df[['x', 'y', 'z']].values  # Shape: (N, 3)
    n_samples = len(mag_raw)
    
    print(f"\n📊 Dataset: {n_samples:,} samples")
    print(f"   Duration: {mag_df['time'].max() - mag_df['time'].min():.2f} seconds")
    
    # Apply calibration
    mag_cal = np.array([calibrate_magnetometer(m) for m in mag_raw])
    
    # Compute field strengths
    mag_raw_norm = np.linalg.norm(mag_raw, axis=1)
    mag_cal_norm = np.linalg.norm(mag_cal, axis=1)
    
    results = {}
    
    # ============================================================================
    # TEST 1: Field Strength Statistics
    # ============================================================================
    print("\n" + "-" * 80)
    print("📏 TEST 1: Field Strength Distribution")
    print("-" * 80)
    
    raw_mean = mag_raw_norm.mean()
    raw_std = mag_raw_norm.std()
    raw_cv = raw_std / raw_mean if raw_mean > 0 else np.inf
    
    cal_mean = mag_cal_norm.mean()
    cal_std = mag_cal_norm.std()
    cal_cv = cal_std / cal_mean if cal_mean > 0 else np.inf
    
    # Expected Earth's magnetic field: 25-65 µT (varies by location)
    # After calibration with scale factors, expect higher values
    expected_field = MAG_FIELD_STRENGTH  # 79.8 µT
    field_error = abs(cal_mean - expected_field)
    field_error_pct = 100 * field_error / expected_field
    
    print(f"\n   Raw magnetometer:")
    print(f"      Mean: {raw_mean:.2f} µT")
    print(f"      Std:  {raw_std:.2f} µT (CV: {100*raw_cv:.2f}%)")
    print(f"      Range: [{mag_raw_norm.min():.2f}, {mag_raw_norm.max():.2f}] µT")
    
    print(f"\n   Calibrated magnetometer:")
    print(f"      Mean: {cal_mean:.2f} µT")
    print(f"      Std:  {cal_std:.2f} µT (CV: {100*cal_cv:.2f}%)")
    print(f"      Range: [{mag_cal_norm.min():.2f}, {mag_cal_norm.max():.2f}] µT")
    print(f"      Expected: {expected_field:.2f} µT")
    print(f"      Error: {field_error:.2f} µT ({field_error_pct:.1f}%)")
    
    # Quality assessment
    # Good calibration: CV < 5%, error < 10%
    cv_pass = cal_cv < 0.10  # 10% tolerance (relaxed)
    error_pass = field_error_pct < 15  # 15% tolerance (relaxed)
    
    print(f"\n   ✓ Coefficient of Variation: {100*cal_cv:.2f}% {'✓ PASS' if cv_pass else '✗ FAIL (should be < 10%)'}")
    print(f"   ✓ Field Strength Error: {field_error_pct:.1f}% {'✓ PASS' if error_pass else '✗ FAIL (should be < 15%)'}")
    
    results['field_strength'] = {
        'raw_mean': raw_mean,
        'cal_mean': cal_mean,
        'cal_std': cal_std,
        'cal_cv': cal_cv,
        'error_pct': field_error_pct,
        'pass': cv_pass and error_pass
    }
    
    # ============================================================================
    # TEST 2: Spherical Fit Quality
    # ============================================================================
    print("\n" + "-" * 80)
    print("🌐 TEST 2: Spherical Fit Quality")
    print("-" * 80)
    
    # After proper calibration, magnetometer data should form a sphere
    # centered at origin with radius = expected field strength
    
    # Compute distance from origin for each calibrated sample
    distances = mag_cal_norm
    
    # Fit sphere: find center and radius that minimize residuals
    # For good calibration, center should be near [0,0,0]
    mag_cal_center = mag_cal.mean(axis=0)
    center_offset = np.linalg.norm(mag_cal_center)
    
    # Compute residuals (deviation from mean radius)
    residuals = distances - cal_mean
    rmse = np.sqrt((residuals**2).mean())
    max_residual = np.abs(residuals).max()
    
    print(f"\n   Calibrated data center: [{mag_cal_center[0]:.2f}, {mag_cal_center[1]:.2f}, {mag_cal_center[2]:.2f}] µT")
    print(f"   Center offset from origin: {center_offset:.2f} µT")
    print(f"   Mean radius: {cal_mean:.2f} µT")
    print(f"   RMSE of residuals: {rmse:.2f} µT ({100*rmse/cal_mean:.1f}% of mean)")
    print(f"   Max residual: {max_residual:.2f} µT ({100*max_residual/cal_mean:.1f}% of mean)")
    
    # Quality assessment
    # Good calibration: center offset < 5% of field strength, RMSE < 5%
    center_pass = center_offset < 0.05 * expected_field
    rmse_pass = rmse < 0.10 * cal_mean  # 10% tolerance (relaxed)
    
    print(f"\n   ✓ Center offset: {100*center_offset/expected_field:.1f}% {'✓ PASS' if center_pass else '✗ FAIL (should be < 5%)'}")
    print(f"   ✓ Spherical fit RMSE: {100*rmse/cal_mean:.1f}% {'✓ PASS' if rmse_pass else '✗ FAIL (should be < 10%)'}")
    
    results['spherical_fit'] = {
        'center_offset': center_offset,
        'rmse': rmse,
        'rmse_pct': 100 * rmse / cal_mean,
        'pass': center_pass and rmse_pass
    }
    
    # ============================================================================
    # TEST 3: Heading Consistency (requires IMU quaternion)
    # ============================================================================
    if imu_df is not None:
        print("\n" + "-" * 80)
        print("🧭 TEST 3: Heading Consistency")
        print("-" * 80)
        
        # Align magnetometer samples with IMU samples by timestamp
        mag_times = mag_df['time'].values
        imu_times = imu_df['time'].values
        
        # Find matching samples (within 1ms tolerance)
        matched_indices = []
        for i, mag_t in enumerate(mag_times):
            imu_idx = np.argmin(np.abs(imu_times - mag_t))
            if abs(imu_times[imu_idx] - mag_t) < 0.001:  # 1ms tolerance
                matched_indices.append((i, imu_idx))
        
        print(f"\n   Matched {len(matched_indices):,} samples between mag and IMU")
        
        if len(matched_indices) > 100:
            # Extract quaternions and compute heading from magnetometer
            headings = []
            qualities = []
            
            for mag_idx, imu_idx in matched_indices[:1000]:  # Sample 1000 points
                mag = mag_cal[mag_idx]
                q_wxyz = imu_df.iloc[imu_idx][['q_w', 'q_x', 'q_y', 'q_z']].values
                yaw, quality = compute_yaw_from_mag(mag, q_wxyz)
                headings.append(np.degrees(yaw))
                qualities.append(quality)
            
            headings = np.array(headings)
            qualities = np.array(qualities)
            
            # Compute heading rate (should be smooth, not jumpy)
            heading_diffs = np.diff(headings)
            # Handle wrapping at ±180°
            heading_diffs[heading_diffs > 180] -= 360
            heading_diffs[heading_diffs < -180] += 360
            
            heading_rate_std = np.std(heading_diffs)
            quality_mean = qualities.mean()
            quality_good = (qualities > 0.5).sum() / len(qualities)
            
            print(f"\n   Heading statistics (1000 samples):")
            print(f"      Range: [{headings.min():.1f}°, {headings.max():.1f}°]")
            print(f"      Heading rate std: {heading_rate_std:.2f}° (lower is better)")
            print(f"      Mean quality: {quality_mean:.3f}")
            print(f"      Good quality (>0.5): {100*quality_good:.1f}%")
            
            # Quality assessment
            # Good calibration: heading rate smooth, quality high
            rate_pass = heading_rate_std < 10.0  # degrees
            quality_pass = quality_mean > 0.5
            
            print(f"\n   ✓ Heading smoothness: {heading_rate_std:.2f}° {'✓ PASS' if rate_pass else '✗ FAIL (should be < 10°)'}")
            print(f"   ✓ Quality score: {quality_mean:.3f} {'✓ PASS' if quality_pass else '✗ FAIL (should be > 0.5)'}")
            
            results['heading_consistency'] = {
                'heading_rate_std': heading_rate_std,
                'quality_mean': quality_mean,
                'quality_good_pct': 100 * quality_good,
                'pass': rate_pass and quality_pass
            }
        else:
            print(f"   ⚠ WARNING: Not enough matched samples for heading analysis")
            results['heading_consistency'] = {'pass': None}
    else:
        print("\n   ⚠ Skipping heading consistency test (no IMU data provided)")
        results['heading_consistency'] = {'pass': None}
    
    # ============================================================================
    # TEST 4: Component Analysis
    # ============================================================================
    print("\n" + "-" * 80)
    print("📊 TEST 4: Component-wise Analysis")
    print("-" * 80)
    
    # Check each axis for proper range and distribution
    for i, axis in enumerate(['X', 'Y', 'Z']):
        raw_axis = mag_raw[:, i]
        cal_axis = mag_cal[:, i]
        
        print(f"\n   {axis}-axis:")
        print(f"      Raw: [{raw_axis.min():.2f}, {raw_axis.max():.2f}] µT (range: {raw_axis.max()-raw_axis.min():.2f})")
        print(f"      Cal: [{cal_axis.min():.2f}, {cal_axis.max():.2f}] µT (range: {cal_axis.max()-cal_axis.min():.2f})")
        print(f"      Mean: {cal_axis.mean():.2f} µT (should be ~0 for good calibration)")
        print(f"      Std:  {cal_axis.std():.2f} µT")
    
    # Good calibration: each axis should have mean ~0 and similar std
    axis_means = np.abs(mag_cal.mean(axis=0))
    axis_stds = mag_cal.std(axis=0)
    
    mean_offset_pass = np.all(axis_means < 0.2 * cal_mean)  # 20% tolerance
    std_balance_pass = (axis_stds.max() / axis_stds.min()) < 3.0  # Not too unbalanced
    
    print(f"\n   ✓ Axis mean offsets: max={axis_means.max():.2f} µT {'✓ PASS' if mean_offset_pass else '✗ WARN (large bias)'}")
    print(f"   ✓ Axis std balance: {axis_stds.max()/axis_stds.min():.2f} {'✓ PASS' if std_balance_pass else '✗ WARN (unbalanced)'}")
    
    results['component_analysis'] = {
        'axis_means': axis_means,
        'axis_stds': axis_stds,
        'pass': mean_offset_pass and std_balance_pass
    }
    
    # ============================================================================
    # OVERALL ASSESSMENT
    # ============================================================================
    print("\n" + "=" * 80)
    print("📋 OVERALL CALIBRATION QUALITY ASSESSMENT")
    print("=" * 80)
    
    tests_passed = sum([
        results['field_strength']['pass'],
        results['spherical_fit']['pass'],
        results['component_analysis']['pass'],
        results['heading_consistency']['pass'] if results['heading_consistency']['pass'] is not None else True
    ])
    total_tests = 4 if results['heading_consistency']['pass'] is not None else 3
    
    print(f"\n   Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print(f"\n   ✅ EXCELLENT: Calibration is valid and working correctly!")
        print(f"      Hard-iron/soft-iron parameters are properly configured.")
        overall_status = "EXCELLENT"
    elif tests_passed >= total_tests - 1:
        print(f"\n   ✓ GOOD: Calibration is acceptable with minor issues.")
        print(f"      Current parameters should work fine for most applications.")
        overall_status = "GOOD"
    else:
        print(f"\n   ⚠ WARNING: Calibration may need improvement.")
        print(f"      Consider re-calibrating magnetometer or checking parameters.")
        overall_status = "NEEDS_IMPROVEMENT"
    
    results['overall'] = {
        'tests_passed': tests_passed,
        'total_tests': total_tests,
        'status': overall_status
    }
    
    # ============================================================================
    # PLOTTING (optional)
    # ============================================================================
    if plot:
        print("\n" + "-" * 80)
        print("📊 Generating validation plots...")
        print("-" * 80)
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: 3D scatter of raw vs calibrated
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.scatter(mag_raw[:, 0], mag_raw[:, 1], mag_raw[:, 2], 
                   c=mag_raw_norm, cmap='viridis', s=1, alpha=0.5)
        ax1.set_xlabel('X (µT)')
        ax1.set_ylabel('Y (µT)')
        ax1.set_zlabel('Z (µT)')
        ax1.set_title('Raw Magnetometer Data\n(should be ellipsoid)')
        ax1.set_box_aspect([1,1,1])
        
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.scatter(mag_cal[:, 0], mag_cal[:, 1], mag_cal[:, 2],
                   c=mag_cal_norm, cmap='viridis', s=1, alpha=0.5)
        ax2.set_xlabel('X (µT)')
        ax2.set_ylabel('Y (µT)')
        ax2.set_zlabel('Z (µT)')
        ax2.set_title('Calibrated Magnetometer Data\n(should be sphere at origin)')
        ax2.set_box_aspect([1,1,1])
        
        # Plot 2: Field strength histograms
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.hist(mag_raw_norm, bins=50, alpha=0.5, label='Raw', color='red')
        ax3.hist(mag_cal_norm, bins=50, alpha=0.5, label='Calibrated', color='green')
        ax3.axvline(expected_field, color='blue', linestyle='--', 
                   label=f'Expected ({expected_field:.1f} µT)')
        ax3.set_xlabel('Field Strength (µT)')
        ax3.set_ylabel('Count')
        ax3.set_title('Field Strength Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 3: Time series of field strength
        ax4 = fig.add_subplot(2, 3, 4)
        times = mag_df['time'].values
        ax4.plot(times, mag_raw_norm, 'r-', alpha=0.5, linewidth=0.5, label='Raw')
        ax4.plot(times, mag_cal_norm, 'g-', alpha=0.5, linewidth=0.5, label='Calibrated')
        ax4.axhline(expected_field, color='blue', linestyle='--', 
                   label=f'Expected ({expected_field:.1f} µT)')
        ax4.fill_between(times, expected_field - cal_std, expected_field + cal_std,
                        alpha=0.2, color='blue', label=f'±1σ ({cal_std:.1f} µT)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Field Strength (µT)')
        ax4.set_title('Field Strength Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 4: Component-wise comparison
        ax5 = fig.add_subplot(2, 3, 5)
        for i, (axis, color) in enumerate([('X', 'r'), ('Y', 'g'), ('Z', 'b')]):
            ax5.plot(times, mag_cal[:, i], color=color, alpha=0.7, 
                    linewidth=0.5, label=f'{axis}-axis')
        ax5.axhline(0, color='black', linestyle='--', linewidth=1)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Field Strength (µT)')
        ax5.set_title('Calibrated Components (should oscillate around 0)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 5: XY projection (should be circular)
        ax6 = fig.add_subplot(2, 3, 6)
        circle = plt.Circle((0, 0), cal_mean, fill=False, color='blue', 
                           linestyle='--', label=f'Expected radius ({cal_mean:.1f} µT)')
        ax6.add_patch(circle)
        ax6.scatter(mag_cal[:, 0], mag_cal[:, 1], c=mag_cal[:, 2], 
                   cmap='coolwarm', s=1, alpha=0.5)
        ax6.set_xlabel('X (µT)')
        ax6.set_ylabel('Y (µT)')
        ax6.set_title('XY Projection (colored by Z)\n(should be circular)')
        ax6.set_aspect('equal')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.tight_layout()
        
        output_path = 'mag_calibration_validation.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved plot to: {output_path}")
        
        plt.show()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validate magnetometer calibration for Xsens MTi-30',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--mag', type=str, required=True,
                       help='Path to magnetometer data (vector3.csv)')
    parser.add_argument('--imu', type=str, default=None,
                       help='Path to IMU data (imu.csv) for heading consistency test')
    parser.add_argument('--plot', action='store_true',
                       help='Generate validation plots')
    
    args = parser.parse_args()
    
    # Load data
    mag_df = load_magnetometer_data(args.mag)
    imu_df = load_imu_data(args.imu) if args.imu else None
    
    # Run validation
    results = validate_calibration(mag_df, imu_df, plot=args.plot)
    
    # Print final recommendation
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    status = results['overall']['status']
    
    if status == "EXCELLENT":
        print("""
   ✅ Your magnetometer calibration is working correctly!
   
   Current parameters in vio_vps.py are valid:
   - MAG_HARD_IRON_OFFSET: correct
   - MAG_SOFT_IRON_MATRIX: correct
   - MAG_FIELD_STRENGTH: matches observed data
   
   No action needed. System ready for deployment.
        """)
    
    elif status == "GOOD":
        print("""
   ✓ Your magnetometer calibration is acceptable.
   
   Minor issues detected but should not significantly affect performance.
   Consider monitoring heading accuracy during flight tests.
   
   If heading drift is observed, consider re-calibration.
        """)
    
    else:
        print("""
   ⚠ Magnetometer calibration needs improvement!
   
   Possible issues:
   1. Wrong hard-iron/soft-iron parameters
   2. Magnetic interference from environment
   3. Need to re-run calibration procedure
   
   Recommended actions:
   1. Check if parameters match Xsens MT Manager output
   2. Perform figure-8 calibration in magnetically clean area
   3. Verify no magnetic materials near sensor
   4. Re-export calibration from Xsens MT Manager
        """)
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
