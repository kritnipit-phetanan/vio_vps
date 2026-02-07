#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground Truth Loader and Error Calculator for VIO Systems

Loads ground truth from flight_log_from_gga.csv and computes trajectory errors
against estimated trajectories from vio_vps.py or vio_for_kh.py.

Author: VIO Project
Date: 2025-11-26
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pyproj import Transformer


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """
    Load ground truth from flight_log_from_gga.csv.
    
    CSV format:
        stamp_log, lat_dd, lon_dd, altitude_MSL_m, xSpeed_mph, ySpeed_mph, zSpeed_mph
    
    Args:
        csv_path: Path to flight_log_from_gga.csv
        
    Returns:
        DataFrame with columns: [timestamp, lat, lon, alt_msl, vx_mph, vy_mph, vz_mph]
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Verify required columns
        required_cols = ['stamp_log', 'lat_dd', 'lon_dd', 'altitude_MSL_m']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Rename for consistency
        df = df.rename(columns={
            'stamp_log': 'timestamp',
            'lat_dd': 'lat',
            'lon_dd': 'lon',
            'altitude_MSL_m': 'alt_msl'
        })
        
        # Optional velocity columns (may not exist in all datasets)
        if 'xSpeed_mph' in df.columns:
            df = df.rename(columns={
                'xSpeed_mph': 'vx_mph',
                'ySpeed_mph': 'vy_mph',
                'zSpeed_mph': 'vz_mph'
            })
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"[GT] Loaded {len(df)} ground truth samples from {csv_path}")
        print(f"[GT] Time range: {df['timestamp'].min():.2f} - {df['timestamp'].max():.2f} s")
        print(f"[GT] Lat range: {df['lat'].min():.6f}° - {df['lat'].max():.6f}°")
        print(f"[GT] Lon range: {df['lon'].min():.6f}° - {df['lon'].max():.6f}°") 
        print(f"[GT] Alt range: {df['alt_msl'].min():.1f} - {df['alt_msl'].max():.1f} m MSL")
        
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Ground truth file not found: {csv_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load ground truth: {e}")


def latlon_to_xy(lat: float, lon: float, 
                 origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """
    Convert lat/lon to local ENU coordinates (meters).
    
    Args:
        lat, lon: Target coordinates (degrees)
        origin_lat, origin_lon: Local origin (degrees)
        
    Returns:
        (x, y) in meters (East, North)
    """
    # UTM projection centered at origin
    transformer = Transformer.from_crs(
        "EPSG:4326",  # WGS84
        f"+proj=tmerc +lat_0={origin_lat} +lon_0={origin_lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84",
        always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def compute_trajectory_errors(
    est_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    origin_lat: float,
    origin_lon: float,
    time_column: str = 'time',
    interpolate: bool = True
) -> pd.DataFrame:
    """
    Compute trajectory errors between estimated and ground truth trajectories.
    
    Args:
        est_df: Estimated trajectory DataFrame with columns:
                [time, lat, lon, px (optional), py (optional), pz (optional)]
        gt_df: Ground truth DataFrame from load_ground_truth()
        origin_lat, origin_lon: Local coordinate origin (degrees)
        time_column: Name of timestamp column in est_df
        interpolate: If True, interpolate GT to match estimated timestamps
        
    Returns:
        DataFrame with error metrics:
            [time, est_lat, est_lon, gt_lat, gt_lon, 
             error_lat, error_lon, error_3d, ...]
    """
    # Align time columns
    est_df = est_df.copy()
    est_df['time'] = est_df[time_column]
    
    # Convert GT lat/lon to ENU coordinates
    gt_df = gt_df.copy()
    gt_xy = [latlon_to_xy(row['lat'], row['lon'], origin_lat, origin_lon) 
             for _, row in gt_df.iterrows()]
    gt_df['gt_px'] = [xy[0] for xy in gt_xy]
    gt_df['gt_py'] = [xy[1] for xy in gt_xy]
    gt_df['gt_pz'] = gt_df['alt_msl']
    
    # Convert estimated lat/lon to ENU coordinates (if available)
    if 'lat' in est_df.columns and 'lon' in est_df.columns:
        est_xy = [latlon_to_xy(row['lat'], row['lon'], origin_lat, origin_lon) 
                  for _, row in est_df.iterrows()]
        est_df['est_px'] = [xy[0] for xy in est_xy]
        est_df['est_py'] = [xy[1] for xy in est_xy]
    elif 'px' in est_df.columns and 'py' in est_df.columns:
        est_df['est_px'] = est_df['px']
        est_df['est_py'] = est_df['py']
    else:
        raise ValueError("Estimated trajectory must contain either (lat, lon) or (px, py)")
    
    if 'pz' in est_df.columns:
        est_df['est_pz'] = est_df['pz']
    elif 'altitude_MSL_m' in est_df.columns:
        est_df['est_pz'] = est_df['altitude_MSL_m']
    else:
        print("[WARNING] No altitude found in estimated trajectory, using 0")
        est_df['est_pz'] = 0.0
    
    # Interpolate GT to match estimated timestamps
    if interpolate:
        from scipy.interpolate import interp1d
        
        # Create interpolators for GT trajectory
        gt_times = gt_df['timestamp'].values
        interp_px = interp1d(gt_times, gt_df['gt_px'].values, 
                             kind='linear', fill_value='extrapolate')
        interp_py = interp1d(gt_times, gt_df['gt_py'].values, 
                             kind='linear', fill_value='extrapolate')
        interp_pz = interp1d(gt_times, gt_df['gt_pz'].values, 
                             kind='linear', fill_value='extrapolate')
        
        # Interpolate to estimated timestamps
        est_times = est_df['time'].values
        est_df['gt_px_interp'] = interp_px(est_times)
        est_df['gt_py_interp'] = interp_py(est_times)
        est_df['gt_pz_interp'] = interp_pz(est_times)
    else:
        # Nearest-neighbor matching (simple)
        est_df['gt_px_interp'] = np.nan
        est_df['gt_py_interp'] = np.nan
        est_df['gt_pz_interp'] = np.nan
        
        for i, row in est_df.iterrows():
            t = row['time']
            idx = (gt_df['timestamp'] - t).abs().idxmin()
            est_df.loc[i, 'gt_px_interp'] = gt_df.loc[idx, 'gt_px']
            est_df.loc[i, 'gt_py_interp'] = gt_df.loc[idx, 'gt_py']
            est_df.loc[i, 'gt_pz_interp'] = gt_df.loc[idx, 'gt_pz']
    
    # Compute errors
    est_df['error_px'] = est_df['est_px'] - est_df['gt_px_interp']
    est_df['error_py'] = est_df['est_py'] - est_df['gt_py_interp']
    est_df['error_pz'] = est_df['est_pz'] - est_df['gt_pz_interp']
    
    # 2D horizontal error (XY plane)
    est_df['error_2d'] = np.sqrt(est_df['error_px']**2 + est_df['error_py']**2)
    
    # 3D error
    est_df['error_3d'] = np.sqrt(
        est_df['error_px']**2 + 
        est_df['error_py']**2 + 
        est_df['error_pz']**2
    )
    
    # Summary statistics
    print("\n" + "="*60)
    print("TRAJECTORY ERROR STATISTICS")
    print("="*60)
    print(f"Number of samples: {len(est_df)}")
    print(f"\n2D Horizontal Error (XY plane):")
    print(f"  Mean:   {est_df['error_2d'].mean():.3f} m")
    print(f"  Median: {est_df['error_2d'].median():.3f} m")
    print(f"  Std:    {est_df['error_2d'].std():.3f} m")
    print(f"  Max:    {est_df['error_2d'].max():.3f} m")
    print(f"  RMSE:   {np.sqrt((est_df['error_2d']**2).mean()):.3f} m")
    
    print(f"\n3D Position Error:")
    print(f"  Mean:   {est_df['error_3d'].mean():.3f} m")
    print(f"  Median: {est_df['error_3d'].median():.3f} m")
    print(f"  Std:    {est_df['error_3d'].std():.3f} m")
    print(f"  Max:    {est_df['error_3d'].max():.3f} m")
    print(f"  RMSE:   {np.sqrt((est_df['error_3d']**2).mean()):.3f} m")
    
    print(f"\nPer-Axis Errors:")
    print(f"  X (East):  mean={est_df['error_px'].mean():+.3f} m, std={est_df['error_px'].std():.3f} m")
    print(f"  Y (North): mean={est_df['error_py'].mean():+.3f} m, std={est_df['error_py'].std():.3f} m")
    print(f"  Z (Up):    mean={est_df['error_pz'].mean():+.3f} m, std={est_df['error_pz'].std():.3f} m")
    print("="*60 + "\n")
    
    return est_df


def save_error_csv(error_df: pd.DataFrame, output_path: str):
    """
    Save error DataFrame to CSV.
    
    Args:
        error_df: DataFrame from compute_trajectory_errors()
        output_path: Output CSV path
    """
    # Select relevant columns
    columns_to_save = [
        'time', 
        'est_px', 'est_py', 'est_pz',
        'gt_px_interp', 'gt_py_interp', 'gt_pz_interp',
        'error_px', 'error_py', 'error_pz',
        'error_2d', 'error_3d'
    ]
    
    # Filter to existing columns
    available_cols = [col for col in columns_to_save if col in error_df.columns]
    
    error_df[available_cols].to_csv(output_path, index=False, float_format='%.6f')
    print(f"[SAVE] Error metrics saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute VIO trajectory errors against ground truth")
    parser.add_argument("--estimated", required=True, help="Path to estimated trajectory CSV (pose.csv)")
    parser.add_argument("--ground_truth", required=True, help="Path to flight_log_from_gga.csv")
    parser.add_argument("--output", default="error_analysis.csv", help="Output error CSV path")
    parser.add_argument("--origin_lat", type=float, required=True, help="Local origin latitude (degrees)")
    parser.add_argument("--origin_lon", type=float, required=True, help="Local origin longitude (degrees)")
    
    args = parser.parse_args()
    
    # Load data
    print("[1/3] Loading ground truth...")
    gt_df = load_ground_truth(args.ground_truth)
    
    print("\n[2/3] Loading estimated trajectory...")
    est_df = pd.read_csv(args.estimated)
    print(f"[EST] Loaded {len(est_df)} estimated poses")
    
    # Compute errors
    print("\n[3/3] Computing errors...")
    error_df = compute_trajectory_errors(
        est_df, gt_df, 
        args.origin_lat, args.origin_lon,
        time_column='time'  # Adjust if needed
    )
    
    # Save results
    save_error_csv(error_df, args.output)
    
    print("\n✅ Error analysis complete!")
