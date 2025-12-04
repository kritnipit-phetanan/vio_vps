#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Evaluation: VIO vs Ground Truth

Computes comprehensive metrics comparing VIO estimates against GPS ground truth:
- Position errors (ATE, RTE, drift rate)
- Orientation errors (yaw, roll, pitch)
- Velocity errors
- Trajectory alignment (scale, rotation)
- Temporal statistics (min, max, mean, median, RMSE, std)

Author: VIO Evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy
from typing import Dict, List, Tuple
import os
import argparse


def load_vio_results(error_log_path: str) -> pd.DataFrame:
    """Load VIO error log with computed errors."""
    df = pd.read_csv(error_log_path)
    df['t_rel'] = df['t'] - df['t'].min()
    return df


def load_gps_ground_truth(gps_path: str) -> pd.DataFrame:
    """Load GPS ground truth data."""
    df = pd.read_csv(gps_path)
    df['t'] = df['stamp_log']
    df['t_rel'] = df['t'] - df['t'].min()
    return df


def compute_ate(pos_errors: np.ndarray) -> Dict[str, float]:
    """
    Absolute Trajectory Error (ATE): measures global consistency.
    
    Returns: dict with RMSE, mean, median, min, max, std
    """
    return {
        'rmse': np.sqrt(np.mean(pos_errors**2)),
        'mean': np.mean(pos_errors),
        'median': np.median(pos_errors),
        'min': np.min(pos_errors),
        'max': np.max(pos_errors),
        'std': np.std(pos_errors),
    }


def compute_rte(positions: np.ndarray, gt_positions: np.ndarray, 
                timestamps: np.ndarray, delta_t: float = 10.0) -> Dict[str, float]:
    """
    Relative Trajectory Error (RTE): measures local consistency over time windows.
    
    Args:
        positions: VIO positions (N, 3)
        gt_positions: GT positions (N, 3)
        timestamps: Timestamps (N,)
        delta_t: Time window in seconds
    
    Returns: dict with mean, median, std of relative errors
    """
    rte_errors = []
    
    for i in range(len(timestamps)):
        t_i = timestamps[i]
        # Find closest timestamp at t_i + delta_t
        j = np.argmin(np.abs(timestamps - (t_i + delta_t)))
        
        if j <= i or timestamps[j] - t_i < delta_t * 0.8:
            continue
        
        # Compute relative motion
        vio_rel = np.linalg.norm(positions[j] - positions[i])
        gt_rel = np.linalg.norm(gt_positions[j] - gt_positions[i])
        
        # Relative error
        rte = np.abs(vio_rel - gt_rel)
        rte_errors.append(rte)
    
    if len(rte_errors) == 0:
        return {'mean': np.nan, 'median': np.nan, 'std': np.nan}
    
    rte_errors = np.array(rte_errors)
    return {
        'mean': np.mean(rte_errors),
        'median': np.median(rte_errors),
        'std': np.std(rte_errors),
        'rmse': np.sqrt(np.mean(rte_errors**2)),
    }


def compute_drift_rate(pos_errors: np.ndarray, timestamps: np.ndarray) -> Dict[str, float]:
    """
    Compute drift rate (m/s) using linear regression.
    
    Returns: dict with slope (m/s), intercept, r_squared
    """
    # Remove NaN values
    valid = ~np.isnan(pos_errors)
    if np.sum(valid) < 2:
        return {'slope_m_per_s': np.nan, 'intercept': np.nan, 'r_squared': np.nan}
    
    t = timestamps[valid]
    e = pos_errors[valid]
    
    # Linear regression: e = a*t + b
    A = np.vstack([t, np.ones(len(t))]).T
    slope, intercept = np.linalg.lstsq(A, e, rcond=None)[0]
    
    # R-squared
    e_pred = slope * t + intercept
    ss_res = np.sum((e - e_pred)**2)
    ss_tot = np.sum((e - np.mean(e))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'slope_m_per_s': slope,
        'intercept_m': intercept,
        'r_squared': r_squared,
    }


def compute_orientation_errors(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute yaw, roll, pitch errors.
    
    Returns: dict with stats for each angle
    """
    errors = {}
    
    # Yaw error (already in error log)
    if 'yaw_error_deg' in df.columns:
        yaw_err = df['yaw_error_deg'].dropna()
        errors['yaw_deg'] = {
            'mean': yaw_err.mean(),
            'median': yaw_err.median(),
            'std': yaw_err.std(),
            'rmse': np.sqrt(np.mean(yaw_err**2)),
            'min': yaw_err.min(),
            'max': yaw_err.max(),
        }
    
    return errors


def compute_velocity_errors(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute velocity errors (magnitude and components).
    
    Returns: dict with stats for vel_error_m_s and components
    """
    errors = {}
    
    # Velocity magnitude error
    if 'vel_error_m_s' in df.columns:
        vel_err = df['vel_error_m_s'].dropna()
        errors['magnitude_m_s'] = {
            'mean': vel_err.mean(),
            'median': vel_err.median(),
            'std': vel_err.std(),
            'rmse': np.sqrt(np.mean(vel_err**2)),
        }
    
    # Velocity component errors
    for comp in ['E', 'N', 'U']:
        col = f'vel_error_{comp}'
        if col in df.columns:
            err = df[col].dropna()
            errors[f'component_{comp}_m_s'] = {
                'mean': err.mean(),
                'median': err.median(),
                'std': err.std(),
                'rmse': np.sqrt(np.mean(err**2)),
            }
    
    return errors


def compute_segment_statistics(df: pd.DataFrame, segments: List[Tuple[float, float, str]]) -> pd.DataFrame:
    """
    Compute statistics for different flight segments.
    
    Args:
        df: Error log dataframe
        segments: List of (t_start, t_end, description)
    
    Returns: DataFrame with per-segment statistics
    """
    results = []
    
    for t_start, t_end, desc in segments:
        seg = df[(df['t_rel'] >= t_start) & (df['t_rel'] < t_end)]
        
        if len(seg) == 0:
            continue
        
        result = {
            'segment': desc,
            't_start': t_start,
            't_end': t_end,
            'duration_s': t_end - t_start,
            'num_samples': len(seg),
            'pos_error_mean_m': seg['pos_error_m'].mean(),
            'pos_error_median_m': seg['pos_error_m'].median(),
            'pos_error_max_m': seg['pos_error_m'].max(),
            'pos_error_std_m': seg['pos_error_m'].std(),
            'vel_error_mean_m_s': seg['vel_error_m_s'].mean() if 'vel_error_m_s' in seg.columns else np.nan,
            'yaw_error_mean_deg': seg['yaw_error_deg'].mean() if 'yaw_error_deg' in seg.columns else np.nan,
            'yaw_error_std_deg': seg['yaw_error_deg'].std() if 'yaw_error_deg' in seg.columns else np.nan,
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def plot_error_timeline(df: pd.DataFrame, output_path: str):
    """Plot error metrics over time."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Position error
    ax = axes[0]
    ax.plot(df['t_rel'], df['pos_error_m'], 'b-', linewidth=0.5, alpha=0.7)
    ax.set_ylabel('Position Error (m)', fontsize=12)
    ax.set_title('VIO Performance Evaluation: Error Timeline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(df['t_rel'].min(), df['t_rel'].max())
    
    # Add mean line
    mean_pos = df['pos_error_m'].mean()
    ax.axhline(mean_pos, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {mean_pos:.2f}m')
    ax.legend(loc='upper right')
    
    # Velocity error
    ax = axes[1]
    if 'vel_error_m_s' in df.columns:
        ax.plot(df['t_rel'], df['vel_error_m_s'], 'g-', linewidth=0.5, alpha=0.7)
        ax.set_ylabel('Velocity Error (m/s)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(df['t_rel'].min(), df['t_rel'].max())
        
        mean_vel = df['vel_error_m_s'].mean()
        ax.axhline(mean_vel, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {mean_vel:.3f}m/s')
        ax.legend(loc='upper right')
    
    # Yaw error
    ax = axes[2]
    if 'yaw_error_deg' in df.columns:
        ax.plot(df['t_rel'], df['yaw_error_deg'], 'orange', linewidth=0.5, alpha=0.7)
        ax.set_ylabel('Yaw Error (°)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(df['t_rel'].min(), df['t_rel'].max())
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
        
        mean_yaw = df['yaw_error_deg'].mean()
        ax.axhline(mean_yaw, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {mean_yaw:.2f}°')
        ax.legend(loc='upper right')
    
    # Altitude error
    ax = axes[3]
    if 'alt_error_m' in df.columns:
        ax.plot(df['t_rel'], df['alt_error_m'], 'purple', linewidth=0.5, alpha=0.7)
        ax.set_ylabel('Altitude Error (m)', fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(df['t_rel'].min(), df['t_rel'].max())
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
        
        mean_alt = df['alt_error_m'].mean()
        ax.axhline(mean_alt, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {mean_alt:.3f}m')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved error timeline: {output_path}")
    plt.close()


def plot_error_distribution(df: pd.DataFrame, output_path: str):
    """Plot error distributions (histograms)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Position error histogram
    ax = axes[0, 0]
    ax.hist(df['pos_error_m'].dropna(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Position Error (m)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Position Error Distribution\nMean={df["pos_error_m"].mean():.2f}m, Median={df["pos_error_m"].median():.2f}m', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axvline(df['pos_error_m'].mean(), color='r', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(df['pos_error_m'].median(), color='g', linestyle='--', linewidth=2, label='Median')
    ax.legend()
    
    # Velocity error histogram
    ax = axes[0, 1]
    if 'vel_error_m_s' in df.columns:
        ax.hist(df['vel_error_m_s'].dropna(), bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Velocity Error (m/s)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Velocity Error Distribution\nMean={df["vel_error_m_s"].mean():.3f}m/s, Median={df["vel_error_m_s"].median():.3f}m/s', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axvline(df['vel_error_m_s'].mean(), color='r', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(df['vel_error_m_s'].median(), color='g', linestyle='--', linewidth=2, label='Median')
        ax.legend()
    
    # Yaw error histogram
    ax = axes[1, 0]
    if 'yaw_error_deg' in df.columns:
        ax.hist(df['yaw_error_deg'].dropna(), bins=50, color='orange', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Yaw Error (°)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Yaw Error Distribution\nMean={df["yaw_error_deg"].mean():.2f}°, Median={df["yaw_error_deg"].median():.2f}°', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axvline(df['yaw_error_deg'].mean(), color='r', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(df['yaw_error_deg'].median(), color='g', linestyle='--', linewidth=2, label='Median')
        ax.legend()
    
    # Altitude error histogram
    ax = axes[1, 1]
    if 'alt_error_m' in df.columns:
        ax.hist(df['alt_error_m'].dropna(), bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Altitude Error (m)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Altitude Error Distribution\nMean={df["alt_error_m"].mean():.3f}m, Median={df["alt_error_m"].median():.3f}m', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axvline(df['alt_error_m'].mean(), color='r', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(df['alt_error_m'].median(), color='g', linestyle='--', linewidth=2, label='Median')
        ax.legend()
    
    plt.suptitle('VIO Error Distributions', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved error distributions: {output_path}")
    plt.close()


def generate_report(df: pd.DataFrame, output_path: str, segments: List[Tuple[float, float, str]] = None):
    """Generate comprehensive text report."""
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" VIO PERFORMANCE EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Dataset info
        f.write("DATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Total samples:        {len(df)}\n")
        f.write(f"Duration:             {df['t_rel'].max():.2f} seconds ({df['t_rel'].max()/60:.2f} minutes)\n")
        f.write(f"Sample rate:          {len(df)/df['t_rel'].max():.1f} Hz (avg)\n")
        f.write(f"\n")
        
        # Position errors (ATE)
        f.write("POSITION ERRORS (Absolute Trajectory Error)\n")
        f.write("-"*80 + "\n")
        ate = compute_ate(df['pos_error_m'].values)
        f.write(f"RMSE:                 {ate['rmse']:.3f} m\n")
        f.write(f"Mean:                 {ate['mean']:.3f} m\n")
        f.write(f"Median:               {ate['median']:.3f} m\n")
        f.write(f"Std Dev:              {ate['std']:.3f} m\n")
        f.write(f"Min:                  {ate['min']:.3f} m\n")
        f.write(f"Max:                  {ate['max']:.3f} m\n")
        f.write(f"Final:                {df['pos_error_m'].iloc[-1]:.3f} m\n")
        f.write(f"\n")
        
        # Position error components
        f.write("POSITION ERROR COMPONENTS (ENU Frame)\n")
        f.write("-"*80 + "\n")
        for comp in ['E', 'N', 'U']:
            col = f'pos_error_{comp}'
            if col in df.columns:
                err = df[col].dropna()
                f.write(f"{comp} (East/North/Up):\n")
                f.write(f"  Mean:               {err.mean():.3f} m\n")
                f.write(f"  Median:             {err.median():.3f} m\n")
                f.write(f"  RMSE:               {np.sqrt(np.mean(err**2)):.3f} m\n")
                f.write(f"  Std Dev:            {err.std():.3f} m\n")
        f.write(f"\n")
        
        # Drift rate
        f.write("DRIFT ANALYSIS\n")
        f.write("-"*80 + "\n")
        drift = compute_drift_rate(df['pos_error_m'].values, df['t_rel'].values)
        f.write(f"Linear drift rate:    {drift['slope_m_per_s']:.5f} m/s ({drift['slope_m_per_s']*60:.3f} m/min)\n")
        f.write(f"Initial offset:       {drift['intercept_m']:.3f} m\n")
        f.write(f"R-squared:            {drift['r_squared']:.4f}\n")
        f.write(f"\n")
        
        # Relative Trajectory Error
        f.write("RELATIVE TRAJECTORY ERROR (10s window)\n")
        f.write("-"*80 + "\n")
        vio_pos = df[['vio_E', 'vio_N', 'vio_U']].values
        
        # Reconstruct GT positions from VIO + errors
        gt_pos = vio_pos - df[['pos_error_E', 'pos_error_N', 'pos_error_U']].values
        
        rte = compute_rte(vio_pos, gt_pos, df['t_rel'].values, delta_t=10.0)
        f.write(f"Mean RTE:             {rte['mean']:.3f} m\n")
        f.write(f"Median RTE:           {rte['median']:.3f} m\n")
        f.write(f"RMSE RTE:             {rte['rmse']:.3f} m\n")
        f.write(f"Std Dev:              {rte['std']:.3f} m\n")
        f.write(f"\n")
        
        # Velocity errors
        f.write("VELOCITY ERRORS\n")
        f.write("-"*80 + "\n")
        vel_errors = compute_velocity_errors(df)
        if 'magnitude_m_s' in vel_errors:
            stats = vel_errors['magnitude_m_s']
            f.write(f"Magnitude:\n")
            f.write(f"  Mean:               {stats['mean']:.4f} m/s\n")
            f.write(f"  Median:             {stats['median']:.4f} m/s\n")
            f.write(f"  RMSE:               {stats['rmse']:.4f} m/s\n")
            f.write(f"  Std Dev:            {stats['std']:.4f} m/s\n")
        
        for comp in ['E', 'N', 'U']:
            key = f'component_{comp}_m_s'
            if key in vel_errors:
                stats = vel_errors[key]
                f.write(f"{comp} component:\n")
                f.write(f"  Mean:               {stats['mean']:.4f} m/s\n")
                f.write(f"  RMSE:               {stats['rmse']:.4f} m/s\n")
        f.write(f"\n")
        
        # Orientation errors
        f.write("ORIENTATION ERRORS\n")
        f.write("-"*80 + "\n")
        orient_errors = compute_orientation_errors(df)
        if 'yaw_deg' in orient_errors:
            stats = orient_errors['yaw_deg']
            f.write(f"Yaw:\n")
            f.write(f"  Mean:               {stats['mean']:.3f}°\n")
            f.write(f"  Median:             {stats['median']:.3f}°\n")
            f.write(f"  RMSE:               {stats['rmse']:.3f}°\n")
            f.write(f"  Std Dev:            {stats['std']:.3f}°\n")
            f.write(f"  Min:                {stats['min']:.3f}°\n")
            f.write(f"  Max:                {stats['max']:.3f}°\n")
        f.write(f"\n")
        
        # Altitude errors
        f.write("ALTITUDE ERRORS\n")
        f.write("-"*80 + "\n")
        if 'alt_error_m' in df.columns:
            alt_err = df['alt_error_m'].dropna()
            f.write(f"Mean:                 {alt_err.mean():.4f} m\n")
            f.write(f"Median:               {alt_err.median():.4f} m\n")
            f.write(f"RMSE:                 {np.sqrt(np.mean(alt_err**2)):.4f} m\n")
            f.write(f"Std Dev:              {alt_err.std():.4f} m\n")
            f.write(f"Min:                  {alt_err.min():.4f} m\n")
            f.write(f"Max:                  {alt_err.max():.4f} m\n")
            f.write(f"Final:                {df['alt_error_m'].iloc[-1]:.4f} m\n")
        f.write(f"\n")
        
        # Segment analysis
        if segments is not None:
            f.write("SEGMENT ANALYSIS\n")
            f.write("-"*80 + "\n")
            seg_stats = compute_segment_statistics(df, segments)
            f.write(seg_stats.to_string(index=False))
            f.write(f"\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Overall Performance:  ")
        if ate['rmse'] < 5:
            f.write("EXCELLENT (< 5m RMSE)\n")
        elif ate['rmse'] < 15:
            f.write("GOOD (< 15m RMSE)\n")
        elif ate['rmse'] < 30:
            f.write("ACCEPTABLE (< 30m RMSE)\n")
        else:
            f.write("NEEDS IMPROVEMENT (> 30m RMSE)\n")
        
        f.write(f"Heading Accuracy:     ")
        if 'yaw_deg' in orient_errors:
            yaw_rmse = orient_errors['yaw_deg']['rmse']
            if yaw_rmse < 5:
                f.write("EXCELLENT (< 5° RMSE)\n")
            elif yaw_rmse < 10:
                f.write("GOOD (< 10° RMSE)\n")
            elif yaw_rmse < 20:
                f.write("ACCEPTABLE (< 20° RMSE)\n")
            else:
                f.write("NEEDS IMPROVEMENT (> 20° RMSE)\n")
        
        f.write(f"Drift Stability:      ")
        if abs(drift['slope_m_per_s']) < 0.01:
            f.write("EXCELLENT (< 0.01 m/s)\n")
        elif abs(drift['slope_m_per_s']) < 0.05:
            f.write("GOOD (< 0.05 m/s)\n")
        elif abs(drift['slope_m_per_s']) < 0.1:
            f.write("ACCEPTABLE (< 0.1 m/s)\n")
        else:
            f.write("NEEDS IMPROVEMENT (> 0.1 m/s)\n")
        
        f.write(f"\n")
        f.write("="*80 + "\n")
    
    print(f"[Report] Saved evaluation report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VIO performance against ground truth')
    parser.add_argument('--error_log', type=str, default='out_vio_imu_ekf/error_log.csv',
                       help='Path to VIO error log CSV')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for reports and plots')
    parser.add_argument('--gps_gt', type=str, default='flight_log_from_gga.csv',
                       help='Path to GPS ground truth CSV (optional)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("VIO PERFORMANCE EVALUATION")
    print("="*80)
    
    # Load VIO results
    print(f"\n[Loading] VIO error log: {args.error_log}")
    df = load_vio_results(args.error_log)
    print(f"[Loaded] {len(df)} samples, duration: {df['t_rel'].max():.2f}s")
    
    # Define flight segments (based on ground truth from video)
    segments = [
        (0, 80, "Hovering"),
        (80, 100, "180° Turn + Takeoff"),
        (100, 200, "Straight Flight 1"),
        (200, 250, "Right Turn"),
        (250, 350, "Straight + U-Turn"),
        (350, 450, "Final Turn + Return"),
    ]
    
    # Generate report
    print("\n[Generating] Evaluation report...")
    report_path = os.path.join(args.output_dir, 'performance_report.txt')
    generate_report(df, report_path, segments)
    
    # Plot error timeline
    print("[Plotting] Error timeline...")
    timeline_path = os.path.join(args.output_dir, 'error_timeline.png')
    plot_error_timeline(df, timeline_path)
    
    # Plot error distributions
    print("[Plotting] Error distributions...")
    dist_path = os.path.join(args.output_dir, 'error_distributions.png')
    plot_error_distribution(df, dist_path)
    
    # Print summary to console
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    ate = compute_ate(df['pos_error_m'].values)
    print(f"Position Error (ATE):")
    print(f"  RMSE:     {ate['rmse']:.3f} m")
    print(f"  Mean:     {ate['mean']:.3f} m")
    print(f"  Median:   {ate['median']:.3f} m")
    print(f"  Final:    {df['pos_error_m'].iloc[-1]:.3f} m")
    
    if 'yaw_error_deg' in df.columns:
        yaw_err = df['yaw_error_deg'].dropna()
        print(f"\nYaw Error:")
        print(f"  Mean:     {yaw_err.mean():.2f}°")
        print(f"  RMSE:     {np.sqrt(np.mean(yaw_err**2)):.2f}°")
        print(f"  Std Dev:  {yaw_err.std():.2f}°")
    
    drift = compute_drift_rate(df['pos_error_m'].values, df['t_rel'].values)
    print(f"\nDrift Rate:")
    print(f"  {drift['slope_m_per_s']:.5f} m/s ({drift['slope_m_per_s']*60:.3f} m/min)")
    
    print("\n" + "="*80)
    print(f"Results saved to: {args.output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
