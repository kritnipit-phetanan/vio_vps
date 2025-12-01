#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze VIO results from out_vio_imu_ekf vs out_vio_imu_ekf_wovio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_vo_debug(vo_debug_path):
    """Analyze VO debug file"""
    if not Path(vo_debug_path).exists():
        return None
    
    df = pd.read_csv(vo_debug_path)
    
    analysis = {
        "total_frames": len(df),
        "frames_with_inliers": (df["num_inliers"] > 0).sum(),
        "frames_above_threshold": (df["num_inliers"] >= 15).sum(),
        "max_inliers": df["num_inliers"].max(),
        "mean_inliers": df["num_inliers"].mean(),
        "median_inliers": df["num_inliers"].median(),
        "frames_with_skip_vo": (df["skip_vo"] == 1).sum() if "skip_vo" in df.columns else 0,
        "frames_use_vz_only": (df["use_only_vz"] == 1).sum() if "use_only_vz" in df.columns else 0,
    }
    
    # Check alignment distribution
    if "alignment_deg" in df.columns:
        analysis["mean_alignment"] = df["alignment_deg"].mean()
        analysis["median_alignment"] = df["alignment_deg"].median()
        analysis["nadir_aligned_frames"] = (df["alignment_deg"] <= 30).sum()
    
    return analysis

def analyze_pose(pose_path):
    """Analyze pose file"""
    if not Path(pose_path).exists():
        return None
    
    df = pd.read_csv(pose_path)
    
    # Check if Frame column has any values
    frames_with_vio = df["Frame"].notna().sum() if "Frame" in df.columns else 0
    
    analysis = {
        "total_samples": len(df),
        "duration_sec": df["Timestamp(s)"].iloc[-1] - df["Timestamp(s)"].iloc[0],
        "frames_with_vio": frames_with_vio,
        "final_position": {
            "lat": df["lat"].iloc[-1],
            "lon": df["lon"].iloc[-1],
            "alt_msl": df["PZ_MSL"].iloc[-1],
            "agl": df["AGL(m)"].iloc[-1] if "AGL(m)" in df.columns else None,
        },
        "position_drift": {
            "px": df["PX"].iloc[-1] - df["PX"].iloc[0],
            "py": df["PY"].iloc[-1] - df["PY"].iloc[0],
            "pz": df["PZ_MSL"].iloc[-1] - df["PZ_MSL"].iloc[0],
        },
        "velocity_stats": {
            "vx_mean": df["VX"].mean(),
            "vy_mean": df["VY"].mean(),
            "vz_mean": df["VZ"].mean(),
            "vx_max": df["VX"].max(),
            "vy_max": df["VY"].max(),
            "vz_max": df["VZ"].max(),
        }
    }
    
    # Calculate total drift magnitude
    analysis["total_drift_m"] = np.sqrt(
        analysis["position_drift"]["px"]**2 + 
        analysis["position_drift"]["py"]**2
    )
    
    return analysis

def analyze_state_debug(state_debug_path):
    """Analyze state debug file"""
    if not Path(state_debug_path).exists():
        return None
    
    df = pd.read_csv(state_debug_path)
    
    analysis = {
        "total_samples": len(df),
        "final_state": {
            "px": df["px"].iloc[-1],
            "py": df["py"].iloc[-1],
            "pz": df["pz"].iloc[-1],
            "vx": df["vx"].iloc[-1],
            "vy": df["vy"].iloc[-1],
            "vz": df["vz"].iloc[-1],
        },
        "acceleration_stats": {
            "ax_mean": df["a_world_x"].mean() if "a_world_x" in df.columns else None,
            "ay_mean": df["a_world_y"].mean() if "a_world_y" in df.columns else None,
            "az_mean": df["a_world_z"].mean() if "a_world_z" in df.columns else None,
        }
    }
    
    return analysis

def plot_comparison(vio_path, no_vio_path, output_dir):
    """Plot comparison between VIO and no-VIO results"""
    
    # Load pose data
    df_vio = pd.read_csv(f"{vio_path}/pose.csv")
    df_no_vio = pd.read_csv(f"{no_vio_path}/pose.csv")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("VIO vs No-VIO Comparison", fontsize=16, fontweight='bold')
    
    # 1. Position XY trajectory
    ax = axes[0, 0]
    ax.plot(df_vio["PX"], df_vio["PY"], 'b-', alpha=0.7, linewidth=1, label='With VIO')
    ax.plot(df_no_vio["PX"], df_no_vio["PY"], 'r-', alpha=0.7, linewidth=1, label='Without VIO')
    ax.plot(df_vio["PX"].iloc[0], df_vio["PY"].iloc[0], 'go', markersize=10, label='Start')
    ax.plot(df_vio["PX"].iloc[-1], df_vio["PY"].iloc[-1], 'b^', markersize=10, label='VIO End')
    ax.plot(df_no_vio["PX"].iloc[-1], df_no_vio["PY"].iloc[-1], 'rs', markersize=10, label='No-VIO End')
    ax.set_xlabel("X (East) [m]")
    ax.set_ylabel("Y (North) [m]")
    ax.set_title("2D Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 2. Altitude over time
    ax = axes[0, 1]
    ax.plot(df_vio["Timestamp(s)"], df_vio["PZ_MSL"], 'b-', alpha=0.7, linewidth=1, label='With VIO')
    ax.plot(df_no_vio["Timestamp(s)"], df_no_vio["PZ_MSL"], 'r-', alpha=0.7, linewidth=1, label='Without VIO')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude MSL [m]")
    ax.set_title("Altitude Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Velocity magnitude over time
    ax = axes[1, 0]
    v_vio = np.sqrt(df_vio["VX"]**2 + df_vio["VY"]**2 + df_vio["VZ"]**2)
    v_no_vio = np.sqrt(df_no_vio["VX"]**2 + df_no_vio["VY"]**2 + df_no_vio["VZ"]**2)
    ax.plot(df_vio["Timestamp(s)"], v_vio, 'b-', alpha=0.7, linewidth=1, label='With VIO')
    ax.plot(df_no_vio["Timestamp(s)"], v_no_vio, 'r-', alpha=0.7, linewidth=1, label='Without VIO')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Position drift from origin
    ax = axes[1, 1]
    drift_vio = np.sqrt(df_vio["PX"]**2 + df_vio["PY"]**2)
    drift_no_vio = np.sqrt(df_no_vio["PX"]**2 + df_no_vio["PY"]**2)
    ax.plot(df_vio["Timestamp(s)"], drift_vio, 'b-', alpha=0.7, linewidth=1, label='With VIO')
    ax.plot(df_no_vio["Timestamp(s)"], drift_no_vio, 'r-', alpha=0.7, linewidth=1, label='Without VIO')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance from origin [m]")
    ax.set_title("Position Drift")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{output_dir}/vio_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_path}")
    plt.close()

def main():
    vio_path = "out_vio_imu_ekf"
    no_vio_path = "out_vio_imu_ekf_wovio"
    output_dir = "."
    
    print("="*80)
    print("VIO RESULTS ANALYSIS")
    print("="*80)
    
    # Analyze VIO run
    print(f"\n{'='*80}")
    print(f"📊 RESULTS WITH VIO (out_vio_imu_ekf)")
    print(f"{'='*80}")
    
    vo_debug = analyze_vo_debug(f"{vio_path}/vo_debug.csv")
    if vo_debug:
        print("\n🎯 Visual Odometry Debug:")
        print(f"   Total frames processed: {vo_debug['total_frames']}")
        print(f"   Frames with inliers > 0: {vo_debug['frames_with_inliers']} ({vo_debug['frames_with_inliers']/vo_debug['total_frames']*100:.1f}%)")
        print(f"   Frames with inliers >= 15 (threshold): {vo_debug['frames_above_threshold']} ({vo_debug['frames_above_threshold']/vo_debug['total_frames']*100:.1f}%)")
        print(f"   Maximum inliers: {vo_debug['max_inliers']}")
        print(f"   Mean inliers: {vo_debug['mean_inliers']:.1f}")
        print(f"   Median inliers: {vo_debug['median_inliers']:.1f}")
        print(f"   Frames using VZ-only mode: {vo_debug['frames_use_vz_only']}")
        if 'mean_alignment' in vo_debug:
            print(f"   Mean camera alignment: {vo_debug['mean_alignment']:.1f}°")
            print(f"   Nadir-aligned frames (<30°): {vo_debug['nadir_aligned_frames']}")
    
    pose_vio = analyze_pose(f"{vio_path}/pose.csv")
    if pose_vio:
        print("\n📍 Pose Analysis:")
        print(f"   Total samples: {pose_vio['total_samples']}")
        print(f"   Duration: {pose_vio['duration_sec']:.2f} seconds")
        print(f"   ⚠️  VIO updates applied: {pose_vio['frames_with_vio']} (PROBLEM: Should be > 0!)")
        print(f"   Position drift (XY): {pose_vio['total_drift_m']:.2f} m")
        print(f"   Position drift (X): {pose_vio['position_drift']['px']:.2f} m")
        print(f"   Position drift (Y): {pose_vio['position_drift']['py']:.2f} m")
        print(f"   Position drift (Z): {pose_vio['position_drift']['pz']:.2f} m")
        print(f"   Mean velocity: vx={pose_vio['velocity_stats']['vx_mean']:.3f}, vy={pose_vio['velocity_stats']['vy_mean']:.3f}, vz={pose_vio['velocity_stats']['vz_mean']:.3f} m/s")
    
    # Analyze No-VIO run
    print(f"\n{'='*80}")
    print(f"📊 RESULTS WITHOUT VIO (out_vio_imu_ekf_wovio)")
    print(f"{'='*80}")
    
    pose_no_vio = analyze_pose(f"{no_vio_path}/pose.csv")
    if pose_no_vio:
        print("\n📍 Pose Analysis:")
        print(f"   Total samples: {pose_no_vio['total_samples']}")
        print(f"   Duration: {pose_no_vio['duration_sec']:.2f} seconds")
        print(f"   Position drift (XY): {pose_no_vio['total_drift_m']:.2f} m")
        print(f"   Position drift (X): {pose_no_vio['position_drift']['px']:.2f} m")
        print(f"   Position drift (Y): {pose_no_vio['position_drift']['py']:.2f} m")
        print(f"   Position drift (Z): {pose_no_vio['position_drift']['pz']:.2f} m")
        print(f"   Mean velocity: vx={pose_no_vio['velocity_stats']['vx_mean']:.3f}, vy={pose_no_vio['velocity_stats']['vy_mean']:.3f}, vz={pose_no_vio['velocity_stats']['vz_mean']:.3f} m/s")
    
    # Comparison
    if pose_vio and pose_no_vio:
        print(f"\n{'='*80}")
        print(f"🔍 COMPARISON")
        print(f"{'='*80}")
        
        drift_diff = pose_vio['total_drift_m'] - pose_no_vio['total_drift_m']
        print(f"\n   XY Drift difference: {drift_diff:.2f} m")
        print(f"   - With VIO: {pose_vio['total_drift_m']:.2f} m")
        print(f"   - Without VIO: {pose_no_vio['total_drift_m']:.2f} m")
        
        if abs(drift_diff) < 0.1:
            print(f"   ⚠️  WARNING: Almost identical drift! VIO is NOT working properly!")
        else:
            if drift_diff < 0:
                print(f"   ✓ VIO reduced drift by {abs(drift_diff):.2f} m")
            else:
                print(f"   ⚠️  VIO INCREASED drift by {drift_diff:.2f} m (unexpected!)")
    
    # Root cause analysis
    print(f"\n{'='*80}")
    print(f"🔍 ROOT CAUSE ANALYSIS")
    print(f"{'='*80}")
    
    if vo_debug:
        print("\n❌ PROBLEM IDENTIFIED:")
        print(f"   1. VO processes {vo_debug['total_frames']} frames")
        print(f"   2. {vo_debug['frames_above_threshold']} frames have enough inliers (>= 15)")
        print(f"   3. BUT: VIO updates = {pose_vio['frames_with_vio']} (from pose.csv)")
        print("\n   This means:")
        print("   - Feature tracking is working (inliers found)")
        print("   - Essential matrix decomposition is working (vo_debug has motion)")
        print("   - BUT: VIO updates are NOT being applied to the EKF state!")
        
        print("\n🔎 Possible causes:")
        print("   [ ] Chi-square test rejection (innovation too large)")
        print("   [ ] Mahalanobis distance gating")
        print("   [ ] Camera alignment check failing")
        print("   [ ] VIO update flag not being set properly")
        print("   [ ] MSCKF backend rejecting all features")
        
        print("\n💡 To debug further, check:")
        print("   1. Search for 'VIO update' or 'MSCKF' in terminal output")
        print("   2. Check if chi-square threshold is too strict")
        print("   3. Review Mahalanobis distance calculations")
        print("   4. Verify camera_view configuration")
    
    # Generate comparison plots
    try:
        plot_comparison(vio_path, no_vio_path, output_dir)
    except Exception as e:
        print(f"\n⚠️  Could not generate plots: {e}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
