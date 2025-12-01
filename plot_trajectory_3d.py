#!/usr/bin/env python3
"""
สคริปต์สำหรับวาด 3D trajectory จากไฟล์ pose.csv
Script for visualizing 3D trajectory from pose.csv file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path


def load_trajectory(csv_path):
    """โหลดข้อมูล trajectory จากไฟล์ CSV"""
    print(f"Loading trajectory from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} poses")
    print(f"Time range: {df['Timestamp(s)'].min():.2f}s to {df['Timestamp(s)'].max():.2f}s")
    return df


def plot_trajectory_3d(df, output_path=None, coordinate_system='local', subsample=1):
    """
    วาด 3D trajectory
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame ที่มีข้อมูล trajectory
    output_path : str or None
        ถ้าระบุจะบันทึกภาพ
    coordinate_system : str
        'local' = ใช้ PX, PY, PZ_MSL
        'geo' = ใช้ lat, lon, AGL
    subsample : int
        แสดงทุก ๆ n จุด (เพื่อประหยัดหน่วยความจำ)
    """
    # Subsample ข้อมูลถ้าต้องการ
    df_plot = df.iloc[::subsample].copy()
    
    # เตรียมข้อมูล
    if coordinate_system == 'local':
        x = df_plot['PX'].values
        y = df_plot['PY'].values
        z = df_plot['PZ_MSL'].values
        xlabel, ylabel, zlabel = 'X (m)', 'Y (m)', 'Z MSL (m)'
        title = '3D Trajectory - Local Coordinates'
    else:  # geo
        x = df_plot['lon'].values
        y = df_plot['lat'].values
        z = df_plot['AGL(m)'].values
        xlabel, ylabel, zlabel = 'Longitude', 'Latitude', 'AGL (m)'
        title = '3D Trajectory - Geographic Coordinates'
    
    # คำนวณความเร็ว (สำหรับ color mapping)
    vx = df_plot['VX'].values
    vy = df_plot['VY'].values
    vz = df_plot['VZ'].values
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # สร้าง figure
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: 3D trajectory with speed colormap
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    scatter = ax1.scatter(x, y, z, c=speed, cmap='jet', s=1, alpha=0.6)
    ax1.plot(x, y, z, 'b-', linewidth=0.5, alpha=0.3)
    ax1.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
    ax1.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='X', label='End')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_zlabel(zlabel)
    ax1.set_title(title + ' (colored by speed)')
    ax1.legend()
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.8)
    cbar.set_label('Speed (m/s)', rotation=270, labelpad=20)
    
    # Plot 2: Top view (X-Y)
    ax2 = fig.add_subplot(2, 2, 2)
    scatter2 = ax2.scatter(x, y, c=z, cmap='terrain', s=1, alpha=0.6)
    ax2.plot(x, y, 'b-', linewidth=0.5, alpha=0.3)
    ax2.scatter(x[0], y[0], c='green', s=100, marker='o', label='Start')
    ax2.scatter(x[-1], y[-1], c='red', s=100, marker='X', label='End')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title('Top View (colored by altitude)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('Altitude (m)', rotation=270, labelpad=20)
    
    # Plot 3: Side view (X-Z)
    ax3 = fig.add_subplot(2, 2, 3)
    distance = np.sqrt(x**2 + y**2)
    ax3.plot(distance, z, 'b-', linewidth=1)
    ax3.scatter(distance[0], z[0], c='green', s=100, marker='o', label='Start')
    ax3.scatter(distance[-1], z[-1], c='red', s=100, marker='X', label='End')
    ax3.set_xlabel('Distance from origin (m)')
    ax3.set_ylabel(zlabel)
    ax3.set_title('Side View - Altitude Profile')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # คำนวณสถิติ
    total_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
    max_altitude = np.max(z)
    min_altitude = np.min(z)
    avg_speed = np.mean(speed)
    max_speed = np.max(speed)
    duration = df_plot['Timestamp(s)'].iloc[-1] - df_plot['Timestamp(s)'].iloc[0]
    
    stats_text = f"""
    Trajectory Statistics
    {'='*40}
    
    Duration: {duration:.2f} seconds ({duration/60:.2f} min)
    Total points: {len(df):,} (showing {len(df_plot):,})
    
    Distance:
      Total distance: {total_distance:.2f} m ({total_distance/1000:.2f} km)
      Horizontal distance: {np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2):.2f} m
    
    Altitude:
      Max: {max_altitude:.2f} m
      Min: {min_altitude:.2f} m
      Range: {max_altitude - min_altitude:.2f} m
      Start: {z[0]:.2f} m
      End: {z[-1]:.2f} m
    
    Speed:
      Average: {avg_speed:.2f} m/s ({avg_speed*3.6:.2f} km/h)
      Maximum: {max_speed:.2f} m/s ({max_speed*3.6:.2f} km/h)
    
    Coordinate Range:
      X: [{np.min(x):.2f}, {np.max(x):.2f}]
      Y: [{np.min(y):.2f}, {np.max(y):.2f}]
      Z: [{np.min(z):.2f}, {np.max(z):.2f}]
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    if output_path:
        print(f"Saving plot to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved successfully!")
    
    plt.show()


def plot_velocity_time(df, output_path=None):
    """วาดกราฟความเร็วตามเวลา"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    time = df['Timestamp(s)'].values
    vx = df['VX'].values
    vy = df['VY'].values
    vz = df['VZ'].values
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # Velocity components
    axes[0].plot(time, vx, label='VX', linewidth=0.5)
    axes[0].plot(time, vy, label='VY', linewidth=0.5)
    axes[0].plot(time, vz, label='VZ', linewidth=0.5)
    axes[0].set_ylabel('Velocity (m/s)')
    axes[0].set_title('Velocity Components over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Total speed
    axes[1].plot(time, speed, 'b-', linewidth=0.8)
    axes[1].set_ylabel('Speed (m/s)')
    axes[1].set_title('Total Speed over Time')
    axes[1].grid(True, alpha=0.3)
    
    # Altitude
    axes[2].plot(time, df['AGL(m)'].values, 'g-', linewidth=0.8)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Altitude AGL (m)')
    axes[2].set_title('Altitude over Time')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Velocity plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot 3D trajectory from pose.csv file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot trajectory from default location
  python plot_trajectory_3d.py
  
  # Plot with custom input file
  python plot_trajectory_3d.py -i path/to/pose.csv
  
  # Plot using geographic coordinates
  python plot_trajectory_3d.py --coord geo
  
  # Subsample data (show every 10th point)
  python plot_trajectory_3d.py --subsample 10
  
  # Save plots to files
  python plot_trajectory_3d.py -o trajectory.png -ov velocity.png
        """
    )
    
    parser.add_argument('-i', '--input', type=str,
                       default='out_vio_msckf_test/pose.csv',
                       help='Path to pose.csv file (default: out_vio_imu_ekf/pose.csv)')
    parser.add_argument('-o', '--output', type=str,
                       help='Output path for 3D trajectory plot (optional)')
    parser.add_argument('-ov', '--output-velocity', type=str,
                       help='Output path for velocity plot (optional)')
    parser.add_argument('--coord', type=str, choices=['local', 'geo'],
                       default='local',
                       help='Coordinate system: local (PX,PY,PZ) or geo (lat,lon,AGL)')
    parser.add_argument('--subsample', type=int, default=1,
                       help='Subsample rate (show every Nth point, default: 1 = all points)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (only save to file)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Load trajectory data
    df = load_trajectory(input_path)
    
    # Plot 3D trajectory
    print(f"\nPlotting 3D trajectory...")
    plot_trajectory_3d(df, args.output, args.coord, args.subsample)
    
    # Plot velocity
    if args.output_velocity or not args.no_show:
        print(f"\nPlotting velocity profile...")
        plot_velocity_time(df, args.output_velocity)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
