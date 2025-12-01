#!/usr/bin/env python3
"""
สคริปต์สำหรับวาด 3D trajectory แบบ interactive ด้วย Plotly
Interactive 3D trajectory visualization using Plotly
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from pathlib import Path


def load_trajectory(csv_path):
    """โหลดข้อมูล trajectory จากไฟล์ CSV"""
    print(f"Loading trajectory from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} poses")
    print(f"Time range: {df['Timestamp(s)'].min():.2f}s to {df['Timestamp(s)'].max():.2f}s")
    return df


def create_interactive_3d_plot(df, coordinate_system='local', subsample=1):
    """
    สร้าง interactive 3D trajectory plot
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame ที่มีข้อมูล trajectory
    coordinate_system : str
        'local' = ใช้ PX, PY, PZ_MSL
        'geo' = ใช้ lat, lon, AGL
    subsample : int
        แสดงทุก ๆ n จุด
    """
    # Subsample ข้อมูล
    df_plot = df.iloc[::subsample].copy()
    
    # เตรียมข้อมูล
    if coordinate_system == 'local':
        x = df_plot['PX'].values
        y = df_plot['PY'].values
        z = df_plot['PZ_MSL'].values
        xlabel, ylabel, zlabel = 'X (m)', 'Y (m)', 'Z MSL (m)'
    else:  # geo
        x = df_plot['lon'].values
        y = df_plot['lat'].values
        z = df_plot['AGL(m)'].values
        xlabel, ylabel, zlabel = 'Longitude', 'Latitude', 'AGL (m)'
    
    # คำนวณความเร็ว
    vx = df_plot['VX'].values
    vy = df_plot['VY'].values
    vz = df_plot['VZ'].values
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    speed_kmh = speed * 3.6
    
    # คำนวณเวลา
    time = df_plot['Timestamp(s)'].values
    
    # สร้าง hover text
    hover_text = [
        f"Time: {t:.2f}s<br>" +
        f"{xlabel}: {xi:.2f}<br>" +
        f"{ylabel}: {yi:.2f}<br>" +
        f"{zlabel}: {zi:.2f}<br>" +
        f"Speed: {s:.2f} m/s ({skm:.2f} km/h)"
        for t, xi, yi, zi, s, skm in zip(time, x, y, z, speed, speed_kmh)
    ]
    
    # สร้าง figure
    fig = go.Figure()
    
    # Add trajectory line
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color=speed_kmh, colorscale='Jet', width=3,
                 colorbar=dict(title="Speed<br>(km/h)", x=1.15)),
        name='Trajectory',
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    # Add start point
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='markers',
        marker=dict(size=10, color='green', symbol='circle'),
        name='Start',
        hovertext=f"START<br>Time: {time[0]:.2f}s",
        hoverinfo='text'
    ))
    
    # Add end point
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='End',
        hovertext=f"END<br>Time: {time[-1]:.2f}s",
        hoverinfo='text'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'3D Trajectory - {coordinate_system.upper()} Coordinates<br>' +
              f'<sub>Duration: {time[-1]-time[0]:.1f}s | Points: {len(df_plot):,} / {len(df):,}</sub>',
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel,
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1200,
        height=800,
        showlegend=True
    )
    
    return fig


def create_velocity_plot(df):
    """สร้างกราฟความเร็วและความสูงตามเวลา"""
    time = df['Timestamp(s)'].values
    vx = df['VX'].values
    vy = df['VY'].values
    vz = df['VZ'].values
    speed = np.sqrt(vx**2 + vy**2 + vz**2) * 3.6  # km/h
    altitude = df['AGL(m)'].values
    
    # สร้าง subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Speed over Time', 'Altitude over Time'),
        vertical_spacing=0.12
    )
    
    # Speed plot
    fig.add_trace(
        go.Scatter(x=time, y=speed, mode='lines', name='Speed',
                  line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # Altitude plot
    fig.add_trace(
        go.Scatter(x=time, y=altitude, mode='lines', name='Altitude',
                  line=dict(color='green', width=1)),
        row=2, col=1
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_yaxes(title_text="Altitude AGL (m)", row=2, col=1)
    
    fig.update_layout(
        height=800,
        width=1200,
        title_text="Velocity and Altitude Profile",
        showlegend=False
    )
    
    return fig


def print_statistics(df):
    """แสดงสถิติของ trajectory"""
    x = df['PX'].values
    y = df['PY'].values
    z = df['PZ_MSL'].values
    
    vx = df['VX'].values
    vy = df['VY'].values
    vz = df['VZ'].values
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    
    total_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
    duration = df['Timestamp(s)'].iloc[-1] - df['Timestamp(s)'].iloc[0]
    
    print("\n" + "="*60)
    print("TRAJECTORY STATISTICS")
    print("="*60)
    print(f"Duration:          {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Total points:      {len(df):,}")
    print(f"\nDistance:")
    print(f"  Total distance:  {total_distance:.2f} m ({total_distance/1000:.2f} km)")
    print(f"  Horizontal dist: {np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2):.2f} m")
    print(f"\nAltitude:")
    print(f"  Maximum:         {np.max(z):.2f} m")
    print(f"  Minimum:         {np.min(z):.2f} m")
    print(f"  Range:           {np.max(z) - np.min(z):.2f} m")
    print(f"  Start:           {z[0]:.2f} m")
    print(f"  End:             {z[-1]:.2f} m")
    print(f"\nSpeed:")
    print(f"  Average:         {np.mean(speed):.2f} m/s ({np.mean(speed)*3.6:.2f} km/h)")
    print(f"  Maximum:         {np.max(speed):.2f} m/s ({np.max(speed)*3.6:.2f} km/h)")
    print(f"  Minimum:         {np.min(speed):.2f} m/s ({np.min(speed)*3.6:.2f} km/h)")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive 3D trajectory visualization from pose.csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default plot
  python plot_trajectory_interactive.py
  
  # Custom input file
  python plot_trajectory_interactive.py -i path/to/pose.csv
  
  # Geographic coordinates
  python plot_trajectory_interactive.py --coord geo
  
  # Subsample for large datasets
  python plot_trajectory_interactive.py --subsample 10
  
  # Save to HTML
  python plot_trajectory_interactive.py -o trajectory.html
        """
    )
    
    parser.add_argument('-i', '--input', type=str,
                       default='out_vio_imu_ekf/pose.csv',
                       help='Path to pose.csv file')
    parser.add_argument('-o', '--output', type=str,
                       help='Output HTML file path (optional)')
    parser.add_argument('--coord', type=str, choices=['local', 'geo'],
                       default='local',
                       help='Coordinate system: local or geo')
    parser.add_argument('--subsample', type=int, default=1,
                       help='Subsample rate (default: 1 = all points)')
    parser.add_argument('--velocity', action='store_true',
                       help='Also show velocity plot')
    parser.add_argument('--stats', action='store_true',
                       help='Print statistics only (no plots)')
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return
    
    df = load_trajectory(input_path)
    
    # Print statistics
    print_statistics(df)
    
    if args.stats:
        return
    
    # Create 3D plot
    print("Creating interactive 3D plot...")
    fig_3d = create_interactive_3d_plot(df, args.coord, args.subsample)
    
    if args.output:
        output_path = Path(args.output)
        print(f"Saving 3D plot to: {output_path}")
        fig_3d.write_html(str(output_path))
        print("Saved successfully!")
    else:
        fig_3d.show()
    
    # Create velocity plot if requested
    if args.velocity:
        print("Creating velocity plot...")
        fig_vel = create_velocity_plot(df)
        
        if args.output:
            vel_path = output_path.with_stem(output_path.stem + '_velocity')
            print(f"Saving velocity plot to: {vel_path}")
            fig_vel.write_html(str(vel_path))
        else:
            fig_vel.show()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
