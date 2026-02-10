
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import calendar

# Attempt to import pyproj logic if available, otherwise use simple fallback
try:
    from pyproj import CRS, Transformer
    def latlon_to_xy(lat, lon, lat0, lon0):
        crs_wgs84 = CRS.from_epsg(4326)
        crs_local = CRS.from_proj4(f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        transformer = Transformer.from_crs(crs_wgs84, crs_local, always_xy=True)
        return transformer.transform(lon, lat)
except ImportError:
    print("Warning: pyproj not found. Using simple equirectangular projection.")
    def latlon_to_xy(lat, lon, lat0, lon0):
        R = 6378137.0  # Earth radius
        dlat = np.radians(lat - lat0)
        dlon = np.radians(lon - lon0)
        x = R * dlon * np.cos(np.radians(lat0))
        y = R * dlat
        return x, y

def load_ppk_trajectory(path: str) -> pd.DataFrame:
    """Load Ground Truth .pos file with GPST timestamp parsing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"GT file not found: {path}")
    
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            # Timestamp: YYYY/MM/DD HH:MM:SS.SSS (GPST)
            gpst_str = f"{parts[0]} {parts[1]}"
            try:
                dt = datetime.strptime(gpst_str, "%Y/%m/%d %H:%M:%S.%f")
                # Convert to Unix timestamp
                # Note: This is technically GPST -> Unix UTC conversion without leap seconds correction
                # but aligns with how VIO loader does it currently.
                stamp = calendar.timegm(dt.timetuple()) + dt.microsecond / 1e6
                
                rows.append({
                    'timestamp': stamp,
                    'lat': float(parts[2]),
                    'lon': float(parts[3]),
                    'height': float(parts[4])  # Ellipsoidal height
                })
            except Exception:
                continue
                
    if not rows:
        raise ValueError("No valid data found in GT file")
        
    df = pd.DataFrame(rows)
    print(f"[GT] Loaded {len(df)} points from {os.path.basename(path)}")
    return df

def load_vio_pose(path: str) -> pd.DataFrame:
    """Load VIO estimated pose.csv."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"VIO pose file not found: {path}")
    
    # Check headers to determine delimiter and column names
    # pose.csv format: Timestamp(s),dt,Frame,PX,PY,PZ_MSL,VX,VY,VZ,lat,lon,AGL(m),...
    try:
        df = pd.read_csv(path)
        # Rename column for consistency
        df = df.rename(columns={'Timestamp(s)': 'timestamp', 'PZ_MSL': 'height'})
        
        # Adjust timestamp: pose.csv stores relative time (t - t0) in first column usually?
        # Actually in main_loop.py: f"{t - self.state.t0:.6f}..."
        # But wait, we need ABSOLUTE timestamp to align with GT.
        # Checking pose.csv columns... usually the logic inside main_loop writes relative time.
        # But `main_loop.py` line 1555: `t - self.state.t0`.
        # This makes alignment hard unless we know t0.
        
        # Let's check how error_log.csv stores time. error_log has absolute time 't'.
        # If pose.csv only has relative time, we might need to deduce absolute time.
        # However, checking the user's previous `head pose.csv` output:
        # Timestamp(s) starts at 0.000000.
        
        # We need the START absolute time.
        # Usually benchmark script or config logs the start time.
        # HACK: Use the first timestamp from error_log.csv or flight_log if possible?
        # Or better: Assume the user provides t0 or we roughly align by start.
        
        # WAIT! If the user wants to compare GT vs Estimate, and Estimate is relative...
        # We should align them.
        # Let's try to find absolute timestamp source.
        # The 'lat' and 'lon' columns in pose.csv are computed from VIO state.
        # We can align using those?
        
        # NOTE: If pose.csv has 'lat', 'lon', we can plot those directly vs GT lat/lon
        # without worrying about exact time alignment for just the TRAJECTORY shape.
        pass
    except Exception as e:
        print(f"Error loading VIO pose: {e}")
        return None
        
    print(f"[VIO] Loaded {len(df)} points from {os.path.basename(path)}")
    return df

def plot_trajectories(gt_df, vio_df, title="Trajectory Comparison"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- Ground Truth Projection ---
    # Set Origin to first GT point
    lat0, lon0 = gt_df.iloc[0]['lat'], gt_df.iloc[0]['lon']
    alt0 = gt_df.iloc[0]['height']
    
    print(f"Origin (GT start): Lat={lat0:.6f}, Lon={lon0:.6f}, Alt={alt0:.2f}m")
    
    gt_x, gt_y = latlon_to_xy(gt_df['lat'].values, gt_df['lon'].values, lat0, lon0)
    gt_z = gt_df['height'].values - alt0
    
    ax.plot(gt_x, gt_y, gt_z, label='Ground Truth (PPK)', color='black', alpha=0.7, linewidth=1.5)
    
    # --- VIO Estimate Projection ---
    # We have 'PX', 'PY', 'PZ_MSL' in pose.csv which are local frame positions.
    # Usually PX, PY are already ENU relative to start.
    # PZ_MSL is absolute MSL.
    
    # If VIO started near GT start, PX/PY should align roughly.
    vio_x = vio_df['PX'].values
    vio_y = vio_df['PY'].values
    
    # For Z, VIO PX/PY are local, but PZ_MSL is absolute (MSL).
    # We should align Z to start at 0 for comparison with GT relative Z,
    # OR compare absolute Z if we knew geoid offset.
    # Let's plot relative Z for trajectory shape comparison.
    vio_z = vio_df['height'].values - vio_df.iloc[0]['height']
    
    # Align VIO to GT origin if needed?
    # Assume VIO initialization set (0,0) to start position.
    
    ax.plot(vio_x, vio_y, vio_z, label='VIO Estimate', color='red', linewidth=2)
    
    # Add start/end markers
    ax.scatter(gt_x[0], gt_y[0], gt_z[0], c='g', marker='^', s=100, label='Start')
    ax.scatter(gt_x[-1], gt_y[-1], gt_z[-1], c='r', marker='x', s=100, label='End')
    
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([gt_x.max()-gt_x.min(), gt_y.max()-gt_y.min(), gt_z.max()-gt_z.min()]).max()
    mid_x = (gt_x.max()+gt_x.min()) * 0.5
    mid_y = (gt_y.max()+gt_y.min()) * 0.5
    mid_z = (gt_z.max()+gt_z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
    
    output_3d = "trajectory_3d.png"
    plt.savefig(output_3d, dpi=300)
    print(f"Saved 3D plot to {output_3d}")
    # plt.show()
    
    # --- Altitude Profile Plot ---
    plt.figure(figsize=(12, 6))
    
    # Timestamps
    # GT has absolute timestamps (Unix).
    # VIO has relative timestamps (0 to end).
    # We align VIO start to match GT start (approximation)
    # Finding alignment:
    # 1. Start of PPK file is NOT necessarily start of VIO.
    # 2. VIO usually starts when bag starts.
    # 3. Best is to plot against Time-since-start.
    
    gt_t = gt_df['timestamp'].values
    gt_t_rel = gt_t - gt_t[0]
    
    vio_t_rel = vio_df['timestamp'].values # Assuming this is time-since-start
    
    # Shift GT time to match VIO?
    # Or just plot both on 0-based time axis.
    # Note: verify if VIO actually starts at same physical location/time as GT start.
    
    # Using 'height' directly
    plt.plot(gt_t_rel, gt_df['height'], 'k--', label='GT Altitude (Ellipsoidal)', alpha=0.5)
    plt.plot(vio_t_rel, vio_df['height'], 'r-', label='VIO Altitude (MSL)')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude Profile Comparison')
    plt.legend()
    plt.grid(True)
    
    output_alt = "altitude_profile.png"
    plt.savefig(output_alt, dpi=300)
    print(f"Saved Altitude plot to {output_alt}")
    # plt.show()
    
    # --- Export CSV ---
    export_comparison_csv(gt_df, vio_df)

def export_comparison_csv(gt_df, vio_df, output_path="trajectory_comparison.csv"):
    """
    Export aligned Ground Truth and VIO data to CSV at VIO resolution.
    Interpolates Ground Truth data to match VIO timestamps.
    """
    print(f"Exporting comparison data to {output_path} (VIO Resolution)...")
    
    # 1. Align Time (Relative to Start)
    t0_gt = gt_df['timestamp'].values[0]
    t_gt_rel = gt_df['timestamp'].values - t0_gt
    
    # VIO timestamps are already relative (start at 0) in pose.csv (usually)
    t_vio_rel = vio_df['timestamp'].values 
    
    # 2. Interpolate GT to VIO timestamps
    # We want to compare at every VIO point (High resolution)
    gt_lat_interp = np.interp(t_vio_rel, t_gt_rel, gt_df['lat'])
    gt_lon_interp = np.interp(t_vio_rel, t_gt_rel, gt_df['lon'])
    gt_alt_interp = np.interp(t_vio_rel, t_gt_rel, gt_df['height'])
    
    # 3. Calculate Errors
    # 3a. Altitude Error
    error_alt = vio_df['height'].values - gt_alt_interp
    
    # 3b. 2D Position Error (Convert Lat/Lon to Meters)
    # Use GT start as origin for conversion
    lat0, lon0 = gt_df['lat'].values[0], gt_df['lon'].values[0]
    
    # Convert VIO Lat/Lon to XY
    vio_x, vio_y = latlon_to_xy(vio_df['lat'].values, vio_df['lon'].values, lat0, lon0)
    
    # Convert Interpolated GT Lat/Lon to XY
    gt_x, gt_y = latlon_to_xy(gt_lat_interp, gt_lon_interp, lat0, lon0)
    
    # Calculate Euclidean distance in 2D
    error_2d = np.sqrt((vio_x - gt_x)**2 + (vio_y - gt_y)**2)
    
    # 3c. 3D Position Error
    error_3d = np.sqrt(error_2d**2 + error_alt**2)
    
    # 4. Create DataFrame
    export_df = pd.DataFrame({
        'Time_Rel_s': t_vio_rel,
        'VIO_Lat': vio_df['lat'].values,
        'VIO_Lon': vio_df['lon'].values,
        'VIO_Alt_MSL': vio_df['height'].values,
        'GT_Lat_Interp': gt_lat_interp,
        'GT_Lon_Interp': gt_lon_interp,
        'GT_Alt_Interp': gt_alt_interp,
        'Error_Alt': error_alt,
        'Error_2D_m': error_2d,
        'Error_3D_m': error_3d
    })
    
    export_df.to_csv(output_path, index=False)
    print(f"Saved {len(export_df)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot VIO vs GT trajectory")
    parser.add_argument("--gt", required=True, help="Path to Ground Truth .pos file")
    parser.add_argument("--vio", required=True, help="Path to VIO output DIRECTORY or pose.csv")
    
    args = parser.parse_args()
    
    gt_path = args.gt
    vio_path = args.vio
    
    if os.path.isdir(vio_path):
        vio_path = os.path.join(vio_path, "preintegration", "pose.csv")
        if not os.path.exists(vio_path):
            # Try fallback location
            vio_path = os.path.join(args.vio, "pose.csv")
            
    print(f"Loading GT: {gt_path}")
    print(f"Loading VIO: {vio_path}")
    
    df_gt = load_ppk_trajectory(gt_path)
    df_vio = load_vio_pose(vio_path)
    
    if df_gt is not None and df_vio is not None:
        plot_trajectories(df_gt, df_vio)
