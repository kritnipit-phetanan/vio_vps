import pandas as pd
import numpy as np
import plotly.graph_objects as go
import rasterio
from pyproj import CRS, Transformer
from geopy.distance import geodesic
import os

# --- 1. กำหนดค่าคงที่และ Path ของไฟล์ (อัปเดตให้ตรงกับข้อมูลปัจจุบัน) ---
# ค่าเริ่มต้นสามารถแก้ไขได้ตามต้องการ
POSE_CSV_PATH = "./out_vio_imu_ekf/pose.csv"  # จากการรันล่าสุด
GROUND_TRUTH_PATH = "./flight_log_from_gga.csv"           # ใช้ flight_log_from_gga.csv ที่มี lat_dd, lon_dd, altitude_MSL_m
DEM_PATH = "./DSM_10_N47_00_W054_00_AOI.tif"  # ใช้ DEM ปัจจุบัน

# หากไม่กำหนด (None) จะอิงเวลาจากไฟล์ ground truth ทั้งช่วง
VIDEO_START_MS = None
VIDEO_END_MS = None
VIDEO_FPS = 30
OUTPUT_HTML_FILE = "trajectory_on_dem_animation.html"
OUTPUT_CSV_FILE = "synced_trajectory.csv"
MPH_TO_KMH = 1.60934
FT_TO_M = 0.3048

def load_vo_data_as_df(file_path):
    if not os.path.exists(file_path):
        return None
    # pose.csv จาก vio_vps.py: มีคอลัมน์ lat, lon, AGL(m)
    df = pd.read_csv(file_path).rename(columns={'lat': 'latitude', 'lon': 'longitude', 'AGL(m)': 'altitude'})
    return df[['latitude', 'longitude', 'altitude']]

def load_dem(dem_path):
    if not os.path.exists(dem_path):
        return None, None, None, None
    with rasterio.open(dem_path) as src:
        dem_z = src.read(1)
        dem_z = dem_z.astype(float)
        if src.nodata is not None:
            dem_z[dem_z == src.nodata] = np.nan
        transform = src.transform
        cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
        dem_x, dem_y = transform * (cols, rows)
        return np.array(dem_x), np.array(dem_y), dem_z, src.crs

def sample_dem_at_latlon(dem_path, lat_arr, lon_arr):
    """คืนค่า DEM สูงเหนือระดับน้ำทะเล (หน่วยเมตร) ที่ตำแหน่ง lat/lon ที่ให้มา"""
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        to_dem = Transformer.from_crs(CRS.from_epsg(4326), dem_crs, always_xy=True)
        x_arr, y_arr = to_dem.transform(lon_arr, lat_arr)
        # rasterio.sample ต้องการลิสต์ของ (x, y)
        samples = list(src.sample(list(zip(x_arr, y_arr))))
        dem_vals = np.array([s[0] if s[0] is not None else np.nan for s in samples], dtype=float)
        # กรองค่าที่ไม่สมเหตุผล
        dem_vals[~np.isfinite(dem_vals)] = np.nan
        return dem_vals

def load_ground_truth_as_df(file_path, start_time_ms=None, end_time_ms=None, dem_path=None):
    if not os.path.exists(file_path):
        return None
    # ใช้ flight_log_from_gga.csv: สกัดคอลัมน์ที่จำเป็นและคำนวณ AGL จาก DEM
    # Column names: stamp_log (in seconds), lat_dd, lon_dd, altitude_MSL_m, xSpeed_mph, ySpeed_mph, zSpeed_mph
    base_usecols = ['stamp_log', 'lat_dd', 'lon_dd']
    # ตรวจสอบคอลัมน์ที่มีในไฟล์ครั้งเดียว
    file_cols = set(pd.read_csv(file_path, nrows=0).columns)
    has_speed = 'xSpeed_mph' in file_cols and 'ySpeed_mph' in file_cols
    has_msl = 'altitude_MSL_m' in file_cols

    usecols = base_usecols + ([ 'xSpeed_mph', 'ySpeed_mph' ] if has_speed else []) + ([ 'altitude_MSL_m' ] if has_msl else [])
    df = pd.read_csv(file_path, usecols=usecols)
    
    # Rename columns to match expected format
    df = df.rename(columns={'stamp_log': 'time_s', 'lat_dd': 'latitude', 'lon_dd': 'longitude'})
    
    # Convert time from seconds to milliseconds
    df['time(millisecond)'] = (df['time_s'] * 1000).astype(int)

    # กรองช่วงเวลา หากกำหนด
    if start_time_ms is not None:
        df = df[df['time(millisecond)'] >= start_time_ms]
    if end_time_ms is not None:
        df = df[df['time(millisecond)'] <= end_time_ms]
    df = df.copy().reset_index(drop=True)

    # คำนวณ AGL: AGL = MSL(m) - DEM(m)
    if has_msl and dem_path is not None and os.path.exists(dem_path):
        msl_m = df['altitude_MSL_m'].astype(float)  # Already in meters
        dem_m = sample_dem_at_latlon(dem_path, df['latitude'].to_numpy(), df['longitude'].to_numpy())
        agl_m = msl_m.to_numpy() - dem_m
        df['altitude'] = agl_m
    else:
        # ไม่สามารถคำนวณ AGL: ตั้งค่าเป็น NaN (จะใช้เฉพาะ XY metrics)
        df['altitude'] = np.nan

    # Calculate total speed from xSpeed and ySpeed if available
    if has_speed:
        df['speed(mph)'] = np.sqrt(df['xSpeed_mph']**2 + df['ySpeed_mph']**2)
    else:
        df['speed(mph)'] = np.nan

    return df[['time(millisecond)', 'latitude', 'longitude', 'altitude', 'speed(mph)']]

def align_and_calculate_metrics(vo_df_with_time, gt_df, transformer):
    gt_timestamps = gt_df['time(millisecond)'].to_numpy()
    vo_timestamps = vo_df_with_time['time(millisecond)'].to_numpy()
    gt_coords_latlon = gt_df[['latitude', 'longitude', 'altitude']].to_numpy()

    vo_resampled_latlon = np.zeros_like(gt_coords_latlon)
    for i in range(3):
        vo_resampled_latlon[:, i] = np.interp(gt_timestamps, vo_timestamps, vo_df_with_time.iloc[:, i])

    origin_offset = vo_resampled_latlon[0] - gt_coords_latlon[0]
    vo_aligned_latlon = vo_resampled_latlon - origin_offset
    
    gt_x, gt_y = transformer.transform(gt_coords_latlon[:, 1], gt_coords_latlon[:, 0])
    vo_x, vo_y = transformer.transform(vo_aligned_latlon[:, 1], vo_aligned_latlon[:, 0])
    
    error_x, error_y, error_z = vo_x - gt_x, vo_y - gt_y, vo_aligned_latlon[:, 2] - gt_coords_latlon[:, 2]
    
    speeds_kmh = np.nan_to_num(gt_df['speed(mph)'].to_numpy() * MPH_TO_KMH)
    
    # FIX: Handle stationary drone (GPS noise causes fake distance accumulation)
    # Check if GT is truly stationary: compute std dev of position
    gt_lat_std = np.std(gt_coords_latlon[:, 0])
    gt_lon_std = np.std(gt_coords_latlon[:, 1])
    # If position std dev is small (< 0.001° ≈ 100m), drone is stationary/hovering
    # Consumer GPS has ~5m accuracy, so 0.001° is reasonable threshold
    IS_STATIONARY = (gt_lat_std < 0.001) and (gt_lon_std < 0.001)
    
    distances, errors_2d = [0.0], []
    for i in range(len(gt_coords_latlon)):
        gt_point, vo_point = (gt_coords_latlon[i, 0], gt_coords_latlon[i, 1]), (vo_aligned_latlon[i, 0], vo_aligned_latlon[i, 1])
        errors_2d.append(geodesic(gt_point, vo_point).meters)
        if i > 0:
            p1, p2 = (vo_aligned_latlon[i-1, 0], vo_aligned_latlon[i-1, 1]), (vo_aligned_latlon[i, 0], vo_aligned_latlon[i, 1])
            # If drone is stationary, don't accumulate distance (only GPS noise)
            if not IS_STATIONARY:
                distances.append(distances[-1] + geodesic(p1, p2).meters)
            else:
                distances.append(distances[-1])  # Stationary - keep previous distance
    
    # GT distance: if stationary, distance should be 0 (only GPS noise)
    gt_distances = [0.0] * len(gt_coords_latlon) if IS_STATIONARY else [0.0]
    if not IS_STATIONARY:
        for i in range(1, len(gt_coords_latlon)):
            p1, p2 = (gt_coords_latlon[i-1, 0], gt_coords_latlon[i-1, 1]), (gt_coords_latlon[i, 0], gt_coords_latlon[i, 1])
            gt_distances.append(gt_distances[-1] + geodesic(p1, p2).meters)

    return vo_aligned_latlon, gt_coords_latlon, {
        'distance': np.array(distances), 'gt_distance': np.array(gt_distances), 'altitude': vo_aligned_latlon[:, 2], 
        'speed': speeds_kmh, 'error_2d': np.array(errors_2d), 'timestamps': gt_timestamps
    }, {'x': error_x, 'y': error_y, 'z': error_z}

def save_synced_csv(gt_df, vo_aligned_latlon, filename):
    synced_df = pd.DataFrame({
        'time(millisecond)': gt_df['time(millisecond)'],
        'gt_latitude': gt_df['latitude'], 'gt_longitude': gt_df['longitude'], 'gt_agl_from_dem(meters)': gt_df['altitude'],
        'vo_latitude': vo_aligned_latlon[:, 0], 'vo_longitude': vo_aligned_latlon[:, 1], 'vo_ALT(m)': vo_aligned_latlon[:, 2]
    })
    synced_df.to_csv(filename, index=False, float_format='%.8f')
    print(f"✅ ข้อมูลที่ซิงค์กันแล้วถูกบันทึกในไฟล์: '{filename}'")

def print_metrics_summary(metrics, error_components):
    gt_distance, pred_distance, avg_speed = metrics['gt_distance'][-1], metrics['distance'][-1], np.mean(metrics['speed'])
    errors_2d = metrics['error_2d']
    avg_error_2d, max_error_2d, min_error_2d = np.mean(errors_2d), np.max(errors_2d), np.min(errors_2d)
    
    rmse_xy = np.sqrt(np.mean(errors_2d**2))
    rmse_h = np.sqrt(np.mean(error_components['z']**2))
    rmse_3d = np.sqrt(np.mean(errors_2d**2 + error_components['z']**2))
    
    cep50, cep90, cep95 = np.percentile(errors_2d, 50), np.percentile(errors_2d, 90), np.percentile(errors_2d, 95)

    print("\n--- Metric Evaluation Summary ---")
    print(f"  - Total Ground Truth Distance: {gt_distance:.2f} meters")
    print(f"  - Total Prediction Distance: {pred_distance:.2f} meters")
    print(f"  - Average Speed:             {avg_speed:.2f} km/h")
    print("  -------------------------------")
    print(f"  - RMSE_xy (m):               {rmse_xy:.4f}")
    print(f"  - RMSE_h (m):                {rmse_h:.4f}")
    print(f"  - RMSE 3D (m):               {rmse_3d:.4f}")
    print("  -------------------------------")
    print(f"  - CEP50 (m):                 {cep50:.4f}")
    print(f"  - CEP90 (m):                 {cep90:.4f}")
    print(f"  - CEP95 (m):                 {cep95:.4f}")
    print("  -------------------------------")
    print(f"  - Average Error 2D (m):      {avg_error_2d:.4f}")
    print(f"  - Maximum Error 2D (m):      {max_error_2d:.4f}")
    print(f"  - Minimum Error 2D (m):      {min_error_2d:.4f}")
    print("---------------------------------")


def create_animated_3d_plot(gt_path_xy, vo_path_xy, dem_x, dem_y, dem_z, metrics, filename):
    # <--- แก้ไข: เพิ่มตัวแปร ANIMATION_STEP ---
    # ยิ่งค่ามาก แอนิเมชันยิ่งเร็วและไฟล์ยิ่งเล็ก (แต่ slider จะละเอียดน้อยลง)
    # ค่าที่แนะนำ: 5, 10, หรือ 20
    ANIMATION_STEP = 30 

    dem_surface = go.Surface(x=dem_x, y=dem_y, z=dem_z, colorscale='viridis', showscale=False, name='DEM Terrain', hoverinfo='none')
    data = [dem_surface]
    data.append(go.Scatter3d(x=gt_path_xy[:, 0], y=gt_path_xy[:, 1], z=gt_path_xy[:, 2], mode='lines', line=dict(color='gray', width=3), name='Full Ground Truth'))
    data.append(go.Scatter3d(x=vo_path_xy[:, 0], y=vo_path_xy[:, 1], z=vo_path_xy[:, 2], mode='lines', line=dict(color='lightpink', width=3), name='Full Visual Odometry'))
    data.append(go.Scatter3d(x=[gt_path_xy[0, 0]], y=[gt_path_xy[0, 1]], z=[gt_path_xy[0, 2]], mode='lines', line=dict(color='blue', width=6), name='GT Path'))
    data.append(go.Scatter3d(x=[vo_path_xy[0, 0]], y=[vo_path_xy[0, 1]], z=[vo_path_xy[0, 2]], mode='lines', line=dict(color='red', width=6), name='VO Path'))
    
    cumulative_rmse_xy = [np.sqrt(np.mean(metrics['error_2d'][:k+1]**2)) for k in range(len(metrics['error_2d']))]

    frames, slider_steps = [], []
    # <--- แก้ไข: เปลี่ยน loop ให้กระโดดทีละ ANIMATION_STEP ---
    for k in range(0, len(gt_path_xy), ANIMATION_STEP):
        time_sec = (metrics['timestamps'][k] - metrics['timestamps'][0]) / 1000.0
        legend_text = (f"<b>Time: {time_sec:.1f} s</b><br>" +
                       f"Distance: {metrics['distance'][k]:.2f} m<br>" +
                       f"Altitude (AGL): {metrics['altitude'][k]:.2f} m<br>" +
                       f"Speed: {metrics['speed'][k]:.1f} km/h<br>" +
                       f"Error to GPS: {metrics['error_2d'][k]:.2f} m<br>" +
                       f"Cumulative Avg.RMSE XY: {cumulative_rmse_xy[k]:.2f} m")
        frames.append(go.Frame(
        data=[go.Scatter3d(x=gt_path_xy[:k+1, 0], y=gt_path_xy[:k+1, 1], z=gt_path_xy[:k+1, 2]),
            go.Scatter3d(x=vo_path_xy[:k+1, 0], y=vo_path_xy[:k+1, 1], z=vo_path_xy[:k+1, 2])],
            traces=[3, 4], name=f'frame{k}',
            layout=go.Layout(annotations=[dict(text=legend_text, align='left', showarrow=False, xref='paper', yref='paper', x=1.05, y=1, bordercolor='black', borderwidth=1)])
        ))
        slider_step = dict(method='animate', args=[[f'frame{k}'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}], label=f'{time_sec:.1f}s')
        slider_steps.append(slider_step)

    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
    title='3D Trajectory Comparison between VIO+VPS and Ground Truth on DEM',
    scene=dict(xaxis_title='Easting (m)', yaxis_title='Northing (m)', zaxis_title='Altitude (AGL, m)', aspectmode='data'),
        legend=dict(x=1.05, y=0.6),
        updatemenus=[dict(
            type='buttons', direction='left', x=0.57, xanchor='center', y=0.05, yanchor='bottom', showactive=False,
            buttons=[
                dict(label='▶ Play', method='animate', args=[None, {"frame": {"duration": 50, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True, "mode": "immediate"}]),
                dict(label='⏸ Pause', method='animate', args=[[None], {"frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}, "mode": "immediate"}]),
                dict(label='⏹ Reset', method='animate', args=[['frame0'], {"frame": {"duration": 50, "redraw": True}, "transition": {"duration": 0}, "mode": "immediate"}])
            ]
        )],
        sliders=[dict(steps=slider_steps, active=0, transition={'duration': 0},
                      currentvalue={'prefix': 'Time: ', 'visible': True, 'xanchor': 'right'},
                      pad={'t': 50, 'b': 10}, len=0.9, x=0.1, y=0)]
    )
    
    fig.add_annotation(text="<b>Metrics at Time: 0.0s</b><br>Use controls to begin", align='left', showarrow=False, xref='paper', yref='paper', x=1.05, y=1, bordercolor='black', borderwidth=1)
    fig.write_html(filename)
    print(f"✅ แอนิเมชัน 3 มิติถูกบันทึกในไฟล์: '{filename}'")

if __name__ == "__main__":
    print("--- 1. Loading Data ---")
    # โหลด DEM พร้อม CRS สำหรับใช้เป็นกรอบอ้างอิงการ plot
    dem_x, dem_y, dem_z, dem_crs = load_dem(DEM_PATH)

    # โหลด GT และคำนวณ AGL จาก DEM (ใช้ทั้งช่วงเวลาหากไม่กำหนด start/end)
    gt_df = load_ground_truth_as_df(GROUND_TRUTH_PATH, VIDEO_START_MS, VIDEO_END_MS, DEM_PATH)
    vo_df = load_vo_data_as_df(POSE_CSV_PATH)

    if vo_df is None or gt_df is None or dem_x is None:
        print("\n[FATAL] ไม่สามารถโหลดข้อมูลที่จำเป็นได้")
    else:
        # ใช้ CRS ของ DEM สำหรับทั้ง metric XY และการ plot เพื่อความสอดคล้องกับผิวภูมิประเทศ
        transformer = Transformer.from_crs(CRS.from_epsg(4326), dem_crs, always_xy=True)

        print("--- 2. Syncing Data & Calculating Metrics ---")
        frame_interval_ms = 1000.0 / VIDEO_FPS
        # หากไม่ได้กำหนด VIDEO_START_MS ให้เริ่มต้นที่เวลาแรกของ GT
        video_start_ms_eff = gt_df['time(millisecond)'].iloc[0] if VIDEO_START_MS is None else VIDEO_START_MS
        vo_df['time(millisecond)'] = video_start_ms_eff + np.arange(len(vo_df)) * frame_interval_ms

        vo_aligned_latlon, gt_path_latlon, metrics, error_components = align_and_calculate_metrics(vo_df, gt_df, transformer)
        print("  - Data synced, origin aligned, and metrics calculated.")

        print_metrics_summary(metrics, error_components)

        print("\n--- 3. Saving Synced Data to CSV ---")
        save_synced_csv(gt_df, vo_aligned_latlon, OUTPUT_CSV_FILE)

        print("\n--- 4. Converting Coordinates to DEM CRS for Plotting ---")
        gt_x, gt_y = transformer.transform(gt_path_latlon[:, 1], gt_path_latlon[:, 0])
        vo_x, vo_y = transformer.transform(vo_aligned_latlon[:, 1], vo_aligned_latlon[:, 0])
        gt_path_xy = np.column_stack((gt_x, gt_y, gt_path_latlon[:, 2]))
        vo_path_xy = np.column_stack((vo_x, vo_y, vo_aligned_latlon[:, 2]))

        print("\n--- 5. Creating 3D Animation ---")
        create_animated_3d_plot(gt_path_xy, vo_path_xy, dem_x, dem_y, dem_z, metrics, OUTPUT_HTML_FILE)