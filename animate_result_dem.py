#!/usr/bin/env python3
"""
Animate VIO+VPS (pose lat,lon) vs GPS (GGA lat_dd,lon_dd) on DEM
โดยยึด timeline จาก GGA เป็นหลัก

Alignment ของเวลา:
  - กำหนด stamp_log = gga_start_stamp (default: 1653512816.50257)
    ให้เท่ากับ Timestamp(s) = 0 ของ pose.csv
  - GGA ทุกแถว: t_gga_rel = stamp_log - gga_start_stamp
  - หา pose แถวที่มี Timestamp(s) ใกล้เคียง t_gga_rel ที่สุด
    -> ใช้ lat, lon, AGL, Frame ของ pose แถวนั้น

Info box:
  1. Distance (km)   - ระยะทางตามวิถี pose (เฉพาะจุดที่ match กับ GGA)
  2. Altitude (AGL)  - AGL จาก pose (แถวที่ match กับ GGA)
  3. Speed (km/h)    - ความเร็วจากวิถี pose ในหน่วยเมตร/วินาที
  4. Error to GPS (m)- ระยะ 2D บนพื้นโลกระหว่าง lat/lon pose vs lat_dd/lon_dd GGA
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show as rio_show
from rasterio.windows import from_bounds
from pyproj import CRS, Transformer
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


# ---------- Loading functions ----------

def load_pose(pose_path: str) -> pd.DataFrame:
    df = pd.read_csv(pose_path)
    required = ['Timestamp(s)', 'lat', 'lon', 'AGL(m)', 'Frame']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {pose_path}")
    df = df.sort_values('Timestamp(s)').reset_index(drop=True)
    return df


def load_gga(gga_path: str) -> pd.DataFrame:
    df = pd.read_csv(gga_path)
    required = ['stamp_log', 'lat_dd', 'lon_dd']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {gga_path}")
    df = df.sort_values('stamp_log').reset_index(drop=True)
    return df


def latlon_to_xy(lat: np.ndarray, lon: np.ndarray, dem) -> tuple[np.ndarray, np.ndarray]:
    """แปลง lat/lon (deg) -> พิกัดใน CRS ของ DEM (ใช้สำหรับการพล็อตบน DEM)"""
    wgs84 = CRS.from_epsg(4326)
    dem_crs = dem.crs
    transformer = Transformer.from_crs(wgs84, dem_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)  # lon, lat
    return np.asarray(x), np.asarray(y)


def latlon_to_local_meters(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    แปลง lat/lon -> local UTM (เมตร) สำหรับคำนวณระยะทาง/สปีด/เออเรอร์
    ใช้ zone จากค่าเฉลี่ย lat/lon ทั้ง flight เพื่อให้ทั้ง pose และ GGA อยู่ใน frame เดียวกัน
    """
    wgs84 = CRS.from_epsg(4326)
    lat0 = float(np.mean(lat))
    lon0 = float(np.mean(lon))
    zone = int((lon0 + 180.0) // 6.0) + 1
    if lat0 >= 0:
        epsg = 32600 + zone  # UTM North
    else:
        epsg = 32700 + zone  # UTM South
    utm_crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    x_m, y_m = transformer.transform(lon, lat)
    return np.asarray(x_m), np.asarray(y_m)


def build_image_list(image_dir: str):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(image_dir, e)))
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    paths = sorted(paths)
    return np.array(paths)


def compute_distance_and_speed(t: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ระยะทางสะสม (km) และความเร็ว (km/h) จากวิถี 2D (x_m, y_m) หน่วยเมตร
    t = เวลาที่สัมพันธ์กับวิถี (เช่น gga_t_rel)
    """
    pos = np.vstack([x_m, y_m]).T
    deltas = np.linalg.norm(np.diff(pos, axis=0), axis=1)  # m
    dist = np.concatenate(([0.0], np.cumsum(deltas)))       # m
    dist_km = dist * 1e-3

    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    speed_mps = deltas / dt
    speed_mps = np.concatenate(([0.0], speed_mps))
    speed_kmh = speed_mps * 3.6

    # handle NaN
    if np.any(np.isnan(speed_kmh)):
        valid = np.where(~np.isnan(speed_kmh))[0]
        if len(valid) == 0:
            speed_kmh[:] = 0.0
        else:
            first = valid[0]
            speed_kmh[:first] = speed_kmh[first]
            for i in range(first + 1, len(speed_kmh)):
                if np.isnan(speed_kmh[i]):
                    speed_kmh[i] = speed_kmh[i - 1]

    return dist_km, speed_kmh


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Animate VIO+VPS (pose) vs GPS (GGA) on DEM driven by GGA time."
    )
    parser.add_argument('--pose_csv', required=True,
                        help="pose.csv (with Timestamp(s), lat, lon, AGL(m), Frame)")
    parser.add_argument('--gga_csv', required=True,
                        help="flight_log_from_gga.csv (with stamp_log, lat_dd, lon_dd)")
    parser.add_argument('--dem', required=True,
                        help="DEM GeoTIFF path")
    parser.add_argument('--images', required=True,
                        help="Camera image directory")
    parser.add_argument('--output', default='',
                        help="Output video file (e.g. pose_vs_gga.mp4). If empty: only show.")
    parser.add_argument('--fps', type=int, default=20,
                        help="FPS when saving animation")
    parser.add_argument('--step', type=int, default=1,
                        help="Subsample GGA frames (step>1 = faster animation)")
    parser.add_argument('--codec', default='mpeg4',
                        help="ffmpeg codec, e.g. 'mpeg4', 'libx264', 'h264'")
    parser.add_argument('--gga_start_stamp', type=float, default=1653512816.50257,
                        help="stamp_log ที่เทียบเท่า pose Timestamp(s)=0")
    args = parser.parse_args()

    # ----- โหลดข้อมูล -----
    pose_df = load_pose(args.pose_csv)
    gga_df = load_gga(args.gga_csv)
    dem = rasterio.open(args.dem)

    # pose time (relative)
    pose_t = pose_df['Timestamp(s)'].to_numpy()
    pose_t_rel = pose_t - pose_t[0]   # ให้ pose เริ่มที่ 0 เสมอ

    # GGA time (relative จาก gga_start_stamp)
    gga_stamp = gga_df['stamp_log'].to_numpy()
    gga_t_rel = gga_stamp - args.gga_start_stamp

    # pose values
    pose_lat = pose_df['lat'].to_numpy()
    pose_lon = pose_df['lon'].to_numpy()
    pose_agl = pose_df['AGL(m)'].to_numpy()
    pose_frame = pose_df['Frame'].to_numpy()

    # GGA values
    gga_lat = gga_df['lat_dd'].to_numpy()
    gga_lon = gga_df['lon_dd'].to_numpy()

    # ----- map GGA time -> nearest pose sample -----
    idx = np.searchsorted(pose_t_rel, gga_t_rel, side='left')
    idx[idx == len(pose_t_rel)] = len(pose_t_rel) - 1
    for i, tg in enumerate(gga_t_rel):
        j = idx[i]
        j0 = max(j - 1, 0)
        if abs(pose_t_rel[j0] - tg) < abs(pose_t_rel[j] - tg):
            idx[i] = j0

    # pose ที่ match กับแต่ละ GGA sample (มีจำนวนเท่ากับ GGA)
    pose_lat_gga = pose_lat[idx]
    pose_lon_gga = pose_lon[idx]
    pose_agl_gga = pose_agl[idx]
    pose_frame_gga = pose_frame[idx]

    # ----- พิกัด DEM สำหรับการพล็อต -----
    pose_x_dem, pose_y_dem = latlon_to_xy(pose_lat_gga, pose_lon_gga, dem)
    gga_x_dem,  gga_y_dem  = latlon_to_xy(gga_lat,      gga_lon,      dem)

    # ----- พิกัดเมตรสำหรับคำนวณระยะ/สปีด/เออเรอร์ -----
    # รวม lat/lon ของ pose และ GGA ร่วมกันเพื่อให้ใช้ local frame เดียวกัน
    lat_all = np.concatenate([pose_lat_gga, gga_lat])
    lon_all = np.concatenate([pose_lon_gga, gga_lon])
    x_all_m, y_all_m = latlon_to_local_meters(lat_all, lon_all)
    n_gga = len(gga_lat)
    pose_x_m = x_all_m[:n_gga]
    pose_y_m = y_all_m[:n_gga]
    gga_x_m  = x_all_m[n_gga:]
    gga_y_m  = y_all_m[n_gga:]

    # Distance & Speed ตามวิถี pose (sample เท่ากับ GGA)
    dist_km, speed_kmh = compute_distance_and_speed(gga_t_rel, pose_x_m, pose_y_m)

    # Error to GPS จาก lat/lon (pose vs GGA) ในหน่วยเมตร (2D ground)
    err_gps = np.sqrt((pose_x_m - gga_x_m) ** 2 + (pose_y_m - gga_y_m) ** 2)

    # AGL ต่อ GGA sampleจาก pose
    agl_gga = pose_agl_gga

    # ----- รูปภาพ: map ผ่าน Frame ของ pose ที่ match กับ GGA -----
    img_paths = build_image_list(args.images)
    n_img = len(img_paths)
    frames_idx = pd.Series(pose_frame_gga)
    # บาง sample อาจไม่มี Frame (NaN) -> ffill/bfill ให้ใช้รูปใกล้เคียง
    frames_idx = frames_idx.ffill().bfill().fillna(0).astype(int).to_numpy()
    frames_idx = np.clip(frames_idx, 0, n_img - 1)

    # >>> NEW: preload รูปทั้งหมดเข้า memory เพื่อลดอาการกระตุก <<<
    img_cache = []
    for p in img_paths:
        im = Image.open(p)
        img_cache.append(im.copy())
        im.close()

    # ----- Subsample GGA frames สำหรับ animation -----
    idx_all = np.arange(n_gga)
    idx_frames = idx_all[::args.step]
    if idx_frames[-1] != idx_all[-1]:
        idx_frames = np.append(idx_frames, idx_all[-1])

    print("pose_x_dem range:", pose_x_dem.min(), pose_x_dem.max())
    print("gga_x_dem range :", gga_x_dem.min(), gga_x_dem.max())
    print("pose_y_dem range:", pose_y_dem.min(), pose_y_dem.max())
    print("gga_y_dem range :", gga_y_dem.min(), gga_y_dem.max())

    # ----- Crop DEM รอบ AOI (pose + GGA) ±1km -----
    all_x = np.concatenate([pose_x_dem, gga_x_dem])
    all_y = np.concatenate([pose_y_dem, gga_y_dem])
    margin_m = 1000.0  # 1 km

    if dem.crs is not None and dem.crs.is_geographic:
        # DEM อยู่ในองศา -> แปลง 1 km เป็น degrees
        lat_mean = float(np.mean(all_y))
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(lat_mean))
        margin_lat = margin_m / meters_per_deg_lat
        margin_lon = margin_m / max(meters_per_deg_lon, 1e-6)
    else:
        # DEM อยู่ในหน่วยเมตร
        margin_lat = margin_m
        margin_lon = margin_m

    minx = float(all_x.min() - margin_lon)
    maxx = float(all_x.max() + margin_lon)
    miny = float(all_y.min() - margin_lat)
    maxy = float(all_y.max() + margin_lat)

    window = from_bounds(minx, miny, maxx, maxy, transform=dem.transform)
    dem_crop = dem.read(1, window=window)
    dem_transform = dem.window_transform(window)

    # ----- Layout figure -----
    fig, axes = plt.subplot_mosaic(
        [['map', 'image'],
         ['map', 'info']],
        figsize=(12, 8)
    )
    ax_map = axes['map']
    ax_img = axes['image']
    ax_info = axes['info']

    # DEM AOI
    rio_show(dem_crop, transform=dem_transform, ax=ax_map)
    ax_map.set_title("VIO+VPS")
    ax_map.set_xlabel("X (DEM CRS)")
    ax_map.set_ylabel("Y (DEM CRS)")
    ax_map.set_xlim(minx, maxx)
    ax_map.set_ylim(miny, maxy)
    ax_map.set_aspect('equal', 'box')

    # เส้นวิถี
    line_pose, = ax_map.plot([], [], '-', lw=2, label="VIO+VPS (pose)")
    line_gps,  = ax_map.plot([], [], '--', lw=2, label="GPS (GGA)")
    pt_pose,   = ax_map.plot([], [], 'o')
    pt_gps,    = ax_map.plot([], [], 'x')
    ax_map.legend(loc='upper left')

    # รูปกล้อง
    ax_img.set_title("Camera image")
    ax_img.axis('off')
    first_img = img_cache[frames_idx[idx_frames[0]]]   # ใช้จาก cache
    im_artist = ax_img.imshow(first_img)

    # Flight info
    ax_info.set_title("Flight info")
    ax_info.axis('off')
    info_text = ax_info.text(
        0.05, 0.95, "",
        transform=ax_info.transAxes,
        va='top', ha='left',
        fontsize=12,
        family='monospace'
    )

    # ----- Animation update -----

    def update(i_gga: int):
        # i_gga คือ index จริงใน array GGA (มาจาก idx_frames)
        i = i_gga

        # วิถีบน DEM
        line_pose.set_data(pose_x_dem[:i + 1], pose_y_dem[:i + 1])
        line_gps.set_data(gga_x_dem[:i + 1],  gga_y_dem[:i + 1])
        pt_pose.set_data([pose_x_dem[i]], [pose_y_dem[i]])
        pt_gps.set_data([gga_x_dem[i]],  [gga_y_dem[i]])

        # รูปกล้อง (ใช้ cache แทน Image.open ทุก frame)
        im_artist.set_data(img_cache[frames_idx[i]])

        # Flight info ...
        txt = (
            f"Distance (km)   : {dist_km[i]:7.3f}\n"
            f"Altitude (AGL) m: {agl_gga[i]:7.1f}\n"
            f"Speed (km/h)    : {speed_kmh[i]:7.1f}\n"
            f"Error to GPS (m): {err_gps[i]:7.2f}"
        )
        info_text.set_text(txt)

        return line_pose, line_gps, pt_pose, pt_gps, im_artist, info_text

    ani = FuncAnimation(
        fig,
        update,
        frames=idx_frames,
        interval=50,
        blit=False
    )

    # Save or show
    if args.output:
        print(f"Saving animation to {args.output} ...")
        try:
            writer = FFMpegWriter(
                fps=args.fps,
                codec=args.codec,
                metadata=dict(artist="vio"),
                bitrate=1800,
            )
            ani.save(args.output, writer=writer, dpi=150)
            print("Done (mp4).")
        except Exception as e:
            print("FFMpegWriter failed:", e)
            gif_out = os.path.splitext(args.output)[0] + ".gif"
            print(f"Trying GIF output instead: {gif_out}")
            writer = PillowWriter(fps=min(args.fps, 15))
            ani.save(gif_out, writer=writer)
            print("Done (gif).")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
