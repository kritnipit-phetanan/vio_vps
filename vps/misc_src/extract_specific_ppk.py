
import pandas as pd
import numpy as np

# Config
ppk_path = "/Users/france/Downloads/vio_dataset/bell412_dataset3/bell412_dataset3_frl.pos"
target_img_ns = 1653512998238136064
target_ts_sec = target_img_ns / 1e9

print(f"Target Image Timestamp: {target_ts_sec:.3f}")

# Load PPK
# Using the same logic as satellite_matcher.py
try:
    ppk_df = pd.read_csv(ppk_path, comment='%', delim_whitespace=True, header=None)
    
    # Parse timestamps
    ppk_datetimes = pd.to_datetime(ppk_df.iloc[:, 0] + ' ' + ppk_df.iloc[:, 1])
    ppk_timestamps = ppk_datetimes.view(np.int64) / 1e9 # seconds
    
    # Find match
    time_diffs = np.abs(ppk_timestamps - target_ts_sec)
    ppk_idx = np.argmin(time_diffs)
    min_diff = time_diffs[ppk_idx]
    
    row = ppk_df.iloc[ppk_idx]
    
    lat = row[2]
    lon = row[3]
    height = row[4]
    yaw = row[16] # col 16 is yaw
    
    print("-" * 30)
    print(f"Matched PPK Row: {ppk_idx}")
    print(f"PPK Timestamp: {ppk_timestamps[ppk_idx]:.3f}")
    print(f"Diff: {min_diff:.3f}s")
    print("-" * 30)
    print(f"Latitude:  {lat:.12f}")
    print(f"Longitude: {lon:.12f}")
    print(f"Height (Ellipsoidal): {height:.3f} m")
    print(f"Yaw (NED): {yaw:.3f} deg")
    
    # Ground elevation assumption
    ground_elev = 80.0
    alt_agl = max(50.0, height - ground_elev)
    print(f"Altitude AGL (est): {alt_agl:.3f} m")
    
except Exception as e:
    print(f"Error: {e}")
