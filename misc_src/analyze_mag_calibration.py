#!/usr/bin/env python3
"""
Magnetometer Calibration Analysis Script

Analyzes MAG data from Bell 412 dataset and computes new calibration parameters
aligned with PPK ground truth at t=0.
"""
import csv
import math

# Load mag data manually (no pandas)
mag_file = '/Users/france/Downloads/vio_dataset/bell412_dataset3/extracted_data_new/imu_data/imu__mag/vector3.csv'

x_vals = []
y_vals = []
z_vals = []

with open(mag_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row['x']))
        y_vals.append(float(row['y']))
        z_vals.append(float(row['z']))

print(f"=== Magnetometer Data Analysis ===")
print(f"Total samples: {len(x_vals)}")

# Statistics
x_mean = sum(x_vals) / len(x_vals)
y_mean = sum(y_vals) / len(y_vals)
z_mean = sum(z_vals) / len(z_vals)

print()
print(f"=== Raw Data Statistics ===")
print(f"X: mean={x_mean:.4f}, min={min(x_vals):.4f}, max={max(x_vals):.4f}")
print(f"Y: mean={y_mean:.4f}, min={min(y_vals):.4f}, max={max(y_vals):.4f}")
print(f"Z: mean={z_mean:.4f}, min={min(z_vals):.4f}, max={max(z_vals):.4f}")

# Magnitude
mag_vals = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(x_vals, y_vals, z_vals)]
mag_mean = sum(mag_vals) / len(mag_vals)
print(f"Magnitude: mean={mag_mean:.4f}")

# Current config values
config_hard_iron = [0.243467, -0.194837, 0.341668]
print()
print(f"=== Current Config Hard Iron ===")
print(f"hard_iron_offset: {config_hard_iron}")

# Calibrated heading from first 100 samples (using current config)
headings_raw = []
headings_cal = []
for i in range(min(100, len(x_vals))):
    x = x_vals[i]
    y = y_vals[i]
    
    # Raw heading (formula: atan2(-y, x))
    h_raw = math.atan2(-y, x)
    headings_raw.append(h_raw)
    
    # Calibrated heading
    x_c = x - config_hard_iron[0]
    y_c = y - config_hard_iron[1]
    h_cal = math.atan2(-y_c, x_c)
    headings_cal.append(h_cal)

h_raw_mean = sum(headings_raw) / len(headings_raw)
h_cal_mean = sum(headings_cal) / len(headings_cal)

print()
print(f"=== Heading Analysis (first 100 samples) ===")
print(f"Raw MAG heading: {math.degrees(h_raw_mean):.1f}°")
print(f"Calibrated MAG heading (current config): {math.degrees(h_cal_mean):.1f}°")

# PPK ground truth at t=0
ppk_yaw_ned = -41.5  # degrees (from PPK file)
ppk_yaw_enu = 90 - ppk_yaw_ned  # NED to ENU = 131.5°
print()
print(f"=== PPK Ground Truth at t=0 ===")
print(f"PPK yaw (NED): {ppk_yaw_ned:.1f}° → ENU: {ppk_yaw_enu:.1f}°")

# Difference (calibration offset)
diff_raw = math.degrees(h_raw_mean) - ppk_yaw_enu
diff_cal = math.degrees(h_cal_mean) - ppk_yaw_enu
print()
print(f"=== Calibration Offset ===")
print(f"Raw MAG vs PPK ENU: {diff_raw:.1f}°")
print(f"Calibrated MAG vs PPK ENU: {diff_cal:.1f}°")

# Compute new declination to correct for offset
# new_declination = current_declination + offset
current_declination_deg = -11.60  # from config
new_declination_deg = current_declination_deg - diff_cal
new_declination_rad = math.radians(new_declination_deg)

print()
print(f"=== Recommended New Declination ===")
print(f"Current: {current_declination_deg:.2f}° ({math.radians(current_declination_deg):.4f} rad)")
print(f"Offset to correct: {-diff_cal:.1f}°")
print(f"New: {new_declination_deg:.2f}° ({new_declination_rad:.4f} rad)")

# Alternative: Compute new hard iron that aligns with PPK
# If MAG heading after calibration = 131.5° (PPK ENU), we need to find hard iron offset
# that makes atan2(-y_c, x_c) = 131.5°
target_heading_rad = math.radians(ppk_yaw_enu)

# For first sample
x0 = x_vals[0]
y0 = y_vals[0]

# Current calibrated heading = h_cal_mean
# We need: atan2(-(y0-hi_y), (x0-hi_x)) = target_heading_rad
# This is underdetermined, but if we assume the offset is mainly in Y:
# For target = 131.5°, tan(131.5°) = -(y0-hi_y)/(x0-hi_x)

print()
print(f"=== Alternative: Adjust Hard Iron for First Sample ===")
print(f"First sample raw: x={x0:.4f}, y={y0:.4f}")
print(f"Current heading for first sample (raw): {math.degrees(math.atan2(-y0, x0)):.1f}°")
print(f"Target heading: {ppk_yaw_enu:.1f}°")
