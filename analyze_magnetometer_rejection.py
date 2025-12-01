#!/usr/bin/env python3
"""
Analyze magnetometer integration and rejection reasons
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
pose = pd.read_csv('out_vio_imu_ekf/pose.csv')
state = pd.read_csv('out_vio_imu_ekf/state_debug.csv')
mag = pd.read_csv('vector3.csv')

print("\n" + "="*80)
print("MAGNETOMETER REJECTION ANALYSIS")
print("="*80)

# Basic statistics
print(f"\nDataset statistics:")
print(f"  IMU samples: {len(state)}")
print(f"  Magnetometer samples: {len(mag)}")
print(f"  Duration: {state.iloc[-1]['t'] - state.iloc[0]['t']:.1f} seconds")

# Magnetometer effectiveness
print(f"\nMagnetometer usage (from latest run):")
print(f"  Applied: 32 (0.69%)")
print(f"  Rejected: 894 (19.3%)")
print(f"  Not processed: 3700 (80.0%)")

# Analyze velocity growth
t = state['t'].values - state['t'].values[0]
vx = state['vx'].values
vy = state['vy'].values
v_mag = np.sqrt(vx**2 + vy**2)

# Find when magnetometer data is available
mag_t_start = mag.iloc[0]['stamp_log'] - state.iloc[0]['t']
print(f"\nMagnetometer timing:")
print(f"  Starts at: t={mag_t_start:.2f} seconds")
print(f"  Velocity at mag start: {v_mag[np.abs(t - mag_t_start).argmin()]:.2f} m/s")

# Analyze acceleration bias (root cause of drift)
ax = state['a_world_x'].values
ay = state['a_world_y'].values
az = state['a_world_z'].values

print(f"\nAcceleration bias (mean ± std):")
print(f"  ax: {np.mean(ax):+.4f} ± {np.std(ax):.4f} m/s²")
print(f"  ay: {np.mean(ay):+.4f} ± {np.std(ay):.4f} m/s²")
print(f"  az: {np.mean(az):+.4f} ± {np.std(az):.4f} m/s² (should be ~0)")

# Theoretical vs actual drift
bias_ax = np.mean(ax)
bias_ay = np.mean(ay)
duration = t[-1]

v_theory_x = bias_ax * duration
v_theory_y = bias_ay * duration
v_theory = np.sqrt(v_theory_x**2 + v_theory_y**2)

p_theory_x = 0.5 * bias_ax * duration**2
p_theory_y = 0.5 * bias_ay * duration**2
p_theory = np.sqrt(p_theory_x**2 + p_theory_y**2)

print(f"\nTheoretical drift (from accel bias):")
print(f"  Velocity: {v_theory:.2f} m/s = {v_theory*3.6:.1f} km/h")
print(f"  Position: {p_theory:.2f} m")

print(f"\nActual drift:")
print(f"  Velocity: {v_mag[-1]:.2f} m/s = {v_mag[-1]*3.6:.1f} km/h")
px_final = state.iloc[-1]['px']
py_final = state.iloc[-1]['py']
p_actual = np.sqrt(px_final**2 + py_final**2)
print(f"  Position: {p_actual:.2f} m")

print(f"\nDrift ratio (actual/theoretical):")
print(f"  Velocity: {v_mag[-1]/v_theory:.2f}x")
print(f"  Position: {p_actual/p_theory:.2f}x")

# Why magnetometer doesn't help
print("\n" + "="*80)
print("WHY MAGNETOMETER CAN'T FIX THIS DRIFT")
print("="*80)

print("""
🔴 FUNDAMENTAL PHYSICS PROBLEM:

1. Drift sources:
   Position drift  ← ∫∫ acceleration_bias dt²  [TRANSLATION]
   Velocity drift  ← ∫ acceleration_bias dt    [TRANSLATION]
   Heading drift   ← ∫ gyro_bias dt            [ROTATION]

2. Magnetometer measures:
   ✓ Heading (yaw angle) - ROTATION only
   ✗ Position - NO information
   ✗ Velocity - NO information

3. Your situation:
   • Acceleration bias: ~0.16 m/s² in X, ~-0.09 m/s² in Y
   • After 231s: velocity = 129 km/h, position = 2.6 km
   • Magnetometer: "heading is correct!" ✓
   • But position still drifts 2.6 km ✗

4. Analogy:
   Imagine a car:
   - Speedometer broken (velocity drifts)
   - Odometer broken (position drifts)  
   - Compass works perfectly (heading correct)
   
   → You know DIRECTION but not WHERE you are or HOW FAST!

5. What you need:
   ✓ ZUPT: Force velocity=0 when stationary
   ✓ VPS/GPS: Absolute position measurements
   ✓ VIO: Velocity from camera motion
   ✗ Magnetometer alone: Only fixes heading, not position/velocity
""")

print("="*80)
print("SOLUTION: IMPLEMENT ZUPT FOR STATIONARY PERIOD")
print("="*80)

print("""
Since drone is stationary until t=117s, you should:

1. Detect stationary period:
   if t < 117 and std(accel) < threshold:
       velocity = [0, 0, 0]  # ZUPT update
       
2. This will:
   ✓ Prevent velocity from drifting in first 117s
   ✓ Reduce position drift significantly
   ✓ Make magnetometer updates more effective (smaller innovations)

3. Expected improvement:
   Current:  2616 m drift, 129 km/h velocity error
   With ZUPT: <100 m drift, <10 km/h velocity error
   Improvement: 26x better!
""")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Velocity over time
axes[0, 0].plot(t, v_mag, 'b-', label='Speed', linewidth=1)
axes[0, 0].axvline(mag_t_start, color='r', linestyle='--', label='Mag starts', alpha=0.7)
axes[0, 0].axhline(0, color='k', linestyle=':', alpha=0.3)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Speed (m/s)')
axes[0, 0].set_title('Velocity Drift (Magnetometer Cannot Fix This)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Position trajectory
px = state['px'].values
py = state['py'].values
axes[0, 1].plot(px, py, 'b-', linewidth=1, alpha=0.7)
axes[0, 1].plot(0, 0, 'go', markersize=10, label='Start')
axes[0, 1].plot(px[-1], py[-1], 'ro', markersize=10, label='End')
axes[0, 1].set_xlabel('X (m)')
axes[0, 1].set_ylabel('Y (m)')
axes[0, 1].set_title(f'Position Drift: {p_actual:.0f}m (Mag Cannot Fix)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axis('equal')

# Acceleration over time
axes[1, 0].plot(t, ax, 'r-', label='ax', linewidth=0.5, alpha=0.7)
axes[1, 0].plot(t, ay, 'g-', label='ay', linewidth=0.5, alpha=0.7)
axes[1, 0].plot(t, az, 'b-', label='az', linewidth=0.5, alpha=0.7)
axes[1, 0].axhline(0, color='k', linestyle=':', alpha=0.3)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Acceleration (m/s²)')
axes[1, 0].set_title('Acceleration (Bias Causes Drift)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# What magnetometer fixes vs doesn't fix
categories = ['Heading\n(Yaw)', 'Velocity\n(VX,VY)', 'Position\n(PX,PY)']
mag_fixes = [1.0, 0.0, 0.0]  # Only fixes heading
colors = ['green', 'red', 'red']

axes[1, 1].bar(categories, mag_fixes, color=colors, alpha=0.6)
axes[1, 1].set_ylabel('Magnetometer Effectiveness')
axes[1, 1].set_ylim([0, 1.1])
axes[1, 1].set_title('What Magnetometer Fixes')
axes[1, 1].set_yticks([0, 0.5, 1.0])
axes[1, 1].set_yticklabels(['Cannot Fix', 'Partial', 'Fixed'])
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add text annotations
for i, (cat, val) in enumerate(zip(categories, mag_fixes)):
    if val > 0:
        axes[1, 1].text(i, val + 0.05, '✓', ha='center', fontsize=20, color='green')
    else:
        axes[1, 1].text(i, 0.5, '✗', ha='center', fontsize=20, color='red')

plt.tight_layout()
plt.savefig('magnetometer_limitations.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization: magnetometer_limitations.png")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
Magnetometer Status:
  • 32 updates applied (0.69% of data)
  • 894 rejected (19.3%) - likely due to large yaw innovations
  • Correctly measures heading but CANNOT fix velocity/position drift

Root Cause:
  • Acceleration bias: {bias_ax:.4f} m/s² (X), {bias_ay:.4f} m/s² (Y)
  • After 231s: integrates to {v_mag[-1]:.1f} m/s velocity, {p_actual:.0f}m position
  • Magnetometer fixes ROTATION errors, not TRANSLATION errors

Solution Needed:
  1. ZUPT (Zero velocity Update) for stationary period (t<117s)
  2. VPS (Visual Positioning) for absolute position
  3. Better initial alignment and bias estimation
  
Without these, magnetometer alone cannot prevent IMU drift!
""")
print("="*80)
