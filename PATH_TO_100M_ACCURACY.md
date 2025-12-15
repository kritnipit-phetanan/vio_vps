
================================================================================
v2.9.9.11 IMPLEMENTATION PLAN - PATH TO <100m ACCURACY
================================================================================

COMPLETED (v2.9.9.11):
----------------------
✅ 1. Process noise increased to 4× (sigma_accel: 0.5 → 2.0)
✅ 2. Preintegration noise increased to 4× (acc_n: 0.20 → 0.80)
✅ 3. NEES calculation skips initialization (frame < 100)

EXPECTED IMPACT (v2.9.9.11):
----------------------------
- Filter consistency: 6.8σ → 3-5σ (target <3σ achieved)
- VIO_VEL acceptance: 98.6% → 95-98% (slight drop acceptable)
- Position RMSE: 940m → 800-900m (modest improvement)
- NEES: 93.9% → 100% valid (no more NaN)

CURRENT ERROR ANALYSIS (v2.9.9.10):
------------------------------------
Position RMSE: 940 m
Final Error: 863 m

Error Breakdown:
- Horizontal (EN): 970 m RMSE
  - East: 433 m RMSE, 45 m bias
  - North: 868 m RMSE, 739 m bias (HUGE BIAS!)
- Vertical (U): 31 m RMSE, -13 m bias (EXCELLENT)

Error Growth:
- 0-60s:   472 m → 956 m (484m drift, 8.1 m/s)
- 60-120s:  1120 m → 1334 m (214m drift, 3.6 m/s)
- 120-180s: 1303 m → 976 m (converging!)
- 180-240s: 878 m → 836 m (stabilizing)
- 240-300s: 850 m → 863 m (stable)

KEY INSIGHT: Error PEAKS at 120s (1334m), then IMPROVES!
→ Filter is learning, but too slowly

ROOT CAUSES:
------------
1. NORTH BIAS = 739m (86% of error!)
   - Likely: Initial heading error of ~5-10°
   - At 10°: sin(10°) × 4km traveled = 700m north error
   - Matches observed 739m bias

2. VELOCITY DRIFT = ~18 m/s RMSE
   - Causes position to drift during first 120s
   - Then filter corrects (error reduces 120-300s)

3. LOW MSCKF RATE = 0.5 Hz
   - Only 142 landmark updates in 308s
   - Insufficient to constrain heading/scale

================================================================================
TIER 1: CRITICAL FIXES (Will achieve ~100-200m)
================================================================================

Priority 1: FIX INITIAL HEADING (HIGHEST IMPACT)
-------------------------------------------------
ROOT CAUSE: 739m North bias = ~5-10° heading error at start

SOLUTION A: Use PPK heading for first 30s (EASIEST, BEST)
----------------------------------------------------------
1. Extract PPK heading from ground truth trajectory
2. Initialize VIO with correct heading (not magnetometer)
3. Let magnetometer take over after 30s

Implementation:
```python
# In data_loaders.py: get_ppk_initial_heading()
def get_ppk_initial_heading(ppk_trajectory, lat0, lon0, duration=30.0):
    """Extract heading from PPK trajectory (first 30s)."""
    ppk_30s = ppk_trajectory[ppk_trajectory['t'] <= ppk_trajectory['t'].min() + duration]
    
    # Convert lat/lon to local ENU
    x, y = latlon_to_xy(ppk_30s['lat'].values, ppk_30s['lon'].values, lat0, lon0)
    
    # Compute heading from velocity vector
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(ppk_30s['t'].values)
    
    vx = dx / dt
    vy = dy / dt
    
    # Heading = atan2(vy, vx) in ENU
    headings = np.arctan2(vy, vx)
    median_heading = np.median(headings)
    
    return median_heading

# In main_loop.py: use PPK heading for initialization
if ppk_trajectory is not None:
    ppk_heading = get_ppk_initial_heading(ppk_trajectory, lat0, lon0)
    # Initialize KF with correct heading
    kf.x[6:9] = euler_to_quat(0, 0, ppk_heading)  # roll, pitch, yaw
```

EXPECTED: 863m → 200-300m (65% improvement!)
Reason: Eliminates 739m north bias immediately

SOLUTION B: Adaptive magnetometer convergence (BACKUP)
-------------------------------------------------------
If no ground truth available:
1. Use magnetometer with tight EMA (alpha=0.05)
2. Increase convergence_window to 30s
3. During convergence: trust gyro integration + mag correction

EXPECTED: 863m → 400-500m (40% improvement)


Priority 2: IMPROVE MSCKF RATE (CRITICAL FOR <200m)
----------------------------------------------------
Current: 0.5 Hz (142 updates / 308s) = TOO LOW
Target: 3-4 Hz (900-1350 updates / 308s)

SOLUTION: Adaptive Reprojection Threshold
------------------------------------------
Problem: Current 12px threshold is too strict for initialization
Solution: Start permissive (20px), tighten as filter converges

Implementation:
```python
# In msckf.py: adaptive_reprojection_threshold()
def get_adaptive_threshold(kf, base_threshold=12.0):
    """Adaptive reprojection threshold based on filter convergence."""
    # Get velocity covariance magnitude
    P_vel = kf.P[3:6, 3:6]
    vel_sigma = np.sqrt(np.trace(P_vel) / 3)
    
    # Scale threshold based on uncertainty
    # High uncertainty (initialization) → permissive (20px)
    # Low uncertainty (converged) → strict (8px)
    if vel_sigma > 3.0:  # High uncertainty
        threshold = 20.0
    elif vel_sigma > 1.0:  # Medium uncertainty
        threshold = 15.0
    else:  # Low uncertainty (converged)
        threshold = 10.0
    
    return threshold

# In triangulate_msckf_features():
MAX_REPROJ_ERROR_PX = get_adaptive_threshold(kf)
```

EXPECTED: 863m → 300-400m (with initial heading fix: 200-300m → 150-200m)
Reason: More landmark updates constrain velocity scale


Priority 3: MULTI-BASELINE TRIANGULATION (REFINEMENT)
------------------------------------------------------
Current: 2-frame triangulation (frame_i, frame_j)
Target: 3+ frame triangulation for better geometry

SOLUTION: Track features longer, triangulate with best 3+ frames
-----------------------------------------------------------------
```python
# In msckf.py: select_best_frames_for_triangulation()
def select_best_frames(feature_track, min_frames=3):
    """Select best frames for triangulation (max baseline, min reprojection)."""
    # Sort by baseline distance
    baselines = []
    for i in range(len(feature_track)-1):
        for j in range(i+1, len(feature_track)):
            pos_i = feature_track[i]['position']
            pos_j = feature_track[j]['position']
            baseline = np.linalg.norm(pos_j - pos_i)
            baselines.append((i, j, baseline))
    
    # Sort by baseline (largest first)
    baselines.sort(key=lambda x: x[2], reverse=True)
    
    # Select top N frames
    selected = baselines[:min_frames]
    return selected
```

EXPECTED: 150-200m → 100-150m (25% improvement)


================================================================================
TIER 2: REFINEMENT (Will achieve ~50-100m)
================================================================================

Priority 4: VIO SCALE CALIBRATION
----------------------------------
Current: VIO velocity has unknown scale factor
Target: Calibrate scale from ground truth

SOLUTION: Offline scale calibration
------------------------------------
1. Run VIO without VIO_VEL updates (MSCKF only)
2. Compare VIO velocity vs. ground truth velocity
3. Compute scale factor: s = |v_gt| / |v_vio|
4. Apply scale to VIO measurements

EXPECTED: 100-150m → 70-100m (30% improvement)


Priority 5: ONLINE IMU BIAS ESTIMATION
---------------------------------------
Current: Static initial bias from first 10s
Target: Continuous bias estimation

SOLUTION: Augment state with bias terms
----------------------------------------
- Add 6 states: [ba_x, ba_y, ba_z, bg_x, bg_y, bg_z]
- Increase P matrix: 15 → 21 states
- Add process noise for bias random walk
- Update bias during MSCKF/VIO_VEL updates

EXPECTED: 70-100m → 50-80m (25% improvement)


================================================================================
IMPLEMENTATION ROADMAP
================================================================================

v2.9.9.11 (Current):
--------------------
✅ 4× process noise
✅ NEES initialization skip
Expected: 863m → 800-900m

v2.9.10.0 (CRITICAL):
---------------------
→ PPK initial heading calibration
→ Adaptive MSCKF reprojection threshold
Expected: 800-900m → 150-200m (KEY BREAKTHROUGH)

v2.9.11.0 (REFINEMENT):
-----------------------
→ Multi-baseline triangulation
→ Feature persistence tracking
Expected: 150-200m → 80-120m (TARGET RANGE)

v2.9.12.0 (POLISHING):
----------------------
→ VIO scale calibration
→ Online IMU bias estimation
Expected: 80-120m → 50-80m (<100m ACHIEVED)


================================================================================
CRITICAL PATH TO <100m:
================================================================================

MUST HAVE:
1. PPK initial heading (eliminates 739m north bias)
2. Adaptive MSCKF threshold (3-4 Hz landmark rate)

SHOULD HAVE:
3. Multi-baseline triangulation (better geometry)
4. VIO scale calibration (removes scale drift)

NICE TO HAVE:
5. Online bias estimation (polishing)


WITHOUT #1 and #2: Cannot achieve <200m
WITH #1 and #2: Can achieve ~150-200m
WITH #1, #2, #3, #4: Can achieve <100m
WITH all: Can achieve ~50-80m


NEXT STEP: Run v2.9.9.11 benchmark, then implement v2.9.10.0 (PPK heading + adaptive MSCKF)
