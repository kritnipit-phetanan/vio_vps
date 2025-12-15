# Plane-Aided MSCKF Integration Plan

## ปัญหาปัจจุบัน

### 1. แยกกันทำงาน (Decoupled)
```python
# Current implementation (main_loop.py):
trigger_msckf_update()  # Standard MSCKF
compute_msckf_with_plane_constraints()  # Separate plane processing
```

**ปัญหา**:
- Triangulate ซ้ำ 2 ครั้ง (ใน MSCKF และ plane_msckf)
- Plane constraints ไม่ช่วย MSCKF triangulation
- `apply_plane_constraint_update()` ไม่มีประโยชน์ (points ไม่ได้อยู่ใน state)

### 2. ฟังก์ชันที่ไม่ถูกใช้งาน

**SLAM Plane Features** (ยังไม่ implement):
- `update_plane_observations()` - Track plane lifetime
- `promote_to_slam_plane()` - Promote persistent planes to SLAM
- `initialize_slam_plane()` - Add plane to EKF state
- `marginalize_slam_plane()` - Remove plane from state
- `plane_intersection_line()` - Geometric constraints

**เหตุผล**: Phase 3 implement แค่ MSCKF Plane (short-term) ยังไม่ถึง SLAM Plane (long-term in state)

## วิธีแก้ที่ถูกต้อง

### Option A: Integrate ใน MSCKF (Recommended)

**แนวคิด**: ให้ plane detection เป็นส่วนหนึ่งของ MSCKF triangulation

```python
# msckf.py: perform_msckf_updates()
def perform_msckf_updates(..., plane_detector=None):
    # 1. Triangulate features (standard)
    triangulated = []
    for feature in features:
        point = triangulate_feature(...)
        if point: triangulated.append((fid, point))
    
    # 2. Detect planes from triangulated points
    if plane_detector:
        points = [p for _, p in triangulated]
        planes = plane_detector.detect_planes(points)
        
        # 3. Re-triangulate features on planes
        for fid, point in triangulated:
            plane = find_nearest_plane(point, planes)
            if plane:
                point_refined = project_to_plane(point, plane)
                # Use refined point for MSCKF update
    
    # 4. Standard MSCKF update with refined points
    for fid, point in triangulated:
        compute_msckf_residual(...)
```

**ข้อดี**:
- ✅ ไม่ triangulate ซ้ำ
- ✅ Plane constraints ช่วยปรับปรุง triangulation ก่อน MSCKF update
- ✅ ใช้ infrastructure MSCKF เดิม

**ข้อเสีย**:
- ต้องแก้ `perform_msckf_updates()` (complex function)

### Option B: Plane as Virtual Measurements

**แนวคิด**: ใช้ plane เป็น measurement แยก (เหมือน VPS/DEM)

```python
# After MSCKF update:
for plane in detected_planes:
    # Find features on plane
    features_on_plane = associate_features(plane)
    
    # Apply plane constraint measurement
    for fid in features_on_plane:
        # Measurement: z = n^T * p + d = 0
        residual, H = compute_plane_residual(plane, cam_states)
        kf.update(residual, H, R)
```

**ปัญหา**: MSCKF features ไม่ได้อยู่ใน state → ไม่สามารถ update ได้!

### Option C: SLAM Plane (Full Implementation)

**แนวคิด**: เก็บ planes ใน EKF state (เหมือน landmarks)

```python
# State augmentation:
state = [p, v, q, ba, bg, cam_clones..., planes...]

# Plane parameters: [n1, n2, d] (3 DOF)

# Measurement model:
z = n^T * p_feature + d = 0

# Update:
H = [∂z/∂cam_states, ∂z/∂plane]
kf.update(z, H, R)
```

**ข้อดี**:
- ✅ Planes persistent across frames
- ✅ Covariance propagation ถูกต้อง
- ✅ ตามทฤษฎี SLAM

**ข้อเสีย**:
- ⚠️ ซับซ้อนมาก
- ⚠️ State dimension ใหญ่ขึ้น
- ⚠️ Marginalization ยุ่งยาก

## แนะนำ: Option A (Quick Win)

### Implementation Steps:

1. **Modify `perform_msckf_updates()` ใน msckf.py**:
   ```python
   def perform_msckf_updates(vio_fe, cam_observations, cam_states, kf,
                             plane_detector=None,  # NEW
                             ...):
   ```

2. **Add plane detection after triangulation**:
   ```python
   # After triangulating all features
   if plane_detector and len(triangulated_points) >= 10:
       points_array = np.array(list(triangulated_points.values()))
       planes = plane_detector.detect_planes(points_array)
       
       # Refine triangulation with plane constraints
       for fid, point in triangulated_points.items():
           plane = find_nearest_plane(point, planes, threshold=0.15)
           if plane:
               point_refined = plane.project_point(point)
               triangulated_points[fid] = point_refined
   ```

3. **Pass plane_detector from main_loop.py**:
   ```python
   num_updates = perform_msckf_updates(
       vio_fe, cam_observations, cam_states, kf,
       plane_detector=self.plane_detector,  # NEW
       ...
   )
   ```

4. **Remove duplicate `compute_msckf_with_plane_constraints()` call**:
   ```python
   # DELETE this from main_loop.py:
   # compute_msckf_with_plane_constraints(...)
   ```

## Next Phase: SLAM Planes (Future Work)

เมื่อ Option A ทำงานได้ดีแล้ว สามารถ extend เป็น SLAM Planes:

1. Implement plane state augmentation
2. Implement plane observation updates
3. Implement plane marginalization
4. Use `update_plane_observations()`, `promote_to_slam_plane()`, etc.

## Summary

**Current Status**: Phase 3 ทำงานแยก (post-processing)
**Recommended Fix**: Integrate plane detection ใน `perform_msckf_updates()` (Option A)
**Expected Improvement**: ลด triangulation failures จาก 43.2% → ~20%

**Unused Functions**: SLAM Plane features → สำหรับ Phase 4 (future work)
