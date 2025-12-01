# VIO System Improvements Summary

## การแก้ไขหลัก (Major Fixes)

### 1. IMU Gravity Compensation ✅
**ปัญหา:** ใช้ gravity compensation ผิด - บวก gravity แทนที่จะลบ

**สาเหตุ:** IMU ส่วนใหญ่ output เป็น **specific force** (แรงเฉพาะ) ซึ่ง**ไม่รวม**แรงโน้มถ่วง
- เมื่อนิ่ง: IMU อ่านค่า a_z ≈ +9.8 m/s² (ค้าน gravity)
- การเคลื่อนที่: a_motion = R * (a_imu - bias) - g

**การแก้ไข:**
```python
# OLD (WRONG):
if gravity_is_included:
    a_world = a_world + g_world  # ผิด!

# NEW (CORRECT):
a_world = a_world - g_world  # ถูกต้อง - ลบ gravity ทุกครั้ง
```

**ผลกระทบ:** แก้ไข vertical drift ที่เกิดจาก double-counting gravity

---

### 2. VIO Velocity Scaling ✅
**ปัญหา:** ใช้ IMU speed ปัจจุบัน (ซึ่งอาจผิดพลาด) มา scale optical flow direction

**การแก้ไข:** คำนวณความเร็วจาก optical flow โดยตรง
```python
# OLD (WRONG):
speed_now = np.linalg.norm(kf.x[3:6,0])  # ใช้ velocity estimate ที่อาจผิด
vel_body = t_body * speed_now

# NEW (CORRECT):
avg_flow_px = np.median(np.linalg.norm(flows, axis=1))  # optical flow magnitude
altitude_m = abs(kf.x[2,0])
gsd = altitude_m / focal_px  # Ground Sampling Distance
speed_from_flow = (avg_flow_px / dt_img) * gsd  # v = flow * GSD / dt
vel_body = t_body * speed_from_flow
```

**สูตร:**
- GSD (m/pixel) = altitude / focal_length
- velocity (m/s) = (flow_pixels / dt) × GSD

**ผลกระทบ:** ทำให้ VIO velocity measurement ถูกต้องและไม่ circular dependency

---

### 3. Camera View Mode Selection ✅
**ฟีเจอร์ใหม่:** เพิ่มการเลือกมุมมองกล้อง 3 แบบ

#### Nadir (ก้มลง - มุมมองจากบนลงล่าง)
```python
"nadir": {
    "extrinsics": "BODY_T_CAMDOWN",
    "nadir_threshold": 30.0,      # Strict Z-alignment check
    "sigma_scale_xy": 1.5,         # Higher XY uncertainty
    "sigma_scale_z": 0.7,          # Lower Z uncertainty (good depth)
    "use_vz_only": True,           # Prefer VZ-only updates
    "min_parallax": 15,            # Need more parallax
    "max_corners": 1500,
}
```

**ลักษณะ:**
- กล้องมองตรงลงพื้น (Z-axis ชี้ลง)
- เหมาะกับ drone/UAV ที่บินปกติ
- เน้น vertical motion (VZ)
- XY motion น้อย → uncertainty สูง

#### Front (มองตรง - มุมมองด้านหน้า)
```python
"front": {
    "extrinsics": "BODY_T_CAMFRONT",
    "nadir_threshold": 60.0,      # Relaxed threshold
    "sigma_scale_xy": 0.8,         # Lower XY uncertainty (good lateral)
    "sigma_scale_z": 1.5,          # Higher Z uncertainty (depth ambiguity)
    "use_vz_only": False,          # Use full 3D velocity
    "min_parallax": 8,             # Lower parallax OK
    "max_corners": 2000,
}
```

**ลักษณะ:**
- กล้องมองตรงไปข้างหน้า
- เหมาะกับรถยนต์, robot ที่เคลื่อนที่ไปข้างหน้า
- ดี XY motion (forward/sideways)
- Z (depth) มี uncertainty สูง

#### Side (มุมเอียง)
```python
"side": {
    "extrinsics": "BODY_T_CAMSIDE",
    "nadir_threshold": 60.0,
    "sigma_scale_xy": 1.0,
    "sigma_scale_z": 1.2,
    "use_vz_only": False,
    "min_parallax": 10,
    "max_corners": 2000,
}
```

**ลักษณะ:**
- กล้องมองไปด้านข้าง
- เหมาะกับ oblique imagery
- Balanced XY/Z uncertainty

---

## การใช้งาน (Usage)

### Command Line
```bash
## วิธีใช้งาน

```bash
# กล้องก้ม (nadir) - มาตรฐาน UAV
python3 vio_vps.py --imu imu.csv --quarry quarry1.csv \
  --images_dir ./camera__image_mono/images \
  --images_index ./camera__image_mono/images_index.csv \
  --dem DSM_10_N47_00_W054_00_AOI.tif --z_state agl \
  --vps vps_result.csv \
  --camera_view nadir

# กล้องมองตรง (front) - สำหรับรถยนต์/robot
python3 vio_vps.py --imu imu.csv --quarry quarry1.csv \
  --images_dir ./camera__image_mono/images \
  --images_index ./camera__image_mono/images_index.csv \
  --dem DSM_10_N47_00_W054_00_AOI.tif --z_state agl \
  --vps vps_result.csv \
  --camera_view front
```

**หมายเหตุ:** 
- `--camera_view` มีผลเฉพาะ **VIO processing** (optical flow → velocity)
- **IMU, VPS, DEM** ยังทำงานเหมือนเดิมทุก mode
- สามารถใช้ DEM/VPS ร่วมกับทุก camera view

# Side camera (กล้องด้านข้าง) - ใช้ DEM/VPS ได้เหมือนกัน
python3 vio_vps.py --imu imu.csv --quarry quarry1.csv \
  --images_dir ./camera__image_mono/images \
  --images_index ./camera__image_mono/images_index.csv \
  --dem DSM_10_N47_00_W054_00_AOI.tif --z_state agl \
  --vps vps_result.csv \
  --camera_view side
```

### สำคัญ! 
- **DEM และ VPS ยังใช้ได้ปกติกับทุก camera view**
- `--camera_view` มีผลเฉพาะกับ **VIO (optical flow)** เท่านั้น
- IMU, VPS, DEM updates ทำงานเหมือนเดิมทุก mode

### ตัวเลือก Camera View
- `--camera_view nadir`: มุมก้ม (UAV standard)
- `--camera_view front`: มุมตรง (car/robot)
- `--camera_view side`: มุมเอียง (oblique)

---

## Technical Details

### Coordinate Frames

#### Body Frame (IMU frame)
```
X: Forward (หน้ารถ/drone)
Y: Right (ขวามือ)
Z: Down (ลงพื้น)
```

#### Camera Frames

**Nadir Camera:**
```
X_cam: Right → Y_body
Y_cam: Down → Z_body
Z_cam: Backward → -X_body
```

**Front Camera:**
```
X_cam: Right → Y_body
Y_cam: Down → Z_body
Z_cam: Forward → X_body
```

### Extrinsics Matrices

**BODY_T_CAMDOWN** (Nadir):
```python
[[ 0.00235643,  0.99997843, -0.00613037, -0.25805624],
 [-0.99960218,  0.00218315, -0.02811962, -0.01138283],
 [-0.02810563,  0.00619420,  0.99958577,  0.09243762],
 [ 0.0,         0.0,         0.0,         1.0       ]]
```

**BODY_T_CAMFRONT** (Front):
```python
[[ 0.0,  0.0,  1.0,  0.1],   # Z_cam → X_body
 [ 1.0,  0.0,  0.0,  0.0],   # X_cam → Y_body
 [ 0.0,  1.0,  0.0, -0.05],  # Y_cam → Z_body
 [ 0.0,  0.0,  0.0,  1.0]]
```

---

## การตรวจสอบ (Validation)

### ตรวจสอบ Gravity Compensation
```python
# เมื่อนิ่ง (stationary):
# IMU: a_z ≈ +9.8 m/s² (specific force ค้าน gravity)
# Expected: a_motion = R * a_imu - g ≈ 0

print(f"[DEBUG][IMU] a_body={rec.lin} a_corr={a_corr} a_world={a_world}")
# ควรเห็น a_world ≈ [0, 0, 0] เมื่อนิ่ง
```

### ตรวจสอบ VIO Velocity
```python
print(f"[DEBUG][VIO] avg_flow_px={avg_flow_px:.2f}, GSD={gsd:.6f}, speed={speed_from_flow:.3f}")
# flow_px ควรสัมพันธ์กับความเร็วจริง
# GSD ขึ้นกับความสูง (สูงขึ้น → GSD มากขึ้น)
```

### ตรวจสอบ Camera View
```python
print(f"[VIO] Camera view mode: {camera_view}")
print(f"[VIO] Using extrinsics: {extrinsics_name}")
# ควรตรงกับที่ตั้งค่า
```

---

## Expected Improvements

### ก่อนแก้ไข:
- ❌ Vertical drift จาก gravity double-counting
- ❌ VIO velocity ผิดเพราะใช้ IMU speed (circular dependency)
- ❌ ไม่สามารถใช้กับ front/side camera

### หลังแก้ไข:
- ✅ Gravity compensation ถูกต้อง
- ✅ VIO velocity คำนวณจาก optical flow โดยตรง
- ✅ รองรับ 3 camera views (nadir/front/side)
- ✅ Adaptive uncertainty สำหรับแต่ละ view mode
- ✅ ไม่มี circular dependency

---

## Debug Output ที่ควรดู

```bash
[DEBUG][IMU] t=1.234 a_body=[...] a_corr=[...] a_world=[...]
[DEBUG][VIO] avg_flow_px=12.34, dt_img=0.0500, GSD=0.001234, speed_from_flow=2.468 m/s
[DEBUG][VIO] t=1.234 vel_meas=[1.2, 0.5, -2.3] alignment_deg=25.3 use_only_vz=True
[VIO] Camera view mode: nadir
[VIO] Using extrinsics: BODY_T_CAMDOWN
```

---

## ข้อควรระวัง (Caveats)

1. **Altitude Dependency:** VIO velocity scaling ขึ้นกับความสูง (altitude)
   - ความสูงต่ำเกินไป → GSD เล็ก → อ่อนไหวต่อ noise
   - ความสูงสูงเกินไป → features ไกล → tracking ยาก

2. **Optical Flow Quality:** ต้องมี features เพียงพอ
   - Texture ดี → flow ดี → velocity ถูก
   - Texture น้อย (ท้องฟ้า, น้ำ) → flow ผิด

3. **Camera Calibration:** Extrinsics ต้องถูกต้อง
   - Nadir view: ใช้ค่าจาก calibration
   - Front/Side: ต้องปรับตาม mounting angle

4. **DEM Fallback:** ยังคงใช้ last valid DEM เมื่อ lookup ล้มเหลว

---

## สรุป

การแก้ไขครั้งนี้แก้ไขปัญหาพื้นฐาน 3 ข้อ:
1. ✅ **Gravity compensation** - แก้ double-counting
2. ✅ **VIO velocity scaling** - ใช้ optical flow แทน IMU
3. ✅ **Camera view modes** - รองรับ nadir/front/side

ระบบควรทำงานได้ดีขึ้นมากสำหรับทั้ง nadir และ front-facing cameras
