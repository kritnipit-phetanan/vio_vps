# Camera View Mode Guide

## คำถามที่พบบ่อย (FAQ)

### Q: ทำไมต้องมี `--camera_view` parameter?
**A:** เพราะ optical flow behavior แตกต่างกันตามมุมมองกล้อง:
- **Nadir (ก้ม)**: motion ส่วนใหญ่อยู่ที่ Z-axis (up/down)
- **Front (ตรง)**: motion ส่วนใหญ่อยู่ที่ X-Y plane (forward/sideways)

### Q: `--camera_view` มีผลกับอะไรบ้าง?
**A:** มีผลเฉพาะ **VIO (Visual Inertial Odometry)** ส่วนเดียว:
- Camera extrinsics (BODY_T_CAM)
- VIO uncertainty weights (sigma_scale_xy, sigma_scale_z)
- Nadir alignment threshold
- Feature tracking parameters (min_parallax, max_corners)

**ไม่มีผลกับ:**
- ❌ IMU propagation
- ❌ VPS position updates
- ❌ DEM height updates
- ❌ MSCKF updates

### Q: ต้องใช้ DEM/VPS กับ front camera ไหม?
**A:** ใช่! **DEM และ VPS ยังใช้ได้ปกติทุก camera view**

```bash
# ✅ ถูกต้อง - ใช้ DEM/VPS กับ front camera
python3 vio_vps.py --imu imu.csv --quarry quarry1.csv \
  --images_dir ./images --images_index ./index.csv \
  --dem DSM.tif --z_state agl --vps vps_result.csv \
  --camera_view front

# ✅ ถูกต้อง - ใช้ DEM/VPS กับ nadir camera
python3 vio_vps.py --imu imu.csv --quarry quarry1.csv \
  --images_dir ./images --images_index ./index.csv \
  --dem DSM.tif --z_state agl --vps vps_result.csv \
  --camera_view nadir
```

### Q: เลือก camera view ยังไง?
**A:** ดูจาก **การติดตั้งกล้อง**:

| Camera Mount | View Mode | Use Case |
|-------------|-----------|----------|
| มองลงพื้น (Z-axis ชี้ลง) | `nadir` | UAV/Drone มาตรฐาน |
| มองตรงไปข้างหน้า (Z-axis ชี้หน้า) | `front` | รถยนต์, Robot, AGV |
| มองด้านข้าง/เอียง | `side` | Oblique imagery |

---

## ตัวอย่างการใช้งาน

### 1. UAV/Drone (Nadir Camera)
```bash
python3 vio_vps.py \
  --imu imu.csv \
  --quarry quarry1.csv \
  --images_dir ./camera__image_mono/images \
  --images_index ./camera__image_mono/images_index.csv \
  --dem DSM_10_N47_00_W054_00_AOI.tif \
  --z_state agl \
  --vps vps_result.csv \
  --camera_view nadir \
  --output out_nadir
```

**คุณสมบัติ:**
- ใช้ VZ-only updates (เน้น vertical motion)
- XY uncertainty สูง (lateral motion น้อย)
- Z uncertainty ต่ำ (depth info ดี)
- Nadir threshold: 30° (strict)

---

### 2. รถยนต์/Robot (Front Camera)
```bash
python3 vio_vps.py \
  --imu imu.csv \
  --quarry quarry1.csv \
  --images_dir ./front_camera/images \
  --images_index ./front_camera/index.csv \
  --dem DSM_10_N47_00_W054_00_AOI.tif \
  --z_state agl \
  --vps vps_result.csv \
  --camera_view front \
  --output out_front
```

**คุณสมบัติ:**
- ใช้ full 3D velocity updates
- XY uncertainty ต่ำ (lateral motion ดี)
- Z uncertainty สูง (depth ambiguity)
- Nadir threshold: 60° (relaxed)

---

### 3. Oblique/Side Camera
```bash
python3 vio_vps.py \
  --imu imu.csv \
  --quarry quarry1.csv \
  --images_dir ./side_camera/images \
  --images_index ./side_camera/index.csv \
  --dem DSM_10_N47_00_W054_00_AOI.tif \
  --z_state agl \
  --vps vps_result.csv \
  --camera_view side \
  --output out_side
```

**คุณสมบัติ:**
- ใช้ full 3D velocity updates
- Balanced XY/Z uncertainty
- Nadir threshold: 60°

---

## Technical Comparison

| Parameter | Nadir | Front | Side |
|-----------|-------|-------|------|
| **Extrinsics** | BODY_T_CAMDOWN | BODY_T_CAMFRONT | BODY_T_CAMSIDE |
| **Nadir Threshold** | 30° (strict) | 60° (relaxed) | 60° |
| **Use VZ-only** | Yes (if aligned) | No (always 3D) | No |
| **σ_xy scale** | 1.5× (higher) | 0.8× (lower) | 1.0× |
| **σ_z scale** | 0.7× (lower) | 1.5× (higher) | 1.2× |
| **Min Parallax** | 15 px | 8 px | 10 px |
| **Max Corners** | 1500 | 2000 | 2000 |

---

## Sensor Fusion Summary

```
┌─────────────────────────────────────────────┐
│          Multi-Sensor Fusion System         │
├─────────────────────────────────────────────┤
│                                             │
│  IMU (400 Hz)  ──► Predict ─────────┐      │
│                                      │      │
│  VIO (20 Hz)   ──► Update Velocity ─┤      │
│  └─ Affected by --camera_view       │      │
│                                      ├──► EKF State
│  VPS (variable) ─► Update Position ─┤      │
│  └─ NOT affected by camera_view     │      │
│                                      │      │
│  DEM (static)  ──► Update Height  ──┘      │
│  └─ NOT affected by camera_view            │
│                                             │
└─────────────────────────────────────────────┘
```

---

## สรุป

### ✅ สิ่งที่ควรจำ:
1. **DEM และ VPS ใช้ได้ทุก camera view** - ไม่จำเป็นต้องเอาออก
2. `--camera_view` มีผลเฉพาะ **VIO processing** เท่านั้น
3. เลือก view mode ตาม**การติดตั้งกล้อง**จริง
4. IMU/VPS/DEM ทำงานเหมือนเดิมทุก mode

### ❌ ข้อควรระวัง:
1. ห้ามใช้ `front` กับกล้องที่ติดตั้งแบบ nadir (ผิดพลาด!)
2. Extrinsics ต้องตรงกับ camera mount จริง
3. ถ้า calibration ผิด → velocity estimate ผิด

---

## Need Help?

ถ้ายังไม่แน่ใจ ให้ดูที่ **camera orientation**:
- กล้องชี้ลง → `nadir`
- กล้องชี้ไปข้างหน้า → `front`
- กล้องชี้ด้านข้าง → `side`

หรือทดสอบทั้ง 3 modes แล้วดูว่า mode ไหนให้ trajectory ที่ดีที่สุด!
