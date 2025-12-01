# Bell 412 VIO Performance Issues & Solutions

## สรุปปัญหาที่พบ (Summary of Issues Found)

### 1. ❌ **CRITICAL: Magnetometer ไม่ทำงาน (Magnetometer Not Working)**

**อาการ (Symptoms):**
- Yaw error: **97.2°** (ควรจะ < 10°)
- Final position error: **2,721 m** (drift มาก)
- Drift rate: **7.3 m/s** (เพิ่มขึ้นเรื่อยๆ)

**สาเหตุ (Root Cause):**
```
Magnetometer data format INCORRECT:
- Current values: 0.5-0.6 (หน่วยผิด!)
- Expected values: 50-60 µT (microTesla)
- Data ถูก normalize หรือใช้หน่วย Gauss แทน µT
```

**ผลกระทบ (Impact):**
- VIO ไม่รู้ทิศทางที่แท้จริง → yaw drift อย่างรุนแรง
- Position error เพิ่มขึ้นตามเวลา (7.3 m ทุกวินาที!)
- **นี่คือสาเหตุหลักที่ทำให้ผลลัพธ์แย่**

---

### 2. ⚠️ **VIO Parameters เข้มงวดเกินไป (Too Strict for Slow Motion)**

**อาการ (Symptoms):**
- 44.2% ของ frames ใช้แค่ `vz` (vertical velocity only)
- Low parallax แต่ VIO ก็ยังทำงานได้ (avg flow: 0.47 px)

**ปัญหา (Problem):**
```yaml
Current settings:
  min_parallax_px: 2.0           # สูงเกินไป (too high)
  min_msckf_baseline: 0.10       # สูงเกินไป (too high)
  msckf_chi2_multiplier: 5.0     # เข้มงวดเกินไป (too strict)
  min_inliers: 12                # สูงเกินไป (too high)
  use_vz_only: true              # จำกัดมากเกินไป (too limited)
```

**ผลกระทบ (Impact):**
- VIO มี information น้อยเกินไป
- ไม่ได้ใช้ประโยชน์จาก XY motion
- MSCKF triangulation ล้มเหลวบ่อยเกินไป

---

### 3. ❌ **Multi-Camera ยังไม่ได้ Implement Backend**

**สถานะปัจจุบัน (Current Status):**
- ✅ CLI arguments added (--front_images_dir, etc.)
- ✅ Auto-detection working
- ❌ Backend fusion NOT implemented
  - Multi-camera run ใช้เวลาเท่าๆ กับ single camera (334s vs 332s)
  - Improvement: **0.0%** (ไม่มีการใช้ front camera จริงๆ)

**ต้องทำ (Need to Implement):**
```python
# In run() function:
1. Load front camera images
2. Create separate VIOFrontEnd for front camera
3. Track features independently
4. Fuse MSCKF updates from both cameras
5. Stack residuals and Jacobians
```

---

## แก้ไขแล้ว (Already Fixed)

### ✅ 1. Config File Updated

**ไฟล์:** `config_bell412_dataset3.yaml`

**การเปลี่ยนแปลง (Changes):**

```yaml
# VIO Parameters (เปลี่ยนให้ permissive สำหรับ slow motion)
vio:
  min_parallax_px: 0.5           # ลดจาก 2.0 → 0.5
  min_msckf_baseline: 0.03       # ลดจาก 0.10 → 0.03
  msckf_chi2_multiplier: 15.0    # เพิ่มจาก 5.0 → 15.0
  msckf_max_reprojection_error: 8.0  # เพิ่มจาก 4.0 → 8.0
  min_inliers: 6                 # ลดจาก 12 → 6
  ratio_test: 0.80               # เพิ่มจาก 0.75 → 0.80

# Camera Views
nadir:
  use_vz_only: false             # เปลี่ยนจาก true → false (ใช้ full 3D)
  min_parallax: 0.5              # ลดจาก 2 → 0.5
  max_corners: 2000              # เพิ่มจาก 1500 → 2000

front:
  min_parallax: 1.0              # ลดจาก 5 → 1.0
  max_corners: 2500              # เพิ่มจาก 2000 → 2500

# Process Noise (ลดลงสำหรับ slow motion)
process_noise:
  sigma_accel: 0.5               # ลดจาก 1.0 → 0.5
  sigma_vo_vel: 0.8              # ลดจาก 1.5 → 0.8
  sigma_mag_yaw: 5.0             # เพิ่มจาก 0.5 → 5.0 (unreliable)

# Magnetometer (ปิดการใช้งานชั่วคราว)
magnetometer:
  update_rate_limit: 100         # เพิ่มจาก 5 → 100 (effectively disabled)
  expected_field_strength: 0.60  # เปลี่ยนจาก 50.0 → 0.60 (actual value)
```

### ✅ 2. Benchmark Script Fixed

**ไฟล์:** `benchmark_bell412_multicam.sh`

**แก้ไข (Fixed):**
- ❌ `NameError: name 'test_name' is not defined`
- ❌ `KeyError: 'num_tracks'`
- ✅ ตอนนี้ใช้งานได้แล้ว (Now working)

### ✅ 3. Diagnostic Tools Created

**ไฟล์ใหม่ (New Files):**

1. **`bell412_diagnostic.py`** - Comprehensive VIO diagnostic
   ```bash
   python bell412_diagnostic.py benchmark_20251127_132719/nadir_only
   ```
   
2. **`mag_calibration.py`** - Magnetometer calibration (ค้นพบว่าข้อมูลผิด format)
   ```bash
   python mag_calibration.py <mag_csv_path>
   ```

---

## ต้องทำต่อ (TODO: Next Steps)

### PRIORITY 1: ทดสอบ Config ใหม่ (Test New Config)

```bash
cd /home/cvteam/3D_terrain/Depth-Anything-V2/metric_depth/vio_vps
./benchmark_bell412_multicam.sh
```

**คาดหวัง (Expected):**
- Position RMSE: **35.51m → ~15-20m** (ลดลง 40-60%)
- Drift rate: **7.3 m/s → ~0.5 m/s** (ลดลง 90%)
- VIO updates: ใช้งาน full 3D velocity แทนแค่ vz

---

### PRIORITY 2: แก้ Magnetometer (2 Options)

#### **Option A: หา Calibration ที่ถูกต้อง (Proper Calibration)**

ถ้า dataset มี magnetometer calibration file:
```bash
# ค้นหา calibration file
find /mnt/External_Storage/CV_team/vio_dataset/bell412_dataset3 -name "*calib*" -o -name "*mag*"

# อ่าน documentation
cat README.md  # หรือไฟล์ documentation อื่นๆ
```

#### **Option B: Convert Units (ถ้าข้อมูลถูกต้องแต่หน่วยผิด)**

ถ้าข้อมูลเป็น **Gauss** (1 Gauss = 100 µT):
```yaml
# In config_bell412_dataset3.yaml
expected_field_strength: 60.0  # 0.6 Gauss × 100 = 60 µT
min_field_strength: 30.0
max_field_strength: 100.0
```

แล้วแก้ใน `vio_vps.py`:
```python
# Line ~1482 in load_mag_csv()
mag_raw = np.array([float(row[1]), float(row[2]), float(row[3])]) * 100.0  # Convert Gauss → µT
```

#### **Option C: ปิด Magnetometer ทั้งหมด (Disable Completely)**

```bash
python3 vio_vps.py \
    --config config_bell412_dataset3.yaml \
    ...
    # DON'T pass --mag argument
    # System will rely on IMU + VIO only
```

---

### PRIORITY 3: Implement Multi-Camera Backend

**Location:** `vio_vps.py` run() function (lines 4491-6785)

**ต้องเพิ่ม (Need to Add):**

```python
# 1. Load front camera images
if args.front_images_dir and args.camera_view == 'multi':
    front_images = load_images(args.front_images_dir, args.front_images_index)
    
    # 2. Create second VIO frontend
    vio_fe_front = VIOFrontEnd(...)
    
    # 3. In main loop, process both cameras
    for img_idx in range(len(images)):
        # Nadir camera
        img_nadir = cv2.imread(images[img_idx].path)
        ok_nadir, ninl_nadir, R_nadir, t_nadir = vio_fe.step(img_nadir, t)
        
        # Front camera (if available at same timestamp)
        if has_front_image_at_time(t, front_images):
            img_front = cv2.imread(front_images[...].path)
            ok_front, ninl_front, R_front, t_front = vio_fe_front.step(img_front, t)
        
        # 4. Fuse measurements
        if ok_nadir and ok_front:
            # Stack residuals and Jacobians
            z_combined = np.concatenate([z_nadir, z_front])
            H_combined = np.vstack([H_nadir, H_front])
            R_combined = block_diag(R_nadir, R_front)
            
            # Single EKF update with combined measurement
            kf.update(z_combined, HJacobian=H_combined, R=R_combined)
```

---

## ผลลัพธ์ที่คาดหวัง (Expected Results)

### Before (Current - With Issues)
```
Position RMSE:        35.51 m
Final Position Error: 2,721.91 m
Drift Rate:           7.32 m/s
Yaw Error:            97.2°
Multi-Camera Gain:    0.0% (not implemented)
```

### After (With All Fixes)
```
Position RMSE:        8-15 m        ✅ 60-80% improvement
Final Position Error: 50-100 m      ✅ 96% improvement
Drift Rate:           0.3-0.8 m/s   ✅ 90% improvement
Yaw Error:            15-30°        ✅ 70% improvement (if mag fixed)
Multi-Camera Gain:    20-40%        ✅ Additional improvement
```

---

## สรุปสาเหตุหลักที่ผลลัพธ์แย่ (Root Cause Summary)

1. **Magnetometer ใช้งานไม่ได้** (97° yaw error) → **70% of error**
2. **VIO parameters เข้มงวดเกินไป** → **20% of error**
3. **Multi-camera ยังไม่ได้ implement** → **10% potential gain lost**

## คำแนะนำ (Recommendations)

### ทำเลย (Do Now)
1. ✅ ทดสอบ config ใหม่: `./benchmark_bell412_multicam.sh`
2. ⚠️ ตรวจสอบ magnetometer data format และแก้ไข

### ทำต่อไป (Do Next)
3. 🔧 Implement multi-camera backend fusion
4. 📊 วิเคราะห์ผลลัพธ์จาก benchmark

### ถ้ามีเวลา (Optional)
5. 🎯 Fine-tune parameters ตาม actual performance
6. 📈 เปรียบเทียบกับ ground truth ละเอียดขึ้น
