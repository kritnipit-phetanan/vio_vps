# VIO/VPS Nadir Camera Examples

สคริปต์ตัวอย่างสำหรับรัน VIO/VPS กับกล้อง Nadir (มุมมองลงล่าง) เหมาะสำหรับ UAV/Drone

## 📋 การใช้งาน

### 1. IMU เท่านั้น (ทดสอบพื้นฐาน)
```bash
./run_nadir_imu_only.sh
```
- ใช้เฉพาะ IMU propagation
- เหมาะสำหรับทดสอบว่า IMU integration ทำงานถูกต้อง

### 2. VIO (Visual-Inertial Odometry)
```bash
./run_nadir_vio.sh
```
- IMU + Camera (visual odometry)
- เหมาะสำหรับพื้นที่ในร่มหรือไม่มี GPS

### 3. Full Fusion (VIO + VPS + DEM)
```bash
./run_nadir_full.sh
```
- IMU + Camera + VPS (GPS-based positioning) + DEM (terrain)
- ระบบเต็มรูปแบบสำหรับการบินกลางแจ้ง

### 4. ตัวอย่างทั่วไป (ปรับแต่งได้)
```bash
./run_nadir_example.sh
```
- Template พร้อม comment อธิบาย
- แก้ไข path ตามข้อมูลของคุณ

## 🔧 การตั้งค่า

แก้ไขไฟล์ `.sh` และเปลี่ยน path เหล่านี้:

```bash
IMU_CSV="path/to/imu.csv"                    # ข้อมูล IMU
QUARRY_CSV="path/to/quarry1.csv"             # ตำแหน่งเริ่มต้น (lat/lon/alt)
IMAGES_DIR="path/to/images"                  # โฟลเดอร์รูปภาพ
IMAGES_INDEX="path/to/images_index.csv"      # Index ของรูป (timestamp)
VPS_CSV="path/to/vps_result.csv"             # ผล VPS (GPS)
DEM_TIFF="path/to/DSM_*.tif"                 # DEM terrain file
```

## 📊 Output Files

หลังจากรันเสร็จ จะได้ไฟล์เหล่านี้ใน output directory:

1. **pose.csv** - ข้อมูล trajectory หลัก
   - Timestamp, Position (px,py,pz), Velocity (vx,vy,vz)
   - lat/lon, AGL (altitude above ground)
   - VO increments (dx,dy,dz,roll,pitch,yaw)

2. **inference_log.csv** - Performance metrics
   - Inference time, FPS

3. **vo_debug.csv** - Visual odometry debugging
   - Number of inliers, rotation angle, alignment
   - VZ-only mode flags

4. **state_debug.csv** - Full EKF state variables
   - Position, velocity, acceleration (world frame)
   - DEM, AGL, MSL values

## 🎯 Nadir Camera Settings

การตั้งค่าเฉพาะสำหรับกล้องมุม Nadir:

```python
--camera_view nadir          # โหมดกล้องลงล่าง
--img_w 1140                 # ความกว้างรูป
--img_h 1080                 # ความสูงรูป
--z_state msl                # ใช้ MSL (mean sea level) หรือ agl
```

### ข้อดีของ Nadir mode:
- ✅ Depth information ดี (แกน Z)
- ✅ เหมาะกับการวัดความสูง (AGL)
- ✅ Track features ได้ดีเมื่อมี parallax จาก altitude change
- ⚠️ XY motion อาจไม่แม่นเท่า forward camera (ต้องพึ่ง VPS/GPS)

### Adaptive uncertainty:
- VZ measurements มี lower uncertainty (`sigma_scale_z: 0.7`)
- VX/VY measurements มี higher uncertainty (`sigma_scale_xy: 1.5`)
- Nadir alignment threshold: 30° (strict)

## 🚀 Quick Start

1. แก้ไข path ในสคริปต์ที่เลือก
2. รัน:
   ```bash
   ./run_nadir_full.sh
   ```
3. ตรวจสอบ output ใน `output_nadir_full/`
4. Visualize trajectory จาก `pose.csv`

## 📖 Command Line Options

ดู options ทั้งหมด:
```bash
python3 vio_vps.py --help
```

### Main arguments:
- `--imu` - IMU CSV file (required)
- `--quarry` - Initial position CSV (required)
- `--output` - Output directory
- `--images_dir` - Image folder
- `--images_index` - Image index CSV
- `--vps` - VPS result CSV
- `--dem` - DEM GeoTIFF file
- `--camera_view` - Camera mode: nadir/front/side
- `--z_state` - Height mode: msl/agl

## 🐛 Debugging

หากพบปัญหา ให้ตรวจสอบ:

1. **IMU drift**: ดู `state_debug.csv` - ตรวจ velocity และ position
2. **VIO tracking**: ดู `vo_debug.csv` - ตรวจ num_inliers และ alignment
3. **Console output**: มี DEBUG prints แสดง:
   - `[DEBUG][IMU]` - IMU propagation
   - `[DEBUG][VPS]` - VPS updates
   - `[DEBUG][VIO]` - Visual odometry
   - `[DEBUG][DEM]` - DEM/height updates

## 💡 Tips

- เริ่มจาก `run_nadir_imu_only.sh` เพื่อทดสอบ IMU ก่อน
- เพิ่ม VIO เมื่อ IMU ทำงานดีแล้ว
- เพิ่ม VPS/DEM สำหรับ absolute positioning
- ปรับ parameters ใน `vio_vps.py` ตามความต้องการ

## 📞 Support

หากมีคำถาม ตรวจสอบ:
- Debug logs in console
- Output CSV files
- Code comments in `vio_vps.py`
