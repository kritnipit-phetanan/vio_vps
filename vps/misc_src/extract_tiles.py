#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use to extract all tiles from mbtiles file
"""
import sqlite3
import os

def extract_all_tiles_to_folder(mbtiles_path, output_folder):
    # 1. ตรวจสอบและสร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"สร้างโฟลเดอร์ใหม่ที่: {output_folder}")
    
    # 2. เชื่อมต่อฐานข้อมูล
    if not os.path.exists(mbtiles_path):
        print(f"Error: ไม่พบไฟล์ .mbtiles ที่ {mbtiles_path}")
        return

    conn = sqlite3.connect(mbtiles_path)
    cursor = conn.cursor()
    
    # 3. ดึงข้อมูลทุกแถวออกมา (เลือก Zoom, X, Y และข้อมูลภาพ)
    print("กำลังอ่านข้อมูลจากฐานข้อมูล...")
    cursor.execute("SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles")
    all_tiles = cursor.fetchall()
    
    total_count = len(all_tiles)
    print(f"พบจำนวนทั้งหมด {total_count} รูป กำลังบันทึกไฟล์...")
    
    saved_count = 0
    for zoom, col, tms_row, tile_blob in all_tiles:
        # 4. แปลงแกน Y จาก TMS (ที่เก็บใน DB) เป็น Google XYZ (มาตรฐานทั่วไป)
        # เพื่อให้ชื่อไฟล์ตรงกับพิกัดที่เราเข้าใจง่ายๆ
        google_row = (2**zoom) - 1 - tms_row
        
        # 5. ตั้งชื่อไฟล์: tile_Z_X_Y.jpg
        # ตัวอย่าง: tile_19_15432_24531.jpg
        filename = f"tile_{zoom}_{col}_{google_row}.jpg"
        file_path = os.path.join(output_folder, filename)
        
        # 6. เขียนข้อมูล Binary ลงไฟล์โดยตรง (เร็วกว่าใช้ OpenCV)
        try:
            with open(file_path, 'wb') as f:
                f.write(tile_blob)
            saved_count += 1
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            
    conn.close()
    print(f"\nเสร็จสิ้น! บันทึกไฟล์เรียบร้อยแล้ว {saved_count}/{total_count} รูป")
    print(f"เช็คไฟล์ได้ที่: {output_folder}")

# --- ส่วนเรียกใช้งาน ---
if __name__ == "__main__":
    # ระบุ Path ไฟล์ .mbtiles ของคุณ
    mbtiles_file = "mission.mbtiles"  # <-- เช็ค Path ให้ถูกนะครับ
    
    # ระบุ Path โฟลเดอร์ปลายทางที่คุณต้องการ
    target_folder = "/Users/france/Downloads/vio_vps/vps/extracted_mbtiles"
    
    extract_all_tiles_to_folder(mbtiles_file, target_folder)