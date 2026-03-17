## C4–C6 Commit-Continuity Plan (Mean-First)  
Baseline lock: `outputs/benchmark_modular_20260311_004927`  
Priority lock: `err3d_mean` first, with `err3d_final` guarded

### Summary
- ปัญหาใหญ่สุดตอนนี้คือ correction เข้าถึง `PROPOSE` ได้ แต่ `COMMIT` เกิดเกือบเฉพาะช่วงต้นไฟลต์ (Q1) แล้ว Q2–Q4 กลายเป็น `HINT_ONLY` จน drift สะสม
- แผนนี้ **ไม่เน้น threshold tightening** แต่แก้ architecture ให้ commit ต่อเนื่องทั้งไฟลต์แบบ bounded/deterministic
- ลำดับบังคับ (one-knob strict):  
  `C4.1 -> C4.2 -> C5.1 -> C5.2 -> C6`  
  (`C2-H` คงเป็น parallel track และไม่แตะ VPS/heading path ใน track นั้น)

## Key Changes

### C4.1 — Probation Continuity Controller (Backend, logic-first)
- เพิ่ม continuity mode ใน supervisor: เมื่อ `no_commit_streak` เกิน trigger และเคสผ่าน direction+magnitude consistency ให้ `quality_reject` บางส่วนถูก route เป็น `bounded COMMIT` แทน `HINT_ONLY`
- bounded commit นี้บังคับ energy cap ตายตัว:
  - cap ต่อ step (`dp_xy`)
  - blend ขั้นต่ำ
  - yaw cap แยก (หรือ freeze yaw ใน lane นี้)
- ห้าม bypass contract/stale guards เดิม  
- เป้าหมาย: เกิด COMMIT ใน Q2–Q4 โดยไม่เกิด burst

### C4.2 — Probation Evidence Stabilization (ไม่แตะ VPS)
- ใช้ monotonic residual evidence 2 ticks แบบ deterministic ใน probation commit
- ลด over-defer จาก `probation_deferred` โดยปรับ replace logic ให้ยอมรับ candidate ใหม่เมื่อมี evidence ดีกว่าจริง (ไม่ใช่ score delta แข็งอย่างเดียว)
- รักษา single source-of-truth ที่ `schedule_backend_correction` + `_process_backend_probation` เท่านั้น

### C5.1 — VPS Local-First Continuity Budget
- ใน `vps_runner` แยก budget เป็น local-first lane ก่อนเสมอ แล้วค่อยใช้ส่วนเหลือกับ global
- ลด fail streak ปลายไฟลต์ด้วย continuity probe ที่ deterministic
- ยังไม่เปลี่ยน matcher หลัก/heading policy ในขั้นนี้

### C5.2 — Anchor Policy for Absolute Stability
- lock anchor เฉพาะ strict/high-confidence เท่านั้น
- failsoft ใช้ได้แค่ bounded correction lane (ไม่ใช้เป็น anchor)
- เป้าหมายคือหยุดแกนอคติ E/N ที่สลับลอยตามช่วงไฟลต์

### C6 — Anti-Tangle Consolidation
- รวม apply/reject authority ให้เหลือ path เดียวใน backend correction path
- ย้าย alias compatibility ไป parse boundary ของ config จุดเดียว
- เพิ่ม orchestrator runtime-effective noop guard ให้ phase ที่ “เปลี่ยน key แต่ไม่กระทบ runtime” ถูกตัดเป็น fail

## Important Interfaces / Outputs
- เพิ่ม telemetry ใน `benchmark_health_summary.csv`:
  - `backend_no_commit_streak_max`
  - `backend_probation_deferred_count`
  - `backend_continuity_bounded_commit_count`
  - `backend_commit_q1_count`, `backend_commit_q2_count`, `backend_commit_q3_count`, `backend_commit_q4_count`
  - `vps_local_first_attempt_count`, `vps_local_first_success_count`, `vps_global_probe_count`
- เพิ่ม field ใน `backend_apply_trace.csv`:
  - `continuity_mode_used`, `streak_at_decision`, `q_bucket`
- reason taxonomy กลางเดียว:
  - `contract_violation`, `stale_reject`, `kinematic_reject`, `quality_reject`, `probation_deferred`, `continuity_bounded_commit`, `snap_reject`

## Test Plan (run between steps)
1. **After C4.1**
   - Unit: continuity route (`quality_reject -> bounded COMMIT`) ต้องเกิดเฉพาะเมื่อ consistency pass
   - Integration full run 1 รอบ เทียบ baseline
2. **After C4.2**
   - Unit: monotonic-evidence and replace policy
   - Integration full run 1 รอบ
3. **After C5.1**
   - Unit: local-first budget split / deterministic probe
   - Integration full run 1 รอบ
4. **After C5.2**
   - Unit: strict-anchor vs failsoft-bounded lane separation
   - Integration full run 1 รอบ
5. **After C6**
   - No-behavior-drift validation + phase noop guard validation

### Pass / Rollback Guards
- Global hard guards:
  - `cov_large_rate == 0`
  - `pmax_max <= 1e6`
  - `policy_conflict_count == 0`
  - `backend_contract_violation_count == 0`
- C4–C6 functional guards:
  - Q2–Q4 ต้องมี `backend_commit_* > 0` อย่างน้อย 2 ไตรมาส
  - `backend_apply_dp_xy_p95` ลดลงจาก baseline อย่างน้อย 10%
  - `backend_apply_count` ไม่ลดเกิน 20% จาก baseline
- Rollback ทันทีเมื่อ:
  - `err3d_mean` regress > 15%
  - `err3d_final` regress > 20%

## Assumptions / Defaults
- โหมดยังเป็น accuracy-first, heading quartet monitor-only
- `matcher_mode=orb` คงเดิม
- baseline comparison หลักใช้ `outputs/benchmark_modular_20260311_004927` ตามที่ล็อกไว้
- เป้าหมาย `err3d_mean <= 100 m` ถือเป็น end-goal ระยะท้าย; รอบนี้เน้นแก้ continuity architecture ให้หยุด drift สะสมก่อน
