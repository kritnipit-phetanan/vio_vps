## Codex Plan Update (Retain C0–C3, Replace C4–C6 with Logic-First Apply Architecture)

### Summary
- สถานะคงเดิม: `C0 PASS`, `C1 PASS`, `C2 PASS(core)`, `C2-H parallel`, `C3 winner = outputs/benchmark_modular_20260309_042205`.
- ปรับแผนโดย **ตัด C4 แบบ threshold-only ที่ ROI ต่ำ** ออกจาก main track และแทนด้วย 6 งาน logic-first:
1. Correction Supervisor (PROPOSE/PROBATION/COMMIT)
2. Time-aligned apply (rewind/repropagate)
3. XY/Yaw decouple authority
4. Pseudo-measurement apply path (แทน raw offset apply)
5. Source reliability model (rolling)
6. Telemetry funnel แบบ end-to-end
- Baseline สำหรับ C4–C6 ใหม่: `outputs/benchmark_modular_20260309_042205`.

### สิ่งที่ “เอาออก” จาก C4–C6 เดิม (main track)
- เอา `C4.1 residual_xy_m tighten`, `C4.2 max_step_dp_xy_m tighten`, `C4.3 min_weight-only` ออกจากเส้นหลัก เพราะเป็น parameter tightening ที่แก้ throughput/trajectory coupling ไม่ตรงจุด.
- คงค่า knob เดิมไว้เป็น fallback only (ไม่ใช้เป็น success path หลัก).
- C5/C6 เดิมที่มีอยู่แล้ว (contract strict + no-op guard) ให้คงไว้ แต่รวมเข้าโครงใหม่ด้านล่าง.

### Revised Phases (Decision-Complete)

#### C4 — Absolute Correction Apply Re-architecture (แทน C4 เดิม)
- `C4.1 Supervisor Lane`
  - เพิ่ม state machine เดียวสำหรับ correction: `PROPOSE -> PROBATION -> COMMIT | REJECT`.
  - correction ใหม่เข้า `PROBATION` ก่อนเสมอ (ไม่ commit เต็มทันที).
  - probation ใช้ fixed window 2 camera ticks; commit เมื่อผ่านทั้ง motion-consistency + innovation-improvement.
  - ถ้าไม่ผ่านให้ `HINT_ONLY` และเพิ่ม reject reason แบบ deterministic.
- `C4.2 Time-Aligned Apply`
  - apply ที่ `t_ref` จริง โดย rewind state จาก ring buffer (state + imu delta), apply correction, แล้ว repropagate กลับ now.
  - ไม่อนุญาต direct-now apply สำหรับ correction ที่ age เกิน threshold.
- `C4.3 XY/Yaw Decouple`
  - authority แยก: XY ผ่านได้แม้ yaw ถูก freeze.
  - yaw reject ห้ามลาก XY lane ให้ reject ตาม.
  - high-speed/unstable window: yaw อยู่ bounded participation mode เท่านั้น.
- Pass C4
  - `err3d_final` ดีขึ้น >= 10% เทียบ C3 winner
  - `backend_apply_count` ไม่ลดเกิน 20% จาก C3 winner
  - `backend_apply_dp_xy_p95` ลด >= 15% จาก C3 winner
  - global guards ผ่านครบ

#### C5 — Contract + Reliability + Pseudo-Measurement Core
- `C5.1 Contract Single Path`
  - ใช้ `BackendCorrectionContractV1` validator path เดียวก่อน apply เสมอ
  - reason taxonomy เดียว: `contract_violation`, `stale_reject`, `kinematic_reject`, `snap_reject`, `quality_reject`
- `C5.2 Pseudo-Measurement Apply`
  - เปลี่ยนจาก raw dp/dyaw inject เป็น weighted pseudo-measurement update ใช้ covariance/residual ที่ normalize แล้ว
  - robust/switchable ใช้ใน path เดียว (ไม่ซ้ำหลายจุด)
- `C5.3 Source Reliability Model`
  - rolling reliability ต่อ source (`VPS`, `LOOP`, `BACKEND`, `MAG`)
  - reliability ต่ำต่อเนื่อง -> auto downweight/temporary quarantine
  - reliability สูงต่อเนื่อง -> อนุญาต commit escalation
- Pass C5
  - `backend_contract_violation_count == 0`
  - `backend_snap_reject_count == 0`
  - `err3d_mean` ดีขึ้น >= 8% เทียบ C4 winner
  - `err3d_final` ดีขึ้น >= 10% เทียบ C4 winner
  - global guards ผ่านครบ

#### C6 — Anti-Tangle Consolidation (No-Behavior-Drift)
- source-of-truth เหลือ 1 จุดสำหรับ classify/apply/reject ใน correction path.
- alias compatibility อยู่ที่ config parse boundary จุดเดียว; downstream ห้ามมี duplicate decision branch.
- orchestrator no-op guard คงไว้และเพิ่ม check “runtime-effective knob” (changed key แต่ไม่กระทบ runtime = FAIL).
- cleanup reason mapping ให้ใช้ dictionary กลางชุดเดียวใน summary + trace.
- Pass C6
  - behavior drift จาก C5 winner <= 3%
  - ไม่มี branch ซ้ำที่ให้ผลขัดกันกับ input เดียว
  - targeted tests ผ่านครบ

### Important Interface / Output Changes
- เพิ่ม type/contract:
  - `CorrectionDecisionState`: `PROPOSE/PROBATION/COMMIT/REJECT/HINT_ONLY`
  - `ApplyDecisionRecord`: include `decision_state`, `reason`, `age_sec`, `t_ref`, `applied_dp_xy`, `applied_dyaw`
- telemetry ใหม่ (mandatory):
  - `backend_proposed_count`
  - `backend_probation_count`
  - `backend_probation_commit_count`
  - `backend_probation_reject_count`
  - `backend_time_aligned_apply_count`
  - `backend_source_reliability_<source>_p50`
  - `backend_reject_<reason>_count` (single taxonomy)
- funnel trace:
  - `backend_apply_trace.csv` ต้องมี `source`, `state`, `reason`, `quality`, `residual_xy`, `dp_xy_in`, `dp_xy_applied`, `age_sec`, `t_ref`, `time_aligned_used`

### Test Cases and Scenarios
- Unit tests
  - supervisor transition correctness (`PROPOSE->PROBATION->COMMIT/REJECT`)
  - time-aligned rewind/repropagate deterministic check
  - XY pass while yaw freeze (decouple rule)
  - contract single-path reject mapping
  - source reliability downweight/quarantine behavior
  - no-op guard runtime-effective check
- Integration sequence (one-knob strict)
  - Run C4.1, compare vs C3 winner
  - Run C4.2, compare vs C4.1 winner
  - Run C4.3, compare vs C4.2 winner
  - Run C5, compare vs C4 winner
  - Run C6, compare vs C5 winner
- Rollback rules (คงเดิม)
  - `cov_large_rate > 0`
  - `pmax_max > 1e6`
  - `policy_conflict_count > 0`
  - `backend_contract_violation_count > 0`
  - `err3d_mean regress > 15%`
  - `err3d_final regress > 20%`

### Assumptions and Defaults
- accuracy-first branch คงเดิม; heading quartet เป็น monitor-only.
- `matcher_mode=orb` คงเดิมจนจบ C6.
- C2-H ยังเป็น parallel track และไม่ block C4–C6.
- C3 winner (`20260309_042205`) เป็น baseline เดียวสำหรับการตัดสินใจ C4–C6.
- threshold-only tuning ที่เคยใช้ใน C4 เดิมเป็น fallback path เท่านั้น ไม่ใช่ pass path หลัก.
