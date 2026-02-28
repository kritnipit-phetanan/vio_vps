"""Deterministic VPS position controller for position-first branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VpsMatchEvidence:
    """Normalized VPS evidence from matcher + state + policy."""

    t_cam: float
    frame_idx: int
    policy_mode: str
    health_key: str
    speed_m_s: float
    yaw_rate_deg_s: float
    abs_offset_vec: np.ndarray
    abs_offset_m: float
    vps_num_inliers: int
    vps_conf: float
    vps_reproj: float
    strict_quality_ok: bool
    failsoft_quality_ok: bool
    vps_match_failsoft: bool
    fs_min_inliers: int
    fs_min_conf: float
    fs_max_reproj: float


@dataclass(frozen=True)
class VpsPositionDecision:
    """Single decision output for VPS position apply chain."""

    quality_mode: str
    policy_reject_note: str
    hard_reject_note: str
    temporal_reject_note: str
    drift_recovery_candidate: bool
    drift_recovery_active: bool
    drift_recovery_note: str
    position_first_soft_active: bool
    position_first_soft_note: str
    position_first_direct_xy_candidate: bool
    position_first_direct_xy_note: str
    allow_direct_xy_apply: bool
    hint_quality: float


class VPSPositionController:
    """Central controller for VPS failsoft/direct-XY conversion."""

    def __init__(self, runner: Any, vio_service: Any):
        self.runner = runner
        self.vio_service = vio_service

    def _compute_hint_quality(
        self,
        evidence: VpsMatchEvidence,
        *,
        position_first_soft_active: bool,
        position_first_direct_xy_candidate: bool,
    ) -> float:
        hint_quality_scale = float(
            self.runner.global_config.get("VPS_XY_DRIFT_RECOVERY_HINT_QUALITY_SCALE", 0.65)
        )
        hint_quality = float(
            np.clip(
                (
                    0.45 * np.clip(float(evidence.vps_conf), 0.0, 1.0)
                    + 0.35 * np.clip(float(evidence.vps_num_inliers) / 80.0, 0.0, 1.0)
                    + 0.20 * np.clip(1.2 / max(float(evidence.vps_reproj), 0.1), 0.0, 1.0)
                )
                * max(0.1, hint_quality_scale),
                0.0,
                1.0,
            )
        )
        if bool(position_first_soft_active):
            hint_quality = float(
                np.clip(
                    hint_quality
                    * float(
                        self.runner.global_config.get(
                            "VPS_POSITION_FIRST_SOFT_HINT_QUALITY_SCALE",
                            0.85,
                        )
                    ),
                    0.0,
                    1.0,
                )
            )
        if bool(position_first_direct_xy_candidate):
            hint_quality = float(
                np.clip(
                    hint_quality
                    * float(
                        self.runner.global_config.get(
                            "VPS_POSITION_FIRST_DIRECT_XY_QUALITY_SCALE",
                            0.95,
                        )
                    ),
                    0.0,
                    1.0,
                )
            )
        return float(hint_quality)

    def _allow_force_failsoft_direct(self, evidence: VpsMatchEvidence, policy_reject_note: str) -> tuple[bool, str]:
        if not (
            bool(evidence.vps_match_failsoft)
            and bool(self.runner.global_config.get("POSITION_FIRST_LANE", False))
            and bool(
                self.runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_FORCE_FAILSOFT_ON_REJECT",
                    True,
                )
            )
            and not bool(policy_reject_note)
        ):
            return False, ""

        fs_force_max_offset = float(
            self.runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_FORCE_FAILSOFT_MAX_OFFSET_M", 180.0)
        )
        fs_force_min_inliers = int(
            round(
                self.runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_FORCE_FAILSOFT_MIN_INLIERS", 5)
            )
        )
        fs_force_max_reproj = float(
            self.runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_FORCE_FAILSOFT_MAX_REPROJ_ERROR", 1.4)
        )
        fs_force_max_speed = float(
            self.runner.global_config.get("VPS_POSITION_FIRST_DIRECT_XY_FORCE_FAILSOFT_MAX_SPEED_M_S", 95.0)
        )
        allow = bool(
            np.isfinite(float(evidence.abs_offset_m))
            and float(evidence.abs_offset_m) <= float(fs_force_max_offset)
            and int(evidence.vps_num_inliers) >= max(1, int(fs_force_min_inliers))
            and np.isfinite(float(evidence.vps_reproj))
            and float(evidence.vps_reproj) <= float(fs_force_max_reproj)
            and (
                (not np.isfinite(float(evidence.speed_m_s)))
                or float(evidence.speed_m_s) <= float(fs_force_max_speed)
            )
        )
        return allow, "position_first_direct_xy_force_failsoft" if allow else ""

    def _allow_failsoft_consensus_fallback(
        self,
        *,
        evidence: VpsMatchEvidence,
        position_first_direct_xy_note: str,
        policy_reject_note: str,
        hard_reject_note: str,
    ) -> tuple[bool, str]:
        if not (
            bool(evidence.vps_match_failsoft)
            and bool(self.runner.global_config.get("POSITION_FIRST_LANE", False))
            and bool(
                self.runner.global_config.get(
                    "VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_CONSENSUS_FALLBACK_ENABLE",
                    True,
                )
            )
            and not bool(policy_reject_note)
            and not bool(hard_reject_note)
        ):
            return False, ""

        note_l = str(position_first_direct_xy_note).strip().lower()
        if note_l not in ("consensus_warmup", "consensus_dev", "consensus_dir"):
            return False, ""

        max_offset = float(
            self.runner.global_config.get(
                "VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_CONSENSUS_FALLBACK_MAX_OFFSET_M",
                120.0,
            )
        )
        min_inliers = max(
            1,
            int(
                round(
                    self.runner.global_config.get(
                        "VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_CONSENSUS_FALLBACK_MIN_INLIERS",
                        5,
                    )
                )
            ),
        )
        min_conf = float(
            self.runner.global_config.get(
                "VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_CONSENSUS_FALLBACK_MIN_CONFIDENCE",
                max(0.06, float(evidence.fs_min_conf)),
            )
        )
        max_reproj = float(
            self.runner.global_config.get(
                "VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_CONSENSUS_FALLBACK_MAX_REPROJ_ERROR",
                max(1.4, float(evidence.fs_max_reproj)),
            )
        )
        max_speed = float(
            self.runner.global_config.get(
                "VPS_POSITION_FIRST_DIRECT_XY_FAILSOFT_CONSENSUS_FALLBACK_MAX_SPEED_M_S",
                85.0,
            )
        )
        allow = bool(
            np.isfinite(float(evidence.abs_offset_m))
            and float(evidence.abs_offset_m) <= float(max_offset)
            and int(evidence.vps_num_inliers) >= int(min_inliers)
            and np.isfinite(float(evidence.vps_conf))
            and float(evidence.vps_conf) >= float(min_conf)
            and np.isfinite(float(evidence.vps_reproj))
            and float(evidence.vps_reproj) <= float(max_reproj)
            and (
                (not np.isfinite(float(evidence.speed_m_s)))
                or float(evidence.speed_m_s) <= float(max_speed)
            )
        )
        if not allow:
            return False, ""
        return True, f"failsoft_consensus_soft_fallback:{note_l}"

    def decide(self, evidence: VpsMatchEvidence) -> VpsPositionDecision:
        runner = self.runner
        if bool(evidence.vps_match_failsoft):
            runner._vps_failsoft_matched_count = int(getattr(runner, "_vps_failsoft_matched_count", 0)) + 1

        quality_mode = "strict" if bool(evidence.strict_quality_ok) else ("failsoft" if bool(evidence.failsoft_quality_ok) else "reject")
        policy_reject_note = ""
        if str(evidence.policy_mode).upper() in ("HOLD", "SKIP"):
            quality_mode = "reject"
            policy_reject_note = f"policy_mode_{str(evidence.policy_mode).lower()}"

        drift_recovery_candidate, drift_recovery_note_candidate = self.vio_service._should_use_vps_xy_drift_recovery(
            t_cam=float(evidence.t_cam),
            abs_offset_m=float(evidence.abs_offset_m),
            speed_m_s=float(evidence.speed_m_s),
            vps_num_inliers=int(evidence.vps_num_inliers),
            vps_conf=float(evidence.vps_conf),
            vps_reproj=float(evidence.vps_reproj),
            policy_mode=str(evidence.policy_mode),
        )
        allow_dir_change_override = bool(
            bool(drift_recovery_candidate)
            and bool(runner.global_config.get("VPS_XY_DRIFT_RECOVERY_ALLOW_DIR_CHANGE_BYPASS", True))
        )

        temporal_reject_note = ""
        hard_reject_note = ""
        hard_ok, hard_note = self.vio_service._check_vps_hard_reject(
            np.asarray(evidence.abs_offset_vec, dtype=float).reshape(-1,),
            float(evidence.abs_offset_m),
            speed_m_s=float(evidence.speed_m_s),
            yaw_rate_deg_s=float(evidence.yaw_rate_deg_s),
            allow_dir_change_override=bool(allow_dir_change_override),
        )
        if not hard_ok:
            quality_mode = "reject"
            hard_reject_note = str(hard_note)

        position_first_direct_xy_candidate, position_first_direct_xy_note = self.vio_service._should_use_vps_position_first_direct_xy(
            t_cam=float(evidence.t_cam),
            abs_offset_vec=np.asarray(evidence.abs_offset_vec, dtype=float).reshape(-1,),
            abs_offset_m=float(evidence.abs_offset_m),
            speed_m_s=float(evidence.speed_m_s),
            vps_num_inliers=int(evidence.vps_num_inliers),
            vps_conf=float(evidence.vps_conf),
            vps_reproj=float(evidence.vps_reproj),
            policy_mode=str(evidence.policy_mode),
            hard_reject_note=str(hard_reject_note),
            failsoft_selected=bool(evidence.vps_match_failsoft),
            failsoft_min_inliers=int(evidence.fs_min_inliers),
            failsoft_min_conf=float(evidence.fs_min_conf),
            failsoft_max_reproj=float(evidence.fs_max_reproj),
        )
        if not bool(position_first_direct_xy_candidate):
            fallback_ok, fallback_note = self._allow_failsoft_consensus_fallback(
                evidence=evidence,
                position_first_direct_xy_note=str(position_first_direct_xy_note),
                policy_reject_note=str(policy_reject_note),
                hard_reject_note=str(hard_reject_note),
            )
            if bool(fallback_ok):
                position_first_direct_xy_candidate = True
                position_first_direct_xy_note = (
                    f"{position_first_direct_xy_note}|{fallback_note}"
                    if str(position_first_direct_xy_note).strip()
                    else fallback_note
                )
            else:
                note_l = str(position_first_direct_xy_note).lower()
                if "consensus_" in note_l:
                    runner._vps_direct_xy_reject_consensus_count = int(
                        getattr(runner, "_vps_direct_xy_reject_consensus_count", 0)
                    ) + 1

        drift_recovery_active = False
        drift_recovery_note = ""
        position_first_soft_active = False
        position_first_soft_note = ""
        if (
            quality_mode == "reject"
            and bool(drift_recovery_candidate)
            and not bool(policy_reject_note)
            and not bool(hard_reject_note)
        ):
            quality_mode = "failsoft"
            drift_recovery_active = True
            drift_recovery_note = str(drift_recovery_note_candidate)

        if quality_mode == "reject":
            pos_first_candidate, pos_first_note_candidate = self.vio_service._should_use_vps_position_first_soft_apply(
                t_cam=float(evidence.t_cam),
                abs_offset_m=float(evidence.abs_offset_m),
                speed_m_s=float(evidence.speed_m_s),
                vps_num_inliers=int(evidence.vps_num_inliers),
                vps_conf=float(evidence.vps_conf),
                vps_reproj=float(evidence.vps_reproj),
                policy_mode=str(evidence.policy_mode),
            )
            if bool(pos_first_candidate) and not bool(policy_reject_note) and not bool(hard_reject_note):
                quality_mode = "failsoft"
                position_first_soft_active = True
                position_first_soft_note = str(pos_first_note_candidate)

        if quality_mode == "failsoft":
            skip_temporal = bool(
                bool(drift_recovery_active)
                and bool(runner.global_config.get("VPS_XY_DRIFT_RECOVERY_SKIP_TEMPORAL_GATE", True))
            )
            if (
                not skip_temporal
                and bool(position_first_soft_active)
                and bool(runner.global_config.get("VPS_POSITION_FIRST_SOFT_SKIP_TEMPORAL_GATE", True))
            ):
                skip_temporal = True
            if not skip_temporal:
                temporal_ok, temporal_note = self.vio_service._evaluate_vps_failsoft_temporal_gate(
                    vps_offset=np.asarray(evidence.abs_offset_vec, dtype=float).reshape(-1,),
                    vps_offset_m=float(evidence.abs_offset_m),
                    speed_m_s=float(evidence.speed_m_s),
                    yaw_rate_deg_s=float(evidence.yaw_rate_deg_s),
                )
                if not temporal_ok:
                    quality_mode = "reject"
                    temporal_reject_note = str(temporal_note)

        hint_quality = float("nan")
        if (
            bool(drift_recovery_candidate)
            or bool(position_first_soft_active)
            or bool(position_first_direct_xy_candidate)
        ):
            hint_quality = self._compute_hint_quality(
                evidence,
                position_first_soft_active=bool(position_first_soft_active),
                position_first_direct_xy_candidate=bool(position_first_direct_xy_candidate),
            )

        allow_direct_xy_apply = False
        if quality_mode == "reject":
            if bool(position_first_direct_xy_candidate):
                allow_direct_xy_apply = True
            else:
                force_allow, forced_note = self._allow_force_failsoft_direct(
                    evidence=evidence, policy_reject_note=policy_reject_note
                )
                if force_allow:
                    allow_direct_xy_apply = True
                    position_first_direct_xy_note = (
                        f"{position_first_direct_xy_note}|{forced_note}"
                        if str(position_first_direct_xy_note).strip()
                        else forced_note
                    )
                elif bool(position_first_soft_active or drift_recovery_candidate):
                    allow_direct_xy_apply = ((not bool(policy_reject_note)) and (not bool(hard_reject_note)))

        return VpsPositionDecision(
            quality_mode=str(quality_mode),
            policy_reject_note=str(policy_reject_note),
            hard_reject_note=str(hard_reject_note),
            temporal_reject_note=str(temporal_reject_note),
            drift_recovery_candidate=bool(drift_recovery_candidate),
            drift_recovery_active=bool(drift_recovery_active),
            drift_recovery_note=str(drift_recovery_note),
            position_first_soft_active=bool(position_first_soft_active),
            position_first_soft_note=str(position_first_soft_note),
            position_first_direct_xy_candidate=bool(position_first_direct_xy_candidate),
            position_first_direct_xy_note=str(position_first_direct_xy_note),
            allow_direct_xy_apply=bool(allow_direct_xy_apply),
            hint_quality=float(hint_quality),
        )

    def apply_direct_xy(self, evidence: VpsMatchEvidence, base_quality: float) -> tuple[bool, str]:
        direct_ok, note = self.vio_service._apply_position_first_direct_xy_correction(
            t_cam=float(evidence.t_cam),
            abs_offset_vec=np.asarray(evidence.abs_offset_vec, dtype=float).reshape(-1,),
            abs_offset_m=float(evidence.abs_offset_m),
            vps_num_inliers=int(evidence.vps_num_inliers),
            vps_conf=float(evidence.vps_conf),
            vps_reproj=float(evidence.vps_reproj),
            base_quality=float(base_quality),
            failsoft_selected=bool(evidence.vps_match_failsoft),
        )
        if bool(direct_ok):
            self.runner._vps_direct_xy_apply_count = int(getattr(self.runner, "_vps_direct_xy_apply_count", 0)) + 1
        elif "budget_exhausted" in str(note).lower():
            self.runner._vps_direct_xy_reject_budget_count = int(
                getattr(self.runner, "_vps_direct_xy_reject_budget_count", 0)
            ) + 1
        return bool(direct_ok), str(note)

    def log_trace(
        self,
        *,
        evidence: VpsMatchEvidence,
        decision: VpsPositionDecision,
        applied: bool,
        reason_code: str,
    ) -> None:
        csv_path = getattr(self.runner, "vps_position_trace_csv", None)
        if not csv_path:
            return
        try:
            match_reason = "matched_failsoft" if bool(evidence.vps_match_failsoft) else "matched_strict"
            with open(csv_path, "a", newline="") as f:
                f.write(
                    f"{float(evidence.t_cam):.6f},{int(evidence.frame_idx)},"
                    f"{match_reason},{decision.quality_mode},{int(decision.allow_direct_xy_apply)},"
                    f"{int(decision.position_first_direct_xy_candidate)},{float(decision.hint_quality):.6f},"
                    f"{float(evidence.abs_offset_m):.3f},{int(evidence.vps_num_inliers)},"
                    f"{float(evidence.vps_conf):.6f},{float(evidence.vps_reproj):.6f},"
                    f"{int(applied)},{reason_code},"
                    f"{decision.policy_reject_note},{decision.hard_reject_note},"
                    f"{decision.temporal_reject_note},{decision.position_first_direct_xy_note}\n"
                )
        except Exception:
            pass
