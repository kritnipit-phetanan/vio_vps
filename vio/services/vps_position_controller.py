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
    msckf_quality_score: float = float("nan")
    msckf_stable_geometry_flag: bool = False
    rescue_trigger_reason: str = "none"
    quality_subscores: str = ""
    temporal_hits: int = 0
    scale_pruned_band: str = "full"


@dataclass(frozen=True)
class VpsPositionDecision:
    """Single decision output for VPS position apply chain."""

    decision_lane: str
    quality_mode: str
    force_hint_only: bool
    apply_score: float
    consensus_score: float
    geometry_score: float
    motion_score: float
    bounded_clamp_m: float
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
    late_reclaim_active: bool
    late_reclaim_note: str


class VPSPositionController:
    """Central controller for VPS failsoft/direct-XY conversion."""

    DECISION_STRICT_APPLY = "STRICT_APPLY"
    DECISION_FAILSOFT_APPLY = "FAILSOFT_APPLY"
    DECISION_BOUNDED_SOFT_APPLY = "BOUNDED_SOFT_APPLY"
    DECISION_HINT_ONLY = "HINT_ONLY"
    DECISION_REJECT = "REJECT"

    SOFT_ACCEPT_REASONS = {
        "soft_accept",
        "abs_corr_soft_apply",
        "bounded_soft_apply",
        "xy_drift_recovery_apply",
        "position_first_soft_apply",
        "position_first_direct_xy_apply",
    }
    SOFT_REJECT_REASONS = {
        "soft_reject",
        "abs_corr_temporal_wait",
    }

    def __init__(self, runner: Any, vio_service: Any):
        self.runner = runner
        self.vio_service = vio_service

    @classmethod
    def is_soft_accept_reason(cls, reason_code: str) -> bool:
        return str(reason_code) in cls.SOFT_ACCEPT_REASONS

    @classmethod
    def is_soft_reject_reason(cls, reason_code: str) -> bool:
        reason = str(reason_code)
        return reason in cls.SOFT_REJECT_REASONS or reason.startswith("soft_reject")

    @staticmethod
    def resolve_reason_code(
        *,
        vps_applied: bool,
        quality_mode: str,
        vps_status: str,
        position_first_direct_xy_applied: bool,
        drift_recovery_active: bool,
        position_first_soft_active: bool,
    ) -> str:
        """Classify VPS decision reason from deterministic decision+status inputs."""
        if bool(drift_recovery_active) and bool(vps_applied):
            reason_code = "xy_drift_recovery_apply"
        elif bool(position_first_soft_active) and bool(vps_applied):
            reason_code = "position_first_soft_apply"
        elif bool(position_first_direct_xy_applied):
            reason_code = "position_first_direct_xy_apply"
        elif str(quality_mode) == "failsoft" and bool(vps_applied):
            reason_code = "soft_accept"
        else:
            reason_code = "normal_accept" if bool(vps_applied) else "hard_reject"

        status_lower = str(vps_status or "").lower()
        if not status_lower:
            return reason_code
        if "clone" in status_lower:
            return "skip_missing_clone"
        if "position_first_soft" in status_lower and "applied" in status_lower:
            return "position_first_soft_apply"
        if "position_first_direct_xy_applied" in status_lower:
            return "position_first_direct_xy_apply"
        if "bounded_soft_apply" in status_lower:
            return "bounded_soft_apply"
        if "xy_drift_recovery_hint_only" in status_lower:
            return "xy_drift_recovery_hint_only"
        if "xy_drift_recovery" in status_lower and "applied" in status_lower:
            return "xy_drift_recovery_apply"
        if "skipped_quality" in status_lower:
            if "soft_reject" in status_lower:
                return "soft_reject"
            return "skip_low_quality"
        if "hard_reject" in status_lower:
            return "abs_corr_hard_reject"
        if "policy_mode_" in status_lower:
            return "policy_hold"
        if "large_offset_pending" in status_lower:
            return "abs_corr_temporal_wait"
        if str(quality_mode) == "failsoft" and "gated" in status_lower:
            return "soft_reject"
        if "gated" in status_lower:
            return "gated"
        if "failed" in status_lower:
            return "soft_reject" if str(quality_mode) == "failsoft" else "hard_reject"
        if "clamped" in status_lower:
            return "abs_corr_clamped_apply"
        if "applied" in status_lower:
            return "abs_corr_soft_apply" if str(quality_mode) == "failsoft" else "normal_accept"
        return reason_code

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

    def _score_apply_lane(
        self,
        *,
        evidence: VpsMatchEvidence,
        strict_quality_ok: bool,
        failsoft_quality_ok: bool,
        position_first_direct_xy_candidate: bool,
        policy_reject_note: str,
        hard_reject_note: str,
        temporal_reject_note: str,
        position_first_direct_xy_note: str,
    ) -> tuple[float, float, float, float, str]:
        cfg = self.runner.global_config
        strict_th = float(cfg.get("VPS_APPLY_GATE_STRICT_SCORE_TH", 0.74))
        failsoft_th = float(cfg.get("VPS_APPLY_GATE_FAILSOFT_SCORE_TH", 0.54))
        hint_th = float(cfg.get("VPS_APPLY_GATE_HINT_SCORE_TH", 0.32))
        bounded_soft_th = float(cfg.get("VPS_APPLY_GATE_BOUNDED_SOFT_SCORE_TH", hint_th))
        bounded_soft_enable = bool(cfg.get("VPS_APPLY_GATE_BOUNDED_SOFT_ENABLE", True))
        msckf_gate_enable = bool(cfg.get("VPS_APPLY_GATE_MSCKF_UNSTABLE_ENABLE", True))
        msckf_min_quality = float(cfg.get("VPS_APPLY_GATE_MSCKF_MIN_QUALITY", 0.34))
        msckf_apply_penalty = float(cfg.get("VPS_APPLY_GATE_MSCKF_UNSTABLE_APPLY_PENALTY", 0.06))
        msckf_failsoft_floor = float(cfg.get("VPS_APPLY_GATE_MSCKF_UNSTABLE_FAILSOFT_TH", 0.58))
        w_cons = float(cfg.get("VPS_APPLY_GATE_CONSENSUS_WEIGHT", 0.35))
        w_geom = float(cfg.get("VPS_APPLY_GATE_GEOMETRY_WEIGHT", 0.45))
        w_motion = float(cfg.get("VPS_APPLY_GATE_MOTION_WEIGHT", 0.20))

        fs_min_inliers = max(1, int(evidence.fs_min_inliers))
        fs_min_conf = max(1e-3, float(evidence.fs_min_conf))
        fs_max_reproj = max(0.05, float(evidence.fs_max_reproj))
        fs_max_offset = max(
            1.0,
            float(cfg.get("VPS_APPLY_FAILSOFT_MAX_OFFSET_M", 180.0)),
        )

        inlier_score = float(np.clip(float(evidence.vps_num_inliers) / float(max(2 * fs_min_inliers, 6)), 0.0, 1.0))
        conf_score = float(np.clip(float(evidence.vps_conf) / float(max(fs_min_conf * 1.8, 0.12)), 0.0, 1.0))
        reproj_score = float(np.clip(fs_max_reproj / max(float(evidence.vps_reproj), 0.05), 0.0, 1.0))
        geometry_score = float(np.clip(0.45 * inlier_score + 0.30 * conf_score + 0.25 * reproj_score, 0.0, 1.0))
        if bool(strict_quality_ok):
            geometry_score = float(np.clip(geometry_score + 0.12, 0.0, 1.0))
        elif not bool(failsoft_quality_ok):
            geometry_score = float(np.clip(geometry_score * 0.70, 0.0, 1.0))

        offset_score = float(np.clip(1.0 - (float(evidence.abs_offset_m) / fs_max_offset), 0.0, 1.0))
        consensus_score = float(offset_score)
        if bool(position_first_direct_xy_candidate):
            consensus_score = float(np.clip(consensus_score + 0.20, 0.0, 1.0))
        note_l = str(position_first_direct_xy_note).lower()
        if "consensus_warmup" in note_l:
            consensus_score = min(consensus_score, 0.40)
        elif "consensus_dev" in note_l:
            consensus_score = min(consensus_score, 0.30)
        elif "consensus_dir" in note_l:
            consensus_score = min(consensus_score, 0.24)

        speed_ref = max(1.0, float(cfg.get("VPS_APPLY_GATE_MOTION_HIGH_SPEED_M_S", 45.0)))
        yaw_ref = max(1.0, float(cfg.get("VPS_APPLY_GATE_MOTION_HIGH_YAWRATE_DEG_S", 55.0)))
        speed_norm = (
            float(np.clip(float(evidence.speed_m_s) / speed_ref, 0.0, 1.0))
            if np.isfinite(float(evidence.speed_m_s))
            else 0.0
        )
        yaw_norm = (
            float(np.clip(abs(float(evidence.yaw_rate_deg_s)) / yaw_ref, 0.0, 1.0))
            if np.isfinite(float(evidence.yaw_rate_deg_s))
            else 0.0
        )
        motion_score = float(np.clip(1.0 - (0.72 * speed_norm + 0.28 * yaw_norm), 0.0, 1.0))

        weight_sum = max(1e-6, w_cons + w_geom + w_motion)
        apply_score = float(
            np.clip(
                (w_cons * consensus_score + w_geom * geometry_score + w_motion * motion_score) / weight_sum,
                0.0,
                1.0,
            )
        )

        msckf_q = float(getattr(evidence, "msckf_quality_score", np.nan))
        msckf_stable = bool(getattr(evidence, "msckf_stable_geometry_flag", False))
        msckf_unstable = bool(
            msckf_gate_enable
            and (
                (not msckf_stable)
                or (np.isfinite(msckf_q) and float(msckf_q) < float(msckf_min_quality))
            )
        )
        if bool(msckf_unstable):
            apply_score = float(np.clip(float(apply_score) - float(msckf_apply_penalty), 0.0, 1.0))
            failsoft_th = max(float(failsoft_th), float(msckf_failsoft_floor))

        if bool(policy_reject_note) or bool(hard_reject_note) or bool(temporal_reject_note):
            # Policy/hard/temporal rejections are not converted to apply lanes.
            return (
                apply_score,
                consensus_score,
                geometry_score,
                motion_score,
                self.DECISION_REJECT,
            )

        if apply_score >= strict_th and bool(strict_quality_ok):
            lane = self.DECISION_STRICT_APPLY
        elif apply_score >= failsoft_th and (bool(failsoft_quality_ok) or bool(position_first_direct_xy_candidate)):
            lane = self.DECISION_FAILSOFT_APPLY
        elif apply_score >= (
            bounded_soft_th if bool(bounded_soft_enable) else hint_th
        ) and (bool(failsoft_quality_ok) or bool(position_first_direct_xy_candidate)):
            lane = (
                self.DECISION_BOUNDED_SOFT_APPLY
                if bool(bounded_soft_enable)
                else self.DECISION_HINT_ONLY
            )
        else:
            lane = self.DECISION_REJECT

        # Accuracy branch default: fail-soft is not apply-default.
        failsoft_default_hint_only = bool(
            cfg.get("VPS_APPLY_GATE_FAILSOFT_DEFAULT_HINT_ONLY", True)
        )
        if (
            lane == self.DECISION_FAILSOFT_APPLY
            and failsoft_default_hint_only
            and not bool(strict_quality_ok)
        ):
            lane = (
                self.DECISION_BOUNDED_SOFT_APPLY
                if bool(bounded_soft_enable)
                else self.DECISION_HINT_ONLY
            )
        return apply_score, consensus_score, geometry_score, motion_score, lane

    def _motion_consistency_gate(self, *, evidence: VpsMatchEvidence) -> tuple[bool, str]:
        """Block VPS apply lanes when correction vector opposes short-term motion."""
        cfg = self.runner.global_config
        if not bool(cfg.get("VPS_APPLY_GATE_MOTION_CONSISTENCY_ENABLE", True)):
            return True, ""

        vps_vec = np.asarray(evidence.abs_offset_vec, dtype=float).reshape(-1,)
        if vps_vec.size < 2:
            return True, ""
        vps_xy = np.asarray(vps_vec[:2], dtype=float).reshape(2,)
        if not np.all(np.isfinite(vps_xy)):
            return True, ""
        vps_norm = float(np.linalg.norm(vps_xy))
        min_offset_m = max(
            0.5,
            float(cfg.get("VPS_APPLY_GATE_MOTION_CONSISTENCY_MIN_OFFSET_M", 10.0)),
        )
        if (not np.isfinite(vps_norm)) or float(vps_norm) < float(min_offset_m):
            return True, ""

        try:
            vel_xy = np.asarray(self.runner.kf.x[3:5, 0], dtype=float).reshape(2,)
        except Exception:
            vel_xy = np.asarray([np.nan, np.nan], dtype=float)
        if not np.all(np.isfinite(vel_xy)):
            return True, ""
        speed_xy = float(np.linalg.norm(vel_xy))
        min_speed = max(
            0.1,
            float(cfg.get("VPS_APPLY_GATE_MOTION_CONSISTENCY_MIN_SPEED_M_S", 14.0)),
        )
        if (not np.isfinite(speed_xy)) or float(speed_xy) < float(min_speed):
            return True, ""

        vps_dir = vps_xy / max(1e-9, float(vps_norm))
        vel_dir = vel_xy / max(1e-9, float(speed_xy))
        ema_alpha = float(np.clip(cfg.get("VPS_APPLY_GATE_MOTION_CONSISTENCY_REF_ALPHA", 0.30), 0.05, 1.0))
        prev_ref = getattr(self.runner, "_vps_motion_consistency_ref_xy", None)
        if isinstance(prev_ref, np.ndarray) and prev_ref.size >= 2 and np.all(np.isfinite(prev_ref[:2])):
            ref = (1.0 - float(ema_alpha)) * np.asarray(prev_ref[:2], dtype=float) + float(ema_alpha) * vel_dir
        else:
            ref = vel_dir
        ref_norm = float(np.linalg.norm(ref))
        if (not np.isfinite(ref_norm)) or ref_norm <= 1e-9:
            return True, ""
        ref_dir = ref / ref_norm
        self.runner._vps_motion_consistency_ref_xy = np.asarray(ref_dir, dtype=float).reshape(2,)

        cosang = float(np.clip(np.dot(vps_dir, ref_dir), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cosang)))

        full_speed = max(
            float(min_speed) + 1e-3,
            float(cfg.get("VPS_APPLY_GATE_MOTION_CONSISTENCY_SPEED_FULL_M_S", 90.0)),
        )
        speed_ratio = float(
            np.clip((float(speed_xy) - float(min_speed)) / (float(full_speed) - float(min_speed)), 0.0, 1.0)
        )
        max_opp_low = float(
            np.clip(
                cfg.get("VPS_APPLY_GATE_MOTION_CONSISTENCY_MAX_OPPOSITION_DEG_LOW_SPEED", 130.0),
                95.0,
                179.0,
            )
        )
        max_opp_high = float(
            np.clip(
                cfg.get("VPS_APPLY_GATE_MOTION_CONSISTENCY_MAX_OPPOSITION_DEG_HIGH_SPEED", 105.0),
                85.0,
                max_opp_low,
            )
        )
        allowed_angle = float(max_opp_low + (max_opp_high - max_opp_low) * speed_ratio)

        yaw_rate = abs(float(evidence.yaw_rate_deg_s)) if np.isfinite(float(evidence.yaw_rate_deg_s)) else 0.0
        yaw_relax_start = max(
            1.0,
            float(cfg.get("VPS_APPLY_GATE_MOTION_CONSISTENCY_YAW_RELAX_START_DEG_S", 35.0)),
        )
        yaw_relax_max = max(
            0.0,
            float(cfg.get("VPS_APPLY_GATE_MOTION_CONSISTENCY_YAW_RELAX_MAX_DEG", 12.0)),
        )
        if yaw_rate > yaw_relax_start and yaw_relax_max > 1e-6:
            relax_ratio = float(np.clip((yaw_rate - yaw_relax_start) / max(1.0, yaw_relax_start), 0.0, 1.0))
            allowed_angle = float(min(179.0, allowed_angle + yaw_relax_max * relax_ratio))

        if float(angle_deg) > float(allowed_angle):
            return (
                False,
                f"motion_consistency_reject:ang={float(angle_deg):.1f}>{float(allowed_angle):.1f},"
                f"spd={float(speed_xy):.1f},off={float(vps_norm):.1f}",
            )
        return True, ""

    def decide_and_classify(self, evidence: VpsMatchEvidence) -> VpsPositionDecision:
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
            # Source-quality temporal consensus gate (2-3 frame maturity)
            # before promoting failsoft/rescue candidates.
            temporal_hits = int(max(0, evidence.temporal_hits))
            rescue_mode = str(getattr(evidence, "rescue_trigger_reason", "none")).strip().lower() not in ("", "none")
            min_hits_failsoft = max(
                1,
                int(runner.global_config.get("VPS_TEMPORAL_CONSENSUS_MIN_HITS_FAILSOFT", 2)),
            )
            min_hits_rescue = max(
                min_hits_failsoft,
                int(runner.global_config.get("VPS_TEMPORAL_CONSENSUS_MIN_HITS_RESCUE", 3)),
            )
            min_hits_required = int(min_hits_rescue if rescue_mode else min_hits_failsoft)
            if temporal_hits < min_hits_required:
                quality_mode = "reject"
                temporal_reject_note = (
                    f"SOFT_REJECT_TEMPORAL_HITS:{temporal_hits}/{min_hits_required}"
                    + ("|rescue" if rescue_mode else "")
                )
                runner._vps_temporal_consensus_block_count = int(
                    getattr(runner, "_vps_temporal_consensus_block_count", 0)
                ) + 1
            else:
                runner._vps_temporal_consensus_pass_count = int(
                    getattr(runner, "_vps_temporal_consensus_pass_count", 0)
                ) + 1

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

        (
            apply_score,
            consensus_score,
            geometry_score,
            motion_score,
            decision_lane,
        ) = self._score_apply_lane(
            evidence=evidence,
            strict_quality_ok=bool(evidence.strict_quality_ok),
            failsoft_quality_ok=bool(evidence.failsoft_quality_ok),
            position_first_direct_xy_candidate=bool(position_first_direct_xy_candidate),
            policy_reject_note=str(policy_reject_note),
            hard_reject_note=str(hard_reject_note),
            temporal_reject_note=str(temporal_reject_note),
            position_first_direct_xy_note=str(position_first_direct_xy_note),
        )

        # Deterministic hold-bypass for accuracy branch:
        # allow bounded soft-apply when policy HOLD persists and match quality is adequate.
        late_reclaim_active = False
        late_reclaim_note = ""
        hard_reject_text = str(hard_reject_note or "")
        allow_offset_hard_reject_in_hold = bool(
            runner.global_config.get(
                "VPS_APPLY_GATE_BOUNDED_SOFT_ALLOW_OFFSET_HARD_REJECT_IN_HOLD",
                True,
            )
        )
        offset_only_hard_reject = bool(
            "HARD_REJECT_OFFSET" in hard_reject_text and "HARD_REJECT_" in hard_reject_text
        )
        hard_block_for_hold = bool(hard_reject_text) and not (
            bool(allow_offset_hard_reject_in_hold) and bool(offset_only_hard_reject)
        )
        if (
            decision_lane == self.DECISION_REJECT
            and str(policy_reject_note).startswith("policy_mode_hold")
            and not bool(hard_block_for_hold)
            and not bool(temporal_reject_note)
            and bool(runner.global_config.get("VPS_APPLY_GATE_BOUNDED_SOFT_ENABLE", True))
            and bool(
                runner.global_config.get(
                    "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_ENABLE",
                    True,
                )
            )
        ):
            hold_bypass_min_no_apply_sec = max(
                0.0,
                float(
                    runner.global_config.get(
                        "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MIN_NO_APPLY_SEC",
                        3.0,
                    )
                ),
            )
            hold_bypass_min_score = float(
                runner.global_config.get(
                    "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MIN_SCORE_TH",
                    max(0.36, float(runner.global_config.get("VPS_APPLY_GATE_BOUNDED_SOFT_SCORE_TH", 0.36))),
                )
            )
            hold_bypass_max_speed = float(
                runner.global_config.get(
                    "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MAX_SPEED_M_S",
                    95.0,
                )
            )
            hold_bypass_max_offset = float(
                runner.global_config.get(
                    "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MAX_OFFSET_M",
                    float(runner.global_config.get("VPS_APPLY_FAILSOFT_MAX_OFFSET_M", 180.0)),
                )
            )
            hold_bypass_min_inliers = max(
                1,
                int(
                    round(
                        float(
                            runner.global_config.get(
                                "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MIN_INLIERS",
                                evidence.fs_min_inliers,
                            )
                        )
                    )
                ),
            )
            hold_bypass_min_conf = float(
                runner.global_config.get(
                    "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MIN_CONFIDENCE",
                    evidence.fs_min_conf,
                )
            )
            hold_bypass_max_reproj = float(
                runner.global_config.get(
                    "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_BYPASS_MAX_REPROJ_ERROR",
                    evidence.fs_max_reproj,
                )
            )
            last_apply_t = float(getattr(runner, "last_vps_update_time", -1e9))
            since_last_apply = float(evidence.t_cam) - float(last_apply_t)
            hold_bypass_ok = bool(
                np.isfinite(float(since_last_apply))
                and float(since_last_apply) >= float(hold_bypass_min_no_apply_sec)
                and np.isfinite(float(apply_score))
                and float(apply_score) >= float(hold_bypass_min_score)
                and np.isfinite(float(evidence.abs_offset_m))
                and float(evidence.abs_offset_m) <= float(hold_bypass_max_offset)
                and int(evidence.vps_num_inliers) >= int(hold_bypass_min_inliers)
                and np.isfinite(float(evidence.vps_conf))
                and float(evidence.vps_conf) >= float(hold_bypass_min_conf)
                and np.isfinite(float(evidence.vps_reproj))
                and float(evidence.vps_reproj) <= float(hold_bypass_max_reproj)
                and (
                    (not np.isfinite(float(evidence.speed_m_s)))
                    or float(evidence.speed_m_s) <= float(hold_bypass_max_speed)
                )
            )
            if bool(hold_bypass_ok):
                decision_lane = self.DECISION_BOUNDED_SOFT_APPLY
                policy_reject_note = "policy_mode_hold_bypass_bounded_soft"
                runner._vps_bounded_soft_hold_bypass_count = int(
                    getattr(runner, "_vps_bounded_soft_hold_bypass_count", 0)
                ) + 1
            elif bool(
                runner.global_config.get(
                    "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_RECLAIM_ENABLE",
                    True,
                )
            ):
                reclaim_min_no_apply_sec = max(
                    float(hold_bypass_min_no_apply_sec),
                    float(
                        runner.global_config.get(
                            "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_RECLAIM_MIN_NO_APPLY_SEC",
                            8.0,
                        )
                    ),
                )
                reclaim_min_score = float(
                    runner.global_config.get(
                        "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_RECLAIM_MIN_SCORE_TH",
                        min(float(hold_bypass_min_score), 0.34),
                    )
                )
                reclaim_max_speed = float(
                    runner.global_config.get(
                        "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_RECLAIM_MAX_SPEED_M_S",
                        float(hold_bypass_max_speed),
                    )
                )
                reclaim_max_offset = float(
                    runner.global_config.get(
                        "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_RECLAIM_MAX_OFFSET_M",
                        float(hold_bypass_max_offset),
                    )
                )
                reclaim_min_inliers = max(
                    1,
                    int(
                        round(
                            float(
                                runner.global_config.get(
                                    "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_RECLAIM_MIN_INLIERS",
                                    max(3, hold_bypass_min_inliers - 1),
                                )
                            )
                        )
                    ),
                )
                reclaim_min_conf = float(
                    runner.global_config.get(
                        "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_RECLAIM_MIN_CONFIDENCE",
                        max(0.06, float(hold_bypass_min_conf) * 0.8),
                    )
                )
                reclaim_max_reproj = float(
                    runner.global_config.get(
                        "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_RECLAIM_MAX_REPROJ_ERROR",
                        max(1.4, float(hold_bypass_max_reproj)),
                    )
                )
                reclaim_ok = bool(
                    np.isfinite(float(since_last_apply))
                    and float(since_last_apply) >= float(reclaim_min_no_apply_sec)
                    and np.isfinite(float(apply_score))
                    and float(apply_score) >= float(reclaim_min_score)
                    and np.isfinite(float(evidence.abs_offset_m))
                    and float(evidence.abs_offset_m) <= float(reclaim_max_offset)
                    and int(evidence.vps_num_inliers) >= int(reclaim_min_inliers)
                    and np.isfinite(float(evidence.vps_conf))
                    and float(evidence.vps_conf) >= float(reclaim_min_conf)
                    and np.isfinite(float(evidence.vps_reproj))
                    and float(evidence.vps_reproj) <= float(reclaim_max_reproj)
                    and (
                        (not np.isfinite(float(evidence.speed_m_s)))
                        or float(evidence.speed_m_s) <= float(reclaim_max_speed)
                    )
                )
                if bool(reclaim_ok):
                    decision_lane = self.DECISION_BOUNDED_SOFT_APPLY
                    policy_reject_note = "policy_mode_hold_reclaim_bounded_soft"
                    late_reclaim_active = True
                    late_reclaim_note = (
                        f"reclaim:no_apply={since_last_apply:.2f}s,score={float(apply_score):.3f}"
                    )
                    runner._vps_bounded_soft_hold_reclaim_count = int(
                        getattr(runner, "_vps_bounded_soft_hold_reclaim_count", 0)
                    ) + 1

        bounded_clamp_m = float("nan")
        if decision_lane == self.DECISION_BOUNDED_SOFT_APPLY:
            bounded_clamp_m = float(
                runner.global_config.get(
                    "VPS_APPLY_GATE_BOUNDED_SOFT_MAX_APPLY_DP_XY_M",
                    12.0,
                )
            )
            if np.isfinite(float(evidence.speed_m_s)):
                hs_th = float(
                    runner.global_config.get(
                        "VPS_APPLY_GATE_BOUNDED_SOFT_HIGH_SPEED_M_S",
                        20.0,
                    )
                )
                if float(evidence.speed_m_s) >= hs_th:
                    bounded_clamp_m = min(
                        float(bounded_clamp_m),
                        float(
                            runner.global_config.get(
                                "VPS_APPLY_GATE_BOUNDED_SOFT_HIGH_SPEED_MAX_APPLY_DP_XY_M",
                                7.0,
                            )
                        ),
                    )
            if bool(late_reclaim_active):
                reclaim_clamp = float(
                    runner.global_config.get(
                        "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_RECLAIM_MAX_APPLY_DP_XY_M",
                        4.5,
                    )
                )
                if np.isfinite(float(evidence.speed_m_s)):
                    hs_th = float(
                        runner.global_config.get(
                            "VPS_APPLY_GATE_BOUNDED_SOFT_HIGH_SPEED_M_S",
                            20.0,
                        )
                    )
                    if float(evidence.speed_m_s) >= hs_th:
                        reclaim_clamp = min(
                            float(reclaim_clamp),
                            float(
                                runner.global_config.get(
                                    "VPS_APPLY_GATE_BOUNDED_SOFT_POLICY_HOLD_RECLAIM_HIGH_SPEED_MAX_APPLY_DP_XY_M",
                                    3.0,
                                )
                            ),
                        )
                bounded_clamp_m = min(float(bounded_clamp_m), float(reclaim_clamp))
            min_interval = max(
                0.0,
                float(
                    runner.global_config.get(
                        "VPS_APPLY_GATE_BOUNDED_SOFT_MIN_INTERVAL_SEC",
                        0.35,
                    )
                ),
            )
            if min_interval > 1e-6:
                last_apply_t = float(getattr(runner, "_vps_bounded_soft_last_apply_time", -1e9))
                if np.isfinite(last_apply_t):
                    dt_apply = float(evidence.t_cam) - float(last_apply_t)
                    if np.isfinite(dt_apply) and dt_apply < min_interval:
                        decision_lane = self.DECISION_REJECT
                        bounded_clamp_m = float("nan")
                        note = f"bounded_soft_min_interval:{dt_apply:.2f}<{min_interval:.2f}"
                        temporal_reject_note = (
                            f"{temporal_reject_note}|{note}"
                            if str(temporal_reject_note).strip()
                            else note
                        )

        if decision_lane in (
            self.DECISION_STRICT_APPLY,
            self.DECISION_FAILSOFT_APPLY,
            self.DECISION_BOUNDED_SOFT_APPLY,
        ):
            motion_ok, motion_note = self._motion_consistency_gate(evidence=evidence)
            if not bool(motion_ok):
                runner._vps_motion_consistency_block_count = int(
                    getattr(runner, "_vps_motion_consistency_block_count", 0)
                ) + 1
                block_to_hint = bool(
                    runner.global_config.get(
                        "VPS_APPLY_GATE_MOTION_CONSISTENCY_BLOCK_TO_HINT_ONLY",
                        True,
                    )
                )
                decision_lane = self.DECISION_HINT_ONLY if block_to_hint else self.DECISION_REJECT
                bounded_clamp_m = float("nan")
                temporal_reject_note = (
                    f"{temporal_reject_note}|{motion_note}"
                    if str(temporal_reject_note).strip()
                    else motion_note
                )

        force_hint_only = bool(decision_lane == self.DECISION_HINT_ONLY)
        allow_direct_xy_apply = False
        if decision_lane in (
            self.DECISION_STRICT_APPLY,
            self.DECISION_FAILSOFT_APPLY,
            self.DECISION_BOUNDED_SOFT_APPLY,
        ):
            quality_mode = "strict" if bool(evidence.strict_quality_ok) else "failsoft"
        elif decision_lane == self.DECISION_HINT_ONLY:
            quality_mode = "reject"
        else:
            quality_mode = "reject"

        if quality_mode == "reject" and not force_hint_only:
            # In REJECT lane, keep only deterministic fallback path;
            # APPLY/HINT lane conversions are handled in classifier above.
            if bool(position_first_soft_active or drift_recovery_candidate):
                allow_direct_xy_apply = ((not bool(policy_reject_note)) and (not bool(hard_reject_note)))

        return VpsPositionDecision(
            decision_lane=str(decision_lane),
            quality_mode=str(quality_mode),
            force_hint_only=bool(force_hint_only),
            apply_score=float(apply_score),
            consensus_score=float(consensus_score),
            geometry_score=float(geometry_score),
            motion_score=float(motion_score),
            bounded_clamp_m=float(bounded_clamp_m),
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
            late_reclaim_active=bool(late_reclaim_active),
            late_reclaim_note=str(late_reclaim_note),
        )

    def decide(self, evidence: VpsMatchEvidence) -> VpsPositionDecision:
        """Backward-compatible alias to the canonical classifier API."""
        return self.decide_and_classify(evidence)

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
            residual_xy = float(evidence.abs_offset_m) if np.isfinite(float(evidence.abs_offset_m)) else float("nan")
            applied_dp_xy = 0.0
            if bool(applied):
                if np.isfinite(float(decision.bounded_clamp_m)) and float(decision.bounded_clamp_m) > 0.0:
                    applied_dp_xy = float(min(max(0.0, residual_xy), float(decision.bounded_clamp_m)))
                else:
                    applied_dp_xy = float(max(0.0, residual_xy))
            reject_reason = "" if bool(applied) else str(reason_code)
            rescue_trigger_reason = str(getattr(evidence, "rescue_trigger_reason", "none")).replace(",", ";")
            quality_subscores = str(getattr(evidence, "quality_subscores", "")).replace(",", ";")
            temporal_hits = int(getattr(evidence, "temporal_hits", 0))
            scale_pruned_band = str(getattr(evidence, "scale_pruned_band", "full")).replace(",", ";")
            with open(csv_path, "a", newline="") as f:
                f.write(
                    f"{float(evidence.t_cam):.6f},{int(evidence.frame_idx)},"
                    f"{match_reason},{decision.quality_mode},{decision.decision_lane},"
                    f"{int(decision.force_hint_only)},{float(decision.apply_score):.6f},"
                    f"{float(decision.consensus_score):.6f},{float(decision.geometry_score):.6f},"
                    f"{float(decision.motion_score):.6f},{float(decision.bounded_clamp_m):.6f},"
                    f"{int(decision.allow_direct_xy_apply)},"
                    f"{int(decision.position_first_direct_xy_candidate)},{float(decision.hint_quality):.6f},"
                    f"{float(evidence.abs_offset_m):.3f},{int(evidence.vps_num_inliers)},"
                    f"{float(evidence.vps_conf):.6f},{float(evidence.vps_reproj):.6f},"
                    f"{int(applied)},{reason_code},{reject_reason},"
                    f"{residual_xy:.6f},{applied_dp_xy:.6f},"
                    f"{decision.policy_reject_note},{decision.hard_reject_note},"
                    f"{decision.temporal_reject_note},{decision.position_first_direct_xy_note},"
                    f"{rescue_trigger_reason},{quality_subscores},{temporal_hits},{scale_pruned_band}\n"
                )
        except Exception:
            pass
