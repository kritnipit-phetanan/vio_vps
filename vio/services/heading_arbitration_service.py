"""Heading arbitration service for MAG yaw authority.

This service is intentionally stateful and lightweight:
- Builds one consistency score from vision/gyro/state agreement
- Limits cumulative MAG yaw injection over a sliding time window
- Applies hold/recover behavior when mismatch persists
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class HeadingArbResult:
    skip: bool
    r_mult: float
    reason: str
    consistency_score: float
    max_update_dyaw_deg: float
    authority: str


class HeadingArbitrationService:
    """Stateful MAG heading authority arbitration."""

    def __init__(self, runner: Any):
        self.runner = runner
        self._score_ema: float = float("nan")
        self._mismatch_streak: int = 0
        self._hold_until: float = -1e9
        self._recovery_active: bool = False
        self._recovery_good_hits: int = 0
        self._yaw_injection_hist: deque[tuple[float, float]] = deque()

    @staticmethod
    def _wrap(rad: float) -> float:
        return float(np.arctan2(np.sin(float(rad)), np.cos(float(rad))))

    @staticmethod
    def _lin_score(delta: float, good: float, bad: float) -> float:
        if not np.isfinite(delta):
            return float("nan")
        if delta <= good:
            return 1.0
        if delta >= bad:
            return 0.0
        return float(1.0 - (delta - good) / max(1e-9, bad - good))

    @staticmethod
    def _policy_extra(policy_decision: Optional[Any], key: str, default: float) -> float:
        if policy_decision is None:
            return float(default)
        try:
            return float(policy_decision.extra(key, default))
        except Exception:
            return float(default)

    def _prune_injection_hist(self, now_t: float, window_sec: float) -> float:
        cutoff = float(now_t) - max(0.01, float(window_sec))
        while self._yaw_injection_hist and self._yaw_injection_hist[0][0] < cutoff:
            self._yaw_injection_hist.popleft()
        if len(self._yaw_injection_hist) == 0:
            return 0.0
        return float(sum(abs(v) for _, v in self._yaw_injection_hist))

    def evaluate_mag(
        self,
        timestamp: float,
        yaw_mag: float,
        yaw_state: float,
        vision_yaw: Optional[float],
        vision_age_sec: float,
        vision_quality: float,
        gyro_consistency_delta_deg: float,
        policy_decision: Optional[Any] = None,
    ) -> HeadingArbResult:
        """Evaluate MAG authority for current sample."""
        cfg = self.runner.global_config

        enabled = bool(cfg.get("MAG_HEADING_ARB_ENABLE", False))
        if policy_decision is not None:
            enabled = bool(self._policy_extra(policy_decision, "heading_arb_enable", 1.0 if enabled else 0.0) > 0.5)
        if not enabled:
            return HeadingArbResult(
                skip=False,
                r_mult=1.0,
                reason="",
                consistency_score=float("nan"),
                max_update_dyaw_deg=float("inf"),
                authority="disabled",
            )

        hard_mismatch_deg = float(cfg.get("MAG_HEADING_ARB_HARD_MISMATCH_DEG", 95.0))
        min_vis_quality = float(cfg.get("MAG_HEADING_ARB_MIN_VISION_QUALITY", 0.55))
        streak_to_hold = int(cfg.get("MAG_HEADING_ARB_STREAK_TO_HOLD", 3))
        hold_sec = float(cfg.get("MAG_HEADING_ARB_HOLD_SEC", 1.8))
        soft_r_mult = float(cfg.get("MAG_HEADING_ARB_SOFT_R_MULT", 2.0))
        max_vis_age = float(cfg.get("MAG_HEADING_ARB_MAX_VISION_AGE_SEC", 1.0))

        score_ema_alpha = float(cfg.get("MAG_HEADING_ARB_SCORE_EMA_ALPHA", 0.20))
        score_soft_thresh = float(cfg.get("MAG_HEADING_ARB_SCORE_SOFT_THRESHOLD", 0.55))
        score_hold_thresh = float(cfg.get("MAG_HEADING_ARB_SCORE_HOLD_THRESHOLD", 0.30))
        score_vis_good_deg = float(cfg.get("MAG_HEADING_ARB_SCORE_VIS_GOOD_DEG", 18.0))
        score_gyro_good_deg = float(cfg.get("MAG_HEADING_ARB_SCORE_GYRO_GOOD_DEG", 10.0))
        score_gyro_bad_deg = float(cfg.get("MAG_HEADING_ARB_SCORE_GYRO_BAD_DEG", 50.0))
        score_state_good_deg = float(cfg.get("MAG_HEADING_ARB_SCORE_STATE_GOOD_DEG", 25.0))
        score_state_bad_deg = float(cfg.get("MAG_HEADING_ARB_SCORE_STATE_BAD_DEG", 110.0))
        w_vis = float(cfg.get("MAG_HEADING_ARB_SCORE_VIS_WEIGHT", 0.50))
        w_gyro = float(cfg.get("MAG_HEADING_ARB_SCORE_GYRO_WEIGHT", 0.30))
        w_state = float(cfg.get("MAG_HEADING_ARB_SCORE_STATE_WEIGHT", 0.20))

        yaw_budget_window_sec = float(cfg.get("MAG_HEADING_ARB_YAW_BUDGET_WINDOW_SEC", 6.0))
        yaw_budget_abs_deg = float(cfg.get("MAG_HEADING_ARB_YAW_BUDGET_ABS_DEG", 8.0))
        yaw_budget_min_remaining_deg = float(
            cfg.get("MAG_HEADING_ARB_YAW_BUDGET_MIN_REMAINING_DEG", 0.6)
        )
        recovery_confirm_hits = int(cfg.get("MAG_HEADING_ARB_RECOVER_CONFIRM_HITS", 3))
        recovery_min_score = float(cfg.get("MAG_HEADING_ARB_RECOVER_MIN_SCORE", 0.60))
        recovery_soft_r_mult = float(cfg.get("MAG_HEADING_ARB_RECOVER_SOFT_R_MULT", 1.6))
        recovery_max_update_dyaw_deg = float(
            cfg.get("MAG_HEADING_ARB_RECOVER_MAX_UPDATE_DYAW_DEG", 0.30)
        )

        # Policy snapshot overrides
        if policy_decision is not None:
            hard_mismatch_deg = self._policy_extra(policy_decision, "heading_arb_hard_mismatch_deg", hard_mismatch_deg)
            min_vis_quality = self._policy_extra(policy_decision, "heading_arb_min_vision_quality", min_vis_quality)
            streak_to_hold = int(self._policy_extra(policy_decision, "heading_arb_streak_to_hold", streak_to_hold))
            hold_sec = self._policy_extra(policy_decision, "heading_arb_hold_sec", hold_sec)
            soft_r_mult = self._policy_extra(policy_decision, "heading_arb_soft_r_mult", soft_r_mult)
            max_vis_age = self._policy_extra(policy_decision, "heading_arb_max_vision_age_sec", max_vis_age)
            score_ema_alpha = self._policy_extra(policy_decision, "heading_arb_score_ema_alpha", score_ema_alpha)
            score_soft_thresh = self._policy_extra(policy_decision, "heading_arb_score_soft_threshold", score_soft_thresh)
            score_hold_thresh = self._policy_extra(policy_decision, "heading_arb_score_hold_threshold", score_hold_thresh)
            score_vis_good_deg = self._policy_extra(policy_decision, "heading_arb_score_vis_good_deg", score_vis_good_deg)
            score_gyro_good_deg = self._policy_extra(policy_decision, "heading_arb_score_gyro_good_deg", score_gyro_good_deg)
            score_gyro_bad_deg = self._policy_extra(policy_decision, "heading_arb_score_gyro_bad_deg", score_gyro_bad_deg)
            score_state_good_deg = self._policy_extra(policy_decision, "heading_arb_score_state_good_deg", score_state_good_deg)
            score_state_bad_deg = self._policy_extra(policy_decision, "heading_arb_score_state_bad_deg", score_state_bad_deg)
            w_vis = self._policy_extra(policy_decision, "heading_arb_score_vis_weight", w_vis)
            w_gyro = self._policy_extra(policy_decision, "heading_arb_score_gyro_weight", w_gyro)
            w_state = self._policy_extra(policy_decision, "heading_arb_score_state_weight", w_state)
            yaw_budget_window_sec = self._policy_extra(policy_decision, "heading_arb_yaw_budget_window_sec", yaw_budget_window_sec)
            yaw_budget_abs_deg = self._policy_extra(policy_decision, "heading_arb_yaw_budget_abs_deg", yaw_budget_abs_deg)
            yaw_budget_min_remaining_deg = self._policy_extra(
                policy_decision, "heading_arb_yaw_budget_min_remaining_deg", yaw_budget_min_remaining_deg
            )
            recovery_confirm_hits = int(
                self._policy_extra(policy_decision, "heading_arb_recover_confirm_hits", recovery_confirm_hits)
            )
            recovery_min_score = self._policy_extra(policy_decision, "heading_arb_recover_min_score", recovery_min_score)
            recovery_soft_r_mult = self._policy_extra(
                policy_decision, "heading_arb_recover_soft_r_mult", recovery_soft_r_mult
            )
            recovery_max_update_dyaw_deg = self._policy_extra(
                policy_decision, "heading_arb_recover_max_update_dyaw_deg", recovery_max_update_dyaw_deg
            )

        now_t = float(timestamp)
        reason_codes = []

        vis_delta_deg = float("nan")
        vis_ok = (
            vision_yaw is not None
            and np.isfinite(float(vision_age_sec))
            and 0.0 <= float(vision_age_sec) <= float(max_vis_age)
            and np.isfinite(float(vision_quality))
            and float(vision_quality) >= float(min_vis_quality)
        )
        if vis_ok:
            vis_delta_deg = abs(np.degrees(self._wrap(float(yaw_mag) - float(vision_yaw))))

        state_delta_deg = abs(np.degrees(self._wrap(float(yaw_mag) - float(yaw_state))))

        score_vis = self._lin_score(vis_delta_deg, score_vis_good_deg, hard_mismatch_deg) if vis_ok else float("nan")
        score_gyro = self._lin_score(float(gyro_consistency_delta_deg), score_gyro_good_deg, score_gyro_bad_deg)
        score_state = self._lin_score(state_delta_deg, score_state_good_deg, score_state_bad_deg)

        score_vals = []
        score_weights = []
        for score, weight in ((score_vis, w_vis), (score_gyro, w_gyro), (score_state, w_state)):
            if np.isfinite(score) and np.isfinite(weight) and weight > 0.0:
                score_vals.append(float(score))
                score_weights.append(float(weight))

        if len(score_vals) == 0:
            score_now = 0.5
        else:
            score_now = float(np.dot(score_vals, score_weights) / max(1e-9, float(np.sum(score_weights))))
        score_now = float(np.clip(score_now, 0.0, 1.0))
        if np.isfinite(self._score_ema):
            a = float(np.clip(score_ema_alpha, 0.02, 0.90))
            self._score_ema = (1.0 - a) * self._score_ema + a * score_now
        else:
            self._score_ema = score_now
        score_eff = float(np.clip(self._score_ema, 0.0, 1.0))

        # Mismatch streak drives hold state.
        mismatch_hit = (
            (vis_ok and np.isfinite(vis_delta_deg) and vis_delta_deg >= hard_mismatch_deg)
            or (score_eff < score_hold_thresh)
        )
        if mismatch_hit:
            self._mismatch_streak += 1
        else:
            self._mismatch_streak = max(0, self._mismatch_streak - 1)

        if self._mismatch_streak >= max(1, int(streak_to_hold)):
            self._mismatch_streak = 0
            self._hold_until = max(self._hold_until, now_t + max(0.0, hold_sec))
            self._recovery_active = True
            self._recovery_good_hits = 0
            return HeadingArbResult(
                skip=True,
                r_mult=1.0,
                reason="arb_hold_start_streak",
                consistency_score=score_eff,
                max_update_dyaw_deg=0.0,
                authority="HOLD",
            )

        if now_t < self._hold_until:
            return HeadingArbResult(
                skip=True,
                r_mult=1.0,
                reason="arb_hold_active",
                consistency_score=score_eff,
                max_update_dyaw_deg=0.0,
                authority="HOLD",
            )

        # Yaw injection budget guard (sliding window).
        used_budget_deg = self._prune_injection_hist(now_t, yaw_budget_window_sec)
        remaining_budget_deg = float(yaw_budget_abs_deg) - float(used_budget_deg)
        if np.isfinite(remaining_budget_deg) and remaining_budget_deg <= float(yaw_budget_min_remaining_deg):
            # Short hold to force recovery behavior and break injection bursts.
            self._hold_until = max(self._hold_until, now_t + max(0.2, 0.35 * hold_sec))
            self._recovery_active = True
            self._recovery_good_hits = 0
            return HeadingArbResult(
                skip=True,
                r_mult=1.0,
                reason="arb_hold_budget_exhausted",
                consistency_score=score_eff,
                max_update_dyaw_deg=0.0,
                authority="HOLD",
            )

        r_mult = 1.0
        authority = "MAG_PRIMARY"
        if score_eff < score_soft_thresh:
            r_mult *= max(1.0, float(soft_r_mult))
            reason_codes.append("arb_soft_authority")
            authority = "MAG_SOFT"

        max_update_dyaw_deg = float("inf")
        if np.isfinite(remaining_budget_deg):
            max_update_dyaw_deg = max(0.05, float(remaining_budget_deg))

        # Recovery: after hold, require consecutive good scores before full authority.
        if self._recovery_active:
            if score_eff >= recovery_min_score:
                self._recovery_good_hits += 1
            else:
                self._recovery_good_hits = 0
            r_mult *= max(1.0, float(recovery_soft_r_mult))
            max_update_dyaw_deg = min(max_update_dyaw_deg, max(0.05, float(recovery_max_update_dyaw_deg)))
            reason_codes.append(f"arb_recover_{self._recovery_good_hits}/{max(1, recovery_confirm_hits)}")
            authority = "RECOVER"
            if self._recovery_good_hits >= max(1, int(recovery_confirm_hits)):
                self._recovery_active = False
                self._recovery_good_hits = 0
                reason_codes.append("arb_recover_done")
                authority = "MAG_PRIMARY"

        return HeadingArbResult(
            skip=False,
            r_mult=float(max(1.0, r_mult)),
            reason="|".join(reason_codes),
            consistency_score=score_eff,
            max_update_dyaw_deg=float(max_update_dyaw_deg),
            authority=authority,
        )

    def register_applied_correction(self, timestamp: float, abs_yaw_deg: float):
        """Track accepted yaw injection for sliding-window budget."""
        if not np.isfinite(abs_yaw_deg):
            return
        val = abs(float(abs_yaw_deg))
        if val <= 1e-6:
            return
        self._yaw_injection_hist.append((float(timestamp), val))
