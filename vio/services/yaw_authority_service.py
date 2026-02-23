"""Global yaw authority service.

Single-owner yaw governance across MAG / LOOP / BACKEND (and VPS soft-only gating):
- select one yaw owner at a time (or HOLD)
- cap cumulative yaw injection in a sliding window
- cap yaw injection rate (deg/s)
- confidence-based owner switching with hysteresis/cooldown
- force soft-only behavior for loop/vps/backend at high speed or unstable state
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class YawAuthorityDecision:
    """Decision for one yaw-injection request."""

    source: str
    owner: str
    allow: bool
    mode: str  # APPLY | SOFT_APPLY | HOLD | SKIP
    reason: str
    r_mult: float = 1.0
    max_update_dyaw_deg: float = float("inf")
    confidence: float = float("nan")


class YawAuthorityService:
    """Stateful global yaw authority manager."""

    _YAW_SOURCES: Tuple[str, ...] = ("MAG", "LOOP", "BACKEND")

    def __init__(self, runner: Any):
        self.runner = runner
        self._owner: str = "HOLD"
        self._owner_switch_t: float = -1e9
        self._hold_until_t: float = -1e9
        self._source_score_ema: Dict[str, float] = {}
        self._yaw_hist: deque[Tuple[float, float, str]] = deque()
        self._source_last_seen_t: Dict[str, float] = {}

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, float(x))))

    @staticmethod
    def _finite_or(val: float, default: float) -> float:
        try:
            f = float(val)
        except Exception:
            return float(default)
        return float(f if np.isfinite(f) else default)

    def _cfg(self, key: str, default: float) -> float:
        cfg = self.runner.global_config if isinstance(self.runner.global_config, dict) else {}
        return self._finite_or(cfg.get(key, default), default)

    def _cfg_bool(self, key: str, default: bool) -> bool:
        cfg = self.runner.global_config if isinstance(self.runner.global_config, dict) else {}
        return bool(cfg.get(key, default))

    def _cfg_map(self, key: str, default: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        cfg = self.runner.global_config if isinstance(self.runner.global_config, dict) else {}
        raw = cfg.get(key, default or {})
        if not isinstance(raw, dict):
            return dict(default or {})
        out: Dict[str, float] = {}
        for k, v in raw.items():
            try:
                out[str(k).upper()] = float(v)
            except Exception:
                continue
        return out

    def _enabled(self) -> bool:
        return bool(self._cfg_bool("YAW_AUTH_ENABLE", False))

    def _stage(self) -> int:
        return int(round(self._cfg("YAW_AUTH_STAGE", 0.0)))

    def _prune_hist(self, t_now: float, window_sec: float) -> None:
        cutoff = float(t_now) - max(0.01, float(window_sec))
        while self._yaw_hist and float(self._yaw_hist[0][0]) < cutoff:
            self._yaw_hist.popleft()

    def _window_sum(self, t_now: float, window_sec: float, source: Optional[str] = None) -> float:
        self._prune_hist(float(t_now), float(window_sec))
        if source is None:
            vals = (float(v) for _, v, _ in self._yaw_hist)
        else:
            source_u = str(source).upper()
            vals = (float(v) for _, v, s in self._yaw_hist if str(s).upper() == source_u)
        return float(sum(abs(v) for v in vals))

    def _update_score_ema(self, source: str, confidence: float) -> float:
        source_u = str(source).upper()
        conf = self._clamp(self._finite_or(confidence, 0.5), 0.0, 1.0)
        alpha = self._clamp(self._cfg("YAW_AUTH_SCORE_EMA_ALPHA", 0.22), 0.02, 0.95)
        prev = self._source_score_ema.get(source_u, np.nan)
        if np.isfinite(prev):
            ema = (1.0 - alpha) * float(prev) + alpha * conf
        else:
            ema = conf
        self._source_score_ema[source_u] = float(self._clamp(ema, 0.0, 1.0))
        return float(self._source_score_ema[source_u])

    def _source_priority(self, source: str) -> float:
        pri = self._cfg_map(
            "YAW_AUTH_SOURCE_PRIORITY",
            {"MAG": 1.0, "LOOP": 1.0, "BACKEND": 0.9},
        )
        return float(pri.get(str(source).upper(), 0.8))

    def _stage12_claim_sources(self) -> set[str]:
        raw = self.runner.global_config.get(
            "YAW_AUTH_STAGE12_CLAIM_SOURCES",
            ["LOOP", "BACKEND"],
        ) if isinstance(self.runner.global_config, dict) else ["LOOP", "BACKEND"]
        if not isinstance(raw, (list, tuple)):
            return {"LOOP", "BACKEND"}
        out = {str(v).upper() for v in raw if str(v).strip() != ""}
        return out if out else {"LOOP", "BACKEND"}

    def _maybe_handoff_stage12(self, source: str, t_now: float, confidence: float, speed_m_s: float) -> str:
        """Controlled owner handoff for stage 1/2 to avoid MAG lockout all flight."""
        if self._stage() >= 3:
            return ""
        if not self._cfg_bool("YAW_AUTH_STAGE12_CLAIM_ENABLE", True):
            return ""

        src = str(source).upper()
        owner = str(self._owner).upper()
        if owner == "HOLD" or src == owner:
            return ""

        min_switch_dt = max(
            0.0,
            self._cfg(
                "YAW_AUTH_STAGE12_MIN_SWITCH_INTERVAL_SEC",
                self._cfg("YAW_AUTH_SWITCH_MIN_INTERVAL_SEC", 0.75),
            ),
        )
        if (float(t_now) - float(self._owner_switch_t)) < min_switch_dt:
            return ""

        claim_sources = self._stage12_claim_sources()
        claim_min_score = self._clamp(self._cfg("YAW_AUTH_STAGE12_CLAIM_MIN_SCORE", 0.45), 0.0, 1.0)
        claim_margin = self._clamp(self._cfg("YAW_AUTH_STAGE12_CLAIM_MARGIN", 0.05), 0.0, 1.0)
        owner_timeout_sec = max(0.1, self._cfg("YAW_AUTH_STAGE12_OWNER_TIMEOUT_SEC", 1.2))
        allow_stale_reclaim_any = self._cfg_bool("YAW_AUTH_STAGE12_ALLOW_STALE_RECLAIM_ANY", True)
        loop_force_claim_min_score = self._clamp(
            self._cfg("YAW_AUTH_STAGE12_LOOP_FORCE_CLAIM_MIN_SCORE", 0.62), 0.0, 1.0
        )
        loop_force_claim_max_speed = max(
            0.0, self._cfg("YAW_AUTH_STAGE12_LOOP_FORCE_CLAIM_MAX_SPEED_M_S", 35.0)
        )

        owner_last_t = float(self._source_last_seen_t.get(owner, -1e9))
        owner_stale = (float(t_now) - owner_last_t) >= owner_timeout_sec

        src_score = self._clamp(self._source_score_ema.get(src, confidence), 0.0, 1.0)
        owner_score = self._clamp(self._source_score_ema.get(owner, 0.0), 0.0, 1.0)
        src_eligible = src_score >= claim_min_score

        if owner_stale and (allow_stale_reclaim_any or src_eligible):
            self._owner = src
            self._owner_switch_t = float(t_now)
            return "stage12_claim_owner_stale"

        if src not in claim_sources:
            return ""
        if owner == "MAG" and src_eligible and (src_score >= (owner_score + claim_margin)):
            self._owner = src
            self._owner_switch_t = float(t_now)
            return "stage12_claim_from_mag"
        if (
            owner == "MAG"
            and src == "LOOP"
            and src_score >= loop_force_claim_min_score
            and (not np.isfinite(speed_m_s) or float(speed_m_s) <= loop_force_claim_max_speed)
        ):
            self._owner = src
            self._owner_switch_t = float(t_now)
            return "stage12_claim_loop_force"
        return ""

    def _select_best_source(self, candidates: Iterable[str]) -> Tuple[str, float]:
        min_score = self._clamp(self._cfg("YAW_AUTH_MIN_SOURCE_SCORE", 0.35), 0.0, 1.0)
        best_src = "HOLD"
        best_val = -1.0
        for src in candidates:
            s = str(src).upper()
            score = self._source_score_ema.get(s, np.nan)
            if not np.isfinite(score) or float(score) < min_score:
                continue
            val = float(score) + 1e-3 * self._source_priority(s)
            if val > best_val:
                best_val = val
                best_src = s
        return best_src, (best_val if best_src != "HOLD" else -1.0)

    def _switch_owner_if_needed(self, t_now: float) -> None:
        stage = self._stage()
        if stage < 3:
            return

        candidates = list(self._YAW_SOURCES)
        best_src, _ = self._select_best_source(candidates)
        if best_src == "HOLD":
            hold_score_th = self._clamp(self._cfg("YAW_AUTH_HOLD_SCORE_THRESHOLD", 0.18), 0.0, 1.0)
            cur_score = float(self._source_score_ema.get(self._owner, 0.0))
            if self._owner != "HOLD" and cur_score < hold_score_th:
                self._owner = "HOLD"
                self._owner_switch_t = float(t_now)
            return

        if self._owner == "HOLD":
            self._owner = str(best_src)
            self._owner_switch_t = float(t_now)
            return

        if str(best_src) == str(self._owner):
            return

        switch_margin = self._clamp(self._cfg("YAW_AUTH_SWITCH_MARGIN", 0.12), 0.0, 1.0)
        switch_min_dt = max(0.0, self._cfg("YAW_AUTH_SWITCH_MIN_INTERVAL_SEC", 0.75))
        cur_score = float(self._source_score_ema.get(str(self._owner).upper(), 0.0))
        best_score = float(self._source_score_ema.get(str(best_src).upper(), 0.0))
        if (float(t_now) - float(self._owner_switch_t)) < switch_min_dt:
            return
        if best_score >= (cur_score + switch_margin):
            self._owner = str(best_src)
            self._owner_switch_t = float(t_now)

    def _state_unstable(
        self,
        health_state: str,
        p_max: float,
        p_cond: float,
    ) -> bool:
        health = str(health_state).upper()
        unstable_health_raw = self.runner.global_config.get(
            "YAW_AUTH_SOFT_ONLY_UNSTABLE_HEALTH",
            ["WARNING", "DEGRADED"],
        ) if isinstance(self.runner.global_config, dict) else ["WARNING", "DEGRADED"]
        unstable_health = set()
        if isinstance(unstable_health_raw, (list, tuple)):
            unstable_health = {str(v).upper() for v in unstable_health_raw}
        pmax_th = self._cfg("YAW_AUTH_SOFT_ONLY_UNSTABLE_PMAX", 1.0e6)
        pcond_th = self._cfg("YAW_AUTH_SOFT_ONLY_UNSTABLE_PCOND", 1.0e12)
        return bool(
            health in unstable_health
            or (np.isfinite(p_max) and float(p_max) > float(pmax_th))
            or (np.isfinite(p_cond) and float(p_cond) > float(pcond_th))
        )

    def should_force_soft_only(
        self,
        source: str,
        timestamp: float,
        speed_m_s: float,
        health_state: str,
        p_max: float,
        p_cond: float,
    ) -> Tuple[bool, str]:
        if not self._enabled() or self._stage() < 4:
            return False, ""
        src = str(source).upper()
        if src not in ("LOOP", "VPS", "BACKEND"):
            return False, ""
        speed_soft = self._cfg("YAW_AUTH_SOFT_ONLY_HIGH_SPEED_M_S", 22.0)
        high_speed = np.isfinite(speed_m_s) and float(speed_m_s) >= float(speed_soft)
        unstable = self._state_unstable(health_state=health_state, p_max=p_max, p_cond=p_cond)
        if high_speed and unstable:
            return True, "high_speed+unstable"
        if high_speed:
            return True, "high_speed"
        if unstable:
            return True, "unstable_state"
        return False, ""

    def request_decision(
        self,
        source: str,
        timestamp: float,
        requested_abs_dyaw_deg: float,
        confidence: float,
        speed_m_s: float,
        health_state: str = "HEALTHY",
        p_max: float = float("nan"),
        p_cond: float = float("nan"),
    ) -> YawAuthorityDecision:
        src = str(source).upper()
        t_now = float(timestamp)
        req_abs = max(0.0, self._finite_or(requested_abs_dyaw_deg, 0.0))
        conf = self._update_score_ema(src, confidence)
        self._source_last_seen_t[src] = float(t_now)

        if not self._enabled() or self._stage() <= 0:
            return YawAuthorityDecision(
                source=src,
                owner=src,
                allow=True,
                mode="APPLY",
                reason="yaw_auth_disabled",
                r_mult=1.0,
                max_update_dyaw_deg=float("inf"),
                confidence=conf,
            )

        claim_reason = ""
        # Stage-1 lazy owner (single owner, no score switching).
        if self._owner == "HOLD" and self._stage() < 3:
            self._owner = src
            self._owner_switch_t = float(t_now)
        elif self._stage() < 3:
            claim_reason = self._maybe_handoff_stage12(
                source=src,
                t_now=float(t_now),
                confidence=conf,
                speed_m_s=float(speed_m_s),
            )

        # Stage-3: confidence/hysteresis switching.
        self._switch_owner_if_needed(float(t_now))

        if float(t_now) < float(self._hold_until_t):
            return YawAuthorityDecision(
                source=src,
                owner="HOLD",
                allow=False,
                mode="HOLD",
                reason="hold_active",
                confidence=conf,
            )

        owner = str(self._owner).upper()
        if owner == "HOLD":
            return YawAuthorityDecision(
                source=src,
                owner=owner,
                allow=False,
                mode="HOLD",
                reason="owner_hold",
                confidence=conf,
            )

        if src != owner:
            return YawAuthorityDecision(
                source=src,
                owner=owner,
                allow=False,
                mode="HOLD",
                reason=f"owner_is_{owner.lower()}",
                confidence=conf,
            )

        mode = "APPLY"
        reason_codes = [claim_reason] if claim_reason else []
        r_mult = 1.0
        max_update = float("inf")

        # Stage-2: cumulative/rate budget.
        if self._stage() >= 2 and req_abs > 1e-6:
            win_sec = max(0.25, self._cfg("YAW_AUTH_YAW_BUDGET_WINDOW_SEC", 8.0))
            max_cum = max(0.2, self._cfg("YAW_AUTH_YAW_BUDGET_ABS_DEG", 10.0))
            max_rate = max(0.2, self._cfg("YAW_AUTH_YAW_RATE_MAX_DEG_S", 4.0))
            used_cum = self._window_sum(t_now=t_now, window_sec=win_sec, source=None)
            used_rate = self._window_sum(t_now=t_now, window_sec=1.0, source=None)
            rem_cum = max(0.0, float(max_cum) - float(used_cum))
            rem_rate = max(0.0, float(max_rate) - float(used_rate))
            max_update = min(max_update, rem_cum, rem_rate)
            if max_update <= 0.05:
                hold_sec = max(0.1, self._cfg("YAW_AUTH_HOLD_SEC", 0.8))
                self._hold_until_t = max(float(self._hold_until_t), float(t_now + hold_sec))
                return YawAuthorityDecision(
                    source=src,
                    owner=owner,
                    allow=False,
                    mode="HOLD",
                    reason="budget_exhausted",
                    confidence=conf,
                )
            if max_update < req_abs:
                mode = "SOFT_APPLY"
                reason_codes.append("budget_soft_cap")
                r_mult *= max(1.0, self._cfg("YAW_AUTH_BUDGET_SOFT_R_MULT", 1.6))

        # Stage-4: soft-only on high-speed/unstable for loop/vps/backend.
        force_soft, soft_reason = self.should_force_soft_only(
            source=src,
            timestamp=t_now,
            speed_m_s=speed_m_s,
            health_state=health_state,
            p_max=p_max,
            p_cond=p_cond,
        )
        if force_soft:
            mode = "SOFT_APPLY"
            reason_codes.append(f"soft_only_{soft_reason}")
            r_mult *= max(1.0, self._cfg("YAW_AUTH_SOFT_ONLY_R_MULT", 1.5))
            max_update = min(max_update, max(0.05, self._cfg("YAW_AUTH_SOFT_ONLY_MAX_DYAW_DEG", 1.2)))

        reason = "|".join(reason_codes) if reason_codes else "owner_apply"
        return YawAuthorityDecision(
            source=src,
            owner=owner,
            allow=True,
            mode=mode,
            reason=reason,
            r_mult=float(max(1.0, r_mult)),
            max_update_dyaw_deg=float(max_update),
            confidence=conf,
        )

    def register_applied(self, source: str, timestamp: float, abs_yaw_deg: float) -> None:
        """Track applied yaw injection for budget/rate guards."""
        if not self._enabled() or self._stage() <= 0:
            return
        val = self._finite_or(abs_yaw_deg, 0.0)
        if val <= 1e-6:
            return
        src = str(source).upper()
        self._yaw_hist.append((float(timestamp), float(abs(val)), src))
        self._prune_hist(float(timestamp), max(0.25, self._cfg("YAW_AUTH_YAW_BUDGET_WINDOW_SEC", 8.0)))
