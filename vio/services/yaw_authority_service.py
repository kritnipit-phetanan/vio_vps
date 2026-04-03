"""Global yaw authority service.

Single-owner yaw governance across MAG / LOOP / BACKEND (and VPS soft-only gating):
- select one yaw owner at a time (or HOLD)
- cap cumulative yaw injection in a sliding window
- cap yaw injection rate (deg/s)
- confidence-based owner switching with hysteresis/cooldown
- force soft-only behavior for loop/vps/backend at high speed or unstable state
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from ..output_utils import log_heading_owner_trace


@dataclass(frozen=True)
class YawAuthorityDecision:
    """Decision for one yaw-injection request."""

    source: str
    owner: str
    allow: bool
    mode: str  # APPLY | SOFT_APPLY | SOFT_SPRING | HOLD | SKIP
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
        self._last_switch_eval_t: float = -1e9
        self._hold_until_t: float = -1e9
        self._source_score_ema: Dict[str, float] = {}
        self._yaw_hist: deque[Tuple[float, float, str]] = deque()
        self._source_last_seen_t: Dict[str, float] = {}
        self._source_request_hist: Dict[str, deque[float]] = defaultdict(deque)
        self._source_apply_hist: Dict[str, deque[float]] = defaultdict(deque)
        self._owner_switch_count: int = 0
        self._owner_source_counts: Dict[str, int] = defaultdict(int)
        self._owner_samples: int = 0
        self._source_accept_hist: Dict[str, deque[Tuple[float, int]]] = defaultdict(deque)
        self._source_owner_block_until_t: Dict[str, float] = defaultdict(lambda: -1e9)
        self._source_owner_block_count: Dict[str, int] = defaultdict(int)
        self._source_last_applied_t: Dict[str, float] = defaultdict(lambda: -1e9)
        self._decision_samples: int = 0
        self._mag_owner_block_samples: int = 0
        self._owner_dead_fallback_count: int = 0
        self._owner_dead_last_t: float = -1e9
        self._last_loop_hold_reclaim_t: float = -1e9

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

    def _cfg_source(self, key: str, source: str, default: float) -> float:
        """Read scalar key with optional source-specific override map."""
        src = str(source).upper()
        base = self._cfg(key, default)
        map_key = f"{key}_MAP"
        ov_map = self._cfg_map(map_key, {})
        if src in ov_map:
            return self._finite_or(ov_map.get(src, base), base)
        return float(base)

    def _enabled(self) -> bool:
        return bool(self._cfg_bool("YAW_AUTH_ENABLE", False))

    def _stage(self) -> int:
        return int(round(self._cfg("YAW_AUTH_STAGE", 0.0)))

    def _owner_min_dwell_sec(self) -> float:
        return max(0.0, self._cfg("YAW_AUTH_OWNER_MIN_DWELL_SEC", 0.0))

    def _set_owner(self, owner: str, t_now: float) -> bool:
        owner_u = str(owner).upper()
        prev = str(self._owner).upper()
        switched = owner_u != prev
        self._owner = owner_u
        if switched:
            self._owner_switch_t = float(t_now)
            self._owner_switch_count += 1
        return switched

    def set_external_confidence(self, source: str, timestamp: float, confidence: float) -> float:
        """Feed non-yaw-update confidence (e.g., MSCKF quality) into switching scores."""
        src = str(source).upper()
        ema = self._update_score_ema(src, confidence)
        self._source_last_seen_t[src] = float(timestamp)
        return float(ema)

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

    def _source_accept_rate(self, source: str, t_now: float, window_sec: float) -> Tuple[float, int]:
        src = str(source).upper()
        hist = self._source_accept_hist.get(src)
        if not hist:
            return float("nan"), 0
        cutoff = float(t_now) - max(0.1, float(window_sec))
        while hist and float(hist[0][0]) < cutoff:
            hist.popleft()
        if len(hist) == 0:
            return float("nan"), 0
        vals = np.asarray([float(v) for _, v in hist], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan"), 0
        return float(np.mean(vals)), int(vals.size)

    def _append_ts_event(
        self,
        hist_map: Dict[str, deque[float]],
        source: str,
        timestamp: float,
        max_keep: int,
    ) -> None:
        src = str(source).upper()
        hist = hist_map[src]
        hist.append(float(timestamp))
        while len(hist) > max(16, int(max_keep)):
            hist.popleft()

    def _count_recent_events(
        self,
        hist_map: Dict[str, deque[float]],
        source: str,
        t_now: float,
        window_sec: float,
    ) -> int:
        src = str(source).upper()
        hist = hist_map.get(src)
        if not hist:
            return 0
        cutoff = float(t_now) - max(0.1, float(window_sec))
        while hist and float(hist[0]) < cutoff:
            hist.popleft()
        return int(len(hist))

    def _source_apply_efficiency(self, source: str, t_now: float) -> Tuple[float, int]:
        """
        Estimate source effectiveness from request/apply events in a rolling window.

        Returns:
            (efficiency in [0,1], request_count)
        """
        if not self._cfg_bool("YAW_AUTH_ACTIVITY_ENABLE", True):
            return 1.0, 0
        win_sec = max(1.0, self._cfg("YAW_AUTH_ACTIVITY_WINDOW_SEC", 8.0))
        req_cnt = self._count_recent_events(
            self._source_request_hist,
            source=source,
            t_now=float(t_now),
            window_sec=win_sec,
        )
        if req_cnt <= 0:
            return 1.0, 0
        app_cnt = self._count_recent_events(
            self._source_apply_hist,
            source=source,
            t_now=float(t_now),
            window_sec=win_sec,
        )
        eff = float(app_cnt) / float(max(1, req_cnt))
        return float(self._clamp(eff, 0.0, 1.0)), int(req_cnt)

    def _non_mag_apply_recent(self, t_now: float) -> int:
        """Count recent LOOP/BACKEND applies for MAG block fallback logic."""
        win_sec = max(1.0, self._cfg("YAW_AUTH_MAG_BLOCK_REQUIRE_ALT_WINDOW_SEC", 8.0))
        loop_n = self._count_recent_events(
            self._source_apply_hist, source="LOOP", t_now=float(t_now), window_sec=win_sec
        )
        back_n = self._count_recent_events(
            self._source_apply_hist, source="BACKEND", t_now=float(t_now), window_sec=win_sec
        )
        return int(loop_n + back_n)

    def _owner_apply_recent_ok(self, source: str, t_now: float) -> bool:
        """
        Owner eligibility gate from recent effective applies.

        Prevents stale owner lock (e.g., BACKEND owns yaw but applies too rarely).
        """
        if not self._cfg_bool("YAW_AUTH_OWNER_REQUIRE_APPLY_RECENT_ENABLE", True):
            return True
        src = str(source).upper()
        if src == "HOLD":
            return True
        # Keep MAG available as weak anchor when no strong alternative is active.
        if src == "MAG":
            return True
        win_sec = max(0.5, self._cfg("YAW_AUTH_OWNER_APPLY_RECENT_WINDOW_SEC", 8.0))
        raw_map = self.runner.global_config.get(
            "YAW_AUTH_OWNER_MIN_APPLY_RECENT_MAP",
            {"BACKEND": 2},
        ) if isinstance(self.runner.global_config, dict) else {"BACKEND": 2}
        min_map = raw_map if isinstance(raw_map, dict) else {"BACKEND": 2}
        try:
            min_recent = int(round(float(min_map.get(src, 0))))
        except Exception:
            min_recent = 0
        if min_recent <= 0:
            return True
        app_cnt = self._count_recent_events(
            self._source_apply_hist,
            source=src,
            t_now=float(t_now),
            window_sec=win_sec,
        )
        if app_cnt >= min_recent:
            return True

        # Request-alive path for sparse/indirect sources.
        # LOOP/BACKEND can legitimately have low "applied yaw" counts while still
        # being active and high-confidence; do not force HOLD immediately in that case.
        if src not in ("BACKEND", "LOOP"):
            return False

        if src == "BACKEND":
            default_enable = self._cfg_bool("YAW_AUTH_BACKEND_REQUEST_ALIVE_ENABLE", True)
        else:
            default_enable = self._cfg_bool("YAW_AUTH_LOOP_REQUEST_ALIVE_ENABLE", True)
        if not default_enable:
            return False

        req_win_default = self._cfg(
            "YAW_AUTH_BACKEND_REQUEST_ALIVE_WINDOW_SEC" if src == "BACKEND" else "YAW_AUTH_LOOP_REQUEST_ALIVE_WINDOW_SEC",
            win_sec,
        )
        req_win = max(
            win_sec,
            self._cfg_source("YAW_AUTH_REQUEST_ALIVE_WINDOW_SEC_MAP", src, req_win_default),
        )
        req_cnt = self._count_recent_events(
            self._source_request_hist,
            source=src,
            t_now=float(t_now),
            window_sec=float(req_win),
        )

        min_req_default = (
            self._cfg("YAW_AUTH_BACKEND_REQUEST_ALIVE_MIN_REQUESTS", self._cfg("YAW_AUTH_BACKEND_BOOTSTRAP_MIN_REQUESTS", 3.0))
            if src == "BACKEND"
            else self._cfg("YAW_AUTH_LOOP_REQUEST_ALIVE_MIN_REQUESTS", 2.0)
        )
        min_req = max(
            1,
            int(round(self._cfg_source("YAW_AUTH_REQUEST_ALIVE_MIN_REQUESTS_MAP", src, min_req_default))),
        )
        min_score_default = (
            self._cfg("YAW_AUTH_BACKEND_REQUEST_ALIVE_MIN_SCORE", self._cfg("YAW_AUTH_BACKEND_BOOTSTRAP_MIN_SCORE", 0.28))
            if src == "BACKEND"
            else self._cfg("YAW_AUTH_LOOP_REQUEST_ALIVE_MIN_SCORE", 0.24)
        )
        min_score = self._clamp(
            self._cfg_source("YAW_AUTH_REQUEST_ALIVE_MIN_SCORE_MAP", src, min_score_default),
            0.0,
            1.0,
        )
        min_fresh = self._clamp(
            self._cfg_source("YAW_AUTH_REQUEST_ALIVE_MIN_FRESHNESS_MAP", src, 0.10),
            0.0,
            1.0,
        )
        src_score = self._clamp(self._source_score_ema.get(src, 0.0), 0.0, 1.0)
        src_fresh = self._source_freshness(src, t_now=float(t_now))
        return bool(req_cnt >= min_req and src_score >= min_score and src_fresh >= min_fresh)

    def _mag_owner_eligible(self, t_now: float) -> bool:
        """Return True when MAG is eligible to own yaw (stage>=3 arbitration)."""
        if (
            bool(self.runner.global_config.get("POSITION_FIRST_LANE", False))
            and bool(self._cfg_bool("YAW_AUTH_POSITION_FIRST_DISABLE_MAG_OWNER", True))
        ):
            return False
        if self._is_source_blocked("MAG", t_now=float(t_now)):
            return False
        # Global effectiveness gate from real applied/rejected outcomes.
        # This avoids persistent MAG ownership when MAG is mostly rejected by EKF updates.
        try:
            state = getattr(self.runner, "state", None)
            mag_updates = float(getattr(state, "mag_updates", 0.0))
            mag_rejects = float(getattr(state, "mag_rejects", 0.0))
            # Owner-arbitration skips are tracked separately and should not demote
            # MAG signal quality in the global accept-rate gate.
            mag_owner_skips = float(getattr(state, "mag_owner_skips", 0.0))
            eff_rejects = max(0.0, float(mag_rejects) - float(mag_owner_skips))
            mag_total = mag_updates + eff_rejects
            min_total = max(1.0, self._cfg("YAW_AUTH_MAG_GLOBAL_MIN_SAMPLES", 80.0))
            min_rate = self._clamp(self._cfg("YAW_AUTH_MAG_GLOBAL_MIN_ACCEPT_RATE", 0.08), 0.0, 1.0)
            if mag_total >= min_total and (mag_updates / max(1.0, mag_total)) < min_rate:
                return False
        except Exception:
            pass
        win_sec = max(1.0, self._cfg("YAW_AUTH_MAG_OWNER_ACCEPT_WINDOW_SEC", 8.0))
        min_samples = max(1, int(round(self._cfg("YAW_AUTH_MAG_OWNER_MIN_SAMPLES", 20.0))))
        min_rate = self._clamp(self._cfg("YAW_AUTH_MAG_OWNER_MIN_ACCEPT_RATE", 0.15), 0.0, 1.0)
        rate, samples = self._source_accept_rate("MAG", t_now=float(t_now), window_sec=win_sec)
        non_mag_req_win = max(0.5, self._cfg("YAW_AUTH_MAG_WARMUP_NONMAG_WINDOW_SEC", 4.0))
        non_mag_requests = int(
            self._count_recent_events(self._source_request_hist, source="LOOP", t_now=float(t_now), window_sec=non_mag_req_win)
            + self._count_recent_events(self._source_request_hist, source="BACKEND", t_now=float(t_now), window_sec=non_mag_req_win)
        )
        if samples < min_samples:
            # Cold-start path: if non-MAG sources are already active, raise MAG ownership bar
            # to avoid early-flight latch to MAG when accept-rate is still unproven.
            warmup_min = self._clamp(self._cfg("YAW_AUTH_MAG_WARMUP_MIN_SCORE", 0.55), 0.0, 1.0)
            if non_mag_requests > 0:
                warmup_min = max(
                    warmup_min,
                    self._clamp(self._cfg("YAW_AUTH_MAG_WARMUP_WITH_NONMAG_MIN_SCORE", 0.85), 0.0, 1.0),
                )
            warmup_min_samples = max(1, int(round(self._cfg("YAW_AUTH_MAG_WARMUP_MIN_SAMPLES", 6.0))))
            if non_mag_requests > 0 and samples < warmup_min_samples:
                return False
            score = self._clamp(self._source_score_ema.get("MAG", 0.0), 0.0, 1.0)
            return bool(score >= warmup_min)
        if not np.isfinite(rate):
            return False
        return bool(float(rate) >= float(min_rate))

    def _owner_has_recent_activity(self, source: str, t_now: float, window_sec: float) -> bool:
        """
        Return True when source is still actively requesting decisions with fresh score.

        This prevents false dead-owner fallback when a source is alive but produces
        tiny/no-op yaw applies for a short period.
        """
        src = str(source).upper()
        req_cnt = self._count_recent_events(
            self._source_request_hist,
            source=src,
            t_now=float(t_now),
            window_sec=float(window_sec),
        )
        raw_map = self.runner.global_config.get(
            "YAW_AUTH_OWNER_DEAD_MIN_REQUESTS_MAP",
            {"BACKEND": 2, "LOOP": 1},
        ) if isinstance(self.runner.global_config, dict) else {"BACKEND": 2, "LOOP": 1}
        min_map = raw_map if isinstance(raw_map, dict) else {"BACKEND": 2, "LOOP": 1}
        try:
            min_req = int(round(float(min_map.get(src, 1))))
        except Exception:
            min_req = 1
        min_req = max(1, min_req)
        freshness = self._source_freshness(src, t_now=float(t_now))
        min_fresh = self._clamp(self._cfg("YAW_AUTH_OWNER_DEAD_MIN_FRESHNESS", 0.20), 0.0, 1.0)
        return bool(req_cnt >= min_req and freshness >= min_fresh)

    def _strict_hold_reclaim_allowed(self, source: str, t_now: float, conf_raw: float, conf_ema: float) -> bool:
        """
        Stage>=3 strict-mode reclaim gate.

        Allows only configured sources (default BACKEND/LOOP) to escape HOLD with
        minimum confidence and freshness, keeping arbitration single-path.
        """
        src = str(source).upper()
        if self._is_source_blocked(src, t_now=float(t_now)):
            return False
        sources_raw = self.runner.global_config.get(
            "YAW_AUTH_STRICT_HOLD_RECLAIM_SOURCES",
            ["BACKEND", "LOOP"],
        ) if isinstance(self.runner.global_config, dict) else ["BACKEND", "LOOP"]
        allowed_sources = {str(v).upper() for v in (sources_raw or []) if str(v).strip() != ""}
        if src not in allowed_sources:
            return False
        base_conf = self._clamp(self._cfg("YAW_AUTH_STRICT_HOLD_RECLAIM_MIN_CONFIDENCE", 0.22), 0.0, 1.0)
        conf_map_raw = self.runner.global_config.get(
            "YAW_AUTH_STRICT_HOLD_RECLAIM_MIN_CONFIDENCE_MAP",
            {},
        ) if isinstance(self.runner.global_config, dict) else {}
        conf_map = conf_map_raw if isinstance(conf_map_raw, dict) else {}
        min_conf = base_conf
        if src in conf_map:
            try:
                min_conf = self._clamp(float(conf_map.get(src)), 0.0, 1.0)
            except Exception:
                min_conf = base_conf
        min_fresh = self._clamp(self._cfg("YAW_AUTH_STRICT_HOLD_RECLAIM_MIN_FRESHNESS", 0.15), 0.0, 1.0)
        src_fresh = self._source_freshness(src, t_now=float(t_now))
        reclaim_conf = max(float(conf_raw), float(conf_ema))
        if reclaim_conf < min_conf or src_fresh < min_fresh:
            return False
        if src == "BACKEND":
            req_win_sec = max(0.5, self._cfg("YAW_AUTH_OWNER_APPLY_RECENT_WINDOW_SEC", 8.0))
            if not self._owner_has_recent_activity(src, t_now=float(t_now), window_sec=float(req_win_sec)):
                return False
        return True

    def _try_hold_interrupt_source(self, source: str, t_now: float, conf_raw: float, conf_ema: float) -> bool:
        """
        Escape HOLD for non-MAG sources when they show fresh, consistent activity.

        This prevents global HOLD lock where frequent MAG soft-updates starve LOOP/BACKEND
        ownership even when those sources are active and confident enough to contribute.
        """
        src = str(source).upper()
        if src not in {"LOOP", "BACKEND"}:
            return False
        if self._is_source_blocked(src, t_now=float(t_now)):
            return False
        min_conf_default = self._clamp(self._cfg("YAW_AUTH_HOLD_INTERRUPT_MIN_CONFIDENCE", 0.22), 0.0, 1.0)
        min_conf = self._clamp(
            self._cfg_source("YAW_AUTH_HOLD_INTERRUPT_MIN_CONFIDENCE_MAP", src, min_conf_default),
            0.0,
            1.0,
        )
        min_fresh_default = self._clamp(self._cfg("YAW_AUTH_HOLD_INTERRUPT_MIN_FRESHNESS", 0.15), 0.0, 1.0)
        min_fresh = self._clamp(
            self._cfg_source("YAW_AUTH_HOLD_INTERRUPT_MIN_FRESHNESS_MAP", src, min_fresh_default),
            0.0,
            1.0,
        )
        req_win_sec = max(
            0.5,
            self._cfg("YAW_AUTH_HOLD_INTERRUPT_WINDOW_SEC", self._cfg("YAW_AUTH_ACTIVITY_WINDOW_SEC", 8.0)),
        )
        min_req_default = max(1, int(round(self._cfg("YAW_AUTH_HOLD_INTERRUPT_MIN_REQUESTS", 1.0))))
        min_req = max(
            1,
            int(round(self._cfg_source("YAW_AUTH_HOLD_INTERRUPT_MIN_REQUESTS_MAP", src, float(min_req_default)))),
        )
        src_score = max(
            float(conf_raw),
            float(conf_ema),
            self._clamp(float(self._source_score_ema.get(src, conf_ema)), 0.0, 1.0),
        )
        src_fresh = self._source_freshness(src, t_now=float(t_now))
        req_cnt = self._count_recent_events(
            self._source_request_hist,
            source=src,
            t_now=float(t_now),
            window_sec=float(req_win_sec),
        )
        if src_score < min_conf or src_fresh < min_fresh or req_cnt < min_req:
            return False
        self._set_owner(src, float(t_now))
        self._hold_until_t = min(float(self._hold_until_t), float(t_now))
        return True

    def _build_hold_mag_anchor_decision(
        self,
        source: str,
        t_now: float,
        conf_raw: float,
        conf_ema: float,
        reason_prefix: str = "",
    ) -> Optional[YawAuthorityDecision]:
        """
        Return weak MAG anchor decision when owner is HOLD and no strong source is active.

        This prevents yaw free-run during long HOLD periods while keeping MAG authority soft.
        """
        if not self._cfg_bool("YAW_AUTH_HOLD_MAG_ANCHOR_ENABLE", True):
            return None
        return self._build_mag_soft_spring_decision(
            source=source,
            t_now=t_now,
            conf_raw=conf_raw,
            conf_ema=conf_ema,
            reason_prefix=reason_prefix if str(reason_prefix).strip() != "" else "hold_mag_anchor",
            owner_hint="HOLD",
        )

    def _build_mag_ineligible_soft_decision(
        self,
        source: str,
        t_now: float,
        conf_raw: float,
        conf_ema: float,
        reason_prefix: str = "",
    ) -> Optional[YawAuthorityDecision]:
        """
        Return bounded MAG soft-apply when MAG is not eligible to own yaw.

        This keeps MAG available as a weak yaw anchor during position-first and
        other owner-eligibility failures instead of dropping it completely.
        """
        return self._build_mag_soft_spring_decision(
            source=source,
            t_now=t_now,
            conf_raw=conf_raw,
            conf_ema=conf_ema,
            reason_prefix=reason_prefix if str(reason_prefix).strip() != "" else "mag_owner_ineligible",
            owner_hint="HOLD",
            ineligible_hint=True,
        )

    def _build_mag_soft_spring_decision(
        self,
        source: str,
        t_now: float,
        conf_raw: float,
        conf_ema: float,
        reason_prefix: str = "",
        owner_hint: Optional[str] = None,
        blocked_hint: Optional[bool] = None,
        ineligible_hint: Optional[bool] = None,
    ) -> Optional[YawAuthorityDecision]:
        """
        Return an aggressively softened MAG ticket.

        This is intentionally permissive: it keeps MAG alive as a weak yaw spring
        even when arbitration would normally drop it, while relying on the update
        path to clamp residuals and inflate R.
        """
        src = str(source).upper()
        if src != "MAG":
            return None
        if not self._cfg_bool("YAW_AUTH_MAG_SOFT_SPRING_ENABLE", True):
            return None
        mag_blocked = bool(self._is_source_blocked("MAG", t_now=float(t_now))) if blocked_hint is None else bool(blocked_hint)
        if mag_blocked and not self._cfg_bool("YAW_AUTH_MAG_SOFT_SPRING_ALLOW_WHEN_BLOCKED", True):
            return None

        soft_conf = max(float(conf_raw), float(conf_ema))
        min_conf = self._clamp(
            self._cfg("YAW_AUTH_MAG_SOFT_SPRING_MIN_CONFIDENCE", 0.0),
            0.0,
            1.0,
        )
        if soft_conf < min_conf:
            return None

        owner_now = str(owner_hint if owner_hint is not None else self._owner).upper()
        non_owner = bool(owner_now not in ("MAG", "HOLD"))
        mag_ineligible = (
            bool(ineligible_hint)
            if ineligible_hint is not None
            else (not self._mag_owner_eligible(float(t_now)))
        )

        base_r_mult = max(1.0, self._cfg("YAW_AUTH_MAG_SOFT_SPRING_R_MULT", 10.0))
        if owner_now == "HOLD":
            base_r_mult = max(base_r_mult, self._cfg("YAW_AUTH_MAG_SOFT_SPRING_HOLD_R_MULT", 8.0))
        if non_owner:
            base_r_mult = max(base_r_mult, self._cfg("YAW_AUTH_MAG_SOFT_SPRING_NONOWNER_R_MULT", 14.0))
        if mag_blocked:
            base_r_mult = max(base_r_mult, self._cfg("YAW_AUTH_MAG_SOFT_SPRING_BLOCKED_R_MULT", 18.0))
        if mag_ineligible:
            base_r_mult = max(base_r_mult, self._cfg("YAW_AUTH_MAG_SOFT_SPRING_INELIGIBLE_R_MULT", 12.0))

        max_dyaw_deg = max(0.05, self._cfg("YAW_AUTH_MAG_SOFT_SPRING_MAX_DYAW_DEG", 0.22))
        if owner_now == "HOLD":
            max_dyaw_deg = min(
                max_dyaw_deg,
                max(0.05, self._cfg("YAW_AUTH_MAG_SOFT_SPRING_HOLD_MAX_DYAW_DEG", 0.18)),
            )
        if non_owner:
            max_dyaw_deg = min(
                max_dyaw_deg,
                max(0.05, self._cfg("YAW_AUTH_MAG_SOFT_SPRING_NONOWNER_MAX_DYAW_DEG", 0.12)),
            )
        if mag_blocked:
            max_dyaw_deg = min(
                max_dyaw_deg,
                max(0.03, self._cfg("YAW_AUTH_MAG_SOFT_SPRING_BLOCKED_MAX_DYAW_DEG", 0.08)),
            )
        if mag_ineligible:
            max_dyaw_deg = min(
                max_dyaw_deg,
                max(0.04, self._cfg("YAW_AUTH_MAG_SOFT_SPRING_INELIGIBLE_MAX_DYAW_DEG", max_dyaw_deg)),
            )

        reason = "mag_soft_spring"
        if str(reason_prefix).strip() != "":
            reason = f"{str(reason_prefix)}|{reason}"
        return YawAuthorityDecision(
            source=src,
            owner=owner_now,
            allow=True,
            mode="SOFT_SPRING",
            reason=reason,
            r_mult=float(base_r_mult),
            max_update_dyaw_deg=float(max_dyaw_deg),
            confidence=float(conf_ema),
        )

    def _is_source_blocked(self, source: str, t_now: float) -> bool:
        src = str(source).upper()
        until_t = float(self._source_owner_block_until_t.get(src, -1e9))
        return bool(float(t_now) < until_t)

    def _update_source_blocks(self, t_now: float) -> None:
        """Temporarily block MAG ownership when rolling accept rate collapses."""
        require_alt = self._cfg_bool("YAW_AUTH_MAG_BLOCK_REQUIRE_ALT_SOURCE", False)
        min_alt_applies = max(0, int(round(self._cfg("YAW_AUTH_MAG_BLOCK_MIN_ALT_APPLIES", 1.0))))
        min_rate = self._clamp(self._cfg("YAW_AUTH_MAG_OWNER_MIN_ACCEPT_RATE", 0.15), 0.0, 1.0)
        force_rate = self._clamp(self._cfg("YAW_AUTH_MAG_BLOCK_FORCE_RATE", 0.03), 0.0, 1.0)
        win_sec = max(1.0, self._cfg("YAW_AUTH_MAG_OWNER_ACCEPT_WINDOW_SEC", 8.0))
        min_samples = max(1, int(round(self._cfg("YAW_AUTH_MAG_OWNER_MIN_SAMPLES", 20.0))))
        ban_sec = max(0.2, self._cfg("YAW_AUTH_MAG_OWNER_BAN_SEC", 2.5))
        force_ban_sec = max(ban_sec, self._cfg("YAW_AUTH_MAG_BLOCK_FORCE_BAN_SEC", 8.0))
        mag_rate, mag_samples = self._source_accept_rate("MAG", t_now=float(t_now), window_sec=win_sec)
        if mag_samples < min_samples or not np.isfinite(mag_rate):
            return
        mag_rate_f = float(mag_rate)
        force_block = bool(mag_rate_f <= force_rate)
        if not force_block and mag_rate_f >= float(min_rate):
            return
        if require_alt and not force_block:
            non_mag_recent = self._non_mag_apply_recent(t_now=float(t_now))
            if non_mag_recent < min_alt_applies:
                # Do not hard-block MAG when no other source is actively applying.
                return
        # Increase block duration as accept-rate collapses to avoid rapid owner relatch.
        deficit = max(0.0, (float(min_rate) - mag_rate_f) / max(1e-6, float(min_rate)))
        dynamic_ban = float(ban_sec) * (1.0 + min(2.0, deficit))
        if force_block:
            dynamic_ban = max(dynamic_ban, float(force_ban_sec))
        prev_until = float(self._source_owner_block_until_t.get("MAG", -1e9))
        next_until = max(prev_until, float(t_now) + float(dynamic_ban))
        if next_until > prev_until + 1e-6:
            self._source_owner_block_count["MAG"] += 1
        self._source_owner_block_until_t["MAG"] = float(next_until)

    def _apply_dead_owner_fallback(self, t_now: float) -> str:
        """
        Fallback owner -> HOLD when current owner is inactive (no applied yaw for too long).

        This prevents stage-2/3 owner latch where LOOP/BACKEND keep ownership despite
        not contributing effective corrections.
        """
        if not self._cfg_bool("YAW_AUTH_OWNER_DEAD_ENABLE", True):
            return ""
        owner = str(self._owner).upper()
        if owner == "HOLD":
            return ""
        dead_rearm_sec = max(0.0, self._cfg("YAW_AUTH_OWNER_DEAD_REARM_SEC", 0.8))
        if (float(t_now) - float(self._owner_dead_last_t)) < dead_rearm_sec:
            return ""
        dead_timeout_sec = self._owner_dead_timeout_sec(owner)
        dead_hold_sec = max(0.0, self._cfg("YAW_AUTH_OWNER_DEAD_HOLD_SEC", 0.35))
        last_applied_t = float(self._source_last_applied_t.get(owner, -1e9))
        time_since_switch = float(t_now) - float(self._owner_switch_t)
        if last_applied_t < -1e8:
            owner_dead = bool(time_since_switch > dead_timeout_sec)
        else:
            owner_dead = bool((float(t_now) - last_applied_t) > dead_timeout_sec)
        if owner_dead and owner in ("BACKEND", "LOOP"):
            # Keep active sources from being forced into HOLD solely because
            # applied yaw is temporarily tiny/zero.
            if self._owner_has_recent_activity(owner, t_now=float(t_now), window_sec=float(dead_timeout_sec)):
                owner_dead = False
        if not owner_dead:
            return ""
        self._set_owner("HOLD", float(t_now))
        if dead_hold_sec > 0.0:
            self._hold_until_t = max(float(self._hold_until_t), float(t_now) + float(dead_hold_sec))
        self._owner_dead_fallback_count += 1
        self._owner_dead_last_t = float(t_now)
        return f"owner_dead_timeout_{owner.lower()}"

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

    def _modulate_confidence_by_msckf(self, source: str, confidence: float) -> float:
        """
        Optionally fuse source confidence with MSCKF quality confidence.

        When enabled, this reduces ownership bias toward a source when MSCKF quality
        is persistently low, and stabilizes switching behavior under weak visual support.
        """
        conf = self._clamp(confidence, 0.0, 1.0)
        if not self._cfg_bool("YAW_AUTH_USE_MSCKF_CONFIDENCE", True):
            return conf
        src = str(source).upper()
        if src not in self._YAW_SOURCES:
            return conf
        msckf_q = self._source_score_ema.get("MSCKF", np.nan)
        if not np.isfinite(msckf_q):
            return conf
        w_map = self._cfg_map(
            "YAW_AUTH_MSCKF_CONF_BLEND",
            {"MAG": 0.60, "LOOP": 0.20, "BACKEND": 0.05},
        )
        w = self._clamp(float(w_map.get(src, 0.25)), 0.0, 0.95)
        return float(self._clamp((1.0 - w) * conf + w * float(msckf_q), 0.0, 1.0))

    def _source_priority(self, source: str) -> float:
        pri = self._cfg_map(
            "YAW_AUTH_SOURCE_PRIORITY",
            {"MAG": 1.0, "LOOP": 1.0, "BACKEND": 0.9},
        )
        return float(pri.get(str(source).upper(), 0.8))

    def _source_min_score(self, source: str, default_min: float) -> float:
        """Per-source minimum score threshold used by owner selector."""
        src = str(source).upper()
        raw = self._cfg_map("YAW_AUTH_MIN_SOURCE_SCORE_MAP", {})
        if src in raw:
            return self._clamp(float(raw[src]), 0.0, 1.0)
        return self._clamp(float(default_min), 0.0, 1.0)

    def _mag_score_scale(self, t_now: float) -> float:
        """
        Continuous MAG score scaling from rolling accept-rate.

        This avoids binary behavior where MAG is either fully dominant or fully blocked.
        """
        if not self._cfg_bool("YAW_AUTH_MAG_SCORE_ACCEPT_ENABLE", True):
            return 1.0
        win_sec = max(1.0, self._cfg("YAW_AUTH_MAG_SCORE_ACCEPT_WINDOW_SEC", 8.0))
        ref_rate = self._clamp(self._cfg("YAW_AUTH_MAG_SCORE_ACCEPT_REF_RATE", 0.20), 0.01, 1.0)
        min_samples = max(1, int(round(self._cfg("YAW_AUTH_MAG_SCORE_MIN_SAMPLES", 20.0))))
        floor = self._clamp(self._cfg("YAW_AUTH_MAG_SCORE_FLOOR", 0.15), 0.0, 1.0)
        cold_start_scale = self._clamp(self._cfg("YAW_AUTH_MAG_SCORE_COLD_START_SCALE", 1.0), 0.0, 1.0)
        mag_rate, mag_samples = self._source_accept_rate("MAG", t_now=float(t_now), window_sec=win_sec)
        if mag_samples < min_samples or not np.isfinite(mag_rate):
            return float(cold_start_scale)
        ratio = float(mag_rate) / float(ref_rate)
        return float(self._clamp(max(floor, ratio), floor, 1.0))

    def _source_freshness(self, source: str, t_now: float) -> float:
        """
        Freshness factor in [0, 1] from recent activity.

        Prefer last-applied timestamp; fallback to last-seen timestamp.
        This suppresses stale owner lock where a source keeps ownership despite
        no recent usable corrections.
        """
        src = str(source).upper()
        stale_sec = max(
            0.2,
            self._cfg_source(
                "YAW_AUTH_SOURCE_STALE_SEC",
                src,
                self._cfg("YAW_AUTH_OWNER_DEAD_TIMEOUT_SEC", 2.5),
            ),
        )
        tau_sec = max(
            0.2,
            self._cfg_source("YAW_AUTH_SOURCE_STALE_TAU_SEC", src, stale_sec),
        )
        last_applied_t = float(self._source_last_applied_t.get(src, -1e9))
        last_seen_t = float(self._source_last_seen_t.get(src, -1e9))
        # Use the most recent activity signal instead of preferring "applied" forever.
        # Otherwise one old apply can make a currently-active source look stale.
        last_activity_t = max(last_applied_t, last_seen_t)
        if not np.isfinite(last_activity_t) or last_activity_t < -1e8:
            return 0.0
        age = max(0.0, float(t_now) - float(last_activity_t))
        if age <= stale_sec:
            return 1.0
        return float(np.exp(-(age - stale_sec) / tau_sec))

    def _owner_dead_timeout_sec(self, source: str) -> float:
        """Owner-dead timeout with per-source override support."""
        return max(
            0.2,
            self._cfg_source("YAW_AUTH_OWNER_DEAD_TIMEOUT_SEC", str(source).upper(), 2.5),
        )

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
        if self._is_source_blocked(src, t_now=float(t_now)):
            return ""
        if owner == "HOLD" or src == owner:
            return ""

        min_owner_dwell = self._owner_min_dwell_sec()
        if (float(t_now) - float(self._owner_switch_t)) < min_owner_dwell:
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
            self._set_owner(src, t_now)
            return "stage12_claim_owner_stale"

        if src not in claim_sources:
            return ""
        if owner == "MAG" and src_eligible and (src_score >= (owner_score + claim_margin)):
            self._set_owner(src, t_now)
            return "stage12_claim_from_mag"
        if (
            owner == "MAG"
            and src == "LOOP"
            and src_score >= loop_force_claim_min_score
            and (not np.isfinite(speed_m_s) or float(speed_m_s) <= loop_force_claim_max_speed)
        ):
            self._set_owner(src, t_now)
            return "stage12_claim_loop_force"
        return ""

    def _select_best_source(self, candidates: Iterable[str], t_now: float) -> Tuple[str, float]:
        min_score = self._clamp(self._cfg("YAW_AUTH_MIN_SOURCE_SCORE", 0.35), 0.0, 1.0)
        min_effective_score = self._clamp(self._cfg("YAW_AUTH_MIN_EFFECTIVE_SCORE", 0.06), 0.0, 1.0)
        act_enable = self._cfg_bool("YAW_AUTH_ACTIVITY_ENABLE", True)
        act_min_samples = max(1, int(round(self._cfg("YAW_AUTH_ACTIVITY_MIN_SAMPLES", 8.0))))
        act_floor = self._clamp(self._cfg("YAW_AUTH_ACTIVITY_RATIO_FLOOR", 0.25), 0.0, 1.0)
        best_src = "HOLD"
        best_val = -1.0
        for src in candidates:
            s = str(src).upper()
            if self._is_source_blocked(s, t_now=float(t_now)):
                continue
            if s == "MAG" and not self._mag_owner_eligible(float(t_now)):
                continue
            if not self._owner_apply_recent_ok(s, t_now=float(t_now)):
                continue
            score = self._source_score_ema.get(s, np.nan)
            if not np.isfinite(score):
                continue
            if s == "MAG":
                score = float(score) * self._mag_score_scale(float(t_now))
            min_score_s = self._source_min_score(s, min_score)
            if float(score) < min_score_s:
                continue
            freshness = self._source_freshness(s, t_now=float(t_now))
            if act_enable:
                eff, req_cnt = self._source_apply_efficiency(s, t_now=float(t_now))
                if req_cnt >= act_min_samples:
                    score = float(score) * max(float(act_floor), float(eff))
            val = float(score) * float(freshness)
            if val < min_effective_score:
                continue
            # Very small tie-breaker only; do not let priority dominate stale quality.
            val += 1e-6 * self._source_priority(s)
            if val > best_val:
                best_val = val
                best_src = s
        return best_src, (best_val if best_src != "HOLD" else -1.0)

    def _switch_owner_if_needed(self, t_now: float) -> None:
        stage = self._stage()
        if stage < 3:
            return
        eval_min_dt = max(0.0, self._cfg("YAW_AUTH_SWITCH_EVAL_MIN_INTERVAL_SEC", 0.25))
        if (float(t_now) - float(self._last_switch_eval_t)) < eval_min_dt:
            return
        self._last_switch_eval_t = float(t_now)

        # Strong MAG demotion path: if rolling accept-rate is poor, do not keep MAG owner.
        owner = str(self._owner).upper()
        if owner == "MAG":
            win_sec = max(1.0, self._cfg("YAW_AUTH_MAG_OWNER_ACCEPT_WINDOW_SEC", 8.0))
            min_samples = max(1, int(round(self._cfg("YAW_AUTH_MAG_OWNER_MIN_SAMPLES", 20.0))))
            min_rate = self._clamp(self._cfg("YAW_AUTH_MAG_OWNER_MIN_ACCEPT_RATE", 0.15), 0.0, 1.0)
            mag_rate, mag_samples = self._source_accept_rate("MAG", t_now=float(t_now), window_sec=win_sec)
            if mag_samples >= min_samples and np.isfinite(mag_rate) and float(mag_rate) < float(min_rate):
                non_mag = [s for s in self._YAW_SOURCES if str(s).upper() != "MAG"]
                alt_src, _ = self._select_best_source(non_mag, t_now=float(t_now))
                self._set_owner(str(alt_src) if alt_src != "HOLD" else "HOLD", float(t_now))
                return

        candidates = list(self._YAW_SOURCES)
        best_src, _ = self._select_best_source(candidates, t_now=float(t_now))
        if best_src == "HOLD":
            if self._owner == "HOLD":
                escape_min = self._clamp(self._cfg("YAW_AUTH_HOLD_ESCAPE_MIN_SCORE", 0.16), 0.0, 1.0)
                escape_min_dt = max(0.0, self._cfg("YAW_AUTH_HOLD_ESCAPE_MIN_SEC", 0.35))
                if (float(t_now) - float(self._owner_switch_t)) >= escape_min_dt:
                    alt_src = "HOLD"
                    alt_val = -1.0
                    for s in self._YAW_SOURCES:
                        if self._is_source_blocked(s, t_now=float(t_now)):
                            continue
                        score = self._source_score_ema.get(s, np.nan)
                        if not np.isfinite(score) or float(score) < escape_min:
                            continue
                        freshness = self._source_freshness(s, t_now=float(t_now))
                        val = float(score) * float(freshness)
                        if val > alt_val:
                            alt_val = val
                            alt_src = s
                    if alt_src != "HOLD":
                        self._set_owner(str(alt_src), t_now)
                        return
            hold_score_th = self._clamp(self._cfg("YAW_AUTH_HOLD_SCORE_THRESHOLD", 0.18), 0.0, 1.0)
            cur_score = float(self._source_score_ema.get(self._owner, 0.0))
            if self._owner != "HOLD" and cur_score < hold_score_th:
                self._set_owner("HOLD", t_now)
            return

        if self._owner == "HOLD":
            self._set_owner(str(best_src), t_now)
            return

        if str(best_src) == str(self._owner):
            return

        switch_margin = self._clamp(self._cfg("YAW_AUTH_SWITCH_MARGIN", 0.12), 0.0, 1.0)
        switch_min_dt = max(0.0, self._cfg("YAW_AUTH_SWITCH_MIN_INTERVAL_SEC", 0.75))
        owner_min_dwell = self._owner_min_dwell_sec()
        cur_score = float(self._source_score_ema.get(str(self._owner).upper(), 0.0))
        best_score = float(self._source_score_ema.get(str(best_src).upper(), 0.0))
        if (float(t_now) - float(self._owner_switch_t)) < switch_min_dt:
            return
        if (float(t_now) - float(self._owner_switch_t)) < owner_min_dwell:
            return
        if best_score >= (cur_score + switch_margin):
            self._set_owner(str(best_src), t_now)

    def _try_loop_high_quality_reclaim(
        self,
        t_now: float,
        loop_conf_raw: float,
        loop_conf_ema: float,
        speed_m_s: float,
        current_dead_reason: str = "",
    ) -> str:
        """
        Deterministic reclaim path: LOOP may reclaim owner from BACKEND when
        LOOP quality is consistently high. This avoids backend-only yaw lock.
        """
        if self._stage() < 3:
            return ""
        if not self._cfg_bool("YAW_AUTH_LOOP_RECLAIM_ENABLE", False):
            return ""
        owner = str(self._owner).upper()
        allow_from_hold = self._cfg_bool("YAW_AUTH_LOOP_RECLAIM_ALLOW_FROM_HOLD", False)
        hold_require_backend_dead = self._cfg_bool(
            "YAW_AUTH_LOOP_RECLAIM_HOLD_REQUIRE_BACKEND_DEAD", True
        )
        hold_allow_no_dead = self._cfg_bool(
            "YAW_AUTH_LOOP_RECLAIM_HOLD_ALLOW_NO_DEAD", False
        )
        backend_dead_ctx = "owner_dead_timeout_backend" in str(current_dead_reason).lower()
        if owner not in ("BACKEND", "HOLD"):
            return ""
        if owner == "HOLD":
            if not allow_from_hold:
                return ""
            if hold_require_backend_dead and not backend_dead_ctx and not hold_allow_no_dead:
                return ""
            hold_min_sec = max(0.0, self._cfg("YAW_AUTH_LOOP_RECLAIM_HOLD_MIN_SEC", 0.35))
            bypass_min_dt_on_backend_dead = self._cfg_bool(
                "YAW_AUTH_LOOP_RECLAIM_BYPASS_MIN_INTERVAL_ON_BACKEND_DEAD",
                True,
            )
            if not (backend_dead_ctx and bypass_min_dt_on_backend_dead):
                if (float(t_now) - float(self._owner_switch_t)) < hold_min_sec:
                    return ""
            hold_cd_sec = max(0.0, self._cfg("YAW_AUTH_LOOP_RECLAIM_HOLD_COOLDOWN_SEC", 1.5))
            if (float(t_now) - float(self._last_loop_hold_reclaim_t)) < hold_cd_sec:
                return ""
        if owner == "BACKEND":
            # BACKEND-owner reclaim should remain deterministic and quality-led.
            pass
        if self._is_source_blocked("LOOP", t_now=float(t_now)):
            return ""

        min_dt = max(0.0, self._cfg("YAW_AUTH_LOOP_RECLAIM_MIN_INTERVAL_SEC", self._cfg("YAW_AUTH_SWITCH_MIN_INTERVAL_SEC", 0.75)))
        bypass_min_dt_on_backend_dead = self._cfg_bool(
            "YAW_AUTH_LOOP_RECLAIM_BYPASS_MIN_INTERVAL_ON_BACKEND_DEAD",
            True,
        )
        if (
            (float(t_now) - float(self._owner_switch_t)) < min_dt
            and not (owner == "HOLD" and backend_dead_ctx and bypass_min_dt_on_backend_dead)
        ):
            return ""

        win_sec = max(0.5, self._cfg("YAW_AUTH_LOOP_RECLAIM_WINDOW_SEC", 6.0))
        min_req = max(1, int(round(self._cfg("YAW_AUTH_LOOP_RECLAIM_MIN_REQUESTS", 3.0))))
        if owner == "HOLD" and backend_dead_ctx:
            min_req = max(
                1,
                int(
                    round(
                        self._cfg(
                            "YAW_AUTH_LOOP_RECLAIM_MIN_REQUESTS_ON_BACKEND_DEAD",
                            1.0,
                        )
                    )
                ),
            )
        req_cnt = self._count_recent_events(
            self._source_request_hist,
            source="LOOP",
            t_now=float(t_now),
            window_sec=win_sec,
        )
        if req_cnt < min_req:
            return ""

        loop_score = max(
            self._clamp(float(loop_conf_raw), 0.0, 1.0),
            self._clamp(float(loop_conf_ema), 0.0, 1.0),
            self._clamp(float(self._source_score_ema.get("LOOP", loop_conf_ema)), 0.0, 1.0),
        )
        back_score = self._clamp(float(self._source_score_ema.get("BACKEND", 0.0)), 0.0, 1.0)
        min_score = self._clamp(self._cfg("YAW_AUTH_LOOP_RECLAIM_MIN_SCORE", 0.62), 0.0, 1.0)
        margin = self._clamp(self._cfg("YAW_AUTH_LOOP_RECLAIM_MARGIN", 0.05), 0.0, 0.8)
        if owner == "HOLD" and backend_dead_ctx:
            min_score = self._clamp(
                self._cfg("YAW_AUTH_LOOP_RECLAIM_MIN_SCORE_ON_BACKEND_DEAD", min_score),
                0.0,
                1.0,
            )
            margin = self._clamp(
                self._cfg("YAW_AUTH_LOOP_RECLAIM_MARGIN_ON_BACKEND_DEAD", margin),
                0.0,
                0.8,
            )
        min_fresh = self._clamp(self._cfg("YAW_AUTH_LOOP_RECLAIM_MIN_FRESHNESS", 0.10), 0.0, 1.0)
        loop_fresh = self._source_freshness("LOOP", t_now=float(t_now))
        if loop_score < min_score or loop_fresh < min_fresh:
            return ""
        if loop_score < (back_score + margin):
            return ""

        max_speed = max(0.0, self._cfg("YAW_AUTH_LOOP_RECLAIM_MAX_SPEED_M_S", 120.0))
        if np.isfinite(speed_m_s) and float(speed_m_s) > float(max_speed):
            return ""
        min_eff_samples = max(0, int(round(self._cfg("YAW_AUTH_LOOP_RECLAIM_MIN_APPLY_SAMPLES", 0.0))))
        min_eff = self._clamp(self._cfg("YAW_AUTH_LOOP_RECLAIM_MIN_APPLY_EFF", 0.0), 0.0, 1.0)
        eff, req_cnt_eff = self._source_apply_efficiency("LOOP", t_now=float(t_now))
        if min_eff_samples > 0 and req_cnt_eff >= min_eff_samples and float(eff) < float(min_eff):
            return ""

        self._set_owner("LOOP", float(t_now))
        if owner == "HOLD":
            self._last_loop_hold_reclaim_t = float(t_now)
            return "loop_reclaim_high_quality_from_hold"
        return "loop_reclaim_high_quality"

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
        switched = False
        prev_score_ema = float(self._source_score_ema.get(src, np.nan))
        req_abs = max(0.0, self._finite_or(requested_abs_dyaw_deg, 0.0))
        backend_min_req = max(0.0, self._cfg("YAW_AUTH_BACKEND_MIN_REQUEST_DYAW_DEG", 0.12))
        backend_noop_req = bool(src == "BACKEND" and req_abs < backend_min_req)
        conf_raw = self._clamp(self._finite_or(confidence, 0.5), 0.0, 1.0)
        conf_in = self._modulate_confidence_by_msckf(src, conf_raw)
        if backend_noop_req:
            conf_in = min(conf_in, self._clamp(self._cfg("YAW_AUTH_BACKEND_NOOP_MAX_SCORE", 0.18), 0.0, 1.0))
        if src == "MAG":
            conf_in = float(conf_in) * self._mag_score_scale(float(t_now))
        conf = self._update_score_ema(src, conf_in)
        if not backend_noop_req:
            self._source_last_seen_t[src] = float(t_now)
        self._decision_samples += 1
        if not backend_noop_req:
            self._append_ts_event(
                self._source_request_hist,
                source=src,
                timestamp=float(t_now),
                max_keep=max(64, int(round(self._cfg("YAW_AUTH_ACTIVITY_WINDOW_SEC", 8.0) * 80.0))),
            )
        self._update_source_blocks(t_now=float(t_now))
        mag_blocked = self._is_source_blocked("MAG", t_now=float(t_now))
        if mag_blocked:
            self._mag_owner_block_samples += 1

        if self._enabled() and src == "MAG":
            if not self._mag_owner_eligible(float(t_now)):
                # Keep ineligible MAG usable as a weak anchor, but do not let its
                # soft-path decisions build enough score to seize yaw ownership.
                owner_score_cap = self._clamp(
                    min(
                        self._cfg("YAW_AUTH_HOLD_SCORE_THRESHOLD", 0.18),
                        max(0.05, self._cfg("YAW_AUTH_MIN_SOURCE_SCORE", 0.35) - 0.02),
                    ),
                    0.0,
                    1.0,
                )
                capped_score = min(float(conf), float(owner_score_cap))
                if np.isfinite(prev_score_ema):
                    capped_score = min(capped_score, float(prev_score_ema), float(owner_score_cap))
                self._source_score_ema["MAG"] = float(capped_score)
                if str(self._owner).upper() == "MAG":
                    self._set_owner("HOLD", float(t_now))
                soft_dec = self._build_mag_ineligible_soft_decision(
                    source=src,
                    t_now=float(t_now),
                    conf_raw=float(conf_raw),
                    conf_ema=float(conf),
                    reason_prefix="mag_owner_ineligible",
                )
                if soft_dec is not None:
                    return soft_dec
                return YawAuthorityDecision(
                    source=src,
                    owner=str(self._owner).upper(),
                    allow=False,
                    mode="HOLD",
                    reason="mag_owner_ineligible",
                    confidence=conf,
                )

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
        strict_single_path = bool(self._stage() >= 3 and self._cfg_bool("YAW_AUTH_STRICT_SINGLE_PATH_ENABLE", True))
        # Stage-1 lazy owner (single owner, no score switching).
        if self._is_source_blocked(str(self._owner).upper(), t_now=float(t_now)):
            self._set_owner("HOLD", t_now=float(t_now))

        # During HOLD cooldown, keep arbitration stable and avoid owner switch churn.
        if float(t_now) < float(self._hold_until_t):
            if self._try_hold_interrupt_source(
                src,
                t_now=float(t_now),
                conf_raw=float(conf_raw),
                conf_ema=float(conf),
            ):
                switched = True
            else:
                return YawAuthorityDecision(
                    source=src,
                    owner="HOLD",
                    allow=False,
                    mode="HOLD",
                    reason="hold_active",
                    confidence=conf,
                )

        if self._owner == "HOLD" and self._stage() < 3 and not self._is_source_blocked(src, t_now=float(t_now)):
            switched = self._set_owner(src, t_now) or switched
        elif self._stage() < 3:
            claim_reason = self._maybe_handoff_stage12(
                source=src,
                t_now=float(t_now),
                confidence=conf,
                speed_m_s=float(speed_m_s),
            )
            if claim_reason != "":
                switched = True

        # Stage-3: confidence/hysteresis switching.
        prev_owner = str(self._owner).upper()
        self._switch_owner_if_needed(float(t_now))
        if str(self._owner).upper() != prev_owner:
            switched = True

        if src == "LOOP":
            reclaim_reason = self._try_loop_high_quality_reclaim(
                t_now=float(t_now),
                loop_conf_raw=float(conf_raw),
                loop_conf_ema=float(conf),
                speed_m_s=float(speed_m_s),
            )
            if reclaim_reason:
                claim_reason = f"{claim_reason}|{reclaim_reason}" if claim_reason else reclaim_reason
                switched = True

        dead_reason = self._apply_dead_owner_fallback(float(t_now))
        if dead_reason:
            switched = True
            if src == "LOOP":
                reclaim_reason = self._try_loop_high_quality_reclaim(
                    t_now=float(t_now),
                    loop_conf_raw=float(conf_raw),
                    loop_conf_ema=float(conf),
                    speed_m_s=float(speed_m_s),
                    current_dead_reason=str(dead_reason),
                )
                if reclaim_reason:
                    switched = True
                    self._hold_until_t = min(float(self._hold_until_t), float(t_now))
                    claim_reason = (
                        f"{claim_reason}|{reclaim_reason}" if claim_reason else reclaim_reason
                    )
                    dead_reason = f"{dead_reason}|{reclaim_reason}"
            # Avoid sticky HOLD after dead-owner fallback: allow the currently requesting
            # source to reclaim ownership immediately when eligible.
            if (
                not strict_single_path
                and
                self._stage() >= 3
                and str(self._owner).upper() == "HOLD"
                and not self._is_source_blocked(src, t_now=float(t_now))
            ):
                reclaim_min_score = self._clamp(
                    self._cfg(
                        "YAW_AUTH_OWNER_DEAD_RECLAIM_MIN_SCORE",
                        max(self._cfg("YAW_AUTH_MIN_SOURCE_SCORE", 0.35), 0.45),
                    ),
                    0.0,
                    1.0,
                )
                reclaim_min_fresh = self._clamp(
                    self._cfg("YAW_AUTH_OWNER_DEAD_RECLAIM_MIN_FRESHNESS", 0.35),
                    0.0,
                    1.0,
                )
                src_score = self._clamp(self._source_score_ema.get(src, conf), 0.0, 1.0)
                src_fresh = self._source_freshness(src, t_now=float(t_now))
                if src_score >= reclaim_min_score and src_fresh >= reclaim_min_fresh:
                    switched = self._set_owner(src, float(t_now)) or switched
                    self._hold_until_t = min(float(self._hold_until_t), float(t_now))
                    dead_reason = f"{dead_reason}|immediate_reclaim_{str(src).lower()}"
                else:
                    dead_reason = f"{dead_reason}|reclaim_deferred_{str(src).lower()}"

        if float(t_now) < float(self._hold_until_t):
            if strict_single_path:
                if self._strict_hold_reclaim_allowed(src, t_now=float(t_now), conf_raw=float(conf_raw), conf_ema=float(conf)):
                    self._set_owner(src, float(t_now))
                    self._hold_until_t = min(float(self._hold_until_t), float(t_now))
                else:
                    anchor_dec = self._build_hold_mag_anchor_decision(
                        source=src,
                        t_now=float(t_now),
                        conf_raw=float(conf_raw),
                        conf_ema=float(conf),
                        reason_prefix=dead_reason if dead_reason else "hold_active",
                    )
                    if anchor_dec is not None:
                        return anchor_dec
                    return YawAuthorityDecision(
                        source=src,
                        owner="HOLD",
                        allow=False,
                        mode="HOLD",
                        reason=dead_reason if dead_reason else "hold_active",
                        confidence=conf,
                    )
            if self._stage() >= 3 and not strict_single_path:
                return YawAuthorityDecision(
                    source=src,
                    owner="HOLD",
                    allow=False,
                    mode="HOLD",
                    reason=dead_reason if dead_reason else "hold_active",
                    confidence=conf,
                )
            if self._stage() >= 3:
                bypass_sources_raw = self.runner.global_config.get(
                    "YAW_AUTH_HOLD_BYPASS_SOURCES", ["LOOP", "BACKEND"]
                ) if isinstance(self.runner.global_config, dict) else ["LOOP", "BACKEND"]
                bypass_sources = {
                    str(v).upper() for v in bypass_sources_raw
                    if str(v).strip() != ""
                }
                bypass_min_conf_default = self._clamp(
                    self._cfg("YAW_AUTH_HOLD_BYPASS_MIN_CONFIDENCE", 0.55), 0.0, 1.0
                )
                bypass_min_fresh = self._clamp(
                    self._cfg("YAW_AUTH_HOLD_BYPASS_MIN_FRESHNESS", 0.20), 0.0, 1.0
                )
                bypass_min_conf_map_raw = self.runner.global_config.get(
                    "YAW_AUTH_HOLD_BYPASS_MIN_CONFIDENCE_MAP", {}
                ) if isinstance(self.runner.global_config, dict) else {}
                bypass_min_conf_map = (
                    bypass_min_conf_map_raw if isinstance(bypass_min_conf_map_raw, dict) else {}
                )
                bypass_min_conf = bypass_min_conf_default
                if src in bypass_min_conf_map:
                    try:
                        bypass_min_conf = self._clamp(float(bypass_min_conf_map.get(src)), 0.0, 1.0)
                    except Exception:
                        bypass_min_conf = bypass_min_conf_default
                src_freshness = self._source_freshness(src, t_now=float(t_now))
                if (
                    src in bypass_sources
                    and max(float(conf_raw), float(conf)) >= bypass_min_conf
                    and src_freshness >= bypass_min_fresh
                    and not self._is_source_blocked(src, t_now=float(t_now))
                ):
                    self._set_owner(src, float(t_now))
                    self._hold_until_t = min(float(self._hold_until_t), float(t_now))
                else:
                    return YawAuthorityDecision(
                        source=src,
                        owner="HOLD",
                        allow=False,
                        mode="HOLD",
                        reason=dead_reason if dead_reason else "hold_active",
                        confidence=conf,
                    )
            else:
                return YawAuthorityDecision(
                    source=src,
                    owner="HOLD",
                    allow=False,
                    mode="HOLD",
                    reason=dead_reason if dead_reason else "hold_active",
                    confidence=conf,
                )

        owner = str(self._owner).upper()
        if owner != "HOLD" and not self._owner_apply_recent_ok(owner, t_now=float(t_now)):
            self._set_owner("HOLD", float(t_now))
            owner = "HOLD"
        self._owner_samples += 1
        self._owner_source_counts[owner] += 1
        trace_reason_parts = []
        if str(claim_reason).strip() != "":
            trace_reason_parts.append(str(claim_reason))
        if str(dead_reason).strip() != "":
            trace_reason_parts.append(str(dead_reason))
        if not trace_reason_parts:
            trace_reason_parts.append("owner_eval")
        log_heading_owner_trace(
            getattr(self.runner, "heading_owner_trace_csv", None),
            t=float(t_now),
            owner=owner,
            source=src,
            mode="HOLD" if owner == "HOLD" else "EVAL",
            reason="|".join(trace_reason_parts),
            score_mag=float(self._source_score_ema.get("MAG", np.nan)),
            score_loop=float(self._source_score_ema.get("LOOP", np.nan)),
            score_backend=float(self._source_score_ema.get("BACKEND", np.nan)),
            score_msckf=float(self._source_score_ema.get("MSCKF", np.nan)),
            switched=bool(switched),
        )

        if owner == "HOLD":
            strict_hold_reclaimed = False
            if strict_single_path:
                if self._strict_hold_reclaim_allowed(src, t_now=float(t_now), conf_raw=float(conf_raw), conf_ema=float(conf)):
                    try:
                        self._owner_source_counts["HOLD"] = max(
                            0, int(self._owner_source_counts.get("HOLD", 0)) - 1
                        )
                    except Exception:
                        pass
                    self._set_owner(src, float(t_now))
                    owner = str(self._owner).upper()
                    self._owner_source_counts[owner] += 1
                    strict_hold_reclaimed = True
                else:
                    anchor_dec = self._build_hold_mag_anchor_decision(
                        source=src,
                        t_now=float(t_now),
                        conf_raw=float(conf_raw),
                        conf_ema=float(conf),
                        reason_prefix="owner_hold",
                    )
                    if anchor_dec is not None:
                        return anchor_dec
                    return YawAuthorityDecision(
                        source=src,
                        owner=owner,
                        allow=False,
                        mode="HOLD",
                        reason="owner_hold",
                        confidence=conf,
                    )
            if self._stage() >= 3 and (not strict_single_path or not strict_hold_reclaimed):
                reclaim_sources_raw = self.runner.global_config.get(
                    "YAW_AUTH_HOLD_RECLAIM_SOURCES", ["LOOP", "BACKEND"]
                ) if isinstance(self.runner.global_config, dict) else ["LOOP", "BACKEND"]
                reclaim_sources = {
                    str(v).upper() for v in reclaim_sources_raw
                    if str(v).strip() != ""
                }
                reclaim_min_conf = self._clamp(
                    self._cfg("YAW_AUTH_HOLD_RECLAIM_MIN_CONFIDENCE", 0.20), 0.0, 1.0
                )
                reclaim_conf = max(float(conf_raw), float(conf))
                if (
                    src in reclaim_sources
                    and reclaim_conf >= reclaim_min_conf
                    and not self._is_source_blocked(src, t_now=float(t_now))
                ):
                    # Reassign this sample from HOLD to reclaimed owner for runtime ratios.
                    try:
                        self._owner_source_counts["HOLD"] = max(
                            0, int(self._owner_source_counts.get("HOLD", 0)) - 1
                        )
                    except Exception:
                        pass
                    self._set_owner(src, float(t_now))
                    owner = str(self._owner).upper()
                    self._owner_source_counts[owner] += 1
                else:
                    anchor_dec = self._build_hold_mag_anchor_decision(
                        source=src,
                        t_now=float(t_now),
                        conf_raw=float(conf_raw),
                        conf_ema=float(conf),
                        reason_prefix="owner_hold",
                    )
                    if anchor_dec is not None:
                        return anchor_dec
                    return YawAuthorityDecision(
                        source=src,
                        owner=owner,
                        allow=False,
                        mode="HOLD",
                        reason="owner_hold",
                        confidence=conf,
                    )
            elif self._stage() < 3:
                return YawAuthorityDecision(
                    source=src,
                    owner=owner,
                    allow=False,
                    mode="HOLD",
                    reason="owner_hold",
                    confidence=conf,
                )

        if src != owner:
            if src == "MAG":
                spring_dec = self._build_mag_soft_spring_decision(
                    source=src,
                    t_now=float(t_now),
                    conf_raw=float(conf_raw),
                    conf_ema=float(conf),
                    reason_prefix=f"owner_is_{owner.lower()}",
                    owner_hint=owner,
                )
                if spring_dec is not None:
                    return spring_dec
            return YawAuthorityDecision(
                source=src,
                owner=owner,
                allow=False,
                mode="HOLD",
                reason=f"owner_is_{owner.lower()}",
                confidence=conf,
            )
        if self._is_source_blocked(owner, t_now=float(t_now)):
            if src == "MAG":
                spring_dec = self._build_mag_soft_spring_decision(
                    source=src,
                    t_now=float(t_now),
                    conf_raw=float(conf_raw),
                    conf_ema=float(conf),
                    reason_prefix=f"owner_blocked_{owner.lower()}",
                    owner_hint=owner,
                    blocked_hint=True,
                )
                if spring_dec is not None:
                    return spring_dec
            return YawAuthorityDecision(
                source=src,
                owner="HOLD",
                allow=False,
                mode="HOLD",
                reason=f"owner_blocked_{owner.lower()}",
                confidence=conf,
            )

        mode = "APPLY"
        reason_codes = [claim_reason] if claim_reason else []
        r_mult = 1.0
        max_update = float("inf")

        # When MAG rolling accept-rate is low, keep it as weak anchor:
        # allow only soft / tiny corrections rather than hard ownership lockout.
        if src == "MAG":
            low_win_sec = max(1.0, self._cfg("YAW_AUTH_MAG_OWNER_ACCEPT_WINDOW_SEC", 8.0))
            low_min_samples = max(1, int(round(self._cfg("YAW_AUTH_MAG_OWNER_MIN_SAMPLES", 20.0))))
            low_min_rate = self._clamp(self._cfg("YAW_AUTH_MAG_OWNER_MIN_ACCEPT_RATE", 0.15), 0.0, 1.0)
            low_rate, low_samples = self._source_accept_rate("MAG", t_now=float(t_now), window_sec=low_win_sec)
            if low_samples >= low_min_samples and np.isfinite(low_rate) and float(low_rate) < float(low_min_rate):
                spring_dec = self._build_mag_soft_spring_decision(
                    source=src,
                    t_now=float(t_now),
                    conf_raw=float(conf_raw),
                    conf_ema=float(conf),
                    reason_prefix="mag_low_accept_soft_only",
                    owner_hint=owner,
                )
                if spring_dec is not None:
                    return spring_dec
                mode = "SOFT_APPLY"
                reason_codes.append("mag_low_accept_soft_only")
                r_mult *= max(1.0, self._cfg("YAW_AUTH_MAG_LOW_ACCEPT_SOFT_R_MULT", 3.0))
                max_update = min(
                    max_update,
                    max(0.05, self._cfg("YAW_AUTH_MAG_LOW_ACCEPT_MAX_DYAW_DEG", 0.35)),
                )

        # Stage-2: cumulative/rate budget.
        if self._stage() >= 2 and req_abs > 1e-6:
            win_sec = max(0.25, self._cfg("YAW_AUTH_YAW_BUDGET_WINDOW_SEC", 8.0))
            max_cum = max(0.2, self._cfg("YAW_AUTH_YAW_BUDGET_ABS_DEG", 10.0))
            max_rate = max(0.2, self._cfg("YAW_AUTH_YAW_RATE_MAX_DEG_S", 4.0))
            per_source_budget = self._cfg_bool("YAW_AUTH_BUDGET_PER_SOURCE_ENABLE", True)
            budget_source = src if per_source_budget else None
            used_cum = self._window_sum(t_now=t_now, window_sec=win_sec, source=budget_source)
            used_rate = self._window_sum(t_now=t_now, window_sec=1.0, source=budget_source)
            rem_cum = max(0.0, float(max_cum) - float(used_cum))
            rem_rate = max(0.0, float(max_rate) - float(used_rate))
            if per_source_budget:
                # Keep a looser global cap for total system stability while preventing
                # one source (often MAG at high cadence) from starving LOOP/BACKEND.
                global_mult = max(1.0, self._cfg("YAW_AUTH_BUDGET_GLOBAL_MULT", 2.0))
                global_used_cum = self._window_sum(t_now=t_now, window_sec=win_sec, source=None)
                global_used_rate = self._window_sum(t_now=t_now, window_sec=1.0, source=None)
                rem_cum = min(
                    rem_cum,
                    max(0.0, float(max_cum) * float(global_mult) - float(global_used_cum)),
                )
                rem_rate = min(
                    rem_rate,
                    max(0.0, float(max_rate) * float(global_mult) - float(global_used_rate)),
                )
            max_update = min(max_update, rem_cum, rem_rate)
            if max_update <= 0.05:
                if src != "MAG":
                    hold_sec = max(0.1, self._cfg("YAW_AUTH_HOLD_SEC", 0.8))
                    self._hold_until_t = max(float(self._hold_until_t), float(t_now + hold_sec))
                elif src == "MAG":
                    spring_dec = self._build_mag_soft_spring_decision(
                        source=src,
                        t_now=float(t_now),
                        conf_raw=float(conf_raw),
                        conf_ema=float(conf),
                        reason_prefix="budget_exhausted_mag",
                        owner_hint=owner,
                    )
                    if spring_dec is not None:
                        return spring_dec
                return YawAuthorityDecision(
                    source=src,
                    owner=owner,
                    allow=False,
                    mode="HOLD",
                    reason="budget_exhausted_mag" if src == "MAG" else "budget_exhausted",
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
        self._source_last_applied_t[src] = float(timestamp)
        self._append_ts_event(
            self._source_apply_hist,
            source=src,
            timestamp=float(timestamp),
            max_keep=max(64, int(round(self._cfg("YAW_AUTH_ACTIVITY_WINDOW_SEC", 8.0) * 80.0))),
        )
        self._yaw_hist.append((float(timestamp), float(abs(val)), src))
        self._prune_hist(float(timestamp), max(0.25, self._cfg("YAW_AUTH_YAW_BUDGET_WINDOW_SEC", 8.0)))

    def register_source_observation(self, source: str, timestamp: float, accepted: bool) -> None:
        """Register source-level accept/reject outcome for owner eligibility gating."""
        src = str(source).upper()
        if src not in self._YAW_SOURCES:
            return
        hist = self._source_accept_hist[src]
        hist.append((float(timestamp), 1 if bool(accepted) else 0))
        max_keep = max(32, int(round(self._cfg("YAW_AUTH_MAG_OWNER_MIN_SAMPLES", 20.0) * 20.0)))
        while len(hist) > max_keep:
            hist.popleft()

    def get_runtime_metrics(self) -> Dict[str, float]:
        """Return yaw-owner runtime summary metrics for health reporting."""
        total = float(max(1, self._owner_samples))
        mag = float(self._owner_source_counts.get("MAG", 0))
        loop = float(self._owner_source_counts.get("LOOP", 0))
        backend = float(self._owner_source_counts.get("BACKEND", 0))
        hold = float(self._owner_source_counts.get("HOLD", 0))
        return {
            "owner_switch_count": float(self._owner_switch_count),
            "owner_mag_ratio": float(mag / total),
            "owner_loop_ratio": float(loop / total),
            "owner_backend_ratio": float(backend / total),
            "owner_hold_ratio": float(hold / total),
            "mag_owner_block_count": float(self._source_owner_block_count.get("MAG", 0)),
            "mag_owner_block_ratio": float(
                float(self._mag_owner_block_samples) / float(max(1, self._decision_samples))
            ),
            "owner_dead_fallback_count": float(self._owner_dead_fallback_count),
        }
