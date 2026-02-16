#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asynchronous fixed-lag backend correction queue.

This backend is intentionally lightweight for near-RT operation:
- frontend EKF never blocks
- backend produces bounded soft corrections from recent absolute hints
- caller polls and applies correction with blend/caps
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BackendCorrection:
    """Backend correction payload returned to frontend."""

    t_ref: float
    dp_enu: np.ndarray
    dyaw_deg: float
    cov_scale: float
    age_sec: float
    quality_score: float


@dataclass
class _Keyframe:
    t: float
    p_enu: np.ndarray
    yaw_deg: float
    quality: float


@dataclass
class _AbsHint:
    t: float
    dp_enu: np.ndarray
    dyaw_deg: float
    quality: float


class BackendOptimizer:
    """Background optimizer that emits soft absolute corrections."""

    def __init__(
        self,
        fixed_lag_window: int = 10,
        optimize_rate_hz: float = 2.0,
        max_iteration_ms: float = 35.0,
        max_correction_age_sec: float = 2.0,
        min_quality_score: float = 0.2,
        max_abs_dp_xy_m: float = 60.0,
        max_abs_dyaw_deg: float = 8.0,
    ):
        self.fixed_lag_window = max(3, int(fixed_lag_window))
        self.optimize_rate_hz = max(0.2, float(optimize_rate_hz))
        self.max_iteration_ms = max(1.0, float(max_iteration_ms))
        self.max_correction_age_sec = max(0.1, float(max_correction_age_sec))
        self.min_quality_score = float(np.clip(min_quality_score, 0.0, 1.0))
        self.max_abs_dp_xy_m = max(1.0, float(max_abs_dp_xy_m))
        self.max_abs_dyaw_deg = max(0.5, float(max_abs_dyaw_deg))

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._keyframes: deque[_Keyframe] = deque(maxlen=max(32, self.fixed_lag_window * 3))
        self._abs_hints: deque[_AbsHint] = deque(maxlen=max(64, self.fixed_lag_window * 6))
        self._pending_corrections: deque[BackendCorrection] = deque(maxlen=16)

        self.stats = {
            "iterations": 0,
            "corrections_emitted": 0,
            "last_iter_ms": 0.0,
        }
        self.last_poll_stale_drops = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=1.5)
        self._thread = None

    def push_keyframe(self, t_ref: float, p_enu: np.ndarray, yaw_deg: float, quality_score: float = 1.0) -> None:
        """Push one frontend keyframe snapshot for fixed-lag context."""
        p = np.asarray(p_enu, dtype=float).reshape(3,)
        if not np.all(np.isfinite(p)):
            return
        kf = _Keyframe(
            t=float(t_ref),
            p_enu=p.copy(),
            yaw_deg=float(yaw_deg) if np.isfinite(yaw_deg) else 0.0,
            quality=float(np.clip(quality_score, 0.0, 1.0)),
        )
        with self._lock:
            self._keyframes.append(kf)

    def report_absolute_hint(self, t_ref: float, dp_enu: np.ndarray, dyaw_deg: float = 0.0, quality_score: float = 0.5) -> None:
        """Report absolute correction hint (e.g., VPS, relocalization)."""
        dp = np.asarray(dp_enu, dtype=float).reshape(3,)
        if not np.all(np.isfinite(dp)):
            return
        hint = _AbsHint(
            t=float(t_ref),
            dp_enu=dp.copy(),
            dyaw_deg=float(dyaw_deg) if np.isfinite(dyaw_deg) else 0.0,
            quality=float(np.clip(quality_score, 0.0, 1.0)),
        )
        with self._lock:
            self._abs_hints.append(hint)

    def poll_correction(self, t_now: Optional[float] = None) -> Optional[BackendCorrection]:
        """Poll latest backend correction; drops stale corrections."""
        now = time.time() if t_now is None else float(t_now)
        dropped = 0
        with self._lock:
            while self._pending_corrections:
                corr = self._pending_corrections.popleft()
                age = max(0.0, now - float(corr.t_ref))
                corr.age_sec = age
                if age <= self.max_correction_age_sec:
                    self.last_poll_stale_drops = dropped
                    return corr
                dropped += 1
            self.last_poll_stale_drops = dropped
        return None

    def _worker_loop(self) -> None:
        period = 1.0 / max(self.optimize_rate_hz, 1e-6)
        while self._running:
            t0 = time.time()
            corr = self._compute_correction()
            iter_ms = (time.time() - t0) * 1000.0
            with self._lock:
                self.stats["iterations"] = int(self.stats.get("iterations", 0)) + 1
                self.stats["last_iter_ms"] = float(iter_ms)
                if corr is not None:
                    self._pending_corrections.append(corr)
                    self.stats["corrections_emitted"] = int(self.stats.get("corrections_emitted", 0)) + 1
            dt = time.time() - t0
            sleep_s = max(0.0, period - dt)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _compute_correction(self) -> Optional[BackendCorrection]:
        t_start = time.time()
        with self._lock:
            keyframes = list(self._keyframes)
            hints = list(self._abs_hints)

        if len(keyframes) < 2 or len(hints) == 0:
            return None

        kf_tail = keyframes[-self.fixed_lag_window :]
        t_min = float(kf_tail[0].t)
        t_max = float(kf_tail[-1].t)

        win_hints = [h for h in hints if (h.t >= t_min and h.t <= t_max)]
        if len(win_hints) == 0:
            return None

        w = np.array([max(1e-3, float(h.quality)) for h in win_hints], dtype=float)
        if not np.all(np.isfinite(w)):
            return None
        w_sum = float(np.sum(w))
        if w_sum <= 1e-9:
            return None
        w /= w_sum

        dp_stack = np.array([h.dp_enu for h in win_hints], dtype=float)
        dyaw_arr = np.array([h.dyaw_deg for h in win_hints], dtype=float)
        if not np.all(np.isfinite(dp_stack)):
            return None

        dp_est = np.sum(dp_stack * w.reshape(-1, 1), axis=0)
        dyaw_est = float(np.sum(dyaw_arr * w)) if np.all(np.isfinite(dyaw_arr)) else 0.0
        q_est = float(np.clip(np.sum(w * np.array([h.quality for h in win_hints], dtype=float)), 0.0, 1.0))

        if q_est < self.min_quality_score:
            return None

        dp_xy_norm = float(np.linalg.norm(dp_est[:2]))
        if dp_xy_norm > self.max_abs_dp_xy_m and dp_xy_norm > 1e-9:
            dp_est[:2] *= float(self.max_abs_dp_xy_m / dp_xy_norm)
        dyaw_est = float(np.clip(dyaw_est, -self.max_abs_dyaw_deg, self.max_abs_dyaw_deg))

        if float(np.linalg.norm(dp_est[:2])) < 0.5 and abs(dyaw_est) < 0.2:
            return None

        elapsed_ms = (time.time() - t_start) * 1000.0
        if elapsed_ms > self.max_iteration_ms:
            return None

        cov_scale = float(np.clip(1.0 + 0.6 * (1.0 - q_est), 1.0, 1.8))
        return BackendCorrection(
            t_ref=float(t_max),
            dp_enu=np.asarray(dp_est, dtype=float),
            dyaw_deg=float(dyaw_est),
            cov_scale=cov_scale,
            age_sec=0.0,
            quality_score=q_est,
        )
