#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asynchronous fixed-lag backend correction queue.

This backend is intentionally lightweight for near-RT operation:
- frontend EKF never blocks
- backend produces bounded soft corrections from recent absolute hints
- caller polls and applies correction with blend/caps

X2.0 scaffold note:
- a factor-lite graph assembly path is available behind config flags
- by default it is disabled and does not change correction behavior
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


def _wrap_angle_deg(angle_deg: float) -> float:
    """Wrap angle (deg) to [-180, 180)."""
    a = float(angle_deg)
    return float((a + 180.0) % 360.0 - 180.0)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median for 1D arrays."""
    if values.size == 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]
    w = np.maximum(weights[order], 0.0)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return float(np.median(values))
    cdf = np.cumsum(w) / w_sum
    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    idx = max(0, min(idx, v.size - 1))
    return float(v[idx])


@dataclass
class BackendCorrection:
    """Backend correction payload returned to frontend."""

    t_ref: float
    dp_enu: np.ndarray
    dyaw_deg: float
    cov_scale: float
    age_sec: float
    quality_score: float
    source_mix: Optional[dict[str, float]] = None
    residual_summary: Optional[dict[str, float]] = None
    contract_version: str = "v1"


@dataclass(frozen=True)
class BackendCorrectionContractV1:
    """Strict, normalized correction contract consumed by frontend apply path."""

    t_ref: float
    age_sec: float
    quality_score: float
    dp_enu: np.ndarray
    dyaw_deg: float
    cov_scale: float
    source_mix: dict[str, float]
    residual_summary: dict[str, float]
    contract_version: str = "v1"

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        strict_version: bool = True,
        expected_version: str = "v1",
        require_source_mix: bool = True,
        require_residual_summary: bool = True,
    ) -> tuple[Optional["BackendCorrectionContractV1"], str]:
        """Best-effort parse/validate correction payload into ContractV1."""

        def _get(key: str, default: Any = None) -> Any:
            if isinstance(payload, Mapping):
                return payload.get(key, default)
            return getattr(payload, key, default)

        required = ("t_ref", "age_sec", "quality_score", "dp_enu", "dyaw_deg", "cov_scale")
        for key in required:
            if _get(key, None) is None:
                return None, f"missing:{key}"

        version = str(_get("contract_version", "") or "").strip()
        if strict_version:
            if not version:
                return None, "missing:contract_version"
            if version != str(expected_version):
                return None, f"bad:contract_version={version}"
        elif not version:
            version = str(expected_version)

        try:
            t_ref = float(_get("t_ref"))
            age_sec = float(_get("age_sec"))
            quality_score = float(_get("quality_score"))
            dyaw_deg = float(_get("dyaw_deg"))
            cov_scale = float(_get("cov_scale"))
        except Exception:
            return None, "bad:scalar_cast"
        if not np.isfinite(t_ref):
            return None, "bad:t_ref_non_finite"
        if (not np.isfinite(age_sec)) or age_sec < 0.0:
            return None, "bad:age_sec"
        if not np.isfinite(quality_score):
            return None, "bad:quality_non_finite"
        if not np.isfinite(dyaw_deg):
            return None, "bad:dyaw_non_finite"
        if (not np.isfinite(cov_scale)) or cov_scale <= 0.0:
            return None, "bad:cov_scale"

        try:
            dp_enu = np.asarray(_get("dp_enu"), dtype=float).reshape(3,)
        except Exception:
            return None, "bad:dp_enu_shape"
        if not np.all(np.isfinite(dp_enu)):
            return None, "bad:dp_enu_non_finite"

        source_mix_raw = _get("source_mix", None)
        if require_source_mix and (not isinstance(source_mix_raw, Mapping) or len(source_mix_raw) == 0):
            return None, "missing:source_mix"
        source_mix: dict[str, float] = {}
        if isinstance(source_mix_raw, Mapping):
            for k, v in source_mix_raw.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    source_mix[str(k)] = float(fv)
        if require_source_mix and len(source_mix) == 0:
            return None, "bad:source_mix"

        residual_raw = _get("residual_summary", None)
        if require_residual_summary and (not isinstance(residual_raw, Mapping) or len(residual_raw) == 0):
            return None, "missing:residual_summary"
        residual_summary: dict[str, float] = {}
        if isinstance(residual_raw, Mapping):
            for k, v in residual_raw.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    residual_summary[str(k)] = float(fv)
        if require_residual_summary and len(residual_summary) == 0:
            return None, "bad:residual_summary"

        return (
            cls(
                t_ref=float(t_ref),
                age_sec=float(age_sec),
                quality_score=float(quality_score),
                dp_enu=np.asarray(dp_enu, dtype=float),
                dyaw_deg=float(dyaw_deg),
                cov_scale=float(cov_scale),
                source_mix=source_mix,
                residual_summary=residual_summary,
                contract_version=str(version),
            ),
            "",
        )


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
    source: str


@dataclass
class _FactorLiteGraph:
    """X2.0 scaffold: per-iteration factor-lite graph summary."""

    node_count: int
    factor_count_total: int
    factor_count_prior: int
    factor_count_abs_xyyaw: int
    factor_count_yaw_only: int
    factor_count_imu: int
    factor_count_visual: int
    factor_count_dem: int


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
        robust_yaw_enable: bool = True,
        robust_yaw_huber_deg: float = 6.0,
        switchable_constraints_enable: bool = True,
        switchable_quality_floor: float = 0.08,
        switchable_residual_xy_m: float = 12.0,
        switchable_residual_yaw_deg: float = 15.0,
        switchable_min_weight: float = 0.15,
        hybrid_factor_lite_enable: bool = False,
        hybrid_factor_lite_window: int = 10,
        hybrid_factor_lite_loss_xy_m: float = 8.0,
        hybrid_factor_lite_loss_yaw_deg: float = 10.0,
        hybrid_factor_lite_use_imu: bool = True,
        hybrid_factor_lite_use_visual: bool = True,
        hybrid_factor_lite_use_vps: bool = True,
        hybrid_factor_lite_use_dem: bool = True,
        hybrid_factor_lite_use_vps_xy: bool = True,
        hybrid_factor_lite_use_vps_yaw: bool = True,
        hybrid_factor_lite_vps_yaw_quality_floor: float = 0.30,
        hybrid_factor_lite_vps_yaw_cap_deg: float = 4.0,
        hybrid_factor_lite_use_loop_yaw: bool = True,
        hybrid_factor_lite_use_mag_yaw: bool = True,
        latest_wins_enable: bool = True,
        drop_stale_on_emit: bool = True,
        min_emit_dp_xy_m: float = 0.5,
        min_emit_dyaw_deg: float = 0.2,
    ):
        self.fixed_lag_window = max(3, int(fixed_lag_window))
        self.optimize_rate_hz = max(0.2, float(optimize_rate_hz))
        self.max_iteration_ms = max(1.0, float(max_iteration_ms))
        self.max_correction_age_sec = max(0.1, float(max_correction_age_sec))
        self.min_quality_score = float(np.clip(min_quality_score, 0.0, 1.0))
        self.max_abs_dp_xy_m = max(1.0, float(max_abs_dp_xy_m))
        self.max_abs_dyaw_deg = max(0.5, float(max_abs_dyaw_deg))
        self.robust_yaw_enable = bool(robust_yaw_enable)
        self.robust_yaw_huber_deg = max(0.1, float(robust_yaw_huber_deg))
        self.switchable_constraints_enable = bool(switchable_constraints_enable)
        self.switchable_quality_floor = float(np.clip(switchable_quality_floor, 0.0, 0.95))
        self.switchable_residual_xy_m = max(0.1, float(switchable_residual_xy_m))
        self.switchable_residual_yaw_deg = max(0.1, float(switchable_residual_yaw_deg))
        self.switchable_min_weight = float(np.clip(switchable_min_weight, 0.0, 1.0))
        self.hybrid_factor_lite_enable = bool(hybrid_factor_lite_enable)
        self.hybrid_factor_lite_window = max(3, int(hybrid_factor_lite_window))
        self.hybrid_factor_lite_loss_xy_m = max(0.1, float(hybrid_factor_lite_loss_xy_m))
        self.hybrid_factor_lite_loss_yaw_deg = max(0.1, float(hybrid_factor_lite_loss_yaw_deg))
        self.hybrid_factor_lite_use_imu = bool(hybrid_factor_lite_use_imu)
        self.hybrid_factor_lite_use_visual = bool(hybrid_factor_lite_use_visual)
        self.hybrid_factor_lite_use_vps = bool(hybrid_factor_lite_use_vps)
        self.hybrid_factor_lite_use_dem = bool(hybrid_factor_lite_use_dem)
        self.hybrid_factor_lite_use_vps_xy = bool(hybrid_factor_lite_use_vps_xy)
        self.hybrid_factor_lite_use_vps_yaw = bool(hybrid_factor_lite_use_vps_yaw)
        self.hybrid_factor_lite_vps_yaw_quality_floor = float(
            np.clip(hybrid_factor_lite_vps_yaw_quality_floor, 0.0, 1.0)
        )
        self.hybrid_factor_lite_vps_yaw_cap_deg = float(
            max(0.05, abs(float(hybrid_factor_lite_vps_yaw_cap_deg)))
        )
        self.hybrid_factor_lite_use_loop_yaw = bool(hybrid_factor_lite_use_loop_yaw)
        self.hybrid_factor_lite_use_mag_yaw = bool(hybrid_factor_lite_use_mag_yaw)
        self.latest_wins_enable = bool(latest_wins_enable)
        self.drop_stale_on_emit = bool(drop_stale_on_emit)
        self.min_emit_dp_xy_m = max(0.0, float(min_emit_dp_xy_m))
        self.min_emit_dyaw_deg = max(0.0, float(min_emit_dyaw_deg))

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._keyframes: deque[_Keyframe] = deque(maxlen=max(32, self.fixed_lag_window * 3))
        self._abs_hints: deque[_AbsHint] = deque(maxlen=max(64, self.fixed_lag_window * 6))
        self._pending_corrections: deque[BackendCorrection] = deque(maxlen=16)
        self._latest_correction: Optional[BackendCorrection] = None

        self.stats = {
            "iterations": 0,
            "corrections_emitted": 0,
            "last_iter_ms": 0.0,
            "last_switch_weight_mean": 1.0,
            "last_switch_weight_min": 1.0,
            "last_switch_weight_max": 1.0,
            "last_hint_count": 0,
            "factor_lite_enabled": int(self.hybrid_factor_lite_enable),
            "factor_lite_node_count": 0,
            "factor_lite_count_total": 0,
            "factor_lite_count_prior": 0,
            "factor_lite_count_abs_xyyaw": 0,
            "factor_lite_count_yaw_only": 0,
            "factor_lite_count_imu": 0,
            "factor_lite_count_visual": 0,
            "factor_lite_count_dem": 0,
            "factor_lite_build_ms": 0.0,
            "corrections_overwritten": 0,
            "corrections_emit_stale_dropped": 0,
            "corrections_polled": 0,
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

    def report_absolute_hint(
        self,
        t_ref: float,
        dp_enu: np.ndarray,
        dyaw_deg: float = 0.0,
        quality_score: float = 0.5,
        source: str = "ABS",
    ) -> None:
        """Report absolute correction hint (e.g., VPS, relocalization)."""
        dp = np.asarray(dp_enu, dtype=float).reshape(3,)
        if not np.all(np.isfinite(dp)):
            return
        hint = _AbsHint(
            t=float(t_ref),
            dp_enu=dp.copy(),
            dyaw_deg=float(dyaw_deg) if np.isfinite(dyaw_deg) else 0.0,
            quality=float(np.clip(quality_score, 0.0, 1.0)),
            source=str(source or "ABS"),
        )
        with self._lock:
            self._abs_hints.append(hint)

    def poll_correction(self, t_now: Optional[float] = None) -> Optional[BackendCorrection]:
        """Poll latest backend correction; drops stale corrections."""
        now = time.time() if t_now is None else float(t_now)
        dropped = 0
        with self._lock:
            if self.latest_wins_enable:
                corr = self._latest_correction
                self._latest_correction = None
                if corr is None:
                    self.last_poll_stale_drops = 0
                    return None
                age = max(0.0, now - float(corr.t_ref))
                corr.age_sec = age
                if age > self.max_correction_age_sec:
                    self.last_poll_stale_drops = 1
                    return None
                self.last_poll_stale_drops = 0
                self.stats["corrections_polled"] = int(self.stats.get("corrections_polled", 0)) + 1
                return corr
            while self._pending_corrections:
                corr = self._pending_corrections.popleft()
                age = max(0.0, now - float(corr.t_ref))
                corr.age_sec = age
                if age <= self.max_correction_age_sec:
                    self.last_poll_stale_drops = dropped
                    self.stats["corrections_polled"] = int(self.stats.get("corrections_polled", 0)) + 1
                    return corr
                dropped += 1
            self.last_poll_stale_drops = dropped
        return None

    def _store_correction(self, corr: Optional[BackendCorrection], emit_now_t: float) -> bool:
        """Store one computed correction according to transport policy."""
        if corr is None:
            return False
        if self.drop_stale_on_emit:
            emit_age = max(0.0, float(emit_now_t) - float(corr.t_ref))
            corr.age_sec = float(emit_age)
            if emit_age > self.max_correction_age_sec:
                self.stats["corrections_emit_stale_dropped"] = int(
                    self.stats.get("corrections_emit_stale_dropped", 0)
                ) + 1
                return False
        if self.latest_wins_enable:
            if self._latest_correction is not None:
                self.stats["corrections_overwritten"] = int(self.stats.get("corrections_overwritten", 0)) + 1
            self._latest_correction = corr
            self.stats["corrections_emitted"] = int(self.stats.get("corrections_emitted", 0)) + 1
            return True
        if len(self._pending_corrections) >= int(self._pending_corrections.maxlen or 0) and len(self._pending_corrections) > 0:
            self.stats["corrections_overwritten"] = int(self.stats.get("corrections_overwritten", 0)) + 1
        self._pending_corrections.append(corr)
        self.stats["corrections_emitted"] = int(self.stats.get("corrections_emitted", 0)) + 1
        return True

    def _worker_loop(self) -> None:
        period = 1.0 / max(self.optimize_rate_hz, 1e-6)
        while self._running:
            t0 = time.time()
            corr = self._compute_correction()
            iter_ms = (time.time() - t0) * 1000.0
            with self._lock:
                self.stats["iterations"] = int(self.stats.get("iterations", 0)) + 1
                self.stats["last_iter_ms"] = float(iter_ms)
                emit_now_t = float(corr.t_ref) if corr is not None else 0.0
                if len(self._keyframes) > 0:
                    emit_now_t = float(self._keyframes[-1].t)
                self._store_correction(corr, emit_now_t=emit_now_t)
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

        # X2.0 scaffold only: assemble factor-lite summary without changing solver behavior.
        if self.hybrid_factor_lite_enable:
            graph_t0 = time.time()
            graph = self._build_factor_lite_graph(kf_tail, win_hints)
            graph_ms = (time.time() - graph_t0) * 1000.0
            with self._lock:
                self.stats["factor_lite_node_count"] = int(graph.node_count)
                self.stats["factor_lite_count_total"] = int(graph.factor_count_total)
                self.stats["factor_lite_count_prior"] = int(graph.factor_count_prior)
                self.stats["factor_lite_count_abs_xyyaw"] = int(graph.factor_count_abs_xyyaw)
                self.stats["factor_lite_count_yaw_only"] = int(graph.factor_count_yaw_only)
                self.stats["factor_lite_count_imu"] = int(graph.factor_count_imu)
                self.stats["factor_lite_count_visual"] = int(graph.factor_count_visual)
                self.stats["factor_lite_count_dem"] = int(graph.factor_count_dem)
                self.stats["factor_lite_build_ms"] = float(graph_ms)
            win_hints = self._select_factor_lite_hints(win_hints)
            if len(win_hints) == 0:
                return None

        q = np.array([float(np.clip(h.quality, 0.0, 1.0)) for h in win_hints], dtype=float)
        if not np.all(np.isfinite(q)):
            return None
        w_base = np.maximum(1e-3, q)

        dp_stack = np.array([h.dp_enu for h in win_hints], dtype=float)
        dyaw_arr = np.array([h.dyaw_deg for h in win_hints], dtype=float)
        if not np.all(np.isfinite(dp_stack)):
            return None
        if not np.all(np.isfinite(dyaw_arr)):
            dyaw_arr = np.zeros_like(dyaw_arr)

        # Switchable constraints: down-weight hints with large residuals to robust center.
        switch_w = np.ones_like(w_base)
        if self.switchable_constraints_enable and len(win_hints) >= 3:
            try:
                center_xy = np.array(
                    [
                        _weighted_median(dp_stack[:, 0], w_base),
                        _weighted_median(dp_stack[:, 1], w_base),
                    ],
                    dtype=float,
                )
                dyaw_wrapped = np.array([_wrap_angle_deg(v) for v in dyaw_arr], dtype=float)
                yaw_center = _weighted_median(dyaw_wrapped, w_base)
                yaw_res = np.array(
                    [abs(_wrap_angle_deg(dy - yaw_center)) for dy in dyaw_arr],
                    dtype=float,
                )
                xy_res = np.linalg.norm(dp_stack[:, :2] - center_xy.reshape(1, 2), axis=1)
                sw_resid = 1.0 / (
                    1.0
                    + (xy_res / self.switchable_residual_xy_m) ** 2
                    + (yaw_res / self.switchable_residual_yaw_deg) ** 2
                )
                sw_quality = np.maximum(self.switchable_quality_floor, q)
                switch_w = np.clip(sw_resid * sw_quality, self.switchable_min_weight, 1.0)
            except Exception:
                switch_w = np.ones_like(w_base)

        w = w_base * switch_w
        w_sum = float(np.sum(w))
        if w_sum <= 1e-9:
            return None
        w /= w_sum

        source_mix: dict[str, float] = {}
        for wi, h in zip(w, win_hints):
            src = str(getattr(h, "source", "ABS")).upper()
            source_mix[src] = float(source_mix.get(src, 0.0) + float(wi))
        if source_mix:
            total_mix = float(sum(source_mix.values()))
            if total_mix > 1e-12:
                for k in list(source_mix.keys()):
                    source_mix[k] = float(source_mix[k] / total_mix)

        dp_est = np.sum(dp_stack * w.reshape(-1, 1), axis=0)

        # Robust yaw aggregation (Huber re-weighted around circular center).
        if self.robust_yaw_enable and len(dyaw_arr) >= 2:
            dyaw_wrapped = np.array([_wrap_angle_deg(v) for v in dyaw_arr], dtype=float)
            yaw_center = _weighted_median(dyaw_wrapped, w)
            yaw_delta = np.array([_wrap_angle_deg(dy - yaw_center) for dy in dyaw_arr], dtype=float)
            abs_delta = np.abs(yaw_delta)
            huber = self.robust_yaw_huber_deg
            huber_w = np.where(abs_delta <= huber, 1.0, huber / np.maximum(abs_delta, 1e-6))
            w_yaw = w * huber_w
            w_yaw_sum = float(np.sum(w_yaw))
            if w_yaw_sum > 1e-9:
                w_yaw = w_yaw / w_yaw_sum
                dyaw_est = float(_wrap_angle_deg(yaw_center + float(np.sum(w_yaw * yaw_delta))))
            else:
                dyaw_est = float(_wrap_angle_deg(yaw_center))
        else:
            dyaw_est = float(_wrap_angle_deg(float(np.sum(dyaw_arr * w))))

        q_est = float(np.clip(np.sum(w * q), 0.0, 1.0))

        if q_est < self.min_quality_score:
            return None

        dp_xy_norm = float(np.linalg.norm(dp_est[:2]))
        if dp_xy_norm > self.max_abs_dp_xy_m and dp_xy_norm > 1e-9:
            dp_est[:2] *= float(self.max_abs_dp_xy_m / dp_xy_norm)
        dyaw_est = float(np.clip(dyaw_est, -self.max_abs_dyaw_deg, self.max_abs_dyaw_deg))

        if float(np.linalg.norm(dp_est[:2])) < self.min_emit_dp_xy_m and abs(dyaw_est) < self.min_emit_dyaw_deg:
            return None

        elapsed_ms = (time.time() - t_start) * 1000.0
        if elapsed_ms > self.max_iteration_ms:
            return None

        with self._lock:
            self.stats["last_switch_weight_mean"] = float(np.mean(switch_w)) if switch_w.size else 1.0
            self.stats["last_switch_weight_min"] = float(np.min(switch_w)) if switch_w.size else 1.0
            self.stats["last_switch_weight_max"] = float(np.max(switch_w)) if switch_w.size else 1.0
            self.stats["last_hint_count"] = int(len(win_hints))

        cov_scale = float(np.clip(1.0 + 0.6 * (1.0 - q_est), 1.0, 1.8))
        residual_summary = {
            "hint_count": float(len(win_hints)),
            "switch_weight_mean": float(np.mean(switch_w)) if switch_w.size else 1.0,
            "switch_weight_min": float(np.min(switch_w)) if switch_w.size else 1.0,
            "switch_weight_max": float(np.max(switch_w)) if switch_w.size else 1.0,
        }
        return BackendCorrection(
            t_ref=float(t_max),
            dp_enu=np.asarray(dp_est, dtype=float),
            dyaw_deg=float(dyaw_est),
            cov_scale=cov_scale,
            age_sec=0.0,
            quality_score=q_est,
            source_mix=source_mix,
            residual_summary=residual_summary,
            contract_version="v1",
        )

    def _build_factor_lite_graph(self, kf_tail: list[_Keyframe], win_hints: list[_AbsHint]) -> _FactorLiteGraph:
        """
        X2.0 scaffold graph assembly.

        This is intentionally structural-only in X2.0:
        - builds factor counts to validate data wiring and runtime budget
        - does not alter correction estimates
        """
        nodes = min(len(kf_tail), self.hybrid_factor_lite_window)
        hints = win_hints[-max(1, self.hybrid_factor_lite_window * 2) :]
        prior_count = 1 if nodes > 0 else 0
        abs_xyyaw = 0
        yaw_only = 0
        for h in hints:
            src = str(h.source).upper()
            if src in ("VPS", "ABS"):
                if self.hybrid_factor_lite_use_vps and self.hybrid_factor_lite_use_vps_xy:
                    abs_xyyaw += 1
                if (
                    self.hybrid_factor_lite_use_vps
                    and self.hybrid_factor_lite_use_vps_yaw
                    and float(h.quality) >= self.hybrid_factor_lite_vps_yaw_quality_floor
                ):
                    yaw_only += 1
                continue
            if src == "DEM":
                if self.hybrid_factor_lite_use_dem:
                    abs_xyyaw += 1
                continue
            if src in ("LOOP", "BACKEND", "BACKEND_YAW"):
                if self.hybrid_factor_lite_use_loop_yaw or self.hybrid_factor_lite_use_visual:
                    yaw_only += 1
                continue
            if src == "MAG":
                if self.hybrid_factor_lite_use_mag_yaw:
                    yaw_only += 1
        imu_count = max(0, nodes - 1) if self.hybrid_factor_lite_use_imu else 0
        visual_count = max(0, int(0.5 * max(0, nodes - 1))) if self.hybrid_factor_lite_use_visual else 0
        dem_count = int(sum(1 for h in hints if str(h.source).upper() == "DEM")) if self.hybrid_factor_lite_use_dem else 0

        total = int(prior_count + abs_xyyaw + yaw_only + imu_count + visual_count + dem_count)
        return _FactorLiteGraph(
            node_count=int(nodes),
            factor_count_total=total,
            factor_count_prior=int(prior_count),
            factor_count_abs_xyyaw=int(abs_xyyaw),
            factor_count_yaw_only=int(yaw_only),
            factor_count_imu=int(imu_count),
            factor_count_visual=int(visual_count),
            factor_count_dem=int(dem_count),
        )

    def _select_factor_lite_hints(self, win_hints: list[_AbsHint]) -> list[_AbsHint]:
        """
        Select hints participating in factor-lite solve.

        X2.1 behavior:
        - use_vps controls absolute factors (VPS/ABS)
        - use_dem controls DEM absolute factor
        - use_visual controls loop-like yaw-only factors
        - use_imu currently has no direct hint source in this lightweight queue
        """
        out: list[_AbsHint] = []
        for h in win_hints:
            src = str(h.source).upper()
            if src in ("VPS", "ABS") and self.hybrid_factor_lite_use_vps:
                if self.hybrid_factor_lite_use_vps_xy or self.hybrid_factor_lite_use_vps_yaw:
                    dp = np.asarray(h.dp_enu, dtype=float).reshape(3,)
                    dyaw = float(h.dyaw_deg)
                    if not self.hybrid_factor_lite_use_vps_xy:
                        dp = np.zeros_like(dp)
                    yaw_allowed = bool(
                        self.hybrid_factor_lite_use_vps_yaw
                        and float(h.quality) >= self.hybrid_factor_lite_vps_yaw_quality_floor
                    )
                    if not yaw_allowed:
                        dyaw = 0.0
                    else:
                        dyaw = float(
                            np.clip(
                                dyaw,
                                -self.hybrid_factor_lite_vps_yaw_cap_deg,
                                self.hybrid_factor_lite_vps_yaw_cap_deg,
                            )
                        )
                    out.append(
                        _AbsHint(
                            t=float(h.t),
                            dp_enu=dp,
                            dyaw_deg=dyaw,
                            quality=float(h.quality),
                            source=str(h.source),
                        )
                    )
                continue
            if src == "DEM" and self.hybrid_factor_lite_use_dem:
                out.append(h)
                continue
            if src in ("LOOP", "BACKEND", "BACKEND_YAW") and (self.hybrid_factor_lite_use_loop_yaw or self.hybrid_factor_lite_use_visual):
                out.append(
                    _AbsHint(
                        t=float(h.t),
                        dp_enu=np.zeros(3, dtype=float),
                        dyaw_deg=float(h.dyaw_deg),
                        quality=float(h.quality),
                        source=str(h.source),
                    )
                )
                continue
            if src == "MAG" and self.hybrid_factor_lite_use_mag_yaw:
                out.append(
                    _AbsHint(
                        t=float(h.t),
                        dp_enu=np.zeros(3, dtype=float),
                        dyaw_deg=float(h.dyaw_deg),
                        quality=float(h.quality),
                        source=str(h.source),
                    )
                )
                continue
        return out
