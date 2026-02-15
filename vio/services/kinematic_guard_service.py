"""Position-velocity consistency guard service for VIORunner."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Tuple

import numpy as np


class KinematicGuardService:
    """
    Enforce short-horizon consistency between position and velocity states.

    Guard objective:
    - detect mismatch `||v_state - Δp/Δt||`
    - inflate velocity covariance when mismatch grows
    - softly blend velocity toward kinematic estimate on severe mismatch
    """

    def __init__(self, runner: Any):
        self.runner = runner
        self._hist: Deque[Tuple[float, np.ndarray]] = deque()
        self._last_action_t: float = -1e12

    def reset(self):
        self._hist.clear()
        self._last_action_t = -1e12

    def apply(self, t: float):
        runner = self.runner
        kf = getattr(runner, "kf", None)
        if kf is None:
            return
        if not bool(runner.global_config.get("KIN_GUARD_ENABLED", True)):
            return

        p_now = np.array(kf.x[0:3, 0], dtype=float).reshape(3,)
        t_now = float(t)
        if not np.all(np.isfinite(p_now)) or not np.isfinite(t_now):
            return

        self._hist.append((t_now, p_now.copy()))
        window_sec = float(runner.global_config.get("KIN_GUARD_WINDOW_SEC", 0.5))
        t_min = t_now - max(0.05, window_sec) - 0.1
        while len(self._hist) >= 2 and self._hist[0][0] < t_min:
            self._hist.popleft()
        if len(self._hist) < 2:
            return

        t_old, p_old = self._hist[0]
        dt = t_now - float(t_old)
        if dt <= 1e-3:
            return

        v_kin = (p_now - p_old) / dt
        v_state = np.array(kf.x[3:6, 0], dtype=float).reshape(3,)
        if not np.all(np.isfinite(v_kin)) or not np.all(np.isfinite(v_state)):
            return

        mismatch = float(np.linalg.norm(v_state - v_kin))
        runner._kin_guard_samples = int(getattr(runner, "_kin_guard_samples", 0)) + 1
        runner._kin_guard_last_mismatch = mismatch

        warn_th = float(runner.global_config.get("KIN_GUARD_VEL_MISMATCH_WARN", 8.0))
        hard_th = float(runner.global_config.get("KIN_GUARD_VEL_MISMATCH_HARD", 15.0))
        max_inflate = float(runner.global_config.get("KIN_GUARD_MAX_INFLATE", 1.25))
        alpha_hard = float(runner.global_config.get("KIN_GUARD_HARD_BLEND_ALPHA", 0.08))
        min_action_dt = float(runner.global_config.get("KIN_GUARD_MIN_ACTION_DT_SEC", 0.25))
        max_state_speed = float(runner.global_config.get("KIN_GUARD_MAX_STATE_SPEED_M_S", 120.0))
        max_blend_speed = float(runner.global_config.get("KIN_GUARD_MAX_BLEND_SPEED_M_S", 60.0))
        max_kin_speed = float(runner.global_config.get("KIN_GUARD_MAX_KIN_SPEED_M_S", 80.0))

        if mismatch <= warn_th:
            return
        if (t_now - self._last_action_t) < max(0.02, min_action_dt):
            return
        self._last_action_t = t_now

        runner._kin_guard_trigger_count = int(getattr(runner, "_kin_guard_trigger_count", 0)) + 1
        frac = (mismatch - warn_th) / max(1e-6, hard_th - warn_th)
        inflate = 1.0 + min(max(0.0, frac), 1.0) * max(0.0, max_inflate - 1.0)
        inflate = float(np.clip(inflate, 1.0, max(1.0, max_inflate)))
        kf.P[3:6, 3:6] *= float(inflate)
        if hasattr(kf, "ensure_covariance_valid"):
            try:
                kf.ensure_covariance_valid(stage="KIN_GUARD")
            except Exception:
                pass

        if mismatch >= hard_th:
            runner._kin_guard_hard_count = int(getattr(runner, "_kin_guard_hard_count", 0)) + 1
            alpha = float(np.clip(alpha_hard, 0.0, 1.0))
            v_state_norm = float(np.linalg.norm(v_state))
            v_kin_norm = float(np.linalg.norm(v_kin))
            allow_blend = (
                alpha > 0.0
                and v_state_norm <= max_blend_speed
                and v_kin_norm <= max_kin_speed
                and mismatch <= (3.0 * hard_th)
            )
            if allow_blend:
                v_new = (1.0 - alpha) * v_state + alpha * v_kin
                kf.x[3:6, 0] = v_new.reshape(3,)

        # Prevent non-physical speed bursts from contaminating downstream updates.
        v_now = np.array(kf.x[3:6, 0], dtype=float).reshape(3,)
        v_now_norm = float(np.linalg.norm(v_now))
        if np.isfinite(v_now_norm) and max_state_speed > 1e-3 and v_now_norm > max_state_speed:
            runner._kin_guard_speed_clamp_count = int(getattr(runner, "_kin_guard_speed_clamp_count", 0)) + 1
            kf.x[3:6, 0] = (v_now * (max_state_speed / v_now_norm)).reshape(3,)

        # Lightweight convention/debug monitor.
        if hasattr(runner, "output_reporting"):
            try:
                runner.output_reporting.log_convention_check(
                    t=t_now,
                    sensor="KIN_GUARD",
                    check="vel_mismatch_m_s",
                    value=mismatch,
                    threshold=warn_th,
                    status="PASS" if mismatch <= warn_th else ("WARN" if mismatch < hard_th else "FAIL"),
                    note=f"dt={dt:.3f},inflate={inflate:.3f}",
                )
            except Exception:
                pass
