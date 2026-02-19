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
        self._hard_exceed_start_t: float | None = None
        self._hard_active: bool = False
        self._speed_exceed_start_t: float | None = None
        self._speed_hard_active: bool = False

    def reset(self):
        self._hist.clear()
        self._last_action_t = -1e12
        self._hard_exceed_start_t = None
        self._hard_active = False
        self._speed_exceed_start_t = None
        self._speed_hard_active = False

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
        v_state_norm = float(np.linalg.norm(v_state))
        v_kin_norm = float(np.linalg.norm(v_kin))
        runner._kin_guard_samples = int(getattr(runner, "_kin_guard_samples", 0)) + 1
        runner._kin_guard_last_mismatch = mismatch

        warn_th = float(runner.global_config.get("KIN_GUARD_VEL_MISMATCH_WARN", 8.0))
        hard_th = float(runner.global_config.get("KIN_GUARD_VEL_MISMATCH_HARD", 15.0))
        hard_hold_sec = float(runner.global_config.get("KIN_GUARD_HARD_HOLD_SEC", 0.30))
        release_ratio = float(runner.global_config.get("KIN_GUARD_RELEASE_HYSTERESIS_RATIO", 0.75))
        max_inflate = float(runner.global_config.get("KIN_GUARD_MAX_INFLATE", 1.25))
        alpha_hard = float(runner.global_config.get("KIN_GUARD_HARD_BLEND_ALPHA", 0.08))
        min_action_dt = float(runner.global_config.get("KIN_GUARD_MIN_ACTION_DT_SEC", 0.25))
        max_state_speed = float(runner.global_config.get("KIN_GUARD_MAX_STATE_SPEED_M_S", 120.0))
        max_blend_speed = float(runner.global_config.get("KIN_GUARD_MAX_BLEND_SPEED_M_S", 60.0))
        max_kin_speed = float(runner.global_config.get("KIN_GUARD_MAX_KIN_SPEED_M_S", 80.0))
        speed_hard_th = float(
            runner.global_config.get("KIN_GUARD_SPEED_HARD_M_S", max_state_speed)
        )
        speed_hold_sec = float(
            runner.global_config.get("KIN_GUARD_SPEED_HARD_HOLD_SEC", hard_hold_sec)
        )
        speed_release_ratio = float(
            runner.global_config.get(
                "KIN_GUARD_SPEED_RELEASE_HYSTERESIS_RATIO",
                release_ratio,
            )
        )
        speed_blend_alpha = float(
            runner.global_config.get("KIN_GUARD_SPEED_BLEND_ALPHA", max(0.2, alpha_hard))
        )
        speed_inflate = float(runner.global_config.get("KIN_GUARD_SPEED_INFLATE", 1.12))
        release_th = float(max(0.0, warn_th * max(0.3, min(1.0, release_ratio))))
        speed_release_th = float(
            max(0.0, speed_hard_th * max(0.3, min(1.0, speed_release_ratio)))
        )

        # Hard state arm/disarm with dwell + hysteresis to avoid rapid toggling.
        if mismatch >= hard_th:
            if self._hard_exceed_start_t is None:
                self._hard_exceed_start_t = t_now
            elif (t_now - float(self._hard_exceed_start_t)) >= max(0.0, hard_hold_sec):
                self._hard_active = True
        else:
            self._hard_exceed_start_t = None
            if self._hard_active and mismatch <= release_th:
                self._hard_active = False

        # Independent speed runaway arm/disarm with dwell + hysteresis.
        if np.isfinite(v_state_norm) and v_state_norm >= speed_hard_th:
            if self._speed_exceed_start_t is None:
                self._speed_exceed_start_t = t_now
            elif (t_now - float(self._speed_exceed_start_t)) >= max(0.0, speed_hold_sec):
                self._speed_hard_active = True
        else:
            self._speed_exceed_start_t = None
            if self._speed_hard_active and (not np.isfinite(v_state_norm) or v_state_norm <= speed_release_th):
                self._speed_hard_active = False

        mismatch_hard_condition = bool(
            self._hard_active or (hard_hold_sec <= 1e-9 and mismatch >= hard_th)
        )
        speed_hard_condition = bool(
            self._speed_hard_active
            or (
                speed_hold_sec <= 1e-9
                and np.isfinite(v_state_norm)
                and v_state_norm >= speed_hard_th
            )
        )
        hard_condition = bool(mismatch_hard_condition or speed_hard_condition)

        if mismatch <= warn_th and not hard_condition:
            return
        if (t_now - self._last_action_t) < max(0.02, min_action_dt):
            return
        self._last_action_t = t_now

        runner._kin_guard_trigger_count = int(getattr(runner, "_kin_guard_trigger_count", 0)) + 1
        frac = (mismatch - warn_th) / max(1e-6, hard_th - warn_th)
        inflate = 1.0 + min(max(0.0, frac), 1.0) * max(0.0, max_inflate - 1.0)
        if speed_hard_condition and np.isfinite(speed_inflate):
            inflate = max(inflate, max(1.0, float(speed_inflate)))
        inflate = float(np.clip(inflate, 1.0, max(1.0, max_inflate)))
        kf.P[3:6, 3:6] *= float(inflate)
        if hasattr(kf, "ensure_covariance_valid"):
            try:
                kf.ensure_covariance_valid(stage="KIN_GUARD")
            except Exception:
                pass

        if hard_condition:
            runner._kin_guard_hard_count = int(getattr(runner, "_kin_guard_hard_count", 0)) + 1
            alpha = float(np.clip(alpha_hard, 0.0, 1.0))
            allow_blend = (
                alpha > 0.0
                and v_state_norm <= max_blend_speed
                and v_kin_norm <= max_kin_speed
                and mismatch <= (3.0 * hard_th)
            )
            if allow_blend and mismatch_hard_condition:
                v_new = (1.0 - alpha) * v_state + alpha * v_kin
                kf.x[3:6, 0] = v_new.reshape(3,)
            elif speed_hard_condition:
                alpha_speed = float(np.clip(max(alpha, speed_blend_alpha), 0.0, 1.0))
                if alpha_speed > 0.0 and np.isfinite(v_kin_norm) and v_kin_norm <= max_kin_speed:
                    v_new = (1.0 - alpha_speed) * v_state + alpha_speed * v_kin
                    kf.x[3:6, 0] = v_new.reshape(3,)
                    runner._kin_guard_speed_blend_count = int(
                        getattr(runner, "_kin_guard_speed_blend_count", 0)
                    ) + 1
                runner._kin_guard_speed_hard_count = int(
                    getattr(runner, "_kin_guard_speed_hard_count", 0)
                ) + 1

        # Prevent non-physical speed bursts from contaminating downstream updates.
        # Clamp when mismatch-hard OR speed-hard is active.
        v_now = np.array(kf.x[3:6, 0], dtype=float).reshape(3,)
        v_now_norm = float(np.linalg.norm(v_now))
        speed_cap = float(min(max_state_speed, speed_hard_th))
        if hard_condition and np.isfinite(v_now_norm) and speed_cap > 1e-3 and v_now_norm > speed_cap:
            runner._kin_guard_speed_clamp_count = int(getattr(runner, "_kin_guard_speed_clamp_count", 0)) + 1
            kf.x[3:6, 0] = (v_now * (speed_cap / v_now_norm)).reshape(3,)

        # Lightweight convention/debug monitor.
        if hasattr(runner, "output_reporting"):
            try:
                v_state_note = f"{v_state_norm:.2f}" if np.isfinite(v_state_norm) else "nan"
                v_kin_note = f"{v_kin_norm:.2f}" if np.isfinite(v_kin_norm) else "nan"
                runner.output_reporting.log_convention_check(
                    t=t_now,
                    sensor="KIN_GUARD",
                    check="vel_mismatch_m_s",
                    value=mismatch,
                    threshold=warn_th,
                    status="PASS" if mismatch <= warn_th else ("WARN" if mismatch < hard_th else "FAIL"),
                    note=(
                        f"dt={dt:.3f},inflate={inflate:.3f},"
                        f"hard={int(mismatch_hard_condition)},speed_hard={int(speed_hard_condition)},"
                        f"v={v_state_note},vkin={v_kin_note}"
                    ),
                )
            except Exception:
                pass
