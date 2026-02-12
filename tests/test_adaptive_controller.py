import math

from vio.adaptive_controller import (
    AdaptiveContext,
    AdaptiveController,
    DEGRADED,
    HEALTHY,
    RECOVERY,
)


def _ctx(t: float,
         phase: int = 2,
         p_cond: float = 1.0,
         p_max: float = 1.0,
         growth: float = 1.0,
         aiding_age: float = 0.1) -> AdaptiveContext:
    return AdaptiveContext(
        timestamp=t,
        phase=int(phase),
        p_cond=p_cond,
        p_max=p_max,
        p_trace=p_max,
        p_growth_ratio=growth,
        aiding_age_sec=aiding_age,
    )


def test_mode_semantics():
    off = AdaptiveController({"mode": "off"})
    d_off = off.step(_ctx(0.0))
    assert d_off.apply_process_noise is False
    assert d_off.apply_measurement is False

    shadow = AdaptiveController({"mode": "shadow"})
    d_shadow = shadow.step(_ctx(0.0))
    assert d_shadow.apply_process_noise is False
    assert d_shadow.apply_measurement is False

    active = AdaptiveController({"mode": "active"})
    d_active = active.step(_ctx(0.0))
    assert d_active.apply_process_noise is True
    assert d_active.apply_measurement is True


def test_health_transition_and_recovery():
    ctl = AdaptiveController({"mode": "active"})

    # Drive into degraded (hold_sec default is 0.25s).
    ctl.step(_ctx(0.0, p_cond=1e13, p_max=1e8, growth=1.2))
    d = ctl.step(_ctx(0.30, p_cond=1e13, p_max=1e8, growth=1.2))
    assert d.health_state == DEGRADED

    # Stay healthy long enough to enter recovery, then healthy.
    t = 0.60
    while t < 4.0:
        d = ctl.step(_ctx(t, p_cond=1.0, p_max=1.0, growth=1.0))
        t += 0.5
    assert d.health_state in (RECOVERY, HEALTHY)

    while t < 7.5:
        d = ctl.step(_ctx(t, p_cond=1.0, p_max=1.0, growth=1.0))
        t += 0.5
    assert d.health_state == HEALTHY


def test_nis_feedback_high_and_low():
    high = AdaptiveController({"mode": "active"})
    for i in range(30):
        high.record_measurement("DEM", accepted=False, nis_norm=10.0, timestamp=0.1 * i)
    d_high = high.step(_ctx(4.0))
    s_high = d_high.sensor_scale("DEM")
    assert s_high["r_scale"] > 1.0
    assert s_high["threshold_scale"] <= 1.0

    low = AdaptiveController({"mode": "active"})
    for i in range(40):
        low.record_measurement("DEM", accepted=True, nis_norm=0.1, timestamp=0.1 * i)
    d_low = low.step(_ctx(4.0))
    s_low = d_low.sensor_scale("DEM")
    assert s_low["r_scale"] < 1.0
    assert s_low["threshold_scale"] >= 1.0


def test_decision_clamp_ranges():
    ctl = AdaptiveController({
        "mode": "active",
        "process_noise": {
            "aiding_multiplier": {"FULL": 10.0, "PARTIAL": 10.0, "NONE": 10.0},
            "health_profile": {
                "HEALTHY": {
                    "sigma_accel_scale": 100.0,
                    "gyr_w_scale": 100.0,
                    "acc_w_scale": 100.0,
                    "sigma_unmodeled_gyr_scale": 100.0,
                    "min_yaw_scale": 100.0,
                }
            },
            "scale_clamp": {
                "sigma_accel": [0.5, 1.5],
                "gyr_w": [0.5, 1.5],
                "acc_w": [0.5, 1.5],
                "sigma_unmodeled_gyr": [0.5, 1.5],
                "min_yaw": [0.5, 1.5],
            },
        },
        "measurement": {
            "sensor_defaults": {"DEM": {"r_scale": 100.0, "threshold_scale": 100.0}},
            "clamp": {
                "r_scale": [0.5, 2.0],
                "chi2_scale": [0.5, 2.0],
                "threshold_scale": [0.5, 2.0],
                "reproj_scale": [0.5, 2.0],
            },
        },
    })

    # Force high NIS path to inflate, then clamp.
    for i in range(20):
        ctl.record_measurement("DEM", accepted=False, nis_norm=100.0, timestamp=0.1 * i)

    d = ctl.step(_ctx(3.0))
    assert 0.5 <= d.sigma_accel_scale <= 1.5
    assert 0.5 <= d.gyr_w_scale <= 1.5
    assert 0.5 <= d.acc_w_scale <= 1.5
    assert 0.5 <= d.sigma_unmodeled_gyr_scale <= 1.5
    assert 0.5 <= d.min_yaw_scale <= 1.5

    s = d.sensor_scale("DEM")
    assert 0.5 <= s["r_scale"] <= 2.0
    assert 0.5 <= s["threshold_scale"] <= 2.0
    assert math.isfinite(s["r_scale"])


def test_zupt_fail_soft_and_phase_profile_fields():
    ctl = AdaptiveController({
        "mode": "active",
        "measurement": {
            "phase_profiles": {
                "ZUPT": {
                    "1": {"chi2_scale": 0.8, "r_scale": 1.4},
                    "2": {"chi2_scale": 1.2, "r_scale": 0.9},
                }
            }
        },
    })

    d1 = ctl.step(_ctx(0.0, phase=1))
    d2 = ctl.step(_ctx(1.0, phase=2))
    s1 = d1.sensor_scale("ZUPT")
    s2 = d2.sensor_scale("ZUPT")

    assert "fail_soft_enable" in s1
    assert "hard_reject_factor" in s1
    assert "soft_r_cap" in s1
    assert "soft_r_power" in s1
    assert s1["chi2_scale"] != s2["chi2_scale"]


def test_gravity_sensor_policy_fields():
    ctl = AdaptiveController({"mode": "active"})
    d = ctl.step(_ctx(0.0, phase=0))
    s = d.sensor_scale("GRAVITY_RP")
    assert "sigma_deg" in s
    assert "acc_norm_tolerance" in s
    assert "max_gyro_rad_s" in s
