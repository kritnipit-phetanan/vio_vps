import numpy as np

from vps.relocalization_policy import (
    build_relocalization_centers,
    extract_xy_sigma_m,
    should_force_global_relocalization,
)


def test_extract_xy_sigma_m_from_covariance():
    cov = np.array([[9.0, 0.0], [0.0, 4.0]], dtype=float)
    sigma = extract_xy_sigma_m(cov)
    assert np.isclose(sigma, 3.0)


def test_force_global_relocalization_rules():
    global_mode, reason = should_force_global_relocalization(
        force_global=False,
        accuracy_mode=True,
        objective="accuracy",
        reloc_enabled=True,
        fail_streak=0,
        fail_streak_trigger=6,
        since_success_sec=1.0,
        stale_success_sec=8.0,
        since_global_sec=15.0,
        global_interval_sec=12.0,
        est_cov_xy=np.eye(2),
        xy_sigma_trigger_m=35.0,
        phase=2,
        force_global_on_warning_phase=False,
    )
    assert global_mode is True
    assert reason == "periodic_global"

    global_mode, reason = should_force_global_relocalization(
        force_global=False,
        accuracy_mode=False,
        objective="stability",
        reloc_enabled=True,
        fail_streak=7,
        fail_streak_trigger=6,
        since_success_sec=1.0,
        stale_success_sec=8.0,
        since_global_sec=1.0,
        global_interval_sec=12.0,
        est_cov_xy=np.eye(2),
        xy_sigma_trigger_m=35.0,
        phase=2,
        force_global_on_warning_phase=False,
    )
    assert global_mode is True
    assert reason == "fail_streak"

    global_mode, reason = should_force_global_relocalization(
        force_global=False,
        accuracy_mode=False,
        objective="stability",
        reloc_enabled=True,
        fail_streak=0,
        fail_streak_trigger=6,
        since_success_sec=1.0,
        stale_success_sec=8.0,
        since_global_sec=1.0,
        global_interval_sec=12.0,
        est_cov_xy=np.diag([50.0 ** 2, 40.0 ** 2]),
        xy_sigma_trigger_m=35.0,
        phase=2,
        force_global_on_warning_phase=False,
    )
    assert global_mode is True
    assert reason == "high_covariance"


def test_build_relocalization_centers_respects_cap_and_keeps_origin_first():
    centers = build_relocalization_centers(
        est_lat=45.0,
        est_lon=-75.0,
        max_centers=5,
        ring_radius_m=[35.0, 80.0],
        ring_samples=8,
    )
    assert len(centers) <= 5
    assert centers[0] == (45.0, -75.0)
    assert len(set((round(a, 8), round(b, 8)) for a, b in centers)) == len(centers)
