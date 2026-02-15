import numpy as np

from vio.loop_closure import LoopClosureDetector


def test_loop_closure_double_confirm_fail_soft():
    detector = LoopClosureDetector(
        position_threshold=30.0,
        min_keyframe_dist=5.0,
        min_keyframe_yaw=5.0,
        min_frame_gap=10,
        min_match_ratio=0.1,
        min_inliers=15,
        quality_gate={
            "min_inliers_hard": 35,
            "min_inliers_failsoft": 20,
            "min_spatial_spread": 0.18,
            "max_reproj_px": 2.5,
            "yaw_residual_bound_deg": 25.0,
            "double_confirm_enable": True,
            "double_confirm_window_sec": 2.0,
            "double_confirm_yaw_deg": 8.0,
        },
        fail_soft={"enable": True},
    )
    detector.keyframes.append(
        {
            "frame_idx": 0,
            "position": np.array([0.0, 0.0], dtype=float),
            "yaw": 0.0,
            "keypoints": [],
            "descriptors": np.zeros((1, 32), dtype=np.uint8),
        }
    )

    def _fake_match(_img, _kf_idx, _K):
        return {
            "yaw_rel": 0.0,
            "num_inliers": 22,  # soft-pass but below hard threshold
            "inlier_ratio": 0.4,
            "spread_ratio": 0.22,
            "reproj_p95_px": 2.8,  # soft pass (2.5 * 1.6)
            "num_good_matches": 40,
        }

    detector.match_keyframe = _fake_match  # type: ignore[method-assign]
    img = np.zeros((64, 64), dtype=np.uint8)
    K = np.eye(3, dtype=float)
    pos = np.array([1.0, 1.0], dtype=float)
    yaw = np.deg2rad(10.0)

    first = detector.check_loop_closure(
        frame_idx=100,
        position=pos,
        yaw=yaw,
        gray_image=img,
        K=K,
        timestamp=1.0,
        phase=2,
        health_state="HEALTHY",
    )
    assert first is None

    second = detector.check_loop_closure(
        frame_idx=101,
        position=pos,
        yaw=yaw,
        gray_image=img,
        K=K,
        timestamp=1.5,
        phase=2,
        health_state="HEALTHY",
    )
    assert second is not None
    assert second["fail_soft"] is True
    assert int(second["num_inliers"]) >= 20

