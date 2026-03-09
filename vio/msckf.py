#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSCKF Backend Module

Multi-State Constraint Kalman Filter implementation for VIO.
Includes feature triangulation, measurement Jacobians, and EKF updates.

References:
- Mourikis & Roumeliotis, "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation"
- OpenVINS (https://github.com/rpng/open_vins)

Author: VIO project
"""

import os
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from scipy.spatial.transform import Rotation as R_scipy

from .math_utils import skew_symmetric, safe_matrix_inverse
from .policy.types import MsckfQualitySnapshot


_MSCKF_LOG_EVERY_N = max(1, int(os.getenv("MSCKF_LOG_EVERY_N", "20")))
_MSCKF_LOG_COUNTER = 0


def _log_msckf_update(message: str):
    """Throttle frequent MSCKF update info logs to reduce console overhead."""
    global _MSCKF_LOG_COUNTER
    _MSCKF_LOG_COUNTER += 1
    if _MSCKF_LOG_COUNTER <= 3 or _MSCKF_LOG_COUNTER % _MSCKF_LOG_EVERY_N == 0:
        print(message)


# =============================================================================
# v2.9.10.0: Adaptive MSCKF Threshold (Priority 2)
# =============================================================================

def get_adaptive_reprojection_threshold(kf: Any,
                                       base_threshold: float = 12.0,
                                       reproj_scale: float = 1.0,
                                       phase: int = 2,
                                       health_state: str = "HEALTHY",
                                       global_config: Optional[dict] = None) -> float:
    """Adaptive MSCKF reprojection threshold based on filter convergence.
    
    v2.9.10.0 Priority 2: Start permissive during initialization (20px),
    tighten as filter converges (10px). This increases MSCKF success rate
    from 0.5 Hz to target 3-4 Hz.
    
    Args:
        kf: Kalman filter with state covariance P
        base_threshold: Baseline threshold (unused, kept for compatibility)
    
    Returns:
        Adaptive threshold in pixels
    """
    try:
        # Get velocity covariance magnitude
        P_vel = kf.P[3:6, 3:6]  # Velocity covariance block
        vel_sigma = np.sqrt(np.trace(P_vel) / 3)  # Average velocity std
        
        # Adaptive thresholding based on uncertainty:
        # - High uncertainty (initialization): permissive (20px)
        # - Medium uncertainty: moderate (15px)
        # - Low uncertainty (converged): strict (10px)
        if vel_sigma > 3.0:  # High uncertainty
            threshold = 20.0
        elif vel_sigma > 1.5:  # Medium uncertainty
            threshold = 15.0
        elif vel_sigma > 0.8:  # Converging
            threshold = 12.0
        else:  # Converged (vel_sigma < 0.8)
            threshold = 10.0
        
        scaled_threshold = threshold * max(1e-3, float(reproj_scale))

        phase_key = str(max(0, min(2, int(phase))))
        health_key = str(health_state).upper()
        phase_gate = {"0": 1.20, "1": 1.08, "2": 1.00}
        health_gate = {"HEALTHY": 1.00, "WARNING": 1.10, "DEGRADED": 1.20, "RECOVERY": 1.05}
        if isinstance(global_config, dict):
            try:
                phase_gate = dict(global_config.get("MSCKF_PHASE_REPROJ_GATE_SCALE", phase_gate))
            except Exception:
                pass
            try:
                health_gate = dict(global_config.get("MSCKF_HEALTH_REPROJ_GATE_SCALE", health_gate))
            except Exception:
                pass
        scaled_threshold *= float(phase_gate.get(phase_key, 1.0))
        scaled_threshold *= float(health_gate.get(health_key, 1.0))
        return float(np.clip(scaled_threshold, 2.0, 100.0))
        
    except Exception:
        # Fallback to permissive threshold if covariance unavailable
        scaled_threshold = 15.0 * max(1e-3, float(reproj_scale))
        return float(np.clip(scaled_threshold, 2.0, 100.0))


def _state_aware_reproj_policy(feature_quality: float,
                               phase: int,
                               health_state: str,
                               global_config: Optional[dict]) -> Dict[str, float]:
    """
    Build state-aware reprojection gate policy from phase/health/feature quality.

    Returns:
        dict with keys:
          - gate_mult: multiplicative factor on reprojection threshold
          - avg_gate_mult: multiplicative factor on average reproj gate
          - low_quality_reject: whether to hard reject low quality features
          - quality_band: high|mid|low
    """
    cfg = global_config if isinstance(global_config, dict) else {}
    q_high = float(cfg.get("MSCKF_REPROJ_QUALITY_HIGH_TH", 0.75))
    q_low = float(cfg.get("MSCKF_REPROJ_QUALITY_LOW_TH", 0.45))
    q_mid_mult = float(cfg.get("MSCKF_REPROJ_QUALITY_MID_GATE_MULT", 1.15))
    low_quality_reject = bool(cfg.get("MSCKF_REPROJ_QUALITY_LOW_REJECT", True))
    warning_mult = float(cfg.get("MSCKF_REPROJ_WARNING_SCALE", 1.20))
    degraded_mult = float(cfg.get("MSCKF_REPROJ_DEGRADED_SCALE", 1.35))

    band = "mid"
    if np.isfinite(feature_quality):
        if feature_quality >= q_high:
            band = "high"
        elif feature_quality < q_low:
            band = "low"

    gate_mult = 1.0
    avg_gate_mult = 1.0
    if band == "mid":
        gate_mult *= q_mid_mult
        avg_gate_mult *= q_mid_mult
    elif band == "low":
        # For low quality we can still fail-soft unless explicit low-reject is enabled.
        gate_mult *= max(q_mid_mult, 1.25)
        avg_gate_mult *= max(q_mid_mult, 1.20)

    health_key = str(health_state).upper()
    if health_key == "WARNING":
        gate_mult *= warning_mult
        avg_gate_mult *= max(1.0, 0.5 * (1.0 + warning_mult))
    elif health_key == "DEGRADED":
        gate_mult *= degraded_mult
        avg_gate_mult *= max(1.0, 0.5 * (1.0 + degraded_mult))

    # Dynamic phases are noisier; avoid reject bursts.
    phase_int = int(phase)
    if phase_int <= 1:
        gate_mult *= 1.08
        avg_gate_mult *= 1.05

    return {
        "gate_mult": float(np.clip(gate_mult, 0.7, 3.0)),
        "avg_gate_mult": float(np.clip(avg_gate_mult, 0.7, 3.0)),
        "low_quality_reject": bool(low_quality_reject),
        "quality_band": band,
    }
from .ekf import ExtendedKalmanFilter, ensure_covariance_valid
from .camera import normalized_to_unit_ray


# Default extrinsics (will be overridden when used)
BODY_T_CAMDOWN = np.eye(4, dtype=np.float64)

# Triangulation constants
# v2.9.9.6: REDUCED baseline threshold for helicopter hover (observed ~0.026m baseline)
MIN_MSCKF_BASELINE = 0.005  # Was 0.15 (too strict), reduced to 0.005m
MIN_PARALLAX_ANGLE_DEG = 0.3

# MSCKF statistics
MSCKF_STATS = {
    'total_attempt': 0,
    'success': 0,
    'fail_few_obs': 0,
    'fail_prefilter_geometry': 0,
    'fail_baseline': 0,
    'fail_parallax': 0,
    'fail_depth_sign': 0,
    'fail_depth_sign_init': 0,
    'fail_depth_sign_post_refine': 0,
    'fail_depth_large': 0,
    'depth_init_fail_count': 0,
    'depth_init_short_track_count': 0,
    'depth_init_parallax_low_count': 0,
    'depth_init_quality_low_count': 0,
    'depth_init_candidate_count': 0,
    'depth_init_routed_count': 0,
    'depth_init_gate_block_count': 0,
    'depth_sparse_recover_candidate_count': 0,
    'depth_sparse_recover_gate_block_count': 0,
    'depth_sparse_recover_parallax_low_count': 0,
    'reproj_eval_attempt': 0,
    'fail_reproj_error': 0,
    'fail_reproj_sparse': 0,
    'fail_reproj_sparse_recoverable': 0,
    'fail_depth_sparse_recoverable': 0,
    'fail_reproj_pixel': 0,
    'fail_reproj_normalized': 0,
    'fail_geometry_borderline': 0,
    'fail_geometry_insufficient_pretri': 0,
    'fail_geometry_insufficient_posttri': 0,
    'reclass_to_geometry_count': 0,
    'geometry_preagg_rebucket_count': 0,
    'preagg_callsite_dense_candidate_count': 0,
    'preagg_callsite_dense_failsoft_skip_count': 0,
    'preagg_callsite_dense_reclass_skip_count': 0,
    'preagg_callsite_dense_invoked_count': 0,
    'preagg_precond_called_count': 0,
    'preagg_precond_pass_count': 0,
    'preagg_precond_fail_invalid_input_count': 0,
    'preagg_precond_fail_avg_err_nan_count': 0,
    'preagg_precond_fail_avg_gate_nan_count': 0,
    'preagg_precond_fail_avg_gate_nonpos_count': 0,
    'preagg_precond_fail_avg_err_le_gate_count': 0,
    'preagg_precond_fail_reproj_bound_count': 0,
    'preagg_precond_fail_err_arr_too_small_count': 0,
    'preagg_predicate_eval_count': 0,
    'preagg_predicate_median_good_count': 0,
    'preagg_predicate_high_tail_count': 0,
    'preagg_predicate_enough_signal_count': 0,
    'preagg_predicate_quality_ok_count': 0,
    'preagg_predicate_all_true_count': 0,
    'preagg_parallax_low_entered_count': 0,
    'retry_lane_defer_count': 0,
    'posttri_retry_defer_count': 0,
    'posttri_retry_recover_defer_count': 0,
    'posttri_retry_recover_exhausted_count': 0,
    'posttri_retry_recover_success_count': 0,
    'posttri_retry_recover_depth_defer_count': 0,
    'posttri_retry_recover_depth_exhausted_count': 0,
    'posttri_retry_recover_depth_success_count': 0,
    'posttri_retry_recover_depth_gate_reject_count': 0,
    'posttri_retry_recover_depth_soft_accept_count': 0,
    'posttri_retry_recover_depth_retry_seen_count': 0,
    'posttri_retry_recover_depth_same_cycle_attempt_count': 0,
    'posttri_retry_recover_depth_same_cycle_success_count': 0,
    'posttri_retry_recover_depth_same_cycle_entered_count': 0,
    'posttri_retry_recover_depth_same_cycle_fail_depth_count': 0,
    'posttri_retry_recover_depth_same_cycle_fail_reproj_count': 0,
    'posttri_retry_recover_depth_same_cycle_fail_geometry_count': 0,
    'posttri_retry_recover_depth_same_cycle_fail_nonlinear_count': 0,
    'posttri_retry_recover_depth_same_cycle_fail_other_count': 0,
    'posttri_retry_recover_depth_same_cycle_clip_proj_count': 0,
    'posttri_retry_recover_depth_gate_override_count': 0,
    'posttri_retry_recover_depth_gate_override_reject_count': 0,
    'posttri_retry_recover_depth_borderline_promote_count': 0,
    'posttri_retry_recover_depth_protected_added_count': 0,
    'posttri_retry_recover_depth_protected_carried_count': 0,
    'posttri_retry_recover_depth_protected_truncated_count': 0,
    'posttri_retry_recover_depth_protected_missing_clone_count': 0,
    'posttri_retry_recover_depth_protected_source_missing_count': 0,
    'posttri_retry_recover_depth_protected_depth_gate_again_count': 0,
    'posttri_retry_recover_depth_relaxed_gate_used_count': 0,
    'posttri_retry_recover_depth_fail_depth_count': 0,
    'posttri_retry_recover_depth_fail_reproj_count': 0,
    'posttri_retry_recover_depth_fail_geometry_count': 0,
    'posttri_retry_recover_depth_fail_nonlinear_count': 0,
    'posttri_retry_recover_depth_fail_other_count': 0,
    'unstable_lane_count': 0,
    'stable_lane_used_count': 0,
    'fail_nonlinear': 0,
    'fail_chi2': 0,
    'fail_solver': 0,
    'fail_other': 0,
}

# Single-source fail-reason priority for post-tri retry routing and diagnostics.
MSCKF_FAIL_REASON_PRIORITY = (
    "fail_depth_sign_init",
    "fail_depth_sign_post_refine",
    "fail_depth_sign",
    "fail_prefilter_geometry",
    "fail_geometry_insufficient_pretri",
    "fail_geometry_insufficient_posttri",
    "fail_reproj_pixel",
    "fail_reproj_normalized",
    "fail_reproj_error",
    "fail_reproj_sparse",
    "fail_reproj_sparse_recoverable",
    "fail_depth_sparse_recoverable",
    "fail_geometry_borderline",
    "fail_parallax",
    "fail_baseline",
    "fail_nonlinear",
    "fail_solver",
    "fail_other",
)


def reset_msckf_stats():
    """Reset MSCKF statistics."""
    global MSCKF_STATS
    for key in MSCKF_STATS:
        MSCKF_STATS[key] = 0


def print_msckf_stats():
    """Print MSCKF triangulation statistics."""
    total = MSCKF_STATS['total_attempt']
    if total == 0:
        print("[MSCKF-STATS] No triangulation attempts")
        return
    
    success = MSCKF_STATS['success']
    print(f"[MSCKF-STATS] Total: {total}, Success: {success} ({100*success/total:.1f}%)")
    
    critical_fail_keys = {
        "fail_depth_sign",
        "fail_depth_sign_init",
        "fail_depth_sign_post_refine",
        "fail_reproj_error",
    }
    for key, val in MSCKF_STATS.items():
        if key.startswith('fail_') and (val > 0 or key in critical_fail_keys):
            print(f"  {key}: {val} ({100*val/total:.1f}%)")
    depth_total_fail = int(
        MSCKF_STATS.get("fail_depth_sign_init", 0)
        + MSCKF_STATS.get("fail_depth_sign_post_refine", 0)
        + MSCKF_STATS.get("fail_depth_large", 0)
    )
    reproj_eval_attempt = int(MSCKF_STATS.get("reproj_eval_attempt", 0))
    reproj_fail = int(MSCKF_STATS.get("fail_reproj_error", 0))
    reproj_rate_eval = (
        float(reproj_fail) / float(max(1, reproj_eval_attempt))
        if reproj_eval_attempt > 0
        else float("nan")
    )
    print(f"  depth_total_fail: {depth_total_fail} ({100*depth_total_fail/total:.1f}%)")
    print(
        "  reproj_eval:"
        f" attempt={reproj_eval_attempt},"
        f" fail={reproj_fail},"
        f" rate={reproj_rate_eval:.6f} per eval"
    )
    print(
        "  sparse_recover:"
        f" recoverable={int(MSCKF_STATS.get('fail_reproj_sparse_recoverable', 0))},"
        f" depth_recoverable={int(MSCKF_STATS.get('fail_depth_sparse_recoverable', 0))},"
        f" defer={int(MSCKF_STATS.get('posttri_retry_recover_defer_count', 0))},"
        f" exhausted={int(MSCKF_STATS.get('posttri_retry_recover_exhausted_count', 0))},"
        f" recovered={int(MSCKF_STATS.get('posttri_retry_recover_success_count', 0))},"
        f" depth_defer={int(MSCKF_STATS.get('posttri_retry_recover_depth_defer_count', 0))},"
        f" depth_exhausted={int(MSCKF_STATS.get('posttri_retry_recover_depth_exhausted_count', 0))},"
        f" depth_recovered={int(MSCKF_STATS.get('posttri_retry_recover_depth_success_count', 0))},"
        f" depth_gate_reject={int(MSCKF_STATS.get('posttri_retry_recover_depth_gate_reject_count', 0))},"
        f" depth_soft_accept={int(MSCKF_STATS.get('posttri_retry_recover_depth_soft_accept_count', 0))},"
        f" retry_seen={int(MSCKF_STATS.get('posttri_retry_recover_depth_retry_seen_count', 0))},"
        f" same_cycle_attempt={int(MSCKF_STATS.get('posttri_retry_recover_depth_same_cycle_attempt_count', 0))},"
        f" same_cycle_success={int(MSCKF_STATS.get('posttri_retry_recover_depth_same_cycle_success_count', 0))},"
        f" same_cycle_entered={int(MSCKF_STATS.get('posttri_retry_recover_depth_same_cycle_entered_count', 0))},"
        f" same_cycle_fail_depth={int(MSCKF_STATS.get('posttri_retry_recover_depth_same_cycle_fail_depth_count', 0))},"
        f" same_cycle_fail_reproj={int(MSCKF_STATS.get('posttri_retry_recover_depth_same_cycle_fail_reproj_count', 0))},"
        f" same_cycle_fail_geometry={int(MSCKF_STATS.get('posttri_retry_recover_depth_same_cycle_fail_geometry_count', 0))},"
        f" same_cycle_fail_nonlinear={int(MSCKF_STATS.get('posttri_retry_recover_depth_same_cycle_fail_nonlinear_count', 0))},"
        f" same_cycle_fail_other={int(MSCKF_STATS.get('posttri_retry_recover_depth_same_cycle_fail_other_count', 0))},"
        f" same_cycle_clip_proj={int(MSCKF_STATS.get('posttri_retry_recover_depth_same_cycle_clip_proj_count', 0))},"
        f" borderline_promote={int(MSCKF_STATS.get('posttri_retry_recover_depth_borderline_promote_count', 0))},"
        f" protected_add={int(MSCKF_STATS.get('posttri_retry_recover_depth_protected_added_count', 0))},"
        f" protected_carry={int(MSCKF_STATS.get('posttri_retry_recover_depth_protected_carried_count', 0))},"
        f" protected_trunc={int(MSCKF_STATS.get('posttri_retry_recover_depth_protected_truncated_count', 0))},"
        f" protected_missing_clone={int(MSCKF_STATS.get('posttri_retry_recover_depth_protected_missing_clone_count', 0))},"
        f" protected_source_missing={int(MSCKF_STATS.get('posttri_retry_recover_depth_protected_source_missing_count', 0))},"
        f" protected_depth_gate_again={int(MSCKF_STATS.get('posttri_retry_recover_depth_protected_depth_gate_again_count', 0))},"
        f" relaxed_gate={int(MSCKF_STATS.get('posttri_retry_recover_depth_relaxed_gate_used_count', 0))},"
        f" gate_override={int(MSCKF_STATS.get('posttri_retry_recover_depth_gate_override_count', 0))},"
        f" gate_override_reject={int(MSCKF_STATS.get('posttri_retry_recover_depth_gate_override_reject_count', 0))}"
    )
    print(
        "  depth_sparse_lane:"
        f" candidate={int(MSCKF_STATS.get('depth_sparse_recover_candidate_count', 0))},"
        f" gate_block={int(MSCKF_STATS.get('depth_sparse_recover_gate_block_count', 0))},"
        f" parallax_low={int(MSCKF_STATS.get('depth_sparse_recover_parallax_low_count', 0))}"
    )
    print(
        "  depth_init_lane:"
        f" fail={int(MSCKF_STATS.get('depth_init_fail_count', 0))},"
        f" short_track={int(MSCKF_STATS.get('depth_init_short_track_count', 0))},"
        f" parallax_low={int(MSCKF_STATS.get('depth_init_parallax_low_count', 0))},"
        f" quality_low={int(MSCKF_STATS.get('depth_init_quality_low_count', 0))},"
        f" candidate={int(MSCKF_STATS.get('depth_init_candidate_count', 0))},"
        f" routed={int(MSCKF_STATS.get('depth_init_routed_count', 0))},"
        f" gate_block={int(MSCKF_STATS.get('depth_init_gate_block_count', 0))}"
    )
    print(
        "  preagg lanes:"
        f" geometry_preagg_rebucket_count={int(MSCKF_STATS.get('geometry_preagg_rebucket_count', 0))}"
    )
    print(
        "  preagg callsites:"
        f" dense_candidate={int(MSCKF_STATS.get('preagg_callsite_dense_candidate_count', 0))},"
        f" dense_failsoft_skip={int(MSCKF_STATS.get('preagg_callsite_dense_failsoft_skip_count', 0))},"
        f" dense_reclass_skip={int(MSCKF_STATS.get('preagg_callsite_dense_reclass_skip_count', 0))},"
        f" dense_invoked={int(MSCKF_STATS.get('preagg_callsite_dense_invoked_count', 0))}"
    )
    print(
        "  preagg preconditions:"
        f" called={int(MSCKF_STATS.get('preagg_precond_called_count', 0))},"
        f" pass={int(MSCKF_STATS.get('preagg_precond_pass_count', 0))},"
        f" fail_invalid={int(MSCKF_STATS.get('preagg_precond_fail_invalid_input_count', 0))},"
        f" fail_avg_err_nan={int(MSCKF_STATS.get('preagg_precond_fail_avg_err_nan_count', 0))},"
        f" fail_avg_gate_nan={int(MSCKF_STATS.get('preagg_precond_fail_avg_gate_nan_count', 0))},"
        f" fail_gate_nonpos={int(MSCKF_STATS.get('preagg_precond_fail_avg_gate_nonpos_count', 0))},"
        f" fail_avg_le_gate={int(MSCKF_STATS.get('preagg_precond_fail_avg_err_le_gate_count', 0))},"
        f" fail_reproj_bound={int(MSCKF_STATS.get('preagg_precond_fail_reproj_bound_count', 0))},"
        f" fail_err_small={int(MSCKF_STATS.get('preagg_precond_fail_err_arr_too_small_count', 0))}"
    )
    print(
        "  preagg predicates:"
        f" eval={int(MSCKF_STATS.get('preagg_predicate_eval_count', 0))},"
        f" median_good={int(MSCKF_STATS.get('preagg_predicate_median_good_count', 0))},"
        f" high_tail={int(MSCKF_STATS.get('preagg_predicate_high_tail_count', 0))},"
        f" enough_signal={int(MSCKF_STATS.get('preagg_predicate_enough_signal_count', 0))},"
        f" quality_ok={int(MSCKF_STATS.get('preagg_predicate_quality_ok_count', 0))},"
        f" all_true={int(MSCKF_STATS.get('preagg_predicate_all_true_count', 0))}"
    )


def _pairwise_parallax_med_px(obs_list: List[dict], norm_scale_px: float) -> float:
    """Estimate median pairwise parallax in pixels from normalized observations."""
    pts: List[np.ndarray] = []
    for obs in obs_list:
        try:
            x, y = obs.get("pt_norm", (np.nan, np.nan))
            pt = np.array([float(x), float(y)], dtype=float)
            if np.all(np.isfinite(pt)):
                pts.append(pt)
        except Exception:
            continue
    if len(pts) < 2:
        return float("nan")
    dists: List[float] = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.linalg.norm(pts[j] - pts[i]))
            if np.isfinite(d):
                dists.append(d)
    if len(dists) == 0:
        return float("nan")
    scale = float(norm_scale_px if np.isfinite(norm_scale_px) and norm_scale_px > 1e-6 else 120.0)
    return float(np.median(np.asarray(dists, dtype=float)) * scale)


def _classify_unstable_reason_code(
    *,
    track_count: int,
    track_min: int,
    inlier_ratio: Optional[float] = None,
    inlier_min: Optional[float] = None,
    parallax_med: Optional[float] = None,
    parallax_min: Optional[float] = None,
    depth_ratio: Optional[float] = None,
    depth_min: Optional[float] = None,
    reproj_p95: Optional[float] = None,
    reproj_max: Optional[float] = None,
    unstable_depth_gate: Optional[bool] = None,
    use_inlier_gate: bool = True,
    use_reproj_gate: bool = True,
) -> str:
    """Single-source unstable reason mapping used by quality snapshot and triangulation lanes."""
    if int(track_count) < max(1, int(track_min)):
        return "track_count_low"

    if bool(use_inlier_gate):
        if inlier_min is not None:
            if (inlier_ratio is None) or (not np.isfinite(inlier_ratio)) or (float(inlier_ratio) < float(inlier_min)):
                return "inlier_ratio_low"

    if parallax_min is not None:
        if (parallax_med is None) or (not np.isfinite(parallax_med)) or (float(parallax_med) < float(parallax_min)):
            return "parallax_low"

    if unstable_depth_gate is None:
        if depth_min is not None:
            if (depth_ratio is None) or (not np.isfinite(depth_ratio)) or (float(depth_ratio) < float(depth_min)):
                return "depth_sign_low"
    elif bool(unstable_depth_gate):
        return "depth_sign_low"

    if bool(use_reproj_gate):
        if reproj_max is not None:
            if (reproj_p95 is None) or (not np.isfinite(reproj_p95)) or (float(reproj_p95) > float(reproj_max)):
                return "reproj_high"

    return "stable"


def _summarize_msckf_quality(
    t: float,
    track_count: int,
    accepted_count: int,
    parallax_px: List[float],
    reproj_p95_norm: List[float],
    depth_positive_ratio: List[float],
    feature_quality: List[float],
    global_config: Optional[dict] = None,
) -> MsckfQualitySnapshot:
    """Build compact MSCKF quality snapshot from per-feature aggregates."""
    tc = int(max(0, track_count))
    acc = int(max(0, accepted_count))
    inlier_ratio = float(acc / tc) if tc > 0 else float("nan")
    px_arr = np.asarray(parallax_px, dtype=float)
    rp_arr = np.asarray(reproj_p95_norm, dtype=float)
    dp_arr = np.asarray(depth_positive_ratio, dtype=float)
    q_arr = np.asarray(feature_quality, dtype=float)
    parallax_med = float(np.nanmedian(px_arr)) if px_arr.size else float("nan")
    reproj_p95 = float(np.nanpercentile(rp_arr, 95)) if rp_arr.size else float("nan")
    depth_ratio = float(np.nanmean(dp_arr)) if dp_arr.size else float("nan")
    q_score = float(np.nanmedian(q_arr)) if q_arr.size else float("nan")
    if not np.isfinite(q_score):
        # Fallback score from geometry consistency if feature quality unavailable.
        score_terms = []
        if np.isfinite(inlier_ratio):
            score_terms.append(float(np.clip(inlier_ratio, 0.0, 1.0)))
        if np.isfinite(reproj_p95):
            score_terms.append(float(np.clip(1.0 / (1.0 + 40.0 * reproj_p95), 0.0, 1.0)))
        if np.isfinite(depth_ratio):
            score_terms.append(float(np.clip(depth_ratio, 0.0, 1.0)))
        q_score = float(np.mean(score_terms)) if score_terms else float("nan")

    cfg = global_config if isinstance(global_config, dict) else {}
    tr_min = max(1, int(cfg.get("MSCKF_QUALITY_GATE_TRACK_MIN", 10)))
    inlier_min = float(cfg.get("MSCKF_QUALITY_GATE_INLIER_MIN", 0.30))
    parallax_min = float(cfg.get("MSCKF_QUALITY_GATE_PARALLAX_MIN_PX", 1.2))
    depth_min = float(cfg.get("MSCKF_QUALITY_GATE_DEPTH_POSITIVE_MIN", 0.62))
    reproj_max = float(cfg.get("MSCKF_QUALITY_GATE_REPROJ_P95_MAX", 0.06))
    camera_view = str(cfg.get("DEFAULT_CAMERA_VIEW", "nadir")).strip().lower()
    depth_floor_nadir = float(cfg.get("MSCKF_QUALITY_GATE_DEPTH_POSITIVE_MIN_FLOOR_NADIR", depth_min))
    # Keep default behavior conservative for sparse test/local configs that do not
    # provide adaptive-depth tuning keys explicitly.
    depth_track_relax_gain = float(cfg.get("MSCKF_QUALITY_GATE_DEPTH_POSITIVE_TRACK_RELAX_GAIN", 0.0))
    depth_parallax_relax_gain = float(cfg.get("MSCKF_QUALITY_GATE_DEPTH_POSITIVE_PARALLAX_RELAX_GAIN", 0.0))
    depth_parallax_relax_cap_ratio = float(
        cfg.get("MSCKF_QUALITY_GATE_DEPTH_POSITIVE_PARALLAX_RELAX_CAP_RATIO", 2.0)
    )
    adaptive_depth_min = float(depth_min)
    if camera_view == "nadir":
        # Nadir relaxation must stay evidence-driven: relax only when track/parallax
        # already indicate usable geometry, never when track support is weak.
        has_track_support = bool(tc >= tr_min)
        has_parallax_support = bool(np.isfinite(parallax_med) and float(parallax_med) >= parallax_min)

        track_relax = 0.0
        if has_track_support:
            track_surplus_ratio = float(
                np.clip(
                    (float(tc) - float(tr_min)) / float(max(1, tr_min)),
                    0.0,
                    1.0,
                )
            )
            track_relax = float(
                np.clip(
                    track_surplus_ratio * max(0.0, depth_track_relax_gain),
                    0.0,
                    max(0.0, depth_track_relax_gain),
                )
            )

        parallax_relax = 0.0
        if has_parallax_support and parallax_min > 1e-6:
            parallax_ratio = float(parallax_med) / float(parallax_min)
            parallax_ratio = float(np.clip(parallax_ratio, 1.0, max(1.0, depth_parallax_relax_cap_ratio)))
            parallax_surplus = max(0.0, parallax_ratio - 1.0)
            parallax_relax = float(
                np.clip(
                    parallax_surplus * max(0.0, depth_parallax_relax_gain),
                    0.0,
                    max(0.0, depth_parallax_relax_gain),
                )
            )

        if has_track_support or has_parallax_support:
            adaptive_depth_min = float(
                np.clip(
                    float(depth_min) - track_relax - parallax_relax,
                    min(float(depth_floor_nadir), float(depth_min)),
                    max(float(depth_floor_nadir), float(depth_min)),
                )
            )

    unstable_reason_code = _classify_unstable_reason_code(
        track_count=tc,
        track_min=tr_min,
        inlier_ratio=inlier_ratio,
        inlier_min=inlier_min,
        parallax_med=parallax_med,
        parallax_min=parallax_min,
        depth_ratio=depth_ratio,
        depth_min=adaptive_depth_min,
        reproj_p95=reproj_p95,
        reproj_max=reproj_max,
        use_inlier_gate=True,
        use_reproj_gate=True,
    )
    stable_geometry_flag = bool(unstable_reason_code == "stable")

    track_health = 0.0
    track_terms: List[float] = []
    if np.isfinite(inlier_ratio):
        track_terms.append(float(np.clip((float(inlier_ratio) - inlier_min) / max(1e-6, 1.0 - inlier_min), 0.0, 1.0)))
    if np.isfinite(parallax_med):
        track_terms.append(float(np.clip(float(parallax_med) / max(1e-6, 2.0 * parallax_min), 0.0, 1.0)))
    if np.isfinite(depth_ratio):
        track_terms.append(
            float(
                np.clip(
                    (float(depth_ratio) - adaptive_depth_min) / max(1e-6, 1.0 - adaptive_depth_min),
                    0.0,
                    1.0,
                )
            )
        )
    if tc > 0:
        track_terms.append(float(np.clip(float(tc) / float(max(tr_min * 2, 1)), 0.0, 1.0)))
    if track_terms:
        track_health = float(np.clip(np.mean(track_terms), 0.0, 1.0))

    risk_terms: List[float] = []
    if np.isfinite(reproj_p95):
        risk_terms.append(float(np.clip(float(reproj_p95) / max(1e-6, reproj_max), 0.0, 3.0)))
    if np.isfinite(depth_ratio):
        risk_terms.append(
            float(
                np.clip(
                    (adaptive_depth_min - float(depth_ratio)) / max(1e-6, adaptive_depth_min),
                    0.0,
                    3.0,
                )
            )
        )
    if np.isfinite(parallax_med):
        risk_terms.append(float(np.clip((parallax_min - float(parallax_med)) / max(1e-6, parallax_min), 0.0, 3.0)))
    conditioning_risk = float(np.clip(np.mean(risk_terms) if risk_terms else np.nan, 0.0, 3.0))

    return MsckfQualitySnapshot(
        timestamp=float(t),
        track_count=tc,
        inlier_ratio=float(inlier_ratio),
        parallax_med_px=float(parallax_med),
        reproj_p95_norm=float(reproj_p95),
        depth_positive_ratio=float(depth_ratio),
        quality_score=float(q_score),
        stable_geometry_flag=bool(stable_geometry_flag),
        conditioning_risk=float(conditioning_risk),
        feature_track_health=float(track_health),
        unstable_reason_code=str(unstable_reason_code),
    )


def _log_msckf_quality_csv(path: Optional[str], snap: MsckfQualitySnapshot) -> None:
    if path is None:
        return
    try:
        with open(path, "a", newline="") as f:
            f.write(
            f"{float(snap.timestamp):.6f},{int(snap.track_count)},"
            f"{float(snap.inlier_ratio):.6f},{float(snap.parallax_med_px):.6f},"
            f"{float(snap.reproj_p95_norm):.6f},{float(snap.depth_positive_ratio):.6f},"
            f"{float(snap.quality_score):.6f},{int(bool(snap.stable_geometry_flag))},"
            f"{float(snap.conditioning_risk):.6f},{float(snap.feature_track_health):.6f},"
            f"{str(snap.unstable_reason_code).replace(',', ';')}\n"
            )
    except Exception:
        pass


def _get_active_body_to_camera_transform(global_config: Optional[dict]) -> np.ndarray:
    """
    Resolve active body->camera extrinsic from compiled config.

    Falls back to BODY_T_CAMDOWN for backward compatibility.
    """
    if isinstance(global_config, dict):
        try:
            camera_view = str(global_config.get("DEFAULT_CAMERA_VIEW", "nadir"))
            view_cfgs = global_config.get("CAMERA_VIEW_CONFIGS", {})
            if isinstance(view_cfgs, dict):
                view_cfg = view_cfgs.get(camera_view, {})
                extr_key = view_cfg.get("extrinsics", None) if isinstance(view_cfg, dict) else None
                if isinstance(extr_key, str) and extr_key in global_config:
                    T = np.asarray(global_config.get(extr_key), dtype=np.float64)
                    if T.shape == (4, 4):
                        return T
            T_down = np.asarray(global_config.get("BODY_T_CAMDOWN", np.eye(4)), dtype=np.float64)
            if T_down.shape == (4, 4):
                return T_down
        except Exception:
            pass

    from .config import BODY_T_CAMDOWN
    return np.asarray(BODY_T_CAMDOWN, dtype=np.float64)


def _decompose_body_to_camera_transform(T_body_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose body->camera transform into forward and inverse components.

    Returns:
        R_BC, t_BC, R_CB, t_CB
    """
    t_mat = np.asarray(T_body_cam, dtype=np.float64)
    R_BC = t_mat[:3, :3]
    t_BC = t_mat[:3, 3]
    R_CB = R_BC.T
    t_CB = -R_CB @ t_BC
    return R_BC, t_BC, R_CB, t_CB


def imu_pose_to_camera_pose(q_imu: np.ndarray, p_imu: np.ndarray, 
                            T_body_cam: np.ndarray = None,
                            global_config: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert IMU/Body pose to camera pose using extrinsics.
    
    Args:
        q_imu: IMU quaternion [w,x,y,z] representing R_WB (body-to-world)
        p_imu: IMU position [x,y,z] in world frame
        T_body_cam: Body→Camera extrinsics T_BC (4x4 matrix)
        global_config: Global config dict (for loading BODY_T_CAMDOWN from YAML)
    
    Returns:
        q_cam: Camera quaternion [w,x,y,z] representing R_WC (camera-to-world)
        p_cam: Camera position [x,y,z] in world frame
    """
    if T_body_cam is None:
        T_body_cam = _get_active_body_to_camera_transform(global_config)

    # Body->camera extrinsics (T_BC) from calibration.
    _, _, R_CB, t_CB = _decompose_body_to_camera_transform(T_body_cam)

    # IMU/body orientation (R_BW: body -> world)
    q_imu_xyzw = np.array([q_imu[1], q_imu[2], q_imu[3], q_imu[0]])
    R_BW = R_scipy.from_quat(q_imu_xyzw).as_matrix()

    # Camera pose in world.
    R_CW = R_BW @ R_CB
    q_cam_xyzw = R_scipy.from_matrix(R_CW).as_quat()
    q_cam = np.array([q_cam_xyzw[3], q_cam_xyzw[0], q_cam_xyzw[1], q_cam_xyzw[2]])

    # t_CB is camera origin in body frame.
    p_cam = p_imu + R_BW @ t_CB

    return q_cam, p_cam


def get_feature_multi_view_observations(fid: int, cam_observations: List[dict]) -> List[dict]:
    """
    Extract all observations of a specific feature across multiple camera poses.
    
    Args:
        fid: Feature ID
        cam_observations: List of observation records
    
    Returns: List of {'cam_id', 'pt_pixel', 'pt_norm', 'quality', 'frame', 't'}
    """
    multi_view_obs = []
    for obs_record in cam_observations:
        if 'observations' in obs_record:
            for obs in obs_record['observations']:
                if obs['fid'] == fid:
                    pt_pixel = obs.get('pt_pixel', obs.get('pt_px', None))
                    multi_view_obs.append({
                        'cam_id': obs_record['cam_id'],
                        'pt_pixel': pt_pixel,
                        'pt_px': pt_pixel,
                        'pt_norm': obs['pt_norm'],
                        'quality': obs['quality'],
                        'frame': obs_record.get('frame', -1),
                        't': obs_record.get('t', 0.0)
                    })
        elif 'tracks' in obs_record:
            for track_fid, pt in obs_record['tracks']:
                if track_fid == fid:
                    multi_view_obs.append({
                        'cam_id': obs_record['cam_id'],
                        'pt_pixel': pt,
                        'pt_px': pt,
                        'pt_norm': pt,
                        'quality': 1.0,
                        'frame': -1,
                        't': 0.0
                    })
    # Keep one observation per cam clone (latest timestamp wins) and sort by time.
    by_cam: Dict[int, dict] = {}
    for obs in multi_view_obs:
        cam_id = int(obs.get('cam_id', -1))
        prev = by_cam.get(cam_id)
        if prev is None or float(obs.get('t', 0.0)) >= float(prev.get('t', 0.0)):
            by_cam[cam_id] = obs

    return sorted(by_cam.values(), key=lambda o: (float(o.get('t', 0.0)), int(o.get('cam_id', -1))))


def find_mature_features_for_msckf(vio_fe, cam_observations: List[dict],
                                   min_observations: int = 3,
                                   global_config: Optional[dict] = None,
                                   max_features: int = 0) -> List[int]:
    """
    Find features ready for MSCKF update.
    
    Args:
        vio_fe: VIO frontend with track information
        cam_observations: List of observation records
        min_observations: Minimum observations required
    
    Returns: List of mature feature IDs
    """
    if vio_fe is None:
        return []
    
    cfg = global_config if isinstance(global_config, dict) else {}
    track_pack_enable = bool(cfg.get("MSCKF_TRACK_PACK_ENABLE", True))
    track_pack_min_inlier = float(cfg.get("MSCKF_TRACK_PACK_MIN_INLIER_RATIO", 0.60))
    track_pack_min_quality = float(cfg.get("MSCKF_TRACK_PACK_MIN_QUALITY_MEDIAN", 0.34))
    track_pack_recent_window = int(cfg.get("MSCKF_TRACK_PACK_RECENT_WINDOW", 6))
    track_pack_recent_inlier = float(cfg.get("MSCKF_TRACK_PACK_MIN_RECENT_INLIER_RATIO", 0.55))
    track_pack_max_tracks = int(cfg.get("MSCKF_TRACK_PACK_MAX_TRACKS", 0))
    track_pack_min_parallax_px = float(cfg.get("MSCKF_TRACK_PACK_MIN_PARALLAX_PX", 0.95))
    track_pack_min_time_span_sec = float(cfg.get("MSCKF_TRACK_PACK_MIN_TIME_SPAN_SEC", 0.10))
    effective_max_tracks = int(max(0, max_features))
    if track_pack_max_tracks > 0 and (effective_max_tracks <= 0 or track_pack_max_tracks < effective_max_tracks):
        effective_max_tracks = int(track_pack_max_tracks)

    mature_tracks: Dict[int, List[dict]]
    if track_pack_enable:
        mature_tracks = vio_fe.get_mature_tracks(
            min_inlier_ratio=float(track_pack_min_inlier),
            min_quality_median=float(track_pack_min_quality),
            min_recent_inlier_ratio=float(track_pack_recent_inlier),
            recent_window=int(track_pack_recent_window),
            max_tracks=int(effective_max_tracks),
        )
    else:
        mature_tracks = vio_fe.get_mature_tracks()
    mature_features = []
    
    for fid in mature_tracks.keys():
        obs = get_feature_multi_view_observations(fid, cam_observations)
        if len(obs) >= min_observations:
            if track_pack_enable and len(obs) >= 2:
                parallax_px = _pairwise_parallax_med_px(obs, 120.0)
                t_arr = np.asarray([float(o.get("t", np.nan)) for o in obs], dtype=float)
                if t_arr.size > 1 and np.isfinite(t_arr).any():
                    t_span = float(np.nanmax(t_arr) - np.nanmin(t_arr))
                else:
                    t_span = float("nan")
                low_parallax = bool(
                    np.isfinite(parallax_px)
                    and float(parallax_px) < float(max(0.1, track_pack_min_parallax_px))
                )
                short_span = bool(
                    (not np.isfinite(t_span))
                    or float(t_span) < float(max(1e-3, track_pack_min_time_span_sec))
                )
                if low_parallax and short_span:
                    continue
            mature_features.append(fid)
    if effective_max_tracks > 0 and len(mature_features) > effective_max_tracks:
        mature_features = mature_features[:effective_max_tracks]
    
    return mature_features


def triangulate_point_linear(observations: List[dict], cam_states: List[dict]) -> Optional[np.ndarray]:
    """
    Linear triangulation using DLT (Direct Linear Transform).
    
    Solves the linear least squares problem for 3D point from multiple views.
    Used as initial estimate for nonlinear refinement.
    
    Args:
        observations: List of observation dicts with 'uv' (normalized coordinates)
        cam_states: List of camera state dicts with 'q' and 'p'
        
    Returns:
        3D point in world frame, or None if insufficient observations
    """
    if len(observations) < 2:
        return None
        
    # Build DLT matrix
    A_rows = []
    
    for obs in observations:
        # Find matching camera state
        cam_state = None
        for cs in cam_states:
            if cs.get('frame_idx') == obs.get('frame_idx') or cs.get('clone_id') == obs.get('clone_id'):
                cam_state = cs
                break
        
        if cam_state is None:
            continue
            
        # Get camera pose (world to camera)
        q_wc = cam_state['q']  # [w,x,y,z]
        p_w = cam_state['p']   # Camera position in world
        
        # Convert quaternion to rotation matrix
        q_xyzw = np.array([q_wc[1], q_wc[2], q_wc[3], q_wc[0]])
        R_wc = R_scipy.from_quat(q_xyzw).as_matrix()  # World to camera

        # Projection: p_cam = R_wc @ (p_world - p_w)
        # Normalized: [u, v, 1]^T ~ R_wc @ (p_world - p_w)
        # Let t = -R_wc @ p_w, then p_cam = R_wc @ p_world + t
        t_cw = -R_wc @ p_w
        
        # Build 2 equations from each observation
        uv = obs['uv']  # Normalized image coordinates
        u, v = uv[0], uv[1]
        
        # Row for u: u*(r3^T @ P + t3) - (r1^T @ P + t1) = 0
        # Row for v: v*(r3^T @ P + t3) - (r2^T @ P + t2) = 0
        r1 = R_wc[0, :]
        r2 = R_wc[1, :]
        r3 = R_wc[2, :]
        
        A_rows.append(u * r3 - r1)
        A_rows.append(v * r3 - r2)
    
    if len(A_rows) < 4:  # Need at least 2 observations
        return None
    
    A = np.array(A_rows)
    
    # Build b vector
    b_rows = []
    for i, obs in enumerate(observations):
        cam_state = None
        for cs in cam_states:
            if cs.get('frame_idx') == obs.get('frame_idx') or cs.get('clone_id') == obs.get('clone_id'):
                cam_state = cs
                break
        if cam_state is None:
            continue
            
        q_wc = cam_state['q']
        p_w = cam_state['p']
        q_xyzw = np.array([q_wc[1], q_wc[2], q_wc[3], q_wc[0]])
        R_wc = R_scipy.from_quat(q_xyzw).as_matrix()
        t_cw = -R_wc @ p_w
        
        uv = obs['uv']
        u, v = uv[0], uv[1]
        
        b_rows.append(t_cw[0] - u * t_cw[2])
        b_rows.append(t_cw[1] - v * t_cw[2])
    
    b = np.array(b_rows)
    
    # Solve least squares: A @ p = b
    try:
        p_world, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return p_world
    except np.linalg.LinAlgError:
        return None


def triangulate_point_nonlinear(observations: List[dict], cam_states: List[dict], 
                                p_init: np.ndarray, kf: ExtendedKalmanFilter,
                                max_iters: int = 10, debug: bool = False,
                                global_config: Optional[dict] = None) -> Optional[np.ndarray]:
    """
    Nonlinear triangulation refinement using Gauss-Newton.
    
    Args:
        observations: List of observations
        cam_states: Camera state metadata
        p_init: Initial 3D point estimate
        kf: EKF with camera poses in state
        max_iters: Maximum iterations
    
    Returns: Refined 3D point or None
    """
    if p_init is None or len(observations) < 2:
        return None
    
    p = p_init.copy()
    
    for iteration in range(max_iters):
        H = []
        r = []
        valid_views = 0
        
        for obs in observations:
            cam_id = obs['cam_id']
            if cam_id >= len(cam_states):
                continue
            
            cs = cam_states[cam_id]
            q_idx = cs['q_idx']
            p_idx = cs['p_idx']
            
            q_imu = kf.x[q_idx:q_idx+4, 0]
            p_imu = kf.x[p_idx:p_idx+3, 0]
            q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu, global_config=global_config)
            
            # Transform point to camera frame
            q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
            R_cw = R_scipy.from_quat(q_xyzw).as_matrix()
            R_wc = R_cw.T
            
            p_c = R_wc @ (p - p_cam)
            
            # Skip invalid views instead of failing the entire feature.
            if p_c[2] <= 0.1:
                continue
            valid_views += 1
            
            # Project to normalized plane
            x_pred = p_c[0] / p_c[2]
            y_pred = p_c[1] / p_c[2]
            
            x_obs, y_obs = obs['pt_norm']
            r.append(x_obs - x_pred)
            r.append(y_obs - y_pred)
            
            # Jacobian
            inv_z = 1.0 / p_c[2]
            inv_z2 = inv_z * inv_z
            
            J_proj = np.array([
                [inv_z, 0, -p_c[0] * inv_z2],
                [0, inv_z, -p_c[1] * inv_z2]
            ])
            
            J = J_proj @ R_wc
            H.append(J)
        
        if len(r) < 4 or valid_views < 2:
            return None
        
        H = np.vstack(H)
        r = np.array(r).reshape(-1, 1)
        
        try:
            HTH = H.T @ H
            HTr = H.T @ r
            
            # v2.9.9.9: More aggressive damping to improve convergence
            # Reduces fail_nonlinear (15.2%) by making optimizer more stable
            lambda_lm = max(1e-2 * np.trace(HTH) / 3.0, 1e-5)  # 10× larger damping
            HTH_damped = HTH + lambda_lm * np.eye(3)
            
            dp = np.linalg.solve(HTH_damped, HTr)
            
            # Limit step size (more conservative)
            dp_norm = np.linalg.norm(dp)
            if dp_norm > 5.0:  # Was 10.0, now 5.0 for stability
                dp = dp * (5.0 / dp_norm)
            
            p = p + dp.reshape(3,)
            
            # v2.9.9.9: Relaxed convergence (1e-6 → 1e-5) for faster termination
            if np.linalg.norm(dp) < 1e-5:
                break
        except np.linalg.LinAlgError:
            return None
    
    return p


# =============================================================================
# v2.9.10.0: Multi-Baseline Triangulation (Priority 3)
# =============================================================================

def select_best_baseline_pairs(observations: List[dict],
                               cam_states: List[dict],
                               kf: ExtendedKalmanFilter,
                               global_config: Optional[dict] = None,
                               max_pairs: int = 6) -> List[Tuple[int, int]]:
    """
    Select observation pairs with best triangulation geometry.

    Score combines baseline, parallax angle, and temporal/frame separation.
    """
    if len(observations) < 2:
        return []

    pose_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for obs in observations:
        cam_id = int(obs.get("cam_id", -1))
        if cam_id < 0 or cam_id >= len(cam_states) or cam_id in pose_cache:
            continue
        cs = cam_states[cam_id]
        q_imu = kf.x[cs['q_idx']:cs['q_idx']+4, 0]
        p_imu = kf.x[cs['p_idx']:cs['p_idx']+3, 0]
        q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu, global_config=global_config)
        q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]], dtype=float)
        R_cw = R_scipy.from_quat(q_xyzw).as_matrix()
        pose_cache[cam_id] = (p_cam, R_cw, q_cam)

    MAX_NORM_COORD = 2.5
    scored_pairs: List[Tuple[float, int, int]] = []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            obs_i = observations[i]
            obs_j = observations[j]
            cam_i = int(obs_i.get("cam_id", -1))
            cam_j = int(obs_j.get("cam_id", -1))
            if cam_i == cam_j or cam_i not in pose_cache or cam_j not in pose_cache:
                continue

            xi, yi = obs_i['pt_norm']
            xj, yj = obs_j['pt_norm']
            if np.hypot(xi, yi) > MAX_NORM_COORD or np.hypot(xj, yj) > MAX_NORM_COORD:
                continue

            p_i, R_i_cw, _ = pose_cache[cam_i]
            p_j, R_j_cw, _ = pose_cache[cam_j]
            baseline = float(np.linalg.norm(p_j - p_i))
            if baseline < MIN_MSCKF_BASELINE * 0.5:
                continue

            ray_i = R_i_cw @ normalized_to_unit_ray(float(xi), float(yi))
            ray_j = R_j_cw @ normalized_to_unit_ray(float(xj), float(yj))
            ray_i /= max(1e-9, np.linalg.norm(ray_i))
            ray_j /= max(1e-9, np.linalg.norm(ray_j))
            parallax_deg = float(np.degrees(np.arccos(np.clip(np.dot(ray_i, ray_j), -1.0, 1.0))))
            if parallax_deg < MIN_PARALLAX_ANGLE_DEG * 0.5:
                continue

            frame_gap = abs(int(obs_j.get("frame", -1)) - int(obs_i.get("frame", -1)))
            time_gap = abs(float(obs_j.get("t", 0.0)) - float(obs_i.get("t", 0.0)))
            score = baseline * max(parallax_deg, 1e-3) * (1.0 + 0.05 * frame_gap + 0.02 * time_gap)
            scored_pairs.append((score, i, j))

    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    return [(i, j) for _, i, j in scored_pairs[:max_pairs]]


def triangulate_feature(fid: int, cam_observations: List[dict], cam_states: List[dict],
                        kf: ExtendedKalmanFilter, use_plane_constraint: bool = True,
                        ground_altitude: float = 0.0, debug: bool = False,
                        dem_reader = None,
                        origin_lat: float = 0.0, origin_lon: float = 0.0,
                        global_config: dict = None,
                        reproj_scale: float = 1.0,
                        phase: int = 2,
                        health_state: str = "HEALTHY",
                        retry_mode: str = "") -> Optional[dict]:
    """
    Triangulate a feature using multi-view observations.
    
    Args:
        fid: Feature ID
        cam_observations: All observation records
        cam_states: Camera state metadata
        kf: EKF with camera poses
        use_plane_constraint: Use ground plane for depth estimation
        ground_altitude: Ground altitude in MSL
        dem_reader: DEM reader for ground plane constraint
        origin_lat/origin_lon: Local projection origin
    
    Returns: {'p_w': np.ndarray, 'observations': List, 'quality': float} or None
    """
    global MSCKF_STATS
    
    obs_list = get_feature_multi_view_observations(fid, cam_observations)
    MSCKF_STATS['total_attempt'] += 1
    
    if len(obs_list) < 2:
        MSCKF_STATS['fail_few_obs'] += 1
        return None
    
    candidate_pairs = select_best_baseline_pairs(
        obs_list,
        cam_states,
        kf,
        global_config=global_config,
        max_pairs=6,
    )
    if len(candidate_pairs) == 0:
        MSCKF_STATS['fail_baseline'] += 1
        return None

    p_init = None
    best_pair_score = -np.inf
    fail_counts = {"baseline": 0, "parallax": 0, "solver": 0, "depth": 0, "other": 0}
    MAX_NORM_COORD = 2.5

    # L2 (logic-first): apply unstable-geometry-aware depth-sign policy in the
    # initial pair-selection stage as well, not only after nonlinear refinement.
    depth_gate_enable = bool(
        (global_config or {}).get("MSCKF_DEPTHSIGN_UNSTABLE_ENABLE", True)
        if isinstance(global_config, dict) else True
    )
    depth_gate_track_min = int(
        (global_config or {}).get("MSCKF_QUALITY_GATE_TRACK_MIN", 10)
        if isinstance(global_config, dict) else 10
    )
    depth_gate_parallax_min_px = float(
        (global_config or {}).get("MSCKF_QUALITY_GATE_PARALLAX_MIN_PX", 1.2)
        if isinstance(global_config, dict) else 1.2
    )
    depth_gate_mult = float(
        (global_config or {}).get("MSCKF_DEPTHSIGN_UNSTABLE_MIN_DEPTH_MULT", 0.35)
        if isinstance(global_config, dict) else 0.35
    )
    depth_gate_floor_m = float(
        (global_config or {}).get("MSCKF_DEPTHSIGN_UNSTABLE_MIN_DEPTH_FLOOR_M", 0.015)
        if isinstance(global_config, dict) else 0.015
    )
    depth_reclass_enable = bool(
        (global_config or {}).get("MSCKF_DEPTHSIGN_UNSTABLE_RECLASSIFY_ENABLE", True)
        if isinstance(global_config, dict) else True
    )
    # L2 one-knob: state-aware depth-sign guard for unstable geometry only.
    # Block weak-depth triangulation from entering updates when positive-depth
    # support is too thin under unstable geometry.
    depth_state_gate_enable = bool(
        (global_config or {}).get("MSCKF_DEPTHSIGN_STATE_AWARE_UNSTABLE_ENABLE", True)
        if isinstance(global_config, dict) else True
    )
    depth_state_gate_min_pos_ratio = float(
        (global_config or {}).get("MSCKF_DEPTHSIGN_STATE_AWARE_UNSTABLE_MIN_POS_RATIO", 0.26)
        if isinstance(global_config, dict) else 0.26
    )
    depth_state_gate_min_valid_obs = int(
        (global_config or {}).get("MSCKF_DEPTHSIGN_STATE_AWARE_UNSTABLE_MIN_VALID_OBS", 3)
        if isinstance(global_config, dict) else 3
    )
    depth_state_gate_min_reject_ratio = float(
        (global_config or {}).get("MSCKF_DEPTHSIGN_STATE_AWARE_UNSTABLE_MIN_REJECT_RATIO", 0.62)
        if isinstance(global_config, dict) else 0.62
    )
    depth_sign_strict_dom_ratio_th = float(
        (global_config or {}).get("MSCKF_DEPTHSIGN_STRICT_DOM_RATIO_TH", 0.92)
        if isinstance(global_config, dict) else 0.92
    )
    depth_sign_strict_min_valid_ratio = float(
        (global_config or {}).get("MSCKF_DEPTHSIGN_STRICT_MIN_VALID_RATIO", 0.22)
        if isinstance(global_config, dict) else 0.22
    )
    depth_sign_strict_min_parallax_px = float(
        (global_config or {}).get(
            "MSCKF_DEPTHSIGN_STRICT_MIN_PARALLAX_PX",
            max(1.4, float(depth_gate_parallax_min_px) * 1.25),
        )
        if isinstance(global_config, dict) else max(1.4, float(depth_gate_parallax_min_px) * 1.25)
    )
    depth_sign_strict_min_track_count = int(
        (global_config or {}).get(
            "MSCKF_DEPTHSIGN_STRICT_MIN_TRACK_COUNT",
            max(4, int(depth_gate_track_min)),
        )
        if isinstance(global_config, dict) else max(4, int(depth_gate_track_min))
    )
    depth_sign_strict_min_quality = float(
        (global_config or {}).get("MSCKF_DEPTHSIGN_STRICT_MIN_QUALITY", 0.45)
        if isinstance(global_config, dict) else 0.45
    )
    depth_sparse_recover_enable = bool(
        (global_config or {}).get("MSCKF_DEPTH_SPARSE_RECOVER_ENABLE", True)
        if isinstance(global_config, dict) else True
    )
    depth_sparse_recover_min_obs = int(
        (global_config or {}).get("MSCKF_DEPTH_SPARSE_RECOVER_MIN_OBS", 5)
        if isinstance(global_config, dict) else 5
    )
    depth_sparse_recover_min_parallax_px = float(
        (global_config or {}).get("MSCKF_DEPTH_SPARSE_RECOVER_MIN_PARALLAX_PX", 1.15)
        if isinstance(global_config, dict) else 1.15
    )
    depth_sparse_recover_min_quality = float(
        (global_config or {}).get("MSCKF_DEPTH_SPARSE_RECOVER_MIN_QUALITY", 0.34)
        if isinstance(global_config, dict) else 0.34
    )
    depth_sparse_recover_min_depth_dom_ratio = float(
        (global_config or {}).get("MSCKF_DEPTH_SPARSE_RECOVER_MIN_DEPTH_DOM_RATIO", 0.55)
        if isinstance(global_config, dict) else 0.55
    )
    depth_init_recover_enable = bool(
        (global_config or {}).get("MSCKF_DEPTH_INIT_RECOVER_ENABLE", True)
        if isinstance(global_config, dict) else True
    )
    depth_init_recover_min_obs = int(
        (global_config or {}).get(
            "MSCKF_DEPTH_INIT_RECOVER_MIN_OBS",
            max(4, int(depth_sparse_recover_min_obs)),
        )
        if isinstance(global_config, dict) else max(4, int(depth_sparse_recover_min_obs))
    )
    depth_init_recover_min_parallax_px = float(
        (global_config or {}).get(
            "MSCKF_DEPTH_INIT_RECOVER_MIN_PARALLAX_PX",
            max(0.35, 0.65 * float(depth_sparse_recover_min_parallax_px)),
        )
        if isinstance(global_config, dict) else max(0.35, 0.65 * float(depth_sparse_recover_min_parallax_px))
    )
    depth_init_recover_max_parallax_px = float(
        (global_config or {}).get(
            "MSCKF_DEPTH_INIT_RECOVER_MAX_PARALLAX_PX",
            max(0.5, 1.05 * float(depth_sparse_recover_min_parallax_px)),
        )
        if isinstance(global_config, dict) else max(0.5, 1.05 * float(depth_sparse_recover_min_parallax_px))
    )
    depth_init_recover_min_quality = float(
        (global_config or {}).get(
            "MSCKF_DEPTH_INIT_RECOVER_MIN_QUALITY",
            max(0.0, float(depth_sparse_recover_min_quality) - 0.02),
        )
        if isinstance(global_config, dict) else max(0.0, float(depth_sparse_recover_min_quality) - 0.02)
    )
    depth_init_recover_min_depth_fail_ratio = float(
        (global_config or {}).get("MSCKF_DEPTH_INIT_RECOVER_MIN_DEPTH_FAIL_RATIO", 0.60)
        if isinstance(global_config, dict) else 0.60
    )
    depth_retry_soft_accept_enable = bool(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_SOFT_ACCEPT_ENABLE", False)
        if isinstance(global_config, dict) else False
    )
    depth_retry_soft_depth_ratio = float(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_SOFT_DEPTH_RATIO", 0.72)
        if isinstance(global_config, dict) else 0.72
    )
    depth_retry_soft_max_promote = int(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_SOFT_MAX_PROMOTE", 2)
        if isinstance(global_config, dict) else 2
    )
    depth_retry_soft_error_mult = float(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_SOFT_ERROR_MULT", 1.15)
        if isinstance(global_config, dict) else 1.15
    )
    depth_retry_full_rescue_enable = bool(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_FULL_RESCUE_ENABLE", False)
        if isinstance(global_config, dict) else False
    )
    depth_retry_bounded_relax_enable = bool(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_BOUNDED_RELAX_ENABLE", False)
        if isinstance(global_config, dict) else False
    )
    depth_retry_bounded_relax_obs_delta = int(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_BOUNDED_RELAX_OBS_DELTA", 1)
        if isinstance(global_config, dict) else 1
    )
    depth_retry_bounded_relax_parallax_mult = float(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_BOUNDED_RELAX_PARALLAX_MULT", 0.88)
        if isinstance(global_config, dict) else 0.88
    )
    depth_retry_bounded_relax_quality_delta = float(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_BOUNDED_RELAX_QUALITY_DELTA", 0.03)
        if isinstance(global_config, dict) else 0.03
    )
    depth_retry_gate_override_enable = bool(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_GATE_OVERRIDE_ENABLE", False)
        if isinstance(global_config, dict) else False
    )
    retry_mode_key = str(retry_mode).strip().lower()
    protected_retry_context = bool(
        retry_mode_key in ("depth_sparse_recover", "depth_sparse_same_cycle_rescue")
    )
    use_depth_same_cycle_full_rescue = bool(
        retry_mode_key == "depth_sparse_same_cycle_rescue"
        and bool(depth_retry_full_rescue_enable)
    )
    use_depth_retry_soft_accept = bool(
        retry_mode_key in ("depth_sparse_recover", "depth_sparse_same_cycle_rescue")
        and bool(depth_retry_soft_accept_enable)
    )
    use_depth_retry_bounded_relax = bool(
        retry_mode_key in ("depth_sparse_recover", "depth_sparse_same_cycle_rescue")
        and bool(depth_retry_bounded_relax_enable)
    )
    depth_retry_soft_depth_ratio_use = float(
        np.clip(
            float(depth_retry_soft_depth_ratio)
            * (0.80 if use_depth_same_cycle_full_rescue else 1.0),
            0.25,
            1.0,
        )
    )
    depth_retry_soft_max_promote_use = int(
        max(0, int(depth_retry_soft_max_promote) + (1 if use_depth_same_cycle_full_rescue else 0))
    )
    depth_retry_soft_error_mult_use = float(
        np.clip(
            float(depth_retry_soft_error_mult)
            * (1.12 if use_depth_same_cycle_full_rescue else 1.0),
            1.0,
            2.5,
        )
    )
    # Deterministic stable-geometry reproj lane (single canonical lane).
    # Legacy stable_norm keys are kept as fallback aliases for backward compatibility.
    stable_lane_enable = bool(
        (global_config or {}).get(
            "MSCKF_STABLE_REPROJ_LANE_ENABLE",
            (global_config or {}).get("MSCKF_REPROJ_STABLE_NORM_ENABLE", True),
        )
        if isinstance(global_config, dict) else True
    )
    stable_lane_min_obs = int(
        (global_config or {}).get(
            "MSCKF_STABLE_REPROJ_LANE_MIN_OBS",
            (global_config or {}).get("MSCKF_REPROJ_STABLE_NORM_MIN_OBS", 8),
        )
        if isinstance(global_config, dict) else 8
    )
    stable_lane_min_quality = float(
        (global_config or {}).get(
            "MSCKF_STABLE_REPROJ_LANE_MIN_QUALITY",
            (global_config or {}).get("MSCKF_REPROJ_STABLE_NORM_MIN_QUALITY", 0.55),
        )
        if isinstance(global_config, dict) else 0.55
    )
    stable_lane_cap_percentile = float(
        (global_config or {}).get(
            "MSCKF_STABLE_REPROJ_LANE_CAP_PERCENTILE",
            (global_config or {}).get("MSCKF_REPROJ_STABLE_NORM_CAP_PERCENTILE", 0.80),
        )
        if isinstance(global_config, dict) else 0.80
    )
    stable_lane_cap_max_mult = float(
        (global_config or {}).get(
            "MSCKF_STABLE_REPROJ_LANE_CAP_MAX_MULT",
            (global_config or {}).get("MSCKF_REPROJ_STABLE_NORM_CAP_MAX_MULT", 1.08),
        )
        if isinstance(global_config, dict) else 1.08
    )
    stable_lane_max_borderline_ratio = float(
        (global_config or {}).get("MSCKF_STABLE_REPROJ_LANE_MAX_BORDERLINE_RATIO", 0.35)
        if isinstance(global_config, dict) else 0.35
    )
    stable_lane_min_valid_obs_ratio = float(
        (global_config or {}).get("MSCKF_STABLE_REPROJ_LANE_MIN_VALID_OBS_RATIO", 0.55)
        if isinstance(global_config, dict) else 0.55
    )
    stable_lane_max_reduction = float(
        (global_config or {}).get(
            "MSCKF_STABLE_REPROJ_LANE_MAX_REDUCTION",
            (global_config or {}).get("MSCKF_REPROJ_STABLE_NORM_MAX_REDUCTION", 0.08),
        )
        if isinstance(global_config, dict) else 0.08
    )
    unstable_lane_enable = bool(
        (global_config or {}).get("MSCKF_UNSTABLE_REPROJ_LANE_ENABLE", True)
        if isinstance(global_config, dict) else True
    )
    unstable_lane_min_obs = int(
        (global_config or {}).get("MSCKF_UNSTABLE_REPROJ_LANE_MIN_OBS", 6)
        if isinstance(global_config, dict) else 6
    )
    unstable_lane_cap_percentile = float(
        (global_config or {}).get("MSCKF_UNSTABLE_REPROJ_LANE_CAP_PERCENTILE", 0.68)
        if isinstance(global_config, dict) else 0.68
    )
    unstable_lane_cap_max_mult = float(
        (global_config or {}).get("MSCKF_UNSTABLE_REPROJ_LANE_CAP_MAX_MULT", 1.04)
        if isinstance(global_config, dict) else 1.04
    )
    unstable_lane_max_borderline_ratio = float(
        (global_config or {}).get("MSCKF_UNSTABLE_REPROJ_LANE_MAX_BORDERLINE_RATIO", 0.55)
        if isinstance(global_config, dict) else 0.55
    )
    unstable_lane_min_valid_obs_ratio = float(
        (global_config or {}).get("MSCKF_UNSTABLE_REPROJ_LANE_MIN_VALID_OBS_RATIO", 0.45)
        if isinstance(global_config, dict) else 0.45
    )
    unstable_lane_max_reduction = float(
        (global_config or {}).get("MSCKF_UNSTABLE_REPROJ_LANE_MAX_REDUCTION", 0.05)
        if isinstance(global_config, dict) else 0.05
    )
    sparse_reproj_recover_enable = bool(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_ENABLE", True)
        if isinstance(global_config, dict) else True
    )
    sparse_reproj_recover_min_obs = int(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_MIN_OBS", 5)
        if isinstance(global_config, dict) else 5
    )
    sparse_reproj_recover_min_parallax_px = float(
        (global_config or {}).get(
            "MSCKF_RETRY_LANE_POSTTRI_RECOVER_MIN_PARALLAX_PX",
            max(1.0, float(depth_gate_parallax_min_px)),
        )
        if isinstance(global_config, dict) else max(1.0, float(depth_gate_parallax_min_px))
    )
    sparse_reproj_recover_min_quality = float(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_MIN_QUALITY", 0.34)
        if isinstance(global_config, dict) else 0.34
    )
    sparse_reproj_recover_max_depth_dom_ratio = float(
        (global_config or {}).get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_MAX_DEPTH_DOM_RATIO", 0.45)
        if isinstance(global_config, dict) else 0.45
    )

    # Use lightweight signals available before triangulation succeeds.
    pretri_gate_enable = bool(
        (global_config or {}).get("MSCKF_PRETRI_GEOMETRY_GATE_ENABLE", True)
        if isinstance(global_config, dict) else True
    )
    pretri_min_obs = int(
        (global_config or {}).get("MSCKF_PRETRI_GEOMETRY_MIN_OBS", 6)
        if isinstance(global_config, dict) else 6
    )
    pretri_min_parallax_px = float(
        (global_config or {}).get(
            "MSCKF_PRETRI_GEOMETRY_MIN_PARALLAX_PX",
            max(0.8, float(depth_gate_parallax_min_px) * 0.9),
        )
        if isinstance(global_config, dict) else max(0.8, float(depth_gate_parallax_min_px) * 0.9)
    )
    pretri_min_time_span_sec = float(
        (global_config or {}).get("MSCKF_PRETRI_GEOMETRY_MIN_TIME_SPAN_SEC", 0.10)
        if isinstance(global_config, dict) else 0.10
    )
    pretri_min_quality = float(
        (global_config or {}).get("MSCKF_PRETRI_GEOMETRY_MIN_QUALITY", 0.34)
        if isinstance(global_config, dict) else 0.34
    )
    pretri_min_fail_signals = int(
        (global_config or {}).get("MSCKF_PRETRI_GEOMETRY_MIN_FAIL_SIGNALS", 2)
        if isinstance(global_config, dict) else 2
    )
    pretri_min_fail_signals = max(1, pretri_min_fail_signals)
    pretri_hard_min_obs = int(
        (global_config or {}).get(
            "MSCKF_PRETRI_GEOMETRY_HARD_MIN_OBS",
            max(3, int(pretri_min_obs) - 2),
        )
        if isinstance(global_config, dict) else max(3, int(pretri_min_obs) - 2)
    )
    pretri_hard_min_obs = max(2, pretri_hard_min_obs)
    parallax_med_px_init = _pairwise_parallax_med_px(obs_list, 120.0)
    t_arr_init = np.asarray([float(o.get("t", np.nan)) for o in obs_list], dtype=float)
    q_arr_init = np.asarray([float(o.get("quality", np.nan)) for o in obs_list], dtype=float)
    t_span_init = (
        float(np.nanmax(t_arr_init) - np.nanmin(t_arr_init))
        if (t_arr_init.size > 1 and np.isfinite(t_arr_init).any())
        else float("nan")
    )
    q_med_init = float(np.nanmedian(q_arr_init)) if (q_arr_init.size > 0 and np.isfinite(q_arr_init).any()) else float("nan")
    unstable_geometry_init = bool(depth_gate_enable and (
        len(obs_list) < max(2, int(depth_gate_track_min))
        or (np.isfinite(parallax_med_px_init) and float(parallax_med_px_init) < float(depth_gate_parallax_min_px))
    ))
    if pretri_gate_enable:
        low_obs = bool(len(obs_list) < max(2, int(pretri_min_obs)))
        low_parallax = bool(
            np.isfinite(parallax_med_px_init)
            and float(parallax_med_px_init) < float(max(0.1, pretri_min_parallax_px))
        )
        short_span = bool(
            (not np.isfinite(t_span_init))
            or float(t_span_init) < float(max(1e-3, pretri_min_time_span_sec))
        )
        low_quality = bool(
            np.isfinite(q_med_init)
            and float(q_med_init) < float(np.clip(pretri_min_quality, 0.0, 1.0))
        )
        fail_signal_count = int(low_obs) + int(low_parallax) + int(short_span) + int(low_quality)
        geometry_insufficient_pretri = bool(
            len(obs_list) < int(pretri_hard_min_obs)
            or (
                fail_signal_count >= int(pretri_min_fail_signals)
                and (
                    (low_parallax and (short_span or low_quality))
                    or (low_obs and low_parallax)
                )
            )
        )
        if geometry_insufficient_pretri:
            MSCKF_STATS['fail_geometry_insufficient_pretri'] += 1
            MSCKF_STATS['fail_geometry_borderline'] += 1
            MSCKF_STATS['reclass_to_geometry_count'] += 1
            return None

    for obs_idx0, obs_idx1 in candidate_pairs:
        obs0 = obs_list[obs_idx0]
        obs1 = obs_list[obs_idx1]
        cam_id0 = int(obs0['cam_id'])
        cam_id1 = int(obs1['cam_id'])
        if cam_id0 >= len(cam_states) or cam_id1 >= len(cam_states) or cam_id0 == cam_id1:
            fail_counts["other"] += 1
            continue

        cs0 = cam_states[cam_id0]
        cs1 = cam_states[cam_id1]
        p_imu0_quick = kf.x[cs0['p_idx']:cs0['p_idx']+3, 0]
        p_imu1_quick = kf.x[cs1['p_idx']:cs1['p_idx']+3, 0]
        baseline_estimate = np.linalg.norm(p_imu1_quick - p_imu0_quick)
        if baseline_estimate < MIN_MSCKF_BASELINE * 0.8:
            fail_counts["baseline"] += 1
            continue

        q_imu0 = kf.x[cs0['q_idx']:cs0['q_idx']+4, 0]
        p_imu0 = kf.x[cs0['p_idx']:cs0['p_idx']+3, 0]
        q0, p0 = imu_pose_to_camera_pose(q_imu0, p_imu0, global_config=global_config)

        q_imu1 = kf.x[cs1['q_idx']:cs1['q_idx']+4, 0]
        p_imu1 = kf.x[cs1['p_idx']:cs1['p_idx']+3, 0]
        q1, p1 = imu_pose_to_camera_pose(q_imu1, p_imu1, global_config=global_config)

        baseline = float(np.linalg.norm(p1 - p0))
        if baseline < MIN_MSCKF_BASELINE:
            fail_counts["baseline"] += 1
            continue

        x0, y0 = obs0['pt_norm']
        x1, y1 = obs1['pt_norm']
        if np.hypot(x0, y0) > MAX_NORM_COORD or np.hypot(x1, y1) > MAX_NORM_COORD:
            fail_counts["other"] += 1
            continue

        q0_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]], dtype=float)
        q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]], dtype=float)
        R0_cw = R_scipy.from_quat(q0_xyzw).as_matrix()
        R1_cw = R_scipy.from_quat(q1_xyzw).as_matrix()
        ray0_w = R0_cw @ normalized_to_unit_ray(float(x0), float(y0))
        ray1_w = R1_cw @ normalized_to_unit_ray(float(x1), float(y1))
        ray0_w /= max(1e-9, np.linalg.norm(ray0_w))
        ray1_w /= max(1e-9, np.linalg.norm(ray1_w))
        ray_angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(ray0_w, ray1_w), -1.0, 1.0))))
        if ray_angle_deg < MIN_PARALLAX_ANGLE_DEG:
            fail_counts["parallax"] += 1
            continue

        w = p0 - p1
        a = np.dot(ray0_w, ray0_w)
        b = np.dot(ray0_w, ray1_w)
        c = np.dot(ray1_w, ray1_w)
        d = np.dot(ray0_w, w)
        e = np.dot(ray1_w, w)
        denom = a * c - b * b
        if abs(denom) < 1e-6:
            fail_counts["solver"] += 1
            continue

        s = (b * e - c * d) / denom
        t_pair = (a * e - b * d) / denom
        p_ray0 = p0 + s * ray0_w
        p_ray1 = p1 + t_pair * ray1_w
        p_candidate = 0.5 * (p_ray0 + p_ray1)
        pair_score = baseline * max(ray_angle_deg, 1e-3)

        # Use camera-frame cheirality check (z>0) for consistency with projection model.
        p_c0 = R0_cw.T @ (p_candidate - p0)
        p_c1 = R1_cw.T @ (p_candidate - p1)
        depth0 = float(p_c0[2])
        depth1 = float(p_c1[2])
        min_depth_m = 0.05
        if bool(unstable_geometry_init):
            min_depth_m = max(
                float(depth_gate_floor_m),
                float(min_depth_m) * float(np.clip(depth_gate_mult, 0.1, 1.0)),
            )
        max_depth_m = 700.0
        if depth0 <= min_depth_m or depth1 <= min_depth_m:
            fail_counts["depth"] += 1
            continue
        if depth0 > max_depth_m or depth1 > max_depth_m:
            fail_counts["depth"] += 1
            continue

        if pair_score > best_pair_score:
            best_pair_score = pair_score
            p_init = p_candidate

    if p_init is None:
        if fail_counts["depth"] > 0:
            MSCKF_STATS['depth_init_fail_count'] += 1
            if len(obs_list) < int(max(2, depth_init_recover_min_obs)):
                MSCKF_STATS['depth_init_short_track_count'] += 1
            if np.isfinite(parallax_med_px_init) and float(parallax_med_px_init) < float(depth_init_recover_min_parallax_px):
                MSCKF_STATS['depth_init_parallax_low_count'] += 1
            if np.isfinite(q_med_init) and float(q_med_init) < float(np.clip(depth_init_recover_min_quality, 0.0, 1.0)):
                MSCKF_STATS['depth_init_quality_low_count'] += 1

            total_pair_fail = int(sum(int(v) for v in fail_counts.values()))
            depth_fail_ratio = float(fail_counts["depth"]) / float(max(1, total_pair_fail))
            parallax_init_ok = bool(
                np.isfinite(parallax_med_px_init)
                and float(parallax_med_px_init) >= float(depth_init_recover_min_parallax_px)
                and float(parallax_med_px_init) <= float(max(depth_init_recover_min_parallax_px, depth_init_recover_max_parallax_px))
            )
            quality_init_ok = bool(
                (not np.isfinite(q_med_init))
                or float(q_med_init) >= float(np.clip(depth_init_recover_min_quality, 0.0, 1.0))
            )
            init_recover_candidate = bool(
                bool(depth_init_recover_enable)
                and bool(depth_sparse_recover_enable)
                and bool(unstable_geometry_init)
                and len(obs_list) >= int(max(2, depth_init_recover_min_obs))
                and depth_fail_ratio >= float(np.clip(depth_init_recover_min_depth_fail_ratio, 0.0, 1.0))
                and parallax_init_ok
                and quality_init_ok
            )
            if init_recover_candidate:
                MSCKF_STATS['depth_init_candidate_count'] += 1
                MSCKF_STATS['depth_init_routed_count'] += 1
                MSCKF_STATS['depth_sparse_recover_candidate_count'] += 1
                MSCKF_STATS['fail_depth_sparse_recoverable'] += 1
                MSCKF_STATS['fail_geometry_borderline'] += 1
                MSCKF_STATS['reclass_to_geometry_count'] += 1
                return None
            MSCKF_STATS['depth_init_gate_block_count'] += 1
            MSCKF_STATS['fail_depth_sign_init'] += 1
            # Taxonomy split:
            # - init-stage depth failures are tracked as depth_large/init-specific,
            #   not as post-refine depth-sign failures.
            MSCKF_STATS['fail_depth_large'] += 1
            if bool(unstable_geometry_init) and bool(depth_reclass_enable):
                MSCKF_STATS['fail_geometry_borderline'] += 1
                MSCKF_STATS['reclass_to_geometry_count'] += 1
        elif fail_counts["parallax"] > 0:
            MSCKF_STATS['fail_parallax'] += 1
        elif fail_counts["baseline"] > 0:
            MSCKF_STATS['fail_baseline'] += 1
        elif fail_counts["solver"] > 0:
            MSCKF_STATS['fail_solver'] += 1
        else:
            MSCKF_STATS['fail_other'] += 1
        return None
    
    # Nonlinear refinement
    p_refined = triangulate_point_nonlinear(
        obs_list,
        cam_states,
        p_init,
        kf,
        debug=debug,
        global_config=global_config,
    )
    
    if p_refined is None:
        MSCKF_STATS['fail_nonlinear'] += 1
        return None
    
    # =========================================================================
    # NEW (Phase 2): MSCKF Reprojection Validation using kannala_brandt_project
    # =========================================================================
    # Validate triangulated point by reprojecting to all observed frames
    # This catches errors from incorrect camera model or calibration issues
    
    from .camera import kannala_brandt_project, make_KD_for_size
    
    # Attempt to get camera intrinsics (K, D) from global config
    # If not available, fall back to normalized coordinate reprojection
    use_pixel_reprojection = False
    K = None
    D = None
    try:
        KB_PARAMS = {}
        if isinstance(global_config, dict):
            KB_PARAMS = global_config.get('KB_PARAMS', {}) or {}
        if not KB_PARAMS:
            from .config import KB_PARAMS
        if KB_PARAMS:
            # Reconstruct K and D from KB params
            K, D = make_KD_for_size(KB_PARAMS, 
                                     int(KB_PARAMS.get('w', 1440)), 
                                     int(KB_PARAMS.get('h', 1080)))
            use_pixel_reprojection = True
    except Exception:
        pass  # Fall back to normalized coordinate method
    
    # Compute feature quality for state-aware reprojection policy.
    quality_vals = []
    for obs in obs_list:
        qv = float(obs.get('quality', np.nan))
        if np.isfinite(qv):
            quality_vals.append(qv)
    feature_quality = float(np.nanmean(quality_vals)) if len(quality_vals) > 0 else np.nan
    reproj_policy = _state_aware_reproj_policy(
        feature_quality=feature_quality,
        phase=int(phase),
        health_state=str(health_state),
        global_config=global_config,
    )
    # State-aware depth-sign handling (L2 one-knob):
    # When geometry is unstable, reduce minimum positive-depth threshold and
    # reclassify depth-dominant triangulation failures as borderline geometry
    # instead of strict depth-sign failures.
    depth_gate_enable = bool(
        (global_config or {}).get("MSCKF_DEPTHSIGN_UNSTABLE_ENABLE", True)
        if isinstance(global_config, dict) else True
    )
    depth_gate_quality_th = float(
        (global_config or {}).get("MSCKF_DEPTHSIGN_UNSTABLE_QUALITY_TH", 0.38)
        if isinstance(global_config, dict) else 0.38
    )
    depth_gate_track_min = int(
        (global_config or {}).get("MSCKF_QUALITY_GATE_TRACK_MIN", 10)
        if isinstance(global_config, dict) else 10
    )
    depth_gate_parallax_min_px = float(
        (global_config or {}).get("MSCKF_QUALITY_GATE_PARALLAX_MIN_PX", 1.2)
        if isinstance(global_config, dict) else 1.2
    )
    depth_gate_mult = float(
        (global_config or {}).get("MSCKF_DEPTHSIGN_UNSTABLE_MIN_DEPTH_MULT", 0.35)
        if isinstance(global_config, dict) else 0.35
    )
    depth_gate_floor_m = float(
        (global_config or {}).get("MSCKF_DEPTHSIGN_UNSTABLE_MIN_DEPTH_FLOOR_M", 0.015)
        if isinstance(global_config, dict) else 0.015
    )
    depth_reclass_enable = bool(
        (global_config or {}).get("MSCKF_DEPTHSIGN_UNSTABLE_RECLASSIFY_ENABLE", True)
        if isinstance(global_config, dict) else True
    )
    unstable_geometry_depth_gate = bool(
        depth_gate_enable and (
            str(reproj_policy.get("quality_band", "mid")) == "low"
            or (np.isfinite(feature_quality) and float(feature_quality) < float(depth_gate_quality_th))
        )
    )
    if bool(unstable_geometry_depth_gate):
        MSCKF_STATS['unstable_lane_count'] += 1

    # Compute reprojection error (ENHANCED with pixel-level validation)
    # v2.9.10.0: Adaptive threshold based on filter convergence (Priority 2)
    # Start permissive (20px) during initialization, tighten to 10px when converged
    MAX_REPROJ_ERROR_PX = get_adaptive_reprojection_threshold(
        kf,
        reproj_scale=reproj_scale,
        phase=int(phase),
        health_state=str(health_state),
        global_config=global_config,
    )
    MAX_REPROJ_ERROR_PX *= float(reproj_policy.get("gate_mult", 1.0))
    total_error = 0.0
    rescue_total_error = 0.0
    max_pixel_error = 0.0
    valid_obs: List[dict] = []
    rescue_valid_obs: List[dict] = []
    obs_error_norm_values: List[float] = []
    rescue_obs_error_norm_values: List[float] = []
    depth_rejects = 0
    reproj_rejects = 0
    pixel_rejects = 0
    norm_rejects = 0
    borderline_rejects = 0
    rescue_borderline_rejects = 0
    depth_retry_promoted_obs = 0
    rescue_promoted_obs = 0
    rescue_obs_set_used = False

    norm_scale_px = 120.0
    try:
        if use_pixel_reprojection and K is not None and np.isfinite(float(K[0, 0])) and float(K[0, 0]) > 1e-3:
            norm_scale_px = float(K[0, 0])
        else:
            kb_cfg = {}
            if isinstance(global_config, dict):
                kb_cfg = global_config.get('KB_PARAMS', {}) or {}
            if not kb_cfg:
                from .config import KB_PARAMS as _KB_PARAMS
                kb_cfg = _KB_PARAMS or {}
            fx_guess = float(kb_cfg.get('mu', kb_cfg.get('fx', norm_scale_px))) if kb_cfg else norm_scale_px
            if np.isfinite(fx_guess) and fx_guess > 1e-3:
                norm_scale_px = fx_guess
    except Exception:
        pass
    norm_threshold = MAX_REPROJ_ERROR_PX / max(1e-6, norm_scale_px)
    ray_threshold_deg = float(np.degrees(np.arctan(MAX_REPROJ_ERROR_PX / max(1e-6, norm_scale_px))))
    ray_threshold_deg = float(np.clip(ray_threshold_deg, 0.3, 8.0))
    ray_soft_factor = 1.8
    pixel_soft_factor = 1.6
    avg_gate_factor = 2.0
    try:
        if isinstance(global_config, dict):
            ray_soft_factor = float(global_config.get("MSCKF_RAY_SOFT_FACTOR", ray_soft_factor))
            pixel_soft_factor = float(global_config.get("MSCKF_PIXEL_SOFT_FACTOR", pixel_soft_factor))
            avg_gate_factor = float(global_config.get("MSCKF_AVG_REPROJ_GATE_FACTOR", avg_gate_factor))
    except Exception:
        pass
    phase_ray_soft = {"0": 1.12, "1": 1.06, "2": 1.00}
    health_ray_soft = {"HEALTHY": 1.00, "WARNING": 1.10, "DEGRADED": 1.20, "RECOVERY": 1.04}
    if isinstance(global_config, dict):
        try:
            phase_ray_soft = dict(global_config.get("MSCKF_PHASE_RAY_SOFT_SCALE", phase_ray_soft))
        except Exception:
            pass
        try:
            health_ray_soft = dict(global_config.get("MSCKF_HEALTH_RAY_SOFT_SCALE", health_ray_soft))
        except Exception:
            pass
    ray_soft_factor = float(np.clip(ray_soft_factor, 1.0, 6.0))
    pixel_soft_factor = float(np.clip(pixel_soft_factor, 1.0, 6.0))
    avg_gate_factor = float(np.clip(avg_gate_factor, 1.0, 6.0))
    phase_key = str(max(0, min(2, int(phase))))
    health_key = str(health_state).upper()
    ray_soft_factor *= float(phase_ray_soft.get(phase_key, 1.0))
    ray_soft_factor *= float(health_ray_soft.get(health_key, 1.0))
    pixel_soft_factor *= float(phase_ray_soft.get(phase_key, 1.0))
    pixel_soft_factor *= float(health_ray_soft.get(health_key, 1.0))
    ray_soft_factor = float(np.clip(ray_soft_factor, 1.0, 8.0))
    pixel_soft_factor = float(np.clip(pixel_soft_factor, 1.0, 8.0))
    phase_avg_gate = {"0": 1.15, "1": 1.05, "2": 1.00}
    health_avg_gate = {"HEALTHY": 1.00, "WARNING": 1.08, "DEGRADED": 1.15, "RECOVERY": 1.04}
    if isinstance(global_config, dict):
        try:
            phase_avg_gate = dict(global_config.get("MSCKF_PHASE_AVG_REPROJ_GATE_SCALE", phase_avg_gate))
        except Exception:
            pass
        try:
            health_avg_gate = dict(global_config.get("MSCKF_HEALTH_AVG_REPROJ_GATE_SCALE", health_avg_gate))
        except Exception:
            pass
    avg_gate_factor *= float(phase_avg_gate.get(phase_key, 1.0))
    avg_gate_factor *= float(health_avg_gate.get(health_key, 1.0))
    avg_gate_factor *= float(reproj_policy.get("avg_gate_mult", 1.0))

    if reproj_policy.get("quality_band", "mid") == "low" and bool(reproj_policy.get("low_quality_reject", True)):
        MSCKF_STATS['fail_geometry_borderline'] += 1
        return None
    
    parallax_med_px = _pairwise_parallax_med_px(obs_list, norm_scale_px)
    if bool(depth_gate_enable):
        if len(obs_list) < max(2, int(depth_gate_track_min)):
            unstable_geometry_depth_gate = True
        if np.isfinite(parallax_med_px) and float(parallax_med_px) < float(depth_gate_parallax_min_px):
            unstable_geometry_depth_gate = True
    unstable_reproj_early_reject_enable = bool(
        (global_config or {}).get("MSCKF_UNSTABLE_REPROJ_EARLY_REJECT_ENABLE", False)
        if isinstance(global_config, dict) else False
    )
    unstable_reproj_proven_track_max_obs = int(
        (global_config or {}).get(
            "MSCKF_UNSTABLE_REPROJ_PROVEN_TRACK_MAX_OBS",
            max(3, min(int(depth_gate_track_min), 5)),
        )
        if isinstance(global_config, dict) else max(3, min(int(depth_gate_track_min), 5))
    )
    unstable_reproj_proven_parallax_mult = float(
        (global_config or {}).get("MSCKF_UNSTABLE_REPROJ_PROVEN_PARALLAX_MULT", 0.85)
        if isinstance(global_config, dict) else 0.85
    )
    geometry_insuff_reclass_enable = bool(
        (global_config or {}).get("MSCKF_GEOMETRY_INSUFF_RECLASSIFY_ENABLE", True)
        if isinstance(global_config, dict) else True
    )
    geometry_insuff_reclass_mult = float(
        (global_config or {}).get("MSCKF_GEOMETRY_INSUFF_RECLASSIFY_MULT", 1.25)
        if isinstance(global_config, dict) else 1.25
    )
    geometry_insuff_max_valid_ratio = float(
        (global_config or {}).get("MSCKF_GEOMETRY_INSUFF_MAX_VALID_RATIO", 0.45)
        if isinstance(global_config, dict) else 0.45
    )
    geometry_insuff_max_track_obs = int(
        (global_config or {}).get("MSCKF_GEOMETRY_INSUFF_MAX_TRACK_OBS", 8)
        if isinstance(global_config, dict) else 8
    )
    geometry_insuff_parallax_mult = float(
        (global_config or {}).get("MSCKF_GEOMETRY_INSUFF_PARALLAX_MULT", 1.10)
        if isinstance(global_config, dict) else 1.10
    )
    preagg_min_track_obs = int(
        (global_config or {}).get(
            "MSCKF_GEOMETRY_PREAGG_MIN_TRACK_OBS",
            max(4, int(depth_gate_track_min) - 2),
        )
        if isinstance(global_config, dict) else max(4, int(depth_gate_track_min) - 2)
    )
    preagg_reproj_high_max_mult = float(
        (global_config or {}).get("MSCKF_GEOMETRY_PREAGG_REPROJ_HIGH_MAX_MULT", 1.08)
        if isinstance(global_config, dict) else 1.08
    )
    if bool(unstable_reproj_early_reject_enable) and bool(unstable_geometry_depth_gate):
        track_obs_cap = max(2, int(unstable_reproj_proven_track_max_obs))
        parallax_proven_mult = float(np.clip(unstable_reproj_proven_parallax_mult, 0.2, 1.2))
        low_track_geometry = bool(len(obs_list) <= track_obs_cap)
        low_parallax_geometry = bool(
            np.isfinite(parallax_med_px)
            and float(parallax_med_px) < float(depth_gate_parallax_min_px) * parallax_proven_mult
        )
        if low_track_geometry or low_parallax_geometry:
            # Deterministic unstable-geometry lane:
            # reject early as geometry-borderline to avoid counting these cases as
            # reprojection failures when the dominant issue is observability.
            MSCKF_STATS['fail_geometry_borderline'] += 1
            MSCKF_STATS['fail_geometry_insufficient_pretri'] += 1
            MSCKF_STATS['reclass_to_geometry_count'] += 1
            return None

    for obs in obs_list:
        cam_id = obs['cam_id']
        if cam_id >= len(cam_states):
            continue
        
        cs = cam_states[cam_id]
        q_imu = kf.x[cs['q_idx']:cs['q_idx']+4, 0]
        p_imu = kf.x[cs['p_idx']:cs['p_idx']+3, 0]
        q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu, global_config=global_config)
        
        q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
        R_cw = R_scipy.from_quat(q_xyzw).as_matrix()
        R_wc = R_cw.T
        
        # Transform point to camera frame
        p_c = R_wc @ (p_refined - p_cam)
        
        # v3.5.0: ADAPTIVE depth check based on position uncertainty
        # When P_pos is large, triangulation is noisy → need relaxed threshold
        # Get position uncertainty from filter
        P_pos_trace = kf.P[0,0] + kf.P[1,1] + kf.P[2,2]  # Total position variance
        pos_sigma = np.sqrt(P_pos_trace / 3.0)  # Average position std
        
        # Adaptive threshold: 0.05m (good case) to 0.5m (high uncertainty)
        min_depth_threshold = 0.05 + min(0.45, pos_sigma * 0.01)
        if bool(unstable_geometry_depth_gate):
            min_depth_threshold = max(
                float(depth_gate_floor_m),
                float(min_depth_threshold) * float(np.clip(depth_gate_mult, 0.1, 1.0)),
            )
        
        promoted_by_depth_retry = False
        promoted_by_same_cycle_rescue = False
        same_cycle_retry_mode = bool(
            bool(use_depth_same_cycle_full_rescue)
            and (retry_mode_key == "depth_sparse_same_cycle_rescue")
            and bool(protected_retry_context)
        )
        if p_c[2] < min_depth_threshold:
            soft_depth_threshold = max(
                float(depth_gate_floor_m) * 0.5,
                float(min_depth_threshold) * float(depth_retry_soft_depth_ratio_use),
            )
            same_cycle_rescue_threshold = max(
                float(depth_gate_floor_m) * 0.35,
                float(min_depth_threshold) * (0.48 if use_depth_same_cycle_full_rescue else 0.60),
            )
            same_cycle_retry_floor_threshold = max(
                float(depth_gate_floor_m) * 0.12,
                float(min_depth_threshold) * 0.22,
            )
            if (
                use_depth_retry_soft_accept
                and depth_retry_promoted_obs < int(depth_retry_soft_max_promote_use)
                and p_c[2] >= soft_depth_threshold
            ):
                promoted_by_depth_retry = True
            elif (
                use_depth_same_cycle_full_rescue
                and rescue_promoted_obs < int(depth_retry_soft_max_promote_use)
                and (
                    p_c[2] >= same_cycle_rescue_threshold
                    or (same_cycle_retry_mode and p_c[2] >= same_cycle_retry_floor_threshold)
                )
            ):
                promoted_by_same_cycle_rescue = True
                if bool(same_cycle_retry_mode) and p_c[2] < same_cycle_rescue_threshold:
                    # Retry-mode-only bounded relaxation for depth-sparse protected lane.
                    MSCKF_STATS['posttri_retry_recover_depth_relaxed_gate_used_count'] += 1
            else:
                depth_rejects += 1
                continue
        
        # Retry-mode same-cycle rescue lane:
        # use positive-depth clipped projection only for promoted observations to
        # avoid near-zero depth blow-ups while keeping legacy depth-gate behavior
        # unchanged for normal lanes.
        p_c_proj = p_c
        if promoted_by_same_cycle_rescue and bool(same_cycle_retry_mode):
            clip_depth_floor = max(
                float(depth_gate_floor_m) * 0.14,
                float(min_depth_threshold) * 0.35,
            )
            if p_c[2] < clip_depth_floor:
                p_c_proj = p_c.copy()
                vec_norm = float(np.linalg.norm(p_c_proj))
                if vec_norm > 1e-9:
                    ray_dir = p_c_proj / vec_norm
                    if ray_dir[2] > 1e-6:
                        p_c_proj = ray_dir * float(clip_depth_floor / ray_dir[2])
                    else:
                        p_c_proj[2] = clip_depth_floor
                else:
                    p_c_proj[2] = clip_depth_floor
                MSCKF_STATS['posttri_retry_recover_depth_same_cycle_clip_proj_count'] += 1

        # Compute normalized coordinates
        x_pred = p_c_proj[0] / p_c_proj[2]
        y_pred = p_c_proj[1] / p_c_proj[2]
        
        x_obs, y_obs = obs['pt_norm']
        norm_error = np.sqrt((x_obs - x_pred)**2 + (y_obs - y_pred)**2)

        # Ray-angle validation is more stable for fisheye than pure pixel/normalized residuals.
        ray_pred = p_c_proj.reshape(3,)
        ray_pred /= max(1e-9, np.linalg.norm(ray_pred))
        ray_obs = normalized_to_unit_ray(float(x_obs), float(y_obs))
        ray_err_deg = float(np.degrees(np.arccos(np.clip(np.dot(ray_pred, ray_obs), -1.0, 1.0))))
        has_pixel_obs = bool(use_pixel_reprojection and ('pt_px' in obs) and (obs['pt_px'] is not None))
        ray_gate = ray_threshold_deg * (ray_soft_factor if has_pixel_obs else 1.0)
        if ray_err_deg > ray_gate:
            reproj_rejects += 1
            norm_rejects += 1
            continue
        obs_error_metric = float(norm_error)
        
        # NEW: Pixel-level reprojection validation (if K, D available)
        if has_pixel_obs:
            # Project 3D point to pixel coordinates
            pts_reproj = kannala_brandt_project(p_c_proj.reshape(1, 3), K, D)
            
            if pts_reproj.size > 0:
                pt_obs_px = np.array(obs['pt_px'], dtype=float).reshape(-1)
                if pt_obs_px.size >= 2:
                    pt_obs_px = pt_obs_px[:2]
                    pt_pred_px = np.array(pts_reproj[0], dtype=float).reshape(2,)
                    pixel_error = float(np.linalg.norm(pt_pred_px - pt_obs_px))
                    max_pixel_error = max(max_pixel_error, pixel_error)

                    pixel_soft = MAX_REPROJ_ERROR_PX * pixel_soft_factor
                    if pixel_error > pixel_soft:
                        reproj_rejects += 1
                        pixel_rejects += 1
                        continue

                    # Fail-soft: keep borderline observations but penalize quality.
                    obs_error_metric = min(pixel_error, pixel_soft) / max(1e-6, norm_scale_px)
                    if pixel_error > MAX_REPROJ_ERROR_PX:
                        obs_error_metric *= 1.15
                        borderline_rejects += 1

        strict_obs_metric = float(obs_error_metric)
        rescue_obs_metric = float(obs_error_metric)
        if promoted_by_depth_retry:
            depth_retry_promoted_obs += 1
            borderline_rejects += 1
            strict_obs_metric = float(strict_obs_metric) * float(
                depth_retry_soft_error_mult_use
            )
            rescue_obs_metric = float(strict_obs_metric)
            MSCKF_STATS['posttri_retry_recover_depth_soft_accept_count'] += 1
        if promoted_by_same_cycle_rescue:
            rescue_promoted_obs += 1
            rescue_borderline_rejects += 1
            rescue_obs_metric = float(rescue_obs_metric) * float(
                min(2.75, depth_retry_soft_error_mult_use * 1.12)
            )
            MSCKF_STATS['posttri_retry_recover_depth_soft_accept_count'] += 1
        else:
            total_error += float(strict_obs_metric)
            obs_error_norm_values.append(float(strict_obs_metric))
            valid_obs.append(obs)
        rescue_total_error += float(rescue_obs_metric)
        rescue_obs_error_norm_values.append(float(rescue_obs_metric))
        rescue_valid_obs.append(obs)

    if (
        use_depth_same_cycle_full_rescue
        and rescue_promoted_obs > 0
        and len(rescue_valid_obs) >= 2
        and len(rescue_valid_obs) > len(valid_obs)
    ):
        p_rescue = triangulate_point_nonlinear(
            rescue_valid_obs,
            cam_states,
            p_refined,
            kf,
            debug=debug,
            global_config=global_config,
        )
        if p_rescue is not None:
            p_refined = p_rescue
            valid_obs = list(rescue_valid_obs)
            obs_error_norm_values = list(rescue_obs_error_norm_values)
            total_error = float(rescue_total_error)
            borderline_rejects = int(borderline_rejects + rescue_borderline_rejects)
            depth_retry_promoted_obs = int(depth_retry_promoted_obs + rescue_promoted_obs)
            rescue_obs_set_used = True
            MSCKF_STATS['posttri_retry_recover_depth_borderline_promote_count'] += 1
        elif retry_mode_key == "depth_sparse_same_cycle_rescue" and len(valid_obs) < 2:
            # Same-cycle rescue fallback:
            # in retry-mode, allow bounded rescue observation set to proceed
            # even if re-triangulation fails, so the update lane can activate.
            valid_obs = list(rescue_valid_obs)
            obs_error_norm_values = list(rescue_obs_error_norm_values)
            total_error = float(rescue_total_error)
            borderline_rejects = int(borderline_rejects + rescue_borderline_rejects)
            depth_retry_promoted_obs = int(depth_retry_promoted_obs + rescue_promoted_obs)
            rescue_obs_set_used = True
            MSCKF_STATS['posttri_retry_recover_depth_borderline_promote_count'] += 1

    same_cycle_rescue_direct_lane_active = bool(
        bool(use_depth_same_cycle_full_rescue)
        and (retry_mode_key == "depth_sparse_same_cycle_rescue")
        and bool(rescue_obs_set_used)
        and int(len(valid_obs)) >= 2
        and int(depth_retry_promoted_obs + (0 if rescue_obs_set_used else rescue_promoted_obs)) > 0
    )
    promoted_obs_total = int(
        depth_retry_promoted_obs + (0 if rescue_obs_set_used else rescue_promoted_obs)
    )
    same_cycle_rescue_direct_max_promoted = int(
        max(1, int(depth_retry_soft_max_promote_use) + 1)
    )

    valid_obs_ratio_now = float(len(valid_obs)) / float(max(1, len(obs_list)))
    def _use_geometry_borderline_preagg(avg_gate: float, avg_err: Optional[float]) -> bool:
        # Deterministic pre-aggregation lane:
        # when residuals are heavy-tailed (spread high) but central tendency is good,
        # classify as geometry-borderline before hard reproj fail.
        MSCKF_STATS['preagg_precond_called_count'] += 1
        if avg_err is None or (not np.isfinite(avg_err)):
            MSCKF_STATS['preagg_precond_fail_invalid_input_count'] += 1
            MSCKF_STATS['preagg_precond_fail_avg_err_nan_count'] += 1
            return False
        if not np.isfinite(avg_gate):
            MSCKF_STATS['preagg_precond_fail_invalid_input_count'] += 1
            MSCKF_STATS['preagg_precond_fail_avg_gate_nan_count'] += 1
            return False
        if avg_gate <= 1e-9:
            MSCKF_STATS['preagg_precond_fail_invalid_input_count'] += 1
            MSCKF_STATS['preagg_precond_fail_avg_gate_nonpos_count'] += 1
            return False
        if float(avg_err) <= float(avg_gate):
            MSCKF_STATS['preagg_precond_fail_avg_err_le_gate_count'] += 1
            return False
        if float(avg_err) > float(avg_gate) * float(max(1.0, preagg_reproj_high_max_mult)):
            MSCKF_STATS['preagg_precond_fail_reproj_bound_count'] += 1
            return False

        err_arr = np.asarray(obs_error_norm_values, dtype=float)
        err_arr = err_arr[np.isfinite(err_arr)]
        if err_arr.size < max(4, int(preagg_min_track_obs) // 2):
            MSCKF_STATS['preagg_precond_fail_err_arr_too_small_count'] += 1
            return False

        MSCKF_STATS['preagg_precond_pass_count'] += 1
        MSCKF_STATS['preagg_predicate_eval_count'] += 1

        med = float(np.median(err_arr))
        p95 = float(np.percentile(err_arr, 95))
        spread_ratio = p95 / max(1e-9, med)
        median_good = bool(med <= (float(avg_gate) * 0.92))
        high_tail_gate_mult = 1.05
        high_tail_spread_min = 1.55
        high_tail = bool(
            p95 > (float(avg_gate) * float(high_tail_gate_mult))
            and spread_ratio >= float(high_tail_spread_min)
        )
        enough_signal = bool(
            float(valid_obs_ratio_now) >= 0.30
            and (borderline_rejects > 0 or norm_rejects > 0 or pixel_rejects > 0)
        )
        quality_ok = bool(
            (not np.isfinite(feature_quality))
            or float(feature_quality) >= float(np.clip(depth_gate_quality_th, 0.05, 1.0))
        )
        if median_good:
            MSCKF_STATS['preagg_predicate_median_good_count'] += 1
        if high_tail:
            MSCKF_STATS['preagg_predicate_high_tail_count'] += 1
        if enough_signal:
            MSCKF_STATS['preagg_predicate_enough_signal_count'] += 1
        if quality_ok:
            MSCKF_STATS['preagg_predicate_quality_ok_count'] += 1

        all_true = bool(median_good and high_tail and enough_signal and quality_ok)
        if all_true:
            MSCKF_STATS['preagg_predicate_all_true_count'] += 1
        return all_true

    if bool(depth_state_gate_enable) and bool(unstable_geometry_depth_gate):
        total_obs = max(1, int(len(obs_list)))
        valid_obs_n = int(len(valid_obs))
        depth_pos_ratio = float(valid_obs_n) / float(total_obs)
        depth_reject_ratio = float(depth_rejects) / float(max(1, depth_rejects + valid_obs_n))
        weak_depth_support = bool(
            valid_obs_n < max(2, int(depth_state_gate_min_valid_obs))
            or depth_pos_ratio < float(np.clip(depth_state_gate_min_pos_ratio, 0.05, 1.0))
            or (
                depth_reject_ratio > float(np.clip(depth_state_gate_min_reject_ratio, 0.0, 1.0))
                and valid_obs_n <= max(2, int(depth_state_gate_min_valid_obs))
            )
        )
        if weak_depth_support:
            # C2 hardening:
            # Do not early-exit sparse weak-depth cases here; route them to the
            # len(valid_obs)<2 lane so depth-sparse recovery can activate.
            # Keep immediate reject only for clearly unrecoverable dense cases.
            if valid_obs_n >= 2:
                same_cycle_direct_bypass_ok = bool(
                    same_cycle_rescue_direct_lane_active
                    and int(promoted_obs_total)
                    <= int(same_cycle_rescue_direct_max_promoted)
                )
                same_cycle_rescue_promote_ok = bool(
                    bool(use_depth_same_cycle_full_rescue)
                    and valid_obs_n >= 2
                    and int(promoted_obs_total) > 0
                    and depth_pos_ratio >= float(np.clip(float(depth_state_gate_min_pos_ratio) * 0.50, 0.10, 1.0))
                    and depth_reject_ratio <= float(np.clip(float(depth_state_gate_min_reject_ratio) * 1.35, 0.0, 0.99))
                    and (
                        (not np.isfinite(parallax_med_px))
                        or float(parallax_med_px) >= float(
                            max(0.1, float(depth_sparse_recover_min_parallax_px) * 0.75)
                        )
                    )
                )
                borderline_parallax_ok = bool(
                    (not np.isfinite(parallax_med_px))
                    or float(parallax_med_px) >= float(
                        max(
                            0.1,
                            float(depth_sparse_recover_min_parallax_px)
                            * float(np.clip(depth_retry_bounded_relax_parallax_mult, 0.5, 1.0)),
                        )
                    )
                )
                borderline_quality_ok = bool(
                    (not np.isfinite(feature_quality))
                    or float(feature_quality) >= float(
                        np.clip(
                            float(depth_sparse_recover_min_quality) - float(depth_retry_bounded_relax_quality_delta),
                            0.0,
                            1.0,
                        )
                    )
                )
                borderline_promote_ok = bool(
                    bool(use_depth_retry_bounded_relax)
                    and bool(protected_retry_context)
                    and valid_obs_n >= max(2, int(depth_state_gate_min_valid_obs) - 1)
                    and depth_pos_ratio >= float(np.clip(float(depth_state_gate_min_pos_ratio) * 0.60, 0.10, 1.0))
                    and depth_reject_ratio <= float(np.clip(float(depth_state_gate_min_reject_ratio) * 1.15, 0.0, 0.98))
                    and borderline_parallax_ok
                    and borderline_quality_ok
                )
                same_cycle_gate_override_ok = bool(
                    bool(depth_retry_gate_override_enable)
                    and bool(use_depth_same_cycle_full_rescue)
                    and (retry_mode_key == "depth_sparse_same_cycle_rescue")
                    and valid_obs_n >= 2
                    and int(promoted_obs_total) > 0
                    and int(promoted_obs_total)
                    <= max(1, int(depth_retry_soft_max_promote_use))
                    and depth_pos_ratio >= float(np.clip(float(depth_state_gate_min_pos_ratio) * 0.45, 0.10, 1.0))
                    and depth_reject_ratio <= float(np.clip(float(depth_state_gate_min_reject_ratio) * 1.50, 0.0, 0.99))
                    and (
                        (not np.isfinite(parallax_med_px))
                        or float(parallax_med_px) >= float(
                            max(0.1, float(depth_sparse_recover_min_parallax_px) * 0.65)
                        )
                    )
                )
                if same_cycle_direct_bypass_ok:
                    # Deterministic bounded bypass in same-cycle retry-mode:
                    # skip legacy unstable depth-gate so rescued observations
                    # can enter update path directly.
                    MSCKF_STATS['posttri_retry_recover_depth_gate_override_count'] += 1
                    MSCKF_STATS['posttri_retry_recover_depth_relaxed_gate_used_count'] += 1
                elif same_cycle_gate_override_ok:
                    MSCKF_STATS['posttri_retry_recover_depth_gate_override_count'] += 1
                    MSCKF_STATS['posttri_retry_recover_depth_relaxed_gate_used_count'] += 1
                elif rescue_obs_set_used and same_cycle_rescue_promote_ok:
                    MSCKF_STATS['posttri_retry_recover_depth_borderline_promote_count'] += 1
                    MSCKF_STATS['posttri_retry_recover_depth_relaxed_gate_used_count'] += 1
                elif borderline_promote_ok:
                    # Bounded dense-borderline promote for retry-depth lane only.
                    # Keep geometry alive long enough to reach the deterministic
                    # update/retry path instead of dying at the unstable depth gate.
                    MSCKF_STATS['posttri_retry_recover_depth_borderline_promote_count'] += 1
                    MSCKF_STATS['posttri_retry_recover_depth_relaxed_gate_used_count'] += 1
                else:
                    if (
                        bool(depth_retry_gate_override_enable)
                        and bool(use_depth_same_cycle_full_rescue)
                        and (retry_mode_key == "depth_sparse_same_cycle_rescue")
                    ):
                        MSCKF_STATS['posttri_retry_recover_depth_gate_override_reject_count'] += 1
                    if bool(depth_reclass_enable):
                        MSCKF_STATS['fail_geometry_borderline'] += 1
                        MSCKF_STATS['fail_depth_large'] += 1
                        MSCKF_STATS['reclass_to_geometry_count'] += 1
                    else:
                        MSCKF_STATS['fail_depth_sign_post_refine'] += 1
                        MSCKF_STATS['fail_depth_sign'] += 1
                    return None

    if len(valid_obs) < 2:
        total_rejects = max(1, int(depth_rejects + reproj_rejects))
        valid_obs_ratio = float(len(valid_obs)) / float(max(1, len(obs_list)))
        low_track_geometry = bool(len(obs_list) <= int(max(2, geometry_insuff_max_track_obs)))
        low_parallax_geometry = bool(
            np.isfinite(parallax_med_px)
            and float(parallax_med_px)
            < float(max(0.1, depth_gate_parallax_min_px) * float(max(0.5, geometry_insuff_parallax_mult)))
        )
        if bool(geometry_insuff_reclass_enable) and bool(unstable_geometry_depth_gate):
            if (
                valid_obs_ratio <= float(np.clip(geometry_insuff_max_valid_ratio, 0.0, 1.0))
                and (low_track_geometry or low_parallax_geometry)
            ):
                MSCKF_STATS['fail_geometry_insufficient_posttri'] += 1
                MSCKF_STATS['fail_geometry_borderline'] += 1
                MSCKF_STATS['reclass_to_geometry_count'] += 1
                return None
        if depth_rejects >= reproj_rejects:
            valid_obs_n = int(len(valid_obs))
            depth_dom_ratio = float(depth_rejects) / float(total_rejects)
            strict_depth_sign = bool(
                (not bool(unstable_geometry_depth_gate))
                and depth_dom_ratio >= float(np.clip(depth_sign_strict_dom_ratio_th, 0.5, 1.0))
                and valid_obs_ratio <= float(np.clip(depth_sign_strict_min_valid_ratio, 0.0, 1.0))
                and len(obs_list) >= max(2, int(depth_sign_strict_min_track_count))
                and (
                    (not np.isfinite(parallax_med_px))
                    or float(parallax_med_px) >= float(max(0.1, depth_sign_strict_min_parallax_px))
                )
                and (
                    (not np.isfinite(feature_quality))
                    or float(feature_quality) >= float(np.clip(depth_sign_strict_min_quality, 0.0, 1.0))
                )
            )
            parallax_recover_ok = bool(
                np.isfinite(parallax_med_px)
                and float(parallax_med_px) >= float(max(0.1, depth_sparse_recover_min_parallax_px))
            )
            parallax_low_for_depth_sparse = bool(
                np.isfinite(parallax_med_px)
                and float(parallax_med_px) < float(max(0.1, depth_sign_strict_min_parallax_px) * 1.10)
            )
            quality_recover_ok = bool(
                (not np.isfinite(feature_quality))
                or float(feature_quality) >= float(np.clip(depth_sparse_recover_min_quality, 0.0, 1.0))
            )
            if parallax_low_for_depth_sparse:
                MSCKF_STATS['depth_sparse_recover_parallax_low_count'] += 1
            strict_min_obs_recover = max(3, int(depth_sparse_recover_min_obs))
            min_obs_recover = int(strict_min_obs_recover)
            if bool(unstable_geometry_depth_gate) and parallax_low_for_depth_sparse:
                min_obs_recover = max(3, min_obs_recover - 1)
            relaxed_min_obs_recover = max(3, strict_min_obs_recover - max(0, int(depth_retry_bounded_relax_obs_delta)))
            strict_parallax_lane_ok = bool(
                parallax_recover_ok
                or (
                    bool(unstable_geometry_depth_gate)
                    and parallax_low_for_depth_sparse
                    and depth_dom_ratio >= float(np.clip(depth_sparse_recover_min_depth_dom_ratio, 0.0, 1.0))
                )
            )
            relaxed_parallax_lane_ok = bool(
                np.isfinite(parallax_med_px)
                and float(parallax_med_px) >= float(
                    max(
                        0.1,
                        float(depth_sparse_recover_min_parallax_px)
                        * float(np.clip(depth_retry_bounded_relax_parallax_mult, 0.5, 1.0)),
                    )
                )
            )
            relaxed_quality_recover_ok = bool(
                (not np.isfinite(feature_quality))
                or float(feature_quality) >= float(
                    np.clip(
                        float(depth_sparse_recover_min_quality) - float(depth_retry_bounded_relax_quality_delta),
                        0.0,
                        1.0,
                    )
                )
            )
            parallax_lane_ok = bool(
                strict_parallax_lane_ok
                or (
                    bool(use_depth_retry_bounded_relax)
                    and bool(relaxed_parallax_lane_ok)
                    and depth_dom_ratio >= float(np.clip(depth_sparse_recover_min_depth_dom_ratio, 0.0, 1.0))
                )
            )
            depth_sparse_candidate = bool(
                depth_sparse_recover_enable
                and len(obs_list) >= min_obs_recover
                and depth_dom_ratio >= float(np.clip(depth_sparse_recover_min_depth_dom_ratio, 0.0, 1.0))
                and parallax_lane_ok
                and (quality_recover_ok or (bool(use_depth_retry_bounded_relax) and bool(relaxed_quality_recover_ok)))
            )
            relaxed_depth_sparse_candidate = bool(
                bool(use_depth_retry_bounded_relax)
                and depth_sparse_recover_enable
                and len(obs_list) >= relaxed_min_obs_recover
                and depth_dom_ratio >= float(np.clip(depth_sparse_recover_min_depth_dom_ratio, 0.0, 1.0))
                and bool(relaxed_parallax_lane_ok)
                and bool(relaxed_quality_recover_ok)
            )
            if depth_sparse_candidate:
                MSCKF_STATS['depth_sparse_recover_candidate_count'] += 1
                if (not strict_parallax_lane_ok or len(obs_list) < strict_min_obs_recover or (not quality_recover_ok)) and relaxed_depth_sparse_candidate:
                    MSCKF_STATS['posttri_retry_recover_depth_relaxed_gate_used_count'] += 1
            recoverable_depth_sparse = bool(
                depth_sparse_candidate
                and (bool(unstable_geometry_depth_gate) or parallax_low_for_depth_sparse)
                and (
                    valid_obs_n >= 1
                    or (
                        bool(unstable_geometry_depth_gate)
                        and parallax_low_for_depth_sparse
                        and depth_rejects >= max(2, int(0.55 * len(obs_list)))
                    )
                )
            )
            if (not strict_depth_sign) and recoverable_depth_sparse:
                if bool(use_depth_same_cycle_full_rescue):
                    # Same-cycle full-rescue lane should not bounce back to
                    # fail_depth_sparse_recoverable. Keep it in geometry bucket
                    # so retry diagnostics can separate lane-entry from legacy depth-gate loops.
                    MSCKF_STATS['fail_geometry_insufficient_posttri'] += 1
                    MSCKF_STATS['fail_geometry_borderline'] += 1
                    MSCKF_STATS['reclass_to_geometry_count'] += 1
                    return None
                MSCKF_STATS['fail_depth_sparse_recoverable'] += 1
                if borderline_rejects > 0:
                    MSCKF_STATS['fail_geometry_borderline'] += 1
                return None
            if depth_sparse_candidate and (not recoverable_depth_sparse):
                MSCKF_STATS['depth_sparse_recover_gate_block_count'] += 1
            if (not strict_depth_sign) and bool(depth_reclass_enable):
                MSCKF_STATS['fail_geometry_borderline'] += 1
                MSCKF_STATS['fail_depth_large'] += 1
                MSCKF_STATS['reclass_to_geometry_count'] += 1
            else:
                MSCKF_STATS['fail_depth_sign_post_refine'] += 1
                MSCKF_STATS['fail_depth_sign'] += 1
        else:
            # Sparse post-tri path (len(valid_obs)<2): route recoverable reproj-sparse
            # cases to a retry lane before final rejection.
            depth_dom_ratio = float(depth_rejects) / float(total_rejects)
            reproj_dom_ratio = float(reproj_rejects) / float(total_rejects)
            parallax_recover_ok = bool(
                np.isfinite(parallax_med_px)
                and float(parallax_med_px) >= float(max(0.1, sparse_reproj_recover_min_parallax_px))
            )
            quality_recover_ok = bool(
                (not np.isfinite(feature_quality))
                or float(feature_quality) >= float(np.clip(sparse_reproj_recover_min_quality, 0.0, 1.0))
            )
            recoverable_sparse = bool(
                sparse_reproj_recover_enable
                and len(obs_list) >= max(3, int(sparse_reproj_recover_min_obs))
                and reproj_dom_ratio >= 0.50
                and depth_dom_ratio
                <= float(np.clip(sparse_reproj_recover_max_depth_dom_ratio, 0.0, 1.0))
                and parallax_recover_ok
                and quality_recover_ok
            )
            if recoverable_sparse:
                MSCKF_STATS['fail_reproj_sparse_recoverable'] += 1
                if borderline_rejects > 0:
                    MSCKF_STATS['fail_geometry_borderline'] += 1
            else:
                MSCKF_STATS['fail_reproj_sparse'] += 1
                if borderline_rejects > 0:
                    MSCKF_STATS['fail_geometry_borderline'] += 1
        return None

    avg_error_raw = float(total_error / len(valid_obs))
    avg_error = float(avg_error_raw)
    stable_lane_used = False
    stable_lane_cap = float("nan")
    stable_lane_borderline_ratio = float(borderline_rejects) / float(max(1, len(obs_list)))
    stable_lane_valid_obs_ratio = float(len(valid_obs)) / float(max(1, len(obs_list)))
    avg_reproj_gate = (avg_gate_factor * norm_threshold)
    stable_geometry_for_lane = bool(
        (not bool(unstable_geometry_depth_gate))
        and len(valid_obs) >= max(2, int(stable_lane_min_obs))
        and (
            (not np.isfinite(parallax_med_px))
            or float(parallax_med_px) >= float(max(0.1, depth_sign_strict_min_parallax_px))
        )
        and (
            (not np.isfinite(feature_quality))
            or float(feature_quality) >= float(np.clip(stable_lane_min_quality, 0.0, 1.0))
        )
    )
    # Deterministic stable lane (single path):
    # bounded per-feature residual capping with reduction floor.
    if bool(stable_lane_enable) and bool(stable_geometry_for_lane):
        err_arr = np.asarray(obs_error_norm_values, dtype=float)
        err_arr = err_arr[np.isfinite(err_arr)]
        pctl = float(np.clip(stable_lane_cap_percentile, 0.50, 0.99))
        cap_mult = float(np.clip(stable_lane_cap_max_mult, 1.0, 1.30))
        max_borderline = float(np.clip(stable_lane_max_borderline_ratio, 0.0, 1.0))
        min_valid_ratio = float(np.clip(stable_lane_min_valid_obs_ratio, 0.0, 1.0))
        max_reduction_lane = float(np.clip(stable_lane_max_reduction, 0.0, 0.30))
        if (
            err_arr.size >= max(4, int(stable_lane_min_obs))
            and stable_lane_borderline_ratio <= max_borderline
            and stable_lane_valid_obs_ratio >= min_valid_ratio
        ):
            per_feature_cap = float(np.quantile(err_arr, pctl))
            stable_lane_cap = float(
                min(
                    avg_reproj_gate * cap_mult,
                    max(norm_threshold, per_feature_cap),
                )
            )
            capped_arr = np.minimum(err_arr, stable_lane_cap)
            lane_candidate = float(np.mean(capped_arr))
            floor_error_lane = float(avg_error_raw * (1.0 - max_reduction_lane))
            lane_avg_error = float(min(avg_error_raw, max(floor_error_lane, lane_candidate)))
            if lane_avg_error < avg_error:
                avg_error = lane_avg_error
                stable_lane_used = True
                MSCKF_STATS['stable_lane_used_count'] += 1
    elif bool(unstable_lane_enable):
        # Deterministic unstable-geometry reproj lane:
        # apply conservative capped aggregation to reduce reject bursts without
        # globally relaxing reproj gates.
        err_arr = np.asarray(obs_error_norm_values, dtype=float)
        err_arr = err_arr[np.isfinite(err_arr)]
        pctl = float(np.clip(unstable_lane_cap_percentile, 0.50, 0.95))
        cap_mult = float(np.clip(unstable_lane_cap_max_mult, 1.0, 1.15))
        max_borderline = float(np.clip(unstable_lane_max_borderline_ratio, 0.0, 1.0))
        min_valid_ratio = float(np.clip(unstable_lane_min_valid_obs_ratio, 0.0, 1.0))
        max_reduction_lane = float(np.clip(unstable_lane_max_reduction, 0.0, 0.20))
        if err_arr.size >= max(4, int(unstable_lane_min_obs)):
            strict_lane_ok = bool(
                stable_lane_borderline_ratio <= max_borderline
                and stable_lane_valid_obs_ratio >= min_valid_ratio
            )
            if strict_lane_ok:
                reduction_scale = 1.0
                lane_cap_mult = cap_mult
                per_feature_cap = float(np.quantile(err_arr, pctl))
                lane_cap = float(
                    min(
                        avg_reproj_gate * lane_cap_mult,
                        max(norm_threshold, per_feature_cap),
                    )
                )
                capped_arr = np.minimum(err_arr, lane_cap)
                lane_candidate = float(np.mean(capped_arr))
                floor_error_lane = float(avg_error_raw * (1.0 - max_reduction_lane * reduction_scale))
                lane_avg_error = float(min(avg_error_raw, max(floor_error_lane, lane_candidate)))
                if lane_avg_error < avg_error:
                    avg_error = lane_avg_error

    reproj_failsoft_applied = False
    reproj_quality_scale = 1.0
    MSCKF_STATS['reproj_eval_attempt'] += 1
    if avg_error > avg_reproj_gate:
        unstable_reproj_reclass_enable = bool(
            (global_config or {}).get("MSCKF_UNSTABLE_REPROJ_RECLASSIFY_ENABLE", False)
            if isinstance(global_config, dict) else False
        )
        unstable_reproj_reclass_mult = float(
            (global_config or {}).get("MSCKF_UNSTABLE_REPROJ_RECLASSIFY_MULT", 1.0)
            if isinstance(global_config, dict) else 1.0
        )
        unstable_reproj_reclass_min_obs = int(
            (global_config or {}).get("MSCKF_UNSTABLE_REPROJ_RECLASSIFY_MIN_OBS", 4)
            if isinstance(global_config, dict) else 4
        )
        failsoft_enable = bool(
            (global_config or {}).get("MSCKF_REPROJ_FAILSOFT_ENABLE", True)
            if isinstance(global_config, dict) else True
        )
        failsoft_max_mult = float(
            (global_config or {}).get("MSCKF_REPROJ_FAILSOFT_MAX_MULT", 1.35)
            if isinstance(global_config, dict) else 1.35
        )
        failsoft_min_obs = int(
            (global_config or {}).get("MSCKF_REPROJ_FAILSOFT_MIN_OBS", 3)
            if isinstance(global_config, dict) else 3
        )
        failsoft_min_quality = float(
            (global_config or {}).get("MSCKF_REPROJ_FAILSOFT_MIN_QUALITY", 0.35)
            if isinstance(global_config, dict) else 0.35
        )
        failsoft_gate = avg_reproj_gate * max(1.0, failsoft_max_mult)
        allow_failsoft = bool(
            failsoft_enable
            and np.isfinite(avg_error)
            and avg_error <= failsoft_gate
            and len(valid_obs) >= max(2, failsoft_min_obs)
            and (
                (not np.isfinite(feature_quality))
                or (float(feature_quality) >= failsoft_min_quality)
                or (reproj_policy.get("quality_band", "mid") != "low")
            )
            and not (
                reproj_policy.get("quality_band", "mid") == "low"
                and bool(reproj_policy.get("low_quality_reject", True))
            )
        )
        if allow_failsoft:
            MSCKF_STATS['preagg_callsite_dense_failsoft_skip_count'] += 1
            reproj_failsoft_applied = True
            reproj_quality_scale = float(np.clip(avg_reproj_gate / max(avg_error, 1e-9), 0.35, 0.92))
            if borderline_rejects > 0 or reproj_policy.get("quality_band", "mid") != "high":
                MSCKF_STATS['fail_geometry_borderline'] += 1
            if debug:
                _log_msckf_update(
                    f"[MSCKF] reproj fail-soft: fid={fid}, avg={avg_error:.4f}, "
                    f"gate={avg_reproj_gate:.4f}, soft_gate={failsoft_gate:.4f}, "
                    f"q_scale={reproj_quality_scale:.2f}"
                )
        else:
            MSCKF_STATS['preagg_callsite_dense_candidate_count'] += 1
            valid_obs_ratio = float(len(valid_obs)) / float(max(1, len(obs_list)))
            low_track_geometry = bool(len(obs_list) <= int(max(2, geometry_insuff_max_track_obs)))
            low_parallax_geometry = bool(
                np.isfinite(parallax_med_px)
                and float(parallax_med_px)
                < float(max(0.1, depth_gate_parallax_min_px) * float(max(0.5, geometry_insuff_parallax_mult)))
            )
            # C2 one-knob: allow near-threshold reproj re-bucket when geometry is
            # unstable either by observability (track/parallax) or by low feature
            # quality. This keeps the update behavior ("no update") unchanged while
            # avoiding hard reproj over-counting for borderline unstable cases.
            quality_unstable_for_reclass = bool(
                np.isfinite(feature_quality)
                and float(feature_quality) < float(np.clip(depth_gate_quality_th, 0.05, 1.0))
            )
            unstable_near_threshold = bool(
                unstable_reproj_reclass_enable
                and len(valid_obs) >= max(2, unstable_reproj_reclass_min_obs)
                and np.isfinite(avg_error)
                and avg_error <= (avg_reproj_gate * max(1.0, unstable_reproj_reclass_mult))
                and (unstable_geometry_depth_gate or quality_unstable_for_reclass)
            )
            unstable_geometry_reclass = bool(
                unstable_geometry_depth_gate
                and bool(geometry_insuff_reclass_enable)
                and valid_obs_ratio <= float(np.clip(geometry_insuff_max_valid_ratio, 0.0, 1.0))
                and (low_track_geometry or low_parallax_geometry)
                and np.isfinite(avg_error)
                and avg_error <= (avg_reproj_gate * max(1.0, geometry_insuff_reclass_mult))
            )
            if unstable_near_threshold or unstable_geometry_reclass:
                MSCKF_STATS['preagg_callsite_dense_reclass_skip_count'] += 1
                # Deterministic unstable-lane reclassification:
                # keep "no update" behavior but classify near-threshold unstable
                # failures as borderline geometry, not hard reproj failures.
                MSCKF_STATS['fail_geometry_borderline'] += 1
                MSCKF_STATS['reclass_to_geometry_count'] += 1
                if unstable_geometry_reclass:
                    MSCKF_STATS['fail_geometry_insufficient_posttri'] += 1
                return None
            MSCKF_STATS['preagg_callsite_dense_invoked_count'] += 1
            if _use_geometry_borderline_preagg(
                avg_gate=float(avg_reproj_gate),
                avg_err=float(avg_error),
            ):
                MSCKF_STATS['fail_geometry_borderline'] += 1
                MSCKF_STATS['reclass_to_geometry_count'] += 1
                MSCKF_STATS['geometry_preagg_rebucket_count'] += 1
                return None
            MSCKF_STATS['fail_reproj_error'] += 1
            MSCKF_STATS['fail_reproj_normalized'] += 1
            if reproj_policy.get("quality_band", "mid") == "mid" or borderline_rejects > 0:
                MSCKF_STATS['fail_geometry_borderline'] += 1
            # [DIAGNOSTIC] Log reprojection failures to identify root cause
            if MSCKF_STATS['fail_reproj_error'] % 500 == 1 and debug:
                print(f"[MSCKF-DIAG] fail_reproj_error #{MSCKF_STATS['fail_reproj_error']}: fid={fid}, avg_norm_error={avg_error:.4f}, threshold={norm_threshold:.4f}")
                print(f"  norm_scale_px={norm_scale_px:.2f}")
            return None
    
    MSCKF_STATS['success'] += 1
    quality_base = 1.0 / (1.0 + avg_error * 100.0)
    result = {
        'p_w': p_refined,
        'observations': valid_obs,
        'quality': float(quality_base * reproj_quality_scale),
        'avg_reproj_error': avg_error,
        'reproj_p95_norm': float(np.percentile(obs_error_norm_values, 95)) if len(obs_error_norm_values) > 0 else np.nan,
        'depth_positive_ratio': float(len(valid_obs) / max(1, len(obs_list))),
        'parallax_med_px': float(parallax_med_px),
        'num_obs_used': len(valid_obs),
        'num_obs_total': len(obs_list),
        'quality_band': str(reproj_policy.get("quality_band", "mid")),
        'feature_quality': float(feature_quality) if np.isfinite(feature_quality) else np.nan,
        'reproj_failsoft_applied': bool(reproj_failsoft_applied),
        'reproj_quality_scale': float(reproj_quality_scale),
        'stable_reproj_lane_used': bool(stable_lane_used),
        'stable_reproj_lane_cap': float(stable_lane_cap) if np.isfinite(stable_lane_cap) else np.nan,
        'stable_reproj_lane_borderline_ratio': float(stable_lane_borderline_ratio),
        'stable_reproj_lane_valid_obs_ratio': float(stable_lane_valid_obs_ratio),
        'retry_mode_key': str(retry_mode_key),
        'rescue_obs_set_used': bool(rescue_obs_set_used),
        'rescue_promoted_obs_total': int(promoted_obs_total),
        'rescue_promoted_obs_cap': int(same_cycle_rescue_direct_max_promoted),
        'same_cycle_rescue_direct_lane_active': bool(same_cycle_rescue_direct_lane_active),
    }
    
    # Add pixel-level error if available
    if use_pixel_reprojection and max_pixel_error > 0:
        result['max_pixel_reproj_error'] = max_pixel_error
    
    return result


def compute_huber_weights(normalized_residuals: np.ndarray, 
                          threshold: float = 1.345) -> np.ndarray:
    """Compute Huber robust loss weights."""
    r_abs = np.abs(normalized_residuals)
    weights = np.ones_like(normalized_residuals)
    
    outlier_mask = r_abs > threshold
    weights[outlier_mask] = threshold / r_abs[outlier_mask]
    
    return weights


def compute_measurement_jacobian(p_w: np.ndarray, cam_state: dict, 
                                 kf: ExtendedKalmanFilter,
                                 err_state_size: int, 
                                 use_preint_jacobians: bool = True,
                                 T_cam_imu: np.ndarray = None,
                                 global_config: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute measurement Jacobian for one camera observation.
    
    Args:
        p_w: 3D point in world frame
        cam_state: Camera state metadata
        kf: EKF with full state
        err_state_size: Error state dimension
        use_preint_jacobians: Add IMU preintegration terms
        T_cam_imu: Camera-to-IMU extrinsics
        global_config: Global config dict (for loading BODY_T_CAMDOWN from YAML)
    
    Returns: (H_cam, H_feat)
    """
    if T_cam_imu is None:
        T_cam_imu = _get_active_body_to_camera_transform(global_config)
    R_BC, _, _, t_CB = _decompose_body_to_camera_transform(T_cam_imu)
    
    # Get IMU pose (FEJ if available)
    if 'q_fej' in cam_state and 'p_fej' in cam_state:
        q_imu = cam_state['q_fej']
        p_imu = cam_state['p_fej']
    else:
        q_idx = cam_state['q_idx']
        p_idx = cam_state['p_idx']
        q_imu = kf.x[q_idx:q_idx+4, 0]
        p_imu = kf.x[p_idx:p_idx+3, 0]
    
    # Transform to camera pose
    q_imu_xyzw = np.array([q_imu[1], q_imu[2], q_imu[3], q_imu[0]])
    R_BW = R_scipy.from_quat(q_imu_xyzw).as_matrix()
    R_CW = R_BW @ R_BC.T
    R_WC = R_CW.T
    p_cam = p_imu + R_BW @ t_CB
    
    # Transform point to camera frame
    p_rel = p_w - p_cam
    p_c = R_WC @ p_rel
    
    # Projection Jacobian
    inv_z = 1.0 / p_c[2]
    inv_z2 = inv_z * inv_z
    
    j_proj = np.array([
        [inv_z, 0, -p_c[0] * inv_z2],
        [0, inv_z, -p_c[1] * inv_z2]
    ])
    
    # Jacobian w.r.t. feature
    h_feat = j_proj @ R_WC
    
    # Jacobian w.r.t. error state
    h_cam = np.zeros((2, err_state_size))
    
    err_theta_idx = cam_state['err_q_idx']
    err_p_idx = cam_state['err_p_idx']
    
    # Position Jacobian
    h_cam[:, err_p_idx:err_p_idx+3] = j_proj @ (-R_WC)
    
    # Rotation Jacobian (body error-state): p_c = R_BC * R_BW^T * (p_w - p_imu) + t_BC
    p_wi = p_w - p_imu
    p_b = R_BW.T @ p_wi
    j_rot = j_proj @ (-R_BC @ skew_symmetric(p_b))
    h_cam[:, err_theta_idx:err_theta_idx+3] = j_rot
    
    # Preintegration Jacobians (bias coupling)
    # Try preint object first, then stored jacobians
    J_R_bg = J_p_bg = J_p_ba = None
    if use_preint_jacobians:
        if 'preint' in cam_state and cam_state['preint'] is not None:
            preint = cam_state['preint']
            J_R_bg, _, _, J_p_bg, J_p_ba = preint.get_jacobians()
        elif 'J_R_bg' in cam_state and cam_state['J_R_bg'] is not None:
            # Use stored Jacobians from clone time
            J_R_bg = cam_state['J_R_bg']
            J_p_bg = cam_state['J_p_bg']
            J_p_ba = cam_state['J_p_ba']
    
    # Apply bias Jacobians only if valid 3x3 matrices
    if J_R_bg is not None and hasattr(J_R_bg, 'shape') and J_R_bg.shape == (3, 3):
        
        R_clone = R_BW
        
        # Gyro bias
        h_cam[:, 9:12] += j_rot @ J_R_bg
        j_pos = j_proj @ (-R_WC)
        h_cam[:, 9:12] += j_pos @ R_clone @ J_p_bg
        
        # Accel bias
        h_cam[:, 12:15] += j_pos @ R_clone @ J_p_ba
    
    return h_cam, h_feat


def compute_observability_nullspace(kf: ExtendedKalmanFilter, 
                                   num_clones: int) -> np.ndarray:
    """
    Compute nullspace basis for unobservable directions.
    
    Unobservable modes: global X, Y position and global yaw.
    
    Args:
        kf: EKF with current state
        num_clones: Number of camera clones
    
    Returns: (err_state_size, 3) nullspace matrix
    """
    err_state_size = 18 + 6 * num_clones  # v3.9.7: 18 core + 6*clones
    
    U = np.zeros((err_state_size, 3), dtype=float)
    
    # Global X translation
    U[0, 0] = 1.0
    for i in range(num_clones):
        clone_p_idx = 18 + 6*i + 3  # v3.9.7: 18 core error dim
        U[clone_p_idx, 0] = 1.0
    
    # Global Y translation
    U[1, 1] = 1.0
    for i in range(num_clones):
        clone_p_idx = 18 + 6*i + 3  # v3.9.7: 18 core error dim
        U[clone_p_idx + 1, 1] = 1.0
    
    # Global yaw rotation
    U[8, 2] = 1.0
    
    p_imu = kf.x[0:3, 0]
    U[0, 2] = -p_imu[1]
    U[1, 2] = p_imu[0]
    
    core_size = 19  # v3.9.7: nominal state with mag_bias
    for i in range(num_clones):
        clone_p_idx = core_size + 7*i + 4
        p_cam = kf.x[clone_p_idx:clone_p_idx+3, 0]
        
        err_theta_idx = 18 + 6*i  # v3.9.7: 18 core error dim
        err_p_idx = 18 + 6*i + 3
        
        U[err_theta_idx + 2, 2] = 1.0
        U[err_p_idx, 2] = -p_cam[1]
        U[err_p_idx + 1, 2] = p_cam[0]
    
    # Orthonormalize
    U_ortho = np.zeros_like(U)
    for j in range(3):
        u_j = U[:, j].copy()
        for k in range(j):
            u_j -= np.dot(U_ortho[:, k], u_j) * U_ortho[:, k]
        norm = np.linalg.norm(u_j)
        if norm > 1e-10:
            U_ortho[:, j] = u_j / norm
    
    return U_ortho


def msckf_measurement_update(fid: int, triangulated: dict, cam_observations: List[dict],
                             cam_states: List[dict], kf: ExtendedKalmanFilter,
                             measurement_noise: float = 1e-4,
                             huber_threshold: float = 1.345,
                             chi2_max_dof: float = 15.36,
                             chi2_scale: float = 1.0,
                             global_config: dict = None) -> Tuple[bool, float, float]:
    """
    MSCKF measurement update with observability constraints.
    
    Args:
        fid: Feature ID
        triangulated: Triangulation result
        cam_observations: Observation records
        cam_states: Camera state metadata
        kf: EKF to update
        measurement_noise: Measurement noise variance
        huber_threshold: Huber loss threshold
        chi2_max_dof: Max chi-square per DoF
    
    Returns: (success, innovation_norm, chi2_test)
    """
    # CRITICAL: Check P matrix validity BEFORE any computation (Step 1: catch explosion early)
    if not np.all(np.isfinite(kf.P)):
        print(f"[MSCKF] CRITICAL: P matrix contains inf/nan, skipping MSCKF update")
        return (False, np.nan, np.nan)
    
    # Check for large P values that will cause overflow
    max_p_val = np.max(np.abs(kf.P))
    if max_p_val > 1e10:
        print(f"[MSCKF] WARNING: P matrix has very large values (max={max_p_val:.2e}), may overflow")
        kf.P = ensure_covariance_valid(
            kf.P,
            label="MSCKF-entry",
            max_value=getattr(kf, "covariance_max_value", 1e8),
            symmetrize=True,
            check_psd=True,
            conditioner=kf,
            timestamp=float("nan"),
            stage="MSCKF_ENTRY",
        )
    
    p_w = triangulated['p_w']
    obs_list = triangulated['observations']
    
    if len(obs_list) < 2:
        return (False, np.nan, np.nan)
    
    err_state_size = kf.P.shape[0]
    
    # Compute residuals and Jacobians
    residuals = []
    h_x_stack = []
    h_f_stack = []
    
    for obs in obs_list:
        cam_id = obs['cam_id']
        if cam_id >= len(cam_states):
            continue
        
        cs = cam_states[cam_id]
        
        # Predicted measurement
        q_imu = kf.x[cs['q_idx']:cs['q_idx']+4, 0]
        p_imu = kf.x[cs['p_idx']:cs['p_idx']+3, 0]
        q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu, global_config=global_config)
        
        q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
        r_cw = R_scipy.from_quat(q_xyzw).as_matrix()
        r_wc = r_cw.T
        p_c = r_wc @ (p_w - p_cam)
        
        if p_c[2] <= 0.1:
            continue
        
        z_pred = np.array([p_c[0] / p_c[2], p_c[1] / p_c[2]])
        z_obs = np.array([obs['pt_norm'][0], obs['pt_norm'][1]])
        
        r = z_obs - z_pred
        residuals.append(r)
        
        h_cam, h_feat = compute_measurement_jacobian(p_w, cs, kf, err_state_size, 
                                                      global_config=global_config)
        h_x_stack.append(h_cam)
        h_f_stack.append(h_feat)
    
    if len(residuals) < 2:
        return (False, np.nan, np.nan)
    
    r_o = np.vstack(residuals).reshape(-1, 1)
    h_x = np.vstack(h_x_stack)
    h_f = np.vstack(h_f_stack)
    
    # DEBUG: Check H matrix bias columns before any projection (disabled for performance)
    # h_bias_raw = np.linalg.norm(h_x[:, 9:15])
    # if h_bias_raw > 1e-10:
    #     print(f"[MSCKF-BIAS-RAW] H[:,9:15] norm={h_bias_raw:.6f}")
    # else:
    #     print(f"[MSCKF-BIAS-RAW] H[:,9:15] is ZERO!")
    
    # ROOT CAUSE FIX: Check Jacobian magnitude BEFORE any matmul
    # Large Jacobians indicate bad feature (bad depth/parallax) → overflow in matmul
    h_max = np.max(np.abs(h_x))
    if h_max > 1e6:  # Threshold for reasonable Jacobian magnitude
        return (False, np.nan, np.nan)
    
    # EARLY EXIT: Validate input matrices before any computation
    if not np.all(np.isfinite(h_x)) or not np.all(np.isfinite(h_f)) or not np.all(np.isfinite(r_o)):
        return (False, np.nan, np.nan)
    
    # Nullspace projection
    try:
        u_mat, s_mat, vh_mat = np.linalg.svd(h_f, full_matrices=True)
        tol = 1e-6 * s_mat[0] if len(s_mat) > 0 else 1e-6
        rank = np.sum(s_mat > tol)
        null_space = u_mat[:, rank:]
        
        # Check null_space validity before matmul
        if not np.all(np.isfinite(null_space)):
            h_proj = h_x
            r_proj = r_o
        else:
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                h_proj = null_space.T @ h_x
                r_proj = null_space.T @ r_o
            # Validate result
            if not np.all(np.isfinite(h_proj)) or not np.all(np.isfinite(r_proj)):
                h_proj = h_x
                r_proj = r_o
    except np.linalg.LinAlgError:
        h_proj = h_x
        r_proj = r_o
    
    # Observability constraint
    num_clones = (err_state_size - 18) // 6  # v3.9.7: 18 core error dim
    
    try:
        U_obs = compute_observability_nullspace(kf, num_clones)
        
        # EARLY EXIT: Validate U_obs before matmul
        if not np.all(np.isfinite(U_obs)):
            h_constrained = h_proj
            r_constrained = r_proj
        else:
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                projection_matrix = np.eye(err_state_size) - U_obs @ U_obs.T
            
            # Check projection matrix validity before matmul
            if not np.all(np.isfinite(projection_matrix)):
                h_constrained = h_proj
                r_constrained = r_proj
            else:
                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                    h_constrained = h_proj @ projection_matrix
                r_constrained = r_proj
                
                # Validate result
                if not np.all(np.isfinite(h_constrained)):
                    h_constrained = h_proj
                    r_constrained = r_proj
        
        # DEBUG: Check if bias Jacobians are non-zero before/after projection (disabled for performance)
        # h_bias_before = np.linalg.norm(h_proj[:, 9:15])
        # h_bias_after = np.linalg.norm(h_constrained[:, 9:15])
        # if h_bias_before > 1e-10:
        #     print(f"[MSCKF-BIAS] H_bias before={h_bias_before:.6f}, after={h_bias_after:.6f}, ratio={h_bias_after/h_bias_before:.3f}")
    except (np.linalg.LinAlgError, ValueError):
        h_constrained = h_proj
        r_constrained = r_proj
    
    # Huber weighting
    measurement_std = np.sqrt(measurement_noise)
    r_normalized = r_constrained / measurement_std
    weights = compute_huber_weights(r_normalized.flatten(), threshold=huber_threshold)
    weight_matrix = np.diag(np.sqrt(weights))
    
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        h_weighted = weight_matrix @ h_constrained
        r_weighted = weight_matrix @ r_constrained
    
    # Check for inf/nan after Huber weighting
    if not np.all(np.isfinite(h_weighted)) or not np.all(np.isfinite(r_weighted)):
        return (False, np.nan, np.nan)
    
    meas_dim = r_weighted.shape[0]
    r_cov_original = np.eye(meas_dim) * measurement_noise
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        r_cov = weight_matrix @ r_cov_original @ weight_matrix.T
    
    if not np.all(np.isfinite(r_cov)):
        return (False, np.nan, np.nan)
    
    # CRITICAL: Check P matrix validity before S = H @ P @ H.T
    if not np.all(np.isfinite(kf.P)):
        print(f"[MSCKF] CRITICAL: P matrix corrupted before S computation, aborting")
        return (False, np.nan, np.nan)
    
    # Innovation and chi-square gating
    try:
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            s_mat = h_weighted @ kf.P @ h_weighted.T + r_cov
    except (FloatingPointError, RuntimeWarning) as e:
        print(f"[MSCKF] WARNING: Matmul overflow in S matrix: {e}")
        return (False, np.nan, np.nan)
    
    if not np.all(np.isfinite(s_mat)):
        print(f"[MSCKF] WARNING: S matrix contains inf/nan after computation")
        return (False, np.nan, np.nan)
    
    innovation_norm = float(np.linalg.norm(r_weighted))
    
    try:
        s_inv = np.linalg.inv(s_mat)
        chi2_test = float(r_weighted.T @ s_inv @ r_weighted)
        
        dof = meas_dim
        chi2_threshold = max(1e-6, float(chi2_scale) * chi2_max_dof * dof)
        
        if chi2_test > chi2_threshold:
            return (False, innovation_norm, chi2_test)
    except np.linalg.LinAlgError:
        return (False, innovation_norm, np.nan)
    
    # EKF update
    try:
        # Check S_inv before using it
        if not np.all(np.isfinite(s_inv)):
            print(f"[MSCKF] WARNING: S_inv contains inf/nan")
            return (False, innovation_norm, np.nan)
        
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            k_gain = kf.P @ h_weighted.T @ s_inv
        
        if not np.all(np.isfinite(k_gain)):
            print(f"[MSCKF] WARNING: Kalman gain contains inf/nan")
            return (False, innovation_norm, np.nan)
        
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            delta_x = k_gain @ r_weighted
        
        # DEBUG: Check bias correction magnitude (disabled for performance)
        # dbg = delta_x[9:12, 0]
        # dba = delta_x[12:15, 0]
        # dbg_norm = np.linalg.norm(dbg)
        # dba_norm = np.linalg.norm(dba)
        # if dbg_norm > 1e-10 or dba_norm > 1e-10:
        #     print(f"[MSCKF-BIAS-UPDATE] δb_g={dbg_norm:.2e} rad/s, δb_a={dba_norm:.2e} m/s²")
        
        kf._apply_error_state_correction(delta_x)
        
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            i_kh = np.eye(err_state_size) - k_gain @ h_weighted
        
        if not np.all(np.isfinite(i_kh)):
            print(f"[MSCKF] WARNING: (I - KH) matrix contains inf/nan")
            return (False, innovation_norm, np.nan)
        
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            kf.P = i_kh @ kf.P @ i_kh.T + k_gain @ r_cov @ k_gain.T
        
        kf.P = ensure_covariance_valid(
            kf.P,
            label="MSCKF-Update",
            symmetrize=True,
            check_psd=True,
            max_value=getattr(kf, "covariance_max_value", 1e8),
            conditioner=kf,
            timestamp=float("nan"),
            stage="MSCKF_UPDATE",
        )
        
        kf.x_post = kf.x.copy()
        kf.P_post = kf.P.copy()
        if hasattr(kf, "log_cov_health"):
            kf.log_cov_health(update_type="MSCKF", timestamp=float('nan'), stage="post_update")
        
        return (True, innovation_norm, chi2_test)
    except (np.linalg.LinAlgError, ValueError):
        return (False, innovation_norm, np.nan)


def msckf_measurement_update_with_plane(fid: int, triangulated: dict, 
                                       cam_observations: List[dict],
                                       cam_states: List[dict], 
                                       kf: ExtendedKalmanFilter,
                                       plane,
                                       plane_config: dict,
                                       chi2_scale: float = 1.0,
                                       global_config: dict = None) -> Tuple[bool, float, float]:
    """
    MSCKF measurement update with stacked plane constraint.
    
    Stacked measurement model (following OpenVINS ov_plane):
        [z_bearing]   [H_bearing  ]   [0]
        [z_plane  ] = [H_plane    ] + [ε]
    
    where:
        - z_bearing: 2D bearing measurements (standard MSCKF)
        - z_plane: 1D point-on-plane constraint (n^T * p + d = 0)
    
    Args:
        fid: Feature ID
        triangulated: Triangulation result with 'p_w' (3D point)
        cam_observations: All observations
        cam_states: Camera states
        kf: EKF
        plane: Detected plane (with normal n and distance d)
        plane_config: Plane configuration dict
    
    Returns: (success, innovation_norm, chi2_test)
    """
    # CRITICAL: Check P matrix validity BEFORE any computation
    if not np.all(np.isfinite(kf.P)):
        print(f"[MSCKF-PLANE] CRITICAL: P matrix contains inf/nan, skipping")
        return (False, np.nan, np.nan)
    
    # Check for large P values that will cause overflow
    max_p_val = np.max(np.abs(kf.P))
    if max_p_val > 1e10:
        print(f"[MSCKF-PLANE] WARNING: P matrix has very large values (max={max_p_val:.2e})")
        kf.P = ensure_covariance_valid(
            kf.P,
            label="MSCKF-Plane-entry",
            max_value=getattr(kf, "covariance_max_value", 1e8),
            symmetrize=True,
            check_psd=True,
            conditioner=kf,
            timestamp=float("nan"),
            stage="MSCKF_PLANE_ENTRY",
        )
    # =========================================================================
    # Part 1: Standard MSCKF bearing measurements
    # =========================================================================
    point_world = triangulated['p_w']
    obs_list = triangulated['observations']
    
    if len(obs_list) < 2:
        return (False, 0.0, 0.0)
    
    # Build bearing Jacobian (same as standard MSCKF)
    # Dynamically compute error state size from actual covariance dimensions
    # This handles cases where state has additional elements (e.g., SLAM features)
    err_state_size = kf.P.shape[0]
    meas_dim_bearing = 2 * len(obs_list)
    
    h_bearing = np.zeros((meas_dim_bearing, err_state_size))
    r_bearing = np.zeros(meas_dim_bearing)
    
    for i, obs_data in enumerate(obs_list):
        cam_id = obs_data['cam_id']
        if cam_id >= len(cam_states):
            continue
        
        cs = cam_states[cam_id]
        q_imu = kf.x[cs['q_idx']:cs['q_idx']+4, 0]
        p_imu = kf.x[cs['p_idx']:cs['p_idx']+3, 0]
        
        q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu, global_config=global_config)
        q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
        R_cw = R_scipy.from_quat(q_xyzw).as_matrix()
        R_wc = R_cw.T
        
        p_c = R_wc @ (point_world - p_cam)
        
        if p_c[2] < 0.1:
            return (False, 0.0, 0.0)
        
        # Bearing measurement
        xn_pred = p_c[0] / p_c[2]
        yn_pred = p_c[1] / p_c[2]
        
        xn_obs, yn_obs = obs_data['pt_norm']
        
        r_bearing[2*i] = xn_obs - xn_pred
        r_bearing[2*i+1] = yn_obs - yn_pred
        
        # Jacobian (standard MSCKF)
        z_inv = 1.0 / p_c[2]
        z_inv2 = z_inv * z_inv
        
        J_proj = np.array([
            [z_inv, 0, -p_c[0]*z_inv2],
            [0, z_inv, -p_c[1]*z_inv2]
        ])
        
        J_q = -J_proj @ R_cw @ skew_symmetric(p_c)
        J_p = -J_proj @ R_wc
        
        clone_idx_err = 18 + 6 * cam_id  # v3.9.7: 18 core error dim
        h_bearing[2*i:2*i+2, clone_idx_err:clone_idx_err+3] = J_q
        h_bearing[2*i:2*i+2, clone_idx_err+3:clone_idx_err+6] = J_p
    
    # =========================================================================
    # Part 2: Plane constraint measurement
    # =========================================================================
    # Measurement: z_plane = n^T * p + d = 0 (ideal)
    # Residual: r_plane = n^T * p_w + d
    
    residual_plane = plane.point_distance(point_world)
    
    # Jacobian w.r.t. point (not used here, but for reference)
    # H_point = n^T (1x3)
    
    # Jacobian w.r.t. camera states (through triangulated point)
    # Rigorous computation: ∂(n^T*p + d)/∂cam_states using triangulation sensitivity
    # Reference: OpenVINS ov_plane, Section III.B
    
    h_plane = np.zeros((1, err_state_size))
    n = plane.normal
    
    # Compute ∂p_w/∂cam_states analytically from triangulation equations
    # For two-view triangulation: p_w = c0 + λ0*r0 where λ0 = f(poses, bearings)
    # Chain rule: ∂p_w/∂cam = ∂p_w/∂λ0 * ∂λ0/∂cam + ∂p_w/∂c0 * ∂c0/∂cam
    
    # Get first two views for triangulation sensitivity
    if len(obs_list) >= 2:
        obs0 = obs_list[0]
        obs1 = obs_list[1]
        cam_id0 = obs0['cam_id']
        cam_id1 = obs1['cam_id']
        
        if cam_id0 < len(cam_states) and cam_id1 < len(cam_states):
            cs0 = cam_states[cam_id0]
            cs1 = cam_states[cam_id1]
            
            # Camera poses
            q_imu0 = kf.x[cs0['q_idx']:cs0['q_idx']+4, 0]
            p_imu0 = kf.x[cs0['p_idx']:cs0['p_idx']+3, 0]
            q0, c0 = imu_pose_to_camera_pose(q_imu0, p_imu0, global_config=global_config)
            
            q_imu1 = kf.x[cs1['q_idx']:cs1['q_idx']+4, 0]
            p_imu1 = kf.x[cs1['p_idx']:cs1['p_idx']+3, 0]
            q1, c1 = imu_pose_to_camera_pose(q_imu1, p_imu1, global_config=global_config)
            
            # Ray directions in world frame
            q0_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]])
            R0_wc = R_scipy.from_quat(q0_xyzw).as_matrix().T
            x0, y0 = obs0['pt_norm']
            ray0_c = normalized_to_unit_ray(x0, y0)
            r0 = R0_wc @ ray0_c
            r0 = r0 / np.linalg.norm(r0)
            
            q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]])
            R1_wc = R_scipy.from_quat(q1_xyzw).as_matrix().T
            x1, y1 = obs1['pt_norm']
            ray1_c = normalized_to_unit_ray(x1, y1)
            r1 = R1_wc @ ray1_c
            r1 = r1 / np.linalg.norm(r1)
            
            # Triangulation geometry: p_w = c0 + λ0*r0 = c1 + λ1*r1
            # Solving for λ0, λ1 using least-squares (midpoint method)
            w = c0 - c1
            a = np.dot(r0, r0)
            b = np.dot(r0, r1)
            c_dot = np.dot(r1, r1)
            d = np.dot(r0, w)
            e = np.dot(r1, w)
            denom = a * c_dot - b * b
            
            if abs(denom) > 1e-6:
                λ0 = (b * e - c_dot * d) / denom
                
                # ∂λ0/∂c0: derivative of depth w.r.t. first camera position
                # From λ0 = (b*e - c*d) / denom where d = r0^T*w, e = r1^T*w, w = c0 - c1
                # ∂λ0/∂c0 = ∂/∂c0[(b*(r1^T*w) - c*(r0^T*w)) / denom]
                #          = (b*r1 - c*r0) / denom
                dλ0_dc0 = (b * r1 - c_dot * r0) / denom
                
                # ∂λ0/∂c1 = -∂λ0/∂c0 (since w = c0 - c1)
                dλ0_dc1 = -dλ0_dc0
                
                # ∂p_w/∂c0 = I + r0 ⊗ dλ0/∂c0 (direct position + depth sensitivity)
                # ∂p_w/∂c1 = r0 ⊗ dλ0/∂c1 (only depth sensitivity)
                dp_dc0 = np.eye(3) + np.outer(r0, dλ0_dc0)
                dp_dc1 = np.outer(r0, dλ0_dc1)
                
                # ∂p_w/∂θ0: derivative w.r.t. first camera orientation
                # Rotation affects ray direction: r0 = R0_wc @ ray0_c
                # ∂r0/∂θ0 = -R0_wc @ [ray0_c]×
                # Then ∂p_w/∂θ0 = λ0 * ∂r0/∂θ0
                dr0_dθ0 = -R0_wc @ skew_symmetric(ray0_c)
                dp_dθ0 = λ0 * dr0_dθ0
                
                # Similar for second camera
                # ∂λ0/∂θ1 comes from ∂r1/∂θ1 affecting dot products
                # For simplicity, use numerical approximation or neglect (small)
                dp_dθ1 = np.zeros((3, 3))  # Approximation: first-order only
                
                # ∂(n^T*p_w + d)/∂cam_states = n^T * ∂p_w/∂cam_states
                clone_idx0 = 18 + 6 * cam_id0  # v3.9.7: 18 core error dim
                clone_idx1 = 18 + 6 * cam_id1
                
                h_plane[0, clone_idx0:clone_idx0+3] = n @ dp_dθ0
                h_plane[0, clone_idx0+3:clone_idx0+6] = n @ dp_dc0
                
                h_plane[0, clone_idx1:clone_idx1+3] = n @ dp_dθ1
                h_plane[0, clone_idx1+3:clone_idx1+6] = n @ dp_dc1
            else:
                # Fallback: use simplified approximation if geometry is degenerate
                for i, obs_data in enumerate(obs_list):
                    cam_id = obs_data['cam_id']
                    if cam_id >= len(cam_states):
                        continue
                    
                    cs = cam_states[cam_id]
                    q_imu = kf.x[cs['q_idx']:cs['q_idx']+4, 0]
                    p_imu = kf.x[cs['p_idx']:cs['p_idx']+3, 0]
                    
                    q_cam, p_cam = imu_pose_to_camera_pose(q_imu, p_imu, global_config=global_config)
                    q_xyzw = np.array([q_cam[1], q_cam[2], q_cam[3], q_cam[0]])
                    R_cw = R_scipy.from_quat(q_xyzw).as_matrix()
                    R_wc = R_cw.T
                    
                    p_c = R_wc @ (point_world - p_cam)
                    
                    clone_idx_err = 18 + 6 * cam_id  # v3.9.7: 18 core error dim
                    J_p_theta = -R_wc @ skew_symmetric(p_c)
                    J_p_pos = -R_wc
                    
                    h_plane[0, clone_idx_err:clone_idx_err+3] += n @ J_p_theta / len(obs_list)
                    h_plane[0, clone_idx_err+3:clone_idx_err+6] += n @ J_p_pos / len(obs_list)
    
    # =========================================================================
    # Part 3: Stack measurements and update
    # =========================================================================
    meas_dim_total = meas_dim_bearing + 1
    h_stacked = np.vstack([h_bearing, h_plane])
    # Ensure r_stacked is 2D column vector (same as standard MSCKF)
    r_stacked = np.concatenate([r_bearing, [residual_plane]]).reshape(-1, 1)
    
    # EARLY EXIT: Validate stacked matrices before any computation
    if not np.all(np.isfinite(h_stacked)) or not np.all(np.isfinite(r_stacked)):
        return (False, np.nan, np.nan)
    
    # Measurement noise
    sigma_bearing = 1e-4  # Standard MSCKF bearing noise
    sigma_plane = plane_config.get('PLANE_SIGMA', 0.05)
    
    R_stacked = np.eye(meas_dim_total)
    R_stacked[:meas_dim_bearing, :meas_dim_bearing] *= sigma_bearing
    R_stacked[-1, -1] = sigma_plane ** 2
    
    # Observability constraint (same as standard MSCKF)
    U_nullspace = compute_observability_nullspace(kf, len(cam_states))
    
    if U_nullspace is not None and U_nullspace.shape[0] == h_stacked.shape[1]:
        # Check U_nullspace validity before matmul
        if not np.all(np.isfinite(U_nullspace)):
            h_weighted = h_stacked
            r_weighted = r_stacked
        else:
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                A_proj = np.eye(h_stacked.shape[1]) - U_nullspace @ U_nullspace.T
            
            # Check projection matrix validity before matmul
            if not np.all(np.isfinite(A_proj)):
                print(f"[MSCKF-PLANE] WARNING: Projection matrix contains inf/nan")
                h_weighted = h_stacked
                r_weighted = r_stacked
            else:
                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                    h_weighted = h_stacked @ A_proj
                r_weighted = r_stacked
                
                # Check h_weighted validity after matmul
                if not np.all(np.isfinite(h_weighted)):
                    return (False, np.nan, np.nan)
    else:
        h_weighted = h_stacked
        r_weighted = r_stacked
    
    # CRITICAL: Check P matrix validity before S = H @ P @ H.T
    if not np.all(np.isfinite(kf.P)):
        print(f"[MSCKF-PLANE] CRITICAL: P matrix corrupted before S computation")
        return (False, np.nan, np.nan)
    
    # Chi-square gating
    try:
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            s_mat = h_weighted @ kf.P @ h_weighted.T + R_stacked
    except (FloatingPointError, RuntimeWarning) as e:
        print(f"[MSCKF-PLANE] WARNING: Matmul overflow in S matrix: {e}")
        return (False, np.nan, np.nan)
    
    if not np.all(np.isfinite(s_mat)):
        print(f"[MSCKF-PLANE] WARNING: S matrix contains inf/nan")
        return (False, np.nan, np.nan)
    
    innovation_norm = float(np.linalg.norm(r_weighted))
    
    try:
        s_inv = safe_matrix_inverse(s_mat, damping=1e-9, method='cholesky')
        chi2_test = float(r_weighted.T @ s_inv @ r_weighted)
        
        chi2_threshold = max(1e-6, float(chi2_scale) * 15.36 * meas_dim_total)
        
        if chi2_test > chi2_threshold:
            return (False, innovation_norm, chi2_test)
    except np.linalg.LinAlgError:
        return (False, innovation_norm, np.nan)
    
    # EKF update
    try:
        # Check S_inv before using it
        if not np.all(np.isfinite(s_inv)):
            print(f"[MSCKF-PLANE] WARNING: S_inv contains inf/nan")
            return (False, innovation_norm, np.nan)
        
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            k_gain = kf.P @ h_weighted.T @ s_inv
        
        if not np.all(np.isfinite(k_gain)):
            print(f"[MSCKF-PLANE] WARNING: Kalman gain contains inf/nan")
            return (False, innovation_norm, np.nan)
        
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            delta_x = k_gain @ r_weighted
        
        kf._apply_error_state_correction(delta_x)
        
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            i_kh = np.eye(err_state_size) - k_gain @ h_weighted
        
        if not np.all(np.isfinite(i_kh)):
            print(f"[MSCKF-PLANE] WARNING: (I - KH) matrix contains inf/nan")
            return (False, innovation_norm, np.nan)
        
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            kf.P = i_kh @ kf.P @ i_kh.T + k_gain @ R_stacked @ k_gain.T
        
        kf.P = ensure_covariance_valid(
            kf.P,
            label="MSCKF-Plane-Update",
            symmetrize=True,
            check_psd=True,
            max_value=getattr(kf, "covariance_max_value", 1e8),
            conditioner=kf,
            timestamp=float("nan"),
            stage="MSCKF_PLANE_UPDATE",
        )
        
        kf.x_post = kf.x.copy()
        kf.P_post = kf.P.copy()
        if hasattr(kf, "log_cov_health"):
            kf.log_cov_health(update_type="MSCKF_PLANE", timestamp=float('nan'), stage="post_update")
        
        return (True, innovation_norm, chi2_test)
    except (np.linalg.LinAlgError, ValueError):
        return (False, innovation_norm, np.nan)


def perform_msckf_updates(vio_fe, cam_observations: List[dict],
                          cam_states: List[dict], kf: ExtendedKalmanFilter,
                          min_observations: int = 3, max_features: int = 50,
                          msckf_dbg_path: str = None,
                          dem_reader = None,
                          origin_lat: float = 0.0, origin_lon: float = 0.0,
                          plane_detector = None,
                          plane_config: dict = None,
                          global_config: dict = None,
                          chi2_scale: float = 1.0,
                          reproj_scale: float = 1.0,
                          phase: int = 2,
                          health_state: str = "HEALTHY",
                          stats_out: Optional[Dict[str, float]] = None,
                          quality_out: Optional[Dict[str, float]] = None,
                          timestamp: float = float("nan")) -> int:
    """
    Perform MSCKF updates for mature features.
    
    Optionally uses plane-aided MSCKF with stacked measurements:
    - Standard bearing measurements (2D per observation)
    - Point-on-plane constraint (1D per plane-associated feature)
    
    Args:
        vio_fe: VIO frontend
        cam_observations: Observation records
        cam_states: Camera state metadata
        kf: EKF to update
        min_observations: Minimum observations per feature
        max_features: Maximum features to process
        msckf_dbg_path: Debug log path
        dem_reader: DEM reader for ground constraint
        origin_lat/origin_lon: Local projection origin
        plane_detector: PlaneDetector for plane-aided MSCKF (optional)
        plane_config: Plane configuration dict (optional)
    
    Returns: Number of successful updates
    """
    if vio_fe is None or len(cam_states) < 2:
        return 0
    
    mature_fids = find_mature_features_for_msckf(
        vio_fe,
        cam_observations,
        min_observations,
        global_config=global_config,
        max_features=max_features,
    )
    
    num_successful = 0
    num_attempted = 0
    dof_samples: List[int] = []
    chi2_norm_samples: List[float] = []
    parallax_samples: List[float] = []
    reproj_p95_samples: List[float] = []
    depth_ratio_samples: List[float] = []
    feature_quality_samples: List[float] = []
    runtime_verbosity = (
        str(global_config.get("LOG_RUNTIME_VERBOSITY", "debug")).lower()
        if isinstance(global_config, dict)
        else "debug"
    )
    runtime_quiet = runtime_verbosity in ("release", "quiet", "minimal")
    prefilter_cfg = global_config if isinstance(global_config, dict) else {}
    prefilter_enable = bool(prefilter_cfg.get("MSCKF_L2_PREFILTER_ENABLE", False))
    prefilter_min_obs = int(prefilter_cfg.get("MSCKF_L2_PREFILTER_MIN_OBS", max(3, int(min_observations))))
    prefilter_min_parallax_px = float(prefilter_cfg.get("MSCKF_L2_PREFILTER_MIN_PARALLAX_PX", 1.15))
    prefilter_min_time_span_sec = float(prefilter_cfg.get("MSCKF_L2_PREFILTER_MIN_TIME_SPAN_SEC", 0.085))
    prefilter_min_quality = float(prefilter_cfg.get("MSCKF_L2_PREFILTER_MIN_QUALITY", 0.32))
    retry_lane_enable = bool(prefilter_cfg.get("MSCKF_RETRY_LANE_ENABLE", True))
    retry_lane_max_cycles = int(max(0, prefilter_cfg.get("MSCKF_RETRY_LANE_MAX_CYCLES", 2)))
    retry_lane_sparse_track_max_obs = int(
        max(2, prefilter_cfg.get("MSCKF_RETRY_LANE_SPARSE_TRACK_MAX_OBS", max(4, int(min_observations) + 1)))
    )
    retry_lane_min_parallax_px = float(
        prefilter_cfg.get("MSCKF_RETRY_LANE_MIN_PARALLAX_PX", prefilter_min_parallax_px)
    )
    retry_lane_min_time_span_sec = float(
        prefilter_cfg.get("MSCKF_RETRY_LANE_MIN_TIME_SPAN_SEC", prefilter_min_time_span_sec)
    )
    posttri_retry_defer_enable = bool(
        prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_DEFER_ENABLE", True)
    )
    posttri_retry_max_cycles = int(
        max(
            0,
            prefilter_cfg.get(
                "MSCKF_RETRY_LANE_POSTTRI_MAX_CYCLES",
                min(1, int(retry_lane_max_cycles)),
            ),
        )
    )
    posttri_retry_recover_enable = bool(
        prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_ENABLE", True)
    )
    posttri_retry_recover_max_cycles = int(
        max(
            0,
            prefilter_cfg.get(
                "MSCKF_RETRY_LANE_POSTTRI_RECOVER_MAX_CYCLES",
                max(1, int(posttri_retry_max_cycles) + 1),
            ),
        )
    )
    posttri_retry_recover_min_obs = int(
        max(2, prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_MIN_OBS", 5))
    )
    posttri_retry_recover_min_parallax_px = float(
        prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_MIN_PARALLAX_PX", 1.15)
    )
    posttri_retry_recover_min_quality = float(
        prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_MIN_QUALITY", 0.34)
    )
    posttri_retry_recover_depth_same_cycle_enable = bool(
        prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_SAME_CYCLE_ENABLE", False)
    )
    posttri_retry_recover_depth_full_rescue_enable = bool(
        prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_FULL_RESCUE_ENABLE", False)
    )
    posttri_retry_recover_protect_enable = bool(
        prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_PROTECT_ENABLE", False)
    )
    posttri_retry_recover_protect_cycles = int(
        max(0, prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_PROTECT_CYCLES", 2))
    )
    posttri_retry_recover_depth_bounded_relax_enable = bool(
        prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_BOUNDED_RELAX_ENABLE", False)
    )
    posttri_retry_recover_depth_bounded_relax_obs_delta = int(
        max(0, prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_BOUNDED_RELAX_OBS_DELTA", 1))
    )
    posttri_retry_recover_depth_bounded_relax_parallax_mult = float(
        prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_BOUNDED_RELAX_PARALLAX_MULT", 0.88)
    )
    posttri_retry_recover_depth_bounded_relax_quality_delta = float(
        prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_BOUNDED_RELAX_QUALITY_DELTA", 0.03)
    )
    posttri_retry_recover_depth_gate_enable = bool(
        prefilter_cfg.get("MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_GATE_ENABLE", True)
    )
    # Deterministic caps for retry-mode same-cycle depth rescue.
    posttri_retry_same_cycle_max_promoted_obs = int(
        np.clip(
            prefilter_cfg.get(
                "MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_SAME_CYCLE_MAX_PROMOTED_OBS",
                3,
            ),
            1,
            8,
        )
    )
    posttri_retry_same_cycle_max_avg_reproj_norm = float(
        np.clip(
            prefilter_cfg.get(
                "MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_SAME_CYCLE_MAX_AVG_REPROJ_NORM",
                0.06,
            ),
            0.005,
            0.30,
        )
    )
    posttri_retry_same_cycle_min_depth_ratio = float(
        np.clip(
            prefilter_cfg.get(
                "MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_SAME_CYCLE_MIN_DEPTH_RATIO",
                0.30,
            ),
            0.05,
            1.00,
        )
    )
    posttri_retry_same_cycle_chi2_scale_mult = float(
        np.clip(
            prefilter_cfg.get(
                "MSCKF_RETRY_LANE_POSTTRI_RECOVER_DEPTH_SAME_CYCLE_CHI2_SCALE_MULT",
                0.85,
            ),
            0.2,
            1.0,
        )
    )
    retry_state: Dict[int, int] = {}
    posttri_retry_state: Dict[int, int] = {}
    posttri_retry_source: Dict[int, str] = {}
    posttri_retry_protect_state: Dict[int, int] = {}
    if vio_fe is not None:
        try:
            retry_state = dict(getattr(vio_fe, "_msckf_retry_state", {}))
        except Exception:
            retry_state = {}
        try:
            posttri_retry_state = dict(getattr(vio_fe, "_msckf_posttri_retry_state", {}))
        except Exception:
            posttri_retry_state = {}
        try:
            posttri_retry_source = dict(getattr(vio_fe, "_msckf_posttri_retry_source", {}))
        except Exception:
            posttri_retry_source = {}
        try:
            posttri_retry_protect_state = dict(getattr(vio_fe, "_msckf_posttri_retry_protect_state", {}))
        except Exception:
            posttri_retry_protect_state = {}
    if posttri_retry_protect_state:
        posttri_retry_protect_state = {
            int(fid): int(cnt)
            for fid, cnt in posttri_retry_protect_state.items()
            if int(cnt) > 0
        }
    valid_fid_set = {int(fid) for fid in mature_fids}
    if posttri_retry_protect_state:
        valid_fid_set.update(int(fid) for fid in posttri_retry_protect_state.keys())
    if retry_state:
        retry_state = {
            int(fid): int(cnt)
            for fid, cnt in retry_state.items()
            if int(fid) in valid_fid_set
        }
    if posttri_retry_state:
        posttri_retry_state = {
            int(fid): int(cnt)
            for fid, cnt in posttri_retry_state.items()
            if int(fid) in valid_fid_set
        }
    if posttri_retry_source:
        posttri_retry_source = {
            int(fid): str(src)
            for fid, src in posttri_retry_source.items()
            if int(fid) in valid_fid_set
        }

    def _record_depth_retry_failure(reason: str) -> None:
        reason = str(reason or "")
        if reason in (
            "fail_depth_sign_init",
            "fail_depth_sign_post_refine",
            "fail_depth_sign",
            "fail_depth_large",
            "fail_depth_sparse_recoverable",
        ):
            MSCKF_STATS['posttri_retry_recover_depth_fail_depth_count'] += 1
        elif reason in (
            "fail_reproj_pixel",
            "fail_reproj_normalized",
            "fail_reproj_error",
            "fail_reproj_sparse",
            "fail_reproj_sparse_recoverable",
        ):
            MSCKF_STATS['posttri_retry_recover_depth_fail_reproj_count'] += 1
        elif reason in (
            "fail_prefilter_geometry",
            "fail_geometry_insufficient_pretri",
            "fail_geometry_insufficient_posttri",
            "fail_geometry_borderline",
            "fail_parallax",
            "fail_baseline",
            "fail_solver",
        ):
            MSCKF_STATS['posttri_retry_recover_depth_fail_geometry_count'] += 1
        elif reason == "fail_nonlinear":
            MSCKF_STATS['posttri_retry_recover_depth_fail_nonlinear_count'] += 1
        else:
            MSCKF_STATS['posttri_retry_recover_depth_fail_other_count'] += 1

    def _record_same_cycle_retry_failure(reason: str) -> None:
        reason = str(reason or "")
        if reason in (
            "fail_depth_sign_init",
            "fail_depth_sign_post_refine",
            "fail_depth_sign",
            "fail_depth_large",
            "fail_depth_sparse_recoverable",
        ):
            MSCKF_STATS['posttri_retry_recover_depth_same_cycle_fail_depth_count'] += 1
        elif reason in (
            "fail_reproj_pixel",
            "fail_reproj_normalized",
            "fail_reproj_error",
            "fail_reproj_sparse",
            "fail_reproj_sparse_recoverable",
        ):
            MSCKF_STATS['posttri_retry_recover_depth_same_cycle_fail_reproj_count'] += 1
        elif reason in (
            "fail_prefilter_geometry",
            "fail_geometry_insufficient_pretri",
            "fail_geometry_insufficient_posttri",
            "fail_geometry_borderline",
            "fail_parallax",
            "fail_baseline",
            "fail_solver",
        ):
            MSCKF_STATS['posttri_retry_recover_depth_same_cycle_fail_geometry_count'] += 1
        elif reason == "fail_nonlinear":
            MSCKF_STATS['posttri_retry_recover_depth_same_cycle_fail_nonlinear_count'] += 1
        else:
            MSCKF_STATS['posttri_retry_recover_depth_same_cycle_fail_other_count'] += 1

    def _depth_recover_gate_ok(quick_obs_local: List[dict], quick_parallax_local: float, q_med_local: float) -> Tuple[bool, bool]:
        strict_depth_recover_gate_ok = bool(
            len(quick_obs_local) >= int(max(2, posttri_retry_recover_min_obs))
            and np.isfinite(quick_parallax_local)
            and float(quick_parallax_local) >= float(max(0.1, posttri_retry_recover_min_parallax_px))
            and (
                (not np.isfinite(q_med_local))
                or float(q_med_local) >= float(np.clip(posttri_retry_recover_min_quality, 0.0, 1.0))
            )
        )
        relaxed_depth_recover_gate_ok = bool(
            bool(posttri_retry_recover_depth_bounded_relax_enable)
            and len(quick_obs_local) >= int(max(2, posttri_retry_recover_min_obs - posttri_retry_recover_depth_bounded_relax_obs_delta))
            and np.isfinite(quick_parallax_local)
            and float(quick_parallax_local) >= float(
                max(
                    0.1,
                    posttri_retry_recover_min_parallax_px
                    * np.clip(posttri_retry_recover_depth_bounded_relax_parallax_mult, 0.5, 1.0),
                )
            )
            and (
                (not np.isfinite(q_med_local))
                or float(q_med_local) >= float(
                    np.clip(
                        posttri_retry_recover_min_quality - posttri_retry_recover_depth_bounded_relax_quality_delta,
                        0.0,
                        1.0,
                    )
                )
            )
        )
        return bool(strict_depth_recover_gate_ok or relaxed_depth_recover_gate_ok), bool(
            (not strict_depth_recover_gate_ok) and relaxed_depth_recover_gate_ok
        )

    protected_retry_carried_fids = set()
    if posttri_retry_recover_protect_enable and posttri_retry_protect_state:
        protected_fids = []
        cleaned_protect_state = {}
        for fid, cycles_left in sorted(posttri_retry_protect_state.items()):
            if cycles_left <= 0:
                continue
            src = str(posttri_retry_source.get(int(fid), "")).strip().lower()
            if src != "depth_sparse":
                MSCKF_STATS['posttri_retry_recover_depth_protected_source_missing_count'] += 1
                posttri_retry_source.pop(int(fid), None)
                continue
            obs = get_feature_multi_view_observations(fid, cam_observations)
            valid_clone_obs = [
                o for o in obs
                if 0 <= int(o.get("cam_id", -1)) < len(cam_states)
            ]
            if len(valid_clone_obs) >= max(2, int(min_observations)):
                protected_fids.append(int(fid))
                protected_retry_carried_fids.add(int(fid))
                cleaned_protect_state[int(fid)] = int(cycles_left)
            else:
                MSCKF_STATS['posttri_retry_recover_depth_protected_missing_clone_count'] += 1
                posttri_retry_source.pop(int(fid), None)
        posttri_retry_protect_state = cleaned_protect_state
        if protected_fids:
            merged = []
            seen = set()
            for fid in protected_fids + mature_fids:
                fid_i = int(fid)
                if fid_i in seen:
                    continue
                seen.add(fid_i)
                merged.append(fid_i)
            mature_fids = merged

    if len(mature_fids) == 0:
        return 0

    if len(mature_fids) > max_features:
        truncated_fids = set(int(fid) for fid in mature_fids[max_features:])
        truncated_protected = truncated_fids & protected_retry_carried_fids
        if truncated_protected:
            MSCKF_STATS['posttri_retry_recover_depth_protected_truncated_count'] += int(len(truncated_protected))
        mature_fids = mature_fids[:max_features]

    # =========================================================================
    # Plane Detection (if enabled)
    # =========================================================================
    detected_planes = []
    triangulated_points = {}
    
    if plane_detector is not None and plane_config is not None:
        # First pass: triangulate all mature features to build point cloud
        for fid in mature_fids:
            tri_result = triangulate_feature(fid, cam_observations, cam_states, kf,
                                            use_plane_constraint=True, ground_altitude=0.0,
                                            debug=False,
                                            dem_reader=dem_reader,
                                            origin_lat=origin_lat, origin_lon=origin_lon,
                                            reproj_scale=reproj_scale,
                                            phase=phase,
                                            health_state=health_state)
            if tri_result is not None:
                triangulated_points[fid] = tri_result['p_w']
        
        # Detect planes from triangulated point cloud
        if len(triangulated_points) >= 10:
            points_array = np.array(list(triangulated_points.values()))
            try:
                detected_planes = plane_detector.detect_planes(points_array)
                if len(detected_planes) > 0:
                    if not runtime_quiet:
                        print(
                            f"[MSCKF-PLANE] Detected {len(detected_planes)} planes from {len(points_array)} points"
                        )
            except Exception as e:
                print(f"[MSCKF-PLANE] Plane detection failed: {e}")
    
    # =========================================================================
    # MSCKF Updates with Optional Plane Constraints
    # =========================================================================
    for i, fid in enumerate(mature_fids):
        quick_obs = get_feature_multi_view_observations(fid, cam_observations)
        protected_retry_carried = bool(int(fid) in protected_retry_carried_fids)
        if len(quick_obs) == 0:
            if protected_retry_carried:
                MSCKF_STATS['posttri_retry_recover_depth_protected_missing_clone_count'] += 1
            retry_state.pop(int(fid), None)
            posttri_retry_state.pop(int(fid), None)
            posttri_retry_source.pop(int(fid), None)
            posttri_retry_protect_state.pop(int(fid), None)
            continue
        if protected_retry_carried:
            valid_clone_obs = sum(
                1 for o in quick_obs if 0 <= int(o.get("cam_id", -1)) < len(cam_states)
            )
            if valid_clone_obs < max(2, int(min_observations)):
                MSCKF_STATS['posttri_retry_recover_depth_protected_missing_clone_count'] += 1
                retry_state.pop(int(fid), None)
                posttri_retry_state.pop(int(fid), None)
                posttri_retry_source.pop(int(fid), None)
                posttri_retry_protect_state.pop(int(fid), None)
                continue
            MSCKF_STATS['posttri_retry_recover_depth_protected_carried_count'] += 1
        quick_parallax = _pairwise_parallax_med_px(quick_obs, 120.0)
        t_vals = [float(o.get("t", np.nan)) for o in quick_obs]
        t_arr = np.asarray(t_vals, dtype=float)
        q_vals = [float(o.get("quality", np.nan)) for o in quick_obs]
        q_arr = np.asarray(q_vals, dtype=float)
        q_med = float(np.nanmedian(q_arr)) if (q_arr.size > 0 and np.isfinite(q_arr).any()) else float("nan")
        time_span_sec = (
            float(np.nanmax(t_arr) - np.nanmin(t_arr))
            if (t_arr.size > 1 and np.isfinite(t_arr).any())
            else float("nan")
        )
        sparse_short_track = bool(len(quick_obs) <= int(retry_lane_sparse_track_max_obs))
        sparse_low_parallax = bool(
            np.isfinite(quick_parallax)
            and float(quick_parallax) < float(max(0.1, retry_lane_min_parallax_px))
        )
        sparse_short_span = bool(
            (not np.isfinite(time_span_sec))
            or float(time_span_sec) < float(max(1e-3, retry_lane_min_time_span_sec))
        )
        in_posttri_recover_retry = bool(int(posttri_retry_state.get(int(fid), 0)) > 0)
        depth_sparse_retry_source = bool(
            str(posttri_retry_source.get(int(fid), "")).strip().lower() == "depth_sparse"
        )
        in_depth_sparse_retry_path = bool(
            depth_sparse_retry_source and (in_posttri_recover_retry or protected_retry_carried)
        )
        if retry_lane_enable and retry_lane_max_cycles > 0:
            if (not in_depth_sparse_retry_path) and sparse_short_track and (sparse_low_parallax or sparse_short_span):
                cur_retry = int(retry_state.get(int(fid), 0))
                if cur_retry < int(retry_lane_max_cycles):
                    retry_state[int(fid)] = cur_retry + 1
                    MSCKF_STATS['retry_lane_defer_count'] += 1
                    continue
            retry_state.pop(int(fid), None)
        if in_depth_sparse_retry_path:
            MSCKF_STATS['posttri_retry_recover_depth_retry_seen_count'] += 1

        # L2 (logic-first): prefilter weak geometry before triangulation so depth-sign
        # failures are concentrated to truly stable/strict cases only.
        if prefilter_enable and (not in_posttri_recover_retry):
            if len(quick_obs) < max(2, prefilter_min_obs):
                MSCKF_STATS['fail_prefilter_geometry'] += 1
                continue
            weak_short_track = bool(
                len(quick_obs) <= max(3, prefilter_min_obs)
                and (
                    (not np.isfinite(time_span_sec))
                    or time_span_sec < float(max(1e-3, prefilter_min_time_span_sec))
                )
                and (
                    np.isfinite(quick_parallax)
                    and float(quick_parallax) < float(prefilter_min_parallax_px * 1.25)
                )
            )
            weak_low_quality = bool(
                np.isfinite(q_med)
                and float(q_med) < float(np.clip(prefilter_min_quality, 0.0, 1.0))
                and np.isfinite(quick_parallax)
                and float(quick_parallax) < float(prefilter_min_parallax_px * 1.6)
            )
            weak_low_parallax = bool(
                np.isfinite(quick_parallax)
                and float(quick_parallax) < float(max(0.1, prefilter_min_parallax_px))
                and (
                    (not np.isfinite(time_span_sec))
                    or time_span_sec < float(max(1e-3, prefilter_min_time_span_sec * 1.6))
                )
            )
            if weak_short_track or weak_low_quality or weak_low_parallax:
                MSCKF_STATS['fail_prefilter_geometry'] += 1
                continue

        stats_before = dict(MSCKF_STATS)
        enable_debug = (i < 3)
        retry_mode = ""
        if int(posttri_retry_state.get(int(fid), 0)) > 0:
            retry_mode = str(posttri_retry_source.get(int(fid), "")).strip().lower()
            if retry_mode == "depth_sparse":
                retry_mode = "depth_sparse_recover"
            elif retry_mode == "reproj_sparse":
                retry_mode = "reproj_sparse_recover"
        retry_mode_same_cycle_rescue_active = bool(
            bool(posttri_retry_recover_depth_same_cycle_enable)
            and bool(posttri_retry_recover_depth_full_rescue_enable)
            and bool(in_posttri_recover_retry)
            and bool(depth_sparse_retry_source)
            and len(quick_obs) >= 2
            and bool(protected_retry_carried)
        )
        if retry_mode_same_cycle_rescue_active:
            retry_mode = "depth_sparse_same_cycle_rescue"
            MSCKF_STATS['posttri_retry_recover_depth_same_cycle_entered_count'] += 1
            MSCKF_STATS['posttri_retry_recover_depth_same_cycle_attempt_count'] += 1
        triangulated = triangulate_feature(fid, cam_observations, cam_states, kf, 
                                          use_plane_constraint=True, ground_altitude=0.0,
                                          debug=enable_debug,
                                          dem_reader=dem_reader,
                                          origin_lat=origin_lat, origin_lon=origin_lon,
                                          global_config=global_config,
                                          reproj_scale=reproj_scale,
                                          phase=phase,
                                          health_state=health_state,
                                          retry_mode=retry_mode)
        
        if triangulated is None:
            fail_reason = "triangulation_failed"
            for key in MSCKF_FAIL_REASON_PRIORITY:
                if int(MSCKF_STATS.get(key, 0)) > int(stats_before.get(key, 0)):
                    fail_reason = key
                    break
            if in_posttri_recover_retry and str(posttri_retry_source.get(int(fid), "")) == "depth_sparse":
                _record_depth_retry_failure(fail_reason)
                if bool(retry_mode_same_cycle_rescue_active):
                    _record_same_cycle_retry_failure(fail_reason)
            if protected_retry_carried and fail_reason == "fail_depth_sparse_recoverable":
                MSCKF_STATS['posttri_retry_recover_depth_protected_depth_gate_again_count'] += 1
            if (
                triangulated is None
                and (not in_posttri_recover_retry)
                and bool(posttri_retry_recover_enable)
                and bool(posttri_retry_recover_depth_same_cycle_enable)
                and fail_reason == "fail_depth_sparse_recoverable"
            ):
                depth_recover_gate_ok, used_relaxed_gate = _depth_recover_gate_ok(
                    quick_obs_local=quick_obs,
                    quick_parallax_local=quick_parallax,
                    q_med_local=q_med,
                )
                if depth_recover_gate_ok and used_relaxed_gate:
                    MSCKF_STATS['posttri_retry_recover_depth_relaxed_gate_used_count'] += 1
                if depth_recover_gate_ok:
                    MSCKF_STATS['posttri_retry_recover_depth_same_cycle_entered_count'] += 1
                    MSCKF_STATS['posttri_retry_recover_depth_same_cycle_attempt_count'] += 1
                    retry_stats_before = dict(MSCKF_STATS)
                    retry_mode_same_cycle = "depth_sparse_recover"
                    if (
                        bool(posttri_retry_recover_depth_full_rescue_enable)
                        and bool(protected_retry_carried)
                        and len(quick_obs) >= 2
                    ):
                        retry_mode_same_cycle = "depth_sparse_same_cycle_rescue"
                    triangulated_retry = triangulate_feature(
                        fid,
                        cam_observations,
                        cam_states,
                        kf,
                        use_plane_constraint=True,
                        ground_altitude=0.0,
                        debug=enable_debug,
                        dem_reader=dem_reader,
                        origin_lat=origin_lat,
                        origin_lon=origin_lon,
                        global_config=global_config,
                        reproj_scale=reproj_scale,
                        phase=phase,
                        health_state=health_state,
                        retry_mode=retry_mode_same_cycle,
                    )
                    if triangulated_retry is not None:
                        triangulated = triangulated_retry
                        MSCKF_STATS['posttri_retry_recover_depth_same_cycle_success_count'] += 1
                    else:
                        fail_reason_retry = "triangulation_failed"
                        for key in MSCKF_FAIL_REASON_PRIORITY:
                            if int(MSCKF_STATS.get(key, 0)) > int(retry_stats_before.get(key, 0)):
                                fail_reason_retry = key
                                break
                        if protected_retry_carried and fail_reason_retry == "fail_depth_sparse_recoverable":
                            MSCKF_STATS['posttri_retry_recover_depth_protected_depth_gate_again_count'] += 1
                        _record_same_cycle_retry_failure(fail_reason_retry)
                        _record_depth_retry_failure(fail_reason_retry)
            if (
                triangulated is None
                and
                posttri_retry_defer_enable
                and posttri_retry_recover_enable
                and posttri_retry_recover_max_cycles > 0
                and fail_reason in ("fail_reproj_sparse_recoverable", "fail_depth_sparse_recoverable")
            ):
                if fail_reason == "fail_depth_sparse_recoverable" and posttri_retry_recover_depth_gate_enable:
                    depth_recover_gate_ok, used_relaxed_gate = _depth_recover_gate_ok(
                        quick_obs_local=quick_obs,
                        quick_parallax_local=quick_parallax,
                        q_med_local=q_med,
                    )
                    if depth_recover_gate_ok and used_relaxed_gate:
                        MSCKF_STATS['posttri_retry_recover_depth_relaxed_gate_used_count'] += 1
                    if not depth_recover_gate_ok:
                        MSCKF_STATS['posttri_retry_recover_depth_gate_reject_count'] += 1
                        if protected_retry_carried:
                            MSCKF_STATS['posttri_retry_recover_depth_protected_depth_gate_again_count'] += 1
                        posttri_retry_source.pop(int(fid), None)
                        posttri_retry_state.pop(int(fid), None)
                        posttri_retry_protect_state.pop(int(fid), None)
                        if msckf_dbg_path:
                            num_obs = sum(
                                1 for cam_obs in cam_observations
                                for obs in cam_obs.get('observations', [])
                                if obs.get('fid') == fid
                            )
                            with open(msckf_dbg_path, "a", newline="") as mf:
                                mf.write(
                                    f"{vio_fe.frame_idx},{fid},{num_obs},0,nan,nan,0,nan,posttri_retry_depth_gate_reject\n"
                                )
                        continue
                cur_retry = abs(int(posttri_retry_state.get(int(fid), 0)))
                if cur_retry < int(posttri_retry_recover_max_cycles):
                    posttri_retry_state[int(fid)] = cur_retry + 1
                    posttri_retry_source[int(fid)] = (
                        "depth_sparse" if fail_reason == "fail_depth_sparse_recoverable" else "reproj_sparse"
                    )
                    if (
                        fail_reason == "fail_depth_sparse_recoverable"
                        and bool(posttri_retry_recover_protect_enable)
                        and int(posttri_retry_recover_protect_cycles) > 0
                    ):
                        posttri_retry_protect_state[int(fid)] = max(
                            int(posttri_retry_protect_state.get(int(fid), 0)),
                            int(posttri_retry_recover_protect_cycles) + 1,
                        )
                        MSCKF_STATS['posttri_retry_recover_depth_protected_added_count'] += 1
                    MSCKF_STATS['posttri_retry_recover_defer_count'] += 1
                    if fail_reason == "fail_depth_sparse_recoverable":
                        MSCKF_STATS['posttri_retry_recover_depth_defer_count'] += 1
                    if msckf_dbg_path:
                        num_obs = sum(
                            1 for cam_obs in cam_observations
                            for obs in cam_obs.get('observations', [])
                            if obs.get('fid') == fid
                        )
                        with open(msckf_dbg_path, "a", newline="") as mf:
                            mf.write(
                                f"{vio_fe.frame_idx},{fid},{num_obs},0,nan,nan,0,nan,posttri_retry_recover_defer\n"
                            )
                    continue
                MSCKF_STATS['posttri_retry_recover_exhausted_count'] += 1
                if str(posttri_retry_source.get(int(fid), "")) == "depth_sparse":
                    MSCKF_STATS['posttri_retry_recover_depth_exhausted_count'] += 1
                posttri_retry_source.pop(int(fid), None)
                posttri_retry_protect_state.pop(int(fid), None)
            if (
                posttri_retry_defer_enable
                and posttri_retry_max_cycles > 0
                and fail_reason == "fail_reproj_sparse"
                and sparse_short_track
                and (sparse_low_parallax or sparse_short_span)
            ):
                cur_retry = abs(int(posttri_retry_state.get(int(fid), 0)))
                if cur_retry < int(posttri_retry_max_cycles):
                    posttri_retry_state[int(fid)] = -(cur_retry + 1)
                    MSCKF_STATS['posttri_retry_defer_count'] += 1
                    if msckf_dbg_path:
                        num_obs = sum(
                            1 for cam_obs in cam_observations
                            for obs in cam_obs.get('observations', [])
                            if obs.get('fid') == fid
                        )
                        with open(msckf_dbg_path, "a", newline="") as mf:
                            mf.write(
                                f"{vio_fe.frame_idx},{fid},{num_obs},0,nan,nan,0,nan,posttri_retry_defer\n"
                            )
                    continue
            posttri_retry_state.pop(int(fid), None)
            posttri_retry_source.pop(int(fid), None)
            posttri_retry_protect_state.pop(int(fid), None)
            if msckf_dbg_path:
                num_obs = sum(1 for cam_obs in cam_observations 
                             for obs in cam_obs.get('observations', []) 
                             if obs.get('fid') == fid)
                with open(msckf_dbg_path, "a", newline="") as mf:
                    mf.write(f"{vio_fe.frame_idx},{fid},{num_obs},0,nan,nan,0,nan,{fail_reason}\n")
            continue

        parallax_val = float(triangulated.get("parallax_med_px", np.nan))
        reproj_p95_val = float(triangulated.get("reproj_p95_norm", np.nan))
        depth_ratio_val = float(triangulated.get("depth_positive_ratio", np.nan))
        quality_val = float(triangulated.get("quality", np.nan))
        if np.isfinite(parallax_val):
            parallax_samples.append(parallax_val)
        if np.isfinite(reproj_p95_val):
            reproj_p95_samples.append(reproj_p95_val)
        if np.isfinite(depth_ratio_val):
            depth_ratio_samples.append(depth_ratio_val)
        if np.isfinite(quality_val):
            feature_quality_samples.append(quality_val)
        
        # Check if feature is associated with a plane
        associated_plane = None
        if len(detected_planes) > 0 and plane_config is not None:
            point_3d = triangulated['p_w']
            distance_threshold = plane_config.get('PLANE_DISTANCE_THRESHOLD', 0.15)
            
            # Find nearest plane
            min_dist = float('inf')
            for plane in detected_planes:
                dist = abs(plane.point_distance(point_3d))
                if dist < min_dist and dist < distance_threshold:
                    min_dist = dist
                    associated_plane = plane
        
        # MSCKF update with optional plane constraint
        num_obs = sum(
            1 for cam_obs in cam_observations
            for obs in cam_obs.get('observations', [])
            if obs.get('fid') == fid
        )
        use_plane_constraint = (
            associated_plane is not None
            and plane_config is not None
            and plane_config.get('PLANE_USE_CONSTRAINTS', True)
        )
        dof_est = max(1, 2 * max(1, num_obs) + (1 if use_plane_constraint else 0))
        num_attempted += 1
        dof_samples.append(dof_est)
        retry_mode_key_used = str(triangulated.get("retry_mode_key", "")).strip().lower()
        same_cycle_retry_update_mode = bool(
            retry_mode_key_used == "depth_sparse_same_cycle_rescue"
            and bool(in_posttri_recover_retry)
            and bool(depth_sparse_retry_source)
            and bool(protected_retry_carried)
        )
        if same_cycle_retry_update_mode:
            promoted_obs = int(max(0, triangulated.get("rescue_promoted_obs_total", 0)))
            avg_reproj_err = float(triangulated.get("avg_reproj_error", np.nan))
            depth_pos_ratio = float(triangulated.get("depth_positive_ratio", np.nan))
            if promoted_obs <= 0 or promoted_obs > int(posttri_retry_same_cycle_max_promoted_obs):
                _record_same_cycle_retry_failure("fail_geometry_insufficient_posttri")
                _record_depth_retry_failure("fail_geometry_insufficient_posttri")
                continue
            if (not np.isfinite(avg_reproj_err)) or (
                avg_reproj_err > float(posttri_retry_same_cycle_max_avg_reproj_norm)
            ):
                _record_same_cycle_retry_failure("fail_reproj_sparse_recoverable")
                _record_depth_retry_failure("fail_reproj_sparse_recoverable")
                continue
            if (not np.isfinite(depth_pos_ratio)) or (
                depth_pos_ratio < float(posttri_retry_same_cycle_min_depth_ratio)
            ):
                _record_same_cycle_retry_failure("fail_depth_sparse_recoverable")
                _record_depth_retry_failure("fail_depth_sparse_recoverable")
                continue
        local_chi2_scale = float(chi2_scale)
        if same_cycle_retry_update_mode:
            local_chi2_scale *= float(posttri_retry_same_cycle_chi2_scale_mult)

        if associated_plane is not None and plane_config.get('PLANE_USE_CONSTRAINTS', True):
            # Stacked measurement: bearing (2D per obs) + plane constraint (1D)
            success, innovation_norm, chi2_test = msckf_measurement_update_with_plane(
                fid, triangulated, cam_observations, cam_states, kf,
                associated_plane, plane_config,
                chi2_scale=local_chi2_scale,
                global_config=global_config)
        else:
            # Standard MSCKF update (bearing only)
            success, innovation_norm, chi2_test = msckf_measurement_update(
                fid, triangulated, cam_observations, cam_states, kf,
                chi2_scale=local_chi2_scale,
                global_config=global_config)
        
        if np.isfinite(chi2_test):
            chi2_norm_samples.append(float(chi2_test) / float(max(1, dof_est)))
        
        if msckf_dbg_path:
            avg_reproj = triangulated.get('avg_reproj_error', np.nan)
            q_band = triangulated.get("quality_band", "")
            status_reason = "success" if success else f"update_reject_{q_band}"
            with open(msckf_dbg_path, "a", newline="") as mf:
                mf.write(f"{vio_fe.frame_idx},{fid},{num_obs},1,{avg_reproj:.3f},"
                        f"{innovation_norm:.3f},{int(success)},{chi2_test:.3f},{status_reason}\n")
        
        if success:
            num_successful += 1
            retry_state.pop(int(fid), None)
            if int(posttri_retry_state.get(int(fid), 0)) > 0:
                MSCKF_STATS['posttri_retry_recover_success_count'] += 1
                if str(posttri_retry_source.get(int(fid), "")) == "depth_sparse":
                    MSCKF_STATS['posttri_retry_recover_depth_success_count'] += 1
                    if bool(same_cycle_retry_update_mode):
                        MSCKF_STATS['posttri_retry_recover_depth_same_cycle_success_count'] += 1
            posttri_retry_state.pop(int(fid), None)
            posttri_retry_source.pop(int(fid), None)
            posttri_retry_protect_state.pop(int(fid), None)
        elif bool(same_cycle_retry_update_mode):
            _record_same_cycle_retry_failure("fail_reproj_sparse_recoverable")
            _record_depth_retry_failure("fail_reproj_sparse_recoverable")

    if posttri_retry_protect_state:
        posttri_retry_protect_state = {
            int(fid): int(cnt) - 1
            for fid, cnt in posttri_retry_protect_state.items()
            if int(cnt) - 1 > 0
        }

    if vio_fe is not None:
        try:
            setattr(vio_fe, "_msckf_retry_state", retry_state)
        except Exception:
            pass
        try:
            setattr(vio_fe, "_msckf_posttri_retry_state", posttri_retry_state)
        except Exception:
            pass
        try:
            setattr(vio_fe, "_msckf_posttri_retry_source", posttri_retry_source)
        except Exception:
            pass
        try:
            setattr(vio_fe, "_msckf_posttri_retry_protect_state", posttri_retry_protect_state)
        except Exception:
            pass
    
    if len(chi2_norm_samples) > 0:
        nis_norm = float(np.mean(chi2_norm_samples))
    else:
        nis_norm = np.nan
    quality_snap = _summarize_msckf_quality(
        t=float(timestamp),
        track_count=int(num_attempted),
        accepted_count=int(num_successful),
        parallax_px=parallax_samples,
        reproj_p95_norm=reproj_p95_samples,
        depth_positive_ratio=depth_ratio_samples,
        feature_quality=feature_quality_samples,
        global_config=global_config,
    )

    if stats_out is not None:
        stats_out.clear()
        stats_out.update({
            "sensor": "MSCKF",
            "accepted": bool(num_successful > 0),
            "dof": int(round(float(np.mean(dof_samples)))) if len(dof_samples) > 0 else 1,
            "nis_norm": nis_norm,
            "chi2": float(nis_norm) if np.isfinite(nis_norm) else np.nan,
            "threshold": np.nan,
            "r_scale_used": 1.0,
            "attempted": int(num_attempted),
            "accepted_count": int(num_successful),
            "quality_score": float(quality_snap.quality_score),
            "inlier_ratio": float(quality_snap.inlier_ratio),
            "parallax_med_px": float(quality_snap.parallax_med_px),
            "reproj_p95_norm": float(quality_snap.reproj_p95_norm),
            "depth_positive_ratio": float(quality_snap.depth_positive_ratio),
            "stable_geometry_flag": float(bool(quality_snap.stable_geometry_flag)),
            "conditioning_risk": float(quality_snap.conditioning_risk),
            "feature_track_health": float(quality_snap.feature_track_health),
            "unstable_reason_code": str(quality_snap.unstable_reason_code),
        })
    if quality_out is not None:
        quality_out.clear()
        quality_out.update(quality_snap.as_dict())
    
    return num_successful


# =============================================================================
# High-level MSCKF Trigger Function
# =============================================================================

def trigger_msckf_update(kf, cam_states: list, cam_observations: list,
                         vio_fe, t: float = 0.0,
                         msckf_dbg_csv: Optional[str] = None,
                         dem_reader=None, origin_lat: float = 0.0,
                         origin_lon: float = 0.0,
                         plane_detector=None,
                         plane_config: dict = None,
                         global_config: dict = None,
                         chi2_scale: float = 1.0,
                         reproj_scale: float = 1.0,
                         phase: int = 2,
                         health_state: str = "HEALTHY",
                         adaptive_info: Optional[Dict[str, Any]] = None,
                         policy_decision: Optional[Any] = None,
                         runner: Optional[Any] = None) -> int:
    """
    Trigger MSCKF multi-view geometric update.
    
    This function decides when to perform MSCKF updates based on:
    - Number of mature features (seen in multiple views)
    - Number of camera clones in sliding window
    - Frame intervals
    
    Args:
        kf: ExtendedKalmanFilter instance
        cam_states: List of camera clone states
        cam_observations: List of camera observations
        vio_fe: VIO frontend
        t: Current timestamp for logging
        msckf_dbg_csv: Optional debug CSV path
        dem_reader: Optional DEM reader for ground constraints
        origin_lat/origin_lon: Local projection origin
    
    Returns:
        Number of successful MSCKF updates
    """
    if policy_decision is not None:
        mode = str(getattr(policy_decision, "mode", "APPLY")).upper()
        if mode in ("HOLD", "SKIP"):
            if adaptive_info is not None:
                adaptive_info.clear()
                adaptive_info.update({
                    "sensor": "MSCKF",
                    "accepted": False,
                    "dof": 1,
                    "nis_norm": np.nan,
                    "chi2": np.nan,
                    "threshold": np.nan,
                    "r_scale_used": 1.0,
                    "attempted": 0,
                    "reason_code": f"policy_mode_{mode.lower()}",
                })
            return 0
        try:
            chi2_scale = float(getattr(policy_decision, "chi2_scale", chi2_scale))
        except Exception:
            pass
        try:
            reproj_scale = float(getattr(policy_decision, "reproj_scale", reproj_scale))
        except Exception:
            pass
        try:
            phase = int(round(float(policy_decision.extra("phase", float(phase)))))
        except Exception:
            pass
        try:
            health_state = str(policy_decision.extra_str("health_state", str(health_state))).upper()
        except Exception:
            pass

    if vio_fe is None or len(cam_states) < 2:
        if adaptive_info is not None:
            adaptive_info.clear()
            adaptive_info.update({
                "sensor": "MSCKF",
                "accepted": False,
                "dof": 1,
                "nis_norm": np.nan,
                "chi2": np.nan,
                "threshold": np.nan,
                "r_scale_used": 1.0,
                "phase": int(phase),
            })
        return 0
    
    # Count mature features
    feature_obs_count = {}
    for obs_set in cam_observations:
        for obs in obs_set['observations']:
            fid = obs['fid']
            feature_obs_count[fid] = feature_obs_count.get(fid, 0) + 1
    
    num_mature = sum(1 for c in feature_obs_count.values() if c >= 2)
    
    # Decide if we should update
    # TUNED: Reduced from 20→10 to process features earlier (was causing 77.5% to wait for full window)
    should_update = (
        num_mature >= 10 or                                    # Many mature features (reduced from 20)
        len(cam_states) >= 4 or                               # Window getting full
        (vio_fe.frame_idx % 5 == 0 and len(cam_states) >= 3)  # Periodic update
    )
    
    if should_update:
        try:
            stats = {}
            quality_stats: Dict[str, float] = {}
            num_updates = perform_msckf_updates(
                vio_fe,
                cam_observations,
                cam_states,
                kf,
                min_observations=2,
                max_features=50,
                msckf_dbg_path=msckf_dbg_csv,
                dem_reader=dem_reader,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                plane_detector=plane_detector,
                plane_config=plane_config,
                global_config=global_config,
                chi2_scale=chi2_scale,
                reproj_scale=reproj_scale,
                phase=int(phase),
                health_state=str(health_state),
                stats_out=stats,
                quality_out=quality_stats,
                timestamp=float(t),
            )
            if len(quality_stats) > 0:
                snap = MsckfQualitySnapshot(
                    timestamp=float(quality_stats.get("timestamp", t)),
                    track_count=int(quality_stats.get("track_count", 0)),
                    inlier_ratio=float(quality_stats.get("inlier_ratio", np.nan)),
                    parallax_med_px=float(quality_stats.get("parallax_med_px", np.nan)),
                    reproj_p95_norm=float(quality_stats.get("reproj_p95_norm", np.nan)),
                    depth_positive_ratio=float(quality_stats.get("depth_positive_ratio", np.nan)),
                    quality_score=float(quality_stats.get("quality_score", np.nan)),
                    stable_geometry_flag=bool(float(quality_stats.get("stable_geometry_flag", 0.0)) >= 0.5),
                    conditioning_risk=float(quality_stats.get("conditioning_risk", np.nan)),
                    feature_track_health=float(quality_stats.get("feature_track_health", np.nan)),
                    unstable_reason_code=str(quality_stats.get("unstable_reason_code", "stable")),
                )
                if runner is not None:
                    try:
                        runner._msckf_quality_snapshot = snap
                        hist = getattr(runner, "_msckf_quality_history", None)
                        if isinstance(hist, list) and np.isfinite(float(snap.quality_score)):
                            hist.append(float(snap.quality_score))
                            if len(hist) > 20000:
                                del hist[:-20000]
                        st_hist = getattr(runner, "_msckf_stable_geometry_history", None)
                        if isinstance(st_hist, list):
                            st_hist.append(1.0 if bool(snap.stable_geometry_flag) else 0.0)
                            if len(st_hist) > 20000:
                                del st_hist[:-20000]
                        _log_msckf_quality_csv(getattr(runner, "msckf_quality_csv", None), snap)
                        if (
                            getattr(runner, "yaw_authority_service", None) is not None
                            and bool(global_config.get("YAW_AUTH_USE_MSCKF_CONFIDENCE", True))
                        ):
                            runner.yaw_authority_service.set_external_confidence(
                                source="MSCKF",
                                timestamp=float(t),
                                confidence=float(np.clip(snap.quality_score, 0.0, 1.0))
                                if np.isfinite(float(snap.quality_score))
                                else 0.0,
                            )
                    except Exception:
                        pass
            if adaptive_info is not None:
                adaptive_info.clear()
                if len(stats) > 0:
                    adaptive_info.update(stats)
                    adaptive_info["phase"] = int(phase)
                else:
                    adaptive_info.update({
                        "sensor": "MSCKF",
                        "accepted": bool(num_updates > 0),
                        "dof": 1,
                        "nis_norm": np.nan,
                        "chi2": np.nan,
                        "threshold": np.nan,
                        "r_scale_used": 1.0,
                        "phase": int(phase),
                    })
            if num_updates > 0:
                _log_msckf_update(f"[MSCKF] Updated {num_updates} features at t={t:.3f}s")
            return num_updates
        except Exception as e:
            print(f"[MSCKF] Error: {e}")
            if adaptive_info is not None:
                adaptive_info.clear()
                adaptive_info.update({
                    "sensor": "MSCKF",
                    "accepted": False,
                    "dof": 1,
                    "nis_norm": np.nan,
                    "chi2": np.nan,
                    "threshold": np.nan,
                    "r_scale_used": 1.0,
                    "phase": int(phase),
                })
            return 0
    
    if adaptive_info is not None:
        adaptive_info.clear()
        adaptive_info.update({
            "sensor": "MSCKF",
            "accepted": False,
            "dof": 1,
            "nis_norm": np.nan,
            "chi2": np.nan,
            "threshold": np.nan,
            "r_scale_used": 1.0,
            "attempted": 0,
            "phase": int(phase),
        })
    
    return 0
