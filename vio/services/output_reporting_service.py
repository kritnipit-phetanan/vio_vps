"""Output and reporting service for VIORunner.

This module contains CSV logging and end-of-run summary/reporting logic so
`main_loop.py` can stay focused on orchestration.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy

from ..data_loaders import interpolate_time_ref
from ..math_utils import quaternion_to_yaw
from ..msckf import print_msckf_stats
from ..output_utils import (
    append_benchmark_health_summary,
    log_convention_event,
    print_error_statistics,
)
from ..state_manager import imu_to_gnss_position


def build_sensor_time_audit_rows(sensor_times: Dict[str, np.ndarray],
                                 reference_sensor: str = "IMU",
                                 in_range_frac_warn: float = 0.995,
                                 nn_dt_p95_warn_sec: float = 0.005) -> List[Dict[str, float]]:
    """
    Build runtime time-base audit rows for all sensors against a reference stream.

    Returns one dict per sensor with overlap and nearest-neighbor dt statistics.
    """
    rows: List[Dict[str, float]] = []
    if reference_sensor not in sensor_times:
        return rows

    ref = np.asarray(sensor_times.get(reference_sensor, []), dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return rows
    ref = np.sort(ref)
    ref_start = float(ref[0])
    ref_end = float(ref[-1])

    for sensor, arr in sensor_times.items():
        t = np.asarray(arr, dtype=float)
        t = t[np.isfinite(t)]
        t = np.sort(t)
        n = int(t.size)
        if n == 0:
            rows.append({
                "sensor": sensor,
                "start_t": np.nan,
                "end_t": np.nan,
                "samples": 0,
                "overlap_start": np.nan,
                "overlap_end": np.nan,
                "overlap_sec": 0.0,
                "in_range_frac": 0.0,
                "nn_dt_mean_s": np.nan,
                "nn_dt_p95_s": np.nan,
                "nn_dt_max_s": np.nan,
                "warn": 1.0,
            })
            continue

        start_t = float(t[0])
        end_t = float(t[-1])
        overlap_start = float(max(start_t, ref_start))
        overlap_end = float(min(end_t, ref_end))
        overlap_sec = float(max(0.0, overlap_end - overlap_start))

        in_range = (t >= ref_start) & (t <= ref_end)
        in_range_frac = float(np.mean(in_range)) if n > 0 else 0.0

        idx = np.searchsorted(ref, t, side="left")
        idx_right = np.clip(idx, 0, ref.size - 1)
        idx_left = np.clip(idx - 1, 0, ref.size - 1)
        dt_right = np.abs(t - ref[idx_right])
        dt_left = np.abs(t - ref[idx_left])
        nn_dt = np.minimum(dt_left, dt_right)
        nn_dt_mean = float(np.nanmean(nn_dt)) if nn_dt.size > 0 else np.nan
        nn_dt_p95 = float(np.nanpercentile(nn_dt, 95)) if nn_dt.size > 0 else np.nan
        nn_dt_max = float(np.nanmax(nn_dt)) if nn_dt.size > 0 else np.nan

        warn = (
            (in_range_frac < float(in_range_frac_warn))
            or (np.isfinite(nn_dt_p95) and nn_dt_p95 > float(nn_dt_p95_warn_sec))
        )
        rows.append({
            "sensor": sensor,
            "start_t": start_t,
            "end_t": end_t,
            "samples": n,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "overlap_sec": overlap_sec,
            "in_range_frac": in_range_frac,
            "nn_dt_mean_s": nn_dt_mean,
            "nn_dt_p95_s": nn_dt_p95,
            "nn_dt_max_s": nn_dt_max,
            "warn": 1.0 if warn else 0.0,
        })
    return rows


class OutputReportingService:
    """Encapsulates output logging and benchmark reporting for a VIORunner."""

    def __init__(self, runner: Any):
        self.runner = runner

    def log_time_sync_debug(self,
                            t_filter: float,
                            t_gt_mapped: float,
                            dt_gt: float,
                            gt_idx: int,
                            gt_stamp_log: float,
                            matched: int):
        """Append one per-frame time sync debug row."""
        if not self.runner.time_sync_csv:
            return
        try:
            with open(self.runner.time_sync_csv, "a", newline="") as f:
                f.write(
                    f"{t_filter:.6f},{t_gt_mapped:.6f},{dt_gt:.6f},{gt_idx},"
                    f"{gt_stamp_log:.6f},{matched},{self.runner.error_time_mode}\n"
                )
        except Exception:
            pass

    def log_convention_check(self,
                             t: float,
                             sensor: str,
                             check: str,
                             value: float,
                             threshold: float,
                             status: str,
                             note: str = ""):
        """Soft-monitor for frame/time convention consistency (never fail-fast)."""
        log_convention_event(
            self.runner.convention_csv,
            t=float(t),
            sensor=str(sensor),
            check=str(check),
            value=float(value),
            threshold=float(threshold),
            status=str(status),
            note=str(note),
        )
        if str(status).upper() in ("FAIL", "WARN"):
            key = f"{sensor}:{check}"
            count = int(self.runner._convention_warn_counts.get(key, 0)) + 1
            self.runner._convention_warn_counts[key] = count
            if count <= 5 or count % 200 == 0:
                print(
                    f"[CONVENTION] {sensor}/{check} {status}: "
                    f"value={value:.6g}, threshold={threshold:.6g} {note}"
                )

    def run_bootstrap_convention_checks(self):
        """One-shot frame/convention sanity checks after EKF init."""
        t0 = float(self.runner.state.t0) if np.isfinite(float(self.runner.state.t0)) else 0.0
        identity = np.eye(3, dtype=float)
        for name in ("BODY_T_CAMDOWN", "BODY_T_CAMFRONT", "BODY_T_CAMSIDE"):
            t_mat = self.runner.global_config.get(name, None)
            if t_mat is None:
                continue
            try:
                t_arr = np.array(t_mat, dtype=float)
                if t_arr.shape != (4, 4):
                    self.log_convention_check(
                        t0, "FRAME", "extrinsics_shape", float(t_arr.size), 16.0, "WARN", f"{name}"
                    )
                    continue
                r_mat = t_arr[:3, :3]
                ortho_err = float(np.linalg.norm(r_mat.T @ r_mat - identity, ord="fro"))
                det_val = float(np.linalg.det(r_mat))
                status_r = "PASS" if ortho_err <= 1e-2 else "WARN"
                status_d = "PASS" if abs(det_val - 1.0) <= 5e-2 else "WARN"
                self.log_convention_check(t0, "FRAME", "orthogonality", ortho_err, 1e-2, status_r, name)
                self.log_convention_check(t0, "FRAME", "determinant", abs(det_val - 1.0), 5e-2, status_d, name)
            except Exception as exc:
                self.log_convention_check(t0, "FRAME", "extrinsics_parse", 1.0, 0.0, "WARN", f"{name}:{exc}")

        g_norm = float(self.runner.global_config.get("IMU_PARAMS", {}).get("g_norm", 9.8066))
        g_world_z = -g_norm
        self.log_convention_check(
            t0,
            "FRAME",
            "enu_gravity_sign",
            g_world_z,
            0.0,
            "PASS" if g_world_z < 0.0 else "WARN",
            "ENU expects gravity on -Z",
        )

    def run_sensor_time_audit(self):
        """Write one runtime time-base audit table for loaded sensors."""
        audit_csv = getattr(self.runner, "sensor_time_audit_csv", None)
        if not audit_csv:
            return

        sensor_times: Dict[str, np.ndarray] = {
            "IMU": np.array([rec.t for rec in (self.runner.imu or [])], dtype=float),
            "CAM": np.array([item.t for item in (self.runner.imgs or [])], dtype=float),
            "MAG": np.array([item.t for item in (self.runner.mag_list or [])], dtype=float),
            "GT": np.array(
                pd.to_numeric(
                    (self.runner.ppk_trajectory["stamp_log"] if self.runner.ppk_trajectory is not None and "stamp_log" in self.runner.ppk_trajectory else []),
                    errors="coerce",
                ),
                dtype=float,
            ),
        }
        if getattr(self.runner, "msl_interpolator", None) is not None and hasattr(self.runner.msl_interpolator, "times"):
            sensor_times["MSL"] = np.array(self.runner.msl_interpolator.times, dtype=float)
        if getattr(self.runner, "vps_list", None):
            sensor_times["VPS_LIST"] = np.array([item.t for item in self.runner.vps_list], dtype=float)

        rows = build_sensor_time_audit_rows(sensor_times, reference_sensor="IMU")
        rows.extend(self._build_pps_mapping_rows(sensor_times.get("IMU", np.array([], dtype=float))))
        if len(rows) == 0:
            return

        with open(audit_csv, "a", newline="") as f:
            for row in rows:
                f.write(
                    f"{row['sensor']},{row['start_t']:.6f},{row['end_t']:.6f},{int(row['samples'])},"
                    f"{row['overlap_start']:.6f},{row['overlap_end']:.6f},{row['overlap_sec']:.6f},"
                    f"{row['in_range_frac']:.6f},{row['nn_dt_mean_s']:.6f},{row['nn_dt_p95_s']:.6f},"
                    f"{row['nn_dt_max_s']:.6f},{int(row['warn'])}\n"
                )

        warns = [r for r in rows if int(r.get("warn", 0)) == 1]
        if warns:
            print("[TIME-AUDIT] WARN sensors:")
            for row in warns:
                print(
                    f"  - {row['sensor']}: in_range_frac={row['in_range_frac']:.4f}, "
                    f"nn_dt_p95={row['nn_dt_p95_s']*1000.0:.2f}ms"
                )
        else:
            print("[TIME-AUDIT] PASS all sensors (reference=IMU)")

    def _build_pps_mapping_rows(self, imu_ref_times: np.ndarray) -> List[Dict[str, float]]:
        """
        Build additional audit rows from /imu/time_ref_pps mapping quality.

        Row semantics:
        - sensor: "<SOURCE>_PPS_MAP"
        - nn_dt_* columns store absolute mapping residual statistics (seconds)
          between source time_ref and PPS-predicted time_ref from source ROS stamps.
        """
        rows: List[Dict[str, float]] = []
        pps_csv = getattr(self.runner, "time_ref_pps_csv", None)
        if not pps_csv or not os.path.isfile(pps_csv):
            return rows

        try:
            pps_df = pd.read_csv(pps_csv)
        except Exception:
            return rows

        if "time_ref" not in pps_df.columns:
            return rows

        ros_col = next((c for c in ("stamp_msg", "stamp_bag", "stamp_log") if c in pps_df.columns), None)
        if ros_col is None:
            return rows

        map_df = pps_df[[ros_col, "time_ref"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if len(map_df) < 2:
            return rows
        map_df[ros_col] = pd.to_numeric(map_df[ros_col], errors="coerce")
        map_df["time_ref"] = pd.to_numeric(map_df["time_ref"], errors="coerce")
        map_df = map_df.dropna().sort_values(ros_col).drop_duplicates(subset=[ros_col], keep="first")
        if len(map_df) < 2:
            return rows

        pairs = list(zip(map_df[ros_col].to_numpy(dtype=float), map_df["time_ref"].to_numpy(dtype=float)))
        ros_min = float(map_df[ros_col].iloc[0])
        ros_max = float(map_df[ros_col].iloc[-1])
        ref = np.asarray(imu_ref_times, dtype=float)
        ref = ref[np.isfinite(ref)]
        ref_start = float(np.min(ref)) if ref.size else np.nan
        ref_end = float(np.max(ref)) if ref.size else np.nan

        source_streams: List[tuple[str, np.ndarray, np.ndarray]] = []

        # IMU mapping: stamp_* -> time_ref from imu_with_ref.csv
        try:
            imu_df = pd.read_csv(self.runner.config.imu_path, usecols=["time_ref", ros_col])
            imu_df = imu_df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(imu_df) > 0:
                source_streams.append((
                    "IMU",
                    pd.to_numeric(imu_df[ros_col], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(imu_df["time_ref"], errors="coerce").to_numpy(dtype=float),
                ))
        except Exception:
            pass

        # Camera time_ref mapping (timeref_csv) if present.
        if getattr(self.runner.config, "timeref_csv", None) and os.path.isfile(self.runner.config.timeref_csv):
            try:
                cam_df = pd.read_csv(self.runner.config.timeref_csv, usecols=["time_ref", ros_col])
                cam_df = cam_df.replace([np.inf, -np.inf], np.nan).dropna()
                if len(cam_df) > 0:
                    source_streams.append((
                        "CAM_TIMEREF",
                        pd.to_numeric(cam_df[ros_col], errors="coerce").to_numpy(dtype=float),
                        pd.to_numeric(cam_df["time_ref"], errors="coerce").to_numpy(dtype=float),
                    ))
            except Exception:
                pass

        # Magnetometer mapping: stamp_* from vector3.csv paired with converted mag_list time_ref.
        if getattr(self.runner.config, "mag_csv", None) and os.path.isfile(self.runner.config.mag_csv) and getattr(self.runner, "mag_list", None):
            try:
                mag_df = pd.read_csv(self.runner.config.mag_csv, usecols=[ros_col]).replace([np.inf, -np.inf], np.nan).dropna()
                mag_ros = pd.to_numeric(mag_df[ros_col], errors="coerce").to_numpy(dtype=float)
                mag_ref = np.array([float(m.t) for m in self.runner.mag_list], dtype=float)
                n = int(min(mag_ros.size, mag_ref.size))
                if n > 0:
                    source_streams.append(("MAG", mag_ros[:n], mag_ref[:n]))
            except Exception:
                pass

        for src_name, t_ros_raw, t_ref_raw in source_streams:
            t_ros = np.asarray(t_ros_raw, dtype=float)
            t_ref = np.asarray(t_ref_raw, dtype=float)
            n = int(min(t_ros.size, t_ref.size))
            if n < 2:
                continue
            t_ros = t_ros[:n]
            t_ref = t_ref[:n]
            mask = np.isfinite(t_ros) & np.isfinite(t_ref)
            if int(np.sum(mask)) < 2:
                continue
            t_ros = t_ros[mask]
            t_ref = t_ref[mask]

            t_ref_pred = interpolate_time_ref(t_ros, pairs)
            residual = t_ref_pred - t_ref
            residual_abs = np.abs(residual)
            in_range = (t_ros >= ros_min) & (t_ros <= ros_max)
            in_range_frac = float(np.mean(in_range)) if t_ros.size > 0 else 0.0

            residual_eval = residual_abs[in_range] if np.any(in_range) else residual_abs
            mean_res = float(np.nanmean(residual_eval)) if residual_eval.size > 0 else np.nan
            p95_res = float(np.nanpercentile(residual_eval, 95)) if residual_eval.size > 0 else np.nan
            max_res = float(np.nanmax(residual_eval)) if residual_eval.size > 0 else np.nan

            start_t = float(np.nanmin(t_ref))
            end_t = float(np.nanmax(t_ref))
            overlap_start = float(max(start_t, ref_start)) if np.isfinite(ref_start) else start_t
            overlap_end = float(min(end_t, ref_end)) if np.isfinite(ref_end) else end_t
            overlap_sec = float(max(0.0, overlap_end - overlap_start))
            warn = (in_range_frac < 0.995) or (np.isfinite(p95_res) and p95_res > 0.005)

            rows.append({
                "sensor": f"{src_name}_PPS_MAP",
                "start_t": start_t,
                "end_t": end_t,
                "samples": int(t_ref.size),
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "overlap_sec": overlap_sec,
                "in_range_frac": in_range_frac,
                "nn_dt_mean_s": mean_res,
                "nn_dt_p95_s": p95_res,
                "nn_dt_max_s": max_res,
                "warn": 1.0 if warn else 0.0,
            })
        return rows

    def log_error(self, t: float):
        """Log VIO error vs ground truth."""
        gt_df = self.runner.ppk_trajectory
        if gt_df is None or len(gt_df) == 0:
            self.log_time_sync_debug(
                t_filter=float(t),
                t_gt_mapped=float("nan"),
                dt_gt=float("nan"),
                gt_idx=-1,
                gt_stamp_log=float("nan"),
                matched=0,
            )
            return

        try:
            t_gt = self.runner.bootstrap_service.filter_time_to_gt_time(t)

            # Find closest ground truth
            gt_diffs = np.abs(gt_df["stamp_log"].values - t_gt)
            gt_idx = int(np.argmin(gt_diffs))
            gt_row = gt_df.iloc[gt_idx]
            dt_gt = float(gt_diffs[gt_idx])
            gt_stamp_log = float(gt_row["stamp_log"]) if "stamp_log" in gt_row else float("nan")
            is_matched = 1 if dt_gt <= 1.0 else 0
            self.log_time_sync_debug(
                t_filter=float(t),
                t_gt_mapped=float(t_gt),
                dt_gt=dt_gt,
                gt_idx=gt_idx,
                gt_stamp_log=gt_stamp_log,
                matched=is_matched,
            )
            if dt_gt > 1.0:
                if not self.runner._warned_gt_time_mismatch:
                    print(
                        f"[ERROR_LOG] WARNING: Large GT time mismatch ({dt_gt:.3f}s). "
                        f"mode={self.runner.error_time_mode}. Skipping unmatched rows."
                    )
                    self.runner._warned_gt_time_mismatch = True
                return

            use_ppk = self.runner.ppk_trajectory is not None

            if use_ppk:
                gt_lat = gt_row["lat"]
                gt_lon = gt_row["lon"]
                gt_alt = gt_row["height"]
            else:
                gt_lat = gt_row["lat_dd"]
                gt_lon = gt_row["lon_dd"]
                gt_alt = gt_row["altitude_MSL_m"]

            gt_E, gt_N = self.runner.proj_cache.latlon_to_xy(gt_lat, gt_lon, self.runner.lat0, self.runner.lon0)
            gt_U = gt_alt

            # VIO prediction (IMU position)
            vio_E = float(self.runner.kf.x[0, 0])
            vio_N = float(self.runner.kf.x[1, 0])
            vio_U = float(self.runner.kf.x[2, 0])

            # Ground truth is GNSS position - convert VIO (IMU) to GNSS for fair comparison
            if np.linalg.norm(self.runner.lever_arm) > 0.01:
                q_vio = self.runner.kf.x[6:10, 0]
                q_xyzw = np.array([q_vio[1], q_vio[2], q_vio[3], q_vio[0]])
                r_body_to_world = R_scipy.from_quat(q_xyzw).as_matrix()

                p_imu_enu = np.array([vio_E, vio_N, vio_U])
                p_gnss_enu = imu_to_gnss_position(p_imu_enu, r_body_to_world, self.runner.lever_arm)
                vio_E, vio_N, vio_U = p_gnss_enu[0], p_gnss_enu[1], p_gnss_enu[2]

            # Errors
            err_E = vio_E - gt_E
            err_N = vio_N - gt_N
            err_U = vio_U - gt_U
            pos_error = np.sqrt(err_E ** 2 + err_N ** 2 + err_U ** 2)

            # Velocity error
            vel_error = 0.0
            vel_err_E = vel_err_N = vel_err_U = 0.0

            if use_ppk and "ve" in gt_row and "vn" in gt_row and "vu" in gt_row:
                gt_vel_E = float(gt_row["ve"])
                gt_vel_N = float(gt_row["vn"])
                gt_vel_U = float(gt_row["vu"])

                vio_vel_E = float(self.runner.kf.x[3, 0])
                vio_vel_N = float(self.runner.kf.x[4, 0])
                vio_vel_U = float(self.runner.kf.x[5, 0])

                vel_err_E = vio_vel_E - gt_vel_E
                vel_err_N = vio_vel_N - gt_vel_N
                vel_err_U = vio_vel_U - gt_vel_U
                vel_error = np.sqrt(vel_err_E ** 2 + vel_err_N ** 2 + vel_err_U ** 2)
            elif gt_idx > 0 and gt_idx < len(gt_df) - 1:
                gt_row_prev = gt_df.iloc[gt_idx - 1]
                gt_row_next = gt_df.iloc[gt_idx + 1]

                dt = gt_row_next["stamp_log"] - gt_row_prev["stamp_log"]
                if dt > 0.01:
                    if use_ppk:
                        gt_E_prev, gt_N_prev = self.runner.proj_cache.latlon_to_xy(
                            gt_row_prev["lat"], gt_row_prev["lon"], self.runner.lat0, self.runner.lon0
                        )
                        gt_E_next, gt_N_next = self.runner.proj_cache.latlon_to_xy(
                            gt_row_next["lat"], gt_row_next["lon"], self.runner.lat0, self.runner.lon0
                        )
                        gt_U_prev = gt_row_prev["height"]
                        gt_U_next = gt_row_next["height"]
                    else:
                        gt_E_prev, gt_N_prev = self.runner.proj_cache.latlon_to_xy(
                            gt_row_prev["lat_dd"], gt_row_prev["lon_dd"], self.runner.lat0, self.runner.lon0
                        )
                        gt_E_next, gt_N_next = self.runner.proj_cache.latlon_to_xy(
                            gt_row_next["lat_dd"], gt_row_next["lon_dd"], self.runner.lat0, self.runner.lon0
                        )
                        gt_U_prev = gt_row_prev["altitude_MSL_m"]
                        gt_U_next = gt_row_next["altitude_MSL_m"]

                    gt_vel_E = (gt_E_next - gt_E_prev) / dt
                    gt_vel_N = (gt_N_next - gt_N_prev) / dt
                    gt_vel_U = (gt_U_next - gt_U_prev) / dt

                    vio_vel_E = float(self.runner.kf.x[3, 0])
                    vio_vel_N = float(self.runner.kf.x[4, 0])
                    vio_vel_U = float(self.runner.kf.x[5, 0])

                    vel_err_E = vio_vel_E - gt_vel_E
                    vel_err_N = vio_vel_N - gt_vel_N
                    vel_err_U = vio_vel_U - gt_vel_U
                    vel_error = np.sqrt(vel_err_E ** 2 + vel_err_N ** 2 + vel_err_U ** 2)

            # Yaw
            q_vio = self.runner.kf.x[6:10, 0]
            yaw_vio = np.rad2deg(quaternion_to_yaw(q_vio))

            yaw_gt = np.nan
            yaw_error = np.nan
            if use_ppk and "yaw" in gt_row:
                yaw_gt = 90.0 - np.rad2deg(gt_row["yaw"])
                yaw_error = ((yaw_vio - yaw_gt + 180) % 360) - 180

            with open(self.runner.error_csv, "a", newline="") as ef:
                ef.write(
                    f"{t:.6f},{pos_error:.3f},{err_E:.3f},{err_N:.3f},{err_U:.3f},"
                    f"{vel_error:.3f},{vel_err_E:.3f},{vel_err_N:.3f},{vel_err_U:.3f},"
                    f"{err_U:.3f},{yaw_vio:.2f},{yaw_gt:.2f},{yaw_error:.2f},"
                    f"{gt_lat:.8f},{gt_lon:.8f},{gt_alt:.3f},"
                    f"{vio_E:.3f},{vio_N:.3f},{vio_U:.3f}\n"
                )
        except Exception:
            pass

    def log_pose(self,
                 t: float,
                 dt: float,
                 used_vo: bool,
                 vo_data: Optional[Dict] = None,
                 msl_now: Optional[float] = None,
                 agl_now: Optional[float] = None,
                 lat_now: Optional[float] = None,
                 lon_now: Optional[float] = None):
        """Log current pose to CSV."""
        if lat_now is None or lon_now is None:
            lat_now, lon_now = self.runner.proj_cache.xy_to_latlon(
                self.runner.kf.x[0, 0], self.runner.kf.x[1, 0],
                self.runner.lat0, self.runner.lon0,
            )

        if msl_now is None or agl_now is None:
            dem_now = self.runner.dem.sample_m(lat_now, lon_now) if self.runner.dem.ds else 0.0
            if dem_now is None:
                dem_now = 0.0

            msl_now = self.runner.kf.x[2, 0]
            agl_now = msl_now - dem_now

        frame_str = str(self.runner.state.vio_frame) if used_vo else ""

        vo_dx = vo_data.get("dx", np.nan) if vo_data else np.nan
        vo_dy = vo_data.get("dy", np.nan) if vo_data else np.nan
        vo_dz = vo_data.get("dz", np.nan) if vo_data else np.nan
        vo_r = vo_data.get("roll", np.nan) if vo_data else np.nan
        vo_p = vo_data.get("pitch", np.nan) if vo_data else np.nan
        vo_y = vo_data.get("yaw", np.nan) if vo_data else np.nan

        with open(self.runner.pose_csv, "a", newline="") as f:
            f.write(
                f"{t - self.runner.state.t0:.6f},{dt:.6f},{frame_str},"
                f"{self.runner.kf.x[0,0]:.3f},{self.runner.kf.x[1,0]:.3f},{msl_now:.3f},"
                f"{self.runner.kf.x[3,0]:.3f},{self.runner.kf.x[4,0]:.3f},{self.runner.kf.x[5,0]:.3f},"
                f"{lat_now:.8f},{lon_now:.8f},{agl_now:.3f},"
                f"{'' if np.isnan(vo_dx) else f'{vo_dx:.6f}'},"
                f"{'' if np.isnan(vo_dy) else f'{vo_dy:.6f}'},"
                f"{'' if np.isnan(vo_dz) else f'{vo_dz:.6f}'},"
                f"{'' if np.isnan(vo_r) else f'{vo_r:.3f}'},"
                f"{'' if np.isnan(vo_p) else f'{vo_p:.3f}'},"
                f"{'' if np.isnan(vo_y) else f'{vo_y:.3f}'}\n"
            )

    def write_benchmark_health_summary(self):
        """Write one-row benchmark health summary CSV for before/after comparison."""
        if not self.runner.benchmark_health_summary_csv:
            return

        projection_count = 0
        first_projection_time = float("nan")
        pcond_max_stats = float("nan")
        if self.runner.kf is not None and hasattr(self.runner.kf, "get_conditioning_stats"):
            try:
                cstats = self.runner.kf.get_conditioning_stats()
                projection_count = int(cstats.get("projection_count", 0))
                first_projection_time = float(cstats.get("first_projection_time", float("nan")))
                pcond_max_stats = float(cstats.get("max_cond_seen", float("nan")))
            except Exception:
                pass

        pcond_max_cov = float("nan")
        pmax_max = float("nan")
        cov_large_rate = float("nan")
        if self.runner.cov_health_csv and os.path.isfile(self.runner.cov_health_csv):
            try:
                cov_df = pd.read_csv(self.runner.cov_health_csv)
                if len(cov_df) > 0:
                    pcond_max_cov = float(pd.to_numeric(cov_df["p_cond"], errors="coerce").max())
                    pmax_max = float(pd.to_numeric(cov_df["p_max"], errors="coerce").max())
                    cov_large_rate = float(pd.to_numeric(cov_df["large_flag"], errors="coerce").mean())
            except Exception:
                pass

        pcond_max = pcond_max_stats
        if np.isfinite(pcond_max_cov):
            pcond_max = max(pcond_max, pcond_max_cov) if np.isfinite(pcond_max) else pcond_max_cov

        pos_rmse = float("nan")
        final_pos_err = float("nan")
        final_alt_err = float("nan")
        if self.runner.error_csv and os.path.isfile(self.runner.error_csv):
            try:
                err_df = pd.read_csv(self.runner.error_csv)
                if len(err_df) > 0:
                    pos_vals = pd.to_numeric(err_df["pos_error_m"], errors="coerce").to_numpy(dtype=float)
                    alt_vals = pd.to_numeric(err_df["alt_error_m"], errors="coerce").to_numpy(dtype=float)
                    pos_rmse = float(np.sqrt(np.nanmean(pos_vals ** 2)))
                    final_pos_err = float(pos_vals[-1])
                    final_alt_err = float(alt_vals[-1])
            except Exception:
                pass

        output_dir_norm = os.path.normpath(self.runner.config.output_dir)
        if os.path.basename(output_dir_norm) == "preintegration":
            run_id = os.path.basename(os.path.dirname(output_dir_norm))
        else:
            run_id = os.path.basename(output_dir_norm)
        if not run_id:
            run_id = output_dir_norm

        append_benchmark_health_summary(
            self.runner.benchmark_health_summary_csv,
            run_id=run_id,
            projection_count=projection_count,
            first_projection_time=first_projection_time,
            pcond_max=pcond_max,
            pmax_max=pmax_max,
            cov_large_rate=cov_large_rate,
            pos_rmse=pos_rmse,
            final_pos_err=final_pos_err,
            final_alt_err=final_alt_err,
        )
        print(
            f"[SUMMARY] projection_count={projection_count}, "
            f"first_projection_time={first_projection_time:.3f}, "
            f"pcond_max={pcond_max:.2e}, cov_large_rate={cov_large_rate:.3f}"
        )

    def print_summary(self):
        """Print final summary statistics."""
        print("\n\n--- Done ---")
        print(f"Total IMU samples: {len(self.runner.imu)}")
        print(f"Images used: {self.runner.state.vio_frame}")
        print(f"VPS used: {self.runner.state.vps_idx}")
        print(
            f"Magnetometer: {self.runner.state.mag_updates} updates | "
            f"{self.runner.state.mag_rejects} rejected"
        )
        print(
            f"ZUPT: {self.runner.state.zupt_applied} applied | "
            f"{self.runner.state.zupt_rejected} rejected | "
            f"{self.runner.state.zupt_detected} detected"
        )

        if self.runner.kf is not None and hasattr(self.runner.kf, "get_cov_growth_summary"):
            cov_summary = self.runner.kf.get_cov_growth_summary()
            if len(cov_summary) > 0:
                print("\nCovariance Growth Sources (top 8):")
                for row in cov_summary[:8]:
                    print(
                        f"  {row['update_type']}: samples={row['samples']}, "
                        f"growth={row['growth_events']} ({100.0 * row['growth_rate']:.1f}%), "
                        f"largeP={row['large_events']} ({100.0 * row['large_rate']:.1f}%), "
                        f"max|P|={row['max_pmax']:.2e}"
                    )

        print_msckf_stats()
        print_error_statistics(self.runner.error_csv)
        self.write_benchmark_health_summary()
