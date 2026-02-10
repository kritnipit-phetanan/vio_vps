"""
Event-driven VIO Loop Implementation (v3.8.0)

This module implements the event-driven architecture for sensor fusion,
following the OpenVINS/PX4 EKF2 pattern:

Architecture Overview:
----------------------
1. **Event Scheduler**: Priority queue orders all sensor events by timestamp
2. **Propagate-to-Event**: State propagates EXACTLY to measurement time before update
3. **Fast-Propagate Output**: Separate output predictor for low-latency logging

Key Principle:
  "Before ANY update, filter_time == measurement_time ALWAYS"

This differs from IMU-driven mode where:
- IMU-driven: Iterate all IMU → poll sensors → state can lag by ~2.5ms
- Event-driven: Find next event → propagate to it → apply update → zero lag

References:
-----------
[1] OpenVINS Propagator: select IMU between time0-time1 + interpolate
    https://docs.openvins.com/classov__msckf_1_1Propagator.html
[2] PX4 EKF2 Output Predictor: IMU buffered propagate from fusion to current
    https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf
[3] OpenVINS fast_state_propagate for low-latency output
    https://docs.ros.org/en/melodic/api/ov_msckf/html/ROS1Visualizer_8cpp_source.html

Author: VIO project
"""

import time
import heapq
import copy
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from .imu_preintegration import IMUPreintegration
from .propagation import (
    process_imu, get_flight_phase, VibrationDetector
)
from .magnetometer import reset_mag_filter_state, set_mag_constants
from .trn import create_trn_from_config
from .data_loaders import ProjectionCache
from .output_utils import log_state_debug
from .ekf import ExtendedKalmanFilter


# =============================================================================
# Event Types and Priority
# =============================================================================

class EventType(IntEnum):
    """
    Sensor event types with natural ordering.
    
    Lower value = higher priority when timestamps equal.
    
    CRITICAL FIX: IMU must come FIRST at timestamp t!
    Reason: Sensor updates at t need IMU data up to t for propagate_to_event().
    Wrong order: sensor@t → propagate(no IMU@t yet) → extrapolate fallback ❌
    Correct order: IMU@t → sensor@t → propagate(has IMU@t) → exact ✅
    """
    IMU = 1      # Highest priority: MUST process first for propagation data
    VPS = 2      # Position correction (needs propagate to t first)
    MAG = 3      # Magnetometer yaw (needs propagate to t first)
    CAMERA = 4   # Visual features / MSCKF (needs propagate to t first)
    DEM = 5      # Terrain height (needs propagate to t first)


@dataclass
class SensorEvent:
    """
    A timestamped sensor event for the priority queue.
    
    Attributes:
        timestamp: Event time (seconds)
        event_type: Type of sensor event
        data: Sensor-specific data (IMU record, image path, etc.)
        index: Index into original sensor array (for tracking progress)
    """
    timestamp: float
    event_type: EventType
    data: Any
    index: int
    
    def __lt__(self, other):
        """Priority queue ordering: time first, then event type."""
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.event_type < other.event_type


# =============================================================================
# IMU Interpolation (OpenVINS-style)
# =============================================================================

def interpolate_imu(imu_before, imu_after, t_target: float):
    """
    Linear interpolation of IMU measurement at target time.
    
    Following OpenVINS approach:
    - Linear interpolation for gyro and accel
    - Timestamp must be within [imu_before.t, imu_after.t]
    
    Args:
        imu_before: IMU record before target time
        imu_after: IMU record after target time
        t_target: Target timestamp for interpolation
    
    Returns:
        Interpolated IMU record (namedtuple or dataclass)
    """
    if imu_before is None or imu_after is None:
        raise ValueError("Cannot interpolate with None IMU records")
    
    if t_target < imu_before.t or t_target > imu_after.t:
        raise ValueError(f"Target time {t_target} not in range [{imu_before.t}, {imu_after.t}]")
    
    # Linear interpolation factor
    dt_total = imu_after.t - imu_before.t
    if dt_total < 1e-10:
        # Times are equal, return before
        return imu_before
    
    alpha = (t_target - imu_before.t) / dt_total
    
    # Interpolate gyro and accel
    ang_interp = (1.0 - alpha) * imu_before.ang + alpha * imu_after.ang
    lin_interp = (1.0 - alpha) * imu_before.lin + alpha * imu_after.lin
    
    # Create interpolated record (assume same structure as input)
    from .data_loaders import IMURecord
    return IMURecord(
        t=t_target,
        ang=ang_interp,
        lin=lin_interp
    )


def select_imu_segment(imu_buffer: 'IMUBuffer', t_start: float, t_end: float,
                       interpolate_endpoints: bool = True) -> List:
    """
    Select IMU segment for propagation with endpoint interpolation.
    
    OpenVINS-style IMU selection:
    1. Find all IMU samples in (t_start, t_end)
    2. Optionally interpolate IMU at exact t_start and t_end
    3. Return [imu@t_start, imu_samples..., imu@t_end]
    
    This ensures propagation integrates from EXACT start to EXACT end time.
    
    Args:
        imu_buffer: IMU buffer containing measurements
        t_start: Start time (will interpolate IMU at this time)
        t_end: End time (will interpolate IMU at this time)
        interpolate_endpoints: If True, add interpolated IMU at endpoints
    
    Returns:
        List of IMU records for propagation [oldest ... newest]
    
    Raises:
        ValueError: If insufficient IMU data for interpolation
    """
    result = []
    
    # Find IMU before and after t_start for interpolation
    imu_before_start = None
    imu_after_start = None
    
    for rec in imu_buffer.buffer:
        if rec.t <= t_start:
            imu_before_start = rec
        elif rec.t > t_start and imu_after_start is None:
            imu_after_start = rec
            break
    
    # Add interpolated IMU at t_start
    if interpolate_endpoints:
        if imu_before_start is None:
            raise ValueError(f"No IMU data before t_start={t_start:.6f}")
        
        if imu_after_start is None or t_start >= imu_after_start.t:
            # Edge case: t_start is at or after last IMU, use last IMU
            if imu_before_start.t == t_start:
                result.append(imu_before_start)
            else:
                raise ValueError(f"No IMU data after t_start={t_start:.6f} for interpolation")
        elif imu_before_start.t == t_start:
            # Exact match, no interpolation needed
            result.append(imu_before_start)
        else:
            # Interpolate at t_start
            imu_start = interpolate_imu(imu_before_start, imu_after_start, t_start)
            result.append(imu_start)
    
    # Add all IMU samples strictly between (t_start, t_end)
    for rec in imu_buffer.buffer:
        if t_start < rec.t < t_end:
            result.append(rec)
    
    # Find IMU before and after t_end for interpolation
    imu_before_end = None
    imu_after_end = None
    
    for rec in imu_buffer.buffer:
        if rec.t <= t_end:
            imu_before_end = rec
        elif rec.t > t_end and imu_after_end is None:
            imu_after_end = rec
            break
    
    # Add interpolated IMU at t_end
    if interpolate_endpoints:
        if imu_before_end is None:
            raise ValueError(f"No IMU data before t_end={t_end:.6f}")
        
        if imu_after_end is None:
            # Edge case: t_end is at or after last IMU
            if imu_before_end.t == t_end:
                # Only add if not already added
                if len(result) == 0 or result[-1].t != t_end:
                    result.append(imu_before_end)
            else:
                # Cannot interpolate, use last IMU (extrapolation warning)
                print(f"[WARNING] Cannot interpolate at t_end={t_end:.6f}, using last IMU at {imu_before_end.t:.6f}")
                if len(result) == 0 or result[-1].t != imu_before_end.t:
                    result.append(imu_before_end)
        elif imu_before_end.t == t_end:
            # Exact match
            if len(result) == 0 or result[-1].t != t_end:
                result.append(imu_before_end)
        else:
            # Interpolate at t_end
            imu_end = interpolate_imu(imu_before_end, imu_after_end, t_end)
            result.append(imu_end)
    
    return result


# =============================================================================
# IMU Buffer for Propagation
# =============================================================================

class IMUBuffer:
    """
    Ring buffer for IMU measurements with efficient range queries.
    
    Optimized for event-driven propagation:
    - O(1) append
    - O(1) range query with index tracking (avoids O(N) scan)
    - Automatic pruning of old samples
    
    Thread-safe for single producer / single consumer.
    """
    
    def __init__(self, max_size: int = 2000):
        """
        Initialize IMU buffer.
        
        Args:
            max_size: Maximum samples to keep (~5 seconds @ 400Hz)
        """
        self.buffer = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.last_query_idx = 0  # Track last consumed index for O(1) range query
    
    def append(self, imu_record):
        """Add IMU measurement to buffer."""
        self.buffer.append(imu_record)
        self.timestamps.append(imu_record.t)
    
    def get_range(self, t_start: float, t_end: float) -> List:
        """
        Get IMU measurements in time range (t_start, t_end].
        
        OPTIMIZED: Use last_query_idx to avoid full scan (O(1) amortized).
        
        Args:
            t_start: Start time (exclusive)
            t_end: End time (inclusive)
        
        Returns:
            List of IMU records in range
        """
        result = []
        
        # Start from last queried index (amortized O(1))
        start_idx = max(0, self.last_query_idx)
        
        for i in range(start_idx, len(self.buffer)):
            rec = self.buffer[i]
            if rec.t > t_end:
                break
            if t_start < rec.t <= t_end:
                result.append(rec)
                self.last_query_idx = i  # Update for next query
        
        return result
    
    def get_closest_before(self, t: float):
        """Get IMU measurement closest to but not after t."""
        closest = None
        for rec in self.buffer:
            if rec.t <= t:
                closest = rec
            else:
                break
        return closest
    
    def prune_before(self, t: float):
        """Remove samples before timestamp t (optional, deque handles this)."""
        while len(self.buffer) > 0 and self.buffer[0].t < t:
            self.buffer.popleft()
            self.timestamps.popleft()
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# Event Scheduler
# =============================================================================

class EventScheduler:
    """
    Priority-queue based event scheduler for sensor fusion.
    
    Manages events from all sensors and returns them in chronological order.
    Supports dynamic event insertion (for sensors arriving asynchronously).
    """
    
    def __init__(self):
        """Initialize empty event queue."""
        self.queue = []  # heapq
        self.processed_count = {et: 0 for et in EventType}
    
    def add_event(self, event: SensorEvent):
        """Add event to priority queue."""
        heapq.heappush(self.queue, event)
    
    def add_events_batch(self, events: List[SensorEvent]):
        """Add multiple events efficiently."""
        for ev in events:
            heapq.heappush(self.queue, ev)
    
    def peek_next(self) -> Optional[SensorEvent]:
        """Look at next event without removing it."""
        if len(self.queue) == 0:
            return None
        return self.queue[0]
    
    def pop_next(self) -> Optional[SensorEvent]:
        """Remove and return next event."""
        if len(self.queue) == 0:
            return None
        event = heapq.heappop(self.queue)
        self.processed_count[event.event_type] += 1
        return event
    
    def has_events(self) -> bool:
        """Check if queue has remaining events."""
        return len(self.queue) > 0
    
    def __len__(self):
        return len(self.queue)


# =============================================================================
# Filter Time Tracker
# =============================================================================

@dataclass
class FilterState:
    """
    Tracks the "reference time" of the filter with delayed fusion horizon.
    
    CRITICAL: kf.x and kf.P are ONLY valid at filter_time!
    Any update must propagate to measurement_time first.
    
    Delayed Fusion Pattern (PX4 EKF2-style):
    - fusion_time: Time where filter fuses measurements (delayed)
    - output_time: Current time for output (ahead of fusion)
    - fusion_delay: Time lag for handling out-of-order measurements
    
    Attributes:
        filter_time: Current fusion time of kf.x/kf.P (delayed)
        output_time: Current output time (real-time)
        fusion_delay: Delay in seconds (default 50ms)
        t0: Initial timestamp (for relative time logging)
        imu_buffer: Ring buffer of recent IMU measurements
        ongoing_preint: Preintegration buffer for MSCKF Jacobians
        measurement_buffer: Buffer for out-of-order measurements
    """
    filter_time: float = 0.0
    output_time: float = 0.0
    fusion_delay: float = 0.05  # 50ms delay for out-of-order handling
    t0: float = 0.0
    imu_buffer: IMUBuffer = None
    ongoing_preint: IMUPreintegration = None
    measurement_buffer: List = None  # Buffer for measurements arriving late
    
    def __post_init__(self):
        if self.imu_buffer is None:
            self.imu_buffer = IMUBuffer()
        if self.measurement_buffer is None:
            self.measurement_buffer = []


# =============================================================================
# Propagate-to-Event Functions
# =============================================================================

def propagate_to_event(kf: ExtendedKalmanFilter,
                       filter_state: FilterState,
                       target_time: float,
                       imu_params: dict,
                       estimate_imu_bias: bool = False) -> bool:
    """
    Propagate filter state EXACTLY to target timestamp (Pure Event-Driven).
    
    NEW APPROACH (OpenVINS-style):
    1. Select IMU segment from buffer: [filter_time, target_time]
    2. Endpoints are INTERPOLATED for exact times (no fractional steps)
    3. Integrate each IMU sample sequentially
    4. Update filter_time to target_time
    5. Accumulate preintegration for MSCKF Jacobians
    
    After this call: filter_time == target_time EXACTLY.
    
    CRITICAL: No IMU events needed! IMU buffer is data source only.
    
    Args:
        kf: Extended Kalman Filter
        filter_state: Contains filter_time, imu_buffer, ongoing_preint
        target_time: Desired timestamp to propagate to
        imu_params: IMU noise parameters
        estimate_imu_bias: Whether bias estimation is enabled
    
    Returns:
        True if propagation succeeded
    
    Raises:
        ValueError: If insufficient IMU data for propagation
    """
    current_time = filter_state.filter_time
    
    if target_time <= current_time:
        # Already at or past target time
        return True
    
    # Select IMU segment with endpoint interpolation
    try:
        imu_samples = select_imu_segment(
            filter_state.imu_buffer,
            t_start=current_time,
            t_end=target_time,
            interpolate_endpoints=True
        )
    except ValueError as e:
        print(f"[PROPAGATE] ERROR: Cannot select IMU segment [{current_time:.6f}, {target_time:.6f}]")
        print(f"  Reason: {e}")
        print(f"  Buffer size: {len(filter_state.imu_buffer)}")
        if len(filter_state.imu_buffer) > 0:
            print(f"  Buffer range: [{filter_state.imu_buffer.buffer[0].t:.6f}, {filter_state.imu_buffer.buffer[-1].t:.6f}]")
        raise  # Re-raise to stop execution (no silent fallback)
    
    if len(imu_samples) == 0:
        # Should not happen after select_imu_segment, but check anyway
        raise ValueError(f"No IMU samples selected for propagation [{current_time:.6f}, {target_time:.6f}]")
    
    # Propagate through each IMU sample
    # First sample should be at current_time (interpolated)
    # Last sample should be at target_time (interpolated)
    t_prev = current_time
    
    for i, imu in enumerate(imu_samples):
        dt = imu.t - t_prev
        
        if dt < 0:
            raise ValueError(f"IMU samples not in chronological order: {t_prev:.6f} -> {imu.t:.6f}")
        
        if dt > 1e-9:  # Skip tiny timesteps
            _propagate_single_step(
                kf, imu, dt, filter_state,
                imu_params, estimate_imu_bias,
                target_time=imu.t
            )
        
        t_prev = imu.t
    
    # Verify we reached target time
    if abs(t_prev - target_time) > 1e-6:
        raise ValueError(
            f"Propagation ended at {t_prev:.6f}, expected {target_time:.6f} "
            f"(error: {abs(t_prev - target_time):.9f}s)"
        )
    
    # Update filter time
    filter_state.filter_time = target_time
    
    return True


def _propagate_single_step(kf: ExtendedKalmanFilter,
                           imu_record,
                           dt: float,
                           filter_state: FilterState,
                           imu_params: dict,
                           estimate_imu_bias: bool,
                           target_time: float = None):
    """
    Propagate state by single IMU step.
    
    CRITICAL FIX: Must pass target_time (not imu_record.t) for fractional steps!
    
    Updates:
    1. kf.x (nominal state)
    2. kf.P (error-state covariance)
    3. ongoing_preint (for MSCKF Jacobians)
    
    Args:
        target_time: Actual target time for this step (for fractional propagation)
    """
    # Use target_time if provided (for fractional steps), otherwise use imu_record.t
    t_actual = target_time if target_time is not None else imu_record.t
    
    # Propagate state and covariance
    process_imu(
        kf, imu_record, dt,
        estimate_imu_bias=estimate_imu_bias,
        t=t_actual,  # FIXED: Use actual target time, not imu_record.t
        t0=filter_state.t0,
        imu_params=imu_params
    )
    
    # Accumulate preintegration for MSCKF Jacobians
    if filter_state.ongoing_preint is not None:
        filter_state.ongoing_preint.integrate_measurement(
            imu_record.ang, imu_record.lin, dt
        )


# =============================================================================
# Fast-Propagate Output Layer
# =============================================================================

def fast_propagate_output(kf: ExtendedKalmanFilter,
                          filter_state: FilterState,
                          output_time: float,
                          imu_params: dict,
                          propagate_covariance: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fast-propagate state from filter_time to output_time for logging.
    
    This is the "output predictor" pattern from PX4 EKF2:
    - Does NOT modify the main filter state (kf)
    - Uses IMU buffer to extrapolate from fusion horizon to current time
    - Provides low-latency output without disrupting fusion pipeline
    
    CRITICAL: This function uses a COPY of state, never modifies kf!
    
    NOTE: Currently NOT USED in run_event_driven_loop().
    Reason: We propagate at every IMU sample anyway (hybrid architecture).
    Future: Enable this for true event-driven (skip IMU propagation between events).
    
    Args:
        kf: Extended Kalman Filter (will NOT be modified)
        filter_state: Contains filter_time and imu_buffer
        output_time: Desired output timestamp
        imu_params: IMU noise parameters
        propagate_covariance: If True, also propagate P (slower)
    
    Returns:
        x_out: Propagated state vector (16x1)
        P_out: Propagated covariance (16x16) if propagate_covariance=True, else None
    """
    current_time = filter_state.filter_time
    
    if output_time <= current_time:
        # Already at or past output time - return current state
        return kf.x.copy(), kf.P.copy() if propagate_covariance else None
    
    # Make a lightweight copy of state
    x_out = kf.x.copy()
    P_out = kf.P.copy() if propagate_covariance else None
    
    # Get IMU samples for extrapolation
    imu_samples = filter_state.imu_buffer.get_range(current_time, output_time)
    
    if len(imu_samples) == 0:
        # No new IMU - return current state
        return x_out, P_out
    
    # Fast propagation (nominal state only, optionally covariance)
    t_prev = current_time
    for imu in imu_samples:
        dt = imu.t - t_prev
        if dt > 1e-6:
            _fast_propagate_step(x_out, P_out, imu, dt, imu_params, propagate_covariance)
        t_prev = imu.t
    
    # Handle fractional timestep
    if t_prev < output_time and len(imu_samples) > 0:
        last_imu = imu_samples[-1]
        dt_frac = output_time - t_prev
        _fast_propagate_step(x_out, P_out, last_imu, dt_frac, imu_params, propagate_covariance)
    
    return x_out, P_out


def _fast_propagate_step(x: np.ndarray, P: Optional[np.ndarray],
                         imu_record, dt: float,
                         imu_params: dict,
                         propagate_covariance: bool):
    """
    Single step of fast propagation for output predictor.
    
    Simpler than full process_imu() - focuses on speed for output layer.
    """
    # Extract state components
    p = x[0:3, 0]
    v = x[3:6, 0]
    q = x[6:10, 0]  # [w,x,y,z]
    bg = x[10:13, 0]
    ba = x[13:16, 0]
    
    # Bias-corrected IMU measurements
    gyro = imu_record.ang - bg
    accel = imu_record.lin - ba
    
    # Current rotation (body to world)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    R_wb = R_scipy.from_quat(q_xyzw).as_matrix()
    
    # Gravity in world frame (ENU: Z up)
    g_world = np.array([0.0, 0.0, -imu_params.get('g_norm', 9.80665)])
    
    # Acceleration in world frame
    a_world = R_wb @ accel + g_world
    
    # Update position (simple Euler integration for speed)
    x[0:3, 0] = p + v * dt + 0.5 * a_world * dt**2
    
    # Update velocity
    x[3:6, 0] = v + a_world * dt
    
    # Update orientation (first-order quaternion integration)
    omega_mag = np.linalg.norm(gyro)
    if omega_mag > 1e-10:
        axis = gyro / omega_mag
        angle = omega_mag * dt
        dq = R_scipy.from_rotvec(axis * angle).as_quat()  # [x,y,z,w]
        q_new = R_scipy.from_quat(q_xyzw) * R_scipy.from_quat(dq)
        q_new_xyzw = q_new.as_quat()
        x[6:10, 0] = [q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]]
    
    # Optionally propagate covariance (slower but more accurate uncertainty)
    if propagate_covariance and P is not None:
        # Simplified covariance propagation (diagonal approximation for speed)
        # Full propagation would use process_imu() - this is just for output
        sigma_g = imu_params.get('gyr_n', 0.01)
        sigma_a = imu_params.get('acc_n', 0.1)
        
        # Add process noise to diagonal (approximate)
        P[0:3, 0:3] += np.eye(3) * (sigma_a * dt**2)**2
        P[3:6, 3:6] += np.eye(3) * (sigma_a * dt)**2
        P[6:9, 6:9] += np.eye(3) * (sigma_g * dt)**2


# =============================================================================
# Event Handlers
# =============================================================================

def handle_camera_event(runner, event: SensorEvent, filter_state: FilterState,
                        imu_params: dict) -> Tuple[bool, Optional[Dict]]:
    """
    Handle camera event: propagate to camera time, then process VIO.
    
    Args:
        runner: VIORunner instance
        event: Camera SensorEvent
        filter_state: Current filter state
        imu_params: IMU parameters
    
    Returns:
        (used_vo, vo_data) tuple
    """
    t_cam = event.timestamp
    
    # Step 1: Propagate to exact camera time
    propagate_to_event(
        runner.kf, filter_state, t_cam,
        imu_params, runner.config.estimate_imu_bias
    )
    
    # Step 2: Process VIO at exact timestamp
    # Get a dummy IMU record for rotation rate check
    last_imu = filter_state.imu_buffer.get_closest_before(t_cam)
    
    used_vo, vo_data = runner.process_vio(last_imu, t_cam, filter_state.ongoing_preint)
    
    return used_vo, vo_data


def handle_vps_event(runner, event: SensorEvent, filter_state: FilterState,
                     imu_params: dict):
    """
    Handle VPS event: propagate to VPS time, then apply update.
    """
    t_vps = event.timestamp
    
    # Propagate to exact VPS time
    propagate_to_event(
        runner.kf, filter_state, t_vps,
        imu_params, runner.config.estimate_imu_bias
    )
    
    # Apply VPS update (runner.process_vps already handles the actual update)
    # We need to call it at the exact timestamp
    runner._process_single_vps(event.data, t_vps)


def handle_magnetometer_event(runner, event: SensorEvent, filter_state: FilterState,
                              imu_params: dict):
    """
    Handle magnetometer event: propagate to mag time, then apply update.
    """
    t_mag = event.timestamp
    
    # Propagate to exact magnetometer time
    propagate_to_event(
        runner.kf, filter_state, t_mag,
        imu_params, runner.config.estimate_imu_bias
    )
    
    # Apply magnetometer update
    runner._process_single_mag(event.data, t_mag)


def handle_dem_event(runner, event: SensorEvent, filter_state: FilterState,
                     imu_params: dict):
    """
    Handle DEM height event: propagate to time, then apply update.
    """
    t_dem = event.timestamp
    
    # Propagate to exact time
    propagate_to_event(
        runner.kf, filter_state, t_dem,
        imu_params, runner.config.estimate_imu_bias
    )
    
    # Apply DEM height update
    runner.process_dem_height(t_dem)


# =============================================================================
# Main Event-Driven Loop
# =============================================================================

def run_event_driven_loop(runner):
    """
    Event-driven VIO loop (v3.8.0).
    
    Current Architecture (Hybrid: NOT pure event-driven):
    ======================================================
    1. Load all sensor data and create event queue
    2. Process events in chronological order (priority queue)
    3. IMU events: propagate state, update helpers, log at IMU rate
    4. Sensor events (VPS/MAG/CAM): propagate_to_event() then apply update
    5. DEM updates: polled during IMU events (position-dependent)
    
    Key guarantee: Before ANY sensor update, filter_time == measurement_time.
    
    CRITICAL LIMITATIONS:
    ====================
    ❌ Still propagates at EVERY IMU sample (400Hz) - no computation savings
    ❌ Adds overhead: priority queue + buffer scanning + event creation
    ❌ fast_propagate_output() not used (no output predictor benefit)
    ⚠️  Extrapolation fallback can hide timing bugs silently
    
    TODO: Migrate to Pure Event-Driven (OpenVINS-style)
    ===================================================
    Changes needed for true event-driven architecture:
    
    1. **Remove IMU from event queue**
       - Keep only: CAMERA, VPS, MAG (DEM optional)
       - IMU buffer is data source, not event source
    
    2. **Implement IMU segment selection + interpolation**
       - propagate_to(t_event) selects IMU in range [filter_time, t_event]
       - Interpolate IMU at exact t_event if between samples
       - See OpenVINS Propagator::select_imu_readings()
    
    3. **Add fusion time horizon**
       - State fuses at delayed horizon (e.g., t - 50ms)
       - Buffer out-of-order measurements
       - See PX4 EKF2 delayed fusion
    
    4. **Enable fast_propagate_output()**
       - Fusion at t_fusion → output at t_current via IMU extrapolation
       - Provides low-latency output without disrupting fusion
    
    5. **Remove/assert extrapolation fallback**
       - Should never trigger if architecture is correct
       - Add assertion/error instead of silent fallback
    
    6. **1-shot measurement consume**
       - Handler processes single measurement at exact timestamp
       - No while loops scanning forward
    
    Benefits of Pure Event-Driven:
    - Skip IMU propagation between sparse measurements (computation savings)
    - Better handling of out-of-order/delayed measurements
    - Proper separation: fusion time ≠ output time
    - Matches production systems (PX4 EKF2, OpenVINS)
    
    Args:
        runner: VIORunner instance with initialized config and state
    """
    # =========================================================================
    # Step 1: Initialize (same as IMU-driven mode)
    # =========================================================================
    runner.load_data()
    runner.initialize_ekf()
    runner.initialize_vio_frontend()
    runner._initialize_rectifier()
    runner._initialize_loop_closure()
    
    # Initialize magnetometer filter
    reset_mag_filter_state()
    mag_ema_alpha = runner.global_config.get('MAG_EMA_ALPHA', 0.3)
    mag_max_yaw_rate_deg = runner.global_config.get('MAG_MAX_YAW_RATE_DEG', 30.0)
    mag_gyro_threshold_deg = runner.global_config.get('MAG_GYRO_THRESHOLD_DEG', 10.0)
    mag_r_inflate = runner.global_config.get('MAG_R_INFLATE', 5.0)
    set_mag_constants(
        ema_alpha=mag_ema_alpha,
        max_yaw_rate=np.radians(mag_max_yaw_rate_deg),
        gyro_threshold=np.radians(mag_gyro_threshold_deg),
        consistency_r_inflate=mag_r_inflate
    )
    print(f"[MAG-FILTER] Initialized: EMA_α={mag_ema_alpha:.2f}")
    
    # Initialize vibration detector
    imu_params = runner.global_config.get('IMU_PARAMS', {})
    vib_buffer_size = runner.global_config.get('VIBRATION_WINDOW_SIZE', 50)
    vib_threshold_mult = runner.global_config.get('VIBRATION_THRESHOLD_MULT', 5.0)
    runner.vibration_detector = VibrationDetector(
        buffer_size=vib_buffer_size,
        threshold=imu_params.get('acc_n', 0.1) * vib_threshold_mult
    )
    
    # Initialize TRN
    if runner.global_config.get('TRN_ENABLED', False):
        runner.trn = create_trn_from_config(runner.dem, runner.global_config)
        if runner.trn:
            print(f"[TRN] Initialized")
    else:
        print("[TRN] Disabled")
    
    runner.initial_msl = runner.msl0
    runner.setup_output_files()
    
    # =========================================================================
    # Step 2: Create Filter State Tracker
    # =========================================================================
    filter_state = FilterState()
    filter_state.t0 = runner.imu[0].t
    filter_state.filter_time = filter_state.t0
    filter_state.imu_buffer = IMUBuffer(max_size=2000)
    
    # Initialize preintegration cache for MSCKF
    filter_state.ongoing_preint = IMUPreintegration(
        bg=runner.kf.x[10:13, 0].reshape(3,),
        ba=runner.kf.x[13:16, 0].reshape(3,),
        sigma_g=imu_params.get('gyr_n', 0.01),
        sigma_a=imu_params.get('acc_n', 0.1),
        sigma_bg=imu_params.get('gyr_w', 0.0001),
        sigma_ba=imu_params.get('acc_w', 0.001)
    )
    
    # Also store in runner.state for compatibility
    runner.state.t0 = filter_state.t0
    runner.state.last_t = filter_state.t0
    
    # =========================================================================
    # Step 3: Build Event Queue
    # =========================================================================
    scheduler = EventScheduler()
    
    print("\n[EVENT-DRIVEN] Building event queue...")
    
    # =========================================================================
    # PURE EVENT-DRIVEN: IMU is DATA SOURCE, not EVENT SOURCE
    # =========================================================================
    # Load all IMU into buffer (no events generated)
    # propagate_to_event() will select IMU segments from buffer as needed
    print(f"[EVENT-DRIVEN] Loading {len(runner.imu)} IMU samples into buffer...")
    for imu_rec in runner.imu:
        filter_state.imu_buffer.append(imu_rec)
    print(f"[EVENT-DRIVEN] IMU buffer ready: {len(filter_state.imu_buffer)} samples")
    print(f"  Time range: [{filter_state.imu_buffer.buffer[0].t:.3f}, {filter_state.imu_buffer.buffer[-1].t:.3f}]")
    
    # Add camera events
    for i, img_rec in enumerate(runner.imgs):
        scheduler.add_event(SensorEvent(
            timestamp=img_rec.t,
            event_type=EventType.CAMERA,
            data=img_rec,
            index=i
        ))
    
    # Add VPS events
    for i, vps_rec in enumerate(runner.vps_list):
        scheduler.add_event(SensorEvent(
            timestamp=vps_rec.t,
            event_type=EventType.VPS,
            data=vps_rec,
            index=i
        ))
    
    # Add magnetometer events
    for i, mag_rec in enumerate(runner.mag_list):
        scheduler.add_event(SensorEvent(
            timestamp=mag_rec.t,
            event_type=EventType.MAG,
            data=mag_rec,
            index=i
        ))
    
    # NOTE: DEM height updates are NOT event-based
    # DEM is sampled at current position during IMU events (polling pattern)
    # Reason: DEM updates depend on state position, not fixed timestamps
    # Alternative: Could generate periodic DEM events (e.g., every 0.1s)
    
    print(f"[EVENT-DRIVEN] Total events: {len(scheduler)} (sensors only)")
    print(f"  Camera: {len(runner.imgs)}, VPS: {len(runner.vps_list)}, MAG: {len(runner.mag_list)}")
    print(f"  IMU: {len(runner.imu)} samples (data source, not events)")
    print(f"  DEM: On-demand at sensor events (position-dependent)")
    
    # =========================================================================
    # Step 4: Main Event Loop (Pure Event-Driven with Fusion Delay)
    # =========================================================================
    print("\n=== Running (Pure Event-driven mode) ===")
    print(f"[FUSION] Fusion delay: {filter_state.fusion_delay*1000:.1f}ms")
    tic_all = time.time()
    
    last_a_world = np.array([0.0, 0.0, 0.0])
    last_output_time = filter_state.t0
    output_interval = 0.01  # Output at 100Hz (reduced from 400Hz for efficiency)
    
    event_count = 0
    
    # Initialize output_time to first IMU time
    filter_state.output_time = filter_state.t0
    
    while scheduler.has_events():
        event = scheduler.pop_next()
        t_measurement = event.timestamp
        event_count += 1
        
        tic_iter = time.time()
        
        # Update output time to current measurement time
        filter_state.output_time = t_measurement
        
        # Calculate fusion time (delayed by fusion_delay)
        t_fusion = t_measurement - filter_state.fusion_delay
        
        # Skip events before we have enough IMU data
        if t_fusion < filter_state.t0:
            continue
        
        # ---------------------------------------------------------------------
        # Process Fusion (at delayed time)
        # ---------------------------------------------------------------------
        # Propagate filter to fusion time (if needed)
        if t_fusion > filter_state.filter_time:
            try:
                propagate_to_event(
                    runner.kf, filter_state, t_fusion,
                    imu_params, runner.config.estimate_imu_bias
                )
            except ValueError as e:
                print(f"[ERROR] Failed to propagate to fusion time {t_fusion:.6f}: {e}")
                continue
        
        # ---------------------------------------------------------------------
        # Handle event by type (at fusion time)
        # ---------------------------------------------------------------------
        if event.event_type == EventType.CAMERA:
            # Camera event: already at fusion time, process VIO
            runner.state.img_idx = event.index
            try:
                used_vo, vo_data = handle_camera_event(runner, event, filter_state, imu_params)
            except Exception as e:
                print(f"[ERROR] Camera event failed at {t_measurement:.6f}: {e}")
            
        elif event.event_type == EventType.VPS:
            # VPS event: already at fusion time, apply update
            try:
                handle_vps_event(runner, event, filter_state, imu_params)
                runner.state.vps_idx = event.index + 1
            except Exception as e:
                print(f"[ERROR] VPS event failed at {t_measurement:.6f}: {e}")
            
        elif event.event_type == EventType.MAG:
            # Magnetometer event: already at fusion time, apply update
            try:
                handle_magnetometer_event(runner, event, filter_state, imu_params)
                runner.state.mag_idx = event.index + 1
            except Exception as e:
                print(f"[ERROR] MAG event failed at {t_measurement:.6f}: {e}")
        
        # ---------------------------------------------------------------------
        # Output logging (at output_time using fast propagation)
        # ---------------------------------------------------------------------
        if filter_state.output_time - last_output_time >= output_interval:
            # Fast-propagate from fusion_time to output_time for logging
            try:
                x_output, _ = fast_propagate_output(
                    runner.kf,
                    filter_state,
                    filter_state.output_time,
                    imu_params,
                    propagate_covariance=False
                )
            except Exception as e:
                print(f"[WARNING] Fast propagation failed: {e}, using fusion state")
                x_output = runner.kf.x
            
            # Get current position for logging (using output state)
            lat_now, lon_now = runner.proj_cache.xy_to_latlon(
                x_output[0, 0], x_output[1, 0],
                runner.lat0, runner.lon0
            )
            dem_now = runner.dem.sample_m(lat_now, lon_now) if runner.dem.ds else 0.0
            if dem_now is None:
                dem_now = 0.0
            
            msl_now = x_output[2, 0]
            agl_now = msl_now - dem_now
            
            # Save current fusion state and temporarily swap for logging
            x_backup = runner.kf.x.copy()
            runner.kf.x = x_output
            
            # Log pose (using output state at real-time)
            dt = filter_state.output_time - runner.state.last_t
            used_vo = event.event_type == EventType.CAMERA
            runner.log_pose(filter_state.output_time, dt, used_vo, None, msl_now, agl_now, lat_now, lon_now)
            
            # Log error
            runner.log_error(filter_state.output_time)
            
            # Restore fusion state
            runner.kf.x = x_backup
            
            # Log inference timing
            toc_iter = time.time()
            with open(runner.inf_csv, "a", newline="") as f:
                dt_proc = toc_iter - tic_iter
                fps = (1.0 / dt_proc) if dt_proc > 0 else 0.0
                f.write(f"{event_count},{dt_proc:.6f},{fps:.2f}\n")
            
            runner.state.last_t = filter_state.output_time
            last_output_time = filter_state.output_time
        
        # Progress display
        if event_count % 1000 == 0:
            time_elapsed = t_measurement - filter_state.t0
            speed_ms = float(np.linalg.norm(runner.kf.x[3:6, 0]))
            fusion_lag = filter_state.output_time - filter_state.filter_time
            print(f"t={time_elapsed:8.3f}s | speed={speed_ms*3.6:5.1f}km/h | "
                  f"lag={fusion_lag*1000:.1f}ms | events={event_count}", end="\r")
    
    # =========================================================================
    # Step 5: Finish
    # =========================================================================
    toc_all = time.time()
    
    runner.print_summary()
    print(f"\n[PURE EVENT-DRIVEN] Processed {event_count} sensor events")
    print(f"  Camera: {scheduler.processed_count.get(EventType.CAMERA, 0)}")
    print(f"  VPS: {scheduler.processed_count.get(EventType.VPS, 0)}")
    print(f"  MAG: {scheduler.processed_count.get(EventType.MAG, 0)}")
    print(f"  IMU: {len(filter_state.imu_buffer)} samples (data source only)")
    
    print(f"\n[FUSION TIMING]")
    print(f"  Delay: {filter_state.fusion_delay*1000:.1f}ms")
    print(f"  Final fusion time: {filter_state.filter_time:.3f}s")
    print(f"  Final output time: {filter_state.output_time:.3f}s")
    print(f"  Final lag: {(filter_state.output_time - filter_state.filter_time)*1000:.1f}ms")
    
    print(f"\n=== Finished in {toc_all - tic_all:.2f} seconds ===")
