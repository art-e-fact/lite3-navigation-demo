"""Reusable harness for launching and managing simulation + control processes in tests."""

import atexit
import enum
import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, NamedTuple

import rclpy
import rclpy.context
import yaml
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PlanMsg
from rclpy.executors import SingleThreadedExecutor
from rosgraph_msgs.msg import Clock


STOP_TIMEOUT_SEC = 10.0
KILL_TIMEOUT_SEC = 5.0
SPIN_TIMEOUT_SEC = 0.1  # how long one executor spin waits for a message
WAIT_POLL_SEC = 0.05  # how often wait() re-checks its predicate
SPIN_JOIN_TIMEOUT_SEC = 5.0  # how long stop() waits for the spin thread to finish
LOG_TAIL_LINES = 80  # enough to show a crash's traceback in a failure message
DOMAIN_ID_BASE = 200  # ROS domain ids from here up, clear of the default 0
DOMAIN_ID_SPAN = 30  # the pid picks one of this many, so parallel runs rarely collide
ODOM_QUEUE_DEPTH = 20  # odom is the fastest topic, so it gets the deepest queue
CLOCK_QUEUE_DEPTH = 10
PLAN_QUEUE_DEPTH = 10
GOAL_ROUND_DIGITS = 1  # plans are keyed by goal rounded to 0.1 m (see _plans)
NANOSECONDS_PER_SECOND = 1e9
# Signals sent to a process group in turn, each with how long to wait for the exit.
STOP_SIGNALS = ((signal.SIGINT, STOP_TIMEOUT_SEC), (signal.SIGTERM, KILL_TIMEOUT_SEC), (signal.SIGKILL, KILL_TIMEOUT_SEC))

_PROCESS_GROUP_IDS: dict[int, int] = {}  # pid -> pgid of every process this module started


class TraceSample(NamedTuple):
    """One /odom message, as the reports consume it."""

    sim_time_s: float
    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float


class StopReason(enum.Enum):
    """Reason the harness stopped spinning."""

    PREDICATE = 'predicate'
    TIMEOUT = 'timeout'
    SIM_TIMEOUT = 'sim_timeout'
    STALLED = 'stalled'
    SIM_EXITED = 'sim_exited'
    CONTROL_EXITED = 'control_exited'
    STOPPED = 'stopped'


def _killpg(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def _kill(pid: int, signum: int) -> None:
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        pass


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / NANOSECONDS_PER_SECOND


def _cleanup_process_groups() -> None:
    for pgid in set(_PROCESS_GROUP_IDS.values()):
        _killpg(pgid, signal.SIGKILL)
    if _INITIAL_RERUN_VIEWER_PIDS is not None:
        _terminate_rerun_viewers(_rerun_viewer_pids() - _INITIAL_RERUN_VIEWER_PIDS)


atexit.register(_cleanup_process_groups)


# Rerun viewer cleanup
VIEWER_STOP_TIMEOUT_SEC = 3.0
VIEWER_POLL_SEC = 0.1
_INITIAL_RERUN_VIEWER_PIDS: set[int] | None = None


def _is_rerun_viewer_argv(argv: list[str]) -> bool:
    if len(argv) < 2:
        return False
    if not any(arg == '--port' or arg.startswith('--port=') for arg in argv):
        return False
    return any(Path(arg).name == 'rerun' for arg in argv[:2])


def _rerun_viewer_pids() -> set[int]:
    """PIDs of running Rerun viewer processes (Linux via /proc, otherwise via ps)."""
    pids: set[int] = set()
    proc_dir = Path('/proc')
    if proc_dir.is_dir():
        for entry in proc_dir.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / 'cmdline').read_bytes()
            except OSError:
                continue
            argv = [part.decode('utf-8', errors='replace') for part in raw.split(b'\0') if part]
            if _is_rerun_viewer_argv(argv):
                pids.add(int(entry.name))
        return pids
    try:
        output = subprocess.run(
            ['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return pids
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and _is_rerun_viewer_argv(parts[1].split()):
            pids.add(int(parts[0]))
    return pids


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_rerun_viewers(pids: set[int]) -> None:
    if not pids:
        return
    for pid in pids:
        _kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + VIEWER_STOP_TIMEOUT_SEC
    while time.monotonic() < deadline and any(_pid_alive(pid) for pid in pids):
        time.sleep(VIEWER_POLL_SEC)
    for pid in pids:
        if _pid_alive(pid):
            _kill(pid, signal.SIGKILL)


def _tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    if not path.exists():
        return ''
    return ''.join(
        path.read_text(encoding='utf-8', errors='replace').splitlines(keepends=True)[-lines:]
    )


def _start_process(
    cmd: list[str],
    log_path: Path,
    domain_id: str,
    cwd: str,
    extra_env: dict | None = None,
):
    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = domain_id
    env['PYTHONUNBUFFERED'] = '1'
    if extra_env:
        env.update(extra_env)
    log_file = log_path.open('w', encoding='utf-8')
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
    _PROCESS_GROUP_IDS[process.pid] = os.getpgid(process.pid)
    return process, log_file


def _stop_process(process, log_file) -> None:
    pgid = _PROCESS_GROUP_IDS.get(process.pid)
    if pgid is None:
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            pgid = None
    try:
        # Escalate SIGINT -> SIGTERM -> SIGKILL, waiting after each; the last wait may raise.
        for signum, timeout_sec in STOP_SIGNALS:
            if pgid is not None:
                _killpg(pgid, signum)
            if process.poll() is not None:
                break
            try:
                process.wait(timeout=timeout_sec)
                break
            except subprocess.TimeoutExpired:
                if signum == signal.SIGKILL:
                    raise
        if pgid is not None:
            _killpg(pgid, signal.SIGKILL)  # any children left in the group
    finally:
        if pgid is not None and process.poll() is not None:
            _PROCESS_GROUP_IDS.pop(process.pid, None)
        log_file.close()


class SimControlHarness:
    """Launch and manage simulation and control processes for integration tests.

    Subscribes to ``/odom`` for distance tracking and ``/clock`` for sim-time
    measurement. A background spin thread runs all checks and stops the harness
    automatically when a deadline or failure condition is reached.
    """

    def __init__(
        self,
        sim_config: dict,
        *,
        config_path: Path,
        log_dir: Path,
        repo_root: Path,
        sim_package_root: Path,
        control_launch_args: dict[str, str],
        control_package: str = 'nav2_demo',
        control_launch_file: str = 'nav2_route.launch.py',
        domain_id: str | None = None,
        max_runtime_sec: float,
        sim_timeout_sec: float | None = None,
        stall_timeout_sec: float | None = None,
        stall_min_step_m: float = 0.01,
        odom_topic: str = '/odom',
        clock_topic: str = '/clock',
        max_odom_step_m: float = 1.0,
        sim_pixi_env: str | None = None,
        sim_ready_timeout_sec: float = 180.0,
        sim_extra_args: list[str] | None = None,
    ):
        """Initialise the harness; call ``start()`` or use as a context manager to run it.

        The control stack is launched only once the simulator publishes its first
        ``/clock`` message (or after ``sim_ready_timeout_sec``), so the controller never
        starts against a simulator that is still loading or JIT-compiling kernels.

        ``sim_extra_args`` are appended to the simulator command line, for example
        ``["--set", "robot.start_pose.x=1.0"]`` to override a config field.
        """
        self._sim_config = sim_config
        self._sim_extra_args = list(sim_extra_args or [])
        self._sim_ready_timeout_sec = sim_ready_timeout_sec
        self._config_path = Path(config_path)
        self._log_dir = Path(log_dir)
        self._repo_root = Path(repo_root)
        self._sim_package_root = Path(sim_package_root)
        self._control_launch_args = control_launch_args
        self._control_package = control_package
        self._control_launch_file = control_launch_file
        self._domain_id = (
            domain_id if domain_id is not None else str(DOMAIN_ID_BASE + (os.getpid() % DOMAIN_ID_SPAN))
        )
        self._sim_pixi_env = sim_pixi_env or subprocess.run(
            ['scripts/sim_pixi_env.sh'],
            cwd=str(self._repo_root),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self._max_runtime_sec = max_runtime_sec
        self._sim_timeout_sec = sim_timeout_sec
        self._stall_timeout_sec = stall_timeout_sec
        self._stall_min_step_m = stall_min_step_m
        self._odom_topic = odom_topic
        self._clock_topic = clock_topic
        self._max_odom_step_m = max_odom_step_m

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._spin_thread: threading.Thread | None = None

        self._sim_proc = None
        self._sim_log_file = None
        self._sim_log_path: Path | None = None
        self._control_proc = None
        self._control_log_file = None
        self._control_log_path: Path | None = None

        self._context: rclpy.context.Context | None = None
        self._node = None
        self._executor: SingleThreadedExecutor | None = None

        self._total_distance_m = 0.0
        self._message_count = 0
        self._trace: list[TraceSample] = []  # one per odom message, for reports
        # Nav2's first plan for each goal, keyed by the goal (x, y) rounded to 0.1 m:
        # the planner republishes every cycle from wherever the robot is, so later
        # plans for the same goal are not the route it meant to drive.
        self._plans: dict[tuple[float, float], list[tuple[float, float]]] = {}
        self._previous_xy: tuple[float, float] | None = None
        self._last_xy: tuple[float, float] | None = None
        self._first_clock_sec: float | None = None
        self._last_clock_sec: float | None = None
        self._first_clock_wall_time: float | None = None
        self._last_clock_wall_time: float | None = None
        self._first_odom_wall_time: float | None = None
        self._last_moved_wall_time: float | None = None
        self._stop_reason: StopReason | None = None
        self._wall_start: float | None = None
        self._previous_domain_id: str | None = None

    def start(self) -> None:
        """Write the sim config, launch both processes, and start the background spin loop."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(
            yaml.safe_dump(self._sim_config, sort_keys=False),
            encoding='utf-8',
        )

        self._previous_domain_id = os.environ.get('ROS_DOMAIN_ID')
        os.environ['ROS_DOMAIN_ID'] = self._domain_id
        self._log_dir.mkdir(parents=True, exist_ok=True)

        global _INITIAL_RERUN_VIEWER_PIDS
        self._rerun_viewers_before = _rerun_viewer_pids()
        if _INITIAL_RERUN_VIEWER_PIDS is None:
            _INITIAL_RERUN_VIEWER_PIDS = set(self._rerun_viewers_before)

        pixi_exe = os.environ.get('PIXI_EXE', 'pixi')
        sim_pythonpath = os.pathsep.join(
            path for path in [str(self._sim_package_root), os.environ.get('PYTHONPATH', '')] if path
        )
        sim_python = 'python'
        self._context = rclpy.context.Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node('sim_control_harness', context=self._context)
        self._node.get_logger().info(
            f'Launching simulator with pixi env: {self._sim_pixi_env}'
        )
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)

        self._node.create_subscription(Odometry, self._odom_topic, self._odom_callback, ODOM_QUEUE_DEPTH)
        self._node.create_subscription(Clock, self._clock_topic, self._clock_callback, CLOCK_QUEUE_DEPTH)
        self._node.create_subscription(PlanMsg, '/plan', self._plan_callback, PLAN_QUEUE_DEPTH)

        self._sim_log_path = self._log_dir / 'sim.log'
        self._sim_proc, self._sim_log_file = _start_process(
            cmd=[
                pixi_exe, 'run', '-e', self._sim_pixi_env,
                sim_python, '-m', 'simulation_package.start_simulation',
                '--config', str(self._config_path),
                *self._sim_extra_args,
            ],
            log_path=self._sim_log_path,
            domain_id=self._domain_id,
            cwd=str(self._repo_root),
            extra_env={'PYTHONPATH': sim_pythonpath},
        )

        self._wait_for_sim_ready()

        self._control_log_path = self._log_dir / 'control.log'
        self._control_proc, self._control_log_file = _start_process(
            cmd=(
                ['ros2', 'launch', self._control_package, self._control_launch_file]
                + [f'{name}:={value}' for name, value in self._control_launch_args.items()]
            ),
            log_path=self._control_log_path,
            domain_id=self._domain_id,
            cwd=str(self._repo_root),
        )

        self._wall_start = time.monotonic()
        self._spin_thread = threading.Thread(
            target=self._spin_loop, daemon=True, name='sim_control_harness_spin'
        )
        self._spin_thread.start()

    def _wait_for_sim_ready(self) -> None:
        """Spin until the simulator publishes ``/clock``, it exits, or the timeout elapses."""
        started = time.monotonic()
        deadline = started + self._sim_ready_timeout_sec
        while time.monotonic() < deadline:
            with self._lock:
                if self._first_clock_sec is not None:
                    self._node.get_logger().info(
                        f'simulator ready after {time.monotonic() - started:.1f} s'
                    )
                    return
            if self._sim_proc is not None and self._sim_proc.poll() is not None:
                # Let the spin loop report SIM_EXITED with the log tail.
                return
            self._executor.spin_once(timeout_sec=SPIN_TIMEOUT_SEC)
        self._node.get_logger().warn(
            f'no /clock from the simulator within {self._sim_ready_timeout_sec:.0f} s; '
            'launching the control stack anyway'
        )

    def _odom_callback(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        sample = TraceSample(_stamp_to_sec(msg.header.stamp), float(position.x), float(position.y), yaw, speed)
        now = time.monotonic()
        with self._lock:
            self._message_count += 1
            self._trace.append(sample)
            self._last_xy = (sample.x_m, sample.y_m)
            if self._first_odom_wall_time is None:
                self._first_odom_wall_time = now
            previous_xy = self._previous_xy
            self._previous_xy = self._last_xy
            if previous_xy is not None:
                step_m = math.hypot(sample.x_m - previous_xy[0], sample.y_m - previous_xy[1])
                if step_m <= self._max_odom_step_m:
                    self._total_distance_m += step_m
                    if step_m >= self._stall_min_step_m:
                        self._last_moved_wall_time = now

    def _plan_callback(self, msg: PlanMsg) -> None:
        if not msg.poses:
            return
        goal = msg.poses[-1].pose.position
        with self._lock:
            self._plans.setdefault((round(goal.x, GOAL_ROUND_DIGITS), round(goal.y, GOAL_ROUND_DIGITS)),
                                   [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses])

    def _clock_callback(self, msg: Clock) -> None:
        stamp_sec = _stamp_to_sec(msg.clock)
        now = time.monotonic()
        with self._lock:
            if self._first_clock_sec is None:
                self._first_clock_sec = stamp_sec
                self._first_clock_wall_time = now
            self._last_clock_sec = stamp_sec
            self._last_clock_wall_time = now

    def _request_stop(self, reason: StopReason) -> None:
        """Record *reason* (the first one wins) and tell the spin loop to stop."""
        with self._lock:
            if self._stop_reason is None:
                self._stop_reason = reason
        self._stop_event.set()

    def _stop_reason_due(self) -> 'StopReason | None':
        """The deadline or process-exit condition met right now, if any."""
        now = time.monotonic()
        if self._wall_start is not None and now - self._wall_start >= self._max_runtime_sec:
            return StopReason.TIMEOUT
        if self._sim_timeout_sec is not None and self.elapsed_sim_sec >= self._sim_timeout_sec:
            return StopReason.SIM_TIMEOUT
        if self._stall_timeout_sec is not None:
            with self._lock:
                first_odom_time = self._first_odom_wall_time
                last_moved = self._last_moved_wall_time
            if first_odom_time is not None:
                reference = last_moved if last_moved is not None else first_odom_time
                if now - reference >= self._stall_timeout_sec:
                    return StopReason.STALLED
        if self._sim_proc is not None and self._sim_proc.poll() is not None:
            return StopReason.SIM_EXITED
        if self._control_proc is not None and self._control_proc.poll() is not None:
            return StopReason.CONTROL_EXITED
        return None

    def _spin_loop(self) -> None:
        while not self._stop_event.is_set():
            reason = self._stop_reason_due()
            if reason is not None:
                self._request_stop(reason)
                break
            self._executor.spin_once(timeout_sec=SPIN_TIMEOUT_SEC)

    def wait(
        self,
        predicate: Callable[[], bool] | None = None,
        timeout: float | None = None,
    ) -> 'StopReason | None':
        """Block until a stop condition is met or the optional wall-clock timeout expires.

        Returns the ``StopReason`` that caused the harness to stop.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            if predicate is not None and predicate():
                self._request_stop(StopReason.PREDICATE)
                return StopReason.PREDICATE
            if self._stop_event.is_set():
                return self.stop_reason
            if deadline is not None and time.monotonic() >= deadline:
                return self.stop_reason
            time.sleep(WAIT_POLL_SEC)

    def stop(self) -> None:
        """Gracefully stop both processes and tear down ROS resources."""
        self._stop_event.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=SPIN_JOIN_TIMEOUT_SEC)
        self._log_real_time_factor()
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=0.0)
        if self._node is not None:
            self._node.destroy_node()
        if self._context is not None:
            rclpy.shutdown(context=self._context)
        # Simulator first: stopping the control stack first leaves the sim stepping an
        # uncontrolled robot, and those frames land in the .rrd the assertions read.
        if self._sim_proc is not None and self._sim_log_file is not None:
            _stop_process(self._sim_proc, self._sim_log_file)
        if self._control_proc is not None and self._control_log_file is not None:
            _stop_process(self._control_proc, self._control_log_file)
        self._close_spawned_rerun_viewers()
        if self._previous_domain_id is None:
            os.environ.pop('ROS_DOMAIN_ID', None)
        else:
            os.environ['ROS_DOMAIN_ID'] = self._previous_domain_id
        self._request_stop(StopReason.STOPPED)

    def _log_real_time_factor(self) -> None:
        """Report sim seconds per wall second between the first and last ``/clock``."""
        with self._lock:
            if self._first_clock_wall_time is None or self._node is None:
                return
            sim_sec = self._last_clock_sec - self._first_clock_sec
            wall_sec = self._last_clock_wall_time - self._first_clock_wall_time
        rtf = sim_sec / wall_sec if wall_sec > 0 else float('nan')
        self._node.get_logger().info(
            f'simulator ran {sim_sec:.1f} sim s in {wall_sec:.1f} wall s '
            f'(real-time factor {rtf:.3f}), {self.message_count} odom messages'
        )

    def kill(self) -> None:
        """Immediately kill all process groups with SIGKILL."""
        self._stop_event.set()
        for proc, log_file in [
            (self._control_proc, self._control_log_file),
            (self._sim_proc, self._sim_log_file),
        ]:
            if proc is not None:
                pgid = _PROCESS_GROUP_IDS.get(proc.pid)
                if pgid is not None:
                    _killpg(pgid, signal.SIGKILL)
            if log_file is not None:
                log_file.close()
        self._close_spawned_rerun_viewers()

    def _close_spawned_rerun_viewers(self) -> None:
        """Close Rerun viewers that appeared while this harness was running."""
        before = getattr(self, '_rerun_viewers_before', None)
        if before is None:
            return
        _terminate_rerun_viewers(_rerun_viewer_pids() - before)

    def __enter__(self) -> 'SimControlHarness':
        """Start the harness."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the harness."""
        self.stop()

    @property
    def total_distance_m(self) -> float:
        """Total XY distance the robot has travelled, in metres."""
        with self._lock:
            return self._total_distance_m

    @property
    def last_xy(self) -> tuple[float, float] | None:
        """Most recent (x, y) position from odometry, or ``None`` if no messages yet."""
        with self._lock:
            return self._last_xy

    @property
    def trace(self) -> list[TraceSample]:
        """Every odom sample so far."""
        with self._lock:
            return list(self._trace)

    @property
    def plans(self) -> list[list[tuple[float, float]]]:
        """Nav2's first plan for each goal it was sent, as (x m, y m) points in the map frame."""
        with self._lock:
            return list(self._plans.values())

    @property
    def message_count(self) -> int:
        """Number of odometry messages received."""
        with self._lock:
            return self._message_count

    @property
    def elapsed_sim_sec(self) -> float:
        """Elapsed simulation time in seconds, derived from ``/clock``."""
        with self._lock:
            if self._first_clock_sec is None or self._last_clock_sec is None:
                return 0.0
            return self._last_clock_sec - self._first_clock_sec

    @property
    def stop_reason(self) -> 'StopReason | None':
        """The reason the harness stopped, or ``None`` if still running."""
        with self._lock:
            return self._stop_reason

    @property
    def sim_alive(self) -> bool:
        """``True`` if the simulation process is still running."""
        return self._sim_proc is not None and self._sim_proc.poll() is None

    @property
    def control_alive(self) -> bool:
        """``True`` if the control process is still running."""
        return self._control_proc is not None and self._control_proc.poll() is None

    def sim_log_tail(self, lines: int = LOG_TAIL_LINES) -> str:
        """Return the last *lines* lines of the simulation process log."""
        return _tail(self._sim_log_path, lines) if self._sim_log_path else ''

    def control_log_tail(self, lines: int = LOG_TAIL_LINES) -> str:
        """Return the last *lines* lines of the control process log."""
        return _tail(self._control_log_path, lines) if self._control_log_path else ''