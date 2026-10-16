#!/usr/bin/env python3

import argparse
import math
import signal
import sys
import time
from pathlib import Path

import warp as wp

CURRENT_DIR = Path(__file__).resolve().parent
# Sibling modules import each other by bare name, from either the source tree or the install.
for directory in (CURRENT_DIR, CURRENT_DIR.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import rclpy

from rerun_recorder import NewtonRerunRecorder
from ros_bridge import NewtonRosBridge
from simulation import DT, ROS_SPIN_EVERY_STEPS, ROBOT_PROFILES, NewtonSimulation, RobotProfile, create_newton_viewer
from sensors.newton.geometry import find_body_index
from sensors.newton.sensor_manager import NewtonSensorManager
from simulation_config import SimulationConfig, load_validated_config
from utils import log, step_interval

RENDER_EVERY_STEPS = 5

VIEWER_CAMERA_DISTANCE_M = 5.0
VIEWER_CAMERA_HEIGHT_M = 2.5

# Physics steps run before the simulator goes live, so Warp/MuJoCo kernel compilation
# happens before the first /clock rather than stalling the controller. Cheap on GPU
# (the CUDA graph capture has already compiled most of it), essential on CPU.
WARMUP_STEPS = 10
RERUN_FLUSH_EVERY_RECORDS = 100


def _aim_viewer_at_robot(viewer, sim: NewtonSimulation):
    """Point the optional Newton viewer at the robot's spawn."""
    set_camera = getattr(viewer, "set_camera", None)
    if set_camera is None:
        return

    start_x, start_y, start_yaw = sim.robot_start_pose
    camera_pos = wp.vec3(
        float(start_x) - VIEWER_CAMERA_DISTANCE_M * math.cos(start_yaw),
        float(start_y) - VIEWER_CAMERA_DISTANCE_M * math.sin(start_yaw),
        sim.profile.base_height + VIEWER_CAMERA_HEIGHT_M,
    )
    pitch_deg = -math.degrees(math.atan2(VIEWER_CAMERA_HEIGHT_M, VIEWER_CAMERA_DISTANCE_M))
    set_camera(camera_pos, pitch_deg, math.degrees(start_yaw))


def warm_up(sim: NewtonSimulation, sensors: NewtonSensorManager | None, recorder: NewtonRerunRecorder | None):
    """Compile every kernel the live loop needs, then rewind to the initial state.

    Nothing here is published as simulated time: the harness gates the control stack on
    the first ``/clock``, so the controller only ever sees a simulator running at speed.
    """
    started = time.perf_counter()
    for _ in range(WARMUP_STEPS):
        sim.step()
    if sensors is not None:
        sensors.warmup(sim.state_0, sim.timestamp)
    sim.rewind()
    if sensors is not None:
        sensors.reset_cameras()
    if recorder is not None:
        # After the rewind, so the t=0 body transforms are the ones the run starts from.
        recorder.log_model(sim.state_0)
    return time.perf_counter() - started


def run_loop(
    sim: NewtonSimulation,
    ros: NewtonRosBridge,
    state_every_steps: int,
    sensors: NewtonSensorManager | None = None,
    recorder: NewtonRerunRecorder | None = None,
):
    next_step_time = time.perf_counter()
    records_since_flush = 0
    while rclpy.ok() and not ros.should_exit():
        sleep_time = next_step_time - time.perf_counter()
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        else:
            next_step_time = time.perf_counter()
        next_step_time += DT

        if sim.step_count % ROS_SPIN_EVERY_STEPS == 0:
            ros.spin_once()
        sim.set_command(ros.read_latest_action())
        sim.step()
        ros.publish_clock(sim.timestamp)

        if sim.step_count % state_every_steps == 0:
            state = sim.state_snapshot()
            ros.publish_state(sim.timestamp, state, sim.last_tau)
            ros.publish_odom_and_tf(sim.timestamp, state)
            if recorder is not None:
                # Tests resample at 20 ms
                recorder.record(sim.state_0, sim.timestamp)
                records_since_flush += 1
                if records_since_flush >= RERUN_FLUSH_EVERY_RECORDS:
                    recorder.flush()
                    records_since_flush = 0

        if sensors is not None:
            sensors.update(sim.state_0, sim.step_count, sim.timestamp)

        if sim.step_count % RENDER_EVERY_STEPS == 0:
            sim.render()


def _robot_body_index(sim: NewtonSimulation, profile: RobotProfile) -> int:
    """Index of the robot's root body in ``state.body_q``.

    The robot is added to the builder first, so body 0 is its root; the label check
    keeps that assumption honest if the build order ever changes.
    """
    try:
        return find_body_index(sim.model, profile.root_body)
    except ValueError:
        log(sim.logger, "warn", f"Root body '{profile.root_body}' not found in the Newton model; using body 0",
            tag_logger=False)
        return 0


def run_newton(config: SimulationConfig, ros_args: list[str] | None = None):
    rclpy.init(args=ros_args)
    model_path = config.resolved_robot_description()
    scene_path = config.resolved_scene()
    profile = ROBOT_PROFILES.get(config.robot.model_name)
    if profile is None:
        raise SystemExit(f"No Newton robot profile for '{config.robot.model_name}'")
    viewer = None if config.headless else create_newton_viewer()
    ros = NewtonRosBridge(headless=config.headless, model_path=model_path, profile=profile)
    sim = NewtonSimulation(
        model_path=model_path,
        scene_path=scene_path,
        headless=config.headless,
        viewer=viewer,
        logger=ros.get_logger(),
        sensor_options=config.sensors,
        profile=profile,
        start_pose=config.robot.start_pose,
    )
    if viewer is not None:
        _aim_viewer_at_robot(viewer, sim)

    recorder = None
    if config.rerun.enabled:
        recorder = NewtonRerunRecorder(
            sim.model,
            config.robot.model,
            config.rerun,
            headless=config.headless,
            logger=ros.get_logger(),
        )

    sensors = NewtonSensorManager(
        sim.model,
        sim.state_0,
        ros.node,
        DT,
        config.sensors,
        robot_body_index=_robot_body_index(sim, profile),
        viewer=viewer,
    )

    state_every_steps = step_interval(config.robot.state_frequency_hz, DT)

    try:
        warmup_sec = warm_up(sim, sensors, recorder)
        log(ros.node, "info", f"Newton warm-up finished in {warmup_sec:.1f} s; publishing /clock")
        run_loop(sim, ros, state_every_steps, sensors, recorder)
    except KeyboardInterrupt:
        log(ros.node, "info", "Newton simulation interrupted; shutting down")
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                ros.get_logger().error(f"Failed to close the Rerun recorder: {exc}")
        sensors.close()
        ros.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Run Lite3 Newton ROS2 simulation")
    parser.add_argument("--config", default=None, help="Path to simulation YAML config")
    args, ros_args = parser.parse_known_args()
    run_newton(load_validated_config(args.config, {"simulator": "newton"}), ros_args)


if __name__ == "__main__":
    main()
