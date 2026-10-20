"""Drive a recorded route with Nav2 on the Lite3, in a world extruded from the
route's map, and check that the robot got round.

    artefacts run test-nav2-route

The route is driven once per run (the module-scoped ``run`` fixture); each test
then checks one aspect of the result with the toolkit's assert helpers, so the
dashboard shows them one by one.

Runs under `artefacts run` only: it reads the scenario's params from artefacts.yaml
(headless, rviz, linear_vel; a list drives the route once per value) and writes
everything into the run's upload directory. It drives the route recorded last with
`artefacts-route record`. The simulator (Newton) runs as a subprocess
with the robot spawned at the route's start; Nav2 and rl_deploy are launched
via nav2_route.launch.py; the fixture drives the route with
artefacts_toolkit_navigation.follow_route().
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from artefacts_toolkit.config import get_artefacts_params
from artefacts_toolkit_navigation import (
    assert_each_waypoint_reached, assert_final_pose, assert_max_recoveries, assert_min_clearance,
    assert_route_completed, follow_route, load_route, map_to_world, route_launch_args)

from report import path_chart, speed_chart, trace_metrics, waypoint_chart, write_trace_csv
from sim_control_harness import SimControlHarness, StopReason

PARAMS = get_artefacts_params()  # this scenario's params from artefacts.yaml
OUTPUT_FOLDER = Path(os.environ['ARTEFACTS_SCENARIO_UPLOAD_DIR'])  # uploaded with the run
REPO_ROOT = Path(__file__).resolve().parents[3]  # test/ -> nav2_demo/ -> src/ -> repo
SIM_PACKAGE_ROOT = REPO_ROOT / 'src' / 'simulation_package'
METRICS = REPO_ROOT / 'metrics.json'  # artefacts.yaml's `metrics:`, read from where `artefacts run` runs
METRIC_DECIMALS = 3

ROUTE_TIMEOUT_SEC = 300.0
NAV2_STARTUP_TIMEOUT_SEC = 180.0
HARNESS_MARGIN_SEC = 120.0  # slack on top of startup + route time before the harness gives up
LIDAR_RANGE_M = 12.0
FOLLOW_ELEVATION_DEG = -35.0
OVERHEAD_ELEVATION_DEG = -89.0  # not -90: looking straight down leaves the camera up vector undefined
OVERHEAD_DISTANCE_M = 10.0
OVERHEAD_SMOOTHING = 0.03  # per-frame blend towards the robot (0..1); low keeps the top-down view steady
WAYPOINT_TOLERANCE_M = 0.5  # how close to a waypoint counts as reached
MAX_FINAL_ERROR_M = 0.5  # how close to the last waypoint counts as arrived
MIN_CLEARANCE_M = 0.2  # lidar to nearest wall; the lidar sits mid-body, so this is roughly the body's half width
MAX_RECOVERIES = 2  # spins/backups Nav2 may need before the run counts as a struggle
VIDEO_FPS = 10.0  # each camera frame costs ~7 ms of wall time (the viewer re-uploads every wall shape)


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    """Drive the route once; the result and the harness (for its log tails) for every test."""
    METRICS.unlink(missing_ok=True)  # or a run that dies early reports the previous run's
    route = load_route()  # the route recorded last

    # The world: the route's map with its occupied cells raised into walls.
    world = map_to_world(REPO_ROOT / route.map.file, OUTPUT_FOLDER / f'{route.name}_world.xml',
                         format='mjcf')

    sim_config = {
        'simulator': 'newton',
        'scene': world,
        'headless': PARAMS['headless'],
        'robot': {'model': 'lite3'},
        'sensors': {
            'lidar_2d': {'enabled': True, 'range_max': LIDAR_RANGE_M},
            # Two videos of the run, recorded by the simulator (headless too): a chase
            # camera, steeper than the default 18 deg so the 1 m walls do not hide the
            # robot, and a top-down view that keeps the map's heading.
            'follow_camera': {'enabled': True, 'video_path': str(OUTPUT_FOLDER / f'{route.name}_follow.mp4'),
                              'elevation_deg': FOLLOW_ELEVATION_DEG, 'fps': VIDEO_FPS},
            'overhead_camera': {'enabled': True, 'video_path': str(OUTPUT_FOLDER / f'{route.name}_overhead.mp4'),
                                'elevation_deg': OVERHEAD_ELEVATION_DEG, 'follow_yaw': False, 'distance_m': OVERHEAD_DISTANCE_M,
                                'smoothing': OVERHEAD_SMOOTHING, 'fps': VIDEO_FPS},
        },
        'rerun': {'enabled': False},
    }
    # Spawn the robot where the route starts: the simulator's config fields,
    # named the way its --set overrides want them.
    start = route_launch_args(
        route, names=('robot.start_pose.x', 'robot.start_pose.y', 'robot.start_pose.yaw'),
        map_arg=None)
    sim_extra_args = [arg for key, value in start.items() for arg in ('--set', f'{key}={value}')]

    with SimControlHarness(
        sim_config,
        config_path=OUTPUT_FOLDER / f'{route.name}_sim.yaml',
        log_dir=tmp_path_factory.mktemp('harness'),
        repo_root=REPO_ROOT,
        sim_package_root=SIM_PACKAGE_ROOT,
        control_package='nav2_demo',
        control_launch_file='nav2_route.launch.py',
        control_launch_args={'use_rviz': str(PARAMS['rviz']).lower(), 'linear_vel': str(PARAMS['linear_vel'])},
        sim_extra_args=sim_extra_args,
        max_runtime_sec=NAV2_STARTUP_TIMEOUT_SEC + ROUTE_TIMEOUT_SEC + HARNESS_MARGIN_SEC,
    ) as harness:
        result = follow_route(
            route, timeout_s=ROUTE_TIMEOUT_SEC, startup_timeout_s=NAV2_STARTUP_TIMEOUT_SEC,
            label=route.name, results_dir=str(OUTPUT_FOLDER))
        reason = harness.stop_reason
        trace = harness.trace
        plans = harness.plans

    # The run's data, pass or fail: the odom trace as CSV, the charts and the metrics.
    metrics = result.metrics()
    waypoint_chart(result, OUTPUT_FOLDER / f'{route.name}_waypoints.html')
    if trace:
        write_trace_csv(trace, OUTPUT_FOLDER / f'{route.name}_odom.csv')
        path_chart(REPO_ROOT / route.map.file, route, trace, OUTPUT_FOLDER / f'{route.name}_path.html', plans)
        speed_chart(trace, route, OUTPUT_FOLDER / f'{route.name}_speed.html')
        metrics.update(trace_metrics(trace, route, plans))
    if result.waypoints:
        metrics['time_to_first_waypoint_s'] = result.waypoints[0].time_s
    METRICS.write_text(json.dumps({name: round(float(value), METRIC_DECIMALS) for name, value in metrics.items()},
                                  indent=2))

    if reason is StopReason.SIM_EXITED:
        pytest.fail(f'simulation exited unexpectedly:\n{harness.sim_log_tail()}')
    if reason is StopReason.CONTROL_EXITED:
        pytest.fail(f'Nav2 launch exited unexpectedly:\n{harness.control_log_tail()}')

    return SimpleNamespace(result=result, harness=harness)


def test_route_completed(run):
    try:
        assert_route_completed(run.result)
    except AssertionError as error:
        pytest.fail(f'{error}\n--- nav log ---\n{run.harness.control_log_tail(40)}')


def test_each_waypoint_reached(run):
    assert_each_waypoint_reached(run.result, tolerance_m=WAYPOINT_TOLERANCE_M)


def test_final_pose(run):
    assert_final_pose(run.result, max_pos_error_m=MAX_FINAL_ERROR_M)


def test_min_clearance(run):
    assert_min_clearance(run.result, MIN_CLEARANCE_M)


def test_max_recoveries(run):
    assert_max_recoveries(run.result, MAX_RECOVERIES)
