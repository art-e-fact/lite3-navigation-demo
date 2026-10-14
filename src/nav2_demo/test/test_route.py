"""Drive a recorded route with Nav2 on the Lite3, in a world extruded from the
route's map, and check that the robot got round.

    pixi run test-nav2-route            # or: pixi run -e nav-test pytest src/nav2_demo/test -s

The simulator (Newton) runs in its own pixi environment with the
robot spawned at the route's start; Nav2 and rl_deploy run from this
environment via nav2_route.launch.py; the test itself drives the route with
artefacts_toolkit_navigation.follow_route(). Param, also settable from
Artefacts scenarios: headless. Environment: ROUTE picks a route file
(default: the route recorded last, else the bundled depot route); RVIZ=true
opens Nav2's RViz view alongside (the simulator's own viewer follows headless).
"""

import json
import os
from pathlib import Path

import pytest
from artefacts_toolkit_navigation import follow_route, load_route, map_to_world, route_launch_args

from report import path_chart, speed_chart, trace_metrics, waypoint_chart, write_trace_csv
from sim_control_harness import SimControlHarness, StopReason

# Where the world, the simulator config and the route result go: Artefacts' upload
# directory when it sets one, else results/ (gitignored).
OUTPUT_FOLDER = Path(os.getenv('ARTEFACTS_SCENARIO_UPLOAD_DIR', 'results'))
REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SIM_PACKAGE_ROOT = REPO_ROOT / 'src' / 'simulation_package'
METRICS = REPO_ROOT / 'metrics.json'  # artefacts.yaml's `metrics:`, read from where `artefacts run` runs
METRIC_DECIMALS = 3
# The route to drive: ROUTE=... if given, else the route recorded last (artefacts-route
# record saves into ./routes), else the bundled depot route.
ROUTE = (os.environ.get('ROUTE') or (None if Path('routes').is_dir() else PACKAGE_ROOT / 'routes' / 'depot.yaml'))
RVIZ = os.environ.get('RVIZ', 'false').lower() in ('1', 'true', 'yes')

ROUTE_TIMEOUT_SEC = 300.0
NAV2_STARTUP_TIMEOUT_SEC = 180.0
HARNESS_MARGIN_SEC = 120.0  # slack on top of startup + route time before the harness gives up
LIDAR_RANGE_M = 12.0
FOLLOW_ELEVATION_DEG = -35.0
OVERHEAD_ELEVATION_DEG = -89.0  # not -90: looking straight down leaves the camera up vector undefined
OVERHEAD_DISTANCE_M = 10.0
OVERHEAD_SMOOTHING = 0.03  # per-frame blend towards the robot (0..1); low keeps the top-down view steady
MAX_FINAL_ERROR_M = 0.5  # how close to the last waypoint counts as arrived


def test_route_is_completed(tmp_path, headless):
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    METRICS.unlink(missing_ok=True)  # or a run that dies early reports the previous run's
    route = load_route(ROUTE)

    # The world: the route's map with its occupied cells raised into walls.
    world = map_to_world(REPO_ROOT / route.map.file, OUTPUT_FOLDER / f'{route.name}_world.xml',
                         format='mjcf')

    sim_config = {
        'simulator': 'newton',
        'scene': world,
        'headless': headless,
        'robot': {'model': 'lite3'},
        'sensors': {
            'lidar_2d': {'enabled': True, 'range_max': LIDAR_RANGE_M},
            # Two videos of the run, recorded by the simulator (headless too): a chase
            # camera, steeper than the default 18 deg so the 1 m walls do not hide the
            # robot, and a top-down view that keeps the map's heading.
            'follow_camera': {'enabled': True, 'video_path': str(OUTPUT_FOLDER / f'{route.name}_follow.mp4'),
                              'elevation_deg': FOLLOW_ELEVATION_DEG},
            'overhead_camera': {'enabled': True, 'video_path': str(OUTPUT_FOLDER / f'{route.name}_overhead.mp4'),
                                'elevation_deg': OVERHEAD_ELEVATION_DEG, 'follow_yaw': False, 'distance_m': OVERHEAD_DISTANCE_M,
                                'smoothing': OVERHEAD_SMOOTHING},
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
        log_dir=tmp_path,
        repo_root=REPO_ROOT,
        sim_package_root=SIM_PACKAGE_ROOT,
        control_package='nav2_demo',
        control_launch_file='nav2_route.launch.py',
        control_launch_args={'use_rviz': str(RVIZ).lower(), **({'route': str(ROUTE)} if ROUTE else {})},
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

    assert result.succeeded, (
        f'{result.outcome}: {result.error}\n--- nav log ---\n{harness.control_log_tail(40)}')
    assert result.final_pos_error_m is not None and result.final_pos_error_m < MAX_FINAL_ERROR_M, (
        f'ended {result.final_pos_error_m:.2f} m from the last waypoint')
