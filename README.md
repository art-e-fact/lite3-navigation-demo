# lite3-navigation-demo

This demo showcases [artefacts-toolkit-navigation](https://github.com/art-e-fact/artefacts-toolkit-navigation), running the DeepRobotics Lite3 through an automatically generated scene (based on a rviz map), with a user recorded route.

## Requirements

* ROS 2 Jazzy.
* Optional: an NVIDIA GPU which Newton will use if found.

## Build

_Note: This project can also be run with `pixi`. Please see the [pixi branch](https://github.com/art-e-fact/lite3-navigation-demo/tree/pixi)._

```bash
git submodule update --init --recursive
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -y --skip-keys "rail_inspector simple_local_heightmap"
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements.txt
colcon build --symlink-install
source install/setup.bash
```
_Note_ `rail_inspector` and `simple_local_heightmap` are part of the `artefacts` fork of the Deep Robotics `sdk_deploy`. They are unused in this demo and so can be skipped.

## Run the demo

### 1. Record a route

```bash
artefacts-route record --map src/nav2_demo/maps/depot.yaml
```

RViz opens with the map: **2D Pose Estimate** where the robot should start,
**Publish Point** for each waypoint, enter in the terminal to save it as a
timestamped file under `routes/`.

### 2. Drive it

The test runs through [Artefacts](https://docs.artefacts.com).

```bash
artefacts login   # once; set `project` in artefacts.yaml to yours
artefacts run test-nav2-route
```

This takes the route recorded last, starts the simulator with the Lite3 at
its start, launches rl_deploy and Nav2 on its map, drives the route once per
`linear_vel` and asserts that the robot got round. World, simulator config,
route result, two videos (chase and overhead), the odom trace as CSV, charts
and metrics are uploaded with each run. The params in `artefacts.yaml`:

- `headless`: `false` shows the Newton viewer; `rviz: true` also opens Nav2's RViz view.
- `linear_vel`: FollowPath's speed in m/s; a list drives the route once per value.

See [src/nav2_demo](src/nav2_demo/README.md) for the pieces.

## Layout

- `src/nav2_demo`: the launch file, Nav2 parameters, maps and the test.
- `src/simulation_package`: the Lite3 in Newton with a 2D lidar, publishing to ROS 2.
- `src/sdk_deploy` (submodule): DeepRobotics' `rl_deploy`, the locomotion controller that turns `/cmd_vel` into joint commands.
