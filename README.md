# lite3-navigation-demo

Nav2 drives the DeepRobotics Lite3 quadruped along a route recorded in RViz,
in a simulated world extruded from the route's map, using
[artefacts-toolkit-navigation](https://github.com/art-e-fact/artefacts-toolkit-navigation).

Split out of [lite3-ros-demos](https://github.com/art-e-fact/lite3-ros-demos).

## Requirements

* Ubuntu 24.04 with [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
  (`ros-jazzy-ros-base` is enough; rosdep installs the rest), `python3-colcon-common-extensions`,
  `python3-rosdep`
* Optional: an NVIDIA GPU. Newton uses it when a driver is found; on CPU it runs
  well below real time.

## Build

```bash
git submodule update --init --recursive
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -y --skip-keys "rail_inspector simple_local_heightmap"
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements.txt
colcon build --symlink-install
source install/setup.bash
```

The two skipped keys are stale entries in the sdk_deploy submodule's manifest.
Every later shell needs the three `source` lines. (`uv venv --system-site-packages`
and `uv pip install -r requirements.txt` work too.)

## Run the demo

### 1. Record a route

```bash
artefacts-route record --name lab --map src/nav2_demo/maps/depot.yaml
```

RViz opens with the map: **2D Pose Estimate** where the robot should start,
**Publish Point** for each waypoint, enter in the terminal to save
`routes/lab.yaml`.

### 2. Drive it

The test runs through [Artefacts](https://docs.artefacts.com), which passes it
the scenario params from `artefacts.yaml` and collects what it writes:

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
