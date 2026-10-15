# lite3-navigation-demo

Nav2 drives the DeepRobotics Lite3 quadruped along a route recorded in RViz,
in a simulated world extruded from the route's map, using
[artefacts-toolkit-navigation](https://github.com/art-e-fact/artefacts-toolkit-navigation).

Split out of [lite3-ros-demos](https://github.com/art-e-fact/lite3-ros-demos).
Tested on Linux (x64); the simulator also runs on macOS (arm64).

## Requirements

* Linux (tested on Ubuntu 24.04) or macOS
* [pixi](https://pixi.prefix.dev/latest/installation/)
* Optional: an NVIDIA GPU with CUDA 12 drivers. Newton runs on it when one is found
  (`scripts/sim_pixi_env.sh`); on CPU it runs well below real time.

## Build

```bash
pixi run build
```

The simulator (Newton, ROS 2 Kilted) and the Nav2 side (ROS 2 Jazzy)
need different ROS versions, so they are built into separate workspaces under
`rosbuild/` and run as separate processes. Delete `rosbuild/` for a clean build.

## Run the demo

### 1. Record a route

```bash
pixi run record-route --name lab     # on src/nav2_demo/maps/depot.yaml; --map for another
```

RViz opens with the map: **2D Pose Estimate** where the robot should start,
**Publish Point** for each waypoint, enter in the terminal to save
`routes/lab.yaml`.

### 2. Drive it

The test runs through [Artefacts](https://docs.artefacts.com), which passes it
the scenario params from `artefacts.yaml` and collects what it writes:

```bash
pixi run -e artefacts artefacts login   # once; set `project` in artefacts.yaml to yours
pixi run -e artefacts artefacts run test-nav2-route
SIM_PIXI_ENV=sim pixi run -e artefacts artefacts run test-nav2-route   # force the CPU Newton build
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
