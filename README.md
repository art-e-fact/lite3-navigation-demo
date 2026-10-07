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

The simulator (Newton/MuJoCo, ROS 2 Kilted) and the Nav2 side (ROS 2 Jazzy)
need different ROS versions, so they are built into separate workspaces under
`rosbuild/` and run as separate processes. Delete `rosbuild/` for a clean build.

## Run the demo

```bash
pixi run test-nav2-route --headless  # Newton, no windows
pixi run test-nav2-route             # with the Newton viewer
RVIZ=true pixi run test-nav2-route   # ... and Nav2's RViz view
pixi run test-nav2-route-mujoco      # the same on MuJoCo
pixi run test-nav2-route-cpu         # force the CPU Newton build
```

This starts the simulator with the Lite3 at the route's start, launches
rl_deploy and Nav2 on the route's map, drives the route and asserts that the
robot got round. World, simulator config and route result land in `results/`.

### Record your own route

```bash
pixi run record-route --name lab     # on src/nav2_demo/maps/depot.yaml; --map for another
pixi run test-nav2-route             # drives the route recorded last
ROUTE=routes/lab.yaml pixi run test-nav2-route   # or a particular one
```

RViz opens with the map: **2D Pose Estimate** where the robot should start,
**Publish Point** for each waypoint, enter in the terminal to save
`routes/lab.yaml`. See [src/nav2_demo](src/nav2_demo/README.md) for the pieces.

## Run with Artefacts

```bash
pixi run -e artefacts artefacts login   # once; set `project` in artefacts.yaml to yours
pixi run -e artefacts artefacts run test-nav2-route
pixi run -e artefacts artefacts run test-nav2-route-mujoco
```

## Layout

- `src/nav2_demo`: the launch file, Nav2 parameters, maps, bundled route and the test.
- `src/simulation_package`: the Lite3 in Newton or MuJoCo with a 2D lidar, publishing to ROS 2.
- `src/sdk_deploy` (submodule): DeepRobotics' `rl_deploy`, the locomotion controller that turns `/cmd_vel` into joint commands.
