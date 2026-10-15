# nav2_demo

A route, driven by Nav2 on the Lite3 quadruped, in a world made from the
route's map. Uses
[artefacts-toolkit-navigation](https://github.com/art-e-fact/artefacts-toolkit-navigation):

- `artefacts-route record` records a route by clicking in RViz on a map.
  `routes/depot.yaml` is one recorded that way on Nav2's depot map.
- `map_to_world()` raises the map's occupied cells into walls, giving a
  simulator world at the map's coordinates (`format="mjcf"` for this
  simulator). It takes milliseconds, so the test makes it fresh each run.
- `follow_route()` sends the route to Nav2 and reports whether it was
  completed.

## Record a route

From the repository root, on one of the maps here (or your own):

```bash
pixi run record-route --name lab   # --map <other.yaml> for another map
```

This opens RViz with the map. Click **2D Pose Estimate** where the robot
should start, then **Publish Point** for each waypoint in order; press enter
in the terminal (or close RViz) to save `routes/lab.yaml`. Keep the points on
open floor: the Lite3 is 0.56 x 0.30 m and walks at 0.3 m/s.

## Run it

```bash
artefacts run test-nav2-route
```

Drives the route recorded last. The params in `artefacts.yaml` pick the viewers
(`headless`, `rviz` for Nav2's RViz view: map, costmaps, path, robot) and the
speed (`linear_vel`).

## Pieces

- `launch/nav2_route.launch.py`: rl_deploy and Nav2's `bringup_launch.py` on
  the route's map with `config/nav2_lite3.yaml`. Nav2 is started a few
  seconds after rl_deploy (`nav2_delay`), once the robot is standing, so the
  lidar has stopped sweeping the ground.
- `config/nav2_lite3.yaml`: Nav2's default parameters with the Lite3's frames,
  footprint and speeds, and Regulated Pure Pursuit as the controller; the
  header lists every change.
- `test/sim_control_harness.py`: starts the simulator in its own pixi
  environment, then the nav launch once `/clock` ticks, and watches both.
- `test/test_route.py`: the pytest. It extrudes the world from the route's
  map, starts the simulator (Newton, the Lite3 with its 2D lidar) with
  `--set robot.start_pose.*` from the route, launches the nav side, and
  asserts on `follow_route()`'s result. The world, the simulator config, the
  route result, a chase-camera and an overhead video, the odom trace as CSV and
  three charts (`test/report.py`: path on the map, speed, per-waypoint figures)
  and `metrics.json` are written for Artefacts to upload.
