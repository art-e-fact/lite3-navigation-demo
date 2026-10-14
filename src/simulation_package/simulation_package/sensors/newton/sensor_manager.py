"""Orchestrates Newton sensors at their own publish rates."""

from dataclasses import dataclass, field

from tf2_ros import StaticTransformBroadcaster

from sensors.common.transforms import sim_time_stamp
from simulation_config import FollowCameraConfig, Lidar2DConfig
from sensors.newton.bvh import NewtonBvh
from sensors.newton.follow_camera_recorder import NewtonFollowCameraRecorder
from sensors.newton.lidar_sensor import NewtonLidarSensor
from utils import step_interval


@dataclass
class NewtonSensorOptions:
    lidar_2d: Lidar2DConfig = field(default_factory=Lidar2DConfig)
    follow_camera: FollowCameraConfig = field(default_factory=FollowCameraConfig)
    overhead_camera: FollowCameraConfig = field(default_factory=FollowCameraConfig)

    @property
    def enable_lidar(self) -> bool:
        return self.lidar_2d.enabled

    @property
    def enable_follow_camera(self) -> bool:
        return self.follow_camera.enabled or self.overhead_camera.enabled


class NewtonSensorManager:
    """Runs every Newton ray sensor plus the follow camera at their own rates.

    All of them raytrace against the same shape BVH, so the manager refits it once
    per step for whichever sensors are due and then renders them all from that fit.
    """

    def __init__(
        self,
        model,
        state,
        node,
        dt: float,
        options: NewtonSensorOptions,
        robot_body_index: int = 0,
        viewer=None,
    ):
        self.model = model
        self.node = node
        self.static_tf_broadcaster = StaticTransformBroadcaster(node)

        self.lidar = NewtonLidarSensor(model, node, config=options.lidar_2d)
        # The follow camera raytraces the same BVH as the lidar, so it belongs in the
        # same due-list rather than owning a second refit.
        self.follow_camera = NewtonFollowCameraRecorder(
            model,
            node,
            config=options.follow_camera,
            robot_body_index=robot_body_index,
            viewer=viewer,
            bvh=None,  # assigned below once the shared BVH exists
        )
        # Never the interactive window: that one is the follow camera's. This camera
        # renders off its own offscreen viewer (or the raytracer without GL).
        self.overhead_camera = NewtonFollowCameraRecorder(
            model,
            node,
            config=options.overhead_camera,
            robot_body_index=robot_body_index,
            viewer=None,
            bvh=None,
        )
        self.cameras = [
            (self.follow_camera, step_interval(options.follow_camera.fps, dt)),
            (self.overhead_camera, step_interval(options.overhead_camera.fps, dt)),
        ]

        # Truncated, unlike step_interval(): kept so the sensor rates stay bit-identical.
        self.lidar_step_interval = max(1, int(1.0 / (options.lidar_2d.frequency_hz * dt)))
        self.bvh = NewtonBvh(model, state) if self.enabled else None
        for camera, _ in self.cameras:
            camera.bvh = self.bvh
        self._publish_static_transforms()

    @property
    def enabled(self) -> bool:
        return (
            self.lidar.enabled
            or any(camera.enabled for camera, _ in self.cameras)
        )

    def update(self, state, step_count: int, timestamp: float):
        if not self.enabled:
            return

        due_lidar = self.lidar.enabled and step_count % self.lidar_step_interval == 0
        due_cameras = [camera for camera, interval in self.cameras
                       if camera.enabled and step_count % interval == 0]
        if not (due_lidar or due_cameras):
            return

        # Refit once for all due sensors; sensor rendering then sees the same world pose.
        self.bvh.refit(state)
        if due_lidar:
            self.lidar.update(state, timestamp)
        for camera in due_cameras:
            camera.update(state, refit=False, timestamp=timestamp)

    def warmup(self, state, timestamp: float = 0.0):
        """Render every enabled sensor once so their Warp kernels are compiled.

        Publishes one message per sensor topic; the follow-camera frame is rendered but
        deliberately not written to the video.
        """
        if not self.enabled:
            return

        self.bvh.refit(state)
        if self.lidar.enabled:
            self.lidar.update(state, timestamp)
        for camera, _ in self.cameras:
            if camera.enabled:
                camera.render(state, refit=False, timestamp=timestamp)

    def reset_cameras(self):
        """Forget the cameras' smoothed look-at points (after a warm-up rewind)."""
        for camera, _ in self.cameras:
            camera.reset()

    def close(self):
        for camera, _ in self.cameras:
            camera.close()

    def _publish_static_transforms(self):
        # Sim time, like every other message the simulator publishes: the nav stack runs
        # with use_sim_time and cannot match a wall-clock TF against sim-clock data.
        stamp = sim_time_stamp(0.0)
        transforms = []
        transforms.extend(self.lidar.get_static_transforms(stamp))
        if transforms:
            self.static_tf_broadcaster.sendTransform(transforms)
