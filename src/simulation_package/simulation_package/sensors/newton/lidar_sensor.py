"""2D LiDAR for Newton using custom SensorTiledCamera rays."""

import math
import newton
import numpy as np
from sensor_msgs.msg import LaserScan

from constants import SENSOR_QUEUE_DEPTH
from sensors.common.lidar import planar_lidar_dirs
from sensors.common.transforms import sim_time_stamp
from sensors.newton.geometry import find_builder_shape_index, find_site_index, site_static_transform
from sensors.newton.ray_buffers import SiteRayCaster, rays_from_dirs
from simulation_config import Lidar2DConfig

MAX_ALL_INF_DEBUG_LOGS = 5
# Site marker cylinder (radius, half height, unused) [m]: a 5 cm puck.
LIDAR_MARKER_SCALE = (0.025, 0.01, 0.0)
# Site marker colour, RGB 0..1: dark grey.
LIDAR_MARKER_COLOR = (0.1, 0.1, 0.1)
# Hit-body sentinel for missed rays; distinct from every body index and from -1 (world).
NO_HIT_BODY = -9999


class NewtonLidarSensor:
    @staticmethod
    def init_visuals(builder, config: Lidar2DConfig | None = None):
        config = config or Lidar2DConfig()
        site_index = find_builder_shape_index(builder, config.site_name)
        builder.shape_type[site_index] = newton.GeoType.CYLINDER
        builder.shape_scale[site_index] = LIDAR_MARKER_SCALE
        builder.shape_color[site_index] = LIDAR_MARKER_COLOR
        # Not VISIBLE: the rays start at this shape's centre and SensorTiledCamera casts
        # against every rendered shape, so a visible marker is hit by all 360 rays at
        # 2.5 cm (its own wall) and the scan is all inf.
        builder.shape_flags[site_index] &= ~int(newton.ShapeFlags.VISIBLE)

    def __init__(self, model, node, enabled: bool = False, config: Lidar2DConfig | None = None):
        self.config = config or Lidar2DConfig(enabled=enabled)
        self.model = model
        self.node = node
        self.enabled = self.config.enabled
        self.site_index = -1

        if not self.enabled:
            node.get_logger().info("[INFO] Newton LiDAR disabled")
            return

        self.site_index = find_site_index(model, self.config.site_name)
        self.body_id = int(model.shape_body.numpy()[self.site_index])
        self.shape_body = model.shape_body.numpy()
        self.local_dirs = planar_lidar_dirs(self.config.num_rays)
        self.rays = rays_from_dirs(self.local_dirs)
        self.caster = SiteRayCaster(model, self.site_index, self.config.num_rays)

        self.pub = node.create_publisher(LaserScan, self.config.topic, SENSOR_QUEUE_DEPTH)
        self.angle_increment = 2.0 * math.pi / self.config.num_rays
        self.scan_time = 1.0 / self.config.frequency_hz
        self.time_increment = self.scan_time / max(1, self.config.num_rays - 1)
        self._all_inf_debug_logs = 0
        node.get_logger().info(
            f"[INFO] Newton LiDAR initialized ({self.config.num_rays} rays @ {self.config.frequency_hz} Hz)"
        )

    def update(self, state, timestamp: float):
        if not self.enabled:
            return

        ranges, shape_indices, site_pos = self.caster.cast(state, self.rays)
        raw_hit_count = int(np.count_nonzero(ranges > 0.0))

        ranges[ranges <= 0.0] = float("inf")
        ranges[(ranges < self.config.range_min) | (ranges > self.config.range_max)] = float("inf")
        range_hit_count = int(np.count_nonzero(ranges < float("inf")))

        # SensorTiledCamera has no bodyexclude option, so filter hits on the mounting body.
        valid_shape = shape_indices < len(self.shape_body)
        hit_bodies = np.full(shape_indices.shape, NO_HIT_BODY, dtype=np.int32)
        hit_bodies[valid_shape] = self.shape_body[shape_indices[valid_shape].astype(np.int64)]
        ranges[hit_bodies == self.body_id] = float("inf")
        self_filtered_count = int(np.count_nonzero(ranges < float("inf")))

        if self._all_inf_debug_logs < MAX_ALL_INF_DEBUG_LOGS and not np.any(ranges < float("inf")):
            self._all_inf_debug_logs += 1
            self.node.get_logger().warn(
                "Newton LiDAR produced all inf ranges "
                f"(raw_hits={raw_hit_count}, in_range={range_hit_count}, "
                f"after_self_filter={self_filtered_count}, site_z={site_pos[2]:.3f})"
            )

        msg = LaserScan()
        msg.header.stamp = sim_time_stamp(timestamp)
        msg.header.frame_id = self.config.frame_id
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = self.angle_increment
        msg.time_increment = self.time_increment
        msg.scan_time = self.scan_time
        msg.range_min = self.config.range_min
        msg.range_max = self.config.range_max
        msg.ranges = ranges.astype(np.float32).tolist()
        self.pub.publish(msg)

    def get_static_transforms(self, stamp):
        if not self.enabled or self.site_index < 0:
            return []
        return [site_static_transform(stamp, self.model, self.site_index, self.config.frame_id)]
