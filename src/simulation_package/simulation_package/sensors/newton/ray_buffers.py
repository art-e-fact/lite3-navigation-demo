"""Warp ray buffers and site-mounted ray casting for Newton SensorTiledCamera."""

import numpy as np
import warp as wp
from newton._src.sensors.sensor_tiled_camera import SensorTiledCamera

from sensors.newton.geometry import camera_transforms, site_world_pose


def rays_from_dirs(local_dirs: np.ndarray, height: int = 1):
    dirs = np.asarray(local_dirs, dtype=np.float32).reshape(height, -1, 3)
    origins = np.zeros_like(dirs, dtype=np.float32)
    rays = np.stack((origins, dirs), axis=2)
    rays = rays.reshape(1, height, dirs.shape[1], 2, 3)
    return wp.array(rays, dtype=wp.vec3f)


class SiteRayCaster:
    """Casts one row of *ray_count* rays from a site shape, as the LiDAR sensors do."""

    def __init__(self, model, site_index: int, ray_count: int):
        self.model = model
        self.site_index = site_index
        self.sensor = SensorTiledCamera(model, load_textures=False)
        self.depth_image = self.sensor.utils.create_depth_image_output(ray_count, 1)
        self.shape_index_image = self.sensor.utils.create_shape_index_image_output(ray_count, 1)

    def cast(self, state, rays) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-ray range (0 for a miss) and hit shape index, plus the site's world position."""
        site_pos, site_rot, _ = site_world_pose(self.model, state, self.site_index)
        transforms = camera_transforms(site_pos, site_rot, self.model.world_count)
        self.sensor.update(
            state,
            transforms,
            rays,
            depth_image=self.depth_image,
            shape_index_image=self.shape_index_image,
        )
        ranges = self.depth_image.numpy()[0, 0, 0].astype(np.float32)
        shape_indices = self.shape_index_image.numpy()[0, 0, 0]
        return ranges, shape_indices, site_pos
