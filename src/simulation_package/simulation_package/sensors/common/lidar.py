"""Shared LiDAR constants and ray pattern helpers."""

import math
import numpy as np

# Sensor settings (sites, frames, topics, ranges) live in simulation_config; these are
# only the fallbacks for callers that build a sensor without a config.
NUM_RAYS = 360


def planar_lidar_dirs(num_rays: int = NUM_RAYS) -> np.ndarray:
    angles = np.linspace(-math.pi, math.pi, num_rays, endpoint=False)
    return np.column_stack((
        np.cos(angles),
        np.sin(angles),
        np.zeros(num_rays),
    )).astype(np.float32)
