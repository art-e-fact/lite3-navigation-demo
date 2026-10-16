"""Shared LiDAR ray pattern helpers; sensor settings live in simulation_config."""

import math
import numpy as np


def planar_lidar_dirs(num_rays: int) -> np.ndarray:
    """Unit ray directions in the sensor's XY plane, evenly spaced from -pi to pi."""
    angles = np.linspace(-math.pi, math.pi, num_rays, endpoint=False)
    return np.column_stack((
        np.cos(angles),
        np.sin(angles),
        np.zeros(num_rays),
    )).astype(np.float32)
