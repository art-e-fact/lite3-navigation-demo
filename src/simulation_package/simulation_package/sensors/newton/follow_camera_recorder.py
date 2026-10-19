"""Offscreen third-person follow camera video recorder for the Newton backend.

Mirrors ``sensors/mujoco/follow_camera_recorder.FollowCameraRecorder``: an
offscreen RGB camera that trails the robot base and writes an mp4.

Frames come from Newton's GL viewer via ``ViewerGL.get_frame`` so the video looks
like the Newton screen, headless or not (like MuJoCo's EGL recorder):

* ``headless: false``: the interactive window, at its own resolution. Each frame
  is drawn with the viewer's own renderer at the follow pose but never presented,
  and the user's camera is put back, so the window stays free to orbit. The mp4
  is locked to the window's resolution at start; if the framebuffer later changes
  size (e.g. a macOS Retina backing-scale change when the window moves between
  displays), later frames are resampled back to the locked size instead of crashing.
* ``headless: true``: a private EGL-backed ``ViewerGL`` at ``width`` x ``height``
  that needs no display server.  The physics loop never sees it, so the CUDA
  graph stays enabled.

If no GL context can be created at all, it falls back to ``SensorTiledCamera``
(pure Warp raytracer, works on the CPU device too, but slow there).

Camera placement replicates MuJoCo's free-camera semantics so both backends
frame the robot the same way:

    lookat  = smoothed(robot position) + target_height_m
    azimuth = robot yaw + azimuth_offset_deg
    forward = (cos(el)cos(az), cos(el)sin(az), sin(el))
    position = lookat - distance_m * forward

``SensorTiledCamera`` uses the OpenGL camera convention (+X right, +Y up,
-Z forward), which is also what ``sensors/common/transforms`` encodes.
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
from newton._src.sensors.sensor_tiled_camera import SensorTiledCamera
import numpy as np
from PIL import Image
import warp as wp

from sensors.common.transforms import yaw_from_quat_xyzw
from sensors.newton.geometry import camera_transforms
from simulation_config import FollowCameraConfig
from utils import log, resolve_logger, split_pose

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
# libx264 macro block edge [px]: frames are cropped to a multiple of it (800x600 is not
# a multiple of 16, so 8 rather than imageio's default 16).
VIDEO_MACRO_BLOCK_PX = 8
# Below this |forward x up| the camera looks straight up/down and "right" is undefined.
PARALLEL_EPSILON = 1e-6


def _floor_to_macro_block(pixels: int) -> int:
    return pixels - pixels % VIDEO_MACRO_BLOCK_PX


def _look_rotation(forward: np.ndarray) -> np.ndarray:
    """Camera rotation matrix (columns = camera X/Y/Z in world) looking along *forward*."""
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, WORLD_UP)
    norm = np.linalg.norm(right)
    if norm < PARALLEL_EPSILON:  # looking straight up/down: pick an arbitrary stable right vector
        right = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    else:
        right = right / norm
    z_axis = -forward
    up = np.cross(z_axis, right)
    return np.column_stack([right, up, z_axis])


class NewtonFollowCameraRecorder:
    """Render a follow camera into an mp4 while the Newton simulation runs.

    Args:
        model: finalized ``newton.Model``.
        node_or_logger: ROS node (``.get_logger()``), a bare logger, or ``None``.
        config: ``simulation_config.FollowCameraConfig``.
        robot_body_index: index into ``state.body_q`` of the robot base body.
        bvh: shared ``sensors.newton.bvh.NewtonBvh`` (or ``None`` to own one).
        viewer: open ``newton.viewer.ViewerGL`` to grab frames from, or ``None``
            to raytrace offscreen.
    """

    def __init__(
        self,
        model,
        node_or_logger,
        config: FollowCameraConfig | None = None,
        robot_body_index: int = 0,
        bvh=None,
        viewer=None,
    ):
        self.model = model
        self.logger = resolve_logger(node_or_logger)
        self.config = config or FollowCameraConfig()
        self.robot_body_index = int(robot_body_index)
        self.bvh = bvh
        self.viewer = viewer if hasattr(viewer, "get_frame") else None
        self._owns_viewer = False
        self.enabled = bool(self.config.enabled)
        self.video_path = str(self.config.video_path).strip()
        self.fps = float(self.config.fps)
        # FollowCameraConfig.fov_deg defaults to MuJoCo's free-camera fovy (45 deg) so
        # both backends frame the robot the same way.
        self.fov_deg = float(self.config.fov_deg)
        self.closed = True
        self.frame_count = 0
        self._lookat = None

        if not self.enabled:
            self._log_info("Newton follow camera recorder disabled")
            self.sensor = None
            self.writer = None
            return

        if not self.video_path:
            raise ValueError("video_path must be set when the camera is enabled")

        output_path = Path(self.video_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.video_path = str(output_path)

        width = int(self.config.width)
        height = int(self.config.height)
        self.sensor = None
        if self.viewer is None:
            self.viewer = _create_headless_viewer(model, width, height, self.logger)
            self._owns_viewer = self.viewer is not None
        if self.viewer is not None:
            # The mp4 takes the viewer's window size; width/height size the raytracer.
            width, height = self.viewer.renderer.window.get_framebuffer_size()
            width, height = _floor_to_macro_block(width), _floor_to_macro_block(height)  # see render()
            source = "Newton viewer (EGL)" if self._owns_viewer else "Newton viewer"
        else:
            self.sensor = SensorTiledCamera(model, load_textures=False)
            self.sensor.default_render_config.enable_shadows = True
            # Textures are a CPU-device liability and the scene has none worth showing.
            self.sensor.default_render_config.enable_textures = False
            self.sensor.utils.create_default_light(enable_shadows=True)
            self.rays = self.sensor.utils.compute_camera_rays_pinhole(
                width, height, camera_fovs=math.radians(self.fov_deg)
            )
            self.color_image = self.sensor.utils.create_color_image_output(
                width, height, camera_count=1
            )
            source = "Warp raytracer"

        self.frame_size = (height, width)  # locked (H, W); later frames are resampled to this
        self._resize_warned = False
        self.writer = imageio.get_writer(
            self.video_path,
            fps=self.fps,
            codec="libx264",
            quality=self.config.quality,
            macro_block_size=VIDEO_MACRO_BLOCK_PX,
        )
        self.closed = False
        self._log_info(
            f"Newton follow camera recording enabled: {self.video_path} "
            f"({width}x{height} @ {self.fps:.1f} Hz, fov {self.fov_deg:.0f} deg, {source})"
        )

    def render(self, state, refit: bool = True, timestamp: float = 0.0,
               log_state: bool = True) -> np.ndarray | None:
        """Render one RGB frame (H, W, 3) uint8 without writing it to the video.

        ``log_state=False`` when another camera already logged *state* into the shared viewer.
        """
        if not self.enabled or self.closed:
            return None
        position, forward, azimuth, elevation = self._camera_pose(state)

        if self.viewer is not None:
            viewer, camera = self.viewer, self.viewer.camera
            own_view = (wp.vec3(*camera.pos), camera.pitch, camera.yaw)
            viewer.set_camera(wp.vec3(*position), math.degrees(elevation), math.degrees(azimuth))
            if log_state:
                viewer.begin_frame(timestamp)
                viewer.log_state(state)
            # The viewer's own draw call without presenting it, so the window (shared
            # with the user and the other camera) keeps its own camera.
            viewer.renderer.render(camera, viewer.objects, viewer.lines, viewer.wireframe_shapes, viewer.arrows)
            frame = viewer.get_frame().numpy()
            viewer.set_camera(*own_view)
            # Crop to the writer's macro block so odd window sizes are not resampled.
            frame_height, frame_width = frame.shape[:2]
            return np.ascontiguousarray(
                frame[: _floor_to_macro_block(frame_height), : _floor_to_macro_block(frame_width)]
            )

        if refit and self.bvh is not None:
            self.bvh.refit(state)
        elif refit:
            self.model.bvh_refit_shapes(state)

        transforms = camera_transforms(position, _look_rotation(forward), self.model.world_count)
        self.sensor.update(
            state,
            transforms,
            self.rays,
            color_image=self.color_image,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )
        rgba = self.sensor.utils.to_rgba_from_color(self.color_image).numpy()
        return np.ascontiguousarray(rgba[0, :, :, :3])

    def update(self, state, refit: bool = True, timestamp: float = 0.0, log_state: bool = True) -> None:
        """Render one frame and append it to the video."""
        frame = self.render(state, refit=refit, timestamp=timestamp, log_state=log_state)
        if frame is None:
            return
        if frame.shape[:2] != self.frame_size:
            # The viewer's framebuffer resized mid-run (e.g. a macOS Retina scale change);
            # resample back to the size the writer was opened with instead of crashing.
            locked_height, locked_width = self.frame_size
            if not self._resize_warned:
                self._resize_warned = True
                self._log_info(
                    f"Newton follow camera frame size changed to "
                    f"{frame.shape[1]}x{frame.shape[0]}; resampling to {locked_width}x{locked_height}"
                )
            frame = np.asarray(Image.fromarray(frame).resize((locked_width, locked_height), Image.BILINEAR))
        self.writer.append_data(frame)
        self.frame_count += 1

    def reset(self) -> None:
        """Forget the smoothed look-at point (used after a warm-up rewind)."""
        self._lookat = None

    def close(self) -> None:
        if not self.enabled or self.closed:
            return
        self.closed = True
        self.writer.close()
        if self._owns_viewer:
            self.viewer.close()
        self._log_info(
            f"Newton follow camera video saved: {self.video_path} ({self.frame_count} frames)"
        )

    # -- camera placement ---------------------------------------------------

    def _camera_pose(self, state) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Camera position, forward unit vector, azimuth and elevation (radians)."""
        target, quat_xyzw = split_pose(state.body_q.numpy()[self.robot_body_index])
        target[2] += float(self.config.target_height_m)
        if self._lookat is None:
            self._lookat = target
        else:
            alpha = float(np.clip(self.config.smoothing, 0.0, 1.0))
            self._lookat = (1.0 - alpha) * self._lookat + alpha * target

        yaw = yaw_from_quat_xyzw(quat_xyzw) if self.config.follow_yaw else 0.0
        azimuth = yaw + math.radians(self.config.azimuth_offset_deg)
        elevation = math.radians(self.config.elevation_deg)
        forward = np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float64,
        )
        position = self._lookat - float(self.config.distance_m) * forward
        return position, forward, azimuth, elevation

    # -- logging ------------------------------------------------------------

    def _log_info(self, message: str) -> None:
        log(self.logger, "info", message)


def _create_headless_viewer(model, width: int, height: int, logger):
    """Invisible EGL-backed ``ViewerGL`` (no display server needed), or ``None``."""
    try:
        import pyglet

        pyglet.options["headless"] = True
        import newton.viewer

        viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
        viewer.set_model(model)
        return viewer
    except Exception as exc:  # no EGL/GL at all: fall back to the raytracer
        log(logger, "warn", f"No headless GL viewer for the follow camera, raytracing instead: {exc}")
        return None
