import math
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

import newton
import newton.examples
import warp as wp
from newton import JointTargetMode
from newton.sensors import SensorIMU

from constants import (
    DEFAULT_ROBOT_START_POSE,
    FREE_JOINT_ANGVEL,
    FREE_JOINT_LINVEL,
    FREE_JOINT_Q_COUNT,
    FREE_JOINT_QD_COUNT,
    LITE3_ROOT_BODY,
)
from sensors.common.transforms import rotate_world_to_body
from sensors.newton.lidar_sensor import NewtonLidarSensor
from simulation_config import MJCF_SUFFIXES, SensorsConfig
from utils import POSE_POSITION, POSE_QUATERNION, log


DT = 0.004
ROS_SPIN_EVERY_STEPS = 1
LEG_COUNT = 4
# Lite3 standing pose of one leg (hip abduction, hip, knee) [rad]; every leg is the same.
LITE3_LEG_STANCE_RAD = (0.0, -1.35453, 2.54948)
# Per-DOF |tau| clamp [Nm], mirrors the MJCF actuator ctrlrange.
LITE3_EFFORT_LIMIT_NM = 30.0
# M20 leg: three leg joints and a wheel.
M20_LEG_EFFORT_LIMIT_NM = (76.4, 76.4, 76.4, 21.6)
DEFAULT_STIFFNESS = 30.0
DEFAULT_DAMPING = 1.0
# Rotor inertia the MJCFs omit. The PD torque is integrated explicitly (host side,
# into control.joint_f), so a DOF is only stable while kd*DT/M_ii < 2; the Lite3
# knee's M_ii is 0.0041 kg m^2, and rl_deploy's stand-up gains (kd=2.5) put it at
# 2.46 at DT=0.004 -- the legs thrashed at 30+ rad/s. 0.01 kg m^2 is a geared
# actuator's reflected rotor inertia and brings the worst DOF back to 0.71,
# which is what MuJoCo instead buys by running the same PD at DT=0.001.
ARMATURE = 0.01
# Each pyramidal condim=3 contact costs 4 rows in MuJoCo's constraint solver.
CONSTRAINT_ROWS_PER_CONTACT = 4
# Joint-limit penalty stiffness and damping; hand-tuned, no derivation recorded.
JOINT_LIMIT_STIFFNESS = 1.0e2
JOINT_LIMIT_DAMPING = 1.0e0
# Default contact stiffness ke, damping kd and friction stiffness kf; hand-tuned.
CONTACT_STIFFNESS = 5.0e4
CONTACT_DAMPING = 5.0e2
CONTACT_FRICTION_STIFFNESS = 1.0e3
# Default Coulomb friction coefficient; hand-tuned.
CONTACT_FRICTION = 0.75
# Standard gravity [m/s^2], applied along -Z.
GRAVITY_MPS2 = 9.81
# State arrays captured at construction and restored by rewind().
STATE_ARRAYS = ("joint_q", "joint_qd", "body_q", "body_qd")

# M20 sim-to-policy frame, mirrors mujoco_simulation_ros2.py (DdsInterface::Handler).
M20_JOINT_DIR = np.array(
    [1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1],
    dtype=np.float32,
)
M20_POS_OFFSET_RAD = np.radians(
    [-25, -131, 160, 0, 25, -131, 160, 0, -25, 131, -160, 0, 25, 131, -160, 0]
).astype(np.float32)


@dataclass(frozen=True)
class RobotProfile:
    name: str
    base_height: float
    default_joint_pos: np.ndarray
    effort_limit: np.ndarray  # per-DOF |tau| clamp, mirrors MJCF actuator ctrlrange
    root_body: str  # MJCF root body name; the follow camera and Rerun key off it
    joint_dir: np.ndarray | None = None  # pub<->raw joint calibration, None = identity
    pos_offset_rad: np.ndarray | None = None

    @property
    def num_dofs(self) -> int:
        return len(self.default_joint_pos)


ROBOT_PROFILES: dict[str, RobotProfile] = {
    "Lite3": RobotProfile(
        name="Lite3",
        base_height=0.43,
        default_joint_pos=np.tile(LITE3_LEG_STANCE_RAD, LEG_COUNT).astype(np.float32),
        effort_limit=np.full(len(LITE3_LEG_STANCE_RAD) * LEG_COUNT, LITE3_EFFORT_LIMIT_NM, dtype=np.float32),
        root_body=LITE3_ROOT_BODY,
    ),
    "M20": RobotProfile(
        name="M20",
        base_height=1.0,
        default_joint_pos=np.array(
            [-0.438, -1.16, 2.76, 0, 0.438, -1.16, 2.76, 0, -0.438, 1.16, -2.76, 0, 0.438, 1.16, -2.76, 0],
            dtype=np.float32,
        ),
        effort_limit=np.tile(M20_LEG_EFFORT_LIMIT_NM, LEG_COUNT).astype(np.float32),
        root_body="base_link",
        joint_dir=M20_JOINT_DIR,
        pos_offset_rad=M20_POS_OFFSET_RAD,
    ),
}


@dataclass
class JointCommand:
    """Per-DOF PD gains, targets and feed-forward torque."""

    kp: np.ndarray
    kd: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray

    @classmethod
    def standing(cls, profile: RobotProfile) -> "JointCommand":
        """Fresh arrays holding the robot's standing pose with the default gains."""
        num_dofs = profile.num_dofs
        return cls(
            kp=np.full(num_dofs, DEFAULT_STIFFNESS, dtype=np.float32),
            kd=np.full(num_dofs, DEFAULT_DAMPING, dtype=np.float32),
            position=profile.default_joint_pos.copy(),
            velocity=np.zeros(num_dofs, dtype=np.float32),
            torque=np.zeros(num_dofs, dtype=np.float32),
        )

    def copy(self) -> "JointCommand":
        return JointCommand(**{field.name: getattr(self, field.name).copy() for field in fields(self)})

    def assign(self, other: "JointCommand") -> None:
        """Copy *other*'s values into this command's arrays in place."""
        for field in fields(self):
            getattr(self, field.name)[:] = getattr(other, field.name)


@dataclass
class SimulationState:
    position: np.ndarray
    quat_xyzw: np.ndarray
    linvel_body: np.ndarray
    angvel_body: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    imu_acc: np.ndarray  # specific force at imu_site, sensor frame
    imu_gyro: np.ndarray  # angular rate at imu_site, sensor frame


def create_newton_viewer():
    parser = newton.examples.create_parser()
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        viewer, _ = newton.examples.init(parser)
        return viewer
    finally:
        sys.argv = saved_argv


def warp_to_numpy(array, fallback_size: int) -> np.ndarray:
    if array is None:
        return np.zeros(fallback_size, dtype=np.float32)
    try:
        return np.asarray(array.numpy(), dtype=np.float32).copy()
    except Exception:
        try:
            return np.asarray(array, dtype=np.float32).copy()
        except Exception:
            return np.zeros(fallback_size, dtype=np.float32)


class NewtonSimulation:
    def __init__(
        self,
        model_path: str,
        scene_path: str | None = None,
        headless: bool = True,
        viewer=None,
        logger=None,
        sensor_options: SensorsConfig | None = None,
        profile: RobotProfile | None = None,
        start_pose=None,
    ):
        self.model_path = model_path
        self.scene_path = scene_path
        self.headless = headless
        self.viewer = None if headless else viewer
        self.logger = logger
        self.sensor_options = sensor_options
        self.profile = profile if profile is not None else ROBOT_PROFILES["Lite3"]
        self.num_dofs = self.profile.num_dofs
        # Where the actuated DOFs sit in joint_q / joint_qd, after the free joint.
        self._dof_q = slice(FREE_JOINT_Q_COUNT, FREE_JOINT_Q_COUNT + self.num_dofs)
        self._dof_qd = slice(FREE_JOINT_QD_COUNT, FREE_JOINT_QD_COUNT + self.num_dofs)
        self.device = wp.get_device()
        self.timestamp = 0.0
        self.step_count = 0

        self.robot_start_pose = DEFAULT_ROBOT_START_POSE
        if start_pose is not None:
            # robot.start_pose from the config: fields set there win over the default start.
            self.robot_start_pose = start_pose.apply(self.robot_start_pose)

        self.command = JointCommand.standing(self.profile)
        self.last_tau = np.zeros(self.num_dofs, dtype=np.float32)

        self.model, self.solver, self.collision_pipeline, self.state_0, self.state_1, self.control = self._build_newton_model(model_path, scene_path)
        self.contacts = self.collision_pipeline.contacts()
        self.graph = None
        self.use_cuda_graph = False
        self._joint_f_buffer = np.zeros(self.model.joint_dof_count, dtype=np.float32)

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.vsync = True
            # newton.examples.init() shows a splash that examples.run() would normally hide.
            if hasattr(self.viewer, "hide_loading_splash"):
                self.viewer.hide_loading_splash()

        self._setup_cuda_graph()
        # A warm-up rewind restores exactly the world the first published step would have seen.
        self._initial_state = self._capture_state()
        self._log_info(f"Newton {self.profile.name} simulation loaded: {self.model_path}")
        if self.scene_path is not None:
            self._log_info(f"Newton environment scene loaded: {self.scene_path}")
        if self.viewer is not None:
            self._log_info("Newton viewer enabled because headless is false.")
        self._log_info("Optional sensors are managed by the ROS runner.")

    def set_command(self, command: JointCommand):
        self.command.assign(command)

    def _capture_state(self) -> dict:
        values = {}
        for name in STATE_ARRAYS:
            array = getattr(self.state_0, name, None)
            if array is not None:
                values[name] = array.numpy().copy()
        return values

    def rewind(self):
        """Restore the state captured at construction and zero the clock.

        Used after the warm-up pass so JIT compilation does not leak into the
        published trajectory. Both state buffers are written because an uncaptured
        step swaps them.
        """
        for state in (self.state_0, self.state_1):
            for name, values in self._initial_state.items():
                array = getattr(state, name, None)
                if array is not None:
                    array.assign(values)
        self.timestamp = 0.0
        self.step_count = 0
        self.last_tau[:] = 0.0
        self._joint_f_buffer[:] = 0.0
        self.control.joint_f.assign(self._joint_f_buffer)

    def step(self):
        """Advance one physics step."""
        self._apply_control()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_physics_step(apply_viewer_forces=True, keep_state_buffers=False)
        self.timestamp += DT
        self.step_count += 1

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.timestamp)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None and hasattr(self.viewer, "log_contacts"):
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def state_snapshot(self) -> SimulationState:
        joint_q, joint_qd = self._joint_state()
        quat_xyzw = joint_q[POSE_QUATERNION]
        self.imu.update(self.state_0)
        return SimulationState(
            position=joint_q[POSE_POSITION],
            quat_xyzw=quat_xyzw,
            linvel_body=rotate_world_to_body(quat_xyzw, joint_qd[FREE_JOINT_LINVEL]),
            angvel_body=rotate_world_to_body(quat_xyzw, joint_qd[FREE_JOINT_ANGVEL]),
            joint_position=joint_q[self._dof_q],
            joint_velocity=joint_qd[self._dof_qd],
            # .numpy() aliases the sensor's persistent buffer on CPU devices, so copy out.
            imu_acc=self.imu.accelerometer.numpy()[0].copy(),
            imu_gyro=self.imu.gyroscope.numpy()[0].copy(),
        )

    def _joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Host copies of the generalised coordinates and velocities (free joint + DOFs)."""
        joint_q = warp_to_numpy(self.state_0.joint_q, FREE_JOINT_Q_COUNT + self.num_dofs)
        joint_qd = warp_to_numpy(self.state_0.joint_qd, FREE_JOINT_QD_COUNT + self.num_dofs)
        return joint_q, joint_qd

    def _setup_cuda_graph(self):
        if not self.device.is_cuda:
            self._log_info("CUDA graph disabled because Newton is not running on a CUDA device.")
            return
        if not wp.is_mempool_enabled(self.device):
            self._log_info("CUDA graph disabled because the Warp memory pool is not enabled.")
            return
        try:
            with wp.ScopedCapture() as capture:
                self._simulate_physics_step(apply_viewer_forces=True, keep_state_buffers=True)
            self.graph = capture.graph
            self.use_cuda_graph = True
            self._log_info("Using CUDA graph for Newton physics stepping.")
        except Exception as exc:
            self.graph = None
            self.use_cuda_graph = False
            self._log_warn(f"CUDA graph capture failed; falling back to uncaptured Newton stepping: {exc}")

    def _build_newton_model(self, model_path: str, scene_path: str | None = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Lite3 model description not found: {model_path}")

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=ARMATURE, limit_ke=JOINT_LIMIT_STIFFNESS, limit_kd=JOINT_LIMIT_DAMPING
        )
        builder.default_shape_cfg.ke = CONTACT_STIFFNESS
        builder.default_shape_cfg.kd = CONTACT_DAMPING
        builder.default_shape_cfg.kf = CONTACT_FRICTION_STIFFNESS
        builder.default_shape_cfg.mu = CONTACT_FRICTION

        scene_loaded = self._add_environment_scene(builder, scene_path, model_path)
        scene_shape_count = builder.shape_count

        if Path(model_path).suffix.lower() == ".xml":
            builder.add_mjcf(
                model_path,
                xform=wp.transform_identity(),
                collapse_fixed_joints=True,
                enable_self_collisions=False,
                parse_visuals=True,
                parse_meshes=True,
                parse_sites=True,
                ignore_names=("floor",),
            )
        else:
            builder.add_usd(
                model_path,
                xform=wp.transform_identity(),
                collapse_fixed_joints=True,
                enable_self_collisions=False,
                joint_ordering="dfs",
                hide_collision_shapes=True,
            )

        self._init_sensor_visuals(builder)
        # Hull robot meshes only. Environment meshes stay triangle colliders;
        # convex-hulling a landscape wraps it in one blob.
        # keep_visual_shapes: ray sensors (SensorTiledCamera) cast against render meshes,
        # which plain convex-hulling would discard.
        hull_indices = list(range(scene_shape_count, builder.shape_count))
        if hull_indices:
            builder.approximate_meshes("convex_hull", shape_indices=hull_indices, keep_visual_shapes=True)
        if not scene_loaded:
            builder.add_ground_plane()

        start_x, start_y, start_yaw = (float(value) for value in self.robot_start_pose)
        builder.joint_q[POSE_POSITION] = [start_x, start_y, self.profile.base_height]
        builder.joint_q[POSE_QUATERNION] = [0.0, 0.0, math.sin(0.5 * start_yaw), math.cos(0.5 * start_yaw)]
        builder.joint_q[self._dof_q] = self.profile.default_joint_pos.tolist()

        for dof_index in range(self.num_dofs):
            target_index = FREE_JOINT_QD_COUNT + dof_index
            # PD runs on CPU and enters via control.joint_f; solver-side servo stays off.
            builder.joint_target_ke[target_index] = 0.0
            builder.joint_target_kd[target_index] = 0.0
            builder.joint_armature[target_index] = ARMATURE
            builder.joint_target_mode[target_index] = int(JointTargetMode.NONE)

        model = builder.finalize()
        model.set_gravity((0.0, 0.0, -GRAVITY_MPS2))
        # Matches the MJCF's <accelerometer>/<gyro> on imu_site, which Newton's MJCF importer
        # ignores. Must precede model.state(): the sensor requests the body_qdd state attribute.
        self.imu = SensorIMU(model, sites="*imu_site")
        # Newton generates the contacts (use_mujoco_contacts=False) so the environment stays a
        # triangle collider. MuJoCo convexifies every mesh geom it collides itself, which turns a
        # lidar landscape into a ~300-vertex blob no matter what approximate_meshes() did.
        # Static-static and static-kinematic pairs are pure overhead in the explicit broad phase.
        collision_pipeline = newton.CollisionPipeline(model, include_static_kinematic_pairs=False)
        # MuJoCo transfers only min(rigid_contact_max, naconmax) contacts per step, so sizing
        # nconmax to the pipeline's own cap means a Newton contact can never be silently dropped.
        nconmax = collision_pipeline.rigid_contact_max
        njmax = CONSTRAINT_ROWS_PER_CONTACT * (nconmax + model.joint_dof_count)
        solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            solver="newton",
            use_mujoco_contacts=False,
            nconmax=nconmax,
            njmax=njmax,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        if control.joint_f is None:
            raise RuntimeError("Newton Control.joint_f is unavailable; cannot apply joint torques")
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        newton.eval_fk(model, state_1.joint_q, state_1.joint_qd, state_1)
        return model, solver, collision_pipeline, state_0, state_1, control

    def _add_environment_scene(self, builder, scene_path: str | None, model_path: str) -> bool:
        if not scene_path:
            return False

        scene_file = Path(scene_path)
        if scene_file.resolve() == Path(model_path).resolve():
            return False
        if scene_file.suffix.lower() not in MJCF_SUFFIXES:
            return False

        builder.add_mjcf(
            str(scene_file),
            xform=wp.transform_identity(),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            parse_visuals=True,
            parse_meshes=True,
            parse_sites=False,
        )
        return True

    def _init_sensor_visuals(self, builder):
        options = self.sensor_options
        if options is not None and options.lidar_2d.enabled:
            NewtonLidarSensor.init_visuals(builder, options.lidar_2d)

    def _simulate_physics_step(self, apply_viewer_forces: bool, keep_state_buffers: bool):
        self.state_0.clear_forces()
        if apply_viewer_forces and self.viewer is not None and hasattr(self.viewer, "apply_forces"):
            self.viewer.apply_forces(self.state_0)
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)
        if keep_state_buffers:
            self.state_0.assign(self.state_1)
        else:
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _apply_control(self):
        joint_q, joint_qd = self._joint_state()
        command = self.command
        tau = (
            command.kp * (command.position - joint_q[self._dof_q])
            + command.kd * (command.velocity - joint_qd[self._dof_qd])
            + command.torque
        )
        limit = self.profile.effort_limit
        self.last_tau = np.clip(tau, -limit, limit)

        self._joint_f_buffer[self._dof_qd] = self.last_tau
        self.control.joint_f.assign(self._joint_f_buffer)

    def _log_info(self, message: str):
        log(self.logger, "info", message, tag_logger=False)

    def _log_warn(self, message: str):
        log(self.logger, "warn", message, tag_logger=False)
