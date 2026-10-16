"""Transform, quaternion and ROS geometry-message helpers used by sensor backends."""

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Quaternion, TransformStamped, Vector3
from scipy.spatial.transform import Rotation as R_scipy

NANOSECONDS_PER_SECOND = 1e9


def sim_time_stamp(timestamp: float) -> Time:
    """Build a ROS stamp from simulated seconds.

    Every message a simulator publishes must be stamped on the same clock it puts on
    ``/clock``; the navigation stack runs with ``use_sim_time`` and will not match a
    sensor stamped from the wall clock against a TF stamped from simulated time.
    """
    stamp = Time()
    seconds = int(timestamp)
    stamp.sec = seconds
    stamp.nanosec = int((timestamp - seconds) * NANOSECONDS_PER_SECOND)
    return stamp


def _xyz_kwargs(xyz) -> dict[str, float]:
    return {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}


def vector3_msg(xyz) -> Vector3:
    return Vector3(**_xyz_kwargs(xyz))


def point_msg(xyz) -> Point:
    return Point(**_xyz_kwargs(xyz))


def quaternion_msg(quat_xyzw) -> Quaternion:
    return Quaternion(
        x=float(quat_xyzw[0]), y=float(quat_xyzw[1]),
        z=float(quat_xyzw[2]), w=float(quat_xyzw[3])
    )


def make_transform(stamp, parent: str, child: str, translation, quat_xyzw) -> TransformStamped:
    transform = TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = parent
    transform.child_frame_id = child
    transform.transform.translation = vector3_msg(translation)
    transform.transform.rotation = quaternion_msg(quat_xyzw)
    return transform


def quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return R_scipy.from_matrix(matrix).as_quat()


def quat_xyzw_to_rpy(quat_xyzw) -> np.ndarray:
    """Roll, pitch, yaw [rad] (ZYX Tait-Bryan) of an xyzw quaternion."""
    return R_scipy.from_quat(quat_xyzw).as_euler("xyz").astype(np.float32)


def yaw_from_quat_xyzw(quat_xyzw) -> float:
    return float(quat_xyzw_to_rpy(quat_xyzw)[2])


def rotate_world_to_body(quat_xyzw, vec_world) -> np.ndarray:
    """Express a world-frame vector in the body frame of an xyzw orientation."""
    return R_scipy.from_quat(quat_xyzw).inv().apply(vec_world).astype(np.float32)
