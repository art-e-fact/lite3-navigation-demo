"""Small helpers shared by the Newton simulator, its ROS bridge and its sensors."""

import numpy as np

# A Newton pose (body_q, shape_transform, joint_q root) is 7 floats: xyz then quaternion xyzw.
POSE_POSITION = slice(0, 3)
POSE_QUATERNION = slice(3, 7)


def log(node_or_logger, level: str, message: str, tag_logger: bool = True) -> None:
    """Send *message* to a ROS node's logger (or a bare logger) at *level*, or print it.

    *node_or_logger* may be a ROS node (``.get_logger()``), a logger with ``.info()`` /
    ``.warn()``, or ``None``. The printed fallback is always tagged "[LEVEL] ";
    *tag_logger* adds the same tag to the logger line as well.
    """
    tagged = f"[{level.upper()}] {message}"
    logger = resolve_logger(node_or_logger)
    if logger is not None:
        getattr(logger, level)(tagged if tag_logger else message)
    else:
        print(tagged)


def resolve_logger(node_or_logger):
    """The logger behind a ROS node, a bare logger as-is, or ``None``."""
    get_logger = getattr(node_or_logger, "get_logger", None)
    return get_logger() if get_logger is not None else node_or_logger


def step_interval(frequency_hz: float, dt: float) -> int:
    """Physics steps between two events at *frequency_hz*, rounded, at least 1."""
    return max(1, int(round(1.0 / (frequency_hz * dt))))


def label_text(label) -> str:
    """A Newton label (bytes or str) as text."""
    return label.decode() if isinstance(label, bytes) else str(label)


def label_short_name(label) -> str:
    """The last segment of a Newton label path, e.g. "robot/TORSO" -> "TORSO"."""
    text = label_text(label)
    return text.rsplit("/", 1)[-1] or text


def split_pose(pose) -> tuple[np.ndarray, np.ndarray]:
    """A 7-float Newton pose as (position xyz, quaternion xyzw) float64 copies."""
    values = np.asarray(pose, dtype=np.float64)
    return values[POSE_POSITION].copy(), values[POSE_QUATERNION].copy()


if __name__ == "__main__":
    assert step_interval(10.0, 0.004) == 25
    assert step_interval(15.0, 0.004) == 17  # 16.67 rounds up
    assert step_interval(1000.0, 0.004) == 1  # faster than the physics never goes below 1

    class _Recorder:
        def info(self, message):
            self.line = message

    class _Node:
        def __init__(self):
            self.logger = _Recorder()

        def get_logger(self):
            return self.logger

    recorder = _Recorder()
    log(recorder, "info", "hi")
    assert recorder.line == "[INFO] hi"
    log(recorder, "info", "hi", tag_logger=False)
    assert recorder.line == "hi"
    node = _Node()
    log(node, "info", "via node")
    assert node.logger.line == "[INFO] via node"

    assert label_short_name(b"robot/TORSO") == "TORSO"
    assert label_short_name("TORSO") == "TORSO"
    position, quaternion = split_pose([1, 2, 3, 0, 0, 0, 1])
    assert position.tolist() == [1.0, 2.0, 3.0] and quaternion.tolist() == [0.0, 0.0, 0.0, 1.0]
    print("utils self-check OK")
