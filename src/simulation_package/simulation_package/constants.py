"""Names shared by the Newton simulator, its ROS bridge and its sensors."""

# Robot base TF frame; sensor frames hang off it and odometry reports it.
BASE_FRAME = "base_link"
# Odometry TF frame.
ODOM_FRAME = "odom"
# Lite3 MJCF root body; the follow camera and Rerun key off it.
LITE3_ROOT_BODY = "TORSO"
# Publisher queue depth for sensor topics: ~1 s of backlog at the 10 Hz scan rate.
SENSOR_QUEUE_DEPTH = 10

# Newton free joint: 7 generalised coordinates (xyz + quaternion) and 6 velocities.
FREE_JOINT_Q_COUNT = 7
FREE_JOINT_QD_COUNT = 6
# (x, y, yaw) of the robot spawn in a static scene.
DEFAULT_ROBOT_START_POSE = (-5.0, 0.0, 0.0)
