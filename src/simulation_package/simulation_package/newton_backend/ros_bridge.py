import numpy as np
import rclpy
from drdds.msg import ImuData, ImuDataValue, JointData, JointsData, JointsDataCmd, JointsDataValue, MetaType
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster

from constants import BASE_FRAME, ODOM_FRAME
from sensors.common.transforms import make_transform, point_msg, quat_xyzw_to_rpy, quaternion_msg, sim_time_stamp, vector3_msg
from simulation import ROBOT_PROFILES, JointCommand, RobotProfile

# drdds JointsData/JointsDataCmd carry a fixed 16 slots (the M20); the Lite3 fills 12.
JOINT_MSG_SLOTS = 16
# Joint name field: four ASCII spaces, i.e. a blank 4-character name.
BLANK_JOINT_NAME = [32, 32, 32, 32]
# drdds status_word the sim reports in every joint slot.
JOINT_STATUS_WORD = 1
# Motor and driver temperatures [deg C] the sim reports: fixed placeholders.
SIM_MOTOR_TEMP_C = 40.0
SIM_DRIVER_TEMP_C = 45.0
# Publisher/subscriber queue depths; IMU and joints publish every state step, so get the deepest.
IMU_QUEUE_DEPTH = 200
JOINTS_QUEUE_DEPTH = 200
ODOM_QUEUE_DEPTH = 50
CLOCK_QUEUE_DEPTH = 10
JOINTS_CMD_QUEUE_DEPTH = 10


def _meta_header(stamp) -> MetaType:
    """drdds message header: the sim has no hardware frame id, so it is always 0."""
    header = MetaType()
    header.frame_id = 0
    header.stamp = stamp
    return header


class NewtonRosBridge:
    def __init__(self, headless: bool, model_path: str, profile: RobotProfile | None = None):
        self.node = Node("newton_simulation")
        self.node.declare_parameter("headless", headless)
        self.node.declare_parameter("model_path", model_path)
        self._shutdown_requested = False

        self.profile = profile if profile is not None else ROBOT_PROFILES["Lite3"]
        self.num_dofs = self.profile.num_dofs
        # The latest command from /JOINTS_CMD, in the simulator's raw joint frame.
        self.command = JointCommand.standing(self.profile)

        self.imu_pub = self.node.create_publisher(ImuData, "/IMU_DATA", IMU_QUEUE_DEPTH)
        self.imu_std_pub = self.node.create_publisher(Imu, "/IMU_SIM", IMU_QUEUE_DEPTH) # adding "_SIM" to avoid publishing to the same topic in real robot
        self.joints_pub = self.node.create_publisher(JointsData, "/JOINTS_DATA", JOINTS_QUEUE_DEPTH)
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", ODOM_QUEUE_DEPTH)
        # The navigation stack runs with use_sim_time, so /clock is the only time source.
        self.clock_pub = self.node.create_publisher(Clock, "/clock", CLOCK_QUEUE_DEPTH)
        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.cmd_sub = self.node.create_subscription(JointsDataCmd, "/JOINTS_CMD", self._cmd_callback, JOINTS_CMD_QUEUE_DEPTH)

    def get_logger(self):
        return self.node.get_logger()

    def spin_once(self):
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def should_exit(self) -> bool:
        return self._shutdown_requested or not rclpy.ok()

    def destroy(self):
        self.node.destroy_node()

    def read_latest_action(self) -> JointCommand:
        return self.command.copy()

    def publish_clock(self, timestamp: float):
        """Publish simulated time. Called every step"""
        msg = Clock()
        msg.clock = sim_time_stamp(timestamp)
        self.clock_pub.publish(msg)

    def publish_state(self, timestamp: float, state, last_tau: np.ndarray):
        stamp = sim_time_stamp(timestamp)
        roll_deg, pitch_deg, yaw_deg = np.degrees(quat_xyzw_to_rpy(state.quat_xyzw))

        imu_msg = ImuData()
        imu_msg.header = _meta_header(stamp)
        imu_msg.data = ImuDataValue()
        imu_msg.data.roll = float(roll_deg)
        imu_msg.data.pitch = float(pitch_deg)
        imu_msg.data.yaw = float(yaw_deg)
        imu_msg.data.omega_x = float(state.imu_gyro[0])
        imu_msg.data.omega_y = float(state.imu_gyro[1])
        imu_msg.data.omega_z = float(state.imu_gyro[2])
        imu_msg.data.acc_x = float(state.imu_acc[0])
        imu_msg.data.acc_y = float(state.imu_acc[1])
        imu_msg.data.acc_z = float(state.imu_acc[2])
        self.imu_pub.publish(imu_msg)

        imu_std_msg = Imu()
        imu_std_msg.header.frame_id = '' # matches with /IMU from real M20
        imu_std_msg.header.stamp = stamp # use this for sim only
        # imu_std_msg.header.stamp = self.node.get_clock().now().to_msg() # use this for HIL -> TODO: somehow difference between timestamps of lidar vs imu topics causes drifting in localization by on-board SLAM binaries
        imu_std_msg.orientation = quaternion_msg(state.quat_xyzw)
        imu_std_msg.angular_velocity = vector3_msg(state.imu_gyro)
        imu_std_msg.linear_acceleration = vector3_msg(state.imu_acc)
        self.imu_std_pub.publish(imu_std_msg)

        joints_msg = JointsData()
        joints_msg.header = _meta_header(stamp)
        joints_msg.data = JointsDataValue()
        joints_msg.data.joints_data = [JointData() for _ in range(JOINT_MSG_SLOTS)]
        pub_pos, pub_vel, pub_tau = self._raw_to_pub(state.joint_position, state.joint_velocity, last_tau)
        for index, joint in enumerate(joints_msg.data.joints_data):
            joint.status_word = JOINT_STATUS_WORD
            if index >= self.num_dofs:
                continue
            joint.name = list(BLANK_JOINT_NAME)
            joint.data_id = 0
            joint.position = float(pub_pos[index])
            joint.velocity = float(pub_vel[index])
            joint.torque = float(pub_tau[index])
            joint.motion_temp = SIM_MOTOR_TEMP_C
            joint.driver_temp = SIM_DRIVER_TEMP_C
        self.joints_pub.publish(joints_msg)

    def publish_odom_and_tf(self, timestamp: float, state):
        stamp = sim_time_stamp(timestamp)

        transform = make_transform(stamp, ODOM_FRAME, BASE_FRAME, state.position, state.quat_xyzw)
        self.tf_broadcaster.sendTransform(transform)

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = ODOM_FRAME
        odom_msg.child_frame_id = BASE_FRAME
        odom_msg.pose.pose.position = point_msg(state.position)
        odom_msg.pose.pose.orientation = transform.transform.rotation
        odom_msg.twist.twist.linear = vector3_msg(state.linvel_body)
        odom_msg.twist.twist.angular = vector3_msg(state.angvel_body)
        self.odom_pub.publish(odom_msg)

    def _cmd_callback(self, msg: JointsDataCmd):
        if len(msg.data.joints_data) not in (self.num_dofs, JOINT_MSG_SLOTS):
            self.node.get_logger().warn("Received JointsDataCmd with incorrect number of joints")
            return

        joints = msg.data.joints_data[: self.num_dofs]
        self.command.kp[:] = [joint.kp for joint in joints]
        self.command.kd[:] = [joint.kd for joint in joints]
        self.command.torque[:] = [joint.torque for joint in joints]  # tau_ff no processing, matches MuJoCo bridge
        pub_pos = np.array([joint.position for joint in joints], dtype=np.float32)
        pub_vel = np.array([joint.velocity for joint in joints], dtype=np.float32)
        self.command.position[:], self.command.velocity[:] = self._pub_to_raw(pub_pos, pub_vel)

    # The drdds (published) joint frame and the simulator's raw frame differ per robot by a
    # sign flip and offset per DOF; the Lite3's are identity.

    def _pub_to_raw(self, position: np.ndarray, velocity: np.ndarray):
        if self.profile.joint_dir is None:
            return position, velocity
        direction = self.profile.joint_dir
        return position * direction + self.profile.pos_offset_rad, velocity * direction

    def _raw_to_pub(self, position: np.ndarray, velocity: np.ndarray, torque: np.ndarray):
        if self.profile.joint_dir is None:
            return position, velocity, torque
        direction = self.profile.joint_dir
        return (position - self.profile.pos_offset_rad) * direction, velocity * direction, torque * direction
