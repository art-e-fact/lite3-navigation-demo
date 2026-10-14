import numpy as np
import rclpy
from drdds.msg import ImuData, ImuDataValue, JointData, JointsData, JointsDataCmd, JointsDataValue, MetaType
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster

from constants import BASE_FRAME, ODOM_FRAME
from sensors.common.transforms import make_transform, quaternion_msg, sim_time_stamp
from simulation import ROBOT_PROFILES, JointCommand, RobotProfile, default_joint_command_arrays, quat_xyzw_to_rpy

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


class NewtonRosBridge:
    def __init__(self, headless: bool, model_path: str, profile: RobotProfile | None = None):
        self.node = Node("newton_simulation")
        self.node.declare_parameter("headless", headless)
        self.node.declare_parameter("model_path", model_path)
        self._shutdown_requested = False

        self.profile = profile if profile is not None else ROBOT_PROFILES["Lite3"]
        self.num_dofs = self.profile.num_dofs
        self.kp_cmd, self.kd_cmd, self.pos_cmd, self.vel_cmd, self.tau_ff = default_joint_command_arrays(self.profile)

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
        return JointCommand(
            kp=self.kp_cmd.copy(),
            kd=self.kd_cmd.copy(),
            position=self.pos_cmd.copy(),
            velocity=self.vel_cmd.copy(),
            torque=self.tau_ff.copy(),
        )

    def publish_clock(self, timestamp: float):
        """Publish simulated time. Called every step"""
        msg = Clock()
        msg.clock = sim_time_stamp(timestamp)
        self.clock_pub.publish(msg)

    def publish_state(self, timestamp: float, state, last_tau: np.ndarray):
        stamp = sim_time_stamp(timestamp)
        rpy_deg = np.degrees(quat_xyzw_to_rpy(state.quat_xyzw))

        imu_msg = ImuData()
        imu_msg.header = MetaType()
        imu_msg.header.frame_id = 0
        imu_msg.header.stamp = stamp
        imu_msg.data = ImuDataValue()
        imu_msg.data.roll = float(rpy_deg[0])
        imu_msg.data.pitch = float(rpy_deg[1])
        imu_msg.data.yaw = float(rpy_deg[2])
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
        imu_std_msg.angular_velocity.x = float(state.imu_gyro[0])
        imu_std_msg.angular_velocity.y = float(state.imu_gyro[1])
        imu_std_msg.angular_velocity.z = float(state.imu_gyro[2])
        imu_std_msg.linear_acceleration.x = float(state.imu_acc[0])
        imu_std_msg.linear_acceleration.y = float(state.imu_acc[1])
        imu_std_msg.linear_acceleration.z = float(state.imu_acc[2])
        self.imu_std_pub.publish(imu_std_msg)

        joints_msg = JointsData()
        joints_msg.header = MetaType()
        joints_msg.header.frame_id = 0
        joints_msg.header.stamp = stamp
        joints_msg.data = JointsDataValue()
        joints_msg.data.joints_data = [JointData() for _ in range(JOINT_MSG_SLOTS)]
        pub_pos, pub_vel, pub_tau = self._raw_to_pub(state.joint_position, state.joint_velocity, last_tau)
        for index in range(self.num_dofs):
            joint = joints_msg.data.joints_data[index]
            joint.name = list(BLANK_JOINT_NAME)
            joint.data_id = 0
            joint.status_word = JOINT_STATUS_WORD
            joint.position = float(pub_pos[index])
            joint.velocity = float(pub_vel[index])
            joint.torque = float(pub_tau[index])
            joint.motion_temp = SIM_MOTOR_TEMP_C
            joint.driver_temp = SIM_DRIVER_TEMP_C
        for index in range(self.num_dofs, JOINT_MSG_SLOTS):
            joints_msg.data.joints_data[index].status_word = JOINT_STATUS_WORD
        self.joints_pub.publish(joints_msg)

    def publish_odom_and_tf(self, timestamp: float, state):
        stamp = sim_time_stamp(timestamp)
        linvel_body = state.linvel_body

        transform = make_transform(stamp, ODOM_FRAME, BASE_FRAME, state.position, state.quat_xyzw)
        self.tf_broadcaster.sendTransform(transform)

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = ODOM_FRAME
        odom_msg.child_frame_id = BASE_FRAME
        odom_msg.pose.pose.position.x = float(state.position[0])
        odom_msg.pose.pose.position.y = float(state.position[1])
        odom_msg.pose.pose.position.z = float(state.position[2])
        odom_msg.pose.pose.orientation = transform.transform.rotation
        odom_msg.twist.twist.linear.x = float(linvel_body[0])
        odom_msg.twist.twist.linear.y = float(linvel_body[1])
        odom_msg.twist.twist.linear.z = float(linvel_body[2])
        odom_msg.twist.twist.angular.x = float(state.angvel_body[0])
        odom_msg.twist.twist.angular.y = float(state.angvel_body[1])
        odom_msg.twist.twist.angular.z = float(state.angvel_body[2])
        self.odom_pub.publish(odom_msg)

    def _cmd_callback(self, msg: JointsDataCmd):
        if len(msg.data.joints_data) not in (self.num_dofs, JOINT_MSG_SLOTS):
            self.node.get_logger().warn("Received JointsDataCmd with incorrect number of joints")
            return

        pub_pos = np.zeros(self.num_dofs, dtype=np.float32)
        pub_vel = np.zeros(self.num_dofs, dtype=np.float32)
        for index in range(self.num_dofs):
            joint_cmd = msg.data.joints_data[index]
            self.kp_cmd[index] = joint_cmd.kp
            self.kd_cmd[index] = joint_cmd.kd
            pub_pos[index] = joint_cmd.position
            pub_vel[index] = joint_cmd.velocity
            self.tau_ff[index] = joint_cmd.torque  # tau_ff no processing, matches MuJoCo bridge

        if self.profile.joint_dir is not None:
            self.pos_cmd[:] = pub_pos * self.profile.joint_dir + self.profile.pos_offset_rad
            self.vel_cmd[:] = pub_vel * self.profile.joint_dir
        else:
            self.pos_cmd[:] = pub_pos
            self.vel_cmd[:] = pub_vel

    def _raw_to_pub(self, position: np.ndarray, velocity: np.ndarray, torque: np.ndarray):
        if self.profile.joint_dir is None:
            return position, velocity, torque
        direction = self.profile.joint_dir
        return (position - self.profile.pos_offset_rad) * direction, velocity * direction, torque * direction
