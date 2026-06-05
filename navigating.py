import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
import math
import cv2.aruco as aruco
import cv2
import pyrealsense2 as rs
import numpy as np


class WaypointFollower(Node):

    def __init__(self):

        print("Initializing Waypoint Follower Node...")
        super().__init__('driver')

        self.waypoints = [
            # (0.736309, -1.38571),
            # (0.982407, -1.69943),
            # (1.04191, -1.94773),
            # (1.13994, -2.12903),
            # (1.2739, -2.53038),
            # (1.43386, -2.69148),
            # (1.67081, -2.70303),
            # (1.35214, -3.15403),
            # (1.74137, -3.44432),
            # (1.79611, -3.75871),
            # (2.07021, -4.00688),
            # (2.27255, -4.41041),
            # (2.77686, -4.4279),
            # (3.10408, -4.67795),
            # (3.20963, -5.20914),
            # (3.49568, -5.20405),
            # (3.67604, -5.20382),
            # (3.88456, -5.21132)

            # (0.736309, -1.38571),
            # (0.982407, -1.69943),
            # (1.04191, -1.94773),
            # (1.13994, -2.12903),
            # (1.2739, -2.53038),
            # (1.43386, -2.69148),
            # (1.18509, -3.03328),
            # (1.35214, -3.15403),
            # (1.74137, -3.44432),
            # (1.79611, -3.75871),
            # (2.07021, -4.00688),
            # (2.27255, -4.41041),
            # (2.3227, -4.65372),
            # (2.58982, -4.56805)

            # (0.235625, 1.8627),
            # (0.167202, 2.14641),
            # (0.62452, 2.38739),
            # (0.686011, 2.94524),
            # (0.732943, 3.31037),
            # (0.73514, 3.76693),
            # (0.767027, 4.2642),
            # (0.466161, 4.23851),
            # (0.48823, 4.66716)

                        
        
        ]

        self.current_waypoint = 0

        self.linear_speed = 0.05
        self.lookahead_distance = 0.4
        self.max_angular_speed = 0.35
        self.waypoint_tolerance = 0.08

        self.position = None
        self.yaw = 0.0

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_z = 0.0

        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_set = False

        self.origin_odom_x = 0.0
        self.origin_odom_y = 0.0

        self.intial_yaw = None

        self.code_dict = {}

        self.odom_sub = self.create_subscription(
            Odometry,
            '/a200_1074/platform/odom',
            self.odom_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/a200_1074/cmd_vel',
            10
        )

        self.pipe = rs.pipeline()
        cfg = rs.config()

        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.pipe.start(cfg)

        profile = self.pipe.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.fx, self.fy = intrinsics.fx, intrinsics.fy
        self.ppx, self.ppy = intrinsics.ppx, intrinsics.ppy

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.target = False
        self.search = False

        self.targetInput = int(input("Enter target marker ID: "))

        self.get_logger().info("Waypoint Follower Node Initialized.")

        self.timer = self.create_timer(0.1, self.control_loop)

        self.camera_thread = threading.Thread(target=self.process_camera, daemon=True)
        self.camera_thread.start()


    def odom_callback(self, msg):

        self.position = msg.pose.pose.position

        self.odom_x = self.position.x
        self.odom_y = self.position.y

        orientation = msg.pose.pose.orientation
        q = [orientation.x, orientation.y, orientation.z, orientation.w]

        siny_cosp = 2 * (q[3] * q[2] + q[0] * q[1])
        cosy_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])

        self.raw_yaw = math.atan2(siny_cosp, cosy_cosp)

        if self.intial_yaw is None:
            self.intial_yaw = self.raw_yaw
        
        self.yaw = math.atan2(math.sin(self.intial_yaw - self.raw_yaw), math.cos(self.intial_yaw - self.raw_yaw))
        # self.yaw = math.atan2(
        #     2 * (q[3] * q[2] + q[0] * q[1]),
        #     1 - 2 * (q[1] * q[1] + q[2] * q[2])
        # )

    def camera_rotation(self, x, y, z):

        sin_y = math.sin(self.yaw)
        cos_y = math.cos(self.yaw)

        new_x = self.odom_x + (cos_y * x - sin_y * y)
        new_y = self.odom_y + (sin_y * x + cos_y * y)
        new_z = self.odom_z + z

        return new_x, new_y, new_z


    def process_camera(self):


        while rclpy.ok():

            frames = self.pipe.poll_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            try:
                corners, ids, _ = self.detector.detectMarkers(color_image)
            except Exception as e:
                self.get_logger().error(f"Error detecting markers: {e}")
                continue

            if ids is not None:

                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

                for i, corner in enumerate(corners):

                    x, y = np.mean(corner[0], axis=0).astype(int)

                    depth_value = depth_frame.get_distance(x, y)

                    Y = (x - self.ppx) * depth_value / self.fx
                    Z = (y - self.ppy) * depth_value / self.fy
                    X = math.sqrt(depth_value**2 - Z**2)

                    marker_id = ids[i][0]

                    new_x, new_y, new_z = self.camera_rotation(X, Y, Z)

                    if marker_id == 1 and marker_id not in self.code_dict and not self.origin_set:

                        self.origin_odom_x = self.odom_x
                        self.origin_odom_y = self.odom_y

                        self.origin_set = True
                        self.origin_x = new_x - 0.35
                        self.origin_y = new_y

                        self.code_dict[marker_id] = [
                            new_x - self.origin_odom_x + self.odom_x,
                            new_y - self.origin_odom_y + self.odom_y
                        ]

                        self.yaw = math.atan2(math.sin(self.intial_yaw - self.raw_yaw), math.cos(self.intial_yaw - self.raw_yaw))



                        print("Marker:", marker_id, self.origin_x, self.origin_y)
                    
                    if marker_id == self.targetInput and self.search and marker_id not in self.code_dict and X <= 0.7 and self.origin_set:
                        
                        self.target = True
                        print("Target Marker Detected")

            cv2.imshow("Camera", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

    def control_loop(self):

        cmd = TwistStamped()

        if self.position is None:
            return

        if not self.origin_set:
            self.get_logger().info("Waiting for origin to be set...")
            return

        
        if self.current_waypoint >= len(self.waypoints):
            self.search = True
            if self.target == True:
                print("Target Reached. Stopping robot.")
                self.stop_robot()
                quit()
                return
            else:
                print("All waypoints reached, but target not detected. Searching for target...")   
                cmd.twist.linear.x = 0.0
                cmd.twist.angular.z = 0.15
                cmd.header.stamp = self.get_clock().now().to_msg()
                self.cmd_pub.publish(cmd)
                return
        
        

        target_x, target_y = self.waypoints[self.current_waypoint]

        robot_x = (self.odom_x - 0.35) + self.origin_odom_x - self.origin_x
        robot_y = (self.odom_y) + self.origin_odom_y - self.origin_y

        dx = target_x - robot_x
        dy = target_y - robot_y

        distance = math.sqrt(dx**2 + dy**2)

        if distance < self.waypoint_tolerance:

            self.get_logger().info(f"Reached waypoint {self.current_waypoint}")

            self.current_waypoint += 1
            return

        target_heading = math.atan2(dy, dx)

        alpha = target_heading - self.yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        if abs(alpha) > 1.2:

            cmd.twist.angular.z = max(-0.15, min(0.15, 1.2 * alpha))
            cmd.twist.linear.x = 0.0

            cmd.header.stamp = self.get_clock().now().to_msg()

            self.cmd_pub.publish(cmd)
            print("robot_x:", robot_x, "robot_y:", robot_y, "target_x:", target_x, "target_y:", target_y, "alpha:", alpha)

            return
        
        Ld = min(self.lookahead_distance, distance)
        Ld = max(Ld, 0.15)

        curvature = (2 * math.sin(alpha)) / Ld

        linear = self.linear_speed * min(distance / 0.4, 1.0)
        linear = max(0.01, linear)

        angular = linear * curvature

        angular = max(-self.max_angular_speed,
                      min(self.max_angular_speed, angular))

        cmd = TwistStamped()

        cmd.twist.linear.x = linear
        cmd.twist.angular.z = angular

        cmd.header.stamp = self.get_clock().now().to_msg()
        print("robot_x:", robot_x, "robot_y:", robot_y, "target_x:", target_x, "target_y:", target_y, "alpha:", alpha)

        self.cmd_pub.publish(cmd)


    def stop_robot(self):

        cmd = TwistStamped()

        cmd.twist.linear.x = 0.0
        cmd.twist.angular.z = 0.0

        cmd.header.stamp = self.get_clock().now().to_msg()

        self.cmd_pub.publish(cmd)

def main(args=None):

    print("Starting Waypoint Follower...")

    rclpy.init(args=args)

    node = WaypointFollower()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        print("Shutting down...")

    finally:

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()