#!/usr/bin/env python3
import math

import pyrealsense2 as rs
import numpy as np
import cv2
import rclpy
import threading
import time
import cv2.aruco as aruco
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry

class QRReaction(Node):
    def __init__(self):
        super().__init__('qr_rover_controller')
        self.yaw = 0
        self.origin_x = 0
        self.origin_y = 0
        self.origin_z = 0
        self.odom_subscriber = self.create_subscription(Odometry, '/a200_1074/platform/odom', self.odom_callback, 10)

        #Realsense Camera set up
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

        #Aruco code detector setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.last_data = None

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_z = 0.0

        self.origin_set = False

        self.camera_thread = threading.Thread(target=self.process_camera, daemon=True)
        self.camera_thread.start()
        self.get_logger().info("Program Running...")

        self.code_dict = {}

    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
        1 - 2 * (q.y * q.y + q.z * q.z))

    def camera_rotation(self, x, y, z):
        sin = math.sin(self.yaw)
        cos = math.cos(self.yaw)

        new_x = self.odom_x + (cos * x - sin * y)
        new_y = self.odom_y + (sin * x + cos * y)
        new_z = self.odom_z + z

        return new_x, new_y, new_z

    def process_camera(self):
        
        origin_odom_x = self.odom_x
        origin_odom_y = self.odom_y
        while rclpy.ok():

            if self.origin_set:
                with open("data.txt", "a") as f:
                    f.write(f"{(self.odom_x - origin_odom_x)-self.origin_x}, {(self.odom_y - origin_odom_y)-self.origin_y}\n")
                    print((self.odom_x - origin_odom_x)-self.origin_x,(self.odom_y - origin_odom_y)-self.origin_y)
                    time.sleep(1.0)



            frames = self.pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            #depth_image = np.asanyarray(depth_frame.get_data())

            try:
                corners, ids, rejected = self.detector.detectMarkers(color_image)
            except Exception as e:
                print(f"Error in detecting markers: {e}")
                continue

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                for i, corner in enumerate(corners):
                    x, y = np.mean(corner[0], axis=0).astype(int)
                    depth_value = depth_frame.get_distance(x, y)
                    X =  depth_value
                    Y = (x - self.ppx) * depth_value / self.fx
                    Z = (y - self.ppy) * depth_value / self.fy

                    marker_id = int(ids[i][0])

                    #new_x, new_y, new_z = self.camera_rotation(X,Y,Z)

                    # x -> forward y -> left z -> up

                    if marker_id is not None and marker_id not in self.code_dict and X<= 0.7:
                        self.code_dict[marker_id] = [self.origin_x,self.origin_y]
                        print("Marker: ",marker_id, self.odom_x, self.odom_y)

def main(args=None):
    rclpy.init(args=args)
    node = QRReaction()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()