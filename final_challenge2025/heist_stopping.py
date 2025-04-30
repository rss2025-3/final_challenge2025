#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult
from vs_msgs.msg import ConeLocation, ConeLocationPixel
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from math import inf

#from safety_controller.visualization_tools import VisualizationTools

from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class HeistStoppingController(Node):

    def __init__(self):
        super().__init__("safety_controller")
        # Declare parameters
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("safety_topic", "default")
        self.declare_parameter("stopping_time", 0.0)
        self.declare_parameter('odom_topic', "default")

        # Fetch constants from the ROS parameter server
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SAFETY_TOPIC = self.get_parameter('safety_topic').get_parameter_value().string_value
        self.STOPPING_TIME = self.get_parameter('stopping_time').get_parameter_value().double_value
        self.ODOM_TOPIC = self.get_parameter('odom_topic').get_parameter_value().string_value


        self.drive_publisher_ = self.create_publisher(AckermannDriveStamped, self.SAFETY_TOPIC, 10)

        self.ackermann_sub = self.create_subscription(
            AckermannDriveStamped,
            self.DRIVE_TOPIC,
            self.acker_callback,
            10)
        
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.listener_callback,
            10)

        self.ground_truth_subscriber = self.create_subscription(Odometry, '/pf/pose/odom', self.get_car_pose, 1)
        self.stoplight_loc = (-12.4, 14.6)
        self.stoplight_dist = float(inf)
        self.stoplight_red = False
        self.stoplight = self.create_subscription(ConeLocation, '/relative_stoplight', self.at_stoplight,10)
        self.detect_stoplight = self.create_subscription(Bool, '/detect_stoplight', self.detect_stoplight,10)
        # self.bananas = self.create_subscription(PhysicalLocation, '/banana_loc', self.at_banana,10)
        # self.guards = self.create_subscription(Bool, '/detect_guards', self.listener_callback,10)
        
        # velocity that gets updated by listening to the drive command
        self.velocity = 1
        self.banana_stop_time = 30 #based on how often drive commands are published
        self.banana_cooldown = 30
        self.ignore_bananas = False

    def detect_stoplight(self, msg):
        self.stoplight_red = msg.data

    def get_car_pose(self,msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self.stoplight_dist = np.sqrt((x-self.stoplight_loc[0])**2+(y-self.stoplight_loc[1])**2)

    def acker_callback(self, msg):
        self.velocity = msg.drive.speed
        
        # if self.ignore_bananas and self.banana_cooldown:
        #     self.banana_cooldown -= 1
        # else: #reset
        #     self.ignore_bananas = False
        #     self.banana_cooldown = 30
        #     self.banana_stop_time = 30

    def detect_stoplight(self, msg):
        self.light_is_red = msg.data

    def set_stoplight_loc(self, msg):
        self.stoplight_loc = (msg.x, msg.y)
        self.get_logger().info(f"Stoplight location received: {self.stoplight_loc}")

    def set_banana_loc(self, msg):
        self.banana_loc = (msg.x, msg.y)
        self.get_logger().info(f"Banana location received: {self.banana_loc}")

    def listener_callback(self, laser_scan):
        
        scan_data = laser_scan.ranges

        # calculates the forward obstacle distance by averaging 10 LIDAR values in front
        # index 50 is the front middle and so 45:55 is the region around it
        forward_dist = min(scan_data[480:600]) # TODO: check the dimensions of actual LIDAR scan
        time_to_collision = forward_dist / (self.velocity+0.01)
        stop = time_to_collision < self.STOPPING_TIME or forward_dist < 0.3
        
        
        if stop: # TODO: if this doesn't work maybe we also check distance
            # Do something to stop
            current_time = self.get_clock().now()
            drive_stamped = AckermannDriveStamped()
            drive_stamped.header.frame_id = "drive_frame_id"
            drive_stamped.header.stamp = current_time.to_msg()
            drive_stamped.drive.speed = 0.0
            drive_stamped.drive.steering_angle = 0.0
            self.get_logger().info(f'{time_to_collision}, {forward_dist}')
            #self.get_logger().info(f'STOPPING!!! {time_to_collision=}')
            #self.get_logger().info(f'STOP DISTANCe {forward_dist=}')

            # TODO: uncomment this to send actual stop commands
            self.drive_publisher_.publish(drive_stamped)

    def at_stoplight(self, msg):
        if self.stoplight_dist < 2:
            vel = max(self.velocity, 0.01)
            stopping_dist = (vel**2) / (2*1.4*9.8)
            stop = self.stoplight_dist-0.75 < stopping_dist

            if stop and self.stoplight_red:
                self.get_logger().info(f"Stopping at red light: distance={self.stoplight_dist:.2f}, velocity={self.velocity:.2f}")
                current_time = self.get_clock().now()
                drive_stamped = AckermannDriveStamped()
                drive_stamped.header.frame_id = "drive_frame_id"
                drive_stamped.header.stamp = current_time.to_msg()
                drive_stamped.drive.speed = 0.0
                drive_stamped.drive.steering_angle = 0.0
            
                self.drive_publisher_.publish(drive_stamped)

def main():
    rclpy.init()
    heist_safety_controller = HeistStoppingController()
    rclpy.spin(heist_safety_controller)
    heist_safety_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()