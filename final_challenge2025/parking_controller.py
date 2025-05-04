#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
import math

class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic")
        DRIVE_TOPIC = self.get_parameter("drive_topic").value # set in launch file; different for simulator vs racecar

        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.parked_pub = self.create_publisher(Bool, "/is_parked", 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.create_subscription(ConeLocation, "/relative_stoplight", self.relative_stoplight_callback, 10)
        self.create_subscription(ConeLocation, "/relative_banana", self.relative_banana_callback, 10)
        self.create_subscription(Bool, "/stoplight_detected", self.stoplight_detected_callback, 10)
        self.create_subscription(Bool, "/banana_close", self.banana_detected_callback, 10)


        self.parking_distance = 0.05 # meters; try playing with this number!
        self.acceptable_angle = 10 * math.pi / 180
        self.relative_x = 0
        self.relative_y = 0

        self.L = 0.1
        self.L_ah = 0.33
        self.speed = 1.0
        self.backwards = False

        self.get_logger().info("Parking Controller Initialized")

    def stoplight_detected_callback(self, msg):
        self.stoplight_detected = msg.data

    def banana_detected_callback(self, msg):
        self.banana_detected = msg.data

    # def relative_stoplight_callback(self, msg):
    #     if self.stoplight_detected:
    #         self.execute_parking(msg.x, msg.y + 0.06)  # adjust y if needed
    #         self.get_logger().info("Parking toward STOPLIGHT")

    def relative_banana_callback(self, msg):
        if self.banana_detected:
            self.execute_parking(msg.x, msg.y + 0.06)
            self.get_logger().info("Parking toward BANANA")

    def execute_parking(self, x_rel, y_rel):
        self.relative_x = x_rel
        self.relative_y = y_rel
        self.get_logger().info(f"relative x {self.relative_x}")
        self.get_logger().info(f"relative y {self.relative_y}")
        eta = math.atan(self.relative_y / self.relative_x)
        delta = math.atan(2 * self.L * math.sin(eta) / self.L_ah)

        current_time = self.get_clock().now()
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.frame_id = "base_link"
        drive_cmd.header.stamp = current_time.to_msg()
        parked_msg = Bool()
        
        error_dist = abs(self.relative_x)-self.parking_distance
        if self.backwards is False:
            if error_dist <= 0.2 and error_dist >=0 and abs(eta) < self.acceptable_angle:#) <= self.parking_distance:# and abs(eta) < self.acceptable_angle:
                # we're parked
                drive_cmd.drive.speed = 0.0
                drive_cmd.drive.steering_angle = 0.0
                parked_msg.data = True 

            elif self.relative_x < self.parking_distance:
                # we're too close
                drive_cmd.drive.speed = -1 * self.speed
                drive_cmd.drive.steering_angle = 0.0
                self.backwards = True
                parked_msg.data = False
                
            else:
                drive_cmd.drive.speed = 1 * self.speed
                drive_cmd.drive.steering_angle = delta
                parked_msg.data = False
        else:
            if self.relative_x > self.parking_distance:
                drive_cmd.drive.speed = 1 * self.speed
                drive_cmd.drive.steering_angle = delta
                self.backwards = False
                parked_msg.data = False
            else:
                drive_cmd.drive.speed = -1 * self.speed
                drive_cmd.drive.steering_angle = 0.0
                parked_msg.data = False


        self.parked_pub.publish(parked_msg)
        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()
        error_msg.x_error = float(self.relative_x)
        error_msg.y_error = float(self.relative_y)
        error_msg.distance_error = float(math.sqrt(self.relative_x ** 2 + self.relative_y ** 2))

        #################################

        # YOUR CODE HERE
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)

        #################################
        
        self.error_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
