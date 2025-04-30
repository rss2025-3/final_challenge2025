#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from vs_msgs.msg import ConeLocation, ParkingError
import csv
from rclpy.time import Time
import numpy as np
import math

# ros2 run visual_servoing parking_controller_analysis --ros-args -p filename:=cone.csv

class ParkingControllerAnalyzer(Node):
    def __init__(self):
        super().__init__("wall_follower_analyzer")

        # Declare filename parameter with default value
        self.declare_parameter('filename', 'cone_metrics.csv')
        filename = self.get_parameter('filename').value

        self.CONE_TOPIC = "/relative_cone"

        self.csv_file = open(f"/home/racecar/racecar_ws/src/visual_servoing/{filename}", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'x_pos', 'y_pos'])
        
        self.start_time = None

        # Create subscription to cone topic
        self.cone_sub = self.create_subscription(
            ConeLocation,
            self.CONE_TOPIC,
            self.analyze_cone,
            10
        )

    def analyze_cone(self, msg):
        # Get current ROS time instead of using message header
        current_time = self.get_clock().now()
        
        if self.start_time is None:
            self.start_time = current_time
        
        time_since_start = (current_time - self.start_time).nanoseconds / 1e9

        x = float(msg.x_pos)
        y = float(msg.y_pos)

        print(f"Recorded: {x}, {y}")

        # Log the data
        self.csv_writer.writerow([time_since_start, x, y])
        
    def destroy_node(self):
        # Close the CSV file when the node is destroyed
        self.csv_file.close()
        super().destroy_node()

def main():
    rclpy.init()
    analyzer = ParkingControllerAnalyzer()
    rclpy.spin(analyzer)
    analyzer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()