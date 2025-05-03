#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocationPixel
from std_msgs.msg import Bool

# import your color segmentation algorithm; call this function in ros_image_callback!
from color_segmentation import detect_traffic_light


class StoplightDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self):
        super().__init__("stoplight_detector")
        # toggle line follower vs cone parker
        self.LineFollower = False

        # Subscribe to ZED camera RGB frames
        self.stoplight_pub = self.create_publisher(ConeLocationPixel, "/relative_stoplight_px", 10)
        self.debug_pub = self.create_publisher(Image, "/cone_debug_img", 10)
        self.stop_pub = self.create_publisher(Bool, "/detect_stoplight", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        self.get_logger().info("Stoplight Detector Initialized")

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
	
        detected_color, b_box = detect_traffic_light(image, min_area=100)
        
        self.get_logger().info(f'Stoplight Detected Color: {detected_color}')
        
        if detected_color == 'red':
            bool_msg = Bool()
            bool_msg.data = True
            self.stop_pub.publish(bool_msg)
        else:
            bool_msg = Bool()
            bool_msg.data = False
            self.stop_pub.publish(bool_msg)

        x = float(abs(b_box[1][0]+b_box[0][0])/2)
        #y = float(abs(b_box[1][1]+b_box[0][1])/2)
        y = float(b_box[1][1])

        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)
        pixel_msg = ConeLocationPixel()
        pixel_msg.u = x
        pixel_msg.v = y
        #self.get_logger().info(f"{b_box=}")
        #self.get_logger().info(f"pixel y {y}")

        self.stoplight_pub.publish(pixel_msg)

def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
