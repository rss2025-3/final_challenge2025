import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from .detector import Detector
from vs_msgs.msg import ConeLocation, ConeLocationPixel
import numpy as np
from std_msgs.msg import Bool

# import your color segmentation algorithm; call this function in ros_image_callback!
from .color_segmentation import detect_traffic_light
from .traffic_light_detection import detect_traffic_light_stack

class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector")
        self.detector = Detector()
        self.detector.set_threshold(0.2)
        self.stoplight_pub = self.create_publisher(ConeLocationPixel, "/stoplight_px", 10)
        self.debug_pub1 = self.create_publisher(Image, "/light_debug_img", 10)
        self.stop_pub = self.create_publisher(Bool, "/detect_stoplight", 10)
        self.publisher = self.create_publisher(ConeLocationPixel, "/banana_px", 10)
        self.subscriber = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.callback, 1)
        self.debug_pub = self.create_publisher(Image, "/banana_debug_img", 10)
        
        self.create_subscription(Bool, "/banana_close", self.banana_detected_callback, 10)
        self.banana_close = False

        self.bridge = CvBridge()
        self.get_logger().info("Detector Initialized")
        self.counter = 0
        self.banana_save_count = 0
    def banana_detected_callback(self, msg):
        self.banana_close = msg.data
        if msg.data is False:
            self.banana_save_count += 1
    def callback(self, img_msg):

        self.counter += 1
        if self.counter % 1 != 0:
            return

        # Process image with CV Bridge

        img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        predictions = detect_traffic_light_stack(img)
        colors = [d['color'] for d in predictions] 
        if 'red' in colors:
            bool_msg = Bool()
            bool_msg.data = True
            self.stop_pub.publish(bool_msg)
        else:
            bool_msg = Bool()
            bool_msg.data = False
            self.stop_pub.publish(bool_msg)
        predictions = [((start[0], start[1], end[0], end[1]), color) for (start, end), color in
             [((d['bbox'][0], d['bbox'][1]), d['color']) for d in predictions]] 

        model = Detector()    
        out = model.draw_box(np.array(img), predictions, draw_all=True)
        
        if self.banana_close is True:
            model.set_threshold(0.01)
            results = model.predict(img)
        
            predictions = results["predictions"]
            original_image = results["original_image"]
            out = model.draw_box(np.array(out), predictions, draw_all=True)
            #self.get_logger().info(f'{predictions=}')
            banana = None
            for p in predictions:
                if p[1] in ['banana', 'boat', 'kite', 'bird']:
                    banana = p

            if banana is not None:
                (x1, y1, x2, y2), label = banana
                b_box = ((x1, y1), (x2, y2))
                banana = ((x1, y1, x2, y2), "banana")  # Override label
                out = model.draw_box(original_image.copy(), [banana])
                self.get_logger().info('banana found')
                x = float(abs(b_box[1][0]+b_box[0][0])/2)
                #y = float(abs(b_box[1][1]+b_box[0][1])/2)
                y = float(b_box[1][1])

                #debug_msg = self.bridge.cv2_to_imgmsg(np.array(out), "bgr8")
                #self.debug_pub.publish(debug_msg)
            
                pixel_msg = ConeLocationPixel()
                pixel_msg.u = x
                pixel_msg.v = y
                #self.get_logger().info(f"{b_box=}")
                #self.get_logger().info(f"pixel y {y}")

                self.publisher.publish(pixel_msg)
                # Save image if it's the 1st or 2nd detection
                if self.banana_save_count < 2:
                    filename = f"banana{self.banana_save_count + 1}.jpg"
                    cv2.imwrite(filename, np.array(out))
                    self.get_logger().info(f"Saved banana detection image as {filename}")
            # save_path = "demo_output.png"
            # out.save(save_path)
            # print(f"Saved demo to {save_path}!")

        debug_msg = self.bridge.cv2_to_imgmsg(np.array(out), "bgr8")
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    detector = DetectorNode()
    rclpy.spin(detector)
    rclpy.shutdown()

if __name__=="__main__":
    main()
