import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from .detector import Detector

class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector")
        self.detector = Detector()
        self.publisher = self.create_publisher(ConeLocationPixel, "/relative_banana_px", 10)
        self.subscriber = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.callback, 1)
        self.bridge = CvBridge()
        self.get_logger().info("Detector Initialized")

    def callback(self, img_msg):
        # Process image with CV Bridge

        img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        model = Detector()
        model.set_threshold(0.5)
        results = model.predict(img)
        
        predictions = results["predictions"]
        original_image = results["original_image"]
            
        out = model.draw_box(original_image, predictions, draw_all=True)

        (x1, y1, x2, y2), label = predictions[0]
        b_box = ((x1, y1), (x2, y2))
        
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

        self.publisher.publish(pixel_msg)

        # save_path = "demo_output.png"
        # out.save(save_path)
        # print(f"Saved demo to {save_path}!")


        #TODO: 

def main(args=None):
    rclpy.init(args=args)
    detector = DetectorNode()
    rclpy.spin(detector)
    rclpy.shutdown()

if __name__=="__main__":
    main()
