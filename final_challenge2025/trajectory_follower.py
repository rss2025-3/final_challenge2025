import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
import numpy as np
import math
from .utils import LineTrajectory
import tf_transformations as tf
from std_msgs.msg import Bool

class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.drive_topic = "/vesc/high_level/input/nav_2"
        self.speed = 1.0  # FILL IN #
        #self.lookahead = 2.0 * self.speed  # FILL IN #
        self.lookahead = 2.0
        self.wheelbase_length = 0.33  # FILL IN #
    
        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
                                                 
        
        self.pose_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)

        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        
        self.point_pub = self.create_publisher(PointStamped,
                                               '/lookahead',
                                               1)
        
        self.initialized_traj = False
        
        self.banana_close = False
        self.banana_sub = self.create_subscription(Bool, "/banana_close", self.banana_update, 10)
        self.stoplight_loc = (-12.4, 14.6)
        self.num_traj = 0

    def banana_update(self, msg):
        self.banana_close = msg.data
        
    def pose_callback(self, odometry_msg):
        if self.initialized_traj is True and self.banana_close is False:
            #self.get_logger().info('PURE PURSUITTTTTT!!!')
            map_x = odometry_msg.pose.pose.position.x
            map_y = odometry_msg.pose.pose.position.y
            theta = tf.euler_from_quaternion((odometry_msg.pose.pose.orientation.x, odometry_msg.pose.pose.orientation.y, odometry_msg.pose.pose.orientation.z, odometry_msg.pose.pose.orientation.w))[2]

            closest_pt, min_dist, min_idx = self.closest_point_vectorized((map_x, map_y))
            lookahead_point = self.find_lookahead_point([map_x, map_y], min_idx)
            #self.get_logger().info(f'{min_idx=}')

            if lookahead_point is not None:
                lookahead_point = lookahead_point[1]
                point_msg = PointStamped()

                # Populate the header (time + frame)
                point_msg.header.stamp = self.get_clock().now().to_msg()
                point_msg.header.frame_id = "map"  # or "odom" or "base_link", etc.

                # Set the point coordinates
                point_msg.point.x = lookahead_point[0]
                point_msg.point.y = lookahead_point[1]
                point_msg.point.z = 0.0

                # Publish the point
                self.point_pub.publish(point_msg)

                eta = math.atan2((lookahead_point[1] - map_y),  (lookahead_point[0] - map_x)) - theta
                # self.get_logger().info(f"lookahead1:{(lookahead_point[1] - map_y)}, 2: {(lookahead_point[0] - map_x)}")
                delta = math.atan2(2 * self.wheelbase_length * math.sin(eta),  self.lookahead)
                #self.get_logger().info(f'{eta=}')
                current_time = self.get_clock().now()
                drive_cmd = AckermannDriveStamped()
                drive_cmd.header.frame_id = "base_link"
                drive_cmd.header.stamp = current_time.to_msg()

                last_pt = np.array(self.trajectory.points[-1])
                car_pos = np.array([map_x, map_y])
                dist_to_goal = np.linalg.norm(last_pt - car_pos)
                
                # backwards hack
                dist_to_stoplight = math.sqrt((map_x-self.stoplight_loc[0])**2 + (map_y-self.stoplight_loc[1])**2)
                if self.num_traj == 3:
                    drive_cmd.drive.speed = 1.5 * self.speed
                elif dist_to_stoplight > 3:
                    drive_cmd.drive.speed = 1 * self.speed
                else:
                    drive_cmd.drive.speed = 0.85 * self.speed

                if dist_to_goal <= 1:
                    drive_cmd.drive.speed *= dist_to_goal
                    if dist_to_goal <= 0.1:
                        drive_cmd.drive.speed = 0.0
                        self.initialized_traj =  False

                drive_cmd.drive.steering_angle = delta
                self.drive_pub.publish(drive_cmd)
            else:
                last_pt = np.array(self.trajectory.points[-1])
                car_pos = np.array([map_x, map_y])
                current_time = self.get_clock().now()
                dist_to_goal = np.linalg.norm(last_pt - car_pos)
                # self.get_logger().info(f"dist_to ")

                drive_cmd = AckermannDriveStamped()
                drive_cmd.header.frame_id = "base_link"
                drive_cmd.header.stamp = current_time.to_msg()

                drive_cmd.drive.speed = 1 * self.speed

                if dist_to_goal <= 1:
                    drive_cmd.drive.speed *= dist_to_goal
                    if dist_to_goal <= 0.1:
                        drive_cmd.drive.speed = 0.0

                drive_cmd.drive.steering_angle = 0.0
                self.drive_pub.publish(drive_cmd)

        else:
            current_time = self.get_clock().now()
            drive_cmd = AckermannDriveStamped()
            drive_cmd.header.frame_id = "base_link"
            drive_cmd.header.stamp = current_time.to_msg()
            drive_cmd.drive.speed = 0.0

            drive_cmd.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_cmd)

    def closest_point_vectorized(self, car_position):
        if len(self.trajectory.points) < 2:
            return None, float('inf'), -1

        pts = np.array(self.trajectory.points)  # shape (N, 2)
        V = pts[:-1] # segment starts
        W = pts[1:]  # segment ends

        p = np.array(car_position)

        VW = W - V
        seg_lens_sq = np.einsum('ij,ij->i', VW, VW)  # shape (N-1)
        pV = p - V 

        dot_vals = np.einsum('ij,ij->i', pV, VW)  # shape (N-1)


        seg_lens_sq_safe = np.maximum(seg_lens_sq, 1e-12)

        # Param t along each segment
        t_vals = dot_vals / seg_lens_sq_safe
        t_vals = np.clip(t_vals, 0.0, 1.0)  # shape (N-1)

        # projection_i = V[i] + t_vals[i] * (W[i] - V[i])
        projections = V + (t_vals[:, np.newaxis] * VW)  # shape ((N-1), 2)

        # Distance from the car to each projection
        diff = projections - p  # shape ((N-1), 2)
        dist_sq = np.einsum('ij,ij->i', diff, diff)  # shape (N-1)
        distances = np.sqrt(dist_sq)                 # shape (N-1)

        # Find the index of the closest projection
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        closest_pt = projections[min_idx]

        return (tuple(closest_pt), float(min_dist), int(min_idx))
    
    def find_lookahead_point(self, car_pos, start_segment_idx):        
        for seg_idx in range(start_segment_idx, len(self.trajectory.points)-1):
            p1 = np.array(self.trajectory.points[seg_idx])
            p2 = np.array(self.trajectory.points[seg_idx + 1])
            intersection = self.circle_segment_intersection(
                p1, p2,
                np.array(car_pos),
                self.lookahead
            )
            if intersection is not None:
                return intersection

        # If we exhaust all segments with no intersection found, return None or last point, up to you
        return None

    def circle_segment_intersection(self, p1, p2, center, radius, t_min=0.0):
        v = p2 - p1

        a = np.dot(v, v)
        b = 2.0 * np.dot(v, p1-center)
        c = np.dot(p1, p1) + np.dot(center, center) - 2.0 * np.dot(p1, center) - radius ** 2


        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        
        # Usually two solutions t1 < t2
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        # We'll consider the smaller positive t first, because that's "first" intersection
        # But we only accept t if it's in [t_min, 1.0].
        # for t_candidate in sorted([t1, t2]):
        #     if t_min <= t_candidate <= 1.0:
        #         # valid intersection
        #         ix = p1[0] + t_candidate * v[0]
        #         iy = p1[1] + t_candidate * v[1]
        #         return (ix, iy)
        
        return ((p1[0] + t1 * v[0], p1[1] + t1 * v[1]), (p1[0] + t2 * v[0], p1[1] + t2 * v[1]))

        # If neither t1 nor t2 is valid, no intersection on this segment
        return None


    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")
        # self.get_logger().info(f"Receiving new trajectory {msg.poses} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True
        self.num_traj += 1


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
