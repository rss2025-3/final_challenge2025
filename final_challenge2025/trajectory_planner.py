import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from .a_star_final import a_star_final, plot_path
#from .rrt import RRT
from std_msgs.msg import Bool
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import tf_transformations as tf
import numpy as np


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")
        self.declare_parameter("drive_topic", "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter("drive_topic").value # set in launch file; different for simulator vs racecar
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.goals_clicked = 0
        self.curr_pos = None
        self.state = None
        self.goal_index = 0
        self.goals = []
        self.park_start_time = None
        self.banana_detected = False
        self.parked = False
        self.backup_start_time = None

        self.banana_close = self.create_publisher(Bool, "/banana_close", 10)
        self.is_parked = self.create_subscription(Bool, "/is_parked",self.check_parked,10)
        self.localize = self.create_subscription(Odometry, "/pf/pose/odom",self.localize_cb,10)
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        self.current_position = (0,0)
        self.viz_timer = self.create_timer(1/5, self.state_machine_cb)

    def localize_cb(self, msg):
        self.curr_pos = msg

        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ) 

    def check_parked(self,msg):
        self.parked = msg.data

    def map_cb(self, map_msg):
        occupancy_data = np.array(map_msg.data)
        binary_occupancy_grid = np.where((occupancy_data >= 0) & (occupancy_data <= 50), 0, 1)
        binary_occupancy_grid = binary_occupancy_grid.reshape((map_msg.info.height, map_msg.info.width))
        self.binary_occupancy_grid = binary_occupancy_grid

        try:
            self.dilated_occupancy_grid = np.load("dilated_occupancy_grid.npy")

            self.get_logger().info("Precomputed binary occupancy grid loaded.")
        except:
            # Invert and dilate the binary occupancy grid
            self.dilated_occupancy_grid = binary_dilation(self.binary_occupancy_grid, iterations=5)
            self.dilated_occupancy_grid = ~self.dilated_occupancy_grid 
            #np.save(self.dilated_occupancy_grid)
            
            np.save("dilated_occupancy_grid.npy", self.dilated_occupancy_grid)

            self.get_logger().info("Binary occupancy grid created and dilated.")
        
        # Save the grid as an image
        self.save_grid_image()

        self.resolution = map_msg.info.resolution
        self.origin_x = map_msg.info.origin.position.x
        self.origin_y = map_msg.info.origin.position.y

        # orientation of the map's origin in terms of rotation around the Z-axis
        self.origin_theta = tf.euler_from_quaternion((
            map_msg.info.origin.orientation.x,
            map_msg.info.origin.orientation.y,
            map_msg.info.origin.orientation.z,
            map_msg.info.origin.orientation.w))[2]

        self.inv_rotation_matrix = np.array([
            [np.cos(-self.origin_theta), -np.sin(-self.origin_theta)],
            [np.sin(-self.origin_theta), np.cos(-self.origin_theta)]
        ])

        self.rotation_matrix = np.array([
            [np.cos(self.origin_theta), -np.sin(self.origin_theta)],
            [np.sin(self.origin_theta), np.cos(self.origin_theta)]
        ])

        self.get_logger().info("Defined transformations")

    def pixel_to_map(self, pixel_x, pixel_y):
        # Convert pixel coordinates to map coordinates
        pixel_coords = np.array([[pixel_x], [pixel_y]])
        map_coords = np.dot(self.inv_rotation_matrix, pixel_coords) * self.resolution
        map_x = map_coords[0] + self.origin_x
        map_y = map_coords[1] + self.origin_y
        return map_x, map_y

    def map_to_pixel(self, map_x, map_y):
        # Convert map coordinates to pixel coordinates
        map_coords = np.array([[map_x - self.origin_x], [map_y - self.origin_y]])
        pixel_coords = np.dot(self.rotation_matrix, map_coords) / self.resolution
        pixel_x = pixel_coords[0]
        pixel_y = pixel_coords[1]
        return pixel_x, pixel_y

    def pose_cb(self, pose):
        self.get_logger().info("in pose_cb")
        clicked_x = pose.pose.pose.position.x
        clicked_y = pose.pose.pose.position.y

        self.current_position = (clicked_x, clicked_y)

    def goal_cb(self, msg):
        self.get_logger().info("in goal_cb")
        goal = (msg.pose.position.x, msg.pose.position.y)
        # self.plan_path(self.current_position, goal, self.dilated_occupancy_grid, a_star=True)
        self.get_logger().info(f"Received goal: {goal}")
        self.goals.append(goal)
        self.goals_clicked += 1

        if self.goals_clicked == 2:
            self.get_logger().info(f"cur pose: {self.curr_pos}")
            tempx = self.curr_pos.pose.pose.position.x
            tempy = self.curr_pos.pose.pose.position.y
            return_loc = (tempx, tempy)

            self.return_to_start_goal = return_loc  
            self.goals.append(return_loc)

            # tempx = self.curr_pos.pose.pose.position.x
            # tempy = self.curr_pos.pose.pose.position.y
            # return_loc = (tempx, tempy)
            # self.goals.append(return_loc)
            self.state = "START"

    def plan_path(self, start_point, end_point, map, a_star=True):
        start_time = self.get_clock().now()
        if a_star:
            self.trajectory.clear()
            start_px = self.map_to_pixel(*start_point)  # (x, y) in meters → pixels
            goal_px = self.map_to_pixel(*end_point)
            self.get_logger().info(f"Start: {start_point}, End: {end_point}")
            path = a_star_final(map, start_px, goal_px, block_size=2)
            if path != None:
                path = [(float(x), float(y)) for x, y in path]
                self.get_logger().info(f"Path found! (from A*): {path}")
                for point in path:
                    point = self.pixel_to_map(*point)
                    
                    self.trajectory.addPoint((float(point[0]),float(point[1])))
            else:
                self.get_logger().info("No path found (from A*)")

        end_time = self.get_clock().now()
        runtime = (end_time - start_time).nanoseconds / 1e9  # Convert to seconds
        self.get_logger().info(f"A* Runtime: {runtime} seconds")

        path2 = path

        # distance = 0
        # for i in range(1, len(path)):
        #     new_distance = (path[i][0]-path[i-1][0])**2 + (path[i][1]-path[i-1][1])**2
        #     distance += new_distance**0.5
        # self.get_logger().info(f"A* Distance: {distance} pixels")

        # start_time2 = self.get_clock().now()
        # if a_star: #rrt
        #     self.trajectory.clear()
        #     start_px = self.map_to_pixel(*start_point)  # (x, y) in meters → pixels
        #     goal_px = self.map_to_pixel(*end_point)
        #     self.get_logger().info(f"Running RRT, Start: {start_px}, End: {goal_px}, Map shape: {self.dilated_occupancy_grid.shape}")
            
        #     rrt = RRT(
        #         start=start_px,
        #         goal=goal_px,
        #         obstacles=map,#self.dilated_occupancy_grid,
        #         x_bound=(0, self.dilated_occupancy_grid.shape[1] * self.resolution),
        #         y_bound=(0, self.dilated_occupancy_grid.shape[0] * self.resolution),
        #         map_resolution=self.resolution,
        #         origin_x=self.origin_x,
        #         origin_y=self.origin_y,
        #         map_to_pixel=self.map_to_pixel
        #     )

        #     path = rrt.plan(do_prune = False)

        #     if path != None:
        #         for point in reversed(path):  
        #             point = self.pixel_to_map(*point)
        #             self.trajectory.addPoint((float(point[0]),float(point[1])))
        #     else:
        #         self.get_logger().info("No path found (from RRT)")

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

        # end_time2 = self.get_clock().now()
        # runtime2 = (end_time2 - start_time2).nanoseconds / 1e9  # Convert to seconds

        # # Log and publish runtime
        # self.get_logger().info(f"RRT Runtime: {runtime2} seconds")

        # distance = 0
        # if path != None:
        #     for i in range(1, len(path)):
        #         new_distance = (path[i][0]-path[i-1][0])**2 + (path[i][1]-path[i-1][1])**2
        #         distance += new_distance**0.5
        # self.get_logger().info(f"RRT Distance: {distance} pixels")

        # plot_path(~map.T, path, start_px, goal_px, filename='path_plot.png', path2=path2)

    def plan_path_midpoint(self, start_point, mid_point, end_point, map, a_star=True):
        start_time = self.get_clock().now()
        if a_star:
            self.trajectory.clear()
            start_px = self.map_to_pixel(*start_point)  # (x, y) in meters → pixels
            midpoint_px = self.map_to_pixel(*mid_point)
            goal_px = self.map_to_pixel(*end_point)
            self.get_logger().info(f"Start: {start_point}, End: {end_point}")
            path1 = a_star_final(map, start_px, midpoint_px, block_size=2)
            path2 = a_star_final(map, midpoint_px, goal_px, block_size=2)
            if path1 != None and path2 != None:
                path1.extend(path2[1:])
                path = [(float(x), float(y)) for x, y in path1]
                self.get_logger().info(f"Path found! (from A*): {path}")
                for point in path:
                    point = self.pixel_to_map(*point)

                    self.trajectory.addPoint((float(point[0]),float(point[1])))
            else:
                self.get_logger().info("No path found (from A*)")

        end_time = self.get_clock().now()
        runtime = (end_time - start_time).nanoseconds / 1e9  # Convert to seconds
        self.get_logger().info(f"A* Runtime: {runtime} seconds")

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    def save_grid_image(self, filename="src/path_planning/binary_occupancy_grid.png", dpi=1000):
        np.save("src/path_planning/binary_occupancy_grid.npy", self.binary_occupancy_grid)
        np.save("src/path_planning/dilated_occupancy_grid.npy", self.dilated_occupancy_grid)

        plt.imshow(self.binary_occupancy_grid, cmap='gray', origin='lower')

        masked_dilated_grid = np.ma.masked_where(self.dilated_occupancy_grid == 0, self.dilated_occupancy_grid)

        plt.imshow(masked_dilated_grid, cmap='Blues', origin='lower', alpha=1)

        plt.title("Binary Occupancy Grid with Dilation")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.colorbar(label='Occupancy')
        plt.savefig(filename, dpi=dpi)
        plt.close()
        self.get_logger().info(f"Binary occupancy grid saved as {filename} with DPI {dpi}")

    def state_machine_cb(self):
        self.get_logger().info(self.state)
        if self.goals:
            self.get_logger().info(f"goals: {self.goals}")
            if self.state == "START":
                self.get_logger().info("switch to planning")
                self.state = "PLANNING"
            elif self.state == "PLANNING":
                goal = self.goals[self.goal_index]
                self.plan_path(self.current_position, goal, self.dilated_occupancy_grid, a_star=True)
                self.get_logger().info("switch to driving")
                self.state = "DRIVING"
            elif self.state == "DRIVING":
                goal = self.goals[self.goal_index]
                dist = np.linalg.norm(np.array(goal) - np.array(self.current_position))
                self.get_logger().info(f"dist: {dist}")
                if dist < 1:
                    self.state = "DETECTING"
                    self.start_time = self.get_clock().now()
                    self.get_logger().info("switch to detecting")
            elif self.state == "DETECTING":
                self.goal_index += 1
                self.get_logger().info(f"{self.goal_index}")
                # if self.goal_index < 2:
                #     self.state = "PLANNING"
                # else:
                #     self.state = "DONE"
                #     self.get_logger().info("All goals visited. Returning to start.")
                # self.banana_close.publish(Bool(data=True))
                if self.parked:
                    if self.park_start_time is None:
                        self.park_start_time = self.get_clock().now()
                    else:
                        time_parked = (self.get_clock().now() - self.park_start_time).nanoseconds / 1e9
                        if time_parked >= 5.0:  # wait 5 seconds
                            self.get_logger().info("5 seconds parked. Starting to back up.")
                            self.park_start_time = None
                            self.backup_start_time = self.get_clock().now()
                            self.state = "BACKING_UP"

            elif self.state == "BACKING_UP":
                if self.backup_start_time is None:
                    self.backup_start_time = self.get_clock().now()

                time_backing = (self.get_clock().now() - self.backup_start_time).nanoseconds / 1e9
                backup_speed = -1.0  # m/s, adjust as needed
                target_distance = 1.0  # meters

                if time_backing < target_distance / abs(backup_speed):
                    drive_msg = AckermannDriveStamped()
                    drive_msg.drive.speed = backup_speed
                    drive_msg.drive.steering_angle = 0.0
                    self.drive_pub.publish(drive_msg)
                else:
                    # Stop the car after backing up
                    stop_msg = AckermannDriveStamped()
                    stop_msg.drive.speed = 0.0
                    stop_msg.drive.steering_angle = 0.0
                    self.drive_pub.publish(stop_msg)

                    self.backup_start_time = None
                    if self.goal_index < 2:
                        self.state = "PLANNING"
                        self.get_logger().info("Finished backing up. Planning next goal.")
                    else:
                        self.state = "DONE"
                        self.get_logger().info("All goals visited. Returning to start.")
            elif self.state == "DONE":
                self.state == "FINISHED"
                goal = self.goals[self.goal_index]
                self.get_logger().info(f"start goal: {goal}")
                midpoint = (-54.54682159423828, 26.789297103881836)
                self.plan_path_midpoint(self.current_position, midpoint, goal, self.dilated_occupancy_grid, a_star=True)
                self.goals_clicked = 0
                self.get_logger().info("going back to start")
            elif self.state == "FINISHED":
                return
            else:
                pass

        else:
            return

            




def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
