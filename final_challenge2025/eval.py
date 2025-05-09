import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PointStamped, PoseArray
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

class PoseEvaluator(Node):
    def __init__(self):
        super().__init__('pose_evaluator')
        
        # Set up TF2 listener for ground truth
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscribe to particle filter's pose estimate
        self.pf_sub = self.create_subscription(
            Odometry,
            '/pf/pose/odom',
            self.pf_callback,
            10
        )
        
        # Subscribe to clicked points
        self.click_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10
        )
        
        # Subscribe to planned trajectory
        self.traj_sub = self.create_subscription(
            PoseArray,
            '/trajectory/current',
            self.trajectory_callback,
            10
        )
        
        # Subscribe to map
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            1
        )
        
        # Initialize data storage
        self.start_time = time.time()
        
        # Add timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create visualization directory for this run
        self.viz_dir = f'src/final_challenge2025/logs/viz_{self.run_timestamp}'
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Add flag to track if we have received a trajectory
        self.has_trajectory = False
        
        # Add timer for periodic visualization (5 seconds)
        self.create_timer(5.0, self.periodic_visualization)
        self.get_logger().info(f'Pose evaluator started.')
        
        # Store latest PF estimate
        self.latest_pf_x = None
        self.latest_pf_y = None
        
        # Store clicked points and trajectory
        self.clicked_points = []
        self.current_trajectory = []
        
        # Store map data
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = None
        self.map_height = None
        
        # Add storage for particle filter history
        self.pf_position_history = []
        # Flag to know when to start recording positions
        self.recording_positions = False
        
        # Add storage for crosstrack error history with timestamps
        self.crosstrack_error_history = []  # Will store tuples of (timestamp, error)
        self.trajectory_start_time = None
    
    def map_callback(self, msg):
        # Store map data and metadata
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.get_logger().info("Received new map data")
    
    def world_to_map(self, x, y):
        # Convert world coordinates to map coordinates
        mx = int((x - self.map_origin[0]) / self.map_resolution)
        my = int((y - self.map_origin[1]) / self.map_resolution)
        return mx, my
    
    def pf_callback(self, msg):
        self.latest_pf_x = msg.pose.pose.position.x
        self.latest_pf_y = msg.pose.pose.position.y
        
        # Store position if we're recording
        if self.recording_positions:
            current_time = time.time()
            self.pf_position_history.append((self.latest_pf_x, self.latest_pf_y))
            
            # Compute and store crosstrack error with timestamp
            error = self.compute_crosstrack_error((self.latest_pf_x, self.latest_pf_y))
            if error is not None and self.trajectory_start_time is not None:
                elapsed_time = current_time - self.trajectory_start_time
                self.crosstrack_error_history.append((elapsed_time, error))
            
        self.get_logger().info(f"Latest pose estimate: x={self.latest_pf_x:.2f}, y={self.latest_pf_y:.2f}")
    
    def clicked_point_callback(self, msg):
        point = (msg.point.x, msg.point.y)
        self.clicked_points.append(point)
        self.get_logger().info(f"New clicked point: x={point[0]:.2f}, y={point[1]:.2f}")
        
    def trajectory_callback(self, msg):
        self.current_trajectory = [(pose.position.x, pose.position.y) for pose in msg.poses]
        self.get_logger().info(f"Received new trajectory with {len(self.current_trajectory)} points")
        
        # Reset tracking when new trajectory received
        self.has_trajectory = True
        self.recording_positions = True
        self.pf_position_history = []
        self.crosstrack_error_history = []
        self.trajectory_start_time = time.time()
        
        # Plot initial trajectory visualization
        self.plot_trajectory()
    
    def plot_trajectory(self, save_path=None):
        if not self.current_trajectory or self.map_data is None:
            return
            
        plt.figure(figsize=(12, 12))
        
        # Normalize map data to have mid-dark gray background
        # Convert map data to float and normalize
        map_viz = np.array(self.map_data, dtype=float)
        # Set free space (0) to 0.3 for mid-dark gray, keep obstacles (100) as black (0.0)
        map_viz[map_viz == 0] = 0.3  # 0.3 gives a mid-dark gray
        map_viz[map_viz == 100] = 0.0  # 0.0 is black
        map_viz[map_viz == -1] = 0.2  # Slightly darker gray for unknown space
        
        # Plot the map
        plt.imshow(map_viz, cmap='gray', origin='lower')
        
        # Transform trajectory points to map coordinates
        transformed_trajectory = []
        for point in self.current_trajectory:
            pixel_x = -(point[0] - self.map_origin[0]) / self.map_resolution
            pixel_y = -(point[1] - self.map_origin[1]) / self.map_resolution
            transformed_trajectory.append((pixel_x, pixel_y))
        
        # Plot transformed trajectory
        traj_x = [p[0] for p in transformed_trajectory]
        traj_y = [p[1] for p in transformed_trajectory]
        
        # Debug print coordinates
        self.get_logger().info(f"Map dimensions: {self.map_width}x{self.map_height}")
        self.get_logger().info(f"Original trajectory start: ({self.current_trajectory[0][0]:.2f}, {self.current_trajectory[0][1]:.2f})")
        self.get_logger().info(f"Transformed trajectory start: ({traj_x[0]:.2f}, {traj_y[0]:.2f})")
        
        plt.plot(traj_x, traj_y, 'b-', label='Planned Trajectory', linewidth=3, zorder=10)
        
        # Plot transformed start and end points
        if transformed_trajectory:
            plt.plot(transformed_trajectory[0][0], transformed_trajectory[0][1], 
                    'go', label='Start', markersize=12, zorder=11)
            plt.plot(transformed_trajectory[-1][0], transformed_trajectory[-1][1], 
                    'ro', label='End', markersize=12, zorder=11)
        
        # Transform and plot all particle filter positions
        if self.pf_position_history:
            transformed_pf_history = []
            for pf_pos in self.pf_position_history:
                pixel_x = -(pf_pos[0] - self.map_origin[0]) / self.map_resolution
                pixel_y = -(pf_pos[1] - self.map_origin[1]) / self.map_resolution
                transformed_pf_history.append((pixel_x, pixel_y))
            
            # Plot all historical positions with smaller dots
            pf_x = [p[0] for p in transformed_pf_history]
            pf_y = [p[1] for p in transformed_pf_history]
            plt.scatter(pf_x, pf_y, c='orange', marker='.', label='Robot Path', s=10, zorder=11, alpha=0.6)
            
            # Plot current position with a larger marker
            plt.plot(pf_x[-1], pf_y[-1], 
                    'yx', label='Current Position', markersize=15, markeredgewidth=3, zorder=12)
        
        # Transform and plot clicked points
        if self.clicked_points:
            transformed_clicks = []
            for point in self.clicked_points:
                pixel_x = -(point[0] - self.map_origin[0]) / self.map_resolution
                pixel_y = -(point[1] - self.map_origin[1]) / self.map_resolution
                transformed_clicks.append((pixel_x, pixel_y))
            
            click_x = [p[0] for p in transformed_clicks]
            click_y = [p[1] for p in transformed_clicks]
            plt.scatter(click_x, click_y, c='r', marker='x', label='Clicked Points', 
                       s=100, zorder=11)
        
        plt.title('Planned Trajectory and Robot Path on Map')
        plt.xlabel('Pixels')
        plt.ylabel('Pixels')
        plt.legend()
        plt.grid(True)
        
        # Save plot to specified path or default location
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'{self.viz_dir}/trajectory_map_{timestamp}.png'
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def compute_crosstrack_error(self, robot_pos):
        """Compute minimum perpendicular distance from robot to trajectory line segments."""
        if not self.current_trajectory:
            return None
            
        robot = np.array(robot_pos)
        min_distance = float('inf')
        
        # Iterate through consecutive pairs of points defining line segments
        for i in range(len(self.current_trajectory) - 1):
            p1 = np.array(self.current_trajectory[i])
            p2 = np.array(self.current_trajectory[i + 1])
            
            # Vector from p1 to p2
            segment = p2 - p1
            # Vector from p1 to robot
            to_robot = robot - p1
            
            # Length of the line segment
            segment_length = np.linalg.norm(segment)
            
            if segment_length == 0:
                # If points are the same, just compute distance to the point
                distance = np.linalg.norm(to_robot)
            else:
                # Normalize segment vector
                segment_unit = segment / segment_length
                
                # Project robot position onto line segment
                projection_length = np.dot(to_robot, segment_unit)
                
                if projection_length < 0:
                    # Robot is before segment start
                    distance = np.linalg.norm(to_robot)
                elif projection_length > segment_length:
                    # Robot is after segment end
                    distance = np.linalg.norm(robot - p2)
                else:
                    # Robot projects onto segment
                    # Compute perpendicular distance
                    projection = p1 + projection_length * segment_unit
                    distance = np.linalg.norm(robot - projection)
            
            min_distance = min(min_distance, distance)
        
        return float(min_distance)
    
    def plot_crosstrack_error(self, save_path):
        """Create a plot of crosstrack error over time."""
        if not self.crosstrack_error_history:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Extract times and errors from history
        times = [t for t, _ in self.crosstrack_error_history]
        errors = [e for _, e in self.crosstrack_error_history]
        
        # Plot error over time
        plt.plot(times, errors, 'b-', linewidth=2)
        
        plt.title('Crosstrack Error Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Crosstrack Error (meters)')
        plt.grid(True)
        
        # Add mean error text
        mean_error = np.mean(errors)
        plt.text(0.02, 0.98, f'Mean Error: {mean_error:.3f} m', 
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def periodic_visualization(self):
        # Only create visualizations if we have received a trajectory
        if self.has_trajectory and self.map_data is not None:
            timestamp = datetime.now().strftime("%H%M%S")
            
            # Save trajectory visualization
            traj_save_path = f'{self.viz_dir}/trajectory_viz_{timestamp}.png'
            self.plot_trajectory(traj_save_path)
            
            # Save crosstrack error plot
            error_save_path = f'{self.viz_dir}/crosstrack_error_{timestamp}.png'
            self.plot_crosstrack_error(error_save_path)
            
            self.get_logger().info(f"Saved visualizations with timestamp {timestamp}")

def main(args=None):
    rclpy.init(args=args)
    evaluator = PoseEvaluator()
    rclpy.spin(evaluator)
    evaluator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()