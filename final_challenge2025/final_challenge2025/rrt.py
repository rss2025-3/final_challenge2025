import numpy as np
import heapq
import matplotlib.pyplot as plt

class RRT:
    class Node:
        def __init__(self,pos):
            self.pos = (float(pos[0]), float(pos[1]))
            self.parent = []
    
    def __init__(
        self,
        start,
        goal,
        obstacles,
        x_bound,
        y_bound,
        max = 5000,
        resolution = 60.0, #0.2,
        goal_dist = 20.0,#0.2,
        map_resolution = None,
        origin_x = 0,
        origin_y = 0,
        map_to_pixel = None
    ):
        self.start = self.Node(start)
        self.goal = self.Node(goal)
        self.obstacles = obstacles
        self.x_bound = (0, obstacles.shape[1])  
        self.y_bound = (0, obstacles.shape[0])
        self.x_width = x_bound[1] - x_bound[0]
        self.y_height = y_bound[1] - y_bound[0]
        self.max = max
        self.resolution = resolution
        self.goal_dist = goal_dist
        self.node_list = []
        self.map_resolution = map_resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.map_to_pixel = map_to_pixel

    def plan(self, do_prune=False):
        #print("planning")
        self.node_list = [self.start]
        for i in range(self.max):
            random = self.get_random()
            nearest = self.get_nearest(self.node_list, random)
            new = self.steer(nearest, random)

            if not self.collision(nearest, new):
                #print(f"no collision, new:{new.pos}")
                self.node_list.append(new)
            # else:
                # print("collided")

            if self.is_goal_reached(new):
                #print("goal reached")
                path = self.create_path(len(self.node_list)-1)
                if do_prune:
                    # Line-of-sight pruning
                    grid = ~self.obstacles
                    path = self.smooth_path(path, grid) if do_prune else path
                    #path = self.smooth_path(path, self.downscale_binary(grid.T, 2)) if do_prune else path


                self.plot_path(self.obstacles, path, self.start, self.goal, filename='rrt_test.png')
                return path

        return None

    def steer(self, nearest, random):
        theta = np.arctan2(random.pos[1] - nearest.pos[1], random.pos[0] - nearest.pos[0])
        new_pos = (nearest.pos[0] + self.resolution * np.cos(theta), nearest.pos[1] + self.resolution * np.sin(theta))
        new_node = self.Node(new_pos)
        new_node.parent = nearest

        return new_node

    def get_random(self):
        if np.random.rand() < 0.1:
            return self.goal
        else:
            x_rand = np.random.uniform(self.x_bound[0], self.x_bound[1])
            y_rand = np.random.uniform(self.y_bound[0], self.y_bound[1])
            return self.Node((x_rand, y_rand))
        # x_rand = np.random.uniform(self.x_bound[0], self.x_bound[1])
        # y_rand = np.random.uniform(self.y_bound[0], self.y_bound[1])
        # return self.Node((x_rand, y_rand))
        
        # rand = np.random.randint(0, self.x_width * self.y_height)
        # x_rand = float(rand % self.x_width)
        # y_rand = float(rand // self.y_height)
        # rand_node = self.Node((x_rand,y_rand))

        # return rand_node

    def get_nearest(self, node_list, random):
        dist = [np.linalg.norm(np.array(node.pos) - np.array(random.pos)) for node in node_list]
        nearest_index = np.argmin(dist)

        return self.node_list[nearest_index]

    def collision(self, nearest, new):
        x0, y0 = nearest.pos
        x1, y1 = new.pos
        dist = np.linalg.norm([x1 - x0, y1 - y0])
        steps = max(1, int(dist / self.resolution))  # 👈 fix here
        # steps = int(dist / self.resolution)

        for i in range(steps + 1):
            t = i / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)

            px = int(np.round(x))
            py = int(np.round(y))
            #print(f"px:{px}, py:{py}, obstacle:{self.obstacles[py,px]}")

            if px < 0 or py < 0 or py >= self.obstacles.shape[0] or px >= self.obstacles.shape[1]:
                #print(f"out of bounds, px:{px}, py:{py}, obstacles shape[1]:{self.obstacles.shape[1]}")
                return True 

            if self.obstacles[py,px] == 0:
                return True

        return False


    def is_goal_reached(self, current_node):
        dx = current_node.pos[0] - self.goal.pos[0]
        dy = current_node.pos[1] - self.goal.pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        #print(f"Distance to goal: {dist}")
        return dist < self.goal_dist

    def create_path(self, goal_index):
        path = [self.goal.pos]
        node = self.node_list[goal_index]
        
        while node != self.start:
            path.append(node.pos)
            node = node.parent
        
        path.append(node.pos)
        return path

    def downscale_binary(self, arr, block_size=10):
        new_shape = (arr.shape[0] // block_size, arr.shape[1] // block_size)
        downscaled = np.zeros(new_shape, dtype=int)
        for i in range(0, arr.shape[0], block_size):
            for j in range(0, arr.shape[1], block_size):
                block = arr[i:i+block_size, j:j+block_size]
                downscaled[i//block_size, j//block_size] = np.mean(block) > 0
        return downscaled

    def has_line_of_sight(self, grid, p1, p2):
        """Bresenham's algorithm to check line of sight on grid."""
        x0, y0 = p1
        x1, y1 = p2
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if grid[int(round(x)), int(round(y))] != 0:
                    return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if grid[int(round(x)), int(round(y))] != 0:
                    return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return grid[int(round(x)), int(round(y))] == 0

    def smooth_path(self, path, grid):
        if not path or len(path) < 2:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.has_line_of_sight(grid, path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    def plot_path(self, grid, path, start, goal, filename='rrt_path_plot.png'):
        fig, ax = plt.subplots()
        flipped_grid = np.flipud(grid)
        ax.imshow(~grid, cmap='Greys', origin='upper')
        if path:
            px, py = zip(*path)
            ax.plot(px, py, color='red')
        ax.plot(start.pos[0], start.pos[1], 'go')  # Start
        ax.plot(goal.pos[0], goal.pos[1], 'bo')    # Goal
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
