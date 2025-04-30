#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 17:21:56 2025

@author: blammers
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt

class Node:
    def __init__(self, position, g=0, h=0):
        self.position = position
        self.g = g
        self.h = h
    def f(self):
        return self.g + self.h
    def __lt__(self, other):  # For heapq
        return self.f() < other.f()

def euclidean(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

def get_neighbors(pos, grid, diagonal=True):
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    if diagonal:
        directions += [(-1,-1), (-1,1), (1,-1), (1,1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = pos[0]+dx, pos[1]+dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if grid[nx, ny] == 0:
                neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def astar(grid, start, goal, diagonal=True):
    open_list = []
    heapq.heappush(open_list, (0, Node(start, g=0, h=euclidean(start, goal))))
    
    came_from = {}
    cost_so_far = {start: 0}
    
    expanded = 0
    max_q = 1

    while open_list:
        if len(open_list) > max_q:
            max_q = len(open_list)

        _, current_node = heapq.heappop(open_list)
        current_pos = current_node.position

        if current_pos == goal:
            return reconstruct_path(came_from, current_pos), expanded, max_q
        
        expanded += 1

        for neighbor in get_neighbors(current_pos, grid, diagonal):
            new_cost = cost_so_far[current_pos] + euclidean(current_pos, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + euclidean(neighbor, goal)
                heapq.heappush(open_list, (priority, Node(neighbor, g=new_cost, h=euclidean(neighbor, goal))))
                came_from[neighbor] = current_pos

    return None, expanded, max_q

def downscale_binary(arr, block_size=10):
    new_shape = (arr.shape[0] // block_size, arr.shape[1] // block_size)
    downscaled = np.zeros(new_shape, dtype=int)
    for i in range(0, arr.shape[0], block_size):
        for j in range(0, arr.shape[1], block_size):
            block = arr[i:i+block_size, j:j+block_size]
            downscaled[i//block_size, j//block_size] = np.mean(block) > 0
    return downscaled

def downscale_coord(coord, block_size):
    return (int(coord[0]) // block_size, int(coord[1]) // block_size)

def upscale_path(path, block_size):
    scaled_path = [(i * block_size + block_size // 2, j * block_size + block_size // 2) for i, j in path]
    
    return scaled_path

def plot_path(grid, path, start, goal, filename='path_plot.png', path2=None):
    fig, ax = plt.subplots()

    # Desired hex color
    hex_color = '#737373'
    rgba_color = plt.matplotlib.colors.to_rgba(hex_color)

    # Create RGBA image: shape (rows, cols, 4)
    image = np.zeros(grid.shape + (4,))
    image[grid == 1] = rgba_color  # Apply color where grid == 1

    ax.imshow(image, cmap='Greys', origin='upper')
    if path:
        px, py = zip(*path)
        ax.plot(py, px, color='#4285f4', linewidth=2)
    if path2:
        px, py = zip(*path2)
        ax.plot(py, px, color='#e69138', linewidth=2)
    ax.plot(start[1], start[0], marker='o', color='#cc0000')  # Start
    ax.plot(goal[1], goal[0], marker='o', color='#6aa84f')    # Goal
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def has_line_of_sight(grid, p1, p2):
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
            if grid[x, y] != 0:
                return False
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            if grid[x, y] != 0:
                return False
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return grid[x, y] == 0

def smooth_path(path, grid):
    if not path or len(path) < 2:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if has_line_of_sight(grid, path[i], path[j]):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed

from scipy.interpolate import splprep, splev

def spline_smooth_path(path, num_points=100):
    if len(path) < 3:
        return path
    x, y = zip(*path)
    tck, u = splprep([x, y], s=2)
    unew = np.linspace(0, 1, num_points)
    out = splev(unew, tck)
    return list(zip(out[0], out[1]))

def moving_average_path(path, window_size=5):
    """
    Smooth path using a simple moving average filter.

    Parameters:
        path : list of (x, y) tuples
        window_size : number of neighboring points to average over (must be odd)

    Returns:
        list of smoothed (x, y) tuples
    """
    if len(path) < window_size:
        return path

    padded = [path[0]] * (window_size//2) + path + [path[-1]] * (window_size//2)
    smoothed = []

    for i in range(len(path)):
        window = padded[i:i+window_size]
        avg_x = np.mean([p[0] for p in window])
        avg_y = np.mean([p[1] for p in window])
        smoothed.append((avg_x, avg_y))

    return smoothed


def a_star_final(grid, start, goal, block_size=10, do_prune=True, do_spline=False):
    """

    Parameters
    ----------
    grid : np.array
        2D Occupancy grid with 1s as obstacles and 0s as free space
    start : tuple
        (x,y)
    goal : tuple
        (x,y)
    block_size : int, optional
        Factor to downscale occupancy grid to improve A* runtime. The default is 10.

    Returns
    -------
    list of (x,y) tuples representing the path from start to goal.

    """
    grid = ~grid
    start_ds = downscale_coord(start, block_size)
    goal_ds = downscale_coord(goal, block_size)
    grid_array = downscale_binary(grid.T, block_size)
    path, expanded, max_q = astar(grid_array, start_ds, goal_ds)

    if path is None:
        return None

    if do_prune:
        # Line-of-sight pruning
        path = smooth_path(path, grid_array) if do_prune else path

    # Optional spline smoothing
    if do_spline:
        pruned = spline_smooth_path(path, num_points=len(pruned) * 10)

    # Upscale coordinates
    upscaled = upscale_path(path, block_size)

    plot_path(grid, [(j, i) for (i, j) in upscaled], (start[1], start[0]), (goal[1], goal[0]), filename='path_plot.png')

    return upscaled
    
