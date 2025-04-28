import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    resized = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    
    # Focus on bottom half
    mask = np.zeros_like(binary)
    mask[binary.shape[0]//2:] = 255
    masked = cv2.bitwise_and(binary, mask)
    
    return masked, resized

def detect_lines(edge_image):
    edges = cv2.Canny(edge_image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=30)
    if lines is not None:
        return [line[0] for line in lines]
    return []

def compute_line_parameters(line):
    x1, y1, x2, y2 = line
    avg_x = (x1 + x2) / 2
    slope = (y2 - y1) / (x2 - x1 + 1e-6)
    return avg_x, slope

def find_parallel_pair_split_by_half(lines, image_width):
    if len(lines) < 2:
        print("Not enough lines to find a pair.")
        return None, None

    mid_x = image_width / 2
    left_lines = []
    right_lines = []

    for line in lines:
        avg_x, slope = compute_line_parameters(line)
        if avg_x < mid_x:
            left_lines.append((line, avg_x, slope))
        else:
            right_lines.append((line, avg_x, slope))

    print(f"Total lines: {len(lines)} | Left: {len(left_lines)} | Right: {len(right_lines)}")

    if not left_lines:
        print("No left-side lines detected.")
    if not right_lines:
        print("No right-side lines detected.")

    best_pair = None
    min_slope_diff = float('inf')

    for line_left, _, slope_left in left_lines:
        for line_right, _, slope_right in right_lines:
            slope_diff = abs(slope_left - slope_right)
            if slope_diff < min_slope_diff:
                min_slope_diff = slope_diff
                best_pair = (line_left, line_right)

    if best_pair:
        print("Found a valid left-right pair with slope diff:", min_slope_diff)
        return best_pair
    else:
        print("No valid pair found, fallback to extreme lines.")
        # Fallback: return the leftmost and rightmost lines
        leftmost_line = min(lines, key=lambda line: compute_line_parameters(line)[0], default=None)
        rightmost_line = max(lines, key=lambda line: compute_line_parameters(line)[0], default=None)
        return leftmost_line, rightmost_line

def draw_lines(image, lines, color=(0, 255, 0), thickness=3):
    output = image.copy()
    if lines is None:
        return output
    if isinstance(lines, (tuple, list)) and len(lines) == 4 and all(isinstance(i, (int, float, np.integer)) for i in lines):
        x1, y1, x2, y2 = lines
        cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    elif isinstance(lines, (list, tuple)):
        for line in lines:
            if line is not None and len(line) == 4:
                x1, y1, x2, y2 = line
                cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return output

# ----------- MAIN SCRIPT -----------

image_filename = '/Users/blammers/Documents/RSS/racecar_docker/home/racecar_ws/src/final_challenge2025/racetrack_images/lane_3/image45.png'
img = cv2.imread(image_filename)
if img is None:
    raise FileNotFoundError(f"Could not load {image_filename}")

binary_mask, resized_img = preprocess_image(img)
all_lines = detect_lines(binary_mask)

# Visualize all detected lines
overlay_all = draw_lines(resized_img, all_lines, color=(100, 255, 100), thickness=1)
all_rgb = cv2.cvtColor(overlay_all, cv2.COLOR_BGR2RGB)

# Find best parallel pair: one from left half, one from right half
image_width = resized_img.shape[1]
parallel_left, parallel_right = find_parallel_pair_split_by_half(all_lines, image_width)

# Draw parallel pair
overlay_parallel = resized_img.copy()
if parallel_left is not None:
    overlay_parallel = draw_lines(overlay_parallel, parallel_left, color=(255, 0, 0), thickness=3)
if parallel_right is not None:
    overlay_parallel = draw_lines(overlay_parallel, parallel_right, color=(0, 0, 255), thickness=3)

# Convert to RGB for matplotlib
original_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
parallel_rgb = cv2.cvtColor(overlay_parallel, cv2.COLOR_BGR2RGB)

# Plot results
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(all_rgb)
plt.title("All Detected Lines")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(parallel_rgb)
plt.title("Best Parallel Lines (Left/Right Halves)")
plt.axis("off")

plt.tight_layout()
plt.show()
