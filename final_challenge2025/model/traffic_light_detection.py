import cv2
import numpy as np
import os
import glob

def hsv_mask(hsv, lower, upper):
    return cv2.inRange(hsv, lower, upper)

def find_blobs(mask, min_area, color_name):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            blobs.append({'bbox': (x, y, x + w, y + h), 'center': (cx, cy), 'color': color_name})
    return blobs

def is_similar_size(b1, b2, tolerance=0.8):
    x1, y1, x2, y2 = b1['bbox']
    w1, h1 = x2 - x1, y2 - y1
    x1, y1, x2, y2 = b2['bbox']
    w2, h2 = x2 - x1, y2 - y1
    return (abs(w1 - w2) / max(w1, w2) < tolerance) and (abs(h1 - h2) / max(h1, h2) < tolerance)

def is_evenly_spaced(r, y, g, tol=0.7):
    d1 = y['center'][1] - r['center'][1]
    d2 = g['center'][1] - y['center'][1]
    return abs(d1 - d2) / max(d1, d2) < tol

def iou(box1, box2):
    x1_min, y1_min = box1[0]
    x1_max, y1_max = box1[1]
    x2_min, y2_min = box2[0]
    x2_max, y2_max = box2[1]

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def remove_redundant_boxes(detections, iou_threshold=0.5):
    kept = []
    for i, d in enumerate(detections):
        redundant = False
        for k in kept:
            if iou(d['bbox'], k['bbox']) > iou_threshold:
                redundant = True
                break
        if not redundant:
            kept.append(d)
    return kept

def detect_traffic_light_stack(img, min_area_ratio=0.00001):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = img.shape[:2]
    min_area = min_area_ratio * width * height

    red_mask = hsv_mask(hsv, np.array([0, 50, 5]), np.array([8, 255, 255]))
    yellow_mask = hsv_mask(hsv, np.array([8, 150, 60]), np.array([20, 255, 255]))
    green_mask = hsv_mask(hsv, np.array([60, 50, 5]), np.array([85, 255, 255]))

    red_blobs = find_blobs(red_mask, min_area, 'red')
    yellow_blobs = find_blobs(yellow_mask, min_area, 'yellow')
    green_blobs = find_blobs(green_mask, min_area, 'green')

    traffic_lights = []

    for r in red_blobs:
        for y in yellow_blobs:
            if abs(r['center'][0] - y['center'][0]) < 20 and r['center'][1] < y['center'][1]:
                if not is_similar_size(r, y): continue
                for g in green_blobs:
                    if abs(r['center'][0] - g['center'][0]) < 20 and y['center'][1] < g['center'][1]:
                        if not (is_similar_size(y, g) and is_similar_size(r, g)):
                            continue
                        if not is_evenly_spaced(r, y, g):
                            continue

                        # Determine brightness from V channel
                        v_red = np.mean(hsv[r['bbox'][1]:r['bbox'][3], r['bbox'][0]:r['bbox'][2], 2])
                        v_yellow = np.mean(hsv[y['bbox'][1]:y['bbox'][3], y['bbox'][0]:y['bbox'][2], 2])
                        v_green = np.mean(hsv[g['bbox'][1]:g['bbox'][3], g['bbox'][0]:g['bbox'][2], 2])
                        brightness = {'red': v_red, 'yellow': v_yellow, 'green': v_green}
                        color_on = max(brightness, key=brightness.get)

                        xs = [r['bbox'][0], y['bbox'][0], g['bbox'][0], r['bbox'][2], y['bbox'][2], g['bbox'][2]]
                        ys = [r['bbox'][1], y['bbox'][1], g['bbox'][1], r['bbox'][3], y['bbox'][3], g['bbox'][3]]

                        traffic_lights.append({
                            'bbox': ((min(xs), min(ys)), (max(xs), max(ys))),
                            'color': color_on
                        })

    return remove_redundant_boxes(traffic_lights)

def process_training_folder(folder='training'):
    img_extensions = ('*.jpg', '*.jpeg', '*.png')
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(folder, ext)))

    if not img_paths:
        print(f"No images found in '{folder}' folder.")
        return

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to read {path}")
            continue

        detections = detect_traffic_light_stack(img)

        for det in detections:
            start, end = det['bbox']
            color = det['color']
            cv2.rectangle(img, start, end, (0, 255, 0), 2)
            cv2.putText(img, color.upper(), (start[0], start[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"{len(detections)} traffic light(s) detected in {os.path.basename(path)}")

        cv2.imshow("Traffic Light Detection", img)
        key = cv2.waitKey(2000)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_training_folder("training")

