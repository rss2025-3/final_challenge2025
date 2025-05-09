import cv2
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(200)
	cv2.destroyAllWindows()
def detect_traffic_light(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set min_area to 5% of image area
    img_area = img.shape[0] * img.shape[1]
    min_area = 0.002 * img_area
    # Tuned HSV ranges
    lower_red1 = np.array([0, 45, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 45, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([50, 40, 100])
    upper_green = np.array([105, 255, 255])
    lower_yellow = np.array([15, 45, 100])
    upper_yellow = np.array([35, 255, 255])

    # Masks
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Morph
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    masks = {'red': red_mask, 'yellow': yellow_mask, 'green': green_mask}
    expected_y = {'red': 0.2, 'yellow': 0.5, 'green': 0.8}  # within the cropped image

    v_channel = hsv[:, :, 2]
    height = img.shape[0]

    best_score = 0
    detected_color = None
    detected_bbox = None

    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            center_y = y + h / 2
            relative_y = center_y / height
            position_weight = 1 - abs(relative_y - expected_y[color])  # higher is better

            # Mask just this blob
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [c], -1, 255, thickness=cv2.FILLED)

            brightness = np.sum(v_channel[contour_mask > 0])
            score = brightness * position_weight

            if score > best_score:
                best_score = score
                detected_color = color
                detected_bbox = ((x, y), (x + w, y + h))

    return (detected_color, detected_bbox) if detected_color else (None, None)
