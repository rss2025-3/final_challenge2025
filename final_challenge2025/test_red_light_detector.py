import cv2
import os
import glob
from color_segmentation import detect_traffic_light, image_print  # Import your function from the file you wrote

def test_folder(folder_path):
    """
    Test red light detection on all images in the specified folder.
    Args:
        folder_path (str): Path to the folder containing images.
    """
    # Get all image file paths recursively
    image_files = glob.glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True) + \
              glob.glob(os.path.join(folder_path, "**", "*.png"), recursive=True) + \
              glob.glob(os.path.join(folder_path, "**", "*.jpeg"), recursive=True)

    if not image_files:
        print("No images found in folder:", folder_path)
        return

    for img_path in image_files:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image {img_path}")
            continue


        height, width = img.shape[:2]
        
        # Define "small" however you want — here's a simple rule
        if height < 100 or width < 100:
            # Scale up by 10x
            img = cv2.resize(img, (width * 10, height * 10), interpolation=cv2.INTER_NEAREST)

        # Detect red light
        _, bbox = detect_traffic_light(img)

        # Draw bounding box if detected
        if bbox is not None:
            (x1, y1), (x2, y2) = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show image
        image_print(img)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = "red_light_images/"  # <-- CHANGE this to your images folder
    test_folder(folder_path)
