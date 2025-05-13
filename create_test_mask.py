import cv2
import numpy as np
from PIL import Image

# Load the test image
image = cv2.imread('data/segmentation/train/images/test_image.jpg')
height, width = image.shape[:2]

# Create a complex mask with multiple tumor regions
mask = np.zeros((height, width), dtype=np.uint8)

# Create main tumor region
center_x, center_y = width // 2, height // 2
radius = min(width, height) // 4
cv2.circle(mask, (center_x, center_y), radius, 255, -1)

# Add smaller tumor regions
small_radius = radius // 3
# Top right tumor
cv2.circle(mask, (center_x + radius, center_y - radius), small_radius, 255, -1)
# Bottom left tumor
cv2.circle(mask, (center_x - radius, center_y + radius), small_radius, 255, -1)

# Add some irregular shapes
pts = np.array([[center_x - radius//2, center_y - radius//2],
                [center_x + radius//2, center_y - radius//2],
                [center_x, center_y + radius//2]], np.int32)
cv2.fillPoly(mask, [pts], 255)

# Save the mask
cv2.imwrite('data/segmentation/train/masks/test_image.jpg', mask)

# Also create validation data by copying the same files
image = Image.open('data/segmentation/train/images/test_image.jpg')
mask = Image.open('data/segmentation/train/masks/test_image.jpg')

image.save('data/segmentation/val/images/test_image.jpg')
mask.save('data/segmentation/val/masks/test_image.jpg') 