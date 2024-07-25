import cv2
import numpy as np

# Load the image
image = cv2.imread('Calibration data\Test 3\intrinsic_3.png')

cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# key = cv2.waitKey(1)
cv2.imshow("image", image)

# Display the image
# cv2.imshow('image', image)

# Wait indefinitely until a key is pressed
cv2.waitKey(0)

# Destroy all OpenCV windows
cv2.destroyAllWindows()
