import matplotlib.pyplot as plt
import cv2
import numpy as np


img = cv2.imread("TEL330_fish_tracking/fish_fillet_black.png")



img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray)
plt.show()

gray = cv2.medianBlur(img_gray, 5)
plt.imshow(gray)
plt.show()
# Detect circles
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                           param1=50, param2=40, minRadius=1, maxRadius=40)

# Draw results
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(gray, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw center
        cv2.circle(gray, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('Detected Circles', gray)
cv2.waitKey(0)

"""img_gray = cv2.equalizeHist(img_gray)
plt.imshow(img_gray)
plt.show()"""

