import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        src = np.asanyarray(color_frame.get_data())

        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

        # create mask
        mask = cv2.inRange(hsv, (0, 72, 120), (11, 255, 255))

        # find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        out = src.copy()

        # draw only the largest contour
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(out, [largest], -1, (0, 255, 0), 2)

        cv2.imshow("output", out)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
