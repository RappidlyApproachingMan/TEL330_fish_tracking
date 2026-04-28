import rtde_control
import rtde_receive
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

"""
Using a realsense camera, we detect the 3d position of the point on the fish.
using the results from the calibration script we can transform the point from camera coordinates to robot base coordinates.
we can then use moveL to move the robot asynchronously to the point, and use the opencv window to start and stop the movement.

"""





"""
    cheat sheet.
    Bounds:
        X,    Y,   Z
    -0.75,0.90,0.20
    -0.30,0.90,0.00
    -0.75,-0.90,0.20
    -0.30,-0.90,0.20
"""

ROBOT_IP = "192.168.56.101"
SPEED = 0.2


class Camera():
    def __init__(self):
        """ Initializes the RealSense camera and sets up the OpenCV window for display and user input. """
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)


        #openCV window for displaying the camera feed and getting user input
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    
    def get_frames(self):
        """ aligns the depth and color frames and returns them as numpy arrays for use in opencv """
        frames = self.pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image
    
    def display_frames(self, depth_image, color_image):
        """ displays the depth and color frames side by side in an opencv window """
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow("Camera", images)
        cv2.waitKey(1)

    def get_point(self):
        depth_image, color_image = self.get_frames()
        if depth_image is None or color_image is None:
            return None
        # Process the color image to find the point on the fish (e.g., using color thresholding)
        # For simplicity, let's assume we have a function that returns the pixel coordinates of the point
        pixel_coords = self.detect_fish_point(color_image)

        if pixel_coords is None:
            return None

        # Get the depth value at the detected pixel coordinates
        depth_value = depth_image[pixel_coords[1], pixel_coords[0]]

        # Convert pixel coordinates and depth value to camera coordinates
        point_3d_camera = self.pixel_to_camera_coordinates(pixel_coords, depth_value)

        return point_3d_camera

    def detect_fish_point(self, color_image):
        # Placeholder for fish point detection logic (e.g., color thresholding)
        # This should return the (x, y) pixel coordinates of the detected point on the fish
        pass

    def pixel_to_camera_coordinates(self, pixel_coords, depth_value):
        # Placeholder for converting pixel coordinates and depth value to camera coordinates
        pass

    def pixel_to_robot_coordinates(self, point_3d_camera):
        # Placeholder for converting camera coordinates to robot base coordinates using calibration results
        pass

class Controller():
    def __init__(self):
        self.rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        
        self.camera = Camera()
        self.current_point = None

        self.END_EFFECTOR_HEIGHT = 0.40

    def update_point(self):
        point_3d_camera = self.camera.get_point()
        if point_3d_camera is not None:
            self.current_point = self.camera.pixel_to_robot_coordinates(point_3d_camera)
            # note that this gives the x,y,z of the point. We want the arm to follow from above, so z should be the height of the arm.
            self.current_point[2] = self.END_EFFECTOR_HEIGHT

    def follow_point(self, point_3d_robot):
        # Move asynchronously to the detected point in robot coordinates
        while True:
            self.update_point()
            if point_3d_robot is not None:
                self.rtde_c.moveL(point_3d_robot, SPEED, 0.5, True)


    def stop(self):
        self.rtde_c.stopL()
        self.update_thread.join()
        self.follow_thread.join()
        cv2.destroyAllWindows()

    def run(self):
        while True:
            self.camera.display_frames(*self.camera.get_frames())
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC to quit
                print("Exiting...")
                self.stop()
                break
            elif key == ord("e"):  # 'e' to stop movement immediately
                print("Stopping movement...")
                self.rtde_c.stopL()
            elif key == ord("s"):  # 's' to start movement immediately
                print("Starting movement...")
                self.follow_point(self.current_point)



            

if __name__ == "__main__":
    controller = Controller()
    controller.run()
    
    