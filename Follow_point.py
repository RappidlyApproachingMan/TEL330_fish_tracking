import time

import rtde_control
import rtde_receive
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import threading as thrd




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

#read t_cam_to_base from  T_base_camera.npy
#T_CAM_TO_BASE = np.load("TEL330_fish_tracking/T_base_camera.npy")
#T_CAM_TO_BASE = np.load("TEL330_fish_tracking/T_camera_base.npy")
#T_CAM_TO_BASE = np.load("TEL330_fish_tracking/cam2gripper.npy")


def read_calibration(filename="calibration.npz"):
        data = np.load(filename, allow_pickle=True)
        print(data.files)
        R = data["R"]
        t = data["t"]
        return R, t
#T_CAM_TO_BASE = read_calibration("TEL330_fish_tracking/cam2gripper.npz")
T_CAM_TO_BASE = read_calibration("TEL330_fish_tracking/calibration.npz")

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

        self.profile = self.pipeline.get_active_profile()
        self.color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = self.color_stream.get_intrinsics()
        self.fx = self.intrinsics.fx   # Focal length x
        self.fy = self.intrinsics.fy   # Focal length y
        self.cx = self.intrinsics.ppx  # Principal point x
        self.cy = self.intrinsics.ppy  # Principal point y

        # Get depth scale (converts raw depth units → meters)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.align = rs.align(rs.stream.color)
    

        self.T_cam_to_base = T_CAM_TO_BASE # Set the transformation matrix from camera to robot base frame

        #openCV window for displaying the camera feed and getting user input
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    
    def get_frames(self):
        """ aligns the depth and color frames and returns them as numpy arrays for use in opencv """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        
        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return depth_image, color_image
    
    def display_frames(self, color_image, masked_image):
        """ displays the depth and color frames side by side in an opencv window """
        if masked_image is not None:
            images = np.hstack((color_image, masked_image))
        cv2.imshow("Camera", images)
        cv2.waitKey(1)

    def get_point(self):
        depth_image, color_image = self.get_frames()
        if depth_image is None or color_image is None:
            return None
        # Process the color image to find the point on the fish (e.g., using color thresholding)
        if self.get_fish_point(color_image) is not None:
            pixel_coords, out = self.get_fish_point(color_image)
            print(f"Pixel coordinates of detected point: {pixel_coords}")
        else:
            #home position
            pixel_coords = (80, 200)  #found experimentally
            fake_depth = 1030 # measured
            point_3d_camera = self.pixel_to_camera_coordinates(pixel_coords, fake_depth)
            print(f"Using home position pixel coordinates: {pixel_coords}, 3D camera coordinates: {point_3d_camera}")
            return point_3d_camera

        if pixel_coords is None:
            return None

        # Get the depth value at the detected pixel coordinates
        #it needs to be the mean of a small window around the pixel coordinates to be more robust to noise

        depth_value = np.mean(depth_image[pixel_coords[1]-2:pixel_coords[1]+3, pixel_coords[0]-2:pixel_coords[0]+3])
        print(f"Depth value at detected point: {depth_value} (raw units), {depth_value * self.depth_scale:.3f} meters")


        # Convert pixel coordinates and depth value to camera coordinates
        point_3d_camera = self.pixel_to_camera_coordinates(pixel_coords, depth_value)

        return point_3d_camera # returns (X, Y, Z) in camera coordinates, or None if detection fails


    def get_fish_point(self, masked_image):
        # Placeholder for fish point detection logic (e.g., color thresholding)
        # This should return the (x, y) pixel coordinates of the detected point on the fish


        src = masked_image.copy() # already an array, no need to convert

        if len(src.shape) == 2:  # If the image is grayscale, convert to BGR
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

        # create mask
        mask = cv2.inRange(hsv, (0, 72, 120), (11, 255, 255))

        # find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        MIN_AREA = 200  # adjust this value

        # filter out small contours
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]

        out = src.copy()

        # draw only the largest contour
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(out, [largest], -1, (0, 255, 0), 2)
            # compute centroid of the largest contour
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                return (cX, cY), out
            else:
                return None

        
    


        


    def pixel_to_camera_coordinates(self, pixel_coords, depth_value):
        """
        Convert pixel coordinates and depth value to 3D camera coordinates
            (X, Y, Z) coordinates in the camera frame (in meters), or None if invalid
        """
        if depth_value == 0:
            return None  # No depth data at this pixel

        # RealSense depth is in millimeters — convert to meters
        depth = depth_value * self.depth_scale  # depth_scale is usually 0.001

        u, v = pixel_coords
        #use SDK helper function to deproject pixel to point in camera space
        point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth)
        X, Y, Z = point_3d   

        #print(point_3d)

        # # Unproject using pinhole camera model
        # X = (u - self.cx) * Z / self.fx
        # Y = (v - self.cy) * Z / self.fy

        return (X, Y, Z)
    def pixel_to_robot_coordinates(self, point_3d_camera):
        """Transform a 3D point from camera frame to robot base frame."""
        #NOTE: named incorrectly, this takes a coordinate in the camera 3d cordinate space
        if point_3d_camera is None:
            return None

        X_c, Y_c, Z_c = point_3d_camera

        # Homogeneous point in camera space 
        # P_cam = np.array([X_c, Y_c, Z_c]) 

        # # get current pose 
        # tcp = rtde_receive.getActualTCPPose() 
        # t_g2b = np.array(tcp[:3]).reshape(3, 1) 
        # R_g2b = R.from_rotvec(tcp[3:]).as_matrix() 

        # R_cam2gripper = self.T_cam_to_base[:3, :3] 
        # t_cam2gripper = self.T_cam_to_base[:3, 3].reshape(3, 1)

        # # Compute the transformation from camera to robot base frame 
        # R_cam2base = R_g2b @ R_cam2gripper 
        # t_cam2base = R_g2b @ t_cam2gripper + t_g2b

        # Homogeneous point in camera space
        P_cam = np.array([X_c, Y_c, Z_c])

        R_cam2base , t_cam2base = self.T_cam_to_base


        #manual R and t
        R =[
            [0.0527717,-0.81727689,0.5738237 ],
            [-0.99467694, 0.0079081,   0.1027387 ],
            [-0.08850382, -0.5761909,  -0.81250915]]
        t = [-1.07155554,  0.4456387,   1.06818035]

        # this is the same fucking matrix that i am reading from the file but that one won't work. i bet it's because of how i save it.
        R2 = [
            [ 0.0505512,  -0.8501962,   0.52403339],
            [-0.99856541, -0.03375126,  0.04156889],
              [-0.01765493, -0.52538297, -0.85068269]]
        t2 = [-0.99032283,  0.55015257,  1.19973231]
        #make np arrays
        R = np.array(R)
        t = np.array(t)
        R2 = np.array(R2)
        t2 = np.array(t2)

        # Apply transform
        P_robot = R_cam2base @ P_cam + t_cam2base
        #P_robot = R @ P_cam + t
        #P_robot = R2 @ P_cam + t2

        return list(P_robot[:3])  # [X, Y, Z] in robot base frame



class Controller():
    def __init__(self):
        self.rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

        self.camera = Camera()
        self.current_point = None

        self.END_EFFECTOR_HEIGHT = 0.40

        self._running = False
        self._lock = thrd.Lock()

        self.KP = 0.8                 # proportional gain
        self.MAX_SPEED = 0.25        # max TCP speed, m/s NOTE: this acts as a multiplier.  If robot is set to 20% speed, then this is 0.2 * 0.2 = 0.04 m/s max speed.  If robot is set to 100% speed, then this is 0.2 m/s max speed.
        self.ACCELERATION = 0.5
        self.SPEEDL_DT = 0.05         # command duration, seconds

    def update_point(self):
        point_3d_camera = self.camera.get_point()

        if point_3d_camera is not None:
            camera_point = self.camera.pixel_to_robot_coordinates(point_3d_camera)

            target_point = camera_point + [0.0, 3.14, 0.0]
            target_point[2] = self.END_EFFECTOR_HEIGHT

            with self._lock:
                self.current_point = target_point

            print(f"Updated point: {self.current_point}")

    def follow_point_speedL(self, target_pose):
        if target_pose is None:
            return
        
        current_pose = self.rtde_r.getActualTCPPose()
        offset = -0.15 

        dx = target_pose[0] - current_pose[0]
        dy = target_pose[1] + offset - current_pose[1]
        dz = target_pose[2] - current_pose[2]

        vx = self.KP * dx
        vy = self.KP * dy
        vz = self.KP * dz

        speed_norm = (vx**2 + vy**2 + vz**2) ** 0.5
        print(speed_norm)

        if speed_norm > self.MAX_SPEED:
            scale = self.MAX_SPEED / speed_norm
            vx *= scale
            vy *= scale
            vz *= scale

        # Keep orientation fixed; only translate
        speed_vector = [vx, vy, vz, 0.0, 0.0, 0.0]

        self.rtde_c.speedL(
            speed_vector,
            self.ACCELERATION,
            self.SPEEDL_DT
        )

    def stop(self):
        self._running = False
        self.rtde_c.speedStop()
        cv2.destroyAllWindows()

    def run(self):

        def detection_loop():
            while True:
                self.update_point()
                time.sleep(0.05)

        def movement_loop():
            while self._running:
                with self._lock:
                    point = self.current_point

                if point is not None:
                    self.follow_point_speedL(point)

                time.sleep(self.SPEEDL_DT)

            self.rtde_c.speedStop()

        update_point_thread = thrd.Thread(target=detection_loop)
        update_point_thread.daemon = True
        update_point_thread.start()

        start_position = [
            -0.400,
            0.60,
            self.END_EFFECTOR_HEIGHT,
            0.0,
            3.14,
            0.0
        ]

        # It is okay to use moveL once for startup positioning
        self.rtde_c.moveL(start_position, 0.05 , 0.5, True)

        masked = None

        while True:
            depth, color = self.camera.get_frames()

            fish_result = self.camera.get_fish_point(color)
            if fish_result is not None:
                _, masked = fish_result

                

            self.camera.display_frames(color, masked)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                print("Starting movement...")
                if not self._running:
                    self._running = True
                    movement_thread = thrd.Thread(
                        target=movement_loop,
                        daemon=True
                    )
                    movement_thread.start()

            elif key == ord("q") or key == 27:
                print("Exiting...")
                self.stop()
                break

            elif key == ord("e"):
                print("Stopping movement...")
                self._running = False
                self.rtde_c.speedStop()     

if __name__ == "__main__":
    controller = Controller()
    controller.run()
    
    