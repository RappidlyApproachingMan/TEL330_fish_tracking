import numpy as np
import cv2
import pyrealsense2 as rs
from rtde_receive import RTDEReceiveInterface

ROBOT_IP = "192.168.56.101"


class laserCalibration():
    def __init__(self):
        self.CAM_X = 640
        self.CAM_Y = 480

        self.rtde_r = RTDEReceiveInterface(ROBOT_IP)

        # realsense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.CAM_X, self.CAM_Y, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.CAM_X, self.CAM_Y, rs.format.bgr8, 30)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        self.color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr_params = self.color_stream.get_intrinsics()

        #camera matrix
        self.cam_matrix = np.array([
            [self.intr_params.fx, 0,             self.intr_params.ppx],
            [0,             self.intr_params.fy, self.intr_params.ppy],
            [0,             0,             1             ]
        ])
        #distortion coefficients
        self.dist_coeffs = np.array(self.intr_params.coeffs).reshape(5, 1)

        #STORAGE
        self.camera_points = []   # points in the camera coordinate frame
        self.robot_points  = []   # robot tcp pos
        self.depth_frame = None   # used  mouse callback

        self.MINIMUM_DEPTH = 0.1   
        self.MAXIMUM_DEPTH = 2.0   
        self.DEPTH_AREA = 5   # Depth varies from pixel to pixel.
        self.MIN_POINTS = 10




    def img2cam(self, x, y, depth):
        # covert 2d pixel coordinate to 3d camera coordinate)
        half_area = self.DEPTH_AREA // 2
        depth_values = []

        for i in range(-half_area, half_area + 1):
            for j in range(-half_area, half_area + 1):
                pixel_x = x + i
                pixel_y = y + j
                if 0 <= pixel_x < self.CAM_X and 0 <= pixel_y < self.CAM_Y:
                    depth_value = self.depth_frame.get_distance(pixel_x, pixel_y)
                    if self.MINIMUM_DEPTH < depth_value < self.MAXIMUM_DEPTH:
                        depth_values.append(depth_value)
        
        if not depth_values:
            print("No valid depth values at selected position.")
            return None  # No valid depth values
        
        avg_depth = np.mean(depth_values)
        point = rs.rs2_deproject_pixel_to_point(self.intr_params, [x, y], avg_depth)
        point = np.array(point)
        return point


    def svd_calib(self):
        # solve for R, t such that robot_point = R @ cam_point + t
        A = np.array(self.camera_points)
        B = np.array(self.robot_points) 

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        A_centered = A - centroid_A
        B_centered = B - centroid_B

        H = A_centered.T @ B_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = centroid_B - R @ centroid_A

        return R, t
    
    def mouse(self,event, x, y, flags, param):

        if self.depth_frame is None:
            return
        
        # laser dot in camera frame
        cam_point = self.img2cam(x, y, self.depth_frame)
        if cam_point is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # get tcp position in robot base frame
            tcp_pos = self.rtde_r.getActualTCPPose()
            robot_point = np.array(tcp_pos[:3]) # x, y, z, not rx, ry, rz

            self.camera_points.append(cam_point)
            self.robot_points.append(robot_point)

            print(f"Point {len(self.camera_points)} recorded:")
            print(f"  Camera: {cam_point}")
            print(f"  Robot : {robot_point}")
        

    def save(self, R, t, filename="calibration.npz"):
        np.savez(filename, {"R": R, "t": t})
        print(f"Calibration saved to {filename}")

    def read_calibration(self, filename="calibration.npz"):
        
        data = np.load(filename, allow_pickle=True).item()
        R = data["R"]
        t = data["t"]
        return R, t
    
    def main(self):
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse)

        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            self.depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not self.depth_frame or not color_frame:
                continue

            self.depth_frame = self.depth_frame.as_depth_frame()
            image = np.asanyarray(color_frame.get_data())

            for i, camera_p in enumerate(self.camera_points):
                # project 3D camera point back to pixel coordinates for visualization
                px = int(self.intr_params.ppx + self.intr_params.fx * camera_p[0] / camera_p[2])
                py = int(self.intr_params.ppy + self.intr_params.fy * camera_p[1] / camera_p[2])
                cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(image, f"P{i+1}", (px + 5, py - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(image, f"Points: {len(self.camera_points)}/{self.MIN_POINTS}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Calibration", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.pipeline.stop()
        cv2.destroyAllWindows()

        #calibrate 
        if len(self.camera_points) >= self.MIN_POINTS:
            print(f"Calibrating with {len(self.camera_points)} points...")
            R, t = self.svd_calib()
            self.save(R, t)
            print("Calibration complete.")
        else:
            print(f"Not enough points for calibration. {len(self.camera_points)}/{self.MIN_POINTS} collected.")


if __name__ == "__main__":
    calib = laserCalibration()
    calib.main()