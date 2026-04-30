import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_receive
from scipy.spatial.transform import Rotation

# --- ArUco config ---
MARKER_ID     = 50
MARKER_SIZE   = 0.05   # meters (5 cm)
ARUCO_DICT    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
ARUCO_PARAMS  = cv2.aruco.DetectorParameters()
DETECTOR      = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# 3D corners of the marker in marker frame (z=0 plane)
# Order: top-left, top-right, bottom-right, bottom-left
HALF = MARKER_SIZE / 2.0
MARKER_OBJP = np.array([
    [-HALF,  HALF, 0],
    [ HALF,  HALF, 0],
    [ HALF, -HALF, 0],
    [-HALF, -HALF, 0],
], dtype=np.float32)

# --- RealSense setup ---
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(cfg)

intr = profile.get_stream(rs.stream.color)\
              .as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([[intr.fx, 0,       intr.ppx],
                           [0,       intr.fy, intr.ppy],
                           [0,       0,       1       ]])
dist_coeffs = np.array(intr.coeffs)

# --- Robot connection ---
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.56.101")

# --- Storage ---
R_gripper2base_list = []
t_gripper2base_list = []
R_target2cam_list    = []
t_target2cam_list    = []
R_base2gripper_list = []
t_base2gripper_list = []

print("=== Hand-Eye Calibration (ArUco) ===")
print("Move the robot to a new pose, then press SPACE to record it.")
print("Aim for 15-25 varied poses. Press ESC when done.\n")

try:
    while True:
        frames      = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray        = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners_all, ids, rejected = DETECTOR.detectMarkers(gray)

        display  = color_image.copy()
        target_corners = None

        if ids is not None:
            # Check if our target marker is among detected markers
            ids_flat = ids.flatten()
            if MARKER_ID in ids_flat:
                idx            = np.where(ids_flat == MARKER_ID)[0][0]
                target_corners = corners_all[idx].reshape(4, 2)   # (4, 2) float32

                # Draw all detected markers and highlight target
                cv2.aruco.drawDetectedMarkers(display, corners_all, ids)
                cv2.putText(display,
                            f"Marker {MARKER_ID} found! Poses: {len(R_gripper2base_list)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.aruco.drawDetectedMarkers(display, corners_all, ids)
                cv2.putText(display, f"Marker {MARKER_ID} NOT found (others visible)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            cv2.putText(display, "No markers detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        # --- SPACE: record this pose ---
        if key == ord(" "):
            if target_corners is None:
                print("Target marker not visible — move robot until marker is detected")
                continue

            # Solve marker pose in camera frame via solvePnP
            success, rvec, tvec = cv2.solvePnP(
                MARKER_OBJP,
                target_corners,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success:
                print("solvePnP failed — try again")
                continue

            #calcilate the transformation from marker to camera

            """
            Steps:
            1. Fix calibration pattern on the robot gripper.
            2. Collect aruco poses, and robot poses. Robot pose is expressed as [x, y, z, w, p, r] (Euler angles)
            3. R_target2cam, t_target2cam are obtained from aruco poses. This is converted to H_target2cam.
            4. R_gripper2base, t_gripper2base are obtained from robot poses which are obtained in step 2.
            5. Call cv.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
            6. The above step returns R_cam2gripper, t_cam2gripper. Convert this to H_cam2gripper (4x4 homogenous matrix)
            7. Get the pose of the target by calculating H_target2gripper = np.matmul(H_cam2gripper, H_target2cam) 
            """

            #implementation of step 3
            R_target2cam, _ = cv2.Rodrigues(rvec)
            t_target2cam    = tvec.reshape(3, 1)
            H_target2cam = np.eye(4)
            H_target2cam[:3, :3] = R_target2cam
            H_target2cam[:3,  3] = t_target2cam.flatten()
            #implementation of step 4
            tcp = rtde_r.getActualTCPPose()
            t_g2b = np.array(tcp[:3]).reshape(3, 1)
            R_g2b = Rotation.from_rotvec(tcp[3:]).as_matrix()
            # transform g2b to b2g
            R_b2g = R_g2b.T
            t_b2g = -R_b2g @ t_g2b
            R_base2gripper_list.append(R_b2g)
            t_base2gripper_list.append(t_b2g)

            # H_gripper2base = np.eye(4) # this matrix is wrong. use the one above
            # H_gripper2base[:3, :3] = R_g2b.T # which becomes R_b2g
            # H_gripper2base[:3,  3] = t_g2b.flatten() 
            R_gripper2base_list.append(R_g2b[:3, :3])
            t_gripper2base_list.append(t_g2b[:3,  0].reshape(3, 1))
            R_target2cam_list.append(H_target2cam[:3, :3])
            t_target2cam_list.append(H_target2cam[:3,  3].reshape(3, 1))
        

            print(f"Pose {len(R_gripper2base_list)} recorded — "
                  f"TCP: X={tcp[0]:.3f} Y={tcp[1]:.3f} Z={tcp[2]:.3f}")

        # --- ESC: finish and solve ---
        elif key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# --- Solve hand-eye calibration ---
if len(R_gripper2base_list) < 5:
    print(f"Not enough poses ({len(R_gripper2base_list)}) — need at least 5, ideally 15+")
else:
    print(f"\nSolving with {len(R_gripper2base_list)} poses...")
# step 5
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base_list,
        t_gripper2base=t_gripper2base_list,
        R_target2cam=R_target2cam_list,
        t_target2cam=t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # step 6
    H_cam2gripper = np.eye(4)
    H_cam2gripper[:3, :3] = R_cam2gripper
    H_cam2gripper[:3,  3] = t_cam2gripper.flatten()
    # step 7
    # H_target2gripper = np.matmul(H_cam2gripper, H_target2cam)

    print("\H_cam2gripper:")
    print(H_cam2gripper)

    np.save("H_cam2gripper.npy", H_cam2gripper)
    print("\nSaved to H_cam2gripper.npy")