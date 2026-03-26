import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_receive
from scipy.spatial.transform import Rotation

# --- Checkerboard config (must match your printed board) ---
BOARD_SIZE  = (8, 6)      # inner corners (cols, rows)
SQUARE_SIZE = 0.025       # meters — measure your actual printed squares

objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

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
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.100")

# --- Storage ---
R_gripper2base_list = []
t_gripper2base_list = []
R_board2cam_list    = []
t_board2cam_list    = []

print("=== Hand-Eye Calibration ===")
print("Move the robot to a new pose, then press SPACE to record it.")
print("Aim for 15-25 varied poses. Press ESC when done.\n")

try:
    while True:
        # Grab color frame
        frames      = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray        = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Try to find checkerboard
        flags  = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, flags)

        # Draw corners if found
        display = color_image.copy()
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, BOARD_SIZE, corners, found)
            cv2.putText(display, f"Board found! Poses recorded: {len(R_gripper2base_list)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Board NOT found",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        # --- SPACE: record this pose ---
        if key == ord(" "):
            if not found:
                print("Board not visible — move robot until board is detected")
                continue

            # Solve board pose in camera frame
            success, rvec, tvec = cv2.solvePnP(
                objp, corners, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                print("solvePnP failed — try again")
                continue

            R_board2cam, _ = cv2.Rodrigues(rvec)
            t_board2cam    = tvec.reshape(3, 1)

            # Get TCP pose from robot
            tcp = rtde_r.getActualTCPPose()
            t_g2b = np.array(tcp[:3]).reshape(3, 1)
            R_g2b = Rotation.from_rotvec(tcp[3:]).as_matrix()

            R_gripper2base_list.append(R_g2b)
            t_gripper2base_list.append(t_g2b)
            R_board2cam_list.append(R_board2cam)
            t_board2cam_list.append(t_board2cam)

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

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base_list,
        t_gripper2base_list,
        R_board2cam_list,
        t_board2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_base_camera = np.eye(4)
    T_base_camera[:3, :3] = R_cam2base
    T_base_camera[:3,  3] = t_cam2base.flatten()

    print("\nT_base_camera:")
    print(T_base_camera)

    np.save("T_base_camera.npy", T_base_camera)
    print("\nSaved to T_base_camera.npy")