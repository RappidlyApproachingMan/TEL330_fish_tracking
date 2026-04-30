import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_receive
from scipy.spatial.transform import Rotation

# --- Checkerboard config ---
BOARD_SIZE  = (9, 6)      # inner corners (cols, rows) — adjust to your board
SQUARE_SIZE = 0.022       # meters — measure your actual printed squares

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

print("Camera matrix:")
print(camera_matrix)
print("Distortion coeffs:", dist_coeffs)

# --- Robot connection ---
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.56.101")

# --- Storage ---
R_gripper2base_list = []
t_gripper2base_list = []
R_board2cam_list    = []
t_board2cam_list    = []
tcp_poses           = []   # for logging

print("\n=== Hand-Eye Calibration ===")
print("Tips for good calibration:")
print("  - Move the board to cover all corners of the camera FOV")
print("  - Include large tilts (±45 degrees) in all directions")
print("  - Aim for 20+ varied poses")
print("  - Keep the board fully visible and flat")
print("\nPress SPACE to record pose, D to delete last pose, ESC when done.\n")

try:
    while True:
        frames      = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray        = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        flags  = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, flags)

        display = color_image.copy()

        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, BOARD_SIZE, corners, found)
            cv2.putText(display, f"Board found!  Poses: {len(R_gripper2base_list)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"Board NOT found  Poses: {len(R_gripper2base_list)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show instructions on screen
        cv2.putText(display, "SPACE=record  D=delete last  ESC=solve",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        # --- SPACE: record pose ---
        if key == ord(" "):
            if not found:
                print("Board not visible — reposition until board is detected")
                continue

            success, rvec, tvec = cv2.solvePnP(
                objp, corners, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                print("solvePnP failed — try again")
                continue

            R_board2cam, _ = cv2.Rodrigues(rvec)
            t_board2cam    = tvec.reshape(3, 1)

            tcp = rtde_r.getActualTCPPose()
            t_g2b = np.array(tcp[:3]).reshape(3, 1)
            R_g2b = Rotation.from_rotvec(tcp[3:]).as_matrix()

            R_gripper2base_list.append(R_g2b)
            t_gripper2base_list.append(t_g2b)
            R_board2cam_list.append(R_board2cam)
            t_board2cam_list.append(t_board2cam)
            tcp_poses.append(tcp)

            print(f"Pose {len(R_gripper2base_list):02d} recorded — "
                  f"TCP: X={tcp[0]:.3f} Y={tcp[1]:.3f} Z={tcp[2]:.3f} "
                  f"Rx={tcp[3]:.3f} Ry={tcp[4]:.3f} Rz={tcp[5]:.3f}")

        # --- D: delete last pose ---
        elif key == ord("d"):
            if R_gripper2base_list:
                R_gripper2base_list.pop()
                t_gripper2base_list.pop()
                R_board2cam_list.pop()
                t_board2cam_list.pop()
                tcp_poses.pop()
                print(f"Deleted last pose — {len(R_gripper2base_list)} remaining")
            else:
                print("Nothing to delete")

        # --- ESC: solve ---
        elif key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# --- Solve ---
n = len(R_gripper2base_list)
if n < 5:
    print(f"Not enough poses ({n}) — need at least 5, ideally 20+")
else:
    print(f"\nSolving with {n} poses...")

    # Try all methods and pick the best one
    methods = {
        "TSAI":    cv2.CALIB_HAND_EYE_TSAI,
        "PARK":    cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":  cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    best_method = None
    best_error  = float("inf")
    best_R      = None
    best_t      = None

    for name, method in methods.items():
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base_list,
            t_gripper2base_list,
            R_board2cam_list,
            t_board2cam_list,
            method=method
        )

        # Reprojection error check
        errors = []
        for i in range(n):
            T_g2b = np.eye(4)
            T_g2b[:3,:3] = R_gripper2base_list[i]
            T_g2b[:3, 3] = t_gripper2base_list[i].flatten()

            T_b2c = np.eye(4)
            T_b2c[:3,:3] = R_board2cam_list[i]
            T_b2c[:3, 3] = t_board2cam_list[i].flatten()

            T_c2b = np.eye(4)
            T_c2b[:3,:3] = R_cam2base
            T_c2b[:3, 3] = t_cam2base.flatten()

            T_board2base = T_c2b @ T_b2c
            err = np.linalg.norm(T_board2base[:3,3] - T_g2b[:3,3])
            errors.append(err)

        mean_err = np.mean(errors)
        max_err  = np.max(errors)
        print(f"  {name:12s} — mean={mean_err*100:.2f} cm   max={max_err*100:.2f} cm")

        if mean_err < best_error:
            best_error  = mean_err
            best_method = name
            best_R      = R_cam2base
            best_t      = t_cam2base

    print(f"\nBest method: {best_method} (mean error {best_error*100:.2f} cm)")

    T_base_camera = np.eye(4)
    T_base_camera[:3,:3] = best_R
    T_base_camera[:3, 3] = best_t.flatten()

    print("\nT_base_camera:")
    print(T_base_camera)

    # Sanity check — translation should match physical setup
    print(f"\nCamera position in base frame:")
    print(f"  X={best_t[0,0]:.3f} m  (expected ≈ -1.0)")
    print(f"  Y={best_t[1,0]:.3f} m  (expected ≈  0.5)")
    print(f"  Z={best_t[2,0]:.3f} m  (expected ≈  0.68)")

    det = np.linalg.det(best_R)
    print(f"\nRotation matrix determinant: {det:.6f}  (should be 1.0)")

    np.save("T_base_camera_2.npy", T_base_camera)
    print("\nSaved to T_base_camera.npy")