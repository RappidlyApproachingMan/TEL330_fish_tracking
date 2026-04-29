import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_receive
from scipy.spatial.transform import Rotation


# =========================
# Checkerboard config
# =========================
BOARD_SIZE = (9, 6)       # inner corners: columns, rows
SQUARE_SIZE = 0.022       # meters

objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE


# =========================
# Helper functions
# =========================
def invert_transform(R, t):
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


# =========================
# RealSense setup
# =========================
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(cfg)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array(intr.coeffs, dtype=np.float64)


# =========================
# Robot connection
# =========================
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.56.101")


# =========================
# Storage
# =========================
R_gripper2base_list = []
t_gripper2base_list = []

R_board2cam_list = []
t_board2cam_list = []


print("=== Eye-to-Hand Calibration ===")
print("Camera is fixed beside the robot.")
print("Checkerboard must be rigidly attached to the TCP/gripper.")
print("Move robot to varied poses, then press SPACE to record.")
print("Use 15-25 varied poses if possible.")
print("Press ESC to finish and solve.\n")


# =========================
# Capture loop
# =========================
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, flags)

        display = color_image.copy()

        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )

            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            cv2.drawChessboardCorners(display, BOARD_SIZE, corners, found)

            cv2.putText(
                display,
                f"Board found | Poses: {len(R_gripper2base_list)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                display,
                f"Board NOT found | Poses: {len(R_gripper2base_list)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow("Eye-to-Hand Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        # =========================
        # SPACE: record pose
        # =========================
        if key == ord(" "):
            if not found:
                print("Board not visible — pose not recorded.")
                continue

            success, rvec, tvec = cv2.solvePnP(
                objp,
                corners,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                print("solvePnP failed — pose not recorded.")
                continue

            R_board2cam, _ = cv2.Rodrigues(rvec)
            t_board2cam = tvec.reshape(3, 1)

            tcp = rtde_r.getActualTCPPose()

            t_gripper2base = np.array(tcp[:3], dtype=np.float64).reshape(3, 1)
            R_gripper2base = Rotation.from_rotvec(tcp[3:]).as_matrix()

            R_gripper2base_list.append(R_gripper2base)
            t_gripper2base_list.append(t_gripper2base)

            R_board2cam_list.append(R_board2cam)
            t_board2cam_list.append(t_board2cam)

            print(
                f"Pose {len(R_gripper2base_list)} recorded | "
                f"TCP x={tcp[0]:.3f}, y={tcp[1]:.3f}, z={tcp[2]:.3f}"
            )

        # =========================
        # ESC: finish
        # =========================
        elif key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()


# =========================
# Solve eye-to-hand calibration
# =========================
if len(R_gripper2base_list) < 5:
    print(f"Not enough poses: {len(R_gripper2base_list)}")
    print("Need at least 5, ideally 15-25.")
else:
    print(f"\nSolving with {len(R_gripper2base_list)} poses...")

    R_base2gripper_list = []
    t_base2gripper_list = []

    for R_g2b, t_g2b in zip(R_gripper2base_list, t_gripper2base_list):
        R_b2g, t_b2g = invert_transform(R_g2b, t_g2b)
        R_base2gripper_list.append(R_b2g)
        t_base2gripper_list.append(t_b2g)

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper_list,
        t_base2gripper_list,
        R_board2cam_list,
        t_board2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_base_camera = make_T(R_cam2base, t_cam2base)

    print("\nT_base_camera:")
    print(T_base_camera)

    np.save("T_base_camera.npy", T_base_camera)
    print("\nSaved: T_base_camera.npy")

    R_base2cam, t_base2cam = invert_transform(R_cam2base, t_cam2base)
    T_camera_base = make_T(R_base2cam, t_base2cam)

    np.save("T_camera_base.npy", T_camera_base)
    print("Saved: T_camera_base.npy")


# =========================
# Example usage
# =========================
print("\nTo transform a camera point to robot base coordinates:")
print("""
T_base_camera = np.load("T_base_camera_NEW.npy")

point_camera = np.array([x, y, z, 1.0]).reshape(4, 1)
point_base = T_base_camera @ point_camera

print(point_base[:3])
""")
