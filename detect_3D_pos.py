import cv2
import numpy as np
import pyrealsense2 as rs   
import rtde_control
import rtde_receive
import time

# --- Checkerboard config ---
# These are the INNER corners (squares - 1) in (columns, rows)
BOARD_SIZE    = (9, 6)       # e.g. a 9x7 square board has 8x6 inner corners
SQUARE_SIZE   = 0.022        # meters — measure your printed board square size

# Build the real-world 3D corner positions (flat on Z=0 plane)
objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE          # scale to meters

def get_checkerboard_3d_position(color_frame, depth_frame, intrinsics,
                                  camera_matrix, dist_coeffs):
    """
    Detect checkerboard, return its center position in camera frame.
    """
    color_image = np.asanyarray(color_frame.get_data())
    gray        = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # --- Find inner corners ---
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, flags)

    if not found:
        return None, None

    # --- Refine corner locations to sub-pixel accuracy ---
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # --- Solve PnP: get board pose relative to camera ---
    success, rvec, tvec = cv2.solvePnP(
        objp, corners, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None

    # tvec is the board ORIGIN (top-left inner corner) in camera frame
    # Instead, compute the CENTER of the board
    center_objp = np.array([[
        (BOARD_SIZE[0] - 1) * SQUARE_SIZE / 2,
        (BOARD_SIZE[1] - 1) * SQUARE_SIZE / 2,
        0.0
    ]], dtype=np.float32)

    # Project center point to camera frame
    center_cam, _ = cv2.projectPoints(center_objp, rvec, tvec,
                                       camera_matrix, dist_coeffs)
    u, v = int(center_cam[0][0][0]), int(center_cam[0][0][1])

    # --- Get depth at projected center (median patch) ---
    patch = 5
    depths = []
    for du in range(-patch, patch + 1):
        for dv in range(-patch, patch + 1):
            d = depth_frame.get_distance(
                int(np.clip(u + du, 0, 639)),
                int(np.clip(v + dv, 0, 479))
            )
            if d > 0:
                depths.append(d)

    if not depths:
        return None, None

    depth = np.median(depths)

    # --- Back-project center pixel to 3D ---
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
    return np.array(point_3d), corners   # return corners for visualization


pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile  = pipeline.start(cfg)

align = rs.align(rs.stream.color)
intr  = profile.get_stream(rs.stream.color)\
               .as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([[intr.fx, 0,       intr.ppx],
                           [0,       intr.fy, intr.ppy],
                           [0,       0,       1       ]])
dist_coeffs   = np.array(intr.coeffs)


T_base_camera = np.load("T_base_camera.npy")

HOVER_HEIGHT = 0.15
MIN_Z_BASE   = 0.05
MOVE_SPEED   = 0.3
MOVE_ACCEL   = 0.5
POSITION_TOL = 0.005

rtde_c = rtde_control.RTDEControlInterface("192.168.1.100")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.100")

class LowPassFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None
    def update(self, x):
        self.value = x if self.value is None \
                     else self.alpha * x + (1 - self.alpha) * self.value
        return self.value.copy()

position_filter = LowPassFilter(alpha=0.3)
last_target     = None

try:
    while True:
        frames      = pipeline.wait_for_frames()
        aligned     = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # 1. Detect checkerboard center in camera frame
        P_cam, corners = get_checkerboard_3d_position(
            color_frame, depth_frame, intr, camera_matrix, dist_coeffs
        )

        if P_cam is None:
            print("Checkerboard not detected — holding position")
            time.sleep(0.05)
            continue

        # 2. Transform to robot base frame
        P_base = (T_base_camera @ np.append(P_cam, 1.0))[:3]

        # 3. Hover above
        target    = P_base.copy()
        target[2] = P_base[2] + HOVER_HEIGHT
        target[2] = max(target[2], MIN_Z_BASE)

        # 4. Smooth
        target = position_filter.update(target)

        # 5. Dead-band check
        if last_target is not None:
            if np.linalg.norm(target - last_target) < POSITION_TOL:
                time.sleep(0.02)
                continue

        # 6. Move robot
        tcp_now     = rtde_r.getActualTCPPose()
        target_pose = [target[0], target[1], target[2],
                       tcp_now[3], tcp_now[4], tcp_now[5]]

        rtde_c.moveL(target_pose, MOVE_SPEED, MOVE_ACCEL, asynchronous=True)
        last_target = target.copy()

        print(f"Tracking → X={target[0]:.3f}  Y={target[1]:.3f}  Z={target[2]:.3f}")
        time.sleep(0.033)

finally:
    pipeline.stop()
    rtde_c.stopL(0.5)
    rtde_c.disconnect()