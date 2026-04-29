import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_receive
from scipy.spatial.transform import Rotation
import rtde_control
import rtde_receive
import time


# --- Connections ---
rtde_c = rtde_control.RTDEControlInterface("192.168.56.101")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.56.101")


# Load saved calibration
T_base_camera = np.load("T_base_camera.npy")

# A point detected by the RealSense in camera frame (meters)
P_camera = np.array([0.3, -0.1, 0.8, 1.0])  # homogeneous

# Transform to robot base frame
P_base = T_base_camera @ P_camera
x, y, z = P_base[:3]

print(f"Target in robot base frame: X={x:.4f} Y={y:.4f} Z={z:.4f}")

# Send to robot (keeping current orientation)
tcp_current = rtde_r.getActualTCPPose()
target_pose = [x, y, z, tcp_current[3], tcp_current[4], tcp_current[5]]

# Move via URScript or rtde_control
rtde_c = rtde_control.RTDEControlInterface("192.168.1.100")
rtde_c.moveL(target_pose, speed=0.1, acceleration=0.5)




##### STEP 1: Getting objects 3D position with the camera. ##### 

def get_object_3d_position(color_frame, depth_frame, intrinsics):
    """
    Detect object in color image, return its 3D position in camera frame.
    This example tracks a bright object — replace with your detector.
    """
    color_image = np.asanyarray(color_frame.get_data())

    # --- Object detection (swap this block with YOLO, template match, etc.) ---
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 72, 120), (11, 255, 255))  # red object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Largest contour centroid
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])

    # --- Get depth at centroid (median over small patch for robustness) ---
    patch = 5
    depths = []
    for du in range(-patch, patch+1):
        for dv in range(-patch, patch+1):
            d = depth_frame.get_distance(
                np.clip(u+du, 0, 639),
                np.clip(v+dv, 0, 479)
            )
            if d > 0:
                depths.append(d)

    if not depths:
        return None

    depth = np.median(depths)

    # --- Back-project to 3D (camera frame) ---
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
    return np.array(point_3d)  # [X, Y, Z] in meters, camera frame





##### STEP 2: The tracking loop. ##### 


# --- Load calibration ---
T_base_camera = np.load("T_base_camera.npy")

HOVER_HEIGHT   = 0.15   # meters above the object
MIN_Z_BASE     = 0.05   # safety floor — never go below this in base frame
MOVE_SPEED     = 0.3    # m/s  — tune to object speed
MOVE_ACCEL     = 0.5
POSITION_TOL   = 0.005  # 5mm — skip move if object hasn't moved more than this



# --- RealSense setup ---
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(cfg)

align    = rs.align(rs.stream.color)
intr     = profile.get_stream(rs.stream.color)\
                  .as_video_stream_profile().get_intrinsics()

last_target = None

try:
    while True:
        # 1. Grab aligned frames
        frames        = pipeline.wait_for_frames()
        aligned       = align.process(frames)
        color_frame   = aligned.get_color_frame()
        depth_frame   = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # 2. Detect object in camera frame
        P_cam = get_object_3d_position(color_frame, depth_frame, intr)
        if P_cam is None:
            print("Object not detected — holding position")
            time.sleep(0.05)
            continue

        # 3. Transform to robot base frame
        P_cam_h = np.append(P_cam, 1.0)          # homogeneous
        P_base  = (T_base_camera @ P_cam_h)[:3]

        # 4. Add hover offset in Z (base frame Z = up for UR10)
        target = np.array([
            P_base[0],
            P_base[1],
            P_base[2] + HOVER_HEIGHT
        ])

        # 5. Safety clamp
        target[2] = max(target[2], MIN_Z_BASE)

        # 6. Skip if object hasn't moved enough
        if last_target is not None:
            delta = np.linalg.norm(target - last_target)
            if delta < POSITION_TOL:
                time.sleep(0.02)
                continue

        # 7. Get current TCP orientation (keep it fixed — only translate)
        tcp_now     = rtde_r.getActualTCPPose()
        target_pose = [
            target[0], target[1], target[2],
            tcp_now[3], tcp_now[4], tcp_now[5]  # fixed orientation
        ]

        # 8. Non-blocking moveL — robot starts moving immediately
        rtde_c.moveL(target_pose, MOVE_SPEED, MOVE_ACCEL, asynchronous=True)

        last_target = target.copy()
        print(f"Tracking → X={target[0]:.3f}  Y={target[1]:.3f}  Z={target[2]:.3f}")

        time.sleep(0.033)  # ~30 Hz loop

finally:
    pipeline.stop()
    rtde_c.stopL(0.5)   # decelerate to stop
    rtde_c.disconnect()