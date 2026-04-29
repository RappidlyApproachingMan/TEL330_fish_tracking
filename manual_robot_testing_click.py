import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_control
import rtde_receive
from moving_robot import get_object_3d_position  # or paste the function in



clicked_point = None

def on_mouse_click(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", on_mouse_click)





# --- Config ---
HOVER_HEIGHT = 0.15
MIN_Z_BASE   = 0.05
MOVE_SPEED   = 0.05   # very slow for testing!
MOVE_ACCEL   = 0.3

# --- Init ---
T_base_camera = np.load("T_base_camera.npy")
rtde_c = rtde_control.RTDEControlInterface("192.168.56.101")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.56.101")

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile  = pipeline.start(cfg)
align    = rs.align(rs.stream.color)
intr     = profile.get_stream(rs.stream.color)\
                  .as_video_stream_profile().get_intrinsics()

print("Press SPACE to capture and move, ESC to quit.\n")

try:
    while True:
        frames      = pipeline.wait_for_frames()
        aligned     = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Show live feed so you can see what the camera sees
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("Camera", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key == ord(" "):  # SPACE — capture and move
            if clicked_point is None:
                    print("Click a point in the image first")
                    continue

            u, v = clicked_point
            depth = depth_frame.get_distance(u, v)

            if depth == 0:
                print("No depth at that pixel — try another spot")
                continue

            P_cam = np.array(rs.rs2_deproject_pixel_to_point(intr, [u, v], depth))
            print(f"Clicked pixel ({u}, {v}), depth={depth:.3f}m")
            
            # Transform to base frame
            P_cam_h = np.append(P_cam, 1.0)
            P_base  = (T_base_camera @ P_cam_h)[:3]
            print(f"P_base  : X={P_base[0]:.4f}  Y={P_base[1]:.4f}  Z={P_base[2]:.4f}")

            # Apply hover and safety floor
            target    = P_base.copy()
            target[2] = max(P_base[2] + HOVER_HEIGHT, MIN_Z_BASE)
            print(f"Target  : X={target[0]:.4f}  Y={target[1]:.4f}  Z={target[2]:.4f}")

            # Ask for confirmation before moving
            confirm = input("Send to robot? (y/n): ").strip().lower()
            if confirm != "y":
                print("Skipped.")
                continue

            tcp_now     = rtde_r.getActualTCPPose()
            target_pose = [
                target[0], target[1], target[2],
                tcp_now[3], tcp_now[4], tcp_now[5]
            ]

            print("Moving...")
            rtde_c.moveL(target_pose, MOVE_SPEED, MOVE_ACCEL)  # blocking
            print("Done.\n")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    rtde_c.disconnect()