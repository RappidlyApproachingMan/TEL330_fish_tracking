import cv2
import numpy as np
import pyrealsense2 as rs

# --- Checkerboard config --- must match your board!
BOARD_SIZE  = (9, 6)      # inner corners (cols, rows)
SQUARE_SIZE = 0.022       # meters — measure with a ruler

objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# --- RealSense setup ---
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile  = pipeline.start(cfg)

obj_points = []   # 3D points in real world
img_points = []   # 2D points in image
gray       = None

print("=== Camera Intrinsic Calibration ===")
print("Hold the checkerboard in front of the camera by hand (robot not needed)")
print("Move it to many different positions, angles and distances")
print("Aim for 20+ samples covering the full FOV")
print("\nPress SPACE to record, D to delete last, ESC to solve and save\n")

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
            cv2.putText(display, f"Board found!  Samples: {len(obj_points)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"Board NOT found  Samples: {len(obj_points)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display, "SPACE=record  D=delete last  ESC=solve & save",
                    (10, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        cv2.imshow("Camera Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if not found:
                print("Board not visible — reposition and try again")
                continue
            obj_points.append(objp)
            img_points.append(corners)
            print(f"Sample {len(obj_points):02d} recorded")

        elif key == ord("d"):
            if obj_points:
                obj_points.pop()
                img_points.pop()
                print(f"Deleted last sample — {len(obj_points)} remaining")
            else:
                print("Nothing to delete")

        elif key == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# --- Solve ---
n = len(obj_points)
if n < 15:
    print(f"Not enough samples ({n}) — need at least 15, ideally 20+")
else:
    print(f"\nCalibrating with {n} samples...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    print(f"\nReprojection error: {ret:.4f} px  (should be under 1.0, ideally under 0.5)")
    print("\nCamera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)

    if ret > 1.0:
        print("\nWARNING: reprojection error is high — consider recollecting samples")
        print("Make sure the board is flat, well lit, and covers all corners of the FOV")
    else:
        print("\nCalibration looks good!")
        np.save("camera_matrix.npy", camera_matrix)
        np.save("dist_coeffs.npy",   dist_coeffs)
        print("Saved camera_matrix.npy and dist_coeffs.npy")