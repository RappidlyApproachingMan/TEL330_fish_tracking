"""
laser_calibration.py
──────────────────────────────────────────────────────────────────────────────
Laser-dot hand-eye calibration for a downward-pointing laser at fixed height.

Workflow
--------
1. Run this script.
2. Move the robot to a position over the table and lower the laser.
3. In the live window, LEFT-CLICK the laser dot.
4. When prompted in the terminal, enter the robot's world XY (e.g. "0.3 0.15").
5. Repeat for at least 4 positions (10-15 recommended for accuracy).
6. Press 'c' to compute and save the homography.
7. Press 'q' to quit at any time.

Controls
--------
  Left-click   – mark laser dot position
  'u'          – undo last point
  'c'          – compute homography (needs ≥ 4 points)
  'q'          – quit

Output
------
  laser_calibration.npz  – homography matrix + raw point pairs
"""

import sys
import time
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARN] pyrealsense2 not found – running in DEMO mode with webcam.")


# ── Configuration ─────────────────────────────────────────────────────────────

WINDOW_NAME   = "Laser Calibration  |  click dot · 'c' compute · 'u' undo · 'q' quit"
OUTPUT_FILE   = "laser_calibration.npz"
MARKER_COLOUR = (0, 255, 0)    # green crosshair for confirmed points
PENDING_COLOUR= (0, 165, 255)  # orange for the click awaiting world input
FONT          = cv2.FONT_HERSHEY_SIMPLEX


# ── Camera helpers ─────────────────────────────────────────────────────────────

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        self.pipeline.start(cfg)
        # Warm-up frames
        for _ in range(10):
            self.pipeline.wait_for_frames()

    def read(self):
        frames = self.pipeline.wait_for_frames()
        colour = frames.get_color_frame()
        if not colour:
            return None
        return np.asanyarray(colour.get_data())

    def release(self):
        self.pipeline.stop()


class WebcamCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()


# ── Calibration logic ──────────────────────────────────────────────────────────

class LaserCalibration:
    def __init__(self):
        self.pixel_points = []   # confirmed (u, v)
        self.world_points  = []  # confirmed (X, Y)
        self.H = None

    # ── point management ──────────────────────────────────────────────────────

    def add_point(self, pixel_uv, world_xy):
        self.pixel_points.append(pixel_uv)
        self.world_points.append(world_xy)
        idx = len(self.pixel_points)
        print(f"  ✓ Point {idx:02d} saved: pixel={pixel_uv}  →  world=({world_xy[0]:.4f}, {world_xy[1]:.4f}) m")

    def undo(self):
        if self.pixel_points:
            px = self.pixel_points.pop()
            wy = self.world_points.pop()
            print(f"  ✗ Removed last point: pixel={px}  world={wy}")
        else:
            print("  Nothing to undo.")

    # ── homography ────────────────────────────────────────────────────────────

    def compute(self):
        n = len(self.pixel_points)
        if n < 4:
            print(f"  [ERROR] Need at least 4 points (have {n}).")
            return False

        src = np.array(self.pixel_points, dtype=np.float32)
        dst = np.array(self.world_points,  dtype=np.float32)

        self.H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=0.005)

        if self.H is None:
            print("  [ERROR] Homography computation failed – check your points.")
            return False

        inliers = int(mask.sum())
        print(f"\n  Homography computed  ({inliers}/{n} inliers)")
        self._print_reprojection(src, dst, mask)
        return True

    def _print_reprojection(self, src, dst, mask):
        errors = []
        for i, (s, d) in enumerate(zip(src, dst)):
            if mask[i]:
                pred = np.array(self.pixel_to_world(tuple(s)))
                err  = np.linalg.norm(pred - d)
                errors.append(err)
        mean_mm = np.mean(errors) * 1000
        max_mm  = np.max(errors)  * 1000
        print(f"  Reprojection error → mean: {mean_mm:.2f} mm   max: {max_mm:.2f} mm")
        if mean_mm > 5:
            print("  [WARN] Error > 5 mm – consider re-clicking noisy points ('u' to undo).")

    def pixel_to_world(self, pixel_uv):
        """Return (X, Y) world coords for a pixel; requires compute() first."""
        assert self.H is not None, "Call compute() first."
        pt = np.array([[[float(pixel_uv[0]), float(pixel_uv[1])]]], dtype=np.float32)
        wp = cv2.perspectiveTransform(pt, self.H)
        return float(wp[0, 0, 0]), float(wp[0, 0, 1])

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path=OUTPUT_FILE):
        np.savez(path,
                 H=self.H,
                 pixel_points=np.array(self.pixel_points),
                 world_points=np.array(self.world_points))
        print(f"  Calibration saved → {path}")

    @staticmethod
    def load(path=OUTPUT_FILE):
        cal = LaserCalibration()
        data = np.load(path)
        cal.H            = data["H"]
        cal.pixel_points = data["pixel_points"].tolist()
        cal.world_points = data["world_points"].tolist()
        print(f"  Calibration loaded from {path}  ({len(cal.pixel_points)} points)")
        return cal


# ── HUD / overlay ──────────────────────────────────────────────────────────────

def draw_overlay(frame, cal, pending_px):
    out = frame.copy()

    # confirmed points
    for px in cal.pixel_points:
        u, v = int(px[0]), int(px[1])
        cv2.drawMarker(out, (u, v), MARKER_COLOUR,
                       cv2.MARKER_CROSS, 18, 2)

    # pending (clicked but world XY not yet entered)
    if pending_px is not None:
        u, v = int(pending_px[0]), int(pending_px[1])
        cv2.drawMarker(out, (u, v), PENDING_COLOUR,
                       cv2.MARKER_CROSS, 18, 2)
        cv2.putText(out, "Enter world XY in terminal", (u + 12, v - 8),
                    FONT, 0.45, PENDING_COLOUR, 1, cv2.LINE_AA)

    # point count
    n = len(cal.pixel_points)
    status = f"Points: {n}  ({'READY – press c' if n >= 4 else f'need {4 - n} more'})"
    cv2.putText(out, status, (10, 22), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # homography result
    if cal.H is not None:
        cv2.putText(out, "Homography OK  |  saved", (10, 46),
                    FONT, 0.55, (0, 255, 100), 1, cv2.LINE_AA)

    return out


# ── World-XY prompt ────────────────────────────────────────────────────────────

def prompt_world_xy():
    """Block on terminal input for 'X Y' in metres. Returns (X, Y) or None."""
    print("\n  Enter robot world XY in metres (e.g.  0.30 0.15 ) or 's' to skip: ", end="", flush=True)
    try:
        line = input().strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if line.lower() == 's':
        return None
    parts = line.split()
    if len(parts) != 2:
        print("  [WARN] Expected two numbers – point skipped.")
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        print("  [WARN] Could not parse numbers – point skipped.")
        return None


# ── Mouse callback ─────────────────────────────────────────────────────────────

class ClickHandler:
    def __init__(self):
        self.pending = None   # pixel coords waiting for world input

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending = (x, y)
            print(f"\n  Clicked pixel ({x}, {y})")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Camera
    if REALSENSE_AVAILABLE:
        print("[INFO] Connecting to RealSense …")
        cam = RealSenseCamera()
    else:
        print("[INFO] Using webcam (demo mode).")
        cam = WebcamCamera()

    cal     = LaserCalibration()
    handler = ClickHandler()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    #time.sleep(1)  # allow camera to warm up
    #cv2.setMouseCallback(WINDOW_NAME, handler.callback)


    print("\n─── Laser Calibration ───────────────────────────────────────────")
    print("  Move robot, shine laser, LEFT-CLICK the dot, enter world XY.")
    print("  Keys:  'c' compute & save  |  'u' undo last point  |  'q' quit")
    print("─────────────────────────────────────────────────────────────────\n")

    while True:
        frame = cam.read()
        if frame is None:
            continue

        # Handle a fresh click – pause for terminal input
        if handler.pending is not None:
            px = handler.pending
            handler.pending = None

            # Show the pending marker while waiting
            preview = draw_overlay(frame, cal, px)
            cv2.imshow(WINDOW_NAME, preview)
            cv2.waitKey(1)

            world_xy = prompt_world_xy()
            if world_xy is not None:
                cal.add_point(px, world_xy)

        display = draw_overlay(frame, cal, handler.pending)
        cv2.imshow(WINDOW_NAME, display)
        cv2.setMouseCallback(WINDOW_NAME, handler.callback)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n  Quitting.")
            break

        elif key == ord('u'):
            cal.undo()

        elif key == ord('c'):
            if cal.compute():
                cal.save()
                print("\n  Press 'q' to quit or keep adding points for higher accuracy.")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()