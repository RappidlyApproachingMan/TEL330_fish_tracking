import cv2
import numpy as np
import os
import glob

# --------------------------------------------------
# Settings
# --------------------------------------------------
image_folder = "fish"
image_pattern = os.path.join(image_folder, "*.png")
output_video = "tracked_fish_sequence_NEWW.mp4"

fps = 5

# --------------------------------------------------
# Function: find salmon ROI
# --------------------------------------------------
def find_salmon_roi(img):
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_salmon = np.array([0, 40, 80])
    upper_salmon = np.array([30, 255, 255])

    salmon_mask_full = cv2.inRange(hsv_full, lower_salmon, upper_salmon)

    kernel = np.ones((7, 7), np.uint8)
    salmon_mask_full = cv2.morphologyEx(salmon_mask_full, cv2.MORPH_CLOSE, kernel)
    salmon_mask_full = cv2.morphologyEx(salmon_mask_full, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        salmon_mask_full,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)

    if cv2.contourArea(c) < 200:
        return None

    x, y, w, h = cv2.boundingRect(c)

    pad = 50
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)

    return x1, y1, x2, y2


# --------------------------------------------------
# Function: detect circle inside dynamic ROI
# --------------------------------------------------
def detect_tracking_circle(img):
    roi_box = find_salmon_roi(img)

    if roi_box is None:
        return None, None

    x1, y1, x2, y2 = roi_box
    roi = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, fish_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    gray_masked = cv2.bitwise_and(gray, fish_mask)

    gray_masked = cv2.normalize(gray_masked, None, 0, 255, cv2.NORM_MINMAX)

    blur = cv2.GaussianBlur(gray_masked, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=25,
        param1=80,
        param2=14,
        minRadius=2,
        maxRadius=12
    )

    if circles is None:
        return None, roi_box

    circles = np.round(circles[0]).astype(int)

    best_circle = None
    best_score = -1

    for cx, cy, r in circles:
        patch_size = 8
        px1 = max(0, cx - patch_size)
        py1 = max(0, cy - patch_size)
        px2 = min(gray.shape[1], cx + patch_size)
        py2 = min(gray.shape[0], cy + patch_size)

        patch = gray[py1:py2, px1:px2]
        score = np.mean(patch)

        if score > best_score:
            best_score = score
            best_circle = (cx, cy, r)

    if best_circle is None:
        return None, roi_box

    cx, cy, r = best_circle

    full_x = cx + x1
    full_y = cy + y1

    return (full_x, full_y, r, best_score), roi_box


# --------------------------------------------------
# Load image sequence
# --------------------------------------------------
image_paths = sorted(glob.glob(image_pattern))

if len(image_paths) == 0:
    raise FileNotFoundError(f"No images found with pattern: {image_pattern}")

frames = []
valid_paths = []

for path in image_paths:
    frame = cv2.imread(path)

    if frame is None:
        print(f"Warning: could not read {path}")
        continue

    frames.append(frame)
    valid_paths.append(path)

if len(frames) == 0:
    raise RuntimeError("No valid images loaded.")

h, w = frames[0].shape[:2]

# Make sure all frames have same size
fixed_frames = []
for frame in frames:
    if frame.shape[:2] != (h, w):
        frame = cv2.resize(frame, (w, h))
    fixed_frames.append(frame)

frames = fixed_frames

# --------------------------------------------------
# Prepare video writer
# --------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

if not writer.isOpened():
    raise RuntimeError("Could not open video writer.")

# --------------------------------------------------
# Detect circle in every frame and draw track line
# --------------------------------------------------
track_points = []
last_valid_point = None
last_radius = 5

for i, frame in enumerate(frames):
    out = frame.copy()

    detected, roi_box = detect_tracking_circle(frame)

    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if detected is not None:
        x, y, r, score = detected
        last_valid_point = (x, y)
        last_radius = r
        track_points.append((x, y))
        status_text = f"Detected score: {score:.1f}"
    else:
        # If detection fails in one frame, keep previous point visually
        if last_valid_point is not None:
            x, y = last_valid_point
            r = last_radius
            track_points.append((x, y))
            status_text = "Detection failed, using last point"
        else:
            x, y, r = None, None, None
            status_text = "Detection failed"

    # Draw detected/tracked point
    if x is not None:
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)
        cv2.circle(out, (x, y), 2, (0, 0, 255), 3)

    # Draw tracking line
    if len(track_points) > 1:
        pts = np.array(track_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=False, color=(255, 0, 0), thickness=2)

    cv2.putText(
        out,
        f"Frame: {i + 1}/{len(frames)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        out,
        status_text,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    writer.write(out)

writer.release()

print(f"Saved video to: {os.path.abspath(output_video)}")
