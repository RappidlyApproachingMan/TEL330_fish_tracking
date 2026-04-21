import cv2
import numpy as np
import os

# --------------------------------------------------
# Settings
# --------------------------------------------------
image_path = "TEL330_fish_tracking/fish_fillet_black.png"
output_video = "tracked_simulation.mp4"

num_frames = 60
dx_per_frame = 4     # horizontal conveyor movement
dy_per_frame = 1     # vertical movement if you want slight drift
fps = 10

search_margin = 35
template_pad = 12

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def shift_image(image, dx, dy):
    h, w = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return shifted

def add_motion_blur(image, ksize=7):
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    return cv2.filter2D(image, -1, kernel)

def add_gaussian_noise(image, sigma=4):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# --------------------------------------------------
# Load image
# --------------------------------------------------
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

h, w = img.shape[:2]
gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------
# Detect circle in first frame
# --------------------------------------------------
_, fish_mask = cv2.threshold(gray0, 20, 255, cv2.THRESH_BINARY)
gray_masked = cv2.bitwise_and(gray0, fish_mask)

blur = cv2.GaussianBlur(gray_masked, (11, 11), 2)

circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=80,
    param2=20,
    minRadius=2,
    maxRadius=12
)

if circles is None:
    raise RuntimeError("No circles detected in first frame.")

circles = np.round(circles[0]).astype(int)
x0, y0, r0 = circles[0]
print(f"Initial circle detected at x={x0}, y={y0}, r={r0}")

# --------------------------------------------------
# Build template around detected circle
# --------------------------------------------------
x1 = max(0, x0 - r0 - template_pad)
y1 = max(0, y0 - r0 - template_pad)
x2 = min(w, x0 + r0 + template_pad)
y2 = min(h, y0 + r0 + template_pad)

template = gray0[y1:y2, x1:x2].copy()
template_h, template_w = template.shape[:2]

if template_h < 3 or template_w < 3:
    raise RuntimeError("Template region is too small.")

# --------------------------------------------------
# Generate simulated frames
# --------------------------------------------------
frames = []

rng = np.random.default_rng(42)
y_jitter = 0

for i in range(num_frames):
    dx = i * dx_per_frame

    # stronger wavy and drifting path
    y_wave = 18 * np.sin(i * 0.22) + 8 * np.sin(i * 0.07)
    y_jitter += rng.integers(-3, 4)
    y_jitter = np.clip(y_jitter, -20, 20)

    dy = int(y_wave + y_jitter)

    frame = shift_image(img, dx, dy)
    frame = add_motion_blur(frame, ksize=5)
    frame = add_gaussian_noise(frame, sigma=3)

    frames.append(frame)

# --------------------------------------------------
# Prepare video writer
# --------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

if not writer.isOpened():
    raise RuntimeError("Could not open video writer.")

# --------------------------------------------------
# Track circle and draw path
# --------------------------------------------------
prev_x, prev_y = x0, y0
track_points = []

for i, frame in enumerate(frames):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Search only near previous location
    sx1 = max(0, prev_x - search_margin - template_w // 2)
    sy1 = max(0, prev_y - search_margin - template_h // 2)
    sx2 = min(w, prev_x + search_margin + template_w // 2)
    sy2 = min(h, prev_y + search_margin + template_h // 2)

    search_region = gray[sy1:sy2, sx1:sx2]

    if search_region.shape[0] >= template_h and search_region.shape[1] >= template_w:
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        match_x = sx1 + max_loc[0] + template_w // 2
        match_y = sy1 + max_loc[1] + template_h // 2

        prev_x, prev_y = match_x, match_y
        score = max_val
    else:
        match_x, match_y = prev_x, prev_y
        score = 0.0

    track_points.append((match_x, match_y))

    # Draw tracked circle
    out = frame.copy()
    cv2.circle(out, (match_x, match_y), r0, (0, 255, 0), 2)
    cv2.circle(out, (match_x, match_y), 2, (0, 0, 255), 3)

    # Draw trail line of all tracked positions
    if len(track_points) > 1:
        pts = np.array(track_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=False, color=(255, 0, 0), thickness=2)

    # Add some text
    cv2.putText(
        out,
        f"Frame: {i+1}/{num_frames}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    cv2.putText(
        out,
        f"Match score: {score:.2f}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    writer.write(out)

writer.release()
print(f"Saved video to: {os.path.abspath(output_video)}")