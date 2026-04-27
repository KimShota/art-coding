"""
Luffy Rubber Arm Effect - Real arm stretching, infinite canvas
--------------------------------------------------------------
pip install opencv-python mediapipe numpy

python luffy_rubber_arms.py
"""

import cv2
import mediapipe as mp
import numpy as np
import sys, os, time, urllib.request

MODEL_PATH = os.path.expanduser("~/pose_landmarker_full.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "pose_landmarker/pose_landmarker_full/float16/latest/"
              "pose_landmarker_full.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading pose model (~5MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

options = mp_vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.4,
    min_pose_presence_confidence=0.4,
    min_tracking_confidence=0.4,
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)

L_SHOULDER, L_ELBOW, L_WRIST = 11, 13, 15
R_SHOULDER, R_ELBOW, R_WRIST = 12, 14, 16

SENSITIVITY  = 0.18
MAX_STRETCH  = 20.0   # allow huge stretch
# How much padding to add around the frame so arms can go off-screen
PAD          = 800    # pixels of padding on each side


def get_pt(lm, idx, w, h):
    p = lm[idx]
    vis = getattr(p, 'visibility', 1.0)
    if vis is not None and vis < 0.2:
        return None
    return np.array([p.x * w, p.y * h], dtype=np.float32)


def stretch_factor(shoulder, elbow, wrist, sensitivity):
    if shoulder is None or elbow is None or wrist is None:
        return 1.0
    upper = np.linalg.norm(elbow - shoulder)
    actual = np.linalg.norm(wrist - shoulder)
    natural = upper * 2.0 * sensitivity
    return max(1.0, actual / max(natural, 1.0))


def warp_arm(canvas, shoulder, elbow, wrist, sf, pad):
    """
    Warp the actual forearm pixels. canvas is a padded version of frame.
    All coordinates are offset by pad.
    """
    if shoulder is None or elbow is None or wrist is None:
        return canvas
    if sf <= 1.05:
        return canvas

    sf = min(sf, MAX_STRETCH)
    ch, cw = canvas.shape[:2]

    # Offset points into padded canvas space
    sh = shoulder + pad
    el = elbow    + pad
    wr = wrist    + pad

    lower_dir = wr - el
    lower_len = np.linalg.norm(lower_dir)
    if lower_len < 5:
        return canvas
    lower_unit = lower_dir / lower_len

    upper_dir = el - sh
    upper_len = np.linalg.norm(upper_dir)
    if upper_len < 5:
        return canvas

    arm_width = max(35, upper_len * 0.55)
    perp = np.array([-lower_unit[1], lower_unit[0]])

    # Source quad: actual forearm (elbow -> wrist)
    src = np.float32([
        el + perp * arm_width,
        el - perp * arm_width,
        wr - perp * arm_width * 0.75,
        wr + perp * arm_width * 0.75,
    ])

    # New stretched wrist — can go way beyond canvas bounds, we'll clamp later
    stretched_len = lower_len * sf
    new_wrist = el + lower_unit * stretched_len

    # Destination quad
    dst = np.float32([
        el + perp * arm_width,
        el - perp * arm_width,
        new_wrist - perp * arm_width * 0.75,
        new_wrist + perp * arm_width * 0.75,
    ])

    # Bounding box covering both src and dst, clamped to canvas
    all_pts = np.vstack([src, dst])
    x_min = max(0, int(all_pts[:, 0].min()) - 20)
    x_max = min(cw, int(all_pts[:, 0].max()) + 20)
    y_min = max(0, int(all_pts[:, 1].min()) - 20)
    y_max = min(ch, int(all_pts[:, 1].max()) + 20)

    if x_max <= x_min or y_max <= y_min:
        return canvas

    pw = x_max - x_min
    ph = y_max - y_min

    offset = np.float32([x_min, y_min])
    src_local = src - offset
    dst_local = dst - offset

    M = cv2.getPerspectiveTransform(src_local, dst_local)
    patch = canvas[y_min:y_max, x_min:x_max].copy()
    warped = cv2.warpPerspective(patch, M, (pw, ph),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

    # Mask for destination region
    mask = np.zeros((ph, pw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_local.astype(np.int32), 255)
    mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_f = mask_blur.astype(np.float32) / 255.0
    mask_3 = np.stack([mask_f]*3, axis=2)

    region = canvas[y_min:y_max, x_min:x_max].astype(np.float32)
    blended = region * (1 - mask_3) + warped.astype(np.float32) * mask_3
    canvas[y_min:y_max, x_min:x_max] = blended.astype(np.uint8)

    # GOMU text near stretched wrist (clamped to canvas for display)
    tx = int(np.clip(new_wrist[0], 0, cw - 200))
    ty = int(np.clip(new_wrist[1], 30, ch - 10))
    if sf > 1.4:
        label = "GOMU GOMU NO..." if sf < 2.5 else "PISTOL!!!"
        fscale = min(2.0, sf * 0.5)
        cv2.putText(canvas, label, (tx+2, ty+2),
                    cv2.FONT_HERSHEY_DUPLEX, fscale, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, fscale, (0, 220, 255), 2, cv2.LINE_AA)

    return canvas


def draw_hud(frame, l_sf, r_sf, sensitivity, debug, pose_detected):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (360, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    pd_color = (0, 220, 80) if pose_detected else (0, 80, 220)
    cv2.putText(frame, "Pose: DETECTED" if pose_detected else "Pose: NOT FOUND",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pd_color, 1, cv2.LINE_AA)

    def bar(y, label, val):
        filled = int(min(val - 1, MAX_STRETCH - 1) / (MAX_STRETCH - 1) * 200)
        color = (0, 200, 100) if val < 2 else (0, 180, 255) if val < 4 else (0, 80, 255)
        cv2.rectangle(frame, (10, y), (210, y+14), (60,60,60), -1)
        cv2.rectangle(frame, (10, y), (10+filled, y+14), color, -1)
        cv2.putText(frame, f"{label}: {val:.1f}x", (10, y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1, cv2.LINE_AA)

    bar(36, "Left arm",  l_sf)
    bar(66, "Right arm", r_sf)
    cv2.putText(frame, f"Sens:{sensitivity:.2f} [+/-]  Debug[D]  Quit[Q]",
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140,140,140), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    sensitivity = SENSITIVITY
    debug       = False
    start_time  = time.time()

    # Get actual frame size
    ret, test = cap.read()
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame size: {fw}x{fh}, padding: {PAD}px each side")
    print("Controls: Q=quit  +/-=sensitivity  D=debug")

    # Output window size = original frame (we crop back after warping)
    cv2.namedWindow("Gomu Gomu no... STRETCH!", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gomu Gomu no... STRETCH!", fw, fh)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Detect on original frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Create padded canvas (black border)
        canvas = np.zeros((h + PAD*2, w + PAD*2, 3), dtype=np.uint8)
        canvas[PAD:PAD+h, PAD:PAD+w] = frame

        l_sf = r_sf = 1.0
        pose_detected = bool(results.pose_landmarks)

        if pose_detected:
            lm = results.pose_landmarks[0]

            l_shoulder = get_pt(lm, L_SHOULDER, w, h)
            l_elbow    = get_pt(lm, L_ELBOW,    w, h)
            l_wrist    = get_pt(lm, L_WRIST,    w, h)
            r_shoulder = get_pt(lm, R_SHOULDER, w, h)
            r_elbow    = get_pt(lm, R_ELBOW,    w, h)
            r_wrist    = get_pt(lm, R_WRIST,    w, h)

            l_sf = stretch_factor(l_shoulder, l_elbow, l_wrist, sensitivity)
            r_sf = stretch_factor(r_shoulder, r_elbow, r_wrist, sensitivity)

            canvas = warp_arm(canvas, l_shoulder, l_elbow, l_wrist, l_sf, PAD)
            canvas = warp_arm(canvas, r_shoulder, r_elbow, r_wrist, r_sf, PAD)

            if debug:
                for idx in [L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST]:
                    pt = get_pt(lm, idx, w, h)
                    if pt is not None:
                        cx = int(pt[0]) + PAD
                        cy = int(pt[1]) + PAD
                        cv2.circle(canvas, (cx, cy), 8, (0, 255, 0), -1)

        # Crop back to original frame area for display
        display = canvas[PAD:PAD+h, PAD:PAD+w]

        draw_hud(display, l_sf, r_sf, sensitivity, debug, pose_detected)
        cv2.imshow("Gomu Gomu no... STRETCH!", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('+'), ord('=')):
            sensitivity = min(1.0, round(sensitivity + 0.05, 2))
            print(f"Sensitivity: {sensitivity}")
        elif key == ord('-'):
            sensitivity = max(0.1, round(sensitivity - 0.05, 2))
            print(f"Sensitivity: {sensitivity}")
        elif key == ord('d'):
            debug = not debug

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()