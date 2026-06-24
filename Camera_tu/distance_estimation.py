import argparse
import json
import os
import sys

import cv2
import numpy as np

CALIB_FILE = "calibration.json"
HSV_FILE   = "hsv_bounds.json"

# Reasonable starting points for each colour; the user tunes from here.
# HSV with H in [0, 179], S/V in [0, 255]. Red wraps around 0/180, so we
# encode it as lo[0] > hi[0]; detect_object() handles that wrap.
DEFAULT_HSV = {
    "red":    ([170, 120,  70], [ 10, 255, 255]),
    "blue":   ([100, 120,  70], [130, 255, 255]),
    "green":  ([ 35,  80,  60], [ 85, 255, 255]),
    "yellow": ([ 20, 120,  90], [ 35, 255, 255]),
    "orange": ([  5, 120,  90], [ 18, 255, 255]),
    # Black: any hue, any saturation, very low V. The V_upper is the key knob —
    # raise it to admit darker grays, lower it to demand pure black.
    "black":  ([  0,   0,   0], [180, 255,  50]),
}
COLORS = list(DEFAULT_HSV.keys())


def nothing(_):
    pass


def make_hsv_window(window_name="HSV Tuner",
                    initial=(0, 120, 70, 10, 255, 255)):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 400, 300)
    h_lo, s_lo, v_lo, h_hi, s_hi, v_hi = initial
    cv2.createTrackbar("H lo", window_name, h_lo, 179, nothing)
    cv2.createTrackbar("S lo", window_name, s_lo, 255, nothing)
    cv2.createTrackbar("V lo", window_name, v_lo, 255, nothing)
    cv2.createTrackbar("H hi", window_name, h_hi, 179, nothing)
    cv2.createTrackbar("S hi", window_name, s_hi, 255, nothing)
    cv2.createTrackbar("V hi", window_name, v_hi, 255, nothing)
    return window_name


def read_hsv_bounds(window_name):
    lo = np.array([
        cv2.getTrackbarPos("H lo", window_name),
        cv2.getTrackbarPos("S lo", window_name),
        cv2.getTrackbarPos("V lo", window_name),
    ])
    hi = np.array([
        cv2.getTrackbarPos("H hi", window_name),
        cv2.getTrackbarPos("S hi", window_name),
        cv2.getTrackbarPos("V hi", window_name),
    ])
    return lo, hi


class HSVPicker:
    """Mouse-driven HSV sampler.

    - Hover prints the HSV under the cursor.
    - Click samples a small patch around the cursor and produces (h, s, v)
      means to seed the trackbars with sensible bounds.
    """
    def __init__(self):
        self.frame_hsv  = None
        self.hover_hsv  = None
        self.clicked    = None        # last sampled (h, s, v), consumed by main loop

    def set_frame(self, frame_hsv):
        self.frame_hsv = frame_hsv

    def on_mouse(self, event, x, y, flags, _):
        if self.frame_hsv is None:
            return
        H, W = self.frame_hsv.shape[:2]
        if not (0 <= x < W and 0 <= y < H):
            return
        self.hover_hsv = tuple(int(v) for v in self.frame_hsv[y, x])
        if event == cv2.EVENT_LBUTTONDOWN:
            half = 5
            y0, y1 = max(0, y - half), min(H, y + half)
            x0, x1 = max(0, x - half), min(W, x + half)
            patch  = self.frame_hsv[y0:y1, x0:x1].reshape(-1, 3)
            self.clicked = tuple(int(v) for v in patch.mean(axis=0))


def _apply_clicked_bounds_to_trackbars(tuner, hsv_sample,
                                       h_pad=8, s_pad=30, v_pad=30):
    """Given an (h, s, v) sample, seed the trackbars with sensible bounds:
    H ± h_pad (wraps for red), S in [s - s_pad, 255], V in [v - v_pad, 255]."""
    h, s, v = hsv_sample
    h_lo = (h - h_pad) % 180
    h_hi = (h + h_pad) % 180
    s_lo = max(0, s - s_pad)
    v_lo = max(0, v - v_pad)
    cv2.setTrackbarPos("H lo", tuner, int(h_lo))
    cv2.setTrackbarPos("H hi", tuner, int(h_hi))
    cv2.setTrackbarPos("S lo", tuner, int(s_lo))
    cv2.setTrackbarPos("S hi", tuner, 255)
    cv2.setTrackbarPos("V lo", tuner, int(v_lo))
    cv2.setTrackbarPos("V hi", tuner, 255)


def detect_object(frame, lower_hsv, upper_hsv, min_area=500):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if lower_hsv[0] <= upper_hsv[0]:
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    else:
        # Hue wraps around (e.g. red): build two ranges and OR them.
        lo1 = lower_hsv.copy()
        hi1 = upper_hsv.copy()
        hi1[0] = 179
        lo2 = lower_hsv.copy()
        lo2[0] = 0
        hi2 = upper_hsv.copy()
        mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < min_area:
        return None, mask

    x, y, w, h = cv2.boundingRect(biggest)
    return (x, y, w, h), mask


def save_calibration(focal_length_px, real_width):
    with open(CALIB_FILE, "w") as f:
        json.dump({"focal_length_px": float(focal_length_px),
                   "real_width": float(real_width)}, f, indent=2)


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        return None
    with open(CALIB_FILE) as f:
        return json.load(f)


# ---------- multi-colour HSV-bounds persistence ----------

def _load_all_bounds():
    """Return {color: {"lower": [...], "upper": [...]}}.
    Tolerates the old single-colour format by promoting it to {"red": ...}."""
    if not os.path.exists(HSV_FILE):
        return {}
    with open(HSV_FILE) as f:
        data = json.load(f)
    if isinstance(data, dict) and "lower" in data and "upper" in data:
        return {"red": {"lower": data["lower"], "upper": data["upper"]}}
    return data


def load_hsv_bounds(color):
    """(lower_list, upper_list) for the given colour, or its default."""
    all_b = _load_all_bounds()
    if color in all_b:
        entry = all_b[color]
        return list(entry["lower"]), list(entry["upper"])
    lo, hi = DEFAULT_HSV[color]
    return list(lo), list(hi)


def save_hsv_bounds(color, lower, upper):
    """Update only the selected colour; preserve the others.
    Coerces numpy ints to plain Python ints so json.dump won't choke."""
    all_b = _load_all_bounds()
    all_b[color] = {
        "lower": [int(v) for v in lower],
        "upper": [int(v) for v in upper],
    }
    with open(HSV_FILE, "w") as f:
        json.dump(all_b, f, indent=2)


# ---------- main flows ----------

def calibrate(cap, real_width, known_distance, color):
    """Compute focal length: F = (P * D) / W, where
    P = perceived pixel width, D = known distance, W = real width."""
    window = f"Calibration [{color}] - SPACE to capture, q to quit"
    lo_init, hi_init = load_hsv_bounds(color)
    tuner = make_hsv_window(f"HSV Tuner (calibration: {color})",
                            initial=(*lo_init, *hi_init))
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    picker = HSVPicker()
    cv2.setMouseCallback(window, picker.on_mouse)
    print(f"\nCALIBRATION [{color}]: hold the {color} marker at "
          f"{known_distance} units from the camera.\n"
          f"Click the marker to auto-seed the HSV bounds, fine-tune with the "
          f"trackbars, then press SPACE.\n")

    focal_length = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.", file=sys.stderr)
            return None

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        picker.set_frame(hsv_frame)
        if picker.clicked is not None:
            _apply_clicked_bounds_to_trackbars(tuner, picker.clicked)
            picker.clicked = None

        lo, hi = read_hsv_bounds(tuner)
        bbox, mask = detect_object(frame, lo, hi)

        display = frame.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, f"pixel width: {w}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display,
                    f"[{color}] SPACE=capture D={known_distance} W={real_width}  click=sample HSV",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        if picker.hover_hsv is not None:
            cv2.putText(display,
                        f"hover HSV: H={picker.hover_hsv[0]} "
                        f"S={picker.hover_hsv[1]} V={picker.hover_hsv[2]}",
                        (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow(window, display)
        cv2.imshow(f"Mask ({color})", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and bbox is not None:
            pixel_width = bbox[2]
            focal_length = (pixel_width * known_distance) / real_width
            save_calibration(focal_length, real_width)
            save_hsv_bounds(color, lo, hi)
            print(f"Captured. focal_length = {focal_length:.2f} px "
                  f"(HSV bounds saved for {color})")
            break

    cv2.destroyWindow(window)
    cv2.destroyWindow(f"Mask ({color})")
    cv2.destroyWindow(f"HSV Tuner (calibration: {color})")
    return focal_length


def run_estimation(cap, focal_length, real_width, color):
    window = f"Distance Estimation [{color}] - q to quit, s to save HSV, click to sample"
    lo_init, hi_init = load_hsv_bounds(color)
    tuner = make_hsv_window(f"HSV Tuner ({color})",
                            initial=(*lo_init, *hi_init))
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    picker = HSVPicker()
    cv2.setMouseCallback(window, picker.on_mouse)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.", file=sys.stderr)
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        picker.set_frame(hsv_frame)
        if picker.clicked is not None:
            _apply_clicked_bounds_to_trackbars(tuner, picker.clicked)
            print(f"Sampled HSV {picker.clicked} -> trackbars seeded for {color}.")
            picker.clicked = None

        lo, hi = read_hsv_bounds(tuner)
        bbox, mask = detect_object(frame, lo, hi)

        display = frame.copy()
        if bbox is not None:
            x, y, w, h = bbox
            distance = (real_width * focal_length) / w   # similar triangles
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, f"[{color}] {distance:.2f} units",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"px width: {w}", (x, y + h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"[{color}] no object detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if picker.hover_hsv is not None:
            cv2.putText(display,
                        f"hover HSV: H={picker.hover_hsv[0]} "
                        f"S={picker.hover_hsv[1]} V={picker.hover_hsv[2]}",
                        (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow(window, display)
        cv2.imshow(f"Mask ({color})", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            save_hsv_bounds(color, lo, hi)
            print(f"HSV bounds saved for {color}.")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Monocular distance estimation via color detection.")
    parser.add_argument("--color", choices=COLORS, default="red",
                        help="Colour to calibrate / detect (default red).")
    parser.add_argument("--width", type=float, default=7.5,
                        help="Real width of the marker in cm (default 7.5 — "
                             "the longer side of the 7.5x6.5 marker; orient it "
                             "with that side running left-right in the frame).")
    parser.add_argument("--distance", type=float,
                        help="Known distance during calibration "
                             "(same units as --width). Required on first run "
                             "or with --recalibrate.")
    parser.add_argument("--camera", type=int, default=2,
                        help="Camera index (default 2 = secondary webcam; "
                             "0 is the built-in. /dev/video1 and "
                             "/dev/video3 are typically metadata-only nodes.)")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Force a fresh focal-length capture even if one is saved.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.", file=sys.stderr)
        sys.exit(1)

    # Lock auto-white-balance and autofocus so hues / sharpness stay
    # stable while tuning. Some C270 firmware ignores these via OpenCV;
    # if so, the v4l2-ctl commands in the README handle it for you.
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    calib = load_calibration()
    needs_calib = args.recalibrate or calib is None or \
        calib.get("real_width") != args.width

    if needs_calib:
        if args.distance is None:
            print("First run (or --recalibrate / changed --width) needs "
                  "--distance.", file=sys.stderr)
            cap.release()
            sys.exit(1)
        focal_length = calibrate(cap, args.width, args.distance, args.color)
        if focal_length is None:
            cap.release()
            sys.exit(1)
    else:
        focal_length = calib["focal_length_px"]
        print(f"Loaded calibration: focal_length = {focal_length:.2f} px "
              f"(tuning HSV for {args.color}; press 's' in the window to save).")

    run_estimation(cap, focal_length, args.width, args.color)
    cap.release()


if __name__ == "__main__":
    main()
