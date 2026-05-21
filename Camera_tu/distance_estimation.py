import argparse
import json
import os
import sys

import cv2
import numpy as np

CALIB_FILE = "calibration.json"


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
        json.dump({"focal_length_px": focal_length_px,
                   "real_width": real_width}, f, indent=2)


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        return None
    with open(CALIB_FILE) as f:
        return json.load(f)


def calibrate(cap, real_width, known_distance):
    """Compute focal length: F = (P * D) / W, where
    P = perceived pixel width, D = known distance, W = real width."""
    window = "Calibration - press SPACE when object is locked, q to quit"
    tuner = make_hsv_window("HSV Tuner (calibration)")
    print(f"\nCALIBRATION: hold the object at {known_distance} units from "
          f"the camera.\nTune the HSV trackbars until the object is cleanly "
          f"isolated, then press SPACE to capture.\n")

    focal_length = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.", file=sys.stderr)
            return None

        lo, hi = read_hsv_bounds(tuner)
        bbox, mask = detect_object(frame, lo, hi)

        display = frame.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, f"pixel width: {w}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display,
                    f"SPACE = capture at D={known_distance}, W={real_width}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
        cv2.imshow(window, display)
        cv2.imshow("Mask (calibration)", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and bbox is not None:
            pixel_width = bbox[2]
            focal_length = (pixel_width * known_distance) / real_width
            save_calibration(focal_length, real_width)
            # Also persist the HSV bounds the user dialed in.
            with open("hsv_bounds.json", "w") as f:
                json.dump({"lower": lo.tolist(), "upper": hi.tolist()}, f)
            print(f"Captured. focal_length = {focal_length:.2f} px")
            break

    cv2.destroyWindow(window)
    cv2.destroyWindow("Mask (calibration)")
    cv2.destroyWindow(tuner)
    return focal_length


def run_estimation(cap, focal_length, real_width):
    window = "Distance Estimation - q to quit"
    initial = (0, 120, 70, 10, 255, 255)
    if os.path.exists("hsv_bounds.json"):
        with open("hsv_bounds.json") as f:
            saved = json.load(f)
        initial = (*saved["lower"], *saved["upper"])
    tuner = make_hsv_window("HSV Tuner", initial=initial)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.", file=sys.stderr)
            break

        lo, hi = read_hsv_bounds(tuner)
        bbox, mask = detect_object(frame, lo, hi)

        display = frame.copy()
        if bbox is not None:
            x, y, w, h = bbox
            # Similar triangles: D = (W * F) / P
            distance = (real_width * focal_length) / w
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, f"{distance:.2f} units",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"px width: {w}", (x, y + h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(display, "no object detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(window, display)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Monocular distance estimation via color detection.")
    parser.add_argument("--width", type=float, default=7.5,
                        help="Real width of the object in cm (default 7.5).")
    parser.add_argument("--distance", type=float,
                        help="Known distance during calibration "
                             "(same units as --width). Required on first run.")
    parser.add_argument("--camera", type=int, default=2,
                        help="Camera index (default 2 = secondary webcam; "
                             "0 is the built-in. Note: /dev/video1 and "
                             "/dev/video3 are metadata-only sub-devices.)")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Force a fresh calibration even if one is saved.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.", file=sys.stderr)
        sys.exit(1)

    calib = load_calibration()
    needs_calib = args.recalibrate or calib is None or \
        calib.get("real_width") != args.width

    if needs_calib:
        if args.distance is None:
            print("First run (or --recalibrate / changed --width) needs "
                  "--distance.", file=sys.stderr)
            cap.release()
            sys.exit(1)
        focal_length = calibrate(cap, args.width, args.distance)
        if focal_length is None:
            cap.release()
            sys.exit(1)
    else:
        focal_length = calib["focal_length_px"]
        print(f"Loaded calibration: focal_length = {focal_length:.2f} px")

    run_estimation(cap, focal_length, args.width)
    cap.release()


if __name__ == "__main__":
    main()
