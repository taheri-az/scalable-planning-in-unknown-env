"""
Monocular color-based label detector.

Captures one frame from a USB camera, looks for a red square (the agreed
marker for label `a`), and returns the perceived label string plus the slant
distance from the camera to the marker in meters. Uses the calibration and
HSV bounds saved by Camera_tu/distance_estimation.py.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np


CALIB_PATH      = Path(__file__).parent / "Camera_tu" / "calibration.json"
HSV_BOUNDS_PATH = Path(__file__).parent / "Camera_tu" / "hsv_bounds.json"

# Colour → label string used by the planner / DFA.
COLOR_LABEL = {
    "red": "a && !b && !c",
}
EMPTY_LABEL = "!a && !b && !c"


class LabelDetector:
    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        warmup_seconds: float = 2.0,
        min_area_px: int = 500,
    ):
        with open(CALIB_PATH) as f:
            calib = json.load(f)
        self.focal_px      = float(calib["focal_length_px"])
        self.real_width_cm = float(calib["real_width"])

        with open(HSV_BOUNDS_PATH) as f:
            bounds = json.load(f)
        self.hsv_lo = np.array(bounds["lower"], dtype=np.int32)
        self.hsv_hi = np.array(bounds["upper"], dtype=np.int32)

        self.min_area_px = min_area_px

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        # Let auto-exposure / white-balance settle.
        t0 = time.time()
        while time.time() - t0 < warmup_seconds:
            self.cap.read()

    def _color_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo, hi = self.hsv_lo, self.hsv_hi
        if lo[0] <= hi[0]:
            mask = cv2.inRange(hsv, lo, hi)
        else:
            # Hue wraps around (e.g. red).
            lo1, hi1 = lo.copy(), hi.copy(); hi1[0] = 179
            lo2, hi2 = lo.copy(), hi.copy(); lo2[0] = 0
            mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def detect(self):
        """
        Grab one frame, look for the marker, return (label_str, distance_m).
        If nothing is detected, returns (None, None).
        """
        ok, frame = self.cap.read()
        if not ok:
            return None, None

        mask = self._color_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < self.min_area_px:
            return None, None

        _, _, pix_w, _ = cv2.boundingRect(biggest)
        if pix_w <= 0:
            return None, None

        # Pinhole: D = (W * F) / P. Calibration was in cm; convert to meters.
        distance_cm = (self.real_width_cm * self.focal_px) / pix_w
        distance_m  = distance_cm / 100.0
        return COLOR_LABEL["red"], distance_m

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        try: self.close()
        except Exception: pass
