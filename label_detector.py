"""
Monocular color-based label detector.

Captures one frame from a USB camera, looks for a red square (the agreed
marker for label `a`), and returns the perceived label string plus the slant
distance from the camera to the marker in meters. Uses the calibration and
HSV bounds saved by Camera_tu/distance_estimation.py.
"""

import atexit
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np


CALIB_PATH      = Path(__file__).parent / "Camera_tu" / "calibration.json"
HSV_BOUNDS_PATH = Path(__file__).parent / "Camera_tu" / "hsv_bounds.json"

# Colour → label string used by the planner / DFA.
# Mapping for this experiment: red → a, yellow → b, green → c.
# Blue and orange are detected and shown in the recording but produce no DFA
# transition (label = None) — useful for verifying perception independently.
COLOR_LABEL = {
    "red":    "a && !b && !c",
    "yellow": "!a && b && !c",
    "green":  "!a && !b && c",
    "blue":   None,
    "orange": None,
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
        record_path: str = None,
        record_fps: float = 20.0,
    ):
        with open(CALIB_PATH) as f:
            calib = json.load(f)
        self.focal_px      = float(calib["focal_length_px"])
        self.real_width_cm = float(calib["real_width"])

        # Load HSV bounds; supports new multi-colour format
        # {color: {lower, upper}} and the legacy single-colour {lower, upper}.
        with open(HSV_BOUNDS_PATH) as f:
            raw = json.load(f)
        if "lower" in raw and "upper" in raw:
            raw = {"red": {"lower": raw["lower"], "upper": raw["upper"]}}
        self._color_bounds = {
            color: (np.array(entry["lower"], dtype=np.int32),
                    np.array(entry["upper"], dtype=np.int32))
            for color, entry in raw.items()
        }

        self.min_area_px = min_area_px
        self._frame_size = (frame_width, frame_height)

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        # Lock auto-white-balance and autofocus so colour bounds tuned on the
        # laptop carry over to the robot without hue drift.
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # Force a single-frame buffer so cap.read() always returns the
        # freshest frame instead of any buffered older one. Without this,
        # detect() sometimes runs on a frame from before the last turn.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Let auto-exposure / white-balance settle.
        t0 = time.time()
        while time.time() - t0 < warmup_seconds:
            self.cap.read()

        # Optional background video recorder. The grab thread owns cap.read();
        # detect() reads from the shared latest-frame buffer so we don't
        # double-open the camera.
        self._latest_frame = None
        self._frame_lock   = threading.Lock()
        self._shutdown     = False

        self._writer = None
        self._record_fps = record_fps
        if record_path is not None:
            # mp4v in .mp4: much smaller than MJPG at the same visual quality.
            # The atexit handler below releases the writer on clean exits so
            # the moov atom gets written and the file stays playable.
            # IMPORTANT: only interrupt with Ctrl-C (never Ctrl-Z) — suspending
            # the process prevents close() from running.
            self._writer = cv2.VideoWriter(
                record_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                record_fps,
                self._frame_size,
            )

        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

        # Make sure the writer + camera get released even on KeyboardInterrupt
        # or other unclean exits.
        atexit.register(self.close)

    def _grab_loop(self):
        write_interval = 1.0 / self._record_fps if self._record_fps > 0 else 0.0
        last_write = 0.0
        while not self._shutdown:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._frame_lock:
                self._latest_frame = frame
            if self._writer is None:
                continue
            now = time.time()
            if now - last_write < write_interval:
                continue
            annotated = frame.copy()
            label, dist_m, bbox, color = self._detect_in_frame(annotated)
            self._annotate(annotated, label, dist_m, bbox, color)
            self._writer.write(annotated)
            last_write = now

    def _detect_in_frame(self, frame):
        """
        Scan every configured colour, pick the largest valid blob across all,
        and return (label_str_or_None, distance_m, (x, y, w, h), color_name).
        Returns (None, None, None, None) if nothing meets `min_area_px`.

        label_str is None when the detected colour isn't mapped to a DFA
        proposition (e.g. yellow / orange) — useful for verifying perception
        without affecting the planner.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        best_area = 0
        best = None
        for color, (lo, hi) in self._color_bounds.items():
            mask = self._color_mask(hsv, lo, hi)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            biggest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(biggest)
            if area < self.min_area_px or area <= best_area:
                continue
            x, y, w, h = cv2.boundingRect(biggest)
            if w <= 0:
                continue
            best_area = area
            best = (color, x, y, w, h)
        if best is None:
            return None, None, None, None
        color, x, y, w, h = best
        distance_cm = (self.real_width_cm * self.focal_px) / w
        distance_m  = distance_cm / 100.0
        return COLOR_LABEL.get(color), distance_m, (x, y, w, h), color

    @staticmethod
    def _annotate(frame, label, distance_m, bbox, color=None):
        if bbox is None:
            return
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_top = color if color is not None else str(label)
        cv2.putText(frame, text_top, (x, max(y - 28, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"{distance_m * 100:.1f} cm", (x, max(y - 6, 36)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    @staticmethod
    def _color_mask(hsv, lo, hi):
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
        Look at the most recent frame, find the strongest colour marker, and
        return (label_str_or_None, distance_m, color_name). The label is None
        if the detected colour isn't mapped to a DFA proposition. Returns
        (None, None, None) if nothing meets the minimum area.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None, None, None
            frame = self._latest_frame.copy()
        label, dist_m, _bbox, color = self._detect_in_frame(frame)
        return label, dist_m, color

    def close(self):
        self._shutdown = True
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=2.0)
            self._grab_thread = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        try: self.close()
        except Exception: pass
