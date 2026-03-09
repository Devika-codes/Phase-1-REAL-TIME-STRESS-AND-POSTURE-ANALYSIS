"""
blink_detector.py
-----------------
Detects eye blinks via the Eye Aspect Ratio (EAR) computed from
MediaPipe FaceMesh landmarks, and flags a low-blink-rate alert.

EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

When EAR drops below EAR_THRESHOLD a blink is registered.
If blinks per minute fall below BLINK_RATE_THRESHOLD → alert.

MediaPipe FaceMesh landmark indices (right eye used as reference):
    p1=33, p2=160, p3=158, p4=133, p5=153, p6=144   (left eye)
    p1=362, p2=385, p3=387, p4=263, p5=373, p6=380  (right eye)
"""

import time
import cv2
import numpy as np
from utils import calculate_distance


class BlinkDetector:
    """
    Tracks blink events and warns when blink rate is too low.

    Healthy blink rate: ~15-20 blinks per minute.
    Alert is raised when sustained blink rate < BLINK_RATE_THRESHOLD.
    """

    # ── Thresholds ──────────────────────────────────────────────────────────────
    EAR_THRESHOLD         = 0.22    # EAR below this = eye closed
    EAR_CONSEC_FRAMES     = 2       # Min consecutive frames for a valid blink
    BLINK_RATE_THRESHOLD  = 12      # Blinks per minute — below this = alert
    MEASUREMENT_WINDOW    = 60.0    # Seconds over which blink rate is computed

    def __init__(self):
        # MediaPipe landmark indices for both eyes
        self._LEFT_EYE  = [33,  160, 158, 133, 153, 144]
        self._RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # State
        self._frame_counter  = 0      # Consecutive frames with eyes closed
        self._blink_times: list[float] = []  # Timestamps of detected blinks
        self._total_blinks   = 0

        self.ear             = 0.0
        self.blink_rate      = 0.0    # Blinks per minute (rolling)
        self.alert_needed    = False

    # ── Public API ───────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray, face_landmarks) -> tuple[np.ndarray, float, float, bool]:
        """
        Process facial landmarks for blink detection.

        Args:
            frame           – BGR frame (for drawing)
            face_landmarks  – mediapipe FaceMesh landmark object (or None)

        Returns:
            frame           – annotated frame
            ear             – current Eye Aspect Ratio
            blink_rate      – blinks per minute (rolling 60 s window)
            alert_needed    – True when blink rate is dangerously low
        """
        if face_landmarks is None:
            return frame, self.ear, self.blink_rate, False

        lm  = face_landmarks.landmark
        h, w = frame.shape[:2]

        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        # ── Compute EAR for both eyes ────────────────────────────────────────────
        left_ear  = self._eye_aspect_ratio([px(i) for i in self._LEFT_EYE])
        right_ear = self._eye_aspect_ratio([px(i) for i in self._RIGHT_EYE])
        self.ear  = (left_ear + right_ear) / 2.0

        # ── Blink detection state machine ────────────────────────────────────────
        if self.ear < self.EAR_THRESHOLD:
            self._frame_counter += 1
        else:
            if self._frame_counter >= self.EAR_CONSEC_FRAMES:
                # A complete blink was registered
                self._total_blinks += 1
                self._blink_times.append(time.time())
            self._frame_counter = 0

        # ── Prune blink history older than the measurement window ─────────────────
        now = time.time()
        cutoff = now - self.MEASUREMENT_WINDOW
        self._blink_times = [t for t in self._blink_times if t > cutoff]

        # ── Rolling blink rate (blinks per minute) ────────────────────────────────
        elapsed = min(now - (self._blink_times[0] if self._blink_times else now),
                      self.MEASUREMENT_WINDOW)
        if elapsed > 5:   # Only compute once we have 5 s of data
            self.blink_rate = (len(self._blink_times) / elapsed) * 60.0
        else:
            self.blink_rate = 20.0  # Assume healthy until enough data

        # ── Alert logic ───────────────────────────────────────────────────────────
        self.alert_needed = self.blink_rate < self.BLINK_RATE_THRESHOLD

        # ── Draw eye landmarks and EAR ────────────────────────────────────────────
        self._draw_eyes(frame, [px(i) for i in self._LEFT_EYE],
                                [px(i) for i in self._RIGHT_EYE])

        return frame, self.ear, self.blink_rate, self.alert_needed

    # ── Private helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _eye_aspect_ratio(eye_pts: list) -> float:
        """
        Compute Eye Aspect Ratio.
        eye_pts: list of 6 (x,y) numpy arrays [p1..p6].
        """
        # Vertical distances
        A = calculate_distance(eye_pts[1], eye_pts[5])
        B = calculate_distance(eye_pts[2], eye_pts[4])
        # Horizontal distance
        C = calculate_distance(eye_pts[0], eye_pts[3])
        return (A + B) / (2.0 * C + 1e-6)

    @staticmethod
    def _draw_eyes(frame, left_pts, right_pts) -> None:
        """Draw eye contour polygons on the frame."""
        for pts in (left_pts, right_pts):
            hull = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [hull], isClosed=True,
                          color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
