"""
posture_detector.py
-------------------
Uses MediaPipe Pose to extract skeletal landmarks and calculate
posture quality based on shoulder alignment and spine angles.

Detects:
    1. Slouching     - spine angle too low (hunched forward)
    2. Sulking       - head dropped down, chin toward chest
    3. Uneven shoulders
    4. Forward head  - head jutting forward
    5. Lateral lean  - leaning to one side
"""

import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle, calculate_distance


class PostureDetector:

    # ── Thresholds (tightened for real detection) ──────────────────────────────

    # Shoulder level: vertical difference between left/right shoulder Y (normalised)
    SHOULDER_LEVEL_THRESHOLD = 0.05    # > 0.05 = uneven shoulders

    # Forward head: horizontal distance nose vs shoulder midpoint (normalised)
    FORWARD_HEAD_THRESHOLD   = 0.09    # > 0.09 = head jutting forward

    # Spine angle: ear → shoulder → hip angle in degrees
    # Upright = ~170-180°, slouching brings this DOWN
    SPINE_ANGLE_THRESHOLD    = 155     # < 155° = slouching / hunching

    # Lateral lean: shoulder midpoint vs hip midpoint horizontal offset
    SHOULDER_HIP_THRESHOLD   = 0.08   # > 0.08 = leaning sideways

    # Sulking / head drop: nose Y vs shoulder Y (normalised)
    # When head drops, nose Y increases toward shoulders
    # Ratio = (nose_y - shoulder_mid_y): negative = head up (normal)
    # Closer to 0 or positive = head dropped toward chest
    HEAD_DROP_THRESHOLD      = -0.12  # > -0.12 = head drooping / sulking

    # Neck angle: nose → shoulder midpoint vertical drop
    # Measures how far the neck is bent forward
    NECK_FORWARD_THRESHOLD   = 0.18   # > 0.18 = neck bent forward

    # How many issues needed to mark as Bad (1 = very strict, 2 = moderate)
    MIN_ISSUES_FOR_BAD = 1

    def __init__(self):
        self.mp_pose           = mp.solutions.pose
        self.mp_drawing        = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode        = False,
            model_complexity         = 1,
            smooth_landmarks         = True,
            min_detection_confidence = 0.6,
            min_tracking_confidence  = 0.6,
        )

        self.posture_status = "Good"
        self.posture_issues = []
        self.spine_angle    = 180.0

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, str, list]:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        self.posture_issues = []

        if results.pose_landmarks:
            self._draw_skeleton(frame, results)
            self._analyse_posture(results.pose_landmarks.landmark, frame.shape, frame)

        self.posture_status = (
            "Bad" if len(self.posture_issues) >= self.MIN_ISSUES_FOR_BAD else "Good"
        )
        return frame, self.posture_status, self.posture_issues

    # ── Private helpers ───────────────────────────────────────────────────────

    def _draw_skeleton(self, frame: np.ndarray, results) -> None:
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

    def _analyse_posture(self, landmarks, shape: tuple, frame: np.ndarray) -> None:
        h, w = shape[:2]
        lm   = landmarks

        def get(landmark_enum):
            pt = lm[landmark_enum.value]
            return np.array([pt.x, pt.y])

        # ── Key landmark positions (normalised 0–1) ───────────────────────────
        nose       = get(self.mp_pose.PoseLandmark.NOSE)
        l_shoulder = get(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        r_shoulder = get(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        l_hip      = get(self.mp_pose.PoseLandmark.LEFT_HIP)
        r_hip      = get(self.mp_pose.PoseLandmark.RIGHT_HIP)
        l_ear      = get(self.mp_pose.PoseLandmark.LEFT_EAR)
        r_ear      = get(self.mp_pose.PoseLandmark.RIGHT_EAR)

        shoulder_mid = (l_shoulder + r_shoulder) / 2
        hip_mid      = (l_hip      + r_hip)      / 2
        ear_mid      = (l_ear      + r_ear)      / 2

        # ── 1. SLOUCHING — spine angle ear→shoulder→hip ───────────────────────
        # When you slouch forward, this angle collapses below threshold
        self.spine_angle = calculate_angle(l_ear, l_shoulder, l_hip)
        if self.spine_angle < self.SPINE_ANGLE_THRESHOLD:
            self.posture_issues.append(
                f"Slouching — spine angle {self.spine_angle:.0f}° (need >{self.SPINE_ANGLE_THRESHOLD}°)"
            )

        # ── 2. SULKING / HEAD DROP — nose drops toward shoulder level ─────────
        # nose_y increases downward in image coords; shoulders are below face
        # head_drop_ratio = how close nose Y is to shoulder Y
        # Normal: nose is well ABOVE shoulders → large negative value
        # Sulking: nose drops toward chest → ratio approaches 0 or goes positive
        head_drop_ratio = nose[1] - shoulder_mid[1]   # normalised
        if head_drop_ratio > self.HEAD_DROP_THRESHOLD:
            self.posture_issues.append(
                f"Head dropped / sulking — chin toward chest"
            )

        # ── 3. FORWARD HEAD / NECK BEND ───────────────────────────────────────
        # Ear midpoint should be roughly above shoulder midpoint
        # If ear is significantly in FRONT of (horizontal offset from) shoulder → forward head
        neck_forward = abs(ear_mid[0] - shoulder_mid[0])
        if neck_forward > self.NECK_FORWARD_THRESHOLD:
            self.posture_issues.append("Neck bent forward — head jutting out")

        # ── 4. FORWARD HEAD (nose offset) ─────────────────────────────────────
        head_offset = abs(nose[0] - shoulder_mid[0])
        if head_offset > self.FORWARD_HEAD_THRESHOLD:
            self.posture_issues.append("Forward head posture")

        # ── 5. UNEVEN SHOULDERS ───────────────────────────────────────────────
        shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
        if shoulder_diff > self.SHOULDER_LEVEL_THRESHOLD:
            self.posture_issues.append("Uneven shoulders")

        # ── 6. LATERAL LEAN ───────────────────────────────────────────────────
        lateral_offset = abs(shoulder_mid[0] - hip_mid[0])
        if lateral_offset > self.SHOULDER_HIP_THRESHOLD:
            self.posture_issues.append("Leaning to one side")

        # ── Draw visual indicators on frame ───────────────────────────────────
        self._draw_posture_indicators(
            frame, l_ear, l_shoulder, l_hip,
            shoulder_diff, head_offset, head_drop_ratio, neck_forward
        )

    def _draw_posture_indicators(self, frame, l_ear, l_shoulder, l_hip,
                                  shoulder_diff, head_offset,
                                  head_drop_ratio, neck_forward) -> None:
        """
        Draw spine line and colour-coded status indicators directly on the frame.
        GREEN = within threshold, RED = exceeded threshold.
        """
        h, w = frame.shape[:2]

        # Convert normalised → pixel coords
        def to_px(pt):
            return (int(pt[0] * w), int(pt[1] * h))

        ear_px  = to_px(l_ear)
        sho_px  = to_px(l_shoulder)
        hip_px  = to_px(l_hip)

        # Draw spine line ear→shoulder→hip with colour based on angle
        spine_col = (0, 200, 80) if self.spine_angle >= self.SPINE_ANGLE_THRESHOLD else (0, 60, 220)
        cv2.line(frame, ear_px, sho_px, spine_col, 3, cv2.LINE_AA)
        cv2.line(frame, sho_px, hip_px, spine_col, 3, cv2.LINE_AA)
        cv2.circle(frame, ear_px, 6, spine_col, -1, cv2.LINE_AA)
        cv2.circle(frame, sho_px, 6, spine_col, -1, cv2.LINE_AA)
        cv2.circle(frame, hip_px, 6, spine_col, -1, cv2.LINE_AA)

        # Metrics readout — bottom left above stress panel
        metrics = [
            (f"Spine angle  : {self.spine_angle:.1f} deg",
             self.spine_angle >= self.SPINE_ANGLE_THRESHOLD),

            (f"Head drop    : {head_drop_ratio:.3f}",
             head_drop_ratio <= self.HEAD_DROP_THRESHOLD),

            (f"Neck forward : {neck_forward:.3f}",
             neck_forward <= self.NECK_FORWARD_THRESHOLD),

            (f"Shoulder diff: {shoulder_diff:.3f}",
             shoulder_diff <= self.SHOULDER_LEVEL_THRESHOLD),
        ]

        base_y = h - 230
        for i, (text, is_good) in enumerate(metrics):
            colour = (0, 200, 80) if is_good else (0, 60, 220)
            cv2.putText(frame, text, (10, base_y + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, colour, 1, cv2.LINE_AA)