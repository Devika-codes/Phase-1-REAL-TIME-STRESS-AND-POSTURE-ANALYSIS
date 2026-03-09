"""
stress_detector.py
------------------
Emotion-aware stress detection using MediaPipe FaceMesh (468 landmarks).

EMOTION → STRESS MAPPING:
    Angry expression → High stress   (furrowed brows, stress lines, nose flare,
                                       tight lips, glabella compression)
    Neutral / Happy  → Low stress

ANGRY signals (each 0-1, weighted and summed into angry_score):
    A1. Furrowed brows         – inner brows descend toward nose      weight 2.0
    A2. Stress lines (glabella)– brow squeeze + vertical compression  weight 3.0
    A3. Brow-eye gap           – brows press toward eyes              weight 1.5
    A4. Tight / pressed lips   – lip distance shrinks                 weight 1.0
    A5. Nose snarl             – upper lip rises toward nose          weight 0.8
    A6. Wide staring eyes      – upper lid retraction                 weight 0.5

CLASSIFICATION:
    angry_score >= ANGRY_THRESHOLD  → High stress  (Angry)
    else                            → Low stress   (Neutral)

Auto-calibration to the user's neutral face over first 60 frames.
"""

import cv2
import numpy as np
import mediapipe as mp
from utils import calculate_distance


# ── Colour constants (BGR) ─────────────────────────────────────────────────────
_GREEN  = (0,   210,  80)
_YELLOW = (0,   210, 255)
_RED    = (0,    60, 220)
_WHITE  = (240, 240, 240)
_GREY   = (160, 160, 160)


class StressDetector:
    """
    Detects stress via anger recognition from facial geometry.
    ANGRY face → High stress
    Neutral    → Low stress
    """

    # ── Classification threshold ───────────────────────────────────────────────
    ANGRY_THRESHOLD = 1.5   # angry_score must reach this for High stress

    # ── ANGRY signal thresholds ────────────────────────────────────────────────
    # A1: Furrowed brows — inner brow Y descends toward nose (norm by face_h)
    ANGRY_BROW_FURROW_THRESHOLD = 0.295

    # A2: Stress lines — glabella height (norm by face_h); smaller = squashed
    ANGRY_GLABELLA_H_THRESHOLD  = 0.110

    # A2b: Glabella width (norm by face_w); smaller = brows squeezed together
    ANGRY_GLABELLA_W_THRESHOLD  = 0.072

    # A3: Brow-eye gap (norm by face_h); smaller = brows glaring toward eyes
    ANGRY_BROW_EYE_THRESHOLD    = 0.068

    # A4: Lip gap (norm by face_h); smaller = lips pressed together in anger
    ANGRY_LIP_PRESS_THRESHOLD   = 0.012

    # A5: Upper lip to nose (norm by face_h); smaller = snarl / nose raise
    ANGRY_NOSE_RAISE_THRESHOLD  = 0.095

    # A6: Eye openness (norm by face_h); larger = wide angry stare
    ANGRY_EAR_WIDE_THRESHOLD    = 0.040

    # Frames stress lines must be sustained before confirming
    STRESS_LINE_SUSTAIN_FRAMES  = 8

    # Temporal smoothing window (frames)
    HISTORY_LEN = 40

    def __init__(self):
        self.mp_face_mesh      = mp.solutions.face_mesh
        self.mp_drawing        = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode        = False,
            max_num_faces            = 1,
            refine_landmarks         = True,
            min_detection_confidence = 0.55,
            min_tracking_confidence  = 0.55,
        )

        # Rolling score history for temporal smoothing
        self._angry_history: list[float] = []

        # Auto-calibration state
        self._calibration_frames  = 0
        self._calib_brow_furrow   = []
        self._calib_inter_brow    = []
        self._calib_glabella_h    = []
        self._calib_glabella_w    = []
        self._baseline_furrow     = None
        self._baseline_inter_brow = None
        self._baseline_glabella_h = None
        self._baseline_glabella_w = None

        # Stress lines sustain counter
        self._stress_line_counter = 0
        self.stress_lines_active  = False

        # Exposed state
        self.stress_level  = "Low"
        self.emotion_label = "Neutral"   # "Neutral" | "Angry"
        self.angry_score   = 0.0
        self.stress_score  = 0.0
        self.indicators: list[tuple[str, float]] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray):
        """
        Run anger-based stress detection on a BGR frame.
        Returns: (annotated_frame, stress_level, indicator_strings)
        """
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        self.indicators = []
        raw_angry = 0.0

        if results.multi_face_landmarks:
            face_lm = results.multi_face_landmarks[0]
            self._draw_mesh(frame, face_lm)
            raw_angry = self._analyse(face_lm.landmark, frame)

        # ── Temporal smoothing (75th percentile over last 40 frames) ──────────
        self._angry_history.append(raw_angry)
        if len(self._angry_history) > self.HISTORY_LEN:
            self._angry_history.pop(0)

        if len(self._angry_history) >= 10:
            self.angry_score = float(np.percentile(self._angry_history, 75))
        else:
            self.angry_score = float(np.mean(self._angry_history)) if self._angry_history else 0.0

        self.stress_score = self.angry_score

        # ── Classification ────────────────────────────────────────────────────
        if self.angry_score >= self.ANGRY_THRESHOLD:
            self.emotion_label = "Angry"
            self.stress_level  = "High"
        else:
            self.emotion_label = "Neutral"
            self.stress_level  = "Low"

        self._draw_hud(frame)
        return frame, self.stress_level, [i[0] for i in self.indicators]

    # ── Private: analysis ──────────────────────────────────────────────────────

    def _analyse(self, lm, frame: np.ndarray) -> float:
        """Compute raw angry_score from facial landmarks. Returns float score."""
        h, w = frame.shape[:2]

        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        face_h = max(calculate_distance(px(10), px(152)), 1.0)
        face_w = max(calculate_distance(px(234), px(454)), 1.0)

        self._run_calibration(lm, px, face_h, face_w)

        # Use personalised baselines if calibration is done, else use defaults
        furrow_thresh     = self._baseline_furrow     or self.ANGRY_BROW_FURROW_THRESHOLD
        glabella_h_thresh = self._baseline_glabella_h or self.ANGRY_GLABELLA_H_THRESHOLD
        glabella_w_thresh = self._baseline_glabella_w or self.ANGRY_GLABELLA_W_THRESHOLD

        angry_score = 0.0

        forehead_y     = px(10)[1]
        l_inner_brow_y = px(107)[1]
        r_inner_brow_y = px(336)[1]
        avg_furrow     = ((l_inner_brow_y + r_inner_brow_y) / 2 - forehead_y) / face_h

        # ── A1: Furrowed brows (weight 2.0) ───────────────────────────────────
        # Inner brow corners drop toward nose when angry — corrugator supercilii
        if avg_furrow > furrow_thresh:
            conf = min((avg_furrow - furrow_thresh) / 0.04, 1.0)
            angry_score += 2.0 * conf
            self.indicators.append((f"Angry: furrowed brows ({conf*100:.0f}%)", conf))
            self._highlight_brows(frame, px, _RED)
        else:
            self._highlight_brows(frame, px, _GREEN)

        # ── A2: Stress lines / glabella compression (weight 3.0) ──────────────
        # Procerus squashes glabella vertically + corrugator pinches it horizontally
        # Together they create the "11 lines" stress crease between the brows
        glabella_h = calculate_distance(px(9), px(6))   / face_h
        glabella_w = calculate_distance(px(55), px(285)) / face_w

        glabella_compressed = glabella_h < glabella_h_thresh
        glabella_pinched    = glabella_w < glabella_w_thresh
        brows_furrowed      = avg_furrow  > furrow_thresh

        # All three must be true simultaneously
        if brows_furrowed and glabella_compressed and glabella_pinched:
            self._stress_line_counter = min(
                self._stress_line_counter + 1, self.STRESS_LINE_SUSTAIN_FRAMES + 5)
        else:
            self._stress_line_counter = max(self._stress_line_counter - 1, 0)

        self.stress_lines_active = (
            self._stress_line_counter >= self.STRESS_LINE_SUSTAIN_FRAMES)

        if self.stress_lines_active:
            compress_conf = min((glabella_h_thresh - glabella_h) / 0.02, 1.0)
            pinch_conf    = min((glabella_w_thresh  - glabella_w) / 0.02, 1.0)
            conf          = (compress_conf + pinch_conf) / 2
            angry_score  += 3.0 * conf
            self.indicators.append((f"Angry: STRESS LINES between brows ({conf*100:.0f}%)", conf))
            self._draw_stress_line_marker(frame, px(9), px(6), px(55), px(285), conf)
        else:
            self._draw_stress_line_marker(frame, px(9), px(6), px(55), px(285), 0.0)

        # ── A3: Brow-eye gap compression (weight 1.5) ─────────────────────────
        # Brows pressing down toward eyelids creates the glaring angry look
        l_gap   = (px(159)[1] - px(107)[1]) / face_h
        r_gap   = (px(386)[1] - px(336)[1]) / face_h
        avg_gap = (l_gap + r_gap) / 2

        if avg_gap < self.ANGRY_BROW_EYE_THRESHOLD:
            conf = min((self.ANGRY_BROW_EYE_THRESHOLD - avg_gap) / 0.02, 1.0)
            angry_score += 1.5 * conf
            self.indicators.append((f"Angry: brows pressing toward eyes ({conf*100:.0f}%)", conf))

        # ── A4: Tight / pressed lips (weight 1.0) ─────────────────────────────
        # Orbicularis oris compresses lips together in suppressed anger
        lip_gap = calculate_distance(px(13), px(14)) / face_h
        if lip_gap < self.ANGRY_LIP_PRESS_THRESHOLD:
            conf = min((self.ANGRY_LIP_PRESS_THRESHOLD - lip_gap) / 0.008, 1.0)
            angry_score += 1.0 * conf
            self.indicators.append((f"Angry: tight / pressed lips ({conf*100:.0f}%)", conf))

        # ── A5: Nose snarl / upper lip raise (weight 0.8) ─────────────────────
        # Levator labii superioris pulls upper lip toward nose — classic snarl
        nose_ul_dist = calculate_distance(px(0), px(4)) / face_h
        if nose_ul_dist < self.ANGRY_NOSE_RAISE_THRESHOLD:
            conf = min((self.ANGRY_NOSE_RAISE_THRESHOLD - nose_ul_dist) / 0.015, 1.0)
            angry_score += 0.8 * conf
            self.indicators.append((f"Angry: nose snarl / upper lip raise ({conf*100:.0f}%)", conf))

        # ── A6: Wide staring eyes (weight 0.5) ────────────────────────────────
        # Levator palpebrae retracts upper lid — eyes open wider than normal
        l_ear   = calculate_distance(px(159), px(145)) / face_h
        r_ear   = calculate_distance(px(386), px(374)) / face_h
        avg_ear = (l_ear + r_ear) / 2

        if avg_ear > self.ANGRY_EAR_WIDE_THRESHOLD:
            conf = min((avg_ear - self.ANGRY_EAR_WIDE_THRESHOLD) / 0.01, 1.0)
            angry_score += 0.5 * conf
            self.indicators.append((f"Angry: wide staring eyes ({conf*100:.0f}%)", conf))

        return angry_score

    # ── Private: calibration ──────────────────────────────────────────────────

    def _run_calibration(self, lm, px, face_h, face_w) -> None:
        """Collect first 60 frames at neutral expression to personalise thresholds."""
        CALIB_FRAMES = 60
        if self._calibration_frames >= CALIB_FRAMES:
            return

        forehead_y = px(10)[1]
        avg_furrow = ((px(107)[1] + px(336)[1]) / 2 - forehead_y) / face_h
        inter_brow = calculate_distance(px(107), px(336)) / face_w
        glabella_h = calculate_distance(px(9),   px(6))   / face_h
        glabella_w = calculate_distance(px(55),  px(285))  / face_w

        self._calib_brow_furrow.append(avg_furrow)
        self._calib_inter_brow.append(inter_brow)
        self._calib_glabella_h.append(glabella_h)
        self._calib_glabella_w.append(glabella_w)
        self._calibration_frames += 1

        if self._calibration_frames == CALIB_FRAMES:
            self._baseline_furrow     = float(np.mean(self._calib_brow_furrow)) + 0.025
            self._baseline_inter_brow = float(np.mean(self._calib_inter_brow))  - 0.020
            self._baseline_glabella_h = float(np.mean(self._calib_glabella_h))  - 0.012
            self._baseline_glabella_w = float(np.mean(self._calib_glabella_w))  - 0.018
            print(f"[StressDetector] Calibration complete  "
                  f"furrow={self._baseline_furrow:.3f}  "
                  f"glabella_h={self._baseline_glabella_h:.3f}  "
                  f"glabella_w={self._baseline_glabella_w:.3f}")

    # ── Private: drawing helpers ──────────────────────────────────────────────

    def _draw_mesh(self, frame: np.ndarray, face_landmarks) -> None:
        self.mp_drawing.draw_landmarks(
            image                   = frame,
            landmark_list           = face_landmarks,
            connections             = self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec   = None,
            connection_drawing_spec = self.mp_drawing_styles
                                          .get_default_face_mesh_contours_style(),
        )

    @staticmethod
    def _highlight_brows(frame: np.ndarray, px, colour: tuple) -> None:
        """Colour-coded dots and line across inner brow landmarks."""
        for idx in (107, 66, 336, 296):
            pt = px(idx).astype(int)
            cv2.circle(frame, tuple(pt), 5, colour, -1, cv2.LINE_AA)
        cv2.line(frame,
                 tuple(px(107).astype(int)),
                 tuple(px(336).astype(int)),
                 colour, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_stress_line_marker(frame, g_top, g_bottom, g_left, g_right,
                                  confidence: float) -> None:
        """Draw the '11 lines' marker in the glabella zone between the brows."""
        cx = int((g_left[0]  + g_right[0]) / 2)
        cy = int((g_top[1]   + g_bottom[1]) / 2)

        if confidence > 0:
            r_val  = int(255 * confidence)
            g_val  = int(140 * (1 - confidence))
            colour = (0, g_val, r_val)
            size   = int(10 + confidence * 8)
            offset = max(4, int(confidence * 7))
            # Two vertical lines mimicking the "11" stress crease
            for x_off in (-offset, offset):
                cv2.line(frame, (cx + x_off, cy - size),
                                (cx + x_off, cy + size), colour, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), size + 4, colour, 1, cv2.LINE_AA)
            label = "STRESS LINES" if confidence > 0.5 else "Brow tension"
            cv2.putText(frame, label, (cx - 55, cy - size - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), 4, (80, 80, 80), 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Draw the stress analysis HUD panel in the bottom-left corner."""
        h, w  = frame.shape[:2]
        px_   = 10
        py_   = h - 185
        pw    = 440
        ph    = 175

        # Semi-transparent dark background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (px_, py_), (px_ + pw, py_ + ph), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        level_colour  = _RED   if self.stress_level == "High" else _GREEN
        emotion_colour= _RED   if self.emotion_label == "Angry" else _GREEN

        # ── Title ─────────────────────────────────────────────────────────────
        cv2.putText(frame, "STRESS MONITOR",
                    (px_ + 8, py_ + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, _GREY, 1, cv2.LINE_AA)

        # ── Emotion + stress badges ────────────────────────────────────────────
        emoji = ":-@" if self.emotion_label == "Angry" else ":-|"
        cv2.putText(frame,
                    f"{self.emotion_label.upper()}  {emoji}",
                    (px_ + 8, py_ + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, emotion_colour, 2, cv2.LINE_AA)

        cv2.putText(frame, f"Stress: {self.stress_level}",
                    (px_ + 270, py_ + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, level_colour, 2, cv2.LINE_AA)

        # ── Angry score bar ────────────────────────────────────────────────────
        bar_x = px_ + 70
        bar_w = pw - 120
        max_s = 5.0

        cv2.putText(frame, "ANGRY",
                    (px_ + 8, py_ + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, _RED, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x, py_ + 62),
                      (bar_x + bar_w, py_ + 78), (50, 50, 50), -1)
        fill = int(min(self.angry_score / max_s, 1.0) * bar_w)
        if fill > 0:
            cv2.rectangle(frame, (bar_x, py_ + 62),
                          (bar_x + fill, py_ + 78), _RED, -1)
        cv2.rectangle(frame, (bar_x, py_ + 62),
                      (bar_x + bar_w, py_ + 78), _GREY, 1)
        # Threshold marker line
        tx = bar_x + int(self.ANGRY_THRESHOLD / max_s * bar_w)
        cv2.line(frame, (tx, py_ + 59), (tx, py_ + 81), _WHITE, 1)
        cv2.putText(frame, f"{self.angry_score:.1f}",
                    (bar_x + bar_w + 6, py_ + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GREY, 1, cv2.LINE_AA)

        # ── Active signals list ────────────────────────────────────────────────
        cv2.putText(frame, "Signals detected:",
                    (px_ + 8, py_ + 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GREY, 1, cv2.LINE_AA)

        sorted_ind = sorted(self.indicators, key=lambda x: x[1], reverse=True)
        if sorted_ind:
            for i, (label, conf) in enumerate(sorted_ind[:3]):
                y_pos = py_ + 114 + i * 20
                dot_c = _RED if conf > 0.6 else _YELLOW
                cv2.circle(frame, (px_ + 14, y_pos - 4), 4, dot_c, -1)
                cv2.putText(frame, label[:60], (px_ + 25, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, _WHITE, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "None — face looks relaxed",
                        (px_ + 25, py_ + 114),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, _GREEN, 1, cv2.LINE_AA)

        # ── Stress lines status ────────────────────────────────────────────────
        sl_text   = "Stress Lines: ACTIVE" if self.stress_lines_active else "Stress Lines: -"
        sl_colour = _RED if self.stress_lines_active else _GREY
        cv2.putText(frame, sl_text,
                    (px_ + 8, py_ + ph - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, sl_colour, 1, cv2.LINE_AA)

        # ── Calibration progress ───────────────────────────────────────────────
        if self._calibration_frames < 60:
            pct = int(self._calibration_frames / 60 * 100)
            cv2.putText(frame,
                        f"Calibrating... {pct}%  (keep neutral face)",
                        (px_ + 180, py_ + ph - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, _YELLOW, 1, cv2.LINE_AA)