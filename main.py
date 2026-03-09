"""
main.py
-------
Real-Time Stress & Posture Detection System
Runs silently in the background — no visible OpenCV window.
Notifications appear as native Windows toast alerts (top-right corner).

Usage:
    python main.py [OPTIONS]

Options:
    --camera   INT   Webcam index (default: 0)
    --width    INT   Frame width  (default: 640)   ← lower res for background use
    --height   INT   Frame height (default: 480)
    --fps      INT   Target FPS   (default: 15)    ← lower fps = less CPU usage
    --no-voice       Disable TTS voice alerts
    --no-tray        Skip system tray icon (useful on headless / CI)

Controls (system tray right-click menu):
    Toggle Voice  – enable / disable voice alerts
    Screenshot    – save a debug frame to ./screenshots/
    Quit          – stop the background monitor
"""

import argparse
import os
import sys
import time
import threading
import cv2
import mediapipe as mp

from posture_detector import PostureDetector
from stress_detector  import StressDetector
from blink_detector   import BlinkDetector
from alert_system     import AlertSystem


# ── Shared state (read by tray, written by detection loop) ────────────────────
_state = {
    "running":        True,
    "voice_on":       True,
    "save_screenshot": False,
    "posture":        "Good",
    "stress":         "Low",
    "blink_rate":     20.0,
}


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Background Stress & Posture Monitor")
    p.add_argument("--camera",   type=int,  default=0,   help="Webcam index")
    p.add_argument("--width",    type=int,  default=640, help="Frame width")
    p.add_argument("--height",   type=int,  default=480, help="Frame height")
    p.add_argument("--fps",      type=int,  default=15,  help="Target FPS (lower = less CPU)")
    p.add_argument("--no-voice", action="store_true",    help="Disable voice alerts")
    p.add_argument("--no-tray",  action="store_true",    help="No system tray icon")
    return p.parse_args()


# ── Detection loop (runs on background thread) ────────────────────────────────
def detection_loop(args, alert_sys):
    """
    Core CV loop — runs entirely headless (no imshow).
    Captures frames, runs all three detectors, fires alerts.
    """
    global _state

    print("[System] Initialising detectors …")
    posture_det = PostureDetector()
    stress_det  = StressDetector()
    blink_det   = BlinkDetector()

    # Shared FaceMesh for blink detector
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode        = False,
        max_num_faces            = 1,
        refine_landmarks         = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )

    print(f"[System] Opening camera {args.camera} at {args.width}×{args.height} …")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"[Error] Cannot open camera {args.camera}")
        _state["running"] = False
        return

    # Posture smoothing counters
    BAD_POSTURE_FRAMES_NEEDED = 15
    bad_posture_counter  = 0
    good_posture_counter = 0

    frame_delay = 1.0 / max(args.fps, 1)
    os.makedirs("screenshots", exist_ok=True)

    print("[System] Background monitor running.  Use the tray icon to control it.\n")

    try:
        while _state["running"]:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)

            # ── Shared FaceMesh inference ──────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb)
            face_lm = (
                face_results.multi_face_landmarks[0]
                if face_results and face_results.multi_face_landmarks
                else None
            )

            # ── 1. Posture ─────────────────────────────────────────────────────
            frame, posture_status, posture_issues = posture_det.process(frame)

            if posture_status == "Bad":
                bad_posture_counter  = min(bad_posture_counter + 1,
                                           BAD_POSTURE_FRAMES_NEEDED + 10)
                good_posture_counter = 0
            else:
                good_posture_counter = min(good_posture_counter + 1,
                                           BAD_POSTURE_FRAMES_NEEDED + 10)
                bad_posture_counter  = max(bad_posture_counter - 1, 0)

            if bad_posture_counter >= BAD_POSTURE_FRAMES_NEEDED:
                smoothed_posture = "Bad"
            elif good_posture_counter >= BAD_POSTURE_FRAMES_NEEDED:
                smoothed_posture = "Good"
            else:
                smoothed_posture = posture_status

            # ── 2. Stress ──────────────────────────────────────────────────────
            frame, stress_level, stress_indicators = stress_det.process(frame)

            # ── 3. Blink ───────────────────────────────────────────────────────
            frame, ear, blink_rate, blink_alert = blink_det.process(frame, face_lm)

            # ── Update shared state ────────────────────────────────────────────
            _state["posture"]    = smoothed_posture
            _state["stress"]     = stress_level
            _state["blink_rate"] = blink_rate

            # ── 4. Alerts ──────────────────────────────────────────────────────
            _handle_alerts(
                alert_sys,
                smoothed_posture, posture_issues,
                stress_level, stress_det.emotion_label, stress_indicators,
                blink_alert, blink_rate,
            )

            # ── Screenshot request from tray ───────────────────────────────────
            if _state["save_screenshot"]:
                _state["save_screenshot"] = False
                ts   = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join("screenshots", f"capture_{ts}.png")
                cv2.imwrite(path, frame)
                print(f"[System] Screenshot saved → {path}")
                alert_sys.notify(
                    title="Screenshot saved",
                    message=f"Saved to screenshots/{os.path.basename(path)}",
                    category="info",
                )

            # ── Throttle to target FPS ─────────────────────────────────────────
            elapsed = time.time() - t_start
            sleep_t = frame_delay - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except Exception as exc:
        print(f"[Error] Detection loop crashed: {exc}")
        import traceback; traceback.print_exc()

    finally:
        alert_sys.shutdown()
        cap.release()
        face_mesh.close()
        print("[System] Camera released.  Goodbye!")


# ── Alert dispatch ────────────────────────────────────────────────────────────
def _handle_alerts(alert_sys, posture_status, posture_issues,
                   stress_level, emotion_label, stress_indicators,
                   blink_alert, blink_rate):

    # ── Posture ───────────────────────────────────────────────────────────────
    if posture_status == "Bad":
        issues_str = " | ".join(posture_issues) if posture_issues else "Check alignment"
        alert_sys.notify(
            title       = "⚠️ Posture Alert",
            message     = f"Sit straight — {issues_str}",
            category    = "posture",
            speech_text = "Please sit straight and correct your posture.",
        )
    else:
        alert_sys.notify(
            title       = "✅ Good Posture",
            message     = "Great alignment — keep it up!",
            category    = "good",
            speech_text = "",
        )

    # ── Stress / Anger ────────────────────────────────────────────────────────
    stress_lines = any(
        "STRESS LINES" in ind for ind in (stress_indicators or [])
    )

    if stress_lines:
        alert_sys.notify(
            title       = "😤 Stress Detected",
            message     = "Stress lines between brows — you look very tense. Take a break!",
            category    = "stress",
            speech_text = "Stress lines detected. You look very tense. Please relax.",
        )
    elif emotion_label == "Angry" or stress_level == "High":
        alert_sys.notify(
            title       = "😠 High Stress",
            message     = "You look tense — take a deep breath and a short break.",
            category    = "stress",
            speech_text = "You look angry or frustrated. Please take a deep breath.",
        )

    # ── Blink ─────────────────────────────────────────────────────────────────
    if blink_alert:
        alert_sys.notify(
            title       = "👁️ Blink Reminder",
            message     = f"Blink rate low ({blink_rate:.0f}/min) — remember to blink!",
            category    = "blink",
            speech_text = "Your blink rate is too low. Please blink your eyes.",
        )


# ── System tray icon ──────────────────────────────────────────────────────────
def _start_tray(args):
    """
    Create a system tray icon with a right-click context menu.
    Runs on the main thread (required by most tray libraries on Windows).
    Falls back gracefully if pystray / Pillow are not installed.
    """
    try:
        import pystray
        from PIL import Image, ImageDraw

        # Build a simple coloured icon (green circle on dark bg)
        size   = 64
        img    = Image.new("RGB", (size, size), color=(30, 30, 30))
        draw   = ImageDraw.Draw(img)
        draw.ellipse([8, 8, size - 8, size - 8], fill=(0, 200, 80))

        def toggle_voice(icon, item):
            _state["voice_on"] = not _state["voice_on"]
            print(f"[Tray] Voice {'ON' if _state['voice_on'] else 'OFF'}")

        def take_screenshot(icon, item):
            _state["save_screenshot"] = True

        def quit_app(icon, item):
            _state["running"] = False
            icon.stop()

        def voice_label(item):
            return "🔇 Disable voice" if _state["voice_on"] else "🔊 Enable voice"

        menu = pystray.Menu(
            pystray.MenuItem("Stress & Posture Monitor", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(voice_label, toggle_voice),
            pystray.MenuItem("📷 Screenshot", take_screenshot),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("❌ Quit", quit_app),
        )

        icon = pystray.Icon("StressMonitor", img, "Stress & Posture Monitor", menu)
        print("[Tray] System tray icon active — right-click to control.")
        icon.run()   # blocks until quit

    except ImportError:
        print("[Tray] pystray/Pillow not installed — no tray icon.")
        print("       Install with:  pip install pystray pillow")
        print("       Press Ctrl+C to stop the monitor.\n")
        # Just keep the main thread alive
        try:
            while _state["running"]:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[System] Interrupted.")
            _state["running"] = False


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    _state["voice_on"] = not args.no_voice

    alert_sys = AlertSystem(voice_enabled=_state["voice_on"])

    # Detection loop on a background daemon thread
    loop_thread = threading.Thread(
        target=detection_loop,
        args=(args, alert_sys),
        daemon=True,
        name="DetectionLoop",
    )
    loop_thread.start()

    # System tray blocks the main thread (or falls back to a sleep loop)
    if not args.no_tray:
        _start_tray(args)
    else:
        print("[System] Running without tray icon.  Press Ctrl+C to stop.")
        try:
            while _state["running"]:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[System] Interrupted.")
            _state["running"] = False

    loop_thread.join(timeout=3)
    print("[System] Shut down complete.")


if __name__ == "__main__":
    main()