"""
alert_system.py
---------------
Fires native OS toast notifications (top-right corner of screen).
No OpenCV overlay needed — the app runs fully in the background.

Notification backend priority (auto-detected at startup):
    1. winotify     – Windows 10/11 toast (best, actionable)
    2. plyer        – cross-platform wrapper (Windows / macOS / Linux)
    3. win10toast   – Windows fallback
    4. subprocess   – PowerShell toast on Windows (zero-install fallback)
    5. print()      – last resort (always works, just prints to console)

Install recommended backends:
    pip install winotify          # Windows 10/11  ← recommended
    pip install plyer             # cross-platform alternative
    pip install pyttsx3           # voice alerts

Alert categories and cooldowns:
    posture  – 10 s
    stress   – 20 s
    blink    – 15 s
    good     – 30 s
    info     –  5 s  (screenshots, status messages)
"""

import time
import threading
import queue
import subprocess
import sys


# ── Detect best available notification backend ────────────────────────────────

def _detect_backend():
    """Return a callable notify(title, message) using the best available backend."""

    # ── 1. winotify (Windows 10 / 11 native toasts) ───────────────────────────
    try:
        from winotify import Notification, audio

        def _winotify(title: str, message: str, icon_path: str = ""):
            toast = Notification(
                app_id   = "Stress & Posture Monitor",
                title    = title,
                msg      = message,
                duration = "short",          # "short" = 5 s, "long" = 25 s
                icon     = icon_path or "",
            )
            toast.set_audio(audio.Default, loop=False)
            toast.show()

        print("[AlertSystem] Using winotify (Windows native toasts).")
        return _winotify, "winotify"

    except ImportError:
        pass

    # ── 2. plyer (cross-platform) ─────────────────────────────────────────────
    try:
        from plyer import notification as _plyer_notif

        def _plyer(title: str, message: str, icon_path: str = ""):
            _plyer_notif.notify(
                title       = title,
                message     = message,
                app_name    = "Stress & Posture Monitor",
                app_icon    = icon_path or "",
                timeout     = 5,
            )

        print("[AlertSystem] Using plyer notifications.")
        return _plyer, "plyer"

    except ImportError:
        pass

    # ── 3. win10toast ─────────────────────────────────────────────────────────
    try:
        from win10toast import ToastNotifier
        _win10_toaster = ToastNotifier()

        def _win10toast(title: str, message: str, icon_path: str = ""):
            _win10_toaster.show_toast(
                title,
                message,
                icon_path  = icon_path or None,
                duration   = 5,
                threaded   = True,
            )

        print("[AlertSystem] Using win10toast notifications.")
        return _win10toast, "win10toast"

    except ImportError:
        pass

    # ── 4. PowerShell BurntToast / Windows fallback (no extra packages) ───────
    if sys.platform == "win32":
        def _powershell(title: str, message: str, icon_path: str = ""):
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$n = New-Object System.Windows.Forms.NotifyIcon; "
                "$n.Icon = [System.Drawing.SystemIcons]::Information; "
                "$n.Visible = $True; "
                f"$n.ShowBalloonTip(5000, '{title}', '{message}', "
                "[System.Windows.Forms.ToolTipIcon]::Info)"
            )
            subprocess.Popen(
                ["powershell", "-WindowStyle", "Hidden",
                 "-NonInteractive", "-Command", ps_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=0x08000000,   # CREATE_NO_WINDOW
            )

        print("[AlertSystem] Using PowerShell balloon tooltip (no extra packages needed).")
        return _powershell, "powershell"

    # ── 5. macOS osascript ─────────────────────────────────────────────────────
    if sys.platform == "darwin":
        def _macos(title: str, message: str, icon_path: str = ""):
            subprocess.Popen(
                ["osascript", "-e",
                 f'display notification "{message}" with title "{title}"'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        print("[AlertSystem] Using macOS osascript notifications.")
        return _macos, "osascript"

    # ── 6. Linux notify-send ──────────────────────────────────────────────────
    if sys.platform.startswith("linux"):
        def _linux(title: str, message: str, icon_path: str = ""):
            subprocess.Popen(
                ["notify-send", "-t", "5000", title, message],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        print("[AlertSystem] Using notify-send (Linux).")
        return _linux, "notify-send"

    # ── 7. Console fallback ───────────────────────────────────────────────────
    def _console(title: str, message: str, icon_path: str = ""):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] 🔔 {title}: {message}")

    print("[AlertSystem] No toast backend found — printing to console.")
    return _console, "console"


_NOTIFY_FN, _BACKEND = _detect_backend()


# ── TTS availability ──────────────────────────────────────────────────────────
try:
    import pyttsx3 as _pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[AlertSystem] pyttsx3 not found — voice alerts disabled.")


# ── AlertSystem class ─────────────────────────────────────────────────────────

class AlertSystem:
    """
    Manages toast notifications and optional voice feedback.

    Usage:
        alert = AlertSystem(voice_enabled=True)
        alert.notify("⚠️ Posture Alert", "Sit straight!", category="posture")
        alert.shutdown()
    """

    # Per-category cooldowns (seconds)
    COOLDOWNS = {
        "posture": 10.0,
        "stress":  20.0,
        "blink":   15.0,
        "good":    30.0,
        "info":     5.0,
    }

    # Optional: path to a .ico/.png icon shown in the toast
    ICON_PATH = ""   # e.g. "assets/icon.ico"

    def __init__(self, voice_enabled: bool = True):
        self._last_triggered: dict[str, float] = {}
        self._voice_enabled = voice_enabled and TTS_AVAILABLE

        # Voice queue (background thread so it never blocks CV loop)
        self._speech_queue: queue.Queue = queue.Queue()
        if self._voice_enabled:
            self._tts_thread = threading.Thread(
                target=self._tts_worker, daemon=True, name="TTS")
            self._tts_thread.start()

        # Notification dispatch is also non-blocking
        self._notif_queue: queue.Queue = queue.Queue()
        self._notif_thread = threading.Thread(
            target=self._notif_worker, daemon=True, name="Notifications")
        self._notif_thread.start()

        print(f"[AlertSystem] Ready  (backend={_BACKEND}, "
              f"voice={'on' if self._voice_enabled else 'off'})")

    # ── Public API ────────────────────────────────────────────────────────────

    def notify(self, title: str, message: str,
               category: str = "info",
               speech_text: str = "") -> None:
        """
        Fire a native OS toast notification if the cooldown for *category*
        has elapsed.  Voice alert queued separately.

        Args:
            title       – bold heading shown in the toast
            message     – body text of the notification
            category    – rate-limit bucket: posture | stress | blink | good | info
            speech_text – words spoken aloud; empty string = silent
        """
        now      = time.time()
        cooldown = self.COOLDOWNS.get(category, 10.0)

        if now - self._last_triggered.get(category, 0) < cooldown:
            return   # still in cooldown

        self._last_triggered[category] = now

        # Queue the OS toast (non-blocking)
        self._notif_queue.put((title, message))

        # Queue voice if enabled and text provided
        if speech_text and self._voice_enabled:
            self._speech_queue.put(speech_text)

    def shutdown(self) -> None:
        """Gracefully stop background threads."""
        self._notif_queue.put(None)
        if self._voice_enabled:
            self._speech_queue.put(None)

    # ── Legacy OpenCV draw methods (kept as no-ops for compatibility) ──────────
    # If any other module calls these they won't crash — they just do nothing.

    def draw_alerts(self, frame):
        return frame

    def draw_status_bar(self, frame, posture="", stress="",
                        ear=0.0, blink_rate=0.0):
        return frame

    def trigger(self, category: str, message: str,
                colour_key: str = "red", speech_text: str = "") -> None:
        """Legacy shim — maps old trigger() calls to notify()."""
        _TITLE_MAP = {
            "posture": "⚠️ Posture Alert",
            "stress":  "😤 Stress Alert",
            "blink":   "👁️ Blink Reminder",
            "good":    "✅ Good Posture",
            "lean":    "↗️ Screen Lean",
        }
        title = _TITLE_MAP.get(category, "ℹ️ Alert")
        self.notify(title=title, message=message,
                    category=category, speech_text=speech_text)

    # ── Private workers ───────────────────────────────────────────────────────

    def _notif_worker(self) -> None:
        """Background thread: sends queued toasts one by one."""
        while True:
            item = self._notif_queue.get()
            if item is None:
                break
            title, message = item
            try:
                _NOTIFY_FN(title, message, self.ICON_PATH)
            except Exception as exc:
                print(f"[AlertSystem] Toast error: {exc}")

    def _tts_worker(self) -> None:
        """Background thread: speaks queued phrases via pyttsx3."""
        engine = _pyttsx3.init()
        engine.setProperty("rate",   160)
        engine.setProperty("volume", 0.9)

        while True:
            text = self._speech_queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[TTS] Error: {exc}")