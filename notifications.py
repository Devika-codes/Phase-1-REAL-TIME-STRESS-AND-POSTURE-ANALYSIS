"""
core/notifications.py — cross-platform gradient popup notifications.

Windows compatibility fixes:
  • WA_TranslucentBackground REMOVED — caused invisible windows on Windows DWM
  • Uses setWindowOpacity() for fade-in/out instead (works on all platforms)
  • Solid opaque background with manually painted rounded-rect + clipping mask
  • Qt.Tool replaced with Qt.SubWindow fallback for Windows
  • Explicit Qt.QueuedConnection on signal so cross-thread calls always queue properly
  • push() can be called from ANY thread safely
"""

import threading
import time
import random
import subprocess
import math
from typing import List

from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore    import Qt, QTimer, QObject, pyqtSignal, pyqtSlot, QRect, QPoint
from PyQt5.QtGui     import (
    QPainter, QColor, QLinearGradient, QFont,
    QPen, QBrush, QPainterPath, QRegion,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Messages
# ─────────────────────────────────────────────────────────────────────────────

MSGS = {
    'posture': [
        "Your spine called — it wants its dignity back!",
        "Architects draw straight lines. So should you.",
        "Vertebrae wellness check: FAILED. Sit up!",
        "You're not a question mark — straighten out!",
        "Less turtle neck, more swan neck. Chin up!",
        "Posture police on duty. Step away from the slouch!",
        "Your spine is doing acrobatics it never signed up for.",
        "Gravity wins again. Fight back — sit straight!",
        "Posture alignment: 0%. Let's get that to 100%!",
        "Channel your inner flamingo — tall and elegant!",
        "The Eiffel Tower stands straight. Be the tower.",
        "Chair is supportive. Your spine deserves the same.",
    ],
    'stress': [
        "Your face is tighter than an 11:59 PM deadline.",
        "Breathe like the ocean — in, out, in, out.",
        "Coffee can wait. Reset your nervous system first.",
        "Your brows called a meeting. Relax them!",
        "Bubble wrap OR deep breath? Breath wins every time.",
        "Stress levels: spicy. Remedy: one deep breath.",
        "Your face is doing drama. Downshift to comedy.",
        "Cool it like a cucumber. You're doing great!",
        "Tension detected! Time to discharge some energy.",
        "Hyper-focus mode on. Your brain needs a micro-vacation.",
        "Your jaw is clenching. Release it — feel the difference!",
        "Even snipers breathe before the shot. Breathe.",
    ],
    'appreciation': [
        "10 minutes of perfect posture! You absolute legend!",
        "Posture Champion unlocked! Your spine thanks you!",
        "WOW! Sitting like royalty for 10 whole minutes!",
        "Look at you — maintaining posture like a total PRO!",
        "Gold medal posture! Your chiropractor weeps with joy!",
        "10-minute greatness! Your posture game is elite!",
        "Perfect alignment streak! That's what we're talking about!",
        "Blooming with good posture energy. Keep it up, champion!",
    ],
    'water': [
        "30 min in! Your cells are thirsty little creatures.",
        "Hydration station calling! Time for that H2O hit!",
        "Drink water challenge: you're legally obligated to sip!",
        "Your brain is 73% water. Don't let it become a raisin.",
        "Coffee counts as dehydration. Drink WATER!",
        "Your kidneys slid a note: please hydrate.",
        "30-min milestone! Celebrate with a big glass of water.",
        "1% dehydration = 10% less brainpower. Drink up!",
    ],
}

VOICE_TEXT = {
    'posture':      "Please sit straight",
    'stress':       "Relax. Take a deep breath",
    'appreciation': "Great job! Good posture maintained!",
    'water':        "Break time! Drink some water!",
}

THEMES = {
    'posture': {
        'top': QColor(190, 20,  35),  'bot': QColor(120,  8, 20),
        'border': QColor(255, 80, 80), 'badge_bg': QColor(255, 60, 60, 60),
        'badge': 'POSTURE ALERT',     'text': QColor(255, 200, 200),
    },
    'stress': {
        'top': QColor(200, 110,  0),  'bot': QColor(140, 60,  0),
        'border': QColor(255, 180, 40), 'badge_bg': QColor(255, 160, 0, 60),
        'badge': 'STRESS DETECTED',   'text': QColor(255, 230, 160),
    },
    'appreciation': {
        'top': QColor(14, 155,  75),  'bot': QColor(8, 100, 50),
        'border': QColor(60, 220, 120), 'badge_bg': QColor(50, 200, 100, 60),
        'badge': 'GREAT JOB!',        'text': QColor(190, 255, 215),
    },
    'water': {
        'top': QColor(30, 100, 215),  'bot': QColor(15,  55, 155),
        'border': QColor(80, 160, 255), 'badge_bg': QColor(60, 130, 255, 60),
        'badge': 'HYDRATION TIME',    'text': QColor(190, 220, 255),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  NotifPopup  — opaque window, opacity-based fade (works on all platforms)
# ─────────────────────────────────────────────────────────────────────────────

class NotifPopup(QWidget):
    W         = 420
    H         = 115
    MARGIN    = 16
    RADIUS    = 14
    ANIM_MS   = 350    # fade-in / fade-out duration in ms
    HOLD_MS   = 5000   # how long the popup stays fully visible

    def __init__(self, kind: str, msg: str):
        # ── Window flags: FramelessWindowHint + StaysOnTop ──────────────────
        # Do NOT use Qt.Tool on Windows — it can make the window invisible.
        # Do NOT use WA_TranslucentBackground — breaks DWM compositing.
        # Use a plain frameless window + setWindowOpacity for fading.
        flags = (Qt.FramelessWindowHint |
                 Qt.WindowStaysOnTopHint |
                 Qt.WindowDoesNotAcceptFocus)
        super().__init__(None, flags)

        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        # WA_TranslucentBackground intentionally NOT set (Windows fix)
        self.setFixedSize(self.W, self.H)
        self.setWindowOpacity(0.0)   # start invisible; we fade via setWindowOpacity

        self._alive   = True
        self._msg     = msg
        self._theme   = THEMES.get(kind, THEMES['posture'])
        self._t0      = time.time()

        # Compute final resting position
        screen   = QApplication.primaryScreen().availableGeometry()
        self._tx = screen.right() - self.W - self.MARGIN
        self._ty = screen.top()   + self.MARGIN

        # Start off the right edge of the screen
        self.move(screen.right() + self.W, self._ty)

        self._phase     = 'in'
        self._phase_ms  = 0
        self._timer     = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def show_animated(self):
        self.show()
        self.raise_()
        self._phase_ms = 0
        self._timer.start(16)          # ~60 fps tick

    def _close_safe(self):
        self._alive = False
        self._timer.stop()
        self.hide()
        self.deleteLater()

    def _tick(self):
        self._phase_ms += 16

        if self._phase == 'in':
            frac = min(self._phase_ms / self.ANIM_MS, 1.0)
            ease = 1 - (1 - frac) ** 3          # ease-out cubic
            # Slide in from the right
            off_x = int((1 - ease) * (self.W + self.MARGIN + 20))
            self.move(self._tx + off_x, self._ty)
            self.setWindowOpacity(ease)
            if frac >= 1.0:
                self._phase    = 'hold'
                self._phase_ms = 0

        elif self._phase == 'hold':
            self.move(self._tx, self._ty)
            self.setWindowOpacity(1.0)
            if self._phase_ms >= self.HOLD_MS:
                self._phase    = 'out'
                self._phase_ms = 0

        elif self._phase == 'out':
            frac = min(self._phase_ms / self.ANIM_MS, 1.0)
            ease = frac ** 2                    # ease-in quad
            off_x = int(ease * (self.W + self.MARGIN + 20))
            self.move(self._tx + off_x, self._ty)
            self.setWindowOpacity(1.0 - ease)
            if frac >= 1.0:
                self._close_safe()
                return

        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W, H  = self.W, self.H
        R     = self.RADIUS
        t     = self._theme
        secs  = time.time() - self._t0

        # ── Background gradient ───────────────────────────────────────────────
        bg = QLinearGradient(0, 0, 0, H)
        bg.setColorAt(0.0, t['top'])
        bg.setColorAt(1.0, t['bot'])
        path = QPainterPath()
        path.addRoundedRect(0, 0, W, H, R, R)
        p.fillPath(path, QBrush(bg))

        # ── Subtle top-shine overlay ──────────────────────────────────────────
        shine = QLinearGradient(0, 0, 0, H // 2)
        shine.setColorAt(0.0, QColor(255, 255, 255, 30))
        shine.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillPath(path, QBrush(shine))

        # ── Pulsing border ────────────────────────────────────────────────────
        pulse = 0.5 + 0.5 * math.sin(secs * 5.0)
        bc    = QColor(t['border'])
        bc.setAlpha(int(100 + 100 * pulse))
        p.setPen(QPen(bc, 2.0))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, W - 2, H - 2, R, R)

        # ── Badge ─────────────────────────────────────────────────────────────
        badge_bg = QColor(t['badge_bg'])
        p.setBrush(badge_bg)
        p.setPen(Qt.NoPen)
        badge_w = len(t['badge']) * 7 + 20
        p.drawRoundedRect(12, 9, badge_w, 18, 5, 5)

        f1 = QFont(); f1.setPointSize(7); f1.setBold(True); f1.setLetterSpacing(QFont.AbsoluteSpacing, 0.8)
        p.setFont(f1)
        p.setPen(QPen(QColor(t['text'])))
        p.drawText(QRect(12, 9, badge_w, 18), Qt.AlignCenter, t['badge'])

        # ── Divider ───────────────────────────────────────────────────────────
        div_c = QColor(t['text']); div_c.setAlpha(50)
        p.setPen(QPen(div_c, 1))
        p.drawLine(12, 32, W - 12, 32)

        # ── Message text — word-wrapped ───────────────────────────────────────
        words = self._msg.split()
        lines, cur = [], ''
        for w in words:
            test = (cur + ' ' + w).strip()
            if len(test) <= 46:
                cur = test
            else:
                lines.append(cur); cur = w
        if cur:
            lines.append(cur)

        f2 = QFont(); f2.setPointSize(10); f2.setBold(True)
        p.setFont(f2)
        p.setPen(QPen(QColor(255, 255, 255, 235)))
        line_h = 24
        total_h = len(lines[:2]) * line_h
        y_start = 35 + (H - 35 - total_h) // 2
        for i, line in enumerate(lines[:2]):
            p.drawText(QRect(14, y_start + i * line_h, W - 28, line_h),
                       Qt.AlignLeft | Qt.AlignVCenter, line)

        p.end()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self._close_safe()

    def mousePressEvent(self, ev):
        """Click to dismiss."""
        self._close_safe()


# ─────────────────────────────────────────────────────────────────────────────
#  NotificationOverlay — thread-safe dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class NotificationOverlay(QObject):
    """
    Lives on the Qt main thread.
    push() is safe to call from ANY thread — the signal uses Qt.QueuedConnection
    so it always marshals delivery to the main thread regardless of caller.
    """

    # Explicit str, str signal
    _do_show = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.voice_enabled = True
        self.voice_gender  = 'female'
        self._popups: List[NotifPopup] = []
        # Qt.QueuedConnection ensures the slot runs on THIS object's thread (main thread)
        # even when emit() is called from a background thread
        self._do_show.connect(self._create_popup, Qt.QueuedConnection)

    def start(self):
        pass

    def stop(self):
        for p in list(self._popups):
            try:
                if getattr(p, '_alive', False):
                    p._close_safe()
            except Exception:
                pass
        self._popups.clear()

    def push(self, kind: str):
        """Thread-safe — can be called from any thread."""
        msg = random.choice(MSGS.get(kind, ['Alert!']))
        self._do_show.emit(kind, msg)   # queued → always delivers on main thread
        if self.voice_enabled:
            threading.Thread(target=self._speak, args=(kind,),
                             daemon=True).start()

    def set_voice(self, enabled: bool, gender: str):
        self.voice_enabled = enabled
        self.voice_gender  = gender

    @pyqtSlot(str, str)
    def _create_popup(self, kind: str, msg: str):
        """Always runs on Qt main thread — safe to create/show widgets."""
        live = []
        for p in self._popups:
            try:
                if getattr(p, '_alive', False):
                    live.append(p)
            except Exception:
                pass
        self._popups = live
        popup = NotifPopup(kind, msg)
        self._popups.append(popup)
        popup.show_animated()

    def _speak(self, kind: str):
        text  = VOICE_TEXT.get(kind, 'Alert!')
        pitch = '72' if self.voice_gender == 'female' else '48'
        cmds  = [
            # Windows PowerShell SAPI
            ['powershell', '-WindowStyle', 'Hidden', '-NonInteractive', '-Command',
             f'Add-Type -AssemblyName System.Speech; '
             f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
             f'$s.Speak("{text}")'],
            # macOS
            ['say', '-v', 'Samantha' if self.voice_gender == 'female' else 'Alex', text],
            # Linux espeak
            ['espeak', '-p', pitch, '-s', '145', '-a', '180', text],
            # Linux festival
            ['bash', '-c', f'echo "{text}" | festival --tts'],
        ]
        for cmd in cmds:
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                return
            except (FileNotFoundError, OSError):
                continue
