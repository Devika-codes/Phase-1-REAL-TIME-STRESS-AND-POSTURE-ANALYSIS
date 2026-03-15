"""
ui/tray_app.py
──────────────────────────────────────────────────────
PyQt5 system-tray application — notifications only, no dashboard.

Right-click the tray icon to:
  ▶  Start Session
  ⏹  End Session
  ✕  Quit
"""

import sys
import os
from typing import Optional

from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
from PyQt5.QtCore    import Qt, QObject
from PyQt5.QtGui     import QIcon, QPixmap, QPainter, QColor, QBrush, QPen, QRadialGradient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.detector      import StressPostureDetector
from core.notifications import NotificationOverlay


# ── Tray icon ─────────────────────────────────────────────────────────────────

def _make_tray_icon() -> QIcon:
    px = QPixmap(64, 64)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)

    grad = QRadialGradient(32, 32, 30)
    grad.setColorAt(0.0,  QColor('#8b5cf6'))
    grad.setColorAt(0.75, QColor('#6d28d9'))
    grad.setColorAt(1.0,  QColor(0, 0, 0, 0))
    p.setBrush(QBrush(grad))
    p.setPen(Qt.NoPen)
    p.drawEllipse(4, 4, 56, 56)

    p.setBrush(QColor('#1a0a3a'))
    p.drawEllipse( 9, 11, 19, 20)
    p.drawEllipse(36, 11, 19, 20)
    p.drawEllipse(19, 23, 26, 24)

    pen = QPen(QColor('white'), 2)
    p.setPen(pen)
    p.drawLine(18, 19, 18, 29)
    p.drawLine(32, 13, 32, 36)
    p.drawLine(46, 19, 46, 29)

    p.end()
    return QIcon(px)


# ── TrayApp ───────────────────────────────────────────────────────────────────

_CARD_BG    = '#0d1321'
_BORDER     = '#1e2a3a'
_TEXT       = '#e2e8f0'
_MUTED      = '#64748b'
_PURPLE     = '#8b5cf6'
_PURPLE_D   = '#4c1d95'


class TrayApp(QObject):

    def __init__(self, app: QApplication):
        super().__init__()
        self._app = app
        self.detector: Optional[StressPostureDetector] = None

        # Notification overlay (unchanged from original)
        self.overlay = NotificationOverlay()
        self.overlay.start()

        # ── Tray icon & menu ──────────────────────────────────────────────
        self._icon = QSystemTrayIcon(_make_tray_icon(), self._app)
        self._icon.setToolTip('Stress & Posture Monitor')

        menu = QMenu()
        menu.setStyleSheet(f"""
            QMenu{{
                background:{_CARD_BG};
                color:{_TEXT};
                border:1px solid {_BORDER};
                border-radius:8px;
                padding:4px;
            }}
            QMenu::item{{
                padding:8px 24px;
                border-radius:6px;
            }}
            QMenu::item:selected{{
                background:{_PURPLE}44;
                color:#c4b5fd;
            }}
            QMenu::separator{{
                height:1px;
                background:{_BORDER};
                margin:4px 8px;
            }}
        """)

        self._start_act = QAction('▶  Start Session', self._app)
        self._start_act.triggered.connect(self.start_session)
        menu.addAction(self._start_act)

        self._stop_act = QAction('⏹  End Session', self._app)
        self._stop_act.setEnabled(False)
        self._stop_act.triggered.connect(self.stop_session)
        menu.addAction(self._stop_act)

        menu.addSeparator()

        quit_act = QAction('✕  Quit', self._app)
        quit_act.triggered.connect(self._quit)
        menu.addAction(quit_act)

        self._icon.setContextMenu(menu)
        self._icon.show()

        self._icon.showMessage(
            'Stress & Posture Monitor',
            'Running in tray. Right-click to start a session.',
            QSystemTrayIcon.Information, 3000)

    # ── Session control ───────────────────────────────────────────────────────

    def start_session(self):
        if self.detector and self.detector._running:
            return
        self.detector = StressPostureDetector(on_alert=self._on_alert)
        self.detector.start()
        self._start_act.setEnabled(False)
        self._stop_act.setEnabled(True)
        self._icon.showMessage(
            'Session Started',
            'Camera active. Sit naturally for ~30 s to calibrate.',
            QSystemTrayIcon.Information, 5000)
        print('  → Session started. Calibrating for ~30 s…')

    def stop_session(self):
        if not self.detector:
            return
        detector = self.detector
        self.detector = None
        self._start_act.setEnabled(True)
        self._stop_act.setEnabled(False)
        sd = detector.stop()
        self._icon.showMessage(
            'Session Ended',
            f'Duration: {sd.duration_min} min  |  '
            f'Posture alerts: {sd.posture_alerts}  |  '
            f'Stress alerts: {sd.stress_alerts}',
            QSystemTrayIcon.Information, 5000)
        print(f'  → Session ended. Duration: {sd.duration_min} min, '
              f'posture alerts: {sd.posture_alerts}, '
              f'stress alerts: {sd.stress_alerts}')

    # ── Alert callback (called from detector's background thread) ─────────────

    def _on_alert(self, kind: str, score: float):
        # overlay.push() uses Qt.QueuedConnection internally, so this is
        # safe to call from any thread.
        self.overlay.push(kind)

    # ── Quit ──────────────────────────────────────────────────────────────────

    def _quit(self):
        if self.detector:
            try:
                self.detector.stop()
            except Exception:
                pass
            self.detector = None
        try:
            self.overlay.stop()
        except Exception:
            pass
        self._app.quit()
