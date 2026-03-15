#!/usr/bin/env python3
"""
main.py — Stress & Posture Awareness System (no dashboard)
───────────────────────────────────────────────────────────
Launches the PyQt5 system-tray application.
Notifications pop in the top-right corner exactly as before.
The dashboard has been removed; use the tray right-click menu to control sessions.

Usage:
    python3 main.py

Requirements:
    pip install PyQt5 opencv-python numpy mediapipe

Voice (optional):
    Linux:  sudo apt install espeak
    macOS:  nothing extra (uses built-in 'say')
"""

import sys
import os

# Ensure user site-packages is on path (e.g. when run from Cursor/IDE)
try:
    import site
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.insert(0, user_site)
except Exception:
    pass


# ── Dependency check ──────────────────────────────────────────────────────────

def check_deps():
    missing = []
    try:
        import PyQt5
    except ImportError:
        missing.append('PyQt5')
    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    try:
        import mediapipe
    except ImportError:
        missing.append('mediapipe')

    if missing:
        print('╔══════════════════════════════════════════════╗')
        print('║  Missing dependencies detected               ║')
        print('╚══════════════════════════════════════════════╝')
        parts = []
        for pkg in missing:
            if pkg == 'mediapipe':
                parts.append('"mediapipe>=0.10.0"')
            else:
                parts.append(pkg)
        python_exe = sys.executable
        print(f'\nUsing Python: {python_exe}')
        print('Install for this Python (run in terminal):')
        if sys.platform == 'win32' and ' ' in python_exe:
            print(f'  & "{python_exe}" -m pip install {" ".join(parts)}')
        else:
            print(f'  {python_exe} -m pip install {" ".join(parts)}')
        print()
        sys.exit(1)


check_deps()


# ── Launch ────────────────────────────────────────────────────────────────────

from PyQt5.QtWidgets import QApplication, QSystemTrayIcon
from PyQt5.QtCore    import Qt

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Stress & Posture Monitor')
    app.setQuitOnLastWindowClosed(False)   # keep alive in tray

    if not QSystemTrayIcon.isSystemTrayAvailable():
        print('ERROR: No system tray found on this platform.')
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(__file__))
    from ui.tray_app import TrayApp

    tray = TrayApp(app)

    print('╔═══════════════════════════════════════════════════════╗')
    print('║   🧠  Stress & Posture Monitor is running in tray    ║')
    print('║   → Right-click tray icon: Start / Stop / Quit       ║')
    print('║   → Notifications appear top-right when alerts fire  ║')
    print('╚═══════════════════════════════════════════════════════╝')

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
