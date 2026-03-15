# 🧠 Stress & Posture Awareness System (No Dashboard)

A real-time wellness monitor that silently watches your posture and stress via your webcam.  
Lives in the **system tray** — always running, never in your way.  
Notifications slide in from the top-right corner exactly as before. No dashboard window.

---

## 🚀 Quick Start

```bash
# 1. Install
pip install PyQt5 opencv-python numpy mediapipe

# 2. (Optional) Voice alerts on Linux
sudo apt install espeak

# 3. Run
python3 main.py
```

The app will appear as a **tray icon** (top-right on macOS, bottom-right on Windows/Linux).

---

## 🖱️ How to Use

| Action | Result |
|--------|--------|
| **Right-click tray icon → Start Session** | Camera starts, calibration runs (~30 s) |
| **Right-click tray icon → End Session** | Camera stops, summary shown as tray notification |
| **Right-click tray icon → Quit** | Exits the app |

### Session Flow
```
Run app → tray icon appears
     ↓
Right-click → "Start Session"
     ↓
Sit naturally for ~30 seconds (calibration)
     ↓
Work normally — notifications pop top-right when issues are detected
     ↓
Right-click → "End Session"
```

---

## 🔔 Notifications

Gradient cards slide in from the top-right corner, stay for 5 seconds, then slide out.

| Color | When | Voice | Sample message |
|-------|------|-------|----------------|
| 🔴 **Red** | Posture issue detected (3 s sustained) | *"Please sit straight"* | "Your spine called — it wants its dignity back!" |
| 🟡 **Yellow** | Any stress signal (3 s sustained) | *"Relax. Take a deep breath!"* | "Your face is tighter than an 11:59 PM deadline." |
| 🟢 **Green** | 10-min good posture streak | *"Good posture maintained!"* | "Posture Champion unlocked! 🏆" |
| 🔵 **Blue** | Every 30 minutes | *"Break time! Drink some water!"* | "Your brain is 73% water. Don't let it become a raisin." |

Press **ESC** or **click** any notification to dismiss it early.

---

## ⚙️ Project Structure

```
stress_posture_app/
├── main.py                  ← Entry point
├── requirements.txt
├── core/
│   ├── detector.py          ← Camera + signal processing (unchanged)
│   └── notifications.py     ← Gradient popup notifications (unchanged)
└── ui/
    └── tray_app.py          ← Tray icon + session control (no dashboard)
```

---

## 🔧 Troubleshooting

**No tray icon on Linux?**
```bash
sudo apt install libappindicator3-1 gnome-shell-extension-appindicator
```

**Camera not found?**
```bash
python3 -c "import cv2; c=cv2.VideoCapture(0); print(c.read()[0])"
```

**No voice?**
```bash
# Linux
sudo apt install espeak && espeak "hello"
# macOS — just works: say "hello"
```
