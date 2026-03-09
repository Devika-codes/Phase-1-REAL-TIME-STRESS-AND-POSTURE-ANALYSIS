# 🧘 Real-Time Stress & Posture Detection System

A Python computer-vision mini-project that uses your webcam to continuously
monitor posture, facial stress indicators, and eye blink rate — then alerts
you with colour-coded on-screen notifications and optional voice feedback.

---

## 📁 Project Structure

```
stress_posture_system/
│
├── main.py               ← Entry point & main processing loop
├── posture_detector.py   ← MediaPipe Pose skeleton analysis
├── stress_detector.py    ← FaceMesh heuristic stress scoring
├── blink_detector.py     ← Eye Aspect Ratio blink tracker
├── alert_system.py       ← HUD overlay + pyttsx3 voice alerts
├── utils.py              ← Shared math helpers
├── requirements.txt      ← Python dependencies
└── README.md             ← This file
```

---

## 🚀 Installation & Setup

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.10 or 3.11 recommended |
| Webcam | Built-in or USB |
| pip | Up to date (`pip install --upgrade pip`) |

### Step 1 — Clone / Download

```bash
# If using git:
git clone <repo-url>
cd stress_posture_system

# Or just copy the folder and cd into it
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Activate:
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Linux only** — pyttsx3 needs `espeak`:
> ```bash
> sudo apt-get install espeak espeak-data libespeak-dev ffmpeg
> ```

### Step 4 — Run in VS Code

1. Open the `stress_posture_system/` folder in VS Code.
2. Select your `.venv` Python interpreter (`Ctrl+Shift+P` → *Python: Select Interpreter*).
3. Open `main.py` and press **F5** (or use the Run menu).

**Or from the terminal:**
```bash
python main.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit the application |
| `V` | Toggle voice alerts on / off |
| `S` | Save a screenshot to `./screenshots/` |

---

## ⚙️ Command-Line Options

```
python main.py [OPTIONS]

  --camera   INT   Webcam index (default: 0)
  --width    INT   Frame width  (default: 1280)
  --height   INT   Frame height (default: 720)
  --fps      INT   Target FPS   (default: 30)
  --no-voice       Disable TTS voice alerts
```

Examples:
```bash
python main.py --camera 1              # Use secondary camera
python main.py --no-voice              # Silent mode
python main.py --width 640 --height 480  # Lower resolution for slow machines
```

---

## 🔍 How It Works

### 1. Posture Detection (`posture_detector.py`)

Uses **MediaPipe Pose** to extract 33 body keypoints every frame.

| Metric | Threshold | Alert |
|--------|-----------|-------|
| Shoulder vertical difference | > 0.05 (normalised) | Uneven shoulders |
| Nose offset from shoulder midpoint | > 0.08 | Forward head posture |
| Ear → Shoulder → Hip angle | < 165° | Hunching / curved spine |
| Shoulder midpoint vs hip midpoint | > 0.06 | Leaning to one side |

### 2. Stress Detection (`stress_detector.py`)

Uses **MediaPipe FaceMesh** (468 landmarks) with three heuristic indicators:

| Indicator | Threshold | Meaning |
|-----------|-----------|---------|
| Inner brow distance to nose bridge | < 0.018 | Furrowed brows (tension) |
| Eye vertical span / face height | > 0.038 | Wide eyes (alertness) |
| Mouth corner Y vs mouth top Y | < −0.008 | Frowning |

Scores are smoothed over a 20-frame rolling window:

| Score | Level |
|-------|-------|
| 0 | Low |
| 1 | Mild |
| 2–3 | High |

### 3. Blink Detection (`blink_detector.py`)

Computes **Eye Aspect Ratio (EAR)**:

```
EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
```

| Parameter | Value |
|-----------|-------|
| EAR closed threshold | 0.22 |
| Min consecutive closed frames | 2 |
| Alert blink rate | < 12 blinks/min |
| Measurement window | 60 seconds |

### 4. Alert System (`alert_system.py`)

Colour-coded overlay notifications with per-category cooldowns:

| Colour | Meaning | Cooldown |
|--------|---------|----------|
| 🟢 Green | Posture is good | 30 s |
| 🟡 Yellow | Mild stress warning | 20 s |
| 🔴 Red | Bad posture / high stress | 10 s |
| 🔵 Blue | Low blink rate reminder | 15 s |

Voice alerts run on a **background thread** so they never block the video loop.

---

## 🖥️ Screen Layout

```
┌─────────────────────────────────────────────────────────┐
│ Posture: Good  Stress: Low  EAR: 0.32  Blinks/min: 18  │  ← HUD bar
├─────────────────────────────────────────────────────────┤
│                                             ┌──────────┐│
│                                             │ ✓ Great  ││ ← Alert panel
│                                             │  posture ││   (top-right)
│         [WEBCAM FEED + SKELETON]            └──────────┘│
│                                                         │
│ Shoulder diff: 0.012                                    │
│ Head offset  : 0.031                                    │
│ Spine angle  : 172.3 deg                                │
│ Brow tension: 0.0241  Eye: 0.0312  Mouth: −0.003        │
│                                      FPS: 28.4          │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Tuning Thresholds

All thresholds are class-level constants — easy to adjust:

```python
# posture_detector.py
PostureDetector.SPINE_ANGLE_THRESHOLD    = 165   # degrees
PostureDetector.FORWARD_HEAD_THRESHOLD   = 0.08

# stress_detector.py
StressDetector.EYE_WIDE_THRESHOLD        = 0.038
StressDetector.STRESS_HIGH_SCORE         = 2

# blink_detector.py
BlinkDetector.EAR_THRESHOLD              = 0.22
BlinkDetector.BLINK_RATE_THRESHOLD       = 12    # blinks/min
```

---

## 🧩 Extending with a CNN Stress Classifier

The heuristic stress scorer can be replaced with a CNN:

1. Uncomment `tensorflow` in `requirements.txt`.
2. Train a model on a facial expression dataset (e.g., FER-2013).
3. In `stress_detector.py`, replace `_compute_stress_score()` with model inference:

```python
import tensorflow as tf

model = tf.keras.models.load_model("stress_model.h5")

def _compute_stress_score(self, lm, shape):
    face_roi = self._crop_face(frame, lm)   # extract & resize to 48×48
    pred = model.predict(face_roi[np.newaxis, ...])[0]
    return int(np.argmax(pred))   # 0=Low, 1=Mild, 2=High
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not found | Try `--camera 1` or `--camera 2` |
| Low FPS | Use `--width 640 --height 480` or set `model_complexity=0` in detectors |
| No voice on Linux | `sudo apt-get install espeak` |
| No voice on Windows | pyttsx3 uses SAPI5 — ensure Windows TTS is enabled |
| MediaPipe import error | Reinstall: `pip install mediapipe --force-reinstall` |

---

## 📄 License

MIT — free to use, modify, and distribute.
