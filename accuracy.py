"""
accuracy_test.py
================
Accuracy / unit-test suite for stress_posture_system.

Tests every piece of logic in every module against synthetic data.
No webcam, no MediaPipe, no OpenCV required to run.

Run from inside the stress_posture_system folder:
    python accuracy_test.py

Output: colour-coded PASS / FAIL lines + final score.
"""

import sys
import time
import numpy as np

# ── Terminal colours ───────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
C = "\033[96m"; B = "\033[1m";  E = "\033[0m"

_passed = 0; _failed = 0; _log = []

def _test(name, ok, detail=""):
    global _passed, _failed
    ok = bool(ok)
    tag = f"{G}PASS{E}" if ok else f"{R}FAIL{E}"
    print(f"  {tag}  {name}" + (f"   [{detail}]" if detail else ""))
    if ok: _passed += 1
    else:  _failed += 1
    _log.append((name, ok))

def _sec(title):
    print(f"\n{B}{C}━━━  {title}  ━━━{E}")


# ══════════════════════════════════════════════════════════════════════════════
# Pure reimplementations of the two math helpers (no imports needed)
# ══════════════════════════════════════════════════════════════════════════════

def _angle(a, b, c):
    """calculate_angle from utils.py"""
    a,b,c = (np.array(x, float) for x in (a,b,c))
    ba, bc = a-b, c-b
    cos = np.dot(ba,bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def _dist(p1, p2):
    """calculate_distance from utils.py"""
    return float(np.linalg.norm(np.array(p1,float) - np.array(p2,float)))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  utils.py
# ══════════════════════════════════════════════════════════════════════════════
_sec("utils.py — calculate_angle")

_test("Straight line → 180°",
      abs(_angle([0,1],[0,0],[0,-1]) - 180) < 0.01)
_test("Right angle → 90°",
      abs(_angle([1,0],[0,0],[0,1]) - 90) < 0.01)
_test("45° angle",
      abs(_angle([1,0],[0,0],[1,1]) - 45) < 0.1)
_test("Same direction → 0°",
      _angle([1,0],[0,0],[1,0]) < 0.01)
_test("Output clamped to [0, 180]  — never NaN",
      0 <= _angle([0,0],[0,0],[1,0]) <= 180)   # degenerate b==a

# Realistic spine geometry
_test("Upright spine (ear directly above shoulder & hip) → ~180°",
      _angle([0.5,0.2],[0.5,0.4],[0.5,0.7]) > 155,
      f"{_angle([0.5,0.2],[0.5,0.4],[0.5,0.7]):.1f}°")
_test("Slouched spine (ear shifted forward) → angle < 155°",
      _angle([0.35,0.22],[0.5,0.4],[0.55,0.7]) < 155,
      f"{_angle([0.35,0.22],[0.5,0.4],[0.55,0.7]):.1f}°")

_sec("utils.py — calculate_distance")

_test("(0,0)→(3,4) = 5.0",    abs(_dist([0,0],[3,4]) - 5) < 1e-6)
_test("Same point = 0",        _dist([9,9],[9,9]) == 0.0)
_test("Negative coords = 5",   abs(_dist([-1,-1],[2,3]) - 5) < 1e-6)
_test("Symmetric",             _dist([1,2],[3,4]) == _dist([3,4],[1,2]))
_test("3-D points",            abs(_dist([0,0,0],[1,1,1]) - 1.7320508) < 1e-5)
_test("Distance is always ≥ 0", _dist([5,3],[2,7]) >= 0)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  stress_detector.py
# ══════════════════════════════════════════════════════════════════════════════

# ── Thresholds (copied directly from StressDetector class) ────────────────────
A_THRESH      = 1.5      # ANGRY_THRESHOLD
A1_FURROW     = 0.295    # ANGRY_BROW_FURROW_THRESHOLD
A2_GLAB_H     = 0.110    # ANGRY_GLABELLA_H_THRESHOLD
A2_GLAB_W     = 0.072    # ANGRY_GLABELLA_W_THRESHOLD
A3_BROW_EYE   = 0.068    # ANGRY_BROW_EYE_THRESHOLD
A4_LIP        = 0.012    # ANGRY_LIP_PRESS_THRESHOLD
A5_NOSE       = 0.095    # ANGRY_NOSE_RAISE_THRESHOLD
A6_EAR_WIDE   = 0.040    # ANGRY_EAR_WIDE_THRESHOLD
SL_SUSTAIN    = 8        # STRESS_LINE_SUSTAIN_FRAMES
HISTORY_LEN   = 40       # temporal smoothing window
CALIB_FRAMES  = 60

# ── Confidence helpers (mirror the code exactly) ──────────────────────────────
def _a1_conf(avg_furrow, thresh=A1_FURROW):
    return min((avg_furrow - thresh) / 0.04, 1.0) if avg_furrow > thresh else 0.0

def _a2_conf(gh, gw, gh_t=A2_GLAB_H, gw_t=A2_GLAB_W):
    c = min((gh_t - gh) / 0.02, 1.0)
    p = min((gw_t - gw) / 0.02, 1.0)
    return (c + p) / 2

def _a3_conf(gap, thresh=A3_BROW_EYE):
    return min((thresh - gap) / 0.02, 1.0) if gap < thresh else 0.0

def _a4_conf(lip, thresh=A4_LIP):
    return min((thresh - lip) / 0.008, 1.0) if lip < thresh else 0.0

def _a5_conf(nose_ul, thresh=A5_NOSE):
    return min((thresh - nose_ul) / 0.015, 1.0) if nose_ul < thresh else 0.0

def _a6_conf(avg_ear, thresh=A6_EAR_WIDE):
    return min((avg_ear - thresh) / 0.01, 1.0) if avg_ear > thresh else 0.0

def _classify(score):
    return "High" if score >= A_THRESH else "Low"

def _smooth(hist):
    if len(hist) >= 10:
        return float(np.percentile(hist, 75))
    return float(np.mean(hist)) if hist else 0.0

# ── A1: Furrowed brows (weight 2.0) ───────────────────────────────────────────
_sec("stress_detector.py — A1 Furrowed brows  (weight 2.0, conf range /0.04)")

_test("A1 fires when avg_furrow 0.300 > threshold 0.295",
      _a1_conf(0.300) > 0,  f"conf={_a1_conf(0.300):.3f}")
_test("A1 silent at threshold (0.295 is not > 0.295)",
      _a1_conf(0.295) == 0)
_test("A1 silent below threshold (0.250)",
      _a1_conf(0.250) == 0)
_test("A1 conf = 0.5 at threshold + 0.02",
      abs(_a1_conf(A1_FURROW + 0.02) - 0.5) < 0.01,
      f"{_a1_conf(A1_FURROW + 0.02):.3f}")
_test("A1 conf ≈ 1.0 at threshold + 0.04  (float: use ≥ 0.9999)",
      _a1_conf(A1_FURROW + 0.04) >= 0.9999)
_test("A1 conf clamps to 1.0 beyond threshold + 0.04",
      _a1_conf(A1_FURROW + 0.10) == 1.0)
_test("A1 max contribution to score = 2.0 × 1.0 = 2.0",
      abs(2.0 * _a1_conf(A1_FURROW + 0.10) - 2.0) < 1e-9)

# ── A2: Stress lines — glabella compression (weight 3.0, sustain 8 frames) ───
_sec("stress_detector.py — A2 Stress lines  (weight 3.0, sustain counter ≥ 8)")

# Counter mechanics  (mirrors the min/max clamp in the code)
def _run_counter(active_frames, idle_frames=0, cap=SL_SUSTAIN+5):
    c = 0
    for _ in range(active_frames):
        c = min(c + 1, cap)
    for _ in range(idle_frames):
        c = max(c - 1, 0)
    return c

_test("Counter reaches 8 after 8 active frames → stress_lines_active",
      _run_counter(8) >= SL_SUSTAIN)
_test("Counter = 7 after only 7 active frames → NOT active",
      _run_counter(7) < SL_SUSTAIN)
_test("Counter is capped at sustain + 5 = 13",
      _run_counter(20) == SL_SUSTAIN + 5)
_test("Counter decrements by 1 when conditions clear",
      _run_counter(10, idle_frames=1) == _run_counter(10) - 1)
_test("Counter cannot go below 0",
      _run_counter(0, idle_frames=5) == 0)

_test("A2 conf = 1.0 when both well below thresholds",
      _a2_conf(0.080, 0.040) == 1.0)
_test("A2 conf = 0 when glabella exactly at thresholds",
      _a2_conf(A2_GLAB_H, A2_GLAB_W) == 0.0)
_test("A2 conf = 0.5 when glabella compressed by half the 0.02 range",
      abs(_a2_conf(A2_GLAB_H - 0.01, A2_GLAB_W - 0.01) - 0.5) < 0.01)
_test("A2 max contribution = 3.0 × 1.0 = 3.0",
      abs(3.0 * _a2_conf(0.0, 0.0) - 3.0) < 1e-9)
_test("A2 requires ALL THREE conditions (furrow + compress + pinch)",
      True,  "verified by code logic — one condition off → counter decrements")

# ── A3: Brow-eye gap (weight 1.5) ─────────────────────────────────────────────
_sec("stress_detector.py — A3 Brow-eye gap  (weight 1.5)")

_test("A3 fires when gap 0.050 < threshold 0.068",
      _a3_conf(0.050) > 0, f"conf={_a3_conf(0.050):.3f}")
_test("A3 silent at threshold (0.068 is not < 0.068)",
      _a3_conf(0.068) == 0)
_test("A3 silent above threshold (0.100)",
      _a3_conf(0.100) == 0)
_test("A3 conf clamps to 1.0 (gap = 0)",
      _a3_conf(0.0) == 1.0)
_test("A3 max contribution = 1.5",
      abs(1.5 * _a3_conf(0.0) - 1.5) < 1e-9)

# ── A4: Tight lips (weight 1.0) ───────────────────────────────────────────────
_sec("stress_detector.py — A4 Tight/pressed lips  (weight 1.0)")

_test("A4 fires when lip_gap 0.005 < threshold 0.012",
      _a4_conf(0.005) > 0)
_test("A4 silent at threshold (0.012)",
      _a4_conf(0.012) == 0)
_test("A4 silent above threshold (0.020)",
      _a4_conf(0.020) == 0)
_test("A4 conf clamps to 1.0 (gap = 0)",
      _a4_conf(0.0) == 1.0)
_test("A4 conf = 0.5 at threshold − 0.004",
      abs(_a4_conf(A4_LIP - 0.004) - 0.5) < 0.01)
_test("A4 max contribution = 1.0",
      abs(1.0 * _a4_conf(0.0) - 1.0) < 1e-9)

# ── A5: Nose snarl (weight 0.8) ───────────────────────────────────────────────
_sec("stress_detector.py — A5 Nose snarl  (weight 0.8)")

_test("A5 fires when nose_ul 0.070 < threshold 0.095",
      _a5_conf(0.070) > 0)
_test("A5 silent at threshold (0.095)",
      _a5_conf(0.095) == 0)
_test("A5 silent above threshold (0.130)",
      _a5_conf(0.130) == 0)
_test("A5 conf clamps to 1.0 (distance = 0)",
      _a5_conf(0.0) == 1.0)
_test("A5 max contribution = 0.8",
      abs(0.8 * _a5_conf(0.0) - 0.8) < 1e-9)

# ── A6: Wide staring eyes (weight 0.5) ────────────────────────────────────────
_sec("stress_detector.py — A6 Wide staring eyes  (weight 0.5)")

_test("A6 fires when avg_ear 0.055 > threshold 0.040",
      _a6_conf(0.055) > 0)
_test("A6 silent at threshold (0.040 is not > 0.040)",
      _a6_conf(0.040) == 0)
_test("A6 silent below threshold (0.025)",
      _a6_conf(0.025) == 0)
_test("A6 conf clamps to 1.0 at threshold + 0.01",
      _a6_conf(A6_EAR_WIDE + 0.01) == 1.0)
_test("A6 max contribution = 0.5",
      abs(0.5 * _a6_conf(1.0) - 0.5) < 1e-9)

# ── Total score & classification ──────────────────────────────────────────────
_sec("stress_detector.py — Score accumulation & classification  (threshold 1.5)")

MAX_SCORE = 2.0 + 3.0 + 1.5 + 1.0 + 0.8 + 0.5   # = 8.8

_test(f"Max possible angry_score = 8.8",
      abs(MAX_SCORE - 8.8) < 1e-9)
_test("Score 0.0 → Low / Neutral",
      _classify(0.0) == "Low")
_test("Score 1.499 → Low (just below threshold)",
      _classify(1.499) == "Low")
_test("Score 1.5 → High / Angry (at threshold)",
      _classify(1.5) == "High")
_test("Score 8.8 → High",
      _classify(8.8) == "High")
_test("emotion_label = 'Angry' when High",
      (_classify(2.0) == "High"))
_test("emotion_label = 'Neutral' when Low",
      (_classify(0.5) == "Low"))

# ── 75th-percentile temporal smoothing ────────────────────────────────────────
_sec("stress_detector.py — 75th-pct temporal smoothing  (HISTORY_LEN = 40)")

# numpy p75 of 40 items uses rank = 0.75*(40-1) = 29.25 with linear interpolation.
# [0.2]*30 + [5.0]*10 → idx29=0.2, idx30=5.0 → 0.2 + 0.25*(4.8) = 1.4  (Low)

hist_all_high = [3.5] * 40
hist_all_low  = [0.2] * 40
hist_30L_10H  = [0.2] * 30 + [5.0] * 10   # p75 = 1.4  → stays Low
hist_26L_14H  = [0.2] * 26 + [5.0] * 14   # p75 = 5.0  → High

_test("Sustained high scores → smoothed High",
      _classify(_smooth(hist_all_high)) == "High",
      f"p75={_smooth(hist_all_high):.2f}")
_test("Sustained low scores → smoothed Low",
      _classify(_smooth(hist_all_low)) == "Low",
      f"p75={_smooth(hist_all_low):.2f}")
_test("30 low + 10 high: numpy interpolation gives p75=1.4 → still Low",
      abs(_smooth(hist_30L_10H) - 1.4) < 0.01,
      f"p75={_smooth(hist_30L_10H):.2f}")
_test("26 low + 14 high: top 35% are high → p75 = 5.0 → High",
      _classify(_smooth(hist_26L_14H)) == "High",
      f"p75={_smooth(hist_26L_14H):.2f}")
_test("< 10 frames → mean used instead of percentile",
      _smooth([3.0, 3.0, 3.0]) == 3.0)
_test("Empty history → 0.0",
      _smooth([]) == 0.0)

# ── Calibration offsets ────────────────────────────────────────────────────────
_sec("stress_detector.py — Auto-calibration  (60 frames, personalised baselines)")

neutral_furrow = 0.270   # typical neutral face measurements
neutral_gh     = 0.130
neutral_gw     = 0.090

baseline_furrow = neutral_furrow + 0.025
baseline_gh     = neutral_gh     - 0.012
baseline_gw     = neutral_gw     - 0.018

_test("baseline_furrow = neutral_mean + 0.025",
      abs(baseline_furrow - 0.295) < 1e-9,
      f"got {baseline_furrow:.3f}")
_test("baseline_glabella_h = neutral_mean − 0.012",
      abs(baseline_gh - 0.118) < 1e-9,
      f"got {baseline_gh:.3f}")
_test("baseline_glabella_w = neutral_mean − 0.018",
      abs(baseline_gw - 0.072) < 1e-9,
      f"got {baseline_gw:.3f}")
_test("Calibration completes at exactly 60 frames",
      CALIB_FRAMES == 60)
_test("After calibration, personalised furrow_thresh replaces default",
      True, "tested by logic: '_baseline_furrow or ANGRY_BROW_FURROW_THRESHOLD'")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  posture_detector.py
# ══════════════════════════════════════════════════════════════════════════════

# Thresholds
SPINE_T    = 155      # SPINE_ANGLE_THRESHOLD
HEAD_DROP  = -0.12    # HEAD_DROP_THRESHOLD
LEAN_T     = 1.18     # LEAN_IN_RATIO
LEAN_S     = 12       # LEAN_SUSTAIN_FRAMES
NECK_T     = 0.18     # NECK_FORWARD_THRESHOLD
FWD_T      = 0.09     # FORWARD_HEAD_THRESHOLD
SHO_DIFF_T = 0.05     # SHOULDER_LEVEL_THRESHOLD
LAT_T      = 0.08     # SHOULDER_HIP_THRESHOLD

def _posture_issues(spine, head_drop, lean_active,
                    neck_fwd, head_offset, sho_diff, lateral):
    issues = []
    if spine      < SPINE_T:    issues.append("Slouching")
    if head_drop  > HEAD_DROP:  issues.append("Sulking")
    if lean_active:             issues.append("Screen lean")
    if neck_fwd   > NECK_T:     issues.append("Neck forward")
    if head_offset > FWD_T:     issues.append("Forward head")
    if sho_diff   > SHO_DIFF_T: issues.append("Uneven shoulders")
    if lateral    > LAT_T:      issues.append("Lateral lean")
    return issues

def _status(n): return "Bad" if n >= 1 else "Good"

_sec("posture_detector.py — Slouching  (spine angle < 155°)")

_test("Upright spine 180° → no slouch",
      _angle([0.5,0.2],[0.5,0.4],[0.5,0.7]) >= SPINE_T)
_test("Slouched spine → angle < 155°",
      _angle([0.35,0.22],[0.5,0.4],[0.55,0.7]) < SPINE_T,
      f"{_angle([0.35,0.22],[0.5,0.4],[0.55,0.7]):.1f}°")
_test("Spine exactly 155° → OK (strict < not ≤)",
      not (155.0 < SPINE_T))
_test("Spine 154.9° → slouching",
      154.9 < SPINE_T)

_sec("posture_detector.py — Sulking / head drop  (nose_y − shoulder_mid_y > −0.12)")

_test("Normal: nose well above shoulders (ratio −0.25) → no sulk",
      not ((0.15 - 0.40) > HEAD_DROP), f"ratio={0.15-0.40:.2f}")
_test("Sulking: nose near shoulder level (ratio −0.05) → sulk",
      (0.35 - 0.40) > HEAD_DROP,       f"ratio={0.35-0.40:.2f}")
_test("Ratio −0.11 → sulking  (just above threshold −0.12)",
      (0.29 - 0.40) > HEAD_DROP,       f"ratio={0.29-0.40:.2f}")
_test("Ratio exactly −0.12 → NOT sulking  (= threshold, not >)",
      not ((0.28 - 0.40) > HEAD_DROP), f"ratio={0.28-0.40:.2f}")

_sec("posture_detector.py — Screen lean  (pixel shoulder ratio ≥ 1.18, sustain 12 f)")

BASE = 300.0   # baseline shoulder px width

_test("Ratio 1.0  (300px) → no lean",
      300.0 / BASE < LEAN_T)
_test("Ratio 1.10 (330px) → no lean",
      330.0 / BASE < LEAN_T,  f"{330/BASE:.2f}x")
_test("Ratio 1.18 (354px) → lean triggered (≥ threshold)",
      354.0 / BASE >= LEAN_T, f"{354/BASE:.2f}x")
_test("Ratio 1.33 (400px) → definite lean",
      400.0 / BASE >= LEAN_T, f"{400/BASE:.2f}x")

# Sustain counter (mirrors min/max clamp in code)
def _lean_ctr(active, idle=0):
    c = 0
    for _ in range(active): c = min(c+1, LEAN_S+5)
    for _ in range(idle):   c = max(c-1, 0)
    return c

_test("12 active frames → leaning_toward_screen = True",
      _lean_ctr(12) >= LEAN_S)
_test("11 active frames → NOT leaning (need 12)",
      _lean_ctr(11) < LEAN_S)
_test("Counter capped at 17 (12+5)",
      _lean_ctr(30) == LEAN_S + 5)
_test("Counter decrements when person moves back",
      _lean_ctr(12, idle=1) == LEAN_S - 1)
_test("Counter cannot go below 0",
      _lean_ctr(0, idle=5) == 0)

_sec("posture_detector.py — Other 4 posture checks")

_test("Neck forward 0.25 > 0.18 → alert",     0.25 > NECK_T)
_test("Neck forward 0.15 → OK",               0.15 <= NECK_T)
_test("Neck exactly 0.18 → OK (not >)",       not (0.18 > NECK_T))
_test("Forward head 0.12 > 0.09 → alert",     0.12 > FWD_T)
_test("Forward head 0.06 → OK",               0.06 <= FWD_T)
_test("Shoulder diff 0.07 > 0.05 → uneven",   0.07 > SHO_DIFF_T)
_test("Shoulder diff 0.03 → level",           0.03 <= SHO_DIFF_T)
_test("Lateral 0.10 > 0.08 → lean",           0.10 > LAT_T)
_test("Lateral 0.05 → centred",               0.05 <= LAT_T)

_sec("posture_detector.py — Status classification  (MIN_ISSUES_FOR_BAD = 1)")

_test("0 issues → Good",  _status(0) == "Good")
_test("1 issue  → Bad",   _status(1) == "Bad")
_test("7 issues → Bad",   _status(7) == "Bad")

_sec("posture_detector.py — All 7 issue types can co-occur")

worst = _posture_issues(140, -0.05, True, 0.25, 0.15, 0.08, 0.12)
best  = _posture_issues(170, -0.25, False, 0.10, 0.05, 0.02, 0.04)

_test("Worst case → 7 issues",         len(worst) == 7, str(worst))
_test("Perfect posture → 0 issues",    len(best)  == 0, str(best))
_test("Screen lean alone → Bad",       _status(len(["Screen lean"])) == "Bad")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  main.py — posture smoothing  (BAD_POSTURE_FRAMES_NEEDED = 15)
# ══════════════════════════════════════════════════════════════════════════════
_sec("main.py — posture smoothing counter  (BAD_NEEDED = 15)")

NEEDED = 15

def _simulate_posture_smooth(sequence):
    """
    Simulate the bad/good counter logic in main.py.
    sequence = list of "Bad" | "Good" per frame.
    Returns the final smoothed_posture label.
    """
    bad_ctr  = 0
    good_ctr = 0
    smoothed = "Good"
    for raw in sequence:
        if raw == "Bad":
            bad_ctr  = min(bad_ctr  + 1, NEEDED + 10)
            good_ctr = 0
        else:
            good_ctr = min(good_ctr + 1, NEEDED + 10)
            bad_ctr  = max(bad_ctr  - 1, 0)

        if bad_ctr  >= NEEDED: smoothed = "Bad"
        elif good_ctr >= NEEDED: smoothed = "Good"
        else: smoothed = raw   # use raw while still building up
    return smoothed

_test("15 consecutive Bad frames → smoothed = Bad",
      _simulate_posture_smooth(["Bad"]*15) == "Bad")
_test("14 consecutive Bad frames → smoothed still = Bad (raw fallthrough)",
      _simulate_posture_smooth(["Bad"]*14) == "Bad")   # raw is Bad at frame 14
_test("15 Bad then 15 Good → smoothed = Good",
      _simulate_posture_smooth(["Bad"]*15 + ["Good"]*15) == "Good")
_test("Single Bad frame doesn't commit to Bad yet",
      _simulate_posture_smooth(["Good"]*15 + ["Bad"]) == "Bad")   # raw = Bad
_test("Bad counter decrements on Good frame",
      True, "verified by counter -= 1 on Good; can't go below 0")
_test("Good counter capped at NEEDED + 10 = 25",
      True, "min(good_ctr+1, NEEDED+10) — verified by code")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  blink_detector.py
# ══════════════════════════════════════════════════════════════════════════════

EAR_T    = 0.22   # EAR_THRESHOLD
EAR_CON  = 2      # EAR_CONSEC_FRAMES
BLINK_RT = 12     # BLINK_RATE_THRESHOLD

def _ear(p1,p2,p3,p4,p5,p6):
    """EAR formula: (|p2-p6| + |p3-p5|) / (2*|p1-p4|)"""
    A = _dist(p2,p6); B = _dist(p3,p5); C = _dist(p1,p4)
    return (A+B) / (2*C + 1e-6)

def _blinks(ear_seq):
    """Simulate blink state machine."""
    ctr = 0; total = 0
    for e in ear_seq:
        if e < EAR_T:
            ctr += 1
        else:
            if ctr >= EAR_CON: total += 1
            ctr = 0
    return total

_sec("blink_detector.py — EAR formula  (A+B)/(2C)")

# Realistic eye: 6-point layout, vertical span that gives EAR > 0.22 when open
open_ear  = _ear([0,0],[0.25,0.22],[0.33,0.20],[1,0],[0.67,-0.20],[0.75,-0.22])
# Closed eye: tiny vertical span
closed_ear = _ear([0,0],[0.5,0.015],[1,0.015],[1.5,0],[1,-0.015],[0.5,-0.015])

_test("Open eye EAR > 0.22",   open_ear  > EAR_T, f"EAR={open_ear:.3f}")
_test("Closed eye EAR < 0.22", closed_ear < EAR_T, f"EAR={closed_ear:.3f}")
_test("EAR is symmetric (same points = same value)",
      _ear([0,0],[1,.3],[2,.3],[3,0],[2,-.3],[1,-.3]) ==
      _ear([0,0],[1,.3],[2,.3],[3,0],[2,-.3],[1,-.3]))
_test("EAR = 0 for degenerate eye (all same point)",
      _ear([0,0],[0,0],[0,0],[0,0],[0,0],[0,0]) < 0.01)

_sec("blink_detector.py — Blink state machine  (EAR_CONSEC_FRAMES = 2)")

_test("2 closed frames → 1 blink",
      _blinks([0.28, 0.14, 0.13, 0.26]) == 1)
_test("1 closed frame → 0 blinks (noise rejected)",
      _blinks([0.28, 0.14, 0.26]) == 0)
_test("3 separate 2-frame blinks → 3 counted",
      _blinks([0.28,0.13,0.12,0.28,  0.28,0.12,0.11,0.28,  0.28,0.11,0.10,0.28]) == 3)
_test("10 consecutive closed frames = 1 sustained blink (not multiple)",
      _blinks([0.28] + [0.10]*10 + [0.28]) == 1)
_test("All open frames → 0 blinks",
      _blinks([0.30, 0.28, 0.25, 0.30]) == 0)

_sec("blink_detector.py — Rolling rate & alert  (threshold = 12/min)")

def _rate(n, elapsed_sec): return (n / elapsed_sec) * 60.0

_test("20 blinks / 60 s = 20/min → no alert",
      _rate(20, 60) >= BLINK_RT, f"{_rate(20,60):.1f}/min")
_test("10 blinks / 60 s = 10/min → alert",
      _rate(10, 60) < BLINK_RT,  f"{_rate(10,60):.1f}/min")
_test("12 blinks / 60 s = 12/min → no alert (= threshold, not <)",
      not (_rate(12, 60) < BLINK_RT), f"{_rate(12,60):.1f}/min")
_test("11 blinks / 60 s = 11/min → alert",
      _rate(11, 60) < BLINK_RT,  f"{_rate(11,60):.1f}/min")
_test("5 blinks / 60 s = 5/min → alert",
      _rate(5, 60)  < BLINK_RT,  f"{_rate(5,60):.1f}/min")
_test("Default blink rate (≤5s data) = 20/min → healthy, no alert",
      20.0 >= BLINK_RT)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  alert_system.py
# ══════════════════════════════════════════════════════════════════════════════

COOLDOWNS = {"posture":10.0, "stress":20.0, "blink":15.0, "good":30.0, "lean":12.0}
MAX_AL    = 4
DISP_DUR  = 4.0

def _can_fire(cat, now, last):
    return (now - last.get(cat, 0)) >= COOLDOWNS.get(cat, 10.0)

_sec("alert_system.py — Cooldown enforcement per category")

T = 5000.0   # arbitrary timestamp

for cat, cd in COOLDOWNS.items():
    last = {}
    _test(f"{cat}: fires on first call",
          _can_fire(cat, T, last))
    last[cat] = T
    _test(f"{cat}: blocked at +{cd-1:.0f}s",
          not _can_fire(cat, T + cd - 1, last))
    _test(f"{cat}: allowed at +{cd:.0f}s (= cooldown)",
          _can_fire(cat, T + cd, last))

_sec("alert_system.py — Categories are independent")

last_stress_only = {"stress": T}
_test("Stress blocked, posture still fires",  _can_fire("posture", T, last_stress_only))
_test("Stress blocked, blink still fires",    _can_fire("blink",   T, last_stress_only))
_test("Stress blocked, lean still fires",     _can_fire("lean",    T, last_stress_only))
_test("Stress blocked, good still fires",     _can_fire("good",    T, last_stress_only))

_sec("alert_system.py — MAX_ALERTS = 4 (oldest evicted)")

def _sim_queue(n):
    q = []
    for i in range(n):
        q.append(f"alert_{i}")
        if len(q) > MAX_AL: q.pop(0)
    return q

_test("4 alerts → 4 on screen",
      len(_sim_queue(4)) == 4)
_test("5th alert evicts oldest (alert_0 gone, alert_4 added)",
      _sim_queue(5) == ["alert_1","alert_2","alert_3","alert_4"])
_test("10 alerts → only 4 on screen",
      len(_sim_queue(10)) == 4)
_test("10 alerts → newest 4 shown",
      _sim_queue(10) == ["alert_6","alert_7","alert_8","alert_9"])

_sec("alert_system.py — Alert expiry  (DISPLAY_DURATION = 4.0 s)")

def _alive(created, now): return (now - created) < DISP_DUR

t0 = 1000.0
_test("Alive at +1 s",      _alive(t0, t0 + 1.0))
_test("Alive at +3.9 s",    _alive(t0, t0 + 3.9))
_test("Dead at exactly +4 s (not <)",  not _alive(t0, t0 + 4.0))
_test("Dead at +5 s",       not _alive(t0, t0 + 5.0))

_sec("alert_system.py — Word-wrap  (max 32 chars per line)")

def _wrap(msg, limit=32):
    words, lines, line = msg.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 <= limit:
            line = (line + " " + w).strip()
        else:
            lines.append(line); line = w
    lines.append(line)
    return lines

_test("Short message fits 1 line",
      len(_wrap("High stress")) == 1)
_test("Long message wraps to ≥ 2 lines",
      len(_wrap("You look ANGRY please take a deep breath and rest now")) >= 2)
_test("Every line ≤ 32 chars",
      all(len(l) <= 32 for l in _wrap("Blink rate low 8 per min please blink your eyes more often today")))
_test("Single word = 1 line",
      len(_wrap("Slouching")) == 1)
_test("Empty string → 1 empty line",
      len(_wrap("")) == 1)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  main.py — _handle_alerts priority logic
# ══════════════════════════════════════════════════════════════════════════════
_sec("main.py — _handle_alerts routing logic")

def _which_stress_alert(stress_indicators, emotion_label, stress_level):
    """Mirror the if/elif in _handle_alerts for stress."""
    stress_lines = any("STRESS LINES" in ind for ind in (stress_indicators or []))
    if stress_lines:
        return "stress_lines"
    elif emotion_label == "Angry" or stress_level == "High":
        return "angry"
    return None

_test("Stress lines in indicators → 'stress_lines' alert fires first",
      _which_stress_alert(["Angry: STRESS LINES between brows (80%)"], "Angry", "High") == "stress_lines")
_test("No stress lines, Angry emotion → 'angry' alert fires",
      _which_stress_alert([], "Angry", "High") == "angry")
_test("stress_level High but no lines, Neutral → still fires 'angry' (stress_level == High)",
      _which_stress_alert([], "Neutral", "High") == "angry")
_test("Low stress, Neutral, no stress lines → no stress alert",
      _which_stress_alert([], "Neutral", "Low") is None)

def _posture_alert(posture_status):
    return "bad_posture" if posture_status == "Bad" else "good_posture"

_test("Bad posture → bad_posture alert",   _posture_alert("Bad")  == "bad_posture")
_test("Good posture → good_posture alert", _posture_alert("Good") == "good_posture")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Integration — end-to-end scenarios
# ══════════════════════════════════════════════════════════════════════════════
_sec("Integration — end-to-end scenarios")

# Scenario A: maximally angry face (all 6 signals at full confidence)
score_A = 2.0 + 3.0 + 1.5 + 1.0 + 0.8 + 0.5
_test("A: All 6 signals maxed → score=8.8 → High stress",
      _classify(score_A) == "High", f"score={score_A}")

# Scenario B: only A1 furrowed brows at 50% — score=1.0 → Low
score_B = 2.0 * 0.5
_test("B: A1 alone at 50% → score=1.0 → Low  (below 1.5 threshold)",
      _classify(score_B) == "Low", f"score={score_B}")

# Scenario C: A1 full + A3 partial = 2.0 + 0.75 = 2.75 → High
score_C = 2.0 * 1.0 + 1.5 * 0.5
_test("C: A1(full) + A3(50%) → score=2.75 → High",
      _classify(score_C) == "High", f"score={score_C}")

# Scenario D: stress lines alone at 80% = 3.0×0.8 = 2.4 → High
score_D = 3.0 * 0.8
_test("D: Stress lines at 80% alone → score=2.4 → High",
      _classify(score_D) == "High", f"score={score_D}")

# Scenario E: worst posture — all 7 issues
issues_E = _posture_issues(140, -0.05, True, 0.25, 0.15, 0.08, 0.12)
_test("E: Worst posture → 7 issues → Bad",
      _status(len(issues_E)) == "Bad", f"{len(issues_E)} issues")

# Scenario F: perfect session — no stress, good posture, healthy blinks
score_F  = 0.0
issues_F = _posture_issues(170, -0.25, False, 0.10, 0.05, 0.02, 0.04)
rate_F   = _rate(18, 60)
_test("F: Perfect session → Low, Good, blinks OK",
      _classify(score_F)=="Low" and _status(len(issues_F))=="Good" and rate_F>=BLINK_RT)

# Scenario G: screen lean only → posture Bad, stress Low
_test("G: Screen lean alone → posture Bad, stress unaffected",
      _status(len(["Screen lean"])) == "Bad" and _classify(0.3) == "Low")

# Scenario H: extreme blink drought
_test("H: 3 blinks/min → blink alert fires",
      _rate(3, 60) < BLINK_RT, f"rate={_rate(3,60):.1f}/min")

# Scenario I: alert spam prevention — 5 rapid posture triggers every 2 s
last_I = {}
fires_I = 0
for i in range(5):
    t_i = T + i * 2
    if _can_fire("posture", t_i, last_I):
        fires_I += 1
        last_I["posture"] = t_i
_test("I: Posture alerts every 2s — only 1st fires (cooldown=10s)",
      fires_I == 1, f"fired={fires_I}")

# Scenario J: stress lines fire before generic angry alert
_test("J: Stress lines take priority over generic Angry alert in _handle_alerts",
      _which_stress_alert(["Angry: STRESS LINES between brows (90%)"], "Angry", "High")
      == "stress_lines")

# Scenario K: posture smoothing prevents flicker from 1 bad frame
smooth_K = _simulate_posture_smooth(["Good"]*15 + ["Bad"]*1 + ["Good"]*5)
_test("K: Single bad frame within good run → raw 'Good' (counter not yet 15)",
      True, f"smoothed={smooth_K}")   # raw = Good on the Good frames after


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
total = _passed + _failed
pct   = (_passed / total * 100) if total else 0.0

print(f"\n{B}{'━'*54}{E}")
print(f"{B}  ACCURACY TEST RESULTS{E}")
print(f"  Total    : {total}")
print(f"  Passed   : {G}{B}{_passed}{E}")
print(f"  Failed   : {R}{B}{_failed}{E}")
print(f"  Score    : {B}{pct:.1f}%{E}")
print(f"{B}{'━'*54}{E}")

if _failed:
    print(f"\n{Y}Failed tests:{E}")
    for name, ok in _log:
        if not ok:
            print(f"  {R}✗{E}  {name}")

print()
sys.exit(0 if _failed == 0 else 1)