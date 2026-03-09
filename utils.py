"""
utils.py
--------
Shared mathematical utility functions used across all detector modules.
"""

import numpy as np


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate the angle (in degrees) at point B formed by the vectors BA and BC.

    Args:
        a, b, c – 2-D or 3-D numpy arrays representing points.

    Returns:
        Angle in degrees [0, 180].
    """
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Euclidean distance between two points.

    Args:
        p1, p2 – numpy arrays of any dimension.

    Returns:
        Scalar distance.
    """
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def draw_rounded_rect(img: np.ndarray, pt1: tuple, pt2: tuple,
                      colour: tuple, radius: int = 10,
                      thickness: int = -1) -> None:
    """
    Draw a rectangle with rounded corners on *img* in-place.

    Args:
        img       – BGR frame.
        pt1       – top-left (x, y).
        pt2       – bottom-right (x, y).
        colour    – BGR colour tuple.
        radius    – corner radius in pixels.
        thickness – -1 fills the rectangle.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    # Straight segments
    cv2_import()
    import cv2
    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), colour, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), colour, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), colour, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), colour, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), colour, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), colour, thickness)

    # Corner arcs
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90,  colour, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90,  colour, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90,  colour, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90,  colour, thickness)


def cv2_import():
    """Lazy import guard (avoids circular imports)."""
    pass


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def moving_average(history: list, new_val: float, max_len: int = 30) -> tuple[list, float]:
    """
    Append new_val to history, trim to max_len, return (history, mean).
    """
    history.append(new_val)
    if len(history) > max_len:
        history.pop(0)
    return history, float(np.mean(history))
