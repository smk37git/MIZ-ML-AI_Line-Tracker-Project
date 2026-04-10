"""
Queue ROI helpers for the adjustable straight-line detection box.
"""
import cv2
import numpy as np
from config import QUEUE_ROI

_queue_roi_points = np.array(QUEUE_ROI, dtype=np.int32)


def get_queue_roi_points():
    """Return the configured queue box corner points."""
    return _queue_roi_points.copy()


def set_queue_roi_point(index, x, y):
    """Move one queue box corner point."""
    _queue_roi_points[index] = [int(x), int(y)]


def reset_queue_roi_points():
    """Reset the live queue box back to config.py values."""
    global _queue_roi_points
    _queue_roi_points = np.array(QUEUE_ROI, dtype=np.int32)


def point_inside_queue_roi(point):
    """Return True if an (x, y) point is inside the queue box."""
    return cv2.pointPolygonTest(get_queue_roi_points(), point, False) >= 0


def draw_queue_roi(frame, color=(0, 255, 0), thickness=2):
    """Draw the queue box on a frame."""
    points = get_queue_roi_points()
    cv2.polylines(frame, [points], True, color, thickness)
    for i, point in enumerate(points):
        x, y = map(int, point)
        cv2.circle(frame, (x, y), 7, color, -1)
        cv2.putText(frame, str(i + 1), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
