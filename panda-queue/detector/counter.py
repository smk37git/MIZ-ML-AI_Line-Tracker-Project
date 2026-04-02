"""
Queue counter with speed-based filtering.
 
How it works:
1. YOLO detects people. BoT-SORT assigns persistent track IDs.
2. For each tracked person inside the ROI, we record their position each frame.
3. We calculate their average speed over the last N frames.
4. If speed < threshold, they're standing in line. Count them.
5. If speed > threshold, they're walking through. Ignore them.
6. Wait time = count of stationary people x calibrated service rate.
"""
import time
import cv2
import numpy as np
from config import (
    QUEUE_ROI, SPEED_THRESHOLD, SPEED_HISTORY_FRAMES,
    MIN_FRAMES_TO_COUNT, SERVICE_RATE
)
 
 
class QueueCounter:
 
    def __init__(self):
        # track_id -> list of (cx, cy, timestamp)
        self.position_history = {}
        self.people_count = 0
        self.estimated_wait_sec = 0.0
 
    def update(self, boxes):
        """
        Process tracked detections for one frame.
 
        Args:
            boxes: results[0].boxes from model.track()
 
        Returns:
            dict with people_count, estimated_wait_sec/min,
            and lists of boxes for display
        """
        now = time.time()
        current_ids = set()
        boxes_in_roi = []        # For drawing: (x1,y1,x2,y2,tid,in_line)
        stationary_count = 0
 
        for box in boxes:
            if box.id is None:
                continue
 
            track_id = int(box.id[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
 
            # Check if person is inside the queue ROI
            if cv2.pointPolygonTest(QUEUE_ROI, (cx, cy), False) < 0:
                continue  # Outside ROI, skip entirely
 
            current_ids.add(track_id)
 
            # Record position history
            if track_id not in self.position_history:
                self.position_history[track_id] = []
            self.position_history[track_id].append((cx, cy, now))
 
            # Keep only recent history
            hist = self.position_history[track_id]
            if len(hist) > SPEED_HISTORY_FRAMES:
                self.position_history[track_id] = hist[-SPEED_HISTORY_FRAMES:]
                hist = self.position_history[track_id]
 
            # Need enough frames to calculate speed reliably
            if len(hist) < MIN_FRAMES_TO_COUNT:
                boxes_in_roi.append((x1, y1, x2, y2, track_id, False))
                continue  # Not enough data yet, don't count
 
            # Calculate average speed (pixels per second)
            dx = hist[-1][0] - hist[0][0]
            dy = hist[-1][1] - hist[0][1]
            dt = hist[-1][2] - hist[0][2]
            speed = (dx**2 + dy**2)**0.5 / max(dt, 0.01)
 
            if speed < SPEED_THRESHOLD:
                stationary_count += 1
                boxes_in_roi.append((x1, y1, x2, y2, track_id, True))
            else:
                boxes_in_roi.append((x1, y1, x2, y2, track_id, False))
 
        # Clean up stale tracks (not seen this frame)
        stale = [tid for tid in self.position_history if tid not in current_ids]
        for tid in stale:
            del self.position_history[tid]
 
        # Calculate wait time
        self.people_count = stationary_count
        self.estimated_wait_sec = stationary_count * SERVICE_RATE
 
        return {
            'people_count': self.people_count,
            'estimated_wait_sec': round(self.estimated_wait_sec, 1),
            'estimated_wait_min': round(self.estimated_wait_sec / 60, 1),
            'service_rate': SERVICE_RATE,
            'boxes_in_roi': boxes_in_roi,  # For visualization
        }

