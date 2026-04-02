"""
All configuration for the queue detector.
Change these values when moving from laptop to Jetson.
"""
import numpy as np
 
# ============ CAMERA ============
CAMERA_INDEX = 0                     # 0 = default webcam. Try 1 if external.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
 
# ============ MODEL ============
MODEL_PATH = 'yolov8n.pt'           # Change to 'yolov8n.engine' on Jetson
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0                  # COCO class 0 = person
 
# ============ QUEUE ROI ============
# Pixel coordinates for the queue area polygon.
# Run roi_tool.py to determine these for your camera position.
# Draw TIGHT around just the standing-in-line area.
# Do NOT include the pickup counter or walkways.
QUEUE_ROI = np.array([
    [200, 150],
    [900, 150],
    [900, 600],
    [200, 600],
], dtype=np.int32)
 
# ============ SPEED FILTER ============
# People standing in line are nearly stationary.
# People walking through move fast. Filter them out.
SPEED_THRESHOLD = 50.0               # pixels/second. Below this = in line.
SPEED_HISTORY_FRAMES = 10            # Frames of position history to average.
MIN_FRAMES_TO_COUNT = 8              # Must be seen this many frames before counting.
 
# ============ WAIT TIME ============
# Manually calibrated: go time 15-20 people at Panda Express,
# measure total time from joining line to getting food,
# divide by number of people. Update this number weekly.
SERVICE_RATE = 50.0                  # Seconds per person (calibrate this!)
 
# ============ BACKEND ============
BACKEND_URL = 'http://127.0.0.1:8000/api/update/'
SEND_INTERVAL = 10                   # Seconds between sends
 
# ============ DISPLAY ============
SHOW_PREVIEW = True                  # OpenCV window. Set False on Jetson.
PREVIEW_SCALE = 0.75

