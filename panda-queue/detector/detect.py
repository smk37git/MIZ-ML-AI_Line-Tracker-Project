# Updated File
"""
Main detection script. Ties together YOLO, counting, and sending.
Usage: python detect.py
"""
import cv2
import time
from ultralytics import YOLO
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    MODEL_PATH, CONFIDENCE_THRESHOLD, PERSON_CLASS_ID,
    QUEUE_ROI, SHOW_PREVIEW, PREVIEW_SCALE, SERVICE_RATE
)
from counter import QueueCounter
from sender import DataSender
 
 
def main():
    # Load model
    print('Loading YOLOv8 model...')
    model = YOLO(MODEL_PATH)
    print(f'Model loaded on: {model.device}')
    print(f'Service rate: {SERVICE_RATE} sec/person')
 
    counter = QueueCounter()
    sender = DataSender()
 
    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
 
    if not cap.isOpened():
        print('ERROR: Cannot open camera.')
        print('  Try changing CAMERA_INDEX in config.py (0, 1, or 2)')
        return
 
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Camera: {actual_w}x{actual_h}')
    print(f'ROI: {len(QUEUE_ROI)} points')
    print(f'Preview: {"ON" if SHOW_PREVIEW else "OFF"}')
    print('Starting detection. Press Ctrl+C to stop (or q in preview window).')
    print()
 
    fps_time = time.time()
    frame_count = 0
 
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('WARNING: Frame read failed, retrying...')
                time.sleep(0.1)
                continue
 
            # YOLO detection + tracking
            results = model.track(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                persist=True,
                verbose=False,
            )
 
            # Count people in queue (with speed filtering)
            data = counter.update(results[0].boxes)
 
            # Send to Django
            sender.maybe_send(data)
 
            # FPS logging (every 2 seconds)
            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                print(f'[DETECT] {data["people_count"]} in line | '
                      f'{data["estimated_wait_min"]} min wait | '
                      f'{fps:.1f} FPS')
                frame_count = 0
                fps_time = time.time()
 
            # Preview window
            if SHOW_PREVIEW:
                display = frame.copy()
 
                # Draw ROI
                cv2.polylines(display, [QUEUE_ROI], True, (0, 255, 0), 2)
 
                # Draw detected people
                for (x1, y1, x2, y2, tid, in_line) in data['boxes_in_roi']:
                    color = (0, 255, 0) if in_line else (0, 0, 255)
                    label = f'ID:{tid}' + (' [LINE]' if in_line else ' [MOVING]')
                    cv2.rectangle(display, (int(x1), int(y1)),
                                  (int(x2), int(y2)), color, 2)
                    cv2.putText(display, label, (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
                # Stats overlay
                cv2.putText(display,
                    f'In Line: {data["people_count"]} | '
                    f'Wait: {data["estimated_wait_min"]} min',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
 
                if PREVIEW_SCALE != 1.0:
                    h, w = display.shape[:2]
                    display = cv2.resize(display,
                        (int(w * PREVIEW_SCALE), int(h * PREVIEW_SCALE)))
 
                cv2.imshow('Panda Queue Detector', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
 
    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        print('Done.')
 
 
if __name__ == '__main__':
    main()
