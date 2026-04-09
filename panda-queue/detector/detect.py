"""
Main detection script. Ties together YOLO, counting, and sending.
Usage: python detect.py
"""
import cv2
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from ultralytics import YOLO
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    MODEL_PATH, CONFIDENCE_THRESHOLD, PERSON_CLASS_ID,
    QUEUE_ROI, SHOW_PREVIEW, PREVIEW_SCALE, SERVICE_RATE,
    STREAM_PORT, STREAM_FPS
)
from counter import QueueCounter
from sender import DataSender
from frame_buffer import FrameBuffer

# Global frame buffer shared between detector loop and MJPEG server
frame_buffer = FrameBuffer()


class MJPEGHandler(BaseHTTPRequestHandler):
    """Serves the latest annotated frame as a multipart MJPEG stream."""

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        try:
            while True:
                jpeg = frame_buffer.wait_for_frame(timeout=2.0)
                if jpeg:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpeg)}\r\n'.encode())
                    self.wfile.write(b'\r\n')
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass  # suppress per-request logs


def start_stream_server():
    """Run the MJPEG HTTP server in a daemon thread."""
    server = HTTPServer(('0.0.0.0', STREAM_PORT), MJPEGHandler)
    print(f'MJPEG stream server running on http://0.0.0.0:{STREAM_PORT}/')
    server.serve_forever()


def load_model_async(result):
    """Load YOLO model in a background thread."""
    print('Loading YOLOv8 model...')
    result['model'] = YOLO(MODEL_PATH)
    print(f'Model loaded on: {result["model"].device}')
    result['ready'] = True


def main():
    # Start MJPEG stream server first
    stream_thread = threading.Thread(target=start_stream_server, daemon=True)
    stream_thread.start()

    # Open camera immediately so we can stream raw frames while model loads
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

    # Start loading YOLO model in background
    model_result = {'model': None, 'ready': False}
    model_thread = threading.Thread(target=load_model_async, args=(model_result,))
    model_thread.start()

    counter = QueueCounter()
    sender = DataSender()

    print(f'Service rate: {SERVICE_RATE} sec/person')
    print(f'ROI: {len(QUEUE_ROI)} points')
    print(f'Preview: {"ON" if SHOW_PREVIEW else "OFF"}')
    print(f'MJPEG stream: http://localhost:{STREAM_PORT}/')
    print('Streaming camera while model loads...')
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

            if model_result['ready']:
                # --- Full detection mode ---
                model = model_result['model']
                results = model.track(
                    frame,
                    classes=[PERSON_CLASS_ID],
                    conf=CONFIDENCE_THRESHOLD,
                    persist=True,
                    verbose=False,
                )
                data = counter.update(results[0].boxes)
                sender.maybe_send(data)

                # FPS logging
                frame_count += 1
                elapsed = time.time() - fps_time
                if elapsed >= 2.0:
                    fps = frame_count / elapsed
                    print(f'[DETECT] {data["people_count"]} in line | '
                          f'{data["estimated_wait_min"]} min wait | '
                          f'{fps:.1f} FPS')
                    frame_count = 0
                    fps_time = time.time()

                # Build annotated frame
                display = frame.copy()
                cv2.polylines(display, [QUEUE_ROI], True, (0, 255, 0), 2)
                for (x1, y1, x2, y2, tid, in_line) in data['boxes_in_roi']:
                    color = (0, 255, 0) if in_line else (0, 0, 255)
                    label = f'ID:{tid}' + (' [LINE]' if in_line else ' [MOVING]')
                    cv2.rectangle(display, (int(x1), int(y1)),
                                  (int(x2), int(y2)), color, 2)
                    cv2.putText(display, label, (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(display,
                    f'In Line: {data["people_count"]} | '
                    f'Wait: {data["estimated_wait_min"]} min',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # --- Loading mode: stream raw camera with overlay ---
                display = frame.copy()
                cv2.polylines(display, [QUEUE_ROI], True, (0, 255, 0), 2)
                cv2.putText(display, 'Loading model...',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            # Push frame to MJPEG stream
            frame_buffer.update(display)

            # Optional OpenCV preview window
            if SHOW_PREVIEW:
                preview = display
                if PREVIEW_SCALE != 1.0:
                    h, w = display.shape[:2]
                    preview = cv2.resize(display,
                        (int(w * PREVIEW_SCALE), int(h * PREVIEW_SCALE)))
                cv2.imshow('Panda Queue Detector', preview)
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
