"""
Thread-safe frame buffer for MJPEG streaming.
The detector writes annotated frames here; the MJPEG server reads them.
"""
import threading
import cv2


class FrameBuffer:
    """Holds the latest JPEG-encoded annotated frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None  # raw JPEG bytes
        self._event = threading.Event()

    def update(self, cv_frame):
        """Encode an OpenCV BGR frame to JPEG and store it."""
        _, jpeg = cv2.imencode('.jpg', cv_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with self._lock:
            self._frame = jpeg.tobytes()
        self._event.set()

    def get_jpeg(self):
        """Return the latest JPEG bytes, or None if no frame yet."""
        with self._lock:
            return self._frame

    def wait_for_frame(self, timeout=1.0):
        """Block until a new frame is available, then return it."""
        self._event.wait(timeout=timeout)
        self._event.clear()
        with self._lock:
            return self._frame
