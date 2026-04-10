"""
Interactive ROI calibration tool.
Drag the 4 green corners to fit the queue area, then press 'q' to print points.
Press 'r' to reset back to the current config.
"""
import cv2
import numpy as np
from config import CAMERA_INDEX, QUEUE_ROI
 
HANDLE_RADIUS = 12
points = np.array(QUEUE_ROI, dtype=np.int32).tolist()
selected_point = None
 
 
def click_handler(event, x, y, flags, param):
    global selected_point

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = find_nearest_point(x, y)
    elif event == cv2.EVENT_MOUSEMOVE and selected_point is not None:
        points[selected_point] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_point = None


def find_nearest_point(x, y):
    for i, (px, py) in enumerate(points):
        if ((px - x) ** 2 + (py - y) ** 2) ** 0.5 <= HANDLE_RADIUS:
            return i
    return None
 
 
def main():
    global points

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print('ERROR: Cannot open camera. Check CAMERA_INDEX in config.py.')
        return
 
    print('=== ROI Calibration Tool ===')
    print('Drag the 4 green corners until the box matches the queue area.')
    print('Draw TIGHT around just the standing-in-line area.')
    print('Do NOT include the pickup counter or walkways.')
    print()
    print('Controls: drag corners = adjust, r = reset, q = done')
    print()
 
    cv2.namedWindow('ROI Tool')
    cv2.setMouseCallback('ROI Tool', click_handler)
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        roi_points = np.array(points, np.int32)
        cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)

        for i, pt in enumerate(points):
            cv2.circle(frame, tuple(pt), 6, (0, 255, 0), -1)
            cv2.putText(frame, str(i + 1), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
        cv2.putText(frame, 'Drag corners | q=done r=reset',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('ROI Tool', frame)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points = np.array(QUEUE_ROI, dtype=np.int32).tolist()
            print('Points reset.')
 
    cap.release()
    cv2.destroyAllWindows()
 
    print()
    print('=' * 50)
    print('COPY THIS INTO config.py AS QUEUE_ROI:')
    print('=' * 50)
    print('QUEUE_ROI = np.array([')
    for pt in points:
        print(f'    [{pt[0]}, {pt[1]}],')
    print('], dtype=np.int32)')
 
 
if __name__ == '__main__':
    main()
