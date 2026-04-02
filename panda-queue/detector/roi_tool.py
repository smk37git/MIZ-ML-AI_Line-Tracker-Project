"""
Interactive ROI calibration tool.
Click 4+ corners of the queue area, press 'q' to finish.
Press 'r' to reset, 'q' to quit and print coordinates.
"""
import cv2
import numpy as np
from config import CAMERA_INDEX
 
points = []
 
 
def click_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f'  Point {len(points)}: [{x}, {y}]')
 
 
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print('ERROR: Cannot open camera. Check CAMERA_INDEX in config.py.')
        return
 
    print('=== ROI Calibration Tool ===')
    print('Click the corners of the queue area (at least 4 points).')
    print('Draw TIGHT around just the standing-in-line area.')
    print('Do NOT include the pickup counter or walkways.')
    print()
    print('Controls: click = add point, r = reset, q = done')
    print()
 
    cv2.namedWindow('ROI Tool')
    cv2.setMouseCallback('ROI Tool', click_handler)
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        for i, pt in enumerate(points):
            cv2.circle(frame, tuple(pt), 6, (0, 255, 0), -1)
            cv2.putText(frame, str(i + 1), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points, np.int32)], True, (0, 255, 0), 2)
 
        cv2.putText(frame, f'Points: {len(points)} | q=done r=reset',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('ROI Tool', frame)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and len(points) >= 3:
            break
        elif key == ord('r'):
            points.clear()
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
