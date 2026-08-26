import cv2
from picamera2 import Picamera2
from datetime import datetime

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (640, 480)}
))
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cv2.putText(frame, "Team Name", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        height, width = frame.shape[:2]
        cv2.putText(frame, f"{width} x {height}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, timestamp, (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        hsv_display = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        combined = cv2.hconcat([frame, hsv_display])

        cv2.imshow("Day 3 Annotated", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite(
                '/home/lhsengr09/Hallway-traffic-counter/cv_project/images/snapshot.jpg',
                frame
            )
            print("Snapshot saved!")

        elif key == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
