import cv2
import numpy as np
from picamera2 import Picamera2

# HSV values that work with your object
LOWER = np.array([1, 100, 100])
UPPER = np.array([179, 255, 255])

# Start the camera
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (640, 480)}
    )
)
picam2.start()

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Convert camera image to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create color mask
        mask = cv2.inRange(hsv, LOWER, UPPER)

        # Remove small noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Find the largest detected object
            largest = max(contours, key=cv2.contourArea)

            # Ignore tiny objects
            if cv2.contourArea(largest) > 300:

                # Find enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(largest)

                # Calculate center
                M = cv2.moments(largest)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Green circle around object
                    cv2.circle(
                        frame,
                        (int(x), int(y)),
                        int(radius),
                        (0, 255, 0),
                        2
                    )

                    # Red center point
                    cv2.circle(
                        frame,
                        (cx, cy),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    # Show coordinates
                    cv2.putText(
                        frame,
                        f"({cx},{cy})",
                        (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

        # Show camera
        cv2.imshow("Color Tracking", frame)

        # Press q to quit
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
