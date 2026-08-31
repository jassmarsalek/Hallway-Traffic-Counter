import cv2
import numpy as np
from picamera2 import Picamera2

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

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur the image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Find edges
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Shape counters
        triangles = 0
        rectangles = 0
        circles = 0

        for contour in contours:

            # Ignore tiny objects/noise
            area = cv2.contourArea(contour)

            if area < 1000:
                continue

            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)

            # Approximate the contour
            approx = cv2.approxPolyDP(
                contour,
                0.04 * perimeter,
                True
            )

            # Number of corners
            corners = len(approx)

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)

            # Classify shape
            if corners == 3:
                shape = "Triangle"
                triangles += 1

            elif corners == 4:
                shape = "Rectangle"
                rectangles += 1

            elif corners > 6:
                shape = "Circle"
                circles += 1

            else:
                shape = "Unknown"

            # Draw contour
            cv2.drawContours(
                frame,
                [approx],
                -1,
                (0, 255, 0),
                2
            )

            # Display shape name
            cv2.putText(
                frame,
                shape,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Display running totals
        cv2.putText(
            frame,
            f"Triangles: {triangles}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Rectangles: {rectangles}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Circles: {circles}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Show camera
        cv2.imshow("Shape Detection", frame)

        # Press q to quit
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
