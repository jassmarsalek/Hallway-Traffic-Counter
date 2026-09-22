import cv2
import mediapipe as mp
import math
from picamera2 import Picamera2


# Calculate the angle formed by three points
def calculate_angle(a, b, c):
    # Create vectors from point B
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])

    # Dot product
    dot_product = ab[0] * cb[0] + ab[1] * cb[1]

    # Magnitudes
    magnitude_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    magnitude_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    # Prevent division by zero
    if magnitude_ab == 0 or magnitude_cb == 0:
        return 0

    # Calculate cosine
    cosine_angle = dot_product / (magnitude_ab * magnitude_cb)

    # Prevent rounding errors
    cosine_angle = max(-1, min(1, cosine_angle))

    # Convert to degrees
    return math.degrees(math.acos(cosine_angle))


# Set up MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Set up Raspberry Pi camera
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)

picam2.configure(config)
picam2.start()


# Rep counter variables
counter = 0
stage = None


# Start MediaPipe Pose
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:

        # Capture camera frame
        frame = picam2.capture_array()

        # Process frame with MediaPipe
        results = pose.process(frame)

        # Check for detected pose
        if results.pose_landmarks:

            # Get frame size
            height, width, _ = frame.shape

            # Get right arm landmarks
            shoulder_landmark = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]

            elbow_landmark = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_ELBOW
            ]

            wrist_landmark = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]

            # Convert landmarks to pixel coordinates
            shoulder = (
                int(shoulder_landmark.x * width),
                int(shoulder_landmark.y * height)
            )

            elbow = (
                int(elbow_landmark.x * width),
                int(elbow_landmark.y * height)
            )

            wrist = (
                int(wrist_landmark.x * width),
                int(wrist_landmark.y * height)
            )

            # Calculate elbow angle
            angle = calculate_angle(
                shoulder,
                elbow,
                wrist
            )

            # -------------------------
            # REP COUNTER LOGIC
            # -------------------------

            # Arm is bent
            if angle < 90:
                stage = "down"

            # Arm becomes straight after being bent
            if angle > 160 and stage == "down":
                stage = "up"
                counter += 1

            # Draw pose skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Show elbow angle
            cv2.putText(
                frame,
                f"Angle: {int(angle)}",
                (elbow[0], elbow[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Display rep count
        cv2.putText(
            frame,
            f"Reps: {counter}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        # Display current stage
        cv2.putText(
            frame,
            f"Stage: {stage}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Show camera feed
        cv2.imshow(
            "Day 10 - Rep Counter (Press Q to Quit)",
            frame
        )

        # Quit with Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# Clean up
picam2.stop()
cv2.destroyAllWindows()
