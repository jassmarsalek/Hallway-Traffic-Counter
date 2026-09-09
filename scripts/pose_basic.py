import cv2
import mediapipe as mp
from picamera2 import Picamera2

# Set up MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Set up the camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# Start pose detection
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Detect body pose
        results = pose.process(frame)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # Display frame
        cv2.imshow("MediaPipe Pose - Press Q to Quit", frame)

        # Quit when Q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

picam2.stop()
cv2.destroyAllWindows()
