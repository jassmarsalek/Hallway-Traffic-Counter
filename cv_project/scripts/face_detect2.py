import cv2
import mediapipe as mp
from picamera2 import Picamera2

# Set up MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Set up PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (640, 480)}
))
picam2.start()

try:
    while True:
        # Capture frame from PiCamera2
        frame = picam2.capture_array()
        
        # Picamera2 outputs BGRA; convert to BGR for display and RGB for MediaPipe
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)

        # Draw the face mesh grid if faces are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        # Display output window
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
