import cv2
import numpy as np
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

# Model and label paths
MODEL_PATH = "../models/detect.tflite"
LABELS_PATH = "../models/labelmap.txt"

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load TensorFlow Lite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]["shape"][1]
input_width = input_details[0]["shape"][2]

# Set up camera
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        h, w, _ = frame.shape

        # Resize frame for the TFLite model
        resized = cv2.resize(frame, (input_width, input_height))
        input_data = np.expand_dims(resized, axis=0)

        # Run object detection
        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )
        interpreter.invoke()

        boxes = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        classes = interpreter.get_tensor(
            output_details[1]["index"]
        )[0]

        scores = interpreter.get_tensor(
            output_details[2]["index"]
        )[0]

        # Count all detected objects
        objects_detected = 0

        for i in range(len(scores)):
            if scores[i] < CONFIDENCE_THRESHOLD:
                continue

            # Get detected object label
            class_id = int(classes[i])

            if class_id >= len(labels):
                continue

            label = labels[class_id]

            objects_detected += 1

            # Get bounding box
            ymin, xmin, ymax, xmax = boxes[i]

            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display label and confidence
            cv2.putText(
                frame,
                f"{label} {scores[i]:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Display number of detected objects
        cv2.putText(
            frame,
            f"Objects Detected: {objects_detected}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # Show camera feed
        cv2.imshow(
            "Hallway Traffic Counter - Detection Test",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
