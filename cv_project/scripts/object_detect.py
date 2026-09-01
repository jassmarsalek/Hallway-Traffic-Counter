import cv2
import numpy as np
from picamera2 import Picamera2

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH = "/home/lhsengr09/Hallway-Traffic-Counter/cv_project/models/detect.tflite"
LABEL_PATH = "/home/lhsengr09/Hallway-Traffic-Counter/cv_project/models/labelmap.txt"

CONFIDENCE_THRESHOLD = 0.7

with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]["shape"][1]
input_width = input_details[0]["shape"][2]

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (640, 480)}
))
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frame_h, frame_w, _ = frame.shape

        resized = cv2.resize(frame, (input_width, input_height))
        input_data = np.expand_dims(resized, axis=0)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        classes = interpreter.get_tensor(output_details[1]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]["index"])[0]

        for i in range(len(scores)):
            if scores[i] >= CONFIDENCE_THRESHOLD:
                ymin, xmin, ymax, xmax = boxes[i]

                x1 = int(xmin * frame_w)
                y1 = int(ymin * frame_h)
                x2 = int(xmax * frame_w)
                y2 = int(ymax * frame_h)

                class_id = int(classes[i])

                if class_id < len(labels):
                    label = labels[class_id]
                else:
                    label = "Unknown"

                confidence = int(scores[i] * 100)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    f"{label}: {confidence}%",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
