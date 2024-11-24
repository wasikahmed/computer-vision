import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os

# Import required classes from MediaPipe
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to the model
model_path = os.path.join(os.getcwd(), 'models/blaze_face_short_range.tflite')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Global variables for results
display_text = ""
face_detected = False
bounding_boxes = []
face_landmarks = []
confidence_scores = []

def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    """
    Callback to process the face detection results.
    """
    global display_text, face_detected, bounding_boxes, face_landmarks, confidence_scores
    display_text = ""
    face_detected = False
    bounding_boxes = []
    face_landmarks = []
    confidence_scores = []

    if result.detections:
        face_detected = True
        for i, detection in enumerate(result.detections):
            # Confidence score of detection
            score = detection.categories[0].score
            confidence_scores.append(score)

            face_text = f"Face {i + 1}: Confidence ({score:.2f})"
            display_text += face_text + "\n"

            # Extract bounding box coordinates
            bbox = detection.bounding_box
            x_min = int(bbox.origin_x)
            y_min = int(bbox.origin_y)
            x_max = int(bbox.origin_x + bbox.width)
            y_max = int(bbox.origin_y + bbox.height)
            bounding_boxes.append((x_min, y_min, x_max, y_max))

            # Save landmarks for drawing
            if detection.keypoints:
                landmarks = [
                    (int(kp.x * output_image.width), int(kp.y * output_image.height)) 
                    for kp in detection.keypoints
                ]
                face_landmarks.append(landmarks)

# Configure MediaPipe FaceDetector
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    min_detection_confidence=0.5,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Set up webcam for capturing video
cap = cv2.VideoCapture(0)

# Create FaceDetector instance
with FaceDetector.create_from_options(options) as detector:
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Convert the frame to a MediaPipe Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Get a monotonically increasing timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Process the image
        detector.detect_async(mp_image, timestamp_ms)

        # Display the results
        if face_detected:
            for i, box in enumerate(bounding_boxes):
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                # Add confidence text next to the bounding box
                confidence_text = f"Conf: {confidence_scores[i]:.2f}"
                cv2.putText(frame, confidence_text, (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            for landmarks in face_landmarks:
                for landmark in landmarks:
                    cv2.circle(frame, landmark, 2, (0, 0, 255), -1)

        # Display total faces detected
        total_faces_text = f"Faces Detected: {len(bounding_boxes)}"
        cv2.putText(frame, total_faces_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('MediaPipe Face Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
