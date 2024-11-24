import cv2
import mediapipe as mp
import time
import os

# Import required classes from MediaPipe
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand connections for skeletal structure
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky finger
    (0, 17)  # Palm
]

# Global variable to store recognized gestures for display
display_text = ""
gesture_detected = False  # Track if any gesture is detected
bounding_boxes = []  # Store bounding box coordinates for detected hands
hand_landmarks = []  # Store landmarks for detected hands

# Callback function to handle gesture recognition results
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global display_text, gesture_detected, bounding_boxes, hand_landmarks
    display_text = ""
    gesture_detected = False
    bounding_boxes = []  # Reset bounding boxes for each frame
    hand_landmarks = []  # Reset landmarks for each frame

    if result.gestures:
        for i, gesture in enumerate(result.gestures):
            # Add detected gesture information
            gesture_text = f"Hand {i + 1}: {gesture[0].category_name} ({gesture[0].score:.2f})"
            display_text += gesture_text + "\n"
            gesture_detected = True

        # Extract bounding box information and landmarks
        for hand_landmark in result.hand_landmarks:
            # Bounding box coordinates
            x_min = min([lm.x for lm in hand_landmark]) * output_image.width
            y_min = min([lm.y for lm in hand_landmark]) * output_image.height
            x_max = max([lm.x for lm in hand_landmark]) * output_image.width
            y_max = max([lm.y for lm in hand_landmark]) * output_image.height
            bounding_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))

            # Save landmarks for drawing
            hand_landmarks.append([(int(lm.x * output_image.width), int(lm.y * output_image.height)) for lm in hand_landmark])

model_path = os.path.join(os.getcwd(), 'models/gesture_recognizer.task')
print(f"Model path: {model_path}")

# Check if the file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file was not found at {model_path}")

# Configure GestureRecognizerOptions
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Start webcam capture
cap = cv2.VideoCapture(0)

# Create GestureRecognizer instance
with GestureRecognizer.create_from_options(options) as recognizer:
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

        # Perform gesture recognition asynchronously
        recognizer.recognize_async(mp_image, timestamp_ms)

        # Overlay recognized gestures and draw borders/landmarks on the video feed
        if display_text:
            y0, dy = 30, 30
            for i, line in enumerate(display_text.split("\n")):
                y = y0 + i * dy
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw bounding boxes around detected hands
        for bbox in bounding_boxes:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box

        # Draw hand landmarks and connections
        for landmarks in hand_landmarks:
            # Draw connections
            for start, end in HAND_CONNECTIONS:
                if start < len(landmarks) and end < len(landmarks):
                    cv2.line(frame, landmarks[start], landmarks[end], (255, 0, 0), 2)  # Blue lines for connections
            # Draw landmarks
            for x, y in landmarks:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red dots for landmarks

        # Display the webcam feed
        cv2.imshow('Gesture Recognition', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
