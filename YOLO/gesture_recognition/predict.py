from ultralytics import YOLO
import cv2
import yaml

# Load the YOLO model
model = YOLO("train2_best.pt")

# Load class names from the YAML file
with open('data.yaml', 'r') as file:
    data = yaml.safe_load(file)
class_names = data['names']

# Open a connection to the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Make predictions on the frame
    results = model.predict(source=frame, conf=0.3)  # Set a confidence threshold

    # Get the gesture with the highest confidence
    detected_gesture = None
    highest_confidence = 0

    for result in results:
        for box in result.boxes:
            confidence = box.conf[0]  # Extract confidence
            class_id = int(box.cls[0])  # Extract class ID
            
            if confidence > highest_confidence:  # Update if this gesture has the highest confidence
                highest_confidence = confidence
                detected_gesture = class_names[class_id]

    # Display the detected gesture at the top of the frame
    if detected_gesture:
        text = f"Gesture: {detected_gesture} ({highest_confidence:.2f})"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Gesture Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
